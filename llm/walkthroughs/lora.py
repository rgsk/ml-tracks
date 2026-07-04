"""
WALKTHROUGH: LoRA (Low-Rank Adaptation), one layer at a time.

SFT fine-tuned ALL ~0.5M weights of the base model. LoRA asks: do we have to?
Empirically, the UPDATE a pretrained model needs to learn a new task has low
"intrinsic rank" — it lives in a tiny subspace. So instead of learning a full
weight change dW (out x in numbers), LoRA freezes the original weight W0 and
learns dW as a product of two skinny matrices:

    W_effective = W0 + (alpha/r) * B @ A       A: (r, in)   B: (out, r)   r << in,out
                  ^^ frozen        ^^^^^^^^^ the only trainable params

r is the rank (4-8 typical). That's r*(in+out) params instead of in*out — often
100-1000x fewer — for nearly the same task quality. QLoRA = this + a 4-bit frozen
base so huge models fit on one GPU.

We build it on the CHAR base + reverse task (full SFT hit 100% there), so we can
compare LoRA vs full fine-tuning apples-to-apples.

Layers (run each `exp_*`, watch the output, then say "next"):
  1. the low-rank idea — dW = B@A: the param arithmetic + why it's expressive.  (here)
  2. LoRALinear — wrap a frozen Linear, add the A/B branch; B=0 => starts as base.
  3. freezing — requires_grad=False on the base; the optimizer sees only A,B.
  4. inject — swap the GPT's Linears for LoRALinear; count trainable params.
  5. train — fine-tune reverse with <1% of the params (reuse the SFT masked loop).
  6. merge — fold W0 + (alpha/r)B@A into one Linear: zero inference overhead.

This is the SCRATCH file; you rebuild a clean lora.py yourself afterwards.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import random
import sys

# this file lives in llm/walkthroughs/; its building blocks live in the parent
# llm/ package dir — put that on the import path first.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LLM = os.path.dirname(_HERE)
sys.path.insert(0, _LLM)

import torch
import torch.nn as nn

from config import GPTConfig
from model import GPT
# reuse the REAL SFT pipeline unchanged — LoRA only changes WHAT'S trainable, not
# the data/loss/loop. (sft.py resolves to llm/sft.py: _LLM is first on sys.path.)
from sft import (build_example, format_reverse, generate_answer, load_tokenizer,
                 make_words, sft_batch)

_CHAR_CKPT = os.path.join(_LLM, "artifacts", "checkpoints", "char", "best.pt")


def load_char_base():
    """Rebuild the pretrained CHAR base GPT from its checkpoint (full SFT hit 100%
    on reverse here). This is what LoRA adapts."""
    ckpt = torch.load(_CHAR_CKPT, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _banner(*lines):
    print("=" * 70)
    for line in lines:
        print(line)
    print("=" * 70)


# ---------------------------------------------------------------------------
# LAYER 1: the low-rank idea — dW = B @ A.
#
# A Linear's weight W is (out, in). A full fine-tuning update dW is the same
# shape: out*in free numbers. LoRA replaces dW with a PRODUCT of two skinny
# matrices:
#     A: (r, in)      B: (out, r)      dW = B @ A   -> shape (out, in)  ✓
# The product is a valid (out, in) matrix, but it only has r*(in+out) free
# numbers, and its RANK is at most r. So we're betting the needed update is
# low-rank — a bet that holds in practice because adapting a big pretrained
# model moves it within a small subspace, not all over weight space.
#
# This layer is pure arithmetic: show the param count drop, and confirm B@A is
# a full-shape but rank<=r matrix. No model yet.
# ---------------------------------------------------------------------------
def exp_1_low_rank_idea(seed=0):
    """For a few real Linear shapes from our GPT (n_embd=96), compare full-update
    params (in*out) vs LoRA params r*(in+out), and verify B@A is (out,in) with
    rank r."""
    _banner("LAYER 1: dW = B@A — a rank-r update, r*(in+out) params not in*out")
    torch.manual_seed(seed)

    # (name, in, out) for the Linear layers our 96-dim GPT actually has
    shapes = [
        ("attn q/k/v (each)", 96, 96),
        ("attn out-proj",     96, 96),
        ("mlp up-proj",       96, 384),
        ("mlp down-proj",     384, 96),
    ]
    r = 8
    print(f"  rank r = {r}\n")
    print("  layer               |   in x out |  full dW |  LoRA r(in+out) | shrink")
    print("  --------------------+------------+----------+-----------------+-------")
    for name, din, dout in shapes:
        full = din * dout
        lora = r * (din + dout)
        print(f"  {name:19s} | {din:4d} x {dout:4d} | {full:8d} | {lora:15d} | {full/lora:4.1f}x")

    # concretely build one dW = B@A and check its shape + rank
    din, dout = 384, 96
    A = torch.randn(r, din)
    B = torch.randn(dout, r)
    dW = B @ A
    print(f"\n  build dW = B@A with A{tuple(A.shape)}, B{tuple(B.shape)}:")
    print(f"    dW shape = {tuple(dW.shape)}   (matches a real (out,in) weight)")
    print(f"    rank(dW) = {torch.linalg.matrix_rank(dW).item()}   (<= r = {r}: it's low-rank BY CONSTRUCTION)")
    print(f"    free params: B@A has {r*(din+dout)}, a full {dout}x{din} update has {din*dout}")

    print("\n  The whole bet: the UPDATE a task needs is low-rank, so r*(in+out) knobs")
    print("  suffice. Next (layer 2): wrap a frozen Linear with this A/B branch so the")
    print("  module computes W0 x + (alpha/r) B@A x, trainable only in A,B.")


# ---------------------------------------------------------------------------
# LAYER 2: LoRALinear — a frozen Linear + the trainable A/B branch.
#
# The module keeps the original nn.Linear as `base` and adds two small
# parameters A (r,in), B (out,r). Its forward is:
#     out = base(x)  +  (alpha/r) * (x @ A^T) @ B^T          # = W0 x + scaling*B@A x
#
# Two init details that matter A LOT:
#   * B = 0  -> the whole LoRA branch outputs 0 at init, so LoRALinear(x) EXACTLY
#              equals base(x). Fine-tuning therefore STARTS at the pretrained model
#              (no random jolt), which is why it's stable.
#   * A = random (NOT zero). If BOTH were zero the branch would be stuck: the
#              gradient wrt A is proportional to B (=0), so A could never move.
#              With A random, grad flows to B (grad wrt B is proportional to A),
#              so B lifts off zero and training begins. (We prove this asymmetry.)
#
# alpha/r is a fixed scale (not learned): it lets you change r later without
# re-tuning the LR, since the branch magnitude stays ~constant (layer-1 Q3).
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base                                  # the pretrained Linear
        self.r = r
        self.scaling = alpha / r
        din, dout = base.in_features, base.out_features
        self.A = nn.Parameter(torch.empty(r, din))
        self.B = nn.Parameter(torch.zeros(dout, r))       # ZERO: branch starts at 0
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # A random (must be nonzero)

    def forward(self, x):
        return self.base(x) + self.scaling * (x @ self.A.T) @ self.B.T


def exp_2_lora_linear(seed=0):
    """Wrap a real Linear in LoRALinear and show: (1) at init it equals the base
    exactly (B=0), (2) the grad asymmetry that forces A random / B zero, (3) once B
    moves, the output shifts."""
    _banner("LAYER 2: LoRALinear = frozen base + (alpha/r)*B@A branch; B=0 at init")
    torch.manual_seed(seed)

    din, dout = 96, 96
    base = nn.Linear(din, dout)
    layer = LoRALinear(base, r=8, alpha=16)
    x = torch.randn(4, din)

    # (1) at init the LoRA branch is a no-op -> output == base output, exactly
    with torch.no_grad():
        same = torch.allclose(layer(x), base(x))
    print(f"  A init: nonzero (kaiming), norm {layer.A.norm():.3f}")
    print(f"  B init: zeros,             norm {layer.B.norm():.3f}")
    print(f"  scaling alpha/r = 16/8 = {layer.scaling}")
    print(f"  layer(x) == base(x) at init?  {same}   (LoRA starts as the base model)")

    # (2) the gradient asymmetry: with B=0, A gets NO gradient; B does.
    layer.zero_grad()
    layer(x).sum().backward()
    print(f"\n  after one backward (B still 0 at this instant):")
    print(f"    |grad A| = {layer.A.grad.norm():.4e}   (zero: grad wrt A ∝ B = 0)")
    print(f"    |grad B| = {layer.B.grad.norm():.4e}   (nonzero: grad wrt B ∝ A ≠ 0)")
    print("    -> so B lifts off zero first; if A were ALSO zero, nothing would move.")

    # (3) push B off zero by hand -> the output now differs from the base
    with torch.no_grad():
        layer.B += 0.1 * torch.randn_like(layer.B)
        delta = (layer(x) - base(x)).abs().max().item()
    print(f"\n  nudge B, then max|layer(x) - base(x)| = {delta:.4f}  (branch now active)")

    print("\n  Next (layer 3): freeze the base so requires_grad is True ONLY for A,B —")
    print("  that's where the memory/compute win actually comes from.")


# ---------------------------------------------------------------------------
# LAYER 3: freezing — the memory/compute win.
#
# Wrapping a Linear (layer 2) didn't save anything yet: the base weight still had
# requires_grad=True, so the optimizer would still update it. LoRA's win is that
# we FREEZE the base — requires_grad_(False) on base.weight/bias — and build the
# optimizer over the trainable params only. Two concrete consequences:
#   * backward computes NO gradient for the frozen base (base.weight.grad stays
#     None) -> no gradient memory for 99% of the weights.
#   * the optimizer holds state for the trainable params only. This is the big
#     one for Adam: it keeps TWO extra tensors (m, v) per trainable param, so
#     freezing 99% of the model cuts optimizer memory ~99% too.
# (Forward still runs through the frozen base, so activations aren't saved — LoRA
# trims gradient + optimizer memory, not activation memory.)
# ---------------------------------------------------------------------------
def freeze_base(layer: LoRALinear):
    """Turn off grad for the wrapped base Linear; A and B stay trainable."""
    for p in layer.base.parameters():
        p.requires_grad_(False)


def _counts(module):
    total = sum(p.numel() for p in module.parameters())
    train = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, train


def exp_3_freezing(seed=0):
    """Freeze the base of one LoRALinear and show trainable params collapse to just
    A,B; the base gets no gradient; and the optimizer only manages A,B."""
    _banner("LAYER 3: freeze the base -> only A,B train (grad + optimizer memory win)")
    torch.manual_seed(seed)

    base = nn.Linear(96, 96)
    layer = LoRALinear(base, r=8, alpha=16)

    total, train_before = _counts(layer)
    print(f"  before freeze: {total} total params, {train_before} trainable  "
          f"(base is still trainable — layer-2 Q3)")

    freeze_base(layer)
    _, train_after = _counts(layer)
    print(f"  after  freeze: {total} total params, {train_after} trainable "
          f"(= A {layer.A.numel()} + B {layer.B.numel()}) -> {train_after/total:.1%} of the module")

    # backward: no gradient is stored for the frozen base
    layer.zero_grad(set_to_none=True)
    layer(torch.randn(4, 96)).sum().backward()
    print(f"\n  after backward:")
    print(f"    base.weight.grad is None? {layer.base.weight.grad is None}   (frozen -> no grad memory)")
    print(f"    A.grad / B.grad present?  {layer.A.grad is not None and layer.B.grad is not None}")

    # the optimizer only sees the trainable params
    opt = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=1e-3)
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    print(f"\n  optimizer built over requires_grad params only: manages {n_opt} params")
    print(f"    Adam keeps 2 state tensors (m,v) per trainable param -> "
          f"{2*n_opt} numbers, vs {2*total} if the base were trainable.")

    print("\n  (This module is small, so trainable % looks big; at the FULL-model level")
    print("  next, the frozen embeddings + attention dominate and LoRA is <1%.)")
    print("  Next (layer 4): inject LoRALinear across the real GPT and count params.")


# ---------------------------------------------------------------------------
# LAYER 4: inject — swap the GPT's Linears for LoRALinear, count trainable params.
#
# We walk the model, replace each target nn.Linear with a LoRALinear that WRAPS it
# (so the pretrained weights are kept), then freeze everything except the A/B
# adapters. We target the attention (qkv, out-proj) and MLP (up, down) Linears —
# the standard set — and skip lm_head/embeddings.
#
# Collect-then-replace (not replace-while-walking): once we insert a LoRALinear,
# its own `.base` child is an nn.Linear too, and a naive walk would wrap that
# again. So we snapshot the targets from the ORIGINAL tree first.
# ---------------------------------------------------------------------------
def apply_lora(model, r: int = 8, alpha: int = 16, skip=("lm_head",)):
    """Replace target nn.Linear layers with LoRALinear, then freeze all base
    weights so only the A,B adapters train. Returns the count of layers adapted."""
    targets = []
    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Linear) and name not in skip:
                targets.append((parent, name, child))
    for parent, name, child in targets:
        setattr(parent, name, LoRALinear(child, r=r, alpha=alpha))

    for p in model.parameters():                 # freeze everything ...
        p.requires_grad_(False)
    for m in model.modules():                    # ... then re-enable just A,B
        if isinstance(m, LoRALinear):
            m.A.requires_grad_(True)
            m.B.requires_grad_(True)
    return len(targets)


def exp_4_inject(r=8, alpha=16):
    """Load the char base, inject LoRA across its attention+MLP Linears, and report
    how many layers were adapted and the trainable fraction."""
    _banner(f"LAYER 4: inject LoRALinear across the GPT (r={r}) + count trainable")
    model = load_char_base()

    total_before, _ = _counts(model)
    n = apply_lora(model, r=r, alpha=alpha)
    total_after, train_after = _counts(model)

    print(f"  base model: {total_before:,} params, all trainable if fully fine-tuned")
    print(f"  injected LoRA into {n} Linear layers (attn qkv/proj + mlp up/down x "
          f"{n//4} blocks)\n")
    print(f"  total params now: {total_after:,}  (+{total_after-total_before:,} for A,B)")
    print(f"  TRAINABLE params: {train_after:,}  = {train_after/total_after:.1%} of the model")
    print(f"  full fine-tuning would train {total_before:,} "
          f"({total_before/train_after:.0f}x more)")

    print("\n  Reconciling the layer-3 teaser: I said '<1%', but here it's ~"
          f"{train_after/total_after:.0%} — because this model is TINY (n_embd 96) and r=8")
    print("  is a big fraction of it. On a real LLM (n_embd 4096, same r=8) the adapters")
    print("  are the same tiny size while the frozen base is ~2000x bigger -> well under 1%.")
    print("  Next (layer 5): train ONLY these adapters on reverse, reusing the SFT loop.")


# ---------------------------------------------------------------------------
# LAYER 5: train ONLY the adapters — reuse the SFT masked loop verbatim.
#
# This is the payoff. We take the char base, inject LoRA (base frozen), build the
# optimizer over the ~10% trainable A,B, and run the EXACT SFT loop (same masked
# batches from sft.py, same forward/backward). The gradient only reaches A,B; the
# pretrained weights never move. If the low-rank bet holds, we recover full-SFT
# quality (100% on reverse) while training a fraction of the params.
#
# LoRA usually wants a HIGHER learning rate than full fine-tuning: the adapter
# starts at zero and its update is scaled by alpha/r, so it needs a bigger step to
# get going (full SFT used 5e-4; we use ~1e-3).
# ---------------------------------------------------------------------------
def exp_5_train(steps=2000, batch_size=32, lr=1e-3, r=8, alpha=16, seed=0):
    """Fine-tune ONLY the LoRA adapters on reverse and report held-out accuracy vs
    full SFT (which hit 100%). Same data/loss/loop as sft.py — only A,B train."""
    _banner(f"LAYER 5: train ONLY the adapters (r={r}, lr={lr}) on reverse")
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = load_tokenizer("char")
    model = load_char_base()
    n = apply_lora(model, r=r, alpha=alpha)
    model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(trainable, lr=lr)

    train_ex = [build_example(*format_reverse(w), tok) for w in make_words(4000, seed=1)]
    test_words = make_words(200, seed=3)
    rng = random.Random(seed)

    def acc():
        model.eval()
        return sum(generate_answer(model, tok, w, device) == w[::-1] for w in test_words) / len(test_words)

    print(f"  training {n_train:,} / {n_total:,} params ({n_train/n_total:.1%}) across {n} adapters")
    print(f"  held-out accuracy before training: {acc():.1%}\n")
    print("   step | train loss | held-out acc")
    print("  ------+------------+-------------")
    for step in range(1, steps + 1):
        model.train()
        batch = [train_ex[rng.randrange(len(train_ex))] for _ in range(batch_size)]
        x, y = sft_batch(batch, device)
        opt.zero_grad(set_to_none=True)
        loss = model(x, y)[1]
        loss.backward()
        opt.step()
        if step == 1 or step % 400 == 0:
            print(f"  {step:5d} |   {loss.item():6.3f}   |   {acc():.1%}")

    final = acc()
    print(f"\n  LoRA final held-out accuracy: {final:.1%}   (full SFT was 100%)")
    print(f"  reached with only {n_train/n_total:.1%} of the params trainable — the frozen")
    print("  base never moved; all the task knowledge is in the tiny A,B adapters.")
    print("  Next (layer 6): MERGE W0 + (alpha/r)B@A into plain Linears -> zero overhead.")


# ---------------------------------------------------------------------------
# LAYER 6: merge — fold W0 + (alpha/r)B@A into one Linear (zero inference cost).
#
# During training LoRA's forward runs TWO paths: base(x) PLUS the A/B branch — a
# little extra compute per layer (layer-4 Q1). But dW = (alpha/r)B@A is just a
# weight delta, so for deployment we ADD it into the base weight once:
#     W_merged = W0 + (alpha/r) * (B @ A)
# and replace the LoRALinear with a plain nn.Linear holding W_merged. Now the
# forward is a single matmul again — identical outputs, NO LoRA overhead, and the
# A,B params disappear.
#
# Trade-off: merged = fast but frozen to ONE task (you lose the swap-a-different-
# adapter trick). Keep it unmerged if you want to hot-swap tasks; merge a copy
# when you deploy a single task. (You can always un-merge by subtracting dW.)
# ---------------------------------------------------------------------------
def merge_lora(model):
    """Fold every LoRALinear's (alpha/r)B@A into its base weight and swap the
    wrapper for the plain (now-merged) Linear. Returns how many were merged."""
    to_merge = [(parent, name, child)
                for parent in model.modules()
                for name, child in parent.named_children()
                if isinstance(child, LoRALinear)]
    for parent, name, child in to_merge:
        with torch.no_grad():
            child.base.weight += child.scaling * (child.B @ child.A)   # W0 += dW
        setattr(parent, name, child.base)                              # drop the wrapper
    return len(to_merge)


def exp_6_merge(r=8, alpha=16, seed=0):
    """Inject LoRA, activate the adapters (nonzero B, as if trained), then merge and
    show: outputs are identical, the A,B params vanish, no LoRALinear remains."""
    _banner("LAYER 6: merge W0 + (alpha/r)B@A -> plain Linear, identical & overhead-free")
    torch.manual_seed(seed)

    model = load_char_base()
    apply_lora(model, r=r, alpha=alpha)
    # simulate a trained adapter: give B a nonzero value so the branch is active
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.B.copy_(0.05 * torch.randn_like(m.B))
    model.eval()

    idx = torch.randint(0, model.config.vocab_size, (2, 16))
    with torch.no_grad():
        before = model(idx)[0]
    params_before = sum(p.numel() for p in model.parameters())
    n_lora = sum(isinstance(m, LoRALinear) for m in model.modules())

    merged = merge_lora(model)

    with torch.no_grad():
        after = model(idx)[0]
    params_after = sum(p.numel() for p in model.parameters())
    n_lora_after = sum(isinstance(m, LoRALinear) for m in model.modules())

    max_diff = (before - after).abs().max().item()
    print(f"  LoRALinear modules: {n_lora} -> {n_lora_after}  (merged {merged}, all now plain Linear)")
    print(f"  params: {params_before:,} -> {params_after:,}  (A,B folded away: -{params_before-params_after:,})")
    print(f"  max |logits_before - logits_after| = {max_diff:.2e}   (identical up to float rounding)")
    assert torch.allclose(before, after, atol=1e-4), "merge changed the function!"
    print("  outputs match -> the merged model computes the SAME thing with a single")
    print("  matmul per layer: zero LoRA overhead at inference.  ✅")

    print("\n  That's LoRA end-to-end: low-rank dW = B@A, frozen base + trainable A/B")
    print("  (B=0 start), inject across the model, train ~10% of params to full quality,")
    print("  then merge for free inference. Next: rebuild a clean lora.py yourself.")


def run_experiments():
    # exp_1_low_rank_idea()
    # exp_2_lora_linear()
    # exp_3_freezing()
    # exp_4_inject()
    # exp_5_train()
    exp_6_merge()


@contextlib.contextmanager
def _tee(path):
    """Print to BOTH the terminal and `path` (long runs survive scrollback)."""
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)

        def flush(self):
            for st in self.streams:
                st.flush()

    with open(path, "w") as f:
        with contextlib.redirect_stdout(_Tee(sys.stdout, f)):
            yield
    print(f"(output also written to {path})", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", metavar="FILE", help="also write all output to FILE")
    args = parser.parse_args()

    if args.out:
        with _tee(args.out):
            run_experiments()
    else:
        run_experiments()
