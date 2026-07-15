# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# LLM · 09 — RMSNorm: drop the centering

**Phase B opens here.** `01`–`08` built a GPT-2-shaped model and drove it properly;
every box is open. What's left is that the parts themselves have *changed*. Llama,
Mistral, Qwen and Gemma do not run 2019's transformer — they run this one with four
specific pieces swapped out. Phase B does them **one delta per notebook**, each a small
measured upgrade on the exact model you just built.

This is the smallest of the four, and the best one to start on because the *code* is
three lines and all the interest is in the *why*.

LayerNorm does two things to each token vector, and `06` only ever cared that it did
*something*:

    y = (x - mean) / sqrt(var + eps) * gamma + beta
         └── 1 ──┘   └──── 2 ─────┘         └─ 3 ─┘

1. **re-centering** — subtract the mean, so the vector's features average 0
2. **re-scaling** — divide by the standard deviation, so it has unit spread
3. **affine** — a learned per-feature scale `gamma` and shift `beta`

RMSNorm keeps **only step 2** (measured against the raw vector rather than the centered
one) and half of step 3:

    y = x / sqrt(mean(x^2) + eps) * gamma

No mean subtraction. No bias. Zhang & Sennrich (2019) reported you lose essentially
nothing by doing this, and the whole industry believed them — so the question worth a
notebook isn't *what* the code is, it's **why deleting half of a load-bearing layer is
free.** That's what we measure:

- **the two are nearly the same function on real data** — we hook the trained model and
  find that centering deletes only a few percent of the typical token vector, because
  the mean is already close to 0. So RMSNorm ≈ LayerNorm before you train anything.
- **and when it isn't, the centering is *absorbable*** — subtracting the mean is a
  **linear projection**, so the very next `Linear` can learn it (or unlearn it) for
  free. The same goes for `beta`, which is *exactly* redundant with the next layer's
  bias — we prove both by fusing them into a Linear's weights and matching to ~0.
- **the re-scaling is the part that can't be faked**, because it's **nonlinear** —
  `RMSNorm(2x) == RMSNorm(x)`, a scale-invariance no matrix can reproduce. That
  asymmetry *is* the answer: step 2 does the load-bearing work, steps 1 and 3 were
  always things the network could do itself.

Then the honest A/B: swap every norm in `06`'s GPT and retrain **3 seeds each**, so we
can tell a real difference from seed noise.

And then the part that surprised me. The standard story is "RMSNorm is *simpler and
faster*" — one reduction instead of two. We go benchmark that at Llama width, expecting
to bank the win our toy is too small to show... and **it isn't there**: torch's fused
LayerNorm and fused RMSNorm run at exactly the same speed. The saving is real in the
arithmetic and *invisible* on the hardware, for a reason worth understanding — so we
chase it down, find where the win does show up, and end with an honest answer to why the
industry switched anyway. It isn't the one in the abstract.
"""

# %%
from pathlib import Path

import math
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def _repo_root() -> Path:
    """Walk up from the cwd to the folder holding pyproject.toml, so paths work no
    matter where the kernel is launched from."""
    here = Path.cwd()
    for d in (here, *here.parents):
        if (d / "pyproject.toml").exists():
            return d
    return here


ROOT = _repo_root()
DATA = ROOT / "nb" / "llm" / "data" / "input.txt"  # tinyshakespeare, shared with 01–08
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
print(f"device: {DEV}")

# %% [markdown]
"""
## Same data skeleton as `03`–`08`
"""

# %%
text = DATA.read_text()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

BLOCK = 64
BATCH = 32


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK, (BATCH,))
    x = torch.stack([d[i:i + BLOCK] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + BLOCK] for i in ix])
    return x.to(DEV), y.to(DEV)


print(f"corpus {len(text):,} chars, vocab {vocab_size}, block {BLOCK}")

# %% [markdown]
"""
## Both norms, from scratch

Three lines each, and the diff between them is the notebook. Written out rather than
imported, because they *are* the lesson — and each is checked against its `torch` twin
to ~0, so we know we're studying the real thing and not our own bug.
"""


# %%
class LayerNorm(nn.Module):
    """(x - mean) / sqrt(var + eps) * gamma + beta — re-center, re-scale, affine."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # gamma: per-feature scale
        self.bias = nn.Parameter(torch.zeros(dim))    # beta:  per-feature shift

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        # population variance (divide by C, not C-1) — what nn.LayerNorm uses
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.weight + self.bias


class RMSNorm(nn.Module):
    """x / sqrt(mean(x^2) + eps) * gamma — re-scale only. No centering, no bias."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))   # gamma only — there is no beta

    def forward(self, x):
        # mean-of-squares, NOT variance: no mean is subtracted. The .mean (not .sum)
        # is what makes the norm independent of the feature count C.
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        x_hat = x / torch.sqrt(ms + self.eps)         # eps INSIDE the sqrt
        return x_hat * self.weight


# both must match torch exactly (same init: gamma=1, beta=0)
_x = torch.randn(4, 8, 32)
print(f"LayerNorm vs nn.LayerNorm : max diff "
      f"{(LayerNorm(32)(_x) - nn.LayerNorm(32)(_x)).abs().max():.2e}")
print(f"RMSNorm   vs nn.RMSNorm   : max diff "
      f"{(RMSNorm(32)(_x) - nn.RMSNorm(32)(_x)).abs().max():.2e}")
n_ln = sum(p.numel() for p in LayerNorm(32).parameters())
n_rms = sum(p.numel() for p in RMSNorm(32).parameters())
print(f"\nparams per norm of width 32: LayerNorm {n_ln} (gamma+beta), "
      f"RMSNorm {n_rms} (gamma only)")

# %% [markdown]
"""
### `mean(x^2)` is not the variance

The single most common misreading of that code is that RMSNorm divides by the standard
deviation like LayerNorm does, just written differently. It doesn't. Recall:

```-
var(x) = mean(x^2) - mean(x)^2
         └───┬────┘   └───┬───┘
        RMSNorm uses      the term RMSNorm never computes
        THIS alone
```

So `mean(x^2) == var(x)` **only when `mean(x) == 0`** — which is exactly the condition
LayerNorm manufactures by centering and RMSNorm skips. RMSNorm's denominator is the raw
**root-mean-square** of the vector, not its standard deviation. The two coincide only
for an already-centered input:
"""

# %%
x_centered = torch.randn(1000)
x_shifted = torch.randn(1000) + 3.0            # same spread, mean moved to ~3

for name, v in [("mean ~0", x_centered), ("mean ~3", x_shifted)]:
    ms, var = v.pow(2).mean().item(), v.var(unbiased=False).item()
    agree = "AGREE" if abs(ms - var) < 0.05 else f"DIFFER by {ms - var:.2f}"
    print(f"{name}: mean {v.mean():>6.3f} | mean(x^2) {ms:>6.3f} | "
          f"var {var:>6.3f} | they {agree}")
print("\nmean(x^2) - var = mean(x)^2, exactly — so the gap is the mean, squared.")

# %% [markdown]
"""
And the direct consequence: **LayerNorm removes a shift, RMSNorm does not.** Feed both a
vector whose features average 5 and look at what comes out.
"""

# %%
x_off = torch.randn(4, 32) + 5.0
ln_out, rms_out = LayerNorm(32)(x_off), RMSNorm(32)(x_off)
rms_of = lambda t: t.pow(2).mean().sqrt()
print(f"input             : mean {x_off.mean():>6.3f}   rms {rms_of(x_off):>5.3f}")
print(f"LayerNorm output  : mean {ln_out.mean():>6.3f}   rms {rms_of(ln_out):>5.3f}"
      f"   <- mean forced to 0")
print(f"RMSNorm output    : mean {rms_out.mean():>6.3f}   rms {rms_of(rms_out):>5.3f}"
      f"   <- mean SURVIVES, only the scale is fixed")
print("\nBoth pin the scale. Only LayerNorm pins the location. Whether that matters is")
print("the rest of this notebook.")

# %% [markdown]
"""
## The model, norm-swappable

`06`'s GPT with one change: the norm is a **constructor argument**. Every norm site —
`ln1`, `ln2` in each Block and the final `ln_f` — is built by whatever class we pass, so
`GPT(norm=LayerNorm)` and `GPT(norm=RMSNorm)` differ in exactly one respect and nothing
else. That's what makes the A/B honest.

Note we're using our own from-scratch classes above, not torch's — they match to ~0, and
this way the thing being compared is the thing we read.
"""


# %%
class CausalSelfAttention(nn.Module):
    """All heads at once: qkv projection, reshape to (B, nh, T, hs), masked softmax."""

    def __init__(self, n_embd, n_head, block_size=BLOCK):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.n_embd = n_head, n_embd
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.proj.RESIDUAL_PROJ = True
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        nh, hs = self.n_head, self.head_size
        q, k, v = self.qkv(x).split(self.n_embd, dim=-1)
        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) * hs ** -0.5
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        out = F.softmax(scores, dim=-1) @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    """Position-wise MLP: expand 4x, GELU, project back into the residual stream."""

    def __init__(self, n_embd):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.proj.RESIDUAL_PROJ = True

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    """06's Block, with the norm class injected rather than hard-coded."""

    def __init__(self, n_embd, n_head, norm, block_size=BLOCK):
        super().__init__()
        self.ln1 = norm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = norm(n_embd)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """06's GPT. `norm` selects LayerNorm or RMSNorm at all 2*n_layer + 1 sites."""

    def __init__(self, vocab_size, norm=LayerNorm, n_embd=128, n_head=4, n_layer=4,
                 block_size=BLOCK):
        super().__init__()
        self.block_size, self.n_layer = block_size, n_layer
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, norm, block_size) for _ in range(n_layer)]
        )
        self.ln_f = norm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss


# %%
def apply_gpt2_init(model):
    """07's Knob 1: N(0, 0.02), residual projections scaled by 1/sqrt(2*n_layer)."""
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            std = 0.02 * ((2 * model.n_layer) ** -0.5
                          if getattr(mod, "RESIDUAL_PROJ", False) else 1.0)
            nn.init.normal_(mod.weight, std=std)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Embedding):
            nn.init.normal_(mod.weight, std=0.02)
    return model


def get_lr(step, *, warmup, max_steps, max_lr, min_lr):
    """07's Knob 3: linear warmup to max_lr, then half-cosine decay to min_lr."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    ratio = min((step - warmup) / (max_steps - warmup), 1.0)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


@torch.no_grad()
def full_val_loss(m, bs=256):
    """The same deterministic whole-val-split metric `03`–`08` reported."""
    m.eval()
    nwin = (len(val_data) - 1) // BLOCK
    x = val_data[:nwin * BLOCK].view(nwin, BLOCK)
    y = val_data[1:nwin * BLOCK + 1].view(nwin, BLOCK)
    total = count = 0
    for i in range(0, nwin, bs):
        xb, yb = x[i:i + bs].to(DEV), y[i:i + bs].to(DEV)
        total += m(xb, yb)[1].item() * yb.numel()
        count += yb.numel()
    m.train()
    return total / count


def train(m, max_iters=3000, max_lr=3e-3, min_lr=3e-4, warmup=100, log=False):
    """07's recipe, unchanged: GPT-2 init, AdamW 2-group decay, warmup+cosine, clip."""
    apply_gpt2_init(m)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": [p for p in params if p.dim() >= 2], "weight_decay": 0.1},
         {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}],
        lr=max_lr, betas=(0.9, 0.95),
    )
    for it in range(max_iters + 1):
        for g in opt.param_groups:
            g["lr"] = get_lr(it, warmup=warmup, max_steps=max_iters,
                             max_lr=max_lr, min_lr=min_lr)
        if log and it % 1000 == 0:
            print(f"  step {it:>4} : val {full_val_loss(m):.3f}")
        xb, yb = get_batch("train")
        _, loss = m(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return m


# the LayerNorm baseline — the model every measurement below is taken on
torch.manual_seed(1337)
baseline = train(GPT(vocab_size, norm=LayerNorm).to(DEV), log=True)
baseline.eval()
print(f"\nLayerNorm baseline: val {full_val_loss(baseline):.3f}")

# %% [markdown]
"""
## Why is the centering free? Reason 1: the mean is already ~0

Start empirically, on the trained model rather than on a story. Hook every norm site and
look at the tensor **going in**: what is its per-token mean, next to its RMS?

The ratio of those two is exactly the quantity we want, and it has a clean geometric
reading. Centering subtracts `mean * 1` (the all-ones direction). That removed piece has
length `|mean| * sqrt(C)`, while the vector itself has length `rms * sqrt(C)`. So:

```-
       length of what centering deletes     |mean| * sqrt(C)     |mean|
      ----------------------------------- = ---------------- =  --------
             length of the vector             rms * sqrt(C)        rms
```

**`|mean| / rms` is the fraction of the token vector that centering throws away.** If
it's small, LayerNorm and RMSNorm are near-identical functions and there is nothing to
lose by dropping step 1.
"""


# %%
@torch.no_grad()
def norm_site_stats(m, batches=8):
    """Per-token |mean| and RMS of the tensor entering each norm, via forward hooks."""
    stats, handles = {}, []

    def hook(name):
        def fn(mod, inp, out):
            x = inp[0]
            stats.setdefault(name, []).append(
                (x.mean(-1).abs().mean().item(), x.pow(2).mean(-1).sqrt().mean().item())
            )
        return fn

    for i, b in enumerate(m.blocks):
        handles.append(b.ln1.register_forward_hook(hook(f"block{i}.ln1")))
        handles.append(b.ln2.register_forward_hook(hook(f"block{i}.ln2")))
    handles.append(m.ln_f.register_forward_hook(hook("ln_f")))
    for _ in range(batches):
        m(get_batch("val")[0])
    for h in handles:
        h.remove()
    return {k: (sum(a for a, _ in v) / len(v), sum(b for _, b in v) / len(v))
            for k, v in stats.items()}


stats = norm_site_stats(baseline)
print(f"{'norm site':<14} | {'|mean|':>8} | {'rms':>8} | {'|mean|/rms':>10}")
print("-" * 50)
ratios = []
for site, (mu, rms) in stats.items():
    ratios.append(mu / rms)
    print(f"{site:<14} | {mu:>8.3f} | {rms:>8.3f} | {mu / rms:>9.1%}")
print(f"\nmean over sites: centering deletes {sum(ratios) / len(ratios):.1%} of the "
      f"typical token vector.")

# %% [markdown]
"""
Now the direct consequence — how different are the two functions *on this model's actual
activations*? Take the real tensor entering each norm and push it through both (with
`gamma=1, beta=0`, so we compare the normalization itself and not a learned scale), then
measure the relative difference.
"""


# %%
@torch.no_grad()
def norm_output_gap(m, batches=4):
    """Relative L2 gap between LayerNorm(x) and RMSNorm(x) on real activations."""
    gaps, handles = {}, []

    def hook(name):
        def fn(mod, inp, out):
            x = inp[0].float()
            C = x.size(-1)
            ln = LayerNorm(C).to(x.device)(x)      # gamma=1, beta=0
            rms = RMSNorm(C).to(x.device)(x)
            rel = (ln - rms).norm(dim=-1) / ln.norm(dim=-1)
            gaps.setdefault(name, []).append(rel.mean().item())
        return fn

    for i, b in enumerate(m.blocks):
        handles.append(b.ln1.register_forward_hook(hook(f"block{i}.ln1")))
    handles.append(m.ln_f.register_forward_hook(hook("ln_f")))
    for _ in range(batches):
        m(get_batch("val")[0])
    for h in handles:
        h.remove()
    return {k: sum(v) / len(v) for k, v in gaps.items()}


for site, gap in norm_output_gap(baseline).items():
    print(f"{site:<14}: LayerNorm(x) vs RMSNorm(x) differ by {gap:>6.1%} (relative L2)")
print("\nSame input, two 'different' layers, single-digit-percent apart on most sites.")
print("They were never doing very different things — centering had little left to do.")

# %% [markdown]
"""
## Reason 2: centering is *linear*, so the next layer can do it itself

The empirical answer above is satisfying but circumstantial — it says the mean *happens*
to be small here. The structural answer is stronger, and it's the one that transfers.

Subtracting the mean is a **linear map**. Concretely, `x - mean(x)*1` is a matrix
multiply by `P = I - (1/C) * 11^T` — a projection that deletes the all-ones component
and leaves everything else alone:
"""

# %%
C = 8
P = torch.eye(C) - torch.ones(C, C) / C
x = torch.randn(5, C)
print(f"x @ P.T  ==  x - x.mean(-1, keepdim=True) ? "
      f"{torch.allclose(x @ P.T, x - x.mean(-1, keepdim=True), atol=1e-6)}")
print(f"P is a projection (P @ P == P)? {torch.allclose(P @ P, P, atol=1e-6)}")
print(f"rank of P: {torch.linalg.matrix_rank(P).item()} of {C}"
      f"  <- it deletes exactly ONE of the {C} dimensions")

# %% [markdown]
"""
And every norm in this model is immediately followed by a `Linear` — `ln1` feeds `qkv`,
`ln2` feeds the FeedForward's `fc`, `ln_f` feeds `lm_head`. So the composition is
`W(Px)`, and matrix multiplication is associative:

```-
W(Px) = (WP)x
```

**`WP` is just another weight matrix.** Which means centering isn't a capability the
architecture provides — it's a *setting of the next layer's weights*. If centering
helps, a model without it can learn `W' = WP` and get the identical function; if it
doesn't help, a model with it has been forced into a subspace for no reason. Either way,
nothing is lost by removing it. Proof by fusion:
"""

# %%
lin = nn.Linear(C, 5, bias=False)
fused = nn.Linear(C, 5, bias=False)
with torch.no_grad():
    fused.weight.copy_(lin.weight @ P)     # fold the centering INTO the weights

centered_then_linear = lin(x @ P.T)        # what a LayerNorm-style model computes
just_linear = fused(x)                     # what an RMSNorm-style model CAN learn
gap = (centered_then_linear - just_linear).abs().max()
print(f"Linear(center(x)) == FusedLinear(x)? "
      f"{torch.allclose(centered_then_linear, just_linear, atol=1e-6)}"
      f"   (max diff {gap:.2e})")
print("\nSo 'centering + Linear' and 'Linear' are the same family of functions.")
print("The centering was never adding expressive power — only fixing a choice.")

# %% [markdown]
"""
### The same argument kills `beta`, exactly

LayerNorm's learned shift `beta` gets added, and then the next `Linear` multiplies by
`W` and adds its own bias `b`:

```-
W(z + beta) + b  =  Wz + (W*beta + b)
                          └──── a constant vector ────┘
```

`W*beta + b` is just... a bias. `beta` is **exactly** redundant with the next layer's
bias term — not approximately, not usually, always. Anything `beta` can express, `b` can
already express. So RMSNorm dropping it costs precisely nothing:
"""

# %%
lin_b = nn.Linear(C, 5)                    # this one HAS a bias
beta = torch.randn(C)
fused_b = nn.Linear(C, 5)
with torch.no_grad():
    fused_b.weight.copy_(lin_b.weight)
    fused_b.bias.copy_(lin_b.bias + lin_b.weight @ beta)   # fold beta into the bias

gap_b = (lin_b(x + beta) - fused_b(x)).abs().max()
print(f"Linear(x + beta) == FusedLinear(x)? "
      f"{torch.allclose(lin_b(x + beta), fused_b(x), atol=1e-6)}"
      f"   (max diff {gap_b:.2e})")
print("\nbeta is absorbed with zero residual — it is a duplicate parameter, always.")

# %% [markdown]
"""
## Reason 3: the re-scaling is the part nothing else can do

So if steps 1 and 3 are absorbable, why not delete step 2 as well and drop the norm
entirely? Because dividing by the RMS is **not linear**, and that's exactly what makes
it irreplaceable.

The tell is **scale invariance**. Double the input and a linear map doubles its output —
that's what linear *means*. A norm doesn't: it returns the same thing, because it
divides by a length that doubled too.

    RMSNorm(2x) == RMSNorm(x)        # but  Linear(2x) == 2 * Linear(x)

No fixed matrix can do that, at any width, ever. So the re-scaling is a genuine
capability the architecture must supply — and it's the one both norms keep. That's the
whole asymmetry: **RMSNorm deletes the two steps a `Linear` could have done, and keeps
the one it couldn't.**
"""

# %%
rms, lin_ref = RMSNorm(C), nn.Linear(C, C, bias=False)
inv = torch.allclose(rms(2 * x), rms(x), atol=1e-6)
equi = torch.allclose(lin_ref(2 * x), 2 * lin_ref(x), atol=1e-6)
doubles = torch.allclose(rms(2 * x), 2 * rms(x), atol=1e-6)
print(f"RMSNorm(2x) == RMSNorm(x)?   {inv}   <- scale-INVARIANT (nonlinear)")
print(f"Linear(2x)  == 2*Linear(x)?  {equi}   <- scale-EQUIVARIANT (linear)")
print(f"RMSNorm(2x) == 2*RMSNorm(x)? {doubles}  <- no linear map can reproduce a norm")
print("\n06 measured WHY that matters: it keeps every sublayer's input unit-scaled")
print("while the residual stream grows through depth. RMSNorm keeps that job intact.")

# %% [markdown]
"""
## The A/B — and it needs more than one run

Now swap the norm and retrain. But a single pair of runs cannot support "same quality":
if LayerNorm lands at 1.573 and RMSNorm at 1.579, is that RMSNorm being worse, or is it
the seed? The only honest way to claim *no difference* is to **measure the noise** — so
we run **3 seeds each** and compare the gap between the norms against the spread within
each norm.
"""

# %%
SEEDS = [1337, 7, 42]
results = {}
for norm_cls in (LayerNorm, RMSNorm):
    losses = []
    for s in SEEDS:
        torch.manual_seed(s)
        m = train(GPT(vocab_size, norm=norm_cls).to(DEV))
        losses.append(full_val_loss(m))
    results[norm_cls.__name__] = losses
    print(f"{norm_cls.__name__:<10}: " + "  ".join(f"{l:.4f}" for l in losses)
          + f"   mean {sum(losses) / len(losses):.4f}")

ln_l, rms_l = results["LayerNorm"], results["RMSNorm"]
ln_mean, rms_mean = sum(ln_l) / len(ln_l), sum(rms_l) / len(rms_l)
spread = max(max(ln_l) - min(ln_l), max(rms_l) - min(rms_l))
print(f"\ngap between norms      : {abs(rms_mean - ln_mean):.4f}")
print(f"worst within-norm spread: {spread:.4f}  (same norm, different seed)")
verdict = ("SMALLER than seed noise" if abs(rms_mean - ln_mean) < spread
           else "larger than noise")
print(f"-> the norm choice is worth {verdict}.")

# %%
plt.figure(figsize=(5.6, 3.4))
for i, (name, losses) in enumerate(results.items()):
    plt.scatter([i] * len(losses), losses, s=45, zorder=3,
                label=f"{name} (mean {sum(losses) / len(losses):.4f})")
    plt.hlines(sum(losses) / len(losses), i - 0.18, i + 0.18, lw=2)
plt.xticks([0, 1], list(results))
plt.ylabel("final val loss")
plt.title(f"3 seeds each — the norms overlap")
plt.legend(fontsize=8)
plt.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.show()

# %% [markdown]
"""
### What the swap actually bought, on this model

Be blunt about the size of the prize here. RMSNorm drops one `beta` vector per norm
site, and this model has `2 * n_layer + 1 = 9` of them, each of width `n_embd = 128`:
"""

# %%
p_ln = sum(p.numel() for p in GPT(vocab_size, norm=LayerNorm).parameters())
p_rms = sum(p.numel() for p in GPT(vocab_size, norm=RMSNorm).parameters())
n_sites = 2 * 4 + 1
print(f"LayerNorm GPT : {p_ln:,} params")
print(f"RMSNorm   GPT : {p_rms:,} params")
print(f"saved         : {p_ln - p_rms:,}  = {n_sites} sites x 128 width"
      f"  = {(p_ln - p_rms) / p_ln:.2%} of the model")

# %% [markdown]
"""
**A rounding error in the parameter count, and no loss difference.** That is a
completely unconvincing reason to change anything, and it would be dishonest to dress it
up — on a char model this size, RMSNorm is not an upgrade, it's a tie. Nine vectors is
noise at any scale, so the parameter count was never the real argument anyway.

The real argument is supposed to be **speed**. Let's go collect it.
"""

# %% [markdown]
"""
## The speed argument, and why it doesn't survive contact with a GPU

Look again at what each norm has to compute:

```-
LayerNorm:  mean = sum(x)/C          -> reduction 1 over the C features
            var  = sum((x-mean)^2)/C -> reduction 2 (it NEEDS the mean first)
            y    = (x - mean) / sqrt(var + eps) * gamma + beta

RMSNorm:    ms   = sum(x^2)/C        -> ONE reduction, no dependency
            y    = x / sqrt(ms + eps) * gamma
```

LayerNorm's two reductions are **serial** — you can't start the variance until the mean
is done — plus it does an extra subtract and an extra add per element. RMSNorm halves
the reductions and skips both. That's a real saving in the arithmetic, and it's the
headline of every RMSNorm explanation you'll read.

Our `n_embd=128` model is far too small to show it, so let's measure at Llama-7B width —
`8192` tokens of width `4096` — and bank the win. We'll time **two** versions of each
norm: torch's (fused CUDA kernels) and the from-scratch ones from the top of this
notebook (plain PyTorch ops, unfused).
"""


# %%
def bench(mod, x, iters=50, bwd=True):
    """Median ms per call, after warmup. CUDA-synced: time the GPU, not the queue."""
    x = x.clone().requires_grad_(bwd)
    for _ in range(10):                                   # warmup / autotune
        y = mod(x)
        if bwd:
            y.sum().backward()
    if DEV == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        y = mod(x)
        if bwd:
            y.sum().backward()
        if DEV == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


WIDTH, TOKENS = 4096, 8192                                # Llama-7B width, a real batch
xb = torch.randn(TOKENS, WIDTH, device=DEV)
print(f"width {WIDTH}, {TOKENS} tokens — median ms of 50\n")
print(f"{'':<24} | {'LayerNorm':>11} | {'RMSNorm':>11} | {'speedup':>8}")
print("-" * 62)
for tag, bwd in [("fwd", False), ("fwd+bwd", True)]:
    for impl, ln_mod, rms_mod in [
        ("torch (fused)", nn.LayerNorm(WIDTH).to(DEV), nn.RMSNorm(WIDTH).to(DEV)),
        ("ours (unfused)", LayerNorm(WIDTH).to(DEV), RMSNorm(WIDTH).to(DEV)),
    ]:
        t_ln, t_rms = bench(ln_mod, xb, bwd=bwd), bench(rms_mod, xb, bwd=bwd)
        print(f"{tag + ', ' + impl:<24} | {t_ln:>9.3f}ms | {t_rms:>9.3f}ms | "
              f"{t_ln / t_rms:>7.2f}x")

# %% [markdown]
"""
### Read that table again — the headline win is **not there**

With torch's fused kernels, RMSNorm is **not faster than LayerNorm**. Not by a few
percent; by nothing at all, forward or backward, at Llama width. Half the reductions,
identical wall-clock. Meanwhile our clumsy unfused versions show a solid ~1.4x forward.
The saving is real in the *arithmetic* and absent from the *hardware*, and the reason is
the single most useful fact about this kind of op.

**A norm is memory-bandwidth bound.** It does ~5 floating-point operations per element
while moving 4 bytes in and 4 bytes out. Arithmetic is not the cost — *moving the data*
is. So:

- **Fused** (torch): the kernel reads each token's `C` features from HBM **once** into
  on-chip memory, does whatever reductions it likes on data that's already there, and
  writes the result **once**. LayerNorm's second reduction happens in registers, at a
  cost of essentially zero HBM traffic. Same bytes in, same bytes out, same time. The
  arithmetic saving is real and *free to skip* — you were never paying for it.
- **Unfused** (ours): every intermediate is a separate kernel launch round-tripping
  through HBM — `x.mean()` reads `x`, `x.var()` reads `x` **again**, `(x - mean)` reads
  and writes, `/sqrt` reads and writes, `* weight` reads and writes. Now each extra
  operation is extra *traffic*, and LayerNorm's extra steps cost real time. Hence 1.4x.

Which reframes the famous result. Zhang & Sennrich reported 7–64% speedups in 2019 — on
RNN implementations that looked like **our unfused column**, because that's what the
frameworks did then. Their measurement was honest and their code was the norm of its
day. Then kernel fusion arrived and quietly ate the entire argument.
"""

# %% [markdown]
"""
### So why did everyone switch?

Not for the speed — we just failed to find it. The honest answer is the least dramatic
one, and it's the reason to trust it:

**RMSNorm is the same, with less in it.** It matches LayerNorm's quality (measured:
inside seed noise), and it deletes two things — a mean and a bias — that we *proved* the
next `Linear` can do for itself. Given a tie, you take the version with fewer moving
parts: one less parameter tensor to init, decay, shard, checkpoint and quantize; one
less opportunity for a bug. When Llama picked it up the ecosystem followed, and there
has never been a reason to add the centering back.

That's a genuinely useful shape of engineering result to have seen: **a change that
survives because it costs nothing, not because it wins anything.** Most of Phase B is
not like this one — `10` and `11` both buy real, measurable capability. But "simpler,
and provably loses nothing" is its own kind of argument, and it's the whole of this
notebook.

(Caveat, honestly: this is one GPU, one torch version, one shape. RMSNorm can still edge
ahead where the kernel *isn't* fused, where memory is tight, or in the backward pass of
some implementations. The transferable lesson isn't the exact ratio — it's that on a
bandwidth-bound op, counting arithmetic predicts nothing, and you have to measure.)
"""

# %% [markdown]
"""
## The payoff: read the RMSNorm model

The last thing left is the only one that matters to a reader. Train the RMSNorm GPT and
let it write, using `08`'s decode settings (`T=0.8`, `top_p=0.9`). Same Shakespeare, one
fewer operation per norm, half the norm parameters, and text you can't tell apart from
`08`'s.
"""


# %%
def nucleus_mask(probs, p):
    """08's top-p: the smallest set of tokens whose cumulative probability reaches p."""
    p = p + 1e-6
    sorted_probs, sorted_idx = probs.sort(descending=True, dim=-1)
    prefix = sorted_probs.cumsum(dim=-1) - sorted_probs
    drop = torch.zeros_like(prefix > p).scatter(-1, sorted_idx, prefix > p)
    return ~drop


@torch.no_grad()
def generate(m, idx, max_new_tokens, temperature=0.8, top_p=0.9):
    """08's decode step, condensed: temperature -> softmax -> top-p -> multinomial."""
    for _ in range(max_new_tokens):
        logits = m(idx[:, -m.block_size:])[0][:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        probs = probs.masked_fill(~nucleus_mask(probs, top_p), 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
    return idx


torch.manual_seed(1337)
rms_model = train(GPT(vocab_size, norm=RMSNorm).to(DEV))
rms_model.eval()
print(f"RMSNorm GPT: val {full_val_loss(rms_model):.3f}  "
      f"(LayerNorm baseline: {full_val_loss(baseline):.3f})\n")
print("=" * 78)
start = torch.tensor([[stoi["\n"]]], device=DEV)
print(decode(generate(rms_model, start, 500)[0].tolist()))

# %% [markdown]
"""
## What you built, and where `10` goes next

The first Phase B delta is in, and it's a *deletion*:

- **LayerNorm does three things** — re-center, re-scale, affine. **RMSNorm keeps one and
  a half**: divide by the raw root-mean-square (`mean(x^2)`, which is *not* the variance
  unless the mean is already 0) and apply `gamma`. No mean pass, no `beta`.
- **why the centering is free, three ways.** Empirically it deletes only a few percent
  of the typical token vector (`|mean|/rms`), so the two layers were computing nearly
  the same function anyway. Structurally it's a **linear projection** `P`, and the next
  `Linear` can fold it in: `W(Px) = (WP)x` — proved by fusion to ~0. And `beta` is
  *exactly* a duplicate of the next layer's bias, `W(z+beta)+b = Wz+(W*beta+b)` — also
  proved by fusion.
- **why the re-scaling isn't** — it's nonlinear, and `RMSNorm(2x) == RMSNorm(x)` is a
  scale-invariance no matrix can imitate. RMSNorm deletes the parts a `Linear` could
  have done and keeps the part it couldn't. That's the whole design.
- **measured honestly** — 3 seeds each, and the gap between the norms came out **smaller
  than the gap between seeds of the same norm**. A tie, for a fraction of a percent of
  the parameters.
- **and the famous speed win didn't reproduce.** At Llama width, torch's fused LayerNorm
  and fused RMSNorm are the same speed — while our *unfused* versions show ~1.4x. A norm
  is memory-bandwidth bound: a fused kernel reads and writes the same bytes either way,
  so the second reduction is free and skipping it saves nothing. The 2019 paper's 7–64%
  was real for 2019's unfused code, and fusion erased it. RMSNorm survives because it's
  **the same with less in it** — not because it's fast.

That last point is the one to keep: on a bandwidth-bound op, **counting arithmetic
predicts nothing**. Measure, or you'll repeat a speedup that stopped being true years
ago.

That's also the Phase B pattern in miniature: a modern LLM is a GPT-2 with the fat
trimmed off the parts that turned out not to be load-bearing.

The next delta goes the other way — instead of deleting a piece of the FeedForward,
**SwiGLU** *adds* a branch to it. But we owe a debt before we can read that change. `06`
built the FFN as `Linear → GELU → Linear` and justified the `4x` expansion and the
position-wise property, while saying almost nothing about the GELU sitting in the middle;
SwiGLU replaces exactly that. So `10` asks the prior question — **why is there a
nonlinearity there at all?** — and the answer is not a detail: two stacked `Linear`s are
one `Linear`, so without it the entire `4x` hidden layer collapses into a single matrix.
Then *which* one, where the zoo (ReLU, GELU, SiLU, tanh, sigmoid) collapses into **one
knob** and a surprise: SiLU, the activation Llama runs and the one inside SwiGLU, loses to
plain GELU here. `11` then builds SwiGLU on top of that.
"""
