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
# LLM · 07 — the training loop, properly

`06` finished the *architecture*: the FeedForward, the residual, pre-norm, the Block —
four stacked Blocks reproduced `01`'s **1.594** and there is nothing structural left to
build. `01`'s box is fully open. What's left is the other half of getting a low number:
**how well you drive the model you already have.**

Every notebook since `03` has called the same tiny `train()` on faith:

    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)   # one flat LR, decay on everything
    for it in range(max_iters):
        ... loss.backward(); opt.step()                # no init, no schedule, no clip

That plain loop was fine for *seeing ideas work*. It is not how a GPT is actually
trained. This notebook opens it and turns four knobs — in the order they bite during a real
run. Two of them **set this run's number**, and two are **correctness / insurance** whose
payoff is a clean start and a run that doesn't blow up, not a lower digit on a 3000-step
toy. We measure all four honestly, and say plainly which is which:

1. **Weight init** *(diagnostic + scale)* — before a single gradient step, a freshly-built
   model should sit at loss `≈ ln(vocab)` (it knows nothing, so it should predict
   ~uniform). The PyTorch defaults miss it — and, because `ln_f` normalizes the stream
   before the head, the fix is *entirely* the **output head's** init scale (GPT-2's
   `std=0.02`); the separate `1/sqrt(2·n_layer)` residual-projection scaling is a *depth*
   stabilizer this 4-layer toy can't even show move. That `ln(vocab)` sanity check is the
   transferable win; on *this* toy the init doesn't lower the final loss at all (measured)
   — its payoff is at depth/scale.
2. **AdamW & decoupled weight decay** *(sets the number)* — *why* Adam over SGD, measured:
   AdamW reaches a loss SGD can't touch even at 30× the LR. Plus the nanoGPT two-group
   trick — decay the 2-D weight matrices, never the 1-D biases and LayerNorm gains — where
   the "W" is decay kept *out* of the adaptive update.
3. **LR warmup + cosine decay** *(sets the number)* — a fresh Adam's moment estimates are
   garbage for the first few steps (warmup eases in) and the schedule lets us safely run a
   *higher* peak LR than a flat run dares; a slow cosine finish then lets the model settle.
   Measured against a flat LR.
4. **Gradient clipping** *(insurance)* — cap the global grad norm so one freak batch can't
   spike the weights to NaN. We log the norm and watch it fire on the early spikes and go
   quiet after — cheap cover that rarely changes the toy's number but saves the bad-luck
   run.

Then the payoff: assemble all four into `train()`, retrain `06`'s exact GPT, and put the
properly-driven curve next to the plain one — same model, lower number. Each knob then gets
turned back *off*, one at a time, so you see what each was actually worth.

No new architecture here — same `06` model, same data, same deterministic metric. Just
the training craft.
"""

# %%
from pathlib import Path

import math

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
DATA = ROOT / "nb" / "llm" / "data" / "input.txt"  # tinyshakespeare, shared with 01–06
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
print(f"device: {DEV}")

# %% [markdown]
"""
## Same data skeleton as `03`–`06`

Nothing changes here — character tokenizer, one long id tensor, 90/10 split, random
`BLOCK`-length chunks with targets shifted one step left. Every training run below reads
through this exact `get_batch`, so their losses land on the same scoreboard as the bigram,
concat, attention, and `06`'s reproduced GPT.
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

BLOCK = 64        # sequence length per row
BATCH = 32


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK, (BATCH,))
    x = torch.stack([d[i:i + BLOCK] for i in ix])          # (BATCH, BLOCK)
    y = torch.stack([d[i + 1:i + 1 + BLOCK] for i in ix])  # targets = x shifted +1
    return x.to(DEV), y.to(DEV)


print(f"corpus {len(text):,} chars, vocab {vocab_size}, block {BLOCK}")

# %% [markdown]
"""
## The model `06` handed us: the 4-Block GPT

We're tuning *how this trains*, not what it is, so here it is unchanged — the exact model
`06` assembled and used to recover 1.594. Read it as a black box; the only detail that
matters for `07` is flagged in the code: the two projections that **write back into the
residual stream** (`attn.proj` and the FeedForward's output Linear) each carry a
`RESIDUAL_PROJ = True` tag. Knob 1 (weight init) is the first thing that reads it.
"""


# %%
class CausalSelfAttention(nn.Module):
    """All heads at once: qkv projection, reshape to (B, nh, T, hs), masked softmax."""

    def __init__(self, n_embd, n_head, block_size=BLOCK):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.proj.RESIDUAL_PROJ = True            # writes back into the residual stream
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        nh, hs = self.n_head, self.head_size
        q, k, v = self.qkv(x).split(self.n_embd, dim=-1)
        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) * hs ** -0.5           # (B, nh, T, T)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(scores, dim=-1)
        out = w @ v                                             # (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    """Position-wise MLP: expand 4x, GELU, project back into the residual stream."""

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.proj.RESIDUAL_PROJ = True            # writes back into the residual stream
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """One transformer layer: pre-norm residual attention, then pre-norm residual MLP."""

    def __init__(self, n_embd, n_head, block_size=BLOCK, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """`06`'s core architecture: embed -> N Blocks -> final norm -> vocab head."""

    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=BLOCK):
        super().__init__()
        self.block_size = block_size
        self.n_layer = n_layer
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# %% [markdown]
"""
## Knob 1 — weight init: land the untrained model at `ln(vocab)`

Here is a sanity check you can run *before training touches the model at all*, and it
catches more init bugs than any other single number. A freshly-built LM knows nothing, so
for any input it should predict **roughly uniform** over the vocab. The cross-entropy of a
uniform prediction over `V` classes is:

    -ln(1/V) = ln(V)                       # ln(65) ≈ 4.174 here

So the *first forward pass, no training*, should read `≈ ln(vocab)`. If it reads much
higher, the model starts out **confidently wrong** — the logits are large and spread out,
so a wrong class gets high probability and the loss with it. Training then has to spend its
first hundreds of steps just walking that back before it can learn anything. A loss far
*below* `ln(vocab)` at init would be even more suspicious (it can't know anything yet).

Let's read it with the model as PyTorch builds it by default.
"""

# %%
ln_vocab = math.log(vocab_size)


@torch.no_grad()
def init_loss(model):
    """Loss on one fresh batch, no training — the untrained-model sanity read."""
    model.eval()
    xb, yb = get_batch("val")
    loss = model(xb, yb)[1].item()
    model.train()
    return loss


torch.manual_seed(0)
default_model = GPT(vocab_size).to(DEV)
print(f"target  ln(vocab)         : {ln_vocab:.4f}")
print(f"PyTorch-default init loss : {init_loss(default_model):.4f}   "
      f"(over by {init_loss(default_model) - ln_vocab:+.3f})")

# %% [markdown]
"""
### Why the default overshoots — it's the output head, and *only* the head

The whole path from network to loss is one line: `logits = lm_head(ln_f(x))`. And `ln_f`
is a LayerNorm — it renormalizes `x` to unit scale *per token* right before the head. So
**however large or small the residual stream got, `lm_head` always sees a unit-scale
input** (we measure `std(ln_f(x)) ≈ 1.000` below, for any init). The head is blind to the
stream's magnitude.

That single fact pins down the cause. With the head's input fixed at unit scale, the spread
of the logits is set *entirely* by `lm_head`'s own weight scale. PyTorch's default `Linear`
init draws weights with std `≈ 1/sqrt(n_embd) ≈ 0.051`, which makes the untrained logits
spread with std `≈ 0.58` — wide enough that softmax is visibly non-uniform, so cross-entropy
against random targets sits *above* `ln(vocab)`. Shrink just that one matrix to std `0.02`
and the logits collapse toward uniform and the loss onto `ln(vocab)`.

Nothing upstream matters for this number: not the `N(0,1)` embeddings, not `06`'s
residual-stream growth — `ln_f` erases all of it before the head. (The textbook-dramatic
version of this bug is **weight tying**: if `lm_head` *shares* the embedding's `N(0,1)`
weights, the logits truly explode. Here the two are separate, so it's only the milder
`0.051` default.) Let's isolate it — fix `lm_head` alone, leave everything else at PyTorch
defaults, and watch the init loss land anyway:
"""


# %%
@torch.no_grad()
def head_stats(model):
    """std of the head's INPUT (after ln_f) and of the resulting logits, on one batch."""
    xb, _ = get_batch("val")
    x = model.tok_emb(xb) + model.pos_emb(torch.arange(xb.size(1), device=DEV))
    for b in model.blocks:
        x = b(x)
    head_in = model.ln_f(x)
    return head_in.std().item(), model.lm_head(head_in).std().item()


torch.manual_seed(0)
dm = GPT(vocab_size).to(DEV)
h_std, lg_std = head_stats(dm)
print(f"default: std(ln_f output) = {h_std:.3f}  <- ~1 whatever the stream did (ln_f norms)")
print(f"default: lm_head.weight std {dm.lm_head.weight.std():.4f} -> logit std {lg_std:.3f}")

torch.manual_seed(0)
hm = GPT(vocab_size).to(DEV)                        # everything at PyTorch defaults...
nn.init.normal_(hm.lm_head.weight, std=0.02)        # ...except lm_head, shrunk to 0.02
nn.init.zeros_(hm.lm_head.bias)
_, lg2 = head_stats(hm)
print(f"lm_head@0.02 ONLY: logit std {lg2:.3f} -> init loss {init_loss(hm):.4f}  "
      f"(≈ ln vocab {ln_vocab:.4f}; nothing else touched)")

# %% [markdown]
"""
### The GPT-2 init function

The real recipe applies `std=0.02` to *every* Linear and Embedding — the extra ones are
harmless for the init loss (`ln_f` and the pre-norms normalize their scale away) but are
the scheme that trains at scale — plus a `1/sqrt(2·n_layer)` factor on the residual
projections we'll examine next. Building the full thing confirms it also lands on
`ln(vocab)`:
"""


# %%
def apply_gpt2_init(model):
    """GPT-2 weight init: N(0, 0.02) everywhere, residual projections scaled down by
    1/sqrt(2·n_layer) so the residual stream doesn't blow up through depth. Turning
    this OFF = leaving PyTorch's defaults (the overshoot above)."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "RESIDUAL_PROJ", False):
                std *= (2 * model.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    return model


torch.manual_seed(0)
init_model = apply_gpt2_init(GPT(vocab_size).to(DEV))
print(f"target  ln(vocab)         : {ln_vocab:.4f}")
print(f"GPT-2 init loss           : {init_loss(init_model):.4f}   "
      f"<- lands on ln(vocab): the model starts out honestly uncertain")

# %% [markdown]
"""
### What the `1/sqrt(2·n_layer)` residual scaling is actually for

So if the head can't see the stream, why scale the residual projections at all? Because it
targets a *different* problem, one `ln_f` doesn't hide: the size of the residual stream
*inside* the stack. Each Block adds `attn(...)` and `ffwd(...)`, so over `2·n_layer`
additions the stream's variance accumulates (`06`'s lesson). GPT-2 shrinks each residual
projection by `1/sqrt(2·n_layer)` to keep that growth flat — a *depth* stabilizer, not a
logit fix, which is why only the two `RESIDUAL_PROJ`-tagged Linears get the factor.

The demonstration is a **depth sweep**: build this exact model at growing depths, init it
with and without the `1/sqrt(2·n_layer)` scaling, and read the residual RMS entering `ln_f`.
With the scaling the stream is **pinned flat at every depth** (what the factor is calibrated
to do); without it the `2·n_layer` additions pile up and the RMS **climbs like `sqrt(depth)`
— a ~9× spread by depth 48**. That growth is exactly what the scaling cancels, and exactly
why it's a *depth* concern our 4-layer toy barely feels. Yet through the whole sweep **both
columns' init loss stays on `ln(vocab)`**: `ln_f` normalizes the stream away before the
head, so none of this reaches the logits. The two jobs, cleanly separated:
"""


# %%
@torch.no_grad()
def resid_rms(model):
    """RMS of the residual stream entering ln_f, on one fresh batch."""
    xb, _ = get_batch("val")
    x = model.tok_emb(xb) + model.pos_emb(torch.arange(xb.size(1), device=DEV))
    for block in model.blocks:
        x = block(x)
    return x.std().item()


def build_init(n_layer, scale_resid):
    """Fresh GPT at a given depth, 0.02 init, with/without the 1/sqrt(2N) residual scale."""
    torch.manual_seed(0)
    m = GPT(vocab_size, n_layer=n_layer).to(DEV)
    for mod in m.modules():
        if isinstance(mod, nn.Linear):
            std = 0.02
            if scale_resid and getattr(mod, "RESIDUAL_PROJ", False):
                std *= (2 * n_layer) ** -0.5
            nn.init.normal_(mod.weight, std=std)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Embedding):
            nn.init.normal_(mod.weight, std=0.02)
    return m


print("             scaled 1/sqrt(2N)  |  plain 0.02 (no scale)")
print(f"{'depth':>5} | {'resid RMS':>9} {'loss':>6} | {'resid RMS':>9} {'loss':>6}")
print("-" * 45)
for nl in [2, 4, 8, 16, 32, 48]:
    ms, mu = build_init(nl, True), build_init(nl, False)
    marker = "  <- our model" if nl == 4 else ""
    print(f"{nl:>5} | {resid_rms(ms):>9.3f} {init_loss(ms):>6.3f} | "
          f"{resid_rms(mu):>9.3f} {init_loss(mu):>6.3f}{marker}")
print(f"\n(ln vocab = {ln_vocab:.3f}) scaled: RMS pinned ~0.05 at EVERY depth. unscaled: it")
print("climbs ~sqrt(depth), a ~9x spread by depth 48 — the accumulation the scaling cancels.")
print("Both columns' init loss stays on ln(vocab): ln_f hides the stream from the head.")

# %% [markdown]
"""
### Why `sqrt(depth)`? The residual stream is a random walk

Here's the piece that makes the sweep click. We measured two facts: each block *adds* its
sublayer output to the stream (`x = x + delta`), and each delta is the **same size at every
depth** — because the sublayer's input is `ln(x)` (unit scale, always) and its weights are
init'd identically per block, so same-scale-in gives same-scale-out. So why doesn't the
stream grow *linearly*, `2N` blocks × a constant step `δ`? Why `sqrt`?

Because the deltas point in **random, mostly-different directions**, so they don't stack up
— they partly cancel. The one-line reason is that **variances add, standard deviations
don't**: for independent zero-mean vectors,

    Var(a + b) = Var(a) + Var(b)        ->   RMS(a + b) = sqrt(RMS(a)^2 + RMS(b)^2)

not `RMS(a) + RMS(b)`. Apply that to the stream, a sum of `2N` roughly-independent,
equal-size steps:

    Var(stream) ≈ Var(emb) + 2N · δ^2       ->   RMS(stream) ≈ sqrt(2N) · δ

That's a **random walk** (the "drunkard's walk"): take `2N` steps of size `δ` in random
directions and you drift `sqrt(2N)·δ` from the origin, not `2N·δ`. The steps never get
bigger with depth — there are just *more* of them, and a longer random walk drifts as
`sqrt(steps)`. That is the `sqrt(depth)` in the unscaled column.

And it's exactly why `1/sqrt(2·n_layer)` is the cancelling fix: shrink every step by
`1/sqrt(2N)` and the drift becomes `sqrt(2N) · δ/sqrt(2N) = δ` — the *same* distance no
matter how many steps. Let's watch the steps stay flat while the walk accumulates, and
confirm the `sqrt`-sum predicts the stream exactly:
"""


# %%
@torch.no_grad()
def block_deltas(model):
    """Per-block sublayer output RMS (the random-walk 'step') and the running stream RMS.
    Also accumulate Var(emb) + sum(step^2) — the predicted stream variance if the steps
    were independent."""
    xb, _ = get_batch("val")
    x = model.tok_emb(xb) + model.pos_emb(torch.arange(xb.size(1), device=DEV))
    rows, var_pred = [], x.var().item()               # start from the embedding variance
    for blk in model.blocks:
        da_out = blk.attn(blk.ln1(x)); da = da_out.std().item(); x = x + da_out
        df_out = blk.ffwd(blk.ln2(x)); df = df_out.std().item(); x = x + df_out
        var_pred += da ** 2 + df ** 2                  # variances add (independent steps)
        rows.append((da, df, x.std().item()))
    return rows, var_pred ** 0.5


torch.manual_seed(0)
rows, rms_pred = block_deltas(build_init(8, scale_resid=False))   # unscaled, 8 blocks
print("unscaled, 8 blocks — each block adds a ~constant-size step:")
print(f"{'blk':>3} | {'step_attn':>9} {'step_ff':>8} | {'stream RMS':>10}")
for i, (da, df, s) in enumerate(rows):
    print(f"{i:>3} | {da:>9.4f} {df:>8.4f} | {s:>10.3f}")
print(f"\nsteps stay flat with depth; stream = sqrt(emb^2 + sum of steps^2) = {rms_pred:.3f}")
print(f"  -> matches the measured {rows[-1][2]:.3f}: constant steps, sqrt(N) random-walk drift.")

# %% [markdown]
"""
### So do we even need `apply_gpt2_init` here? Honestly, no.

Worth stating plainly. On this 4-layer, 3000-step toy, neither half of the scheme earns its
keep in the loss:

- the `1/sqrt(2·n_layer)` residual scaling does nothing measurable — we just watched it
  leave the init loss untouched, and the ablation at the end shows it costs ~0 in final
  loss too;
- even the `lm_head` fix, which *does* land the init loss on `ln(vocab)`, doesn't lower the
  *final* loss — a small model recovers from PyTorch's larger default init inside a few
  hundred steps. In fact pure PyTorch defaults train to the *same* place (the end ablation's
  "init off" row confirms it, a hair lower if anything).

So why is it in the recipe — and in this notebook? Two reasons a toy can't reward but that
decide real runs. (1) The **`ln(vocab)` sanity check** is the cheapest bug-catcher you have:
a fresh model that *doesn't* read `≈ ln(vocab)` has something wrong (a mis-tied head, a
forgotten scale, an exploded embedding) — you want that signal on every job, and it's the
`lm_head` scale that gives it. (2) The residual scaling is **stability insurance at depth**,
where the accumulation we measured as "mild" turns destabilizing over dozens of layers.
GPT-2 init is the init that *keeps working* as you scale up — not the one that wins a
4-layer toy's third decimal.
"""

# %% [markdown]
"""
## Knob 2 — AdamW: why Adam, and decoupled weight decay

Two separate questions live in the optimizer, and it's worth splitting them.

**(a) Why Adam at all, not plain SGD?** In a transformer, different parameters see wildly
different gradient scales — an embedding row for a rare char barely moves, a LayerNorm gain
gets a strong signal every step. SGD uses one global LR for all of them, so it either
crawls for the small-gradient params or diverges on the large ones. **Adam keeps a
per-parameter running estimate of the gradient's magnitude** (the second moment `v`) and
divides by it, giving each parameter its *own* effective step size. On this model the
difference is not subtle — we measure it once the training loop is defined, below, and even
handing SGD a 30× larger LR doesn't close the gap.

**(b) The "W": decoupled weight decay.** Weight decay is L2 regularization — pull every
weight a little toward 0 each step to curb overfitting. Plain Adam folds that pull *into
the gradient*, so it flows through the `1/sqrt(v)` division and a param with a big gradient
history gets *less* decay — the decay strength becomes an accident of each param's gradient
scale. AdamW applies `w -= lr·wd·w` **directly, outside** the adaptive update, so every
weight decays by the same fraction. That decoupling is literally what the "W" is.

> We drive with `torch.optim` here, but the update rules themselves are only a few lines
> each. `custom/optimizers.py` builds **SGD → Adam → AdamW** from scratch — each the
> previous plus one idea, each validated against its `torch.optim` twin to ~0 — and isolates
> the single line that moves between Adam and AdamW that *is* this paragraph.
"""


# %% [markdown]
"""
### The nanoGPT two-group trick: what gets decayed

One more refinement, and it's a one-liner. You do **not** want to decay *every* parameter:

- the big **2-D weight matrices** (Linear weights, embeddings) hold ~all the params and do
  the actual feature mixing — decay these, it's real regularization;
- the **1-D params** (biases, LayerNorm gain/bias) are few and structural. Decaying a
  LayerNorm gain (init 1.0) toward 0 just *fights the normalization* it exists to do;
  decaying a bias removes the offset it's there to provide, for no benefit.

The clean heuristic (nanoGPT's): **decay every tensor with `dim >= 2`, skip every tensor
with `dim < 2`.** No name-matching, one split by dimensionality.
"""


# %%
def configure_optimizer(model, lr, weight_decay=0.1, betas=(0.9, 0.95)):
    """AdamW with two param groups: decay the >=2-D weights, not the 1-D biases/norms."""
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]      # weight matrices, embeddings
    no_decay = [p for p in params if p.dim() < 2]    # biases, LayerNorm gain/bias
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


# show the partition on 06's GPT
_probe = apply_gpt2_init(GPT(vocab_size))
_opt = configure_optimizer(_probe, lr=3e-4)
_d, _nd = _opt.param_groups
n_d = sum(p.numel() for p in _d["params"])
n_nd = sum(p.numel() for p in _nd["params"])
print(f"decay    (dim>=2): {len(_d['params']):>2} tensors, {n_d:>7,} params  "
      f"wd={_d['weight_decay']}")
print(f"no-decay (dim<2) : {len(_nd['params']):>2} tensors, {n_nd:>7,} params  "
      f"wd={_nd['weight_decay']}")
print(f"  -> {n_nd / (n_d + n_nd) * 100:.1f}% of params (the 1-D ones) are never decayed.")

# %% [markdown]
"""
## Knob 3 — LR warmup + cosine decay

Adam's moment estimates `m, v` both start at **0** and are pure noise for the first handful
of steps — taking a full-size step then, on a fresh model, can knock it into a bad region
it never recovers from. **Warmup** eases in: ramp the LR linearly from ~0 up to the peak
over the first `warmup` steps. Then **cosine decay** glides from the peak down to a small
floor along half a cosine — smoother than a step drop, and the slow finish lets the model
settle into the minimum instead of bouncing around it. The shape every GPT uses:

    lr
  max ┤        .-''''-.
      │      /          `-.
      │    /                `-.
  min ┤  /                      `--------
      └──┬──────────┬──────────────────── step
         0        warmup                max_steps
"""


# %%
def get_lr(step, *, warmup, max_steps, max_lr, min_lr):
    """Linear warmup to max_lr, then half-cosine decay to min_lr, then hold the floor."""
    if step < warmup:                                    # phase 1: linear ramp
        return max_lr * (step + 1) / warmup
    if step > max_steps:                                 # phase 3: clamp to floor
        return min_lr
    ratio = (step - warmup) / (max_steps - warmup)       # phase 2: 0 -> 1 through window
    coeff = 0.5 * (1 + math.cos(math.pi * ratio))        # 1 -> 0 along half a cosine
    return min_lr + coeff * (max_lr - min_lr)


_steps = range(3000)
_curve = [get_lr(s, warmup=100, max_steps=3000, max_lr=3e-3, min_lr=3e-4) for s in _steps]
plt.figure(figsize=(6, 2.6))
plt.plot(_steps, _curve)
plt.axvline(100, ls="--", c="gray", lw=0.8)
plt.text(120, 3e-3 * 0.5, "warmup ends", fontsize=8, color="gray")
plt.xlabel("step")
plt.ylabel("learning rate")
plt.title("warmup (100) → cosine decay to min_lr")
plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Knob 4 — gradient clipping: cap the step so one bad batch can't wreck the run

Deep nets — transformers especially, early in training — occasionally hit a batch that
produces a **huge** gradient: a loss spike, an outlier sequence, an unlucky init. One
oversized step can blow the weights to NaN and destroy the whole run. **Gradient clipping**
caps the step size so a single bad batch can't.

It's done **by global norm**: treat *all* gradients as one big vector `g`, and if its total
L2 norm exceeds `max_norm`, scale the whole vector down by the same factor:

    total_norm = sqrt( sum over every grad element of g_i^2 )
    if total_norm > max_norm:
        g *= max_norm / total_norm        # every grad scaled by the SAME factor

One shared factor **preserves the direction** of the update (the relative sizes between
params) and only shortens its length — per-element clamping would distort the direction,
which is the whole reason we clip by norm. `torch.nn.utils.clip_grad_norm_` does exactly
this and *returns the pre-clip norm*, which is what we log to watch for spikes. We'll see
below that the un-clipped norm is largest in the first few steps (the fresh-model danger
zone warmup also targets) and settles after.
"""

# %% [markdown]
"""
## Assemble it: one configurable `train()`

All four knobs in one loop, each behind a flag so we can ablate them. It logs the val curve
(same deterministic `full_val_loss` as `03`–`06`) and the per-step **pre-clip grad norm**,
and returns both so we can plot. The plain baseline = every knob off: PyTorch-default init,
flat LR, `weight_decay=0`, no clip — i.e. `03`–`06`'s loop.
"""


# %%
@torch.no_grad()
def full_val_loss(m, bs=256):
    """Mean cross-entropy over the WHOLE validation split in non-overlapping BLOCK
    windows — the identical deterministic metric `03`–`06` used."""
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


def train(m, *, optimizer="adamw", gpt2_init=True, weight_decay=0.1, schedule=True,
          grad_clip=1.0, max_lr=3e-3, min_lr=3e-4, warmup=100, max_iters=3000,
          eval_every=500, log=True):
    """Train `m` with the four knobs individually switchable. Returns
    (final_val, val_curve, gradnorm_trace). Every knob off == the plain 03–06 loop.
    optimizer='sgd' swaps AdamW for SGD+momentum (the Knob-2 counterfactual)."""
    if gpt2_init:
        apply_gpt2_init(m)
    if optimizer == "sgd":
        opt = torch.optim.SGD(m.parameters(), lr=max_lr, momentum=0.9)
    else:
        opt = configure_optimizer(m, lr=max_lr, weight_decay=weight_decay)
    val_curve, grad_trace = [], []
    for it in range(max_iters + 1):
        if schedule:                                     # knob 3: set this step's LR
            lr_now = get_lr(it, warmup=warmup, max_steps=max_iters,
                            max_lr=max_lr, min_lr=min_lr)
            for g in opt.param_groups:
                g["lr"] = lr_now
        if it % eval_every == 0:
            v = full_val_loss(m)
            val_curve.append((it, v))
            if log:
                print(f"  step {it:>4} : val {v:.3f}")
        xb, yb = get_batch("train")
        _, loss = m(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:                                # knob 4: cap the global norm
            gn = torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip)
            grad_trace.append(gn.item())
        opt.step()
    return full_val_loss(m), val_curve, grad_trace


# %% [markdown]
"""
### Knob 2, measured: AdamW vs SGD

Back to the "why Adam" claim, now that we can train. Same model, same warmup+cosine
schedule, same clipping — only the optimizer changes. SGD gets **momentum and a peak LR
30× higher** than AdamW's, every advantage short of adaptivity, and still can't reach it.
That gap is Adam's per-parameter step size doing its job on a model whose gradients span
very different scales.
"""

# %%
torch.manual_seed(1337)
adamw_m = GPT(vocab_size).to(DEV)
adamw_final, _, _ = train(adamw_m, optimizer="adamw", max_lr=3e-3, log=False)

torch.manual_seed(1337)
sgd_m = GPT(vocab_size).to(DEV)
sgd_final, _, _ = train(sgd_m, optimizer="sgd", weight_decay=0.0, max_lr=1e-1, log=False)

print(f"AdamW (peak 3e-3)          : {adamw_final:.3f}")
print(f"SGD+momentum (peak 1e-1)   : {sgd_final:.3f}   "
      f"<- 30x the LR, still {sgd_final - adamw_final:+.3f} behind")

# %% [markdown]
"""
### The measured payoff: properly-driven vs the plain loop

Two runs of `06`'s exact GPT, same seed, same 3000 iters. **Plain** is every knob off (what
we've used on faith since `03`); **well-driven** turns all four on. Watch the well-driven
curve start *lower* (Knob 1: it begins at ln(vocab) instead of already-confused) and end
lower.
"""

# %%
torch.manual_seed(1337)
plain = GPT(vocab_size).to(DEV)
print("plain loop (every knob OFF): default init, flat LR, no decay, no clip")
plain_final, plain_curve, _ = train(
    plain, gpt2_init=False, weight_decay=0.0, schedule=False, grad_clip=0.0,
    max_lr=1e-3,          # the flat 1e-3 the old loop used
)

torch.manual_seed(1337)
well = GPT(vocab_size).to(DEV)
print("\nwell-driven (every knob ON): GPT-2 init, warmup+cosine, wd=0.1, clip=1.0")
well_final, well_curve, well_grads = train(well)

print(f"\nplain loop        : {plain_final:.3f}")
print(f"well-driven       : {well_final:.3f}   <- same model, same steps, better driven")

# %%
plt.figure(figsize=(6.5, 3.2))
plt.plot(*zip(*plain_curve), "o-", label=f"plain ({plain_final:.3f})", ms=3)
plt.plot(*zip(*well_curve), "o-", label=f"well-driven ({well_final:.3f})", ms=3)
plt.xlabel("step")
plt.ylabel("val loss")
plt.legend()
plt.title("same GPT, plain loop vs all four knobs on")
plt.tight_layout()
plt.show()

# %% [markdown]
"""
### The grad-norm trace: where clipping actually fires

The pre-clip global grad norm from the well-driven run, per step. It's largest in the first
few dozen steps — the fresh-model danger zone — where the odd batch pokes above the
`clip=1.0` line and gets scaled back; after warmup it settles well under the cap and
clipping mostly stops firing. That early spike is exactly the "one bad step wrecks a fresh
model" risk clipping (and warmup) exist to cover.
"""

# %%
plt.figure(figsize=(6.5, 2.8))
plt.plot(well_grads, lw=0.6)
plt.axhline(1.0, ls="--", c="red", lw=0.9, label="clip threshold = 1.0")
plt.xlabel("step")
plt.ylabel("pre-clip grad norm")
plt.yscale("log")
plt.legend()
plt.title("global grad norm per step (clipped back to 1.0 when above)")
plt.tight_layout()
plt.show()
fired = sum(g > 1.0 for g in well_grads)
early = sum(g > 1.0 for g in well_grads[:100])
print(f"clipping fired on {fired}/{len(well_grads)} steps "
      f"({early} of them in the first 100).")

# %% [markdown]
"""
## Turn each knob back off, one at a time

Start from the all-on `train()` and disable exactly one setting per run, same seed and
steps, and read the final val as a signed delta against the all-on baseline. This is the
honest test of what each is worth *on this model* — and honesty means reporting the two
knobs whose delta is ~0 (or even slightly negative) just as plainly as the one that bites.
The header already sorted them into "sets the number" vs "correctness/insurance"; this is
that sort, measured.
"""


# %%
def run(label, base=None, **overrides):
    torch.manual_seed(1337)
    m = GPT(vocab_size).to(DEV)
    final, _, _ = train(m, log=False, **overrides)
    delta = f"  ({final - base:+.3f})" if base is not None else ""
    print(f"  {label:<30}: {final:.3f}{delta}")
    return final


print("ablation — all knobs on, then ONE turned off per row:\n")
base = run("all four knobs on")
run("schedule off (flat max_lr)", base, schedule=False)
run("weight decay off (wd=0)", base, weight_decay=0.0)
run("GPT-2 init off (defaults)", base, gpt2_init=False)
run("grad clip off", base, grad_clip=0.0)
print(f"\nbaseline (all on) = {base:.3f}. Delta = what dropping ONE knob costs (+) or "
      f"saves (-).")

# %% [markdown]
"""
Reading the table (exact numbers wobble a little with hardware/seed; the *sort* is the
point, and it matches the header's split):

- **schedule off** is the one that clearly bites (`+0.1`-ish). A flat LR must either stay
  conservative (slow) or go aggressive (unstable at the end); warmup+cosine gets the fast
  middle *and* the clean finish — and it's what let us safely run the 3e-3 peak in the first
  place. This is the biggest single lever on this run's number (with the AdamW-vs-SGD choice
  measured above).
- **weight decay off** barely moves it — a *tiny* char model on a *short* run doesn't
  overfit, so the regularizer has little to do. It's in the recipe for the larger,
  trained-to-convergence regime, not to rescue this toy.
- **GPT-2 init off** comes out ~flat or even *slightly lower* — exactly the honest caveat
  from Knob 1. This model recovers from PyTorch's larger default init inside 3000 steps, so
  the final digit doesn't reward the safe `0.02` init. Its payoff is the `ln(vocab)` sanity
  check and stability at *depth* — neither of which a 4-layer toy's final loss can show.
  Keep it anyway: you're learning the recipe that works at scale, and the sanity check earns
  its place on every real run.
- **grad clip off** is also ~free here — a well-init'd small model rarely throws a norm big
  enough to matter (the trace above fired on ~half the first 100 steps, then almost never —
  a handful out of 3000). It's *insurance*: cheap, and the one run in a hundred where an
  early spike would have gone to NaN is the one it pays for.

The takeaway is the real one, not a tidy one: on this scale the **optimizer** and the **LR
schedule** set the number; **init** and **clipping** are correctness and insurance whose
value you verify at the start (`ln(vocab)`) and on the tail-risk step (the grad trace), and
cash in when you scale the model up. That's why all four are in every serious training
recipe — even the two a toy can't reward.
"""

# %% [markdown]
"""
## Read the babble from the well-driven model

Same qualitative check as `06`, now from the properly-trained GPT — seed a newline, sample
400 chars. It's the same tiny char model, so still Shakespeare-*shaped* rather than
sensible, but the better-driven run gets there on fewer wasted steps.
"""

# %%
start = torch.tensor([[stoi["\n"]]], device=DEV)
print(decode(well.generate(start, max_new_tokens=400)[0].tolist()))

# %% [markdown]
"""
## What you built, and where `08` goes next

The `train()` call we'd reused on faith since `03` is fully open, and each piece is now a
knob you've measured — including *where* it does its work:

- **AdamW** *(sets the number)* — per-parameter adaptive steps beat SGD here by a margin SGD
  can't close at 30× the LR; with **decoupled** weight decay via the nanoGPT two-group
  split: decay the 2-D weights, never the 1-D biases and norms.
- **LR warmup + cosine decay** *(sets the number)* — ease past the fresh-Adam danger zone,
  which lets you safely run a higher peak LR, then glide to a floor so the model settles.
  The clearest single lever in the ablation.
- **weight init** *(diagnostic + scale)* — GPT-2's `std=0.02`, residual projections scaled
  by `1/sqrt(2·n_layer)`, lands the untrained model at `ln(vocab)` and holds `06`'s residual
  stream flat. Doesn't lower this toy's final loss — measured — but it's the sanity check
  you run on every job and the init that actually trains at depth.
- **gradient clipping** *(insurance)* — cap the global norm so one freak batch can't NaN the
  run; we watched it fire on the early spikes (~half the first 100 steps) and go quiet after
  (a handful of the remaining 2900). Near-free here, decisive on the bad-luck run.

The model is built (`04`–`06`) and now *well-trained* (`07`). What's left is the other end:
turning the trained model's logits into text. `08` opens `generate`'s sampling — **greedy**
as the `T→0` limit, **temperature** to sharpen or flatten, **top-k** to keep the k most
likely, and **top-p / nucleus** to keep the smallest set summing to `p` — and shows how each
knob changes the samples, and which ones matter for open-ended text vs exact-answer tasks.
"""
