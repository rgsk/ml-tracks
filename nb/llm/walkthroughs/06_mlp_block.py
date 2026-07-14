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
# LLM · 06 — the MLP and the Block

`05` ended on a deliberately honest scoreboard. A single attention head, and even four
heads, cleared the bigram floor but **did not beat `04`'s concat** (1.84). The reason
was stated, not yet fixed: **attention only mixes.** It moves information *between*
tokens — a learned, causal, weighted average of the past — and then hands each position
a single *linear* readout to the vocabulary. There is no place in that model for a token
to **think on its own** once it has gathered its context.

This notebook adds that missing half and then the wrapper that lets you stack it:

1. The **FeedForward** (a per-token MLP): `Linear → GELU → Linear`, applied
   *independently at every position*. Attention gathers; the MLP computes. We prove it's
   position-wise and measure the drop it buys over attention-alone — though a *single*
   layer still trails concat. Beating concat takes depth, which is the rest of the
   notebook.
2. The **depth problem**, measured: naively stacking `attn → MLP` layers doesn't train —
   the forward signal and the backward gradient both fall apart with depth. We *see* it
   happen on real tensors.
3. The two fixes that solve it, each measured: the **residual** (`x + sublayer(x)`)
   gives the gradient a straight path through depth, and **pre-norm** (`LayerNorm` on
   each sublayer's *input*) keeps every sublayer well-scaled no matter how tall the
   stack.
4. Assemble the **Block** — `x = x + attn(ln1(x)); x = x + ffwd(ln2(x))` — stack four of
   them, and **recover `01`'s 1.594**. That's the whole core architecture, built.

Still no RoPE, no KV-cache, and the training loop is still the plain one from `03`–`05`
(AdamW, fixed LR) — those are `07`, `11`, `12`. One delta at a time: this notebook is
the **MLP and the Block**.
"""

# %%
from pathlib import Path

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
DATA = ROOT / "nb" / "llm" / "data" / "input.txt"  # tinyshakespeare, shared with 01–05
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
print(f"device: {DEV}")

# %% [markdown]
"""
## Same data skeleton as `03`–`05`

Character tokenizer, one long id tensor, 90/10 split, random `BLOCK`-length chunks with
targets shifted one step left. Every model below trains through this exact `get_batch`,
so its loss lands on the same scoreboard as the bigram, concat, and `05`'s attention.
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
## The box `05` handed us: fused causal self-attention

We build *on top of* attention, so here it is unchanged — the fused
`CausalSelfAttention` `05` ended on (all heads in a couple of batched matmuls, validated
against `F.scaled_dot_product_attention` to ~0). Nothing new; we just need it as a
component. Read it as a black box that does one thing: **mix each token with a learned,
causal weighted average of the tokens before it**, in and out at width `n_embd`.
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


# %% [markdown]
"""
## The missing half: a per-token MLP

Attention's output at position `t` is a weighted sum of *other tokens'* values — a mix.
But mixing is linear routing; the model still needs to take each mixed vector and
**transform it nonlinearly, on its own**. That's the FeedForward: the *same* little MLP
applied independently to every position.

    C  --Linear-->  4C  --GELU-->  4C  --Linear-->  C

Three design choices, each with a reason:

- **Expand to `4C`, then back.** The hidden layer is 4× wider than the residual stream
  (the GPT-2 ratio). Attention has no per-token nonlinearity at all; this wide hidden
  layer is where most of the model's *knowledge* — the learned per-token transformations
  — actually lives. (In a real GPT the FeedForward holds ~2/3 of the parameters.)
- **GELU**, not ReLU. A smooth gate — `x · Φ(x)` — that GPT-2 and most transformers use;
  the smoothness gives cleaner gradients than ReLU's hard corner.
- **Position-wise.** No token index appears anywhere in it. Row `t` of the output
  depends *only* on row `t` of the input — the exact opposite of attention, which is
  *all* about cross-position interaction. Attention is the only place tokens talk; the
  MLP is where each one thinks in private. We prove that next.

(A `Dropout` sits at the end for regularization; we set `dropout=0` here so the measured
numbers are deterministic.)
"""


# %%
class FeedForward(nn.Module):
    """Position-wise MLP: expand 4x, GELU, project back into the residual stream."""

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# %% [markdown]
"""
### See it: the MLP is position-wise, attention is not

The claim that separates the two sublayers, made falsifiable. Take a batch, perturb
**only the last position**, and re-run each module. If a module is position-wise, every
*earlier* output is byte-for-byte unchanged — information cannot have crossed positions.

- The **FeedForward** must pass: it never looks across the time axis, so touching row
  `T-1` can't move rows `0..T-2`.
- **Attention** must *fail* this exact test in the forward direction — mixing across
  positions is its entire job. (It stays causal the *other* way: the last token can read
  earlier ones, just not vice-versa. Here we perturb the last token and look at earlier
  outputs, which attention leaves alone; so we perturb an *earlier* token instead to
  show attention genuinely moves information forward.)

Two modules, the same probe, opposite answers — that contrast *is* the division of
labor.
"""

# %%
torch.manual_seed(0)
B, T, C = 4, BLOCK, 48
x = torch.randn(B, T, C)

ff = FeedForward(C).eval()
attn = CausalSelfAttention(C, n_head=4).eval()

# perturb ONLY the last position
x_last = x.clone()
x_last[:, -1, :] = torch.randn(B, C)
ff_pw = torch.allclose(ff(x)[:, :-1], ff(x_last)[:, :-1], atol=1e-6)
print(f"FeedForward: perturbing pos T-1 leaves earlier outputs unchanged? {ff_pw}")
print("  -> position-wise: each token is transformed on its own.")

# perturb an EARLY position; a later output must move (attention read it)
x_early = x.clone()
x_early[:, 0, :] = torch.randn(B, C)
attn_moves = not torch.allclose(attn(x)[:, -1], attn(x_early)[:, -1], atol=1e-6)
print(f"\nAttention: perturbing pos 0 changes the LAST output?          {attn_moves}")
print("  -> cross-position: it mixes info forward. That's the half the MLP can't do.")

# %% [markdown]
"""
## Does the MLP actually help? One attn+MLP layer vs `05`'s attention-only

The direct test. Take `05`'s multi-head attention LM and insert one FeedForward after
the attention — a *single* layer, `attn → MLP`, no residual or norm yet (one layer
trains fine without them; we add those for *depth*, next). Train both — with and without
the MLP — on the same loop and compare to `04`'s concat.

The prediction from the theory: attention alone gives a *linear* readout over a mixed
vector; adding a nonlinear per-token transform should measurably lower the loss. But
note the honest ceiling — *one* attn+MLP layer improves on attention-alone yet **still
trails concat's 1.84**. A single mixing layer isn't enough compute; crossing concat is
what **depth** buys, and that's the payoff waiting at the end of this notebook.
"""


# %%
class OneLayerLM(nn.Module):
    """token+pos embed -> attention -> per-token MLP -> vocab.
    One layer, no residual."""

    def __init__(self, vocab_size, n_embd=64, n_head=4, mlp=True, block_size=BLOCK):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ffwd = FeedForward(n_embd) if mlp else None
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.attn(x)
        if self.ffwd is not None:
            x = self.ffwd(x)
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


# %%
@torch.no_grad()
def full_val_loss(m, bs=256):
    """Mean cross-entropy over the WHOLE validation split in non-overlapping BLOCK
    windows — the identical deterministic metric `03`–`05` used."""
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


def train(m, max_iters=3000, eval_every=1000, lr=1e-3):
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for it in range(max_iters + 1):
        if it % eval_every == 0:
            print(f"  step {it:>4} : val {full_val_loss(m):.3f}")
        xb, yb = get_batch("train")
        _, loss = m(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return full_val_loss(m)


attn_only = OneLayerLM(vocab_size, mlp=False).to(DEV)
print("training attention-only (1 layer, no MLP):")
attn_only_floor = train(attn_only)

attn_mlp = OneLayerLM(vocab_size, mlp=True).to(DEV)
print("\ntraining attention + MLP (1 layer):")
attn_mlp_floor = train(attn_mlp)

print(f"\nconcat/Bengio (04)          : 1.840")
print(f"attention only (1 layer)    : {attn_only_floor:.3f}")
print(f"attention + MLP (1 layer)   : {attn_mlp_floor:.3f}   <- the MLP's contribution")

# %% [markdown]
"""
## Now stack it — and watch depth break

One `attn → MLP` layer is good; `01` used **four**. The naive way to go deeper is to
just keep applying the pair: `x = mlp(attn(x))`, over and over. It doesn't work, and the
failure is worth *seeing* rather than being told, because the fix (residual + norm) only
makes sense once you've watched the thing it fixes.

Two forces break a deep plain stack, and we can measure both on a freshly-initialized
network (no training needed — it's about signal propagation through the layers):

- **Forward:** the activation scale drifts as it passes through many nonlinear layers —
  it can shrink toward zero or blow up — so by the top the signal is degenerate.
- **Backward:** the gradient that reaches the *early* layers, having been multiplied by
  each layer's Jacobian on the way down, **shrinks geometrically with depth** — the
  vanishing gradient. Early layers get almost no learning signal.

Below: a plain stack of `Linear→GELU→Linear` sublayers, no residual, no norm. Sweep the
depth and, for each, measure (a) the RMS of the final activation and (b) the norm of the
gradient arriving back at the input. Read the trend, not any single row.
"""


# %%
def signal_probe(depth, residual, norm, C=64, seed=0):
    """Push a random input through `depth` sublayers and one scalar loss backward.
    Returns (final activation RMS, gradient norm reaching the input)."""
    torch.manual_seed(seed)
    x = torch.randn(1, 16, C, requires_grad=True)
    sublayers = [
        nn.Sequential(nn.Linear(C, C), nn.GELU(), nn.Linear(C, C)) for _ in range(depth)
    ]
    norms = [nn.LayerNorm(C) for _ in range(depth)] if norm else None
    h = x
    for i, layer in enumerate(sublayers):
        inp = norms[i](h) if norm else h
        out = layer(inp)
        h = h + out if residual else out          # the one-line difference
    h.pow(2).mean().backward()                    # a generic scalar objective
    return h.detach().std().item(), x.grad.norm().item()


print(f"{'depth':>6} | {'plain: act_rms  grad_norm':>28} | "
      f"{'residual+prenorm: act_rms  grad_norm':>36}")
print("-" * 78)
for depth in [1, 2, 4, 8, 16, 32]:
    p_rms, p_g = signal_probe(depth, residual=False, norm=False)
    r_rms, r_g = signal_probe(depth, residual=True, norm=True)
    print(f"{depth:>6} | {p_rms:>12.3f}  {p_g:>12.2e} | {r_rms:>16.3f}  {r_g:>14.2e}")

print("\nplain: gradient to the input collapses toward 0 as depth grows (vanishing);")
print("residual+prenorm: gradient stays O(1) and activations stay well-scaled.")

# %% [markdown]
"""
## Fix 1 — the residual: give the gradient a straight road

Change one character of the update: instead of `x = sublayer(x)`, write

    x = x + sublayer(x)

Now each layer learns a **delta** to add to a running "residual stream," rather than
replacing it. The payoff is in the backward pass. Differentiate `x + sublayer(x)` w.r.t.
`x`:

    d/dx [x + sublayer(x)] = I + J        (J = the sublayer's Jacobian)

The **identity `I`** is the key. In a plain stack the input gradient is a *product* of
Jacobians, `J_L · … · J_1`, which shrinks or explodes geometrically. With the residual,
every factor is `(I + J)`: expand the product and one term is `I · I · … · I = I` — a
**direct, undamped path** from the loss to every layer. The gradient can always flow
straight down the identity highway, no matter how many layers sit on top. That's the
`residual=True` column above: `grad_norm` stopped collapsing.

(This is exactly `05`'s `04`-shape lesson echoed structurally: the residual stream is
the thing every sublayer reads from and writes back into, the same stream `01` carried
from embedding to final norm.)
"""

# %% [markdown]
"""
## Fix 2 — pre-norm: keep every sublayer's input well-scaled

The residual fixes the gradient, but introduces a forward-side wrinkle: because each
sublayer *adds* to the stream, the stream's magnitude **grows with depth** — variances
accumulate. Feed an ever-larger vector into the next attention/MLP and things get
ill-conditioned again.

The fix is **pre-norm**: put a `LayerNorm` on the *input* to each sublayer, not on the
residual stream itself:

    x = x + attn(ln1(x))
    x = x + ffwd(ln2(x))

`ln(x)` re-centers and re-scales *per token, across the C features*, so every sublayer
always sees a unit-scale input regardless of how tall the stack is — while the raw
residual stream `x` is left free to grow and carry information straight through
(un-normalized) to the top. Crucially it's `x + sublayer(ln(x))`, **not** `ln(x +
sublayer(x))`: the identity highway from Fix 1 must stay un-normalized, or we'd break
the clean gradient path we just built. (Post-norm — LayerNorm *after* the add — is the
original 2017 Transformer; pre-norm is what GPT-2 onward use because it trains far more
stably at depth.)

Measure the split: as depth grows, the **residual stream RMS climbs**, but the
**normalized input** each sublayer receives stays pinned near 1.
"""

# %%
torch.manual_seed(0)
C = 64
ln = nn.LayerNorm(C)
x = torch.randn(1, 16, C)
sub = lambda: nn.Sequential(nn.Linear(C, C), nn.GELU(), nn.Linear(C, C))

print(f"{'depth':>6} | {'residual-stream RMS':>20} | "
      f"{'RMS of ln(x) into sublayer':>28}")
print("-" * 62)
h = x
for depth in range(1, 33):
    into = ln(h)
    h = h + sub()(into)          # pre-norm residual step
    if depth in (1, 2, 4, 8, 16, 32):
        print(f"{depth:>6} | {h.std().item():>20.3f} | {into.std().item():>28.3f}")

print("\nstream grows with depth (info accumulates); "
      "the normed input stays ~1 (stable).")

# %% [markdown]
"""
## Assemble the Block

Both fixes, wrapped around the two sublayers, is the transformer **Block** — the unit
`01` stacked four of. Attention (mix) and FeedForward (per-token compute), each in its
own pre-norm residual:

    x = x + attn(ln1(x))     # gather context, add the delta
    x = x + ffwd(ln2(x))     # think per-token, add the delta

That's the entire core architecture. Everything after this notebook (`07`–`13`) either
tunes *how* it's trained or swaps one piece inside this exact shape.
"""


# %%
class Block(nn.Module):
    """One transformer layer: pre-norm residual attention, then pre-norm residual
    MLP."""

    def __init__(self, n_embd, n_head, block_size=BLOCK, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))    # NOT ln(x + attn(x)) — keep the highway clean
        x = x + self.ffwd(self.ln2(x))
        return x


# %% [markdown]
"""
## Stack four Blocks → recover `01`'s 1.594

Now the payoff. Token + position embedding → **four Blocks** → a final `LayerNorm` → the
vocab head. This is `01`'s architecture exactly, at `01`'s size (`n_embd=128`, 4 heads,
4 layers) — assembled from the pieces we built and understand. The `ln_f` before the
head is the same idea one more time: normalize the residual stream once at the top
before reading out logits.

Trained on the same loop, it should land right around `01`'s **1.594** — the number that
has sat at the bottom of the scoreboard since `05`, now reproduced from scratch. (It
won't match to the decimal — different seed, plain LR schedule instead of `01`'s — but
it's the same model, so it's the same neighborhood.)
"""


# %%
class GPT(nn.Module):
    """The core architecture: embed -> N Blocks -> final norm -> vocab head."""

    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=BLOCK):
        super().__init__()
        self.block_size = block_size
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

    generate = OneLayerLM.generate       # identical autoregressive loop


gpt = GPT(vocab_size).to(DEV)
print(f"GPT params : {sum(p.numel() for p in gpt.parameters()):,}")
print("\ntraining 4-Block GPT:")
gpt_floor = train(gpt, max_iters=5000, eval_every=1000)

# %% [markdown]
"""
## The full scoreboard, from bigram to reproduced GPT

Everything this track has measured, on the same data and the same deterministic metric.
The story in one column: each row changed *one* idea, and the loss moved.
"""

# %%
print(f"bigram         (03, CTX=1)         : 2.450   <- the floor")
print(f"average        (04, uniform mix)   : 2.950   <- worse: no order, no focus")
print(f"concat/Bengio  (04, rigid window)  : 1.840")
print(f"attention only (06, 1 layer)       : {attn_only_floor:.3f}   "
      "<- mix, linear readout")
print(f"attention+MLP  (06, 1 layer)       : {attn_mlp_floor:.3f}   "
      "<- + per-token compute")
print(f"4-Block GPT    (06, this notebook) : {gpt_floor:.3f}   "
      "<- + depth (residual+prenorm)")
print(f"transformer    (01, reference)     : 1.594")

# %% [markdown]
"""
## Read the babble

Generate from the trained GPT. Compared to `05`'s single-layer output, real **words**
and short phrases start to hold together — depth plus per-token MLPs is what lets the
model assemble structure, not just echo character statistics. It's still a tiny char
model on a few minutes of training, so don't expect sense; expect Shakespeare-shaped
text with words that mostly exist.
"""

# %%
start = torch.tensor([[stoi["\n"]]], device=DEV)
print(decode(gpt.generate(start, max_new_tokens=400)[0].tolist()))

# %% [markdown]
"""
## What you built, and where `07` goes next

The core transformer is complete and reproduced from scratch:

- the **FeedForward** is the per-token nonlinear compute attention lacks —
  position-wise, 4× wide, where most of the model's knowledge sits;
- the **residual** (`x + sublayer(x)`) gives the gradient an identity highway through
  depth — we watched the vanishing gradient it cures;
- **pre-norm** keeps every sublayer's input unit-scaled while the residual stream grows
  freely — measured, both halves;
- the **Block** wraps attention and MLP each in a pre-norm residual, and four stacked
  Blocks reproduce `01`'s 1.594.

There's nothing architectural left in the core model — `01`'s box is fully open. What's
left is *how well you drive it*. `07` opens the `train()` call we've been reusing on
faith since `03`: **AdamW** and decoupled weight decay, principled **weight init** (and
the `≈ ln(vocab)` init-loss sanity check), **LR warmup + cosine decay**, and **gradient
clipping** — each a knob you can turn off and watch the loss curve get worse. Same
model, a better-driven training run, and a lower number.
"""
