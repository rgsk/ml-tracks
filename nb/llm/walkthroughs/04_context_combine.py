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
# LLM · 04 — combining context: concat & average (the road to attention)

`03` left us at the **bigram floor**: predict the next token from the *single* current
token. The only way down from that floor is to look at **more than one** previous
token. So the question this notebook answers is narrow and mechanical:

> we have a window of `T` previous token-vectors — how do we collapse them into one
> vector to predict from?

There are two classic answers, and building both is the cleanest possible setup for
attention (`05`), because attention is precisely the fix for where they fail:

1. **Concatenate** — stack the window's vectors side by side, one weight-slot per
   position, then a small MLP. This is the **Bengio 2003 neural LM**. It *keeps order*
   but the window is rigid and the parameters grow with it.
2. **Average** — mean-pool the window into one vector (a **bag of chars**). It scales to
   any length for free but *destroys order* and can't focus on the tokens that matter.

We train both on the same character data as `03`, put their val loss next to the bigram
floor and `01`'s transformer, and — the real payoff — *measure* each one's weakness
directly. The closer shows the one-line trick that turns "average" into "attention".
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
DATA = ROOT / "nb" / "llm" / "data" / "input.txt"  # tinyshakespeare, shared with 01–03
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
print(f"device: {DEV}")

# %% [markdown]
"""
## Same data skeleton as `03`

Character tokenizer, one long id tensor, 90/10 split, random `BLOCK`-length chunks with
targets shifted one step left. Nothing new — every model below trains through this exact
`get_batch`, so their losses are directly comparable to the bigram floor.
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

BLOCK = 64        # sequence length per batch row
BATCH = 32
CTX = 8           # how many previous tokens each prediction may look at (the window)


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK, (BATCH,))
    x = torch.stack([d[i:i + BLOCK] for i in ix])          # (BATCH, BLOCK)
    y = torch.stack([d[i + 1:i + 1 + BLOCK] for i in ix])  # targets = x shifted +1
    return x.to(DEV), y.to(DEV)


print(f"corpus {len(text):,} chars, vocab {vocab_size}, context window CTX={CTX}")

# %% [markdown]
"""
## The one shared move: causal windows

Both models need, for every output position `t`, the `CTX` tokens *ending at* `t`
(token `t` itself plus the `CTX-1` before it — strictly the past, so it stays causal).
We build all those windows at once with a left-pad + `unfold`, turning `(B, T)` ids into
`(B, T, CTX)` windows. That one tensor op is the only bookkeeping here; after it, "concat
vs average" is a single line of difference.

Left-padding with id `0` gives the first few positions a stand-in "start" context. With
`BLOCK=64` and `CTX=8` only 7/64 positions per row are affected, and it hits both models
equally — so it never changes which one wins.
"""


# %%
def causal_windows(idx, ctx):
    """(B, T) ids -> (B, T, ctx): for each t, the ctx tokens ending at position t."""
    padded = F.pad(idx, (ctx - 1, 0), value=0)   # left-pad the time axis
    return padded.unfold(1, ctx, 1)              # slide a width-ctx window, step 1


_demo = torch.tensor([encode("hello wo")])       # (1, 8)
print(f"ids     : {_demo.tolist()[0]}")
print(f"windows : {tuple(causal_windows(_demo, CTX).shape)}  (one CTX-row per position)")
print("last row (context for the final char) =",
      causal_windows(_demo, CTX)[0, -1].tolist())

# %% [markdown]
"""
## Model A — concatenate (the Bengio 2003 neural LM)

Embed each token in the window to a `C`-dim vector, **flatten** the `CTX` of them into one
`CTX*C` vector, and push it through `Linear -> GELU -> Linear -> vocab`. Because each
window slot lands in its own stretch of the input, the first Linear has *separate weights
per position* — so the model can tell `"ab"` from `"ba"` and can learn "the char right
before me matters most". Order is baked in by the layout; no position embedding needed.

Two things to notice:

- With `CTX=1` and no hidden layer this collapses *exactly* to `03`'s bigram — the bigram
  is the one-token special case of this model.
- The input width is `CTX*C`, so the first layer's parameter count grows linearly with the
  window. Doubling context doubles those weights — the scaling wall we'll call out below.

`forward`/`generate` keep the `(logits, loss)` interface from `03`, so every metric and
the generator carry over unchanged.
"""


# %%
class ConcatMLP(nn.Module):
    """Bengio-style fixed-window LM: concat the window's embeddings, then an MLP."""

    def __init__(self, vocab_size, ctx=CTX, n_embd=24, n_hidden=128):
        super().__init__()
        self.ctx = ctx
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.net = nn.Sequential(
            nn.Linear(ctx * n_embd, n_hidden),
            nn.GELU(),                              # Bengio used tanh; GELU is the modern default
            nn.Linear(n_hidden, vocab_size),
        )

    def forward(self, idx, targets=None):
        win = causal_windows(idx, self.ctx)         # (B, T, ctx)
        e = self.tok_emb(win)                       # (B, T, ctx, C)
        e = e.flatten(2)                            # (B, T, ctx*C)  <- concat the window
        logits = self.net(e)                        # (B, T, vocab)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.ctx:]           # only the last ctx tokens matter
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# %% [markdown]
"""
## Model B — average (bag of chars)

Same embeddings, same window. The *only* change is the collapse step: **mean** over the
`CTX` axis instead of concatenating, then a linear head to the vocab.

That one change is a big deal. A mean is **permutation-invariant**: shuffle the window and
the output is identical. So this model literally cannot use order — `mean("ab") ==
mean("ba")`. It also can't *focus*: every token in the window contributes exactly `1/CTX`,
so the immediately-preceding char (usually the strongest cue for the next one) is weighted
no more than a token far back. We'll show both failures with numbers right after training.

The upside — which attention keeps — is that the mean works for *any* window length with a
parameter count that doesn't depend on `CTX` at all.
"""


# %%
class AverageBOW(nn.Module):
    """Bag-of-chars LM: mean-pool the window's embeddings, then a linear head."""

    def __init__(self, vocab_size, ctx=CTX, n_embd=24):
        super().__init__()
        self.ctx = ctx
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        win = causal_windows(idx, self.ctx)         # (B, T, ctx)
        e = self.tok_emb(win)                       # (B, T, ctx, C)
        e = e.mean(dim=2)                           # (B, T, C)  <- order is gone here
        logits = self.head(e)                       # (B, T, vocab)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.ctx:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# %% [markdown]
"""
## Train both on the shared loop

Same AdamW loop and the same deterministic `full_val_loss` as `03`, so all three numbers
(bigram, average, concat) are measured identically. Watch concat pull clearly below
average, and both below the bigram floor.
"""


# %%
import math


@torch.no_grad()
def full_val_loss(m, bs=256):
    """Mean cross-entropy over the WHOLE validation split in non-overlapping BLOCK
    windows — the same deterministic metric `03` used."""
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


def train(m, max_iters=3000, eval_every=1000, lr=1e-2):
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


avg_model = AverageBOW(vocab_size).to(DEV)
concat_model = ConcatMLP(vocab_size).to(DEV)

print(f"average params : {sum(p.numel() for p in avg_model.parameters()):,}")
print(f"concat  params : {sum(p.numel() for p in concat_model.parameters()):,}")
print("\ntraining average (bag of chars):")
avg_floor = train(avg_model)
print("\ntraining concat (Bengio MLP):")
concat_floor = train(concat_model)

# %% [markdown]
"""
## The scoreboard

The bigram number is `03`'s floor (≈ **2.45** on this data); `01`'s transformer sits at
**1.594**. Here's the twist: **concat crushes the floor (≈1.84), but average lands *above*
it (≈2.95) — worse than the bigram that sees only one token.** Seeing 8× more context made
it *worse*. That inversion is the whole lesson: *how* you combine context matters more than
*how much* you have. Concat keeps order and focus; average — a uniform mean — throws both
away and dilutes the one token that carried the signal. Order and focus are most of what
language is.
"""

# %%
bigram_floor = 2.45   # from 03 (same data, same metric)
print(f"uniform  ln(vocab)         : {math.log(vocab_size):.3f}")
print(f"bigram        (03, CTX=1)  : {bigram_floor:.3f}   <- the floor")
print(f"average       (CTX={CTX}, no order): {avg_floor:.3f}")
print(f"concat/Bengio (CTX={CTX}, order)   : {concat_floor:.3f}")
print(f"transformer   (01)         : 1.594   <- context used *selectively*")
print(f"\norder alone is worth ~{avg_floor - concat_floor:.2f} nats/token "
      f"(concat vs average, same window)")

# %% [markdown]
"""
## Measuring the two weaknesses directly

Numbers on a scoreboard are easy to hand-wave. Let's *prove* the claims.

**Weakness 1 — average is order-blind.** Take one context and a reordering of it. The
average model must give the identical next-token distribution (a mean can't see order);
the concat model must give a different one. `torch.allclose` settles it.
"""


# %%
@torch.no_grad()
def next_probs(m, context_ids):
    x = torch.tensor([context_ids], device=DEV)
    logits, _ = m(x)
    return F.softmax(logits[0, -1], dim=-1)         # distribution over the next token


base = encode("to be or")[:CTX]            # exactly one CTX-window
perm = base[::-1]                          # same chars, reversed order
print(f"base context : {decode(base)!r}")
print(f"permuted     : {decode(perm)!r}")

avg_same = torch.allclose(next_probs(avg_model, base),
                          next_probs(avg_model, perm), atol=1e-6)
concat_same = torch.allclose(next_probs(concat_model, base),
                             next_probs(concat_model, perm), atol=1e-6)
print(f"\naverage: reorder changes prediction? {not avg_same}   "
      f"(False -> it is blind to order)")
print(f"concat : reorder changes prediction? {not concat_same}   "
      f"(True  -> it uses order)")

# %% [markdown]
"""
**Weakness 2 — average can't focus, and it costs it.** In a bag of chars every window token
counts `1/CTX`. But for predicting the next character the *immediately previous* token
carries far more signal than one seven steps back. A fixed uniform weight can't express
that — and the damage is measurable: the average model, with 8 tokens of context, scores
*worse* than the bigram with 1. Pooling the strong last-token cue in with 7 weaker ones
washes it out faster than the extra tokens help. More context, used indiscriminately, is a
net loss. (That is the exact hole attention's *learned, non-uniform* weights climb out of.)
"""

# %%
print(f"average floor {avg_floor:.3f}  vs  bigram floor {bigram_floor:.3f}")
print(f"-> uniform pooling is ~{avg_floor - bigram_floor:.2f} nats WORSE than the bigram,")
print("   even with 8x the context: the informative last token gets diluted away.")

# %% [markdown]
"""
## Read the babble — concat is visibly more word-like

Generate from the concat model. Against `03`'s bigram it should hold short letter-runs
together better — because it is choosing each char from the previous `CTX`, not just one —
though whole words still won't cohere and there's no long-range grammar. That remaining
gap to `01` is what selective, unbounded context (attention) closes.
"""

# %%
start = torch.tensor([[stoi["\n"]]], device=DEV)
print("--- concat / Bengio ---")
print(decode(concat_model.generate(start, max_new_tokens=300)[0].tolist()))

# %% [markdown]
"""
## The bridge to `05`: average is attention with the weights frozen

Here's the move that makes attention feel inevitable. Averaging over the past is the same
as multiplying by a **lower-triangular matrix of uniform weights** — row `t` puts `1/(t+1)`
on every position `0..t` and `0` on the future. Below, a hand-built weight matrix
reproduces a causal running average *exactly* via one matmul.
"""

# %%
torch.manual_seed(0)
B, T, C = 1, 5, 2
x = torch.randn(B, T, C)

# uniform causal weights: row t averages positions 0..t
w = torch.tril(torch.ones(T, T))
w = w / w.sum(dim=1, keepdim=True)          # each row sums to 1
avg_via_matmul = w @ x                       # (T, T) @ (T, C) -> running mean

# the same thing, computed the slow explicit way
avg_via_loop = torch.stack([x[0, :t + 1].mean(dim=0) for t in range(T)])[None]

print("uniform causal weight matrix (row t = average over 0..t):")
print(w.round(decimals=2))
print(f"\nmatmul == explicit running mean ? "
      f"{torch.allclose(avg_via_matmul, avg_via_loop, atol=1e-6)}")

# %% [markdown]
"""
Stare at that weight matrix. The **only** thing wrong with it for language is that the
weights are *fixed and uniform* — every past token forced to count the same. **Attention
(`05`) keeps this exact structure — a causal, row-normalized weighted sum of past values —
but *learns* the weights from the tokens themselves**, so each position can put its mass on
the few earlier tokens that actually matter (and add a position signal back in). Concat had
focus but a rigid, parameter-hungry window; average scaled but couldn't focus. Attention is
the weighted average that gets both — and that's the box `05` opens.
"""
