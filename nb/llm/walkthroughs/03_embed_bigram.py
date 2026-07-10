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
# LLM · 03 — embeddings + the bigram floor

`01` fed the model token ids and the very first thing it did was turn each id into a
**vector**. `02` showed where the ids come from. This notebook opens the next box: how
an id *becomes* a vector, and the simplest possible language model you can build on
top of that — the **bigram** — which is exactly the floor `01`'s transformer had to
beat.

Two ideas, one payoff:

1. **The embedding table.** `nn.Embedding` is nothing but a lookup table of learned
   vectors — one row per id. We'll show it's literally a one-hot vector times a weight
   matrix, and why a transformer *also* needs a **position** vector on top.
2. **The bigram model.** Predict the next token from the **current token alone** — no
   context, no attention. It's an embedding table whose rows *are* the next-token
   scores. Train it with the exact same loop `01` used and watch it converge to a floor
   well above `01`'s number. That gap is what context buys us — `04` starts closing it and
   `05`'s attention finishes the job.

We stay **character-level** here (same 65-token vocab as `01`) on purpose: the bigram's
loss is then in the *same units* as `01`'s transformer, so "the floor it had to beat" is
a real, directly comparable number — not an apples-to-oranges BPE count.
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
DATA = ROOT / "nb" / "llm" / "data" / "input.txt"  # tinyshakespeare, shared with 01/02
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
print(f"device: {DEV}")

# %% [markdown]
"""
## The data + batches — the skeleton every later notebook reuses

Same character tokenizer and same `get_batch` as `01`. This little harness — encode
the corpus to one long tensor, split 90/10, grab random `BLOCK`-length chunks with
targets shifted one step left — is the **reusable training skeleton**. Every later
notebook swaps in a bigger model but keeps this exact data path, so it's worth fixing
here once.
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

BLOCK = 64        # context length (the bigram ignores it; 04's attention will use it)
BATCH = 32


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK, (BATCH,))
    x = torch.stack([d[i:i + BLOCK] for i in ix])          # (BATCH, BLOCK)
    y = torch.stack([d[i + 1:i + 1 + BLOCK] for i in ix])  # targets = x shifted +1
    return x.to(DEV), y.to(DEV)


print(f"corpus {len(text):,} chars, vocab {vocab_size}")
xb, yb = get_batch("train")
print(f"x {tuple(xb.shape)}, y {tuple(yb.shape)}")

# %% [markdown]
"""
## An embedding is a lookup table (= one-hot × weight)

`nn.Embedding(num, dim)` holds a `(num, dim)` weight matrix and, given an id `i`,
returns **row `i`**. That's the whole operation — a differentiable lookup. The rows
start random and are learned by backprop like any other parameter.

The "lookup" framing and the "matrix multiply" framing are the *same thing*: looking up
row `i` equals multiplying a one-hot vector (1 at position `i`) by the weight matrix.
`nn.Embedding` just skips building the one-hot and indexes directly — same result, far
cheaper. Let's prove that equivalence numerically.
"""

# %%
demo_emb = nn.Embedding(vocab_size, 24)   # 65 ids -> 24-dim vectors
ids = torch.tensor([stoi["h"], stoi["i"]])

lookup = demo_emb(ids)                                     # the indexing path
onehot = F.one_hot(ids, vocab_size).float() @ demo_emb.weight  # the matmul path
print(f"emb('h') is a {tuple(demo_emb(torch.tensor(stoi['h'])).shape)}-vector")
print(f"lookup == one-hot @ weight ? {torch.allclose(lookup, onehot, atol=1e-6)}")

# %% [markdown]
"""
## Why a transformer also needs *position*

The token embedding gives every id a vector, but it says nothing about **where** in the
sequence a token sits — the same `"e"` gets the same vector whether it's first or
last. Attention (`05`) is order-blind by construction, so `01` added a second table,
`pos_emb`, indexed by *position* `0, 1, 2, …`, and summed the two: `token_emb[id] +
pos_emb[pos]`. Now each input vector encodes both *what* the token is and *where* it is.

The **bigram** below is the one model that genuinely doesn't need this: it looks only at
the single current token, so position is irrelevant to it. That's exactly why it's the
simplest baseline — and exactly why it can't learn anything that depends on word
order.
"""

# %%
# what 01 does, in one line, for illustration (the bigram will NOT use pos)
tok_emb_tbl = nn.Embedding(vocab_size, 24)
pos_emb_tbl = nn.Embedding(BLOCK, 24)
pos = torch.arange(BLOCK)
combined = tok_emb_tbl(xb.cpu()) + pos_emb_tbl(pos)       # (BATCH, BLOCK, 24)
print(f"token+position input to a transformer: {tuple(combined.shape)}")

# %% [markdown]
"""
## The bigram model — the simplest LM there is

Collapse the embedding dimension all the way down to `vocab_size` and read the rows
**as logits directly**: `nn.Embedding(vocab, vocab)`. Row `i` is the model's score for
every possible *next* token, given that the *current* token is `i`. No context, no
position, no attention — the prediction at each position depends on that position's
token and nothing else. It's a learned count table of "what usually follows what".

Note the `forward` / `generate` interface is **identical** to `01`'s GPT (returns
`(logits, loss)`; `generate` samples-appends-repeats). That's deliberate — every later
model is a drop-in for this one.
"""


# %%
class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # (vocab, vocab): row = next-token logits for the current token
        self.logits_table = nn.Embedding(vocab_size, vocab_size)
        # start every row at 0 -> every next-token distribution is exactly uniform,
        # so the init loss lands exactly on ln(vocab) (see the sanity check below).
        # Zero-init is safe here: each row gets a different gradient (its rows see
        # different targets), so there's no symmetry to break like in a deep net.
        nn.init.zeros_(self.logits_table.weight)

    def forward(self, idx, targets=None):
        logits = self.logits_table(idx)                    # (B, T, vocab)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)                           # bigram: no context to crop
            probs = F.softmax(logits[:, -1, :], dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


model = BigramLM(vocab_size).to(DEV)
print(f"params: {sum(p.numel() for p in model.parameters()):,}  "
      f"(= vocab² = {vocab_size}² = {vocab_size**2})")

# %% [markdown]
"""
## Sanity check: the init loss should be ≈ ln(vocab)

We zero-init the table, so before training every row is `0` → softmax gives a perfectly
*uniform* distribution over the 65 next-token options. A uniform guess over `V` classes
has cross-entropy exactly `ln(V)` — the same `≈ ln(vocab)` baseline `01` printed. If the
measured init loss were far from `ln(vocab)`, something would be miswired (bad init
scale, wrong reduction, …). It's the cheapest correctness test there is, and every later
notebook keeps it. (With the default random `nn.Embedding` init the rows aren't zero, so
the init loss sits a bit *above* `ln(vocab)` — still a useful check, just less exact.)
"""

# %%
import math

with torch.no_grad():
    _, init_loss = model(*get_batch("val"))
print(f"ln(vocab) = {math.log(vocab_size):.3f}")
print(f"measured init loss = {init_loss.item():.3f}   (zero-init -> lands exactly)")

# %% [markdown]
"""
## Train it — same loop as `01`

AdamW over the shared `get_batch` harness, with a deterministic full-validation pass for
a stable curve. Watch the loss fall from `≈ ln(vocab)` and then **flatten** — the
bigram saturates fast because there's not much to learn from single-token context.
"""


# %%
@torch.no_grad()
def full_val_loss(m, bs=256):
    """Mean cross-entropy over the WHOLE validation split in non-overlapping BLOCK
    windows — a real, deterministic metric (not a noisy one-batch sample)."""
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


MAX_ITERS, EVAL_EVERY, LR = 3000, 500, 1e-2
opt = torch.optim.AdamW(model.parameters(), lr=LR)

for it in range(MAX_ITERS + 1):
    if it % EVAL_EVERY == 0:
        print(f"  step {it:>4} : val {full_val_loss(model):.3f}")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

bigram_floor = full_val_loss(model)

# %% [markdown]
"""
## The floor — and the gap `01` closed

Put the numbers side by side:

- **uniform guess**: `ln(vocab) ≈ 4.17`
- **bigram** (this notebook): the current token only
- **transformer** (`01`): the *whole* preceding context via attention → val ≈ **1.6**

The bigram already crushes the uniform baseline just by learning letter-pair statistics
(`q`→`u`, space after `.`, capitals after `\n`). But it plateaus far above the
transformer, because "the single previous character" throws away almost everything.
Every 0.1 nats below this floor is the model learning to *use context* — which is what
`04` (combining more than one previous token) and then `05`'s attention exist to do.
"""

# %%
print(f"uniform  ln(vocab) : {math.log(vocab_size):.3f}")
print(f"bigram (this nb)   : {bigram_floor:.3f}   <- the floor")
print(f"transformer (01)   : 1.594   <- what context buys, from the SAME data")
print(f"\ncontext is worth ~{bigram_floor - 1.594:.2f} nats/token here")

# %% [markdown]
"""
## The payoff — read the babble

Generate from the trained bigram. It gets *local* structure right — plausible letter
pairs, word-ish blobs, capitals after newlines — but no words hold together and
there's no grammar, because each character is chosen from the previous one alone.
Compare this to `01`'s output: that jump in coherence is the whole rest of the track.
"""

# %%
start = torch.tensor([[stoi["\n"]]], device=DEV)
print(decode(model.generate(start, max_new_tokens=300)[0].tolist()))

# %% [markdown]
"""
You now have the two primitives every later notebook stands on: **embeddings** (id →
vector) and the **train/generate skeleton**, plus a concrete **floor** to beat. Next,
`04` takes the first step past the floor — **combining more than one previous token**
(concatenate vs average) — and shows why a *learned, weighted* combination is needed,
which is exactly the **self-attention** that `05` builds.
"""
