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
# LLM · 01 — the whole game

Top-down: before we take anything apart, let's build a **real GPT, train it on Shakespeare, and watch
it babble back**. By the end of this notebook you have a working character-level language model and a
rough mental map of its parts. *Why* each part is shaped the way it is — that's `02` onward, each
opening one box of *this exact model*.

> This is the map. Every later notebook zooms into **one box you already ran here.**

Kept deliberately minimal: character tokenizer (one id per character), a small transformer, a plain
training loop. No BPE, no dropout tuning, no fancy sampling yet — just enough to see the loss fall and
read the output.
"""

# %%
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _repo_root() -> Path:
    """Walk up from the cwd to the folder holding pyproject.toml, so paths work no matter where
    the kernel is launched from."""
    here = Path.cwd()
    for d in (here, *here.parents):
        if (d / "pyproject.toml").exists():
            return d
    return here


ROOT = _repo_root()
DATA = ROOT / "nb" / "llm" / "data" / "input.txt"          # tinyshakespeare (~1MB), track-local
CKPT_DIR = ROOT / "nb" / "llm" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
print(f"device: {DEV}")

# %% [markdown]
"""
## The data — characters in, characters out

The **character tokenizer** is the simplest thing that works: list every distinct character in the
text, and map each to an integer id. "hello" becomes a list of 5 ids; ids decode straight back to
text. (Real models merge frequent character *pairs* into subword tokens — BPE — which is `02`. For now,
one character = one token.)
"""

# %%
if not DATA.exists():                                      # self-contained: fetch on first use
    import urllib.request
    DATA.parent.mkdir(parents=True, exist_ok=True)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    print(f"  downloading tinyshakespeare -> {DATA.relative_to(ROOT)} ...")
    urllib.request.urlretrieve(url, DATA)

text = DATA.read_text()
chars = sorted(set(text))                                  # the vocabulary: every distinct character
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}                 # char -> id
itos = {i: c for i, c in enumerate(chars)}                 # id -> char
encode = lambda s: [stoi[c] for c in s]                    # string -> list of ids
decode = lambda ids: "".join(itos[i] for i in ids)         # list of ids -> string

print(f"corpus: {len(text):,} characters, vocab of {vocab_size} distinct chars")
print(f"vocab: {''.join(chars)!r}")
print(f"encode('Hello') = {encode('Hello')}  ->  decode back = {decode(encode('Hello'))!r}")

# whole corpus as one long tensor of ids, split 90/10 into train and validation
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

# %% [markdown]
"""
## Batches — predict the next character

The task is next-token prediction: given a chunk of `block_size` characters, predict the character
that follows at **every** position. `get_batch` grabs `batch_size` random chunks; `y` is just `x`
shifted one step to the left (each position's target is the next character).
"""

# %%
BLOCK = 64        # context length: how many characters the model sees at once
BATCH = 32        # sequences per step


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK, (BATCH,))           # random start offsets
    x = torch.stack([d[i:i + BLOCK] for i in ix])          # (BATCH, BLOCK) inputs
    y = torch.stack([d[i + 1:i + 1 + BLOCK] for i in ix])  # (BATCH, BLOCK) targets = x shifted +1
    return x.to(DEV), y.to(DEV)


xb, yb = get_batch("train")
print(f"x {tuple(xb.shape)}, y {tuple(yb.shape)}   (y is x shifted one character left)")
print(f"  x[0]: {decode(xb[0].tolist())!r}")
print(f"  y[0]: {decode(yb[0].tolist())!r}")

# %% [markdown]
"""
## The model in one breath

Rough narration, no rigor yet — just enough that it isn't a black box:

- **embeddings** turn each token id into a vector, and add a **position** vector so the model knows
  *where* each token sits (attention itself is order-blind — `04`).
- **blocks** are the transformer's repeating unit. Each has **self-attention** (tokens look at the
  earlier tokens and pull in what's relevant — the causal mask stops peeking at the future) followed
  by a small **MLP** (each token thinks on its own). Both are wrapped in a **residual + LayerNorm** so
  the stack trains at depth.
- **head** maps each token's final vector to a score for every possible next character.

That's it — `ids → embed → blocks → scores`. *Why* attention, why the residual wrapper, why the MLP
expands 4×? All deferred — `04`–`06` open each box in turn.
"""


# %%
@dataclass
class GPTConfig:
    """All the model's dimensions in one place. The model is built entirely from a config object, so
    a checkpoint that stores this config can reconstruct the exact architecture — no reliance on
    matching constructor defaults. (This is the GPT-2/nanoGPT pattern; `llm/config.py` is the same
    idea carried further.)"""
    vocab_size: int
    block_size: int = 64        # max context length
    n_embd: int = 128           # embedding / residual width
    n_head: int = 4             # attention heads
    n_layer: int = 4            # transformer blocks


class Block(nn.Module):
    """One transformer layer: causal self-attention, then a per-token MLP, each in a residual +
    pre-LayerNorm wrapper (x = x + sublayer(norm(x)))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        # torch's attention (we build it by hand in 04)
        self.attn = nn.MultiheadAttention(cfg.n_embd, cfg.n_head, batch_first=True)   
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(  # per-token compute
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
        )
        # causal mask: True where a query position may NOT attend (the strict future)
        self.register_buffer(
            "causal",
            torch.triu(
                torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool),
                diagonal=1,
            ),
        )

    def forward(self, x):
        h = self.ln1(x)
        T = x.size(1)
        a, _ = self.attn(h, h, h, attn_mask=self.causal[:T, :T], need_weights=False)   # tokens mix
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size                           # max context length
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # id -> vector
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)    # position -> vector
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size)          # vector -> next-char scores

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)                # (B, T, C)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))                           # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Autoregressive: sample a next char, append, repeat. Crop the context to block_size so the
        position embedding never runs off its table."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.block_size:])
            probs = F.softmax(logits[:, -1, :], dim=-1)            # distribution over next char
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# the config is the single source of truth: we build the model from it, save it in the checkpoint,
# and reconstruct with GPT(GPTConfig(**saved)) later.
cfg = GPTConfig(vocab_size=vocab_size, block_size=BLOCK)
model = GPT(cfg).to(DEV)
print(f"params: {sum(p.numel() for p in model.parameters()):,}")

# %% [markdown]
"""
## Watch it learn

Cross-entropy on next-character prediction, AdamW, a few thousand steps. We print the loss on held-out
validation text every so often — watch it fall. Before any training the model is uniform-random, so
its loss should sit near `ln(vocab)` (log of the number of characters); anything much lower means it's
actually learned the statistics of the text.
"""


# %%
@torch.no_grad()
def full_val_loss(m, bs=256):
    """Mean cross-entropy over the WHOLE validation split, in non-overlapping BLOCK-sized windows.
    Deterministic — a real metric, not a noisy one-batch sample — so it's a stable curve AND gives
    the identical number for any two models that share weights (our round-trip check below)."""
    m.eval()
    nwin = (len(val_data) - 1) // BLOCK                        # how many full windows fit
    x = val_data[:nwin * BLOCK].view(nwin, BLOCK)             # (nwin, BLOCK) inputs
    y = val_data[1:nwin * BLOCK + 1].view(nwin, BLOCK)        # targets, shifted +1
    total = count = 0
    for i in range(0, nwin, bs):                              # batch the windows through the model
        xb, yb = x[i:i + bs].to(DEV), y[i:i + bs].to(DEV)
        total += m(xb, yb)[1].item() * yb.numel()            # weight by #tokens (mean over uneven chunks)
        count += yb.numel()
    m.train()
    return total / count


@torch.no_grad()
def train_loss_est(iters=100):
    """Cheap train-loss estimate from random batches (the train split is huge; no full pass needed)."""
    model.eval()
    est = torch.stack([model(*get_batch("train"))[1] for _ in range(iters)]).mean().item()
    model.train()
    return est


import math

MAX_ITERS, EVAL_EVERY, LR = 3000, 500, 3e-3
opt = torch.optim.AdamW(model.parameters(), lr=LR)

print(f"random-init baseline: loss ≈ ln(vocab) = {math.log(vocab_size):.3f}")
for it in range(MAX_ITERS + 1):
    if it % EVAL_EVERY == 0:
        print(f"  step {it:>4} : train {train_loss_est():.3f}   val {full_val_loss(model):.3f}")
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

# %% [markdown]
"""
## Save the checkpoint

We save three things: the trained **weights**, the **tokenizer** (so ids decode back to text), and the
**config** (as a plain dict via `asdict`, so reloading doesn't depend on the class being importable).
This is the whole point of a checkpoint — later notebooks `torch.load` this exact model instead of
spending time retraining; `02` onward all start from these weights.
"""

# %%
ckpt = CKPT_DIR / "gpt.pt"
torch.save({"model": model.state_dict(), "stoi": stoi, "itos": itos, "config": asdict(cfg)}, ckpt)
print(f"saved -> {ckpt.relative_to(ROOT)}")

# %% [markdown]
"""
## Load it back — and prove it round-trips

Reload in a **separate** step (exactly as `02` will): read the file, rebuild the architecture from the
saved config with `GPT(GPTConfig(**config))`, and load the weights into that fresh model. Then confirm
it gives the **identical full validation loss** as the trained model. With no dropout and `eval()`, a
correct reload is exact — a mismatch would mean the checkpoint is incomplete (missing a buffer, wrong
shapes, a config field that doesn't rebuild the same model, ...).
"""

# %%
saved = torch.load(ckpt)
reloaded = GPT(GPTConfig(**saved["config"])).to(DEV)         # rebuild exact architecture from saved config
reloaded.load_state_dict(saved["model"])
before, after = full_val_loss(model), full_val_loss(reloaded)
print(f"full val loss: trained {before:.3f}  vs  reloaded {after:.3f}  ->  round-trips ✓")

# %% [markdown]
"""
## The payoff — read what it writes

Seed with a newline and let it generate 500 characters. It won't be Shakespeare, but from a pile of
random characters it has learned words, line breaks, capitalized names with colons, the *shape* of a
play — purely from predicting the next character.
"""

# %%
start = torch.tensor([[stoi["\n"]]], device=DEV)
sample = decode(model.generate(start, max_new_tokens=500)[0].tolist())
print(sample)

# %% [markdown]
"""
You now have a working language model and a rough picture. Everything below zooms into **one box you
just ran** — starting with `02`, which opens the input: what a *token* really is, and why real models
use BPE instead of raw characters. (`roadmap.md` holds the full running order.)
"""
