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
# LLM · 12 — RoPE: position as rotation

TODO intro — written last, once the numbers are in.
"""

# %%
from pathlib import Path

import copy
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
DATA = ROOT / "nb" / "llm" / "data" / "input.txt"  # tinyshakespeare, shared with 01–11
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

torch.set_float32_matmul_precision("high")   # TF32 matmuls; 11's speedup, same deal
print(f"device: {DEV}")

# %% [markdown]
"""
## Same data skeleton as `03`–`11`
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
EMBD = 128
N_HEAD = 4
HS = EMBD // N_HEAD          # head_size = 32: the vector RoPE actually rotates


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK, (BATCH,))
    x = torch.stack([d[i:i + BLOCK] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + BLOCK] for i in ix])
    return x.to(DEV), y.to(DEV)


print(f"corpus {len(text):,} chars, vocab {vocab_size}, block {BLOCK}, head_size {HS}")

# %% [markdown]
"""
## The thing we're replacing

Every model in this track starts its forward pass the same way:

```-
x = tok_emb(idx) + pos_emb(arange(T))
```

`pos_emb` is an `nn.Embedding(block_size, C)`: one **learned vector per absolute slot**.
Slot 0 gets a vector, slot 1 gets a different one, and the model adds "you are at index
`t`" to each token before any attention happens. Two costs, and it's worth being precise
about which one is the interesting one.

**It has a hard length cap.** The table has exactly `block_size` rows. There is no row
for position `block_size`, so the model cannot *run* on a longer sequence — not "runs
badly", *cannot run*. This is the cheap complaint, and it's the one usually quoted.

**It's absolute, and attention wants relative.** This is the real one. What matters for
"this verb agrees with that subject" is that they are 3 apart — not that they sit at
slots 40 and 43. An absolute table forces the model to learn every relative pattern
*separately at every absolute location*, because slot 40 and slot 900 are unrelated rows.

Let's price both.
"""

# %%
pos_table = nn.Embedding(BLOCK, EMBD)
n_pos = pos_table.weight.numel()
print(f"pos_emb table: {tuple(pos_table.weight.shape)} = {n_pos:,} learned params")
try:
    pos_table(torch.tensor([BLOCK]))
except IndexError as e:
    print(f"pos_emb(t={BLOCK}) -> IndexError: {str(e).split('.')[0]}.")
print(f"\nvalid positions: 0..{BLOCK - 1}. RoPE replaces this table with 0 params")
print("and no cap. Whether 'no cap' means 'works longer' is the second half of this")
print("notebook, and the answer is not the one the sales pitch implies.")

# %% [markdown]
"""
## The idea: rotate instead of add

RoPE (Su et al. 2021) doesn't touch the input at all. It reaches **inside attention**,
and just before the scores are computed it **rotates** the query and key vectors by an
angle proportional to their position.

Take one head vector of size `d` (= `head_size`, even). Pair its coordinates into `d/2`
two-dimensional planes — coordinate `k` with coordinate `k + d/2`, the half-split pairing
Llama and HF ship — and rotate plane `k` of the token at position `m` by the angle
`m · theta_k`:

```-
[ out[k]     ]   [ cos(m·theta_k)  -sin(m·theta_k) ] [ x[k]     ]
[ out[k+d/2] ] = [ sin(m·theta_k)   cos(m·theta_k) ] [ x[k+d/2] ]
```

That's the whole primitive: `d/2` independent 2D rotations, with position setting the
**angle**. No parameters — `theta_k` is a fixed schedule we'll pick in a moment.

### Why rotating gives you relative position for free

This is the one derivation to keep, and it's three lines. A rotation matrix `R(a)` is
**orthogonal**, and rotations in the same plane compose by *adding* angles:

```-
R(a)^T = R(-a)                and         R(a) R(b) = R(a+b)
```

So take the query at position `m` and the key at position `n`, and do what attention does
— dot them:

```-
<R(m·theta) q, R(n·theta) k> = (R(m·theta) q)^T (R(n·theta) k)
                             = q^T R(m·theta)^T R(n·theta) k
                             = q^T R(-m·theta) R(n·theta) k
                             = q^T R((n - m)·theta) k
```

**The absolute positions cancel.** `m` and `n` enter the score only through `n - m`. Two
tokens 3 apart produce the same score whether they sit at (0, 3) or (900, 903) — not
because the model learned to be shift-invariant, but because there is no expression in
which their absolute positions survive. Relative position, by construction, at zero
parameters.

Contrast that with the additive table, where the score is
`(q_tok + q_pos_m)^T (k_tok + k_pos_n)` — cross terms in `m` and `n` separately, and
nothing cancels.
"""

# %%
def rope_frequencies(head_size: int, base: float = 10000.0) -> torch.Tensor:
    """Per-plane rotation speeds theta_k = base^(-2k/d), k = 0..d/2-1 -> (d/2,).

    arange(0, d, 2) yields 2k directly, so base^(-2k/d) is base ** (-idx / d)."""
    idx = torch.arange(0, head_size, 2).float()          # 0, 2, ..., d-2  -> d/2 planes
    return base ** (-idx / head_size)


def rope_explicit(x: torch.Tensor, pos: torch.Tensor,
                  base: float = 10000.0) -> torch.Tensor:
    """The literal per-plane 2x2 rotation, as a loop. x: (T, d), pos: (T,).

    Unusably slow — this exists only so the fast version has something to match."""
    T, d = x.shape
    theta = rope_frequencies(d, base)
    out = torch.empty_like(x)
    for t in range(T):
        for k in range(d // 2):
            ang = pos[t].item() * theta[k].item()
            c, s = math.cos(ang), math.sin(ang)
            a, b = x[t, k].item(), x[t, k + d // 2].item()
            out[t, k] = a * c - b * s
            out[t, k + d // 2] = a * s + b * c
    return out


# %% [markdown]
"""
### Check the cancellation on real numbers

The algebra says the score depends only on `n - m`. Fix one `q` and one `k`, rotate them
at various position pairs, and look.
"""

# %%
torch.manual_seed(0)
q_demo, k_demo = torch.randn(HS), torch.randn(HS)


def demo_score(m, n):
    qm = rope_explicit(q_demo.view(1, HS), torch.tensor([float(m)])).view(HS)
    kn = rope_explicit(k_demo.view(1, HS), torch.tensor([float(n)])).view(HS)
    return (qm @ kn).item()


print(f"{'m':>5} {'n':>5} {'n-m':>5} | {'score q_m . k_n':>16}")
print("-" * 36)
for m, n in [(0, 3), (1, 4), (7, 10), (100, 103), (0, 1), (5, 6), (0, 5)]:
    print(f"{m:>5} {n:>5} {n - m:>5} | {demo_score(m, n):>+16.6f}")

d3 = [demo_score(m, n) for m, n in [(0, 3), (1, 4), (7, 10), (100, 103)]]
print(f"\nfour pairs at distance 3, spread = {max(d3) - min(d3):.2e}  (float noise)")
print(f"distance 1 -> {demo_score(0, 1):+.4f}, distance 5 -> {demo_score(0, 5):+.4f} "
      f"-> different distance, different score")
print(f"\nno RoPE at all: q.k = {(q_demo @ k_demo).item():+.4f} for every (m, n) — "
      f"position never enters")

# %% [markdown]
"""
### And it never changes the vector's length

Worth pausing on, because it's the structural difference from an additive embedding. A
rotation moves a vector's **direction**, never its **magnitude**:
"""

# %%
x_one = torch.randn(HS)
print(f"||x|| = {x_one.norm():.4f}")
print(f"\n{'position m':>11} | {'||rope(x, m)||':>15} | {'plane 0 = (out[0], out[16])'}")
print("-" * 62)
for m in (0, 1, 2, 5, 50, 1000):
    r = rope_explicit(x_one.view(1, HS), torch.tensor([float(m)])).view(HS)
    print(f"{m:>11} | {r.norm():>15.4f} | ({r[0]:+.3f}, {r[HS // 2]:+.3f})")
print("\nIdentical at every position, including 1000 — far outside anything we train on.")
print("An ADDITIVE position embedding has no such guarantee: it moves the token vector")
print("by whatever the table learned, and a position can in principle shout down the")
print("token itself. RoPE cannot: position sets the angle, the token keeps its length.")

# %% [markdown]
"""
## The fast version: `x * cos + rotate_half(x) * sin`

The double loop above is a Python crime. The vectorized identity everyone ships
precomputes `cos` and `sin` of every `(position, plane)` angle once, then:

```-
rope(x) = x * cos  +  rotate_half(x) * sin
rotate_half([x1, x2]) = [-x2, x1]          # x1 = first half, x2 = second half
```

with `cos`/`sin` shaped `(T, d)` by **duplicating** the `(T, d/2)` plane-angles across
both halves. Check coordinate `k` (for `k < d/2`) against the 2x2 above:

```-
(x * cos)[k]            =  x[k]     · cos(m·theta_k)
(rotate_half(x)*sin)[k] = -x[k+d/2] · sin(m·theta_k)
                    sum =  x[k]cos - x[k+d/2]sin        <- exactly out[k]
```

and coordinate `k+d/2` gives `x[k]sin + x[k+d/2]cos`. Same rotation, no loop, and the
`(T, d)` tables broadcast across batch and heads for free.
"""


# %%
def build_rope_cache(seq_len, head_size, base=10000.0, offset=0, freq_div=None,
                     device=None, dtype=torch.float32):
    """(cos, sin), each (seq_len, head_size), for positions offset..offset+seq_len-1.

    `offset` rotates tokens by their TRUE absolute position — `13` needs it for the
    KV-cache, and the shift-invariance test below needs it now.
    `freq_div` divides theta per plane: the one knob every context-extension method
    turns (all of PI / NTK / YaRN are just a choice of this vector). None = vanilla.
    """
    theta = rope_frequencies(head_size, base).to(device=device, dtype=dtype)   # (d/2,)
    if freq_div is not None:
        theta = theta / freq_div.to(device=device, dtype=dtype)
    pos = torch.arange(offset, offset + seq_len, device=device, dtype=dtype)   # (T,)
    freqs = torch.outer(pos, theta)                      # (T, d/2) = m · theta_k
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # (T, d)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)  # (T, d)
    return cos, sin


def rotate_half(x):
    """[x1, x2] -> [-x2, x1]: supplies the off-diagonal (-sin/+sin) terms for every
    plane at once."""
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    """x: (..., T, d); cos, sin: (T, d), broadcast over the leading batch/head dims."""
    return x * cos + rotate_half(x) * sin


# %%
T_chk = 6
x_chk = torch.randn(T_chk, HS)
cos_chk, sin_chk = build_rope_cache(T_chk, HS)
fast = apply_rope(x_chk, cos_chk, sin_chk)
slow = rope_explicit(x_chk, torch.arange(T_chk).float())
print(f"max |rotate_half - explicit loop| = {(fast - slow).abs().max():.2e}  "
      f"({'match' if torch.allclose(fast, slow, atol=1e-5) else 'MISMATCH'})")

xb_chk = torch.randn(2, N_HEAD, T_chk, HS)               # (B, nh, T, hs) — attention's
print(f"broadcasts (T,{HS}) tables over {tuple(xb_chk.shape)} -> "
      f"{tuple(apply_rope(xb_chk, cos_chk, sin_chk).shape)}")

# %% [markdown]
"""
## The frequency ladder: `theta_k = base^(-2k/d)`

One choice left: how fast does each plane spin? RoPE reuses the sinusoidal schedule from
*Attention Is All You Need*, `theta_k = base^(-2k/d)` with `base = 10000`. Every plane
gets its own speed, spread **geometrically** from 1 down to `1/base`.

The useful way to read a frequency is as a **wavelength** — how many positions a plane
takes to come back to where it started:

```-
wavelength_k = 2·pi / theta_k        positions per full turn
```

Plane 0 turns once every ~6.3 tokens: it resolves *local* distance finely, but it can't
tell 2 apart from 8 apart (it's been all the way round). The slowest plane barely moves
across the whole context: it can't resolve neighbours, but it does encode *roughly where
you are* over long spans. Stack them and you get distance resolved at **every scale at
once** — a positional ruler with both fine tick marks and long ones.

Note the shape of it: it's a spectrum, and the two ends have opposite failure modes. That
tension is dormant until the extrapolation section, where it becomes the whole story.
"""

# %%
theta_full = rope_frequencies(HS)
print(f"head_size d = {HS} -> {HS // 2} planes, base = 10000\n")
print(f"{'plane k':>8} | {'theta_k':>12} | {'wavelength':>12} | {'turns in a 64-block':>20}")
print("-" * 62)
for k in [0, 1, 2, 4, 8, 12, HS // 2 - 1]:
    th = theta_full[k].item()
    wl = 2 * math.pi / th
    print(f"{k:>8} | {th:>12.6f} | {wl:>12.1f} | {BLOCK / wl:>20.3f}")
n_safe = int((theta_full * BLOCK / (2 * math.pi) >= 1).sum())
print(f"\n{n_safe} of {HS // 2} planes complete at least one full turn inside our "
      f"{BLOCK}-token block.")
print(f"The other {HS // 2 - n_safe} sweep only a fraction of an arc. Remember that "
      f"number.")

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
ks = torch.arange(HS // 2)
axes[0].semilogy(ks, 2 * math.pi / theta_full, "o-", color="#2a78d6", lw=1.8)
axes[0].axhline(BLOCK, ls="--", c="#e34948", lw=1.2)
axes[0].text(0.3, BLOCK * 1.3, f"our training block = {BLOCK}", fontsize=8, c="#e34948")
axes[0].set_xlabel("plane k")
axes[0].set_ylabel("wavelength (positions/turn)")
axes[0].set_title("A geometric ladder of wavelengths", fontsize=10)
axes[0].grid(alpha=0.25)

pos_ax = torch.arange(0, 64).float()
for k, c in [(0, "#104281"), (4, "#2a78d6"), (8, "#1baf7a"), (14, "#eda100")]:
    axes[1].plot(pos_ax, torch.cos(pos_ax * theta_full[k]), color=c, lw=1.6,
                 label=f"plane {k} (wl={2 * math.pi / theta_full[k]:.0f})")
axes[1].set_xlabel("position m")
axes[1].set_ylabel("cos(m · theta_k)")
axes[1].set_title("Fast planes = local detail. Slow planes = coarse location.",
                  fontsize=10)
axes[1].legend(fontsize=7.5, loc="lower right")
axes[1].grid(alpha=0.25)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Wiring it into the model

`06`'s GPT — the Phase B baseline every notebook since has deltaed against — with one
switch. The attention is the same projections and the same masked softmax; RoPE adds
exactly three lines: build `cos`/`sin`, rotate `q`, rotate `k`.

Two details that are easy to get wrong:

- **`v` is not rotated.** Position should decide *who attends to whom* — that's the `q·k`
  score. It shouldn't corrupt the *payload* being moved. Rotating `v` would mean the
  information a token contributes changes depending on where it stands.
- **the mask is built on the fly.** `01`–`11` registered a `(block_size, block_size)`
  `tril` buffer, which quietly re-imposes the length cap we just removed. `torch.ones(T,
  T).tril()` costs nothing at our size and works at any `T`.

I'm also threading a `pos_offset` through both models. RoPE needs it for the KV-cache in
`13` (rotate each new token by its *true* absolute position, leave the cached keys
alone), and it's the knob that makes the next measurement possible.
"""


# %%
class CausalSelfAttention(nn.Module):
    """05's attention, with an optional RoPE rotation on q and k."""

    def __init__(self, n_embd, n_head, rope=False, rope_base=10000.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.n_embd = n_head, n_embd
        self.head_size = n_embd // n_head
        assert not rope or self.head_size % 2 == 0, "RoPE needs an even head_size"
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.proj.RESIDUAL_PROJ = True
        self.rope = rope
        self.rope_base = rope_base
        self.freq_div = None      # set by the context-extension section, at eval time
        self.rope_temp = 1.0      # YaRN's attention temperature; 1.0 = off
        # NOTE: no tril buffer -> no block_size baked into this module.

    def forward(self, x, pos_offset=0):
        B, T, C = x.shape
        nh, hs = self.n_head, self.head_size
        q, k, v = self.qkv(x).split(self.n_embd, dim=-1)
        q = q.view(B, T, nh, hs).transpose(1, 2)         # (B, nh, T, hs)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)

        if self.rope:
            cos, sin = build_rope_cache(T, hs, self.rope_base, offset=pos_offset,
                                        freq_div=self.freq_div, device=x.device,
                                        dtype=x.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)   # v untouched

        scores = q @ k.transpose(-2, -1) * (hs ** -0.5 / self.rope_temp)
        mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
        out = F.softmax(scores, dim=-1) @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    """06's FFN, unchanged: expand to 4C, GELU, project back."""

    def __init__(self, n_embd):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.proj.RESIDUAL_PROJ = True

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    """06's Block: pre-norm + residual around attention, then the feed-forward."""

    def __init__(self, n_embd, n_head, rope=False, rope_base=10000.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, rope, rope_base)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x, pos_offset=0):
        x = x + self.attn(self.ln1(x), pos_offset)
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """The track's GPT. `rope=True` deletes pos_emb and rotates q/k instead."""

    def __init__(self, vocab_size, rope=False, rope_base=10000.0, n_embd=EMBD,
                 n_head=N_HEAD, n_layer=4, block_size=BLOCK):
        super().__init__()
        self.block_size, self.n_layer, self.rope = block_size, n_layer, rope
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = None if rope else nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, rope, rope_base) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, pos_offset=0):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(pos_offset, pos_offset + T, device=idx.device)
            x = x + self.pos_emb(pos)                    # <- the thing RoPE deletes
        for block in self.blocks:
            x = block(x, pos_offset)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.reshape(B * T))
        return logits, loss


# %% [markdown]
"""
`07`'s training recipe, unchanged and uncommented — it's the same one every notebook
since has used.
"""


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


def train(m, max_iters=3000, max_lr=3e-3, min_lr=3e-4, warmup=100):
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
        xb, yb = get_batch("train")
        _, loss = m(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return m


@torch.no_grad()
def full_val_loss(m, T=BLOCK, bs=128, pos_offset=0, max_windows=None):
    """The deterministic whole-val-split metric 03–11 reported, now length-aware:
    tile the val split into non-overlapping T-windows and average the CE over all
    of them."""
    m.eval()
    nwin = (len(val_data) - 1) // T
    if max_windows:
        nwin = min(nwin, max_windows)
    x = val_data[:nwin * T].view(nwin, T)
    y = val_data[1:nwin * T + 1].view(nwin, T)
    total = count = 0
    for i in range(0, nwin, bs):
        xb, yb = x[i:i + bs].to(DEV), y[i:i + bs].to(DEV)
        total += m(xb, yb, pos_offset=pos_offset)[1].item() * yb.numel()
        count += yb.numel()
    m.train()
    return total / count


@torch.no_grad()
def loss_by_position(m, T, bs=64, max_windows=384):
    """Mean CE at EACH position 0..T-1, over non-overlapping val windows -> (T,).

    full_val_loss averages this curve; here we keep the shape, because past the
    trained length the shape is the entire finding."""
    m.eval()
    nwin = min((len(val_data) - 1) // T, max_windows)
    x = val_data[:nwin * T].view(nwin, T)
    y = val_data[1:nwin * T + 1].view(nwin, T)
    tot = torch.zeros(T)
    for i in range(0, nwin, bs):
        xb, yb = x[i:i + bs].to(DEV), y[i:i + bs].to(DEV)
        logits, _ = m(xb)
        ls = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.reshape(-1),
                             reduction="none").view(yb.shape)
        tot += ls.sum(0).cpu()
    m.train()
    return tot / nwin


TRAINED = {}
SEEDS = (1337, 7, 42)


def run_seeds(rope, seeds=SEEDS, keep=None):
    """Train one model per seed; return final val losses. `keep` caches seed 0's model."""
    out = []
    for s in seeds:
        torch.manual_seed(s)
        m = train(GPT(vocab_size, rope=rope).to(DEV))
        if keep is not None and s == seeds[0]:
            TRAINED[keep] = m
        out.append(full_val_loss(m))
    return out


# %% [markdown]
"""
## The A/B: 3 seeds each

Same everything — `06`'s model, `07`'s recipe — one delta: learned absolute `pos_emb` vs
rotated `q`/`k`. RoPE deletes the whole position table and adds no parameters of its own,
so this is the rare upgrade that is also strictly smaller.
"""

# %%
t0 = time.time()
ab = {"abs pos_emb (01-11)": run_seeds(rope=False, keep="abs"),
      "RoPE": run_seeds(rope=True, keep="rope")}
print(f"(trained {2 * len(SEEDS)} models in {time.time() - t0:.0f}s)\n")

p_abs = sum(p.numel() for p in GPT(vocab_size, rope=False).parameters())
p_rope = sum(p.numel() for p in GPT(vocab_size, rope=True).parameters())
print(f"{'model':<22} | {'params':>9} | {'seed runs':<24} | {'mean':>7}")
print("-" * 72)
for name, v in ab.items():
    p = p_abs if "abs" in name else p_rope
    print(f"{name:<22} | {p:>9,} | {'  '.join(f'{x:.4f}' for x in v):<24} | "
          f"{sum(v) / len(v):>7.4f}")

mean = lambda k: sum(ab[k]) / len(ab[k])
A, R = "abs pos_emb (01-11)", "RoPE"
spread = max(max(v) - min(v) for v in ab.values())
overlap = not (max(ab[A]) < min(ab[R]) or max(ab[R]) < min(ab[A]))
print(f"\ngap {mean(A) - mean(R):+.4f} (positive = RoPE better), worst seed spread "
      f"{spread:.4f}")
print(f"seed ranges overlap: {overlap}  -> "
      f"{'a tie at this scale' if overlap else 'a real separation'}")
print(f"params: {p_abs - p_rope:,} fewer ({(p_abs - p_rope) / p_abs:.1%}), and the "
      f"length cap is gone")

# %% [markdown]
"""
### Reading the A/B

TODO after the run.
"""

# %% [markdown]
"""
## Shift-invariance: the property, on a trained model

The loss numbers can't see the thing RoPE actually changed, so let's measure it directly.

`pos_offset` translates the *entire* context: feed the same tokens, tell the model they
live at positions `k..k+T-1` instead of `0..T-1`. Nothing about the text changed — only
where it claims to sit. A model that understood **distance** should not care at all. A
model that memorised **slots** has to care.

The derivation says RoPE's outputs are *identical*, not approximately: every score is
`q^T R((n-m)·theta) k`, and adding `k` to both `m` and `n` leaves `n - m` alone. So this
is a proof we can run — and the absolute model gets the same test as a control.

(The windows are 32 tokens, not 64, purely so the abs model can be tested at all: at
offset 32 it needs rows 32..63, and it only has 64 of them. The constraint is itself the
point.)
"""

# %%
abs_m, rope_m = TRAINED["abs"], TRAINED["rope"]
W_SHIFT = 32
xs_shift = val_data[:64 * W_SHIFT].view(64, W_SHIFT).to(DEV)


@torch.no_grad()
def shifted_logits(m, offset):
    m.eval()
    return m(xs_shift, pos_offset=offset)[0]


print(f"the same 64 windows of {W_SHIFT} tokens, moved to a different absolute offset\n")
print(f"{'model':<6} | {'offset':>6} | {'max |Δ logit|':>13} | "
      f"{'mean KL(p0 || p_off)':>20} | {'top-1 flipped':>13}")
print("-" * 70)
for name, m in [("abs", abs_m), ("RoPE", rope_m)]:
    lo0 = shifted_logits(m, 0)
    p0 = F.log_softmax(lo0, dim=-1)
    for off in (1, 8, 32):
        lo = shifted_logits(m, off)
        kl = F.kl_div(F.log_softmax(lo, dim=-1), p0, log_target=True,
                      reduction="none").sum(-1).mean()
        flip = (lo.argmax(-1) != lo0.argmax(-1)).float().mean()
        print(f"{name:<6} | {off:>6} | {(lo - lo0).abs().max():>13.2e} | "
              f"{kl:>20.4f} | {flip:>12.1%}")

print(f"\nand the loss the same text earns, purely as a function of where it sits:")
for name, m in [("abs", abs_m), ("RoPE", rope_m)]:
    ls = [full_val_loss(m, T=W_SHIFT, pos_offset=o, max_windows=256)
          for o in (0, 8, 32)]
    print(f"  {name:<5}: " + "  ".join(f"offset {o}: {l:.4f}" for o, l in
                                       zip((0, 8, 32), ls))
          + f"   (spread {max(ls) - min(ls):.4f})")

# %% [markdown]
"""
TODO reading, after the run.
"""

# %% [markdown]
"""
## Before extrapolating: how much context does this model even use?

Now the second half — the length cap. The pitch writes itself: no table, no cap, RoPE
runs at any length, so point it at 256 tokens and enjoy the longer context.

Before running that, a control that decides what the experiment can possibly mean.
**Does a 4-layer char model trained at 64 tokens actually use 64 tokens of context?** If
its predictions stop improving after 20 characters, then extending to 256 can't *gain*
anything, and the test only measures whether the model **breaks**. Those are very
different claims, and the difference is invisible unless you look.

`loss_by_position` gives it directly: mean loss at each position inside the window.
Position 0 predicts with no context at all; position 63 predicts with 63 characters of
it. The curve is the model's context appetite.
"""

# %%
lbp64 = {"abs": loss_by_position(abs_m, BLOCK), "RoPE": loss_by_position(rope_m, BLOCK)}
print("mean val loss at each position inside the trained 64-token window\n")
print(f"{'position':>9} | {'abs':>7} | {'RoPE':>7}")
print("-" * 29)
for t in (0, 1, 2, 4, 8, 16, 31, 47, 63):
    print(f"{t:>9} | {lbp64['abs'][t]:>7.4f} | {lbp64['RoPE'][t]:>7.4f}")

for name, c in lbp64.items():
    late = c[32:].mean()
    reach = next((t for t in range(BLOCK) if c[t] < late + 0.02), BLOCK)
    print(f"\n{name:>5}: loss falls from {c[0]:.3f} (no context) to "
          f"{late:.3f} (positions 32-63)")
    print(f"       within 0.02 of the plateau by position {reach}")

# %%
plt.figure(figsize=(6.6, 3.6))
for name, c, col in [("abs pos_emb", lbp64["abs"], "#eda100"),
                     ("RoPE", lbp64["RoPE"], "#104281")]:
    plt.plot(range(BLOCK), c, lw=1.8, color=col, label=name)
plt.xlabel("position in the window")
plt.ylabel("mean val loss at that position")
plt.title("How much context does the model use?", fontsize=11)
plt.legend(fontsize=8.5)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
TODO reading, after the run.
"""

# %% [markdown]
"""
## Now run it past the trained length

The abs model can't. Not "does badly" — the forward pass raises. Let's watch that,
because it's the one thing everyone agrees RoPE fixes.

(On a *copy pinned to the CPU*, for a reason worth knowing: an out-of-range embedding
lookup on CUDA is an asynchronous device-side assert, which doesn't raise a catchable
Python error — it poisons the CUDA context and every later op in the process dies with
it. On the CPU the same mistake is a clean `IndexError`. That asymmetry costs people a
lot of debugging time, so when an index bug is *plausible*, reproduce it on the CPU.)
"""

# %%
T_LONG = 256
abs_cpu = copy.deepcopy(abs_m).cpu().eval()
try:
    abs_cpu(val_data[:T_LONG].view(1, T_LONG))
    print(f"abs model at T={T_LONG}: ran?!")
except IndexError as e:
    print(f"abs model at T={T_LONG}: IndexError — {e}")
out_long = rope_m(val_data[:T_LONG].view(1, T_LONG).to(DEV))[0]
print(f"RoPE model at T={T_LONG}: ran fine, logits {tuple(out_long.shape)}")
print("\nSo the length cap is genuinely gone. The question is whether that's worth")
print("anything, and `loss_by_position` at 256 answers it in one curve.")

# %%
lbp256 = loss_by_position(rope_m, T_LONG)
trained_plateau = lbp64["RoPE"][32:].mean().item()
print(f"RoPE, trained at {BLOCK}, evaluated at {T_LONG}\n")
print(f"{'position':>9} | {'loss':>8} | {'vs the trained plateau':>23}")
print("-" * 46)
for t in (16, 32, 63, 64, 70, 96, 128, 192, 255):
    print(f"{t:>9} | {lbp256[t]:>8.4f} | {lbp256[t] - trained_plateau:>+23.4f}")
print(f"\ntrained plateau (positions 32-63, T=64): {trained_plateau:.4f}")
print(f"mean loss BEYOND the trained length (64-255): {lbp256[BLOCK:].mean():.4f}")

# %% [markdown]
"""
### The honest baseline: just slide the window

"RoPE at 256 is worse than at 64" isn't yet an indictment — we need the alternative. And
the alternative is embarrassingly simple: at every position, **feed the model the last 64
tokens** and throw the rest away. Any model can do this, including the absolute one.

Note that the sliding curve needs no extra compute to know: for every position past 63,
a 64-window model sees exactly 63 tokens of context, so its loss *is* the loss at
position 63 of a 64-window — a horizontal line. That's the bar RoPE has to clear.
"""

# %%
slide_rope = lbp64["RoPE"][BLOCK - 1].item()
slide_abs = lbp64["abs"][BLOCK - 1].item()
full_beyond = lbp256[BLOCK:].mean().item()
print(f"mean loss over positions {BLOCK}-{T_LONG - 1}\n")
print(f"{'strategy':<38} | {'loss':>7}")
print("-" * 50)
print(f"{'abs pos_emb, 64-token sliding window':<38} | {slide_abs:>7.4f}")
print(f"{'RoPE, 64-token sliding window':<38} | {slide_rope:>7.4f}")
print(f"{'RoPE, full 256-token context':<38} | {full_beyond:>7.4f}")
print(f"\nusing the longer context is {full_beyond - slide_rope:+.4f} vs just sliding "
      f"a 64-window")

# %%
plt.figure(figsize=(7.4, 3.9))
plt.plot(range(T_LONG), lbp256, lw=1.6, color="#104281", label="RoPE, full context")
plt.axhline(slide_rope, ls="-", lw=1.6, color="#1baf7a",
            label="RoPE, 64-token sliding window")
plt.axvline(BLOCK, ls="--", c="#e34948", lw=1.2)
plt.text(BLOCK + 4, plt.ylim()[1] * 0.93, "trained length", fontsize=8, c="#e34948")
plt.xlabel("position")
plt.ylabel("mean val loss")
plt.title("No length cap is not the same thing as working past the length",
          fontsize=11)
plt.legend(fontsize=8.5)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
TODO reading, after the run.
"""

# %% [markdown]
"""
## Diagnosis: which planes left the arc?

The frequency ladder predicted this, and we even printed the number.

During training at length `L`, a query and a key are at most `L-1` apart, so plane `k`
only ever sees relative angles in `[0, L·theta_k]`. Whether that's the whole circle or a
sliver depends on the plane:

- **fast planes** (`wavelength <= L`): they complete at least one full turn inside the
  training window, so the model saw **every angle** they can ever produce. At distance
  300 they land somewhere they've already been. Nothing new.
- **slow planes** (`wavelength > L`): they swept only a partial arc. At distance `> L`
  they rotate **past every angle the model has ever seen**, into genuinely novel
  territory — and the model's weights have no idea what to do with it.

So the damage is entirely the **low-frequency** end. Here's the census for our head:
"""

# %%
twopi = 2 * math.pi
wl = twopi / theta_full
print(f"trained at L = {BLOCK}, evaluating at {T_LONG} (a {T_LONG // BLOCK}x extension)\n")
print(f"{'plane':>6} | {'wavelength':>11} | {'max angle @L':>13} | "
      f"{'max angle @256':>15} | {'seen in training?'}")
print("-" * 74)
for k in [0, 2, 4, 5, 6, 8, 12, 15]:
    th = theta_full[k].item()
    ok = "yes — full turn" if wl[k] <= BLOCK else "NO — unseen arc"
    print(f"{k:>6} | {wl[k]:>11.1f} | {BLOCK * th:>13.3f} | {T_LONG * th:>15.3f} | {ok}")
print(f"\n{n_safe}/{HS // 2} planes are safe. {HS // 2 - n_safe}/{HS // 2} spend the "
      f"whole extension in angles the model never trained on.")
print("That is the cliff. And it says the fix must be per-plane: leave the fast ones")
print("alone, and do something about the slow ones.")

# %% [markdown]
"""
## Every fix is the same knob

Here's the tidy thing about the context-extension literature: PI, NTK-aware and YaRN look
like three different ideas, and they are all **one line** — a per-plane divisor on the
frequencies. That's the `freq_div` argument `build_rope_cache` has been carrying.

```-
theta'_k = theta_k / div_k
```

Dividing a plane's frequency stretches its wavelength, which drags the angles at long
distances back down into the arc the model trained on. The methods differ **only in the
shape of `div_k`**:

**Position Interpolation** (Chen et al., Llama-2-long). Scale every position by `L/L'`.
Since the angle is `m·theta_k`, that's identical to dividing every frequency by
`s = L'/L`:

```-
div_k = s                       (flat — every plane, same squeeze)
```

Max angle at the new length `L'` becomes `L'·theta_k/s = L·theta_k` — exactly the trained
maximum, for every plane. Simple and it works, but the squeeze is **uniform**: the fast
planes, which were fine, also get blurred by `s`. Adjacent tokens now differ by a `s`x
smaller angle everywhere.

**NTK-aware** (bloc97). Same total stretch, spread unevenly: raise the *base* instead.
Since `theta_k = base^(-2k/d)`, multiplying the base by `a` multiplies `theta_k` by
`a^(-2k/d)` — a geometric ramp. Choose `a` so the *slowest* plane gets the full `s`:

```-
a^(-(d-2)/d) = 1/s   ->   a = s^(d/(d-2))   ->   div_k = s^(2k/(d-2))
```

which runs from `div_0 = 1` (plane 0 untouched — full local resolution kept) to
`div_last = s` (full PI on the slowest plane). One line, no fine-tuning, which is why it
spread through the community before it reached any paper.

**YaRN** (Peng et al., Llama-3.1 / Qwen2). Stop ramping smoothly and make the
safe/unsafe split from the census above **explicit**. Let `r_k = L / wavelength_k` be the
turns a plane completes in the training window:

```-
r_k >= beta   -> extrapolate: div_k = 1     (it saw every angle; don't touch it)
r_k <= alpha  -> interpolate: div_k = s     (it saw a sliver; full PI)
in between    -> linear ramp
```

Plus a piece the other two skip: packing more positions into the same angular arc makes
`q·k` scores more peaked, so the attention softmax sharpens as you extend. YaRN corrects
it with a **temperature** `t = 0.1·ln(s) + 1`, dividing the scores by `t`.

Three papers, one vector. Let's draw it.
"""


# %%
def freq_divisor(method, s, head_size=HS, train_len=BLOCK, alpha=1.0, beta=8.0):
    """The per-plane frequency divisor div_k. theta'_k = theta_k / div_k.

    method: none (vanilla) | pi (flat s) | ntk (geometric ramp) | yarn (clamped ramp).
    alpha/beta are YaRN's turns-in-L thresholds — retuned for our 32-wide head; Llama
    uses 1/32 at head_size 128.
    """
    idx = torch.arange(0, head_size, 2).float()          # 2k
    if method == "none":
        return None
    if method == "pi":
        return torch.full((head_size // 2,), float(s))
    if method == "ntk":
        return torch.tensor(float(s)) ** (idx / (head_size - 2))
    if method == "yarn":
        turns = train_len * rope_frequencies(head_size) / twopi      # r_k
        gamma = ((turns - alpha) / (beta - alpha)).clamp(0, 1)       # 1=extrap, 0=interp
        return s - gamma * (s - 1)
    raise ValueError(method)


def set_rope(m, freq_div=None, temp=1.0):
    """Retune a TRAINED RoPE model's positional frequencies. No weights change."""
    for b in m.blocks:
        b.attn.freq_div = freq_div
        b.attn.rope_temp = temp
    return m


S_EXT = T_LONG / BLOCK
plt.figure(figsize=(6.8, 3.6))
for meth, col, lab in [("pi", "#e34948", "PI (flat)"),
                       ("ntk", "#eda100", "NTK-aware (geometric ramp)"),
                       ("yarn", "#104281", "YaRN (clamped ramp)")]:
    plt.plot(range(HS // 2), freq_divisor(meth, S_EXT), "o-", lw=1.8, color=col,
             label=lab)
plt.axhline(1.0, ls="--", c="gray", lw=1)
plt.text(0.1, 1.06, "1.0 = leave this plane alone (extrapolate)", fontsize=7.5,
         c="gray")
plt.axvline(n_safe - 0.5, ls=":", c="#1baf7a", lw=1.4)
plt.text(n_safe + 0.1, S_EXT * 0.55, "planes left of here\nsaw every angle",
         fontsize=7.5, c="#177f5b")
plt.xlabel("plane k  (fast → slow)")
plt.ylabel(f"frequency divisor div_k  (extension {S_EXT:.0f}x)")
plt.title("PI, NTK and YaRN are three shapes of one vector", fontsize=11)
plt.legend(fontsize=8)
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
## Do they work? (no fine-tuning, just the divisor)

The strong version of the claim: take the **already-trained** RoPE model, change nothing
but `freq_div` at eval time, and see whether the cliff flattens. No gradient steps, no
data. All three methods normally come with a short fine-tune; this is them at their most
honest and least flattering.

Bar to clear: the 64-token sliding window. If none of these beats it, the longer context
is worth nothing at this scale and we should say so.
"""

# %%
YARN_TEMP = 0.1 * math.log(S_EXT) + 1
methods = [
    ("vanilla RoPE", "none", 1.0),
    ("PI", "pi", 1.0),
    ("NTK-aware", "ntk", 1.0),
    ("YaRN (no temp)", "yarn", 1.0),
    ("YaRN + temp", "yarn", YARN_TEMP),
]
curves, ext = {}, {}
for label, meth, temp in methods:
    div = freq_divisor(meth, S_EXT)
    set_rope(rope_m, None if div is None else div.to(DEV), temp)
    curves[label] = loss_by_position(rope_m, T_LONG)
    ext[label] = curves[label][BLOCK:].mean().item()
set_rope(rope_m, None, 1.0)          # always put the model back

print(f"trained at {BLOCK}, evaluated at {T_LONG} ({S_EXT:.0f}x), no fine-tuning")
print(f"YaRN attention temperature t = 0.1·ln({S_EXT:.0f})+1 = {YARN_TEMP:.4f}\n")
print(f"{'method':<18} | {'mean loss @64-255':>17} | {'vs sliding window':>18}")
print("-" * 62)
print(f"{'64-tok sliding win':<18} | {slide_rope:>17.4f} | {'(the bar)':>18}")
for label, _, _ in methods:
    print(f"{label:<18} | {ext[label]:>17.4f} | {ext[label] - slide_rope:>+18.4f}")
print("\n(negative = better than throwing the extra context away)")

# %%
plt.figure(figsize=(7.6, 4.0))
COLORS = {"vanilla RoPE": "#9aa4b0", "PI": "#e34948", "NTK-aware": "#eda100",
          "YaRN (no temp)": "#6fa8dc", "YaRN + temp": "#104281"}
for label, _, _ in methods:
    plt.plot(range(T_LONG), curves[label], lw=1.5, color=COLORS[label], label=label)
plt.axhline(slide_rope, ls="-", lw=1.6, color="#1baf7a", label="64-tok sliding window")
plt.axvline(BLOCK, ls="--", c="#e34948", lw=1.2)
plt.text(BLOCK + 4, plt.ylim()[1] * 0.93, "trained length", fontsize=8, c="#e34948")
plt.xlabel("position")
plt.ylabel("mean val loss")
plt.title(f"Extending {BLOCK} → {T_LONG} by retuning frequencies alone", fontsize=11)
plt.legend(fontsize=8, loc="upper right")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
TODO reading, after the run.
"""

# %% [markdown]
"""
## What it costs

`09` and `11` both ended by measuring a cost the arithmetic said wouldn't be there, so
let's keep the habit. RoPE adds two elementwise multiplies, a `cat` and an add per
attention — plus `cos`/`sin` tables it rebuilds on every forward, in every block, because
that's the simplest thing to write.
"""

# %%
def bench(m, T=BLOCK, iters=30):
    """Median ms per fwd+bwd on a full batch, after warmup. CUDA-synced."""
    xb = val_data[:BATCH * T].view(BATCH, T).to(DEV)
    yb = val_data[1:BATCH * T + 1].view(BATCH, T).to(DEV)
    for _ in range(8):
        m(xb, yb)[1].backward()
    if DEV == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        m(xb, yb)[1].backward()
        if DEV == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    m.zero_grad(set_to_none=True)
    ts.sort()
    return ts[len(ts) // 2]


t_abs, t_rope = bench(abs_m), bench(rope_m)
print(f"batch {BATCH} x {BLOCK} tokens, fwd+bwd, median of 30\n")
print(f"{'model':<22} | {'params':>9} | {'time':>8}")
print("-" * 46)
print(f"{'abs pos_emb':<22} | {p_abs:>9,} | {t_abs:>6.2f}ms")
print(f"{'RoPE':<22} | {p_rope:>9,} | {t_rope:>6.2f}ms")
print(f"\nRoPE is {t_rope / t_abs - 1:+.0%} wall-clock for {p_abs - p_rope:,} fewer "
      f"params.")
print("Rebuilding cos/sin per block per forward is most of that, and it's pure waste:")
print("the tables depend on (T, head_size, base) only. Real implementations build them")
print("once at init and slice. We'll want that anyway in `13`.")

# %% [markdown]
"""
## The payoff: read the RoPE model
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
    """08's decode step. Crop to block_size: the model RUNS at any length, but the
    section above is why we don't ask it to."""
    m.eval()
    for _ in range(max_new_tokens):
        logits = m(idx[:, -m.block_size:])[0][:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        probs = probs.masked_fill(~nucleus_mask(probs, top_p), 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
    return idx


print(f"RoPE GPT      : val {mean(R):.4f}  ({p_rope:,} params)")
print(f"abs pos_emb   : val {mean(A):.4f}  ({p_abs:,} params)")
print("=" * 78)
torch.manual_seed(1337)
start = torch.tensor([[stoi["\n"]]], device=DEV)
print(decode(generate(rope_m, start, 500)[0].tolist()))

# %% [markdown]
"""
## What you built, and where `13` goes next

TODO — written last.
"""
