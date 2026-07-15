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
# LLM · 10 — activation functions: the one line the FFN is built around

`06` built the FeedForward and justified two of its three design choices. It explained
the `4x` expansion, it proved the position-wise property — and then it said this about
the third:

> **GELU**, not ReLU. A smooth gate — `x · Φ(x)` — that GPT-2 and most transformers use;
> the smoothness gives cleaner gradients than ReLU's hard corner.

That's a claim about *which* activation, and we never asked the prior question: **why is
there an activation there at all?** `10` pays that debt, because `11` can't be read
without it — SwiGLU changes exactly this line, and you can't judge the change without
knowing what the line does.

The answer turns out to be less of a detail than the notebook order suggests. Delete the
activation and the FeedForward doesn't get *slightly worse* — it **stops existing**. Two
stacked `Linear`s are one `Linear`:

    W2·(W1·x + b1) + b2 = (W2·W1)·x + (W2·b1 + b2)
                          └─ W ─┘       └─ b ─┘

so `C → 4C → C` without a nonlinearity collapses into a single `C → C` map. All 131,712
parameters of the FFN express what 16,512 could. The `4x` expansion `06` was so careful
to justify buys **literally nothing**. We prove that by fusion, then pay the bill on the
scoreboard.

Then the harder question — *which* one — and here the measurements got interesting:

- **the zoo disappoints.** Saturating activations (sigmoid, tanh) do badly, for the
  vanishing-gradient reason `06` taught. Fine. But **SiLU** — the modern one, the one
  Llama runs, the one inside SwiGLU — comes out clearly *worse* than plain GELU on this
  model, across every seed. That looked like a bug.
- **it isn't a bug, and one knob explains the whole zoo.** Swish is `x · σ(β·x)`, and
  `β` is a *sharpness* dial: `β → 0` is a straight line, `β = 1` **is** SiLU, `β ≈
  1.702` reproduces GELU to 0.02, `β → ∞` is ReLU. They aren't different ideas — they're
  one gate at different sharpness. We sweep `β` and the loss curve explains every result
  in the table, including SiLU's.
- **and the reason connects to the top of the notebook.** For small `x`, `x·σ(βx) ≈ 0.5x
  + (β/4)x²` — the *nonlinear* term is proportional to `β`. A softer gate is literally
  closer to a straight line, and we opened by measuring exactly what a straight line
  costs.
- **so we let the model choose.** `β` is a learnable parameter (that's Swish's original
  definition). We train it and read off what it picks.

> Framing and the `β` walkthrough follow Carlos Roldán's *What is SwiGLU?*
> (jcarlosroldan.com/post/348), which sets this up more clearly than the papers do. The
> numbers here are ours.
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
DATA = ROOT / "nb" / "llm" / "data" / "input.txt"  # tinyshakespeare, shared with 01–09
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
print(f"device: {DEV}")

# %% [markdown]
"""
## Same data skeleton as `03`–`09`
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
## Two linear layers are one linear layer

Start with the algebra, because everything else follows from it. A `Linear` computes
`W·x + b`. Stack two:

```-
linear2(linear1(x)) = W2·(W1·x + b1) + b2
                    = W2·W1·x + W2·b1 + b2
                    = (W2·W1)·x + (W2·b1 + b2)
                      └─ W ─┘      └─── b ───┘
                    = W·x + b            <- a single Linear
```

Depth bought nothing. `W2·W1` is just some matrix; `W2·b1 + b2` is just some vector. Any
function two stacked linear layers can compute, **one** linear layer already computes —
and the same argument telescopes to any depth. This is why a nonlinearity has to sit
between them, and it's the entire reason activation functions exist.

`09` proved things by fusion and we'll do it again, because a fusion that matches to ~0
is the strongest form of "these are the same function":
"""

# %%
C = 16
x = torch.randn(8, C)

lin1, lin2 = nn.Linear(C, 32), nn.Linear(32, C)
fused = nn.Linear(C, C)
with torch.no_grad():
    fused.weight.copy_(lin2.weight @ lin1.weight)          # W = W2 @ W1
    fused.bias.copy_(lin2.weight @ lin1.bias + lin2.bias)  # b = W2 @ b1 + b2

stacked, single = lin2(lin1(x)), fused(x)
print(f"Linear(Linear(x)) == a single Linear(x)? "
      f"{torch.allclose(stacked, single, atol=1e-5)}"
      f"   (max diff {(stacked - single).abs().max():.2e})")

# %% [markdown]
"""
It telescopes. Stack **six** linear layers, fold them all into one matrix, and the
6-layer network is *exactly* reproduced by a single `Linear` — no matter how wide the
hidden layers were:
"""

# %%
torch.manual_seed(0)
widths = [C, 64, 128, 64, 32, 64, C]
deep = nn.Sequential(*[nn.Linear(a, b) for a, b in zip(widths, widths[1:])])

W, b = torch.eye(C), torch.zeros(C)          # fold the whole stack into (W, b)
for layer in deep:
    W, b = layer.weight @ W, layer.weight @ b + layer.bias
one = nn.Linear(C, C)
with torch.no_grad():
    one.weight.copy_(W)
    one.bias.copy_(b)

n_deep = sum(p.numel() for p in deep.parameters())
n_one = sum(p.numel() for p in one.parameters())
print(f"a {len(deep)}-layer linear net ({n_deep:,} params) "
      f"vs one Linear ({n_one:,} params)")
print(f"identical outputs? {torch.allclose(deep(x), one(x), atol=1e-4)}"
      f"   (max diff {(deep(x) - one(x)).abs().max():.2e})")
print(f"-> {n_deep:,} parameters expressing exactly {n_one:,} params' worth of map.")

# %% [markdown]
"""
## What that means for `06`'s FeedForward

Now apply it to the actual layer. `06`'s FFN is `Linear(C, 4C) → GELU → Linear(4C, C)`.
Take the GELU out and it's two stacked linear layers — so by the argument above it is
**one** `C → C` linear map, and the `4x` hidden layer that `06` called "where most of
the model's knowledge lives" is an elaborate way of writing a `128×128` matrix.

Let's fuse a real FFN and count the damage:
"""


# %%
class FeedForward(nn.Module):
    """06's FFN, with the activation injected so we can swap or delete it."""

    def __init__(self, n_embd, act):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.proj.RESIDUAL_PROJ = True
        self.act = act

    def forward(self, x):
        return self.proj(self.act(self.fc(x)))


EMBD = 128
ff_none = FeedForward(EMBD, nn.Identity())          # the FFN with NO activation
xb = torch.randn(4, 16, EMBD)

collapsed = nn.Linear(EMBD, EMBD)
with torch.no_grad():
    collapsed.weight.copy_(ff_none.proj.weight @ ff_none.fc.weight)
    collapsed.bias.copy_(ff_none.proj.weight @ ff_none.fc.bias + ff_none.proj.bias)

p_ffn = sum(p.numel() for p in ff_none.parameters())
p_col = sum(p.numel() for p in collapsed.parameters())
print(f"FFN without an activation == one Linear({EMBD}, {EMBD})? "
      f"{torch.allclose(ff_none(xb), collapsed(xb), atol=1e-4)}")
print(f"  FFN params            : {p_ffn:,}")
print(f"  the map it computes   : {p_col:,}  ({p_ffn / p_col:.1f}x fewer)")
print("\nThe 4x expansion is real parameters doing NO extra work. The activation is")
print("what makes the hidden layer count — without it, the width is decoration.")

# %% [markdown]
"""
### The same thing, said with a picture

The clearest way to see it: a linear map can only ever draw a **straight line**. Stack a
hundred of them and it's still a straight line. One `GELU` in the middle and the same
two layers can bend:
"""

# %%
torch.manual_seed(0)
xs = torch.linspace(-3, 3, 300).unsqueeze(1)
plt.figure(figsize=(6.5, 3.0))
for name, act in [("no activation (identity)", nn.Identity()), ("GELU", nn.GELU())]:
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(1, 64), act, nn.Linear(64, 1))
    plt.plot(xs, net(xs).detach(), label=name, lw=1.8)
plt.axhline(0, c="gray", lw=0.5)
plt.xlabel("input")
plt.ylabel("output of a random 1→64→1 net")
plt.title("same architecture, same weights — one line is a line")
plt.legend(fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
## The model, activation-swappable

`06`'s GPT with the FFN's activation as a constructor argument — the only thing that
varies in every experiment below.
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


class Block(nn.Module):
    """06's Block, with the FFN's activation injected."""

    def __init__(self, n_embd, n_head, act_fn, block_size=BLOCK):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, act_fn())

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """06's GPT. `act_fn` is a zero-arg factory: a FRESH activation per block, so a
    learnable one (Swish's beta) gets its own parameter in each FFN."""

    def __init__(self, vocab_size, act_fn=nn.GELU, n_embd=EMBD, n_head=4, n_layer=4,
                 block_size=BLOCK):
        super().__init__()
        self.block_size, self.n_layer = block_size, n_layer
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, act_fn, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
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
    """The same deterministic whole-val-split metric `03`–`09` reported."""
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


def run_seeds(act_fn, seeds=(1337, 7, 42)):
    """Train one model per seed, return the list of final val losses."""
    out = []
    for s in seeds:
        torch.manual_seed(s)
        out.append(full_val_loss(train(GPT(vocab_size, act_fn).to(DEV))))
    return out


# %% [markdown]
"""
### Pay the bill: what deleting the activation actually costs

The fusion proof says the FFN collapses. Here's the scoreboard version — `06`'s exact
model, GELU vs nothing at all. (The model isn't *entirely* linear without it:
attention's softmax is still a nonlinearity. This measures only the FFN's contribution
collapsing.)
"""

# %%
gelu_seeds = run_seeds(nn.GELU)
none_seeds = run_seeds(nn.Identity)
gelu_mean = sum(gelu_seeds) / len(gelu_seeds)
none_mean = sum(none_seeds) / len(none_seeds)
print(f"GELU              : " + "  ".join(f"{v:.4f}" for v in gelu_seeds)
      + f"   mean {gelu_mean:.4f}")
print(f"no activation     : " + "  ".join(f"{v:.4f}" for v in none_seeds)
      + f"   mean {none_mean:.4f}")
print(f"\ncost of deleting one function call: {none_mean - gelu_mean:+.4f} val loss")
print("Same parameters, same steps, same everything. That gap is the whole reason the")
print("FeedForward is a FeedForward and not a matrix.")

# %% [markdown]
"""
## So which one? Meet the zoo

Every activation here is trying to be the same thing: **cheap, nonlinear, and easy to
push a gradient through**. They differ in how they trade those off.

- **ReLU** — `max(0, x)`. The default for a decade because it is almost free: a compare
  and a conditional move. Its derivative is `1` for `x > 0` and `0` otherwise — no
  saturation on the positive side, which is what fixed the deep-net training problem
  sigmoid had.
- **sigmoid** — `1 / (1 + e^-x)`. Squashes anything into `(0, 1)`, which is why it's
  perfect for *probabilities* and terrible for *hidden layers*: push `|x|` past ~4 and
  the curve goes flat, so the derivative goes to ~0 and the gradient dies. That's the
  vanishing-gradient problem `06` measured from the other direction.
- **tanh** — the same S-curve rescaled to `(-1, 1)`. Zero-centred, which helps, but it
  saturates exactly the same way.
- **GELU** — `x · Φ(x)`, where `Φ` is the Gaussian CDF. Instead of a hard `0/1` switch,
  it multiplies `x` by *how likely* it is to be positive. Smooth, and it lets a little
  negative signal through.
- **SiLU / Swish** — `x · σ(x)`. The same idea with the logistic sigmoid as the gate.

The shapes matter less than the **derivatives**, so we plot both — one column per
activation, with the other four ghosted in grey behind each so you can still see the
whole field without five lines fighting in one axes.

The **bottom row** is where the story is. Backprop multiplies by `f'(x)` at every layer
it crosses, so a flat derivative means no gradient means no learning. Read it left to
right: ReLU, GELU and SiLU all get their derivative up to ~1 and hold it (and the top
row shows why they look interchangeable — the colour barely leaves the grey). `tanh`'s
derivative is a narrow bump that decays to nothing either side of zero. And sigmoid's
never gets off the floor *at all*.
"""

# %%
ACTS = [
    ("ReLU", torch.relu, "#2a78d6"),
    ("GELU", F.gelu, "#1baf7a"),
    ("SiLU (Swish, b=1)", F.silu, "#eda100"),
    ("tanh", torch.tanh, "#4a3aa7"),
    ("sigmoid", torch.sigmoid, "#e34948"),
]
GHOST = "#d6d6d4"

xs = torch.linspace(-5, 5, 1001, requires_grad=True)
# (value, derivative) for each activation, once
curves = [(fn(xs), torch.autograd.grad(fn(xs).sum(), xs, retain_graph=True)[0])
          for _n, fn, _c in ACTS]
xd = xs.detach()

# One panel per activation (not five lines in one axes): the focal curve in colour,
# the other four ghosted behind it so every panel still shows the whole field.
fig, axes = plt.subplots(2, 5, figsize=(13, 4.6), sharex=True, sharey="row")
for col, (name, _fn, color) in enumerate(ACTS):
    for row in (0, 1):
        ax = axes[row][col]
        for j, pair in enumerate(curves):
            if j != col:
                ax.plot(xd, pair[row].detach(), color=GHOST, lw=1, zorder=1)
        ax.plot(xd, curves[col][row].detach(), color=color, lw=2, zorder=3)
        ax.axhline(0, color="#999", lw=0.5, zorder=0)
        ax.grid(alpha=0.18, lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][col].set_title(name, fontsize=9.5, color="#222")

axes[0][0].set_ylim(-1.3, 5.2)
axes[1][0].set_ylim(-0.25, 1.3)
axes[0][0].set_ylabel("f(x)", fontsize=10)
axes[1][0].set_ylabel("f'(x)\n(what backprop multiplies by)", fontsize=9)
for ax in axes[1]:
    ax.set_xlabel("x", fontsize=9)

# the one number that matters, labelled on the panel it belongs to
axes[1][4].axhline(0.25, ls="--", lw=1, color="#e34948", zorder=2)
axes[1][4].annotate("peaks at 0.25", xy=(-4.6, 0.36), fontsize=8, color="#222")
fig.suptitle("Each panel: one activation in colour, the other four in grey for scale",
             fontsize=10, y=0.99, color="#444")
fig.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# %%
# Backprop multiplies by f'(x) at every activation it crosses, so f' IS the gradient's
# transmission factor. Two readings of it: the CEILING (the biggest f' ever gets), and
# the TYPICAL value at the input scale this model really runs at — the pre-activation
# std we measure further down, ~1.64.
#
# The .sum() is the standard elementwise-derivative trick: autograd needs a scalar, and
# since output i depends only on input i, d(sum)/dx_i = f'(x_i) — one backward pass
# returns f' at all 20,001 grid points at once.
torch.manual_seed(0)
real_scale = torch.randn(200_000) * 1.64          # the pre-activations this model sees
ceiling, typical = "max f'(x)", "mean |f'(x)| at our scale"
print(f"{'activation':<18} | {ceiling:>10} | {typical:>25}")
print("-" * 60)
for name, fn, _c in ACTS:
    xg = torch.linspace(-8, 8, 20001, requires_grad=True)
    g, = torch.autograd.grad(fn(xg).sum(), xg)             # f' on a dense grid
    xs = real_scale.clone().requires_grad_(True)
    gs, = torch.autograd.grad(fn(xs).sum(), xs)     # f' where the data actually is
    print(f"{name:<18} | {g.max():>10.3f} | {gs.abs().mean():>25.3f}")

# %% [markdown]
"""
**Sigmoid's problem is not that it saturates at the ends — it's that it never passes a
gradient at all.** Its derivative *peaks* at `0.25`, so every sigmoid a gradient crosses
shrinks it by at least `4x`, before saturation even enters the argument. Compounded over
a stack, that's the vanishing gradient in one number. `tanh` fixes the ceiling (its
derivative peaks at `1.0`) and still averages well under the others, because at our
input scale it spends most of its time out on the flat parts. The three modern ones all
pass roughly half the gradient through.

Keep the limits of this metric in mind, though: it explains why the **saturating** pair
lose, and that's all it explains. It does *not* rank ReLU, GELU and SiLU correctly
against each other — as we're about to see, ReLU passes *less* gradient than SiLU and
still beats it. A different mechanism decides that fight.
"""

# %% [markdown]
"""
### The A/B, 3 seeds each

`09` taught us not to trust a single run, so: same model, same recipe, only the FFN's
activation changes.
"""

# %%
TABLE = [
    ("no activation", nn.Identity),
    ("sigmoid", nn.Sigmoid),
    ("tanh", nn.Tanh),
    ("SiLU (Swish b=1)", nn.SiLU),
    ("GELU", nn.GELU),
    ("ReLU", nn.ReLU),
]
results = {"no activation": none_seeds, "GELU": gelu_seeds}   # already trained above
for name, fn in TABLE:
    if name not in results:
        results[name] = run_seeds(fn)

print(f"{'activation':<18} | {'seed runs':<24} | {'mean':>7} | {'spread':>7}")
print("-" * 66)
for name, _fn in TABLE:
    v = results[name]
    print(f"{name:<18} | {'  '.join(f'{x:.4f}' for x in v):<24} | "
          f"{sum(v) / len(v):>7.4f} | {max(v) - min(v):>7.4f}")

# %% [markdown]
"""
Two things to read off, one expected and one not.

**Expected:** no activation is a disaster, and the two *saturating* activations are
clearly bad — sigmoid worst, tanh better (it's zero-centred), both far behind. That's
the vanishing-gradient story, and the derivative plot predicted it.

**Not expected:** **SiLU loses to GELU**, and not by a whisker — by more than the seed
spread, on every seed. That is genuinely strange. SiLU is the *modern* choice: Llama
runs it, and it's the activation inside the SwiGLU that `11` is about to sell you. It's
also nearly the same curve as GELU — they never differ by more than ~0.19 anywhere:
"""

# %%
xr = torch.linspace(-5, 5, 10001)
d_silu = (F.gelu(xr) - F.silu(xr)).abs().max()
d_relu = (F.gelu(xr) - torch.relu(xr)).abs().max()
print(f"max |GELU(x) - SiLU(x)| over x in [-5,5] : {d_silu:.4f}")
print(f"max |GELU(x) - ReLU(x)| over x in [-5,5] : {d_relu:.4f}")
print("\nNearly the same function, reliably different loss. Something real is up.")

# %% [markdown]
"""
## One knob explains the entire table

Here's the resolution, and it's the best idea in this notebook. **Swish** is not really
a function — it's a *family*:

    Swish(x) = x · sigmoid(beta * x)

`beta` is a **sharpness** dial on the gate, and the family swallows the zoo:

- **`beta -> 0`**: `sigmoid(0) = 0.5`, so `Swish(x) -> x/2` — a **straight line**. The
  activation stops activating.
- **`beta = 1`**: exactly **SiLU**. This is the definition.
- **`beta ≈ 1.702`**: reproduces **GELU** to within 0.02 — the classic approximation.
- **`beta -> ∞`**: `sigmoid(beta*x)` becomes a hard 0/1 step, so `Swish(x) -> max(0, x)`
  — **ReLU**.

So SiLU, GELU and ReLU are *not* competing ideas. They are **one gate at three
sharpnesses**, and the table above is secretly a measurement of `beta`. Verify the two
endpoints numerically before we lean on them:
"""


# %%
class Swish(nn.Module):
    """x * sigmoid(beta*x). beta is a sharpness dial: 0 -> linear, 1 -> SiLU,
    1.702 -> ~GELU, inf -> ReLU. Optionally learnable (Swish's original definition)."""

    def __init__(self, beta=1.0, learnable=False):
        super().__init__()
        b = torch.tensor(float(beta))
        if learnable:
            self.beta = nn.Parameter(b)
        else:
            self.register_buffer("beta", b)

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


xr = torch.linspace(-5, 5, 10001)
g_silu = (Swish(1.0)(xr) - F.silu(xr)).abs().max()
g_gelu = (Swish(1.702)(xr) - F.gelu(xr)).abs().max()
g_relu = (Swish(100.0)(xr) - torch.relu(xr)).abs().max()
g_lin = (Swish(0.001)(xr) - xr / 2).abs().max()
print(f"beta=1.000 vs SiLU        : max diff {g_silu:.2e}   <- Swish(1) IS SiLU")
print(f"beta=1.702 vs GELU        : max diff {g_gelu:.4f}  <- the classic approx")
print(f"beta=100   vs ReLU        : max diff {g_relu:.4f}  <- a hard step = ReLU")
print(f"beta=0.001 vs the line x/2: max diff {g_lin:.4f}  <- no activation")

# %%
# beta is an ORDERED magnitude, not five unrelated categories — so it gets one hue
# stepped light→dark, and you read the knob turning as the curve darkens.
FAMILY = [
    (0.01, "#86b6ef", "beta→0: x/2, a straight line"),
    (0.5, "#5598e7", "beta=0.5"),
    (1.0, "#2a78d6", "beta=1: SiLU"),
    (1.702, "#1c5cab", "beta=1.702: ≈GELU"),
    (50.0, "#104281", "beta→∞: ReLU"),
]
xf = torch.linspace(-4, 3, 1401)          # zoomed to where the family actually differs
fig, ax = plt.subplots(figsize=(8.4, 3.6))
for b, color, label in FAMILY:
    ax.plot(xf, Swish(b)(xf), color=color, lw=2, label=label, zorder=3)
ax.axhline(0, color="#999", lw=0.5, zorder=0)
ax.axvline(0, color="#999", lw=0.5, zorder=0)
ax.grid(alpha=0.18, lw=0.5)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.set_xlabel("x")
ax.set_ylabel("x * sigmoid(beta*x)")
ax.set_title("One knob: beta dials the gate from a straight line to ReLU",
             fontsize=11, color="#222")
# legend OUTSIDE the axes (it used to sit on the data), reversed so its order matches
# the order the curves stack in at the right-hand edge
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], fontsize=8.5, frameon=False,
          loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.tight_layout()
plt.show()

# %% [markdown]
"""
### Sweep the knob and the table falls out

If the zoo really is one dial, then training at a few `beta` values should reproduce
every row of the A/B — and tell us *why* SiLU lost.
"""

# %%
BETAS = [0.5, 1.0, 1.702, 3.0, 6.0]
beta_res = {}
for b in BETAS:
    beta_res[b] = run_seeds(lambda b=b: Swish(b), seeds=(1337, 7))
    tag = {1.0: "  <- SiLU", 1.702: "  <- ≈GELU"}.get(b, "")
    v = beta_res[b]
    print(f"  beta={b:<6}: " + "  ".join(f"{x:.4f}" for x in v)
          + f"   mean {sum(v) / len(v):.4f}{tag}")

# %%
plt.figure(figsize=(6.4, 3.4))
means = [sum(beta_res[b]) / len(beta_res[b]) for b in BETAS]
plt.plot(BETAS, means, "o-", lw=1.6)
for b, m in zip(BETAS, means):
    lbl = {1.0: "SiLU", 1.702: "≈GELU"}.get(b)
    if lbl:
        plt.annotate(lbl, (b, m), fontsize=8, xytext=(4, 6), textcoords="offset points")
plt.axhline(gelu_mean, ls="--", c="gray", lw=0.8)
plt.text(4.2, gelu_mean + 0.002, "GELU (measured)", fontsize=8, color="gray")
plt.xlabel("beta — gate sharpness")
plt.ylabel("final val loss")
plt.title("softer gate = more linear = worse")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
### Why softer is worse — and it's the first section again

The curve is monotone through the interesting range: the softer the gate, the worse the
loss. `beta = 0.5` is bad, SiLU (`beta = 1`) is better, GELU (`beta ≈ 1.702`) better
still, and it flattens out around ReLU.

The reason is a two-line Taylor expansion. Near `x = 0`, `sigmoid(z) ≈ 0.5 + z/4`, so:

```-
x · sigmoid(beta*x)  ≈  x · (0.5 + beta*x/4)  =  0.5*x  +  (beta/4)*x^2
                                                 └ linear ┘   └ the nonlinear part ┘
```

**The nonlinear term is proportional to `beta`.** A soft gate isn't a different flavour
of nonlinearity — it's *less* nonlinearity. Send `beta` to 0 and the `x²` term vanishes
and you're left with `x/2`: a straight line, the thing we spent the first half of this
notebook proving is worthless. SiLU sits at `beta = 1` where GELU sits at `1.702`, so
**SiLU is ~40% less curved than GELU at the same input scale**, and that's the entire
gap.

"At the same input scale" is the load-bearing part. What counts is `beta` measured
against how big the pre-activations actually are — a gate that switches over a range of
`1/beta` is sharp for data spread over `10` and nearly linear for data spread over
`0.1`. So the right `beta` is a property of *your* model's scale, not a universal
constant. Ours:
"""


# %%
@torch.no_grad()
def preact_std(m):
    """std of the tensor entering the FFN's activation, per block."""
    stats, handles = [], []
    for b in m.blocks:
        handles.append(b.ffwd.fc.register_forward_hook(
            lambda mod, i, o: stats.append(o.std().item())))
    m(get_batch("val")[0])
    for h in handles:
        h.remove()
    return stats


torch.manual_seed(1337)
probe = train(GPT(vocab_size, nn.GELU).to(DEV))
probe.eval()
stds = preact_std(probe)
print("pre-activation std entering the FFN's activation, per block:")
for i, s in enumerate(stds):
    print(f"  block{i}: {s:.3f}")
print(f"\nmean {sum(stds) / len(stds):.3f} — the gate switches over roughly this")
print("range. That scale is what makes beta=1 too soft and beta~2-3 right HERE.")

# %% [markdown]
"""
## Let the model pick `beta`

Swish was *defined* with `beta` learnable — that's the point of it. So stop guessing:
make it a parameter, give each block its own, and read off what the model chooses. Our
sweep says the optimum is north of SiLU's `1`; the model has never seen that sweep.
"""

# %%
torch.manual_seed(1337)
learned = train(GPT(vocab_size, lambda: Swish(1.0, learnable=True)).to(DEV))
betas = [b.ffwd.act.beta.item() for b in learned.blocks]
print("learned beta per block (all initialized at 1.0 = SiLU):")
for i, b in enumerate(betas):
    print(f"  block{i}: {b:.3f}")
print(f"\nmean learned beta: {sum(betas) / len(betas):.3f}")
print(f"val loss with learned beta: {full_val_loss(learned):.4f}"
      f"   (fixed SiLU {sum(results['SiLU (Swish b=1)']) / 3:.4f}, "
      f"GELU {gelu_mean:.4f})")
print("\nStarted at SiLU's beta=1 and every block walked the SAME direction — toward a")
print("sharper gate, exactly where the sweep said the loss was lower.")

# %% [markdown]
"""
**Every block moved, and every block moved the same way.** All four started at SiLU's
`beta = 1` and climbed toward `~2` — into the region the sweep independently found was
better, which the model had no way of knowing. When you let the network shape its own
activation, it asks for a *sharper gate than SiLU gives it*. That's the cleanest
confirmation of the mechanism we could ask for: two completely different experiments,
one answer.

It does **not** overtake fixed GELU, and it's worth being precise about why rather than
hiding it. `beta` starts at `1` and has to be *learned*, so this run spent most of its
3000 steps at a gate softer than GELU's fixed `1.702` — paying the curvature tax the
whole way, and still climbing when the clock ran out. It beats the fixed SiLU it started
from, in the predicted direction. That's the claim the experiment supports, and no more.
"""

# %% [markdown]
"""
## `beta` and the input scale are the same knob

There's a loose end in everything above, and pulling it tightens the whole argument.
We keep saying a gate is "sharp" or "soft" — but sharp *compared to what*? A gate that
switches over a width of `1/beta` is a hard step for inputs spread over `±10` and a
straight line for inputs spread over `±0.1`. So `beta` alone can't be the quantity that
matters. And indeed it isn't — there's an exact identity hiding in `x·sigmoid(beta*x)`:

```-
swish_beta(s*x)  =  s*x · sigmoid(beta*s*x)
                 =  s · [ x · sigmoid((beta*s)*x) ]
                 =  s · swish_(beta*s)(x)
```

**Scaling the input by `s` IS multiplying `beta` by `s`** — the two are the same
operation. (The leftover factor `s` on the outside is a scalar, and `09` proved the next
`Linear` absorbs one of those for free.) So neither `beta` nor the pre-activation scale
is separately meaningful. The only real quantity is the **product `beta*s`** — call it
the *effective sharpness*.
"""

# %%
xi = torch.randn(50_000) * 1.64                     # pre-activations at our scale
print("identity check:  swish_beta(s*x) == s * swish_(beta*s)(x)")
for s in (0.5, 2.0, 3.3):
    for b in (1.0, 1.702):
        lhs = Swish(b)(s * xi)
        rhs = s * Swish(b * s)(xi)
        print(f"  s={s:<4} beta={b:<6} -> max diff {(lhs - rhs).abs().max():.2e}")

# %% [markdown]
"""
### The sweep was an effective-sharpness sweep all along

That identity re-reads every number in this notebook. Our pre-activations sit at
`s ≈ 1.64`, so the `beta` sweep was never measuring `beta` — it was measuring `beta*s`:
"""

# %%
S_HERE = 1.64
print(f"{'beta':>6} | {'beta*s (effective)':>18} | {'val loss':>9}")
print("-" * 42)
for b in BETAS:
    v = sum(beta_res[b]) / len(beta_res[b])
    tag = {1.0: "  <- SiLU", 1.702: "  <- ~GELU"}.get(b, "")
    print(f"{b:>6} | {b * S_HERE:>18.2f} | {v:>9.4f}{tag}")
best = min(BETAS, key=lambda b: sum(beta_res[b]) / len(beta_res[b]))
print(f"\nbest effective sharpness here: beta*s = {best * S_HERE:.2f}")
print(f"SiLU (beta=1) would hit that if the pre-activations sat at "
      f"s = {best * S_HERE:.2f} instead of {S_HERE}.")

# %% [markdown]
"""
So the honest statement of this notebook's surprise is **not** "SiLU is worse than
GELU". It's:

> `beta*s = 1.0 x 1.64 = 1.64` is too soft for this model. The loss wants
> `beta*s ≈ 3-5`.

Nothing about SiLU is defective — it's *mismatched to the scale our init and our
LayerNorm happen to produce*. Give this model pre-activations at `s ≈ 3.3` and SiLU
would sit exactly on the optimum, and GELU would be the one overshooting. That's the
real content of the "Llama runs SiLU and is fine" caveat: at their scale, `beta=1` lands
somewhere else on this curve.

And note what the identity buys us: we get that conclusion **for free, without training
anything**. "Re-run the sweep with the pre-activations doubled" sounds like a good
experiment, but by the identity it is the *bit-identical computation* to the `beta`
sweep we already ran — same graph, same numbers. It's a corollary, not an experiment.
"""

# %% [markdown]
"""
### A hypothesis, and its death

Here's a tempting idea. If `beta*s` is what matters, and each block gets its own
learnable `beta`, then maybe every block tunes itself to the *same* effective sharpness
— the model's preferred operating point — and the varying `beta`s are just each block
compensating for its own `s`.

It's testable in one line: measure `s` and `beta` **in the same model** and multiply.
(The same-model part is the whole test. Taking `s` from one run and `beta` from another
would be comparing two different networks and calling it a pattern.)
"""


# %%
@torch.no_grad()
def preact_std_avg(m, batches=8):
    """Per-block pre-activation std, averaged over several val batches. Hooks fire in
    block order, so stats[i::n_blocks] are the readings for block i."""
    stats, handles = [], []
    for b in m.blocks:
        handles.append(b.ffwd.fc.register_forward_hook(
            lambda mod, i, o: stats.append(o.std().item())))
    for _ in range(batches):
        m(get_batch("val")[0])
    for h in handles:
        h.remove()
    nb = len(m.blocks)
    return [sum(stats[i::nb]) / len(stats[i::nb]) for i in range(nb)]


learned.eval()
s_learned = preact_std_avg(learned)              # same model the betas came from
prods = [b * s for b, s in zip(betas, s_learned)]
print(f"{'block':<6} | {'s (preact std)':>14} | {'learned beta':>12} | {'beta*s':>7}")
print("-" * 50)
for i, (s, b, p) in enumerate(zip(s_learned, betas, prods)):
    print(f"{i:<6} | {s:>14.3f} | {b:>12.3f} | {p:>7.3f}")
spread = lambda v: max(v) - min(v)
print(f"\nbeta alone : spread {spread(betas):.3f}  (nearly constant across blocks)")
print(f"beta*s     : spread {spread(prods):.3f}  (varies several-fold)")

# %% [markdown]
"""
**The hypothesis is dead.** It's the opposite of what happens: `beta*s` is *not*
constant — it falls steeply with depth — while **`beta` alone is nearly constant**. The
blocks are not each tuning to a shared operating point. If anything, `beta` barely moves
and `s` does all the varying.

Two things worth taking from the wreckage.

**First, the earlier "pattern" was an artifact, and this is why the same-model rule
matters.** Multiply the per-block `beta`s by the *GELU probe's* `s` values (which grow
with depth) and you get a suspiciously flat-looking product — a pattern assembled from
two different networks. Measured inside one model, where `s` actually *falls* with
depth, it evaporates. Same arithmetic, two sources, opposite conclusions.

**Second, the deepest block runs its FFN nearly linear** — `beta*s ≈ 1`, down at the
soft end of the sweep where we measured the loss to be *worst*. And the model chose that
freely: `s` is set by `fc`'s learned weights, so any block could have scaled itself into
the sharp regime. The last one declined. We measured what the optimum is *when every
block shares one `beta`*; it does not follow that every block wants it, and this says
they don't.

Which is a good note to leave the notebook on. `beta*s` is the right quantity — that
part is an identity and it's solid. "Every block wants the same `beta*s`" was a story,
and the measurement refused it.
"""

# %% [markdown]
"""
## What you built, and where `11` goes next

The line `06` waved at is the line the FeedForward is built around:

- **two stacked `Linear`s are one `Linear`** — `W2(W1x+b1)+b2 = (W2W1)x + (W2b1+b2)`,
  proved by fusion to ~0, and it telescopes to any depth. So without an activation
  `06`'s `C → 4C → C` FFN collapses to a single `C → C` matrix: **131,712 parameters
  expressing 16,512 parameters' worth of function**, and the `4x` expansion buys
  nothing. Measured on the scoreboard, deleting that one function call is the most
  expensive thing we've done to this model.
- **the zoo is one knob.** `Swish(x) = x · sigmoid(beta*x)` contains all of it: `beta→0`
  is a straight line, `beta=1` **is** SiLU, `beta≈1.702` reproduces GELU to 0.02,
  `beta→∞` is ReLU. Verified numerically at every endpoint.
- **so "which activation" is really "how sharp".** Sweeping `beta` reproduced the whole
  A/B table, including the surprise that SiLU underperforms GELU here — because
  `x·sigmoid(beta*x) ≈ 0.5x + (beta/4)x²`, the curvature *is* `beta`, and SiLU is the
  softer gate. Saturating activations (sigmoid, tanh) lose for the other reason: their
  derivative flatlines and the gradient dies.
- **and the model agrees.** Made learnable from SiLU's `beta=1`, all four blocks climbed
  toward `~2` — into the region the sweep found better, which the model never saw. It
  beats the fixed SiLU it started from; it doesn't catch fixed GELU, because `beta`
  spent most of training still climbing.
- **`beta` alone was never the real quantity.** `swish_beta(s*x) = s*swish_(beta*s)(x)`
  is an exact identity, so scaling the input *is* scaling `beta` — only the product
  `beta*s` means anything. The sweep was an effective-sharpness sweep in disguise, and
  the honest form of the surprise is "`beta*s = 1.64` is too soft **here**", not "SiLU
  is worse": at `s ≈ 3.3` SiLU would *be* the optimum. That follows from the identity
  with no extra training — which is also why "re-run the sweep at double scale" isn't an
  experiment, it's the same computation.
- **and one story died on contact.** "Each block tunes `beta` to hit a shared `beta*s`"
  is a lovely hypothesis; measured inside a single model it's false — `beta` is nearly
  constant across blocks while `beta*s` falls several-fold with depth, and the deepest
  block chooses to run its FFN nearly *linear*. The version that looked true came from
  multiplying one model's `beta` by another model's `s`.
- **two failures, two mechanisms.** Saturating activations (sigmoid, tanh) lose on
  *gradient flow* — sigmoid's derivative can't exceed `0.25`, so it divides the gradient
  by `4x` at every layer before saturation is even mentioned. SiLU loses to GELU on
  *curvature*, which gradient flow gets backwards (ReLU passes less gradient than SiLU
  and wins anyway). Don't reach for the vanishing-gradient story when the answer is
  sharpness.

One honest caveat before `11`: none of this makes GELU "better than SiLU" in general.
The right sharpness depends on the pre-activation scale your init and norms produce, and
ours is a 128-wide char model. Llama runs SiLU at scale and is fine.

Which sets up the real question. Look at what Swish actually is — a value `x`,
multiplied by a **gate** `sigmoid(beta*x)` that decides how much of it gets through:

```-
Swish(x) = x · sigmoid(beta * x)
           ^         ^
           value     gate — a FIXED function of the value itself
```

The gate can only ever look at `x`. Its sharpness is learnable, but *what it looks at*
isn't. `11` asks the obvious next question: **what if the gate were its own learned
projection?** Let one matrix compute the value and a *different* matrix compute the
gate:

```-
GLU(x)    = (W·x + b) · sigmoid(V·x + c)      # gate = a separate learned projection
SwiGLU(x) = (W·x + b) · Swish(V·x + c)        # ...with Swish instead of sigmoid
```

That's **SwiGLU**, and it's what Llama, PaLM and Mistral run instead of `06`'s FFN. It
costs a third matrix — so the honest comparison has to hold parameters equal, which is
where the famous `8C/3` hidden size comes from. And it buys something a fixed gate
provably cannot: a **product of two learned functions of the input**, which is to say
`x²`-shaped terms — genuine multiplicative interaction. `11` builds it, sizes it fairly,
and measures whether the gate earns its matrix.
"""
