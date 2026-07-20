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
# Diffusion · 01 — the whole game

Top-down: before we derive a single equation, let's build a **real diffusion model, train it, and
watch it invent digits out of pure static**. By the end of this notebook you have a working MNIST
generator and a rough mental map of its three moving parts. *Why* each part is shaped the way it is —
that's `02` onward, each opening one box of *this exact pipeline*.

The three ideas, in one breath:

1. **forward** — take a clean digit and add Gaussian noise on a schedule until it's pure static.
2. **model** — train a small network to look at a noisy image (and *how* noisy it is) and predict the
   noise that was added.
3. **reverse** — start from pure static and use that network to strip a little noise at a time, step
   by step, until a clean digit is left. That digit was never in the training set — the model made
   it up.

> This is the map. Every later notebook zooms into **one box you already ran here.**
"""

# %%
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def _repo_root() -> Path:
    """Walk up from the cwd to the folder holding pyproject.toml, so paths work no matter where
    the kernel is launched from."""
    here = Path.cwd()
    for d in (here, *here.parents):
        if (d / "pyproject.toml").exists():
            return d
    return here


ROOT = _repo_root()
DATA = ROOT / "nb" / "data" / "mnist.npz"          # shared MNIST cache (no torchvision)
CKPT_DIR = ROOT / "nb" / "diffusion" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEV}")


def load_mnist(train=True):
    """MNIST images (N,1,28,28) in [-1,1], from the cached npz. No labels (01 is unconditional:
    it generates *some* digit, not a chosen one — class conditioning is 09). No torchvision."""
    if not DATA.exists():                                  # self-contained: fetch on first use
        DATA.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        print(f"  downloading MNIST npz (~11MB) -> {DATA.relative_to(ROOT)} ...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", DATA)
    d = np.load(DATA)
    x = d["x_train"] if train else d["x_test"]            # (N,28,28) uint8 [0,255]
    return (torch.from_numpy(x).float() / 127.5 - 1.0).unsqueeze(1)   # (N,1,28,28) in [-1,1]


def to_img(x):
    """(1,28,28)-ish tensor in [-1,1] -> HxW numpy in [0,1] for imshow."""
    return ((x.squeeze().clamp(-1, 1) + 1) / 2).cpu().numpy()


# %% [markdown]
"""
## The pieces in one breath

Rough narration, no rigor yet — just enough that none of it is a black box:

- **the schedule** (`make_schedule`) — `T` noise levels. `alpha_bar[t]` is *how much of the original
  signal survives* to level `t` (≈1 at `t=0`, ≈0 at `t=T`). It lets us jump straight to any noise
  level in one shot: `x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε`. (opened in `02`)
- **the model** (`TinyUNet`) — takes a noisy image `x_t` **and** the timestep `t`, and predicts the
  noise `ε` inside it. It's a U-Net: downsample `28→14→7` (see globally), upsample back with **skip
  connections** (keep the fine detail). The timestep is fed in so it knows *how* noisy the input is.
  (target opened in `03`, architecture in `04`)
- **sampling** (`sample`) — start at pure noise `x_T ~ N(0,I)` and walk backward `T→0`, subtracting a
  little predicted noise each step, until a clean digit remains. (opened in `05`)

*Why* predict the noise and not the image? Why a U-Net? Why walk back step by step instead of one
shot? All deferred — `02`–`05` open each box in turn.
"""

# %%
def make_schedule(T=1000, device="cpu"):
    """Linear DDPM schedule. Returns betas, alphas, alpha_bar (each shape (T,))."""
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)          # how much signal survives to step t
    return betas, alphas, alpha_bar


def timestep_embedding(t, dim):
    """Sinusoidal embedding of the integer timestep t (B,) -> (B, dim). Same trick as positional
    encodings in the llm/ track — turn a scalar into a smooth many-frequency vector."""
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _Block(nn.Module):
    """Residual block: (GroupNorm -> SiLU -> conv) twice, with the timestep added in the middle.
    GroupNorm + residual are the standard diffusion-U-Net minimum for stable training and crisp
    samples; the why is `04`."""

    def __init__(self, cin, cout, temb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(temb_dim, cout)                  # inject the timestep here
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]              # add "how noisy" into the features
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    """Predicts the noise ε from (x_t, t). Down 28->14->7 (channels grow), up 7->14->28 with skip
    connections; the timestep t is injected into every block. This is the CNN track's conv/downsample
    trunk, rewired for image->image instead of image->label."""

    def __init__(self, base=32, temb_dim=128):
        super().__init__()
        self.temb_dim = temb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim))
        self.stem = nn.Conv2d(1, base, 3, padding=1)                    # 1 -> base, 28x28
        self.down1 = _Block(base, base, temb_dim)                       # 28
        self.down2 = _Block(base, base * 2, temb_dim)                   # 14 (after pool)
        self.down3 = _Block(base * 2, base * 4, temb_dim)              # 7  (after pool)
        self.mid = _Block(base * 4, base * 4, temb_dim)                 # 7
        self.up2 = _Block(base * 4 + base * 2, base * 2, temb_dim)      # 14 (+down2 skip)
        self.up1 = _Block(base * 2 + base, base, temb_dim)             # 28 (+down1 skip)
        self.out_norm = nn.GroupNorm(8, base)
        self.out = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x, t):
        temb = self.time_mlp(timestep_embedding(t, self.temb_dim))
        h1 = self.down1(self.stem(x), temb)                            # (B, base,  28, 28)
        h2 = self.down2(F.avg_pool2d(h1, 2), temb)                     # (B, 2base, 14, 14)
        h3 = self.down3(F.avg_pool2d(h2, 2), temb)                     # (B, 4base,  7,  7)
        m = self.mid(h3, temb)
        u = F.interpolate(m, scale_factor=2, mode="nearest")           # 7 -> 14
        u = self.up2(torch.cat([u, h2], 1), temb)
        u = F.interpolate(u, scale_factor=2, mode="nearest")           # 14 -> 28
        u = self.up1(torch.cat([u, h1], 1), temb)
        return self.out(F.silu(self.out_norm(u)))                      # (B,1,28,28) predicted ε


@torch.no_grad()
def sample(model, n, T, betas, alphas, alpha_bar, device):
    """Ancestral DDPM sampling: start from pure noise, walk back T -> 0, one denoise step at a time."""
    model.eval()
    x = torch.randn(n, 1, 28, 28, device=device)
    for i in reversed(range(T)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        eps_hat = model(x, t)
        a, ab, b = alphas[i], alpha_bar[i], betas[i]
        mean = (x - (1 - a) / (1 - ab).sqrt() * eps_hat) / a.sqrt()    # DDPM posterior mean
        if i > 0:
            var = b * (1 - alpha_bar[i - 1]) / (1 - ab)                # posterior variance
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = mean                                                   # last step: no added noise
    return x


torch.manual_seed(0)
model = TinyUNet().to(DEV)
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params:,} total")

# %% [markdown]
"""
## Watch it learn

The training loop *is* the forward process plus one MSE line. For each batch:

1. pick a **random noise level `t`** per image (one batch spans easy and hard levels),
2. draw noise `ε` and jump straight to `x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε` (the forward process, one line),
3. ask the model to predict `ε` from `(x_t, t)` and minimise `MSE(ε̂, ε)`.

That's the whole objective. Loss starts near **1.0** (predicting all-zeros scores `Var(ε)=1`) and
falls as the model learns to see the noise. We also keep an **EMA** (exponential moving average) of
the weights and sample from *those* — averaged weights give noticeably cleaner digits for ~free (the
`08` box).

> On a GPU this is ~2–3 minutes; on CPU it's slow (the 1000-step sampling at the end especially).
> Drop `EPOCHS` for a quick smoke test — the digits get rougher but still recognisable.
"""

# %%
EPOCHS, BATCH, LR, T = 20, 128, 2e-4, 1000

x0 = load_mnist(train=True)                                            # (60000,1,28,28) in [-1,1]
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x0), batch_size=BATCH, shuffle=True)
betas, alphas, alpha_bar = make_schedule(T, device=DEV)

opt = torch.optim.Adam(model.parameters(), lr=LR)
ema = {k: v.detach().clone() for k, v in model.state_dict().items()}   # averaged weights for sampling

print(f"training on {len(x0)} images ({DEV}), {len(loader)} steps/epoch. loss = MSE(ε̂, ε):")
for ep in range(1, EPOCHS + 1):
    model.train()
    run = 0.0
    for (xb,) in loader:
        xb = xb.to(DEV)
        t = torch.randint(0, T, (xb.shape[0],), device=DEV)           # a random noise level per image
        eps = torch.randn_like(xb)
        ab = alpha_bar[t].view(-1, 1, 1, 1)
        xt = ab.sqrt() * xb + (1 - ab).sqrt() * eps                    # forward: noise the batch
        loss = F.mse_loss(model(xt, t), eps)                           # predict the noise
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():                                         # nudge the EMA copy
            for k, v in model.state_dict().items():
                if ema[k].is_floating_point():
                    ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
                else:
                    ema[k].copy_(v)
        run += loss.item()
    print(f"  epoch {ep:>2}: train loss {run / len(loader):.4f}")

# %% [markdown]
"""
Loss slides from ~1.0 down toward ~0.03–0.04. A low ε-loss means: *given a noisy image and its noise
level, the model can pick out the noise* — and if you can see the noise, you can subtract it. We save
the EMA weights (and `T`) so `02`–`16` can load this exact model instead of retraining.
"""

# %%
ckpt = CKPT_DIR / "ddpm.pt"
torch.save({"ema": ema, "T": T, "base": 32, "temb_dim": 128}, ckpt)
print(f"saved trained weights -> {ckpt.relative_to(ROOT)}")

# %% [markdown]
"""
## The payoff — digits out of pure static

Sampling is the reverse process: start from a `64`-image batch of **pure Gaussian noise** and let the
trained model denoise it step by step, `T → 0`. What comes out are digits the model *invented* — not
copies of training images. (This runs the network `T=1000` times, so it's the slow part.)
"""

# %%
model.load_state_dict(ema)                                             # sample from the averaged weights
print("sampling 64 digits from pure noise (reverse process, T steps)...")
imgs = sample(model, 64, T, betas, alphas, alpha_bar, DEV)

rows, cols = 8, 8
fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.9, rows * 0.9))
for ax, k in zip(axes.flat, range(rows * cols)):
    ax.imshow(to_img(imgs[k]), cmap="gray")
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("generated MNIST digits — sampled from pure noise", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()
print("64 digits the model invented from static. That's the whole game.")

# %% [markdown]
"""
## The map — what we open next

You now have a working generator and a rough picture. Every notebook below picks **one box you just
ran** and explains why it's built that way — measured, not asserted:

| next | opens | the question |
|---|---|---|
| `02` | `make_schedule` + the forward line | how a digit dissolves into noise; why the `√` coefficients |
| `03` | `MSE(ε̂, ε)` | why predict the *noise*, not the image? (+ the wiring checks) |
| `04` | `TinyUNet` | why a U-Net (down/up + skips), and why the timestep must be an input |
| `05` | `sample` | why generation is *iterative* — one-shot from noise is mush |
| `06`+ | faster sampling, conditioning, scale | DDIM, CFG, latent, DiT, flow matching |

Next: **`02` — the forward process: how we add noise, and why `√ᾱ·x0 + √(1-ᾱ)·ε`.**
"""
