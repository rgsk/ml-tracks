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
# Diffusion · 03 — the training target

In `01` the entire objective was one line inside the training loop:

```-
loss = F.mse_loss(model(xt, t), eps)      # predict the noise ε
```

The net looks at a noisy image `x_t` (and its level `t`) and predicts the **noise ε** that was added —
not the clean image `x_0`. This notebook opens that choice. Three questions:

1. **Isn't predicting `ε` and predicting `x_0` the same thing?** Given `x_t`, yes — they're a fixed,
   invertible map apart. So the choice is really about *how the loss weights the noise levels*.
2. **Then why `ε`?** Because `ε ~ N(0, I)` at **every** `t` — a target of constant scale — so one
   plain `MSE` already balances all levels, with no per-`t` reweighting. Predicting `x_0` under plain
   `MSE` quietly over-weights the near-hopeless high-noise levels and gives blurrier samples.
3. **Is the loop even wired right?** Two cheap checks — borrowed from the CNN track — that you run
   *before* spending three minutes: untrained loss ≈ `1`, and overfit-one-batch collapses.

Still no U-Net internals here (that's `04`); the model is a black box we feed a target to.
"""

# %%
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def _repo_root() -> Path:
    here = Path.cwd()
    for d in (here, *here.parents):
        if (d / "pyproject.toml").exists():
            return d
    return here


ROOT = _repo_root()
DATA = ROOT / "nb" / "data" / "mnist.npz"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_mnist(train=True):
    """MNIST images (N,1,28,28) in [-1,1], from the cached npz. No torchvision."""
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        print(f"  downloading MNIST npz (~11MB) -> {DATA.relative_to(ROOT)} ...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", DATA)
    d = np.load(DATA)
    x = d["x_train"] if train else d["x_test"]
    return (torch.from_numpy(x).float() / 127.5 - 1.0).unsqueeze(1)   # (N,1,28,28) in [-1,1]


def to_img(x):
    return ((x.squeeze().clamp(-1, 1) + 1) / 2).cpu().numpy()


def make_schedule(T=1000, device="cpu"):
    """Linear DDPM schedule (same as 01/02). Returns betas, alphas, alpha_bar."""
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


def timestep_embedding(t, dim):
    """Sinusoidal embedding of the integer timestep (B,) -> (B, dim) — the llm/ positional trick."""
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


T = 1000
betas, alphas, alpha_bar = make_schedule(T)

# %% [markdown]
"""
## Two views of the same target

The forward line from `02` ties `x_0`, `ε`, and `x_t` together:

```-
x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε
```

Solve it for each unknown — **given `x_t` and `t`**, either one determines the other:

```-
ε   = (x_t − √ᾱ_t·x_0) / √(1-ᾱ_t)
x_0 = (x_t − √(1-ᾱ_t)·ε) / √ᾱ_t
```

So a network that sees `x_t` and outputs `ε` is, up to a fixed invertible linear map, the *same*
network that outputs `x_0`. **Predicting the noise and predicting the image carry identical
information.** Let's confirm the round-trip is exact, then see where the two choices actually differ.
"""

# %%
x = load_mnist(train=False)
torch.manual_seed(0)
x0 = x[:256]                                          # a batch of real held-out digits

print("  given x_t, recover each view from the other (round-trip error), and compare losses:")
for t in (50, 200, 500, 900):
    ab = alpha_bar[t]
    eps = torch.randn_like(x0)
    xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps       # the noisy image the net actually sees

    # round-trip: rebuild x0 from the true ε, and ε from the true x0 — should be exact
    x0_from_eps = (xt - (1 - ab).sqrt() * eps) / ab.sqrt()
    eps_from_x0 = (xt - ab.sqrt() * x0) / (1 - ab).sqrt()
    rt = max((x0_from_eps - x0).abs().max(), (eps_from_x0 - eps).abs().max()).item()

    # loss identity: pretend the net is a bit off on x0; map that same error into ε-space
    x0_hat = x0 + 0.1 * torch.randn_like(x0)
    eps_hat = (xt - ab.sqrt() * x0_hat) / (1 - ab).sqrt()
    loss_x0 = ((x0 - x0_hat) ** 2).mean().item()
    loss_eps = ((eps - eps_hat) ** 2).mean().item()
    snr = (ab / (1 - ab)).item()                      # signal-to-noise ratio ᾱ/(1-ᾱ)
    print(f"  t={t:>3} | round-trip err={rt:.1e} | ε-loss/x0-loss={loss_eps/loss_x0:7.2f}"
          f"  vs  SNR=ᾱ/(1-ᾱ)={snr:7.2f}")

# %% [markdown]
"""
The round-trip error is machine-epsilon — the two views really are the same target. But look at the
last two columns: the **ε-loss is exactly `SNR = ᾱ_t/(1-ᾱ_t)` times the `x_0`-loss**. That's not a
coincidence; it drops straight out of the map above:

```-
DERIVATION — how a prediction error in ε maps into x0-space (same net, x_t & t held fixed):
  x0(ε) = (x_t − √(1-ᾱ)·ε) / √ᾱ                            (the x0 view above)
  perturb the truth ε → ε̂ = ε + δ, plug into the same formula, subtract:
    x̂0 = (x_t − √(1-ᾱ)·(ε+δ)) / √ᾱ ,   x0 = (x_t − √(1-ᾱ)·ε) / √ᾱ
    x̂0 − x0 = [ −√(1-ᾱ)·δ ] / √ᾱ = −(√(1-ᾱ)/√ᾱ) · δ         (x_t cancels; linear in the error δ)
  square and average:
    MSE_x0 = ((1-ᾱ)/ᾱ) · MSE_ε = MSE_ε / SNR(t)            SNR(t) = ᾱ/(1-ᾱ)   (from 02)
  ⇔  MSE_ε = SNR(t) · MSE_x0
```

So **training on `ε` with plain `MSE` is identical to training on `x_0` with an `SNR`-weighted `MSE`.**
The choice of target is secretly a choice of *per-`t` loss weight*. The next cell shows why the `ε`
weighting is the good one.
"""

# %%
ts = torch.arange(1, T)
snr = alpha_bar[ts] / (1 - alpha_bar[ts])

# empirical check that the ε-target is unit-scale at *every* t (it's drawn N(0,1), t-independent)
torch.manual_seed(0)
probe_t = torch.arange(0, T, 50)
eps_var = [torch.randn(20000).var().item() for _ in probe_t]     # Var(ε) ≈ 1, flat in t
print(f"  Var(ε-target) across t: {min(eps_var):.3f} .. {max(eps_var):.3f}  (flat ≈ 1)")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].plot(alpha_bar.sqrt().numpy(), label="√ᾱ_t  (signal in x_t — what an x0-net recovers)")
ax[0].plot((1 - alpha_bar).sqrt().numpy(), label="√(1-ᾱ_t)  (noise in x_t — what an ε-net recovers)")
ax[0].axhline(1.0, color="gray", ls=":", lw=1)
ax[0].set_title("the ε-target keeps unit scale; the signal fades"); ax[0].set_xlabel("t"); ax[0].legend()
ax[1].semilogy(ts.numpy(), snr.numpy(), color="C3")
ax[1].set_title("SNR = ᾱ/(1-ᾱ): the weight ε-loss puts on the x0 error")
ax[1].set_xlabel("t"); ax[1].set_ylabel("weight (log)")
fig.tight_layout(); plt.show()

# %% [markdown]
"""
## Why `ε`: a target that never changes scale

Two things the plot makes concrete:

- **`ε` is scale-stable.** The regression target has `Var(ε) ≈ 1` at *every* noise level, so a single
  unweighted `MSE` already treats all `t` on equal footing — no per-`t` normalisation to tune. This is
  also why sampling can start from `N(0,I)` (`02`) and why the *untrained* loss is ≈ 1 (next section):
  a net that outputs `≈ 0` scores `E‖ε‖² = 1`.
- **The implied `x_0` weighting is the sane one.** `SNR = ᾱ/(1-ᾱ)` is huge at low `t` (where `x_0` is
  easily recoverable and worth nailing) and `→ 0` at high `t` (where `x_t` is almost pure noise and
  `x_0` is unrecoverable). Training on `x_0` with *plain* `MSE` throws equal weight at those hopeless
  high-noise levels — capacity spent chasing detail it can't recover, which shows up as blurrier
  samples. Predicting `ε` bakes in the down-weighting for free.

So `ε`-prediction is the DDPM default not because `x_0` is *wrong*, but because plain `MSE` on `ε`
happens to be the well-weighted objective. (`07` revisits this: `v`-prediction is a *third* view —
a scale-balanced mix of `x_0` and `ε` — that's even steadier at low step counts.)

## Two wiring checks — run these *before* you wait

The objective is `MSE(ε̂, ε)` and nothing more. Before launching `01`'s three-minute run, two cheap
checks — straight from the CNN track — tell you the loop is actually hooked up. We use a small
throwaway denoiser (the real architecture is `04`; treat this as a black box).
"""

# %%
class TimeNet(nn.Module):
    """A small black-box denoiser — just enough net to exercise the loss wiring. Down 28→14→7, up
    with skips, timestep injected. The real U-Net (and *why* it's shaped this way) is `04`."""

    def __init__(self, base=32, temb_dim=64):
        super().__init__()
        self.temb_dim = temb_dim
        self.time = nn.Sequential(
            nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim))
        self.t_down, self.t_mid, self.t_up = (nn.Linear(temb_dim, c) for c in (base, base * 2, base))
        self.stem = nn.Conv2d(1, base, 3, padding=1)
        self.down = nn.Conv2d(base, base, 3, padding=1)
        self.mid = nn.Conv2d(base, base * 2, 3, padding=1)
        self.up = nn.Conv2d(base * 2 + base, base, 3, padding=1)      # + skip from stem
        self.out = nn.Conv2d(base, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # start at ε̂ ≈ 0

    def forward(self, x, t):
        temb = self.time(timestep_embedding(t, self.temb_dim))
        h1 = F.silu(self.stem(x) + self.t_down(temb)[:, :, None, None])          # 28
        h = F.silu(self.down(F.avg_pool2d(h1, 2)))                               # 14
        h = F.silu(self.mid(F.avg_pool2d(h, 2)) + self.t_mid(temb)[:, :, None, None])  # 7
        h = F.interpolate(h, size=28, mode="nearest")                            # 7 -> 28
        h = F.silu(self.up(torch.cat([h, h1], 1)) + self.t_up(temb)[:, :, None, None])
        return self.out(h)


torch.manual_seed(0)
net = TimeNet()
print(f"  stand-in denoiser: {sum(p.numel() for p in net.parameters()):,} params")

# check 1 — untrained loss ≈ 1
xb = x[:64]
t = torch.randint(0, T, (64,))
eps = torch.randn_like(xb)
ab = alpha_bar[t].view(-1, 1, 1, 1)
xt = ab.sqrt() * xb + (1 - ab).sqrt() * eps
with torch.no_grad():
    loss0 = F.mse_loss(net(xt, t), eps).item()
print(f"  check 1 — untrained MSE(ε̂, ε) = {loss0:.3f}   (want ≈ 1.0 = Var(ε); ε̂≈0 by zero-init)")

# %% [markdown]
"""
`≈ 1.0` confirms two things at once: the target has unit variance, and the net starts by predicting
`≈ 0` (we zero-init the output layer — standard in diffusion). A number far from `1` means something
is mis-scaled (wrong data range, missing `√(1-ᾱ)`, a stray normalisation) — catch it *now*, not after
20 epochs.

Now the second check: can the loop drive the loss **down**? Freeze one tiny batch and train only on
it. If gradients flow and the target is learnable, the loss collapses — proof the whole path
(forward line → model → `MSE` → `backward` → `opt.step`) is connected.
"""

# %%
torch.manual_seed(0)
net = TimeNet()
opt = torch.optim.Adam(net.parameters(), lr=2e-3)

xb = x[:16]                                           # one small, *fixed* batch
t = torch.randint(0, T, (16,))
eps = torch.randn_like(xb)
ab = alpha_bar[t].view(-1, 1, 1, 1)
xt = ab.sqrt() * xb + (1 - ab).sqrt() * eps           # freeze (x_t, t, ε) — memorise this exact map

losses = []
for step in range(600):
    loss = F.mse_loss(net(xt, t), eps)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
print(f"  check 2 — overfit one batch: loss {losses[0]:.3f} -> {losses[-1]:.4f} over {len(losses)} steps")

with torch.no_grad():
    eps_hat = net(xt, t)
k = 0                                                 # eyeball one image's predicted vs true noise
fig, ax = plt.subplots(1, 3, figsize=(8.5, 3.1))
ax[0].semilogy(losses); ax[0].set_title("overfit-one-batch loss (log)"); ax[0].set_xlabel("step")
ax[1].imshow(eps[k, 0], cmap="gray"); ax[1].set_title("true ε"); ax[1].axis("off")
ax[2].imshow(eps_hat[k, 0], cmap="gray"); ax[2].set_title("predicted ε̂"); ax[2].axis("off")
fig.tight_layout(); plt.show()
print("  loss collapses and ε̂ matches ε — the training loop is wired. Now 01's full run is worth it.")

# %% [markdown]
"""
## Recap

- Given `x_t`, predicting the **noise `ε`** and predicting the **image `x_0`** are the *same target*,
  one invertible linear map apart — we verified the round-trip is exact.
- The difference is the **implied loss weighting**: plain `MSE` on `ε` equals `SNR = ᾱ/(1-ᾱ)`-weighted
  `MSE` on `x_0`. That weighting (big at low noise, `→0` at high noise) is the sane one, and `ε` is
  **scale-stable** (`Var(ε) ≈ 1` at every `t`), so one unweighted `MSE` needs no per-`t` tuning.
- Two **wiring checks** guard the three-minute run: **untrained loss ≈ 1** (target is unit-scale, net
  outputs `≈ 0`) and **overfit-one-batch collapses** (gradients flow, target is learnable).

Next: **`04` — the denoiser: why a U-Net (down/up + skip connections), and why the timestep `t` has to
be an input** — we open the black box we just trained.
"""
