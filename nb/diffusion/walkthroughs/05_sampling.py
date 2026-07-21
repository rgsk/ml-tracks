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
# Diffusion · 05 — sampling: the reverse process

`01`–`04` built and understood the *forward* half (noise a digit) and the *model* (predict that
noise). This notebook opens the last box: **generation** — the `sample` loop that walks pure static
back into a digit. It closes Phase A: after this you understand the whole working generator.

The model gives us `ε̂`, and the forward identity (`02`) turns any `ε̂` into an estimate of the clean
image: `x̂_0 = (x_t − √(1-ᾱ_t)·ε̂) / √ᾱ_t`. So the obvious idea is: start from noise, predict `ε̂`
**once**, jump straight to `x̂_0`. Three questions:

1. **Why doesn't that one-shot jump work?** We'll do it and *watch* it — from pure noise it's a blurry
   grey blob. The reason: at high noise `x̂_0` is the **average of every digit that noise could become**.
2. **What does the iterative loop do instead?** It takes one *small* step toward `x̂_0`, re-adds a
   little noise, and re-predicts — the **reverse posterior** `q(x_{t-1}|x_t, x_0)`. Each step resolves
   a bit more, so the average collapses onto *one* committed digit.
3. **Why re-add noise every step?** Because we're **sampling** from a distribution, not maximising a
   mean. Drop the noise and the trajectory drifts back toward the same over-smoothed average.
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
CKPT = ROOT / "nb" / "diffusion" / "checkpoints" / "ddpm.pt"
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
    """Linear DDPM schedule (same as 01–04). Returns betas, alphas, alpha_bar."""
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
betas, alphas, alpha_bar = make_schedule(T, device=DEV)

# %% [markdown]
"""
## The model, verbatim from `01`

Same `TinyUNet` you trained — copied so this notebook stands alone (the *why* of its shape was `04`).
We load `01`'s EMA weights, or quick-train a rough stand-in if the checkpoint is missing.
"""

# %%
class _Block(nn.Module):
    def __init__(self, cin, cout, temb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(temb_dim, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    def __init__(self, base=32, temb_dim=128):
        super().__init__()
        self.temb_dim = temb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim))
        self.stem = nn.Conv2d(1, base, 3, padding=1)
        self.down1 = _Block(base, base, temb_dim)
        self.down2 = _Block(base, base * 2, temb_dim)
        self.down3 = _Block(base * 2, base * 4, temb_dim)
        self.mid = _Block(base * 4, base * 4, temb_dim)
        self.up2 = _Block(base * 4 + base * 2, base * 2, temb_dim)
        self.up1 = _Block(base * 2 + base, base, temb_dim)
        self.out_norm = nn.GroupNorm(8, base)
        self.out = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x, t):
        temb = self.time_mlp(timestep_embedding(t, self.temb_dim))
        h1 = self.down1(self.stem(x), temb)
        h2 = self.down2(F.avg_pool2d(h1, 2), temb)
        h3 = self.down3(F.avg_pool2d(h2, 2), temb)
        m = self.mid(h3, temb)
        u = F.interpolate(m, scale_factor=2, mode="nearest")
        u = self.up2(torch.cat([u, h2], 1), temb)
        u = F.interpolate(u, scale_factor=2, mode="nearest")
        u = self.up1(torch.cat([u, h1], 1), temb)
        return self.out(F.silu(self.out_norm(u)))


def load_or_train():
    model = TinyUNet(base=32, temb_dim=128).to(DEV)
    if CKPT.exists():
        model.load_state_dict(torch.load(CKPT, map_location=DEV)["ema"])
        print(f"  loaded 01's trained EMA weights <- {CKPT.relative_to(ROOT)}")
    else:
        print("  no checkpoint — quick-training a rough stand-in (~600 steps)...")
        x0 = load_mnist(True).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=2e-4)
        for _ in range(600):
            xb = x0[torch.randint(0, len(x0), (128,), device=DEV)]
            tt = torch.randint(0, T, (128,), device=DEV)
            eps = torch.randn_like(xb)
            ab = alpha_bar[tt].view(-1, 1, 1, 1)
            loss = F.mse_loss(model(ab.sqrt() * xb + (1 - ab).sqrt() * eps, tt), eps)
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


model = load_or_train()


def predict_x0(x_t, t_int):
    """The net's estimate of the clean image from x_t: x̂_0 = (x_t − √(1-ᾱ)·ε̂)/√ᾱ  (the 02 identity)."""
    t = torch.full((x_t.shape[0],), t_int, device=DEV, dtype=torch.long)
    with torch.no_grad():
        eps_hat = model(x_t, t)
    ab = alpha_bar[t_int]
    return (x_t - (1 - ab).sqrt() * eps_hat) / ab.sqrt()

# %% [markdown]
"""
## 1 · Why one shot fails: `x̂_0` is an *average* of digits

Take real digits, noise them to a level `t`, and ask the model for its one-shot `x̂_0`. At low `t`
(barely noised) the recovery is crisp. As `t` climbs, the noisy image stops pinning down *which* digit
it was, so the best guess `x̂_0` becomes the **conditional mean** `E[x_0 | x_t]` — a blur of every digit
that noise is compatible with. At `t≈T` (pure static) that's the average of *all* digits: grey mush.
"""

# %%
x0 = load_mnist(train=False).to(DEV)[:8]              # 8 real held-out digits
torch.manual_seed(0)
show_t = [50, 200, 500, 800, 999]

fig, axes = plt.subplots(3, len(show_t), figsize=(len(show_t) * 1.5, 4.4))
print("  one-shot x̂_0 error vs the true digit, as a function of the noise level it's launched from:")
for c, t in enumerate(show_t):
    ab = alpha_bar[t]
    eps = torch.randn_like(x0)
    xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    x0_hat = predict_x0(xt, t)
    mse = F.mse_loss(x0_hat.clamp(-1, 1), x0).item()
    print(f"    t={t:>3} (√ᾱ={ab.sqrt():.2f}): one-shot MSE(x̂_0, x_0) = {mse:.3f}")
    for r, im, lab in [(0, xt[0], "x_t (input)"), (1, x0_hat[0], "x̂_0 (1 shot)"), (2, x0[0], "true x_0")]:
        axes[r, c].imshow(to_img(im), cmap="gray"); axes[r, c].axis("off")
        if c == 0:
            axes[r, c].set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9)
        if r == 0:
            axes[r, c].set_title(f"t={t}", fontsize=9)
fig.suptitle("one-shot denoise: crisp at low noise, a blurry digit-average at high noise", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95)); plt.show()
print("  from near-pure-noise the single best guess is the MEAN of all digits — that's why 1 shot mushes.")

# %% [markdown]
"""
The error climbs steeply with `t`, and the images tell you *why*: the model isn't wrong, it's
**hedging**. Given ambiguous input it outputs the average of all consistent completions, which is the
mathematically-best single guess under MSE but a lousy *sample*. To get one sharp digit we must not
force a decision from pure noise — we let the model commit gradually. That's the reverse process.

## 2 · The reverse posterior: one small step back

Instead of jumping `x_t → x̂_0`, take one step `x_t → x_{t-1}`, only `1/T` of the way. The forward chain
is Gaussian, so the exact one-step reverse — *given* the true `x_0` — is Gaussian too, and Bayes gives
it in closed form:

```-
DERIVATION — the reverse posterior q(x_{t-1} | x_t, x_0)   (Bayes on two Gaussians):
  one forward step:      q(x_t     | x_{t-1}) = N(√α_t · x_{t-1},   β_t · I)
  forward from x_0:      q(x_{t-1} | x_0)     = N(√ᾱ_{t-1} · x_0,  (1-ᾱ_{t-1}) · I)
  Bayes:  q(x_{t-1}|x_t,x_0) ∝ q(x_t|x_{t-1})·q(x_{t-1}|x_0)   — product of Gaussians → Gaussian.
  collect the terms quadratic & linear in x_{t-1}, complete the square:
    variance   β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) · β_t
    mean       μ̃_t = (√ᾱ_{t-1}·β_t)/(1-ᾱ_t) · x_0  +  (√α_t·(1-ᾱ_{t-1}))/(1-ᾱ_t) · x_t
  at sample time we don't have x_0 — plug in the net's estimate x̂_0 = (x_t − √(1-ᾱ_t)·ε̂)/√ᾱ_t.
  the x_t terms combine and simplify to the ε-form the sampler actually uses:
    μ̃_t = (1/√α_t) · ( x_t − (β_t/√(1-ᾱ_t)) · ε̂ )
```

So one reverse step = **nudge `x_t` by a sliver of the predicted noise** (the mean), then **add fresh
Gaussian noise** of variance `β̃_t` (we're drawing a sample, not taking the mean). Iterate `T→0`.
"""

# %%
@torch.no_grad()
def sample(model, x, T, betas, alphas, alpha_bar, stochastic=True, record=(), gen=None):
    """Ancestral DDPM sampling from a given starting noise x. Returns (final, {t: (x_t, x̂_0)} snaps).
    stochastic=False drops the added noise (uses the posterior mean only)."""
    model.eval()
    snaps = {}
    for i in reversed(range(T)):
        t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
        eps_hat = model(x, t)
        a, ab, b = alphas[i], alpha_bar[i], betas[i]
        x0_hat = (x - (1 - ab).sqrt() * eps_hat) / ab.sqrt()          # running estimate of the clean img
        mean = (x - (1 - a) / (1 - ab).sqrt() * eps_hat) / a.sqrt()   # μ̃_t, ε-form from the derivation
        if i in record:
            snaps[i] = (x.clamp(-1, 1).clone(), x0_hat.clamp(-1, 1).clone())
        if i > 0 and stochastic:
            var = b * (1 - alpha_bar[i - 1]) / (1 - ab)               # β̃_t
            x = mean + var.sqrt() * torch.randn(x.shape, device=x.device, generator=gen)
        else:
            x = mean                                                  # last step, or mean-only mode
    return x, snaps


N = 16
gen = torch.Generator(device=DEV).manual_seed(0)
xT = torch.randn(N, 1, 28, 28, device=DEV, generator=gen)             # the SAME start for both methods

# one-shot straight from xT (predict ε̂ once, jump to x̂_0)
oneshot = predict_x0(xT, T - 1)

# the full iterative sampler from the same xT, recording a few waypoints for the trajectory
rec = [999, 800, 600, 400, 200, 100, 40, 0]
print(f"  running the {T}-step reverse loop on {N} images (slow part — {T} net calls)...")
final, snaps = sample(model, xT.clone(), T, betas, alphas, alpha_bar, record=rec, gen=gen)
print("  done. one-shot vs iterative from identical noise:")

# %% [markdown]
"""
## The payoff — same noise, two endings

Left: the one-shot `x̂_0` from `x_T`. Right: the iterative reverse loop from the *identical* `x_T`. Same
starting static — the loop is the entire difference between a grey smear and a digit.
"""

# %%
fig, axes = plt.subplots(N // 4, 8, figsize=(11, N // 4 * 1.4))
for k in range(N):
    r, c = divmod(k, 4)
    axes[r, c].imshow(to_img(oneshot[k]), cmap="gray"); axes[r, c].axis("off")
    axes[r, c + 4].imshow(to_img(final[k]), cmap="gray"); axes[r, c + 4].axis("off")
axes[0, 1].set_title("one-shot  x̂_0  (blurry means)", fontsize=10, loc="center")
axes[0, 5].set_title(f"iterative  ({T} steps)  → digits", fontsize=10, loc="center")
fig.tight_layout(); plt.show()
print("  identical x_T → one big jump gives mush; a thousand small steps give a digit. Why: iteration.")

# %% [markdown]
"""
## 3 · Watch the digit resolve

The trajectory makes the mechanism visible. **Top row** — the actual state `x_t` as the loop runs
`T→0`: static slowly organising. **Bottom row** — the model's `x̂_0` *estimate* at each step: it starts
as the same blurry digit-average the one-shot gave, and **commits to a single digit** as the small
steps (plus the re-injected noise) break the tie. The loop isn't denoising a fixed image; it's steering
an ambiguous average into one concrete sample.
"""

# %%
k = 0
cols = sorted(snaps.keys(), reverse=True)
fig, axes = plt.subplots(2, len(cols), figsize=(len(cols) * 1.35, 3.2))
for c, i in enumerate(cols):
    xt_i, x0_i = snaps[i]
    axes[0, c].imshow(to_img(xt_i[k]), cmap="gray"); axes[0, c].axis("off")
    axes[1, c].imshow(to_img(x0_i[k]), cmap="gray"); axes[1, c].axis("off")
    axes[0, c].set_title(f"t={i}", fontsize=9)
axes[0, 0].set_ylabel("x_t", rotation=0, ha="right", va="center", fontsize=10)
axes[1, 0].set_ylabel("x̂_0", rotation=0, ha="right", va="center", fontsize=10)
fig.suptitle("reverse trajectory: state x_t (top) organising, estimate x̂_0 (bottom) committing", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94)); plt.show()
print("  bottom row: a blurry average at t=999 sharpens into one digit by t=0 — that's a sample forming.")

# %% [markdown]
"""
## 4 · Why re-add noise: mean-only over-smooths

The `β̃_t` term looks optional — why not just follow the mean every step and get a clean path? Because
the mean at each step still points at that hedged average; without the injected noise to knock the
trajectory onto a *specific* digit, ancestral sampling drifts toward smoother, greyer, less varied
images. Run the identical `x_T` with the noise term switched off and compare the spread of outputs.
"""

# %%
gen2 = torch.Generator(device=DEV).manual_seed(0)
mean_only, _ = sample(model, xT.clone(), T, betas, alphas, alpha_bar, stochastic=False, gen=gen2)

diversity = lambda b: torch.pdist(b.flatten(1)).mean().item()        # mean pairwise L2 across samples
print(f"  sample diversity (mean pairwise distance):  stochastic {diversity(final):.2f}"
      f"   |  mean-only {diversity(mean_only):.2f}   (higher = more varied)")

fig, axes = plt.subplots(2, 8, figsize=(11, 3))
for k in range(8):
    axes[0, k].imshow(to_img(final[k]), cmap="gray"); axes[0, k].axis("off")
    axes[1, k].imshow(to_img(mean_only[k]), cmap="gray"); axes[1, k].axis("off")
axes[0, 0].set_ylabel("stochastic", rotation=0, ha="right", va="center", fontsize=9)
axes[1, 0].set_ylabel("mean-only", rotation=0, ha="right", va="center", fontsize=9)
fig.suptitle("the variance term keeps samples sharp & varied (top) vs smoother mean-only (bottom)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92)); plt.show()
print("  the noise isn't a nuisance — it's what makes this SAMPLING (draw a digit) not averaging.")

# %% [markdown]
"""
> Mean-only DDPM isn't *wrong* — it's a deterministic path, just a poorly-shaped one at `T` tiny steps.
> `06` (DDIM) rebuilds a **principled** deterministic sampler that's both sharp *and* lets you cut the
> step count by 20–100×. So hold onto this: the stochastic term is one design choice, not a law.

## Recap

- **One shot fails** because `x̂_0` from a noisy image is the **conditional mean** `E[x_0|x_t]` — at
  high noise, the blurry average of every digit that noise could become. Best single guess, worst
  sample. We measured the error climbing with `t` and saw the mush.
- **Iterative sampling** replaces the jump with the **reverse posterior** `q(x_{t-1}|x_t,x_0)`: nudge by
  a sliver of `ε̂` (the mean `μ̃_t`) and add `β̃_t` noise, `T→0`. Derived via Bayes on two Gaussians;
  the mean simplifies to `(1/√α_t)(x_t − (β_t/√(1-ᾱ_t))·ε̂)` — exactly `01`'s `sample` line.
- **The trajectory** shows the `x̂_0` estimate morph from a digit-average into one committed digit — a
  sample *forming*, not an image being cleaned.
- **The injected noise** is what makes it sampling; drop it and outputs get smoother and less varied.

That closes **Phase A** — you now understand the full generator end to end: forward (`02`), target
(`03`), model (`04`), reverse (`05`).

Next: **`06` — DDIM: deterministic, few-step sampling** — the same trained model, `1000 → 50 → 10`
steps, a free inference-time speed knob (and the principled deterministic path promised above).
"""
