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
# Diffusion · 02 — the forward process

In `01` we wrote two things and moved on:

- `make_schedule(T)` — some `betas`, `alphas`, `alpha_bar`,
- one forward line — `x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε`.

This notebook opens that box. Three questions:

1. **What is the schedule?** Noising is a slow Markov chain — add a *little* Gaussian noise `T` times.
   `betas` say how much each step; `alpha_bar` says how much of the original signal survives.
2. **Why one line instead of a 1000-step loop?** There's a closed form that jumps straight to any
   level `t` in one shot. We'll verify it matches the step-by-step chain *in distribution*.
3. **Why the `√` coefficients?** They make the process **variance-preserving** — `Var(x_t) ≈ 1` at
   every `t` — which is also why the chain ends at clean `N(0,I)` noise (the `torch.randn` that `01`'s
   sampler started from).

No network here — the forward process has **no learnable parameters**. It's pure, fixed noising.
"""

# %%
from pathlib import Path

import numpy as np
import torch
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


# %% [markdown]
"""
## The schedule: β → α → ᾱ

One noising **step** takes the current image and mixes in a splash of fresh Gaussian noise:

```-
x_t = √(1 - β_t)·x_{t-1}  +  √(β_t)·ε_t        ε_t ~ N(0, I),  independent each step
```

`β_t` is the *noise fraction* at step `t` — tiny at first, larger later. The linear schedule ramps it
from `1e-4` to `0.02` over `T=1000` steps. Two derived quantities:

- `α_t = 1 - β_t` — the *signal fraction* kept in a single step (≈ 0.99, barely below 1).
- `ᾱ_t = α_1·α_2·⋯·α_t` (cumulative product) — the signal kept after *all* `t` steps.

The point of `ᾱ` is **compounding**: each step keeps ~99% of the signal, but multiply a thousand
near-one numbers and it still decays to ~0. Watch the three curves.
"""


# %%
def make_schedule(T=1000, device="cpu"):
    """Linear DDPM schedule. Returns betas, alphas, alpha_bar (each shape (T,))."""
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


T = 1000
betas, alphas, alpha_bar = make_schedule(T)

for t in (0, 100, 250, 500, 750, 999):
    print(f"  t={t:>4}: β={betas[t]:.4f}  α={alphas[t]:.4f}  "
          f"ᾱ={alpha_bar[t]:.4f}  → signal √ᾱ={alpha_bar[t].sqrt():.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].plot(betas.numpy(), label="β_t (noise added per step)")
ax[0].plot(alphas.numpy(), label="α_t = 1-β_t (signal kept per step)")
ax[0].set_title("per-step (barely leaves 1.0)"); ax[0].set_xlabel("t"); ax[0].legend()
ax[1].plot(alpha_bar.numpy(), label="ᾱ_t (signal kept after t steps)")
ax[1].plot(alpha_bar.sqrt().numpy(), "--", label="√ᾱ_t (signal amplitude in x_t)")
ax[1].set_title("cumulative — the compounding decay to ~0"); ax[1].set_xlabel("t"); ax[1].legend()
fig.tight_layout(); plt.show()

# %% [markdown]
"""
`α ≈ 0.99` every step, yet `ᾱ_500 ≈ 0.08` — a thousand tiny multiplications compound into near-total
erasure. `√ᾱ_t` (dashed) is what actually multiplies the digit in `x_t`, so it's the honest "how much
of the picture is left" curve: still ~0.7 at `t=250`, down to ~0.03 by `t=999`.

## The closed form: skip the loop

The step rule is a Markov chain — to get `x_t` you'd apply it `t` times. But each step is *linear in
the image plus independent Gaussian noise*, and Gaussians have a gift: **a sum of independent
Gaussians is Gaussian, with variances adding**. So the whole chain collapses to one jump. Composing
two steps:

```-
x_t = √α_t·x_{t-1} + √(1-α_t)·ε
    = √α_t(√α_{t-1}·x_{t-2} + √(1-α_{t-1})·ε') + √(1-α_t)·ε
    = √(α_t α_{t-1})·x_{t-2}  +  [√α_t√(1-α_{t-1})·ε' + √(1-α_t)·ε]
```

The two noise terms are independent Gaussians; their variances add:
`α_t(1-α_{t-1}) + (1-α_t) = 1 - α_t α_{t-1}`. So they merge into a *single* `N(0,I)` with amplitude
`√(1 - α_t α_{t-1})`. Induct down to `x_0` and every `α` collects into `ᾱ_t`:

```-
x_t = √ᾱ_t·x_0  +  √(1-ᾱ_t)·ε        ε ~ N(0, I)      ← the one line from 01
```

One draw of noise, one level `t`, no loop. Let's confirm it's not just algebra — that the one-jump
formula and the actual `t`-step chain produce the **same distribution**.

To have something exact to check against, we start from a single fixed pixel `x0 = 0.6` (the code
stacks `N` copies into `x0_batch` only to *estimate* the mean/std empirically). The analytic
mean/std come straight off the closed form `x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε` (with `ε ~ N(0,1)`):

```-
Note: x0 constant → mean(x0) = x0, Var(x0)=0
mean(x_t) = √ᾱ_t·mean(x0) + √(1-ᾱ_t)·mean(ε) = √ᾱ_t·x0 + 0 = √ᾱ_t·x0
→ pred_mean = alpha_bar[t].sqrt() * 0.6

Var(x_t)  = ᾱ_t·Var(x0) + (1-ᾱ_t)·Var(ε) = ᾱ_t·0 + (1-ᾱ_t)·1 = 1-ᾱ_t
std(x_t)  = √(1-ᾱ_t)
→ pred_std = (1 - alpha_bar[t]).sqrt()
```

Those `pred_*` are the analytic ground truth the iterative and closed-form runs below should match.
"""


# %%
def forward_iterative(x0, t, alphas, betas):
    """Apply the noising STEP t+1 times (the honest Markov chain). x0 shape (N,) = N independent runs."""
    x = x0.clone()
    for i in range(t + 1):
        x = alphas[i].sqrt() * x + betas[i].sqrt() * torch.randn_like(x)
    return x


def forward_closed_form(x0, t, alpha_bar):
    """One jump to level t."""
    ab = alpha_bar[t]
    return ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn_like(x0)


torch.manual_seed(0)
N = 40000
for t in (50, 200, 500):
    x0 = 0.6                                         # the starting pixel value (matches the math)
    x0_batch = torch.full((N,), x0)                  # N independent runs of that same pixel
    xi = forward_iterative(x0_batch, t, alphas, betas)   # step-by-step
    xc = forward_closed_form(x0_batch, t, alpha_bar)     # one jump
    pred_mean = alpha_bar[t].sqrt() * x0
    pred_std = (1 - alpha_bar[t]).sqrt()
    print(f"  t={t:>3}  | iterative mean={xi.mean():+.3f} std={xi.std():.3f}"
          f"  | closed-form mean={xc.mean():+.3f} std={xc.std():.3f}"
          f"  | predicted mean={pred_mean:+.3f} std={pred_std:.3f}")

# %% [markdown]
"""
All three columns agree to ~2 decimals: the `t`-step chain, the one-line jump, and the analytic
`(√ᾱ_t·x0, √(1-ᾱ_t))` mean/std. The loop was never needed — **and this is what made `01`'s training
loop cheap**: for a random `t` per image we noise the whole batch in a single line, no `t`-step
simulation.

## Why the `√`: variance-preserving

Why `√ᾱ` and `√(1-ᾱ)`, not `ᾱ` and `(1-ᾱ)`? Because the square roots keep the **total variance
constant**. If `x0` has unit variance and `ε ~ N(0,1)` is independent, then

```-
Var(x_t) = (√ᾱ_t)²·Var(x0) + (√(1-ᾱ_t))²·Var(ε) = ᾱ_t·1 + (1-ᾱ_t)·1 = 1   for every t
```

The signal and noise variances trade off along `ᾱ_t + (1-ᾱ_t) = 1`, so `x_t` stays ~`N(0,1)`-scaled
the whole way — the network sees inputs of a stable magnitude at every noise level. Drop the `√`
(coefficients `ᾱ`, `1-ᾱ`) and variance sags to `ᾱ² + (1-ᾱ)²`, which dips to **0.5** in the middle. We
measure both.
"""

# %%
torch.manual_seed(0)
x0 = torch.randn(50000)                               # unit-variance signal, so the law is clean
ts = torch.arange(0, T, 20)
var_sqrt, var_nosqrt = [], []
for t in ts:
    ab = alpha_bar[t]
    eps = torch.randn_like(x0)
    var_sqrt.append((ab.sqrt() * x0 + (1 - ab).sqrt() * eps).var().item())         # correct
    var_nosqrt.append((ab * x0 + (1 - ab) * eps).var().item())                     # no sqrt
print(f"  with √  : Var(x_t) ranges {min(var_sqrt):.3f} .. {max(var_sqrt):.3f}  (flat ≈ 1)")
print(f"  without : Var(x_t) dips to {min(var_nosqrt):.3f}  (variance destroyed mid-way)")

plt.figure(figsize=(6.5, 3.4))
plt.plot(ts.numpy(), var_sqrt, label="√ᾱ, √(1-ᾱ)  → Var ≈ 1 (variance-preserving)")
plt.plot(ts.numpy(), var_nosqrt, label="ᾱ, (1-ᾱ)  → Var dips to 0.5")
plt.axhline(1.0, color="gray", ls=":", lw=1)
plt.xlabel("t"); plt.ylabel("Var(x_t)"); plt.legend(); plt.title("why the √ coefficients")
plt.tight_layout(); plt.show()

# %% [markdown]
"""
## The signal-to-noise ratio

Those same two variances, taken as a **ratio** instead of a sum, are the **signal-to-noise ratio** —
how much signal there is per unit of noise at level `t`. Same terms as the `Var(x_t)` line above,
divided rather than added:

```-
SNR(t) = signal variance / noise variance
       = (√ᾱ_t)²·Var(x0) / [(√(1-ᾱ_t))²·Var(ε)]
       = ᾱ_t·1 / [(1-ᾱ_t)·1]
       = ᾱ_t / (1-ᾱ_t)
```

Huge at `t=0` (almost all signal), `→ 0` as `t→T` (almost all noise) — the same story `√ᾱ` told,
compressed into one number. We reuse it in `03`, where it turns out to be exactly the per-`t` weight
relating the two training targets (predict the noise vs predict the image). Let's measure it and check
it matches the closed form.
"""

# %%
torch.manual_seed(0)
x0 = torch.randn(50000)                               # unit-variance signal
ts = torch.arange(0, T, 20)
snr_measured, snr_formula = [], []
for t in ts:
    ab = alpha_bar[t]
    signal = ab.sqrt() * x0                           # the √ᾱ·x0 part of x_t
    noise = (1 - ab).sqrt() * torch.randn_like(x0)    # the √(1-ᾱ)·ε part of x_t
    snr_measured.append((signal.var() / noise.var()).item())
    snr_formula.append((ab / (1 - ab)).item())

for t in (0, 100, 250, 500, 750, 999):
    ab = alpha_bar[t]
    print(f"  t={t:>4}: SNR = ᾱ/(1-ᾱ) = {(ab / (1 - ab)):9.3f}")

plt.figure(figsize=(6.5, 3.4))
plt.semilogy(ts.numpy(), snr_formula, label="ᾱ_t / (1-ᾱ_t)  (closed form)")
plt.semilogy(ts.numpy(), snr_measured, "o", ms=3, label="measured  Var(signal)/Var(noise)")
plt.xlabel("t"); plt.ylabel("SNR (log)"); plt.legend(); plt.title("signal-to-noise ratio across t")
plt.tight_layout(); plt.show()
print("  all signal at t=0, ~0 by t=T — reused in 03 as the weight that relates the two targets.")

# %% [markdown]
"""
## The endpoint — and why sampling starts from noise

As `t → T`, `ᾱ_t → 0`, so the closed form becomes

```-
x_T = √ᾱ_T·x0 + √(1-ᾱ_T)·ε  ≈  0·x0 + 1·ε  =  ε  ~  N(0, I)
```

The original digit is *gone* — every image, whatever it started as, lands at the same standard
Gaussian. That's the whole reason generation can run **backwards from pure noise**: the forward
process funnels all data to one known distribution, so `01`'s sampler could start at `torch.randn`
and have somewhere valid to begin. Now let's *see* a digit make that trip.
"""

# %%
x = load_mnist(train=False)
torch.manual_seed(1)
digits = x[torch.randperm(len(x))[:4]]               # 4 held-out digits
show_t = [0, 50, 150, 300, 500, 750, 999]

fig, axes = plt.subplots(len(digits), len(show_t), figsize=(len(show_t) * 1.2, len(digits) * 1.2))
for r, d in enumerate(digits):
    eps = torch.randn_like(d)                         # same noise across the row → smooth dissolve
    for c, t in enumerate(show_t):
        ab = alpha_bar[t]
        xt = ab.sqrt() * d + (1 - ab).sqrt() * eps
        ax = axes[r, c]
        ax.imshow(to_img(xt), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(f"t={t}\n√ᾱ={ab.sqrt():.2f}", fontsize=8)
fig.suptitle("the forward process — a digit dissolving into N(0,I) via the closed form", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95)); plt.show()
print("left: the digit (√ᾱ≈1).  right: pure static (√ᾱ≈0).  Same ε down each row.")

# %% [markdown]
"""
## Recap

- The forward process is a **fixed, parameter-free** Markov chain: add a little Gaussian noise `T`
  times on the `β` schedule.
- `ᾱ_t` (cumulative signal) lets us **skip the loop** — one jump `x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε`, which
  we verified matches the chain in distribution and is exactly what made training cheap.
- The `√` coefficients make it **variance-preserving** (`Var(x_t) ≈ 1`), and the chain **ends at
  `N(0,I)`** — the noise `01`'s sampler starts from.

A knob we didn't turn: the *shape* of the schedule (linear here). Cosine holds signal longer and is
the better default — that's the `07` box.

Next: **`03` — the training target: why the network predicts the noise `ε`, not the image** (and the
two wiring checks that tell you the loop is even hooked up).
"""
