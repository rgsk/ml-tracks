"""
CHILD WALKTHROUGH (digs into ddpm exp_2): the FORWARD PROCESS, top-down.

exp_1 (in the parent ddpm.py) trained a U-Net and sampled digits from noise — the whole game.
The forward process is the FIRST box we open: exactly how a clean digit x0 is turned into the
noise the network learns to predict. It's a big enough topic to get its own folder — so we run
the SAME move here, fractally: see the whole thing work first, THEN break it apart.

The one equation everything rests on — noise a digit to ANY level t in a single jump:

    x_t = √ᾱ_t · x0  +  √(1-ᾱ_t) · ε ,      ε ~ N(0, I)

where β_t is the schedule, α_t = 1 - β_t, and ᾱ_t = ∏_{s≤t} α_s. Because of this closed form the
train loop never noises step-by-step: pick t, draw ε, done (that's the parent's train loop).

Layers (each an `exp_*`; run it, read the output, then say "next"):
  1. the WHOLE GAME    — watch a real digit dissolve into static across t (linear schedule). Rough
                         narration only: ᾱ is a dial 1→0, the √ keeps signal+noise balanced, it ends
                         at pure noise.                                                        (here)
  then open the boxes — each a "why" about that picture:
  2. the SCHEDULE       — what ᾱ actually is: β_t, α_t, ᾱ_t=∏α; tiny nibbles COMPOUND.
  3. the √ / CLOSED FORM — one-jump == step-by-step (Monte-Carlo), and why the coefficients are √
                          (variance-preserving: Var(x_t)≈1 for all t).
  4. the ENDPOINT       — by t=T, ᾱ≈0 so x_T ≈ N(0,I) regardless of x0 → why sampling starts from
                          pure noise.
  5. the SCHEDULE SHAPE — the cosine schedule: declare ᾱ(t)=cos²(...) and back-solve β (the reverse
                          of how the linear schedule is built).
  6. LINEAR vs COSINE   — compare the ᾱ curves + count "wasted" near-pure-noise steps; a second
                          dissolve (linear vs cosine rows) makes "cosine keeps signal longer" VISIBLE.
  7. the SNR            — SNR(t)=ᾱ/(1-ᾱ), log-SNR: the difficulty curriculum the model sees at each
                          t, and the coordinate modern schedules (EDM, flow) are defined in.

  (NOT here — the next box up: the training TARGET (predict ε, ε-vs-x0 weighting) and batch assembly
   are the parent's exp_3 "training target" dig-in. This child stays about the forward process.)

Content mirrors the older bottom-up diffusion/walkthroughs/forward_process.py, re-sequenced
top-down as a child dig-in under the ddpm track. No torchvision; shared MNIST at
new/diffusion/data/mnist.npz.
"""
from __future__ import annotations

import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))                 # .../forward_process/walkthroughs
# walk up to the shared new/diffusion/ root (holds data/):
#   walkthroughs -> forward_process -> children -> ddpm -> walkthroughs -> diffusion
_DIFF = os.path.abspath(os.path.join(_HERE, *([".."] * 5)))        # new/diffusion
_FIGS = os.path.join(_HERE, "figures", "experiments")


def _banner(*lines):
    print("=" * 70)
    for line in lines:
        print(line)
    print("=" * 70)


def _to_img(x):
    """(1,28,28)-ish tensor in [-1,1] -> HxW numpy in [0,1] for imshow."""
    return ((x.squeeze().clamp(-1, 1) + 1) / 2).cpu().numpy()


def _mnist(train=True):
    """MNIST images (N,1,28,28) in [-1,1], from the cached npz. No torchvision."""
    import numpy as np
    path = os.path.join(_DIFF, "data", "mnist.npz")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import urllib.request
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        print(f"  downloading MNIST npz (~11MB) -> {path} ...")
        urllib.request.urlretrieve(url, path)
    d = np.load(path)
    x = d["x_train"] if train else d["x_test"]                     # (N,28,28) uint8 [0,255]
    return (torch.from_numpy(x).float() / 127.5 - 1.0).unsqueeze(1)  # (N,1,28,28) in [-1,1]


def make_linear_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    """The original DDPM schedule: β_t linearly spaced. Returns (betas, alphas, alpha_bars),
    each shape (T,), indexed t=0..T-1. (This is the SAME schedule the parent ddpm.py trained
    exp_1 on — we build it here to read it apart in exp_2.)"""
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)          # ∏ of alphas up to each t
    return betas, alphas, alpha_bars


# ---------------------------------------------------------------------------
# LAYER 1 (the whole game): SEE the forward process before we explain any of it.
#
# Take a few real digits and apply the closed form  x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε  at a row of
# growing t, and watch each digit melt into static. Rough narration only — three things are
# going on, each gets its own box next:
#   - ᾱ_t is a DIAL from all-signal (t=0) to all-noise (t=T).            (exp_2: the schedule)
#   - the coefficients are √ so signal + noise stay balanced (no blow-up). (exp_3: the √)
#   - by the last column the digit is gone — pure N(0,I).                 (exp_4: the endpoint)
# Use the SAME ε across a row, so the only thing changing left→right is HOW MUCH of it we mix in.
# ---------------------------------------------------------------------------
def exp_1_dissolve(seed=0, T=1000):
    """The whole game of the forward process: dissolve real MNIST digits into noise across t and
    save the grid. No derivations yet — see it happen, get the map. exp_2..exp_4 open each box."""
    _banner("LAYER 1: the whole game — watch a real digit dissolve into noise")

    torch.manual_seed(seed)
    _, _, alpha_bars = make_linear_schedule(T=T)

    print("  the forward process, in one breath:")
    print("    x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε    — mix a fixed digit x0 with fresh noise ε")
    print("    ᾱ_t is a DIAL: t=0 → all signal (ᾱ=1), t=T → all noise (ᾱ≈0)")
    print("    the coefficients are √ so signal + noise stay balanced; by t=T the digit is gone.\n")

    imgs = _mnist(train=False)
    picks = imgs[[1, 3, 5, 7]]                                      # four different held-out digits
    eps = torch.randn_like(picks)                                  # ONE noise field per row, scaled in
    ts = [0, 50, 100, 200, 400, 600, 800, T - 1]

    import matplotlib
    matplotlib.use("Agg")                                          # headless: save, don't show
    import matplotlib.pyplot as plt
    rows, cols = picks.shape[0], len(ts)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.15))
    for r in range(rows):
        for c, t in enumerate(ts):
            ab = alpha_bars[t]
            x_t = ab.sqrt() * picks[r] + (1 - ab).sqrt() * eps[r]  # the closed form
            ax = axes[r, c]
            ax.imshow(_to_img(x_t), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={t}\n√ᾱ={ab.sqrt().item():.2f}", fontsize=9)
    fig.suptitle("the forward process: x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε  — same ε, more of it as t grows",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "01_dissolve.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"  wrote {out} — left: the digit; right: pure static. That's the whole forward process.")
    print("  Every later box explains a 'why' about a picture you've now seen. Next (exp_2): open")
    print("  the first box — the SCHEDULE — and see what ᾱ_t actually is and why it collapses.")


# ---------------------------------------------------------------------------
# LAYER 2 (first box): the schedule — β_t, α_t, and the cumulative ᾱ_t.
#
# exp_1 showed ᾱ_t acting as a dial. Here's what it IS. From a per-step noise level β_t:
#   β_t    how much fresh noise we mix in at step t (grows over time)
#   α_t    = 1 - β_t : the fraction of signal that survives ONE step
#   ᾱ_t    = α_1·…·α_t : the fraction of the ORIGINAL signal that survives ALL t steps (a
#            cumulative PRODUCT — this is why a barely-lossy per-step α collapses ᾱ fast)
# ---------------------------------------------------------------------------
def exp_2_schedule():
    """Open the schedule and READ it: for a few times t, show β_t (noise added this step), α_t
    (signal surviving one step), ᾱ_t (signal surviving all t steps), and the mix fractions √ᾱ
    and √(1-ᾱ) — the exact numbers behind the dissolve you just watched."""
    _banner("LAYER 2: the schedule — β, α, and the cumulative ᾱ")

    T = 1000                                           # canonical DDPM T
    betas, alphas, alpha_bars = make_linear_schedule(T=T)

    print(f"  linear schedule, T={T}, β_t linearly from {betas[0]:.1e} to {betas[-1]:.1e}")
    print("    β_t   = how much NOISE we mix in at step t (grows over time)")
    print("    α_t   = 1 - β_t = fraction of signal surviving ONE step")
    print("    ᾱ_t   = α_1·…·α_t = fraction of the ORIGINAL surviving t steps\n")

    print(f"  {'t':>4} | {'β_t':>8} | {'α_t':>8} | {'ᾱ_t':>10} | "
          f"{'signal √ᾱ':>10} | {'noise √(1-ᾱ)':>13}")
    print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*13}")
    for t in [0, 100, 250, 500, 750, T - 1]:
        b, a, ab = betas[t].item(), alphas[t].item(), alpha_bars[t].item()
        print(f"  {t:>4} | {b:>8.5f} | {a:>8.5f} | {ab:>10.5f} | {ab**0.5:>10.4f} | {(1-ab)**0.5:>13.4f}")
    print()
    print("  Read it: at t=500, α_t is still ~0.99 (one step barely changes the image), yet ᾱ_t")
    print("  has already collapsed to ~0.08 — the tiny per-step losses COMPOUND over 500")
    print("  multiplications. So 'signal √ᾱ' shrinks 1 -> 0 and 'noise √(1-ᾱ)' grows 0 -> 1 as t")
    print("  rises — exactly the fade you saw in exp_1's grid. ᾱ_t is the single dial for that mix.")
    print("  Next (exp_3): prove that dialing with ᾱ == adding noise step-by-step, and why the √.")


# ---------------------------------------------------------------------------
# LAYER 3 (box): the √ / closed form — is the one-jump REALLY the step-by-step process?
#
# exp_1's dissolve and exp_2's table both used the one-line jump  x_t = √ᾱ·x0 + √(1-ᾱ)·ε. But the
# "real" forward process adds a little noise T times:  x_s = √α_s·x_{s-1} + √(1-α_s)·z_s. Do they
# agree? They can't match sample-by-sample (each draws its own noise) but they match in
# DISTRIBUTION — and seeing why also reveals why the coefficients are √: variance preservation.
#
# DERIVATION — why T single steps collapse into one jump (and why √):
#   one step:   x_s = √α_s·x_{s-1} + √(1-α_s)·z_s,   z_s ~ N(0,I) iid.
#   compose two:
#     x_2 = √α_2·(√α_1·x0 + √(1-α_1)·ε_1) + √(1-α_2)·ε_2
#         = √(α_1α_2)·x0 + [√α_2·√(1-α_1)·ε_1 + √(1-α_2)·ε_2]
#   the bracket is two independent zero-mean Gaussians, so their VARIANCES add:
#     α_2(1-α_1) + (1-α_2) = 1 - α_1α_2,
#   hence it's one fresh √(1-α_1α_2)·ε. With ᾱ_2 = α_1α_2:  x_2 = √ᾱ_2·x0 + √(1-ᾱ_2)·ε.
#   induction over t →  x_t = √ᾱ·x0 + √(1-ᾱ)·ε.
#   variance (x0 ⟂ ε, Var(x0)=1):  Var(x_t) = (√ᾱ)²·1 + (√(1-ᾱ))²·1 = ᾱ + (1-ᾱ) = 1  for EVERY t.
#   → the √ is chosen precisely so signal-power ᾱ + noise-power (1-ᾱ) = 1: no blow-up, no fade.
# ---------------------------------------------------------------------------
def forward_iterative(x0, betas, t_idx, n_samples, generator=None):
    """Step-by-step forward process from x0 through steps 0..t_idx, run n_samples independent
    times. Each step: x <- √(1-β)·x + √β·z, fresh z ~ N(0,1). x0 is a scalar tensor; returns a
    (n_samples,) tensor of x_t draws."""
    x = x0.expand(n_samples).clone()
    for s in range(t_idx + 1):
        z = torch.randn(n_samples, generator=generator)
        x = torch.sqrt(1.0 - betas[s]) * x + torch.sqrt(betas[s]) * z
    return x


def exp_3_closed_form(seed=0, T=1000):
    """Two 'why's about the dissolve. (A) Monte-Carlo: the slow step-by-step noising lands in the
    SAME distribution as the one-line jump (matching mean √ᾱ·x0, std √(1-ᾱ)). (B) why the
    coefficients are √: the process is variance-preserving — Var(x_t) ≈ 1 at every t.
    (Derivation of both in the comment above.)"""
    _banner("LAYER 3: the √ / closed form — one-jump == step-by-step, and why √")
    g = torch.Generator().manual_seed(seed)
    betas, _, alpha_bars = make_linear_schedule(T=T)

    # (A) we can't compare single samples (each draws its own noise); compare the DISTRIBUTION.
    x0 = torch.tensor(2.0)
    n = 40000
    print("  (A) step-by-step vs one-jump. Noise a fixed x0=2.0 step-by-step 40k times to level t,")
    print("      then compare its empirical mean/std to the closed form's √ᾱ·x0 and √(1-ᾱ):\n")
    print(f"  {'t':>4} | {'iter mean':>10} {'iter std':>9} | {'√ᾱ·x0':>8} {'√(1-ᾱ)':>8}")
    print(f"  {'-'*4}-+-{'-'*10}-{'-'*9}-+-{'-'*8}-{'-'*8}")
    for t in [100, 500, T - 1]:
        xt = forward_iterative(x0, betas, t, n, generator=g)
        ab = alpha_bars[t]
        print(f"  {t:>4} | {xt.mean():>+10.4f} {xt.std():>9.4f} | "
              f"{(ab.sqrt()*x0).item():>+8.4f} {(1-ab).sqrt().item():>8.4f}")
    print("\n      match → we can SKIP the T-step loop and jump straight to any t. That one-liner is")
    print("      exactly what exp_1's dissolve and the parent's train loop use.\n")

    # (B) why √: variance preservation. x0 ~ N(0,1) → Var(x_t) = ᾱ + (1-ᾱ) = 1 for every t.
    print("  (B) why the coefficients are √:  (√ᾱ)² + (√(1-ᾱ))² = ᾱ + (1-ᾱ) = 1. On unit-variance")
    print("      data Var(x_t) stays ≈ 1 at EVERY t — signal power drains, noise power fills, sum=1:\n")
    x0v = torch.randn(n, generator=g)
    print(f"  {'t':>4} | {'ᾱ':>8} {'1-ᾱ':>8} | {'Var(x_t)':>9}")
    print(f"  {'-'*4}-+-{'-'*8}-{'-'*8}-+-{'-'*9}")
    for t in [0, 250, 500, 750, T - 1]:
        ab = alpha_bars[t]
        eps = torch.randn(n, generator=g)
        xt = ab.sqrt() * x0v + (1 - ab).sqrt() * eps
        print(f"  {t:>4} | {ab.item():>8.4f} {(1-ab).item():>8.4f} | {xt.var():>9.4f}")
    print("\n      Var pinned at ~1: THAT is why it's √ and not, say, ᾱ and (1-ᾱ) directly — the")
    print("      squares are what must sum to 1, so the amplitudes are their square roots. No")
    print("      blow-up, no fade — x_t just trades signal for noise as t grows.")
    print("  Next (exp_4): where the process ENDS — x_T ~ N(0,I), the digit fully erased.")


def run_experiments():
    # exp_1_dissolve()
    # exp_2_schedule()
    exp_3_closed_form()


if __name__ == "__main__":
    run_experiments()
