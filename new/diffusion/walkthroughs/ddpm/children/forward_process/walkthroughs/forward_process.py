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

import math
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
# (Full derivation in the ''' block just below forward_iterative.)
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


def forward_closed_form(x0, alpha_bars, t_idx, n_samples, generator=None):
    """The ONE-JUMP forward process: x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε, one fresh ε ~ N(0,1) per sample.
    Same signature/return as forward_iterative so the two can be compared draw-for-draw — this is
    the loop above collapsed into a single step by the derivation in the comment block."""
    ab = alpha_bars[t_idx]
    eps = torch.randn(n_samples, generator=generator)
    return ab.sqrt() * x0 + (1.0 - ab).sqrt() * eps


'''
DERIVATION — the forward closed form  x_t = √ᾱ·x0 + √(1−ᾱ)·ε : why the t single steps
collapse into ONE jump.
  Single step:  x_s = √α_s·x_{s−1} + √(1−α_s)·z_s,   z_s ~ N(0,I) iid.
  Three facts:
     (A) scaling by c scales VARIANCE by c²;
     (B) independent Gaussians added → their variances add;
     (C) a sum of Gaussians is Gaussian.
  Compose two steps:
      x_1 = √α₁·x0 + √(1−α₁)·ε₁
      x_2 = √α₂·x_1 + √(1−α₂)·ε₂
          = √α₂·[√α₁·x0 + √(1−α₁)·ε₁] + √(1−α₂)·ε₂
          = √(α₁α₂)·x0 + [ √α₂·√(1−α₁)·ε₁ + √(1−α₂)·ε₂ ]
  The bracket is two independent zero-mean Gaussians; by A+B its variance is
      α₂(1−α₁) + (1−α₂)  =  1 − α₁α₂,
  and by C it is one fresh noise √(1−α₁α₂)·ε. With ᾱ₂ = α₁α₂:
      x_2 = √ᾱ₂·x0 + √(1−ᾱ₂)·ε.
  By induction over t (each step folds one more α in):
      x_t = √ᾱ·x0 + √(1−ᾱ)·ε,   ᾱ = ∏_{s≤t} α_s.        [QED]

Why the coefficients are square roots — VARIANCE PRESERVATION (Var(x0)=1, x0 ⟂ ε):
      Var(x_t) = (√ᾱ)²·Var(x_0) + (√(1−ᾱ))²·Var(ε)
               =    ᾱ   ·   1      +    (1−ᾱ)   ·   1
               =    ᾱ + (1 − ᾱ)
               =    1                        ← for EVERY t
signal power ᾱ + noise power (1−ᾱ) always sum to 1, so x_t neither blows up nor
fades — it just trades signal for noise.
'''


def exp_3_closed_form(seed=0, T=1000):
    """Two 'why's about the dissolve. (A) Monte-Carlo: the slow step-by-step noising lands in the
    SAME distribution as the one-line jump (matching mean √ᾱ·x0, std √(1-ᾱ)). (B) why the
    coefficients are √: the process is variance-preserving — Var(x_t) ≈ 1 at every t.
    (Derivation of both in the comment above.)"""
    _banner("LAYER 3: the √ / closed form — one-jump == step-by-step, and why √")
    g = torch.Generator().manual_seed(seed)
    betas, _, alpha_bars = make_linear_schedule(T=T)

    # (A) run BOTH procedures as Monte-Carlo and compare. We can't match single samples (each
    #     draws its own noise), so compare the DISTRIBUTION each lands in — iterative (T-step loop)
    #     vs closed form (one jump). Both should agree, and both should hit the analytic √ᾱ·x0 /
    #     √(1-ᾱ) that the derivation predicts.
    x0 = torch.tensor(2.0)
    n = 40000
    print("  (A) step-by-step vs one-jump. Noise a fixed x0=2.0 to level t, 40k times, TWO ways —")
    print("      forward_iterative (the T-step loop) and forward_closed_form (the single jump) —")
    print("      then read off each one's empirical mean/std. They should match each other, and")
    print("      both should match the analytic √ᾱ·x0 and √(1-ᾱ):\n")
    print(f"  {'t':>4} | {'iter mean':>10} {'iter std':>9} | {'closed mean':>11} {'closed std':>10} "
          f"| {'√ᾱ·x0':>8} {'√(1-ᾱ)':>8}")
    print(f"  {'-'*4}-+-{'-'*10}-{'-'*9}-+-{'-'*11}-{'-'*10}-+-{'-'*8}-{'-'*8}")
    for t in [100, 500, T - 1]:
        xi = forward_iterative(x0, betas, t, n, generator=g)
        xc = forward_closed_form(x0, alpha_bars, t, n, generator=g)
        ab = alpha_bars[t]
        print(f"  {t:>4} | {xi.mean():>+10.4f} {xi.std():>9.4f} | "
              f"{xc.mean():>+11.4f} {xc.std():>10.4f} | "
              f"{(ab.sqrt()*x0).item():>+8.4f} {(1-ab).sqrt().item():>8.4f}")
    print("\n      all three columns agree → the T-step loop and the one jump land in the SAME")
    print("      distribution, and it's exactly the √ᾱ·x0 / √(1-ᾱ) the derivation predicts. So we")
    print("      can SKIP the loop and jump straight to any t — what exp_1's dissolve and the")
    print("      parent's train loop use.\n")

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


# ---------------------------------------------------------------------------
# LAYER 4 (box): the endpoint — where the dissolve ENDS, and why it's the same for every digit.
#
# The last column of exp_1's grid was pure static, identical-looking for all four digits. Here's
# why. From exp_3, x_t has mean √ᾱ·x0 and std √(1-ᾱ). As t→T, ᾱ→0: the signal term √ᾱ·x0 VANISHES
# and the std →1, so x_T ~ N(0,I) no matter what x0 was. We noise two very different images and
# watch them start distinguishable (different means mid-way) but converge to the SAME N(0,1) by
# t=T. That universal endpoint is what lets generation START from a fresh scoop of noise.
# ---------------------------------------------------------------------------
def exp_4_endpoint(seed=0, T=1000):
    """Noise two very different x0 (+2.0 and -5.0) to several t. Mid-way their distributions differ
    (means ≈ √ᾱ·x0); by t=T both are ≈ N(0,1) — x0 fully erased. Hence the sampler can begin from
    a fresh scoop of N(0,1)."""
    _banner("LAYER 4: the endpoint — x_T ~ N(0,I), the digit fully erased")
    g = torch.Generator().manual_seed(seed)
    betas, _, alpha_bars = make_linear_schedule(T=T)
    n = 40000
    x0_a, x0_b = torch.tensor(2.0), torch.tensor(-5.0)

    print("  From exp_3: x_t has mean √ᾱ·x0 and std √(1-ᾱ). As t→T, ᾱ→0, so the signal term √ᾱ·x0")
    print("  vanishes and std→1. Noise two very different images (x0=+2.0 and x0=-5.0) to a few")
    print("  levels t, 40k times each, and watch their distributions converge:\n")
    print(f"  {'t':>4} | {'ᾱ':>9} | {'+2.0 mean':>9} {'std':>7} | {'-5.0 mean':>9} {'std':>7}")
    print(f"  {'-'*4}-+-{'-'*9}-+-{'-'*9}-{'-'*7}-+-{'-'*9}-{'-'*7}")
    for t in [0, 250, 500, 750, T - 1]:
        xa = forward_iterative(x0_a, betas, t, n, generator=g)
        xb = forward_iterative(x0_b, betas, t, n, generator=g)
        ab = alpha_bars[t].item()
        print(f"  {t:>4} | {ab:>9.5f} | {xa.mean():>+9.4f} {xa.std():>7.4f} | "
              f"{xb.mean():>+9.4f} {xb.std():>7.4f}")
    print()
    print("  At t=0 the two are maximally distinct — mean ≈ x0 (+2.0 vs -5.0) with std ≈ 0 (ᾱ≈1, so")
    print("  almost no noise yet): all the digit-specific information is intact.")
    print("  At t=250 the two are still far apart (means ≈ √ᾱ·x0: +1.44 vs -3.61) — the digit still")
    print("  shows through, exactly the mid columns of exp_1's dissolve. But as ᾱ→0 the signal term")
    print("  dies, and by t=999 BOTH are ≈ N(0,1): mean~0, std~1, indistinguishable. The forward")
    print("  process has ERASED x0 — the last column looks the same for ANY digit.\n")
    print("  Why this is the linchpin of GENERATION: the endpoint is the SAME universal N(0,I) for")
    print("  every image. So to generate, grab a fresh scoop of N(0,1) static (free to sample) — a")
    print("  valid x_T for SOME digit — and run the process BACKWARDS, denoising step by step, into")
    print("  a brand-new digit. That reverse is exactly the parent ddpm.py's sampler.")
    print("  Next (exp_5): the schedule SHAPE — the cosine schedule, and why it keeps signal longer.")


# ---------------------------------------------------------------------------
# LAYER 5 (box): the schedule SHAPE — the cosine schedule.
#
# Everything so far used the LINEAR schedule (exp_2): set β_t directly, ᾱ falls out as a cumprod.
# The cosine schedule (Nichol & Dhariwal 2021, "Improved DDPM") flips the CONSTRUCTION: declare
# the signal curve ᾱ(t) you want — a gentle cos² falloff from 1 to 0 — then back-solve the per-step
# betas from the ratio of consecutive ᾱ:  β_t = 1 - ᾱ_t/ᾱ_{t-1}  (clamped < 0.999 so the tail can't
# demand more than 100% noise). Same three arrays, same closed-form jump; only the SHAPE of the
# noise-over-time curve changes. This box builds it and reads it; exp_6 puts it beside linear.
# ---------------------------------------------------------------------------
def make_cosine_schedule(T=1000, s=0.008, max_beta=0.999):
    """Cosine schedule: DECLARE ᾱ(t) as a cos² falloff, then back-solve betas. Returns
    (betas, alphas, alpha_bars), each shape (T,) — same interface as make_linear_schedule."""
    steps = torch.arange(T + 1, dtype=torch.float32)          # 0..T (T+1 points)
    f = torch.cos(((steps / T) + s) / (1 + s) * (math.pi / 2)) ** 2
    alpha_bars_full = f / f[0]                                 # normalize so ᾱ(0)=1
    # β_t from the drop between consecutive ᾱ, then clamp the tail.
    betas = (1.0 - alpha_bars_full[1:] / alpha_bars_full[:-1]).clamp(max=max_beta)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)                  # recompute from clamped betas
    return betas, alphas, alpha_bars


def exp_5_cosine_schedule():
    """Build the cosine schedule and READ it: β_t (tiny early, ramps late) and its ᾱ_t curve. The
    point of THIS box is the CONSTRUCTION — ᾱ is set directly as cos², β back-solved, the reverse
    of how the linear schedule is defined."""
    _banner("LAYER 5: the cosine schedule — declare ᾱ as cos², back-solve β")

    T = 1000
    betas, _, alpha_bars = make_cosine_schedule(T=T)

    print("  linear (exp_2):  set β_t  ->  ᾱ = cumprod(1-β)   (ᾱ falls out)")
    print("  cosine (here):   set ᾱ = cos²(...)  ->  β_t back-solved from it")
    print("    β_t = 1 - ᾱ_t / ᾱ_{t-1}   (clamped < 0.999 at the tail)\n")

    print(f"  {'t':>4} | {'β_t':>8} | {'ᾱ_t':>10} | {'signal √ᾱ':>10} | {'noise √(1-ᾱ)':>13}")
    print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*13}")
    for t in [0, 100, 250, 500, 750, T - 1]:
        b, ab = betas[t].item(), alpha_bars[t].item()
        print(f"  {t:>4} | {b:>8.5f} | {ab:>10.5f} | {ab**0.5:>10.4f} | {(1-ab)**0.5:>13.4f}")
    print()
    print("  β_t is NOT linear here: it stays tiny early (keeping the image crisp longer), then")
    print("  ramps up toward the end. And ᾱ_t descends gently — at t=500 ᾱ≈0.49 (√ᾱ≈0.70) vs")
    print("  linear's ᾱ≈0.078 (√ᾱ≈0.28): the digit is still clearly there where linear is mush.")
    print("  Same forward math as exp_2/3/4 — only the noise-over-time CURVE changed.")
    print("  Next (exp_6): put the two curves side by side + a second dissolve, and see why cosine")
    print("  keeps signal alive longer (which matters most for small images like MNIST).")


# ---------------------------------------------------------------------------
# LAYER 6 (box): linear vs cosine — why the shape matters, made VISIBLE.
#
# Put the two ᾱ curves side by side. Linear plunges to ~0 well before t=T (many late steps are
# already pure noise = wasted capacity) AND rushes the informative middle; cosine descends gently
# and keeps usable signal much longer. We measure it (count near-pure-noise steps) and then SEE it:
# a second dissolve with a LINEAR row and a COSINE row, same digit and same ε, so the only
# difference is the schedule. This is where exp_1's dissolve returns as a comparison.
# ---------------------------------------------------------------------------
def exp_6_linear_vs_cosine(seed=0, T=1000):
    """Compare the two schedules' ᾱ curves, count how many steps each spends in the near-pure-noise
    regime, and save a linear-vs-cosine dissolve grid (same digit, same ε) that makes 'cosine keeps
    signal alive longer' literally visible."""
    _banner("LAYER 6: linear vs cosine — cosine keeps signal alive longer")

    torch.manual_seed(seed)
    _, _, ab_lin = make_linear_schedule(T=T)
    _, _, ab_cos = make_cosine_schedule(T=T)

    # (A) the two curves, side by side.
    print("  ᾱ (fraction of the ORIGINAL signal still present) at each t:\n")
    print(f"  {'t':>4} | {'linear ᾱ':>10} {'√ᾱ':>7} | {'cosine ᾱ':>10} {'√ᾱ':>7}")
    print(f"  {'-'*4}-+-{'-'*10}-{'-'*7}-+-{'-'*10}-{'-'*7}")
    for t in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, T - 1]:
        al, ac = ab_lin[t].item(), ab_cos[t].item()
        print(f"  {t:>4} | {al:>10.5f} {al**0.5:>7.4f} | {ac:>10.5f} {ac**0.5:>7.4f}")
    print()

    # (B) "wasted" = ᾱ so small the image is basically gone → denoising there teaches little.
    thr = 0.01                                    # ᾱ < 0.01  <=>  signal √ᾱ < 0.1
    waste_lin, waste_cos = int((ab_lin < thr).sum()), int((ab_cos < thr).sum())
    first_lin = int((ab_lin < thr).float().argmax())     # first t below threshold
    first_cos = int((ab_cos < thr).float().argmax())
    print(f"  'nearly pure noise' = ᾱ < {thr} (signal √ᾱ < {thr**0.5:.1f}):")
    print(f"    linear: crosses at t={first_lin}, so {waste_lin}/{T} steps ({waste_lin/T:.0%}) contribute little")
    print(f"    cosine: crosses at t={first_cos}, so {waste_cos}/{T} steps ({waste_cos/T:.0%}) contribute little")
    print("  Linear hits near-pure-noise early and burns a big chunk of its steps there (nothing")
    print("  left to remove = nothing to learn), AND rushes the informative middle. Cosine holds")
    print("  signal deep into the schedule — far more steps land in the useful regime. This is the")
    print("  Improved-DDPM motivation; it matters MOST for small images (MNIST) with little redundancy.\n")

    # (C) SEE it: same digit, same ε, linear row vs cosine row.
    imgs = _mnist(train=False)
    x0 = imgs[0]                                   # one held-out digit
    eps = torch.randn_like(x0)                     # ONE fixed noise, shared across every cell
    ts = [0, 100, 250, 500, 750, T - 1]
    schedules = [("linear", ab_lin), ("cosine", ab_cos)]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(schedules), len(ts), figsize=(len(ts) * 1.4, len(schedules) * 1.7))
    for r, (name, ab_all) in enumerate(schedules):
        for c, t in enumerate(ts):
            ab = ab_all[t]
            x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            ax = axes[r, c]
            ax.imshow(_to_img(x_t), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={t}", fontsize=9)
            ax.set_xlabel(f"√ᾱ={ab.sqrt().item():.2f}", fontsize=8)   # differs per row → per-cell
            if c == 0:
                ax.set_ylabel(name, fontsize=11)
    fig.suptitle("same digit, same ε — linear (top) vs cosine (bottom): cosine keeps it legible longer",
                 fontsize=11)
    # h_pad gives the top row's per-cell √ᾱ labels room; without it they collapse against the
    # cosine row below.
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "06_linear_vs_cosine.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"  wrote {out} — top row (linear) is mush by the middle columns; bottom row (cosine)")
    print("  keeps the digit legible far later. The Layer-2 point, now literally visible.")
    print("  Next (exp_7): SNR — repackage ᾱ as the per-t DIFFICULTY, the axis modern schedules use.")


# ---------------------------------------------------------------------------
# LAYER 7 (box): the SNR — ᾱ repackaged as the per-t DIFFICULTY.
#
# Everything so far tracked ᾱ (signal fraction) and its mirror 1-ᾱ (noise). SNR just takes their
# RATIO:
#     SNR(t) = signal power / noise power = ᾱ / (1 - ᾱ)
# (from x_t = √ᾱ·x0 + √(1-ᾱ)·ε: signal amplitude √ᾱ → power ᾱ, noise amplitude √(1-ᾱ) → power 1-ᾱ.)
# SNR is the single number for how HARD denoising is at step t: high SNR (small t) = barely noised =
# easy; low SNR (big t) = mostly noise = hard. A training step picks a random t, so the schedule
# decides how the training budget spreads across difficulties. SNR spans ~9 orders of magnitude, so
# we look at log-SNR — the coordinate modern schedules (EDM, flow matching) are actually defined in.
# ---------------------------------------------------------------------------
def exp_7_snr(T=1000):
    """Repackage ᾱ as SNR(t)=ᾱ/(1-ᾱ) = the per-t difficulty; show it in log-SNR and compare linear
    vs cosine. Cosine's log-SNR is a more even ramp; linear crams the easy end then flatlines deep
    negative — the same wasted-steps story from exp_6, in the coordinate papers actually use."""
    _banner("LAYER 7: signal-to-noise ratio — SNR(t) = ᾱ/(1-ᾱ), and log-SNR")

    _, _, ab_lin = make_linear_schedule(T=T)
    _, _, ab_cos = make_cosine_schedule(T=T)
    snr_lin = ab_lin / (1 - ab_lin)                    # signal power / noise power
    snr_cos = ab_cos / (1 - ab_cos)

    print("  SNR(t) = ᾱ / (1 - ᾱ) = signal power / noise power.")
    print("  high SNR = barely noised = EASY denoise ; low SNR = mostly noise = HARD.\n")
    print(f"  {'t':>4} | {'lin SNR':>10} {'lin logSNR':>10} | {'cos SNR':>10} {'cos logSNR':>10}")
    print(f"  {'-'*4}-+-{'-'*10}-{'-'*10}-+-{'-'*10}-{'-'*10}")
    for t in [0, 100, 250, 500, 750, 900, T - 1]:
        sl, sc = snr_lin[t].item(), snr_cos[t].item()
        lsl = math.log(sl) if sl > 0 else float("-inf")
        lsc = math.log(sc) if sc > 0 else float("-inf")
        print(f"  {t:>4} | {sl:>10.3f} {lsl:>10.2f} | {sc:>10.3f} {lsc:>10.2f}")
    print()

    # logSNR = 0 ⇔ SNR = 1 ⇔ signal power == noise power = the "halfway hard" point.
    first_lin = int((snr_lin < 1).float().argmax())    # first t past logSNR=0
    first_cos = int((snr_cos < 1).float().argmax())
    print("  Two reads:")
    print("  (1) SNR is just ᾱ in disguise (a monotonic repackaging) — but it NAMES the difficulty:")
    print("      t=0 → SNR ~1e4 (almost clean, trivial); t=T → SNR ~1e-5 (basically noise, maximal).")
    print("  (2) log-SNR is the honest axis (SNR spans ~9 orders). logSNR=0 ⇔ SNR=1 ⇔ signal power ==")
    print("      noise power = 'halfway hard'.")
    print(f"      linear crosses logSNR=0 at t≈{first_lin} (front-loads the easy end, then the tail is")
    print(f"      crammed at extreme low SNR); cosine at t≈{first_cos} (a far more even ramp).\n")

    # SEE it: log-SNR vs t, both schedules (clamp the floor so the pure-noise tail doesn't dominate).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = torch.arange(T)
    ls_lin = torch.log(snr_lin.clamp_min(1e-5))
    ls_cos = torch.log(snr_cos.clamp_min(1e-5))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ts, ls_lin, label="linear", color="tab:blue")
    ax.plot(ts, ls_cos, label="cosine", color="tab:orange")
    ax.axhline(0, color="gray", ls="--", lw=1)
    ax.annotate("logSNR=0\n(signal power == noise power)", xy=(T * 0.62, 0.3), fontsize=8, color="gray")
    ax.set_xlabel("t (noise level)"); ax.set_ylabel("log-SNR = log(ᾱ/(1-ᾱ))")
    ax.set_title("difficulty curriculum: cosine spreads it evenly, linear crams the tail")
    ax.legend()
    fig.tight_layout()
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "07_snr.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"  wrote {out} — cosine's log-SNR is a near-straight, even ramp; linear drops fast through")
    print("  the easy end then flatlines deep negative (redundant near-noise steps). Same wasted-steps")
    print("  story as exp_6, in the coordinate modern schedules place their steps along directly.")
    print("  🎉 forward_process COMPLETE — the first box of the ddpm whole game, fully opened.")


def run_experiments():
    # exp_1_dissolve()
    # exp_2_schedule()
    # exp_3_closed_form()
    # exp_4_endpoint()
    # exp_5_cosine_schedule()
    # exp_6_linear_vs_cosine()
    exp_7_snr()


if __name__ == "__main__":
    run_experiments()
