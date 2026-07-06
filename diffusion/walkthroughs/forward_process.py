"""
WALKTHROUGH: the forward (noising) process of a diffusion model, one layer at a time.

A diffusion model generates by REVERSING a process that slowly destroys an image with
noise. Before we can learn the reverse (the fun part — noise -> image), we have to
nail the forward part cold, because the training target and the sampler both fall out
of its algebra.

The forward process is dead simple: take a clean image x_0 and, over T steps, keep
adding a little Gaussian noise until nothing is left but pure static:

    x_t = sqrt(1 - beta_t) * x_{t-1}  +  sqrt(beta_t) * z_t        z_t ~ N(0, I)

The one fact that makes diffusion practical is that you DON'T have to run those T steps
to noise an image to level t. Composing all those Gaussians collapses to a single jump:

    x_t = sqrt(alpha_bar_t) * x_0  +  sqrt(1 - alpha_bar_t) * eps     eps ~ N(0, I)

where alpha_t = 1 - beta_t and alpha_bar_t = prod_{s<=t} alpha_s. So for ANY t we can
produce a training example in one line: pick t, draw eps, done. That closed form is the
spine of the whole method — the loss (predict eps), the sampler, and DDIM all reuse it.

Layers (run each `exp_*`, watch the output, then say "next"):
  1. the forward process, in three small bites:
       1a. the schedule — beta_t, alpha_t, and the cumulative alpha_bar_t; read off how
           much signal vs noise survives at each t.                                  (here)
       1b. the closed-form jump — iterative step-by-step noising == the one-line jump
           x_t = sqrt(ab) x0 + sqrt(1-ab) eps (verified in distribution by Monte Carlo).
       1c. the endpoint — at t=T, alpha_bar ~ 0, so x_T ~ N(0, I) no matter what x0 was.
  2. the schedule SHAPE, in two bites:
       2a. build the cosine schedule — set the alpha_bar curve directly as cos^2, then
           back-solve the betas (the reverse of how the linear schedule is defined).
       2b. linear vs cosine — compare the alpha_bar curves; why cosine keeps signal
           alive longer (better for small images like MNIST).
  3. signal-to-noise ratio — SNR(t) = alpha_bar / (1 - alpha_bar); log-SNR; what the
     model actually "sees" at each t.
  4. the training target lives here, in two bites:
       4a. isolate the noise — rearrange the closed form to
           eps = (x_t - sqrt(alpha_bar) x0) / sqrt(1 - alpha_bar); confirm it recovers the
           exact eps we added. At train time eps is a DETERMINED target to regress to.
       4b. eps vs x0 — predicting eps is equivalent to predicting x0 (same equation), and
           why DDPM predicts eps: it's a normalized ~N(0,1) target at every t. Seeds the loss.
  5. a training batch — draw t ~ Uniform, eps ~ N(0,I), build (x_0, t, x_t, eps) the way
     the real train loop will.
  6. SEE it — dissolve an image across t and save a grid (the picture of "add noise").

Reference for the real (torch) versions we'll build later: diffusion/schedule.py and
diffusion/diffusion.py. From-scratch verified copies go in diffusion/custom/ (like
llm/custom/). This is the SCRATCH file you learn from, then rebuild clean.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))   # diffusion/walkthroughs/


def _banner(*lines):
    print("=" * 70)
    for line in lines:
        print(line)
    print("=" * 70)


# ---------------------------------------------------------------------------
# LAYER 1a: the schedule — beta_t, alpha_t, and the cumulative alpha_bar_t.
#
# Before we can noise anything we need a SCHEDULE: how much noise to add at each of the
# T steps. From beta_t we derive two friends:
#   beta_t       how much fresh noise we mix in at step t (grows over time)
#   alpha_t      = 1 - beta_t : the fraction of signal that survives ONE step
#   alpha_bar_t  = alpha_1*...*alpha_t : the fraction of the ORIGINAL signal that
#                  survives ALL t steps so far (a cumulative product)
#
# alpha_bar_t is the star. A noised image is always  sqrt(alpha_bar_t)*x0 +
# sqrt(1-alpha_bar_t)*noise, so alpha_bar_t is the single dial from "all signal" (t=0)
# to "all noise" (t=T). This layer just BUILDS the schedule and reads that dial off at
# a few times t. (1b proves the dial really equals step-by-step noising; 1c shows where
# it ends up.)
# ---------------------------------------------------------------------------
def make_linear_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    """The original DDPM schedule: beta_t linearly spaced. Returns (betas, alphas,
    alpha_bars), each shape (T,), indexed t=0..T-1 (so 'step t' below is 1-indexed as
    t..1 conceptually, but we just use array positions)."""
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)          # prod of alphas up to each t
    return betas, alphas, alpha_bars


def exp_1a_schedule():
    """Build the linear noise schedule and READ it: for a few times t, show beta_t
    (noise added this step), alpha_t (signal surviving one step), alpha_bar_t (signal
    surviving all t steps), and the resulting signal/noise mix fractions."""
    _banner("LAYER 1a: the schedule — beta, alpha, and the cumulative alpha_bar")

    T = 1000                                           # canonical DDPM T
    betas, alphas, alpha_bars = make_linear_schedule(T=T)

    print(f"  linear schedule, T={T}, beta_t linearly from {betas[0]:.1e} to {betas[-1]:.1e}")
    print("    beta_t       = how much NOISE we mix in at step t (grows over time)")
    print("    alpha_t      = 1 - beta_t = fraction of signal surviving ONE step")
    print("    alpha_bar_t  = alpha_1*...*alpha_t = fraction of the ORIGINAL surviving t steps\n")

    print(f"  {'t':>4} | {'beta_t':>8} | {'alpha_t':>8} | {'alpha_bar_t':>12} | "
          f"{'signal √ab':>10} | {'noise √(1-ab)':>13}")
    print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*12}-+-{'-'*10}-+-{'-'*13}")
    for t in [0, 100, 250, 500, 750, T - 1]:
        b, a, ab = betas[t].item(), alphas[t].item(), alpha_bars[t].item()
        print(f"  {t:>4} | {b:>8.5f} | {a:>8.5f} | {ab:>12.5f} | {ab**0.5:>10.4f} | {(1-ab)**0.5:>13.4f}")
    print()
    print("  Read it: at t=500, alpha_t is still ~0.99 (one step barely changes the image),")
    print("  yet alpha_bar_t has collapsed to ~0.08 — the tiny per-step losses COMPOUND over")
    print("  500 multiplications. So 'signal √ab' shrinks 1 -> 0 and 'noise √(1-ab)' grows")
    print("  0 -> 1 as t rises. A noised image is:  √ab * x0  +  √(1-ab) * noise  — and")
    print("  alpha_bar_t is the single dial setting that mix.")
    print("  Next (1b): prove that dialing with alpha_bar == actually adding noise step-by-step.")


# ---------------------------------------------------------------------------
# LAYER 1b: the closed-form jump. Adding noise step-by-step T times lands in the SAME
# DISTRIBUTION as one jump  x_t = sqrt(ab) x0 + sqrt(1-ab) eps. We can't compare single
# samples (different random draws), so we Monte-Carlo it: run the iterative process many
# times to a target t and check its empirical mean/std match the closed form's. Then we
# show WHY the coefficients are those square roots: the process is variance-preserving.
# ---------------------------------------------------------------------------
def forward_iterative(x0, betas, t_idx, n_samples, generator=None):
    """Step-by-step forward process from x0 through steps 0..t_idx, run n_samples
    independent times. Each step: x <- sqrt(1-beta)*x + sqrt(beta)*z, fresh z ~ N(0,1).
    x0 is a scalar tensor; returns a (n_samples,) tensor of x_t draws."""
    x = x0.expand(n_samples).clone()
    for s in range(t_idx + 1):
        z = torch.randn(n_samples, generator=generator)
        x = torch.sqrt(1.0 - betas[s]) * x + torch.sqrt(betas[s]) * z
    return x

'''
DERIVATION — the forward closed form  x_t = √ab·x0 + √(1−ab)·ε : why the t single steps
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
  and by C it is one fresh noise √(1−α₁α₂)·ε. With ab₂ = α₁α₂:
      x_2 = √ab₂·x0 + √(1−ab₂)·ε.
  By induction over t (each step folds one more α in):
      x_t = √ab·x0 + √(1−ab)·ε,   ab = ∏_{s≤t} α_s.        [QED]

Why the coefficients are square roots — VARIANCE PRESERVATION (Var(x0)=1, x0 ⟂ ε):
      Var(x_t) = (√ab)²·Var(x_0) + (√(1−ab))²·Var(ε)
               =    ab   ·   1      +    (1−ab)   ·   1
               =    ab + (1 − ab)
               =    1                        ← for EVERY t
signal power ab + noise power (1−ab) always sum to 1, so x_t neither blows up nor
fades — it just trades signal for noise.
'''
def forward_closed_form(x0, alpha_bars, t_idx, eps):
    """The one-jump shortcut: x_t = √ab·x0 + √(1-ab)·eps. (Derivation in comment above.)"""
    ab = alpha_bars[t_idx]
    return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * eps


def exp_1b_iterative_equals_closed_form(seed=0):
    """Monte-Carlo proof that step-by-step noising and the one-jump closed form give the
    SAME distribution (matching mean sqrt(ab)*x0 and std sqrt(1-ab)), then show the
    process is variance-PRESERVING (Var stays ~1), which is why those coefficients are
    the square roots — they keep signal-power ab + noise-power (1-ab) = 1."""
    _banner("LAYER 1b: iterative noising == closed-form jump (same distribution)")
    g = torch.Generator().manual_seed(seed)

    T = 1000
    betas, _, alpha_bars = make_linear_schedule(T=T)

    # We can't compare single samples: the slow loop and the one jump each draw their own
    # random noise, so exact values differ. What must match is the DISTRIBUTION. For a
    # fixed x0 both produce a Gaussian; check its mean and std by Monte Carlo — run the
    # iterative process n times to step t, compare to the closed form's predicted
    # mean sqrt(ab)*x0 and std sqrt(1-ab).
    x0 = torch.tensor(2.0)
    n = 40000
    print("  fixed x0 = 2.0 ; noise it step-by-step 40k times to step t, compare stats to")
    print("  the closed-form prediction (mean = √ab·x0, std = √(1-ab)):\n")
    print(f"  {'t':>4} | {'iter mean':>10} {'iter std':>9} | {'√ab·x0':>8} {'√(1-ab)':>8}")
    print(f"  {'-'*4}-+-{'-'*10}-{'-'*9}-+-{'-'*8}-{'-'*8}")
    for t in [100, 500, T - 1]:
        xt = forward_iterative(x0, betas, t_idx=t, n_samples=n, generator=g)
        ab = alpha_bars[t]
        sm = (torch.sqrt(ab) * x0).item()
        ss = torch.sqrt(1 - ab).item()
        print(f"  {t:>4} | {xt.mean():>+10.4f} {xt.std():>9.4f} | {sm:>+8.4f} {ss:>8.4f}")
    print("\n  -> mean and std match. So we can SKIP the loop: to get x_t at any level,")
    print("     draw one eps and jump  x_t = √ab·x0 + √(1-ab)·eps.  (forward_closed_form)\n")

    # WHY these exact coefficients? The process is VARIANCE-PRESERVING. If the signal x0
    # has variance 1, then Var(x_t) = ab·Var(x0) + (1-ab)·Var(eps) = ab + (1-ab) = 1 for
    # every t (x0 and eps independent). The square roots are chosen precisely so signal
    # power ab and noise power (1-ab) always sum to 1 — the image never blows up or fades,
    # it just trades signal for noise. Check it: x0 ~ N(0,1), measure Var(x_t) across t.
    print("  variance preservation: x0 ~ N(0,1) (a unit-variance signal). Var(x_t) should")
    print("  stay ~1 for ALL t, because signal power ab + noise power (1-ab) = 1:\n")
    x0v = torch.randn(n, generator=g)
    print(f"  {'t':>4} | {'ab':>8} {'1-ab':>8} | {'Var(x_t)':>9}")
    print(f"  {'-'*4}-+-{'-'*8}-{'-'*8}-+-{'-'*9}")
    for t in [0, 250, 500, 750, T - 1]:
        ab = alpha_bars[t]
        eps = torch.randn(n, generator=g)
        xt = torch.sqrt(ab) * x0v + torch.sqrt(1 - ab) * eps
        print(f"  {t:>4} | {ab.item():>8.4f} {(1-ab).item():>8.4f} | {xt.var():>9.4f}")
    print("\n  -> Var stays ~1 throughout: signal power drains from 1->0, noise power fills")
    print("     0->1, always summing to 1. THAT is why the coefficients are √ab and √(1-ab).")
    print("     Next (1c): where the process ENDS — x_T ~ N(0, I), x0 fully erased.")


# ---------------------------------------------------------------------------
# LAYER 1c: the endpoint. The mean of x_t is sqrt(ab)*x0 (from 1b). As t -> T,
# alpha_bar -> 0, so that signal term sqrt(ab)*x0 vanishes and x_T -> N(0, I) no matter
# what x0 was. We noise two very different images and watch them start distinguishable
# (different means mid-way) but converge to the SAME standard-normal static by t=T. That
# universal endpoint is what lets generation START from pure noise.
# ---------------------------------------------------------------------------
def exp_1c_endpoint_is_pure_noise(seed=0):
    """Noise two very different x0's (+2.0 and -5.0) to several levels t. Mid-way their
    distributions differ (means ~ sqrt(ab)*x0); by t=T both are ~N(0,1), indistinguishable
    — x0 is fully erased. Hence the sampler can begin from a fresh scoop of N(0,1)."""
    _banner("LAYER 1c: the endpoint — x_T ~ N(0, I), x0 fully erased")
    g = torch.Generator().manual_seed(seed)

    T = 1000
    betas, _, alpha_bars = make_linear_schedule(T=T)
    n = 40000
    x0_a, x0_b = torch.tensor(2.0), torch.tensor(-5.0)

    print("  noise two very different images (x0 = +2.0 and x0 = -5.0) to a few levels t,")
    print("  each 40k times; watch their distributions converge as alpha_bar -> 0:\n")
    print(f"  {'t':>4} | {'alpha_bar':>9} | {'+2.0 mean':>9} {'std':>7} | {'-5.0 mean':>9} {'std':>7}")
    print(f"  {'-'*4}-+-{'-'*9}-+-{'-'*9}-{'-'*7}-+-{'-'*9}-{'-'*7}")
    for t in [250, 500, 750, T - 1]:
        xa = forward_iterative(x0_a, betas, t, n, generator=g)
        xb = forward_iterative(x0_b, betas, t, n, generator=g)
        ab = alpha_bars[t].item()
        print(f"  {t:>4} | {ab:>9.5f} | {xa.mean():>+9.4f} {xa.std():>7.4f} | "
              f"{xb.mean():>+9.4f} {xb.std():>7.4f}")
    print()
    print("  At t=250 the two are still far apart (means ~ sqrt(ab)*x0: +1.44 vs -3.61) — the")
    print("  image still shows through. But as alpha_bar -> 0 the signal term sqrt(ab)*x0")
    print("  vanishes, and by t=999 BOTH are ~N(0,1): mean~0, std~1, indistinguishable. The")
    print("  forward process has ERASED x0 entirely.\n")
    print("  Why this is the linchpin of generation: the endpoint is the SAME universal")
    print("  N(0, I) for every image. So to generate, grab a fresh scoop of N(0,1) static")
    print("  (free to sample) — a valid x_T for SOME image — and run the process BACKWARDS,")
    print("  denoising step by step, to land on a brand-new image. Learning that reverse is")
    print("  the next walkthrough. LAYER 1 COMPLETE: forward = signal -> universal noise.")


# ---------------------------------------------------------------------------
# LAYER 2a: the cosine schedule — set the alpha_bar CURVE, then back-solve the betas.
#
# The linear schedule (1a) sets beta_t directly and lets alpha_bar fall out as a
# cumulative product. The cosine schedule (Nichol & Dhariwal 2021, "Improved DDPM")
# flips that: it declares the SIGNAL curve it wants,
#
#   alpha_bar(t) = cos^2( ((t/T + s)/(1+s)) * pi/2 )  / normalizer     (s = 0.008)
#
# a gentle cosine falloff from 1 to 0, then RECOVERS the per-step betas from the ratio of
# consecutive alpha_bars:  beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}  (clamped < 0.999
# so the final steps don't demand more than 100% noise). This layer builds it and reads
# it; 2b compares its curve to the linear one.
# ---------------------------------------------------------------------------
def make_cosine_schedule(T=1000, s=0.008, max_beta=0.999):
    """Cosine schedule: define alpha_bar(t) as a cos^2 falloff, then back-solve betas.
    Returns (betas, alphas, alpha_bars), each shape (T,), consistent with the linear one."""
    steps = torch.arange(T + 1, dtype=torch.float32)          # 0..T (T+1 points)
    f = torch.cos(((steps / T) + s) / (1 + s) * (math.pi / 2)) ** 2
    alpha_bars_full = f / f[0]                                 # normalize so alpha_bar(0)=1
    # beta_t from the drop between consecutive alpha_bars, then clamp the tail.
    betas = (1.0 - alpha_bars_full[1:] / alpha_bars_full[:-1]).clamp(max=max_beta)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)                  # recompute from clamped betas
    return betas, alphas, alpha_bars


def exp_2a_cosine_schedule():
    """Build the cosine schedule and read it: its beta_t (small early, ramps late) and its
    alpha_bar_t curve. The point of THIS layer is the construction — alpha_bar is set
    directly as cos^2 and betas are back-solved, the reverse of the linear schedule."""
    _banner("LAYER 2a: the cosine schedule — declare alpha_bar as cos^2, back-solve betas")

    T = 1000
    betas, _, alpha_bars = make_cosine_schedule(T=T)

    print("  linear schedule (1a):  set beta_t  ->  alpha_bar = cumprod(1 - beta)  (falls out)")
    print("  cosine schedule (here): set alpha_bar = cos^2(...) ->  beta_t back-solved from it")
    print("    beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}   (clamped < 0.999 at the tail)\n")

    print(f"  {'t':>4} | {'beta_t':>8} | {'alpha_bar_t':>12} | {'signal √ab':>10} | {'noise √(1-ab)':>13}")
    print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*12}-+-{'-'*10}-+-{'-'*13}")
    for t in [0, 100, 250, 500, 750, T - 1]:
        b, ab = betas[t].item(), alpha_bars[t].item()
        print(f"  {t:>4} | {b:>8.5f} | {ab:>12.5f} | {ab**0.5:>10.4f} | {(1-ab)**0.5:>13.4f}")
    print()
    print("  beta_t is NOT linear here: it stays tiny early (keeping the image crisp longer),")
    print("  then ramps up toward the end. Same three arrays, same closed-form jump as 1b —")
    print("  only the SHAPE of the noise-over-time curve changed. Next (2b): put this")
    print("  alpha_bar curve beside the linear one and see why cosine keeps signal alive")
    print("  longer — which matters most for small images like MNIST.")


# ---------------------------------------------------------------------------
# LAYER 2b: linear vs cosine. Put the two alpha_bar curves side by side. The linear
# schedule plunges to ~0 well before t=T (many late steps are already pure noise = wasted
# capacity), while cosine descends gently and keeps usable signal much longer. Matters
# most for small images, where the linear schedule destroys structure too fast.
# ---------------------------------------------------------------------------
def exp_2b_linear_vs_cosine():
    """Compare the two schedules' alpha_bar curves across all t, and count how many steps
    each spends in the 'nearly pure noise' regime (little signal left to learn from)."""
    _banner("LAYER 2b: linear vs cosine — cosine keeps signal alive longer")

    T = 1000
    _, _, ab_lin = make_linear_schedule(T=T)
    _, _, ab_cos = make_cosine_schedule(T=T)

    print("  alpha_bar (fraction of the ORIGINAL signal still present) at each t:\n")
    print(f"  {'t':>4} | {'linear ab':>10} {'√ab':>7} | {'cosine ab':>10} {'√ab':>7}")
    print(f"  {'-'*4}-+-{'-'*10}-{'-'*7}-+-{'-'*10}-{'-'*7}")
    for t in [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, T - 1]:
        al, ac = ab_lin[t].item(), ab_cos[t].item()
        print(f"  {t:>4} | {al:>10.5f} {al**0.5:>7.4f} | {ac:>10.5f} {ac**0.5:>7.4f}")
    print()

    # "nearly pure noise" = alpha_bar so small the image is basically gone, so denoising
    # there teaches the network little. Count how many of the T steps fall in that regime,
    # and at which t each schedule first crosses into it.
    thr = 0.01                                    # alpha_bar < 0.01  <=>  signal √ab < 0.1
    waste_lin, waste_cos = int((ab_lin < thr).sum()), int((ab_cos < thr).sum())
    first_lin = int((ab_lin < thr).float().argmax())     # first t below threshold
    first_cos = int((ab_cos < thr).float().argmax())
    print(f"  'nearly pure noise' = alpha_bar < {thr} (signal √ab < {thr**0.5:.1f}):")
    print(f"    linear: crosses at t={first_lin}, so {waste_lin}/{T} steps ({waste_lin/T:.0%}) contribute little")
    print(f"    cosine: crosses at t={first_cos}, so {waste_cos}/{T} steps ({waste_cos/T:.0%}) contribute little")
    print()
    print("  Read it: linear hits near-pure-noise well before the end and spends a big chunk")
    print("  of its steps there (barely any signal to remove = barely anything to learn),")
    print("  AND it rushes through the informative middle. Cosine holds signal deep into the")
    print("  schedule, so far more of its 1000 steps sit in the useful regime where signal")
    print("  and noise coexist.\n")
    print("  Why this matters MOST for small images (MNIST 28x28): a small image has little")
    print("  redundancy, so linear's fast early destruction wipes out its structure almost")
    print("  immediately — the model barely sees lightly-noised digits. Cosine's gentle early")
    print("  descent preserves those informative low-noise steps. That's exactly why")
    print("  'Improved DDPM' switched to cosine for small-resolution data.")
    print("  LAYER 2 COMPLETE: same forward math (1a/1b), a better-shaped noise curriculum.")


# ---------------------------------------------------------------------------
# LAYER 3: signal-to-noise ratio — SNR(t) = alpha_bar / (1 - alpha_bar).
#
# Layers 1-2 gave us alpha_bar (fraction of signal) and its mirror 1-alpha_bar (noise).
# SNR just repackages the two into their RATIO:
#
#   SNR(t) = signal power / noise power = alpha_bar / (1 - alpha_bar)
#
# (from the closed form x_t = √ab·x0 + √(1-ab)·eps: signal amplitude √ab -> power ab,
# noise amplitude √(1-ab) -> power 1-ab.) SNR is the single number that says how HARD the
# denoising task is at step t: high SNR (t small) = barely noised = easy; low SNR (t big)
# = mostly noise = hard. Since a training step picks a random t, the schedule decides how
# our training budget is spread across difficulties. SNR spans many orders of magnitude
# (~1e4 down to ~1e-5), so we look at LOG-SNR — the natural coordinate modern schedules
# are written in — where the schedule becomes a near-straight ramp and we can SEE that
# cosine spreads difficulty more evenly than linear.
# ---------------------------------------------------------------------------
def exp_3_snr():
    """Repackage alpha_bar as SNR(t) = ab/(1-ab) (signal power / noise power) = the task
    difficulty at each t. Show it on a log scale (log-SNR) and compare linear vs cosine:
    cosine's log-SNR is a more even ramp, linear crams the easy end then flatlines deep
    negative — the same wasted-steps story from 2b, in the coordinate papers actually use."""
    _banner("LAYER 3: signal-to-noise ratio — SNR(t) = ab/(1-ab), and log-SNR")

    T = 1000
    _, _, ab_lin = make_linear_schedule(T=T)
    _, _, ab_cos = make_cosine_schedule(T=T)
    snr_lin = ab_lin / (1 - ab_lin)                    # signal power / noise power
    snr_cos = ab_cos / (1 - ab_cos)

    print("  SNR(t) = alpha_bar / (1 - alpha_bar) = signal power / noise power.")
    print("  high SNR = barely noised = EASY denoise ; low SNR = mostly noise = HARD.\n")
    print(f"  {'t':>4} | {'lin ab':>8} {'lin SNR':>10} {'lin logSNR':>10} | "
          f"{'cos ab':>8} {'cos SNR':>10} {'cos logSNR':>10}")
    print(f"  {'-'*4}-+-{'-'*8}-{'-'*10}-{'-'*10}-+-{'-'*8}-{'-'*10}-{'-'*10}")
    for t in [0, 100, 250, 500, 750, 900, T - 1]:
        al, ac = ab_lin[t].item(), ab_cos[t].item()
        sl, sc = snr_lin[t].item(), snr_cos[t].item()
        lsl = math.log(sl) if sl > 0 else float("-inf")
        lsc = math.log(sc) if sc > 0 else float("-inf")
        print(f"  {t:>4} | {al:>8.5f} {sl:>10.3f} {lsl:>10.2f} | "
              f"{ac:>8.5f} {sc:>10.3f} {lsc:>10.2f}")
    print()
    print("  Two reads:")
    print("  (1) SNR is just alpha_bar in disguise — a monotonic repackaging — but it's the")
    print("      form that names the DIFFICULTY directly: at t=0 SNR is ~1e4 (almost clean,")
    print("      trivial), by t=T it's ~1e-5 (basically noise, maximally ambiguous).")
    print("  (2) log-SNR is the honest axis: SNR crosses ~9 orders of magnitude, so only in")
    print("      log does the schedule look like a smooth ramp. Notice logSNR passes through")
    print("      0 (SNR=1, signal power == noise power) — the 'halfway hard' point.\n")
    print("  Linear vs cosine in log-SNR: cosine's logSNR falls in a more EVEN, near-linear")
    print("  ramp, so training budget spreads evenly across difficulties. Linear drops fast")
    print("  through the easy (high-SNR) end, then spends its whole tail crammed at extreme")
    print("  low SNR (redundant, near-identical hard steps) — the SAME wasted-steps story")
    print("  from 2b, now in the coordinate modern schedules (EDM, etc.) are DEFINED in:")
    print("  they place steps directly along log-SNR instead of choosing betas at all.")
    print("  LAYER 3 COMPLETE: alpha_bar, repackaged as the difficulty curriculum the model sees.")


# ---------------------------------------------------------------------------
# LAYER 4a: isolate the noise. The closed form x_t = sqrt(ab)*x0 + sqrt(1-ab)*eps is ONE
# equation linking three things (x_t, x0, eps) at a given t. Solve it for eps:
#
#   eps = (x_t - sqrt(ab)*x0) / sqrt(1-ab)
#
# So given (x_t, x0, t) the noise eps is fully DETERMINED. That's what lets us build a
# supervised target: at training time we HAVE x0 and eps (we drew them), so we can form
# x_t and know the exact eps the network should output from (x_t, t). This layer does the
# rearrange and confirms it recovers eps (and, same equation solved the other way, x0).
# ---------------------------------------------------------------------------
def exp_4a_isolate_eps():
    """Rearrange the closed form to isolate eps = (x_t - sqrt(ab)*x0)/sqrt(1-ab), and
    confirm it exactly recovers the noise we added (and x0, symmetrically). The point:
    eps is a determined function of (x_t, x0, t) — the supervised target for training."""
    _banner("LAYER 4a: isolate the noise — eps = (x_t - √ab·x0) / √(1-ab)")

    T = 1000
    _, _, alpha_bars = make_linear_schedule(T=T)
    g = torch.Generator().manual_seed(0)

    # a little "image" x0 (8 pixels), a chosen noise level t, and the exact noise eps.
    x0 = torch.randn(8, generator=g)
    t = 400
    eps = torch.randn(8, generator=g)
    ab = alpha_bars[t]

    # forward: build the noised x_t (the closed form from 1b).
    x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps
    # rearrange to recover eps; and the SAME equation solved for the other unknown, x0.
    eps_rec = (x_t - torch.sqrt(ab) * x0) / torch.sqrt(1 - ab)
    x0_rec = (x_t - torch.sqrt(1 - ab) * eps) / torch.sqrt(ab)

    print(f"  t={t}, alpha_bar={ab.item():.5f}. Built x_t = √ab·x0 + √(1-ab)·eps, solved back:\n")
    print(f"  {'i':>2} | {'x0':>8} {'eps':>8} {'x_t':>8} | {'eps_rec':>8} {'x0_rec':>8}")
    print(f"  {'-'*2}-+-{'-'*8}-{'-'*8}-{'-'*8}-+-{'-'*8}-{'-'*8}")
    for i in range(8):
        print(f"  {i:>2} | {x0[i]:>+8.4f} {eps[i]:>+8.4f} {x_t[i]:>+8.4f} | "
              f"{eps_rec[i]:>+8.4f} {x0_rec[i]:>+8.4f}")
    print()
    print(f"  max |eps_rec - eps| = {(eps_rec - eps).abs().max():.2e}    "
          f"max |x0_rec - x0| = {(x0_rec - x0).abs().max():.2e}")
    print("  -> exact recovery. The closed form ties x_t, x0, eps together at a given t;")
    print("  fix any two and the third is pinned. So eps is a determined function of")
    print("  (x_t, x0, t).\n")
    print("  Why it matters: at TRAINING time we drew x0 and eps ourselves, so we can build")
    print("  x_t AND know the exact eps target. The network sees only (x_t, t) and learns to")
    print("  predict that eps. Next (4b): predicting eps is equivalent to predicting x0, and")
    print("  why eps is the target DDPM actually regresses to.")


# ---------------------------------------------------------------------------
# LAYER 4b: eps vs x0. Given (x_t, t), the 4a equations convert an eps-prediction into an
# x0-prediction and back — so they are INTERCHANGEABLE targets carrying identical info;
# "which to output" is a design choice. They differ only in LOSS WEIGHTING across noise
# levels: a plain MSE on eps equals an SNR(t)-weighted MSE on x0 (MSE_eps = SNR * MSE_x0).
# DDPM predicts eps with unweighted MSE (the "simple" loss) — which also makes the target
# ~N(0,1) at every t — and this empirically gives better samples. Seeds the DDPM loss.
# ---------------------------------------------------------------------------
'''
DERIVATION — MSE_eps = SNR(t)·MSE_x0.  From the x0-recovery formula (closed form solved
for x0), true for the real ε and any ε_hat since x_t, t are fixed:
    x0     = (x_t − √(1−ab)·ε    ) / √ab
    x0_hat = (x_t − √(1−ab)·ε_hat) / √ab
Subtract; the x_t/√ab terms cancel:
    x0_hat − x0 = −√((1−ab)/ab)·(ε_hat − ε).
Square and average over the batch (the constant scale √((1−ab)/ab) comes out squared):
    MSE_x0 = mean( (x0_hat − x0)² )
           = (1−ab)/ab · mean( (ε_hat − ε)² )
           = (1−ab)/ab · MSE_eps.
Since SNR = ab/(1−ab) ⇒ (1−ab)/ab = 1/SNR:
    MSE_x0 = MSE_eps/SNR  ⇒  MSE_eps = SNR(t)·MSE_x0.   [QED]
The two errors are the SAME vector rescaled by 1/√SNR, so their MSEs differ by SNR.
'''
def exp_4b_eps_vs_x0():
    """(1) eps- and x0-prediction are interchangeable (convert via the 4a formulas given
    (x_t,t)). (2) They differ only by a per-t factor: MSE_eps = SNR(t) * MSE_x0, so
    'predict eps with plain MSE' = 'predict x0 with SNR-weighted MSE'. DDPM picks eps.
    (Derivation in comment above.)"""
    _banner("LAYER 4b: eps vs x0 — interchangeable targets, different loss weighting")

    T = 1000
    _, _, alpha_bars = make_linear_schedule(T=T)
    g = torch.Generator().manual_seed(0)

    n = 4096
    x0 = torch.randn(n, generator=g)                      # data ~ N(0,1) (normalized pixels)

    def eps_to_x0(x_t, eps_hat, ab):                      # the 4a equation, solved for x0
        return (x_t - torch.sqrt(1 - ab) * eps_hat) / torch.sqrt(ab)

    # (1) interchangeability: feed the TRUE eps through eps->x0 and get the TRUE x0 back.
    t = 500
    ab = alpha_bars[t]
    eps = torch.randn(n, generator=g)
    x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps
    x0_from_eps = eps_to_x0(x_t, eps, ab)
    print("  (1) interchangeable: given (x_t, t), the 4a formulas turn one prediction into")
    print("      the other. Passing the true eps through eps->x0 returns the true x0:")
    print(f"      max |x0_from_eps - x0| = {(x0_from_eps - x0).abs().max():.2e}  (deterministic, exact)")
    print("      => an eps-predictor IS an x0-predictor; 'which to output' is a design choice.\n")

    # (2) same info, different loss SCALE. An eps error D implies an x0 error D*sqrt((1-ab)/ab)
    # = D / sqrt(SNR), so the squared errors relate by  MSE_eps = SNR(t) * MSE_x0.
    print("  (2) same info, different LOSS SCALE. An eps error D implies an x0 error")
    print("      D*sqrt((1-ab)/ab), so  MSE_eps = SNR(t) * MSE_x0.  Verify with a noisy")
    print("      'prediction' eps_hat = eps + small noise, converted to x0_hat:\n")
    print(f"  {'t':>4} | {'SNR(t)':>10} | {'MSE_eps':>10} {'MSE_x0':>10} {'ratio':>10}")
    print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-{'-'*10}-{'-'*10}")
    for t in [100, 300, 500, 700, 900]:
        ab = alpha_bars[t]
        eps = torch.randn(n, generator=g)
        x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps
        eps_hat = eps + 0.1 * torch.randn(n, generator=g)         # a noisy prediction
        x0_hat = eps_to_x0(x_t, eps_hat, ab)                      # its implied x0 prediction
        mse_eps = ((eps_hat - eps) ** 2).mean().item()
        mse_x0 = ((x0_hat - x0) ** 2).mean().item()
        snr = (ab / (1 - ab)).item()
        print(f"  {t:>4} | {snr:>10.4f} | {mse_eps:>10.4f} {mse_x0:>10.4f} {mse_eps / mse_x0:>10.4f}")
    print()
    print("  ratio == SNR(t): predicting eps with a plain MSE is the SAME objective as")
    print("  predicting x0 with an SNR-weighted MSE — the choice is a per-noise-level")
    print("  weighting, not a different task. DDPM predicts eps with UNWEIGHTED MSE (the")
    print("  'simple' loss); vs the true ELBO this downweights the low-t/high-SNR terms, and")
    print("  empirically gives better samples. Bonus: eps ~ N(0,1) at EVERY t, so the target")
    print("  is scale-stable across noise levels — nice for optimization.")
    print("  LAYER 4 COMPLETE: the target is eps; MSE(eps_pred, eps) is the DDPM loss.")


# ---------------------------------------------------------------------------
# LAYER 5: a training batch. Everything above was about ONE example / one t. The real
# train loop batches it: take a batch of clean images x0, draw a random noise level t PER
# EXAMPLE, draw eps ~ N(0,I), and build x_t with the closed form (broadcasting alpha_bar
# over the pixel dims). The supervised pair is inputs (x_t, t) -> target eps. Two practical
# points live here: per-example t (one batch spans many noise levels) and the alpha_bar
# reshape/broadcast (the classic diffusion batch bug if you forget it).
# ---------------------------------------------------------------------------
def exp_5_batch():
    """Assemble a training batch as train.py will: x0 batch -> random per-example t ->
    eps ~ N(0,I) -> x_t (closed form, broadcasting alpha_bar over pixels). The pair fed to
    the denoiser is (x_t, t) -> eps; it never sees x0 or alpha_bar."""
    _banner("LAYER 5: a training batch — (x0, t, eps) -> x_t, packaged as (x_t, t) -> eps")

    T = 1000
    _, _, alpha_bars = make_linear_schedule(T=T)
    g = torch.Generator().manual_seed(0)

    B, D = 6, 8                                    # 6 "images" of 8 pixels (real: MNIST 1x28x28)
    x0 = torch.randn(B, D, generator=g)            # a batch of clean data

    # (1) one random noise level PER EXAMPLE (not a single t shared by the whole batch).
    t = torch.randint(0, T, (B,), generator=g)     # (B,)
    eps = torch.randn(B, D, generator=g)           # fresh noise, same shape as x0

    # (2) gather alpha_bar per example, reshape to broadcast over the pixel dim, jump.
    ab = alpha_bars[t].view(B, 1)                  # (B,) -> (B,1) so it scales all D pixels
    x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps   # (B,D), each row noised at ITS t

    print(f"  x0 {tuple(x0.shape)}, per-example t {tuple(t.shape)}, eps {tuple(eps.shape)}, "
          f"alpha_bar[t] {tuple(alpha_bars[t].shape)} -> view (B,1) to broadcast\n")
    print(f"  {'ex':>2} | {'t':>4} {'alpha_bar':>10} | {'√ab':>6} {'√(1-ab)':>8} | first 4 pixels of x_t")
    print(f"  {'-'*2}-+-{'-'*4}-{'-'*10}-+-{'-'*6}-{'-'*8}-+-{'-'*24}")
    for i in range(B):
        abi = alpha_bars[t[i]].item()
        row = " ".join(f"{v:+.2f}" for v in x_t[i][:4])
        print(f"  {i:>2} | {t[i].item():>4} {abi:>10.5f} | {abi**0.5:>6.3f} {(1-abi)**0.5:>8.3f} | {row} ...")
    print()
    print("  The training pair is:  inputs (x_t, t)  ->  target eps.  The network never sees")
    print("  x0 or alpha_bar; it predicts eps from the noised image and its timestep alone.\n")
    print("  Two things the real train loop relies on here:")
    print("  (1) PER-EXAMPLE t — one batch spans many noise levels, so every step trains all")
    print("      difficulties together (balanced, low-variance gradients; see 4b).")
    print("  (2) BROADCASTING — alpha_bar is ONE scalar per image; reshape (B,1) here, (B,1,1,1)")
    print("      for real images, so it scales every pixel/channel of THAT image at ITS t.")
    print("      Forgetting the reshape = shape error or wrong per-pixel scaling (classic bug).")
    print("  LAYER 5 COMPLETE: this (x_t, t, eps) triple is exactly what the denoiser trains on.")


# ---------------------------------------------------------------------------
# LAYER 6: SEE it. Everything above was numbers; this layer makes the forward process a
# PICTURE. Load a real MNIST digit, normalize to [-1,1], and apply the closed form
# x_t = √ab·x0 + √(1-ab)·ε at a row of t values — with the SAME ε everywhere so the only
# thing changing is the noise level. Do it for BOTH schedules so the Layer-2 point ("cosine
# keeps signal alive longer") becomes literally visible. Needs numpy + matplotlib (MNIST is
# pulled once as a small npz — no torchvision).
# ---------------------------------------------------------------------------
def _load_mnist_digit(label=7, index=0):
    """Load one MNIST image (28x28) normalized to [-1,1] from a cached npz, downloading it
    on first use. label = which digit; index = which occurrence of that digit."""
    import numpy as np
    data_dir = os.path.join(os.path.dirname(_HERE), "data")   # diffusion/data
    path = os.path.join(data_dir, "mnist.npz")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        import urllib.request
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        print(f"  downloading MNIST npz (~11MB) -> {path} ...")
        urllib.request.urlretrieve(url, path)
    d = np.load(path)
    x, y = d["x_train"], d["y_train"]
    img = x[np.where(y == label)[0][index]] if label is not None else x[index]
    return torch.from_numpy(img).float() / 127.5 - 1.0        # uint8 [0,255] -> [-1,1]


def exp_6_visualize(seed=0):
    """Dissolve a real MNIST digit across t with the closed form, for BOTH schedules, and
    save a grid (rows = linear vs cosine, cols = t). Same eps in every cell, so any
    difference is purely the schedule — the Layer-2 story, made visible."""
    _banner("LAYER 6: SEE it — a real digit dissolving into noise across t")
    import matplotlib
    matplotlib.use("Agg")                                     # headless: save, don't show
    import matplotlib.pyplot as plt

    torch.manual_seed(seed)
    T = 1000
    _, _, ab_lin = make_linear_schedule(T=T)
    _, _, ab_cos = make_cosine_schedule(T=T)

    x0 = _load_mnist_digit(label=7)                           # a '7', (28,28) in [-1,1]
    eps = torch.randn_like(x0)                                # ONE fixed noise, shared everywhere
    ts = [0, 100, 250, 500, 750, 999]
    schedules = [("linear", ab_lin), ("cosine", ab_cos)]

    fig, axes = plt.subplots(len(schedules), len(ts),
                             figsize=(len(ts) * 1.6, len(schedules) * 2.1))
    for r, (name, ab_all) in enumerate(schedules):
        for c, t in enumerate(ts):
            ab = ab_all[t]
            x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps          # closed form
            disp = ((x_t.clamp(-1, 1) + 1) / 2).numpy()                   # -> [0,1] to show
            ax = axes[r, c]
            ax.imshow(disp, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={t}", fontsize=9)                        # shared column header
            ax.set_xlabel(f"√ab={ab.item()**0.5:.2f}", fontsize=8)        # per-cell: differs per row
            if c == 0:
                ax.set_ylabel(name, fontsize=11)
    fig.suptitle("forward process:  x_t = √ab·x0 + √(1−ab)·ε   (same ε in every cell)", fontsize=10)
    fig.tight_layout()

    out_dir = os.path.join(_HERE, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "forward_dissolve.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"  loaded a real MNIST '7' (28x28), noised with the SAME eps at t={ts},")
    print(f"  for both schedules (closed form x_t = √ab·x0 + √(1−ab)·ε).")
    print(f"  saved grid -> {out}\n")
    print(f"  {'schedule':>8} | " + " ".join(f"t={t:<5}" for t in ts))
    for name, ab_all in schedules:
        cells = " ".join(f"{ab_all[t].item()**0.5:6.2f}" for t in ts)
        print(f"  {name:>8} | {cells}   (signal fraction √ab per cell)")
    print()
    print("  Open the PNG: the top row (linear) is mush by the middle columns, while the")
    print("  bottom row (cosine) keeps the '7' legible much later — the Layer-2 point, now")
    print("  literally visible. This is the whole forward process in one picture; every")
    print("  later layer/walkthrough LEARNS TO REVERSE it. forward_process.py COMPLETE.")


def run_experiments():
    # exp_1a_schedule()
    # exp_1b_iterative_equals_closed_form()
    # exp_1c_endpoint_is_pure_noise()
    # exp_2a_cosine_schedule()
    # exp_2b_linear_vs_cosine()
    # exp_3_snr()
    # exp_4a_isolate_eps()
    # exp_4b_eps_vs_x0()
    # exp_5_batch()
    exp_6_visualize()


@contextlib.contextmanager
def _tee(path):
    """Print to BOTH the terminal and `path` (long runs survive scrollback)."""
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)

        def flush(self):
            for st in self.streams:
                st.flush()

    with open(path, "w") as f:
        with contextlib.redirect_stdout(_Tee(sys.stdout, f)):
            yield
    print(f"(output also written to {path})", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", metavar="FILE", help="also write all output to FILE")
    args = parser.parse_args()

    if args.out:
        with _tee(args.out):
            run_experiments()
    else:
        run_experiments()
