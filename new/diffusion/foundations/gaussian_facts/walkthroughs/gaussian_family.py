"""
FOUNDATIONS · Section 2: THE GAUSSIAN FAMILY.

(See ../roadmap.md.) Section 1 built the mean/variance algebra for ANY distribution. Now we
specialize to the NORMAL (Gaussian) and collect the payoffs the diffusion algebra runs on. The
linear-op laws from Section 1 still hold — but the Gaussian adds a gift no other distribution has:
it stays Gaussian under scaling, shifting, and (independent) summing. Section 3 puts it to work.

The forward diffusion step  x_s = √α·x_{s-1} + √(1-α)·z  is exactly "scale a Gaussian, add an
independent Gaussian" — so by the end of this section that step is not a black box, it's closure.

Layers (each an `exp_*`; run it, read it, say "next"):
  1. THE NORMAL: μ & σ² are the whole story, the bell curve, standard normal N(0,1), z-scores.  (here)
  2. REPARAMETERIZATION:  X = μ + σ·ε with ε~N(0,1) — the identity behind diffusion & VAEs.
  3. AFFINE stays Gaussian:  aX+b ~ N(aμ+b, a²σ²) — same bell, predictable parameters.
  4. CLOSURE: sum of INDEPENDENT Gaussians is Gaussian — and why that's special (uniforms aren't).
"""
from __future__ import annotations

import math
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))                 # .../gaussian_facts/walkthroughs
_FIGS = os.path.join(_HERE, "figures", "experiments")


def _banner(*lines):
    print("=" * 70)
    for line in lines:
        print(line)
    print("=" * 70)


def _normal_pdf(x, mu=0.0, sigma=1.0):
    """The Gaussian density  N(x; μ, σ²) = exp(-½((x-μ)/σ)²) / (σ·√(2π))."""
    z = (x - mu) / sigma
    return torch.exp(-0.5 * z ** 2) / (sigma * math.sqrt(2 * math.pi))


# ---------------------------------------------------------------------------
# exp_1: THE NORMAL DISTRIBUTION — two numbers, the bell, N(0,1), and z-scores.
#
# A NORMAL (Gaussian) random variable is the famous bell curve. Its entire shape is pinned down by
# just TWO numbers — the same mean and variance from Section 1:
#   N(μ, σ²):   μ  = the mean  = where the bell is CENTERED
#               σ² = the variance = how WIDE the bell is (σ = std = the spread in original units)
# The density (height of the bell at x) is
#   N(x; μ, σ²) = exp( -½·((x-μ)/σ)² ) / (σ·√(2π))
# — a peak at μ falling off symmetrically, faster when σ is small. No other parameters: fix μ and σ
# and you've fixed EVERYTHING about a Gaussian (skew 0, kurtosis 3, all higher structure determined).
#
# The STANDARD NORMAL N(0,1) (μ=0, σ=1) is the reference bell. Every other Gaussian is just this one
# shifted and stretched — which is why a single operation, the Z-SCORE, maps ANY normal onto it:
#   z = (X - μ) / σ   →   z ~ N(0,1)     (subtract the center, divide by the spread)
# This is Section 1's affine law read backwards: E[z]=0, Var(z)=1 by the ×a / +d rules. And it comes
# with the "empirical rule": for a normal, 68% of samples land within 1σ, 95% within 2σ, 99.7% in 3σ.
# ---------------------------------------------------------------------------
def exp_1_normal(seed=0):
    """The normal N(μ,σ²): μ centers the bell, σ² sets its width — and that's ALL of it. The standard
    normal N(0,1) is the reference bell; z=(X-μ)/σ standardizes ANY normal onto it (mean 0, var 1),
    with 68/95/99.7% of mass within 1/2/3σ. Figure: bells for different (μ,σ), and z-scoring them all
    collapsing onto the SAME N(0,1)."""
    _banner("SECTION 2 · exp_1: THE NORMAL — μ, σ², the bell, N(0,1), and z-scores")

    torch.manual_seed(seed)
    n = 500_000

    print("  a NORMAL is described by exactly TWO numbers — the mean and variance from Section 1:")
    print("    N(μ, σ²):  μ = center of the bell,  σ² = its spread (σ = std).")
    print("    density   N(x;μ,σ²) = exp(-½((x-μ)/σ)²) / (σ·√(2π))\n")

    cases = [(0.0, 1.0), (2.0, 0.5), (-1.0, 1.5)]     # (μ, σ): the reference + a narrow + a wide
    print(f"  {'distribution':<14} | {'μ (mean)':>16} | {'σ² (variance)':>18}")
    print(f"  {'-'*14}-+-{'-'*16}-+-{'-'*18}")
    samples = {}
    for mu, sigma in cases:
        X = torch.normal(mean=mu, std=sigma, size=(n,))
        samples[(mu, sigma)] = X
        tag = "  ← N(0,1)" if (mu, sigma) == (0.0, 1.0) else ""
        print(f"  N({mu:>+.1f}, {sigma**2:>.2f})    | true {mu:>+.2f}  meas {X.mean():>+.3f} | "
              f"true {sigma**2:>.3f} meas {X.var():>.3f}{tag}")
    print("\n  μ and σ² ARE literally the mean and variance — measuring the samples recovers them.\n")

    # z-SCORE: map any normal onto the standard normal by (X-μ)/σ. Verify mean→0, var→1.
    print("  z-score  z = (X-μ)/σ  maps ANY normal onto the standard normal N(0,1):")
    print(f"    {'from':<12} | {'mean(z)':>8} | {'var(z)':>7}")
    print(f"    {'-'*12}-+-{'-'*8}-+-{'-'*7}")
    for (mu, sigma), X in samples.items():
        z = (X - mu) / sigma
        print(f"    N({mu:>+.1f},{sigma**2:>.2f})  | {z.mean():>+8.3f} | {z.var():>7.3f}")
    print("    → all collapse to mean 0, var 1. (This is Section 1's ×a/+d laws run backwards.)\n")

    # the 68-95-99.7 EMPIRICAL RULE, measured on the standard normal.
    zref = (samples[(0.0, 1.0)])
    print("  empirical rule — fraction of a normal's mass within k·σ of the mean:")
    for k, target in [(1, 68.27), (2, 95.45), (3, 99.73)]:
        frac = (zref.abs() < k).float().mean().item() * 100
        print(f"    within {k}σ:  measured {frac:>5.2f}%   (theory {target:.2f}%)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # LEFT: three different bells — μ slides the peak, σ sets the width. PDF curves + one histogram.
    grid = torch.linspace(-6, 6, 400)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for (mu, sigma), c in zip(cases, colors):
        axL.plot(grid.numpy(), _normal_pdf(grid, mu, sigma).numpy(), color=c, lw=2,
                 label=f"N({mu:+.0f}, {sigma**2:.2f})")
    axL.hist(samples[(2.0, 0.5)].numpy(), bins=140, density=True, alpha=0.3, color="tab:orange")
    axL.set_title("μ centers the bell, σ sets its width"); axL.set_xlabel("x"); axL.set_ylabel("density")
    axL.set_xlim(-6, 6); axL.legend()

    # RIGHT: z-score every one of them → they land on the SAME standard normal, with 68/95/99.7 bands.
    for (mu, sigma), c in zip(cases, colors):
        z = ((samples[(mu, sigma)] - mu) / sigma)
        axR.hist(z.numpy(), bins=140, density=True, alpha=0.35, color=c,
                 label=f"z of N({mu:+.0f},{sigma**2:.2f})")
    axR.plot(grid.numpy(), _normal_pdf(grid).numpy(), color="black", lw=2, label="N(0,1)")
    for k, a in [(1, 0.18), (2, 0.10), (3, 0.05)]:
        axR.axvspan(-k, k, color="gray", alpha=a)
    axR.set_title("z=(X-μ)/σ collapses them ALL onto N(0,1)\nshaded: 68% / 95% / 99.7% within 1/2/3σ")
    axR.set_xlabel("z"); axR.set_xlim(-5, 5); axR.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "normal_and_zscore.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out} — left: three bells (μ shifts, σ widens); right: z-scoring collapses every")
    print("  one onto the single standard normal N(0,1). Next (exp_2): reparameterization — running")
    print("  that collapse BACKWARDS to sample any Gaussian from one fixed ε ~ N(0,1).")


def run_experiments():
    exp_1_normal()


if __name__ == "__main__":
    run_experiments()
