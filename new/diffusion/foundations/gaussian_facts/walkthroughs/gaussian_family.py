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
  2. HISTOGRAM & DENSITY: build a histogram BY HAND — bucketing, counts→density (÷ N·width), area=1.
  3. REPARAMETERIZATION:  X = μ + σ·ε with ε~N(0,1) — the identity behind diffusion & VAEs.
  4. AFFINE stays Gaussian:  aX+b ~ N(aμ+b, a²σ²) — same bell, predictable parameters.
  5. CLOSURE: sum of INDEPENDENT Gaussians is Gaussian — and why that's special (uniforms aren't).
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
    print("  one onto the single standard normal N(0,1). Next (exp_2): where those density CURVES and")
    print("  histograms come from — building one by hand from raw sample counts.")


# ---------------------------------------------------------------------------
# exp_2: HISTOGRAM & DENSITY — build the y-axis of exp_1 BY HAND.
#
# exp_1 drew smooth density curves and filled histograms and asserted they line up. Here we earn that:
# a histogram is nothing but "chop the x-axis into bins, count how many samples fall in each," and
# DENSITY is those counts put on a scale that doesn't depend on how many samples or how wide the bins:
#   bin index of a sample x:  floor((x - lo) / width)        (which slot does it land in)
#   count_i                 = how many samples landed in bin i
#   density_i               = count_i / (N · width)          ← THE normalization matplotlib's
#                                                              density=True applies for you
# The payoff — a probability is an AREA, and the total area is 1:
#   Σ_i  density_i · width  =  (Σ_i count_i) / N  =  1
# Dividing by N kills the sample-count dependence; dividing by width kills the bin-width dependence,
# so density is comparable across runs and across bin choices — which is exactly why exp_1 could plot
# a measured histogram and an analytic PDF on the SAME axis. Bin width is a resolution knob (too few
# → blocky and the peak is understated; too many → noisy), but the area is 1 no matter what.
# ---------------------------------------------------------------------------
def exp_2_histogram_density(seed=0):
    """Build a histogram from scratch: bucket samples by floor((x-lo)/width), count per bin, then turn
    counts into density via ÷(N·width) — the exact thing `density=True` does. Verify Σ density·width = 1,
    match the hand-rolled bars to matplotlib's hist AND the analytic PDF, and show bin width is a
    resolution knob that never changes the unit area. Figure: counts vs density (same bars, two rulers)
    + a bin-width sweep."""
    _banner("SECTION 2 · exp_2: HISTOGRAM & DENSITY — counts → density (÷ N·width), area = 1")

    torch.manual_seed(seed)
    N = 200_000
    mu, sigma = 2.0, 0.5                              # reuse exp_1's narrow bell N(2, 0.25)
    X = torch.normal(mean=mu, std=sigma, size=(N,))

    lo, hi, nbins = 0.0, 4.0, 20                      # μ ± 4σ covers essentially all the mass
    width = (hi - lo) / nbins
    print(f"  X ~ N({mu}, {sigma**2}),  N={N:,} samples.  Bin the range [{lo}, {hi}] into {nbins} "
          f"bins of width {width}.\n")

    # STEP 1 — bucket & count, the whole of "a histogram", by hand.
    bin_of = ((X - lo) / width).floor().long()        # which bin each sample lands in
    inside = (bin_of >= 0) & (bin_of < nbins)         # drop the (few) samples outside [lo, hi]
    counts = torch.bincount(bin_of[inside], minlength=nbins).float()
    centers = lo + (torch.arange(nbins) + 0.5) * width
    Nin = counts.sum()                                # in-range sample count (≈ N here)

    # STEP 2 — counts → density, and the area-is-1 payoff.
    density = counts / (Nin * width)                  # ← THE formula density=True applies
    area = (density * width).sum().item()

    print("  a few bins around the peak (counts are raw tallies; density = count / (N·width)):")
    print(f"    {'bin center':>10} | {'count':>7} | {'count/N':>8} | {'density = count/(N·width)':>26}")
    print(f"    {'-'*10}-+-{'-'*7}-+-{'-'*8}-+-{'-'*26}")
    for i in range(6, 14):                            # bins straddling the peak at x=2
        print(f"    {centers[i]:>10.2f} | {int(counts[i]):>7d} | {counts[i]/Nin:>8.4f} | "
              f"{density[i]:>26.4f}")
    print(f"\n  counts depend on N and width; DENSITY doesn't. And the area is exactly 1:")
    print(f"    Σ density·width = {area:.4f}   (= Σcount / N = {int(counts.sum())}/{int(Nin)} = 1)\n")

    # STEP 3 — three-way agreement: hand-rolled density == matplotlib density=True == analytic PDF.
    peak_center = centers[density.argmax()].item()
    print("  three independent computations of the peak-bin density agree:")
    print(f"    hand-rolled   density[peak] = {density.max():.4f}   (at x≈{peak_center:.2f})")
    print(f"    analytic PDF  N({peak_center:.1f};μ,σ²) = {_normal_pdf(torch.tensor(peak_center), mu, sigma):.4f}")
    print(f"    (matplotlib's density=True computes the SAME count/(N·width) — plotted in the figure)\n")

    # STEP 4 — bin width is a RESOLUTION knob; area stays 1 regardless.
    print("  bin width is a resolution knob — area stays 1, only the detail changes:")
    for nb in [8, 30, 120, 600]:
        w = (hi - lo) / nb
        b = ((X - lo) / w).floor().long()
        ins = (b >= 0) & (b < nb)
        c = torch.bincount(b[ins], minlength=nb).float()
        d = c / (c.sum() * w)
        print(f"    {nb:>4} bins (width {w:.3f}):  peak density {d.max():>6.3f}   area {(d*w).sum():.4f}")
    print("    too FEW bins → blocky, peak UNDERSTATED; too MANY → spiky noise; area always 1.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = torch.linspace(lo, hi, 400)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # LEFT: the SAME bars, two rulers — left axis = counts, right axis = density (count ÷ N·width).
    axL.bar(centers.numpy(), counts.numpy(), width=width * 0.92, color="tab:orange", alpha=0.55,
            edgecolor="white", linewidth=0.5)
    axL.set_ylabel("count  (raw tally)"); axL.set_xlabel("x"); axL.set_ylim(0, counts.max().item() * 1.12)
    axT = axL.twinx()                                 # second ruler on the SAME bars
    axT.plot(grid.numpy(), _normal_pdf(grid, mu, sigma).numpy(), color="black", lw=2,
             label="analytic density N(2,0.25)")
    axT.set_ylabel("density  (= count / (N·width))")
    axT.set_ylim(0, counts.max().item() * 1.12 / (Nin.item() * width))   # align the two rulers
    axT.legend(loc="upper right", fontsize=8)
    axL.set_title("same bars, two rulers: counts (left) vs density (right)")

    # RIGHT: bin-width sweep — coarse→fine, all integrating to area 1 under the same PDF.
    for nb, c in zip([8, 30, 120, 600], ["tab:red", "tab:green", "tab:blue", "tab:gray"]):
        axR.hist(X.numpy(), bins=nb, range=(lo, hi), density=True, histtype="step", lw=1.5,
                 color=c, label=f"{nb} bins")
    axR.plot(grid.numpy(), _normal_pdf(grid, mu, sigma).numpy(), color="black", lw=2.2, label="PDF")
    axR.set_title("bin width = resolution knob (area always 1)"); axR.set_xlabel("x")
    axR.set_ylabel("density"); axR.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "histogram_density.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out} — left: ONE set of bars read two ways (÷N·width turns counts into density,")
    print("  matching the PDF); right: coarse→fine bins, all area 1. Next (exp_3): reparameterization —")
    print("  sampling any Gaussian from one fixed ε ~ N(0,1), the z-score of exp_1 run forwards.")


def run_experiments():
    # exp_1_normal()
    exp_2_histogram_density()


if __name__ == "__main__":
    run_experiments()
