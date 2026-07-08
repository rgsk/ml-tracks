"""
FOUNDATIONS · Section 1: MEAN & VARIANCE under linear operations.

(See ../roadmap.md.) Bottom-up probability background: how the MEAN and VARIANCE of a random
variable change when you scale it, shift it, or add two together. NO Gaussian yet — everything here
holds for ANY distribution. Section 2 specializes to the Gaussian; Section 3 puts it to work.

This is the algebra the diffusion forward step  x_s = √α·x_{s-1} + √(1-α)·z  leans on, learned from
the ground up so nothing is a black box.

Layers (each an `exp_*`; run it, read it, say "next"):
  1. MEAN under scaling/shift:  E[aX] = a·E[X],  E[X+d] = E[X]+d.  (both LINEAR)              (here)
  2. VARIANCE under scaling/shift:  Var(aX) = a²·Var(X)  (std ×|a|),  Var(X+d) = Var(X).
  3. SUMS — linearity of expectation:  E[aX+bY] = aE[X]+bE[Y]  (always, even dependent).
  4. VARIANCE OF SUMS + covariance:  Var(X+Y)=VarX+VarY+2Cov;  independent ⇒ variances ADD.
"""
from __future__ import annotations

import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))                 # .../gaussian_facts/walkthroughs
_FIGS = os.path.join(_HERE, "figures", "experiments")


def _banner(*lines):
    print("=" * 70)
    for line in lines:
        print(line)
    print("=" * 70)


# ---------------------------------------------------------------------------
# exp_1: the MEAN under scaling and shifting.
#
# The MEAN (expectation) E[X] is the average / balance point of a distribution — where the samples
# center. Two linear rules, true for ANY distribution:
#   E[a·X]  = a·E[X]     scaling the variable scales the mean by the SAME factor a
#   E[X+d]  = E[X] + d   shifting the variable shifts the mean by d
# Combined: E[a·X + d] = a·E[X] + d. The mean moves LINEARLY. (The variance won't — it scales by a²,
# not a; that contrast is exp_2, and it's the whole reason diffusion uses √ coefficients.)
# We use a NON-Gaussian X (uniform) on purpose, to show this isn't a Gaussian-only fact.
# ---------------------------------------------------------------------------
def exp_1_mean(seed=0):
    """The MEAN E[X] and how it moves under scaling/shifting: E[aX]=a·E[X], E[X+d]=E[X]+d — both
    linear, for any distribution. Verified on a uniform (non-Gaussian) X; figure shows E[aX] is a
    straight line in a."""
    _banner("SECTION 1 · exp_1: the MEAN under scaling and shifting")

    torch.manual_seed(seed)
    n = 500_000
    X = torch.rand(n) * 2.0                       # X ~ Uniform[0,2]: any old distribution. true mean 1

    print("  the MEAN E[X] = the average / balance point of a distribution.")
    print("    X ~ Uniform[0,2]  (NOT Gaussian — everything here holds for any distribution)")
    print(f"    E[X]       true 1.0     measured {X.mean():.4f}\n")

    a, d = 3.0, 5.0
    print("  scale by a and shift by d — the mean moves LINEARLY:")
    print(f"    E[a·X]     a·E[X]     = {a}·1   = {a*1.0:>4.1f}     measured {(a*X).mean():.4f}")
    print(f"    E[X+d]     E[X]+d     = 1+{d:.0f}  = {1+d:>4.1f}     measured {(X+d).mean():.4f}")
    print(f"    E[a·X+d]   a·E[X]+d   = {a}·1+{d:.0f} = {a*1.0+d:>4.1f}     measured {(a*X+d).mean():.4f}\n")
    print("  scale by a → mean ×a ; shift by d → mean +d. Simple and linear. (Heads up: the VARIANCE")
    print("  will NOT scale by a — it scales by a². That asymmetry is exp_2, and it's exactly why")
    print("  diffusion's coefficients are square roots.)")

    # figure: E[aX] is a straight line through the origin (slope E[X]).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a_vals = torch.linspace(-2, 3, 21)
    measured = torch.tensor([(av * X).mean() for av in a_vals])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(a_vals.numpy(), (a_vals * X.mean()).numpy(), color="tab:red", lw=2, label="predicted  a·E[X]")
    ax.scatter(a_vals.numpy(), measured.numpy(), s=18, color="tab:blue", label="measured  E[a·X]", zorder=3)
    ax.axhline(0, color="gray", lw=0.7); ax.axvline(0, color="gray", lw=0.7)
    ax.set_xlabel("a (scale factor)"); ax.set_ylabel("E[a·X]")
    ax.set_title("the mean scales LINEARLY with a:  E[a·X] = a·E[X]")
    ax.legend(); fig.tight_layout()
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "mean_scales_linearly.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out} — a straight line through the origin, slope E[X].")
    print("  Next (exp_2): the VARIANCE, which scales by a² — a parabola, not a line.")


# ---------------------------------------------------------------------------
# exp_2: the VARIANCE under scaling and shifting.
#
# VARIANCE Var(X) = E[(X - E[X])²] = the average SQUARED distance from the mean = how spread out the
# samples are (std = √Var is the spread in the original units). Under linear ops it behaves DIFFERENTLY
# from the mean:
#   Var(a·X) = a²·Var(X)     scaling by a multiplies variance by a² (std by |a|) — because the
#                            distances from the mean get scaled by a, and variance squares them
#   Var(X+d) = Var(X)        shifting slides everything together, so the SPREAD is unchanged
# So scaling hits the mean by a but the variance by a². That asymmetry (line vs parabola) is the
# whole reason diffusion uses √ coefficients: only the SQUARES (a²) add up the way we want.
# ---------------------------------------------------------------------------
def exp_2_variance(seed=0):
    """The VARIANCE and how it moves: Var(aX)=a²·Var(X) (std ×|a|), Var(X+d)=Var(X). Shown on the
    same uniform X; figure = histograms visibly widening + the a² parabola vs the mean's a line."""
    _banner("SECTION 1 · exp_2: the VARIANCE under scaling and shifting")

    torch.manual_seed(seed)
    n = 500_000
    X = torch.rand(n) * 2.0   # Uniform[0,2]: true Var = (2-0)²/12 = 1/3 ≈ 0.3333
    varX = X.var()

    print("  the VARIANCE Var(X) = average SQUARED distance from the mean = the spread.")
    print("    (std = √Var is that spread in the original units.)")
    print(f"    X ~ Uniform[0,2]:  Var(X) true 0.3333  measured {varX:.4f}   std {varX.sqrt():.4f}\n")

    print("  scale by a → variance ×a²  (NOT ×a):")
    for a in [2.0, 3.0, -2.0]:
        print(f"    Var({a:>4}·X)   a²·Var(X) = {a**2:>2.0f}·0.3333 = {a**2*0.3333:>7.4f}   "
              f"measured {(a*X).var():>7.4f}")
    print("    note Var(-2·X) = Var(2·X): the sign is gone (a² is positive) — std scales by |a|.\n")

    d = 5.0
    print(f"  shift by d → variance UNCHANGED (spread doesn't move):")
    print(f"    Var(X+{d:.0f})    Var(X) = 0.3333            measured {(X+d).var():>7.4f}\n")
    print("  So the SAME operation hits the mean and the variance differently:")
    print("    scale by a:  mean ×a   (linear)   |   variance ×a²  (quadratic)")
    print("    shift by d:  mean +d              |   variance unchanged")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))

    # LEFT: SEE the spread widen as we scale. Wider box = bigger variance.
    for a, color in [(1, "tab:blue"), (2, "tab:orange"), (3, "tab:green")]:
        axL.hist((a * X).numpy(), bins=120, density=True, alpha=0.5, color=color,
                 label=f"{a}·X   (var {(a*X).var():.2f})")
    axL.set_title("scaling widens the spread — variance grows"); axL.set_xlabel("value")
    axL.set_ylabel("density"); axL.legend()

    # RIGHT: the a² law as a parabola, with the mean's line (from exp_1) for contrast.
    a_vals = torch.linspace(-3, 3, 25)
    var_meas = torch.tensor([(av * X).var() for av in a_vals])
    axR.plot(a_vals.numpy(), (a_vals ** 2 * varX).numpy(), color="tab:red", lw=2,
             label="variance:  a²·Var(X)  (parabola)")
    axR.scatter(a_vals.numpy(), var_meas.numpy(), s=14, color="tab:red", zorder=3,
                label="measured Var(a·X)")
    axR.plot(a_vals.numpy(), (a_vals * X.mean()).numpy(), color="tab:blue", ls="--", lw=1.5,
             label="mean:  a·E[X]  (line, exp_1)")
    axR.axhline(0, color="gray", lw=0.7); axR.axvline(0, color="gray", lw=0.7)
    axR.set_title("variance ∝ a² (parabola)  vs  mean ∝ a (line)")
    axR.set_xlabel("a (scale factor)"); axR.legend()

    fig.tight_layout()
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "variance_scales_quadratically.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out} — left: the spread visibly widening; right: variance is a PARABOLA in a")
    print("  while the mean is a LINE. Next (exp_3): adding two variables — linearity of expectation.")


# ---------------------------------------------------------------------------
# exp_3: SUMS — linearity of expectation.
#
# Add two variables (with weights) and the MEAN splits cleanly:
#   E[a·X + b·Y] = a·E[X] + b·E[Y]
# The average of a combination is the combination of the averages. The remarkable part: this holds
# ALWAYS — even when X and Y are strongly DEPENDENT. (Averaging is linear; it never "sees" how X and
# Y relate.) Keep this in mind: the VARIANCE of a sum will NOT be so forgiving — it needs a
# covariance term, and only drops it when X ⟂ Y (exp_4). So: means always add; variances only add
# when independent.
# ---------------------------------------------------------------------------
def exp_3_linearity(seed=0):
    """Linearity of expectation: E[aX+bY]=aE[X]+bE[Y], verified for an INDEPENDENT pair and a
    strongly DEPENDENT pair (Y=X²) — it holds either way. Figure: the dependence (scatter) + the
    identity matching in both cases."""
    _banner("SECTION 1 · exp_3: SUMS — linearity of expectation  E[aX+bY] = aE[X]+bE[Y]")

    torch.manual_seed(seed)
    n = 500_000
    X = torch.rand(n) * 2.0                       # E[X] = 1
    a, b = 2.0, -3.0

    Y_ind = torch.rand(n) * 4.0                   # independent of X, E[Y] = 2
    Y_dep = X ** 2                                 # FULLY determined by X (E[X²]=4/3≈1.333)

    print(f"  a={a}, b={b}. Claim: the mean of the combination = combination of the means,\n")
    print(f"  {'case':<26} | {'measured E[aX+bY]':>18} | {'aE[X]+bE[Y]':>12}")
    print(f"  {'-'*26}-+-{'-'*18}-+-{'-'*12}")
    results = {}
    for name, Y in [("independent  (Y~U[0,4])", Y_ind), ("DEPENDENT    (Y = X²)", Y_dep)]:
        lhs = (a * X + b * Y).mean().item()
        rhs = a * X.mean().item() + b * Y.mean().item()
        results[name] = (lhs, rhs)
        print(f"  {name:<26} | {lhs:>18.4f} | {rhs:>12.4f}")
    print("\n  BOTH match. The mean splits over the sum no matter how X and Y are related — averaging")
    print("  is linear and never looks at the dependence. (Contrast exp_4: the variance of a sum")
    print("  DOES depend on it.)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))

    # LEFT: make the dependence VISIBLE — Y=X² is a curve, X and Y are tightly linked.
    idx = torch.randperm(n)[:2000]
    axL.scatter(X[idx].numpy(), Y_dep[idx].numpy(), s=5, alpha=0.3, color="tab:purple")
    axL.set_title("Y = X²  is strongly DEPENDENT on X"); axL.set_xlabel("X"); axL.set_ylabel("Y")

    # RIGHT: measured vs predicted E[aX+bY] for both cases — equal in each pair.
    labels = list(results.keys())
    meas = [results[k][0] for k in labels]
    pred = [results[k][1] for k in labels]
    xpos = torch.arange(len(labels)).numpy()
    axR.bar(xpos - 0.2, meas, width=0.4, color="tab:blue", label="measured E[aX+bY]")
    axR.bar(xpos + 0.2, pred, width=0.4, color="tab:red", alpha=0.7, label="predicted aE[X]+bE[Y]")
    axR.axhline(0, color="gray", lw=0.7)
    axR.set_xticks(xpos); axR.set_xticklabels(["independent", "dependent"])
    axR.set_title("identity holds in BOTH cases"); axR.legend()

    fig.tight_layout()
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "linearity_of_expectation.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out} — left: X,Y clearly dependent; right: E[aX+bY] still splits, both cases.")
    print("  Next (exp_4): the VARIANCE of a sum — where dependence finally matters (covariance).")


def run_experiments():
    # exp_1_mean()
    # exp_2_variance()
    exp_3_linearity()


if __name__ == "__main__":
    run_experiments()
