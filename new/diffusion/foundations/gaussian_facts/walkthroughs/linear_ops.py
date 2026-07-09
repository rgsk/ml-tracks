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


# ---------------------------------------------------------------------------
# exp_4: VARIANCE OF SUMS — where dependence finally bites.
#
# The mean split cleanly over a sum no matter what (exp_3). The VARIANCE does not:
#   Var(X + Y) = Var(X) + Var(Y) + 2·Cov(X, Y)
# with the COVARIANCE  Cov(X,Y) = E[(X-EX)(Y-EY)] = E[XY] - E[X]E[Y]  measuring how X and Y move
# TOGETHER (+ when they rise together, − when one rises as the other falls, 0 when unrelated).
#   independent ⇒ Cov = 0  ⇒  variances simply ADD  (Var(X+Y) = Var(X)+Var(Y))
#   positively dependent ⇒ Cov > 0 ⇒ sum is MORE spread than the naive add
#   negatively dependent ⇒ Cov < 0 ⇒ sum is LESS spread (can even cancel to zero)
# We show all three with the SAME marginals so only the dependence differs:
#   Y_ind = U[0,2] independent  |  Y_pos = X (rises with X)  |  Y_neg = 2−X (falls as X rises)
# All three Y's have the same mean and variance as X; only Cov changes — so any difference in
# Var(X+Y) is PURELY the covariance term. This is "fact B" of the diffusion closed form: the noise z
# is drawn INDEPENDENTLY of x, so Cov=0 and the signal/noise variances add — no cross term to spoil
# the √α·x + √(1−α)·z bookkeeping.
# ---------------------------------------------------------------------------
def exp_4_variance_of_sums(seed=0):
    """Var(X+Y)=Var(X)+Var(Y)+2Cov(X,Y). Same marginals, three dependence structures (independent,
    Y=X, Y=2−X) so only Cov differs: variances ADD only when independent; positive dep inflates the
    sum's spread, negative dep cancels it. Figure: the three scatters + measured vs naive-add bars."""
    _banner("SECTION 1 · exp_4: VARIANCE OF SUMS  Var(X+Y) = Var(X)+Var(Y)+2·Cov(X,Y)")

    torch.manual_seed(seed)
    n = 500_000
    X = torch.rand(n) * 2.0                       # X ~ Uniform[0,2], Var = 1/3
    varX = X.var().item()

    Y_ind = torch.rand(n) * 2.0                   # independent, SAME marginal as X (U[0,2])
    Y_pos = X.clone()                             # Y = X:    perfectly, positively dependent
    Y_neg = 2.0 - X                               # Y = 2−X:  perfectly, negatively dependent
    # all three Y's are Uniform[0,2] → identical mean & variance; only the DEPENDENCE on X differs.

    def cov(A, B):                                # Cov(A,B) = E[AB] − E[A]E[B]
        return (A * B).mean().item() - A.mean().item() * B.mean().item()

    print("  the VARIANCE of a sum carries a CROSS term the mean never had:")
    print("    Var(X+Y) = Var(X) + Var(Y) + 2·Cov(X,Y)")
    print("  Cov(X,Y) = E[(X−EX)(Y−EY)]: + move together, − move oppositely, 0 unrelated.\n")
    print(f"  All three Y have the SAME marginal (U[0,2], Var≈{varX:.3f}) — only the dependence differs.")
    print(f"  So any gap between Var(X+Y) and the naive Var(X)+Var(Y) is PURELY 2·Cov.\n")

    print(f"  {'case':<24} | {'Cov(X,Y)':>8} | {'Var(X)+Var(Y)':>13} | {'measured Var(X+Y)':>17}")
    print(f"  {'-'*24}-+-{'-'*8}-+-{'-'*13}-+-{'-'*17}")
    cases = [
        ("independent   (Y⟂X)", Y_ind),
        ("positive dep  (Y=X)", Y_pos),
        ("negative dep  (Y=2−X)", Y_neg),
    ]
    results = {}
    for name, Y in cases:
        c = cov(X, Y)
        naive = varX + Y.var().item()             # what you'd get if you (wrongly) just added
        meas = (X + Y).var().item()               # the truth, cross term included
        results[name] = (c, naive, meas)
        print(f"  {name:<24} | {c:>8.4f} | {naive:>13.4f} | {meas:>17.4f}")

    print("\n  independent → Cov≈0 → variances ADD (naive = measured).")
    print("  Y=X → Cov=Var(X)>0 → Var(2X)=4·Var(X), DOUBLE the naive add (the +2Cov piece).")
    print("  Y=2−X → Cov=−Var(X)<0 → X+Y is the constant 2, Var=0: the spreads CANCEL.")
    print("  Same marginals throughout — the ONLY thing that moved was how X and Y are related.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.3))

    # LEFT: SEE the three dependence structures — blob vs up-line vs down-line.
    idx = torch.randperm(n)[:1500]
    for (name, Y), color in zip(cases, ["tab:blue", "tab:green", "tab:red"]):
        axL.scatter(X[idx].numpy(), Y[idx].numpy(), s=5, alpha=0.3, color=color,
                    label=name.split("(")[0].strip())
    axL.set_title("same marginals, different dependence"); axL.set_xlabel("X"); axL.set_ylabel("Y")
    axL.legend(loc="upper center", fontsize=8)

    # RIGHT: naive add (assumes Cov=0) vs the truth — they only agree when independent.
    labels = list(results.keys())
    naive = [results[k][1] for k in labels]
    meas = [results[k][2] for k in labels]
    xpos = torch.arange(len(labels)).numpy()
    axR.bar(xpos - 0.2, naive, width=0.4, color="tab:gray", label="naive Var(X)+Var(Y)")
    axR.bar(xpos + 0.2, meas, width=0.4, color="tab:purple", label="true Var(X+Y)")
    axR.axhline(0, color="gray", lw=0.7)
    axR.set_xticks(xpos); axR.set_xticklabels(["independent", "Y=X", "Y=2−X"])
    axR.set_title("variances ADD only when independent (else off by 2·Cov)")
    axR.set_ylabel("variance"); axR.legend()

    fig.tight_layout()
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "variance_of_sums.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out} — left: three dependence shapes; right: naive add matches the truth ONLY")
    print("  for the independent pair. That's fact B: diffusion's z⟂x is what makes the variances add.")


# ---------------------------------------------------------------------------
# exp_5: CORRELATION — covariance, normalized.
#
# Covariance (exp_4) carries the units and spread of X and Y, so its raw size means nothing on an
# absolute scale. CORRELATION strips that out by dividing by each variable's own std:
#   ρ(X,Y) = Cov(X,Y) / (σ_X · σ_Y)     (measure X and Y in units of their own std, then covary)
# The payoff: ρ is DIMENSIONLESS and always lives in [-1, +1]:
#   ρ = +1  perfect positive line (Y = aX+b, a>0)   |   ρ = 0  no LINEAR co-movement
#   ρ = -1  perfect negative line (a<0)              |   0<|ρ|<1  a loose tendency
# The bound is forced by "variance can't be negative": with z-scores U,V (var 1 each),
#   Var(U±V) = 2 ± 2ρ ≥ 0  ⇒  -1 ≤ ρ ≤ 1, and ρ=±1 exactly when Var(U∓V)=0, i.e. a perfect line.
# Two traps this experiment makes visible:
#   • ρ=0 does NOT mean independent — only no LINEAR structure. Y=X² is fully dependent yet ρ≈0.
#   • ρ=±1 means an exact straight line, not merely "strongly related".
# ---------------------------------------------------------------------------
def exp_5_correlation(seed=0):
    """Correlation ρ=Cov/(σ_X·σ_Y): dimensionless, always in [-1,1]. Dial a target ρ by mixing signal
    and independent noise (Y = ρ·X_z + √(1−ρ²)·Z) and confirm measured ρ tracks it; show the extremes
    ρ=±1 are perfect lines and the ρ≈0 trap (independent AND the dependent Y=X²). Figure: a grid of
    scatters, each titled with its measured ρ."""
    _banner("SECTION 1 · exp_5: CORRELATION  ρ(X,Y) = Cov(X,Y) / (σ_X·σ_Y)  ∈ [-1, 1]")

    torch.manual_seed(seed)
    n = 500_000
    X = torch.rand(n) * 2.0                        # Uniform[0,2]
    Xz = (X - X.mean()) / X.std()                  # standardize X: mean 0, std 1 (a "z-score")

    def corr(A, B):                                # ρ = Cov / (σ_A σ_B), all population (÷n)
        a, b = A - A.mean(), B - B.mean()
        cov = (a * b).mean()
        return (cov / (a.pow(2).mean().sqrt() * b.pow(2).mean().sqrt())).item()

    print("  covariance carries units & spread, so its raw size is unreadable. Normalize it:")
    print("    ρ(X,Y) = Cov(X,Y) / (σ_X · σ_Y)  — dimensionless, and ALWAYS in [-1, +1].\n")

    # DIAL a target ρ: Y = ρ·X_z + √(1-ρ²)·Z, Z⟂X standard normal. Then Cov(X_z,Y)=ρ, Var(Y)=1 ⇒ ρ.
    print("  dial a target ρ by mixing signal X with independent noise Z:  Y = ρ·X_z + √(1−ρ²)·Z")
    print(f"    {'target ρ':>9} | {'measured ρ':>10}")
    print(f"    {'-'*9}-+-{'-'*10}")
    targets = [-1.0, -0.5, 0.0, 0.5, 1.0]
    panels = []                                    # (title, X-for-plot, Y, measured ρ)
    for r in targets:
        Z = torch.randn(n)
        Y = r * Xz + (1 - r ** 2) ** 0.5 * Z
        rho = corr(X, Y)
        panels.append((f"target ρ={r:+.1f}", X, Y, rho))
        print(f"    {r:>+9.1f} | {rho:>+10.3f}")

    print("\n  measured ρ tracks the dial and never leaves [-1,1]. The extremes are special:")
    print("    ρ=+1 → the noise term vanishes → Y is an exact straight line in X (a>0).")
    print("    ρ=-1 → same, negative slope.  These are the ONLY ways to hit ±1.\n")

    # THE ρ≈0 TRAP: independent (ρ≈0, as expected) vs Y=X² (fully DEPENDENT, yet ρ≈0).
    # NOTE: X² only decorrelates when X is SYMMETRIC around its mean (up- & down-branches cancel).
    # Our U[0,2] X sits on one side, so there X² is nearly linear (ρ≈+0.97). Use a symmetric U[-1,1].
    Xs = torch.rand(n) * 2.0 - 1.0                  # Uniform[-1,1]: symmetric about 0
    Y_ind = torch.rand(n) * 2.0
    Y_sq = Xs ** 2
    rho_ind, rho_sq = corr(X, Y_ind), corr(Xs, Y_sq)
    panels.append(("Y=X²  (X sym, dependent!)", Xs, Y_sq, rho_sq))
    print("  ρ=0 does NOT mean independent — only NO LINEAR co-movement:")
    print(f"    independent Y            ρ = {rho_ind:+.3f}   (expected ≈0)")
    print(f"    Y = X², X~U[-1,1] (dep.)  ρ = {rho_sq:+.3f}   ← ρ≈0 yet Y is fully determined by X!")
    print("    (X² decorrelates only when X is SYMMETRIC — the parabola's two branches cancel.)")
    print("  correlation is BLIND to nonlinear structure — it only sees straight-line co-movement.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    idx = torch.randperm(n)[:1500]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (title, XX, YY, rho) in zip(axes.ravel(), panels):
        color = "tab:purple" if title.startswith("Y=X²") else "teal"
        ax.scatter(XX[idx].numpy(), YY[idx].numpy(), s=5, alpha=0.3, color=color)
        ax.set_title(f"{title}\nmeasured ρ = {rho:+.2f}", fontsize=10)
        ax.set_xlabel("X"); ax.set_ylabel("Y")
    fig.suptitle("correlation: +1 and −1 are perfect lines; ρ≈0 misses nonlinear dependence (Y=X²)",
                 fontsize=12)
    fig.tight_layout()
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "correlation.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out} — scatters tightening to a line as |ρ|→1, and the Y=X² panel: a clear")
    print("  curve with ρ≈0. That's the punchline: correlation measures LINEAR co-movement only.")

    # THE SYMMETRY CAVEAT, side by side: same rule Y=X², two supports for X.
    #   one-sided X~U[0,2]  → only the RIGHT branch → nearly monotonic → ρ≈+0.97 (looks linear!)
    #   symmetric X~U[-1,1] → BOTH branches → they cancel → ρ≈0
    rho_sq_1side = corr(X, X ** 2)
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    ax1.scatter(X[idx].numpy(), (X[idx] ** 2).numpy(), s=6, alpha=0.3, color="tab:orange")
    ax1.set_title(f"one-sided  X~U[0,2]:  ρ = {rho_sq_1side:+.2f}\nonly the right branch → nearly a line",
                  fontsize=10)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y = X²")
    ax2.scatter(Xs[idx].numpy(), Y_sq[idx].numpy(), s=6, alpha=0.3, color="tab:purple")
    ax2.set_title(f"symmetric  X~U[-1,1]:  ρ = {rho_sq:+.2f}\nboth branches cancel → decorrelated",
                  fontsize=10)
    ax2.set_xlabel("X"); ax2.set_ylabel("Y = X²")
    fig2.suptitle("SAME rule Y=X², opposite ρ — the ρ≈0 trap needs X symmetric about its mean",
                  fontsize=12)
    fig2.tight_layout()
    out2 = os.path.join(_FIGS, "correlation_x2_symmetry.png")
    fig2.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"  wrote {out2} — SAME Y=X², but one-sided X gives ρ≈+0.97 (looks linear) while symmetric")
    print("  X gives ρ≈0. Decorrelation is not a property of X² — it needs the SYMMETRY.")


def run_experiments():
    # exp_1_mean()
    # exp_2_variance()
    # exp_3_linearity()
    # exp_4_variance_of_sums()
    exp_5_correlation()


if __name__ == "__main__":
    run_experiments()
