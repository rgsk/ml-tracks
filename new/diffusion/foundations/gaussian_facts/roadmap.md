# Gaussian facts — roadmap (foundations)

Reusable probability background behind a lot of diffusion (and ML) algebra: how **mean** and
**variance** behave under linear operations, what makes the **Gaussian** family special, and the
payoffs (the diffusion forward collapse, the reparameterization trick). Lives in `foundations/`
because it crosses subtopics — the forward process, the reverse posterior, DDIM, and VAEs all link
here instead of re-deriving it.

**This is bigger than one file, so it's CHUNKED into sections — each section = one walkthrough `.py`
+ one note `.md`.** And unlike the top-down tracks, this primer is **bottom-up**: we build up from
the mean, because the whole point is to not skip the background. Run each `exp_*`, read it, say
"next".

---

## Sections

### 1. Mean & variance under linear operations
`walkthroughs/linear_ops.py` · `notes/01_linear_ops.md` — general random-variable algebra, **no
Gaussian yet** (holds for *any* distribution):
- `exp_1` **MEAN** under scaling/shift: `E[aX] = a·E[X]`, `E[X+d] = E[X]+d` (both linear)
- `exp_2` **VARIANCE** under scaling/shift: `Var(aX) = a²·Var(X)` (std ×`|a|`), `Var(X+d) = Var(X)`
- `exp_3` **SUMS** — linearity of expectation: `E[aX+bY] = aE[X]+bE[Y]` (always, even dependent)
- `exp_4` **VARIANCE OF SUMS** + covariance: `Var(X+Y) = Var(X)+Var(Y)+2Cov(X,Y)`; independent ⇒
  variances **add** (and *dependent* breaks it — why "independent" is load-bearing)

### 2. The Gaussian family
`walkthroughs/gaussian_family.py` · `notes/02_gaussian_family.md` — now specialize to the normal:
- `exp_1` the normal distribution: `μ, σ²`, the bell curve, the standard normal `N(0,1)`, z-scores
- `exp_2` **histogram & density** built by hand: bucketing, counts→density (`÷ N·width`), area = 1 —
  why a measured histogram and an analytic PDF share an axis
- `exp_3` **reparameterization** `X = μ + σ·ε` (`ε~N(0,1)`) — the identity behind diffusion & VAEs
- `exp_4` affine of a Gaussian stays Gaussian: `aX+b ~ N(aμ+b, a²σ²)`
- `exp_5` **sum of independent Gaussians is Gaussian** (closure) — and why that's special

### 3. Vectors & payoffs
`walkthroughs/vectors_payoffs.py` · `notes/03_vectors_payoffs.md` — put it to work:
- multivariate `N(0, I)`: isotropic, per-component independent — why diffusion works elementwise
  over pixels
- checking normality: histogram / Q-Q / skew & kurtosis
- **capstone synthesis**: `a·X + b·Y` is a predictable Gaussian (the variance-preserving `a²+b²=1`
  mix) → straight into the diffusion forward collapse and the reparam trick
- *(optional aside)* the CLT — why Gaussians show up everywhere

---

The three facts the forward-process closed form leans on — *scaling→c²·variance*, *independent→
variances add*, *sum-is-Gaussian* — are `exp_2`/`exp_4` of Section 1 and the closure item of
Section 2. Everything that needs them links here.

*Run: `python walkthroughs/<section>.py`. Notes in `walkthroughs/notes/`, figures in
`walkthroughs/figures/experiments/`.*
