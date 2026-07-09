# Section 2 — The Gaussian family

A **foundations** primer (see [roadmap.md](../../roadmap.md)). Section 1 built the mean/variance
algebra that holds for **any** distribution. Now we specialize to the **normal** (Gaussian) and
collect the three payoffs the diffusion algebra runs on. The Section-1 laws still apply — but the
Gaussian adds a gift no other distribution has: **it stays Gaussian** under scaling, shifting, and
(independent) summing. That closure is what makes the forward diffusion step
`x_s = √α·x_{s-1} + √(1-α)·z` collapse to a clean closed form.

This note grows one `exp_*` at a time. Run with `python gaussian_family.py`.

---

## exp_1 — the normal: μ, σ², the bell, N(0,1), and z-scores

### A Gaussian is just two numbers

A **normal** (Gaussian) random variable is the familiar bell curve, and its _entire_ shape is pinned
down by exactly **two numbers** — the same mean and variance from Section 1:

```
  N(μ, σ²):   μ  = the mean     = where the bell is CENTERED
              σ² = the variance = how WIDE the bell is   (σ = std = spread in original units)
```

The **density** (the height of the bell at `x`) is

```
  N(x; μ, σ²) = exp( -½·((x-μ)/σ)² ) / (σ·√(2π))
```

a peak at `μ`, falling off symmetrically, and falling off _faster_ when `σ` is small. There are **no
other parameters**: fix `μ` and `σ` and you've fixed everything — the skew is 0, every higher-order
feature is determined. That two-number completeness is special (a uniform needs its two endpoints, a
general distribution needs infinitely many moments); it's the reason a Gaussian survives the algebra
so cleanly later.

Measuring the samples just recovers `μ` and `σ²` — they _are_ the mean and variance:

```
  distribution     |  μ (mean)          |  σ² (variance)
  -----------------+--------------------+-------------------
  N(+0.0, 1.00)    |  true 0.00 meas 0.00 | true 1.000 meas 1.002   ← N(0,1)
  N(+2.0, 0.25)    |  true 2.00 meas 2.00 | true 0.250 meas 0.249
  N(-1.0, 2.25)    |  true -1.0 meas -1.0 | true 2.250 meas 2.247
```

### The standard normal N(0,1), and the z-score that reaches it

The **standard normal** `N(0,1)` (`μ=0`, `σ=1`) is the _reference_ bell. Every other Gaussian is
just this one **shifted by μ and stretched by σ** — so a single operation, the **z-score**, undoes
the shift-and-stretch and maps _any_ normal back onto it:

```
  z = (X - μ) / σ     →     z ~ N(0,1)      (subtract the center, divide by the spread)
```

This is nothing but Section 1's affine laws run **backwards** — `z = (X-μ)/σ` is an affine
transform `a·X + d` with `a = 1/σ`, `d = -μ/σ`, so we just feed it through the mean and variance
laws.

**Mean → 0.** Pull the constant `1/σ` out first, then handle `X - μ` as one piece:

```
  E[ (X-μ)/σ ] = (1/σ)·E[ X - μ ]         pull the constant 1/σ out of E[·]
               = (1/σ)·( E[X] - μ )        E[X - μ] = E[X] - μ   (shift by -μ, exp_1 of Sec 1)
               = (1/σ)·( μ - μ )           E[X] = μ
               = 0
```

The middle step is worth naming on its own: **`E[X - μ] = 0`** — the expected _deviation_ from the
mean is always zero (the `+` and `−` deviations balance at the balance point). That's exactly why
variance can't just average `X - μ` (it'd be 0) and has to **square** it: `Var(X) = E[(X-μ)²]`.

**Variance → 1.** Same factor-out, but the constant comes out **squared** and the shift drops:

```
  Var[ (X-μ)/σ ] = (1/σ²)·Var[ X - μ ]     constant pulls out SQUARED  (Var(aZ)=a²Var(Z), exp_2)
                 = (1/σ²)·Var[ X ]          subtracting the constant μ doesn't change spread
                 = (1/σ²)·σ²                Var(X) = σ²
                 = 1
```

Same scale `1/σ` came out of the mean **linearly** but out of the variance **squared** (`1/σ²`), and
that `²` is what cancels the `σ²`. Same `a`, two different powers — the Section-1 asymmetry doing
real work. Standardizing three _different_ Gaussians all lands on mean 0, variance 1:

```
  from          |  mean(z) |  var(z)
  --------------+----------+--------
  N(+0.0,1.00)  |  -0.003  |  1.002
  N(+2.0,0.25)  |  -0.000  |  0.998
  N(-1.0,2.25)  |  +0.001  |  0.999
```

And because a standardized normal is _always_ the same bell, it has a fixed, memorizable mass
profile — the **68–95–99.7 empirical rule**:

```
  within 1σ:  measured 68.21%   (theory 68.27%)
  within 2σ:  measured 95.46%   (theory 95.45%)
  within 3σ:  measured 99.72%   (theory 99.73%)
```

### The picture: many bells, one standard normal

![three bells, and z-scoring collapses them all onto N(0,1)](../figures/experiments/normal_and_zscore.png)

**Left:** three Gaussians — `μ` slides the peak left/right, `σ` sets the width (small `σ` → tall and
narrow, large `σ` → short and wide). **Right:** z-score each one and the three histograms fall
_exactly_ on top of one another under the single `N(0,1)` curve; the shaded bands are the 68 / 95 /
99.7% within 1 / 2 / 3σ. **One reference bell, and everything else is an affine of it** — which is
precisely the door into the next three experiments.

### Reading the y-axis: "density" is a height, not a probability

The vertical axis is **probability density**, which is _not_ probability. For a continuous variable
the probability of landing at _exactly_ `x = 2.0` is zero (infinitely many reals to hit). What's
meaningful is the probability of landing in an **interval**, and that's an **area**:

```
  P(a ≤ X ≤ b) = area under the curve from a to b  ≈  density(x) · width
```

So density is the thing you **multiply by a width** to get a probability. The **total area under
any of these curves is exactly 1** (the variable lands _somewhere_). Two objects share the panel,
both in density units so they're comparable:

- **The smooth curves** are the analytic formula on a grid — `_normal_pdf(x, μ, σ)`, i.e.
  `exp(-½((x-μ)/σ)²)/(σ·√(2π))`. At the peak (`x=μ`) the `exp` is 1, so the height is just
  `1/(σ·√(2π))`. For `N(0,1)` that's `1/√(2π) ≈ 0.399` — the blue curve tops out at ~0.4.
- **The filled histogram** is _measured_ from the 500k samples with `density=True`, which normalizes
  each bar so the total area is 1:  `bar height = count_in_bin / (total · bin_width)`. Dividing by
  the bin width is what turns raw counts into density — and why the bars sit right under the curve.

**Why the narrow orange bell is _taller_ (~0.8) than the blue one (~0.4):** all three enclose area 1,
so a bell squeezed into a narrow base must go higher to compensate. `N(2, 0.25)` has `σ=0.5` →
`1/(0.5·√(2π)) ≈ 0.798`; the wide `N(-1, 2.25)` (`σ=1.5`) is the shortest (~0.266). **Narrow ⇒ tall,
wide ⇒ short.** A density value _can_ exceed 1 (a very narrow bell peaks well above 1) — that's fine,
because it's a height. Only the **area** is capped at 1.

> **Why this framing matters for diffusion.** "Every Gaussian is an affine image of `N(0,1)`" is the
> seed of the **reparameterization trick** (exp_2): if you want a sample from `N(μ,σ²)`, don't sample
> it directly — sample one `ε ~ N(0,1)` and compute `μ + σ·ε`. That single idea separates the
> randomness (a fixed `ε`) from the parameters (`μ, σ`), and it's what lets gradients flow through
> the noise in both VAEs and diffusion.

---

_Numbers + figure: `python gaussian_family.py` (`exp_1_normal`)._
