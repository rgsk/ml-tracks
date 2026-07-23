# Forward process · exp_7 — signal-to-noise ratio

Last box. Everything so far tracked `ᾱ` (signal fraction). SNR just repackages it as a **ratio** —
the single number for _how hard denoising is_ at each `t` — and `log-SNR` is the coordinate the
modern schedules (EDM, flow matching) are actually written in. Run it with
`python forward_process.py` (`exp_7_snr`).

---

## SNR = ᾱ, repackaged as difficulty

From the closed form `x_t = √ᾱ·x0 + √(1-ᾱ)·ε`, SNR is the ratio of the two terms' _powers_ (variance
= amplitude²). The `√` amplitudes square to exactly `ᾱ` and `1-ᾱ` — and this is the **same
unit-variance normalization from exp_3** (`Var(x0) = Var(ε) = 1`) that makes those squares land clean:

```
  SNR(t) = signal variance / noise variance
         = (√ᾱ_t)²·Var(x0) / [ (√(1-ᾱ_t))²·Var(ε) ]
         = ᾱ_t·1 / [ (1-ᾱ_t)·1 ]
         = ᾱ_t / (1 - ᾱ_t)
```

SNR measures **how much of the clean image survives** — i.e. how _recoverable_ `x0` is:

- **high SNR** (small `t`) = barely corrupted = `x0` easy to **recover**
- **low SNR** (big `t`) = mostly noise = `x0` gone, maximally **ambiguous**

### Careful — "difficulty" depends on the target

The above is the difficulty of recovering the _signal_ `x0`. But the parent `ddpm.py` trains the net
to predict the **noise `ε`**, and that difficulty runs the **opposite** way. At large `t`,
`x_t = √ᾱ·x0 + √(1-ᾱ)·ε ≈ ε`, so the net can nearly echo its input — predicting `ε` is _easy_. At
small `t`, `ε` enters scaled by `√(1-ᾱ) ≈ 0`, an almost-invisible perturbation buried under the full
image — predicting _which_ `ε` was added is _hard_. The two targets are exact mirrors (Gaussian
`x0, ε ~ N(0,1)`):

```
  Var(x0 | x_t) = 1 - ᾱ     recover-x0 error:  ~0 at small t (easy) → ~1 at large t (hard)
  Var(ε  | x_t) =     ᾱ     predict-ε error:   ~1 at small t (hard) → ~0 at large t (easy)
```

**What `Var(x0 | x_t)` means:** the uncertainty _left_ about `x0` once you've seen `x_t` — the error
of the best-possible predictor, the floor no network beats. `≈0` ⇒ `x_t` pins `x0` down (easy);
`≈1` ⇒ `x_t` told you nothing new (hard, since `Var(x0)=1` to begin with).

**Where `1-ᾱ` and `ᾱ` come from** — one line of Gaussian algebra, `Var(A|B) = Var(A) - Cov(A,B)²/Var(B)`.
Everything has variance 1 (`Var(x_t)=ᾱ+(1-ᾱ)=1`), so only the covariances matter:

```
  Cov(x0, x_t) = √ᾱ·Var(x0)    = √ᾱ       →  Var(x0|x_t) = 1 - (√ᾱ)²     = 1 - ᾱ
  Cov(ε,  x_t) = √(1-ᾱ)·Var(ε) = √(1-ᾱ)   →  Var(ε |x_t) = 1 - (√(1-ᾱ))² =     ᾱ
```

Those covariances are just `x_t`'s definition fed through covariance's **bilinearity** (constants
pull out, it distributes over the sum), plus `Cov(A,A)=Var(A)` and `Cov(x0,ε)=0` (independent):

```
  Cov(x0, x_t) = Cov( x0 , √ᾱ·x0 + √(1-ᾱ)·ε )
               = √ᾱ·Cov(x0,x0) + √(1-ᾱ)·Cov(x0,ε)
               = √ᾱ·Var(x0)    + √(1-ᾱ)·0        = √ᾱ·Var(x0)
```

Intuition: `x0` can only co-vary with the _part of `x_t` that contains it_ (`√ᾱ·x0`); the fresh-noise
term `√(1-ᾱ)·ε` is unrelated to `x0` and drops out. The `ε` line is the mirror — now `√ᾱ·x0` is the
unrelated term that dies, leaving `Cov(ε, x_t) = √(1-ᾱ)·Var(ε)`.

They're mirrors because `x_t = √ᾱ·x0 + √(1-ᾱ)·ε` is _one_ equation in _two_ unknowns: the more of
`x0` you read off `x_t` (big `ᾱ`), the more the `ε` term is squeezed out of view. The leftover
uncertainties `1-ᾱ` and `ᾱ` always sum to 1.

**Why "each is the other scaled by `1/√SNR`":** given `x_t`, knowing `ε` _is_ knowing `x0` —
`x0 = (x_t - √(1-ᾱ)·ε)/√ᾱ`. Since `x_t` is fixed, an error in your `ε` guess propagates straight in:

```
  (x0 error) = -( √(1-ᾱ)/√ᾱ )·(ε error) = -(1/√SNR)·(ε error)       [ SNR = ᾱ/(1-ᾱ) ]
```

So the two problems are the _same_ problem up to the per-`t` factor `1/√SNR`. Squaring it recovers
the table: `Var(x0|x_t) = (1/SNR)·Var(ε|x_t) = ((1-ᾱ)/ᾱ)·ᾱ = 1-ᾱ`. _Example:_ at `ᾱ=0.25`,
`SNR=1/3`, so an `ε`-error of `0.10` becomes an `x0`-error of `0.10·√3 ≈ 0.17`.

That single scale factor `1/√SNR` **is** the ε-vs-`x0` loss-weighting knob. This mismatch — SNR-as-
difficulty vs. the ε-target the model actually uses — is exactly what the **training-target box**
(parent exp_3: ε-vs-`x0` parametrization and loss weighting) is about. Here we only build the SNR
axis; that box decides what to _do_ with it.

---

### Proof it's literally that ratio.

Split `x_t` into its two independent pieces — `√ᾱ·x0` (signal)
and `√(1-ᾱ)·ε` (noise) — draw 200k unit-variance `x0` and `ε`, and measure each piece's variance:

```
   t | Var(√ᾱ·x0) Var(√(1-ᾱ)·ε) | their ratio | ᾱ/(1-ᾱ)
 -----+-----------------------------+-------------+---------
  100 |   0.8955      0.1051        |   8.5194    |  8.5366
  250 |   0.5216      0.4797        |   1.0873    |  1.0895
  500 |   0.0778      0.9244        |   0.0842    |  0.0844
  750 |   0.0033      0.9991        |   0.0033    |  0.0033
```

`Var(√ᾱ·x0) ≈ ᾱ` and `Var(√(1-ᾱ)·ε) ≈ 1-ᾱ` (the `√` amplitudes squared, on unit-variance draws), so
their ratio matches the analytic `ᾱ/(1-ᾱ)` at every `t` — the derivation above, confirmed empirically.

Since each training step draws a random `t`, the schedule decides how the training budget spreads
across difficulties.

```
   t |   lin SNR  lin logSNR |   cos SNR  cos logSNR
 -----+----------------------+---------------------
    0 | 9997.341       9.21  | 24243.53      10.10
  100 |    8.537       2.14  |    34.181      3.53
  250 |    1.090       0.09  |     5.489      1.70
  500 |    0.084      -2.47  |     0.970     -0.03
  750 |    0.003      -5.71  |     0.167     -1.79
  900 |    0.000      -8.22  |     0.024     -3.72
  999 |    0.000     -10.12  |     0.000    -19.84
```

---

## Two reads

**(1) SNR is just `ᾱ` in disguise** — a monotonic repackaging — but it _names the difficulty_
directly: `t=0` is `SNR ~1e4` (almost clean, trivial), `t=T` is `SNR ~1e-5` (basically noise,
maximally ambiguous).

**(2) log-SNR is the honest axis.** SNR crosses ~9 orders of magnitude, so only in log does the
schedule look like a smooth ramp. The key landmark is `logSNR = 0` ⇔ `SNR = 1` ⇔ **signal power ==
noise power** = the "halfway hard" point.

`SNR = ᾱ/(1-ᾱ) = 1` means `ᾱ = 1-ᾱ`, i.e. `ᾱ = 0.5`, so
signal power is `0.5` and noise power is `1 - 0.5 = 0.5` — `SNR = 0.5 / (1 - 0.5) = 1`:

```
  linear crosses logSNR=0 at t ≈ 259   → front-loads the easy end, then crams the tail at extreme low SNR
  cosine crosses logSNR=0 at t ≈ 496   → a far more even ramp
```

---

## See it

![log-SNR vs t, linear vs cosine](../figures/experiments/07_snr.png)

Cosine's log-SNR is a near-straight, **even ramp** — training difficulty spreads evenly across `t`.
Linear drops fast through the easy (high-SNR) end, then **flatlines deep negative**: a long tail of
redundant, near-identical hard steps. It's the **same wasted-steps story from exp_6**, now in the
coordinate that matters: modern schedules skip choosing `β` at all and place their steps _directly
along log-SNR_.

---

## 🎉 forward_process — complete

That closes the **first box of the ddpm whole game**. The full arc:

| exp | box                   | takeaway                                                           |
| --- | --------------------- | ------------------------------------------------------------------ |
| 1   | whole game (dissolve) | a digit melts into static via `x_t = √ᾱ·x0 + √(1-ᾱ)·ε`             |
| 2   | the schedule          | `ᾱ = ∏α` is the dial; tiny per-step nibbles compound               |
| 3   | the √ / closed form   | one-jump == step-by-step; √ ⇒ variance-preserving                  |
| 4   | the endpoint          | `x_T ~ N(0,I)` regardless of `x0` → why sampling starts from noise |
| 5   | cosine schedule       | declare `ᾱ = cos²`, back-solve `β`                                 |
| 6   | linear vs cosine      | cosine wastes 6% vs 33% of steps; dissolve makes it visible        |
| 7   | SNR                   | `ᾱ/(1-ᾱ)` = per-`t` difficulty; log-SNR = the modern coordinate    |

**Next box up** (back in the parent `ddpm.py`): the **training target** (exp_3) — why the net
predicts `ε`, the `ε`-vs-`x0` weighting, and how a training batch is assembled. That's where the
forward process feeds into learning.

---

_Numbers + figure: `python forward_process.py` (`exp_7_snr`)._
