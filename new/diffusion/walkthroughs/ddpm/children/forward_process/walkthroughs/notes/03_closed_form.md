# Forward process · exp_3 — the √ / closed form

Third box. exp_1 dissolved a digit and exp_2 read the schedule — **both used the one-line jump**
`x_t = √ᾱ·x0 + √(1-ᾱ)·ε` without justifying it. This page pays that debt: (A) the one-jump really
*is* the slow step-by-step noising, and (B) that's *why* the coefficients are square roots. Run it
with `python forward_process.py` (`exp_3_closed_form`).

---

## The two forms

```
  step-by-step (the "real" process):  x_s = √α_s · x_{s-1} + √(1-α_s) · z_s     T times, fresh z each
  one jump      (what we actually use): x_t = √ᾱ_t · x0    + √(1-ᾱ_t) · ε        one ε, straight to t
```

If these agree we never have to run the T-step loop — we can make a training example at any noise
level in one line. They can't match *sample-by-sample* (each draws its own random noise), but they
match **in distribution**.

---

## (A) one-jump == step-by-step

Noise a fixed `x0 = 2.0` step-by-step 40k times to level `t`, and compare the empirical
mean/std of those draws to the closed form's predicted `√ᾱ·x0` and `√(1-ᾱ)`:

```
   t |  iter mean  iter std |   √ᾱ·x0   √(1-ᾱ)
 -----+----------------------+-----------------
  100 |   +1.8919   0.3240   |  +1.8922   0.3238
  500 |   +0.5553   0.9571   |  +0.5578   0.9603
  999 |   +0.0177   1.0060   |  +0.0127   1.0000
```

Mean and std match at every `t`. So the loop and the jump are the **same distribution** — we skip
the loop. That one-liner is exactly what exp_1's dissolve drew and what the parent's train loop uses.

**Why they collapse into one jump** — compose two steps, leaning on three facts:

```
DERIVATION — the forward closed form  x_t = √ᾱ·x0 + √(1−ᾱ)·ε

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
  and by C it is one fresh noise √(1−α₁α₂)·ε.  With ᾱ₂ = α₁α₂:
      x_2 = √ᾱ₂·x0 + √(1−ᾱ₂)·ε.
  By induction over t (each step folds one more α in):
      x_t = √ᾱ·x0 + √(1−ᾱ)·ε,   ᾱ = ∏_{s≤t} α_s.        [QED]
```

The T steps fold into one because each step just multiplies one more `α` into `ᾱ`.

---

## (B) why the coefficients are √ — variance preservation

Take unit-variance data `x0 ~ N(0,1)` and measure `Var(x_t)` across `t`:

```
   t |    ᾱ      1-ᾱ   | Var(x_t)
 -----+----------------+---------
    0 |  0.9999  0.0001 |  1.0122
  250 |  0.5214  0.4786 |  1.0152
  500 |  0.0778  0.9222 |  0.9958
  750 |  0.0033  0.9967 |  1.0024
  999 |  0.0000  1.0000 |  0.9994
```

`Var(x_t)` is pinned at **≈ 1 for every `t`**. The reason (`x0 ⊥ ε`, `Var(x0)=1`):

```
Why the coefficients are square roots — VARIANCE PRESERVATION (Var(x0)=1, x0 ⟂ ε):
      Var(x_t) = (√ᾱ)²·Var(x_0) + (√(1−ᾱ))²·Var(ε)
               =    ᾱ   ·   1      +    (1−ᾱ)   ·   1
               =    ᾱ + (1 − ᾱ)
               =    1                        ← for EVERY t
signal power ᾱ + noise power (1−ᾱ) always sum to 1, so x_t neither blows up nor
fades — it just trades signal for noise.
```

That's the whole point of the square roots: it's the **squares** that must sum to 1 (signal power
`ᾱ` + noise power `1-ᾱ`), so the **amplitudes** have to be `√ᾱ` and `√(1-ᾱ)`. Pick `ᾱ` and `1-ᾱ`
directly instead and the variance would blow up early and collapse late. With the √, `x_t` never
blows up or fades — it just **trades signal for noise** as `t` climbs. (This is called a
*variance-preserving* process; it's also why `x_T` sits at a clean unit-variance `N(0,I)` — the
next box.)

---

## What's next

Next: **exp_4 — the endpoint.** We've seen the *mean* is `√ᾱ·x0`; as `ᾱ → 0` that signal term
vanishes, so `x_T` becomes pure `N(0,I)` *regardless of `x0`* — the universal starting point that
lets generation begin from static.

---

*Numbers: `python forward_process.py` (`exp_3_closed_form`).*
