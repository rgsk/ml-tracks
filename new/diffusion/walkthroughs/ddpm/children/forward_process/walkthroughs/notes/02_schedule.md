# Forward process · exp_2 — the schedule

**First box, opened after the whole game** ([01_dissolve.md](01_dissolve.md)). exp_1 showed a digit
melting into static and called `ᾱ_t` a "dial." This page is *what that dial actually is* — the
schedule behind the fade. Run it with `python forward_process.py` (`exp_2_schedule`).

---

## The one equation (recap)

The whole forward process is this single closed form — noise a digit `x0` to **any** level `t` in
one jump:

```
  x_t = √ᾱ_t · x0  +  √(1-ᾱ_t) · ε        ε ~ N(0, I)
```

exp_1 *ran* this to make the dissolve grid. Here we build the schedule that defines `ᾱ_t` and read
it apart.

---

## Three quantities, built from one

Pick a per-step noise level `β_t` (the *schedule*), and two friends fall out:

```
  β_t   how much fresh noise we mix in at step t        (grows over time)
  α_t   = 1 - β_t   fraction of signal surviving ONE step
  ᾱ_t   = α_1 · α_2 · … · α_t   fraction of the ORIGINAL signal surviving ALL t steps
```

`ᾱ_t` (`alpha_bar`, a **cumulative product** of the `α`s) is the star: it's the single dial that
sets the signal/noise mix in the equation above — `√ᾱ_t` scales the digit, `√(1-ᾱ_t)` scales the
noise. `t=0` → all signal; `t=T` → all noise.

The linear DDPM schedule just spaces `β_t` evenly from `1e-4` to `0.02`:

```python
betas      = torch.linspace(1e-4, 0.02, T)   # T = 1000
alphas     = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)    # the dial
```

This is the **same** schedule the parent `ddpm.py` trained on — we're just pulling it apart to read.

---

## Read the dial

```
   t |    β_t   |   α_t    |    ᾱ_t     | signal √ᾱ | noise √(1-ᾱ)
 -----+----------+----------+------------+-----------+-------------
    0 |  0.00010 |  0.99990 |   0.99990  |   0.9999  |    0.0100
  100 |  0.00209 |  0.99791 |   0.89514  |   0.9461  |    0.3238
  250 |  0.00508 |  0.99492 |   0.52142  |   0.7221  |    0.6918
  500 |  0.01006 |  0.98994 |   0.07780  |   0.2789  |    0.9603
  750 |  0.01504 |  0.98496 |   0.00330  |   0.0574  |    0.9983
  999 |  0.02000 |  0.98000 |   0.00004  |   0.0064  |    1.0000
```

**The one thing to take away: tiny per-step nibbles COMPOUND.** At `t=500`, a single step still
keeps `α_t ≈ 0.99` of the signal — one step barely touches the image. But `ᾱ_t` is the *product*
of 500 such factors, and `0.99^500 ≈ 0.08`: it has already collapsed. So the **signal** column
`√ᾱ` drains `1 → 0` and the **noise** column `√(1-ᾱ)` fills `0 → 1` as `t` climbs. By `t=999` the
digit contributes `√ᾱ ≈ 0.006` — essentially gone.

That single number, `ᾱ_t`, *is* the forward process: it decides how much digit vs. how much static
lives in `x_t` — the exact numbers behind the fade in [01_dissolve.md](01_dissolve.md).

---

## What's next

| next | opens | the question |
|---|---|---|
| **exp_3** | the closed-form **jump** / the `√` | does dialing `ᾱ` really equal adding noise step-by-step? why are the coefficients `√`? |

Next: **exp_3** — Monte-Carlo verify that the one-line jump matches the slow step-by-step process,
and see why the `√` makes the process *variance-preserving*.

---

*Numbers: `python forward_process.py` (`exp_2_schedule`).*
