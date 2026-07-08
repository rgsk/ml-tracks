# Forward process · exp_5 — the cosine schedule

Fifth box, and a shift in topic: the core forward *mechanics* are done (exp_1–4). The last three
boxes are about the **shape** of the schedule — the `ᾱ` curve — not new math. This one builds a
*second* schedule, the **cosine** schedule, by flipping the construction around. Run it with
`python forward_process.py` (`exp_5_cosine_schedule`).

---

## Two ways to build a schedule

```
  linear (exp_2):  set β_t directly  ->  ᾱ_t = cumprod(1 - β)     ᾱ falls out
  cosine (here):   set the ᾱ curve   ->  β_t back-solved from ᾱ    β falls out
```

The **linear** schedule picks the per-step noise `β_t` and lets the signal curve `ᾱ_t` emerge. The
**cosine** schedule (Nichol & Dhariwal 2021, *Improved DDPM*) does the reverse: it *declares the
signal curve it wants* — a gentle `cos²` falloff from 1 to 0 —

```
  ᾱ(t) = cos²( ((t/T + s) / (1 + s)) · π/2 ) / (normalizer)      s = 0.008
```

then **recovers** the per-step betas from the drop between consecutive `ᾱ`:

```
  β_t = 1 - ᾱ_t / ᾱ_{t-1}          clamped < 0.999 at the tail
```

The clamp stops the final steps from demanding more than 100% noise. Same three arrays
(`betas, alphas, alpha_bars`), same closed-form jump as before — only the curve's *shape* changes.

---

## Read it

```
   t |    β_t    |    ᾱ_t    | signal √ᾱ | noise √(1-ᾱ)
 -----+-----------+-----------+-----------+-------------
    0 |  0.00004  |  0.99996  |   1.0000  |    0.0064
  100 |  0.00053  |  0.97158  |   0.9857  |    0.1686
  250 |  0.00133  |  0.84589  |   0.9197  |    0.3926
  500 |  0.00316  |  0.49229  |   0.7016  |    0.7125
  750 |  0.00758  |  0.14318  |   0.3784  |    0.9256
  999 |  0.99900  |  0.00000  |   0.0000  |    1.0000
```

Two things to notice:

- **`β_t` is not linear.** It stays *tiny* early (`0.00004` at t=0) so the image stays crisp longer,
  then **ramps up** toward the end (the `0.999` at t=999 is the clamp kicking in — the schedule
  wants the last sliver of signal gone fast).
- **`ᾱ_t` descends gently.** At `t=500` cosine still has `ᾱ ≈ 0.49` (`√ᾱ ≈ 0.70`) — the digit is
  clearly there — where the **linear** schedule was already at `ᾱ ≈ 0.078` (`√ᾱ ≈ 0.28`), basically
  mush. Cosine spends far more of its steps in the informative middle.

---

## What's next

We now have *two* schedules with the same mechanics but different curves. The next box makes the
difference concrete:

- **exp_6 — linear vs cosine**: the two `ᾱ` curves side by side, a count of "wasted" near-pure-noise
  steps, and a **second dissolve** (linear row vs cosine row) that makes "cosine keeps signal alive
  longer" literally visible.

Next: **exp_6 — linear vs cosine.**

---

*Numbers: `python forward_process.py` (`exp_5_cosine_schedule`).*
