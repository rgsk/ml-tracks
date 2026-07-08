# Forward process · exp_4 — the endpoint

Fourth box. exp_1's dissolve ended in pure static — and the **last column looked the same for every
digit**. This page is why: as `t → T` the digit is erased and `x_T` becomes a universal `N(0,I)`,
*regardless of `x0`*. That's the fact the whole generator hangs on. Run it with
`python forward_process.py` (`exp_4_endpoint`).

---

## The claim

From exp_3, the noised image has

```
  mean(x_t) = √ᾱ_t · x0        std(x_t) = √(1-ᾱ_t)
```

As `t → T`, `ᾱ_t → 0`, so the **signal term `√ᾱ·x0` vanishes** and the **std → 1**. Whatever `x0`
was, `x_T ≈ N(0, I)`. The digit-specific part dies; only universal static is left.

---

## Watch two very different digits converge

Noise `x0 = +2.0` and `x0 = -5.0` (deliberately far apart) to several `t`, 40k times each, and read
their mean/std:

```
   t |    ᾱ     | +2.0 mean   std  | -5.0 mean   std
 -----+----------+------------------+-----------------
  250 |  0.52142 |  +1.4458  0.6884 |  -3.6127  0.6935
  500 |  0.07780 |  +0.5519  0.9616 |  -1.3923  0.9608
  750 |  0.00330 |  +0.1132  1.0006 |  -0.2822  0.9965
  999 |  0.00004 |  +0.0213  0.9965 |  -0.0362  1.0028
```

- **Mid-way (`t=250`)** the two are still far apart — means `≈ √ᾱ·x0` = `+1.45` vs `-3.61`. The
  original still shows through (those are the *middle* columns of exp_1's dissolve, where the digit
  is grainy but legible).
- **By `t=999`** both collapse to `mean ≈ 0, std ≈ 1` — statistically **indistinguishable**. `x0` is
  gone. That's the identical-looking last column, for *any* digit.

---

## Why this is the linchpin of generation

The endpoint is the **same universal `N(0, I)` for every image**. That's what makes generation
possible at all:

```
  to GENERATE a new digit:
    1. grab a fresh scoop of N(0,1) static      ← free to sample; a valid x_T for SOME digit
    2. run the forward process BACKWARDS         ← denoise step by step
    3. land on a brand-new digit
```

If the endpoint depended on `x0`, you couldn't start from noise — you'd need to know the answer
first. Because it *doesn't*, pure static is a legitimate starting point, and the only missing piece
is the **reverse** process. That reverse is exactly the sampler the parent `ddpm.py` ran in exp_1.

That closes the core forward process: **a digit → a universal noise endpoint.**

---

## What's next

The core forward *mechanics* are done. The remaining boxes are about the *shape* of the schedule:

- **exp_5** — the **cosine schedule**: declare the `ᾱ` curve directly and back-solve `β` (the
  reverse of how we built the linear one).

Next: **exp_5 — the cosine schedule.**

---

*Numbers: `python forward_process.py` (`exp_4_endpoint`).*
