# CNN · exp 3 — why conv at all, not the flatten+MLP we already had?

exp_1's `head` already starts with `nn.Flatten`. So we *have* a flatten+dense on hand — why not drop
`features` entirely and feed flattened pixels straight into a dense net? Because a flatten throws away
the two things that make an image an image, and a conv keeps both: **(A)** locality + translation
structure (shift a digit → a dense net sees a near-new input, a conv's output just shifts), and
**(B)** the parameter count. Run it with `python cnn.py` (`exp_3_why_conv`).

> Open this in the Markdown preview (`Cmd/Ctrl+Shift+V`) so the figure shows inline.

---

## 1. The thing we're arguing against: flatten + dense

`nn.Flatten` lays the `(1, 28, 28)` grid into a `784`-vector, row-major, so pixel `(r, c)` lands at
index:

```
i = r·28 + c
```

A dense layer then computes `y = W @ x_flat`, and `W[h, i]` is the weight from hidden unit `h` to
**pixel index `i`**. Two consequences, and they're the whole problem:

- **Index = absolute position.** `W[h, i]` is tied to one fixed pixel. The layer has no idea that
  index `i` (pixel `(r,c)`) and `i+1` (pixel `(r,c+1)`) are physical neighbors — you could permute
  all 784 indices, retrain, and get the identical model. Spatial adjacency isn't represented.
- **Every position gets its own weights.** Nothing learned at one location is shared with any other.

---

## 2. A shift wrecks the flattened vector

Take a clean `7` and shift it down-and-right by 4px. To your eye it's the *same digit*. But the ink
at index `i = r·28 + c` moves to:

```
(r+4)·28 + (c+4) = i + 4·28 + 4 = i + 116
```

Every ink pixel jumps ~116 slots. The original and shifted vectors have their ink in nearly
**disjoint** coordinate sets. Measure the overlap as a cosine — but first map to `[0,1]` (background
0, ink 1), so the dot product counts **only ink overlap**, not the hundreds of shared background
pixels (in the native `[-1,1]`, two background pixels contribute `(−1)·(−1) = +1` and dominate):

```python
a = x01.flatten();  b = x01_shift.flatten()
cos_pix = (a @ b) / (a.norm() * b.norm())     # ≈ 0.08
```

```
pixel-overlap cosine(orig, shifted) = 0.08     (far below 1)
```

**Takeaway:** to a dense layer that learned "ink at these indices ⇒ 7", the shifted 7 is almost a
new input. It gets **no free ride** — it must relearn the digit at every position. The cure is the
convolution.

---

## 3. A conv is translation-*equivariant* (the exact 0)

A conv slides **one** kernel over every position (exp_2). The claim the code verifies:

```
featmap( shift(x) )  ==  shift( featmap(x) )
```

Let `S` = shift, `C` = correlate-with-kernel. Equivariance is `C(S x) = S(C x)`. The **whole proof**
is three lines, and it's *why* convs are special:

```
Def:   (C x)[p] = Σⱼ k[j] · x[p + j]        # same kernel k at every p
       (S x)[q] = x[q − s]                   # shift by s

C(S x)[p] = Σⱼ k[j] · (S x)[p+j] = Σⱼ k[j] · x[p + j − s]
S(C x)[p] = (C x)[p − s]         = Σⱼ k[j] · x[p − s + j]

⇒  C(S x)[p] = S(C x)[p]      identical, term by term.
```

It works **only** because `C` uses the same `k` everywhere and depends purely on the *relative*
offset `j`. Slide the picture, slide the answer — exactly. In code:

```python
fmap          = F.conv2d(x, kernel, padding=1)                 # C(x)
fmap_of_shift = F.conv2d(x_shift, kernel, padding=1)           # C(S x)
shift_of_fmap = torch.roll(fmap, shifts=(shift, shift), ...)   # S(C x)
diff = (fmap_of_shift - shift_of_fmap)[interior].abs().max()   # = 0.00e+00
```

```
max | featmap(shift(x)) − shift(featmap(x)) |  =  0.0     (on the interior)
```

Two details that make it *exactly* 0: we shift with **`torch.roll`** (circular — matches the proof's
infinite/wrapping grid; a zero-pad shift would differ at the boundary), and we compare only the
**interior** (roll wraps a `shift+1` band around the border where the two wrappings don't correspond
— visible as the faint vertical lines in the figure).

![translation equivariance](../figures/experiments/03_equivariance.png)

**How to read it:** top row is the input `7` and its shift. **Bottom-middle** = shift *then* convolve
(`C(S x)`); **bottom-right** = convolve *then* shift (`S(C x)`). The digit region is pixel-identical —
that *is* the identity above. A conv gets the shifted digit's response **for free**; the same stroke
detector fires wherever the stroke goes.

> **The deep point.** A conv **is** a linear map — a dense matrix with special structure (one kernel
> tiled across a huge, mostly-zero matrix), precisely the **shift-equivariant** subclass. So
> `conv ⊂ dense`: a dense layer *could* represent this conv, but only by learning all 602k weights
> into that exact pattern, and only if training showed it every shift. The conv **hard-codes** the
> structure in ~160 weights — that's the *inductive bias*.

---

## 4. The parameter blow-up

Weight sharing isn't just elegant, it's **cheap**. Count the *first* layer:

| First layer | Formula | Params |
|---|---|---|
| **Dense** `784 → 768` | `784·768 + 768` | **602,880** |
| **Conv** `1→16`, 3×3 (exp_1's stem) | `16·1·3·3 + 16` | **160** |

```
dense 784 → 768   :  602,880   # one independent weight per (pixel, hidden) pair
conv  1 → 16, 3×3 :      160   # one 3×3 kernel per out channel, reused at all 784 spots
                               ~3,768× fewer parameters
```

And the dense layer *grows with image size*, while the conv's kernel count is size-independent. Fewer
params **and** the right bias baked in — which is why exp_1's model leads with a conv trunk.

---

## Recap

| part | claim | payoff |
|---|---|---|
| flatten | ties every weight to an absolute pixel index | a 4px shift → cosine 0.08 (near-new input) |
| equivariance | one shared kernel ⇒ `featmap(shift x) = shift(featmap x)` | verified to `0.0` on the interior (§3) |
| params | share one kernel over all positions | 602,880 → 160, **~3,768×** fewer (§4) |

**One-sentence compression:** a flatten+dense layer ties every weight to an *absolute* pixel, so a
4px shift is a brand-new input (cosine 0.08) it must relearn with 600k position-specific weights; a
conv slides one *shared* 160-weight kernel, so a shift just shifts the output (proven exact) — far
fewer weights and the right bias for free.

Next: **exp_4 — why `conv → relu → conv`?** Depth grows the receptive field, and *without* the ReLU
a stack of convs collapses back to a single conv (we'll measure the collapse, and that a ReLU breaks
it).

---

*Numbers + figure: `python cnn.py` (`exp_3_why_conv`).*
