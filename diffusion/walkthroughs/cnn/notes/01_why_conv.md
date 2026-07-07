# CNN — Layer 1: why a convolution beats a flatten + MLP

The first idea in the CNN track. Four short sections, top to bottom, each one concept.
Verify the numbers yourself with `../cnn.py` (`exp_1_why_conv`); regenerate the images with
`python figs.py`.

> Open this in the Markdown preview (`Cmd/Ctrl+Shift+V`) so the picture and the GIF show inline.

---

## 0. The setup

Our denoiser so far is an **MLP**: it does `x = x.flatten(1)`, turning the `(1, 28, 28)`
image into a **784-vector**, then hits it with dense layers. That throws away the one thing an
image has going for it — its **2-D spatial structure**. A pixel only means something together
with its neighbors (an edge, a stroke, a loop), and the *same* stroke means the same thing
whether it sits top-left or center.

A flattened dense layer knows none of that. Every one of the 784 inputs is an independent knob,
and a digit nudged a few pixels over is, to it, an almost entirely new input.

The rest of this page shows **why**, with a picture and some numbers.

---

## 1. Flatten hard-codes meaning to absolute position

Take a clean `7`. Shift it 4 pixels down-and-right. To your eye it's the *same digit*.
But compare the two **flattened 784-vectors** by cosine similarity (mapped to `[0,1]`, so the
score = "how much ink lands on the same coordinates"):

```
cos( flatten(7), flatten(shifted 7) )  ≈  0.08     # tiny!
```

Why so small? Ink at pixel `(r, c)` sits at flat index `i = r·28 + c`. Shift by `(4,4)` and it
moves to `i + 4·28 + 4 = i + 116` — every ink pixel jumps ~116 slots, so the two vectors have
their ink in almost **disjoint coordinates**. The dot product (in `[0,1]`) literally counts
pixels where *both* have ink; after the shift, almost none line up.

So a flatten + MLP gets **no free ride** from translation: it must relearn the digit at every
position. That's the disease. The cure is the next section.

---

## 2. A convolution is translation-EQUIVARIANT

A conv slides **one** small kernel over *every* position. So shifting the input doesn't scramble
the output — it just **shifts the output too**:

```
featmap( shift(x) )  ==  shift( featmap(x) )
```

We verify it by computing both sides and subtracting. On the interior (away from the border,
where `torch.roll` wraps around) the two are **identical to numerical noise**:

```
max | featmap(shift(x)) − shift(featmap(x)) |  =  0.0
```

![translation equivariance](figs/01_equivariance.png)

**How to read it:**
- Top row: the input `7`, and the same `7` shifted.
- Bottom-left: the edge detector firing on the original (red / blue = the two edge polarities).
- **Bottom-middle** = shift *then* convolve. **Bottom-right** = convolve *then* shift.
- The **digit region is pixel-identical** between the two → that *is* the identity above.
- The faint border artifacts come from `torch.roll` wrapping at the edge — exactly why the
  code compares only the interior.

**Why it's true (one line of algebra).** With the same kernel `k` at every position,
`featmap(x)[p] = Σⱼ k[j]·x[p+j]`. Shift the input by `s` and
`featmap(shift x)[p] = Σⱼ k[j]·x[p+j−s] = featmap(x)[p−s] = shift(featmap x)[p]`.
Slide the whole image over → every neighborhood arrives at a new spot → every response arrives
at that same new spot. The feature map moves rigidly with the input.

---

## 3. One kernel, reused everywhere (weight sharing)

The reason equivariance holds is that the **same** 3×3 kernel is applied at every position.
Watch it walk across the digit — the feature map fills in pixel by pixel, all from *one* tiny
detector:

![kernel sliding](figs/01_slide.gif)

The same stroke detector fires wherever the stroke goes. Nothing is position-specific.

---

## 4. And it's far cheaper — the parameter blow-up

Weight sharing isn't just elegant, it's **cheap**. Count the parameters in the *first layer*:

| First layer | Formula | Params |
|---|---|---|
| **Dense** `784 → 768` | `784·768 + 768` | **602,880** |
| **Conv** `1→16`, 3×3 | `16·1·3·3 + 16` | **160** |

```
dense : 602,880   (one weight per pixel, per unit — grows with image size)
conv  :     160   (one tiny kernel, reused at every position — size-independent)
                  ~3,700× fewer parameters
```

The dense layer learns a separate weight for every (pixel, unit) pair. The conv learns **one**
9-number kernel per in/out channel pair and reuses it across all 784 positions. Fewer params
**and** the right inductive bias.

---

## Summary

| | flatten + MLP | convolution |
|---|---|---|
| locality (neighbors) | ✗ destroyed by flatten | ✓ 3×3 window |
| translation | ✗ relearn each position | ✓ equivariant (§2) |
| weight sharing | ✗ every pixel independent | ✓ one kernel everywhere (§3) |
| first-layer params | 602,880 | 160 (§4) |

That's why **every** image model — including the U-Net we'll swap in for the MLP denoiser — is
built from convs. Next: the convolution **op** itself (cross-correlation, edge kernels, the
output-size formula) → `02_conv_op.md`.

---

*Numbers: `python ../cnn.py`. Figures: `python figs.py`.*
