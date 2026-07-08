# CNN — Layer 2: the convolution op itself

Layer 1 leaned on `featmap(x) = F.conv2d(x, kernel)` as a black box — enough to prove *why* convs
beat a flatten. This layer **opens the box**: what that sliding kernel actually computes, seen three
ways — built from scratch with plain loops, fired as hand-set kernels you can *see*, and pinned down
by the output-size formula. Verify with `../cnn.py` (`exp_2_conv_op`); regenerate the image with
`python figs.py`.

> Open this in the Markdown preview (`Cmd/Ctrl+Shift+V`) so the picture shows inline.

---

## 1. The op: line up a window, multiply, sum

A 2-D conv slides a small kernel `k` of shape `(out_ch, in_ch, kH, kW)` over the image. At each
output position `(y, x)` and output channel `o`:

```
out[o, y, x] = bias[o] + Σ_c Σ_i Σ_j  k[o, c, i, j] · in[c,  y·s + i − p,  x·s + j − p]
```

Read it as: **line the kernel up over a `kH×kW` window of the input, multiply element-wise, sum.**
Then slide by **stride** `s`, having first **padded** the input with `p` zeros on each side so windows
fit at the border. Three pieces of structure live in that one formula:

- **Share over space.** The *same* `k` is used at every `(y,x)` — the weight sharing from Layer 1.
- **Own kernel per output channel.** Output channel `o` has its own kernel `k[o]`.
- **Sum over input channels.** `k[o]` is summed over all `Cin` input channels `c`.

Layer 1's "1 kernel" was the `Cin = Cout = 1` special case. Layers 3–4 do nothing new to the op —
they just **stack** and **stride** it. (The multi-channel `Σ_c` is Layer 3's story; here we stay at
one channel so the picture is legible.)

### It really is just loops

To prove there's no magic, `exp_2` builds cross-correlation from scratch — three nested loops (out
channel, output row, output col), each cell a windowed multiply-and-sum — and matches `F.conv2d`:

```python
window = xp[0, :, y0:y0+kH, x0:x0+kW]     # (Cin, kH, kW)
out[0, o, y, xx] = (w[o] * window).sum()  # multiply element-wise, sum
```

```
ONE output cell (top-left) by hand:  window·kernel summed  =  F.conv2d[0,0,0,0]   (same number)
whole map: max|from-scratch − F.conv2d|  =  0.0            -> our loops ARE F.conv2d
```

---

## 2. Cross-correlation vs convolution (the flip)

Notice the formula uses `in[..., y+i, ...]` — the kernel index `i` and the input index move in the
**same** direction. Textbook *convolution* flips the kernel: `i → −i, j → −j`. Every DL framework
skips the flip and still calls it "conv" — so `F.conv2d` is really **cross-correlation**.

Does it matter? For a **learned** kernel, no — the net simply learns whatever orientation it needs;
the flip would just relabel which weight is which. It matters **only** when you hand-*set* a kernel
and want it to mean what you drew — which is exactly what we do next, and why the note calls it out.

---

## 3. Hand-set kernels → feature maps (see what a conv detects)

Fire four fixed 3×3 kernels at a real `3` (with `padding=1`, keeping 28×28):

```
identity          [[0,0,0],[0,1,0],[0,0,0]]          copies the input (sanity check)
vertical edge     [[-1,0,1],[-2,0,2],[-1,0,1]]       Sobel-x: left↔right brightness change
horizontal edge   [[-1,-2,-1],[0,0,0],[1,2,1]]       Sobel-y: up↕down brightness change
blur (box)        [[1/9]*3]*3                         local average: smooths
```

![feature maps from hand-set kernels](figs/02_feature_maps.png)

**How to read it.** `identity` returns the `3` untouched — the sanity check that the op does what we
think. The two **Sobel** kernels sum to zero, so they read ~0 on flat regions (background, stroke
interiors) and **spike on strokes**: the vertical-edge map traces the digit's left/right borders, the
horizontal-edge map its top/bottom borders. `blur` averages each 3×3 neighborhood, softening the
digit. The whole of a CNN is: **learn** kernels like these instead of hand-drawing them — that's the
*only* difference from here on.

Why the edge kernels fire only on edges is the sum-to-zero argument from Layer 1: `+` and `−` weights
cancel over a constant patch, so nonzero output requires a *change* in brightness under the window.

---

## 4. The output-size formula

How big is the output? A window of width `kW` starting at the left needs `kW` columns; padding adds
`2p` columns; stride `s` takes every `s`-th start. So along **each** axis:

```
out = floor( (in + 2p − k) / s ) + 1
```

The three cases we use downstream, checked against `F.conv2d`:

| in | k | s | p | out | role |
|---|---|---|---|---|---|
| 28 | 3 | 1 | 1 | **28** | **same size** — the workhorse conv (`padding = k//2`) |
| 28 | 3 | 2 | 1 | **14** | **half** — the Layer-4 downsample |
| 28 | 5 | 1 | 0 | 24 | shrink by `k−1`, no padding |
| 28 | 3 | 2 | 0 | 13 | strided, no pad |

Two facts worth memorizing: `k3, s1, p1` **preserves** size (so you can stack convs without the map
shrinking), and `k3, s2, p1` **halves** it (the resolution pyramid `28 → 14 → 7` of Layer 4). The
formula matches `F.conv2d` exactly.

### Practice: the two important configs on small inputs

Same two rows, applied to `in=6` (even) and `in=5` (odd) — small enough to hand-derive and check:

| in | k | s | p | out | role |
|---|---|---|---|---|---|
| 6 | 3 | 1 | 1 | **6** | same size — preserved |
| 5 | 3 | 1 | 1 | **5** | same size — preserved |
| 6 | 3 | 2 | 1 | **3** | half — even in halves exactly (`6/2`) |
| 5 | 3 | 2 | 1 | **3** | half — odd in rounds up (`ceil(5/2)`) |

`k3, s1, p1` always returns `in`; `k3, s2, p1` returns `ceil(in/2)`.

---

## Summary

| piece | what it is |
|---|---|
| the op | window × kernel, summed; share over space, own kernel per out-channel, sum over in-channels |
| from scratch | three loops of multiply-and-sum reproduce `F.conv2d` to 0 |
| cross-correlation | no kernel flip; irrelevant for learned kernels, matters only for hand-set ones |
| feature maps | edge kernels (sum-to-zero) trace strokes; blur smooths; a CNN *learns* these |
| output size | `floor((in + 2p − k)/s) + 1`; `k3s1p1`→same, `k3s2p1`→half |

Next: stack these + a **ReLU**. Depth grows the receptive field, and *without* a nonlinearity a stack
of convs collapses back to a single conv — we'll verify that → `03_stack_and_relu.md`.

---

*Numbers: `python ../cnn.py`. Figures: `python figs.py`.*
