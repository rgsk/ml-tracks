# CNN — Layer 3: depth — channels, receptive field, and why the ReLU is load-bearing

Layers 1–2 lived with **one** kernel making **one** feature map. That's not a network — it's a single
detector. This layer adds the two things that turn a conv op into a *deep* net, and then isolates the
one ingredient without which depth is a lie:

1. **Channels** — a layer maps `Cin` maps → `Cout` maps, so it learns many detectors at once.
2. **Receptive field** — stacking small kernels lets a late cell "see" a large input region.
3. **The ReLU** — a stack of *linear* convs collapses back to a single conv; the nonlinearity is
   the only reason depth buys capacity.

Verify every number with `../cnn.py` (`exp_3_stack_and_relu`). This experiment is console-only — no
figure — so the payoffs below are measurements you run and read.

---

## 1. Channels: a layer maps `Cin` maps → `Cout` maps

Layer 1's conv was the toy case `Cin = Cout = 1`: one image in, one map out. A real conv **layer**
takes a stack of `Cin` input maps and produces a stack of `Cout` output maps. Its weight is a 4-D
tensor:

```
weight shape = (Cout, Cin, kH, kW)
```

Read it as: **output channel `o` owns its own kernel `k[o]`**, and that kernel is itself a stack of
`Cin` little `kH×kW` filters — one per input channel — whose responses are **summed**:

```
out[o]  =  bias[o]  +  Σ_c  corr( in[c], k[o, c] )       # sum over the Cin input channels
```

So each output channel is a *different* detector that looks at **every** input channel at once. A
layer with `Cout = 8` learns 8 such detectors in parallel. The typical first layer is `1 → C` (turn
one greyscale image into `C` edge/blob maps), then `C → C'`, and so on.

```python
conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
# weight shape (8, 3, 3, 3) = (Cout, Cin, kH, kW),  bias (8,)
```

**Param count, ground-up.** Each of the `Cout` output channels has a `Cin·kH·kW` kernel plus one
bias:

```
params = Cout·(Cin·kH·kW) + Cout = 8·(3·3·3) + 8 = 216 + 8 = 224
```

Notice what's *absent*: the image size. A conv layer's parameter count depends on channels and
kernel size only — never on `H×W`. That's the weight sharing from Layer 1, now with many channels.

---

## 2. Receptive field: depth widens the window for free

The **receptive field (RF)** of an output cell is the region of the *original input* that can affect
it. For one `k×k` conv it's obviously `k×k`. The interesting part is what stacking does.

**Why two `k3` convs give a `5×5` RF, from the ground up.** Take an output cell of layer 2. It reads
a `3×3` window of layer-1's map. But each of *those* 9 layer-1 cells was itself computed from a `3×3`
window of the input. Two horizontally-adjacent layer-1 cells look at input windows offset by 1 pixel,
so the union of a row of 3 of them spans:

```
3  +  (3 - 1)  =  5      # the kernel width, plus one (k-1) step for each extra layer-1 cell
```

Same in both axes → a `5×5` input region. Add a third `k3` layer and it grows by another `k−1 = 2`
to `7×7`. The general rule (stride-1 case):

```
RF_1 = k
RF_L = RF_{L-1} + (k - 1)          # each extra k3, stride-1 layer adds (k-1) = 2
     = 1 + L·(k - 1)               # for k=3:  RF_L = 1 + 2L
```

Small kernels stacked deep reach as far as one big kernel would — but with far fewer parameters and a
nonlinearity between each (see §3). Two `3×3`s (10 weights + 2 biases per channel-pair) cover the
same `5×5` a single `5×5` (25 weights) would, and see two rounds of ReLU on the way.

### Measure it: perturb one pixel, count what moves

You don't have to trust the formula — `exp_3` **measures** the RF directly. Light a *single* center
pixel, push it through `L` stacked `k3` convs (all-ones kernels, so contributions can't cancel and
you're counting *reach*, not values), and count how many output cells changed vs. the all-zero input:

```python
pert = base.clone(); pert[0, 0, S//2, S//2] = 1.0     # one lit pixel
moved = (n_layers(pert, L) - n_layers(base, L)).abs() > 0
# bounding box of `moved` = the receptive field
```

```
1 conv : moved block 3x3     (formula RF = 1 + 1·2 = 3)
2 convs: moved block 5x5     (formula RF = 1 + 2·2 = 5)
3 convs: moved block 7x7     (formula RF = 1 + 3·2 = 7)
```

Measurement matches the formula exactly. This is how a small-kernel net eventually "sees" a whole
`28×28` digit: ~3 `k3` convs plus a stride-2 downsample (Layer 4) already cover most of it.

---

## 3. Why the ReLU is load-bearing

Here's the trap depth walks into. **A convolution is a linear map.** Compose two linear maps and you
get… a linear map. And from Layer 1 we know a *shift-equivariant* linear map on an image **is** a
convolution. So:

```
conv(·, w2) ∘ conv(·, w1)   =   conv(·, keq)          for a single kernel keq
```

Two stacked `3×3` convs with **no activation** are *exactly equal* to one `5×5` conv. Depth bought
you nothing but a bigger kernel — no extra expressive power at all.

### Recover the equivalent kernel `keq` (the impulse-response trick)

Any linear shift-invariant system is fully described by its **impulse response**: feed in a delta
(a single `1` surrounded by zeros) and the output *is* the equivalent kernel. One subtlety —
`F.conv2d` is cross-*correlation*, not convolution, so its impulse response comes out **flipped**;
flip it back to get a kernel that reproduces the stack under `F.conv2d`:

```python
delta = torch.zeros(1,1,9,9); delta[0,0,4,4] = 1.0
imp = linear_stack(delta)[:, :, 2:7, 2:7]     # 5x5 impulse response around the center
keq = torch.flip(imp, dims=(2, 3))            # flip back -> the equivalent 5x5 kernel
```

Then check that `conv(x, keq)` reproduces the whole two-layer stack:

```
max | linear_stack(x) - conv(x, keq) |  =  ~1e-6      # zero up to float error
```

Zero. The stack really was one `5×5` conv all along.

### Drop in a ReLU → superposition breaks

The fix is a **nonlinearity** between the convs. The modern default is `ReLU(z) = max(0, z)` (over
the old `sigmoid`/`tanh`). Why does it rescue depth? Because a linear map obeys **superposition**:

```
f(x1 + x2)  =  f(x1) + f(x2)            # true for any linear f
```

If that identity holds, `f` is linear, and (shift-equivariant) linear ⇒ a single conv. So `exp_3`
tests it on both stacks:

```python
lin_resid  = | linear_stack(x1+x2) - linear_stack(x1) - linear_stack(x2) |.max()
relu_resid = |  relu_stack (x1+x2) -  relu_stack (x1) -  relu_stack (x2) |.max()
```

```
linear stack residual = ~3e-6      (superposition holds  -> IS a single conv)
ReLU   stack residual = 2.653      (superposition broken -> NOT any single conv)
```

The nonzero residual is the proof: with a ReLU between them, the two layers can no longer be written
as **any** single convolution. `max(0, ·)` clips different inputs by different amounts depending on
their sign, so `f(x1+x2) ≠ f(x1)+f(x2)` — and that broken linearity is precisely the extra capacity.
It's what lets stacked layers compose `edges → strokes → parts` instead of collapsing back to one
edge detector.

**The one-line version:** without a nonlinearity, depth is a no-op; the ReLU is the entire reason a
deep conv net is more than a single big-kernel conv.

---

## Summary

| piece | what it is | the payoff you run |
|---|---|---|
| channels | layer maps `Cin→Cout`, weight `(Cout,Cin,kH,kW)`; each out-channel = own kernel summed over in-channels | `Conv2d(3→8,k3)` → `(8,3,3,3)`, 224 params |
| receptive field | stacking `k3` grows the window: `RF_L = 1 + L·(k−1)` | perturb 1 pixel → moved block `3×3 → 5×5 → 7×7` |
| linear collapse | two convs, no activation = one bigger conv (`keq`) | `|stack − conv(x,keq)| ≈ 1e-6` |
| ReLU | nonlinearity breaks superposition, so depth = real capacity | ReLU residual `≈ 2.65` (≠ 0) |

**One-sentence compression:** channels let a layer learn many detectors at once, stacking small
kernels grows the receptive field cheaply (`RF = 1 + 2L` for `k3`), but *without* a ReLU the whole
stack collapses to a single conv (verified to ~0) — so the nonlinearity is the load-bearing piece
that makes depth compose features instead of buying a bigger kernel.

Next (Layer 4): **downsampling** — a stride-2 conv halves `H,W` while channels grow, the
`28 → 14 → 7` pyramid that finally gives a late cell a receptive field covering the whole digit →
`04_downsample.md`.

---

*Numbers: `python ../cnn.py` (`exp_3_stack_and_relu`).*
