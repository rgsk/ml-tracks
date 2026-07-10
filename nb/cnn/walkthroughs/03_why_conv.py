# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# CNN · 03 — why conv at all, not the flatten+MLP we already had?

`01`'s `head` already starts with `nn.Flatten`. So we *have* a flatten+dense on hand — why not drop
`features` entirely and feed flattened pixels straight into a dense net? Because a flatten throws
away the two things that make an image an image, and a conv keeps both:

- **(A)** locality + translation structure — shift a digit and a dense net sees a near-new input,
  while a conv's output just *shifts* (translation **equivariance**),
- **(B)** the parameter count — one shared kernel vs a weight per (pixel, hidden) pair.
"""

# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from cnn.train import load_mnist, to_img

# %% [markdown]
"""
## (A) The thing we're arguing against: flatten + dense

`nn.Flatten` lays the `(1, 28, 28)` grid into a `784`-vector, row-major, so pixel `(r, c)` lands at
index `i = r·28 + c`. A dense layer then computes `y = W @ x_flat`, and `W[h, i]` is the weight from
hidden unit `h` to **pixel index `i`**. Two consequences, and they're the whole problem:

- **Index = absolute position.** `W[h, i]` is tied to one fixed pixel. The layer has no idea that
  index `i` and `i+1` are physical neighbors — you could permute all 784 indices, retrain, and get
  the identical model. Spatial adjacency isn't represented.
- **Every position gets its own weights.** Nothing learned at one location is shared with any other.

### A shift wrecks the flattened vector

Take a clean `7` and shift it down-and-right by 4px. To your eye it's the *same digit*. But the ink
at index `i = r·28 + c` moves to `(r+4)·28 + (c+4) = i + 116` — every ink pixel jumps ~116 slots, so
the original and shifted vectors have their ink in nearly **disjoint** coordinate sets. Measure the
overlap as a cosine — but first map to `[0,1]` (background 0, ink 1), so the dot product counts
**only ink overlap**, not the hundreds of shared background pixels (in the native `[-1,1]`, two
background pixels contribute `(−1)·(−1) = +1` and dominate the sum).
"""

# %%
xte, yte = load_mnist(train=False)
x = xte[(yte == 7).nonzero()[0].item():][:1]        # a clean '7' (1,1,28,28) in [-1,1]
shift = 4                                           # pixels down and right

x01 = (x + 1) / 2                                    # [0,1]: background 0, ink 1
x01_shift = torch.roll(x01, shifts=(shift, shift), dims=(2, 3))
a, b = x01.flatten(), x01_shift.flatten()
cos_pix = (a @ b) / (a.norm() * b.norm() + 1e-12)

print("a flatten+dense ties every weight to an ABSOLUTE pixel index.")
print(f"  shift the SAME '7' by ({shift},{shift}) px and compare the flattened 784-vectors:")
print(f"    pixel-overlap cosine(orig, shifted) = {cos_pix.item():.2f}   (far below 1)")
print("  -> to a dense layer the shifted digit is almost a NEW input; it must relearn the digit")
print("     at every position. Nothing is shared across space.")

# %% [markdown]
r"""
## A conv is translation-*equivariant* (the exact 0)

A conv slides **one** kernel over every position (`02`). The claim: `featmap(shift(x)) =
shift(featmap(x))`. Let `S` = shift, `C` = correlate-with-kernel; equivariance is `C(Sx) = S(Cx)`.
The whole proof is three lines, and it's *why* convs are special:

$$(Cx)[p] = \sum_j k[j]\, x[p+j], \qquad (Sx)[q] = x[q-s]$$

$$C(Sx)[p] = \sum_j k[j]\,(Sx)[p+j] = \sum_j k[j]\, x[p+j-s]$$

$$S(Cx)[p] = (Cx)[p-s] = \sum_j k[j]\, x[p-s+j]$$

The last two are identical term by term, so `C(Sx)[p] = S(Cx)[p]`. It works **only** because `C`
uses the same `k` everywhere and depends purely on the *relative* offset `j`. Slide the picture,
slide the answer — exactly. Let's verify it to numerical noise.
"""

# %%
# one fixed 3x3 kernel as our "feature": a diagonal edge detector (sum-to-zero) traces the 7's outline
kernel = torch.tensor([[-1., -1., 0.],
                       [-1., 0., 1.],
                       [0., 1., 1.]]).view(1, 1, 3, 3)

fmap = F.conv2d(x, kernel, padding=1)                                    # C(x)
x_shift = torch.roll(x, shifts=(shift, shift), dims=(2, 3))              # S(x), in [-1,1]
fmap_of_shift = F.conv2d(x_shift, kernel, padding=1)                     # C(S x)
shift_of_fmap = torch.roll(fmap, shifts=(shift, shift), dims=(2, 3))     # S(C x)

# compare on the INTERIOR: torch.roll wraps at the border (matches the proof's infinite grid), so
# crop a margin = shift+1 where the two wrappings correspond.
m = shift + 1
interior = (slice(None), slice(None), slice(m, 28 - m), slice(m, 28 - m))
diff = (fmap_of_shift[interior] - shift_of_fmap[interior]).abs().max().item()
scale = fmap.abs().max().item()

print("a conv slides ONE kernel over every position, so shifting the input just shifts the map:")
print("  featmap(shift(x)) = shift(featmap(x)).  verify (interior):")
print(f"    max|featmap(shift(x)) - shift(featmap(x))| = {diff:.2e}   (feature scale ~{scale:.1f})")
print("  -> equal to numerical noise. The conv gets the shifted digit's response FOR FREE.")

# %%
def _fm(t):
    return t.squeeze().detach().cpu().numpy()


fig, ax = plt.subplots(2, 3, figsize=(9, 6))
panels = [
    (0, 0, to_img(x), "input  x", "gray"),
    (0, 1, to_img(x_shift), "shift(x)", "gray"),
    (0, 2, None, "", None),
    (1, 0, _fm(fmap), "featmap(x)", "coolwarm"),
    (1, 1, _fm(fmap_of_shift), "featmap(shift(x))\n[shift, then conv]", "coolwarm"),
    (1, 2, _fm(shift_of_fmap), "shift(featmap(x))\n[conv, then shift]", "coolwarm"),
]
for r, c, img, title, cmap in panels:
    a_ = ax[r][c]
    a_.set_xticks([]); a_.set_yticks([])
    if img is None:
        a_.axis("off"); continue
    kw = dict(cmap=cmap) if cmap == "gray" else dict(cmap=cmap, vmin=-scale, vmax=scale)
    a_.imshow(img, **kw)
    a_.set_title(title, fontsize=11)
fig.suptitle(f"a conv is translation-EQUIVARIANT:  bottom-middle == bottom-right  (max diff {diff:.1e})",
             fontsize=12)
fig.tight_layout()
plt.show()

# %% [markdown]
"""
**How to read it:** top row is the input `7` and its shift. **Bottom-middle** = shift *then*
convolve (`C(Sx)`); **bottom-right** = convolve *then* shift (`S(Cx)`). The digit region is
pixel-identical — that *is* the identity above. The same stroke detector fires wherever the stroke
goes. (The faint border band is where `torch.roll` wraps and the two wrappings don't correspond —
why we compared only the interior.)

> **The deep point.** A conv **is** a linear map — a dense matrix with special structure (one kernel
> tiled across a huge, mostly-zero matrix), precisely the **shift-equivariant** subclass. So
> `conv ⊂ dense`: a dense layer *could* represent this conv, but only by learning all ~600k weights
> into that exact pattern, and only if training showed it every shift. The conv **hard-codes** the
> structure in ~160 weights — that's the *inductive bias*.
"""

# %% [markdown]
"""
## (B) The parameter blow-up

Weight sharing isn't just elegant, it's **cheap**. Count the *first* layer — a dense layer ties one
independent weight to every (pixel, hidden) pair, while a conv learns one small kernel per output
channel and reuses it at all 784 positions.
"""

# %%
H = 768                                 # 01's MLP-first-layer aside
dense_params = 784 * H + H
conv_out = 16                           # 01's first conv: Conv2d(1, 16, 3)
conv_params = conv_out * 1 * 3 * 3 + conv_out

print("parameter count of the FIRST layer:")
print(f"  dense 784 -> {H}       : {dense_params:>8,} weights (every pixel<->hidden pair)")
print(f"  conv 1 -> {conv_out} ch, 3x3  : {conv_params:>8,} weights (one small kernel, reused at 784 spots)")
print(f"  -> ~{dense_params // conv_params:,}x fewer weights AND the right bias built in.")
print("     And the dense layer GROWS with image size; the conv's kernel count is size-independent.")

# %% [markdown]
"""
## Recap

| part | claim | payoff |
|---|---|---|
| flatten | ties every weight to an absolute pixel index | a 4px shift → cosine ≈ 0.08 (near-new input) |
| equivariance | one shared kernel ⇒ `featmap(shift x) = shift(featmap x)` | verified to ~0 on the interior |
| params | share one kernel over all positions | 602,880 → 160, **~3,768×** fewer |

**One-sentence compression:** a flatten+dense layer ties every weight to an *absolute* pixel, so a
4px shift is a brand-new input (cosine ≈ 0.08) it must relearn with ~600k position-specific weights;
a conv slides one *shared* 160-weight kernel, so a shift just shifts the output (proven exact) — far
fewer weights and the right bias for free.

Next: **`04` — why `conv → relu → conv`?** Depth grows the receptive field, and *without* the ReLU a
stack of convs collapses back to a single conv (we'll measure the collapse, and that a ReLU breaks
it).
"""
