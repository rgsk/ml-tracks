# ---
# jupyter:
#   jupytext:
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
# # CNN · 02 — open `features`: what a `Conv2d` really computes
#
# In `01` you ran a model whose first box was `features = [Conv2d → ReLU] × 3`. We treated `Conv2d` as
# magic. It isn't. This notebook opens *one* conv three ways:
#
# - **(a)** rebuild it from scratch and match PyTorch to floating-point noise,
# - **(b)** hand-set kernels so you can *see* what a conv detects,
# - **(c)** the output-size formula that produced `01`'s `28 → 14 → 7` pyramid.

# %%
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def _repo_root() -> Path:
    here = Path.cwd()
    for d in (here, *here.parents):
        if (d / "pyproject.toml").exists():
            return d
    return here


ROOT = _repo_root()
DATA = ROOT / "new" / "diffusion" / "data" / "mnist.npz"   # shared MNIST cache
sys.path.insert(0, str(ROOT / "nb" / "cnn"))               # so `from custom... import ...` works


def load_mnist(train=True):
    d = np.load(DATA)
    x = d["x_train"] if train else d["x_test"]
    y = d["y_train"] if train else d["y_test"]
    x = (torch.from_numpy(x).float() / 127.5 - 1.0).unsqueeze(1)   # (N,1,28,28) in [-1,1]
    return x, torch.from_numpy(y).long()


def to_img(x):
    return ((x.squeeze().clamp(-1, 1) + 1) / 2).cpu().numpy()

# %% [markdown]
# ## (a) A conv is a sliding-window multiply-and-sum
#
# A conv layer maps input `(N, Cin, H, W)` with weight `(Cout, Cin, kH, kW)` to `(N, Cout, Ho, Wo)`.
# Output channel `o` slides its own kernel over the (padded) input; **each output cell is one windowed
# multiply-and-sum**:
#
# $$\text{out}[o, i, j] = \sum_{\text{over the } C_{in} \times k_H \times k_W \text{ window}} \text{kernel}[o] \cdot \text{input\_window}$$
#
# That's the *entire* operation — three loops (out-channel, row, col), each cell a dot product. The
# from-scratch version lives in `custom/naive_conv2d.py`:
#
# ```python
# for o in range(Cout):            # each output channel = its own detector
#     for i in range(Hout):        # slide down
#         for j in range(Wout):    # slide across
#             window = x[:, :, i*s:i*s+kH, j*s:j*s+kW]     # (N, Cin, kH, kW)
#             out[:, o, i, j] = (window * weight[o]).sum(dim=(1,2,3))
# ```
#
# Let's confirm it *is* `F.conv2d` — one cell by hand, then the whole map.

# %%
from custom.naive_conv2d import naive_conv2d

torch.manual_seed(0)
x = torch.randn(1, 1, 6, 6)
w = torch.randn(1, 1, 3, 3)
mine = naive_conv2d(x, w, stride=1, padding=0)          # (1,1,4,4)
ref = F.conv2d(x, w, stride=1, padding=0)

window = x[0, 0, 0:3, 0:3]                              # the top-left 3x3 window
print("a conv cell = line the kernel over a window, multiply element-wise, sum.")
print(f"  ONE cell by hand (top-left):  (window * kernel).sum() = {(window * w[0, 0]).sum().item():+.4f}")
print(f"  F.conv2d[0,0,0,0]                                     = {ref[0, 0, 0, 0].item():+.4f}")
print(f"  whole map: max|naive_conv2d - F.conv2d| = {(mine - ref).abs().max().item():.2e}")
print("  -> our 3 loops ARE F.conv2d.")

# %% [markdown]
# Two structural facts fall out of that loop — and they're the whole reason convs beat dense layers
# (that's `03`):
#
# - **Weight sharing** — the *same* `weight[o]` is used at every `(i, j)`. One detector, applied
#   everywhere on the image.
# - **Locality** — `out[o,i,j]` depends only on a small `kH×kW` window, not the whole image.
#
# > The kernel index and the input index move in the *same* direction, so this is technically
# > cross-**correlation**, not textbook convolution (which flips the kernel). Every framework does
# > this and calls it "conv." It only matters when you *hand-set* a kernel (next); a *learned* kernel
# > just learns whichever orientation it needs.

# %% [markdown]
# ## (b) Hand-set kernels → see what a conv detects
#
# A trained conv *learns* its kernels. To build intuition we **set** four `3×3` kernels by hand and
# fire them at a real `3` (with `padding=1`, keeping 28×28):
#
# ```-
# identity          vertical edge     horizontal edge    blur (box)
# copies input      Sobel-x: L↔R      Sobel-y: U↕D       local average
#
#  0  0  0          -1  0  1          -1 -2 -1          1/9 1/9 1/9
#  0  1  0          -2  0  2           0  0  0          1/9 1/9 1/9
#  0  0  0          -1  0  1           1  2  1          1/9 1/9 1/9
# ```

# %%
xte, yte = load_mnist(train=False)
img = xte[(yte == 3).nonzero()[0].item():][:1]          # a real '3' (1,1,28,28)

kernels = {
    # copies the input (sanity)
    "identity": [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
    ],  
    # Sobel-x: left<->right change
    "vertical edge": [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ], 
    # Sobel-y: up<->down change
    "horizontal edge": [ 
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ],  
    # local average: smooths
    "blur (box)": [[1 / 9] * 3] * 3,  
}

print("fire four hand-SET 3x3 kernels at a real '3' (padding=1 keeps 28x28) -> feature maps:")
for name, k in kernels.items():
    fm = F.conv2d(img, torch.tensor(k).float().view(1, 1, 3, 3), padding=1)
    print(f"  {name:<16} range [{fm.min():+.2f}, {fm.max():+.2f}]")

fig, axes = plt.subplots(1, 5, figsize=(5 * 2.0, 2.3))
axes[0].imshow(to_img(img), cmap="gray"); axes[0].set_title("input '3'", fontsize=9)
for ax, (name, k) in zip(axes[1:], kernels.items()):
    fm = F.conv2d(img, torch.tensor(k).float().view(1, 1, 3, 3), padding=1).squeeze()
    if name in ("identity", "blur (box)"):
        ax.imshow(fm.numpy(), cmap="gray")                # near-nonnegative: just a filtered image
    else:
        m = fm.abs().max().item()                         # signed edge map: 0 = neutral center
        ax.imshow(fm.numpy(), cmap="coolwarm", vmin=-m, vmax=m)
    ax.set_title(name, fontsize=9)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("one input, four hand-set kernels -> four feature maps (edges, blur)", fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
# - **identity** returns the digit untouched — proof the op does what we think.
# - The two **Sobel** kernels sum to zero, so they read ~0 on flat regions (background, stroke
#   interiors) and **spike on edges**: vertical-edge traces the left/right borders of strokes,
#   horizontal-edge the top/bottom ones. Red/blue = the two edge directions (signed output, centered
#   at 0).
# - **blur** averages each `3×3` neighborhood, softening the digit.
#
# Why edge kernels fire only on edges: a sum-to-zero kernel cancels over any constant patch, so nonzero
# output requires a *change* in brightness under the window. **A CNN's whole job is to learn kernels
# like these — thousands of them — instead of us hand-drawing four.** That's the only difference
# between this and `features`.

# %% [markdown]
# ## (c) The output-size formula (where `28 → 14 → 7` came from)
#
# A window of width `k` starting at the left needs `k` columns; padding adds `2p`; stride `s` takes
# every `s`-th start. So along **each** axis:
#
# $$\text{out} = \left\lfloor \frac{\text{in} + 2p - k}{s} \right\rfloor + 1$$
#
# The three convs from `01`'s `features`, checked against `F.conv2d` — this *is* the pyramid you
# already watched.

# %%
def out_size(inp, k, s, p):
    return (inp + 2 * p - k) // s + 1


print("each Conv2d's output size: out = floor((in + 2p - k)/s) + 1 — this IS 01's pyramid:")
for inp, k, s, p, role in [(28, 3, 1, 1, "stem  k3,s1,p1"),
                           (28, 3, 2, 1, "down1 k3,s2,p1"),
                           (14, 3, 2, 1, "down2 k3,s2,p1")]:
    real = F.conv2d(torch.zeros(1, 1, inp, inp), torch.zeros(1, 1, k, k), stride=s, padding=p).shape[-1]
    print(f"  {role}:  {inp:>2} -> {out_size(inp, k, s, p):>2}   (F.conv2d gives {real})")
print("  -> 28 -> 14 -> 7, exactly the shrink you saw.")

# %% [markdown]
# **Two configs are worth memorizing**, because every CNN is built from them:
#
# - `k3, s1, p1` → **preserves** size (`out = in`) — stack convs without the map shrinking.
# - `k3, s2, p1` → **halves** it, and exactly `out = ceil(in/2)` — the rule the whole resolution
#   pyramid rides on (`05`). Worth pinning down on both even and odd inputs:

# %%
print("k3,s1,p1 preserves size; k3,s2,p1 halves it (= ceil(in/2)) — even and odd:")
for inp, k, s, p, why in [(6, 3, 1, 1, "same size — preserved"),
                          (5, 3, 1, 1, "same size — preserved"),
                          (6, 3, 2, 1, "half — even halves exactly (6/2)"),
                          (5, 3, 2, 1, "half — odd rounds up (ceil(5/2)=3)")]:
    print(f"  in={inp} k{k} s{s} p{p} -> {out_size(inp, k, s, p)}   {why}")

# %% [markdown]
# ### What is the padding actually doing?
#
# Both clean sizes came from `p = k//2`. Turn padding **off** (`p = 0`) and the same convs stop
# behaving — stride-1 shrinks by `k−1` every layer, and strided lands on 13 instead of the clean 14.

# %%
print("turn padding off (p=0) and watch the sizes break:")
for inp, k, s, p, role in [(28, 3, 1, 0, "k3,s1,p0 (stem, unpadded)"),
                           (28, 5, 1, 0, "k5,s1,p0 (bigger kernel)"),
                           (28, 3, 2, 0, "k3,s2,p0 (down, unpadded)")]:
    real = F.conv2d(torch.zeros(1, 1, inp, inp), torch.zeros(1, 1, k, k), stride=s, padding=p).shape[-1]
    print(f"  {role:<26}:  {inp:>2} -> {out_size(inp, k, s, p):>2}   (F.conv2d gives {real})")
print("  -> s1 unpadded SHRINKS by k-1 each layer (28->26): stack 3 and you've lost 6 pixels.")
print("     p = k//2 (k3 -> p1) cancels that exactly -> same size. s2 unpadded gives 13, not 14.")

# %% [markdown]
# So padding isn't cosmetic — it counteracts the `k−1` a window loses at the border:
#
# - **stride 1:** `p = k//2` cancels the shrink *exactly*, giving `out = in`. Without it, every layer
#   peels off `k−1` pixels — stack three unpadded `k3` convs and you've silently lost 6 (`28 → 22`).
# - **stride 2:** `p = 1` is what lands you on the clean `ceil(in/2)` (14) instead of 13, so the
#   `28 → 14 → 7` pyramid stays exact.
#
# The same-size padding `p = k//2` only comes out whole for **odd** `k` (`k3→1, k5→2, k7→3`) — an even
# kernel can't be symmetrically padded to preserve size, which is why conv kernels are almost always
# odd (3, 5, 7).

# %% [markdown]
# ## Recap
#
# | part | claim | payoff |
# |---|---|---|
# | the op | a conv = sliding-window multiply-and-sum (weight sharing + locality) | `naive_conv2d` matches `F.conv2d` to ~5e-7 |
# | feature maps | hand-set edge kernels trace strokes; a CNN *learns* such kernels | the 4-kernel figure |
# | output size | `floor((in+2p−k)/s)+1` | reproduces `01`'s `28 → 14 → 7` |
#
# Next: **`03` — why conv at all?** We had a flatten+MLP already; the next notebook shows what the
# conv's weight sharing and locality buy that a dense layer throws away (translation equivariance +
# a huge parameter cut).
