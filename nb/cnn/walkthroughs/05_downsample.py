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
# CNN · 05 — why `stride=2` (H/2, W/2) and ×2 channels?

`04` stacked stride-1 convs: the map stays `28×28` and the receptive field creeps up by `k−1 = 2`
per layer — you'd need ~14 layers for one cell to see a whole digit. `01` didn't do that. Two of its
convs used `stride=2`, **halving** `H,W` (`28 → 14 → 7`) while **doubling** channels
(`16 → 32 → 64`). This notebook measures the three things that buys:

- **(A)** the resolution pyramid `28 → 14 → 7`,
- **(B)** the receptive field *explodes* — stride multiplies the reach of every later layer,
- **(C)** channels grow *for free* — quartering positions keeps the footprint bounded.
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from cnn.train import load_mnist, to_img


torch.manual_seed(0)

# %% [markdown]
"""
## (A) The resolution pyramid (`28 → 14 → 7`)

A `stride=2`, `k3`, `p1` conv outputs `ceil(in/2)` — the `02` size formula with `s=2`. `01`'s
`features` is a stride-1 stem followed by two stride-2 downsamples, doubling channels at each. Let's
push a tensor through `01`'s exact stack and read the shapes.
"""

# %%
x = torch.randn(1, 1, 28, 28)
stem = nn.Conv2d(1, 16, kernel_size=3, padding=1)               # 28 -> 28 (stride 1)
down1 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)   # 28 -> 14
down2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)   # 14 -> 7
h0 = stem(x); h1 = down1(F.relu(h0)); h2 = down2(F.relu(h1))

print("01's features: a stride-1 stem, then two STRIDE-2 convs, DOUBLING channels each:")
print(f"  {tuple(x.shape)} --stem(1->16,s1)--> {tuple(h0.shape)}"
      f" --down1(16->32,s2)--> {tuple(h1.shape)} --down2(32->64,s2)--> {tuple(h2.shape)}")
print("  spatial pyramid 28 -> 28 -> 14 -> 7  (k3,s2,p1 -> out = ceil(in/2), the 02 formula)")

# %% [markdown]
"""
Spatial `28 → 28 → 14 → 7`, channels `1 → 16 → 32 → 64`. This *is* the pyramid you watched in `01` —
now we know why each arrow is there.
"""

# %% [markdown]
r"""
## (B) Stride makes the receptive field *explode*

The receptive-field recurrence from `04`, with the stride term made explicit:

$$\text{RF}_L = \text{RF}_{L-1} + (k-1)\cdot \prod(\text{strides of earlier layers})$$

The key word is **product**: once you've downsampled by 2, every later `(k−1)` step is worth *2
input pixels*, and after two downsamples, *4*. So RF compounds instead of adding. We measure it
**exactly** with autograd — pick one central output cell, backprop to the input, and the input
pixels with nonzero gradient are precisely the ones it depends on.
"""

# %%
S = 31
ones = torch.ones(1, 1, 3, 3)


def rf(n_layers, stride):
    xin = torch.zeros(1, 1, S, S, requires_grad=True)
    y = xin
    for _ in range(n_layers):
        y = F.conv2d(y, ones, stride=stride, padding=1)
    c = y.shape[-1] // 2                              # a central output cell
    y[0, 0, c, c].backward()
    m = xin.grad[0, 0].abs() > 0                      # input pixels it depends on
    ys, _ = m.nonzero(as_tuple=True)
    return ys.max().item() - ys.min().item() + 1     # RF side length, in input pixels


print("receptive field of ONE output cell, in INPUT pixels (autograd: which input pixels the cell")
print("actually depends on), for stride-1 vs stride-2 stacks of k3 convs:")
print("  layers |  stride-1 RF  |  stride-2 RF")
for L in (1, 2, 3):
    print(f"     {L}    |      {rf(L, 1):>2}       |      {rf(L, 2):>2}")
print("  stride-1 crawls 3->5->7; stride-2 leaps 3->7->15 (and ->31 at layer 4).")
print("  -> ~14 stride-1 convs to reach RF 28, but only ~4 stride-2: stride is how a small-kernel")
print("     net comes to 'see' the whole digit.")

# %%
# draw the exact RF box (from rf() above) centered on a real '3' after 3 convs, both strides
xte, yte = load_mnist(train=False)
digit = to_img(xte[(yte == 3).nonzero()[0].item():][:1])     # a real '3', 28x28 in [0,1]

fig, axes = plt.subplots(1, 2, figsize=(2 * 2.6, 2.9))
for ax, stride in zip(axes, (1, 2)):
    r = rf(3, stride)                                         # RF side after 3 convs
    ax.imshow(digit, cmap="gray")
    lo = 14 - r / 2                                           # box centered on the 28x28 digit
    rect = plt.Rectangle((lo, lo), r, r, fill=False, edgecolor="red", lw=2)
    ax.add_patch(rect)
    ax.set_title(f"3 stride-{stride} convs\nRF {r}x{r} pixels", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("what ONE late output cell sees — same depth, stride-2 reaches far more of the digit",
             fontsize=9)
fig.tight_layout()
plt.show()

# %% [markdown]
"""
Same depth — **3 convs** — but the stride-2 cell's `15×15` window covers half the digit where the
stride-1 cell's `7×7` sees a sliver. *That's* what stride buys: reach, cheaply.
"""

# %% [markdown]
"""
## (C) Channels grow *for free* — the footprint stays bounded

If bigger reach were the whole story we'd just stack more stride-1 convs. The other half is
**compute**. Each stride-2 stage quarters the number of positions (`H·W → H/2·W/2`), so **doubling**
the channel count still shrinks the activation footprint `C·H·W`.
"""

# %%
print("footprint C·H·W per stage. once channels DOUBLE while area QUARTERS, each downsample")
print("multiplies the footprint by 2·(1/4) = 1/2 — resolution traded for depth cheaply:")
prev = None
for name, t in (("input", x), ("stem  1->16", h0), ("down1 16->32", h1), ("down2 32->64", h2)):
    c, hh, ww = t.shape[1], t.shape[2], t.shape[3]
    v = c * hh * ww
    ratio = "(input)" if prev is None else f"x{v / prev:.2f} vs prev"
    print(f"  {name:<12} {c:>2}·{hh}·{ww} = {v:>5} values   {ratio}")
    prev = v
print("  -> the stem is the one expensive, full-res layer; after that, area ÷4 pays for channels ×2.")
print("     stride-2 conv FLOPs drop ~4x per stage too, which makes deep, wide late layers affordable.")

# %% [markdown]
"""
Each downsample multiplies the footprint by `2 · (1/4) = 1/2`. So the network can afford to get
**richer per cell** (more channels = more distinct detectors) exactly where it got **coarser in
space**. That's the universal CNN shape:

> **resolution down, channels up.**
"""

# %% [markdown]
"""
## Recap

| part | claim | payoff |
|---|---|---|
| pyramid | stride-2 `k3` conv halves H,W (`ceil(in/2)`) | `28 → 14 → 7`, `01`'s exact stack |
| receptive field | stride *multiplies* every later step | `3/5/7` (s1) vs `3/7/15` (s2), by autograd |
| channels | area ÷4 lets channels ×2 for half the footprint | `C·H·W` ×1/2 per downsample |

**One-sentence compression:** a stride-2 conv halves the resolution, which makes the receptive field
compound (so a few layers see the whole digit instead of ~14) *and* quarters the positions (so
doubling channels still halves the footprint) — reach and richness, both cheap, which is why every
CNN funnels resolution down while fanning channels up.

Next: **`06` — the head + loss.** Turn the final `64×7×7` map into 10 class scores: `flatten →
Linear` vs global-average-pool, cross-entropy, and two wiring checks (untrained loss ≈ `ln 10`,
overfit one batch → 0).
"""
