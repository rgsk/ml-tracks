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
# CNN · 04 — why `conv → relu → conv`?

`01`'s `features` didn't stack bare convs — it stacked `conv → relu → conv → relu → …`. Two
questions hide in that pattern, both with measured answers:

- **(A)** why stack convs at all — depth grows the *receptive field*,
- **(B)** why put a ReLU *between* them — without it, a stack of convs collapses back into a single
  conv, so depth buys nothing.
"""

# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0)

# %% [markdown]
r"""
## (A) Why stack — depth grows the receptive field

A single `3×3` conv output sees a `3×3` patch of its input. Put a second `3×3` conv on top: each of
*its* outputs sees a `3×3` patch of the first map, and each of those cells already spans `3×3` of
the original — so it sees `5×5` of the **original image**. Depth widens the window **without bigger
kernels**:

$$\text{RF}_L = 1 + L\,(k-1) \qquad (k=3,\ \text{stride }1:\ +2 \text{ per layer} \to 3, 5, 7, \dots)$$

We don't assert this — we **measure** it. Perturb *one* input pixel, run `L` stacked convs (all-ones
kernels so contributions can't cancel — we're counting reach, not values), and count how many output
pixels move.
"""

# %%
S = 15
base = torch.zeros(1, 1, S, S)
pert = base.clone(); pert[0, 0, S // 2, S // 2] = 1.0     # a single lit pixel at the center
ones = torch.ones(1, 1, 3, 3)


def n_layers(x, n):
    for _ in range(n):
        x = F.conv2d(x, ones, padding=1)
    return x


masks = {}
print("perturb ONE center input pixel; count OUTPUT pixels that move (the receptive field),")
print("after 1/2/3 stacked k3 convs:")
for L in (1, 2, 3):
    moved = (n_layers(pert, L) - n_layers(base, L)).abs() > 0
    masks[L] = moved[0, 0]
    ys, xs = moved[0, 0].nonzero(as_tuple=True)
    h = ys.max().item() - ys.min().item() + 1
    w = xs.max().item() - xs.min().item() + 1
    rf = 1 + L * (3 - 1)                                  # RF_L = 1 + L·(k-1) for k3, stride 1
    print(f"  {L} conv{'s' if L > 1 else ' '}: moved block {h}x{w}     (formula RF = 1 + {L}·2 = {rf})")
print("  -> depth widens the window with small kernels; ~3 k3 convs + a stride-2 (05) already let")
print("     an output cell 'see' most of a 28x28 digit.")

# %%
fig, axes = plt.subplots(1, 3, figsize=(3 * 2.4, 2.7))
for ax, L in zip(axes, (1, 2, 3)):
    rf = 1 + L * (3 - 1)
    ax.imshow(masks[L].cpu().numpy(), cmap="Reds", vmin=0, vmax=1)
    ax.plot(S // 2, S // 2, "b+", markersize=10, markeredgewidth=2)   # the one perturbed pixel
    ax.set_title(f"{L} conv{'s' if L > 1 else ''}  ->  RF {rf}x{rf}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("receptive field of ONE output cell grows with depth (blue + = perturbed input pixel)",
             fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
"""
The blue `+` is the single perturbed input pixel; the red square is every output cell it can reach.
It grows `3×3 → 5×5 → 7×7` — one cell more of context per layer, per side. (Stride-1 growth is slow;
`05`'s stride-2 downsampling is what makes it *explode* so late layers see the whole `28×28` digit
without dozens of layers.)
"""

# %% [markdown]
"""
## (B) Why the ReLU — without it the stack collapses to one conv

Here's the trap. A convolution is a **linear** map. Compose two linear maps and you get… a single
linear map. So two stacked convs *with no activation between them* are exactly **one** conv with a
bigger kernel — the depth bought nothing.

**Proof by impulse response.** Any linear shift-invariant map is fully described by its response to a
single spike (`delta`). Feed a delta through the 2-conv linear stack; the response is a `5×5` kernel
`keq`. Then `conv(x, keq)` — *one* conv — reproduces the whole stack.
"""

# %%
C = 8
w1 = torch.randn(C, 1, 3, 3) * 0.3       # 1 -> C
w2 = torch.randn(1, C, 3, 3) * 0.3       # C -> 1


def linear_stack(x):                     # two convs, NO activation
    return F.conv2d(F.conv2d(x, w1, padding=1), w2, padding=1)


def relu_stack(x):                       # same, ReLU between them
    return F.conv2d(F.relu(F.conv2d(x, w1, padding=1)), w2, padding=1)


# the equivalent single kernel = the stack's IMPULSE RESPONSE, flipped. F.conv2d is cross-correlation,
# so its impulse response comes out kernel-flipped; flip back to get a kernel usable under F.conv2d.
delta = torch.zeros(1, 1, 9, 9); delta[0, 0, 4, 4] = 1.0
imp = linear_stack(delta)[:, :, 2:7, 2:7]        # 5x5 impulse response around center
keq = torch.flip(imp, dims=(2, 3))               # -> the equivalent 5x5 kernel

x = torch.randn(1, 1, 20, 20)
single = F.conv2d(x, keq, padding=2)             # ONE 5x5 conv
interior = (slice(None), slice(None), slice(2, 18), slice(2, 18))
collapse = (linear_stack(x)[interior] - single[interior]).abs().max().item()
print("a conv is LINEAR, so composing two (no activation) is still ONE conv.")
print("the 2-layer linear stack's impulse response is a 5x5 kernel keq; conv(x, keq) reproduces it:")
print(f"  max|linear_stack(x) - conv(x, keq)| = {collapse:.2e}")
print("  -> depth WITHOUT a nonlinearity buys nothing: two layers = one bigger kernel.")

# %% [markdown]
"""
> **Why flip?** `F.conv2d` is cross-*correlation* (`02`), so its impulse response comes out
> kernel-flipped. Flip it back and `keq` is a genuine kernel that reproduces the stack under
> `F.conv2d`.

**The cure, tested.** Drop a ReLU between the two convs and check superposition — the defining
property of a linear map, `f(x₁+x₂) = f(x₁)+f(x₂)`.
"""

# %%
x1, x2 = torch.randn(1, 1, 20, 20), torch.randn(1, 1, 20, 20)
lin_resid = (linear_stack(x1 + x2) - linear_stack(x1) - linear_stack(x2)).abs().max().item()
relu_resid = (relu_stack(x1 + x2) - relu_stack(x1) - relu_stack(x2)).abs().max().item()
print("drop a ReLU between the two convs and test superposition  f(x1+x2) =? f(x1)+f(x2):")
print(f"  linear stack residual = {lin_resid:.2e}   (linear: holds — so it IS some single conv)")
print(f"  ReLU   stack residual = {relu_resid:.3f}      (broken: NOT any single conv)")
print("  -> ReLU is what lets stacked convs compose edges->strokes->parts instead of")
print("     collapsing to one edge detector. Nonlinearity is what makes depth mean something.")

# %% [markdown]
"""
Once superposition fails, the stack **cannot** equal any single conv, no matter the kernel. That
broken linearity is the whole point: the ReLU is what lets stacked layers compose
`edges → strokes → parts` instead of collapsing back into one edge detector.

Why ReLU specifically (`max(0, ·)`)? It's the modern default — cheap, no saturation, gradients that
don't vanish the way `sigmoid`/`tanh` do. *Any* nonlinearity breaks the collapse; ReLU is the one
that also trains well.
"""

# %% [markdown]
"""
## Recap

| part | claim | payoff |
|---|---|---|
| stacking | depth grows the receptive field, `RF = 1 + L(k−1)` | measured `3×3 → 5×5 → 7×7` |
| linear collapse | two convs, no activation = **one** bigger conv | `conv(x, keq)` reproduces the stack to ~1e-6 |
| the ReLU | a nonlinearity breaks the collapse | superposition residual `~0 → ~2.9` with a ReLU inserted |

**One-sentence compression:** stacking convs widens the receptive field one `(k−1)` step per layer,
but a stack of *linear* convs is algebraically just one bigger conv — the ReLU between them breaks
that superposition, which is the only reason depth adds capacity instead of a fancier kernel.

Next: **`05` — why `stride=2` (H/2, W/2) and ×2 channels?** Stride-2 downsampling makes the
receptive field cover more input pixels in *fewer* layers, and "resolution down, channels up" keeps
compute bounded while features get richer — the `28 → 14 → 7` pyramid you watched in `01`.
"""
