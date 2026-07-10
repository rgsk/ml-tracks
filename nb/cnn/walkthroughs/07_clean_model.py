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
# CNN · 07 — the clean model, and what carries into the diffusion U-Net

This is the capstone. There's nothing new to *measure* — because the track is top-down, the clean
model already appeared in `01`. What `02`–`06` did was **justify every line of it**: the conv op and
the `28→14→7` pyramid (`02`), conv over a flatten+MLP (`03`), the `conv → relu → conv` stack (`04`),
stride-2 downsampling with growing channels (`05`), the `flatten → Linear` head + cross-entropy
(`06`).

So this notebook does two small things:

1. Pull the pieces out of the notebooks into two importable modules — `nb/cnn/model.py` (the pure
   `SmallCNN`) and `nb/cnn/train.py` (the loop, eval, checkpointing, and prediction figure) — and
   prove it's the *same* model by loading `01`'s trained weights into it.
2. Note which pieces carry straight into the U-Net we'll build for diffusion.

`01` and `06` keep their loops inline on purpose (seeing the loop *is* the lesson); here, where we're
just demonstrating the finished artifact, we import the reusable helpers.
"""

# %%
import torch
import matplotlib.pyplot as plt

from cnn.train import load_mnist, test_acc, plot_predictions, CKPT, ROOT, DEV

print(f"device: {DEV}")

# %% [markdown]
"""
## 1. The clean, importable model

`model.py` is `01`'s `SmallCNN`, verbatim — a `features` conv trunk then a `flatten → Linear` head.
Sanity checks: the parameter count is exactly the **54,666** we saw in `01`, and a dummy batch flows
to `(B, 10)` class scores.
"""

# %%
from cnn.model import SmallCNN

model = SmallCNN().to(DEV)
n_params = sum(p.numel() for p in model.parameters())
print("SmallCNN imported from nb/cnn/model.py")
print(model)
print(f"\nparam count: {n_params:,}")
assert n_params == 54_666, f"expected 54,666 params, got {n_params:,}"

dummy = torch.randn(4, 1, 28, 28, device=DEV)      # a batch of 4 fake 28x28 images
out = model(dummy)
print(f"forward: {tuple(dummy.shape)} -> {tuple(out.shape)}   (B images -> B x 10 class scores)")
assert out.shape == (4, 10)
print("-> param count and forward shape match 01. ✓")

# %% [markdown]
"""
## Prove it's the *same* model — load `01`'s trained weights

`model.py`'s module layout (`features.{0,2,4}`, `head.1`) is identical to `01`'s, so the state dict
`01` saved should load with **no missing or unexpected keys** (`strict=True`). And if these really
are the trained weights, the imported module should read held-out digits at ~99% — not the ~10% of a
fresh net.
"""

# %%
if CKPT.exists():
    state = torch.load(CKPT, map_location=DEV)
    model.load_state_dict(state)                   # strict=True (default): raises on ANY key mismatch
    print(f"loaded {CKPT.relative_to(ROOT)} into the imported SmallCNN — no key mismatch. ✓")

    xte, yte = load_mnist(train=False)                         # test_acc moves batches to the device
    print(f"test accuracy of the loaded weights: {test_acc(model, xte, yte) * 100:.2f}%")
    print("-> ~99%, not ~10%: these ARE 01's trained weights, in the extracted module. Same model.")
else:
    print(f"no checkpoint at {CKPT.relative_to(ROOT)} — run 01_whole_game first to create it.")

# %% [markdown]
"""
### Read held-out digits

And the point of it all — the imported-and-loaded module reading held-out digits it never trained on
(green = correct, red = wrong). Same payoff as `01`, now from the clean modules. This is
`train.plot_predictions` — the very same helper the `python nb/cnn/train.py` CLI uses to save the
figure to `outputs/cnn/`, here rendered inline (we pass no `save_path`).
"""

# %%
if CKPT.exists():
    fig, n_correct = plot_predictions(model, xte, yte)         # no save_path -> render inline
    plt.show()
    print(f"{n_correct}/40 correct on this held-out sample.")

# %% [markdown]
"""
## 2. What carries into the diffusion U-Net

The point of learning CNNs *here* is that the diffusion model's denoiser is a **U-Net**, and its
whole encoder half is exactly the machinery we just justified. What transfers straight across:

- **The conv trunk.** `conv → relu` stacks over feature maps — the same primitive, same reasons
  (locality + weight sharing + translation equivariance, `03`).
- **The downsampling pyramid.** "resolution **down**, channels **up**" via stride-2 (`05`) *is* the
  U-Net encoder: `28 → 14 → 7 …`, channels growing as area shrinks, so a few layers see the whole
  image at bounded cost.
- **Receptive field via depth + stride** (`04`, `05`) — how a small-kernel net comes to see globally.

What's **new** in a U-Net (the next track picks these up):

- It isn't a classifier: instead of a `flatten → Linear` head that throws the spatial grid away, the
  encoder pyramid is **mirrored by a decoder** that *upsamples* back to `28×28`, with **skip
  connections** from encoder to decoder so fine detail survives the bottleneck. Output is
  **image-shaped** (the predicted noise), not 10 scores.
- **Conditioning:** the denoiser also takes the diffusion timestep `t` (and often a class/text
  embedding), injected into the conv blocks.

So `SmallCNN.features` ≈ the U-Net's downsampling encoder. Build the mirror-image decoder + skips +
timestep conditioning around it, and the classifier becomes a denoiser.

**That's the CNN walkthrough.** Every box of the model in `01` is opened, justified, and re-assembled
— and the encoder we built is the half of the U-Net we'll reuse for diffusion.
"""
