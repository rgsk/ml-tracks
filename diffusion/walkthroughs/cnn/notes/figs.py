"""
figs — generates every figure the CNN notes embed. One function per layer's figures; outputs
go to figs/ namespaced by layer number (01_*, 02_*, ...). The experiments file (../cnn.py) never
plots — all images come from here.

Run:  python figs.py            (regenerates all figures)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

_HERE = os.path.dirname(os.path.abspath(__file__))               # .../cnn/notes
_WALK = os.path.dirname(os.path.dirname(_HERE))                  # .../walkthroughs (for denoiser_and_loss)
if _WALK not in sys.path:
    sys.path.insert(0, _WALK)
_OUT = os.path.join(_HERE, "figs")
os.makedirs(_OUT, exist_ok=True)


def _to_img(x):
    """(...,28,28) in [-1,1] -> 28x28 numpy in [0,1] for imshow."""
    return ((x.squeeze().clamp(-1, 1) + 1) / 2).cpu().numpy()


def _seven():
    from denoiser_and_loss import MNISTData
    test = MNISTData(train=False)
    idx = (test.y == 7).nonzero()[0].item()
    return test.x[idx:idx + 1]                                  # (1,1,28,28) in [-1,1]


# the fixed 3x3 diagonal edge detector used throughout Layer 1.
KERNEL = torch.tensor([[-1., -1., 0.], [-1., 0., 1.], [0., 1., 1.]]).view(1, 1, 3, 3)
SHIFT = 4


# --- Layer 1 figures -------------------------------------------------------
def fig_1_equivariance():
    """2x3 grid: bottom-middle (shift then conv) and bottom-right (conv then shift) are pixel-
    identical on the interior -> that IS featmap(shift x) == shift(featmap x)."""
    x = _seven()
    x_shift = torch.roll(x, (SHIFT, SHIFT), (2, 3))
    fmap = F.conv2d(x, KERNEL, padding=1)
    fmap_of_shift = F.conv2d(x_shift, KERNEL, padding=1)         # shift, then conv
    shift_of_fmap = torch.roll(fmap, (SHIFT, SHIFT), (2, 3))     # conv, then shift

    fig, ax = plt.subplots(2, 3, figsize=(9, 6))
    panels = [
        (0, 0, x, "input  x", "gray"),
        (0, 1, x_shift, "shift(x)", "gray"),
        (0, 2, None, "", None),
        (1, 0, fmap, "featmap(x)", "coolwarm"),
        (1, 1, fmap_of_shift, "featmap(shift(x))\n[shift, then conv]", "coolwarm"),
        (1, 2, shift_of_fmap, "shift(featmap(x))\n[conv, then shift]", "coolwarm"),
    ]
    for r, c, t, title, cmap in panels:
        a = ax[r][c]
        a.set_xticks([]); a.set_yticks([])
        if t is None:
            a.axis("off"); continue
        a.imshow(_to_img(t) if cmap == "gray" else t.squeeze().detach().numpy(),
                 cmap=cmap, vmin=None if cmap == "gray" else -6, vmax=None if cmap == "gray" else 6)
        a.set_title(title, fontsize=11)

    m = SHIFT + 1
    diff = (fmap_of_shift[..., m:-m, m:-m] - shift_of_fmap[..., m:-m, m:-m]).abs().max().item()
    fig.suptitle(f"a conv is translation-EQUIVARIANT:  bottom-middle == bottom-right  "
                 f"(max diff {diff:.1e})", fontsize=12)
    out = os.path.join(_OUT, "01_equivariance.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_1_slide():
    """Animate the 3x3 kernel walking across the digit; the feature map fills in as it goes.
    Makes 'ONE kernel, reused at every position' (weight sharing) visceral."""
    x = _seven()
    img = _to_img(x)
    fmap = F.conv2d(x, KERNEL, padding=1).squeeze().detach().numpy()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8, 4))
    axL.imshow(img, cmap="gray"); axL.set_title("input + sliding 3x3 kernel")
    axL.set_xticks([]); axL.set_yticks([])
    rect = plt.Rectangle((-1.5, -1.5), 3, 3, fill=False, edgecolor="red", lw=2)
    axL.add_patch(rect)

    revealed = np.full((28, 28), np.nan, dtype=np.float32)
    imR = axR.imshow(revealed, cmap="coolwarm", vmin=-6, vmax=6)
    axR.set_title("feature map, filling in"); axR.set_xticks([]); axR.set_yticks([])

    coords = [(r, c) for r in range(28) for c in range(28)]
    step = 7                                                    # reveal 7 pixels/frame

    def update(f):
        for k in range(f * step, min((f + 1) * step, len(coords))):
            r, c = coords[k]
            revealed[r, c] = fmap[r, c]
        imR.set_data(revealed)
        r, c = coords[min((f + 1) * step - 1, len(coords) - 1)]
        rect.set_xy((c - 1.5, r - 1.5))
        return imR, rect

    frames = (len(coords) + step - 1) // step
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=40, blit=False)
    out = os.path.join(_OUT, "01_slide.gif")
    anim.save(out, writer=animation.PillowWriter(fps=25))
    plt.close(fig)
    print(f"wrote {out}")


ALL = [fig_1_equivariance, fig_1_slide]


if __name__ == "__main__":
    for fn in ALL:
        fn()
