"""
figures — EXTRA figures for the CNN notes that no experiment already produces. Anything an `exp_*`
in ../cnn.py plots is written by that experiment straight into figures/experiments/ (single source of
truth); this file only holds figures with no experiment counterpart (e.g. animations). To keep
ownership obvious, these go in figures/generated/ (experiments/ = experiment-owned, handmade/ =
hand-drawn diagrams), namespaced by exp number (01_*, 02_*, ...).

Run:  python figures.py            (regenerates the generated/ figures)
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

_HERE = os.path.dirname(os.path.abspath(__file__))               # .../cnn/figures
_TOPIC = os.path.dirname(_HERE)                                  # .../cnn (holds cnn.py)
if _TOPIC not in sys.path:
    sys.path.insert(0, _TOPIC)                                   # so `from cnn import ...` works
_OUT = os.path.join(_HERE, "generated")                         # extras only; experiment figures live in experiments/
os.makedirs(_OUT, exist_ok=True)

# reuse cnn.py's cached-npz loader and imshow helper — one source of truth for data + display.
# pyrefly: ignore [missing-import]
from cnn import _mnist, _to_img


def _three():
    """A real '3' (1,1,28,28) in [-1,1] — the same digit exp_2 fires its hand-set kernels at."""
    xte, yte = _mnist(train=False)
    idx = (yte == 3).nonzero()[0].item()
    return xte[idx:idx + 1]


# the vertical-edge (Sobel-x) kernel from exp_2 — sum-to-zero, so it fires only where brightness changes.
KERNEL = torch.tensor(
    [
        [-1.0, 0.0, 1.0],
        [-2.0, 0.0, 2.0],
        [-1.0, 0.0, 1.0],
    ]
).view(1, 1, 3, 3)


# --- exp 2 figures ---------------------------------------------------------
# NB: exp_2's feature-map still (02_feature_maps.png) is produced by exp_2_open_features in ../cnn.py,
# not here. This animation has no experiment counterpart, so it lives here.
def fig_2_slide():
    """Animate the 3x3 kernel walking across the digit; the feature map fills in as it goes.
    Makes 'ONE kernel, reused at every position' (weight sharing) visceral."""
    x = _three()
    img = _to_img(x)
    fmap = F.conv2d(x, KERNEL, padding=1).squeeze().detach().numpy()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8, 4))
    axL.imshow(img, cmap="gray"); axL.set_title("input + sliding 3x3 kernel")
    axL.set_xticks([]); axL.set_yticks([])
    rect = plt.Rectangle((-1.5, -1.5), 3, 3, fill=False, edgecolor="red", lw=2)
    axL.add_patch(rect)

    revealed = np.full((28, 28), np.nan, dtype=np.float32)
    m = float(np.abs(fmap).max())
    imR = axR.imshow(revealed, cmap="coolwarm", vmin=-m, vmax=m)
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
    out = os.path.join(_OUT, "02_slide.gif")
    anim.save(out, writer=animation.PillowWriter(fps=25))
    plt.close(fig)
    print(f"wrote {out}")


ALL = [fig_2_slide]


if __name__ == "__main__":
    for fn in ALL:
        fn()
