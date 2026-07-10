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
# CNN · 01 — the whole game

Top-down: before we take anything apart, let's build a **real CNN, train it, and watch it read
digits**. By the end of this notebook you have a working ~99% MNIST classifier and a rough mental
map of its parts. *Why* each part is shaped the way it is — that's `02` onward, each opening one box
of *this exact model*.

> This is the map. Every later notebook zooms into **one box you already ran here.**
"""

# %%
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def _repo_root() -> Path:
    """Walk up from the cwd to the folder holding pyproject.toml, so paths work no matter where
    the kernel is launched from."""
    here = Path.cwd()
    for d in (here, *here.parents):
        if (d / "pyproject.toml").exists():
            return d
    return here


ROOT = _repo_root()
DATA = ROOT / "nb" / "data" / "mnist.npz"                  # nb-local MNIST cache (downloaded on demand, no torchvision)
CKPT_DIR = ROOT / "nb" / "cnn" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEV}")


def load_mnist(train=True):
    """MNIST images (N,1,28,28) in [-1,1] and labels (N,), from the cached npz. No torchvision."""
    if not DATA.exists():                                          # self-contained: fetch on first use
        DATA.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        print(f"  downloading MNIST npz (~11MB) -> {DATA.relative_to(ROOT)} ...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", DATA)
    d = np.load(DATA)
    x = d["x_train"] if train else d["x_test"]                 # (N,28,28) uint8 [0,255]
    y = d["y_train"] if train else d["y_test"]
    x = (torch.from_numpy(x).float() / 127.5 - 1.0).unsqueeze(1)   # (N,1,28,28) in [-1,1]
    return x, torch.from_numpy(y).long()


def to_img(x):
    """(1,28,28)-ish tensor in [-1,1] -> HxW numpy in [0,1] for imshow."""
    return ((x.squeeze().clamp(-1, 1) + 1) / 2).cpu().numpy()


# %% [markdown]
"""
## The model in one breath

Rough narration, no rigor yet — just enough that it isn't a black box:

- **`features`** turns raw pixels into feature maps. Each `Conv2d` slides small learnable filters
  over the image to detect local patterns (edges → strokes → parts). The `stride=2` convs **halve
  the spatial size** (28 → 14 → 7) while we **double the channels** (16 → 32 → 64): coarser in
  space, richer per location. Out comes a small `64 × 7 × 7` grid of feature vectors.
- **`head`** flattens that grid into one `3136`-vector and a `Linear` maps it to **10 class scores**.
- **`ReLU`** is the nonlinearity between convs (why it's load-bearing is `04`).

That's it — `raw pixels → features → 10 scores`. *Why* conv (not a dense net)? Why `stride=2` and
doubling channels? Why flatten (not average-pool)? All deferred — `02`–`06` open each box in turn.
"""


# %%
class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),              # 28x28, find local patterns
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 28->14, channels 16->32
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 14->7,  channels 32->64
        )
        self.head = nn.Sequential(
            nn.Flatten(),                                          # (B,64,7,7) -> (B,3136)
            nn.Linear(64 * 7 * 7, n_classes),                      # -> (B,10) class scores
        )

    def forward(self, x):
        return self.head(self.features(x))


torch.manual_seed(0)
model = SmallCNN().to(DEV)
n_params = sum(p.numel() for p in model.parameters())
print(f"params: {n_params:,} total")
print(f"(for scale, one MLP dense layer 784x768 alone is {784 * 768:,})")

# %% [markdown]
"""
## Watch it learn

Train on all 60k MNIST images, cross-entropy loss, Adam, 5 epochs. We print test accuracy before
training (≈ chance) and after each epoch — watch it climb.
"""

# %%
EPOCHS, BATCH, LR = 5, 128, 1e-3

xtr, ytr = load_mnist(train=True)
xte, yte = load_mnist(train=False)
xte, yte = xte.to(DEV), yte.to(DEV)
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(xtr, ytr), batch_size=BATCH, shuffle=True
)


@torch.no_grad()
def test_acc():
    model.eval()
    correct = 0
    for i in range(0, len(xte), 2000):
        correct += (model(xte[i:i + 2000]).argmax(1) == yte[i:i + 2000]).sum().item()
    model.train()
    return correct / len(xte)


opt = torch.optim.Adam(model.parameters(), lr=LR)
print(f"training on {len(xtr)} images ({DEV}), {len(loader)} steps/epoch")
print(f"  before training : test acc {test_acc() * 100:5.2f}%   (~chance, 10 classes)")
for ep in range(1, EPOCHS + 1):
    run = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEV), yb.to(DEV)
        loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        run += loss.item()
    print(f"  epoch {ep}         : train loss {run / len(loader):.4f}   test acc {test_acc() * 100:5.2f}%")

# %% [markdown]
"""
From ~chance to **~99% in a few seconds** — a real digit reader in ~55k params, while a single MLP
dense layer (`784×768`) alone would be ~600k. The convolutional structure does a lot with very few
weights; *why* is the whole rest of the walkthrough.

We save the trained weights so the later notebooks can load this exact model instead of retraining.
"""

# %%
ckpt = CKPT_DIR / "smallcnn.pt"
torch.save(model.state_dict(), ckpt)
print(f"saved trained weights -> {ckpt.relative_to(ROOT)}")

# %% [markdown]
"""
## Read held-out digits

The payoff — predictions on 40 test digits the model never trained on (green = correct, red = wrong).
"""

# %%
model.eval()
torch.manual_seed(0)
idxs = torch.randperm(len(xte))[:40]
with torch.no_grad():
    preds = model(xte[idxs]).argmax(1).cpu()

rows, cols = 5, 8
fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.25))
for ax, k in zip(axes.flat, range(len(idxs))):
    i = idxs[k].item()
    pred, true = preds[k].item(), yte[i].item()
    ok = pred == true
    ax.imshow(to_img(xte[i]), cmap="gray")
    ax.set_title(f"{pred}" if ok else f"{pred}≠{true}", color=("green" if ok else "red"), fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("held-out MNIST digits — CNN predictions (green = correct, red = wrong)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()

n_correct = (preds == yte[idxs].cpu()).sum().item()
print(f"{n_correct}/40 correct on this held-out sample.")

# %% [markdown]
"""
## The map — what we open next

You now have a working model and a rough picture. Every notebook below picks **one box you just
ran** and explains why it's built that way — measured, not asserted:

| next | opens | the question |
|---|---|---|
| `02` | `Conv2d` inside `features` | what does a conv actually *compute*? (build one from scratch) |
| `03` | conv vs the MLP we had | why not just flatten + dense? (locality, translation equivariance) |
| `04` | `conv → relu → conv` | why depth, and why the ReLU between them is load-bearing |
| `05` | `stride=2`, ×2 channels | why downsample — more reach in fewer layers, bounded cost |
| `06` | `flatten → Linear` head | is flatten the right head? (vs global-average-pool) + wiring checks |
| `07` | the whole model | assemble the clean version; what carries into the diffusion U-Net |

Next: **`02` — open `features`: what is a `Conv2d`, really?**
"""
