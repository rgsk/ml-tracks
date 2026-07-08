"""
WALKTHROUGH: cnn (top-down) — build a real CNN, watch it read MNIST, THEN open up each piece.

The teaching order is deliberately top-down (see roadmap.md). exp_1 assembles a whole working
classifier and trains it to ~99% on MNIST, with only a rough one-line narration per part — the point
is to SEE it work and get a mental map. Every later experiment opens ONE box of this exact model and
explains the why, measured:

  1. the whole game        — build SmallCNN, train on MNIST, watch acc climb to ~99%, read digits. (here)
  2. open `features`       — what a Conv2d actually computes; naive_conv2d from scratch; feature maps.
  3. why conv, not MLP     — locality, translation equivariance, the parameter blow-up.
  4. conv -> relu -> conv  — depth grows the receptive field; the ReLU stops it collapsing to 1 conv.
  5. stride 2, 2x channels — downsampling: more input pixels in fewer layers; resolution-down/chan-up.
  6. the head + loss        — flatten vs global-average-pool; cross-entropy; wiring checks.
  7. clean model            — assemble the tidy version in new/cnn/, note what carries into the U-Net.

Reuses the cached mnist.npz under diffusion/data (no torchvision, no re-download).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))                 # new/cnn/walkthroughs/cnn (this topic)
_NEW = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))     # new/ (holds diffusion/data)
_FIGS = os.path.join(_HERE, "figures", "experiments")              # experiment-generated figures


def _banner(*lines):
    print("=" * 70)
    for line in lines:
        print(line)
    print("=" * 70)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _to_img(x):
    """(1,28,28)-ish tensor in [-1,1] -> HxW numpy in [0,1] for imshow."""
    return ((x.squeeze().clamp(-1, 1) + 1) / 2).cpu().numpy()


def _mnist(train=True):
    """MNIST images (N,1,28,28) in [-1,1] and labels (N,), from the cached npz. No torchvision."""
    import numpy as np
    path = os.path.join(_NEW, "diffusion", "data", "mnist.npz")    # shared cache under new/
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import urllib.request
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        print(f"  downloading MNIST npz (~11MB) -> {path} ...")
        urllib.request.urlretrieve(url, path)
    d = np.load(path)
    x = d["x_train"] if train else d["x_test"]                     # (N,28,28) uint8 [0,255]
    y = d["y_train"] if train else d["y_test"]
    x = (torch.from_numpy(x).float() / 127.5 - 1.0).unsqueeze(1)   # (N,1,28,28) in [-1,1]
    return x, torch.from_numpy(y).long()


# ---------------------------------------------------------------------------
# The model. Rough narration only — every piece gets its own experiment later.
#
#   features:  a stack of conv -> relu that (a) turns pixels into feature maps and (b) HALVES the
#              spatial size while DOUBLING channels at each downsample (stride 2). So a 28x28 image
#              becomes a small 7x7 grid of 64-dim feature vectors: coarser in space, richer per cell.
#   head:      FLATTEN that 64x7x7 grid into one vector and a Linear -> 10 class scores.
#
# Why conv (not a dense net)? Why relu between convs? Why stride 2 and doubling channels? Why flatten
# (and not average-pool)? All deferred — exp_2..exp_6 open each of these boxes in turn.
# ---------------------------------------------------------------------------
class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),             # 28x28, find local patterns
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # 28->14, channels 16->32
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),  # 14->7,  channels 32->64
        )
        self.head = nn.Sequential(
            nn.Flatten(),                                                      # (B,64,7,7) -> (B,3136)
            nn.Linear(64 * 7 * 7, n_classes),                                  # -> (B,10) class scores
        )

    def forward(self, x):
        return self.head(self.features(x))


def exp_1_whole_game(seed=0, epochs=5, batch_size=128, lr=1e-3):
    """The whole game, top-down: assemble SmallCNN, train it on all of MNIST, and WATCH test accuracy
    climb from ~chance to ~99% — a real digit reader in ~55k params. Then save a grid of held-out
    digits with the model's predictions. No derivations yet; the point is to see it work and get the
    map. exp_2..exp_6 open each piece of this exact model and explain the why."""
    _banner("EXP 1: the whole game — build a CNN, train on MNIST, watch it read digits (~99%)")

    torch.manual_seed(seed)
    dev = _device()

    xtr, ytr = _mnist(train=True)
    xte, yte = _mnist(train=False)
    train = torch.utils.data.TensorDataset(xtr, ytr)
    loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    xte, yte = xte.to(dev), yte.to(dev)

    model = SmallCNN().to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    print("  the model, in one breath:")
    print("    features: conv->relu x3, downsampling 28 -> 14 -> 7 while channels grow 1 -> 16 -> 32 -> 64")
    print("    head:     flatten the 64x7x7 grid -> Linear -> 10 class scores")
    print(f"    params:   {n_params:,} total (compare: an MLP's first dense layer 784x768 = 602,880)\n")

    @torch.no_grad()
    def test_acc():
        model.eval()
        correct = 0
        for i in range(0, len(xte), 2000):
            correct += (model(xte[i:i + 2000]).argmax(1) == yte[i:i + 2000]).sum().item()
        model.train()
        return correct / len(xte)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"  training on {len(xtr)} images ({dev}), {len(loader)} steps/epoch. watch test accuracy:")
    print(f"        before training : test acc {test_acc() * 100:5.2f}%   (~chance, 10 classes)")
    for ep in range(1, epochs + 1):
        run = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
        print(f"        epoch {ep}         : train loss {run / len(loader):.4f}   test acc {test_acc() * 100:5.2f}%")

    # ---- the payoff: read held-out digits -------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model.eval()
    torch.manual_seed(seed)
    idxs = torch.randperm(len(xte))[:40]
    with torch.no_grad():
        preds = model(xte[idxs]).argmax(1).cpu()
    rows, cols = 5, 8
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.25))
    for ax, k in zip(axes.flat, range(len(idxs))):
        i = idxs[k].item()
        pred, true = preds[k].item(), yte[i].item()
        ok = pred == true
        ax.imshow(_to_img(xte[i]), cmap="gray")
        ax.set_title(f"{pred}" if ok else f"{pred}≠{true}", color=("green" if ok else "red"), fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("held-out MNIST digits — CNN predictions (green = correct, red = wrong)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "01_predictions.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out} — held-out digits with the CNN's predictions (nearly all green).")
    print("  That's the whole game: raw pixels -> ~99% digit reader. Next (exp_2) we open the first")
    print("  box — `features` — and ask what a single Conv2d actually computes.")


def run_experiments():
    exp_1_whole_game()


if __name__ == "__main__":
    run_experiments()
