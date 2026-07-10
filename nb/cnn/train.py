"""train.py — the reusable training/eval/plot procedure for the CNN walkthrough (nb/cnn).

Separation of concerns (mirrors llm/): `model.py` holds the pure `SmallCNN` nn.Module; this file owns
everything *procedural* — loading MNIST, the training loop, checkpointing, evaluation, and the
held-out prediction figure. The teaching notebooks 01 and 06 deliberately keep their loops INLINE
(seeing the loop is the lesson); this module is what 07 imports to demonstrate the finished artifact,
and what runs as a script to (re)generate the checkpoint + figure:

    python -m cnn.train           # load-or-train -> checkpoints/smallcnn.pt, figure -> outputs/cnn/

Every plotting helper RETURNS its figure and only writes a file when given `save_path`, so the same
function renders inline in a notebook and saves to outputs/ from the CLI.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from cnn.model import SmallCNN

# --- paths (anchored to the repo root, so this works from any cwd) ---
_HERE = Path(__file__).resolve().parent                        # nb/cnn


def _repo_root() -> Path:
    for d in (_HERE, *_HERE.parents):
        if (d / "pyproject.toml").exists():
            return d
    return _HERE


ROOT = _repo_root()
DATA = ROOT / "nb" / "data" / "mnist.npz"                      # nb-local MNIST cache (downloaded on demand, no torchvision)
CKPT = ROOT / "nb" / "cnn" / "checkpoints" / "smallcnn.pt"     # trained weights (gitignored)
OUTPUTS = _HERE / "outputs"                                    # saved figures (gitignored)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_mnist(train=True):
    """MNIST images (N,1,28,28) in [-1,1] and labels (N,), from the cached npz. No torchvision."""
    if not DATA.exists():                                          # self-contained: fetch on first use
        DATA.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        print(f"  downloading MNIST npz (~11MB) -> {DATA.relative_to(ROOT)} ...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", DATA)
    d = np.load(DATA)
    x = d["x_train"] if train else d["x_test"]
    y = d["y_train"] if train else d["y_test"]
    x = (torch.from_numpy(x).float() / 127.5 - 1.0).unsqueeze(1)   # (N,1,28,28) in [-1,1]
    return x, torch.from_numpy(y).long()


def to_img(x):
    """(1,28,28)-ish tensor in [-1,1] -> HxW numpy in [0,1] for imshow."""
    return ((x.squeeze().clamp(-1, 1) + 1) / 2).cpu().numpy()


@torch.no_grad()
def test_acc(model, xte, yte):
    """Top-1 accuracy over the test set. Data may live on CPU; batches move to the model's device."""
    dev = next(model.parameters()).device
    was_training = model.training
    model.eval()
    correct = 0
    for i in range(0, len(xte), 2000):
        pred = model(xte[i:i + 2000].to(dev)).argmax(1).cpu()
        correct += (pred == yte[i:i + 2000]).sum().item()
    if was_training:
        model.train()
    return correct / len(xte)


def train(model, xtr, ytr, xte, yte, epochs=5, batch=128, lr=1e-3, log=print):
    """Train `model` on MNIST with Adam + cross-entropy. Returns [(epoch, train_loss, test_acc), ...]."""
    dev = next(model.parameters()).device
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xtr, ytr), batch_size=batch, shuffle=True
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    log(f"training on {len(xtr)} images ({dev}), {len(loader)} steps/epoch")
    log(f"  before training : test acc {test_acc(model, xte, yte) * 100:5.2f}%   (~chance, 10 classes)")
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
        acc = test_acc(model, xte, yte)
        history.append((ep, run / len(loader), acc))
        log(f"  epoch {ep}         : train loss {run / len(loader):.4f}   test acc {acc * 100:5.2f}%")
    return history


def load_or_train(dev=DEV, epochs=5):
    """Return a SmallCNN with 01's weights: load checkpoints/smallcnn.pt if present, else train + save.
    Returns (model, trained) where `trained` is True if it had to train from scratch."""
    model = SmallCNN().to(dev)
    if CKPT.exists():
        model.load_state_dict(torch.load(CKPT, map_location=dev))
        return model, False
    xtr, ytr = load_mnist(train=True)
    xte, yte = load_mnist(train=False)
    train(model, xtr, ytr, xte, yte, epochs=epochs)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT)
    return model, True


def plot_predictions(model, xte, yte, n=40, save_path=None):
    """Grid of `n` held-out digits with the model's predictions (green=correct, red=wrong).
    Returns (fig, n_correct). Writes a PNG only if `save_path` is given — otherwise leaves the figure
    open so a notebook can render it inline."""
    dev = next(model.parameters()).device
    model.eval()
    torch.manual_seed(0)
    idxs = torch.randperm(len(xte))[:n]
    with torch.no_grad():
        preds = model(xte[idxs].to(dev)).argmax(1).cpu()

    rows, cols = 5, 8
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.25))
    for ax, k in zip(axes.flat, range(len(idxs))):
        i = idxs[k].item()
        pred, true = preds[k].item(), yte[i].item()
        ok = pred == true
        ax.imshow(to_img(xte[i]), cmap="gray")
        ax.set_title(f"{pred}" if ok else f"{pred}≠{true}", color=("green" if ok else "red"), fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("held-out MNIST digits — SmallCNN predictions (green = correct, red = wrong)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    n_correct = (preds == yte[idxs].cpu()).sum().item()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
    return fig, n_correct


def main():
    """CLI: (re)build the trained model and its held-out-prediction figure from scratch."""
    print(f"device: {DEV}")
    model, trained = load_or_train()
    print("trained from scratch" if trained else f"loaded existing {CKPT.relative_to(ROOT)}")
    xte, yte = load_mnist(train=False)
    print(f"test accuracy: {test_acc(model, xte, yte) * 100:.2f}%")
    out = OUTPUTS / "predictions.png"
    _, n_correct = plot_predictions(model, xte, yte, save_path=out)
    print(f"{n_correct}/40 correct on the held-out sample; wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
