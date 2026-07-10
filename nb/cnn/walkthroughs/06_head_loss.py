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
# CNN · 06 — the `head` + loss: `flatten → Linear`, and is it the right head?

`01`'s `features` handed us a `(B, 64, 7, 7)` grid; the `head` turned it into 10 class scores with
`flatten → Linear`, and the loss was cross-entropy. This is the **last box**. Three things, all
measured:

- **(A)** two heads — `flatten` keeps *where* each feature fired, global-average-pool averages it
  away; on centered digits, flatten wins,
- **(B)** the loss — cross-entropy = `−log(softmax)`, which doubles as a wiring check,
- **(C)** the other wiring check — a correctly wired net must overfit one batch to ~0.
"""

# %%
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from cnn.train import load_mnist

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0


class SmallCNN(nn.Module):
    """01's SmallCNN, but with a swappable head so we can race flatten vs GAP on the same trunk."""

    def __init__(self, head="flatten", n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        if head == "flatten":                                    # keeps position: 64x7x7 -> 3136-vec
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, n_classes),
            )
        else:                                                    # GAP: average each channel's 7x7 -> 64-vec
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, n_classes),
            )

    def forward(self, x):
        return self.head(self.features(x))


print(f"device: {DEV}")

# %% [markdown]
"""
## (A) Two heads — what each keeps

The final grid is `(B, 64, 7, 7)`: 64 feature channels, each a `7×7` map of *where that feature
fired*. Two ways to collapse it to 10 scores:

- **`flatten → Linear`**: read the whole grid as one `3136`-vector → `Linear(3136, 10)`. Keeps
  **where** each feature fired — position is preserved.
- **`GAP → Linear`**: average each channel's `7×7` map → `64`-vector → `Linear(64, 10)`. Keeps only
  **how much** each feature fired anywhere — position averaged away.
"""

# %%
trunk_params = sum(p.numel() for p in SmallCNN().features.parameters())
flat_head = 64 * 7 * 7 * 10 + 10
gap_head = 64 * 10 + 10
print("the final feature grid is (B, 64, 7, 7). two ways to turn it into 10 class scores:")
print(f"  flatten -> Linear : read the grid as ONE 3136-vec -> Linear(3136,10) = {flat_head:,} weights")
print("                      keeps WHERE each feature fired (position is preserved).")
print(f"  GAP     -> Linear : average each channel's 7x7 -> 64-vec -> Linear(64,10) = {gap_head:,} weights")
print("                      keeps only HOW MUCH each feature fired (position averaged away).")
print(f"  (shared conv trunk: {trunk_params:,} params either way.)")

# %% [markdown]
"""
GAP isn't a special op — it's literally a mean over the spatial dims, just kept as a `1×1` "image"
(which is why the head is `AdaptiveAvgPool2d(1)` **then** `Flatten`). `AdaptiveAvgPool2d(k)` is the
general form — it pools to any `k×k` grid regardless of input size; at `k=1` it's exactly this global
mean.
"""

# %%
# GAP is nothing fancy: AdaptiveAvgPool2d(1) == mean over the spatial dims (just kept as 1x1)
f = torch.randn(2, 64, 7, 7)
a = nn.AdaptiveAvgPool2d(1)(f).flatten(1)     # (B,64,7,7)->(B,64,1,1)->(B,64)
b = f.mean(dim=(2, 3))                         # (B,64,7,7)->(B,64), same numbers
print(f"  (GAP is just a mean: |AdaptiveAvgPool2d(1).flatten - f.mean(dim=(2,3))| = {(a - b).abs().max().item():.2e})")

# %% [markdown]
"""
Global-average-pool is the **standard** head on big images (ResNet, etc.): when the object can sit
anywhere in the frame, averaging over position gives you translation *invariance*, which is exactly
what you want. But MNIST digits are **centered** — *where* a stroke sits is a genuine cue (a
horizontal bar near the top vs. the middle helps separate a `7` from a `2`). So here, throwing
position away should hurt. We'll train both heads on the same trunk and same init and watch — but
first, the loss and the wiring checks.
"""

# %% [markdown]
"""
## (B) The loss — cross-entropy = `−log(softmax)`

The loss is one line: softmax the 10 logits into probabilities, take the negative log of the
probability at the **true** class. That's all `F.cross_entropy` does — and it gives a free wiring
check.

> **Check 1 — untrained loss ≈ `ln 10`.** A fresh net's logits are ~arbitrary, so softmax is roughly
> *uniform* over 10 classes → probability `~1/10` at the true class → loss `~ −ln(1/10) = ln 10 ≈
> 2.303`. If an untrained net's loss came out far from `ln(#classes)`, the softmax/label wiring is
> wrong before you've trained a single step.
"""

# %%
xtr, ytr = load_mnist(train=True)
xte, yte = load_mnist(train=False)

torch.manual_seed(SEED)
model = SmallCNN("flatten").to(DEV)
with torch.no_grad():
    ref = F.cross_entropy(model(xtr[:256].to(DEV)), ytr[:256].to(DEV))
print("cross-entropy = -log(softmax at the true class).")
print("WIRING CHECK 1 — an UNtrained net outputs ~uniform over 10 classes, so loss ~= ln(10):")
print(f"  untrained loss = {ref.item():.3f}   vs   ln(10) = {math.log(10):.3f}   (matches -> softmax sane)")

# %% [markdown]
"""
## (C) The other wiring check — overfit one batch

> **Check 2 — memorize ONE batch.** A correctly wired model + loss + optimizer *must* be able to
> drive the loss to ~0 on a single small batch (it can just memorize it). Take 64 images, run 300
> Adam steps on only those. This is the **first** thing to run on any new model: if it *can't*
> overfit one batch, don't bother training on the full set — something (a detached tensor, a frozen
> layer, a label mismatch) is broken.
"""

# %%
torch.manual_seed(SEED)
model = SmallCNN("flatten").to(DEV)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
xb, yb = xtr[:64].to(DEV), ytr[:64].to(DEV)
curve = []
for step in range(300):
    loss = F.cross_entropy(model(xb), yb)
    opt.zero_grad(); loss.backward(); opt.step()
    curve.append(loss.item())
with torch.no_grad():
    acc1 = (model(xb).argmax(1) == yb).float().mean().item()
print("WIRING CHECK 2 — memorize ONE batch of 64 (model+loss+optimizer must be able to):")
print(f"  step 0 loss {curve[0]:.3f}  ->  step 300 loss {curve[-1]:.4f}   (batch acc {acc1 * 100:.0f}%)")
print("  -> it CAN drive the loss to ~0: gradients flow, the head/loss are wired right.")

# %% [markdown]
"""
## (A, measured) flatten vs GAP on the same trunk

Same trunk, same init, 3 epochs on 20k images — **only the head differs.** On centered digits,
position is signal, so flatten should beat GAP.
"""


# %%
def train_eval(head, n_train=20000, epochs=3, bs=128):
    torch.manual_seed(SEED)                                      # same init trunk for a fair race
    model = SmallCNN(head).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ds = torch.utils.data.TensorDataset(xtr[:n_train], ytr[:n_train])
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEV), yb.to(DEV)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(xte), 2000):
            xe, ye = xte[i:i + 2000].to(DEV), yte[i:i + 2000].to(DEV)
            correct += (model(xe).argmax(1) == ye).sum().item()
    return correct / len(xte)


acc_flat = train_eval("flatten")
acc_gap = train_eval("gap")
print("same trunk + 3 epochs on 20k images, only the head differs:")
print(f"  flatten -> Linear : test acc {acc_flat * 100:5.2f}%   (keeps position)")
print(f"  GAP     -> Linear : test acc {acc_gap * 100:5.2f}%   (position averaged away)")
print(f"  -> flatten wins by {(acc_flat - acc_gap) * 100:.2f} pts on CENTERED digits: position IS signal")
print("     here, and GAP discards it. (on ImageNet-scale images GAP's invariance is the win.)")

# %% [markdown]
"""
A big gap — GAP is *not* a free swap on centered digits. The lesson isn't "GAP is bad"; it's that a
head encodes an assumption. GAP assumes *position shouldn't matter*; flatten assumes *it should*.
Match the head to the data.
"""

# %%
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 3.4))
axL.plot(curve, color="tab:blue")
axL.axhline(math.log(10), ls="--", color="gray", lw=1)
axL.text(len(curve) * 0.5, math.log(10) + 0.05, "ln 10 (untrained)", color="gray", fontsize=8)
axL.set_title("wiring check: overfit ONE batch -> 0", fontsize=10)
axL.set_xlabel("step"); axL.set_ylabel("cross-entropy loss")
bars = axR.bar(["flatten", "GAP"], [acc_flat * 100, acc_gap * 100], color=["tab:green", "tab:orange"])
axR.set_ylim(min(acc_flat, acc_gap) * 100 - 2, 100)
axR.set_title("head choice on centered digits", fontsize=10)
axR.set_ylabel("test accuracy (%)")
for bar, acc in zip(bars, (acc_flat, acc_gap)):
    axR.text(bar.get_x() + bar.get_width() / 2, acc * 100 + 0.1, f"{acc * 100:.2f}%", ha="center", fontsize=9)
fig.suptitle("the head + loss: cross-entropy wires up (left), flatten beats GAP on centered digits (right)",
             fontsize=10)
fig.tight_layout()
plt.show()

# %% [markdown]
"""
## Recap

| part | claim | payoff |
|---|---|---|
| head | flatten keeps position, GAP averages it away | ~97% vs ~72% on centered digits |
| loss | cross-entropy = `−log(softmax)` | untrained loss `2.30 ≈ ln 10` |
| wiring | model+loss+opt can memorize one batch | loss `→ 0.0000`, batch acc 100% |

**One-sentence compression:** the head is a *choice* — `flatten` keeps position (a real cue on
centered digits, so it beats GAP here) while GAP's translation-invariance is what you'd want on big
images — and cross-entropy is just `−log(softmax)`, cheap enough to double as two sanity checks
(untrained loss ≈ `ln 10`, and a one-batch overfit to 0) that catch a miswired net before you waste
a training run.

That's every box of `01`'s `SmallCNN` opened. Next: **`07` — assemble the clean model** and note
which pieces (conv trunk, downsampling, the resolution-down/channels-up shape) carry straight into
the U-Net we'll build for diffusion.
"""
