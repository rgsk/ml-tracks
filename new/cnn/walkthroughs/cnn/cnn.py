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
_CNN = os.path.dirname(os.path.dirname(_HERE))                     # new/cnn (holds custom/)
_NEW = os.path.dirname(_CNN)                                        # new/ (holds diffusion/data)
_FIGS = os.path.join(_HERE, "figures", "experiments")              # experiment-generated figures
if _CNN not in sys.path:
    sys.path.insert(0, _CNN)                                        # so `from custom.naive_conv2d import ...` works


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


# ---------------------------------------------------------------------------
# EXP 2: open the first box, `features` — what does a single Conv2d compute?
#
# exp_1's `features` was a stack of Conv2d. Here we open ONE of them. Three things:
#   (a) it's just a SLIDING WINDOW multiply-and-sum. We rebuild it from scratch (custom/naive_conv2d,
#       three loops) and match F.conv2d to floating-point noise — no magic, no special math.
#   (b) with hand-SET kernels (not learned) we can SEE what a conv detects: edge kernels trace strokes,
#       a blur smooths. A trained conv just LEARNS kernels like these.
#   (c) the output-size formula out = floor((in + 2p - k)/s) + 1 — it's what set the 28->14->7 pyramid
#       you already watched in exp_1.
# ---------------------------------------------------------------------------
def exp_2_open_features(seed=0):
    """Open `features`: (a) a Conv2d is a windowed multiply-and-sum — naive_conv2d (from scratch) matches
    F.conv2d to ~0; (b) fire hand-set edge/blur kernels at a real digit and SEE the feature maps;
    (c) the output-size formula that produced exp_1's 28->14->7 pyramid."""
    _banner("EXP 2: open `features` — a Conv2d is a sliding multiply-and-sum (feature maps, sizes)")

    # pyrefly: ignore [missing-import]
    from custom.naive_conv2d import naive_conv2d

    # ---- (a) rebuild the conv op from scratch and match F.conv2d --------------------------
    torch.manual_seed(seed)
    x = torch.randn(1, 1, 6, 6)
    w = torch.randn(1, 1, 3, 3)
    mine = naive_conv2d(x, w, stride=1, padding=0)                  # (1,1,4,4)
    ref = F.conv2d(x, w, stride=1, padding=0)
    window = x[0, 0, 0:3, 0:3]
    print("  (a) a conv cell = line the kernel over a window, multiply element-wise, sum.")
    print(f"        ONE cell by hand (top-left):  (window * kernel).sum() = {(window * w[0, 0]).sum().item():+.4f}")
    print(f"        F.conv2d[0,0,0,0]                                     = {ref[0, 0, 0, 0].item():+.4f}")
    print(f"        whole map: max|naive_conv2d - F.conv2d| = {(mine - ref).abs().max().item():.2e}"
          "   -> our 3 loops ARE F.conv2d.\n")

    # ---- (b) hand-set kernels on a real digit -> feature maps -----------------------------
    xte, yte = _mnist(train=False)
    x = xte[(yte == 3).nonzero()[0].item():][:1]                   # a real '3' (1,1,28,28)
    kernels = {
        "identity": [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],  # copies the input (sanity)
        "vertical edge": [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ],  # Sobel-x: left<->right change
        "horizontal edge": [
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1],
        ],  # Sobel-y: up<->down change
        "blur (box)": [[1 / 9] * 3] * 3,  # local average: smooths
    }
    print("  (b) fire four hand-SET 3x3 kernels at a real '3' (padding=1 keeps 28x28) -> feature maps:")
    for name, k in kernels.items():
        fm = F.conv2d(x, torch.tensor(k).float().view(1, 1, 3, 3), padding=1)
        print(f"        {name:<16} range [{fm.min():+.2f}, {fm.max():+.2f}]")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(5 * 2.0, 2.3))
    axes[0].imshow(_to_img(x), cmap="gray"); axes[0].set_title("input '3'", fontsize=9)
    for ax, (name, k) in zip(axes[1:], kernels.items()):
        fm = F.conv2d(x, torch.tensor(k).float().view(1, 1, 3, 3), padding=1).squeeze()
        if name in ("identity", "blur (box)"):
            ax.imshow(fm.numpy(), cmap="gray")                     # near-nonnegative: just a filtered image
        else:
            m = fm.abs().max().item()                              # signed edge map: 0 = neutral center
            ax.imshow(fm.numpy(), cmap="coolwarm", vmin=-m, vmax=m)
        ax.set_title(name, fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("one input, four hand-set kernels -> four feature maps (edges, blur)", fontsize=10)
    fig.tight_layout()
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "02_feature_maps.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"      wrote {out}  (edge kernels trace strokes; blur smooths; a CNN LEARNS kernels like these)\n")

    # ---- (c) the output-size formula that made exp_1's 28->14->7 pyramid ------------------
    def out_size(inp, k, s, p):
        return (inp + 2 * p - k) // s + 1

    print("  (c) each Conv2d's output size: out = floor((in + 2p - k)/s) + 1 — this IS exp_1's pyramid:")
    for inp, k, s, p, role in [(28, 3, 1, 1, "stem  k3,s1,p1"),
                               (28, 3, 2, 1, "down1 k3,s2,p1"),
                               (14, 3, 2, 1, "down2 k3,s2,p1")]:
        real = F.conv2d(torch.zeros(1, 1, inp, inp), torch.zeros(1, 1, k, k), stride=s, padding=p).shape[-1]
        print(f"        {role}:  {inp:>2} -> {out_size(inp, k, s, p):>2}   (F.conv2d gives {real})")
    print("      -> 28 -> 14 -> 7, exactly the shrink you saw.\n")

    # what does the padding p do? drop it (p=0) and the same convs no longer keep the clean sizes:
    print("      the padding's whole job: turn off p and watch the sizes break.")
    for inp, k, s, p, role in [(28, 3, 1, 0, "k3,s1,p0 (stem, unpadded)"),
                               (28, 5, 1, 0, "k5,s1,p0 (bigger kernel)"),
                               (28, 3, 2, 0, "k3,s2,p0 (down, unpadded)")]:
        real = F.conv2d(torch.zeros(1, 1, inp, inp), torch.zeros(1, 1, k, k), stride=s, padding=p).shape[-1]
        print(f"        {role:<26}:  {inp:>2} -> {out_size(inp, k, s, p):>2}   (F.conv2d gives {real})")
    print("      -> s1 unpadded SHRINKS by k-1 every layer (28->26): stack 3 and you've lost 6")
    print("         pixels for nothing. p = k//2 (for k3, 3//2 = p1) exactly cancels that -> same size.")
    print("         s2 unpadded gives 13.")
    print("         not the clean 14 = ceil(28/2). So p keeps stride-1 same-size and stride-2 at ceil(in/2).")
    print("      Next (exp_3): why conv at all, vs the flatten+MLP we could have reused?")


# ---------------------------------------------------------------------------
# EXP 3: why conv at all — vs the flatten+MLP we could have reused?
#
# exp_1's `head` already used nn.Flatten. So we *have* a flatten+dense on hand — why not skip
# `features` and feed flattened pixels straight to a dense net? Because a flatten throws away the
# two things that make an image an image, and a conv keeps both:
#
#   (A) LOCALITY + TRANSLATION. Flattening lays the 28x28 grid into a 784-vector, tying every weight
#       to an ABSOLUTE pixel index — it has no idea index i and i+1 are neighbors, and the SAME digit
#       shifted a few px lands on a nearly disjoint set of indices, so a dense layer sees an almost-new
#       input it must relearn per position. A conv slides ONE kernel over every position, so shifting
#       the input just SHIFTS the feature map: featmap(shift(x)) = shift(featmap(x)) — translation
#       EQUIVARIANCE, which we verify to numerical noise.
#   (B) PARAMETER BLOW-UP. A dense first layer 784 -> H costs 784*H weights (exp_1's aside: 784*768 =
#       602,880). A conv learns one tiny 3x3 kernel per channel and REUSES it at all 784 positions —
#       exp_1's first conv is 160 weights. ~3,700x fewer, and the right bias baked in.
# ---------------------------------------------------------------------------
def exp_3_why_conv(seed=0):
    """Why the conv trunk earns its place over the flatten+MLP exp_1's head already contains:
    (A) shift a digit and the flattened 784-vectors barely overlap (dense must relearn each position),
    yet a conv's feature map merely SHIFTS with the input (equivariance, verified to ~0); (B) a dense
    first layer costs ~600k weights vs a conv kernel's ~160. Save the digit, its shift, and the two
    feature maps lining up."""
    _banner("EXP 3: why conv, not flatten+MLP — locality, translation equivariance, param count")

    # grab a clean '7' to shift around (shifts cleanly, mostly-centered strokes).
    xte, yte = _mnist(train=False)
    x = xte[(yte == 7).nonzero()[0].item():][:1]                   # (1,1,28,28) in [-1,1]
    shift = 4                                                      # pixels down and right

    # ---- (A) the flatten kills translation: pixel-overlap of a digit vs its shift ------------
    # measure overlap in an INK-vs-background sense: map to [0,1] (background 0, ink 1) so the cosine
    # reflects how much INK lands on the same coordinates, not the shared black background.
    x01 = (x + 1) / 2                                              # [0,1]: background 0, ink 1
    x01_shift = torch.roll(x01, shifts=(shift, shift), dims=(2, 3))
    a, b = x01.flatten(), x01_shift.flatten()
    cos_pix = (a @ b) / (a.norm() * b.norm() + 1e-12)             # pixel-overlap of the two vectors
    print("  (A) a flatten+dense ties every weight to an ABSOLUTE pixel index.")
    print(f"      shift the SAME '7' by ({shift},{shift}) px and compare the flattened 784-vectors:")
    print(f"        pixel-overlap cosine(orig, shifted) = {cos_pix.item():.2f}"
          "   (far below 1: little ink lands on the same coordinates)")
    print("      -> to a dense layer the shifted digit is almost a NEW input; it must relearn the")
    print("         digit at every position. Nothing is shared across space.\n")

    # ---- a conv is translation-EQUIVARIANT: featmap(shift(x)) == shift(featmap(x)) -----------
    # one fixed 3x3 kernel as our "feature". a DIAGONAL edge detector (sum-to-zero) traces the whole
    # outline of the 7 — both the top horizontal stroke and the diagonal — so the map reads clearly.
    kernel = torch.tensor([[-1., -1., 0.],
                           [-1., 0., 1.],
                           [0., 1., 1.]]).view(1, 1, 3, 3)        # (out=1,in=1,3,3)
    fmap = F.conv2d(x, kernel, padding=1)                         # feature map of the original
    x_shift = torch.roll(x, shifts=(shift, shift), dims=(2, 3))   # same digit, shifted in [-1,1]
    fmap_of_shift = F.conv2d(x_shift, kernel, padding=1)          # conv of the shifted input
    shift_of_fmap = torch.roll(fmap, shifts=(shift, shift), dims=(2, 3))  # shift of the original's map

    # compare on the INTERIOR: torch.roll wraps at the border (matches an infinite-grid shift), so
    # crop a margin = shift+1 and check where the two ways of wrapping correspond.
    m = shift + 1
    interior = (slice(None), slice(None), slice(m, 28 - m), slice(m, 28 - m))
    diff = (fmap_of_shift[interior] - shift_of_fmap[interior]).abs().max().item()
    scale = fmap.abs().max().item()
    print("  a convolution slides ONE kernel over every position, so shifting the input just")
    print("  shifts the feature map:  featmap(shift(x)) = shift(featmap(x)).  verify (interior):")
    print(f"        max|featmap(shift(x)) - shift(featmap(x))| = {diff:.2e}   (feature scale ~{scale:.1f}"
          " -> equal to numerical noise)")
    print("      -> the conv gets the shifted digit's response FOR FREE. The same stroke detector")
    print("         fires wherever the stroke goes.\n")

    # ---- (B) the parameter count -------------------------------------------------------------
    H = 768                                                       # exp_1's MLP-first-layer aside
    dense_params = 784 * H + H
    conv_out = 16                                                 # exp_1's first conv: Conv2d(1,16,3)
    conv_params = conv_out * 1 * 3 * 3 + conv_out                 # one 3x3 kernel per out channel
    print("  (B) parameter count of the FIRST layer:")
    print(f"        dense 784 -> {H}       : {dense_params:>8,} weights (every pixel<->hidden pair)")
    print(f"        conv 1 -> {conv_out} ch, 3x3  : {conv_params:>8,} weights (one small kernel, reused at 784 spots)")
    print(f"      -> the conv does it with ~{dense_params // conv_params:,}x fewer weights AND the right")
    print("         bias built in. This is why exp_1's model leads with a conv trunk, not a dense net.\n")

    # ---- the payoff: digit, its shift, and the two feature maps lining up ---------------------
    # bottom-middle (shift, then conv) and bottom-right (conv, then shift) are pixel-identical on the
    # interior -> that IS featmap(shift x) == shift(featmap x). edge maps: diverging cmap centered at 0.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _fm(t):
        return t.squeeze().detach().cpu().numpy()

    fig, ax = plt.subplots(2, 3, figsize=(9, 6))
    panels = [
        (0, 0, _to_img(x), "input  x", "gray"),
        (0, 1, _to_img(x_shift), "shift(x)", "gray"),
        (0, 2, None, "", None),
        (1, 0, _fm(fmap), "featmap(x)", "coolwarm"),
        (1, 1, _fm(fmap_of_shift), "featmap(shift(x))\n[shift, then conv]", "coolwarm"),
        (1, 2, _fm(shift_of_fmap), "shift(featmap(x))\n[conv, then shift]", "coolwarm"),
    ]
    for r, c, img, title, cmap in panels:
        a = ax[r][c]
        a.set_xticks([]); a.set_yticks([])
        if img is None:
            a.axis("off"); continue
        kw = dict(cmap=cmap) if cmap == "gray" else dict(cmap=cmap, vmin=-scale, vmax=scale)
        a.imshow(img, **kw)
        a.set_title(title, fontsize=11)
    fig.suptitle(f"a conv is translation-EQUIVARIANT:  bottom-middle == bottom-right  "
                 f"(max diff {diff:.1e})", fontsize=12)
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "03_equivariance.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} — bottom-middle (shift, then conv) and bottom-right (conv, then shift) are")
    print("  the same image: the property a flatten throws away and a conv keeps. Next (exp_4): why")
    print("  conv -> relu -> conv, and what the ReLU between two convs actually buys.")


# ---------------------------------------------------------------------------
# EXP 4: why `conv -> relu -> conv`? Depth, the receptive field, and the load-bearing ReLU.
#
# exp_1's `features` didn't stack bare convs — it stacked conv -> relu -> conv -> relu -> ... Two
# questions that stack answers, and both need measuring:
#
#   (A) WHY STACK AT ALL (receptive field). One k3 conv: each output sees a 3x3 input patch. Put a
#       second k3 on top and each of ITS outputs sees a 3x3 patch of the first map, which each spans
#       3x3 of the input -> 5x5 of the ORIGINAL. Depth widens the window WITHOUT bigger kernels:
#           RF_L = 1 + L·(k-1)      (k3, stride 1: +2 per layer -> 3, 5, 7, ...)
#       We perturb ONE input pixel and count how many outputs move: 3x3 after one conv, 5x5 after two.
#       This is how a small-kernel net eventually "sees" a whole 28x28 digit.
#   (B) WHY THE RELU (it's load-bearing). A conv is LINEAR, and two linear convs compose into... ONE
#       linear conv (a bigger kernel) — depth bought NOTHING. We prove it: the 2-conv no-activation
#       stack's IMPULSE RESPONSE is a 5x5 kernel keq, and conv(x, keq) reproduces the whole stack to
#       ~0. Drop a ReLU between the convs and superposition breaks (f(x1+x2) != f(x1)+f(x2)), so it can
#       no longer equal ANY single conv. The ReLU is exactly what lets stacked layers compose
#       edges -> strokes -> parts instead of collapsing back to one edge detector.
# ---------------------------------------------------------------------------
def exp_4_stack_and_relu(seed=0):
    """Why exp_1 stacks conv -> relu -> conv: (A) depth grows the receptive field — perturb one input
    pixel, count moved outputs: 3x3 after one k3 conv, 5x5 after two, 7x7 after three (RF = 1+L(k-1));
    (B) the ReLU is load-bearing — a linear 2-conv stack collapses to a single 5x5 conv (reproduced to
    ~0), and a ReLU between them breaks that (superposition fails), so depth+nonlinearity buys real
    capacity. Save the growing receptive field."""
    _banner("EXP 4: conv -> relu -> conv — receptive field grows with depth, and ReLU makes it matter")

    torch.manual_seed(seed)

    # ---- (A) receptive field grows with depth --------------------------------------------
    # perturb ONE center pixel and count how many OUTPUT pixels move, after 1/2/3 stacked k3 convs.
    # use ones-kernels so contributions can't cancel — we're counting reach, not values.
    S = 15
    base = torch.zeros(1, 1, S, S)
    pert = base.clone(); pert[0, 0, S // 2, S // 2] = 1.0          # a single lit pixel at the center
    ones = torch.ones(1, 1, 3, 3)

    def n_layers(x, n):
        for _ in range(n):
            x = F.conv2d(x, ones, padding=1)
        return x

    masks = {}
    print("  (A) perturb ONE center input pixel; count OUTPUT pixels that move (an output cell's")
    print("      receptive field), after 1/2/3 stacked k3 convs:")
    for L in (1, 2, 3):
        moved = (n_layers(pert, L) - n_layers(base, L)).abs() > 0
        masks[L] = moved[0, 0]
        ys, xs = moved[0, 0].nonzero(as_tuple=True)
        h = ys.max().item() - ys.min().item() + 1
        w = xs.max().item() - xs.min().item() + 1
        rf = 1 + L * (3 - 1)                                       # RF_L = 1 + L·(k-1) for k3, stride1
        print(f"        {L} conv{'s' if L > 1 else ' '}: moved block {h}x{w}     (formula RF = 1 + {L}·2 = {rf})")
    print("      -> depth widens the window with small kernels; ~3 k3 convs + a stride-2 (exp_5)")
    print("         already let an output cell 'see' most of a 28x28 digit.\n")

    # ---- (B) why the ReLU: a LINEAR conv stack collapses to one conv ----------------------
    C = 8
    w1 = torch.randn(C, 1, 3, 3) * 0.3                             # 1 -> C
    w2 = torch.randn(1, C, 3, 3) * 0.3                             # C -> 1

    def linear_stack(x):                                          # two convs, NO activation
        return F.conv2d(F.conv2d(x, w1, padding=1), w2, padding=1)

    def relu_stack(x):                                            # same, ReLU between them
        return F.conv2d(F.relu(F.conv2d(x, w1, padding=1)), w2, padding=1)

    # the equivalent single kernel = the stack's IMPULSE RESPONSE (delta in -> response out), FLIPPED:
    # F.conv2d is cross-correlation, so its impulse response is the kernel flipped; flip back to
    # recover a kernel that reproduces the stack under F.conv2d.
    delta = torch.zeros(1, 1, 9, 9); delta[0, 0, 4, 4] = 1.0
    imp = linear_stack(delta)[:, :, 2:7, 2:7]                      # 5x5 impulse response around center
    keq = torch.flip(imp, dims=(2, 3))                            # -> the equivalent 5x5 kernel

    x = torch.randn(1, 1, 20, 20)
    single = F.conv2d(x, keq, padding=2)                          # ONE 5x5 conv
    interior = (slice(None), slice(None), slice(2, 18), slice(2, 18))
    collapse = (linear_stack(x)[interior] - single[interior]).abs().max().item()
    print("  (B) a convolution is LINEAR, so composing two of them (no activation) is still ONE conv.")
    print("      the 2-layer linear stack's impulse response is a 5x5 kernel keq; conv(x, keq)")
    print(f"      reproduces the whole stack:  max|linear_stack(x) - conv(x, keq)| = {collapse:.2e}")
    print("      -> depth WITHOUT a nonlinearity buys nothing: two layers = one bigger kernel.\n")

    # superposition test: a linear map satisfies f(x1+x2)=f(x1)+f(x2); ReLU breaks it.
    x1, x2 = torch.randn(1, 1, 20, 20), torch.randn(1, 1, 20, 20)
    lin_resid = (linear_stack(x1 + x2) - linear_stack(x1) - linear_stack(x2)).abs().max().item()
    relu_resid = (relu_stack(x1 + x2) - relu_stack(x1) - relu_stack(x2)).abs().max().item()
    print("      now drop a ReLU between the two convs and test superposition f(x1+x2) =? f(x1)+f(x2):")
    print(f"        linear stack residual = {lin_resid:.2e}   (linear: holds)")
    print(f"        ReLU   stack residual = {relu_resid:.3f}      (broken: NOT any single conv)")
    print("      -> ReLU is what lets stacked convs compose edges->strokes->parts instead of")
    print("         collapsing to one edge detector. Nonlinearity is what makes depth mean something.\n")

    # ---- the payoff: watch the receptive field grow 3x3 -> 5x5 -> 7x7 ---------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "04_receptive_field.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} — the window an output cell sees widens 3x3 -> 5x5 -> 7x7 with each stacked")
    print("  conv. Next (exp_5): stride-2 downsampling (28->14->7) + growing channels — how late")
    print("  layers get a receptive field covering the WHOLE digit without dozens of layers.")


# ---------------------------------------------------------------------------
# EXP 5: why `stride=2` (H/2, W/2) and ×2 channels? The 28->14->7 resolution pyramid.
#
# exp_4 stacked stride-1 convs: the map stays 28x28 and the receptive field crept up by (k-1)=2 per
# layer -> you'd need ~14 layers for one cell to "see" a whole 28x28 digit. exp_1 instead DOWNSAMPLED:
# two of its convs used stride=2, halving H,W (28->14->7) while DOUBLING channels (16->32->64). Three
# things that buys, each measured:
#
#   (A) THE PYRAMID. A stride-2 k3,p1 conv outputs ceil(in/2) (the exp_2 size formula) -> 28->14->7.
#       We rebuild exp_1's exact features stack and check the shapes.
#   (B) RECEPTIVE FIELD EXPLODES. RF_L = RF_{L-1} + (k-1)·∏(strides so far): once you've downsampled,
#       every later (k-1) step is worth `stride` INPUT pixels. Measured by autograd (which input pixels
#       a central output cell actually depends on): stride-1 gives RF 3/5/7, stride-2 gives 3/7/15 —
#       ~4 stride-2 layers cover 28 where stride-1 needs ~14.
#   (C) CHANNELS GROW FOR FREE. Each stride-2 stage cuts positions 4x, so DOUBLING channels still
#       multiplies the activation footprint C·H·W by 2·(1/4) = 1/2 (and conv FLOPs drop too). That's
#       the universal CNN shape: resolution DOWN, channels UP.
# ---------------------------------------------------------------------------
def exp_5_downsample(seed=0):
    """Why exp_1 downsamples with stride-2 and doubles channels: (A) a stride-2 k3 conv halves H,W —
    the 28->14->7 pyramid, shapes checked; (B) the receptive field (measured in INPUT pixels via
    autograd) grows far faster with stride (3/7/15 vs 3/5/7) because the stride product multiplies
    every later step; (C) each downsample quarters positions, so doubling channels still shrinks the
    footprint C·H·W. Save the receptive field of a late cell on a real digit, stride-1 vs stride-2."""
    _banner("EXP 5: stride-2 downsampling — 28->14->7, channels up: the resolution pyramid")

    torch.manual_seed(seed)

    # ---- (A) the resolution pyramid: exp_1's exact features stack -------------------------
    x = torch.randn(1, 1, 28, 28)
    stem = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)               # 28 -> 28 (stride 1)
    down1 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)   # 28 -> 14
    down2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)   # 14 -> 7
    h0 = stem(x); h1 = down1(F.relu(h0)); h2 = down2(F.relu(h1))
    print("  (A) exp_1's features: a stride-1 stem, then two STRIDE-2 convs, DOUBLING channels each:")
    print(f"      {tuple(x.shape)} --stem(1->16,s1)--> {tuple(h0.shape)}"
          f" --down1(16->32,s2)--> {tuple(h1.shape)} --down2(32->64,s2)--> {tuple(h2.shape)}")
    print("      spatial pyramid 28 -> 28 -> 14 -> 7  (k3,s2,p1 -> out = ceil(in/2), the exp_2 formula)\n")

    # ---- (B) receptive field: stride MULTIPLIES the reach of every later step -------------
    # Measure RF in INPUT pixels EXACTLY: pick one central OUTPUT cell, backprop to the input, and the
    # input pixels with nonzero gradient are precisely the ones it depends on.
    S = 31
    ones = torch.ones(1, 1, 3, 3)

    def rf(n_layers, stride):
        xin = torch.zeros(1, 1, S, S, requires_grad=True)
        y = xin
        for _ in range(n_layers):
            y = F.conv2d(y, ones, stride=stride, padding=1)
        c = y.shape[-1] // 2                                   # a central output cell
        y[0, 0, c, c].backward()
        m = xin.grad[0, 0].abs() > 0                           # input pixels it depends on
        ys, _ = m.nonzero(as_tuple=True)
        return ys.max().item() - ys.min().item() + 1          # RF side length, in input pixels

    print("  (B) receptive field of ONE output cell, in INPUT pixels (autograd: which input pixels")
    print("      the cell actually depends on), for stride-1 vs stride-2 stacks of k3 convs:")
    print("        layers |  stride-1 RF  |  stride-2 RF")
    for L in (1, 2, 3):
        print(f"           {L}    |      {rf(L, 1):>2}       |      {rf(L, 2):>2}")
    print("      RF_L = RF_{L-1} + (k-1)·∏(strides): the stride PRODUCT multiplies every later step.")
    print("      -> ~14 stride-1 convs to reach RF 28, but only ~4 stride-2 (3->7->15->31): stride is")
    print("         how a small-kernel net comes to 'see' the whole digit.\n")

    # ---- (C) why channels grow as resolution shrinks: the footprint stays bounded ---------
    print("  (C) footprint C·H·W per stage. once channels DOUBLE while area QUARTERS, each downsample")
    print("      multiplies the footprint by 2·(1/4) = 1/2 — resolution traded for depth cheaply:")
    prev = None
    for name, t in (("input", x), ("stem  1->16", h0), ("down1 16->32", h1), ("down2 32->64", h2)):
        c, hh, ww = t.shape[1], t.shape[2], t.shape[3]
        v = c * hh * ww
        ratio = "(input)" if prev is None else f"×{v / prev:.2f} vs prev"
        print(f"        {name:<12} {c:>2}·{hh}·{ww} = {v:>5} values   {ratio}")
        prev = v
    print("      -> stride-2 conv FLOPs drop ~4x per stage too, which is what makes deep, wide late")
    print("         layers affordable.\n")

    # ---- the payoff: RF of a late cell on a real digit, stride-1 vs stride-2 --------------
    # draw the exact RF box (from rf() above) centered on a real '3' after 3 convs, both strides.
    xte, yte = _mnist(train=False)
    digit = _to_img(xte[(yte == 3).nonzero()[0].item():][:1])     # a real '3', 28x28 in [0,1]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "05_downsample_rf.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} — same 3 convs, but the stride-2 cell's receptive field (15x15) covers half")
    print("  the digit where the stride-1 cell (7x7) sees a sliver. Next (exp_6): the head + loss —")
    print("  flatten vs global-average-pool, cross-entropy, and two wiring checks.")


def run_experiments():
    # exp_1_whole_game()
    exp_2_open_features()
    # exp_3_why_conv()
    # exp_4_stack_and_relu()
    # exp_5_downsample()


if __name__ == "__main__":
    run_experiments()
