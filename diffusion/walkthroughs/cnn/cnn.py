"""
WALKTHROUGH: cnn — why a convolution beats a flatten-then-MLP on images, built one layer
at a time, ending in a CNN that reads MNIST digits.

Our denoiser so far (denoiser_and_loss.py) is an MLP: it FLATTENS the 28x28 image into a
784-vector and hits it with dense layers. That works, but it throws away the one thing an
image has going for it — its 2-D spatial structure. A pixel only means something in concert
with its NEIGHBORS (an edge, a stroke, a loop), and the same stroke means the same thing
whether it sits in the top-left or the middle. A flattened dense layer knows none of this:
every one of the 784 inputs is an independent knob, and a digit nudged a few pixels over is,
to it, an almost entirely different input it must relearn from scratch.

A CONVOLUTION fixes exactly that. It slides ONE small kernel over every position, so it (1)
shares weights across the whole image (a stroke detector learned once fires everywhere) and
(2) is translation-EQUIVARIANT (shift the input -> the feature map just shifts too). That is
why every image model — including the U-Net we'll swap in for the MLP denoiser later — is
built from convs. This walkthrough builds that machinery from the op up, then trains a small
CNN classifier so we can WATCH accuracy climb, before we reuse the pieces for image diffusion.

Layers (run each `exp_*`, watch the output, then say "next"):
  1. WHY conv, not a flatten+MLP — the flatten destroys locality and translation structure:
     a digit shifted a few pixels is a wildly different 784-vector (so an MLP must relearn
     every position), while a conv's feature map merely SHIFTS. Plus the parameter blow-up:
     one dense layer = 100,000s of weights; one conv kernel = a few hundred, reused everywhere. (here)
  2. the convolution op itself — cross-correlation of a small kernel sliding over the image;
     hand-set edge kernels -> SEE the feature maps; the output-size formula.
  3. depth: stacking convs + channels + the ReLU — receptive field grows with depth, and
     WITHOUT a nonlinearity a stack of convs collapses to a single conv (verify it) — ReLU is
     what makes depth buy anything.
  4. downsampling — a stride-2 conv halves H,W and we grow channels: the resolution pyramid
     28 -> 14 -> 7, receptive field covering the whole digit.
  5. the head + loss — global-average-pool the final map to a vector, Linear -> 10 logits,
     cross-entropy; untrained loss ~ ln 10, overfit one batch -> 0 (wiring test).
  6. train it — the real loop on MNIST: accuracy climbs to ~99%, far above the MLP for a
     fraction of the params; save a grid of held-out digits with predicted labels.

Reuses denoiser_and_loss.py's MNISTData (the npz loader + [-1,1] normalization). No torchvision.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))   # diffusion/walkthroughs/cnn
_WALK = os.path.dirname(_HERE)                        # diffusion/walkthroughs (holds denoiser_and_loss)
_FIGS = os.path.join(_HERE, "notes", "figs")          # single home for note figures (00_*, 01_*, ...)
if _WALK not in sys.path:
    sys.path.insert(0, _WALK)                         # so `import denoiser_and_loss` works


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


# ---------------------------------------------------------------------------
# LAYER 1: why conv, not a flatten+MLP.
#
# The MLP denoiser starts with `x = x.flatten(1)` — the (1,28,28) image becomes a 784-vector,
# and then a dense layer W (784 -> hidden) mixes all 784 pixels with independent weights. Two
# things are wrong with that for images:
#
#   (A) NO locality / translation structure. Flattening throws away which pixels are neighbors.
#       Worse, it hard-codes MEANING TO ABSOLUTE POSITION: the weight on "pixel 407" is learned
#       separately from the weight on "pixel 408". So the SAME digit shifted a few pixels lands
#       on a different set of input coordinates and looks, to the dense layer, like a nearly new
#       input it must relearn. We measure this below: shift a '7' by a few pixels and the pixel-
#       overlap between the two flattened vectors collapses — an MLP gets no free ride from the
#       shift. A convolution, by contrast, applies ONE kernel at every position, so shifting the
#       input just SHIFTS the output (translation equivariance): featmap(shift(x)) = shift(featmap(x)).
#       We verify that equality holds to numerical noise.
#
#   (B) PARAMETER BLOW-UP. A dense layer 784 -> H has 784·H weights (H=768 as in our denoiser =>
#       ~600k weights in the FIRST layer alone). A conv learns one tiny kernel (e.g. 3x3 = 9
#       weights per in/out channel pair) and REUSES it across all 784 positions — a few hundred
#       weights do the whole layer. Fewer params AND the right inductive bias.
#
# So this layer proves, with numbers and a picture, the two reasons every image net is convs.
# ---------------------------------------------------------------------------
def exp_1_why_conv(seed=0):
    """Show the two reasons to prefer conv over flatten+MLP on images: (A) a small pixel shift
    destroys the overlap of the flattened vectors (so an MLP must relearn each position), yet a
    conv's feature map merely SHIFTS with the input (equivariance, verified to ~0); (B) a dense
    first layer costs ~600k weights vs a conv kernel's few hundred. Save a picture of the digit,
    its shift, and their feature maps lining up."""
    _banner("LAYER 1: why conv, not flatten+MLP — locality, translation equivariance, param count")

    from denoiser_and_loss import MNISTData
    test = MNISTData(train=False)

    # grab a clean '7' to shift around.
    idx = (test.y == 7).nonzero()[0].item()
    x = test.x[idx:idx + 1]                                   # (1,1,28,28) in [-1,1]
    shift = 4                                                 # pixels down and right

    # ---- (A) the flatten kills translation: pixel-overlap of a digit vs its shift ------------
    # measure overlap in an INK-vs-background sense: map to [0,1] (background 0, ink 1) so the
    # cosine similarity reflects how much INK lands on the same coordinates, not the shared black.
    x01 = (x + 1) / 2                                         # [0,1]: background 0, ink 1
    x01_shift = torch.roll(x01, shifts=(shift, shift), dims=(2, 3))
    a = x01.flatten()
    b = x01_shift.flatten()
    cos_pix = (a @ b) / (a.norm() * b.norm() + 1e-12)         # pixel-overlap of the two vectors
    print("  (A) a flatten+MLP hard-codes meaning to absolute pixel position.")
    print(f"      shift the SAME '7' by ({shift},{shift}) px and compare the flattened 784-vectors:")
    print(f"        pixel-overlap cosine(orig, shifted) = {cos_pix.item():.2f}"
          "   (far below 1: little ink lands on the same coordinates)")
    print("      -> to a dense layer the shifted digit is almost a NEW input; it must learn the")
    print("         digit again at every position. Nothing is shared across space.\n")

    # ---- a conv is translation-EQUIVARIANT: featmap(shift(x)) == shift(featmap(x)) -----------
    # use one fixed 3x3 kernel (a diagonal edge detector) as our "feature". padding=1 keeps 28x28.
    kernel = torch.tensor([[-1., -1., 0.],
                           [-1., 0., 1.],
                           [0., 1., 1.]]).view(1, 1, 3, 3)    # (out=1,in=1,3,3)
    fmap = F.conv2d(x, kernel, padding=1)                     # feature map of the original
    x_shift = torch.roll(x, shifts=(shift, shift), dims=(2, 3))     # same digit, shifted in [-1,1]
    fmap_of_shift = F.conv2d(x_shift, kernel, padding=1)      # feature map of the shifted input
    shift_of_fmap = torch.roll(fmap, shifts=(shift, shift), dims=(2, 3))  # shift of the original's map

    # compare on the INTERIOR (roll wraps at the border; the digit is centered so the interior is
    # where the equality is exact up to the wrap). crop a margin = shift.
    m = shift + 1
    interior = (slice(None), slice(None), slice(m, 28 - m), slice(m, 28 - m))
    diff = (fmap_of_shift[interior] - shift_of_fmap[interior]).abs().max().item()
    scale = fmap.abs().max().item()
    print("  a convolution slides ONE kernel over every position, so shifting the input just")
    print("  shifts the feature map:  featmap(shift(x)) = shift(featmap(x)).  verify (interior):")
    print(f"        max|featmap(shift(x)) - shift(featmap(x))| = {diff:.2e}   (feature scale ~{scale:.1f}"
          " -> equal to numerical noise)")
    print("      -> the conv gets the shifted digit's response FOR FREE from the original's. The")
    print("         same stroke detector fires wherever the stroke goes.\n")

    # ---- (B) the parameter count ------------------------------------------------------------
    H = 768                                                  # our denoiser's hidden width
    dense_params = 784 * H + H
    conv_out = 16
    conv_params = conv_out * 1 * 3 * 3 + conv_out            # one 3x3 kernel per out channel
    print("  (B) parameter count of the FIRST layer:")
    print(f"        dense 784 -> {H}         : {dense_params:>8,} weights (every pixel<->hidden pair)")
    print(f"        conv 1 -> {conv_out} ch, 3x3    : {conv_params:>8,} weights (one small kernel, reused at all 784 spots)")
    print(f"      -> the conv does the first layer with ~{dense_params // conv_params:,}x fewer weights AND the")
    print("         right bias built in. This is why every image model is built from convs.\n")

    # ---- a picture: digit, its shift, and the two feature maps lining up ---------------------
    # 2x3 grid: bottom-middle (shift, then conv) and bottom-right (conv, then shift) are pixel-
    # identical on the interior -> that IS featmap(shift x) == shift(featmap x). edge maps use a
    # diverging colormap centered at 0 (the sum-to-zero kernel is signed).
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
    out = os.path.join(_FIGS, "01_equivariance.png")
    os.makedirs(_FIGS, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")
    print("  Open it: bottom-middle (shift, then conv) and bottom-right (conv, then shift) are the")
    print("  same image. That equality is the property a flatten throws away and a conv keeps.\n")
    print("  Next (Layer 2): the convolution op itself — what that sliding kernel computes, hand-set")
    print("  edge kernels you can SEE in the feature maps, and the output-size formula.")


# ---------------------------------------------------------------------------
# LAYER 2: the convolution op itself — what the sliding kernel computes.
#
# Layer 1 leaned on featmap(x) = F.conv2d(x, kernel) as a black box. Open it. A 2-D conv (really
# CROSS-CORRELATION in every DL framework — see below) slides a small kernel k of shape
# (out_ch, in_ch, kH, kW) over the image and, at each output position (y,x) and output channel o:
#
#     out[o, y, x] = bias[o] + Σ_c Σ_i Σ_j  k[o, c, i, j] · in[c, y·s + i - p,  x·s + j - p]
#
# i.e. line the kernel up over a kH×kW WINDOW of the input, multiply element-wise, sum. Slide by
# STRIDE s, and PAD the input with p zeros on each side first so windows fit at the border. The
# SAME k is used at every (y,x) — the weight sharing from Layer 1 — and out channel o has its own
# kernel k[o] summed over all in channels c. That triple structure (share over space, own kernel
# per out-channel, sum over in-channels) is the entire op; Layers 3-4 just stack and stride it.
#
# CROSS-CORRELATION vs CONVOLUTION: true convolution flips the kernel (i,j -> -i,-j). Frameworks
# skip the flip (the formula above, no flip) and still call it "conv". For a LEARNED kernel it's
# irrelevant — the net learns whatever orientation it needs. It only matters when, like here, we
# hand-SET a kernel and want it to mean what we drew.
#
# OUTPUT SIZE. A window of width kW starting at the left needs kW columns; padding adds 2p; stride
# s takes every s-th start. So along each axis:
#
#     out = floor( (in + 2p - k) / s ) + 1
#
# The three cases we use downstream: (k3,s1,p1)->same size (28->28); (k3,s2,p1)->half (28->14),
# the downsample of Layer 4; (k5,s1,p0)->shrink by k-1 (28->24), no padding. This bite (a) builds
# the op from scratch with plain loops and matches F.conv2d to numerical noise, (b) fires hand-set
# edge/blur kernels at a real digit and SAVES the feature maps, (c) checks the size formula.
# ---------------------------------------------------------------------------
def _naive_conv2d(x, w, stride=1, pad=0):
    """Cross-correlation from scratch (plain loops), to SHOW what F.conv2d computes.
    x: (1, Cin, H, W), w: (Cout, Cin, kH, kW). Returns (1, Cout, outH, outW). No bias."""
    Cout, _Cin, kH, kW = w.shape
    xp = F.pad(x, (pad, pad, pad, pad))                       # zero-pad H and W
    _, _, H, W = xp.shape
    outH = (H - kH) // stride + 1
    outW = (W - kW) // stride + 1
    out = torch.zeros(1, Cout, outH, outW)
    for o in range(Cout):
        for y in range(outH):
            for xx in range(outW):
                y0, x0 = y * stride, xx * stride
                window = xp[0, :, y0:y0 + kH, x0:x0 + kW]     # (Cin, kH, kW)
                out[0, o, y, xx] = (w[o] * window).sum()      # multiply element-wise, sum
    return out


def exp_2_conv_op(seed=0):
    """Open the conv black box: (a) build cross-correlation from scratch with loops, show ONE
    output cell by hand, and match F.conv2d to ~0; (b) fire hand-set edge/blur/identity kernels
    at a real digit and save the feature maps (SEE what a conv detects); (c) verify the output-
    size formula out = floor((in+2p-k)/s)+1 against F.conv2d for the strides we'll use later."""
    _banner("LAYER 2: the convolution op — sliding-kernel cross-correlation, feature maps, sizes")

    # ---- (a) build it from scratch and match F.conv2d --------------------------------------
    torch.manual_seed(seed)
    x_small = torch.randn(1, 1, 6, 6)
    w_small = torch.randn(1, 1, 3, 3)
    mine = _naive_conv2d(x_small, w_small, stride=1, pad=0)     # (1,1,4,4)
    ref = F.conv2d(x_small, w_small, stride=1, padding=0)
    print("  (a) cross-correlation is: line the kernel over a window, multiply element-wise, sum.")
    print("      ONE output cell (top-left, y=x=0) by hand:")
    window = x_small[0, 0, 0:3, 0:3]
    print(f"        window·kernel summed = {(window * w_small[0, 0]).sum().item():+.4f}")
    print(f"        F.conv2d[0,0,0,0]    = {ref[0, 0, 0, 0].item():+.4f}   (same number)")
    print(f"      whole map: max|from-scratch - F.conv2d| = {(mine - ref).abs().max().item():.2e}"
          "  -> our loops ARE F.conv2d.\n")

    # ---- (b) hand-set kernels on a real digit -> feature maps -------------------------------
    from denoiser_and_loss import MNISTData
    test = MNISTData(train=False)
    idx = (test.y == 3).nonzero()[0].item()
    x = test.x[idx:idx + 1]                                    # (1,1,28,28) in [-1,1]

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
        ],  # Sobel-x: |left-right| change
        "horizontal edge": [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ],  # Sobel-y: up-down change
        "blur (box)": [[1 / 9] * 3] * 3,  # local average: smooths
    }
    print("  (b) fire hand-set 3x3 kernels at a real '3' (padding=1 keeps 28x28):")
    for name, k in kernels.items():
        w = torch.tensor(k, dtype=torch.float32).view(1, 1, 3, 3)
        fm = F.conv2d(x, w, padding=1)
        print(f"        {name:<16} -> feature map range [{fm.min():+.2f}, {fm.max():+.2f}]"
              f"  (edge kernels: ~0 on flat regions, spike on strokes)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 5, figsize=(5 * 2.0, 2.3))
    axes[0].imshow(_to_img(x), cmap="gray"); axes[0].set_title("input '3'", fontsize=9)
    for ax, (name, k) in zip(axes[1:], kernels.items()):
        w = torch.tensor(k, dtype=torch.float32).view(1, 1, 3, 3)
        fm = F.conv2d(x, w, padding=1).squeeze()
        if name in ("identity", "blur (box)"):
            ax.imshow(fm.cpu().numpy(), cmap="gray")           # near-nonnegative: just a filtered image
        else:
            M = fm.abs().max().item()                          # signed edge map: center 0 on neutral
            ax.imshow(fm.cpu().numpy(), cmap="coolwarm", vmin=-M, vmax=M)
        ax.set_title(name, fontsize=9)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("one input, four hand-set kernels -> four feature maps (edges, blur)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(_FIGS, "02_feature_maps.png")
    os.makedirs(_FIGS, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"      saved -> {out}")
    print("      Open it: 'identity' returns the 3; the edge kernels trace its strokes (vertical")
    print("      vs horizontal); blur softens it. A CNN LEARNS kernels like these instead of us")
    print("      hand-drawing them — that's the only difference from here on.\n")

    # ---- (c) the output-size formula -------------------------------------------------------
    print("  (c) output size along each axis:  out = floor((in + 2p - k)/s) + 1")
    print(f"      {'in':>3} {'k':>2} {'s':>2} {'p':>2} | {'formula':>7} | {'F.conv2d':>8}")
    print(f"      {'-'*3} {'-'*2} {'-'*2} {'-'*2}-+-{'-'*7}-+-{'-'*8}")
    for (k, s, p) in [(3, 1, 1), (3, 2, 1), (5, 1, 0), (3, 2, 0)]:
        w = torch.zeros(1, 1, k, k)
        got = F.conv2d(x, w, stride=s, padding=p).shape[-1]
        formula = (28 + 2 * p - k) // s + 1
        tag = "same" if formula == 28 else ("half" if formula == 14 else "")
        print(f"      {28:>3} {k:>2} {s:>2} {p:>2} | {formula:>7} | {got:>8}   {tag}")
    print("      -> (k3,s1,p1) keeps size (the workhorse conv); (k3,s2,p1) halves it (Layer-4")
    print("         downsample). Formula matches F.conv2d exactly.\n")
    print("  Next (Layer 3): stack these + a ReLU. Depth grows the receptive field, and WITHOUT a")
    print("  nonlinearity a stack of convs collapses back to a single conv — we'll verify that.")


# ---------------------------------------------------------------------------
# LAYER 3: depth — stacking convs, channels, and why the ReLU is load-bearing.
#
# One conv gives ONE feature map from one image. Real nets do two things to that:
#
# CHANNELS. A conv layer maps Cin feature maps -> Cout feature maps; its weight is
# (Cout, Cin, kH, kW). Output channel o has its OWN kernel over all Cin inputs, summed:
# out[o] = Σ_c corr(in[c], k[o,c]). So a layer learns Cout different detectors, each looking
# at every input channel. Layer-1's "1 kernel" was the Cin=Cout=1 special case; a real first
# layer is 1 -> C (C edge/blur/blob detectors at once), then C -> C', etc.
#
# RECEPTIVE FIELD. One k3 conv: each output sees a 3x3 input patch. Stack a second k3 on top and
# each of ITS outputs sees a 3x3 patch OF THE FIRST MAP, which itself spans 3x3 of the input —
# so it sees 5x5 of the ORIGINAL. Depth grows the window without bigger kernels:
#
#     RF_1 = k ;   RF_L = RF_{L-1} + (k - 1)·∏_{i<L} s_i        (stride-1: +(k-1) per layer)
#
# Two k3 layers -> RF 5; three -> 7; add a stride-2 (Layer 4) and it jumps. We MEASURE it below
# by perturbing one input pixel and counting how many outputs move: 3x3 after one conv, 5x5 after
# two. This is how a small-kernel net eventually "sees" a whole 28x28 digit.
#
# WHY A NONLINEARITY. A conv is LINEAR. Compose two linear convs and you get... a single linear
# conv (a bigger kernel) — depth bought NOTHING. We prove it: the two-layer no-activation stack's
# IMPULSE RESPONSE is a 5x5 kernel keq, and conv(x, keq) reproduces the whole stack to ~0. Then we
# drop a ReLU between the two convs: superposition breaks (f(x1+x2) != f(x1)+f(x2)), so it can no
# longer equal ANY single conv. ReLU (max(0,·), the modern default over sigmoid/tanh) is exactly
# what lets stacked layers compose edges->strokes->parts instead of collapsing back to one edge.
# ---------------------------------------------------------------------------
def exp_3_stack_and_relu(seed=0):
    """Three facts about depth: (a) a conv layer maps Cin->Cout, weight (Cout,Cin,kH,kW), each
    out-channel its own kernel summed over in-channels; (b) receptive field grows with depth —
    perturb one input pixel, count moved outputs: 3x3 after one conv, 5x5 after two; (c) the ReLU
    is load-bearing — a linear 2-conv stack collapses to a single 5x5 conv (verify to ~0), and a
    ReLU between them breaks that (superposition fails), so depth+nonlinearity buys real capacity."""
    _banner("LAYER 3: depth — channels, receptive field, and why ReLU makes stacking matter")

    torch.manual_seed(seed)

    # ---- (a) channels: a conv layer maps Cin -> Cout -------------------------------------
    conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
    print("  (a) a conv LAYER maps Cin feature maps -> Cout feature maps.")
    print(f"      Conv2d(3 -> 8, k3): weight shape {tuple(conv.weight.shape)} = (Cout, Cin, kH, kW),"
          f" bias {tuple(conv.bias.shape)}")
    print("      -> 8 output channels, each with its OWN 3x3 kernel over ALL 3 input channels,")
    print(f"         summed. params = 8·3·3·3 + 8 = {8 * 3 * 3 * 3 + 8}. Layer-1's single kernel was")
    print("         just the Cin=Cout=1 case.\n")

    # ---- (b) receptive field grows with depth -------------------------------------------
    # perturb ONE center pixel and count how many OUTPUT pixels change, after 1 vs 2 convs.
    # use ones-kernels so contributions can't cancel — we're counting reach, not values.
    S = 15
    base = torch.zeros(1, 1, S, S)
    pert = base.clone(); pert[0, 0, S // 2, S // 2] = 1.0      # a single lit pixel at the center
    ones = torch.ones(1, 1, 3, 3)

    def n_layers(x, n):
        for _ in range(n):
            x = F.conv2d(x, ones, padding=1)
        return x

    print("  (b) perturb ONE center input pixel; count the OUTPUT pixels that move (the receptive")
    print("      field of an output cell), after 1/2/3 stacked k3 convs:")
    for L in (1, 2, 3):
        moved = (n_layers(pert, L) - n_layers(base, L)).abs() > 0
        ys, xs = moved[0, 0].nonzero(as_tuple=True)
        h = ys.max().item() - ys.min().item() + 1
        w = xs.max().item() - xs.min().item() + 1
        rf = 1 + L * (3 - 1)                                   # RF_L = 1 + L·(k-1) for k3, stride1
        print(f"        {L} conv{'s' if L > 1 else ' '}: moved block {h}x{w}     (formula RF = 1 + {L}·2 = {rf})")
    print("      -> depth widens the window with small kernels; ~3 k3 convs + a stride-2 already")
    print("         let an output cell 'see' most of a 28x28 digit.\n")

    # ---- (c) why the ReLU: a LINEAR conv stack collapses to one conv ---------------------
    C = 8
    w1 = torch.randn(C, 1, 3, 3) * 0.3                         # 1 -> C
    w2 = torch.randn(1, C, 3, 3) * 0.3                         # C -> 1

    def linear_stack(x):                                       # two convs, NO activation
        return F.conv2d(F.conv2d(x, w1, padding=1), w2, padding=1)

    def relu_stack(x):                                         # same, ReLU between them
        return F.conv2d(F.relu(F.conv2d(x, w1, padding=1)), w2, padding=1)

    # the equivalent single kernel = the stack's IMPULSE RESPONSE (delta in -> response out),
    # FLIPPED: F.conv2d is cross-correlation, so its impulse response is the kernel flipped; flip
    # back to recover a kernel that reproduces the stack under F.conv2d.
    delta = torch.zeros(1, 1, 9, 9); delta[0, 0, 4, 4] = 1.0
    imp = linear_stack(delta)[:, :, 2:7, 2:7]                  # 5x5 impulse response around center
    keq = torch.flip(imp, dims=(2, 3))                         # -> the equivalent 5x5 kernel

    x = torch.randn(1, 1, 20, 20)
    single = F.conv2d(x, keq, padding=2)                       # ONE 5x5 conv
    interior = (slice(None), slice(None), slice(2, 18), slice(2, 18))
    collapse = (linear_stack(x)[interior] - single[interior]).abs().max().item()
    print("  (c) a convolution is LINEAR, so composing two of them (no activation) is still ONE conv.")
    print(f"      the 2-layer linear stack's impulse response is a 5x5 kernel keq; conv(x, keq)")
    print(f"      reproduces the whole stack:  max|linear_stack(x) - conv(x, keq)| = {collapse:.2e}")
    print("      -> depth WITHOUT a nonlinearity buys nothing: two layers = one bigger kernel.\n")

    # superposition test: linear map satisfies f(x1+x2)=f(x1)+f(x2); ReLU breaks it.
    x1, x2 = torch.randn(1, 1, 20, 20), torch.randn(1, 1, 20, 20)
    lin_resid = (linear_stack(x1 + x2) - linear_stack(x1) - linear_stack(x2)).abs().max().item()
    relu_resid = (relu_stack(x1 + x2) - relu_stack(x1) - relu_stack(x2)).abs().max().item()
    print("      now drop a ReLU between the two convs and test superposition f(x1+x2) =? f(x1)+f(x2):")
    print(f"        linear stack residual = {lin_resid:.2e}   (linear: holds)")
    print(f"        ReLU   stack residual = {relu_resid:.3f}      (broken: NOT any single conv)")
    print("      -> ReLU is what lets stacked convs compose edges->strokes->parts instead of")
    print("         collapsing to one edge detector. Nonlinearity is what makes depth mean something.\n")
    print("  Next (Layer 4): downsampling — a stride-2 conv halves H,W while channels grow, the")
    print("  28->14->7 pyramid that gives late layers a receptive field covering the whole digit.")


def run_experiments():
    # exp_1_why_conv()
    exp_2_conv_op()
    # exp_3_stack_and_relu()
    # exp_4_downsample()
    # exp_5_head_and_loss()
    # exp_6_train()


@contextlib.contextmanager
def _tee(path):
    """Print to BOTH the terminal and `path` (long runs survive scrollback)."""
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)

        def flush(self):
            for st in self.streams:
                st.flush()

    with open(path, "w") as f:
        with contextlib.redirect_stdout(_Tee(sys.stdout, f)):
            yield
    print(f"(output also written to {path})", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", metavar="FILE", help="also write all output to FILE")
    args = parser.parse_args()

    if args.out:
        with _tee(args.out):
            run_experiments()
    else:
        run_experiments()
