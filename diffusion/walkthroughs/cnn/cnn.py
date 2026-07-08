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
  6. train it — the real loop on MNIST: the GAP net climbs but PLATEAUS ~95%, short of ~99%;
     still a working reader in ~24k params. Save a grid of held-out digits with predicted labels.
  7. why GAP stalled — global pooling is translation-INVARIANT, so it discards WHERE features fire;
     centered digits are told apart by position. A flatten head (same conv trunk) keeps position and
     hits ~99%. flatten-on-features is fine; only flatten-on-raw-pixels (Layer 1) was the mistake.

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


# ---------------------------------------------------------------------------
# LAYER 4: downsampling — a stride-2 conv halves H,W, and we grow channels.
#
# Layer 3 stacked stride-1 convs: the map stayed 28x28 and the receptive field crept up by
# (k-1)=2 per layer. To "see" a whole 28x28 digit that way you'd need ~14 layers. Real nets
# instead build a RESOLUTION PYRAMID: periodically DOWNSAMPLE with a stride-2 conv, which
#
#   (1) HALVES H and W (28 -> 14 -> 7 -> ..., the Layer-2 formula with k3,s2,p1 = ceil(in/2)), and
#   (2) MULTIPLIES receptive-field growth. RF_L = RF_{L-1} + (k-1)·∏_{i<L} s_i : once you've
#       downsampled, every later (k-1) step is worth `stride` INPUT pixels, so RF explodes.
#       Pure stride-2 stack: RF 3 -> 7 -> 15 -> 31 (four layers already cover 28).
#
# And downsampling is what makes GROWING CHANNELS affordable: each stride-2 stage cuts the number
# of positions 4x, so doubling the channel count still leaves the activation footprint C·H·W (and
# the conv FLOPs) SHRINKING. That's the universal CNN shape: resolution down, channels up. We use
# a learned stride-2 conv (the modern default) rather than a fixed max-pool, so the downsampler
# itself is trained. Below we (a) build the 28->14->7 pyramid and check shapes, (b) MEASURE the
# receptive field in input pixels for stride-1 vs stride-2 stacks, (c) tally the footprint per stage.
# ---------------------------------------------------------------------------
def exp_4_downsample(seed=0):
    """Three facts about downsampling: (a) a stride-2 k3 conv halves H,W while channels grow —
    the 28->14->7 pyramid, shapes checked against the Layer-2 formula; (b) receptive field, measured
    in INPUT pixels via autograd, grows far faster with stride (RF 3/7/15 for stride-2 vs 3/5/7 for
    stride-1) because the stride product multiplies every later step; (c) each downsample cuts
    positions 4x, so doubling channels still shrinks the activation footprint C·H·W."""
    _banner("LAYER 4: downsampling — stride-2 halves H,W, channels grow: the 28->14->7 pyramid")

    torch.manual_seed(seed)

    # ---- (a) the resolution pyramid: stride-2 halves H,W while channels grow --------------
    x = torch.randn(1, 1, 28, 28)
    conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)    # 28 -> 14
    conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)   # 14 -> 7
    h1 = conv1(x)
    h2 = conv2(h1)
    print("  (a) a STRIDE-2 conv (k3, p1) halves H,W; we DOUBLE channels at each step.")
    print(f"      {tuple(x.shape)}  --conv(1->16, s2)-->  {tuple(h1.shape)}"
          f"  --conv(16->32, s2)-->  {tuple(h2.shape)}")
    print("      spatial pyramid 28 -> 14 -> 7  (k3,s2,p1 -> out = ceil(in/2), the Layer-2 formula)")
    print(f"      params: conv1 = 16·1·9+16 = {16 * 1 * 9 + 16}, conv2 = 32·16·9+32 = {32 * 16 * 9 + 32}\n")

    # ---- (b) receptive field: stride MULTIPLIES the reach of every later step -------------
    # Measure RF in INPUT pixels exactly: pick one central OUTPUT cell, backprop to the input,
    # and the input pixels with nonzero gradient are precisely the ones it depends on.
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

    print("  (b) receptive field of ONE output cell, in INPUT pixels (autograd: which input pixels")
    print("      the cell actually depends on), for stride-1 vs stride-2 stacks of k3 convs:")
    print("        layers |  stride-1 RF  |  stride-2 RF")
    for L in (1, 2, 3):
        print(f"           {L}    |      {rf(L, 1):>2}       |      {rf(L, 2):>2}")
    print("      RF_L = RF_{L-1} + (k-1)·∏(strides): the stride PRODUCT multiplies every later step.")
    print("      -> ~14 stride-1 convs to reach RF 28, but only ~4 stride-2 (3->7->15->31): stride is")
    print("         how a small-kernel net comes to 'see' the whole digit.\n")

    # ---- (c) why channels grow as resolution shrinks: the footprint stays bounded ---------
    print("  (c) footprint C·H·W per stage. once channels DOUBLE while area QUARTERS, each downsample")
    print("      multiplies the footprint by 2·(1/4) = 1/2 — resolution traded for depth cheaply:")
    prev = None
    for name, t in (("input", x), ("after conv1", h1), ("after conv2", h2)):
        c, hh, ww = t.shape[1], t.shape[2], t.shape[3]
        v = c * hh * ww
        ratio = "(input)" if prev is None else f"×{v / prev:.2f} vs prev"
        print(f"        {name:<12} {c:>2}·{hh}·{ww} = {v:>5} values   {ratio}")
        prev = v
    print("      -> stride-2 conv FLOPs drop 4x per stage too, which is what makes deep, wide late")
    print("         layers affordable. Next (Layer 5): global-average-pool the 7x7 map -> a vector,")
    print("         Linear -> 10 logits, cross-entropy.")


# ---------------------------------------------------------------------------
# LAYER 5: the head + loss — turn the final feature map into class scores, and score them.
#
# Layers 2-4 give us a feature stack: (B,1,28,28) -> (B,64,7,7), a small grid of 64-channel
# feature vectors. To classify we need (B,10) logits and a scalar loss. Two pieces:
#
# THE HEAD. How do we go from a (64,7,7) map to a 10-vector? The MLP-era answer was FLATTEN then
# Linear — but that ties the head to an exact spatial size (3136 inputs) and burns 31k weights.
# The modern answer is GLOBAL AVERAGE POOLING: average each channel's 7x7 map down to ONE number,
# giving a 64-vector ("how strongly is feature c present ANYWHERE?"), then a single Linear(64->10).
# It's spatial-size independent (any HxW pools to 64) and ~50x cheaper. That 64-vector -> 10 logits.
#
# THE LOSS. Cross-entropy: softmax the logits to a distribution p, take -log p[true class]. A useful
# SANITY CHECK falls out of the math: an untrained net has ~equal logits, so p_true ~ 1/C and the
# loss ~ ln C = ln 10 ~ 2.30. If your untrained loss isn't near that, something is miswired.
#
# THE WIRING TEST. Before training on all of MNIST, OVERFIT A SINGLE BATCH: a correctly-wired model
# with enough capacity must drive one batch's loss to ~0 and accuracy to 100%. If it can't, the bug
# is in the model/plumbing, not the data or schedule — the single most useful debugging habit there is.
# ---------------------------------------------------------------------------
class SmallCNN(torch.nn.Module):
    """Layers 2-4 assembled + a classifier head: (B,1,28,28) -> (B,10) logits. A stride-1 stem then
    two stride-2 downsamples (28->14->7, channels 1->16->32->64), then a head.

    head="gap"     : global-average-pool the (64,7,7) map to a 64-vector, Linear(64->10). The Layer-5
                     default; cheap and position-INVARIANT (Layers 5-6).
    head="flatten" : flatten the (64,7,7) map to 3136 and Linear(3136->10), keeping POSITION. Layer 7
                     shows this is what actually reaches ~99% on centered MNIST."""

    def __init__(self, n_classes=10, head="gap"):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1), torch.nn.ReLU(),            # 28x28
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), torch.nn.ReLU(),  # 14x14
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), torch.nn.ReLU(),  # 7x7
        )
        self.head_kind = head
        self.head = torch.nn.Linear(64 if head == "gap" else 64 * 7 * 7, n_classes)

    def forward(self, x):
        f = self.features(x)                                  # (B, 64, 7, 7)
        v = f.mean(dim=(2, 3)) if self.head_kind == "gap" else f.flatten(1)   # (B,64) or (B,3136)
        return self.head(v)                                   # (B, 10)


def exp_5_head_and_loss(seed=0):
    """Assemble the full classifier and check three things: (a) the GAP head turns a (B,64,7,7) map
    into (B,10) logits, spatial-size-independent and ~50x cheaper than flatten+Linear; (b) an
    untrained net's cross-entropy loss sits at ~ln 10 (a calibration check the softmax math predicts);
    (c) OVERFIT one batch -> loss ~0, acc 100%, proving the whole thing is wired and gradients flow."""
    _banner("LAYER 5: the head + loss — global-average-pool -> logits, cross-entropy, wiring tests")

    import math
    torch.manual_seed(seed)
    dev = _device()

    from denoiser_and_loss import MNISTData
    train = MNISTData(train=True)
    B = 32
    xb = train.x[:B].to(dev)                                   # (B,1,28,28) in [-1,1]
    yb = train.y[:B].to(dev)                                   # (B,) labels 0..9
    model = SmallCNN().to(dev)

    # ---- (a) the head: global-average-pool -> Linear --------------------------------------
    with torch.no_grad():
        f = model.features(xb)
        v = f.mean(dim=(2, 3))
        logits = model.head(v)
    print("  (a) GLOBAL-AVERAGE-POOL the final map, then a Linear head -> class logits.")
    print(f"      features {tuple(f.shape)}  --mean over H,W-->  {tuple(v.shape)}"
          f"  --Linear(64->10)-->  {tuple(logits.shape)}")
    print("      GAP = collapse each channel's 7x7 map to ONE number ('how much of feature c anywhere').")
    print(f"        GAP + Linear(64->10)       = 64·10+10   = {64 * 10 + 10} params  (works for ANY H,W)")
    print(f"        flatten + Linear({64*7*7=}->10) = 3136·10+10 = {3136 * 10 + 10} params  (~50x more, size-locked)\n")

    # ---- (b) cross-entropy: untrained loss ~ ln(class count) ------------------------------
    loss0 = F.cross_entropy(logits, yb).item()
    print("  (b) cross-entropy = -log( softmax(logits)[true class] ) = -log(p_true).")
    print(f"      at init the logits are ~equal, so p_true ~ 1/10 and loss ~ ln 10 = {math.log(10):.4f}")
    print(f"        measured untrained loss = {loss0:.4f}   (matches -> softmax/labels/init all wired right)\n")

    # ---- (c) overfit ONE batch -> loss ~0 (the wiring test) ------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    print("  (c) OVERFIT one batch: a correctly-wired net with enough capacity must drive a single")
    print("      batch's loss to ~0 and accuracy to 100% (proves gradients flow and the head connects):")
    for step in range(201):
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 40 == 0 or step == 200:
            acc = (logits.argmax(1) == yb).float().mean().item()
            print(f"        step {step:>3}: loss {loss.item():.4f}   acc {acc * 100:5.1f}%")
    print("      -> loss collapses to ~0, acc 100%: the full CNN (Layers 2-5) is wired right.\n")
    print("  Next (Layer 6): train on ALL of MNIST and watch test accuracy climb, for a fraction of the")
    print("  MLP's params, and save a grid of held-out digits with their predicted labels.")


# ---------------------------------------------------------------------------
# LAYER 6: train it — the real loop on MNIST, and read held-out digits.
#
# Everything is assembled: SmallCNN (Layers 2-5) with the GAP head maps a digit to 10 logits,
# cross-entropy scores them, and the one-batch overfit test proved it's wired. Now the actual job:
# minimize cross-entropy over ALL 60k training images, and MEASURE generalization on the 10k test
# images the net never sees. The loop is the standard four lines — forward, loss, backward, step —
# over shuffled minibatches, repeated for a few epochs. We print test accuracy after each epoch so
# you WATCH it climb from ~chance (10%) to the low-to-mid 90s. It's a working digit reader in ~24k
# params (smaller than the MLP's first dense layer alone) — but it PLATEAUS around ~95%, short of the
# ~99% MNIST easily allows. That shortfall is real and intentional: Layer 7 diagnoses WHY the GAP head
# caps out here and swaps in the head that reaches ~99%. The payoff figure is the digit reader: a grid
# of held-out digits with predictions (green = correct, red = wrong), a few of them wrong at ~95%.
# ---------------------------------------------------------------------------
def exp_6_train(seed=0, epochs=8, batch_size=128, lr=2e-3):
    """Train the GAP SmallCNN on all of MNIST and watch test accuracy climb — from ~chance to the
    low-to-mid 90s, where it PLATEAUS (short of the ~99% MNIST allows; Layer 7 explains why). Save a
    grid of held-out test digits with the CNN's predictions (green=correct, red=wrong)."""
    _banner("LAYER 6: train the GAP CNN on MNIST — accuracy climbs to ~95%, then plateaus")

    torch.manual_seed(seed)
    dev = _device()

    from denoiser_and_loss import MNISTData
    train = MNISTData(train=True)
    test = MNISTData(train=False)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

    model = SmallCNN(head="gap").to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: SmallCNN(GAP head), {n_params:,} params TOTAL — smaller than the MLP's first dense")
    print(f"         layer alone (784·768 = 602,880 from Layer 1). Training on {len(train)} images on {dev}.\n")

    xte, yte = test.x.to(dev), test.y.to(dev)

    @torch.no_grad()
    def test_acc():
        model.eval()
        correct = 0
        for i in range(0, len(xte), 2000):                     # chunk the 10k test set
            correct += (model(xte[i:i + 2000]).argmax(1) == yte[i:i + 2000]).sum().item()
        model.train()
        return correct / len(xte)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"  epochs={epochs}, batch={batch_size}, lr={lr}, {len(train_loader)} steps/epoch."
          "  test accuracy after each epoch:")
    print(f"        before training : test acc {test_acc() * 100:5.2f}%   (~chance, 10 classes)")
    for ep in range(1, epochs + 1):
        run = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
        print(f"        epoch {ep}        : train loss {run / len(train_loader):.4f}   "
              f"test acc {test_acc() * 100:5.2f}%")

    # ---- the payoff: an actual digit reader on held-out images ----------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model.eval()
    torch.manual_seed(seed)
    idxs = torch.randperm(len(test))[:40]                      # 40 held-out digits
    with torch.no_grad():
        preds = model(xte[idxs]).argmax(1).cpu()
    rows, cols = 5, 8
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.25))
    for ax, k in zip(axes.flat, range(len(idxs))):
        i = idxs[k].item()
        pred, true = preds[k].item(), test.y[i].item()
        ok = pred == true
        ax.imshow(_to_img(test.x[i]), cmap="gray")
        ax.set_title(f"{pred}" if ok else f"{pred}≠{true}", color=("green" if ok else "red"), fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("held-out MNIST digits — CNN predictions (green = correct, red = wrong)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(_FIGS, "06_predictions.png")
    os.makedirs(_FIGS, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out} — held-out digits with the GAP CNN's predictions (a few wrong at ~95%).")
    print("  It works, but it stalled short of ~99%. Next (Layer 7): WHY the GAP head caps out on")
    print("  centered MNIST, and the flatten head that fixes it -> ~99%.")


# ---------------------------------------------------------------------------
# LAYER 7: why GAP stalled — global pooling throws position away; flatten keeps it, and wins.
#
# Layer 6's GAP net plateaued around ~95%, not the ~99% MNIST allows. This layer diagnoses it and
# fixes it, with the SAME conv stack — only the head changes.
#
# THE CULPRIT: GAP is largely translation-INVARIANT. Layer 1 showed conv features are translation-
# EQUIVARIANT (shift the input -> the feature MAP shifts). GAP then AVERAGES each channel's map over
# space, and the average of a shifted map changes far less than a position-keeping readout would -> the
# pooled 64-vector moves ~4x less than the flatten vector when the digit shifts (measured below; it's
# not perfectly invariant here because stride-2 aliasing and boundaries leak some position). So GAP
# mostly sees how much of a feature exists, not WHERE — it discards position. That invariance is
# a feature for big natural-image nets (an object can be anywhere), but MNIST digits are CENTERED and
# told apart largely by position — the top loop of a 9 vs the bottom loop of a 6, a 1's central stroke.
# Throwing position away is throwing away the signal. (GAP also bottlenecks 64·7·7=3136 numbers down to
# 64 — a 49x crush — while flatten keeps all 3136.)
#
# THE FIX: a FLATTEN head. Lay the (64,7,7) map into a 3136-vector (keeping which feature fired WHERE)
# and Linear(3136->10). It costs more params in the head (31k vs 650) but reaches ~99%. And this does
# NOT contradict Layer 1's "don't flatten": there we flattened RAW PIXELS (before any conv), destroying
# the spatial structure; here we flatten CONV FEATURES the stack already built with weight-sharing and
# locality. Flatten-on-pixels is the disease; flatten-on-features is a fine readout (it's what LeNet did).
# We (a) train both heads head-to-head, (b) MEASURE the shift-invariance that explains the gap, (c) save
# the ~99% reader.
# ---------------------------------------------------------------------------
def exp_7_gap_vs_flatten(seed=0, epochs=5, batch_size=128):
    """Diagnose Layer 6's plateau: (a) same conv stack, GAP vs flatten head, trained head-to-head —
    GAP ~95%, flatten ~99%; (b) MEASURE why: shift the digit a few px and the GAP vector barely moves
    (translation-invariant -> blind to position) while the flatten vector changes a lot; (c) save the
    flatten model's held-out predictions — the ~99% digit reader."""
    _banner("LAYER 7: why GAP stalled — position-blind pooling vs position-keeping flatten (-> ~99%)")

    torch.manual_seed(seed)
    dev = _device()

    from denoiser_and_loss import MNISTData
    train = MNISTData(train=True)
    test = MNISTData(train=False)
    loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    xte, yte = test.x.to(dev), test.y.to(dev)

    def train_head(head, lr):
        torch.manual_seed(seed)                                # same init/shuffle -> fair comparison
        m = SmallCNN(head=head).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        for _ in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(dev), yb.to(dev)
                loss = F.cross_entropy(m(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return m

    @torch.no_grad()
    def acc(m):
        m.eval()
        c = 0
        for i in range(0, len(xte), 2000):
            c += (m(xte[i:i + 2000]).argmax(1) == yte[i:i + 2000]).sum().item()
        return c / len(xte)

    # ---- (a) same conv stack, two heads, head-to-head ------------------------------------
    gap = train_head("gap", lr=2e-3)
    flat = train_head("flatten", lr=1e-3)
    p_gap = sum(p.numel() for p in gap.parameters())
    p_flat = sum(p.numel() for p in flat.parameters())
    print(f"  (a) SAME conv stack ({epochs} epochs each), only the head differs:")
    print(f"        GAP head      ( 64 -> 10):  {p_gap:>6,} params   test acc {acc(gap) * 100:5.2f}%")
    print(f"        flatten head  (3136-> 10):  {p_flat:>6,} params   test acc {acc(flat) * 100:5.2f}%")
    print("      -> flatten wins by several points. The conv features are identical; the head is the")
    print("         whole difference. Why?\n")

    # ---- (b) MEASURE the cause: GAP is translation-invariant, flatten isn't --------------
    # conv features are translation-EQUIVARIANT; GAP averages over space (invariant), flatten keeps it.
    xb = xte[:512]
    xs = torch.roll(xb, shifts=(2, 2), dims=(2, 3))            # shift the digit 2px down-and-right
    with torch.no_grad():
        f0 = flat.features(xb)
        f1 = flat.features(xs)
        gap0, gap1 = f0.mean(dim=(2, 3)), f1.mean(dim=(2, 3))        # GAP representation
        flt0, flt1 = f0.flatten(1), f1.flatten(1)                    # flatten representation
    gap_move = ((gap1 - gap0).norm(dim=1) / (gap0.norm(dim=1) + 1e-9)).mean().item()
    flt_move = ((flt1 - flt0).norm(dim=1) / (flt0.norm(dim=1) + 1e-9)).mean().item()
    print("  (b) shift the digit 2px and watch how much each head's REPRESENTATION moves:")
    print(f"        GAP vector     moves {gap_move * 100:5.1f}%   (averaging over space washes out position)")
    print(f"        flatten vector moves {flt_move * 100:5.1f}%   (it encodes WHERE each feature fired)")
    print(f"      -> flatten is ~{flt_move / gap_move:.0f}x more position-sensitive. GAP largely averages position")
    print("         away — but centered MNIST digits are told apart BY position (a 6's loop is low, a 9's")
    print("         is high), so the flatten head keeps exactly the signal GAP discards.\n")

    # ---- (c) the payoff: the ~99% reader --------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    flat.eval()
    torch.manual_seed(seed)
    idxs = torch.randperm(len(test))[:40]
    with torch.no_grad():
        preds = flat(xte[idxs]).argmax(1).cpu()
    rows, cols = 5, 8
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.1, rows * 1.25))
    for ax, k in zip(axes.flat, range(len(idxs))):
        i = idxs[k].item()
        pred, true = preds[k].item(), test.y[i].item()
        ok = pred == true
        ax.imshow(_to_img(test.x[i]), cmap="gray")
        ax.set_title(f"{pred}" if ok else f"{pred}≠{true}", color=("green" if ok else "red"), fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("held-out MNIST digits — flatten CNN predictions (~99%)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(_FIGS, "07_predictions_flatten.png")
    os.makedirs(_FIGS, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  (c) wrote {out} — the flatten model's held-out predictions (~99%, nearly all green).")
    print("      Lesson: match the head's invariance to the task. GAP's position-invariance is right for")
    print("      big natural images, wrong for small centered digits — flatten keeps the position they")
    print("      need. Same conv trunk either way, and that trunk is what we carry into the U-Net next.")


def run_experiments():
    # exp_1_why_conv()
    # exp_2_conv_op()
    # exp_3_stack_and_relu()
    # exp_4_downsample()
    # exp_5_head_and_loss()
    # exp_6_train()
    exp_7_gap_vs_flatten()


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
