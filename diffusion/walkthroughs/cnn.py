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

_HERE = os.path.dirname(os.path.abspath(__file__))   # diffusion/walkthroughs/
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)                         # so `import denoiser_and_loss` works


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
    fmap_of_shift = F.conv2d(x01_shift * 2 - 1, kernel, padding=1)  # feature map of the shifted input
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        (_to_img(x), "original 7"),
        (_to_img(x01_shift * 2 - 1), f"shifted ({shift},{shift})"),
        (shift_of_fmap.squeeze().cpu().numpy(), "shift( featmap(orig) )"),
        (fmap_of_shift.squeeze().cpu().numpy(), "featmap( shifted )"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(4 * 2.1, 2.4))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("shift the input -> the conv feature map just shifts (right two panels match)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(_HERE, "outputs", "cnn_equivariance.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")
    print("  Open it: the two right panels — shift-then-featmap vs featmap-of-shift — are the same")
    print("  image. That equality is the property a flatten throws away and a conv keeps.\n")
    print("  Next (Layer 2): the convolution op itself — what that sliding kernel computes, hand-set")
    print("  edge kernels you can SEE in the feature maps, and the output-size formula.")


def run_experiments():
    exp_1_why_conv()
    # exp_2_conv_op()
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
