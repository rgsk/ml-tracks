# ---
# jupyter:
#   jupytext:
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
# # `naive_conv2d` — 2-D convolution from scratch
#
# To prove `F.conv2d` is no magic. This module is **imported** by `02_features` (`from
# custom.naive_conv2d import naive_conv2d`) and also runs **standalone as its own self-test**:
#
# ```bash
# python nb/cnn/custom/naive_conv2d.py     # checks against torch's F.conv2d
# ```
#
# ## What it computes
#
# A conv **layer** maps input `(N, Cin, H, W)` with weight `(Cout, Cin, kH, kW)` to output
# `(N, Cout, Hout, Wout)`. Each output channel `o` slides its own `(Cin, kH, kW)` kernel over the
# (zero-padded) input; every output cell is a windowed multiply-and-sum over all input channels:
#
# $$\text{out}[o, i, j] = \sum_{\text{over the } C_{in} \times k_H \times k_W \text{ window}} \text{kernel}[o]\cdot\text{input\_window}$$
#
# That's the whole operation — **three loops** (output channel, output row, output col), each cell one
# dot product. Two structural facts fall out and are the whole reason convs beat dense layers (`03`):
# **weight sharing** (the same `kernel[o]` at every `(i,j)`) and **locality** (each output depends only
# on a small `kH×kW` window).
#
# This matches torch's `F.conv2d` (single group, dilation 1) to floating-point noise.
#
# > Like every DL framework, this is cross-**correlation** (kernel *not* flipped) — see `03` for the
# > flip. It only matters for hand-set kernels; a learned kernel just learns whichever orientation.

# %%
import torch


def naive_conv2d(x, weight, bias=None, stride=1, padding=0):
    """x: (N, Cin, H, W), weight: (Cout, Cin, kH, kW), bias: (Cout,) or None. Returns (N, Cout, Ho, Wo)."""
    N, Cin, H, W = x.shape
    Cout, Cin_w, kH, kW = weight.shape
    assert Cin == Cin_w, f"in-channels mismatch: x has {Cin}, weight expects {Cin_w}"

    # zero-pad H and W by `padding` on each side (done by hand so nothing is hidden)
    if padding > 0:
        xp = torch.zeros(N, Cin, H + 2 * padding, W + 2 * padding, dtype=x.dtype)
        xp[:, :, padding:padding + H, padding:padding + W] = x
        x = xp
    Hp, Wp = x.shape[2], x.shape[3]

    Hout = (Hp - kH) // stride + 1                     # the output-size formula (see 02 part c)
    Wout = (Wp - kW) // stride + 1
    out = torch.zeros(N, Cout, Hout, Wout, dtype=x.dtype)

    for o in range(Cout):                              # each output channel = its own detector
        for i in range(Hout):                          # slide down
            for j in range(Wout):                      # slide across
                y0, x0 = i * stride, j * stride
                window = x[:, :, y0:y0 + kH, x0:x0 + kW]        # (N, Cin, kH, kW)
                out[:, o, i, j] = (window * weight[o]).sum(dim=(1, 2, 3))   # multiply, sum
        if bias is not None:
            out[:, o] += bias[o]
    return out


# %% [markdown]
# ## Self-check vs `torch.F.conv2d`
#
# Compare against torch across the configs we actually use (stride, padding, channels, batch, bias,
# larger kernels), including the exact convs inside `SmallCNN.features` — the `28 → 14 → 7` pyramid.
# The `if __name__ == "__main__"` guard means this runs when the file is executed (as a script *or* as
# this notebook) but stays quiet when `02` imports the function.

# %%
if __name__ == "__main__":
    import torch.nn.functional as F

    def check(name, x, w, bias=None, stride=1, padding=0):
        mine = naive_conv2d(x, w, bias=bias, stride=stride, padding=padding)
        ref = F.conv2d(x, w, bias=bias, stride=stride, padding=padding)
        assert mine.shape == ref.shape, f"{name}: shape {tuple(mine.shape)} != torch {tuple(ref.shape)}"
        diff = (mine - ref).abs().max().item()
        print(f"  {name:<32} out {str(tuple(mine.shape)):<16} max|mine-ref| = {diff:.2e}")
        # atol=1e-4: residual is pure float32 accumulation-order noise (hundreds of products
        # summed per cell in a different order than cuDNN/MKL), not a correctness gap.
        assert torch.allclose(mine, ref, atol=1e-4), f"{name}: does not match F.conv2d"

    torch.manual_seed(0)
    print("naive_conv2d vs F.conv2d:")
    check("1->1  k3 s1 p0", torch.randn(1, 1, 6, 6), torch.randn(1, 1, 3, 3))
    check("1->1  k3 s1 p1 (same size)", torch.randn(1, 1, 8, 8), torch.randn(1, 1, 3, 3), padding=1)
    check("1->1  k3 s2 p1 (halve)", torch.randn(1, 1, 8, 8), torch.randn(1, 1, 3, 3), stride=2, padding=1)
    check("3->8  k3 s1 p1, N=4", torch.randn(4, 3, 10, 10), torch.randn(8, 3, 3, 3), padding=1)
    check("3->8  k3 s2 p1 + bias", torch.randn(2, 3, 12, 12), torch.randn(8, 3, 3, 3),
          bias=torch.randn(8), stride=2, padding=1)
    check("1->2  k5 s1 p0 (shrink k-1)", torch.randn(1, 1, 12, 12), torch.randn(2, 1, 5, 5))

    # the exact convs inside SmallCNN.features (01): the 28 -> 14 -> 7 pyramid
    check("stem  1->16  k3 s1 p1", torch.randn(1, 1, 28, 28), torch.randn(16, 1, 3, 3), padding=1)
    check("down1 16->32 k3 s2 p1", torch.randn(1, 16, 28, 28), torch.randn(32, 16, 3, 3), stride=2, padding=1)
    check("down2 32->64 k3 s2 p1", torch.randn(1, 32, 14, 14), torch.randn(64, 32, 3, 3), stride=2, padding=1)

    print("all configs match F.conv2d ✅")
