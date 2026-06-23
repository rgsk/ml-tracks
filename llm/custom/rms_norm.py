"""
RMSNorm from scratch (drop-in for nn.RMSNorm).

RMSNorm is "LayerNorm with the re-CENTERING removed". For each token vector x of
length C (the last dim) it divides by the root-mean-square of its features and
applies a learned per-feature scale (gamma) — no mean subtraction, no bias:

    rms(x) = sqrt(mean(x^2) + eps)          # over the last dim, C features
    y      = x / rms(x) * gamma

Compare to LayerNorm (see custom/layer_norm.py):
    LayerNorm:  y = (x - mean) / sqrt(var + eps) * gamma + beta
    RMSNorm:    y =  x         / sqrt(mean(x^2) + eps) * gamma

Two things are GONE versus LayerNorm:
  - the mean subtraction (x - mean): RMSNorm does NOT re-center to mean 0. It
    only fixes the SCALE of the vector, not its location.
  - the bias term (beta): just a scale, no shift.

Why anyone does this:
  - cheaper: no mean pass, no subtraction — one reduction (sum of squares)
    instead of two (mean, then variance). One fewer parameter vector too.
  - the Zhang & Sennrich (2019) finding: most of LayerNorm's benefit comes from
    re-scaling, not re-centering — so you can drop the centering and lose almost
    nothing. This is why LLaMA, T5, Gemma, etc. all use RMSNorm.

Note: mean(x^2) is NOT the variance. variance = mean(x^2) - mean(x)^2. They only
coincide when the mean is 0 — which is exactly the centering RMSNorm skips. So
RMSNorm's denominator is the raw RMS, not the std.

Run `python -m custom.rms_norm` from the llm/ dir to verify against nn.RMSNorm.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

        # per-feature scale only — no bias (RMSNorm has no shift term).
        # nn.Parameter so it trains; init to identity (all ones).
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mean of squares over the last dim only (per token, across its C
        # features). This is mean-of-squares, NOT variance — there is no mean
        # subtraction, and the .mean (not .sum) is what makes the norm
        # scale-invariant to the feature dimension C.
        ms = (x**2).mean(-1, keepdim=True)               # (..., 1)

        # divide by the root-mean-square; eps goes INSIDE the sqrt (same place
        # LayerNorm adds it) for numerical safety when the token is near-zero.
        x_hat = x / torch.sqrt(ms + self.eps)            # (..., C)

        # learned per-feature scale only — no bias (RMSNorm has no shift).
        return x_hat * self.weight


# ---------------------------------------------------------------------------
# Self-check: compares against nn.RMSNorm. Run from the llm/ dir:
#   python -m custom.rms_norm
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, C = 4, 8, 32
    x = torch.randn(B, T, C)

    mine = RMSNorm(C, eps=1e-5)
    ref = nn.RMSNorm(C, eps=1e-5)  # same default init: weight=1

    out_mine = mine(x)
    out_ref = ref(x)

    assert out_mine.shape == x.shape, f"bad shape {out_mine.shape}"
    assert torch.allclose(out_mine, out_ref, atol=1e-6), "does not match nn.RMSNorm"

    # sanity 1: with weight=1, the RMS of each normalized token should be ~1
    # (rms, NOT std — RMSNorm fixes the scale, not the spread around a mean).
    rms = out_mine.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4), "rms not ~1"

    # sanity 2: RMSNorm does NOT re-center. Feed an input with a large per-token
    # mean; LayerNorm would zero it out, RMSNorm must keep a nonzero mean.
    x_shift = torch.randn(B, T, C) + 5.0
    m = mine(x_shift).mean(dim=-1)
    assert (m.abs() > 1e-3).all(), "RMSNorm should NOT zero the mean (no centering)"

    # sanity 3: the learned scale really scales. Set weight=2 -> output rms ~2.
    with torch.no_grad():
        mine.weight.fill_(2.0)
    rms2 = mine(x).pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms2, 2 * torch.ones_like(rms2), atol=1e-4), "scale not applied"

    print("matches nn.RMSNorm ✅")
