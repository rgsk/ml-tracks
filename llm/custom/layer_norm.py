"""
LayerNorm from scratch (drop-in for nn.LayerNorm).

For each token vector x of length C (the last dim), LayerNorm:
  1. normalizes across the C features to mean 0, variance 1
  2. then applies a learned per-feature scale (gamma) and shift (beta)

    y = (x - mean) / sqrt(var + eps) * gamma + beta

Key points vs BatchNorm:
  - statistics are computed PER TOKEN over its own C features (the last dim),
    NOT across the batch. So there is no running mean/var and no difference
    between train and eval — what makes it the right choice for sequences.
  - the variance is the BIASED/population variance (divide by C, not C-1);
    that's what nn.LayerNorm uses, so we match it (unbiased=False).

Fill in the TODOs, then run `python -m custom.layer_norm` from the llm/ dir.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

        # per-feature scale/shift; nn.Parameter so they train. Init to identity.
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # normalize over the last dim only (per token, across its C features)
        mean = x.mean(dim=-1, keepdim=True)                 # (..., 1)
        var = x.var(dim=-1, keepdim=True, unbiased=False)   # (..., 1), population var
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.weight + self.bias


# ---------------------------------------------------------------------------
# Self-check: compares against nn.LayerNorm. Run from the llm/ dir:
#   python -m custom.layer_norm
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, T, C = 4, 8, 32
    x = torch.randn(B, T, C)

    mine = LayerNorm(C)
    ref = nn.LayerNorm(C)  # same default init: weight=1, bias=0

    out_mine = mine(x)
    out_ref = ref(x)

    assert out_mine.shape == x.shape, f"bad shape {out_mine.shape}"
    assert torch.allclose(out_mine, out_ref, atol=1e-6), "does not match nn.LayerNorm"

    # sanity: each normalized token (before affine, but weight=1/bias=0 here)
    # should have ~0 mean and ~1 std across its C features.
    m = out_mine.mean(dim=-1)
    s = out_mine.std(dim=-1, unbiased=False)
    assert torch.allclose(m, torch.zeros_like(m), atol=1e-5), "mean not ~0"
    assert torch.allclose(s, torch.ones_like(s), atol=1e-4), "std not ~1"

    print("matches nn.LayerNorm ✅")
