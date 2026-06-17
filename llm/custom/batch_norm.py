"""
BatchNorm1d from scratch (drop-in for nn.BatchNorm1d), the counterpart to
custom.layer_norm — implement it side by side to feel the difference.

BatchNorm normalizes each FEATURE across the BATCH (and any extra dims), the
opposite axis from LayerNorm (which normalizes each SAMPLE across its features):

    input x: (B, C)
    LayerNorm  -> stats over C, per row   (independent of other samples)
    BatchNorm  -> stats over B, per column (depends on the whole batch)

Because batch statistics aren't available at inference (you may predict one
sample at a time), BatchNorm keeps RUNNING estimates of mean/var during training
and uses them at eval. Hence it behaves differently in train vs eval mode — the
thing LayerNorm avoids, and a big reason transformers use LayerNorm.

    y = (x - mean) / sqrt(var + eps) * gamma + beta
      train: mean/var are this batch's stats (and update the running estimates)
      eval : mean/var are the stored running estimates

Fill in the TODOs, then run `python -m custom.batch_norm` from the llm/ dir.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BatchNorm1d(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        # Learnable affine params, per feature (same idea as LayerNorm).
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

        # Running estimates used at EVAL time. These are NOT parameters (no
        # gradient) but must persist/move with the module -> register_buffer.
        # Standard init: running_mean=0, running_var=1.
        self.register_buffer("running_mean", torch.zeros(dim))
        self.register_buffer("running_var", torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, dim). Normalize over the BATCH dim (dim=0), per feature.
        if self.training:
            mean = x.mean(dim=0)                      # (dim,)
            var = x.var(dim=0, unbiased=False)        # biased, like nn.BatchNorm
            # update running estimates; bookkeeping, not part of the graph
            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean, var = self.running_mean, self.running_var

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return x_hat * self.weight + self.bias


# ---------------------------------------------------------------------------
# Self-check: compares against nn.BatchNorm1d. Run from the llm/ dir:
#   python -m custom.batch_norm
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    B, C = 16, 32
    x = torch.randn(B, C)

    mine = BatchNorm1d(C)
    ref = nn.BatchNorm1d(C)

    # --- TRAIN mode: output uses this batch's stats; should match nn exactly ---
    mine.train()
    ref.train()
    out_mine = mine(x)
    out_ref = ref(x)
    assert out_mine.shape == x.shape, f"bad shape {out_mine.shape}"
    assert torch.allclose(out_mine, out_ref, atol=1e-5), "train output mismatch"

    # each feature (column) should be ~0 mean / ~1 std across the batch
    assert torch.allclose(out_mine.mean(dim=0), torch.zeros(C), atol=1e-5), "mean not ~0"
    assert torch.allclose(out_mine.std(dim=0, unbiased=False), torch.ones(C), atol=1e-4), "std not ~1"

    # running estimates must have moved away from their init (0 and 1)
    assert not torch.allclose(mine.running_mean, torch.zeros(C)), "running_mean not updating"

    # --- EVAL mode: output must NOT depend on the other samples in the batch ---
    # (the defining contrast with train mode, where a row's output shifts with
    # the rest of the batch). In eval, fixed running stats are used, so a row's
    # output is identical whether it's alone or batched with others.
    mine.eval()
    row = torch.randn(1, C)
    batched = torch.cat([row, torch.randn(7, C)], dim=0)
    assert torch.allclose(mine(row), mine(batched)[:1], atol=1e-6), \
        "eval output must not depend on other samples in the batch"

    print("matches nn.BatchNorm1d ✅")
