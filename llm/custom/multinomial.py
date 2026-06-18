"""
multinomial sampling (num_samples=1) from scratch — the core of how generate()
turns a probability vector into a token id.

The method is INVERSE-TRANSFORM SAMPLING (a.k.a. inverse-CDF). To draw a category
from probs = [p0, p1, ..., p_{V-1}] (summing to 1):

  1. Build the cumulative distribution:  cdf[i] = p0 + p1 + ... + pi.
     This carves [0, 1) into V buckets; bucket i has width p_i (its probability).
        probs:  0.1   0.2    0.05   0.65
        cdf:    0.1   0.3    0.35   1.0
        line: [--0.1--|---0.3---|0.35|--------1.0--------]
                bucket0  bucket1  b2        bucket3
  2. Draw u ~ Uniform[0, 1).
  3. Return the bucket u landed in = the first index where cdf[i] >= u. Since a
     bucket's width is exactly its probability, u lands in bucket i with prob p_i —
     which is exactly what we wanted.

The slick vectorized step: the first index where cdf >= u equals the COUNT of
buckets with cdf < u. So `(cdf < u).sum(-1)` is the sampled index, no loop needed.
(torch.searchsorted(cdf, u) does the same thing.)

We work on a batched (B, V) probs tensor -> (B, 1) ids, matching generate's use.

Fill in the TODOs, then run `python -m custom.multinomial` from the llm/ dir.
"""

from __future__ import annotations

import torch


def multinomial_sample(probs: torch.Tensor) -> torch.Tensor:
    """Sample one category per row. probs: (B, V), each row sums to 1.

    Returns (B, 1) long tensor of sampled indices — matches
    torch.multinomial(probs, num_samples=1) in shape and distribution.
    """
    B, V = probs.shape

    cdf = probs.cumsum(dim=-1)                   # (B, V), bucket boundaries on [0,1)
    u = torch.rand(B, 1)                         # (B, 1), one uniform draw per row
    # first index with cdf >= u == count of buckets strictly below u (branchless)
    idx = (cdf < u).sum(dim=-1, keepdim=True)    # (B, 1)
    # boundary guard: if u is a hair below 1 but float cumsum made cdf[-1] < u,
    # the count would be V (out of range) — clamp it back to the last index.
    return idx.clamp(max=V - 1)


# ---------------------------------------------------------------------------
# Self-check: compares the empirical distribution against torch.multinomial.
# Run from the llm/ dir:  python -m custom.multinomial
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    probs = torch.tensor([[0.10, 0.20, 0.05, 0.65]])
    V = probs.size(-1)
    N = 100_000

    # draw N samples by repeating the row N times and sampling each
    batch = probs.expand(N, V).contiguous()
    mine = multinomial_sample(batch)                     # (N, 1)

    # 1. shape + dtype + range
    assert mine.shape == (N, 1), f"bad shape {mine.shape}"
    assert mine.dtype == torch.long, f"ids must be long, got {mine.dtype}"
    assert int(mine.min()) >= 0 and int(mine.max()) < V, "index out of range"

    # 2. empirical frequencies ≈ the input probs (law of large numbers)
    freq = torch.bincount(mine.flatten(), minlength=V).float() / N
    assert torch.allclose(freq, probs[0], atol=0.01), \
        f"empirical dist {freq.tolist()} far from {probs[0].tolist()}"

    # 3. and they track torch.multinomial's own empirical frequencies
    ref = torch.multinomial(probs, N, replacement=True).flatten()
    ref_freq = torch.bincount(ref, minlength=V).float() / N
    assert torch.allclose(freq, ref_freq, atol=0.01), "differs from torch.multinomial dist"

    # 4. degenerate distribution: all mass on one index -> always that index
    one_hot = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    picks = multinomial_sample(one_hot.expand(1000, 4).contiguous())
    assert (picks == 2).all(), "a point mass must always sample its own index"

    print(f"freq      = {freq.tolist()}")
    print(f"torch freq= {ref_freq.tolist()}")
    print("matches torch.multinomial ✅")
