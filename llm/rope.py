"""
RoPE — Rotary Position Embeddings.

Instead of ADDING a learned position vector at the input, RoPE ROTATES the query
and key vectors (inside attention) by an angle proportional to their position. The
payoff: the attention score q_m . k_n then depends only on the RELATIVE distance
(m - n), for free, with zero learned parameters and no fixed length cap.

    q_m . k_n = <R(m*theta) q, R(n*theta) k> = q^T R((n-m)*theta) k    # only n-m survives

Mechanism (per head vector of size d = head_size, d even):
  * split the d coords into d/2 planes, HALF-SPLIT pairing: coord k with coord k+d/2
  * rotate plane k by angle (position m) * theta_k, with theta_k = base^(-2k/d)
    (fast planes = fine local distance; slow planes = smooth long-range position)

These are the pure helpers; CausalSelfAttention (attention.py) calls apply_rope on
q,k when config.use_rope is set. Derived step-by-step in walkthroughs/rope.py.

Run `python rope.py` from the llm/ dir to self-check.
"""

from __future__ import annotations

import torch


def rope_frequencies(head_size: int, base: float = 10000.0) -> torch.Tensor:
    """Per-plane rotation frequencies theta_k = base^(-2k/d), k = 0..d/2-1.

    Returns a (d/2,) tensor. arange(0, d, 2) yields 2k directly, so base^(-2k/d)
    is base ** (-idx / d) with idx in {0, 2, ..., d-2}.
    """
    idx = torch.arange(0, head_size, 2).float()          # 0,2,4,...,d-2  -> length d/2
    return base ** (-idx / head_size)                    # base^(-2k/d)


def build_rope_cache(seq_len: int, head_size: int, base: float = 10000.0,
                     offset: int = 0, device=None, dtype: torch.dtype = torch.float32):
    """Precompute (cos, sin), each (seq_len, head_size), for absolute positions
    offset..offset+seq_len-1.

    offset lets the KV-cache rotate NEW tokens by their true absolute position
    (offset = number of cached tokens), so cached keys keep the rotation they got
    when they were new. The (T, d/2) plane-angles are duplicated across both halves
    so cos/sin broadcast against a full (..., T, d) head tensor.
    """
    theta = rope_frequencies(head_size, base).to(device=device, dtype=dtype)  # (d/2,)
    pos = torch.arange(offset, offset + seq_len, device=device, dtype=dtype)   # (T,)
    freqs = torch.outer(pos, theta)                      # (T, d/2) = m * theta_k
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # (T, d)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)  # (T, d)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[x1, x2] -> [-x2, x1], splitting the last dim in half. Supplies the
    off-diagonal (-sin / +sin) terms of the 2D rotation for every plane at once."""
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate every 2D plane of x by its per-position angle:
        rope(x) = x * cos + rotate_half(x) * sin
    x: (..., T, d);  cos, sin: (T, d) broadcast over the leading (batch/head) dims.
    """
    return x * cos + rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Self-check: run `python rope.py` from the llm/ dir.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math

    torch.manual_seed(1337)

    # --- the vectorized formula equals the literal per-plane rotation ---------
    def rope_explicit(x, pos, base=10000.0):
        """Literal 2x2 rotation of plane (k, k+d/2) by pos*theta_k. x: (T, d)."""
        T, d = x.shape
        theta = rope_frequencies(d, base)
        out = torch.empty_like(x)
        for t in range(T):
            for k in range(d // 2):
                ang = pos[t].item() * theta[k].item()
                c, s = math.cos(ang), math.sin(ang)
                a, b = x[t, k].item(), x[t, k + d // 2].item()
                out[t, k] = a * c - b * s
                out[t, k + d // 2] = a * s + b * c
        return out

    T, d = 6, 8
    x = torch.randn(T, d)
    pos = torch.arange(T).float()
    cos, sin = build_rope_cache(T, d)
    fast = apply_rope(x, cos, sin)
    slow = rope_explicit(x, pos)
    assert torch.allclose(fast, slow, atol=1e-5), "rotate_half != explicit rotation!"
    print("rotate_half formula matches the explicit 2D rotation ✅")

    # --- norm preservation: a rotation never changes a vector's length --------
    assert torch.allclose(fast.norm(dim=-1), x.norm(dim=-1), atol=1e-5), \
        "RoPE changed the vector norm — not a pure rotation!"
    print("norm preserved at every position ✅")

    # --- THE property: q_m . k_n depends only on the distance (n - m) ---------
    q, k = torch.randn(d), torch.randn(d)

    def rope_vec(v, m):
        c, s = build_rope_cache(1, d, offset=m)
        return apply_rope(v.view(1, d), c, s).view(d)

    def score(m, n):
        return (rope_vec(q, m) @ rope_vec(k, n)).item()

    d3 = [score(m, n) for m, n in [(0, 3), (1, 4), (7, 10)]]   # all distance 3
    assert max(d3) - min(d3) < 1e-4, f"distance-3 scores not equal: {d3}"
    assert abs(score(0, 1) - score(0, 3)) > 1e-3, "distances 1 and 3 should differ"
    print(f"relative property: distance-3 scores all ≈ {d3[0]:.4f} (spread {max(d3)-min(d3):.1e}) ✅")

    # --- broadcasts over the (B, nh, T, d) shape attention uses ---------------
    B, nh = 2, 4
    xb = torch.randn(B, nh, T, d)
    cosb, sinb = build_rope_cache(T, d)
    assert apply_rope(xb, cosb, sinb).shape == xb.shape
    print("broadcasts over (B, nh, T, d) ✅")
    print("all checks passed ✅")
