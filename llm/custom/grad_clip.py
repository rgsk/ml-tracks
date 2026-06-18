"""
Gradient clipping (by global norm) from scratch — drop-in for
torch.nn.utils.clip_grad_norm_.

WHY clip at all: deep nets (esp. RNNs/Transformers early in training) can hit a
batch that produces a huge gradient — a loss spike, a bad init, an outlier
sequence. One oversized step can blow the weights to NaN and destroy the run.
Clipping caps the *step size* so a single bad batch can't wreck training.

WHY by global norm (not per-parameter, not by value): we treat ALL gradients as
one big vector g and look at its total L2 norm ||g||. If that exceeds max_norm,
we scale the WHOLE vector down by the same factor:

    total_norm = sqrt( sum over every grad element of g_i^2 )
    clip_coef  = max_norm / (total_norm + eps)
    if clip_coef < 1:            # i.e. total_norm > max_norm
        g *= clip_coef           # every grad scaled by the same factor

Scaling everything by one shared factor preserves the *direction* of the update
(the relative sizes between parameters), only shortening its length. Per-value
clipping (clamping each element to [-c, c]) would distort the direction. That
direction-preserving property is the whole point, and the usual interview answer.

Note total_norm is measured BEFORE clipping (that's what we return, and what you
log to watch for spikes).

Fill in the TODOs, then run `python -m custom.grad_clip` from the llm/ dir.
"""

from __future__ import annotations

import torch


def clip_grad_norm(parameters, max_norm: float, eps: float = 1e-6) -> torch.Tensor:
    """Scale `.grad` of every parameter in-place so their global L2 norm <= max_norm.

    Returns the total norm BEFORE clipping (for logging). Mirrors the behavior of
    torch.nn.utils.clip_grad_norm_ with the default norm_type=2.
    """
    # parameters may be a generator (model.parameters()); we iterate it twice
    # (once for the norm, once to scale), so materialize it into a list first.
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.tensor(0.0)

    # global L2 norm = norm of the per-tensor norms, i.e. sqrt(sum_p ||g_p||^2).
    # NOT the sum of the norms (that overcounts via the triangle inequality).
    total_norm = torch.norm(torch.stack([p.grad.norm(2) for p in params]), 2)

    # eps guards against divide-by-zero when every gradient is exactly 0.
    clip_coef = max_norm / (total_norm + eps)

    # only scale when it would SHRINK the grad (total_norm > max_norm); never scale
    # up. One shared scalar keeps the direction, just shortens the step. In-place:
    # the optimizer reads p.grad immediately after.
    if clip_coef < 1:
        for p in params:
            p.grad.mul_(clip_coef)
    return total_norm   # pre-clip value, for logging


# ---------------------------------------------------------------------------
# Self-check: compares against torch.nn.utils.clip_grad_norm_. Run from llm/:
#   python -m custom.grad_clip
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # gradient clipping only touches .grad tensors — no model needed. Build a set
    # of fake "parameters" with assorted shapes and random grads.
    shapes = [(4, 4), (8,), (16, 3), (1,)]

    def make_params(grads):
        ps = []
        for g in grads:
            p = torch.nn.Parameter(torch.zeros_like(g))
            p.grad = g.clone()           # each copy gets its OWN grad (clip is in-place)
            ps.append(p)
        return ps

    base_grads = [torch.randn(s) for s in shapes]

    # --- case A: grads are large, so clipping actually fires ------------------
    max_norm = 1.0
    mine = make_params(base_grads)
    ref = make_params(base_grads)

    tn_mine = clip_grad_norm(mine, max_norm)
    tn_ref = torch.nn.utils.clip_grad_norm_(ref, max_norm)

    # 1. returned total norm (pre-clip) matches torch
    assert torch.allclose(tn_mine, tn_ref, atol=1e-6), \
        f"total_norm mismatch: {tn_mine} vs {tn_ref}"
    assert tn_mine > max_norm, "test setup: grads should exceed max_norm so clipping fires"

    # 2. every clipped grad matches torch's, element for element
    for pm, pr in zip(mine, ref):
        assert torch.allclose(pm.grad, pr.grad, atol=1e-6), "clipped grads differ from torch"

    # 3. after clipping, the global norm is exactly max_norm
    post = torch.norm(torch.stack([p.grad.norm(2) for p in mine]), 2)
    assert torch.allclose(post, torch.tensor(max_norm), atol=1e-5), \
        f"post-clip norm should equal max_norm, got {post}"

    # 4. direction preserved: clipped grad is a positive scalar multiple of original
    flat_before = torch.cat([g.flatten() for g in base_grads])
    flat_after = torch.cat([p.grad.flatten() for p in mine])
    cos = torch.dot(flat_before, flat_after) / (flat_before.norm() * flat_after.norm())
    assert torch.allclose(cos, torch.tensor(1.0), atol=1e-5), "clipping changed the direction!"

    # --- case B: grads already small, clipping must be a NO-OP ----------------
    big = 1e9
    mine2 = make_params(base_grads)
    clip_grad_norm(mine2, big)
    for pm, g in zip(mine2, base_grads):
        assert torch.equal(pm.grad, g), "grad within budget must be left untouched"

    # --- case C: all grads None -> returns 0, no crash -----------------------
    p_none = torch.nn.Parameter(torch.zeros(3))   # .grad is None
    assert clip_grad_norm([p_none], 1.0).item() == 0.0

    print("matches torch.nn.utils.clip_grad_norm_ ✅")
