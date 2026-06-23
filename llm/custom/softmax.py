"""
Softmax from scratch (drop-in replacement for F.softmax).

    softmax(x, dim)[i] = exp(x[i]) / sum_j exp(x[j])     # over the given dim

Two things make this interview-worthy despite the one-line definition:

  1. Numerical stability — exp() overflows for large inputs (exp(1000)=inf).
    softmax is shift-invariant: subtracting any constant c from every element
    leaves the result unchanged, because exp(x-c)/sum exp(x-c) = exp(x)/sum exp(x)
    (the exp(-c) factors cancel top and bottom). So we subtract the per-row MAX,
    making the largest exponent exactly exp(0)=1 — no overflow, ever.

  2. The `dim` argument — softmax normalizes along ONE axis. Using keepdim=True
    on the max and sum lets them broadcast back against x of any rank, so the
    same code works for (N,), (N,V), (B,T,V), etc.

You'll also implement log_softmax (the numerically-honest log(softmax(x))) and
the backward pass (the Jacobian-vector product), since those are the parts
people actually trip on.

Run `python -m custom.softmax` from the llm/ dir to check against torch.
"""

from __future__ import annotations

import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Stable softmax along `dim`. Same result as F.softmax(x, dim=dim)."""
    m = x.max(dim=dim, keepdim=True).values     # shift-invariance: largest exponent -> 1
    e = torch.exp(x - m)
    return e / e.sum(dim=dim, keepdim=True)

def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Stable log(softmax(x)). Don't just do `softmax(x).log()` — that logs the
    tiny probabilities directly and loses precision. Use the logsumexp form:

        log_softmax(x) = (x - m) - log( sum_j exp(x - m) )

    where m is the per-`dim` max (keepdim). Notice the (x - m) term survives
    OUTSIDE the log — that's what keeps very-negative log-probs accurate.

    Why not softmax(x).log()? For a very negative logit, softmax produces a
    tiny probability that underflows toward 0, so .log() gives -inf. In this
    form that same log-prob is computed as z - logsumexp, a subtraction of two
    FINITE numbers, so it stays accurate. That's why cross-entropy is built on
    top of log_softmax, never softmax-then-log."""
    z = x - x.max(dim=dim, keepdim=True).values     # stabilize: max -> 0
    # logsumexp = torch.log(torch.exp(z).sum(dim=dim, keepdim=True))
    # below is shorter form of above
    logsumexp = torch.logsumexp(z, dim, keepdim=True)
    return z - logsumexp

# see llm/experiments/softmax_backward_derivation.ipynb 
def softmax_backward(grad_out: torch.Tensor, y: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Backward pass. Given y = softmax(x) (the FORWARD output) and grad_out =
    dL/dy, return dL/dx.

    The Jacobian of softmax is  J_ij = y_i (delta_ij - y_j), so

        dL/dx_i = sum_j grad_out_j * y_i (delta_ij - y_j)
                = y_i * ( grad_out_i - sum_j grad_out_j * y_j )

    i.e. dx = y * (grad_out - (grad_out * y).sum(dim, keepdim=True)).
    The subtracted term is the same for every i in the row — it's the weighted
    average of grad_out under the softmax distribution."""
    return y * (grad_out - (grad_out * y).sum(dim, keepdim=True))

# ---------------------------------------------------------------------------
# Self-check: compares against torch on random data.
# Run from the llm/ directory:  python -m custom.softmax
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch.nn.functional as F

    torch.manual_seed(0)

    # --- forward: softmax, multiple shapes/dims ---
    print("--- softmax ---")
    for shape, dim in [((11,), -1), ((7, 11), -1), ((7, 11), 0), ((3, 4, 5), 1)]:
        x = torch.randn(*shape)
        mine, ref = softmax(x, dim), F.softmax(x, dim=dim)
        assert torch.allclose(mine, ref, atol=1e-6), f"softmax mismatch at {shape} dim={dim}"
        assert torch.allclose(mine.sum(dim=dim), torch.ones_like(mine.sum(dim=dim))), "rows don't sum to 1"
    # overflow trap: huge logits must not produce inf/nan
    big = torch.tensor([[1000.0, 1001.0, 999.0]])
    assert torch.isfinite(softmax(big)).all(), "softmax overflowed — did you subtract the max?"
    print("matches F.softmax (incl. large-logit stability) ✅")

    # --- log_softmax ---
    print("--- log_softmax ---")
    x = torch.randn(7, 11)
    assert torch.allclose(log_softmax(x), F.log_softmax(x, dim=-1), atol=1e-6), "log_softmax mismatch"
    assert torch.isfinite(log_softmax(big)).all(), "log_softmax overflowed"
    print("matches F.log_softmax ✅")

    # --- backward: check against autograd ---
    print("--- softmax_backward ---")
    x = torch.randn(7, 11, requires_grad=True)
    y = F.softmax(x, dim=-1)
    g = torch.randn(7, 11)            # arbitrary upstream gradient
    y.backward(g)
    dx = softmax_backward(g, y.detach(), dim=-1)
    assert torch.allclose(dx, x.grad, atol=1e-6), "backward mismatch vs autograd"
    print("matches autograd ✅")
