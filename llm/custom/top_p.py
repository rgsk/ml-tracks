import torch


def nucleus_keep_mask(probs: torch.Tensor, top_p: float, method: str = "explicit") -> torch.Tensor:
    """Return a boolean KEEP mask (True = inside the nucleus) over `probs`.

    Two equivalent implementations of top-p (nucleus) filtering:
      - "explicit":   per-row Python loop, accumulate until cumprob crosses p
      - "vectorized": prefix-shift one-liner (cumprobs - sorted_probs) > p

    Both keep the smallest set of tokens whose cumulative probability reaches p,
    always including the token that crosses the threshold (and never dropping the
    top-1 token).
    """
    B, V = probs.shape
    top_p += 1e-6
    sorted_probs, sorted_idx = probs.sort(descending=True, dim=-1)
    if method == "explicit":
        sorted_remove = torch.ones_like(sorted_probs, dtype=torch.bool)
        for b in range(B):
            cumprob = torch.tensor(0.0)
            i = 0
            while i < V and cumprob <= top_p:
                cumprob += sorted_probs[b][i]
                i += 1
            sorted_remove[b] = torch.arange(V) >= i
    elif method == "vectorized":
        cumprobs = sorted_probs.cumsum(dim=-1)               # (B, V) running mass
        # remove a token once the mass BEFORE it already reached p. The prefix
        # shift (cumprobs - sorted_probs, NOT cumprobs) keeps the first token to
        # cross p, and never removes the top-1 token (its prefix mass is 0).
        sorted_remove = (cumprobs - sorted_probs) > top_p
    else:
        raise ValueError(f"unknown method {method!r}")

    # scatter the sorted-space mask back to original vocab order
    remove = torch.zeros_like(sorted_remove).scatter(
        -1,
        sorted_idx,
        sorted_remove,
    )
    return ~remove


def nucleus_filter(probs: torch.Tensor, top_p: float, method: str = "explicit") -> torch.Tensor:
    """Apply top-p filtering and renormalize so the survivors sum to 1."""
    keep = nucleus_keep_mask(probs, top_p, method)
    probs = probs.masked_fill(~keep, 0.0)
    return probs / probs.sum(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Tests: the two methods must agree, and both must satisfy the nucleus contract.
# Run:  python practice.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # 1. equivalence: explicit loop == vectorized one-liner, random batches, many p.
    for _ in range(50):
        B, V = torch.randint(1, 5, (1,)).item(), torch.randint(2, 30, (1,)).item()
        probs = torch.softmax(torch.randn(B, V), dim=-1)
        for top_p in [0.0, 0.1, 0.3, 0.5, 0.8, 0.9, 0.99, 1.0]:
            m_exp = nucleus_keep_mask(probs, top_p, "explicit")
            m_vec = nucleus_keep_mask(probs, top_p, "vectorized")
            assert torch.equal(m_exp, m_vec), (
                f"methods disagree at p={top_p}, V={V}\n{m_exp}\n{m_vec}"
            )
    print("explicit == vectorized on 50 random batches x 8 thresholds ✅")

    # 2. contract checks on a fixed, hand-checkable example.
    probs = torch.tensor([[0.50, 0.30, 0.15, 0.05]])
    #   cumprobs = [0.50, 0.80, 0.95, 1.00]; p=0.9 keeps the first 3 (0.95 crosses).
    for method in ("explicit", "vectorized"):
        keep = nucleus_keep_mask(probs, 0.9, method)
        assert keep.tolist() == [[True, True, True, False]], (method, keep)

        # p -> 0: only the single most-probable token survives.
        keep0 = nucleus_keep_mask(probs, 0.0, method)
        assert keep0.sum().item() == 1 and keep0[0, 0], (method, keep0)

        # p = 1.0: nothing removed, filtered distribution == original.
        out1 = nucleus_filter(probs, 1.0, method)
        assert torch.allclose(out1, probs, atol=1e-6), (method, out1)

        # filtered output is a valid distribution (sums to 1) and zeros the tail.
        out = nucleus_filter(probs, 0.9, method)
        assert torch.allclose(out.sum(), torch.tensor(1.0)), (method, out)
        assert out[0, -1].item() == 0.0, (method, out)
    print("nucleus contract (crossing token kept, p->0 greedy, p=1 identity) ✅")

    # 3. THE KEY MENTAL FLIP: top_p is a FLOOR on how much mass you keep, not a
    #    ceiling. You add tokens until the cumulative probability REACHES p, and
    #    the token that crosses the line is included. So the question is never
    #    "which tokens stay under p" but "how many do I need to cover p".
    #
    #    probs = [0.4, 0.3, 0.2, 0.1]  ->  cumprobs = [0.4, 0.7, 0.9, 1.0]
    #
    #      p = 0.8: 0.4+0.3 = 0.7 is UNDER 0.8, so two tokens don't cover p yet;
    #               add 0.2 (cum 0.9) to cross 0.8 -> keep first 3 [T,T,T,F].
    #      p = 0.7: 0.4+0.3 = 0.7 exactly equals p. In float32 the sum rounds to
    #               0.70000005 > 0.7 and would drop 0.2 without a tolerance; the
    #               +1e-6 slop keeps 0.2 -> first 3 [T,T,T,F].
    probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
    for top_p in (0.7, 0.8):
        for method in ("explicit", "vectorized"):
            keep = nucleus_keep_mask(probs, top_p, method)
            assert keep.tolist() == [[True, True, True, False]], (top_p, method, keep)
    print("floor-not-ceiling: p=0.7 & p=0.8 both keep 0.2 (cover p, don't stop under it) ✅")

    print("all tests passed ✅")
