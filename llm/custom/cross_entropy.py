"""
Cross-entropy loss from scratch (drop-in replacement for F.cross_entropy).

Signature matches how model.py calls it:
    cross_entropy(logits, targets)
      logits:  (N, V)  raw scores (NOT softmaxed), N = B*T, V = vocab_size
      targets: (N,)    integer class ids in [0, V)
    returns: scalar tensor = mean over the N examples of  -log p(correct class)

Definition for one example with logits z and correct class c:
    loss = -log( softmax(z)[c] )
         = -( z[c] - log sum_j exp(z[j]) )      # log-softmax form

Two equivalent implementations:
  cross_entropy        - log-softmax form (more numerically robust)
  cross_entropy_simple - normalize-then-log form (easier to read)

Both use the standard max-subtraction trick so exp() never overflows.
`cross_entropy` also supports `ignore_index` (the SFT loss-masking mechanism):
positions whose target == ignore_index contribute zero loss and zero gradient,
and the mean is over the kept positions only.
Run `python -m custom.cross_entropy` from the llm/ dir to check against torch.
"""

from __future__ import annotations

import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                  ignore_index: int = -100) -> torch.Tensor:
    """Log-softmax form: never materializes the probabilities, so it stays
    accurate even when some log-probs are very negative.

    Positions where targets == ignore_index are dropped: they add no loss and
    receive no gradient, and the average is over the KEPT positions only. (This
    is how SFT masks the prompt: label = -100 there.)
    """
    N, V = logits.shape
    z = logits - logits.max(dim=-1, keepdim=True).values     # stabilize: max -> 0
    logsumexp = torch.log(torch.exp(z).sum(dim=-1, keepdim=True))
    log_probs = z - logsumexp                                # log-softmax, (N, V)

    keep = targets != ignore_index                           # (N,) True where counted
    # THE TRAP: -100 is a valid NEGATIVE index in torch (row V-100), so a raw gather
    # wouldn't error — it'd silently pick the wrong class. Sanitize ignored ids first.
    safe_targets = targets.masked_fill(~keep, 0)
    correct_log_probs = log_probs[torch.arange(N), safe_targets]   # (N,)
    # zero the ignored positions and average over the KEPT count, not N (that count
    # is why SFT's masked loss averaged over 26 tokens, not 92). all-ignored -> nan.
    return -(correct_log_probs * keep).sum() / keep.sum()


def cross_entropy_simple(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Textbook form: normalize to probabilities, then take log. Easier to read;
    slightly less stable because it logs the small probs directly."""
    N, V = logits.shape
    probs = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
    probs /= probs.sum(dim=-1, keepdim=True)
    log_probs = torch.log(probs)
    correct_log_probs = log_probs[torch.arange(N), targets]
    return -correct_log_probs.mean()

# ---------------------------------------------------------------------------
# Self-check: compares against torch's F.cross_entropy on random data.
# Run from the llm/ directory:  python -m custom.cross_entropy
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch.nn.functional as F

    def test(fn):
        torch.manual_seed(0)
        N, V = 7, 11
        logits = torch.randn(N, V)
        targets = torch.randint(0, V, (N,))

        mine = fn(logits, targets)
        ref = F.cross_entropy(logits, targets)
        print(f"mine = {mine.item():.6f}   ref = {ref.item():.6f}")
        assert torch.allclose(mine, ref, atol=1e-6), "does not match F.cross_entropy"

        # survives large logits thanks to the max-subtraction trick
        big = torch.tensor([[1000.0, 1001.0, 999.0]])
        assert torch.isfinite(fn(big, torch.tensor([1]))).all(), "overflowed"
        print("matches F.cross_entropy ✅")

    for name, fn in (("log-softmax", cross_entropy), ("simple", cross_entropy_simple)):
        print(f"--- {name} ---")
        test(fn)

    # --- ignore_index (only cross_entropy implements it) ---------------------
    print("--- ignore_index ---")
    torch.manual_seed(1)
    N, V = 8, 6
    logits = torch.randn(N, V, requires_grad=True)
    targets = torch.randint(0, V, (N,))
    targets[[1, 4, 5]] = -100                         # mask three of the eight

    mine = cross_entropy(logits, targets)             # default ignore_index=-100
    ref = F.cross_entropy(logits, targets)            # torch's default is -100 too
    print(f"mine = {mine.item():.6f}   ref = {ref.item():.6f}")
    assert torch.allclose(mine, ref, atol=1e-6), "ignore_index mismatch vs torch"

    # denominator is the KEPT count (5), not N (8): equals a hand mean over kept rows
    keep = targets != -100
    assert keep.sum().item() == 5
    manual = F.cross_entropy(logits.detach()[keep], targets[keep])
    assert torch.allclose(mine.detach(), manual, atol=1e-6), "not a mean over KEPT only"

    # ignored positions must receive exactly zero gradient (the whole point)
    mine.backward()
    assert logits.grad[~keep].abs().sum().item() == 0.0, "masked rows leaked gradient"
    assert logits.grad[keep].abs().sum().item() > 0.0, "kept rows got no gradient"

    # not hardcoded to -100: a custom ignore_index must work too
    t2 = torch.randint(0, V, (N,))
    t2[2] = 0
    assert torch.allclose(cross_entropy(logits.detach(), t2, ignore_index=0),
                          F.cross_entropy(logits.detach(), t2, ignore_index=0), atol=1e-6), \
        "custom ignore_index=0 mismatch"

    print("ignore_index matches F.cross_entropy ✅ (kept-only mean + zero grad on masked)")
