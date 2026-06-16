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
Run `python -m custom.cross_entropy` from the llm/ dir to check against torch.
"""

from __future__ import annotations

import torch


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Log-softmax form: never materializes the probabilities, so it stays
    accurate even when some log-probs are very negative."""
    N, V = logits.shape
    z = logits - logits.max(dim=-1, keepdim=True).values     # stabilize: max -> 0
    logsumexp = torch.log(torch.exp(z).sum(dim=-1, keepdim=True))
    log_probs = z - logsumexp                                # log-softmax, (N, V)
    correct_log_probs = log_probs[torch.arange(N), targets]  # (N,)
    return -correct_log_probs.mean()


def cross_entropy_simple(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Textbook form: normalize to probabilities, then take log. Easier to read;
    slightly less stable because it logs the small probs directly."""
    N, V = logits.shape
    probs = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
    probs /= probs.sum(dim=-1, keepdim=True)
    correct_log_probs = torch.log(probs)[torch.arange(N), targets]
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
