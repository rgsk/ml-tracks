"""
Gradient accumulation: take ONE optimizer step per N micro-batches, to simulate a
batch N times larger than fits in memory.

The mechanism rests on one fact: .backward() ADDS into .grad (it doesn't replace).
So if you run backward N times without zeroing in between, the gradients sum. Then
you step once. Effective batch size = micro_batch_size * N.

  zero_grad()                      # ONCE, before the micro-batch loop
  for each of N micro-batches:
      loss = forward(...) / N      # <-- the crucial division (see below)
      loss.backward()              # ADDS this micro-batch's grads into .grad
  clip; optimizer.step();

THE GOTCHA — why divide the loss by N:
  cross_entropy uses reduction="mean", so each micro-batch loss is already a MEAN
  over its B samples. Summing N of those means gives a gradient N times too large
  versus a single mean over all N*B samples. Dividing each micro-loss by N rescales
  the sum back to a proper mean:
      sum_j (1/N) * mean_over_B(grad)  =  (1/(N*B)) * sum over all N*B samples
  ...which is EXACTLY the gradient of one big batch of size N*B. (The self-check
  proves this equality, and shows that without the /N the grads are N times too big.)

Composes with mixed precision: each micro-batch forward runs under autocast, and
the scaler scales every backward by the same S, so the accumulated grads are just
the (scaled) sum — unscale once at the end before clipping/stepping.

Fill in the TODOs, then run `python grad_accum.py` from the llm/ dir.
"""

from __future__ import annotations

import torch


def grad_accum_step(model, micro_batches, optimizer, scaler, *, amp_dtype, device_type,
                    grad_clip: float = 0.0):
    """One optimizer step over len(micro_batches) accumulated micro-batches.

    micro_batches: a list of (xb, yb). Returns the mean micro-batch loss (a float).
    """
    n = len(micro_batches)

    # zero ONCE, before the loop — we want grads to accumulate across micro-batches
    optimizer.zero_grad(set_to_none=True)

    running_total = 0.0
    for xb, yb in micro_batches:
        with torch.autocast(device_type=device_type, dtype=amp_dtype):
            _, loss = model(xb, yb)
            loss = loss / n                  # rescale so the sum of means is a mean
        scaler.scale(loss).backward()        # ADDS into .grad (no zero between)
        running_total += loss.item() * n     # undo /n to report the real mean loss

    # after accumulating: unscale -> clip -> step -> update (as in amp_train_step)
    if grad_clip:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    return running_total / n


# ---------------------------------------------------------------------------
# Self-check: proves N accumulated micro-batches == one big batch of size N*B.
# Run from the llm/ dir:  python grad_accum.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math

    from config import GPTConfig
    from model import GPT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    # dropout=0 so the two paths are deterministic and directly comparable
    cfg = GPTConfig(vocab_size=65, block_size=16, n_embd=64, n_head=4, n_layer=2, dropout=0.0)
    model = GPT(cfg).to(device)

    N, B, T = 4, 8, cfg.block_size       # accumulate N micro-batches of size B
    big_x = torch.randint(0, cfg.vocab_size, (N * B, T), device=device)
    big_y = torch.randint(0, cfg.vocab_size, (N * B, T), device=device)
    micro = [(big_x[i*B:(i+1)*B], big_y[i*B:(i+1)*B]) for i in range(N)]

    def grads_of(fn):
        model.zero_grad(set_to_none=True)
        fn()
        return [p.grad.detach().clone() for p in model.parameters()]

    # (a) one big batch of size N*B
    big_grads = grads_of(lambda: model(big_x, big_y)[1].backward())

    # (b) N micro-batches, each loss divided by N, accumulated
    def accumulate():
        for xb, yb in micro:
            (model(xb, yb)[1] / N).backward()
    accum_grads = grads_of(accumulate)

    # (c) the SAME but WITHOUT the /N — should be N times too big
    def accumulate_no_div():
        for xb, yb in micro:
            model(xb, yb)[1].backward()
    nodiv_grads = grads_of(accumulate_no_div)

    for gb, ga, gn in zip(big_grads, accum_grads, nodiv_grads):
        assert torch.allclose(gb, ga, atol=1e-5), \
            f"accumulated grads != big-batch grads (max diff {(gb-ga).abs().max():.2e})"
        assert torch.allclose(gn, N * gb, atol=1e-4), \
            "without /N the grads should be exactly N times too large"
    print(f"{N} micro-batches of {B}  ==  one batch of {N*B}  (grads match to 1e-5) ✅")
    print(f"without the /N division: grads are exactly {N}x too big ✅")

    # (d) the full step function actually trains under bf16
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(device, enabled=False)   # bf16 -> no scaler
    losses = []
    for _ in range(20):
        mb = [(torch.randint(0, cfg.vocab_size, (B, T), device=device),
               torch.randint(0, cfg.vocab_size, (B, T), device=device)) for _ in range(N)]
        losses.append(grad_accum_step(model, mb, opt, scaler, amp_dtype=torch.bfloat16,
                                      device_type=device, grad_clip=1.0))
    assert all(math.isfinite(l) for l in losses), "loss went non-finite"
    print(f"grad_accum_step trains: loss {losses[0]:.3f} -> {losses[-1]:.3f} ✅")
    print("gradient accumulation works ✅")
