"""
Learning-rate schedule: linear warmup -> cosine decay.

The shape every GPT uses (nanoGPT, GPT-3):

    lr
  max ┤        .-''''-.
      │      /          `-.
      │    /                `-.
  min ┤  /                      `--------
      └──┬──────────┬──────────────────── step
         0     warmup_steps           max_steps

Three phases of get_lr(step):
  1. WARMUP (step < warmup_steps): ramp linearly from ~0 up to max_lr. Adam's
     moment estimates start at 0 and are garbage for the first few steps; taking
     full-size steps then can blow up a fresh model. Warmup eases in.
  2. COSINE DECAY (warmup_steps <= step <= max_steps): glide from max_lr down to
     min_lr along half a cosine. Smoother than a step/linear drop, and the slow
     finish lets the model settle into a minimum.
  3. AFTER (step > max_steps): hold at min_lr (don't divide by zero / go negative).

This returns a number; the training loop writes it into optimizer.param_groups
each step. (torch has CosineAnnealingLR/LambdaLR, but the manual get_lr is the
common, transparent idiom — and it's not shadowing a single torch primitive.)

Fill in the TODOs, then run `python lr_schedule.py` from the llm/ dir.
"""

from __future__ import annotations

import math


def get_lr(step: int, *, warmup_steps: int, max_steps: int,
           max_lr: float, min_lr: float) -> float:
    """Learning rate at `step`: linear warmup to max_lr, then cosine decay to min_lr."""
    # 1. WARMUP — linear ramp. step+1 so step 0 isn't a dead 0-LR step; at
    #    step=warmup_steps-1 this hits max_lr exactly (continuous with phase 2).
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # 3. AFTER decay window — clamp to the floor (don't let the ratio exceed 1).
    if step > max_steps:
        return min_lr

    # 2. COSINE DECAY. decay_ratio is the fraction through the window (0 -> 1);
    #    coeff = 0.5*(1+cos(pi*ratio)) bends it 1 -> 0 along a half-cosine; then
    #    lerp: coeff=1 gives max_lr (start), coeff=0 gives min_lr (end).
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ---------------------------------------------------------------------------
# Self-check: asserts the schedule's shape and endpoints. Run from llm/:
#   python lr_schedule.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    warmup, total = 100, 1000
    max_lr, min_lr = 3e-4, 3e-5

    def lr(s):
        return get_lr(s, warmup_steps=warmup, max_steps=total, max_lr=max_lr, min_lr=min_lr)

    # 1. warmup endpoints: starts small and positive, peaks at max_lr at warmup-1
    assert 0 < lr(0) < max_lr, f"step 0 LR should be a small positive ramp value, got {lr(0)}"
    assert math.isclose(lr(warmup - 1), max_lr, rel_tol=1e-9), "warmup must end exactly at max_lr"

    # 2. warmup is strictly increasing
    warm = [lr(s) for s in range(warmup)]
    assert all(b > a for a, b in zip(warm, warm[1:])), "warmup must be monotonically increasing"

    # 3. cosine start == max_lr (continuous with warmup), end == min_lr
    assert math.isclose(lr(warmup), max_lr, rel_tol=1e-9), "decay should start at max_lr"
    assert math.isclose(lr(total), min_lr, rel_tol=1e-9), "decay should end at min_lr"

    # 4. decay is strictly decreasing across the window
    decay = [lr(s) for s in range(warmup, total + 1)]
    assert all(b < a for a, b in zip(decay, decay[1:])), "cosine decay must be monotonically decreasing"

    # 5. the cosine midpoint is the average of the two LRs (defining property)
    mid = (warmup + total) // 2
    assert math.isclose(lr(mid), (max_lr + min_lr) / 2, rel_tol=1e-2), "midpoint should be the LR average"

    # 6. past the window, hold at the floor
    assert lr(total + 500) == min_lr, "after max_steps the LR must stay at min_lr"
    assert lr(total) >= min_lr, "LR must never dip below min_lr"

    print(f"lr(0)={lr(0):.2e}  lr(peak)={lr(warmup-1):.2e}  "
          f"lr(mid)={lr(mid):.2e}  lr(end)={lr(total):.2e}")
    print("schedule shape verified ✅")
