"""
SGD from scratch — with momentum and weight decay. Drop-in for torch.optim.SGD
(default dampening=0, nesterov=False).

The update, in order, for each parameter w with gradient g:

  1. weight decay (COUPLED — folded into the gradient):
        g  <-  g + weight_decay * w
     This is the L2-penalty view: it's exactly the gradient of (wd/2)||w||^2.
     For plain SGD, adding it to g and decaying w directly are identical — the
     distinction only starts to matter for Adam vs AdamW (next exercises).

  2. momentum (an exponentially-decaying running sum of past gradients):
        buf  <-  momentum * buf + g      (first step: buf = g)
        g    <-  buf
     Momentum smooths the trajectory: it keeps rolling in a consistent direction
     and damps oscillation across steep valleys, so you can use a larger lr.

  3. the step:
        w  <-  w - lr * g

Why a velocity buffer? A single gradient is noisy; momentum averages them, so the
update reflects the persistent downhill direction rather than each batch's jitter.

Fill in the TODOs, then run `python -m custom.sgd` from the llm/ dir.
"""

from __future__ import annotations

import torch


class SGD:
    def __init__(self, params, lr: float, momentum: float = 0.0,
                 weight_decay: float = 0.0):
        # materialize (params may be a generator) — we index into it each step
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        # one momentum buffer per param, created lazily on the first step
        self.buffers: list[torch.Tensor | None] = [None] * len(self.params)

    @torch.no_grad()
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad

            # coupled weight decay: fold into the gradient. Out-of-place (g = g + ...)
            # so we don't mutate p.grad; uses the current, pre-update weight p.
            if self.weight_decay != 0:
                g = g + self.weight_decay * p

            # momentum: a persistent running sum of gradients, one buffer per param.
            # first step seeds it with g; afterward it decays the old sum and adds g.
            if self.momentum != 0:
                buf = self.buffers[i]
                buf = g.clone() if buf is None else self.momentum * buf + g
                self.buffers[i] = buf
                g = buf

            p.add_(g, alpha=-self.lr)        # p = p - lr * g (in-place on the leaf)

    def zero_grad(self):
        for p in self.params:
            p.grad = None


# ---------------------------------------------------------------------------
# Self-check: runs identical optimizations with ours and torch.optim.SGD and
# asserts the weights stay equal. Run from the llm/ dir:  python -m custom.sgd
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def make_params():
        # reseed so "mine" and "ref" start from the exact same values
        torch.manual_seed(0)
        return [torch.nn.Parameter(torch.randn(8, 4)),
                torch.nn.Parameter(torch.randn(4))]

    def loss_fn(ps):
        # an arbitrary smooth objective whose gradient changes every step (so the
        # momentum buffer actually accumulates non-trivially)
        W, b = ps
        return 0.5 * W.pow(2).sum() + (W.sum(dim=0) + b).sin().sum()

    configs = [
        ("vanilla",      0.0, 0.0),
        ("momentum",     0.9, 0.0),
        ("weight_decay", 0.0, 0.05),
        ("momentum+wd",  0.9, 0.05),
    ]

    for name, mom, wd in configs:
        mine_p, ref_p = make_params(), make_params()
        mine = SGD(mine_p, lr=0.05, momentum=mom, weight_decay=wd)
        ref = torch.optim.SGD(ref_p, lr=0.05, momentum=mom, weight_decay=wd)

        for _ in range(40):
            mine.zero_grad(); loss_fn(mine_p).backward(); mine.step()
            ref.zero_grad(); loss_fn(ref_p).backward(); ref.step()

        for a, b in zip(mine_p, ref_p):
            assert torch.allclose(a, b, atol=1e-6), \
                f"[{name}] diverged from torch.optim.SGD (max diff {(a-b).abs().max():.2e})"
        print(f"{name:13s} matches torch.optim.SGD ✅")

    print("\nall SGD configs match ✅")
