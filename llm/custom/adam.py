"""
Adam from scratch — drop-in for torch.optim.Adam (with coupled weight decay).

Adam = momentum (1st moment) + per-parameter adaptive step (2nd moment) + a
bias-correction fix for the cold start. Two buffers per parameter, both init 0:

    m = b1*m + (1-b1)*g            # 1st moment: running mean of g  (like SGD momentum)
    v = b2*v + (1-b2)*g*g          # 2nd moment: running mean of g^2 (grad magnitude)

    m_hat = m / (1 - b1^t)         # bias correction (see below)
    v_hat = v / (1 - b2^t)

    w = w - lr * m_hat / (sqrt(v_hat) + eps)

Two new ideas vs SGD:

  * ADAPTIVE STEP (the v buffer). Dividing by sqrt(v) — the recent RMS of the
    gradient — gives every parameter its OWN effective learning rate. A weight
    whose gradients are consistently large gets a smaller step; a weight with tiny
    gradients gets a larger one. So Adam is robust to wildly different gradient
    scales across parameters, which is why it "just works" without much lr tuning.

  * BIAS CORRECTION. m and v start at 0, so for the first few steps they're biased
    toward 0 (an average that includes the fake initial zero). E.g. at t=1,
    m = (1-b1)*g, which is way smaller than g. Dividing by (1 - b1^t) exactly
    undoes that: at t=1 it divides by (1-b1), recovering g; as t grows, b1^t -> 0
    and the correction fades to 1. Without it the early steps are far too small.

Weight decay here is COUPLED (added into g, like SGD) — that's plain Adam. Moving
that one line is what turns this into AdamW (the next, final exercise).

Fill in the TODOs, then run `python -m custom.adam` from the llm/ dir.
"""

from __future__ import annotations

import torch


class Adam:
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        # per-param state: 1st moment, 2nd moment, and a step counter (for bias corr)
        self.m: list[torch.Tensor | None] = [None] * len(self.params)
        self.v: list[torch.Tensor | None] = [None] * len(self.params)
        self.t: list[int] = [0] * len(self.params)

    @torch.no_grad()
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad

            # coupled weight decay (same as SGD): fold into the gradient, out-of-place.
            if self.weight_decay != 0:
                g = g + self.weight_decay * p

            # lazily create the zero-initialized buffers on first use
            if self.m[i] is None:
                self.m[i] = torch.zeros_like(p)
                self.v[i] = torch.zeros_like(p)
            self.t[i] += 1                  # persists across step() calls
            t = self.t[i]

            # exponential moving averages: 1st moment (mean of g), 2nd moment (mean of g^2)
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g

            # bias correction: buffers started at 0, so divide out the (1 - b^t) bias.
            # strong at the cold start, fades to 1 as t grows (b^t -> 0).
            m_hat = self.m[i] / (1 - self.b1 ** t)
            v_hat = self.v[i] / (1 - self.b2 ** t)

            # adaptive step: normalize the momentum by the grad RMS. addcdiv_(a, b,
            # value=v) does p += v*(a/b), so this is p -= lr * m_hat/(sqrt(v_hat)+eps).
            p.addcdiv_(m_hat, v_hat.sqrt() + self.eps, value=-self.lr)

    def zero_grad(self):
        for p in self.params:
            p.grad = None


# ---------------------------------------------------------------------------
# Self-check: runs identical optimizations with ours and torch.optim.Adam and
# asserts the weights stay equal. Run from the llm/ dir:  python -m custom.adam
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def make_params():
        torch.manual_seed(0)
        return [torch.nn.Parameter(torch.randn(8, 4)),
                torch.nn.Parameter(torch.randn(4))]

    def loss_fn(ps):
        W, b = ps
        return 0.5 * W.pow(2).sum() + (W.sum(dim=0) + b).sin().sum()

    # (name, betas, weight_decay)
    configs = [
        ("default",      (0.9, 0.999), 0.0),
        ("weight_decay", (0.9, 0.999), 0.05),
        ("custom_betas", (0.8, 0.9),   0.0),
    ]

    for name, betas, wd in configs:
        mine_p, ref_p = make_params(), make_params()
        mine = Adam(mine_p, lr=1e-2, betas=betas, weight_decay=wd)
        ref = torch.optim.Adam(ref_p, lr=1e-2, betas=betas, weight_decay=wd)

        for _ in range(50):
            mine.zero_grad(); loss_fn(mine_p).backward(); mine.step()
            ref.zero_grad(); loss_fn(ref_p).backward(); ref.step()

        for a, b in zip(mine_p, ref_p):
            assert torch.allclose(a, b, atol=1e-6), \
                f"[{name}] diverged from torch.optim.Adam (max diff {(a-b).abs().max():.2e})"
        print(f"{name:13s} matches torch.optim.Adam ✅")

    print("\nall Adam configs match ✅")
