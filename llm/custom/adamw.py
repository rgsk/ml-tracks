"""
AdamW from scratch — drop-in for torch.optim.AdamW. This is Adam with ONE change:
weight decay is DECOUPLED from the gradient.

Side by side, that's the entire difference:

  Adam  (coupled):                      AdamW (decoupled):
    g = g + wd*w        # into grad        # g is left as the raw gradient
    m = b1*m + (1-b1)*g                    m = b1*m + (1-b1)*g
    v = b2*v + (1-b2)*g^2                  v = b2*v + (1-b2)*g^2
    ...bias correct...                     ...bias correct...
                                           w = w - lr*wd*w     # decay, straight on w
    w = w - lr*m_hat/(sqrt(v_hat)+eps)     w = w - lr*m_hat/(sqrt(v_hat)+eps)

Why it matters (the whole point of the detour):
  In coupled Adam the wd*w term flows through m and v, so it gets divided by
  sqrt(v_hat) along with the gradient. A weight with a large gradient history
  (big v) therefore gets LESS decay than one with a small history — the decay
  strength becomes an accident of each parameter's gradient scale, which is not
  what "shrink all weights by a fixed fraction" should mean.
  AdamW applies w -= lr*wd*w directly, untouched by v, so every weight decays by
  the same proportion each step. That decoupling is literally what the "W" is.
  (For plain SGD the two are identical — sqrt(v) is what breaks the equivalence.)

Note PyTorch's AdamW scales the decay by lr: w -= lr*wd*w. That decay term is
exactly the one SGD produces when you expand its coupled form:
    SGD:  w = w - lr*(g + wd*w) = w - lr*g - lr*wd*w   # clean -lr*wd*w falls out
But Adam does NOT produce that clean term: folding wd into g sends it through m, v
and the sqrt(v_hat) division, so coupled Adam's effective decay is -lr*wd*w/sqrt(v_hat)
— per-parameter, not uniform. AdamW = the uniform SGD-style decay, kept out of v.

Fill in the TODOs, then run `python -m custom.adamw` from the llm/ dir.
"""

from __future__ import annotations

import torch


class AdamW:
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m: list[torch.Tensor | None] = [None] * len(self.params)
        self.v: list[torch.Tensor | None] = [None] * len(self.params)
        self.t: list[int] = [0] * len(self.params)

    @torch.no_grad()
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad                      # raw gradient — weight decay is NOT folded in

            # decoupled weight decay: shrink the weight directly, separate from the
            # adaptive update. p *= (1 - lr*wd) is exactly p -= lr*wd*p, in-place.
            # This single line is the entire difference from Adam.
            if self.weight_decay != 0:
                p.mul_(1 - self.lr * self.weight_decay)

            if self.m[i] is None:
                self.m[i] = torch.zeros_like(p)
                self.v[i] = torch.zeros_like(p)
            self.t[i] += 1
            t = self.t[i]

            # the Adam core, unchanged — driven by the raw gradient g (no decay in it)
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*g
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*g*g
            m_hat = self.m[i] / (1 - self.b1**t)
            v_hat = self.v[i] / (1 - self.b2**t)
            p.addcdiv_(m_hat, v_hat.sqrt() + self.eps, value=-self.lr)

    def zero_grad(self):
        for p in self.params:
            p.grad = None


# ---------------------------------------------------------------------------
# Self-check: matches torch.optim.AdamW, AND shows coupled != decoupled. Run from
# the llm/ dir:  python -m custom.adamw
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch.optim as optim

    def make_params():
        torch.manual_seed(0)
        return [torch.nn.Parameter(torch.randn(8, 4)),
                torch.nn.Parameter(torch.randn(4))]

    def loss_fn(ps):
        W, b = ps
        return 0.5 * W.pow(2).sum() + (W.sum(dim=0) + b).sin().sum()

    # 1. match torch.optim.AdamW across a few configs
    configs = [
        ("default",      (0.9, 0.999), 0.01),
        ("heavy_decay",  (0.9, 0.999), 0.2),
        ("custom_betas", (0.8, 0.9),   0.05),
    ]
    for name, betas, wd in configs:
        mine_p, ref_p = make_params(), make_params()
        mine = AdamW(mine_p, lr=1e-2, betas=betas, weight_decay=wd)
        ref = optim.AdamW(ref_p, lr=1e-2, betas=betas, weight_decay=wd)
        for _ in range(50):
            mine.zero_grad(); loss_fn(mine_p).backward(); mine.step()
            ref.zero_grad(); loss_fn(ref_p).backward(); ref.step()
        for a, b in zip(mine_p, ref_p):
            assert torch.allclose(a, b, atol=1e-6), \
                f"[{name}] diverged from torch.optim.AdamW (max diff {(a-b).abs().max():.2e})"
        print(f"{name:13s} matches torch.optim.AdamW ✅")

    # 2. the payoff: with wd>0, decoupled (AdamW) and coupled (Adam) must DIFFER.
    #    Same lr/betas/wd, same data — only the decay coupling changes.
    wp, ap = make_params(), make_params()
    adamw = optim.AdamW(wp, lr=1e-2, weight_decay=0.2)
    adam = optim.Adam(ap, lr=1e-2, weight_decay=0.2)   # coupled
    for _ in range(50):
        adamw.zero_grad(); loss_fn(wp).backward(); adamw.step()
        adam.zero_grad(); loss_fn(ap).backward(); adam.step()
    max_gap = max((a - b).abs().max().item() for a, b in zip(wp, ap))
    assert max_gap > 1e-3, "coupled and decoupled should diverge when wd>0"
    print(f"\ncoupled (Adam) vs decoupled (AdamW), wd=0.2 -> weights differ by {max_gap:.4f} ✅")
    print("all AdamW checks pass ✅")
