# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# `optimizers` — SGD, Adam, AdamW from scratch

`07` drove the training loop with `torch.optim.SGD` and `torch.optim.AdamW` as black boxes:
it measured *that* AdamW beats SGD and *that* decoupled decay is the "W", but never opened
the update rule itself. This notebook opens all three — and because they're written as
plain, importable classes with the exact `torch.optim` signatures, a walkthrough can
`from custom.optimizers import AdamW` and train with them for real, not just read about them.

An **optimizer** has exactly one job: given the gradients `p.grad` that backprop just filled
in, decide how to change each parameter `p` so the loss goes down. Everything below is a
variation on the oldest answer — take a step downhill, `w <- w - lr * g` — where each
optimizer adds **one idea** to fix a concrete failure of the one before it:

    SGD    step downhill, but smooth the noisy gradient with momentum
    Adam   give every parameter its OWN step size — divide by its gradient's recent size
    AdamW  fix how weight decay interacts with that per-parameter scaling  (the "W")

Reading them back to back is the whole point: each class is a two-line diff on the previous
one, and those two lines *are* the idea it adds. So we build each optimizer, then
immediately **check it against its `torch.optim` twin** — same init, same objective, and the
weights must agree to floating-point noise. That agreement is the proof that the handful of
lines above really is the entire algorithm.

This file is a normal importable module *and* runs standalone as its own test:

```bash
python nb/llm/custom/optimizers.py     # builds each optimizer and checks it vs torch.optim
```
"""

# %%
from __future__ import annotations

import torch

# %% [markdown]
"""
## SGD — step downhill, with momentum

Start from the simplest rule there is, **gradient descent**. The gradient `g` points in the
direction the loss *increases* fastest, so to decrease the loss we step the opposite way:

    w  <-  w - lr * g

This works, but two things hurt it:

- **The gradient is noisy.** `g` is measured on one mini-batch, so it's a jittery estimate
  of the true descent direction; consecutive steps wobble around the real path.
- **Ravines.** When the loss is much steeper in some directions than others, plain descent
  ping-pongs across the steep walls while barely crawling along the gentle floor.

**Momentum** fixes both with a single idea: don't step along the raw gradient, step along a
*running average* of recent gradients. Keep a velocity buffer `buf` and, each step:

    buf  <-  momentum * buf + g          # first step: buf = g
    w    <-  w - lr * buf

Picture a ball rolling downhill. It picks up speed in directions the gradient *consistently*
points, while the back-and-forth noise averages out to near zero. `momentum` (typically
0.9) sets the memory length: at 0.9 each gradient keeps ~90% of its influence into the next
step, so `buf` is roughly the average of the last ~10 gradients. That both damps the jitter
and accelerates progress along the valley floor — which is exactly why you can afford a
larger `lr` with momentum than without.

**Weight decay** is a separate knob, and it's a *regularizer*, not a descent trick: nudge
every weight a little toward 0 each step so the model can't lean on large weights (which
tend to overfit). It's the gradient of an L2 penalty `(wd/2)·||w||^2` — whose derivative is
`wd·w` — so "add an L2 penalty to the loss" is identical to "add `wd·w` to the gradient":

    g  <-  g + weight_decay * w

For SGD, folding decay into `g` like this (**coupled**) and shrinking `w` directly are the
*exact same operation*. Hold that thought: for Adam they stop being the same, and that
difference is the entire reason AdamW exists.

Order within a step: **weight decay → momentum → step**. (That matches `torch.optim.SGD`
with its defaults `dampening=0, nesterov=False`, so our self-check lines up with it to the
last bit.)
"""


# %%
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


# %% [markdown]
"""
### Check it: SGD vs `torch.optim.SGD`

The proof that the lines above are the whole algorithm: run ours and torch's from the
*same* initial weights on the *same* objective, step both many times, and assert the two
sets of weights never drift apart by more than floating-point noise. The objective's
gradient changes every step, so the momentum buffer genuinely accumulates — a static
gradient would let bugs hide.

The shared `make_params` / `loss_fn` / `match` helpers are defined here and **reused by the
Adam and AdamW checks below**. The `if __name__ == "__main__"` guard runs all of this when
the file is executed as a script or read top-to-bottom as this notebook, but stays silent
when a walkthrough just imports the optimizer classes.
"""

# %%
if __name__ == "__main__":
    import torch.optim as optim

    def make_params():
        # reseed so "mine" and "ref" start from the exact same values
        torch.manual_seed(0)
        return [torch.nn.Parameter(torch.randn(8, 4)),
                torch.nn.Parameter(torch.randn(4))]

    def loss_fn(ps):
        # an arbitrary smooth objective whose gradient changes every step, so the
        # momentum / moment buffers accumulate non-trivially
        W, b = ps
        return 0.5 * W.pow(2).sum() + (W.sum(dim=0) + b).sin().sum()

    def match(name, Mine, Ref, kw, steps=50):
        """Optimize the same params with our optimizer and torch's; assert they agree."""
        mine_p, ref_p = make_params(), make_params()
        mine, ref = Mine(mine_p, **kw), Ref(ref_p, **kw)
        for _ in range(steps):
            mine.zero_grad(); loss_fn(mine_p).backward(); mine.step()
            ref.zero_grad(); loss_fn(ref_p).backward(); ref.step()
        diff = max((a - b).abs().max().item() for a, b in zip(mine_p, ref_p))
        assert diff < 1e-6, f"[{name}] diverged from torch (max diff {diff:.2e})"
        print(f"  {name:<26} matches torch  (max diff {diff:.1e}) ✅")

    print("SGD vs torch.optim.SGD:")
    for nm, mom, wd in [("vanilla", 0.0, 0.0), ("momentum", 0.9, 0.0),
                        ("weight_decay", 0.0, 0.05), ("momentum+wd", 0.9, 0.05)]:
        match(nm, SGD, optim.SGD, dict(lr=0.05, momentum=mom, weight_decay=wd))

# %% [markdown]
"""
## Adam — a per-parameter learning rate

SGD still uses one global `lr` for every parameter. But in a transformer the gradients live
on wildly different scales — an embedding row for a rare character barely moves, a LayerNorm
gain gets a strong signal every step. No single `lr` is right for both: large enough to move
the small-gradient params is large enough to make the big-gradient ones diverge. Momentum
smooths *direction*; it does nothing about this *scale* mismatch.

**Adam** gives each parameter its own effective step size. It tracks two exponential moving
averages per parameter — a **1st moment** `m` (the running mean of the gradient, i.e.
momentum again) and a **2nd moment** `v` (the running mean of the gradient *squared*, an
estimate of its recent magnitude):

    m  <-  b1*m + (1-b1)*g            # where it's heading   (b1 ~ 0.9)
    v  <-  b2*v + (1-b2)*g*g          # how big it's been     (b2 ~ 0.999)

then steps with `m` **divided by the square root of `v`**:

    w  <-  w - lr * m / (sqrt(v) + eps)

That division is the whole idea. `sqrt(v)` is roughly the recent RMS size of that
parameter's own gradient, so dividing by it **normalizes every parameter to a comparable
step**: a weight with consistently huge gradients gets scaled down, one with tiny gradients
gets scaled up, and both advance at a similar *relative* pace. That's why Adam "just works"
across very different layers with little `lr` tuning — and concretely why `07` measured
AdamW reaching a loss SGD couldn't touch even at 30× the `lr`. `eps` (~1e-8) is just a floor
so a parameter with near-zero gradient history doesn't divide by ~0.

**Bias correction.** One subtlety makes Adam correct at the cold start. `m` and `v` both
start at 0, so early on they're biased *toward* 0 — they're averaging in that fake initial
zero. At the very first step `m = (1-b1)*g`, about 10× *smaller* than `g` for `b1=0.9`;
using it raw would make the first steps far too timid. The fix divides each moment by
`1 - b^t`, where `t` is the step count:

    m_hat = m / (1 - b1^t)            v_hat = v / (1 - b2^t)

At `t=1` this divides `m` by `(1-b1)`, recovering `g` exactly; as `t` grows `b^t -> 0` and
the correction fades to 1, so it only matters for the first handful of steps. (The counter
`t` is per-parameter state that must persist across `step()` calls — that's why it lives on
the optimizer, not as a local.)

Weight decay here is **coupled** — added into `g` exactly as in SGD — which makes this
*plain* Adam. Lifting that one line back out of `g` is the only thing AdamW changes.
"""


# %%
class Adam:
    def __init__(self, params, lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
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

            # EMAs: 1st moment (mean of g), 2nd moment (mean of g^2)
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


# %% [markdown]
"""
### Check it: Adam vs `torch.optim.Adam`

Same harness (`match`, reused from the SGD cell), now sweeping `betas` and weight decay. The
`custom_betas` case is the interesting one — it confirms the bias-correction `1 - b^t` uses
the right per-moment `b` (a common off-by-one bug is using `b1` for both).
"""

# %%
if __name__ == "__main__":
    print("Adam vs torch.optim.Adam:")
    for nm, betas, wd in [("default", (0.9, 0.999), 0.0),
                          ("weight_decay", (0.9, 0.999), 0.05),
                          ("custom_betas", (0.8, 0.9), 0.0)]:
        match(nm, Adam, optim.Adam, dict(lr=1e-2, betas=betas, weight_decay=wd))

# %% [markdown]
"""
## AdamW — decouple the weight decay

Here is the catch that plain Adam's *coupled* weight decay creates. We fold decay into the
gradient, `g <- g + wd·w`, and then that combined `g` flows through **everything** — the
moments, and crucially the `1/sqrt(v)` division. So the shrink a weight actually receives is
not `wd·w`; it's `wd·w / sqrt(v)`. A weight with a large gradient history (big `v`) gets
*less* decay; one with a small history gets more. The strength of your regularizer has
turned into an accident of each parameter's gradient scale — which is not what "shrink every
weight by the same small fraction" is supposed to mean.

**AdamW** ("Adam with decoupled **W**eight decay") takes the decay *out* of the gradient and
applies it straight to the weight, untouched by the adaptive `1/sqrt(v)`:

    w  <-  w * (1 - lr*wd)          # decay: a uniform fractional shrink, before the step
    ...then the ordinary Adam step, on the RAW gradient g (no wd folded in)...

Now every weight decays by the same proportion `lr*wd` per step, independent of its gradient
history — exactly the uniform L2-style shrink you meant. That single relocation of one line
*is* the "W". Side by side, it's the entire diff between the two classes:

    Adam  (coupled):                      AdamW (decoupled):
      g = g + wd*w        # into grad        w = w * (1 - lr*wd)   # straight on w
      m = b1*m + (1-b1)*g                    m = b1*m + (1-b1)*g   # g is the RAW grad
      v = b2*v + (1-b2)*g^2                  v = b2*v + (1-b2)*g^2
      w -= lr*m_hat/(sqrt(v_hat)+eps)        w -= lr*m_hat/(sqrt(v_hat)+eps)

Why didn't this matter for SGD? Because SGD has no `sqrt(v)` division: there, `g += wd·w`
followed by `w -= lr·g` expands to `w -= lr·wd·w` — already the decoupled form. It's
precisely Adam's per-parameter scaling that breaks the equivalence, so the split is
invisible in SGD and decisive here. (This is also what `07`'s two parameter groups were
for: decoupled decay on the 2-D weight matrices, none on the 1-D biases and LayerNorm gains
— now you can see it's literally `w *= (1 - lr*wd)` applied only to the group that wants it.)
The check below makes the gap concrete: same `wd>0`, coupled vs decoupled, watch the weights
drift apart.
"""


# %%
class AdamW:
    def __init__(self, params, lr: float = 1e-3,
                 betas: tuple[float, float] = (0.9, 0.999),
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
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1 ** t)
            v_hat = self.v[i] / (1 - self.b2 ** t)
            p.addcdiv_(m_hat, v_hat.sqrt() + self.eps, value=-self.lr)

    def zero_grad(self):
        for p in self.params:
            p.grad = None


# %% [markdown]
"""
### Check it: AdamW vs `torch.optim.AdamW`, and the payoff

First match torch across a few configs, including a heavy `wd=0.2`. Then the measurement
that motivated the whole section: run coupled `Adam` and decoupled `AdamW` with the *same*
`lr`, `betas`, and `wd`, and confirm their weights end up meaningfully apart. That gap is
the "W", isolated — same code everywhere except which side of the `1/sqrt(v)` the decay sits
on.
"""

# %%
if __name__ == "__main__":
    print("AdamW vs torch.optim.AdamW:")
    for nm, betas, wd in [("default", (0.9, 0.999), 0.01),
                          ("heavy_decay", (0.9, 0.999), 0.2),
                          ("custom_betas", (0.8, 0.9), 0.05)]:
        match(nm, AdamW, optim.AdamW, dict(lr=1e-2, betas=betas, weight_decay=wd))

    # the payoff: with wd>0, coupled (Adam) and decoupled (AdamW) MUST diverge. Same
    # lr/betas/wd, same data — only the side of the sqrt(v) division the decay sits on.
    wp, ap = make_params(), make_params()
    adamw = AdamW(wp, lr=1e-2, weight_decay=0.2)
    adam = Adam(ap, lr=1e-2, weight_decay=0.2)          # coupled
    for _ in range(50):
        adamw.zero_grad(); loss_fn(wp).backward(); adamw.step()
        adam.zero_grad(); loss_fn(ap).backward(); adam.step()
    gap = max((a - b).abs().max().item() for a, b in zip(wp, ap))
    assert gap > 1e-3, "coupled and decoupled should diverge when wd>0"
    print(f"\ncoupled Adam vs decoupled AdamW (wd=0.2): weights differ by {gap:.4f} "
          f"<- that gap IS the 'W' ✅")
    print("\nall three optimizers match torch.optim ✅")
