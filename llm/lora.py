"""
LoRA (Low-Rank Adaptation) — a reusable module: LoRALinear + apply/merge helpers.

Fine-tuning all weights is expensive. LoRA freezes the pretrained weight W0 and
learns the UPDATE as a low-rank product, so only a tiny adapter trains:

    out = W0 x  +  (alpha/r) * B @ A @ x        A: (r, in)   B: (out, r)   r << in,out
          ^^^^^ frozen         ^^^^^^^^^ the only trainable params

Three things to implement (fill the TODOs), each with an interview-relevant detail:
  1. LoRALinear   — wrap a frozen Linear; B=0 at init (starts as the base) and A
                    random (or gradients can't flow — grad wrt A is proportional to B).
  2. apply_lora   — swap target Linears for LoRALinear, then freeze the base so only
                    A,B train. Collect-then-replace (don't re-wrap a new wrapper's .base).
  3. merge_lora   — fold (alpha/r)B@A back into the base weight for zero-overhead
                    inference; the module becomes a plain Linear again.

This is the real, reusable module (sft.py imports it for `--lora`). Its from-scratch
build-up lives in walkthroughs/lora.py. Run `python lora.py` to self-check.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus a trainable low-rank adapter (A, B)."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r
        din, dout = base.in_features, base.out_features
        self.A = nn.Parameter(torch.empty(r, din))
        self.B = nn.Parameter(torch.zeros(dout, r))       # ZERO -> branch is a no-op at init
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # A random (grad wrt B needs A != 0)

    def forward(self, x):
        # W0 x + (alpha/r) B@A x, computed as two skinny matmuls (never form B@A)
        return self.base(x) + self.scaling * (x @ self.A.T) @ self.B.T


def apply_lora(model: nn.Module, r: int = 8, alpha: int = 16, skip=("lm_head",)) -> int:
    """Replace each target nn.Linear with a LoRALinear wrapping it, then freeze all
    base weights so ONLY the A,B adapters train. Returns how many layers were adapted.

    `skip` = child names to leave alone (lm_head / embeddings by default).

    IMPORTANT: collect the targets from the ORIGINAL tree FIRST, then replace — if
    you replace while walking, you'll revisit the new LoRALinear and wrap its own
    `.base` (an nn.Linear too) all over again.
    """
    targets = []
    for parent in model.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Linear) and name not in skip:
                targets.append((parent, name, child))
    for parent, name, child in targets:
        setattr(parent, name, LoRALinear(child, r=r, alpha=alpha))

    for p in model.parameters():                 # freeze everything ...
        p.requires_grad_(False)
    for m in model.modules():                    # ... then re-enable just A,B
        if isinstance(m, LoRALinear):
            m.A.requires_grad_(True)
            m.B.requires_grad_(True)
    return len(targets)


def merge_lora(model: nn.Module) -> int:
    """Fold every LoRALinear's (alpha/r)*B@A into its base weight and replace the
    wrapper with the plain (now-merged) Linear. Returns how many were merged.

    After this the forward is a single matmul per layer again — identical outputs,
    zero LoRA overhead — but you lose the ability to swap adapters.
    """
    to_merge = [(parent, name, child)
                for parent in model.modules()
                for name, child in parent.named_children()
                if isinstance(child, LoRALinear)]
    for parent, name, child in to_merge:
        with torch.no_grad():
            child.base.weight += child.scaling * (child.B @ child.A)   # W0 += dW
        setattr(parent, name, child.base)                              # drop the wrapper
    return len(to_merge)


# ---------------------------------------------------------------------------
# Self-check: run `python lora.py` from the llm/ dir. Don't edit below this line.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # ---- LoRALinear: init + forward + grad asymmetry -------------------
    base = nn.Linear(16, 24)
    layer = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(3, 16)

    with torch.no_grad():
        assert torch.allclose(layer(x), base(x)), "at init LoRALinear must equal base (B=0)"
    assert layer.A.abs().sum() > 0, "A must be initialized nonzero"
    assert layer.B.abs().sum().item() == 0.0, "B must be initialized to zero"
    assert layer.scaling == 8 / 4, "scaling must be alpha/r"

    layer.zero_grad(set_to_none=True)
    layer(x).sum().backward()
    assert layer.A.grad.norm() == 0, "grad wrt A must be 0 at init (proportional to B=0)"
    assert layer.B.grad.norm() > 0, "grad wrt B must be nonzero (proportional to A)"

    # ---- apply_lora: swap + freeze, skip lm_head -----------------------
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(16, 32)
            self.fc2 = nn.Linear(32, 16)
            self.lm_head = nn.Linear(16, 8)

        def forward(self, x):
            h = torch.relu(self.fc1(x))
            h = torch.relu(self.fc2(h))
            return self.lm_head(h)

    m = Tiny()
    xb = torch.randn(2, 16)
    with torch.no_grad():
        before = m(xb).clone()

    n = apply_lora(m, r=4, alpha=8)
    assert n == 2, f"should adapt fc1 + fc2 (skip lm_head), adapted {n}"
    assert isinstance(m.fc1, LoRALinear) and isinstance(m.fc2, LoRALinear), "fc1/fc2 not adapted"
    assert isinstance(m.lm_head, nn.Linear) and not isinstance(m.lm_head, LoRALinear), \
        "lm_head must be skipped"

    with torch.no_grad():
        assert torch.allclose(m(xb), before), "at init (B=0) apply_lora must not change outputs"

    # only adapter A,B train; the base is frozen
    trainable = {id(p) for p in m.parameters() if p.requires_grad}
    adapter_ids = {id(mod.A) for mod in m.modules() if isinstance(mod, LoRALinear)} | \
                  {id(mod.B) for mod in m.modules() if isinstance(mod, LoRALinear)}
    assert trainable == adapter_ids, "exactly the A,B adapters must be trainable, nothing else"

    m.zero_grad(set_to_none=True)
    m(xb).sum().backward()
    assert m.fc1.base.weight.grad is None, "frozen base must receive no gradient"
    assert m.fc1.A.grad is not None, "adapter A must receive gradient"

    # ---- merge_lora: identical outputs, wrapper gone -------------------
    with torch.no_grad():                       # give the adapters a nonzero effect
        for mod in m.modules():
            if isinstance(mod, LoRALinear):
                mod.B.copy_(0.1 * torch.randn_like(mod.B))
    m.eval()
    with torch.no_grad():
        pre = m(xb).clone()

    k = merge_lora(m)
    assert k == 2, f"should merge 2 adapters, merged {k}"
    assert not any(isinstance(mod, LoRALinear) for mod in m.modules()), \
        "no LoRALinear should remain after merge"
    with torch.no_grad():
        assert torch.allclose(m(xb), pre, atol=1e-5), "merge must preserve outputs"

    print("all checks passed ✅  (LoRALinear init/forward/grad, apply_lora, merge_lora)")
