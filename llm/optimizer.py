"""
Build an AdamW optimizer with two parameter groups: weight decay on the matmul
weights, none on everything else. The nanoGPT/GPT-3 `configure_optimizers` trick.

Weight decay is L2 regularization — it pulls weights toward 0 each step to curb
overfitting. That helps the big 2-D weight matrices (Linear / embedding weights),
which hold almost all the parameters and do the actual feature mixing.

But you do NOT want to decay the 1-D parameters:
  - LayerNorm gain (init 1.0) and bias (init 0.0): these exist to rescale/shift a
    normalized signal. Pulling the gain toward 0 just fights the normalization.
  - biases: a handful of numbers that shift activations; decaying them removes the
    offset they're there to provide, for no regularization benefit.
These are few in number and don't drive overfitting, so decaying them only hurts.

The clean heuristic (nanoGPT's): decay every tensor with dim >= 2 (matrices/
embeddings), skip every tensor with dim < 2 (biases, LayerNorm params). One line,
no name matching.

Fill in the TODOs, then run `python optimizer.py` from the llm/ dir.
"""

from __future__ import annotations

import torch


def configure_optimizers(model, weight_decay: float, learning_rate: float,
                         betas: tuple[float, float] = (0.9, 0.95)) -> torch.optim.AdamW:
    """AdamW with weight decay applied only to >=2-D params."""
    # only parameters that actually train
    params = [p for p in model.parameters() if p.requires_grad]

    # split purely by dimensionality: >=2-D weights/embeddings get decayed, the
    # 1-D biases and LayerNorm gain/bias do not (decaying those just hurts).
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    # lr/betas are top-level kwargs; each group's weight_decay still applies
    return torch.optim.AdamW(groups, lr=learning_rate, betas=betas)


# ---------------------------------------------------------------------------
# Self-check: builds a GPT and inspects the resulting param groups. Run from llm/:
#   python optimizer.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from config import GPTConfig
    from model import GPT

    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=65, block_size=8, n_embd=32, n_head=4, n_layer=2)
    model = GPT(cfg)

    wd, lr, betas = 0.1, 1e-3, (0.9, 0.95)
    opt = configure_optimizers(model, weight_decay=wd, learning_rate=lr, betas=betas)

    # 1. exactly two groups, with the right weight_decay each
    assert len(opt.param_groups) == 2, f"expected 2 groups, got {len(opt.param_groups)}"
    decay_g, nodecay_g = opt.param_groups
    assert decay_g["weight_decay"] == wd, "decay group must carry weight_decay"
    assert nodecay_g["weight_decay"] == 0.0, "no-decay group must have weight_decay 0"

    # 2. lr + betas threaded through to both groups
    for g in opt.param_groups:
        assert g["lr"] == lr and g["betas"] == betas, "lr/betas not applied"

    # 3. membership is purely by dimensionality
    assert all(p.dim() >= 2 for p in decay_g["params"]), "a <2-D param leaked into decay"
    assert all(p.dim() < 2 for p in nodecay_g["params"]), "a >=2-D param leaked into no-decay"

    # 4. EVERY trainable param appears exactly once (none dropped, none duplicated) —
    #    a silent partition bug here means some weights never get an update.
    all_ids = {id(p) for p in model.parameters() if p.requires_grad}
    group_ids = [id(p) for g in opt.param_groups for p in g["params"]]
    assert len(group_ids) == len(all_ids), "params missing or duplicated across groups"
    assert set(group_ids) == all_ids, "group params don't cover the model"

    # 5. spot-check the intent on named params
    decay_set = {id(p) for p in decay_g["params"]}
    nodecay_set = {id(p) for p in nodecay_g["params"]}
    assert id(model.ln_f.weight) in nodecay_set, "LayerNorm gain must NOT be decayed"
    assert id(model.ln_f.bias) in nodecay_set, "LayerNorm bias must NOT be decayed"
    assert id(model.lm_head.weight) in decay_set, "Linear weight SHOULD be decayed"
    assert id(model.lm_head.bias) in nodecay_set, "Linear bias must NOT be decayed"
    assert id(model.token_embedding.weight) in decay_set, "embedding SHOULD be decayed"

    n_decay = sum(p.numel() for p in decay_g["params"])
    n_nodecay = sum(p.numel() for p in nodecay_g["params"])
    print(f"decay: {len(decay_g['params'])} tensors / {n_decay} params | "
          f"no-decay: {len(nodecay_g['params'])} tensors / {n_nodecay} params")
    print("param groups configured correctly ✅")
