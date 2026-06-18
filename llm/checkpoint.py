"""
Phase 4 - Exercise: model checkpointing (save / resume).

A checkpoint must hold *everything you need to resume training as if you never
stopped* — not just the weights:

    - model.state_dict()      the learned parameters
    - optimizer.state_dict()  Adam's per-parameter moments (m, v) + step counts.
                              Drop this and resuming restarts Adam cold: the
                              first few steps after resume take a big, badly
                              scaled jump. This is the #1 checkpointing gotcha.
    - step                    so logging / LR schedules continue from the right place
    - config                  the architecture, so you can REBUILD the model
                              before loading weights into it

Two interview-grade subtleties baked into the skeleton below:
  * Save state_dict()s, NOT the Module/optimizer objects. Pickling whole objects
    couples the file to your exact class paths and breaks on refactor.
  * Load with map_location so a GPU checkpoint can be restored on a CPU box
    (and vice-versa) without "RuntimeError: ... CUDA device".

Fill in the TODOs, then run `python checkpoint.py` from the llm/ dir.
"""

from __future__ import annotations

import os
from dataclasses import asdict

import torch

from config import GPTConfig


def save_checkpoint(path, model, optimizer, step, config, **extra):
    """Bundle model + optimizer + step + config into one file at `path`.

    `**extra` lets callers tuck in anything else (best val loss, rng state...).
    """
    # state_dict()s, not the objects: a plain dict of tensors isn't coupled to
    # our class paths, so the file survives a later refactor. config as a dict
    # (not the dataclass) for the same reason.
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": asdict(config),
        **extra,
    }
    # atomic write: torch.save direct to `path` can leave a half-written, corrupt
    # file if the process dies mid-save. Write to a temp file, then os.replace —
    # an atomic rename on the same filesystem — so `path` only ever points at a
    # complete checkpoint (the old one or the new one, never a partial).
    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, map_location=None):
    """Restore `model` (and `optimizer` if given) from the checkpoint at `path`.

    Returns the metadata dict (everything that isn't the two state_dicts), so the
    caller can read back `step`, `config`, and any extras.

    `model` must already be built with the matching architecture — load_state_dict
    copies values into existing parameter tensors, it does not create them. (That's
    why we save `config`: rebuild GPT(GPTConfig(**ckpt["config"])) first, then load.)
    """
    # map_location=None restores tensors onto their original device; pass "cpu"
    # to force everything to CPU regardless (so a GPU checkpoint loads anywhere).
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:        # no optimizer at inference time
        optimizer.load_state_dict(ckpt["optimizer"])
    return {k: v for k, v in ckpt.items() if k not in ("model", "optimizer")}


# ---------------------------------------------------------------------------
# Self-check: run `python checkpoint.py`. Don't edit below this line.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    from model import GPT

    torch.manual_seed(1337)
    cfg = GPTConfig(vocab_size=65, block_size=8, n_embd=32, n_head=4, n_layer=2)

    def fresh():
        m = GPT(cfg)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        return m, opt

    # --- train a tiny model a few steps so the optimizer state is non-trivial ---
    model, optimizer = fresh()
    B, T = 4, cfg.block_size
    for _ in range(5):
        xb = torch.randint(0, cfg.vocab_size, (B, T))
        yb = torch.randint(0, cfg.vocab_size, (B, T))
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Adam should have built up per-parameter state by now (sanity on the setup).
    assert len(optimizer.state) > 0, "optimizer has no state — did the steps run?"

    path = os.path.join(tempfile.mkdtemp(), "ckpt.pt")
    save_checkpoint(path, model, optimizer, step=5, config=cfg, best_val=1.234)
    assert os.path.exists(path), "checkpoint file was not written"
    assert not os.path.exists(path + ".tmp"), "temp file left behind — atomic rename?"

    # --- a probe input + reference logits from the trained model (eval mode: no dropout) ---
    model.eval()
    probe = torch.randint(0, cfg.vocab_size, (1, T))
    with torch.no_grad():
        ref_logits, _ = model(probe)

    # --- build a brand-new model/optimizer (different random init) and restore ---
    torch.manual_seed(0)
    model2, optimizer2 = fresh()
    # they really should differ before loading, else the test proves nothing
    same_before = all(
        torch.equal(a, b)
        for a, b in zip(model.state_dict().values(), model2.state_dict().values())
    )
    assert not same_before, "fresh model already equals the saved one (?!)"

    meta = load_checkpoint(path, model2, optimizer2, map_location="cpu")

    # 1. metadata came back, weights/optimizer state excluded from it
    assert meta["step"] == 5, f"bad step {meta.get('step')}"
    assert meta["best_val"] == 1.234, "extra payload (best_val) was lost"
    assert meta["config"]["n_layer"] == cfg.n_layer, "config not round-tripped"
    assert "model" not in meta and "optimizer" not in meta, "meta leaked state_dicts"

    # 2. every parameter now matches the saved model exactly
    for (k, a), (k2, b) in zip(model.state_dict().items(), model2.state_dict().items()):
        assert torch.equal(a, b), f"param mismatch after load: {k}"

    # 3. optimizer moments restored (compare the flattened state_dicts)
    s1 = optimizer.state_dict()["state"]
    s2 = optimizer2.state_dict()["state"]
    assert s1.keys() == s2.keys(), "optimizer state keys differ"
    for pid in s1:
        assert torch.equal(s1[pid]["exp_avg"], s2[pid]["exp_avg"]), "Adam m not restored"
        assert torch.equal(s1[pid]["exp_avg_sq"], s2[pid]["exp_avg_sq"]), "Adam v not restored"

    # 4. the restored model is functionally identical: same logits on the probe
    model2.eval()
    with torch.no_grad():
        got_logits, _ = model2(probe)
    assert torch.equal(ref_logits, got_logits), "restored model produces different logits"

    # 5. rebuild-from-config path: nothing but the file is needed to reconstruct
    rebuilt = GPT(GPTConfig(**meta["config"]))
    load_checkpoint(path, rebuilt)  # no optimizer, default map_location
    rebuilt.eval()
    with torch.no_grad():
        reb_logits, _ = rebuilt(probe)
    assert torch.equal(ref_logits, reb_logits), "config-rebuilt model differs"

    print("all checks passed ✅  (params, optimizer state, step, config, logits all round-tripped)")
