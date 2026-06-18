"""
Mixed-precision training: run the heavy math in 16-bit, keep a 32-bit master copy.
On a modern GPU this is ~2-3x faster and roughly halves memory, for ~free.

Three pieces, and the WHY of each:

1. autocast (torch.autocast): a context manager around the FORWARD + LOSS. Inside
   it, big ops (matmuls) run in 16-bit while numerically-touchy ops (softmax,
   LayerNorm, the loss reduction) stay in fp32 automatically. You do NOT wrap
   backward — autograd replays each op in whatever dtype the forward used.

2. GradScaler (fp16 only): fp16 has a tiny exponent range, so small gradients
   UNDERFLOW to 0 and the signal is lost. Fix: multiply the loss by a big scale S
   before backward, so every gradient is scaled up by S into fp16's representable
   range; then UNSCALE (divide by S) before the optimizer step. The scaler also
   auto-tunes S and skips any step that produced inf/nan (overflow).
     - bf16 has the SAME exponent range as fp32 (just fewer mantissa bits), so it
       doesn't underflow -> no scaler needed (we disable it). fp16 needs it.

3. ordering with grad clipping: after backward, the grads are still scaled by S.
   To clip by the TRUE norm you must unscale FIRST, then clip, then step:
       scaler.scale(loss).backward()
       scaler.unscale_(optimizer)        # grads now real-valued
       clip_grad_norm_(...)              # clip the real grads
       scaler.step(optimizer)           # skips the step if grads are inf/nan
       scaler.update()                  # adjust S for next iter

Master weights stay fp32 the whole time — only the forward math is 16-bit.

Fill in the TODO, then run `python mixed_precision.py` from the llm/ dir.
"""

from __future__ import annotations

import torch


def amp_train_step(model, xb, yb, optimizer, scaler, *, amp_dtype, device_type,
                   grad_clip: float = 0.0):
    """One mixed-precision training step. Returns the loss tensor.

    Works for both bf16 (pass a disabled scaler -> the scaler ops are no-ops) and
    fp16 (pass an enabled scaler). Same code path either way.
    """
    optimizer.zero_grad(set_to_none=True)

    # forward + loss in 16-bit; autocast wraps ONLY the forward, never backward.
    with torch.autocast(device_type=device_type, dtype=amp_dtype):
        _, loss = model(xb, yb)

    # scaled backward: grads come out multiplied by the scaler's factor S (fp16);
    # for bf16 the scaler is disabled, so this is a plain loss.backward().
    scaler.scale(loss).backward()

    # clip on the REAL grads: unscale first, otherwise we'd clip the S-inflated ones.
    if grad_clip:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    scaler.step(optimizer)     # unscales (if not already) and steps; skips on inf/nan
    scaler.update()            # grow/shrink S for next iteration
    return loss


# ---------------------------------------------------------------------------
# Self-check: demonstrates the fp16-underflow problem, then trains a tiny GPT in
# bf16 (and fp16 on CUDA). Run from the llm/ dir:  python mixed_precision.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math

    from config import GPTConfig
    from model import GPT

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=65, block_size=16, n_embd=64, n_head=4, n_layer=2)

    # --- 1. WHY GradScaler exists: fp16 underflows a small gradient to 0, and
    #        loss-scaling rescues it. (Pure-tensor demo, no model needed.) ---
    tiny = torch.tensor(2.0 ** -25)                 # ~3e-8, below fp16's min subnormal
    assert tiny.half().item() == 0.0, "expected fp16 underflow to 0"
    recovered = (tiny * 2**15).half().float() / 2**15   # scale up -> fp16 -> unscale
    assert recovered.item() == tiny.item(), "loss scaling should preserve the value"
    print(f"fp16 underflow: {tiny.item():.2e} -> {tiny.half().item():.0f}  |  "
          f"scaled then unscaled -> {recovered.item():.2e} preserved ✅")

    # --- 2. a real AMP training step: trains, runs forward in 16-bit, keeps fp32 weights ---
    def run(amp_dtype, use_scaler):
        torch.manual_seed(0)
        model = GPT(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler(device, enabled=use_scaler)

        # autocast really does drop the forward output to 16-bit
        xb0 = torch.randint(0, cfg.vocab_size, (8, cfg.block_size), device=device)
        with torch.autocast(device_type=device, dtype=amp_dtype):
            logits, _ = model(xb0)
        assert logits.dtype == amp_dtype, f"autocast should yield {amp_dtype}, got {logits.dtype}"

        losses = []
        for _ in range(30):
            xb = torch.randint(0, cfg.vocab_size, (8, cfg.block_size), device=device)
            yb = torch.randint(0, cfg.vocab_size, (8, cfg.block_size), device=device)
            loss = amp_train_step(model, xb, yb, opt, scaler, amp_dtype=amp_dtype,
                                  device_type=device, grad_clip=1.0)
            losses.append(loss.item())

        assert all(math.isfinite(l) for l in losses), "loss went non-finite under AMP"
        # autocast doesn't touch stored params — EVERY master weight stays fp32
        assert all(p.dtype == torch.float32 for p in model.parameters()), \
            "master weights must stay fp32"
        return losses

    bf = run(torch.bfloat16, use_scaler=False)      # bf16: no scaler needed
    print(f"bf16  (no scaler) : loss {bf[0]:.3f} -> {bf[-1]:.3f}")
    if device == "cuda":
        fp = run(torch.float16, use_scaler=True)    # fp16: GradScaler required
        print(f"fp16  (GradScaler): loss {fp[0]:.3f} -> {fp[-1]:.3f}")
    else:
        print("fp16 path skipped (no CUDA) — GradScaler is a CUDA-fp16 thing")
    print("mixed precision works ✅")
