"""
Param counting, FLOPs, and MFU (Model FLOPs Utilization).

Three interview-classic estimates, all derivable from the config alone:

1. PARAM COUNT — how many learnable numbers are in the model. Derived
   analytically from the architecture (then verified against the real
   sum(p.numel())). Where the params live tells you a lot:
     - token_embedding  vocab_size * n_embd      (a lookup table, not a matmul)
     - position_embed   block_size * n_embd       (lookup table)
     - per block:
         ln1, ln2         2 * n_embd each          (weight + bias)
         attn.qkv         n_embd * 3*n_embd        (bias=False)
         attn.proj        n_embd * n_embd + n_embd
         ffwd up          n_embd * 4*n_embd + 4*n_embd
         ffwd down        4*n_embd * n_embd + n_embd
     - ln_f               2 * n_embd
     - lm_head            n_embd * vocab_size + vocab_size  (a real matmul)
   In big models the per-block matmuls (attn + the 4x MLP) dominate; in tiny
   models the embedding tables do.

2. FLOPs PER TOKEN — the famous "6N" rule. A matmul that uses a weight does one
   multiply + one add per element = 2 FLOPs per parameter per token. So:
       forward  ≈ 2 * N FLOPs/token
       backward ≈ 4 * N FLOPs/token   (≈ 2x forward: grads wrt inputs AND weights)
       total    ≈ 6 * N FLOPs/token
   Here N is the NON-EMBEDDING parameter count: embedding lookups are gathers,
   not matmuls, so they cost ~0 FLOPs and are excluded. The lm_head IS a matmul,
   so it stays in N.
       The 6N rule misses attention's score/value matmuls (q@k^T and wei@v),
   because those aren't parameterized — they scale with T^2, not with params. The
   correction term is 12 * n_layer * n_head * head_size * block_size per token
   (a 6N-style 2x-fwd, 2x-bwd count of the two T*T*hs matmuls). It's small for
   short context, but grows with T and eventually dominates.

3. MFU — what fraction of the GPU's peak FLOP/s you actually achieve:
       achieved_flops_per_sec = flops_per_iter / seconds_per_iter
       MFU = achieved_flops_per_sec / gpu_peak_flops_per_sec
   ~0.3-0.5 is good for a well-tuned training run; low MFU means you're bottle-
   necked on memory bandwidth / launch overhead / data loading, not on math.
   Peak refs (dense, no sparsity): A100 bf16 312e12, H100 bf16 ~990e12,
   V100 fp16 125e12, 4090 bf16 ~165e12.

Run `python flops.py` from the llm/ dir to self-check.
"""

from __future__ import annotations

from config import GPTConfig


def num_params(config: GPTConfig, *, embedding: bool = True) -> int:
    """Analytic parameter count from the config.

    embedding=True  -> the full model count (matches sum(p.numel())).
    embedding=False -> subtract the token + position embedding tables. This is
                       the N to feed the 6N FLOPs rule (lookups do no matmul
                       FLOPs). The lm_head is a matmul, so it is NOT subtracted.
    """
    V, T, C = config.vocab_size, config.block_size, config.n_embd
    L = config.n_layer

    # two lookup tables (no bias)
    tok_emb = V * C
    pos_emb = T * C

    # one transformer block: two LayerNorms (weight + bias), attention
    # (qkv is C->3C bias-free, proj is C->C with bias), and the 4x MLP.
    ln1 = ln2 = 2 * C
    attn = (C * 3 * C) + (C * C + C)
    ffwd = (C * 4 * C + 4 * C) + (4 * C * C + C)
    per_block = ln1 + ln2 + attn + ffwd

    ln_f = 2 * C
    lm_head = C * V + V                                  # a real matmul (C->V)

    total = tok_emb + pos_emb + L * per_block + ln_f + lm_head
    if not embedding:
        # drop the two lookup tables (no matmul FLOPs); keep lm_head
        total -= tok_emb + pos_emb
    return total


# check flops_per_token_calc.png for explaination
def flops_per_token(config: GPTConfig) -> int:
    """FLOPs for one token, forward + backward, via 6N + attention correction."""
    L = config.n_layer
    H = config.n_head
    Q = config.n_embd // config.n_head          # head_size
    T = config.block_size

    # 6N: every non-embedding param does one multiply-add (2 FLOPs) on the
    # forward pass and ~twice that on the backward = 6 FLOPs/param/token.
    N = num_params(config, embedding=False)
    matmul_flops = 6 * N

    # attention's two unparameterized matmuls (q@k^T and wei@v) aren't in N;
    # they scale with T (T^2 per sequence). Same 6N-style fwd+bwd counting.
    attn_flops = 12 * L * H * Q * T

    return matmul_flops + attn_flops


def flops_per_iter(config: GPTConfig, tokens_per_iter: int) -> int:
    """Total FLOPs for one optimizer step that processed `tokens_per_iter` tokens.

    tokens_per_iter = batch_size * block_size * grad_accum_steps (every token in
    the iteration pays flops_per_token, which already includes fwd + bwd).
    """
    return flops_per_token(config) * tokens_per_iter


def estimate_mfu(config: GPTConfig, tokens_per_iter: int, dt: float,
                 gpu_peak_flops: float) -> float:
    """Model FLOPs Utilization: achieved FLOP/s as a fraction of the GPU peak.

    dt = wall-clock seconds for the iteration. gpu_peak_flops = the device's
    advertised peak (e.g. 312e12 for an A100 in bf16).
    """
    achieved = flops_per_iter(config, tokens_per_iter) / dt   # FLOP/s this iter
    return achieved / gpu_peak_flops


# ---------------------------------------------------------------------------
# Self-check: run `python flops.py` from the llm/ dir. Don't edit below this line.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import torch
    from model import GPT

    # --- 1. analytic param count must match the real model, exactly ----------
    cfg = GPTConfig(vocab_size=65, block_size=8, n_embd=32, n_head=4, n_layer=3)
    model = GPT(cfg)
    real = sum(p.numel() for p in model.parameters())
    mine = num_params(cfg)
    assert mine == real, f"analytic params {mine} != real {real}"
    print(f"param count: analytic {mine} == real {real} ✅")

    # check it tracks the real model at a second, differently-shaped config too
    cfg2 = GPTConfig(vocab_size=100, block_size=16, n_embd=64, n_head=8, n_layer=4)
    real2 = sum(p.numel() for p in GPT(cfg2).parameters())
    assert num_params(cfg2) == real2, f"{num_params(cfg2)} != {real2}"

    # non-embedding count subtracts exactly the two lookup tables
    emb = (cfg.vocab_size + cfg.block_size) * cfg.n_embd
    assert num_params(cfg, embedding=False) == real - emb, "non-embedding subtraction wrong"
    print(f"non-embedding params: {num_params(cfg, embedding=False)} "
          f"(dropped {emb} embedding params)")

    # --- 2. FLOPs per token --------------------------------------------------
    fpt = flops_per_token(cfg)
    N = num_params(cfg, embedding=False)
    assert fpt > 6 * N, "total must exceed the bare 6N (attention term is positive)"
    # the attention correction is exactly the surplus over 6N
    attn_term = fpt - 6 * N
    assert attn_term == 12 * cfg.n_layer * cfg.n_head * (cfg.n_embd // cfg.n_head) * cfg.block_size
    print(f"flops/token: {fpt}  (6N={6*N}, attn={attn_term})")

    # longer context => bigger attention share (T^2 cost), so attn fraction grows
    long_cfg = GPTConfig(vocab_size=65, block_size=1024, n_embd=32, n_head=4, n_layer=3)
    frac_short = (flops_per_token(cfg) - 6 * num_params(cfg, embedding=False)) / flops_per_token(cfg)
    frac_long = (flops_per_token(long_cfg) - 6 * num_params(long_cfg, embedding=False)) / flops_per_token(long_cfg)
    assert frac_long > frac_short, "attention's FLOP share should grow with context length"
    print(f"attention FLOP share: T=8 -> {frac_short:.4f}, T=1024 -> {frac_long:.2%}")

    # --- 3. flops_per_iter scales linearly in tokens -------------------------
    tokens = 4 * cfg.block_size            # B=4
    assert flops_per_iter(cfg, tokens) == fpt * tokens
    assert flops_per_iter(cfg, 2 * tokens) == 2 * flops_per_iter(cfg, tokens)

    # --- 4. MFU --------------------------------------------------------------
    peak = 312e12                          # A100 bf16
    # if achieved exactly equals peak, MFU must be 1.0
    work = flops_per_iter(cfg, tokens)
    dt_full = work / peak                  # the dt that saturates the GPU
    assert abs(estimate_mfu(cfg, tokens, dt_full, peak) - 1.0) < 1e-9, "saturating dt should give MFU=1"
    # twice as slow => half the MFU
    m1 = estimate_mfu(cfg, tokens, dt_full, peak)
    m2 = estimate_mfu(cfg, tokens, 2 * dt_full, peak)
    assert abs(m2 - m1 / 2) < 1e-12, "doubling dt should halve MFU"
    assert 0 < m2 < m1, "slower iter must mean lower utilization"
    print(f"MFU: dt={dt_full:.2e}s -> {m1:.1%}, dt={2*dt_full:.2e}s -> {m2:.1%} ✅")

    # --- sanity: a GPT-2-124M-ish config lands near the famous numbers --------
    gpt2 = GPTConfig(vocab_size=50257, block_size=1024, n_embd=768, n_head=12, n_layer=12)
    p = num_params(gpt2)
    print(f"GPT-2(124M)-shaped config: {p:,} params  (~124M expected, sans weight-tying diff)")
    print("all checks passed ✅")
