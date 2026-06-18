"""
Transformer building blocks.

FeedForward: a position-wise MLP. Attention moves information BETWEEN positions;
the FeedForward then lets each position "think" on its own (the same MLP applied
independently per position). It expands 4x, applies GELU, and projects back:

    C  --Linear-->  4C  --GELU-->  4C  --Linear-->  C  --dropout-->  C

Block: one transformer layer = attention + feedforward, each wrapped in a
residual connection with pre-LayerNorm.

Run `python transformer.py` from the llm/ dir to self-check.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from attention import CausalSelfAttention
from config import GPTConfig


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # expand 4x (more capacity), GELU (GPT-2's smooth activation), project back
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    """Exercise 8: one transformer block = attention + feedforward, each wrapped
    in a residual connection with pre-LayerNorm.

    GPT-2 uses the PRE-norm formulation:
        x = x + attn(ln1(x))
        x = x + ffwd(ln2(x))

    Two ideas doing the heavy lifting:
      - Residual (x + ...): the block LEARNS a delta to add to the stream, so
        information (and gradient) flows straight through even in deep stacks.
        This is what makes training many layers stable.
      - Pre-LayerNorm: normalize the INPUT to each sub-layer (not the output).
        Normalizing per-token across the C features keeps activations well-scaled
        so the residual stream doesn't blow up through depth.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        # one LayerNorm before each sub-layer (pre-norm), normalizing over C
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ffwd = FeedForward(config)

    def forward(self, x: torch.Tensor, kv_cache=None):
        # pre-norm + residual, twice. Note: x + sublayer(ln(x)), NOT ln(x + ...).
        # attention returns its updated KV cache, which we thread back out so the
        # caller (GPT) can keep one cache per layer across generation steps.
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache)
        x = x + attn_out
        x = x + self.ffwd(self.ln2(x))
        return x, new_cache


# ---------------------------------------------------------------------------
# Self-check: run `python transformer.py` from the llm/ dir.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(1337)

    cfg = GPTConfig(vocab_size=65, block_size=8, n_embd=32, n_head=4, dropout=0.0)
    ff = FeedForward(cfg)
    ff.eval()

    B, T = 4, cfg.block_size
    x = torch.randn(B, T, cfg.n_embd)
    out = ff(x)

    # shape preserved (it's a residual-stream module)
    assert out.shape == (B, T, cfg.n_embd), f"bad shape {out.shape}"

    # position-wise: row t of the output depends ONLY on row t of the input.
    # Changing position T-1 must not change the output at earlier positions.
    x2 = x.clone()
    x2[:, -1, :] = torch.randn(B, cfg.n_embd)
    assert torch.allclose(ff(x)[:, :-1], ff(x2)[:, :-1], atol=1e-6), \
        "FeedForward leaked across positions — it must act per-position!"

    n_params = sum(p.numel() for p in ff.parameters())
    print("ff out shape:", tuple(out.shape))
    print(f"feedforward works ✅  (params: {n_params})")

    # --- Block ---
    block = Block(cfg)
    block.eval()

    bout, _ = block(x)
    assert bout.shape == (B, T, cfg.n_embd), f"bad block shape {bout.shape}"

    # the block contains attention, so it must STILL be causal end-to-end
    b1, _ = block(x)
    b2, _ = block(x2)
    assert torch.allclose(b1[:, :-1], b2[:, :-1], atol=1e-6), \
        "causality violated in Block!"

    print("block out shape:", tuple(bout.shape))
    print("transformer block works ✅")
