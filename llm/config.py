"""
Model + training hyperparameters in one place.

Using a dataclass (rather than scattering magic numbers across the code) is the
pattern GPT-2/nanoGPT use: the model takes a single `config` object, so the same
class can be instantiated at any size (124M, 350M, ...) just by changing fields.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 65      # set from the tokenizer
    block_size: int = 256     # max context length (sequence length)
    n_embd: int = 384         # embedding / residual stream width (d_model)
    n_head: int = 6           # number of QUERY heads
    # number of KEY/VALUE heads (GQA/MQA). None => n_kv_head = n_head = plain MHA.
    # Must divide n_head. n_kv_head == 1 is MQA (all query heads share one K/V);
    # 1 < n_kv_head < n_head is GQA (query heads grouped, each group shares a K/V).
    n_kv_head: int | None = None
    n_layer: int = 6          # number of transformer blocks
    dropout: float = 0.2
    tie_weights: bool = True  # share one (V,C) tensor for token_embedding & lm_head (GPT-2 style)
