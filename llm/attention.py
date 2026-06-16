"""
Phase 2 - Exercise 4: single-head causal self-attention.

Each position becomes a weighted average of the VALUES of all positions it is
allowed to see (itself + everything before it). The weights come from comparing
this position's QUERY against every other position's KEY.

Shapes:  B = batch, T = time/seq, C = n_embd, hs = head_size
    x          (B, T, C)
    q, k, v    (B, T, hs)
    wei        (B, T, T)   row t = how much position t attends to each position
    out        (B, T, hs)

Run `python attention.py` from the llm/ dir to self-check.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


class Head(nn.Module):
    def __init__(self, config: GPTConfig, head_size: int):
        super().__init__()
        self.head_size = head_size

        # projections from the residual stream (C) to head_size; bias=False per GPT
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)

        # causal mask as a buffer: fixed (no grad), but moves with .to(device)
        # and is saved in state_dict (a plain attribute would do neither)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size)),
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        k = self.key(x)                                          # (B, T, hs)
        q = self.query(x)                                        # (B, T, hs)

        # scores: wei[b, i, j] = q[b, i] . k[b, j], scaled to keep variance ~1
        wei = q @ k.transpose(-2, -1) * self.head_size ** -0.5   # (B, T, T)
        # mask future keys with -inf so they get 0 weight after softmax
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)                             # rows sum to 1
        wei = self.dropout(wei)

        v = self.value(x)                                        # (B, T, hs)
        out = wei @ v                                            # (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """Exercise 5 (explicit version): n_head independent Heads in parallel.

    Each head produces (B, T, head_size); we concat them along the channel dim
    back to (B, T, C), then apply an output projection (the "mixing" linear that
    lets information from different heads interact).

    Why multiple heads? One head learns one kind of relationship (e.g. "attend
    to the previous verb"). Multiple smaller heads let the model attend to
    several different things at once, then combine them.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must divide by n_head"
        head_size = config.n_embd // config.n_head

        # ModuleList (not a plain list) so the heads' params register/move/save
        self.heads = nn.ModuleList(
            [Head(config, head_size) for _ in range(config.n_head)]
        )
        # mixes information across heads after concatenation
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, C)
        out = self.dropout(self.proj(out))                    # (B, T, C)
        return out


class CausalSelfAttention(nn.Module):
    """Exercise 6 (fused version): same math as MultiHeadAttention, but all
    heads computed in a few big batched matmuls instead of a Python loop.

    The trick: project x to Q, K, V each of width C in ONE linear, then reshape
    C into (n_head, head_size) and treat n_head as an extra batch dimension, so
    a single batched matmul does all heads at once.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head

        # one projection producing Q, K, V stacked (C -> 3C); split in forward
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size)),
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        nh, hs = self.n_head, self.head_size

        # one matmul, then split into q, k, v each (B, T, C)
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.n_embd, dim=-1)

        # split C into (nh, hs) and move nh to the batch position -> (B, nh, T, hs)
        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)

        # batched over (B, nh): scores -> mask -> softmax -> weighted values
        wei = q @ k.transpose(-2, -1) * hs ** -0.5               # (B, nh, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)
        out = wei @ v                                            # (B, nh, T, hs)

        # merge heads back to (B, T, C). transpose breaks contiguity, so we must
        # .contiguous() before .view (the .view-vs-.reshape gotcha)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.proj(out))
        return out


# ---------------------------------------------------------------------------
# Self-check: run `python attention.py` from the llm/ dir.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(1337)

    cfg = GPTConfig(vocab_size=65, block_size=8, n_embd=32, n_head=4, dropout=0.0)
    assert cfg.n_embd % cfg.n_head == 0, "n_embd must divide by n_head"
    head_size = cfg.n_embd // cfg.n_head
    head = Head(cfg, head_size)
    head.eval()  # disable dropout so the causality test is deterministic

    B, T = 4, cfg.block_size
    x = torch.randn(B, T, cfg.n_embd)

    out = head(x)
    assert out.shape == (B, T, head_size), f"bad out shape {out.shape}"

    # works for T < block_size too (the tril slice)
    short = head(torch.randn(B, 5, cfg.n_embd))
    assert short.shape == (B, 5, head_size), f"bad short shape {short.shape}"

    # THE causality test: perturbing a FUTURE position must not change the
    # output at earlier positions. This is the causal mask, operationalized.
    out1 = head(x)
    x2 = x.clone()
    x2[:, -1, :] = torch.randn(B, cfg.n_embd)  # change only the last position
    out2 = head(x2)
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-6), \
        "causality violated: an earlier position depends on a future token!"

    print("out shape:", tuple(out.shape))
    print("causal self-attention works ✅")

    # --- MultiHeadAttention ---
    mha = MultiHeadAttention(cfg)
    mha.eval()

    mout = mha(x)
    # output keeps the full channel width C (heads concatenated + projected)
    assert mout.shape == (B, T, cfg.n_embd), f"bad mha shape {mout.shape}"

    # causality must still hold through multi-head + projection
    m1, m2 = mha(x), mha(x2)
    assert torch.allclose(m1[:, :-1], m2[:, :-1], atol=1e-6), \
        "causality violated in multi-head attention!"

    print("mha out shape:", tuple(mout.shape))
    print("multi-head attention works ✅")

    # --- CausalSelfAttention (fused) ---
    csa = CausalSelfAttention(cfg)
    csa.eval()

    cout = csa(x)
    assert cout.shape == (B, T, cfg.n_embd), f"bad csa shape {cout.shape}"

    c1, c2 = csa(x), csa(x2)
    assert torch.allclose(c1[:, :-1], c2[:, :-1], atol=1e-6), \
        "causality violated in fused attention!"

    # parameter count should match the explicit version (same math, same params:
    # 3 projections of C->C for qkv, plus a C->C output proj). Note: the explicit
    # MHA uses bias on its qkv-equivalent? No — Head uses bias=False, so counts
    # match only if qkv is bias=False too (it is).
    n_csa = sum(p.numel() for p in csa.parameters())
    n_mha = sum(p.numel() for p in mha.parameters())
    assert n_csa == n_mha, f"param mismatch: fused {n_csa} vs explicit {n_mha}"

    print("csa out shape:", tuple(cout.shape))
    print(f"fused attention works ✅  (params match explicit: {n_csa})")
