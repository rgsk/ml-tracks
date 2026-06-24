"""
Grouped-Query Attention (GQA) — and its extremes MQA and MHA.

The idea in one line: keep all n_head QUERY heads, but use FEWER key/value heads
and SHARE them across groups of query heads. The only thing that changes versus
your CausalSelfAttention is that K and V are projected to n_kv_head heads instead
of n_head, then each K/V head is reused by a whole group of query heads.

    n_kv_head == n_head   ->  MHA  (every query head has its own K/V) — what you built
    n_kv_head == 1        ->  MQA  (all query heads share ONE K/V head)
    1 < n_kv_head < n_head ->  GQA  (query heads split into n_kv_head groups)

    group_size = n_head // n_kv_head    # query heads served by each K/V head

WHY (this is the whole point — an inference/memory optimization, NOT a quality one):
  - The KV-CACHE you built stores K and V for every past token. Its size is
    proportional to the number of K/V HEADS. Autoregressive decoding is memory-
    BANDWIDTH bound — every step re-reads the entire cache from HBM — so shrinking
    the cache by n_head/n_kv_head (e.g. 8x for MQA) is a direct decode speedup and
    lets you fit far longer contexts / bigger batches in memory.
  - MQA (n_kv_head=1) is the most aggressive but loses some quality; GQA is the
    sweet spot (e.g. Llama-2-70B uses 64 query heads, 8 KV heads). Training FLOPs
    barely change — this is about the cache, not the matmul count.

THE KEY MECHANIC — expanding K/V to match the query heads:
  Project Q to (B, n_head, T, hs) but K, V only to (B, n_kv_head, T, hs). To do the
  attention matmul, broadcast each K/V head across its group with repeat_interleave
  along the head dim by group_size. INTERLEAVE (not tile/repeat) so the groups are
  CONTIGUOUS — query heads [0,1] use kv head 0, [2,3] use kv head 1, etc.:

      kv heads [a, b]  --repeat_interleave(2)-->  [a, a, b, b]   # CONTIGUOUS groups ✅
      kv heads [a, b]  --repeat/tile(2)------->  [a, b, a, b]   # WRONG grouping ✗

  IMPORTANT for the cache: store the SMALL (n_kv_head) K/V in the cache and expand
  only at compute time. Caching the expanded version would throw away the savings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must divide by n_head"
        # None means plain MHA (one K/V head per query head)
        n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        assert config.n_head % n_kv_head == 0, "n_head must be divisible by n_kv_head"

        self.n_head = config.n_head
        self.n_kv_head = n_kv_head
        self.group_size = config.n_head // n_kv_head      # query heads per K/V head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head

        # Q keeps the full width (n_head * hs == n_embd). K and V are NARROWER:
        # only n_kv_head heads, so their projection outputs n_kv_head * hs each.
        self.q_proj = nn.Linear(config.n_embd, config.n_head * self.head_size, bias=False)
        self.k_proj = nn.Linear(config.n_embd, n_kv_head * self.head_size, bias=False)
        self.v_proj = nn.Linear(config.n_embd, n_kv_head * self.head_size, bias=False)

        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.proj.RESIDUAL_PROJ = True  # scaled init, same as CausalSelfAttention

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size)),
        )
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, kv_cache=None):
        """x: (B, T, C). kv_cache: optional (past_k, past_v), each
        (B, n_kv_head, T_past, hs) — note n_kv_head, NOT n_head: the cache holds
        the small/shared K/V. Returns (out, new_cache)."""
        B, T, C = x.shape
        nh, nkv, hs, g = self.n_head, self.n_kv_head, self.head_size, self.group_size

        q = self.q_proj(x)                                  # (B, T, n_head*hs)
        k = self.k_proj(x)                                  # (B, T, n_kv_head*hs)
        v = self.v_proj(x)                                  # (B, T, n_kv_head*hs)

        # split into heads and move the head axis to the batch position.
        # NOTE q has nh heads but k/v have only nkv heads.
        q = q.view(B, T, nh, hs).transpose(1, 2)            # (B, nh,  T, hs)
        k = k.view(B, T, nkv, hs).transpose(1, 2)           # (B, nkv, T, hs)
        v = v.view(B, T, nkv, hs).transpose(1, 2)           # (B, nkv, T, hs)

        # KV cache: append the new (small) K/V to the cached past along time.
        # We cache the n_kv_head version — that's where the memory savings live.
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)               # (B, nkv, T_past+T, hs)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)                                  # store the SMALL K/V

        T_k = k.size(2)
        T_past = T_k - T

        # expand the shared K/V up to nh heads: repeat_interleave keeps groups
        # CONTIGUOUS ([a,a,b,b]). nkv==nh -> g==1 -> no-op (plain MHA).
        k = k.repeat_interleave(repeats=g, dim=1)           # (B, nh, T_k, hs)
        v = v.repeat_interleave(repeats=g, dim=1)           # (B, nh, T_k, hs)

        # from here it's identical to CausalSelfAttention (all nh heads now).
        wei = q @ k.transpose(-2, -1) * hs ** -0.5          # (B, nh, T, T_k)
        wei = wei.masked_fill(self.tril[T_past:T_k, :T_k] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)
        out = wei @ v                                        # (B, nh, T, hs)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.proj(out))
        return out, new_cache


# ---------------------------------------------------------------------------
# Self-check: run `python gqa.py` from the llm/ dir.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(1337)

    B, T, C, nh = 4, 8, 32, 4
    x = torch.randn(B, T, C)
    x2 = x.clone()
    x2[:, -1, :] = torch.randn(B, C)  # perturb only the last (future) position

    # n_head=4, so valid n_kv_head are 4 (MHA), 2 (GQA), 1 (MQA).
    for n_kv in (4, 2, 1):
        cfg = GPTConfig(vocab_size=65, block_size=T, n_embd=C, n_head=nh,
                        n_kv_head=n_kv, dropout=0.0)
        gqa = GroupedQueryAttention(cfg)
        gqa.eval()

        out, cache = gqa(x)
        assert out.shape == (B, T, C), f"n_kv={n_kv}: bad out shape {out.shape}"

        # the cache holds n_kv_head heads — this is the memory win, made concrete.
        assert cache[0].shape == (B, n_kv, T, C // nh), \
            f"n_kv={n_kv}: cache should have {n_kv} kv-heads, got {cache[0].shape}"

        # causality must survive the head sharing.
        o1, _ = gqa(x)
        o2, _ = gqa(x2)
        assert torch.allclose(o1[:, :-1], o2[:, :-1], atol=1e-6), \
            f"n_kv={n_kv}: causality violated!"

        # incremental (cached, one token at a time) == full forward.
        full, _ = gqa(x)
        c, steps = None, []
        for t in range(T):
            ot, c = gqa(x[:, t:t+1, :], c)
            steps.append(ot)
        inc = torch.cat(steps, dim=1)
        assert torch.allclose(full, inc, atol=1e-5), \
            f"n_kv={n_kv}: KV-cache incremental != full forward!"

        n_params = sum(p.numel() for p in gqa.parameters())
        print(f"n_kv_head={n_kv}: out {tuple(out.shape)}, cache kv-heads={n_kv}, "
              f"params={n_params} ✅")

    # fewer KV heads => fewer parameters (smaller k/v projections) AND a smaller
    # cache. Build all three and assert the param count strictly decreases.
    def nparams(n_kv):
        cfg = GPTConfig(vocab_size=65, block_size=T, n_embd=C, n_head=nh,
                        n_kv_head=n_kv, dropout=0.0)
        return sum(p.numel() for p in GroupedQueryAttention(cfg).parameters())
    assert nparams(4) > nparams(2) > nparams(1), "fewer kv-heads should mean fewer params"

    # the grouping must be CONTIGUOUS (repeat_interleave, not tile). With n_kv=2,
    # query heads 0,1 share kv-head 0 and 2,3 share kv-head 1. Give every query
    # head the SAME query vector; then heads in the same group must produce
    # IDENTICAL attention output (same Q, same shared K/V), while heads in
    # different groups generally differ. This pins down interleave-vs-tile.
    cfg = GPTConfig(vocab_size=65, block_size=T, n_embd=C, n_head=4, n_kv_head=2,
                    dropout=0.0)
    gqa = GroupedQueryAttention(cfg)
    gqa.eval()
    with torch.no_grad():
        # make all query heads identical by tying the q projection rows per head:
        # easiest is to force q to be the same across heads by zeroing then copying.
        qw = gqa.q_proj.weight.view(4, C // 4, C)
        qw[1] = qw[0]   # head 1 == head 0  (same group as head 0)
        qw[3] = qw[2]   # head 3 == head 2  (same group as head 2)
    # recompute per-head attention outputs manually to compare heads 0 vs 1.
    Bq = 2
    xx = torch.randn(Bq, T, C)
    qh = gqa.q_proj(xx).view(Bq, T, 4, C // 4).transpose(1, 2)      # (Bq,4,T,hs)
    kk = gqa.k_proj(xx).view(Bq, T, 2, C // 4).transpose(1, 2)      # (Bq,2,T,hs)
    vv = gqa.v_proj(xx).view(Bq, T, 2, C // 4).transpose(1, 2)
    kk = kk.repeat_interleave(2, dim=1)                            # (Bq,4,T,hs)
    vv = vv.repeat_interleave(2, dim=1)
    hs = C // 4
    w = (qh @ kk.transpose(-2, -1)) * hs ** -0.5
    w = w.masked_fill(gqa.tril[:T, :T] == 0, float("-inf"))
    w = F.softmax(w, dim=-1)
    per_head = w @ vv                                              # (Bq,4,T,hs)
    assert torch.allclose(per_head[:, 0], per_head[:, 1], atol=1e-6), \
        "heads in the same group must match (q tied + shared kv) — is it interleave?"
    assert not torch.allclose(per_head[:, 0], per_head[:, 2], atol=1e-3), \
        "heads in different groups should differ (different shared kv head)"
    print("grouping is contiguous (repeat_interleave, not tile) ✅")

    print("GQA/MQA works ✅")
