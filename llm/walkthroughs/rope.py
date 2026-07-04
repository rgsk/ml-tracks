"""
WALKTHROUGH: RoPE (Rotary Position Embeddings), one layer at a time.

Our GPT injects position with a LEARNED lookup table: position_embedding is an
(block_size, C) matrix, and we ADD row t to the token at position t. Two problems:

  1. HARD LENGTH CAP. The table has exactly block_size rows. Position block_size
     doesn't exist -> the model literally cannot run on a longer sequence. That's
     the assert you saw in generate()/KV-cache.
  2. IT'S ABSOLUTE, but attention cares about DISTANCE. What matters for "the verb
     agrees with its subject" is that they're 3 apart, not that they sit at
     positions 40 and 43 vs 900 and 903. Absolute embeddings make the model
     re-learn every relative pattern at every absolute location.

RoPE fixes both. Instead of ADDING a position vector at the input, it ROTATES the
query and key vectors (inside attention) by an angle proportional to their
position. The magic: the dot product of a query at position m and a key at
position n then depends ONLY on (m - n) — relative position falls out of the math,
for free, with ZERO learned parameters and NO length cap.

    q_m . k_n  =  <R(mθ) q,  R(nθ) k>  =  q^T R((n-m)θ) k      # only n-m survives

This is what Llama, GPT-NeoX, Mistral, Qwen all use.

Layers (run each `exp_*`, watch the output, then say "next"):
  1. the problem — learned absolute pos emb: the length cap + no distance sense.  (here)
  2. rotate pairs — split a head vector into 2D planes, rotate each by m*theta; norms are preserved.
  3. the relative-position magic — <R_m q, R_n k> depends only on n-m (the whole point).
  4. frequencies — theta_k = base^(-2k/d): a spectrum of rotation speeds, local -> global.
  5. rotate_half — the cheap production formula (x*cos + rotate_half(x)*sin) == the explicit rotation.
  6. RoPESelfAttention — drop it into the real head layout; causal, KV-cache-correct, and NO length cap.

Context extension (layers 7-10): "no length cap" means RoPE RUNS at any length, but a
model TRAINED at length L still DEGRADES past L. Why, and the three standard fixes:
  7. the problem — past L the SLOW (long-wavelength) planes rotate into angles never seen in training.
  8. Position Interpolation (PI) — scale positions by L/L' so every distance maps back into the trained arc.
  9. NTK-aware — raise the base so slow planes stretch but fast planes keep local resolution (one-line, no fine-tune).
  10. YaRN — explicit extrapolate-fast / interpolate-slow split + an attention-temperature fix (Llama-3.1/Qwen2).

This is the SCRATCH file; you rebuild a clean rope.py yourself afterwards.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys

# this file lives in llm/walkthroughs/; its building blocks live in the parent
# llm/ package dir — put that on the import path first.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LLM = os.path.dirname(_HERE)
sys.path.insert(0, _LLM)

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import CausalSelfAttention
from config import GPTConfig
from model import GPT


def _banner(*lines):
    print("=" * 70)
    for line in lines:
        print(line)
    print("=" * 70)


# ---------------------------------------------------------------------------
# LAYER 1: the problem — a learned ABSOLUTE position table.
#
# Our GPT does (model.py):  x = token_embedding(idx) + position_embedding(pos)
# where position_embedding is nn.Embedding(block_size, C): one LEARNED vector per
# absolute slot 0..block_size-1, added at the input.
#
# Two costs, both of which RoPE removes:
#   * a length CAP: there is no row for position >= block_size, so the model can't
#     process (or extrapolate to) a longer context. (This is the KV-cache assert.)
#   * ABSOLUTE, not relative: the model sees "you are at slot 43", never "you are 3
#     after the subject" directly — it must re-learn relative patterns everywhere.
#
# This layer just makes the cost concrete: the table's shape/params and the cap.
# ---------------------------------------------------------------------------
def exp_1_the_problem(seed=0):
    """Show the learned absolute position table on a real GPT: its (block_size, C)
    shape, its parameter count, and the hard length cap it imposes."""
    _banner("LAYER 1: learned ABSOLUTE position emb — a (block_size,C) table + a length cap")
    torch.manual_seed(seed)

    cfg = GPTConfig(vocab_size=65, block_size=256, n_embd=384, n_head=6, n_layer=6)
    model = GPT(cfg)

    pos = model.position_embedding.weight                # (block_size, C)
    n_pos = pos.numel()
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  position_embedding.weight shape = {tuple(pos.shape)}  (one learned vector per slot)")
    print(f"  learned position params: {n_pos:,}  ({n_pos/n_total:.1%} of the {n_total:,}-param model)")
    print(f"  -> RoPE will replace ALL of these with 0 learned params.\n")

    # the length cap: there is no embedding row for position == block_size.
    print(f"  block_size = {cfg.block_size}: valid positions are 0..{cfg.block_size-1}.")
    try:
        model.position_embedding(torch.tensor([cfg.block_size]))
        print("  (unexpected: lookup at block_size succeeded)")
    except IndexError:
        print(f"  position_embedding(pos={cfg.block_size}) -> IndexError: the table has no such row.")
    print("  So the model CANNOT run beyond block_size — no extrapolation. (= the KV-cache assert.)")

    print("\n  ABSOLUTE vs RELATIVE: this table encodes 'which slot am I', but attention")
    print("  really wants 'how far apart are query and key'. RoPE injects position so")
    print("  that the query-key score depends only on the DISTANCE (m-n), not the slot.")
    print("  Next (layer 2): the primitive — rotate a head vector's 2D planes by m*theta.")


# ---------------------------------------------------------------------------
# LAYER 2: rotate pairs — the RoPE primitive.
#
# Take one head vector of size d (= head_size, must be even). Pair up its
# coordinates into d/2 two-dimensional planes. We use the HALF-SPLIT pairing
# (Llama/HF): coordinate k is paired with coordinate k + d/2. Each plane is
# rotated by an angle (position m) * (per-plane frequency theta_k):
#
#     [ out[k]     ]   [ cos(m*theta_k)  -sin(m*theta_k) ] [ x[k]     ]
#     [ out[k+d/2] ] = [ sin(m*theta_k)   cos(m*theta_k) ] [ x[k+d/2] ]
#
# That's it — RoPE is just d/2 independent 2D rotations, with the rotation ANGLE
# set by the token's position. Two consequences we verify here:
#   * a rotation preserves LENGTH: ||rope(x, m)|| == ||x|| for every position m.
#     Position changes the DIRECTION of each plane, never the magnitude — so it
#     can't blow up or wash out the vector the way an additive embedding could.
#   * as m grows, each plane sweeps around a circle at its own speed theta_k.
# ---------------------------------------------------------------------------
def rope_frequencies(head_size: int, base: float = 10000.0) -> torch.Tensor:
    """theta_k = base^(-2k/d) for k = 0..d/2-1. Returns a (d/2,) tensor of the
    per-plane rotation frequencies (layer 4 explains the base^... choice)."""
    idx = torch.arange(0, head_size, 2).float()          # 0,2,4,...,d-2  -> length d/2
    return base ** (-idx / head_size)                    # base^(-2k/d)


def apply_rope_explicit(x: torch.Tensor, pos: torch.Tensor,
                        base: float = 10000.0) -> torch.Tensor:
    """The literal per-plane rotation, written as a loop for clarity.
    x: (T, d) one sequence of head vectors. pos: (T,) absolute positions.
    Rotates plane (k, k+d/2) of row t by pos[t]*theta_k. (Layer 5 vectorizes this.)"""
    T, d = x.shape
    theta = rope_frequencies(d, base)                    # (d/2,)
    out = torch.empty_like(x)
    for t in range(T):
        for k in range(d // 2):
            ang = pos[t].item() * theta[k].item()
            c, s = math.cos(ang), math.sin(ang)
            a, b = x[t, k].item(), x[t, k + d // 2].item()
            out[t, k] = a * c - b * s
            out[t, k + d // 2] = a * s + b * c
    return out


def exp_2_rotate_pairs(seed=0):
    """Rotate one head vector across positions 0..5 and show (1) the norm is
    preserved at every position, (2) one 2D plane sweeps around a circle."""
    _banner("LAYER 2: RoPE = d/2 independent 2D rotations by angle (position * theta_k)")
    torch.manual_seed(seed)

    d = 8                                                 # a small head_size
    x = torch.randn(d)
    positions = list(range(6))

    print(f"  head vector d = {d}  ->  {d//2} rotation planes, pairing dim k with dim k+{d//2}")
    print(f"  original ||x|| = {x.norm():.4f}\n")
    print("   pos m | ||rope(x, m)|| | plane0 = (out[0], out[4])")
    print("  -------+----------------+---------------------------")
    for m in positions:
        r = apply_rope_explicit(x.unsqueeze(0), torch.tensor([float(m)])).squeeze(0)
        plane0 = (r[0].item(), r[d // 2].item())
        print(f"    {m:3d}  |    {r.norm():.4f}      | ({plane0[0]:+.3f}, {plane0[1]:+.3f})")

    print("\n  Every norm equals ||x|| — rotation changes DIRECTION, never magnitude.")
    print("  plane0 traces a circle of radius sqrt(x[0]^2 + x[4]^2) = "
          f"{math.hypot(x[0].item(), x[d//2].item()):.4f} as m advances.")
    print("  Next (layer 3): why this rotation makes the q.k score depend only on n-m.")


# ---------------------------------------------------------------------------
# LAYER 3: the relative-position magic — the whole reason RoPE works.
#
# Rotate the query at its position m and the key at its position n, THEN take their
# dot product (which is exactly what attention does: score = q . k). Because each
# plane's rotation is orthogonal, R(mθ)^T R(nθ) = R((n-m)θ), so:
#
#     <R(mθ) q,  R(nθ) k>  =  q^T R(mθ)^T R(nθ) k  =  q^T R((n-m)θ) k
#
# The absolute positions m and n CANCEL — only their difference n-m survives. So
# two query/key pairs that are the same distance apart get the SAME attention
# score, no matter where in the sequence they sit. That's relative position,
# delivered by construction, with no learned parameters.
#
# We verify it numerically: fix q and k, then sweep (m, n). Rows with equal n-m
# must show equal scores; different n-m gives different scores.
# ---------------------------------------------------------------------------
def exp_3_relative(seed=0):
    """Fix a query q and key k, rotate them at various position pairs (m, n), and
    show the attention score q_m . k_n depends only on the distance n-m."""
    _banner("LAYER 3: <R_m q, R_n k> depends ONLY on (n-m) — relative position for free")
    torch.manual_seed(seed)

    d = 8
    q, k = torch.randn(d), torch.randn(d)

    def score(m, n):
        qm = apply_rope_explicit(q.unsqueeze(0), torch.tensor([float(m)])).squeeze(0)
        kn = apply_rope_explicit(k.unsqueeze(0), torch.tensor([float(n)])).squeeze(0)
        return (qm @ kn).item()

    print("     m    n   n-m |   score q_m . k_n")
    print("  -------+--------+------------------")
    # three pairs all at distance 3 (different absolute positions), then distances 1, 5
    cases = [(0, 3), (1, 4), (7, 10), (0, 1), (5, 6), (0, 5)]
    for m, n in cases:
        print(f"   {m:4d} {n:4d} {n-m:4d} |     {score(m, n):+8.4f}")

    d3 = [score(m, n) for m, n in [(0, 3), (1, 4), (7, 10)]]
    print(f"\n  the three distance-3 rows: {[f'{s:+.4f}' for s in d3]}")
    print(f"    spread = {max(d3) - min(d3):.2e}  (identical to float precision — absolute m,n cancelled)")
    print(f"  distance 1 -> {score(0,1):+.4f}, distance 5 -> {score(0,5):+.4f}  (different distance, different score)")

    # for contrast: plain (unrotated) dot product ignores position entirely
    print(f"\n  (no RoPE: q.k = {(q @ k).item():+.4f} — same for ALL (m,n): position never enters.)")
    print("  Next (layer 4): the theta_k spectrum — why different planes rotate at different speeds.")


# ---------------------------------------------------------------------------
# LAYER 4: frequencies — theta_k = base^(-2k/d), a spectrum of speeds.
#
# All d/2 planes rotate, but at DIFFERENT frequencies. Plane k turns by theta_k per
# position, with theta_k = base^(-2k/d) (base = 10000, the standard):
#     k = 0        -> theta = 1            (fastest: a full turn every 2*pi ≈ 6.3 tokens)
#     k = d/2 - 1  -> theta ≈ 1/base       (slowest: wavelength ~ base*2*pi tokens)
# The WAVELENGTH of plane k (positions per full revolution) is 2*pi / theta_k.
#
# Why a spectrum? It's the same idea as the sinusoidal positional encoding: fast
# planes give fine, LOCAL distance resolution (adjacent tokens look very different);
# slow planes vary gently and carry LONG-RANGE position. Together the planes encode
# distance at every scale, and the slow ones are what let RoPE generalize to
# contexts longer than anything seen in training.
# ---------------------------------------------------------------------------
def exp_4_frequencies(base=10000.0):
    """Print the per-plane frequency theta_k and its wavelength (positions per full
    rotation) for a real head_size, showing the local -> global spectrum."""
    _banner(f"LAYER 4: theta_k = base^(-2k/d), base={base:g} — a local->global speed spectrum")

    d = 16                                               # head_size (8 planes)
    theta = rope_frequencies(d, base)
    print(f"  head_size d = {d}  ->  {d//2} planes\n")
    print("   plane k | theta_k = base^(-2k/d) | wavelength 2*pi/theta_k (positions/turn)")
    print("  ---------+------------------------+-----------------------------------------")
    for k in range(d // 2):
        wl = 2 * math.pi / theta[k].item()
        print(f"     {k:3d}   |      {theta[k].item():10.6f}        |      {wl:12.1f}")

    print("\n  plane 0 spins fastest (wavelength ~6.3 tokens): fine LOCAL distance.")
    print("  the last plane barely moves (huge wavelength): smooth LONG-RANGE position.")
    print("  A big base spreads the wavelengths further apart -> longer effective context")
    print("  (this is the knob people crank when extending a model to 100k+ tokens).")
    print("  Next (layer 5): compute all these rotations at once with the rotate_half trick.")


# ---------------------------------------------------------------------------
# LAYER 5: rotate_half — the cheap production formula.
#
# The layer-2 double loop is unusably slow. The vectorized identity everyone ships:
# precompute cos and sin of every angle (position m, plane k) ONCE, then
#
#     rope(x) = x * cos  +  rotate_half(x) * sin
#     rotate_half([x1, x2]) = [-x2, x1]      # x1 = first half (:d/2), x2 = second half
#
# with cos, sin shaped (T, d) by DUPLICATING the (T, d/2) plane-angles across both
# halves. Check it lines up with the layer-2 rotation, coordinate k (k < d/2):
#     (x*cos)[k]       = x[k] * cos(mθ_k)
#     (rot_half*sin)[k] = -x[k+d/2] * sin(mθ_k)
#     sum              = x[k]cos - x[k+d/2]sin    ✓  (exactly out[k] from layer 2)
# and coordinate k+d/2 gives x[k]sin + x[k+d/2]cos  ✓. Same rotation, no Python loop.
# ---------------------------------------------------------------------------
def build_rope_cache(seq_len: int, head_size: int, base: float = 10000.0,
                     offset: int = 0):
    """Precompute (cos, sin), each (seq_len, head_size), for positions
    offset..offset+seq_len-1. The (T, d/2) plane-angles are duplicated across both
    halves so they broadcast against a full (..., T, d) head tensor."""
    theta = rope_frequencies(head_size, base)            # (d/2,)
    pos = torch.arange(offset, offset + seq_len).float() # (T,)
    freqs = torch.outer(pos, theta)                      # (T, d/2) = m * theta_k
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # (T, d)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)  # (T, d)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[x1, x2] -> [-x2, x1], splitting the last dim in half."""
    d = x.size(-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (..., T, d); cos, sin: (T, d) broadcast over the leading dims."""
    return x * cos + rotate_half(x) * sin


def exp_5_rotate_half(seed=0):
    """Apply the vectorized rotate_half formula and confirm it equals the explicit
    per-plane rotation from layer 2, bit-for-bit (up to float rounding)."""
    _banner("LAYER 5: rope(x) = x*cos + rotate_half(x)*sin  ==  the explicit rotation")
    torch.manual_seed(seed)

    T, d = 6, 8
    x = torch.randn(T, d)
    pos = torch.arange(T).float()

    # explicit (layer 2) vs vectorized (this layer)
    slow = apply_rope_explicit(x, pos)
    cos, sin = build_rope_cache(T, d)
    fast = apply_rope(x, cos, sin)

    max_diff = (slow - fast).abs().max().item()
    print(f"  x: {tuple(x.shape)}, positions 0..{T-1}")
    print(f"  cos, sin tables: {tuple(cos.shape)} each (built once, reused for every batch/head)")
    print(f"  max |explicit - rotate_half| = {max_diff:.2e}")
    assert torch.allclose(slow, fast, atol=1e-5), "rotate_half does NOT match the explicit rotation!"
    print("  identical -> the loop was just a slow way to write x*cos + rotate_half(x)*sin.  ✅")

    # it broadcasts cleanly over (B, nh, T, d) — the shape attention actually uses
    B, nh = 2, 4
    xb = torch.randn(B, nh, T, d)
    yb = apply_rope(xb, cos, sin)
    print(f"\n  broadcast check: rope on {tuple(xb.shape)} (B, nh, T, d) -> {tuple(yb.shape)}")
    print("  (cos/sin are (T,d); they broadcast across batch and heads for free.)")
    print("  Next (layer 6): wire this into attention on the real (B, nh, T, hs) head layout.")


# ---------------------------------------------------------------------------
# LAYER 6: RoPESelfAttention — drop RoPE into the real attention.
#
# Same as our CausalSelfAttention (attention.py), with ONE change: after projecting
# and reshaping to (B, nh, T, hs), rotate q and k with RoPE before the scores.
# v is NOT rotated — position should shape WHO attends to whom (the q.k scores), not
# the payload being moved. Three things that must hold, all verified below:
#   * still causal (a future token can't change an earlier output).
#   * KV-cache correct: rotate the NEW q,k by their ABSOLUTE positions (offset by the
#     cached length). Cached keys were already rotated when they were new, so we
#     never touch them again — RoPE composes with the cache precisely because the
#     rotation is a function of absolute position.
#   * NO length cap: the causal mask is built on the fly (no block_size-sized table)
#     and cos/sin are computed for whatever T we get — so it runs BEYOND block_size,
#     which the learned absolute table (layer 1) simply cannot do.
# ---------------------------------------------------------------------------
class RoPESelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, base: float = 10000.0):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert (config.n_embd // config.n_head) % 2 == 0, "head_size must be even for RoPE"
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head
        self.base = base

        # identical projections to CausalSelfAttention — RoPE adds NO parameters
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.proj.RESIDUAL_PROJ = True
        # NOTE: no position_embedding, and deliberately no fixed-size tril buffer.

    def forward(self, x: torch.Tensor, kv_cache=None):
        B, T, C = x.shape
        nh, hs = self.n_head, self.head_size

        q, k, v = self.qkv(x).split(self.n_embd, dim=-1)
        q = q.view(B, T, nh, hs).transpose(1, 2)         # (B, nh, T, hs)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)

        # absolute position of the first NEW token = how many keys are already cached
        past_len = kv_cache[0].size(2) if kv_cache is not None else 0

        # rotate the NEW q, k by their absolute positions (offset by past_len). The
        # cached keys were rotated when THEY were new, so we leave them untouched.
        cos, sin = build_rope_cache(T, hs, self.base, offset=past_len)
        cos, sin = cos.to(q), sin.to(q)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)                       # v is NOT rotated

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = torch.cat([past_k, k], dim=2)             # (B, nh, T_past + T, hs)
            v = torch.cat([past_v, v], dim=2)
        new_cache = (k, v)

        T_k = k.size(2)
        T_past = T_k - T

        wei = q @ k.transpose(-2, -1) * hs ** -0.5        # (B, nh, T, T_k)
        # offset-aware causal mask, built on the fly (no block_size cap): query row i
        # is at absolute position T_past+i and may attend keys j <= T_past+i.
        allowed = torch.ones(T, T_k, device=x.device).tril(diagonal=T_past)
        wei = wei.masked_fill(allowed == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v                                     # (B, nh, T, hs)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)
        return out, new_cache


def exp_6_attention(seed=0):
    """Run RoPESelfAttention and verify: causal, KV-cache matches the full forward,
    zero position params, and it runs BEYOND block_size (extrapolation)."""
    _banner("LAYER 6: RoPESelfAttention — causal, KV-cache-correct, and no length cap")
    torch.manual_seed(seed)

    cfg = GPTConfig(vocab_size=65, block_size=8, n_embd=32, n_head=4, dropout=0.0)
    attn = RoPESelfAttention(cfg)
    attn.eval()

    B, T = 2, cfg.block_size
    hs = cfg.n_embd // cfg.n_head
    x = torch.randn(B, T, cfg.n_embd)

    out, _ = attn(x)
    print(f"  head_size = {hs} (even, ok). out shape = {tuple(out.shape)}")

    # (1) causality: perturbing the LAST position must not change earlier outputs
    x2 = x.clone()
    x2[:, -1, :] = torch.randn(B, cfg.n_embd)
    o1, _ = attn(x)
    o2, _ = attn(x2)
    causal = torch.allclose(o1[:, :-1], o2[:, :-1], atol=1e-6)
    print(f"  causal? {causal}  (changing token {T-1} left tokens 0..{T-2} untouched)")

    # (2) KV-cache: feed one token at a time (past_len grows each step) == full forward.
    #     This is the real RoPE+cache test: the offset must rotate each new token at
    #     its absolute position, and cached keys must keep their original rotation.
    full, _ = attn(x)
    cache, steps = None, []
    for t in range(T):
        ot, cache = attn(x[:, t:t + 1, :], cache)
        steps.append(ot)
    inc = torch.cat(steps, dim=1)
    cache_ok = torch.allclose(full, inc, atol=1e-5)
    print(f"  KV-cache incremental == full forward? {cache_ok}  (absolute-position offset handled)")

    # (3) zero position params vs the learned table it replaces
    n_attn = sum(p.numel() for p in attn.parameters())
    n_csa = sum(p.numel() for p in CausalSelfAttention(cfg).parameters())
    saved = cfg.block_size * cfg.n_embd
    print(f"\n  params: RoPE attn {n_attn} == plain attn {n_csa}  (RoPE itself adds 0)")
    print(f"  and the GPT can DROP its position_embedding entirely: "
          f"-{saved:,} params ({cfg.block_size}x{cfg.n_embd} table).")

    # (4) extrapolation: run at T = block_size + 4. The absolute table (layer 1)
    #     has no rows past block_size; RoPE builds cos/sin + the mask for any T.
    T_long = cfg.block_size + 4
    x_long = torch.randn(B, T_long, cfg.n_embd)
    out_long, _ = attn(x_long)
    xl2 = x_long.clone()
    xl2[:, -1, :] = torch.randn(B, cfg.n_embd)
    long_causal = torch.allclose(attn(x_long)[0][:, :-1], attn(xl2)[0][:, :-1], atol=1e-6)
    print(f"\n  ran on T = {T_long} > block_size = {cfg.block_size}: out {tuple(out_long.shape)}, "
          f"still causal? {long_causal}")
    print("  the learned absolute table CANNOT do this (IndexError past block_size, layer 1).")

    assert causal and cache_ok and long_causal, "a RoPE attention invariant failed!"
    print("\n  That's RoPE end-to-end: rotate q,k by position so scores depend only on")
    print("  distance (n-m), computed as x*cos + rotate_half(x)*sin, dropped into")
    print("  attention with 0 params, KV-cache-correct, and no length cap.  ✅")
    print("  Next (layer 7): it RUNS past the trained length, but does it WORK? The")
    print("  degradation problem + the fixes (PI / NTK-aware / YaRN) that give 128k context.")


# ===========================================================================
# CONTEXT EXTENSION (layers 7-10): "no length cap" (layer 6) means RoPE RUNS at
# any length — but a model TRAINED at length L still degrades past L, because
# beyond L the rotations land at angle-combinations it never saw. Layers 7-10 are
# why, and the three standard fixes. These are pure position/frequency math (no
# training), so we reason about angles, not sample quality. Maps to advanced_
# roadmap.md item 2b (Position Interpolation, NTK-aware, YaRN).
# ===========================================================================


# ---------------------------------------------------------------------------
# LAYER 7: why vanilla RoPE degrades past L — the SLOW planes leave their arc.
#
# During training at length L, a query and key are at most L-1 apart, so plane k
# only ever rotates through relative angles up to ~L*theta_k. The model's weights
# are tuned ONLY on that arc. What arc is that, per plane?
#   * FAST planes (short wavelength): L*theta_k >> 2*pi, so they sweep the FULL
#     circle many times during training -> every angle was seen -> they extrapolate
#     FINE to any distance.
#   * SLOW planes (wavelength > L): L*theta_k < 2*pi, so they only ever swept a
#     small ARC of their circle. At distance > L they rotate PAST that arc into
#     angles the model never saw -> this is where extrapolation breaks.
# So the culprit is precisely the low-frequency / long-wavelength planes. The
# clean criterion (YaRN's): a plane is "safe" iff its wavelength <= L (it completes
# a full turn within the trained window). That split drives every fix below.
# ---------------------------------------------------------------------------
def exp_7_extrapolation_problem(train_len=64, d=64):
    """For a model trained at context length L, show per plane how much of its
    rotation circle was seen in training — fast planes see it all (extrapolate
    fine), slow planes see only a sliver (break past L)."""
    _banner(f"LAYER 7: past the trained length L={train_len}, the SLOW planes hit unseen angles")
    theta = rope_frequencies(d)                       # (d/2,)
    twopi = 2 * math.pi

    print(f"  head_size d = {d} ({d//2} planes), trained context length L = {train_len}\n")
    print("  plane |   theta_k    | wavelength | turns in L (L/wl) | arc seen | extrapolates?")
    print("  ------+--------------+------------+-------------------+----------+--------------")
    for k in [0, d // 8, d // 4, 3 * d // 8, d // 2 - 1]:
        th = theta[k].item()
        wl = twopi / th
        turns = train_len / wl                        # = L*theta/2pi
        arc = min(train_len * th, twopi)              # radians actually swept (cap 1 turn)
        ok = "yes (all seen)" if wl <= train_len else "NO (unseen past L)"
        print(f"   {k:3d}  | {th:.4e} | {wl:10.1f} |     {turns:8.3f}      | {arc:5.2f} ra | {ok}")

    safe = int((theta * train_len / twopi >= 1).sum())   # wavelength <= L
    print(f"\n  {safe}/{d//2} planes complete a full turn within L (wavelength <= L) -> safe to")
    print(f"  extrapolate. The other {d//2 - safe} (slow, long-wavelength) planes swept only a")
    print("  partial arc in training, so a longer context rotates them into NOVEL angles the")
    print("  model was never tuned on -> that's the degradation. The fixes all target these")
    print("  slow planes. Next (layer 8): Position Interpolation — squeeze positions back in.")


# ---------------------------------------------------------------------------
# LAYER 8: Position Interpolation (PI) — squeeze every position into the arc.
#
# Simplest fix (Chen et al. / Meta, used by Llama-2 long-context). To extend L -> L'
# scale EVERY position by s = L/L' before rotating: position m becomes m*s. Then the
# largest relative angle at the new max distance L' is L'*s*theta_k = L*theta_k =
# exactly the trained maximum -> every distance now maps to an angle the model saw.
# Needs a short fine-tune, but far less than training long from scratch.
#
# The cost is that the squeeze is UNIFORM across planes: the per-position angle step
# shrinks by s for EVERY plane, including the fast ones that were already fine. So
# adjacent tokens now differ by a smaller angle everywhere -> local/high-frequency
# resolution is blurred. Layers 9-10 fix exactly this over-squeezing.
# ---------------------------------------------------------------------------
def exp_8_position_interpolation(train_len=64, ext_len=256, d=64):
    """Show PI: scale positions by s=L/L' so the max angle at L' equals the trained
    max (back in range) — but the per-step angle shrinks uniformly (the downside)."""
    _banner(f"LAYER 8: Position Interpolation — scale positions by s=L/L' to fit L'={ext_len}")
    theta = rope_frequencies(d)
    s = train_len / ext_len                           # < 1, the PI position scale
    print(f"  extend L={train_len} -> L'={ext_len} (x{ext_len//train_len}).  PI scale s = L/L' = {s:.3f}\n")

    print("  plane | step angle (orig) | step angle (PI) | max ang @L' no-PI | @L' with PI | trained max")
    print("  ------+-------------------+-----------------+-------------------+-------------+------------")
    for k in [0, d // 4, d // 2 - 1]:
        th = theta[k].item()
        trained_max = train_len * th
        print(f"   {k:3d}  |     {th:8.4f}      |    {s*th:8.4f}     |     {ext_len*th:9.3f}     |  "
              f"{ext_len*s*th:8.4f}   |  {trained_max:8.4f}")

    print(f"\n  With PI, max angle at L' == trained max ({train_len}*theta) for EVERY plane -> in range.")
    print(f"  But the per-step angle shrank x{s:.3f} UNIFORMLY — even plane 0 (fast, already safe)")
    print("  now resolves adjacent tokens {:.0f}x more coarsely. Over-squeezing the fast planes".format(1/s))
    print("  is PI's weakness. Next (layer 9): NTK-aware — stretch slow planes, spare fast ones.")


# ---------------------------------------------------------------------------
# LAYER 9: NTK-aware scaling — interpolate UNEVENLY across frequencies.
#
# Insight (bloc97): PI over-squeezes the fast planes. Instead, change the single
# knob RoPE already has — the base — so the interpolation is spread across planes:
# fast (high-freq) planes barely move, slow (low-freq) planes stretch the most.
#     base' = base * s_ext^(d/(d-2))     with s_ext = L'/L  (> 1)
# Since theta_k = base^(-2k/d), scaling the base multiplies theta_k by
# s_ext^(-2k/(d-2)): plane 0 -> x1 (untouched), the slowest plane -> x(1/s_ext)
# (wavelength x s_ext, so it now spans the extended window). This is a ONE-LINE
# change (just pass a bigger rope_base) and needs NO fine-tune — it's why it caught
# on (CodeLlama, NousResearch). Trade-off: less precise than a tuned YaRN, but free.
# ---------------------------------------------------------------------------
def exp_9_ntk_aware(train_len=64, ext_len=256, d=64, base=10000.0):
    """Show NTK-aware: bump the base so wavelengths stretch NON-uniformly — fast
    planes ~unchanged (local resolution kept), slow planes x s_ext (cover L')."""
    _banner(f"LAYER 9: NTK-aware — raise the base so slow planes stretch, fast planes don't")
    s_ext = ext_len / train_len                       # > 1
    base2 = base * s_ext ** (d / (d - 2))
    theta1 = rope_frequencies(d, base)
    theta2 = rope_frequencies(d, base2)
    twopi = 2 * math.pi
    print(f"  extend x{s_ext:.0f}:  base {base:.0f} -> base' = base*s^(d/(d-2)) = {base2:.0f}\n")

    print("  plane | wavelength (base) | wavelength (NTK) | ratio | PI would use (uniform)")
    print("  ------+-------------------+------------------+-------+-----------------------")
    for k in [0, d // 4, d // 2 - 1]:
        wl1 = twopi / theta1[k].item()
        wl2 = twopi / theta2[k].item()
        print(f"   {k:3d}  |     {wl1:10.1f}    |    {wl2:10.1f}    | {wl2/wl1:4.2f}x |     {s_ext:4.2f}x")

    print(f"\n  PI multiplies EVERY wavelength by s_ext={s_ext:.1f} (uniform, blurs fast planes).")
    print("  NTK ramps the ratio from ~1.0x (plane 0, local detail preserved) up to ~s_ext")
    print("  (slowest plane, now covers L'). Same extension, fast-plane resolution kept.")
    print("  Next (layer 10): YaRN — make that fast/slow split explicit + fix the softmax temp.")


# ---------------------------------------------------------------------------
# LAYER 10: YaRN — NTK-by-parts + an attention-temperature correction.
#
# YaRN (Peng et al.; Llama-3.1 / Qwen2 long-context) makes layer-7's safe/unsafe
# split EXPLICIT instead of NTK's smooth curve:
#   * a plane's "rotations within L" is r_k = L / wavelength_k = L*theta_k/2pi.
#   * r_k >= beta  (many turns, high-freq): EXTRAPOLATE — leave theta_k alone (it's
#     safe, layer 7). r_k <= alpha (< 1 turn, low-freq): INTERPOLATE — full PI
#     (theta_k / s_ext). Linear RAMP in between. (alpha, beta are tuned thresholds.)
# Plus one thing PI/NTK ignore: packing more positions into the same arc makes the
# q.k scores (and thus the softmax) more peaked, shifting attention entropy. YaRN
# corrects it with an attention TEMPERATURE t = 0.1*ln(s_ext)+1: scale the scores by
# 1/t (equivalently scale q,k by 1/sqrt(t)). That entropy fix is YaRN's real edge.
# ---------------------------------------------------------------------------
def exp_10_yarn(train_len=64, ext_len=256, d=64, alpha=1.0, beta=8.0):
    """Show YaRN's per-plane decision (extrapolate high-freq / interpolate low-freq /
    ramp between) via r_k = L/wavelength, plus the attention-temperature factor."""
    _banner("LAYER 10: YaRN — extrapolate fast planes, interpolate slow ones, fix the temp")
    s_ext = ext_len / train_len
    theta = rope_frequencies(d)
    twopi = 2 * math.pi
    rotations = train_len * theta / twopi             # r_k = L/wavelength per plane
    gamma = ((rotations - alpha) / (beta - alpha)).clamp(0, 1)  # 1=extrapolate, 0=interpolate
    divisor = s_ext - gamma * (s_ext - 1)             # 1 (keep) .. s_ext (full PI)
    print(f"  extend x{s_ext:.0f}, ramp thresholds alpha={alpha:g} (interp below) beta={beta:g} (extrap above)\n")

    print("  plane | turns in L (r_k) | gamma | freq divisor | action")
    print("  ------+------------------+-------+--------------+----------------------")
    for k in [0, d // 8, d // 4, 3 * d // 8, d // 2 - 1]:
        r = rotations[k].item()
        g = gamma[k].item()
        div = divisor[k].item()                       # theta_yarn = theta / div
        act = ("extrapolate (keep)" if g > 0.99 else
               "interpolate (full PI)" if g < 0.01 else "ramp (blend)")
        print(f"   {k:3d}  |     {r:9.3f}    | {g:.2f}  |    {div:6.3f}    | {act}")

    n_ex = int((gamma > 0.99).sum())
    n_in = int((gamma < 0.01).sum())
    temp = 0.1 * math.log(s_ext) + 1
    print(f"\n  {n_ex} fast planes extrapolated (untouched), {n_in} slow planes interpolated, "
          f"{d//2 - n_ex - n_in} ramped.")
    print(f"  attention temperature t = 0.1*ln(s_ext)+1 = {temp:.4f}: scale scores by 1/t "
          f"(q,k by 1/sqrt(t)={1/math.sqrt(temp):.4f})")
    print("  to counter the sharper softmax from denser positions — the piece PI/NTK skip.")
    print("\n  Summary of the family: PI = uniform squeeze (simple, blurs local); NTK = uneven")
    print("  squeeze via base (free, no fine-tune); YaRN = explicit fast/slow split + temp fix")
    print("  (best quality, Llama-3.1/Qwen2). All three only rescale the SAME rope_base/theta")
    print("  knob you built — nothing else in attention changes.")


def run_experiments():
    # exp_1_the_problem()
    # exp_2_rotate_pairs()
    # exp_3_relative()
    # exp_4_frequencies()
    # exp_5_rotate_half()
    # exp_6_attention()
    exp_7_extrapolation_problem()
    exp_8_position_interpolation()
    exp_9_ntk_aware()
    exp_10_yarn()


@contextlib.contextmanager
def _tee(path):
    """Print to BOTH the terminal and `path` (long runs survive scrollback)."""
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)

        def flush(self):
            for st in self.streams:
                st.flush()

    with open(path, "w") as f:
        with contextlib.redirect_stdout(_Tee(sys.stdout, f)):
            yield
    print(f"(output also written to {path})", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", metavar="FILE", help="also write all output to FILE")
    args = parser.parse_args()

    if args.out:
        with _tee(args.out):
            run_experiments()
    else:
        run_experiments()
