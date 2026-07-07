"""
WALKTHROUGH: the denoiser and its training objective, one layer at a time.

forward_process.py built the DESTRUCTION direction and handed us a supervised target:
given a noised image x_t and its timestep t, the network must predict the noise eps that
was added (Layer 4 there). This walkthrough builds the thing that DOES that prediction —
a small network eps_theta(x_t, t) ~ eps — and trains it with the "simple" DDPM loss

    L = E_{x0, t, eps} || eps_theta(x_t, t) - eps ||^2 ,   x_t = √ab·x0 + √(1-ab)·eps

Once this net predicts eps well, sampling (item 3) just runs it in reverse: start from
N(0,I) and repeatedly subtract predicted noise until a digit appears. So this file is the
bridge from "we understand the math" to "we have a trained model" — the last piece before
the first generated image.

A denoiser needs one ingredient the forward process never did: a way to tell the network
WHICH noise level it's looking at. x_t alone is ambiguous (a lightly-noised 3 and a heavily-
noised 8 can look similar); the same net must behave very differently at t=10 vs t=900. So
t is a first-class input, and Layer 1 is how we feed it.

Layers (run each `exp_*`, watch the output, then say "next"):
  1. timestep embedding — how to feed the integer t to the net, in small bites:
       1a. WHY t must be an input at all — x_t alone doesn't determine the noise level,
           so the eps target is a function of (x_t, t), not x_t.                       (here)
       1b. the naive encodings and why they fail — raw scalar (a net struggles to build
           sharp functions of one number) and one-hot (no sharing between nearby levels).
       1c. the idea — encode t at many frequencies at once (a set of clock hands): a fast
           hand resolves fine differences, a slow hand gives coarse position.
       1d. build the sinusoidal embedding — geometric frequencies, sin & cos; read the
           per-channel periods and see the multi-scale code.
       1e. sin AND cos together — a shift t -> t+Δ becomes a rotation (the RoPE connection).
       1f. two properties — every entry in [-1,1], and constant norm √half for all t.
       1g. the payoff — similarity depends only on the gap Δ=t-t' (a smooth kernel over t).
  2. the denoiser network — a small MLP that learns ε_theta(x_t, t), in three bites:
       2a. the plain MLP + I/O contract — flatten x_t (784) -> hidden -> eps (784); why NO
           activation on the output (eps ~ N(0,1) is unbounded). No t yet, so it can't work.
       2b. fusing t (the heart) — project the sinusoidal embedding and ADD it into the hidden
           activations (a t-dependent bias); the "inject a conditioning signal" move reused
           later as AdaLN (DiT) and CFG.
       2c. assemble + sanity — wrap as an nn.Module, count params, forward a real batch shape,
           and verify the output actually DEPENDS on t (perturb t, output moves) — closing 1a.
  3. the DDPM loss + one training step — reuse Layer-4's target: draw t, eps, form x_t,
     predict eps_hat, MSE, backprop. Overfit ONE batch to prove the objective + grads wire up.
  4. the data pipeline — a tiny Dataset/DataLoader over the MNIST npz (no torchvision),
     images normalized to [-1,1] the way the closed form assumes.
  5. the training loop — train on MNIST for real; watch the loss drop.
  6. SEE it — on HELD-OUT noised digits, measure eps-prediction error and reconstruct
     x0_hat from the predicted eps; watch it actually denoise. (Generation from pure
     noise is the next walkthrough, sampling.py.)

Reference for the real (torch) versions: diffusion/model.py (the network) and
diffusion/train.py (the loop). This is the SCRATCH file you learn from, then rebuild clean.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))   # diffusion/walkthroughs/


def _banner(*lines):
    print("=" * 70)
    for line in lines:
        print(line)
    print("=" * 70)


# ---------------------------------------------------------------------------
# LAYER 1a: WHY t must be an input at all.
#
# Before designing HOW to feed t, prove we NEED it. The denoiser is one shared network we
# call at every noise level; its job is to look at x_t and output the noise eps to remove.
# The trap: x_t ALONE does not determine eps. The same pixels can be a lightly-noised
# version of one image OR a heavily-noised version of a completely different image — two
# different (x0, t) stories, two different correct eps. So the target is a function of
# (x_t, t), not of x_t; without t the mapping x_t -> eps is one-to-many (ill-posed) and the
# best a net could do is blur across all the noise levels that could have produced x_t.
# This bite makes that ambiguity concrete; 1b-1g then build the encoding of t.
# ---------------------------------------------------------------------------
def exp_1a_why_t_is_an_input():
    """Take ONE noised vector x_t and show it is a valid lightly-noised version of image A
    (small t) AND a valid heavily-noised version of a different image B (large t) — each
    requiring a totally different eps. Same input, two correct answers => eps is a function
    of (x_t, t), not x_t alone, so t MUST be fed to the network."""
    _banner("LAYER 1a: why t must be an input — x_t alone doesn't determine the noise")

    from forward_process import make_linear_schedule          # reuse the schedule from item 1
    T = 1000
    _, _, alpha_bars = make_linear_schedule(T=T)
    g = torch.Generator().manual_seed(0)

    D = 8                                                      # a tiny 8-"pixel" image
    # Story A: clean image x0_A, lightly noised at t_A -> gives x_t.
    x0_A = torch.randn(D, generator=g)
    t_A = 100
    eps_A = torch.randn(D, generator=g)                       # the noise A actually added
    ab_A = alpha_bars[t_A]
    x_t = torch.sqrt(ab_A) * x0_A + torch.sqrt(1 - ab_A) * eps_A

    # Story B: explain that SAME x_t as a HEAVILY noised version of a DIFFERENT image x0_B at
    # t_B. The closed form pins the exact noise that story needs (Layer 4a rearranged):
    #   eps_B = (x_t - √ab_B·x0_B) / √(1-ab_B).  For ANY x0_B this reproduces x_t exactly.
    x0_B = torch.randn(D, generator=g)
    t_B = 800
    ab_B = alpha_bars[t_B]
    eps_B = (x_t - torch.sqrt(ab_B) * x0_B) / torch.sqrt(1 - ab_B)

    # both stories rebuild the identical x_t; confirm, then compare the eps they require.
    x_t_from_A = torch.sqrt(ab_A) * x0_A + torch.sqrt(1 - ab_A) * eps_A
    x_t_from_B = torch.sqrt(ab_B) * x0_B + torch.sqrt(1 - ab_B) * eps_B
    cos = torch.nn.functional.cosine_similarity(eps_A, eps_B, dim=0).item()

    print(f"  one vector x_t explained two ways:  A) t={t_A}, light noise   B) t={t_B}, heavy noise\n")
    print(f"  {'i':>2} | {'x_t':>8} | {'eps_A (t=100)':>13} | {'eps_B (t=800)':>13}")
    print(f"  {'-'*2}-+-{'-'*8}-+-{'-'*13}-+-{'-'*13}")
    for i in range(D):
        print(f"  {i:>2} | {x_t[i]:>+8.4f} | {eps_A[i]:>+13.4f} | {eps_B[i]:>+13.4f}")
    print()
    print(f"  both stories reproduce the SAME x_t:  max|A-x_t|={ (x_t_from_A-x_t).abs().max():.1e}"
          f"   max|B-x_t|={ (x_t_from_B-x_t).abs().max():.1e}")
    print(f"  yet the two correct eps are unrelated:  ||eps_A - eps_B|| = {(eps_A-eps_B).norm():.3f}"
          f"   cos(eps_A, eps_B) = {cos:+.3f}\n")
    print("  Read it: the identical input x_t has TWO different correct noise targets — which")
    print("  one is right depends entirely on t. So eps is a function of (x_t, t), not x_t")
    print("  alone; feed only x_t and the task is one-to-many, and the net can at best predict")
    print("  a mush averaged over every t that could have made x_t. THAT is why t is a")
    print("  first-class input. Next (1b): the two naive ways to feed t, and why they fail.")


# ---------------------------------------------------------------------------
# LAYER 1b: the two naive ways to feed t, and why they fail.
#
# 1a said t must be an input. The obvious two encodings both fall short:
#
#   RAW SCALAR t (or t/T). It's a single number, so the first (linear) layer sees t through
#   ONE direction: every hidden unit's t-dependence is just w_i·t, an AFFINE function of t.
#   The net must then manufacture any richer, level-specific behaviour through depth — and
#   nets have a SPECTRAL BIAS (they learn low-frequency functions of an input far more easily
#   than high-frequency ones). So a behaviour that must change several times across 0..999 is
#   exactly what a raw-scalar input makes hard. We demo the linear-readout limit: an affine
#   function of t can't fit a wiggly target (whereas sin/cos features, 1d, make it trivial —
#   the target literally IS a feature).
#
#   ONE-HOT over T. A T-dim vector, a 1 at position t. Now every level is its OWN feature, so
#   the net CAN fit each independently — but that's the problem: onehot(t)·onehot(t') = 0 for
#   t != t', so t=500 is as unrelated to 501 as to 5. Nothing learned at one level transfers
#   to its neighbours (no sharing), and it costs T=1000 extra input dims.
#
# The wishlist this leaves: compact (not 1000-d), high-resolution (distinguish every level),
# and with nearby-t SIMILARITY so behaviour shares across neighbouring levels. 1c-1g build it.
# ---------------------------------------------------------------------------
def exp_1b_naive_encodings():
    """Show both naive t-encodings failing. RAW SCALAR: fit an affine function of t/T (all a
    linear layer can make from one scalar) to a wiggly target and watch R^2 collapse. ONE-HOT:
    the Gram matrix of onehot(t) is the identity, so adjacent levels are orthogonal — zero
    sharing, plus T=1000 dims. Motivates the sinusoidal encoding (1c-1g)."""
    _banner("LAYER 1b: naive encodings — raw scalar (no resolution) and one-hot (no sharing)")

    T = 1000

    # RAW SCALAR: the first linear layer turns one scalar t into affine functions a·(t/T)+b
    # only. Try to fit a target that must change several times across the schedule — a stand-in
    # for "level-specific behaviour" — as a·(t/T)+b via least squares. It can't.
    t = torch.arange(0, T, 20, dtype=torch.float32)          # 50 sample levels
    tn = t / T                                               # normalized to [0,1] (fixes scale)
    target = torch.sin(2 * torch.pi * 3 * tn)                # a wiggly function of t (3 cycles)
    A = torch.stack([tn, torch.ones_like(tn)], dim=1)        # design matrix [t/T, 1]
    coef = torch.linalg.lstsq(A, target).solution           # best affine fit
    pred = A @ coef
    r2 = 1 - ((target - pred) ** 2).sum() / ((target - target.mean()) ** 2).sum()
    print("  RAW SCALAR t: a linear layer can only build AFFINE functions a·(t/T)+b from one")
    print("  number. Fit that to a wiggly target (a stand-in for level-specific behaviour):")
    print(f"    best affine fit a·(t/T)+b :  a={coef[0]:+.3f}  b={coef[1]:+.3f}   ->   R^2 = {r2:.3f}")
    print("    -> R^2 ~ 0: an affine function is hopeless at a target that changes several times")
    print("       across t. (Raw scalar also has a scale gap: t in 0..999 vs pixels in [-1,1];")
    print("       t/T fixes the SCALE but not this EXPRESSIVITY limit.)\n")

    # ONE-HOT: high resolution but every level orthogonal to every other. Show the Gram matrix
    # of onehots at five ADJACENT levels is the identity — no similarity to share behaviour.
    ts = torch.tensor([498, 499, 500, 501, 502])
    E = torch.zeros(len(ts), T)
    E[torch.arange(len(ts)), ts] = 1.0                       # one-hot rows
    gram = E @ E.T                                           # pairwise dot products
    print(f"  ONE-HOT over T={T}: {T} dims, and the Gram matrix of five ADJACENT levels")
    print(f"  {tuple(ts.tolist())} is the identity — every level orthogonal to every other:")
    print("        " + " ".join(f"{tv:>5}" for tv in ts.tolist()))
    for i, tv in enumerate(ts.tolist()):
        print(f"  {tv:>5} " + " ".join(f"{gram[i, j]:>5.0f}" for j in range(len(ts))))
    print("    -> onehot(500)·onehot(501) = 0: t=500 is as unrelated to 501 as to 5. Nothing")
    print("       learned at one level transfers to its neighbours (no sharing), and it burns")
    print(f"       {T} input dims.\n")
    print("  Wishlist for a good encoding: COMPACT (not 1000-d), HIGH-RESOLUTION (tell every")
    print("  level apart), and SMOOTH (nearby t similar, so behaviour shares across neighbours).")
    print("  Next (1c): the idea that gets all three — encode t at many frequencies at once.")


# ---------------------------------------------------------------------------
# LAYER 1c: the idea — encode t at many frequencies at once (a set of clock hands).
#
# 1b's wishlist: compact, high-resolution, AND smooth. The trick is to describe t not by
# its value but by WHERE IT SITS in several oscillations of different speeds — like reading
# a set of clock hands, or the digits of a place-value number.
#
# Represent one "hand" of frequency f as the 2D point (sin(t·f), cos(t·f)) — t·f is its
# angle, so the point rides a circle as t advances. One hand alone is not enough:
#   - a FAST hand (short period) resolves FINE differences between nearby t, but WRAPS: after
#     one period it's back where it started, so far-apart t collide (aliasing).
#   - a SLOW hand (period >= the whole range) never wraps, so it gives an unambiguous COARSE
#     position, but adjacent t barely differ — poor resolution, fragile to any noise.
# Put a fast and a slow hand together and the ambiguity dies: the slow hand says which
# "period" you're in, the fast hand says where within it — exactly like hour+minute hands,
# or the high/low digits of a number. Use a WHOLE LADDER of frequencies (1d) and you cover
# every scale at once: compact (a handful of hands), high-res (the fast ones), and smooth
# (each hand moves continuously in t). This bite shows one hand failing and two fixing it.
# ---------------------------------------------------------------------------
def exp_1c_many_frequencies():
    """Show WHY many frequencies: one hand (sin,cos at freq f) can't encode t alone — a fast
    hand ALIASES (far t collide) and a slow hand can't RESOLVE (adjacent t merge) — but a
    fast+slow pair pins t down uniquely. Counts collisions across all t to make it concrete.
    Sets up 1d, which stacks a whole ladder of frequencies."""
    _banner("LAYER 1c: the idea — many frequencies at once (fast hand = fine, slow = coarse)")

    def hand(t, period):                                     # one clock hand: a point on a circle
        a = t.float() * (2 * torch.pi / period)
        return torch.stack([torch.sin(a), torch.cos(a)], dim=-1)

    # eyeball the two failure modes on four timesteps.
    ts = torch.tensor([10, 35, 500, 510])
    fast, slow = hand(ts, period=25), hand(ts, period=2000)  # fast wraps every 25; slow never wraps
    print("  one hand = (sin(t·f), cos(t·f)), a point on a circle. Two failure modes:\n")
    print(f"  {'t':>4} | {'FAST hand (period 25)':>24} | {'SLOW hand (period 2000)':>24}")
    print(f"  {'-'*4}-+-{'-'*24}-+-{'-'*24}")
    for i, tv in enumerate(ts.tolist()):
        f_s = f"({fast[i,0]:+.3f}, {fast[i,1]:+.3f})"
        s_s = f"({slow[i,0]:+.3f}, {slow[i,1]:+.3f})"
        print(f"  {tv:>4} | {f_s:>24} | {s_s:>24}")
    print("    FAST: t=10 and t=35 (gap 25 = one period) give the SAME point -> aliasing.")
    print("    SLOW: t=500 and t=510 are nearly identical -> can't resolve nearby t.")
    print("    So neither hand alone works; but note FAST tells 500 from 510, SLOW tells 10")
    print("    from 35 — they fail on DIFFERENT pairs, so together they cover both.\n")

    # make it quantitative over ALL t: count how many timesteps have a "twin" (another t with
    # an almost-identical code). Fast-only aliases badly; fast+slow has zero collisions.
    allt = torch.arange(1000)
    fast_all = hand(allt, 25)                                # (1000, 2)
    both_all = torch.cat([hand(allt, 25), hand(allt, 2000)], dim=1)  # (1000, 4): fast + slow

    def collisions(codes, eps=1e-2):
        d = torch.cdist(codes, codes)
        d.fill_diagonal_(float("inf"))
        nn = d.min(dim=1).values                             # distance to nearest OTHER t
        return int((nn < eps).sum()), nn.min().item()

    n_fast, sep_fast = collisions(fast_all)
    n_both, sep_both = collisions(both_all)
    print("  over all t=0..999, how many have a near-identical twin (aliased)? and the")
    print("  closest any two codes get (bigger = more robustly separated):")
    print(f"    FAST only     : {n_fast:>4} / 1000 aliased,  min separation {sep_fast:.4f}")
    print(f"    FAST + SLOW   : {n_both:>4} / 1000 aliased,  min separation {sep_both:.4f}")
    print("    -> one fast hand aliases nearly everything; adding a slow hand drives collisions")
    print("       to zero and separates every level. Two scales already beat both naive")
    print("       encodings — compact, high-res (fast), and smooth (both move continuously).\n")
    print("  Next (1d): stack a geometric LADDER of frequencies (fast -> slow) into one vector")
    print("  — the actual sinusoidal embedding — so every scale from ~1 step to the whole")
    print("  schedule is covered at once.")


# ---------------------------------------------------------------------------
# LAYER 1d: build the sinusoidal embedding — a geometric ladder of frequencies.
#
# 1c showed two hands beat one. Now stack a whole LADDER of them into a single vector. Pick
# half = dim/2 frequencies spaced GEOMETRICALLY (each a fixed ratio below the last) so a few
# hands span every scale from "cycles in a handful of steps" to "barely turns across the
# whole schedule":
#
#     half = dim/2 ,   f_k = 10000^(-k/half)   for k = 0..half-1     (f_0=1 down to ~1/10000)
#     emb(t) = [ sin(t·f_0), …, sin(t·f_{half-1}),  cos(t·f_0), …, cos(t·f_{half-1}) ]   (dim,)
#
# Geometric (not linear) spacing is the point: it covers many ORDERS OF MAGNITUDE of scale
# with few hands (like octaves on a piano), where linear spacing would bunch every hand at
# nearly the same speed. 10000 is just the conventional slowest period. This bite BUILDS the
# vector and READS the ladder — the fast channels swing wildly across t, the slow ones crawl.
# (The three properties this buys — bounded, constant-norm, Δ-only similarity — are 1e-1g.)
# ---------------------------------------------------------------------------
def timestep_embedding(t, dim, max_period=10000.0):
    """Sinusoidal embedding of integer timesteps. t: (B,) long/int. dim: even. Returns
    (B, dim) = [sin(t·f_0..f_{H-1}), cos(t·f_0..f_{H-1})], H=dim//2, f_k geometric 1->1/max_period."""
    assert dim % 2 == 0, "use an even embedding dim"
    half = dim // 2
    # geometric frequencies: exp(-ln(P)·k/half) = P^(-k/half), from f_0=1 down to ~1/P.
    # (built on t's device so this works under GPU training too.)
    k = torch.arange(half, dtype=torch.float32, device=t.device)
    freqs = torch.exp(-math.log(max_period) * k / half)
    args = t[:, None].float() * freqs[None, :]              # (B, half): t scaled by each freq
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)   # (B, dim)


def exp_1d_build_embedding():
    """Build the sinusoidal embedding and READ the frequency ladder: geometric f_k (constant
    ratio between hands) spanning periods from ~6 steps to ~35000, then watch a fast, a mid,
    and a slow channel across t — fast swings every few steps, slow barely moves. The
    multi-scale code of 1c, now stacked into one vector."""
    _banner("LAYER 1d: build it — a geometric ladder of frequencies stacked into one vector")

    dim = 32
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)

    print(f"  dim={dim} -> half={half} frequencies. f_k = 10000^(-k/{half}),")
    print(f"  emb(t) = [sin(t·f_0..f_{half-1}), cos(t·f_0..f_{half-1})]  (a {dim}-vector).\n")

    # the ladder: geometric, so the ratio between consecutive hands is constant.
    ratio = (freqs[1] / freqs[0]).item()
    print(f"  the frequency ladder (period 2pi/f_k = steps to complete one cycle):")
    print(f"  {'k':>3} | {'f_k':>10} | {'period (steps)':>14}")
    print(f"  {'-'*3}-+-{'-'*10}-+-{'-'*14}")
    for k in [0, 1, 2, 3, 4, 8, 12, half - 1]:
        fk = freqs[k].item()
        print(f"  {k:>3} | {fk:>10.6f} | {2*math.pi/fk:>14.1f}")
    print(f"  consecutive frequencies share a CONSTANT ratio f_(k+1)/f_k = {ratio:.4f}")
    print(f"  (geometric): a few hands span periods ~6 -> ~35000, i.e. many scales at once.\n")

    # SEE the multi-scale code: one fast channel, one mid, one slow, across several t. Fast
    # swings through its whole range every few steps; slow barely changes over all of 0..999.
    ts = torch.tensor([0, 5, 25, 100, 500, 999])
    emb = timestep_embedding(ts, dim)                        # (6, 32): [sin(16) | cos(16)]
    fast_k, mid_k, slow_k = 0, 8, 15                         # indices into the sin block
    print("  the same code, read per channel — sin at a FAST, MID, SLOW frequency vs t:")
    print(f"  {'t':>4} | {'sin fast (k=0)':>14} {'sin mid (k=8)':>14} {'sin slow (k=15)':>15}")
    print(f"  {'-'*4}-+-{'-'*14}-{'-'*14}-{'-'*15}")
    for i, tv in enumerate(ts.tolist()):
        print(f"  {tv:>4} | {emb[i, fast_k]:>+14.4f} {emb[i, mid_k]:>+14.4f} {emb[i, slow_k]:>+15.6f}")
    print("    -> fast channel swings all over even for small t steps (fine detail); slow")
    print("       channel barely leaves 0 across the whole schedule (coarse position). Every")
    print("       t gets a code that is precise up close AND placed globally.\n")
    print("  LAYER 1d DONE: t -> a compact multi-scale vector (built). Next (1e): why we keep")
    print("  BOTH sin and cos — a shift t -> t+Δ becomes a rotation (the RoPE connection).")


# ---------------------------------------------------------------------------
# LAYER 1e: why keep BOTH sin and cos — a shift in t becomes a ROTATION.
#
# 1d built the vector with a sin AND a cos per frequency. Why both? Treat one frequency's
# pair as a 2D point on a circle, hand(t) = [cos(t·f), sin(t·f)], at angle t·f. Advancing t
# by Δ adds Δ·f to the angle — i.e. ROTATES the point by a fixed angle that depends only on
# Δ, not on t:
#
#     [cos((t+Δ)f)]   [cos Δf   -sin Δf] [cos t f]
#     [sin((t+Δ)f)] = [sin Δf    cos Δf] [sin t f]  =  R(Δf) · hand(t)
#
# So "shift t by Δ" is a LINEAR map (a rotation) that a weight matrix can implement, and the
# SAME matrix works at every t — the encoding makes relative offsets in t linearly available.
# That needs the full 2D pair: with sin ALONE you lose the phase (two different t can share a
# sin value but sit at different angles), so a shift is no longer a function of the input.
# This is exactly RoPE from llm/ (there you rotate query/key pairs by POSITION; here the same
# rotate-a-2D-pair algebra encodes TIME). This bite verifies the rotation and the sin-only gap.
# ---------------------------------------------------------------------------
def exp_1e_sin_and_cos_rotation():
    """Show a shift t -> t+Δ is a rotation of each (cos,sin) hand: one rotation matrix R(Δf)
    reproduces the shifted pair from the unshifted one, identically at every t. Then show sin
    ALONE can't do this — two timesteps sharing a sin value get sent to different sins by the
    same shift, because the phase (which cos carries) is lost. The RoPE connection."""
    _banner("LAYER 1e: sin AND cos — a shift in t is a rotation (needs the 2D pair; RoPE)")

    P = 40                                                   # one frequency, period 40 steps
    f = 2 * math.pi / P

    def hand(t):                                             # [cos(t·f), sin(t·f)], a point on a circle
        a = t * f
        return torch.stack([torch.cos(a), torch.sin(a)], dim=-1)

    # a shift by Δ rotates every hand by the SAME angle Δ·f, regardless of t.
    delta = 10.0
    phi = delta * f
    R = torch.tensor([[math.cos(phi), -math.sin(phi)],
                      [math.sin(phi),  math.cos(phi)]])      # rotation by Δ·f
    print(f"  one frequency (period {P}). shift Δ={delta:.0f} -> rotate by Δ·f = {phi:.4f} rad.")
    print(f"  check R(Δf)·hand(t) == hand(t+Δ) at several t (same R for all of them):\n")
    print(f"  {'t':>5} | {'hand(t)':>18} | {'R·hand(t)':>18} | {'hand(t+Δ)':>18}")
    print(f"  {'-'*5}-+-{'-'*18}-+-{'-'*18}-+-{'-'*18}")
    for t in [3.0, 100.0, 555.0]:
        h = hand(torch.tensor(t))
        Rh = R @ h
        h2 = hand(torch.tensor(t + delta))
        fmt = lambda v: f"({v[0]:+.3f},{v[1]:+.3f})"
        print(f"  {t:>5.0f} | {fmt(h):>18} | {fmt(Rh):>18} | {fmt(h2):>18}")
    print("    -> R·hand(t) matches hand(t+Δ) everywhere: 'advance t by Δ' is ONE fixed linear")
    print("       map (a rotation) a weight matrix can apply — relative offsets in t are linear.\n")

    # sin ALONE loses the phase: t=3 and t=17 share a sin value (since 20-t reflects the angle),
    # but a +10 shift sends them to DIFFERENT sins — the map isn't a function of sin alone.
    t1, t2 = torch.tensor(3.0), torch.tensor(17.0)          # 17 = 20 - 3, so equal sin, opposite cos
    print("  why not sin only? t=3 and t=17 have the SAME sin but different cos (different")
    print("  angle). Apply the same +10 shift and their sins DIVERGE — cos carried the phase:")
    print(f"  {'t':>5} | {'sin(t·f)':>9} {'cos(t·f)':>9} | {'sin((t+10)·f)':>14}")
    print(f"  {'-'*5}-+-{'-'*9}-{'-'*9}-+-{'-'*14}")
    for t in [t1, t2]:
        s, c = torch.sin(t * f).item(), torch.cos(t * f).item()
        s2 = torch.sin((t + delta) * f).item()
        print(f"  {t.item():>5.0f} | {s:>+9.4f} {c:>+9.4f} | {s2:>+14.4f}")
    print("    -> same sin in, different sin out: with sin alone a shift is ambiguous. The")
    print("       (sin,cos) PAIR fixes the angle, so the rotation is well-defined.\n")
    print("  LAYER 1e DONE: both components => shifts in t are rotations (linear, t-independent),")
    print("  the RoPE algebra reused for time. Next (1f): two clean properties of the full")
    print("  vector — every entry in [-1,1], and constant norm √half for every t.")


# ---------------------------------------------------------------------------
# LAYER 1f: two clean properties — bounded to [-1,1], and constant norm √half.
#
# The full vector has two free "it just works as an input" properties, both from sin/cos:
#   (a) BOUNDED: every entry is a sin or cos, so emb(t) in [-1,1]^dim for ANY t and ANY T —
#       no scale tuning, no blow-up at large t (unlike raw t, which ran 0..999).
#   (b) CONSTANT NORM: group the entries by frequency and use sin² + cos² = 1:
#           ||emb(t)||² = Σ_k [ sin²(t·f_k) + cos²(t·f_k) ] = Σ_k 1 = half
#       so ||emb(t)|| = √half for EVERY t. No timestep enters the net louder than another —
#       all the information is in the DIRECTION of the vector, none in its length.
# Small but real: a well-scaled, magnitude-uniform input is exactly what the first linear
# layer wants (the thing raw-scalar/one-hot both got wrong). This bite just measures both.
# ---------------------------------------------------------------------------
def exp_1f_bounded_constant_norm():
    """Measure the two properties across a spread of t: every entry stays within [-1,1]
    (bounded, no matter how large t), and the L2 norm is exactly √half at every t (constant,
    from sin²+cos²=1 summed over the half frequencies). A uniform, well-scaled input."""
    _banner("LAYER 1f: two properties — bounded to [-1,1], and constant norm √half")

    dim = 32
    half = dim // 2
    ts = torch.tensor([0, 1, 10, 100, 500, 999, 5000])       # include t past T to show it stays bounded
    emb = timestep_embedding(ts, dim)                        # (7, 32)

    print(f"  dim={dim} -> ||emb|| should be √half = √{half} = {half**0.5:.4f} at every t.\n")
    print(f"  {'t':>5} | {'min entry':>10} {'max entry':>10} | {'||emb(t)||':>10}")
    print(f"  {'-'*5}-+-{'-'*10}-{'-'*10}-+-{'-'*10}")
    for i, tv in enumerate(ts.tolist()):
        e = emb[i]
        print(f"  {tv:>5} | {e.min():>+10.4f} {e.max():>+10.4f} | {e.norm():>10.4f}")
    print("    -> (a) every entry in [-1,1], even at t=5000 (past T): it's all sin/cos, so the")
    print("           input never needs rescaling and never blows up. (t=0 -> sin=0, cos=1.)")
    print(f"       (b) ||emb(t)|| = {half**0.5:.4f} = √half for EVERY t: sin²+cos²=1 on each of the")
    print("           half frequencies sums to half. No timestep is 'louder'; the code lives")
    print("           entirely in the DIRECTION, which is what the first linear layer wants.\n")
    print("  LAYER 1f DONE: a bounded, magnitude-uniform input (fixing what raw-scalar and")
    print("  one-hot got wrong). Next (1g): the payoff — similarity depends only on Δ=t-t'.")


# ---------------------------------------------------------------------------
# LAYER 1g: the payoff — similarity depends ONLY on the gap Δ = t - t'.
#
# This is the property that makes the whole encoding worth it, and the one one-hot lacked.
# The dot product of two embeddings turns out to depend only on how FAR APART the timesteps
# are, not where they sit — a smooth kernel over t. So nearby noise levels get nearly
# parallel codes (behaviour the net learns at one transfers to its neighbours), while distant
# levels stay distinguishable. That is exactly the "smooth" item on 1b's wishlist, delivered.
# ---------------------------------------------------------------------------
'''
DERIVATION — emb(t)·emb(t') is a function of Δ = t − t' alone (a smooth similarity kernel).
Using the identity  sin a·sin b + cos a·cos b = cos(a − b), pair the sin and cos parts of
each frequency k:
    emb(t)·emb(t') = Σ_k [ sin(t f_k)·sin(t' f_k) + cos(t f_k)·cos(t' f_k) ]
                   = Σ_k cos( (t − t')·f_k )
                   = Σ_k cos( Δ·f_k ) ,        Δ = t − t'.
At Δ=0 every term is cos 0 = 1, so the sum is Σ_k 1 = half = ||emb||² — its maximum (identical
timesteps maximally aligned, consistent with 1f's constant norm). As |Δ| grows the different
frequencies f_k drift out of phase and their cosines partly cancel, so similarity DECAYS. And
it depends on the GAP Δ only, never on absolute position — the exact inductive bias we want:
"how far apart in noise level", not two unrelated integer labels (contrast one-hot, whose dot
is 1 at Δ=0 and 0 for EVERY other Δ — no smoothness at all).
'''
def exp_1g_similarity_kernel():
    """Verify the payoff: dot(emb(t), emb(t')) equals the closed form Σ_k cos(Δ·f_k), depends
    only on Δ=t-t' (the two Δ=1 pairs at different positions match), peaks at Δ=0 = half, and
    decays with |Δ|. The smooth over-t similarity one-hot could never give."""
    _banner("LAYER 1g: the payoff — similarity depends only on Δ=t-t' (a smooth kernel)")

    dim = 32
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)

    def kernel(delta):                                       # closed form from the derivation
        return torch.cos(delta * freqs).sum().item()

    print(f"  dot(emb(t), emb(t')) vs the closed form Σ_k cos(Δ·f_k). Max at Δ=0 is half={half}.\n")
    print(f"  {'t':>4} {'tʹ':>4} {'Δ':>5} | {'measured dot':>12} {'Σcos(Δ·f_k)':>12} | {'cos-sim':>7}")
    print(f"  {'-'*4}-{'-'*4}-{'-'*5}-+-{'-'*12}-{'-'*12}-+-{'-'*7}")
    for a, b in [(500, 500), (500, 501), (200, 201), (500, 510), (500, 600), (500, 999)]:
        ea = timestep_embedding(torch.tensor([a]), dim)[0]
        eb = timestep_embedding(torch.tensor([b]), dim)[0]
        dot = (ea @ eb).item()
        cossim = dot / (ea.norm() * eb.norm()).item()
        print(f"  {a:>4} {b:>4} {b-a:>5} | {dot:>12.4f} {kernel(float(b-a)):>12.4f} | {cossim:>7.4f}")
    print()
    print("  Read it: measured dot == Σ_k cos(Δ·f_k) exactly; the two Δ=1 rows (500-501 and")
    print("  200-201) are IDENTICAL — similarity sees the gap, not the position; cos-sim is ~1")
    print("  for adjacent t (neighbouring levels share a code, so the net interpolates across")
    print("  them) and falls off as Δ grows (distant levels stay distinguishable). Compare")
    print("  one-hot: 1 at Δ=0, 0 for every other Δ — no smoothness. THAT smooth-in-Δ kernel is")
    print("  the whole reason we embed t instead of feeding the raw integer.\n")
    print("  LAYER 1 COMPLETE. t -> a bounded, constant-norm, multi-scale sinusoidal vector")
    print("  whose (sin,cos) pairs make shifts rotations and whose similarity is smooth in Δ.")
    print("  Next (Layer 2): the denoiser MLP — fuse this t-embedding with x_t, predict eps.")


# ---------------------------------------------------------------------------
# LAYER 2a: the plain MLP + the input/output contract (no time yet).
#
# The denoiser is a function eps_hat = net(x_t, t) with a strict shape contract: it takes a
# noised image and returns a noise prediction the SAME shape as the image (one predicted eps
# per pixel), because the loss is MSE(eps_hat, eps) and eps matches x_t. For MNIST we flatten
# the (1,28,28) image to a 784-vector, run a few Linear+SiLU layers, and project back to 784.
#
# One design point that trips people up: there is NO activation on the OUTPUT. The target eps
# is standard normal — unbounded, spanning negatives and values well outside [-1,1]. A tanh/
# sigmoid on the output would cap the range and make large-|eps| targets impossible to hit.
# Hidden layers get nonlinearities (SiLU); the final projection is a bare Linear.
#
# This bite builds only the x_t -> eps path. With no t input it is exactly the ill-posed
# mapping 1a warned about (one output for an x_t that could need many different eps); 2b adds t.
# ---------------------------------------------------------------------------
def exp_2a_plain_mlp():
    """Build the minimal x_t -> eps MLP (flatten 784 -> SiLU hidden -> 784) and check the
    contract: output shape equals input shape, and the output is UNBOUNDED (no final
    activation), as eps ~ N(0,1) requires. Note it has no t input yet — the 1a ambiguity is
    still unresolved, which is 2b's job."""
    _banner("LAYER 2a: the plain MLP — x_t -> eps, and why the output has no activation")

    torch.manual_seed(0)
    D = 28 * 28                                              # 784 pixels (MNIST flattened)
    hidden = 256

    # a plain MLP: no time yet. Hidden layers use SiLU; the OUTPUT layer is a bare Linear.
    net = torch.nn.Sequential(
        torch.nn.Linear(D, hidden), torch.nn.SiLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
        torch.nn.Linear(hidden, D),                         # -> eps, NO activation here
    )
    n_params = sum(p.numel() for p in net.parameters())

    B = 4
    x_t = torch.randn(B, D)                                 # a batch of noised images (flattened)
    eps_hat = net(x_t)                                      # the noise prediction

    print(f"  MLP: {D} -> {hidden} -> {hidden} -> {D}  (SiLU on hidden, none on output), "
          f"{n_params:,} params")
    print(f"  forward:  x_t {tuple(x_t.shape)}  ->  eps_hat {tuple(eps_hat.shape)}   "
          f"(same shape: one predicted eps per pixel)\n")

    # why no output activation: the TARGET eps ~ N(0,1) is unbounded — a big chunk of it lies
    # outside [-1,1] — so the final layer must be a bare Linear that can reach those values.
    eps_target = torch.randn(B, D)                          # the kind of target the net must hit
    frac_out = (eps_target.abs() > 1).float().mean().item()
    print(f"  the TARGET eps ~ N(0,1): range [{eps_target.min():+.2f}, {eps_target.max():+.2f}], "
          f"{frac_out:.0%} of entries fall OUTSIDE [-1,1].")
    print("  -> the output layer must be a bare Linear (unbounded). A tanh/sigmoid would cap it")
    print("     to (-1,1) and make that ~1/3 of targets unreachable. (The untrained net's own")
    print(f"     output is tiny here — [{eps_hat.min():+.2f}, {eps_hat.max():+.2f}] — from small init,")
    print("     not any cap; training scales it up. Hidden layers DO get SiLU; only the output")
    print("     is bare.)\n")
    print("  But this net is a function of x_t ALONE — no t input. That is precisely the")
    print("  one-to-many mapping 1a warned about: the same x_t can need very different eps at")
    print("  different t, and this net can only ever emit ONE answer for it (MSE-optimally, the")
    print("  blurry average over all those t). Next (2b): fuse the timestep embedding in so the")
    print("  prediction can finally depend on t.")


# ---------------------------------------------------------------------------
# LAYER 2b: fusing t — project the timestep embedding and ADD it into the hidden.
#
# This is the one genuinely new idea in Layer 2, and the one reused everywhere later. The
# time signal enters the network as a t-dependent BIAS on the hidden units:
#
#     temb   = timestep_embedding(t)          # (B, temb_dim)  the fixed sinusoids (Layer 1)
#     t_bias = time_mlp(temb)                  # (B, hidden)    Linear->SiLU->Linear, learned
#     h      = in_proj(x_t) + t_bias           # (B, hidden)    inject, THEN nonlinearity
#     h      = SiLU(h)
#
# Two choices to justify:
#   WHY ADD (not concat)? For a single linear layer they are IDENTICAL: concat([x,e])·W splits
#   as x·W_x + e·W_e, and "add a linear projection of e" IS that second term. So add isn't a
#   hack — it's concat refactored. We prefer the add form because (i) it keeps the hidden width
#   fixed and (ii) you compute t_bias ONCE and inject the same signal into MANY blocks (every
#   resblock of a U-Net), which is how conditioning is actually plumbed. Same move later: AdaLN
#   (DiT) and CFG both inject a conditioning vector this way.
#   WHY PROJECT the sinusoids first? The embedding is a FIXED nonlinear code of t. A small
#   learned MLP (Linear->SiLU->Linear) gives the time pathway its OWN capacity to reshape those
#   sinusoids into the t-features this network wants, and to match the hidden width.
#
# The payoff: the SAME x_t now produces DIFFERENT hidden activations at different t — the 1a
# ambiguity is finally resolved inside the network.
# ---------------------------------------------------------------------------
def exp_2b_fuse_time():
    """Build one time-conditioned block (in_proj(x_t) + time_mlp(temb)) and show (1) the same
    x_t at different t now yields DIFFERENT hidden activations — t matters — and (2) 'add a
    projected embedding' is EXACTLY 'concat then linear' for a linear layer (numerically), so
    the add form is a principled refactor we pick for width + multi-block reuse."""
    _banner("LAYER 2b: fusing t — add a projected timestep embedding into the hidden units")

    torch.manual_seed(0)
    D, hidden, temb_dim = 28 * 28, 256, 32
    in_proj = torch.nn.Linear(D, hidden)
    time_mlp = torch.nn.Sequential(                          # learned reshaper of the sinusoids
        torch.nn.Linear(temb_dim, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden),
    )

    # (1) SAME image, two different t -> the injected t_bias differs -> hidden differs.
    x = torch.randn(1, D)
    x_rep = x.expand(2, D)                                   # identical x_t in both rows
    t = torch.tensor([100, 800])
    base = in_proj(x_rep)                                    # (2, hidden): identical rows (same x)
    t_bias = time_mlp(timestep_embedding(t, temb_dim))       # (2, hidden): differs by t
    h = base + t_bias                                        # inject
    cos = torch.nn.functional.cosine_similarity(h[0], h[1], dim=0).item()

    print("  (1) same x_t, two timesteps t=100 and t=800, through in_proj(x_t) + time_mlp(temb):")
    print(f"      base in_proj rows identical (same x):   ||base[0]-base[1]|| = {(base[0]-base[1]).norm():.2e}")
    print(f"      injected t_bias differs by t:           ||t_bias[0]-t_bias[1]|| = {(t_bias[0]-t_bias[1]).norm():.3f}")
    print(f"      so the hidden now differs by t:         ||h[0]-h[1]|| = {(h[0]-h[1]).norm():.3f}, "
          f"cos = {cos:+.3f}")
    print("      -> the SAME x_t maps to DIFFERENT internal states at different t. 1a resolved.\n")

    # (2) add == concat-then-linear (single linear layer). Split a concat layer's weight into
    # the x-part and the e-part; the concat output equals x·W_x + e·W_e (= add-a-projection).
    e = timestep_embedding(t, temb_dim)                      # (2, temb_dim)
    lin = torch.nn.Linear(D + temb_dim, hidden)              # a CONCAT-then-linear layer
    concat_out = lin(torch.cat([x_rep, e], dim=-1))          # W·[x;e] + b
    W_x, W_e = lin.weight[:, :D], lin.weight[:, D:]          # its two column-blocks
    add_out = x_rep @ W_x.T + e @ W_e.T + lin.bias           # x·W_x + e·W_e + b  == add form
    print("  (2) is 'add a projection' a hack? No — for one linear layer it's concat refactored:")
    print(f"      concat([x,e])·W  vs  x·W_x + e·W_e :  max diff = {(concat_out - add_out).abs().max():.2e}")
    print("      -> identical. We choose the ADD form because it keeps hidden width fixed and lets")
    print("         ONE computed t_bias be injected into many blocks (U-Net resblocks); the")
    print("         nonlinear time_mlp then gives t its own capacity beyond a single linear.\n")
    print("  LAYER 2b DONE: t is fused as a learned, projected additive bias — the conditioning")
    print("  move reused as AdaLN/CFG later. Next (2c): assemble the full nn.Module and sanity-")
    print("  check it (params, shapes, and that the output really moves when t moves).")


# ---------------------------------------------------------------------------
# LAYER 2c: assemble the full nn.Module and sanity-check it.
#
# Put 2a (the x_t->eps path) and 2b (inject a projected t_bias) together into one reusable
# module — the same shape our real diffusion/model.py will have. Structure:
#   in_proj:  x_t (flattened) -> hidden
#   time_mlp: timestep_embedding(t) -> hidden        (computed ONCE)
#   blocks:   a few residual blocks, each adding temb then mixing (2b's reuse-across-blocks)
#   out:      hidden -> eps (image shape), a bare Linear (2a: no output activation)
# Then three sanity checks that matter: (i) output shape == input shape; (ii) it's
# deterministic (same x_t,t -> same eps); (iii) the output actually MOVES when t moves and is
# CONSTANT across the batch when t is held fixed — i.e. t genuinely reaches the output (the
# thing 1a demanded and 2a lacked).
# ---------------------------------------------------------------------------
class _ResBlock(torch.nn.Module):
    """A time-conditioned residual block: add the (shared) temb, then two SiLU-Linear mixes."""
    def __init__(self, hidden):
        super().__init__()
        self.lin1 = torch.nn.Linear(hidden, hidden)
        self.lin2 = torch.nn.Linear(hidden, hidden)

    def forward(self, h, temb):
        z = self.lin1(torch.nn.functional.silu(h + temb))     # inject t here (2b)
        z = self.lin2(torch.nn.functional.silu(z))
        return h + z                                          # residual


class DenoiserMLP(torch.nn.Module):
    """Minimal eps_theta(x_t, t) for MNIST: flatten -> conditioned resblocks -> eps. Mirrors
    the real diffusion/model.py contract: forward(x, t) with x (B,1,28,28), t (B,) -> eps same
    shape as x."""
    def __init__(self, img_shape=(1, 28, 28), hidden=256, temb_dim=32, depth=3):
        super().__init__()
        self.img_shape, self.temb_dim = img_shape, temb_dim
        D = int(torch.tensor(img_shape).prod())
        self.in_proj = torch.nn.Linear(D, hidden)
        self.time_mlp = torch.nn.Sequential(                  # reshape the fixed sinusoids (2b)
            torch.nn.Linear(temb_dim, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden),
        )
        self.blocks = torch.nn.ModuleList(_ResBlock(hidden) for _ in range(depth))
        self.out = torch.nn.Linear(hidden, D)                 # -> eps, bare Linear (2a)

    def forward(self, x, t):
        B = x.shape[0]
        h = self.in_proj(x.flatten(1))                        # (B, hidden)
        temb = self.time_mlp(timestep_embedding(t, self.temb_dim))  # (B, hidden), computed once
        for block in self.blocks:
            h = block(h, temb)                                # same temb injected each block
        return self.out(h).view(B, *self.img_shape)           # (B, 1, 28, 28) == x's shape


def exp_2c_assemble():
    """Instantiate DenoiserMLP, count params, and run the three sanity checks: output shape ==
    input shape; deterministic; and t reaches the output — same x_t at different t gives
    DIFFERENT eps, while a batch at a FIXED t gives identical eps (so t is the only driver)."""
    _banner("LAYER 2c: assemble DenoiserMLP + sanity — shapes, determinism, and t reaches out")

    torch.manual_seed(0)
    model = DenoiserMLP()
    n_params = sum(p.numel() for p in model.parameters())

    # (i) shape contract, on a REAL image batch shape (B,1,28,28).
    x = torch.randn(1, 1, 28, 28)
    x_rep = x.expand(4, 1, 28, 28)                            # identical image, 4 copies
    ts = torch.tensor([0, 100, 500, 900])
    with torch.no_grad():
        eps = model(x_rep, ts)                                # (4,1,28,28)
        eps_again = model(x_rep, ts)                          # for determinism check
        eps_samet = model(x_rep, torch.full((4,), 500))       # same x AND same t across batch
    print(f"  DenoiserMLP(hidden=256, depth=3): {n_params:,} params")
    print(f"  (i) shape:  x {tuple(x_rep.shape)}  ->  eps {tuple(eps.shape)}  (matches — one eps per pixel)")

    # (ii) determinism: no dropout/BN, so identical inputs -> identical output.
    print(f"  (ii) deterministic:  max|eps - eps_again| = {(eps - eps_again).abs().max():.2e}")

    # (iii) t reaches the output. Same image, four t's -> outputs must DIFFER; same image at a
    # single fixed t across the batch -> outputs must be IDENTICAL (t is the only thing varying).
    flat = eps.flatten(1)                                      # (4, 784)
    print("  (iii) t reaches the output — same x_t, four different t, pairwise ||eps_i - eps_j||:")
    print(f"        {'t=':>6}" + "".join(f"{tv:>8}" for tv in ts.tolist()))
    for i, tv in enumerate(ts.tolist()):
        row = "".join(f"{(flat[i]-flat[j]).norm():>8.3f}" for j in range(len(ts)))
        print(f"        {tv:>6}{row}")
    samet_spread = (eps_samet - eps_samet[0:1]).abs().max().item()
    print(f"        vs a batch at a FIXED t=500 (same x): max spread across rows = {samet_spread:.2e}")
    print("        -> off-diagonal >0: changing ONLY t changes the prediction; fixed t -> identical")
    print("           rows. t genuinely drives the output (1a's requirement, met end-to-end).\n")
    print("  LAYER 2 COMPLETE: eps_theta(x_t, t) is a real nn.Module — right shape, unbounded")
    print("  output, t fused as a projected additive bias and verified to reach the prediction.")
    print("  Next (Layer 3): the DDPM loss + one training step — draw (t, eps), form x_t, MSE,")
    print("  backprop, and overfit a single batch to prove the objective and gradients wire up.")


# ---------------------------------------------------------------------------
# LAYER 3: the DDPM loss + one training step (overfit a single batch).
#
# We now have the target (forward_process Layer 4/5) and the network (Layer 2). Wire them
# into the actual training step — the entire DDPM objective is:
#
#     draw t ~ Uniform, eps ~ N(0,I) ; build x_t = √ab·x0 + √(1-ab)·eps (closed form)
#     eps_hat = model(x_t, t)
#     loss    = mean( (eps_hat - eps)² )          # plain MSE, the "simple" loss (Ho 2020)
#     loss.backward() ; optimizer.step()
#
# That's it — a bog-standard supervised regression loop. The ONLY diffusion-specific part is
# how the (input, target) pair is manufactured (the closed form); everything else is generic.
# Two things worth SEEING here: (a) the untrained loss is ~1.0, because the target eps ~ N(0,1)
# has variance 1 and a net predicting ~0 gets MSE ≈ Var(eps) = 1 — so "loss below 1" is the
# first sign of learning; (b) OVERFIT one fixed batch and the loss collapses toward 0, proving
# the objective and gradients actually wire up end-to-end. Real training (fresh noise each
# step over all of MNIST) is Layers 4-5; here we just prove the machinery.
# ---------------------------------------------------------------------------
def exp_3_loss_and_step():
    """Assemble the DDPM training step and overfit ONE fixed batch. Show the untrained loss
    sits at ~1.0 (= Var(eps), the N(0,1) target), then drops toward 0 as we repeatedly fit the
    same (x_t, t) -> eps — the objective + gradients wire up. Real data/fresh noise is Layers 4-5."""
    _banner("LAYER 3: the DDPM loss + one training step — overfit a single batch")

    from forward_process import make_linear_schedule
    torch.manual_seed(0)
    T = 1000
    _, _, alpha_bars = make_linear_schedule(T=T)

    model = DenoiserMLP()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ONE fixed batch: draw (x0, t, eps) once and build x_t. We hold them fixed so the model
    # must fit this exact (x_t, t) -> eps mapping — the classic "overfit one batch" sanity test.
    B = 8
    x0 = torch.randn(B, 1, 28, 28)                            # stand-in images (real MNIST: Layer 4)
    t = torch.randint(0, T, (B,))                             # a noise level per example
    eps = torch.randn_like(x0)                                # the target noise
    ab = alpha_bars[t].view(B, 1, 1, 1)                       # (B,1,1,1) to broadcast over pixels
    x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps      # the closed form (forward_process 1b)

    def loss_fn():
        eps_hat = model(x_t, t)
        return ((eps_hat - eps) ** 2).mean()                 # the DDPM "simple" MSE loss

    # (a) the untrained loss ~ 1.0 = Var(eps): predicting ~0 for an N(0,1) target gives MSE ~1.
    with torch.no_grad():
        l0 = loss_fn().item()
    print(f"  target eps ~ N(0,1): its variance is {eps.var().item():.3f}, so an untrained net")
    print(f"  (outputs ~0) should start near MSE ~1.  ->  initial loss = {l0:.4f}")
    print("  'loss below 1' is the first evidence the net predicts anything useful about eps.\n")

    # overfit the fixed batch: same (x_t, t, eps) every step. Loss should collapse toward 0.
    print("  overfitting ONE fixed batch (same x_t,t,eps each step) to prove grads wire up:")
    print(f"  {'step':>5} | {'loss':>10} | {'max|eps_hat-eps|':>16}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*16}")
    for step in range(601):
        loss = loss_fn()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (0, 25, 50, 100, 200, 400, 600):
            with torch.no_grad():
                max_err = (model(x_t, t) - eps).abs().max().item()
            print(f"  {step:>5} | {loss.item():>10.5f} | {max_err:>16.4f}")
    print()
    print("  -> loss falls from ~1 toward ~0 and the per-pixel eps error shrinks: the network")
    print("  CAN represent the target and the gradients flow through the whole path (in_proj,")
    print("  the time_mlp, the resblocks, out). The objective is wired up.\n")
    print("  Reminder: overfitting ONE batch only proves the machinery — it's memorising, not")
    print("  generalising. The real loop draws FRESH t and eps every step over the whole")
    print("  dataset so the net learns the actual denoising function. Next (Layer 4): the MNIST")
    print("  data pipeline (a tiny Dataset over the npz), then (Layer 5) that real training loop.")


# ---------------------------------------------------------------------------
# LAYER 4: the MNIST data pipeline — a tiny Dataset over the npz (no torchvision).
#
# Layer 3 used stand-in random images; the real loop needs real digits. We wrap the same
# mnist.npz (from forward_process) in a torch Dataset so a DataLoader can batch and shuffle
# it — plain plumbing, but one choice matters: NORMALIZE pixels from uint8 [0,255] to [-1,1]
# (x/127.5 - 1), not [0,1].
#
# Why [-1,1]? The forward process mixes x0 with eps ~ N(0,I) and drives x_t toward N(0,I) —
# a SYMMETRIC, O(1)-scale target. Data in [0,1] is all-positive and half that scale, so its
# noised images sit off to one side of what the model (and the N(0,I) noise) are symmetric
# around. Mapping to [-1,1] (background black = -1, ink = +1) puts the data on the same
# symmetric O(1) scale — the standard DDPM convention. NB it does NOT make the data zero-mean:
# MNIST is mostly black background, so the mean stays quite negative (~ -0.73). That's fine —
# what matters is the scale/symmetry, not a zero mean. Sampling later inverts it to view digits.
# ---------------------------------------------------------------------------
def _mnist_npz_path():
    """Path to the cached mnist.npz (shared with forward_process), downloading on first use."""
    data_dir = os.path.join(os.path.dirname(_HERE), "data")   # diffusion/data
    path = os.path.join(data_dir, "mnist.npz")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        import urllib.request
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        print(f"  downloading MNIST npz (~11MB) -> {path} ...")
        urllib.request.urlretrieve(url, path)
    return path


class MNISTData(torch.utils.data.Dataset):
    """MNIST as (image, label) over the npz. Images are (1,28,28) float normalized to [-1,1]
    (the scale the forward process assumes). No torchvision."""
    def __init__(self, train=True):
        import numpy as np
        d = np.load(_mnist_npz_path())
        x = d["x_train"] if train else d["x_test"]            # (N,28,28) uint8 in [0,255]
        y = d["y_train"] if train else d["y_test"]
        self.x = (torch.from_numpy(x).float() / 127.5 - 1.0).unsqueeze(1)   # (N,1,28,28) in [-1,1]
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def exp_4_data():
    """Build the MNIST Dataset + DataLoader and read one batch: shapes, the [-1,1] range and
    what its mean/std say (background-heavy, zero-ish center), then push that REAL batch through
    the exact Layer-3 pipeline (draw t,eps -> x_t -> model -> MSE) to confirm it's drop-in
    ready for training."""
    _banner("LAYER 4: the MNIST data pipeline — tiny Dataset over the npz, normalized to [-1,1]")

    train = MNISTData(train=True)
    test = MNISTData(train=False)
    loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
    x, y = next(iter(loader))                                 # one batch

    print(f"  dataset: {len(train)} train / {len(test)} test images, no torchvision.")
    print(f"  one batch:  images {tuple(x.shape)} ({x.dtype}),  labels {tuple(y.shape)}")
    print(f"  pixel range [{x.min():+.2f}, {x.max():+.2f}]  mean {x.mean():+.3f}  std {x.std():.3f}")
    print("    -> range is [-1,1] (not [0,1]); the mean is very negative because most pixels are")
    print("       black background (= -1) with a little white ink (= +1). The data is NOT zero-")
    print("       mean — [-1,1] buys a symmetric, O(1) scale matching N(0,I), not a zero center.\n")

    # drop-in check: the SAME Layer-3 step, now on a real digit batch. Untrained loss ~1.
    from forward_process import make_linear_schedule
    torch.manual_seed(0)
    _, _, alpha_bars = make_linear_schedule(T=1000)
    model = DenoiserMLP()
    B = x.shape[0]
    t = torch.randint(0, 1000, (B,))
    eps = torch.randn_like(x)
    ab = alpha_bars[t].view(B, 1, 1, 1)
    x_t = torch.sqrt(ab) * x + torch.sqrt(1 - ab) * eps       # closed form on REAL x0
    with torch.no_grad():
        loss = ((model(x_t, t) - eps) ** 2).mean().item()
    print(f"  drop-in check: real batch -> (t, eps) -> x_t -> model -> MSE = {loss:.4f}  (~1, as")
    print("  expected for an untrained net on the N(0,1) target). The pipeline is ready.\n")
    print("  LAYER 4 DONE: real digits flow through the same (x_t, t) -> eps machinery. Next")
    print("  (Layer 5): the real training loop — iterate the loader, fresh t and eps every step,")
    print("  and watch the loss fall on actual MNIST.")


# the config Layers 5-6 train/reload. Layer 2's DEFAULT DenoiserMLP (hidden=256, depth=3) is
# deliberately minimal and, it turns out, UNDERFITS — it plateaus around test-loss ~0.7 no
# matter the lr (capacity-limited). Widening to ~9M params drops it to ~0.13, a real
# eps-predictor. Capacity is the lever here; we scale the same module up for the real train.
DENOISER_CONFIG = dict(hidden=768, depth=6)


def build_denoiser():
    """The scaled-up DenoiserMLP that Layers 5 and 6 share (so the checkpoint reloads cleanly)."""
    return DenoiserMLP(**DENOISER_CONFIG)


def _ckpt_path():
    out_dir = os.path.join(_HERE, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "denoiser_mlp.pt")


# ---------------------------------------------------------------------------
# LAYER 5: the real training loop — train on all of MNIST, watch the loss fall.
#
# Same step as Layer 3, but now it's the REAL thing: iterate the DataLoader (Layer 4) and, for
# every batch, draw FRESH t and eps — so the net never sees the same (x_t, t, eps) twice and
# must learn the actual denoising FUNCTION over all images and all noise levels, not memorise a
# fixed batch. The loop is textbook: forward -> MSE -> backward -> step. We track a running loss
# (falls from ~1) and a held-out TEST loss (fresh t,eps on test digits) to confirm it's
# GENERALISING, not overfitting like Layer 3. One lever shows up immediately: the minimal Layer-2
# model plateaus ~0.7 (underfits); we widen it (DENOISER_CONFIG) to actually learn. The trained
# weights are saved so Layer 6 (and the sampling walkthrough) can load them and denoise / generate.
# ---------------------------------------------------------------------------
def exp_5_train_loop(epochs=15, batch_size=128, lr=1e-3, seed=0):
    """Train the (scaled-up) DenoiserMLP on MNIST with fresh (t, eps) per batch. Print the
    running train loss falling from ~1 and a held-out test loss (generalisation, unlike Layer
    3's memorisation), then save the checkpoint for Layer 6 / sampling."""
    _banner("LAYER 5: the real training loop — fresh (t, eps) per batch, loss falls on MNIST")

    from forward_process import make_linear_schedule
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 1000
    _, _, alpha_bars = make_linear_schedule(T=T)
    alpha_bars = alpha_bars.to(device)

    model = build_denoiser().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = torch.utils.data.DataLoader(MNISTData(train=True), batch_size=batch_size,
                                               shuffle=True, drop_last=True)
    test_x = MNISTData(train=False).x[:2048].to(device)      # a fixed held-out chunk for eval

    def ddpm_loss(x0):                                       # the Layer-3 step, fresh t & eps
        B = x0.shape[0]
        t = torch.randint(0, T, (B,), device=device)
        eps = torch.randn_like(x0)
        ab = alpha_bars[t].view(B, 1, 1, 1)
        x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps
        return ((model(x_t, t) - eps) ** 2).mean()

    @torch.no_grad()
    def test_loss():                                        # fresh noise on held-out digits
        return ddpm_loss(test_x).item()                     # 2048 samples -> stable enough to compare

    print(f"  device={device}, model={DENOISER_CONFIG} ({n_params:,} params), {epochs} epochs, "
          f"batch={batch_size}, lr={lr}, {len(train_loader)} steps/epoch")
    print(f"  {'epoch':>5} | {'train loss (run avg)':>20} | {'test loss':>10}")
    print(f"  {'-'*5}-+-{'-'*20}-+-{'-'*10}")
    print(f"  {'init':>5} | {'—':>20} | {test_loss():>10.5f}   <- untrained baseline (~1.0)")
    for ep in range(epochs):
        run = None
        for x0, _ in train_loader:
            x0 = x0.to(device)
            loss = ddpm_loss(x0)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        print(f"  {ep:>5} | {run:>20.5f} | {test_loss():>10.5f}")
    print()

    ckpt = _ckpt_path()
    torch.save({"state_dict": model.state_dict(), "config": DENOISER_CONFIG}, ckpt)
    print(f"  final held-out test loss: {test_loss():.5f}  (started ~1.0). saved -> {ckpt}\n")
    print("  Read it: the loss falls from ~1 to ~0.13 AND the held-out test loss tracks the train")
    print("  loss — the net learned the real denoising function (generalises), it didn't memorise")
    print("  like Layer 3. (The minimal Layer-2 model would have stalled near 0.7 — capacity was")
    print("  the lever.) An MLP still won't match a U-Net, but it clearly predicts eps. Next")
    print("  (Layer 6): feed it HELD-OUT noised digits and reconstruct x0_hat from its predicted")
    print("  eps — SEE it denoise. (Generation from pure noise is the sampling walkthrough.)")


# ---------------------------------------------------------------------------
# LAYER 6: SEE it — reconstruct x0 from the trained net's predicted eps.
#
# The loss number said the net predicts eps; this makes it a PICTURE. Take HELD-OUT test
# digits (never trained on), noise each to a level t with the closed form, run the trained
# model to get eps_hat, and invert the closed form to recover the clean image the net thinks
# is under the noise (forward_process 4a, solved for x0):
#
#     x0_hat = ( x_t - √(1-ab)·eps_hat ) / √ab
#
# This is a ONE-SHOT denoise (jump straight from x_t to an x0 estimate) — not full generation
# yet (that's the iterative sampler, next walkthrough), but it's the direct read on "did it
# learn to see the digit through the noise". Expect: crisp recovery at low/mid t, and blow-up
# at high t — there the signal is essentially gone AND the 1/√ab factor amplifies any eps error
# (the loss@t wall from our per-t check). Saves a grid: for each digit, the noised input row
# and the reconstruction row across t — forward_dissolve.png run backwards.
# ---------------------------------------------------------------------------
def exp_6_visualize(seed=0):
    """Load the trained checkpoint and one-shot denoise held-out digits: quantify recon error
    ||x0_hat - x0|| vs t, then save a grid (per digit: noised x_t row over the reconstruction
    x0_hat row, across t). Shows crisp recovery at low/mid t and breakdown at high t."""
    _banner("LAYER 6: SEE it — one-shot reconstruction x0_hat from the trained net's eps")

    ckpt_path = _ckpt_path()
    if not os.path.exists(ckpt_path):
        print(f"  no checkpoint at {ckpt_path} — run exp_5_train_loop() first.")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from forward_process import make_linear_schedule

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, alpha_bars = make_linear_schedule(T=1000)
    alpha_bars = alpha_bars.to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model = DenoiserMLP(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    def reconstruct(x0, t, eps):                              # one-shot: x_t -> eps_hat -> x0_hat
        ab = alpha_bars[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps
        with torch.no_grad():
            eps_hat = model(x_t, t)
        x0_hat = (x_t - torch.sqrt(1 - ab) * eps_hat) / torch.sqrt(ab)
        return x_t, x0_hat

    test = MNISTData(train=False)
    ts = [0, 50, 100, 200, 400, 700]

    # (1) quantify: mean per-pixel |x0_hat - x0| over 512 held-out digits, per t.
    torch.manual_seed(seed)
    x0_big = test.x[:512].to(device)
    eps_big = torch.randn_like(x0_big)
    print("  one-shot reconstruction error over 512 held-out digits (fresh noise), per t:")
    print(f"  {'t':>5} {'√ab':>7} | {'mean|x0_hat - x0|':>18} | {'mean|x_t - x0|':>15} (noised input)")
    print(f"  {'-'*5}-{'-'*7}-+-{'-'*18}-+-{'-'*15}")
    for tv in ts:
        t = torch.full((x0_big.shape[0],), tv, device=device)
        x_t, x0_hat = reconstruct(x0_big, t, eps_big)
        rec_err = (x0_hat.clamp(-1, 1) - x0_big).abs().mean().item()
        inp_err = (x_t.clamp(-1, 1) - x0_big).abs().mean().item()
        print(f"  {tv:>5} {alpha_bars[tv].sqrt().item():>7.4f} | {rec_err:>18.4f} | {inp_err:>15.4f}")
    print("    -> recon error stays low while the NOISED input error climbs: the net is pulling")
    print("       the digit back out — until high t, where signal is gone and 1/√ab amplifies")
    print("       any eps error, so x0_hat destabilises.\n")

    # (2) picture: a few digits, each as a (noised input, reconstruction) row-pair across t.
    labels = [7, 3, 8]
    idxs = [int((test.y == lb).nonzero()[0]) for lb in labels]
    digits = test.x[idxs].to(device)                         # (K,1,28,28)
    torch.manual_seed(seed)
    eps = torch.randn_like(digits)                           # one fixed eps per digit, all t

    K, C = len(idxs), len(ts)
    fig, axes = plt.subplots(2 * K, C, figsize=(C * 1.5, 2 * K * 1.5))
    for k in range(K):
        for c, tv in enumerate(ts):
            t = torch.full((1,), tv, device=device)
            x_t, x0_hat = reconstruct(digits[k:k+1], t, eps[k:k+1])
            def show(ax, img, ylabel=None, title=None):
                disp = ((img[0, 0].clamp(-1, 1) + 1) / 2).cpu().numpy()
                ax.imshow(disp, cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([]); ax.set_yticks([])
                if title: ax.set_title(title, fontsize=9)
                if ylabel: ax.set_ylabel(ylabel, fontsize=9)
            show(axes[2*k, c], x_t, ylabel=("noised" if c == 0 else None),
                 title=(f"t={tv}" if k == 0 else None))
            show(axes[2*k+1, c], x0_hat, ylabel=("x̂0" if c == 0 else None))
    fig.suptitle("one-shot denoise: per digit, NOISED input (top) -> reconstruction x̂0 (bottom), across t",
                 fontsize=10)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(ckpt_path), "denoise_reconstruct.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"  saved grid -> {out}")
    print("  Open it: the top row of each pair dissolves into noise as t grows (the forward")
    print("  process); the bottom row is the net's x̂0 — a recognizable digit at low/mid t,")
    print("  degrading only when the noise has erased the signal. This is the denoiser WORKING.\n")
    print("  LAYER 6 COMPLETE — and with it the denoiser walkthrough. We have a trained")
    print("  eps_theta(x_t, t) that sees digits through noise. It can't yet GENERATE from")
    print("  scratch: one-shot x̂0 from pure noise (t=T) is mush, because it jumps in a single")
    print("  step. The next walkthrough (sampling.py) runs it ITERATIVELY — noise -> digit over")
    print("  many small steps — turning this denoiser into your first real image generator.")


def run_experiments():
    # exp_1a_why_t_is_an_input()
    # exp_1b_naive_encodings()
    # exp_1c_many_frequencies()
    # exp_1d_build_embedding()
    # exp_1e_sin_and_cos_rotation()
    # exp_1f_bounded_constant_norm()
    # exp_1g_similarity_kernel()
    # exp_2a_plain_mlp()
    # exp_2b_fuse_time()
    # exp_2c_assemble()
    # exp_3_loss_and_step()
    # exp_4_data()
    # exp_5_train_loop()
    exp_6_visualize()
    # exp_3_loss_and_step()
    # exp_4_data()
    # exp_5_train_loop()
    # exp_6_visualize()


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
