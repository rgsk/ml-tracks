"""
WALKTHROUGH: sampling — turning the trained denoiser into a generator, one layer at a time.

denoiser_and_loss.py trained eps_theta(x_t, t) ~ eps and, in its Layer 6, showed it can
recover a clean digit from a NOISED one in a single jump (one-shot x0_hat) — as long as
the noise level isn't too high. But that isn't generation yet: it starts from a real
digit's noised image. To GENERATE we must start from PURE noise x_T ~ N(0,I) — a blank
canvas that contains no digit — and end at a brand-new digit the net dreamed up.

The naive idea "just one-shot x0_hat from x_T" fails: at t=T the signal is entirely gone,
the 1/√ab factor explodes, and you get mush (Layer 1 here makes that concrete). The fix is
the REVERSE PROCESS: don't jump: walk back one noise level at a time, x_T -> x_{T-1} -> ...
-> x_0, re-running the denoiser at every step so it can keep correcting course. Each step is
a small, easy denoise; a thousand of them compose into a clean image. THIS is the first real
generator — the artifact the whole track is aiming at.

The reverse step is a Gaussian:  x_{t-1} = mu_theta(x_t, t) + sigma_t · z,  z ~ N(0,I),
whose mean comes from the tractable forward posterior q(x_{t-1} | x_t, x0) with the net's
x0_hat plugged in for the (unknown) x0. Layer 2 derives and verifies it; Layer 3 asks what
the injected sigma_t·z is for; Layer 4 runs the loop and generates a grid of digits; Layer 5
watches a single chain crawl from noise to a digit.

Layers (run each `exp_*`, watch the output, then say "next"):
  1. WHY iterate, not one-shot — one-shot x0_hat from PURE noise (t=T) is mush; the reverse
     process replaces one impossible jump with many easy small steps.                    (here)
  2. the reverse step (the core math), in two bites:
       2a. the tractable posterior q(x_{t-1} | x_t, x0) — its closed-form mean mu~ and
           variance beta~; MC-verify them in 1-D.
       2b. swap the true x0 for the net's x0_hat -> the DDPM update; rearrange to the
           eps-form  x_{t-1} = (1/√alpha)(x_t - beta/√(1-ab)·eps_hat) + sigma_t·z.
  3. the injected noise sigma_t·z — ablate it: deterministic mean-only vs full stochastic
     ancestral sampling; what re-adding noise each step actually buys.
  4. the full sampler — the reverse loop from N(0,I) to a digit; generate a GRID of brand-new
     digits and save it. The first real generator.
  5. SEE the trajectory — snapshots of ONE chain x_T -> x_0, and the net's x0_hat estimate
     sharpening from mush into a digit as the walk proceeds.

Reuses denoiser_and_loss.py (the trained DenoiserMLP + checkpoint) and forward_process.py
(the schedule + closed form). Reference for the real torch version: diffusion/sample.py.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))   # diffusion/walkthroughs/
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)                         # so `import forward_process` etc. work


def _banner(*lines):
    print("=" * 70)
    for line in lines:
        print(line)
    print("=" * 70)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_trained_model(device):
    """Load the DenoiserMLP checkpoint trained in denoiser_and_loss.py Layer 5."""
    from denoiser_and_loss import DenoiserMLP, _ckpt_path
    path = _ckpt_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no checkpoint at {path} — run denoiser_and_loss.py exp_5_train_loop() first."
        )
    ckpt = torch.load(path, map_location=device)
    model = DenoiserMLP(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def _to_img(x):
    """(1,28,28)-ish tensor in [-1,1] -> HxW numpy in [0,1] for imshow."""
    return ((x.squeeze().clamp(-1, 1) + 1) / 2).cpu().numpy()


# ---------------------------------------------------------------------------
# LAYER 1: why iterate, not one-shot — one-shot x0_hat from PURE noise is mush.
#
# The denoiser gives us, at any t, a one-shot estimate of the clean image (forward_process
# 4a, solved for x0):
#
#     x0_hat = ( x_t - √(1-ab_t)·eps_hat ) / √ab_t
#
# Layer 6 of denoiser_and_loss showed this recovers the digit crisply at LOW/MID t and breaks
# down at HIGH t. Generation needs to start at the very TOP, t=T, from pure noise x_T ~ N(0,I)
# — exactly where it breaks down worst, and for two compounding reasons:
#
#   (1) x_T carries essentially NO signal. ab_T is ~1e-5 for the linear schedule, so
#       x_T = √ab_T·x0 + √(1-ab_T)·eps is ~99.99% noise — whatever x0 "underlies" it is
#       unrecoverable from one look. The net can only guess the population-average digit.
#   (2) the 1/√ab_T factor is HUGE (ab_T~4e-5 so 1/√ab_T ≈ 160). Any error in eps_hat gets
#       amplified ~160x in x0_hat, so even a tiny mistake blows the estimate far outside [-1,1].
#
# So a single jump from noise to image is hopeless. The reverse process replaces that one
# impossible jump with a LADDER of easy ones: from x_t, don't leap all the way to x0 — take
# one small step to x_{t-1} (a slightly-less-noisy image), then RE-RUN the denoiser there,
# where ab is a bit larger and the job a bit easier, and repeat. Each rung is a gentle
# denoise the net is good at; composing ~T of them walks pure noise down into a real digit.
# This bite shows the one-shot-from-noise mush (numbers + a picture) to motivate the ladder.
# ---------------------------------------------------------------------------
def exp_1_why_iterate(seed=0):
    """Take the trained net, start from PURE noise x_T ~ N(0,I), and one-shot x0_hat at a
    ladder of t. Show the estimate is fine at low t but explodes into mush as t -> T (signal
    gone AND 1/√ab amplifies error), and save a picture of the t=T one-shot next to a real
    digit. Motivates the iterative reverse process (Layers 2-4)."""
    _banner("LAYER 1: why iterate — one-shot x0_hat from PURE noise (t=T) is mush")

    from forward_process import make_linear_schedule
    device = _device()
    T = 1000
    _, _, alpha_bars = make_linear_schedule(T=T)
    alpha_bars = alpha_bars.to(device)
    model = _load_trained_model(device)

    # start from PURE noise — the blank canvas generation must begin from. No x0 anywhere.
    torch.manual_seed(seed)
    B = 512
    x_T = torch.randn(B, 1, 28, 28, device=device)             # ~ N(0,I): the top of the ladder

    # one-shot x0_hat = (x_t - √(1-ab)·eps_hat)/√ab, PRETENDING this pure noise sits at level t.
    # (For a real forward image, x_t would be √ab·x0 + √(1-ab)·eps; here there is no x0 — we are
    #  asking the net to hallucinate one straight from noise, which is the generation task.)
    def one_shot_stats(tv):
        t = torch.full((B,), tv, device=device)
        ab = alpha_bars[tv]
        with torch.no_grad():
            eps_hat = model(x_T, t)
        x0_hat = (x_T - torch.sqrt(1 - ab) * eps_hat) / torch.sqrt(ab)
        frac_out = (x0_hat.abs() > 1).float().mean().item()    # fraction outside the valid [-1,1]
        return ab.item(), x0_hat.abs().max().item(), x0_hat.std().item(), frac_out

    print("  from PURE noise x_T~N(0,I), one-shot x0_hat at each level t (valid pixels live in")
    print("  [-1,1]; a healthy estimate stays there):\n")
    print(f"  {'t':>5} | {'ab_t':>10} | {'1/√ab_t':>9} | {'max|x0_hat|':>11} | {'std':>7} | {'% outside[-1,1]':>15}")
    print(f"  {'-'*5}-+-{'-'*10}-+-{'-'*9}-+-{'-'*11}-+-{'-'*7}-+-{'-'*15}")
    for tv in [50, 200, 400, 600, 800, 999]:
        ab, mx, sd, fo = one_shot_stats(tv)
        print(f"  {tv:>5} | {ab:>10.2e} | {1/ab**0.5:>9.1f} | {mx:>11.1f} | {sd:>7.2f} | {fo:>14.0%}")
    print("    -> as t climbs, ab_t collapses toward ~4e-5, 1/√ab_t blows up past ~150, and the")
    print("       one-shot estimate explodes far outside [-1,1] — nearly every pixel invalid at")
    print("       t=T. A single jump from pure noise can't produce a digit.\n")

    # a picture: the t=T one-shot (clamped) next to a real digit. It's just noise.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from denoiser_and_loss import MNISTData

    t999 = torch.full((8,), 999, device=device)
    with torch.no_grad():
        eps_hat = model(x_T[:8], t999)
    x0_hat_T = (x_T[:8] - torch.sqrt(1 - alpha_bars[999]) * eps_hat) / torch.sqrt(alpha_bars[999])
    real = MNISTData(train=False).x[:8]                         # what a real digit looks like

    fig, axes = plt.subplots(2, 8, figsize=(8 * 1.4, 2 * 1.4))
    for j in range(8):
        axes[0, j].imshow(_to_img(x0_hat_T[j]), cmap="gray", vmin=0, vmax=1)
        axes[1, j].imshow(_to_img(real[j]), cmap="gray", vmin=0, vmax=1)
        for r in (0, 1):
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
    axes[0, 0].set_ylabel("one-shot\nfrom noise", fontsize=9)
    axes[1, 0].set_ylabel("real digit", fontsize=9)
    fig.suptitle("one-shot x0_hat from PURE noise at t=T (top) is mush, not a digit (bottom)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(_HERE, "outputs", "sampling_oneshot_mush.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out}")
    print("  Open it: the top row (one-shot from noise) is grey mush; there is no digit to find")
    print("  in a single leap from N(0,I).\n")
    print("  The fix is the REVERSE PROCESS: instead of one impossible jump x_T -> x0, take many")
    print("  small, easy steps x_T -> x_{T-1} -> ... -> x_0, re-running the denoiser at each level")
    print("  so it keeps correcting course. Next (Layer 2): the reverse STEP — the posterior")
    print("  q(x_{t-1}|x_t,x0), and the DDPM update you get by plugging the net's x0_hat into it.")


# ---------------------------------------------------------------------------
# LAYER 2a: the tractable forward posterior q(x_{t-1} | x_t, x0).
#
# Layer 1 said: don't jump x_t -> x0, step x_t -> x_{t-1}. What IS that step? If we knew the
# clean image x0, the exact distribution of the slightly-less-noisy x_{t-1} given both the
# current x_t and x0 is a Gaussian we can write in closed form — the "forward posterior". It
# is the reverse step's backbone; Layer 2b then removes the cheat of knowing x0.
#
# Two facts we already have are both Gaussian in x_{t-1}:
#   • ONE forward step:      q(x_t | x_{t-1}) = N(√alpha_t · x_{t-1},  beta_t · I)
#   • the closed form:       q(x_{t-1} | x0)  = N(√ab_{t-1} · x0,      (1-ab_{t-1}) · I)
# Bayes: q(x_{t-1} | x_t, x0) ∝ q(x_t | x_{t-1}) · q(x_{t-1} | x0). A Gaussian times a Gaussian
# (in x_{t-1}) is Gaussian, so the posterior is N(mu~_t, beta~_t · I) with (derivation below):
#
#     beta~_t = (1 - ab_{t-1}) / (1 - ab_t) · beta_t
#     mu~_t   = ( √ab_{t-1}·beta_t / (1-ab_t) )·x0  +  ( √alpha_t·(1-ab_{t-1}) / (1-ab_t) )·x_t
#
# Read mu~: it's a weighted blend of "where x0 says to go" and "where x_t already is". And
# beta~_t < beta_t: knowing BOTH endpoints x_t and x0 pins x_{t-1} down TIGHTER than a blind
# forward step would. This bite MC-verifies both formulas in 1-D by conditioning on x_t.
# ---------------------------------------------------------------------------
'''
DERIVATION — q(x_{t-1} | x_t, x0) = N(mu~_t, beta~_t) (Gaussian × Gaussian in the exponent).
Drop the two Gaussians' exponents and collect the x_{t-1} terms (write y = x_{t-1}):
    -1/2 [ (x_t - √alpha_t·y)² / beta_t  +  (y - √ab_{t-1}·x0)² / (1-ab_{t-1}) ]
The y² coefficient is  1/2·( alpha_t/beta_t + 1/(1-ab_{t-1}) ) = 1/2·(1/beta~_t), and using
alpha_t = 1-beta_t with ab_t = alpha_t·ab_{t-1} it collapses to
    1/beta~_t = alpha_t/beta_t + 1/(1-ab_{t-1}) = (1-ab_t) / ( beta_t·(1-ab_{t-1}) )
      =>  beta~_t = (1-ab_{t-1})/(1-ab_t) · beta_t.
The mean is (linear-in-y coefficient)·beta~_t:
    mu~_t = beta~_t·( √alpha_t/beta_t·x_t + √ab_{t-1}/(1-ab_{t-1})·x0 )
          = √alpha_t(1-ab_{t-1})/(1-ab_t)·x_t + √ab_{t-1}·beta_t/(1-ab_t)·x0.
So the posterior is fully determined by (x_t, x0) and the schedule — no network needed YET.
'''
def exp_2a_posterior(seed=0):
    """MC-verify the forward posterior q(x_{t-1}|x_t,x0) in 1-D. Simulate millions of chains
    x0 -> x_{t-1} -> x_t (fixed scalar x0), condition on x_t landing in a narrow band, and
    compare the empirical mean/std of x_{t-1} to the closed-form mu~_t and √beta~_t. Also show
    beta~_t < beta_t (observing both endpoints tightens x_{t-1})."""
    _banner("LAYER 2a: the tractable posterior q(x_{t-1} | x_t, x0) — mean mu~, variance beta~")

    import math
    from forward_process import make_linear_schedule
    T = 1000
    betas, alphas, alpha_bars = make_linear_schedule(T=T)

    t = 200                                                    # a mid noise level
    b_t = betas[t].item()
    a_t = alphas[t].item()
    ab_t = alpha_bars[t].item()
    ab_tm1 = alpha_bars[t - 1].item()                         # ab_{t-1}

    # closed-form posterior for a fixed scalar x0.
    x0 = 0.5
    beta_tilde = (1 - ab_tm1) / (1 - ab_t) * b_t
    # we will condition on x_t sitting at the CENTER of its q(x_t|x0) spread (most samples there).
    xt_star = math.sqrt(ab_t) * x0
    mu_tilde = (math.sqrt(ab_tm1) * b_t / (1 - ab_t)) * x0 \
        + (math.sqrt(a_t) * (1 - ab_tm1) / (1 - ab_t)) * xt_star

    # MC: x0 (fixed) -> x_{t-1} (closed form) -> x_t (ONE forward step), millions of chains.
    g = torch.Generator().manual_seed(seed)
    N = 4_000_000
    x_tm1 = math.sqrt(ab_tm1) * x0 + math.sqrt(1 - ab_tm1) * torch.randn(N, generator=g)
    x_t = math.sqrt(a_t) * x_tm1 + math.sqrt(b_t) * torch.randn(N, generator=g)

    std_xt = math.sqrt(1 - ab_t)
    band = 0.02 * std_xt                                      # keep chains whose x_t ~ xt_star
    mask = (x_t - xt_star).abs() < band
    sel = x_tm1[mask]                                         # x_{t-1} for those chains
    emp_mean, emp_std = sel.mean().item(), sel.std().item()

    print(f"  t={t}: beta_t={b_t:.5f}, alpha_t={a_t:.5f}, ab_t={ab_t:.4f}, ab_(t-1)={ab_tm1:.4f}\n")
    print(f"  condition on x_t ≈ {xt_star:+.4f} (center of q(x_t|x0), x0={x0}); {mask.sum().item():,} of")
    print(f"  {N:,} simulated chains land in the band. Distribution of x_(t-1) among them:\n")
    print(f"  {'':>18} | {'mean':>10} | {'std':>10}")
    print(f"  {'-'*18}-+-{'-'*10}-+-{'-'*10}")
    print(f"  {'MC empirical':>18} | {emp_mean:>+10.4f} | {emp_std:>10.4f}")
    print(f"  {'closed form':>18} | {mu_tilde:>+10.4f} | {beta_tilde**0.5:>10.4f}   (mu~, √beta~)")
    print()
    print(f"  the two components of mu~ (blend of x0-pull and x_t-position):")
    w_x0 = math.sqrt(ab_tm1) * b_t / (1 - ab_t)
    w_xt = math.sqrt(a_t) * (1 - ab_tm1) / (1 - ab_t)
    print(f"    mu~ = {w_x0:.4f}·x0 + {w_xt:.4f}·x_t   (a single step barely moves — x_t dominates; the")
    print(f"                                        tiny x0-pull is what accumulates over ~T steps into a digit)\n")
    print(f"  variance shrinks by conditioning on BOTH endpoints:")
    print(f"    blind forward-step var beta_t = {b_t:.5f}  ->  posterior var beta~_t = {beta_tilde:.5f}"
          f"  (beta~ < beta)\n")
    print("  Read it: the MC mean/std of x_(t-1) match mu~_t and √beta~_t — the closed-form")
    print("  posterior is exactly the distribution of the less-noisy image given (x_t, x0). This")
    print("  is the reverse STEP's target. Next (2b): we don't know x0 at sampling time — plug in")
    print("  the net's x0_hat and the update becomes the eps-form DDPM sampler.")


def run_experiments():
    # exp_1_why_iterate()
    exp_2a_posterior()
    # exp_2b_ddpm_update()
    # exp_3_injected_noise()
    # exp_4_sample_grid()
    # exp_5_trajectory()


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
