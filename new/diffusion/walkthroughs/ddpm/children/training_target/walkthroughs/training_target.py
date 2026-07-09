"""
CHILD WALKTHROUGH (digs into ddpm exp_3): the TRAINING TARGET, top-down.

The parent ddpm.py trained a U-Net and sampled digits from noise. Its whole train loop was three
lines: noise a batch, ask the net for the noise, minimize MSE. This box opens the middle line —
WHAT does the net learn to output, and why is that the right thing to regress on?

The one sentence everything here rests on:

    given a noised image x_t and its noise level t, PREDICT THE NOISE ε that was added.
    loss = MSE(ε̂, ε).      ε̂ = net(x_t, t),   ε ~ N(0, I)

That's it — diffusion training is a plain supervised regression. No sampling, no reverse process,
no special loss. This child makes that concrete and then answers the "why"s hiding in it.

Layers (each an `exp_*`; run it, read the output, then say "next"):
  1. the WHOLE GAME    — assemble a real training batch (x0 -> per-example t -> ε -> x_t), then two
                         wiring checks you can SEE: an untrained net scores loss ≈ 1.0, and letting a
                         tiny net memorize ONE batch drives the loss -> 0. Rough narration only.  (here)
  then open the boxes — each a "why" about that picture:
  2. WHY ≈ 1 UNTRAINED  — the do-nothing floor: predicting all-zeros scores E‖ε‖² = Var(ε) = 1, so
                         "1" is the baseline and anything below it is genuinely learned noise.
  3. ε OR x0?           — the target isn't unique: given (x_t, t), ε and x0 pin each other down via
                         the closed form. Isolate ε = (x_t - √ᾱ·x0)/√(1-ᾱ); recover it to ~1e-7.
  4. WHY ε WINS         — ε ~ N(0,1) at EVERY t = a scale-stable O(1) target; MSE_x0 swings wildly
                         across t while MSE_ε stays flat (MSE_ε = SNR·MSE_x0). This is the reason.
  5. per-t DIFFICULTY   — even with the ε target some t are intrinsically harder; ties back to the
                         SNR curve from forward_process exp_7 and explains the front-loaded loss.

Content mirrors the older bottom-up diffusion/ Layer-4 (isolate ε, ε-vs-x0) + the loss/wiring bits
of denoiser_and_loss, re-sequenced top-down as a child dig-in. No torchvision; shared MNIST at
new/diffusion/data/mnist.npz.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))                 # .../training_target/walkthroughs
# walk up to the shared new/diffusion/ root (holds data/):
#   walkthroughs -> training_target -> children -> ddpm -> walkthroughs -> diffusion
_DIFF = os.path.abspath(os.path.join(_HERE, *([".."] * 5)))        # new/diffusion
_FIGS = os.path.join(_HERE, "figures", "experiments")


def _banner(*lines):
    print("=" * 70)
    for line in lines:
        print(line)
    print("=" * 70)


def _device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _to_img(x):
    """(1,28,28)-ish tensor in [-1,1] -> HxW numpy in [0,1] for imshow."""
    return ((x.squeeze().clamp(-1, 1) + 1) / 2).cpu().numpy()


def _mnist(train=True):
    """MNIST images (N,1,28,28) in [-1,1], from the cached npz. No torchvision."""
    import numpy as np
    path = os.path.join(_DIFF, "data", "mnist.npz")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import urllib.request
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        print(f"  downloading MNIST npz (~11MB) -> {path} ...")
        urllib.request.urlretrieve(url, path)
    d = np.load(path)
    x = d["x_train"] if train else d["x_test"]                     # (N,28,28) uint8 [0,255]
    return (torch.from_numpy(x).float() / 127.5 - 1.0).unsqueeze(1)  # (N,1,28,28) in [-1,1]


def make_linear_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    """The SAME linear DDPM schedule the parent trained on. Returns (betas, alphas, alpha_bars),
    each shape (T,). ᾱ_t = ∏_{s≤t} α_s = how much original signal survives to step t."""
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def timestep_embedding(t, dim):
    """Sinusoidal embedding of the integer timestep t (B,) -> (B, dim). (Built here so this box
    stands alone; the WHY-a-net-needs-t question is the exp_4 denoiser box, not this one.)"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _TinyDenoiser(nn.Module):
    """A minimal (x_t, t) -> ε̂ regressor, just enough to run the wiring checks. Flatten the image,
    fold in the timestep embedding, a couple of hidden layers, back to 784. NOT the real model —
    the U-Net and why it's shaped that way is the exp_4 box. Output is BARE (no activation): ε is
    unbounded ~N(0,1), so squashing it would be wrong."""

    def __init__(self, dim=784, hidden=512, temb_dim=128):
        super().__init__()
        self.temb_dim = temb_dim
        self.in_proj = nn.Linear(dim, hidden)
        self.time_mlp = nn.Sequential(nn.Linear(temb_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.h1 = nn.Linear(hidden, hidden)
        self.h2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, dim)

    def forward(self, x, t):
        B = x.shape[0]
        h = self.in_proj(x.view(B, -1))
        h = h + self.time_mlp(timestep_embedding(t, self.temb_dim))   # inject t (ADD, like the U-Net)
        h = F.silu(self.h1(F.silu(h)))
        h = F.silu(self.h2(h))
        return self.out(h).view_as(x)


def make_training_pair(x0, alpha_bars, T):
    """Assemble ONE training example/batch exactly as the parent train loop does.

        x0    (B,1,28,28) clean images in [-1,1]
        t     (B,)        a RANDOM noise level PER example  -> one batch spans many difficulties
        ε     (B,1,28,28) fresh Gaussian noise, ε ~ N(0, I)
        x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε                          (the forward closed form, from exp_2)

    Returns (x_t, t, ε). The net will see only (x_t, t); ε is the label it must predict. Note the
    ᾱ reshape to (B,1,1,1) so it broadcasts over the pixels — the classic forgotten-reshape bug."""
    B = x0.shape[0]
    t = torch.randint(0, T, (B,), device=x0.device)
    eps = torch.randn_like(x0)
    ab = alpha_bars[t].view(B, 1, 1, 1)                            # (B,1,1,1) broadcast over pixels
    x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    return x_t, t, eps


# ---------------------------------------------------------------------------
# LAYER 1 (the whole game): SEE that the training target is just ε, and that the objective is real.
#
# Two wiring checks, both a single number you can read:
#   (A) an UNTRAINED net scores loss ≈ 1.0  — it hasn't learned anything, and 1.0 is the "predict
#       nothing" floor (why exactly 1 is the exp_2 box).
#   (B) let a tiny net MEMORIZE one fixed batch (overfit) — loss collapses toward 0. If the target
#       were miswired (wrong ε, wrong reshape, squashed output), this could NOT happen. Watching
#       loss go 1 -> ~0 on one batch is the standard "is my training code even correct?" test.
# Everything else in this child (why ε and not x0, why 1, per-t difficulty) is a "why" about this.
# ---------------------------------------------------------------------------
def exp_1_whole_game(seed=0, T=1000, batch_size=16, steps=600, lr=1e-3):
    """The whole game of the training target: assemble a real (x_t, t) -> ε batch, show an untrained
    net scores ≈ 1.0, then overfit that one batch to ~0. No derivations yet — see the objective work,
    get the map. exp_2..exp_5 open each box."""
    _banner("LAYER 1: the whole game — the target is ε, and MSE(ε̂, ε) actually trains")

    torch.manual_seed(seed)
    dev = _device()
    _, _, alpha_bars = make_linear_schedule(T=T)
    alpha_bars = alpha_bars.to(dev)

    print("  the training target, in one breath:")
    print("    given a noised image x_t and its level t, PREDICT THE NOISE ε that was added.")
    print("    loss = MSE(ε̂, ε).   that's it — plain supervised regression, no reverse process.\n")

    # ---- assemble a real training batch, the way the parent train loop does ------------------
    x0 = _mnist(train=True)[:batch_size].to(dev)                   # (B,1,28,28)
    x_t, t, eps = make_training_pair(x0, alpha_bars, T)
    print("  one training batch, assembled:")
    print(f"    x0   {tuple(x0.shape)}   clean digits in [-1,1]")
    print(f"    t    {tuple(t.shape)}      a random level per example: {t.tolist()[:8]}...")
    print(f"    ε    {tuple(eps.shape)}   the LABEL, ε~N(0,1): mean {eps.mean():+.3f}, var {eps.var():.3f}")
    print(f"    x_t  {tuple(x_t.shape)}   = √ᾱ_t·x0 + √(1-ᾱ_t)·ε  (net sees only x_t and t)\n")

    # ---- (A) untrained net: the do-nothing floor --------------------------------------------
    net = _TinyDenoiser().to(dev)
    with torch.no_grad():
        eps_hat = net(x_t, t)                                      # what the fresh net actually outputs
        loss0 = F.mse_loss(eps_hat, eps).item()
    zeros_loss = F.mse_loss(torch.zeros_like(eps), eps).item()     # predict nothing at all
    print("  (A) wiring check — an UNTRAINED net has learned nothing:")
    # WHY it matches the floor: look at what it outputs. At init the weights are tiny, so the
    # output is ≈0 — centered at 0 with a spread ~30x smaller than ε's. It IS the zero-predictor.
    print(f"        untrained output      : mean {eps_hat.mean():+.4f}, std {eps_hat.std():.4f}   (≈ 0 — outputs almost nothing)")
    print(f"        untrained loss vs ε   = {loss0:.4f}")
    print(f"        predict-all-zeros loss= {zeros_loss:.4f}   (= E‖ε‖² = Var(ε) ≈ 1 — the floor)")
    print("        outputs ≈0  ⇒  loss ≈ Var(ε) = 1. (a constant c scores c²+1, so ONLY c=0 hits 1; exp_2.)\n")

    # ---- (B) overfit ONE batch: proof the objective is learnable ----------------------------
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    losses = []
    print(f"  (B) wiring check — MEMORIZE this one batch ({batch_size} imgs, {steps} steps):")
    for step in range(steps):
        loss = F.mse_loss(net(x_t, t), eps)                       # SAME fixed batch every step
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % (steps // 6) == 0 or step == steps - 1:
            print(f"        step {step:>4}: loss {loss.item():.2e}")
    print(f"    1.0 -> {losses[-1]:.1e}: the net drove MSE(ε̂, ε) to ~0 by fitting ε on this batch.")
    print("    that can only happen if the target is wired right — the standard sanity test.\n")

    # ---- the payoff figure: watch the loss fall 1 -> 0 --------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(losses, color="#3b6", lw=1.8, label="overfit one batch: MSE(ε̂, ε)")
    ax.axhline(1.0, color="#888", ls="--", lw=1.2, label="untrained floor ≈ Var(ε) = 1.0")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("loss = MSE(ε̂, ε)  (log)")
    ax.set_title("the training target is ε — and the objective is learnable\n"
                 "(overfit one batch: loss 1.0 → ~0, the wiring check)", fontsize=10)
    ax.set_yscale("log"); ax.legend(frameon=False)
    fig.tight_layout()
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "01_wiring.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} — the loss curve for the wiring check.")
    print("  That's the whole game of the target: predict ε, minimize MSE, and it trains. Next")
    print("  (exp_2): why the untrained floor is EXACTLY 1 — the predict-nothing baseline.")


# ---------------------------------------------------------------------------
# LAYER 2 (why ≈ 1): the do-nothing floor is Var(target). Make it exact, then measure it.
#
# exp_1 showed an untrained net outputs ≈0 and scores ≈1. Both facts come from one little theorem:
# the best CONSTANT you can predict for a target Y is its mean, and the loss you're left with is its
# variance. Nothing about diffusion — it's the "predict the average" baseline every regression has.
#
#   DERIVATION  — best constant c for target Y, under squared error:
#     E[(c - Y)²] = E[c² - 2cY + Y²]                       expand
#                 = c² - 2c·E[Y] + E[Y²]                   linearity of E
#                 = c² - 2c·μ + (σ² + μ²)                  E[Y²] = Var + mean² = σ² + μ²
#                 = (c - μ)² + σ²                          complete the square   (μ=E[Y], σ²=Var Y)
#     minimized at c = μ, with minimum value σ².     →  best constant = the MEAN, its loss = the VARIANCE.
#
#   For the diffusion target Y = ε ~ N(0, I):  μ = 0,  σ² = 1   →   the floor is exactly 1.
#   (And ε is unit-variance at EVERY t, so this floor is a flat 1 across all noise levels — the thing
#    that makes ε such a convenient target. The contrast with a Var(x0) that MOVES with t is exp_4.)
# ---------------------------------------------------------------------------
def exp_2_why_one(seed=0, n=8192):
    """Why the untrained floor is exactly 1: predicting the best constant for ε costs Var(ε). Sweep
    the constant c, watch MSE(c, ε) trace the parabola (c-μ)²+σ², bottoming out at c=mean=0 with
    value var=1. Anything a real net scores BELOW 1 is learned structure."""
    _banner("LAYER 2: why ≈ 1 — the do-nothing floor is Var(target), and for ε that's 1")

    torch.manual_seed(seed)
    eps = torch.randn(n, 1, 28, 28)                                # a big pile of the target ε ~ N(0,I)
    mu, var = eps.mean().item(), eps.var().item()
    print("  the target here is ε ~ N(0, I):")
    print(f"    measured  mean(ε) = {mu:+.4f}   var(ε) = {var:.4f}   (→ theory says floor = var = 1)\n")

    # ---- sweep the constant predictor c and watch the parabola ------------------------------
    cs = torch.linspace(-1.0, 1.0, 41)
    losses = torch.tensor([F.mse_loss(torch.full_like(eps, c.item()), eps).item() for c in cs])
    parabola = (cs - mu) ** 2 + var                                # the derived (c-μ)²+σ²
    i_min = int(torch.argmin(losses))
    print("  predict a CONSTANT c for every pixel, measure MSE(c, ε):")
    print("      c      MSE(c,ε)   (c-μ)²+σ²")
    for c, m, p in list(zip(cs, losses, parabola))[::8]:           # every 8th row
        mark = "  <- min" if abs(c.item() - cs[i_min].item()) < 1e-6 else ""
        print(f"    {c.item():+.2f}    {m.item():.4f}     {p.item():.4f}{mark}")
    print(f"    best constant c* = {cs[i_min].item():+.2f}  (= mean ε)   loss there = {losses[i_min].item():.4f}  (= var ε = the floor)\n")

    # ---- what "below the floor" means --------------------------------------------------------
    parent_loss = 0.0235                                           # the parent's trained loss (ddpm exp_1)
    print("  so the floor isn't hard-coded — it's what 'predict the average' costs:")
    print(f"    a trained net that scores {parent_loss} explains 1 - {parent_loss}/{var:.2f} = "
          f"{1 - parent_loss / var:.1%} of ε's variance (an R²).")
    print("    every value BELOW 1 is genuinely-learned noise structure; at/above 1 = no better than guessing.\n")

    # ---- payoff figure: the parabola, its minimum, the 'learning' region --------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.axhspan(0, var, color="#3b6", alpha=0.08)
    ax.text(0.0, var * 0.5, "below the floor =\nlearned structure", ha="center", va="center",
            fontsize=9, color="#2a7")
    ax.plot(cs, parabola, color="#888", lw=1.4, label="(c − μ)² + σ²  (derived)")
    ax.scatter(cs, losses, s=16, color="#36c", zorder=3, label="measured  MSE(c, ε)")
    ax.axhline(var, color="#c33", ls="--", lw=1.2, label=f"floor = Var(ε) = {var:.2f}")
    ax.scatter([cs[i_min]], [losses[i_min]], s=70, color="#c33", zorder=4,
               marker="v", label=f"best constant  c*=mean(ε)={cs[i_min].item():.2f}")
    ax.set_xlabel("constant prediction c"); ax.set_ylabel("loss = MSE(c, ε)")
    ax.set_title("the do-nothing floor: best constant = mean(ε) = 0, loss = Var(ε) = 1\n"
                 "(any constant other than the mean scores ABOVE 1)", fontsize=10)
    ax.set_ylim(bottom=0); ax.legend(frameon=False, fontsize=8, loc="upper center")
    fig.tight_layout()
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "02_floor.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} — the loss parabola, its minimum at the mean, the floor at Var(ε).")
    print("  The floor = Var(target). Next (exp_3): the target isn't even unique — given (x_t, t) we")
    print("  could regress on x0 instead of ε, and the two pin each other down.")


# ---------------------------------------------------------------------------
# LAYER 3 (ε or x0?): the target isn't unique. The closed form is ONE linear equation tying three
# quantities at a given t, so fixing any two pins the third:
#
#     x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε
#
#   solve for ε :   ε  = (x_t − √ᾱ_t · x0) / √(1-ᾱ_t)
#   solve for x0:   x0 = (x_t − √(1-ᾱ_t) · ε) / √ᾱ_t
#
# The net only ever sees (x_t, t). Given that, x0 and ε determine each other EXACTLY — so "predict ε"
# and "predict x0" are the SAME job in two coordinate systems. A net trained on one can be converted
# to the other analytically, no retraining. (Which one to actually pick is exp_4 — they are NOT equally
# easy to learn, even though they're interchangeable here.)
# ---------------------------------------------------------------------------
def exp_3_eps_or_x0(seed=0, T=1000, batch_size=64):
    """The target isn't unique: given (x_t, t), ε and x0 pin each other down. Recover each from the
    other with the rearranged closed form and confirm it's exact (~1e-6). Then SEE it: from x_t and the
    true ε, reconstruct the clean digit at every t."""
    _banner("LAYER 3: ε or x0? — one closed form, two interchangeable targets")

    torch.manual_seed(seed)
    dev = _device()
    _, _, alpha_bars = make_linear_schedule(T=T)
    alpha_bars = alpha_bars.to(dev)

    # ---- numeric check: recover ε from x0, and x0 from ε, over a real batch -------------------
    x0 = _mnist(train=False)[:batch_size].to(dev)
    x_t, t, eps = make_training_pair(x0, alpha_bars, T)
    ab = alpha_bars[t].view(-1, 1, 1, 1)
    eps_from_x0 = (x_t - ab.sqrt() * x0) / (1 - ab).sqrt()          # solve the closed form for ε
    x0_from_eps = (x_t - (1 - ab).sqrt() * eps) / ab.sqrt()         # ...and for x0
    print("  the closed form x_t = √ᾱ·x0 + √(1-ᾱ)·ε ties (x_t, x0, ε): fix two, the third is pinned.")
    print("  the net sees only (x_t, t) — given that, ε and x0 determine each other. recover each:")
    print(f"    ε  from (x_t, x0):  max|ε̂ − ε|   = {(eps_from_x0 - eps).abs().max().item():.2e}")
    print(f"    x0 from (x_t, ε) :  max|x̂0 − x0| = {(x0_from_eps - x0).abs().max().item():.2e}")
    print("    both ~machine-zero → the two targets carry the SAME information (interchangeable).\n")
    print("  so 'predict ε' vs 'predict x0' is a choice of PARAMETRIZATION, not of information; a net")
    print("  trained on one converts to the other with these formulas. WHICH to pick (they aren't equally")
    print("  easy to learn) is exp_4.\n")

    # ---- see it: from x_t and the true ε, rebuild the clean digit at every t -----------------
    digit = _mnist(train=False)[7:8].to(dev)                        # one held-out digit
    ts = [50, 100, 200, 400, 600, 800]
    e = torch.randn_like(digit)                                     # ONE noise field, mixed in more each col
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, cols = 3, len(ts)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.15, rows * 1.25))
    row_labels = ["x0  (clean)", "x_t  (noised)", "x̂0 from (x_t, ε)"]
    for j, ti in enumerate(ts):
        abj = alpha_bars[ti]
        x_ti = abj.sqrt() * digit + (1 - abj).sqrt() * e
        x0_hat = (x_ti - (1 - abj).sqrt() * e) / abj.sqrt()         # recover x0 from the TRUE ε
        for i, img in enumerate([digit, x_ti, x0_hat]):
            ax = axes[i, j]
            ax.imshow(_to_img(img), cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"t={ti}", fontsize=9)
            if j == 0:
                ax.set_ylabel(row_labels[i], fontsize=9)
    fig.suptitle("given x_t and the noise ε, the clean image is PINNED:  x̂0 = (x_t − √(1-ᾱ)·ε)/√ᾱ\n"
                 "exact recovery at every t → ε and x0 are the same information", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "03_eps_or_x0.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} — bottom row (recovered x0) matches the top row (clean) at every t.")
    print("  (Recovery is EXACT because ε is the true noise; with a net's ε̂ the 1/√ᾱ factor amplifies")
    print("   its error at high t — a sampling concern, not a target one.)")
    print("  Next (exp_4): ε and x0 are interchangeable, but NOT equally learnable — why we pick ε.")


def run_experiments():
    # exp_1_whole_game()
    # exp_2_why_one()
    exp_3_eps_or_x0()


if __name__ == "__main__":
    run_experiments()
