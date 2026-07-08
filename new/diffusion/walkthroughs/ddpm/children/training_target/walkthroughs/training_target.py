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
        loss0 = F.mse_loss(net(x_t, t), eps).item()
    zeros_loss = F.mse_loss(torch.zeros_like(eps), eps).item()     # predict nothing at all
    print("  (A) wiring check — an UNTRAINED net has learned nothing:")
    print(f"        untrained loss        = {loss0:.4f}")
    print(f"        predict-all-zeros loss= {zeros_loss:.4f}   (= E‖ε‖² = Var(ε) ≈ 1 — the floor)")
    print("        both ≈ 1.0: guessing gets you here. below 1 = real learning.  (why 1 exactly: exp_2)\n")

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


def run_experiments():
    exp_1_whole_game()


if __name__ == "__main__":
    run_experiments()
