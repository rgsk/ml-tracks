"""
WALKTHROUGH: ddpm (top-down) — train a diffusion model, watch it GENERATE MNIST digits from noise,
THEN open up each piece.

Top-down (see ../../roadmap.md): exp_1 trains a tiny time-conditioned U-Net to predict the noise added
to a digit, then SAMPLES — starts from pure Gaussian noise and iteratively denoises into a digit. You
watch digits appear from static, with only a rough narration. Later experiments open one box each:

  1. the whole game       — train the model, sample digits from noise, watch generation work. (here)
  2. the forward process  — how we add noise: x_t = √ᾱ·x0 + √(1-ᾱ)·ε, the schedule, √ = variance-preserving.
  3. the training target  — why predict ε (not the image); loss = MSE(ε̂, ε); wiring checks.
  4. the denoiser         — the U-Net (down/up + skips) and why the timestep t must be an input.
  5. sampling             — why generation is ITERATIVE (one-shot is mush); the DDPM reverse step.

The three ideas, in one breath: add noise on a schedule (forward) -> train a net to predict that noise
-> walk back from pure noise to a digit (reverse). Reuses the cached mnist.npz under new/diffusion/data.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))                 # new/diffusion/walkthroughs/ddpm
_DIFF = os.path.dirname(os.path.dirname(_HERE))                    # new/diffusion (holds data/)
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


# ---------------------------------------------------------------------------
# The pieces. Rough narration only — each gets its own experiment later.
#
#   schedule:  betas ramp up over T steps; ᾱ_t (cumulative) is how much ORIGINAL signal survives to
#              step t. Forward noising in one jump:  x_t = √ᾱ_t · x0 + √(1-ᾱ_t) · ε.  (exp_2)
#   model:     a tiny U-Net that takes a noisy image x_t AND the timestep t, and predicts the noise ε
#              that was added. Down/up with skip connections; t is injected so it knows how noisy the
#              input is. (exp_3 = why predict ε; exp_4 = why this architecture.)
#   sample:    start from pure noise x_T ~ N(0,I) and walk backwards one step at a time, subtracting a
#              bit of predicted noise each step, until a clean digit remains. (exp_5)
# ---------------------------------------------------------------------------
def make_schedule(T=1000, device="cpu"):
    """Linear DDPM schedule. Returns betas, alphas, alpha_bar (all shape (T,))."""
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


def timestep_embedding(t, dim):
    """Sinusoidal embedding of the integer timestep t (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _Block(nn.Module):
    """A residual block: (GroupNorm -> SiLU -> conv) twice, with the timestep added in the middle.
    GroupNorm + residual are the standard diffusion-U-Net minimum that makes training stable and
    samples crisp; the `why` is exp_4."""

    def __init__(self, cin, cout, temb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(temb_dim, cout)                            # inject the timestep
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    """Predicts the noise ε from (x_t, t). Down 28->14->7 (channels grow), up 7->14->28 with skip
    connections; the timestep t is injected into every block."""

    def __init__(self, base=32, temb_dim=128):
        super().__init__()
        self.temb_dim = temb_dim
        self.time_mlp = nn.Sequential(nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim))
        self.stem = nn.Conv2d(1, base, 3, padding=1)                     # 1 -> base, 28x28
        self.down1 = _Block(base, base, temb_dim)                        # 28
        self.down2 = _Block(base, base * 2, temb_dim)                    # 14 (after pool)
        self.down3 = _Block(base * 2, base * 4, temb_dim)                # 7  (after pool)
        self.mid = _Block(base * 4, base * 4, temb_dim)                  # 7
        self.up2 = _Block(base * 4 + base * 2, base * 2, temb_dim)       # 14 (concat down2 skip)
        self.up1 = _Block(base * 2 + base, base, temb_dim)               # 28 (concat down1 skip)
        self.out_norm = nn.GroupNorm(8, base)
        self.out = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x, t):
        temb = self.time_mlp(timestep_embedding(t, self.temb_dim))
        h1 = self.down1(self.stem(x), temb)                              # (B, base,   28, 28)
        h2 = self.down2(F.avg_pool2d(h1, 2), temb)                       # (B, 2base,  14, 14)
        h3 = self.down3(F.avg_pool2d(h2, 2), temb)                       # (B, 4base,   7,  7)
        m = self.mid(h3, temb)
        u = self.up2(torch.cat([F.interpolate(m, scale_factor=2, mode="nearest"), h2], 1), temb)   # 14
        u = self.up1(torch.cat([F.interpolate(u, scale_factor=2, mode="nearest"), h1], 1), temb)   # 28
        return self.out(F.silu(self.out_norm(u)))                        # (B, 1, 28, 28) predicted ε


@torch.no_grad()
def sample(model, n, T, betas, alphas, alpha_bar, device):
    """Ancestral DDPM sampling: start from pure noise, walk back T -> 0, one denoise step at a time."""
    model.eval()
    x = torch.randn(n, 1, 28, 28, device=device)
    for i in reversed(range(T)):
        t = torch.full((n,), i, device=device, dtype=torch.long)
        eps_hat = model(x, t)
        a, ab, b = alphas[i], alpha_bar[i], betas[i]
        mean = (x - (1 - a) / (1 - ab).sqrt() * eps_hat) / a.sqrt()      # DDPM posterior mean
        if i > 0:
            var = b * (1 - alpha_bar[i - 1]) / (1 - ab)                  # posterior variance
            x = mean + var.sqrt() * torch.randn_like(x)
        else:
            x = mean                                                     # last step: no added noise
    return x


def exp_1_whole_game(seed=0, epochs=24, batch_size=128, lr=2e-4, T=1000):
    """The whole game, top-down: train TinyUNet to predict the noise in a corrupted digit, then SAMPLE
    brand-new digits from pure noise and save the grid. No derivations yet — see it generate, get the
    map. exp_2..exp_5 open each piece (forward process, target, denoiser, sampling)."""
    _banner("EXP 1: the whole game — train a diffusion model, sample MNIST digits from pure noise")

    torch.manual_seed(seed)
    dev = _device()

    x0 = _mnist(train=True)                                             # (60000,1,28,28) in [-1,1]
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x0),
                                         batch_size=batch_size, shuffle=True)
    betas, alphas, alpha_bar = make_schedule(T, device=dev)

    model = TinyUNet().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print("  the pipeline, in one breath:")
    print(f"    forward:  x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε   (add noise on a {T}-step schedule)")
    print("    model:    TinyUNet(x_t, t) -> predicts the noise ε   (down 28->14->7, up with skips)")
    print("    reverse:  start at pure noise, subtract predicted noise step by step -> a digit")
    print(f"    params:   {n_params:,}\n")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}   # EMA weights -> crisper samples
    print(f"  training on {len(x0)} images ({dev}), {len(loader)} steps/epoch. loss = MSE(ε̂, ε):")
    for ep in range(1, epochs + 1):
        model.train()
        run = 0.0
        for (xb,) in loader:
            xb = xb.to(dev)
            t = torch.randint(0, T, (xb.shape[0],), device=dev)        # a random noise level per image
            eps = torch.randn_like(xb)
            ab = alpha_bar[t].view(-1, 1, 1, 1)
            xt = ab.sqrt() * xb + (1 - ab).sqrt() * eps                 # forward: noise the batch
            loss = F.mse_loss(model(xt, t), eps)                       # predict the noise
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():                                      # update the EMA copy
                for k, v in model.state_dict().items():
                    ema[k].mul_(0.999).add_(v.detach(), alpha=0.001) if ema[k].is_floating_point() else ema[k].copy_(v)
            run += loss.item()
        print(f"        epoch {ep:>2}: train loss {run / len(loader):.4f}")

    # ---- the payoff: generate digits from pure noise (with the EMA weights) ----------------
    print("\n  sampling 64 digits from pure noise (reverse process, T steps)...")
    model.load_state_dict(ema)                                          # sample from the averaged weights
    imgs = sample(model, 64, T, betas, alphas, alpha_bar, dev)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, cols = 8, 8
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.9, rows * 0.9))
    for ax, k in zip(axes.flat, range(rows * cols)):
        ax.imshow(_to_img(imgs[k]), cmap="gray")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("generated MNIST digits — sampled from pure noise by the trained diffusion model", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "01_samples.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} — 64 digits the model invented from static.")
    print("  That's the whole game: noise -> a trained denoiser -> new digits. Next (exp_2): open the")
    print("  first box — the forward process — and see exactly how a digit dissolves into noise.")


def run_experiments():
    exp_1_whole_game()


if __name__ == "__main__":
    run_experiments()
