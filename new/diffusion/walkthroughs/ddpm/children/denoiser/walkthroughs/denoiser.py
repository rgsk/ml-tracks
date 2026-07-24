"""
CHILD WALKTHROUGH (digs into ddpm exp_4): the DENOISER, top-down.

The parent ddpm.py trained a U-Net and sampled digits from noise; the training_target box then
showed WHAT it learns (predict ε, loss = MSE(ε̂, ε)). This box opens the net itself — the thing
that computes ε̂ = net(x_t, t). Two questions hide in that call:

    WHY a U-Net (down/up with skip connections), and WHY must the timestep t be an input?

The one sentence everything here rests on:

    the denoiser is a plain IMAGE-TO-IMAGE map: (x_t, t) -> ε̂, same spatial size in and out.
    the only two non-obvious ingredients are SKIP CONNECTIONS and TIME CONDITIONING.

Top-down: before any "why", SEE the denoiser work — feed it a noised digit and a level, and watch
it output a noise map that, subtracted off, RECOVERS the clean digit. Run it with
`python denoiser.py` (`exp_1_whole_game`).

Layers (each an `exp_*`; run it, read the output, then say "next"):
  1. the WHOLE GAME    — the real TinyUNet as an image->image map: trace its shape flow
                         (28->14->7->14->28), count its params, then train it briefly and SEE it
                         denoise (x_t -> ε̂ -> recovered x̂0). Rough narration only.            (here)
  then open the boxes — each a "why" about that picture:
  2. WHY SKIPS          — ablate the skip connections: down/up alone throws away spatial detail, so
                         the recovered digit goes blurry; the skips carry the fine structure across.
  3. WHY DOWN/UP        — receptive field: pooling lets a small conv net SEE the whole 28x28 digit
                         (global shape) cheaply; a flat full-res stack can't reach that far.
  4. WHY t IS AN INPUT  — the SAME x_t means different things at different noise levels; drop the
                         time conditioning and the net must predict a blur-average -> loss worsens.
  5. HOW t ENTERS       — sinusoidal embedding -> small MLP -> ADDED into every block (FiLM-lite);
                         why sinusoidal, why inject everywhere instead of once at the input.
  6. THE BLOCK          — the residual GroupNorm->SiLU->conv unit: why these are the modern default
                         (stable training, crisp samples) over BatchNorm / ReLU / plain conv stacks.

Content re-sequences the old bottom-up diffusion/ denoiser pieces top-down as a child dig-in. The
model here is the REAL parent TinyUNet (this box's whole job is to open it). No torchvision; shared
MNIST at new/diffusion/data/mnist.npz.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))                 # .../denoiser/walkthroughs
# walk up to the shared new/diffusion/ root (holds data/):
#   walkthroughs -> denoiser -> children -> ddpm -> walkthroughs -> diffusion
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


def make_training_pair(x0, alpha_bars, T):
    """Assemble ONE training batch exactly as the parent train loop does (see the training_target box).
        x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε,   t random per example,   ε~N(0,I) is the label. Returns (x_t,t,ε)."""
    B = x0.shape[0]
    t = torch.randint(0, T, (B,), device=x0.device)
    eps = torch.randn_like(x0)
    ab = alpha_bars[t].view(B, 1, 1, 1)                            # (B,1,1,1) broadcast over pixels
    x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
    return x_t, t, eps


# ===========================================================================
# THE REAL MODEL — this is the box we are opening. It is the parent ddpm.py's TinyUNet, verbatim.
# exp_1 only USES it (shape flow + it denoises); every named piece below gets its own "why" later:
#   sinusoidal timestep_embedding + time_mlp .......... how t enters   (exp_5)
#   _Block = GroupNorm->SiLU->conv ×2, +temb, residual . the block     (exp_6), how t enters (exp_5)
#   down1/2/3 + avg_pool, up2/up1 + interpolate ........ why down/up    (exp_3)
#   torch.cat([..., h2/h1], 1) skip concats ............ why skips      (exp_2)
# ===========================================================================
def timestep_embedding(t, dim):
    """Sinusoidal embedding of the integer timestep t (B,) -> (B, dim). (WHY sinusoidal: exp_5.)"""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class _Block(nn.Module):
    """Residual block: (GroupNorm -> SiLU -> conv) twice, with the timestep ADDED in the middle.
    GroupNorm + residual are the standard diffusion-U-Net minimum; the `why` is exp_6."""

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
    connections; the timestep t is injected into every block. Output is BARE (no activation) — ε is
    unbounded ~N(0,1). `use_skips=False` zeroes the two skip highways (the exp_2 ablation)."""

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

    def forward(self, x, t, use_skips=True):
        temb = self.time_mlp(timestep_embedding(t, self.temb_dim))
        h1 = self.down1(self.stem(x), temb)                              # (B, base,   28, 28)
        h2 = self.down2(F.avg_pool2d(h1, 2), temb)                       # (B, 2base,  14, 14)
        h3 = self.down3(F.avg_pool2d(h2, 2), temb)                       # (B, 4base,   7,  7)
        m = self.mid(h3, temb)
        s2 = h2 if use_skips else torch.zeros_like(h2)                   # the two skip highways:
        s1 = h1 if use_skips else torch.zeros_like(h1)                   #   off -> the cats see zeros
        u = self.up2(torch.cat([F.interpolate(m, scale_factor=2, mode="nearest"), s2], 1), temb)   # 14
        u = self.up1(torch.cat([F.interpolate(u, scale_factor=2, mode="nearest"), s1], 1), temb)   # 28
        return self.out(F.silu(self.out_norm(u)))                        # (B, 1, 28, 28) predicted ε


class FlatNet(nn.Module):
    """The ablation for 'why down/up': same DEPTH as TinyUNet (stem + 6 _Blocks + out = 14 convs) but
    NO pooling — every block runs at full 28x28, channels held at `base`. Same conv COUNT means the same
    theoretical receptive field, yet with no pool the EFFECTIVE receptive field grows only ~√depth, so a
    center output pixel never really sees the whole digit (measured in exp_3). Callable as net(x, t)."""

    def __init__(self, base=32, temb_dim=128):
        super().__init__()
        self.temb_dim = temb_dim
        self.time_mlp = nn.Sequential(nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim))
        self.stem = nn.Conv2d(1, base, 3, padding=1)
        self.blocks = nn.ModuleList([_Block(base, base, temb_dim) for _ in range(6)])  # 6 blocks, all 28x28
        self.out_norm = nn.GroupNorm(8, base)
        self.out = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x, t):
        temb = self.time_mlp(timestep_embedding(t, self.temb_dim))
        h = self.stem(x)
        for b in self.blocks:
            h = b(h, temb)                                               # stays 28x28 the whole way
        return self.out(F.silu(self.out_norm(h)))


def _recover_x0(x_t, eps_hat, ab):
    """Invert the forward closed form for x0 given a noise estimate: x̂0 = (x_t - √(1-ᾱ)·ε̂)/√ᾱ.
    ab is ᾱ_t reshaped to broadcast over pixels. (This is exactly the algebra the sampler uses.)"""
    return (x_t - (1 - ab).sqrt() * eps_hat) / ab.sqrt()


def _hf(e):
    """High-frequency-energy proxy: the variance LEFT after a 3x3 local blur. A grainy ε̂ (fine
    pixel detail, like real ε) scores high; a smooth low-frequency ghost scores near 0."""
    return (e - F.avg_pool2d(e, 3, 1, 1)).pow(2).mean().item()


def _effective_rf(make_net, dev, seed, n_avg=8, size=28):
    """Measure the EFFECTIVE receptive field of the CENTER output pixel: how much each input pixel
    actually influences it. For fresh random nets (RF is an ARCHITECTURE property, no training needed)
    we push a random image through, backprop from output[center], and accumulate |∂out_center/∂input|.
    Averaging over n_avg random (net, input) draws smooths the map (the standard ERF recipe). Returns
    (mean influence map (size,size), effective radius = RMS spread in px, coverage = frac pixels >1% max)."""
    c = size // 2
    accum = torch.zeros(size, size)
    for k in range(n_avg):
        torch.manual_seed(seed + k)
        net = make_net().to(dev).eval()
        x = torch.randn(1, 1, size, size, device=dev, requires_grad=True)  # normal operating regime
        t = torch.randint(0, 1000, (1,), device=dev)
        net.zero_grad(set_to_none=True)
        net(x, t)[0, 0, c, c].backward()                                # influence of every input pixel
        accum += x.grad.detach().abs()[0, 0].cpu()
    g = accum / n_avg
    p = g / g.sum()                                                     # normalize to a distribution
    ys, xs = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    dist2 = (ys - c).float() ** 2 + (xs - c).float() ** 2
    eff_radius = (p * dist2).sum().sqrt().item()                        # RMS distance of influence
    coverage = (g > 0.01 * g.max()).float().mean().item()              # frac of pixels that matter
    return g, eff_radius, coverage


def _quick_train(net, x0, alpha_bars, T, steps, batch_size, lr, seed, use_skips=True):
    """Train a denoiser from scratch on random (x_t,t)->ε batches and return its final train loss.
    `use_skips` is threaded into every forward so we can train a net that NEVER gets the skips."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    last = float("nan")
    for _ in range(steps):
        idx = torch.randint(0, x0.shape[0], (batch_size,), device=x0.device)
        x_t, t, eps = make_training_pair(x0[idx], alpha_bars, T)
        loss = F.mse_loss(net(x_t, t, use_skips=use_skips), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = loss.item()
    net.eval()
    return last


# ---------------------------------------------------------------------------
# LAYER 1 (the whole game): SEE that the denoiser is just an image->image net, and that it works.
#
# Three things you can read/see:
#   (i)   it's an IMAGE-TO-IMAGE map — (B,1,28,28) in, (B,1,28,28) out; we trace the real shape flow
#         28->14->7->14->28 with a forward hook (measured, not just documented) and count the params.
#   (ii)  UNTRAINED it recovers nothing — ε̂≈0, so x̂0 ≈ x_t/√ᾱ is garbage (a number you can read).
#   (iii) train it briefly and it DENOISES — feed x_t, subtract the predicted ε̂, and the clean digit
#         reappears. The payoff figure is the row [clean x0 | noised x_t | predicted ε̂ | recovered x̂0].
# Everything else in this child (why skips, why down/up, why/how t) is a "why" about this picture.
# ---------------------------------------------------------------------------
def exp_1_whole_game(seed=0, T=1000, base=32, n_train=4000, batch_size=128, steps=300, lr=2e-4):
    """The whole game of the denoiser: the real TinyUNet is an image->image map (x_t,t)->ε̂. Trace its
    shape flow, count its params, then train it briefly and SEE it recover clean digits from noise. No
    derivations yet — see the net work, get the map. exp_2..exp_6 open each box (skips, down/up, t)."""
    _banner("LAYER 1: the whole game — the denoiser is an image->image net (x_t,t)->ε̂ that denoises")

    torch.manual_seed(seed)
    dev = _device()
    _, _, alpha_bars = make_linear_schedule(T=T)
    alpha_bars = alpha_bars.to(dev)

    print("  the denoiser, in one breath:")
    print("    it's a plain IMAGE-TO-IMAGE network: (x_t, t) -> ε̂, same 28x28 in and out.")
    print("    the only two non-obvious parts are SKIP CONNECTIONS and TIME CONDITIONING.\n")

    net = TinyUNet(base=base).to(dev)
    n_params = sum(p.numel() for p in net.parameters())

    # ---- (i) it's an image->image map: trace the real shape flow with forward hooks -----------
    x0_probe = _mnist(train=True)[:4].to(dev)
    x_t_probe, t_probe, _ = make_training_pair(x0_probe, alpha_bars, T)
    shapes = {}
    hooks = []
    for name in ["stem", "down1", "down2", "down3", "mid", "up2", "up1", "out"]:
        mod = getattr(net, name)
        hooks.append(mod.register_forward_hook(
            lambda m, i, o, name=name: shapes.__setitem__(name, tuple(o.shape))))
    with torch.no_grad():
        out_probe = net(x_t_probe, t_probe)
    for h in hooks:
        h.remove()
    print(f"  (i) an image-to-image map — in {tuple(x_t_probe.shape)}  ->  out {tuple(out_probe.shape)}"
          f"   (same spatial size), {n_params:,} params:")
    print(f"        stem  {shapes['stem']}   ┐ DOWN: pool 28->14->7, channels grow")
    print(f"        down1 {shapes['down1']}   │")
    print(f"        down2 {shapes['down2']}   │")
    print(f"        down3 {shapes['down3']}     ┘")
    print(f"        mid   {shapes['mid']}       bottleneck (whole digit in view)")
    print(f"        up2   {shapes['up2']}     ┐ UP: interpolate 7->14->28, concat the skip")
    print(f"        up1   {shapes['up1']}   ┘")
    print(f"        out   {shapes['out']}    bare ε̂ (no activation — ε is unbounded)\n")

    # ---- (ii) untrained: recovers nothing -----------------------------------------------------
    x0 = _mnist(train=True)[:n_train].to(dev)
    torch.manual_seed(seed + 1)
    # a fixed display batch at ONE moderate level, so "before vs after" is a fair comparison
    n_show = 8
    t_show_val = 250
    x0_show = x0[:n_show]
    ab_show = alpha_bars[t_show_val].view(1, 1, 1, 1)
    eps_show = torch.randn_like(x0_show)
    x_t_show = ab_show.sqrt() * x0_show + (1 - ab_show).sqrt() * eps_show
    t_show = torch.full((n_show,), t_show_val, device=dev, dtype=torch.long)

    @torch.no_grad()
    def recovery_mse():
        eps_hat = net(x_t_show, t_show)
        x0_hat = _recover_x0(x_t_show, eps_hat, ab_show)
        return F.mse_loss(x0_hat.clamp(-1, 1), x0_show).item(), eps_hat, x0_hat

    mse_before, eps_hat_before, x0_hat_before = recovery_mse()
    print(f"  (ii) UNTRAINED, at t={t_show_val}: it recovers nothing —")
    print(f"        ε̂ ≈ 0 (std {eps_hat_before.std():.3f}), so x̂0 = (x_t-√(1-ᾱ)ε̂)/√ᾱ is just noise.")
    print(f"        recover MSE(x̂0, x0) = {mse_before:.4f}   (high = garbage)\n")

    # ---- (iii) train it briefly and watch it denoise ------------------------------------------
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    print(f"  (iii) train the real U-Net briefly ({n_train} imgs, batch {batch_size}, {steps} steps):")
    for step in range(steps):
        idx = torch.randint(0, x0.shape[0], (batch_size,), device=dev)
        x_t, t, eps = make_training_pair(x0[idx], alpha_bars, T)
        loss = F.mse_loss(net(x_t, t), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % (steps // 6) == 0 or step == steps - 1:
            print(f"        step {step:>4}: train loss {loss.item():.4f}")
    net.eval()

    mse_after, eps_hat_after, x0_hat_after = recovery_mse()
    print(f"    recover MSE(x̂0, x0): {mse_before:.4f} -> {mse_after:.4f}  — the net now DENOISES.\n")

    # ---- the payoff figure: clean / noised / predicted ε̂ / recovered x̂0 ----------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [
        ("clean x0",        [x0_show[i] for i in range(n_show)]),
        (f"noised x_t (t={t_show_val})", [x_t_show[i] for i in range(n_show)]),
        ("predicted ε̂",     [eps_hat_after[i] for i in range(n_show)]),
        ("recovered x̂0",    [x0_hat_after[i] for i in range(n_show)]),
    ]
    fig, axes = plt.subplots(len(rows), n_show, figsize=(n_show * 1.05, len(rows) * 1.15))
    for r, (label, imgs) in enumerate(rows):
        for c in range(n_show):
            ax = axes[r, c]
            ax.imshow(_to_img(imgs[c]), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center", labelpad=38)
    fig.suptitle("the denoiser is an image->image map that WORKS: subtract the predicted noise ε̂,\n"
                 "and the clean digit x̂0 reappears (real TinyUNet, a few hundred steps)", fontsize=10)
    fig.tight_layout(rect=(0.06, 0, 1, 0.94))
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "01_denoise.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(out, _HERE)} — clean / noised / predicted ε̂ / recovered x̂0.")
    print("  That's the whole game of the denoiser: an image->image net that denoises. Next (exp_2):")
    print("  WHY the skip connections — ablate them and watch the recovered digit go blurry.")


# ---------------------------------------------------------------------------
# LAYER 2 (why skips): the up path bottoms out at a 7x7 bottleneck — coarse. So where does the
# pixel-sharp ε̂ come from? From the SKIP CONNECTIONS: the down side's high-res feature maps are
# stapled (concat) back onto the up side at the matching resolution, routing fine detail AROUND the
# funnel. We prove it by TRAINING TWO nets from scratch — one with skips, one that never gets them —
# and reading three things: the final loss, the recover-MSE, and the high-frequency energy of ε̂
# (plus a picture where the no-skip ε̂ is a smooth ghost). The 7x7 core can decide WHAT/WHERE; only
# the skips can carry the pixel grain.
# ---------------------------------------------------------------------------
def exp_2_why_skips(seed=0, T=1000, base=32, n_train=4000, batch_size=128, steps=300, lr=2e-4):
    """Why the skip connections: the 7x7 bottleneck is too coarse to emit pixel-sharp noise, so the
    down-path features are concatenated back onto the up path (the skips). Train one net WITH skips and
    one that NEVER sees them, then compare loss / recover-MSE / high-freq energy and SEE the no-skip ε̂
    smear into a low-frequency ghost. Same init, same data, same steps — the only difference is skips."""
    _banner("LAYER 2: why skips — the 7x7 bottleneck is coarse; the skip highways carry the detail")

    torch.manual_seed(seed)
    dev = _device()
    _, _, alpha_bars = make_linear_schedule(T=T)
    alpha_bars = alpha_bars.to(dev)

    print("  the puzzle: the up path starts from a 7x7 bottleneck (28x28 pooled down 4x). 7x7 is far")
    print("  too coarse to name every pixel of a noise map. So how does ε̂ come out pixel-sharp?")
    print("  the two `torch.cat([up, downN], 1)` lines: the DOWN side's high-res maps are stapled back")
    print("  onto the up side at 14x14 and 28x28 — detail routed AROUND the funnel. use_skips=False")
    print("  feeds zeros there instead, so we can measure exactly what they buy.\n")

    x0 = _mnist(train=True)[:n_train].to(dev)

    # ---- train two nets from the SAME init: one with skips, one that never gets them -----------
    net_skip = TinyUNet(base=base).to(dev)
    net_none = TinyUNet(base=base).to(dev)
    net_none.load_state_dict(net_skip.state_dict())                # identical starting weights = fair
    print(f"  training two nets from the same init ({n_train} imgs, {steps} steps each):")
    loss_skip = _quick_train(net_skip, x0, alpha_bars, T, steps, batch_size, lr, seed=seed + 7, use_skips=True)
    loss_none = _quick_train(net_none, x0, alpha_bars, T, steps, batch_size, lr, seed=seed + 7, use_skips=False)
    print(f"    final train loss   WITH skips {loss_skip:.4f}   |   NO skips {loss_none:.4f}   (lower = better)\n")

    # ---- a fixed display batch at ONE moderate level, scored both ways -------------------------
    torch.manual_seed(seed + 1)
    n_show = 8
    t_show_val = 250
    x0_show = x0[:n_show]
    ab_show = alpha_bars[t_show_val].view(1, 1, 1, 1)
    eps_show = torch.randn_like(x0_show)
    x_t_show = ab_show.sqrt() * x0_show + (1 - ab_show).sqrt() * eps_show
    t_show = torch.full((n_show,), t_show_val, device=dev, dtype=torch.long)

    with torch.no_grad():
        eps_skip = net_skip(x_t_show, t_show, use_skips=True)
        eps_none = net_none(x_t_show, t_show, use_skips=False)
    x0_skip = _recover_x0(x_t_show, eps_skip, ab_show)
    x0_none = _recover_x0(x_t_show, eps_none, ab_show)
    mse_skip = F.mse_loss(x0_skip.clamp(-1, 1), x0_show).item()
    mse_none = F.mse_loss(x0_none.clamp(-1, 1), x0_show).item()

    print(f"  scored on a held-fixed batch at t={t_show_val}:")
    print(f"    recover MSE(x̂0, x0)        WITH skips {mse_skip:.4f}   |   NO skips {mse_none:.4f}")
    print(f"    high-freq energy of ε̂      true ε {_hf(eps_show):.3f}  |  skips {_hf(eps_skip):.3f}"
          f"  |  no skips {_hf(eps_none):.3f}")
    print("    NO-skip ε̂ has almost no high-freq energy: it's a smooth ghost. The 7x7 core got the")
    print("    coarse WHAT/WHERE, but with the highways cut it cannot emit the pixel grain.\n")

    # ---- payoff figure: the detail is present with skips, gone without ------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [
        ("clean x0",         [x0_show[i] for i in range(n_show)]),
        ("true ε",           [eps_show[i] for i in range(n_show)]),
        ("ε̂  WITH skips",    [eps_skip[i] for i in range(n_show)]),
        ("ε̂  NO skips",      [eps_none[i] for i in range(n_show)]),
        ("x̂0 WITH skips",    [x0_skip[i] for i in range(n_show)]),
        ("x̂0 NO skips",      [x0_none[i] for i in range(n_show)]),
    ]
    fig, axes = plt.subplots(len(rows), n_show, figsize=(n_show * 1.05, len(rows) * 1.12))
    for r, (label, imgs) in enumerate(rows):
        for c in range(n_show):
            ax = axes[r, c]
            ax.imshow(_to_img(imgs[c]), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center", labelpad=40)
    fig.suptitle("why skips: the 7x7 bottleneck carries WHAT/WHERE, the skips carry the DETAIL\n"
                 "no-skip ε̂ is a smooth ghost (little high-freq) → recovered digit blurs", fontsize=10)
    fig.tight_layout(rect=(0.07, 0, 1, 0.94))
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "02_why_skips.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(out, _HERE)} — with-vs-without skips: ε̂ grain and recovered digit.")
    print("  So the U-Net = autoencoder + detail highways. Next (exp_3): WHY down/up at all — the")
    print("  receptive field, i.e. why pooling to 7x7 lets a small net see the WHOLE digit cheaply.")


# ---------------------------------------------------------------------------
# LAYER 3 (why down/up): exp_2 showed the 7x7 bottleneck forces us to add skips — so why pool down to
# 7x7 at ALL? Because a plain 3x3 conv only sees its 8 neighbours; to denoise a stroke CONSISTENTLY the
# net must know the GLOBAL digit (is this arc part of a 3 or an 8?). Pooling is how a SMALL conv net
# buys that reach: each pool DOUBLES how far a later conv sees, in original pixels. We prove it by
# measuring the EFFECTIVE receptive field of a center output pixel for the real U-Net vs a FLAT net with
# the SAME number of convs but no pooling — same conv count (same theoretical reach), yet the flat net's
# effective reach is a tiny central blob (~√depth), while the U-Net's spans the whole 28x28. And it's
# CHEAP: a conv at 7x7 costs (7/28)^2 = 1/16 the FLOPs of one at 28x28. Reach AND cost, both from pool.
# ---------------------------------------------------------------------------
def exp_3_why_down_up(seed=0, base=32, n_avg=8):
    """Why down/up at all: to denoise a stroke consistently the net needs the GLOBAL digit, but a 3x3
    conv sees only its neighbours. Pooling buys reach cheaply. Measure the EFFECTIVE receptive field of a
    center output pixel for the real TinyUNet vs a FLAT net with the SAME conv count but no pool: same
    theoretical reach, but the flat net's effective reach is a tiny blob while the U-Net's covers 28x28.
    Architectural (no training needed) — averaged over a few random inits. exp_4 opens WHY t is an input."""
    _banner("LAYER 3: why down/up — pooling buys a big receptive field cheaply; a flat net can't reach")

    dev = _device()
    print("  the puzzle: a 3x3 conv sees only its 8 neighbours. But to predict the noise on a stroke")
    print("  CONSISTENTLY, the net must know the whole digit (is this arc part of a 3 or an 8?). How")
    print("  does a SMALL conv net see all 28x28? pooling: each pool DOUBLES a later conv's reach in")
    print("  original pixels, so the 7x7 mid block has the entire digit in view.\n")

    print(f"  measuring the EFFECTIVE receptive field of the center output pixel (avg over {n_avg} random")
    print("  inits — it's an architecture property, no training needed):")
    print("    real TinyUNet (down/up, 3 pools)   vs   FlatNet (same 14 convs, NO pool, all at 28x28)\n")

    g_flat, r_flat, cov_flat = _effective_rf(lambda: FlatNet(base=base), dev, seed=seed, n_avg=n_avg)
    g_unet, r_unet, cov_unet = _effective_rf(lambda: TinyUNet(base=base), dev, seed=seed, n_avg=n_avg)

    print(f"    effective radius (RMS spread)   FlatNet {r_flat:5.2f} px   |   TinyUNet {r_unet:5.2f} px")
    print(f"    coverage (frac of pixels >1% max) FlatNet {cov_flat:5.1%}   |   TinyUNet {cov_unet:5.1%}")
    print("    SAME 14 convs, so the same reach ON PAPER — but with no pool the flat net's influence")
    print("    stays a small central blob (~√depth); the U-Net's spans the whole digit. And it's cheap:")
    print("    a conv at 7x7 costs (7/28)^2 = 1/16 the FLOPs of one at 28x28 — deep reach, small bill.\n")

    # ---- payoff figure: the two effective-receptive-field maps, side by side ------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    size = g_flat.shape[0]
    c = size // 2
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9))
    for ax, g, name, r in [(axes[0], g_flat, "FlatNet (no pool)", r_flat),
                           (axes[1], g_unet, "TinyUNet (down/up)", r_unet)]:
        ax.imshow((g / g.max()).numpy(), cmap="magma", vmin=0, vmax=1)
        ax.add_patch(Circle((c, c), r, fill=False, color="cyan", lw=1.4, ls="--"))  # effective radius
        ax.plot(c, c, "+", color="cyan", ms=8)
        ax.set_title(f"{name}\neff. radius {r:.1f}px", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("effective receptive field of the CENTER output pixel (brighter = more influence)\n"
                 "same 14 convs — but pooling lets the U-Net's center pixel SEE the whole 28x28 digit",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "03_why_down_up.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {os.path.relpath(out, _HERE)} — effective receptive field: flat blob vs full-digit reach.")
    print("  So down/up isn't just about the bottleneck — it's how a small net gets GLOBAL sight cheaply.")
    print("  Next (exp_4): WHY t is an input — the same x_t means different things at different levels.")


def run_experiments():
    # exp_1_whole_game()
    # exp_2_why_skips()
    exp_3_why_down_up()
    # exp_4_why_t_input()
    # exp_5_how_t_enters()
    # exp_6_the_block()


if __name__ == "__main__":
    run_experiments()
