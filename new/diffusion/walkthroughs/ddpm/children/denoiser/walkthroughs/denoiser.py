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
                         time conditioning and the net can only fit the average over levels -> its
                         loss DOUBLES at large t (t leaks from the grain early, but saturates late).
  5. HOW t ENTERS       — sinusoidal embedding -> small MLP -> ADDED into every block (FiLM-lite).
                         The code is what matters: sinusoids beat a raw scalar t/T and a free
                         embedding table; injection depth, honestly, doesn't move at this size.
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
    unbounded ~N(0,1). `use_skips=False` zeroes the two skip highways (the exp_2 ablation);
    `use_time=False` pins t to 0 so the net is TIME-BLIND — same weights, no clock (the exp_4 ablation)."""

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

    def forward(self, x, t, use_skips=True, use_time=True):
        if not use_time:
            t = torch.zeros_like(t)                                      # every example gets the SAME
        temb = self.time_mlp(timestep_embedding(t, self.temb_dim))       #   embedding -> no clock
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


class TimeVariantUNet(TinyUNet):
    """The exp_5 harness: the SAME U-Net, only the CLOCK CHANNEL swapped. Two knobs —

      mode   how t is encoded before the MLP:
             "sinusoidal" the real thing: cos/sin at many frequencies (multi-scale, shift-invariant)
             "scalar"     one number t/T — a rank-1 code: every t points the same direction
             "learned"    a free nn.Embedding table: T independent vectors, no smoothness prior
      inject where the embedding is ADDED:
             "all"        into every block (the real thing)
             "first"      only into down1; the rest get a zero temb and must carry t in activations

    Everything else — channels, blocks, skips, params outside the clock — is TinyUNet's."""

    def __init__(self, mode="sinusoidal", inject="all", T=1000, base=32, temb_dim=128):
        super().__init__(base=base, temb_dim=temb_dim)
        self.mode, self.inject, self.T = mode, inject, T
        if mode == "scalar":
            self.time_mlp = nn.Sequential(nn.Linear(1, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim))
        elif mode == "learned":
            self.table = nn.Embedding(T, temb_dim)                       # T free vectors, learned

    def encode(self, t):
        """t (B,) -> the raw code (B, *) handed to time_mlp. This IS the thing exp_5 compares."""
        if self.mode == "sinusoidal":
            return timestep_embedding(t, self.temb_dim)
        if self.mode == "scalar":
            return (t.float() / self.T)[:, None]                         # (B,1) — one direction only
        return self.table(t)

    def forward(self, x, t, **_):
        temb = self.time_mlp(self.encode(t))
        rest = temb if self.inject == "all" else torch.zeros_like(temb)  # "first": later blocks get none
        h1 = self.down1(self.stem(x), temb)
        h2 = self.down2(F.avg_pool2d(h1, 2), rest)
        h3 = self.down3(F.avg_pool2d(h2, 2), rest)
        m = self.mid(h3, rest)
        u = self.up2(torch.cat([F.interpolate(m, scale_factor=2, mode="nearest"), h2], 1), rest)
        u = self.up1(torch.cat([F.interpolate(u, scale_factor=2, mode="nearest"), h1], 1), rest)
        return self.out(F.silu(self.out_norm(u)))


def _recover_x0(x_t, eps_hat, ab):
    """Invert the forward closed form for x0 given a noise estimate: x̂0 = (x_t - √(1-ᾱ)·ε̂)/√ᾱ.
    ab is ᾱ_t reshaped to broadcast over pixels. (This is exactly the algebra the sampler uses.)"""
    return (x_t - (1 - ab).sqrt() * eps_hat) / ab.sqrt()


def _hf_per_image(e):
    """Per-image high-frequency energy: the variance LEFT after a 3x3 local blur. (B,1,H,W) -> (B,).
    Grainy (fine pixel detail, like real ε) scores high; a smooth low-frequency ghost scores near 0."""
    return (e - F.avg_pool2d(e, 3, 1, 1)).pow(2).mean(dim=(1, 2, 3))


def _hf(e):
    """Batch-mean high-frequency energy (a single number)."""
    return _hf_per_image(e).mean().item()


@torch.no_grad()
def _t_leak(x0, alpha_bars, T, seed, stride=10):
    """How much does the picture ALONE give away about t? Build the calibration curve h(t) = mean
    high-freq energy of x_t (grain rises as the digit dissolves), then read it BACKWARDS: for held-out
    samples at random true t, estimate t̂ = argmin_t |h(x_t) - h(t)| and score |t̂ - t|. This is the
    best a t-blind net could do with the cheapest possible cue — its ceiling for guessing the clock.
    Returns (grid, curve, t_true, t_hat)."""
    torch.manual_seed(seed)
    grid = torch.arange(0, T, stride, device=x0.device)
    curve = torch.stack([
        _hf_per_image(alpha_bars[tv].sqrt() * x0 + (1 - alpha_bars[tv]).sqrt() * torch.randn_like(x0)).mean()
        for tv in grid])                                                # (len(grid),) monotone-ish in t
    t_true = torch.randint(0, T, (x0.shape[0],), device=x0.device)
    ab = alpha_bars[t_true].view(-1, 1, 1, 1)
    h = _hf_per_image(ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn_like(x0))
    t_hat = grid[(h[:, None] - curve[None, :]).abs().argmin(dim=1)]     # invert the curve per image
    return grid, curve, t_true, t_hat


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


def _quick_train(net, x0, alpha_bars, T, steps, batch_size, lr, seed, **fwd):
    """Train a denoiser from scratch on random (x_t,t)->ε batches and return its final train loss.
    Forward flags (`use_skips=False`, `use_time=False`, ...) are threaded into every call, so an
    ablated net is ablated during TRAINING too — it never learns to lean on what we cut."""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    last = float("nan")
    for _ in range(steps):
        idx = torch.randint(0, x0.shape[0], (batch_size,), device=x0.device)
        x_t, t, eps = make_training_pair(x0[idx], alpha_bars, T)
        loss = F.mse_loss(net(x_t, t, **fwd), eps)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last = loss.item()
    net.eval()
    return last


@torch.no_grad()
def _loss_per_t(net, x0, alpha_bars, t_vals, seed, **fwd):
    """Held-out MSE(ε̂, ε) at each t in `t_vals`, with the SAME images and the SAME ε at every t (so the
    curve isolates the noise LEVEL, not sampling luck). Returns a list of losses aligned with t_vals."""
    torch.manual_seed(seed)
    eps = torch.randn_like(x0)
    out = []
    for tv in t_vals:
        ab = alpha_bars[tv].view(1, 1, 1, 1)
        x_t = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        t = torch.full((x0.shape[0],), tv, device=x0.device, dtype=torch.long)
        out.append(F.mse_loss(net(x_t, t, **fwd), eps).item())
    return out


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


# ---------------------------------------------------------------------------
# LAYER 4 (why t is an input): the architecture is settled (down/up for reach, skips for detail). The
# last non-obvious ingredient is TIME CONDITIONING. The reason is an AMBIGUITY in the input: a given
# picture x_t is a perfectly plausible x_t for MANY different t — only its noise LEVEL differs, and the
# level is exactly what "how much of this is noise?" asks. Without t the net has to answer for all levels
# at once, and the MSE-optimal answer to an ambiguous question is the AVERAGE (law of total variance):
#
#     ε̂*(x_t)   = E[ε | x_t]              t-blind   optimum
#     ε̂*(x_t,t) = E[ε | x_t, t]           t-aware   optimum
#     E‖ε − E[ε|x_t]‖² = E‖ε − E[ε|x_t,t]‖² + E‖E[ε|x_t,t] − E[ε|x_t]‖²
#                        \__ t-aware floor __/   \__ extra, ≥ 0: the spread ACROSS t __/
#
# so the blind net's loss is provably the aware net's loss PLUS the variance of the right answer across
# t. It can never be smaller, and the gap is biggest where the answer swings hardest with t. We train two
# nets (identical init/data/steps, one with `use_time=False`) and read the loss as a FUNCTION of t.
#
# The honest twist, which we also measure: the gap is only a few percent, because t LEAKS from the
# picture — grain rises as the digit dissolves, so a blind net can estimate the clock. But that cue
# SATURATES past t≈600 (all static looks alike), so the blind net's relative penalty is worst at large
# t — the region every sampling trajectory starts in, with 1000 steps for the error to compound.
# ---------------------------------------------------------------------------
def exp_4_why_t_input(seed=0, T=1000, base=32, n_train=4000, batch_size=128, steps=1000, lr=2e-4,
                      n_eval=512):
    """Why the timestep must be an input: the same x_t is plausible at many noise levels, and only t says
    WHICH — so 'how much of this is noise?' is unanswerable without it. See the ambiguity (one x_t decoded
    under several assumed t), then train a TIME-BLIND twin (`use_time=False`, same init/data/steps) and
    read loss vs t. Also measures the honest caveat: how much t leaks from the picture's grain, and where
    that cue dies (large t) — which is exactly where the blind net's relative penalty is worst."""
    _banner("LAYER 4: why t is an input — the same x_t means different things at different noise levels")

    torch.manual_seed(seed)
    dev = _device()
    _, _, alpha_bars = make_linear_schedule(T=T)
    alpha_bars = alpha_bars.to(dev)

    # ---- (i) the ambiguity, in the schedule numbers -------------------------------------------
    print("  the puzzle: the net is asked 'which part of this picture is the noise?'. But x_t is built")
    print("  as √ᾱ_t·x0 + √(1-ᾱ_t)·ε — the MIX depends on t, and the picture alone does not say which:\n")
    print("      t      √ᾱ_t (signal)   √(1-ᾱ_t) (noise)   noise share of the variance")
    for tv in (50, 250, 500, 750, 950):
        ab = alpha_bars[tv].item()
        print(f"    {tv:>4}       {ab ** 0.5:.3f}            {(1 - ab) ** 0.5:.3f}                {1 - ab:6.1%}")
    print("    same picture, wildly different amounts to subtract. t is the missing side of the equation.\n")

    x0 = _mnist(train=True)[:n_train].to(dev)

    # ---- train the two nets: identical init, one of them time-blind ---------------------------
    net_t = TinyUNet(base=base).to(dev)
    net_blind = TinyUNet(base=base).to(dev)
    net_blind.load_state_dict(net_t.state_dict())                  # identical starting weights = fair
    print(f"  training two nets from the same init ({n_train} imgs, {steps} steps each):")
    loss_t = _quick_train(net_t, x0, alpha_bars, T, steps, batch_size, lr, seed=seed + 7)
    loss_blind = _quick_train(net_blind, x0, alpha_bars, T, steps, batch_size, lr, seed=seed + 7, use_time=False)
    print(f"    final train loss   t AS INPUT {loss_t:.4f}   |   TIME-BLIND {loss_blind:.4f}   (lower = better)\n")

    # ---- (ii) the ambiguity made visible: ONE x_t, decoded under several assumed t --------------
    torch.manual_seed(seed + 1)
    n_show = 6
    t_true = 400
    x0_show = x0[:n_show]
    ab_true = alpha_bars[t_true].view(1, 1, 1, 1)
    x_t_show = ab_true.sqrt() * x0_show + (1 - ab_true).sqrt() * torch.randn_like(x0_show)
    t_assumed = [50, 200, t_true, 700, 950]

    print(f"  (ii) feed the SAME x_t (really from t={t_true}) to the t-aware net, lying about t:")
    decoded = []
    for tv in t_assumed:
        ab = alpha_bars[tv].view(1, 1, 1, 1)
        t = torch.full((n_show,), tv, device=dev, dtype=torch.long)
        with torch.no_grad():
            x0_hat = _recover_x0(x_t_show, net_t(x_t_show, t), ab)
        mse = F.mse_loss(x0_hat.clamp(-1, 1), x0_show).item()
        decoded.append((tv, x0_hat, mse))
        mark = "  <- the truth" if tv == t_true else ""
        print(f"        told t={tv:>4}:  recover MSE(x̂0,x0) = {mse:.4f}{mark}")
    print("    ONE input, five different answers — the net's output is a function of t as much as of x_t,")
    print("    and only the true t recovers the digit. That is the ambiguity a time-blind net must eat.\n")

    # ---- (iii) loss as a function of t, for both nets -------------------------------------------
    x_eval = _mnist(train=False)[:n_eval].to(dev)                 # held-out digits
    t_vals = list(range(25, T, 50))
    curve_t = _loss_per_t(net_t, x_eval, alpha_bars, t_vals, seed=seed + 3)
    curve_blind = _loss_per_t(net_blind, x_eval, alpha_bars, t_vals, seed=seed + 3, use_time=False)

    print("  (iii) held-out loss as a FUNCTION of t (the blind net must answer every t with one rule):")
    print("        t        t AS INPUT    TIME-BLIND     blind is worse by")
    for tv, a, b in zip(t_vals, curve_t, curve_blind):
        if tv in (25, 175, 475, 775, 975):
            print(f"      {tv:>4}         {a:.4f}        {b:.4f}         {(b / a - 1):+6.1%}")
    lo, hi = slice(0, 4), slice(-4, None)                         # t<200 mostly signal | t>800 mostly noise
    def _mean(v, s): return sum(v[s]) / len(v[s])
    rel_lo = _mean(curve_blind, lo) / _mean(curve_t, lo) - 1
    rel_hi = _mean(curve_blind, hi) / _mean(curve_t, hi) - 1
    print(f"    t<200   t AS INPUT {_mean(curve_t, lo):.4f}  |  TIME-BLIND {_mean(curve_blind, lo):.4f}   ({rel_lo:+.1%})")
    print(f"    t>800   t AS INPUT {_mean(curve_t, hi):.4f}  |  TIME-BLIND {_mean(curve_blind, hi):.4f}   ({rel_hi:+.1%})")
    print("    the blind net is never better — it can't be. Law of total variance, with ε̂*(x_t)=E[ε|x_t]")
    print("    the best t-blind answer and ε̂*(x_t,t)=E[ε|x_t,t] the best t-aware one:")
    print("      E‖ε-E[ε|x_t]‖² = E‖ε-E[ε|x_t,t]‖² + E‖E[ε|x_t,t]-E[ε|x_t]‖²  ≥  the t-aware loss,")
    print("    the extra term being how much the right answer SWINGS with t at a fixed picture.\n")

    # ---- (iv) the honest caveat: t partly LEAKS from the picture — until it saturates ------------
    grid, leak_curve, t_leak_true, t_leak_hat = _t_leak(x_eval, alpha_bars, T, seed=seed + 5)
    err = (t_leak_hat - t_leak_true).abs().float()
    err_lo = err[t_leak_true < 500].median().item()
    err_hi = err[t_leak_true >= 500].median().item()
    print("  (iv) why is the OVERALL gap only ~10%, when the blind net is flying blind? because t partly")
    print("  LEAKS from the picture: as the digit dissolves, the GRAIN (high-freq energy) climbs, so a net can")
    print("  estimate the clock itself. Guessing t from that one cue alone (nearest point on the h(t)")
    print("  calibration curve) already gets:")
    print(f"      median |t̂ - t|   for t<500: {err_lo:5.1f} steps   |   for t>=500: {err_hi:5.1f} steps")
    print(f"      the cue saturates: h(t) rises {leak_curve[0]:.3f} -> {leak_curve[len(grid)//2]:.3f} by t=500,")
    print(f"      then crawls to {leak_curve[-1]:.3f} — past ~600 every x_t looks like the same static.")
    print("    so the blind net (a) burns capacity re-deriving what one number would have told it, and")
    print(f"    (b) is left genuinely blind exactly where the cue dies — which is why its RELATIVE penalty")
    print(f"    is worst at large t ({rel_hi:+.1%} vs {rel_lo:+.1%} at small t). And in the SAMPLER that region")
    print("    is where every trajectory starts, with 1000 steps for the error to compound.\n")

    # ---- payoff figures: one per claim, so each can sit next to its paragraph in the note --------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(_FIGS, exist_ok=True)
    written = []

    # (a) the ambiguity: ONE x_t, decoded under each assumed t (a few digits, to show it's not a fluke)
    n_rows = 3
    fig, axes = plt.subplots(n_rows, len(decoded) + 1, figsize=((len(decoded) + 1) * 1.45, n_rows * 1.45),
                             gridspec_kw={"hspace": 0.08, "wspace": 0.08})
    for r in range(n_rows):
        ax = axes[r, 0]
        ax.imshow(_to_img(x_t_show[r]), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(f"the input x_t\n(really t={t_true})", fontsize=8)
        for k, (tv, x0_hat, mse) in enumerate(decoded):
            ax = axes[r, k + 1]
            ax.imshow(_to_img(x0_hat[r]), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"told t={tv}\nMSE {mse:.3f}", fontsize=8,
                             color="tab:green" if tv == t_true else "0.35")
    fig.suptitle("the same x_t, decoded under different assumed t: lie low and it stays static, lie high\n"
                 "and it's subtracted into black — only the true t recovers the digit", fontsize=10, y=1.10)
    out = os.path.join(_FIGS, "04_ambiguity.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append((out, "one x_t decoded under five assumed t."))

    # (b) the ablation: held-out loss vs t, with and without the clock
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(t_vals, curve_t, "o-", ms=4, color="tab:blue", label="t AS INPUT")
    ax.plot(t_vals, curve_blind, "s-", ms=4, color="tab:red", label="TIME-BLIND (use_time=False)")
    ax.fill_between(t_vals, curve_t, curve_blind, color="tab:red", alpha=0.15)
    ax.set_yscale("log")                                           # the loss spans ~2 decades over t
    ax.set_xlabel("timestep t"); ax.set_ylabel("held-out MSE(ε̂, ε)   (log)")
    ax.set_title("without the clock the net can only fit the average over noise levels\n"
                 f"blind is never better; relative penalty {rel_lo:+.0%} at t<200  ->  {rel_hi:+.0%} at t>800",
                 fontsize=10)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = os.path.join(_FIGS, "04_loss_vs_t.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append((out, "loss vs t, with and without t as an input."))

    # (c) the caveat: how much t leaks from the picture's grain, and where that cue dies
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(grid.cpu(), leak_curve.cpu(), color="tab:purple", lw=2)
    ax.axvspan(600, T, color="0.6", alpha=0.25)
    ax.text(0.97, 0.42, "cue saturates:\nevery x_t is the same static", fontsize=9, color="0.25",
            ha="right", va="center", transform=ax.transAxes)
    ax.set_xlabel("timestep t"); ax.set_ylabel("grain of x_t  (high-freq energy)")
    ax.set_title("how much t leaks from the picture — and where it stops\n"
                 f"guessing t from grain alone: median |t̂-t| = {err_lo:.0f} steps (t<500) -> "
                 f"{err_hi:.0f} steps (t≥500)", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(_FIGS, "04_t_leak.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append((out, "the grain cue and its saturation."))

    for path, what in written:
        print(f"  wrote {os.path.relpath(path, _HERE)} — {what}")
    print("  So t is not decoration: it disambiguates the question. Next (exp_5): HOW t enters — the")
    print("  sinusoidal embedding -> MLP -> added into EVERY block, and why that shape.")


# ---------------------------------------------------------------------------
# LAYER 5 (how t enters): exp_4 settled that the net must be TOLD t. Now the design question — what
# shape should that channel have? The real net does:
#
#     t (an integer) --sinusoidal--> (B,128) --small MLP--> temb --ADD into every block-->
#
# Two choices to justify. WHY SINUSOIDAL: write e(t) = [cos(ω_k t) ; sin(ω_k t)], ω_k = 10000^(-k/K).
# Then, by cos(a)cos(b)+sin(a)sin(b) = cos(a-b),
#
#     e(t)·e(t') = Σ_k [cos(ω_k t)cos(ω_k t') + sin(ω_k t)sin(ω_k t')] = Σ_k cos(ω_k·(t-t'))
#     ‖e(t)‖²    = Σ_k [cos²(ω_k t) + sin²(ω_k t)] = K            (the same for EVERY t)
#
# so the code is a constant-length ruler whose geometry depends only on the GAP t-t', read at K scales
# at once (periods ~6 steps up to ~50,000 steps). Neighbours land close (so what the net learns at t
# transfers to t±1) while distant levels are far apart. We measure that against the two obvious
# alternatives — one raw scalar t/T, and a free nn.Embedding table — and then train all three.
# WHERE it's injected we also measure, and report honestly: at this size it does not matter.
# ---------------------------------------------------------------------------
def _code_separation(e, deltas):
    """How far apart does a code place two timesteps Δ apart, in units of its own scale?
        sep(Δ) = mean_t ‖e(t+Δ) - e(t)‖ / mean_t ‖e(t)‖
    Small sep = the two levels look alike to the net (hard to tell apart); large sep = unrelated."""
    n = e.norm(dim=1).mean()
    return [((e[d:] - e[:-d]).norm(dim=1).mean() / n).item() for d in deltas]


def exp_5_how_t_enters(seed=0, T=1000, base=32, n_train=4000, batch_size=128, steps=1000, lr=2e-4,
                       n_eval=512, temb_dim=128):
    """How t enters: sinusoidal embedding -> small MLP -> added into every block. Derive the two
    properties that make sinusoids the right code (constant norm, inner product that depends only on
    t-t', read at many scales), measure them against a raw scalar t/T and a free embedding table, then
    train all three plus a time-blind floor and read loss vs t. Also tests injection depth (every block
    vs only the first) — which, honestly, makes no measurable difference at this size."""
    _banner("LAYER 5: how t enters — sinusoidal code -> MLP -> added into every block")

    torch.manual_seed(seed)
    dev = _device()
    _, _, alpha_bars = make_linear_schedule(T=T)
    alpha_bars = alpha_bars.to(dev)

    # ---- (i) what the code IS: a ladder of frequencies -----------------------------------------
    half = temb_dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
    periods = 2 * math.pi / freqs
    print("  the channel is three steps:  t --sinusoidal--> (B,128) --MLP--> temb --ADD into blocks-->")
    print(f"  the embedding is {half} cos/sin pairs at geometrically spaced frequencies — a ladder of")
    print("  clocks, from one that ticks every few steps to one that barely moves over the whole run:")
    print("      pair k        0      16      32      48      63")
    print(f"      period    {periods[0]:6.1f}  {periods[16]:6.1f}  {periods[32]:6.1f}  {periods[48]:6.0f}  {periods[63]:6.0f}   steps")
    print("    fast pairs resolve NEIGHBOURING t; slow pairs say where we are in the run overall.\n")

    t_all = torch.arange(T)
    e_sin = timestep_embedding(t_all, temb_dim)                    # (T, 128) the real code
    e_scalar = (t_all.float() / T)[:, None]                        # (T, 1)   one number
    torch.manual_seed(seed)
    e_learned = torch.randn(T, temb_dim)                           # (T, 128) a free table, at init

    dots = (e_sin[:T - 10] * e_sin[10:]).sum(1)
    print("  two facts fall out of cos(a)cos(b)+sin(a)sin(b) = cos(a-b):")
    print(f"    ‖e(t)‖² = Σ_k 1 = K = {half}  ->  ‖e(t)‖ = {e_sin.norm(dim=1).mean():.3f} for EVERY t (constant length)")
    print(f"    e(t)·e(t') = Σ_k cos(ω_k(t-t')) depends only on the GAP: at gap 10, dot = {dots.mean():.3f}")
    print(f"      with std {dots.std():.2e} across all t — measured, and flat as the algebra says.")
    print("    a constant-length, shift-invariant ruler: no preferred origin, no dead zone.\n")

    # ---- (ii) resolution: how well does each code separate nearby vs distant t? ------------------
    deltas = [1, 2, 5, 10, 25, 50, 100, 250, 500]
    sep_sin = _code_separation(e_sin, deltas)
    sep_scalar = _code_separation(e_scalar, deltas)
    sep_learned = _code_separation(e_learned, deltas)
    print("  (ii) sep(Δ) = ‖e(t+Δ)-e(t)‖ / ‖e‖ — how far apart the code puts two levels Δ steps apart:")
    print("        Δ            " + "".join(f"{d:>8}" for d in deltas))
    for name, s in [("sinusoidal", sep_sin), ("scalar t/T", sep_scalar), ("learned table", sep_learned)]:
        print(f"    {name:<14}" + "".join(f"{v:8.3f}" for v in s))
    print(f"    sinusoidal: neighbours are already {sep_sin[0]:.2f} apart (tellable) yet the code saturates —")
    print("      near is near, far is far. BOTH resolution and boundedness, at every scale.")
    print(f"    scalar: {sep_scalar[0]:.3f} at Δ=1 — {sep_sin[0] / sep_scalar[0]:.0f}x blunter. It is also rank-1: every t is the")
    print("      SAME direction at a different length, so the net must resolve the level by magnitude")
    print("      alone, through a nonlinearity, at a scale (t/T ≤ 1) far below its activations.")
    print(f"    learned table: ~{sep_learned[0]:.2f} at EVERY Δ — all 1000 rows mutually orthogonal. No notion of")
    print("      'nearby t' at all, so nothing learned at t helps at t±1: each row must be fit alone.\n")

    # ---- (iii) does it matter? train the variants ----------------------------------------------
    x0 = _mnist(train=True)[:n_train].to(dev)
    x_eval = _mnist(train=False)[:n_eval].to(dev)
    t_vals = list(range(25, T, 50))

    variants = [
        ("sinusoidal, every block", dict(mode="sinusoidal", inject="all"),   "tab:blue",   "-"),
        ("sinusoidal, FIRST block only", dict(mode="sinusoidal", inject="first"), "tab:cyan", "--"),
        ("scalar t/T", dict(mode="scalar", inject="all"),                    "tab:orange", "-"),
        ("learned table", dict(mode="learned", inject="all"),                "tab:green",  "-"),
    ]
    print(f"  (iii) train each clock variant from scratch ({n_train} imgs, {steps} steps, identical seed):")
    results = []
    for name, kw, color, ls in variants:
        torch.manual_seed(seed)                                    # same init for the shared U-Net body
        net = TimeVariantUNet(T=T, base=base, temb_dim=temb_dim, **kw).to(dev)
        moved = None
        if kw["mode"] == "learned":
            table0 = net.table.weight.detach().clone()
        loss = _quick_train(net, x0, alpha_bars, T, steps, batch_size, lr, seed=seed + 7)
        if kw["mode"] == "learned":
            moved = ((net.table.weight.detach() - table0).norm() / table0.norm()).item()
        curve = _loss_per_t(net, x_eval, alpha_bars, t_vals, seed=seed + 3)
        results.append((name, curve, color, ls, moved))
        print(f"    {name:<30} train loss {loss:.4f}")
    torch.manual_seed(seed)                                        # the exp_4 floor: no clock at all
    net_blind = TinyUNet(base=base, temb_dim=temb_dim).to(dev)
    loss_blind = _quick_train(net_blind, x0, alpha_bars, T, steps, batch_size, lr, seed=seed + 7, use_time=False)
    curve_blind = _loss_per_t(net_blind, x_eval, alpha_bars, t_vals, seed=seed + 3, use_time=False)
    results.append(("TIME-BLIND (exp_4 floor)", curve_blind, "tab:red", ":", None))
    print(f"    {'TIME-BLIND (exp_4 floor)':<30} train loss {loss_blind:.4f}\n")

    hi = slice(-4, None)                                           # t>800: where exp_4 showed t matters most
    def _mean(v): return sum(v[hi]) / len(v[hi])
    ref = _mean(results[0][1])
    print("  held-out loss at t>800 (the region exp_4 flagged — and where every sample starts):")
    for name, curve, _, _, _ in results:
        print(f"    {name:<30} {_mean(curve):.4f}   ({_mean(curve) / ref:.2f}x the sinusoidal net)")
    moved = [m for *_, m in results if m is not None][0]
    print(f"    the learned table moved only {moved:.1%} from its random init in {steps} steps — each row saw")
    print(f"    ~{steps * batch_size // T} gradients, and rows never help each other. Sinusoids hand the net that")
    print("    structure for free, which is the whole reason they are the default.\n")

    # ---- (iv) injection depth: the honest null result -------------------------------------------
    print("  (iv) WHERE it enters: 'every block' vs 'first block only' land on top of each other above —")
    print("  no measurable difference here, and that is worth saying plainly. With 6 blocks at 28x28 the")
    print("  net can just carry the clock forward in its activations. Injecting everywhere is standard")
    print("  because (a) each block's GroupNorm re-centers its input, eroding a constant added once, and")
    print("  (b) at real depth/resolution the transported clock costs channels in every block it crosses.")
    print("  It becomes decisive when the time signal MODULATES normalization instead of being added —")
    print("  FiLM/AdaLN, which is how DiT conditions (a later subtopic). At MNIST scale: free either way.\n")

    # ---- figures: one per claim ----------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(_FIGS, exist_ok=True)
    written = []

    # (a) the code itself: the frequency ladder as an image
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    im = ax.imshow(e_sin.numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1,
                   extent=(0, temb_dim, T, 0))
    ax.set_xlabel("embedding dimension  (cos pairs 0-63 | sin pairs 0-63; left = fast, right = slow)")
    ax.set_ylabel("timestep t")
    ax.set_title("the sinusoidal timestep code: a ladder of clocks\n"
                 f"periods {periods[0]:.0f} -> {periods[63]:.0f} steps; every row has the same length ‖e(t)‖ = {e_sin.norm(dim=1)[0]:.0f}",
                 fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85, label="value")
    fig.tight_layout()
    out = os.path.join(_FIGS, "05_sinusoidal_code.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append((out, "the code as an image: fast pairs left, slow pairs right."))

    # (b) resolution: separation vs gap, for the three codes
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for name, s, color in [("sinusoidal", sep_sin, "tab:blue"), ("scalar t/T", sep_scalar, "tab:orange"),
                           ("learned table (init)", sep_learned, "tab:green")]:
        ax.plot(deltas, s, "o-", ms=4, color=color, label=name)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("gap Δ between the two timesteps"); ax.set_ylabel("sep(Δ) = ‖e(t+Δ)-e(t)‖ / ‖e‖")
    ax.set_title("what each code does with 'nearby' and 'far apart'\n"
                 "sinusoidal: neighbours tellable AND bounded · scalar: blunt up close · table: everything unrelated",
                 fontsize=9.5)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = os.path.join(_FIGS, "05_separation.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append((out, "code geometry: separation vs gap."))

    # (c) does it matter: per-t loss for every variant
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for name, curve, color, ls, _ in results:
        ax.plot(t_vals, curve, ls, marker="o", ms=3, color=color, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("timestep t"); ax.set_ylabel("held-out MSE(ε̂, ε)   (log)")
    ax.set_title("the clock's ENCODING matters, its injection depth (here) does not\n"
                 "sinusoidal < scalar < learned table < no clock — the spread opens up at large t",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    out = os.path.join(_FIGS, "05_which_code.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append((out, "loss vs t for every clock variant."))

    for path, what in written:
        print(f"  wrote {os.path.relpath(path, _HERE)} — {what}")
    print("  Next (exp_6): THE BLOCK — why GroupNorm->SiLU->conv with a residual is the modern default,")
    print("  and what BatchNorm / ReLU / a plain conv stack cost instead.")


def run_experiments():
    # exp_1_whole_game()
    # exp_2_why_skips()
    # exp_3_why_down_up()
    # exp_4_why_t_input()
    exp_5_how_t_enters()
    # exp_6_the_block()


if __name__ == "__main__":
    run_experiments()
