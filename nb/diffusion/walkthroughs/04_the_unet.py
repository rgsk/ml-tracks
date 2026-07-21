# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# Diffusion · 04 — the denoiser: why a U-Net + time conditioning

`03` fed a target to a black box. Now we open the box — the `TinyUNet` that `01` trained. Its whole
job: read a noisy image `x_t` and its level `t`, and output a **full 28×28 map** — the predicted noise
`ε̂`. Three questions, each answered by *running* the model, not by assertion:

1. **Why image → image, not image → label?** The CNN track (`nb/cnn`) collapses `28×28 → 10` and stops.
   Here the output is the same shape as the input. That one fact reshapes the whole architecture: no
   flatten-to-a-vector head, and we must *rebuild* spatial resolution on the way out.
2. **Why down / up with skip connections?** Downsampling `28→14→7` buys a cheap **global view** (one
   output pixel ends up seeing the whole digit). But squeezing through a `7×7` bottleneck throws away
   fine detail — so **skip connections** route the high-res features around the funnel. We'll delete
   the skips and *watch* the prediction go blurry.
3. **Why must the timestep `t` be an input?** The same pixels `x_t` mean different things at different
   noise levels. We'll feed the model the *wrong* `t` and watch its error blow up — proof it genuinely
   uses `t`.

This is the CNN track's conv + downsample trunk (`nb/cnn/05_downsample`), rewired for image→image.
"""

# %%
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def _repo_root() -> Path:
    here = Path.cwd()
    for d in (here, *here.parents):
        if (d / "pyproject.toml").exists():
            return d
    return here


ROOT = _repo_root()
DATA = ROOT / "nb" / "data" / "mnist.npz"
CKPT = ROOT / "nb" / "diffusion" / "checkpoints" / "ddpm.pt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_mnist(train=True):
    """MNIST images (N,1,28,28) in [-1,1], from the cached npz. No torchvision."""
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        print(f"  downloading MNIST npz (~11MB) -> {DATA.relative_to(ROOT)} ...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", DATA)
    d = np.load(DATA)
    x = d["x_train"] if train else d["x_test"]
    return (torch.from_numpy(x).float() / 127.5 - 1.0).unsqueeze(1)   # (N,1,28,28) in [-1,1]


def to_img(x):
    return ((x.squeeze().clamp(-1, 1) + 1) / 2).cpu().numpy()


def make_schedule(T=1000, device="cpu"):
    """Linear DDPM schedule (same as 01–03). Returns betas, alphas, alpha_bar."""
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bar


def timestep_embedding(t, dim):
    """Sinusoidal embedding of the integer timestep (B,) -> (B, dim) — the llm/ positional trick."""
    half = dim // 2
    freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


T = 1000
betas, alphas, alpha_bar = make_schedule(T, device=DEV)

# %% [markdown]
"""
## The model, verbatim from `01` (plus one switch)

Same `TinyUNet` you already trained — copied here so this notebook stands alone. The **only** addition
is a `use_skips` flag in `forward`, so we can turn the skip connections off and measure what they buy.
The weights don't know about the flag, so it loads `01`'s checkpoint unchanged.
"""

# %%
class _Block(nn.Module):
    """Residual block: (GroupNorm -> SiLU -> conv) twice, timestep added in the middle."""

    def __init__(self, cin, cout, temb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(temb_dim, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class TinyUNet(nn.Module):
    """Predicts ε from (x_t, t). Down 28->14->7 (channels grow), up 7->14->28 with skip connections;
    the timestep is injected into every block. `use_skips=False` zeroes the two U-Net skips."""

    def __init__(self, base=32, temb_dim=128):
        super().__init__()
        self.temb_dim = temb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(temb_dim, temb_dim), nn.SiLU(), nn.Linear(temb_dim, temb_dim))
        self.stem = nn.Conv2d(1, base, 3, padding=1)
        self.down1 = _Block(base, base, temb_dim)
        self.down2 = _Block(base, base * 2, temb_dim)
        self.down3 = _Block(base * 2, base * 4, temb_dim)
        self.mid = _Block(base * 4, base * 4, temb_dim)
        self.up2 = _Block(base * 4 + base * 2, base * 2, temb_dim)     # + down2 skip
        self.up1 = _Block(base * 2 + base, base, temb_dim)             # + down1 skip
        self.out_norm = nn.GroupNorm(8, base)
        self.out = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x, t, use_skips=True):
        temb = self.time_mlp(timestep_embedding(t, self.temb_dim))
        h1 = self.down1(self.stem(x), temb)                           # (B, base,  28, 28)
        h2 = self.down2(F.avg_pool2d(h1, 2), temb)                    # (B, 2base, 14, 14)
        h3 = self.down3(F.avg_pool2d(h2, 2), temb)                    # (B, 4base,  7,  7)
        m = self.mid(h3, temb)                                        # bottleneck: 7x7
        s2 = h2 if use_skips else torch.zeros_like(h2)                # the two skip highways
        s1 = h1 if use_skips else torch.zeros_like(h1)
        u = F.interpolate(m, scale_factor=2, mode="nearest")          # 7 -> 14
        u = self.up2(torch.cat([u, s2], 1), temb)
        u = F.interpolate(u, scale_factor=2, mode="nearest")          # 14 -> 28
        u = self.up1(torch.cat([u, s1], 1), temb)
        return self.out(F.silu(self.out_norm(u)))                     # (B,1,28,28) predicted ε


def load_or_train():
    """Load 01's EMA weights if present; else train a quick stand-in so the notebook runs alone."""
    model = TinyUNet(base=32, temb_dim=128).to(DEV)
    if CKPT.exists():
        ck = torch.load(CKPT, map_location=DEV)
        model.load_state_dict(ck["ema"])
        print(f"  loaded 01's trained EMA weights <- {CKPT.relative_to(ROOT)}")
    else:
        print("  no checkpoint found — training a quick stand-in (~1 epoch, rougher than 01)...")
        x0 = load_mnist(True).to(DEV)
        opt = torch.optim.Adam(model.parameters(), lr=2e-4)
        for step in range(600):
            xb = x0[torch.randint(0, len(x0), (128,), device=DEV)]
            tt = torch.randint(0, T, (128,), device=DEV)
            eps = torch.randn_like(xb)
            ab = alpha_bar[tt].view(-1, 1, 1, 1)
            loss = F.mse_loss(model(ab.sqrt() * xb + (1 - ab).sqrt() * eps, tt), eps)
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


model = load_or_train()
print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

# %% [markdown]
"""
## 1 · Image → image: watch the resolution collapse and come back

Run one batch through and record the tensor shape at each stage. Two things to watch: the **spatial
size** funnels `28→14→7` and then rebuilds `7→14→28`, and the **channel count** grows `32→64→128` on
the way down. A CNN classifier would keep shrinking to `(B, 10)` and never come back — that flatten-
to-a-label head is exactly what a denoiser *cannot* have, because it has to emit a full image.
"""

# %%
x = load_mnist(train=False).to(DEV)
xb = x[:8]
tb = torch.full((8,), 500, device=DEV)

shapes, order = {}, ["stem", "down1", "down2", "down3", "mid", "up2", "up1", "out"]
hooks = [getattr(model, n).register_forward_hook(
    lambda m, i, o, n=n: shapes.__setitem__(n, tuple(o.shape[1:]))) for n in order]
with torch.no_grad():
    _ = model(xb, tb)
for h in hooks:
    h.remove()

print(f"  input                : {tuple(xb.shape[1:])}")
for n in order:
    c, hh, ww = shapes[n]
    stage = {"stem": "in", "mid": "bottleneck", "out": "out"}.get(n, "down" if n[:4] == "down" else "up")
    print(f"  {n:<6} ({stage:<10}): {(c, hh, ww)}   activation volume C·H·W = {c*hh*ww:>6,}")
print(f"  output               : {tuple(_.shape[1:])}   <- same H×W as the input, 1 channel = ε̂")
print("  (a classifier would end at (10,) — a label. Here we rebuild the full 28×28.)")

# %% [markdown]
"""
The volume column is the payoff of downsampling: the bottleneck (`mid`, `128×7×7 = 6,272`) is *smaller*
than the top layers (`32×28×28 = 25,088`) even though it has 4× the channels. Halving each spatial side
quarters the pixel count, so we can afford to quadruple the channels — richer, more abstract features —
and still do **less** work per layer. Most of the network's capacity lives at low resolution, cheaply.

But a plain down→up stack (an autoencoder) has a problem: every fine detail has to squeeze through that
`7×7` funnel and be reconstructed from scratch. The two `torch.cat` lines are the fix — but first, what
does downsampling actually *buy* us up top?
"""

# %% [markdown]
"""
## 2 · Why downsample: one output pixel sees the whole digit

Downsampling isn't just cheap — it grows the **receptive field**. To denoise a pixel well, the model
needs *context*: is this speck part of a stroke, or stray noise? We measure the receptive field
directly. Set one output pixel as the target, backprop to the input, and see how far its influence
spreads — the magnitude of `∂ out[14,14] / ∂ input` at every input pixel.
"""

# %%
probe = x[:1].clone().requires_grad_(True)
t1 = torch.full((1,), 500, device=DEV)

model(probe, t1)[0, 0, 14, 14].backward()             # gradient of the centre output pixel
g = probe.grad[0, 0].abs()
rf_frac = (g > 0).float().mean().item()
print(f"  the centre output pixel depends on {rf_frac*100:.0f}% of the 28×28 input — a *global*")
print(f"  receptive field. The influence peaks locally and tapers out across the whole grid:")

fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.4))
im = ax.imshow((g / g.max()).log10().clamp(min=-4).cpu(), cmap="magma")
ax.plot(14, 14, "c+", ms=10); ax.set_title("|∂ out[14,14] / ∂ input|  (log)", fontsize=10); ax.axis("off")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout(); plt.show()
print("  the deep down/up path lets one output pixel draw on the whole image — global context.")

# %% [markdown]
"""
The downsampled path gives each output pixel a **global** receptive field — it can relate a stroke here
to a stroke there, which local convolutions on the full `28×28` grid would take many more layers to
reach. That global view is what the bottleneck is *for*. The cost is spatial precision: coarse features
know *what* is where only roughly. Which is exactly the trade the skip connections buy back.

## 3 · Why skips: the bottleneck blurs; the highways restore detail

Delete nothing, add nothing — just flip `use_skips=False` so the two `cat`s see zeros instead of the
high-res features `h1`, `h2`. Everything must now flow through the `7×7` funnel and be upsampled with
`nearest`. Predict `ε` on a mildly-noised digit both ways and compare to the true `ε`.
"""

# %%
torch.manual_seed(0)
xd = x[:1]
td = torch.full((1,), 120, device=DEV)                # low-ish noise: fine detail still matters
eps = torch.randn_like(xd)
ab = alpha_bar[120]
xt = ab.sqrt() * xd + (1 - ab).sqrt() * eps           # the noisy image the net sees

with torch.no_grad():
    eps_skip = model(xt, td, use_skips=True)
    eps_none = model(xt, td, use_skips=False)

mse_skip = F.mse_loss(eps_skip, eps).item()
mse_none = F.mse_loss(eps_none, eps).item()
hf = lambda e: (e - F.avg_pool2d(e, 3, 1, 1)).pow(2).mean().item()   # high-frequency energy proxy
print(f"  MSE(ε̂, ε):  with skips {mse_skip:.3f}   |  skips off {mse_none:.3f}   (lower = better)")
print(f"  high-freq energy of ε̂:  true {hf(eps):.3f}  |  skips {hf(eps_skip):.3f}"
      f"  |  skips off {hf(eps_none):.3f}   (skips-off is smoothed out)")

fig, ax = plt.subplots(1, 4, figsize=(10, 3))
for a, im, ttl in zip(ax, [to_img(xt), eps[0, 0].cpu(), eps_skip[0, 0].cpu(), eps_none[0, 0].cpu()],
                      ["noisy x_t (t=120)", "true ε", "ε̂ with skips", "ε̂ skips off"]):
    a.imshow(im, cmap="gray"); a.set_title(ttl, fontsize=9); a.axis("off")
fig.tight_layout(); plt.show()
print("  skips off → the prediction loses its fine grain and the loss jumps. The skips ARE the detail.")

# %% [markdown]
"""
With the skips off, `ε̂` is a smeared, low-frequency ghost and the MSE jumps — the `7×7` bottleneck
simply can't carry the pixel-level structure, and `nearest` upsampling can't invent it back. **A U-Net
is an autoencoder plus skip highways**: the deep path decides *what* and *where* globally, the skips
hand the decoder the *fine detail* it would otherwise have lost. That combination is why the same
trunk is the default for every image→image job (segmentation, super-res, and here, denoising).

## 4 · Why `t` must be an input: the same pixels, a different answer

The forward process reaches the *same* `x_t` from many `(x_0, ε)` pairs at many levels. A given noisy
image at `t=100` (barely noised) and at `t=800` (mostly noise) call for very different amounts of
noise to be identified. So the network can't answer from the pixels alone — it needs to be *told*
which level it's at. We test this bluntly: build `x_t` at a known true `t`, then **lie** to the model
about `t` and watch `MSE(ε̂, ε)` as a function of the `t` we claim.
"""

# %%
torch.manual_seed(1)
xc = x[:64]
claimed = torch.arange(0, T, 25, device=DEV)          # the t-values we feed the model
fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.6))
for true_t in (100, 500, 800):
    eps = torch.randn_like(xc)
    ab = alpha_bar[true_t]
    xt = ab.sqrt() * xc + (1 - ab).sqrt() * eps        # x_t built at the TRUE level
    curve = []
    with torch.no_grad():
        for j in claimed:
            tj = torch.full((64,), int(j), device=DEV)
            curve.append(F.mse_loss(model(xt, tj), eps).item())   # feed the CLAIMED level
    curve = np.array(curve)
    ax.plot(claimed.cpu(), curve, label=f"x_t built at true t={true_t}")
    ax.axvline(true_t, color="gray", ls=":", lw=1)
    print(f"  true t={true_t:>3}: best MSE at claimed t={int(claimed[curve.argmin()]):>3}"
          f"  (min {curve.min():.3f} vs {curve.max():.2f} at the worst lie)")
ax.set_xlabel("timestep t we CLAIM (dotted = the true t)"); ax.set_ylabel("MSE(ε̂, ε)")
ax.set_title("feed the wrong t → the error blows up. The net genuinely uses t."); ax.legend()
fig.tight_layout(); plt.show()

# %% [markdown]
"""
Each curve bottoms out right at its true `t` (the dotted line) and climbs steeply on either side — tell
the model the image is noisier or cleaner than it is, and `ε̂` is wrong. That's the proof the timestep
is a *real* input, not decoration: the sinusoidal embedding (`timestep_embedding`, borrowed from the
`llm/` positional trick) is fed into **every** block via the `temb` `Linear`, so the whole network
recomputes its behaviour per level. One set of weights covers all `T` denoisers because `t` selects
which one to be.

## Recap

- **Image → image, not image → label.** No flatten-to-a-vector head: the output is a full `28×28`
  map (`ε̂`). That forces the down→up shape — rebuild the resolution you funnelled away.
- **Down / up = cheap global context.** Halving each side lets channels grow while doing *less* work;
  the bottleneck gives each output pixel a receptive field over the whole digit. We measured it.
- **Skip connections = the fine detail.** Turned off, `ε̂` goes blurry and the loss jumps — the `7×7`
  funnel can't carry pixel structure, so the skips hand it straight to the decoder. U-Net = autoencoder
  + detail highways.
- **`t` is a genuine input.** Feeding the wrong `t` blows up the error; the sinusoidal embedding is
  injected into every block, so one network is all `T` denoisers, selected by `t`.

Next: **`05` — sampling: the reverse process** — why generation has to be *iterative* (a one-shot `x̂_0`
straight from pure noise is mush), closing Phase A with the full working generator understood end to end.
"""
