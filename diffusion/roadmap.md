# Diffusion Roadmap — image generation from scratch

A fourth from-scratch track, sibling to `llm/` and `rl/`, same exercise-driven style.
Goal: build image generation the way diffusion is actually practiced today — DDPM →
DDIM → classifier-free guidance → latent diffusion → DiT → flow matching — and **watch
sample quality visibly climb as each piece goes in.** That "add a lever, see the output
get better/faster/more controllable" loop is the whole reason we picked diffusion for
the image-gen goal.

Fast on this machine (MNIST/CIFAR, RTX 4060) — the feedback loop is minutes, like
char-Shakespeare was for `llm/`.

## Three kinds of code (mirrors the `llm/` track)
1. **`walkthroughs/`** — layered scratch files, one concept at a time. Each layer =
   a builder fn + an `exp_*` that prints/plots so you SEE it. `run_experiments()`
   comment-toggles which fire; `--out FILE` tees. Model to match: `llm/walkthroughs/rope.py`.
   **We build ONE layer at a time and pause for review.**
2. **The real runnable package** — `config.py`, `schedule.py`, `model.py`,
   `diffusion.py`, `train.py`, `sample.py`. Uses torch ops (never `custom/`), actually
   trains on MNIST and **generates image grids** — the working artifact, like `llm/`'s
   `train.py`/`model.py`. This is where "run it and see digits appear" lives.
3. **`custom/`** — from-scratch reimplementations of the diffusion-specific math
   (schedule, closed-form forward, the DDPM loss, the DDIM step, CFG), each verified
   against a reference — to understand what the library does under the hood, exactly
   like `llm/custom/cross_entropy.py`.

> Numbers below are build order. Every item names its **lever → visible effect** so the
> impact is never abstract.

---

## Phase 0 — DDPM core (the foundation, on MNIST)

**1. Forward process & noise schedule** `walkthroughs/forward_process.py`
Progressively add Gaussian noise to an image; the closed-form jump to any noise level
`x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε`; linear vs cosine β schedule; signal-to-noise
ratio. **Lever → effect:** watch an image dissolve into pure N(0,I) as t grows; cosine
vs linear changes how fast. **← START HERE**

**2. The denoiser & training objective** `walkthroughs/denoiser_and_loss.py` + `model.py`
The model predicts the noise ε; the "simple" MSE loss (Ho et al. 2020) and why
predicting ε ⇔ predicting x_0. A small U-Net (or MLP for a 1st pass). **Lever → effect:**
loss drops → the ε-prediction on held-out noised images gets accurate.

**3. Sampling — the reverse loop** `walkthroughs/sampling.py` + `sample.py`
Ancestral sampling: start from noise, denoise step by step back to an image. **This is
the first working generator** — train on MNIST, watch noise → digits. **Lever → effect:**
the artifact itself; a grid of sampled digits.

## Phase 1 — Faster & better sampling (levers on the working generator)

**4. DDIM — deterministic, few-step sampling** `custom/ddim.py`
Non-Markovian reverse process; same model, far fewer steps. **Lever → effect:** sampling
steps 1000 → 50 → 10, watch quality-vs-speed trade off — a pure inference knob (free,
instant), the diffusion analogue of your KV-cache speedup.

**5. Schedule & parameterization** `custom/schedule.py`
Cosine schedule, `v`-prediction vs `ε`-prediction. **Lever → effect:** sharper, more
stable samples, especially at low step counts.

**6. EMA of weights** (fold into `train.py`)
Exponential moving average of model weights for sampling. **Lever → effect:** noticeably
smoother, cleaner samples for ~free.

## Phase 2 — Conditional generation

**7. Class conditioning + Classifier-Free Guidance (CFG)** `walkthroughs/cfg.py`
Condition on a class label; train with label-dropout; at sampling, mix conditional and
unconditional predictions. **Lever → effect:** the guidance scale `w` — crank it and
watch samples get sharper and more on-class but less diverse (the diffusion cousin of
your `temperature`/`top-k` knob). Generate the digit you ask for.

## Phase 3 — Scale up

**8. Latent diffusion** `walkthroughs/latent_diffusion.py`
Train a VAE (or reuse an autoencoder) to compress images; run diffusion in the small
latent instead of pixels. **Lever → effect:** move from MNIST to CIFAR/color at higher
res without the compute blowing up — the trick that makes Stable Diffusion tractable.

**9. DiT — Diffusion Transformer** `walkthroughs/dit.py`
Replace the U-Net with a transformer backbone + **AdaLN-Zero** conditioning. **Reuses
your `llm/` transformer and the AdaLN idea directly.** **Lever → effect:** quality scales
with depth/width — where diffusion meets everything else you've built.

## Phase 4 — The modern reframe

**10. Flow matching / rectified flow** `walkthroughs/flow_matching.py`
Reframe diffusion as learning a straight-line velocity field from noise to data (SD3,
Flux). **Lever → effect:** fewer, straighter sampling steps; the current SOTA framing.

---

## Threads that recur (call out as they appear, not separate items)
- **The forward/closed-form identity** — item 1's `x_t = sqrt(ᾱ)x_0 + sqrt(1-ᾱ)ε` is
  reused by the loss (item 2), sampling (item 3), and DDIM (item 4).
- **ε ⇔ x_0 ⇔ v parameterization** — same target, three views; item 5 makes it explicit.
- **Inference-time levers** (steps, guidance scale) vs **train-time levers** (schedule,
  EMA, capacity) — which give instant feedback and which need a retrain.
- **Conditioning plumbing** — CFG (item 7) and DiT's AdaLN (item 9) both inject a signal;
  same idea as `llm` SFT-conditioning and world_model's AdaLN.

## Setup notes
- Needs `torchvision` (MNIST/CIFAR) once we hit item 2's training — add to the env then.
  Item 1 is self-contained (torch + matplotlib only).
- Data/checkpoints/sample grids get gitignored; the working artifact writes PNG grids
  so "did it improve?" is always a picture, not just a loss number.
