# Diffusion — roadmap (top-down, image-gen-from-scratch, notebook edition)

**How this is taught: the whole game first.** `01` builds a real diffusion model — a tiny
time-conditioned U-Net — trains it on MNIST, and **samples brand-new digits out of pure noise in the
first minute**, so you *see it work*: watch a grid of static resolve into digits the model was never
shown. Every notebook after that **opens one box of that exact pipeline** and explains the *why* —
measured, not asserted. You always know where a detail lives, because you already ran the model it's
part of.

It's the same Karpathy/fast.ai move the other tracks use: get a basic version *generating* with a
rough mental map of what's happening, then dig in and break each piece apart. This is the notebook
re-presentation of the bottom-up `diffusion/` build — same concepts, retold top-down, one runnable
lesson at a time.

The through-line of the whole track is one loop: **add a lever, watch the samples get
better / faster / more controllable.** That "see the output improve as each piece goes in" payoff is
the whole reason diffusion is the right vehicle for the image-gen goal — the feedback is a *picture*,
not just a loss number, and on MNIST (RTX 4060) it's minutes, like char-Shakespeare was for `llm/`.

---

## Format (jupytext paired notebooks)

Each experiment is **one notebook** — a lesson you read top-to-bottom, with prose, code, and figures
inline. Every notebook is a **jupytext pair**:

- `walkthroughs/NN_name.py` — the **source of truth** (`py:percent`); clean git diffs, editable in the
  editor. This is what you edit.
- `walkthroughs/NN_name.ipynb` — the **rendered pair**: same cells plus executed outputs (prints,
  sample grids, figures). This is what you read. It's **gitignored** and regenerated.

Rebuild/execute a notebook after editing its `.py`:

```bash
uv run jupytext --to ipynb --execute nb/diffusion/walkthroughs/01_whole_game.py   # re-run all cells
```

```-
nb/diffusion/
  roadmap.md                 <- this file: the table of contents + how it's taught
  walkthroughs/
    01_whole_game.py     ⇄ .ipynb  the whole game: tiny U-Net, train on MNIST, sample digits from noise
    02_forward_process.py⇄ .ipynb  add noise on a schedule: the closed-form jump to any noise level
    03_training_target.py⇄ .ipynb  predict the noise ε (not the image); the MSE loss + wiring checks
    04_the_unet.py       ⇄ .ipynb  open the model: down/up + skips, and why the timestep t is an input
    05_sampling.py       ⇄ .ipynb  the reverse loop: why generation is iterative (one-shot is mush)
    06_ddim.py           ⇄ .ipynb  deterministic few-step sampling: 1000 → 50 → 10, a free inference knob
    07_parameterization.py⇄ .ipynb ε vs v-prediction, linear vs cosine schedule — sharper at low steps
    08_ema.py            ⇄ .ipynb  average the weights for sampling: cleaner samples for ~free
    09_cfg.py            ⇄ .ipynb  class-conditioning + classifier-free guidance; the guidance-scale knob
    10_latent.py         ⇄ .ipynb  diffuse in a VAE latent, not pixels — the Stable-Diffusion trick
    11_dit.py            ⇄ .ipynb  swap the U-Net for a transformer + AdaLN-Zero (reuses the llm/ track)
    12_flow_matching.py  ⇄ .ipynb  reframe as a straight-line velocity field: the SD3/Flux SOTA framing
    ── advanced (optional; the current frontier, past the fundamentals) ──
    13_edm.py            ⇄ .ipynb  Karras EDM: schedule + preconditioning + loss-weighting as one design
    14_fast_solvers.py   ⇄ .ipynb  high-order ODE samplers (DPM-Solver++, Heun): fewer steps, no retrain
    15_distillation.py   ⇄ .ipynb  consistency / LCM / Turbo: distill the sampler to 1–4 steps
    16_text_cond.py      ⇄ .ipynb  text encoder + cross-attention / MMDiT: label → real text prompts
    17_guidance_plus.py  ⇄ .ipynb  guidance refinements: autoguidance, limited-interval CFG
  custom/                    <- from-scratch impls (noise schedule, closed-form forward, the DDPM step,
                                the DDIM step, CFG mix); each runs standalone as a self-test vs torch
  model.py                   <- the cleaned-up U-Net + sampler, assembled once we understand each piece
  data/                      <- shared MNIST cache lives at nb/data/mnist.npz ([-1,1], no torchvision)
  checkpoints/               <- ddpm.pt: trained by 01, loaded by later notebooks (no retrain)
  samples/                   <- generated image grids (gitignored) — the artifact "did it improve?"
```

Notebooks stay independent: `01` trains and **saves `checkpoints/ddpm.pt`**; later notebooks **load it
if present, else train a quick one**. So you can open any notebook on its own, and editing `05` never
re-runs `01`.

---

## Phase A — the DDPM core (build & understand the generator, on MNIST)

**`01` — the whole game.** Build a tiny **time-conditioned U-Net**, train it on MNIST to predict the
noise added to an image, then **sample**: start from pure Gaussian noise and iteratively denoise into
a digit. **Watch a grid of generated digits appear from static.** One-line narration per part — three
ideas, no rigor yet: *add noise on a schedule (forward), train a net to predict that noise (loss),
walk back from noise to a digit (reverse).*
> After this you have the map. Everything below zooms into **one box you already ran.**

**`02` — the forward process.** How we add noise: the closed form
`x_t = sqrt(ᾱ_t)·x_0 + sqrt(1-ᾱ_t)·ε`, the noise **schedule** (`β_t → α_t → ᾱ_t`), and *why* the
coefficients are square roots — **variance-preserving**, so `Var(x_t) ≈ 1` at every `t`. **Lever →
effect:** watch a digit dissolve into pure `N(0,I)` as `t` grows; linear vs cosine changes how fast.
This one identity is reused by the loss (`03`), sampling (`05`), and DDIM (`06`).

**`03` — the training target.** Why the net predicts **ε** (the noise), not the image, and why that's
the *scale-stable* target (`ε ~ N(0,1)` at every `t`, so no per-`t` rescaling of the loss). The whole
objective is just `MSE(ε̂, ε)`. Wiring checks that carry over from the CNN track: untrained loss ≈ 1,
overfit-one-batch → 0. Note the ε ⇔ x_0 equivalence — same target, two views (`07` adds a third).

**`04` — the denoiser: why a U-Net + time conditioning.** Open the model `01` ran. **Down/up with skip
connections** — downsample to see globally, upsample back, and the skips carry the fine spatial detail
the bottleneck would lose (an image-to-image job, unlike the CNN classifier's collapse-to-a-label
head). Reuses the conv/downsample pieces straight from `nb/cnn`. And *why the timestep `t` must be an
input*: the same `x_t` means different things at different noise levels, so the net has to be told
*which* level it's denoising.

**`05` — sampling: the reverse process.** Why generation is **iterative**. One-shot `x̂_0` from pure
noise is a blurry mush (we *see* it); ancestral DDPM walks back one small step at a time — the reverse
**posterior mean/variance**, and the noise→digit trajectory across steps. This closes Phase A: you now
understand the full working generator end to end.

## Phase B — faster & better sampling (levers on the working generator)

**`06` — DDIM: deterministic, few-step sampling.** A non-Markovian reverse process that uses the *same
trained model* with far fewer steps. **Lever → effect:** steps `1000 → 50 → 10` — watch quality trade
against speed. A pure **inference-time** knob (free, instant, no retrain) — the diffusion analogue of
the `llm/` KV-cache speedup.

**`07` — parameterization & schedule.** The prediction target and the noise schedule as two tunable
choices: **`ε` vs `v`-prediction** (`v` = the scale-balanced mix that's stable across all `t`) and
**linear vs cosine** `β`. **Lever → effect:** sharper, more stable samples, *especially at the low
step counts* `06` unlocked.

**`08` — EMA of the weights.** Keep an exponential moving average of the model weights and **sample
from the average, not the last step.** **Lever → effect:** noticeably smoother, cleaner samples for
~free — a train-time bookkeeping trick with an outsized visible payoff.

## Phase C — conditional generation (ask for what you want)

**`09` — class conditioning + classifier-free guidance (CFG).** Condition the U-Net on a class label;
train with **label dropout** so one model learns both the conditional and unconditional scores; at
sampling, **mix** the two predictions. **Lever → effect:** the **guidance scale `w`** — crank it and
samples get sharper and more on-class but less diverse (the diffusion cousin of the `llm/`
`temperature` / `top-k` knob). Now you generate the *specific* digit you ask for, not a random one.

## Phase D — scale up (past MNIST)

**`10` — latent diffusion.** Train a small VAE (or reuse an autoencoder) to **compress** images, then
run the *entire* diffusion pipeline in the small latent instead of in pixels. **Lever → effect:** move
from MNIST to **CIFAR / color** at higher resolution without the compute blowing up — the trick that
makes Stable Diffusion tractable.

**`11` — DiT: the Diffusion Transformer.** Replace the U-Net backbone with a **transformer** +
**AdaLN-Zero** conditioning (the timestep/class signal modulates each block's norm). **Reuses the
`nb/llm` transformer directly** — this is where the image and language tracks meet. **Lever → effect:**
sample quality scales with depth/width, the same lever that drives LLMs.

## Phase E — the modern reframe

**`12` — flow matching / rectified flow.** Reframe diffusion as learning a **straight-line velocity
field** that transports noise to data (SD3, Flux). **Lever → effect:** fewer, *straighter* sampling
steps for the same quality — the current SOTA framing, and a clean capstone that recasts everything in
Phases A–D as one instance of a more general idea.

## Advanced — the current frontier (optional, past the fundamentals)

Phases A–E already teach the SOTA *foundations* and the SOTA *framing* (flow matching) — enough to
reason about any 2026 model. This section is where the **actual frontier** lives: each item is a
natural extension of a notebook you already built, not a new track. Take them once the core runs.

**`13` — the EDM framework (Karras 2022).** The single most clarifying "modern reframe" between DDPM
and flow matching: recast the **noise schedule, the network preconditioning, and the loss weighting**
as *one* design space parameterized by the noise level `sigma`, instead of three ad-hoc choices. **Lever
→ effect:** the same model trains more stably and samples better at low step counts — and everything in
Phases A–B (`ε`/`v`, cosine, the difficulty-per-`t` story) falls out as *special cases* of one clean
formulation. If you add only one advanced notebook, add this one; it makes the rest click.

**`14` — high-order ODE samplers.** DDIM (`06`) is a *first-order* solver of the reverse ODE. Real
models use **higher-order** solvers — DPM-Solver++, EDM's Heun (2nd-order) — that take bigger, more
accurate steps. **Lever → effect:** `06`'s 50-step DDIM quality in ~10–20 steps, still **training-free**
(a pure sampler swap on the model you already have). This is what production inference actually runs.

**`15` — distillation & few-step models.** The frontier of *fast* sampling: instead of a better solver,
**train a student to jump many steps at once** — **consistency models**, **LCM** (latent consistency),
**Turbo**-style adversarial distillation. **Lever → effect:** `1–4` step generation (vs 1000), the tech
behind real-time image gen. Ties straight back to your `llm/` intuition that distillation trades a
one-time training cost for a permanent inference win.

**`16` — text conditioning: from labels to prompts.** Phases A–D condition on a *class label* (one of
10). Real text-to-image swaps that for a **frozen text encoder (T5 / CLIP)** feeding the DiT via
**cross-attention**, or the **MMDiT** joint text↔image attention SD3 uses. **Lever → effect:** the leap
from "generate a 7" to "generate *what the sentence says*" — and it directly reuses the attention +
conditioning plumbing from `nb/llm` and `09`/`11`. (Needs a bigger dataset than MNIST — this is the
bridge from a toy to a real system.)

**`17` — guidance refinements.** CFG (`09`) is a blunt global knob; the frontier makes it surgical:
**autoguidance** (guide with a weaker version of the *same* model), **limited-interval CFG** (only apply
guidance at the noise levels where it helps). **Lever → effect:** the sharpness of high guidance without
the diversity collapse and artifacts — a small, high-return polish on the knob you already built.

> Two natural stopping points: after **`12`** you understand modern diffusion end to end (interview-
> ready); after **`13`–`15`** you understand how it's actually *deployed* (fast, stable, production).
> `16`–`17` are the reach into real text-to-image systems.

---

### Threads that recur (called out as they appear, not separate items)
- **The forward/closed-form identity** — `02`'s `x_t = sqrt(ᾱ)x_0 + sqrt(1-ᾱ)ε` powers the loss
  (`03`), sampling (`05`), and DDIM (`06`); learn it once, reuse it everywhere.
- **ε ⇔ x_0 ⇔ v** — one target, three views; `03` introduces the first two, `07` makes it explicit.
- **Inference-time levers** (steps, guidance scale) vs **train-time levers** (schedule, EMA, capacity)
  — which give instant feedback and which need a retrain.
- **Conditioning plumbing** — CFG (`09`) and DiT's AdaLN (`11`) both inject a signal into the network;
  the same idea as `llm/` SFT-conditioning and the AdaLN modulation used across the other tracks.
- **What carries in from `nb/cnn`** — the U-Net's down/up trunk *is* the CNN track's conv + downsample
  stack, now wired for image-to-image instead of image-to-label.
- **Did it improve?** — every experiment's payoff is a **sample grid** written to `samples/`, so the
  answer is always a picture.

## Setup notes
- MNIST comes from the shared `nb/data/mnist.npz` (keras npz, no torchvision, already in `[-1,1]` —
  exactly diffusion's expected input range). `01` loads it the same way `nb/cnn` does.
- `torchvision` is only needed at `10` (CIFAR / the VAE) — add it to the env then; Phases A–C are
  torch + matplotlib only.
- Checkpoints and sample grids are gitignored; the working artifact writes PNG grids so progress is
  always visible.
