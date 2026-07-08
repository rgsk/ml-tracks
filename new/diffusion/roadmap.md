# Diffusion — roadmap (top-down)

**Whole game first.** `exp_1` trains a small diffusion model on MNIST and then **samples brand-new
digits out of pure noise** — you watch generation *work* before we derive a single equation. Then
each experiment opens **one box** of that exact pipeline and explains the why, measured. Same move as
`new/cnn/`: see the result, get the map, then dig in.

Diffusion is a **big** topic, so `walkthroughs/` holds several subtopic folders. We start with the
core method (**`ddpm/`**); DDIM, guidance, latent, DiT, flow-matching become sibling folders later.

---

## Folder layout

```
new/diffusion/
  roadmap.md
  custom/                      from-scratch math (schedule, closed-form forward, DDPM step…), self-testing
  walkthroughs/
    ddpm/                      the core method (whole game + its boxes)
      ddpm.py                  experiments exp_1 .. exp_N
      notes/*.md               write-ups + figures
      figures/{experiments,generated,handmade}/
    ddim/  guidance/  latent/  dit/  flow/     ← later subtopics, each its own folder
  data/mnist.npz               shared cache (keras npz, no torchvision, normalized to [-1,1])
  <root>                       the cleaned-up model/schedule/sampler, assembled after the walkthroughs
```

---

## The experiments (`ddpm/`)

**`exp_1` — the whole game.** Build a tiny time-conditioned U-Net, train it on MNIST to predict the
noise added to an image, then **sample**: start from pure Gaussian noise and iteratively denoise into
a digit. **Watch a grid of generated digits appear from static.** Rough narration only — three ideas:
*add noise on a schedule (forward), train a net to predict that noise, walk back from noise to a digit
(reverse).*
> After this you have the map. Everything below opens one box of this pipeline.

**`exp_2` — the forward process.** How we add noise: the closed form `x_t = √ᾱ_t·x_0 + √(1−ᾱ_t)·ε`,
the noise **schedule** (`β_t → α_t → ᾱ_t`), and why the coefficients are `√` (variance-preserving:
`Var(x_t) ≈ 1` for all `t`). *See* a digit dissolve into noise across `t`.

**`exp_3` — the training target.** Why the net predicts **ε** (the noise), not the image; the loss is
just `MSE(ε̂, ε)`. Why ε is the scale-stable target (~N(0,1) at every `t`), and how per-`t` difficulty
varies. Untrained loss ≈ 1; overfit-one-batch → 0 (the wiring checks).

**`exp_4` — the denoiser (why a U-Net + time conditioning).** Open the model: down/up with skip
connections (keep spatial detail), and *why the timestep `t` must be an input* (the same `x_t` means
different things at different noise levels). Reuses the conv/downsample pieces from the CNN track.

**`exp_5` — sampling (the reverse process).** Why generation is **iterative**: one-shot `x̂_0` from
pure noise is mush; ancestral DDPM walks back step by step. The posterior mean/variance, and the
noise-→-digit trajectory.

**Later folders:** `ddim/` (few-step deterministic sampling) · `guidance/` (class-conditioning + CFG,
the guidance-scale knob) · `latent/` (diffuse in a VAE latent → CIFAR) · `dit/` (transformer + AdaLN)
· `flow/` (rectified flow / flow matching).

---

*Run: `python walkthroughs/ddpm/ddpm.py`. Notes in `walkthroughs/ddpm/notes/`, figures in
`walkthroughs/ddpm/figures/`.*
