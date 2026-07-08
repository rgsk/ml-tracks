# CNN — roadmap (top-down)

**How this one is taught: the whole game first.** Unlike a bottom-up build (primitive → primitive →
… → model, where the working thing only shows up at the end), here `exp_1` builds a real CNN and
trains it to ~99% on MNIST **in the first few minutes**, so you *see it work* and get a rough mental
map. Every experiment after that **opens up one piece of that exact model** and explains the *why* —
measured, not asserted. You always know where a detail lives, because you already ran the model it's a
part of.

It's the Karpathy/fast.ai move: get a basic version working with a rough estimate of what's
happening, then dig in and break each piece apart.

---

## Folder layout (this is the reusable format)

```
new/cnn/
  roadmap.md              <- this file: how we'll teach the topic
  custom/                 <- from-scratch impls we build to prove "no magic" (naive_conv2d, …);
                             each runs standalone as its own self-test vs torch
  walkthroughs/
    cnn/                  <- one folder per topic (cnn now; diffusion subtopics later)
      cnn.py             <- the experiments (exp_1 .. exp_N); run each, watch the output
      notes/
        *.md             <- one write-up per experiment, with figures
      figures/
        figures.py       <- generates illustrative diagrams (no experiment counterpart)
        experiments/     <- figures the experiments (exp_*) produce
        generated/       <- figures figures.py produces
        handmade/        <- hand-drawn diagrams
  <root: new/cnn/>        <- the cleaned-up model, assembled once we understand each piece
                             (sometimes built with TODOs to fill in, sometimes straight)
```

Small topic (CNN) → **one** topic folder with **one** walkthrough file. A big topic (diffusion) →
several topic folders under `walkthroughs/`, each with its own file(s), notes, and figures.

---

## The experiments

**`exp_1` — the whole game.** Build `SmallCNN`: a `features` stack (`conv → relu → conv → relu → …`
that halves H,W and grows channels) then a `flatten → Linear` head. Train on MNIST, **watch test
accuracy climb to ~99%**, and read a grid of held-out digits. Just a one-line rough narration per
part — no rigor yet. Payoff on screen immediately.
> After this you have the map. Everything below zooms into **one box you already ran.**

**`exp_2` — open `features`: what is a `Conv2d`, really?** The sliding kernel (cross-correlation),
weight sharing, the output-size formula. Build `naive_conv2d` from scratch in `custom/` and match
`F.conv2d` to ~0. Fire hand-set edge kernels → **see** the feature maps.

**`exp_3` — why conv, not the flatten+MLP we already had?** Locality, translation *equivariance*
(shift the input → the feature map just shifts), and the parameter blow-up. Why the conv trunk earns
its place over a dense net.

**`exp_4` — why `conv → relu → conv`?** Depth grows the receptive field; and *without* the ReLU a
stack of convs collapses back to a single conv (we measure the collapse, and that a ReLU breaks it).

**`exp_5` — why `stride=2` (H/2, W/2) and ×2 channels?** Downsampling: stride makes the receptive
field cover more input pixels in *fewer* layers, and "resolution down, channels up" keeps compute
bounded while features get richer.

**`exp_6` — why `flatten → Linear`, and is it the right head?** The head + cross-entropy; the
tempting global-average-pool alternative and *why it underperforms* on centered digits; plus two
wiring checks (untrained loss ≈ `ln 10`, overfit one batch → 0).

**`exp_7` — assemble the clean model** (`new/cnn/` root) and note which pieces carry straight into the
U-Net we'll build for diffusion.

---

*Run: `python walkthroughs/cnn/cnn.py`. Notes land in `walkthroughs/cnn/notes/`, figures in
`walkthroughs/cnn/figures/`.*
