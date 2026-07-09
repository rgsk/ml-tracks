# CNN — roadmap (top-down, notebook edition)

**How this is taught: the whole game first.** `01` builds a real CNN and trains it to ~99% on MNIST
**in the first few seconds**, so you *see it work* and get a rough mental map. Every notebook after
that **opens one box of that exact model** and explains the *why* — measured, not asserted. You always
know where a detail lives, because you already ran the model it's part of.

It's the Karpathy/fast.ai move: get a basic version working with a rough estimate of what's happening,
then dig in and break each piece apart.

---

## Format (jupytext paired notebooks)

Each experiment is **one notebook** — a lesson you read top-to-bottom, with prose, code, and figures
inline. Every notebook is a **jupytext pair**:

- `walkthroughs/NN_name.py` — the **source of truth** (`py:percent`); clean git diffs, editable in the
  editor. This is what you edit.
- `walkthroughs/NN_name.ipynb` — the **rendered pair**: same cells plus executed outputs (prints,
  figures). This is what you read.

Rebuild/execute a notebook after editing its `.py`:

```bash
uv run jupytext --to ipynb --execute nb/cnn/walkthroughs/01_whole_game.py   # re-run all cells
uv run jupytext --sync            nb/cnn/walkthroughs/01_whole_game.py       # propagate edits, keep outputs
```

```
nb/cnn/
  roadmap.md                 <- this file: the table of contents + how it's taught
  walkthroughs/
    01_whole_game.py  ⇄ .ipynb   the whole game: build SmallCNN, train, read digits
    02_features.py    ⇄ .ipynb   open `features`: what a Conv2d computes (naive_conv2d)
    03_why_conv.py    ⇄ .ipynb   why conv, not flatten+MLP (locality, equivariance, params)
    04_stack_relu.py  ⇄ .ipynb   conv→relu→conv: receptive field + why ReLU is load-bearing
    05_downsample.py  ⇄ .ipynb   stride-2 + ×2 channels: reach in fewer layers, bounded cost
    06_head_loss.py   ⇄ .ipynb   flatten→Linear vs global-avg-pool; cross-entropy; wiring checks
    07_clean_model.py ⇄ .ipynb   assemble the clean model; what carries into the diffusion U-Net
  custom/                    <- from-scratch impls (naive_conv2d, …); each runs standalone as a self-test
  model.py                   <- the cleaned-up SmallCNN, assembled once we understand each piece
  checkpoints/               <- smallcnn.pt: trained by 01, loaded by later notebooks (no retrain)
```

Notebooks stay independent: `01` trains and **saves `checkpoints/smallcnn.pt`**; later notebooks
**load it if present, else train a quick one**. So you can open any notebook on its own, and editing
`05` never re-runs `01`.

---

## The notebooks

**`01` — the whole game.** Build `SmallCNN` (a `features` stack of `conv → relu` that halves H,W and
grows channels, then a `flatten → Linear` head). Train on MNIST, **watch test accuracy climb to
~99%**, read a grid of held-out digits. One-line narration per part — no rigor yet. Payoff on screen
immediately.
> After this you have the map. Everything below zooms into **one box you already ran.**

**`02` — open `features`: what is a `Conv2d`, really?** The sliding kernel (cross-correlation), weight
sharing, the output-size formula. Build `naive_conv2d` from scratch in `custom/` and match `F.conv2d`
to ~0. Fire hand-set edge kernels → **see** the feature maps.

**`03` — why conv, not the flatten+MLP we already had?** Locality, translation *equivariance* (shift
the input → the feature map just shifts), and the parameter blow-up. Why the conv trunk earns its
place over a dense net.

**`04` — why `conv → relu → conv`?** Depth grows the receptive field; and *without* the ReLU a stack
of convs collapses back to a single conv (we measure the collapse, and that a ReLU breaks it).

**`05` — why `stride=2` (H/2, W/2) and ×2 channels?** Downsampling: stride makes the receptive field
cover more input pixels in *fewer* layers, and "resolution down, channels up" keeps compute bounded
while features get richer.

**`06` — why `flatten → Linear`, and is it the right head?** The head + cross-entropy; the tempting
global-average-pool alternative and *why it underperforms* on centered digits; plus two wiring checks
(untrained loss ≈ `ln 10`, overfit one batch → 0).

**`07` — assemble the clean model** (`model.py`) and note which pieces carry straight into the U-Net
we'll build for diffusion.
