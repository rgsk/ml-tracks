# CNN — Layer 5: the head + loss — from feature map to a scored prediction

Layers 2–4 built a feature extractor: `(B,1,28,28) → (B,64,7,7)`, a little `7×7` grid of 64-channel
feature vectors. That's not a prediction yet. This layer bolts on the two final pieces — a **head**
that turns the map into 10 class scores, and a **loss** that scores them — and then runs two cheap
sanity checks that catch almost every wiring bug before you waste a training run. Verify every number
with `../cnn.py` (`exp_5_head_and_loss`). Console-only, so the payoffs are measurements you run.

---

## 1. The head: global average pooling → a Linear

We have a `(64, 7, 7)` feature map per image and need a 10-vector of class logits. Two ways to bridge
that gap.

**The old way — flatten + Linear.** Lay the `64·7·7 = 3136` numbers into one vector and hit it with
`Linear(3136 → 10)`. It works, but it's tied to an *exact* `7×7` spatial size and costs `3136·10 + 10
= 31,370` weights — and it isn't translation-invariant (a feature in the top-left feeds different
weights than the same feature in the middle).

**The modern way — global average pooling (GAP).** Collapse each channel's whole `7×7` map to a
single number: its average. Ground-up, for channel `c`:

```
v[c] = mean over H,W of f[c, :, :]  =  (1 / (H·W)) · Σ_h Σ_w  f[c, h, w]
```

That gives one number per channel → a **64-vector**, read as *"how strongly is feature `c` present
anywhere in the image?"* Then a single `Linear(64 → 10)` maps it to logits:

```python
f = self.features(x)          # (B, 64, 7, 7)
v = f.mean(dim=(2, 3))        # GAP over H,W -> (B, 64)     <- no params, any H,W
logits = self.head(v)         # Linear(64->10) -> (B, 10)
```

```
features (32, 64, 7, 7)  --mean over H,W-->  (32, 64)  --Linear(64->10)-->  (32, 10)
```

Two wins, same flavor as the conv's weight-sharing:

```
GAP + Linear(64->10)       = 64·10 + 10   =    650 params   (works for ANY H,W)
flatten + Linear(3136->10) = 3136·10 + 10 = 31,370 params   (~50x more, locked to 7x7)
```

- **~50× fewer params** (GAP itself has *zero* parameters).
- **Translation-invariant and size-independent** — GAP averages over position, so *where* a feature
  fires doesn't matter, only *how much*; and any `H×W` pools to the same 64-vector. This position-
  invariance is a big win for *large* natural-image nets (an object can be anywhere), which is why
  GAP is the modern default there.

> ⚠️ That same invariance is a **liability** on small *centered* digits, where position is exactly
> what tells classes apart — it's why the GAP model plateaus around ~95% and a position-keeping
> **flatten** head reaches ~99% (Layer 7 measures and explains this). Match the head's invariance to
> the task.

---

## 2. The loss: cross-entropy, and why untrained ≈ `ln 10`

The 10 logits `z` aren't probabilities. **Softmax** turns them into a distribution:

```
p[i] = exp(z[i]) / Σ_j exp(z[j])           # non-negative, sums to 1
```

**Cross-entropy** is then the negative log-probability the model assigned to the *true* class `t`:

```
loss = − log( p[t] )  =  − log( softmax(z)[true class] )
```

Minimizing it pushes `p[t] → 1`. `F.cross_entropy` fuses the softmax and the log for numerical
stability, so you hand it **raw logits**, not probabilities (applying softmax yourself first is a
classic double-softmax bug).

**The calibration check that falls out of the math.** At initialization the weights are small and
random, so the 10 logits are all ≈ equal, which makes softmax ≈ uniform: `p[t] ≈ 1/10`. So the
untrained loss should sit at:

```
loss ≈ − log(1/10) = ln 10 ≈ 2.3026
```

```
measured untrained loss = 2.3140     (≈ ln 10  -> softmax, labels, and init are all wired right)
```

This is a genuinely useful check: if your untrained loss is far from `ln C` (say 15, or 0.02),
something is wrong *before* you train — labels out of range, logits pre-softmaxed, a wrong class
count, or an exploding init. For `C` classes the target is always `ln C`.

---

## 3. The wiring test: overfit one batch → loss ≈ 0

Before spending a real training run, prove the machine *can* learn at all. Take a **single batch**
(32 images) and minimize its loss over and over. A correctly-wired model with enough capacity must be
able to **memorize 32 examples** — driving loss to ~0 and accuracy to 100%:

```python
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for step in range(201):
    logits = model(xb)
    loss = F.cross_entropy(logits, yb)
    opt.zero_grad(); loss.backward(); opt.step()
```

```
step   0: loss 2.3140   acc   6.2%      <- starts at ln 10, chance accuracy
step  40: loss 1.4805   acc  40.6%
step  80: loss 0.4690   acc  90.6%
step 120: loss 0.0349   acc 100.0%
step 160: loss 0.0101   acc 100.0%
step 200: loss 0.0055   acc 100.0%
```

Loss collapses from `ln 10` to ≈ 0; accuracy hits 100%. **That is the whole point of the test** — it
isolates *model/plumbing* bugs from *data/generalization* ones:

- If a batch **won't** overfit, the bug is in the wiring — a detached graph, frozen (`requires_grad
  = False`) params, a shape mismatch silently broadcasting, a dead ReLU, or a hopeless learning rate.
  These are invisible in a full training run (you just see "loss not going down") but obvious here.
- If it **does** overfit but the full run generalizes poorly, the model is fine and the problem is
  data/regularization/schedule — a completely different place to look.

Overfitting is *desired* here: we're testing capacity and gradient flow, not generalization. Real
training (Layer 6) uses all of MNIST, where memorizing isn't possible and accuracy reflects genuine
learning.

---

## Summary

| piece | what it is | the payoff you run |
|---|---|---|
| head (GAP) | average each channel's map to 1 number → 64-vec → `Linear(64→10)` | `(32,64,7,7)→(32,64)→(32,10)`, 650 vs 31,370 params |
| loss | `−log(softmax(z)[true])`; feed raw logits | — |
| calibration | untrained logits ≈ uniform ⇒ loss ≈ `ln C` | measured `2.31 ≈ ln 10` |
| wiring test | overfit one batch → loss ≈ 0, acc 100% | `2.31 → 0.005`, `6% → 100%` |

**One-sentence compression:** global average pooling turns the `(64,7,7)` map into a translation-
invariant 64-vector for ~50× fewer params than flatten, a `Linear(64→10)` gives logits, cross-entropy
scores them (untrained loss `≈ ln 10`, a free calibration check), and overfitting one batch to ~0
proves the whole net is wired before any real training.

Next (Layer 6): **train it** — the real loop on all of MNIST, accuracy climbing to ~99% (far above
the MLP for a fraction of the params), and a saved grid of held-out digits with predicted labels →
`06_train.md`.

---

*Numbers: `python ../cnn.py` (`exp_5_head_and_loss`).*
