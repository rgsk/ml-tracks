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
# LLM · 08 — from logits to text: sampling

`07` finished the model's *training*: the architecture is built (`04`–`06`) and now
well-driven (`07`), and the val loss is as low as this toy gets. **Nothing below
changes a single weight.** Every experiment here happens at *decode time*, with
`torch.no_grad()`, on the finished model — and yet the text it writes ranges from a
stuck repeating loop to fluent Shakespeare-shaped babble to pure noise. Same weights,
same loss, wildly different output. That range is entirely the work of the four knobs
in this notebook.

The model doesn't emit text. At each step it emits **one vector of 65 logits** — a score
for every character that could come next. Turning that vector into an actual character
is a *choice*, and `01`'s `generate` made the simplest one on faith:

    probs = F.softmax(logits[:, -1, :], dim=-1)     # the model's raw distribution
    nxt = torch.multinomial(probs, num_samples=1)   # ...draw from it, tail and all

That's *one* option among many, and not the one any real system ships. Here are the
four, in the order they compose into a decode step:

1. **Greedy** *(the baseline that fails)* — always take the argmax. Deterministic, and
   the most likely token at every step. We watch it walk straight into a **repeating
   loop**, then measure *why*: real text is not the most-likely text. This is the
   notebook's central idea, and every knob after it is a way of not doing this.
2. **Temperature** — scale the logits by `1/T` before the softmax. We derive what that
   actually does to the distribution (it's a **power transform**, `p^(1/T)`
   renormalized), show that the tempting `probs / T` version is a *literal no-op*, and
   read off greedy as the `T → 0` limit.
3. **Top-k** — keep the `k` highest logits, mask the rest to `-inf`. Kills the tail, but
   with a **fixed** cutoff that ignores how confident the model is at this step.
4. **Top-p / nucleus** — keep the smallest set of tokens whose cumulative probability
   reaches `p`. The cutoff **adapts** to the model's confidence — we plot it moving from
   1 to ~20 characters across a single sentence while top-k's line sits flat.

Then the payoff both ways round. The same knobs get measured in **two regimes on the
same model**: open-ended generation (where a real-word rate and a repetition rate say
temperature+top-p wins) and exact-answer prediction (where next-char accuracy says
**greedy** wins and sampling is strictly self-harm). Which knob is right depends
entirely on which of those two jobs you're doing — that's the transferable lesson, and
it's the same fork that separates "write me a story" from "extract this field as JSON".
"""

# %%
from pathlib import Path

import math
import re

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


def _repo_root() -> Path:
    """Walk up from the cwd to the folder holding pyproject.toml, so paths work no
    matter where the kernel is launched from."""
    here = Path.cwd()
    for d in (here, *here.parents):
        if (d / "pyproject.toml").exists():
            return d
    return here


ROOT = _repo_root()
DATA = ROOT / "nb" / "llm" / "data" / "input.txt"  # tinyshakespeare, shared with 01–07
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
print(f"device: {DEV}")

# %% [markdown]
"""
## Same data skeleton as `03`–`07`

Unchanged: character tokenizer, one long id tensor, 90/10 split. The only piece `08`
leans on heavily is `decode` — this whole notebook is about what comes out of it.
"""

# %%
text = DATA.read_text()
chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: "".join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

BLOCK = 64
BATCH = 32


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK, (BATCH,))
    x = torch.stack([d[i:i + BLOCK] for i in ix])
    y = torch.stack([d[i + 1:i + 1 + BLOCK] for i in ix])
    return x.to(DEV), y.to(DEV)


print(f"corpus {len(text):,} chars, vocab {vocab_size}, block {BLOCK}")

# %% [markdown]
"""
## The model, trained once with `07`'s recipe

`06`'s architecture driven by `07`'s loop — GPT-2 init, AdamW with the two-group decay
split, warmup+cosine, grad clip. Read it as settled: `08` never touches a weight after
this cell. Note what `generate` is **missing** — the model class here has a `forward`
and nothing else. Decoding is not part of the model; we build it, piece by piece, below.
"""


# %%
class CausalSelfAttention(nn.Module):
    """All heads at once: qkv projection, reshape to (B, nh, T, hs), masked softmax."""

    def __init__(self, n_embd, n_head, block_size=BLOCK):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.n_embd = n_head, n_embd
        self.head_size = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.proj.RESIDUAL_PROJ = True
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        nh, hs = self.n_head, self.head_size
        q, k, v = self.qkv(x).split(self.n_embd, dim=-1)
        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) * hs ** -0.5
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        out = F.softmax(scores, dim=-1) @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    """Position-wise MLP: expand 4x, GELU, project back into the residual stream."""

    def __init__(self, n_embd):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.proj.RESIDUAL_PROJ = True

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    """One transformer layer: pre-norm residual attn, then pre-norm residual MLP."""

    def __init__(self, n_embd, n_head, block_size=BLOCK):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    """`06`'s architecture. NOTE: forward only — no `generate`. That's this notebook."""

    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4, block_size=BLOCK):
        super().__init__()
        self.block_size, self.n_layer = block_size, n_layer
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss


# %%
def apply_gpt2_init(model):
    """07's Knob 1: N(0, 0.02), residual projections scaled by 1/sqrt(2*n_layer)."""
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            std = 0.02 * ((2 * model.n_layer) ** -0.5
                          if getattr(mod, "RESIDUAL_PROJ", False) else 1.0)
            nn.init.normal_(mod.weight, std=std)
            if mod.bias is not None:
                nn.init.zeros_(mod.bias)
        elif isinstance(mod, nn.Embedding):
            nn.init.normal_(mod.weight, std=0.02)
    return model


def get_lr(step, *, warmup, max_steps, max_lr, min_lr):
    """07's Knob 3: linear warmup to max_lr, then half-cosine decay to min_lr."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    ratio = min((step - warmup) / (max_steps - warmup), 1.0)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (max_lr - min_lr)


@torch.no_grad()
def full_val_loss(m, bs=256):
    """The same deterministic whole-val-split metric `03`–`07` reported."""
    m.eval()
    nwin = (len(val_data) - 1) // BLOCK
    x = val_data[:nwin * BLOCK].view(nwin, BLOCK)
    y = val_data[1:nwin * BLOCK + 1].view(nwin, BLOCK)
    total = count = 0
    for i in range(0, nwin, bs):
        xb, yb = x[i:i + bs].to(DEV), y[i:i + bs].to(DEV)
        total += m(xb, yb)[1].item() * yb.numel()
        count += yb.numel()
    m.train()
    return total / count


def train(m, max_iters=3000, max_lr=3e-3, min_lr=3e-4, warmup=100):
    """07's four knobs, condensed — this is `07`'s 'all on' run, nothing new."""
    apply_gpt2_init(m)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": [p for p in params if p.dim() >= 2], "weight_decay": 0.1},
         {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}],
        lr=max_lr, betas=(0.9, 0.95),
    )
    for it in range(max_iters + 1):
        for g in opt.param_groups:
            g["lr"] = get_lr(it, warmup=warmup, max_steps=max_iters,
                             max_lr=max_lr, min_lr=min_lr)
        if it % 1000 == 0:
            print(f"  step {it:>4} : val {full_val_loss(m):.3f}")
        xb, yb = get_batch("train")
        _, loss = m(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    return m


torch.manual_seed(1337)
model = train(GPT(vocab_size).to(DEV))
model.eval()  # decode-time from here on: no dropout, no grads, no updates
VAL = full_val_loss(model)
print(f"\ntrained. val loss {VAL:.3f} ({VAL / math.log(2):.2f} bits/char) — frozen.")

# %% [markdown]
"""
## One decode step: a distribution over 65 characters

Strip generation down to its atom. Feed a context, take the logits at the **last**
position (the only ones that predict something new — the earlier positions predict
characters we already have), and softmax them. That's the model's entire opinion about
what comes next: 65 numbers that sum to 1.

Two contexts, deliberately chosen to sit at opposite ends of the confidence range — and
the gap between them is what the rest of the notebook is about:
"""


# %%
@torch.no_grad()
def next_token_logits(m, idx):
    """The (B, vocab) logits for the position right after idx — one decode step's input.
    Crop to block_size so the position embedding never runs off its table."""
    return m(idx[:, -m.block_size:])[0][:, -1, :]


def ctx(s):
    """Encode a prompt string to a (1, T) context tensor on the device."""
    return torch.tensor([encode(s)], device=DEV)


def show_dist(prompt, top=8):
    """Print the model's next-char distribution for a prompt: top entries + the tail."""
    probs = F.softmax(next_token_logits(model, ctx(prompt)), dim=-1)[0]
    p, ix = probs.sort(descending=True)
    ent = -(probs * probs.clamp_min(1e-12).log2()).sum().item()
    print(f"context {prompt!r}")
    body = "  ".join(f"{itos[i.item()]!r}:{v:.3f}" for v, i in zip(p[:top], ix[:top]))
    print(f"  top {top}: {body}")
    print(f"  tail (the other {vocab_size - top}): {p[top:].sum():.4f} of the mass"
          f"   |   entropy {ent:.2f} bits = ~{2 ** ent:.1f} chars' worth of choice")
    return probs


CONFIDENT = "\nROMEO"          # after 'ROMEO' a ':' is all but certain
UNCERTAIN = "\nROMEO:\nI am "  # a fresh word could start with almost any letter
p_conf = show_dist(CONFIDENT)
print()
p_unc = show_dist(UNCERTAIN)

# %% [markdown]
"""
Two facts to carry forward.

**The model's confidence swings enormously between steps.** At `'\\nROMEO'` it is
effectively choosing between ~1 character — it *knows* the colon comes next, at 99.7%.
Nine characters later, at the start of a new word, it's choosing between ~19 of the 65.
That's not a small difference in degree; it's the difference between a step with no
decision in it and a step that's almost maximally open. Any decode rule worth using has
to cope with both, and this is precisely where fixed-cutoff top-k will struggle and
adaptive top-p will not.

**There is always a tail, and `01` sampled from it.** Define the tail honestly, though —
not as "the characters outside the top 8" (at the uncertain step those are *plausible*
letters, not junk) but as **the characters the model itself considers implausible right
now**: say, everything it gives under 1% to. Individually each is a non-candidate. But
there are dozens of them, and their **sum** is what you actually draw from:
"""

# %%
JUNK = 0.01  # a char the model gives under 1% to is no serious candidate right now
for name, probs in [("confident", p_conf), ("uncertain", p_unc)]:
    junk = probs[probs < JUNK]
    print(f"{name:>9} step: {len(junk):>2} of {vocab_size} chars are each under "
          f"{JUNK:.0%}, but together they hold {junk.sum():.4f} of the mass"
          f"  -> ~{junk.sum() * 500:.0f} per 500 draws")

# %% [markdown]
"""
Small is not zero, and you draw 500 times to write a paragraph. Every one of those draws
is a character the model just told you it didn't believe in — and worse, once emitted it
becomes *context*, so the model must condition on its own typo and write around it.

Notice too that the junk set is **not a fixed set of characters**. It's whatever is
implausible *at this step*, and it changes size completely between our two contexts.
That already tells you the cutoff has to move with the model's confidence — the entire
argument between top-k and top-p, visible before we've implemented either.

And this gets much sharper at scale: our vocab is 65, but a real 50k-vocab LM has
~49,990 tail tokens whose individual probabilities round to nothing while their sum is a
few percent — a few percent of *garbage*, sampled several times per paragraph.
Truncation exists to make that tail unreachable rather than merely unlikely.
"""

# %% [markdown]
"""
## Knob 1 — greedy: always take the argmax (and why it fails)

Start with the decode rule that sounds obviously correct. The model was trained to
maximize the probability of the true next character, so at decode time... just take the
character it thinks is most likely? No randomness, fully deterministic, reproducible.

    next_id = logits.argmax(dim=-1, keepdim=True)

Watch what it writes.
"""


# %%
@torch.no_grad()
def greedy_generate(m, idx, max_new_tokens):
    """Deterministic decoding: the argmax at every step."""
    for _ in range(max_new_tokens):
        nxt = next_token_logits(m, idx).argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


greedy_text = decode(greedy_generate(model, ctx("\n"), 400)[0].tolist())
print(greedy_text)

# %% [markdown]
"""
### The likelihood trap

It collapses into a **loop** — a short phrase repeated until we stop it. And it's not a
bug in the model: the model's loss is the best this notebook has produced, and the exact
same weights write perfectly good Shakespeare-shaped text a few cells from now. It's a
bug in the *decode rule*.

Here's the mechanism. Greedy is deterministic, so **state → next character is a
function**. If the last `block_size` characters ever repeat a window it has already
produced, the following character must be the same one it produced last time, and so
must the one after — the sequence has entered a cycle it can never leave. Sampling
breaks the cycle for free (any draw off the argmax is a different state); greedy has no
escape hatch.

The deeper reason is the one worth carrying to any model: **real text is not the
most-likely text.** A language model trained to put high probability on human text does
*not* imply human text is a chain of locally-maximal characters. Human writing
constantly takes low-probability turns — that's what makes it informative. A decode rule
that maximizes probability at every step therefore drifts into the flattest, most
generic, most repetitive region of the distribution, which is exactly where loops live.
(This is Holtzman et al., 2020, *The Curious Case of Neural Text Degeneration* — the
paper that introduced top-p, motivated by precisely this figure.)

Let's measure it rather than assert it. **Surprisal** of a character is `-log2 p(char |
context)` under the model — how many bits of "I didn't see that coming" the model spends
on it. Take three 320-character passages — *real* held-out Shakespeare, the greedy
sample, and a plain `T=1` sample — and plot the model's per-character surprisal along
each:
"""


# %%
@torch.no_grad()
def surprisals(ids, m, block=BLOCK):
    """Per-char surprisal -log2 p(id_t | the previous `block` chars), for t >= block.
    One batched forward: every predicted char gets a full block-length context."""
    ids = torch.as_tensor(ids)
    contexts = torch.stack([ids[i - block:i] for i in range(block, len(ids))]).to(DEV)
    targets = ids[block:].to(DEV)
    logp = F.log_softmax(m(contexts)[0][:, -1, :].float(), dim=-1)
    return (-logp[torch.arange(len(targets)), targets] / math.log(2)).cpu()


N = 320
real_ids = val_data[5000:5000 + N].tolist()
greedy_ids = greedy_generate(model, ctx("\n"), N)[0].tolist()[:N]

torch.manual_seed(0)


@torch.no_grad()
def plain_sample(m, idx, max_new_tokens):
    """01's decode rule: softmax the raw logits, draw, repeat. Tail and all."""
    for _ in range(max_new_tokens):
        probs = F.softmax(next_token_logits(m, idx), dim=-1)
        idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
    return idx


sampled_ids = plain_sample(model, ctx("\n"), N)[0].tolist()[:N]

series = [
    ("real Shakespeare (held-out)", real_ids, "tab:green"),
    ("greedy sample", greedy_ids, "tab:red"),
    ("plain T=1 sample", sampled_ids, "tab:blue"),
]
plt.figure(figsize=(7, 3.4))
for label, ids, color in series:
    s = surprisals(ids, model)
    plt.plot(s, lw=0.9, color=color, label=f"{label} (mean {s.mean():.2f} bits)")
plt.xlabel("character position")
plt.ylabel("surprisal (bits)")
plt.legend(fontsize=8)
plt.title("what the model 'expects' along each passage")
plt.tight_layout()
plt.show()

# %% [markdown]
"""
Three lines, three means, and each number has an exact meaning worth naming.

**Green — real held-out Shakespeare, ~2.2 bits.** This is what real writing looks like
to the model, and it's no coincidence that it lands on the val loss in bits: the val
loss *is* the mean of this line over the whole split. It's the **cross-entropy** between
real text and the model. Note the shape as much as the mean — it **fluctuates hard**,
spiking past 6 bits wherever a human made a genuinely unpredictable choice.

**Blue — the model's own `T=1` samples, ~1.8 bits.** Lower than real text, and not by
accident. Sampling from your own distribution and scoring the surprisal of what you drew
gives you the model's average **entropy**, whereas real text scores the
**cross-entropy**. Those differ by exactly the KL divergence — how far the model still
is from the truth:

```-
real text  -> H(real, model) = H(real) + KL(real || model)   = the val loss  (~2.2 bits)
own sample -> H(model)                                       = entropy       (~1.8 bits)
```

So a model always finds its own writing less surprising than the real thing, and **the
gap is its error**. A perfect model would close it.

**Red — greedy, ~1.2 bits.** The lowest of the three, by construction: greedy picks the
argmax at every step, so it walks the *minimum-surprisal path* through the distribution.
And look past the mean to the spread, where the table below is blunt: **0.0%** of
greedy's characters cost more than 4 bits — not a few, *none* — against ~20% of real
Shakespeare's. Greedy never once takes a turn it didn't expect. That's the trap. It
isn't producing good text; it's producing **maximally unsurprising** text, and those are
different targets. Real writing lives a bit above even the model's own entropy and it
*spikes*; greedy sits a full bit below and flattens. Straight down that valley are the
loops.

(Greedy's ~1.2 isn't ~0, and that's worth understanding: the argmax is only
*unsurprising* where the model is confident. Picking the top character out of a
genuinely open step still costs a couple of bits. Greedy minimizes surprisal step by
step — it doesn't get to zero it.)

So the target isn't *minimum* surprisal. It's the surprisal profile of real text: the
green band, spikes and all. Every knob below is a way of steering toward it — pull down
the junk at the top of the range without collapsing onto the red line at the bottom.
"""

# %%
print(f"{'passage':<28} | {'mean':>5} | {'std':>5} | "
      f"{'frac > 4 bits (real surprises)':>30}")
print("-" * 78)
for label, ids, _c in series:
    s = surprisals(ids, model)
    print(f"{label:<28} | {s.mean():>5.2f} | {s.std():>5.2f} | "
          f"{(s > 4).float().mean():>29.1%}")
print(f"\nval loss in bits = {VAL / math.log(2):.2f}"
      f"  <- what real text scores, by definition")
print("greedy: lowest mean AND the fewest surprises — it never takes an unlikely turn.")

# %% [markdown]
"""
## Knob 2 — temperature: one number that sharpens or flattens

Divide the logits by `T` **before** the softmax:

    probs = softmax(logits / T)

`T = 1` leaves the distribution exactly as trained; `T < 1` sharpens it toward the top
pick; `T > 1` flattens it toward uniform. That's the usual description, and it's true,
but it's worth deriving what the operation actually *is* — because the derivation hands
you both the `T → 0` limit and the reason the obvious alternative implementation is
nonsense.

### It's a power transform on the probabilities

Start from `p_i = exp(z_i) / Z` where `Z = sum_j exp(z_j)`, and ask what
`softmax(z/T)` is in terms of `p`:

```-
exp(z_i / T) = (exp z_i)^(1/T)            # a^(b/T) = (a^b)^(1/T)
             = (p_i * Z)^(1/T)            # since exp(z_i) = p_i * Z
             = Z^(1/T) * p_i^(1/T)        # Z^(1/T) is the SAME for every i

                       Z^(1/T) * p_i^(1/T)          p_i^(1/T)
softmax(z/T)_i = --------------------------- = -------------------
                  sum_j Z^(1/T) * p_j^(1/T)     sum_j p_j^(1/T)
```

The `Z^(1/T)` cancels top and bottom, and temperature is revealed as: **raise every
probability to the power `1/T`, then renormalize.** Everything falls out of that one
line:

- **`T < 1` sharpens.** The exponent `1/T > 1`, and raising numbers in `(0,1)` to a
  power above 1 shrinks them all — but shrinks the *small* ones proportionally much
  harder (`0.5^2 = 0.25`, half as big; `0.01^2 = 0.0001`, a hundredth as big).
  Renormalize and the mass has moved to the leaders.
- **`T > 1` flattens.** The exponent `1/T < 1` is a root, which pulls everything toward
  1 and compresses the gaps.
- **`T → 0` is greedy.** Look at the ratio of any token to the top one: `(p_i /
  p_max)^(1/T)`. The base is `< 1` for every non-argmax token, and the exponent `→ ∞`,
  so every ratio `→ 0`: the argmax takes all the mass. Greedy isn't a separate algorithm
  bolted on — it is the `T → 0` limit of this knob. (In *code* we still special- case
  it, because `logits / 0` is `inf` and `softmax(inf)` is `nan`.)
- **`T → ∞` is uniform.** The exponent `→ 0` and every `p_i^0 = 1`: all 65 characters
  equally likely, i.e. the tokenizer's noise floor.

### Why not `probs / T`?

Because it does *nothing at all*. Dividing every probability by the same constant and
renormalizing is the identity — the constant cancels, exactly like `Z^(1/T)` did above.
It's a tempting one-liner precisely because it *looks* like it should sharpen something,
and it silently doesn't. Both claims, measured:
"""

# %%
p = F.softmax(next_token_logits(model, ctx(UNCERTAIN)), dim=-1)[0]

# the bug: divide the PROBS and renormalize -> mathematically the identity
naive = p / 0.5
naive = naive / naive.sum()
print(f"probs/T renormalized == probs untouched? {torch.allclose(naive, p)}"
      f"   (max diff {(naive - p).abs().max():.2e})  <- a no-op, not a knob")

# the real thing: divide the LOGITS -> equals the p^(1/T) power transform
T = 0.5
correct = F.softmax(next_token_logits(model, ctx(UNCERTAIN)) / T, dim=-1)[0]
power = p ** (1 / T)
power = power / power.sum()
print(f"softmax(logits/T) == p^(1/T) renormalized? "
      f"{torch.allclose(correct, power, atol=1e-6)}"
      f"   (max diff {(correct - power).abs().max():.2e})  <- the derivation, verified")

top = p.argmax().item()
print(f"\ntop char {itos[top]!r}: p={p[top]:.3f} at T=1  ->  {correct[top]:.3f} at T=.5"
      f"  (sharpened)   ->  {naive[top]:.3f} via the buggy probs/T (unchanged)")

# %% [markdown]
"""
### The whole range in one picture

Sweep `T` and read the **entropy** of the resulting distribution, in bits, at both of
our contexts. Entropy is the honest summary of "how much choice is left": `2^entropy` is
the *effective* number of characters the model is choosing between. The two curves start
far apart at `T = 1` (that's the confidence gap from earlier) and both get squeezed to
the same two limits — 0 bits (greedy, 1 effective char) as `T → 0`, and `log2(65) =
6.02` bits (uniform, all 65) as `T` grows.
"""

# %%
Ts = torch.logspace(math.log10(0.05), math.log10(20), 60)


def entropy_at(logits, T):
    q = F.softmax(logits / T, dim=-1)
    return -(q * q.clamp_min(1e-12).log2()).sum().item()


lg_conf = next_token_logits(model, ctx(CONFIDENT))
lg_unc = next_token_logits(model, ctx(UNCERTAIN))

plt.figure(figsize=(6.5, 3.2))
plt.plot(Ts, [entropy_at(lg_unc, t) for t in Ts], label=f"uncertain ctx {UNCERTAIN!r}")
plt.plot(Ts, [entropy_at(lg_conf, t) for t in Ts], label=f"confident ctx {CONFIDENT!r}")
plt.axhline(math.log2(vocab_size), ls="--", c="gray", lw=0.8)
plt.text(0.06, math.log2(vocab_size) - 0.45, "uniform = log2(65)", fontsize=8,
         color="gray")
plt.axvline(1.0, ls=":", c="k", lw=0.8)
plt.text(1.05, 0.2, "T=1: as trained", fontsize=8)
plt.xscale("log")
plt.xlabel("temperature T")
plt.ylabel("entropy of next-char distribution (bits)")
plt.legend(fontsize=8)
plt.title("T→0: all mass on the argmax   |   T→∞: uniform noise")
plt.tight_layout()
plt.show()

# %% [markdown]
"""
### Read four temperatures

Same model, same seed, same 300 characters. Only `T` changes.
"""


# %%
@torch.no_grad()
def temp_generate(m, idx, max_new_tokens, T):
    for _ in range(max_new_tokens):
        logits = next_token_logits(m, idx)
        if T == 0:                                   # the T->0 limit, done exactly
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(logits / T, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


for T in [0.25, 0.8, 1.0, 1.6]:
    torch.manual_seed(0)
    out = decode(temp_generate(model, ctx("\nROMEO:\n"), 300, T)[0].tolist())
    print(f"{'=' * 78}\nT = {T}\n{'=' * 78}\n{out}\n")

# %% [markdown]
"""
Reading them in order: **T=0.25** is nearly greedy — clean, common words, but flat and
already leaning on repetition. **T=0.8** is the sweet spot most people ship: words hold
together, the play's structure survives, there's still variety. **T=1.0** (the raw
trained distribution) is looser, with more invented words creeping in from the tail.
**T=1.6** is falling apart — flattening the distribution promotes exactly the junk the
tail is made of.

And the `T → 0` limit isn't just a story — it's a testable identity:
"""

# %%
torch.manual_seed(0)
a = temp_generate(model, ctx("\nROMEO:\n"), 60, T=0)[0].tolist()
b = greedy_generate(model, ctx("\nROMEO:\n"), 60)[0].tolist()
print(f"temperature T=0 path == greedy argmax path? {a == b}   <- same rule, two names")

# %% [markdown]
"""
## Knob 3 — top-k: keep the k best, delete the rest

Temperature reshapes the distribution but never *removes* anything: at `T=1.6` the junk
gets more likely, and even at `T=0.25` every one of the 65 characters keeps a nonzero
probability. It's a rescaling, not a filter. Truncation is the other axis.

**Top-k** is the blunt version: keep the `k` largest logits, set every other logit to
`-inf`, and let softmax do the rest — `exp(-inf) = 0`, so those characters become
*exactly* unreachable rather than merely unlikely. That distinction is the point: over
500 draws, "unlikely" happens.
"""


# %%
def apply_top_k(logits, k):
    """Mask every logit below the k-th largest to -inf (softmax then gives it a 0)."""
    k = min(k, logits.size(-1))                  # never ask for more than the vocab
    kth = torch.topk(logits, k, dim=-1).values[:, [-1]]   # (B,1): k-th largest logit
    return logits.masked_fill(logits < kth, float("-inf"))


lg = next_token_logits(model, ctx(UNCERTAIN))
for k in [1, 5, 40]:
    q = F.softmax(apply_top_k(lg, k), dim=-1)[0]
    alive = (q > 0).sum().item()
    print(f"top_k={k:>2}: {alive:>2} chars survive, they hold "
          f"{F.softmax(lg, dim=-1)[0][q > 0].sum():.4f} of the ORIGINAL mass")

# %% [markdown]
"""
### The fixed-cutoff problem

`k` is a constant, but the model's confidence is not — we measured that swing in the
very first experiment, and here's the bill for ignoring it. The same `k=40` that behaves
at an uncertain step is absurd at a confident one, and a `k` small enough to be safe at
the confident step would gag the model at the uncertain one. Put both contexts side by
side:
"""

# %%
print(f"{'k':>4} | {'confident ctx: mass kept':>26} | {'uncertain ctx: mass kept':>26}")
print("-" * 62)
for k in [1, 3, 10, 40, 65]:
    row = []
    for lgts in (lg_conf, lg_unc):
        probs = F.softmax(lgts, dim=-1)[0]
        keep = F.softmax(apply_top_k(lgts, k), dim=-1)[0] > 0
        row.append(probs[keep].sum().item())
    print(f"{k:>4} | {row[0]:>26.6f} | {row[1]:>26.6f}")
print("\nAt the confident step k=3 already keeps ~everything — k=40 is 37 chars of")
print("pure tail, exactly what truncation was for. At the uncertain step, a k small")
print("enough to be safe there would be cutting real, plausible continuations.")

# %% [markdown]
"""
One `k` cannot be right for both, because `k` answers the wrong question. It asks "how
many characters?" when what you care about is "how much probability?". That's the gap
top-p closes.

(Honest scale caveat: with a 65-char vocab this is a mild annoyance. At a real vocab of
50k it's the whole ballgame — `k=40` at a confident step admits a tail of ~49,960
tokens' worth of junk, and the reason nobody ships top-k alone.)
"""

# %% [markdown]
"""
## Knob 4 — top-p / nucleus: keep the smallest set that covers p

Ask the question the right way round. Sort the characters by probability and walk down
the list adding them up until the running total **reaches** `p` (say 0.9). That set —
the **nucleus** — is what you keep; everything else is zeroed and the survivors are
renormalized back to a valid distribution.

The cutoff is now **adaptive by construction**. When the model is sure, one character
already covers 0.9 and the nucleus is a single token. When it's torn, it takes a dozen
characters to reach 0.9 and the nucleus opens up to hold them. Same `p`, different width
at every step — which is precisely what top-k couldn't do.
"""


# %%
def nucleus_mask(probs, p):
    """True where a token is INSIDE the nucleus: the smallest set of tokens whose
    cumulative probability reaches p."""
    p = p + 1e-6                                  # float slop — see the note below
    sorted_probs, sorted_idx = probs.sort(descending=True, dim=-1)
    prefix = sorted_probs.cumsum(dim=-1) - sorted_probs   # mass strictly BEFORE each
    drop_sorted = prefix > p                  # drop once the mass before it reached p
    # scatter the sorted-space decision back into vocab order
    drop = torch.zeros_like(drop_sorted).scatter(-1, sorted_idx, drop_sorted)
    return ~drop


def apply_top_p(probs, p):
    """Zero everything outside the nucleus, then renormalize to sum to 1 again."""
    probs = probs.masked_fill(~nucleus_mask(probs, p), 0.0)
    return probs / probs.sum(dim=-1, keepdim=True)


demo = torch.tensor([[0.50, 0.30, 0.15, 0.05]])
print(f"probs    {[round(x, 2) for x in demo.tolist()[0]]}"
      f"   (cumulative: [0.50, 0.80, 0.95, 1.00])")
print(f"p=0.9 -> keep {nucleus_mask(demo, 0.9).tolist()[0]}"
      f"   the 0.15 CROSSES 0.9 and is kept; 0.05 is outside")
print("      -> renormalized "
      f"{[round(x, 3) for x in apply_top_p(demo, 0.9).tolist()[0]]}")

# %% [markdown]
"""
### Two subtleties worth the ink

**(1) The prefix shift.** The mask is `(cumsum - sorted_probs) > p`, not `cumsum > p`.
The subtraction makes the comparison look at the mass *strictly before* each token
rather than including it, and that buys two properties:

- **the crossing token is kept.** Top-p is a **floor** on the mass you keep, not a
  ceiling: you add tokens until the total *reaches* `p`, so the token that pushes you
  over the line is inside the nucleus. With `[0.5, 0.3, 0.15, 0.05]` at `p=0.9`, plain
  `cumsum > p` would drop `0.15` (its cumsum is 0.95) and keep only 0.8 of the mass —
  *under* the p we asked for. The prefix version compares `0.15`'s *prefix* of 0.8,
  which is below 0.9, so it stays.
- **the top-1 token can never be removed.** Its prefix mass is exactly 0, and `0 > p` is
  False for any `p >= 0`. So the nucleus is never empty and `p → 0` degenerates
  gracefully to greedy — no special case needed.

**(2) The `+1e-6`.** Float arithmetic makes exact-boundary cases lie. With `[0.4, 0.3,
0.2, 0.1]` at `p=0.7`, the prefix `0.4 + 0.3` rounds to `0.70000005` in float32 while
`0.7` itself stores as `0.69999999` — so `prefix > p` is *True* and the `0.2` gets
dropped even though the mass before it did not really exceed `0.7`. A tolerance of
`1e-6` absorbs it. (The principled fix is a float64 cumsum, where both sides round to
the same double. In practice neither matters for real softmax outputs — exact ties are
measure-zero — but it's a fun demonstration of a boundary bug that only fires on
hand-made numbers.)
"""

# %%
tie = torch.tensor([[0.4, 0.3, 0.2, 0.1]])                # prefix of 0.2 is exactly 0.7
sp, _ = tie.sort(descending=True)
print(f"float32: 0.4 + 0.3 = {(sp[0, 0] + sp[0, 1]).item():.8f}   vs p = "
      f"{torch.tensor(0.7).item():.8f}  -> the sum 'exceeds' p")


def nucleus_mask_noslop(probs, p):
    sorted_probs, sorted_idx = probs.sort(descending=True, dim=-1)
    drop_sorted = (sorted_probs.cumsum(dim=-1) - sorted_probs) > p
    return ~torch.zeros_like(drop_sorted).scatter(-1, sorted_idx, drop_sorted)


print(f"p=0.7 without the slop: keep {nucleus_mask_noslop(tie, 0.7).tolist()[0]}"
      f"   <- drops 0.2, keeping only 0.7 of the mass")
print(f"p=0.7 with    the slop: keep {nucleus_mask(tie, 0.7).tolist()[0]}"
      f"   <- correct: cover p, don't stop under it")

# the two contract properties, on the real model's distribution
probs_unc = F.softmax(lg_unc, dim=-1)
for pp in [0.0, 0.5, 0.9, 1.0]:
    m = nucleus_mask(probs_unc, pp)
    kept = probs_unc[m].sum().item()
    assert m[0, probs_unc.argmax()], "top-1 must never leave the nucleus"
    assert kept >= pp - 1e-5, "the nucleus must COVER p, not stop under it"
    print(f"p={pp:<4}: nucleus = {m.sum().item():>2} chars, holding {kept:.4f} of mass")

# %% [markdown]
"""
### The money plot: watch the cutoff breathe

This is the whole case for top-p in one figure. Take a single generated passage, and at
every decode step record how many characters the `p=0.9` nucleus contains. Top-k's
answer is a horizontal line by definition. Top-p's is not:
"""

# %%
torch.manual_seed(0)
probe = temp_generate(model, ctx("\nROMEO:\nWhat is the matter"), 120, T=1.0)[0]
sizes, marks = [], []
for i in range(len(ctx("\nROMEO:\nWhat is the matter")[0]), len(probe)):
    q = F.softmax(next_token_logits(model, probe[None, :i]), dim=-1)
    sizes.append(nucleus_mask(q, 0.9).sum().item())
    marks.append(itos[probe[i].item()])

plt.figure(figsize=(7.5, 3.2))
plt.plot(sizes, lw=1.0, color="tab:purple", label="top-p = 0.9: nucleus size")
plt.axhline(40, ls="--", c="tab:orange", lw=1.0, label="top-k = 40: always 40")
plt.axhline(1, ls=":", c="gray", lw=0.8)
plt.xlabel("decode step (of the generated passage)")
plt.ylabel("characters kept")
plt.legend(fontsize=8)
plt.title("the nucleus adapts to confidence; k cannot")
plt.tight_layout()
plt.show()

wide = max(range(len(sizes)), key=lambda i: sizes[i])
narrow = min(range(len(sizes)), key=lambda i: sizes[i])
print(f"nucleus size ranges {min(sizes)}..{max(sizes)} chars over {len(sizes)} steps "
      f"(mean {sum(sizes) / len(sizes):.1f})")
back = lambda i: "".join(marks[max(0, i - 18):i])
print(f"  narrowest, {sizes[narrow]} char(s), after ...{back(narrow)!r}"
      f"  -> the model is sure")
print(f"  widest,   {sizes[wide]:>2} char(s), after ...{back(wide)!r}"
      f"  -> genuinely open")
print(f"\ntop-k=40 would keep 40 at BOTH — {40 - sizes[narrow]} chars of pure tail at")
print("the narrow step, while top-p spends its budget only where there's real choice.")

# %% [markdown]
"""
## Assemble the decode step

All four knobs in one function, in the order they have to run:

```-
logits ──► /T ──► top-k ──► softmax ──► top-p ──► multinomial ──► id
           │      │                     │
           │      │                     └─ thresholds cumulative PROBABILITY,
           │      │                        so it must come after the softmax
           │      └─ thresholds LOGITS, so it comes before (and -inf ──► exactly 0)
           └─ must precede the softmax: softmax(z/T), never probs/T (a no-op)
```

Two order facts are forced, not stylistic. **Temperature before softmax** — we derived
why: after the softmax, dividing is the identity. **Top-k before, top-p after** — top-k
thresholds against a logit value, top-p against accumulated probability mass, which only
exists once you've normalized. And they compose sensibly: top-k trims the tail, then
temperature reshapes what's left, then top-p takes the adaptive slice of *that*.
"""


# %%
def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    """One decode step: (B, vocab) logits -> (B, 1) sampled ids."""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)      # the T->0 limit, exactly
    logits = logits / temperature                       # sharpen/flatten (pre-softmax!)
    if top_k is not None:
        logits = apply_top_k(logits, top_k)             # hard cutoff on logits
    probs = F.softmax(logits, dim=-1)
    if top_p is not None:
        probs = apply_top_p(probs, top_p)               # adaptive cutoff on prob mass
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(m, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
    """01's generate, with the sampling opened up. Still just: predict, pick, append."""
    for _ in range(max_new_tokens):
        nxt = sample_next(next_token_logits(m, idx), temperature, top_k, top_p)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


# every degenerate case collapses onto greedy — one rule, four spellings
seed = ctx("\nROMEO:\n")
ref = greedy_generate(model, seed, 40)[0].tolist()
for label, kw in [("temperature=0", dict(temperature=0)),
                  ("top_k=1", dict(top_k=1)),
                  ("top_p→0", dict(top_p=0.0)),
                  ("temperature=0.01", dict(temperature=0.01))]:
    torch.manual_seed(0)
    got = generate(model, seed, 40, **kw)[0].tolist()
    print(f"  {label:<18} == greedy? {got == ref}")

# %% [markdown]
"""
## The payoff: the same knobs judged in two regimes

Now the part that matters, and the reason "what's the best temperature?" has no answer.
We'll score the *same frozen model* under the *same settings* on two different jobs.

**Regime 1 — open-ended generation.** There's no right answer; we want text that reads
well. Two crude but honest proxies for a char model, both computable from the corpus:

- **real-word rate**: of the whitespace-separated words it writes, what fraction
  actually appear somewhere in tinyshakespeare? This punishes the tail — junk characters
  make junk words.
- **distinct-word rate**: unique words ÷ total words. This punishes the loop — greedy
  will score beautifully on the first metric and catastrophically here.

You need *both*. Either one alone has a trivial degenerate winner, which is exactly the
tension the knobs are trading against.
"""


# %%
CORPUS_WORDS = set(re.findall(r"[a-z']+", text.lower()))


def score_open_ended(sample):
    """Real-word rate (punishes tail junk) and distinct-word rate (punishes loops)."""
    words = re.findall(r"[a-z']+", sample.lower())
    if not words:
        return 0.0, 0.0
    real = sum(w in CORPUS_WORDS for w in words) / len(words)
    distinct = len(set(words)) / len(words)
    return real, distinct


SETTINGS = [
    ("greedy (T=0)", dict(temperature=0)),
    ("T=0.25", dict(temperature=0.25)),
    ("T=0.8", dict(temperature=0.8)),
    ("T=1.0 (raw, = 01)", dict(temperature=1.0)),
    ("T=1.6", dict(temperature=1.6)),
    ("T=1.0, top_k=10", dict(temperature=1.0, top_k=10)),
    ("T=1.0, top_p=0.9", dict(temperature=1.0, top_p=0.9)),
    ("T=0.8, top_p=0.9", dict(temperature=0.8, top_p=0.9)),
    ("T=0.8, top_k=40, top_p=0.95", dict(temperature=0.8, top_k=40, top_p=0.95)),
]

print(f"{'setting':<30} | {'real-word':>9} | {'distinct':>9}   (3000 chars each)")
print("-" * 66)
open_scores = {}
for label, kw in SETTINGS:
    torch.manual_seed(0)
    s = decode(generate(model, ctx("\n"), 3000, **kw)[0].tolist())
    real, distinct = score_open_ended(s)
    open_scores[label] = (real, distinct)
    print(f"{label:<30} | {real:>8.1%} | {distinct:>8.1%}")

# %%
plt.figure(figsize=(6.2, 4.2))
for label, (real, distinct) in open_scores.items():
    plt.scatter(distinct, real, s=32)
    plt.annotate(label, (distinct, real), fontsize=7,
                 xytext=(4, 3), textcoords="offset points")
plt.xlabel("distinct-word rate  →  less repetition")
plt.ylabel("real-word rate  →  less junk")
plt.title("open-ended: the two failure modes pull opposite ways")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# %% [markdown]
"""
The scatter says it plainly: the two failure modes sit at **opposite corners**, and the
knobs trade along the curve between them.

**Greedy** parks in the top-left: ~100% real words — it only ever writes the safest
continuation, so it literally cannot misspell — and ~2% distinct, because it's the same
handful of words on repeat. A perfect score on one axis, a catastrophic one on the
other, from the decode rule that sounded obviously correct. **T=1.6** is the mirror
image in the bottom-right: the most variety of anything in the table, roughly half of it
invented gibberish. Neither is usable, and neither metric alone would have told you
that.

The one clean, uncontested win in the table is the head-to-head we set up two sections
ago. At the same `T=1.0`, **top-p beats top-k on _both_ axes at once** — more real words
*and* more distinct ones. That's not a trade, that's a dominated setting: spending the
same truncation budget adaptively is simply better than spending it at a fixed width,
for exactly the reason the breathing-nucleus plot showed. It also edges out plain
`T=0.8`, which is the interesting part: the adaptive cutoff buys you the
tail-suppression you'd otherwise pay a lower temperature for, *without* paying that
temperature's cost in diversity.

Everything else is a genuine frontier — no point dominates, so there's no "best" setting
to look up, only a position to choose on the curve. Which is why `temperature=0.8,
top_p=0.9` is an industry *default* and not a *solution*.
"""

# %% [markdown]
"""
### Regime 2 — the exact-answer task

Now change the job, not the model. Sometimes there **is** a right answer: extract the
field, evaluate the expression, classify the sentiment, complete the identifier. The
char-model version of that job is the one it was literally trained on — *given real
held-out context, predict the character that actually comes next* — and it's scored by
accuracy, not by whether it reads nicely.

Same knobs, same model. Watch the ranking inverts completely:
"""


# %%
@torch.no_grad()
def next_char_accuracy(m, n_pos=4000, **kw):
    """Accuracy at predicting the ACTUAL next char of held-out text, per decode rule."""
    nwin = min(n_pos, (len(val_data) - 1) // BLOCK)
    x = val_data[:nwin * BLOCK].view(nwin, BLOCK).to(DEV)
    y = val_data[BLOCK:nwin * BLOCK + 1:BLOCK].to(DEV)   # true char after each window
    logits = m(x)[0][:, -1, :]                           # (nwin, vocab)
    pred = sample_next(logits, **kw)[:, 0]
    return (pred.cpu() == y.cpu()[:len(pred)]).float().mean().item()


acc = {}
print(f"{'setting':<30} | {'next-char accuracy':>18}")
print("-" * 52)
for label, kw in SETTINGS:
    torch.manual_seed(0)
    acc[label] = next_char_accuracy(model, **kw)
    print(f"{label:<30} | {acc[label]:>17.1%}")
print(f"\ngreedy is {acc['greedy (T=0)'] - acc['T=1.6']:.1%} ahead of T=1.6, and "
      f"{acc['greedy (T=0)'] - acc['T=1.0 (raw, = 01)']:.1%} ahead of 01's raw T=1.0.")

# %% [markdown]
"""
**Greedy wins, and it isn't close.** The ordering is essentially the reverse of the
open-ended table: every knob that bought diversity there is spending accuracy here, and
`T=1.6` — the champion of the distinct-word axis — is dead last, ~18 points behind
greedy.

The reason is worth stating exactly, because it generalizes far past char models. If you
**sample** from the model's own distribution, your accuracy is the *expected*
probability the model assigned to the true character, `E[p(true)]`. If you take the
**argmax**, your accuracy is how often the model's top pick *is* the true character,
`P(argmax = true)`. The second can never be smaller — you're always better off betting
your mode than drawing from your own distribution. And it's not a loose analogy; the
theory predicts the measured number:
"""


# %%
@torch.no_grad()
def expected_true_prob(m, n_pos=4000):
    """E[p(true char)] — what sampling at T=1 should score, if the theory is right."""
    nwin = min(n_pos, (len(val_data) - 1) // BLOCK)
    x = val_data[:nwin * BLOCK].view(nwin, BLOCK).to(DEV)
    y = val_data[BLOCK:nwin * BLOCK + 1:BLOCK].to(DEV)
    probs = F.softmax(m(x)[0][:, -1, :], dim=-1)
    return probs[torch.arange(len(y)), y].mean().item()


print(f"E[p(true)]  — theory says T=1 sampling scores this : "
      f"{expected_true_prob(model):.1%}")
print(f"measured accuracy of T=1.0 sampling                : "
      f"{acc['T=1.0 (raw, = 01)']:.1%}")
print(f"measured accuracy of greedy = P(argmax = true)     : {acc['greedy (T=0)']:.1%}"
      f"  <- strictly better, always")

# %% [markdown]
"""
**Sampling on a task with one right answer is voluntarily adding noise to your own
answer.**

That's the fork, and it's the whole lesson of the notebook:

- **Open-ended** (story, chat, brainstorm): you want the surprisal profile of real text,
  spikes and all — not the flat, never-surprised line greedy walks. Sample:
  `T ≈ 0.7–1.0` with `top_p ≈ 0.9`. Greedy's loop is a real failure here, and the
  diversity is a feature.
- **Exact-answer** (extraction, classification, arithmetic, code completion, a JSON
  field): there is a mode and you want it. `T = 0`. Every point of temperature is
  accuracy you're donating to the RNG.

Two production footnotes that follow from the same logic. (1) `T=0` is *not* the
most-likely **sequence** — it's a chain of locally-best characters, which is why it can
walk into a loop no globally-likely sequence would contain; searching for the best whole
sequence is *beam search*, a different algorithm that helps exact-answer tasks like
translation and (per Holtzman) makes open-ended text *worse*. (2) The one place sampling
beats greedy on an exact-answer task is **self-consistency**: sample `N` reasoning
chains at `T > 0` and majority-vote the final answers. The diversity is doing real work
there — exploring different derivations — while the vote still collapses to a single
answer at the end. It's the exception that confirms the rule, because the *output* is
still a mode; you just took it over samples instead of over tokens.
"""

# %% [markdown]
"""
## The final read

`01`'s sample was `T=1.0` with no truncation, straight off the raw distribution. Here's
the same model with the settings this notebook argues for — and, right after it, the
greedy loop again, so the gap between two decode rules over identical weights is on one
screen.
"""

# %%
torch.manual_seed(1337)
print("=" * 78)
print("T=0.8, top_p=0.9  — the settings the open-ended table likes")
print("=" * 78)
print(decode(generate(model, ctx("\n"), 500, temperature=0.8, top_p=0.9)[0].tolist()))

print("\n" + "=" * 78)
print("greedy — same weights, same seed, same prompt")
print("=" * 78)
print(decode(greedy_generate(model, ctx("\n"), 500)[0].tolist()))

# %% [markdown]
"""
## What you built, and where `09` goes next

`generate`'s sampling is fully open, and none of it required a gradient:

- **greedy** *(the instructive failure)* — the argmax at every step. Deterministic, so a
  repeated context is a repeated future: it enters a **loop** and can't leave. The
  surprisal plot showed *why* it's the wrong target: greedy walks the minimum-surprisal
  path (~1.2 bits, and it never takes an unlikely turn) while real text lives a full bit
  higher and *spikes*. The most-likely text is not the good text — and the same plot
  measured the model's own entropy (~1.8) sitting below the cross-entropy on real text
  (~2.2), the gap being exactly the model's remaining error.
- **temperature** — `softmax(z/T)`, which we derived is exactly `p^(1/T)` renormalized:
  `T<1` sharpens, `T>1` flattens, `T→0` *is* greedy and `T→∞` is uniform. The
  `probs / T` version is a measured no-op.
- **top-k** — mask all but the `k` best logits to `-inf`, making the tail exactly
  unreachable. A **fixed** cutoff, so it's simultaneously too wide at confident steps
  and too narrow at uncertain ones — measured on both.
- **top-p / nucleus** — the smallest set whose mass reaches `p`, via the prefix-shift
  mask (which keeps the crossing token and can never drop the top-1). The cutoff
  **breathes** with the model's confidence — 1 to ~20 characters across one sentence,
  where top-k's line is flat.
- **the fork** — same model, opposite winners: sampling with `top_p` for open-ended
  text, `T=0` for anything with a right answer.

That closes **Phase A**. The model is built (`04`–`06`), well-trained (`07`), and now
decoded properly (`08`) — `01`'s box is empty, every part of it explained and measured.

**Phase B** changes the parts themselves: what Llama/Mistral/Qwen actually run instead
of 2019's GPT-2, one delta per notebook, each a small measured upgrade on the exact
model you just built. `09` starts with the smallest of them — **RMSNorm**: drop
LayerNorm's re-centering and just scale by the root-mean-square. One fewer reduction,
one fewer parameter vector, no measurable quality cost — and the interesting question is
*why* the centering, which LayerNorm's own paper argued for, turned out not to matter.
"""
