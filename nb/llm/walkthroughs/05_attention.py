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
# LLM · 05 — the heart: self-attention

`04` ended on a single, loaded sentence: **averaging over the past is a matmul with a
lower-triangular matrix of uniform weights**, and the *only* thing wrong with it for
language is that the weights are fixed. Concat had focus but a rigid, parameter-hungry
window; average scaled to any length but couldn't focus and was blind to order.

This notebook builds the thing that fixes both at once. Keep `04`'s picture in your
head:

    out[t] = sum over s<=t of  W[t, s] * value[s]      # a causal, row-normalized mix

Attention keeps this *exact* shape — a causal, row-normalized weighted sum of past
values — but **learns `W` from the tokens themselves**, so each position can put its
mass on the few earlier tokens that actually matter. That one change is the whole
transformer.

The plan, all measured, nothing asserted:

1. Recover `04`'s frozen-weight average as a matmul (one cell), then make the weights
   **content-dependent** via query·key. Build a single `Head` from scratch.
2. Train it into the same LM loop as `03`/`04` → it clears the bigram floor and buries
   the uniform average (the *learned* weighting works, where the frozen one failed).
3. *Prove* the two `04` weaknesses are gone: it uses **order**, and it **focuses** —
   with an attention-weight heatmap you can read.
4. Single head → **multi-head** → the **fused** `CausalSelfAttention`. That fused module
   is exactly the box `01` ran as `nn.MultiheadAttention`; we check it against torch's
   own kernel to ~0.

Still no MLP, no residual/norm, no RoPE, no KV-cache — those are `06`, `11`, `12`. One
delta at a time: this notebook is *only* attention.
"""

# %%
from pathlib import Path

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
DATA = ROOT / "nb" / "llm" / "data" / "input.txt"  # tinyshakespeare, shared with 01–04
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)
print(f"device: {DEV}")

# %% [markdown]
"""
## Same data skeleton as `03`/`04`

Character tokenizer, one long id tensor, 90/10 split, random `BLOCK`-length chunks with
targets shifted one step left. Every model below trains through this exact `get_batch`,
so its loss is directly comparable to the bigram floor and to `04`'s concat/average.

One difference from `04`: there's no `causal_windows` gadget here. Attention runs over
the **whole** `BLOCK`-length row at once and handles "only look at the past" internally,
with a mask. That's the point — the window stops being a fixed hyperparameter you slice
by hand.
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

BLOCK = 64        # sequence length per row: every position may attend to earlier ones
BATCH = 32


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK, (BATCH,))
    x = torch.stack([d[i:i + BLOCK] for i in ix])          # (BATCH, BLOCK)
    y = torch.stack([d[i + 1:i + 1 + BLOCK] for i in ix])  # targets = x shifted +1
    return x.to(DEV), y.to(DEV)


print(f"corpus {len(text):,} chars, vocab {vocab_size}, block {BLOCK}")

# %% [markdown]
"""
## Where `04` left off: the average IS a matmul with frozen weights

One cell to reload the punchline. Build the lower-triangular uniform-weight matrix and
multiply: row `t` spreads `1/(t+1)` over positions `0..t` and `0` on the future. The
result is a causal running mean — computed with a single `W @ x`, no loop.

This `W` is the skeleton every attention layer keeps. Attention changes exactly one
thing about it: the numbers inside. Watch what stays the same as we go.
"""

# %%
torch.manual_seed(0)
T_demo, C_demo = 5, 2
x_demo = torch.randn(T_demo, C_demo)

w_uniform = torch.tril(torch.ones(T_demo, T_demo))
w_uniform = w_uniform / w_uniform.sum(dim=1, keepdim=True)   # each row sums to 1
avg = w_uniform @ x_demo                                     # causal running mean

print("frozen uniform causal weights W (row t = mean over 0..t):")
print(w_uniform.round(decimals=2))
print("\nlower-triangular (causal), rows sum to 1 — attention keeps BOTH of these.")

# %% [markdown]
"""
## The one idea: let each token *choose* its weights

The uniform mean forces `W[t, s] = 1/(t+1)` for every `s`. We want `W[t, s]` to be
**large when token `s` is relevant to token `t`, small otherwise** — and we want the
model to learn what "relevant" means. Attention scores relevance with a dot product
between two learned views of each token:

- **query** `q[t]` — "what is position `t` looking for?"
- **key**   `k[s]` — "what does position `s` offer?"
- **value** `v[s]` — "what does position `s` contribute if attended to?"

The raw weight of `s` for `t` is the match `q[t] · k[s]`. Turn the row of matches into a
causal, normalized weighting with two ops we've already met — **mask the future to
`-inf`, then softmax** — and mix the values:

    scores = q @ k.transpose        # (T, T): scores[t, s] = q[t] · k[s]
    scores = scale * scores         # keep variance ~1 (why: two cells down)
    scores = mask_future(scores)    # -inf above the diagonal -> causal
    W      = softmax(scores)        # rows sum to 1, mass on the past  <- the 04 shape
    out    = W @ v                  # weighted sum of past values

Same `W @ v` as the average — but `W` is now built from the tokens, per row, and
trained. q, k, v are just three linear projections of the input; nothing else is new.
"""


# %%
class Head(nn.Module):
    """One head of causal self-attention (built from scratch)."""

    def __init__(self, n_embd, head_size, block_size=BLOCK):
        super().__init__()
        self.head_size = head_size
        # three learned views of each token. bias=False is the GPT convention.
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # causal mask as a buffer: no grad, but moves with .to(device) and saves in
        # state_dict. 1 on/below the diagonal (allowed), 0 above (the future).
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x, return_weights=False):
        B, T, C = x.shape
        q = self.query(x)                                    # (B, T, hs)
        k = self.key(x)                                      # (B, T, hs)
        # scores[b, t, s] = q[b, t] · k[b, s], scaled so variance stays ~1
        scores = q @ k.transpose(-2, -1) * self.head_size ** -0.5   # (B, T, T)
        # a token may not read the future: set those scores to -inf so softmax -> 0
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(scores, dim=-1)                        # (B, T, T), rows sum to 1
        v = self.value(x)                                    # (B, T, hs)
        out = w @ v                                          # (B, T, hs)
        return (out, w) if return_weights else out


# %% [markdown]
"""
## Wire it into a language model — and why we still add position

Minimal attention LM, deliberately stripped to isolate what *attention alone* buys:
token embedding + **position** embedding → one `Head` → a linear head to the vocab. No
MLP, no residual, no norm (that's `06`).

Why keep the position embedding? Attention's value-mixing is **permutation-
equivariant**: shuffle the tokens and the outputs shuffle with them but carry the same
content — the mechanism has no built-in notion of *where* a token sits. The `q·k` score
depends on token *content*, not on the distance between positions. So order enters the
model through the `pos_emb` added at the input (exactly as in `01`/`03`); attention then
learns to route with both content and position available in each vector. (`11`/RoPE will
move that position signal *inside* attention; here it rides in on the embedding,
unchanged from `03`.)
"""

# %% [markdown]
"""
### See it: attention alone is position-blind

Proof of that claim before we build on it. Strip attention to its **value-mixing** — the
per-token `q/k/v` projections and `softmax(q·kᵀ) @ v`, with **no causal mask and no
position** — and feed it a sequence `x` and a shuffled copy `x[perm]`. Permutation-
equivariance means the two orders of operation agree exactly:

    attend(x)[perm]  ==  attend(x[perm])

Reorder the tokens and the outputs just reorder with them — each token's result is
*unchanged*. The mechanism literally cannot tell where a token sits. (We drop the causal
mask here on purpose: the mask is *itself* a position signal, so it — like `pos_emb` —
injects order too. Dropping it isolates the pure content-based routing.)
"""

# %%
torch.manual_seed(0)
T, C = 5, 16
proj_q = nn.Linear(C, C, bias=False)
proj_k = nn.Linear(C, C, bias=False)
proj_v = nn.Linear(C, C, bias=False)


def attend_no_pos(x):
    """Unmasked self-attention, no position added: the pure value-mixing operation."""
    q, k, v = proj_q(x), proj_k(x), proj_v(x)
    w = F.softmax(q @ k.transpose(-2, -1) * C ** -0.5, dim=-1)
    return w @ v


x = torch.randn(T, C)                        # T token vectors (embeddings, no pos)
perm = torch.randperm(T)                     # a reordering of the T positions

attend_then_perm = attend_no_pos(x)[perm]    # attend, THEN shuffle the outputs
perm_then_attend = attend_no_pos(x[perm])    # shuffle the inputs, THEN attend

print(f"permutation of positions : {perm.tolist()}")
print(f"attend(x)[perm] == attend(x[perm]) ? "
      f"{torch.allclose(attend_then_perm, perm_then_attend, atol=1e-6)}")
print("-> reordering tokens only reorders the outputs; the values are unchanged.")
print("   attention has no built-in sense of position — pos_emb (at the input) and")
print("   the causal mask are what put order back in.")

# %% [markdown]
"""
#### Why the matmul forces this

Write "reorder the rows" as a permutation matrix `P` (`x[perm]` = `P @ x`). `P` is
orthogonal, so `Pᵀ @ P = I` — remember that, it's the whole trick. Push `P @ x` through
each step and track where the `P`s go:

    1. per-row projections    q' = (Px)Wqᵀ = P(xWqᵀ) = P q     (same for k, v)
    2. scores  S = q kᵀ       S' = (Pq)(Pk)ᵀ = P S Pᵀ
    3. W = softmax(S)         W' = softmax(P S Pᵀ) = P W Pᵀ
    4. out = W v              out' = (P W Pᵀ)(P v) = P W (PᵀP) v = P W v = P out

- Projections pick up a **left** `P` because they act on the *feature* axis (right
  index), which commutes with `P` on the *position* axis (left): `(Px)W = P(xW)`.
- A score couples *two* positions, so `P` lands on **both** sides: `S'[i,j] = S[πi,πj]`.
- Softmax is per-row and permutation-equivariant within a row, so it carries `P … Pᵀ`
  straight through.
- Keys and values share the same positions, so the weights' right `Pᵀ` meets `v`'s left
  `P` and they **annihilate** (`PᵀP = I`). Only the query-side `P` survives → it
  reorders the outputs, and nothing else moves.

Strip the matrices and one row is `out[i] = Σ_j softmax_j(q_i·k_j) · v_j` — a **sum over
the key/value axis `j`**, and a sum ignores term order. Each `v_j` is addressed by its
content-derived weight, never by *which slot* it sits in; the only position that matters
is `i`, which query you're computing. That set-aggregation over an unordered bag of
(key, value) pairs is exactly why attention is order-blind until `pos_emb` is added.
"""

# %% [markdown]
"""
#### The same four lines, in code

If the `P` algebra didn't land, watch it happen on real tensors. We reuse `x`, `perm`,
and the `proj_*` from the demo, build the actual permutation matrix `P`, and check each
numbered line prints `True`. `P @ z` reorders the rows of `z` exactly like `z[perm]` —
that's all `P` is — and `Pᵀ @ P = I` is the identity that collapses step 4.
"""

# %%
T, C = x.shape                               # reuse x, perm, proj_* from the demo above
eq = lambda a, b: torch.allclose(a, b, atol=1e-6)

# P is the perm as a matrix: row i has a single 1 in column perm[i]
P = torch.zeros(T, T)
P[torch.arange(T), perm] = 1.0
print("P (permutation matrix):")
print(P.int())
print(f"x[perm] == P @ x ?  {eq(x[perm], P @ x)}   (P reorders rows)")
print(f"Pᵀ @ P == I ?       {eq(P.T @ P, torch.eye(T))}   (the step-4 trick)")


def parts(inp):
    """Run attention, returning every intermediate so we can compare them."""
    q, k, v = proj_q(inp), proj_k(inp), proj_v(inp)
    S = q @ k.transpose(-2, -1) * C ** -0.5
    W = F.softmax(S, dim=-1)
    return q, S, W, W @ v


q, S, W, out = parts(x)                       # on the original order
qp, Sp, Wp, outp = parts(x[perm])             # on the shuffled order (x[perm] == P @ x)

print(f"\n1. q'   == P q      {eq(qp, P @ q)}   (per-row proj -> left P)")
print(f"2. S'   == P S Pᵀ   {eq(Sp, P @ S @ P.T)}   (two positions -> both sides)")
print(f"3. W'   == P W Pᵀ   {eq(Wp, P @ W @ P.T)}   (softmax carries it)")
print(f"4. out' == P out    {eq(outp, P @ out)}   (the payoff)")

# step 4 spelled out: out' = W'v' = (P W Pᵀ)(P v) = P W (PᵀP) v = P W v
step4 = (P @ W @ P.T) @ (P @ proj_v(x))       # = W' v', the real computation on x[perm]
cancel = P @ W @ (P.T @ P) @ proj_v(x)        # insert PᵀP -> it is the identity
print(f"\nout' = (PWPᵀ)(Pv) = PW(PᵀP)v = P(Wv):  {eq(step4, cancel)}")
print("-> the inner PᵀP vanishes; only the query-side P remains -> outputs reorder.")


# %%
class AttnLM(nn.Module):
    """token+position embed -> one causal self-attention head -> vocab logits."""

    def __init__(self, vocab_size, n_embd=24, block_size=BLOCK):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)      # order lives here
        self.head = Head(n_embd, head_size=n_embd, block_size=block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)            # (B, T, C)
        a = self.head(x)                                     # (B, T, C)
        logits = self.lm_head(a)                             # (B, T, vocab)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]      # crop to what pos_emb covers
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# %% [markdown]
"""
## Train it on the shared loop

Same deterministic `full_val_loss` and AdamW loop as `03`/`04`, so the number lands on
the same scoreboard. One attention head, ~a couple thousand steps.
"""


# %%
import math


@torch.no_grad()
def full_val_loss(m, bs=256):
    """Mean cross-entropy over the WHOLE validation split in non-overlapping BLOCK
    windows — the identical deterministic metric `03`/`04` used."""
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


def train(m, max_iters=3000, eval_every=1000, lr=1e-2):
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    for it in range(max_iters + 1):
        if it % eval_every == 0:
            print(f"  step {it:>4} : val {full_val_loss(m):.3f}")
        xb, yb = get_batch("train")
        _, loss = m(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return full_val_loss(m)


head_model = AttnLM(vocab_size).to(DEV)
print(f"single-head attn params : {sum(p.numel() for p in head_model.parameters()):,}")
print("\ntraining single-head attention:")
head_floor = train(head_model)

# %% [markdown]
"""
## Scoreboard: attention clears the floor the average couldn't

The earlier numbers on the same data and metric: bigram `04` floor **2.45**, uniform
average **2.95** (worse than the bigram!), Bengio concat **1.84**, `01`'s full
transformer **1.594**. A *single* attention head is the **same weighted-average shape**
as `04`'s average — but with *learned* weights, and that alone flips a model that scored
*worse* than the bigram into one comfortably below it (~2.38). Same mechanism, one
change (frozen → learned), opposite side of the floor.

It does **not** beat concat yet, and that's the honest, useful part: concat has a
nonlinear hidden MLP, while this model is a *linear* readout over a single weighted mix.
**Attention is the mixing op — it moves information between tokens but does no per-token
"thinking."** Give it the per-token MLP (`06`) and depth and you get `01`'s 1.594.
Multi-head, next, takes the first step down.
"""

# %%
print(f"bigram        (04, CTX=1)        : 2.450   <- the floor")
print(f"average       (04, uniform mix)  : 2.950   <- worse: no order, no focus")
print(f"concat/Bengio (04, rigid window) : 1.840")
print(f"attention     (05, 1 head)       : {head_floor:.3f}   <- learned weighted mix")
print(f"transformer   (01, 4 heads x 4)  : 1.594")

# %% [markdown]
"""
## Weakness 1 fixed: attention uses order

`04`'s average was **order-blind**: `mean("ab") == mean("ba")`, so reordering a context
never changed its prediction. Attention isn't — the query·key scores route different
mass to different positions, and the added position embedding lets identical tokens in
different slots behave differently. Same test as `04`: feed a context and a reversal,
and check the next-token distribution actually moves.
"""


# %%
@torch.no_grad()
def next_probs(m, context_ids):
    x = torch.tensor([context_ids], device=DEV)
    logits, _ = m(x)
    return F.softmax(logits[0, -1], dim=-1)


base = encode("to be or")
perm = base[::-1]
print(f"base context : {decode(base)!r}")
print(f"reversed     : {decode(perm)!r}")

same = torch.allclose(next_probs(head_model, base),
                      next_probs(head_model, perm), atol=1e-6)
print(f"\nreorder changes the prediction? {not same}   "
      f"(True -> attention uses order; 04's average gave False)")

# %% [markdown]
"""
## Weakness 2 fixed: attention focuses — read the weights

The other `04` failure was uniform weight: every token in the window counted `1/CTX`, so
the informative last token got diluted. Attention's `W` is learned and non-uniform, and
— unlike a black-box improvement — **we can look at it**. Pull the softmax weights `W[t,
s]` straight out of the trained head for a real line and plot them.

Read the heatmap as: **row `t` = the token being predicted-from, column `s` = which past
token it leans on**, brightness = weight. It must be lower-triangular (zero above the
diagonal — the causal mask, visible), the rows must sum to 1, and crucially it is
**not** a flat `1/(t+1)`: bright cells mark the positions this head decided matter
(often the recent few, sometimes a specific earlier char). That structure is exactly
what the uniform average could not express.
"""

# %%
import matplotlib.pyplot as plt

sample = "To be, or not to"
ids = torch.tensor([encode(sample)], device=DEV)
head_model.eval()
with torch.no_grad():
    x = head_model.tok_emb(ids) + head_model.pos_emb(
        torch.arange(ids.size(1), device=DEV)
    )
    _, W = head_model.head(x, return_weights=True)
head_model.train()
W = W[0].cpu()                                   # (T, T) weights for this one line

row_sums = W.sum(dim=1)
ones = torch.ones_like(row_sums)
print(f"rows sum to 1? {torch.allclose(row_sums, ones, atol=1e-5)}")
print(f"strictly causal (no mass above diagonal)? "
      f"{torch.triu(W, diagonal=1).abs().max().item() < 1e-9}")

fig, ax = plt.subplots(figsize=(5.2, 4.6))
im = ax.imshow(W, cmap="viridis")
ax.set_xticks(range(len(sample)))
ax.set_yticks(range(len(sample)))
ax.set_xticklabels(list(sample))
ax.set_yticklabels(list(sample))
ax.set_xlabel("attended-to position s (the past)")
ax.set_ylabel("query position t")
ax.set_title("attention weights W[t, s]  (lower-triangular, learned, non-uniform)")
fig.colorbar(im, ax=ax, fraction=0.046, label="weight")
plt.tight_layout()
plt.show()

# %% [markdown]
"""
## The causal mask, operationalized

The heatmap being triangular is one view; here's the behavioral one. Causality means a
prediction at position `t` **cannot depend on any token after `t`**. So if we perturb
only the *last* token of a sequence, every earlier position's output must be byte-for-
byte unchanged. This is the single most important invariant of a decoder — get the mask
wrong and the model "cheats" by reading the answer, training loss looks great, and
generation is garbage.
"""

# %%
xb, _ = get_batch("val")
head_model.eval()
with torch.no_grad():
    out1, _ = head_model(xb)
    xb2 = xb.clone()
    xb2[:, -1] = torch.randint(vocab_size, (xb.size(0),), device=DEV)  # last only
    out2, _ = head_model(xb2)
head_model.train()
unchanged = torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-6)
print(f"perturbing the last token leaves all earlier outputs identical? {unchanged}")
print("-> the causal mask holds: no position reads its future.")

# %% [markdown]
"""
## Why the `1/sqrt(head_size)` scale? (measure it)

The scores are dot products over `head_size` dimensions. If `q` and `k` have ~unit-
variance entries, `q·k` sums `head_size` independent products, so its **variance grows
with `head_size`** — big `head_size` → large-magnitude scores. Push large values through
softmax and it **saturates**: nearly all mass collapses onto one position, the
distribution goes one-hot, and its gradient (`p_i(δ - p_j)`) vanishes — the head can
barely learn *where* to look. Dividing by `sqrt(head_size)` rescales the variance back
to ~1. Let's watch it in the *actual* `Head.forward` layout: a query block `(T, hs)` and
a key block `(T, hs)`, whose product `q @ k.T` is the full `(T, T)` score matrix. Every
entry is a length-`hs` dot product, so the scores carry variance ~`hs` until we scale.
Read one query's row (its scores over all `T` keys) through softmax, scale off vs on.
"""

# %%
torch.manual_seed(0)
T, hs = 8, 64
q = torch.randn(T, hs)                       # a query block, exactly as inside Head
k = torch.randn(T, hs)                       # a key block

scores = q @ k.transpose(-2, -1)             # (T, T): scores[i, j] = q[i] · k[j]
scores_scaled = scores * hs ** -0.5          # the 1/sqrt(hs) fix

print(f"scores shape {tuple(scores.shape)}  (T x T, one length-hs dot product each)")
print(f"var(scores)  unscaled = {scores.var().item():6.2f}  (~ head_size = {hs})")
print(f"var(scores)  scaled   = {scores_scaled.var().item():6.2f}  (~ 1)")

# one query's row = its score distribution over all T keys; softmax it both ways
i = 0
w_unscaled = F.softmax(scores[i], dim=-1)
w_scaled = F.softmax(scores_scaled[i], dim=-1)
fmt = lambda w: "[" + " ".join(f"{v:.2f}" for v in w.tolist()) + "]"
print(f"\nrow {i} softmax unscaled : {fmt(w_unscaled)}")
print(f"  -> peak {w_unscaled.max().item():.3f}  (one key dominates -> near one-hot)")
print(f"row {i} softmax scaled   : {fmt(w_scaled)}")
print(f"  -> peak {w_scaled.max().item():.3f}  (spread over keys -> learns freely)")

# %% [markdown]
"""
## Single head → multi-head: several relationships at once

One head learns *one* way to route — maybe "look at the previous character," or "find
the last space." Language needs many such relationships simultaneously. **Multi-head
attention** runs `n_head` independent heads in parallel, each with its own small `q/k/v`
of size `head_size = n_embd / n_head`, then **concatenates** their outputs back to width
`n_embd` and applies an **output projection** that lets the heads' information mix.

Total width is unchanged (`n_head * head_size = n_embd`), so this isn't "more compute
per token" so much as "split the channels into several independent attention patterns."
Here's the explicit version — a literal list of `Head`s — to make that concrete before
we fuse it.
"""


# %%
class MultiHeadAttention(nn.Module):
    """n_head independent Heads in parallel, concatenated, then mixed."""

    def __init__(self, n_embd, n_head, block_size=BLOCK):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must divide by n_head"
        head_size = n_embd // n_head
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_embd, n_embd)   # lets heads' outputs interact

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)   # (B, T, n_embd)
        return self.proj(out)


# %% [markdown]
"""
Drop multi-head into the same tiny LM (swap the single `Head` for `MultiHeadAttention`)
and train it on the same loop. Four heads over the same 24-wide embedding should edge
below the single head — more routing patterns, same width.
"""


# %%
class MultiHeadLM(nn.Module):
    def __init__(self, vocab_size, n_embd=24, n_head=4, block_size=BLOCK):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        a = self.attn(x)
        logits = self.lm_head(a)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    generate = AttnLM.generate       # identical autoregressive loop


mh_model = MultiHeadLM(vocab_size).to(DEV)
print(f"multi-head (4) params : {sum(p.numel() for p in mh_model.parameters()):,}")
print("\ntraining 4-head attention:")
mh_floor = train(mh_model)
print(f"\nsingle head : {head_floor:.3f}")
print(f"4 heads     : {mh_floor:.3f}   <- more routing patterns, same width")

# %% [markdown]
"""
## The fused version — the box `01` actually ran

The `ModuleList` of heads is a Python `for` loop: readable, but it launches `n_head`
separate matmuls. The production form computes **all heads in a couple of big batched
matmuls**. The trick: project `x` to `Q, K, V` each of width `n_embd` in one `Linear`,
then reshape `n_embd → (n_head, head_size)` and move `n_head` next to the batch dim, so
a single batched matmul does every head at once.

This `CausalSelfAttention` is training-time only — **no KV-cache, no RoPE** (those are
`12` and `11`; adding them here is a one-line change *to this exact module*, which is
why they're kept separate). It's the same math and the same parameter budget as the
explicit multi-head above, and it's what `01`'s `nn.MultiheadAttention` computes under
the hood.
"""


# %%
class CausalSelfAttention(nn.Module):
    """All heads at once in batched matmuls: qkv proj, reshape, masked softmax."""

    def __init__(self, n_embd, n_head, block_size=BLOCK):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_size = n_embd // n_head
        # one projection producing Q, K, V stacked (C -> 3C); split in forward
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape
        nh, hs = self.n_head, self.head_size
        q, k, v = self.qkv(x).split(self.n_embd, dim=-1)     # each (B, T, C)
        # split C into (nh, hs) and move nh beside the batch dim -> (B, nh, T, hs)
        q = q.view(B, T, nh, hs).transpose(1, 2)
        k = k.view(B, T, nh, hs).transpose(1, 2)
        v = v.view(B, T, nh, hs).transpose(1, 2)
        # batched over (B, nh): scores -> causal mask -> softmax -> weighted values
        scores = q @ k.transpose(-2, -1) * hs ** -0.5        # (B, nh, T, T)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(scores, dim=-1)
        out = w @ v                                          # (B, nh, T, hs)
        # transpose breaks contiguity, so .contiguous() before .view (the view gotcha)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


# %% [markdown]
"""
## Check the fused attention against torch's own kernel

The honest test: our hand-rolled scaled-dot-product-with-causal-mask-and-softmax must
match `F.scaled_dot_product_attention` — the exact fused kernel `nn.MultiheadAttention`
(and thus `01`) dispatches to. Same `q, k, v`, `is_causal=True`, compare to ~0. If this
passes, every line of softmax attention we wrote is validated against the library.
"""

# %%
torch.manual_seed(0)
B, T, nh, hs = 2, 16, 4, 8
q = torch.randn(B, nh, T, hs)
k = torch.randn(B, nh, T, hs)
v = torch.randn(B, nh, T, hs)

# ours: the same three lines as CausalSelfAttention.forward
tril = torch.tril(torch.ones(T, T))
scores = q @ k.transpose(-2, -1) * hs ** -0.5
scores = scores.masked_fill(tril == 0, float("-inf"))
ours = F.softmax(scores, dim=-1) @ v

# torch's fused kernel, causal flag on
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

print(f"max abs diff vs F.scaled_dot_product_attention: "
      f"{(ours - ref).abs().max().item():.2e}")
print(f"match to ~0? {torch.allclose(ours, ref, atol=1e-6)}")

# and the fused module is causal + right-shaped end to end
csa = CausalSelfAttention(n_embd=24, n_head=4).to(DEV)
xb, _ = get_batch("val")
emb = nn.Embedding(vocab_size, 24).to(DEV)(xb)
o1 = csa(emb)
emb2 = emb.clone()
emb2[:, -1] = torch.randn_like(emb2[:, -1])
o2 = csa(emb2)
print(f"fused output shape {tuple(o1.shape)}, still causal? "
      f"{torch.allclose(o1[:, :-1], o2[:, :-1], atol=1e-6)}")

# %% [markdown]
"""
## Read the babble

Generate from the trained multi-head model. It's clearly text-shaped — line breaks,
capitalized speaker names with colons — each character chosen from a *learned,
unbounded* look back rather than a rigid 8-token window. Don't expect it to read cleaner
than `04`'s concat yet: a single mixing layer with no per-token MLP and no depth can't
assemble real words or grammar. Stacking the MLP (`06`) and layers on top of this exact
attention is what closes that gap.
"""

# %%
start = torch.tensor([[stoi["\n"]]], device=DEV)
print(decode(mh_model.generate(start, max_new_tokens=300)[0].tolist()))

# %% [markdown]
"""
## What you built, and the box `06` opens next

You now have the transformer's defining operation, from scratch and validated:

- attention is `04`'s causal weighted average with the weights **learned per row** from
  `q·k` — same `W @ v` shape, non-uniform content-dependent `W`;
- it recovers **order** (via position + scoring) and **focus** (non-uniform, readable
  weights), the two things `04`'s average threw away;
- the `1/sqrt(head_size)` scale keeps softmax out of its saturated, no-gradient regime;
- **multi-head** runs several routing patterns in parallel; the **fused**
  `CausalSelfAttention` does it in batched matmuls and matches torch's kernel to ~0.

One thread to carry forward: the score matrix is `T×T`, so attention is **O(T²)** in
sequence length — the cost that later motivates the KV-cache (`12`) and GQA (`13`).

Attention *mixes* information between tokens. It does **no per-token processing** —
after mixing, each token still needs to *think* on its own, and the stack needs residual
+ norm to train at depth. That's `06`: the per-token MLP and the Block that wraps
attention and MLP so `01`'s four layers actually train. That's the last box of the core
architecture.
"""
