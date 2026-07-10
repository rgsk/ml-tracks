# LLM — roadmap (top-down, GPT-from-scratch, notebook edition)

**How this is taught: the whole game first.** `01` builds a real GPT — token embeddings, causal
self-attention, MLP blocks — and **trains it on tinyshakespeare in the first minute**, so you *see it
work*: watch the loss fall past the bigram floor and read back Shakespeare-shaped text. Every notebook
after that **opens one box of that exact model** and explains the *why* — measured, not asserted. You
always know where a detail lives, because you already ran the model it's part of.

It's the Karpathy `makemore`→`nanoGPT` move: get a basic version generating text with a rough mental
map of what's happening, then dig in and break each piece apart. This is the notebook re-presentation
of the bottom-up `llm/` build — same concepts, retold top-down, one runnable lesson at a time.

Alignment (RLHF/PPO/DPO/GRPO) is **not** here — it lives in `nb/rl/` Phase 3, which plugs the model
this track builds straight into the RL loop. This track ends at the supervised edge (SFT, LoRA) and
hands off there, so nothing is taught twice.

---

## Format (jupytext paired notebooks)

Each experiment is **one notebook** — a lesson you read top-to-bottom, with prose, code, and figures
inline. Every notebook is a **jupytext pair**:

- `walkthroughs/NN_name.py` — the **source of truth** (`py:percent`); clean git diffs, editable in the
  editor. This is what you edit.
- `walkthroughs/NN_name.ipynb` — the **rendered pair**: same cells plus executed outputs (prints,
  samples, figures). This is what you read. It's **gitignored** and regenerated.

Rebuild/execute a notebook after editing its `.py`:

```bash
uv run jupytext --to ipynb --execute nb/llm/walkthroughs/01_whole_game.py   # re-run all cells
```

```
nb/llm/
  roadmap.md                 <- this file: the table of contents + how it's taught
  walkthroughs/
    01_whole_game.py   ⇄ .ipynb   the whole game: build a small GPT, train, generate text
    02_tokenizer.py    ⇄ .ipynb   open the input: chars → BPE, what a "token" is (encode/decode)
    03_embed_bigram.py ⇄ .ipynb   token/position embeddings + the bigram baseline (the floor to beat)
    04_context_combine.py ⇄ .ipynb combine >1 previous token: concat (Bengio MLP) vs average — road to attn
    05_attention.py    ⇄ .ipynb   the heart: self-attention — Q/K/V, the causal mask, single→multi-head
    06_mlp_block.py    ⇄ .ipynb   per-token MLP + the Block: residuals + pre-norm, why depth stacks
    07_train_craft.py  ⇄ .ipynb   the training loop: AdamW, weight init, LR warmup+cosine, grad clip
    08_sampling.py     ⇄ .ipynb   turning logits into text: temperature, top-k, top-p, greedy
    09_rmsnorm.py      ⇄ .ipynb   drop LayerNorm's centering — scale by RMS (simpler, same quality)
    10_swiglu.py       ⇄ .ipynb   a gated feed-forward that learns more per parameter
    11_rope.py         ⇄ .ipynb   rotary Q/K → attention depends on relative distance (longer context)
    12_kv_cache.py     ⇄ .ipynb   don't recompute past K/V each step — the core inference speedup
    13_gqa.py          ⇄ .ipynb   share K/V across head groups to shrink the KV-cache
    14_sft.py          ⇄ .ipynb   base → instruction-follower: continue training with loss masking
    15_lora.py         ⇄ .ipynb   freeze the base, train tiny low-rank adapters (~1% of the params)
  custom/                    <- from-scratch impls (softmax, cross_entropy, layer/rms-norm, attention);
                                each runs standalone as a self-test, matched against torch to ~0
  model.py                   <- the cleaned-up GPT, assembled once we understand each piece
  artifacts/                 <- generated files (gitignored), mirrors llm/artifacts/ layout
    tokenizer/               <- trained BPE merges cached here (built by 02, loaded by later notebooks)
    checkpoints/             <- gpt.pt: trained by 01, loaded by later notebooks (no retrain)
```

Notebooks stay independent: `01` trains and **saves `checkpoints/gpt.pt`**; later notebooks **load it
if present, else train a quick one**. So you can open any notebook on its own, and editing `05` never
re-runs `01`.

---

## Phase A — build & understand the model

**`01` — the whole game.** Assemble a small GPT (token+position embeddings → a stack of
`attention → MLP` blocks → a linear head over the vocab), train it on tinyshakespeare, **watch val
loss drop below the bigram floor**, and generate a passage. One-line narration per part — no rigor
yet. Payoff on screen immediately.
> After this you have the map. Everything below zooms into **one box you already ran.**

**`02` — open the input: what is a token?** The character tokenizer (every char = one id) as the
baseline, then **BPE** built from scratch in `custom/`: merge frequent byte pairs into subword chunks
so sequences get shorter and tokens carry meaning. Encode/decode round-trips, the compression win,
and *why* it matters (fewer, more meaningful tokens = more effective context per step).

**`03` — embeddings + the bigram baseline.** How ids become vectors (token embedding table), why the
model also needs **position** information, and the simplest possible LM — predict the next token from
the current one alone. This is the **floor** `01` had to beat; it also fixes the training-loop and
`generate` skeleton every later notebook reuses.

**`04` — combining context: concat vs average.** The bigram only sees one token; to beat its floor
you must use *more than one*. The two classic ways to collapse a window of token-vectors into a
prediction: **concatenate** them (the Bengio 2003 neural LM — keeps order, but a rigid, parameter-
hungry window) and **average** them (a bag-of-chars — scales to any length but destroys order and
can't focus). Measured payoff: concat crushes the floor, while a uniform average lands *worse* than
the bigram despite 8× the context. That "how you combine matters more than how much" is exactly what
sets up attention — a *learned, weighted* average.

**`05` — the heart: self-attention.** Why a fixed context window needs tokens to *look at each other*.
Build attention from scratch: **query/key/value**, the scaled dot-product, the **causal mask** (a
token may only attend to the past), softmax weights → weighted sum of values. Then single-head →
**multi-head** (several attention "views" in parallel) and the fused projection. The one box that most
defines a transformer.

**`06` — the MLP and the Block.** After tokens mix (attention), each token is processed independently
by a small **MLP** (Linear → activation → Linear) — the per-token "compute". Then the **Block** that
`01` stacked: attention and MLP each wrapped in a **residual + pre-norm**, and *why* that wrapping is
what lets you stack depth without the signal blowing up or vanishing (measured).

**`07` — the training loop, properly.** Open `01`'s `train()` call: **AdamW** (and why decoupled weight
decay), sane **weight init** (GPT-2 scaled-residual std) and the init-loss sanity check (`≈ ln(vocab)`),
**LR warmup + cosine decay**, and **gradient clipping**. Each is a knob you can turn off and watch the
loss curve get worse.

**`08` — from logits to text: sampling.** The decode-time trio: **temperature** (sharpen/flatten the
distribution), **top-k** (keep the k most likely), **top-p / nucleus** (keep the smallest set summing to
p), and greedy as the T→0 limit. See how each changes the samples; note which knobs matter for
open-ended text vs. exact-answer tasks.

## Phase B — what modern LLMs changed

Swap the 2019 pieces for what Llama/Mistral/Qwen actually use — **one delta per notebook**, each a
small measured upgrade on the model you built.

**`09` — RMSNorm.** Drop LayerNorm's centering; just scale by root-mean-square. Simpler, one fewer
param vector, same quality — and *why* the re-centering turned out not to matter.

**`10` — SwiGLU.** Replace the Linear→GELU→Linear feed-forward with a **gated** variant (a second
branch multiplied element-wise) that learns more per parameter. Used by Llama/PaLM.

**`11` — RoPE.** Replace the learned absolute position table with **rotary** Q/K so the attention
score depends only on the *relative* distance between tokens — no length cap baked in. The change that
unlocks longer context and cache-friendly generation.

**`12` — KV-cache.** During generation, don't recompute past keys/values every step — cache and reuse
them. The core inference speedup; and why RoPE (`11`) makes a real sliding-window cache possible.

**`13` — GQA / MQA.** The KV-cache stores K/V for every head. Share them across **groups** of heads to
shrink the cache a lot with minimal quality loss — how long contexts get served cheaply.

## Phase C — base model → assistant (supervised)

**`14` — SFT.** A base model only predicts next tokens; it doesn't *follow instructions*. Continue
training on (instruction, good answer) pairs with **loss masking** — only the answer positions count.
The first step from raw LM to assistant.

**`15` — LoRA.** Freeze the base, train tiny low-rank **adapters** instead — same task quality, ~1% of
the trainable params, hot-swappable per task. The last *supervised* rung.
> Preference/RL alignment — reward modeling, PPO-RLHF, DPO, GRPO — is the **`nb/rl/` Phase 3** track,
> which imports this model directly. That's the bridge; follow it there.

---

### Threads that recur (called out as they appear, not separate items)
- **Attention cost**: the T×T score matrix is O(T²) — motivates the KV-cache, GQA, and FlashAttention.
- **Normalization & residuals**: pre-norm + residual is what makes depth trainable; LayerNorm→RMSNorm.
- **Position**: learned absolute → RoPE relative — the change that unlocks length extrapolation.
- **Cross-tokenizer metric**: per-token loss isn't comparable across vocabs; normalize to **bits/byte**.
- **What fine-tuning can and can't do**: SFT/LoRA reshape *behavior*; knowledge comes from pretraining.
```
