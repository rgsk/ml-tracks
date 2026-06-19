# Advanced LLM Roadmap — remaining topics

Exhaustive list of topics not yet built in `llm/`, ranked most → least important
for a GPT-from-scratch interview build. Each has a plain-language "what problem
it solves." Items 2–3 are already on the Phase 5 plan; included since they're
still remaining.

> This file is a personal reference — maintained by me (Rahul), not auto-updated
> by Claude. Tick items off as I go.

---

## Architecture & core mechanics

**1. BPE / subword tokenizer** — Right now every *character* is a token, so "the"
costs 3 tokens and the model wastes capacity learning spelling. BPE merges
frequent character pairs into subword chunks ("the", "ing"), making sequences
shorter and tokens meaningful. What GPT-2/3/4 actually use — and `tokenizer.py`
already flags it as the next step.

**2. RoPE (rotary position embeddings)** — The model learns one vector per
absolute position, so it can't go beyond its trained length and bakes in position
awkwardly. RoPE *rotates* query/key vectors by an angle based on position, so
attention naturally depends on the *distance* between tokens. Enables longer
context and a real sliding-window cache.

**3. FlashAttention** — Attention builds a big T×T score matrix in slow GPU
memory — the bottleneck for long sequences. FlashAttention computes attention in
tiles that stay in fast on-chip memory and never materializes the full matrix:
same math, far less memory and time.

**4. GQA / MQA (grouped / multi-query attention)** — The KV-cache stores
keys/values for every head, eating memory during generation. MQA/GQA share K/V
across groups of heads, shrinking the cache a lot with minimal quality loss. Why
Llama-2/Mistral serve long contexts cheaply. Pairs directly with the KV-cache work.

**5. RMSNorm** — LayerNorm centers *and* scales activations; RMSNorm drops the
centering (just scales by root-mean-square) — simpler, slightly faster, no quality
loss. Llama uses it. Classic "why does this still work?" question.

**6. SwiGLU / GeGLU** — The feed-forward is Linear→GELU→Linear. Gated variants add
a second "gate" branch multiplied element-wise, which learns more per parameter.
Used by Llama/PaLM.

## Generation / inference

**7. Top-p (nucleus) sampling** — We have temperature + top-k. Top-k always keeps a
*fixed* number of tokens even when the model is very sure or very unsure. Top-p
keeps the smallest set of tokens whose probabilities sum to p (e.g. 0.9), adapting
to the model's confidence. Completes the sampling trio.

**8. Activation (gradient) checkpointing** — Training stores every layer's
activations for the backward pass, which dominates memory in deep models.
Checkpointing discards most and *recomputes* them during backward — trading
compute for big memory savings, so you fit bigger models/batches. (Different from
saving model checkpoints to disk.)

## Fine-tuning & alignment

**9. LoRA / QLoRA (parameter-efficient fine-tuning)** — Fine-tuning all weights of
a big model is expensive and memory-hungry. LoRA freezes the original weights and
trains tiny low-rank "adapter" matrices instead — 100–1000× fewer trainable
params, same task quality. QLoRA adds 4-bit quantization so you can fine-tune huge
models on one GPU. One of *the* most-asked topics right now.

**10. Supervised fine-tuning (SFT) / instruction tuning** — A base model just
predicts next tokens; it doesn't follow instructions. SFT continues training on
(instruction, good answer) pairs so it learns to respond helpfully. The first step
from raw LLM to assistant.

**11. RLHF (reward model + PPO)** — SFT teaches imitation; RLHF teaches human
*preferences*. Train a reward model on human comparisons, then use RL (PPO) to push
the LLM toward higher-reward answers. How ChatGPT-style alignment works. Overlaps
the planned RL track.

**12. DPO (direct preference optimization)** — RLHF-with-PPO is finicky and
unstable. DPO gets the same "prefer the better answer" effect with a simple
supervised loss directly on preference pairs — no separate reward model, no RL
loop. The modern, simpler alternative.

## Scaling & systems

**13. Distributed Data Parallel (DDP)** — One GPU is too slow/small for real
training. DDP puts a full model copy on each GPU, each sees different data, then
averages (all-reduces) gradients to stay in sync. The standard multi-GPU method.

**14. FSDP / ZeRO + tensor/pipeline parallelism** — When the model itself won't fit
on one GPU, you must shard it. FSDP/ZeRO split parameters, gradients, and optimizer
states across GPUs; tensor/pipeline parallelism split individual layers or the
layer stack. How models bigger than a single GPU get trained.

**15. Quantization (int8 / 4-bit, PTQ/QAT, GPTQ/AWQ)** — fp32/bf16 weights take
lots of memory and bandwidth at inference. Quantization stores them in 8- or 4-bit
integers (~4–8× smaller, faster serving), with schemes to limit accuracy loss. Key
for deploying LLMs.

**16. torch.compile / kernel fusion** — Running ops one-by-one has launch overhead
and extra memory traffic. torch.compile traces the model into a graph and fuses ops
into fewer, faster kernels — often a near-free 1.3–2× speedup. Ties straight into
the profiling work.

## Evaluation & data

**17. Perplexity** — We report loss; perplexity is just exp(loss) — the standard LM
quality metric ("how many tokens it's effectively choosing between"). Trivial to
add, but it's the named metric interviewers expect.

**18. Dataset / DataLoader + sequence packing** — `get_batch` is a custom sampler.
Real pipelines use torch DataLoader (parallel workers, shuffling) and "pack"
multiple short documents into one sequence to avoid wasting compute on padding.
Data-throughput engineering.

**19. Experiment tracking (Weights & Biases)** — Printing loss to the terminal
doesn't scale across many runs. W&B logs metrics/configs/samples to a dashboard so
you can compare runs and catch divergence early. (Flagged in `todo.txt`.)

## Alternatives & nice-to-knows

**20. ALiBi (attention with linear biases)** — Another position scheme: add a
distance-based penalty to attention scores (farther = more negative), no learned
embeddings. Simpler than RoPE and extrapolates to longer contexts. The thing you
compare RoPE against.

**21. Repetition / frequency / presence penalties** — Models loop ("the the the").
These penalties lower the probability of already-generated tokens to keep output
diverse. Common generation knobs.

**22. Speculative decoding** — Generation is slow because tokens come one at a time.
A small "draft" model proposes several tokens; the big model verifies them in one
pass, accepting the correct ones — 2–3× faster with identical output. Hot inference
topic.

**23. Mixture of Experts (MoE)** — Instead of every token using the whole network,
MoE has many "expert" FFNs and a router sends each token to just a few. Big-model
capacity at small-model compute cost. Used by Mixtral (and rumored GPT-4).

**24. Beam search** — Rather than sampling one token at a time, keep the top-k most
likely *sequences* and expand them, choosing the best overall. Standard for
translation/summarization; less used for chat but expected knowledge.

**25. Sinusoidal positional encoding** — The original Transformer's fixed
(non-learned) position signal from sine/cosine waves at different frequencies.
Worth knowing as the historical baseline that learned/RoPE positions replaced.

## Minor / tooling

**26. Label smoothing** — Targets say "100% this token," making the model
overconfident. Label smoothing spreads a little probability to other tokens,
improving calibration. Small regularization trick.

**27. Weight EMA (exponential moving average)** — Keep a slowly-updated average of
the weights alongside the training ones; the averaged version is often smoother and
generalizes better at inference.

**28. Gradient / backward hooks** — A debugging tool: attach a function to a tensor
to inspect or modify its gradient mid-backprop — e.g. to locate exploding/vanishing
gradients.

**29. CUDA memory profiling** — Beyond timing (done), tools like
`torch.cuda.max_memory_allocated` and memory snapshots show where GPU memory goes —
for fixing OOMs and right-sizing batches.

**30. Contrastive search / other decoding** — Niche decoders that balance likelihood
vs. diversity to cut repetition without randomness. Good to be aware of, rarely
required.
