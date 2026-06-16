"""
Phase 2 - Exercise: a minimal training loop.

Goal: train the baseline model and watch the loss fall from ~4.17 (ln 65), then
sample text so you can see what this position+bigram model can and cannot do.
(Spoiler: it learns letter frequencies but produces gibberish — it can't look
across positions. That limitation is the whole motivation for attention next.)

Fill in the TODOs, then run `python llm/train.py`.
"""

from __future__ import annotations

import torch

from config import GPTConfig
from data import load_data, get_batch
from model import GPT

# --- hyperparameters (small, so it runs fast on CPU) ---
BLOCK_SIZE = 32
BATCH_SIZE = 32
N_EMBD = 64
MAX_ITERS = 3000
EVAL_INTERVAL = 300
EVAL_ITERS = 100
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """Average loss over EVAL_ITERS batches for train and val.

    Why a separate function (not just the latest training-batch loss)?
    A single batch is noisy; averaging many gives a stable read. And we wrap it
    in torch.no_grad() + model.eval() so it doesn't build a graph or apply any
    train-only behavior (dropout later).

    """
    model.eval()
    result = {}
    for split, data in (("train", train_data), ("val", val_data)):
        total = 0.0
        for _ in range(EVAL_ITERS):
            xb, yb = get_batch(data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
            _, loss = model(xb, yb)
            total += loss.item()
        result[split] = total / EVAL_ITERS
    model.train()
    return result


def main():
    torch.manual_seed(1337)

    # 1. data + tokenizer. NOTE: vocab_size for the config comes from the
    #    tokenizer, not a magic number.
    tokenizer, train_data, val_data = load_data()

    # config's vocab_size comes from the tokenizer, not a magic number
    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
    )
    model = GPT(cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(MAX_ITERS):
        # periodic eval (and a final read on the last step)
        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")

        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)   # grads accumulate by default; clear them
        loss.backward()
        optimizer.step()

    # sample from the trained model, starting from a single newline/token id 0
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    out = model.generate(context, max_new_tokens=500)[0].tolist()
    print(tokenizer.decode(out))


if __name__ == "__main__":
    main()
