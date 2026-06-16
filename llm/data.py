"""
Phase 1 - Exercise 2: From corpus to training batches.

A GPT is trained on next-token prediction: given tokens [t0..t_{n-1}], predict
the token at each next position. So every training example is a pair (x, y)
where y is x shifted left by one:

    x = tokens[i      : i+block_size]
    y = tokens[i + 1  : i+1+block_size]

block_size (a.k.a. context length / sequence length) is how many tokens of
history the model sees at once.

This file has two jobs:
  1. load_data: read input.txt, encode it, split into train/val tensors.
  2. get_batch: sample a random batch of (x, y) pairs.

Fill in the TODOs. Run `python data.py` to self-check.
"""

from __future__ import annotations

import os

import torch

from tokenizer import CharTokenizer

# resolve input.txt relative to THIS file, so it works from any cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "input.txt")


def load_data(path: str = DATA_PATH, train_frac: float = 0.9):
    """Read corpus, build tokenizer, encode, split into train/val tensors.

    Returns (tokenizer, train_data, val_data), where the data tensors are
    1-D LongTensors of token ids. The split is by position, not shuffled, so
    the val region stays genuinely unseen (no sequence leakage into train).
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(train_frac * len(data))
    train_data, val_data = data[:n], data[n:]
    return tokenizer, train_data, val_data


def get_batch(data: torch.Tensor, block_size: int, batch_size: int,
              device: str = "cpu"):
    """Sample one batch of (x, y) for next-token prediction.

    Returns x, y each of shape (batch_size, block_size), dtype long, where y
    is x shifted left by one (the target at position t is the token at t+1).
    """
    # high is exclusive, so this leaves room for the +1 shift in y
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Self-check: run `python data.py`. Don't edit below this line.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(1337)

    tok, train_data, val_data = load_data()
    print(f"vocab_size = {tok.vocab_size}")
    print(f"train tokens = {len(train_data)}, val tokens = {len(val_data)}")

    block_size, batch_size = 8, 4
    x, y = get_batch(train_data, block_size, batch_size)

    # shape checks
    assert x.shape == (batch_size, block_size), f"bad x shape {x.shape}"
    assert y.shape == (batch_size, block_size), f"bad y shape {y.shape}"
    assert x.dtype == torch.long and y.dtype == torch.long, "must be long"

    # the y-is-x-shifted-by-one property: y[:, :-1] should equal x[:, 1:]
    assert torch.equal(x[:, 1:], y[:, :-1]), "y must be x shifted by one"

    # readable demo of the prediction targets in row 0
    print("\nrow 0 demo (what the model learns):")
    for t in range(block_size):
        context = x[0, : t + 1].tolist()
        target = y[0, t].item()
        print(f"  {context} -> {target}")
    print("\nall checks passed ✅")
