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

from bpe import BPETokenizer
from tokenizer import CharTokenizer

# resolve input.txt relative to THIS file, so it works from any cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "input.txt")


def load_data(path: str = DATA_PATH, train_frac: float = 0.9,
              tokenizer: str = "char", bpe_vocab_size: int = 1024):
    """Read corpus, build tokenizer, encode, split into train/val tensors.

    `tokenizer` is "char" (one token per character) or "bpe" (byte-level BPE,
    what GPT-2 uses). Returns (tokenizer, train_data, val_data) — 1-D LongTensors
    of token ids. The split is by position, not shuffled, so the val region stays
    genuinely unseen (no sequence leakage into train).

    BPE training + encoding the full corpus is slow (pure-Python, ~minutes), so
    the merges AND the encoded ids are cached to disk keyed by vocab size; later
    runs just load them.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    if tokenizer == "char":
        tok = CharTokenizer(text)
        ids = tok.encode(text)
    elif tokenizer == "bpe":
        tok, ids = _load_or_build_bpe(text, bpe_vocab_size)
    else:
        raise ValueError(f"unknown tokenizer {tokenizer!r} (use 'char' or 'bpe')")

    data = torch.tensor(ids, dtype=torch.long)
    n = int(train_frac * len(data))
    train_data, val_data = data[:n], data[n:]
    return tok, train_data, val_data


def _load_or_build_bpe(text: str, vocab_size: int):
    """Return (BPETokenizer, encoded_ids), training + encoding once then caching."""
    cache_dir = os.path.join(_HERE, "artifacts", "tokenizer")
    os.makedirs(cache_dir, exist_ok=True)
    model_path = os.path.join(cache_dir, f"bpe_v{vocab_size}.json")
    ids_path = os.path.join(cache_dir, f"bpe_v{vocab_size}_ids.pt")

    if os.path.exists(model_path) and os.path.exists(ids_path):
        tok = BPETokenizer.load(model_path)
        ids = torch.load(ids_path).tolist()
        return tok, ids

    print(f"[data] training BPE (vocab {vocab_size}) + encoding corpus — one-time, ~minutes...")
    tok = BPETokenizer()
    tok.train(text, vocab_size=vocab_size)
    ids = tok.encode(text)
    tok.save(model_path)
    torch.save(torch.tensor(ids, dtype=torch.long), ids_path)
    print(f"[data] cached to {os.path.basename(model_path)} / {os.path.basename(ids_path)}")
    return tok, ids


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
