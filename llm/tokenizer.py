"""
Character-level tokenizer.

A tokenizer maps between text and the integer token ids a model consumes:
  - encode: str -> list[int]
  - decode: list[int] -> str

Every unique character in the corpus is one token. This is the simplest scheme;
later we swap in BPE (what GPT-2 uses), keeping this same encode/decode interface.
"""

from __future__ import annotations


class CharTokenizer:
    def __init__(self, text: str):
        # sorted() makes the vocab deterministic across runs (set order is not)
        chars = sorted(set(text))
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(chars)}
        self.itos: dict[int, str] = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


# ---------------------------------------------------------------------------
# Self-check: run `python llm/tokenizer.py` from the repo root.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "input.txt"), "r", encoding="utf-8") as f:
        text = f.read()

    tok = CharTokenizer(text)

    # round-trip property: decode(encode(x)) == x
    sample = "Hello, there!\nFirst Citizen:"
    assert tok.decode(tok.encode(sample)) == sample, "round-trip failed"

    ids = tok.encode(sample)
    assert all(isinstance(i, int) for i in ids), "encode must return ints"
    assert all(0 <= i < tok.vocab_size for i in ids), "ids out of vocab range"

    print(f"vocab_size = {tok.vocab_size}")
    print(f"first 20 ids of sample = {ids[:20]}")
    print("all checks passed ✅")
