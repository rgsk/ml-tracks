"""
Byte-Pair Encoding (BPE) tokenizer — the scheme GPT-2/3/4 actually use.

WHY BPE (vs the CharTokenizer in tokenizer.py):
  - Char-level: every character is a token. "the" costs 3 tokens, sequences are
    long, and the model wastes capacity re-learning spelling.
  - Word-level: one token per word, but the vocab explodes and any unseen word
    (typos, names, "antidisestablishment...") is out-of-vocabulary.
  - BPE is the middle ground: start from bytes, then greedily MERGE the most
    frequent adjacent pair into a new token, over and over. Frequent chunks
    ("the", "ing", " the") become single tokens; rare stuff stays in small
    pieces. Shorter sequences, meaningful tokens, and ZERO out-of-vocab.

WHY BYTE-LEVEL (start from 0..255, not from the characters in the corpus):
  - Any text is UTF-8 bytes, and there are only 256 possible byte values, so the
    base vocab is a fixed 256. decode(encode(x)) == x for ANY string — even
    emoji or characters never seen in training. (CharTokenizer would KeyError.)

Same interface as CharTokenizer: encode(str)->list[int], decode(list[int])->str,
plus a `.vocab_size` attribute, so this can drop into the model unchanged.

Reference algorithm: Karpathy's minbpe.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Two tiny helpers do all the real work. Implement these first.
# ---------------------------------------------------------------------------

def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count how often each adjacent pair appears: [1,2,1,2,3] -> {(1,2):2,(2,1):1,(2,3):1}."""
    counts: dict[tuple[int, int], int] = {}
    for i in range(len(ids) - 1):
        counts[(ids[i], ids[i + 1])] = counts.get((ids[i], ids[i + 1]), 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """Replace every occurrence of `pair` with `idx`: merge([1,2,1,2,3],(1,2),99) -> [99,99,3].

    The i += 2 on a hit is what handles overlaps correctly: [1,1,1] merging
    (1,1) gives [99,1], not [99,99] — the middle element isn't reused.
    """
    new_ids: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


# ---------------------------------------------------------------------------
# The tokenizer.
# ---------------------------------------------------------------------------

class BPETokenizer:
    def __init__(self):
        # learned during train(); merges records the ORDER pairs were merged in.
        self.merges: dict[tuple[int, int], int] = {}   # (a, b) -> new_id
        self.vocab: dict[int, bytes] = {}              # id -> the bytes it expands to
        self.vocab_size = 0

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Learn vocab_size - 256 merges: greedily merge the most frequent pair, repeat.

        The new ids count up 256, 257, ... and self.merges' values ARE that order,
        which encode() must replay (earliest first) so a later merge only fires
        after the earlier one it builds on (e.g. t+h->th before th+e->the).
        """
        assert vocab_size >= 256, "vocab_size must be at least 256 (the byte base)"
        ids = list(text.encode("utf-8"))  # 0..255
        num_merges = vocab_size - 256
        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break  # nothing left to merge
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = merge(ids, pair, idx)
            self.merges[pair] = idx

        # id -> bytes, so decode can expand. Built in learned order so each
        # merged id's two halves already exist when we concatenate them.
        self.vocab = {i: bytes([i]) for i in range(256)}
        for pair, idx in self.merges.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        self.vocab_size = vocab_size

    def encode(self, s: str) -> list[int]:
        """str -> list[int], replaying merges in LEARNED order (not by frequency).

        Each pass merges the pair with the lowest merge index = learned earliest;
        the inf default makes never-learned pairs sort last so they're never picked.
        (len >= 2 guarantees stats is non-empty, so min() is safe.)
        """
        ids = list(s.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # no remaining pair is mergeable
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        # errors="replace": ids can form invalid UTF-8 mid-character (e.g. a model
        # mid-generation); replace emits U+FFFD instead of crashing.
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    # --- persistence: training is slow (~minutes on 1MB), so cache the result ---
    def save(self, path: str) -> None:
        # only the merges (in learned order) + vocab_size need saving; vocab is
        # derived from them. JSON-friendly: tuple keys -> [a, b] pairs.
        import json
        with open(path, "w") as f:
            json.dump({"vocab_size": self.vocab_size,
                       "merges": [[a, b] for (a, b) in self.merges]}, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        import json
        with open(path) as f:
            obj = json.load(f)
        tok = cls()
        tok.vocab_size = obj["vocab_size"]
        # file order IS learned order, so ids count up 256, 257, ... again
        tok.merges = {(a, b): 256 + i for i, (a, b) in enumerate(obj["merges"])}
        tok.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in tok.merges.items():
            tok.vocab[idx] = tok.vocab[a] + tok.vocab[b]
        return tok


# ---------------------------------------------------------------------------
# Self-check: run `python llm/bpe.py` from the repo root.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os

    # --- unit-test the two helpers in isolation ---
    assert get_stats([1, 2, 1, 2, 3]) == {(1, 2): 2, (2, 1): 1, (2, 3): 1}, "get_stats wrong"
    assert merge([1, 2, 1, 2, 3], (1, 2), 99) == [99, 99, 3], "merge wrong"
    assert merge([1, 1, 1], (1, 1), 99) == [99, 1], "merge overlap case wrong"
    print("helpers ok ✅")

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "input.txt"), "r", encoding="utf-8") as f:
        text = f.read()

    # train on a slice so the self-check is fast
    train_text = text[:20000]
    tok = BPETokenizer()
    tok.train(train_text, vocab_size=512, verbose=True)

    assert tok.vocab_size == 512
    assert len(tok.merges) == 512 - 256, f"expected 256 merges, got {len(tok.merges)}"

    # round-trip on text it trained on...
    assert tok.decode(tok.encode(train_text)) == train_text, "round-trip on train text failed"
    # ...and on text it never saw, including a non-ASCII char (byte-level wins here)
    novel = "Hello, world! 你好 🌍 — never seen this."
    assert tok.decode(tok.encode(novel)) == novel, "round-trip on novel text failed"
    print("round-trip ok ✅")

    # BPE should COMPRESS vs raw bytes: fewer tokens than UTF-8 bytes.
    raw = len(train_text.encode("utf-8"))
    enc = len(tok.encode(train_text))
    ratio = raw / enc
    assert enc < raw, "BPE did not compress"
    print(f"compression: {raw} bytes -> {enc} tokens ({ratio:.2f}x)")

    # all encoded ids must be valid vocab entries
    ids = tok.encode(novel)
    assert all(0 <= i < tok.vocab_size for i in ids), "ids out of vocab range"

    # save -> load round-trips: reloaded tokenizer encodes identically
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    tok.save(tmp)
    tok2 = BPETokenizer.load(tmp)
    os.remove(tmp)
    assert tok2.merges == tok.merges and tok2.vocab == tok.vocab, "save/load mismatch"
    assert tok2.encode(novel) == ids, "reloaded tokenizer encodes differently"
    print("save/load ok ✅")

    print("all checks passed ✅")
