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
# LLM · 02 — what is a token?

`01` fed the model a **character** tokenizer: every distinct character got its own
id (a vocab of 65). It round-trips perfectly and needs almost no code — but it is
the *most expensive* way to feed text to a transformer. Every character is a
separate token, so sequences are long, and each token carries almost no meaning on
its own (a lone `"t"` tells the model very little). The model burns context length
and capacity just re-learning how words are spelled.

This notebook opens that box. Real LLMs use **BPE** (byte-pair encoding): start from
raw bytes, then repeatedly glue the single most frequent adjacent pair into a new
token. Frequent chunks — `" the"`, `"ing"`, `"\n\n"` — collapse into one token each,
so the *same* text becomes a **shorter** sequence of **more meaningful** tokens, and
it never chokes on input it has never seen.

We build BPE from scratch and watch two things happen, both as encode/decode
round-trips that never lose a byte:

1. the sequence **compresses** (fewer tokens than characters), and
2. **subwords emerge** on their own — nobody hands the tokenizer a word list.
"""

# %%
import pickle
from pathlib import Path


def _repo_root() -> Path:
    """Walk up from the cwd to the folder holding pyproject.toml, so paths work no
    matter where the kernel is launched from."""
    here = Path.cwd()
    for d in (here, *here.parents):
        if (d / "pyproject.toml").exists():
            return d
    return here


ROOT = _repo_root()
DATA = ROOT / "nb" / "llm" / "data" / "input.txt"  # tinyshakespeare, shared with 01
text = DATA.read_text()

# %% [markdown]
"""
## Recall the cost — one id per character

Rebuild `01`'s character tokenizer in two lines and look at what it does to a short
sentence. The vocab is tiny (65), but *every character is a token*: a 41-character
line is 41 tokens. That is the length the transformer has to pay attention over.
"""

# %%
chars = sorted(set(text))                                   # the 65-symbol vocab
char_stoi = {c: i for i, c in enumerate(chars)}
char_encode = lambda s: [char_stoi[c] for c in s]           # str -> list[int]

sample_line = "To be, or not to be, that is the question"
print(f"corpus: {len(text):,} characters, char-vocab of {len(chars)}")
print(f"char tokenizer: {sample_line!r}")
print(f"  -> {len(char_encode(sample_line))} tokens (one per character)")

# %% [markdown]
"""
## Start from bytes, not characters

Before merging anything, we need a base alphabet. BPE uses **raw UTF-8 bytes**: any
string encodes to bytes, and there are exactly **256** possible byte values, so the
base vocab is a fixed 256 — no matter what text you throw at it.

This is what makes BPE robust: `decode(encode(x)) == x` for *any* string, including
emoji or characters that never appeared in training. A character tokenizer would
`KeyError` on an unseen character; a byte tokenizer cannot, because every byte
0–255 is already in the vocab.

(tinyshakespeare is pure ASCII, so here one character happens to be exactly one
byte. The win of bytes is the guarantee, not the count — it shows up the moment you
feed it anything non-ASCII.)
"""

# %%
raw = list(sample_line.encode("utf-8"))                     # str -> list of 0..255
print(f"first 16 bytes: {raw[:16]}")
print(f"byte vocab: 256 symbols, always round-trips -> {bytes(raw).decode('utf-8')!r}")
print(f"({len(raw)} bytes == {len(sample_line)} chars here, since ASCII)")

# %% [markdown]
"""
## The two helpers that do all the work

BPE is just two tiny operations applied in a loop:

- **`get_stats`** — count how often each adjacent pair of ids appears.
- **`merge`** — replace every occurrence of one chosen pair with a single new id.

The `i += 2` on a hit in `merge` is what handles overlaps correctly: merging `(1,1)`
in `[1,1,1]` gives `[new,1]`, not `[new,new]` — the middle element is consumed by the
first merge and can't be reused. Here's one merge step on a toy string.
"""


# %%
def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count how often each adjacent pair (ids[i], ids[i+1]) occurs."""
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], idx: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with the single new id `idx`."""
    out: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


# one step of BPE on a toy string, so the mechanics are visible
demo = list("aaabdaaabac".encode("utf-8"))
top = max(get_stats(demo), key=get_stats(demo).get)
print(f"demo ids : {demo}")
print(f"top pair : {top}  (chars {bytes(top).decode()!r}) -> mint id 256")
print(f"after    : {merge(demo, top, 256)}   (sequence got shorter)")

# %% [markdown]
"""
## Train BPE on the whole corpus

Now the loop: from the byte ids, find the most frequent pair, merge it into a new id
(`256`, then `257`, …), and repeat. Each pass shortens the sequence and grows the
vocab by one. We do `VOCAB_SIZE - 256` merges — here `512 - 256 = 256` of them.

`merges` records the pairs **in the order they were learned**, and that order *is*
the id numbering (`256, 257, …`). encode() below has to replay merges in exactly
this order, so a merge only fires after the earlier merges it is built on (e.g.
`t`+`h` → `th` must happen before `th`+`e` → `the`).

(This runs the full ~1M-char corpus through 256 merges — a slow Python loop, tens of
seconds. Production tokenizers do this once, offline, then cache the result, which is
exactly what we do at the end.)
"""


# %%
VOCAB_SIZE = 512


def train_bpe(text: str, vocab_size: int) -> dict[tuple[int, int], int]:
    """Learn vocab_size - 256 merges by greedily merging the most frequent pair."""
    ids = list(text.encode("utf-8"))                        # 0..255
    merges: dict[tuple[int, int], int] = {}
    for i in range(vocab_size - 256):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)                    # most frequent pair
        idx = 256 + i
        ids = merge(ids, pair, idx)
        merges[pair] = idx
    return merges


def build_vocab(merges: dict[tuple[int, int], int]) -> dict[int, bytes]:
    """Map every id back to the bytes it stands for (base bytes + merged pieces)."""
    vocab = {i: bytes([i]) for i in range(256)}
    for (a, b), idx in merges.items():                  # learned order: halves exist
        vocab[idx] = vocab[a] + vocab[b]
    return vocab


merges = train_bpe(text, VOCAB_SIZE)
vocab = build_vocab(merges)
print(f"learned {len(merges)} merges -> vocab of {len(vocab)}\n")

print("first 12 merges (the most frequent pairs in the corpus):")
for (a, b), idx in list(merges.items())[:12]:
    print(f"  {idx}: {vocab[a]!r} + {vocab[b]!r} -> {vocab[idx]!r}")

# %% [markdown]
"""
## Subwords emerge on their own

Nobody told BPE what a word is — it only ever counted adjacent pairs. Yet the tokens
it built up are recognizable English chunks: common words, suffixes, and word +
leading-space pieces (the space matters — `" the"` and `"the"` are different tokens,
which is why models are picky about spacing). Here are the longest tokens it learned.
"""

# %%
longest = sorted(vocab.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]
print("longest learned tokens (whole subwords emerge):")
for idx, b in longest:
    print(f"  {idx}: {b.decode('utf-8', errors='replace')!r}")

# %% [markdown]
"""
## encode / decode

**decode** is trivial: look each id up in `vocab` and concatenate the bytes.
(`errors="replace"` because a *model* mid-generation can emit ids that don't form
valid UTF-8 at a character boundary — we want U+FFFD, not a crash.)

**encode** replays the learned merges on new text. Each pass merges the pair with the
**lowest merge id** — i.e. the one learned *earliest* — using `float("inf")` as the
key for pairs that were never learned, so they sort last and are never chosen. When
no remaining pair is in `merges`, we stop. Replaying in learned order is what keeps
encode consistent with training.
"""


# %%
def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    return b"".join(vocab[i] for i in ids).decode("utf-8", errors="replace")


def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break                                           # nothing left to merge
        ids = merge(ids, pair, merges[pair])
    return ids


# round-trip on a 200k slice (encode is O(merges·len), so a slice keeps it quick)
slice_ = text[:200_000]
enc = encode(slice_, merges)
assert decode(enc, vocab) == slice_, "round-trip lost information!"
print("round-trip on 200k chars: exact ✓")
print(f"  {slice_[:41]!r}")
print(f"  -> {len(encode(sample_line, merges))} BPE tokens "
      f"(vs {len(sample_line)} chars)")

# %% [markdown]
"""
## The payoff — compression

Same text, three tokenizers. Char and byte are ~1 token per character; BPE at a vocab
of just 512 already roughly **halves** the sequence. Bigger vocabs compress more:
production tokenizers (GPT-2's 50k, Llama's 128k) reach ~4 characters per token.

Fewer tokens for the same text is a direct win everywhere it matters — the attention
cost is O(T²) in sequence length T, so halving T quarters the attention compute, and
a fixed context window (`BLOCK`) now holds twice as much actual text.
"""

# %%
BLOCK = 64
n_char = len(char_encode(slice_))
n_byte = len(slice_.encode("utf-8"))
n_bpe = len(enc)

print(f"{'tokenizer':<12}{'tokens':>12}{'chars/token':>14}")
for name, n in [("char", n_char), ("byte", n_byte), ("bpe-512", n_bpe)]:
    print(f"{name:<12}{n:>12,}{len(slice_) / n:>14.2f}")

print(f"\ncompression vs chars: {n_char / n_bpe:.2f}x fewer tokens")
print(f"effective context at BLOCK={BLOCK}: char {BLOCK} chars  "
      f"vs  bpe ~{round(BLOCK * len(slice_) / n_bpe)} chars")

# %% [markdown]
"""
## Cache the tokenizer

Training BPE is slow, but the result is small — just the `merges` table (the vocab is
derived from it). Save it so later notebooks load the tokenizer instead of retraining,
the same way `01` cached the model weights. Tuple keys pickle fine, so this is a
one-liner.
"""

# %%
TOK_DIR = ROOT / "nb" / "llm" / "artifacts" / "tokenizer"
TOK_DIR.mkdir(parents=True, exist_ok=True)
tok_path = TOK_DIR / "bpe.pkl"
with open(tok_path, "wb") as f:
    pickle.dump({"merges": merges, "vocab_size": VOCAB_SIZE}, f)
print(f"saved -> {tok_path.relative_to(ROOT)}  ({len(merges)} merges)")

# %% [markdown]
"""
That's the input box opened: text → bytes → learned subword tokens, compressing the
sequence with zero out-of-vocab and no lost bytes. Next, `03` follows those token ids
one step further — into the **embedding table** that turns each id into a vector, and
the bigram baseline that is the floor `01` had to beat.
"""
