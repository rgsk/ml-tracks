"""
Fine-tuning Exercise 1 (SFT / instruction tuning) — data + LOSS MASKING.

A *base* model only predicts the next token; it doesn't "follow instructions."
Supervised fine-tuning just continues next-token training, but on curated
(instruction, good-answer) pairs formatted with a fixed template. That alone
turns a raw LM into something that responds in the expected shape.

The one mechanic that makes SFT *SFT* (and the thing interviewers probe) is
LOSS MASKING: we concatenate  prompt + response  into one sequence, but we only
want the model to be graded on producing the RESPONSE. We do NOT want to train
it to generate the instruction — the instruction is given to it at inference.
So we set the label of every PROMPT position to IGNORE_INDEX (-100), which
`F.cross_entropy` skips (its default `ignore_index` is already -100, so the
existing model.py loss "just works" once we feed masked labels).

Toy task (self-contained on our pretrained Shakespeare base): reverse a short
string.  prompt "reverse: hello\n"   response "olleh<EOS>".  The base has never
seen this, so SFT teaches the *format*.

Full pipeline, reusing train.py's machinery (optimizer / LR schedule / grad_accum
/ checkpoint) — only the base ckpt, the (masked) data, and the small LR differ:
  build_example / collate   — the DATA half (tokenize + mask + pad)
  sft_batch / evaluate / generate_answer — shift, masked-loss eval, greedy decode
  finetune                  — the loop; save + sample + accuracy

Usage:
  python sft.py --check              # data-half self-check only
  python sft.py                      # fine-tune the char base (default; reversal is
                                     #   easy char-level — no BPE re-chunking, ~100%)
  python sft.py --tokenizer bpe      # fine-tune the BPE base instead (~64%)
"""

from __future__ import annotations

import os
import random

import torch

from bpe import BPETokenizer
from checkpoint import load_checkpoint, save_checkpoint
from config import GPTConfig
from data import DATA_PATH
from grad_accum import grad_accum_step
from lora import apply_lora, merge_lora
from lr_schedule import get_lr
from model import GPT
from optimizer import configure_optimizers
from tokenizer import CharTokenizer

# --- special ids ---------------------------------------------------------
# cross_entropy skips target positions equal to this (torch's default too).
IGNORE_INDEX = -100

# id 0 is our END-OF-SEQUENCE *and* PAD for BOTH tokenizers — and it's a safe stop
# token either way because our answers are pure lowercase letters:
#   BPE:  id 0 = the NUL byte  (absent from printable text)
#   char: id 0 = '\n'          (absent from a letters-only answer)
# One id for EOS and PAD is fine: PAD positions get label IGNORE_INDEX (never
# predicted), and right-padding + causal attention keeps PAD out of real tokens.
EOS_ID = 0
PAD_ID = 0

# The PROMPT is everything through the trailing newline; the answer follows it.
# NOTE: we deliberately avoid '>' (the old "->" marker) — it's not in the char
# tokenizer's vocab (tinyshakespeare has no '>'), so a newline is the marker that
# works for BOTH tokenizers.
PROMPT_TEMPLATE = "reverse: {inp}\n"

# --- paths + SFT hyperparameters -----------------------------------------
# SFT = continued training FROM the pretrained base, so we point at its ckpt and
# save the fine-tuned result alongside. The LR is much smaller than pretraining's
# (3e-4): the base already holds useful features and we don't want to blow them
# away ("catastrophic forgetting"). Everything else reuses the train.py machinery.
_HERE = os.path.dirname(os.path.abspath(__file__))
_CKPTS = os.path.join(_HERE, "artifacts", "checkpoints")
BPE_TOK_PATH = os.path.join(_HERE, "artifacts", "tokenizer", "bpe_v1024.json")
# per-tokenizer pretrained bases (both trained by train.py) + where SFT saves.
BASE_CKPT = {"bpe": os.path.join(_CKPTS, "bpe1024", "best.pt"),
             "char": os.path.join(_CKPTS, "char", "best.pt")}
OUT_CKPT = {"bpe": os.path.join(_CKPTS, "sft_reverse_bpe.pt"),
            "char": os.path.join(_CKPTS, "sft_reverse_char.pt")}

SFT_STEPS = 2000        # 500 badly underfits this task (val loss still plunging); ~14s
SFT_LR = 5e-4            # << pretraining's 3e-4-with-decay peak; SFT is gentle
SFT_BATCH_SIZE = 32


# --- task / template (given — this is just plumbing) ---------------------
def format_reverse(inp: str) -> tuple[str, str]:
    """Turn a raw input string into (prompt_str, response_str) for the task."""
    return PROMPT_TEMPLATE.format(inp=inp), inp[::-1]


def make_reverse_dataset(n: int, min_len: int = 1, max_len: int = 8,
                         seed: int = 0) -> list[tuple[str, str]]:
    """n random (prompt_str, response_str) pairs of the reverse task."""
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    out = []
    for _ in range(n):
        length = rng.randint(min_len, max_len)
        inp = "".join(rng.choice(alphabet) for _ in range(length))
        out.append(format_reverse(inp))
    return out


# --- the SFT mechanics (TODO) --------------------------------------------
def build_example(prompt_str: str, response_str: str,
                  tok: BPETokenizer) -> tuple[list[int], list[int]]:
    """Build one SFT training example: (input_ids, labels).

    Returns two equal-length python int lists:
      input_ids : the full  prompt + response + EOS  token sequence.
      labels    : a copy of input_ids, EXCEPT every PROMPT position is set to
                  IGNORE_INDEX so loss lands only on the response (+ EOS).

    KEY subtlety — tokenize the prompt and the response SEPARATELY, then
    concatenate the two id lists. That way `len(prompt_ids)` gives you the
    exact mask boundary. (If you encoded the joined string in one call, BPE
    could merge a pair straddling the prompt/response seam and the boundary
    would become ambiguous.)
    """
    # tokenize the two halves SEPARATELY so len(prompt_ids) is the exact boundary
    prompt_ids = tok.encode(prompt_str)
    response_ids = tok.encode(response_str) + [EOS_ID]   # EOS teaches it to STOP
    input_ids = prompt_ids + response_ids
    labels = list(input_ids)                             # copy (a plain = would alias)
    for i in range(len(prompt_ids)):                     # mask every PROMPT position
        labels[i] = IGNORE_INDEX
    return input_ids, labels


def collate(batch: list[tuple[list[int], list[int]]],
            pad_id: int = PAD_ID,
            ignore_index: int = IGNORE_INDEX) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of (input_ids, labels) examples into two (B, T) tensors.

    Examples have different lengths, so pad every row on the RIGHT to the
    batch's max length:
      - pad input_ids with `pad_id`      (a real, valid token id)
      - pad labels    with `ignore_index` (so pad positions contribute NO loss)
    Return (input_ids, labels) as LongTensors of shape (B, T_max).

    Why right-padding is safe here (no attention mask needed): attention is
    causal, so a real token at position t only ever attends to positions <= t,
    which are all real. PAD sits to the right and can never leak backward.
    """
    T = max(len(ids) for ids, _ in batch)
    xs, ys = [], []
    for ids, lab in batch:
        pad = T - len(ids)
        xs.append(ids + [pad_id] * pad)            # inputs padded with a real id
        ys.append(lab + [ignore_index] * pad)      # labels padded with -100
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


# ===========================================================================
# Training pipeline. SFT reuses the SAME machinery as pretraining (train.py):
# configure_optimizers, get_lr, grad_accum_step, save/load_checkpoint. Only three
# things change — init from the base ckpt, feed masked SFT batches, use a small LR.
# ===========================================================================
def make_words(n: int, min_len: int = 1, max_len: int = 8, seed: int = 0) -> list[str]:
    """n random lowercase words of length [min_len, max_len]. (given)"""
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(min_len, max_len)))
            for _ in range(n)]


def load_tokenizer(tokenizer: str):
    """char = CharTokenizer built from the corpus; bpe = the cached BPE merges."""
    if tokenizer == "bpe":
        return BPETokenizer.load(BPE_TOK_PATH)
    if tokenizer == "char":
        return CharTokenizer(open(DATA_PATH, encoding="utf-8").read())
    raise ValueError(f"unknown tokenizer {tokenizer!r}")


def load_base(tokenizer: str, device: str = "cpu") -> tuple[GPT, object]:
    """Rebuild the pretrained base for `tokenizer` (char/bpe) from its checkpoint,
    load its weights, and pair it with the matching tokenizer. SFT starts here."""
    ckpt = torch.load(BASE_CKPT[tokenizer], map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model, load_tokenizer(tokenizer)


def sft_batch(examples, device: str):
    """collate a list of (input_ids, labels), then shift to (x, y) for the loss:
    x = ids[:, :-1], y = labels[:, 1:] (get_batch's convention). .contiguous()
    because model.forward does targets.view()."""
    x, y = collate(examples)
    return x[:, :-1].contiguous().to(device), y[:, 1:].contiguous().to(device)


@torch.no_grad()
def evaluate(model, val_examples, device: str, chunk: int = 64) -> float:
    """Mean masked val loss over val_examples (cross_entropy skips the -100 labels)."""
    model.eval()
    chunks = [val_examples[k:k + chunk] for k in range(0, len(val_examples), chunk)]
    return sum(model(*sft_batch(slice, device))[1].item() for slice in chunks) / len(chunks)


@torch.no_grad()
def generate_answer(model, tok, word: str, device: str, max_new_tokens: int = 14) -> str:
    """Greedily generate the model's answer for one word, stopping at the first EOS."""
    prompt = format_reverse(word)[0]
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.0)
    new = out[0, ids.shape[1]:].tolist()
    if EOS_ID in new:
        new = new[:new.index(EOS_ID)]          # stop at the learned EOS
    return tok.decode(new)


@torch.no_grad()
def accuracy(model, tok, words, device: str) -> float:
    """Exact-match reverse accuracy over `words`. (given — uses generate_answer)"""
    model.eval()
    return sum(generate_answer(model, tok, w, device) == w[::-1] for w in words) / len(words)


@torch.no_grad()
def print_samples(model, tok, words, device: str):
    """Print a want-vs-generated table for a few held-out words. (given)"""
    model.eval()
    print("\n  word     | want     | generated | ok?")
    print("  ---------+----------+-----------+----")
    for w in words:
        got = generate_answer(model, tok, w, device).replace("\n", "\\n")
        ok = "YES" if got == w[::-1] else "no"
        print(f"  {w:7s}  | {w[::-1]:7s} | {got[:9]:9s} | {ok}")


def finetune(tokenizer: str = "char", steps: int = SFT_STEPS, lr: float = SFT_LR,
             batch_size: int = SFT_BATCH_SIZE, grad_accum_steps: int = 1,
             warmup: int = 50, eval_interval: int = 100, seed: int = 0,
             lora: bool = False, rank: int = 8):
    """Fine-tune the base model on the reverse task and save it. The loop body is
    train.py's, verbatim in spirit — only the base, the data, and the LR differ.

    lora=True freezes the base and trains low-rank adapters instead of all weights
    (see lora.py). Same data/loss/loop — configure_optimizers already filters to
    requires_grad params, so the optimizer picks up only the adapters. We merge the
    adapters back into the weights before saving, so the checkpoint is a plain GPT.
    """
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_base(tokenizer, device)              # <-- init from the BASE
    cfg = model.config

    if lora:
        apply_lora(model, r=rank)                          # freeze base, inject adapters
        model.to(device)                                   # adapters were created on CPU
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"  LoRA (r={rank}): training {n_train:,}/{n_all:,} params ({n_train/n_all:.1%})")

    # data: masked (prompt, answer) pairs. val words are a disjoint seed from train,
    # so val loss / accuracy measure generalization, not memorization.
    train_ex = [build_example(*format_reverse(w), tok) for w in make_words(4000, seed=1)]
    val_ex = [build_example(*format_reverse(w), tok) for w in make_words(400, seed=2)]

    # the SAME optimizer setup as pretraining, just a smaller LR (no weight decay —
    # SFT runs are short). grad_accum_step handles autocast+backward+clip+step.
    optimizer = configure_optimizers(model, weight_decay=0.0, learning_rate=lr)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    amp_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    scaler = torch.amp.GradScaler(device_type, enabled=False)   # bf16/fp32 -> no scaler
    rng = random.Random(seed)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"SFT [{tokenizer}] on {device} | vocab {tok.vocab_size} | {n_params/1e6:.2f}M params | "
          f"{len(train_ex)} train / {len(val_ex)} val examples | lr {lr:g}")
    print(f"  base val loss (before SFT): {evaluate(model, val_ex, device):.3f}")

    for step in range(steps):
        # short warmup then cosine decay to lr/10 — reuse the pretraining schedule
        lr_now = get_lr(step, warmup_steps=warmup, max_steps=steps, max_lr=lr, min_lr=lr / 10)
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        if step % eval_interval == 0:
            print(f"  step {step:4d} | lr {lr_now:.2e} | val loss {evaluate(model, val_ex, device):.3f}")
            model.train()

        # a fresh random batch (optionally several micro-batches) of masked SFT
        # examples, fed straight into the pretraining step function.
        micro = [sft_batch([train_ex[rng.randrange(len(train_ex))] for _ in range(batch_size)],
                           device)
                 for _ in range(grad_accum_steps)]
        grad_accum_step(model, micro, optimizer, scaler, amp_dtype=amp_dtype,
                        device_type=device_type, grad_clip=1.0)

    val = evaluate(model, val_ex, device)
    if lora:
        merge_lora(model)                # fold adapters into the weights -> plain GPT
    out_ckpt = OUT_CKPT[tokenizer].replace(".pt", "_lora.pt" if lora else ".pt")
    # after a merge the saved optimizer state refers to the (now-gone) adapters, so
    # it's dead weight — fine, a merged LoRA checkpoint is for inference, not resume.
    save_checkpoint(out_ckpt, model, optimizer, steps, cfg, task="reverse", val_loss=val)

    test_words = make_words(200, seed=3)                 # held out from train/val
    print_samples(model, tok, test_words[:12], device)   # eyeball a dozen
    acc = accuracy(model, tok, test_words, device)        # score all 200
    tag = f"{tokenizer}{'+lora' if lora else ''}"
    print(f"\ndone [{tag}] | final val loss {val:.3f} | held-out reverse accuracy {acc:.1%}")
    print(f"saved -> {os.path.relpath(out_ckpt, _HERE)}")


# ---------------------------------------------------------------------------
# Self-check for the DATA half (build_example + collate). Run `python sft.py --check`.
# ---------------------------------------------------------------------------
def _data_self_check():
    tok = BPETokenizer.load(BPE_TOK_PATH)

    # ---- build_example -------------------------------------------------
    prompt_str, response_str = format_reverse("hello")
    input_ids, labels = build_example(prompt_str, response_str, tok)

    prompt_ids = tok.encode(prompt_str)
    response_ids = tok.encode(response_str) + [EOS_ID]
    n_prompt, n_resp = len(prompt_ids), len(response_ids)

    assert len(input_ids) == len(labels), "input_ids and labels must be equal length"
    assert input_ids == prompt_ids + response_ids, "input_ids must be prompt+response+EOS"
    assert input_ids[-1] == EOS_ID, "sequence must end with EOS (model learns to stop)"

    # prompt positions masked, response positions kept
    assert all(l == IGNORE_INDEX for l in labels[:n_prompt]), "prompt labels must be masked"
    assert labels[n_prompt:] == input_ids[n_prompt:], "response labels must be unmasked = input"
    n_unmasked = sum(1 for l in labels if l != IGNORE_INDEX)
    assert n_unmasked == n_resp, f"exactly {n_resp} response labels should train, got {n_unmasked}"

    print(f"one example — prompt {n_prompt} tok (masked), response {n_resp} tok (trained):")
    print(f"  prompt   : {prompt_str!r}")
    print(f"  response : {response_str + '<EOS>'!r}")
    print(f"  labels   : {labels}   (-100 = ignored)")

    # ---- collate -------------------------------------------------------
    examples = [build_example(*format_reverse(s), tok) for s in ("ab", "abcdef", "x")]
    x, y = collate(examples)
    T = x.shape[1]
    assert x.shape == y.shape == (3, T), f"bad batch shapes {x.shape} {y.shape}"
    assert x.dtype == torch.long and y.dtype == torch.long, "tensors must be long"
    assert T == max(len(ids) for ids, _ in examples), "T must be the max row length"

    for r, (ids, lab) in enumerate(examples):
        n = len(ids)
        assert x[r, :n].tolist() == ids, f"row {r} real input corrupted"
        assert y[r, :n].tolist() == lab, f"row {r} real labels corrupted"
        assert (x[r, n:] == PAD_ID).all(), f"row {r} input pad must be PAD_ID"
        assert (y[r, n:] == IGNORE_INDEX).all(), f"row {r} label pad must be IGNORE_INDEX"

    print(f"\ncollated batch x,y shape = {tuple(x.shape)}; pad safely masked ✅")
    print("\nall checks passed ✅")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SFT the base model on the reverse task")
    parser.add_argument("--check", action="store_true",
                        help="run the data-half self-check (build_example + collate) and exit")
    parser.add_argument("--tokenizer", choices=["bpe", "char"], default="char",
                        help="which pretrained base + tokenizer to fine-tune")
    parser.add_argument("--lora", action="store_true",
                        help="freeze the base and train low-rank adapters (lora.py) instead "
                             "of all weights; merged into the weights before saving")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank r (with --lora)")
    parser.add_argument("--steps", type=int, default=SFT_STEPS)
    parser.add_argument("--lr", type=float, default=None,
                        help="learning rate; default 5e-4 (full) or 3e-3 (--lora: adapters "
                             "start at 0 and train fewer params, so they want a bigger step)")
    parser.add_argument("--batch-size", type=int, default=SFT_BATCH_SIZE)
    args = parser.parse_args()

    if args.check:
        _data_self_check()
    else:
        lr = args.lr if args.lr is not None else (3e-3 if args.lora else SFT_LR)
        finetune(tokenizer=args.tokenizer, steps=args.steps, lr=lr,
                 batch_size=args.batch_size, lora=args.lora, rank=args.rank)
