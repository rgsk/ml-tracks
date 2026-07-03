"""
WALKTHROUGH: Supervised fine-tuning (SFT / instruction tuning), one layer at a time.

A *base* LM only does one thing: predict the next token of whatever text you give
it. It has no notion of "instruction" vs "answer" — feed it "reverse: hello\\n->"
and it just continues the string in the style of its training data (Shakespeare,
here). SFT fixes that WITHOUT changing the architecture or the loss function: we
just keep doing next-token training, but on curated (instruction, good-answer)
pairs, and we mask the loss so the model is graded ONLY on producing the answer.
That single change turns a raw LM into something that follows a format.

The whole point interviewers probe is LOSS MASKING, so we build up to it slowly
and then watch a real base model learn our toy task (reverse a short string).

Layers (run each `exp_*`, watch the output, then say "next"):
  1. the base model CAN'T follow instructions — see it ignore the task.   (here)
  2. one SFT example — the chat template + tokenize (prompt+response+EOS).
  3. LOSS MASKING — labels with the prompt set to -100, and WHY.
  4. aligning labels with the loss — the shift x=ids[:-1], y=labels[1:].
  5. batching — collate + right-padding (why it's safe under causal attention).
  6. one SFT gradient step — masked loss on a batch, backward, step.
  7. the fine-tune loop — train on the reverse task, watch the loss fall.
  8. before vs after — the payoff: the fine-tuned model reverses held-out strings.
  9. tokenization decides difficulty — char (100%) vs BPE (~80%) on the SAME task.

This is the SCRATCH file. The real exercise (sft.py) stays untouched so you can
rebuild it yourself afterwards.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import random
import sys

# this file lives in llm/walkthroughs/, but its building blocks and artifacts are
# in the parent llm/ package dir — put that on the import path first.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LLM = os.path.dirname(_HERE)
sys.path.insert(0, _LLM)

import torch

from bpe import BPETokenizer
from checkpoint import load_checkpoint, save_checkpoint
from config import GPTConfig
from data import DATA_PATH
from model import GPT
from tokenizer import CharTokenizer

# artifacts (checkpoints, tokenizer cache) live under llm/, not next to this file
_CKPT = os.path.join(_LLM, "artifacts", "checkpoints", "bpe1024", "best.pt")
_CHAR_CKPT = os.path.join(_LLM, "artifacts", "checkpoints", "char", "best.pt")
_SFT_CKPT = os.path.join(_LLM, "artifacts", "checkpoints", "sft_reverse.pt")
_TOK = os.path.join(_LLM, "artifacts", "tokenizer", "bpe_v1024.json")

# Byte-level BPE has no reserved special tokens, so we borrow the NUL byte
# (id 0) as END-OF-SEQUENCE. It never appears in printable toy data, so it's a
# clean single stop token the model can learn to emit to say "I'm done."
EOS_ID = 0
# PAD fills the short rows when we stack variable-length examples into one tensor.
# We reuse id 0 (same as EOS): it's safe because pad POSITIONS get label -100 (no
# loss) and right-padding + causal attention keeps them from ever being read by a
# real token. The safety comes from the -100 label, NOT from a unique pad id.
PAD_ID = 0
# cross_entropy skips target positions equal to this (it's torch's default too),
# so setting a label to IGNORE_INDEX means "don't train on this position."
IGNORE_INDEX = -100
# Everything up to and including "->" is the PROMPT; the answer follows.
PROMPT_TEMPLATE = "reverse: {inp}\n->"


def format_reverse(inp: str, template: str = PROMPT_TEMPLATE) -> tuple[str, str]:
    """Raw input string -> (prompt_str, response_str) for the reverse task."""
    return template.format(inp=inp), inp[::-1]


def _banner(*lines):
    print("=" * 70)
    for line in lines:
        print(line)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Shared setup: load the pretrained BASE model + its tokenizer. This is the
# thing SFT starts from — a plain next-token LM trained on tinyshakespeare
# (BPE vocab 1024, block_size 64, ~0.5M params, val loss 4.41). Everything
# below fine-tunes THIS.
# ---------------------------------------------------------------------------
def load_base(tokenizer: str = "bpe"):
    """Rebuild the base GPT (bpe or char) from its checkpoint, load weights, eval
    mode. Returns (model, tokenizer). Layers 1-8 use the default 'bpe'; layer 9
    loads both to compare them."""
    ckpt_path = _CKPT if tokenizer == "bpe" else _CHAR_CKPT
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["config"]))
    load_checkpoint(ckpt_path, model, map_location="cpu")   # copies weights in
    model.eval()
    tok = (BPETokenizer.load(_TOK) if tokenizer == "bpe"
           else CharTokenizer(open(DATA_PATH, encoding="utf-8").read()))
    return model, tok


@torch.no_grad()
def greedy_complete(model, tok, prompt: str, max_new_tokens: int = 24) -> str:
    """Encode `prompt`, greedily (temperature=0 => argmax) extend it, and return
    ONLY the newly generated text (the model's 'answer'), decoded back to a string."""
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)      # (1, T)
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.0)
    new_ids = out[0, ids.shape[1]:].tolist()                        # drop the prompt
    return tok.decode(new_ids)


# ---------------------------------------------------------------------------
# LAYER 1: the base model can't follow instructions.
#
# Before writing a single line of fine-tuning code, let's SEE the problem. We
# hand the base LM our task prompt — "reverse: hello\n->" — and let it complete.
# A base model is a text-continuation engine: it has never been shown that "->"
# means "now emit the reversed string," so it just produces the most likely
# CONTINUATION under its Shakespeare training — not the answer we want. This is
# exactly the gap SFT closes: same weights, same loss, but retrained on
# (instruction, answer) pairs so this completion becomes the reversed string.
# ---------------------------------------------------------------------------
def exp_1_base_cant_follow():
    """Prompt the untouched base model with several reverse-tasks and print what it
    generates. Expect gibberish / Shakespeare-ish continuation, NOT the reversal —
    the motivation for everything that follows."""
    _banner("LAYER 1: the BASE model ignores the instruction (why SFT exists)")
    model, tok = load_base()

    tests = ["hello", "cat", "world", "abc"]
    print("  prompt is 'reverse: <word>\\n->' ; we want <word> reversed.\n")
    print("   word  | want  | base model actually generates")
    print("  -------+-------+--------------------------------------------")
    for w in tests:
        prompt = f"reverse: {w}\n->"
        got = greedy_complete(model, tok, prompt, max_new_tokens=16)
        got_1line = got.replace("\n", "\\n")            # keep the table on one line
        print(f"  {w:5s}  | {w[::-1]:5s} | {got_1line!r}")

    print("\n  The base model does NOT reverse anything — it just continues the text")
    print("  in the style it was pretrained on. It literally has no concept that '->'")
    print("  is a request. Nothing is broken; it was never taught the task.")
    print("\n  SFT will retrain these SAME weights on (prompt, reversed-answer) pairs so")
    print("  that this exact completion becomes the reversal. Next (layer 2): build ONE")
    print("  training example — the chat template + how it tokenizes.")


# ---------------------------------------------------------------------------
# LAYER 2: build ONE training example — chat template + tokenize.
#
# An SFT example is nothing exotic: it's a SINGLE token sequence formed by
# concatenating the prompt tokens and the answer tokens, plus an EOS token so
# the model learns where to STOP:
#
#     input_ids = [ prompt tokens ...........  answer tokens ...  EOS ]
#                 |<------ we show this ----->||<-- model must produce -->|
#
# The "chat template" is just the fixed text scaffolding around the content
# (here: "reverse: {inp}\n->"). Real chat models use fancier templates with
# special role tokens (<|user|>, <|assistant|>), but the idea is identical.
#
# KEY DECISION (answers layer-1 Q3): tokenize the prompt and the answer
# SEPARATELY, then concatenate the id lists. Then len(prompt_ids) is the EXACT
# boundary between "given" and "to-produce" — which layer 3 needs for masking.
# If you tokenized the joined string in one call, BPE could merge a pair that
# straddles the "->answer" seam, and the boundary would no longer be a clean
# token index. We prove that merge actually happens below.
# ---------------------------------------------------------------------------
def exp_2_one_example():
    """Build one (prompt, answer) example for 'hello', tokenize prompt/answer
    separately, append EOS, and print the pieces + the boundary. Also show that
    JOINT tokenization gives a different sequence at the seam (why we split)."""
    _banner("LAYER 2: one SFT example = prompt_ids + answer_ids + EOS")
    tok = BPETokenizer.load(_TOK)

    prompt_str, answer_str = format_reverse("hello")
    prompt_ids = tok.encode(prompt_str)
    answer_ids = tok.encode(answer_str) + [EOS_ID]     # <-- EOS teaches it to stop
    input_ids = prompt_ids + answer_ids

    print(f"  prompt_str = {prompt_str!r}")
    print(f"  answer_str = {answer_str!r}   (+ EOS id {EOS_ID})\n")
    print(f"  prompt_ids ({len(prompt_ids)} tok) = {prompt_ids}")
    print(f"  answer_ids ({len(answer_ids)} tok) = {answer_ids}")
    print(f"  input_ids  ({len(input_ids)} tok) = {input_ids}")
    print(f"  boundary   = len(prompt_ids) = {len(prompt_ids)}  "
          "(everything before this index is 'given')")

    # round-trip each half back to text so the ids are concrete
    print(f"\n  decode(prompt_ids) = {tok.decode(prompt_ids)!r}")
    print(f"  decode(answer_ids[:-1]) = {tok.decode(answer_ids[:-1])!r}   (EOS dropped)")

    # verify by hand: the whole is exactly the two halves stitched together
    assert input_ids == prompt_ids + answer_ids
    print(f"\n  check: len(input) {len(input_ids)} == "
          f"len(prompt) {len(prompt_ids)} + len(answer) {len(answer_ids)}  ✅")

    # WHY separate: joint encoding can differ at the seam. For OUR template it
    # happens not to (prompt ends in "->", which doesn't merge with a letter),
    # but the risk is real — a contrived seam proves it: "th" + "e".
    joint = tok.encode(prompt_str + answer_str)
    separate = tok.encode(prompt_str) + tok.encode(answer_str)
    print(f"\n  our seam:  joint {len(joint)} tok vs separate {len(separate)} tok "
          f"-> differ={joint != separate} (safe here, seam didn't merge)")
    a, b = "th", "e"
    jm, sm = tok.encode(a + b), tok.encode(a) + tok.encode(b)
    print(f"  merging seam {a!r}+{b!r}:  joint={jm}  separate={sm}  -> differ={jm != sm}")
    print("  -> when a merge DOES straddle the seam, joint collapses two tokens into one,")
    print("     so the prompt/answer boundary stops being a clean index. Tokenizing the")
    print("     two halves separately sidesteps it entirely — len(prompt_ids) stays exact.")

    print("\n  Next (layer 3): the actual SFT move — build `labels` from input_ids but")
    print("  mask the prompt positions to -100 so loss trains ONLY the answer + EOS.")


# ---------------------------------------------------------------------------
# LAYER 3: LOSS MASKING — the one move that makes SFT "SFT".
#
# We have input_ids = prompt(10) + answer(4). We're going to train the model
# next-token style on this sequence. But we do NOT want to train it to produce
# the PROMPT: the prompt is what the USER supplies at inference — teaching the
# model to generate "reverse: hello\n->" is wasted capacity at best and teaches
# it to hallucinate instructions at worst. We only want it graded on the ANSWER.
#
# The mechanism: build `labels` as a copy of input_ids, then overwrite every
# PROMPT position with IGNORE_INDEX (-100). cross_entropy ignores those, so the
# gradient only flows from the answer (+ EOS) positions. That's it — that single
# masking step is the entire difference between "continue pretraining" and "SFT."
#
# (Note: labels here are aligned position-for-position with input_ids — the HF
# convention. The fact that predicting token t uses label t+1 is the SHIFT, and
# we handle that carefully in layer 4. Here we only decide WHICH tokens count.)
# ---------------------------------------------------------------------------
def _show_tok(tok, tid: int) -> str:
    """Readable one-token preview, e.g. 309 -> 'rev'. (repr keeps \\n, NUL visible.)"""
    if tid == IGNORE_INDEX:
        return "----"
    return repr(tok.decode([tid]))


def exp_3_loss_masking():
    """Build masked labels for the 'hello' example and lay them next to input_ids,
    token by token, so you can see exactly which positions train. Contrast with the
    UNMASKED version to make the 'why' concrete (unmasked trains the prompt too)."""
    _banner("LAYER 3: loss masking — labels = input_ids, prompt set to -100")
    tok = BPETokenizer.load(_TOK)

    prompt_str, answer_str = format_reverse("hello")
    prompt_ids = tok.encode(prompt_str)
    answer_ids = tok.encode(answer_str) + [EOS_ID]
    input_ids = prompt_ids + answer_ids
    boundary = len(prompt_ids)

    # THE masking step
    labels = list(input_ids)                       # copy, aligned to input_ids
    for i in range(boundary):                      # every PROMPT position ...
        labels[i] = IGNORE_INDEX                   # ... is ignored by the loss

    print(f"  boundary = {boundary}  (positions 0..{boundary-1} are prompt = masked)\n")
    print("  idx | token   | input_id | label | trains?")
    print("  ----+---------+----------+-------+--------")
    for i, (tid, lab) in enumerate(zip(input_ids, labels)):
        trains = "no (prompt)" if lab == IGNORE_INDEX else "YES"
        print(f"  {i:3d} | {_show_tok(tok, tid):7s} | {tid:8d} | {lab:5d} | {trains}")

    n_train = sum(1 for l in labels if l != IGNORE_INDEX)
    print(f"\n  trained positions = {n_train}  (should equal len(answer+EOS) = {len(answer_ids)})")
    assert n_train == len(answer_ids)
    print(f"  and EOS (id {EOS_ID}) is among them -> the model IS trained to stop.  ✅")

    # the 'why', by contrast: no masking would train all 14 positions
    print(f"\n  without masking, labels would train ALL {len(input_ids)} positions —")
    print("  i.e. we'd also teach the model to GENERATE 'reverse: hello\\n->', the very")
    print("  thing the user hands it. Masking spends every gradient on the answer.")

    print("\n  Next (layer 4): line labels up with predictions. Our model computes loss")
    print("  position-aligned (no internal shift), and get_batch pre-shifts y — so we")
    print("  must feed x = ids[:-1], y = labels[1:]. Get this off-by-one wrong and you")
    print("  silently train on the wrong tokens.")


# ---------------------------------------------------------------------------
# LAYER 4: the SHIFT — line targets up with predictions.
#
# A GPT at position t outputs a distribution over the token at position t+1
# (next-token prediction). So the logits AT t must be scored against the token
# AT t+1. Two ways to arrange that off-by-one:
#   - HF style: model.forward shifts internally (logits[:-1] vs labels[1:]).
#   - THIS codebase: the model does NOT shift; the DATA is pre-shifted. Look at
#     get_batch: x = data[i:i+B], y = data[i+1:i+1+B]. y is just x moved one left.
# We must match the codebase's convention, so for our example we do the same:
#     x = input_ids[:-1]     # drop the last token (nothing to predict after EOS)
#     y = labels[1:]         # drop the first label (nothing predicts position 0)
# Then position t of x predicts y[t] = labels[t+1]. Getting this wrong is the
# classic SFT/LM bug: shift twice and you're two tokens off; shift zero times and
# the target at t IS the input at t — the model "predicts" the token it's already
# reading (a trivial copy through causal attention), loss collapses to ~0, and the
# model learns nothing. We'll show that copy explicitly.
# ---------------------------------------------------------------------------
def exp_4_shift():
    """Take the masked 'hello' example, apply x=ids[:-1] / y=labels[1:], and print
    the aligned (input-position -> target) table. Show the pivotal transition where
    the LAST PROMPT token '>' predicts the FIRST answer token, and show why the
    unshifted alignment is a trivial copy."""
    _banner("LAYER 4: shift — x = input_ids[:-1], y = labels[1:] (match get_batch)")
    tok = BPETokenizer.load(_TOK)

    prompt_str, answer_str = format_reverse("hello")
    prompt_ids = tok.encode(prompt_str)
    answer_ids = tok.encode(answer_str) + [EOS_ID]
    input_ids = prompt_ids + answer_ids
    labels = list(input_ids)
    for i in range(len(prompt_ids)):
        labels[i] = IGNORE_INDEX

    x = input_ids[:-1]        # inputs the model reads
    y = labels[1:]            # target for each input position (shifted)

    print(f"  input_ids ({len(input_ids)}) -> x = input_ids[:-1] ({len(x)}),  "
          f"y = labels[1:] ({len(y)})\n")
    print("   t | reads x[t] | predicts y[t] | trains?")
    print("  ---+------------+---------------+--------")
    for t in range(len(x)):
        tgt = _show_tok(tok, y[t])
        trains = "no" if y[t] == IGNORE_INDEX else "YES"
        star = "  <-- '->' predicts first answer token" if (y[t] != IGNORE_INDEX and
               (t == 0 or y[t - 1] == IGNORE_INDEX)) else ""
        print(f"  {t:2d} | {_show_tok(tok, x[t]):10s} | {tgt:13s} | {trains}{star}")

    n_train = sum(1 for v in y if v != IGNORE_INDEX)
    print(f"\n  trained transitions = {n_train}  (still the 4 answer tokens)")
    assert n_train == len(answer_ids)

    # the pivotal transition, spelled out
    pivot = len(prompt_ids) - 1                 # index of '>' in x
    print(f"\n  position {pivot}: model READS {_show_tok(tok, x[pivot])} (the masked '>') and is")
    print(f"  trained to PREDICT {_show_tok(tok, y[pivot])} (first answer token). So '>' being")
    print("  masked as a LABEL didn't stop it being a useful INPUT — condition-on vs grade-on")
    print("  are different roles (your layer-3 Q2).")

    # why NOT shifting is a silent disaster: target == input at every position
    print("\n  if we FORGOT to shift (y = labels, aligned):")
    for t in (10, 11):
        print(f"    t={t}: reads {tok.decode([input_ids[t]])!r}, 'predicts' "
              f"{tok.decode([labels[t]])!r}  -> identical, a trivial copy")
    print("  causal attention lets position t see x[t] itself, so predicting x[t] is free:")
    print("  loss -> ~0, gradients -> ~0, model learns nothing. Always sanity-check the shift.")

    print("\n  Next (layer 5): batching — real training does many examples at once, but")
    print("  they have different lengths, so we PAD. Padding + masking interact; we'll do")
    print("  it so pad tokens contribute zero loss and never corrupt real positions.")


# Consolidate layers 2-4 into one reusable builder now that we understand it.
def build_example(inp: str, tok, template: str = PROMPT_TEMPLATE) -> tuple[list[int], list[int]]:
    """(input_ids, labels) for one reverse-task example, prompt masked to -100.
    labels are aligned to input_ids (the shift happens later, at loss time)."""
    prompt_str, answer_str = format_reverse(inp, template)
    prompt_ids = tok.encode(prompt_str)
    answer_ids = tok.encode(answer_str) + [EOS_ID]
    input_ids = prompt_ids + answer_ids
    labels = list(input_ids)
    for i in range(len(prompt_ids)):
        labels[i] = IGNORE_INDEX
    return input_ids, labels


# ---------------------------------------------------------------------------
# LAYER 5: batching — pad variable-length examples into one (B, T) tensor.
#
# Training runs B examples at once for GPU efficiency, as one (B, T) tensor. But
# our examples have different lengths (short word vs long word). So we RIGHT-PAD
# every row up to the batch's longest length:
#   - inputs pad with PAD_ID   (a real, valid id — reused EOS/0)
#   - labels pad with -100     (so pad positions contribute ZERO loss)
#
# Why right-padding needs NO attention mask here: attention is CAUSAL, so a real
# token at position t only ever attends to positions <= t — all real, because pad
# sits to the RIGHT. Pad can never leak backward into a real token's context. The
# model still computes logits at pad positions, but their -100 labels drop them
# from the loss. (LEFT-padding would break this — pad would sit before real tokens
# and, without a mask, pollute them. Right-pad is the causal-LM default.)
# ---------------------------------------------------------------------------
def collate(batch, pad_id=PAD_ID, ignore_index=IGNORE_INDEX):
    """Right-pad a list of (input_ids, labels) into two (B, T_max) LongTensors."""
    T = max(len(ids) for ids, _ in batch)
    xs, ys = [], []
    for ids, lab in batch:
        pad = T - len(ids)
        xs.append(ids + [pad_id] * pad)            # inputs padded with a real id
        ys.append(lab + [ignore_index] * pad)      # labels padded with -100
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def exp_5_batching():
    """Collate three different-length reverse examples into one padded batch and
    verify: shapes, that the real prefix is intact, and that padding is masked in
    the labels (so it can't contribute to the loss)."""
    _banner("LAYER 5: batching — right-pad to (B, T_max), pad labels with -100")
    tok = BPETokenizer.load(_TOK)

    words = ["ab", "abcdef", "x"]
    batch = [build_example(w, tok) for w in words]
    for w, (ids, _) in zip(words, batch):
        print(f"  {w!r:8s} -> {len(ids)} tokens")

    x, y = collate(batch)
    print(f"\n  collated: x,y shape = {tuple(x.shape)}  (T_max = "
          f"{max(len(ids) for ids,_ in batch)})\n")

    print("  row | real len | input tail (real | PAD)      | label tail (real | -100)")
    print("  ----+----------+------------------------------+-------------------------")
    for r, (ids, lab) in enumerate(batch):
        n = len(ids)
        xt = x[r].tolist()
        yt = y[r].tolist()
        # show the last few real cells then the pad cells
        xi = " ".join(str(v) for v in xt[max(0, n - 2):n]) + " | " + \
             " ".join(str(v) for v in xt[n:])
        yi = " ".join(str(v) for v in yt[max(0, n - 2):n]) + " | " + \
             " ".join(str(v) for v in yt[n:])
        print(f"  {r:3d} | {n:8d} | {xi:28s} | {yi}")

        # verify by hand: real prefix untouched, pad region masked
        assert x[r, :n].tolist() == ids and y[r, :n].tolist() == lab
        assert (x[r, n:] == PAD_ID).all() and (y[r, n:] == IGNORE_INDEX).all()

    print(f"\n  every real prefix preserved; every pad cell has input {PAD_ID} and label -100.")
    print("  So padding rounds out the tensor but contributes nothing to the loss, and")
    print("  (right-pad + causal) never corrupts a real token. No attention mask needed.  ✅")
    print("\n  Note the shift still happens LATER, on the whole batch: x[:, :-1], y[:, 1:].")
    print("  Because pad labels are -100, the real-EOS -> first-PAD transition is masked")
    print("  automatically — we never train 'after EOS, emit PAD'.")

    print("\n  Next (layer 6): feed one padded batch through the base model, compute the")
    print("  MASKED loss, and confirm masking changes the number (vs training everything).")


# ---------------------------------------------------------------------------
# LAYER 6: one masked forward + one gradient step through the BASE model.
#
# Now wire it to the real model. Our GPT.forward already calls
#   F.cross_entropy(logits.view(-1, V), targets.view(-1))
# and cross_entropy's default ignore_index is -100 — so feeding masked labels
# needs ZERO model changes; the -100 positions just drop out of the loss. We:
#   1. collate a batch, shift (x[:, :-1], y[:, 1:]), get the MASKED loss.
#   2. compare to the loss with the PROMPT UNMASKED (only pad ignored) — isolates
#      what masking actually does to the number.
#   3. take ONE AdamW step on the masked loss and watch it drop — proving the whole
#      pipeline (mask -> shift -> forward -> loss) is differentiable and learns.
# ---------------------------------------------------------------------------
def _labels_all(inp: str, tok: BPETokenizer) -> tuple[list[int], list[int]]:
    """Like build_example but WITHOUT masking the prompt — every real token trains
    (only padding will later be ignored). For the masked-vs-unmasked contrast."""
    ids, _ = build_example(inp, tok)
    return ids, list(ids)               # labels = ids, nothing set to -100


def exp_6_masked_loss():
    """Feed one padded batch through the base model. Show masked loss vs prompt-
    unmasked loss (and how many tokens each averages over), then one AdamW step and
    the loss dropping on that same batch."""
    _banner("LAYER 6: masked loss through the base model + one gradient step")
    model, tok = load_base()

    words = ["ab", "cat", "hello", "xy", "dog", "world", "q", "reverse"]
    masked_batch = [build_example(w, tok) for w in words]
    all_batch = [_labels_all(w, tok) for w in words]

    x, y_masked = collate(masked_batch)
    _, y_all = collate(all_batch)
    # the shift, on the whole batch. .contiguous() because model.forward does
    # targets.view(B*T), which rejects the non-contiguous slice a stride view leaves.
    xin = x[:, :-1].contiguous()
    y_masked = y_masked[:, 1:].contiguous()
    y_all = y_all[:, 1:].contiguous()

    n_masked = int((y_masked != IGNORE_INDEX).sum())
    n_all = int((y_all != IGNORE_INDEX).sum())

    model.eval()
    with torch.no_grad():
        loss_masked = model(xin, y_masked)[1].item()
        loss_all = model(xin, y_all)[1].item()

    print(f"  batch: {len(words)} examples, padded to x shape {tuple(xin.shape)}\n")
    print(f"  masked loss (answer tokens only)  = {loss_masked:.3f}   over {n_masked} tokens")
    print(f"  unmasked loss (prompt + answer)   = {loss_all:.3f}   over {n_all} tokens")
    print(f"  -> masking averages the loss over only the {n_masked} tokens we care about,")
    print(f"     not all {n_all}. The number is a per-TOKEN mean, so a long answer")
    print("     contributes more terms than a short one (your layer-3 Q3).")

    # one gradient step on the MASKED loss -> it should drop on this same batch
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    opt.zero_grad(set_to_none=True)
    loss = model(xin, y_masked)[1]
    loss.backward()
    opt.step()

    model.eval()
    with torch.no_grad():
        loss_after = model(xin, y_masked)[1].item()

    print(f"\n  one AdamW step on the masked loss:")
    print(f"    masked loss before = {loss_masked:.3f}")
    print(f"    masked loss after  = {loss_after:.3f}   ({'down' if loss_after < loss_masked else 'up'})")
    print("  The masked loss is differentiable end-to-end; one step already moves it.")
    print("  That's SFT in miniature — repeat over many batches and the model learns the task.")

    print("\n  Next (layer 7): wrap this in a real fine-tune loop over a dataset of reverse")
    print("  examples and watch the loss fall over many steps.")


# ---------------------------------------------------------------------------
# LAYER 7: the fine-tune loop — many masked steps over a dataset.
#
# Layer 6 was one step on one batch. SFT is just that, repeated: sample a batch
# of (instruction, answer) examples, mask + shift + forward + backward + step,
# over and over, until the model reliably produces the answer. We hold out a
# validation set of UNSEEN words and track its masked loss — that (not the train
# loss) is the honest signal that the model learned the TASK, not the batch.
#
# LR note (answers layer-6 Q2): we fine-tune with a smaller LR than pretraining.
# The base already holds useful features; too big a step and it "catastrophically
# forgets" them. Here the task is tiny so we're forgiving, but the instinct — SFT
# LR << pretraining LR — is the real-world rule.
# ---------------------------------------------------------------------------
def make_words(n: int, min_len: int = 1, max_len: int = 8, seed: int = 0) -> list[str]:
    """n random lowercase words of length [min_len, max_len]."""
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return ["".join(rng.choice(alphabet) for _ in range(rng.randint(min_len, max_len)))
            for _ in range(n)]


def _sft_batch(examples):
    """collate a list of (input_ids, labels), then shift to (xin, y) for the loss."""
    x, y = collate(examples)
    return x[:, :-1].contiguous(), y[:, 1:].contiguous()


def exp_7_finetune(steps: int = 500, batch_size: int = 32, lr: float = 5e-4, seed: int = 0):
    """Fine-tune the base model on the reverse task; print train/val masked loss as
    it falls; save the result to _SFT_CKPT for layer 8. Val words are disjoint from
    train (different seed), so val loss measures generalization, not memorization."""
    _banner(f"LAYER 7: fine-tune loop ({steps} steps, bs {batch_size}, lr {lr})")
    model, tok = load_base()

    train_ex = [build_example(w, tok) for w in make_words(4000, seed=1)]
    val_ex = [build_example(w, tok) for w in make_words(400, seed=2)]
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed)

    @torch.no_grad()
    def val_loss():
        model.eval()
        chunks = [val_ex[k:k + 64] for k in range(0, len(val_ex), 64)]
        return sum(model(*_sft_batch(c))[1].item() for c in chunks) / len(chunks)

    print("   step | train loss | val loss")
    print("  ------+------------+---------")
    print(f"  {0:5d} |     —      | {val_loss():.3f}   (base, before any SFT)")
    for step in range(1, steps + 1):
        model.train()
        batch = [train_ex[rng.randrange(len(train_ex))] for _ in range(batch_size)]
        xin, y = _sft_batch(batch)
        opt.zero_grad(set_to_none=True)
        loss = model(xin, y)[1]
        loss.backward()
        opt.step()
        if step == 1 or step % 100 == 0:
            print(f"  {step:5d} |   {loss.item():6.3f}   | {val_loss():.3f}")

    save_checkpoint(_SFT_CKPT, model, opt, step=steps, config=model.config, task="reverse")
    print(f"\n  saved fine-tuned model -> {os.path.relpath(_SFT_CKPT, _HERE)}")
    print("  val loss fell far below the base's — the model learned to predict reversed")
    print("  tokens on WORDS IT NEVER SAW. Loss is a proxy, though; layer 8 checks the")
    print("  real thing: does it actually GENERATE correct reversals?")


# ---------------------------------------------------------------------------
# LAYER 8: before vs after — the payoff.
#
# Loss going down is a proxy. The real question SFT is judged on: does the model
# now DO the task on inputs it never saw? We greedily generate an answer from both
# the base model and the fine-tuned one, stop at EOS, and check it against the true
# reversal — on a fresh held-out word set. Base ~0% (layer 1), SFT should follow
# the format and get many right. Where it MISSES is itself instructive: reversal
# (emit tokens in reverse order) is genuinely hard for a small LM, so SFT teaches
# the FORMAT reliably even when the underlying skill stays imperfect. SFT aligns
# behavior; it can't conjure capability the base doesn't have.
# ---------------------------------------------------------------------------
def load_sft():
    """Load the layer-7 fine-tuned model from _SFT_CKPT."""
    ckpt = torch.load(_SFT_CKPT, map_location="cpu")
    model = GPT(GPTConfig(**ckpt["config"]))
    load_checkpoint(_SFT_CKPT, model, map_location="cpu")
    model.eval()
    return model


@torch.no_grad()
def generate_answer(model, tok, word: str, template: str = PROMPT_TEMPLATE,
                    max_new_tokens: int = 14) -> str:
    """Greedily complete the reverse-prompt, cut at the first EOS, decode to text."""
    prompt = format_reverse(word, template)[0]
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.0)
    new = out[0, ids.shape[1]:].tolist()
    if EOS_ID in new:
        new = new[:new.index(EOS_ID)]          # stop at the learned EOS
    return tok.decode(new)


def exp_8_before_after():
    """Generate reversals from base vs fine-tuned model on held-out words; print a
    sample table and the accuracy of each. This is the base->assistant transition."""
    _banner("LAYER 8: before vs after — does the fine-tuned model REVERSE?")
    tok = BPETokenizer.load(_TOK)
    base = load_base()[0]
    sft = load_sft()

    test_words = make_words(200, max_len=3, seed=3)       # fresh, unseen in train/val
    show = test_words[:12]

    print("   word    | want    | base gen        | sft gen         | sft ok?")
    print("  ---------+---------+-----------------+-----------------+--------")
    for w in show:
        want = w[::-1]
        b = generate_answer(base, tok, w).replace("\n", "\\n")
        s = generate_answer(sft, tok, w)
        ok = "YES" if s == want else "no"
        print(f"  {w:7s}  | {want:7s} | {b[:15]:15s} | {s[:15]:15s} | {ok}")

    def acc(model):
        return sum(generate_answer(model, tok, w) == w[::-1] for w in test_words) / len(test_words)

    base_acc, sft_acc = acc(base), acc(sft)
    print(f"\n  exact-match accuracy on {len(test_words)} unseen words:")
    print(f"    base model      : {base_acc:6.1%}")
    print(f"    fine-tuned (SFT): {sft_acc:6.1%}")
    print("\n  The base can't do it at all; SFT taught it to read '->', emit letters, and")
    print("  STOP (EOS) — the instruction-following FORMAT — on words it never trained on.")
    print("  That gap is exactly what SFT buys: same weights + same next-token loss, but")
    print("  retrained on (prompt, answer) pairs with the prompt masked. That's the whole")
    print("  first step from raw LM to assistant.")


# ---------------------------------------------------------------------------
# LAYER 9: TOKENIZATION decides whether the task is easy or hard.
#
# Layer 8 showed BPE-SFT gets long words wrong (near-misses in the middle). Is
# that a limit of SFT, or of the TOKENIZER? We rerun the identical experiment —
# same 0.5M model, same steps, same LR, same template — changing ONLY the
# tokenizer, and fine-tune both the char base and the BPE base on reverse.
#
# Prediction: char wins big. With char, every token is one character, so the
# reversed answer is EXACTLY the input tokens in reverse order — a clean positional
# COPY the model can nail. With BPE, the answer is re-chunked differently from the
# input ("hello"->[h,ell,o] but "olleh"->[oll,e,h]), so there's no copy to learn,
# only a messy chunk->chunk remap. Same task, but the representation makes it easy
# or hard. (We use the marker "reverse: {inp}\n" — '>' isn't in the char vocab.)
# ---------------------------------------------------------------------------
COMPARE_TEMPLATE = "reverse: {inp}\n"      # no '>' (char vocab lacks it) -> works for both


def exp_9_tokenization_matters(steps: int = 2000, batch_size: int = 32,
                               lr: float = 5e-4, seed: int = 0):
    """Fine-tune the char base and the BPE base on the SAME reverse task and compare
    held-out accuracy. The only difference between the two runs is the tokenizer."""
    _banner("LAYER 9: TOKENIZATION decides difficulty — char vs BPE, same everything")
    test_words = make_words(200, seed=3)             # held out from train/val
    results = {}

    for name in ("char", "bpe"):
        print('----')
        print(f'{name} tokenizer')
        print('----')
        model, tok = load_base(name)
        train_ex = [build_example(w, tok, COMPARE_TEMPLATE) for w in make_words(4000, seed=1)]
        val_ex = [build_example(w, tok, COMPARE_TEMPLATE) for w in make_words(400, seed=2)]
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        rng = random.Random(seed)
        model.train()
        @torch.no_grad()
        def val_loss():
            model.eval()
            chunks = [val_ex[k:k + 64] for k in range(0, len(val_ex), 64)]
            return sum(model(*_sft_batch(c))[1].item() for c in chunks) / len(chunks)
        print("   step | train loss | val loss")
        print("  ------+------------+---------")
        print(f"  {0:5d} |     —      | {val_loss():.3f}   (base, before any SFT)")
        for step in range(steps):
            batch = [train_ex[rng.randrange(len(train_ex))] for _ in range(batch_size)]
            xin, y = _sft_batch(batch)
            opt.zero_grad(set_to_none=True)
            loss = model(xin, y)[1]
            loss.backward()
            opt.step()
            if step == 1 or step % 100 == 0:
                print(f"  {step:5d} |   {loss.item():6.3f}   | {val_loss():.3f}")

        model.eval()
        acc = sum(generate_answer(model, tok, w, COMPARE_TEMPLATE) == w[::-1]
                  for w in test_words) / len(test_words)
        samples = [(w, w[::-1], generate_answer(model, tok, w, COMPARE_TEMPLATE))
                   for w in test_words[:8]]
        results[name] = (tok.vocab_size, acc, samples)
        print(f"  [{name:4s}] vocab {tok.vocab_size:4d} | {steps} steps | held-out acc {acc:6.1%}")

    for name in ("char", "bpe"):
        vocab, acc, samples = results[name]
        print(f"\n  {name} (vocab {vocab}, acc {acc:.1%}) — a few held-out words:")
        for w, want, got in samples:
            got = got.replace("\n", "\\n")
            mark = "✓" if got == want else "✗"
            print(f"    {w:8s}  want {want:8s}  got {got[:8]:8s}  {mark}")

    ca, ba = results["char"][1], results["bpe"][1]
    print(f"\n  SAME task, SAME 0.5M model, SAME {steps} steps — only the tokenizer changed:")
    print(f"    char {ca:.0%}   vs   bpe {ba:.0%}")
    print("  char makes reversal a clean 'copy input tokens backwards'; BPE re-chunks the")
    print("  answer differently from the input, so it must learn a hard chunk->chunk remap")
    print("  and slips in the middle of long words. Tokenization isn't preprocessing — it")
    print("  sets whether a task is a trivial copy or a hard learned mapping. (It also")
    print("  previews why length-general copy/indexing wants better positions, e.g. RoPE.)")


def run_experiments():
    # exp_1_base_cant_follow()
    # exp_2_one_example()
    # exp_3_loss_masking()
    # exp_4_shift()
    # exp_5_batching()
    # exp_6_masked_loss()
    # exp_7_finetune()
    # exp_8_before_after()
    exp_9_tokenization_matters()


@contextlib.contextmanager
def _tee(path):
    """Print to BOTH the terminal and `path` (long runs survive scrollback)."""
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)

        def flush(self):
            for st in self.streams:
                st.flush()

    with open(path, "w") as f:
        with contextlib.redirect_stdout(_Tee(sys.stdout, f)):
            yield
    print(f"(output also written to {path})", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", metavar="FILE", help="also write all output to FILE")
    args = parser.parse_args()

    if args.out:
        with _tee(args.out):
            run_experiments()
    else:
        run_experiments()
