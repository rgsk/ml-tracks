"""
Phase 2 - Exercise: a minimal training loop.

Goal: train the baseline model and watch the loss fall from ~4.17 (ln 65), then
sample text so you can see what this position+bigram model can and cannot do.
(Spoiler: it learns letter frequencies but produces gibberish — it can't look
across positions. That limitation is the whole motivation for attention next.)

Fill in the TODOs, then run `python llm/train.py`.
"""

from __future__ import annotations

import argparse
import os

import torch

from checkpoint import load_checkpoint, save_checkpoint
from config import GPTConfig
from data import load_data, get_batch
from model import GPT

# --- hyperparameters (small, so it runs in a few minutes on CPU) ---
BLOCK_SIZE = 64
BATCH_SIZE = 32
N_EMBD = 96
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1
MAX_ITERS = 3000
EVAL_INTERVAL = 300
EVAL_ITERS = 100
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# two distinct checkpoints, two distinct jobs (see --resume below):
#   last.pt — newest state, for resuming an interrupted run (fault tolerance)
#   best.pt — lowest-val snapshot, for inference, or a warm restart at lower LR
CKPT_DIR = "checkpoints"
LAST_PATH = os.path.join(CKPT_DIR, "last.pt")
BEST_PATH = os.path.join(CKPT_DIR, "best.pt")


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", nargs="?", const="last", choices=["last", "best"], default=None,
        help="resume an interrupted run from last.pt (--resume), or warm-restart "
             "from best.pt (--resume best), typically with a lower --lr",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="override the learning rate; needed because loading the optimizer "
             "state restores the *saved* LR (use with --resume best to fine-tune)",
    )
    args = parser.parse_args()

    torch.manual_seed(1337)

    # 1. data + tokenizer. NOTE: vocab_size for the config comes from the
    #    tokenizer, not a magic number.
    tokenizer, train_data, val_data = load_data()

    # config's vocab_size comes from the tokenizer, not a magic number
    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
    )
    model = GPT(cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"training on {DEVICE} | {n_params/1e6:.2f}M params")
    lr = args.lr if args.lr is not None else LEARNING_RATE
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # resume: restore weights + optimizer + where we left off + the best val so
    # far (so a later run doesn't overwrite a better checkpoint with a worse one).
    #   --resume        -> last.pt: continue an interrupted run from its newest state
    #   --resume best   -> best.pt: warm-restart from the best snapshot, usually --lr lower
    start_step = 0
    best_val = float("inf")
    if args.resume is not None:
        path = LAST_PATH if args.resume == "last" else BEST_PATH
        if os.path.exists(path):
            meta = load_checkpoint(path, model, optimizer, map_location=DEVICE)
            start_step = meta["step"] + 1          # +1: don't redo the saved step
            best_val = meta.get("best_val", best_val)
            # load_state_dict just overwrote the optimizer's LR with the saved one;
            # re-apply the override so --lr actually takes effect on a warm restart.
            if args.lr is not None:
                for group in optimizer.param_groups:
                    group["lr"] = args.lr
            print(f"resumed from {path} @ step {meta['step']} "
                  f"(best val {best_val:.4f}, lr {optimizer.param_groups[0]['lr']:g})")

    os.makedirs(CKPT_DIR, exist_ok=True)

    for step in range(start_step, MAX_ITERS):
        # periodic eval (and a final read on the last step)
        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
            # best.pt only when val improves (model selection); last.pt every time
            # (so an interrupted run resumes from its newest state, not the best one).
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(BEST_PATH, model, optimizer, step, cfg, best_val=best_val)
            save_checkpoint(LAST_PATH, model, optimizer, step, cfg, best_val=best_val)

        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, DEVICE)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)   # grads accumulate by default; clear them
        loss.backward()
        optimizer.step()

    # sample from the trained model, starting from a single newline/token id 0.
    # temperature 0.8 sharpens slightly (less rambly than 1.0); top_k 40 clips the
    # noisy tail so it can't emit junk characters — the usual readable-sample combo.
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    out = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=40)[0].tolist()
    print(tokenizer.decode(out))


if __name__ == "__main__":
    main()
