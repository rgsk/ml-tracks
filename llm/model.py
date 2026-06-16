"""
GPT model, built bottom-up.

Current stage: embeddings + a projection to vocab logits, no attention yet —
effectively a "neural bigram with positions". It can learn token frequencies
but cannot mix information across positions; attention (next) fixes that.

Shape vocabulary used throughout:
    B = batch_size, T = block_size (time/sequence), C = n_embd (channels)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # one learned vector per vocab id, and per position in the context
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        # project the C-dim representation to vocab logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (B, T) token ids. targets: (B, T) or None.

        Returns (logits, loss); loss is None when targets is None.
        """
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)                       # (B, T, C)
        pos = torch.arange(T, device=idx.device)                  # (T,)
        pos_emb = self.position_embedding(pos)                    # (T, C)
        x = tok_emb + pos_emb                                     # broadcast -> (B, T, C)
        logits = self.lm_head(x)                                  # (B, T, vocab_size)

        # cross_entropy wants (N, vocab) and (N,), so collapse B and T
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, self.config.vocab_size),
                targets.view(B * T),
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressively extend idx by max_new_tokens.

        idx: (B, T) current context. Returns (B, T + max_new_tokens). Each new
        token is sampled from the model's distribution and fed back in as part
        of the context for the next step.
        """
        for _ in range(max_new_tokens):
            # crop to block_size: position_embedding only has block_size rows
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)                          # (B, T, vocab)
            logits = logits[:, -1, :]                           # (B, vocab) last step
            probs = F.softmax(logits, dim=-1)                   # (B, vocab)
            next_id = torch.multinomial(probs, num_samples=1)   # (B, 1)
            idx = torch.cat((idx, next_id), dim=1)
        return idx



# ---------------------------------------------------------------------------
# Self-check: run `python model.py`. Don't edit below this line.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(1337)

    cfg = GPTConfig(vocab_size=65, block_size=8, n_embd=32)
    model = GPT(cfg)

    B, T = 4, cfg.block_size
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    targets = torch.randint(0, cfg.vocab_size, (B, T))

    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, cfg.vocab_size), f"bad logits {logits.shape}"
    assert loss is not None and loss.ndim == 0, "loss should be a scalar"

    # no-targets path
    logits_only, none_loss = model(idx)
    assert none_loss is None, "loss must be None without targets"

    # generate: starting from a single token, produce 20 more
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=20)
    assert out.shape == (1, 21), f"bad generate shape {out.shape}"
    # feeding more than block_size tokens must still work (cropping)
    long_start = torch.zeros((1, cfg.block_size + 5), dtype=torch.long)
    out2 = model.generate(long_start, max_new_tokens=3)
    assert out2.shape == (1, cfg.block_size + 8), f"bad crop shape {out2.shape}"

    # sanity: untrained loss ≈ -ln(1/vocab_size) = ln(vocab_size)
    import math
    expected = math.log(cfg.vocab_size)
    print(f"loss = {loss.item():.4f}  (random-init expectation ≈ {expected:.4f})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params = {n_params}")
    print("all checks passed ✅")
