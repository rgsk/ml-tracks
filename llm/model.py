"""
GPT model.

Full architecture:
    idx -> token_embedding + position_embedding
        -> n_layer transformer Blocks (attention + feedforward)
        -> final LayerNorm
        -> lm_head -> logits over the vocab

Shape vocabulary used throughout:
    B = batch_size, T = block_size (time/sequence), C = n_embd (channels)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import GPTConfig
from transformer import Block


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # one learned vector per vocab id, and per position in the context
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        # n_layer transformer blocks, then a final LayerNorm before the head
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

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

        x = self.blocks(x)                                        # (B, T, C)
        x = self.ln_f(x)                                          # (B, T, C)

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
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """Autoregressively extend idx by max_new_tokens.

        idx: (B, T) current context. Returns (B, T + max_new_tokens). Each new
        token is sampled from the model's distribution and fed back in as part
        of the context for the next step.

        temperature scales the logits BEFORE softmax and controls randomness:
            T = 1.0   unchanged — sample from the model's true distribution
            T < 1.0   sharpens  — mass concentrates on high-logit tokens (toward greedy)
            T > 1.0   flattens   — distribution toward uniform (more random/diverse)
            T -> 0    the argmax wins all the mass = greedy decoding

        top_k (if set) restricts sampling to the k highest-logit tokens; the rest
        are masked out entirely. This kills the long noisy tail — without it, the
        thousands of tiny-probability tokens together hold enough mass to
        occasionally emit garbage. top_k and temperature compose: clip to the top
        k, then temperature reshapes the distribution over just those survivors.
        """
        for _ in range(max_new_tokens):
            # crop to block_size: position_embedding only has block_size rows
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)                          # (B, T, vocab)
            logits = logits[:, -1, :]                           # (B, vocab) last step

            if temperature == 0:
                # greedy: dividing by zero would give inf/nan, so take the argmax
                # (top_k is irrelevant here — the argmax is always inside the top k)
                next_id = logits.argmax(dim=-1, keepdim=True)   # (B, 1)
            else:
                # scale logits BEFORE softmax — softmax(z/T). Scaling probs instead
                # is the classic bug; it doesn't even stay a valid distribution.
                logits = logits / temperature                   # (B, vocab)

                if top_k is not None:
                    # keep only the k largest logits; mask the rest to -inf so they
                    # become exactly 0 probability after softmax (the noisy tail is
                    # truly unreachable, not just unlikely).
                    k = min(top_k, logits.size(-1))     # don't ask for more than vocab
                    v, _ = torch.topk(logits, k)        # (B, k), sorted descending
                    # v[:, [-1]] is the k-th largest logit per row, shape (B, 1)
                    logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))

                probs = F.softmax(logits, dim=-1)               # (B, vocab)
                next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)

            idx = torch.cat((idx, next_id), dim=1)
        return idx



# ---------------------------------------------------------------------------
# Self-check: run `python model.py`. Don't edit below this line.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(1337)

    cfg = GPTConfig(vocab_size=65, block_size=8, n_embd=32, n_head=4, n_layer=3)
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

    # --- temperature ---------------------------------------------------------
    model.eval()

    # 1. T == 0 is exact greedy decoding. Build the reference by hand (argmax at
    #    each step) and assert generate(temperature=0) reproduces it exactly.
    def greedy_ref(idx, n):
        for _ in range(n):
            cond = idx[:, -cfg.block_size:]
            lg, _ = model(cond)
            nxt = lg[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat((idx, nxt), dim=1)
        return idx

    g_start = torch.zeros((1, 1), dtype=torch.long)
    ref = greedy_ref(g_start, 30)
    got = model.generate(g_start, max_new_tokens=30, temperature=0.0)
    assert torch.equal(ref, got), "temperature=0 should be exact greedy decoding"

    # 2. monotonicity: lower temperature must concentrate more mass on the argmax.
    #    Estimate P(next == argmax) at the first step by Monte Carlo; it must be
    #    strictly higher for low T than high T (guaranteed for a unique max).
    ctx = torch.zeros((1, 1), dtype=torch.long)
    top = model(ctx)[0][:, -1, :].argmax().item()   # the greedy token id

    def p_argmax(temp, n=600):
        torch.manual_seed(0)
        hits = sum(
            model.generate(ctx, max_new_tokens=1, temperature=temp)[0, -1].item() == top
            for _ in range(n)
        )
        return hits / n

    p_low, p_high = p_argmax(0.3), p_argmax(3.0)
    assert p_low > p_high, f"low T should favor argmax more: {p_low:.2f} !> {p_high:.2f}"
    print(f"P(argmax): T=0.3 -> {p_low:.2f}, T=3.0 -> {p_high:.2f}  (lower T concentrates)")

    # --- top-k ---------------------------------------------------------------
    # 1. top_k == 1 leaves only the argmax unmasked, so sampling is deterministic
    #    = greedy, for ANY temperature. (Even huge T can't escape a 1-token set.)
    got_k1 = model.generate(g_start, max_new_tokens=30, temperature=5.0, top_k=1)
    assert torch.equal(ref, got_k1), "top_k=1 should collapse to greedy decoding"

    # 2. every sampled token must lie inside the model's top-k set for that context.
    #    Fix the context, take its top-k ids once, then sample many times and assert
    #    each draw is a member — the masked tail must be truly unreachable.
    k = 5
    allowed = set(model(ctx)[0][:, -1, :].topk(k).indices[0].tolist())
    torch.manual_seed(0)
    for _ in range(300):
        tok = model.generate(ctx, max_new_tokens=1, temperature=1.5, top_k=k)[0, -1].item()
        assert tok in allowed, f"sampled {tok} outside the top-{k} set {allowed}"
    print(f"top-{k}: 300/300 samples stayed within the allowed set ✅")

    # sanity: untrained loss ≈ -ln(1/vocab_size) = ln(vocab_size)
    import math
    expected = math.log(cfg.vocab_size)
    print(f"loss = {loss.item():.4f}  (random-init expectation ≈ {expected:.4f})")

    # causality must hold end-to-end through the whole model: changing a future
    # input token must not change the logits at earlier positions.
    model.eval()
    base = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    perturbed = base.clone()
    perturbed[0, -1] = (base[0, -1] + 1) % cfg.vocab_size  # change last token
    l1, _ = model(base)
    l2, _ = model(perturbed)
    assert torch.allclose(l1[:, :-1], l2[:, :-1], atol=1e-5), \
        "causality violated in the full GPT!"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params = {n_params}")
    print("all checks passed ✅")
