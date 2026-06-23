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

        # n_layer transformer blocks, then a final LayerNorm before the head.
        # ModuleList (not Sequential) so we can thread a per-layer KV cache and a
        # position offset through the stack; state_dict keys (blocks.0, ...) are
        # identical to Sequential, so existing checkpoints still load.
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)

        # project the C-dim representation to vocab logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        # WEIGHT TYING (GPT-2): the token embedding (V,C) and the lm_head weight
        # (out,in)=(V,C) are inverse views of the SAME id<->vector map, so GPT-2
        # shares ONE tensor for both — saving V*C params and tying their grads.
        # The alias means parameters() counts that (V,C) matrix only once.
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # GPT-2 weight init on every submodule. Must run AFTER tying so the shared
        # embedding/lm_head tensor ends up at the 0.02 scale (not nn.Embedding's
        # default N(0,1), which would otherwise make the output logits explode).
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """GPT-2 init scheme, called on every submodule via self.apply.

        Overrides PyTorch's per-layer defaults so (a) the tied lm_head doesn't
        inherit the embedding's N(0,1) scale, and (b) the residual stream stays
        ~unit-variance through depth. LayerNorms are left at their defaults
        (weight=1, bias=0), which is already what GPT-2 wants.
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            # layers that write back into the residual stream are scaled down by
            # 1/sqrt(2*n_layer): each block adds to the stream TWICE (attn + ffwd),
            # so over n_layer blocks the variance would grow ~linearly with the
            # number of adds. Shrinking each contributing projection keeps it flat.
            if getattr(module, "RESIDUAL_PROJ", False):
                std *= (2 * self.config.n_layer) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                kv_caches=None, use_cache: bool = False):
        """idx: (B, T) token ids. targets: (B, T) or None.

        Returns (logits, loss) normally, or (logits, loss, new_caches) when
        use_cache=True. kv_caches is a list (one (k, v) per layer) from a previous
        step; when present, idx is just the NEW token(s) and the cache supplies the
        history — so the position embedding must start at the cached length, not 0.
        """
        B, T = idx.shape

        # absolute position of the first new token = how many positions are cached
        past_len = 0
        if kv_caches is not None and kv_caches[0] is not None:
            past_len = kv_caches[0][0].size(2)                    # cache k: (B, nh, T_past, hs)

        tok_emb = self.token_embedding(idx)                       # (B, T, C)
        pos = torch.arange(past_len, past_len + T, device=idx.device)  # absolute positions
        pos_emb = self.position_embedding(pos)                    # (T, C)
        x = tok_emb + pos_emb                                     # broadcast -> (B, T, C)

        new_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, nc = block(x, layer_cache)                         # (B, T, C), updated cache
            if use_cache:
                new_caches.append(nc)

        x = self.ln_f(x)                                          # (B, T, C)
        logits = self.lm_head(x)                                  # (B, T, vocab_size)

        # cross_entropy wants (N, vocab) and (N,), so collapse B and T
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, self.config.vocab_size),
                targets.view(B * T),
            )

        if use_cache:
            return logits, loss, new_caches
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None,
                 top_p: float | None = None,
                 use_cache: bool = False) -> torch.Tensor:
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

        top_p (nucleus sampling, if set) keeps the smallest set of tokens whose
        cumulative probability reaches p (e.g. 0.9), then renormalizes and samples
        from just those. Unlike top_k's FIXED cutoff, the nucleus ADAPTS to the
        model's confidence: when one token is very likely the set is tiny, when the
        model is unsure the set widens. Thresholds on cumulative PROBABILITY, so it
        runs after softmax (top_k thresholds logits, before). All three compose:
        temperature -> top_k -> softmax -> top_p -> sample.

        use_cache: KV-cache the past K/V instead of recomputing the whole context
        each step — O(T) per token instead of O(T^2). The first step processes the
        full prompt to fill the cache; later steps feed only the newest token. The
        cache stores absolute positions, so total length can't exceed block_size
        (the limit of learned absolute position embeddings — relative schemes like
        RoPE are partly motivated by lifting exactly this restriction).
        """
        if use_cache:
            # the cache holds absolute positions and can't slide, so the whole
            # sequence (prompt + everything generated) must fit in block_size.
            # Fail fast with a clear message instead of an opaque index error when
            # position_embedding / tril run off the end mid-generation.
            total = idx.size(1) + max_new_tokens
            assert total <= self.config.block_size, (
                f"use_cache needs prompt+max_new_tokens ({idx.size(1)}+{max_new_tokens}"
                f"={total}) <= block_size ({self.config.block_size}); the absolute-"
                f"position cache can't exceed it. Use use_cache=False to crop+recompute."
            )

        kv_caches = None
        for _ in range(max_new_tokens):
            if use_cache:
                # first step: feed the whole prompt to build the cache; afterward
                # just the last token (its K/V is appended to the running cache).
                idx_cond = idx if kv_caches is None else idx[:, -1:]
                logits, _, kv_caches = self(idx_cond, kv_caches=kv_caches, use_cache=True)
            else:
                # no cache: re-feed the (cropped) context every step. crop to
                # block_size since position_embedding only has block_size rows.
                idx_cond = idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond)                      # (B, T, vocab)
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

                if top_p is not None:
                    # NUCLEUS SAMPLING — zero out everything OUTSIDE the smallest
                    # set of tokens whose cumulative probability reaches top_p,
                    # then renormalize so the survivors sum to 1 again. Sort by prob
                    # so the nucleus is a prefix; keep the original indices to put
                    # the mask back in vocab order.
                    sorted_probs, sorted_idx = probs.sort(descending=True, dim=-1)
                    
                    cumprobs = sorted_probs.cumsum(dim=-1)       # (B, vocab) running mass
                    # remove a token once the mass BEFORE it already reached p. The
                    # prefix shift (cumprobs - sorted_probs, NOT cumprobs) keeps the
                    # first token that crosses p, and never removes the top-1 token
                    # (its prefix mass is 0, and 0 > p is False for any p > 0).
                    sorted_remove = (cumprobs - sorted_probs) > top_p

                    # scatter the sorted-space mask back to original vocab order
                    remove = torch.zeros_like(sorted_remove).scatter(
                        -1, sorted_idx, sorted_remove
                    )
                    probs = probs.masked_fill(remove, 0.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True)  # renormalize

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

    # --- top-p (nucleus) -----------------------------------------------------
    # 1. p -> 0+ keeps only the single most-probable token, so sampling is
    #    deterministic = greedy (the prefix-shift guarantees the top-1 survives).
    got_p = model.generate(g_start, max_new_tokens=30, temperature=1.0, top_p=1e-8)
    assert torch.equal(ref, got_p), "top_p->0 should collapse to greedy decoding"

    # 2. p = 1.0 removes nothing, so it must match plain sampling token-for-token
    #    under the same seed (renormalization is a no-op when nothing is masked).
    torch.manual_seed(0)
    plain_s = model.generate(ctx, max_new_tokens=15, temperature=1.0)
    torch.manual_seed(0)
    full_p = model.generate(ctx, max_new_tokens=15, temperature=1.0, top_p=1.0)
    assert torch.equal(plain_s, full_p), "top_p=1.0 should equal unfiltered sampling"

    # 3. membership: every sampled token must lie in the nucleus for that context.
    #    Build the nucleus by hand (sort, cumsum, prefix-shift) and assert each draw
    #    is a member — and that the nucleus is a STRICT subset (something was cut).
    p = 0.9
    probs0 = F.softmax(model(ctx)[0][:, -1, :], dim=-1)
    sp, si = probs0.sort(descending=True, dim=-1)
    keep = (sp.cumsum(-1) - sp) <= p                 # prefix-shift: top-1 always kept
    nucleus = set(si[0][keep[0]].tolist())
    assert 0 < len(nucleus) < cfg.vocab_size, f"degenerate nucleus size {len(nucleus)}"
    torch.manual_seed(0)
    for _ in range(300):
        tok = model.generate(ctx, max_new_tokens=1, temperature=1.0, top_p=p)[0, -1].item()
        assert tok in nucleus, f"sampled {tok} outside the nucleus {nucleus}"
    print(f"top-p={p}: nucleus={len(nucleus)}/{cfg.vocab_size} tokens, 300/300 samples inside ✅")

    # --- KV-cache ------------------------------------------------------------
    # Cached generation must produce the EXACT same tokens as the non-cached path.
    # Use greedy (temperature=0) so it's deterministic, and stay within block_size
    # (the cache holds absolute positions, so total length can't exceed it).
    n_new = cfg.block_size - 1
    plain = model.generate(g_start, max_new_tokens=n_new, temperature=0.0, use_cache=False)
    cached = model.generate(g_start, max_new_tokens=n_new, temperature=0.0, use_cache=True)
    assert torch.equal(plain, cached), "KV-cache changed the generated tokens!"
    print(f"KV-cache: cached generation == non-cached, {n_new} tokens ✅")

    # overflowing block_size with the cache must fail fast (not an opaque index error)
    try:
        model.generate(g_start, max_new_tokens=cfg.block_size, use_cache=True)
        raise SystemExit("expected an assertion: cache cannot exceed block_size")
    except AssertionError:
        print("KV-cache: refuses to exceed block_size ✅")

    # sanity: with proper init the untrained loss ≈ -ln(1/vocab_size) = ln(vocab).
    # Logits should start ~uniform, so cross-entropy ≈ ln(vocab). A loss far above
    # this means the init is wrong — e.g. weight tying leaking nn.Embedding's
    # N(0,1) scale into the output projection, which blows the logits up.
    import math
    expected = math.log(cfg.vocab_size)
    print(f"loss = {loss.item():.4f}  (random-init expectation ≈ {expected:.4f})")
    assert abs(loss.item() - expected) < 0.5, (
        f"init loss {loss.item():.2f} is far from ln(vocab)={expected:.2f} — is "
        "_init_weights implemented? Without it, tying leaks the embedding's N(0,1) "
        "scale into the output projection and the logits explode."
    )

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

    # weight tying: when on, lm_head and token_embedding must be ONE shared tensor
    if cfg.tie_weights:
        assert model.lm_head.weight is model.token_embedding.weight, \
            "tie_weights=True but lm_head.weight is a separate tensor (did you copy instead of alias?)"
        print("weight tying: lm_head.weight is token_embedding.weight ✅")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params = {n_params}")
    print("all checks passed ✅")
