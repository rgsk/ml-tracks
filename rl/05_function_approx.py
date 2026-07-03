"""
Function approximation — the first NEURAL exercise in the track, and the one that
explains why the next three exist. Exercises 1-4 kept a number for every (s) or
(s,a): a TABLE. That dies the moment the state space is large or continuous —
CartPole's state is four real numbers, so there is no "cell" to index. The fix is
to APPROXIMATE the value function with a parameterized function:

    tabular:   Q is a lookup table, one free number per (s,a)
    approx:    Q(s,a; θ) = a function (linear, or a neural net) with WEIGHTS θ

Now states aren't independent: nudging θ to fix Q at one state moves Q at every
similar state too. That GENERALIZATION is the whole point (you can't visit a
continuous space cell-by-cell) — but it's also where everything gets dangerous.

THE UPDATE: SEMI-GRADIENT TD. We still want Q to satisfy the same TD relation as
exercises 3-4, but now we move WEIGHTS instead of table entries. Treat the TD error
as a loss and take a gradient step:

    target  = r + γ * v(s'; θ)            # the bootstrap (same as before)
    δ       = target - v(s; θ)            # TD error
    θ      <- θ + α * δ * ∇_θ v(s; θ)     # move θ to shrink δ

The subtle, MUST-KNOW word is "SEMI". A true gradient of δ² would also differentiate
the target (it contains θ too, via v(s';θ)). We DON'T — we treat the bootstrap target
as a fixed constant and only take the gradient of the prediction v(s;θ). That's why
it's a *semi*-gradient. (For a linear v = θ·φ(s), ∇_θ v = φ(s), so the update is just
θ += α·δ·φ(s). For a net, "treat the target as constant" = `.detach()` it in torch.)
Every part of this exercise is the same one line; only ∇v changes.

THE CATCH — THE DEADLY TRIAD. Tabular TD always converges. Add function
approximation and it can DIVERGE — weights run off to infinity — when THREE things
are present together (Sutton & Barto, ch. 11):

    1. FUNCTION APPROXIMATION   (weights shared across states — generalization)
    2. BOOTSTRAPPING            (TD target built from your own current estimate)
    3. OFF-POLICY               (training on a distribution ≠ the policy you evaluate)

Any TWO is fine; all THREE can blow up. We make this concrete three times, in
increasing realism:

  PART A  Linear semi-gradient TD on a random walk. ON-policy, so triad incomplete
          (1 + 2, no 3) → converges to the true values. FA works.
  PART B  Baird's counterexample. The famous minimal MDP where all three are present
          → weights diverge to infinity even though a PERFECT solution (V≡0) is
          representable. This is THE interview demonstration of the triad.
  PART C  A real neural net (torch) doing naive online Q-learning on CartPole. The
          off-policy max-bootstrap + a function approximator + correlated online
          samples = unstable: it spikes upward then collapses, never holding a
          solve. That instability is exactly what exercise 6 (DQN) fixes with a
          REPLAY BUFFER and a TARGET NETWORK. Consider this the cliffhanger.

Fill in the three TODO functions (semi_gradient_td, baird_sweep, q_learning_step);
the envs, the net, and the self-check are written for you. Run from the repo root:
    uv run python rl/05_function_approx.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ===========================================================================
# PART A — Linear semi-gradient TD prediction (ON-policy → stable)
# ===========================================================================
# Sutton & Barto's 19-state random walk. States 1..19 in a line; step left/right
# with equal probability; terminate at 0 (reward 0) or 20 (reward +1), γ=1.
# Because reward only comes from the right end, the TRUE value is exactly linear:
# V(s) = s / 20. So a linear approximator with a normalized-position feature can
# represent it perfectly, and on-policy semi-gradient TD should find it.

class RandomWalk:
    """reset()/step() random walk. step() ignores its action (policy is fixed:
    uniform left/right) — this is PREDICTION, not control, like exercise 3."""

    N = 19

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.s = None

    @property
    def true_V(self) -> np.ndarray:
        return np.array([s / (self.N + 1) for s in range(1, self.N + 1)])

    def reset(self) -> int:
        self.s = (self.N + 1) // 2          # start in the middle
        return self.s

    def step(self, _a=None):
        self.s += int(self.rng.choice([-1, 1]))
        if self.s == 0:
            return self.s, 0.0, True
        if self.s == self.N + 1:
            return self.s, 1.0, True
        return self.s, 0.0, False


def phi(s: int, N: int = RandomWalk.N) -> np.ndarray:
    """Feature vector for state s: normalized position + a bias term.
    v(s; w) = w · phi(s). The true V is w=[1, 0] (V(s) = s/(N+1))."""
    return np.array([s / (N + 1), 1.0])


def semi_gradient_td(env: RandomWalk, gamma: float, num_episodes: int,
                     alpha: float, rng: np.random.Generator) -> np.ndarray:
    """Linear semi-gradient TD(0) prediction. Returns the weight vector w (shape 2).

    For a linear value v(s) = w · phi(s), the gradient ∇_w v(s) is just phi(s), so
    the semi-gradient update is:

        s = env.reset()
        loop:
            s', r, done = env.step()
            target = r + gamma * (w · phi(s'))   * (1 - done)   # bootstrap; 0 past end
            delta  = target - (w · phi(s))                      # TD error
            w     += alpha * delta * phi(s)                     # ∇v = phi(s)
            s = s';  break on done

    Anneal alpha linearly from `alpha` to 0 over training (alpha * (1 - ep/num_episodes))
    — constant-alpha leaves a noise floor; annealing converges tightly to the true V.

    Note `target` is computed from w but we do NOT differentiate it — we just plug the
    number in. THAT is the "semi-" in semi-gradient. Start from w = zeros(2).
    """
    # TODO: implement linear semi-gradient TD(0) as described above.
    raise NotImplementedError


# ===========================================================================
# PART B — Baird's counterexample (the DEADLY TRIAD → divergence)
# ===========================================================================
# 7 states, 8 weights, all rewards 0, γ=0.99. The value is LINEAR & OVER-PARAMETERIZED:
#   upper states i=0..5:  v = 2*w[i] + w[7]
#   lower state   6:      v = 1*w[6] + 2*w[7]
# encoded by this feature matrix Phi (7 x 8), so v = Phi @ w:
_BAIRD_PHI = np.zeros((7, 8))
for _i in range(6):
    _BAIRD_PHI[_i, _i] = 2.0
    _BAIRD_PHI[_i, 7] = 1.0
_BAIRD_PHI[6, 6] = 1.0
_BAIRD_PHI[6, 7] = 2.0

# The three ingredients, all present:
#   (1) FUNCTION APPROXIMATION — 7 states share 8 weights via Phi above.
#   (2) BOOTSTRAPPING          — TD target uses the current estimate v(s').
#   (3) OFF-POLICY             — we UPDATE all states uniformly (the behavior policy's
#       distribution), but the TARGET policy is "solid": from every state it goes to
#       the lower state (index 6). Training distribution ≠ evaluated policy.
# A perfect solution exists: w = 0 gives v ≡ 0, which IS the true value (rewards are 0).
# Off-policy semi-gradient TD nevertheless drives ||w|| -> infinity.

def baird_sweep(Phi: np.ndarray, w: np.ndarray, gamma: float,
                alpha: float) -> np.ndarray:
    """ONE synchronous off-policy semi-gradient TD(0) update over all 7 states.
    Returns the new weight vector (don't mutate `w` in place — return a fresh array).

        v       = Phi @ w                  # current value of every state, shape (7,)
        v_next  = v[6]                      # target policy 'solid' -> lower state (6)
        delta   = (0 + gamma * v_next) - v # TD error for each state (reward is 0)
        dw      = alpha * sum_s delta[s] * Phi[s]      # = alpha * (delta @ Phi)
        return w + dw

    The update sums the semi-gradient over every state (each weighted equally — the
    behavior policy visits all states uniformly), but every state BOOTSTRAPS off the
    target policy's successor v[6]. That mismatch (off-policy) + bootstrap + shared
    weights is the triad. Watch ||w|| grow every sweep in the self-check.

    Hint: delta is shape (7,), Phi is (7, 8); `delta @ Phi` gives the (8,) weight delta.
    """
    # TODO: implement one synchronous off-policy semi-gradient TD sweep.
    raise NotImplementedError


# ===========================================================================
# PART C — Naive neural Q-learning on CartPole (torch → UNSTABLE)
# ===========================================================================
# Now the approximator is a real neural net and we do CONTROL (Q-learning, exercise 4)
# online: one gradient step per environment transition, NO replay buffer, NO target
# network. This is the deadly triad with a nonlinear approximator — it learns enough
# to spike upward, but can't hold a solve. Exercise 6 (DQN) adds the two fixes.

class QNet(nn.Module):
    """Tiny MLP: CartPole state (4 numbers) -> Q-value for each of the 2 actions."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def q_learning_step(q: QNet, opt: torch.optim.Optimizer, s, a: int, r: float,
                    ns, term: bool, gamma: float) -> float:
    """One online semi-gradient Q-learning update on a single transition.
    Returns the scalar loss (a float). This is exercise 4's Q-learning update, but
    on a NETWORK: the table entry Q[s,a] becomes the net output q(s)[a].

        st  = float tensor of s ;  nt = float tensor of ns
        # bootstrap target — the SEMI-GRADIENT part: do NOT backprop through it.
        with torch.no_grad():
            target = r + gamma * q(nt).max() * (1 - term)   # off-policy max; 0 if terminal
        pred = q(st)[a]                                      # current estimate Q(s,a;θ)
        loss = (pred - target) ** 2                          # shrink the TD error
        opt.zero_grad(); loss.backward(); opt.step()
        return float(loss)

    The `torch.no_grad()` (or equivalently `.detach()` on the target) is exactly the
    "treat the bootstrap as a constant" rule from the top of the file — without it you'd
    be differentiating through your own moving target, which is wrong AND unstable.
    `term` is True only when the pole actually fell (not on time-limit truncation), so
    we don't bootstrap past a real terminal state.
    """
    # TODO: implement the online semi-gradient Q-learning step in torch.
    raise NotImplementedError


def train_cartpole(num_episodes: int, seed: int) -> np.ndarray:
    """Naive online Q-learning on CartPole using q_learning_step. Returns the array of
    per-episode returns. (Written for you — the only TODO is q_learning_step.)"""
    import gymnasium as gym

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")
    q = QNet()
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    gamma = 0.99
    returns = []
    for ep in range(num_episodes):
        eps = max(0.05, 1.0 - ep / 150)           # linear ε decay, then floor 0.05
        s, _ = env.reset(seed=seed + ep)
        done, G = False, 0.0
        while not done:
            if rng.random() < eps:
                a = int(rng.integers(2))
            else:
                with torch.no_grad():
                    a = int(q(torch.tensor(s, dtype=torch.float32)).argmax())
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            q_learning_step(q, opt, s, a, float(r), ns, term, gamma)
            s = ns
            G += r
        returns.append(G)
    return np.array(returns)


# ===========================================================================
# Self-check.  uv run python rl/05_function_approx.py
# ===========================================================================
if __name__ == "__main__":
    # --- PART A: on-policy linear FA converges to the true values -------------
    rng = np.random.default_rng(0)
    env = RandomWalk(rng)
    w = semi_gradient_td(env, gamma=1.0, num_episodes=30000, alpha=0.02, rng=rng)
    est = np.array([w @ phi(s) for s in range(1, env.N + 1)])
    rms = float(np.sqrt(np.mean((est - env.true_V) ** 2)))
    assert rms < 0.03, f"linear semi-gradient TD did not converge: RMS {rms:.4f}"
    print(f"PART A  linear semi-gradient TD fits the random walk   (RMS {rms:.4f}) ✅")
    print(f"        w = {w.round(3)}  (true is ~[1, 0]; v(s) = w·phi(s) ≈ s/20)")
    print("        est vs true at states 1,5,10,15,19: "
          + ", ".join(f"{est[i]:.2f}/{env.true_V[i]:.2f}" for i in (0, 4, 9, 14, 18)))

    # --- PART B: the deadly triad makes Baird's weights diverge ---------------
    w_b = np.array([1., 1., 1., 1., 1., 1., 10., 1.])     # Baird's classic init
    norms = [np.linalg.norm(w_b)]
    print("\nPART B  Baird's counterexample (off-policy + bootstrap + FA):")
    for sweep in range(1, 1001):
        w_b = baird_sweep(_BAIRD_PHI, w_b, gamma=0.99, alpha=0.01)
        norms.append(np.linalg.norm(w_b))
        if sweep in (1, 10, 100, 500, 1000):
            print(f"        sweep {sweep:4d}:  ||w|| = {norms[-1]:14.1f}   "
                  f"v(state 0) = {(_BAIRD_PHI @ w_b)[0]:12.1f}")
    assert all(b > a for a, b in zip(norms, norms[1:])), "||w|| should grow every sweep"
    assert norms[-1] > 1e6, f"expected divergence, got final ||w|| {norms[-1]:.1f}"
    print(f"        ||w|| blew up {norms[-1] / norms[0]:.1e}x — yet w=0 (V≡0) is the "
          "exact answer. Remove ANY one of the three and it would converge. ✅")

    # --- PART C: a real net learns but can't stay stable (→ exercise 6) -------
    print("\nPART C  naive online neural Q-learning on CartPole (this takes ~15s)...")
    rets = train_cartpole(num_episodes=250, seed=0)
    best_window = float(max(np.convolve(rets, np.ones(50) / 50, "valid")))
    print(f"        max episode return : {rets.max():.0f}   (it CAN balance — spikes up)")
    print(f"        best 50-ep average : {best_window:.0f}   (CartPole-v1 'solved' = 475)")
    print(f"        last 20 eps        : mean {rets[-20:].mean():.0f}, "
          f"std {rets[-20:].std():.0f}   (high variance — it won't HOLD a solve)")
    assert rets.max() > 200, f"net never learned to balance at all: max {rets.max():.0f}"
    assert best_window < 450, \
        f"naive online Q unexpectedly stable ({best_window:.0f}) — that's DQN's job (ex 6)"
    print("        learns yet stays UNSTABLE — replay + target net fix this in ex 6. ✅")

    print("\nfunction approximation + the deadly triad ✅")
