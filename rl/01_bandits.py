"""
Multi-Armed Bandits — the simplest RL problem, and the cleanest place to meet the
one tension that never goes away: EXPLORATION vs EXPLOITATION.

Setup: a slot machine with k arms. Pulling arm a pays a random reward drawn from a
fixed (unknown) distribution with TRUE mean q*(a). There is only ONE state and no
transitions — pulling an arm doesn't change anything — so there's no sequential
credit assignment yet (that's MDPs, exercise 2). What's left is pure: with finite
pulls, how do you SPEND them between trying arms you're unsure about (explore) and
milking the best arm so far (exploit)?

THE CORE OBJECT — action-value estimates Q(a):
  Q(a) is our running estimate of q*(a). The honest estimate is just the average
  reward seen from arm a. Storing a sum and recomputing wastes memory, so we keep
  it INCREMENTALLY. After the N-th pull of arm a giving reward R:

      Q(a) <- Q(a) + (1/N) * (R - Q(a))

  Read it as: nudge the old estimate toward the new reward by step 1/N. The
  bracket (R - Q) is an ERROR (sample minus prediction) — the same shape as a TD
  error and a gradient step later. (A CONSTANT step alpha instead of 1/N tracks a
  NON-stationary arm by forgetting old rewards — we keep 1/N here, stationary.)

THREE WAYS TO PICK AN ARM (you implement all three):
  - greedy:     argmax_a Q(a).  Pure exploitation — gets stuck on whatever looked
                good early, never discovers a better arm. The baseline that FAILS.
  - epsilon-greedy: greedy, but with prob epsilon pick a UNIFORMLY RANDOM arm.
                Dead simple, explores forever at a constant low rate.
  - UCB (upper confidence bound): pick argmax_a [ Q(a) + c * sqrt(ln(t) / N(a)) ].
                The second term is an "optimism" bonus that's LARGE for arms pulled
                few times (small N(a)) and shrinks as you learn — so it explores
                the genuinely uncertain arms instead of random ones. Smarter
                exploration, no randomness needed.

We measure agents by REGRET: how much reward you lost versus always pulling the
best arm. Lower = better. greedy plateaus; epsilon-greedy and UCB keep closing in.
"""

from __future__ import annotations

import numpy as np


class GaussianBandit:
    """k arms. Arm a pays reward ~ Normal(q*(a), 1). Stationary (q* fixed)."""

    def __init__(self, k: int = 10, seed: int | None = None):
        self.k = k
        self.rng = np.random.default_rng(seed)
        # true action values q*(a), drawn once from Normal(0, 1).
        self.q_star = self.rng.normal(0.0, 1.0, size=k)
        self.best_arm = int(np.argmax(self.q_star))

    def pull(self, a: int) -> float:
        """Return a noisy reward for arm a: q*(a) plus unit Gaussian noise."""
        return float(self.q_star[a] + self.rng.normal(0.0, 1.0))


class BanditAgent:
    """Holds Q(a) estimates + pull counts N(a), and the three selection rules."""

    def __init__(self, k: int, epsilon: float = 0.0, c: float = 0.0,
                 optimistic: float = 0.0, seed: int | None = None):
        self.k = k
        self.epsilon = epsilon          # exploration rate for epsilon-greedy
        self.c = c                      # exploration strength for UCB (0 => off)
        self.rng = np.random.default_rng(seed)
        # optimistic init: start Q high so every arm looks great until tried,
        # which forces early exploration even for plain greedy (epsilon=0).
        self.Q = np.full(k, float(optimistic))
        self.N = np.zeros(k, dtype=int)
        self.t = 0                      # total pulls so far (for UCB's ln(t))

    def select(self) -> int:
        """Choose an arm. UCB if c > 0, else epsilon-greedy (epsilon may be 0,
        which is plain greedy)."""
        self.t += 1

        if self.c > 0:
            # UCB: argmax_a [ Q(a) + c * sqrt(ln(t) / N(a)) ]. Untried arms get an
            # infinite bonus so they're pulled first (and we dodge the 0-divide).
            bonus = np.full(self.k, np.inf)
            tried = self.N > 0
            bonus[tried] = self.c * np.sqrt(np.log(self.t) / self.N[tried])
            return int(np.argmax(self.Q + bonus))

        # epsilon-greedy: random arm w.p. epsilon, else greedy (epsilon=0 => greedy).
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.k))
        return int(np.argmax(self.Q))

    def update(self, a: int, reward: float) -> None:
        """Incremental sample-average update of Q(a) after pulling arm a:
        Q[a] <- Q[a] + (1/N[a]) * (reward - Q[a])  (the (R - Q) error-nudge)."""
        self.N[a] += 1
        self.Q[a] += (reward - self.Q[a]) / self.N[a]


def run(agent: BanditAgent, bandit: GaussianBandit, steps: int):
    """Run one agent on one bandit for `steps` pulls. Returns (rewards, optimal),
    arrays of per-step reward and whether the chosen arm was the best one."""
    rewards = np.zeros(steps)
    optimal = np.zeros(steps)
    for t in range(steps):
        a = agent.select()
        r = bandit.pull(a)
        agent.update(a, r)
        rewards[t] = r
        optimal[t] = 1.0 if a == bandit.best_arm else 0.0
    return rewards, optimal


# ---------------------------------------------------------------------------
# Self-check: run `uv run python rl/01_bandits.py` from the repo root.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    K, STEPS, RUNS = 10, 1000, 200

    def evaluate(make_agent) -> tuple[float, float]:
        """Average over RUNS independent bandits. Returns (mean reward over the
        LAST 100 steps, fraction-optimal over the last 100 steps)."""
        last_r, last_opt = [], []
        for run_i in range(RUNS):
            bandit = GaussianBandit(K, seed=run_i)
            agent = make_agent(seed=1000 + run_i)
            r, o = run(agent, bandit, STEPS)
            last_r.append(r[-100:].mean())
            last_opt.append(o[-100:].mean())
        return float(np.mean(last_r)), float(np.mean(last_opt))

    # --- 0) incremental average is exactly the sample mean -------------------
    a = BanditAgent(k=3, seed=0)
    samples = [1.0, 3.0, 2.0, 8.0]
    for s in samples:
        a.update(0, s)
    assert abs(a.Q[0] - np.mean(samples)) < 1e-9, \
        f"incremental Q[0]={a.Q[0]} should equal sample mean {np.mean(samples)}"
    assert a.N[0] == len(samples), "N[0] should count the pulls"
    print(f"incremental update == sample mean ✅  (Q={a.Q[0]:.3f})")

    # --- evaluate the three strategies ---------------------------------------
    greedy_r, greedy_opt = evaluate(
        lambda seed: BanditAgent(K, epsilon=0.0, seed=seed))
    eps_r, eps_opt = evaluate(
        lambda seed: BanditAgent(K, epsilon=0.1, seed=seed))
    ucb_r, ucb_opt = evaluate(
        lambda seed: BanditAgent(K, c=2.0, seed=seed))

    print(f"greedy        : reward={greedy_r:.3f}  optimal={greedy_opt:.1%}")
    print(f"epsilon-greedy: reward={eps_r:.3f}  optimal={eps_opt:.1%}")
    print(f"UCB           : reward={ucb_r:.3f}  optimal={ucb_opt:.1%}")

    # --- 1) exploration beats pure greedy ------------------------------------
    # greedy latches onto an early-lucky arm and rarely finds the best one;
    # exploration keeps probing and ends up picking the optimal arm far more.
    assert eps_opt > greedy_opt + 0.15, \
        f"epsilon-greedy ({eps_opt:.1%}) should pick optimal far more than greedy ({greedy_opt:.1%})"
    assert ucb_opt > greedy_opt + 0.15, \
        f"UCB ({ucb_opt:.1%}) should pick optimal far more than greedy ({greedy_opt:.1%})"
    print("exploration beats pure greedy ✅")

    # --- 2) UCB is at least as good as epsilon-greedy here -------------------
    # directed (uncertainty-driven) exploration edges out uniform-random.
    assert ucb_opt >= eps_opt - 0.02, \
        f"UCB ({ucb_opt:.1%}) shouldn't trail epsilon-greedy ({eps_opt:.1%})"
    print("UCB's directed exploration holds up vs epsilon-greedy ✅")

    # --- 3) optimistic greedy explores WITHOUT randomness --------------------
    # Q starts at +5 (way above true means ~0), so every untried arm looks best;
    # plain greedy is then forced to sweep all arms early and finds the good one.
    opt_r, opt_opt = evaluate(
        lambda seed: BanditAgent(K, epsilon=0.0, optimistic=5.0, seed=seed))
    print(f"optimistic    : reward={opt_r:.3f}  optimal={opt_opt:.1%}")
    assert opt_opt > greedy_opt + 0.15, \
        f"optimistic greedy ({opt_opt:.1%}) should beat plain greedy ({greedy_opt:.1%})"
    print("optimistic init drives exploration with zero randomness ✅")

    print("\nbandits work ✅")
