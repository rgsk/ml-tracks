"""
Tabular Q-learning & SARSA — the first model-free CONTROL. Exercise 3 only PREDICTED
V^π for a FIXED policy from samples. Now the agent has to IMPROVE the policy from
experience: discover a good way to act with no model and no policy handed to it.

The shift from prediction to control forces two new ideas:

  1. LEARN Q, NOT V. To act greedily you need to compare actions, and without a model
     you can't turn V into a one-step lookahead (that needs p(s'|s,a)). So we estimate
     the ACTION-value Q(s,a) directly — then "act greedily" is just argmax_a Q(s,a),
     no model required. This is why control learns Q while prediction could get away
     with V.

  2. EXPLORATION is now the agent's job. In prediction the policy was given, so every
     state got visited (exercise 3 even forced it with exploring starts). In control
     the policy IS what we're learning: act purely greedily and you'll lock onto the
     first half-decent action and never discover better ones. The fix is the bandit
     idea from exercise 1, now per-state: ε-GREEDY behaviour — usually take argmax Q,
     but with probability ε take a uniformly random action to keep sampling. This is
     the "exploration" thread again (bandits → here → entropy bonus → KL-to-ref later).

THE TWO ALGORITHMS — same TD bootstrap as exercise 3, but the target now bootstraps
off Q, and the question is WHICH next-action value you bootstrap from:

  SARSA (on-policy). Use the action a' you ACTUALLY took next (sampled ε-greedy):
      Q(s,a) <- Q(s,a) + α [ r + γ Q(s',a') - Q(s,a) ]
  The name is literally the tuple it uses: (S, A, R, S', A'). It evaluates the policy
  it's FOLLOWING, exploration and all — so it learns the value of "act ε-greedily".

  Q-LEARNING (off-policy). Bootstrap off the BEST next action, regardless of what you
  did:
      Q(s,a) <- Q(s,a) + α [ r + γ max_{a'} Q(s',a') - Q(s,a) ]
  The max means it evaluates the GREEDY (optimal) policy while still BEHAVING
  ε-greedily. Behaviour policy ≠ target policy ⇒ "off-policy". That max is the
  optimality (Bellman-*) backup from exercise 2, now sampled — the same operator value
  iteration swept, just one transition at a time. It converges to Q* directly.

ON- vs OFF-POLICY is the headline of this exercise, and it shows up as a real
behavioural difference — see the CLIFF WALKING demo at the bottom. SARSA learns the
SAFE path (it knows its own ε-exploration would occasionally step off the cliff, so it
stays away from the edge); Q-learning learns the OPTIMAL path right along the edge (its
target assumes greedy behaviour, so it ignores the exploration risk). Classic
Sutton & Barto Example 6.6, and a favourite interview question.

ONE MORE THING TO NOTICE (a hook for exercise 7): Q-learning's greedy values tend to
come out a touch HIGH vs the true V*. Taking a max over noisy estimates is biased
upward — E[max] ≥ max[E] — the MAXIMIZATION BIAS. That's exactly the bug Double
Q-learning / Double DQN fix later. So below we grade control quality by the VALUE OF
THE LEARNED GREEDY POLICY (what actually matters), not by the raw Q magnitudes.

ENV: exercise 2's 4x3 slippery gridworld again, wrapped reset()/step()-only like
exercise 3 — but now reset() goes to the FIXED start (2,0). No more exploring starts:
ε-greedy is what guarantees coverage in control. Ground truth is value iteration from
exercise 2, so we can grade exactly.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

# Reuse exercise 2's GridWorld + DP ground-truth solvers (digit-led module name, so
# load it by path rather than `import`).
_spec = importlib.util.spec_from_file_location(
    "mdp_dp", pathlib.Path(__file__).with_name("02_mdp_dp.py"))
mdp_dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mdp_dp)
GridWorld = mdp_dp.GridWorld

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
_ARROWS = {UP: "↑", RIGHT: "→", DOWN: "↓", LEFT: "←"}


class GridSampler:
    """Gym-like wrapper over GridWorld: reset()/step(a) only, never env.P.

    Unlike exercise 3's exploring-starts sampler, reset() goes to the FIXED start —
    in control, ε-greedy (not random starts) is what drives exploration. Exposes nS/nA
    so the algorithms below stay env-agnostic (they also run on Cliff, below).
    """

    def __init__(self, env: GridWorld, rng: np.random.Generator):
        self.env = env
        self.rng = rng
        self.nS, self.nA = env.nS, env.nA
        self.s = None

    def reset(self) -> int:
        self.s = self.env.s2i[self.env.start]
        return self.s

    def step(self, a: int):
        """One sampled transition (s', r, done). Env rolls its own slip dice; the
        agent never sees a probability — same black-box contract as exercise 3."""
        env = self.env
        cell = env.states[self.s]
        actual = int(self.rng.choice([a, (a + 1) % 4, (a - 1) % 4],
                                     p=[1 - env.noise, env.noise / 2, env.noise / 2]))
        nxt = env._move(cell, actual)
        reward = env.terminals.get(nxt, env.step_reward)
        done = nxt in env.terminals
        self.s = env.s2i[nxt]
        return self.s, reward, done


# ---------------------------------------------------------------------------
# THE EXERCISE: fill in the three routines below. They may use ONLY env.reset(),
# env.step(a), env.nS, env.nA — never the underlying model. `env` here is any
# reset()/step() world (GridSampler or Cliff).
# ---------------------------------------------------------------------------

def epsilon_greedy(Q: np.ndarray, s: int, eps: float,
                   rng: np.random.Generator, nA: int) -> int:
    """ε-greedy action selection over the current Q row.

    With probability eps return a uniformly random action (EXPLORE); otherwise return
    argmax_a Q[s, a] (EXPLOIT). Break ties among equal-max actions RANDOMLY — early on
    Q is all zeros, and always taking argmax=0 would bias exploration badly.

    Returns an int action in [0, nA).
    """
    if rng.random() < eps:
        return int(rng.integers(nA))
    q = Q[s]
    return int(rng.choice(np.flatnonzero(q == q.max())))


def sarsa(env, gamma: float, num_episodes: int, alpha: float,
          rng: np.random.Generator, eps0: float = 1.0,
          eps_min: float = 0.05) -> np.ndarray:
    """SARSA — ON-policy TD control. Bootstrap off the action you ACTUALLY take next.

    Per episode (ε decays linearly from eps0 to eps_min over training — anneal toward
    greedy as Q sharpens):
        s = env.reset();  a = epsilon_greedy(Q, s, eps, rng, nA)
        loop:
            s', r, done = env.step(a)
            a' = epsilon_greedy(Q, s', eps, rng, nA)          # the extra A in SARSA
            target = r + gamma * Q[s', a'] * (1 - done)       # no bootstrap past end
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = s', a';   break on done

    Note a' is chosen ONCE and reused as next step's a — that's what makes SARSA
    on-policy (it learns the value of the policy it actually follows). Returns Q (nS,nA).
    """
    Q = np.zeros((env.nS, env.nA))
    for ep in range(num_episodes):
        eps = max(eps_min, eps0 * (1 - ep / num_episodes))   # linear ε anneal
        s = env.reset()
        a = epsilon_greedy(Q, s, eps, rng, env.nA)
        for _ in range(1000):
            ns, r, done = env.step(a)
            na = epsilon_greedy(Q, ns, eps, rng, env.nA)      # the second A in SARSA
            target = r + gamma * Q[ns, na] * (1.0 - done)     # bootstrap off a'
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = ns, na                                     # reuse a' as next a
            if done:
                break
    return Q


def q_learning(env, gamma: float, num_episodes: int, alpha: float,
               rng: np.random.Generator, eps0: float = 1.0,
               eps_min: float = 0.05) -> np.ndarray:
    """Q-learning — OFF-policy TD control. Bootstrap off the BEST next action.

    Per episode (same linear ε decay):
        s = env.reset()
        loop:
            a = epsilon_greedy(Q, s, eps, rng, nA)            # behave ε-greedily
            s', r, done = env.step(a)
            target = r + gamma * Q[s'].max() * (1 - done)     # but TARGET the greedy
            Q[s, a] += alpha * (target - Q[s, a])
            s = s';   break on done

    The max (vs SARSA's Q[s',a']) is the only line that differs — and it's the whole
    on/off-policy story. Returns Q (nS, nA).
    """
    Q = np.zeros((env.nS, env.nA))
    for ep in range(num_episodes):
        eps = max(eps_min, eps0 * (1 - ep / num_episodes))
        s = env.reset()
        for _ in range(1000):
            a = epsilon_greedy(Q, s, eps, rng, env.nA)        # behave ε-greedily
            ns, r, done = env.step(a)
            target = r + gamma * Q[ns].max() * (1.0 - done)   # ...but TARGET the max
            Q[s, a] += alpha * (target - Q[s, a])
            s = ns
            if done:
                break
    return Q


# ---------------------------------------------------------------------------
# Self-check helpers (implemented for you) + the self-check itself.
# Run from the repo root:  uv run python rl/04_q_learning_sarsa.py
# ---------------------------------------------------------------------------

def greedy_policy(Q: np.ndarray) -> np.ndarray:
    """Deterministic greedy policy as a one-hot (nS, nA) matrix."""
    pi = np.zeros_like(Q)
    pi[np.arange(len(Q)), Q.argmax(axis=1)] = 1.0
    return pi


def policy_value(env: GridWorld, Q: np.ndarray, gamma: float) -> np.ndarray:
    """TRUE value of Q's greedy policy under the model (exact linear solve from ex 2).
    This is the control-quality metric — robust to Q's maximization-bias magnitude."""
    return mdp_dp.analytic_policy_value(env, greedy_policy(Q), gamma)


def rms(a: np.ndarray, b: np.ndarray, idx: list[int]) -> float:
    return float(np.sqrt(np.mean((a[idx] - b[idx]) ** 2)))


def render_policy(env: GridWorld, Q: np.ndarray) -> str:
    grid = [[" ·" for _ in range(env.cols)] for _ in range(env.rows)]
    for (r, c) in env.walls:
        grid[r][c] = " #"
    for (r, c), val in env.terminals.items():
        grid[r][c] = "+1" if val > 0 else "-1"
    pi = Q.argmax(axis=1)
    for s, cell in enumerate(env.states):
        if cell in env.terminals:
            continue
        grid[cell[0]][cell[1]] = " " + _ARROWS[int(pi[s])]
    return "\n".join("".join(row) for row in grid)


# --- A second env, purely for the on/off-policy demo (fully implemented) ----------
class Cliff:
    """Sutton & Barto Cliff Walking (Example 6.6). 4x12, deterministic moves.
    Start bottom-left, goal bottom-right; the bottom-row cells between them are a CLIFF
    (reward -100, teleport back to start). Every other step costs -1. Undiscounted."""
    rows, cols, nA = 4, 12, 4
    _D = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}

    def __init__(self):
        self.nS = self.rows * self.cols
        self.start, self.goal = (3, 0), (3, 11)
        self.s = None

    def _i(self, rc):
        return rc[0] * self.cols + rc[1]

    def reset(self):
        self.s = self._i(self.start)
        return self.s

    def step(self, a):
        r, c = divmod(self.s, self.cols)
        dr, dc = self._D[a]
        nr = min(max(r + dr, 0), self.rows - 1)
        nc = min(max(c + dc, 0), self.cols - 1)
        if nr == 3 and 1 <= nc <= 10:                # stepped onto the cliff
            self.s = self._i(self.start)
            return self.s, -100.0, False
        self.s = self._i((nr, nc))
        done = (nr, nc) == self.goal
        return self.s, -1.0, done


def render_cliff(env: Cliff, Q: np.ndarray) -> str:
    pi = Q.argmax(axis=1)
    rows = []
    for r in range(env.rows):
        row = ""
        for c in range(env.cols):
            if (r, c) == env.goal:
                row += " G"
            elif r == 3 and 1 <= c <= 10:
                row += " C"
            elif (r, c) == env.start:
                row += " S"
            else:
                row += " " + _ARROWS[int(pi[env._i((r, c))])]
        rows.append(row)
    return "\n".join(rows)


def greedy_path_min_row(env: Cliff, Q: np.ndarray, maxlen: int = 100):
    """Follow the greedy (no-exploration) path from start; return (min row reached,
    whether it reached the goal). Lower row number = farther from the bottom cliff."""
    s = env.reset()
    rows = []
    for _ in range(maxlen):
        r, _c = divmod(s, env.cols)
        rows.append(r)
        ns, _r, done = env.step(int(Q[s].argmax()))
        if done:
            rows.append(ns // env.cols)
            return min(rows), True
        if ns == s:
            break
        s = ns
    return min(rows), False


if __name__ == "__main__":
    env = GridWorld()
    g = env.gamma
    nonterminal = [s for s, c in enumerate(env.states) if c not in env.terminals]

    # Ground truth: optimal value/policy from value iteration (exercise 2).
    vi_pol, V_star = mdp_dp.value_iteration(env, g)

    # --- 0) epsilon_greedy behaves correctly ----------------------------------
    rng = np.random.default_rng(0)
    Qprobe = np.zeros((env.nS, env.nA))
    Qprobe[0] = [0.0, 1.0, 0.0, 0.0]                 # action 1 is the unique greedy
    counts = np.bincount([epsilon_greedy(Qprobe, 0, 0.3, rng, env.nA)
                          for _ in range(20000)], minlength=env.nA)
    freq = counts / counts.sum()
    # greedy prob = (1-eps) + eps/nA = 0.7 + 0.075 = 0.775; others = eps/nA = 0.075
    assert abs(freq[1] - 0.775) < 0.02, f"greedy action freq off: {freq}"
    assert all(abs(freq[a] - 0.075) < 0.02 for a in (0, 2, 3)), \
        f"explore freq off: {freq}"
    assert epsilon_greedy(Qprobe, 0, 0.0, rng, env.nA) == 1, "eps=0 must be greedy"
    print("epsilon_greedy explores/exploits at the right rates ✅")

    # --- 1) Q-learning's greedy policy is near-optimal -------------------------
    Qq = q_learning(GridSampler(env, np.random.default_rng(0)), g,
                    num_episodes=30000, alpha=0.05, rng=np.random.default_rng(0))
    err_q = rms(policy_value(env, Qq, g), V_star, nonterminal)
    assert err_q < 0.12, f"Q-learning greedy policy too far from optimal: {err_q:.4f}"
    print(f"Q-learning recovers a near-optimal policy   (polval RMS {err_q:.4f}) ✅")

    # --- 2) SARSA's greedy policy is near-optimal too --------------------------
    Qs = sarsa(GridSampler(env, np.random.default_rng(1)), g,
               num_episodes=30000, alpha=0.05, rng=np.random.default_rng(1))
    err_s = rms(policy_value(env, Qs, g), V_star, nonterminal)
    assert err_s < 0.12, f"SARSA greedy policy too far from optimal: {err_s:.4f}"
    print(f"SARSA recovers a near-optimal policy         (polval RMS {err_s:.4f}) ✅")

    # --- 3) both greedy rollouts walk from start to the +1 goal ----------------
    for name, Q in (("Q-learning", Qq), ("SARSA", Qs)):
        cell = env.start
        for _ in range(50):
            if cell in env.terminals:
                break
            cell = env._move(cell, int(Q[env.s2i[cell]].argmax()))
        assert cell == (0, 3), f"{name} greedy policy didn't reach the +1 goal"
    print("both greedy policies walk start → +1 goal ✅")

    print("\noptimal policy (value iteration):")
    print(render_policy(env, vi_pol))
    print("\nQ-learning greedy policy:")
    print(render_policy(env, Qq))
    print("\nSARSA greedy policy:")
    print(render_policy(env, Qs))

    # --- 4) ON- vs OFF-POLICY: the famous Cliff Walking divergence -------------
    cliff = Cliff()
    Qc_s = sarsa(cliff, gamma=1.0, num_episodes=500, alpha=0.5,
                 rng=np.random.default_rng(0), eps0=0.1, eps_min=0.1)   # const ε=0.1
    Qc_q = q_learning(cliff, gamma=1.0, num_episodes=500, alpha=0.5,
                      rng=np.random.default_rng(0), eps0=0.1, eps_min=0.1)
    s_minrow, s_ok = greedy_path_min_row(cliff, Qc_s)
    q_minrow, q_ok = greedy_path_min_row(cliff, Qc_q)
    assert s_ok and q_ok, "both should reach the goal greedily"
    # SARSA stays farther from the bottom cliff (smaller row index) than Q-learning.
    assert s_minrow < q_minrow, \
        f"expected SARSA safer (higher) than Q-learning: {s_minrow} vs {q_minrow}"
    print(f"\nCliff Walking — SARSA takes the SAFE path (min row {s_minrow}), "
          f"Q-learning the OPTIMAL cliff-edge path (min row {q_minrow}) ✅")
    print("\nSARSA (on-policy → safe, away from the edge):")
    print(render_cliff(cliff, Qc_s))
    print("\nQ-learning (off-policy → optimal, hugging the cliff edge):")
    print(render_cliff(cliff, Qc_q))

    print("\nmodel-free CONTROL works ✅")
