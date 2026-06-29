"""
MDPs & Dynamic Programming — add STATES and TRANSITIONS to the bandit picture, and
plan optimally when you KNOW the model.

In bandits there was one state: pulling an arm changed nothing. A Markov Decision
Process (MDP) adds sequential structure: you're in a state s, take action a, land in
a new state s' (stochastically), collect reward r, and repeat. The whole problem is
captured by the MODEL:

    p(s' | s, a)   transition probabilities
    r(s, a, s')    rewards
    gamma          discount (future reward is worth gamma^t of immediate)

"Markov" = the next state depends only on the CURRENT (s, a), not the history.

THE TWO VALUE FUNCTIONS:
    V^π(s)   = expected discounted return starting in s, then following policy π.
    Q^π(s,a) = same, but FORCE action a first, then follow π.
They're tied together by the BELLMAN EQUATIONS — a value equals immediate reward
plus discounted value of where you land:

    Q^π(s,a) = sum_{s'} p(s'|s,a) [ r + gamma * V^π(s') ]      (one-step lookahead)
    V^π(s)   = sum_a π(a|s) Q^π(s,a)                            (expectation backup)

The OPTIMAL value takes the best action instead of averaging over π:

    V*(s) = max_a sum_{s'} p(s'|s,a) [ r + gamma * V*(s') ]     (optimality backup)

DYNAMIC PROGRAMMING turns those equations into algorithms (model KNOWN):
  - policy evaluation : sweep the expectation backup until V^π stops changing.
  - policy iteration  : evaluate π, then act greedily w.r.t. V^π to get a strictly
                        better π; repeat. Provably converges to π*.
  - value iteration   : sweep the OPTIMALITY backup directly (skip full evaluation),
                        then read the greedy policy off V*.

This file is the ground truth every later, model-FREE method (MC, TD, Q-learning,
...) is secretly trying to approximate from samples.

ENV — the classic Russell & Norvig 4x3 "slippery" gridworld (fully implemented for
you; the DP algorithms below are the exercise):

        . . . +1          start at bottom-left (2,0)
        . # . -1          # is a wall you can't enter
        . . . .           +1 / -1 are terminal (absorbing) states

Actions move UP/RIGHT/DOWN/LEFT, but the floor is slippery: with prob 0.8 you go
where you intended, and 0.1 each you veer to one of the two PERPENDICULAR
directions. Bumping a wall or edge leaves you in place. Every non-terminal move
costs -0.04 (a "living reward" that makes dawdling expensive), so the slip is what
makes p(s'|s,a) interesting and the policy near the trap non-obvious.
"""

from __future__ import annotations

import numpy as np

# action encoding (clockwise, so (a+1)%4 / (a-1)%4 are the perpendicular slips)
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
_DIRS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}
_ARROWS = {UP: "↑", RIGHT: "→", DOWN: "↓", LEFT: "←"}


class GridWorld:
    """The 4x3 slippery gridworld as an EXPLICIT MDP model.

    Exposes exactly what the DP algorithms consume — and nothing gridworld-specific,
    so the algorithms below work on any tabular MDP:
        nS, nA, gamma
        P[s][a] = list of (prob, s_next, reward, done) outcomes summing to prob 1.
    """

    def __init__(self, noise: float = 0.2, step_reward: float = -0.04,
                 gamma: float = 0.9):
        self.rows, self.cols = 3, 4
        self.walls = {(1, 1)}
        self.terminals = {(0, 3): +1.0, (1, 3): -1.0}
        self.start = (2, 0)
        self.noise = noise              # total slip prob, split over the 2 perps
        self.step_reward = step_reward
        self.gamma = gamma

        self.states = [(r, c) for r in range(self.rows) for c in range(self.cols)
                       if (r, c) not in self.walls]
        self.nS = len(self.states)
        self.nA = 4
        self.s2i = {cell: i for i, cell in enumerate(self.states)}
        self.P = self._build_model()

    def _move(self, cell: tuple[int, int], a: int) -> tuple[int, int]:
        """Where action a lands you from `cell`, ignoring slip. Wall/edge => stay."""
        dr, dc = _DIRS[a]
        nxt = (cell[0] + dr, cell[1] + dc)
        if (not 0 <= nxt[0] < self.rows or not 0 <= nxt[1] < self.cols
                or nxt in self.walls):
            return cell
        return nxt

    def _build_model(self) -> dict:
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for cell in self.states:
            s = self.s2i[cell]
            if cell in self.terminals:
                # absorbing: every action self-loops with zero reward => V=0 there.
                for a in range(self.nA):
                    P[s][a] = [(1.0, s, 0.0, True)]
                continue
            for a in range(self.nA):
                outcomes = {a: 1 - self.noise,
                            (a + 1) % 4: self.noise / 2,
                            (a - 1) % 4: self.noise / 2}
                agg: dict[int, list] = {}    # next_state -> [prob, reward, done]
                for act, prob in outcomes.items():
                    nxt = self._move(cell, act)
                    reward = self.terminals.get(nxt, self.step_reward)
                    done = nxt in self.terminals
                    ns = self.s2i[nxt]
                    if ns in agg:            # two slips can land in the same cell
                        agg[ns][0] += prob
                    else:
                        agg[ns] = [prob, reward, done]
                P[s][a] = [(p, ns, r, d) for ns, (p, r, d) in agg.items()]
        return P


# ---------------------------------------------------------------------------
# THE EXERCISE: fill in the five DP routines below. They take a generic `env`
# exposing env.nS, env.nA, env.P and operate on any tabular MDP.
# ---------------------------------------------------------------------------

def q_from_v(env: GridWorld, V: np.ndarray, s: int, gamma: float) -> np.ndarray:
    """One-step lookahead — the Bellman backup core, reused by everything below.

        Q(s,a) = sum_{s'} p(s'|s,a) [ r + gamma * V(s') ]   for each action a.

    Returns an array of shape (nA,).
    """
    # `done` needs no special case: terminal states self-loop with reward 0 and
    # keep V=0, so gamma*V[s'] is already 0 for them.
    q = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, ns, r, done in env.P[s][a]:
            q[a] += prob * (r + gamma * V[ns])
    return q


def policy_evaluation(env: GridWorld, policy: np.ndarray, gamma: float,
                      theta: float = 1e-10) -> np.ndarray:
    """Iterative policy evaluation: repeat the EXPECTATION backup until V^π settles.

        V(s) <- sum_a π(a|s) * Q(s,a)

    `policy` is a stochastic matrix of shape (nS, nA) whose rows sum to 1.
    Returns V of shape (nS,).
    """
    # In-place sweep (reads the freshly-updated V) — converges a touch faster
    # than a synchronous copy.
    V = np.zeros(env.nS)
    while True:
        delta = 0.0
        for s in range(env.nS):
            v_old = V[s]
            V[s] = policy[s] @ q_from_v(env, V, s, gamma)
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    return V


def policy_improvement(env: GridWorld, V: np.ndarray, gamma: float) -> np.ndarray:
    """Greedy improvement: act greedily w.r.t. the one-step lookahead on V.

    Returns a DETERMINISTIC policy encoded as a one-hot matrix of shape (nS, nA).
    """
    policy = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
        best_a = int(np.argmax(q_from_v(env, V, s, gamma)))
        policy[s, best_a] = 1.0
    return policy


def policy_iteration(env: GridWorld, gamma: float,
                     theta: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Alternate evaluation and greedy improvement until the policy stops changing.

    Returns (policy, V) with policy a one-hot (nS, nA) matrix.
    """
    policy = np.ones((env.nS, env.nA)) / env.nA   # start from uniform-random
    V = np.zeros(env.nS)
    while True:
        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V, gamma)
        if (np.argmax(policy, axis=1) == np.argmax(new_policy, axis=1)).all():
            break
        policy = new_policy
    return policy, V


def value_iteration(env: GridWorld, gamma: float,
                    theta: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Sweep the OPTIMALITY backup directly, then read off the greedy policy.

        V(s) <- max_a Q(s,a)

    Returns (policy, V).
    """
    V = np.zeros(env.nS)
    policy = np.zeros((env.nS, env.nA))

    while True:
        delta = 0.0
        for s in range(env.nS):
            v_old = V[s]
            V[s] = np.max(q_from_v(env, V, s, gamma))
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    policy = policy_improvement(env, V, gamma)
    return policy, V


# ---------------------------------------------------------------------------
# Self-check helpers (implemented for you) + the self-check itself.
# Run from the repo root:  uv run python rl/02_mdp_dp.py
# ---------------------------------------------------------------------------

def analytic_policy_value(env: GridWorld, policy: np.ndarray,
                          gamma: float) -> np.ndarray:
    """Ground-truth V^π by SOLVING the linear Bellman system, no iteration:

        V = (I - gamma * P^π)^{-1} r^π

    where P^π(s,s') = sum_a π(a|s) p(s'|s,a) and r^π(s) = expected immediate reward.
    This is the exact thing policy_evaluation converges to — a clean cross-check,
    and a nice interview point (evaluation is just a linear solve; the DP sweep is
    one cheap way to do it without inverting a matrix).
    """
    n = env.nS
    Ppi = np.zeros((n, n))
    rpi = np.zeros(n)
    for s in range(n):
        for a in range(env.nA):
            pa = policy[s, a]
            for prob, ns, r, _ in env.P[s][a]:
                Ppi[s, ns] += pa * prob
                rpi[s] += pa * prob * r
    return np.linalg.solve(np.eye(n) - gamma * Ppi, rpi)


def render_policy(env: GridWorld, policy: np.ndarray) -> str:
    grid = [[" ·" for _ in range(env.cols)] for _ in range(env.rows)]
    for (r, c) in env.walls:
        grid[r][c] = " #"
    for (r, c), val in env.terminals.items():
        grid[r][c] = "+1" if val > 0 else "-1"
    for s, cell in enumerate(env.states):
        if cell in env.terminals:
            continue
        grid[cell[0]][cell[1]] = " " + _ARROWS[int(np.argmax(policy[s]))]
    return "\n".join("".join(row) for row in grid)


if __name__ == "__main__":
    env = GridWorld()
    g = env.gamma

    # --- 0) q_from_v is a correct one-step lookahead -------------------------
    # With V=0, Q(s,a) collapses to the expected IMMEDIATE reward; check that.
    s0 = env.s2i[env.start]
    q0 = q_from_v(env, np.zeros(env.nS), s0, g)
    expected = np.array([sum(p * r for p, ns, r, d in env.P[s0][a])
                         for a in range(env.nA)])
    assert np.allclose(q0, expected), f"q_from_v wrong: {q0} vs {expected}"
    print("q_from_v matches expected immediate reward ✅")

    # --- 1) iterative policy evaluation == analytic linear solve -------------
    rand_pi = np.ones((env.nS, env.nA)) / env.nA
    V_iter = policy_evaluation(env, rand_pi, g)
    V_exact = analytic_policy_value(env, rand_pi, g)
    assert np.allclose(V_iter, V_exact, atol=1e-6), \
        f"policy_evaluation disagrees with the linear solve:\n{V_iter}\n{V_exact}"
    print("policy_evaluation matches the (I-γPπ)⁻¹rπ linear solve ✅")

    # --- 2) policy iteration and value iteration agree on V* and π* ----------
    pi_pol, pi_V = policy_iteration(env, g)
    vi_pol, vi_V = value_iteration(env, g)
    assert np.allclose(pi_V, vi_V, atol=1e-6), \
        f"PI and VI disagree on V*:\n{pi_V}\n{vi_V}"
    assert (np.argmax(pi_pol, 1) == np.argmax(vi_pol, 1)).all(), \
        "PI and VI disagree on the greedy policy"
    print("policy iteration and value iteration agree on V* and π* ✅")

    # --- 3) the optimal value dominates the random policy everywhere ---------
    assert (vi_V >= V_exact - 1e-9).all(), "V* should dominate V^random"
    print("V* dominates V^random in every state ✅")

    # --- 4) following the optimal policy (intended moves) reaches the +1 goal -
    cell = env.start
    path = [cell]
    for _ in range(50):
        if cell in env.terminals:
            break
        cell = env._move(cell, int(np.argmax(vi_pol[env.s2i[cell]])))
        path.append(cell)
    assert cell == (0, 3), f"greedy policy didn't reach the +1 goal: {path}"
    print("greedy rollout walks from start to the +1 goal ✅")

    print("\noptimal policy:")
    print(render_policy(env, vi_pol))
    print("\nMDP dynamic programming works ✅")
