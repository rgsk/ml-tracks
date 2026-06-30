"""
WALKTHROUGH: Tabular Q-learning & SARSA (exercise 4), built one layer at a time.

The shift from exercise 3: that was PREDICTION — a policy was handed to us and we
estimated its V^pi from samples. Now we do CONTROL — no policy is given, the agent
has to DISCOVER a good one from experience alone (still no model: only reset/step).

Two new ideas the prediction setting let us dodge, revealed one layer at a time
(run each `exp_*`, watch the output, then say "next"):

  1. LEARN Q, NOT V — to act well you must COMPARE actions, and without a model you
     can't turn V into a choice. Q(s,a) makes "act greedily" = argmax_a Q[s,a], no
     model needed. (this layer)
  2. EXPLORATION is now the agent's job — ε-greedy behaviour, the bandit idea per
     state, since acting purely greedily starves you of the data you'd need to improve.
  3. SARSA — on-policy TD control: bootstrap off the action you ACTUALLY take next.
  4. Q-LEARNING — off-policy TD control: bootstrap off the BEST next action (the max).
  5. ON- vs OFF-POLICY made visible — Cliff Walking: SARSA's safe path vs Q-learning's
     optimal-but-risky cliff-edge path (Sutton & Barto Example 6.6).
  6. (bonus) MAXIMIZATION BIAS — why Q-learning's values come out high, the seed of
     Double Q-learning / Double DQN.

Ground truth for grading comes from exercise 2's value iteration / exact linear solve.
"""

import argparse
import contextlib
import importlib.util
import pathlib
import sys

import numpy as np

# Reuse exercise 2's gridworld (the MDP model) + its exact solvers, loaded by path
# (the module name starts with a digit, so a plain `import` won't parse).
_spec = importlib.util.spec_from_file_location(
    "mdp_dp", pathlib.Path(__file__).with_name("02_mdp_dp.py"))
mdp_dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mdp_dp)
GridWorld = mdp_dp.GridWorld
_ARROWS = mdp_dp._ARROWS


class GridSampler:
    """Gym-like view of the MDP: reset()/step(a) only, never env.P.

    Same black-box contract as exercise 3's Sampler, but reset() goes to the FIXED
    start (2,0) — in CONTROL, ε-greedy (not exploring starts) is what drives coverage.
    """

    def __init__(self, env, rng):
        self.env = env
        self.rng = rng
        self.nS, self.nA = env.nS, env.nA
        self.s = None

    def reset(self):
        self.s = self.env.s2i[self.env.start]
        return self.s

    def step(self, a):
        """One sampled transition (s', r, done). The env rolls its own slip dice; the
        agent only ever sees the single (s', r) that falls out — never a probability."""
        env = self.env
        cell = env.states[self.s]
        actual = int(self.rng.choice([a, (a + 1) % 4, (a - 1) % 4],
                                     p=[1 - env.noise, env.noise / 2, env.noise / 2]))
        nxt = env._move(cell, actual)
        reward = env.terminals.get(nxt, env.step_reward)
        done = nxt in env.terminals
        self.s = env.s2i[nxt]
        return self.s, reward, done


def _banner(*lines):
    print("=" * 66)
    for line in lines:
        print(line)
    print("=" * 66)


# ---------------------------------------------------------------------------
# LAYER 1: why CONTROL learns Q, not V.
#
# In exercise 2 we could be greedy because we had the model: the best action is
#     argmax_a  sum_{s'} p(s'|s,a) [ r + gamma * V(s') ]                 (q_from_v)
# Look at that formula — it NEEDS p(s'|s,a). A model-free agent doesn't have it, so
# it cannot turn V into a decision. The fix: estimate the ACTION-value directly,
#     Q(s,a) = expected return if you take a now, then act well after,
# because then "act greedily" collapses to argmax_a Q[s,a] — no model at decision
# time. This layer just PROVES that point: we Monte-Carlo-estimate Q(s0, a) for each
# action from SAMPLES ALONE, and show argmax recovers the same best action the
# model-based q_from_v gives — but without ever touching env.P.
# ---------------------------------------------------------------------------

def mc_q_estimate(sampler, policy, gamma, s0, a0, n, rng, max_steps=1000):
    """Monte-Carlo estimate of Q(s0, a0): force action a0 from s0, then FOLLOW
    `policy` to the end, and average the discounted return over n rollouts.
    Uses only reset/step — never env.P. (This is exactly what Q^pi means.)"""
    env = sampler.env
    total = 0.0
    for _ in range(n):
        sampler.s = s0                       # force the start state
        ns, r, done = sampler.step(a0)       # ...and the first action
        G, disc, s = r, gamma, ns
        steps = 0
        while not done and steps < max_steps:
            a = int(rng.choice(env.nA, p=policy[s]))
            ns, r, done = sampler.step(a)
            G += disc * r
            disc *= gamma
            s = ns
            steps += 1
        total += G
    return total / n


def exp_why_q(cell=(2, 0), n=20000, seed=0):
    """For the start cell, estimate Q(s0, a) from SAMPLES for all four actions and
    compare to the model's q_from_v. The argmax (the greedy action) must agree —
    that's the whole point: Q makes greedy choice model-free."""
    env = GridWorld()
    gamma = env.gamma
    rng = np.random.default_rng(seed)
    sampler = GridSampler(env, rng)

    # ground-truth optimal policy + its exact V*, both from exercise 2.
    opt_policy, _ = mdp_dp.value_iteration(env, gamma)
    V_star = mdp_dp.analytic_policy_value(env, opt_policy, gamma)
    s0 = env.s2i[cell]

    _banner(f"LAYER 1: control needs Q, not V   (state {cell}, n={n} rollouts/action)")

    # model-based action values (what exercise 2 reads off env.P) — needs the model.
    q_model = mdp_dp.q_from_v(env, V_star, s0, gamma)
    # model-FREE action values, Monte-Carlo'd from samples — needs only reset/step.
    q_mc = np.array([mc_q_estimate(sampler, opt_policy, gamma, s0, a, n, rng)
                     for a in range(env.nA)])

    print("  a    arrow |  model Q (uses env.P) |  sampled Q (reset/step only)")
    print("-------------+-----------------------+-----------------------------")
    for a in range(env.nA):
        print(f"  {a}     {_ARROWS[a]}   |        {q_model[a]:+.3f}        |"
              f"          {q_mc[a]:+.3f}")
    print(f"\n  greedy action  argmax_a Q :  model -> {_ARROWS[int(q_model.argmax())]} "
          f"  |  sampled -> {_ARROWS[int(q_mc.argmax())]}   "
          f"(match={q_model.argmax() == q_mc.argmax()})")
    print("\n  Point: argmax over Q is a model-FREE decision. argmax over V would need")
    print("  sum_s' p(s'|s,a)[r+gamma V(s')] — the model we no longer have. So control")
    print("  learns a Q-TABLE. Next: how to learn it for ALL states from one stream.")


# ---------------------------------------------------------------------------
# LAYER 2: ε-greedy — exploration is now the agent's job.
#
# In prediction the policy was given, so coverage was free. In control the policy
# IS what we're learning, and a purely greedy agent traces ONE line through the
# world: at each state it only ever CHOOSES its current-best action, so every other
# (s,a) gets zero data and its Q can never improve. But TD control only converges if
# every (s,a) is visited infinitely often. ε-greedy buys exactly that: usually take
# argmax_a Q[s,a] (EXPLOIT), but with probability ε take a uniformly random action
# (EXPLORE). Same exploration/exploitation tension as the bandits in exercise 1, now
# per state. Tie-break the greedy branch RANDOMLY — at init Q is all zeros, so every
# action ties, and always returning argmax=0 would bias exploration hard.
# ---------------------------------------------------------------------------

def epsilon_greedy(Q, s, eps, rng, nA):
    """ε-greedy: random action w.p. eps, else argmax_a Q[s,a] with random tie-break."""
    if rng.random() < eps:
        return int(rng.integers(nA))
    q = Q[s]
    return int(rng.choice(np.flatnonzero(q == q.max())))


def exp_epsilon_greedy(seed=0):
    """Two views of ε-greedy:
    A) the explore/exploit RATES on a fixed Q row — check them against the formula.
    B) (s,a) COVERAGE: behave (no learning) under greedy vs ε-greedy and count which
       state-action pairs ever get sampled. Greedy starves most of the table; ε
       reaches all of it — that's the condition TD control needs to converge."""
    env = GridWorld()
    rng = np.random.default_rng(seed)
    nA = env.nA

    _banner("LAYER 2A: ε-greedy explore/exploit rates (fixed Q row, greedy action = →)")
    Qrow = np.zeros((1, nA))
    Qrow[0, 1] = 1.0                              # action 1 (→) is the unique best
    n = 40000
    print("  eps  | P(greedy →) |  P(each other) |  formula (1-eps)+eps/nA")
    print("-------+-------------+----------------+------------------------")
    for eps in (0.0, 0.1, 0.5, 1.0):
        counts = np.bincount([epsilon_greedy(Qrow, 0, eps, rng, nA) for _ in range(n)],
                             minlength=nA) / n
        print(f"  {eps:.1f}  |    {counts[1]:.3f}    |     {counts[0]:.3f}      |"
              f"         {(1 - eps) + eps / nA:.3f}")

    # --- B) coverage of (s,a) pairs while merely BEHAVING (no learning) ---------
    opt_policy, _ = mdp_dp.value_iteration(env, env.gamma)
    V_star = mdp_dp.analytic_policy_value(env, opt_policy, env.gamma)
    Q_star = np.array([mdp_dp.q_from_v(env, V_star, s, env.gamma)
                       for s in range(env.nS)])     # a fully-trained Q to behave on
    nonterm = [s for s, c in enumerate(env.states) if c not in env.terminals]

    episodes = 8000
    _banner(f"LAYER 2B: (s,a) coverage over {episodes} episodes — greedy vs ε-greedy")
    print(f"  {len(nonterm)} non-terminal states x {nA} actions = "
          f"{len(nonterm) * nA} (s,a) pairs to cover")
    print("   eps  | distinct (s,a) sampled | (s,a) never sampled | min visits")
    print("--------+------------------------+---------------------+-----------")
    for eps in (0.0, 0.1):
        sampler = GridSampler(env, np.random.default_rng(seed))
        visits = np.zeros((env.nS, nA))
        for _ in range(episodes):
            s = sampler.reset()
            for _ in range(1000):
                a = epsilon_greedy(Q_star, s, eps, sampler.rng, nA)
                visits[s, a] += 1
                ns, _r, done = sampler.step(a)
                s = ns
                if done:
                    break
        sub = visits[nonterm]                       # only the pairs we could control
        covered = int((sub > 0).sum())
        never = int((sub == 0).sum())
        print(f"  {eps:.2f}  |          {covered:2d}            |"
              f"         {never:2d}          |    {int(sub.min())}")
    print("\n  Greedy only ever CHOOSES one action per state, so ~one (s,a) per state")
    print("  gets data and the rest stay at zero visits — unlearnable. ε-greedy hits")
    print("  every pair (min visits > 0): the coverage SARSA / Q-learning rely on.")


# ---------------------------------------------------------------------------
# Shared grading helpers (model-based ground truth, used by every control layer).
# ---------------------------------------------------------------------------

def greedy_policy(Q):
    """Deterministic greedy policy as a one-hot (nS, nA) matrix."""
    pi = np.zeros_like(Q)
    pi[np.arange(len(Q)), Q.argmax(axis=1)] = 1.0
    return pi


def policy_value(env, Q, gamma):
    """TRUE value of Q's greedy policy under the model (exact linear solve from ex 2).
    The control-quality metric — robust to Q's magnitude (which can drift high)."""
    return mdp_dp.analytic_policy_value(env, greedy_policy(Q), gamma)


def _rms(a, b, idx):
    return float(np.sqrt(np.mean((a[idx] - b[idx]) ** 2)))


def render_policy(env, Q, title):
    """Draw the greedy action arrow in each cell."""
    print(title)
    pi = Q.argmax(axis=1)
    for r in range(env.rows):
        row = []
        for c in range(env.cols):
            cell = (r, c)
            if cell in env.walls:
                row.append("#")
            elif cell in env.terminals:
                row.append("+1" if env.terminals[cell] > 0 else "-1")
            else:
                row.append(_ARROWS[int(pi[env.s2i[cell]])])
        print("  ".join(f"{x:>2}" for x in row))
    print()


# ---------------------------------------------------------------------------
# LAYER 3: SARSA — the first real CONTROL loop. ON-policy TD.
#
# Take the TD(0) bootstrap from exercise 3 and move it from V to Q. After a
# transition (s,a,r,s'), pick the NEXT action a' the way you'll actually act
# (ε-greedy), and bootstrap off THAT action's value:
#     Q(s,a) <- Q(s,a) + alpha [ r + gamma*Q(s',a') - Q(s,a) ]
# The update literally consumes the tuple (S, A, R, S', A') — hence "SARSA". Because
# a' is the action you go on to take, SARSA evaluates the policy it is FOLLOWING,
# exploration and all. And the magic of CONTROL: we never told it the optimal policy.
# Acting greedily w.r.t. an improving Q, while that same Q is updated toward the
# returns of that behaviour, ratchets the policy up to optimal (generalized policy
# iteration, one transition at a time). ε decays toward 0 so behaviour anneals to
# greedy as Q sharpens.
# ---------------------------------------------------------------------------

def sarsa(env, gamma, num_episodes, alpha, rng, eps0=1.0, eps_min=0.05):
    """ON-policy TD control. Returns the learned Q (nS, nA)."""
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


def exp_sarsa(seed=1, alpha=0.05, num_episodes=30000):
    """Watch SARSA learn: checkpoint the VALUE of its current greedy policy vs the
    exact V* as episodes accumulate, then show the final greedy policy on the grid."""
    env = GridWorld()
    gamma = env.gamma
    rng = np.random.default_rng(seed)
    sampler = GridSampler(env, np.random.default_rng(seed))
    opt_policy, _ = mdp_dp.value_iteration(env, gamma)
    V_star = mdp_dp.analytic_policy_value(env, opt_policy, gamma)
    nonterm = [s for s, c in enumerate(env.states) if c not in env.terminals]

    opt_act = opt_policy.argmax(1)
    V_rand = mdp_dp.analytic_policy_value(
        env, np.ones((env.nS, env.nA)) / env.nA, gamma)    # baseline: random walk

    _banner(f"LAYER 3: SARSA control (alpha={alpha}, ε: 1.0 -> 0.05 over training)")
    print(f"  baselines (mean V over non-terminals):  random π = {V_rand[nonterm].mean():+.3f}"
          f"   |   optimal π* = {V_star[nonterm].mean():+.3f}")
    print("episodes |  greedy actions = π*  |  mean greedy V  |  value RMS vs V*")
    print("---------+----------------------+-----------------+-----------------")
    checkpoints = [10, 30, 100, 300, 1000, 5000, 30000]
    prev = 0
    Q = np.zeros((env.nS, env.nA))
    for cp in checkpoints:
        for ep in range(prev, cp):
            eps = max(0.05, 1.0 * (1 - ep / num_episodes))
            s = sampler.reset()
            a = epsilon_greedy(Q, s, eps, rng, env.nA)
            for _ in range(1000):
                ns, r, done = sampler.step(a)
                na = epsilon_greedy(Q, ns, eps, rng, env.nA)
                Q[s, a] += alpha * (r + gamma * Q[ns, na] * (1.0 - done) - Q[s, a])
                s, a = ns, na
                if done:
                    break
        prev = cp
        vpi = policy_value(env, Q, gamma)
        match = int((Q.argmax(1)[nonterm] == opt_act[nonterm]).sum())
        print(f"{cp:8d} |        {match}/{len(nonterm)}          |"
              f"      {vpi[nonterm].mean():+.3f}     |       {_rms(vpi, V_star, nonterm):.4f}")

    print()
    render_policy(env, np.eye(env.nA)[opt_policy.argmax(1)],
                  "optimal policy (value iteration):")
    render_policy(env, Q, "SARSA greedy policy (final):")


# ---------------------------------------------------------------------------
# LAYER 4: Q-learning — change ONE line, and the meaning changes. OFF-policy TD.
#
# SARSA bootstrapped off a' = the action it actually takes next. Q-learning instead
# bootstraps off the BEST next action, whatever it ends up doing:
#     Q(s,a) <- Q(s,a) + alpha [ r + gamma * max_{a'} Q(s',a') - Q(s,a) ]
#                                          ^^^^^^^^^^^^^^^^  the only change
# That max is the optimality (Bellman-*) backup from exercise 2 — the operator value
# iteration swept — but applied to ONE sampled transition at a time. So Q-learning's
# target is the GREEDY policy's value while its BEHAVIOUR is still ε-greedy: behaviour
# policy ≠ target policy ⇒ OFF-policy. It converges straight to Q*, independent of how
# much it explored. Side effect to notice (Layer 6): taking a max over noisy estimates
# is biased UP (E[max] ≥ max E), so the learned max-Q tends to sit ABOVE the true V*.
# ---------------------------------------------------------------------------

def q_learning(env, gamma, num_episodes, alpha, rng, eps0=1.0, eps_min=0.05):
    """OFF-policy TD control. Returns the learned Q (nS, nA)."""
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


def exp_qlearning(seed=1, alpha=0.05, num_episodes=30000):
    """Same checkpoint table as SARSA — Q-learning converges to V* too — plus the
    overestimation tell: the learned max-Q sits a bit ABOVE the true V* (the seed of
    maximization bias / Double-Q, Layer 6). The greedy-POLICY value stays honest."""
    env = GridWorld()
    gamma = env.gamma
    rng = np.random.default_rng(seed)
    sampler = GridSampler(env, np.random.default_rng(seed))
    opt_policy, _ = mdp_dp.value_iteration(env, gamma)
    V_star = mdp_dp.analytic_policy_value(env, opt_policy, gamma)
    opt_act = opt_policy.argmax(1)
    nonterm = [s for s, c in enumerate(env.states) if c not in env.terminals]

    _banner(f"LAYER 4: Q-learning control (alpha={alpha}, ε: 1.0 -> 0.05)")
    print(f"  optimal mean V over non-terminals: π* = {V_star[nonterm].mean():+.3f}")
    print("episodes | actions=π* | mean greedy V | RMS vs V* | mean learned max-Q")
    print("---------+------------+---------------+-----------+-------------------")
    checkpoints = [10, 30, 100, 300, 1000, 5000, 30000]
    prev = 0
    Q = np.zeros((env.nS, env.nA))
    for cp in checkpoints:
        for ep in range(prev, cp):
            eps = max(0.05, 1.0 * (1 - ep / num_episodes))
            s = sampler.reset()
            for _ in range(1000):
                a = epsilon_greedy(Q, s, eps, rng, env.nA)
                ns, r, done = sampler.step(a)
                Q[s, a] += alpha * (r + gamma * Q[ns].max() * (1.0 - done) - Q[s, a])
                s = ns
                if done:
                    break
        prev = cp
        vpi = policy_value(env, Q, gamma)
        match = int((Q.argmax(1)[nonterm] == opt_act[nonterm]).sum())
        print(f"{cp:8d} |    {match}/{len(nonterm)}     |    {vpi[nonterm].mean():+.3f}     |"
              f"   {_rms(vpi, V_star, nonterm):.4f}  |       {Q.max(1)[nonterm].mean():+.3f}")

    print(f"\n  true mean V* = {V_star[nonterm].mean():+.3f}, learned mean max-Q = "
          f"{Q.max(1)[nonterm].mean():+.3f}. On this mild env the max-bias is small and")
    print("  partly hidden by constant-α lag (which pulls Q low). Layer 6 builds a setup")
    print("  designed to expose the systematic OVERESTIMATION cleanly.\n")
    render_policy(env, np.eye(env.nA)[opt_act], "optimal policy (value iteration):")
    render_policy(env, Q, "Q-learning greedy policy (final):")


# ---------------------------------------------------------------------------
# LAYER 5: ON- vs OFF-policy made visible — Cliff Walking (Sutton & Barto Ex 6.6).
#
# On the gridworld the two agents landed in basically the same place. Here they
# DON'T — and the reason is exactly on- vs off-policy. The map (S start, G goal, the
# bottom row between them is a CLIFF that costs -100 and teleports you back to start;
# every other step costs -1, undiscounted):
#
#     . . . . . . . . . . . .
#     . . . . . . . . . . . .
#     . . . . . . . . . . . .
#     S C C C C C C C C C C G
#
# The OPTIMAL path skirts the cliff edge (row 2): shortest, 13 steps. But the agent
# behaves ε-greedily — and one random step DOWN from the edge falls off. SARSA is
# on-policy: its Q[s',a'] sometimes IS that fatal step, so the edge looks bad and it
# learns the SAFE detour along the top. Q-learning is off-policy: its max target
# assumes a perfect greedy next move, so the edge looks fine — it learns the OPTIMAL
# edge path while still occasionally falling off as it explores. The twist: during
# TRAINING, Q-learning earns LESS reward (more cliff falls) even though its final
# policy is "better". Optimal policy ≠ best online behaviour when you must explore.
# ---------------------------------------------------------------------------

UP, RIGHT, DOWN, LEFT = mdp_dp.UP, mdp_dp.RIGHT, mdp_dp.DOWN, mdp_dp.LEFT


class Cliff:
    """4x12 Cliff Walking. Deterministic moves; bottom row (cols 1..10) is the cliff."""
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
        if nr == 3 and 1 <= nc <= 10:                 # stepped onto the cliff
            self.s = self._i(self.start)
            return self.s, -100.0, False
        self.s = self._i((nr, nc))
        done = (nr, nc) == self.goal
        return self.s, -1.0, done


def render_cliff(env, Q, title):
    print(title)
    pi = Q.argmax(axis=1)
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
        print(row)
    print()


def _greedy_path(env, Q, maxlen=100):
    """Follow the greedy (no-exploration) path from start: return (length, min row
    reached, reached_goal). Lower row = farther from the bottom cliff = safer."""
    s = env.reset()
    rows, steps = [], 0
    for _ in range(maxlen):
        rows.append(s // env.cols)
        ns, _r, done = env.step(int(Q[s].argmax()))
        steps += 1
        if done:
            rows.append(ns // env.cols)
            return steps, min(rows), True
        if ns == s:
            break
        s = ns
    return steps, min(rows), False


def exp_cliff(seed=0, num_episodes=500, alpha=0.5, eps=0.1):
    """Train SARSA and Q-learning on the cliff (constant ε=0.1), recording the ONLINE
    return of each training episode. Then show the two greedy policies + the online
    performance gap — the headline on/off-policy result."""
    env = Cliff()

    def train(kind):
        rng = np.random.default_rng(seed)
        Q = np.zeros((env.nS, env.nA))
        online = []
        for _ in range(num_episodes):
            s = env.reset()
            G = 0.0
            a = epsilon_greedy(Q, s, eps, rng, env.nA)        # SARSA needs a upfront
            for _ in range(2000):
                if kind == "q":
                    a = epsilon_greedy(Q, s, eps, rng, env.nA)   # Q picks fresh each step
                ns, r, done = env.step(a)
                G += r
                if kind == "sarsa":
                    na = epsilon_greedy(Q, ns, eps, rng, env.nA)
                    Q[s, a] += alpha * (r + Q[ns, na] * (1.0 - done) - Q[s, a])
                    s, a = ns, na
                else:
                    Q[s, a] += alpha * (r + Q[ns].max() * (1.0 - done) - Q[s, a])
                    s = ns
                if done:
                    break
            online.append(G)
        return Q, np.array(online)

    Qs, on_s = train("sarsa")
    Qq, on_q = train("q")

    _banner(f"LAYER 5: Cliff Walking — SARSA vs Q-learning (ε={eps} constant)")
    render_cliff(env, Qs, "SARSA greedy policy (on-policy -> SAFE, hugs the top):")
    render_cliff(env, Qq, "Q-learning greedy policy (off-policy -> OPTIMAL, hugs the edge):")

    ls, mrs, oks = _greedy_path(env, Qs)
    lq, mrq, okq = _greedy_path(env, Qq)
    print(f"  greedy path:  SARSA  len={ls:2d}  min-row={mrs} (top) reached={oks}")
    print(f"                Q-learn len={lq:2d}  min-row={mrq} (edge) reached={okq}"
          f"   <- shorter = optimal\n")
    print(f"  ONLINE return DURING training (mean over last 100 episodes):")
    print(f"     SARSA      = {on_s[-100:].mean():+.1f}   (rarely falls off)")
    print(f"     Q-learning = {on_q[-100:].mean():+.1f}   (keeps falling off while exploring)")
    print("\n  So Q-learning's FINAL policy is optimal, yet its TRAINING return is worse:")
    print("  its target ignores the exploration risk it is actually running. SARSA's")
    print("  target feels that risk and trades a few extra steps for safety. Same algorithms,")
    print("  same ε — the only difference is max vs Q[s',a'].")


# ---------------------------------------------------------------------------
# LAYER 6 (bonus): MAXIMIZATION BIAS — why Q-learning over-values, and the fix.
#
# Q-learning's target is gamma * max_a Q(s',a). The max is taken over ESTIMATES, and
# E[max of noisy estimates] >= max of the true values, even when every true value is
# equal. So the SAME numbers are used to (1) pick the best action and (2) report its
# value — a single noisy table doing both jobs systematically inflates the target.
#
# Sutton & Barto's Example 6.7 turns that statistical fact into a CONTROL error. From
# state A: 'right' ends the episode at reward 0; 'left' goes to B at reward 0. From B,
# many actions all end the episode with reward ~ N(-0.1, 1). So every route through B
# has expected return -0.1 < 0 — the optimal action at A is RIGHT. But max_a Q(B,a)
# over noisy estimates comes out POSITIVE early, so Q-learning thinks 'left' is great
# and takes it far more than the 5% an ε-greedy optimal agent would. DOUBLE Q-learning
# fixes it with two tables: one PICKS the argmax, the OTHER reports its value, so the
# selection noise and the evaluation noise are independent and don't compound.
# ---------------------------------------------------------------------------

def exp_max_bias(seed=0):
    """A) the statistical root: E[max of k unbiased estimates] drifts above the truth
    and grows with k.  B) the control error on the A/B MDP, and how Double Q fixes it."""
    rng = np.random.default_rng(seed)

    _banner("LAYER 6A: E[max] >= max E — the bias is in the MAX over estimates")
    print("  k actions, each TRUE value 0, estimated from n=10 samples (mean 0, var 1):")
    print("    k   |  E[max_a Q_hat(a)]   (should be 0; >0 = upward bias, grows with k)")
    print("  ------+-------------------")
    trials, n = 20000, 10
    for k in (1, 2, 5, 10, 50):
        est = rng.standard_normal((trials, k, n)).mean(axis=2)   # (trials,k) sample means
        print(f"    {k:2d}  |       {est.max(axis=1).mean():+.3f}")

    # --- B) the A/B MDP: does it cause a wrong CHOICE? -------------------------
    def run_ab(double, runs=2000, episodes=300, alpha=0.1, eps=0.1, nB=10, gamma=1.0):
        rng = np.random.default_rng(seed)
        left_frac = np.zeros(episodes)

        def pick(q):                          # ε-greedy with random tie-break
            if rng.random() < eps:
                return int(rng.integers(len(q)))
            return int(rng.choice(np.flatnonzero(q == q.max())))

        for _ in range(runs):
            QA = [np.zeros(2), np.zeros(2)]    # [table0, table1]; plain Q uses table0 only
            QB = [np.zeros(nB), np.zeros(nB)]
            for ep in range(episodes):
                qA = (QA[0] + QA[1]) if double else QA[0]
                aA = pick(qA)
                if aA == 1:                    # 'left' = the tempting-but-bad action
                    left_frac[ep] += 1
                    qB = (QB[0] + QB[1]) if double else QB[0]
                    aB = pick(qB)
                    r = rng.normal(-0.1, 1.0)
                    if double:                 # update one B-table toward terminal r
                        j = rng.integers(2)
                        QB[j][aB] += alpha * (r - QB[j][aB])
                        i = rng.integers(2)    # decoupled target for the A-left value
                        astar = int(np.argmax(QB[i]))
                        QA[i][1] += alpha * (gamma * QB[1 - i][astar] - QA[i][1])
                    else:
                        QB[0][aB] += alpha * (r - QB[0][aB])
                        QA[0][1] += alpha * (gamma * QB[0].max() - QA[0][1])
                else:                          # 'right' = optimal, ends at reward 0
                    if double:
                        i = rng.integers(2)
                        QA[i][0] += alpha * (0.0 - QA[i][0])
                    else:
                        QA[0][0] += alpha * (0.0 - QA[0][0])
        return left_frac / runs

    lq = run_ab(double=False)
    ld = run_ab(double=True)
    _banner("LAYER 6B: A/B MDP — % of episodes that take 'left' (optimal would be ~5%)")
    print("  episode |  Q-learning %left  |  Double-Q %left   (optimal floor = ε/2 = 5%)")
    print("  --------+--------------------+-----------------")
    for ep in (1, 5, 10, 25, 50, 100, 300):
        print(f"   {ep:4d}   |       {100 * lq[ep - 1]:5.1f}%       |"
              f"       {100 * ld[ep - 1]:5.1f}%")
    print("\n  Q-learning chases 'left' far above 5% (its max over B inflates left's")
    print("  value); Double-Q's decoupled estimate stays near the 5% ε-floor — it knows")
    print("  'right' is optimal. This is exactly the Double-DQN fix, one exercise early.")


def run_experiments():
    # exp_why_q()
    # exp_epsilon_greedy()
    # exp_sarsa()
    # exp_qlearning()
    # exp_cliff()
    exp_max_bias()
    # (all six layers built — uncomment any to re-run it)


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
    parser.add_argument("--out", metavar="FILE",
                        help="also write all output to FILE")
    args = parser.parse_args()

    if args.out:
        with _tee(args.out):
            run_experiments()
    else:
        run_experiments()
