"""
WALKTHROUGH: Function approximation (exercise 5), built one layer at a time.

The hinge of the whole track. Exercises 2-4 varied two axes and froze a third:

    axis                         ex 2 (DP)    ex 3 (MC/TD)   ex 4 (Q/SARSA)
    ------------------------------------------------------------------------
    (a) know the model?            YES          no              no
    (b) how is the target formed?  full expect  G_t / TD(0)     TD(0) + max
    (c) where do the numbers LIVE? TABLE        TABLE           TABLE

Exercise 5 moves axis (c): instead of one free number per state, we keep WEIGHTS
and COMPUTE the value, v(s; w). Nothing about the Bellman equations changes — but
weights are SHARED across states, so an update at one state moves every similar
state too. That generalization is the only way to handle a continuous space, and
it's also what makes deep RL fragile. Everything after this exercise (DQN, PG,
PPO, RLHF) lives on this side of the line.

Layers (run each `exp_*`, read the output, then say "next"):

  1. THE TABLE DIES — CartPole states never repeat, and any binning is squeezed
     between "too coarse to see your own action" and "never revisited". (this layer)
  2. FA WITHOUT RL — fitting a known V by gradient descent is just regression;
     watch one update at one state move the value of every other state.
  3. THE TARGET — MC target (true SGD) vs TD target (SEMI-gradient), and what goes
     wrong if you take the FULL gradient through the bootstrap.
  4. TABULAR IS A SPECIAL CASE — one-hot features reproduce ex 3 exactly; coarsen
     them into state aggregation and the staircase bias appears.
  5. THE DEADLY TRIAD — Baird's counterexample diverges, then remove one leg at a
     time (bootstrap / off-policy / shared weights) and watch each ablation converge.
  6. NONLINEAR + CONTROL — a torch net doing naive online Q-learning on CartPole:
     it learns, spikes, and collapses. Measure why → the cliffhanger into DQN (ex 6).
"""

import argparse
import contextlib
import sys

import numpy as np

# ---------------------------------------------------------------------------
# LAYER 1: the table dies.
#
# The tabular methods of ex 2-4 all rested on one silent assumption: that you can
# INDEX the state. V[s] means s is a small integer, and learning means visiting
# each s many times. CartPole's state is four real numbers — there is no s to
# index, and (as this layer shows) you never see the same state twice, so the
# tabular update `V[s] += alpha * delta` would run exactly once per cell forever.
#
# The obvious dodge is to bin each dimension and call the cell "the state". This
# layer shows that dodge failing from BOTH sides at once:
#   - coarse bins  -> the state doesn't change for many steps, so the agent can't
#                     see the effect of its own action (aliasing);
#   - fine bins    -> bins^4 explodes and almost none of the cells are revisited.
# There is no good middle. That squeeze is the argument for approximation.
# ---------------------------------------------------------------------------

DIMS = ["x (cart pos)", "x_dot (cart vel)", "theta (pole angle)", "theta_dot (pole vel)"]

# The region we'd bin over. x and theta use CartPole's own termination limits
# (outside them the episode is already over, so there is nothing to store). The
# VELOCITIES are formally unbounded — the env declares [-inf, inf] — so any table
# forces us to invent a clip. That arbitrary choice is itself a cost of going tabular.
BOX = np.array([[-2.4, 2.4], [-3.0, 3.0], [-0.2095, 0.2095], [-3.0, 3.0]])


def _banner(*lines):
    print("=" * 76)
    for line in lines:
        print(line)
    print("=" * 76)


def rollout_states(num_episodes=200, seed=0):
    """Roll a random policy on CartPole. Returns (states, deltas): every state
    visited, and the per-step change |s' - s| that produced the next one."""
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    rng = np.random.default_rng(seed)
    states, deltas = [], []
    for ep in range(num_episodes):
        s, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            states.append(s)
            ns, _r, term, trunc, _ = env.step(int(rng.integers(2)))
            deltas.append(np.abs(ns - s))
            done = term or trunc
            s = ns
        states.append(s)
    return np.array(states), np.array(deltas)


def to_cell(states, bins):
    """Bin each dimension into `bins` buckets over BOX and flatten to ONE integer
    per state — i.e. force the continuous state back into a table index."""
    lo, hi = BOX[:, 0], BOX[:, 1]
    idx = ((states - lo) / (hi - lo) * bins).astype(int)
    idx = np.clip(idx, 0, bins - 1)                 # anything outside BOX piles up
    return np.ravel_multi_index(idx.T, (bins,) * 4)


def exp_table_dies(num_episodes=200, seed=0):
    """Four measurements on the same 200 random-policy episodes:
    A) do raw states ever repeat?   B) what region do they cover?
    C) how big is the binned table, and how much of it gets visited?
    D) how many steps does it take to LEAVE a cell? (the aliasing jaw of the vise)"""
    import gymnasium as gym

    states, deltas = rollout_states(num_episodes, seed)
    T = len(states)

    _banner("LAYER 1: the table dies — CartPole has no cell to index",
            f"  {num_episodes} random-policy episodes, {T:,} states visited")

    # --- A) exact repeats: the tabular precondition, tested --------------------
    uniq = len(np.unique(states, axis=0))
    print("\n  A) does any raw state EVER repeat?")
    print(f"       distinct raw states : {uniq:,} / {T:,}      repeats: {T - uniq}")
    print("       gridworld (ex 2-4)  : 11 states, each revisited thousands of times.")
    print("       V[s] += alpha*delta only learns through REPEATED visits to s. Here every")
    print("       state is new, so every table cell gets written exactly once. Nothing sticks.")

    # --- B) the region, and the unbounded dims --------------------------------
    space = gym.make("CartPole-v1").observation_space
    print("\n  B) what the state actually looks like")
    print("       dim                    | observed min | observed max | env bounds")
    print("     -------------------------+--------------+--------------+---------------")
    for i, name in enumerate(DIMS):
        bound = f"[{space.low[i]:.2f}, {space.high[i]:.2f}]"
        print(f"       {name:22s} |   {states[:, i].min():+7.3f}    |   "
              f"{states[:, i].max():+7.3f}    | {bound}")
    print("       -> two dims are formally UNBOUNDED. A table needs a finite grid, so we")
    print(f"          must invent a clip (here +-{BOX[1, 1]:.0f}). A function of s needs no grid.")

    # --- C) the combinatorics of binning --------------------------------------
    print("\n  C) force it into a table: bin every dim, cell = 'the state'")
    print("       bins/dim | table cells | (s,a) pairs | cells seen | coverage | visits/seen")
    print("     -----------+-------------+-------------+------------+----------+------------")
    for bins in (3, 5, 10, 20, 50):
        cells = bins ** 4
        c = to_cell(states, bins)
        counts = np.bincount(c)
        seen = int((counts > 0).sum())
        print(f"       {bins:5d}    | {cells:11,d} | {2 * cells:11,d} | {seen:10,d} |"
              f"  {100 * seen / cells:6.2f}% | {counts[counts > 0].mean():10.1f}")
    print(f"       ({T:,} transitions of data here. At 10 bins you'd need ~{2 * 10 ** 4 * 100:,}")
    print("        transitions just for 100 visits per (s,a) — and that is the SMALL grid.)")

    # --- D) the other jaw of the vise: coarse cells alias your own action ------
    print("\n  D) so use fewer bins? then the state stops MOVING between updates.")
    print("       one step changes the state by (mean |s' - s| per dim):")
    for i, name in enumerate(DIMS):
        print(f"         {name:22s} {deltas[:, i].mean():.4f}")
    print("\n       bins/dim | theta bin width | steps to cross one theta bin | mean steps")
    print("                |                 |    (bin width / mean step)   | inside a cell")
    print("     -----------+-----------------+------------------------------+--------------")
    for bins in (3, 5, 10, 20, 50):
        width = (BOX[2, 1] - BOX[2, 0]) / bins
        cross = width / deltas[:, 2].mean()
        c = to_cell(states, bins)
        changes = int((c[1:] != c[:-1]).sum())        # consecutive steps in same cell
        dwell = len(c) / max(changes, 1)
        print(f"       {bins:5d}    |     {width:.4f}      |            {cross:5.1f}"
              f"             |     {dwell:5.2f}")

    print("\n     Read the squeeze. Coarse (3-5 bins): the agent sits in the SAME cell for")
    print("     several steps, so s' == s and the TD error r + gamma*V[s'] - V[s] can hardly")
    print("     tell that the action did anything — state aliasing. Crank the bins up to fix")
    print("     that and section C explodes: 50 bins = 6.25M cells, 0.04% ever visited,")
    print("     with fewer than 2 visits each — every write is also the last.")
    print("     One knob, two failure modes, no good setting.")
    print("\n     That is why we stop storing a number per state and start storing WEIGHTS:")
    print("     v(s; w) returns a value for states never visited, and nearby states share")
    print("     it automatically. Next layer: what 'sharing' actually does to the numbers.")


# ---------------------------------------------------------------------------
# LAYER 2: function approximation with NO reinforcement learning in it.
#
# Before mixing in bootstrapping, strip the problem down: SUPPOSE someone handed
# you the true values V(s) for every state. Storing them is a table. FITTING them
# with a parameterized v(s; w) is ordinary supervised regression — least squares —
# and that is ALL "function approximation" means. Every RL-specific difficulty in
# the next layers comes from not having those true values, never from this step.
#
# The env: Sutton & Barto's 19-state random walk. States 1..19 in a line, step
# left/right with prob 1/2 each, terminate at 0 (reward 0) or 20 (reward +1). The
# policy is FIXED, so this is prediction, and V^pi is exactly solvable:
#       V(s) = 0.5*gamma*V(s-1) + 0.5*(r + gamma*V(s+1))
# a linear system (I - gamma*P) V = r — exercise 2's analytic policy value.
#
# Three things to see here:
#   A) 19 numbers compressed into 2 weights, exactly.
#   B) GENERALIZATION is a property of phi, not of RL: one update at ONE state
#      moves other states' values, and HOW FAR that spreads is your design choice.
#   C) CAPACITY: when the true V is not in the span of phi there is an error FLOOR.
#      The best point in the span is the PROJECTION of V onto it — remember that
#      word, layer 3 is about which point TD actually lands on.
# ---------------------------------------------------------------------------

class RandomWalk:
    """19-state random walk. Fixed uniform-random policy => PREDICTION, like ex 3.
    step() ignores its action, and true_V() solves the MDP exactly (no sampling)."""

    N = 19

    def __init__(self, rng=None):
        self.rng = rng
        self.s = None

    def reset(self):
        self.s = (self.N + 1) // 2                      # start in the middle
        return self.s

    def step(self, _a=None):
        self.s += int(self.rng.choice([-1, 1]))
        if self.s == 0:
            return self.s, 0.0, True
        if self.s == self.N + 1:
            return self.s, 1.0, True
        return self.s, 0.0, False

    def model(self):
        """The MDP as matrices over the N non-terminal states (row i = state i+1):
        P (N,N) interior transition probs, rbar (N,) expected immediate reward.
        Terminals are not states — their rows/cols are simply absent, which is the
        `done` convention: nothing to bootstrap off. Used ONLY for ground truth."""
        N = self.N
        P = np.zeros((N, N))
        rbar = np.zeros(N)
        for i, s in enumerate(range(1, N + 1)):
            for ns in (s - 1, s + 1):
                if ns == 0:
                    continue                            # left end: reward 0, episode over
                if ns == N + 1:
                    rbar[i] += 0.5 * 1.0                # right end: reward +1, episode over
                    continue
                P[i, ns - 1] += 0.5
        return P, rbar

    def true_V(self, gamma):
        """Exact V^pi for states 1..N by linear solve (ground truth for grading)."""
        P, rbar = self.model()
        return np.linalg.solve(np.eye(self.N) - gamma * P, rbar)

    def visit_dist(self):
        """mu(s): the ON-POLICY state distribution — expected visits to s per episode,
        normalized. d = e_start (I - P)^-1 sums P^t over all t. Every claim in layer 3
        is about accuracy weighted by THIS: states you visit often matter more."""
        P, _ = self.model()
        e = np.zeros(self.N)
        e[self.reset() - 1] = 1.0
        d = np.linalg.solve(np.eye(self.N) - P.T, e)
        return d / d.sum()


def make_phi(kind, N=RandomWalk.N):
    """Feature MATRIX (N, d): row i is phi(s) for state s = i+1. v(s;w) = phi(s) . w.
    The choice of phi is the choice of how much states share — the whole ballgame."""
    x = np.arange(1, N + 1) / (N + 1)                   # normalized position in (0,1)
    if kind == "const":
        return np.ones((N, 1))                          # one weight for the whole world
    if kind.startswith("poly"):
        deg = int(kind[4:])
        return np.stack([x ** k for k in range(deg + 1)], axis=1)
    if kind == "agg5":                                  # state aggregation: 5 groups
        g = np.minimum((np.arange(N) // 4), 4)
        return np.eye(5)[g]
    if kind == "rbf5":                                  # 5 gaussian bumps, width 0.125
        c = np.linspace(0.1, 0.9, 5)
        return np.exp(-((x[:, None] - c[None, :]) ** 2) / (2 * 0.125 ** 2))
    if kind == "onehot":
        return np.eye(N)                                # == the TABLE (layer 4)
    raise ValueError(kind)


def sgd_fit(Phi, targets, alpha, n_steps, rng):
    """Plain SGD least-squares fit of v(s;w) = Phi[s] . w to KNOWN targets.
    Sample a state uniformly, take one gradient step on (target - v)^2 / 2:
        delta = target[s] - Phi[s] . w
        w    += alpha * delta * Phi[s]        # grad_w v = Phi[s] for a linear v
    This is genuine SGD — the target is a constant, nothing to be 'semi' about yet.
    alpha is annealed linearly to 0: constant alpha would leave a noise floor and we
    want to compare the converged point against the exact projection."""
    w = np.zeros(Phi.shape[1])
    for t in range(n_steps):
        s = int(rng.integers(len(Phi)))
        w += alpha * (1 - t / n_steps) * (targets[s] - Phi[s] @ w) * Phi[s]
    return w


def _rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def exp_fa_is_regression(seed=0):
    """A) fit the true V with 2 weights instead of 19 numbers.
    B) one update at ONE state — how far does it spread, for four choices of phi?
    C) capacity: curve the true V (gamma<1) and watch an error FLOOR appear."""
    rng = np.random.default_rng(seed)
    env = RandomWalk()

    # --- A) 19 numbers -> 2 weights, by regression ----------------------------
    V = env.true_V(gamma=1.0)
    _banner("LAYER 2A: fitting a KNOWN V is just least squares",
            "  19-state random walk, gamma=1  =>  true V(s) = s/20 exactly")
    print(f"  check the exact solve against the closed form s/20: max diff "
          f"{np.abs(V - np.arange(1, 20) / 20).max():.2e}")
    print("\n  phi(s) = [1, s/20]  ->  2 weights for 19 states.")
    Phi = make_phi("poly1")
    print("   SGD steps |    w = [bias, slope]    | RMS vs true V")
    print("  -----------+-------------------------+---------------")
    w = np.zeros(2)
    prev = 0
    for n in (10, 100, 1000, 10000, 100000):
        for _ in range(n - prev):                       # keep fitting the same w
            s = int(rng.integers(len(Phi)))
            w += 0.05 * (V[s] - Phi[s] @ w) * Phi[s]    # sgd_fit's step, constant alpha
        prev = n
        print(f"  {n:9,d}  |  [{w[0]:+.4f}, {w[1]:+.4f}]     |    {_rms(Phi @ w, V):.5f}")
    print("\n  w -> [0, 1]: v(s) = 0*1 + 1*(s/20) = s/20. The 19-number table is now TWO")
    print("  numbers, with zero error. Note what got destroyed to buy that: the states are")
    print("  no longer independent parameters. Which is the point — and the danger.")

    # --- B) generalization is a property of phi -------------------------------
    _banner("LAYER 2B: ONE update at state 5 — where does it land?")
    print("  start from w = 0, do a single update toward the true V(5) = 0.25 (alpha=0.5),")
    print("  then print how much v(s) moved AT EVERY OTHER STATE.\n")
    s0 = 5 - 1                                          # state 5 -> row index 4
    kinds = ["onehot", "agg5", "rbf5", "poly1"]
    labels = {"onehot": "one-hot (= the TABLE)", "agg5": "agg5    (5 groups of 4)",
              "rbf5": "rbf5    (5 gaussian bumps)", "poly1": "poly1   ([1, s/20])"}
    shown = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 19]
    header = "  phi                       | " + " ".join(f"{s:5d}" for s in shown)
    print(header)
    print("  " + "-" * (len(header) - 2))
    dvs = {}
    for kind in kinds:
        P = make_phi(kind)
        dw = 0.5 * (V[s0] - 0.0) * P[s0]                # one SGD step from w = 0
        dv = P @ dw                                     # change in value at EVERY state
        dvs[kind] = dv
        print(f"  {labels[kind]:25s} | " + " ".join(f"{dv[s - 1]:+.3f}" for s in shown))
    print("\n  same delta, same alpha, four different worlds:")
    print("    one-hot : only state 5 moves. That IS the tabular update — no sharing, so")
    print("              19 states need 19 independent visits (layer 1's death sentence).")
    print("    agg5    : states 5-8 move by the SAME amount — they are indistinguishable")
    print("              to the approximator. Sharing by force = the ALIASING of layer 1.")
    print("    rbf5    : a smooth local bump — nearby states move a lot, far ones a little.")
    print("              Sharing by SIMILARITY: evidence spreads where it is plausible.")
    print("    poly1   : every state moves, including state 19 at the far end. Global")
    print("              sharing: cheap and fast, but one bad target perturbs everything.")
    print("\n  So 'generalization' is not something RL does — it is what you asked for when")
    print("  you picked phi. Aliasing (agg5) and generalization (rbf5) are the same")
    print("  mechanism at different widths; only the second one is a choice you control.")

    # --- C) capacity and the error floor --------------------------------------
    gamma = 0.9
    Vg = env.true_V(gamma)
    _banner(f"LAYER 2C: capacity — at gamma={gamma} the true V is CURVED, not a line")
    print("  discounting kills the far-away +1, so V bends. Now some phi's simply cannot")
    print("  represent it. 'Best possible' = the PROJECTION of V onto span(phi) (lstsq).\n")
    print("   phi     | weights | best possible RMS |  SGD reached  | gap to best")
    print("  ---------+---------+-------------------+---------------+------------")
    for kind in ("const", "poly1", "poly2", "poly3", "agg5", "rbf5", "onehot"):
        P = make_phi(kind)
        w_star = np.linalg.lstsq(P, Vg, rcond=None)[0]  # the projection
        w_sgd = sgd_fit(P, Vg, alpha=0.05, n_steps=200000, rng=rng)
        print(f"   {kind:7s} |   {P.shape[1]:3d}   |      {_rms(P @ w_star, Vg):.5f}      |"
              f"    {_rms(P @ w_sgd, Vg):.5f}    |   {_rms(P @ w_sgd, P @ w_star):.5f}")

    print("  (poly3 is the one row where SGD hasn't reached its projection: raw powers")
    print("   1,x,x^2,x^3 are nearly collinear, so that lstsq solution sits in a long")
    print("   narrow valley SGD crawls along. Conditioning of phi is a real cost too.)")

    print("\n  values at a few states (true vs what each phi can manage at BEST):")
    print("   state |  true V  |  const  |  poly1  |  poly2  |  agg5   | onehot")
    print("  -------+----------+---------+---------+---------+---------+---------")
    cols = ("const", "poly1", "poly2", "agg5", "onehot")
    best = {k: make_phi(k) @ np.linalg.lstsq(make_phi(k), Vg, rcond=None)[0] for k in cols}
    for s in (1, 5, 10, 15, 19):
        row = " | ".join(f"{best[k][s - 1]:+.4f}" for k in cols)
        print(f"    {s:3d}  | {Vg[s - 1]:+.4f}  | {row}")
    print("  note poly1 reports v(1) = -0.12: a NEGATIVE value in a walk where every")
    print("  return is 0 or +1. Fitting globally, it spends error where it must — an")
    print("  approximator will happily state things the true value function never could.")

    print("\n  Three lessons to carry into layer 3:")
    print("   1. SGD lands on the projection (gap-to-best is ~0, i.e. small next to the")
    print("      floor itself) — with TRUE targets, FA is solved. Regression, nothing more.")
    print("   2. The floor is set by phi alone, and it is a real constraint here: at")
    print("      gamma=0.9 the curve is steep, so each extra polynomial term only halves")
    print("      the error. onehot (the table) always hits 0 — the table is just the")
    print("      maximum-capacity, zero-generalization end of this same spectrum.")
    print("   3. We do NOT have the true V. Layer 3 replaces `targets` with something we")
    print("      can sample — and the moment that target contains w itself, this clean")
    print("      regression story breaks in a specific, nameable way.")


# ---------------------------------------------------------------------------
# LAYER 3: where does the target come from? (and what "SEMI-gradient" means)
#
# Layer 2 assumed a teacher handing us V(s). There is no teacher. We have samples,
# and exactly two ways to turn them into a regression target — the same two from
# exercise 3, now aimed at WEIGHTS instead of table cells:
#
#   MC :  target = G_t                     the return actually observed. Contains no
#                                          w at all => a genuine constant => plain SGD.
#   TD :  target = r + gamma * v(s'; w)    contains w. THIS is where it gets subtle.
#
# The TD update is written the same way regardless:
#       delta = target - v(s; w)
#       w    += alpha * delta * grad_w v(s; w)
# but note what we did NOT differentiate: the target. It contains w through v(s';w),
# so the true gradient of delta^2 has a second piece. We throw that piece away and
# pretend the target is a constant. That amputation is the entire meaning of the word
# "SEMI-gradient" (in torch: `.detach()` / `torch.no_grad()` around the target).
#
#       semi-gradient TD :  w += alpha * delta * phi(s)
#       FULL gradient    :  w += alpha * delta * (phi(s) - gamma * phi(s'))
#                                                          ^^^^^^^^^^^^^^^ the piece
#                                                          semi-gradient discards
#
# It looks like laziness. It is not: this layer measures four things and shows the
# amputated update is both FASTER and CLOSER to the truth than the honest one.
#
#   A) the two targets side by side: MC is unbiased but wildly noisy; TD is nearly
#      noise-free but is only as good as the w inside it. Bias/variance, made of numbers.
#   B) a step-size sweep: TD reaches a lower error AND tolerates ~30x bigger alpha.
#   C) they converge to DIFFERENT PLACES. MC lands on layer 2's projection (the best
#      point in span(phi)); semi-gradient TD lands on the "TD fixed point", which is
#      worse than the projection but provably not too much worse.
#   D) the FULL gradient converges too — to something worse than both. It minimizes
#      the Bellman ERROR, not the distance to V. And the naive sampled version doesn't
#      even reach that: it needs two independent successors per state (DOUBLE SAMPLING),
#      so with one sample it optimizes variance instead of value.
# ---------------------------------------------------------------------------

def run_episode(env, rng, start=None):
    """One sampled episode as a list of (s, r, s_next, done). Uses only reset/step —
    the model is never consulted (it exists in this file solely to grade the result)."""
    env.rng = rng
    s = env.reset() if start is None else start
    env.s = s
    out = []
    while True:
        ns, r, done = env.step()
        out.append((s, r, ns, done))
        if done:
            return out
        s = ns


def linear_prediction(kind, env, Phi, gamma, num_episodes, alpha, rng, anneal=True):
    """Linear prediction from SAMPLES. The three `kind`s differ in ONE line each —
    that line is the whole content of this layer.

        mc  : target is the observed return G_t.        w += a * (G - v(s)) * phi(s)
        td  : target bootstraps off v(s'), NOT differentiated (semi-gradient).
                                                        w += a * delta * phi(s)
        rg  : same delta, but differentiate the target too (full/'residual' gradient).
                                                        w += a * delta * (phi(s) - g*phi(s'))
    """
    w = np.zeros(Phi.shape[1])
    zero = np.zeros(Phi.shape[1])
    for ep in range(num_episodes):
        a = alpha * (1 - ep / num_episodes) if anneal else alpha
        traj = run_episode(env, rng)
        if kind == "mc":
            G = 0.0
            for (s, r, _ns, _done) in reversed(traj):       # walk backwards for G_t
                G = r + gamma * G                           # the FULL observed return
                w += a * (G - Phi[s - 1] @ w) * Phi[s - 1]
        else:
            for (s, r, ns, done) in traj:
                phi_next = zero if done else Phi[ns - 1]    # `done` => nothing to bootstrap
                delta = r + gamma * (phi_next @ w) - Phi[s - 1] @ w
                if kind == "td":
                    w += a * delta * Phi[s - 1]                        # semi-gradient
                else:
                    w += a * delta * (Phi[s - 1] - gamma * phi_next)   # full gradient
    return w


# --- the four points these algorithms can converge to, computed EXACTLY ------
# (all in the mu-weighted norm: error at a state counts as much as you visit it)

def projection(Phi, V, mu):
    """argmin_w ||Phi w - V||_mu — layer 2's best-possible point. MC converges here."""
    D = np.diag(mu)
    return np.linalg.solve(Phi.T @ D @ Phi, Phi.T @ D @ V)


def td_fixed_point(Phi, P, rbar, mu, gamma):
    """The w where the EXPECTED semi-gradient TD update is zero:
    Phi^T D (I - gamma P) Phi w = Phi^T D rbar. Not the projection — a different point."""
    D = np.diag(mu)
    A = Phi.T @ D @ (np.eye(len(P)) - gamma * P) @ Phi
    return np.linalg.solve(A, Phi.T @ D @ rbar)


def msbe_min(Phi, P, rbar, mu, gamma):
    """argmin of the mean-squared BELLMAN error ||(I - gamma P) Phi w - rbar||_mu.
    What the FULL gradient is honestly trying to minimize (given exact expectations)."""
    D = np.diag(mu)
    M = (np.eye(len(P)) - gamma * P) @ Phi
    return np.linalg.solve(M.T @ D @ M, M.T @ D @ rbar)


def naive_rg_min(Phi, env, mu, gamma):
    """argmin of E[(r + gamma v(s') - v(s))^2] over SINGLE sampled transitions — the
    thing the full-gradient update actually reaches. It differs from msbe_min by the
    variance of the successor: E[X^2] = (E X)^2 + Var X, and the update can shrink the
    loss by shrinking that Var — i.e. by flattening v where the future is uncertain.
    Killing that term needs TWO independent successors per update (DOUBLE SAMPLING)."""
    N = env.N
    rows, tgts, wts = [], [], []
    for i, s in enumerate(range(1, N + 1)):
        for ns in (s - 1, s + 1):                       # each sampled successor is a row
            done = ns in (0, N + 1)
            r = 1.0 if ns == N + 1 else 0.0
            phi_next = np.zeros(Phi.shape[1]) if done else Phi[ns - 1]
            rows.append(gamma * phi_next - Phi[i])      # residual = row @ w + r
            tgts.append(-r)
            wts.append(mu[i] * 0.5)
    A = np.array(rows) * np.sqrt(wts)[:, None]
    b = np.array(tgts) * np.sqrt(wts)
    return np.linalg.lstsq(A, b, rcond=None)[0]


def _murms(Phi, w, V, mu):
    """RMS error weighted by the on-policy visit distribution."""
    return float(np.sqrt(np.sum(mu * (Phi @ w - V) ** 2)))


def exp_the_target(seed=0):
    """A) MC vs TD targets: bias and variance, measured.
    B) step-size sweep — which method learns faster from the same episodes.
    C) they converge to DIFFERENT fixed points; both predicted exactly by theory.
    D) the FULL gradient: stable, honest, and worse. Plus the double-sampling trap."""
    env = RandomWalk()
    mu = env.visit_dist()
    P, rbar = env.model()

    # --- A) what the two targets look like as random variables -----------------
    gamma = 1.0
    V = env.true_V(gamma)
    rng = np.random.default_rng(seed)
    n = 20000
    _banner("LAYER 3A: the two targets, as random variables (gamma=1, 20k samples)",
            "  MC target = G_t (the whole observed episode) | TD target = r + gamma*v(s')")
    print("\n  first with a PERFECT bootstrap (v = true V), so both are unbiased and only")
    print("  the NOISE differs:")
    print("    s | true V |   MC target       |   TD target       | MC std / TD std")
    print("  ----+--------+-------------------+-------------------+----------------")
    for s0 in (5, 10, 15):
        mc_t, td_t = [], []
        for _ in range(n):
            traj = run_episode(env, rng, start=s0)
            mc_t.append(sum(r for (_s, r, _n, _d) in traj))      # gamma=1 => plain sum
            s, r, ns, done = traj[0]                             # ONE step for TD
            td_t.append(r + gamma * (0.0 if done else V[ns - 1]))
        mc_t, td_t = np.array(mc_t), np.array(td_t)
        print(f"   {s0:2d} |  {V[s0 - 1]:.2f}  | {mc_t.mean():+.3f} +- {mc_t.std():.3f} |"
              f" {td_t.mean():+.3f} +- {td_t.std():.3f} |      {mc_t.std() / td_t.std():.1f}x")
    print("\n  Both means hit the true value — both targets are unbiased WHEN the bootstrap")
    print("  is right. The spread is the story. At gamma=1 the return is literally Bernoulli")
    print("  (you end at +1 or 0), so MC's std is sqrt(V(1-V)) ~ 0.5: every single target is")
    print("  0 or 1, never the 0.5 you want. TD's target is r + V(s+-1) = V(s) +- 0.05, so its")
    print("  std is 0.05 — TD replaced 'the rest of the episode' with ONE step plus a stored")
    print("  estimate, and threw away all the randomness in between.")

    print("\n  now the catch — the same TD target with an UNTRAINED w = 0:")
    print("    s | true V | TD target (w=0)   | comment")
    print("  ----+--------+-------------------+---------------------------------")
    for s0 in (5, 10, 15, 19):
        td0 = []
        for _ in range(4000):
            _s, r, _ns, _done = run_episode(env, rng, start=s0)[0]
            td0.append(r + 0.0)                          # v(s'; 0) = 0 everywhere
        td0 = np.array(td0)
        note = "reward only fires next to the goal" if s0 == 19 else "target says 0. it is not 0."
        print(f"   {s0:2d} |  {V[s0 - 1]:.2f}  | {td0.mean():+.3f} +- {td0.std():.3f} | {note}")
    print("\n  Zero variance, maximum BIAS. That is the trade in one table: MC's noise comes")
    print("  from the sampled future, TD's error comes from trusting its own estimate. MC's")
    print("  shrinks with more episodes; TD's shrinks as w improves — it bootstraps itself up.")

    # --- B) which one actually learns faster ----------------------------------
    Phi = make_phi("poly1")                             # can represent V exactly at gamma=1
    _banner("LAYER 3B: step-size sweep, 1000 episodes, poly1 features, gamma=1",
            "  (mean on-policy RMS over 6 seeds; phi CAN represent V exactly here,",
            "   so both methods are aiming at the same target — only the ROUTE differs)")
    print("\n     alpha  |    MC    |    TD")
    print("   ---------+----------+---------")
    best = {"mc": (9, 0), "td": (9, 0)}
    for alpha in (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1):
        errs = {}
        for kind in ("mc", "td"):
            e = np.mean([_murms(Phi, linear_prediction(
                kind, env, Phi, gamma, 1000, alpha, np.random.default_rng(s), anneal=False),
                V, mu) for s in range(6)])
            errs[kind] = e
            if e < best[kind][0]:
                best[kind] = (e, alpha)
        print(f"    {alpha:.4f} |  {errs['mc']:.4f}  |  {errs['td']:.4f}")
    print(f"\n   best MC: {best['mc'][0]:.4f} at alpha={best['mc'][1]}     "
          f"best TD: {best['td'][0]:.4f} at alpha={best['td'][1]}")
    print(f"   TD is better at its best alpha AND that alpha is {best['td'][1] / best['mc'][1]:.0f}x"
          " larger. Why: MC feeds")
    print("   the SAME 0/1 return to every state in the episode, so with shared weights one")
    print("   lucky episode yanks the whole function; you must use tiny steps to survive it.")
    print("   TD's targets are small local corrections, so it can take big steps safely.")

    # --- C) the fixed points: MC and TD do not converge to the same w ----------
    gamma = 0.9
    Vg = env.true_V(gamma)
    Phi = make_phi("agg5")                              # deliberately CANNOT represent Vg
    w_proj = projection(Phi, Vg, mu)
    w_td = td_fixed_point(Phi, P, rbar, mu, gamma)
    _banner("LAYER 3C: same data, same features — DIFFERENT answers",
            "  gamma=0.9 + agg5 features (5 groups): V is NOT representable now, so",
            "  where each method settles becomes visible")
    print("  5000 episodes, alpha annealed to 0.\n")
    w_mc_run = linear_prediction("mc", env, Phi, gamma, 5000, 0.05, np.random.default_rng(2))
    w_td_run = linear_prediction("td", env, Phi, gamma, 5000, 0.05, np.random.default_rng(2))
    print("   method            | predicted by theory | actually reached | on-policy RMS")
    print("  -------------------+---------------------+------------------+--------------")
    print(f"   MC  (true SGD)    |  the PROJECTION     |     matches      |   "
          f"{_murms(Phi, w_mc_run, Vg, mu):.5f}   (theory {_murms(Phi, w_proj, Vg, mu):.5f})")
    print(f"   TD  (semi-grad)   |  the TD FIXED POINT |     matches      |   "
          f"{_murms(Phi, w_td_run, Vg, mu):.5f}   (theory {_murms(Phi, w_td, Vg, mu):.5f})")
    print(f"\n   w from MC run : {np.round(w_mc_run, 4)}")
    print(f"   w projection  : {np.round(w_proj, 4)}   <- MC's destination")
    print(f"   w from TD run : {np.round(w_td_run, 4)}")
    print(f"   w TD fixedpt  : {np.round(w_td, 4)}   <- TD's destination")
    ratio = _murms(Phi, w_td, Vg, mu) / _murms(Phi, w_proj, Vg, mu)
    print(f"\n   TD's answer is {ratio:.2f}x worse than the best point in span(phi). It is NOT")
    print("   minimizing distance to V — it is solving its own Bellman-shaped equation, and")
    print("   the bootstrap drags it off the projection. The classic bound says it can't be")
    print(f"   arbitrarily bad: TD error <= 1/(1-gamma) x best error = "
          f"{1 / (1 - gamma):.0f}x here (we got {ratio:.2f}x).")
    print("   That 1/(1-gamma) is why long-horizon problems make people nervous.")

    # --- D) the full gradient: honest, stable, and worse -----------------------
    w_be = msbe_min(Phi, P, rbar, mu, gamma)
    w_rg_theory = naive_rg_min(Phi, env, mu, gamma)
    w_rg_run = linear_prediction("rg", env, Phi, gamma, 5000, 0.05, np.random.default_rng(2))
    _banner("LAYER 3D: so why not take the FULL gradient?")
    print("  Same setup, but differentiate the target too:")
    print("     w += alpha * delta * (phi(s) - gamma * phi(s'))     <- 'residual gradient'")
    print("  This IS honest gradient descent on delta^2, so it always converges. Look where.\n")
    print("   where a method lands                          | on-policy RMS vs true V")
    print("  ----------------------------------------------+------------------------")
    for label, w in (("projection (best possible in span(phi))", w_proj),
                     ("MC run                                ", w_mc_run),
                     ("semi-gradient TD run                  ", w_td_run),
                     ("full-gradient (residual) run          ", w_rg_run),
                     ("  ...its ideal target: min Bellman err", w_be),
                     ("  ...what one sample actually reaches ", w_rg_theory)):
        print(f"   {label:44s} |        {_murms(Phi, w, Vg, mu):.5f}")
    print("\n  Two separate problems, both visible above:")
    print("   1. WRONG OBJECTIVE. Even with exact expectations, minimizing the Bellman error")
    print(f"      lands at {_murms(Phi, w_be, Vg, mu):.5f} — much worse than semi-gradient TD's "
          f"{_murms(Phi, w_td_run, Vg, mu):.5f}. Small")
    print("      Bellman residual does not mean close to V; you can trade a lot of value")
    print("      accuracy for a slightly more self-consistent (but wrong) function.")
    print("   2. DOUBLE SAMPLING. The run lands at "
          f"{_murms(Phi, w_rg_run, Vg, mu):.5f}, matching "
          f"{_murms(Phi, w_rg_theory, Vg, mu):.5f} — NOT its own")
    print("      ideal. delta appears TWICE in the gradient, so a single sampled s' gives a")
    print("      biased product: E[delta * delta] = (E delta)^2 + Var(delta). The update")
    print("      happily shrinks that variance term by FLATTENING v wherever the future is")
    print("      uncertain. Fixing it needs two independent successors from the same state —")
    print("      free in a simulator you can reset, impossible from a stream of experience.")
    print("\n  So 'semi-gradient' is not a shortcut people apologize for. It converges faster,")
    print("  it lands closer to V, and it needs only one sample per transition. Everything")
    print("  from here — DQN, A2C, PPO, the value head in RLHF — is a semi-gradient method,")
    print("  and `.detach()` on the target is where you will see it in torch.")
    print("\n  The bill comes due in layer 5: semi-gradient TD is not descending on ANY fixed")
    print("  objective, so nothing guarantees it converges at all. On-policy it does. Add")
    print("  off-policy data and shared weights and it can diverge — the deadly triad.")


def run_experiments():
    # exp_table_dies()
    # exp_fa_is_regression()
    exp_the_target()
    # (layers 4-6 to come)


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
