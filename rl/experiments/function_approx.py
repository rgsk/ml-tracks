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

    def true_V(self, gamma):
        """Exact V^pi for states 1..N by linear solve (ground truth for grading)."""
        N = self.N
        P = np.zeros((N, N))                            # non-terminal transitions
        r = np.zeros(N)
        for i, s in enumerate(range(1, N + 1)):
            for ns in (s - 1, s + 1):
                if ns == 0:
                    continue                            # left end: reward 0, absorbing
                if ns == N + 1:
                    r[i] += 0.5 * 1.0                   # right end: reward +1
                    continue
                P[i, ns - 1] += 0.5
        return np.linalg.solve(np.eye(N) - gamma * P, r)


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


def run_experiments():
    exp_table_dies()
    # exp_fa_is_regression()
    # (layers 3-6 to come)


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
