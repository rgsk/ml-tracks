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


def run_experiments():
    exp_table_dies()
    # (layers 2-6 to come)


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
