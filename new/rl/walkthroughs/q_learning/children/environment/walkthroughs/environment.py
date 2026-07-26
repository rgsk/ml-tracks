"""
CHILD WALKTHROUGH (digs into q_learning exp_2): the ENVIRONMENT, top-down.

The parent q_learning.py built an agent — a table, ε-greedy behaviour, one update line — and watched
it solve a gridworld. This box opens the OTHER half of that picture, the half we waved at as "the
world": the thing behind `reset()` and `step(a)`.

It matters more than it looks. The agent has no goals, no preferences, no idea what a wall is; every
bit of that lives in the environment. So the environment isn't scenery — it IS the problem statement:

    an MDP is the tuple (S, A, P, R, γ), and it defines what "good behaviour" even MEANS.
    change one entry and the correct policy changes with it — the agent's code doesn't move at all.

Top-down: before any formalism, SEE that. exp_1 takes the parent's agent VERBATIM and runs it on four
worlds that differ only in P (how slippery) and R (what a step costs), and the learned behaviour comes
out completely different each time — including one world where the right move is to end the episode in
the -1 trap on purpose. Run it with `python environment.py` (`exp_1_whole_game`).

Layers (each an `exp_*`; run it, read the output, then say "next"):
  1. the WHOLE GAME     — same agent, four worlds: turn the P dial and the R dial, watch the optimal
                          behaviour flip. The environment is the problem.                      (here)
  then open the boxes — one entry of the tuple at a time:
  2. THE FORMAL OBJECT  — (S, A, P, R, γ) read straight off this gridworld: print P[s][a] and find the
                          0.8/0.1/0.1; the MARKOV property, and what has to be in a state for it to
                          hold (the thing that goes wrong first in real problems).
  3. THE BLACK BOX      — reset/step vs P: the samples ARE the model. Estimate P̂ from experience and
                          watch it converge to the true P — the bridge to every model-free method.
  4. THE RETURN G_t     — episodes, trajectories, and G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …; what the
                          agent is actually maximizing, and why it's a SUM over the future.
  5. γ, THE PATIENCE DIAL — sweep γ and watch the route change; bounded return ≤ r_max/(1-γ); why an
                          infinite horizon needs discounting at all.
  6. REWARD DESIGN      — the -0.04 is a CHOICE. Shaping, and reward hacking in miniature (the agent
                          optimizes what you wrote, not what you meant) — the seed of RLHF's KL leash.
  7. TERMINATION vs TRUNCATION — absorbing states, the 200-step cap, and the classic bug: bootstrapping
                          through a time limit as if the world had really ended.

Uses the parent's agent (`q_learning`, `evaluate`, `epsilon_greedy`) unchanged — this box changes the
WORLD, never the learner — and the shared GridWorld from new/rl/envs.py.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))              # .../environment/walkthroughs
# walk up:  walkthroughs -> environment -> children -> q_learning (the parent walkthrough)
_PARENT = os.path.abspath(os.path.join(_HERE, *([".."] * 3)))    # .../walkthroughs/q_learning
_RL = os.path.abspath(os.path.join(_HERE, *([".."] * 5)))        # new/rl (holds envs.py)
_FIGS = os.path.join(_HERE, "figures", "experiments")
sys.path[:0] = [_RL, _PARENT]

from envs import ARROWS, GridWorld                              # noqa: E402
from q_learning import evaluate, epsilon_greedy, q_learning      # noqa: E402  (the parent's agent)


def _banner(*lines):
    print("=" * 100)
    for line in lines:
        print(line)
    print("=" * 100)


# The four worlds. Same grid, same start, same terminals, same agent — only P and R move.
WORLDS = [
    ("the world from exp_1", dict(noise=0.2, step_reward=-0.04),
     "80% you go where you aimed; a step costs almost nothing"),
    ("dry floor (P)", dict(noise=0.0, step_reward=-0.04),
     "no slip at all: actions do exactly what they say"),
    ("black ice (P)", dict(noise=0.6, step_reward=-0.04),
     "only 40% you go where you aimed; 30% each side"),
    ("harsh living cost (R)", dict(noise=0.2, step_reward=-2.0),
     "every step costs -2.0 — being alive is expensive"),
]


def intended_route(env, Q, max_len=12):
    """The cells the greedy policy MEANS to walk through from the start (slip ignored).

    Not what happens — the floor is slippery — but the plan the table encodes, which is what we want
    to compare across worlds. Returns (route, aims_at_wall): a policy that aims into a wall goes
    nowhere on paper, and on black ice that turns out to be deliberate."""
    cell, route = env.start, [env.start]
    for _ in range(max_len):
        if cell in env.terminals:
            break
        nxt = env.intended_move(cell, int(Q[env.s2i[cell]].argmax()))
        if nxt == cell:                                          # aims into a wall/edge: stays put
            return route, True
        cell = nxt
        route.append(cell)
    return route, False


def exp_1_whole_game(seed=0, episodes=4000, alpha=0.1):
    """The whole game for this box: the environment IS the problem. Run the parent's agent, untouched,
    on four worlds that differ only in P (slip) and R (step cost), and watch the learned policy come
    out different every time. No formalism yet — exp_2 onward name the pieces we're turning."""
    _banner("EXP 1: the whole game — same agent, four worlds. The environment is the problem.")

    print("  The agent below is the parent's q_learning() imported verbatim: 44 numbers, ε-greedy,")
    print("  one update line. NOTHING about it changes across the four runs. All that changes is the")
    print("  world behind reset()/step(a) — how slippery the floor is (P), and what a step costs (R).\n")

    results = []
    for name, kw, blurb in WORLDS:
        env = GridWorld(seed=seed, **kw)
        Q, hist, _ = q_learning(env, episodes, alpha=alpha, seed=seed)
        ev = evaluate(env, lambda s, rng: epsilon_greedy(Q, s, 0.0, rng))
        route, wall = intended_route(env, Q)
        results.append((name, kw, blurb, env, Q, ev, route, wall))

        print("-" * 100)
        print(f"  {name}   (noise={kw['noise']}, step_reward={kw['step_reward']})")
        print(f"    {blurb}\n")
        print("    " + env.render(lambda s, cell: ARROWS[int(Q[s].argmax())]).replace("\n", "\n    "))
        print(f"\n    learned route from S:  {' -> '.join(str(c) for c in route)}"
              f"{'  then AIMS INTO A WALL (deliberately — see below)' if wall else ''}")
        print(f"    greedy performance:    mean return {ev['ret']:+7.3f}   "
              f"reached +1 in {ev['win']:5.1f}%   {ev['steps']:.1f} steps/episode")

    # ---- the payoff: put the four side by side ------------------------------------------------
    print("-" * 100)
    print("\n  FOUR WORLDS, ONE AGENT:\n")
    print(f"    {'world':<24}{'noise':>7}{'step R':>8}   {'route from S':<26}{'return':>9}{'win %':>8}{'steps':>8}")
    for name, kw, _b, env, Q, ev, route, wall in results:
        short = " ".join(f"{c[0]}{c[1]}" for c in route) + ("  ⊣wall" if wall else "")
        print(f"    {name:<24}{kw['noise']:>7}{kw['step_reward']:>8}   {short:<26}"
              f"{ev['ret']:>+9.3f}{ev['win']:>8.1f}{ev['steps']:>8.1f}")

    print("\n  Same code, four different answers — and each dial does something DIFFERENT:")
    print("    · exp_1's world  — up the left column, then right along the top: 5 moves, and the one")
    print("      that keeps the most distance from the -1. Baseline: +0.75, wins 98.8%, 6.7 steps.")
    print("    · dry floor (P)  — turn the slip OFF and the plan doesn't move at all: the same route,")
    print("      start to finish. What changes is what that plan is WORTH — 5.0 steps instead of 6.7")
    print("      (no slips to undo), 100% instead of 98.8%, +0.84 instead of +0.75. A dial can change")
    print("      the VALUE without changing the POLICY. (The whole optimal policy is in fact identical")
    print("      in these two worlds; proving that needs exp_3's exact values. From learned tables you")
    print("      can only trust the route — off-route cells are the near-ties the parent exp_1 flagged.)")
    print("    · black ice (P)  — crank the slip to 60% and the policy IS rewritten. Return collapses")
    print("      to +0.27 and it takes 19.2 steps to travel 5 cells, yet it still wins 100%: it stops")
    print("      trying to make progress and starts avoiding drift (the cell below).")
    print("    · harsh cost (R) — the shocker: it dives into the -1 on purpose. The trap is 4 moves")
    print("      away and the goal is 5, so at -2.0 a step the nearest EXIT wins. Return -8.82. The 11%")
    print("      that still end on +1 aren't a change of heart: a slip carries it into the top row,")
    print("      where +1 becomes the nearest exit (note the top row still points at it). That is")
    print("      CORRECT: you wrote 'existing is worse than the worst outcome' into R, and it believed you.")

    # ---- the black-ice detail: aiming at a wall on purpose -------------------------------------
    ice = next(r for r in results if r[0].startswith("black ice"))
    env_i, Q_i = ice[3], ice[4]
    s = env_i.s2i[(1, 2)]
    print("\n  Worth staring at: on black ice, cell (1,2) — directly LEFT of the -1 trap:\n")
    for a, nm in enumerate(("up", "right", "down", "left")):
        mark = "  <- greedy" if a == Q_i[s].argmax() else ""
        print(f"      {ARROWS[a]} {nm:<6} {Q_i[s, a]:+.3f}{mark}")
    print("    'left' aims straight into the wall at (1,1). It makes NO progress by design — but the")
    print("    two ways it can slip are UP and DOWN, so the trap becomes literally unreachable this")
    print("    step. Aiming 'up' (toward the goal row) would slip RIGHT into the -1 30% of the time.")
    print("    When the floor is that unreliable, not-drifting beats advancing.")
    print("\n  Nobody edited the agent. Every one of those behaviours was written in the environment.")

    _figure(results)

    print("\n  So the environment is not scenery: it's the problem statement, and 'optimal' is only")
    print("  defined relative to it. Next (exp_2): name the pieces we just turned — the MDP tuple")
    print("  (S, A, P, R, γ) read straight off this gridworld, and the Markov property it rests on.")


def _figure(results):
    """Payoff figure: the learned policy + intended route in each of the four worlds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, (name, kw, _b, env, Q, ev, route, wall) in zip(axes, results):
        vals = np.array([Q[s].max() for s in range(env.nS)])
        lim = max(1e-6, np.abs(vals).max())
        grid = np.full((env.rows, env.cols), np.nan)
        for s, cell in enumerate(env.states):
            grid[cell] = Q[s].max()
        ax.imshow(grid, cmap="RdYlGn", vmin=-lim, vmax=lim)
        for (r, c) in env.walls:
            ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="0.4"))
        # the intended route, drawn on top
        ys, xs = [c[0] for c in route], [c[1] for c in route]
        ax.plot(xs, ys, color="black", lw=2.5, alpha=.35, solid_capstyle="round", zorder=1)
        for cell, val in env.terminals.items():
            ax.text(cell[1], cell[0], f"{val:+.0f}", ha="center", va="center",
                    fontsize=13, weight="bold", zorder=3)
        for s, cell in enumerate(env.states):
            if cell not in env.terminals:
                txt = ARROWS[int(Q[s].argmax())] if Q[s].max() != Q[s].min() else "?"
                ax.text(cell[1], cell[0], txt, ha="center", va="center", fontsize=17, zorder=3)
        ax.text(env.start[1], env.start[0] + .34, "S", ha="center", va="center",
                fontsize=9, color="0.25", zorder=3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{name}\nnoise={kw['noise']}, step R={kw['step_reward']}\n"
                     f"return {ev['ret']:+.2f}, +1 in {ev['win']:.0f}%", fontsize=9)

    fig.suptitle("Same agent, four environments — the policy is a property of the WORLD, "
                 "not just the learner", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "01_four_worlds.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out}")
    print("    arrows = learned greedy policy, colour = max_a Q[s,a], grey line = the route it means")
    print("    to walk from S. Four panels, one agent, four different ideas of 'correct'.")


# ---------------------------------------------------------------------------
# exp_2 — THE FORMAL OBJECT. Two bites: (a) read the tuple (S,A,P,R,γ) off this gridworld, and
# (b) the assumption hiding inside P — the MARKOV property — measured, broken, and repaired.
# ---------------------------------------------------------------------------
class Relabelled:
    """Same world, but every state and action gets a random new NAME (index).

    Nothing about the problem changes — only the labels. Used to show that S and A are index sets to
    the agent, carrying no geometry: it has no idea that state 7 is 'next to' state 8, or that action
    0 means 'up'."""

    def __init__(self, env, seed=0):
        rng = np.random.default_rng(seed)
        self.env = env
        self.s_perm = rng.permutation(env.nS)
        self.a_perm = rng.permutation(env.nA)
        self.nS, self.nA, self.gamma = env.nS, env.nA, env.gamma

    def reset(self):
        return int(self.s_perm[self.env.reset()])

    def step(self, a):
        s, r, done = self.env.step(int(self.a_perm[a]))
        return int(self.s_perm[s]), r, done


def exp_2a_the_tuple(seed=0, episodes=4000):
    """(S, A, P, R, γ) read straight off this gridworld — including what P[s][a] literally contains,
    and the fact that S and A are nothing but labels to the agent."""
    _banner("EXP 2a: the formal object — (S, A, P, R, γ) read off this gridworld")

    env = GridWorld(seed=seed)
    print("  S — the STATE SPACE. 3x4 cells minus the wall = 11 states. The agent gets an integer;")
    print("  the (row, col) is our bookkeeping, not something it can see:\n")
    print("   " + env.render(lambda s, cell: str(s)).replace("\n", "\n   "))
    print(f"\n  (the two terminals are states {env.s2i[(0, 3)]} and {env.s2i[(1, 3)]}; the grid shows their reward instead)")
    print(f"\n  A — the ACTION SPACE: {env.nA} integers. 'up/right/down/left' is a label we chose;")
    print("  the agent only ever sees 0,1,2,3 and finds out what they do by trying them.")
    print(f"\n  So the table from exp_1 has exactly |S|x|A| = {env.nS}x{env.nA} = {env.nS * env.nA} entries —")
    print("  one score per (state, action) pair. That's the whole memory of a tabular agent.")

    # --- P: the transition distribution ---------------------------------------------------------
    print("\n  P — the TRANSITION MODEL: p(s' | s, a), one distribution per (s, a) pair. Here it is")
    print("  for the start cell, action 'up' — this is where the 80/10/10 lives:\n")
    for cell, a, aname in (((2, 0), 0, "up"), ((2, 3), 0, "up")):
        s = env.s2i[cell]
        print(f"    P[s={s} {cell}][a={a} {aname}]:")
        for prob, ns, r, done in sorted(env.P[s][a], key=lambda o: -o[0]):
            nxt = env.states[ns]
            why = ""
            if nxt == cell:
                why = "   (slipped into a wall/edge -> stay put)"
            elif nxt in env.terminals:
                why = "   (TERMINAL)"
            print(f"        prob {prob:.1f} -> s'={ns:>2} {nxt}   r={r:+.2f}   done={done!s:<5}{why}")
        r_sa = sum(p * r for p, _ns, r, _d in env.P[s][a])
        print(f"      expected immediate reward r(s,a) = Σ p·r = {r_sa:+.3f}\n")
    bad = [(s, a) for s in range(env.nS) for a in range(env.nA)
           if abs(sum(p for p, *_ in env.P[s][a]) - 1.0) > 1e-12]
    print(f"    all {env.nS * env.nA} distributions sum to 1: {'✅' if not bad else f'❌ {bad}'}")
    print("    Note the 0.1 that comes back to (2,0): two different outcomes can land in the SAME")
    print("    state, so they merge into one entry. P is over STATES, not over intentions.")

    # --- R and γ ---------------------------------------------------------------------------------
    print(f"\n  R — the REWARD FUNCTION. Here reward depends on where you LAND, so it's R(s,a,s'):")
    print(f"    {env.step_reward:+.2f} for landing on any ordinary cell,  {'+1.00'} / {'-1.00'} for the two terminals.")
    print("    Terminals are absorbing and pay 0 forever after, which is why their value is 0.")
    print(f"\n  γ — the DISCOUNT: {env.gamma}. It's part of the PROBLEM, not the algorithm — it says how")
    print("    much a reward one step later is worth. exp_5 turns this dial and the route changes.")

    # --- the payoff: S and A are just labels ----------------------------------------------------
    print("\n  PAYOFF — if S and A really are just labels, then RENAMING them can't matter. Let's")
    print("  randomly permute all 11 state ids and all 4 action ids and retrain the same agent:\n")
    Q, _h, _s = q_learning(env, episodes, seed=seed)
    base = evaluate(env, lambda s, rng: epsilon_greedy(Q, s, 0.0, rng))
    shuf = Relabelled(GridWorld(seed=seed), seed=1)
    Qr, _h, _s = q_learning(shuf, episodes, seed=seed)
    rel = evaluate(shuf, lambda s, rng: epsilon_greedy(Qr, s, 0.0, rng))
    print(f"    original labels:  mean return {base['ret']:+.3f}   reached +1 in {base['win']:.1f}%")
    print(f"    shuffled labels:  mean return {rel['ret']:+.3f}   reached +1 in {rel['win']:.1f}%")
    print("    Identical (up to sampling noise). The agent never used the geometry — it can't; it")
    print("    only knows 'from label 7, action 2 tended to pay'. Every bit of structure in this")
    print("    problem lives in P and R. That is ALSO the tabular agent's fatal flaw: nothing is")
    print("    shared between neighbouring states, so it must visit each of the 44 pairs itself.")
    print("    (Rung 2 fixes exactly that by replacing the table with a network.)")

    print("\n  Next (exp_2b): the assumption hiding inside 'p(s'|s,a)' — the Markov property.")


class InertiaGrid:
    """The SAME grid, but the floor has INERTIA: turning is harder than carrying on.

    You still pick one of the four actions, but whether the world obeys depends on how sharply you are
    asking it to turn from the direction you are already travelling:

        `obey = (along, perpendicular, reverse)`  probabilities of your action being honoured;
        otherwise you are carried on in your current direction. Then the usual small slip applies.

    Built to break the Markov property on purpose — and to break it in a way the agent could ACT on:
    the best action here genuinely depends on which way you are already moving, which the cell index
    doesn't tell you. `augment=True` puts that back in the state, as (cell, current direction).
    """

    def __init__(self, obey=(0.98, 0.5, 0.05), noise=0.1, step_reward=-0.04, gamma=0.9,
                 seed=None, augment=False):
        self.grid = GridWorld(noise=noise, step_reward=step_reward, gamma=gamma)   # geometry + rewards
        self.obey, self.noise, self.step_reward, self.gamma = obey, noise, step_reward, gamma
        self.augment = augment
        self.nA = 4
        self.n_cells = self.grid.nS
        self.nS = self.n_cells * 5 if augment else self.n_cells      # 5th "direction" = just reset
        self.rng = np.random.default_rng(seed)
        self.cell_i, self.last = None, None

    def _obs(self):
        """What the agent is shown: either the bare cell, or the cell PLUS its current direction."""
        if not self.augment:
            return self.cell_i
        return self.cell_i * 5 + (4 if self.last is None else self.last)

    def reset(self):
        self.cell_i, self.last = self.grid.s2i[self.grid.start], None
        return self._obs()

    def step(self, a: int):
        cell = self.grid.states[self.cell_i]
        if self.last is None:                                     # standing still: no inertia yet
            eff = a
        else:
            rel = (a - self.last) % 4                             # 0 along, 1/3 perpendicular, 2 reverse
            p_obey = {0: self.obey[0], 1: self.obey[1], 3: self.obey[1], 2: self.obey[2]}[rel]
            eff = a if self.rng.random() < p_obey else self.last  # else: carried on
        actual = int(self.rng.choice([eff, (eff + 1) % 4, (eff - 1) % 4],
                                     p=[1 - self.noise, self.noise / 2, self.noise / 2]))
        nxt = self.grid.intended_move(cell, actual)
        self.cell_i, self.last = self.grid.s2i[nxt], actual
        reward = self.grid.terminals.get(nxt, self.step_reward)
        return self._obs(), reward, nxt in self.grid.terminals


def _next_cell_dist(env, cell, arrived_heading, action, trials=20000):
    """Empirical p(next cell | this cell, `action`) when you arrived travelling `arrived_heading`.

    Forces the env into (cell, last=arrived_heading) and samples one step, many times. This is exactly
    the quantity the Markov property claims cannot depend on the history."""
    counts = np.zeros(env.n_cells)
    for _ in range(trials):
        env.reset()
        env.cell_i, env.last = env.grid.s2i[cell], arrived_heading
        obs, _r, _d = env.step(action)
        counts[obs // 5 if env.augment else obs] += 1
    return counts / trials


BUDGETS = (2500, 5000, 10000, 20000)


def exp_2b_markov(seed=0, obey=(0.98, 0.5, 0.05), seeds=(0, 1, 2)):
    """The assumption inside p(s'|s,a): the MARKOV property. State it, watch it hold in our gridworld,
    break it with a floor that has inertia, and then repair it — by changing the STATE, not the
    algorithm. ~20 s (it trains 6 tables)."""
    _banner("EXP 2b: the Markov property — the assumption inside P, and what it costs when it's false")

    print("  The claim packed into 'p(s' | s, a)' is that the pair (s, a) is ENOUGH — given where you")
    print("  are and what you do, the past adds nothing:")
    print("      p(s_t+1 | s_t, a_t, s_t-1, a_t-1, ...)  =  p(s_t+1 | s_t, a_t)")
    print("  In our gridworld that's true BY CONSTRUCTION: look at GridWorld.step() in envs.py — it")
    print("  reads self.s and a and nothing else. There is no history for it to read. That's what the")
    print("  Markov property looks like as code, and it's why a table indexed by (s, a) can work.\n")

    print("  So let's break it. InertiaGrid: same cells, same rewards, but the floor has INERTIA —")
    print(f"  your action is honoured with prob {obey[0]} if it continues the way you're already going,")
    print(f"  {obey[1]} if it's a perpendicular turn, and only {obey[2]} if it's a reversal. Otherwise you're")
    print("  carried on. Nothing here reads the cell index; it reads your DIRECTION OF TRAVEL.\n")

    # --- (i) measure the property, holding and broken ------------------------------------------
    cell, probe = (2, 2), 0                                        # stand at (2,2) and aim UP
    print(f"  Stand in cell {cell}, aim 'up' 20,000 times — but arrive there two different ways:\n")
    rows = []
    for label, kw in (("plain gridworld (obey always)", dict(obey=(1.0, 1.0, 1.0))),
                      (f"inertia grid {obey}", dict(obey=obey))):
        env = InertiaGrid(seed=seed, **kw)
        d_along = _next_cell_dist(env, cell, 0, probe)              # arrived heading UP
        d_turn = _next_cell_dist(env, cell, 1, probe)               # arrived heading RIGHT
        tv = 0.5 * np.abs(d_along - d_turn).sum()
        rows.append((label, d_along, d_turn, tv))
        keep = sorted(range(env.n_cells), key=lambda i: -max(d_along[i], d_turn[i]))[:4]
        print(f"    {label}:")
        print(f"      {'landed in':<14}{'arrived heading UP':>20}{'arrived heading RIGHT':>23}")
        for i in keep:
            print(f"      {str(env.grid.states[i]):<14}{d_along[i]:>20.3f}{d_turn[i]:>23.3f}")
        print(f"      total-variation distance between the two distributions: {tv:.3f}\n")

    print("  Top block: identical to within sampling noise. The history left no trace, so the cell")
    print("  alone IS a Markov state. Bottom block: same cell, same action, and yet a completely")
    print("  different future — arriving sideways, half your 'up' requests get overruled and you sail")
    print("  on to (2,3) instead. p(s'|s,a) isn't just unknown here, it isn't well DEFINED: any number")
    print("  you write in a cell-indexed table is an average over histories you can't see.\n")

    # --- (ii) what it costs, and the repair ----------------------------------------------------
    print("  What does that cost? Train the parent's Q-learning on the inertia world two ways — same")
    print(f"  algorithm, same hyperparameters, {len(seeds)} seeds each. The only difference is what we call a")
    print("  STATE. Evaluated at four training budgets (mean return over seeds ± spread):\n")
    runs = {}
    print(f"    {'episodes trained':<36}   " + "  ".join(f"{b:>12,}" for b in BUDGETS))
    for label, augment in (("state = cell            (11 states)", False),
                           ("state = (cell, heading) (55 states)", True)):
        per_budget = {b: [] for b in BUDGETS}
        for sd in seeds:
            env = InertiaGrid(obey=obey, seed=sd, augment=augment)
            _Q, _h, snaps = q_learning(env, BUDGETS[-1], alpha=0.05, seed=sd, snapshots=BUDGETS)
            for b in BUDGETS:
                ev = evaluate(InertiaGrid(obey=obey, seed=100 + sd, augment=augment),
                              lambda s, rng: epsilon_greedy(snaps[b], s, 0.0, rng), episodes=1000)
                per_budget[b].append(ev["ret"])
        runs[label] = {b: (float(np.mean(v)), float(np.std(v))) for b, v in per_budget.items()}
        cells = "  ".join(f"{runs[label][b][0]:+.3f}±{runs[label][b][1]:.3f}" for b in BUDGETS)
        print(f"    {label}   {cells}")

    a = runs["state = cell            (11 states)"][BUDGETS[-1]][0]
    b = runs["state = (cell, heading) (55 states)"][BUDGETS[-1]][0]
    flat = runs["state = cell            (11 states)"]
    print(f"\n    Two things to read off that table:")
    print(f"    1. The cell-only agent PLATEAUS: {flat[BUDGETS[1]][0]:+.3f} at {BUDGETS[1]:,} episodes and "
          f"{flat[BUDGETS[-1]][0]:+.3f} at {BUDGETS[-1]:,}.")
    print("       Four times the experience buys nothing. This is a CEILING, not slow learning: the")
    print("       information it needs is not in what it's being shown, and no amount of data adds it.")
    print(f"    2. Telling it which way it's travelling is worth {b - a:+.3f} return — and it wins at EVERY")
    print("       budget, not just the last one. Nothing about the learner changed. We changed the state.")
    print("\n    MARKOV IS A PROPERTY OF YOUR STATE REPRESENTATION, NOT OF THE WORLD. If the past")
    print("    matters, your state is missing something — put it in.")
    print("    Honest note: 5x the states normally costs data (that's why nobody stacks 1000 frames),")
    print("    and here it visibly doesn't — the drift-aware agent is ahead even at 2,500 episodes.")
    print("    Its targets are CONSISTENT, and on a world this small that beats having fewer entries.")
    print("    This is the same move as Atari DQN stacking 4 frames so velocity is in the state; when")
    print("    you can't put the missing piece in (poker, dialogue), you need memory instead — a POMDP.")

    _figure_2b(rows, runs, obey)

    print("\n  Next (exp_3): the black box. We could print P above only because we WROTE this env — an")
    print("  agent never gets it. Next we watch experience stand in for it: estimate P̂ from samples.")


def _figure_2b(rows, runs, obey):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = GridWorld()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    for ax, (label, d_along, d_turn, tv) in zip(axes[:2], rows):
        keep = [i for i in range(len(d_along)) if max(d_along[i], d_turn[i]) > 0.005]
        x = np.arange(len(keep))
        ax.bar(x - 0.2, d_along[keep], 0.4, label="arrived heading UP", color="tab:blue")
        ax.bar(x + 0.2, d_turn[keep], 0.4, label="arrived heading RIGHT", color="tab:orange")
        ax.set_xticks(x)
        ax.set_xticklabels([str(grid.states[i]) for i in keep], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("p(next cell)")
        ax.set_title(f"{label}\nstand in (2,2), aim UP  —  TV distance {tv:.3f}", fontsize=10)
        ax.legend(fontsize=8)

    ax = axes[2]
    for (label, per_b), color in zip(runs.items(), ("tab:red", "tab:green")):
        xs = list(per_b)
        ys = [per_b[b][0] for b in xs]
        es = [per_b[b][1] for b in xs]
        ax.errorbar(xs, ys, yerr=es, marker="o", lw=1.6, capsize=3, color=color, label=label)
    ax.set_xscale("log")
    ax.minorticks_off()                                            # log minor labels collide here
    ax.set_xticks(list(BUDGETS)); ax.set_xticklabels([f"{b:,}" for b in BUDGETS], fontsize=8)
    ax.set_xlabel("training episodes"); ax.set_ylabel("mean return of the greedy policy")
    ax.set_title(f"same agent on the inertia world (obey={obey}):\nthe STATE is what changed", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("The Markov property: measured, broken, and repaired by fixing the STATE", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "02_markov.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out}")


def run_experiments():
    # exp_1_whole_game()
    exp_2a_the_tuple()
    exp_2b_markov()
    # exp_3_black_box()
    # exp_4_return()
    # exp_5_gamma()
    # exp_6_reward_design()
    # exp_7_termination()


if __name__ == "__main__":
    run_experiments()
