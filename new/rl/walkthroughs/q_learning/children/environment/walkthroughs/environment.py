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


def run_experiments():
    exp_1_whole_game()
    # exp_2_formal_object()
    # exp_3_black_box()
    # exp_4_return()
    # exp_5_gamma()
    # exp_6_reward_design()
    # exp_7_termination()


if __name__ == "__main__":
    run_experiments()
