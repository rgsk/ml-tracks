"""
WALKTHROUGH: q_learning (top-down) — drop an agent that knows NOTHING into a gridworld, watch it turn
into one that reliably walks to the goal, THEN open up each piece.

Rung 1 of the ladder (see ../../roadmap.md): tabular, no neural nets, so nothing hides. exp_1 runs the
whole game — a table of scores, an ε-greedy actor, and one update line — and you watch the learning
curve climb. Later experiments open one box each:

  1. the whole game        — Q-table + ε-greedy + one update line; watch it learn. (here)
  2. the environment       — what an MDP is: (S,A,P,R,γ), the return G_t, and what γ actually buys.
  3. the value functions   — Bellman, and the EXACT answer via DP: what exp_1 was crawling toward.
  4. the target            — where `r + γ·max Q` comes from: MC (full return) vs TD (bootstrap).
  5. exploration           — the ε knob, stripped down to bandits (children/exploration/).
  6. off- vs on-policy     — swap `max Q[s']` for `Q[s',a']` and you get SARSA; the Cliff split.

The whole game in one breath: score every (state, action) pair, act mostly-greedily on those scores,
and after every step nudge the score you just used toward what actually happened next.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))          # new/rl/walkthroughs/q_learning
_RL = os.path.dirname(os.path.dirname(_HERE))               # new/rl (holds envs.py, custom/)
_FIGS = os.path.join(_HERE, "figures", "experiments")
sys.path.insert(0, _RL)

from envs import ARROWS, GridWorld                          # noqa: E402


def _banner(*lines):
    print("=" * 100)
    for line in lines:
        print(line)
    print("=" * 100)


# ---------------------------------------------------------------------------
# The pieces. Rough narration only — each gets its own experiment later.
#
#   Q[s, a]      a table of SCORES: "how good is taking action a in state s, if I act well after?"
#                One number per (state, action). Starts at all zeros = knows nothing. (exp_3 = what
#                these numbers actually mean, and what their exact values are.)
#   ε-greedy     how we ACT: usually take the best-scoring action, but with probability ε take a
#                random one, because you can't discover a better route without occasionally leaving
#                the one you know. (exp_5)
#   the update   after every single step (s, a, r, s'): move Q[s,a] a fraction α toward
#                "reward I just got + γ · the best I could do from where I landed". (exp_4)
# ---------------------------------------------------------------------------
def epsilon_greedy(Q: np.ndarray, s: int, eps: float, rng: np.random.Generator) -> int:
    """Act: random action with prob eps (EXPLORE), else the best-scoring one (EXPLOIT).

    Ties are broken RANDOMLY — Q starts all-zero, so always taking argmax (=action 0) would send the
    agent marching UP forever and it would never see the rest of the grid."""
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    q = Q[s]
    return int(rng.choice(np.flatnonzero(q == q.max())))


def q_learning(env, episodes: int, alpha: float = 0.1, eps0: float = 1.0, eps_min: float = 0.05,
               anneal_frac: float = 0.5, max_steps: int = 200, snapshots=(),
               seed: int = 0):
    """Tabular Q-learning. Returns (Q, history, snapshots) — the ENTIRE algorithm is 6 lines.

    ε anneals linearly from eps0 to eps_min over the first `anneal_frac` of training: explore a lot
    while the table is garbage, then mostly exploit what you've learned.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.nS, env.nA))
    hist = {"ret": np.zeros(episodes), "steps": np.zeros(episodes), "eps": np.zeros(episodes),
            "visits": np.zeros(env.nS, dtype=int)}
    snaps = {0: Q.copy()} if 0 in snapshots else {}

    for ep in range(episodes):
        eps = max(eps_min, eps0 + (eps_min - eps0) * ep / (anneal_frac * episodes))
        s = env.reset()
        total, n = 0.0, 0
        for n in range(1, max_steps + 1):
            a = epsilon_greedy(Q, s, eps, rng)                      # behave: mostly greedy
            ns, r, done = env.step(a)                               # the world answers
            target = r + env.gamma * Q[ns].max() * (1.0 - done)     # what happened + best from there
            Q[s, a] += alpha * (target - Q[s, a])                   # nudge the score we just used
            hist["visits"][s] += 1
            s, total = ns, total + r
            if done:
                break
        hist["ret"][ep], hist["steps"][ep], hist["eps"][ep] = total, n, eps
        if (ep + 1) in snapshots:
            snaps[ep + 1] = Q.copy()
    return Q, hist, snaps


def evaluate(env, policy, episodes: int = 2000, max_steps: int = 200, seed: int = 123):
    """Run a FIXED policy (no learning, no exploration) and report how it actually does.

    `policy(s, rng) -> action`. Returns mean undiscounted return, % of episodes ending on +1, and
    mean episode length. The env is still slippery, so even a perfect policy sometimes loses."""
    rng = np.random.default_rng(seed)
    rets, wins, lens = np.zeros(episodes), 0, np.zeros(episodes)
    for ep in range(episodes):
        s = env.reset()
        total, n, last = 0.0, 0, 0.0
        for n in range(1, max_steps + 1):
            s, r, done = env.step(policy(s, rng))
            total, last = total + r, r
            if done:
                break
        rets[ep], lens[ep] = total, n
        wins += int(last > 0)
    return {"ret": rets.mean(), "win": 100 * wins / episodes, "steps": lens.mean()}


def _arrow(q_row: np.ndarray) -> str:
    """Greedy arrow for one Q row — '?' if every action still scores the same (nothing learned yet)."""
    return "?" if q_row.max() == q_row.min() else ARROWS[int(q_row.argmax())]


def exp_1_whole_game(seed=0, episodes=4000, alpha=0.1):
    """The whole game, top-down: a Q-table learns to cross a slippery gridworld from nothing but
    (state, action, reward) tuples. No Bellman equations, no derivations — see it work, get the map.
    exp_2..exp_6 open each piece."""
    _banner("EXP 1: the whole game — a Q-table learns to cross a slippery gridworld from experience")

    env = GridWorld(seed=seed)
    print("  the world (the agent is NOT told any of this — it only gets reset()/step(a)):\n")
    print("   " + env.render().replace("\n", "\n   "))
    print(f"\n    S = start ({env.start}),  +1 / -1 = terminal,  # = wall")
    print(f"    every move costs {env.step_reward},  and the floor is SLIPPERY: {1 - env.noise:.0%} you go where")
    print(f"    you aimed, {env.noise / 2:.0%} each you veer to one of the two perpendicular directions")
    print(f"    γ = {env.gamma}  (future reward is worth {env.gamma}x per step of delay)\n")
    print("  the agent, in one breath:")
    print(f"    a table   Q[s, a], {env.nS} states x {env.nA} actions = {env.nS * env.nA} numbers, all starting at 0")
    print("    act       ε-greedy: usually argmax_a Q[s,a], sometimes random (or you never discover better)")
    print("    learn     Q[s,a] += α · ( r + γ·max_a' Q[s',a'] − Q[s,a] )   after EVERY step\n")

    # ---- before: what does knowing nothing look like? ----------------------------------------
    rand = evaluate(env, lambda s, rng: int(rng.integers(env.nA)))
    print("  BEFORE — a uniformly random agent (the zero-knowledge baseline):")
    print(f"    mean return {rand['ret']:+.3f}   reached +1 in {rand['win']:.1f}% of episodes   "
          f"{rand['steps']:.1f} steps/episode\n")

    # ---- train ------------------------------------------------------------------------------
    snap_at = (0, 10, 100, episodes)
    print(f"  TRAINING — {episodes} episodes of Q-learning (α={alpha}, ε: 1.0 -> 0.05):")
    Q, hist, snaps = q_learning(env, episodes, alpha=alpha, snapshots=snap_at, seed=seed)
    step = max(episodes // 8, 1)
    for lo in range(0, episodes, step):                             # block averages: watch it climb
        hi = lo + step
        print(f"    episodes {lo:>5}-{hi:<5}  ε={hist['eps'][lo]:.2f}   "
              f"mean return {hist['ret'][lo:hi].mean():+.3f}   {hist['steps'][lo:hi].mean():5.1f} steps")

    # ---- after: the same measurement, with the learned table ---------------------------------
    greedy = evaluate(env, lambda s, rng: epsilon_greedy(Q, s, 0.0, rng))
    print("\n  AFTER — acting greedily on the learned table:")
    print(f"    mean return {greedy['ret']:+.3f}   reached +1 in {greedy['win']:.1f}% of episodes   "
          f"{greedy['steps']:.1f} steps/episode")
    print(f"    -> return {rand['ret']:+.3f} -> {greedy['ret']:+.3f},  wins {rand['win']:.0f}% -> {greedy['win']:.0f}%")

    # ---- read the policy off the table -------------------------------------------------------
    print("\n  the learned policy (argmax_a Q[s,a] in each cell):\n")
    print("   " + env.render(lambda s, cell: _arrow(Q[s])).replace("\n", "\n   "))
    print("\n  and the value it attaches to each cell (max_a Q[s,a]):\n")
    print("   " + env.render(lambda s, cell: f"{Q[s].max():+.2f}", width=7).replace("\n", "\n   "))

    # The interesting cell: (2,3) sits directly BELOW the -1 trap. "Up" points at the goal column
    # but slips into -1; the table figures out to go LEFT, away from the goal.
    s_trap = env.s2i[(2, 3)]
    print(f"\n  look at cell (2,3), directly below the -1 trap — its four scores:")
    for a, name in enumerate(("up", "right", "down", "left")):
        mark = "  <- greedy" if a == Q[s_trap].argmax() else ""
        print(f"      {ARROWS[a]} {name:<6} {Q[s_trap, a]:+.3f}{mark}")
    print("    'up' heads straight for the goal column and is the WORST action here: a 10% slip lands")
    print("    you in -1. Nobody told the agent about the trap — it lost enough episodes to find out.")

    # ---- honest caveat: the table is only as good as the visits behind it ---------------------
    visits = hist["visits"]
    print("\n  how many times the agent actually STOOD in each cell while learning:\n")
    print("   " + env.render(lambda s, cell: f"{visits[s]}", width=7).replace("\n", "\n   "))
    nonterm = [s for s, c in enumerate(env.states) if c not in env.terminals]
    off_route = min(nonterm, key=lambda s: visits[s])
    tie = min(nonterm, key=lambda s: Q[s].max() - Q[s].min())
    spread = Q[tie].max() - Q[tie].min()
    print(f"    Wildly uneven: the route cells get thousands of visits, {env.states[off_route]} only "
          f"{visits[off_route]}.")
    print("    A score is only as good as the experience behind it, so the off-route cells stay rough.")
    print(f"    At {env.states[tie]} all four actions score within {spread:.2f} of each other — a near-tie that")
    print("    NOISE decides, and the arrow you see there may well be wrong. It costs nothing (the agent")
    print("    never goes there), but it's the crack exp_3 measures exactly and exp_5 (exploration) fixes.")

    _figure(env, hist, snaps, snap_at, rand, greedy)

    print("\n  That's the whole game: scores, ε-greedy behaviour, one update line -> an agent that")
    print("  solves a world it was never given a map of. Next (exp_2): open the first box — the")
    print("  ENVIRONMENT itself — what an MDP is, what the return G_t is, and what γ actually buys.")


def _figure(env, hist, snaps, snap_at, rand, greedy):
    """Payoff figure: the learning curves + the policy at four moments in training."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def smooth(x, w=100):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1], hspace=0.35, wspace=0.25)

    ax = fig.add_subplot(gs[0, :2])
    ax.plot(smooth(hist["ret"]), lw=1.2, color="tab:blue")
    ax.axhline(rand["ret"], ls="--", lw=1, color="tab:red", label=f"random agent ({rand['ret']:+.2f})")
    ax.axhline(greedy["ret"], ls="--", lw=1, color="tab:green",
               label=f"learned greedy policy ({greedy['ret']:+.2f})")
    ax.set_xlabel("episode"); ax.set_ylabel("episode return (100-ep moving avg)")
    ax.set_title("it learns: return per episode"); ax.legend(loc="lower right", fontsize=8)

    ax = fig.add_subplot(gs[0, 2:])
    ax.plot(smooth(hist["steps"]), lw=1.2, color="tab:purple")
    ax.set_xlabel("episode"); ax.set_ylabel("steps to terminal (100-ep moving avg)")
    ax.set_title("and it stops wandering: episode length")

    vmax = max(1e-6, max(abs(s).max() for s in snaps.values()))
    for k, ep in enumerate(snap_at):
        Qs = snaps[ep]
        ax = fig.add_subplot(gs[1, k])
        grid = np.full((env.rows, env.cols), np.nan)
        for s, cell in enumerate(env.states):
            grid[cell] = Qs[s].max()
        ax.imshow(grid, cmap="RdYlGn", vmin=-vmax, vmax=vmax)
        for (r, c) in env.walls:
            ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="0.4"))
        for cell, val in env.terminals.items():
            ax.text(cell[1], cell[0], f"{val:+.0f}", ha="center", va="center", fontsize=13, weight="bold")
        for s, cell in enumerate(env.states):
            if cell not in env.terminals:
                ax.text(cell[1], cell[0], _arrow(Qs[s]), ha="center", va="center", fontsize=17)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"after {ep} episodes", fontsize=10)

    fig.suptitle("Q-learning on the 4x3 slippery gridworld — from flailing to solved, on reward alone",
                 fontsize=13)
    os.makedirs(_FIGS, exist_ok=True)
    out = os.path.join(_FIGS, "01_whole_game.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out}")
    print("    top: the curves. bottom: the greedy policy (arrows) over max_a Q[s,a] (colour) at four")
    print("    moments — '?' = every action still scores the same, i.e. nothing learned there yet.")


def run_experiments():
    exp_1_whole_game()


if __name__ == "__main__":
    run_experiments()
