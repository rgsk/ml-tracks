"""
The hand-coded tabular environments for the RL track (rung 1: `walkthroughs/q_learning/`).

Deliberately small enough to PRINT and check by hand — the whole point of starting tabular is that
there's nowhere for a bug (or a misunderstanding) to hide. Gymnasium takes over from `dqn/` on.

Each env exposes TWO faces, and which one you're allowed to touch is the lesson:

  the BLACK BOX (what an agent gets):   reset() -> s,  step(a) -> (s', r, done),  nS, nA, gamma
  the MODEL     (what a PLANNER gets):  P[s][a] = [(prob, s_next, reward, done), ...]

`step()` rolls its own dice — it never reads `P` — so an agent using only the black box genuinely
learns from experience. `P` exists for exp_3, where dynamic programming solves the env exactly and
gives us the ground truth that the sampling methods are approximating.
"""

from __future__ import annotations

import numpy as np

# Clockwise encoding, so the two PERPENDICULAR slips of action `a` are (a+1)%4 and (a-1)%4.
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIRS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}
ARROWS = {UP: "↑", RIGHT: "→", DOWN: "↓", LEFT: "←"}


class GridWorld:
    """The classic Russell & Norvig 4x3 SLIPPERY gridworld.

        . . . +1        start at bottom-left (2,0)
        . # . -1        # is a wall (can't be entered)
        S . . .         +1 / -1 are terminal

    Actions move UP/RIGHT/DOWN/LEFT but the floor is slippery: with prob `1 - noise` you go where you
    intended, and `noise/2` each you veer to one of the two PERPENDICULAR directions. Walking into a
    wall or off the edge leaves you where you were. Every move costs `step_reward` = -0.04, so
    dawdling is expensive.

    That 20% slip is what makes this interesting: the shortest route to +1 runs right past the -1
    trap, and a slip there is fatal — so the optimal policy is not the one you'd guess.
    """

    def __init__(self, noise: float = 0.2, step_reward: float = -0.04,
                 gamma: float = 0.9, seed: int | None = None):
        self.rows, self.cols = 3, 4
        self.walls = {(1, 1)}
        self.terminals = {(0, 3): +1.0, (1, 3): -1.0}
        self.start = (2, 0)
        self.noise = noise
        self.step_reward = step_reward
        self.gamma = gamma

        self.states = [(r, c) for r in range(self.rows) for c in range(self.cols)
                       if (r, c) not in self.walls]
        self.nS, self.nA = len(self.states), 4
        self.s2i = {cell: i for i, cell in enumerate(self.states)}
        self.P = self._build_model()            # the MODEL — for planning (exp_3), not for agents

        self.rng = np.random.default_rng(seed)
        self.s: int | None = None

    # --- the black box: this is all an agent may use -----------------------------------------
    def reset(self) -> int:
        """Back to the fixed start. (No 'exploring starts' — in control, exploration is the
        agent's job, which is exp_5.)"""
        self.s = self.s2i[self.start]
        return self.s

    def step(self, a: int):
        """One sampled transition -> (s_next, reward, done). Rolls the slip dice itself; the agent
        never sees a probability."""
        cell = self.states[self.s]
        actual = int(self.rng.choice([a, (a + 1) % 4, (a - 1) % 4],
                                     p=[1 - self.noise, self.noise / 2, self.noise / 2]))
        nxt = self._move(cell, actual)
        reward = self.terminals.get(nxt, self.step_reward)
        done = nxt in self.terminals
        self.s = self.s2i[nxt]
        return self.s, reward, done

    # --- the model (used from exp_3 on) ------------------------------------------------------
    def _move(self, cell: tuple[int, int], a: int) -> tuple[int, int]:
        """Where action `a` lands you from `cell`, ignoring slip. Wall/edge -> stay put."""
        dr, dc = DIRS[a]
        nxt = (cell[0] + dr, cell[1] + dc)
        if (not 0 <= nxt[0] < self.rows or not 0 <= nxt[1] < self.cols
                or nxt in self.walls):
            return cell
        return nxt

    def _build_model(self) -> dict:
        """P[s][a] = list of (prob, s_next, reward, done), summing to prob 1."""
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for cell in self.states:
            s = self.s2i[cell]
            if cell in self.terminals:
                for a in range(self.nA):        # absorbing, zero reward => V = 0 there
                    P[s][a] = [(1.0, s, 0.0, True)]
                continue
            for a in range(self.nA):
                outcomes = {a: 1 - self.noise,
                            (a + 1) % 4: self.noise / 2,
                            (a - 1) % 4: self.noise / 2}
                agg: dict[int, list] = {}       # next_state -> [prob, reward, done]
                for act, prob in outcomes.items():
                    nxt = self._move(cell, act)
                    ns = self.s2i[nxt]
                    if ns in agg:               # two different slips can land in the same cell
                        agg[ns][0] += prob
                    else:
                        agg[ns] = [prob, self.terminals.get(nxt, self.step_reward),
                                   nxt in self.terminals]
                P[s][a] = [(p, ns, r, d) for ns, (p, r, d) in agg.items()]
        return P

    # --- printing ---------------------------------------------------------------------------
    def render(self, cell_text=None, width: int = 3) -> str:
        """The grid as text. `cell_text(s, cell) -> str` fills non-terminal, non-wall cells."""
        out = []
        for r in range(self.rows):
            row = []
            for c in range(self.cols):
                if (r, c) in self.walls:
                    row.append("#")
                elif (r, c) in self.terminals:
                    row.append(f"{self.terminals[(r, c)]:+.0f}")
                elif cell_text is None:
                    row.append("S" if (r, c) == self.start else ".")
                else:
                    row.append(cell_text(self.s2i[(r, c)], (r, c)))
            out.append("".join(f"{t:>{width}}" for t in row))
        return "\n".join(out)

    def render_policy(self, actions) -> str:
        """Grid of arrows for a (nS,) array of greedy actions."""
        return self.render(lambda s, cell: ARROWS[int(actions[s])])
