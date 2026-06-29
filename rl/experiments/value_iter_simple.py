"""
Value iteration on a SIMPLE deterministic gridworld — watch V propagate step by step.

    col:   0    1    2    3
        +----+----+----+----+
 row 0  | .  | .  | .  | +1 |
        +----+----+----+----+
 row 1  | .  | #  | .  |  . |
        +----+----+----+----+
 row 2  | .  | .  | .  | .  |
        +----+----+----+----+

Rules (kept deliberately minimal so the value updates are obvious):
  - Deterministic: the intended action ALWAYS works (no slip).
  - Moving into the +1 cell gives reward +1; every other step gives 0.
  - (0,3) is terminal/absorbing: once there you stop (V = 0 there).
  - Bumping a wall or the edge leaves you in place.
  - gamma = 0.9

Because step reward is 0, the optimal V of a cell is just 0.9^(steps-to-goal - 1):
the value "wave" spreads one ring further from the goal on each sweep.

    Bellman optimality backup:  V(s) <- max_a [ r(s,a,s') + gamma * V(s') ]
"""

ROWS, COLS = 3, 4
WALL = (1, 1)
GOAL = (0, 3)
GAMMA = 0.9

UP, RIGHT, DOWN, LEFT = (-1, 0), (0, 1), (1, 0), (0, -1)
ACTIONS = [UP, RIGHT, DOWN, LEFT]
ARROWS = {UP: "↑", RIGHT: "→", DOWN: "↓", LEFT: "←"}


def step(cell, action):
    """Deterministic transition. Returns (next_cell, reward).

    Moving into GOAL pays +1; anything else pays 0. Walls/edges => stay put.
    """
    r, c = cell[0] + action[0], cell[1] + action[1]
    nxt = (r, c)
    if not (0 <= r < ROWS and 0 <= c < COLS) or nxt == WALL:
        nxt = cell                       # bumped a wall/edge: stay
    reward = 1.0 if nxt == GOAL else 0.0
    return nxt, reward


def is_terminal(cell):
    return cell == GOAL


def cells():
    """All non-wall cells (the states)."""
    return [(r, c) for r in range(ROWS) for c in range(COLS) if (r, c) != WALL]


def show(V, title):
    """Pretty-print the value function laid out on the grid."""
    print(title)
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            if (r, c) == WALL:
                row.append("  ##  ")
            else:
                row.append(f"{V[(r, c)]:+5.2f} ")
        print("  ".join(row))
    print()


def show_policy(V, gamma=GAMMA):
    """Read the greedy action off V: in each cell point to the best neighbor."""
    print("greedy policy (argmax_a [r + gamma*V(s')]):")
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            cell = (r, c)
            if cell == WALL:
                row.append("#")
            elif is_terminal(cell):
                row.append("+1")
            else:
                def q(a, cell=cell):
                    nxt, reward = step(cell, a)
                    return reward + gamma * V[nxt]
                best_a = max(ACTIONS, key=q)
                row.append(ARROWS[best_a])
        print("  ".join(f"{x:>2}" for x in row))
    print()


def value_iteration(gamma=GAMMA, theta=1e-6):
    # V starts at 0 everywhere — including the goal, which stays 0 (absorbing).
    V = {cell: 0.0 for cell in cells()}
    show(V, "iter 0 (initial V = 0 everywhere):")

    it = 0
    while True:
        it += 1
        newV = dict(V)                   # synchronous: backups read the OLD V
        delta = 0.0
        for cell in cells():
            if is_terminal(cell):
                continue                 # terminal value pinned at 0
            # one-step lookahead over the 4 actions, take the best
            q = []
            for a in ACTIONS:
                nxt, reward = step(cell, a)
                q.append(reward + gamma * V[nxt])
            best = max(q)
            newV[cell] = best
            delta = max(delta, abs(best - V[cell]))
        V = newV
        show(V, f"iter {it}  (max change this sweep = {delta:.4f}):")
        if delta < theta:
            print(f"converged after {it} sweeps (delta {delta:.2e} < {theta}).\n")
            break

    show_policy(V, gamma)
    return V


if __name__ == "__main__":
    value_iteration()
