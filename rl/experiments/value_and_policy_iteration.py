"""
Value iteration on SIMPLE deterministic gridworlds — watch V propagate step by step.

Default grid:

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
  - Reward depends on where you LAND: goal_reward into the goal, else step_reward.
  - The goal is terminal/absorbing: once there you stop (V = 0 there).
  - Bumping a wall or the edge leaves you in place.

    Bellman optimality backup:  V(s) <- max_a [ r(s,a,s') + gamma * V(s') ]
"""

import argparse
import contextlib
import sys
from dataclasses import dataclass, field

GAMMA = 0.9

UP, RIGHT, DOWN, LEFT = (-1, 0), (0, 1), (1, 0), (0, -1)
ACTIONS = [UP, RIGHT, DOWN, LEFT]
ARROWS = {UP: "↑", RIGHT: "→", DOWN: "↓", LEFT: "←"}


@dataclass(frozen=True)
class Grid:
    rows: int
    cols: int
    goal: tuple
    walls: frozenset = field(default_factory=frozenset)


DEFAULT = Grid(rows=3, cols=4, goal=(0, 3), walls=frozenset({(1, 1)}))

# A bigger grid with a 2-cell pocket (P) fully sealed by walls (#): the agent can
# wander between the two P cells but can NEVER reach the +1 goal from there.
#
#     . . . . . +1
#     . . . . . .
#     . . # # . .
#     . # P P # .
#     . . # # . .
SEALED = Grid(
    rows=5, cols=6, goal=(0, 5),
    walls=frozenset({(2, 2), (2, 3), (4, 2), (4, 3), (3, 1), (3, 4)}),
)
POCKET = {(3, 2), (3, 3)}        # the unreachable cells, for annotation only


def step(grid, cell, action, step_reward=0.0, goal_reward=1.0):
    """Deterministic transition. Returns (next_cell, reward).

    Reward depends on where you LAND: `goal_reward` for entering the goal,
    otherwise `step_reward`. Walls/edges => stay put.
    """
    r, c = cell[0] + action[0], cell[1] + action[1]
    nxt = (r, c)
    if not (0 <= r < grid.rows and 0 <= c < grid.cols) or nxt in grid.walls:
        nxt = cell                       # bumped a wall/edge: stay
    reward = goal_reward if nxt == grid.goal else step_reward
    return nxt, reward


def is_terminal(grid, cell):
    return cell == grid.goal


def cells(grid):
    """All non-wall cells (the states)."""
    return [(r, c) for r in range(grid.rows) for c in range(grid.cols)
            if (r, c) not in grid.walls]


def show(grid, V, title):
    """Pretty-print the value function laid out on the grid."""
    print(title)
    for r in range(grid.rows):
        row = []
        for c in range(grid.cols):
            if (r, c) in grid.walls:
                row.append("   ##  ")
            else:
                row.append(f"{V[(r, c)]:+6.2f} ")
        print(" ".join(row))
    print()


def show_policy(grid, V, step_reward, goal_reward, gamma=GAMMA):
    """Read the greedy action off V: in each cell point to the best neighbor."""
    print("greedy policy (argmax_a [r + gamma*V(s')]):")
    for r in range(grid.rows):
        row = []
        for c in range(grid.cols):
            cell = (r, c)
            if cell in grid.walls:
                row.append("#")
            elif is_terminal(grid, cell):
                row.append("+1")
            else:
                def q(a, cell=cell):
                    nxt, reward = step(grid, cell, a, step_reward, goal_reward)
                    return reward + gamma * V[nxt]
                best_a = max(ACTIONS, key=q)
                row.append(ARROWS[best_a])
        print("  ".join(f"{x:>2}" for x in row))
    print()


def render_policy(grid, policy, title="policy:"):
    """Draw arrows from an EXPLICIT policy dict {cell: action}."""
    print(title)
    for r in range(grid.rows):
        row = []
        for c in range(grid.cols):
            cell = (r, c)
            if cell in grid.walls:
                row.append("#")
            elif is_terminal(grid, cell):
                row.append("+1")
            else:
                row.append(ARROWS[policy[cell]])
        print("  ".join(f"{x:>2}" for x in row))
    print()


def value_iteration(grid=DEFAULT, step_reward=0.0, goal_reward=1.0, gamma=GAMMA,
                    theta=1e-6, v_init=0.0, max_iters=1000, verbose=True):
    # Non-terminal cells start at v_init; the GOAL is pinned at 0 (it's absorbing
    # — its true value really IS 0, a boundary condition, not a guess).
    V = {cell: (0.0 if is_terminal(grid, cell) else v_init) for cell in cells(grid)}
    if verbose:
        show(grid, V, f"iter 0 (initial V = {v_init} on non-terminals, 0 at goal):")

    for it in range(1, max_iters + 1):
        newV = dict(V)                   # synchronous: backups read the OLD V
        delta = 0.0
        for cell in cells(grid):
            if is_terminal(grid, cell):
                continue                 # terminal value pinned at 0
            q = []
            for a in ACTIONS:
                nxt, reward = step(grid, cell, a, step_reward, goal_reward)
                q.append(reward + gamma * V[nxt])
            best = max(q)
            newV[cell] = best
            delta = max(delta, abs(best - V[cell]))
        V = newV
        if verbose:
            show(grid, V, f"iter {it}  (max change this sweep = {delta:.4f}):")
        if delta < theta:
            print(f"converged after {it} sweeps (delta {delta:.2e} < {theta}).\n")
            break
    else:
        print(f"did NOT converge in {max_iters} sweeps "
              f"(delta still {delta:.4f} — diverging on unreachable cells).\n")

    show_policy(grid, V, step_reward, goal_reward, gamma)
    return V


def policy_evaluation(grid, policy, step_reward, goal_reward, gamma,
                      theta=1e-6, max_iters=1000):
    """Compute V^pi for a DETERMINISTIC policy {cell: action} by sweeping the
    EXPECTATION backup (here just a single action per state) until V settles.

        V(s) <- r(s, pi(s)) + gamma * V(s')
    """
    v_init = 0.0
    V = {cell: (0.0 if is_terminal(grid, cell) else v_init) for cell in cells(grid)}
    verbose = False
    if verbose:
        show(grid, V, f"iter 0 (initial V = {v_init} on non-terminals, 0 at goal):")
    
    for it in range(1, max_iters + 1):
        newV = dict(V)                # synchronous: backups read the OLD V
        delta = 0.0
        for cell in cells(grid):
            if is_terminal(grid, cell):
                continue
            nxt, reward = step(grid, cell, policy[cell], step_reward, goal_reward)
            v_new = reward + gamma * V[nxt]
            delta = max(delta, abs(v_new - V[cell]))
            newV[cell] = v_new
        V = newV
        if verbose:
            show(grid, V, f"iter {it}  (max change this sweep = {delta:.4f}):")
        if delta < theta:
            break
    return V


def greedy_policy(grid, V, step_reward, goal_reward, gamma):
    """One-step-lookahead improvement: pick the best action in every state."""
    policy = {}
    for cell in cells(grid):
        if is_terminal(grid, cell):
            continue

        def q(a, cell=cell):
            nxt, reward = step(grid, cell, a, step_reward, goal_reward)
            return reward + gamma * V[nxt]
        policy[cell] = max(ACTIONS, key=q)
    return policy


def policy_iteration(grid=DEFAULT, step_reward=0.0, goal_reward=1.0, gamma=GAMMA,
                     theta=1e-6, max_iters=1000, verbose=True):
    """Alternate EVALUATE (V^pi) and IMPROVE (greedy) until the policy stops
    changing. Provably converges to pi* — usually in very few outer rounds."""
    # arbitrary start policy: point everyone UP (deliberately suboptimal)
    policy = {cell: UP for cell in cells(grid) if not is_terminal(grid, cell)}
    if verbose:
        render_policy(grid, policy, "round 0: initial policy (all UP):")

    V = {}
    for it in range(1, max_iters + 1):
        V = policy_evaluation(grid, policy, step_reward, goal_reward, gamma, theta)
        new_policy = greedy_policy(grid, V, step_reward, goal_reward, gamma)
        changed = sum(new_policy[c] != policy[c] for c in new_policy)
        if verbose:
            show(grid, V, f"round {it}: V^pi for the current policy")
            render_policy(grid, new_policy,
                          f"round {it}: improved (greedy) policy "
                          f"[{changed} action(s) changed]:")
        if new_policy == policy:
            print(f"policy stable after {it} round(s) — optimal pi*.\n")
            break
        policy = new_policy
    return policy, V


def _banner(*lines):
    print("=" * 60)
    for line in lines:
        print(line)
    print("=" * 60)


def value_iteration_experiments():
    def exp_a():
        _banner("VARIATION A: +1 for reaching goal, 0 per step")
        value_iteration(step_reward=0.0, goal_reward=1.0)

    def exp_b():
        _banner("VARIATION B: -1 per step, 0 for reaching goal")
        value_iteration(step_reward=-1.0, goal_reward=0.0)

    def exp_b_init50():
        _banner("VARIATION B', SAME but V initialized at -50 (still converges to V*)")
        value_iteration(step_reward=-1.0, goal_reward=0.0, v_init=-50.0)

    def exp_a_gamma1():
        _banner("VARIATION A @ gamma=1: +1 goal, 0 step, NO discount",
                "  -> dawdling is free; V=1 everywhere, policy degenerates")
        value_iteration(step_reward=0.0, goal_reward=1.0, gamma=1.0)

    def exp_b_gamma1():
        _banner("VARIATION B @ gamma=1: -1 step, 0 goal, NO discount",
                "  -> step cost still forces shortest path; V = -(steps to goal)")
        value_iteration(step_reward=-1.0, goal_reward=0.0, gamma=1.0)

    def exp_b_init50_gamma1():
        _banner("VARIATION B' @ gamma=1: same, V init -50, NO discount",
                "  -> even with gamma=1 (no contraction) the goal anchor still",
                "     locks the true values one ring per sweep")
        value_iteration(step_reward=-1.0, goal_reward=0.0, gamma=1.0, v_init=-50.0)

    def exp_sealed_gamma09():
        _banner("SEALED POCKET @ gamma=0.9: trapped cells converge to -10")
        value_iteration(SEALED, step_reward=-1.0, goal_reward=0.0, gamma=0.9)

    def exp_sealed_gamma09_init50():
        _banner("SEALED POCKET @ gamma=0.9, V init -50: pocket still settles to -10",
                "  -> init -50 is BELOW the trapped value, so the pocket climbs UP to -10")
        value_iteration(SEALED, step_reward=-1.0, goal_reward=0.0, gamma=0.9,
                        v_init=-50.0)

    def exp_sealed_gamma1():
        _banner("SEALED POCKET @ gamma=1.0: trapped cells DIVERGE (never converge)",
                "  (capped at 8 sweeps so we can watch -1 pile up)")
        value_iteration(SEALED, step_reward=-1.0, goal_reward=0.0, gamma=1.0,
                        max_iters=15)

    def exp_sealed_gamma1_init50():
        _banner("SEALED POCKET @ gamma=1.0, V init -50: pocket DIVERGES from -50 down",
                "  -> no fixed point at gamma=1; reachable cells still converge",
                "  (capped at 8 sweeps)")
        value_iteration(SEALED, step_reward=-1.0, goal_reward=0.0, gamma=1.0,
                        v_init=-50.0, max_iters=15)

    # comment/uncomment the experiments you want to run
    exp_a()
    # exp_b()
    # exp_b_init50()
    # exp_a_gamma1()
    # exp_b_gamma1()
    # exp_b_init50_gamma1()
    # exp_sealed_gamma09()
    # exp_sealed_gamma09_init50()
    # exp_sealed_gamma1()
    # exp_sealed_gamma1_init50()


def policy_iteration_experiments():
    def exp_a():
        _banner("POLICY ITERATION, VARIATION A: +1 goal, 0 step",
                "  -> evaluate V^pi, act greedily, repeat; converges to pi* fast")
        policy_iteration(step_reward=0.0, goal_reward=1.0)

    def exp_b():
        _banner("POLICY ITERATION, VARIATION B: -1 step, 0 goal")
        policy_iteration(step_reward=-1.0, goal_reward=0.0)

    def exp_sealed():
        _banner("POLICY ITERATION, SEALED POCKET @ gamma=0.9",
                "  -> reachable cells find pi*; pocket settles to the -10 trap")
        policy_iteration(SEALED, step_reward=-1.0, goal_reward=0.0, gamma=0.9)

    # comment/uncomment the experiments you want to run
    # exp_a()
    exp_b()
    # exp_sealed()


def run_experiments():
    # value_iteration_experiments()
    policy_iteration_experiments()


@contextlib.contextmanager
def _tee(path):
    """Print to BOTH the terminal and `path` (so long runs aren't lost to scrollback)."""
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
                        help="also write all output to FILE (full run, no scrollback limit)")
    args = parser.parse_args()

    if args.out:
        with _tee(args.out):
            run_experiments()
    else:
        run_experiments()
