"""
Rebuilding 02_mdp_dp.py from scratch, one small step at a time.

STEP 1 — the geometry of the world.

Before any "MDP" or "value" talk, we just need a map. Our world is a 3-row,
4-column grid (Russell & Norvig's classic):

    col:   0    1    2    3
        +----+----+----+----+
 row 0  | .  | .  | .  | +1 |
        +----+----+----+----+
 row 1  | .  | #  | .  | -1 |
        +----+----+----+----+
 row 2  | .  | .  | .  | .  |
        +----+----+----+----+

We address every cell by (row, col), with row 0 at the TOP. So:
  - (0, 3) is the good terminal  (+1 reward, episode ends)
  - (1, 3) is the bad terminal   (-1 reward, episode ends)
  - (1, 1) is a WALL  (you can never stand here)
  - (2, 0) is where the agent starts

A "state" is just a cell the agent can actually occupy — i.e. every cell that
is NOT a wall. There are 12 cells, 1 wall, so 11 states.

We also need to talk about states as plain integers 0..10 (the DP code later
indexes numpy arrays by state), so we build two lookups:
    states : list where states[i] = the (row, col) of state i
    s2i    : dict where s2i[(row, col)] = the integer i
"""

from __future__ import annotations

import numpy as np

rows, cols = 3, 4
walls = {(1, 1)}
terminals = {(0, 3): +1.0, (1, 3): -1.0}
start = (2, 0)

# every occupiable cell, scanned top-to-bottom, left-to-right
states = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in walls]
s2i = {cell: i for i, cell in enumerate(states)}

# ---------------------------------------------------------------------------
# STEP 2 — deterministic movement.
#
# Four actions. We encode them as 0..3 going CLOCKWISE (UP, RIGHT, DOWN, LEFT).
# That clockwise order is deliberate: later, a "slip" is just stepping one slot
# left or right in this ring, i.e. (a+1)%4 and (a-1)%4 are the two directions
# perpendicular to a. Keep that in your back pocket; we use it in Step 3.
#
# Remember row 0 is the TOP, so UP = row-1, DOWN = row+1.
# ---------------------------------------------------------------------------
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
DIRS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}
ARROWS = {UP: "↑", RIGHT: "→", DOWN: "↓", LEFT: "←"}


def move(cell, a):
    """Where action a lands you from `cell`, with NO slip.

    Rule: try to step in direction a. If that would leave the grid OR enter a
    wall, you bounce and stay put. Otherwise you move.
    """
    dr, dc = DIRS[a]
    nxt = (cell[0] + dr, cell[1] + dc)
    off_grid = not (0 <= nxt[0] < rows) or not (0 <= nxt[1] < cols)
    if off_grid or nxt in walls:
        return cell          # bumped an edge or wall -> stay
    return nxt


# ---------------------------------------------------------------------------
# STEP 3 — the slippery floor (stochastic actions).
#
# The floor is slippery, so your INTENDED action isn't always what happens:
#     prob 0.8  -> you go where you intended
#     prob 0.1  -> you veer to one perpendicular direction
#     prob 0.1  -> you veer to the OTHER perpendicular direction
# You NEVER slip backwards. The 0.2 total slip is split over the two perps.
#
# Using the clockwise ring from Step 2, the perpendiculars of a are (a+1)%4 and
# (a-1)%4. So the distribution over *intended directions* is easy:
#     {a: 0.8, (a+1)%4: 0.1, (a-1)%4: 0.1}
# Then we push each intended direction through `move` to get the actual cell.
#
# noise = total slip probability (0.2 here). With noise=0 actions are
# deterministic again, which is a handy sanity knob.
# ---------------------------------------------------------------------------
noise = 0.2


def action_outcomes(cell, a):
    """List of (prob, next_cell) for taking action a in `cell`, with slip.

    Note: two different slips can `move` you into the SAME cell (e.g. when both
    veer into a wall and bounce back to where you stand). We merge those so each
    next_cell appears once and the probabilities still sum to 1.
    """
    intended = {a: 1 - noise, (a + 1) % 4: noise / 2, (a - 1) % 4: noise / 2}
    merged = {}                       # next_cell -> total prob
    for act, prob in intended.items():
        nxt = move(cell, act)
        merged[nxt] = merged.get(nxt, 0.0) + prob
    return [(prob, nxt) for nxt, prob in merged.items()]


# ---------------------------------------------------------------------------
# STEP 4 — rewards, termination, and the full model P.
#
# Each move earns a reward and may end the episode:
#     - land on the +1 goal  -> reward +1, done
#     - land on the -1 trap   -> reward -1, done
#     - land anywhere else     -> reward -0.04 (the "living cost"), not done
# The -0.04 per step is what makes dawdling expensive and gives the agent a
# reason to actually head for the goal.
#
# Terminal states are ABSORBING: once you're on +1 or -1 the episode is over.
# We model that as every action self-looping with reward 0 and done=True. This
# little convention pays off later: V is pinned at 0 on terminals, so the DP
# backups need no special-casing for "the episode ended".
#
# Final shape — the thing every DP routine reads:
#     P[s][a] = list of (prob, s_next, reward, done)
# with s, s_next integer state indices (via s2i) and the probs summing to 1.
# ---------------------------------------------------------------------------
step_reward = -0.04
gamma = 0.9        # discount: a reward one step later is worth 0.9x as much


def build_model():
    P = {s: {a: [] for a in range(4)} for s in range(len(states))}
    for cell in states:
        s = s2i[cell]
        if cell in terminals:                      # absorbing terminal
            for a in range(4):
                P[s][a] = [(1.0, s, 0.0, True)]
            continue
        for a in range(4):
            outcomes = []
            for prob, nxt in action_outcomes(cell, a):
                reward = terminals.get(nxt, step_reward)
                done = nxt in terminals
                outcomes.append((prob, s2i[nxt], reward, done))
            P[s][a] = outcomes
    return P


# ---------------------------------------------------------------------------
# STEP 5 — q_from_v: the one-step Bellman lookahead.
#
#     Q(s,a) = sum over (prob, s', r) in P[s][a] of  prob * (r + gamma * V[s'])
#
# Given a value estimate V (one number per state), return Q for ALL 4 actions
# in state s, as an array of shape (4,).
#
# Why no special case for `done`? A terminal s' self-loops with V[s']=0, so
# gamma * V[s'] is already 0 there -- the absorbing convention from Step 4
# does the bookkeeping for us.
# ---------------------------------------------------------------------------
def q_from_v(P, V, s):
    q = np.zeros(4)
    for a in range(4):
        for prob, ns, r, done in P[s][a]:
            q[a] += prob * (r + gamma * V[ns])
    return q


# ---------------------------------------------------------------------------
# STEP 6 — policy_evaluation: how good is a FIXED policy?
#
# Repeat the expectation backup over all states until V stops moving:
#     V(s) <- sum_a policy[s,a] * Q(s,a)   = policy[s] . q_from_v(P, V, s)
#
# We sweep IN PLACE (each V[s] update immediately visible to later states in
# the same pass). That's fine -- it converges to the same V^pi, a touch faster.
# theta is the stop threshold: quit when the largest change in a sweep is tiny.
# ---------------------------------------------------------------------------
def policy_evaluation(P, policy, theta=1e-10):
    V = np.zeros(len(states))
    while True:
        delta = 0.0
        for s in range(len(states)):
            v_old = V[s]
            V[s] = policy[s] @ q_from_v(P, V, s)
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    return V


def render_values(V):
    grid = [["  ##  " for _ in range(cols)] for _ in range(rows)]
    for i, (r, c) in enumerate(states):
        grid[r][c] = f"{V[i]:+5.2f} "
    return "\n".join(" ".join(row) for row in grid)


# ---------------------------------------------------------------------------
# STEP 7 — policy_improvement: act greedily w.r.t. V.
#
# For each state, pick a* = argmax_a Q(s,a) and commit to it. The result is a
# DETERMINISTIC policy, stored one-hot: policy[s] is all zeros except a 1.0 on
# the chosen action.
# ---------------------------------------------------------------------------
def policy_improvement(P, V):
    nS = len(states)
    policy = np.zeros((nS, 4))
    for s in range(nS):
        best_a = int(np.argmax(q_from_v(P, V, s)))
        policy[s, best_a] = 1.0
    return policy


def render_policy(policy):
    grid = [["  ·" for _ in range(cols)] for _ in range(rows)]
    for (r, c) in walls:
        grid[r][c] = "  #"
    for (r, c), val in terminals.items():
        grid[r][c] = " +1" if val > 0 else " -1"
    for s, (r, c) in enumerate(states):
        if (r, c) in terminals:
            continue
        grid[r][c] = "  " + ARROWS[int(np.argmax(policy[s]))]
    return "\n".join("".join(row) for row in grid)


# ---------------------------------------------------------------------------
# STEP 8 — policy_iteration: evaluate -> improve, repeat to optimality.
#
# Start from any policy. Each round:
#     V          = policy_evaluation(P, policy)   # how good is it, exactly?
#     new_policy = policy_improvement(P, V)        # greedy w.r.t. that V
#     stop if the greedy ACTION is unchanged in every state (stable => optimal)
# Compare argmax(policy) vs argmax(new_policy), not the raw matrices.
# ---------------------------------------------------------------------------
def policy_iteration(P):
    nS = len(states)
    policy = np.ones((nS, 4)) / 4          # start from uniform-random
    V = np.zeros(nS)
    rounds = 0
    while True:
        rounds += 1
        V = policy_evaluation(P, policy)
        new_policy = policy_improvement(P, V)
        if (np.argmax(policy, axis=1) == np.argmax(new_policy, axis=1)).all():
            break
        policy = new_policy
    return policy, V, rounds


# ---------------------------------------------------------------------------
# STEP 9 — value_iteration: one max-backup per state, repeated.
#
#     V(s) <- max_a Q(s,a)     (Bellman OPTIMALITY backup)
#
# No policy is tracked during the loop -- the max IS the improvement. Sweep
# until V settles, then read the greedy policy off the converged V* once.
# ---------------------------------------------------------------------------
def value_iteration(P, theta=1e-10):
    nS = len(states)
    V = np.zeros(nS)
    sweeps = 0
    while True:
        sweeps += 1
        delta = 0.0
        for s in range(nS):
            v_old = V[s]
            V[s] = np.max(q_from_v(P, V, s))
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    policy = policy_improvement(P, V)
    return policy, V, sweeps


if __name__ == "__main__":
    P = build_model()

    pi_pol, pi_V, pi_rounds = policy_iteration(P)
    vi_pol, vi_V, vi_sweeps = value_iteration(P)

    print(f"policy iteration: {pi_rounds} rounds")
    print(f"value iteration : {vi_sweeps} sweeps\n")

    print("value iteration V*:")
    print(render_values(vi_V))
    print("\nvalue iteration pi*:")
    print(render_policy(vi_pol))

    same_V = np.allclose(pi_V, vi_V, atol=1e-6)
    same_pi = (np.argmax(pi_pol, 1) == np.argmax(vi_pol, 1)).all()
    print(f"\nPI and VI agree on V*: {same_V}")
    print(f"PI and VI agree on pi*: {same_pi}")
