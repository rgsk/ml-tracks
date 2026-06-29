"""
Scratch build for the MDP/DP exercise.

Step 1: represent the grid as states with integer indices.

    col:   0    1    2    3
        +----+----+----+----+
 row 0  | .  | .  | .  | +1 |
        +----+----+----+----+
 row 1  | .  | #  | .  | -1 |
        +----+----+----+----+
 row 2  | .  | .  | .  | .  |
        +----+----+----+----+
"""

from __future__ import annotations


ROWS, COLS = 3, 4
WALLS = {(1, 1)}
TERMINALS = {(0, 3): +1.0, (1, 3): -1.0}
START = (2, 0)

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
ACTIONS = [UP, RIGHT, DOWN, LEFT]
ACTION_NAMES = {UP: "UP", RIGHT: "RIGHT", DOWN: "DOWN", LEFT: "LEFT"}
ACTION_ARROWS = {UP: "^", RIGHT: ">", DOWN: "v", LEFT: "<"}
DIRS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}
NOISE = 0.2
STEP_REWARD = -0.04


def build_states() -> tuple[list[tuple[int, int]], dict[tuple[int, int], int]]:
    """List every usable cell and give each one a compact integer id."""
    states = []
    for r in range(ROWS):
        for c in range(COLS):
            cell = (r, c)
            if cell not in WALLS:
                states.append(cell)

    state_to_index = {cell: i for i, cell in enumerate(states)}
    return states, state_to_index


def move(cell: tuple[int, int], action: int) -> tuple[int, int]:
    """Apply one intended action. Edges and walls bounce you back in place."""
    dr, dc = DIRS[action]
    next_cell = (cell[0] + dr, cell[1] + dc)

    outside_grid = not (0 <= next_cell[0] < ROWS and 0 <= next_cell[1] < COLS)
    hits_wall = next_cell in WALLS
    if outside_grid or hits_wall:
        return cell

    return next_cell


def action_outcomes(action: int, noise: float = NOISE) -> dict[int, float]:
    """Return actual movement directions caused by one intended action."""
    return {
        action: 1.0 - noise,
        (action + 1) % 4: noise / 2.0,
        (action - 1) % 4: noise / 2.0,
    }


def next_cell_distribution(
    cell: tuple[int, int],
    intended_action: int,
    noise: float = NOISE,
) -> dict[tuple[int, int], float]:
    """Return probabilities over cells after slip and wall/edge bounces."""
    distribution = {}
    for actual_action, prob in action_outcomes(intended_action, noise).items():
        next_cell = move(cell, actual_action)
        distribution[next_cell] = distribution.get(next_cell, 0.0) + prob
    return distribution


def transitions(
    cell: tuple[int, int],
    intended_action: int,
    state_to_index: dict[tuple[int, int], int],
) -> list[tuple[float, int, float, bool]]:
    """Return full MDP outcomes: probability, next state id, reward, terminal."""
    state = state_to_index[cell]
    if cell in TERMINALS:
        return [(1.0, state, 0.0, True)]

    outcomes = []
    for next_cell, prob in next_cell_distribution(cell, intended_action).items():
        reward = TERMINALS.get(next_cell, STEP_REWARD)
        done = next_cell in TERMINALS
        next_state = state_to_index[next_cell]
        outcomes.append((prob, next_state, reward, done))
    return outcomes


def build_model(
    states: list[tuple[int, int]],
    state_to_index: dict[tuple[int, int], int],
) -> dict[int, dict[int, list[tuple[float, int, float, bool]]]]:
    """Build P[s][a], the explicit transition model used by dynamic programming."""
    model = {}
    for state, cell in enumerate(states):
        model[state] = {}
        for action in ACTIONS:
            model[state][action] = transitions(cell, action, state_to_index)
    return model


def q_from_v(
    model: dict[int, dict[int, list[tuple[float, int, float, bool]]]],
    values: list[float],
    state: int,
    gamma: float,
) -> list[float]:
    """One-step Bellman lookahead: value each action using the current V guesses."""
    q_values = []
    for action in ACTIONS:
        q = 0.0
        for prob, next_state, reward, done in model[state][action]:
            q += prob * (reward + gamma * values[next_state])
        q_values.append(q)
    return q_values


def uniform_random_policy(n_states: int) -> list[list[float]]:
    """Every state chooses uniformly among UP, RIGHT, DOWN, LEFT."""
    action_prob = 1.0 / len(ACTIONS)
    return [[action_prob for _ in ACTIONS] for _ in range(n_states)]


def policy_evaluation(
    model: dict[int, dict[int, list[tuple[float, int, float, bool]]]],
    policy: list[list[float]],
    gamma: float,
    theta: float = 1e-10,
) -> list[float]:
    """Evaluate a fixed policy by repeatedly applying the expectation backup."""
    values = [0.0 for _ in model]

    while True:
        delta = 0.0
        for state in model:
            old_value = values[state]
            q_values = q_from_v(model, values, state, gamma)
            values[state] = sum(
                policy[state][action] * q_values[action]
                for action in ACTIONS
            )
            delta = max(delta, abs(old_value - values[state]))

        if delta < theta:
            break

    return values


def greedy_policy_from_values(
    model: dict[int, dict[int, list[tuple[float, int, float, bool]]]],
    values: list[float],
    gamma: float,
) -> list[list[float]]:
    """Improve a value function into a deterministic greedy policy."""
    policy = []
    for state in model:
        q_values = q_from_v(model, values, state, gamma)
        best_action = max(ACTIONS, key=lambda action: q_values[action])
        action_probs = [0.0 for _ in ACTIONS]
        action_probs[best_action] = 1.0
        policy.append(action_probs)
    return policy


def greedy_actions(policy: list[list[float]]) -> list[int]:
    """Return the chosen action id in every state."""
    return [action_probs.index(max(action_probs)) for action_probs in policy]


def policy_iteration(
    model: dict[int, dict[int, list[tuple[float, int, float, bool]]]],
    gamma: float,
    theta: float = 1e-10,
) -> tuple[list[list[float]], list[float], int]:
    """Alternate policy evaluation and greedy improvement until actions stabilize."""
    policy = uniform_random_policy(len(model))
    values = [0.0 for _ in model]
    iterations = 0

    while True:
        iterations += 1
        values = policy_evaluation(model, policy, gamma, theta)
        improved_policy = greedy_policy_from_values(model, values, gamma)

        if greedy_actions(improved_policy) == greedy_actions(policy):
            return improved_policy, values, iterations

        policy = improved_policy


def value_iteration(
    model: dict[int, dict[int, list[tuple[float, int, float, bool]]]],
    gamma: float,
    theta: float = 1e-10,
) -> tuple[list[list[float]], list[float], int]:
    """Find optimal values by repeatedly applying the max Bellman backup."""
    values = [0.0 for _ in model]
    iterations = 0

    while True:
        iterations += 1
        delta = 0.0
        for state in model:
            old_value = values[state]
            values[state] = max(q_from_v(model, values, state, gamma))
            delta = max(delta, abs(old_value - values[state]))

        if delta < theta:
            break

    policy = greedy_policy_from_values(model, values, gamma)
    return policy, values, iterations


def render_values_grid(
    values: list[float],
    state_to_index: dict[tuple[int, int], int],
) -> str:
    """Format one value per grid cell, keeping walls visibly blocked."""
    cell_width = 6
    row_label_width = 7
    rows = [
        f"{'row/col':>{row_label_width}} "
        + " ".join(f"{c:>{cell_width}}" for c in range(COLS))
    ]
    for r in range(ROWS):
        row = [f"{r:>{row_label_width}}"]
        for c in range(COLS):
            cell = (r, c)
            if cell in WALLS:
                row.append(f"{'#':>{cell_width}}")
            else:
                state = state_to_index[cell]
                row.append(f"{values[state]:{cell_width}.3f}")
        rows.append(" ".join(row))
    return "\n".join(rows)


def render_policy_grid(
    policy: list[list[float]],
    state_to_index: dict[tuple[int, int], int],
) -> str:
    """Format a policy as arrows on the original grid."""
    cell_width = 6
    row_label_width = 7
    rows = [
        f"{'row/col':>{row_label_width}} "
        + " ".join(f"{c:>{cell_width}}" for c in range(COLS))
    ]
    for r in range(ROWS):
        row = [f"{r:>{row_label_width}}"]
        for c in range(COLS):
            cell = (r, c)
            if cell in WALLS:
                label = "#"
            elif cell in TERMINALS:
                label = "+1" if TERMINALS[cell] > 0 else "-1"
            else:
                state = state_to_index[cell]
                label = "".join(
                    ACTION_ARROWS[action]
                    for action in ACTIONS
                    if policy[state][action] > 0.0
                )
            row.append(f"{label:>{cell_width}}")
        rows.append(" ".join(row))
    return "\n".join(rows)


if __name__ == "__main__":
    states, s2i = build_states()

    print("states:")
    for i, cell in enumerate(states):
        print(f"{i:2d}: {cell}")

    print(f"\nnumber of usable states: {len(states)}")
    print(f"start index: {s2i[START]}")
    print(f"+1 terminal index: {s2i[(0, 3)]}")
    print(f"-1 terminal index: {s2i[(1, 3)]}")

    print("\nmovement checks:")
    checks = [
        ((2, 0), UP),
        ((2, 0), LEFT),
        ((1, 0), RIGHT),
        ((0, 2), RIGHT),
        ((0, 3), RIGHT),
    ]
    for cell, action in checks:
        print(f"{cell} + {ACTION_NAMES[action]:5s} -> {move(cell, action)}")

    print("\nslip checks:")
    for intended_action in ACTIONS:
        outcomes = action_outcomes(intended_action)
        readable = {
            ACTION_NAMES[action]: prob
            for action, prob in outcomes.items()
        }
        print(f"intend {ACTION_NAMES[intended_action]:5s}: {readable}")

    print("\nnext-cell distribution checks:")
    distribution_checks = [
        ((2, 0), UP),
        ((0, 1), RIGHT),
        ((0, 2), RIGHT),
    ]
    for cell, intended_action in distribution_checks:
        dist = next_cell_distribution(cell, intended_action)
        print(f"{cell} intending {ACTION_NAMES[intended_action]:5s}: {dist}")

    print("\ntransition checks:")
    transition_checks = [
        ((2, 0), UP),
        ((0, 2), RIGHT),
        ((0, 3), LEFT),
    ]
    for cell, intended_action in transition_checks:
        result = transitions(cell, intended_action, s2i)
        print(f"{cell} intending {ACTION_NAMES[intended_action]:5s}: {result}")

    P = build_model(states, s2i)
    print("\nmodel checks:")
    start_state = s2i[START]
    plus_terminal = s2i[(0, 3)]
    print(f"P[start][UP]       = {P[start_state][UP]}")
    print(f"P[+1 terminal][UP] = {P[plus_terminal][UP]}")
    print(f"number of states   = {len(P)}")
    print(f"actions per state  = {len(P[start_state])}")

    print("\nq_from_v checks:")
    gamma = 0.9
    V = [0.0 for _ in states]
    print(f"Q(start, all actions) with V=0: {q_from_v(P, V, start_state, gamma)}")
    print(f"Q((0, 2), all actions) with V=0: {q_from_v(P, V, s2i[(0, 2)], gamma)}")

    V[plus_terminal] = 10.0
    print(f"Q((0, 2), all actions) with V(+1)=10: {q_from_v(P, V, s2i[(0, 2)], gamma)}")

    print("\npolicy evaluation checks:")
    random_policy = uniform_random_policy(len(states))
    print("random policy:")
    print(render_policy_grid(random_policy, s2i))

    random_values = policy_evaluation(P, random_policy, gamma)
    for cell in [(0, 0), (0, 2), START, (0, 3), (1, 3)]:
        print(f"V_random({cell}) = {random_values[s2i[cell]]:.3f}")

    print("\nrandom-policy values as a grid:")
    print(render_values_grid(random_values, s2i))

    print("\ngreedy improvement checks:")
    improved_policy = greedy_policy_from_values(P, random_values, gamma)
    print(render_policy_grid(improved_policy, s2i))

    for cell in [(0, 0), (0, 2), (1, 2), START]:
        state = s2i[cell]
        q_values = q_from_v(P, random_values, state, gamma)
        best_action = improved_policy[state].index(1.0)
        readable_q = {
            ACTION_NAMES[action]: round(q_values[action], 3)
            for action in ACTIONS
        }
        print(
            f"{cell}: Q={readable_q} -> greedy {ACTION_NAMES[best_action]}"
        )

    print("\npolicy iteration checks:")
    optimal_policy, optimal_values, n_iterations = policy_iteration(P, gamma)
    print(f"policy stabilized after {n_iterations} improvement rounds")
    print("\noptimal policy:")
    print(render_policy_grid(optimal_policy, s2i))
    print("\noptimal values:")
    print(render_values_grid(optimal_values, s2i))

    print("\nvalue iteration checks:")
    vi_policy, vi_values, vi_iterations = value_iteration(P, gamma)
    print(f"values converged after {vi_iterations} max-backup sweeps")
    print("\nvalue-iteration policy:")
    print(render_policy_grid(vi_policy, s2i))
    print("\nvalue-iteration values:")
    print(render_values_grid(vi_values, s2i))
    print(f"\npolicy iteration and value iteration agree: {vi_policy == optimal_policy}")
