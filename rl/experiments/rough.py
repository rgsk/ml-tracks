import numpy as np
import gymnasium as gym
DIMS = ["x (cart pos)", "x_dot (cart vel)", "theta (pole angle)", "theta_dot (pole vel)"]
BOX = np.array([[-2.4, 2.4], [-3.0, 3.0], [-0.2095, 0.2095], [-3.0, 3.0]])
def rule(header):
    """Separator for a '|'-joined header: '+' under every '|', '-' everywhere else."""
    start = max(len(header) - len(header.lstrip()), 0)
    return " " * start + "".join("+" if c == "|" else "-" for c in header[start:])
def table_header(header):
    print(header)
    print(rule(header))

def header(columns, debug=False):
    """Header block for an ASCII table: the label rows, then the rule.

    A column is a string, or a tuple of 2+ lines whose extra rows are centered
    under the first. Column width = the widest of that column's own lines.
    debug=True prepends a row showing each column's computed width."""
    cols = [(c,) if isinstance(c, str) else tuple(c) for c in columns]
    widths = [max(map(len, c)) for c in cols]

    lines = []
    for r in range(max(map(len, cols))):
        cells = []
        for col, w in zip(cols, widths):
            text = col[r] if r < len(col) else ""
            cells.append(text.ljust(w) if r == 0 else text.center(w))
        lines.append(" | ".join(cells))

    rule = "".join("+" if ch == "|" else "-" for ch in lines[0])
    if debug:
        lines.insert(0, " | ".join(str(w).center(w) for w in widths))
    lines.append(rule)
    print("\n".join(line.rstrip() for line in lines))


def rollout_states(num_episodes=200, seed=0):
    env = gym.make('CartPole-v1')
    rng = np.random.default_rng(seed)
    states, deltas = [], []
    for ep in range(num_episodes):
        s, _ = env.reset(seed = seed + ep)
        done = False
        while not done:
            states.append(s)
            action = int(rng.integers(2))
            ns, _r, term, trunc, _ = env.step(action)
            deltas.append(np.abs(ns - s))
            done = term or trunc
            s = ns
        states.append(s)
    return np.array(states), np.array(deltas)

def to_cell(states, bins):
    lo, hi = BOX[:, 0], BOX[:, 1]
    idx = ((states - lo) / (hi - lo) * bins).astype(int)
    idx = np.clip(idx, 0, bins - 1)
    # each state as a 4-digit number written in base bins
    # for bins=3
    # id = i0*bins³ + i1*bins² + i2*bins¹ + i3
    # [1,0,2,1] -> 34
    result = np.ravel_multi_index(idx.T, (bins,) * 4)
    return result

def exp_table_dies(num_episodes=200, seed=0):
    states, deltas = rollout_states(num_episodes, seed)
    print(f'{states.shape=}, {deltas.shape=}')
    T = len(states)
    print(f'{num_episodes} random-policy episodes, {T:,} states visited')

    uniq = len(np.unique(states, axis=0))
    print('A')
    print(f'distinct raw states: {uniq:,} / {T:,}')
    print(f'repeats: {T - uniq}')
    print('B')
    env = gym.make('CartPole-v1')
    space = env.observation_space
    table_header(f'{'dim':22s} | observed min | observed max | env bounds')

    for i, name in enumerate(DIMS):
        bound = f'[{space.low[i]:.2f}, {space.high[i]:.2f}]'
        print(
            f"{name:22s} | {states[:, i].min():^+12.3f} | {states[:, i].max():^+12.3f}"
            f" | {bound}"
        )
    print('C')
    table_header(f"bins/dim | table cells | (s,a) pairs | cells seen | coverage | visits/seen")
    for bins in (3, 5, 10, 20, 50):
        cells = bins ** 4
        c = to_cell(states, bins)
        counts = np.bincount(c)
        seen = int((counts > 0).sum())
        # 2 actions per state
        sa = 2 * cells
        print(
            f"{bins:8d} | {cells:11,d} | {sa:11,d} | {seen:10,d}"
            f" | {100 * seen/cells:7.2f}% | {counts[counts > 0].mean():11.1f}"
        )
    print('D')
    print("one step changes the state by (mean |s' - s| per dim):")
    for i, name in enumerate(DIMS):
        print(f'{name:22} {deltas[:, i].mean():.4f}')

    print()
    header([
        'bins/dim',
        'theta bin width',
        ('steps to cross one theta bin', '(bin width / mean step)'),
        ('mean steps', 'inside a cell'),
    ])
    for bins in (3, 5, 10, 20, 50):
        width = (BOX[2, 1] - BOX[2, 0]) / bins
        cross = width / deltas[:, 2].mean()
        c = to_cell(states, bins)
        changes = int((c[1:] != c[:-1]).sum())
        dwell = len(c) / (changes + 1)
        print(f'{bins:8d} | {width:^15.4f} | {f'{cross:>4.1f}':^28} | {dwell:^13.2f}')



    print('\n\n----End----')


def run_experiments():
    exp_table_dies()

if __name__ == "__main__":
    run_experiments()
