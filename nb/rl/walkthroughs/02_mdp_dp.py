# ---
# jupyter:
#   jupytext:
#     cell_markers: '"""'
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
"""
# RL · 02 — MDPs & dynamic programming: valuing a sequence of decisions

In `01` there was **one state** — pulling an arm changed nothing. A **Markov Decision Process** adds
the thing that makes RL *RL*: you're in a state `s`, take action `a`, land in a new state `s'`
(stochastically), collect reward `r`, and repeat. Now a choice has *consequences down the line*, so a
good action isn't the one with the best immediate reward — it's the one that sets up the best *future*.

When you **know the model** (`p(s'|s,a)` and `r`), you don't need to learn from experience at all —
you can **plan** the optimal policy directly with **dynamic programming**. That's this notebook, and
it matters because it's the **ground truth** every later model-free method (`03` MC/TD, `04`
Q-learning) is secretly trying to approximate from samples.

Top-down: we'll build the classic 4×3 slippery gridworld, then **run value iteration and watch the
optimal value light up the grid and the arrows snap to the best path** — before deriving why the
backups converge.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# action encoding (clockwise, so (a+1)%4 / (a-1)%4 are the two perpendicular slips)
UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
_DIRS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}
_ARROWS = {UP: "↑", RIGHT: "→", DOWN: "↓", LEFT: "←"}

# %% [markdown]
"""
## The environment — a slippery gridworld as an explicit MDP

The Russell & Norvig 4×3 world:

```
  . . . +1        start at bottom-left (2,0)
  . # . -1        # is a wall you can't enter
  . . . .         +1 / -1 are terminal (absorbing)
```

Actions move UP/RIGHT/DOWN/LEFT, but the floor is **slippery**: with prob `0.8` you go where you
intended, and `0.1` each you veer to one of the two **perpendicular** directions. Bumping a wall or
edge leaves you in place. Every non-terminal move costs `-0.04` (a "living reward" that makes dawdling
expensive) — so the slip is what makes `p(s'|s,a)` interesting and the policy near the `-1` trap
non-obvious (you'll see the optimal policy deliberately steer *away* from the trap even when that's
the long way round).

The class exposes exactly what the DP algorithms consume, and nothing gridworld-specific — so the
routines below work on **any** tabular MDP:

- `nS, nA, gamma`
- `P[s][a] = list of (prob, s_next, reward, done)` outcomes summing to prob 1.
"""


# %%
class GridWorld:
    """The 4x3 slippery gridworld as an EXPLICIT MDP model (nS, nA, P)."""

    def __init__(self, noise: float = 0.2, step_reward: float = -0.04, gamma: float = 0.9):
        self.rows, self.cols = 3, 4
        self.walls = {(1, 1)}
        self.terminals = {(0, 3): +1.0, (1, 3): -1.0}
        self.start = (2, 0)
        self.noise = noise                  # total slip prob, split over the 2 perps
        self.step_reward = step_reward
        self.gamma = gamma

        self.states = [(r, c) for r in range(self.rows) for c in range(self.cols)
                       if (r, c) not in self.walls]
        self.nS = len(self.states)
        self.nA = 4
        self.s2i = {cell: i for i, cell in enumerate(self.states)}
        self.P = self._build_model()

    def _move(self, cell, a):
        """Where action a lands you from `cell`, ignoring slip. Wall/edge => stay."""
        dr, dc = _DIRS[a]
        nxt = (cell[0] + dr, cell[1] + dc)
        if (not 0 <= nxt[0] < self.rows or not 0 <= nxt[1] < self.cols or nxt in self.walls):
            return cell
        return nxt

    def _build_model(self):
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for cell in self.states:
            s = self.s2i[cell]
            if cell in self.terminals:
                # absorbing: every action self-loops with zero reward => V=0 there.
                for a in range(self.nA):
                    P[s][a] = [(1.0, s, 0.0, True)]
                continue
            for a in range(self.nA):
                outcomes = {a: 1 - self.noise,
                            (a + 1) % 4: self.noise / 2,
                            (a - 1) % 4: self.noise / 2}
                agg = {}                    # next_state -> [prob, reward, done]
                for act, prob in outcomes.items():
                    nxt = self._move(cell, act)
                    reward = self.terminals.get(nxt, self.step_reward)
                    done = nxt in self.terminals
                    ns = self.s2i[nxt]
                    if ns in agg:            # two slips can land in the same cell
                        agg[ns][0] += prob
                    else:
                        agg[ns] = [prob, reward, done]
                P[s][a] = [(p, ns, r, d) for ns, (p, r, d) in agg.items()]
        return P


env = GridWorld()
print(f"nS={env.nS} states, nA={env.nA} actions, gamma={env.gamma}")
print("outcomes of action RIGHT from the start (2,0) — note the 0.8/0.1/0.1 slip:")
for prob, ns, r, done in env.P[env.s2i[env.start]][RIGHT]:
    print(f"  p={prob:.1f} -> {env.states[ns]}  reward={r:+.2f}  done={done}")

# %% [markdown]
"""
A small helper to **see** any `(V, policy)` on the grid — a value heatmap with the greedy action
drawn as an arrow in each cell. We'll lean on this for the payoff.
"""


# %%
def draw_grid(env, V, policy=None, ax=None, title=""):
    """Heatmap of V over the grid (blue=+, red=−), with greedy-action arrows if policy given."""
    grid = np.full((env.rows, env.cols), np.nan)
    for s, (r, c) in enumerate(env.states):
        grid[r, c] = V[s]
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3.2))
    vmax = np.nanmax(np.abs(grid)) or 1.0
    ax.imshow(grid, cmap="RdBu", norm=TwoSlopeNorm(0.0, -vmax, vmax))
    for (r, c) in env.walls:
        ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="0.4"))
    for s, (r, c) in enumerate(env.states):
        if (r, c) in env.terminals:
            ax.text(c, r, f"{env.terminals[(r, c)]:+.0f}", ha="center", va="center",
                    fontsize=13, weight="bold")
            continue
        ax.text(c, r - .28, f"{V[s]:.2f}", ha="center", va="center", fontsize=7, color="0.15")
        if policy is not None:
            ax.text(c, r + .12, _ARROWS[int(np.argmax(policy[s]))], ha="center", va="center",
                    fontsize=15)
    ax.text(*env.start[::-1], "S", ha="center", va="bottom", fontsize=8, color="green", weight="bold")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)
    return ax


# %% [markdown]
r"""
## The two value functions, and the Bellman equations

- `V^π(s)` = expected discounted return starting in `s`, then following policy `π`.
- `Q^π(s,a)` = same, but **force action `a` first**, then follow `π`.

They're tied together by the **Bellman equations** — a value equals immediate reward plus discounted
value of where you land:

$$Q^\pi(s,a) = \sum_{s'} p(s'\mid s,a)\,\big[\,r + \gamma\,V^\pi(s')\,\big] \qquad\text{(one-step lookahead)}$$
$$V^\pi(s) = \sum_a \pi(a\mid s)\,Q^\pi(s,a) \qquad\text{(expectation backup)}$$

The **optimal** value takes the best action instead of averaging over `π`:

$$V^*(s) = \max_a \sum_{s'} p(s'\mid s,a)\,\big[\,r + \gamma\,V^*(s')\,\big] \qquad\text{(optimality backup)}$$

`q_from_v` is that one-step lookahead — the shared core every routine below calls.
"""


# %%
def q_from_v(env, V, s, gamma):
    """One-step lookahead: Q(s,a) = sum_s' p(s'|s,a)[ r + gamma V(s') ], shape (nA,)."""
    # `done` needs no special case: terminals self-loop with reward 0 and keep V=0,
    # so gamma*V[s'] is already 0 for them.
    q = np.zeros(env.nA)
    for a in range(env.nA):
        for prob, ns, r, done in env.P[s][a]:
            q[a] += prob * (r + gamma * V[ns])
    return q


# %% [markdown]
r"""
## Policy evaluation — and why it's *secretly a linear solve*

**Iterative policy evaluation**: sweep the expectation backup `V(s) ← Σ_a π(a|s) Q(s,a)` until `V^π`
stops moving. But the Bellman expectation equation is **linear** in `V`, so `V^π` has a closed form:

$$V^\pi = (I - \gamma P^\pi)^{-1} r^\pi$$

where `P^π(s,s') = Σ_a π(a|s) p(s'|s,a)` and `r^π(s)` is the expected immediate reward. That's a nice
interview point — *evaluation is just a linear system*; the DP sweep is one cheap iterative way to
solve it without inverting a matrix. We implement both and check they agree.
"""


# %%
def policy_evaluation(env, policy, gamma, theta=1e-10):
    """Iterative policy evaluation: V(s) <- sum_a π(a|s) Q(s,a), swept to convergence.
    `policy` is a stochastic (nS, nA) matrix; returns V of shape (nS,)."""
    # In-place sweep (reads freshly-updated V) — converges a touch faster than a sync copy.
    V = np.zeros(env.nS)
    while True:
        delta = 0.0
        for s in range(env.nS):
            v_old = V[s]
            V[s] = policy[s] @ q_from_v(env, V, s, gamma)
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    return V


def analytic_policy_value(env, policy, gamma):
    """Ground-truth V^π by the linear solve V = (I - γP^π)^{-1} r^π (no iteration)."""
    n = env.nS
    Ppi = np.zeros((n, n))
    rpi = np.zeros(n)
    for s in range(n):
        for a in range(env.nA):
            pa = policy[s, a]
            for prob, ns, r, _ in env.P[s][a]:
                Ppi[s, ns] += pa * prob
                rpi[s] += pa * prob * r
    return np.linalg.solve(np.eye(n) - gamma * Ppi, rpi)


rand_pi = np.ones((env.nS, env.nA)) / env.nA        # uniform-random policy
V_iter = policy_evaluation(env, rand_pi, env.gamma)
V_exact = analytic_policy_value(env, rand_pi, env.gamma)
assert np.allclose(V_iter, V_exact, atol=1e-6)
print(f"iterative eval matches the (I-γPπ)⁻¹rπ solve  (max diff {np.abs(V_iter-V_exact).max():.2e}) ✅")

# %% [markdown]
"""
## Policy iteration and value iteration

Two ways to reach the *optimal* policy from the model:

- **Policy iteration**: evaluate `π`, then act **greedily** w.r.t. `V^π` to get a strictly better `π`;
  repeat. Provably converges to `π*` (usually in very few iterations).
- **Value iteration**: skip full evaluation — just sweep the **optimality** backup `V(s) ← max_a
  Q(s,a)` directly, then read the greedy policy off `V*`.
"""


# %%
def policy_improvement(env, V, gamma):
    """Greedy improvement: one-hot (nS, nA) policy that's argmax over the one-step lookahead."""
    policy = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
        policy[s, int(np.argmax(q_from_v(env, V, s, gamma)))] = 1.0
    return policy


def policy_iteration(env, gamma, theta=1e-10):
    """Alternate evaluation + greedy improvement until the policy stops changing."""
    policy = np.ones((env.nS, env.nA)) / env.nA     # start from uniform-random
    while True:
        V = policy_evaluation(env, policy, gamma, theta)
        new_policy = policy_improvement(env, V, gamma)
        if (np.argmax(policy, 1) == np.argmax(new_policy, 1)).all():
            break
        policy = new_policy
    return policy, V


def value_iteration(env, gamma, theta=1e-10):
    """Sweep the optimality backup V(s) <- max_a Q(s,a), then read off the greedy policy."""
    V = np.zeros(env.nS)
    while True:
        delta = 0.0
        for s in range(env.nS):
            v_old = V[s]
            V[s] = np.max(q_from_v(env, V, s, gamma))
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    return policy_improvement(env, V, gamma), V


# %% [markdown]
"""
**Wiring checks** — the two algorithms must agree on `V*` and `π*`; the optimal value must dominate
the random policy everywhere; and the greedy policy (intended moves) must actually walk from the
start to the `+1` goal.
"""

# %%
pi_pol, pi_V = policy_iteration(env, env.gamma)
vi_pol, vi_V = value_iteration(env, env.gamma)
assert np.allclose(pi_V, vi_V, atol=1e-6)
assert (np.argmax(pi_pol, 1) == np.argmax(vi_pol, 1)).all()
assert (vi_V >= V_exact - 1e-9).all()               # V* dominates V^random

cell, path = env.start, [env.start]
for _ in range(50):
    if cell in env.terminals:
        break
    cell = env._move(cell, int(np.argmax(vi_pol[env.s2i[cell]])))
    path.append(cell)
assert cell == (0, 3)
print("PI == VI on V* and π*  ✅")
print("V* dominates V^random everywhere  ✅")
print(f"greedy rollout reaches +1 goal:  {path}  ✅")

# %% [markdown]
"""
## The payoff — the optimal plan, and V spreading out from the goal

**Top row:** the random policy's values vs the optimal `V*` with the optimal policy's arrows. Notice
the optimal policy in the cell just below the `-1` trap points **away** from it — it would rather take
the slippery long way than risk a slip into `-1`.
"""

# %%
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
draw_grid(env, V_exact, rand_pi, ax=a1, title="V^π  (uniform-random policy)")
draw_grid(env, vi_V, vi_pol, ax=a2, title="V*  and optimal policy π*")
fig.tight_layout()
plt.show()

# %% [markdown]
"""
**Watch V propagate.** Value iteration starts at `V=0` everywhere; each sweep pushes value one more
step outward from the `+1` goal. Here are the first few sweeps — the "good news" of the goal
diffusing backward through the grid until it stabilises.
"""

# %%
snapshots, V = [], np.zeros(env.nS)
for k in range(1, 9):
    for s in range(env.nS):
        V[s] = np.max(q_from_v(env, V, s, env.gamma))
    snapshots.append((k, V.copy()))

show = [snapshots[i] for i in (0, 1, 2, 4, 7)]
fig, axes = plt.subplots(1, len(show), figsize=(3.0 * len(show), 2.7))
for ax, (k, Vk) in zip(axes, show):
    draw_grid(env, Vk, ax=ax, title=f"after sweep {k}")
fig.suptitle("value iteration — V diffusing backward from the +1 goal", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
plt.show()

# %% [markdown]
"""
## The map — what we open next

We just **planned** optimally by *knowing* the model. That's the ceiling; the rest of the tabular
track earns the same result **without** the model — from samples alone:

| next | drops | the new question |
|---|---|---|
| `03` | the known `p(s'|s,a)` | estimate `V^π` from **sampled episodes** — MC vs TD(0), bias/variance |
| `04` | the model *and* evaluation-only | learn the **optimal** policy from experience (Q-learning, SARSA) |

The one-step lookahead `q_from_v` and the backup `V ← max_a Q` are the exact targets those
model-free methods approximate: MC/TD estimate the `Σ p[r + γV(s')]` expectation by *averaging real
transitions* instead of reading `env.P`.

Next: **`03` — Monte Carlo & TD(0): estimating value from samples, no model.**
"""
