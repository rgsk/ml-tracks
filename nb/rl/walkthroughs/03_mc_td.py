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
# RL · 03 — Monte Carlo & TD(0): value from samples, no model

`02` computed `V^π` by *reading* `env.P[s][a]` — the exact transition probabilities and rewards.
That's planning, and no real agent gets it. A robot doesn't know the probability that its gripper
slips; it only knows that this time, it did.

So take the model away. The agent may now only **act and observe**: `reset()` to a state, `step(a)`
to get back **one** sampled `(s', r)`. Same question as `02` — what is a fixed policy `π` worth? —
but answered from a **stream of experience**.

There are exactly two honest answers, and the gap between them is the single most reused idea in the
rest of this track:

- **Monte Carlo**: wait for the episode to end, look at the return that actually happened, average it.
- **TD(0)**: don't wait. After one step you already own a guess of the rest — `V(s')` — so **bootstrap**
  off your own estimate.

Top-down as usual: we run both against `02`'s exact answer and **watch the sampled values snap onto
the ground truth in the first figure**, then go back and open each box — why MC is unbiased but
twitchy, why TD is biased but calm, and why the naive "MC vs TD" table people quote is measuring the
wrong thing.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3
_DIRS = {UP: (-1, 0), RIGHT: (0, 1), DOWN: (1, 0), LEFT: (0, -1)}
_ARROWS = {UP: "↑", RIGHT: "→", DOWN: "↓", LEFT: "←"}

# %% [markdown]
"""
## Carried over from `02` (the environment and the ground truth)

Unchanged from the previous notebook, folded into one cell you can skim past: the slippery 4×3
gridworld as an explicit MDP, plus value iteration and the exact `(I − γPπ)⁻¹rπ` solve.

Here they play a different role. `env.P` is now the **hidden truth of the world** — the algorithms
below never touch it — and the linear solve is our **grader**: the number the samples have to find.
"""


# %%
class GridWorld:
    """The 4x3 slippery gridworld as an explicit MDP model (nS, nA, P). From 02."""

    def __init__(self, noise=0.2, step_reward=-0.04, gamma=0.9):
        self.rows, self.cols = 3, 4
        self.walls = {(1, 1)}
        self.terminals = {(0, 3): +1.0, (1, 3): -1.0}
        self.start = (2, 0)
        self.noise, self.step_reward, self.gamma = noise, step_reward, gamma
        self.states = [(r, c) for r in range(self.rows) for c in range(self.cols)
                       if (r, c) not in self.walls]
        self.nS, self.nA = len(self.states), 4
        self.s2i = {cell: i for i, cell in enumerate(self.states)}
        self.P = self._build_model()

    def _move(self, cell, a):
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
                for a in range(self.nA):
                    P[s][a] = [(1.0, s, 0.0, True)]
                continue
            for a in range(self.nA):
                outcomes = {a: 1 - self.noise, (a + 1) % 4: self.noise / 2,
                            (a - 1) % 4: self.noise / 2}
                agg = {}
                for act, prob in outcomes.items():
                    nxt = self._move(cell, act)
                    ns = self.s2i[nxt]
                    if ns in agg:
                        agg[ns][0] += prob
                    else:
                        agg[ns] = [prob, self.terminals.get(nxt, self.step_reward),
                                   nxt in self.terminals]
                P[s][a] = [(p, ns, r, d) for ns, (p, r, d) in agg.items()]
        return P


def q_from_v(env, V, s, gamma):
    """One-step lookahead Q(s,a) = sum_s' p(s'|s,a)[r + gamma V(s')]. From 02."""
    return np.array([sum(p * (r + gamma * V[ns]) for p, ns, r, _ in env.P[s][a])
                     for a in range(env.nA)])


def value_iteration(env, gamma, theta=1e-10):
    """Sweep V(s) <- max_a Q(s,a), then read off the greedy policy. From 02."""
    V = np.zeros(env.nS)
    while True:
        delta = 0.0
        for s in range(env.nS):
            v_old = V[s]
            V[s] = np.max(q_from_v(env, V, s, gamma))
            delta = max(delta, abs(v_old - V[s]))
        if delta < theta:
            break
    policy = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
        policy[s, int(np.argmax(q_from_v(env, V, s, gamma)))] = 1.0
    return policy, V


def analytic_policy_value(env, policy, gamma):
    """Exact V^π by the linear solve V = (I - γP^π)^{-1} r^π. From 02. THE GRADER."""
    n = env.nS
    Ppi, rpi = np.zeros((n, n)), np.zeros(n)
    for s in range(n):
        for a in range(env.nA):
            for prob, ns, r, _ in env.P[s][a]:
                Ppi[s, ns] += policy[s, a] * prob
                rpi[s] += policy[s, a] * prob * r
    return np.linalg.solve(np.eye(n) - gamma * Ppi, rpi)


def draw_grid(env, V, ax=None, title="", vmax=None):
    """Heatmap of V over the grid (blue=+, red=−), value printed per cell. From 02."""
    grid = np.full((env.rows, env.cols), np.nan)
    for s, (r, c) in enumerate(env.states):
        grid[r, c] = V[s]
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3.2))
    vmax = vmax or (np.nanmax(np.abs(grid)) or 1.0)
    ax.imshow(grid, cmap="RdBu", norm=TwoSlopeNorm(0.0, -vmax, vmax))
    for (r, c) in env.walls:
        ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="0.4"))
    for s, (r, c) in enumerate(env.states):
        if (r, c) in env.terminals:
            ax.text(c, r, f"{env.terminals[(r, c)]:+.0f}", ha="center", va="center",
                    fontsize=13, weight="bold")
        else:
            ax.text(c, r, f"{V[s]:.3f}", ha="center", va="center", fontsize=9, color="0.1")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)
    return ax


env = GridWorld()
GAMMA = env.gamma
pi_star, _ = value_iteration(env, GAMMA)            # the policy we will evaluate
V_true = analytic_policy_value(env, pi_star, GAMMA)  # what the samples must find
NONTERM = [s for s, c in enumerate(env.states) if c not in env.terminals]
print(f"target: V^π of the optimal policy, {len(NONTERM)} non-terminal states")
print("exact values:", np.round(V_true[NONTERM], 3))

# %% [markdown]
"""
## The keyhole — all the agent is allowed to see

`Sampler` wraps the MDP and exposes **only** `reset()` and `step(a)`. Inside, it rolls its own dice
(which action actually fires after the slip) and applies the same `_move`; it never assembles a
probability table. The learner just receives the single `(s', r, done)` that fell out.

Two details worth naming:

- **Exploring starts.** Each episode begins at a uniformly random *non-terminal* state. A
  deterministic policy from the fixed start `(2,0)` would trace one corridor forever and leave most
  of the grid unvisited — and you cannot average returns for a state you never stand in. (Control in
  `04` fixes this the grown-up way, with ε-greedy behaviour.)
- **Same dynamics as `env.P`, by construction** — the sampler reuses the noise and `_move` that
  built the model, so any disagreement between samples and DP is estimation error, never a bug.
"""


# %%
class Sampler:
    """Gym-like view of the MDP: reset() to a start state, step(a) for ONE sample."""

    def __init__(self, env, rng):
        self.env, self.rng = env, rng
        self.s = None

    def reset(self):
        """Exploring start: a uniformly random non-terminal state."""
        self.s = int(self.rng.choice(NONTERM))
        return self.s

    def step(self, a):
        """One sampled transition. Returns (s', r, done) — no probabilities, ever."""
        env = self.env
        cell = env.states[self.s]
        # the slip, rolled per step: intended action w.p. 1-noise, each perpendicular w.p. noise/2
        actual = int(self.rng.choice([a, (a + 1) % 4, (a - 1) % 4],
                                     p=[1 - env.noise, env.noise / 2, env.noise / 2]))
        nxt = env._move(cell, actual)             # wall/edge bounce lives in _move
        self.s = env.s2i[nxt]
        return self.s, env.terminals.get(nxt, env.step_reward), nxt in env.terminals


def generate_episode(sampler, policy, rng, max_steps=1000):
    """Roll ONE episode under `policy`. Returns [(s, r, s_next, done), ...] in time order."""
    episode, s = [], sampler.reset()
    for _ in range(max_steps):
        a = int(rng.choice(sampler.env.nA, p=policy[s]))
        ns, r, done = sampler.step(a)
        episode.append((s, r, ns, done))
        s = ns
        if done:
            break
    return episode


# %% [markdown]
r"""
## The two estimators, in eight lines each

**Monte Carlo.** The definition of value *is* an expected return, so estimate the expectation the
most literal way possible — average the returns you saw:

$$G_t = r_{t+1} + \gamma r_{t+2} + \dots, \qquad V(s) \approx \operatorname{mean}\{G_t : s_t = s\}$$

Every `G_t` in one episode comes from a single backward sweep: `G = r + γ·G`, walked from the end.
*First-visit* MC credits only the first occurrence of a state per episode, which keeps the samples
independent across episodes.

**TD(0).** Take the Bellman expectation equation from `02`, `V(s) = E[r + γV(s')]`, and note that a
sampled transition gives you a one-draw estimate of that expectation. So nudge `V(s)` toward it:

$$\delta_t = \underbrace{r + \gamma V(s')}_{\text{TD target}} - V(s), \qquad V(s) \leftarrow V(s) + \alpha\,\delta_t$$

`δ` is the **TD error** — the same `(sample − prediction)` shape as the bandit update in `01`, except
the "sample" now contains your own current guess `V(s')`. No episode end needed: this updates every
step, online.

Both functions below take a list of `checkpoints` and snapshot `V` at each one, so a whole learning
curve costs a single stream of episodes.
"""


# %%
def returns_from_episode(episode, gamma):
    """All G_t in one O(T) backward sweep: G = r + gamma*G, walked from the end."""
    G, out = 0.0, [0.0] * len(episode)
    for t in range(len(episode) - 1, -1, -1):
        G = episode[t][1] + gamma * G
        out[t] = G
    return out


def mc_prediction(env, sampler, policy, gamma, rng, checkpoints):
    """First-visit MC: V(s) = mean of returns observed from s. Snapshots V per checkpoint."""
    total, count, snapshots, done_eps = np.zeros(env.nS), np.zeros(env.nS), [], 0
    for cp in checkpoints:
        for _ in range(cp - done_eps):
            ep = generate_episode(sampler, policy, rng)
            G = returns_from_episode(ep, gamma)
            seen = set()
            for t, (s, _r, _ns, _d) in enumerate(ep):
                if s not in seen:                 # first visit only
                    seen.add(s)
                    total[s] += G[t]
                    count[s] += 1
        done_eps = cp
        snapshots.append(np.divide(total, count, out=np.zeros(env.nS), where=count > 0))
    return snapshots


def td0_prediction(env, sampler, policy, gamma, rng, checkpoints, alpha=0.05):
    """TD(0): V(s) += step * (r + gamma*V(s') - V(s)), every transition.
    alpha=None uses a per-state 1/count step — the same decaying average MC runs on."""
    V, count, snapshots, done_eps = np.zeros(env.nS), np.zeros(env.nS), [], 0
    for cp in checkpoints:
        for _ in range(cp - done_eps):
            for (s, r, ns, done) in generate_episode(sampler, policy, rng):
                count[s] += 1
                step = alpha if alpha is not None else 1.0 / count[s]
                target = r + gamma * V[ns] * (1.0 - done)   # no bootstrap past the end
                V[s] += step * (target - V[s])
        done_eps = cp
        snapshots.append(V.copy())
    return snapshots


def rms(V_hat, V_ref=None):
    """RMS error over non-terminal states (terminals are 0 for everyone — pure dilution)."""
    d = V_hat[NONTERM] - (V_true if V_ref is None else V_ref)[NONTERM]
    return float(np.sqrt(np.mean(d ** 2)))


# %% [markdown]
"""
## The payoff — DP's answer, recovered from experience alone

Both learners get the same budget and the same policy. Nobody reads `env.P`.
"""

# %%
CHECKPOINTS = [100, 300, 1000, 3000, 10000, 30000]

rng = np.random.default_rng(0)
mc_snaps = mc_prediction(env, Sampler(env, rng), pi_star, GAMMA, rng, CHECKPOINTS)
rng = np.random.default_rng(0)                      # identical episode stream
td_snaps = td0_prediction(env, Sampler(env, rng), pi_star, GAMMA, rng, CHECKPOINTS, alpha=0.05)

fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))
vmax = np.abs(V_true).max()
draw_grid(env, V_true, ax=axes[0], title="exact V^π  (linear solve, needs the model)", vmax=vmax)
draw_grid(env, mc_snaps[-1], ax=axes[1],
          title=f"MC, {CHECKPOINTS[-1]} episodes  (RMS {rms(mc_snaps[-1]):.3f})", vmax=vmax)
draw_grid(env, td_snaps[-1], ax=axes[2],
          title=f"TD(0), α=0.05  (RMS {rms(td_snaps[-1]):.3f})", vmax=vmax)
fig.suptitle("model-free prediction recovers the planner's values", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.90))
plt.show()

# %% [markdown]
"""
Three decimal places of agreement, from nothing but `(s, r, s')` tuples. Now the learning curves —
and the first crack: one of these keeps improving and the other one **stops**.
"""

# %%
fig, ax = plt.subplots(figsize=(5.6, 3.4))
ax.plot(CHECKPOINTS, [rms(V) for V in mc_snaps], "o-", label="MC (first-visit)")
ax.plot(CHECKPOINTS, [rms(V) for V in td_snaps], "s-", label="TD(0), α=0.05")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("episodes"); ax.set_ylabel("RMS error vs exact V^π")
ax.set_title("both find V^π — but only one keeps going", fontsize=10)
ax.grid(alpha=.3, which="both"); ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
"""
Hold that thought — the plateau is **not** what most write-ups claim it is. We'll pin the blame
precisely once the boxes are open.

---

# Opening the boxes

## 1. Why any of this can work: enough samples *are* the model

DP needed `p(s'|s,a)`. MC and TD never estimate it — but they don't have to, because a **frequency is
a probability estimate**. Take one action from one state a few thousand times and count where you
land: you get `env.P`'s row back, plus noise that shrinks like `1/√n`.

Nothing below uses this table; it exists to make the premise concrete. Model-free methods skip
straight to the *values* the model would imply, which is cheaper — you never pay to estimate
dynamics you don't need.
"""

# %%
cell, action, N = (2, 0), UP, 4000
rng = np.random.default_rng(1)
probe = Sampler(env, rng)
s0 = env.s2i[cell]

counts = np.zeros(env.nS)
rsum = 0.0
for _ in range(N):
    probe.s = s0                                    # force the same start every draw
    ns, r, _ = probe.step(action)
    counts[ns] += 1
    rsum += r

model = {ns: p for p, ns, _r, _d in env.P[s0][action]}
model_r = sum(p * r for p, _ns, r, _d in env.P[s0][action])
order = sorted(model, key=lambda ns: -model[ns])

fig, ax = plt.subplots(figsize=(5.4, 3.0))
x = np.arange(len(order))
ax.bar(x - .18, [model[ns] for ns in order], .36, label="model  env.P[s][a]")
ax.bar(x + .18, [counts[ns] / N for ns in order], .36, label=f"samples  (n={N})")
ax.set_xticks(x, [str(env.states[ns]) for ns in order])
ax.set_ylabel("probability"); ax.set_title(f"from {cell}, action {_ARROWS[action]}", fontsize=10)
ax.legend(); ax.grid(axis="y", alpha=.3)
fig.tight_layout()
plt.show()
print(f"E[r]: model {model_r:+.4f}   sampled {rsum / N:+.4f}")

# %% [markdown]
"""
## 2. One episode, and the returns MC averages

An episode is a path; `returns_from_episode` turns it into one number per step. `G_t` **grows** down
the column: the later you are, the fewer `-0.04` tolls remain and the less the `+1` gets discounted.
That gradient along a single sampled path is the same "value spreads backward from the goal" picture
from `02`, now built out of one trajectory instead of a full backup.

(The episode shown is the longest of 20 rollouts, so it's a slippy one — the arrow is the *intended*
action, and it often isn't where the agent landed.)
"""

# %%
rng = np.random.default_rng(4)
sampler = Sampler(env, rng)
ep = max((generate_episode(sampler, pi_star, rng) for _ in range(20)), key=len)
G = returns_from_episode(ep, GAMMA)

print(" t | s_t     a   r_t+1 | s_t+1  |    G_t")
print("---+-------------------+--------+---------")
for t, (s, r, ns, _d) in enumerate(ep):
    a = _ARROWS[int(np.argmax(pi_star[s]))]
    print(f"{t:2d} | {str(env.states[s]):7} {a}  {r:+.2f} | {str(env.states[ns]):6} | {G[t]:+.4f}")

brute = sum(GAMMA ** k * ep[k][1] for k in range(len(ep)))
print(f"\nbackward sweep G_0 = {G[0]:+.6f}   forward sum = {brute:+.6f}   "
      f"match={np.isclose(G[0], brute)}")
print(f"G_0 is ONE sample of V^π({env.states[ep[0][0]]}) = {V_true[ep[0][0]]:+.4f} — "
      "unbiased, but this single draw is off by "
      f"{abs(G[0] - V_true[ep[0][0]]):.3f}.")

# %% [markdown]
"""
## 3. What each target actually is

This is the whole notebook in one picture. Fix a state, roll many episodes from it, and histogram the
**target** each method would learn from:

- MC's target is `G_t` — the full sampled tail. Centred on the truth (**unbiased**) but wide: every
  action choice and every slip for the rest of the episode is in there.
- TD's target is `r + γV(s')` — one real reward plus a *guess*. Narrow (**low variance**), but
  centred wherever `V(s')` currently is, which early in training is nowhere near the truth
  (**biased**).

Below, the same state's targets under a **cold** `V = 0` and a **warm** `V = V^π`. The cold panel is
the extreme case worth remembering: with `V = 0` the TD target collapses to the single reward `r =
-0.04` — a **spike**, zero variance, and completely wrong. TD's bias is a property of the current
`V`, not of the method, and it evaporates as `V` improves. MC's spread, meanwhile, never shrinks — no
amount of learning makes the environment less random.
"""

# %%
s_probe = env.s2i[(2, 2)]          # far enough from the goal that MC's tail is genuinely long
M = 3000
rng = np.random.default_rng(2)
sampler = Sampler(env, rng)

mc_targets, first_steps = [], []
for _ in range(M):
    sampler.s = s_probe                                   # force the start state
    ep, s = [], s_probe
    for _ in range(1000):
        a = int(rng.choice(env.nA, p=pi_star[s]))
        ns, r, done = sampler.step(a)
        ep.append((s, r, ns, done))
        s = ns
        if done:
            break
    mc_targets.append(returns_from_episode(ep, GAMMA)[0])
    first_steps.append((ep[0][1], ep[0][2], ep[0][3]))     # (r, s', done) of step 1

bins = np.linspace(-1.1, 1.1, 45)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), sharey=True)
for ax, (label, V_boot) in zip(axes, [("cold  V = 0", np.zeros(env.nS)),
                                      ("warm  V = V^π", V_true)]):
    td_targets = [r + GAMMA * V_boot[ns] * (1 - d) for (r, ns, d) in first_steps]
    ax.hist(mc_targets, bins=bins, alpha=.55,
            label=f"MC target G_t (std {np.std(mc_targets):.2f})")
    ax.hist(td_targets, bins=bins, alpha=.55,
            label=f"TD target r+γV(s') (std {np.std(td_targets):.2f})")
    ax.axvline(V_true[s_probe], color="k", ls="--", lw=1.2, label="true V^π(s)")
    ax.axvline(np.mean(td_targets), color="C1", ls=":", lw=1.6,
               label=f"TD mean (bias {np.mean(td_targets) - V_true[s_probe]:+.2f})")
    ax.set_title(f"{label}", fontsize=10)
    ax.set_xlabel("target value")
axes[0].set_ylabel("count"); axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
fig.suptitle(f"targets for state {env.states[s_probe]} — bias vs variance, made of histograms",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.90))
plt.show()

# %% [markdown]
"""
That trade-off is the origin of a thread that runs to the end of the track: **the target is a design
choice**. n-step returns, GAE(λ), the DQN target network, PPO's advantage estimate — all of them are
edits to this one histogram.

## 4. The honest MC-vs-TD comparison (the step size is a confound)

Here's the mistake in the payoff plot above. MC and TD differed in **two** knobs at once:

| | target | step size |
|---|---|---|
| MC | full return `G_t` | `1/n` — a true running average |
| TD | `r + γV(s')` | `α = 0.05` — constant |

A constant step never stops chasing the newest sample, so it can't converge to a point; that alone
could explain the plateau. To isolate the bootstrap, run a third learner: **TD with MC's own `1/n`
step**. Same episode streams for everyone, 12 seeds, mean ± std.
"""

# %%
BUDGETS = [200, 1000, 5000, 25000]
SEEDS = 12


def sweep(learner):
    """Run `learner(sampler, rng) -> [V per budget]` on seeds 0..SEEDS-1 (identical
    streams across learners), returning (mean, std) of RMS error per budget."""
    errs = np.zeros((SEEDS, len(BUDGETS)))
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        for j, V in enumerate(learner(Sampler(env, rng), rng)):
            errs[seed, j] = rms(V)
    return errs.mean(0), errs.std(0)


runs = {
    "MC  (1/n step)":     sweep(lambda sp, r: mc_prediction(env, sp, pi_star, GAMMA, r, BUDGETS)),
    "TD  (α=0.05)":       sweep(lambda sp, r: td0_prediction(env, sp, pi_star, GAMMA, r, BUDGETS, 0.05)),
    "TD  (1/n step)":     sweep(lambda sp, r: td0_prediction(env, sp, pi_star, GAMMA, r, BUDGETS, None)),
}

fig, ax = plt.subplots(figsize=(5.8, 3.6))
for (label, (m, sd)), style in zip(runs.items(), ["o-", "s--", "^-"]):
    ax.plot(BUDGETS, m, style, label=label)
    ax.fill_between(BUDGETS, m - sd, m + sd, alpha=.15)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("episodes"); ax.set_ylabel("RMS error vs exact V^π")
ax.set_title(f"step size vs bootstrap, disentangled ({SEEDS} seeds)", fontsize=10)
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

for label, (m, sd) in runs.items():
    print(f"{label:16} " + "  ".join(f"{v:.4f}±{s:.3f}" for v, s in zip(m, sd)))

# %% [markdown]
"""
Read it as two comparisons, never one:

- **`TD α=0.05` vs `TD 1/n`** — same target, different step. The plateau moves with the *step size*:
  give TD a decaying step and it keeps converging. So the flat line in the payoff plot was never
  evidence about bootstrapping.
- **`MC 1/n` vs `TD 1/n`** — same step, different target. *This* is the MC-vs-TD comparison, and on
  this board **MC wins at every budget** (about 3× lower error at 25000 episodes).

Two reasons, and neither is "bootstrapping is bad":

1. **The env flatters MC.** Episodes are ~4 steps and `γ=0.9`, so `G_t` has barely any tail to be
   noisy about. TD's variance advantage is worth real money when episodes are long or never
   terminate — CartPole, LLM rollouts, every environment after this notebook.
2. **`1/n` is a bad step schedule for a bootstrapper.** MC's targets are drawn from a fixed
   distribution, so averaging them all equally is exactly right. TD's targets are *non-stationary* —
   the early ones were computed against a `V` that was still wrong — yet `1/n` decays the step before
   those early mistakes can be overwritten. Bootstrapping wants a slower decay (`n^-0.6`ish) or a
   small constant.

The spread column shows the other half of the story. At 200 episodes `TD α=0.05` has by far the
**tightest** band across seeds (±0.012 vs MC's ±0.025) and the **worst** error: low-variance updates,
still anchored near the `V=0` it started from. That is the bias/variance picture from §3 playing out
in a learning curve — TD is steady long before it is accurate.

## 5. The noise floor is `≈ c·√α`

Since the plateau belongs to the step size, measure it. A constant `α` pulls `V` a fixed fraction
toward a noisy target forever, so `V` random-walks in a ball around `V^π` whose radius is set by `α`
and **not** by how much data you collect.
"""

# %%
ALPHAS = [0.2, 0.05, 0.01]
floors = {a: sweep(lambda sp, r, a=a: td0_prediction(env, sp, pi_star, GAMMA, r, BUDGETS, a))[0]
          for a in ALPHAS}

fig, ax = plt.subplots(figsize=(5.6, 3.4))
for a in ALPHAS:
    ax.plot(BUDGETS, floors[a], "o-", label=f"α = {a}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("episodes"); ax.set_ylabel("RMS error")
ax.set_title("constant-α TD: each step size buys its own floor", fontsize=10)
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

ratios = [floors[a][-1] / np.sqrt(a) for a in ALPHAS]
print("floor / √α at the largest budget: " + ", ".join(f"{r:.3f}" for r in ratios))
print(f"≈ constant ⇒ floor ≈ {np.mean(ratios):.2f}·√α. Halving α buys ~1.4× accuracy "
      "and costs ~2× the episodes to get there.")

# %% [markdown]
"""
Look along a **row** as well as down a column: at 200 episodes the *largest* `α` is winning (it has
actually moved off `V=0`), by 25000 it's losing. Fast start versus low floor, one dial — the reason
every deep-RL recipe decays its learning rate.

The theory behind it is **Robbins–Monro**: stochastic approximation converges when
`Σα = ∞` (steps big enough to reach anywhere) and `Σα² < ∞` (steps shrink enough to settle).
`1/n` satisfies both; a constant fails the second, which is exactly the floor you just measured.

Constants aren't a mistake, though — they're a choice. When the target is **non-stationary** (a
changing policy in `04`+, a moving target network in `06`), you *want* to keep chasing, and you pay
for it in precision.

## 6. Same data, different answers: the batch A/B example

The sharpest statement of what bootstrapping buys (Sutton & Barto, Example 6.4). Eight episodes,
`γ=1`, two states:

```
1×   A →(0) B →(0) end
6×   B →(1) end
1×   B →(0) end
```

Show both methods this batch repeatedly until each stops moving. They agree on `V(B) = 0.75` and
disagree flatly on `V(A)`.
"""

# %%
A_, B_ = 0, 1
BATCH = ([[(A_, 0.0, B_, False), (B_, 0.0, -1, True)]]
         + [[(B_, 1.0, -1, True)]] * 6
         + [[(B_, 0.0, -1, True)]])


def batch_mc(episodes, gamma=1.0):
    """MC fixed point: V(s) = mean observed return from s."""
    total, count = {}, {}
    for ep in episodes:
        G = 0.0
        for (s, r, _ns, _d) in reversed(ep):
            G = r + gamma * G
            total[s] = total.get(s, 0.0) + G
            count[s] = count.get(s, 0) + 1
    return {s: total[s] / count[s] for s in total}


def batch_td(episodes, gamma=1.0, alpha=0.01, sweeps=20000):
    """TD(0) fixed point: replay all transitions until V stops moving."""
    V = {A_: 0.0, B_: 0.0}
    for _ in range(sweeps):
        for ep in episodes:
            for (s, r, ns, done) in ep:
                V[s] += alpha * (r + gamma * (0.0 if done else V[ns]) - V[s])
    return V


vmc, vtd = batch_mc(BATCH), batch_td(BATCH)
print(f"MC :  V(A) = {vmc[A_]:+.3f}   V(B) = {vmc[B_]:+.3f}")
print(f"TD :  V(A) = {vtd[A_]:+.3f}   V(B) = {vtd[B_]:+.3f}")

# %% [markdown]
"""
MC is right about its own training data: the one episode through `A` returned 0, so `V(A)=0`. TD
answers a different question — it builds the **maximum-likelihood MDP** implied by the batch
(`A → B` with probability 1, `V(B)=0.75`) and solves it, giving `V(A)=0.75`. That's the
**certainty-equivalence** estimate.

The mechanism is the Markov property. Under it, what happens after you reach `B` is independent of
how you got there, so all eight `B`-episodes are evidence about `A`'s future. MC treats `A`'s value
as a question only `A`'s own episode can answer and throws away seven eighths of the relevant data.
Bootstrapping is how TD *uses the Markov assumption* — which is also why it degrades when that
assumption is false (partial observability), and MC doesn't.

## 7. The dial between them: n-step returns

MC and TD aren't a binary; they're the ends of one axis:

```
G(n) = r_1 + γr_2 + ... + γ^(n-1) r_n + γ^n · V(s_n)
        \________ n real rewards ______/   \_ a guess _/

n = 1   → TD(0)          n = ∞ → MC
```

More real reward pushes **bias** down (you rely less on a wrong `V`); a longer sampled tail pushes
**variance** up. So measure both directly: roll many episodes from each state, form `G(n)` for each
`n`, and report the bias and spread of that target. Once with a **cold** bootstrap (`V=0`, i.e. early
training) and once with a **warm** one (`V=V^π`, i.e. converged).
"""

# %%
def nstep_return(ep, n, gamma, V_boot):
    """n real rewards from the start of ep, then bootstrap V_boot(s_n) if not terminated."""
    m = min(n, len(ep))
    G, disc = 0.0, 1.0
    for k in range(m):
        G += disc * ep[k][1]
        disc *= gamma
    if n < len(ep):                       # s_n exists and is non-terminal
        G += disc * V_boot[ep[n - 1][2]]
    return G


def rollout_from(sampler, policy, rng, s0, max_steps=1000):
    """One episode forced to START at s0 (not an exploring start)."""
    ep, s = [], s0
    sampler.s = s0
    for _ in range(max_steps):
        a = int(rng.choice(sampler.env.nA, p=policy[s]))
        ns, r, done = sampler.step(a)
        ep.append((s, r, ns, done))
        s = ns
        if done:
            break
    return ep


N_LIST = [1, 2, 3, 5, 10, 30]
rng = np.random.default_rng(0)
sampler = Sampler(env, rng)
rollouts = {s: [rollout_from(sampler, pi_star, rng, s) for _ in range(2000)] for s in NONTERM}

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
for ax, (label, V_boot) in zip(axes, [("cold bootstrap  V = 0", np.zeros(env.nS)),
                                      ("warm bootstrap  V = V^π", V_true)]):
    bias, std, rmse = [], [], []
    for n in N_LIST:
        b = [np.mean([nstep_return(ep, n, GAMMA, V_boot) for ep in rollouts[s]]) - V_true[s]
             for s in NONTERM]
        v = [np.var([nstep_return(ep, n, GAMMA, V_boot) for ep in rollouts[s]]) for s in NONTERM]
        bias.append(np.sqrt(np.mean(np.square(b))))
        std.append(np.sqrt(np.mean(v)))
        rmse.append(np.sqrt(bias[-1] ** 2 + std[-1] ** 2))
    ax.plot(N_LIST, bias, "o-", label="|bias| of the target")
    ax.plot(N_LIST, std, "s-", label="std of the target")
    ax.plot(N_LIST, rmse, "^--", color="0.3", label="total error √(bias²+var)")
    ax.set_xscale("log"); ax.set_xticks(N_LIST, [str(n) for n in N_LIST])
    ax.set_xlabel("n  (1 = TD(0)  →  30 ≈ MC)")
    ax.set_title(label, fontsize=10)
    ax.grid(alpha=.3)
    print(f"{label}: best n = {N_LIST[int(np.argmin(rmse))]}")
axes[0].set_ylabel("value units"); axes[0].legend(fontsize=8)
fig.suptitle("the n-step dial: bias falls, variance rises", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.90))
plt.show()

# %% [markdown]
"""
The two panels say something practical: **the best `n` depends on how good your `V` already is.**

- Cold `V`: error falls with `n`. Early in training the bias term dominates and only real reward can
  fix it — MC-like targets win.
- Warm `V`: `n=1` is best. At convergence there's no bias left to remove, so extra steps buy nothing
  but variance.

Real training walks from the left panel to the right one, which means the ideal `n` **shrinks over
training** — and is a good reason to want a *soft* version of this dial instead of an integer.
That's GAE(λ) in `09`: an exponentially weighted average over all `n` at once, with `λ` as the knob.

## The map — what we open next

We can now *evaluate* a fixed policy from experience. What we still can't do is **improve** one:
`V(s)` alone doesn't tell you which action to take unless you can look ahead through the model — and
we just gave that up.

| next | changes | the new question |
|---|---|---|
| `04` | learn `Q(s,a)`, not `V(s)`, and let the policy chase it | **control** from experience: Q-learning (off-policy) vs SARSA (on-policy) |

The fix is one substitution: estimate `Q(s,a)` instead, and greedy improvement needs no model at all
— `argmax_a Q(s,a)` is just a lookup. The TD error you just built survives verbatim into that update,
and from there into DQN, A2C and PPO.

Next: **`04` — tabular Q-learning & SARSA: the first policy that improves itself.**
"""
