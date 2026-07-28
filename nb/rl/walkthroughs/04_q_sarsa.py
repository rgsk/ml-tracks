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
# RL · 04 — Q-learning & SARSA: the first policy that improves itself

Everything so far has been handed a policy. `02` planned with a known model; `03` dropped the model
but still *evaluated* a policy someone gave us — and look where that policy came from: value
iteration, i.e. the model. The one thing we have never done is **find a good way to act from
experience alone**.

That's **control**, and it is the whole job. From here to the end of the track — DQN, PPO, RLHF —
every algorithm is a control algorithm. This notebook builds the tabular original, and two of the
three ideas in it survive verbatim into all of them.

The setting changes in exactly two ways, and each forces a new idea:

- **No policy is given.** To pick an action you must compare actions, and `V(s)` can't do that
  without a model. So learn `Q(s,a)`.
- **Nobody guarantees coverage.** `03` used exploring starts to visit every state. A control agent
  starts where the world puts it and generates its **own** training data — so it has to deliberately
  try things it doesn't currently believe in. That's ε-greedy, the bandit tension from `01`, now per
  state.

Then one design choice splits the field in half — *which* next-action value you bootstrap from — and
it produces two algorithms that behave visibly differently on the same board. Top-down as usual: we
watch an agent go from random to optimal first, then open every box.
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
## Carried over from `02`/`03` (skim past)

The slippery 4×3 gridworld as an explicit MDP, value iteration, and the exact `(I − γPπ)⁻¹rπ` solve.
As in `03`, `env.P` is the **hidden truth** — no algorithm below reads it — and the linear solve is
the **grader**.

The grading question is different now, though, and it's worth being precise about. We are no longer
asking "is `V` accurate?" but **"is the learned policy any good?"** So the metric throughout is: take
the agent's greedy policy `argmax_a Q(s,a)`, and compute its *true* value under the model. That number
is honest even when the `Q` table itself is off — which, as §6 shows, it systematically is.
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
    """One-step lookahead Q(s,a) = sum_s' p(s'|s,a)[r + gamma V(s')]. From 02. NEEDS THE MODEL."""
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


env = GridWorld()
GAMMA = env.gamma
NONTERM = [s for s, c in enumerate(env.states) if c not in env.terminals]
pi_star, _ = value_iteration(env, GAMMA)
V_star = analytic_policy_value(env, pi_star, GAMMA)                    # the ceiling
Q_star = np.array([q_from_v(env, V_star, s, GAMMA) for s in range(env.nS)])
V_rand = analytic_policy_value(env, np.ones((env.nS, env.nA)) / env.nA, GAMMA)   # the floor
OPT_ACT = pi_star.argmax(1)


def greedy_policy(Q):
    """Deterministic greedy policy as a one-hot (nS, nA) matrix."""
    pi = np.zeros_like(Q)
    pi[np.arange(len(Q)), Q.argmax(1)] = 1.0
    return pi


def policy_value(Q):
    """THE METRIC: the TRUE value of the agent's greedy policy, from the model."""
    return analytic_policy_value(env, greedy_policy(Q), GAMMA)


def score(Q):
    """One number: mean true value of the greedy policy over non-terminal states."""
    return float(policy_value(Q)[NONTERM].mean())


print(f"floor   (random policy) : {V_rand[NONTERM].mean():+.3f}")
print(f"ceiling (optimal π*)    : {V_star[NONTERM].mean():+.3f}")
print(f"start state (2,0):  random {V_rand[env.s2i[env.start]]:+.3f}   "
      f"optimal {V_star[env.s2i[env.start]]:+.3f}")

# %% [markdown]
"""
## The keyhole, minus the training wheels

Same black-box `reset`/`step` contract as `03` — with one deliberate downgrade: **`reset()` returns
the fixed start `(2,0)`**, not a random state.

`03`'s exploring starts were a cheat we could afford because the policy was fixed. A control agent
lives where it lands: it always begins at the start cell, and the only way it ever sees the rest of
the board is by *choosing* to go there. Coverage is now the agent's problem, not the environment's —
which is exactly what §2 is about.
"""


# %%
class GridSampler:
    """Gym-like view of the MDP: reset()/step(a) only, never env.P. Fixed start."""

    def __init__(self, env, rng):
        self.env, self.rng = env, rng
        self.nS, self.nA = env.nS, env.nA
        self.s = None

    def reset(self):
        self.s = self.env.s2i[self.env.start]
        return self.s

    def step(self, a):
        """One sampled transition (s', r, done). The env rolls its own slip dice."""
        e = self.env
        cell = e.states[self.s]
        actual = int(self.rng.choice([a, (a + 1) % 4, (a - 1) % 4],
                                     p=[1 - e.noise, e.noise / 2, e.noise / 2]))
        nxt = e._move(cell, actual)
        self.s = e.s2i[nxt]
        return self.s, e.terminals.get(nxt, e.step_reward), nxt in e.terminals


# %% [markdown]
r"""
## The whole agent

Three short functions, and the third is the second with one word changed.

**ε-greedy** — act on your current beliefs, but not always. Ties are broken *randomly*: at init `Q`
is all zeros, so every action ties, and always taking `argmax`'s first index would bias exploration
into a corner.

**SARSA** — take `03`'s TD update and move it from `V` to `Q`. After `(s, a, r, s')`, choose the next
action `a'` the way you will actually act, and bootstrap off **that**:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\,[\,r + \gamma\,Q(s',a') - Q(s,a)\,]$$

The update literally consumes the tuple `(S, A, R, S', A')` — that's the name. Because `a'` is the
action it goes on to take, SARSA evaluates the policy it is **following**, exploration included.

**Q-learning** — bootstrap off the *best* next action instead, whatever you actually do:

$$Q(s,a) \leftarrow Q(s,a) + \alpha\,[\,r + \gamma\,\max_{a'} Q(s',a') - Q(s,a)\,]$$

That `max` is the Bellman **optimality** backup from `02` — the operator value iteration swept over
the whole model — applied to one sampled transition at a time.

And the part that is easy to miss on first read: **nobody told either of them what good behaviour
is.** They act greedily w.r.t. a `Q` that is being updated toward the returns of that very behaviour.
Evaluation nudges the policy, the improved policy generates better data, which nudges evaluation —
*generalized policy iteration*, one transition at a time, with ε keeping the loop from closing too
early. `eps` may be a constant or a function of training progress (`0 → 1`).
"""


# %%
def epsilon_greedy(Q, s, eps, rng):
    """Random action w.p. eps (EXPLORE), else argmax_a Q[s,a] with random tie-break (EXPLOIT)."""
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    q = Q[s]
    return int(rng.choice(np.flatnonzero(q == q.max())))


def sarsa(sampler, gamma, checkpoints, alpha, rng, eps=0.1, q_init=0.0, max_steps=200):
    """ON-policy TD control. Snapshots Q at each checkpoint (episode count)."""
    Q = np.full((sampler.nS, sampler.nA), q_init)
    snaps, done_eps, total = [], 0, checkpoints[-1]
    for cp in checkpoints:
        for ep in range(done_eps, cp):
            e = eps(ep / total) if callable(eps) else eps
            s = sampler.reset()
            a = epsilon_greedy(Q, s, e, rng)              # the first A
            for _ in range(max_steps):
                ns, r, done = sampler.step(a)
                na = epsilon_greedy(Q, ns, e, rng)        # the second A — chosen ONCE
                target = r + gamma * Q[ns, na] * (1.0 - done)     # <-- bootstrap off a'
                Q[s, a] += alpha * (target - Q[s, a])
                s, a = ns, na                             # ...and actually taken next
                if done:
                    break
        done_eps = cp
        snaps.append(Q.copy())
    return snaps


def q_learning(sampler, gamma, checkpoints, alpha, rng, eps=0.1, q_init=0.0, max_steps=200):
    """OFF-policy TD control. Identical to sarsa() except for the target line."""
    Q = np.full((sampler.nS, sampler.nA), q_init)
    snaps, done_eps, total = [], 0, checkpoints[-1]
    for cp in checkpoints:
        for ep in range(done_eps, cp):
            e = eps(ep / total) if callable(eps) else eps
            s = sampler.reset()
            for _ in range(max_steps):
                a = epsilon_greedy(Q, s, e, rng)          # behave ε-greedily...
                ns, r, done = sampler.step(a)
                target = r + gamma * Q[ns].max() * (1.0 - done)   # <-- ...but TARGET the max
                Q[s, a] += alpha * (target - Q[s, a])
                s = ns
                if done:
                    break
        done_eps = cp
        snaps.append(Q.copy())
    return snaps


def anneal(eps0=1.0, eps_min=0.05):
    """Linear ε decay over training: explore hard early, exploit once Q sharpens."""
    return lambda p: max(eps_min, eps0 * (1.0 - p))


# %% [markdown]
"""
## The payoff — a policy, learned from nothing but experience

Both agents start from `Q = 0`, always at `(2,0)`, and are never told the goal exists. Five seeds
each; the curve is the **true value of the greedy policy** at each checkpoint, graded by the model
the agents can't see.
"""

# %%
CHECKPOINTS = [1, 10, 30, 100, 300, 1000, 3000, 10000]
SEEDS = 5


def run_seeds(algo, seeds=SEEDS, checkpoints=CHECKPOINTS, alpha=0.05, eps=None, **kw):
    """Run `algo` on seeds 0..seeds-1, returning a list of snapshot-lists."""
    eps = anneal() if eps is None else eps
    return [algo(GridSampler(env, np.random.default_rng(s)), GAMMA, checkpoints, alpha,
                 np.random.default_rng(s), eps=eps, **kw) for s in range(seeds)]


runs = {"Q-learning": run_seeds(q_learning), "SARSA": run_seeds(sarsa)}
curves = {k: np.array([[score(Q) for Q in snaps] for snaps in v]) for k, v in runs.items()}

print("episodes | " + " | ".join(f"{k:>10}" for k in curves))
for j, cp in enumerate(CHECKPOINTS):
    print(f"{cp:8d} | " + " | ".join(f"{curves[k][:, j].mean():+10.3f}" for k in curves))
print(f"{'random':>8} | " + f"{V_rand[NONTERM].mean():+10.3f}")
print(f"{'optimal':>8} | " + f"{V_star[NONTERM].mean():+10.3f}")


# %%
def draw_policy(env, Q, ax, title="", vmax=None, show_ties=True):
    """Greedy arrows over a max_a Q heatmap; '?' where every action still ties."""
    grid = np.full((env.rows, env.cols), np.nan)
    for s, (r, c) in enumerate(env.states):
        grid[r, c] = Q[s].max()
    vmax = vmax or (np.nanmax(np.abs(grid)) or 1.0)
    ax.imshow(grid, cmap="RdBu", norm=TwoSlopeNorm(0.0, -vmax, vmax))
    for (r, c) in env.walls:
        ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, color="0.4"))
    for s, (r, c) in enumerate(env.states):
        if (r, c) in env.terminals:
            ax.text(c, r, f"{env.terminals[(r, c)]:+.0f}", ha="center", va="center",
                    fontsize=12, weight="bold")
        elif show_ties and np.ptp(Q[s]) < 1e-12:
            ax.text(c, r, "?", ha="center", va="center", fontsize=13, color="0.35")
        else:
            ax.text(c, r, _ARROWS[int(Q[s].argmax())], ha="center", va="center", fontsize=16)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)


fig = plt.figure(figsize=(10.5, 5.4))
gs = fig.add_gridspec(2, 4, height_ratios=[1.5, 1.0], hspace=0.35)
ax = fig.add_subplot(gs[0, :])
for (label, c), style in zip(curves.items(), ["o-", "s--"]):
    m, sd = c.mean(0), c.std(0)
    ax.plot(CHECKPOINTS, m, style, label=label)
    ax.fill_between(CHECKPOINTS, m - sd, m + sd, alpha=.15)
ax.axhline(V_star[NONTERM].mean(), color="k", ls=":", lw=1.2, label="optimal π* (value iteration)")
ax.axhline(V_rand[NONTERM].mean(), color="0.5", ls=":", lw=1.2, label="random policy")
ax.set_xscale("log")
ax.set_xlabel("episodes of experience"); ax.set_ylabel("true value of the greedy policy")
ax.set_title(f"control from experience: no model, no policy given ({SEEDS} seeds)", fontsize=11)
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8, loc="lower right")

strip = [0, 1, 4, 7]                      # 1, 10, 300, 10000 episodes
for k, j in enumerate(strip):
    draw_policy(env, runs["Q-learning"][0][j], fig.add_subplot(gs[1, k]),
                title=f"{CHECKPOINTS[j]} episode" + ("s" if CHECKPOINTS[j] > 1 else ""))
fig.text(0.5, 0.40, "Q-learning's greedy policy as experience accumulates  "
                    "(colour = max_a Q,  ? = no opinion yet)",
         ha="center", fontsize=9)
plt.show()

# %% [markdown]
"""
From `-0.365` to within a whisker of the `+0.522` ceiling, and the bottom strip shows *how*. After one
episode most of the board is a `?` — every action still tied at zero, no opinion to be greedy about.
After ten, the whole grid has arrows but almost no *value*: a faint negative tint everywhere from the
`-0.04` tolls, with the only positive cells sitting next to the `+1`. Then the blue spreads outward,
goal-first, until it reaches the start corner. Nobody wrote down "avoid the `-1`" or "the goal is at
`(0,3)`"; both fell out of the arithmetic.

Two things not to gloss over. The curves **wobble** near the top instead of settling — Q-learning
even dips around 1000 episodes — and they stop a hair short of the ceiling; §3 tracks down exactly
which decision is responsible. And the two algorithms land in nearly the same place, so on this board
they are hard to tell apart — which is why §5 goes to a board where they aren't.

---

# Opening the boxes

## 1. Why control learns `Q` and not `V`

In `02` the greedy action came from a one-step lookahead:

`a* = argmax_a Σ_s' p(s'|s,a) [ r + γ V(s') ]`

Read that formula as an agent, not a mathematician: to use it you must be able to *ask the world*
where each action would take you. That's `p(s'|s,a)` — the thing we gave up. A perfectly accurate
`V` is useless for choosing without it.

`Q(s,a)` is the same information, pre-chewed: the value of *committing to `a` here*, then acting
well. Greedy becomes `argmax_a Q[s,a]` — a table lookup, no model at decision time. That's the whole
reason control learns the bigger table.

Below: estimate `Q(s0, a)` for all four actions from **samples only** (force the action, then follow
π*, average the returns), and check the argmax against the model's `q_from_v`. The rollout policy is
π* only so that the quantity being estimated is well defined (`Q^{π*}`); the point is the *decision
rule* — `argmax` over a table of numbers you can measure, versus a lookahead you cannot.
"""


# %%
def mc_q_estimate(sampler, policy, s0, a0, n, rng, max_steps=200):
    """Q(s0,a0) by Monte Carlo: force a0, then follow `policy` to the end. Uses only reset/step."""
    total = 0.0
    for _ in range(n):
        sampler.s = s0                                  # force the state...
        ns, r, done = sampler.step(a0)                  # ...and the first action
        G, disc, s, steps = r, GAMMA, ns, 0
        while not done and steps < max_steps:
            a = int(rng.choice(sampler.nA, p=policy[s]))
            ns, r, done = sampler.step(a)
            G += disc * r
            disc *= GAMMA
            s, steps = ns, steps + 1
        total += G
    return total / n


rng = np.random.default_rng(0)
sampler = GridSampler(env, rng)
s0 = env.s2i[env.start]
q_sampled = np.array([mc_q_estimate(sampler, pi_star, s0, a, 5000, rng) for a in range(env.nA)])

print(f"state {env.start}")
print("  a       |  model Q (reads env.P) | sampled Q (reset/step only)")
print("----------+------------------------+----------------------------")
for a in range(env.nA):
    print(f"  {a}  {_ARROWS[a]}    |         {Q_star[s0, a]:+.3f}         |"
          f"           {q_sampled[a]:+.3f}")
print(f"\n  greedy action:  model → {_ARROWS[int(Q_star[s0].argmax())]}   "
      f"sampled → {_ARROWS[int(q_sampled.argmax())]}   "
      f"(match = {Q_star[s0].argmax() == q_sampled.argmax()})")
print("  A decision made without ever touching a transition probability.")

# %% [markdown]
"""
## 2. Exploration is now the agent's job

`03` got coverage for free: the policy was fixed and exploring starts dropped the agent everywhere.
Now the agent writes its own curriculum, and a purely greedy one writes a very short one — it takes
its current-best action in every state, so the other actions get **zero** data forever and can never
be revalued. TD control's convergence guarantee is built on visiting every `(s,a)` infinitely often;
greedy behaviour violates it on episode one.

Left/middle panels: `(s,a)` visit counts while *behaving* (no learning) under a fully-trained `Q`.
Greedy tries 9 of the 36 pairs and will never try a tenth. ε-greedy reaches 35 in 3000 episodes — the
holdout is a corner pair behind the trap, which is its own lesson: uniform ε explores the *far* parts
of a world very slowly, and that gap is what exploration bonuses (and `01`'s UCB) exist to close.

But there's a trap in the obvious demo, and it's worth walking into on purpose. Set `ε = 0` and
train, and the agent... mostly still solves it. That is not a reprieve — it's `01`'s **optimistic
initialization** sneaking in. Every reward here is negative, so `Q = 0` is a wildly optimistic start:
any untried action looks better than one that has been tried and disappointed, and greedy exploration
happens *by accident*. Initialize pessimistically (`Q = -1`) and the accident stops (right panel).
"""

# %%
COV_EPISODES = 3000
visit_maps = {}
for eps in (0.0, 0.1):
    sp = GridSampler(env, np.random.default_rng(0))
    visits = np.zeros((env.nS, env.nA))
    for _ in range(COV_EPISODES):
        s = sp.reset()
        for _ in range(200):
            a = epsilon_greedy(Q_star, s, eps, sp.rng)
            visits[s, a] += 1
            s, _r, done = sp.step(a)
            if done:
                break
    visit_maps[eps] = visits[NONTERM]
    sub = visits[NONTERM]
    print(f"ε={eps}: covered {int((sub > 0).sum()):2d}/{sub.size} (s,a) pairs, "
          f"never sampled {int((sub == 0).sum()):2d}, fewest visits {int(sub.min())}")

# %%
INIT_BUDGETS = [200, 2000, 10000]
init_runs = {}
for q_init in (0.0, -1.0):
    for eps in (0.0, 0.1):
        rs = run_seeds(q_learning, seeds=6, checkpoints=INIT_BUDGETS, eps=eps, q_init=q_init)
        init_runs[(q_init, eps)] = np.array([[score(Q) for Q in snaps] for snaps in rs])
        print(f"Q init {q_init:+.1f}, ε={eps}: greedy-policy value " +
              " ".join(f"{v:+.3f}" for v in init_runs[(q_init, eps)].mean(0)) +
              f"   (worst seed at the end {init_runs[(q_init, eps)][:, -1].min():+.3f})")

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))
for ax, eps in zip(axes[:2], (0.0, 0.1)):
    # log scale: the interesting difference is "rarely" vs "never", not the busy pairs
    im = ax.imshow(np.log10(visit_maps[eps] + 1), cmap="viridis", aspect="auto",
                   vmin=0, vmax=np.log10(max(v.max() for v in visit_maps.values()) + 1))
    ax.set_xticks(range(4), [_ARROWS[a] for a in range(4)], fontsize=13)
    ax.set_yticks(range(len(NONTERM)), [str(env.states[s]) for s in NONTERM], fontsize=7)
    ax.set_title(f"(s,a) visits while behaving, ε={eps}\n"
                 f"{int((visit_maps[eps] > 0).sum())}/36 pairs ever tried", fontsize=9)
    for i in range(len(NONTERM)):
        for j in range(4):
            if visit_maps[eps][i, j] == 0:
                ax.text(j, i, "0", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=.046, label="log₁₀(1 + visits)")

ax = axes[2]
x = np.arange(len(INIT_BUDGETS))
for k, ((q_init, eps), c) in enumerate(init_runs.items()):
    ax.plot(x, c.mean(0), "o-" if eps else "s--",
            label=f"Q₀={q_init:+.0f}, ε={eps}")
ax.axhline(V_star[NONTERM].mean(), color="k", ls=":", lw=1.2, label="optimal")
ax.axhline(V_rand[NONTERM].mean(), color="0.5", ls=":", lw=1.2, label="random")
ax.set_xticks(x, [str(b) for b in INIT_BUDGETS])
ax.set_xlabel("episodes"); ax.set_ylabel("value of greedy policy")
ax.set_title("greedy (ε=0) survives only by\nlucky optimistic init", fontsize=9)
ax.grid(alpha=.3); ax.legend(fontsize=7)
fig.tight_layout()
plt.show()

# %% [markdown]
"""
The pessimistic-greedy line is the one to stare at: **flat**, at `-0.269`, identical at 200 and at
10000 episodes. Not "slower" — frozen. It committed to a route in its first few episodes and then
spent ten thousand more confirming it, because the alternative actions never got a single sample. Add
ε to that same agent with that same init and it climbs back into the normal range (`+0.47` by 10000
and still rising) — the init only cost it a slow start, not the task.

Two takeaways that outlive the tabular setting: exploration bugs look like *plateaus*, not crashes;
and how you initialize your value estimates is itself an exploration decision (`01`'s optimistic
init, and later the entropy bonus in `10`, do the same job with different machinery).

## 3. How good is "solved", exactly?

The payoff curve flattens just under the ceiling. That last sliver is worth chasing because the
reason for it is not about control at all — it's `03`'s noise floor, arriving on schedule.

Count how many of the 9 non-terminal states get the **optimal action**, and look at where the misses
are. First, how tight is each decision? For every state, the true gap between the best action and the
runner-up:
"""

# %%
print("state   |  true Q*(s,·)  [↑ → ↓ ←]              | best | gap to runner-up")
print("--------+---------------------------------------+------+-----------------")
for s in NONTERM:
    q = Q_star[s]
    gap = q.max() - np.sort(q)[-2]
    print(f" {str(env.states[s]):6} | {np.array2string(q, precision=3, sign='+'):38} |"
          f"  {_ARROWS[int(q.argmax())]}   | {gap:.3f}"
          + ("   <-- tightest" if gap < 0.04 else ""))

for name, snaps_list in runs.items():
    match = [int((snaps[-1].argmax(1)[NONTERM] == OPT_ACT[NONTERM]).sum()) for snaps in snaps_list]
    missed = sorted({env.states[s] for snaps in snaps_list for s in NONTERM
                     if snaps[-1][s].argmax() != OPT_ACT[s]})
    print(f"\n{name}: optimal actions per seed {match} / 9   "
          f"mean policy value {np.mean([score(s[-1]) for s in snaps_list]):+.4f} "
          f"(optimal {V_star[NONTERM].mean():+.4f})")
    print(f"{'':{len(name)}}  states ever given the wrong action: {missed}")

# %% [markdown]
"""
The misses cluster on `(2,1)` and `(2,3)` — the two tightest decisions on the board (gaps of `0.039`
and `0.037`, versus `0.067–0.127` everywhere else). And the culprit is the **constant step size**: `α=0.05`
keeps `Q` random-walking in a ball around the truth forever (`03` §5 measured that floor as `≈0.21·√α`),
and a `0.039` gap sits inside that ball.

So this is a *precision* failure, not a control failure — and it should be fixable by the same dial
`03` used. Rerun with a smaller `α`, and with a per-`(s,a)` `1/n` step:
"""

# %%
FINE_BUDGET = [30000]


def q_learning_step(sampler, gamma, checkpoints, alpha, rng, eps, max_steps=200):
    """Q-learning with alpha=None meaning a per-(s,a) 1/count step (03's decaying average)."""
    Q = np.zeros((sampler.nS, sampler.nA))
    cnt = np.zeros((sampler.nS, sampler.nA))
    snaps, done_eps, total = [], 0, checkpoints[-1]
    for cp in checkpoints:
        for ep in range(done_eps, cp):
            e = eps(ep / total) if callable(eps) else eps
            s = sampler.reset()
            for _ in range(max_steps):
                a = epsilon_greedy(Q, s, e, rng)
                ns, r, done = sampler.step(a)
                cnt[s, a] += 1
                step = alpha if alpha is not None else 1.0 / cnt[s, a]
                Q[s, a] += step * (r + gamma * Q[ns].max() * (1.0 - done) - Q[s, a])
                s = ns
                if done:
                    break
        done_eps = cp
        snaps.append(Q.copy())
    return snaps


s21 = env.s2i[(2, 1)]
print("step size   | seeds recovering π* exactly | mean policy value | learned Q(2,1) [↑ → ↓ ←]")
print("------------+-----------------------------+-------------------+--------------------------")
for label, alpha in (("α = 0.05", 0.05), ("α = 0.01", 0.01), ("1/n", None)):
    Qs = [q_learning_step(GridSampler(env, np.random.default_rng(s)), GAMMA, FINE_BUDGET, alpha,
                          np.random.default_rng(s), anneal())[-1] for s in range(6)]
    exact = sum(int((Q.argmax(1)[NONTERM] == OPT_ACT[NONTERM]).all()) for Q in Qs)
    print(f" {label:10} |            {exact}/6              |      {np.mean([score(Q) for Q in Qs]):+.4f}      |"
          f" {np.array2string(np.mean(Qs, 0)[s21], precision=3, sign='+')}")
print(f"{'':12}|                             |      {V_star[NONTERM].mean():+.4f}      |"
      f" {np.array2string(Q_star[s21], precision=3, sign='+')}   <- truth")

# %% [markdown]
"""
Diagnosis confirmed: shrink the step and the near-tie resolves, in every seed, to the exactly-optimal
policy. The `α=0.05` agent was never confused about the task — it was just vibrating.

Worth carrying forward: **"my policy is 98% of optimal" is usually a step-size or exploration
statement, not an algorithm statement.** Check those two before rewriting the algorithm.

## 4. The one line that differs, and what each one converges to

Now the real subject of this notebook. Same data, same behaviour, one word:

```-
SARSA        target = r + γ·Q[s', a']      a' ~ ε-greedy — the action it WILL take
Q-learning   target = r + γ·max_a Q[s', a]           the best action, taken or not
```

That changes *what the table means*:

- SARSA's target is the value of the policy it is actually running — ε and all. It converges to
  `Q^{π_ε}`, the value of **ε-greedy behaviour**. Its behaviour policy and its target policy are the
  same object: **on-policy**.
- Q-learning's `max` prices in a perfectly greedy future, no matter how carelessly it is behaving. It
  converges to `Q*` — the optimal values — while following whatever exploratory policy you like:
  **off-policy**. (That property is the reason DQN can learn from a replay buffer of stale
  experience, and why `06` works at all.)

This is testable. Sweep ε, train both to convergence, and ask what the tables believe. For SARSA,
compare `Σ_a π_ε(a|s)·Q(s,a)` — the value of its own ε-greedy behaviour — against the model's exact
value for the best ε-soft policy. For Q-learning, compare `max_a Q(s,a)` against `V*`.
"""


# %%
def eps_soft_matrix(greedy_act, eps):
    """The ε-greedy policy around a set of greedy actions, as a (nS, nA) matrix."""
    P = np.full((env.nS, env.nA), eps / env.nA)
    P[np.arange(env.nS), greedy_act] += 1 - eps
    return P


def eps_soft_optimal(eps, iters=200):
    """Policy iteration INSIDE the ε-soft class: improve greedily, but be graded on the
    ε-greedy policy you actually run. The best an ε-exploring agent can possibly do."""
    ga = np.zeros(env.nS, dtype=int)
    for _ in range(iters):
        V = analytic_policy_value(env, eps_soft_matrix(ga, eps), GAMMA)
        new = np.array([q_from_v(env, V, s, GAMMA).argmax() for s in range(env.nS)])
        if (new == ga).all():
            break
        ga = new
    return V


EPS_GRID = [0.05, 0.1, 0.2, 0.3, 0.4]
SWEEP_BUDGET = [10000]
rows = {"SARSA (on-policy)": [], "Q-learning (off-policy)": []}
soft_opt = []
for eps in EPS_GRID:
    soft_opt.append(eps_soft_optimal(eps)[NONTERM].mean())
    sa, ql = [], []
    for seed in range(3):
        Qs = sarsa(GridSampler(env, np.random.default_rng(seed)), GAMMA, SWEEP_BUDGET, 0.02,
                   np.random.default_rng(seed), eps=eps)[-1]
        Qq = q_learning(GridSampler(env, np.random.default_rng(seed)), GAMMA, SWEEP_BUDGET, 0.02,
                        np.random.default_rng(seed), eps=eps)[-1]
        sa.append((eps_soft_matrix(Qs.argmax(1), eps) * Qs)[NONTERM].sum(1).mean())
        ql.append(Qq.max(1)[NONTERM].mean())
    rows["SARSA (on-policy)"].append(np.mean(sa))
    rows["Q-learning (off-policy)"].append(np.mean(ql))

fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(EPS_GRID, soft_opt, "k--", lw=1.4, label="exact value of ε-greedy behaviour (model)")
ax.plot(EPS_GRID, rows["SARSA (on-policy)"], "s-", label="SARSA: Σ π_ε(a|s) Q(s,a)")
ax.plot(EPS_GRID, rows["Q-learning (off-policy)"], "o-", label="Q-learning: max_a Q(s,a)")
ax.axhline(V_star[NONTERM].mean(), color="0.3", ls=":", lw=1.2, label="V* (optimal)")
ax.set_xlabel("ε used during training (constant)"); ax.set_ylabel("mean value over states")
ax.set_title("what each table converges to", fontsize=10)
ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

for k, v in rows.items():
    print(f"{k:24} " + "  ".join(f"{x:+.3f}" for x in v))
print(f"{'ε-soft optimum (exact)':24} " + "  ".join(f"{x:+.3f}" for x in soft_opt))
print(f"{'V* (exact)':24} " + "  ".join(f"{V_star[NONTERM].mean():+.3f}" for _ in EPS_GRID))

# %% [markdown]
"""
Two flat-vs-sloping lines, and that's the entire on/off-policy distinction as a measurement. SARSA
lands on the dashed curve at every ε (`+0.478` vs `+0.491`, `+0.388` vs `+0.390`, `+0.242` vs
`+0.234`): its table really is reporting the value of the exploring policy it runs. Q-learning's stays
flat at `V*` to within seed noise — it is answering "what if I acted perfectly from here?", a question
whose answer does not depend on ε at all.

(Q-learning's line is not just flat but slightly *above* `V*` at the larger ε — `+0.529` at `ε=0.3`
against a true `+0.522`. That overshoot is not noise; it is the `max` inflating its own estimates,
which is §6.)

Neither is "more correct". They answer different questions, and §5 is the case where the difference
in the *answer* becomes a difference in the *behaviour*.

## 5. Cliff Walking — where the two visibly disagree

Sutton & Barto's Example 6.6, the classic. A 4×12 board, deterministic moves, no discounting; every
step costs `-1`; the bottom row between start and goal is a cliff worth `-100` and a teleport back to
the start.

```-
 .  .  .  .  .  .  .  .  .  .  .  .
 .  .  .  .  .  .  .  .  .  .  .  .
 .  .  .  .  .  .  .  .  .  .  .  .     <- row 2: the optimal route, along the edge
 S  C  C  C  C  C  C  C  C  C  C  G
```

The optimal policy hugs the cliff edge: 13 steps. But the agent explores with ε, and a random step
down from the edge is a `-100`. So there are two defensible answers, and the two algorithms each pick
one — not because we tuned them, but because of that single `max`.
"""


# %%
class Cliff:
    """Cliff Walking (S&B Example 6.6). Deterministic; bottom row (cols 1..10) is the cliff."""
    rows, cols, nA = 4, 12, 4

    def __init__(self):
        self.nS = self.rows * self.cols
        self.start, self.goal = (3, 0), (3, 11)
        self.s = None

    def _i(self, rc):
        return rc[0] * self.cols + rc[1]

    def reset(self):
        self.s = self._i(self.start)
        return self.s

    def step(self, a):
        r, c = divmod(self.s, self.cols)
        dr, dc = _DIRS[a]
        nr, nc = min(max(r + dr, 0), self.rows - 1), min(max(c + dc, 0), self.cols - 1)
        if nr == 3 and 1 <= nc <= 10:                  # stepped off
            self.s = self._i(self.start)
            return self.s, -100.0, False
        self.s = self._i((nr, nc))
        return self.s, -1.0, (nr, nc) == self.goal


def train_cliff(kind, seed=0, episodes=500, alpha=0.5, eps=0.1, max_steps=2000):
    """Train on the cliff, recording the ONLINE return actually earned each episode."""
    cliff, rng = Cliff(), np.random.default_rng(seed)
    Q, online = np.zeros((cliff.nS, cliff.nA)), []
    for ep in range(episodes):
        e = eps(ep / episodes) if callable(eps) else eps
        s, G = cliff.reset(), 0.0
        a = epsilon_greedy(Q, s, e, rng)
        for _ in range(max_steps):
            if kind == "q":
                a = epsilon_greedy(Q, s, e, rng)
            ns, r, done = cliff.step(a)
            G += r
            if kind == "sarsa":
                na = epsilon_greedy(Q, ns, e, rng)
                Q[s, a] += alpha * (r + Q[ns, na] * (1.0 - done) - Q[s, a])
                s, a = ns, na
            else:
                Q[s, a] += alpha * (r + Q[ns].max() * (1.0 - done) - Q[s, a])
                s = ns
            if done:
                break
        online.append(G)
    return Q, np.array(online), cliff


def render_cliff(cliff, Q, title):
    print(title)
    for r in range(cliff.rows):
        row = ""
        for c in range(cliff.cols):
            if (r, c) == cliff.goal:
                row += " G"
            elif r == 3 and 1 <= c <= 10:
                row += " ▓"
            elif (r, c) == cliff.start:
                row += " S"
            else:
                row += " " + _ARROWS[int(Q[cliff._i((r, c))].argmax())]
        print(row)


def greedy_path(cliff, Q, maxlen=100):
    """Follow the greedy path from S: (length, closest row to the cliff, reached goal?)."""
    s, rows, steps = cliff.reset(), [], 0
    for _ in range(maxlen):
        rows.append(s // cliff.cols)
        ns, _r, done = cliff.step(int(Q[s].argmax()))
        steps += 1
        if done:
            return steps, min(rows), True
        if ns == s:
            break
        s = ns
    return steps, min(rows), False


CLIFF_SEEDS = 10
cliff_curves, cliff_Q = {}, {}
for kind, name in (("sarsa", "SARSA"), ("q", "Q-learning")):
    runs_c = [train_cliff(kind, seed=s) for s in range(CLIFF_SEEDS)]
    cliff_curves[name] = np.array([o for _Q, o, _c in runs_c])
    cliff_Q[name] = runs_c[0][0]
    paths = [greedy_path(runs_c[0][2], Q) for Q, _o, _c in runs_c]
    print(f"{name:11}: greedy path reaches the goal in {sum(p[2] for p in paths)}/{CLIFF_SEEDS} seeds, "
          f"lengths {sorted(p[0] for p in paths if p[2])}")
    print(f"{'':11}  closest row to the cliff (2 = the optimal edge route): {[p[1] for p in paths]}")
    print(f"{'':11}  online return, last 100 episodes: "
          f"{cliff_curves[name][:, -100:].mean():+.1f}")

cliff_env = Cliff()
print()
render_cliff(cliff_env, cliff_Q["SARSA"], "SARSA (on-policy) — the safe detour:")
print()
render_cliff(cliff_env, cliff_Q["Q-learning"], "Q-learning (off-policy) — straight along the edge:")

# %%
fig, ax = plt.subplots(figsize=(6.4, 3.6))
w = 20
for name, c in cliff_curves.items():
    m = c.mean(0)
    sm = np.convolve(m, np.ones(w) / w, mode="valid")
    ax.plot(np.arange(len(sm)) + w, sm, label=name)
ax.set_ylim(-120, 0)
ax.set_xlabel("episode"); ax.set_ylabel(f"online return (mean of {CLIFF_SEEDS} seeds, smoothed)")
ax.set_title("the reward actually earned while learning (ε=0.1 throughout)", fontsize=10)
ax.grid(alpha=.3); ax.legend(fontsize=9)
fig.tight_layout()
plt.show()

# %% [markdown]
"""
**Q-learning finds the optimal policy and earns less reward doing it.** Its greedy path is 13 steps,
exactly optimal, in all ten seeds; SARSA's is 17 or 19, hugging the top row. Yet during training SARSA
collects roughly `-27` per episode against Q-learning's `-47`, because Q-learning keeps walking the
edge with a 10% chance of a `-100` plunge on every step.

(SARSA's greedy path completes in 8 of 10 seeds; in the other two it stalls in a loop up in the top
rows. Those cells are visited a handful of times at most — with a constant `ε` and `α=0.5` their
arrows are still noise. It never walks the edge in any seed, which is the claim that matters here.)

That is not a bug in either one. It is the definition of on- vs off-policy showing up as money:
optimal-policy-if-you-act-perfectly ≠ best-behaviour-while-you-are-still-exploring.

Look at the actual numbers at an edge cell to see the mechanism — the `Q` row at `(2,5)`, one square
above the cliff:
"""

# %%
for cell in [(2, 5), (1, 5)]:
    i = cliff_env._i(cell)
    print(f"state {cell}   [↑ → ↓ ←]" + ("        (edge cell, cliff directly below)"
                                         if cell == (2, 5) else "        (one row higher)"))
    for name in ("SARSA", "Q-learning"):
        q = cliff_Q[name][i]
        print(f"  {name:11} {np.array2string(q, precision=1, sign='+', floatmode='fixed'):32}"
              f"  greedy {_ARROWS[int(q.argmax())]}")
    print()
print("optimal cost-to-go from (2,5) if you never slip: -7  (6 steps right, 1 down)")

# %% [markdown]
"""
Q-learning's `→` at the edge is worth exactly `-7.0` — the perfect-play distance to the goal. Its
`max` target assumes the *next* move is greedy too, so the ε-risk it is actually running never enters
the arithmetic. SARSA's `→` is much worse than its `↑`/`←`: bootstrapping off `a'` means the 10%
chance that `a'` is a step into the void gets averaged into the value of *being there at all*, and
that penalty propagates back along the whole edge. One row up, at `(1,5)`, the two even choose
opposite directions — Q-learning steps down toward the shortest path, SARSA climbs away.

**An honest caveat, since the textbook version invites the wrong conclusion.** "SARSA is being
cautious because ε is high, so anneal ε and it will converge to the optimal path" sounds right and is
not what happens here:
"""

# %%
print("closest row to the cliff (2 = optimal edge route, 0 = top row), 4 seeds each")
print(" α    ε      | SARSA        | Q-learning")
print("-------------+--------------+-----------")
for alpha in (0.5, 0.1):
    for eps in (0.1, 0.01):
        out = {}
        for kind in ("sarsa", "q"):
            out[kind] = [greedy_path(c, Q)[1]
                         for Q, _o, c in (train_cliff(kind, seed=s, episodes=1000,
                                                      alpha=alpha, eps=eps) for s in range(4))]
        print(f" {alpha:<4} {eps:<6} | {str(out['sarsa']):12} | {out['q']}")

# %% [markdown]
"""
Q-learning walks the edge in every single run; SARSA never does, at any of these settings. Shrinking
ε does not bring it back, and the reason is the same coverage argument as §2, wearing a different
hat: **once SARSA prefers the safe route it stops walking the edge, so the data that would revalue
the edge never arrives.** The theory ("as ε→0 SARSA's fixed point → `Q*`") assumes every `(s,a)` is
visited infinitely often — a nearly-greedy agent quietly stops paying that bill.

Which is the real lesson to take to the deep-RL notebooks: off-policy methods can *keep* learning
about actions they no longer take. On-policy methods only learn about the road they are on. That is
precisely the trade you re-make when you choose PPO (on-policy, stable, sample-hungry) over
DQN/SAC (off-policy, replay-driven, sample-efficient) in `06`–`10`.

## 6. The dark side of that `max`

One more consequence of `max`, and it becomes a named bug in `07`. Q-learning's target uses
`max_a Q(s',a)` where every `Q` is a **noisy estimate**. The maximum of noisy estimates is biased
upward — `E[max] ≥ max E` — even when every true value is identical, because the max preferentially
selects whichever action got lucky:
"""

# %%
rng = np.random.default_rng(0)
print("k equal-value actions, each estimated from 10 samples of N(0,1). True max = 0.000")
for k in (1, 2, 4, 10, 50):
    est = rng.standard_normal((20000, k, 10)).mean(2)
    print(f"  k = {k:2d}   E[max_a Q̂(a)] = {est.max(1).mean():+.3f}")

# %% [markdown]
"""
The same table both *picks* the argmax and *reports* its value, so selection noise and evaluation
noise are the same noise, and they compound instead of cancelling.

S&B's Example 6.7 turns that statistic into a wrong decision. From state `A`: `right` ends the
episode at `0`; `left` moves to `B`, from which 10 actions each end the episode with reward
`N(-0.1, 1)`. Every route through `B` is worth `-0.1`, so the optimal action at `A` is `right`, and an
ε-greedy optimal agent would take `left` about `ε/2 = 5%` of the time.

**Double Q-learning** is the fix and it is almost trivial: keep two tables, let one *choose* the
argmax and the other *report* its value. The two noises are now independent, so they no longer
conspire.
"""


# %%
def run_ab(double, runs=1000, episodes=300, alpha=0.1, eps=0.1, nB=10, seed=0):
    """S&B Example 6.7. Returns the fraction of runs taking 'left' at A, per episode."""
    rng = np.random.default_rng(seed)
    left = np.zeros(episodes)

    def pick(q):
        if rng.random() < eps:
            return int(rng.integers(len(q)))
        return int(rng.choice(np.flatnonzero(q == q.max())))

    for _ in range(runs):
        QA = [np.zeros(2), np.zeros(2)]                  # plain Q-learning uses table 0 only
        QB = [np.zeros(nB), np.zeros(nB)]
        for ep in range(episodes):
            if pick((QA[0] + QA[1]) if double else QA[0]) == 1:        # 1 = 'left'
                left[ep] += 1
                aB = pick((QB[0] + QB[1]) if double else QB[0])
                r = rng.normal(-0.1, 1.0)
                if double:
                    j = int(rng.integers(2))
                    QB[j][aB] += alpha * (r - QB[j][aB])
                    i = int(rng.integers(2))
                    # table i CHOOSES the argmax, table 1-i REPORTS its value
                    QA[i][1] += alpha * (QB[1 - i][int(QB[i].argmax())] - QA[i][1])
                else:
                    QB[0][aB] += alpha * (r - QB[0][aB])
                    QA[0][1] += alpha * (QB[0].max() - QA[0][1])       # the same table does both
            else:
                i = int(rng.integers(2)) if double else 0
                QA[i][0] += alpha * (0.0 - QA[i][0])
    return left / runs


left_q, left_d = run_ab(double=False), run_ab(double=True)

fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.plot(100 * left_q, label="Q-learning")
ax.plot(100 * left_d, label="Double Q-learning")
ax.axhline(5, color="k", ls=":", lw=1.2, label="optimal (ε/2 = 5%)")
ax.set_xlabel("episode"); ax.set_ylabel("% of runs taking the bad action 'left'")
ax.set_title("maximization bias: a statistical artefact becomes a wrong decision", fontsize=10)
ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout()
plt.show()

print("episode |  Q-learning  |  Double-Q   (optimal ≈ 5%)")
for ep in (1, 5, 10, 25, 50, 100, 300):
    print(f"  {ep:4d}  |    {100 * left_q[ep - 1]:5.1f}%    |   {100 * left_d[ep - 1]:5.1f}%")

# %% [markdown]
"""
Plain Q-learning spends the early episodes convinced that the losing action is great, because one of
`B`'s ten noisy estimates is always flattering. It does recover here — the tabular estimates
eventually average out — but "recovers eventually, given a stationary target and enough visits" is
exactly the promise that breaks once `Q` is a neural network generalizing across states. Then this
bug stops being transient, and `07` fixes it with the identical trick, renamed **Double DQN**.

## The map — what we open next

We now have the complete tabular control loop, and it already contains three of the ideas the rest of
the track keeps reusing: bootstrap a target, act ε-greedily on your own estimates, and choose whether
that target is on- or off-policy.

What breaks next is not the algorithm — it's the **table**. `Q` here has `9 × 4 = 36` entries, one per
`(s,a)`, all learned independently: nothing the agent learns about `(2,0)` transfers to `(2,1)`, and a
state it has never visited has no value at all. CartPole's state is four real numbers, so the table
would need infinitely many rows. The only way forward is to replace the lookup with a **function**
`Q(s,a; θ)` that generalizes between states.

| next | changes | the new question |
|---|---|---|
| `05` | `Q` becomes a neural network | why do the updates that just worked now **diverge**? (the deadly triad) |

Swapping in a network breaks things that were free in this notebook — the targets move, the samples
are correlated, and the bootstrap chases itself — which is what makes `06`'s two fixes (replay buffer,
target network) worth their complexity.

Next: **`05` — function approximation: the Q-table becomes a network, and stops converging.**
"""
