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
# RL · 01 — multi-armed bandits: exploration vs exploitation

The whole RL track starts here, at the **simplest possible problem that is still RL**: a slot machine
with `k` arms. There is **one state** and **no transitions** — pulling an arm doesn't change anything
about the world — so there's no sequential credit assignment yet (that's MDPs, `02`). What's left is
the single tension that never goes away: with a finite budget of pulls, how do you **spend** them
between trying arms you're unsure about (**explore**) and milking the best arm so far (**exploit**)?

Top-down as always: we build three agents, **run them, and watch the learning curves** — greedy
plateaus, ε-greedy and UCB keep closing in. *Then* we go back and open each box: the incremental
value update, why greedy fails, and why "optimism" explores without any randomness at all.

> This is the cleanest place the explore/exploit dial appears. It comes back the whole track long:
> ε-greedy → UCB → entropy bonus → KL-to-reference in RLHF.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

rng_global = np.random.default_rng(0)

# %% [markdown]
"""
## The environment — a Gaussian bandit

`k` arms. Each arm `a` has a fixed but **unknown** true value `q*(a)`, drawn once from `Normal(0,1)`.
Pulling arm `a` pays a noisy reward `~ Normal(q*(a), 1)`. "Stationary" = `q*` never moves. The agent
never sees `q*`; it only sees the rewards it collects, and has to back out which arm is best.
"""


# %%
class GaussianBandit:
    """k arms. Arm a pays reward ~ Normal(q*(a), 1). Stationary (q* fixed)."""

    def __init__(self, k: int = 10, seed: int | None = None):
        self.k = k
        self.rng = np.random.default_rng(seed)
        self.q_star = self.rng.normal(0.0, 1.0, size=k)   # true action values, drawn once
        self.best_arm = int(np.argmax(self.q_star))

    def pull(self, a: int) -> float:
        """Noisy reward for arm a: q*(a) plus unit Gaussian noise."""
        return float(self.q_star[a] + self.rng.normal(0.0, 1.0))


# a quick look at one 10-armed bandit: the true means the agent must discover
b = GaussianBandit(k=10, seed=1)
print("true q*(a):", np.round(b.q_star, 2))
print(f"best arm = {b.best_arm}  (q* = {b.q_star[b.best_arm]:.2f})")

# %% [markdown]
r"""
## The core object — action-value estimates `Q(a)`

`Q(a)` is our running estimate of `q*(a)`. The honest estimate is just the **average reward seen from
arm `a`** so far. Storing every reward and re-averaging wastes memory, so we keep it **incrementally**.
After the `N`-th pull of arm `a` giving reward `R`:

$$Q(a) \leftarrow Q(a) + \frac{1}{N}\,\big(R - Q(a)\big)$$

Read it as: nudge the old estimate toward the new reward by step `1/N`. The bracket `(R − Q)` is an
**error** (sample minus prediction) — the *exact* shape of a TD error (`03`) and of a gradient step
later. Why this equals the plain average: expand the average of the first `N` samples and peel off the
last one,

$$Q_N = \frac{1}{N}\sum_{i=1}^{N} R_i
      = \frac{1}{N}\Big(R_N + (N-1)Q_{N-1}\Big)
      = Q_{N-1} + \frac{1}{N}\big(R_N - Q_{N-1}\big).$$

So the one-line incremental update is bit-for-bit the sample mean — no approximation. (A **constant**
step `α` instead of `1/N` would forget old rewards and *track* a non-stationary arm; we stay with
`1/N` here since `q*` is fixed.)
"""

# %% [markdown]
"""
## Three ways to pick an arm

All three sit on top of the same `Q(a)`; they differ only in *how they explore*:

- **greedy** — `argmax_a Q(a)`. Pure exploitation. Latches onto whatever looked good after a couple of
  lucky early pulls and **never discovers a better arm**. The baseline that *fails*.
- **ε-greedy** — greedy, but with probability `ε` pick a **uniformly random** arm. Dead simple;
  explores forever at a constant low rate. `ε = 0` recovers plain greedy.
- **UCB** (upper confidence bound) — `argmax_a [ Q(a) + c·sqrt(ln t / N(a)) ]`. The second term is an
  **optimism bonus**: large for arms pulled few times (small `N(a)`), shrinking as you learn. So it
  directs exploration at the genuinely *uncertain* arms instead of random ones — no coin-flips needed.

**Optimistic init** is a fourth trick that needs no exploration rule at all: start every `Q(a)` way
above the true means, so *every untried arm looks like the best one* and even plain greedy is forced
to sweep them early.
"""


# %%
class BanditAgent:
    """Holds Q(a) estimates + pull counts N(a), and the selection rules."""

    def __init__(self, k: int, epsilon: float = 0.0, c: float = 0.0,
                 optimistic: float = 0.0, seed: int | None = None):
        self.k = k
        self.epsilon = epsilon              # exploration rate for ε-greedy
        self.c = c                          # exploration strength for UCB (0 => off)
        self.rng = np.random.default_rng(seed)
        self.Q = np.full(k, float(optimistic))   # optimistic init: start high -> forces early sweep
        self.N = np.zeros(k, dtype=int)
        self.t = 0                          # total pulls so far (for UCB's ln t)

    def select(self) -> int:
        """UCB if c > 0, else ε-greedy (ε may be 0, i.e. plain greedy)."""
        self.t += 1

        if self.c > 0:
            # untried arms get an infinite bonus -> pulled first (and dodges 0-divide)
            bonus = np.full(self.k, np.inf)
            tried = self.N > 0
            bonus[tried] = self.c * np.sqrt(np.log(self.t) / self.N[tried])
            return int(np.argmax(self.Q + bonus))

        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.k))
        return int(np.argmax(self.Q))

    def update(self, a: int, reward: float) -> None:
        """Incremental sample-average update: Q[a] += (1/N[a]) * (reward - Q[a])."""
        self.N[a] += 1
        self.Q[a] += (reward - self.Q[a]) / self.N[a]


# %% [markdown]
"""
**Wiring check** — the incremental update really is the sample mean, and `N` counts the pulls:
"""

# %%
a = BanditAgent(k=3, seed=0)
for s in [1.0, 3.0, 2.0, 8.0]:
    a.update(0, s)
assert abs(a.Q[0] - np.mean([1.0, 3.0, 2.0, 8.0])) < 1e-9
assert a.N[0] == 4
print(f"incremental Q[0] = {a.Q[0]:.3f}  ==  sample mean 3.500  ✅")

# %% [markdown]
"""
## Watch it work

One run is noisy, so we do the standard thing (Sutton & Barto §2.3): average over **many independent
bandits**. Each run gets a fresh `GaussianBandit` (new `q*`) and a fresh agent; we record, at every
step, the reward received and whether the arm chosen was the truly-best one. Averaging those across
runs gives the two classic curves.
"""


# %%
def run(agent: BanditAgent, bandit: GaussianBandit, steps: int):
    """Run one agent on one bandit for `steps` pulls.
    Returns (rewards, optimal): per-step reward and whether the best arm was chosen."""
    rewards = np.zeros(steps)
    optimal = np.zeros(steps)
    for t in range(steps):
        a = agent.select()
        r = bandit.pull(a)
        agent.update(a, r)
        rewards[t] = r
        optimal[t] = 1.0 if a == bandit.best_arm else 0.0
    return rewards, optimal


def average_runs(make_agent, k=10, steps=1000, runs=300):
    """Average per-step reward and %-optimal across many independent bandits."""
    R = np.zeros(steps)
    O = np.zeros(steps)
    for i in range(runs):
        bandit = GaussianBandit(k, seed=i)
        agent = make_agent(seed=1000 + i)
        r, o = run(agent, bandit, steps)
        R += r
        O += o
    return R / runs, O / runs


strategies = {
    "greedy (ε=0)":      lambda seed: BanditAgent(10, epsilon=0.0, seed=seed),
    "ε-greedy (ε=0.1)":  lambda seed: BanditAgent(10, epsilon=0.1, seed=seed),
    "ε-greedy (ε=0.01)": lambda seed: BanditAgent(10, epsilon=0.01, seed=seed),
    "UCB (c=2)":         lambda seed: BanditAgent(10, c=2.0, seed=seed),
}

curves = {name: average_runs(make) for name, make in strategies.items()}

# %% [markdown]
"""
The payoff — the two curves every RL course opens with. **Left:** average reward per step climbs as
each agent figures out the good arms. **Right:** the fraction of pulls that hit the *actual* best arm.
"""

# %%
fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.2))
for name, (R, O) in curves.items():
    axL.plot(R, label=name)
    axR.plot(O * 100, label=name)
axL.set(xlabel="step", ylabel="average reward", title="average reward per step")
axR.set(xlabel="step", ylabel="% optimal action", title="% of pulls on the best arm")
axR.set_ylim(0, 100)
for ax in (axL, axR):
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle("10-armed bandit — exploration beats pure greedy (avg of 300 runs)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.95))
plt.show()

# final-100-step summary
print("strategy              reward(last100)   %optimal(last100)")
for name, (R, O) in curves.items():
    print(f"  {name:<18}  {R[-100:].mean():6.3f}          {O[-100:].mean():6.1%}")

# %% [markdown]
"""
Read the curves:

- **greedy flatlines low.** It commits early to whatever arm looked best after a few noisy pulls and
  never re-checks — so on most of the 300 bandits it's stuck on a sub-optimal arm.
- **ε-greedy keeps climbing.** The constant trickle of random pulls means it *keeps finding* the best
  arm. `ε=0.1` learns faster but caps lower (it wastes 10% of pulls exploring forever); `ε=0.01`
  starts slower but overtakes it later — the explore-rate trade-off in one picture.
- **UCB is strongest here.** Directing exploration at *uncertain* arms (not random ones) finds the
  best arm fastest. Its early spike is the initial sweep — every untried arm has an infinite bonus, so
  UCB pulls each once before it starts exploiting.
"""

# %% [markdown]
"""
## Optimism explores with **zero** randomness

Last trick, and a genuinely different mechanism. Set every `Q(a)` far above the true means (here
`+5`, while `q* ~ 0`). Now every *untried* arm looks better than any tried one, so **plain greedy**
(`ε = 0`) is forced to try each arm at least once — the disappointment of real rewards pulls the
inflated estimates down until the truly-good arm wins out. Exploration falls straight out of the
initial values; no coin-flips.
"""

# %%
opt_curve = average_runs(lambda seed: BanditAgent(10, epsilon=0.0, optimistic=5.0, seed=seed))
greedy_curve = curves["greedy (ε=0)"]

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(greedy_curve[1] * 100, label="greedy, Q₀=0 (no exploration)")
ax.plot(opt_curve[1] * 100, label="greedy, Q₀=+5 (optimistic)")
ax.set(xlabel="step", ylabel="% optimal action",
       title="optimistic init drives exploration with no randomness")
ax.set_ylim(0, 100); ax.legend(); ax.grid(alpha=0.3)
plt.show()

print(f"plain greedy   : %optimal(last100) = {greedy_curve[1][-100:].mean():.1%}")
print(f"optimistic Q₀=5: %optimal(last100) = {opt_curve[1][-100:].mean():.1%}")

# %% [markdown]
"""
The optimistic curve overshoots early (that forced sweep) then settles well above plain greedy —
same greedy rule, but the *initial values* did all the exploring. (This trick leans on stationarity:
the optimism is a one-time push at the start, so it does nothing if the arms drift later.)
"""

# %% [markdown]
"""
## The map — what we open next

We isolated **explore vs exploit** in the one setting where nothing else is going on: one state, no
dynamics. Everything from here adds structure back:

| next | adds | the new question |
|---|---|---|
| `02` | states + transitions, a **known** model | how do you value a *sequence* of decisions? (Bellman, DP) |
| `03` | drop the model, learn from **samples** | estimate values without knowing the dynamics (MC vs TD) |
| `04` | **control** from experience | learn the *best* policy, not just evaluate one (Q-learning, SARSA) |

The `(R − Q)` error-nudge you just wrote is the seed of all of it — it becomes the TD error in `03`
and the loss gradient once we swap the table for a network in `05`.

Next: **`02` — MDPs & dynamic programming: valuing a sequence of decisions.**
"""
