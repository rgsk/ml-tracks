# RL — roadmap (from-scratch, classic → RLHF, notebook edition)

A parallel track to `nb/cnn/` and `llm/`, rebuilt in the **notebook style**: every experiment is
**one jupytext-paired notebook** you read top-to-bottom — prose, code, and figures inline — and each
one **ends in a run-and-see payoff** (a learning curve, a solved gridworld, a policy that improves on
screen), not just a passing assert.

**How this is taught: see it work, then open the box.** Each notebook first gets the method *running*
on a tiny env so you watch it succeed, then goes back and explains *why* each piece is shaped that way
— the update rule, the exploration knob, the stability hack. It's the same top-down move as `nb/cnn`,
applied per-method: the RL track is a *progression* of algorithms, so the "whole game" is rebuilt one
rung at a time, each rung the smallest honest delta on the one below.

The sweep runs from tabular foundations (no neural nets) to the RLHF methods that plug straight back
into the LLM track (PPO/DPO/GRPO). Ported faithfully from the original `rl/` track (the explanatory
asides come with it — the "why" parenthetical is usually the whole point).

---

## Format (jupytext paired notebooks)

Each experiment is **one notebook** — a lesson read top-to-bottom. Every notebook is a **jupytext
pair**:

- `walkthroughs/NN_name.py` — the **source of truth** (`py:percent`); clean git diffs, editable in
  the editor. This is what you edit.
- `walkthroughs/NN_name.ipynb` — the **rendered pair**: same cells plus executed outputs (prints,
  figures). This is what you read. It's **gitignored** and regenerated.

Rebuild/execute a notebook after editing its `.py`:

```bash
uv run jupytext --to ipynb --execute nb/rl/walkthroughs/01_bandits.py   # re-run all cells
```

```
nb/rl/
  roadmap.md                 <- this file: the table of contents + how it's taught
  walkthroughs/
    01_bandits.py    ⇄ .ipynb   one state, k arms: exploration vs exploitation (ε-greedy, UCB, optimism)
    02_mdp_dp.py     ⇄ .ipynb   add states+transitions: Bellman, policy/value iteration (known model)
    03_mc_td.py      ⇄ .ipynb   drop the model: predict V from samples — MC vs TD(0) bias/variance
    04_q_sarsa.py    ⇄ .ipynb   first CONTROL from experience: Q-learning (off) vs SARSA (on-policy)
    ...                          (05+ enter neural nets — see the phases below)
  custom/                    <- from-scratch impls where a primitive is worth rebuilding; each self-tests
  checkpoints/               <- weights trained by a notebook, loaded by later ones (no retrain)
```

Notebooks stay independent: a deep-RL notebook that trains a net **saves a checkpoint**; later
notebooks **load it if present, else train a quick one**, so editing one never re-runs another.

---

## Phase 0 — Tabular foundations (no neural nets yet)

**`01` — multi-armed bandits.** One state, k actions, learn which pays best. No sequential dynamics,
so it isolates the one tension that never goes away: **exploration vs exploitation**. Action-value
estimates, the incremental-average update, ε-greedy, UCB, optimistic init. Payoff: the classic
average-reward / %-optimal learning curves. *(env: hand-coded Gaussian bandit)*

**`02` — MDPs & dynamic programming.** Add states + transitions. The Bellman equations, then policy
evaluation, policy iteration, value iteration with a **known** model — the ground truth every later
method approximates. Cross-checked against the analytic `(I − γPπ)⁻¹rπ` solve. *(env: hand-coded
gridworld)*

**`03` — Monte Carlo & TD(0) prediction.** Drop the known model: estimate values from **sampled**
episodes. MC (wait till episode end) vs TD (bootstrap from the next step) — the bias/variance
trade-off at the heart of RL. *(env: gridworld)*

**`04` — tabular Q-learning & SARSA.** First real **control** from experience. Off-policy
(Q-learning) vs on-policy (SARSA), the max-bootstrap, ε-greedy behaviour; the Cliff-Walking split
(SARSA takes the safe path, Q-learning the optimal edge). *(env: gridworld / Cliff Walking)*

## Phase 1 — Deep value-based (enter neural nets)

**`05` — function approximation.** Replace the Q-table with a network `Q(s,a; θ)`. Why naive online
updates diverge (the **deadly triad**) — motivates the DQN tricks. *(env: CartPole)*

**`06` — DQN.** The two stabilizers that made deep RL work: an **experience replay** buffer and a
**target network**. Solve CartPole end-to-end. *(env: CartPole)*

**`07` — DQN improvements.** Double DQN (kills max-overestimation), Dueling heads, Prioritized
Replay — each a small, interview-favourite delta on `06`. *(env: CartPole / LunarLander)*

## Phase 2 — Policy gradient (learn the policy directly)

**`08` — REINFORCE.** The policy-gradient theorem from scratch: push up log-prob of actions weighted
by return. Why a **baseline** cuts variance without adding bias. *(env: CartPole)*

**`09` — actor-critic / A2C.** Learn a value critic to bootstrap the advantage instead of full
returns. Generalized Advantage Estimation (GAE) as the bias/variance knob. *(env: CartPole)*

**`10` — PPO.** The workhorse: clipped surrogate objective for big-but-safe policy steps, multiple
epochs per batch. The exact algorithm RLHF uses — the bridge to Phase 3. *(env: CartPole → continuous)*

## Phase 3 — RLHF (closes the loop with `llm/`)

**`11` — reward modeling.** Turn **pairwise preferences** into a scalar reward model via
Bradley-Terry loss on (preferred, rejected) pairs. *(env: toy preferences → a real LM)*

**`12` — PPO-RLHF.** Put `10` and `11` together: a value head on the LM, reward from `11`, a KL
penalty to the reference model so it doesn't drift / reward-hack. The InstructGPT recipe. *(env: small
LM from `llm/`)*

**`13` — DPO.** Direct Preference Optimization: skip the reward model *and* the RL loop — a clever
loss makes the LM its own implicit reward model. *(env: small LM + preference pairs)*

**`14` — GRPO.** Group Relative Policy Optimization (DeepSeek): drop the value critic, estimate the
advantage from a **group** of sampled answers' rewards. Cheap, and dominates recent reasoning-model
training. *(env: small LM, verifiable rewards)*

---

### Threads that recur (called out as they appear, not separate items)
- **Exploration**: ε-greedy → UCB → entropy bonus → KL-to-reference.
- **Bias/variance of the target**: MC ↔ TD ↔ n-step ↔ GAE(λ).
- **On- vs off-policy**: SARSA/Q-learning, why PPO is "on-policy-ish" with clipping.
- **Stability hacks**: target nets, replay, clipping, KL penalties — each fixes a specific divergence.
