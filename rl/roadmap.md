# RL Roadmap — from-scratch, classic → RLHF

A parallel track to `llm/`, same exercise-driven style: I (Claude) write a skeleton
with TODOs + a `__main__` self-check; you fill the TODOs and run it; we review and
strip scaffolding. Full sweep from tabular foundations to the RLHF methods that
plug back into the LLM track (PPO/DPO/GRPO).

Environments are chosen per-topic for clarity: tiny hand-coded MDPs while the math
is the point, `gymnasium` (CartPole etc.) once we need a real env, scaling to
harder envs later.

> Personal reference — tick items off as you go. Numbers are build order.

---

## Phase 0 — Tabular foundations (no neural nets yet)

**1. Multi-armed bandits** — One state, k actions, learn which pays best. No
sequential dynamics, so it isolates the core RL tension: EXPLORATION vs
EXPLOITATION. Action-value estimates, incremental updates, ε-greedy, UCB,
optimistic init. *(env: hand-coded Gaussian bandit)*

**2. MDPs & dynamic programming** — Add states + transitions. The Bellman
equations, then policy evaluation, policy iteration, and value iteration with a
KNOWN model. This is the ground truth every later method approximates. *(env:
hand-coded gridworld)*

**3. Monte Carlo & TD(0) prediction** — Drop the known model: estimate values from
SAMPLED episodes. MC (wait till episode end) vs TD (bootstrap from the next step)
— the bias/variance trade-off at the heart of RL. *(env: gridworld)*

**4. Tabular Q-learning & SARSA** — First real CONTROL from experience.
Off-policy (Q-learning) vs on-policy (SARSA), the max-bootstrap, ε-greedy
behaviour. Watch it solve a gridworld with a Q-table. *(env: gridworld / FrozenLake)*

## Phase 1 — Deep value-based (enter neural nets)

**5. Function approximation** — Replace the Q-table with a network: Q(s,a; θ).
Why naive online updates diverge (the "deadly triad") — motivates the DQN tricks.
*(env: CartPole)*

**6. DQN** — The two stabilizers that made deep RL work: an EXPERIENCE REPLAY
buffer (break correlation) and a TARGET NETWORK (stable bootstrap target). Solve
CartPole end-to-end. *(env: CartPole)*

**7. DQN improvements** — Double DQN (kills max-overestimation), Dueling heads
(separate value/advantage), Prioritized Replay (sample surprising transitions
more). Each a small, interview-favourite delta on #6. *(env: CartPole / LunarLander)*

## Phase 2 — Policy gradient (learn the policy directly)

**8. REINFORCE** — The policy-gradient theorem from scratch: push up log-prob of
actions weighted by return. Why a BASELINE cuts variance without adding bias.
*(env: CartPole)*

**9. Actor-Critic / A2C** — Learn a value critic to bootstrap the advantage
instead of using full returns. Generalized Advantage Estimation (GAE) as the
bias/variance knob. *(env: CartPole / LunarLander)*

**10. PPO** — The workhorse. Clipped surrogate objective to take big-but-safe
policy steps; multiple epochs per batch. This is the exact algorithm RLHF uses —
the bridge to Phase 3. *(env: CartPole, then continuous control)*

## Phase 3 — RLHF (closes the loop with `llm/`)

**11. Reward modeling** — Turn PAIRWISE PREFERENCES into a scalar reward model.
Bradley-Terry loss on (preferred, rejected) pairs. The "human feedback" half of
RLHF. *(env: toy preference data, then a real LM)*

**12. PPO-RLHF** — Put #10 and #11 together: a value head on the LM, reward from
#11, and a KL penalty to the reference model so it doesn't drift / reward-hack.
The classic InstructGPT recipe. *(env: small LM from `llm/`)*

**13. DPO** — Direct Preference Optimization: skip the reward model AND the RL loop
— a clever loss makes the LM its own implicit reward model. Why it's eaten a lot
of RLHF's lunch. *(env: small LM + preference pairs)*

**14. GRPO** — Group Relative Policy Optimization (DeepSeek). Drop the value
critic; estimate the advantage from a GROUP of sampled answers' rewards. Why it's
cheap and dominates recent reasoning-model training. *(env: small LM, verifiable
rewards)*

---

### Threads that recur (call out as they appear, not separate items)
- **Exploration**: ε-greedy → UCB → entropy bonus → KL-to-reference.
- **Bias/variance of the target**: MC ↔ TD ↔ n-step ↔ GAE(λ).
- **On- vs off-policy**: SARSA/Q-learning, why PPO is "on-policy-ish" with clipping.
- **Stability hacks**: target nets, replay, clipping, KL penalties — each fixes a
  specific divergence.
