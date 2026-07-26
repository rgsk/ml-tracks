# RL — roadmap (top-down)

**Whole game first, one rung at a time.** RL isn't one model with boxes in it — it's a **ladder of
algorithms**, each one the smallest honest fix to what broke on the rung below. So the top-down move
happens *per subtopic*: each folder's `exp_1` builds a **complete, working agent** and lets you
**watch it learn** (a return curve climbing, a gridworld solved, a pole staying up), with only a
rough one-line narration per part. Every experiment after that **opens one box of that exact agent**
and explains the why — measured, not asserted.

Four rungs, each a subtopic folder: **`q_learning/`** (tabular, no nets) → **`dqn/`** (deep
value-based) → **`ppo/`** (policy gradient — the modern workhorse) → **`rlhf/`** (which plugs
straight back into the `llm/` track). Each rung starts from a *failure* of the previous one, so you
never learn a trick before you've felt the problem it solves.

---

## Folder layout

```-
new/rl/
  roadmap.md
  custom/                      from-scratch impls (gae, clipped surrogate, bradley-terry, dpo loss,
                               replay buffer…); each runs standalone as its own self-test vs a reference
  walkthroughs/
    q_learning/                rung 1 — tabular control (whole game + its boxes)
      q_learning.py            experiments exp_1 .. exp_N
      notes/*.md               one write-up per experiment
      figures/{experiments,generated,handmade}/
      children/exploration/    dig-in: bandits (one state, k arms) — the ε knob on its own
    dqn/                       rung 2 — Q with a neural net (CartPole)
    ppo/                       rung 3 — learn the policy directly (CartPole → continuous)
    rlhf/                      rung 4 — align the GPT from llm/ (reward model, PPO-RLHF, DPO, GRPO)
  envs.py                      the hand-coded tabular envs (gridworld, cliff), reset/step like gymnasium
  <root>                       the cleaned-up reusable agent (policy/value nets + PPO update),
                               assembled after ppo/ and imported by rlhf/
```

Envs: **hand-coded** for the tabular rung (a gridworld you can print and check by hand),
**gymnasium** (`classic-control`, already installed) from `dqn/` on. Deep-RL runs stay small enough
for the local 4060; a subtopic that trains a net **saves a checkpoint** so later experiments load
instead of retrain.

---

## Rung 1 — `q_learning/` (tabular control: the vocabulary, no neural nets)

**`exp_1` — the whole game.** A `Q[state, action]` table, an ε-greedy actor, and one update line:
`Q[s,a] += α(r + γ·max Q[s',·] − Q[s,a])`. Drop it in a slippery gridworld and **watch a flailing
random agent turn into one that walks to the goal** — steps-to-goal curve falling, and the greedy
policy printed as arrows. Rough narration only: *score every (state, action), act mostly-greedily,
nudge each score toward what actually happened next.*
> After this you have the vocabulary. Everything below opens one box of this agent.

**`exp_2` — the environment (what an MDP actually is).** `(S, A, P, R, γ)` and the `reset`/`step`
interface; episode, trajectory, the return `G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + …`. Why discount at
all, made visible: **sweep γ and watch the optimal route change** (low γ = impatient, takes the risky
shortcut; high γ = takes the safe long way).

**`exp_3` — the value functions + the ground truth.** `V^π`, `Q^π`, and the Bellman equations as
"value now = reward now + γ·value next". With the model **known**, solve exactly: policy evaluation,
policy iteration, value iteration — cross-checked against the analytic `(I − γPπ)⁻¹rπ` solve. That
exact table is the answer `exp_1` was crawling toward from samples; **measure the gap** between the
two.

**`exp_4` — the target: MC vs TD.** Where `r + γ·max Q` came from. Estimate `V^π` from samples two
ways — full-episode returns (MC: unbiased, high variance) vs one-step bootstrap (TD(0): biased,
low variance) — and **graded against `exp_3`'s exact solve**. The bias/variance knob that reappears
as GAE(λ) in `ppo/`.

**`exp_5` — the exploration knob** → dig-in `children/exploration/`. Greedy gets stuck on the first
thing that worked; ε fixes it, badly. Stripped to **one state and k arms** (a bandit — no dynamics,
so exploration is the *only* tension): ε-greedy, optimistic init, UCB, decay schedules, on
%-optimal-action and regret curves.

**`exp_6` — off-policy vs on-policy.** One character changes: `max Q[s',·]` (learn about the greedy
policy) → `Q[s',a']` for the action you actually took (SARSA). Run both on **Cliff Walking** and
watch the split — SARSA takes the safe path away from the cliff, Q-learning hugs the optimal edge and
falls off while exploring. The distinction that decides what every later method is allowed to reuse.

## Rung 2 — `dqn/` (the table doesn't scale)

**`exp_1` — the whole game.** DQN on CartPole: a small MLP `Q(s; θ)` with one output per action,
replay buffer, target net, ε-decay. **Watch episode return climb 20 → 500** and render a rollout of
the pole standing still.

**`exp_2` — why a network.** CartPole's state is 4 continuous numbers: the table needs discretization,
and the bin count explodes (measured) while *still* generalizing to nothing. `Q(s,a;θ)` shares
structure across nearby states — the whole point.

**`exp_3` — why the naive version diverges.** Take `exp_1` and delete both stabilizers: online
gradient TD updates on correlated samples → **watch Q values blow up**. The **deadly triad**
(bootstrapping + off-policy + function approximation) named at the exact moment you see it happen.

**`exp_4` — the replay buffer.** Consecutive transitions are near-duplicates and strongly correlated;
a buffer restores something closer to i.i.d. batches and reuses each transition many times. Ablation:
buffer off / buffer tiny → measured collapse.

**`exp_5` — the target network.** The regression target `r + γ·max Q(s';θ)` moves every time θ moves —
you're chasing your own tail. Freeze a copy, sync every N steps. Ablation + the sync-period sweep.

**`exp_6` — the deltas that stuck.** One measured improvement each: **Double DQN** (decouple
select-from-evaluate; *measure* the max-operator's overestimation bias directly), **Dueling** heads
(V + advantage), **prioritized replay** (sample by TD error). Small, interview-favourite diffs on
`exp_1`.

## Rung 3 — `ppo/` (learn the policy, not the values)

**`exp_1` — the whole game.** PPO on CartPole: policy net + value net, collect a rollout batch,
compute GAE advantages, take several epochs of clipped-surrogate steps. **Watch return climb to
solved** — and note that this is *the* algorithm the RLHF rung uses, unchanged.

**`exp_2` — why a policy at all.** Value-based control needs an `argmax` over actions — fine for 2
discrete actions, hopeless for a continuous torque vector. Plus stochastic policies are what you want
in partially-observed / adversarial settings. Show a case where the ε-greedy-over-Q formulation can't
express the right behaviour.

**`exp_3` — the policy gradient theorem** (derived inline). `∇J(θ) = E[∇log π(a|s)·G]` — the
log-derivative trick, and why the environment's dynamics *don't* appear in it. That's REINFORCE, ~10
lines: run it, **it learns, and it's visibly noisy** (high-variance curve you can point at).

**`exp_4` — the baseline.** Subtract anything that doesn't depend on the action and the gradient is
unchanged (`E[∇log π·b] = 0`, shown) while the **variance drops** (measured, side-by-side curves).
`G − V(s)` = the advantage — the answer to "was that action better than average *here*", not "did
this episode go well".

**`exp_5` — the critic + GAE.** Bootstrap the advantage with a learned `V` instead of the full return:
n-step targets, then **GAE(λ)** as the single dial from TD (λ=0, biased/low-variance) to MC (λ=1) —
`exp_4` of rung 1, returning in its modern form. Sweep λ, read the scoreboard.

**`exp_6` — why clipping.** Reusing a batch for several epochs makes the data off-policy → the
importance ratio `π/π_old` blows up and one huge step destroys the policy. **Ablate the clip and watch
the collapse**, then read the clipped surrogate as a cheap trust region (and where the old KL-penalty
form went).

**`exp_7` — the plumbing that actually decides whether PPO works.** Advantage normalization, entropy
bonus, value-loss coefficient, vectorized envs, observation normalization, LR schedule — each ablated
and ranked, because in PPO the "implementation details" are the algorithm.

## Rung 4 — `rlhf/` (closes the loop with `llm/`)

**`exp_1` — the whole game.** Take the small GPT from the `llm/` track and **align it with DPO** on
preference pairs: one loss, one loop, no reward model, no rollouts. **Read the before/after samples
and watch the model's style actually shift.** The cheapest complete alignment pipeline that exists —
so it's the whole game.

**`exp_2` — where preferences come from: the reward model.** Humans can't score text, but they can
pick a winner. Bradley-Terry on (preferred, rejected) pairs → a scalar reward head. Train one, then
**test it**: does it rank held-out pairs, and where does it get gamed?

**`exp_3` — the KL leash.** Why unconstrained reward maximization **reward-hacks** into degenerate
text (shown), and how a KL penalty to the frozen reference model keeps it near the original
distribution. The β knob = how far the policy is allowed to move.

**`exp_4` — PPO-RLHF.** Wire `ppo/` to a language model: tokens as actions, a value head on the LM,
reward from `exp_2` at the end of a sequence, KL from `exp_3` per token. The InstructGPT recipe —
and why it's an operational nightmare (four models in memory).

**`exp_5` — DPO, derived.** Go back to `exp_1` and *earn* it: the KL-constrained optimum has a
closed form, invert it and the reward model **disappears into the policy** — a plain classification
loss on pairs. Why it's the default for preference data, and where it's weaker than online RL.

**`exp_6` — GRPO.** Drop the critic entirely: sample a **group** of answers per prompt, use the
group's mean reward as the baseline (advantage = z-score within the group). With *verifiable* rewards
(does the answer parse / is the arithmetic right) this is what today's reasoning models are trained
with. The current end of the ladder.

---

## Optional advanced section (after rung 4 — pick what you need)

Each extends one core subtopic; the ladder above is already interview-complete without them.

- **`sac/`** — off-policy continuous control with entropy regularization (extends `dqn/` + `ppo/`).
  The actual workhorse for **robot learning**: sample-efficient, no on-policy batch to throw away.
- **`offline_rl/`** — learn from a fixed log with no environment access (CQL / IQL): the
  distribution-shift problem and the conservatism fix. The regime real robot data lives in.
- **`model_based/`** — learn the dynamics, then plan or train inside the model (Dyna → world models).
  Wildly more sample-efficient, and the bridge to the parked world-model track.
- **`bandits_pro/`** — Thompson sampling / contextual bandits, if the exploration dig-in leaves you
  wanting the Bayesian version.

**Stop points:** after rung 3 you can implement any standard deep-RL algorithm from scratch; after
rung 4 you can explain and build every alignment method in current use; the advanced section is where
the robotics side of the ladder starts.

---

*Run: `python walkthroughs/<subtopic>/<subtopic>.py`. Notes in `walkthroughs/<subtopic>/notes/`,
figures in `walkthroughs/<subtopic>/figures/`. Content is re-sequenced from the bottom-up `rl/` track
(`01_bandits.py` … `05_function_approx.py` are done there) — same math, top-down order.*
