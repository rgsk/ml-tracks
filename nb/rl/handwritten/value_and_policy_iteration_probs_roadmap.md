# First

Honest framing first: of what's in the notebook now, `uniform_policy` is the _only_ thing the distribution representation actually bought you. The incumbent tie-break, the `max_iters`/`theta` work, the γ=1 oscillation finding — all of those work identically with `policy[r][c] = action`. So the upgrade is currently one cell wide. Here's what actually widens it.

## 1. ε-greedy evaluation next to a pit — the highest-payoff one

```python
grid = Grid(rows=3, cols=4, step_reward=-0.04,
            terminals={(0, 3): 1, (1, 3): -1}, walls={(1, 1)})
```

Take the optimal π\*, smear it into ε-greedy (`1-ε+ε/4` on the best action, `ε/4` on each other), and evaluate. Sweep ε over `{0, 0.05, 0.2, 0.5}`.

Watch `V(1,2)` and `(2,3)` specifically — the cells adjacent to the −1 pit. Their value should fall much faster with ε than cells far from the pit, because exploration there can kill you. That gap _is_ the Cliff Walking result, computed exactly by DP before you write a single line of sampling code. When SARSA later prefers the safe route and Q-learning doesn't, you'll already have the number that explains why.

Impossible before: a deterministic policy can't express "mostly right, sometimes not."

## 2. ε-soft policy iteration — a genuinely different fixed point

Replace the greedy improvement step with an **ε-greedy** improvement step and iterate to convergence. It converges, but not to π* — it converges to the best policy *among ε-soft policies*, and its value is strictly below `V*`.

That's the on-policy/off-policy split in miniature, and it's the single most useful thing you can pre-load before SARSA vs Q-learning: SARSA converges to this, Q-learning to `V*`. Deterministic policies literally cannot represent the answer, so the experiment didn't exist before.

## 3. The value bracket

Compute `V^uniform` and `V^π*` on the same grid and report mean-over-non-terminals for each. That's the floor and ceiling that q_learning_sarsa.py:317 uses to grade learning curves (`analytic_policy_value` with `np.ones((nS,nA))/nA`). You now have both. Cheap, and it makes later plots interpretable instead of arbitrary.

## 4. Interpolated policies

`π_λ = λ·π* + (1-λ)·uniform`, sweep λ from 0 to 1, plot mean V. Gives a smooth curve from random-walk value to optimal value and shows that value is _not_ linear in policy mixing. Also a clean way to see that policy improvement is monotone but not proportional.

## 5. Stochastic tie-breaking — ties directly into the γ=1 bug you just found

When several actions tie in `read_policy`, split probability equally among them instead of taking dict order. Then rerun `policy_iteration(default_grid, gamma=1.0)`.

You already know the default oscillates forever and `pass_incumbent_policy=True` fixes it in 4 rounds. This gives you a third answer to compare, and unlike the incumbent hack it makes the tie _visible_ — you can render the number of tied actions per cell and see exactly how degenerate γ=1 makes the problem. A deterministic policy has to pick one and hide the tie.

---

If you only do two, do **1** and **2** — those are the ones that pay off directly in `mc_td.py` and `q_learning_sarsa.py`. 3 is a five-minute add-on to either.

# Second

Three that are genuinely open, ranked by payoff.

1. Where does the ε-soft optimum first diverge from π*?
   You have two data points: at ε=0.5 ε-soft PI returns exactly π*, at ε=0.9 it flips (2,1) to ←. The threshold between them is the headline number of this whole block — how clumsy do you have to be before the best plan changes?

Sweep ε in fine steps (0.5 → 0.9 by 0.02), record the core each time, and report the first ε where it differs from π\*. Same binary-search style as the −1.9 step-reward threshold in your other notebook.

Two things to log alongside:

V^(eps-soft PI) vs V^(eps_soft(pi_star, eps)) cellwise. Below the threshold they should be identical (the smeared optimum is already optimal); above it the PI version should be strictly better somewhere. That inequality is the formal statement, and you'd have it verified across a whole range instead of at one point.
Which cell flips first, and whether it's always (2,1).
This is directly the SARSA/Q-learning boundary on this grid: below the threshold both algorithms target the same greedy policy; above it they don't.

2. Isolate pit risk from the wandering tax
   The note claims V^eps = V\* − wandering tax − pit risk. That's a decomposition you asserted from a pattern; you can test it.

Rerun the ε sweep with step_reward = -0.001 instead of -0.04. Wandering is now nearly free, so the tax term should collapse toward zero while pit risk is untouched.

Prediction: the far column (0,0)/(1,0)/(2,0) barely moves with ε, while (1,2) and (2,3) still fall off a cliff. If the far column still drops substantially, the decomposition is wrong and something else is going on — which would be the more interesting outcome.

It should also drag the (2,3)-vs-(2,0) crossover to a much lower ε, since the head start is no longer being eaten by step costs.

3. γ sweep at fixed ε
   Completely unexplored here, and the interaction is non-obvious. The +1 goal is several steps away and gets discounted by γ^d; the −1 pit is one slip away and is barely discounted at all. So lowering γ should make the pit look relatively worse — a myopic agent is more pit-averse, not less.

Fix ε=0.2, sweep γ over {0.99, 0.9, 0.7, 0.5}, and watch whether the greedy core starts routing away from the pit at low γ even at modest exploration. If it does, you have two independent knobs (clumsiness and myopia) producing the same behavioural change, which is a nice thing to be able to explain.

Also worth adding while you're in here, cheap: assert the policy improvement theorem inside policy_iteration — after each improvement, check V^(new) >= V^(old) cellwise. It's three lines, it's the one invariant the algorithm actually guarantees, and it would have caught several of the bugs from the last few days as they happened rather than by inspection.

Of the three, #1 is the one I'd do — it converts "the effect exists at ε=0.9" into a number, and it's the result you'd actually cite when SARSA and Q-learning disagree later.
