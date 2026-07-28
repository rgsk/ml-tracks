### Claude chat - Implement value and policy iteration from scratch

- `gamma=1.0` with `step_reward=0, goal_reward=1` → V=1 everywhere, policy degenerates (no urgency).
- `gamma=1.0` with `step_reward=-1` → V = −(steps to goal). Still fine.
- The **sealed pocket** grid (cells walled off from the goal): at γ=0.9 they settle to −10; at γ=1.0 they **diverge**, and your `while delta > 0.001` loop will spin forever — you need `max_iters`.
- `v_init=-50` instead of 0 → still converges to the same V\*. Good demonstration that the fixed point doesn't care about the start.

- **`v_init`** — start V at `50` instead of 0. It still converges to the same V\*, which is the concrete demonstration that the fixed point doesn't depend on initialization. Cheap to add: one parameter in `value_iteration`.
- **The sealed pocket** — a grid where some cells are walled off from the goal entirely. At γ=0.9 they converge to `10` (the trapped value); at γ=1 they diverge. Note `value_iteration` still has **no `max_iters`**, so that second one will hang the kernel — it's the one function you never capped.

Ranked by insight-per-line-of-code:

**1. The living-reward phase transition** — the single most illuminating gridworld experiment, and your grid is one line from it. Add a `-1` terminal at `(1,3)` next to the `+1`, then sweep `step_reward` from `-0.01` down to `-2.0`. There's a threshold where the optimal policy flips from "take the long way around the trap" to "dive into the `-1` to stop the bleeding". Same MDP, same γ, and the arrows reverse. It makes reward design feel dangerous in a way no amount of explanation does.

**2. Truncated policy evaluation → VI and PI are the same algorithm.** Cap `policy_evaluation` at exactly `k` sweeps instead of running to convergence:

```
k = 1        -> this IS value iteration
k = 2,3,5    -> "modified policy iteration", the middle of the spectrum
k = inf      -> this IS policy iteration
```

Plot total backups (not outer rounds) against `k`. You'll usually find the minimum is at small-but-not-1 `k`. This is Sutton & Barto §4.6 and it's the thing that makes the whole chapter click.

**3. In-place (asynchronous) backups.** Delete `V_old` and write straight into `V`. It still converges — the theory doesn't require synchrony — and it's typically ~2× fewer sweeps because information propagates within a sweep. Then flip the sweep order (bottom-right → top-left) and watch the sweep count change dramatically, since you're now propagating _with_ the value flow instead of against it. Good intuition for why prioritized sweeping exists.

**4. `v_init` (still not added, cheapest of all).** `v_init = -50` and `v_init = +100` both converge to the same V\*, one climbing and one falling. Concrete proof the Bellman operator is a contraction with a unique fixed point, and it takes one parameter.

If you want to point at the next rung instead: **make transitions stochastic** (0.8 intended / 0.1 each perpendicular). It's a real change to `step` — returning a list of `(prob, next_cell, reward)` instead of one outcome — but it's what turns the trap experiment genuinely interesting (the optimal policy deliberately steers away from the `-1` even when that's slower), and it's the model shape every model-free method in `03_mc_td` is trying to approximate from samples.

Deterministic transitions make the flip a hard cliff. If you later add slip, the same sweep produces a much richer picture: an intermediate band where the policy takes a _longer_ path specifically to keep distance from the trap, because a slip near it is expensive.
