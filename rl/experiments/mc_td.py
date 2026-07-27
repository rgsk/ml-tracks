"""
WALKTHROUGH: Monte Carlo & TD(0) prediction (exercise 3), built one layer at a time.

The shift from exercise 2: we LOSE the model. There, the DP algorithms read
env.P[s][a] — the exact transition probabilities and rewards. A real agent never
gets that. It can only ACT and OBSERVE: from state s take action a, and the world
returns ONE sample (s', r) drawn from the hidden p(.|s,a).

So everything here is built on a stream of samples instead of a known model. We
build it up in layers (run each, watch the output):

  1. the experience stream — a gym-like Sampler (reset/step), and the punchline
     that makes model-free learning possible: enough samples ARE the model.
  2. generate_episode — roll a whole trajectory under a policy.
  3. returns G_t — turn a trajectory into the numbers we average (Monte Carlo).
  4. mc_prediction — first-visit MC; watch V converge to the truth.
  5. td0_prediction — the bootstrap; the TD error; online learning.
  6. MC vs TD — the bias/variance payoff, side by side.

Ground truth for grading comes from exercise 2's exact linear solve.
"""

import argparse
import contextlib
import importlib.util
import pathlib
import sys

import numpy as np

# Reuse exercise 2's gridworld (the MDP model) + its exact ground-truth solver.
# (Module name starts with a digit, so load it by path rather than `import`.)
_spec = importlib.util.spec_from_file_location(
    "mdp_dp", pathlib.Path(__file__).parent.parent / "02_mdp_dp.py")
mdp_dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mdp_dp)
GridWorld = mdp_dp.GridWorld
_ARROWS = mdp_dp._ARROWS


# ---------------------------------------------------------------------------
# LAYER 1: the experience stream.
#
# The Sampler wraps the MDP and exposes ONLY reset()/step(a). Internally it draws
# from env.P — but the learning algorithms never get to peek at P, they just see
# the (s', r) that comes back. That keyhole is the entire premise of model-free RL.
# ---------------------------------------------------------------------------

class Sampler:
    """Gym-like view of the MDP: reset() to a start state, step(a) for one sample."""

    def __init__(self, env, rng):
        self.env = env
        self.rng = rng
        self._nonterminal = [s for s, cell in enumerate(env.states)
                             if cell not in env.terminals]
        self.s = None

    def reset(self):
        """Exploring start: begin at a uniformly random NON-terminal state, so every
        state eventually gets sampled (a fixed start + fixed policy would only ever
        trace one path)."""
        self.s = int(self.rng.choice(self._nonterminal))
        return self.s

    def step(self, a):
        """One sampled transition from the current state. Returns (s', r, done).

        The ENV rolls its OWN dice — which action actually fires after the slip —
        then applies deterministic movement. It never builds a probability table;
        the agent just sees the single (s', r) that falls out. (Reuses the same
        _move + noise that exercise 2's model was built from, so the sampled
        dynamics match env.P exactly — by construction, not by luck.)
        """
        env = self.env
        cell = env.states[self.s]
        # slip: intended action w.p. 1-noise, each perpendicular w.p. noise/2.
        actual = int(self.rng.choice([a, (a + 1) % 4, (a - 1) % 4],
                                     p=[1 - env.noise, env.noise / 2, env.noise / 2]))
        nxt = env._move(cell, actual)            # wall/edge bounce lives in _move
        reward = env.terminals.get(nxt, env.step_reward)
        done = nxt in env.terminals
        self.s = env.s2i[nxt]
        return self.s, reward, done


def _banner(*lines):
    print("=" * 64)
    for line in lines:
        print(line)
    print("=" * 64)


def exp_samples_are_the_model(cell=(2, 0), action=mdp_dp.UP, n=20000, seed=0):
    """Take action `a` from one state `n` times; tally where we land + mean reward,
    and compare to the model env.P[s][a]. They must agree — counting recovers the
    probabilities DP simply read off. THIS is why model-free prediction can work.
    """
    env = GridWorld()
    rng = np.random.default_rng(seed)
    sampler = Sampler(env, rng)
    s = env.s2i[cell]

    _banner(f"LAYER 1: samples ARE the model    "
            f"(from {cell}, action {_ARROWS[action]}, n={n})")

    # --- the model (what exercise 2 would just read) ---
    print("MODEL  env.P[s][a]  (prob | next cell | reward):")
    model_p = {}
    model_rbar = 0.0
    for (p, ns, r, _done) in env.P[s][action]:
        model_p[ns] = p
        model_rbar += p * r
        print(f"   {p:5.2f}  ->  {env.states[ns]!s:7}  r={r:+.2f}")
    print(f"   expected immediate reward E[r] = {model_rbar:+.4f}\n")

    # --- the samples (all the agent is actually allowed to see) ---
    counts = np.zeros(env.nS)
    rsum = 0.0
    for _ in range(n):
        sampler.s = s                       # force the same start each draw
        ns, r, _done = sampler.step(action)
        counts[ns] += 1
        rsum += r
    print("SAMPLES  empirical frequency over n draws:")
    for ns in sorted(model_p, key=lambda x: -model_p[x]):
        freq = counts[ns] / n
        print(f"   {freq:5.2f}  ->  {env.states[ns]!s:7}   "
              f"(model {model_p[ns]:.2f}, off by {abs(freq - model_p[ns]):.3f})")
    print(f"   sample mean reward = {rsum / n:+.4f}  "
          f"(model {model_rbar:+.4f})\n")


# ---------------------------------------------------------------------------
# LAYER 2: generate_episode — roll a FULL trajectory under a policy.
#
# reset(), then repeatedly: pick a ~ pi(.|s), step it, record the transition, until
# done. This is the unit MC and TD both consume. Two rollouts of the same policy can
# differ (action sampling + slip are both random).
# ---------------------------------------------------------------------------

def generate_episode(sampler, policy, rng, max_steps=1000):
    """Roll ONE episode following `policy` (stochastic matrix (nS, nA)).
    Returns the experience as a list of transitions [(s, r, s_next, done), ...]."""
    episode = []
    s = sampler.reset()
    for _ in range(max_steps):
        a = int(rng.choice(sampler.env.nA, p=policy[s]))
        ns, r, done = sampler.step(a)
        episode.append((s, r, ns, done))
        s = ns
        if done:
            break
    return episode


def _greedy_arrows(env, policy):
    """argmax action arrow per state, for showing what policy we're rolling under."""
    return {s: _ARROWS[int(np.argmax(policy[s]))] for s in range(env.nS)}


def exp_one_episode(num_episodes=4, seed=1):
    """Roll a few episodes under the OPTIMAL policy and print each as a path of
    cells with the reward collected on each move. Watch lengths/paths vary."""
    env = GridWorld()
    rng = np.random.default_rng(seed)
    sampler = Sampler(env, rng)
    opt_policy, _ = mdp_dp.value_iteration(env, env.gamma)
    arrows = _greedy_arrows(env, opt_policy)

    _banner(f"LAYER 2: generate_episode under the OPTIMAL policy "
            f"({num_episodes} rollouts)")
    for k in range(num_episodes):
        ep = generate_episode(sampler, opt_policy, rng)
        start = env.states[ep[0][0]]
        total = sum(r for (_s, r, _ns, _d) in ep)
        # render the path: start cell, then each (intended action) -> landing cell
        parts = [f"{start}"]
        for (s, r, ns, _d) in ep:
            parts.append(f" --{arrows[s]}{r:+.2f}--> {env.states[ns]}")
        print(f"ep {k}:  len={len(ep):2d}  return(undiscounted)={total:+.2f}")
        print("   " + "".join(parts) + "\n")


# ---------------------------------------------------------------------------
# LAYER 3: returns G_t — turn a trajectory into the numbers MC averages.
#
# G_t = r_{t+1} + gamma*r_{t+2} + ... ; computed in ONE backward sweep:
#   G = 0; for t = T-1..0:  G = r_{t+1} + gamma*G    (this G is G_t)
# ---------------------------------------------------------------------------

def returns_from_episode(episode, gamma):
    """Discounted return G_t for every step, via one O(T) backward sweep.
    Returns a list aligned with `episode`: returns[t] is the return of state s_t."""
    G = 0.0
    out = [0.0] * len(episode)
    for t in range(len(episode) - 1, -1, -1):
        r = episode[t][1]
        G = r + gamma * G
        out[t] = G
    return out


def exp_returns(seed=1):
    """Print one episode with its per-step return G_t, and verify the backward
    sweep equals the brute-force forward discounted sum."""
    env = GridWorld()
    gamma = env.gamma
    rng = np.random.default_rng(seed)
    sampler = Sampler(env, rng)
    opt_policy, _ = mdp_dp.value_iteration(env, gamma)

    # roll a handful, show the LONGEST so the backward sweep spans several steps.
    ep = max((generate_episode(sampler, opt_policy, rng) for _ in range(8)),
             key=len)
    G = returns_from_episode(ep, gamma)

    _banner(f"LAYER 3: returns G_t  (gamma={gamma}, episode len {len(ep)})")
    print(" t | s_t    r_{t+1} |   G_t  (= r + gamma*G_{t+1})")
    print("---+----------------+------------------------------")
    for t, (s, r, _ns, _d) in enumerate(ep):
        print(f"{t:2d} | {str(env.states[s]):6} {r:+.2f}  | {G[t]:+.4f}")

    # cross-check: brute-force forward sum for t=0 must match the backward sweep.
    rewards = [r for (_s, r, _ns, _d) in ep]
    G0_brute = sum(gamma ** k * rewards[k] for k in range(len(rewards)))
    print(f"\nbackward G_0 = {G[0]:+.6f}   forward-sum G_0 = {G0_brute:+.6f}   "
          f"match={np.isclose(G[0], G0_brute)}")
    print("note: earlier states have SMALLER G (more -0.04 tolls + more discount "
          "before the +1).")


# ---------------------------------------------------------------------------
# LAYER 4: mc_prediction — average first-visit returns over many episodes.
#   V(s) = mean of the G_t observed on the FIRST visit to s, across episodes.
# ---------------------------------------------------------------------------

def mc_prediction(env, sampler, policy, gamma, num_episodes, rng):
    """First-visit Monte Carlo estimate of V^pi."""
    returns_sum = np.zeros(env.nS)
    returns_cnt = np.zeros(env.nS)
    for _ in range(num_episodes):
        ep = generate_episode(sampler, policy, rng)
        G = returns_from_episode(ep, gamma)
        seen = set()
        for t, (s, _r, _ns, _d) in enumerate(ep):
            if s not in seen:                # first visit only
                seen.add(s)
                returns_sum[s] += G[t]
                returns_cnt[s] += 1
    V = np.zeros(env.nS)
    nz = returns_cnt > 0
    V[nz] = returns_sum[nz] / returns_cnt[nz]
    return V


def _rms(V_hat, V_true, states):
    d = V_hat[states] - V_true[states]
    return float(np.sqrt(np.mean(d ** 2)))


def _show_two(env, V_a, V_b, label_a, label_b):
    """Print two value functions on the grid, side by side, for eyeballing."""
    def grid_lines(V):
        rows = []
        for r in range(env.rows):
            cells = []
            for c in range(env.cols):
                if (r, c) in env.walls:
                    cells.append("  #  ")
                else:
                    cells.append(f"{V[env.s2i[(r, c)]]:+.2f}")
            rows.append(" ".join(cells))
        return rows
    A, B = grid_lines(V_a), grid_lines(V_b)
    print(f"{label_a:<26}   {label_b}")
    for la, lb in zip(A, B):
        print(f"{la:<26}   {lb}")


def exp_mc_prediction(seed=0):
    """Run first-visit MC, checkpointing RMS error vs the exact V^pi as episodes
    accumulate, then show the final estimate next to the ground truth."""
    env = GridWorld()
    gamma = env.gamma
    rng = np.random.default_rng(seed)
    sampler = Sampler(env, rng)
    opt_policy, _ = mdp_dp.value_iteration(env, gamma)
    V_true = mdp_dp.analytic_policy_value(env, opt_policy, gamma)
    nonterm = sampler._nonterminal

    _banner("LAYER 4: first-visit MC prediction (V^pi of the optimal policy)")
    print("episodes |  RMS error vs exact V^pi")
    print("---------+--------------------------")
    checkpoints = [100, 500, 2000, 10000, 50000]
    prev = 0
    returns_sum = np.zeros(env.nS)
    returns_cnt = np.zeros(env.nS)
    V = np.zeros(env.nS)
    for cp in checkpoints:
        for _ in range(cp - prev):           # extend, reusing accumulated counts
            ep = generate_episode(sampler, opt_policy, rng)
            G = returns_from_episode(ep, gamma)
            seen = set()
            for t, (s, _r, _ns, _d) in enumerate(ep):
                if s not in seen:
                    seen.add(s)
                    returns_sum[s] += G[t]
                    returns_cnt[s] += 1
        prev = cp
        nz = returns_cnt > 0
        V[nz] = returns_sum[nz] / returns_cnt[nz]
        print(f"{cp:8d} |  {_rms(V, V_true, nonterm):.4f}")

    print()
    _show_two(env, V_true, V, "exact V^pi (linear solve)", "MC estimate (50k eps)")


# ---------------------------------------------------------------------------
# LAYER 5: td0_prediction — bootstrap; update EVERY step, no episode-end wait.
#   target = r + gamma * V(s') * (1 - done)        # one reward + a GUESS
#   V(s)  += alpha * (target - V(s))               # nudge by the TD error
# ---------------------------------------------------------------------------

def td0_prediction(env, sampler, policy, gamma, num_episodes, alpha, rng):
    """TD(0) estimate of V^pi with constant step size alpha."""
    V = np.zeros(env.nS)
    for _ in range(num_episodes):
        for (s, r, ns, done) in generate_episode(sampler, policy, rng):
            target = r + gamma * V[ns] * (1.0 - done)   # no bootstrap past the end
            V[s] += alpha * (target - V[s])
    return V


def exp_td_prediction(seed=0, alpha=0.05):
    """TD(0) with the SAME checkpoints as MC, so you can compare convergence and
    see the constant-alpha noise floor (it hovers, never fully lands)."""
    env = GridWorld()
    gamma = env.gamma
    rng = np.random.default_rng(seed)
    sampler = Sampler(env, rng)
    opt_policy, _ = mdp_dp.value_iteration(env, gamma)
    V_true = mdp_dp.analytic_policy_value(env, opt_policy, gamma)
    nonterm = sampler._nonterminal

    _banner(f"LAYER 5: TD(0) prediction (alpha={alpha})")
    print("episodes |  RMS error vs exact V^pi")
    print("---------+--------------------------")
    checkpoints = [100, 500, 2000, 10000, 50000]
    prev = 0
    V = np.zeros(env.nS)
    for cp in checkpoints:
        for _ in range(cp - prev):
            for (s, r, ns, done) in generate_episode(sampler, opt_policy, rng):
                V[s] += alpha * (r + gamma * V[ns] * (1.0 - done) - V[s])
        prev = cp
        print(f"{cp:8d} |  {_rms(V, V_true, nonterm):.4f}")

    print()
    _show_two(env, V_true, V, "exact V^pi (linear solve)", "TD(0) estimate (50k eps)")


# ---------------------------------------------------------------------------
# LAYER 6: MC vs TD head-to-head.
#   A) bias/variance over many seeds (mean +/- std of RMS).
#   B) the batch A/B example — same data, different fixed points; TD = the
#      certainty-equivalence (max-likelihood-MDP) estimate, MC = training-set fit.
# ---------------------------------------------------------------------------

def exp_mc_vs_td(seeds=12, alpha=0.05):
    """Run BOTH methods for several episode budgets across many seeds; report mean
    and std of RMS vs the exact V^pi. TD: lower variance early; MC: lower bias
    (its constant-alpha-free 1/N average keeps shrinking, TD hits a noise floor)."""
    env = GridWorld()
    gamma = env.gamma
    opt_policy, _ = mdp_dp.value_iteration(env, gamma)
    V_true = mdp_dp.analytic_policy_value(env, opt_policy, gamma)

    _banner(f"LAYER 6A: MC vs TD(0) bias/variance  ({seeds} seeds, alpha={alpha})")
    print("episodes |     MC  mean+/-std    |     TD  mean+/-std")
    print("---------+-----------------------+----------------------")
    for n in (200, 1000, 5000, 25000):
        mc_errs, td_errs = [], []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            samp = Sampler(env, rng)
            nt = samp._nonterminal
            mc_errs.append(_rms(mc_prediction(env, samp, opt_policy, gamma, n, rng),
                                V_true, nt))
            rng = np.random.default_rng(seed)         # same episodes for TD
            samp = Sampler(env, rng)
            td_errs.append(_rms(td0_prediction(env, samp, opt_policy, gamma, n,
                                               alpha, rng), V_true, nt))
        print(f"{n:8d} |  {np.mean(mc_errs):.4f} +/- {np.std(mc_errs):.4f}   "
              f"|  {np.mean(td_errs):.4f} +/- {np.std(td_errs):.4f}")


# The classic A/B batch (Sutton & Barto, Example 6.4). gamma = 1, two states A,B.
# States: A=0, B=1; terminal flagged by done. Eight episodes:
#   1x   A -(0)-> B -(0)-> end
#   6x   B -(1)-> end
#   1x   B -(0)-> end
A, B = 0, 1
_AB_EPISODES = (
    [[(A, 0.0, B, False), (B, 0.0, -1, True)]]
    + [[(B, 1.0, -1, True)]] * 6
    + [[(B, 0.0, -1, True)]]
)


def batch_mc(episodes, gamma=1.0):
    """Batch (every-visit) MC fixed point: V(s) = mean observed return from s."""
    rsum, rcnt = {}, {}
    for ep in episodes:
        G = 0.0
        for t in range(len(ep) - 1, -1, -1):
            s, r = ep[t][0], ep[t][1]
            G = r + gamma * G
            rsum[s] = rsum.get(s, 0.0) + G
            rcnt[s] = rcnt.get(s, 0) + 1
    return {s: rsum[s] / rcnt[s] for s in rsum}


def batch_td(episodes, gamma=1.0, alpha=0.01, sweeps=20000):
    """Batch TD(0) fixed point: present all transitions repeatedly until V stops
    moving. Converges to the value of the MAXIMUM-LIKELIHOOD MDP built from the
    data (certainty equivalence)."""
    V = {A: 0.0, B: 0.0}
    for _ in range(sweeps):
        for ep in episodes:
            for (s, r, ns, done) in ep:
                target = r + gamma * (0.0 if done else V[ns])
                V[s] += alpha * (target - V[s])
    return V


def exp_batch_ab():
    """Same 8 episodes, two fixed points: MC says V(A)=0 (its one episode returned
    0); TD says V(A)=0.75 (A always leads to B, and V(B)=0.75)."""
    _banner("LAYER 6B: batch A/B — TD is 'right', MC fits the training returns")
    vmc = batch_mc(_AB_EPISODES)
    vtd = batch_td(_AB_EPISODES)
    print("       V(A)     V(B)")
    print(f"MC :  {vmc[A]:+.3f}   {vmc[B]:+.3f}     <- mean return seen from each state")
    print(f"TD :  {vtd[A]:+.3f}   {vtd[B]:+.3f}     <- value of the implied (ML) MDP")
    print("\nBoth agree V(B)=0.75 (6 of 8 B-episodes paid 1). They split on A:")
    print("  MC: the ONE episode through A returned 0  => V(A)=0.")
    print("  TD: A -> B in 100% of data, and V(B)=0.75 => V(A)=0+0.75=0.75.")
    print("  TD's is the certainty-equivalence estimate (build the empirical MDP,")
    print("  solve it) — it generalizes; MC just minimizes error on observed returns.")


# ---------------------------------------------------------------------------
# LAYER 7 (bonus): the bias/variance DIAL — sweep the n-step return from n=1
# (TD(0)) to n=inf (MC) and measure bias and variance of the TARGET directly.
#   G_n = r1 + ... + gamma^(n-1) r_n + gamma^n * V_boot(s_n)
# bias comes from V_boot being WRONG (shrinks as n uses more real reward);
# variance comes from the LENGTH of the sampled tail (grows with n).
# ---------------------------------------------------------------------------

def _rollout_from(sampler, policy, rng, s0, max_steps=1000):
    """One episode forced to START at state s0 (not a random exploring start)."""
    ep = []
    sampler.s = s0
    s = s0
    for _ in range(max_steps):
        a = int(rng.choice(sampler.env.nA, p=policy[s]))
        ns, r, done = sampler.step(a)
        ep.append((s, r, ns, done))
        s = ns
        if done:
            break
    return ep


def _nstep_return(ep, n, gamma, V_boot):
    """n-step return from the START of ep: n real rewards then bootstrap V_boot(s_n).
    If the episode ends within n steps it's the full return (terminal => no bootstrap)."""
    L = len(ep)
    m = min(n, L)
    G, disc = 0.0, 1.0
    for k in range(m):
        G += disc * ep[k][1]
        disc *= gamma
    if n < L:                                # s_n exists and is non-terminal
        G += disc * V_boot[ep[n - 1][2]]     # disc == gamma^n here
    return G


def exp_nstep_dial(M=4000, seed=0):
    """Sweep n and report, aggregated over all non-terminal start states:
    RMS bias, typical target std (sqrt mean variance), and RMSE = sqrt(bias^2+var).
    Run with a WRONG bootstrap (V=0) and a CORRECT one (V=V_true) to localize bias."""
    env = GridWorld()
    gamma = env.gamma
    opt_policy, _ = mdp_dp.value_iteration(env, gamma)
    V_true = mdp_dp.analytic_policy_value(env, opt_policy, gamma)

    ns_list = [1, 2, 3, 5, 10, 30, np.inf]
    for boot_label, V_boot in (("WRONG  bootstrap V=0", np.zeros(env.nS)),
                               ("CORRECT bootstrap V=V_true", V_true)):
        rng = np.random.default_rng(seed)
        sampler = Sampler(env, rng)
        nonterm = sampler._nonterminal
        # M rollouts per start state, reused across all n (fair variance comparison).
        rollouts = {s: [_rollout_from(sampler, opt_policy, rng, s) for _ in range(M)]
                    for s in nonterm}

        _banner(f"LAYER 7: n-step bias/variance dial  ({boot_label}, M={M})")
        print("   n  |  RMS bias  | target std |   RMSE   | dial")
        print("------+------------+------------+----------+" + "-" * 24)
        for n in ns_list:
            nn = env.nS if n == np.inf else n
            biases, varis = [], []
            for s in nonterm:
                Gs = np.array([_nstep_return(ep, nn, gamma, V_boot)
                               for ep in rollouts[s]])
                biases.append(Gs.mean() - V_true[s])
                varis.append(Gs.var())
            rms_bias = float(np.sqrt(np.mean(np.square(biases))))
            tgt_std = float(np.sqrt(np.mean(varis)))
            rmse = float(np.sqrt(rms_bias ** 2 + tgt_std ** 2))
            label = "inf(MC)" if n == np.inf else str(n)
            bar = "B" * int(rms_bias * 20) + "v" * int(tgt_std * 20)
            print(f"{label:>5} |   {rms_bias:.4f}   |   {tgt_std:.4f}   | "
                  f" {rmse:.4f} | {bar}")
        print("   (B = bias units, v = std units; watch B shrink & v grow as n->inf)\n")


def run_experiments():
    # exp_samples_are_the_model()
    exp_one_episode()
    # exp_returns()
    # exp_mc_prediction()
    # exp_td_prediction()
    # exp_mc_vs_td()
    # exp_batch_ab()
    # exp_nstep_dial()


@contextlib.contextmanager
def _tee(path):
    """Print to BOTH the terminal and `path` (long runs survive scrollback)."""
    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                st.write(s)

        def flush(self):
            for st in self.streams:
                st.flush()

    with open(path, "w") as f:
        with contextlib.redirect_stdout(_Tee(sys.stdout, f)):
            yield
    print(f"(output also written to {path})", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", metavar="FILE",
                        help="also write all output to FILE")
    args = parser.parse_args()

    if args.out:
        with _tee(args.out):
            run_experiments()
    else:
        run_experiments()
