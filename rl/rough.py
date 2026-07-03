"""
WALKTHROUGH: Function approximation & the deadly triad (exercise 5), one layer at a time.

The shift from exercises 1-4: those kept a NUMBER for every state (or every (s,a)) —
a TABLE. That works when states are few and discrete. It dies the moment the state
space is large or continuous: CartPole's state is four real numbers, so there is no
"cell" to index, and you never see the exact same state twice. The fix is to
APPROXIMATE the value with a parameterized function and learn its WEIGHTS:

    tabular:   V is a lookup table, one free number per state
    approx:    V(s; w) = w · phi(s)   (linear), or a neural net   — a few weights, shared

Layers (run each `exp_*`, watch the output, then say "next"):
  1. FEATURES — what replaces the table. Built here in small sub-steps:
       1a. a value is a dot product w · phi(s); one-hot features just read a table. (here)
       1b. coarser features SHARE weights across states -> generalization.
       1c. capacity vs structure: which features can represent which values.
  2. semi-gradient TD — LEARN the weights from samples by SGD.
  3. why "semi" — full-gradient (residual) vs semi-gradient.
  4. Baird's counterexample — the DEADLY TRIAD makes the weights diverge.
  5. a real NEURAL net on CartPole — naive online Q-learning is unstable -> exercise 6.
"""

import argparse
import contextlib
import sys

import numpy as np


# ---------------------------------------------------------------------------
# A tiny env we'll reuse for layers 1-3: Sutton & Barto's 19-state random walk.
# States 1..19 in a line; each step goes left/right with equal probability;
# terminate at 0 (reward 0) or 20 (reward +1); gamma = 1. Because reward only
# arrives at the right end, the TRUE value is exactly linear: V(s) = s / 20.
# (No learning yet in layer 1 — we only borrow its states and its true values.)
# ---------------------------------------------------------------------------
N = 19
STATES = np.arange(1, N + 1)                       # the non-terminal states
TRUE_V = STATES / (N + 1)                          # V(s) = s/20, exactly linear


def _banner(*lines):
    print("=" * 70)
    for line in lines:
        print(line)
    print("=" * 70)


# ---------------------------------------------------------------------------
# LAYER 1a: a value is a DOT PRODUCT, and a table is the one-hot special case.
#
# The whole idea of (linear) function approximation is one formula:
#     V(s; w) = w · phi(s)
# phi(s) is a FEATURE VECTOR describing state s as numbers; w is a weight vector;
# the value is their dot product. Different choices of phi give different ways of
# representing V — that's the only knob.
#
# The simplest possible phi is ONE-HOT: phi(s) = e_s, a vector that is 1 in slot s
# and 0 everywhere else. Then the dot product just picks out one weight:
#     w · e_s = w[s].
# So with one-hot features the weight vector w IS the lookup table, entry for entry.
# Everything we did in exercises 1-4 was secretly this. We prove it below: invent a
# table of made-up values, set w = that table, and check w · phi(s) reads it back.
# ---------------------------------------------------------------------------

def phi_onehot(s: int) -> np.ndarray:
    """One-hot feature for state s: a length-N vector, 1 in slot (s-1), else 0.
    (States are numbered 1..N, so state s lives at index s-1.)"""
    v = np.zeros(N)
    v[s - 1] = 1.0
    return v


def exp_1a_onehot(seed=0):
    """Make up an arbitrary 'table' of values, set the weights equal to it, and show
    V(s; w) = w · phi(s) reproduces the table exactly — for one-hot features, w IS the
    table. We print the actual feature vectors for a few states so the dot product is
    concrete, not abstract."""
    rng = np.random.default_rng(seed)
    table = rng.uniform(0, 1, size=N).round(2)     # a made-up value for each state
    w = table.copy()                               # <-- the weights ARE the table

    _banner("LAYER 1a: V(s;w) = w · phi(s);  one-hot features  =>  w is the table")

    # Show the feature vectors for a few states (they are just unit vectors).
    for s in (3, 10, 17):
        phi = phi_onehot(s)
        print(f"  phi({s:2d}) = {phi.astype(int)}   (a 1 only in slot {s - 1})")

    print("\n   s | table[s] |  w · phi(s)  | match")
    print("  ---+----------+--------------+------")
    for s in (3, 10, 17):
        v = float(w @ phi_onehot(s))               # the dot product = the value
        print(f"  {s:2d} |   {table[s - 1]:.2f}   |     {v:.2f}     |  {np.isclose(v, table[s - 1])}")

    # And check it for ALL states at once, not just the three we printed.
    allV = np.array([w @ phi_onehot(s) for s in STATES])
    print(f"\n  over all {N} states: max |w·phi(s) - table[s]| = {np.abs(allV - table).max():.1e}")
    print("\n  So 'a Q-table' was never special — it's linear function approximation with")
    print("  one-hot features, where every state owns one private weight and nothing is")
    print("  shared. Next (1b): pick a SMALLER phi so states start sharing weights.")


# ---------------------------------------------------------------------------
# LAYER 1b: a coarser phi SHARES weights across states  ->  generalization.
#
# One-hot gave every state its own weight (no sharing). Now use FEWER weights than
# states and force sharing. State AGGREGATION is the simplest way: chop the line of
# states into a few contiguous bins, and let phi(s) be one-hot over BINS. Every state
# in a bin reads the SAME weight, so they're locked to the same value. Two consequences
# we'll see: (1) states in a bin always share a value; (2) changing ONE weight moves a
# whole bin of states at once. That second thing is the entire point of FA: an update
# from one state's experience spills over to its neighbours. With continuous states you
# have no choice — you can't store a weight per state, so you MUST share.
# ---------------------------------------------------------------------------

def phi_aggregate(s: int, groups: int = 4) -> np.ndarray:
    """State aggregation: bucket the N states into `groups` contiguous bins; phi is
    one-hot over the BIN. All states in a bin share one weight (their common value)."""
    v = np.zeros(groups)
    g = min((s - 1) * groups // N, groups - 1)
    v[g] = 1.0
    return v


def exp_1b_sharing(groups=4):
    """Show weight-sharing concretely: which states land in which bin, that bin-mates
    get identical values, and that bumping ONE weight moves a whole bin at once."""
    _banner(f"LAYER 1b: aggregate phi ({groups} bins, {N} states)  =>  weights are SHARED")

    # which states share each bin?
    members = {g: [int(s) for s in STATES if phi_aggregate(s, groups).argmax() == g]
               for g in range(groups)}
    print("  bins (states that share one weight):")
    for g in range(groups):
        print(f"    bin {g}:  states {members[g]}")

    # give each bin a weight, then read off every state's value = w[its bin].
    w = np.array([0.10, 0.30, 0.60, 0.90])[:groups]
    values = np.array([w @ phi_aggregate(s, groups) for s in STATES])
    print(f"\n  weights per bin: {w}")
    print("  value each state gets (= its bin's weight):")
    print("    " + "  ".join(f"{int(s)}:{values[i]:.2f}" for i, s in enumerate(STATES)))

    # bump ONE weight; count how many states' values move.
    w2 = w.copy()
    w2[1] += 0.20                                  # nudge bin 1 only
    values2 = np.array([w2 @ phi_aggregate(s, groups) for s in STATES])
    moved = [int(s) for i, s in enumerate(STATES) if not np.isclose(values[i], values2[i])]
    print(f"\n  bumped weight of bin 1 by +0.20  ->  moved states {moved}")
    print(f"  one weight changed, {len(moved)} states moved together. THAT is generalization:")
    print("  experience at one state updates its bin-mates too. (one-hot moves exactly 1.)")
    print("\n  Trade-off teaser: bin-mates are FORCED equal, but the true V differs across")
    print("  them — so aggregation can't be exact. Next (1c): which phi can represent which V.")


# ---------------------------------------------------------------------------
# LAYER 1c: capacity vs structure — which phi can represent which V.
#
# Sharing (1b) buys generalization but spends CAPACITY: a feature map can only
# represent value functions that lie in its "span". To measure that cleanly, we ask
# for the BEST-POSSIBLE weights for a target (least squares, since we happen to know
# the target here) and look at the leftover RMS error:
#     RMS 0  => phi CAN represent this target exactly
#     RMS >0 => phi is too coarse for it; that's a floor no learning can beat.
# We try three feature maps against two targets:
#     A) the TRUE value  V(s)=s/20   (a straight line — has structure)
#     B) an ARBITRARY bumpy table    (no structure at all).
# One more feature map joins here: 'position + bias', phi(s)=[s/20, 1] — a slanted
# line, only 2 weights. The point of the whole layer lands in this table.
# ---------------------------------------------------------------------------

def phi_position(s: int) -> np.ndarray:
    """Normalized position + bias: phi(s) = [s/(N+1), 1]. Two weights for all N states.
    V(s;w) = w0*(s/20) + w1 — a straight line in s (contrast 1b's flat steps)."""
    return np.array([s / (N + 1), 1.0])


def _best_fit_rms(phi, target: np.ndarray) -> tuple[int, float]:
    """Best linear readout of `target` from feature map `phi`, via least squares.
    Returns (#weights, RMS residual). RMS 0 => phi can represent target exactly."""
    Phi = np.stack([phi(s) for s in STATES])       # (N, d) feature matrix
    w, *_ = np.linalg.lstsq(Phi, target, rcond=None)
    return Phi.shape[1], float(np.sqrt(np.mean((Phi @ w - target) ** 2)))


def exp_1c_capacity(seed=0):
    """Fit two targets with three feature maps; the RMS floor shows what each phi CAN
    and CANNOT represent — capacity vs structure, the whole point of layer 1."""
    rng = np.random.default_rng(seed)
    target_linear = TRUE_V                          # has structure (a line)
    target_arbitrary = rng.uniform(0, 1, size=N)    # no structure

    _banner("LAYER 1c: capacity vs structure — best-possible fit per feature map")
    print("  feature map           | #weights | RMS fit to TRUE-linear V | RMS fit to ARBITRARY table")
    print("  ----------------------+----------+--------------------------+---------------------------")
    for name, phi in (("one-hot  (tabular)", phi_onehot),
                      ("aggregate (4 bins)", lambda s: phi_aggregate(s, 4)),
                      ("position + bias   ", phi_position)):
        _d, rms_lin = _best_fit_rms(phi, target_linear)
        d, rms_arb = _best_fit_rms(phi, target_arbitrary)
        print(f"  {name}    |    {d:2d}    |          {rms_lin:.4f}          |          {rms_arb:.4f}")

    print("\n  Reading it:")
    print("   - one-hot: 19 weights, fits ANY target exactly (it's the table). No")
    print("     generalization, and impossible for continuous states.")
    print("   - position+bias: just 2 weights, yet fits the TRUE V EXACTLY (RMS 0) —")
    print("     the env's value really is the line s/20, so w=[1,0] nails it. But it")
    print("     CAN'T fit the arbitrary table: 2 knobs can't bend into a bumpy shape.")
    print("   - aggregate: 4 flat steps — approximate on both, exact on neither.")
    print("  So the RIGHT features (structure matching the value) let a FEW shared weights")
    print("  be exact AND generalize. The wrong ones leave an error floor no learning beats.")
    print("\n  That closes layer 1 (what replaces the table). We cheated by using lstsq —")
    print("  we knew the target. The agent only sees sampled (s, r). Layer 2: LEARN w from")
    print("  samples by SGD, which turns out to be 'semi-gradient TD'.")


# ---------------------------------------------------------------------------
# A reset()/step() view of the random walk, so the agent only ever SEES samples
# (s, r, done) — never the target values. (Layer 1 peeked at TRUE_V; from here on the
# learner is blind to it, exactly like a real agent.)
# ---------------------------------------------------------------------------

class RandomWalk:
    """Gym-like random walk. reset() starts in the middle; step() ignores its action
    (the policy is fixed: uniform left/right) and returns (s', r, done)."""

    def __init__(self, rng):
        self.rng = rng
        self.s = None

    def reset(self):
        self.s = (N + 1) // 2                        # start in the middle (state 10)
        return self.s

    def step(self):
        self.s += int(self.rng.choice([-1, 1]))      # equal-prob left/right
        if self.s == 0:
            return self.s, 0.0, True                 # left terminal, reward 0
        if self.s == N + 1:
            return self.s, 1.0, True                 # right terminal, reward +1
        return self.s, 0.0, False


# ---------------------------------------------------------------------------
# LAYER 2a: learning w from ONE sample — the semi-gradient TD update.
#
# We can't call lstsq anymore: the agent never sees the target values, only sampled
# transitions (s, r, s'). So do what supervised learning does — SGD. Treat the value
# error as a loss and step w downhill:
#     loss(s) = 1/2 (target - V(s;w))^2         where target = r + gamma*V(s';w)
#     d loss / d w = -(target - V(s;w)) * grad_w V(s;w)
# For the LINEAR value V(s;w) = w·phi(s), the gradient grad_w V is simply phi(s)
# (your guess from 1a). So one SGD step is:
#     delta = target - V(s;w)                   # the TD error
#     w    += alpha * delta * phi(s)            # move phi(s)'s direction by the error
# This is exactly TD(0) from exercise 3, but nudging WEIGHTS instead of a table cell.
# (There's a subtlety in calling this a real gradient — the "semi" — but park it until
# layer 3; the update above is what everyone uses.) We do ONE update here and watch
# two things: V(s) moves toward the target, AND other states move too (shared weights).
# ---------------------------------------------------------------------------

def exp_2a_one_update(alpha=0.1, gamma=1.0):
    """A single semi-gradient TD update, by hand. The agent stands on state 19 and
    steps RIGHT into the +1 terminal. Watch w change, V(19) move toward the target,
    and — the surprise — values at states it NEVER visited move too (generalization)."""
    _banner("LAYER 2a: one semi-gradient TD update  (w += alpha * delta * phi(s))")

    w = np.zeros(2)                                  # start blank: V(s)=0 everywhere
    s = 19
    r, ns, done = 1.0, 20, True                      # from 19, step right -> +1 terminal

    v_s = w @ phi_position(s)                         # current estimate V(19)
    target = r + gamma * 0.0 * (1.0 - done)          # done -> no bootstrap; target = r = 1
    delta = target - v_s                             # TD error
    print(f"  before:  w = {w}")
    print(f"  sample:  s=19  ->  r={r}, s'={ns}, done={done}")
    print(f"  phi(19) = {phi_position(19)}   (that's [19/20, 1])")
    print(f"  V(19)   = w·phi(19) = {v_s:.3f}")
    print(f"  target  = r + gamma*V(s')*(1-done) = {target:.3f}   (terminal: just the reward)")
    print(f"  delta   = target - V(19) = {delta:.3f}")

    v_others_before = np.array([w @ phi_position(x) for x in (5, 10, 19)])
    w = w + alpha * delta * phi_position(s)          # THE update
    v_others_after = np.array([w @ phi_position(x) for x in (5, 10, 19)])

    print(f"\n  after:   w = {w.round(4)}   (= alpha*delta*phi(19) = {alpha}*{delta:.0f}*{phi_position(19)})")
    print(f"  V(19):  {v_others_before[2]:.3f}  ->  {v_others_after[2]:.3f}   "
          "(moved toward target 1.0, by one alpha-step)")
    print("\n  the surprise — states we NEVER visited also changed:")
    print("     s  |  V before |  V after")
    print("    ----+-----------+---------")
    for x, b, a in zip((5, 10, 19), v_others_before, v_others_after):
        tag = "  <- visited" if x == 19 else "  <- untouched, yet moved"
        print(f"    {x:2d}  |   {b:.3f}   |  {a:.3f}{tag}")
    print("\n  ONE update to shared weights nudged the WHOLE value function. That spillover")
    print("  is generalization doing its job (and, later, also how things go wrong).")
    print("  Next (2b): run this update over thousands of steps and watch V converge to truth.")


# ---------------------------------------------------------------------------
# LAYER 2b: run the update to convergence — V finds the truth from samples alone.
#
# Same one-line update as 2a, now applied to every transition of thousands of
# episodes, with the EXACT feature map (position+bias, which 1c showed can represent
# V perfectly). No target is ever shown; the learner only sees (s, r, s'). We
# checkpoint the RMS error vs the (secret) TRUE_V as episodes accumulate. alpha is
# annealed toward 0 so the estimate settles instead of jittering around a noise floor.
# Expect RMS -> ~0 and the recovered weights w -> [1, 0]  (i.e. V(s) = s/20).
# ---------------------------------------------------------------------------

def semi_gradient_td(env, phi, gamma, num_episodes, alpha0):
    """Linear semi-gradient TD(0). Returns the learned weights w. alpha anneals
    linearly alpha0 -> 0 over training. (env is a RandomWalk; phi is a feature map.)"""
    w = np.zeros(len(phi(1)))
    for ep in range(num_episodes):
        alpha = alpha0 * (1 - ep / num_episodes)
        s = env.reset()
        while True:
            ns, r, done = env.step()
            target = r + gamma * (w @ phi(ns)) * (1.0 - done)   # semi-grad: target fixed
            w = w + alpha * (target - w @ phi(s)) * phi(s)
            s = ns
            if done:
                break
    return w


def _rms_vs_true(w, phi):
    est = np.array([w @ phi(s) for s in STATES])
    return float(np.sqrt(np.mean((est - TRUE_V) ** 2)))


def exp_2b_converge(seed=0, gamma=1.0, alpha0=0.02, num_episodes=30000):
    """Train semi-gradient TD with the EXACT (position) features, checkpointing RMS vs
    the true V. It should march to ~0 and recover w ≈ [1, 0] — learned from samples,
    no target and no model ever provided."""
    _banner(f"LAYER 2b: semi-gradient TD converges from samples "
            f"(position features, alpha {alpha0}->0)")
    rng = np.random.default_rng(seed)
    env = RandomWalk(rng)
    phi = phi_position
    w = np.zeros(len(phi(1)))
    print("   episodes |  RMS vs true V")
    print("   ---------+---------------")
    prev = 0
    for cp in [100, 500, 2000, 10000, 30000]:
        for ep in range(prev, cp):
            alpha = alpha0 * (1 - ep / num_episodes)
            s = env.reset()
            while True:
                ns, r, done = env.step()
                target = r + gamma * (w @ phi(ns)) * (1.0 - done)
                w = w + alpha * (target - w @ phi(s)) * phi(s)
                s = ns
                if done:
                    break
        prev = cp
        print(f"   {cp:8d} |     {_rms_vs_true(w, phi):.4f}")

    print(f"\n  final w = {w.round(3)}  (true is [1, 0], i.e. V(s) = 1*(s/20) + 0)")
    print("  The SAME nudge from 2a, repeated on samples, recovered the true values —")
    print("  no target ever shown, no model. With features that CAN fit V, TD nails it.")
    print("\n  Lurking question for 2c: back in 1c the aggregate floor was 0.068, but does")
    print("  TD actually REACH that floor when features are too coarse? (Spoiler: no.)")


# ---------------------------------------------------------------------------
# LAYER 2c: TD's fixed point is NOT the best-possible fit.
#
# With EXACT features (2b) TD nailed the truth. With COARSE features it doesn't just
# lose a little to capacity — it converges to a DIFFERENT solution than you'd hope.
# Two learners, same aggregate features, same data:
#   gradient Monte Carlo: target = the actual return G_t (no bootstrap). It directly
#       minimizes value error, so it reaches the capacity floor from 1c (~0.068).
#   semi-gradient TD(0):  target = r + gamma*V(s'). The bootstrap makes it converge to
#       the "projected Bellman" fixed point, WEIGHTED by how often each state is
#       visited (mu) — a different objective. Its error is only bounded by
#           VE(TD) <= 1/(1-gamma) * (best-possible VE),
#       and with gamma=1 that bound is INFINITE — so TD can sit far above the floor.
#       Here it compresses every group toward the middle.
# MC = unbiased-but-noisy target; TD = biased-but-low-variance target. The same
# bias/variance split as exercise 3, now visible in WHERE the weights settle.
# ---------------------------------------------------------------------------

def gradient_mc(env, phi, gamma, num_episodes, alpha0):
    """Gradient Monte Carlo: like semi_gradient_td but the target is the real return
    G_t (computed at episode end), so there is NO bootstrap. Returns weights w."""
    w = np.zeros(len(phi(1)))
    for ep in range(num_episodes):
        alpha = alpha0 * (1 - ep / num_episodes)
        s = env.reset()
        traj = []
        while True:
            ns, r, done = env.step()
            traj.append((s, r))
            s = ns
            if done:
                break
        G = 0.0
        for (st, r) in reversed(traj):             # returns via one backward sweep
            G = r + gamma * G
            w = w + alpha * (G - w @ phi(st)) * phi(st)
    return w


def exp_2c_td_fixed_point(seed=0, gamma=1.0, alpha0=0.02, num_episodes=30000):
    """Same coarse (aggregate) features for both learners. MC reaches the ~0.068
    capacity floor; TD converges elsewhere (~0.20), its group values squashed toward
    the middle. The gap is the TD fixed point, not a lack of capacity."""
    phi = lambda s: phi_aggregate(s, 4)
    _banner("LAYER 2c: TD's fixed point != best-possible fit (aggregate features)")

    Phi = np.stack([phi(s) for s in STATES])
    w_ls, *_ = np.linalg.lstsq(Phi, TRUE_V, rcond=None)   # best-possible (knows target)
    w_mc = gradient_mc(RandomWalk(np.random.default_rng(seed)), phi, gamma,
                       num_episodes, alpha0)
    w_td = semi_gradient_td(RandomWalk(np.random.default_rng(seed)), phi, gamma,
                            num_episodes, alpha0)

    print("  method                | RMS vs true V | per-bin weights")
    print("  ----------------------+---------------+-----------------------------")
    print(f"  best-possible (lstsq) |    {_rms_vs_true(w_ls, phi):.4f}     | {w_ls.round(3)}")
    print(f"  gradient Monte Carlo  |    {_rms_vs_true(w_mc, phi):.4f}     | {w_mc.round(3)}")
    print(f"  semi-gradient TD(0)   |    {_rms_vs_true(w_td, phi):.4f}     | {w_td.round(3)}")
    print(f"\n  ideal per-bin values (mean true V in each bin): "
          f"{np.array([TRUE_V[[s-1 for s in STATES if phi(s).argmax()==g]].mean() for g in range(4)]).round(3)}")

    print("\n  MC's weights match best-possible: its target is the real return, so it")
    print("  minimizes value error and reaches the capacity floor. TD's weights are")
    print("  squashed toward the middle: its bootstrapped target converges to a different")
    print("  (projected-Bellman, mu-weighted) fixed point. Bound VE(TD) <= 1/(1-gamma)*best")
    print("  is vacuous at gamma=1, so TD drifts well above the floor.")
    print("\n  Takeaway: 'can TD beat 0.068?' -> no; and it may not even REACH it. The bias")
    print("  is the SAME bootstrap that made TD low-variance in ex3 — free lunch has a bill.")
    print("  Next (layer 3): the word 'semi' — what we quietly skipped in the gradient.")


def run_experiments():
    # exp_1a_onehot()
    # exp_1b_sharing()
    exp_1c_capacity()
    # exp_2a_one_update()
    # exp_2b_converge()
    exp_2c_td_fixed_point()
    # ... layer 3+ later


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
    parser.add_argument("--out", metavar="FILE", help="also write all output to FILE")
    args = parser.parse_args()

    if args.out:
        with _tee(args.out):
            run_experiments()
    else:
        run_experiments()
