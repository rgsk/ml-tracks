# Step-by-step teaching approach (for hard exercises)

Point me to this file when an exercise feels too hard to attempt cold, and you
want me to **walk you through building it from scratch** instead of handing you a
skeleton-with-TODOs to fill in. This is a *different mode* from the usual
exercise-driven workflow (where I write skeleton + TODOs and you fill them in).

## The deal

- We work in a scratch file (e.g. `rl/rough.py`), NOT the real exercise file.
  The real exercise file stays untouched with its TODOs, so you can still do it
  yourself afterwards from understanding.
- **No TODOs in the scratch file.** In this mode I write the actual code. You
  read, run, and absorb. The point is to *reveal the problem slowly*, not to
  test you yet.
- I reveal **one small piece per turn** and wait for you to say "next". I do not
  dump the whole solution.

## What counts as "one small piece"

Small. Smaller than feels necessary. Even the *environment* gets built in
layers, not assumed. For the MDP/DP exercise the steps were:

1. Geometry (cells, states, integer indexing)
2. Deterministic movement (physics only: cell + action -> cell)
3. Stochastic slip (action -> probability distribution over outcomes)
4. Rewards + termination -> the full model `P`
5. The reusable atom (`q_from_v`, the one-step Bellman backup)
6-9. Each algorithm that wraps the atom, one at a time

Rule of thumb: each step is one function or one concept, and each builds
directly on the last. Pure helpers first (no side concerns), then layer
behavior on top. Keep layers cleanly separated (e.g. movement knows nothing
about rewards or terminals).

## The rhythm of each step

1. **Plain-words intuition first**, before any code. Why does this piece exist,
   what problem does it solve.
2. **Write the code** into the scratch file — small, readable, commented with
   the *why*, matching surrounding style.
3. **Run it immediately** and show real output.
4. **Explain the output** by hand — verify a number or two with arithmetic so
   it's concrete, not hand-wavy.
5. **End with a couple of quick check questions** for you to answer (no pressure,
   they surface the subtlety of that step). Often plant a hook for the next step.
6. Wait for "next".

## Style constraints (mine, honor them)

- I'm an ML-interview candidate, intermediate PyTorch, I learn by implementing.
- No LaTeX in chat — the client doesn't render it. Use ASCII / unicode math in
  code blocks. (LaTeX is fine inside .ipynb cells.)
- Call out the interview-relevant framing when it exists (e.g. "policy
  evaluation is just a linear solve").
- Point out the *famous*/non-obvious results explicitly (e.g. the gridworld
  optimal policy steering AWAY from the goal near the trap).

## After the walkthrough

Once the whole thing is built in scratch, push me back to the **real exercise
file** to reconstruct it myself against its actual signatures — that's what makes
it stick. Offer to review when I'm done.
