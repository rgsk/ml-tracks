# Five-Week Roadmap — Sim-Only Manipulation Sprint

> The time-boxed *when/do*. For the *why/what* (brain-vs-body, track map, RM/WM, world_model),
> see [startup-roadmap.md](startup-roadmap.md).

## Constraints

- **Time:** 5 weeks, deadline ~**Sep 1, 2026** (started Jul 24).
- **Hardware:** **no robot arm.** Everything **sim-only**. Arm purchase is the *outcome*, not a dependency.
- **Compute:** rented **RunPod** GPUs.
- **Goal:** understand the manipulation stack deeply enough to confidently buy an arm and start
  experimenting on general manipulation (pick-and-place first; napkin folding as aspiration).

## Priority

```- 
finish diffusion → Diffusion Policy → ACT → Reward Model → VLA (only if time)
```

- **Core commitment = diffusion / DP / ACT / RM (weeks 1-4).**
- **VLA = week-5 stretch**, done only if ahead of schedule.
- **World model = NOT in this sprint** — parked; see startup-roadmap.md.
- Diffusion Policy is the bridge — it *is* the diffusion→policy step (reuses the denoiser I'm building now).

## Schedule

```- 
Wk 1  Finish diffusion (mid-way now) + stand up LeRobot on RunPod.
      Payoff: Push-T env runs; I understand the demo-data format.

Wk 2  Diffusion Policy on Push-T   (reuses my diffusion denoiser).
      Payoff: watch the policy push the T-block into the goal outline.

Wk 3  ACT on ALOHA transfer-cube   (gym-xarm lift as a warm-up if needed).
      Payoff: bimanual cube hand-off; I understand action chunking.

Wk 4  Reward Model rung 1-2 on MY Wk2-3 rollouts + failure detection.
      Payoff: RM flags which DP/ACT episodes failed → full-circle data loop.

Wk 5  VLA taste on LIBERO — IF ahead of schedule.
      Otherwise: buffer + write the "ready to buy arm" checklist.
```

### Week detail

**Wk 1 — diffusion + LeRobot.** Close out the in-progress diffusion track (denoiser/training). In
parallel, get LeRobot running on RunPod, launch `gym-pusht`, and read/understand the LeRobot demo-dataset
format (this format is what I'll reuse for real teleop data later).

**Wk 2 — Diffusion Policy on Push-T.** Train DP on the Push-T demos. This is the highest-synergy step:
DP is a conditional DDPM that denoises into an *action sequence* instead of pixels, so it reuses the exact
denoiser machinery from the diffusion track. Payoff = the circular pusher nudging the T into its outline.

**Wk 3 — ACT on ALOHA transfer-cube.** Train ACT (CVAE + transformer, predicting action *chunks*) on the
bimanual transfer-cube task. Warm up on single-arm `gym-xarm` lift if transfer-cube is too much at once.
Payoff = right arm picks the cube, hands it to the left.

**Wk 4 — Reward model on my own rollouts.** Rung 1: zero-shot VLM progress estimation on the DP/ACT
rollouts from weeks 2-3. Rung 2 (lite): train a small progress/value head. Use it for **failure
detection** — flag episodes that plateaued or never hit progress=1. This closes the data-flywheel loop
against my *own* policies, which is the most startup-relevant skill of the sprint.

**Wk 5 — VLA taste OR buffer.** If ahead: run/eval a VLA (OpenVLA or pi0) on **LIBERO**, read the code,
feel the "change the sentence → behavior changes" moment. Not mastery — understanding. If behind: use as
buffer and write the arm-buying checklist in startup-roadmap.md.

## Out of scope for these 5 weeks (and why)

- **Deformables / cloth (napkin folding itself)** — no cheap sim does cloth well; waits for the real arm.
- **Real hardware / teleop** — no arm yet; sim pipeline transfers 1:1 on arm-day (LeRobot).
- **Tactile / force control** — body-stack, needs hardware.
- **Full VLA mastery** — week-5 is a taste only.
- **World-model planning (the `world_model` repo)** — parked capstone; needs the RM first anyway.

## Calibration

5 weeks is tight but doable if the target is **"working pipeline + solid understanding," not mastery.**
Weeks 2-3 (DP + ACT) are the core reliable payoff. Week 4 (RM) is the most startup-relevant. Week 5 (VLA)
is genuinely optional. All sim, all RunPod, no arm.
