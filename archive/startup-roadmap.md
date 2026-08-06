# Startup Roadmap — Manipulation-First Robotics

> The durable *why/what*. For the time-boxed *when/do*, see [five-weeks-roadmap.md](five-weeks-roadmap.md).

## North star

Become well-positioned to **co-found a manipulation-first robotics startup** — narrow, high-value,
repetitive dexterous tasks, in the spirit of Dyna Robotics / Physical Intelligence.

- **Aspirational target task:** napkin folding (deformable, contact-rich).
- **Real near-term goal:** get an arm good at *general* manipulation — starting with the simple stuff
  (pick-and-place) and building up. Napkin folding is a direction, **not a hard bound**.
- **Sequencing belief:** all the ML learning tracks are the "brain." They're necessary but they are
  **not the moat.** The moat is data, reliability, and task selection (see below).

## Mental model: brain vs body

The list of learning tracks is a complete curriculum for the **brain** of the robot — and only the brain.
"An arm that's good at manipulation" also needs a **body** stack that the tracks barely touch.

```- 
BRAIN (the learning tracks — my current focus)
  cnn · diffusion · llm · rl · Diffusion Policy · ACT · VLA · reward model · world model

BODY (the robotics/embodiment stack — mostly deferred to arm-day)
  inverse kinematics · compliant/force control · 3D perception · simulation ·
  teleop / data collection · tactile sensing · deformable-object handling
```

The single biggest body-side gap is **compliant / force control** — you can't fold a napkin (or handle
anything softly) with a stiff position controller. That, plus **teleop/data-collection**, is where the
actual company lives, and it's the part that's hard to learn from notebooks. Budget for a light robotics
track alongside the policy work once the arm is in hand.

## Why the architecture isn't the moat

The napkin-folding demos that make companies notable are the same ACT / Diffusion Policy / VLA stack below.
What makes them *work* is three things self-learners under-invest in:

1. **Data-collection infrastructure** — teleoperate cheaply, harvest hundreds of clean demos per task.
   The policy is nearly a commodity; the data pipeline is the product.
2. **Reliability engineering** — 6/10 is a demo, 99/100 on a narrow task is a business. Closing that gap
   is ~90% of the work and almost none of the papers.
3. **Task selection** — pick tasks that are valuable, repetitive, and narrow enough to actually nail.

Building the brain from scratch still matters: when a policy fails at 3am, I'll know *why*. That depth is
the co-founder asset. But keep an eye on the last mile.

## Track → test-bed map

Every track earns a "watch it do the thing" payoff:

| track | what it is (plain) | sim test bed | the "it works!" moment |
|---|---|---|---|
| cnn | see | MNIST | predict digits |
| diffusion | generate images | image dataset | sample new images |
| llm | understand/generate language | text corpus | generate better text |
| rl | learn by trial and error | games | solves a game by hit-and-trial |
| **Diffusion Policy** | the robot's hands (do the task) | **Push-T** (`gym-pusht`) | push a T-block into a target outline |
| **ACT** | hands, chunked & precise | **ALOHA transfer-cube** (`gym-aloha`) | two arms pick a cube and hand it off |
| **VLA** | hands + ears (do what you *ask*) | **LIBERO** (or SIMPLER) | change the sentence → behavior changes |
| reward model | the coach (grades progress) | video rollouts | predict a 0→1 progress curve |
| world model | the imagination (predict futures) | game video | control left/right inside an imagined render |

## LeRobot: the sim = real sandbox

`gym-pusht`, `gym-aloha`, `gym-xarm` — plus datasets and reference Diffusion-Policy/ACT/VLA
implementations — all ship inside **LeRobot** (HuggingFace). LeRobot is *also* the stack that drives the
cheap real arms (SO-100 / SO-101 — almost certainly what I'll buy).

```- 
   SIM                                    REAL
   gym-pusht / gym-aloha  ── same code ──► SO-100 arm
   LeRobot policy (DP/ACT/VLA)             LeRobot policy (DP/ACT/VLA)
   record demos in sim    ── same format ─► teleoperate & record real demos
```

So sim isn't a stepping stone I abandon on arm-day — it's the *same dev loop*, swapping the env backend.
The whole policy pipeline, data recording, and eval harness can be built in sim now, and on arm-day I
change an env config and re-record demos on real hardware. Nothing wasted.

**Caveat:** Push-T and transfer-cube are *rigid*. Napkin folding is *deformable*, and no cheap sim does
cloth well yet. Sim takes me all the way to "working, understood DP/ACT/VLA pipeline"; the deformable-
specific learning genuinely waits for the real arm + a real napkin — which is the right place to hit it.

## Reward model + world model as later on-arm tools

These two are capstones, deferred — but they plug directly into improving the arm. Reliability ranking:

**1. RM → failure-triggered data collection (reliable now, do first).** The RM flags rollouts that
plateaued / never reached done; a human confirms; collect more demos on those cases; retrain. A **data
flywheel** with a human gate, so the RM's errors can't corrupt the policy. This is how manipulation
companies actually run.

**2. RM → cycle-time observability (reliable now, near-free after #1).** Progress model over rollouts →
cycle-time per task, min/max/variance, throughput, stuck-episode rate. In manipulation, **cycle time is
the unit economics** ("folds in 8s vs 14s"). Read-only monitoring, so low-risk.

**3. WM → planning (real, but least reliable today).** The `world_model` repo (below) is the seed. Named
methods: **DreamerV3** (train policy inside imagined rollouts), **TD-MPC2** (plan short-horizon in a latent
model — *the* reliable-in-sim option), **visual MPC / UniPi** (predict future video, score, execute — the
frontier, drifts on long horizons). Cloth makes the world model itself hard, so treat WM planning as the
research-y capstone.

**RM + WM combine into one loop** (this is TD-MPC2 / MuZero):

```- 
   world model: imagine candidate futures  (what if I do action-seq A?)
        │ imagined rollouts
        ▼
   reward model: score each imagined future (which reaches done fastest?)
        │ pick best action sequence
        ▼
   execute on arm ──► repeat
```

Because WM-planning needs the RM to be useful, **build the RM first** (useful standalone) and the WM later.

### The `world_model` repo

`/home/rahul/Documents/codes/ml/world_model` (branch `learn`) — **"Genie 3 Reconstruction."** A three-stage
video world model trained on game footage, **no action labels required**:

```- 
Raw frames
  └► Video Tokenizer     (stage 1: VQ-VAE + FSQ)      frames → discrete patch tokens
       └► Latent Action Model (stage 2: FSQ)          frame pairs → discrete action tokens
            └► Dynamics Model  (stage 3: MaskGIT ST-Transformer)  context + actions → next-frame tokens
                 └► predicted frames (via tokenizer decoder)
```

Role in this roadmap: **parked past the 5-week sprint.** It's the eventual "imagination" engine for the
planning loop above. Learning goal is to understand this repo cold, then later adapt the dynamics model as
a planner. Finishing the diffusion track unblocks the generative machinery it relies on.

## "Ready to buy the arm" checklist

- [ ] Diffusion Policy trains and succeeds on Push-T in sim (understood, not copied).
- [ ] ACT trains and succeeds on ALOHA transfer-cube (action chunking understood).
- [ ] Comfortable end-to-end with LeRobot: record demos, train a policy, run eval — all in sim.
- [ ] A reward model can score my own DP/ACT rollouts and flag failures (data-flywheel loop demonstrated).
- [ ] (stretch) VLA run/understood on LIBERO.
- [ ] Clear on which arm to buy (SO-100/101) and why the sim work transfers 1:1.
