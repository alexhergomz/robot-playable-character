# RPC — Robot Playable Character

Treat a humanoid as a **playable character**, not an autonomous agent: a human
gives coarse intent in language and fine intent on a stick, and a physics-backed
controller animates the robot dynamically in real time. No clip library to pick
from, no goal to plan toward, no waiting on a VLA.

The premise, and what is and is not already in the literature, is in
**[docs/literature-brief.md](docs/literature-brief.md)**. The system layout and
the reasoning behind each boundary is in
**[docs/architecture.md](docs/architecture.md)**.

The generator **animates**; the low-level policy **stabilises**. Neither replaces
the other — see [Division of labour](docs/architecture.md#division-of-labour-the-thing-not-to-lose-sight-of).

```
language (0.2–1 Hz) ─┐
                     ├─ arbiter ─→ phase generator ─→ lookahead ─→ RL tracker ─→ MuJoCo
stick    (50 Hz) ────┘   50 Hz        50 Hz            10 frames     50 Hz        500 Hz
                       stick wins    1.1 ms (GPU)      200 ms        6.3 ms
```

The generator is **not** a PFNN. Experts are blended by a learned gate reading a
phase manifold that periodic autoencoders extract from motion without
supervision, rather than by a cyclic cubic over one scalar phase. PFNN is the
ancestor and remains in `rpc/generator/pfnn.py` as the baseline every comparison
runs against.

**The stick always supersedes language.** Authority transfers in 50 ms and
returns only after the operator has held neutral for 0.4 s. Every handover is
logged, which turns ordinary operation into labelled data about where autonomy
was insufficient.

## Status

| Component | State |
|---|---|
| BVH loader, FK, mirroring, resampling | works; validated against LAFAN1 (77 clips, 1.71 m subject, knee 79–173°) |
| Contact + phase extraction (PFNN and DeepPhase) | works; walking gives 0.61–0.65 stance per foot, 0.71 Hz stride |
| Control features, trajectory window | works; invariants tested |
| Phase-manifold generator (gated MoE) | **trained**; 3000-step rollouts, command gain 0.79–0.93 |
| Per-limb phase (4 groups) | trained; phase is limb-local, the network is not yet — see below |
| Command / arbiter / gamepad / CLIP language | works; 55 tests pass |
| Retargeting onto Unitree G1 | works; 2 cm mean foot error, 1.9 ms/frame |
| MuJoCo sim + tracking env | works; reward discriminates, 21× realtime/env |
| **GEAR-SONIC executor** | **working** — tracks a reference 38 s without falling, recovers from 1.0 m/s shoves, no training |
| PPO tracking policy | superseded by SONIC; kept as an ablation baseline |
| Deployment path (onboard, no reflash) | SONIC's deploy app ported to the G1 Orin (TRT 8.5); generator runs onboard; full chain validated in NVIDIA's own MuJoCo sim. **Not yet run on the robot** — see [`port/HANDOFF.md`](port/HANDOFF.md) |
| Real hardware | not attempted |

## Quick start

```bash
pip install -r requirements.txt

# vendored: Unitree G1 from MuJoCo Menagerie (sparse checkout, 56 MB)
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie.git assets/mujoco_menagerie
cd assets/mujoco_menagerie && git sparse-checkout set unitree_g1 && cd -

# LAFAN1 (144 MB) — the dataset the PFNN line is normally trained on
curl -L -o /tmp/lafan1.zip \
  https://github.com/ubisoft/ubisoft-laforge-animation-dataset/raw/master/lafan1/lafan1.zip
unzip -q /tmp/lafan1.zip -d data/raw/lafan1

# retarget onto the G1, then learn the phase manifold and the generator
python scripts/retarget_gmr.py --match walk run sprint --out data/processed/gmr/g1
python scripts/train_phase.py  --clips data/processed/gmr/g1 --out runs/pae
python scripts/prepare_lmp_data.py --src data/processed/gmr/g1 --pae runs/pae/pae.pt \
       --out data/processed/robot_lmp
python scripts/train_lmp.py --data data/processed/robot_lmp --no-gait --out runs/lmp

# drive it: stick -> generator -> lookahead -> tracker -> physics
python scripts/eval_rpc.py --lmp runs/lmp/generator.pt --device cuda --seconds 40 \
       --save runs/run.npz
python scripts/render_video.py --clip runs/run.npz --out runs/run.mp4
```

Per-limb phase instead of one global clock:

```bash
python scripts/train_phase_limb.py   --clips data/processed/gmr_all/g1 --out runs/pae_limb
python scripts/prepare_limb_data.py  --pae runs/pae_limb --out data/processed/robot_limb
python scripts/train_lmp.py --data data/processed/robot_limb --no-gait --out runs/lmp_limb
python scripts/eval_decouple.py --gen runs/lmp_limb/generator.pt
```

## Layout

```
rpc/motion/     quaternions, BVH, clips, control features, phase (contact + DeepPhase PAE)
rpc/generator/  PFNN, MANN, the shared blended-linear layer, the autoregressive runtime
rpc/control/    Command, TrajectoryPredictor, ChannelArbiter, gamepad/keyboard, CLIP
rpc/sim/        robot spec, anchored IK retargeting, MuJoCo wrapper
rpc/policy/     reference streams, tracking env, PPO
scripts/        prepare_data · train_generator · retarget_clips · train_tracker · demo_rpc
tests/          55 tests: motion invariants, interaction contract, physics
attic/          superseded controllers, kept with the measurements that retired them
```

## Interruptibility

The claim the architecture rests on: an operator can change their mind
mid-motion. Measured with all arms sharing the generator and the tracker, so the
only varied factor is when a changed command is allowed to take effect.

| manoeuvre | continuous | at a motion-phase boundary | after a committed segment |
|---|---|---|---|
| stop, 1.6 → 0 m/s | **640 ms** | 1200 ms | 2560 ms |
| dash, 0.4 → 1.6 m/s | **620 ms** | 680 ms | 2340 ms |
| turn (yaw rate) | **140 ms** | 380 ms | 1520 ms |

Settling time; reaction times are within 150 ms of each other everywhere, so the
advantage is not in starting sooner but in not having to finish what was
started. The committed arm's 1.5 s window is a chosen parameter and the ratios
scale with it; the boundary arm has none, its latency following from the
measured gait period, so that is the load-bearing comparison.

Two qualifications, both measured. Decomposing the latency, the **tracker** lags
400–500 ms on speed changes and 0 ms on turning, so most of the stop is physical
execution and no planner can beat it — but the tracker is common to every arm,
so the comparison stands. And the tracker has no interruption interface at all:
it sees a reference trajectory and nothing else, so a redirect reaches it only
as tracking error. Asked to follow a reference that steps by 0.3 rad on every
joint at once it never falls in 3/3 runs; it silently stops tracking instead.
The failure mode to guard is the robot quietly not doing what was asked, not the
robot falling.

## The executor: GEAR-SONIC

The low-level layer is NVIDIA's pre-trained G1 motion-tracking policy (ONNX,
Apache-2.0 code / NVIDIA Open Model License weights), used **only** as the
stabiliser. SONIC also ships a kinematic planner; that is the generator's slot,
so we do not use it.

```bash
python scripts/eval_sonic.py --probe                 # resolve file conventions
python scripts/eval_sonic.py --order hardware --quat xyzw --steps 1900 --push 1.0
python scripts/eval_sonic.py --our-clip data/processed/retargeted/g1/walk1_subject1.npz
```

| Reference | Survived | Joint error |
|---|---|---|
| SONIC's own sample walk, low-latency variant | **38 s / 38 s** | 0.123 rad |
| ...default (1 s lookahead) variant | **38 s / 38 s** | 0.115 rad |
| ...with 1.0 m/s shoves every 3 s | **20 s / 20 s** | 0.136 rad |
| ...with 1.5 m/s shoves | 3.9 s, fell | 0.192 rad |
| **our** retargeted LAFAN1 `walk1` | 5.9 s, fell | 0.222 rad |
| ...same, upper body pinned to nominal | 20.7 s, fell | **0.145 rad** |

The last two lines are the useful ones. SONIC is a known-good executor, so worse
tracking on our references is evidence about **our retargeting**, not about
SONIC — deviation-as-diagnostic, from [architecture.md](docs/architecture.md).

Chasing that signal found and fixed four real defects in the retargeter, each
verified by measurement against SONIC's own reference as a control:

| Defect | Symptom | After |
|---|---|---|
| Root path integrated from the **unscaled** human root delta | body travels human distances on 0.767x legs | root speed 0.78 -> 0.63 m/s |
| No **foot locking** — mocap skate passed straight through | stance-foot slip 63 cm/s vs SONIC's 0.4 | **median 0.00 cm/s** |
| Ground registration compared the human **ankle joint** to the robot **ankle body** | reference hovered 3-6 cm above the floor | support foot 3.26 cm vs SONIC's 2.89 |
| IK posture prior was the **straight-leg keyframe** | knee pinned at its extension limit 70% of frames | 0.1%; knee now bends 0.48 rad |

Pelvis pitch and roll, previously discarded entirely (pure-yaw base), are also
restored: 5.8 deg mean against the human's 5.5.

**This did not close the gap**, and the remaining error is now localised: pinning
the upper body to nominal takes survival from 5.9 s to 20.7 s and error from
0.222 to 0.145 rad. The legs are close to right; the torso and arms are not.
Prime suspect is the torso scale of 0.488 — the G1's pelvis-to-shoulder span is
0.309 m against the human's 0.633 m, so every degree of human spine motion is
amplified into the robot's 3-DoF waist.

Worth noting what SONIC's own reference looks like: near-zero foot slip *and*
only 3.1 deg of pelvis tilt against the human's 5.5. NVIDIA is not doing faithful
kinematic retargeting either — the directory is called `robot_filtered`. That is
direct evidence that an offline correction pass is required rather than optional.

Two file conventions could not be settled from source and were resolved by
probing all four combinations — the wrong ones fall within 0.3-1.8 s, so the
signal is unambiguous: SONIC's sample motions store `dof` in **hardware** joint
order with **xyzw** quaternions.

## PFNN directly in physics (no tracker, no handoff)

`rpc/motion/robot_dataset.py` + `rpc/generator/physics_runtime.py` put the
generator in the robot's own state space: one network between the stick and the
actuators, all inputs measured from the simulator, its predicted pose *is* the
position command. It runs at 4.5 ms/step against a 20 ms budget.

It does not walk, and the reason is not the network. Open-loop replay — feeding
reference joint angles straight to stiff position servos, no network at all —
isolates the reference from everything else:

| Reference (identical robot, gains, rate, harness) | Open-loop replay | Servo error |
|---|---|---|
| SONIC's `robot_filtered` motion | **9.0 s** | 0.006 rad |
| our kinematic retarget, `walk1` | 1.5 s | 0.011 rad |
| our kinematic retarget, `walk2` | 0.5 s | 0.005 rad |

The servos track the poses to a hundredth of a radian in every case, so the
robot is *executing* our reference faithfully and falling anyway. Our motion is
not physically realisable; a physics-corrected one is 6-19x more executable.

Stacking that against the feedback results gives the whole picture:

| | open loop | + feedback (SONIC) |
|---|---|---|
| physics-corrected reference | 9.0 s | 38 s, no fall |
| our kinematic reference | 0.5-1.5 s | 6-20 s |

**Reference quality dominates; feedback multiplies whatever it is given.** The
closed-loop PFNN manages 0.3-0.5 s — worse than open-loop replay — because
covariate shift is added on top: trained by regression on motion that never
falls, it has never seen a tipping state and has no corrective action to offer.

The conclusion is a constraint on the whole plan, not a bug to fix: **a
generator trained on motion that cannot be executed cannot produce motion that
can be.** Making the reference physically valid is step one, and it is a
prerequisite for the generator, not a later refinement.

One harness bug found on the way: SONIC's PD gains are soft *because SONIC
actively balances* — open loop they cannot hold a standing pose (103 steps),
while the stock gains hold indefinitely (500/500). Applying them to a
non-SONIC controller invalidated the first round of measurements.

## Why human motion is not automatically realisable on a humanoid

Physical validity is a property of the motion **and the body**.  Retargeting
transfers joint angles, which belong to the pose; it does not transfer dynamics.
The two bodies are not close: the G1's pelvis-to-shoulder span is 0.309 m
against the human's 0.633 m, legs scale by 0.767, there is no toe joint and the
mass distribution is unrelated.  Same joint angles on a different body gives a
different centre-of-mass trajectory, and balance conditions that held for the
human stop holding.

Measured with a body-independent criterion — where the centre of pressure would
have to be:

| Reference | ZMP outside support, median | p90 |
|---|---|---|
| SONIC `robot_filtered` | +1.2 cm | +8.6 cm |
| our kinematic retarget | **+18.9 cm** | +2881 cm |

Our reference demands a centre of pressure up to a metre and a half outside the
foot.  The poses are fine; the physics is not.

### The fix: recompute the root, keep everything else

`rpc/sim/dynamics_fix.py`.  The root is where the mass is, so it is the one part
that must be recomputed for the new body — foot placement, arm pose and style
all transfer.  Keep the footprints, solve for a CoM trajectory whose ZMP stays
over them (the cart-table model in batch form: one sparse least-squares solve,
no learning), move the root to realise it, re-plant the feet with leg IK.

```bash
python scripts/fix_dynamics.py     # ~1.2 s per 2500 frames
```

| | before | after | target |
|---|---|---|---|
| ZMP median, 18 clips | +18.9 cm | **+1.1 cm** | +1.2 cm (SONIC) |
| ZMP p90 | +2881 cm | +84.5 cm | +8.6 cm |
| footprints moved | — | 0.5-0.9 cm (walks) | — |

### ...which is necessary and not sufficient

The cart-table model fixes the *linear* term and explicitly neglects the
*angular* one.  Measuring the angular term directly (centroidal angular
momentum, no ablation) on **content-matched** windows — same speed, lower yaw
rate — the gap is real:

| | \|Ldot\| |
|---|---|
| SONIC, yaw 0.28 / 0.42 m/s | 5.65 |
| ours, yaw 0.13 / 0.38 m/s | **15.11** |

SONIC's scales sensibly with content (0.73 standing to 9.59 fast); our floor is
above its ceiling.  Six explanations were tested and refuted by measurement:
upper-body amplitude (a sweep makes it monotonically worse), IK spikes (the gap
is in the *median*), joint velocity (ours is lower), root angular velocity (the
same), root vertical acceleration (corrected to 0.56 m/s2 against SONIC's 0.57,
with no effect on Ldot), and clip content (matched above).

Per-body attribution then settles it without a hypothesis:

| body | mass | SONIC | ours | excess |
|---|---|---|---|---|
| **torso_link** | 7.82 kg | 1.74 | 14.14 | **+12.40** |
| shoulder links (x4) | 2.7 kg | 0.81 | 5.84 | +5.03 |
| knees (x2) | 3.9 kg | 2.72 | 4.26 | +1.55 |
| *total* | | 12.38 | 37.50 | +25.12 |

**Half the excess is one body**, and ~70% is the torso plus the links rigidly
attached to it — and torso_link is the heaviest body on the robot at 23% of its
mass.  The mechanism is the shoulder IK: two position targets (6 constraints) on
a 3-DoF waist, scaled by 0.488 and not lying at the robot's shoulder radius, so
the solve is over-determined and resolves differently frame to frame.

This is now the *third* independent line of evidence for the same defect — after
the SONIC tracking breakdown (waist error +0.320, pinning the upper body gave
3.5x survival) — but this time by attribution rather than inference.

Open-loop replay barely moved (77 -> 61 steps), because the cart-table model
fixes the *linear* term and explicitly neglects the *angular* one.  Isolating it:

**Known bug:** the correction misbehaves on running clips — flight phases have
no support polygon, the ZMP reference holds its last value, and the solver moves
the feet 1.5-2 m.  Walking clips are unaffected (feet move under 1 cm).  Needs a
flight-phase guard before the run/sprint clips are usable.

## Retargeting: GMR, not our own

**Our hand-built retargeter is superseded.** It accumulated seven distinct bugs,
all of the form *"a geometry-dependent quantity transferred with the wrong
ratio"* — root scale, foot locking, sole registration, knee prior,
over-determined waist, arm lateral scale, shoulder-width measured from a rest
pose with folded clavicles. Every one was a symptom of a per-chain scaling
scheme that should not have existed.

`scripts/retarget_gmr.py` uses **GMR** (Ze et al., ICRA 2026, MIT) instead:
task-space IK through `mink` with a QP solver, scaled from the actor's measured
height. Verified before adoption — same 29 joints in the same order, same qpos
layout, and joint limits *narrower* on all 15 that differ, so its output is
always valid in our model.

| same clips, same measurement code | ZMP p90 | \|Ldot\| | SONIC survival |
|---|---|---|---|
| ours (hand-tuned + ZMP correction) | 11.5-33.3 cm | 18.9-23.3 | 1 / 3 |
| **GMR** (no correction at all) | **0.20-6.46 cm** | **5.96-7.28** | **3 / 3** |
| SONIC's own `robot_filtered` reference | 8.60 cm | 5.98 | — |

Across all 18 clips: **12/18 track for the full 20 s** (ours: 3/18), mean
survival 863/1000 (463), mean joint error **0.132** against SONIC's own
reference at 0.12 — reference-quality parity. Walking flight fractions are
0.00-0.02 where ours gave 0.22-0.70.

The ZMP correction still earns its place on top of GMR — it rescues
`run2_subject4` from 328 to 1000 steps and is neutral elsewhere — so the
pipeline is GMR then `fix_dynamics.py`.

### A retracted result

An earlier version of this file scored clips on spine tilt and proposed dropping
11 of 18 as "not walking". That triage predicts nothing: its correlation with
trackability is **+0.049**, the clips it would keep average *worse* survival
(474 vs 663), and four of the five clips that survived a full 20 s were ones it
marked for deletion. It has been removed. Two still frames were enough to make
an invented threshold look convincing; motion has to be judged as motion, which
is what `scripts/render_video.py` is for.

## Testing the fixes

Three interventions, measured on quantities with dynamic range (open-loop replay
is a threshold test on an unstable equilibrium and stays floored regardless).

**1. Torso orientation target — works.**  Two shoulder *position* targets are six
constraints on a 3-DoF waist; one torso *orientation* target is three on three
and is satisfiable exactly.

| | before | after |
|---|---|---|
| torso_link \|Ldot\| | 17.94 | **8.19** |
| all bodies, summed | 87.25 | 68.18 (-22%) |
| ZMP p90, before correction, 18 clips | +2881 cm | **+635 cm** |
| ZMP p90, after correction | +84.5 cm | **+34.0 cm** |
| matched-window \|Ldot\| | 15.11 | **9.43** (SONIC 5.65) |

The torso is no longer the top contributor; the knees are.

**2. Damping the trunk (`--ori-gain 0.3`) — works, smaller effect.**  Copying
human trunk excursion one-for-one is faithful and wrong: the G1 carries 23% of
its mass in the torso, and a physics-corrected reference shows roughly 0.3x the
human's waist range.  Torso \|Ldot\| 9.97 -> 8.19.

**3. Froude / dynamic-similarity time scaling — refuted.**  Dimensional analysis
says the G1 should step 14% faster than the human (time scales by
sqrt(0.764) = 0.874).  Measured, it makes things worse: \|Ldot\| 40.05 -> 52.44,
exactly the 1.144^2 = 1.31 that playing a trajectory faster implies.  Dynamic
similarity requires *geometric* similarity including mass distribution, and a
7.82 kg torso on light limbs is not a scaled human.  The scaling argument is
sound; this body pair does not satisfy its precondition.

### End to end

Using SONIC purely as a diagnostic — a known-good executor, so survival measures
the *reference*:

| | before today | after |
|---|---|---|
| clips tracking 20 s without falling | **0 / 18** | **5 / 18** |
| mean survival | ~300/1000 steps | **590 / 1000** |
| mean joint error | 0.24-0.27 rad | **0.194 rad** |

All five survivors are walks.  Every run and sprint still fails, which is the
known flight-phase bug in the dynamics correction — no support polygon, so the
ZMP reference holds its last value and the solver drags the feet metres.

## Measured

On an RTX 4050 laptop, MuJoCo on CPU:

| | |
|---|---|
| Generator inference, batch 1 | 0.44 ms GPU / 0.59 ms CPU |
| Full generator step incl. trajectory blending | 1.9 ms |
| Retarget (20 IK iterations) | 1.9 ms |
| Frame budget at 30 Hz | 33.3 ms |
| Generator training | 4 s/epoch, 813k samples, 2.17 M params |
| Retarget foot error (18 locomotion clips) | 5.8 cm mean, 2.0 cm on clean walks |
| Tracker env | 1066 steps/s, 21× realtime |

One-step validation on held-out subject 5: MPJPE 96 mm, root-velocity error
0.11 m/s, yaw-rate error 0.53 rad/s.

## Limitations

Read these before trusting anything above.

- **The tracker is not converged.** The env, reward, PPO and reference plumbing
  all work, and the reward separates good behaviour from bad (2.23 mean reward
  standing on-reference; random actions terminate in 49 steps). A run at 780
  steps/s on 24 CPU envs reaches mean return ~30 and episode length ~45 of 300
  after 80k steps — learning, but nowhere near following a walk cycle. Humanoid
  motion tracking wants ~10⁸ steps; this is ~10⁵. Until it converges,
  `demo_rpc.py` runs **kinematically** — it shows what the operator asked for,
  not what a robot would do. The scaling path is MJX; the `_mjx` G1 models are
  already vendored, and `TrackerEnv`'s reward and observation are written as
  pure array operations for that port.
- **Command gain is calibrated only in the walking range.** 0.8 m/s commanded
  gives 0.76, 1.2 gives 1.12; above ~1.8 m/s the inferred gait mix tips to `run`
  and the generator overshoots (2.5 → 3.9 m/s). Response is monotone
  throughout. The handles are `gait_from_speed`'s `sharpness` and `GAIT_SPEED`.
- **Yaw-rate error is high** (0.53 rad/s one-step). Turning is the weakest
  channel; a turn command of 1.0 rad/s achieves ~0.74.
- **Contact labels are heuristic.** Walking gives sensible statistics, but the
  detector still reports ~20% flight during walking, which should be near zero.
  LAFAN1 has real foot sliding, so the speed threshold is deliberately loose.
- **The PAE is implemented but unused.** `PeriodicAutoencoder` trains and
  produces phase manifolds; the generator currently uses contact phase. Wiring
  DeepPhase in as the phase source is the obvious next ablation, and
  `FeatureSpec.n_phase_channels` already reserves the input block.
- **Language is coarse and cheaply indexed.** CLIP over clip-name labels. The
  retrievals are sensible ("start running" → `run1`), but the labels are
  filename-derived; captioning with a VLM is the intended upgrade, and
  `library_from_features` is the one function to change.
- **The retargeter ignores orientation.** It matches body *positions* only, so
  foot roll/pitch and hand orientation are unconstrained. This is likely to
  matter for contact quality once the tracker runs.
- **No hardware, no sim-to-real.** No domain randomisation beyond optional
  pushes, no actuator modelling, no latency modelling.

## What would make this a contribution

From the brief's gap list, the three this codebase is positioned to attack:

- **G1 dual-channel arbitration** — implemented and testable here; nothing in
  the literature fuses a slow semantic channel with a fast analog one under a
  stated contract.
- **G3 authored responsiveness** — `Command.responsiveness` is a runtime knob
  that shapes how fast the near future commits to intent, which is an animator's
  parameter that robotics currently leaves to reward weights.
- **G6 phase as a shared coordinate** — the same phase variable indexes the
  generator's weights, enters the tracker's observation, and defines the contact
  reward.

The measurement gap (**G2**, no metric for feel) is the one this cannot close
alone: it needs human operators, and `ScriptedInput` exists so the same protocol
can be run against a deterministic baseline first.
