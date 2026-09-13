# RPC system architecture

Three layers, three clocks. The split exists because human intent arrives on two
timescales two orders of magnitude apart, and one model cannot serve both — see
the control-rate ladder in [literature-brief.md](literature-brief.md).

```
  OPERATOR
    ├── language      0.2–1 Hz   "walk forward, stay low"
    │      └── CLIP text encoder → similarity over labelled clips
    │             → gait mix + speed prior + style vector          [Command]
    │
    └── stick/keys     30 Hz     analog velocity + turn + responsiveness
           └── deadzone, expo shaping, gait inferred from magnitude [Command]
                              │
                  ┌───────────┴────────────┐
                  │  ChannelArbiter        │  stick wins in 50 ms,
                  │  authority ∈ [0,1]     │  language resumes after 0.4 s
                  └───────────┬────────────┘  hold + 0.8 s release
                              ▼
  LAYER 1  ── kinematic generator ──────────────────────────  30 Hz, 1.9 ms
      TrajectoryPredictor: rolling ±1 s window in the character frame.
        past  = the character's own history, not negotiable
        future = blend(last prediction, operator intent), ramp = responsiveness
      PFNN / MANN: weights blended by gait phase (or a learned gate)
        in  249 = traj_pos 26 | traj_dir 26 | traj_gait 65 | joint_pos 66 | joint_vel 66
        out 294 = joint_pos 66 | joint_vel 66 | rot6d 132 | root_delta 3 | dphase 1
                  | future_pos 12 | future_dir 12 | contacts 2
                              │  reference motion stream (human skeleton)
                              ▼
  LAYER 2  ── retargeting ──────────────────────────────────  30 Hz, 1.9 ms
      Anchored damped-least-squares IK on MuJoCo body Jacobians.
      Per-limb scaling from an anchor (hip for a foot, shoulder for a hand),
      warm-started from the previous frame, then grounded on the support foot.
                              │  robot qpos + desired contacts + phase
                              ▼
  LAYER 3  ── tracking policy ──────────────────────────────  50 Hz
      PPO. Observes proprioception + the reference expressed in its own frame.
      Acts as joint-position offsets from the default pose.
      Rewarded for tracking; terminated for falling. When those conflict,
      not falling wins.
                              ▼
      MuJoCo position servos                                    500 Hz
```

## Division of labour (the thing not to lose sight of)

**The generator animates. The low-level policy stabilises.** They are not
alternatives and neither subsumes the other.

The generator decides *what the body does* — gait, phase, limb coordination,
style, how the motion changes when the operator moves the stick. It learns that
from mocap by supervised regression, which is why it needs no reward shaping, no
exploration and no physical understanding learned from scratch. That is the
whole argument against putting a VLA or an end-to-end policy in this slot: they
must rediscover what walking looks like, implicitly, through reward.

The low-level policy solves a deliberately narrow problem: realise a trajectory
that has already been specified, and do not fall doing it — reject pushes,
absorb terrain, avoid tripping, recover. It is not asked to invent motion.

When a pre-trained tracker (GEAR-SONIC) is used, it fills the **executor** role
only. SONIC also ships a kinematic planner (`planner_sonic.onnx`); that is the
slot the generator occupies, so we do not use it.

### How narrow "just stabilise" actually is

The boundary is a dial, not a fixed line, and where it sits depends on how
physically feasible the reference is:

| Generator trained on | What the low-level layer must do |
|---|---|
| Raw human mocap, retargeted at runtime | A great deal. Wrong mass distribution, wrong limb ratios, no toe joint — the reference is not merely unstable, it is unachievable, so the tracker must deviate from it in a coordinated way. This is what ExBody, BeyondMimic and SONIC are: heavy trackers. |
| Physics-corrected, robot-space motion | Very little. The reference is already dynamically admissible, so a stabiliser around it suffices. |

The architecture the project actually wants is the second row, and reaching it
is what the offline physics-correction stage is for. It is not a side quest —
it is the thing that makes "the low-level policy is just a stabiliser" true
rather than aspirational.

This also gives a diagnostic worth logging: **how far the tracker has to deviate
from the reference is a measure of the generator, not the tracker.** Persistent
deviation on a converged tracker means the generator is emitting motion the
robot cannot perform.

## Why each boundary is where it is

**Command, not goal.** The operator supplies a desired velocity, turn rate and
gait — quantities meaningful on every frame and holdable on a stick. Nothing in
the loop plans, re-solves, or waits for a goal to be reached. This is what makes
the analog channel close inside the ~50 ms perceptual window.

**Reference stream, not pose lookup.** Layer 1 emits dense reference motion and
layer 3 tracks it. The tracker never learns what "walk" means; it learns what a
G1 must do to follow a plausible human trajectory. That indifference is why the
same policy can be driven by a clip during training and a human during
deployment (`ClipReference` and `LiveReference` are interchangeable).

**Physics from frame zero.** The reference is what a *human* can do. Making it
true for a 33 kg robot with 29 actuators is a separate, physical problem, and it
is present from the start rather than added later. A generator tuned against a
kinematic viewer will produce references no robot can follow, and the mismatch
will not show up until it is expensive to fix.

## Load-bearing decisions, and what they cost

| Decision | Why | Cost |
|---|---|---|
| Velocity is derived from the pose difference, never fed back | The network predicts position and velocity independently; nothing forces them to agree, so feeding velocity back closes a loop with no restoring force. Measured: joint velocity mean 0.18 in training, 2.8 after 100 autoregressive steps, root speed saturating at the 6 m/s clamp. | One frame of pose history in the controller. |
| Noise on the autoregressive input blocks during training | Exposure bias. The model is trained on ground-truth history and rolled out on its own. σ = 0.2 in normalised units. | One-step MPJPE 68 → 96 mm; the rollout stops diverging. A trade worth making. |
| The character's own trajectory sample is pinned to the origin | Training data always has exactly zero there. Letting the shifted buffer keep last frame's *planned* position feeds the network 5 cm of drift it has never seen. | None. |
| Blend layer outputs, not weights | `W(p)x = Σ c_k(p)(W_k x)` is exact and avoids materialising a per-sample weight tensor — 522 MB at batch 1024. | K dense GEMMs instead of one batched. |
| IK targets anchored per limb, not to the root | LAFAN1 subjects have a hip half-width of 12% of leg length; the G1's is 8.5%. Root-scaled foot targets sit 27 cm outboard and saturate hip roll every frame. | An anchor pair per target. Foot error 11 cm → 2 cm. |
| Rolling floor for contact detection | walk3's actor goes to the ground for 20 s, which drags a global floor estimate 40 cm down and suppresses contact for the whole clip. | A sliding-window percentile. |
| Gait inferred from stick magnitude | The generator conditions strongly on gait: 1.2 m/s commanded yields 0.6 m/s under `stand`, 1.1 under `walk`, 4.0 under `run`. Exposing gait as a separate control makes the stick feel broken. | Gait is no longer independently addressable from the stick (language still sets it). |

## The arbitration contract

```
stick magnitude > 0.12   →  authority → 1 over 50 ms
stick released           →  hold 0.4 s, then authority → 0 over 0.8 s
output                   =  language.blend(stick, authority)
style                    =  always language's
every crossing of 0.5    →  logged as a handover
```

Asymmetric on purpose. Shared-autonomy arbitration that *infers* whose intent
should win (Dragan & Srinivasa 2013; Javdani et al. 2015) is right when the
human's goal is unknown and the robot's inference is good. Here the human is
driving a character in real time, and any moment the machine second-guesses the
stick is a moment the operator experiences as the robot fighting them. So
authority is asserted, not inferred — but released slowly, because snapping back
to autonomy the instant a thumb leaves the stick is equally unpleasant.

`ChannelArbiter.intervention_segments()` returns the windows where the operator
held authority. That is a free, continuously-generated label for "autonomy was
not good enough here" — the training signal Sirius-style intervention learning
uses, and which nobody collects for whole-body control (gap G5 in the brief).

## Where phase enters

`contact_phase` is the PFNN construction: phase advances π per heel strike, so
its derivative is the instantaneous gait frequency. `PeriodicAutoencoder` is the
DeepPhase construction: unsupervised periodic latents, no contact labels, no
gait assumption.

Phase then appears three more times downstream — as the index into the
generator's blended weights, as the `(sin, cos)` pair in the tracker's
observation, and implicitly as the desired contact schedule the tracker is
rewarded for matching. That is the concrete form of the claim in gap G6: the
animation phase and the robot's contact schedule are the same quantity, and this
code makes them the same variable.

## Scaling path

Everything here runs on one laptop GPU: 4 s per generator epoch, 21× realtime
per tracker env. The tracker is the bottleneck — PPO on 24 CPU envs is roughly
1k steps/s, and humanoid motion tracking wants 10⁸ steps.

The intended route is MJX: `assets/mujoco_menagerie/unitree_g1/g1_mjx.xml` and
`scene_mjx.xml` already ship in the vendored menagerie, and `TrackerEnv`'s
reward and observation are written as pure array operations for that reason.
Nothing in layers 1 and 2 changes.
