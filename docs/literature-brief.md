# Robot Playable Characters — literature brief

*Preliminary scan, 2026-09-05. Scope: PFNN lineage → real hardware. 58 works, 6 tracks.
Purpose: position a thesis, not survey a field.*

---

## The read: the architecture is converged, the interaction contract is not

**Four independent groups have shipped the RPC stack without naming it.**

HOVER, the Masked Humanoid Controller, BeyondMimic and SONIC all do the same structural
thing: a fast, low-parameter whole-body policy that consumes a *partially specified,
streaming reference* from something upstream, where "upstream" can be a joystick, a mocap
stream, a retargeted video, or a language-conditioned generator. Disney's bipedal character
went further and put an *animation engine* upstream of the RL policy, with a human on a
remote for live show performance. That is the thesis, built, on hardware, in 2024.

So the contribution cannot be "use an animation-style controller instead of a VLA" — that
argument is already won in practice, quietly, by people who never framed it as a critique
of autonomy. The contribution has to be the part everyone skipped: **the interaction
contract.** Nobody has specified what the operator's two channels mean, how they arbitrate
when they disagree, what latency each is allowed, or how to measure whether the resulting
thing feels good to drive. Games solved all four, forty years of craft deep, and robotics
has imported none of it.

The second contribution is methodological and genuinely untouched: game animation's
*production* apparatus — motion databases with hand-designed trajectory-matching features,
blend trees and layering, responsiveness tuning as an explicit authored parameter,
animator-in-the-loop iteration. Robotics imports the datasets and the architectures. It has
never imported the pipeline or the craft vocabulary.

### Already load-bearing — do not claim as novel

| | |
|---|---|
| **Streaming kinematic reference + RL tracker** | The dominant humanoid architecture since ExBody. "Animation engine drives a controller" is this. |
| **Multi-modal command spaces** | HOVER distills joystick, root velocity, joint targets and mocap into one policy. MHC does masked partial specification on a real Digit. |
| **Language as the coarse channel** | PADL → SuperPADL in sim; TextOp streams editable text instructions to a real humanoid. |
| **Test-time steering without retraining** | BeyondMimic steers a diffusion policy with cost functions; BFM-Zero and Motivo prompt one latent space with rewards, goals, or clips. |
| **Phase as an inductive bias for legs** | CPG-RL and periodic reward composition got there from biomechanics, independently of PFNN. |

### Structurally absent — this is where a thesis lives

| | |
|---|---|
| **Two channels at once** | Every system above takes language *or* stick. None fuses a slow semantic channel with a fast analog one and defines who wins. |
| **Feel as a measurable quantity** | Robotics has tracking error. Games have responsiveness, controllability, predictability — no robotics paper reports any of them. |
| **Authored responsiveness** | In games the naturalness/latency trade is a knob an animator turns per-action. On robots it is an accident of the reward. |
| **Runtime nearest-neighbour retrieval** | Motion matching is the industry default and has, as far as this scan found, never been the live generator on a real humanoid. |
| **Operator-in-the-loop improvement** | Sirius-style intervention learning exists for tabletop manipulation, not for whole-body character control. |

---

## The control-rate ladder — why the framing has teeth

The strongest technical form of the argument is not philosophical, it is a timing argument.
Human intent arrives on two clocks two orders of magnitude apart, and one autoregressive
model cannot serve both.

```
1 ms        10 ms       100 ms        1 s         10 s
|-----------|-----------|-----------|-----------|
[==]                                              joint torque / PD loop     500 Hz   1–2 ms
            [=]                                   RL whole-body policy        50 Hz    20 ms
            [==]                                  game animation frame     60→30 fps 16.7–33 ms
           [~~~~~]                                PERCEPTIBLE INPUT LAG   15 ms detect / 50 ms budget
                        [==]                      motion-matching search      10 Hz   100 ms
                        [=====]                   VLA action chunk          2–10 Hz 100–500 ms
                                    [=====]       language instruction     0.2–1 Hz    1–5 s
```

Expert players resolve latency down to ~15 ms; games target under 50 ms end-to-end. A VLA
action chunk lands entirely outside that window — which is exactly why Helix, GR00T and
Hi-Robot split into System-1 / System-2.

**The claim is stronger than "VLAs are slow": the analog channel must close inside the
perceptual window, so it must bypass the semantic model entirely, not be distilled from it.**

---

## Track 1 — The PFNN lineage (kinematic neural character control)

The direct answer to "what happened to PFNN". The line runs through phase generalisation
(local, then learned) and terminates in retrieval-based methods the games industry actually
shipped. Worth knowing: the field partly walked *away* from phase-indexed weights toward
learned phase manifolds and codebooks — so "repurpose PFNN" should probably become
"repurpose the phase manifold and the trajectory-feature interface".

| Venue | Work | Why it matters |
|---|---|---|
| SIGGRAPH '17, TOG 36(4) | **Phase-Functioned Neural Networks for Character Control** — Holden, Komura, Saito | The origin. Network weights are a cyclic function of gait phase; input is user control + previous state + terrain geometry. Tiny, real-time, terrain-adaptive. The control interface — a desired *future trajectory*, not a goal — is the part that ages best. |
| SIGGRAPH '18, TOG 37(4) | **Mode-Adaptive Neural Networks for Quadruped Motion Control** — Zhang, Starke, Komura, Saito | Phase → gating. Replaces hand-defined phase with a learned gating network blending expert weights. The moment the lineage stops needing a labelled gait cycle, and the moment it becomes a mixture-of-experts. |
| SIGGRAPH A '19, TOG 38(6) | **Neural State Machine for Character-Scene Interactions** — Starke, Zhang, Komura, Saito | Goal-directed contact: sitting, carrying, opening doors from one model. Closest animation-side analogue to loco-manipulation; solves the goal/phase alignment problem robots hit at every contact event. |
| SIGGRAPH '20, TOG 39(4) | **Local Motion Phases for Learning Multi-Contact Character Movements** — Starke, Zhao, Komura, Zaman | Per-limb phase. Kills the single global phase — each end-effector carries its own contact phase. A humanoid's contact schedule *is* a set of local phases; cleanest published bridge between animation phase and robot contact timing. |
| SIGGRAPH '21, TOG 40(4):92 | **Neural Animation Layering for Synthesizing Martial Arts Movements** — Starke, Zhao, Zinno, Komura | Layered, additively composed controllers — the neural blend tree. The mechanism for "language sets the layer, stick modulates it". |
| SIGGRAPH '22, TOG 41(4) | **DeepPhase: Periodic Autoencoders for Learning Motion Phase Manifolds** — Starke, Mason, Komura | **The successor to cite.** Unsupervised multi-channel phase extraction (frequency, amplitude, offset, shift) from unstructured mocap. Distances in the phase manifold beat pose-space distances for alignment and retrieval. |
| SIGGRAPH '20, TOG 39(4) | **Learned Motion Matching** — Holden, Kanoun, Perepichka, Popa (Ubisoft La Forge) | The industrial default. Three small nets replace projection / stepping / decompression — neural scalability with motion matching's predictability. Read with Clavet's GDC 2016 *For Honor* talk for the trajectory-feature design, which is the real transferable artefact. |
| SIGGRAPH A '19, TOG 38(6) | **DReCon: Data-Driven Responsive Control of Physics-Based Characters** — Bergamin, Clavet, Holden, Forbes | **The hinge paper.** Motion matching produces a kinematic target; RL tracks it in physics. That is the ExBody / HOVER / SONIC architecture, published in graphics five years earlier. |
| SIGGRAPH A '25, arXiv:2510.22632 | **Environment-aware Motion Matching** — Ponton, Andrews, Andujar, Pelechano | Runtime search penalises collisions with a dynamic scene, adjusting pose *and* trajectory together. Kinematic version of the reactive layer an RPC needs between operator intent and world. |
| SIGGRAPH '20, TOG 39(4) | **Character Controllers Using Motion VAEs** — Ling, Zinno, Cheng, van de Panne | Action space over a learned motion manifold, driven by RL. Ancestor of every "prompt the latent space" humanoid paper in Track 3. |
| SIGGRAPH '24 *(unverified)* | **Categorical Codebook Matching for Embodied Character Controllers** — Starke et al. | Retrieval as classification; discrete codebooks in place of continuous regression. **Confirm exact title/venue before citing.** |

## Track 2 — Physics-based character control (the bridge)

Already *playable* controllers with latent or linguistic steering; they just live in
simulation with a free actuator budget. Almost every real-humanoid paper in Track 3 is one
of these with a sim-to-real story bolted on.

| Venue | Work | Why it matters |
|---|---|---|
| SIGGRAPH '18 | **DeepMimic** — Peng, Abbeel, Levine, van de Panne | Reference tracking as reward. The primitive under every humanoid motion-tracking policy since. |
| SIGGRAPH '21 | **AMP: Adversarial Motion Priors** — Peng, Ma, Abbeel, Levine, Kanazawa | Style without a clip: a discriminator supplies "looks like the dataset" so task reward can be sparse. How you keep character in a robot improvising under operator input rather than replaying. |
| SIGGRAPH '22 / '23 | **ASE & CALM** — Peng et al.; Tessler et al. | A latent sphere of skills a high-level controller — or a human — indexes into. The most direct sim-side answer to "what does the right analog stick actually move". |
| SIGGRAPH A '22, arXiv:2301.13868 | **PADL: Language-Directed Physics-Based Character Control** — Juravsky, Guo, Fidler, Peng | Explicitly separates high-level task commands from low-level skill commands in one language interface — the coarse/fine distinction, minus the analog channel. |
| SIGGRAPH '24 | **SuperPADL** — Juravsky et al. | 5,000+ skills, real time on a consumer GPU, natural transitions so a user composes multi-stage animation live. Closest thing in print to an operator directing a character through language in real time. |
| SIGGRAPH A '24, TOG 43(6) | **MaskedMimic: Unified Physics-Based Character Control Through Masked Motion Inpainting** — Tessler et al. | **Partial specification as the interface.** Train on masked motion; at test time any subset of constraints — a hand target, a root path, a text prompt — is a valid command. Cleanest formal statement of the RPC command abstraction anywhere. |
| ICCV '23 / ICLR '24 | **Perpetual Humanoid Control & PULSE** — Luo et al. | PHC tracks indefinitely with failure recovery; PULSE distils it into a reusable latent motion representation. The robustness half of what an operator-driven character needs. |

## Track 3 — Real humanoids (where the thesis is already half-built)

Read defensively: this is where a reviewer says "this exists". The job is to show these
systems all pick *one* input channel and none treats the operator as a designed part of the loop.

| Venue | Work | Why it matters |
|---|---|---|
| RSS '24, arXiv:2402.16796 | **ExBody: Expressive Whole-Body Control** — Cheng, Ji, Yang, Pathak, Wang | Split the body: upper imitates expressive mocap, lower just stays up and follows a joystick root command. ExBody2 (arXiv:2412.13196) adds data curation and a stronger tracker. |
| CoRL '24, arXiv:2406.08858 | **OmniH2O** — He et al. | Full teleop baseline: continuous human keypoint retargeting. The "fully manual" end of the autonomy slider — position against this as much as against VLAs. |
| ICRA '25, arXiv:2410.21229 | **HOVER: Versatile Neural Whole-Body Controller** — He et al. | The multi-mode argument, already made. Distils navigation, loco-manipulation and tabletop modes into one 1.5M-param policy over a unified command space, explicitly designed for joysticks, keyboards, mocap, exoskeletons, VR. Their motivation section is your motivation section. |
| arXiv:2408.07295 | **Masked Humanoid Controller** — Dugar, Shrestha, Yu, van Marum, Fern | **Closest published relative.** Masked target trajectories over arbitrary subsets of state, accepting optimised trajectories, mocap, retargeted video *and* live joystick through one interface — on a real Digit V3. The baseline. |
| arXiv:2508.08241 | **BeyondMimic** — Li et al. (Berkeley Hybrid Robotics) | Test-time steering: classifier guidance turns novel cost functions into behaviour — inpainting, joystick teleop, obstacle avoidance — with no retraining, on hardware. Strongest existing mechanism for "adapts in real time". |
| arXiv:2511.07820 | **SONIC** — NVIDIA GEAR | Behaviour foundation model: scale motion tracking until one policy is a generalist motor substrate; shared latent across robot/human/hybrid motion; low-latency Jetson inference. Likely default lower layer of any RPC built in 2026. |
| arXiv:2504.11054 / 2511.04131 | **Meta Motivo & BFM-Zero** | Forward-backward representations embed motions, goals and rewards in one space, so a single policy is promptable zero-shot. Principled candidate for what the coarse channel writes into. |
| arXiv:2602.07439 *(unverified)* | **TextOp** — TeleHuman | Your coarse channel, alone: streaming language with on-the-fly edits, autoregressive motion diffusion upstairs, tracking policy downstairs. Their motivation is verbatim your gap statement. |
| CoRL '22, arXiv:2212.03238 | **Walk These Ways: Multiplicity of Behavior** — Margolis, Agrawal | **The gamepad paper.** One policy exposing continuous behaviour parameters — footswing height, stance width, gait frequency, posture — modulated live by an operator. A character-controller tuning panel on a quadruped. Under-cited in humanoid work. |
| ICRA '21 | **Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition** — Siekmann et al. | Phase from the robotics side: gaits as periodic contact schedules in the reward. Pair with PFNN — two fields discovering phase is the right coordinate, with no contact between them. |
| arXiv:2407.11562 | **RobotKeyframing** — ETH / Disney Research | Mixes dense and sparse rewards so an operator specifies a variable number of keyframes rather than a dense trajectory. Keyframing is the animator's native verb — clearest case of vocabulary transferring. |

## Track 4 — The Disney precedent (read this first)

The most complete existing instantiation of "robot as playable character", from a team whose
institutional purpose is character performance rather than autonomy.

| Venue | Work | Why it matters |
|---|---|---|
| RSS '24, arXiv:2501.05204 | **Design and Control of a Bipedal Robotic Character** — Grandia, Knoop, Hopkins, Wiedebach, Bishop, Pickles, Müller, Bächer | **Animation engine → RL policy → live operator.** At runtime an animation engine composes and blends multiple animation sources into command signals; an RL controller executes them robustly; an operator interface drives live shows. Ask the question they didn't: *what should that operator interface be, and how do we know when it's good?* |
| SIGGRAPH '24 | **Interactive Design of Stylized Walking Gaits for Robotic Characters** — Disney Research | Authoring, not learning: an artist edits a stylised gait in an interactive tool, a model-based stack keeps it feasible in real time. The production-pipeline half, which robotics almost never writes down. |

## Track 5 — Shared autonomy (the vocabulary you're missing)

The field that spent thirty years on exactly this question — how much does the human say,
how much does the machine infer, who wins on conflict — and is almost entirely absent from
the humanoid-character literature.

| Venue | Work | Why it matters |
|---|---|---|
| RSS '20 (ILIAD) | **Shared Autonomy with Learned Latent Actions** — Jeon, Losey, Sadigh | The core mechanism: embed high-dimensional robot behaviour into a low-dimensional latent the human drives with a 2-DoF joystick. The formal answer to "what should the stick be connected to". Extended version (Auton. Robots 2022, arXiv:2107.02907) adds required properties and a personalised alignment model. |
| RSS '15 / RSS '18 | **Shared Autonomy via Hindsight Optimization**; **via Deep RL** — Javdani, Srinivasa, Bagnell; Reddy, Dragan, Levine | Arbitration formalised: assist under uncertainty about the human's goal without requiring it be declared. Pair with Dragan & Srinivasa's policy-blending formalism (IJRR 2013). |
| RSS '23 | **Sirius: Robot Learning on the Job** — Liu, Zhu et al. | Interventions as training signal during deployment. Nothing like it for whole-body control — and an RPC generates this data continuously by construction. |
| ICRA '19, arXiv:1810.02890 | **HG-DAgger** — Kelly et al. | Human-gated handover. With IWR (Mandlekar et al.) and the 2025 Frontiers interactive-IL survey, the canonical cluster for "the operator decides when to take over". |
| HRI '23, arXiv:2301.02555 | **"No, to the Right" — Online Language Corrections via Shared Autonomy** — Cui, Karamcheti et al. | Language as a *correction* channel mid-execution, not instruction. The missing middle between coarse and fine. |
| JFR '17 | **Director: A UI Designed for Robot Operation with Shared Autonomy** — Marion et al. (MIT DRC) | Prior art on the interface itself. Read with NASA-JSC Affordance Templates and the DRC traded-control literature (Sheridan 1992 underneath all of it). |
| RSS '06 / Sensors '20 | **Sliding / adjustable autonomy** — Sellner et al.; modern HRI variable-autonomy line | Manual → shared → supervisory → autonomous as an explicit, run-time-adjustable axis. "Semi-autonomous" is a point on this axis; say which one, in their terms. |

## Track 6 — Adjacent fields (the unexpected leverage)

Two of these — phase-variable prosthetics and game-feel research — are, as far as this scan
can tell, uncited in humanoid character control, and both are load-bearing.

| Venue | Work | Why it matters |
|---|---|---|
| T-RO / IEEE Access (Gregg lab, U. Michigan) | **Phase-variable control of powered knee-ankle prostheses** — Quintero, Villarreal, Rezazadeh, Gregg | **A third independent discovery of phase.** A continuous mechanical phase variable, estimated live from thigh motion, unifies control across the gait cycle and across tasks — replacing discrete activity classification with a continuous coordinate. Same idea as PFNN, different field, plus something graphics lacks: real-time phase estimation from a human's own body. |
| Neural Networks '08 / RA-L '22 | **Central pattern generators**; **CPG-RL** — Ijspeert; Bellegarda & Ijspeert | Phase from biology: coupled oscillators whose setpoints an RL policy modulates. The pre-existing robotics home for phase — cite it so the PFNN import reads as convergence, not novelty-by-transplant. |
| UPenn tech report / AUC 2018 | **Trade-offs between Responsiveness and Naturalness for Player Characters**, and the game-feel literature | **The metrics robotics doesn't have.** Perceived vs. real latency, responsiveness as an authored parameter, naturalness/snappiness as an explicit design axis. Nobody has run this evaluation on a robot — a self-contained, publishable user study. |
| ICML '24, arXiv:2402.15391 | **Genie: Generative Interactive Environments** — Bruce et al., DeepMind | Latent actions learned unsupervised from unlabelled video; the model infers *which parts of the world are controllable*. With Menapace et al.'s Playable Video Generation (CVPR 2021): "what should the controller be bound to", answered without a designer. |
| GDC '16 (Ubisoft) | **Motion Matching and The Road to Next-Gen Animation** — Simon Clavet | Craft, not research, and worth citing anyway: the feature-design reasoning behind what a controller matches on — future trajectory positions, facing directions, foot velocities. A better command interface for a humanoid than anything in the robotics literature, designed by animators for feel. |

---

## Six gaps this scan could not close

**G1 — Dual-channel arbitration.** No system fuses a slow semantic channel and a fast analog
channel with a defined arbitration rule. HOVER switches between modes; TextOp takes language
only; BeyondMimic takes a joystick only. What happens when the operator says "walk to the
table" and simultaneously pushes the stick left? Shared autonomy has the formal machinery
(policy blending, hindsight optimisation) that character control never applied.
*Nearest: HOVER (mode switching) · MaskedMimic (simultaneous heterogeneous constraints, sim only) · Dragan & Srinivasa 2013.*

**G2 — No metric for feel.** Humanoid papers report tracking error and success rate. Neither
measures whether a human can drive the thing. A benchmark — operator task completion under
varying assistance, with latency and naturalness as manipulated variables — does not exist
and would be cited by everyone building teleop.
*Nearest: game-feel literature (not applied to robots) · assistive-teleop user studies (not whole-body) · Disney's operator interface (no published evaluation).*

**G3 — Responsiveness as an authored parameter.** In games, how much lead-in an action gets
is a per-action decision an animator makes. On a robot it emerges from reward weights nobody
chose deliberately. Exposing a runtime responsiveness knob that stays physically feasible is
a small, defensible contribution with an obvious demo.
*Nearest: DReCon (responsiveness in the title, sim) · Walk These Ways (behaviour parameters, gait shape rather than latency).*

**G4 — Retrieval at runtime on hardware.** This scan found no instance of nearest-neighbour
retrieval acting as the live reference generator for a real humanoid. The reason is probably
real — retrieval is discontinuous, and a kinematic pop is free while a dynamic one is a fall
— but that reason is itself an unpublished finding.
*Nearest: DReCon (motion matching + physics, sim) · Environment-aware Motion Matching (kinematic) · GMR real-time retargeting on CPU.*

**G5 — The operator as a data source.** An RPC produces a continuous stream of human
corrective input by construction. Sirius showed deployment interventions improve manipulation
policies; nothing does this for whole-body control, where the intervention is continuous
modulation rather than a binary takeover — a much richer signal.
*Nearest: Sirius, HG-DAgger, IWR (all tabletop, discrete interventions).*

**G6 — Phase as the shared coordinate.** Animation phase (PFNN, local motion phases,
DeepPhase), robot contact scheduling (periodic reward composition), CPG oscillator phase, and
prosthetic phase-variable control are four independent inventions of the same idea in four
literatures that do not cite each other. A paper unifying them — showing a learned phase
manifold as the interface between operator intent and contact schedule — is a genuine
synthesis, and the version of "repurpose PFNN" with real intellectual content.
*Nearest: DeepPhase (graphics) · Siekmann et al. (robotics) · Ijspeert (biology) · Gregg lab (prosthetics) — no cross-citations found.*

---

## Three objections you will get

**"This is just teleoperation with extra steps."**
OmniH2O already retargets a whole body onto a humanoid in real time. Why is a gamepad better
than a mocap suit?
> **Response.** Because channel bandwidth is the point, not fidelity. Teleoperation demands
> the operator's full body and full attention and produces motion bounded by what the
> operator can physically do. A character controller lets a seated human with two thumbs
> direct motion the robot is better at than they are — an asymmetry standard in games and
> unexplored in robotics. Frame it as bandwidth-appropriate control, not low-fidelity teleop.

**"HOVER and the Masked Humanoid Controller already did this."**
Multi-mode command spaces including joystick input, on real hardware, published.
> **Response.** True — cite them as evidence the architecture is sound, not as competitors.
> They built a controller that *accepts* many command types. What is missing is a designed
> human-facing interaction: which channel is authoritative, what mental model the operator
> should have, what the system does with intent it cannot execute, and whether any of it is
> pleasant to drive. That is an HRI and design contribution on top of their systems layer.

**"PFNN is obsolete — its own authors moved past it."**
The lineage went phase-indexed weights → gating → local phases → learned manifolds →
codebooks, and modern RL trackers get contact timing implicitly.
> **Response.** Concede and reframe. The durable exports are not the phase-indexed MLP: they
> are the *trajectory-feature command interface*, *layered composition*, *learned phase
> manifolds as an alignment metric*, and the production practice around motion databases. Say
> "PFNN and its successors" and let DeepPhase and Learned Motion Matching carry the technical
> weight — the architecture is the historical anchor, not the proposal.

---

## If you read ten things, in this order

1. **Design and Control of a Bipedal Robotic Character** (Disney, RSS 2024) — how much of the thesis is already built, and by whom.
2. **Masked Humanoid Controller** (arXiv:2408.07295) — closest baseline on real hardware; read the command interface carefully.
3. **DReCon** (SIGGRAPH Asia 2019) — the hinge: motion matching upstairs, RL physics downstairs, five years before robotics did it.
4. **MaskedMimic** (SIGGRAPH Asia 2024) — cleanest formal statement of partial specification as a control interface.
5. **DeepPhase** (SIGGRAPH 2022) — the modern form of the PFNN idea, and what you should actually propose to repurpose.
6. **HOVER** (ICRA 2025) — read the multi-mode motivation before writing your own.
7. **Shared Autonomy with Learned Latent Actions** (RSS 2020) — what a joystick axis should be connected to.
8. **BeyondMimic** (arXiv:2508.08241) — test-time steering with no retraining.
9. **Walk These Ways** (CoRL 2022) — behaviour parameters as an operator-facing tuning panel. Short, unusually close to the framing.
10. **Clavet, Motion Matching and the Road to Next-Gen Animation** (GDC 2016) — craft knowledge with no academic equivalent.

---

## Query terms and venues

**Animation side:** phase manifold / periodic autoencoder · local motion phase, contact phase ·
motion matching, trajectory features · neural animation layering · responsive character control ·
motion in-betweening, motion warping · directable / steerable character

**Robotics side:** whole-body controller, command space · masked / partial motion specification ·
behaviour foundation model, promptable · test-time steering, classifier guidance ·
periodic reward composition · multiplicity of behavior · motion tracking policy, retargeting

**Interaction side:** shared autonomy, policy blending · latent actions, control interface ·
sliding / adjustable / variable autonomy · traded control, supervisory control ·
interactive imitation learning · online language correction · game feel, perceived latency

**Venues:** SIGGRAPH / SIGGRAPH Asia (TOG) · SCA · RSS, CoRL, ICRA, IROS, RA-L · HRI ·
GDC vault (unindexed, high value) · `sebastianstarke/AI4Animation` (tracks the lineage) ·
`YanjieZe/awesome-humanoid-robot-learning` (tracks Track 3)

---

## Sourcing

Compiled 2026-09-05 from web search plus full-text reads of the Disney RSS 2024 paper and the
Masked Humanoid Controller. Everything else is from abstracts, project pages and search
summaries — titles, authors and venues for pre-2025 graphics and robotics work are
high-confidence, but **anything marked *(unverified)*, and every 2026 arXiv identifier, should
be verified against the actual paper before citing.** Author lists were not fully verified for
any entry. This is a map for deciding what to read, not a bibliography.
