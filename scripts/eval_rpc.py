#!/usr/bin/env python3
"""The whole Robot Playable Character loop: stick -> PFNN -> SONIC -> physics.

    operator (gamepad / scripted / language)   30-50 Hz
        -> Command (velocity, turn, gait, responsiveness)
        -> PFNN, open loop, robot joint space           1.8 ms/frame
        -> lookahead window of reference poses
        -> SONIC encoder/decoder                        8.4 ms/step
        -> joint targets -> PD -> MuJoCo                50 Hz / 500 Hz

PFNN never sees the robot.  It is an animator: it turns operator intent into a
dense, phase-coherent pose stream and nothing else.  SONIC never sees the
operator.  It is a tracker: it makes that stream physically true and keeps the
robot upright.  Neither knows the other exists.

The lookahead is **regenerated every control step** from PFNN's current state
under the current command, rather than read from a buffer filled 200 ms ago.
Buffering would be cheaper, but it would mean a stick movement takes a full
lookahead window to reach the robot, which is precisely the latency the whole
architecture exists to avoid.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rpc.control.arbiter import ArbiterConfig, ChannelArbiter
from rpc.control.command import Command
from rpc.control.input import GamepadInput, ScriptedInput
from rpc.generator.kinematic_runtime import KinematicCharacterController
from rpc.motion import quat as Q
from rpc.sim import sonic as P
from rpc.sim.robot import REGISTRY
from rpc.sim.sonic_policy import (CONTROL_HZ, SonicPolicy, apply_sonic_armature,
                                  apply_sonic_gains)


def lookahead(gen: KinematicCharacterController, cmd: Command, n: int, fps: float):
    """Roll PFNN forward n frames under the current command, without consuming it."""
    snap = copy.deepcopy(gen)
    prev = gen.joint_pos.copy()          # the frame *before* the window
    dof, quats = [], []
    for _ in range(n):
        f = snap.step(cmd)
        dof.append(f.joint_pos.copy())
        quats.append(Q.yaw_quat(np.array(f.root_yaw)))
    dof = np.stack(dof)
    # prepend the pose preceding the window, not dof[0] itself -- prepending
    # dof[:1] makes the anchor frame's reference velocity identically zero,
    # which tells the tracker to arrest joint motion on every control step.
    dvel = np.diff(dof, axis=0, prepend=prev[None]) * fps
    return dof, dvel, np.stack(quats)


DEFAULT_SCRIPT = [
    (2.0, Command(move=np.zeros(2))),
    (4.0, Command(move=np.array([0.9, 0.0]), responsiveness=0.6)),
    (3.0, Command(move=np.array([1.6, 0.0]), responsiveness=0.7)),
    (3.0, Command(move=np.array([0.9, 0.0]), turn=0.8, responsiveness=0.7)),
    (3.0, Command(move=np.zeros(2))),
    (6.0, Command(move=np.array([1.0, 0.0]), responsiveness=0.6)),
    (5.0, Command(move=np.array([1.0, 0.0]), turn=-0.8, responsiveness=0.7)),
    (6.0, Command(move=np.array([1.4, 0.0]), responsiveness=0.7)),
    (5.0, Command(move=np.array([0.6, 0.0]), responsiveness=0.5)),
    (30.0, Command(move=np.array([1.0, 0.0]), responsiveness=0.6)),
]



def resolve_init(data, clip="walk", frame_in_clip=None):
    """Frame index of a clip whose name starts with ``clip``, in this dataset.

    A bare frame index is not portable: 60007 is inside walk1_subject2 in the
    18-clip locomotion dataset and inside fightAndSports1_subject1 in the
    32-clip one, so reusing it silently initialises the phase state from a
    completely different motion.  Naming the clip keeps evaluations comparable
    across corpora.
    """
    import json
    from pathlib import Path
    man = json.loads((Path(data) / "manifest.json").read_text())
    acc = 0
    for c in man["clips"]:
        if c["split"] != "train":
            continue
        if c["name"].startswith(clip):
            off = c["frames"] // 2 if frame_in_clip is None else frame_in_clip
            return acc + min(off, c["frames"] - 1), c["name"]
        acc += c["frames"]
    raise SystemExit(f"no train clip starting with {clip!r} in {data}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pfnn", default="runs/sweep_n0.0_d0.7_h1024_e90/generator.pt")
    ap.add_argument("--lmp", default="", help="use a Local-Motion-Phase generator instead")
    ap.add_argument("--init-clip", default="",
                    help="initialise the phase state from this clip "
                         "(portable across datasets, unlike a frame index)")
    ap.add_argument("--init-frame", type=int, default=-1,
                    help="seed the LMP phase manifold from this training frame "
                         "instead of the degenerate all-channels-in-unison default")
    ap.add_argument("--sonic", default="assets/sonic")
    ap.add_argument("--variant", default="low_latency")
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--say", default="")
    ap.add_argument("--save", default="")
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda; the generator dominates the step budget")
    args = ap.parse_args()

    from eval_kinematic import load

    if args.lmp:
        import torch
        from rpc.generator.lmp import LMPGenerator
        from rpc.generator.lmp_runtime import LMPCharacterController
        from rpc.motion.features import FeatureConfig
        from rpc.motion.robot_dataset import RobotFeatureSpec
        ck = torch.load(args.lmp, map_location="cpu", weights_only=False)
        man = ck["manifest"]
        cfg = FeatureConfig(fps=man["fps"], n_past=man["n_past"], n_future=man["n_future"],
                            span=man["span"], gaits=tuple(man["gaits"]))
        spec = RobotFeatureSpec(man["n_joints"], cfg, man["n_feet"])
        a = ck["args"]
        C = man["phase_channels"]
        model = LMPGenerator(ck["n_in"], ck["n_out"], gate_dim=4 * C,
                             hidden=a["hidden"], n_experts=a["experts"],
                             dropout=a["dropout"])
        model.load_state_dict(ck["state"])
        gen = LMPCharacterController(model, spec, ck["out_slices"], 4 * C,
                                     n_feet=man["n_feet"],
                                     keep_cols=ck.get("keep_cols"),
                                     curve=ck.get("amplitude_curve"),
                                     device=args.device)
        tag = ("LMP-nogait " if ck.get("no_gait") else "LMP ") + Path(args.lmp).parent.name
    else:
        model, spec, man = load(args.pfnn)
        gen = KinematicCharacterController(model, spec)
        tag = "PFNN " + Path(args.pfnn).parent.name
    robot = REGISTRY[args.robot]

    m = mujoco.MjModel.from_xml_path(robot.path())
    apply_sonic_gains(m)
    apply_sonic_armature(m)
    d = mujoco.MjData(m)
    pol = SonicPolicy(root=args.sonic, variant=args.variant)
    n_look, step = pol.lookahead_frames, pol.step
    sub = max(1, int(round(1.0 / (CONTROL_HZ * m.opt.timestep))))

    device = GamepadInput() if args.live else ScriptedInput(DEFAULT_SCRIPT, dt=1 / CONTROL_HZ)
    arb, lang_cmd = None, None
    if args.say:
        from rpc.control.language import LanguageChannel, MotionEntry
        lib = [MotionEntry(e["name"], tuple(e.get("labels", (e["name"],))),
                           np.array(e["gait"])) for e in man.get("library", [])] or None
        if lib:
            lang_cmd, _ = LanguageChannel().index(lib).interpret(args.say)
            arb = ChannelArbiter(ArbiterConfig(), dt=1 / CONTROL_HZ)

    # start standing, then let PFNN take over from the nominal pose
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.qpos[7:] = P.DEFAULT_ANGLES
    d.qpos[2] = 0.76
    mujoco.mj_forward(m, d)
    if args.lmp and args.init_frame >= 0:
        D = Path("data/processed/robot_lmp")
        M0 = np.load(D / "train_M.npy", mmap_mode="r")[args.init_frame].copy()
        F0 = np.load(D / "train_F.npy", mmap_mode="r")[args.init_frame].copy()
        gen.reset(joint_pos=P.DEFAULT_ANGLES.copy(), root_height=0.76,
                  manifold=M0, foot=F0)
    else:
        gen.reset(joint_pos=P.DEFAULT_ANGLES.copy(), root_height=0.76)

    def gravity():
        o = np.zeros(3)
        mujoco.mju_rotVecQuat(o, np.array([0.0, 0.0, -1.0]),
                              np.array([d.qpos[3], -d.qpos[4], -d.qpos[5], -d.qpos[6]]))
        return o

    pol.reset(d.qpos[7:].copy(), d.qvel[6:].copy(), d.qvel[3:6].copy(), gravity())

    n = int(args.seconds * CONTROL_HZ)
    log, cmds, gen_ms, pol_ms = [], [], [], []
    for t in range(n):
        stick = device.poll()
        cmd = arb.update(lang_cmd, stick, args.say) if arb else stick
        cmd = cmd.clamped()          # the robot's measured ceiling, not a guess

        t0 = time.perf_counter()
        dof, dvel, quats = lookahead(gen, cmd, n_look * max(step, 1), CONTROL_HZ)
        idx = np.arange(n_look) * step
        gen.step(cmd)                                   # advance the real reference
        gen_ms.append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        target = pol.step_policy(dof[idx][:, P.MUJOCO_TO_ISAACLAB],
                                 dvel[idx][:, P.MUJOCO_TO_ISAACLAB],
                                 quats[idx], d.qpos[3:7].copy(),
                                 d.qpos[7:].copy(), d.qvel[6:].copy(),
                                 d.qvel[3:6].copy(), gravity())
        pol_ms.append((time.perf_counter() - t0) * 1e3)

        d.ctrl[:] = np.clip(target, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
        for _ in range(sub):
            mujoco.mj_step(m, d)
        log.append(d.qpos.copy())
        cmds.append((cmd.move[0], cmd.move[1], cmd.turn))
        if d.qpos[2] < 0.45 or d.xmat[1].reshape(3, 3)[2, 2] < 0.3:
            break

    q = np.stack(log)
    c = np.array(cmds)
    fell = d.qpos[2] < 0.45 or d.xmat[1].reshape(3, 3)[2, 2] < 0.3
    print(f"input: {device.mode}   gen: {tag}   SONIC: {args.variant}")
    print(f"  survived {len(q)}/{n} steps ({len(q)/CONTROL_HZ:5.2f} s of {args.seconds:.0f}) "
          f"{'FELL' if fell else 'STILL UP'}")
    print(f"  height mean {q[:,2].mean():.3f}  min {q[:,2].min():.3f}")
    print(f"  timing: PFNN lookahead {np.mean(gen_ms):5.2f} ms   SONIC {np.mean(pol_ms):5.2f} ms"
          f"   budget {1000/CONTROL_HZ:.0f} ms")
    v = np.linalg.norm(np.diff(q[:, :2], axis=0), axis=-1) * CONTROL_HZ
    for lo, hi, lab in [(0, 0.2, "stand"), (0.7, 1.1, "walk 0.9"), (1.3, 1.7, "walk 1.5")]:
        sel = (np.linalg.norm(c[:-1, :2], axis=-1) > lo) & (np.linalg.norm(c[:-1, :2], axis=-1) <= hi)
        if sel.sum() > 10:
            print(f"  commanded {lab:9s} -> achieved {v[sel].mean():.2f} m/s")
    if args.save:
        np.savez_compressed(args.save, qpos=q.astype(np.float32),
                            contacts=np.zeros((len(q), 2), np.float32),
                            phase=np.zeros(len(q), np.float32), fps=CONTROL_HZ)
        print(f"  saved {args.save}")


if __name__ == "__main__":
    main()
