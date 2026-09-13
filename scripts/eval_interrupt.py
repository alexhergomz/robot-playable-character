#!/usr/bin/env python3
"""Interruptibility: how fast can the operator change their mind?

This is the claim the architecture rests on.  A clip- or mode-based controller
commits: once a motion starts it plays to a boundary, which is why a humanoid
running a canned strike cannot be stopped or redirected mid-swing.  A
phase-indexed generator has no boundary -- every frame is a valid place to
change the command.

To measure the mechanism rather than a difference of animators, all three arms
use the *same* generator and the *same* tracker.  Only the policy governing
when a new command is allowed to take effect differs:

    continuous  the command applies on the frame it arrives           (ours)
    boundary    the command waits for the next gait-cycle boundary    (motion
                state machine / mode switch semantics)
    committed   the command waits out a fixed clip duration           (a canned
                animation, e.g. a strike, that must finish)

Reported per arm: reaction latency (when the robot's velocity first responds)
and settle latency (when it reaches 90% of the new steady state).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rpc.control.command import Command
from rpc.generator.lmp import LMPGenerator
from rpc.generator.lmp_runtime import LMPCharacterController
from rpc.motion.features import FeatureConfig
from rpc.motion.robot_dataset import RobotFeatureSpec
from rpc.sim import sonic as P
from rpc.sim.robot import REGISTRY
from rpc.sim.sonic_policy import (CONTROL_HZ, SonicPolicy, apply_sonic_armature,
                                  apply_sonic_gains)
from eval_rpc import lookahead


class Gate:
    """When is a newly issued command allowed to reach the generator?"""

    def __init__(self, mode: str, commit_s: float = 1.5):
        self.mode = mode
        self.commit = int(commit_s * CONTROL_HZ)
        self.held: Command | None = None
        self.pending: Command | None = None
        self.timer = 0

    def __call__(self, cmd: Command, phase_wrapped: bool) -> Command:
        if self.held is None:
            self.held = cmd
            return cmd
        if self.mode == "continuous":
            self.held = cmd
            return cmd
        changed = (not np.allclose(cmd.move, self.held.move)
                   or abs(cmd.turn - self.held.turn) > 1e-9)
        if changed and self.pending is None:
            self.pending = cmd
            self.timer = 0
        if self.pending is not None:
            self.timer += 1
            ready = phase_wrapped if self.mode == "boundary" else self.timer >= self.commit
            if ready:
                self.held = self.pending
                self.pending = None
        return self.held


def resolve_init(data, clip="walk", frame_in_clip=None):
    """Frame index of a clip whose name starts with ``clip``, in this dataset.

    A bare frame index is not portable: 60007 lies inside walk1_subject2 in the
    18-clip locomotion dataset and inside fightAndSports1_subject1 in the
    32-clip one, so reusing it silently initialises the phase state from a
    different motion -- which cost a third of the achieved speed and most of a
    comparison before it was noticed.  Naming the clip keeps evaluations
    comparable across corpora.
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


def build(gen_path: str):
    ck = torch.load(gen_path, map_location="cpu", weights_only=False)
    man, a = ck["manifest"], ck["args"]
    C = man["phase_channels"]
    cfg = FeatureConfig(fps=man["fps"], n_past=man["n_past"], n_future=man["n_future"],
                        span=man["span"], gaits=tuple(man["gaits"]))
    spec = RobotFeatureSpec(man["n_joints"], cfg, man["n_feet"])
    net = LMPGenerator(ck["n_in"], ck["n_out"], gate_dim=4 * C, hidden=a["hidden"],
                       n_experts=a["experts"], dropout=a["dropout"])
    net.load_state_dict(ck["state"])
    return net, spec, ck, man, C


def rollout(mode, gen_path, script, init_frame=None, commit_s=1.5, save=None,
            data="data/processed/robot_lmp", device="cuda"):
    net, spec, ck, man, C = build(gen_path)
    M = np.load(ROOT / data / "train_M.npy", mmap_mode="r")
    F = np.load(ROOT / data / "train_F.npy", mmap_mode="r")
    mo = mujoco.MjModel.from_xml_path(REGISTRY["g1"].path())
    apply_sonic_gains(mo); apply_sonic_armature(mo)
    d = mujoco.MjData(mo)
    pol = SonicPolicy(root=str(ROOT / "assets/sonic"), variant="low_latency")
    sub = max(1, int(round(1.0 / (CONTROL_HZ * mo.opt.timestep))))
    g = LMPCharacterController(net, spec, ck["out_slices"], 4 * C,
                               n_feet=man["n_feet"], keep_cols=ck.get("keep_cols"),
                               curve=ck.get("amplitude_curve"), device=device)
    mujoco.mj_resetDataKeyframe(mo, d, 0)
    d.qpos[7:] = P.DEFAULT_ANGLES; d.qpos[2] = 0.76
    mujoco.mj_forward(mo, d)
    g.reset(joint_pos=P.DEFAULT_ANGLES.copy(), root_height=0.76,
            manifold=M[init_frame].copy(), foot=F[init_frame].copy())

    def grav():
        o = np.zeros(3)
        mujoco.mju_rotVecQuat(o, np.array([0.0, 0.0, -1.0]),
                              np.array([d.qpos[3], -d.qpos[4], -d.qpos[5], -d.qpos[6]]))
        return o

    pol.reset(d.qpos[7:].copy(), d.qvel[6:].copy(), d.qvel[3:6].copy(), grav())
    gate = Gate(mode, commit_s)
    log, applied, prev_ang = [], [], None
    for t, cmd in enumerate(script):
        ang = np.arctan2(g.manifold[C:2 * C][0], g.manifold[:C][0])
        wrapped = prev_ang is not None and ang < prev_ang - np.pi
        prev_ang = ang
        eff = gate(cmd.clamped(), wrapped)
        dof, dvel, qt = lookahead(g, eff, pol.lookahead_frames * max(pol.step, 1), CONTROL_HZ)
        idx = np.arange(pol.lookahead_frames) * pol.step
        g.step(eff)
        tg = pol.step_policy(dof[idx][:, P.MUJOCO_TO_ISAACLAB],
                             dvel[idx][:, P.MUJOCO_TO_ISAACLAB], qt[idx],
                             d.qpos[3:7].copy(), d.qpos[7:].copy(),
                             d.qvel[6:].copy(), d.qvel[3:6].copy(), grav())
        d.ctrl[:] = np.clip(tg, mo.actuator_ctrlrange[:, 0], mo.actuator_ctrlrange[:, 1])
        for _ in range(sub):
            mujoco.mj_step(mo, d)
        log.append(d.qpos.copy())
        applied.append(eff.speed)
        if d.qpos[2] < 0.45:
            break
    q = np.stack(log)
    if save:
        np.savez_compressed(save, qpos=q.astype(np.float32),
                            contacts=np.zeros((len(q), 2), np.float32),
                            phase=np.zeros(len(q), np.float32), fps=CONTROL_HZ)
    return q, np.array(applied)


def _yaw_rate(q, fps):
    w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
    yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return np.diff(yaw) * fps


def latency(q, t_change, before, after, fps=CONTROL_HZ, channel="speed"):
    """Reaction and settle latency in ms, from the robot's own motion track.

    ``channel`` picks what the manoeuvre actually changes: forward speed for a
    stop or a dash, yaw rate for a turn.  Measuring a turn on the speed track
    reports a meaningless zero, since turning barely alters forward speed.
    """
    if channel == "yaw":
        v = _yaw_rate(q, fps)
    else:
        v = np.linalg.norm(np.diff(q[:, :2], axis=0), axis=-1) * fps
    v = np.convolve(v, np.ones(9) / 9, "same")
    base = v[max(0, t_change - 25):t_change].mean()
    tgt = v[min(len(v) - 1, t_change + int(3.0 * fps) - 25):
            min(len(v), t_change + int(3.0 * fps))].mean()
    span = tgt - base
    if abs(span) < 0.05:
        return None, None, base, tgt
    react = settle = None
    for i in range(t_change, min(len(v), t_change + int(4.0 * fps))):
        p = (v[i] - base) / span
        if react is None and p > 0.10:
            react = (i - t_change) / fps * 1000
        if p > 0.90:
            settle = (i - t_change) / fps * 1000
            break
    return react, settle, base, tgt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="runs/lmp_nogait/generator.pt")
    ap.add_argument("--data", default="data/processed/robot_lmp")
    ap.add_argument("--init-clip", default="walk")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--commit", type=float, default=1.5)
    ap.add_argument("--video", action="store_true")
    args = ap.parse_args()

    init, init_name = resolve_init(ROOT / args.data, args.init_clip)
    print(f"phase state initialised from {init_name} (frame {init})\n")

    HZ = CONTROL_HZ
    tests = {
        "emergency stop  (1.6 -> 0)": (
            [Command(move=np.array([1.6, 0.0]), responsiveness=0.8)] * int(5 * HZ)
            + [Command(move=np.zeros(2), responsiveness=0.8)] * int(5 * HZ), int(5 * HZ)),
        "sudden dash     (0.4 -> 1.6)": (
            [Command(move=np.array([0.4, 0.0]), responsiveness=0.8)] * int(5 * HZ)
            + [Command(move=np.array([1.6, 0.0]), responsiveness=0.8)] * int(5 * HZ), int(5 * HZ)),
        "hard turn       (fwd -> turn)": (
            [Command(move=np.array([1.2, 0.0]), responsiveness=0.8)] * int(5 * HZ)
            + [Command(move=np.array([1.2, 0.0]), turn=1.5, responsiveness=0.8)] * int(5 * HZ),
            int(5 * HZ)),
    }
    CHANNEL = {"hard turn       (fwd -> turn)": "yaw"}
    print(f"commitment window for the 'committed' arm: {args.commit:.1f} s\n")
    for name, (script, tc) in tests.items():
        print(name)
        for mode in ("continuous", "boundary", "committed"):
            tag = f"runs/interrupt_{mode}_{name.split()[0]}.npz" if args.video else None
            q, _ = rollout(mode, args.gen, script, init_frame=init, commit_s=args.commit,
                           save=tag, data=args.data, device=args.device)
            ch = CHANNEL.get(name, "speed")
            r, s, b, a = latency(q, tc, None, None, channel=ch)
            rs = f"{r:6.0f} ms" if r is not None else "    n/a"
            ss = f"{s:6.0f} ms" if s is not None else "  >4000 ms"
            u = "rad/s" if ch == "yaw" else "m/s"
            print(f"   {mode:11s} react {rs}   settle {ss}    {b:5.2f} -> {a:5.2f} {u}")
        print()


if __name__ == "__main__":
    main()
