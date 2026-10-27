#!/usr/bin/env python3
"""Run the full streaming path in simulation, through the actual wire bytes.

    generator -> PoseStreamer -> bytes -> receiver model -> SONIC -> MuJoCo

The receiver model reproduces the parts of the deploy app's
StreamedMotionMerger that matter for safety: a frame window keyed by global
frame index, newer chunks overwriting frames from their first index onward, a
catch-up reset whenever a chunk does not end later than the window, and a
playback cursor that holds the final frame once it runs past the window.

Scenarios: a graceful session (stand, walk, stop, settle), and a crash -- the
streamer killed mid-stride with no settling, which is what the deploy app's
hold-last-frame behaviour turns into a frozen stride.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from rpc.control.command import Command
from rpc.deploy import wire
from rpc.deploy.streamer import PoseStreamer, StreamLimits
from rpc.generator.lmp import LMPGenerator
from rpc.generator.lmp_runtime import LMPCharacterController
from rpc.motion.features import FeatureConfig
from rpc.motion.robot_dataset import RobotFeatureSpec
from rpc.sim import sonic as P
from rpc.sim import sonic_scene
from rpc.sim.robot import REGISTRY
from rpc.sim.sonic_policy import (CONTROL_HZ, SonicPolicy, apply_sonic_armature,
                                  apply_sonic_gains)

HZ = CONTROL_HZ


class ReceiverModel:
    def __init__(self, lookahead: int):
        self.win: dict[int, tuple] = {}
        self.end = -1
        self.cursor: int | None = None
        self.lookahead = lookahead
        self.catchups = 0

    def on_message(self, msg: bytes):
        hdr, d = wire.unpack(msg, "pose")
        assert hdr["v"] == 1
        fi = d["frame_index"]
        if self.end >= 0 and fi[-1] <= self.end:
            self.catchups += 1                       # the deploy app would jump here
        for j, k in enumerate(fi):
            self.win[int(k)] = (d["joint_pos"][j], d["joint_vel"][j], d["body_quat"][j, 0])
        self.end = max(self.end, int(fi[-1]))
        if self.cursor is None:
            self.cursor = int(fi[0])

    def frames(self):
        """Lookahead from the cursor; past the window, hold the final frame."""
        last = max(self.win)
        ks = [min(self.cursor + i, last) for i in range(self.lookahead)]
        pos = np.stack([self.win[k][0] for k in ks])
        vel = np.stack([self.win[k][1] if self.cursor + i <= last else np.zeros(29)
                        for i, k in enumerate(ks)])
        quat = np.stack([self.win[k][2] for k in ks])
        return pos, vel, quat, self.cursor + self.lookahead - 1 > last

    def advance(self):
        self.cursor += 1


def build(bundle_path, device, hold_phase_when_idle=False):
    """Generator from a deploy bundle (scripts/export_deploy.py): no datasets."""
    ck = torch.load(bundle_path, map_location="cpu", weights_only=False)
    man, a = ck["manifest"], ck["args"]; C = man["phase_channels"]
    cfg = FeatureConfig(fps=man["fps"], n_past=man["n_past"], n_future=man["n_future"],
                        span=man["span"], gaits=tuple(man["gaits"]))
    spec = RobotFeatureSpec(man["n_joints"], cfg, man["n_feet"])
    net = LMPGenerator(ck["n_in"], ck["n_out"], gate_dim=4 * C, hidden=a["hidden"],
                       n_experts=a["experts"], dropout=a["dropout"])
    net.load_state_dict(ck["state"])
    g = LMPCharacterController(net, spec, ck["out_slices"], 4 * C, n_feet=man["n_feet"],
                               keep_cols=ck.get("keep_cols"),
                               curve=ck["amplitude_curve"], device=device,
                               hold_phase_when_idle=hold_phase_when_idle)
    g.reset(joint_pos=P.DEFAULT_ANGLES.copy(), root_height=0.76,
            manifold=ck["init_manifold"].copy(), foot=ck["init_foot"].copy())
    return g


def run(scenario, args):
    if args.model == "sonic":            # SONIC's own model, per-step torque PD
        mo = sonic_scene.load()
        pd = sonic_scene.PDStep(mo)
    else:                                # Menagerie, position actuators (legacy)
        mo = mujoco.MjModel.from_xml_path(REGISTRY["g1"].path())
        apply_sonic_gains(mo); apply_sonic_armature(mo)
        pd = None
    d = mujoco.MjData(mo)
    sub = max(1, int(round(1.0 / (HZ * mo.opt.timestep))))
    lo, hi = mo.jnt_range[1:30, 0].copy(), mo.jnt_range[1:30, 1].copy()
    pol = SonicPolicy(root=str(ROOT / "assets/sonic"), variant="low_latency")
    rx = ReceiverModel(pol.lookahead_frames)
    clock = [0.0]
    st = PoseStreamer(build(args.bundle, args.device), rx.on_message,
                      joint_lower=lo, joint_upper=hi,
                      mujoco_to_isaaclab=P.MUJOCO_TO_ISAACLAB, horizon=args.horizon,
                      limits=StreamLimits(max_speed=args.speed),
                      stand_pose=P.DEFAULT_ANGLES.copy(), clock=lambda: clock[0])
    if pd is None:
        mujoco.mj_resetDataKeyframe(mo, d, 0)
    d.qpos[3] = 1.0
    d.qpos[7:] = P.DEFAULT_ANGLES; d.qpos[2] = 0.76 if pd is None else 0.79
    mujoco.mj_forward(mo, d)
    if pd is not None:
        pd(d, P.DEFAULT_ANGLES, 50)

    def grav():
        o = np.zeros(3)
        mujoco.mju_rotVecQuat(o, np.array([0.0, 0.0, -1.0]),
                              np.array([d.qpos[3], -d.qpos[4], -d.qpos[5], -d.qpos[6]]))
        return o
    pol.reset(d.qpos[7:].copy(), d.qvel[6:].copy(), d.qvel[3:6].copy(), grav())

    walk = Command(move=np.array([args.speed, 0.0]), responsiveness=0.6)
    stand = Command(move=np.zeros(2), responsiveness=0.6)
    plan = [(2.0, stand), (args.walk_s, walk)]
    phases = [(int(s * HZ), c) for s, c in plan]
    streaming, log, held_from, stop_t = True, [], None, None
    total = int((sum(s for s, _ in plan) + args.after_s + 5.0) * HZ)
    t_cmd = 0
    for t in range(total):
        clock[0] = t / HZ
        if streaming:
            elapsed, cmd = t, None
            for n, c in phases:
                if elapsed < n:
                    cmd = c; break
                elapsed -= n
            if cmd is None:                      # script over: end the session
                if scenario == "graceful" and t_cmd < int(st.limits.brake_s * HZ):
                    cmd = stand                  # brake, then simply stop sending
                    t_cmd += 1
                else:                            # graceful: done braking; crash: dies
                    streaming = False
                    stop_t = t if stop_t is None else stop_t
            if streaming:
                st.tick(cmd)
        pos, vel, quat, holding = rx.frames()
        if holding and held_from is None:
            held_from = t
        tg = pol.step_policy(pos, vel, quat, d.qpos[3:7].copy(), d.qpos[7:].copy(),
                             d.qvel[6:].copy(), d.qvel[3:6].copy(), grav())
        if pd is not None:
            pd(d, tg, sub)
        else:
            d.ctrl[:] = np.clip(tg, mo.actuator_ctrlrange[:, 0], mo.actuator_ctrlrange[:, 1])
            for _ in range(sub):
                mujoco.mj_step(mo, d)
        rx.advance()
        log.append(d.qpos.copy())
        if d.qpos[2] < 0.45:
            break
    q = np.stack(log)
    fell = q[-1, 2] < 0.45
    walk_seg = q[int(2.5 * HZ):int((2.0 + args.walk_s) * HZ)]
    v = np.linalg.norm(walk_seg[-1, :2] - walk_seg[0, :2]) / (len(walk_seg) / HZ)
    after = q[stop_t:] if stop_t is not None else q[-1:]
    drift = np.linalg.norm(after[-1, :2] - after[0, :2])
    v_end = np.linalg.norm(np.diff(q[-int(2 * HZ):, :2], axis=0), axis=-1).mean() * HZ
    if args.save:
        np.savez_compressed(f"{args.save}_{scenario}.npz", qpos=q.astype(np.float32),
                            contacts=np.zeros((len(q), 2), np.float32),
                            phase=np.zeros(len(q), np.float32), fps=HZ)
    return dict(fell=fell, seconds=len(q) / HZ, speed=v, catchups=rx.catchups,
                sent=st.stats.sent, clipped=st.stats.clipped_frames,
                gen_ms=float(np.mean(st.stats.gen_ms)),
                held_s=(len(q) - held_from) / HZ if held_from is not None else 0.0,
                drift_while_held=drift, min_h=float(q[:, 2].min()), v_end=v_end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="runs/deploy/bundle.pt")
    ap.add_argument("--model", choices=("sonic", "menagerie"), default="sonic",
                    help="robot model: SONIC's own (default) or MuJoCo Menagerie's")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--speed", type=float, default=0.3)
    ap.add_argument("--walk-s", type=float, default=8.0)
    ap.add_argument("--after-s", type=float, default=3.0)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    for sc in ("graceful", "crash"):
        r = run(sc, args)
        print(f"{sc:9s} fell={'YES' if r['fell'] else 'no '} {r['seconds']:5.1f}s  "
              f"walk {r['speed']:.2f} m/s  sent {r['sent']}  catch-up resets {r['catchups']}  "
              f"clipped {r['clipped']}  gen {r['gen_ms']:.1f} ms  "
              f"travel after stream ends {r['drift_while_held']:.2f} m  "
              f"speed in last 2 s {r['v_end']:.3f} m/s  min h {r['min_h']:.3f}")


if __name__ == "__main__":
    main()
