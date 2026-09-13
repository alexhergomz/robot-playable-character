#!/usr/bin/env python3
"""Does the controller stand up in physics?

Three modes, deliberately ordered from "is the data even executable" upward, so
a failure can be attributed:

* ``replay``    - feed the reference joint angles straight to the servos, open
                  loop.  This tests the *data*, not the network.  If a clip
                  cannot be replayed it cannot be learned either.
* ``pfnn``      - the trained generator, closed loop through the simulator.
* ``hold``      - hold the first pose.  The floor: anything that survives less
                  than this is worse than doing nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpc.control.command import Command
from rpc.generator.pfnn import PFNN
from rpc.generator.physics_runtime import PhysicsCharacterController
from rpc.motion.features import FeatureConfig
from rpc.motion.robot_dataset import (RobotFeatureSpec, analyse_robot_clip,
                                      build_robot_samples)
from rpc.sim.env import RobotSim, SimConfig
from rpc.sim.robot import G1
from rpc.sim.sonic_policy import apply_sonic_gains


def load_clip(path: Path, fps: float):
    sys.path.insert(0, "scripts")
    from prepare_robot_data import resample

    return resample(np.load(path), fps)


def make_sim(hz: float, sonic_gains: bool):
    sim = RobotSim(G1, SimConfig(control_hz=hz))
    if sonic_gains:
        apply_sonic_gains(sim.model)
    return sim


def seed_state(sim, qpos, i, fps):
    """Start on the reference *with its velocities* — a standing start into a
    mid-stride pose is a disturbance the controller never recovers from."""
    q = sim.default_qpos.copy()
    q[:] = qpos[i]
    v = np.zeros(sim.model.nv)
    if i + 1 < len(qpos):
        v[:3] = (qpos[i + 1, :3] - qpos[i, :3]) * fps
        v[6:] = (qpos[i + 1, 7:] - qpos[i, 7:]) * fps
    return q, v


def run(mode: str, clip: Path, args):
    qpos, ct, ph = load_clip(clip, args.fps)
    sim = make_sim(args.fps, not args.stock_gains)
    start = args.start
    q0, v0 = seed_state(sim, qpos, start, args.fps)

    if mode == "pfnn":
        ck = torch.load(args.model, map_location="cpu", weights_only=False)
        man = ck["manifest"]
        cfg = FeatureConfig(fps=man["fps"], n_past=man["n_past"],
                            n_future=man["n_future"], span=man["span"],
                            gaits=tuple(man["gaits"]))
        spec = RobotFeatureSpec(man["n_joints"], cfg, man.get("n_feet", 2))
        a = ck["args"]
        model = PFNN(spec.n_in, spec.n_out, hidden=a["hidden"],
                     n_control=a["experts"], dropout=a["dropout"])
        model.load_state_dict(ck["state"])
        ctrl = PhysicsCharacterController(model, spec, sim,
                                          action_filter=args.action_filter)
        f = analyse_robot_clip(qpos, ct, ph, args.fps, cfg, clip.stem)
        ctrl.reset(qpos=q0, qvel=v0, phase=float(ph[start]),
                   gait=f.gait[start, cfg.n_past])
        cmd = Command(move=np.array([args.speed, 0.0]), responsiveness=0.6)

    sim.reset(qpos=q0, qvel=v0)
    if mode == "pfnn":
        ctrl.sim = sim
        ctrl.reset(qpos=q0, qvel=v0, phase=float(ph[start]),
                   gait=f.gait[start, cfg.n_past])

    hs, ups, errs, sp = [], [], [], []
    n = min(args.steps, len(qpos) - start - 2)
    for t in range(n):
        if mode == "replay":
            sim.step_position(qpos[start + t, 7:])
        elif mode == "hold":
            sim.step_position(qpos[start, 7:])
        else:
            ctrl.step(cmd)
        hs.append(float(sim.root_pos[2]))
        g = sim.projected_gravity()
        ups.append(float(-g[2]))
        errs.append(float(np.abs(sim.joint_pos - qpos[start + t, 7:]).mean()))
        if t:
            sp.append(float(np.linalg.norm(sim.data.qvel[:2])))
        if sim.fallen():
            break
    steps = t + 1
    return {"mode": mode, "clip": clip.stem, "steps": steps, "seconds": steps / args.fps,
            "fell": sim.fallen(), "height": float(np.mean(hs)),
            "upright": float(np.mean(ups)), "joint_err": float(np.mean(errs)),
            "speed": float(np.mean(sp)) if sp else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/pfnn_robot/generator.pt")
    ap.add_argument("--clips", default="data/processed/retargeted/g1")
    ap.add_argument("--only", nargs="*", default=["walk1_subject1"])
    ap.add_argument("--modes", nargs="*", default=["hold", "replay", "pfnn"])
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--action-filter", type=float, default=0.0)
    ap.add_argument("--stock-gains", action="store_true")
    args = ap.parse_args()

    files = [Path(args.clips) / f"{n}.npz" for n in args.only] if args.only \
        else sorted(Path(args.clips).glob("*.npz"))
    print(f"{'mode':8s} {'clip':20s} {'steps':>10s} {'upright':>8s} {'height':>7s} "
          f"{'jointerr':>9s} {'speed':>6s}")
    for f in files:
        for mode in args.modes:
            if mode == "pfnn" and not Path(args.model).exists():
                continue
            r = run(mode, f, args)
            print(f"{r['mode']:8s} {r['clip']:20s} {r['steps']:4d}/{args.steps:<5d} "
                  f"{r['upright']:8.3f} {r['height']:7.3f} {r['joint_err']:9.3f} "
                  f"{r['speed']:6.2f}{'  FELL' if r['fell'] else '  ok'}")


if __name__ == "__main__":
    main()
