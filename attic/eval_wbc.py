#!/usr/bin/env python3
"""Track a reference with the QP whole-body controller.

Two reference sources:
  --clip   a retargeted mocap clip (tests the controller alone)
  --pfnn   the open-loop generator (tests the full planner + controller stack)

``--no-base`` ablates the base term from the objective, leaving joint tracking
only, so we can measure what the base term buys instead of assuming it.
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
from rpc.generator.kinematic_runtime import KinematicCharacterController
from rpc.sim.robot import REGISTRY
from rpc.sim.estimator import LegOdometry, LowState
from rpc.sim.wbc import WBCConfig, WholeBodyController, apply_torque_actuators


def clip_reference(path, fps_out):
    from prepare_robot_data import resample

    qpos, ct, _ = resample(np.load(path), fps_out)
    return qpos, ct


def pfnn_reference(model_path, n, speeds, fps_out):
    from eval_kinematic import load

    model, spec, _ = load(model_path)
    ctrl = KinematicCharacterController(model, spec)
    ctrl.reset(root_height=0.75)
    per = max(1, n // len(speeds))
    q, ct = [], []
    for sp in speeds:
        for _ in range(per):
            f = ctrl.step(Command(move=np.array([sp, 0.0]), responsiveness=0.6))
            q.append(f.qpos(spec.n_joints))
            ct.append(f.contacts)
    return np.stack(q), np.stack(ct)


def run(qpos, contacts, robot, cfg, steps, fps, start=100, use_schedule=True):
    m = mujoco.MjModel.from_xml_path(robot.path())
    apply_torque_actuators(m)
    d = mujoco.MjData(m)
    n_sub = max(1, int(round(1.0 / (fps * m.opt.timestep))))
    wbc = WholeBodyController(m, robot, cfg)
    est = LegOdometry(m, robot)

    d.qpos[:] = qpos[start]
    d.qvel[:] = 0
    d.qvel[:3] = (qpos[start + 1, :3] - qpos[start, :3]) * fps
    d.qvel[6:] = (qpos[start + 1, 7:] - qpos[start, 7:]) * fps
    mujoco.mj_forward(m, d)

    n = min(steps, len(qpos) - start - 2)
    errs, hs, taus, infeas = [], [], [], 0
    for t in range(n):
        ref = qpos[start + t]
        qd = (qpos[start + t + 1, 7:] - ref[7:]) * fps
        v_ref = (qpos[start + t + 1, :3] - ref[:3]) * fps
        sched = contacts[start + t] if use_schedule else None
        state = LowState.from_sim(m, d)
        e = est.update(state, sched)
        tau = wbc.solve(d, state, e, ref[7:], qd,
                        (float(ref[2]), ref[3:7], v_ref), sched)
        infeas += wbc.last.get("status") != "ok"
        taus.append(wbc.last.get("tau_max", 0.0))
        d.ctrl[:] = tau
        for _ in range(n_sub):
            mujoco.mj_step(m, d)
        errs.append(float(np.abs(d.qpos[7:] - ref[7:]).mean()))
        hs.append(float(d.qpos[2]))
        if d.qpos[2] < 0.43 or -d.xmat[1].reshape(3, 3)[2, 2] > -0.3:
            break
    return {"steps": t + 1, "of": n, "sec": (t + 1) / fps,
            "joint_err": float(np.mean(errs)), "height": float(np.mean(hs)),
            "tau_max": float(np.max(taus)), "infeasible": infeas,
            "fell": bool(d.qpos[2] < 0.43)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="")
    ap.add_argument("--pfnn", default="")
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--speeds", type=float, nargs="*", default=[0.6, 1.2])
    ap.add_argument("--ablate", action="store_true", help="run with and without the base term")
    args = ap.parse_args()

    robot = REGISTRY[args.robot]
    if args.pfnn:
        qpos, contacts = pfnn_reference(args.pfnn, args.steps + 200, args.speeds, args.fps)
        label = f"PFNN {Path(args.pfnn).parent.name}"
    else:
        qpos, contacts = clip_reference(args.clip, args.fps)
        label = Path(args.clip).stem

    variants = [(True, True), (False, True), (True, False)] if args.ablate else [(True, True)]
    print(f"{'reference':22s} {'base':>5s} {'sched':>6s} {'steps':>12s} {'joint err':>10s} "
          f"{'tau max':>8s} {'infeas':>7s}")
    for use_base, use_sched in variants:
        r = run(qpos, contacts, robot, WBCConfig(use_base=use_base), args.steps,
                args.fps, use_schedule=use_sched)
        print(f"{label:22s} {('ON' if use_base else 'OFF'):>5s} "
              f"{('ON' if use_sched else 'OFF'):>6s} "
              f"{r['steps']:5d}/{r['of']:<6d} {r['joint_err']:10.4f} {r['tau_max']:8.1f} "
              f"{r['infeasible']:7d}  {'FELL' if r['fell'] else 'OK'}")


if __name__ == "__main__":
    main()
