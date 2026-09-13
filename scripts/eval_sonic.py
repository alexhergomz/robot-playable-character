#!/usr/bin/env python3
"""Run GEAR-SONIC as the executor in our MuJoCo G1.

Two ambiguities cannot be settled from the source alone — whether a reference
file's ``dof`` is stored in IsaacLab or hardware joint order, and whether its
``root_rot`` is wxyz or xyzw.  Rather than guess, ``--probe`` runs all four
combinations and reports which one keeps the robot upright.  A tracking policy
fed a permuted reference falls over in well under a second, so the signal is
unambiguous.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpc.sim import sonic as P
from rpc.sim.robot import REGISTRY
from rpc.sim.sonic_policy import (CONTROL_HZ, SonicPolicy, apply_sonic_armature,
                                 apply_sonic_gains)


def load_our_clip(path: str, fps_out: float = CONTROL_HZ):
    """Load a clip from ``retarget_clips.py``.

    Our format is MuJoCo-native: ``qpos`` = root pos(3) + wxyz quat(4) + 29
    joints in hardware order.  Feeding it to SONIC is the direct test of
    whether our retargeter produces motion the executor can actually follow —
    tracking error here is a measure of the *reference*, not of SONIC.
    """
    z = np.load(path)
    qpos, fps_in = z["qpos"].astype(np.float64), float(z["fps"])
    n_out = int(len(qpos) * fps_out / fps_in)
    src = np.linspace(0, len(qpos) - 1, n_out)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, len(qpos) - 1)
    a = (src - i0)[:, None]
    q = qpos[i0] * (1 - a) + qpos[i1] * a
    return {"dof": q[:, 7:], "root_rot": q[:, 3:7], "root_trans": q[:, :3], "fps": fps_out}


def load_reference(path: str, fps_out: float = CONTROL_HZ):
    """Load a SONIC-format motion and resample to the control rate."""
    import joblib

    d = joblib.load(path)
    if len(d) == 1 and isinstance(next(iter(d.values())), dict):
        d = next(iter(d.values()))
    fps_in = float(d["fps"])
    dof, rot, trans = d["dof"], d["root_rot"], d["root_trans_offset"]
    n_out = int(len(dof) * fps_out / fps_in)
    src = np.linspace(0, len(dof) - 1, n_out)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, len(dof) - 1)
    a = (src - i0)[:, None]
    return {
        "dof": (dof[i0] * (1 - a) + dof[i1] * a).astype(np.float64),
        "root_rot": (rot[i0] * (1 - a) + rot[i1] * a).astype(np.float64),
        "root_trans": (trans[i0] * (1 - a) + trans[i1] * a).astype(np.float64),
        "fps": fps_out,
    }


def rollout(ref, robot, order: str, quat: str, steps: int, seed: int = 0,
            variant: str = "low_latency", push: float = 0.0, viewer=None,
            sonic_armature: bool = True):
    model = mujoco.MjModel.from_xml_path(robot.path())
    apply_sonic_gains(model)
    if sonic_armature:
        apply_sonic_armature(model)
    data = mujoco.MjData(model)
    n_sub = max(1, int(round(1.0 / (CONTROL_HZ * model.opt.timestep))))

    dof = ref["dof"] if order == "isaaclab" else ref["dof"][:, P.MUJOCO_TO_ISAACLAB]
    dof_hw = dof[:, P.ISAACLAB_TO_MUJOCO]
    q = ref["root_rot"]
    quats = q if quat == "wxyz" else q[:, [3, 0, 1, 2]]
    quats = quats / np.linalg.norm(quats, axis=-1, keepdims=True)
    dvel = np.gradient(dof, axis=0) * ref["fps"]
    trans = ref["root_trans"]

    # start on the reference
    data.qpos[:3] = trans[0]
    data.qpos[2] = max(float(trans[0][2]), 0.60)
    data.qpos[3:7] = quats[0]
    data.qpos[7:] = dof_hw[0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    pol = SonicPolicy(variant=variant)
    grav = lambda: (np.array(mujoco.mju_rotVecQuat(np.zeros(3), np.array([0.0, 0.0, -1.0]),
                                                   np.array([data.qpos[3], -data.qpos[4],
                                                             -data.qpos[5], -data.qpos[6]]))
                             or np.zeros(3)))
    def gravity_dir():
        out = np.zeros(3)
        mujoco.mju_rotVecQuat(out, np.array([0.0, 0.0, -1.0]),
                              np.array([data.qpos[3], -data.qpos[4], -data.qpos[5], -data.qpos[6]]))
        return out

    pol.reset(data.qpos[7:].copy(), data.qvel[6:].copy(), data.qvel[3:6].copy(), gravity_dir())

    n = len(dof)
    step = pol.step
    errs, heights, alive = [], [], 0
    rng = np.random.default_rng(seed)
    for t in range(min(steps, n - 1)):
        idx = np.clip(t + np.arange(pol.lookahead_frames) * step, 0, n - 1)
        target = pol.step_policy(dof[idx], dvel[idx], quats[idx], data.qpos[3:7].copy(),
                                 data.qpos[7:].copy(), data.qvel[6:].copy(),
                                 data.qvel[3:6].copy(), gravity_dir())
        data.ctrl[:] = np.clip(target, model.actuator_ctrlrange[:, 0],
                               model.actuator_ctrlrange[:, 1])
        if push > 0 and t > 0 and t % int(3 * CONTROL_HZ) == 0:
            th = rng.uniform(0, 2 * np.pi)
            data.qvel[0] += push * np.cos(th)
            data.qvel[1] += push * np.sin(th)
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        errs.append(float(np.abs(data.qpos[7:] - dof_hw[t]).mean()))
        heights.append(float(data.qpos[2]))
        alive = t + 1
        if data.qpos[2] < 0.4:
            break
    return {"steps": alive, "seconds": alive / CONTROL_HZ,
            "joint_err": float(np.mean(errs)) if errs else np.nan,
            "height": float(np.mean(heights)) if heights else np.nan,
            "fell": bool(data.qpos[2] < 0.4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion",
                    default="assets/sonic/sample_data/robot_filtered/210531/"
                            "walk_forward_amateur_001__A001.pkl")
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--variant", default="low_latency", choices=["low_latency", "default"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--push", type=float, default=0.0)
    ap.add_argument("--probe", action="store_true",
                    help="try all joint-order / quaternion conventions")
    ap.add_argument("--order", default="isaaclab", choices=["isaaclab", "hardware"])
    ap.add_argument("--quat", default="wxyz", choices=["wxyz", "xyzw"])
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--stock-armature", action="store_true",
                    help="keep the model's uniform 0.010 armature")
    ap.add_argument("--our-clip", default="",
                    help="a .npz from retarget_clips.py instead of a SONIC sample")
    args = ap.parse_args()

    robot = REGISTRY[args.robot]
    if args.our_clip:
        ref = load_our_clip(args.our_clip)
        args.motion = args.our_clip
        if not args.probe:
            args.order, args.quat = "hardware", "wxyz"   # our own convention
    else:
        ref = load_reference(args.motion)
    print(f"reference: {Path(args.motion).name}  {len(ref['dof'])} frames "
          f"@ {ref['fps']:.0f} Hz  ({len(ref['dof'])/ref['fps']:.1f} s)")
    print(f"variant: {args.variant}   lookahead: 10 frames x step "
          f"{1 if args.variant=='low_latency' else 5} = "
          f"{(10 * (1 if args.variant=='low_latency' else 5)) / CONTROL_HZ * 1000:.0f} ms\n")

    combos = ([(o, q) for o in ("isaaclab", "hardware") for q in ("wxyz", "xyzw")]
              if args.probe else [(args.order, args.quat)])
    rows = []
    for order, quat in combos:
        r = rollout(ref, robot, order, quat, args.steps, variant=args.variant,
                    push=args.push, sonic_armature=not args.stock_armature)
        rows.append(((order, quat), r))
        print(f"  dof={order:9s} quat={quat:5s} -> survived {r['steps']:4d}/{args.steps} "
              f"({r['seconds']:5.2f} s)  joint err {r['joint_err']:.3f} rad  "
              f"mean height {r['height']:.3f}{'  FELL' if r['fell'] else ''}")

    best = max(rows, key=lambda kv: (kv[1]["steps"], -kv[1]["joint_err"]))
    if args.probe:
        print(f"\nbest convention: dof={best[0][0]}, quat={best[0][1]}")


if __name__ == "__main__":
    main()
