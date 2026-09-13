#!/usr/bin/env python3
"""Is the generator a good animator?

Open-loop, no simulator: the network feeds itself for the whole rollout.  This
is the test PFNN was designed to pass and it is separate from whether anything
can execute the result.  Four things are checked:

* **divergence** - do joint angles stay inside the robot's limits and do
  velocities stay bounded, or does the autoregressive loop blow up
* **on-manifold** - is the generated pose distribution the one it was trained on
* **gait coherence** - do the feet alternate at a plausible stride frequency
* **command response** - does commanded speed produce proportional motion
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpc.control.command import Command
from rpc.generator.kinematic_runtime import KinematicCharacterController
from rpc.generator.pfnn import PFNN
from rpc.motion.features import FeatureConfig
from rpc.motion.robot_dataset import RobotFeatureSpec
from rpc.sim.robot import REGISTRY


def load(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    man, a = ck["manifest"], ck["args"]
    cfg = FeatureConfig(fps=man["fps"], n_past=man["n_past"], n_future=man["n_future"],
                        span=man["span"], gaits=tuple(man["gaits"]))
    spec = RobotFeatureSpec(man["n_joints"], cfg, man.get("n_feet", 2))
    m = PFNN(spec.n_in, spec.n_out, hidden=a["hidden"], n_control=a["experts"],
             dropout=a["dropout"])
    m.load_state_dict(ck["state"])
    return m, spec, man


def rollout(ctrl, spec, speeds, seconds_each):
    frames, cmds = [], []
    for sp in speeds:
        for _ in range(int(seconds_each * spec.cfg.fps)):
            c = Command(move=np.array([sp, 0.0]), responsiveness=0.6)
            frames.append(ctrl.step(c))
            cmds.append(sp)
    return frames, np.array(cmds)


def measure(model, spec, mj_model, speeds, seconds, ref_Y=None):
    """Roll out open-loop and score the animation.  Shared by the sweep."""
    lo, hi = mj_model.jnt_range[1:spec.n_joints + 1].T
    ctrl = KinematicCharacterController(model, spec)
    ctrl.reset(root_height=0.75)
    frames, cmds = rollout(ctrl, spec, speeds, seconds)
    fps = spec.cfg.fps
    jp = np.stack([f.joint_pos for f in frames])
    h = np.array([f.root_height for f in frames])
    ct = np.stack([f.contacts for f in frames]) > 0.5
    rd = np.stack([f.root_delta for f in frames])
    jv = np.abs(np.diff(jp, axis=0)) * fps

    flips = np.flatnonzero(np.diff(ct[:, 0].astype(int)) != 0)
    runlen = float(np.median(np.diff(flips))) / fps * 1000 if len(flips) > 2 else 0.0
    gains = []
    for sp in sorted(set(speeds)):
        if sp <= 0:
            continue
        m_ = cmds == sp
        gains.append(np.linalg.norm(rd[m_][:, :2], axis=-1).mean() * fps / sp)
    out = {
        "speed_med": float(np.median(jv)), "speed_p99": float(np.percentile(jv, 99)),
        "speed_max": float(jv.max()), "pose_std": float(jp.std(0).mean()),
        "limit_viol": float(((jp < lo) | (jp > hi)).mean() * 100),
        "height_min": float(h.min()), "height_max": float(h.max()),
        "contact_ms": runlen, "gain": float(np.mean(gains)) if gains else np.nan,
        "diverged": bool(not np.isfinite(jp).all() or jp.std(0).mean() > 2.0),
    }
    if ref_Y is not None:
        rv = np.abs(np.diff(ref_Y, axis=0)) * fps
        out["ref_speed_med"] = float(np.median(rv))
        out["ref_pose_std"] = float(ref_Y.std(0).mean())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/pfnn_gmr/generator.pt")
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--data", default="data/processed/robot_gmr")
    ap.add_argument("--speeds", type=float, nargs="*", default=[0.0, 0.6, 1.2, 1.8, 0.6])
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    import mujoco

    model, spec, man = load(args.model)
    robot = REGISTRY[args.robot]
    mj = mujoco.MjModel.from_xml_path(robot.path())
    lo, hi = mj.jnt_range[1:spec.n_joints + 1].T

    ctrl = KinematicCharacterController(model, spec)
    ctrl.reset(root_height=0.75)
    frames, cmds = rollout(ctrl, spec, args.speeds, args.seconds)
    fps = spec.cfg.fps

    jp = np.stack([f.joint_pos for f in frames])
    jv = np.stack([f.joint_vel for f in frames])
    h = np.array([f.root_height for f in frames])
    ct = np.stack([f.contacts for f in frames])
    rd = np.stack([f.root_delta for f in frames])

    print(f"open-loop rollout: {len(frames)} frames = {len(frames)/fps:.1f} s at {fps:.0f} Hz\n")

    print("DIVERGENCE")
    out_lo = (jp < lo).mean() * 100
    out_hi = (jp > hi).mean() * 100
    print(f"  joint angles outside the robot's limits: {out_lo+out_hi:5.2f}% of joint-frames")
    print(f"  |joint velocity|  mean {np.abs(jv).mean():6.3f}  p99 {np.percentile(np.abs(jv),99):6.3f}"
          f"  max {np.abs(jv).max():7.3f} rad/s")
    print(f"  root height       mean {h.mean():.3f}  min {h.min():.3f}  max {h.max():.3f} m")
    drift = np.abs(jp[-int(2*fps):].mean(0) - jp[:int(2*fps)].mean(0)).mean()
    print(f"  mean |pose(last 2 s) - pose(first 2 s)| = {drift:.4f} rad  (drift, not divergence, if small)")

    print("\nON-MANIFOLD")
    Y = np.load(Path(args.data) / "train_Y.npy", mmap_mode="r")
    sl = spec.output_slices["joint_pos"]
    ref = np.asarray(Y[::37, sl])
    print(f"  generated joint std {jp.std(0).mean():.4f} rad   training data {ref.std(0).mean():.4f} rad")
    print(f"  generated joint mean-abs-offset from training mean: "
          f"{np.abs(jp.mean(0) - ref.mean(0)).mean():.4f} rad")

    print("\nGAIT")
    b = ct > 0.5
    flips = np.diff(b[:, 0].astype(int))
    strides = np.flatnonzero(flips > 0)
    if len(strides) > 2:
        print(f"  left-foot contact onsets: {len(strides)} in {len(frames)/fps:.0f} s "
              f"-> {len(strides)/(len(frames)/fps):.2f} Hz")
    else:
        print(f"  left-foot contact onsets: {len(strides)}  (no clear gait cycle)")
    print(f"  contact fraction L/R {b[:,0].mean():.2f}/{b[:,1].mean():.2f}   "
          f"double {(b.sum(1)==2).mean():.2f}  flight {(b.sum(1)==0).mean():.2f}")

    print("\nCOMMAND RESPONSE")
    for sp in sorted(set(args.speeds)):
        m_ = cmds == sp
        got = np.linalg.norm(rd[m_][:, :2], axis=-1).mean() * fps
        print(f"  commanded {sp:4.1f} m/s -> achieved {got:5.2f} m/s")

    if args.save:
        q = np.stack([f.qpos(spec.n_joints) for f in frames])
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save, qpos=q.astype(np.float32),
                            contacts=ct.astype(np.float32),
                            phase=np.array([f.phase for f in frames], np.float32), fps=fps)
        print(f"\nsaved {args.save}")


if __name__ == "__main__":
    main()
