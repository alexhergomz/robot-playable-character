#!/usr/bin/env python3
"""Pre-retarget mocap clips onto the robot and cache them.

Tracker training replays these, so the cost is paid once.  The cache also makes
retargeting quality auditable independently of the policy: if the robot cannot
follow a clip, look here before blaming the reward.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpc.motion.bvh import parse
from rpc.motion.dataset import analyse
from rpc.motion.features import FeatureConfig
from rpc.sim.retarget import Retargeter, RetargetConfig
from rpc.sim.robot import REGISTRY


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/lafan1")
    ap.add_argument("--out", default="data/processed/retargeted")
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--match", nargs="*", default=["walk", "run", "sprint"])
    ap.add_argument("--max-frames", type=int, default=6000, help="per clip, 0 = all")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--ori-gain", type=float, default=0.3,
                    help="how much of the human trunk rotation to reproduce; "
                         "1.0 is faithful and wrong -- the G1 carries 23%% of its "
                         "mass in the torso and a physics-corrected reference "
                         "shows ~0.3x the human's waist excursion")
    ap.add_argument("--lock-feet", action="store_true", default=True,
                    help="pin the stance foot to remove mocap foot skate")
    ap.add_argument("--no-lock-feet", dest="lock_feet", action="store_false")
    args = ap.parse_args()

    spec = REGISTRY[args.robot]
    out = Path(args.out) / args.robot
    out.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(spec.path())

    paths = [p for p in sorted(Path(args.src).glob("*.bvh"))
             if not args.match or any(k in p.stem for k in args.match)]
    print(f"retargeting {len(paths)} clips onto {spec.name}")

    index = []
    for n, p in enumerate(paths):
        clip = parse(p).resample(args.fps)
        feats = analyse(clip, FeatureConfig(fps=args.fps))
        T = len(feats.joint_pos) if not args.max_frames else min(args.max_frames,
                                                                 len(feats.joint_pos))
        rt = Retargeter(model, spec, clip.skeleton,
                        RetargetConfig(iters=args.iters, ori_gain=args.ori_gain),
                        motion_positions=feats.joint_pos)

        qpos = np.zeros((T, model.nq))
        contacts = feats.contacts
        foot_bids = rt.foot_bids
        locked = None          # (foot index, world xy the foot is pinned to)
        t0 = time.time()
        # World root path from the clip's own root deltas, scaled by the leg
        # ratio.  Unscaled, the robot's body travels the *human's* distance while
        # its 0.767x legs can only reach 76.7% as far, so the stance foot has to
        # slide to keep up — measured at 63 cm/s against 0.4 cm/s for a
        # physically consistent reference.
        leg = rt.scale["leg"]
        x, y, yaw = 0.0, 0.0, 0.0
        for i in range(T):
            q = rt.solve(feats.joint_pos[i], yaw, np.array([x, y]),
                         root_rot6d=feats.joint_rot6d[i, 0],
                         joint_rot6d=feats.joint_rot6d[i])

            # Foot locking.  The IK is solved relative to the root, so shifting
            # the root translates the feet rigidly — which means the stance foot
            # can be pinned exactly, with no re-solve and no change to the pose.
            # Without this the reference inherits the mocap's foot skate (~60 cm/s
            # of stance-foot motion in LAFAN1) and no tracker can follow it,
            # because the reference is a body gliding over the ground.
            if args.lock_feet:
                stance = np.flatnonzero(contacts[i] > 0.5)
                if len(stance):
                    rt.data.qpos[:] = q
                    mujoco.mj_kinematics(model, rt.data)
                    k = int(stance[0]) if locked is None or locked[0] not in stance \
                        else locked[0]
                    here = rt.data.xpos[foot_bids[k]][:2].copy()
                    if locked is not None and locked[0] == k:
                        drift = here - locked[1]
                        x -= float(drift[0])
                        y -= float(drift[1])
                        q[0] -= float(drift[0])
                        q[1] -= float(drift[1])
                    else:
                        locked = (k, here)
                else:
                    locked = None
            qpos[i] = q

            d = feats.root_delta[i]
            c, s = np.cos(yaw), np.sin(yaw)
            x += (c * d[0] - s * d[1]) * leg
            y += (s * d[0] + c * d[1]) * leg
            yaw += d[2]
        res = rt.residuals(qpos[-1], feats.joint_pos[T - 1], yaw, np.array([x, y]))

        np.savez_compressed(out / f"{p.stem}.npz", qpos=qpos.astype(np.float32),
                            contacts=feats.contacts[:T].astype(np.float32),
                            phase=feats.phase[:T].astype(np.float32), fps=args.fps)
        index.append({"name": p.stem, "frames": int(T),
                      "foot_residual_cm": round(float(np.mean(
                          [res[k] for k in res if "Foot" in k])) * 100, 2)})
        print(f"[{n+1:3d}/{len(paths)}] {p.stem:24s} {T:6d} frames  "
              f"{time.time()-t0:5.1f}s  foot err {index[-1]['foot_residual_cm']:.2f} cm",
              flush=True)

    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {len(index)} clips to {out}")


if __name__ == "__main__":
    main()
