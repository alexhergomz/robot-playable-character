#!/usr/bin/env python3
"""Retarget LAFAN1 to the G1 with GMR instead of our hand-tuned scaling.

GMR (Ze et al., ICRA 2026, MIT) is purpose-built for this: BVH human motion to
diverse humanoids, task-space IK through `mink` with a QP solver, scaled from
the actor's measured height rather than a set of per-chain ratios chosen by
hand.  Our own retargeter accumulated seven distinct bugs of the form
"geometry-dependent quantity transferred with the wrong ratio"; a library that
does this for a living is the right dependency.

Output is written in the same .npz layout as `retarget_clips.py`, so every
measurement and evaluation script applies to both unchanged.

Verified before use: GMR's G1 has the same 29 joints in the same order and the
same qpos layout as the Menagerie model, and its joint limits are *narrower* on
all 15 joints where they differ — so its output is always valid in our model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "external" / "GMR"))

import mujoco

from rpc.motion.phase import contact_phase
from rpc.motion.contacts import com_and_feet, contacts_from_model
from rpc.sim.robot import REGISTRY


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/lafan1")
    ap.add_argument("--out", default="data/processed/gmr/g1")
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--match", nargs="*", default=["walk", "run", "sprint"])
    ap.add_argument("--max-frames", type=int, default=6000)
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting.utils.lafan1 import load_bvh_file

    spec = REGISTRY[args.robot]
    model = mujoco.MjModel.from_xml_path(spec.path())
    data = mujoco.MjData(model)
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in spec.foot_bodies]
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    sole = float(np.mean([data.xpos[b][2] for b in feet]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = [p for p in sorted(Path(args.src).glob("*.bvh"))
             if not args.match or any(k in p.stem for k in args.match)]
    print(f"GMR retargeting {len(paths)} clips -> {spec.name}")

    for n, p in enumerate(paths):
        frames, human_height = load_bvh_file(str(p), format="lafan1")
        rt = GMR(src_human="bvh_lafan1", tgt_robot="unitree_g1",
                 actual_human_height=human_height, verbose=False)
        T = min(args.max_frames, len(frames))
        qpos = np.zeros((T, model.nq))
        for i in range(T):
            # GMR already calls apply_ground_offset internally; passing
            # offset_to_ground=True grounds a second time and lifts the robot.
            qpos[i] = rt.retarget(frames[i])

        # Ground on the robot's sole.  GMR registers the human's ankle, which
        # sits ~6 cm up; the robot's ankle body sits `sole` above its own sole.
        #
        # A geometry-based replacement was tried and rejected on measurement.
        # Taking the mode of the foot's signed distance to the floor looks more
        # principled, but the retargeted clips have no crisp ground level --
        # support distance is a broad spread from -4 cm to +0.5 cm -- so the
        # mode is dominated by rolling and swinging frames and registers the
        # robot ~1 cm too high, giving walking clips 0.46-0.69 flight fraction.
        # Conditioning on genuine stance sharpens it but drifts up to 0.8 cm
        # with the stance cut, in opposite directions on different clips.
        # Against a stance-conditioned reading this rule is off by ~1.4 mm,
        # which is smaller than the spread of any estimator tried.
        _, fpos = com_and_feet(model, data, qpos, feet)
        support = fpos[..., 2].min(axis=1)
        qpos[:, 2] -= np.percentile(support, 40) - sole
        _, fpos = com_and_feet(model, data, qpos, feet)
        # contact from the model's own collision geometry, not a
        # sole-offset margin and speed threshold
        contacts = contacts_from_model(model, data, qpos).astype(np.float32)
        phase = contact_phase(contacts, args.fps).astype(np.float32)

        np.savez_compressed(out / f"{p.stem}.npz", qpos=qpos.astype(np.float32),
                            contacts=contacts, phase=phase, fps=args.fps)
        low = float(np.percentile(fpos[..., 2].min(axis=1), 50))
        print(f"[{n+1:3d}/{len(paths)}] {p.stem:24s} {T:6d} frames  height {human_height:.2f} m  "
              f"support-foot {low*100:5.2f} cm  flight {1-contacts.any(1).mean():.2f}", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
