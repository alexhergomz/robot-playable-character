#!/usr/bin/env python3
"""Render a reference trajectory to video.

Stills cannot show gait.  A posture metric that looked decisive on two frames
turned out to predict nothing (correlation +0.05 with trackability), so motion
gets judged as motion.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpc.sim.robot import REGISTRY


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--start", type=int, default=300)
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=float, default=0, help="0 = the clip's own rate")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--azimuth", type=float, default=120)
    ap.add_argument("--elevation", type=float, default=-10)
    ap.add_argument("--distance", type=float, default=3.0)
    args = ap.parse_args()

    spec = REGISTRY[args.robot]
    model = mujoco.MjModel.from_xml_path(spec.path())
    data = mujoco.MjData(model)
    z = np.load(args.clip)
    qpos = z["qpos"].astype(np.float64)
    fps = args.fps or float(z["fps"])

    n = min(int(args.seconds * fps), len(qpos) - args.start)
    rend = mujoco.Renderer(model, args.height, args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation, cam.distance = args.azimuth, args.elevation, args.distance

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(args.out, fps=int(fps), macro_block_size=1) as w:
        for k in range(n):
            data.qpos[:] = qpos[args.start + k]
            data.qvel[:] = 0
            mujoco.mj_forward(model, data)
            cam.lookat[:] = qpos[args.start + k][:3]
            cam.lookat[2] = 0.8
            rend.update_scene(data, cam)
            w.append_data(rend.render())
    print(f"{args.out}  {n} frames @ {fps:.0f} fps  ({n/fps:.1f} s)")


if __name__ == "__main__":
    main()
