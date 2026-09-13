#!/usr/bin/env python3
"""Render a reference trajectory to a contact sheet.

Numbers say whether a statistic moved; they do not say whether the motion looks
like walking.  Several defects in this project were mis-attributed for exactly
that reason, so every reference change should be looked at as well as measured.

Needs a headless GL backend: ``MUJOCO_GL=egl``.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpc.sim.robot import REGISTRY


def sheet(model, qpos, idx, cols, w, h, azimuth, elevation, distance, track_root):
    data = mujoco.MjData(model)
    rend = mujoco.Renderer(model, h, w)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
    tiles = []
    for i in idx:
        data.qpos[:] = qpos[i]
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        cam.lookat[:] = (qpos[i][:3] if track_root else qpos[idx[0]][:3])
        cam.lookat[2] = 0.8
        rend.update_scene(data, cam)
        img = rend.render().copy()
        img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = 90     # tile border
        tiles.append(img)
    rows = int(np.ceil(len(tiles) / cols))
    while len(tiles) < rows * cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.concatenate([np.concatenate(tiles[r * cols:(r + 1) * cols], axis=1)
                           for r in range(rows)], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--start", type=int, default=200)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--size", type=int, nargs=2, default=[300, 260])
    ap.add_argument("--azimuth", type=float, default=105)
    ap.add_argument("--elevation", type=float, default=-8)
    ap.add_argument("--distance", type=float, default=2.6)
    ap.add_argument("--no-track", action="store_true")
    args = ap.parse_args()

    spec = REGISTRY[args.robot]
    model = mujoco.MjModel.from_xml_path(spec.path())
    z = np.load(args.clip)
    qpos = z["qpos"].astype(np.float64)
    idx = [min(args.start + k * args.stride, len(qpos) - 1) for k in range(args.frames)]
    img = sheet(model, qpos, idx, args.cols, args.size[0], args.size[1],
                args.azimuth, args.elevation, args.distance, not args.no_track)

    try:
        from PIL import Image
        Image.fromarray(img).save(args.out)
    except ImportError:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(args.out, img)
    print(f"{args.out}  frames {idx[0]}..{idx[-1]}  {img.shape[1]}x{img.shape[0]}")


if __name__ == "__main__":
    main()
