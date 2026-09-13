#!/usr/bin/env python3
"""Make retargeted clips physically executable, offline.

Reads the kinematic retargets, recomputes the root trajectory so the ZMP stays
over the footprints, re-plants the feet, and writes corrected clips.  Purely
deterministic: one sparse least-squares solve and a leg IK pass per clip.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpc.sim.dynamics_fix import DynamicsFixConfig, com_and_feet, correct
from rpc.sim.robot import REGISTRY


def zmp_excursion(model, data, qpos, contacts, foot_bids, half, fps):
    com, feet = com_and_feet(model, data, qpos, foot_bids)
    acc = np.gradient(np.gradient(com, axis=0), axis=0) * fps * fps
    zmp = com[:, :2] - com[:, 2][:, None] * acc[:, :2] / np.maximum(acc[:, 2] + 9.81, 1e-3)[:, None]
    on = feet[:, :, 2] < (np.percentile(feet[:, :, 2], 20) + 0.04)
    dist = np.full(len(qpos), np.inf)
    for k in range(len(foot_bids)):
        dist = np.where(on[:, k],
                        np.minimum(dist, np.linalg.norm(zmp - feet[:, k, :2], axis=-1) - half[k]),
                        dist)
    v = dist[np.isfinite(dist)]
    return float(np.median(v) * 100), float(np.percentile(v, 90) * 100)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/retargeted/g1")
    ap.add_argument("--out", default="data/processed/corrected/g1")
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--smooth", type=float, default=12.0)
    ap.add_argument("--zmp", type=float, default=60.0)
    args = ap.parse_args()

    spec = REGISTRY[args.robot]
    model = mujoco.MjModel.from_xml_path(spec.path())
    data = mujoco.MjData(model)
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in spec.foot_bodies]
    half = [max((float(model.geom_rbound[g]) for g in range(model.ngeom)
                 if model.geom_bodyid[g] == b), default=0.05) for b in feet]
    cfg = DynamicsFixConfig(smooth=args.smooth, zmp=args.zmp)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in sorted(Path(args.src).glob("*.npz")):
        z = np.load(p)
        qpos, ct, ph, fps = (z["qpos"].astype(np.float64), z["contacts"],
                             z["phase"], float(z["fps"]))
        b0 = zmp_excursion(model, data, qpos, ct, feet, half, fps)
        fixed, info = correct(model, qpos, ct, feet, fps, cfg)
        b1 = zmp_excursion(model, data, fixed, ct, feet, half, fps)
        np.savez_compressed(out / p.name, qpos=fixed.astype(np.float32),
                            contacts=ct.astype(np.float32),
                            phase=ph.astype(np.float32), fps=fps)
        print(f"  {p.stem:26s} ZMP median {b0[0]:+6.1f} -> {b1[0]:+6.1f} cm   "
              f"p90 {b0[1]:+7.1f} -> {b1[1]:+7.1f}   root {info['shift_cm']:5.1f} cm  "
              f"feet {info['foot_move_cm']:.2f} cm  flight {info['flight_frac']*100:.0f}%",
              flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
