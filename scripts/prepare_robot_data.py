#!/usr/bin/env python3
"""Build PFNN training arrays in the robot's own state space.

Source is the retargeted clips, resampled to the control rate.  Unlike the
human-space pipeline there is no skeleton and no retargeting downstream: what
comes out of the network here is what goes into the actuators.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpc.motion.features import FeatureConfig
from rpc.motion.robot_dataset import (RobotFeatureSpec, analyse_robot_clip,
                                      build_robot_samples)


def resample(z, fps_out: float):
    qpos, ct, ph, fps = z["qpos"].astype(np.float64), z["contacts"], z["phase"], float(z["fps"])
    if abs(fps_out - fps) < 1e-6:
        return qpos, ct, ph
    n = int(len(qpos) * fps_out / fps)
    src = np.linspace(0, len(qpos) - 1, n)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, len(qpos) - 1)
    a = (src - i0)[:, None]
    q = qpos[i0] * (1 - a) + qpos[i1] * a
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=-1, keepdims=True)
    return q, ct[i0] * (1 - a) + ct[i1] * a, ph[i0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/retargeted/g1")
    ap.add_argument("--out", default="data/processed/robot_g1")
    ap.add_argument("--fps", type=float, default=50.0, help="control rate")
    ap.add_argument("--holdout", default="subject5")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.npz"))
    assert files, f"no retargeted clips in {src}"

    cfg = FeatureConfig(fps=args.fps)
    spec = None
    splits = {"train": {"X": [], "Y": [], "P": []}, "val": {"X": [], "Y": [], "P": []}}
    index = []

    for p in files:
        z = np.load(p)
        qpos, ct, ph = resample(z, args.fps)
        if spec is None:
            spec = RobotFeatureSpec(n_joints=qpos.shape[1] - 7, cfg=cfg,
                                    n_feet=ct.shape[1])
        f = analyse_robot_clip(qpos, ct, ph, args.fps, cfg, p.stem)
        X, Y, P = build_robot_samples(f, spec)
        split = "val" if args.holdout and args.holdout in p.stem else "train"
        splits[split]["X"].append(X)
        splits[split]["Y"].append(Y)
        splits[split]["P"].append(P)
        index.append({"name": p.stem, "split": split, "frames": int(len(X))})
        print(f"  {p.stem:26s} {split:5s} {len(X):6d}", flush=True)

    man = {"space": "robot", "fps": args.fps, "n_joints": spec.n_joints,
           "n_feet": spec.n_feet, "n_in": spec.n_in, "n_out": spec.n_out,
           "in_blocks": spec.in_blocks, "out_blocks": spec.out_blocks,
           "gaits": list(cfg.gaits), "n_past": cfg.n_past, "n_future": cfg.n_future,
           "span": cfg.span, "clips": index}
    for split, d in splits.items():
        if not d["X"]:
            continue
        for k in ("X", "Y", "P"):
            arr = np.concatenate(d[k], axis=0)
            np.save(out / f"{split}_{k}.npy", arr)
            man[f"{split}_{k}_shape"] = list(arr.shape)
        print(f"{split}: {man[f'{split}_X_shape']}")
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
