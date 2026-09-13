#!/usr/bin/env python3
"""Turn a directory of BVH into the generator's training arrays.

Writes memory-mapped .npy so training does not have to hold the whole corpus in
RAM, plus a manifest describing the feature layout the arrays were built with —
a checkpoint is only interpretable next to the spec that produced it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpc.control.language import library_from_features
from rpc.motion.bvh import parse
from rpc.motion.dataset import FeatureSpec, analyse, build_samples
from rpc.motion.features import FeatureConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/raw/lafan1")
    ap.add_argument("--out", default="data/processed/lafan1")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=0, help="0 = all clips")
    ap.add_argument("--mirror", action="store_true", default=True)
    ap.add_argument("--no-mirror", dest="mirror", action="store_false")
    ap.add_argument("--holdout", default="subject5",
                    help="clips whose name contains this go to the validation split")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(src.glob("*.bvh"))
    if args.limit:
        paths = paths[: args.limit]
    assert paths, f"no .bvh under {src}"

    cfg = FeatureConfig(fps=args.fps)
    spec = None
    splits: dict[str, dict[str, list]] = {
        "train": {"X": [], "Y": [], "P": []},
        "val": {"X": [], "Y": [], "P": []},
    }
    library, clip_index = [], []
    t0 = time.time()

    for i, p in enumerate(paths):
        clip = parse(p).resample(args.fps)
        if spec is None:
            spec = FeatureSpec(n_joints=clip.skeleton.n_joints, cfg=cfg)
            skel = clip.skeleton
        split = "val" if args.holdout and args.holdout in p.stem else "train"

        variants = [clip] + ([clip.mirror()] if args.mirror and split == "train" else [])
        n_here = 0
        for v in variants:
            feats = analyse(v, cfg)
            X, Y, P = build_samples(feats, spec)
            splits[split]["X"].append(X)
            splits[split]["Y"].append(Y)
            splits[split]["P"].append(P)
            n_here += len(X)
            if v is clip:
                library.extend(library_from_features([feats]))
        clip_index.append({"name": p.stem, "split": split, "frames": n_here})
        print(f"[{i+1:3d}/{len(paths)}] {p.stem:26s} {split:5s} {n_here:7d} samples", flush=True)

    manifest = {
        "fps": args.fps,
        "n_joints": spec.n_joints,
        "joint_names": list(skel.names),
        "parents": skel.parents.tolist(),
        "offsets": skel.offsets.tolist(),
        "fwd": skel.fwd.tolist(),
        "up": skel.up.tolist(),
        "n_in": spec.n_in,
        "n_out": spec.n_out,
        "in_blocks": spec.in_blocks,
        "out_blocks": spec.out_blocks,
        "gaits": list(cfg.gaits),
        "n_past": cfg.n_past,
        "n_future": cfg.n_future,
        "span": cfg.span,
        "mirrored": args.mirror,
        "clips": clip_index,
        "library": [{"name": e.name, "labels": list(e.labels), "gait": e.gait.tolist()}
                    for e in library],
    }

    for split, d in splits.items():
        if not d["X"]:
            continue
        for key in ("X", "Y", "P"):
            arr = np.concatenate(d[key], axis=0)
            np.save(out / f"{split}_{key}.npy", arr)
            manifest[f"{split}_{key}_shape"] = list(arr.shape)
        print(f"{split}: {manifest[f'{split}_X_shape']}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
