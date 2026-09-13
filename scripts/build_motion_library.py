#!/usr/bin/env python3
"""The motion library CLIP indexes: labels and targets taken from the clips.

The previous language channel mapped an utterance onto a gait one-hot and then
converted that to a speed through ``GAIT_SPEED`` -- a hand-typed dictionary
({"walk": 1.3, "run": 3.2, ...}).  Nothing checked those numbers against the
motions they claimed to describe.

Here each clip contributes what it actually is: its measured mean root speed,
its measured turn rate, and its measured phase manifold (per-channel amplitude
and frequency from the periodic autoencoder).  "Run" then means whatever the
run clips measurably do, and adding a clip to the corpus updates the meaning
of the word for free.

Labels come from the clip filename, which LAFAN1 encodes the action into.
For a corpus without them this is the one place a VLM captioner substitutes;
nothing downstream changes.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rpc.sim.robot import REGISTRY
from prepare_lmp_data import load_pae, manifold
from train_phase import BODIES

# LAFAN1 encodes the action as a leading word plus an index: walk3_subject5.
_LABEL = re.compile(r"^([a-zA-Z]+)")


def labels_for(stem: str) -> tuple[str, ...]:
    m = _LABEL.match(stem)
    base = m.group(1).lower() if m else stem.lower()
    # a couple of natural synonyms so the operator is not forced to guess the
    # corpus's exact vocabulary; these are words, not tuned numbers
    syn = {"walk": ("walk", "walking"), "run": ("run", "running", "jog"),
           "sprint": ("sprint", "sprinting", "run fast"),
           "dance": ("dance", "dancing"), "fight": ("fight", "fighting")}
    return syn.get(base, (base,))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/processed/gmr_corrected/g1")
    ap.add_argument("--pae", default="runs/pae/pae.pt")
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--out", default="data/processed/motion_library.json")
    args = ap.parse_args()

    net, ck = load_pae(args.pae)
    m = mujoco.MjModel.from_xml_path(REGISTRY[args.robot].path())
    d = mujoco.MjData(m)
    ids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in BODIES])
    C = ck["channels"]

    lib = []
    for f in sorted(glob.glob(args.clips + "/*.npz")):
        z = np.load(f)
        q, src = z["qpos"], float(z["fps"])
        if abs(src - args.fps) > 1e-6:
            t0 = np.arange(len(q)) / src
            t1 = np.arange(0.0, t0[-1], 1.0 / args.fps)
            q = np.stack([np.interp(t1, t0, q[:, c]) for c in range(q.shape[1])], 1)
            q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)

        v = np.linalg.norm(np.diff(q[:, :2], axis=0), axis=-1) * args.fps
        v = np.convolve(v, np.ones(int(args.fps)) / int(args.fps), "same")
        w, x, y, zc = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
        yaw = np.unwrap(np.arctan2(2 * (w * zc + x * y), 1 - 2 * (y * y + zc * zc)))
        turn = np.abs(np.diff(yaw)) * args.fps

        M = manifold(q, net, ck, m, d, ids, args.fps)
        amp, frq = M[:, 2 * C:3 * C], M[:, 3 * C:]
        # moving frames only: a clip's identity is what it does while moving
        # A clip's identity is what it does while *performing* the action, not
        # the idle around it.  LAFAN1's sprint clips are mostly standing with
        # short bursts: taking the median over "moving" frames rated sprint at
        # 0.61 m/s, slower than walk.  Threshold relative to the clip's own
        # peak instead -- scale-free, so it adapts to each clip's dynamics.
        mv = v > 0.5 * np.percentile(v, 95)           # over v (length T-1)
        mvM = np.concatenate([mv, [False]])[:len(M)]  # same mask over M (length T)
        stem = Path(f).stem
        lib.append({
            "name": stem,
            "labels": list(labels_for(stem)),
            "speed": float(np.median(v[mv])),
            "speed_p90": float(np.percentile(v, 90)),
            "turn": float(np.median(turn)),
            "amplitude": [float(a) for a in amp[mvM].mean(0)],
            "frequency": [float(a) for a in frq[mvM].mean(0)],
        })
        print(f"  {stem:24s} {','.join(lib[-1]['labels']):22s} "
              f"speed {lib[-1]['speed']:4.2f}  freq {np.mean(lib[-1]['frequency']):4.2f} Hz")

    json.dump({"channels": C, "fps": args.fps, "entries": lib},
              open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}  ({len(lib)} clips)")
    for lab in sorted({l for e in lib for l in e["labels"]}):
        sel = [e for e in lib if lab in e["labels"]]
        print(f"  {lab:12s} n={len(sel)}  median speed {np.median([e['speed'] for e in sel]):4.2f} m/s"
              f"  freq {np.mean([np.mean(e['frequency']) for e in sel]):4.2f} Hz")


if __name__ == "__main__":
    main()
