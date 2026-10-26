#!/usr/bin/env python3
"""Bundle everything the streamer needs into one file.

The robot side must not depend on the training datasets: the streamer used to
read its initial phase state out of a multi-gigabyte dataset by frame index,
which is fragile (indices mean different clips in different datasets) and
meant deployment broke the day the datasets were cleared.

The bundle holds the generator's weights and layout, the amplitude curve, and
one initial state -- the phase manifold and foot frame taken mid-stride from a
named walking clip, recomputed from the raw capture so it does not need any
processed data.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from rpc.motion.contacts import contacts_from_model
from rpc.sim.robot import REGISTRY
from prepare_lmp_data import foot_frame, load_pae, manifold
from train_phase import BODIES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="runs/lmp_nogait/generator.pt")
    ap.add_argument("--pae", default="runs/pae/pae.pt")
    ap.add_argument("--clip", default="walk1_subject1")
    ap.add_argument("--out", default="runs/deploy/bundle.pt")
    ap.add_argument("--fps", type=float, default=50.0)
    args = ap.parse_args()

    ck = torch.load(ROOT / args.gen, map_location="cpu", weights_only=False)
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(ROOT / "scripts/retarget_gmr.py"),
                        "--match", args.clip, "--out", tmp, "--max-frames", "3000"],
                       check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
        z = np.load(Path(tmp) / f"{args.clip}.npz")
        q, src = z["qpos"].astype(np.float64), float(z["fps"])
    t0 = np.arange(len(q)) / src
    t1 = np.arange(0.0, t0[-1], 1.0 / args.fps)
    q = np.stack([np.interp(t1, t0, q[:, c]) for c in range(q.shape[1])], 1)
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)

    m = mujoco.MjModel.from_xml_path(REGISTRY["g1"].path()); d = mujoco.MjData(m)
    net, pck = load_pae(str(ROOT / args.pae))
    ids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in BODIES])
    M = manifold(q, net, pck, m, d, ids, args.fps)
    feet = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
                     for b in ("left_ankle_roll_link", "right_ankle_roll_link")])
    F = foot_frame(q, m, d, feet, contacts_from_model(m, d, q).astype(np.float32))
    k = len(q) // 2
    curve = ck.get("amplitude_curve")
    if curve is None:                        # older checkpoints kept it in a json
        curve = json.loads((ROOT / "rpc/control/amplitude_curve.json").read_text())
    bundle = {k2: ck[k2] for k2 in ("state", "args", "manifest", "out_slices",
                                    "n_in", "n_out") if k2 in ck}
    bundle.update(keep_cols=ck.get("keep_cols"), amplitude_curve=curve,
                  init_manifold=M[k].astype(np.float32), init_foot=F[k].astype(np.float32),
                  provenance={"generator": args.gen, "pae": args.pae,
                              "init_clip": args.clip, "init_frame": int(k)})
    out = ROOT / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)
    print(f"wrote {args.out}  ({out.stat().st_size/1e6:.1f} MB)  init: {args.clip} frame {k}")


if __name__ == "__main__":
    main()
