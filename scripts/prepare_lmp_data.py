#!/usr/bin/env python3
"""Stage 2: dataset carrying the phase manifold, not a scalar phase.

Adds two arrays alongside the existing X/Y/P:

* ``M`` (T, 4*C) -- the phase manifold per frame: for each PAE channel,
  (a*cos(phi), a*sin(phi), amplitude, frequency).  The first two are the
  wrap-free coordinates the network gates on; frequency is in Hz, which is
  what lets the runtime advance phase as 2*pi*f*dt instead of accumulating a
  predicted per-frame delta forever.

* ``F`` (T, n_feet, 4) -- per-foot (x, y, z, contact) in the character frame.
  The measured defect is foot-skate: the generator slides the planted foot at
  ~10x the rate of its own training data, because no term in the objective has
  ever seen "the stance foot must not move".  This array is what lets one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rpc.motion.features import FeatureConfig
from rpc.motion.phase import PeriodicAutoencoder
from rpc.motion.robot_dataset import (RobotFeatureSpec, analyse_robot_clip,
                                      build_robot_samples)
from rpc.sim.robot import REGISTRY
from prepare_robot_data import resample
from train_phase import BODIES, character_frame_velocity


def load_pae(path: str):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = PeriodicAutoencoder(ck["n_in"], ck["window"], n_phases=ck["channels"],
                              fps=ck["fps"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck


@torch.no_grad()
def manifold(qpos, net, ck, model, data, ids, fps, batch=512):
    """(T, 4C) phase manifold per frame, edges held rather than extrapolated."""
    V = character_frame_velocity(qpos, model, data, ids, fps)
    Vn = ((V - ck["mu"]) / ck["sd"]).astype(np.float32)
    W = ck["window"]
    half = W // 2
    pad = np.concatenate([np.repeat(Vn[:1], half, 0), Vn, np.repeat(Vn[-1:], half, 0)])
    out = []
    for i in range(0, len(Vn), batch):
        idx = np.arange(i, min(i + batch, len(Vn)))
        x = np.stack([pad[j:j + W] for j in idx]).transpose(0, 2, 1)
        _, p = net(torch.from_numpy(x))
        a, f, ph = p["amplitude"], p["frequency"], p["phase"]
        out.append(torch.cat([a * torch.cos(ph), a * torch.sin(ph), a, f], -1).numpy())
    return np.concatenate(out).astype(np.float32)


def foot_frame(qpos, model, data, feet_ids, contacts):
    """(T, n_feet, 4): foot position in the yaw-aligned character frame + contact."""
    T = len(qpos)
    pos = np.empty((T, len(feet_ids), 3))
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        pos[t] = data.xpos[feet_ids]
    w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(-yaw), np.sin(-yaw)
    rel = pos - qpos[:, None, :3]
    loc = np.stack([c[:, None] * rel[..., 0] - s[:, None] * rel[..., 1],
                    s[:, None] * rel[..., 0] + c[:, None] * rel[..., 1],
                    rel[..., 2]], -1)
    return np.concatenate([loc, contacts[:, :, None].astype(np.float32)], -1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/gmr_corrected/g1")
    ap.add_argument("--pae", default="runs/pae/pae.pt")
    ap.add_argument("--out", default="data/processed/robot_lmp")
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--holdout", default="subject5")
    args = ap.parse_args()

    net, ck = load_pae(args.pae)
    assert abs(ck["fps"] - args.fps) < 1e-6, f"PAE trained at {ck['fps']} Hz, data at {args.fps}"

    m = mujoco.MjModel.from_xml_path(REGISTRY[args.robot].path())
    d = mujoco.MjData(m)
    ids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in BODIES])
    feet = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
                     for b in ("left_ankle_roll_link", "right_ankle_roll_link")])

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.npz"))
    assert files, f"no clips in {src}"

    cfg = FeatureConfig(fps=args.fps)
    spec = None
    splits = {s: {k: [] for k in ("X", "Y", "P", "M", "F")} for s in ("train", "val")}
    index = []

    for p in files:
        z = np.load(p)
        qpos, ct, ph = resample(z, args.fps)
        if spec is None:
            spec = RobotFeatureSpec(n_joints=qpos.shape[1] - 7, cfg=cfg, n_feet=ct.shape[1])
        f = analyse_robot_clip(qpos, ct, ph, args.fps, cfg, p.stem)
        X, Y, P = build_robot_samples(f, spec)
        n = len(X)
        M = manifold(qpos, net, ck, m, d, ids, args.fps)[:n]
        F = foot_frame(qpos, m, d, feet, ct)[:n]
        split = "val" if args.holdout and args.holdout in p.stem else "train"
        for k, v in (("X", X), ("Y", Y), ("P", P), ("M", M), ("F", F)):
            splits[split][k].append(v)
        index.append({"name": p.stem, "split": split, "frames": int(n)})
        print(f"  {p.stem:26s} {split:5s} {n:6d}", flush=True)

    man = {"space": "robot_lmp", "fps": args.fps, "n_joints": spec.n_joints,
           "n_feet": spec.n_feet, "n_in": spec.n_in, "n_out": spec.n_out,
           "in_blocks": spec.in_blocks, "out_blocks": spec.out_blocks,
           "gaits": list(cfg.gaits), "n_past": cfg.n_past, "n_future": cfg.n_future,
           "span": cfg.span, "phase_channels": ck["channels"],
           "manifold_layout": ["a_cos", "a_sin", "amplitude", "frequency"],
           "pae": args.pae, "clips": index}
    for split, dd in splits.items():
        if not dd["X"]:
            continue
        for k in ("X", "Y", "P", "M", "F"):
            arr = np.concatenate(dd[k], axis=0)
            np.save(out / f"{split}_{k}.npy", arr)
            man[f"{split}_{k}_shape"] = list(arr.shape)
        print(f"{split}: X {man[f'{split}_X_shape']}  M {man[f'{split}_M_shape']}  "
              f"F {man[f'{split}_F_shape']}")
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
