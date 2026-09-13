#!/usr/bin/env python3
"""Dataset carrying a per-limb phase manifold.

Replaces the single 20-D whole-body manifold with the concatenation of four
per-limb manifolds (left leg, right leg, left arm, right arm), 8 phase channels
in total.  Measured on this corpus, arm-leg phase coupling ranges from 0.81
while running to 0.13 while aiming, so one shared clock cannot represent both;
per-limb phase can.

Layout per group, matching the whole-body version: (a*cos(phi), a*sin(phi), a, f)
for each of the group's channels, groups concatenated in ``GROUPS`` order.
"""
from __future__ import annotations

import argparse
import glob
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
from prepare_lmp_data import foot_frame
from prepare_robot_data import resample as resample_samples
from train_phase import character_frame_velocity
from train_phase_limb import GROUPS


def load_limb_paes(root, model):
    man = json.loads((Path(root) / "manifest.json").read_text())
    out = []
    for g in GROUPS:
        ck = torch.load(Path(root) / f"pae_{g}.pt", map_location="cpu", weights_only=False)
        net = PeriodicAutoencoder(ck["n_in"], ck["window"], n_phases=ck["channels"],
                                  fps=ck["fps"])
        net.load_state_dict(ck["state"])
        net.eval()
        ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                        for b in ck["bodies"]])
        out.append((g, net, ck, ids))
    return out, man


@torch.no_grad()
def limb_manifold(qpos, paes, model, data, fps, batch=512):
    """(T, 16*n_groups/…) per-limb manifold; groups concatenated in GROUPS order."""
    # Field-major: [all a_cos][all a_sin][all amplitude][all frequency], with
    # channels ordered by group.  The runtime and the trainer both slice the
    # manifold this way, so emitting it group-major here would silently hand
    # amplitudes where phases are expected.
    cos_, sin_, amp_, frq_ = [], [], [], []
    for _, net, ck, ids in paes:
        V = character_frame_velocity(qpos, model, data, ids, fps)
        Vn = ((V - ck["mu"]) / ck["sd"]).astype(np.float32)
        W = ck["window"]
        half = W // 2
        pad = np.concatenate([np.repeat(Vn[:1], half, 0), Vn, np.repeat(Vn[-1:], half, 0)])
        C, S, A, Fq = [], [], [], []
        for i in range(0, len(Vn), batch):
            idx = np.arange(i, min(i + batch, len(Vn)))
            x = np.stack([pad[j:j + W] for j in idx]).transpose(0, 2, 1)
            _, p = net(torch.from_numpy(x))
            a, f, ph = p["amplitude"], p["frequency"], p["phase"]
            C.append((a * torch.cos(ph)).numpy()); S.append((a * torch.sin(ph)).numpy())
            A.append(a.numpy()); Fq.append(f.numpy())
        cos_.append(np.concatenate(C)); sin_.append(np.concatenate(S))
        amp_.append(np.concatenate(A)); frq_.append(np.concatenate(Fq))
    return np.concatenate([np.concatenate(cos_, 1), np.concatenate(sin_, 1),
                           np.concatenate(amp_, 1), np.concatenate(frq_, 1)],
                          axis=1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/gmr_all/g1")
    ap.add_argument("--pae", default="runs/pae_limb")
    ap.add_argument("--out", default="data/processed/robot_limb")
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--holdout", default="subject5")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(REGISTRY[args.robot].path())
    d = mujoco.MjData(m)
    paes, pman = load_limb_paes(args.pae, m)
    feet = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
                     for b in ("left_ankle_roll_link", "right_ankle_roll_link")])

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.npz"))
    cfg = FeatureConfig(fps=args.fps)
    spec = None
    splits = {s: {k: [] for k in ("X", "Y", "P", "M", "F")} for s in ("train", "val")}
    index = []
    for p in files:
        z = np.load(p)
        qpos, ct, ph = resample_samples(z, args.fps)
        if spec is None:
            spec = RobotFeatureSpec(n_joints=qpos.shape[1] - 7, cfg=cfg, n_feet=ct.shape[1])
        f = analyse_robot_clip(qpos, ct, ph, args.fps, cfg, p.stem)
        X, Y, P = build_robot_samples(f, spec)
        n = len(X)
        M = limb_manifold(qpos, paes, m, d, args.fps)[:n]
        F = foot_frame(qpos, m, d, feet, ct)[:n]
        split = "val" if args.holdout and args.holdout in p.stem else "train"
        for k, v in (("X", X), ("Y", Y), ("P", P), ("M", M), ("F", F)):
            splits[split][k].append(v)
        index.append({"name": p.stem, "split": split, "frames": int(n)})
        print(f"  {p.stem:28s} {split:5s} {n:6d}", flush=True)

    n_ch = sum(pman["groups"][g]["n_in"] and pman["channels"] for g in GROUPS)
    man = {"space": "robot_limb", "fps": args.fps, "n_joints": spec.n_joints,
           "n_feet": spec.n_feet, "n_in": spec.n_in, "n_out": spec.n_out,
           "in_blocks": spec.in_blocks, "out_blocks": spec.out_blocks,
           "gaits": list(cfg.gaits), "n_past": cfg.n_past, "n_future": cfg.n_future,
           "span": cfg.span,
           "phase_channels": n_ch,
           "limb_groups": list(GROUPS), "channels_per_group": pman["channels"],
           "manifold_layout": ["a_cos", "a_sin", "amplitude", "frequency"],
           "pae": args.pae, "clips": index}
    for split, dd in splits.items():
        if not dd["X"]:
            continue
        for k in ("X", "Y", "P", "M", "F"):
            arr = np.concatenate(dd[k], axis=0)
            np.save(out / f"{split}_{k}.npy", arr)
            man[f"{split}_{k}_shape"] = list(arr.shape)
        print(f"{split}: X {man[f'{split}_X_shape']}  M {man[f'{split}_M_shape']}")
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"wrote {out}  ({n_ch} phase channels across {len(GROUPS)} limb groups)")


if __name__ == "__main__":
    main()
