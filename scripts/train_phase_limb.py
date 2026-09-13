#!/usr/bin/env python3
"""Per-limb phase: one periodic autoencoder per body group.

A single whole-body periodic autoencoder was measured on this corpus and does
not separate the limbs: of five channels only one (the step-frequency channel
at ~2.6 Hz) is limb-specific, at 91% legs, and even that dissolves on fight
clips where every channel goes mixed.  Independent per-limb phase therefore has
to be built rather than harvested, which is what the part-wise phase literature
does.

Four groups are fitted separately -- left leg, right leg, left arm, right arm --
so that one limb's phase can be advanced or redirected without touching the
others.  The pelvis is excluded: ``character_frame_velocity`` subtracts the root
position and the pelvis *is* the root, so its three channels are identically
zero and carry nothing.
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

from rpc.motion.phase import PeriodicAutoencoder
from rpc.sim.robot import REGISTRY
from train_phase import character_frame_velocity

#: Body groups, each fitted by its own periodic autoencoder.
GROUPS = {
    "l_leg": ("left_hip_pitch_link", "left_knee_link", "left_ankle_roll_link"),
    "r_leg": ("right_hip_pitch_link", "right_knee_link", "right_ankle_roll_link"),
    "l_arm": ("left_shoulder_pitch_link", "left_elbow_link", "left_wrist_yaw_link"),
    "r_arm": ("right_shoulder_pitch_link", "right_elbow_link", "right_wrist_yaw_link"),
}


def resample(z, fps):
    q, src = z["qpos"], float(z["fps"])
    if abs(src - fps) < 1e-6:
        return q
    t0 = np.arange(len(q)) / src
    t1 = np.arange(0.0, t0[-1], 1.0 / fps)
    qi = np.stack([np.interp(t1, t0, q[:, c]) for c in range(q.shape[1])], 1)
    qi[:, 3:7] /= np.linalg.norm(qi[:, 3:7], axis=1, keepdims=True)
    return qi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/processed/gmr_all/g1")
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--out", default="runs/pae_limb")
    ap.add_argument("--window", type=int, default=61)
    ap.add_argument("--channels", type=int, default=2, help="phase channels per group")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(REGISTRY[args.robot].path())
    d = mujoco.MjData(m)
    files = sorted(glob.glob(args.clips + "/*.npz"))
    assert files, args.clips
    qs = [resample(np.load(f), args.fps) for f in files]
    print(f"{len(files)} clips, {sum(len(q) for q in qs)} frames at {args.fps} Hz")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"groups": {}, "window": args.window, "channels": args.channels,
                "fps": args.fps, "clips": [Path(f).stem for f in files]}

    for name, bodies in GROUPS.items():
        ids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies])
        assert (ids >= 0).all(), bodies
        seqs = [character_frame_velocity(q, m, d, ids, args.fps) for q in qs]
        allv = np.concatenate(seqs)
        mu, sd = allv.mean(0), allv.std(0) + 1e-6
        W = args.window
        wins = []
        for s in seqs:
            s = (s - mu) / sd
            idx = np.arange(0, len(s) - W, 2)
            wins.append(np.stack([s[i:i + W] for i in idx]))
        X = np.concatenate(wins).transpose(0, 2, 1).astype(np.float32)
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(X))
        nv = len(X) // 10
        val, tr = X[perm[:nv]], X[perm[nv:]]

        net = PeriodicAutoencoder(X.shape[1], W, n_phases=args.channels, fps=args.fps)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
        tr_t, val_t = torch.from_numpy(tr), torch.from_numpy(val)
        for ep in range(args.epochs):
            net.train()
            p = torch.randperm(len(tr_t))
            for i in range(0, len(p), args.batch):
                b = tr_t[p[i:i + args.batch]]
                y, _ = net(b)
                loss = torch.nn.functional.mse_loss(y, b)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sch.step()
        net.eval()
        with torch.no_grad():
            vy, vp = net(val_t)
            vl = torch.nn.functional.mse_loss(vy, val_t).item()
            f = vp["frequency"].mean(0).numpy()
        torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "window": W,
                    "channels": args.channels, "fps": args.fps,
                    "bodies": list(bodies), "n_in": X.shape[1]},
                   out / f"pae_{name}.pt")
        manifest["groups"][name] = {"bodies": list(bodies), "n_in": int(X.shape[1]),
                                    "val": vl, "freq": [float(x) for x in f]}
        print(f"  {name:6s} {X.shape[1]:2d} ch in, {len(tr)} windows  "
              f"val {vl:.4f}  freq {np.array2string(f, precision=2)}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
