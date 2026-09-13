#!/usr/bin/env python3
"""Stage 1 of the successor: learn the phase manifold instead of integrating it.

The measured failure of the scalar-phase generator was that `dphase` is a
free regression output which the runtime integrates open-loop, so a systematic
under-prediction compounds with nothing to correct it -- the gait ran at half
the trained cadence and the stride stretched to 121 cm.

DeepPhase (Starke et al. 2022) removes that degree of freedom.  A periodic
autoencoder is fitted to windows of Cartesian joint velocity in the character
frame; the only way it can reconstruct the window is to make its M latent
curves genuinely sinusoidal.  Each channel then reports (amplitude, frequency,
phase), and frequency is in Hz -- a grounded quantity read off the motion, not
a per-frame delta accumulated forever.

The M channels are local: different channels lock to different body parts and
different periodicities, which is what lets one limb be redirected without
resetting the rest (Local Motion Phases, Starke et al. 2021).
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

from rpc.motion.phase import PeriodicAutoencoder
from rpc.sim.robot import REGISTRY

# Bodies whose motion carries the gait.  Feet and hands dominate; knees and
# elbows disambiguate stance from swing; pelvis anchors the frame.
BODIES = ("pelvis",
          "left_hip_pitch_link", "left_knee_link", "left_ankle_roll_link",
          "right_hip_pitch_link", "right_knee_link", "right_ankle_roll_link",
          "left_shoulder_pitch_link", "left_elbow_link", "left_wrist_yaw_link",
          "right_shoulder_pitch_link", "right_elbow_link", "right_wrist_yaw_link")


def character_frame_velocity(qpos: np.ndarray, model, data, ids, fps: float) -> np.ndarray:
    """(T, 3B) Cartesian body velocity expressed in the yaw-aligned root frame."""
    T = len(qpos)
    pos = np.empty((T, len(ids), 3))
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        pos[t] = data.xpos[ids]
    # de-rotate by root yaw and de-translate by root position, so periodicity
    # is not swamped by forward travel
    w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(-yaw), np.sin(-yaw)
    rel = pos - qpos[:, None, :3]
    local = np.stack([c[:, None] * rel[..., 0] - s[:, None] * rel[..., 1],
                      s[:, None] * rel[..., 0] + c[:, None] * rel[..., 1],
                      rel[..., 2]], axis=-1)
    vel = np.gradient(local, 1.0 / fps, axis=0)
    return vel.reshape(T, -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/processed/gmr_corrected/g1")
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--out", default="runs/pae")
    ap.add_argument("--window", type=int, default=61)      # 1.22 s at 50 Hz
    ap.add_argument("--channels", type=int, default=5)
    ap.add_argument("--fps", type=float, default=50.0,
                    help="resample clips to the generator's control rate")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(REGISTRY[args.robot].path())
    d = mujoco.MjData(m)
    ids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in BODIES])
    assert (ids >= 0).all(), [b for b, i in zip(BODIES, ids) if i < 0]

    files = sorted(glob.glob(args.clips + "/*.npz"))
    seqs, fps = [], args.fps
    for f in files:
        z = np.load(f)
        q, src = z["qpos"], float(z["fps"])
        if abs(src - fps) > 1e-6:      # clips are 30 Hz; the generator runs at 50
            t_src = np.arange(len(q)) / src
            t_dst = np.arange(0.0, t_src[-1], 1.0 / fps)
            qi = np.empty((len(t_dst), q.shape[1]))
            for c in range(q.shape[1]):
                qi[:, c] = np.interp(t_dst, t_src, q[:, c])
            n = np.linalg.norm(qi[:, 3:7], axis=1, keepdims=True)
            qi[:, 3:7] /= np.clip(n, 1e-8, None)      # renormalise the quaternion
            q = qi
        seqs.append(character_frame_velocity(q, m, d, ids, fps))
        print(f"  {Path(f).stem:24s} {len(seqs[-1]):5d} frames")

    C = seqs[0].shape[1]
    mu = np.concatenate(seqs).mean(0)
    sd = np.concatenate(seqs).std(0) + 1e-6

    W = args.window
    wins = []
    for s in seqs:
        s = (s - mu) / sd
        idx = np.arange(0, len(s) - W, 2)
        wins.append(np.stack([s[i:i + W] for i in idx]))
    X = np.concatenate(wins).transpose(0, 2, 1).astype(np.float32)   # (N, C, W)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    n_val = len(X) // 10
    val, tr = X[perm[:n_val]], X[perm[n_val:]]
    print(f"\n{C} channels  {len(tr)} train / {len(val)} val windows of {W} @ {fps} Hz")

    net = PeriodicAutoencoder(C, W, n_phases=args.channels, fps=fps)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    tr_t, val_t = torch.from_numpy(tr), torch.from_numpy(val)

    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(tr_t))
        tot = 0.0
        for i in range(0, len(tr_t), args.batch):
            b = tr_t[perm[i:i + args.batch]]
            y, _ = net(b)
            loss = torch.nn.functional.mse_loss(y, b)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        net.eval()
        with torch.no_grad():
            vy, vp = net(val_t)
            vl = torch.nn.functional.mse_loss(vy, val_t).item()
            f = vp["frequency"].mean(0)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:3d}  train {tot/len(tr_t):.4f}  val {vl:.4f}  "
                  f"chan freq {np.array2string(f.numpy(), precision=2)}")

    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd,
                "window": W, "channels": args.channels, "fps": fps,
                "bodies": list(BODIES), "n_in": C}, args.out + "/pae.pt")
    json.dump({"window": W, "channels": args.channels, "fps": fps,
               "bodies": list(BODIES)}, open(args.out + "/manifest.json", "w"), indent=2)
    print(f"\nwrote {args.out}/pae.pt")


if __name__ == "__main__":
    main()
