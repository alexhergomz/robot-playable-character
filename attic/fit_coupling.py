#!/usr/bin/env python3
"""Fit inter-limb phase coupling from the corpus.

Per-limb phase buys limb independence and pays for it: with four clocks
advancing on their own predicted frequencies, a 0.038 Hz difference between the
legs integrated to 2921 degrees of relative drift over 16 s, the gait lost its
antiphase and the robot curved.  A single global clock cannot drift, but it also
cannot let an arm come loose from the gait, which is the whole point of per-limb
phase.

What the corpus actually shows is neither: measured limb phase locking is 0.56
between legs during locomotion and 0.13 between arm and leg while aiming.  So
coupling is real, varies by limb pair, and should be measured rather than
imposed or omitted.

Each limb is modelled as a phase oscillator pulled by the others,

    dphi_i/dt = 2*pi*f_i + sum_j K_ij sin(phi_j - phi_i - psi_ij),

and since K sin(d - psi) = a sin(d) + b cos(d), the unknowns enter linearly and
the fit is least squares.  A pair with no coupling in the data simply returns
K ~ 0, so nothing has to decide in advance which limbs should be coupled.
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
from train_phase_limb import GROUPS, resample


def load(root, model):
    man = json.loads((Path(root) / "manifest.json").read_text())
    out = []
    for g in GROUPS:
        ck = torch.load(Path(root) / f"pae_{g}.pt", map_location="cpu", weights_only=False)
        net = PeriodicAutoencoder(ck["n_in"], ck["window"], n_phases=ck["channels"],
                                  fps=ck["fps"])
        net.load_state_dict(ck["state"]); net.eval()
        ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                        for b in ck["bodies"]])
        out.append((g, net, ck, ids))
    return out, man


@torch.no_grad()
def phase_track(q, paes, model, data, fps, stride=2):
    """Dominant-channel phase and frequency per limb, per frame."""
    ph, fr = {}, {}
    for g, net, ck, ids in paes:
        V = character_frame_velocity(q, model, data, ids, fps)
        Vn = ((V - ck["mu"]) / ck["sd"]).astype(np.float32)
        W = ck["window"]; half = W // 2
        idx = np.arange(half, len(Vn) - half, stride)
        X = np.stack([Vn[i - half:i + half + 1] for i in idx]).transpose(0, 2, 1)
        Pp, Aa, Ff = [], [], []
        for i in range(0, len(X), 512):
            _, p = net(torch.from_numpy(X[i:i + 512]))
            Pp.append(p["phase"].numpy()); Aa.append(p["amplitude"].numpy())
            Ff.append(p["frequency"].numpy())
        Pp, Aa, Ff = np.concatenate(Pp), np.concatenate(Aa), np.concatenate(Ff)
        k = int(np.argmax(Aa.mean(0)))
        ph[g], fr[g] = Pp[:, k], Ff[:, k]
    return ph, fr, stride / fps


def fit(ph, fr, dt, names):
    """Least-squares K, psi for every ordered limb pair."""
    n = len(names)
    K = np.zeros((n, n)); PSI = np.zeros((n, n)); R2 = np.zeros(n)
    for i, gi in enumerate(names):
        d = np.diff(np.unwrap(ph[gi])) / dt
        resid = d - 2 * np.pi * fr[gi][:-1]
        cols, pairs = [], []
        for j, gj in enumerate(names):
            if i == j:
                continue
            delta = ph[gj][:-1] - ph[gi][:-1]
            cols += [np.sin(delta), np.cos(delta)]
            pairs.append(j)
        A = np.stack(cols, 1)
        sol, *_ = np.linalg.lstsq(A, resid, rcond=None)
        pred = A @ sol
        ss = ((resid - pred) ** 2).sum()
        R2[i] = 1 - ss / max(((resid - resid.mean()) ** 2).sum(), 1e-12)
        for t, j in enumerate(pairs):
            a, b = sol[2 * t], sol[2 * t + 1]
            K[i, j] = float(np.hypot(a, b))
            PSI[i, j] = float(np.arctan2(-b, a))
    return K, PSI, R2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/processed/gmr_all/g1")
    ap.add_argument("--pae", default="runs/pae_limb")
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--max-frames", type=int, default=4000)
    ap.add_argument("--out", default="runs/pae_limb/coupling.json")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(REGISTRY[args.robot].path())
    d = mujoco.MjData(m)
    paes, _ = load(args.pae, m)
    names = list(GROUPS)

    sets = {"locomotion": ("walk", "run", "sprint"),
            "action": ("fight", "aiming", "multipleActions", "fightAndSports")}
    results = {}
    for label, pats in sets.items():
        P_, F_ = {g: [] for g in names}, {g: [] for g in names}
        for pat in pats:
            for f in sorted(glob.glob(f"{args.clips}/{pat}*.npz"))[:4]:
                q = resample(np.load(f), args.fps)[:args.max_frames]
                ph, fr, dt = phase_track(q, paes, m, d, args.fps)
                for g in names:
                    P_[g].append(ph[g]); F_[g].append(fr[g])
        ph = {g: np.concatenate(P_[g]) for g in names}
        fr = {g: np.concatenate(F_[g]) for g in names}
        K, PSI, R2 = fit(ph, fr, dt, names)
        results[label] = {"K": K.tolist(), "psi": PSI.tolist(), "r2": R2.tolist()}
        print(f"\n{label}: coupling strength K[i][j] (rad/s of pull on i from j)")
        print("            " + "".join(f"{g:>9s}" for g in names))
        for i, g in enumerate(names):
            print(f"  {g:9s} " + "".join("        ." if i == j else f"{K[i,j]:9.2f}"
                                         for j in range(len(names)))
                  + f"    R2 {R2[i]:.2f}")
        print("  preferred offset psi[i][j], degrees (180 = antiphase)")
        for i, g in enumerate(names):
            print(f"  {g:9s} " + "".join("        ." if i == j else
                                         f"{np.degrees(PSI[i,j])%360:9.0f}"
                                         for j in range(len(names))))
    Path(args.out).write_text(json.dumps({"groups": names, **results}, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
