#!/usr/bin/env python3
"""Does per-limb phase actually decouple the limbs?

The mechanism claim behind per-limb phase is that one limb can be redirected
without disturbing the others.  Independence at the phase-extraction stage is
trivial -- each per-limb autoencoder only ever sees its own group's channels --
so the claim is only testable once a generator is conditioned on the joint
manifold and predicts the whole body from it.

The test rotates one group's phase by pi (the strongest redirect a phase
channel can express) and measures how far each limb's predicted joint
trajectory moves away from an unperturbed rollout of the same generator.  A
decoupled representation gives a diagonal matrix; a shared clock gives a dense
one.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rpc.control.command import Command
from rpc.generator.lmp import LMPGenerator
from rpc.generator.lmp_runtime import LMPCharacterController
from rpc.motion.features import FeatureConfig
from rpc.motion.robot_dataset import RobotFeatureSpec
from rpc.sim import sonic as P

#: Joint indices per limb, in MuJoCo order (waist excluded: it is shared).
LIMB_JOINTS = {
    "l_leg": list(range(0, 6)),
    "r_leg": list(range(6, 12)),
    "l_arm": list(range(15, 22)),
    "r_arm": list(range(22, 29)),
}


def build(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    man, a = ck["manifest"], ck["args"]
    C = man["phase_channels"]
    cfg = FeatureConfig(fps=man["fps"], n_past=man["n_past"], n_future=man["n_future"],
                        span=man["span"], gaits=tuple(man["gaits"]))
    spec = RobotFeatureSpec(man["n_joints"], cfg, man["n_feet"])
    net = LMPGenerator(ck["n_in"], ck["n_out"], gate_dim=4 * C, hidden=a["hidden"],
                       n_experts=a["experts"], dropout=a["dropout"])
    net.load_state_dict(ck["state"])
    return net, spec, ck, man, C


def rollout(net, spec, ck, man, C, data, perturb=None, steps=500, hold=200,
            device="cpu", speed=1.0):
    M = np.load(Path(data) / "train_M.npy", mmap_mode="r")
    F = np.load(Path(data) / "train_F.npy", mmap_mode="r")
    g = LMPCharacterController(net, spec, ck["out_slices"], 4 * C, n_feet=man["n_feet"],
                               keep_cols=ck.get("keep_cols"),
                               curve=ck.get("amplitude_curve"), device=device)
    g.reset(joint_pos=P.DEFAULT_ANGLES.copy(), root_height=0.76,
            manifold=M[60007].copy(), foot=F[60007].copy())
    c = Command(move=np.array([speed, 0.0]), responsiveness=0.6)
    q = []
    for t in range(steps):
        if perturb is not None and t == hold:
            # rotate only this group's channels by pi
            m = g.manifold.copy()
            sl = perturb
            ang = np.arctan2(m[C:2 * C][sl], m[:C][sl]) + np.pi
            amp = m[2 * C:3 * C][sl]
            m[:C][sl] = amp * np.cos(ang)
            m[C:2 * C][sl] = amp * np.sin(ang)
            g.manifold = m
        q.append(g.step(c).joint_pos.copy())
    return np.stack(q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="runs/lmp_limb/generator.pt")
    ap.add_argument("--data", default="data/processed/robot_limb")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--hold", type=int, default=200)
    args = ap.parse_args()

    net, spec, ck, man, C = build(args.gen)
    groups = man.get("limb_groups")
    per = man.get("channels_per_group")
    if not groups:
        print("this generator has a single whole-body phase; per-limb perturbation "
              "is not defined for it")
        return
    base = rollout(net, spec, ck, man, C, args.data, None, device=args.device)
    seg = slice(args.hold, args.hold + 100)          # the 2 s after the redirect
    print(f"Per-limb phase redirect: mean |change| in each limb's joints, rad")
    print(f"  generator {Path(args.gen).parent.name}, {len(groups)} groups x {per} channels\n")
    print(f"  {'perturbed':10s}" + "".join(f"{g:>9s}" for g in LIMB_JOINTS) + "     ratio")
    for i, gname in enumerate(groups):
        sl = slice(i * per, (i + 1) * per)
        q = rollout(net, spec, ck, man, C, args.data, sl, hold=args.hold,
                    device=args.device)
        row = {}
        for lname, idx in LIMB_JOINTS.items():
            row[lname] = np.abs(q[seg][:, idx] - base[seg][:, idx]).mean()
        own = row.get(gname, np.nan)
        others = np.mean([v for k, v in row.items() if k != gname])
        print(f"  {gname:10s}" + "".join(f"{row[l]:9.4f}" for l in LIMB_JOINTS)
              + f"   {own/max(others,1e-9):6.2f}x")
    print("\n  ratio = change in the perturbed limb / mean change in the others.")
    print("  A decoupled representation gives ratios well above 1.")


if __name__ == "__main__":
    main()
