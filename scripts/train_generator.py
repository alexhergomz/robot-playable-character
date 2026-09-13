#!/usr/bin/env python3
"""Train the kinematic generator (PFNN or MANN) on prepared arrays."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpc.generator.pfnn import (DEFAULT_LOSS_WEIGHTS, MANN, PFNN,
                                PHYSICAL_LOSS_WEIGHTS, ROBOT_LOSS_WEIGHTS,
                                loss_terms, physical_loss)
from rpc.motion.dataset import FeatureSpec, Normalizer
from rpc.motion.features import FeatureConfig


def load_split(root: Path, split: str):
    return tuple(np.load(root / f"{split}_{k}.npy", mmap_mode="r") for k in ("X", "Y", "P"))


def spec_from_manifest(man: dict):
    """Human-space or robot-space, decided by the manifest that built the data."""
    cfg = FeatureConfig(fps=man["fps"], n_past=man["n_past"],
                        n_future=man["n_future"], span=man["span"],
                        gaits=tuple(man["gaits"]))
    if man.get("space") == "robot":
        from rpc.motion.robot_dataset import RobotFeatureSpec
        return RobotFeatureSpec(n_joints=man["n_joints"], cfg=cfg,
                                n_feet=man.get("n_feet", 2))
    return FeatureSpec(n_joints=man["n_joints"], cfg=cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/lafan1")
    ap.add_argument("--out", default="runs/generator")
    ap.add_argument("--arch", choices=["pfnn", "mann"], default="pfnn")
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--experts", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=2.5e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--physical-loss", action="store_true",
                    help="loss in radians/metres over the consumed outputs only")
    ap.add_argument("--input-noise", type=float, default=0.15,
                    help="std (in normalised units) added to the autoregressive "
                         "input blocks; cures exposure bias in the rollout")
    args = ap.parse_args()

    root = Path(args.data)
    man = json.loads((root / "manifest.json").read_text())
    spec = spec_from_manifest(man)
    Xtr, Ytr, Ptr = load_split(root, "train")
    Xva, Yva, Pva = load_split(root, "val")
    if args.max_samples:
        Xtr, Ytr, Ptr = Xtr[: args.max_samples], Ytr[: args.max_samples], Ptr[: args.max_samples]

    dev = torch.device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Normalisers on a subsample: the full array is a memmap and the statistics
    # converge long before we have read all of it.
    sub = np.random.default_rng(0).choice(len(Xtr), min(200_000, len(Xtr)), replace=False)
    sub.sort()
    xn = Normalizer.fit(np.asarray(Xtr[sub], dtype=np.float32))
    yn = Normalizer.fit(np.asarray(Ytr[sub], dtype=np.float32))

    if args.arch == "pfnn":
        model = PFNN(spec.n_in, spec.n_out, hidden=args.hidden,
                     n_control=args.experts, dropout=args.dropout)
        gate_slice = None
    else:
        isl = spec.input_slices
        gate_slice = np.r_[isl["traj_pos"], isl["traj_dir"], isl["traj_gait"]]
        model = MANN(spec.n_in, spec.n_out, gate_dim=len(gate_slice), hidden=args.hidden,
                     n_experts=args.experts, dropout=args.dropout)
    model.to(dev)
    model.set_normalizers(xn, yn)
    print(f"{args.arch.upper()}  params {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"| train {len(Xtr):,} val {len(Xva):,} | device {dev}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, len(Xtr) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr * 10, total_steps=args.epochs * steps_per_epoch, pct_start=0.25)
    osl = spec.output_slices
    isl = spec.input_slices
    # In robot space the predicted pose *is* the actuator command, so it is
    # weighted far above the cosmetic terms.
    weights = ROBOT_LOSS_WEIGHTS if man.get("space") == "robot" else DEFAULT_LOSS_WEIGHTS
    # Only the blocks the controller feeds back to itself get noise.  The
    # trajectory window is the operator's, and perturbing it would teach the
    # model to ignore the command.
    noise_cols = torch.zeros(spec.n_in, device=dev)
    for blk in ("joint_pos", "joint_vel", "root_height", "gravity",
                "root_lin_vel", "root_ang_vel"):
        if blk in isl:
            noise_cols[isl[blk]] = 1.0

    def batch(X, Y, P, idx):
        idx = np.sort(idx)
        x = torch.from_numpy(np.asarray(X[idx], dtype=np.float32)).to(dev)
        y = torch.from_numpy(np.asarray(Y[idx], dtype=np.float32)).to(dev)
        p = torch.from_numpy(np.asarray(P[idx], dtype=np.float32)).to(dev)
        xn_ = (x - model.x_mean) / model.x_std
        yn_ = (y - model.y_mean) / model.y_std
        aux = p if args.arch == "pfnn" else xn_[:, gate_slice]
        return xn_, yn_, aux

    rng = np.random.default_rng(0)
    history = []
    best = float("inf")
    for ep in range(args.epochs):
        model.train()
        perm = rng.permutation(len(Xtr))
        tot, t0 = 0.0, time.time()
        for s in range(steps_per_epoch):
            xb, yb, aux = batch(Xtr, Ytr, Ptr, perm[s * args.batch:(s + 1) * args.batch])
            if args.input_noise > 0:
                xb = xb + torch.randn_like(xb) * noise_cols * args.input_noise
            pred = model(xb, aux, normalized=True)
            losses = (physical_loss(pred, yb, model.y_std, osl, PHYSICAL_LOSS_WEIGHTS)
                      if args.physical_loss else loss_terms(pred, yb, osl, weights))
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(losses["total"])

        model.eval()
        vidx = rng.choice(len(Xva), min(40_000, len(Xva)), replace=False)
        with torch.no_grad():
            xb, yb, aux = batch(Xva, Yva, Pva, vidx)
            pred = model(xb, aux, normalized=True)
            vl = (physical_loss(pred, yb, model.y_std, osl, PHYSICAL_LOSS_WEIGHTS)
                  if args.physical_loss else loss_terms(pred, yb, osl, weights))
            # de-normalised, physically meaningful errors
            den = pred * model.y_std + model.y_mean
            tgt = yb * model.y_std + model.y_mean
            jp = (den[:, osl["joint_pos"]] - tgt[:, osl["joint_pos"]])
            jp = jp.reshape(len(den), -1, 3) if jp.shape[1] % 3 == 0 and man.get("space") != "robot" \
                else jp[..., None]
            rd = (den[:, osl["root_delta"]] - tgt[:, osl["root_delta"]])
            mpjpe = float(jp.norm(dim=-1).mean()) * 1000   # mm, or mrad in robot space
            vel_err = float(rd[:, :2].norm(dim=-1).mean()) * man["fps"]
            yaw_err = float(rd[:, 2].abs().mean()) * man["fps"]

        rec = {"epoch": ep, "train": tot / steps_per_epoch, "val": float(vl["total"]),
               "mpjpe_mm": mpjpe, "root_vel_err_mps": vel_err, "yaw_err_rps": yaw_err,
               "secs": time.time() - t0}
        history.append(rec)
        print(f"ep {ep:2d}  train {rec['train']:.4f}  val {rec['val']:.4f}  "
              f"MPJPE {mpjpe:6.1f} mm  root-vel err {vel_err:.3f} m/s  "
              f"yaw err {yaw_err:.3f} rad/s  ({rec['secs']:.0f}s)", flush=True)

        if rec["val"] < best:
            best = rec["val"]
            torch.save({"arch": args.arch, "state": model.state_dict(), "manifest": man,
                        "args": vars(args), "gate_slice": None if gate_slice is None
                        else gate_slice.tolist()}, out / "generator.pt")
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"best val {best:.4f} -> {out/'generator.pt'}")


if __name__ == "__main__":
    main()
