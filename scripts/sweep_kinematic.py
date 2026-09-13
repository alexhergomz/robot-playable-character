#!/usr/bin/env python3
"""Sweep the regularisation and capacity that shape the animation.

The open-loop rollout showed the generator is stable but produces a damped,
chattering pose: 42% of the training data's positional variance with 2-5x its
frame-to-frame speed.  Four candidates were named; this tests them against the
two numbers that matter, matched to the training data:

    speed_med  -> chatter   (target: the data's ~0.81 rad/s)
    pose_std   -> damping   (target: the data's ~0.31 rad)
"""
from __future__ import annotations

import os
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import mujoco

from eval_kinematic import load, measure
from rpc.sim.robot import REGISTRY

PY = os.environ.get("RPC_PYTHON", sys.executable)


def train(tag, noise, dropout, hidden, epochs, data, extra=()):
    out = ROOT / "runs" / f"sweep_{tag}"
    if (out / "generator.pt").exists():
        return out
    cmd = [PY, "-u", str(ROOT / "scripts" / "train_generator.py"),
           "--data", data, "--out", str(out), "--epochs", str(epochs),
           "--dropout", str(dropout), "--input-noise", str(noise),
           "--hidden", str(hidden), "--batch", "2048", *extra]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/robot_gmr")
    ap.add_argument("--noise", type=float, nargs="*", default=[0.0, 0.05, 0.15, 0.25])
    ap.add_argument("--dropout", type=float, nargs="*", default=[0.3, 0.5, 0.7])
    ap.add_argument("--hidden", type=int, nargs="*", default=[512])
    ap.add_argument("--epochs", type=int, nargs="*", default=[90])
    ap.add_argument("--speeds", type=float, nargs="*", default=[0.0, 0.6, 1.2, 1.8])
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()

    mj = mujoco.MjModel.from_xml_path(REGISTRY["g1"].path())
    refY = np.asarray(np.load(Path(args.data) / "train_Y.npy", mmap_mode="r")[:8000, :29])

    print(f"{'noise':>6s} {'drop':>5s} {'hid':>5s} {'ep':>4s} | "
          f"{'speed_med':>9s} {'p99':>7s} {'max':>7s} | {'pose_std':>8s} | "
          f"{'contact_ms':>10s} {'gain':>5s} {'lim%':>5s}", flush=True)
    print(f"{'target':>6s} {'':>5s} {'':>5s} {'':>4s} | "
          f"{np.median(np.abs(np.diff(refY,axis=0))*50):9.2f} {'~7.6':>7s} {'~17.6':>7s} | "
          f"{refY.std(0).mean():8.3f} | {'~600':>10s} {'1.00':>5s} {'0':>5s}", flush=True)

    rows = []
    for n, d, h, e in itertools.product(args.noise, args.dropout, args.hidden, args.epochs):
        tag = f"n{n}_d{d}_h{h}_e{e}"
        try:
            out = train(tag, n, d, h, e, args.data)
            model, spec, _ = load(str(out / "generator.pt"))
            m = measure(model, spec, mj, args.speeds, args.seconds, refY)
        except Exception as exc:                       # keep the sweep going
            print(f"{n:6.2f} {d:5.2f} {h:5d} {e:4d} | FAILED {exc}", flush=True)
            continue
        rows.append((n, d, h, e, m))
        print(f"{n:6.2f} {d:5.2f} {h:5d} {e:4d} | "
              f"{m['speed_med']:9.2f} {m['speed_p99']:7.1f} {m['speed_max']:7.1f} | "
              f"{m['pose_std']:8.3f} | {m['contact_ms']:10.0f} {m['gain']:5.2f} "
              f"{m['limit_viol']:5.2f}{'  DIVERGED' if m['diverged'] else ''}", flush=True)

    (ROOT / "runs" / "sweep_kinematic.json").write_text(json.dumps(
        [{"noise": n, "dropout": d, "hidden": h, "epochs": e, **m} for n, d, h, e, m in rows],
        indent=1))


if __name__ == "__main__":
    main()
