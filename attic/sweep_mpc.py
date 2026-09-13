#!/usr/bin/env python3
"""Sweep MPC horizon and exploration scale against survival."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_mpc import estimated_state
from prepare_robot_data import resample
from rpc.sim.estimator import LegOdometry, LowState
from rpc.sim.mpc import MPCConfig, PredictiveSamplingMPC
from rpc.sim.robot import REGISTRY


def run(qpos, robot, cfg, steps, fps, start=100):
    m = mujoco.MjModel.from_xml_path(robot.path())
    d = mujoco.MjData(m)
    sub = max(1, int(round(1.0 / (fps * m.opt.timestep))))
    mpc = PredictiveSamplingMPC(m, robot, cfg, substeps=sub)
    est = LegOdometry(m, robot)
    d.qpos[:] = qpos[start]
    d.qvel[:] = 0
    d.qvel[:3] = (qpos[start + 1, :3] - qpos[start, :3]) * fps
    d.qvel[6:] = (qpos[start + 1, 7:] - qpos[start, 7:]) * fps
    mujoco.mj_forward(m, d)
    n = min(steps, len(qpos) - start - cfg.horizon - 2)
    errs, imp, t0 = [], [], time.time()
    scratch = mujoco.MjData(m)
    for t in range(n):
        ref = qpos[start + t: start + t + cfg.horizon]
        s = LowState.from_sim(m, d)
        e = est.update(s, None)
        s0 = estimated_state(m, d, None, s, e, mpc.state_size)
        mujoco.mj_setState(m, scratch, s0, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        ctrl = mpc.step(scratch, ref[:, 7:], ref[:, 2])
        imp.append(mpc.last["improvement"])
        d.ctrl[:] = ctrl
        for _ in range(sub):
            mujoco.mj_step(m, d)
        errs.append(float(np.abs(d.qpos[7:] - qpos[start + t, 7:]).mean()))
        R = d.xmat[1].reshape(3, 3)
        if d.qpos[2] < 0.43 or R[2, 2] < 0.3:
            break
    ms = (time.time() - t0) / (t + 1) * 1e3
    mpc.close()
    return t + 1, n, float(np.mean(errs)), float(np.mean(imp)), ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=["walk1_subject1", "walk3_subject2",
                                                   "walk2_subject4"])
    ap.add_argument("--src", default="data/processed/gmr_corrected/g1")
    ap.add_argument("--horizon", type=int, nargs="*", default=[4, 6, 10, 16])
    ap.add_argument("--sigma", type=float, nargs="*", default=[0.03, 0.08, 0.16, 0.30])
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()

    robot = REGISTRY["g1"]
    refs = {c: resample(np.load(f"{args.src}/{c}.npz"), 50.0)[0] for c in args.clips}
    print(f"{'H':>3s} {'sigma':>6s} | " + " ".join(f"{c[:12]:>13s}" for c in args.clips)
          + f" | {'mean':>6s} {'err':>7s} {'improv':>7s} {'ms':>6s}", flush=True)
    rows = []
    for H, sg in itertools.product(args.horizon, args.sigma):
        cfg = MPCConfig(horizon=H, sigma=sg, n_sample=args.samples)
        out, cells = [], []
        for c in args.clips:
            steps, n, err, imp, ms = run(refs[c], robot, cfg, args.steps, 50.0)
            out.append((steps, err, imp, ms))
            cells.append(f"{steps:5d}/{n:<7d}")
        mean = np.mean([o[0] for o in out])
        rows.append({"horizon": H, "sigma": sg, "mean_steps": mean,
                     "err": float(np.mean([o[1] for o in out])),
                     "improvement": float(np.mean([o[2] for o in out])),
                     "ms": float(np.mean([o[3] for o in out]))})
        print(f"{H:3d} {sg:6.2f} | " + " ".join(cells) +
              f" | {mean:6.0f} {rows[-1]['err']:7.4f} {rows[-1]['improvement']:7.1f} "
              f"{rows[-1]['ms']:6.1f}", flush=True)
    (ROOT / "runs" / "sweep_mpc.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
