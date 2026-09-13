#!/usr/bin/env python3
"""Track a reference with predictive-sampling MPC over the pose lookahead.

The rollouts start from the *estimated* state, not the simulator's: base height
and velocity from leg odometry, attitude from the IMU, joints from encoders.
Horizontal position is set to zero because the robot cannot measure it and the
cost never refers to it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rpc.control.command import Command
from rpc.generator.kinematic_runtime import KinematicCharacterController
from rpc.sim.estimator import LegOdometry, LowState
from rpc.sim.mpc import MPCConfig, PredictiveSamplingMPC
from rpc.sim.robot import REGISTRY


def estimated_state(model, data, est_data, state, est, size):
    """Initial rollout state built only from what the robot can measure."""
    s = np.zeros(size)
    mujoco.mj_getState(model, data, s, mujoco.mjtState.mjSTATE_FULLPHYSICS)
    nq = model.nq
    s[1:3] = 0.0                       # base xy: unobservable, and unused by the cost
    s[3] = est.height                  # leg odometry
    s[4:8] = state.quaternion          # IMU
    s[8:1 + nq] = state.q              # encoders
    s[1 + nq:1 + nq + 3] = est.lin_vel     # leg odometry
    s[1 + nq + 3:1 + nq + 6] = state.gyroscope
    s[1 + nq + 6:1 + nq + model.nv] = state.dq
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="")
    ap.add_argument("--pfnn", default="")
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--fps", type=float, default=50.0)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--start", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=6)
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--sigma", type=float, default=0.08)
    ap.add_argument("--speeds", type=float, nargs="*", default=[0.6, 1.2])
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    robot = REGISTRY[args.robot]
    if args.pfnn:
        from eval_kinematic import load
        model, spec, _ = load(args.pfnn)
        gen = KinematicCharacterController(model, spec)
        gen.reset(root_height=0.75)
        q, ct = [], []
        per = max(1, (args.steps + 200) // len(args.speeds))
        for sp in args.speeds:
            for _ in range(per):
                f = gen.step(Command(move=np.array([sp, 0.0]), responsiveness=0.6))
                q.append(f.qpos(spec.n_joints))
        qpos = np.stack(q)
        label = f"PFNN({Path(args.pfnn).parent.name})"
    else:
        from prepare_robot_data import resample
        qpos, _, _ = resample(np.load(args.clip), args.fps)
        label = Path(args.clip).stem

    m = mujoco.MjModel.from_xml_path(robot.path())
    d = mujoco.MjData(m)
    sub = max(1, int(round(1.0 / (args.fps * m.opt.timestep))))
    mpc = PredictiveSamplingMPC(m, robot, MPCConfig(horizon=args.horizon,
                                                    n_sample=args.samples,
                                                    sigma=args.sigma), substeps=sub)
    est = LegOdometry(m, robot)

    st = args.start
    d.qpos[:] = qpos[st]
    d.qvel[:] = 0
    d.qvel[:3] = (qpos[st + 1, :3] - qpos[st, :3]) * args.fps
    d.qvel[6:] = (qpos[st + 1, 7:] - qpos[st, 7:]) * args.fps
    mujoco.mj_forward(m, d)

    n = min(args.steps, len(qpos) - st - args.horizon - 2)
    errs, hs, imp, log = [], [], [], []
    for t in range(n):
        ref = qpos[st + t: st + t + args.horizon]
        state = LowState.from_sim(m, d)
        e = est.update(state, None)
        s0 = estimated_state(m, d, None, state, e, mpc.state_size)
        # roll out from the estimate, but keep the true base xy for the sim itself
        saved = d.qpos[:2].copy()
        scratch = mujoco.MjData(m)
        mujoco.mj_setState(m, scratch, s0, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        ctrl = mpc.step(scratch, ref[:, 7:], ref[:, 2])
        imp.append(mpc.last["improvement"])
        d.ctrl[:] = ctrl
        for _ in range(sub):
            mujoco.mj_step(m, d)
        errs.append(float(np.abs(d.qpos[7:] - qpos[st + t, 7:]).mean()))
        hs.append(float(d.qpos[2]))
        log.append(d.qpos.copy())
        R = d.xmat[1].reshape(3, 3)
        if d.qpos[2] < 0.43 or R[2, 2] < 0.3:
            break
    mpc.close()

    print(f"{label:24s} H={args.horizon} K={args.samples} sigma={args.sigma}")
    print(f"  survived {t+1:4d}/{n} steps ({(t+1)/args.fps:5.2f} s)  "
          f"joint err {np.mean(errs):.4f} rad  height {np.mean(hs):.3f}")
    print(f"  MPC improvement over tracking the reference exactly: "
          f"{np.mean(imp):.1f} mean cost units")
    if args.save:
        np.savez_compressed(args.save, qpos=np.stack(log).astype(np.float32),
                            contacts=np.zeros((len(log), 2), np.float32),
                            phase=np.zeros(len(log), np.float32), fps=args.fps)
        print(f"  saved {args.save}")


if __name__ == "__main__":
    main()
