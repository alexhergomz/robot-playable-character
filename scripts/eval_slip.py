#!/usr/bin/env python3
"""True stance slip: the velocity of the material point actually in contact.

Differencing a foot body's origin does not measure sliding.  A planted foot
rolling heel-to-toe swings its ankle forward at speed while the contact patch
stays put, so the body-origin difference reports large "slip" that is really
rotation -- it put both our system and the training corpus at p90 ~50 cm/s and
made an artifact look like a defect.

Here the contact point is taken from the collision solver, converted to a
body-fixed material point, and followed to the next frame.  A planted foot
scores zero however much the ankle above it moves.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rpc.control.command import Command
from rpc.generator.lmp import LMPGenerator
from rpc.generator.lmp_runtime import LMPCharacterController
from rpc.motion.contacts import floor_geom_ids, foot_geom_ids
from rpc.motion.features import FeatureConfig
from rpc.motion.robot_dataset import RobotFeatureSpec
from rpc.sim import sonic as P
from rpc.sim.robot import REGISTRY
from rpc.sim.sonic_policy import (CONTROL_HZ, SonicPolicy, apply_sonic_armature,
                                  apply_sonic_gains)
from eval_rpc import lookahead

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def contact_slip(model, data, qpos, fps):
    fg = foot_geom_ids(model, FEET)
    floor = set(floor_geom_ids(model).tolist())
    bid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in FEET]
    dt, out = 1.0 / fps, []
    for t in range(len(qpos) - 1):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        keep = []
        for c in range(data.ncon):
            g1, g2 = data.contact[c].geom1, data.contact[c].geom2
            for i, gs in enumerate(fg):
                if (g1 in floor and g2 in gs) or (g2 in floor and g1 in gs):
                    keep.append((i, data.xpos[bid[i]].copy(),
                                 data.xmat[bid[i]].reshape(3, 3).copy(),
                                 data.contact[c].pos.copy()))
        if not keep:
            continue
        data.qpos[:] = qpos[t + 1]
        mujoco.mj_forward(model, data)
        for i, x0, R0, p in keep:
            loc = R0.T @ (p - x0)
            p1 = data.xmat[bid[i]].reshape(3, 3) @ loc + data.xpos[bid[i]]
            out.append(np.linalg.norm((p1 - p)[:2]) / dt)
    return np.array(out)


def rollout(gen_path, speeds=(0.5, 0.9, 1.4), secs=12, init_frame=60007):
    ck = torch.load(gen_path, map_location="cpu", weights_only=False)
    man, a = ck["manifest"], ck["args"]
    C = man["phase_channels"]
    cfg = FeatureConfig(fps=man["fps"], n_past=man["n_past"], n_future=man["n_future"],
                        span=man["span"], gaits=tuple(man["gaits"]))
    spec = RobotFeatureSpec(man["n_joints"], cfg, man["n_feet"])
    net = LMPGenerator(ck["n_in"], ck["n_out"], gate_dim=4 * C, hidden=a["hidden"],
                       n_experts=a["experts"], dropout=a["dropout"])
    net.load_state_dict(ck["state"])
    M = np.load(ROOT / "data/processed/robot_lmp/train_M.npy", mmap_mode="r")
    F = np.load(ROOT / "data/processed/robot_lmp/train_F.npy", mmap_mode="r")
    mo = mujoco.MjModel.from_xml_path(REGISTRY["g1"].path())
    apply_sonic_gains(mo); apply_sonic_armature(mo)
    sub = max(1, int(round(1.0 / (CONTROL_HZ * mo.opt.timestep))))
    res = {}
    for sp in speeds:
        d = mujoco.MjData(mo)
        pol = SonicPolicy(root=str(ROOT / "assets/sonic"), variant="low_latency")
        g = LMPCharacterController(net, spec, ck["out_slices"], 4 * C,
                                   n_feet=man["n_feet"], keep_cols=ck.get("keep_cols"),
                                   curve=ck.get("amplitude_curve"))
        mujoco.mj_resetDataKeyframe(mo, d, 0)
        d.qpos[7:] = P.DEFAULT_ANGLES; d.qpos[2] = 0.76
        mujoco.mj_forward(mo, d)
        g.reset(joint_pos=P.DEFAULT_ANGLES.copy(), root_height=0.76,
                manifold=M[init_frame].copy(), foot=F[init_frame].copy())

        def grav():
            o = np.zeros(3)
            mujoco.mju_rotVecQuat(o, np.array([0.0, 0.0, -1.0]),
                                  np.array([d.qpos[3], -d.qpos[4], -d.qpos[5], -d.qpos[6]]))
            return o

        pol.reset(d.qpos[7:].copy(), d.qvel[6:].copy(), d.qvel[3:6].copy(), grav())
        c = Command(move=np.array([sp, 0.0]), responsiveness=0.6).clamped()
        log = []
        for t in range(int(secs * CONTROL_HZ)):
            dof, dvel, qt = lookahead(g, c, pol.lookahead_frames * max(pol.step, 1), CONTROL_HZ)
            idx = np.arange(pol.lookahead_frames) * pol.step
            g.step(c)
            tg = pol.step_policy(dof[idx][:, P.MUJOCO_TO_ISAACLAB],
                                 dvel[idx][:, P.MUJOCO_TO_ISAACLAB], qt[idx],
                                 d.qpos[3:7].copy(), d.qpos[7:].copy(),
                                 d.qvel[6:].copy(), d.qvel[3:6].copy(), grav())
            d.ctrl[:] = np.clip(tg, mo.actuator_ctrlrange[:, 0], mo.actuator_ctrlrange[:, 1])
            for _ in range(sub):
                mujoco.mj_step(mo, d)
            log.append(d.qpos.copy())
            if d.qpos[2] < 0.45:
                break
        q = np.stack(log)[150:]
        s = contact_slip(mo, d, q, CONTROL_HZ)
        v = np.linalg.norm(q[-1, :2] - q[0, :2]) / (len(q) / CONTROL_HZ)
        res[sp] = (np.median(s) * 100, np.percentile(s, 90) * 100, v, len(log))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    args = ap.parse_args()
    labels = args.labels or [Path(g).parent.name for g in args.gens]
    print("true contact-point slip (corpus reference: med 1.4-4.8, p90 8.9-17.2 cm/s)\n")
    print(f"  {'model':16s} {'cmd':>5s} {'med':>7s} {'p90':>7s} {'speed':>7s}  fell")
    for g, lab in zip(args.gens, labels):
        for sp, (md, p90, v, n) in rollout(g).items():
            print(f"  {lab:16s} {sp:5.1f} {md:7.1f} {p90:7.1f} {v:7.2f}  "
                  f"{'YES' if n < 500 else 'no'}")
        print()


if __name__ == "__main__":
    main()
