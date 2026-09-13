#!/usr/bin/env python3
"""Probe: remove foot skate from retargeted clips by moving the root.

GMR gives the feet the only position cost in its IK, so the robot's foot is
driven to the scaled human ankle position.  Nothing tells the solver "this foot
is on the ground", so the sole ends up 0.4-1.8 cm into the floor, by a
different amount per clip, and a planted foot drifts.

A planted foot that drifts means the root travelled further than the legs
carried it.  The fix does not need IK: subtract that drift from the root.
Corrections are accumulated incrementally rather than applied per stance
interval, so the trajectory stays continuous across touchdown and toe-off.

Stance is detected from horizontal foot speed, which is independent of the
vertical registration this is meant to establish -- and the human source is
clean enough for that to be meaningful (0.4 cm/s slip while planted).
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpc.motion.contacts import contacts_from_model, floor_geom_ids, foot_geom_ids
from rpc.motion.phase import contact_phase
from rpc.sim.robot import REGISTRY

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def foot_tracks(model, data, qpos):
    bid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in FEET]
    fg = foot_geom_ids(model, FEET)
    floor = int(floor_geom_ids(model)[0])
    sc = np.zeros(6)
    pos = np.zeros((len(qpos), 2, 3))
    dist = np.zeros((len(qpos), 2))
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for i, gs in enumerate(fg):
            pos[t, i] = data.xpos[bid[i]]
            dist[t, i] = min(mujoco.mj_geomDistance(model, data, int(g), floor, 2.0, sc)
                             for g in gs)
    return pos, dist


def rest_distance(model, data):
    fg = foot_geom_ids(model, FEET)
    floor = int(floor_geom_ids(model)[0])
    sc = np.zeros(6)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return min(mujoco.mj_geomDistance(model, data, int(g), floor, 2.0, sc)
               for gs in fg for g in gs)


def lock(model, data, qpos, fps, stance_q=20.0):
    pos, dist = foot_tracks(model, data, qpos)
    rest = rest_distance(model, data)
    spd = np.zeros((len(qpos), 2))
    spd[:-1] = np.linalg.norm(np.diff(pos[..., :2], axis=0), axis=-1) * fps
    spd[-1] = spd[-2]
    # a foot is planted when it is slow AND it is the lower of the two
    thr = np.percentile(spd, stance_q)
    lower = dist <= dist.min(axis=1, keepdims=True) + 0.01
    stance = (spd <= thr) & lower

    # Accumulate the correction in the horizontal plane only.  Integrating the
    # vertical component as well makes small per-frame errors compound over
    # thousands of frames and walks the robot off the floor entirely -- it
    # produced clips with a duty factor of 0.00.  Height is a single global
    # registration, not something to integrate.
    off = np.zeros((len(qpos), 3))
    for t in range(1, len(qpos)):
        both = stance[t] & stance[t - 1]
        off[t] = off[t - 1]
        if both.any():
            d = (pos[t, both] - pos[t - 1, both]).mean(axis=0)
            off[t, :2] = off[t - 1, :2] - d[:2]
    # vertical: put the planted sole at the model's own resting distance
    if stance.any():
        off[:, 2] = -(np.median(dist[stance]) - rest)
    q = qpos.astype(np.float64).copy()
    q[:, :3] += off
    return q, stance, rest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/gmr_corrected/g1")
    ap.add_argument("--out", default="data/processed/gmr_locked/g1")
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--stance-q", type=float, default=20.0)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(REGISTRY[args.robot].path())
    d = mujoco.MjData(m)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"{'clip':22s} {'slip before':>12s} {'after':>8s}   "
          f"{'speed before':>13s} {'after':>7s}   duty")
    for f in sorted(glob.glob(args.src + "/*.npz")):
        z = np.load(f)
        q0, fps = z["qpos"].astype(np.float64), float(z["fps"])
        q1, stance, rest = lock(m, d, q0, fps, args.stance_q)

        def slip(q):
            pos, _ = foot_tracks(m, d, q)
            s = np.linalg.norm(np.diff(pos[..., :2], axis=0), axis=-1) * fps
            return np.median(s[stance[:-1]]) * 100

        v0 = np.linalg.norm(np.diff(q0[:, :2], axis=0), axis=-1).sum() / (len(q0) / fps)
        v1 = np.linalg.norm(np.diff(q1[:, :2], axis=0), axis=-1).sum() / (len(q1) / fps)
        ct = contacts_from_model(m, d, q1).astype(np.float32)
        ph = contact_phase(ct, fps).astype(np.float32)
        np.savez_compressed(out / Path(f).name, qpos=q1.astype(np.float32),
                            contacts=ct, phase=ph, fps=fps)
        print(f"{Path(f).stem:22s} {slip(q0):9.2f} cm/s {slip(q1):6.2f}   "
              f"{v0:10.2f} m/s {v1:6.2f}   {ct.mean():.2f}", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
