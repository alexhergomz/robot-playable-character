"""Contact as a geometric fact, not a height threshold.

The heuristic this replaces asked whether a foot body's origin sat below a
rolling low percentile of its own height track, with a 1.5 cm tolerance:

    floor = rolling_percentile(height, window, q=10)
    contact = height - floor < 0.015

Three constants (window, percentile, tolerance) with no physical referent, and
a failure mode we hit before -- on a clip where the actor goes to the ground,
the "floor" follows them down.

MuJoCo already knows the answer.  ``mj_forward`` runs the collision solver, so
for any kinematic pose the model's own geometry says which feet touch the
ground.  No thresholds, and it accounts for foot shape and orientation -- a
foot pivoting onto its toe is in contact, which a body-origin height test
misses.
"""
from __future__ import annotations

import mujoco
import numpy as np


def foot_geom_ids(model, foot_bodies: tuple[str, ...]) -> list[np.ndarray]:
    """Collision geoms belonging to each named foot body."""
    out = []
    for b in foot_bodies:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
        assert bid >= 0, f"no body {b!r}"
        out.append(np.array([g for g in range(model.ngeom)
                             if model.geom_bodyid[g] == bid]))
    return out


def floor_geom_ids(model) -> np.ndarray:
    """Static geoms attached to the world body -- the ground."""
    return np.array([g for g in range(model.ngeom)
                     if model.geom_bodyid[g] == 0])


def contacts_from_model(model, data, qpos: np.ndarray,
                        foot_bodies: tuple[str, ...] = ("left_ankle_roll_link",
                                                        "right_ankle_roll_link"),
                        ) -> np.ndarray:
    """(T, n_feet) boolean contact, from the simulator's collision solver."""
    feet = foot_geom_ids(model, foot_bodies)
    floor = set(floor_geom_ids(model).tolist())
    T = len(qpos)
    out = np.zeros((T, len(feet)), bool)
    for t, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for c in range(data.ncon):
            g1, g2 = data.contact[c].geom1, data.contact[c].geom2
            for i, gs in enumerate(feet):
                if (g1 in floor and g2 in gs) or (g2 in floor and g1 in gs):
                    out[t, i] = True
    return out


def com_and_feet(model, data, qpos: np.ndarray, foot_bids):
    """CoM and foot-body positions along a trajectory.  Pure forward kinematics."""
    T = len(qpos)
    com = np.zeros((T, 3))
    feet = np.zeros((T, len(foot_bids), 3))
    for i, q in enumerate(qpos):
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        com[i] = data.subtree_com[0]
        for k, b in enumerate(foot_bids):
            feet[i, k] = data.xpos[b]
    return com, feet


def support_distance(model, data, qpos, foot_bodies=("left_ankle_roll_link",
                                                    "right_ankle_roll_link"),
                     stride: int = 2) -> np.ndarray:
    """Signed distance from the lower foot's collision geometry to the floor.

    Uses the model's own geometry, so a foot rolled onto its toe reports the
    toe's clearance rather than the ankle origin's height.  ``stride``
    subsamples frames; it affects only how many samples the estimate is built
    from, not what is estimated.
    """
    feet = foot_geom_ids(model, foot_bodies)
    floor = int(floor_geom_ids(model)[0])
    scratch = np.zeros(6)
    out = []
    for t in range(0, len(qpos), stride):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(model, data)
        out.append(min(mujoco.mj_geomDistance(model, data, int(g), floor, 2.0, scratch)
                       for gs in feet for g in gs))
    return np.asarray(out)


def shorth(x: np.ndarray) -> float:
    """Mean of the shortest interval containing half the samples.

    A standard robust mode estimator.  Used instead of picking a percentile:
    the ground is where the support foot *most often* rests, and the shorth
    finds that without anyone choosing which quantile "most often" means.
    """
    s = np.sort(np.asarray(x, float))
    n = len(s)
    h = max(1, n // 2)
    w = s[h - 1:] - s[:n - h + 1]
    i = int(np.argmin(w))
    return float(s[i:i + h].mean())


def ground_offset(model, data, qpos, foot_bodies=("left_ankle_roll_link",
                                                  "right_ankle_roll_link"),
                  stride: int = 2) -> float:
    """How far to lower a retargeted clip so its support foot rests on the floor.

    Replaces ``percentile(ankle_height, 40) - keyframe_ankle_height``, which
    carried two assumptions: that 40% of frames have the support foot planted
    (duty factor is 0.5-0.8 and differs per clip, and running clips have
    flight), and that the ankle body sits a fixed height above the sole (true
    only for the keyframe pose, not for a foot rolled onto heel or toe).

    Here the quantity measured is the real signed distance from the foot's
    collision geometry to the floor, and the offset is its mode -- where the
    foot rests most of the time.  Translating the clip in z shifts every
    distance equally, so this needs one pass and no iteration.
    """
    return shorth(support_distance(model, data, qpos, foot_bodies, stride))
