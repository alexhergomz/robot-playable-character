"""Make a kinematically-retargeted motion physically executable.

The diagnosis this implements: retargeting transfers *joint angles*, which are a
property of the pose, but physical validity is a property of the pose **and the
body**.  The G1's torso is 49% of the human's, its legs 77%, and its mass
distribution is nothing alike, so copying the human's root trajectory puts the
centre of pressure up to 1.5 m outside the support polygon.  The robot cannot
help but tip; no tracker can rescue it.

Everything else about the motion transfers.  Foot placement, contact schedule,
arm pose and style are all fine.  The root is the only part that must be
recomputed, because the root is where the mass is.

So: keep the footprints, solve for a centre-of-mass trajectory whose ZMP stays
over them, move the root to realise it, then re-plant the feet with IK.  This is
the cart-table model (Kajita et al. 2003) in batch form — a linear least-squares
problem, no learning, no reward, no iteration over rollouts.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

G = 9.81


@dataclass
class DynamicsFixConfig:
    smooth: float = 12.0        # weight on CoM acceleration; higher = calmer body
    track: float = 1.0          # weight on staying near the original CoM path
    zmp: float = 60.0           # weight on putting the ZMP over the footprints
    smooth_z: float = 400.0     # weight on *vertical* CoM acceleration
    track_z: float = 1.0
    ik_iters: int = 24
    ik_damping: float = 0.05


def com_and_feet(model, data, qpos: np.ndarray, foot_bids):
    """CoM and foot-body positions along a trajectory."""
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


def robot_contacts(feet: np.ndarray, sole_offset: float, fps: float,
                   margin: float = 0.025, speed_thresh: float = 0.9) -> np.ndarray:
    """Contact read off the *robot's own* feet, not inherited from the mocap.

    The mocap detector is a heuristic tuned on a human skeleton with a toe joint
    and real foot sliding; carried across it reported up to 70% flight on
    walking clips, which releases the ZMP constraint on most frames and makes
    the correction a no-op exactly where it is needed.

    The robot's geometry is known: a foot body sits ``sole_offset`` above the
    ground when its sole is planted, so contact is directly observable.
    """
    z = feet[..., 2]
    v = np.linalg.norm(np.gradient(feet, axis=0)[..., :2] * fps, axis=-1)
    return (z < sole_offset + margin) & (v < speed_thresh)


def desired_zmp(feet: np.ndarray, contacts: np.ndarray):
    """Centre of pressure over the feet that are down, and where it applies.

    Returns ``(zmp_ref, weight)``.  During flight there is no support polygon
    and hence no ZMP constraint at all: the body is ballistic, so the horizontal
    CoM velocity is constant.  Holding the last ZMP value and *enforcing* it —
    which this did originally — asks the solver to satisfy a constraint that
    does not exist, and on running clips it responded by moving the feet 1.5-2 m.
    Zeroing the weight there is not a patch: the smoothness term then drives the
    horizontal CoM to constant velocity, which is exactly the ballistic answer.
    """
    T = len(feet)
    z = np.zeros((T, 2))
    w = np.zeros(T)
    last = feet[0, :, :2].mean(0)
    for i in range(T):
        on = contacts[i] > 0.5
        if on.any():
            last = feet[i, on, :2].mean(0)
            w[i] = 1.0
        z[i] = last
    return z, w


def solve_com(com0: np.ndarray, zmp_ref: np.ndarray, dt: float,
              cfg: DynamicsFixConfig, weight: np.ndarray | None = None) -> np.ndarray:
    """Least-squares CoM trajectory balancing three wishes.

    With a constant CoM height the cart-table relation is *linear* in the CoM,

        zmp_t = com_t - (h / g) * com_ddot_t

    so "put the ZMP over the feet", "stay near the original path" and "do not
    thrash" are all quadratic, and the whole trajectory comes out of one sparse
    solve rather than a forward integration (which is unstable) or an optimiser.
    """
    T = len(com0)
    h = float(np.median(com0[:, 2]))
    k = h / (G * dt * dt)

    # second-difference operator
    rows = np.arange(1, T - 1)
    D2 = sp.coo_matrix(
        (np.concatenate([np.ones(T - 2), -2 * np.ones(T - 2), np.ones(T - 2)]),
         (np.concatenate([rows, rows, rows]),
          np.concatenate([rows - 1, rows, rows + 1]))),
        shape=(T, T)).tocsr()

    A = sp.identity(T, format="csr") - k * D2          # zmp = A @ com
    I = sp.identity(T, format="csr")
    w = np.ones(T) if weight is None else np.asarray(weight, dtype=float)
    W = sp.diags(cfg.zmp * w)
    lhs = ((A.T @ W @ A) + cfg.track * I + cfg.smooth * (D2.T @ D2)).tocsc()
    out = com0.copy()
    for axis in (0, 1):
        rhs = (A.T @ (W @ zmp_ref[:, axis])) + cfg.track * com0[:, axis]
        out[:, axis] = spl.spsolve(lhs, rhs)
    return out


def _smooth_axis(x: np.ndarray, track: float, smooth: float) -> np.ndarray:
    """Least-squares smoothing: stay near ``x``, but penalise acceleration."""
    T = len(x)
    rows = np.arange(1, T - 1)
    D2 = sp.coo_matrix(
        (np.concatenate([np.ones(T - 2), -2 * np.ones(T - 2), np.ones(T - 2)]),
         (np.concatenate([rows, rows, rows]),
          np.concatenate([rows - 1, rows, rows + 1]))),
        shape=(T, T)).tocsr()
    lhs = (track * sp.identity(T, format="csr") + smooth * (D2.T @ D2)).tocsc()
    return spl.spsolve(lhs, track * x)


class FootPlanter:
    """Leg-only IK that pins the feet to given world positions."""

    def __init__(self, model, foot_bids, leg_joint_prefixes=("left_hip", "left_knee",
                                                             "left_ankle", "right_hip",
                                                             "right_knee", "right_ankle")):
        self.model = model
        self.data = mujoco.MjData(model)
        self.foot_bids = list(foot_bids)
        dofs = []
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if any(name.startswith(p) for p in leg_joint_prefixes):
                dofs.append(model.jnt_dofadr[j])
        self.dofs = np.array(dofs, dtype=int)
        self.qadr = np.array([model.jnt_qposadr[model.dof_jntid[d]] for d in self.dofs])
        self._jac = np.zeros((3, model.nv))

    def solve(self, q: np.ndarray, targets: np.ndarray, cfg: DynamicsFixConfig):
        m, d = self.model, self.data
        q = q.copy()
        nd = len(self.dofs)
        for _ in range(cfg.ik_iters):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err = np.empty(3 * len(self.foot_bids))
            J = np.empty((3 * len(self.foot_bids), nd))
            worst = 0.0
            for i, b in enumerate(self.foot_bids):
                e = targets[i] - d.xpos[b]
                worst = max(worst, float(np.linalg.norm(e)))
                err[3 * i:3 * i + 3] = e
                mujoco.mj_jacBody(m, d, self._jac, None, b)
                J[3 * i:3 * i + 3] = self._jac[:, self.dofs]
            if worst < 1e-3:
                break
            A = J.T @ J + (cfg.ik_damping ** 2) * np.eye(nd)
            dq = np.linalg.solve(A, J.T @ err)
            q[self.qadr] += dq
            lo, hi = m.jnt_range[m.dof_jntid[self.dofs]].T
            q[self.qadr] = np.clip(q[self.qadr], lo, hi)
        return q


def correct(model, qpos: np.ndarray, contacts: np.ndarray, foot_bids, fps: float,
            cfg: DynamicsFixConfig | None = None, sole_offset: float | None = None):
    """Return a dynamics-corrected copy of ``qpos`` plus a diagnostic dict."""
    cfg = cfg or DynamicsFixConfig()
    data = mujoco.MjData(model)
    com0, feet0 = com_and_feet(model, data, qpos, foot_bids)
    if sole_offset is not None:
        contacts = robot_contacts(feet0, sole_offset, fps)
    zref, zw = desired_zmp(feet0, contacts)
    com1 = solve_com(com0, zref, 1.0 / fps, cfg, weight=zw)

    # Translating the root translates the whole body, so the CoM moves with it
    # one-for-one; the correction can be applied to the root directly.
    shift = com1[:, :2] - com0[:, :2]
    # Vertical too.  Per-frame grounding copies the human's foot-height signal
    # straight into the root, and that shows up as 1.84 m/s2 of vertical CoM
    # acceleration against 0.57 for a physics-corrected reference -- a 3.2x
    # excess that sits in the ZMP denominator m(z_ddot + g) *and* drives the
    # angular-momentum rate.  A walking human keeps the CoM path smooth by
    # flexing the knee; solving for a smooth height and re-planting the feet
    # makes the legs absorb it the same way.
    z1 = _smooth_axis(com0[:, 2], cfg.track_z, cfg.smooth_z)
    out = qpos.copy()
    out[:, :2] += shift
    out[:, 2] += z1 - com0[:, 2]

    # ...which drags the feet along, so plant them back where they were.
    planter = FootPlanter(model, foot_bids)
    for i in range(len(out)):
        tgt = feet0[i].copy()
        tgt[:, :2] += 0.0                      # footprints stay put in the world
        out[i] = planter.solve(out[i], tgt, cfg)

    com2, feet2 = com_and_feet(model, data, out, foot_bids)
    return out, {"shift_cm": float(np.abs(shift).mean() * 100),
                 "flight_frac": float(1.0 - zw.mean()),
                 "z_shift_cm": float(np.abs(z1 - com0[:, 2]).mean() * 100),
                 "com_before": com0, "com_after": com2,
                 "foot_move_cm": float(np.linalg.norm(feet2 - feet0, axis=-1).mean() * 100)}
