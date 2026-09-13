"""Whole-body controller: QP inverse dynamics.

Deterministic, non-neural, no training.  The generator says "I want this pose";
this decides what torques make that happen *given* the robot's dynamics, the
contacts it currently has, and what the friction and the motors can actually
deliver.

Decision variables are joint accelerations and contact forces together:

    x = [ qddot (nv) ; f (3 per contact point) ]

**Hard constraints — physics, not behaviour:**

* *Underactuated dynamics.*  The floating base has six degrees of freedom and no
  actuators, so its acceleration is whatever gravity and the contact forces
  make it.  Those six rows are an equality the solver cannot dodge, and they
  are the reason a controller that only cares about joint angles will happily
  produce feasible torques that tip the robot over.
* *Contact.*  Points in contact do not accelerate, so the stance foot stays put.
* *Friction cone*, linearised as a pyramid: no force outside what the ground
  can supply.
* *Torque limits*, from the real G1 motor effort limits.

**Objective — what to give up when it cannot have everything.**  Tracking the
reference: joint accelerations from a PD law on the reference pose, and
optionally the base too.  ``use_base`` exists so the base term can be ablated
rather than assumed: with it off, the objective is joint tracking alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import sonic as P
from .robot import RobotSpec

#: Real G1 motor effort limits, by joint, in Nm.
TORQUE_LIMIT = np.array([P.EFFORT_LIMIT[t] for t in P.MOTOR_TYPE])


@dataclass
class WBCConfig:
    kp_joint: float = 250.0
    kd_joint: float = 25.0
    kp_height: float = 150.0         # base height, from leg odometry
    kd_height: float = 25.0
    kp_base_ori: float = 220.0       # base attitude, straight from the IMU
    kd_base_ori: float = 30.0
    kd_base_vel: float = 30.0        # base linear velocity, from leg odometry
    mu: float = 0.8                  # friction coefficient
    w_joint: float = 1.0
    w_base: float = 60.0             # base error is the unrecoverable kind
    w_force: float = 1e-4            # regularisation, keeps the QP well posed
    contact_damping: float = 20.0    # damps contact-point drift in place of Jdot*qdot
    contact_height: float = 0.02     # a foot point this close to the floor is in contact
    use_base: bool = True            # ablation switch
    solver: str = "daqp"


class WholeBodyController:
    def __init__(self, model, spec: RobotSpec, cfg: WBCConfig | None = None):
        self.model = model
        self.spec = spec
        self.cfg = cfg or WBCConfig()
        self.nv = model.nv
        self.nu = model.nu

        # contact points: the sphere geoms on each foot, in body-local coords
        self.points: list[tuple[int, np.ndarray]] = []
        for body in spec.foot_bodies:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == bid and model.geom_contype[g]:
                    self.points.append((bid, model.geom_pos[g].copy()))
        assert self.points, "no contact geoms found on the feet"
        self.n_pt = len(self.points)

        self._M = np.zeros((self.nv, self.nv))
        self._jacp = np.zeros((3, self.nv))
        self.last = {}

    # -- contact set and Jacobian -------------------------------------------
    def _contacts(self, data, schedule=None):
        """Which foot points are load-bearing, and their stacked Jacobian.

        Geometry alone tells you a foot is down; it cannot tell you a foot is
        about to lift.  The controller then constrains a swing foot right up
        until the constraint vanishes underneath it, which is exactly where the
        first version failed — contacts collapsed 8 to 1 at the step transition.
        The generator already predicts the contact schedule, so pass it in.
        """
        per = len(self.points) // 2
        active, J, vel = [], [], []
        for k, (bid, local) in enumerate(self.points):
            if schedule is not None and schedule[min(k // per, len(schedule) - 1)] <= 0.5:
                continue
            world = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ local
            if world[2] > self.cfg.contact_height:
                continue
            mujoco.mj_jac(self.model, data, self._jacp, None, world, bid)
            active.append(k)
            J.append(self._jacp.copy())
            vel.append(self._jacp @ data.qvel)
        if not active:
            return active, np.zeros((0, self.nv)), np.zeros(0)
        return active, np.vstack(J), np.concatenate(vel)

    # -- desired accelerations ----------------------------------------------
    def _desired(self, state, est, q_ref, qd_ref, base_ref):
        """Desired accelerations from estimated state only.

        Note what is *not* here: absolute horizontal position.  The robot cannot
        measure it and integrating velocity drifts, so nothing tracks it.  The
        base task is height, attitude and linear velocity — all of which leg
        odometry and the IMU actually provide.
        """
        c = self.cfg
        a_j = (c.kp_joint * (q_ref - state.q)
               + c.kd_joint * ((qd_ref if qd_ref is not None else 0.0) - state.dq))
        if base_ref is None or not c.use_base:
            return a_j, np.zeros(6)
        h_ref, quat_ref, v_ref = base_ref

        dq = np.empty(4)
        mujoco.mju_mulQuat(dq, quat_ref, np.array([state.quaternion[0], -state.quaternion[1],
                                                   -state.quaternion[2], -state.quaternion[3]]))
        e_ori = np.empty(3)
        mujoco.mju_quat2Vel(e_ori, dq, 1.0)

        a_lin = c.kd_base_vel * (v_ref - est.lin_vel)
        a_lin[2] += c.kp_height * (h_ref - est.height) - c.kd_height * est.lin_vel[2]
        a_ang = c.kp_base_ori * e_ori - c.kd_base_ori * est.ang_vel
        return a_j, np.concatenate([a_lin, a_ang])

    # -- the QP --------------------------------------------------------------
    def solve(self, data, state, est, q_ref, qd_ref=None, base_ref=None,
              schedule=None) -> np.ndarray:
        from qpsolvers import solve_qp

        c = self.cfg
        nv, nu = self.nv, self.nu
        active, Jc, Jv = self._contacts(data, schedule)
        nf = 3 * len(active)
        n = nv + nf

        # MuJoCo >= 3.12 takes MjData here, not the sparse qM array.
        mujoco.mj_fullM(self.model, data, self._M)
        M = self._M
        h = data.qfrc_bias.copy()
        a_j, a_b = self._desired(state, est, q_ref, qd_ref, base_ref)

        # objective: diagonal, because every term is a selection of x
        w = np.concatenate([np.full(6, c.w_base if c.use_base else 1e-6),
                            np.full(nu, c.w_joint), np.full(nf, c.w_force)])
        tgt = np.concatenate([a_b, a_j, np.zeros(nf)])
        Pm = np.diag(2.0 * w)
        qv = -2.0 * w * tgt

        # equalities: underactuated rows, then contact acceleration
        A = np.zeros((6 + nf, n))
        b = np.zeros(6 + nf)
        A[:6, :nv] = M[:6, :]
        b[:6] = -h[:6]
        if nf:
            A[:6, nv:] = -Jc[:, :6].T
            A[6:, :nv] = Jc
            b[6:] = -c.contact_damping * Jv

        # inequalities: friction pyramid, then torque limits
        G, hh = [], []
        for i in range(len(active)):
            z = nv + 3 * i + 2
            for ax in (0, 1):
                for sgn in (1.0, -1.0):
                    row = np.zeros(n)
                    row[nv + 3 * i + ax] = sgn
                    row[z] = -c.mu
                    G.append(row)
                    hh.append(0.0)
        Gt = np.zeros((2 * nu, n))
        Gt[:nu, :nv] = M[6:, :]
        Gt[nu:, :nv] = -M[6:, :]
        if nf:
            Gt[:nu, nv:] = -Jc[:, 6:].T
            Gt[nu:, nv:] = Jc[:, 6:].T
        G.append(Gt)
        hh.append(np.concatenate([TORQUE_LIMIT - h[6:], TORQUE_LIMIT + h[6:]]))
        G = np.vstack([g.reshape(1, -1) if g.ndim == 1 else g for g in G])
        hh = np.concatenate([np.atleast_1d(x) for x in hh])

        lb = np.full(n, -np.inf)
        for i in range(len(active)):
            lb[nv + 3 * i + 2] = 0.0            # unilateral contact: no pulling

        x = solve_qp(Pm, qv, G, hh, A, b, lb=lb, solver=c.solver)
        if x is None:
            self.last = {"status": "infeasible", "n_contact": len(active)}
            return np.zeros(nu)

        qdd, f = x[:nv], x[nv:]
        tau = M[6:, :] @ qdd + h[6:] - (Jc[:, 6:].T @ f if nf else 0.0)
        tau = np.clip(tau, -TORQUE_LIMIT, TORQUE_LIMIT)
        self.last = {"status": "ok", "n_contact": len(active),
                     "tau_max": float(np.abs(tau).max()),
                     "joint_acc_err": float(np.abs(qdd[6:] - a_j).mean()),
                     "base_acc_err": float(np.abs(qdd[:6] - a_b).mean()),
                     "fz": float(f[2::3].sum()) if nf else 0.0}
        return tau


def apply_torque_actuators(model) -> None:
    """Switch the position servos to direct torque control."""
    for i in range(model.nu):
        model.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_NONE
        model.actuator_gainprm[i, :] = 0.0
        model.actuator_gainprm[i, 0] = 1.0
        model.actuator_biasprm[i, :] = 0.0
        model.actuator_ctrlrange[i] = [-TORQUE_LIMIT[i], TORQUE_LIMIT[i]]
        model.actuator_forcerange[i] = [-TORQUE_LIMIT[i], TORQUE_LIMIT[i]]
