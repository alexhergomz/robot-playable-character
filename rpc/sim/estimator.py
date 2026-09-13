"""State estimation from exactly what a real G1 EDU reports.

A deployment reads only this, per NVIDIA's own G1 deploy code:

    imu_state().quaternion()      base orientation
    imu_state().gyroscope()       base angular velocity, body frame
    imu_state().accelerometer()   base linear acceleration, body frame
    motor_state()[i].q(), .dq()   29 joint positions and velocities

There is **no base position, no base linear velocity, and no foot force
sensor**.  A controller that reads MuJoCo's ``qpos[:3]`` or ``qvel[:3]`` is
using information the robot does not have, and any result that depends on it
does not transfer.

What *is* recoverable is recoverable by leg odometry, which is standard
practice: while a foot is planted its world position is fixed, so the base
height follows from the robot's own kinematics and the IMU attitude, and the
base linear velocity follows from how fast the joints are moving the planted
foot underneath it.  Horizontal *position* is not recoverable — integrating
velocity drifts without bound — so nothing should track it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np


@dataclass
class LowState:
    """Exactly the fields the SDK gives, and nothing else."""

    q: np.ndarray                 # (29,) joint positions
    dq: np.ndarray                # (29,) joint velocities
    quaternion: np.ndarray        # (4,) wxyz, IMU attitude
    gyroscope: np.ndarray         # (3,) body-frame angular velocity
    accelerometer: np.ndarray     # (3,) body-frame linear acceleration

    @classmethod
    def from_sim(cls, model, data) -> "LowState":
        """Read a simulator the way the SDK reads the robot.

        Deliberately narrow: this is the only place the sim is touched, so the
        controller physically cannot reach for privileged state.
        """
        acc = np.zeros(3)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "imu_accel")
        if sid >= 0:
            acc = data.sensordata[model.sensor_adr[sid]:model.sensor_adr[sid] + 3].copy()
        else:
            R = data.xmat[1].reshape(3, 3)
            acc = R.T @ (data.cacc[1][3:] + np.array([0.0, 0.0, 9.81])) \
                if hasattr(data, "cacc") else np.array([0.0, 0.0, 9.81])
        return cls(q=data.qpos[7:].copy(), dq=data.qvel[6:].copy(),
                   quaternion=data.qpos[3:7].copy(),
                   gyroscope=data.qvel[3:6].copy(), accelerometer=acc)


@dataclass
class Estimate:
    gravity: np.ndarray           # projected gravity, base frame
    height: float                 # base height above the contact plane
    lin_vel: np.ndarray           # base linear velocity, world frame
    ang_vel: np.ndarray           # base angular velocity, world frame
    quat: np.ndarray              # base attitude (straight from the IMU)
    contact: np.ndarray           # per-foot-point contact flags used for odometry
    valid: bool                   # false when no foot is planted (flight)


class LegOdometry:
    """Base height and velocity from joint kinematics plus IMU attitude."""

    def __init__(self, model, spec, contact_height: float = 0.03,
                 vel_filter: float = 0.7):
        self.model = model
        self.data = mujoco.MjData(model)
        self.vel_filter = vel_filter
        self.contact_height = contact_height
        self.points: list[tuple[int, np.ndarray]] = []
        for body in spec.foot_bodies:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == bid and model.geom_contype[g]:
                    self.points.append((bid, model.geom_pos[g].copy()))
        self._jacp = np.zeros((3, model.nv))
        self._v = np.zeros(3)
        self._h = 0.75

    def _base_frame_kinematics(self, q: np.ndarray):
        """Foot points and their Jacobians with the base pinned at the origin.

        This is what a deployment does with its URDF: the robot knows its own
        shape, so given the encoders it knows where its feet are relative to
        itself.  It does not know where *it* is.
        """
        d = self.data
        d.qpos[:3] = 0.0
        d.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        d.qpos[7:] = q
        d.qvel[:] = 0.0
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        out = []
        for bid, local in self.points:
            p = d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ local
            mujoco.mj_jac(self.model, d, self._jacp, None, p, bid)
            out.append((p.copy(), self._jacp[:, 6:].copy()))
        return out

    def update(self, s: LowState, reference_contact: np.ndarray | None = None) -> Estimate:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, s.quaternion)
        R = R.reshape(3, 3)
        gravity = R.T @ np.array([0.0, 0.0, -1.0])
        omega = R @ s.gyroscope                                 # world frame

        pts = self._base_frame_kinematics(s.q)
        world_rel = [R @ p for p, _ in pts]                     # foot relative to base, world axes

        # Which points are planted.  The reference schedule, when given, says
        # which *foot* should be down; geometry picks the points on it.
        z = np.array([w[2] for w in world_rel])
        lowest = z.min()
        flags = z < lowest + self.contact_height
        if reference_contact is not None and len(self.points) % len(reference_contact) == 0:
            per = len(self.points) // len(reference_contact)
            sched = np.repeat(reference_contact > 0.5, per)
            if sched.any():
                flags = flags & sched

        if not flags.any():
            return Estimate(gravity, self._h, self._v.copy(), omega, s.quaternion,
                            flags, valid=False)

        # Height: a planted foot sits on the plane, so the base is that far above it.
        self._h = float(-np.mean([world_rel[i][2] for i in np.flatnonzero(flags)]))

        # Velocity: the planted point is world-stationary, so the base moves
        # opposite to however the joints are driving that point.
        vs = []
        for i in np.flatnonzero(flags):
            p_rel, J = world_rel[i], pts[i][1]
            vs.append(-(R @ (J @ s.dq)) - np.cross(omega, p_rel))
        v = np.mean(vs, axis=0)
        a = self.vel_filter
        self._v = a * self._v + (1 - a) * v
        return Estimate(gravity, self._h, self._v.copy(), omega, s.quaternion,
                        flags, valid=True)
