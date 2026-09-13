"""Robot-space features: PFNN that actuates the robot directly.

The two-layer version has the generator animate a *human* skeleton and a tracker
turn that into robot motion.  This module removes the seam: the generator's
state space **is** the robot's, its outputs **are** joint position targets, and
physics closes the loop.  There is exactly one network between the operator's
stick and the actuators.

Two things change relative to the human-space features, and both follow from the
loop being closed through physics rather than through a second policy:

* The state includes **projected gravity, base height and base velocities**.  An
  open-loop animator does not need to know it is falling over; a controller
  does, and this is the only channel through which it can find out.
* Inputs at run time are **measured**, not predicted.  The network is trained by
  regression on recorded robot motion but consumes the simulator's state, which
  makes it a feedback controller rather than a playback device.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import quat as Q
from .features import FeatureConfig


@dataclass(frozen=True)
class RobotFeatureSpec:
    n_joints: int
    cfg: FeatureConfig
    n_feet: int = 2

    @property
    def in_blocks(self) -> list[tuple[str, int]]:
        K, G, J = self.cfg.n_traj, len(self.cfg.gaits), self.n_joints
        return [("traj_pos", 2 * K), ("traj_dir", 2 * K), ("traj_gait", G * K),
                ("joint_pos", J), ("joint_vel", J),
                ("root_height", 1), ("gravity", 3),
                ("root_lin_vel", 3), ("root_ang_vel", 3)]

    @property
    def out_blocks(self) -> list[tuple[str, int]]:
        J, nf = self.n_joints, self.cfg.n_future
        return [("joint_pos", J), ("joint_vel", J), ("root_delta", 3),
                ("root_height", 1), ("dphase", 1),
                ("future_pos", 2 * nf), ("future_dir", 2 * nf),
                ("contacts", self.n_feet)]

    @staticmethod
    def _slices(blocks):
        out, off = {}, 0
        for name, n in blocks:
            out[name] = slice(off, off + n)
            off += n
        return out, off

    @property
    def input_slices(self):
        return self._slices(self.in_blocks)[0]

    @property
    def output_slices(self):
        return self._slices(self.out_blocks)[0]

    @property
    def n_in(self) -> int:
        return self._slices(self.in_blocks)[1]

    @property
    def n_out(self) -> int:
        return self._slices(self.out_blocks)[1]


@dataclass
class RobotClipFeatures:
    traj_pos: np.ndarray        # (T, K, 2) heading-frame root path
    traj_dir: np.ndarray        # (T, K, 2)
    gait: np.ndarray            # (T, K, G)
    joint_pos: np.ndarray       # (T, J)
    joint_vel: np.ndarray       # (T, J)
    root_height: np.ndarray     # (T,)
    gravity: np.ndarray         # (T, 3) in the base frame
    root_lin_vel: np.ndarray    # (T, 3) base frame
    root_ang_vel: np.ndarray    # (T, 3) base frame
    root_delta: np.ndarray      # (T, 3) forward / lateral / dyaw, frame t -> t+1
    phase: np.ndarray           # (T,)
    dphase: np.ndarray          # (T,)
    contacts: np.ndarray        # (T, n_feet)
    name: str = ""


def _heading(quat: np.ndarray) -> np.ndarray:
    """Yaw-only rotation of the base.  The robot pelvis is already canonical
    (+x forward, +z up), so no skeleton convention is involved."""
    fwd = Q.rotate(quat, np.array([1.0, 0.0, 0.0]))
    return Q.yaw_quat(np.arctan2(fwd[..., 1], fwd[..., 0]))


def analyse_robot_clip(qpos: np.ndarray, contacts: np.ndarray, phase: np.ndarray,
                       fps: float, cfg: FeatureConfig | None = None,
                       name: str = "") -> RobotClipFeatures:
    cfg = cfg or FeatureConfig(fps=fps)
    qpos = np.asarray(qpos, dtype=np.float64)
    T = len(qpos)
    root_p, root_q, joints = qpos[:, :3], qpos[:, 3:7], qpos[:, 7:]
    root_q = root_q / np.linalg.norm(root_q, axis=-1, keepdims=True)

    hq = _heading(root_q)
    inv_h = Q.conj(hq)
    inv_q = Q.conj(root_q)

    # trajectory window in the heading frame
    ground = root_p.copy()
    ground[:, 2] = 0.0
    idx = np.clip(np.arange(T)[:, None] + cfg.offsets[None, :], 0, T - 1)
    rel = ground[idx] - ground[:, None, :]
    B = lambda q, shape: np.broadcast_to(q[:, None, :], shape[:-1] + (4,))
    traj_pos = Q.rotate(B(inv_h, rel.shape), rel)[..., :2]

    fwd = Q.rotate(root_q, np.array([1.0, 0.0, 0.0]))
    fwd[:, 2] = 0.0
    fwd /= np.clip(np.linalg.norm(fwd, axis=-1, keepdims=True), 1e-8, None)
    traj_dir = Q.rotate(B(inv_h, fwd[idx].shape), fwd[idx])[..., :2]

    # base-frame quantities: what an IMU would report
    gravity = Q.rotate(inv_q, np.array([0.0, 0.0, -1.0]))
    lin_w = np.gradient(root_p, axis=0) * fps
    root_lin_vel = Q.rotate(inv_q, lin_w)
    dq = Q.mul(root_q[1:], Q.conj(root_q[:-1]))
    ang_w = np.concatenate([Q.to_rotvec(dq) * fps, np.zeros((1, 3))], axis=0)
    root_ang_vel = Q.rotate(inv_q, ang_w)

    joint_vel = np.gradient(joints, axis=0) * fps

    # root motion in the previous frame's heading frame
    yaw = np.arctan2(fwd[:, 1], fwd[:, 0])
    d_world = np.zeros_like(ground)
    d_world[:-1] = ground[1:] - ground[:-1]
    d_local = Q.rotate(inv_h, d_world)
    d_yaw = np.zeros(T)
    d_yaw[:-1] = np.angle(np.exp(1j * (yaw[1:] - yaw[:-1])))
    root_delta = np.stack([d_local[:, 0], d_local[:, 1], d_yaw], axis=-1)

    speed = np.linalg.norm(lin_w[:, :2], axis=-1)
    gait = _gait_from_speed(speed, root_p[:, 2], cfg)
    gait_win = gait[idx]

    dphase = np.zeros(T)
    dphase[:-1] = np.angle(np.exp(1j * (phase[1:] - phase[:-1])))
    dphase[-1] = dphase[-2] if T > 1 else 0.0

    return RobotClipFeatures(traj_pos, traj_dir, gait_win, joints, joint_vel,
                             root_p[:, 2], gravity, root_lin_vel, root_ang_vel,
                             root_delta, phase, dphase,
                             np.asarray(contacts, dtype=np.float64), name)


def _gait_from_speed(speed: np.ndarray, height: np.ndarray,
                     cfg: FeatureConfig) -> np.ndarray:
    names = list(cfg.gaits)
    g = np.zeros((len(speed), len(names)))
    ref_h = np.percentile(height, 90)
    g[:, names.index("stand")] = np.clip(1.0 - speed / 0.3, 0, 1)
    g[:, names.index("walk")] = np.clip(1.0 - np.abs(speed - 1.0) / 0.9, 0, 1)
    g[:, names.index("run")] = np.clip((speed - 1.5) / 1.2, 0, 1)
    if "crouch" in names:
        g[:, names.index("crouch")] = (height < 0.9 * ref_h).astype(float)
    return g / np.clip(g.sum(-1, keepdims=True), 1e-6, None)


def build_robot_samples(f: RobotClipFeatures, spec: RobotFeatureSpec):
    """Cut into (X, Y, phase).  Sample i observes frame i, predicts frame i+1."""
    T = len(f.joint_pos)
    n = T - 1
    flat = lambda a: a[:n].reshape(n, -1)
    X = np.concatenate([
        flat(f.traj_pos), flat(f.traj_dir), flat(f.gait),
        f.joint_pos[:n], f.joint_vel[:n],
        f.root_height[:n, None], f.gravity[:n],
        f.root_lin_vel[:n], f.root_ang_vel[:n],
    ], axis=-1).astype(np.float32)

    nxt = slice(1, n + 1)
    nf = spec.cfg.n_future
    Y = np.concatenate([
        f.joint_pos[nxt], f.joint_vel[nxt], f.root_delta[:n],
        f.root_height[nxt, None], f.dphase[:n, None],
        f.traj_pos[nxt][:, -nf:].reshape(n, -1),
        f.traj_dir[nxt][:, -nf:].reshape(n, -1),
        f.contacts[nxt],
    ], axis=-1).astype(np.float32)

    assert X.shape[1] == spec.n_in, (X.shape[1], spec.n_in)
    assert Y.shape[1] == spec.n_out, (Y.shape[1], spec.n_out)
    return X, Y, f.phase[:n].astype(np.float32)
