"""The tracking policy's environment.

The generator produces a kinematic reference that is plausible for a human and
only approximately possible for a robot.  This env is where that gets resolved:
the policy is rewarded for reproducing the reference *and* for staying upright,
and when those conflict, staying upright wins because the episode ends
otherwise.  That is the entire reason the project is built on a simulator
rather than on a kinematic character controller with a robot mesh.

The contact term is the one worth arguing about.  Animation phase and robot
contact schedule are the same quantity discovered by two fields (see the brief,
gap G6), so the reference's contact labels are supervised directly rather than
left to emerge from position tracking.  In practice this is what stops the
policy from tracking the pose beautifully while skating.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ..motion import quat as Q
from ..sim.env import RobotSim, SimConfig
from ..sim.robot import RobotSpec
from .reference import ReferenceStream


@dataclass
class RewardWeights:
    joint_pos: float = 0.55
    joint_vel: float = 0.05
    key_body: float = 0.35
    root_pos: float = 0.25
    root_ori: float = 0.35
    root_lin_vel: float = 0.20
    root_ang_vel: float = 0.10
    contact: float = 0.25
    action_rate: float = -0.02
    torque: float = -2e-5
    alive: float = 0.15


@dataclass
class TrackerConfig:
    control_hz: float = 50.0
    horizon: int = 3                 # reference frames shown to the policy
    episode_len: int = 500
    max_joint_err: float = 1.2       # rad, mean; beyond this the episode is hopeless
    max_root_err: float = 0.5        # metres
    action_scale: float = 0.25
    obs_history: int = 3
    push_impulse: float = 0.0
    push_interval: float = 4.0
    weights: RewardWeights = field(default_factory=RewardWeights)
    key_bodies: tuple[str, ...] = ()


class TrackerEnv:
    """Gymnasium-style env; kept dependency-light so it vectorises trivially."""

    def __init__(self, spec: RobotSpec, reference: ReferenceStream,
                 cfg: TrackerConfig | None = None, seed: int = 0):
        self.cfg = cfg or TrackerConfig()
        self.spec = spec
        self.ref = reference
        self.sim = RobotSim(spec, SimConfig(control_hz=self.cfg.control_hz,
                                            action_scale=self.cfg.action_scale,
                                            push_impulse=self.cfg.push_impulse,
                                            push_interval=self.cfg.push_interval,
                                            seed=seed))
        self.rng = np.random.default_rng(seed)
        m = self.sim.model
        keys = self.cfg.key_bodies or (spec.foot_bodies + tuple(
            t.body for t in spec.targets.values() if "wrist" in t.body or "elbow" in t.body))
        self.key_bids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in keys]
        self.nu = m.nu
        self.n_joints = m.nq - 7

        self.last_action = np.zeros(self.nu)
        self.history: list[np.ndarray] = []
        self.n_obs = len(self._observe(self.ref._sample(0.0) if hasattr(self.ref, "_sample")
                                       else self.ref.reset(self.rng)))
        self.reset()

    # -- observation --------------------------------------------------------
    def _proprio(self) -> np.ndarray:
        s = self.sim
        return np.concatenate([
            s.joint_pos - s.default_qpos[7:],
            s.joint_vel * 0.05,
            s.projected_gravity(),
            s.data.qvel[3:6] * 0.25,
            s.foot_contacts(),
            self.last_action,
        ])

    def _reference_obs(self, frame) -> np.ndarray:
        """Reference expressed *relative to the robot*, in the robot's frame.

        Absolute reference coordinates would make the policy depend on where in
        the world the clip happened to be recorded; differences in the base
        frame are the only part that generalises.
        """
        s = self.sim
        inv = Q.conj(s.root_quat)
        frames = [frame] + self.ref.horizon(self.cfg.horizon)
        out = [np.sin([frame.phase]), np.cos([frame.phase]), frame.contacts]
        for f in frames:
            out.append(f.qpos[7:] - s.joint_pos)
            out.append(Q.rotate(inv, f.qpos[:3] - s.root_pos))
            out.append(Q.to_rotvec(Q.mul(f.qpos[3:7], inv)))
        out.append(Q.rotate(inv, frame.root_lin_vel) * 0.5)
        out.append(Q.rotate(inv, frame.root_ang_vel) * 0.25)
        return np.concatenate(out)

    def _observe(self, frame) -> np.ndarray:
        return np.concatenate([self._proprio(), self._reference_obs(frame)])

    # -- rollout ------------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        frame = self.ref.reset(self.rng)
        # Reference state initialisation: start the robot *on* the reference,
        # with its velocities, so early transitions are in-distribution.
        qvel = np.zeros(self.sim.model.nv)
        qvel[:3] = frame.root_lin_vel
        qvel[3:6] = frame.root_ang_vel
        self.sim.reset(qpos=frame.qpos.copy(), qvel=qvel)
        self.last_action = np.zeros(self.nu)
        self.step_count = 0
        self.frame = frame
        return self._observe(frame)

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.sim.step(action)
        prev_action, self.last_action = self.last_action, action
        self.frame = self.ref.step()
        self.step_count += 1

        rew, info = self._reward(self.frame, action, prev_action)
        term = self._terminated(self.frame)
        trunc = (self.step_count >= self.cfg.episode_len) or getattr(self.ref, "exhausted", False)
        if term:
            rew -= 1.0
        return self._observe(self.frame), rew, term, trunc, info

    # -- reward -------------------------------------------------------------
    def _reward(self, frame, action, prev_action):
        s, w = self.sim, self.cfg.weights
        jp_err = float(np.abs(s.joint_pos - frame.qpos[7:]).mean())
        jv_err = float(np.abs(s.joint_vel).mean())

        ref_key = self._reference_key_bodies(frame)
        key_err = float(np.linalg.norm(s.body_positions(self.key_bids) - ref_key, axis=-1).mean())

        root_err = float(np.linalg.norm(s.root_pos - frame.qpos[:3]))
        ori_err = float(np.linalg.norm(Q.to_rotvec(Q.mul(frame.qpos[3:7], Q.conj(s.root_quat)))))
        lv_err = float(np.linalg.norm(s.data.qvel[:3] - frame.root_lin_vel))
        av_err = float(np.linalg.norm(s.data.qvel[3:6] - frame.root_ang_vel))
        c_err = float(np.abs(s.foot_contacts() - (frame.contacts > 0.5)).mean())

        g = lambda e, k: float(np.exp(-k * e * e))
        terms = {
            "joint_pos": w.joint_pos * g(jp_err, 8.0),
            "joint_vel": w.joint_vel * g(jv_err, 0.02),
            "key_body": w.key_body * g(key_err, 25.0),
            "root_pos": w.root_pos * g(root_err, 12.0),
            "root_ori": w.root_ori * g(ori_err, 6.0),
            "root_lin_vel": w.root_lin_vel * g(lv_err, 2.0),
            "root_ang_vel": w.root_ang_vel * g(av_err, 0.5),
            "contact": w.contact * (1.0 - c_err),
            "action_rate": w.action_rate * float(np.square(action - prev_action).sum()),
            "torque": w.torque * float(np.square(s.data.actuator_force).sum()),
            "alive": w.alive,
        }
        info = {"errors": {"joint": jp_err, "key_body": key_err, "root": root_err,
                           "orientation": ori_err, "contact": c_err}, "terms": terms}
        return float(sum(terms.values())), info

    def _reference_key_bodies(self, frame) -> np.ndarray:
        """Where the reference puts the key bodies, via forward kinematics.

        Uses a scratch MjData so the live simulation state is untouched.
        """
        if not hasattr(self, "_scratch"):
            self._scratch = mujoco.MjData(self.sim.model)
        d = self._scratch
        d.qpos[:] = frame.qpos
        mujoco.mj_kinematics(self.sim.model, d)
        return np.stack([d.xpos[b] for b in self.key_bids])

    def _terminated(self, frame) -> bool:
        if self.sim.fallen():
            return True
        if np.abs(self.sim.joint_pos - frame.qpos[7:]).mean() > self.cfg.max_joint_err:
            return True
        return bool(np.linalg.norm(self.sim.root_pos[:2] - frame.qpos[:2]) > self.cfg.max_root_err)
