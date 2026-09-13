"""MuJoCo wrapper: the physics the reference has to survive.

Thin on purpose.  It owns the model, the data, the action convention and the
disturbances — everything about *this* robot in *this* world — and nothing
about motion, commands or learning.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .robot import RobotSpec


@dataclass
class SimConfig:
    control_hz: float = 50.0        # policy rate; physics runs at the model timestep
    action_scale: float = 0.25      # rad, added to the default pose
    push_interval: float = 4.0      # seconds between random shoves (0 disables)
    push_impulse: float = 0.0       # N.s applied to the pelvis
    seed: int = 0


class RobotSim:
    """A single MuJoCo instance with a position-servo action convention.

    Actions are *offsets from the default pose*, not absolute targets.  That
    keeps the zero action at a standing posture, which means an untrained or
    saturated policy fails by standing still rather than by folding up — and it
    is the convention every sim-to-real humanoid paper uses, so a policy trained
    here is portable.
    """

    def __init__(self, spec: RobotSpec, cfg: SimConfig | None = None):
        self.spec = spec
        self.cfg = cfg or SimConfig()
        self.model = mujoco.MjModel.from_xml_path(spec.path())
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(self.cfg.seed)

        self.n_sub = max(1, int(round(1.0 / (self.cfg.control_hz * self.model.opt.timestep))))
        self.dt = self.n_sub * self.model.opt.timestep

        key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, spec.keyframe)
        self.key_id = max(key, 0)
        self.default_qpos = np.array(self.model.key_qpos[self.key_id])
        # actuators are position servos on named joints; map ctrl -> qpos address
        self.act_qadr = np.array([
            self.model.jnt_qposadr[self.model.actuator_trnid[i, 0]] for i in range(self.model.nu)
        ])
        self.default_ctrl = self.default_qpos[self.act_qadr].copy()
        self.ctrl_range = self.model.actuator_ctrlrange.copy()

        self.root_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, spec.root_body)
        self.foot_bids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, b)
                          for b in spec.foot_bodies]
        self._t_push = 0.0
        self.reset()

    # -- lifecycle ----------------------------------------------------------
    def reset(self, qpos: np.ndarray | None = None, qvel: np.ndarray | None = None) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)
        if qpos is not None:
            self.data.qpos[:] = qpos
        if qvel is not None:
            self.data.qvel[:] = qvel
        else:
            self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.default_ctrl
        mujoco.mj_forward(self.model, self.data)
        self._t_push = 0.0

    def step(self, action: np.ndarray) -> None:
        target = self.default_ctrl + np.clip(action, -1.0, 1.0) * self.cfg.action_scale
        self.data.ctrl[:] = np.clip(target, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self._maybe_push()

    def step_position(self, targets: np.ndarray) -> None:
        """Command absolute joint positions.

        The offset-from-default convention exists so that a *policy* fails by
        standing still.  A generator that predicts poses outright has no such
        origin, so it drives the servos directly.
        """
        self.data.ctrl[:] = np.clip(targets, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)
        self._maybe_push()

    def _maybe_push(self) -> None:
        if self.cfg.push_impulse <= 0 or self.cfg.push_interval <= 0:
            return
        self._t_push += self.dt
        if self._t_push >= self.cfg.push_interval:
            self._t_push = 0.0
            theta = self.rng.uniform(0, 2 * np.pi)
            f = self.cfg.push_impulse / self.dt
            self.data.xfrc_applied[self.root_bid, :3] = [f * np.cos(theta), f * np.sin(theta), 0.0]
        elif self._t_push > self.dt * 2:
            self.data.xfrc_applied[self.root_bid, :3] = 0.0

    # -- observation helpers ------------------------------------------------
    @property
    def root_pos(self) -> np.ndarray:
        return self.data.qpos[:3].copy()

    @property
    def root_quat(self) -> np.ndarray:
        return self.data.qpos[3:7].copy()

    @property
    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[7:].copy()

    @property
    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[6:].copy()

    def body_positions(self, bids) -> np.ndarray:
        return np.stack([self.data.xpos[b] for b in bids])

    def projected_gravity(self) -> np.ndarray:
        """Gravity in the base frame — the standard orientation observation,
        and the one that survives sim-to-real because an IMU measures it."""
        q = self.root_quat
        R = np.empty(9)
        mujoco.mju_quat2Mat(R, q)
        return R.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])

    def foot_contacts(self) -> np.ndarray:
        """Per-foot contact from the actual contact list, not a height heuristic."""
        out = np.zeros(len(self.foot_bids))
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            for b in (self.model.geom_bodyid[c.geom1], self.model.geom_bodyid[c.geom2]):
                for k, fb in enumerate(self.foot_bids):
                    if b == fb or self._is_descendant(b, fb):
                        out[k] = 1.0
        return out

    def _is_descendant(self, body: int, ancestor: int) -> bool:
        while body > 0:
            if body == ancestor:
                return True
            body = self.model.body_parentid[body]
        return False

    def fallen(self, min_height: float | None = None) -> bool:
        h = min_height if min_height is not None else 0.55 * self.spec.stand_height
        return bool(self.data.qpos[2] < h or self.projected_gravity()[2] > -0.3)
