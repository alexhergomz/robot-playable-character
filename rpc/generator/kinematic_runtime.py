"""Open-loop PFNN in robot joint space — the animator, with no physics.

This is PFNN used the way games use it: the network feeds itself.  Its own
predicted pose becomes the next input, the root is integrated from its own
predicted delta, and nothing measures anything.  The output is a *reference*
motion — "I want this pose at this time" — which a separate physical controller
is then responsible for realising.

Running it this way is also the only clean test of whether the generator is a
good animator, separate from whether anything can execute it.

One gap this exposes: the network predicts root height and a planar root delta,
but **not base pitch and roll**, so the projected-gravity input has to be
supplied as nominal upright.  In training that channel carried the reference's
few degrees of tilt, so the rollout is slightly off-distribution there, and a
downstream controller gets no base-orientation reference at all.  For an
open-loop generator feeding a whole-body controller, base orientation should be
an output.  See `EXPECTED_OUTPUT_GAPS`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..control.command import Command, TrajectoryPredictor
from ..motion import quat as Q
from ..motion.robot_dataset import RobotFeatureSpec

EXPECTED_OUTPUT_GAPS = ("base_orientation",)


@dataclass
class KinematicFrame:
    joint_pos: np.ndarray        # (J,) reference joint angles
    joint_vel: np.ndarray        # (J,)
    root_xy: np.ndarray          # (2,) integrated world position
    root_yaw: float
    root_height: float
    contacts: np.ndarray
    phase: float
    root_delta: np.ndarray       # (fwd, lat, dyaw) for the step just taken

    def qpos(self, n_joints: int) -> np.ndarray:
        q = np.zeros(7 + n_joints)
        q[:2] = self.root_xy
        q[2] = self.root_height
        q[3:7] = Q.yaw_quat(np.array(self.root_yaw))
        q[7:] = self.joint_pos
        return q


class KinematicCharacterController:
    """Autoregressive robot-space PFNN.  No simulator, no measurement."""

    def __init__(self, model, spec: RobotFeatureSpec, device: str = "cpu",
                 prediction_weight: float = 0.3, max_speed: float = 4.0):
        self.model = model.to(device).eval()
        self.spec = spec
        self.device = device
        self.prediction_weight = prediction_weight
        self.max_speed = max_speed
        self.dt = 1.0 / spec.cfg.fps
        self.traj = TrajectoryPredictor(spec.cfg)
        self._osl = spec.output_slices
        self.reset()

    def reset(self, joint_pos=None, joint_vel=None, root_height=0.75,
              phase=0.0, gait=None) -> None:
        J = self.spec.n_joints
        self.joint_pos = np.zeros(J) if joint_pos is None else np.array(joint_pos, float)
        self.joint_vel = np.zeros(J) if joint_vel is None else np.array(joint_vel, float)
        self.root_height = float(root_height)
        self.root_xy = np.zeros(2)
        self.root_yaw = 0.0
        self.phase = float(phase)
        self.lin_vel = np.zeros(3)
        self.ang_vel = np.zeros(3)
        self.contacts = np.zeros(self.spec.n_feet)
        self.traj.reset(gait)
        self.n_steps = 0

    def observe(self) -> np.ndarray:
        tp, td, tg = self.traj.sample()
        # base orientation is not predicted, so gravity is supplied as upright
        gravity = np.array([0.0, 0.0, -1.0])
        return np.concatenate([
            tp.ravel(), td.ravel(), tg.ravel(),
            self.joint_pos, self.joint_vel,
            [self.root_height], gravity, self.lin_vel, self.ang_vel,
        ]).astype(np.float32)

    def step(self, cmd: Command) -> KinematicFrame:
        self.traj.apply_command(cmd)
        x = self.observe()
        with torch.no_grad():
            y = self.model.step(torch.from_numpy(x)[None].to(self.device),
                                torch.tensor([self.phase], dtype=torch.float32,
                                             device=self.device))[0].cpu().numpy()
        o = self._osl
        prev = self.joint_pos
        self.joint_pos = y[o["joint_pos"]].copy()
        # Velocity is derived, never fed back: the network predicts position and
        # velocity independently and nothing forces them to agree, so feeding the
        # predicted velocity closes a loop with no restoring force.
        self.joint_vel = (self.joint_pos - prev) * self.spec.cfg.fps
        self.root_height = float(y[o["root_height"]][0])

        rd = y[o["root_delta"]].copy()
        planar = float(np.linalg.norm(rd[:2])) * self.spec.cfg.fps
        if planar > self.max_speed:
            rd[:2] *= self.max_speed / planar

        c, s = np.cos(self.root_yaw), np.sin(self.root_yaw)
        self.root_xy = self.root_xy + np.array([c * rd[0] - s * rd[1],
                                                s * rd[0] + c * rd[1]])
        self.root_yaw = float(np.arctan2(np.sin(self.root_yaw + rd[2]),
                                         np.cos(self.root_yaw + rd[2])))
        self.lin_vel = np.array([rd[0], rd[1], 0.0]) * self.spec.cfg.fps
        self.ang_vel = np.array([0.0, 0.0, rd[2]]) * self.spec.cfg.fps

        self.traj.push_past(rd)
        self.traj.absorb_prediction(y[o["future_pos"]].reshape(-1, 2),
                                    y[o["future_dir"]].reshape(-1, 2),
                                    self.prediction_weight)
        self.contacts = 1.0 / (1.0 + np.exp(-np.clip(8.0 * (y[o["contacts"]] - 0.5), -30, 30)))
        self.phase = float((self.phase + max(float(y[o["dphase"]][0]), 0.0)) % (2 * np.pi))
        self.n_steps += 1
        return KinematicFrame(self.joint_pos.copy(), self.joint_vel.copy(),
                              self.root_xy.copy(), self.root_yaw, self.root_height,
                              self.contacts.copy(), self.phase, rd)
