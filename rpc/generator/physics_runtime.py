"""PFNN as a closed-loop controller in physics.

One network sits between the operator's stick and the actuators.  There is no
tracker, no retargeting at run time and no reference stream: the generator reads
the simulator's state, and its predicted next pose *is* the position command.

The difference from :mod:`rpc.generator.runtime` is entirely in where the inputs
come from.  The kinematic controller feeds itself — its own predicted pose and
its own predicted root motion — which makes it a playback device that happens to
be steerable.  Here the joint angles, base orientation, velocities and root
displacement are all **measured**, so an error does not accumulate silently; it
appears in the next observation and the network gets a chance to answer it.
That is what turns a regression model into feedback control, and it is why the
state includes projected gravity and base height at all.

Trained purely by regression on recorded motion, this is still an open-loop
learner driven closed-loop, so it inherits the usual distribution shift.  The
cure is more data from its own state distribution (a DAgger pass), not a change
of algorithm — no reward, no exploration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..control.command import Command, TrajectoryPredictor
from ..motion import quat as Q
from ..motion.robot_dataset import RobotFeatureSpec


@dataclass
class PhysicsFrame:
    qpos: np.ndarray
    targets: np.ndarray          # what was commanded to the actuators
    contacts: np.ndarray
    phase: float
    root_delta: np.ndarray       # measured, in the previous heading frame
    height: float
    upright: float               # -gravity_z, 1.0 when perfectly upright


class PhysicsCharacterController:
    """Drives a :class:`RobotSim` from a robot-space PFNN."""

    def __init__(self, model, spec: RobotFeatureSpec, sim, device: str = "cpu",
                 action_filter: float = 0.0):
        self.model = model.to(device).eval()
        self.spec = spec
        self.sim = sim
        self.device = device
        self.action_filter = action_filter
        self.traj = TrajectoryPredictor(spec.cfg)
        self._isl = spec.input_slices
        self._osl = spec.output_slices
        self.reset()

    # -- state ---------------------------------------------------------------
    def reset(self, qpos: np.ndarray | None = None, qvel: np.ndarray | None = None,
              phase: float = 0.0, gait: np.ndarray | None = None) -> None:
        self.sim.reset(qpos=qpos, qvel=qvel)
        self.phase = float(phase)
        self.traj.reset(gait)
        self._prev = self._measure_frame()
        self.targets = self.sim.joint_pos.copy()
        self.n_steps = 0

    def _measure_frame(self):
        s = self.sim
        q = s.root_quat
        fwd = Q.rotate(q, np.array([1.0, 0.0, 0.0]))
        yaw = float(np.arctan2(fwd[1], fwd[0]))
        return {"xy": s.root_pos[:2].copy(), "yaw": yaw}

    def _measured_root_delta(self) -> np.ndarray:
        """Displacement since the previous control step, in that step's frame."""
        now = self._measure_frame()
        d = now["xy"] - self._prev["xy"]
        c, s_ = np.cos(-self._prev["yaw"]), np.sin(-self._prev["yaw"])
        local = np.array([c * d[0] - s_ * d[1], s_ * d[0] + c * d[1]])
        dyaw = float(np.angle(np.exp(1j * (now["yaw"] - self._prev["yaw"]))))
        self._prev = now
        return np.array([local[0], local[1], dyaw])

    def observe(self) -> np.ndarray:
        s = self.sim
        q = s.root_quat
        inv = Q.conj(q)
        gravity = Q.rotate(inv, np.array([0.0, 0.0, -1.0]))
        lin = Q.rotate(inv, s.data.qvel[:3].copy())
        ang = Q.rotate(inv, s.data.qvel[3:6].copy())
        tp, td, tg = self.traj.sample()
        return np.concatenate([
            tp.ravel(), td.ravel(), tg.ravel(),
            s.joint_pos, s.joint_vel,
            [s.root_pos[2]], gravity, lin, ang,
        ]).astype(np.float32)

    # -- one control step ----------------------------------------------------
    def step(self, cmd: Command) -> PhysicsFrame:
        self.traj.apply_command(cmd)
        x = self.observe()

        with torch.no_grad():
            xt = torch.from_numpy(x)[None].to(self.device)
            pt = torch.tensor([self.phase], dtype=torch.float32, device=self.device)
            y = self.model.step(xt, pt)[0].cpu().numpy()

        o = self._osl
        targets = y[o["joint_pos"]].copy()
        if self.action_filter > 0:
            a = float(np.clip(self.action_filter, 0.0, 1.0))
            targets = a * self.targets + (1 - a) * targets
        self.targets = targets

        self.sim.step_position(targets)

        # everything below is *measured*, which is the whole point
        root_delta = self._measured_root_delta()
        self.traj.push_past(root_delta)
        fut_p = y[o["future_pos"]].reshape(-1, 2)
        fut_d = y[o["future_dir"]].reshape(-1, 2)
        self.traj.absorb_prediction(fut_p, fut_d, 0.3)

        self.phase = float((self.phase + max(float(y[o["dphase"]][0]), 0.0)) % (2 * np.pi))
        self.n_steps += 1

        g = Q.rotate(Q.conj(self.sim.root_quat), np.array([0.0, 0.0, -1.0]))
        return PhysicsFrame(self.sim.data.qpos.copy(), targets,
                            self.sim.foot_contacts(), self.phase, root_delta,
                            float(self.sim.root_pos[2]), float(-g[2]))

    @property
    def fallen(self) -> bool:
        return self.sim.fallen()
