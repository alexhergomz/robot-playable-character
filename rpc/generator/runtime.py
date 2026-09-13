"""The playable character: one autoregressive loop, driven by a command.

This is the layer the operator actually feels.  Every frame it takes the
current command, bends the trajectory window toward it, asks the generator for
the next pose, and integrates the root through the world.  No planning, no
goal, no re-solve — the character is always mid-motion and the command only
ever changes where it is going next.

The output is a stream of *reference* frames.  Nothing here knows about
physics; making the reference physically true is the tracker's job one layer
down.  Keeping that split is what lets the same generator drive a game
character and a robot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..control.command import Command, TrajectoryPredictor
from ..motion import quat as Q
from ..motion.clip import MotionClip, Skeleton
from ..motion.dataset import ClipFeatures, FeatureSpec


@dataclass
class Frame:
    """One generated reference frame, in both local and world coordinates."""

    joint_pos_local: np.ndarray     # (J, 3) character frame
    joint_rot_local: np.ndarray     # (J, 4) character frame
    joint_pos_world: np.ndarray     # (J, 3)
    root_pos: np.ndarray            # (3,) world
    root_yaw: float
    contacts: np.ndarray            # (n_feet,) in [0, 1]
    phase: float
    root_delta: np.ndarray          # (3,) forward, lateral, dyaw for the step just taken


class CharacterController:
    """Runs a trained generator as an interactive controller."""

    def __init__(self, model, spec: FeatureSpec, skeleton: Skeleton,
                 device: str = "cpu", prediction_weight: float = 0.5,
                 max_speed: float = 6.0, max_turn: float = 4.0):
        self.model = model.to(device).eval()
        self.spec = spec
        self.skeleton = skeleton
        self.device = device
        self.prediction_weight = prediction_weight
        self.max_speed = max_speed
        self.max_turn = max_turn
        self.traj = TrajectoryPredictor(spec.cfg)
        self.dt = 1.0 / spec.cfg.fps
        self._isl = spec.input_slices
        self._osl = spec.output_slices
        self.reset()

    # -- state --------------------------------------------------------------
    def reset(self, seed: ClipFeatures | None = None, frame: int = 0) -> None:
        """Start from a real pose if given, otherwise from the rest pose.

        Seeding from data matters: the generator is autoregressive, so an
        implausible first pose is an out-of-distribution input that it never
        recovers from.
        """
        J = self.skeleton.n_joints
        if seed is not None:
            self.joint_pos = seed.joint_pos[frame].copy()
            self.joint_vel = seed.joint_vel[frame].copy()
            self.phase = float(seed.phase[frame])
            self.traj.reset(seed.gait[frame, self.spec.cfg.n_past])
            self.traj.pos[:] = 0.0
            self.traj.dir[:] = np.array([1.0, 0.0])
        else:
            rest = self.skeleton.rest_positions()
            self.joint_pos = rest - rest[0]
            self.joint_vel = np.zeros((J, 3))
            self.phase = 0.0
            self.traj.reset()
        self.root_pos = np.zeros(3)
        self.root_yaw = 0.0
        self.contacts = np.zeros(self.spec.n_feet)
        self.n_steps = 0

    # -- one control step ---------------------------------------------------
    def _assemble_input(self) -> np.ndarray:
        tp, td, tg = self.traj.sample()
        parts = [tp.ravel(), td.ravel(), tg.ravel(),
                 self.joint_pos.ravel(), self.joint_vel.ravel()]
        return np.concatenate(parts).astype(np.float32)

    def step(self, cmd: Command) -> Frame:
        self.traj.apply_command(cmd)
        x = self._assemble_input()

        with torch.no_grad():
            xt = torch.from_numpy(x)[None].to(self.device)
            pt = torch.tensor([self.phase], dtype=torch.float32, device=self.device)
            y = self.model.step(xt, pt)[0].cpu().numpy()

        o = self._osl
        J = self.skeleton.n_joints
        prev_pos = self.joint_pos
        self.joint_pos = y[o["joint_pos"]].reshape(J, 3)
        rot6d = y[o["joint_rot6d"]].reshape(J, 6)
        joint_rot = Q.unflatten_2axis(rot6d)
        root_delta = y[o["root_delta"]].copy()
        # Safety net: a single wild step would otherwise put the whole
        # trajectory buffer out of distribution and never recover.
        planar = float(np.linalg.norm(root_delta[:2])) * self.spec.cfg.fps
        if planar > self.max_speed:
            root_delta[:2] *= self.max_speed / planar
        root_delta[2] = float(np.clip(root_delta[2], -self.max_turn / self.spec.cfg.fps,
                                      self.max_turn / self.spec.cfg.fps))
        dphase = float(y[o["dphase"]][0])

        # Velocity is *derived*, never fed back.  The network predicts position
        # and velocity independently, so nothing forces them to agree; feeding
        # the predicted velocity into the next input closes a loop with no
        # restoring force and it diverges within a few seconds (measured: joint
        # velocity mean 0.18 in training, 2.8 after 100 autoregressive steps).
        # Reconstructing it from the pose difference makes the two consistent by
        # construction.  The transform below carries the previous pose into the
        # new character frame: a world-stationary point maps to zero velocity.
        c_, s_ = np.cos(-root_delta[2]), np.sin(-root_delta[2])
        R = np.array([[c_, -s_, 0.0], [s_, c_, 0.0], [0.0, 0.0, 1.0]])
        shift = R @ np.array([root_delta[0], root_delta[1], 0.0])
        prev_in_frame = prev_pos @ R.T - shift
        self.joint_vel = (self.joint_pos - prev_in_frame) * self.spec.cfg.fps
        fut_p = y[o["future_pos"]].reshape(-1, 2)
        fut_d = y[o["future_dir"]].reshape(-1, 2)
        self.contacts = 1.0 / (1.0 + np.exp(-np.clip(8.0 * (y[o["contacts"]] - 0.5), -30, 30)))

        # integrate the root through the world
        c, s = np.cos(self.root_yaw), np.sin(self.root_yaw)
        self.root_pos = self.root_pos + np.array(
            [c * root_delta[0] - s * root_delta[1],
             s * root_delta[0] + c * root_delta[1], 0.0]
        )
        self.root_yaw = float(np.arctan2(np.sin(self.root_yaw + root_delta[2]),
                                         np.cos(self.root_yaw + root_delta[2])))

        # advance the window and fold the prediction back into the plan
        self.traj.push_past(root_delta)
        self.traj.absorb_prediction(fut_p, fut_d, self.prediction_weight)

        # phase must stay monotone: a negative step means the gait ran backwards,
        # which desynchronises contacts from the tracker below.
        self.phase = float((self.phase + max(dphase, 0.0)) % (2 * np.pi))
        self.n_steps += 1

        hq = Q.yaw_quat(np.array(self.root_yaw))
        world = Q.rotate(np.broadcast_to(hq, (J, 4)), self.joint_pos) + self.root_pos
        return Frame(self.joint_pos.copy(), joint_rot, world,
                     self.root_pos.copy(), self.root_yaw,
                     self.contacts.copy(), self.phase, root_delta)

    def rollout(self, commands, n: int | None = None) -> list[Frame]:
        """Drive the controller from an input device or an iterable of commands."""
        frames = []
        if hasattr(commands, "poll"):
            assert n is not None, "need a length when driving from a device"
            for _ in range(n):
                frames.append(self.step(commands.poll()))
        else:
            for cmd in commands:
                frames.append(self.step(cmd))
        return frames


def frames_to_clip(frames: list[Frame], skeleton: Skeleton, fps: float,
                   name: str = "generated") -> MotionClip:
    """Package a rollout as a MotionClip so it can be measured, retargeted and
    played back through exactly the same code paths as captured data."""
    root_pos = np.stack([f.root_pos for f in frames])
    root_pos[:, 2] = np.stack([f.joint_pos_local[0, 2] for f in frames])
    rots = np.stack([f.joint_rot_local for f in frames])
    hq = np.stack([Q.yaw_quat(np.array(f.root_yaw)) for f in frames])
    rots[:, 0] = Q.mul(hq, rots[:, 0])
    return MotionClip(skeleton, root_pos, Q.normalize(rots), fps, name)
