"""Reference streams — what the tracker is asked to follow.

The tracker must not care whether the reference came from a mocap clip or from
the live generator; that indifference is the whole architecture.  Training uses
:class:`ClipReference` (retargeted mocap, cheap and resettable); deployment uses
:class:`LiveReference` (the operator driving the generator).  Same interface,
same observation, so the policy transfers without retraining.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class RefFrame:
    qpos: np.ndarray            # (nq,) full robot pose, world
    root_lin_vel: np.ndarray    # (3,) world
    root_ang_vel: np.ndarray    # (3,) world
    contacts: np.ndarray        # (n_feet,) desired contact in [0, 1]
    phase: float


class ReferenceStream(Protocol):
    n_feet: int
    dt: float

    def reset(self, rng: np.random.Generator) -> RefFrame: ...
    def step(self) -> RefFrame: ...
    def horizon(self, k: int) -> list[RefFrame]: ...


class ClipReference:
    """A pre-retargeted clip, replayed at the control rate.

    Random start offsets are what make motion tracking learnable: reset state
    initialisation (DeepMimic) turns a long-horizon imitation problem into many
    short ones, and without it the policy only ever sees the first second.
    """

    def __init__(self, qpos: np.ndarray, contacts: np.ndarray, phase: np.ndarray,
                 fps: float, control_hz: float, episode_len: int = 400):
        self.qpos_src = np.asarray(qpos, dtype=np.float64)
        self.contacts_src = np.asarray(contacts, dtype=np.float64)
        self.phase_src = np.asarray(phase, dtype=np.float64)
        self.fps = fps
        self.dt = 1.0 / control_hz
        self.episode_len = episode_len
        self.n_feet = self.contacts_src.shape[1]
        self._src_dt = 1.0 / fps
        self.t = 0.0
        self._vel = self._finite_difference()

    def _finite_difference(self):
        from ..motion import quat as Q

        lin = np.gradient(self.qpos_src[:, :3], axis=0) * self.fps
        q = self.qpos_src[:, 3:7]
        dq = Q.mul(q[1:], Q.conj(q[:-1]))
        rot = Q.to_rotvec(dq) * self.fps
        ang = np.concatenate([rot, rot[-1:]], axis=0)
        return lin, ang

    def _sample(self, t: float) -> RefFrame:
        from ..motion import quat as Q

        n = len(self.qpos_src)
        f = np.clip(t * self.fps, 0, n - 1 - 1e-6)
        i0 = int(f)
        i1 = min(i0 + 1, n - 1)
        a = f - i0
        qpos = self.qpos_src[i0] * (1 - a) + self.qpos_src[i1] * a
        qpos[3:7] = Q.slerp(self.qpos_src[i0, 3:7], self.qpos_src[i1, 3:7], a)
        lin, ang = self._vel
        return RefFrame(qpos,
                        lin[i0] * (1 - a) + lin[i1] * a,
                        ang[i0] * (1 - a) + ang[i1] * a,
                        self.contacts_src[i0] * (1 - a) + self.contacts_src[i1] * a,
                        float(self.phase_src[i0]))

    def reset(self, rng: np.random.Generator) -> RefFrame:
        span = max(1e-3, (len(self.qpos_src) - 1) / self.fps - self.episode_len * self.dt)
        self.t = float(rng.uniform(0, span))
        return self._sample(self.t)

    def step(self) -> RefFrame:
        self.t += self.dt
        return self._sample(self.t)

    def horizon(self, k: int) -> list[RefFrame]:
        return [self._sample(self.t + (i + 1) * self.dt) for i in range(k)]

    @property
    def exhausted(self) -> bool:
        return self.t * self.fps >= len(self.qpos_src) - 2


class ClipLibraryReference:
    """Samples a clip per episode.  Keeps the policy from overfitting one gait."""

    def __init__(self, streams: list[ClipReference]):
        assert streams
        self.streams = streams
        self.active = streams[0]
        self.n_feet = streams[0].n_feet
        self.dt = streams[0].dt

    def reset(self, rng: np.random.Generator) -> RefFrame:
        self.active = self.streams[rng.integers(len(self.streams))]
        return self.active.reset(rng)

    def step(self) -> RefFrame:
        return self.active.step()

    def horizon(self, k: int):
        return self.active.horizon(k)

    @property
    def exhausted(self) -> bool:
        return self.active.exhausted


class LiveReference:
    """The generator, driven by an operator, retargeted on the fly.

    This is the deployment path.  It is deliberately the same object type as
    the training path so that the only thing that changes between "imitating a
    clip" and "being driven by a human" is which stream is plugged in.
    """

    def __init__(self, controller, retargeter, input_device, control_hz: float,
                 arbiter=None, language_command=None):
        self.controller = controller
        self.retargeter = retargeter
        self.input = input_device
        self.arbiter = arbiter
        self.language_command = language_command
        self.dt = 1.0 / control_hz
        self.n_feet = 2
        self._gen_dt = 1.0 / controller.spec.cfg.fps
        self._acc = 0.0
        self._last = None

    def _advance(self) -> RefFrame:
        from ..motion import quat as Q

        stick = self.input.poll()
        cmd = (self.arbiter.update(self.language_command, stick)
               if self.arbiter is not None and self.language_command is not None else stick)
        frame = self.controller.step(cmd)
        q = self.retargeter.solve(frame.joint_pos_local, frame.root_yaw, frame.root_pos[:2])
        prev = self._last
        lin = np.zeros(3) if prev is None else (q[:3] - prev.qpos[:3]) / self._gen_dt
        ang = np.zeros(3) if prev is None else Q.to_rotvec(
            Q.mul(q[3:7], Q.conj(prev.qpos[3:7]))) / self._gen_dt
        self._last = RefFrame(q.copy(), lin, ang, frame.contacts, frame.phase)
        return self._last

    def reset(self, rng: np.random.Generator) -> RefFrame:
        _ = rng
        self.controller.reset()
        self._last = None
        self._acc = 0.0
        return self._advance()

    def step(self) -> RefFrame:
        # The generator runs at 30 Hz and the tracker at 50 Hz; hold the last
        # reference between generator frames rather than interpolating, because
        # interpolation would add a frame of latency to the operator's input.
        self._acc += self.dt
        while self._acc >= self._gen_dt:
            self._acc -= self._gen_dt
            self._advance()
        return self._last

    def horizon(self, k: int) -> list[RefFrame]:
        return [self._last] * k          # the future is not known; the operator owns it

    @property
    def exhausted(self) -> bool:
        return False
