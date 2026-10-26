"""Stream the generator's reference motion to the G1 deploy app.

The generator runs open loop, as everywhere else in this project; the deploy
app's tracker (SONIC) closes the loop on the robot.  This module only turns
operator commands into a stream of reference frames and enforces the rules the
receiver needs and the robot deserves.

Rules, each tied to a measured property of the receiver:

* **Frame indices follow the wall clock at the control rate.**  The receiver's
  merger keeps a sliding window and *forces a catch-up reset* -- a jump of the
  playback cursor -- whenever an incoming chunk does not end strictly later
  than the previous one.  Two sends inside one 20 ms tick would do exactly
  that, so a send is skipped unless the clock has advanced.
* **Each chunk regenerates the whole future horizon under the live command.**
  The merger keeps old frames only up to where the new chunk starts, so newer
  frames replace stale ones; this is what makes edits take effect immediately.
* **The stream never starts the controller.**  ``command`` messages can, so
  ``start`` exists only as an explicit, separate call; nothing here sends it.
* **Every message ends in a stand (the dead-man tail).**  When the stream
  stops, the receiver plays to the end of its window and then *holds the
  final frame*.  Killed mid-walk, that left a frozen stride, which SONIC
  could not hold still: in simulation the robot kept stepping for 8 s and
  travelled 1.8 m nobody commanded.  So each message carries the horizon and
  then a smooth blend into the standing pose.  The tracker only consumes the
  frames just ahead of its cursor, so while the stream is alive the tail is
  never played; if the stream dies, lags or loses the network, the robot
  finishes its stride, eases into a stand and holds it.  A graceful stop is
  therefore just: brake, then stop sending.
* **Every outgoing frame is checked**: finite, inside the joint limits, and
  produced under a capped speed.  Non-finite output is never sent.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..control.command import Command
from ..motion import quat as Q
from . import wire


@dataclass
class StreamLimits:
    #: Commanded planar speed cap, m/s.  Deliberately far below the ~1.0 m/s the
    #: simulation sustains: first hardware trials should be slow.
    max_speed: float = 0.3
    max_turn: float = 0.5               #: rad/s
    #: Length of the dead-man tail appended after the horizon, s.
    tail_s: float = 1.0
    #: Before stopping gracefully, stream a zero-velocity command this long so
    #: the robot decelerates under the generator rather than inside the tail.
    brake_s: float = 1.5


@dataclass
class StreamStats:
    sent: int = 0
    skipped_same_tick: int = 0
    clipped_frames: int = 0
    max_clip: float = 0.0
    last_end: int = -1
    gen_ms: list = field(default_factory=list)


class PoseStreamer:
    """Turns ``Command``s into protocol-v1 pose messages for the deploy app.

    ``sink`` receives finished message bytes: a ZMQ PUB socket's ``send`` on the
    robot, a list's ``append`` in tests and dry runs.
    """

    def __init__(self, generator, sink: Callable[[bytes], None], *,
                 joint_lower: np.ndarray, joint_upper: np.ndarray,
                 mujoco_to_isaaclab: np.ndarray, horizon: int = 12,
                 hz: float = 50.0, limits: StreamLimits | None = None,
                 stand_pose: np.ndarray | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.gen = generator
        self.sink = sink
        self.lo = np.asarray(joint_lower, float)
        self.hi = np.asarray(joint_upper, float)
        self.remap = np.asarray(mujoco_to_isaaclab)
        self.horizon = int(horizon)
        if stand_pose is None:
            raise ValueError("stand_pose is required: it is where every stream ends")
        self.stand = np.asarray(stand_pose, float)
        self.hz = float(hz)
        self.limits = limits or StreamLimits()
        self.clock = clock
        self.stats = StreamStats()
        self._t0: float | None = None
        self._frame = 0                  # frame the generator's state corresponds to
        self.last_speed = float("inf")

    # -- command shaping -------------------------------------------------------
    def _safe(self, cmd: Command) -> Command:
        c = copy.copy(cmd)
        sp = float(np.linalg.norm(cmd.move))
        if sp > self.limits.max_speed:
            c.move = cmd.move * (self.limits.max_speed / sp)
        c.turn = float(np.clip(cmd.turn, -self.limits.max_turn, self.limits.max_turn))
        return c

    # -- frames ---------------------------------------------------------------
    def _clock_frame(self) -> int:
        if self._t0 is None:
            self._t0 = self.clock()
        return int((self.clock() - self._t0) * self.hz)

    def _horizon_frames(self, cmd: Command):
        snap = copy.deepcopy(self.gen)
        prev = self.gen.joint_pos.copy()
        pos, yaw = [], []
        for _ in range(self.horizon):
            f = snap.step(cmd)
            pos.append(f.joint_pos.copy())
            yaw.append(f.root_yaw)
        pos = np.stack(pos)
        vel = np.diff(pos, axis=0, prepend=prev[None]) * self.hz
        quat = np.stack([Q.yaw_quat(np.array(y)) for y in yaw])
        return pos, vel, quat

    def _tail(self, pos: np.ndarray, quat: np.ndarray):
        """Smoothstep blend from the horizon's last frame to the stand pose."""
        n = max(1, int(round(self.limits.tail_s * self.hz)))
        a = np.linspace(0.0, 1.0, n + 1)[1:]
        a = a * a * (3 - 2 * a)
        tp = (1 - a)[:, None] * pos[-1][None] + a[:, None] * self.stand[None]
        tq = np.repeat(quat[-1:], n, axis=0)           # keep heading; stop turning
        return tp, tq

    def _check(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        if not (np.isfinite(pos).all() and np.isfinite(vel).all()):
            raise FloatingPointError("generator produced a non-finite reference; "
                                     "nothing was sent")
        clipped = np.clip(pos, self.lo, self.hi)
        over = np.abs(clipped - pos).max()
        if over > 0:
            self.stats.clipped_frames += int((np.abs(clipped - pos) > 0).any(1).sum())
            self.stats.max_clip = max(self.stats.max_clip, float(over))
        return clipped

    # -- the loop body ----------------------------------------------------------
    def tick(self, cmd: Command) -> bool:
        """Advance to the current clock frame and send the horizon.  Returns
        False when the clock has not advanced and nothing was sent."""
        k = self._clock_frame()
        if k + self.horizon + int(round(self.limits.tail_s * self.hz)) <= self.stats.last_end:
            self.stats.skipped_same_tick += 1
            return False
        c = self._safe(cmd)
        while self._frame < k:                   # catch the generator up to now
            self.gen.step(c)
            self._frame += 1
        t = time.perf_counter()
        pos, vel, quat = self._horizon_frames(c)
        self.stats.gen_ms.append((time.perf_counter() - t) * 1e3)
        self.last_speed = float(np.abs(vel[0]).mean())   # rad/s of the next frame
        tp, tq = self._tail(pos, quat)
        prev = self.gen.joint_pos.copy()
        pos = np.concatenate([pos, tp])
        quat = np.concatenate([quat, tq])
        vel = np.diff(pos, axis=0, prepend=prev[None]) * self.hz
        pos = self._check(pos, vel)
        idx = np.arange(k + 1, k + 1 + len(pos), dtype=np.int64)
        self.sink(wire.pose_message(pos, vel, quat, idx, self.remap))
        self.stats.sent += 1
        self.stats.last_end = int(idx[-1])
        return True

    def brake(self, sleep: Callable[[float], None] = time.sleep) -> None:
        """Graceful stop: stream zero velocity for ``brake_s``, then return.

        The caller then simply stops sending; the dead-man tail of the last
        message carries the robot into the stand pose and the receiver holds it.
        """
        zero = Command(move=np.zeros(2), turn=0.0)
        end = self.clock() + self.limits.brake_s
        while self.clock() < end:
            self.tick(zero)
            sleep(1.0 / (2 * self.hz))
