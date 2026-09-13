"""Operator input devices, all producing the same :class:`Command`.

A gamepad is the reference device — two sticks map cleanly onto planar velocity
and turn rate, and the analog range is the point.  Keyboard is the fallback and
is explicitly worse: it quantises the very channel the argument depends on.
:class:`ScriptedInput` exists so the loop can be tested and benchmarked without
a human, which is what every experiment in this project needs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..motion.features import GAITS
from .command import FEASIBLE_COMMAND_MAX, Command


@dataclass
class InputConfig:
    #: Full stick = the fastest the robot measurably delivers.  Mapping the
    #: stick to a speed it cannot reach spends most of its travel on a
    #: region where extra command produces foot slip rather than motion.
    max_speed: float = FEASIBLE_COMMAND_MAX      # m/s at full stick
    max_turn: float = 2.5           # rad/s at full stick
    deadzone: float = 0.10
    expo: float = 1.6               # >1 gives fine control near centre


def _shape(v: float, cfg: InputConfig) -> float:
    if abs(v) < cfg.deadzone:
        return 0.0
    s = (abs(v) - cfg.deadzone) / (1 - cfg.deadzone)
    return float(np.sign(v) * s ** cfg.expo)


class GamepadInput:
    """pygame joystick -> Command.  Falls back to keyboard when no pad is found."""

    def __init__(self, cfg: InputConfig | None = None):
        self.cfg = cfg or InputConfig()
        self.available = False
        self.mode = "none"
        try:
            import pygame

            pygame.init()
            pygame.joystick.init()
            self._pg = pygame
            if pygame.joystick.get_count() > 0:
                self.pad = pygame.joystick.Joystick(0)
                self.pad.init()
                self.available = True
                self.mode = f"gamepad:{self.pad.get_name()}"
            else:
                self.mode = "keyboard"
                self.available = True
        except Exception as exc:                       # pragma: no cover - device dependent
            self.mode = f"unavailable ({exc})"

    def poll(self) -> Command:
        if not self.available:
            return Command(move=np.zeros(2), turn=0.0)
        self._pg.event.pump()
        if self.mode.startswith("gamepad"):
            lx, ly = self.pad.get_axis(0), -self.pad.get_axis(1)
            rx = self.pad.get_axis(3) if self.pad.get_numaxes() > 3 else 0.0
            trig = (self.pad.get_axis(5) + 1) / 2 if self.pad.get_numaxes() > 5 else 0.0
            move = np.array([_shape(ly, self.cfg), -_shape(lx, self.cfg)]) * self.cfg.max_speed
            turn = -_shape(rx, self.cfg) * self.cfg.max_turn
            resp = 0.35 + 0.6 * trig       # right trigger = "be snappier"
            return Command(move=move, turn=turn, responsiveness=float(resp))

        keys = self._pg.key.get_pressed()
        K = self._pg.locals
        fwd = keys[K.K_w] - keys[K.K_s]
        lat = keys[K.K_a] - keys[K.K_d]
        turn = keys[K.K_q] - keys[K.K_e]
        boost = 1.0 + 0.9 * keys[K.K_LSHIFT]
        return Command(
            move=np.array([fwd, lat], dtype=float) * self.cfg.max_speed * 0.45 * boost,
            turn=float(turn) * self.cfg.max_turn * 0.6,
            responsiveness=0.85 if keys[K.K_SPACE] else 0.5,
        )


class ScriptedInput:
    """A deterministic operator, for tests and for measuring the thing we claim.

    ``script`` is a list of (duration_seconds, Command) pairs; the input holds
    each command for its duration and then returns to neutral.
    """

    def __init__(self, script: list[tuple[float, Command]], dt: float = 1 / 30):
        self.script = script
        self.dt = dt
        self.t = 0.0
        self.mode = "scripted"
        self.available = True

    def poll(self) -> Command:
        acc = 0.0
        for dur, cmd in self.script:
            acc += dur
            if self.t < acc:
                self.t += self.dt
                return cmd
        self.t += self.dt
        return Command(move=np.zeros(2), turn=0.0, gait=np.eye(len(GAITS))[0])

    def reset(self) -> None:
        self.t = 0.0
