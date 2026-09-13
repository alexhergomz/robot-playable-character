"""Channel arbitration — the part of the system the literature does not have.

Two channels, two clocks.  Language is coarse, arrives every few seconds, and
describes *what*.  The stick is fine, arrives every frame, and describes *how*
and *where*.  The contract:

    the stick always wins, immediately, and language resumes only after the
    operator has visibly let go.

That asymmetry is deliberate.  A shared-autonomy arbitration that reasons about
whose intent is more likely (Dragan & Srinivasa 2013, Javdani et al. 2015) is
the right answer when the human's goal is unknown and the robot's inference is
good.  Here the human is driving a character in real time, and any moment where
the machine second-guesses the stick is a moment the operator experiences as the
robot fighting them.  So authority is not inferred, it is asserted — but it is
released *slowly*, because a hard switch back to autonomy the instant the thumb
leaves the stick is equally unpleasant.

Every handover is logged.  An operator driving a character generates a
continuous record of where autonomy was insufficient, which is the training
signal Sirius-style intervention learning uses and which nobody collects for
whole-body control.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .command import Command


@dataclass
class ArbiterConfig:
    deadzone: float = 0.12          # stick magnitude below which we call it "released"
    attack: float = 0.05            # seconds to full stick authority
    release: float = 0.8            # seconds to hand authority back to language
    hold: float = 0.4               # seconds of stick idle before release even begins
    style_from_language: bool = True  # language keeps the style channel even under stick


@dataclass
class Handover:
    t: float
    direction: str                  # "to_operator" | "to_language"
    language_intent: str
    stick: tuple[float, float, float]


class ChannelArbiter:
    """Blends a language-derived command with a live stick command."""

    def __init__(self, cfg: ArbiterConfig | None = None, dt: float = 1 / 30):
        self.cfg = cfg or ArbiterConfig()
        self.dt = dt
        self.authority = 0.0        # 0 = language drives, 1 = operator drives
        self.idle = 0.0
        self.t = 0.0
        self.log: list[Handover] = []
        self._last_state = "language"

    @staticmethod
    def stick_magnitude(stick: Command) -> float:
        return float(max(np.linalg.norm(stick.move), abs(stick.turn)))

    def update(self, language: Command, stick: Command,
               language_intent: str = "") -> Command:
        """Return the command the generator should act on this frame."""
        self.t += self.dt
        active = self.stick_magnitude(stick) > self.cfg.deadzone

        if active:
            self.idle = 0.0
            self.authority = min(1.0, self.authority + self.dt / max(self.cfg.attack, 1e-6))
        else:
            self.idle += self.dt
            if self.idle > self.cfg.hold:
                self.authority = max(0.0, self.authority - self.dt / max(self.cfg.release, 1e-6))

        state = "operator" if self.authority > 0.5 else "language"
        if state != self._last_state:
            self.log.append(Handover(
                self.t,
                "to_operator" if state == "operator" else "to_language",
                language_intent,
                (float(stick.move[0]), float(stick.move[1]), float(stick.turn)),
            ))
            self._last_state = state

        out = language.blend(stick, self.authority)
        if self.cfg.style_from_language and language.style is not None:
            # Style is semantic, so it stays with the semantic channel even while
            # the operator is steering: "walk angrily" then steer wherever you like.
            out.style = language.style
        return out

    def reset(self) -> None:
        self.authority = 0.0
        self.idle = 0.0
        self.t = 0.0
        self._last_state = "language"
        self.log.clear()

    def intervention_segments(self) -> list[tuple[float, float]]:
        """(start, end) windows where the operator held authority.

        These are the labels for gap G5: the moments the autonomous channel was
        not good enough, marked by a human at no extra cost.
        """
        out, start = [], None
        for h in self.log:
            if h.direction == "to_operator":
                start = h.t
            elif start is not None:
                out.append((start, h.t))
                start = None
        if start is not None:
            out.append((start, self.t))
        return out
