"""What the operator is asking for, and how it becomes a trajectory.

The command is deliberately *not* a goal.  It is a desired planar velocity, a
turn rate, a gait mix and a style — quantities that are meaningful every frame
and that a human can hold on a stick.  Turning it into the trajectory window
the generator consumes is the job of :class:`TrajectoryPredictor`, and the rate
at which it does so is the responsiveness knob.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from ..motion.features import GAITS, FeatureConfig


# Speed at which each gait is the natural choice, m/s.  Shared with the
# language channel so both channels agree what "run" means.
GAIT_SPEED = {"stand": 0.0, "walk": 1.3, "run": 3.2, "crouch": 0.8, "jump": 1.6}


def _feasible():
    """The speed the robot can actually deliver, measured on the full loop.

    Not a safety limit and not a design choice: above ~1.6 m/s commanded the
    generator's reference keeps speeding up (2.26 m/s at command 2.5) but the
    robot does not -- achieved speed plateaus near 1.0 m/s and the surplus
    turns into stance foot slip, 9.2 cm/s against the corpus's 1.4-3.9.  Asking
    for more produces sliding, not travel, so the operator's range ends here.
    """
    import json
    from pathlib import Path
    d = json.loads((Path(__file__).parent / "feasible_speed.json").read_text())
    return float(d["feasible_command_max"])


FEASIBLE_COMMAND_MAX = _feasible()


_GAIT_TABLE: dict | None = None


def _gait_table():
    """Speed -> gait mixture, measured from the corpus rather than assumed.

    The hand-typed version of this placed each gait at a "characteristic
    speed" (stand 0.0, walk 1.3) and blended by distance, which put the
    stand/walk crossover at 0.65 m/s.  Measured on the training corpus the
    crossover is at ~0.28 m/s: anything from 0.3 m/s upward is 95% *walk* with
    exactly zero stand.  The mismatch meant a 0.6 m/s command was handed a
    55% stand vector the network had never seen paired with a moving
    trajectory, so it did what stand means and balanced in place -- a hard
    locomotion stall for every command below 0.65 m/s.
    """
    global _GAIT_TABLE
    if _GAIT_TABLE is None:
        import json
        from pathlib import Path
        d = json.loads((Path(__file__).parent / "gait_table.json").read_text())
        _GAIT_TABLE = {"speed": np.asarray(d["speed"]),
                       "gait": np.asarray(d["gait"]),
                       "gaits": tuple(d["gaits"])}
    return _GAIT_TABLE


def gait_from_speed(speed: float, crouch: float = 0.0,
                    gaits: tuple[str, ...] = GAITS, sharpness: float = 2.2) -> np.ndarray:
    """Soft gait mix implied by how hard the stick is pushed.

    The generator conditions strongly on gait, so this mapping has to agree
    with how the training data was labelled.  It is interpolated from
    :func:`_gait_table`, which is measured from the corpus.  ``sharpness`` is
    retained for call compatibility and no longer used.
    """
    _ = sharpness
    t = _gait_table()
    assert t["gaits"] == tuple(gaits), (t["gaits"], gaits)
    w = np.stack([np.interp(speed, t["speed"], t["gait"][:, i])
                  for i in range(len(gaits))])
    if "crouch" in gaits and crouch > 0:
        w[gaits.index("crouch")] = max(w[gaits.index("crouch")], crouch * w.max())
    if "jump" in gaits:
        w[gaits.index("jump")] = 0.0        # jump is an event, not a speed
    return w / max(w.sum(), 1e-8)


@dataclass
class Command:
    """Operator intent in the character's own frame."""

    move: np.ndarray = field(default_factory=lambda: np.zeros(2))   # m/s, +x forward
    turn: float = 0.0                                               # rad/s, +ccw
    #: ``None`` means "infer from speed", which is what a stick should do.
    gait: np.ndarray | None = None
    style: np.ndarray | None = None      # optional embedding from the language channel
    #: Target phase manifold from the language channel: ``[amplitude(C), frequency(C)]``
    #: measured from the retrieved clips.  ``None`` means "no opinion".
    phase: np.ndarray | None = None
    responsiveness: float = 0.6          # 0 = fully committed to the current motion,
                                         # 1 = snap the whole future window to intent
    crouch: float = 0.0

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.move))

    def clamped(self) -> "Command":
        """This command with ``move`` limited to what the robot can deliver."""
        sp = self.speed
        if sp <= FEASIBLE_COMMAND_MAX or sp < 1e-9:
            return self
        out = copy.copy(self)
        out.move = self.move * (FEASIBLE_COMMAND_MAX / sp)
        return out

    def effective_gait(self) -> np.ndarray:
        return self.gait if self.gait is not None else gait_from_speed(self.speed, self.crouch)

    def blend(self, other: "Command", w: float) -> "Command":
        """Convex blend toward ``other``.  Used by the arbiter for handover."""
        w = float(np.clip(w, 0.0, 1.0))
        style = other.style if w > 0.5 else self.style
        return Command(
            move=self.move * (1 - w) + other.move * w,
            turn=self.turn * (1 - w) + other.turn * w,
            gait=self.effective_gait() * (1 - w) + other.effective_gait() * w,
            style=style,
            responsiveness=self.responsiveness * (1 - w) + other.responsiveness * w,
            crouch=self.crouch * (1 - w) + other.crouch * w,
        )


class TrajectoryPredictor:
    """The rolling trajectory window the generator is conditioned on.

    Past samples are the character's own history and are not negotiable.  Future
    samples are a blend of what the network predicted last frame and where the
    operator's command would take the character — and the blend weight ramps
    with lookahead, so the near future stays committed to the current motion
    while the far future turns immediately.  That ramp is what "responsiveness"
    means in an animation system, and exposing it is cheaper and more honest
    than trying to tune it out of a reward function.
    """

    def __init__(self, cfg: FeatureConfig, half_window: float = 1.0):
        self.cfg = cfg
        self.dt = 1.0 / cfg.fps
        self.n_half = int(round(half_window * cfg.fps))
        n = 2 * self.n_half + 1
        self.pos = np.zeros((n, 2))          # character-local, metres
        self.dir = np.tile(np.array([1.0, 0.0]), (n, 1))
        self.gait = np.tile(np.eye(len(cfg.gaits))[0], (n, 1))
        self.sample_idx = self.n_half + cfg.offsets  # into the dense buffer

    # -- the two halves -----------------------------------------------------
    def push_past(self, root_delta: np.ndarray) -> None:
        """Advance one frame: shift history back and re-express in the new frame.

        ``root_delta`` is (forward, lateral, dyaw) for the step just taken.  The
        whole buffer is rotated and translated into the new character frame so
        that index ``n_half`` is always the character, at the origin, facing +x.
        """
        dx, dy, dyaw = root_delta
        c, s = np.cos(-dyaw), np.sin(-dyaw)
        R = np.array([[c, -s], [s, c]])
        self.pos = (self.pos - np.array([dx, dy])) @ R.T
        self.dir = self.dir @ R.T
        self.pos[:-1] = self.pos[1:]
        self.dir[:-1] = self.dir[1:]
        self.gait[:-1] = self.gait[1:]
        self.pos[-1] = self.pos[-2] + (self.pos[-2] - self.pos[-3])
        self.dir[-1] = self.dir[-2]
        self.gait[-1] = self.gait[-2]
        # Pin the character to its own origin.  After the shift, index n_half
        # holds last frame's *planned* position, which differs from where the
        # character actually went by the planning error — a few centimetres per
        # frame, accumulating.  Training data always has exactly zero here, so
        # letting it drift feeds the network an input it has never seen.
        self.pos[self.n_half] = 0.0
        self.dir[self.n_half] = np.array([1.0, 0.0])

    def apply_command(self, cmd: Command) -> None:
        """Bend the future half toward the operator's intent."""
        n = self.n_half
        # bias > 1 front-loads the ramp, so the near future commits to intent
        # sooner: high responsiveness == snappy, low == the character finishes
        # what it started.  This is the knob an animator would tune per action.
        bias = 1.0 + 6.0 * float(np.clip(cmd.responsiveness, 0.0, 1.0))

        gait = cmd.effective_gait()
        target_dir = self.dir[n].copy()
        if cmd.speed > 1e-3:
            target_dir = cmd.move / cmd.speed

        # All n future frames at once (this runs ~13x per control tick on the
        # robot's CPU, where the per-frame Python loop cost 2.3 ms).  Same
        # arithmetic as the loop it replaces, kept in _apply_command_loop and
        # checked against it in tests/test_control.py.
        k = np.arange(1, n + 1)
        w = (1.0 - (1.0 - k / n) ** bias)[:, None]   # 0 at the character, 1 at the horizon
        yaw = cmd.turn * (k * self.dt)
        c, s = np.cos(yaw), np.sin(yaw)
        steer = np.stack([c * target_dir[0] - s * target_dir[1],
                          s * target_dir[0] + c * target_dir[1]], axis=1)
        # sequential running sum, so rounding matches pos = pos + move*dt
        pos = np.cumsum(np.vstack([self.pos[n], np.tile(cmd.move * self.dt, (n, 1))]), axis=0)[1:]
        sl = slice(n + 1, 2 * n + 1)
        self.pos[sl] = self.pos[sl] * (1 - w) + pos * w
        d = self.dir[sl] * (1 - w) + steer * w
        nrm = np.sqrt((d * d).sum(axis=1))
        ok = nrm > 1e-8
        d[ok] /= nrm[ok, None]
        self.dir[sl] = d
        self.gait[sl] = self.gait[sl] * (1 - w) + gait * w

    def _apply_command_loop(self, cmd: Command) -> None:
        """Reference per-frame form of :meth:`apply_command`, for tests."""
        n = self.n_half
        bias = 1.0 + 6.0 * float(np.clip(cmd.responsiveness, 0.0, 1.0))
        gait = cmd.effective_gait()
        target_dir = self.dir[n].copy()
        if cmd.speed > 1e-3:
            target_dir = cmd.move / cmd.speed
        pos = self.pos[n].copy()
        for k in range(1, n + 1):
            w = 1.0 - (1.0 - k / n) ** bias
            yaw = cmd.turn * (k * self.dt)
            c, s = np.cos(yaw), np.sin(yaw)
            steer = np.array([[c, -s], [s, c]]) @ target_dir
            pos = pos + cmd.move * self.dt
            i = n + k
            self.pos[i] = self.pos[i] * (1 - w) + pos * w
            self.dir[i] = self.dir[i] * (1 - w) + steer * w
            nrm = np.linalg.norm(self.dir[i])
            if nrm > 1e-8:
                self.dir[i] /= nrm
            self.gait[i] = self.gait[i] * (1 - w) + gait * w

    def absorb_prediction(self, future_pos: np.ndarray, future_dir: np.ndarray,
                          weight: float = 0.5) -> None:
        """Fold the generator's own predicted future back into the buffer.

        Without this the future window is pure operator extrapolation and the
        character walks like a cursor.  With it, the network's knowledge of what
        a body can actually do bends the plan — the same role the physics tracker
        plays one layer down.
        """
        idx = self.sample_idx[self.cfg.n_past + 1:]
        self.pos[idx] = self.pos[idx] * (1 - weight) + future_pos * weight
        self.dir[idx] = self.dir[idx] * (1 - weight) + future_dir * weight

    # -- what the network sees ---------------------------------------------
    def sample(self):
        return (self.pos[self.sample_idx].copy(),
                self.dir[self.sample_idx].copy(),
                self.gait[self.sample_idx].copy())

    def reset(self, gait: np.ndarray | None = None) -> None:
        self.pos[:] = 0.0
        self.dir[:] = np.array([1.0, 0.0])
        self.gait[:] = gait if gait is not None else np.eye(len(self.cfg.gaits))[0]
