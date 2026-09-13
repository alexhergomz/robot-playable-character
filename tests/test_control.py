"""The interaction contract, tested as a contract."""
from __future__ import annotations

import numpy as np

from rpc.control.arbiter import ArbiterConfig, ChannelArbiter
from rpc.control.command import Command, TrajectoryPredictor
from rpc.control.input import InputConfig, ScriptedInput, _shape
from rpc.control.language import LanguageChannel, MotionEntry
from rpc.motion.features import GAITS, FeatureConfig

CFG = FeatureConfig(fps=30.0)


class TestTrajectory:
    def test_horizon_reaches_commanded_displacement(self):
        t = TrajectoryPredictor(CFG)
        t.reset()
        t.apply_command(Command(move=np.array([2.0, 0.0]), responsiveness=1.0))
        pos, _, _ = t.sample()
        assert abs(pos[-1, 0] - 2.0) < 0.05          # 2 m/s over a 1 s horizon

    def test_responsiveness_is_monotone(self):
        near = []
        for r in (0.0, 0.25, 0.5, 0.75, 1.0):
            t = TrajectoryPredictor(CFG)
            t.reset()
            t.apply_command(Command(move=np.array([2.0, 0.0]), responsiveness=r))
            pos, _, _ = t.sample()
            near.append(pos[CFG.n_past + 1, 0])
        assert all(b > a for a, b in zip(near, near[1:])), near

    def test_far_horizon_is_pure_intent(self):
        """Whatever the responsiveness, the end of the window is where the
        operator asked to be — responsiveness only shapes the approach."""
        ends = []
        for r in (0.0, 1.0):
            t = TrajectoryPredictor(CFG)
            t.reset()
            t.apply_command(Command(move=np.array([1.5, 0.0]), responsiveness=r))
            ends.append(t.sample()[0][-1, 0])
        assert abs(ends[0] - ends[1]) < 1e-6

    def test_push_past_keeps_character_at_origin(self):
        t = TrajectoryPredictor(CFG)
        t.reset()
        for _ in range(60):
            t.apply_command(Command(move=np.array([1.0, 0.4]), turn=0.3))
            t.push_past(np.array([1.0 / 30, 0.4 / 30, 0.01]))
        pos, direction, _ = t.sample()
        assert np.abs(pos[CFG.n_past]).max() < 1e-9
        assert np.allclose(direction[CFG.n_past], [1, 0], atol=1e-6)

    def test_turn_rotates_the_horizon(self):
        t = TrajectoryPredictor(CFG)
        t.reset()
        t.apply_command(Command(move=np.array([1.0, 0.0]), turn=1.0, responsiveness=1.0))
        _, d, _ = t.sample()
        assert d[-1, 1] > 0.5      # +turn is counter-clockwise, so the horizon faces +y


class TestArbiter:
    def make(self):
        return ChannelArbiter(ArbiterConfig(attack=0.05, release=0.8, hold=0.4), dt=1 / 30)

    def test_stick_wins_immediately(self):
        a = self.make()
        lang = Command(move=np.array([1.0, 0.0]))
        stick = Command(move=np.array([0.0, 2.0]))
        for _ in range(3):
            out = a.update(lang, stick)
        assert a.authority == 1.0
        assert out.move[1] > 1.9 and out.move[0] < 0.1

    def test_language_resumes_only_after_release(self):
        a = self.make()
        lang = Command(move=np.array([1.0, 0.0]))
        for _ in range(30):
            a.update(lang, Command(move=np.array([0.0, 2.0])))
        neutral = Command(move=np.zeros(2))
        for _ in range(12):                     # 0.4 s hold: authority must not move
            a.update(lang, neutral)
        assert a.authority > 0.99
        for _ in range(40):
            a.update(lang, neutral)
        assert a.authority < 0.5

    def test_deadzone_does_not_trigger_handover(self):
        a = self.make()
        lang = Command(move=np.array([1.0, 0.0]))
        for _ in range(60):
            a.update(lang, Command(move=np.array([0.05, 0.05])))
        assert a.authority == 0.0
        assert a.log == []

    def test_style_stays_with_language_under_stick(self):
        a = self.make()
        lang = Command(move=np.array([1.0, 0.0]), style=np.array([0.0, 1.0]))
        for _ in range(10):
            out = a.update(lang, Command(move=np.array([2.0, 0.0])))
        assert np.allclose(out.style, [0.0, 1.0])

    def test_interventions_are_logged_as_segments(self):
        a = self.make()
        lang = Command(move=np.array([1.0, 0.0]))
        for _ in range(30):
            a.update(lang, Command(move=np.zeros(2)))
        for _ in range(60):
            a.update(lang, Command(move=np.array([2.0, 0.0])))
        for _ in range(60):
            a.update(lang, Command(move=np.zeros(2)))
        segs = a.intervention_segments()
        assert len(segs) == 1
        start, end = segs[0]
        assert 0.9 < start < 1.2 and end > start


class TestInput:
    def test_deadzone_and_expo(self):
        cfg = InputConfig()
        assert _shape(0.05, cfg) == 0.0
        assert _shape(1.0, cfg) == 1.0
        assert _shape(0.5, cfg) < 0.5              # expo gives fine control near centre
        assert _shape(-0.8, cfg) == -_shape(0.8, cfg)

    def test_scripted_input_returns_to_neutral(self):
        s = ScriptedInput([(1.0, Command(move=np.array([1.0, 0.0])))], dt=1 / 30)
        assert s.poll().move[0] == 1.0
        for _ in range(45):
            c = s.poll()
        assert np.allclose(c.move, 0)


class TestLanguage:
    def library(self):
        # Entries carry what the clips measurably are, not an authored gait
        # one-hot: speed, turn rate and the phase manifold are all measurements.
        e = lambda n, l, sp, f: MotionEntry(n, l, speed=sp, turn=0.0,
                                            amplitude=np.full(5, sp),
                                            frequency=np.full(5, f))
        return [e("walk1", ("walk",), 0.96, 1.18),
                e("run1", ("run",), 2.33, 1.40),
                e("idle1", ("stand",), 0.05, 0.10)]

    def test_running_is_faster_than_walking(self):
        ch = LanguageChannel().index(self.library())
        walk, dw = ch.interpret("walk slowly forward")
        run, dr = ch.interpret("run fast")
        assert run.speed > walk.speed
        # and the stride frequency ordering must follow, since both come from
        # the same measurement rather than from separate constants
        assert dr["frequency_hz"] > dw["frequency_hz"]

    def test_output_is_a_valid_command(self):
        ch = LanguageChannel().index(self.library())
        cmd, diag = ch.interpret("walk")
        # gait is no longer authored: it is inferred from the measured speed
        assert cmd.gait is None
        assert cmd.effective_gait().shape == (len(GAITS),)
        assert abs(cmd.effective_gait().sum() - 1.0) < 1e-6
        assert cmd.phase is not None and cmd.phase.shape == (10,)
        assert cmd.style is not None and abs(cmd.style.sum() - 1.0) < 1e-6
        assert diag["backend"] in ("clip", "hash")

    def test_language_is_never_snappier_than_a_stick(self):
        ch = LanguageChannel().index(self.library())
        cmd, _ = ch.interpret("walk")
        assert cmd.responsiveness < 0.6
