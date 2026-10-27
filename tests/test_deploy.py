"""The ZMQ stream to the G1 deploy app: wire format and streaming rules."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from rpc.control.command import Command
from rpc.deploy import wire
from rpc.deploy.streamer import PoseStreamer, StreamLimits
from rpc.sim import sonic as P

UPSTREAM = Path(__file__).resolve().parents[1] / "assets/sonic/upstream/zmq_planner_sender.py"


def _upstream():
    if not UPSTREAM.exists():
        pytest.skip("upstream sender not downloaded")
    spec = importlib.util.spec_from_file_location("ups", UPSTREAM)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class TestWire:
    def test_header_is_1280_not_the_documented_1024(self):
        assert wire.HEADER_SIZE == 1280

    def test_round_trip(self):
        d = {"a": np.arange(6, dtype=np.float32).reshape(2, 3),
             "b": np.array([3, 4], np.int64), "c": np.array([1], np.uint8)}
        hdr, out = wire.unpack(wire.pack("pose", d, version=1), "pose")
        assert hdr["v"] == 1
        for k in d:
            np.testing.assert_array_equal(out[k], d[k])

    def test_pose_bytes_match_upstream_sender(self):
        ups = _upstream()
        d = {"joint_pos": np.random.rand(4, 29).astype(np.float32),
             "frame_index": np.arange(4, dtype=np.int64)}
        assert wire.pack("pose", d, version=1) == ups.pack_pose_message(d, version=1)

    def test_command_bytes_match_upstream_sender(self):
        ups = _upstream()
        for st, sp in ((True, False), (False, True)):
            assert wire.command_message(st, sp, False) == \
                ups.build_command_message(st, sp, False)

    def test_remap_to_isaaclab_happens_once(self):
        jp = np.tile(np.arange(29, dtype=np.float32), (3, 1))     # value = mujoco index
        msg = wire.pose_message(jp, jp, np.tile([1, 0, 0, 0], (3, 1)),
                                np.arange(3), P.MUJOCO_TO_ISAACLAB)
        _, out = wire.unpack(msg, "pose")
        np.testing.assert_array_equal(out["joint_pos"][0], P.MUJOCO_TO_ISAACLAB)
        assert out["body_quat"].shape == (3, 1, 4)


class _FakeFrame:
    def __init__(self, q, yaw):
        self.joint_pos, self.root_yaw = q, yaw


class _FakeGen:
    """Joint 0 integrates commanded speed; enough to test the streaming rules."""
    def __init__(self, nan=False):
        self.joint_pos = np.zeros(29); self.yaw = 0.0; self.nan = nan

    def step(self, cmd):
        self.joint_pos = self.joint_pos.copy()
        self.joint_pos[0] += 0.02 * float(np.linalg.norm(cmd.move))
        if self.nan:
            self.joint_pos[3] = np.nan
        return _FakeFrame(self.joint_pos.copy(), self.yaw)


class _Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


def _streamer(gen=None, **kw):
    sent, clock = [], _Clock()
    s = PoseStreamer(gen or _FakeGen(), sent.append,
                     joint_lower=-np.full(29, 3.0), joint_upper=np.full(29, 3.0),
                     mujoco_to_isaaclab=P.MUJOCO_TO_ISAACLAB, clock=clock,
                     stand_pose=np.full(29, 0.5), **kw)
    return s, sent, clock


class TestStreamer:
    walk = Command(move=np.array([2.0, 0.0]))

    def test_chunk_ends_strictly_increase_and_frames_are_contiguous(self):
        s, sent, clock = _streamer()
        ends = []
        for i in range(40):
            clock.t = i * 0.02
            s.tick(self.walk)
        for m in sent:
            _, out = wire.unpack(m, "pose")
            fi = out["frame_index"]
            assert (np.diff(fi) == 1).all()
            ends.append(fi[-1])
        assert all(b > a for a, b in zip(ends, ends[1:]))

    def test_replanning_every_other_frame_never_starves_the_lookahead(self):
        # SONIC reads 10 frames ahead; between re-plans the receiver must still
        # hold 10 generated (non-tail) frames past the current one.
        lookahead, tail = 10, 50
        s, sent, clock = _streamer(replan_every=2)
        for i in range(40):
            clock.t = i * 0.02
            s.tick(self.walk)
        firsts = [int(wire.unpack(m, "pose")[1]["frame_index"][0]) for m in sent]
        assert np.diff(firsts).tolist() == [2] * (len(firsts) - 1)
        for a, b in zip(firsts, firsts[1:]):
            generated_end = a + s.horizon - 1            # last non-tail frame of message a
            for j in range(a - 1, b - 1):                # frames played before message b
                assert generated_end - j >= lookahead
        assert len(sent) == 20

    def test_replan_every_one_is_the_default_behaviour(self):
        a, sa, ca = _streamer(); b, sb, cb = _streamer(replan_every=1)
        for i in range(20):
            ca.t = cb.t = i * 0.02
            a.tick(self.walk); b.tick(self.walk)
        assert sa == sb

    def test_a_second_send_in_the_same_tick_is_skipped(self):
        s, sent, clock = _streamer()
        clock.t = 0.0
        assert s.tick(self.walk) and not s.tick(self.walk)
        assert len(sent) == 1 and s.stats.skipped_same_tick == 1

    def test_speed_is_capped(self):
        s, sent, clock = _streamer(limits=StreamLimits(max_speed=0.3))
        assert np.linalg.norm(s._safe(self.walk).move) == pytest.approx(0.3)

    def test_streaming_never_starts_the_controller(self):
        s, sent, clock = _streamer()
        for i in range(10):
            clock.t = i * 0.02
            s.tick(self.walk)
        assert all(m.startswith(b"pose") for m in sent)

    def test_non_finite_reference_is_never_sent(self):
        s, sent, clock = _streamer(gen=_FakeGen(nan=True))
        with pytest.raises(FloatingPointError):
            s.tick(self.walk)
        assert sent == []

    def test_joint_limits_are_enforced(self):
        s, sent, clock = _streamer()
        s.lo[:] = -0.01; s.hi[:] = 0.01
        for i in range(10):
            clock.t = i * 0.02
            s.tick(self.walk)
        _, out = wire.unpack(sent[-1], "pose")
        assert np.abs(out["joint_pos"]).max() <= 0.01 + 1e-6
        assert s.stats.clipped_frames > 0

    def test_every_message_ends_in_the_stand_pose(self):
        s, sent, clock = _streamer()
        for i in range(5):
            clock.t = i * 0.02
            s.tick(self.walk)
        for m in sent:
            _, out = wire.unpack(m, "pose")
            n = s.horizon + int(round(s.limits.tail_s * s.hz))
            assert out["joint_pos"].shape[0] == n
            last_mj = np.empty(29); last_mj[P.MUJOCO_TO_ISAACLAB] = out["joint_pos"][-1]
            np.testing.assert_allclose(last_mj, 0.5, atol=1e-6)

    def test_tail_starts_after_the_lookahead_the_tracker_reads(self):
        s, sent, clock = _streamer()
        clock.t = 0.0; s.tick(self.walk)
        _, out = wire.unpack(sent[0], "pose")
        head = out["joint_pos"][:s.horizon]
        assert s.horizon > 10                    # SONIC reads 10 frames ahead
        assert not np.allclose(head[-1], out["joint_pos"][-1])


def test_sonic_model_stands_still():
    """Regression for the creep: on SONIC's own model a static stand holds."""
    import mujoco
    from rpc.sim import sonic_scene as S
    from rpc.sim.sonic_policy import SonicPolicy, CONTROL_HZ
    if not (S.MESHES.exists()):
        pytest.skip("Menagerie meshes not present")
    m = S.load(); d = mujoco.MjData(m); pd = S.PDStep(m)
    sub = int(round(1 / (CONTROL_HZ * m.opt.timestep)))
    d.qpos[3] = 1; d.qpos[7:] = P.DEFAULT_ANGLES; d.qpos[2] = 0.79
    mujoco.mj_forward(m, d); pd(d, P.DEFAULT_ANGLES, 50)
    pol = SonicPolicy(root=str(Path(__file__).resolve().parents[1] / "assets/sonic"),
                      variant="low_latency")

    def grav():
        o = np.zeros(3)
        mujoco.mju_rotVecQuat(o, np.array([0., 0., -1.]),
                              np.array([d.qpos[3], -d.qpos[4], -d.qpos[5], -d.qpos[6]]))
        return o
    pol.reset(d.qpos[7:].copy(), d.qvel[6:].copy(), d.qvel[3:6].copy(), grav())
    R = np.tile(P.DEFAULT_ANGLES[P.MUJOCO_TO_ISAACLAB], (10, 1))
    start = d.qpos[:2].copy()
    for _ in range(int(6 * CONTROL_HZ)):
        tg = pol.step_policy(R, np.zeros((10, 29)), np.tile([1., 0, 0, 0], (10, 1)),
                             d.qpos[3:7].copy(), d.qpos[7:].copy(), d.qvel[6:].copy(),
                             d.qvel[3:6].copy(), grav())
        pd(d, tg, sub)
    assert d.qpos[2] > 0.6
    assert np.linalg.norm(d.qpos[:2] - start) < 0.15


BUNDLE = Path(__file__).resolve().parents[1] / "port/results/onboard_bundle"


@pytest.mark.skipif(not (BUNDLE / "weights.npz").exists(), reason="onboard bundle not exported")
class TestIdlePhaseHold:
    """Channels keep the seed frame's relative phases while idle, and only then."""

    stand = Command(move=np.zeros(2))
    walk = Command(move=np.array([0.3, 0.0]))

    @staticmethod
    def _rel(g):
        C = g.C
        ph = np.arctan2(g.manifold[C:2 * C], g.manifold[:C])
        return np.angle(np.exp(1j * (ph - ph[0])))

    def test_off_by_default_and_channels_drift_while_standing(self):
        from rpc.deploy import onboard
        g = onboard.load(BUNDLE, backend="numpy")
        rel0 = self._rel(g)
        for _ in range(100):
            g.step(self.stand)
        assert not g.hold_phase_when_idle
        assert np.abs(np.angle(np.exp(1j * (self._rel(g) - rel0)))).max() > 0.5

    def test_hold_keeps_the_walking_configuration_while_standing(self):
        from rpc.deploy import onboard
        g = onboard.load(BUNDLE, backend="numpy", hold_phase_when_idle=True)
        rel0 = self._rel(g)
        for _ in range(100):
            g.step(self.stand)
        assert np.allclose(np.angle(np.exp(1j * (self._rel(g) - rel0))), 0.0, atol=1e-9)

    def test_hold_does_not_touch_walking_steps(self):
        from rpc.deploy import onboard
        a = onboard.load(BUNDLE, backend="numpy")
        b = onboard.load(BUNDLE, backend="numpy", hold_phase_when_idle=True)
        for _ in range(60):
            fa, fb = a.step(self.walk), b.step(self.walk)
        assert np.array_equal(a.manifold, b.manifold) and np.array_equal(fa.joint_pos, fb.joint_pos)


def _raw(lx=0.0, ly=0.0, rx=0.0, ry=0.0, buttons=()):
    import struct
    from rpc.deploy.remote import BUTTONS
    b = bytearray(40)
    for n in buttons:
        byte, bit = BUTTONS[n]
        b[byte] |= 1 << bit
    b[4:8], b[8:12], b[12:16], b[20:24] = (struct.pack("<f", v) for v in (lx, rx, ry, ly))
    return list(b)


class TestRemote:
    def test_round_trip_matches_unitree_layout(self):
        from rpc.deploy.remote import parse
        s = parse(_raw(lx=0.25, ly=-0.5, rx=0.75, ry=0.1, buttons=("R1", "A", "left")))
        assert (s.lx, s.ly, s.rx) == pytest.approx((0.25, -0.5, 0.75))
        assert s.buttons == {"R1", "A", "left"}

    def test_no_command_without_deadman(self):
        from rpc.deploy.remote import StickMap, parse, to_command
        assert to_command(parse(_raw(ly=1.0)), StickMap()) is None
        assert to_command(parse(_raw(ly=1.0, buttons=("L1",))), StickMap()) is None

    def test_stick_directions_and_scale(self):
        from rpc.deploy.remote import StickMap, parse, to_command
        m = StickMap(max_speed=0.3, max_turn=0.5)
        fwd = to_command(parse(_raw(ly=1.0, buttons=("R1",))), m)
        left = to_command(parse(_raw(lx=-1.0, buttons=("R1",))), m)
        ccw = to_command(parse(_raw(rx=-1.0, buttons=("R1",))), m)
        assert fwd.move == pytest.approx([0.3, 0.0]) and fwd.turn == 0.0
        assert left.move == pytest.approx([0.0, 0.3])
        assert ccw.turn == pytest.approx(0.5)

    def test_deadband_gives_exact_zero(self):
        from rpc.deploy.remote import StickMap, parse, to_command
        c = to_command(parse(_raw(lx=0.05, ly=-0.08, rx=0.09, buttons=("R1",))), StickMap())
        assert np.all(c.move == 0.0) and c.turn == 0.0


class TestLauncherSession:
    held = staticmethod(lambda: __import__("rpc.deploy.remote", fromlist=["x"]).parse(_raw(ly=1.0, buttons=("R1",))))

    def _session(self):
        from rpc.deploy.launcher import Session
        from rpc.deploy.remote import StickMap
        gen = _FakeGen(); calls = []
        orig = gen.step
        gen.step = lambda c: (calls.append(1), orig(c))[1]
        s, sent, clock = _streamer(gen=gen, limits=StreamLimits(max_speed=0.3, brake_s=0.5))
        return Session(s, StickMap(), stale_s=0.25, clock=clock), sent, clock, calls

    def _run(self, sess, clock, t0, t1, remote):
        for i in range(int(round((t1 - t0) * 50))):
            clock.t = t0 + i * 0.02
            sess.step(*remote(clock.t))

    def test_streams_only_while_held_then_brakes_and_goes_quiet(self):
        sess, sent, clock, _ = self._session()
        self._run(sess, clock, 0.0, 1.0, lambda t: (self.held(), t))
        n_held = len(sent)
        assert sess.state == "streaming" and n_held > 0
        self._run(sess, clock, 1.0, 1.5, lambda t: (None, None))
        assert len(sent) > n_held                       # brake messages
        self._run(sess, clock, 1.5, 3.0, lambda t: (None, None))
        n_after = len(sent)
        self._run(sess, clock, 3.0, 4.0, lambda t: (None, None))
        assert sess.state == "idle" and len(sent) == n_after   # nothing more is sent

    def test_stale_remote_counts_as_released(self):
        sess, sent, clock, _ = self._session()
        self._run(sess, clock, 0.0, 0.5, lambda t: (self.held(), t))
        self._run(sess, clock, 0.5, 2.0, lambda t: (self.held(), 0.4))   # last sample stuck at t=0.4
        assert sess.state == "idle"

    def test_resume_after_pause_has_no_catch_up_burst_and_frames_increase(self):
        sess, sent, clock, calls = self._session()
        self._run(sess, clock, 0.0, 1.0, lambda t: (self.held(), t))
        self._run(sess, clock, 1.0, 6.0, lambda t: (None, None))
        before = len(calls)
        clock.t = 6.0
        sess.step(self.held(), 6.0)
        # one real step to reach the clock frame + the horizon rollout, not 250 catch-up steps
        assert len(calls) - before <= 1 + sess.st.horizon
        ends = [int(wire.unpack(m, "pose")[1]["frame_index"][-1]) for m in sent]
        assert all(b > a for a, b in zip(ends, ends[1:]))
