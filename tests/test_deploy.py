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
