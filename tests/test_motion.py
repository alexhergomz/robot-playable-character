"""Invariants that must hold for the motion pipeline to mean anything."""
from __future__ import annotations

import numpy as np
import pytest

from rpc.motion import quat as Q
from rpc.motion.clip import MotionClip, Skeleton
from rpc.motion.dataset import FeatureSpec, Normalizer, analyse, build_samples
from rpc.motion.features import FeatureConfig, root_deltas, sample_trajectory
from rpc.motion.phase import contact_phase, foot_contacts, phase_delta

rng = np.random.default_rng(0)


def rand_q(*shape):
    return Q.normalize(rng.normal(size=(*shape, 4)))


class TestQuaternions:
    def test_matrix_round_trip(self):
        q = rand_q(64)
        assert np.allclose(np.abs((Q.from_matrix(Q.to_matrix(q)) * q).sum(-1)), 1, atol=1e-6)

    def test_rotate_matches_matrix(self):
        q, v = rand_q(64), rng.normal(size=(64, 3))
        assert np.allclose(Q.rotate(q, v), np.einsum("bij,bj->bi", Q.to_matrix(q), v), atol=1e-9)

    def test_euler_matches_scipy(self):
        sp = pytest.importorskip("scipy.spatial.transform")
        e = rng.uniform(-3, 3, size=(64, 3))
        ref = sp.Rotation.from_euler("ZYX", e).as_quat()[:, [3, 0, 1, 2]]
        assert np.allclose(np.abs((Q.from_euler(e, "zyx") * ref).sum(-1)), 1, atol=1e-8)

    def test_6d_round_trip(self):
        q = rand_q(64)
        assert np.allclose(np.abs((Q.unflatten_2axis(Q.flatten_2axis(q)) * q).sum(-1)),
                           1, atol=1e-6)

    def test_slerp_does_not_broadcast_silently(self):
        """A scalar t used to be expanded here, turning (T,J,4) into (T,T,J,4)."""
        a, b = rand_q(7, 5), rand_q(7, 5)
        out = Q.slerp(a, b, np.linspace(0, 1, 7)[:, None, None])
        assert out.shape == (7, 5, 4)


def toy_skeleton() -> Skeleton:
    names = ("Hips", "LeftUpLeg", "LeftFoot", "RightUpLeg", "RightFoot", "Spine", "Head")
    parents = np.array([-1, 0, 1, 0, 3, 0, 5])
    offsets = np.array([[0, 0, 0], [0, 0.1, -0.05], [0, 0, -0.45],
                        [0, -0.1, -0.05], [0, 0, -0.45], [0, 0, 0.2], [0, 0, 0.35]], float)
    return Skeleton(names, parents, offsets).with_inferred_axes()


def toy_clip(T: int = 200, fps: float = 30.0) -> MotionClip:
    skel = toy_skeleton()
    t = np.arange(T) / fps
    root = np.stack([1.0 * t, np.zeros(T), 0.9 + 0.01 * np.sin(2 * np.pi * t)], -1)
    rot = np.tile(np.array([1.0, 0, 0, 0]), (T, skel.n_joints, 1))
    swing = 0.5 * np.sin(2 * np.pi * 1.0 * t)
    rot[:, 1] = Q.from_axis_angle(np.tile([0, 1.0, 0], (T, 1)), swing)
    rot[:, 3] = Q.from_axis_angle(np.tile([0, 1.0, 0], (T, 1)), -swing)
    return MotionClip(skel, root, rot, fps, "toy")


class TestSkeletonAxes:
    def test_infers_forward_and_up(self):
        s = toy_skeleton()
        assert np.allclose(s.up, [0, 0, 1], atol=1e-6)
        assert abs(abs(float(np.dot(s.fwd, [1, 0, 0]))) - 1) < 1e-6

    def test_axes_are_orthonormal(self):
        s = toy_skeleton()
        assert abs(float(np.dot(s.fwd, s.up))) < 1e-8
        assert abs(np.linalg.norm(s.fwd) - 1) < 1e-8


class TestFeatures:
    def test_trajectory_frame_is_character_local(self):
        c = toy_clip()
        gp, gr = c.forward_kinematics()
        cfg = FeatureConfig(fps=c.fps)
        tp, td = sample_trajectory(cfg, gp, gr, c.skeleton.fwd)
        assert np.abs(tp[:, cfg.n_past]).max() < 1e-9        # current sample at origin
        assert np.allclose(td[:, cfg.n_past], [1, 0], atol=1e-6)   # facing +x

    def test_future_leads_past_when_walking_forward(self):
        c = toy_clip()
        gp, gr = c.forward_kinematics()
        cfg = FeatureConfig(fps=c.fps)
        tp, _ = sample_trajectory(cfg, gp, gr, c.skeleton.fwd)
        assert tp[:, -1, 0].mean() > 0.3
        assert tp[:, 0, 0].mean() < -0.3

    def test_root_delta_recovers_speed(self):
        c = toy_clip()
        gp, gr = c.forward_kinematics()
        rd = root_deltas(gp, gr, c.skeleton.fwd)
        assert abs(rd[:-1, 0].mean() * c.fps - 1.0) < 0.05     # 1 m/s by construction
        assert abs(rd[:-1, 1].mean()) < 1e-6                   # no lateral drift

    def test_mirror_preserves_height_and_flips_lateral(self):
        c = toy_clip()
        m = c.mirror()
        assert np.allclose(c.root_pos[:, 2], m.root_pos[:, 2])
        assert np.allclose(c.root_pos[:, 1], -m.root_pos[:, 1])


class TestPhase:
    def test_rolling_floor_survives_a_dropped_actor(self):
        """A segment where the character leaves the standing plane must not
        suppress contact detection for the whole clip."""
        T = 900
        p = np.zeros((T, 2, 3))
        p[:, 0, 2] = 0.02 + 0.08 * np.clip(np.sin(np.arange(T) * 0.2), 0, None)
        p[:, 1, 2] = 0.02 + 0.08 * np.clip(-np.sin(np.arange(T) * 0.2), 0, None)
        p[400:600, :, 2] -= 0.45                       # actor goes to the ground
        ct = foot_contacts(p, [[0], [1]], fps=30.0)
        early = ct[:300].mean()
        assert early > 0.3, f"contacts suppressed by the low segment ({early:.2f})"

    def test_phase_is_bounded_and_advances(self):
        c = toy_clip(600)
        gp, _ = c.forward_kinematics()
        feet = [c.skeleton.index("LeftFoot"), c.skeleton.index("RightFoot")]
        ct = foot_contacts(gp, [[feet[0]], [feet[1]]], c.fps)
        ph = contact_phase(ct, c.fps)
        assert ph.min() >= 0 and ph.max() < 2 * np.pi + 1e-6
        assert phase_delta(ph).mean() > 0


class TestDataset:
    def test_layout_matches_declaration(self):
        c = toy_clip(300)
        cfg = FeatureConfig(fps=c.fps)
        spec = FeatureSpec(c.skeleton.n_joints, cfg)
        X, Y, P = build_samples(analyse(c, cfg), spec)
        assert X.shape[1] == spec.n_in and Y.shape[1] == spec.n_out
        assert len(X) == len(Y) == len(P) == len(c) - 1
        assert np.isfinite(X).all() and np.isfinite(Y).all()

    def test_slices_tile_the_vector(self):
        spec = FeatureSpec(22, FeatureConfig())
        for slices, n in ((spec.input_slices, spec.n_in), (spec.output_slices, spec.n_out)):
            covered = np.zeros(n, int)
            for sl in slices.values():
                covered[sl] += 1
            assert (covered == 1).all()

    def test_normalizer_survives_constant_columns(self):
        x = np.concatenate([rng.normal(size=(100, 4)), np.ones((100, 2))], -1)
        n = Normalizer.fit(x)
        assert np.isfinite(n(x)).all()
