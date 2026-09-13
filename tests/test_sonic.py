"""GEAR-SONIC adapter: layout, conventions, and that it actually stands.

These are the checks that catch a mis-transcribed observation vector.  A wrong
layout does not raise — it produces a policy that falls over, which is much
harder to attribute after the fact.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("onnxruntime")

from rpc.sim import sonic as P
from rpc.sim.robot import G1
from rpc.sim.sonic_policy import (DECODER_LAYOUT, ENCODER_LAYOUT_DEFAULT,
                                  ENCODER_LAYOUT_LOW_LATENCY, SonicPolicy,
                                  apply_sonic_gains, _quat_to_mat, _six_d)

ASSETS = Path("assets/sonic")
HAVE_WEIGHTS = (ASSETS / "low_latency" / "model_encoder.onnx").exists()
SAMPLE = ASSETS / "sample_data/robot_filtered/210531/walk_forward_amateur_001__A001.pkl"


class TestParameters:
    def test_joint_order_matches_menagerie(self):
        m = mujoco.MjModel.from_xml_path(G1.path())
        v = P.verify_against_model(m)
        assert v["joint_order_matches"] and v["default_pose_within_limits"]

    def test_permutations_are_inverses(self):
        assert np.array_equal(P.ISAACLAB_TO_MUJOCO[P.MUJOCO_TO_ISAACLAB],
                              np.arange(P.N_MOTOR))
        assert np.array_equal(P.MUJOCO_TO_ISAACLAB[P.ISAACLAB_TO_MUJOCO],
                              np.arange(P.N_MOTOR))

    def test_default_pose_is_a_crouch_not_the_keyframe(self):
        """Guards the most expensive silent bug: using Menagerie's straight-legged
        stand keyframe as SONIC's action origin."""
        m = mujoco.MjModel.from_xml_path(G1.path())
        assert np.abs(P.DEFAULT_ANGLES - m.key_qpos[0][7:]).max() > 0.5
        assert P.DEFAULT_ANGLES[3] > 0.6 and P.DEFAULT_ANGLES[0] < -0.3   # knee / hip pitch

    def test_action_scale_is_per_joint(self):
        assert P.ACTION_SCALE.shape == (29,)
        assert P.ACTION_SCALE.max() / P.ACTION_SCALE.min() > 5

    def test_sonic_gains_are_softer_than_the_model_default(self):
        m = mujoco.MjModel.from_xml_path(G1.path())
        stock = m.actuator_gainprm[0, 0]
        apply_sonic_gains(m)
        assert m.actuator_gainprm[0, 0] < stock / 4
        assert np.allclose(m.actuator_biasprm[:, 1], -P.KP)


class TestLayouts:
    def test_dimensions_match_the_onnx_graphs(self):
        assert DECODER_LAYOUT.size == 994
        assert ENCODER_LAYOUT_LOW_LATENCY.size == 1247
        assert ENCODER_LAYOUT_DEFAULT.size == 1762

    def test_blocks_tile_without_gaps(self):
        for layout in (DECODER_LAYOUT, ENCODER_LAYOUT_LOW_LATENCY, ENCODER_LAYOUT_DEFAULT):
            covered = np.zeros(layout.size, int)
            for sl in layout.offsets.values():
                covered[sl] += 1
            assert (covered == 1).all()

    def test_six_d_is_interleaved_not_concatenated(self):
        """SONIC reads the first two columns row-major; the project's own
        flatten_2axis concatenates them column-major.  Mixing them up scrambles
        every reference orientation."""
        R = _quat_to_mat(np.array([0.9239, 0.0, 0.0, 0.3827]))
        s = _six_d(R)
        assert np.allclose(s, [R[0, 0], R[0, 1], R[1, 0], R[1, 1], R[2, 0], R[2, 1]])


@pytest.mark.skipif(not HAVE_WEIGHTS, reason="SONIC weights not downloaded")
class TestPolicy:
    def test_onnx_dims_agree_with_layout(self):
        p = SonicPolicy(variant="low_latency")
        assert p.enc.get_inputs()[0].shape[1] == ENCODER_LAYOUT_LOW_LATENCY.size
        assert p.dec.get_inputs()[0].shape[1] == DECODER_LAYOUT.size

    def test_history_is_oldest_first_and_primed(self):
        p = SonicPolicy(variant="low_latency")
        p.reset(np.zeros(29), np.zeros(29), np.zeros(3), np.array([0.0, 0.0, -1.0]))
        assert len(p.history.joint_pos) == 10
        for i in range(12):
            p.history.push(np.full(29, float(i)), np.zeros(29), np.zeros(29),
                           np.zeros(3), np.zeros(3))
        stacked = p.history.stack(p.history.joint_pos).reshape(10, 29)
        assert stacked[0, 0] < stacked[-1, 0]          # oldest first

    @pytest.mark.skipif(not SAMPLE.exists(), reason="sample motion not downloaded")
    def test_tracks_the_reference_without_falling(self):
        import sys
        sys.path.insert(0, "scripts")
        from eval_sonic import load_reference, rollout

        ref = load_reference(str(SAMPLE))
        r = rollout(ref, G1, "hardware", "xyzw", steps=150)
        assert not r["fell"], r
        assert r["joint_err"] < 0.25, r
        assert 0.6 < r["height"] < 0.9, r

    @pytest.mark.skipif(not SAMPLE.exists(), reason="sample motion not downloaded")
    def test_wrong_convention_falls_over(self):
        """The probe is only meaningful if a wrong convention actually fails."""
        import sys
        sys.path.insert(0, "scripts")
        from eval_sonic import load_reference, rollout

        ref = load_reference(str(SAMPLE))
        bad = rollout(ref, G1, "isaaclab", "wxyz", steps=150)
        assert bad["fell"] and bad["steps"] < 60, bad
