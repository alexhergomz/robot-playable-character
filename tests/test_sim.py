"""Physics-layer invariants.  These are slow-ish but they catch the failures
that only appear once the reference meets a real robot."""
from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from rpc.motion.clip import Skeleton
from rpc.policy.reference import ClipReference
from rpc.policy.tracker import TrackerConfig, TrackerEnv
from rpc.sim.env import RobotSim, SimConfig
from rpc.sim.robot import G1, chain_lengths, human_chain_lengths


@pytest.fixture(scope="module")
def sim():
    return RobotSim(G1, SimConfig())


class TestSim:
    def test_stands_under_zero_action(self, sim):
        sim.reset()
        h0 = sim.data.qpos[2]
        for _ in range(250):                       # 5 s
            sim.step(np.zeros(sim.model.nu))
        assert not sim.fallen()
        assert abs(sim.data.qpos[2] - h0) < 0.05

    def test_zero_action_is_the_default_pose(self, sim):
        assert np.allclose(sim.default_ctrl, sim.default_qpos[7:])

    def test_projected_gravity_is_down_when_upright(self, sim):
        sim.reset()
        assert np.allclose(sim.projected_gravity(), [0, 0, -1], atol=1e-6)

    def test_both_feet_contact_when_standing(self, sim):
        sim.reset()
        for _ in range(50):
            sim.step(np.zeros(sim.model.nu))
        assert sim.foot_contacts().sum() == 2

    def test_random_actions_fall(self, sim):
        sim.reset()
        rng = np.random.default_rng(0)
        for _ in range(300):
            sim.step(rng.uniform(-1, 1, sim.model.nu))
            if sim.fallen():
                return
        pytest.fail("random torques should topple an unbalanced humanoid")


def toy_skeleton() -> Skeleton:
    names = ("Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg",
             "RightFoot", "Spine2", "LeftArm", "LeftForeArm", "LeftHand",
             "RightArm", "RightForeArm", "RightHand")
    parents = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 7, 11, 12])
    o = np.zeros((14, 3))
    o[1] = [0, 0.10, -0.06]; o[2] = [0, 0, -0.44]; o[3] = [0, 0, -0.42]
    o[4] = [0, -0.10, -0.06]; o[5] = [0, 0, -0.44]; o[6] = [0, 0, -0.42]
    o[7] = [0, 0, 0.55]; o[8] = [0, 0.18, 0.08]; o[9] = [0, 0, -0.28]; o[10] = [0, 0, -0.25]
    o[11] = [0, -0.18, 0.08]; o[12] = [0, 0, -0.28]; o[13] = [0, 0, -0.25]
    return Skeleton(names, parents, o).with_inferred_axes()


class TestRetarget:
    def test_limb_scales_are_plausible(self):
        m = mujoco.MjModel.from_xml_path(G1.path())
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        skel = toy_skeleton()
        rl, hl = chain_lengths(m, d, G1), human_chain_lengths(skel, G1)
        for k in rl:
            s = rl[k] / hl[k]
            assert 0.3 < s < 1.2, f"{k} scale {s:.3f} is not a plausible body ratio"

    def test_ik_reaches_the_feet_and_grounds_them(self):
        from rpc.sim.retarget import RetargetConfig, Retargeter

        m = mujoco.MjModel.from_xml_path(G1.path())
        skel = toy_skeleton()
        rt = Retargeter(m, G1, skel, RetargetConfig(iters=25))
        rest = skel.rest_positions()
        pose = rest - rest[0] * 0
        pose[:, 2] += 0.92 - pose[0, 2]                  # stand the toy skeleton up
        q = rt.solve(pose, 0.0, np.zeros(2))
        res = rt.residuals(q, pose, 0.0, np.zeros(2))
        assert res["LeftFoot"] < 0.06 and res["RightFoot"] < 0.06, res
        d = rt.data
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        lowest = min(float(d.xpos[b][2]) for b in rt.foot_bids)
        assert abs(lowest) < 0.05, f"feet not grounded: {lowest:.3f} m"


class TestTrackerEnv:
    def make(self, T=400):
        qp = np.tile(np.concatenate([[0, 0, 0.79], [1, 0, 0, 0], np.zeros(29)]), (T, 1))
        for i, v in ((15, 0.2), (18, 1.28), (22, 0.2), (25, 1.28)):
            qp[:, 7 + i] = v
        ref = ClipReference(qp, np.ones((T, 2)), np.linspace(0, 8 * np.pi, T) % (2 * np.pi),
                            30.0, 50.0, 200)
        return TrackerEnv(G1, ref, TrackerConfig(episode_len=200))

    def test_on_reference_beats_random(self):
        env = self.make()
        env.reset(seed=0)
        good = sum(env.step(np.zeros(env.nu))[1] for _ in range(150))
        env.reset(seed=0)
        rng = np.random.default_rng(0)
        bad, n = 0.0, 0
        for _ in range(150):
            _, r, term, trunc, _ = env.step(rng.uniform(-1, 1, env.nu))
            bad += r
            n += 1
            if term or trunc:
                break
        assert good > bad, (good, bad)
        assert n < 150, "random actions should terminate the episode early"

    def test_observation_is_finite_and_stable_length(self):
        env = self.make()
        o = env.reset(seed=1)
        assert o.shape == (env.n_obs,) and np.isfinite(o).all()
        for _ in range(50):
            o, r, term, trunc, _ = env.step(np.zeros(env.nu))
            assert o.shape == (env.n_obs,) and np.isfinite(o).all()
            assert np.isfinite(r)


class TestOrientationTargets:
    """A 3-DoF waist takes exactly three constraints, not six."""

    def test_torso_frame_coincides_with_pelvis_at_zero_waist(self):
        """The orientation target assumes this; if the model ever changes, the
        target silently acquires a constant offset."""
        m = mujoco.MjModel.from_xml_path(G1.path())
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        pid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
        rel = d.xmat[pid].reshape(3, 3).T @ d.xmat[tid].reshape(3, 3)
        assert np.allclose(rel, np.eye(3), atol=1e-6)

    def test_waist_is_not_over_constrained(self):
        """Three waist DoF against the objectives that drive them."""
        n_waist = sum(1 for n in ("waist_yaw", "waist_roll", "waist_pitch"))
        driving = sum(3 for t in G1.ori_targets.values() if t.body == "torso_link")
        driving += sum(3 for t in G1.targets.values() if "shoulder" in t.body)
        assert driving <= n_waist, f"{driving} constraints on {n_waist} DoF"

    def test_position_only_ik_still_works_without_rotations(self):
        """Regression: the orientation loop used to index rotations that a
        position-only caller never supplied."""
        from rpc.sim.retarget import RetargetConfig, Retargeter

        m = mujoco.MjModel.from_xml_path(G1.path())
        skel = toy_skeleton()
        rt = Retargeter(m, G1, skel, RetargetConfig(iters=8))
        rest = skel.rest_positions()
        pose = rest.copy()
        pose[:, 2] += 0.92 - pose[0, 2]
        q = rt.solve(pose, 0.0, np.zeros(2))          # no joint_rot6d
        assert np.isfinite(q).all()
