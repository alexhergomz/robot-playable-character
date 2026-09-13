"""Human reference motion -> robot joint targets.

Damped least squares on MuJoCo's own body Jacobians.  Nothing clever: the
generator produces mocap-skeleton joint positions, we scale them per limb onto
the robot's proportions, and solve for the joint angles that put the robot's
corresponding bodies there.

Two choices are worth stating.  First, the solve runs in the *character frame*
with the floating base pinned, so the base trajectory is set analytically and
only the 23 useful joints are solved — the IK never has to trade base drift
against pose error.  Second, the previous solution seeds the next one, which
makes the retargeter itself temporally coherent; without it the IK jumps
between equally-good solutions and hands the tracker a reference that is
discontinuous at 30 Hz.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ..motion import quat as Q
from ..motion.clip import Skeleton
from .robot import RobotSpec, chain_lengths, human_chain_lengths


@dataclass
class RetargetConfig:
    iters: int = 12
    damping: float = 0.08
    step: float = 0.9
    posture_weight: float = 0.02     # pull toward the default pose in the nullspace
    tol: float = 2e-3                # metres; stop early once every target is this close
    ori_gain: float = 1.0            # <1 damps orientation targets toward the anchor


class Retargeter:
    #: Chains whose human length is meaningless in the rest pose.
    dynamic_chains: tuple[str, ...] = ("shoulder_width",)

    def __init__(self, model, spec: RobotSpec, skeleton: Skeleton,
                 cfg: RetargetConfig | None = None,
                 motion_positions: np.ndarray | None = None):
        self.model = model
        self.spec = spec
        self.skeleton = skeleton
        self.cfg = cfg or RetargetConfig()
        self.data = mujoco.MjData(model)

        key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, spec.keyframe)
        self.default_qpos = np.array(model.key_qpos[key]) if key >= 0 else np.zeros(model.nq)
        if spec.nominal_angles is not None:
            # Seed and regularise toward the robot's nominal operating posture
            # rather than the model's standing keyframe.
            self.default_qpos[7:] = np.asarray(spec.nominal_angles)
        mujoco.mj_resetDataKeyframe(model, self.data, max(key, 0))
        mujoco.mj_forward(model, self.data)

        rl = chain_lengths(model, self.data, spec)
        hl = human_chain_lengths(skeleton, spec)
        # Some chains cannot be measured from the rest pose.  LAFAN1's rest pose
        # has the clavicles folded in, putting the shoulder joints 0.120 m apart
        # against 0.242 m during actual motion -- a rest-pose ratio of 1.67 that
        # would throw the hands *outward* instead of correcting them.  Where
        # motion is available, measure those chains from it.
        if motion_positions is not None:
            for ch in spec.chains:
                if ch.name not in self.dynamic_chains:
                    continue
                total = 0.0
                for a, b in zip(ch.human[:-1], ch.human[1:]):
                    ia, ib = skeleton.index(a), skeleton.index(b)
                    total += float(np.linalg.norm(
                        motion_positions[:, ib] - motion_positions[:, ia], axis=-1).mean())
                hl[ch.name] = total
        self.scale = {k: rl[k] / max(hl[k], 1e-6) for k in rl}

        self.pairs = []
        for mocap_name, t in spec.targets.items():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, t.body)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, t.robot_anchor)
            assert bid >= 0 and aid >= 0, f"missing body for target {mocap_name!r}"
            sc = np.full(3, self.scale[t.chain])
            if t.lateral_chain:
                sc[1] = self.scale[t.lateral_chain]      # y is lateral in the character frame
            self.pairs.append((mocap_name, skeleton.index(mocap_name),
                               skeleton.index(t.human_anchor), bid, aid,
                               t.weight, sc))

        # DoF columns the IK is allowed to move: joints, minus the excluded ones
        free = []
        for j in range(model.njnt):
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name in spec.ik_exclude:
                continue
            free.append(model.jnt_dofadr[j])
        self.dofs = np.array(free, dtype=int)
        self.qadr = np.array([model.jnt_qposadr[model.dof_jntid[d]] for d in self.dofs])
        self.ori_pairs = []
        for name, t in spec.ori_targets.items():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, t.body)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, t.robot_anchor)
            assert bid >= 0 and aid >= 0, f"missing body for ori target {name!r}"
            self.ori_pairs.append((name, skeleton.index(t.human_joint),
                                   skeleton.index(t.human_anchor), bid, aid, t.weight))
        self._jacr = np.zeros((3, model.nv))
        self.foot_bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
                          for b in spec.foot_bodies]
        self.sole_offset = self._measure_sole_offset()
        # Ankle *and* toe: the clip was ground-aligned on both, so the minimum
        # over them is ~0 at contact.  Using the ankle alone registers against a
        # joint that sits 6 cm up even when the foot is flat on the floor.
        self.human_foot_idx = []
        for side in ("Left", "Right"):
            for part in ("Foot", "Toe"):
                try:
                    self.human_foot_idx.append(skeleton.find(side, part))
                except KeyError:
                    pass
        # Canonical basis of the mocap root, as columns in root-local coords:
        # [forward, left, up].  Post-multiplying the root's character-frame
        # rotation by this turns "human root orientation" into "robot pelvis
        # orientation", which is identity when the character stands upright.
        left = np.cross(skeleton.up, skeleton.fwd)
        self.canonical_basis = np.stack([skeleton.fwd, left, skeleton.up], axis=1)
        self.q = self.default_qpos.copy()
        self._jac = np.zeros((3, model.nv))

    def _measure_sole_offset(self) -> float:
        """Height of the foot body's origin above the ground when standing.

        Read straight off the model's own standing keyframe, which already rests
        the robot on z=0 — far more reliable than reasoning about geom extents
        (``geom_rbound`` is a bounding-sphere radius, which for a flat foot
        overestimates the drop by an order of magnitude).

        This matters because the mocap "Foot" joint is the ankle and sits ~6 cm
        up in a human, while the robot's ankle body sits ~3 cm above its sole.
        Grounding one against the other leaves the reference hovering for the
        whole clip, and the tracker spends the episode reaching for a floor that
        is not there.
        """
        mujoco.mj_forward(self.model, self.data)
        off = float(np.mean([self.data.xpos[b][2] for b in self.foot_bids]))
        assert 0.005 < off < 0.15, f"implausible sole offset {off:.3f} m"
        return off

    # -- root placement -----------------------------------------------------
    def root_transform(self, joint_pos_local: np.ndarray, root_yaw: float,
                       root_xy: np.ndarray, root_rot6d: np.ndarray | None = None):
        """Robot base pose from the human root, scaled to the robot's leg length.

        ``root_rot6d`` is the mocap root's rotation in the character frame (the
        heading-removed pitch and roll).  Dropping it — which this code did
        originally, commanding a pure yaw — pins the pelvis rigidly upright.
        A human walking tilts the pelvis 5.5 deg on average and up to 23 deg,
        and every degree of that has to be absorbed by the leg joints instead,
        which shows up as a standing bias in the retargeted hip angles.
        """
        z = float(joint_pos_local[0, 2]) * self.scale["leg"]
        pos = np.array([root_xy[0], root_xy[1], z])
        quat = Q.yaw_quat(np.array(root_yaw))
        if root_rot6d is not None:
            R = Q.to_matrix(Q.unflatten_2axis(np.asarray(root_rot6d)))
            quat = Q.mul(quat, Q.from_matrix(R @ self.canonical_basis))
        return pos, Q.normalize(quat)

    # -- the solve ----------------------------------------------------------
    def _ori_targets(self, joint_rot6d: np.ndarray) -> list[np.ndarray]:
        """Desired world rotation matrices for each orientation target.

        Mocap joint frames are all parallel to the root's at rest (BVH stores
        rest pose as offsets with identity rotations), so one basis change
        converts any relative joint rotation into the robot's convention.  The
        G1's torso frame coincides with its pelvis frame when the waist is at
        zero, which is verified in the tests.
        """
        C = self.canonical_basis
        out = []
        for _n, hj, ha, _b, _a, _w in self.ori_pairs:
            Ra = Q.to_matrix(Q.unflatten_2axis(joint_rot6d[ha]))
            Rj = Q.to_matrix(Q.unflatten_2axis(joint_rot6d[hj]))
            rel = C.T @ (Ra.T @ Rj) @ C                # anchor-relative, robot frame
            if abs(self.cfg.ori_gain - 1.0) > 1e-9:
                # Scale the rotation *vector*, not the matrix: a body with a
                # different inertia distribution should not copy the human's
                # trunk excursion one-for-one.
                rv = Q.to_rotvec(Q.from_matrix(rel)) * self.cfg.ori_gain
                ang = np.linalg.norm(rv)
                axis = rv / ang if ang > 1e-9 else np.array([0.0, 0.0, 1.0])
                rel = Q.to_matrix(Q.from_axis_angle(axis, np.array(ang)))
            out.append(rel)
        return out

    def solve(self, joint_pos_local: np.ndarray, root_yaw: float,
              root_xy: np.ndarray, warm_start: bool = True,
              root_rot6d: np.ndarray | None = None,
              joint_rot6d: np.ndarray | None = None) -> np.ndarray:
        """Return a full qpos tracking the given human pose."""
        m, d = self.model, self.data
        pos, quat = self.root_transform(joint_pos_local, root_yaw, root_xy, root_rot6d)

        q = (self.q if warm_start else self.default_qpos).copy()
        q[:3] = pos
        q[3:7] = quat

        # Offsets are re-anchored each iteration because an anchor can itself
        # move (the shoulder rides on the waist joints), so the arm targets
        # follow wherever the torso ends up rather than fighting it.
        hq = quat
        offsets = np.stack([
            Q.rotate(hq, (joint_pos_local[hj] - joint_pos_local[ha]) * s)
            for (_n, hj, ha, _b, _a, _w, s) in self.pairs
        ])

        rel_ori = self._ori_targets(joint_rot6d) if (joint_rot6d is not None
                                                     and self.ori_pairs) else []
        n_ori = len(rel_ori)
        nd = len(self.dofs)
        for _ in range(self.cfg.iters):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)

            n_pos = len(self.pairs)
            err = np.empty(3 * (n_pos + n_ori))
            Jstack = np.empty((3 * (n_pos + n_ori), nd))
            worst = 0.0
            for i, (_n, _hj, _ha, bid, aid, w, _s) in enumerate(self.pairs):
                e = (d.xpos[aid] + offsets[i]) - d.xpos[bid]
                worst = max(worst, float(np.linalg.norm(e)))
                sw = np.sqrt(w)
                err[3 * i:3 * i + 3] = e * sw
                mujoco.mj_jacBody(m, d, self._jac, None, bid)
                Jstack[3 * i:3 * i + 3] = self._jac[:, self.dofs] * sw
            # Only the targets we actually built rotations for: a caller that
            # passes no joint rotations gets position-only IK.
            for j, (_n, _hj, _ha, bid, aid, w) in enumerate(self.ori_pairs[:n_ori]):
                Rcur = d.xmat[bid].reshape(3, 3)
                Rtgt = d.xmat[aid].reshape(3, 3) @ rel_ori[j]
                e = Q.to_rotvec(Q.from_matrix(Rtgt @ Rcur.T))
                sw = np.sqrt(w)
                r = 3 * (n_pos + j)
                err[r:r + 3] = e * sw
                mujoco.mj_jacBody(m, d, None, self._jacr, bid)
                Jstack[r:r + 3] = self._jacr[:, self.dofs] * sw

            if worst < self.cfg.tol and n_ori == 0:
                break

            # damped least squares with a posture term in the same solve
            A = Jstack.T @ Jstack + (self.cfg.damping ** 2 + self.cfg.posture_weight) * np.eye(nd)
            b = Jstack.T @ err + self.cfg.posture_weight * (
                self.default_qpos[self.qadr] - q[self.qadr])
            dq = np.linalg.solve(A, b)
            q[self.qadr] += self.cfg.step * dq
            lo, hi = m.jnt_range[m.dof_jntid[self.dofs]].T
            q[self.qadr] = np.clip(q[self.qadr], lo, hi)

        # Ground the solution.  The anchored IK fixes the pose but says nothing
        # about absolute height, and a base placed by scaling the human's root
        # leaves the feet a few centimetres through the floor — which the
        # tracker below would spend its whole reward budget fighting.
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        have = min(float(d.xpos[b][2]) for b in self.foot_bids)
        # The human clip is ground-aligned, so its lowest foot joint is ~0 at
        # contact; the robot's ankle body must then sit one sole-offset up.
        want = (min(float(joint_pos_local[j][2]) for j in self.human_foot_idx)
                * self.scale["leg"] + self.sole_offset)
        q[2] += want - have

        self.q = q
        return q

    def residuals(self, q: np.ndarray, joint_pos_local: np.ndarray,
                  root_yaw: float, root_xy: np.ndarray,
                  root_rot6d: np.ndarray | None = None) -> dict[str, float]:
        """Per-target position error, metres — the honest measure of retarget quality."""
        m, d = self.model, self.data
        pos, hq = self.root_transform(joint_pos_local, root_yaw, root_xy, root_rot6d)
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        _ = pos
        out = {}
        for name, hj, ha, bid, aid, _w, s in self.pairs:
            rel = Q.rotate(hq, (joint_pos_local[hj] - joint_pos_local[ha]) * s)
            out[name] = float(np.linalg.norm(d.xpos[aid] + rel - d.xpos[bid]))
        return out
