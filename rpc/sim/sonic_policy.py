"""Driving GEAR-SONIC as the low-level executor.

The generator animates; this stabilises.  SONIC is a pre-trained motion-tracking
policy for the G1: an encoder squeezes a window of *reference* motion into a
64-D token, a decoder turns that token plus a history of proprioception into 29
joint actions.  It is not a substitute for the generator — SONIC's own kinematic
planner fills that slot and we deliberately do not use it.

Everything here is a transcription of NVLabs/GR00T-WholeBodyControl's deploy
reference (``g1_deploy_onnx_ref.cpp``).  The details that silently break a
downloaded policy, all confirmed against that source:

* Observation terms are concatenated **in the order the YAML lists them**, each
  contributing its registry dimension.  The three layouts check out exactly:
  decoder 994, low-latency encoder 1247, default encoder 1762.
* In ``g1`` mode only four encoder terms are populated; every other term is
  zero-filled.  ``encoder_mode_4`` is *not* a one-hot — it is the mode id in
  slot 0 followed by three zeros, so for g1 the whole block is zeros.
* Proprioception is ``q - default_angle``, but the reference joint angles are
  **absolute**.  Assuming that asymmetry away costs you the default pose offset
  on every joint of every reference frame.
* History is **oldest-first** (``newest_first`` is left false everywhere).
* Observations are in IsaacLab joint order, motor commands in hardware order.
* SONIC expects its own PD gains, which are far softer than the Menagerie
  model's baked-in position servos (kp 14-99 against 500).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import sonic as P

HISTORY = 10                 # frames of proprioception the decoder sees
MOTOR_TYPE_BY_INDEX = P.MOTOR_TYPE
CONTROL_HZ = 50.0            # 20 ms, matching the 10-frame/200 ms windows


def _six_d(R: np.ndarray) -> np.ndarray:
    """SONIC's 6-D rotation layout: first two columns, read row-major.

    Note the interleaving — (r00, r01, r10, r11, r20, r21) — which is *not* the
    column-major concatenation used elsewhere in this project.
    """
    return np.array([R[0, 0], R[0, 1], R[1, 0], R[1, 1], R[2, 0], R[2, 1]])


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


@dataclass
class Layout:
    """Named offsets into a flat observation vector."""

    blocks: tuple[tuple[str, int], ...]

    @property
    def offsets(self) -> dict[str, slice]:
        out, off = {}, 0
        for name, n in self.blocks:
            out[name] = slice(off, off + n)
            off += n
        return out

    @property
    def size(self) -> int:
        return sum(n for _, n in self.blocks)


#: Decoder input, from ``observation_config.yaml`` -> 994.
DECODER_LAYOUT = Layout((
    ("token_state", 64),
    ("his_base_angular_velocity_10frame_step1", 30),
    ("his_body_joint_positions_10frame_step1", 290),
    ("his_body_joint_velocities_10frame_step1", 290),
    ("his_last_actions_10frame_step1", 290),
    ("his_gravity_dir_10frame_step1", 30),
))

#: Low-latency encoder input -> 1247.  Only the first four blocks are filled in
#: g1 mode; the teleop and smpl blocks stay zero.
ENCODER_LAYOUT_LOW_LATENCY = Layout((
    ("encoder_mode_4", 4),
    ("motion_joint_positions_10frame_step1", 290),
    ("motion_joint_velocities_10frame_step1", 290),
    ("motion_anchor_orientation_10frame_step1", 60),
    ("motion_anchor_orientation", 6),
    ("motion_joint_positions_lowerbody_10frame_step1", 120),
    ("motion_joint_velocities_lowerbody_10frame_step1", 120),
    ("vr_3point_local_target", 9),
    ("vr_3point_local_orn_target", 12),
    ("smpl_joints_4frame_step1", 288),
    ("smpl_anchor_orientation_4frame_step1", 24),
    ("motion_joint_positions_wrists_4frame_step1", 24),
))

#: Default (1 s lookahead) encoder input -> 1762.
ENCODER_LAYOUT_DEFAULT = Layout((
    ("encoder_mode_4", 4),
    ("motion_joint_positions_10frame_step5", 290),
    ("motion_joint_velocities_10frame_step5", 290),
    ("motion_root_z_position_10frame_step5", 10),
    ("motion_root_z_position", 1),
    ("motion_anchor_orientation", 6),
    ("motion_anchor_orientation_10frame_step5", 60),
    ("motion_joint_positions_lowerbody_10frame_step5", 120),
    ("motion_joint_velocities_lowerbody_10frame_step5", 120),
    ("vr_3point_local_target", 9),
    ("vr_3point_local_orn_target", 12),
    ("smpl_joints_10frame_step1", 720),
    ("smpl_anchor_orientation_10frame_step1", 60),
    ("motion_joint_positions_wrists_10frame_step1", 60),
))

VARIANTS = {
    "low_latency": (ENCODER_LAYOUT_LOW_LATENCY, 1, "low_latency"),
    "default": (ENCODER_LAYOUT_DEFAULT, 5, ""),
}


@dataclass
class ProprioHistory:
    """Oldest-first ring buffers, in IsaacLab joint order."""

    n: int = HISTORY
    joint_pos: list = field(default_factory=list)
    joint_vel: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    ang_vel: list = field(default_factory=list)
    gravity: list = field(default_factory=list)

    def push(self, jp, jv, act, av, g) -> None:
        for buf, v in ((self.joint_pos, jp), (self.joint_vel, jv), (self.actions, act),
                       (self.ang_vel, av), (self.gravity, g)):
            buf.append(np.asarray(v, dtype=np.float64))
            if len(buf) > self.n:
                buf.pop(0)

    def fill(self, jp, jv, act, av, g) -> None:
        """Prime every slot, so the first step is not fed zeros."""
        for _ in range(self.n):
            self.push(jp, jv, act, av, g)

    def stack(self, buf) -> np.ndarray:
        return np.concatenate(buf)          # oldest first


class SonicPolicy:
    """ONNX encoder + decoder, driven by a reference-motion window."""

    def __init__(self, root: str = "assets/sonic", variant: str = "low_latency",
                 providers: list[str] | None = None):
        import onnxruntime as ort
        from pathlib import Path

        layout, step, sub = VARIANTS[variant]
        self.variant = variant
        self.enc_layout = layout
        self.step = step
        self.lookahead_frames = HISTORY
        base = Path(root) / sub
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        prov = providers or ["CPUExecutionProvider"]
        self.enc = ort.InferenceSession(str(base / "model_encoder.onnx"), opts, providers=prov)
        self.dec = ort.InferenceSession(str(base / "model_decoder.onnx"), opts, providers=prov)
        assert self.enc.get_inputs()[0].shape[1] == layout.size, (
            f"encoder expects {self.enc.get_inputs()[0].shape[1]}, layout gives {layout.size}")
        assert self.dec.get_inputs()[0].shape[1] == DECODER_LAYOUT.size

        self.enc_off = layout.offsets
        self.dec_off = DECODER_LAYOUT.offsets
        self.history = ProprioHistory()
        self.last_action = np.zeros(P.N_MOTOR)
        self.token = np.zeros(P.TOKEN_DIM, dtype=np.float32)

    # -- observation assembly ------------------------------------------------
    def encoder_obs(self, ref_dof: np.ndarray, ref_dvel: np.ndarray,
                    ref_quat: np.ndarray, base_quat: np.ndarray) -> np.ndarray:
        """Reference window -> encoder input.

        ``ref_dof``/``ref_dvel`` are (10, 29) in IsaacLab order at the variant's
        frame spacing; ``ref_quat`` is (10, 4) world root orientation.
        """
        x = np.zeros(self.enc_layout.size, dtype=np.float32)
        # encoder_mode_4: mode id then zeros. g1 is mode 0, so this block is zero.
        x[self.enc_off["encoder_mode_4"]] = 0.0
        pos_key = f"motion_joint_positions_10frame_step{self.step}"
        vel_key = f"motion_joint_velocities_10frame_step{self.step}"
        ori_key = f"motion_anchor_orientation_10frame_step{self.step}"
        x[self.enc_off[pos_key]] = ref_dof.reshape(-1)
        x[self.enc_off[vel_key]] = ref_dvel.reshape(-1)

        inv_base = _quat_conj(base_quat)
        ori = np.concatenate([_six_d(_quat_to_mat(_quat_mul(inv_base, q))) for q in ref_quat])
        x[self.enc_off[ori_key]] = ori
        return x[None]

    def decoder_obs(self) -> np.ndarray:
        h = self.history
        x = np.zeros(DECODER_LAYOUT.size, dtype=np.float32)
        x[self.dec_off["token_state"]] = self.token
        x[self.dec_off["his_base_angular_velocity_10frame_step1"]] = h.stack(h.ang_vel)
        x[self.dec_off["his_body_joint_positions_10frame_step1"]] = h.stack(h.joint_pos)
        x[self.dec_off["his_body_joint_velocities_10frame_step1"]] = h.stack(h.joint_vel)
        x[self.dec_off["his_last_actions_10frame_step1"]] = h.stack(h.actions)
        x[self.dec_off["his_gravity_dir_10frame_step1"]] = h.stack(h.gravity)
        return x[None]

    # -- one control step ----------------------------------------------------
    def step_policy(self, ref_dof, ref_dvel, ref_quat, base_quat,
                    joint_pos_hw, joint_vel_hw, ang_vel, gravity) -> np.ndarray:
        """Returns joint position targets in hardware order."""
        jp = (joint_pos_hw - P.DEFAULT_ANGLES)[P.MUJOCO_TO_ISAACLAB]
        jv = joint_vel_hw[P.MUJOCO_TO_ISAACLAB]
        self.history.push(jp, jv, self.last_action, ang_vel, gravity)

        self.token = self.enc.run(
            None, {"obs_dict": self.encoder_obs(ref_dof, ref_dvel, ref_quat, base_quat)}
        )[0][0]
        action = self.dec.run(None, {"obs_dict": self.decoder_obs()})[0][0]
        self.last_action = action.astype(np.float64)
        return P.DEFAULT_ANGLES + action[P.ISAACLAB_TO_MUJOCO] * P.ACTION_SCALE

    def reset(self, joint_pos_hw, joint_vel_hw, ang_vel, gravity) -> None:
        self.history = ProprioHistory()
        self.last_action = np.zeros(P.N_MOTOR)
        self.token = np.zeros(P.TOKEN_DIM, dtype=np.float32)
        jp = (joint_pos_hw - P.DEFAULT_ANGLES)[P.MUJOCO_TO_ISAACLAB]
        jv = joint_vel_hw[P.MUJOCO_TO_ISAACLAB]
        self.history.fill(jp, jv, self.last_action, ang_vel, gravity)


def apply_sonic_armature(model) -> None:
    """Give each joint the rotor inertia SONIC's gains were derived from.

    SONIC computes kp = armature * (2*pi*10)^2 per motor type, so the gains only
    behave as intended if the simulated armature matches.  The Menagerie model
    uses a uniform 0.010 for all 29 joints against SONIC's 0.0036-0.0251 — up to
    2.5x too little rotor inertia at the hips and knees, and 2.8x too much at the
    ankles.  That is a sim-fidelity gap, not a policy defect.
    """
    for i, t in enumerate(MOTOR_TYPE_BY_INDEX):
        jid = i + 1                      # joint 0 is the floating base
        model.dof_armature[model.jnt_dofadr[jid]] = P.ARMATURE[t]


def apply_sonic_gains(model) -> None:
    """Retune the MuJoCo position actuators to SONIC's PD gains.

    Menagerie ships kp=500 / kd=43; SONIC was trained against kp 14-99.  Left
    alone, the model is roughly ten times too stiff and the policy's action
    offsets are swamped by the servo.
    """
    for i in range(model.nu):
        model.actuator_gainprm[i, 0] = P.KP[i]
        model.actuator_biasprm[i, 1] = -P.KP[i]
        model.actuator_biasprm[i, 2] = -P.KD[i]
