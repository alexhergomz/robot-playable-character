"""Parameters for NVIDIA's GEAR-SONIC whole-body policy.

SONIC is a pre-trained humanoid motion-tracking policy for the Unitree G1 —
an encoder that compresses a reference-motion window into a 64-D token, and a
decoder that turns that token plus proprioception into 29 joint actions.  It
gives us a physics-grounded tracker with no training at all, so the project's
own RL is a comparison rather than a dependency.

Every constant here is transcribed from NVLabs/GR00T-WholeBodyControl
(``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp``
and ``robot_parameters.hpp``).  They are exactly the numbers that silently
break a downloaded policy if you guess them:

* SONIC's default pose is a **crouch** (hip -0.31, knee +0.67, ankle -0.36),
  not the Menagerie ``stand`` keyframe's straight legs.  Actions are offsets
  from *its* default, so using the keyframe biases every joint target.
* The policy runs in **IsaacLab** joint order; hardware and MuJoCo use a
  different one.  ``ISAACLAB_TO_MUJOCO`` is the bridge.
* Action scale is per-joint, derived from the motor's effort limit and a
  second-order model, not a single global gain.

Code in that repository is Apache-2.0; the weights are under the NVIDIA Open
Model License (commercial use permitted with attribution).
"""
from __future__ import annotations

import numpy as np

N_MOTOR = 29

# -- motor model -----------------------------------------------------------
ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520,
            "7520_22": 0.025101925, "4010": 0.00425}
EFFORT_LIMIT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
NATURAL_FREQ = 10.0 * 2.0 * np.pi          # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS = {k: v * NATURAL_FREQ ** 2 for k, v in ARMATURE.items()}
DAMPING = {k: 2.0 * DAMPING_RATIO * v * NATURAL_FREQ for k, v in ARMATURE.items()}

#: Motor type per joint, in MuJoCo/hardware order.
MOTOR_TYPE = (
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",        # left leg
    "7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020",        # right leg
    "7520_14", "5020", "5020",                                          # waist
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",            # left arm
    "5020", "5020", "5020", "5020", "5020", "4010", "4010",            # right arm
)
assert len(MOTOR_TYPE) == N_MOTOR

#: ``target = action * ACTION_SCALE + DEFAULT_ANGLES``
ACTION_SCALE = np.array([0.25 * EFFORT_LIMIT[t] / STIFFNESS[t] for t in MOTOR_TYPE])
KP = np.array([STIFFNESS[t] for t in MOTOR_TYPE], dtype=np.float32)
KD = np.array([DAMPING[t] for t in MOTOR_TYPE], dtype=np.float32)

#: SONIC's nominal pose — a slight crouch, unlike the Menagerie stand keyframe.
DEFAULT_ANGLES = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,          # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,          # right leg
    0.0, 0.0, 0.0,                                  # waist yaw / roll / pitch
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,             # left arm
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,            # right arm
])

#: Joint order used inside the policy differs from MuJoCo/hardware order.
ISAACLAB_TO_MUJOCO = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
                               11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28])
MUJOCO_TO_ISAACLAB = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
                               16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])

#: MuJoCo/hardware joint order, for asserting against a loaded model.
JOINT_NAMES = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)

#: Decoder (policy) input, 994-D; encoder output token is 64-D.
TOKEN_DIM = 64
DECODER_DIM = 994
ENCODER_DIM = {"default": 1762, "low_latency": 1247}

#: Reference-motion lookahead for the ``g1`` encoder mode.  ``step`` is in
#: 20 ms policy frames, so the default variant needs a full second of future
#: motion and the low-latency variant needs 200 ms.  For an operator-driven
#: character that difference is the responsiveness budget, so ``low_latency``
#: is the variant this project wants.
LOOKAHEAD = {"default": {"frames": 10, "step": 5}, "low_latency": {"frames": 10, "step": 1}}

#: The reference format SONIC's sample data uses — the same content our
#: retargeter already produces, one rename apart.
#:   root_trans_offset (T,3) | root_rot (T,4) wxyz | dof (T,29) | fps
REFERENCE_FIELDS = ("root_trans_offset", "root_rot", "dof", "fps")


def verify_against_model(model) -> dict:
    """Check a loaded MuJoCo G1 against SONIC's assumptions."""
    import mujoco

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(model.nu)]
    order_ok = names == [f"{n}_joint" for n in JOINT_NAMES]
    perm_ok = bool(np.all(ISAACLAB_TO_MUJOCO[MUJOCO_TO_ISAACLAB] == np.arange(N_MOTOR)))
    lo, hi = model.jnt_range[1:N_MOTOR + 1].T
    within = bool(np.all((DEFAULT_ANGLES >= lo - 1e-6) & (DEFAULT_ANGLES <= hi + 1e-6)))
    return {"n_actuators": model.nu, "joint_order_matches": order_ok,
            "permutation_is_an_involution_pair": perm_ok,
            "default_pose_within_limits": within}
