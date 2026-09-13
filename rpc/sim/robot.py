"""Robot description and the human-to-robot correspondence.

Everything platform-specific lives here: which MuJoCo bodies correspond to
which mocap joints, how the limbs are scaled, and what a safe default pose is.
Swapping robots should mean writing one more :class:`RobotSpec`, not editing
the generator, the retargeter or the tracker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parents[2] / "assets"

#: The G1's nominal posture, taken from the GEAR-SONIC controller: a slight
#: crouch.  Retargeting toward it keeps the reference inside the executor's
#: training distribution as well as away from the knee stop.
_SONIC_NOMINAL = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
])


@dataclass(frozen=True)
class Target:
    """One IK objective.

    The offset is measured from an *anchor* — the hip for a foot, the shoulder
    for a hand — not from the root.  Scaling root-relative offsets instead
    bakes the human's hip width into the robot's stance: LAFAN1 subjects have a
    hip half-width of 12% of leg length against the G1's 8.5%, so a root-scaled
    foot target sits 27 cm outboard and saturates the hip-roll limit on every
    frame.  Anchoring absorbs that difference into the robot's own geometry.
    """

    body: str                      # robot body to move
    weight: float
    human_anchor: str              # mocap joint the offset is measured from
    robot_anchor: str              # robot body it is re-anchored to
    chain: str                     # which limb scale applies
    #: Optional per-axis scale for the lateral component, as a chain name.
    #: A human hand hangs ~12 cm inboard of its shoulder because human shoulders
    #: are wide (0.36 m); the G1's are 0.200 m.  Scaling that inboard offset by
    #: the *arm-length* ratio puts the hand on the centreline and the robot
    #: clasps its hands.  The lateral component belongs to shoulder width, not
    #: arm length.
    lateral_chain: str | None = None


@dataclass(frozen=True)
class OriTarget:
    """An orientation objective.

    A 3-DoF waist has exactly three degrees of freedom, so asking it to satisfy
    two shoulder *position* targets is six constraints on three unknowns.  The
    solve is over-determined, resolves differently frame to frame, and the
    resulting waist trajectory swings ~3x further than a physics-corrected
    reference -- on the heaviest body in the robot.  One orientation target is
    three constraints on three unknowns and is satisfiable exactly.
    """

    body: str                      # robot body whose orientation is matched
    weight: float
    human_joint: str               # mocap joint supplying the orientation
    human_anchor: str              # ...measured relative to this joint
    robot_anchor: str              # ...and applied relative to this body


@dataclass(frozen=True)
class Chain:
    """A limb, for scaling.  ``via`` are the bodies whose lengths we sum."""

    name: str
    via: tuple[str, ...]           # robot body names, root-to-tip
    human: tuple[str, ...]         # mocap joint names, root-to-tip


@dataclass
class RobotSpec:
    name: str
    xml: str
    root_body: str
    targets: dict[str, Target]     # mocap joint name -> objective
    chains: tuple[Chain, ...]
    foot_bodies: tuple[str, ...]
    keyframe: str = "stand"
    ori_targets: dict[str, "OriTarget"] = field(default_factory=dict)
    # joints excluded from IK (wrists/waist roll add nothing to a pose match and
    # cost conditioning); they are held at the default instead
    ik_exclude: tuple[str, ...] = ()
    #: Nominal operating posture, used as the IK seed and the nullspace prior.
    #: Not the same thing as the model's standing keyframe: the keyframe has
    #: straight legs, and near full extension a two-link leg's hip-to-ankle
    #: *distance* barely changes with knee angle, so a straight-leg prior
    #: resolves that near-degenerate direction by locking the knee against its
    #: limit.  Measured on walk1: 70% of frames at the knee stop.
    nominal_angles: np.ndarray | None = None
    action_scale: float = 0.25
    stand_height: float = 0.79

    def path(self) -> str:
        return str(ASSETS / self.xml)


G1 = RobotSpec(
    name="unitree_g1",
    xml="mujoco_menagerie/unitree_g1/scene.xml",
    root_body="pelvis",
    targets={
        # feet dominate: they set contact timing, which is what the phase
        # channel synchronises and what makes the robot fall if it is wrong
        "LeftFoot":     Target("left_ankle_roll_link", 6.0, "LeftUpLeg", "left_hip_pitch_link", "leg"),
        "RightFoot":    Target("right_ankle_roll_link", 6.0, "RightUpLeg", "right_hip_pitch_link", "leg"),
        "LeftLeg":      Target("left_knee_link", 1.5, "LeftUpLeg", "left_hip_pitch_link", "leg"),
        "RightLeg":     Target("right_knee_link", 1.5, "RightUpLeg", "right_hip_pitch_link", "leg"),
        "LeftHand":     Target("left_wrist_yaw_link", 2.0, "LeftArm", "left_shoulder_pitch_link", "arm", "shoulder_width"),
        "RightHand":    Target("right_wrist_yaw_link", 2.0, "RightArm", "right_shoulder_pitch_link", "arm", "shoulder_width"),
        "LeftForeArm":  Target("left_elbow_link", 1.0, "LeftArm", "left_shoulder_pitch_link", "arm", "shoulder_width"),
        "RightForeArm": Target("right_elbow_link", 1.0, "RightArm", "right_shoulder_pitch_link", "arm", "shoulder_width"),
    },
    ori_targets={
        # The waist gets an orientation target instead of two shoulder positions.
        "Spine2": OriTarget("torso_link", 3.0, "Spine2", "Hips", "pelvis"),
    },
    chains=(
        Chain("leg", ("left_hip_pitch_link", "left_knee_link", "left_ankle_roll_link"),
              ("LeftUpLeg", "LeftLeg", "LeftFoot")),
        Chain("arm", ("left_shoulder_pitch_link", "left_elbow_link", "left_wrist_yaw_link"),
              ("LeftArm", "LeftForeArm", "LeftHand")),
        # pelvis-to-shoulder, not pelvis-to-torso_link: the torso body origin
        # sits 4 cm above the pelvis, which would give a scale of 0.14
        Chain("torso", ("pelvis", "left_shoulder_pitch_link"), ("Hips", "LeftArm")),
        Chain("shoulder_width", ("left_shoulder_pitch_link", "right_shoulder_pitch_link"),
              ("LeftArm", "RightArm")),
    ),
    foot_bodies=("left_ankle_roll_link", "right_ankle_roll_link"),
    ik_exclude=("left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
                "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"),
    nominal_angles=_SONIC_NOMINAL,
)

REGISTRY = {"g1": G1}


def chain_lengths(model, data, spec: RobotSpec) -> dict[str, float]:
    """Robot limb lengths in the default pose."""
    import mujoco

    out = {}
    for ch in spec.chains:
        total = 0.0
        for a, b in zip(ch.via[:-1], ch.via[1:]):
            pa = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, a)]
            pb = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)]
            total += float(np.linalg.norm(pb - pa))
        out[ch.name] = total
    return out


def human_chain_lengths(skeleton, spec: RobotSpec) -> dict[str, float]:
    """Mocap limb lengths from the rest pose."""
    rest = skeleton.rest_positions()
    out = {}
    for ch in spec.chains:
        total = 0.0
        for a, b in zip(ch.human[:-1], ch.human[1:]):
            total += float(np.linalg.norm(rest[skeleton.index(b)] - rest[skeleton.index(a)]))
        out[ch.name] = total
    return out
