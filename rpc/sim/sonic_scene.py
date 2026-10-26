"""SONIC's own G1 model, for simulation that matches what SONIC was trained on.

Everything earlier in this project simulated MuJoCo Menagerie's G1.  That model
differs from SONIC's (``g1_29dof_rev_1_0.xml`` in NVlabs/GR00T-WholeBodyControl)
exactly where a tracking policy is most sensitive: each of SONIC's feet touches
the ground through four 5 mm point spheres at the heel and toe corners with
MuJoCo's default friction of 1.0, where Menagerie uses 0.6-friction spheres plus
a mesh.  SONIC's low-latency encoder has no root-position input, so whether the
robot stands still is decided entirely by contact physics.

The model drives joints with torque motors.  ``PDStep`` closes the PD loop at
every physics step -- the way the real motor drivers and NVIDIA's MuJoCo bridge
do -- with SONIC's gains and per-motor effort limits.
"""
from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np

from . import sonic as P

ROOT = Path(__file__).resolve().parents[2]
MODEL = Path(__file__).resolve().parent / "models/g1_29dof_rev_1_0.xml"
MESHES = ROOT / "assets/mujoco_menagerie/unitree_g1/assets"


def load(timestep: float = 0.002) -> mujoco.MjModel:
    xml = MODEL.read_text()
    xml = re.sub(r'meshdir="[^"]*"', f'meshdir="{MESHES}/"', xml)
    # the file is already a complete scene (floor at default friction, light)
    m = mujoco.MjModel.from_xml_string(xml)
    m.opt.timestep = timestep
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, m.njnt)]
    expect = [n + "_joint" for n in P.JOINT_NAMES]
    if names != expect:
        raise ValueError("joint order differs from the order the rest of the code assumes")
    for j in range(29):
        m.dof_armature[6 + j] = P.ARMATURE[P.MOTOR_TYPE[j]]
    return m


class PDStep:
    """tau = kp (q* - q) - kd qdot, clipped to the motor's effort limit, per step."""

    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.kp = np.array([P.STIFFNESS[t] for t in P.MOTOR_TYPE])
        self.kd = np.array([P.DAMPING[t] for t in P.MOTOR_TYPE])
        self.lim = np.array([P.EFFORT_LIMIT[t] for t in P.MOTOR_TYPE])

    def __call__(self, d: mujoco.MjData, target: np.ndarray, substeps: int):
        for _ in range(substeps):
            tau = self.kp * (target - d.qpos[7:]) - self.kd * d.qvel[6:]
            d.ctrl[:] = np.clip(tau, -self.lim, self.lim)
            mujoco.mj_step(self.m, d)
