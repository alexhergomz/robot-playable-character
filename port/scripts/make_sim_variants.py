"""Create the model variants used in the NVIDIA-sim experiments (inside the git-ignored upstream checkout).

* g1_pointfeet/  NVIDIA's 43-DoF scene with SONIC's 4 point contacts per foot and no joint frictionloss.
* g1_sonic_rev1/ SONIC's training model (g1_29dof_rev_1_0) with SONIC's per-motor armature baked in;
                 run with nvsim_run.py --no-hands.

Usage: python port/scripts/make_sim_variants.py
"""
import re
import shutil
import sys
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rpc.sim import sonic as P  # noqa: E402

G = ROOT / "assets/sonic/gr00t_wbc/gear_sonic/data/robot_model/model_data"

pf = G / "g1_pointfeet"
if pf.exists():
    shutil.rmtree(pf)
shutil.copytree(G / "g1", pf, ignore=shutil.ignore_patterns("meshes"))
(pf / "meshes").symlink_to(G / "g1/meshes")
x = (pf / "g1_29dof_with_hand.xml").read_text()
box = ('<geom type="box" size="0.085 0.03 0.005" pos="0.035 0 -0.03"\n'
       '                    rgba="0.2 0.2 0.2 0.4" />')
assert x.count(box) == 2, "foot box geoms not found; upstream model changed"
spheres = "\n                  ".join(f'<geom size="0.005" pos="{px} {py} -0.03" rgba="0.2 0.2 0.2 1" />'
                                     for px, py in ((-0.05, 0.025), (-0.05, -0.025), (0.12, 0.03), (0.12, -0.03)))
x = re.sub(r'frictionloss="[0-9.]+"', 'frictionloss="0"', x.replace(box, spheres))
(pf / "g1_29dof_with_hand.xml").write_text(x)

sr = G / "g1_sonic_rev1"
sr.mkdir(exist_ok=True)
src = (ROOT / "rpc/sim/models/g1_29dof_rev_1_0.xml").read_text()
src = re.sub(r'meshdir="[^"]*"', f'meshdir="{ROOT / "assets/mujoco_menagerie/unitree_g1/assets"}"', src)
(sr / "scene.xml").write_text(src)
spec = mujoco.MjSpec.from_file(str(sr / "scene.xml"))
for j, n in enumerate(P.JOINT_NAMES):
    spec.joint(n + "_joint").armature = P.ARMATURE[P.MOTOR_TYPE[j]]
(sr / "scene.xml").write_text(spec.to_xml())
mujoco.MjModel.from_xml_path(str(sr / "scene.xml"))
print("wrote", pf, "and", sr)
