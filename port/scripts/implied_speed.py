"""Walking speed implied by a joint-angle reference alone (what SONIC's g1 mode sees).

Pelvis fixed upright at the origin, forward kinematics on SONIC's model; in each
frame the lower foot is taken as the stance foot, and its backward speed relative
to the pelvis is the speed the legs would carry the body at with no slip.
"""
import sys
from pathlib import Path
import numpy as np
import mujoco
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rpc.sim import sonic_scene
from rpc.sim import sonic as P

m = sonic_scene.load(); d = mujoco.MjData(m)
feet = [m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id]

def implied(jp_mj, fps=50.0, skip=50):
    pos = []
    for q in jp_mj:
        d.qpos[:] = 0; d.qpos[3] = 1.0; d.qpos[7:36] = q
        mujoco.mj_kinematics(m, d)
        pos.append([d.xpos[f].copy() for f in feet])
    pos = np.array(pos)[skip:]                          # (T, 2, 3)
    stance = np.argmin(pos[:, :, 2], axis=1)            # lower foot
    v = np.diff(pos[:, :, 0], axis=0) * fps             # forward velocity of each foot vs pelvis
    vs = -v[np.arange(len(v)), stance[1:]]              # stance foot sweeping back = body moving forward
    return float(np.median(vs)), float(np.mean(vs))

if __name__ == "__main__":
    inv = np.argsort(P.MUJOCO_TO_ISAACLAB)
    nv = np.loadtxt("assets/sonic/gr00t_wbc/gear_sonic_deploy/reference/example/walking_quip_360_R_002__A428/joint_pos.csv", delimiter=",", skiprows=1)
    print("NVIDIA walk (true 0.57 m/s): implied median %.2f mean %.2f" % implied(nv[:, inv][50:400]))
    for tag, cmd in (("h12r1b", 0.3), ("ours_v06", 0.6)):
        R = np.load(f"port/results/closed_loop/{tag}.phases.ref.npz"); jp = R["joint_pos"]
        # walking phase only: frames 9.5..17.5 s of stream time ~ index offset from first frame
        f0 = R["frame_index"][0]; sel = (R["frame_index"] - f0 > 50 * 9.5) & (R["frame_index"] - f0 < 50 * 17.5)
        print(f"ours cmd {cmd} (sim realised {'0.09' if cmd==0.3 else '0.35'}): implied median %.2f mean %.2f" % implied(jp[sel], skip=0))
