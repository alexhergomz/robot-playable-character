"""NVIDIA's own G1 MuJoCo simulator (gear_sonic/utils/mujoco_sim), headless, recorded.

It talks to the deploy app exactly as the real robot does: rt/lowstate out,
rt/lowcmd in, over DDS on the loopback interface.  This wrapper changes none
of that; it only drives what a person would do in NVIDIA's viewer:

* lowers the elastic band so the feet touch the ground (viewer key 7, twice),
* releases the band when the file given by --release-flag appears (key 9),
* records an mp4 from a tracking camera and an npz log of the base.

Run with port/sim_env/bin/python (Python 3.10, NVIDIA's unitree_sdk2py fork).
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np

GR00T = Path(__file__).resolve().parents[2] / "assets/sonic/gr00t_wbc"
sys.path.insert(0, str(GR00T))

import mujoco  # noqa: E402

from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig  # noqa: E402
from gear_sonic.utils.mujoco_sim.simulator_factory import SimulatorFactory, init_channel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--release-flag", default="/tmp/nvsim_release_band")
    ap.add_argument("--video", default="")
    ap.add_argument("--log", default="nvsim_log.npz")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scene", default="", help="override ROBOT_SCENE (repo-relative), e.g. a variant model")
    ap.add_argument("--no-hands", action="store_true", help="29-DoF model without Dex3 hands")
    args = ap.parse_args()
    args.log = str(Path(args.log).resolve())
    args.video = str(Path(args.video).resolve()) if args.video else ""
    Path(args.release_flag).unlink(missing_ok=True)

    import os
    os.chdir(GR00T)                                    # scene paths in the yaml are repo-relative
    cfg = SimLoopConfig().load_wbc_yaml()
    cfg["ENV_NAME"] = "default"
    if args.scene:
        cfg["ROBOT_SCENE"] = args.scene
    if args.no_hands:
        cfg["NUM_HAND_MOTORS"] = cfg["NUM_HAND_JOINTS"] = 0
        e = cfg["motor_effort_limit_list"]               # legs+waist 15, L arm 7, L hand 7, R arm 7, R hand 7
        cfg["motor_effort_limit_list"] = e[:22] + e[29:36]
    init_channel(config=cfg)
    sim = SimulatorFactory.create_simulator(config=cfg, env_name="default", onscreen=False,
                                            offscreen=False, enable_image_publish=False)
    env = sim.sim_env
    env.elastic_band.length = -0.2                     # viewer key 7 twice: feet on the ground
    mujoco.mj_forward(env.mj_model, env.mj_data)       # body poses valid before the band reads them
    threading.Thread(target=sim.start, daemon=True).start()

    m, d = env.mj_model, env.mj_data
    view = mujoco.MjData(m)
    r = mujoco.Renderer(m, 480, 640) if args.video else None
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = m.body("pelvis").id
    cam.distance, cam.azimuth, cam.elevation = 2.5, 120, -20
    frames, log, released_at = [], [], None
    t0 = time.monotonic()
    print("nvsim: running", flush=True)
    while (t := time.monotonic() - t0) < args.duration:
        if released_at is None and Path(args.release_flag).exists():
            env.elastic_band.enable = False
            released_at = t
            print(f"nvsim: band released at {t:.1f} s", flush=True)
        view.qpos[:] = d.qpos; view.qvel[:] = d.qvel
        mujoco.mj_forward(m, view)
        log.append(np.concatenate([[t, released_at is not None], view.qpos]))
        if r is not None:
            r.update_scene(view, cam)
            frames.append(r.render().copy())
        time.sleep(max(0.0, (len(log)) / args.fps - (time.monotonic() - t0)))
    sim._running = False
    names = [m.joint(i).name for i in range(m.njnt)]
    np.savez(args.log, log=np.array(log), released_at=-1 if released_at is None else released_at,
             joint_names=np.array(names), jnt_qposadr=m.jnt_qposadr.copy())
    if r is not None and frames:
        import imageio.v2 as imageio
        imageio.mimsave(args.video, frames, fps=args.fps, macro_block_size=8)
    print(f"nvsim: done, {len(log)} samples", flush=True)


if __name__ == "__main__":
    main()
