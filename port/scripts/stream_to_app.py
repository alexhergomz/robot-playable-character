"""Operator + generator side of the closed-loop test against NVIDIA's deploy app.

Publishes on tcp://*:5556 exactly what the robot setup will: protocol-v1
``pose`` messages from PoseStreamer.  The app runs with ``--input-type zmq``
(``zmq_manager`` refuses to run without SONIC's planner, which TensorRT 8.5
cannot load), so the operator acts through the app's keyboard: Enter toggles
the ZMQ stream, ``]`` starts control.  Here those keystrokes are written to
the app's stdin pipe (--keys); the streamer itself never starts the controller.

Script: stream standing -> start control (streamed mode) -> release the sim's
elastic band -> stand -> walk 0.3 m/s -> walk + turn -> stand -> brake -> stop
streaming (dead-man tail) -> hold.  Phase times are written for the analysis.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from rpc.control.command import Command  # noqa: E402
from rpc.deploy.streamer import PoseStreamer, StreamLimits  # noqa: E402
from rpc.sim import sonic as P  # noqa: E402
from rpc.sim import sonic_scene  # noqa: E402
from sim_stream import build  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(ROOT / "runs/deploy/bundle.pt"))
    ap.add_argument("--speed", type=float, default=0.3)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--replan-every", type=int, default=1)
    ap.add_argument("--release-flag", default="/tmp/nvsim_release_band")
    ap.add_argument("--phases", default="")
    ap.add_argument("--keys", required=True, help="FIFO connected to the app's stdin")
    ap.add_argument("--hold-phase", action="store_true", help="LMPCharacterController(hold_phase_when_idle=True)")
    ap.add_argument("--short-stand", action="store_true",
                    help="start walking ~1 s after the band is released (total ~2.5 s of standing)")
    args = ap.parse_args()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.bind("tcp://*:5556")
    mo = sonic_scene.load()
    ref = {}                                          # final reference per frame index (mujoco order)
    t_send = {}

    def send(msg):
        from rpc.deploy import wire
        _, d = wire.unpack(msg, "pose")
        inv = np.argsort(P.MUJOCO_TO_ISAACLAB)
        for i, f in enumerate(d["frame_index"].reshape(-1)):
            ref[int(f)] = d["joint_pos"][i][inv]
        t_send[int(d["frame_index"].reshape(-1)[0])] = time.monotonic() - t0
        sock.send(msg)

    t0 = time.monotonic()
    st = PoseStreamer(build(args.bundle, "cuda", hold_phase_when_idle=args.hold_phase), send,
                      joint_lower=mo.jnt_range[1:30, 0].copy(), joint_upper=mo.jnt_range[1:30, 1].copy(),
                      mujoco_to_isaaclab=P.MUJOCO_TO_ISAACLAB, horizon=args.horizon,
                      limits=StreamLimits(max_speed=args.speed), stand_pose=P.DEFAULT_ANGLES.copy(),
                      replan_every=args.replan_every)
    stand = Command(move=np.zeros(2), responsiveness=0.6)
    walk = Command(move=np.array([args.speed, 0.0]), responsiveness=0.6)
    turn = Command(move=np.array([args.speed, 0.0]), turn=0.4, responsiveness=0.6)
    log = []

    def mark(name):
        log.append((name, round(time.monotonic() - t0, 3))); print(f"[{log[-1][1]:6.2f}] {name}", flush=True)

    def stream(cmd, seconds, during=None):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            st.tick(cmd)
            if during: during()
            time.sleep(0.005)

    keys = open(args.keys, "w")

    def key(ch):
        keys.write(ch); keys.flush()

    time.sleep(1.5)                                    # let the app's subscribers connect
    pre = (0.5, 0.25, 0.75, 1.0) if args.short_stand else (2.0, 0.5, 3.0, 3.0)
    mark("stream standing"); stream(stand, pre[0])
    mark("operator: Enter (use ZMQ stream)"); key("\n")
    stream(stand, pre[1])
    mark("operator: ] (start control)"); key("]")
    stream(stand, pre[2])
    mark("operator: release band"); Path(args.release_flag).touch()
    stream(stand, pre[3])
    mark("walk"); stream(walk, 8.0)
    mark("walk + turn"); stream(turn, 6.0)
    mark("stand"); stream(stand, 3.0)
    mark("brake"); st.brake()
    mark("stream stops (dead-man tail)"); time.sleep(6.0)
    mark("end")
    if args.phases:
        idx = np.array(sorted(ref))
        np.savez(Path(args.phases).with_suffix(".ref.npz"), frame_index=idx,
                 joint_pos=np.stack([ref[i] for i in idx]),
                 send_first=np.array(sorted(t_send)), send_t=np.array([t_send[k] for k in sorted(t_send)]))
        Path(args.phases).write_text(json.dumps({"phases": log, "gen_ms": st.stats.gen_ms,
                                                 "sent": st.stats.sent, "clipped": st.stats.clipped_frames}))


if __name__ == "__main__":
    main()
