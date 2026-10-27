"""Time the onboard generator + streamer loop.  Sends nothing anywhere: the sink only counts bytes.

Runs the same stand / walk / turn / brake command script on a virtual 50 Hz
clock and reports the wall time of every PoseStreamer.tick against the 20 ms
budget.  The streamed joint targets are saved so the robot's output can be
compared with the laptop's.

Usage (robot, Python 3.8):  python3 bench_onboard.py BUNDLE_DIR --backend trt|numpy --out result.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # laptop: repo root

from rpc.control.command import Command  # noqa: E402
from rpc.deploy import onboard, wire  # noqa: E402
from rpc.deploy.streamer import PoseStreamer, StreamLimits  # noqa: E402
from rpc.sim import sonic as P  # noqa: E402

HZ = 50.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--backend", default="trt")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--replan-every", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--out", default="bench_result.npz")
    args = ap.parse_args()

    init = np.load(Path(args.bundle) / "init.npz")
    t0 = time.perf_counter()
    gen = onboard.load(args.bundle, backend=args.backend, engine=args.engine, graph=not args.no_graph)
    print(f"load {args.backend}: {time.perf_counter() - t0:.2f} s")

    sent = []
    def sink(msg):                       # records instead of sending
        sent.append(wire.unpack(msg, "pose")[1])
    clock = [0.0]
    st = PoseStreamer(gen, sink, joint_lower=init["joint_lower"], joint_upper=init["joint_upper"],
                      mujoco_to_isaaclab=P.MUJOCO_TO_ISAACLAB, horizon=args.horizon,
                      limits=StreamLimits(max_speed=0.3), stand_pose=init["stand_pose"].copy(),
                      clock=lambda: clock[0], replan_every=args.replan_every)
    plan = ([Command(move=np.zeros(2), responsiveness=0.6)] * 100
            + [Command(move=np.array([0.3, 0.0]), responsiveness=0.6)] * 200
            + [Command(move=np.array([0.3, 0.0]), turn=0.5, responsiveness=0.6)] * 150
            + [Command(move=np.zeros(2), responsiveness=0.6)] * 100)
    dt = []
    for cmd in plan:
        t = time.perf_counter()
        st.tick(cmd)
        dt.append(time.perf_counter() - t)
        clock[0] += 1.0 / HZ
    dt = np.array(dt[5:]) * 1e3             # drop warm-up
    print(f"tick ms over {len(dt)} ticks: mean {dt.mean():.2f}  p50 {np.median(dt):.2f}  "
          f"p99 {np.percentile(dt, 99):.2f}  max {dt.max():.2f}  over 20 ms: {(dt > 20).sum()}")
    # Re-planning every N frames is only real-time if each re-plan finishes
    # before the receiver runs out of generated look-ahead: horizon - N - 10 frames of slack.
    # A message sent at frame k holds generated frames k+1..k+horizon; the next re-plan
    # starts at k+N and must land before frame k+horizon-10+1, when SONIC's 10-frame
    # look-ahead would reach past them into the stand-pose tail.
    slack_ms = (args.horizon - 10 - args.replan_every + 1) * 1e3 / HZ
    print(f"average generator load per 20 ms frame: {dt.sum() / len(dt):.2f} ms; "
          f"worst tick {dt.max():.1f} ms vs slack before the tracker starves ~{slack_ms:.0f} ms")
    last = {}
    for m in sent:                          # newest value of every frame index
        for i, f in enumerate(m["frame_index"].reshape(-1)):
            last[int(f)] = m["joint_pos"][i]
    idx = np.array(sorted(last)); jp = np.stack([last[i] for i in idx])
    np.savez(args.out, tick_ms=dt, frame_index=idx, joint_pos=jp)
    print(f"messages {len(sent)}, frames {len(idx)}, saved {args.out}")


if __name__ == "__main__":
    main()
