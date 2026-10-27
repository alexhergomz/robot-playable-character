"""Onboard launcher: Unitree remote -> generator -> PoseStreamer -> ZMQ -> NVIDIA's deploy app.

Runs on the robot's Orin next to ``g1_deploy_onnx_ref --input-type zmq``
(Python 3.8, numpy 1.17, no torch).  It never starts or stops the controller:
a person does that on the app (``]`` start, ``O`` emergency stop, Enter to use
the ZMQ stream).  It only publishes pose messages on localhost, and only while
the deadman button is held:

    held      -> stream the stick command
    released  -> stream zero velocity for ``brake_s``, then stop sending; the
                 last message's tail blends the reference into the stand pose and
                 the app holds it.  The generator keeps stepping (stand command,
                 phase hold on) so pressing again resumes without a catch-up burst.
    stale     -> remote silent for ``stale_s``: treated as released.

The remote is read from ``rt/lowstate`` (listen only).

Usage on the robot:
    python3 -m rpc.deploy.launcher BUNDLE_DIR --backend trt --iface eth0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ..control.command import Command
from . import onboard
from .remote import RemoteState, StickMap, parse, to_command
from .streamer import PoseStreamer, StreamLimits
from ..sim import sonic as P

HZ = 50.0
STAND = Command(move=np.zeros(2), responsiveness=0.6)


class Session:
    """The launcher's state machine, independent of DDS/ZMQ so it can be tested."""

    def __init__(self, streamer: PoseStreamer, sticks: StickMap, stale_s: float = 0.25,
                 clock=time.monotonic):
        self.st = streamer
        self.sticks = sticks
        self.stale_s = stale_s
        self.clock = clock
        self.state = "idle"               # idle | streaming | braking
        self._brake_until = 0.0
        self.last_cmd: Command | None = None

    def step(self, remote: RemoteState | None, remote_t: float | None) -> str:
        now = self.clock()
        fresh = remote is not None and remote_t is not None and now - remote_t <= self.stale_s
        cmd = to_command(remote, self.sticks) if fresh else None
        if cmd is not None:
            self.state = "streaming"
            self.st.tick(cmd)
        elif self.state == "streaming":
            self.state, self._brake_until = "braking", now + self.st.limits.brake_s
        if self.state == "braking":
            if now < self._brake_until:
                self.st.tick(STAND)
            else:
                self.state = "idle"
        if self.state == "idle":
            self.st.advance(STAND)
        self.last_cmd = cmd
        return self.state


class DDSRemote:
    """Latest ``wireless_remote`` from rt/lowstate.  Subscribes only."""

    def __init__(self, iface: str):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
        ChannelFactoryInitialize(0, iface)
        self.state, self.t = None, None
        self._sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._sub.Init(self._on, 1)

    def _on(self, msg):
        try:
            self.state, self.t = parse(msg.wireless_remote), time.monotonic()
        except ValueError:
            pass                              # malformed sample: keep the previous one, let it go stale

    def read(self):
        return self.state, self.t


class ScriptedRemote:
    """For tests and simulation: [[t, lx, ly, rx, ["R1", ...]], ...] from a JSON file."""

    def __init__(self, path: str, clock=time.monotonic):
        self.events = json.loads(Path(path).read_text())
        self.clock, self.t0 = clock, clock()

    def read(self):
        t = self.clock() - self.t0
        cur = None
        for ev in self.events:
            if ev[0] <= t:
                cur = ev
        if cur is None:
            return None, None
        return RemoteState(lx=cur[1], ly=cur[2], rx=cur[3], buttons=frozenset(cur[4])), self.clock()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle")
    ap.add_argument("--backend", default="trt", choices=("trt", "numpy", "ort"))
    ap.add_argument("--engine", default=None)
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--remote", default="dds", help="'dds' or 'script:FILE.json'")
    ap.add_argument("--bind", default="tcp://127.0.0.1:5556")
    ap.add_argument("--max-speed", type=float, default=0.3)
    ap.add_argument("--max-turn", type=float, default=0.5)
    ap.add_argument("--deadman", default="R1")
    # 16/3 fits the Orin at its default GPU clock; 12/1 needs the GPU clock raised.
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--replan-every", type=int, default=3)
    ap.add_argument("--duration", type=float, default=0.0, help="stop after N s (0 = until Ctrl-C)")
    args = ap.parse_args()

    import zmq
    init = np.load(Path(args.bundle) / "init.npz")
    gen = onboard.load(args.bundle, backend=args.backend, engine=args.engine, hold_phase_when_idle=True)
    sock = zmq.Context().socket(zmq.PUB)
    sock.bind(args.bind)
    limits = StreamLimits(max_speed=args.max_speed, max_turn=args.max_turn)
    st = PoseStreamer(gen, sock.send, joint_lower=init["joint_lower"], joint_upper=init["joint_upper"],
                      mujoco_to_isaaclab=P.MUJOCO_TO_ISAACLAB, horizon=args.horizon,
                      limits=limits, stand_pose=init["stand_pose"].copy(),
                      replan_every=args.replan_every)
    remote = DDSRemote(args.iface) if args.remote == "dds" else ScriptedRemote(args.remote.split(":", 1)[1])
    session = Session(st, StickMap(max_speed=args.max_speed, max_turn=args.max_turn, deadman=args.deadman))
    print(f"launcher: streaming on {args.bind}; hold {args.deadman} to move", flush=True)

    t0 = next_t = time.monotonic()
    last_print, prev_state = t0, None
    try:
        while not args.duration or time.monotonic() - t0 < args.duration:
            state = session.step(*remote.read())
            now = time.monotonic()
            if state != prev_state or now - last_print > 2.0:
                c = session.last_cmd
                ms = np.mean(st.stats.gen_ms[-25:]) if st.stats.gen_ms else 0.0
                print(f"[{now - t0:7.1f}] {state:9s} cmd {'-' if c is None else np.round([*c.move, c.turn], 2)} "
                      f"gen {ms:5.1f} ms sent {st.stats.sent}", flush=True)
                last_print, prev_state = now, state
            next_t += 1.0 / HZ
            time.sleep(max(0.0, next_t - time.monotonic()))
    except KeyboardInterrupt:
        pass
    finally:
        print("launcher: stopped sending; the app holds the stand pose", flush=True)


if __name__ == "__main__":
    main()
