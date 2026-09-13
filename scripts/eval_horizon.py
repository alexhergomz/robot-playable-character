#!/usr/bin/env python3
"""Does the reference horizon have to be regenerated every control step?

The runtime currently rebuilds all ten lookahead frames from the live command
at every step, so a changed command reaches the whole horizon at once.  That
costs eleven generator forward passes per step (52--60 ms against the 20 ms a
50 Hz loop allows) and is the binding constraint on the whole system.

The alternative is what a conventional pipeline does: keep the horizon in a
buffer, generate one new frame per step, drop the consumed one.  One forward
pass instead of eleven -- but the tail of the horizon was produced under the
command in force up to ten frames (200 ms) ago.

Whether that matters is an empirical question, and the measurement in
``eval_interrupt`` already answers it: the tracker itself lags 400--500 ms on
speed changes, so 200 ms of stale tail may be invisible behind it.  This script
runs the same manoeuvres under both policies.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rpc.control.command import Command
from rpc.generator.lmp import LMPGenerator
from rpc.generator.lmp_runtime import LMPCharacterController
from rpc.motion import quat as Q
from rpc.motion.features import FeatureConfig
from rpc.motion.robot_dataset import RobotFeatureSpec
from rpc.sim import sonic as P
from rpc.sim.robot import REGISTRY
from rpc.sim.sonic_policy import (CONTROL_HZ, SonicPolicy, apply_sonic_armature,
                                  apply_sonic_gains)

HZ = CONTROL_HZ


class Regenerated:
    """Rebuild the whole horizon from the live command every step."""
    name = "regenerated"

    def __init__(self, gen, n):
        self.gen, self.n = gen, n

    def horizon(self, cmd):
        snap = copy.deepcopy(self.gen)
        prev = self.gen.joint_pos.copy()
        dof, quats = [], []
        for _ in range(self.n):
            f = snap.step(cmd)
            dof.append(f.joint_pos.copy())
            quats.append(Q.yaw_quat(np.array(f.root_yaw)))
        dof = np.stack(dof)
        dvel = np.diff(dof, axis=0, prepend=prev[None]) * HZ
        frame = self.gen.step(cmd)
        return dof, dvel, np.stack(quats), frame


class Buffered:
    """Keep the horizon; generate one new frame per step, drop the consumed one."""
    name = "buffered"

    def __init__(self, gen, n):
        self.gen, self.n = gen, n
        self.buf = deque()
        self.prev = gen.joint_pos.copy()

    def _emit(self, cmd):
        f = self.gen.step(cmd)
        return (f.joint_pos.copy(), Q.yaw_quat(np.array(f.root_yaw)), f)

    def horizon(self, cmd):
        while len(self.buf) < self.n + 1:          # prime on the first call
            self.buf.append(self._emit(cmd))
        dof = np.stack([b[0] for b in list(self.buf)[:self.n]])
        quats = np.stack([b[1] for b in list(self.buf)[:self.n]])
        dvel = np.diff(dof, axis=0, prepend=self.prev[None]) * HZ
        consumed = self.buf.popleft()
        self.prev = consumed[0]
        self.buf.append(self._emit(cmd))           # one forward pass per step
        return dof, dvel, quats, consumed[2]


class Periodic:
    """Regenerate the whole horizon every ``m`` steps; shift the buffer between.

    A partial regeneration is not well defined: a head rebuilt from the live
    command and a tail carried over from the buffer come from generator states
    that have diverged, so splicing them produces an incoherent horizon (it
    falls).  Regenerating the whole horizon on a slower cadence keeps it
    coherent and still amortises the cost, at the price of a command waiting up
    to ``m`` steps to be seen at all.
    """

    def __init__(self, gen, n, m=3):
        self.gen, self.n, self.m = gen, n, m
        self.k = 0
        self.buf = deque()
        self.prev = gen.joint_pos.copy()

    def horizon(self, cmd):
        if self.k % self.m == 0 or not self.buf:
            snap = copy.deepcopy(self.gen)
            self.prev = self.gen.joint_pos.copy()
            self.buf = deque()
            for _ in range(self.n):
                f = snap.step(cmd)
                self.buf.append((f.joint_pos.copy(), Q.yaw_quat(np.array(f.root_yaw))))
        self.k += 1
        items = list(self.buf)
        dof = np.stack([b[0] for b in items])
        quats = np.stack([b[1] for b in items])
        dvel = np.diff(dof, axis=0, prepend=self.prev[None]) * HZ
        frame = self.gen.step(cmd)                 # advance the real reference
        self.buf.popleft()
        self.buf.append((frame.joint_pos.copy(), Q.yaw_quat(np.array(frame.root_yaw))))
        self.prev = dof[0]
        return dof, dvel, quats, frame


def build(gen_path):
    ck = torch.load(gen_path, map_location="cpu", weights_only=False)
    man, a = ck["manifest"], ck["args"]
    C = man["phase_channels"]
    cfg = FeatureConfig(fps=man["fps"], n_past=man["n_past"], n_future=man["n_future"],
                        span=man["span"], gaits=tuple(man["gaits"]))
    spec = RobotFeatureSpec(man["n_joints"], cfg, man["n_feet"])
    net = LMPGenerator(ck["n_in"], ck["n_out"], gate_dim=4 * C, hidden=a["hidden"],
                       n_experts=a["experts"], dropout=a["dropout"])
    net.load_state_dict(ck["state"])
    return net, spec, ck, man, C


def rollout(policy_cls, gen_path, script, init_frame=60007):
    net, spec, ck, man, C = build(gen_path)
    M = np.load(ROOT / "data/processed/robot_lmp/train_M.npy", mmap_mode="r")
    F = np.load(ROOT / "data/processed/robot_lmp/train_F.npy", mmap_mode="r")
    mo = mujoco.MjModel.from_xml_path(REGISTRY["g1"].path())
    apply_sonic_gains(mo); apply_sonic_armature(mo)
    d = mujoco.MjData(mo)
    pol = SonicPolicy(root=str(ROOT / "assets/sonic"), variant="low_latency")
    sub = max(1, int(round(1.0 / (HZ * mo.opt.timestep))))
    g = LMPCharacterController(net, spec, ck["out_slices"], 4 * C, n_feet=man["n_feet"],
                               keep_cols=ck.get("keep_cols"), curve=ck.get("amplitude_curve"),
                               device=device)
    mujoco.mj_resetDataKeyframe(mo, d, 0)
    d.qpos[7:] = P.DEFAULT_ANGLES; d.qpos[2] = 0.76
    mujoco.mj_forward(mo, d)
    g.reset(joint_pos=P.DEFAULT_ANGLES.copy(), root_height=0.76,
            manifold=M[init_frame].copy(), foot=F[init_frame].copy())

    def grav():
        o = np.zeros(3)
        mujoco.mju_rotVecQuat(o, np.array([0.0, 0.0, -1.0]),
                              np.array([d.qpos[3], -d.qpos[4], -d.qpos[5], -d.qpos[6]]))
        return o

    pol.reset(d.qpos[7:].copy(), d.qvel[6:].copy(), d.qvel[3:6].copy(), grav())
    hp = policy_cls(g, pol.lookahead_frames)
    idx = np.arange(pol.lookahead_frames) * pol.step
    log, ms = [], []
    for cmd in script:
        c = cmd.clamped()
        t0 = time.perf_counter()
        dof, dvel, quats, _ = hp.horizon(c)
        ms.append((time.perf_counter() - t0) * 1e3)
        tg = pol.step_policy(dof[idx][:, P.MUJOCO_TO_ISAACLAB],
                             dvel[idx][:, P.MUJOCO_TO_ISAACLAB], quats[idx],
                             d.qpos[3:7].copy(), d.qpos[7:].copy(),
                             d.qvel[6:].copy(), d.qvel[3:6].copy(), grav())
        d.ctrl[:] = np.clip(tg, mo.actuator_ctrlrange[:, 0], mo.actuator_ctrlrange[:, 1])
        for _ in range(sub):
            mujoco.mj_step(mo, d)
        log.append(d.qpos.copy())
        if d.qpos[2] < 0.45:
            break
    return np.stack(log), np.array(ms)


def latency(q, tc, channel="speed"):
    if channel == "yaw":
        w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
        yaw = np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        v = np.diff(yaw) * HZ
    else:
        v = np.linalg.norm(np.diff(q[:, :2], axis=0), axis=-1) * HZ
    v = np.convolve(v, np.ones(9) / 9, "same")
    base = v[max(0, tc - 25):tc].mean()
    tgt = v[tc + int(3 * HZ) - 25:tc + int(3 * HZ)].mean()
    span = tgt - base
    if abs(span) < 0.05:
        return None, None
    r = s = None
    for i in range(tc, min(len(v), tc + int(4 * HZ))):
        p = (v[i] - base) / span
        if r is None and p > 0.10:
            r = (i - tc) / HZ * 1000
        if p > 0.90:
            s = (i - tc) / HZ * 1000
            break
    return r, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="runs/lmp_nogait/generator.pt")
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda; the generator dominates the step budget")
    args = ap.parse_args()
    n = int(5 * HZ)
    S = lambda **k: Command(responsiveness=0.8, **k)
    tests = {
        "stop 1.6 -> 0": ([S(move=np.array([1.6, 0.]))] * n + [S(move=np.zeros(2))] * n, "speed"),
        "accel 0.4 -> 1.6": ([S(move=np.array([0.4, 0.]))] * n
                             + [S(move=np.array([1.6, 0.]))] * n, "speed"),
        "turn": ([S(move=np.array([1.2, 0.]))] * n
                 + [S(move=np.array([1.2, 0.]), turn=1.5)] * n, "yaw"),
    }
    import functools
    policies = [("buffered", Buffered)]
    for m in (5, 3, 2):
        policies.append((f"regen every {m}", functools.partial(Periodic, m=m)))
    policies.append(("regen every step", Regenerated))
    print(f"{'manoeuvre':18s} {'horizon policy':15s} {'react':>7s} {'settle':>8s} {'gen ms/step':>12s}")
    for name, (script, ch) in tests.items():
        for label, cls in policies:
            q, ms = rollout(cls, args.gen, script)
            r, s = latency(q, n, ch)
            f = lambda x: f"{x:5.0f}" if x is not None else "  n/a"
            print(f"{name:18s} {label:15s} {f(r)} ms {f(s)} ms {np.mean(ms):9.1f} ms"
                  + ("   FELL" if len(q) < len(script) else ""), flush=True)
        print()
    print(f"\nbudget at {HZ:.0f} Hz: {1000/HZ:.0f} ms per control step")


if __name__ == "__main__":
    main()
