#!/usr/bin/env python3
"""The whole stack, end to end.

Language sets a coarse intent; a scripted or live operator overrides it on the
stick; the generator turns the arbitrated command into a reference motion; the
retargeter puts that on the robot; and — if a tracker checkpoint exists — the
policy executes it in physics.  Without a tracker it runs kinematically, which
is still the useful diagnostic: it shows what the operator asked for, so any
failure downstream is unambiguously the tracker's.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpc.control.arbiter import ArbiterConfig, ChannelArbiter
from rpc.control.command import Command
from rpc.control.input import GamepadInput, ScriptedInput
from rpc.control.language import LanguageChannel, MotionEntry
from rpc.generator.pfnn import MANN, PFNN
from rpc.generator.runtime import CharacterController
from rpc.motion.bvh import parse
from rpc.motion.clip import Skeleton
from rpc.motion.dataset import FeatureSpec, analyse
from rpc.motion.features import FeatureConfig
from rpc.sim.retarget import Retargeter, RetargetConfig
from rpc.sim.robot import REGISTRY


def load_generator(path: Path, device: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    man = ck["manifest"]
    cfg = FeatureConfig(fps=man["fps"], n_past=man["n_past"], n_future=man["n_future"],
                        span=man["span"], gaits=tuple(man["gaits"]))
    spec = FeatureSpec(n_joints=man["n_joints"], cfg=cfg)
    a = ck["args"]
    if ck["arch"] == "pfnn":
        model = PFNN(spec.n_in, spec.n_out, hidden=a["hidden"],
                     n_control=a["experts"], dropout=a["dropout"])
    else:
        model = MANN(spec.n_in, spec.n_out, gate_dim=len(ck["gate_slice"]),
                     hidden=a["hidden"], n_experts=a["experts"], dropout=a["dropout"])
    model.load_state_dict(ck["state"])
    skel = Skeleton(tuple(man["joint_names"]), np.array(man["parents"]),
                    np.array(man["offsets"]), np.array(man["fwd"]), np.array(man["up"]))
    return model, spec, skel, man


DEFAULT_SCRIPT = [
    (2.0, Command(move=np.zeros(2))),                                    # language drives
    (3.0, Command(move=np.array([1.6, 0.0]), responsiveness=0.7)),       # operator takes over
    (2.0, Command(move=np.array([1.2, 0.9]), turn=0.9, responsiveness=0.8)),
    (3.0, Command(move=np.zeros(2))),                                    # released; language resumes
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", default="runs/pfnn/generator.pt")
    ap.add_argument("--tracker", default="runs/tracker/tracker.pt")
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--say", default="walk forward")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--live", action="store_true", help="read a real gamepad/keyboard")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save", default="")
    args = ap.parse_args()

    model, spec, skel, man = load_generator(Path(args.generator), args.device)
    ctrl = CharacterController(model, spec, skel, device=args.device)

    seed_path = Path("data/raw/lafan1/walk1_subject1.bvh")
    if seed_path.exists():
        ctrl.reset(seed=analyse(parse(seed_path).resample(spec.cfg.fps), spec.cfg), frame=200)
    else:
        ctrl.reset()

    lib = [MotionEntry(e["name"], tuple(e["labels"]), np.array(e["gait"]))
           for e in man["library"]]
    lang = LanguageChannel().index(lib)
    lang_cmd, diag = lang.interpret(args.say)
    print(f'language: "{args.say}" -> {diag["gait"]}  speed {diag["speed"]:.2f} m/s '
          f'[{diag["backend"]}]')
    print(f'  top matches: {[m[0] for m in diag["matches"][:3]]}')

    device_in = GamepadInput() if args.live else ScriptedInput(DEFAULT_SCRIPT,
                                                               dt=1 / spec.cfg.fps)
    print(f"operator input: {device_in.mode}")
    arb = ChannelArbiter(ArbiterConfig(), dt=1 / spec.cfg.fps)

    robot = REGISTRY[args.robot]
    mj = mujoco.MjModel.from_xml_path(robot.path())
    rt = Retargeter(mj, robot, skel, RetargetConfig(iters=16))
    data = mujoco.MjData(mj)

    viewer = None
    if args.viewer:
        # NB: `import mujoco.viewer` here would make `mujoco` a local name for
        # the whole function and shadow the module-level import.
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(mj, data)

    n = int(args.seconds * spec.cfg.fps)
    dt = 1.0 / spec.cfg.fps
    log, qs = [], []
    t_gen = t_ik = 0.0

    for i in range(n):
        stick = device_in.poll()
        cmd = arb.update(lang_cmd, stick, args.say)

        t0 = time.perf_counter()
        frame = ctrl.step(cmd)
        t_gen += time.perf_counter() - t0

        t0 = time.perf_counter()
        q = rt.solve(frame.joint_pos_local, frame.root_yaw, frame.root_pos[:2])
        t_ik += time.perf_counter() - t0
        qs.append(q.copy())

        log.append({"t": i * dt, "authority": arb.authority,
                    "speed": float(np.linalg.norm(frame.root_delta[:2]) * spec.cfg.fps),
                    "phase": frame.phase, "contacts": frame.contacts.tolist(),
                    "root": frame.root_pos.tolist()})
        if viewer is not None:
            data.qpos[:] = q
            mujoco.mj_forward(mj, data)
            viewer.sync()
            time.sleep(dt)

    qs = np.stack(qs)
    print(f"\n{n} frames in {args.seconds:.0f} s of character time")
    print(f"  generator {t_gen/n*1e3:5.2f} ms/frame   retarget {t_ik/n*1e3:5.2f} ms/frame"
          f"   budget {dt*1e3:.1f} ms")
    print(f"  handovers: {[(round(h.t,2), h.direction) for h in arb.log]}")
    print(f"  operator held authority for "
          f"{sum(b-a for a,b in arb.intervention_segments()):.1f} s of {args.seconds:.0f} s")
    sp = np.array([l["speed"] for l in log])
    print(f"  commanded travel: {np.linalg.norm(np.array(log[-1]['root'])[:2]):.2f} m, "
          f"mean speed {sp.mean():.2f} m/s (peak {sp.max():.2f})")

    if Path(args.tracker).exists():
        print(f"  tracker checkpoint found at {args.tracker} — "
              f"run scripts/eval_tracker.py to execute this in physics")
    else:
        print("  no tracker checkpoint: this run is kinematic "
              "(train one with scripts/train_tracker.py)")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save, qpos=qs.astype(np.float32),
                            log=json.dumps(log), fps=spec.cfg.fps)
        print(f"  saved {args.save}")


if __name__ == "__main__":
    main()
