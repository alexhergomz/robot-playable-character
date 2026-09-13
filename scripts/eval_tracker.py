#!/usr/bin/env python3
"""Execute a reference in physics with a trained tracker.

Two modes, one code path.  ``--clip`` replays a retargeted mocap clip, which is
the training distribution and measures tracking quality.  ``--live`` runs the
generator under an operator, which is the deployment condition and measures
whether the split architecture actually holds up: the tracker has never seen the
generator's output, only retargeted mocap.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpc.control.arbiter import ArbiterConfig, ChannelArbiter
from rpc.control.command import Command
from rpc.control.input import GamepadInput, ScriptedInput
from rpc.control.language import LanguageChannel, MotionEntry
from rpc.policy.ppo import ActorCritic
from rpc.policy.reference import ClipReference, LiveReference
from rpc.policy.tracker import TrackerConfig, TrackerEnv
from rpc.sim.retarget import RetargetConfig, Retargeter
from rpc.sim.robot import REGISTRY


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", default="runs/tracker/tracker.pt")
    ap.add_argument("--clip", default="", help="a .npz from retarget_clips.py")
    ap.add_argument("--live", action="store_true", help="drive the generator instead")
    ap.add_argument("--generator", default="runs/pfnn2/generator.pt")
    ap.add_argument("--say", default="walk forward")
    ap.add_argument("--gamepad", action="store_true")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    ck = torch.load(args.tracker, map_location="cpu", weights_only=False)
    spec = REGISTRY[ck["robot"]]
    cfg = TrackerConfig(episode_len=args.steps)

    if args.live:
        from scripts.demo_rpc import DEFAULT_SCRIPT, load_generator
        from rpc.generator.runtime import CharacterController
        import mujoco

        model, gspec, skel, man = load_generator(Path(args.generator), "cpu")
        ctrl = CharacterController(model, gspec, skel)
        mj = mujoco.MjModel.from_xml_path(spec.path())
        rt = Retargeter(mj, spec, skel, RetargetConfig(iters=16))
        lib = [MotionEntry(e["name"], tuple(e["labels"]), np.array(e["gait"]))
               for e in man["library"]]
        lang_cmd, _ = LanguageChannel().index(lib).interpret(args.say)
        dev_in = GamepadInput() if args.gamepad else ScriptedInput(
            DEFAULT_SCRIPT, dt=1 / gspec.cfg.fps)
        ref = LiveReference(ctrl, rt, dev_in, cfg.control_hz,
                            ChannelArbiter(ArbiterConfig(), dt=1 / gspec.cfg.fps), lang_cmd)
        label = f'live generator, "{args.say}"'
    else:
        path = Path(args.clip) if args.clip else sorted(
            Path("data/processed/retargeted") .glob("*/*.npz"))[0]
        z = np.load(path)
        ref = ClipReference(z["qpos"], z["contacts"], z["phase"], float(z["fps"]),
                            cfg.control_hz, args.steps)
        label = f"clip {path.stem}"

    env = TrackerEnv(spec, ref, cfg)
    ac = ActorCritic(ck["n_obs"], ck["n_act"])
    ac.load_state_dict(ck["state"])
    ac.eval()

    viewer = None
    if args.viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(env.sim.model, env.sim.data)

    print(f"tracker: {args.tracker}   reference: {label}")
    rows = []
    for ep in range(args.episodes):
        obs = env.reset(seed=ep)
        ret, errs, n, fell = 0.0, [], 0, False
        for _ in range(args.steps):
            with torch.no_grad():
                a, _, _ = ac.act(torch.as_tensor(obs, dtype=torch.float32)[None],
                                 deterministic=True)
            obs, r, term, trunc, info = env.step(a[0].numpy())
            ret += r
            errs.append(info["errors"])
            n += 1
            if viewer is not None:
                viewer.sync()
            if term or trunc:
                fell = term
                break
        m = lambda k: float(np.mean([e[k] for e in errs]))
        rows.append((n, ret, m("joint"), m("key_body"), m("root"), m("contact"), fell))
        print(f"  ep {ep}: {n:4d}/{args.steps} steps  return {ret:8.1f}  "
              f"joint {m('joint'):.3f} rad  key-body {m('key_body')*100:5.1f} cm  "
              f"root {m('root')*100:5.1f} cm  contact-mismatch {m('contact'):.2f}"
              f"{'  FELL' if fell else ''}")

    a = np.array([[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows])
    print(f"\nmean over {len(rows)} episodes: survived {a[:,0].mean():.0f}/{args.steps} steps, "
          f"return {a[:,1].mean():.1f}, joint err {a[:,2].mean():.3f} rad, "
          f"key-body {a[:,3].mean()*100:.1f} cm, contact mismatch {a[:,5].mean():.2f}")
    print(f"fall rate {sum(r[6] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
