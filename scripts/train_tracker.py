#!/usr/bin/env python3
"""Train the physics tracking policy on retargeted clips."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpc.policy.ppo import PPO, PPOConfig, VecEnv
from rpc.policy.reference import ClipLibraryReference, ClipReference
from rpc.policy.tracker import TrackerConfig, TrackerEnv
from rpc.sim.robot import REGISTRY


def load_clips(root: Path, control_hz: float, episode_len: int, limit: int = 0):
    files = sorted(root.glob("*.npz"))
    if limit:
        files = files[:limit]
    assert files, f"no retargeted clips in {root} — run scripts/retarget_clips.py"
    out = []
    for f in files:
        z = np.load(f)
        out.append(ClipReference(z["qpos"], z["contacts"], z["phase"],
                                 float(z["fps"]), control_hz, episode_len))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/processed/retargeted/g1")
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--out", default="runs/tracker")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--episode-len", type=int, default=400)
    ap.add_argument("--push", type=float, default=0.0)
    ap.add_argument("--limit-clips", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    spec = REGISTRY[args.robot]
    cfg = TrackerConfig(episode_len=args.episode_len, push_impulse=args.push)
    clips = load_clips(Path(args.clips), cfg.control_hz, args.episode_len, args.limit_clips)
    print(f"{len(clips)} reference clips, {sum(len(c.qpos_src) for c in clips):,} frames")

    def make(i):
        return TrackerEnv(spec, ClipLibraryReference([
            ClipReference(c.qpos_src, c.contacts_src, c.phase_src, c.fps,
                          cfg.control_hz, args.episode_len) for c in clips]),
            cfg, seed=i)

    venv = VecEnv(make, args.envs)
    n_obs, n_act = venv.envs[0].n_obs, venv.envs[0].nu
    agent = PPO(venv, n_obs, n_act, PPOConfig(), device=args.device)
    print(f"obs {n_obs}  act {n_act}  envs {args.envs}  device {args.device}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    per_iter = agent.cfg.rollout * args.envs
    iters = max(1, args.steps // per_iter)
    hist, t0, best = [], time.time(), -1e9

    for it in range(iters):
        batch = agent.collect()
        stats = agent.update(*batch)
        if agent.recent:
            rets = np.array([r for r, _ in agent.recent[-60:]])
            lens = np.array([l for _, l in agent.recent[-60:]])
            rec = {"iter": it, "steps": (it + 1) * per_iter,
                   "return": float(rets.mean()), "ep_len": float(lens.mean()),
                   "sps": (it + 1) * per_iter / (time.time() - t0), **stats}
            hist.append(rec)
            if it % 5 == 0 or it == iters - 1:
                print(f"it {it:4d}  steps {rec['steps']:>9,}  return {rec['return']:8.1f}  "
                      f"ep_len {rec['ep_len']:6.1f}  kl {stats['kl']:+.4f}  "
                      f"{rec['sps']:.0f} sps", flush=True)
            if rec["return"] > best:
                best = rec["return"]
                torch.save({"state": agent.ac.state_dict(), "n_obs": n_obs, "n_act": n_act,
                            "robot": args.robot, "args": vars(args)}, out / "tracker.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    print(f"best mean return {best:.1f} -> {out/'tracker.pt'}")


if __name__ == "__main__":
    main()
