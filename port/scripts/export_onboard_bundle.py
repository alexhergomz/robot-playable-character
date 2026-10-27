"""Split runs/deploy/bundle.pt into the torch-free onboard bundle read by rpc.deploy.onboard.load.

Writes meta.json, init.npz, weights.npz and generator.onnx, then checks that
the numpy backend driven through LMPCharacterController reproduces the torch
controller frame for frame over a stand / walk / turn / brake command script.

Usage: python port/scripts/export_onboard_bundle.py [--bundle runs/deploy/bundle.pt] [--out port/results/onboard_bundle]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(Path(__file__).parent))

from export_generator_onnx import ExportStep  # noqa: E402
from rpc.control.command import Command  # noqa: E402
from rpc.deploy import onboard  # noqa: E402
from rpc.sim import sonic as P  # noqa: E402
from rpc.sim import sonic_scene  # noqa: E402
from sim_stream import build  # noqa: E402

ROOT_HEIGHT = 0.76     # same reset as scripts/sim_stream.build


def script():
    return ([Command(move=np.zeros(2), responsiveness=0.6)] * 100
            + [Command(move=np.array([0.6, 0.0]), responsiveness=0.6)] * 150
            + [Command(move=np.array([0.4, 0.0]), turn=0.5, responsiveness=0.6)] * 150
            + [Command(move=np.zeros(2), responsiveness=0.6)] * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(ROOT / "runs/deploy/bundle.pt"))
    ap.add_argument("--out", default=str(ROOT / "port/results/onboard_bundle"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.bundle, map_location="cpu", weights_only=False)
    m = ck["manifest"]
    meta = {"manifest": {k: m[k] for k in ("fps", "n_past", "n_future", "span", "gaits",
                                            "n_joints", "n_feet", "phase_channels")},
            "out_slices": {k: list(v) for k, v in ck["out_slices"].items()},
            "n_in": ck["n_in"], "n_out": ck["n_out"], "root_height": ROOT_HEIGHT,
            "provenance": {k: str(v) for k, v in ck.get("provenance", {}).items()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    curve = ck["amplitude_curve"]
    np.savez(out / "init.npz", keep_cols=np.asarray(ck["keep_cols"], bool),
             curve_speed=np.asarray(curve["speed"]), curve_amplitude=np.asarray(curve["amplitude"]),
             init_manifold=np.asarray(ck["init_manifold"]), init_foot=np.asarray(ck["init_foot"]),
             stand_pose=P.DEFAULT_ANGLES.copy(),
             # streamer clips to these; the robot has no MuJoCo to read them from
             joint_lower=sonic_scene.load().jnt_range[1:30, 0].copy(),
             joint_upper=sonic_scene.load().jnt_range[1:30, 1].copy())
    np.savez(out / "weights.npz", **{k: v.numpy() for k, v in ck["state"].items()})

    g_torch = build(args.bundle, "cpu")
    exp = ExportStep(g_torch.model.eval()).eval()
    x0 = torch.zeros(1, ck["n_in"]); g0 = torch.zeros(1, 4 * m["phase_channels"])
    torch.onnx.export(exp, (x0, g0), str(out / "generator.onnx"), input_names=["x", "gate_x"],
                      output_names=["y"], opset_version=13, do_constant_folding=True, dynamo=False)

    # The torch-free controller must reproduce the torch one, frame for frame.
    g_np = onboard.load(out, backend="numpy")
    worst = 0.0
    for cmd in script():
        a, b = g_torch.step(cmd), g_np.step(cmd)
        worst = max(worst, float(np.abs(a.joint_pos - b.joint_pos).max()),
                    float(np.abs(a.root_xy - b.root_xy).max()), abs(a.root_yaw - b.root_yaw))
    print(f"numpy controller vs torch controller over {len(script())} steps: max |diff| {worst:.2e}")
    assert worst < 1e-3, "torch-free controller diverges from the torch one"
    for f in sorted(out.iterdir()):
        print(f"  {f.name:16s} {f.stat().st_size / 1e6:7.1f} MB")


if __name__ == "__main__":
    main()
