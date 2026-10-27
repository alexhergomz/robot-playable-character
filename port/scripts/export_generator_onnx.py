"""Export the deploy generator's network to ONNX for TensorRT 8.5 on the Orin.

Only the network moves to ONNX; the rest of ``LMPCharacterController.step``
is numpy and stays on the CPU.  The exported graph is ``LMPGenerator.step``:
un-normalised input and gate in, un-normalised output out.

``BlendedLinear`` uses einsum, which TensorRT 8.5 supports only partially, so
the export wrapper re-expresses the same sum as two matmuls.  The wrapper is
checked against the original module on real inputs before anything is written.

Usage: python port/scripts/export_generator_onnx.py [--bundle runs/deploy/bundle.pt] [--out port/results/generator]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))

from rpc.control.command import Command  # noqa: E402
from sim_stream import build  # noqa: E402


class _Blended(nn.Module):
    """``BlendedLinear`` for batch 1 without einsum: y = c @ (W x + b)."""

    def __init__(self, layer):
        super().__init__()
        K, O, I = layer.weight.shape
        self.K, self.O = K, O
        self.register_buffer("w", layer.weight.detach().reshape(K * O, I).t().contiguous())  # (I, K*O)
        self.register_buffer("b", layer.bias.detach().reshape(1, K * O))

    def forward(self, x, coeff):                       # x (1, I), coeff (1, K)
        per_expert = (x @ self.w + self.b).reshape(self.K, self.O)
        return coeff @ per_expert                      # (1, O)


class ExportStep(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.gate = net.gate
        self.l0, self.l1, self.l2 = _Blended(net.l0), _Blended(net.l1), _Blended(net.l2)
        self.act = nn.ELU()
        for k in ("x_mean", "x_std", "y_mean", "y_std"):
            self.register_buffer(k, getattr(net, k).detach().clone())

    def forward(self, x, gate_x):
        xn = (x - self.x_mean) / self.x_std
        c = torch.softmax(self.gate(gate_x), dim=-1)
        h = self.act(self.l0(xn, c))
        h = self.act(self.l1(h, c))
        return self.l2(h, c) * self.y_std + self.y_mean


def record_inputs(g, n_per_phase=150):
    """Real (x, gate) pairs from the controller: stand, walk, turn, brake."""
    rec = []
    orig = g.model.step
    def step(x, gate):
        rec.append((x.cpu().numpy().copy(), gate.cpu().numpy().copy()))
        return orig(x, gate)
    g.model.step = step
    plan = [Command(move=np.zeros(2), responsiveness=0.6),
            Command(move=np.array([0.6, 0.0]), responsiveness=0.6),
            Command(move=np.array([0.4, 0.0]), turn=0.5, responsiveness=0.6),
            Command(move=np.zeros(2), responsiveness=0.6)]
    for cmd in plan:
        for _ in range(n_per_phase):
            g.step(cmd)
    g.model.step = orig
    return (np.concatenate([r[0] for r in rec]).astype(np.float32),
            np.concatenate([r[1] for r in rec]).astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(ROOT / "runs/deploy/bundle.pt"))
    ap.add_argument("--out", default=str(ROOT / "port/results/generator"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    g = build(args.bundle, "cpu")
    net = g.model.eval()
    X, G = record_inputs(g)
    print(f"recorded {len(X)} real steps: x {X.shape[1]}, gate {G.shape[1]}")

    exp = ExportStep(net).eval()
    with torch.no_grad():
        ref = np.concatenate([net.step(torch.from_numpy(X[i:i + 1]), torch.from_numpy(G[i:i + 1])).numpy()
                              for i in range(len(X))])
        wrap = np.concatenate([exp(torch.from_numpy(X[i:i + 1]), torch.from_numpy(G[i:i + 1])).numpy()
                               for i in range(len(X))])
    err = np.abs(wrap - ref).max()
    print(f"wrapper vs original module: max |err| {err:.2e} (output scale {np.abs(ref).mean():.3f})")
    assert err < 1e-4, "export wrapper does not reproduce the module"

    onnx_path = out / "generator.onnx"
    torch.onnx.export(exp, (torch.from_numpy(X[:1]), torch.from_numpy(G[:1])), str(onnx_path),
                      input_names=["x", "gate_x"], output_names=["y"], opset_version=13,
                      do_constant_folding=True, dynamo=False)

    import onnxruntime as ort
    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ox = np.concatenate([s.run(None, {"x": X[i:i + 1], "gate_x": G[i:i + 1]})[0] for i in range(len(X))])
    print(f"ONNX Runtime vs original module: max |err| {np.abs(ox - ref).max():.2e}")
    ops = sorted({n.op_type for n in __import__("onnx").load(str(onnx_path)).graph.node})
    print("ops:", ops)

    idx = np.linspace(0, len(X) - 1, 64).astype(int)
    np.save(out / "x.npy", X[idx]); np.save(out / "gate.npy", G[idx]); np.save(out / "ref.npy", ref[idx])
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB) and 64 test samples")


if __name__ == "__main__":
    main()
