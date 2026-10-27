"""Torch-free generator for the robot's Orin (Python 3.8, numpy 1.17, no torch).

The deploy bundle (``scripts/export_deploy.py``) is split by
``port/scripts/export_onboard_bundle.py`` into ``meta.json`` (feature layout,
slices, init state), ``weights.npz`` and ``generator.onnx``.  This module
rebuilds the same :class:`LMPCharacterController` from those, with the network
behind ``step_numpy(x, gate)``:

* :class:`NumpyGenerator` -- reference implementation, CPU only.
* :class:`TRTGenerator`   -- TensorRT engine built on the Orin from the ONNX
  (FP32: matches PyTorch to 6e-6), driven through ``ctypes`` into libcudart
  because the robot has no pycuda.
"""
from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np

from ..generator.lmp_runtime import LMPCharacterController
from ..motion.features import FeatureConfig
from ..motion.robot_dataset import RobotFeatureSpec


def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


class NumpyGenerator:
    """``LMPGenerator.step`` in numpy, from ``weights.npz``."""

    def __init__(self, weights: str | Path):
        w = np.load(weights)
        self.w = {k: w[k].astype(np.float32) for k in w.files}
        # (K, O, I) -> (I, K*O) so one matmul evaluates every expert
        self.layers = []
        for name in ("l0", "l1", "l2"):
            W, b = self.w[name + ".weight"], self.w[name + ".bias"]
            K, O, I = W.shape
            self.layers.append((W.reshape(K * O, I).T.copy(), b.reshape(K * O), K, O))

    def step_numpy(self, x: np.ndarray, gate: np.ndarray) -> np.ndarray:
        w = self.w
        xn = (x - w["x_mean"]) / w["x_std"]
        g = _elu(w["gate.0.weight"] @ gate + w["gate.0.bias"])
        g = _elu(w["gate.3.weight"] @ g + w["gate.3.bias"])
        g = w["gate.5.weight"] @ g + w["gate.5.bias"]
        c = np.exp(g - g.max()); c /= c.sum()
        h = xn
        for i, (W, b, K, O) in enumerate(self.layers):
            h = c @ (h @ W + b).reshape(K, O)
            if i < 2:
                h = _elu(h)
        return h * w["y_std"] + w["y_mean"]


class ORTGenerator:
    """``generator.onnx`` on ONNX Runtime (CPU).  Laptop testing only; the robot has no ORT."""

    def __init__(self, onnx_path: str | Path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.s = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])

    def step_numpy(self, x: np.ndarray, gate: np.ndarray) -> np.ndarray:
        return self.s.run(None, {"x": np.asarray(x, np.float32)[None],
                                 "gate_x": np.asarray(gate, np.float32)[None]})[0][0]


class TRTGenerator:
    """TensorRT 8.5 engine with inputs ``x``, ``gate_x`` and output ``y`` (batch 1, FP32).

    One call = two async uploads, the engine, one async download, one sync,
    all on one stream from pinned host buffers.  Blocking cudaMemcpy plus
    ``execute_v2`` cost 3 ms per call on the Orin against 0.65 ms of GPU work.
    With ``graph=True`` that sequence is captured once as a CUDA graph and
    replayed, which removes the per-call launch overhead as well.
    """

    _H2D, _D2H = 1, 2

    def __init__(self, engine: str | Path, cudart: str = "libcudart.so", graph: bool = True):
        import tensorrt as trt

        cu = self._cuda = ctypes.CDLL(cudart)
        logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(Path(engine).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"could not deserialize {engine}")
        self.ctx = self.engine.create_execution_context()
        self.stream = ctypes.c_void_p()
        self._check(cu.cudaStreamCreate(ctypes.byref(self.stream)))
        self.host, self.dev, self.bindings, self._pinned = {}, {}, [], []
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            if self.engine.get_binding_dtype(i) != trt.float32:
                raise TypeError(f"{name}: expected float32")
            n = int(np.prod(shape)) * 4
            hp, dp = ctypes.c_void_p(), ctypes.c_void_p()
            self._check(cu.cudaHostAlloc(ctypes.byref(hp), ctypes.c_size_t(n), ctypes.c_uint(0)))
            self._check(cu.cudaMalloc(ctypes.byref(dp), ctypes.c_size_t(n)))
            self._pinned.append(hp)
            buf = (ctypes.c_float * (n // 4)).from_address(hp.value)
            self.host[name] = np.ctypeslib.as_array(buf).reshape(shape)
            self.dev[name] = dp
            self.bindings.append(dp.value)
        self._x, self._g, self._y = self.host["x"], self.host["gate_x"], self.host["y"]
        self._graph_exec = None
        if graph:
            self._capture()

    def _check(self, err):
        if err != 0:
            raise RuntimeError(f"CUDA runtime error {err}")

    def _enqueue(self):
        cu, s = self._cuda, self.stream
        for name in ("x", "gate_x"):
            h = self.host[name]
            self._check(cu.cudaMemcpyAsync(self.dev[name], ctypes.c_void_p(h.ctypes.data),
                                           ctypes.c_size_t(h.nbytes), ctypes.c_int(self._H2D), s))
        if not self.ctx.execute_async_v2(self.bindings, s.value):
            raise RuntimeError("TensorRT enqueue failed")
        self._check(cu.cudaMemcpyAsync(ctypes.c_void_p(self._y.ctypes.data), self.dev["y"],
                                       ctypes.c_size_t(self._y.nbytes), ctypes.c_int(self._D2H), s))

    def _capture(self):
        cu = self._cuda
        self._enqueue()                       # TensorRT allocates lazily; warm up before capture
        self._check(cu.cudaStreamSynchronize(self.stream))
        g, ge = ctypes.c_void_p(), ctypes.c_void_p()
        self._check(cu.cudaStreamBeginCapture(self.stream, ctypes.c_int(0)))   # global mode
        self._enqueue()
        self._check(cu.cudaStreamEndCapture(self.stream, ctypes.byref(g)))
        self._check(cu.cudaGraphInstantiate(ctypes.byref(ge), g, None, None, ctypes.c_size_t(0)))
        self._graph, self._graph_exec = g, ge

    def step_numpy(self, x: np.ndarray, gate: np.ndarray) -> np.ndarray:
        self._x[...] = np.asarray(x, np.float32).reshape(self._x.shape)
        self._g[...] = np.asarray(gate, np.float32).reshape(self._g.shape)
        if self._graph_exec is not None:
            self._check(self._cuda.cudaGraphLaunch(self._graph_exec, self.stream))
        else:
            self._enqueue()
        self._check(self._cuda.cudaStreamSynchronize(self.stream))
        return self._y.reshape(-1).copy()

    def __del__(self):
        cu = getattr(self, "_cuda", None)
        if cu is None:
            return
        for p in getattr(self, "dev", {}).values():
            cu.cudaFree(p)
        for p in getattr(self, "_pinned", []):
            cu.cudaFreeHost(p)


def load(bundle_dir: str | Path, backend: str = "trt", engine: str | Path | None = None,
         graph: bool = True, hold_phase_when_idle: bool = False):
    """Generator controller from an onboard bundle, reset to the bundled init state."""
    d = Path(bundle_dir)
    meta = json.loads((d / "meta.json").read_text())
    init = np.load(d / "init.npz")
    m = meta["manifest"]
    cfg = FeatureConfig(fps=m["fps"], n_past=m["n_past"], n_future=m["n_future"],
                        span=m["span"], gaits=tuple(m["gaits"]))
    spec = RobotFeatureSpec(m["n_joints"], cfg, m["n_feet"])
    if backend == "trt":
        net = TRTGenerator(engine or d / "generator.fp32.engine", graph=graph)
    elif backend == "numpy":
        net = NumpyGenerator(d / "weights.npz")
    elif backend == "ort":
        net = ORTGenerator(d / "generator.onnx")
    else:
        raise ValueError(backend)
    C = m["phase_channels"]
    g = LMPCharacterController(net, spec, {k: tuple(v) for k, v in meta["out_slices"].items()},
                               4 * C, n_feet=m["n_feet"], keep_cols=init["keep_cols"],
                               curve={k: init["curve_" + k] for k in ("speed", "amplitude")},
                               hold_phase_when_idle=hold_phase_when_idle)
    g.reset(joint_pos=init["stand_pose"].copy(), root_height=float(meta["root_height"]),
            manifold=init["init_manifold"].copy(), foot=init["init_foot"].copy())
    return g
