# Records real SONIC encoder/decoder inputs from a closed-loop sim run (walk at 0.6 m/s, brake, stand),
# plus FP32 ONNX Runtime CPU outputs as the reference for the TensorRT 8.5 comparison.
# Usage: python port/scripts/capture_sonic_io.py OUT_DIR
import sys, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "port/results/trt_check"); OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import rpc.sim.sonic_policy as sp
rec = {"enc": [], "dec": []}
class Rec:
    def __init__(self, sess, key): self.s, self.k = sess, key
    def run(self, out, feeds):
        (name, x), = feeds.items(); rec[self.k].append(np.array(x, np.float32)); return self.s.run(out, feeds)
    def __getattr__(self, a): return getattr(self.s, a)
orig = sp.SonicPolicy.__init__
def init(self, *a, **k):
    orig(self, *a, **k); self.enc = Rec(self.enc, "enc"); self.dec = Rec(self.dec, "dec")
sp.SonicPolicy.__init__ = init
import sim_stream
sys.argv = ["sim_stream", "--speed", "0.6", "--walk-s", "6", "--after-s", "2"]
sim_stream.main()
import onnxruntime as ort
out = str(OUT)
for k, f in (("enc", "model_encoder.onnx"), ("dec", "model_decoder.onnx")):
    X = np.concatenate(rec[k]); idx = np.linspace(0, len(X) - 1, 64).astype(int); X = X[idx]
    s = ort.InferenceSession(str(ROOT / "assets/sonic/low_latency" / f), providers=["CPUExecutionProvider"])
    iname, oname = s.get_inputs()[0].name, s.get_outputs()[0].name
    Y = np.concatenate([s.run(None, {iname: X[i:i+1]})[0] for i in range(len(X))])
    np.save(f"{out}/{k}_in.npy", X); np.save(f"{out}/{k}_ref.npy", Y)
    print(k, "captured", len(rec[k]), "kept", X.shape, "->", Y.shape, "io names", iname, oname)
