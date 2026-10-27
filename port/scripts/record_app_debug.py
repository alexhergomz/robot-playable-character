"""Record the deploy app's per-tick debug stream (tcp://localhost:5557, topic g1_debug, msgpack) to npz."""
import argparse, time
import numpy as np, msgpack, zmq
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--duration", type=float, default=70)
a = ap.parse_args()
s = zmq.Context().socket(zmq.SUB); s.connect("tcp://localhost:5557"); s.setsockopt(zmq.SUBSCRIBE, b"g1_debug")
keys = ["index", "base_quat", "base_ang_vel", "body_q", "body_dq", "last_action", "token_state",
        "body_q_target", "base_quat_target"]
rec = {k: [] for k in keys}; rec["t"] = []
t0 = time.monotonic(); s.RCVTIMEO = 500
while time.monotonic() - t0 < a.duration:
    try: msg = s.recv()
    except zmq.Again: continue
    d = msgpack.unpackb(msg[len(b"g1_debug"):], raw=False)
    if not all(k in d for k in keys): continue
    rec["t"].append(time.monotonic() - t0)
    for k in keys: rec[k].append(np.asarray(d[k], float))
np.savez(a.out, **{k: np.array(v) for k, v in rec.items()}); print("recorded", len(rec["t"]), "ticks")
