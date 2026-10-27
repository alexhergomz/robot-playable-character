"""Operator at the app's keyboard for the launcher test: Enter (use ZMQ stream), ] (start), release the sim band."""
import argparse, json, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--keys", required=True); ap.add_argument("--log", required=True)
ap.add_argument("--release-flag", default="/tmp/nvsim_release_band"); a = ap.parse_args()
k, t0, log = open(a.keys, "w"), time.monotonic(), []
def at(t, name, fn):
    time.sleep(max(0.0, t - (time.monotonic() - t0))); fn(); log.append((name, round(time.monotonic() - t0, 3))); print(log[-1], flush=True)
at(3.0, "Enter", lambda: (k.write("\n"), k.flush()))
at(3.5, "]", lambda: (k.write("]"), k.flush()))
at(6.0, "release band", lambda: Path(a.release_flag).touch())
at(45.0, "end", lambda: None)
Path(a.log).write_text(json.dumps({"phases": log}))
