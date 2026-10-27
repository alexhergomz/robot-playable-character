"""Operator script: play one of NVIDIA's bundled reference motions through the app (no streaming).

Control experiment for the closed-loop sim: how much of a reference's root
travel does SONIC realise in NVIDIA's simulator when the reference is NVIDIA's
own?  Keys go to the app's stdin FIFO: ] start control, N next motion, T play.
"""
import argparse, json, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--keys", required=True)
ap.add_argument("--phases", default="")
ap.add_argument("--next", type=int, default=3, help="how many times to press N from the first motion")
ap.add_argument("--play-s", type=float, default=11.0)
ap.add_argument("--release-flag", default="/tmp/nvsim_release_band")
a = ap.parse_args()
keys, t0, log = open(a.keys, "w"), time.monotonic(), []
def mark(n): log.append((n, round(time.monotonic() - t0, 3))); print(log[-1], flush=True)
def key(c): keys.write(c); keys.flush()
time.sleep(1.5); mark("operator: ] (start control)"); key("]")
time.sleep(3.0); mark("operator: release band"); Path(a.release_flag).touch()
time.sleep(2.0)
for _ in range(a.next): key("N"); time.sleep(0.3)
time.sleep(1.0); mark("operator: T (play)"); key("T")
time.sleep(a.play_s); mark("end")
if a.phases: Path(a.phases).write_text(json.dumps({"phases": log}))
