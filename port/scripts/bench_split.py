"""Split PoseStreamer.tick time into network calls vs everything else (numpy backend, any machine)."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rpc.control.command import Command
from rpc.deploy import onboard
from rpc.deploy.streamer import PoseStreamer, StreamLimits
from rpc.sim import sonic as P
b = sys.argv[1]; init = np.load(Path(b) / "init.npz")
gen = onboard.load(b, backend="numpy")
net = gen.model; orig = net.step_numpy; acc = {"t": 0.0, "n": 0}
def timed(x, g):
    t = time.perf_counter(); y = orig(x, g); acc["t"] += time.perf_counter() - t; acc["n"] += 1; return y
net.step_numpy = timed
clock = [0.0]
st = PoseStreamer(gen, lambda m: None, joint_lower=init["joint_lower"], joint_upper=init["joint_upper"],
                  mujoco_to_isaaclab=P.MUJOCO_TO_ISAACLAB, horizon=12, limits=StreamLimits(max_speed=0.3),
                  stand_pose=init["stand_pose"].copy(), clock=lambda: clock[0])
walk = Command(move=np.array([0.3, 0.0]), responsiveness=0.6)
for _ in range(20): st.tick(walk); clock[0] += 0.02
acc.update(t=0.0, n=0); T = 0.0; N = 200
for _ in range(N):
    t = time.perf_counter(); st.tick(walk); T += time.perf_counter() - t; clock[0] += 0.02
print(f"per tick: total {T/N*1e3:.2f} ms | network calls {acc['n']/N:.1f} taking {acc['t']/N*1e3:.2f} ms | everything else {(T-acc['t'])/N*1e3:.2f} ms")
