# LISTEN-ONLY temperature snapshot: Orin thermal zones + one rt/lowstate sample (motors) + BMS. Publishes nothing.
echo "== Orin thermal zones (C)"
for z in /sys/class/thermal/thermal_zone*; do printf "%s=%.1f  " "$(cat $z/type)" "$(echo "$(cat $z/temp)/1000" | bc -l)"; done; echo
echo "== GPU/CPU load"; head -1 /proc/loadavg
timeout 20 python3 - <<'PY'
import time
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, BmsState_
ChannelFactoryInitialize(0, "eth0")
got = {}
def cb(m): got.setdefault("s", m)
def cbb(m): got.setdefault("b", m)
s = ChannelSubscriber("rt/lowstate", LowState_); s.Init(cb, 1)
b = ChannelSubscriber("rt/lf/bmsstate", BmsState_); b.Init(cbb, 1)
t0 = time.time()
while ("s" not in got or "b" not in got) and time.time() - t0 < 10: time.sleep(0.05)
if "b" in got:
    m_ = got["b"]; print(f"== battery SOC {m_.soc}%  current {m_.current}  temps {list(m_.temperature)[:4]}")
else: print("== battery: no bmsstate")
if "s" not in got: print("no lowstate"); raise SystemExit
m = got["s"]
names = ["L_hip_p","L_hip_r","L_hip_y","L_knee","L_ank_p","L_ank_r","R_hip_p","R_hip_r","R_hip_y","R_knee","R_ank_p","R_ank_r",
         "waist_y","waist_r","waist_p","L_sh_p","L_sh_r","L_sh_y","L_elb","L_wr_r","L_wr_p","L_wr_y","R_sh_p","R_sh_r","R_sh_y","R_elb","R_wr_r","R_wr_p","R_wr_y"]
T = [tuple(m.motor_state[i].temperature) for i in range(29)]
hot = sorted(range(29), key=lambda i: -max(T[i]))
print("== motor temps (C, [surface, winding]) hottest first:")
print("  " + "  ".join(f"{names[i]} {T[i][0]}/{T[i][1]}" for i in hot[:8]))
print(f"  max {max(max(t) for t in T)}  median {sorted(max(t) for t in T)[14]}")
PY
