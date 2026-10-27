"""Render a closed-loop launcher run with a controller/UI panel next to the sim view.

Panel: remote sticks and the deadman button (from the scripted remote), launcher
state (from its log), the command sent, and rolling plots of commanded vs
measured speed and yaw rate (from the sim log).  The remote script runs on the
launcher's clock; --offset maps it to sim time.
"""
import argparse, json, re
from pathlib import Path

import numpy as np
import imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="results/closed_loop/TAG (without suffix)")
ap.add_argument("--remote", required=True)
ap.add_argument("--offset", type=float, required=True, help="sim time = launcher time + offset")
ap.add_argument("--t0", type=float, default=0.0); ap.add_argument("--t1", type=float, default=46.0)
ap.add_argument("--max-speed", type=float, default=0.3); ap.add_argument("--max-turn", type=float, default=0.5)
ap.add_argument("--out", required=True)
a = ap.parse_args()

L = np.load(a.run + ".sim.npz")["log"]; ts, x, y = L[:, 0], L[:, 2], L[:, 3]; q = L[:, 5:9]
yaw = np.unwrap(np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2)))
dt = np.diff(ts, prepend=ts[0] - 1 / 30)
vx = np.r_[0, np.diff(x)] / dt; vy = np.r_[0, np.diff(y)] / dt
v_fwd = vx * np.cos(yaw) + vy * np.sin(yaw)                         # along the robot's heading
k = np.ones(15) / 15
v_fwd = np.convolve(v_fwd, k, "same"); yr = np.convolve(np.r_[0, np.diff(yaw)] / dt, k, "same")

events = json.loads(Path(a.remote).read_text())
def remote_at(t):
    cur = [0, 0, 0, 0, []]
    for ev in events:
        if ev[0] <= t: cur = ev
    return cur
def db(v, d=0.1):
    v = float(np.clip(v, -1, 1)); return 0.0 if abs(v) < d else np.sign(v) * (abs(v) - d) / (1 - d)
def cmd_at(t):
    _, lx, ly, rx, b = remote_at(t)
    if "R1" not in b: return None
    return db(ly) * a.max_speed, -db(lx) * a.max_speed, -db(rx) * a.max_turn

states = [(float(m.group(1)), m.group(2)) for m in
          (re.match(r"\[\s*([\d.]+)\] (\w+)", l) for l in Path(a.run + ".launcher.out").read_text().splitlines()) if m]
def state_at(t):
    s = "idle"
    for tt, ss in states:
        if tt <= t: s = ss
    return s

try:
    F = ImageFont.truetype("DejaVuSans.ttf", 15); FB = ImageFont.truetype("DejaVuSans-Bold.ttf", 18); FS = ImageFont.truetype("DejaVuSans.ttf", 12)
except OSError:
    F = FB = FS = ImageFont.load_default()

BG, FG, DIM = (24, 27, 33), (230, 232, 236), (130, 136, 148)
GREEN, AMBER, GREY, BLUE, ORANGE = (70, 190, 110), (235, 170, 60), (110, 115, 125), (90, 160, 240), (240, 130, 70)
W = 400

def stick(d, cx, cy, sx, sy, label):
    r = 52
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=DIM, width=2)
    d.line([cx - r, cy, cx + r, cy], fill=(50, 54, 62)); d.line([cx, cy - r, cx, cy + r], fill=(50, 54, 62))
    px, py = cx + sx * (r - 8), cy - sy * (r - 8)
    d.ellipse([px - 11, py - 11, px + 11, py + 11], fill=BLUE)
    d.text((cx, cy + r + 14), label, fill=DIM, font=FS, anchor="mm")

def plot(d, box, t, series, lo, hi, title):
    x0, y0, x1, y1 = box; span = 10.0
    d.rectangle(box, outline=(55, 60, 70)); d.text((x0 + 4, y0 + 3), title, fill=DIM, font=FS)
    zero = y1 - (0 - lo) / (hi - lo) * (y1 - y0)
    d.line([x0, zero, x1, zero], fill=(55, 60, 70))
    for tt_arr, vals, col in series:
        m = (tt_arr > t - span) & (tt_arr <= t)
        if m.sum() < 2: continue
        px = x1 - (t - tt_arr[m]) / span * (x1 - x0)
        py = y1 - (np.clip(vals[m], lo, hi) - lo) / (hi - lo) * (y1 - y0)
        d.line(list(zip(px, py)), fill=col, width=2)

reader = iio.get_reader(a.run + ".mp4")
tl = np.arange(a.t0, a.t1, 1 / 30)                                  # launcher-time grid for the command trace
cmd_trace = np.array([(c if c else (0, 0, 0)) for c in map(cmd_at, tl)])
frames = []
for fi, st in enumerate(ts):
    lt = st - a.offset
    if lt < a.t0 or lt > a.t1: continue
    try: img = Image.fromarray(reader.get_data(fi))
    except IndexError: break
    panel = Image.new("RGB", (W, img.height), BG); d = ImageDraw.Draw(panel)
    _, lx, ly, rx, btn = remote_at(lt); held = "R1" in btn
    d.text((14, 10), "Unitree remote (scripted)", fill=FG, font=FB)
    d.rounded_rectangle([W - 120, 10, W - 14, 36], 6, fill=GREEN if held else (60, 64, 72))
    d.text((W - 67, 23), "R1 deadman", fill=(10, 10, 10) if held else DIM, font=FS, anchor="mm")
    stick(d, 105, 108, lx, ly, "left: move")
    stick(d, 295, 108, rx, 0.0, "right: turn")
    s = state_at(lt); col = {"streaming": GREEN, "braking": AMBER}.get(s, GREY)
    d.rounded_rectangle([14, 192, W - 14, 222], 6, fill=col)
    d.text((W / 2, 207), f"launcher: {s.upper()}", fill=(10, 10, 10), font=FB, anchor="mm")
    c = cmd_at(lt)
    d.text((14, 232), "command: " + ("none (not streaming)" if c is None else f"vx {c[0]:+.2f} m/s  turn {c[2]:+.2f} rad/s"), fill=FG, font=F)
    m_now = np.argmin(np.abs(ts - st))
    d.text((14, 252), f"measured: v {v_fwd[m_now]:+.2f} m/s  yaw {yr[m_now]:+.2f} rad/s", fill=FG, font=F)
    sim_tl = tl + a.offset
    plot(d, (14, 292, W - 14, 375), st, [(sim_tl, cmd_trace[:, 0], ORANGE), (ts, v_fwd, BLUE)], -0.1, 0.5, "forward speed (m/s)")
    plot(d, (14, 384, W - 14, 468), st, [(sim_tl, cmd_trace[:, 2], ORANGE), (ts, yr, BLUE)], -0.3, 0.8, "yaw rate (rad/s)")
    d.text((14, 272), "command", fill=ORANGE, font=FS); d.text((84, 272), "measured", fill=BLUE, font=FS)
    di = ImageDraw.Draw(img)
    di.text((10, 8), f"t = {lt:5.1f} s   NVIDIA deploy app + MuJoCo sim, G1", fill=(255, 255, 255), font=F)
    out = Image.new("RGB", (img.width + W, img.height)); out.paste(img, (0, 0)); out.paste(panel, (img.width, 0))
    frames.append(np.asarray(out))
iio.mimsave(a.out, frames, fps=30, macro_block_size=8)
print(f"wrote {a.out}: {len(frames)} frames ({len(frames) / 30:.1f} s), {frames[0].shape[1]}x{frames[0].shape[0]}")
