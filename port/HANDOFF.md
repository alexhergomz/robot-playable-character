# Handoff: running the generator on the real G1 through SONIC

Read this first if you are picking the work up.  Details and measurements are in
`port/README.md`; this file is the map and the runbook.

## Ground rules (from the project owner)

- **No reflashing** the robot.  It stays on JetPack 5.1.1 / TensorRT 8.5.2.
- **No cable** during runs: everything runs on the robot's Orin, wireless.
- On the robot: **no installs, no sudo** unless the owner says so; work from a user folder or `/tmp`.
- Nothing may **publish on the robot's DDS** or move the robot without the owner's explicit go-ahead.
  Listening is fine.  The deploy app is the only thing that commands motors.
- Robot in the **harness** for any motion test; speed capped (0.3 m/s); a person at the app's `O` key.
- Explain a change before implementing it; simplest idea first; report failures plainly; watch videos, don't work blind.
- Check robot temperatures periodically when working on it (`port/scripts/temps.sh`, listen-only).
- The laptop has 14 GB: run SONIC/generator sim jobs **one at a time** (~1.75 GB each).

## Architecture on the robot

```
Unitree remote --(rt/lowstate, read only)--> rpc.deploy.launcher
    generator (TensorRT, torch-free, phase hold on) -> PoseStreamer -> ZMQ localhost:5556 "pose"
NVIDIA g1_deploy_onnx_ref --input-type zmq  (SONIC encoder+decoder, TensorRT 8.5)
    -> rt/lowcmd -> motors
operator at the app's keyboard: Enter = use ZMQ stream, ] = start control, O = emergency stop
```

## Status

| Piece | State |
|---|---|
| Deploy app cross-built for the Orin (no reflash) | done; `port/patches/` (8-line TRT guard + float-parse fallback) |
| SONIC encoder/decoder on TRT 8.5 | FP32 exact / 1.8e-5 rad; FP16 not used |
| SONIC planner | does not build on TRT 8.5, not needed (streamed mode) |
| Generator on the Orin | ONNX -> TRT FP32 (6e-6 vs PyTorch); torch-free runtime `rpc/deploy/onboard.py` |
| Real-time budget | 37 ms/tick at horizon 12; **horizon 16 + replan every 3 fits** (16.7 ms avg); GPU sits at 306 MHz unless its min clock is raised (sudo) |
| Closed loop on the laptop: NVIDIA app (x86) + NVIDIA MuJoCo sim + our stream | works; `port/scripts/sim_closed_loop.sh` |
| Stand->walk speed variance | root cause: phase-channel drift while idle; fixed by `hold_phase_when_idle=True` (on in the launcher) |
| Onboard launcher with remote + deadman | done; tested in the NVIDIA closed loop (`results/.../launcher_basic_overlay.mp4`) |
| On the robot | **not run yet**: nothing has executed the deploy app or launcher on the Orin |

## Runbook for the robot

Bundle: on the laptop `bash port/scripts/make_robot_bundle.sh`, then copy `port/robot_bundle/` to the
robot's user folder (e.g. `~/rpc_bundle/`, via the robot's SSH key). It holds the app, its libraries,
SONIC's models, the generator, the `rpc` code and run scripts.  Nothing is installed.

**A. Listen-only checks (nothing moves)**
1. `bash temps.sh`: Orin and motor temperatures and battery.
2. `python3 -c "import zmq"` on the robot. If it fails, the launcher can't publish: decide with the owner
   (e.g. a user-level pyzmq) before going on.
3. Remote check: watch decoded sticks/buttons while someone uses the remote:
   `PYTHONPATH=py python3 -c "from rpc.deploy.launcher import DDSRemote; import time; r=DDSRemote('eth0'); [print(r.read()[0]) or time.sleep(0.5) for _ in range(40)]"`.
   Confirm: left stick up gives ly > 0, left gives lx < 0, right stick left gives rx < 0; and that **R1** isn't
   part of a Unitree firmware mode-switch combo (pick another deadman with `--deadman` if it is).

**B. Dry run with the motion board unreachable (nothing can move)**
4. `bash build_engine.sh` builds the generator's TRT engine on the Orin (GPU only).
5. Terminal 1: `IFACE=lo bash run_app.sh`. DDS is bound to loopback, so motor commands cannot reach the
   robot. Check that it builds SONIC's TRT engines, validates the observation config and connects to ZMQ.
   It will not enter control (no lowstate on `lo`), which is expected.
6. Terminal 2: `IFACE=eth0 bash run_launcher.sh` (reads the remote over eth0, publishes on localhost only).
   Hold R1: the launcher must report `streaming` with gen time per tick well under 60 ms per 3 ticks,
   and the app must log `Received ZMQ message - topic: 'pose'` after pressing Enter in terminal 1.

**C. First motion test (owner's explicit go-ahead only)**
7. Robot in the harness, in **dev mode** (Unitree's controller off), battery > 40 %, temperatures normal.
8. Terminal 1: `IFACE=eth0 bash run_app.sh`. Terminal 2: `bash run_launcher.sh`. Hold R1 with sticks neutral,
   press Enter, then `]`. Stand only first. Abort with `O` on any oscillation, stumble, or joint noise.
9. Then slow walking with the left stick (capped at 0.3 m/s). Release R1 to stop (brakes, then holds the
   stand pose).

Optional (needs sudo; owner's decision): raise the GPU min clock for horizon 12 / replan 1:
`echo 918000000 | sudo tee /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/min_freq` (resets on reboot).

## Reproducing the laptop side

- Upstream source: sparse clone of NVlabs/GR00T-WholeBodyControl at `port/patches/UPSTREAM_COMMIT` into
  `assets/sonic/gr00t_wbc`, then `git apply port/patches/gear_sonic_deploy_trt85_gcc10.patch`.
- Cross build (aarch64): see `port/README.md` "Rebuild" (robot sysroot copy + `port/Dockerfile` + `port/build.sh`).
- x86 build for the sim: `port/Dockerfile.x86` + `port/build_x86.sh`, TensorRT 10.7 libs from the pip wheel,
  headers from NVIDIA/TensorRT v10.7.0 (+ NvOnnxParser.h from onnx-tensorrt release/10.7-GA).
- Sim env: `uv venv --python 3.10 port/sim_env`, then install numpy 1.26.4, scipy 1.15.3, mujoco, tyro, pin,
  pyyaml, easydict, loguru, cyclonedds 0.10.2, pyzmq, msgpack, imageio[ffmpeg],
  tensorrt-cu12-libs/bindings 10.7.0 (`--extra-index-url https://pypi.nvidia.com`), and NVIDIA's
  unitree_sdk2py fork: `uv pip install --no-deps -e assets/sonic/gr00t_wbc/external_dependencies/unitree_sdk2_python`.
- Onboard generator bundle: `python port/scripts/export_onboard_bundle.py` (needs `runs/deploy/bundle.pt`
  from `scripts/export_deploy.py`).
- Closed loops: `bash port/scripts/sim_closed_loop.sh TAG [--speed 0.3 --hold-phase --short-stand]`,
  `bash port/scripts/launcher_closed_loop.sh TAG port/scripts/remote_script_basic.json`; overlay video:
  `python port/scripts/overlay_video.py --run port/results/closed_loop/TAG --remote ... --offset ...`.
- Sim model variants for the one-factor experiments: `python port/scripts/make_sim_variants.py`.

## Open issues, in priority order

1. Everything in "Runbook" above.
2. A brief speed overshoot (~0.5 m/s at 0.3 commanded) right after resuming a walk; the generator's
   walk onset is jerky in general (worst joint 5-8 rad/s in the first frames).
3. Generator speed gain below 1 (~0.9 at 0.3, ~0.75 at 0.6 m/s).
4. Train the learned PhaseCoupling (`rpc/generator/lmp.py`, `--coupling`): the principled fix for the
   phase drift that `hold_phase_when_idle` patches.
5. Earlier sim results (interruptibility, tracker lag) were measured without the phase hold; re-measure.
6. On the Orin, the `teleimager` service (not ours) crash-loops every ~30 s and reloads the camera driver;
   tell its owner.
