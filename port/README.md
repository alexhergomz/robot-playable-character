# Porting GEAR-SONIC's deploy app to the G1's Orin NX (no reflash)

**Start with [`HANDOFF.md`](HANDOFF.md)** for the current state and the robot runbook; this file is the detailed log.

Robot: Orin NX 16 GB, JetPack 5.1.1 (L4T R35.3.1), Ubuntu 20.04, glibc 2.31,
GCC 9.4, CUDA 11.4, TensorRT 8.5.2. NVIDIA targets JetPack 6 / TensorRT 10.7.
Reflashing is ruled out, and the robot must run untethered, so everything
(app + generator) has to run onboard.

## Status (2026-10-26)

| Step | Result |
|---|---|
| Cross-compile `g1_deploy_onnx_ref` for aarch64 | **Done.** Max GLIBC 2.18, libstdc++ static, all libs resolve on the robot (`ldd`). |
| TRT 8.5 numerics, encoder+decoder (low_latency), 64 real sim inputs | **FP32 passes**: encoder exact (0 error), decoder max 1.8e-5 rad. FP16 flips 13/4096 quantized encoder tokens (NVIDIA default is FP32, keep it). Latency enc 0.83 ms + dec 1.83 ms on the Orin. |
| Planner (`planner_sonic.onnx`, HF nvidia/GEAR-SONIC) | Fails to parse on TRT 8.5 (no plugin for an op, most likely `LayerNormalization`, native only from TRT 8.6). **Not needed**: streamed-motion mode doesn't use it; omit `--planner-file` (null planner is guarded in the app). |
| NVIDIA unit test (`run_tests`, FK only) | Runs on the robot. Quaternion mismatches are all q vs -q sign flips (harmless). Angular velocity mismatches up to 5.5 rad/s are **unexplained**. Next: build the same test natively on x86 against the same data (`reference/example` stands in for the missing `bones_072925_test`) to tell a port bug from a test-data mismatch. |
| Our generator on the Orin | **Not started.** Export `LMPGenerator.forward(x, gate_x)` to ONNX (the rest of `lmp_runtime.py` is numpy), check vs PyTorch, build TRT engine, time a 12-frame horizon against the 20 ms budget. Robot has Python 3.8 / numpy 1.17.4 and no torch. |
| First real run (harness, wireless, 0.3 m/s cap) | Needs explicit go-ahead. This is the only step that moves the robot. |

### Update (2026-10-26, afternoon)

- **NVIDIA unit test cleared**: on the same motion the robot build and a native x86 build fail at the identical 15,586 entries with identical values; the earlier mismatch was directory order (different motion loaded).
- **Generator on the Orin**: exported to ONNX (`scripts/export_generator_onnx.py`), TRT 8.5 FP32 matches PyTorch to 6e-6, 0.65 ms/step in trtexec. Torch-free onboard runtime `rpc/deploy/onboard.py` (TRT via ctypes, numpy reference) reproduces the torch controller to 1e-6 over 500 steps; robot Python 3.8 / numpy 1.17 output matches the laptop to 1.4e-6.
- **Onboard timing** (`scripts/bench_onboard.py`, `bench_split.py`): 73 ms/tick -> 37 ms after vectorising `TrajectoryPredictor.apply_command` (exact to 1.5e-14, test added) and pinned/async/CUDA-graph TRT calls. Remainder is the GPU parked at 306 MHz by its governor (MAXN max 918 MHz). Without sudo, horizon 16 + `replan_every=3` fits (16.7 ms/frame average, worst 58 ms vs 80 ms slack). With `min_freq` raised to 918 MHz (needs sudo) horizon 12 / every frame should fit (~18 ms, not yet measured).
- **Closed loop of NVIDIA's real C++ app** (x86 build, TRT 10.7) against **NVIDIA's MuJoCo sim** (`gear_sonic/utils/mujoco_sim`, their unitree_sdk2py fork) on the laptop: `scripts/sim_closed_loop.sh`, `nvsim_run.py`, `stream_to_app.py`, `play_reference.py`; env in `sim_env/` (uv, Python 3.10). Use `--input-type zmq` (keyboard: Enter = ZMQ stream, `]` = start, `O` = stop); `zmq_manager` stops immediately without a planner.
  - Pipeline works end to end: stream received, merged at 50 Hz, control starts, band released, robot stays up through walk / turn / brake / dead-man stop.
  - NVIDIA's own walk reference realises 100% of its travel (5.00 of 4.98 m).
  - **Ours under-walks in NVIDIA's sim**: cmd 0.3 -> 0.09 m/s, cmd 0.6 -> 0.35 m/s (our Python sim on SONIC's model gave 0.39 / 0.52). Turning is fine (0.43 rad/s for 0.4). Our hip-pitch stride at 0.3 m/s is 0.23 rad p2p vs 0.79 in NVIDIA's walk; NVIDIA's sim uses box soles + per-joint frictionloss 0.2, SONIC's training model uses 4 point contacts per foot. Video: `results/closed_loop/compare_walk.mp4`.

### Update (2026-10-27): the "slow walk" was phase drift while standing

- The under-walking above was **not** the simulator, the app or the port. Matched one factor at a time: the Python SONIC wrapper reproduces the app exactly (reference playback frame-exact, encoder token identical, decoder within logging jitter, same gain formula); physics step and foot model don't matter. The runs differed in how long the generator **stood before walking**.
- Cause (state-swap test: only the phase manifold restores the walk; pose and trajectory buffer don't): the 5 phase channels advance at their own predicted frequencies (1.0-1.4 Hz); walking keeps them coordinated, standing doesn't, so the walk starts from an arbitrary channel configuration. Realised speed at 0.6 m/s commanded: 0.34-0.68 m/s depending only on stand time (0.08-0.52 at 0.3).
- Fix: `LMPCharacterController(hold_phase_when_idle=True)` locks the channels to the seed frame's relative phases on idle steps. Realised speed is then 0.26-0.27 (cmd 0.3) and 0.44-0.45 (cmd 0.6) for 1-15 s stands, pose jitter while standing halves, walking steps are bit-identical. Confirmed in NVIDIA's app + sim: 8.5 s stand 0.09 -> 0.27 m/s. Opt-in (default off); plumbed through `onboard.load`, `sim_stream.build`, `stream_to_app.py --hold-phase`; 3 tests. **The robot launcher should enable it.** The principled fix is training the learned PhaseCoupling.
- Instant re-alignment at walk onset also fixes speed but adds a joint-speed spike (up to 9.9 rad/s); not used.
- Memory: each SONIC-in-the-loop Python run peaks at ~1.75 GB; the laptop has 14 GB. Run them sequentially (10 in parallel caused an OOM).

Nothing has been installed on the robot. Everything ran from `/tmp` and never
published on DDS. The deploy binary has **never been executed**.

## Layout

- `Dockerfile`: Ubuntu 20.04 + `g++-10-aarch64-linux-gnu` (`sonic-cross:focal`).
- `aarch64-g1.cmake`: toolchain against a read-only copy of the robot's rootfs.
- `build.sh`: configure + build (run inside the container, see below).
- `patches/gear_sonic_deploy_trt85_gcc10.patch`: changes to upstream at
  `patches/UPSTREAM_COMMIT` (NVlabs/GR00T-WholeBodyControl, `gear_sonic_deploy/`):
  - TRT-version guards for `kINT64` (>=10), `kBF16` (>=9), `kFP8` (>=8.6).
  - `motor_gain_scaling.cpp`: strtof/snprintf fallback when libstdc++ lacks
    floating-point `from_chars`/`to_chars` (GCC < 11), upstream path kept.
- `deps/` (not versioned, re-download): ONNX Runtime 1.16.3 aarch64,
  msgpack-c cpp-3.3.0, nlohmann json 3.11.3 single header, cppzmq 4.10.0
  headers (robot's cppzmq 4.7.0 lacks `zmq::sockopt`).
- `results/`: TRT check inputs/references (`trt_check/`), robot-side logs,
  outputs and engines (`robot_tmp/`, engines only valid on this Orin + TRT 8.5.2),
  FK test logs, robot API listener logs.
- `scripts/`: `capture_sonic_io.py` (records real SONIC I/O from `scripts/sim_stream.py`;
  its output path points at an old session scratchpad, so update it),
  `temps.sh` (listen-only Orin + motor temps), `listen_api.py`, `robot_survey.sh`.

## Rebuild

```bash
# 1. robot rootfs copy (read-only rsync; ~6 GB) -> assets/sonic/sysroot_g1
#    copy /usr/include, /usr/lib/aarch64-linux-gnu, /lib/aarch64-linux-gnu,
#    /usr/local/cuda-11.4/targets/aarch64-linux, /usr/lib/cmake (skip *.a except libgtest*),
#    then make absolute symlinks relative (they break under another root).
# 2. upstream source
git clone --depth 1 --filter=blob:none --sparse https://github.com/NVlabs/GR00T-WholeBodyControl.git assets/sonic/gr00t_wbc
cd assets/sonic/gr00t_wbc && git sparse-checkout set gear_sonic_deploy && git apply ../../../port/patches/gear_sonic_deploy_trt85_gcc10.patch
# 3. build
docker build -t sonic-cross:focal port
docker run --rm -u $(id -u):$(id -g) \
  -v $PWD/assets/sonic/gr00t_wbc/gear_sonic_deploy:/src \
  -v $PWD/assets/sonic/sysroot_g1:/sysroot:ro -v $PWD/port/deps:/deps:ro \
  -v $PWD/port:/port:ro -v $PWD/port/build:/build sonic-cross:focal /port/build.sh
# -> gear_sonic_deploy/target/release/{g1_deploy_onnx_ref,run_tests}
```

Runtime bundle on the robot: binary + `lib/` with `libonnxruntime.so*` and
the SDK's `libddsc*` (from `thirdparty/unitree_sdk2/thirdparty/lib/aarch64`),
run with `LD_LIBRARY_PATH=<bundle>/lib`. The zmq test tools fail to link
(missing `-pthread`) and aren't needed.
