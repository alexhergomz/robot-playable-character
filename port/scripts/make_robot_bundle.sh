#!/bin/bash
# Assemble everything the G1's Orin needs into port/robot_bundle/ (git-ignored), ready to copy:
#   rsync -a port/robot_bundle/ unitree@<robot>:~/rpc_bundle/
# Nothing here installs anything on the robot.  Prerequisites on the laptop (see port/README.md):
#   port/bin_aarch64/g1_deploy_onnx_ref   (cross build: port/build.sh)
#   port/deps/onnxruntime-linux-aarch64-1.16.3
#   port/results/onboard_bundle            (python port/scripts/export_onboard_bundle.py)
#   assets/sonic/low_latency/*.onnx        (HF nvidia/GEAR-SONIC)
set -eu
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DEP=$REPO/assets/sonic/gr00t_wbc/gear_sonic_deploy
B=$REPO/port/robot_bundle
rm -rf "$B"; mkdir -p "$B"/{app/lib,app/policy,app/reference,generator,py}

# NVIDIA deploy app (aarch64, TRT 8.5 port) + its runtime libraries
cp "$REPO/port/bin_aarch64/g1_deploy_onnx_ref" "$B/app/"
cp -P "$REPO"/port/deps/onnxruntime-linux-aarch64-1.16.3/lib/libonnxruntime.so* "$B/app/lib/"
cp -P "$DEP"/thirdparty/unitree_sdk2/thirdparty/lib/aarch64/libddsc* "$B/app/lib/"
cp "$REPO"/assets/sonic/low_latency/model_encoder.onnx "$REPO"/assets/sonic/low_latency/model_decoder.onnx "$B/app/policy/"
cp "$DEP/policy/release/observation_config_low_latency.yaml" "$B/app/policy/"
cp -r "$DEP/reference/example" "$B/app/reference/"          # required positional arg; we stream instead
cp -r "$DEP/g1" "$B/app/g1"

# Generator (torch-free) + the rpc code the launcher imports
cp "$REPO"/port/results/onboard_bundle/{meta.json,init.npz,weights.npz,generator.onnx} "$B/generator/"
(cd "$REPO" && find rpc -name "*.py" -o -name "*.json" | grep -v __pycache__ | cpio -pdm --quiet "$B/py")
cp "$REPO/port/scripts/temps.sh" "$REPO/port/scripts/listen_api.py" "$B/"

cat > "$B/build_engine.sh" <<'EOS'
#!/bin/bash
# Build the generator's TensorRT engine on the Orin (FP32; matches PyTorch to 6e-6). GPU only.
cd "$(dirname "$0")/generator" && /usr/src/tensorrt/bin/trtexec --onnx=generator.onnx --saveEngine=generator.fp32.engine
EOS
cat > "$B/run_app.sh" <<'EOS'
#!/bin/bash
# NVIDIA's deploy app.  IFACE=lo for a dry run (motor commands cannot reach the motion board),
# IFACE=eth0 for the real robot.  Operator keys in this terminal: Enter = use ZMQ stream,
# ] = start control, O = emergency stop.
IFACE=${IFACE:-lo}
cd "$(dirname "$0")/app"
extra=(); [ "$IFACE" = "lo" ] && extra=(--disable-crc-check)
LD_LIBRARY_PATH=$PWD/lib:/usr/local/cuda/lib64 ./g1_deploy_onnx_ref "$IFACE" policy/model_decoder.onnx reference/example/ \
  --obs-config policy/observation_config_low_latency.yaml --encoder-file policy/model_encoder.onnx \
  --input-type zmq --zmq-host localhost --zmq-port 5556 --zmq-topic pose "${extra[@]}"
EOS
cat > "$B/run_launcher.sh" <<'EOS'
#!/bin/bash
# Remote (rt/lowstate, read only) -> generator (TensorRT) -> ZMQ localhost:5556.  Hold R1 to move.
cd "$(dirname "$0")"
LD_LIBRARY_PATH=/usr/local/cuda/lib64 PYTHONPATH=$PWD/py python3 -m rpc.deploy.launcher generator \
  --backend trt --engine generator/generator.fp32.engine --iface ${IFACE:-eth0} --max-speed ${MAX_SPEED:-0.3} "$@"
EOS
chmod +x "$B"/*.sh
du -sh "$B"; ls "$B"
