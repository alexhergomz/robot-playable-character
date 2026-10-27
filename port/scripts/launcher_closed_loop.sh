#!/bin/bash
# The onboard launcher (ORT backend on the laptop) in NVIDIA's app + sim closed loop, driven by a scripted remote.
# Usage: launcher_closed_loop.sh TAG REMOTE_SCRIPT.json [launcher args]
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
TAG=$1; RS=$2; shift 2
PORT=$REPO/port; R=$PORT/results/closed_loop; mkdir -p $R
DEP=$REPO/assets/sonic/gr00t_wbc/gear_sonic_deploy; SP=$PORT/sim_env/lib/python3.10/site-packages
SONIC=$REPO/assets/sonic/low_latency
MUJOCO_GL=egl $PORT/sim_env/bin/python $PORT/scripts/nvsim_run.py --duration 75 --video $R/$TAG.mp4 --log $R/$TAG.sim.npz > $R/$TAG.sim.out 2>&1 &
SIM=$!; sleep 3
KEYS=/tmp/sonic_app_keys; rm -f $KEYS; mkfifo $KEYS
( cd $DEP && env LD_LIBRARY_PATH=$PORT/bin_x86/lib:$SP/tensorrt_libs:$SP/nvidia/cuda_runtime/lib \
  timeout 80 $PORT/bin_x86/g1_deploy_onnx_ref lo $SONIC/model_decoder.onnx reference/example/ \
  --obs-config policy/release/observation_config_low_latency.yaml --encoder-file $SONIC/model_encoder.onnx \
  --input-type zmq --zmq-host localhost --zmq-port 5556 --zmq-topic pose --disable-crc-check < $KEYS > $R/$TAG.app.log 2>&1 ) &
APP=$!
exec 3>$KEYS
for i in $(seq 1 120); do grep -q "G1Deploy object created successfully" $R/$TAG.app.log 2>/dev/null && break; sleep 0.5; done
echo "app ready after ~$((i/2)) s"
( cd $REPO && python3 -m rpc.deploy.launcher port/results/onboard_bundle --backend ort \
  --remote script:$RS --duration 47 "$@" > $R/$TAG.launcher.out 2>&1 ) &
LAU=$!
python3 $PORT/scripts/operator_keys.py --keys $KEYS --log $R/$TAG.phases.json > $R/$TAG.operator.out 2>&1
wait $LAU; exec 3>&-; wait $APP; wait $SIM
echo "done: $R/$TAG.*"
