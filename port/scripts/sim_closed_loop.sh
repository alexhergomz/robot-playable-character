#!/bin/bash
# Closed loop on the laptop: NVIDIA's MuJoCo sim <-DDS lo-> NVIDIA's deploy app (x86) <-ZMQ- our generator/streamer.
# Usage: port/scripts/sim_closed_loop.sh TAG [extra stream_to_app.py args]
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
TAG=$1; shift
PORT=$REPO/port; R=$PORT/results/closed_loop; mkdir -p $R
DEP=$REPO/assets/sonic/gr00t_wbc/gear_sonic_deploy; SP=$PORT/sim_env/lib/python3.10/site-packages
SONIC=$REPO/assets/sonic/low_latency
MUJOCO_GL=egl $PORT/sim_env/bin/python $PORT/scripts/nvsim_run.py --duration 80 --video $R/$TAG.mp4 --log $R/$TAG.sim.npz ${SCENE:+--scene $SCENE} ${SIM_ARGS:-} > $R/$TAG.sim.out 2>&1 &
SIM=$!
sleep 3
KEYS=/tmp/sonic_app_keys; rm -f $KEYS; mkfifo $KEYS
( cd $DEP && env LD_LIBRARY_PATH=$PORT/bin_x86/lib:$SP/tensorrt_libs:$SP/nvidia/cuda_runtime/lib \
  timeout 85 $PORT/bin_x86/g1_deploy_onnx_ref lo $SONIC/model_decoder.onnx reference/example/ \
  --obs-config policy/release/observation_config_low_latency.yaml --encoder-file $SONIC/model_encoder.onnx \
  --input-type zmq --zmq-host localhost --zmq-port 5556 --zmq-topic pose --disable-crc-check < $KEYS > $R/$TAG.app.log 2>&1 ) &
APP=$!
exec 3>$KEYS                       # hold the app's stdin open (a FIFO blocks until both ends are open)
for i in $(seq 1 120); do grep -q "G1Deploy object created successfully" $R/$TAG.app.log 2>/dev/null && break; sleep 0.5; done
echo "app ready after ~$((i/2)) s"
$PORT/sim_env/bin/python $PORT/scripts/record_app_debug.py --out $R/$TAG.app_debug.npz --duration 45 > $R/$TAG.debug.out 2>&1 &
DBG=$!
${OPERATOR:-python3 $PORT/scripts/stream_to_app.py} --keys $KEYS --phases $R/$TAG.phases.json "$@" > $R/$TAG.stream.out 2>&1
exec 3>&-
wait $DBG; wait $APP; wait $SIM
echo "done: $R/$TAG.*"
