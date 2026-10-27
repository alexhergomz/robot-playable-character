#!/bin/bash
# Cross-builds gear_sonic_deploy for the G1 Orin inside sonic-cross:focal.
# Mounts: /src = gear_sonic_deploy, /sysroot = robot rootfs copy (read-only), /deps = ORT + msgpack, /build = output.
set -e
cd /build
cmake -G Ninja /src \
  -DCMAKE_TOOLCHAIN_FILE=/port/aarch64-g1.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DTensorRT_ROOT=/usr \
  -DTensorRT_INCLUDE_DIR=/sysroot/usr/include/aarch64-linux-gnu \
  -DCUDA_RUNTIME_LIBRARY=/sysroot/usr/local/cuda-11.4/targets/aarch64-linux/lib/libcudart.so \
  -DCUDA_INCLUDE_DIR=/sysroot/usr/local/cuda-11.4/targets/aarch64-linux/include \
  -Donnxruntime_INCLUDE_DIR=/deps/onnxruntime-linux-aarch64-1.16.3/include \
  -Donnxruntime_LIBRARY=/deps/onnxruntime-linux-aarch64-1.16.3/lib/libonnxruntime.so \
  -DMSGPACK_INCLUDE_DIR=/deps/msgpack-c-cpp-3.3.0/include \
  -DCMAKE_CXX_FLAGS="-isystem /deps/cppzmq/include -isystem /deps/nlohmann_json/include -isystem /deps/msgpack-c-cpp-3.3.0/include" \
  "$@"
ninja -k 0 2>&1 | tail -${TAIL:-60}
