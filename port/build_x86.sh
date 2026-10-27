#!/bin/bash
# Native x86 build (inside sonic-x86:noble). /trt_h = TRT 10.7 headers, /trt_lib = TRT libs, /deps = ORT x64.
set -e
cd /build
cmake -G Ninja /src -DCMAKE_BUILD_TYPE=Release \
  -DTensorRT_INCLUDE_DIR=/trt_h -DTensorRT_ROOT=/trt_lib \
  -Donnxruntime_INCLUDE_DIR=/deps/onnxruntime-linux-x64-1.16.3/include \
  -Donnxruntime_LIBRARY=/deps/onnxruntime-linux-x64-1.16.3/lib/libonnxruntime.so \
  -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined"
ninja -k 0 g1_deploy_onnx_ref 2>&1 | tail -${TAIL:-40}
