# On the robot, /tmp/gen_check: GPU only; touches no DDS topic or motor.
cd /tmp/gen_check; T=/usr/src/tensorrt/bin/trtexec
python3 -c "import tensorrt as t; print('python tensorrt', t.__version__)" 2>&1 | tail -1
python3 -c "import pycuda; print('pycuda ok')" 2>&1 | tail -1
for prec in fp32 fp16; do
  flag=""; [ $prec = fp16 ] && flag="--fp16"
  $T --onnx=generator.onnx $flag --saveEngine=gen.$prec.engine > build_$prec.log 2>&1 || { echo "BUILD FAIL $prec"; tail -5 build_$prec.log; continue; }
  grep -E "\] Latency: min|GPU Compute Time: min" build_$prec.log | sed "s/^.*\[I\] /$prec /"
  for i in $(seq -w 0 63); do
    $T --loadEngine=gen.$prec.engine --loadInputs=x:in/x_$i.bin,gate_x:in/g_$i.bin --exportOutput=out/${prec}_$i.json --iterations=1 --warmUp=0 --duration=0 > /dev/null 2>&1 || echo "RUN FAIL $prec $i"
  done
done
ls out | wc -l
