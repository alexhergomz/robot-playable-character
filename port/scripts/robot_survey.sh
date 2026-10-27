#!/bin/bash
# Read-only survey of the G1's onboard computer.  Every command only reads.
sec() { echo; echo "=== $1 ==="; }
sec "host";            hostname; uname -srm; grep -E '^(PRETTY_NAME)=' /etc/os-release
sec "jetpack / L4T";   cat /etc/nv_tegra_release 2>/dev/null || echo "not a Jetson (no nv_tegra_release)"
                       dpkg-query -W -f='${Package} ${Version}\n' nvidia-jetpack 2>/dev/null
sec "cuda";            (nvcc --version 2>/dev/null || /usr/local/cuda/bin/nvcc --version 2>/dev/null) | tail -1
                       ls -d /usr/local/cuda* 2>/dev/null
sec "tensorrt";        dpkg-query -W -f='${Package} ${Version}\n' 'libnvinfer*' 'tensorrt*' 2>/dev/null | head -5
                       python3 -c 'import tensorrt as t; print("python tensorrt", t.__version__)' 2>/dev/null
sec "compute";         nproc; free -h | head -2; df -h / | tail -1
sec "network";         ip -4 -o addr show | awk '{print $2, $4}'
sec "unitree software"; ls -d /unitree /opt/unitree* ~/unitree* 2>/dev/null
                       find /unitree -maxdepth 3 -iname '*version*' 2>/dev/null | head -5 | while read f; do echo "$f: $(head -c 200 "$f" 2>/dev/null | tr '\n' ' ')"; done
sec "unitree sdk";     ls -d /usr/local/include/unitree /opt/unitree_robotics 2>/dev/null
                       python3 -c 'import unitree_sdk2py; print("unitree_sdk2py present")' 2>/dev/null
sec "running services"; systemctl list-units --type=service --state=running --no-pager 2>/dev/null | grep -iE 'unitree|sport|motion|robot|ai_|dds' | head -15
sec "unitree processes"; ps -eo pid,etime,comm 2>/dev/null | grep -iE 'unitree|sport|motion|robot|ai_' | head -15
sec "existing deploy installs"; ls -d ~/GR00T* ~/gear_sonic* ~/*sonic* 2>/dev/null || echo none
