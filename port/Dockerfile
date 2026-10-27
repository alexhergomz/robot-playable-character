# Cross-compiles NVIDIA's gear_sonic_deploy for the G1's Orin (JetPack 5.1.1, Ubuntu 20.04, glibc 2.31).
# Ubuntu 20.04 matches the robot's glibc; gcc-10 is the oldest compiler with the C++20 the app uses.
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        g++-10-aarch64-linux-gnu gcc-10-aarch64-linux-gnu binutils-aarch64-linux-gnu \
        cmake ninja-build make pkg-config ca-certificates file \
    && rm -rf /var/lib/apt/lists/*
