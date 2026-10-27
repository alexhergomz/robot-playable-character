# Cross toolchain: build on x86 against a read-only copy of the G1 Orin's root filesystem.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_SYSROOT /sysroot)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc-10)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++-10)
set(CMAKE_FIND_ROOT_PATH /sysroot /deps /src)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
# The robot's libstdc++ is GCC 9's; carry our own so the binary runs unmodified.
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static-libstdc++ -static-libgcc -Wl,--allow-shlib-undefined -L/sysroot/usr/local/cuda-11.4/targets/aarch64-linux/lib -Wl,-rpath-link,/sysroot/usr/lib/aarch64-linux-gnu/tegra:/sysroot/usr/lib/aarch64-linux-gnu:/sysroot/lib/aarch64-linux-gnu:/sysroot/usr/local/cuda-11.4/targets/aarch64-linux/lib")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "-static-libgcc -Wl,-rpath-link,/sysroot/usr/lib/aarch64-linux-gnu/tegra:/sysroot/usr/lib/aarch64-linux-gnu:/sysroot/lib/aarch64-linux-gnu:/sysroot/usr/local/cuda-11.4/targets/aarch64-linux/lib")
