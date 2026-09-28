#!/bin/bash
# Build and run an sp_vision_25 executable in one step. For use inside a dev
# container (Jetson or x86_64) -- needs cmake/g++ and whatever else the
# Dockerfile installs, not meant to run on a bare host.
#
# Usage: ./run.sh <target> [args passed through to the binary]
#
# Examples:
#   ./run.sh auto_aim_test --config-path=configs/demo_tensorrt.yaml assets/demo/demo
#   ./run.sh auto_aim_test --config-path=configs/demo_cuda.yaml assets/demo/demo
#   ./run.sh standard --config-path=configs/standard4.yaml
#
# <target> is any CMake target this project defines (auto_aim_test, standard,
# mt_standard, auto_buff_test, camera_test, ... -- see the top-level
# CMakeLists.txt for the full list). Only that target (and whatever it
# depends on) gets built, not the whole project, so this stays fast on
# repeat runs once everything's already compiled once.

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <target> [args passed through to the binary]"
  echo "Example: $0 auto_aim_test --config-path=configs/demo_tensorrt.yaml assets/demo/demo"
  exit 1
fi

TARGET="$1"
shift

if [ ! -d build ]; then
  echo "No build/ directory yet -- configuring..."
  cmake -B build -DCMAKE_BUILD_TYPE=Release
fi

cmake --build build -j"$(nproc)" --target "$TARGET"

exec "./build/${TARGET}" "$@"
