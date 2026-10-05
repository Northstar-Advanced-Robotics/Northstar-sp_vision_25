#!/bin/bash
# Build the sp_vision_25 image on a Jetson Orin, matched to whatever JetPack
# the host is running (JetPack 6 / L4T R36 or JetPack 7 / L4T R38+).
#
# Usage: ./docker_build_jetson.sh [extra docker build args...]
#
# The Dockerfile's Jetson defaults are pinned to JetPack 6 (L4T R36.4, CUDA
# 12.6). Rather than hard-coding a second set of JetPack 7 versions, this
# script reads the host's own:
#   - L4T release            (/etc/nv_tegra_release)
#   - NVIDIA apt sources     (/etc/apt/sources.list.d/nvidia-l4t-apt-source.list)
#   - CUDA install           (/usr/local/cuda -> /usr/local/cuda-X.Y)
# and passes them as build args, so the TensorRT/CUDA packages inside the
# image always match the host's driver stack -- the same reason the r36.4
# suite had to match `cat /etc/nv_tegra_release` before (JETSON_ORIN.md §5.6).
#
# Override the base image with BASE_IMAGE=... ./docker_build_jetson.sh

set -e

IMAGE_NAME="sp_vision_25"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$(uname -m)" != "aarch64" ] || [ ! -f /etc/nv_tegra_release ]; then
  echo "This script is for building on a Jetson (needs /etc/nv_tegra_release)." >&2
  echo "On x86_64 just run: docker build -t ${IMAGE_NAME} ." >&2
  exit 1
fi

# "# R36 (release), REVISION: 4.7, ..." -> major=36, minor=4
L4T_MAJOR="$(sed -n 's/^# R\([0-9]*\) (release), REVISION: \([0-9]*\)\..*/\1/p' /etc/nv_tegra_release)"
L4T_MINOR="$(sed -n 's/^# R\([0-9]*\) (release), REVISION: \([0-9]*\)\..*/\2/p' /etc/nv_tegra_release)"
if [ -z "$L4T_MAJOR" ]; then
  echo "Could not parse L4T version from /etc/nv_tegra_release:" >&2
  head -1 /etc/nv_tegra_release >&2
  exit 1
fi

CUDA_DIR="$(readlink -f /usr/local/cuda 2>/dev/null || true)"
CUDA_VERSION="${CUDA_DIR##*/cuda-}"   # e.g. 12.6 / 13.0
if [ -z "$CUDA_DIR" ] || [ "$CUDA_VERSION" = "$CUDA_DIR" ]; then
  echo "Could not find the host CUDA install (/usr/local/cuda -> /usr/local/cuda-X.Y)." >&2
  echo "Install it with: sudo apt install nvidia-jetpack" >&2
  exit 1
fi
CUDA_MAJOR="${CUDA_VERSION%%.*}"

APT_SOURCES_FILE=/etc/apt/sources.list.d/nvidia-l4t-apt-source.list
if [ -f "$APT_SOURCES_FILE" ]; then
  JETSON_APT_SOURCES="$(grep -v '^\s*#' "$APT_SOURCES_FILE" | grep -v '^\s*$')"
else
  echo "Warning: $APT_SOURCES_FILE not found, falling back to the Dockerfile's r36.4 default." >&2
  JETSON_APT_SOURCES=""
fi

if [ -z "$BASE_IMAGE" ]; then
  if [ "$L4T_MAJOR" = "36" ]; then
    BASE_IMAGE="nvcr.io/nvidia/l4t-jetpack:r36.4.0"
  else
    # JetPack 7 is Ubuntu 24.04 based.
    BASE_IMAGE="ubuntu:24.04"
  fi
fi

# The community onnxruntime-gpu wheel (device: CUDA only) is built against
# CUDA 12 -- skip it on anything else; device: TENSORRT doesn't use it.
ORT_ARGS=()
if [ "$CUDA_MAJOR" != "12" ]; then
  ORT_ARGS=(--build-arg ONNXRUNTIME_JETSON_WHEEL_URL=)
fi

echo "Host: L4T R${L4T_MAJOR}.${L4T_MINOR}, CUDA ${CUDA_VERSION} (${CUDA_DIR})"
echo "Base image: ${BASE_IMAGE}"
[ -n "$JETSON_APT_SOURCES" ] && echo "Apt sources:" && echo "$JETSON_APT_SOURCES" | sed 's/^/  /'
[ "$CUDA_MAJOR" != "12" ] && echo "CUDA ${CUDA_MAJOR}: skipping onnxruntime (device: CUDA unavailable, device: TENSORRT unaffected)"

sudo docker build \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg JETSON_APT_SOURCES="${JETSON_APT_SOURCES}" \
  --build-arg JETSON_CUDA_VERSION="${CUDA_VERSION}" \
  "${ORT_ARGS[@]}" \
  -t "${IMAGE_NAME}" \
  "$@" \
  "${REPO_DIR}"
