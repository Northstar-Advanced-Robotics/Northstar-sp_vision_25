# sp_vision_25 build environment.
#
# Builds natively for whatever architecture it's run on: pulls the matching
# OpenVINO 2024.6.0 archive (the version hard-coded in this repo's
# CMakeLists files) and compiles the project against apt-installed deps.
#
# NOTE: Intel does not publish an "ubuntu22" *arm64* archive for OpenVINO
# 2024.6.0 (only ubuntu18/ubuntu20 arm64 builds exist for this version) even
# though this image's base OS is Ubuntu 22.04. On aarch64 we therefore pull
# the ubuntu20 arm64 archive instead -- it works fine on Ubuntu 22.04 because
# OpenVINO bundles its own runtime libs and Ubuntu 22.04's glibc/libstdc++
# are backwards-compatible with binaries built against Ubuntu 20.04's older
# ones. See JETSON_ORIN.md for details.
#
# On a Jetson Orin (Ubuntu 22.04 aarch64, JetPack 6): build directly on the
# device with `docker build .` — no cross-compilation needed. See
# JETSON_ORIN.md for why OpenVINO's GPU plugin does NOT work on Jetson's GPU
# (it's Intel-only) — set `device: CPU` in whatever config you deploy there.
#
# ROS2-gated targets (sentry*, publish_test, subscribe_test, topic_loop_test)
# are skipped since ROS2 isn't installed here — see JETSON_ORIN.md §5.3 to add it.
#
# `device: GPU` in the yaml configs means Intel's GPU plugin, which needs
# Intel's actual compute-runtime (intel-opencl-icd / intel-level-zero-gpu),
# not just the generic ICD loader that OpenVINO's own dependency script
# installs. That runtime is only meaningful on x86_64 (Jetson's GPU is
# NVIDIA, unsupported by this plugin regardless), so it's only installed
# there. At *container run time* you still need to give the container access
# to the host GPU:
#   - native Linux host: `docker run --device=/dev/dri ...`
#   - Windows host via Docker Desktop/WSL2: `docker run --device=/dev/dxg
#     -v /usr/lib/wsl:/usr/lib/wsl ...` (see JETSON_ORIN.md appendix)
#
# `device: CUDA` is a second, independent GPU backend for NVIDIA GPUs, since
# OpenVINO's GPU plugin above can never use one. It runs YOLOV5 through ONNX
# Runtime's CUDA execution provider instead of OpenVINO, using a .onnx
# sibling of assets/yolov5.xml (same weights, converted offline -- see
# JETSON_ORIN.md). Both x86_64 (NVIDIA dev-machine GPU) and aarch64 (Jetson's
# own GPU) are supported, but via two different apt repos/package sets,
# because desktop CUDA and Jetson/JetPack CUDA are different builds:
#   - x86_64: a generic CUDA 12.6 runtime is apt-installed from NVIDIA's
#     desktop repo, paired with the generic onnxruntime-linux-x64-gpu
#     release.
#   - aarch64 (Jetson): CUDA/cuDNN/TensorRT are apt-installed from NVIDIA's
#     Jetson/L4T repo instead (repo.download.nvidia.com/jetson), pinned to
#     match the exact versions already on a reference JetPack 6 device
#     (L4T R36.4.7). Confirmed on real hardware that `docker run --runtime
#     nvidia` does NOT bind-mount these in on JetPack 6 -- unlike JetPack
#     4.x, its NVIDIA Container Runtime CSV mounts only cover GPU device
#     nodes and display/graphics libs, not CUDA/cuDNN/TensorRT -- so, same
#     as x86_64, they have to be baked into the image. `--runtime nvidia` is
#     still required at `docker run` time for the GPU device nodes
#     themselves. ONNX Runtime's own libs come from a community aarch64
#     Jetson build (the generic aarch64 onnxruntime-gpu wheel on PyPI has no
#     Tegra support); the matching public C++ headers are pulled straight
#     from the onnxruntime GitHub repo at the same version tag, since
#     prebuilt wheels don't ship C/C++ headers. See JETSON_ORIN.md §5.6.
# At container run time NVIDIA access on Windows/WSL2 comes through the same
# `--device=/dev/dxg -v /usr/lib/wsl:/usr/lib/wsl` as the Intel path above
# (Docker Desktop mirrors both vendors' driver shims there); on native Linux
# x86_64 use `--gpus all` (needs the NVIDIA Container Toolkit); on Jetson use
# `--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all` (see JETSON_ORIN.md §5.6).
#
# `device: TENSORRT` is a third GPU backend, NVIDIA only, using TensorRT's
# own C++ API directly instead of going through ONNX Runtime at all --
# confirmed on real Jetson hardware at ~10-13ms/frame vs `device: CUDA`'s
# ~40-55ms (FP16 + kernel-level autotuning for the exact GPU) -- see
# JETSON_ORIN.md §5.7, including a critical `sudo jetson_clocks` step on the
# host without which that gap mostly disappears. Costs a one-time engine
# build per device (minutes) the first time a config using it runs.
# `libnvinfer-dev`/`libnvonnxparsers-dev` (TensorRT's build-time
# headers/libs) are installed alongside the CUDA runtime above, from the
# same repo (desktop vs Jetson/L4T) and same architecture gating. NOTE:
# on Jetson this is still not quite self-contained -- the Jetson apt repo
# has no package providing CUDA's own C headers (cuda_runtime_api.h etc,
# needed to compile TensorRT API usage), so compiling `device: TENSORRT`
# there additionally requires bind-mounting the host's own
# /usr/local/cuda-12.6 (from JetPack's SDK-Manager flash) at container run
# time -- see JETSON_ORIN.md §5.7 for the full explanation and the exact
# mount flag.

# BASE_IMAGE defaults to plain ubuntu:22.04 (works for both x86_64 dev
# machines and aarch64) so nothing changes for non-Jetson builds. On Jetson,
# override it to NVIDIA's official l4t-jetpack image (aarch64-only, so never
# the default) -- required for NVIDIA VPI's VIC (hardware image processor)
# backend to work inside a container. A plain ubuntu:22.04 base + apt-installed
# nvidia-vpi crashes VIC with a silent SIGABRT during `import vpi`/first use,
# for reasons that didn't trace back to any specific missing package (tried:
# extra privileges/host namespaces, nvidia-vpi + libegl1/libgles2, forcing
# glvnd to only the NVIDIA EGL vendor ICD, libdrm-tegra0 -- none fixed it).
# l4t-jetpack works out of the box with zero extra flags/packages, so we use
# it as the base instead of continuing to bisect blindly against a
# closed-source driver stack. Build on Jetson with, e.g.:
#   docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0 -t sp_vision_25 .
# (pick the l4t-jetpack tag matching the device's L4T release --
# `cat /etc/nv_tegra_release` -- same rule as the r36.4 apt suite below.)
ARG BASE_IMAGE=ubuntu:22.04
FROM ${BASE_IMAGE}

ARG OPENVINO_VERSION=2024.6.0
ARG OPENVINO_SERIES=2024.6
ARG OPENVINO_BUILD=17404.4c0f47d2335
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        g++ \
        cmake \
        make \
        wget \
        ca-certificates \
        pkg-config \
        can-utils \
        libopencv-dev \
        libfmt-dev \
        libeigen3-dev \
        libspdlog-dev \
        libyaml-cpp-dev \
        libusb-1.0-0-dev \
        nlohmann-json3-dev \
        libceres-dev \
        libgl1 \
        libgomp1 \
        openssh-server \
        screen \
    && rm -rf /var/lib/apt/lists/*

# --- OpenVINO 2024.6.0 runtime (matches the version hard-coded in CMakeLists.txt) ---
# Picks the arm64 (ubuntu20 build, see note above) archive on Jetson/aarch64
# hosts, x86_64 (ubuntu22 build) archive otherwise.
RUN set -eux; \
    arch="$(uname -m)"; \
    case "$arch" in \
        aarch64) ov_arch=arm64; ov_os=ubuntu20 ;; \
        x86_64)  ov_arch=x86_64; ov_os=ubuntu22 ;; \
        *) echo "Unsupported architecture: $arch" >&2; exit 1 ;; \
    esac; \
    url="https://storage.openvinotoolkit.org/repositories/openvino/packages/${OPENVINO_SERIES}/linux/l_openvino_toolkit_${ov_os}_${OPENVINO_VERSION}.${OPENVINO_BUILD}_${ov_arch}.tgz"; \
    wget -q "$url" -O /tmp/openvino.tgz; \
    mkdir -p "/opt/intel/openvino_${OPENVINO_VERSION}"; \
    tar -xf /tmp/openvino.tgz -C "/opt/intel/openvino_${OPENVINO_VERSION}" --strip-components=1; \
    rm /tmp/openvino.tgz; \
    ln -s "/opt/intel/openvino_${OPENVINO_VERSION}" /opt/intel/openvino; \
    if [ -x "/opt/intel/openvino/install_dependencies/install_openvino_dependencies.sh" ]; then \
        /opt/intel/openvino/install_dependencies/install_openvino_dependencies.sh -y || true; \
    fi

# --- Intel GPU compute runtime (needed for `device: GPU` in the yaml configs) ---
# x86_64 only: the OpenVINO GPU plugin only ever talks to an Intel GPU, and
# Jetson's GPU is NVIDIA, so there is nothing to install on aarch64.
RUN set -eux; \
    if [ "$(uname -m)" = "x86_64" ]; then \
        apt-get update && apt-get install -y --no-install-recommends gnupg gpg-agent wget ca-certificates; \
        wget -qO - https://repositories.intel.com/gpu/intel-graphics.key \
            | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg; \
        echo 'deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client' \
            > /etc/apt/sources.list.d/intel-gpu-jammy.list; \
        apt-get update; \
        apt-get install -y --no-install-recommends intel-opencl-icd intel-level-zero-gpu level-zero clinfo; \
        rm -rf /var/lib/apt/lists/*; \
    fi

# --- NVIDIA CUDA runtime + ONNX Runtime GPU (needed for `device: CUDA`) ---
# x86_64: cuda-cudart/cublas/cufft/curand/cusparse/cusolver + cudnn9 + nvrtc
# are the minimal runtime pieces ONNX Runtime's CUDA execution provider
# needs (no nvcc/full toolkit required). Pinned to CUDA 12.6 + cuDNN 9, the
# combination ONNX Runtime 1.20.x's GPU build expects; the container only
# needs a driver new enough to run *some* CUDA 12.x, which is what NVIDIA
# driver backward compatibility guarantees.
ARG ONNXRUNTIME_VERSION=1.20.1
RUN set -eux; \
    if [ "$(uname -m)" = "x86_64" ]; then \
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb; \
        dpkg -i /tmp/cuda-keyring.deb; \
        rm /tmp/cuda-keyring.deb; \
        apt-get update; \
        apt-get install -y --no-install-recommends \
            cuda-cudart-12-6 cuda-nvrtc-12-6 libcublas-12-6 libcufft-12-6 \
            libcurand-12-6 libcusparse-12-6 libcusolver-12-6 libcudnn9-cuda-12 \
            libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev; \
        rm -rf /var/lib/apt/lists/*; \
        url="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz"; \
        wget -q "$url" -O /tmp/onnxruntime.tgz; \
        mkdir -p /opt/onnxruntime; \
        tar -xzf /tmp/onnxruntime.tgz -C /opt/onnxruntime --strip-components=1; \
        rm /tmp/onnxruntime.tgz; \
        echo /opt/onnxruntime/lib > /etc/ld.so.conf.d/onnxruntime.conf; \
        ldconfig; \
    fi

# aarch64 (Jetson): CUDA/cuDNN/TensorRT ARE apt-installed directly into the
# image here, from NVIDIA's Jetson/L4T apt repo -- *not* relying on
# `docker run --runtime nvidia` to mount them from the host. Verified against
# a real JetPack 6 device (L4T R36.4.7): as of JetPack 5+, the NVIDIA
# Container Runtime's CSV auto-mount mechanism only covers GPU *device
# nodes* and display/graphics libs (its devices.csv/drivers.csv) -- it does
# NOT mount CUDA/cuDNN/TensorRT anymore (that was the older JetPack 4.x
# behavior). So `--runtime nvidia` is still required at `docker run` time
# for the actual GPU device nodes, but the compute libraries themselves have
# to be baked into the image, same as the x86_64 path above -- just from
# NVIDIA's Jetson repo instead of the desktop one.
#
# Installs the `cuda-libraries-12-6` meta-package (every CUDA runtime math
# library, apt-resolved -- no nvcc/compiler, we don't compile CUDA kernels
# ourselves) rather than a hand-picked subset of runtime libs. A hand-picked
# list (cudart/cublas/cufft/curand/cusparse/cusolver) looked complete but
# caused a hard-to-diagnose SIGSEGV inside the CUDA EP on real Jetson
# hardware -- CUDA 12.x split libnvJitLink out as its own package, a
# transitive runtime dependency of cuBLAS/cuSOLVER that isn't obvious from
# the shared library names alone, and apt's own dependency resolution
# catches this sort of thing where a manually-curated list won't.
# `cuda-libraries-12-6` (confirmed working on real Jetson hardware) is the
# Jetson repo's runtime-only equivalent of the full toolkit's libraries.
# libnvinfer-dev/libnvonnxparsers-dev (TensorRT's headers, for `device:
# TENSORRT`) are included here too -- TensorRT 10.3 was already natively
# installed on the reference device, these packages just add the matching
# development headers on top of it. If you're on a different JetPack/L4T
# version than R36.4.7, check `apt-cache policy cuda-libraries-12-6` on the
# device first -- the L4T apt suite (`r36.4` below) must match
# `cat /etc/nv_tegra_release`.
RUN set -eux; \
    if [ "$(uname -m)" = "aarch64" ]; then \
        wget -qO /etc/apt/trusted.gpg.d/jetson-ota-public.asc https://repo.download.nvidia.com/jetson/jetson-ota-public.asc; \
        chmod 644 /etc/apt/trusted.gpg.d/jetson-ota-public.asc; \
        echo 'deb https://repo.download.nvidia.com/jetson/common r36.4 main' > /etc/apt/sources.list.d/nvidia-l4t-apt-source.list; \
        echo 'deb https://repo.download.nvidia.com/jetson/t234 r36.4 main' >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list; \
        apt-get update; \
        apt-get install -y --no-install-recommends \
            cuda-libraries-12-6 libcudnn9-cuda-12 \
            libnvinfer-dev libnvinfer-plugin-dev libnvonnxparsers-dev \
            vpi3-dev; \
        rm -rf /var/lib/apt/lists/*; \
    fi

# ONNX Runtime for aarch64+CUDA+TensorRT, extracted from a community Jetson
# build (JetPack 6.x / CUDA 12.6 -- Microsoft's official PyPI package has no
# aarch64+Tegra build). Prebuilt wheels don't ship C/C++ headers, so those
# are fetched separately, straight from the onnxruntime source tree at the
# matching version tag.
ARG ONNXRUNTIME_JETSON_VERSION=1.23.0
ARG ONNXRUNTIME_JETSON_WHEEL_URL=https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
RUN set -eux; \
    if [ "$(uname -m)" = "aarch64" ]; then \
        apt-get update && apt-get install -y --no-install-recommends unzip; \
        rm -rf /var/lib/apt/lists/*; \
        wget -q "$ONNXRUNTIME_JETSON_WHEEL_URL" -O /tmp/onnxruntime.whl; \
        mkdir -p /opt/onnxruntime/lib /opt/onnxruntime/include; \
        unzip -q -j /tmp/onnxruntime.whl 'onnxruntime/capi/libonnxruntime*' -d /opt/onnxruntime/lib; \
        rm /tmp/onnxruntime.whl; \
        ln -sf "libonnxruntime.so.${ONNXRUNTIME_JETSON_VERSION}" /opt/onnxruntime/lib/libonnxruntime.so; \
        for h in onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h \
                 onnxruntime_ep_c_api.h onnxruntime_float16.h; do \
            wget -q "https://raw.githubusercontent.com/microsoft/onnxruntime/v${ONNXRUNTIME_JETSON_VERSION}/include/onnxruntime/core/session/$h" \
                -O "/opt/onnxruntime/include/$h"; \
        done; \
        echo /opt/onnxruntime/lib > /etc/ld.so.conf.d/onnxruntime.conf; \
        ldconfig; \
    fi

# libcuda.so itself (the actual GPU driver, as opposed to the CUDA *runtime*
# libs installed above) is never baked into the image -- on native Linux it
# comes from the host's NVIDIA driver install (standard ldconfig path via
# the NVIDIA Container Toolkit), but on Windows/WSL2 it only exists inside
# the /usr/lib/wsl/lib bind mount (see note above), which isn't in the
# dynamic linker's cache since it's not present at build time. Harmless
# no-op on hosts where that path doesn't exist.
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

WORKDIR /root/sp_vision_25
COPY . .

# RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
#     && cmake --build build -j"$(nproc)"

CMD ["/bin/bash"]
