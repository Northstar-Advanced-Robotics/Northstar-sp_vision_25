# sp_vision_25 — Architecture Overview & Jetson Orin Guide

This document explains what this repository does, how it's put together, and what it
takes to run it on an NVIDIA Jetson Orin instead of the Intel NUC it was designed for.
It also documents the accompanying `Dockerfile`, which builds the project without
requiring you to hand-install every dependency.

## 1. What this project is

`sp_vision_25` is Tongji University SuperPower RoboMaster team's 2025-season vision
stack for **autonomous aiming ("auto-aim")** and **rune/buff hitting ("auto-buff")**
on a combat robot. Given a camera feed and IMU/gimbal feedback from an embedded
controller, it:

1. Detects enemy armor plates (or the rotating "buff" rune panel) in each frame.
2. Estimates the target's 3D pose and motion (position, velocity, yaw spin) with an
   Extended Kalman Filter.
3. Plans a firing trajectory ("轨迹规划器") that predicts where the target will be
   after bullet flight time + system latency, accounting for the gimbal's achievable
   acceleration so the aim point is actually reachable.
4. Decides when to fire based on how well the planned gimbal trajectory matches the
   predicted target trajectory.
5. Sends aim/fire commands to a downstream STM32-based controller ("C板") over
   CAN/serial, and reads IMU quaternions back for gimbal-to-world alignment.

It is pure C++ (no ROS required for the core pipeline); ROS 2 is only used
optionally by the "sentry" (哨兵) build variant to talk to a navigation stack.

## 2. Repository layout

```
sp_vision_25
├── assets/          Demo video + neural net weights (ONNX and OpenVINO IR .xml/.bin)
├── calibration/      Camera intrinsic + hand-eye calibration utilities
├── configs/          Per-robot YAML configs (standard3/4, sentry, uav, ascento, ...)
├── io/                Hardware abstraction layer
│   ├── camera.cpp/hpp        Camera facade
│   ├── hikrobot/              HikRobot industrial camera SDK wrapper (amd64+arm64 .so bundled)
│   ├── mindvision/             MindVision industrial camera SDK wrapper (amd64+arm64 .so bundled)
│   ├── usbcamera/               Generic USB (V4L2/UVC) camera support
│   ├── cboard.cpp/hpp           STM32 controller link: CAN + IMU quaternion queue
│   ├── socketcan.hpp            Linux SocketCAN wrapper
│   ├── gimbal/                   Alternate serial-based gimbal driver
│   ├── dm_imu/                    Damiao IMU driver
│   ├── serial/                    Vendored cross-platform serial library
│   └── ros2/                       Optional nav-stack publish/subscribe bridge
├── tasks/
│   ├── auto_aim/       Detector (traditional CV + YOLOv5/v8/v11 via OpenVINO),
│   │                   classifier, EKF tracker, solver (PnP), trajectory planner
│   │                   (incl. vendored TinyMPC QP solver), aimer, shooter, voter
│   ├── auto_buff/       Rune/buff-specific detector, solver, predictor (RANSAC sine
│   │                   fit), aimer — uses Ceres for nonlinear fitting
│   └── omniperception/    Sentry-specific perception/decision helper
├── tools/             Shared utilities: logging (spdlog), YAML config loader,
│                       EKF, PID, CRC, image tools, ballistic trajectory math,
│                       thread-safe queue/pool, video recorder, PlotJuggler plotter
├── src/               Executable entry points (standard.cpp, sentry.cpp, uav.cpp, ...)
├── tests/             Standalone test/debug programs per module
└── CMakeLists.txt     Top-level build; conditionally builds ROS2/sentry targets
```

### Software layering (io → tasks → src)
`io` wraps all hardware (cameras, CAN, serial). `tasks` implements the
perception/decision algorithms as reusable libraries (`auto_aim`, `auto_buff`,
`omniperception`), independent of any specific robot. `src/*.cpp` are thin
per-robot `main()` programs that wire `io` + `tasks` together differently
depending on the robot type (e.g. `standard.cpp` for an infantry robot,
`sentry.cpp` for the ROS2-connected sentry, `uav.cpp` for a drone). `tools` is a
grab-bag of dependency-free helpers used everywhere.

## 3. Build dependencies

| Dependency | Purpose | Version pinned in repo |
|---|---|---|
| OpenCV | Image I/O, traditional CV armor detection | apt `libopencv-dev` (~4.5.4 on Ubuntu 22.04) |
| **OpenVINO** | Neural-net inference (YOLOv5/v8/v11 armor + buff detectors) | **hard-coded to `/opt/intel/openvino_2024.6.0/`** in 3 CMakeLists files |
| Eigen3 | Linear algebra (EKF, geometry) | apt |
| Ceres Solver | Nonlinear least squares (buff rune sine-motion fit) | apt |
| fmt, spdlog | Formatting/logging | apt |
| yaml-cpp | Per-robot YAML config parsing | apt |
| nlohmann-json | JSON (config/telemetry) | apt |
| libusb-1.0 | USB camera SDK transport | apt |
| TinyMPC | Vendored QP solver for trajectory planning | in-tree source (`tasks/auto_aim/planner/tinympc`) |
| HikRobot MVS SDK | Industrial camera driver | **vendored `.so` for amd64 *and* arm64** already in `io/hikrobot/lib/` |
| MindVision SDK | Industrial camera driver (alternate) | **vendored `.so` for amd64 *and* arm64** already in `io/mindvision/lib/` |
| vendored `serial` lib | Cross-platform serial port I/O | in-tree source (`io/serial`) |
| ROS 2 (rclcpp, std_msgs, `sp_msgs`) | Only for the sentry nav-bridge build | optional, `find_package(... QUIET)` — skipped entirely if absent |

The top-level `CMakeLists.txt` already checks `CMAKE_SYSTEM_PROCESSOR` and picks
`hikrobot/lib/amd64` vs `hikrobot/lib/arm64` (and the same for MindVision)
automatically — **this repo was already set up with ARM64 in mind**, it's just
never been fully wired up for a Jetson.

## 4. Can this run on a Jetson Orin? — Short answer: mostly yes, with one real caveat

Jetson Orin boards (Nano/NX/AGX) run **Ubuntu 22.04 aarch64** under JetPack 6, so
architecture-wise this is a native fit, not an emulation hack. Here's the
component-by-component breakdown:

| Component | Jetson Orin status | Notes |
|---|---|---|
| CPU architecture | ✅ Works | CMake already branches on `aarch64` for HikRobot/MindVision libs |
| HikRobot camera SDK | ✅ Works | arm64 `.so` already vendored in-repo |
| MindVision camera SDK | ✅ Works | arm64 `.so` already vendored in-repo |
| Generic USB/V4L2 camera | ✅ Works | no architecture dependency |
| CAN bus (SocketCAN) | ✅ Works | needs a CAN interface — via a Jetson carrier board's onboard CAN controller, or any USB-CAN adapter (`can0`) |
| Serial (STM32 C-board, IMU) | ✅ Works | vendored `serial` lib is a plain POSIX termios wrapper |
| OpenCV / Eigen / Ceres / fmt / spdlog / yaml-cpp / nlohmann-json | ✅ Works | all available as aarch64 apt packages on Ubuntu 22.04 |
| **OpenVINO inference — `device: CPU`** | ✅ Works | Intel ships an official **arm64 Linux archive** for OpenVINO 2024.6.0 — but only built against **Ubuntu 20.04**, not 22.04 (`l_openvino_toolkit_ubuntu20_2024.6.0.17404.4c0f47d2335_arm64.tgz`; verified this downloads correctly). There is no `ubuntu22`+arm64 combination for this OpenVINO version. This is fine in practice: the archive bundles its own runtime libraries, and Ubuntu 22.04's glibc/libstdc++ are backwards-compatible with binaries built against 20.04's older ones, so it runs unmodified on the Jetson's Ubuntu 22.04 (JetPack 6) userspace. The Dockerfile handles this automatically. CPU plugin runs on ARM. |
| **OpenVINO inference — `device: GPU`** | ❌ Does not work | OpenVINO's `GPU` plugin only targets **Intel** GPUs (iGPU/Arc via oneAPI/Level Zero). It has no backend for Jetson's NVIDIA/CUDA GPU. Every shipped config (`standard3.yaml`, `standard4.yaml`, `sentry.yaml`, `mvs.yaml`, `ascento.yaml`) sets `device: GPU` — **you must change this to `device: CPU` or `device: CUDA`** in whatever config you deploy on a Jetson, or inference will fail to find a device. |
| **`device: CUDA` (Jetson's own GPU)** | ✅ Works | Separate ONNX Runtime CUDA backend, not OpenVINO — see §5.6. Only `YOLOV5` supports it (the model every shipped config actually uses). |
| ROS 2 (sentry mode) | ✅ Works, if wanted | JetPack 6 (Ubuntu 22.04) supports ROS 2 Humble same as any x86 Ubuntu 22.04 box. The `sentry*` executables also depend on a custom `sp_msgs` package that isn't in this repo, so full sentry functionality requires that package from elsewhere regardless of platform. |

**Bottom line:** the codebase itself is portable — nothing in the vision pipeline
is x86-specific. GPU-accelerated inference on Jetson's own NVIDIA GPU is
possible via `device: CUDA` (§5.6), a separate backend from OpenVINO's GPU
plugin (which is Intel-only and simply won't work here). `device: CPU`
(OpenVINO's CPU plugin) remains available as a simpler fallback with no
JetPack/NVIDIA Container Runtime setup required.

Also note the `assets/*.xml`/`*.bin` files are OpenVINO IR models exported for
FP32/INT8 — these load fine on the CPU plugin, no re-export needed to just get
it running (only needed if you later want max CPU-inference speed via
re-quantization, or if you build the TensorRT path).

## 5. Using the Dockerfile

The included `Dockerfile` builds an aarch64 image with every apt-installable
dependency plus OpenVINO 2024.6.0 (matching the version hard-coded in the repo's
CMakeLists files), and compiles the project. It's architecture-aware: build it
directly **on** a Jetson Orin (or any aarch64 host) and it pulls the arm64
(ubuntu20-packaged, see §4) OpenVINO archive automatically; build it on an
x86_64 dev machine and it pulls the x86_64 (ubuntu22-packaged) archive instead.
**This has been built and verified end-to-end for the x86_64 path** (full
compile of every non-ROS2 target succeeds); the arm64 download path was
verified to fetch a valid archive but has not been build-tested on real Jetson
hardware — see §6.

It does **not** install ROS 2 — the ROS2-gated targets (`sentry`, `sentry_bp`,
`sentry_debug`, `sentry_multithread`, `publish_test`, `subscribe_test`,
`topic_loop_test`) are simply skipped by CMake (this is the existing, intended
behavior — see the `find_package(... QUIET)` guard in `CMakeLists.txt`). Every
other executable (`standard`, `mt_standard`, `auto_aim_test`, `auto_buff_test`,
calibration tools, etc.) builds normally. If you need the sentry/ROS2 build,
see §5.3 below.

### 5.1 Build the image

On the Jetson Orin itself (recommended — no cross-compilation needed):

```bash
docker build -t sp_vision_25 .
```

From an x86_64 machine, cross-building for the Jetson via buildx:

```bash
docker buildx build --platform linux/arm64 -t sp_vision_25:arm64 --load .
```

### 5.2 Run it

The container needs access to the camera (USB), the CAN interface, and/or the
serial port for the C-board, depending on what your robot config uses. The
simplest way to get all of that is `--privileged` with the host network and
device tree bind-mounted; for tighter scoping, pass individual `--device` flags
instead:

```bash
docker run --rm -it \
  --privileged \
  --network host \
  -v /dev:/dev \
  -v "$(pwd)/configs:/root/sp_vision_25/configs" \
  -v "$(pwd)/logs:/root/sp_vision_25/logs" \
  sp_vision_25 \
  ./build/standard --config-path=configs/standard4.yaml
```

Note the `--config-path=` (with `=`) — `auto_aim_test`/`standard`/etc. use
OpenCV's `CommandLineParser`, which requires `--flag=value` or `-c=value`;
a bare `configs/standard4.yaml` positional argument is silently interpreted
as something else entirely (for `auto_aim_test`, the input *video* path) and
the config path quietly falls back to its default instead of erroring. Also
note `auto_aim_test` specifically replays a recorded video
(`assets/demo/demo.avi`) — for a live camera/CAN/serial robot like this
example implies, use `./build/standard` instead.

Remember to edit whichever config you mount in to set `device: CPU` or
`device: CUDA` (see §4) — every shipped config defaults to `device: GPU`,
which does not work on Jetson. For `device: CUDA`, add
`--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all` to the `docker run` command
above (see §5.6).

**For iterative dev work on a Jetson with a monitor attached** (GPU +
`cv::imshow` video output on the Jetson's own display + the CUDA-headers
mount §5.7's `device: TENSORRT` needs + persistent build/engine cache across
sessions), use `./docker_enter_jetson.sh` instead of building the `docker
run` command by hand — see the script's own header comment for what each
piece does and why (in particular, why it uses a mounted `.Xauthority`
instead of `xhost`).

**For a pure-SSH session with no display at all** (no monitor, no Xvfb, not
even a logged-in desktop) — `./docker_enter_jetson.sh --headless` skips the
`$DISPLAY`/`.Xauthority` checks/warnings entirely, and when creating a
*fresh* container, skips wiring up X11 (env vars + `.Xauthority`/
`/tmp/.X11-unix` mounts) at all — you get a plain interactive shell in the
container, ready to build and run the `--headless`-capable test binaries
(§6.2/§6.4) with zero X11 dependency. A container created this way can't
show `cv::imshow` windows later without being recreated via a normal
(non-`--headless`) run of the script from a session that does have a
working display — see §6.5 for a lighter one-off alternative if the
container was already created with the X11 mounts from an earlier
non-headless run.

### 5.3 Adding ROS 2 / sentry support (optional)

This repo's sentry executables require a `sp_msgs` ROS2 package that isn't
vendored here. If you have access to it and want the sentry build:

1. Start from `FROM ros:humble-ros-base` (arm64 manifest available) instead of
   plain `ubuntu:22.04`, or `apt install ros-humble-ros-base` on top of this
   Dockerfile.
2. Bring in `sp_msgs` (and any other custom message packages your nav stack
   needs) as an additional `COPY` + `colcon build`, sourced before the CMake
   configure step so `find_package(sp_msgs)` succeeds.
3. Re-run `cmake -B build && cmake --build build` — the `sentry*` targets will
   then be included automatically since the CMake ROS2 guard will pass.

### 5.4 Intel GPU acceleration on Windows (Docker Desktop / WSL2)

Not a Jetson topic, but the same `device: GPU` OpenVINO setting from §4, so
it's documented here: if you're developing on a Windows machine with an Intel
iGPU (not the Jetson itself), the Dockerfile now installs Intel's actual GPU
compute runtime (`intel-opencl-icd`, `intel-level-zero-gpu`) — the OpenVINO
dependency script alone only installs the generic OpenCL ICD loader, which is
not enough for the GPU plugin to find a device.

Even with the runtime installed, the **container still needs the host GPU
device passed through at run time**:

- **Native Linux host:** `docker run --device=/dev/dri ...`
- **Windows host via Docker Desktop/WSL2:** there is no `/dev/dri` in Docker
  Desktop's WSL VM. GPU access instead goes through the WSL GPU
  paravirtualization device `/dev/dxg` plus the vendor driver shim libraries
  Docker Desktop mirrors from the host under `/usr/lib/wsl/lib`:
  ```
  docker run --device=/dev/dxg -v /usr/lib/wsl:/usr/lib/wsl ...
  ```
  `.devcontainer/devcontainer.json` in this repo already sets this up via
  `runArgs`/`mounts`, so opening this repo in VS Code's Dev Containers on this
  machine gets GPU access automatically — rebuild the container
  (**Dev Containers: Rebuild Container**) to pick it up.

This was verified end-to-end: `auto_aim_test` run against `configs/demo.yaml`
(`device: GPU`) inside the container, with the host's Intel UHD Graphics iGPU
passed through this way, ran YOLO inference at ~9-10ms/frame with no
OpenVINO device-not-found error. Note this only works for an **Intel** GPU —
an NVIDIA GPU on the same Windows machine is irrelevant here, since OpenVINO's
GPU plugin never talks to NVIDIA hardware (same restriction as Jetson in §4).

### 5.5 NVIDIA GPU acceleration (`device: CUDA`)

Since OpenVINO's `GPU` device can never use an NVIDIA card, NVIDIA GPU
acceleration is a **second, independent inference backend**: setting
`device: CUDA` in a yaml config makes `YOLOV5` (the only model any shipped
config actually uses — see below) run through **ONNX Runtime's CUDA
execution provider** instead of OpenVINO, while every other class
(`YOLO11`/`YOLOV8`) and every other part of the pipeline (tracker, aimer,
solver, ...) is completely unchanged.

**Why not just point OpenVINO at the NVIDIA GPU?** There's a community
`nvidia_plugin` for OpenVINO (in `openvino_contrib`), but it's pinned to
OpenVINO **2024.1.0** (this repo hard-codes 2024.6.0 — plugins aren't
ABI-compatible across versions), needs OpenVINO built from source plus exact
CUDA 11.8 / cuDNN 8.6.0 / cuTENSOR 1.6.1 versions, and isn't part of Intel's
distribution. Too fragile to depend on. ONNX Runtime's CUDA EP is the
actively maintained, officially supported way to run a model on an NVIDIA GPU
outside OpenVINO.

**Where `assets/yolov5.onnx` came from:** OpenVINO can convert *into* IR from
ONNX, but not back out — there's no supported IR→ONNX path. `assets/yolov5.onnx`
was produced with the third-party
[`openvino2onnx`](https://pypi.org/project/openvino2onnx/) tool
(`python -m openvino2onnx assets/yolov5.xml assets/yolov5.onnx`) from the
exact same `assets/yolov5.xml`/`.bin` weights the OpenVINO path uses — not a
retrained or re-exported model. This was numerically validated (not just
"it loads"): both the IR and the resulting ONNX were run on identical real
video frames through OpenVINO and ONNX Runtime respectively, and the outputs
matched to ~1e-3 (bounding boxes, class scores, and the top detection all
identical). The same approach was tried for `yolo11.xml`, but the converted
model failed ONNX Runtime's own load-time shape inference (a YOLO11-specific
op in its detection head didn't survive conversion) — since no shipped
config actually uses `yolo_name: yolo11` or `yolov8` (every config uses
`yolov5`), this wasn't pursued further. Extending `device: CUDA` to those
classes would mean either fixing that conversion or sourcing the original
pre-IR `.onnx`/`.pt` weights, and duplicating the small amount of
CUDA-backend plumbing added to `yolov5.hpp`/`.cpp` into `yolo11.hpp`/`.cpp`
and `yolov8.hpp`/`.cpp`.

**Build-time requirements** (all x86_64-only, added to the Dockerfile):
CUDA 12.6 runtime + cuDNN 9 (`cuda-cudart-12-6`, `libcublas-12-6`, etc., via
NVIDIA's apt repo — not the full CUDA toolkit, no `nvcc` needed) and the
prebuilt `onnxruntime-linux-x64-gpu` release, unpacked to `/opt/onnxruntime`.
`tasks/auto_aim/CMakeLists.txt` looks for it there; if it's missing, the
project still builds fine, just without `device: CUDA` support (attempting
to use it at runtime throws a clear error instead of silently falling back).

**Run-time GPU passthrough** uses the same mechanism as §5.4: on this
Windows/WSL2 machine, `--device=/dev/dxg -v /usr/lib/wsl:/usr/lib/wsl` covers
*both* the Intel and NVIDIA driver shims Docker Desktop mirrors there (no
separate `--gpus all` needed — the existing `devcontainer.json` config
already provides NVIDIA access as-is). On native Linux, use `--gpus all`
(requires the NVIDIA Container Toolkit on the host).

This was verified end-to-end on the actual dev machine's NVIDIA GPU: with
`configs/demo_cuda.yaml` (a copy of `demo.yaml` with `device: CUDA`),
`auto_aim_test` logged `YOLOV5: using ONNX Runtime CUDA backend`, then ran
250+ frames at ~5-19ms per yolo+tracker+aimer cycle (after an expected
~1s first-frame CUDA/cuDNN JIT warmup) with no errors.

**Not applicable to Jetson.** The CUDA runtime this installs is the generic
x86_64 build; Jetson needs JetPack/L4T's own CUDA build tied to its embedded
driver — see §5.6 for the Jetson-specific `device: CUDA` install instead.

### 5.6 NVIDIA GPU acceleration on Jetson (`device: CUDA`)

Jetson's GPU **is** NVIDIA, so unlike the x86_64 Intel-vs-NVIDIA split in
§5.4/§5.5, this is the natural way to get real GPU acceleration on a Jetson
— but it needs a different install than §5.5's x86_64 path, because
Jetson's CUDA/cuDNN/TensorRT are JetPack/L4T-specific builds tied to
whatever's flashed on the device, not a generic desktop CUDA install.

**How it works, reflecting what a real device (JetPack 6, L4T R36.4.7, CUDA
12.6.68, cuDNN 9.3.0, TensorRT 10.3) showed, including one wrong assumption
this section originally made and had to correct after testing on hardware:**

- **First attempt (wrong):** assumed CUDA/cuDNN/TensorRT should stay off the
  image and get bind-mounted in from the host by the NVIDIA Container
  Runtime at `docker run --runtime nvidia` time — this is genuinely how it
  worked on older JetPack 4.x, and is exactly how the *Intel* GPU path in
  §5.4 works via `/dev/dxg` + `/usr/lib/wsl/lib`. On this JetPack 6 device it
  doesn't apply: `cat /etc/nvidia-container-runtime/config.toml` showed
  `mode = "auto"`, and the only CSV manifests present under
  `/etc/nvidia-container-runtime/host-files-for-container.d/` were
  `devices.csv` (GPU/display device nodes) and `drivers.csv` (display/weston
  libs) — no `cuda.csv`/`cudnn.csv`/`tensorrt.csv`. Confirmed by running with
  `--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all`
  and finding `/usr/local/cuda*/lib64/libcublas*` simply didn't exist inside
  the container. As of JetPack 5+, NVIDIA moved to baking CUDA/cuDNN/TensorRT
  into the container image itself instead (the CSV auto-mount mechanism now
  only covers device nodes and display libs).
- **What actually works:** CUDA/cuDNN are apt-installed straight into the
  image, same as the x86_64 path, just from NVIDIA's **Jetson/L4T** apt repo
  (`repo.download.nvidia.com/jetson/{common,t234}`, suite `r36.4`) instead of
  the desktop one, using the `jetson-ota-public.asc` signing key, and pinned
  to `cuda-cudart-12-6`/`libcudnn9-cuda-12`/etc. — the exact package names
  and versions confirmed already present on the reference device via
  `apt-cache policy`. `--runtime nvidia -e NVIDIA_VISIBLE_DEVICES=all` is
  still required at `docker run` time, just for a narrower purpose now: the
  actual GPU device nodes (`/dev/nvhost-gpu`, `/dev/nvmap`, etc. from
  `devices.csv`), which genuinely do still come from the host, not the image.
- ONNX Runtime's own libraries (`libonnxruntime.so`, plus its CUDA and
  TensorRT execution provider `.so`s) are separate from the CUDA/cuDNN
  install above — extracted from a community-built Jetson wheel
  ([`ultralytics/assets`](https://github.com/ultralytics/assets/releases),
  onnxruntime-gpu 1.23.0 for JetPack 6/CUDA 12.6 — Microsoft's official PyPI
  package has no aarch64+Tegra build). Prebuilt wheels don't ship C/C++
  headers, so the 5 headers `onnxruntime_cxx_api.h` actually needs are
  pulled straight from the `microsoft/onnxruntime` GitHub repo at the
  matching `v1.23.0` tag.

**If you're on a different JetPack/L4T version than R36.4.7**, the apt
suite pinned in the Dockerfile (`r36.4`) needs to match — check
`cat /etc/nv_tegra_release` and `apt-cache policy cuda-cudart-12-6` on the
device and adjust the `r36.4` in the Dockerfile's Jetson CUDA block
accordingly.

**One more bug, found and fixed on real hardware: an onnxruntime CUDA EP
output-retrieval crash.** After fixing the CUDA/cuDNN install (above), a new
SIGSEGV showed up — not at startup, but on the very first inference call.
Bisected with fine-grained tracing added directly to `infer_cuda()` (rebuilt
in-place inside a live container, much faster than a full image rebuild per
iteration) down to one exact line: `ort_session_->Run(...)` completes
successfully — correct output shape `[1, 25200, 22]`, correct element count,
`IsTensor()` true — but calling `GetTensorData<float>()` on the
auto-allocated output `Ort::Value` segfaults. Reproduced in an ~18-line
standalone program outside the whole project, confirmed identical
(byte-for-byte matching library, via `md5sum`) whether the onnxruntime wheel
came from `ultralytics/assets` or `pypi.jetson-ai-lab.io` — this is a bug in
the wheel's build itself, not a preprocessing or project-code issue. Tried
switching to the bundled TensorRT execution provider instead, which hit a
different problem (GPU memory allocation failure during engine autotuning —
plausibly just this device's limited shared memory) and wasn't pursued
further. The fix that worked: **explicit `Ort::IoBinding` with a
caller-owned CPU output buffer**, instead of `Run()`'s simple API which lets
the CUDA EP allocate the output value itself — sidesteps whatever's broken
in that default allocation path entirely. Verified via the same standalone
repro before applying to `yolov5.cpp`'s `infer_cuda()`.

**Status: confirmed working end-to-end on real Jetson Orin hardware.**
`auto_aim_test` against `configs/demo_cuda.yaml` ran cleanly through 100+
frames after the IOBinding fix, all `[N] yolo: ...` log lines present, no
crash — at roughly 40-55ms/frame. (One gotcha hit along the way, worth
recording: after fixing the source, a rebuilt image still showed the old
crash twice in a row — traced to the *host's* working-tree copy of
`yolov5.cpp` having uncommitted local edits that `docker build` picks up
over whatever's committed to git, since it builds from the working
directory, not `HEAD`. `git status`/`git diff` on the host is the first
thing to check if a Dockerfile-based rebuild doesn't seem to reflect a
just-pushed fix.)

### 5.7 TensorRT backend (`device: TENSORRT`) — faster than `device: CUDA`

~40-55ms/frame from §5.6 is fine for a first working GPU path, but slow for
real-time auto-aim, and generic CUDA-EP execution isn't why you'd reach for
a Jetson in the first place. `device: TENSORRT` is a third YOLOV5 backend,
using TensorRT's own C++ API directly (`NvInfer.h`/`NvOnnxParser.h`) instead
of going through ONNX Runtime at all — sidesteps both the output-retrieval
bug in §5.6 (different library entirely) and ONNX Runtime's own TensorRT
execution provider, which hit a GPU-memory allocation failure during engine
autotuning when tried as part of debugging §5.6 (not pursued further, since
the native API avoids it and is the standard, most-tuned path anyway — the
same approach the
[Ultralytics Jetson guide](https://docs.ultralytics.com/guides/nvidia-jetson)
uses).

**How it works:** same `assets/yolov5.onnx` as `device: CUDA` (§5.5/§5.6),
but on first use with `device: TENSORRT`, `YOLOV5`'s constructor builds a
TensorRT engine from it in-process (via `IBuilder`/`INetworkDefinition`/
`nvonnxparser::IParser`, FP16 enabled when the platform supports it) and
caches the serialized result to `assets/yolov5.engine` — subsequent runs
just deserialize that cached file (fast) instead of rebuilding. **The
engine file is tied to the exact GPU + TensorRT + CUDA version it was built
on** (unlike the portable `.onnx`/`.xml` models) — it won't load on a
different device or after a TensorRT/JetPack upgrade; delete the stale
`.engine` file to force a rebuild (the code detects a failed deserialize and
says so in the error message). The first run on any given device will be
slow (engine building/autotuning can take several minutes) — that delay is
a one-time cost per device, not per container restart, as long as
`assets/` (or wherever the `.engine` ends up) persists across runs (e.g.
via the same bind-mounted `configs`/`assets` pattern already used
elsewhere, or simply not deleting/recreating the container).

**Build-time requirements** (both x86_64 and aarch64, added to the
Dockerfile alongside the `device: CUDA` CUDA/cuDNN install): TensorRT's
development headers/libs, `libnvinfer-dev`/`libnvinfer-plugin-dev`/
`libnvonnxparsers-dev`. On the reference Jetson device, TensorRT 10.3's
*runtime* was already present (JetPack ships it), these packages just add
the matching dev headers on top. `tasks/auto_aim/CMakeLists.txt` looks for
`NvInfer.h`/`libnvinfer`/`libnvonnxparser` under both Debian/Ubuntu
multiarch triplet paths (`/usr/include/aarch64-linux-gnu`,
`/usr/lib/aarch64-linux-gnu`, and the `x86_64` equivalents) as well as
plain `/usr/include`/`/usr/lib` — apt installs these under the multiarch
paths, which CMake's bare `find_path`/`find_library` doesn't reliably
search without an explicit hint. If TensorRT isn't found, the project still
builds fine, just without `device: TENSORRT` support (same graceful-absence
pattern as `device: CUDA`'s onnxruntime detection).

**A real gap the Dockerfile doesn't currently close: CUDA's own C headers
(`cuda_runtime_api.h`), needed to compile TensorRT's API usage, aren't
installed by anything in the Jetson apt repo.** Unlike desktop CUDA repos,
which split a `cuda-cudart-dev-*` header-only package out from the runtime,
the Jetson/L4T repo used here has no such package -- confirmed via
`apt-cache search cuda-cudart` (only the runtime `cuda-cudart-12-6` exists,
and it ships zero `.h` files) and `apt-cache policy cuda-toolkit-12-6`
(doesn't exist either). What worked: the Jetson **host** already has a
complete `/usr/local/cuda-12.6/include` (from JetPack's own SDK-Manager
flash, entirely separate from anything apt-installed inside the container) —
bind-mounting that in at container run time, same pattern as the
Windows/WSL2 `/usr/lib/wsl` mount elsewhere in this doc, closes the gap:
```
docker run ... -v /usr/local/cuda-12.6:/usr/local/cuda-12.6:ro ...
```
This is a workaround, not a real fix baked into the Dockerfile — if you're
on a different JetPack/L4T version, adjust the `12.6` to match, and if the
host's own CUDA install is somehow incomplete this won't help. A more
robust fix (not yet done) would find or build a genuine apt-installable
header package for this repo, or vendor the small set of needed headers
directly into the image.

**Status: confirmed working end-to-end on real Jetson Orin hardware,
including the expected large speedup over `device: CUDA`.** With the CMake
multiarch paths and CUDA-headers bind-mount above,
`configs/demo_tensorrt.yaml` ran cleanly with `platformHasFastFp16=1`
(FP16 genuinely active) and a cached `assets/yolov5.engine` reused across
runs. Per-stage timing (added temporarily, since removed) broke the
`infer_tensorrt()` call down as preprocess ~1.3ms, host-to-device copy
~0.7ms, TensorRT inference ~3.4ms, device-to-host copy ~0.5ms — full
`yolo:` (inference + postprocess/NMS) landed at **~10-13ms/frame**, versus
`device: CUDA`'s ~40-55ms. That's roughly the 4-5x speedup expected from
FP16 + kernel autotuning, once one more thing was fixed:

**Critical, easy-to-miss step: run `sudo jetson_clocks` on the host.**
Initial TensorRT numbers were a disappointing ~28-35ms/frame -- barely
faster than `device: CUDA` -- even with the power mode already at
`MAXN_SUPER` (`nvpmodel -q`). `MAXN_SUPER` just permits maximum clocks;
it does not *hold* them there, so short bursty GPU workloads (exactly what
per-frame inference is) can spend real time at reduced clock speed while
DVFS ramps up. `sudo jetson_clocks` locks clocks at their maximum and
disables that ramping -- after running it, `infer` time alone dropped from
~10-15ms to ~3.3-4.9ms with no code changes. This isn't specific to
`device: TENSORRT` (it would help `device: CUDA` too), but it's the kind of
thing worth doing *before* concluding a GPU backend "isn't much faster" on
Jetson -- verify clocks are actually locked first.

#### GPU-side preprocessing and fused NMS

Two further optimizations on top of the baseline above, both specific to
`device: TENSORRT`:

**GPU-resident preprocessing** (`YOLOV5::infer_tensorrt_gpu_preprocess()`,
`preprocess_kernel.cu`): the CPU letterbox resize + `cv::dnn::blobFromImage`
chain (~5ms combined) is replaced by a raw-frame memcpy+H2D followed by a
single custom CUDA kernel fusing letterbox resize + BGR→RGB + normalize +
HWC→CHW, running on the GPU (measured sitting at 15-30% average utilization
otherwise -- plenty of idle headroom). Cut single-threaded `detect()` from
~11.6ms to ~6.7ms.

**Fused NMS** (`tasks/auto_aim/yolos/trt_engine.hpp`,
`scripts/onnx/fuse_efficient_nms.py`): NMS suppression itself now runs
inside the TensorRT engine via NVIDIA's own `EfficientNMS_TRT` plugin,
instead of a 25200-row CPU scan (`YOLOV5::parse()`) every frame.
`assets/yolov5.onnx` was modified in place by
`scripts/onnx/fuse_efficient_nms.py` -- purely additive: the original
`output/sink_port_0` output ([1,25200,22]) is untouched, four more outputs
are added (`num_detections` [1,1] int32, `detection_boxes` [1,64,4],
`detection_scores` [1,64], `detection_classes` [1,64] -- all FIXED shape,
padded to `kMaxNmsOutputBoxes`). C++ recovers which of the 25200 original
rows each surviving detection came from by matching `detection_scores`
against the original output's own objectness column (nearest-match, not
exact equality -- the plugin's internal sigmoid isn't guaranteed
bit-identical to the host-side one), then reads that row's keypoints/color/
class directly -- no `cv::dnn::NMSBoxes` call needed for the TensorRT path
anymore (`YOLOV5::parse_from_efficient_nms()`).

**This supersedes an earlier attempt using the ONNX-standard
`NonMaxSuppression` op** (`scripts/onnx/fuse_nms.py`, `git log` history --
the script is no longer used but left in the repo for reference/rollback).
That approach produced a dynamically-shaped `selected_indices` output,
which forced TensorRT's data-dependent-shape (DDS) machinery
(`nvinfer1::IOutputAllocator`) -- and measurement after full production
integration found `enqueueV3()` became a genuinely blocking call under it
(~6.5ms, essentially the full GPU compute time, instead of near-instant),
which ate most of the intended win and regressed the async ring buffer's
throughput (140.6fps -> 117.3fps). Root cause, confirmed via the TensorRT
changelog: **TensorRT 10.0-10.7 has a documented, NVIDIA-acknowledged
performance regression for DDS networks** ("Fixed performance regression
for TensorRT 10.x compared to TensorRT 8.6 for networks involving
data-dependent shapes (for example, non-max suppression or non-zero
operations)" -- 10.8.0 release notes). This Jetson is pinned to TensorRT
10.3.0 via JetPack 6.1/6.2 (both ship 10.3; no in-place upgrade carries the
fix -- only JetPack 7.2, TensorRT 10.16.2, which is a full L4T major-version
reflash, not attempted here). `EfficientNMS_TRT`'s fixed-shape outputs
sidestep DDS entirely rather than waiting on that fix, at the cost of the
row-recovery step described above (EfficientNMS_TRT doesn't return the
original row index, only derived box/score/class, and this model needs the
original row's actual keypoints -- not just an axis-aligned box -- for the
armor-corner PnP solve).

**If `assets/yolov5.onnx` is ever regenerated** (e.g. a retrained model
re-exported via `openvino2onnx`, see §5.5/§5.6 above for that tool), re-run
`python scripts/onnx/fuse_efficient_nms.py` afterward to re-apply the NMS
fusion (it refuses to run on an already-fused graph -- start from a fresh
`openvino2onnx` export, or `scripts/onnx/yolov5_prefusion_backup.onnx`, a
clean pre-fusion copy checked in for exactly this), and delete any local
`assets/yolov5.engine` so it gets rebuilt from the newly-fused onnx
(`trt_build_or_load_engine()` has no staleness check against the `.onnx` it
was built from -- it'll happily load a stale cached engine with the old I/O
layout, which then fails loudly with an "unexpected fused-NMS layout" error
at construction, not silently). `scripts/onnx/requirements.txt` pins the
exact `onnx`/`onnx_graphsurgeon` versions validated against this graph's
specific op mix -- other versions were found to break during development
(missing/renamed internal APIs).

**Measured impact:** GPU compute time for detection + NMS combined averages
~3.7ms (`EfficientNMS_TRT` itself costs ~0.45ms on top of plain inference,
confirmed via `trtexec --dumpProfile`; no DDS sync barriers appear in the
per-layer profile, unlike the earlier approach) -- essentially the same
total as with no NMS fusion at all. CPU-side `[PARSE-TIMING]` for the
TensorRT path dropped from ~1.4-1.8ms (25200-row scan) to a single cheap
sigmoid pass plus at most `kMaxNmsOutputBoxes` linear score-matching scans.

**Gotcha: `EfficientNMS_TRT` (and any other TensorRT plugin) needs
`initLibNvInferPlugins()` called before it can be parsed OR deserialized.**
`trtexec` does this automatically at startup, so a graph that parses/builds/
runs fine under `trtexec` can still fail under this project's own use of the
TensorRT API with `IPluginRegistry::getPluginCreator: ... Cannot find
plugin: EfficientNMS_TRT` / `Cannot deserialize plugin since corresponding
IPluginCreator not found`. Fixed in `trt_build_or_load_engine()`
(`trt_engine.cpp`) -- one `initLibNvInferPlugins(&logger, "")` call before
either the cached-engine-load or fresh-build path (needed for both: a
cached `.engine`'s plugin layers look the creator up by name at
deserialize time too, not just at parse time). Needs linking
`libnvinfer_plugin` (`CMakeLists.txt`'s `TENSORRT_NVINFER_PLUGIN_LIB`),
separate from `libnvinfer` itself.

## 6. Testing and benchmarking on the Jetson

Get an interactive shell in the dev container first, if you don't have one
already: `./docker_enter_jetson.sh` (needs a working display -- see §5.2)
or, for a plain SSH session with no display of any kind,
`./docker_enter_jetson.sh --headless` (§5.2) -- everything below assumes
you're running from inside the container, at the repo root
(`/root/sp_vision_25`).

### 6.1 Test binaries at a glance

All three video-test binaries below now take a **`--headless`** flag
(added this session): skips all `imshow`/`waitKey`, and passes
`debug=false` down so even the internal `YOLOV5`/`Detector` debug windows
never open -- no X server, Xvfb, or desktop session needed at all, works
with `DISPLAY`/`XAUTHORITY` completely unset. Without it, all three show
GUI windows by default and **will crash hard**
(`cv::Exception: Can't initialize GTK backend`) if entered via
`docker_enter_jetson.sh --headless` (§5.2) or any other session with no X
server reachable -- always pass `--headless` in that situation.

- **`detector_video_test`**: single-threaded, synchronous
  `YOLO::detect()` over a video file. Without `--headless`, shows the
  internal `YOLOV5` debug window (`imshow("detection", ...)`), throttled to
  ~30fps display via `waitKey(33)`.
  ```
  ./build/detector_video_test --headless --config-path=configs/demo_tensorrt.yaml assets/demo/demo.avi
  ```
- **`mt_detector_video_test`**: exercises the async ring-buffered
  `MultiThreadDetector::push()`/`debug_pop()` path over the same video.
  Preloads every frame into RAM up front so video-file decode time doesn't
  pollute the timing numbers (a real camera wouldn't pay that cost either).
  Same invocation as above, plus its own additional flags -- see §6.2.
- **`auto_aim_test`**: the full pipeline -- `YOLO::detect()` +
  `Tracker::track()` + `Aimer::aim()` -- over a video **and a matching
  ground-truth timestamp/orientation `.txt` file**. **Path is given WITHOUT
  the `.avi` extension** (it appends `.avi`/`.txt` itself) -- differs from
  the two binaries above, which want the full filename:
  ```
  ./build/auto_aim_test --headless --config-path=configs/demo_tensorrt.yaml assets/demo/demo
  ```
- **`mt_standard`/`mt_auto_aim_debug`**: the real, camera/gimbal-driven
  production binaries -- never preload frames (read one at a time from a
  live camera), no `--headless` flag (they don't show any GUI window at
  all, `debug=false` in production configs). The `tools::Stats`
  `[STATS]`/`[MT-THROUGHPUT]` logging (§6.3) is baked into the shared
  `YOLOV5`/`MultiThreadDetector` code these link against too, so it
  surfaces automatically in their own logs -- no separate test tool needed
  to get real production timing distributions.

**Also fixed while adding this:** `Detector::detect(const cv::Mat&, int)`
(the traditional-CV full-frame scan, `--tradition`/`use_traditional`) had
an `imshow("binary_img", ...)` that was never gated by its own `debug_`
flag at all (unlike its sibling `show_result()` call right below it in the
same function) -- a pre-existing bug, invisible until `--headless` had no
X server to silently paper over it. Now properly gated.

### 6.2 `mt_detector_video_test` flags

- **`--sync`**: runs the same preloaded frames through the plain
  single-threaded `YOLO::detect()` path instead of the async ring buffer --
  same engine, same frames, no threading -- for a direct, decode-excluded,
  apples-to-apples baseline against the async path.
- **`--realtime`**: paces the internal `YOLOV5` debug window's display to
  the source video's own recorded fps (`cv::VideoCapture::CAP_PROP_FPS`,
  falls back to 30 if unreported), matching `detector_video_test`'s/
  `auto_aim_test`'s `waitKey(30-33)` convention. Default (no flag): fully
  unthrottled, for stress-testing max throughput -- what all the fps
  numbers in this doc use unless stated otherwise.
- **`--headless`**: skips **all** GUI -- no `imshow`, no `waitKey` -- and
  constructs the detector with `debug=false` so even the internal `YOLOV5`
  debug window never opens. No X server, no Xvfb, no desktop session of any
  kind needed; works with `DISPLAY`/`XAUTHORITY` completely unset. This is
  now the recommended way to benchmark over SSH -- see §6.4 for why it
  matters far more than just being convenient.
- **`--max-frames=N`** (note: `=`, OpenCV's `cv::CommandLineParser` doesn't
  accept a space-separated value): caps how many frames get preloaded, to
  reduce memory footprint. Turned out not to matter much once `--headless`
  is used (§6.4) -- the desktop session's own memory pressure was the
  dominant effect, not the size of the preloaded video buffer.

Example, the fully headless/unthrottled stress-test invocation used to
produce the numbers in §6.4:
```
ssh <jetson> "docker exec <container> bash -c 'unset DISPLAY; unset XAUTHORITY; cd /root/sp_vision_25 && ./build/mt_detector_video_test --headless --config-path=configs/demo_tensorrt.yaml assets/demo/demo.avi'"
```

### 6.3 `tools::Stats` -- per-metric running mean/stddev/outliers

New utility (`tools/stats.hpp`/`.cpp`) for exactly the question "the
per-frame numbers vary a lot -- by how much, and which frames are the
outliers?" that eyeballing individual log lines can't answer. Welford's
online algorithm (mean/stddev without retaining every sample). Construct
with a name (`tools::Stats foo_stats_{"name"};`), call `.add(value_ms,
frame_index)` once per sample -- auto-logs a `[STATS]` summary
(count/mean/stddev/min/max + the worst-N samples, each tagged with the
frame index that produced it) every 200 samples, and once more from the
destructor so short runs still get a final report.

Wired into:
- **`YOLOV5`** (single-threaded, both backends): `detect total` -- wraps
  the *entire* `detect()` call, the number that actually dictates
  achievable FPS for this path (the per-bracket stats below are its
  components, not a substitute). Plus the TensorRT-specific brackets: `TRT
  memcpy`/`dispatch`/`enqueueV3`/`gpu_wait_and_d2h`, and `parse` (shared
  with the non-TensorRT backends).
- **`MultiThreadDetector`** (async): `MT push total (excl. backpressure)`
  / `MT pop total` -- each side's own *active* cost, for diagnosing which
  pipeline *stage* is inherently slower. `MT push wall (incl.
  backpressure)` -- wraps the *entire* `push()` call including any wait for
  a free ring slot, i.e. what a real caller (`mt_standard.cpp`'s producer
  loop) actually pays per frame. `MT pipeline_latency` -- end-to-end delay
  from `push(t)` to a result being ready, a *different* question from
  throughput (how stale a result is, not how many frames/sec the pipeline
  sustains). A derived `[MT-THROUGHPUT]` line every 200 pops logs a
  theoretical `ceiling` (from `push_total`/`pop_total`, assumes infinite
  ring buffering) alongside an `achieved` estimate (from `push_wall`).

**Gotcha, found the hard way: `[MT-THROUGHPUT]`'s `achieved` number is not
a precise fps predictor.** With only `kTrtRingSize=3` slots of buffering,
the producer can burst ahead of a slower consumer without the two
threads' costs averaging out to the fully-saturated-queue theoretical
rate implied by `push_wall`'s mean -- measured up to ~25% off from the
real number. **The actual ground truth for real achieved throughput is
the simple `done: ... fps` summary line** (`popped_count / elapsed` wall
clock), not any derived `Stats` mean -- treat `push_wall`'s stats as a
diagnostic (where's the time going, which frames are outliers), not a
throughput oracle.

### 6.4 Performance tuning findings, in order of actual impact (measured)

All numbers from repeated `mt_detector_video_test` runs (unthrottled,
full 687-frame `assets/demo/demo.avi`) on the same code, varying only the
device/environment state:

1. **Reboot the device if it's been under heavy, repeated TensorRT/CUDA
   engine-build churn for a long time.** Pinned/CMA memory (a small,
   physically-contiguous carveout, separate from and in addition to
   regular RAM) can fragment from many alloc/free cycles across many
   processes, causing `NvMapMemAllocInternalTagged: error 12` and hard
   crashes (`cudaMallocHost failed`, `CUDA initialization failure`). A
   `docker restart` does **not** fix this -- the carveout is host-kernel
   state, not container-scoped. Only a full device reboot resets it.
   **Single biggest factor measured this session: ~100-130fps -> 183fps**,
   same code, purely from a clean reboot.
2. **`sudo jetson_clocks`** (§5.7) -- still real and worth doing (locks
   GPU/CPU clocks at max, disabling DVFS ramp-down/up between bursty
   inference calls), but measured smaller than expected *on top of* a
   clean reboot: **~4% (183fps -> 191fps)**. Doesn't survive a reboot --
   re-run it after every one. Don't assume it alone explains a big
   performance gap; check the memory/reboot state first.
3. **Close/log out of any desktop GUI session before benchmarking.** A
   running GNOME session costs ~1-3GB RAM baseline. Combined with
   `mt_detector_video_test`'s frame-preload design (~4.3GB for the full
   demo video -- deliberately holding it all in memory so file-decode time
   doesn't pollute the timing), this pushes total memory usage close
   enough to the device's 7.6GB limit to cause real kernel-level
   memory-pressure stalls. Confirmed via `tegrastats` run concurrently with
   the benchmark and correlated against the worst outlier frames' exact
   timestamps: CPU cores pegged near 100%, GPU near 97% utilization, RAM at
   ~81% peak during those windows -- not GPU clock ramping (already ruled
   out by clocks being locked), not thermal throttling (temps stayed a mild
   ~57-58°C throughout). Measured: logging out cut peak RAM from ~6.2GB
   (81%) to ~5.6GB (74%), **fps 191 -> 233**, and worst-case per-frame
   stalls (excluding the one-time first-frame warmup) from ~30-50ms down to
   ~7-8ms.
4. **Best: run fully `--headless` (§6.2), no desktop or X server at all.**
   Strictly less overhead than just logging out, which still leaves GDM's
   login-greeter `Xorg` process and related services running. Measured:
   **287fps**, worst-case stalls (excl. warmup) down to ~3.5-4ms -- the
   best result measured all session, on the exact same code as the
   100-130fps starting point. `--max-frames` (reducing the preloaded video
   buffer itself) made no further meaningful difference once headless --
   confirms the desktop session's overhead was the dominant effect, not the
   video buffer's absolute size.

**Caveat:** `mt_standard.cpp`/the actual production path never preloads
hundreds of frames like this test does -- it reads one frame at a time
from a live camera. So the specific memory-pressure mechanism above is
very likely a benchmark-tool artifact, not something the deployed robot
hits in the field. Still worth keeping the Jetson's desktop light (or
better, run headless) *while benchmarking*, since jetson_clocks/reboot
findings 1-2 apply generally, not just to this test tool.

### 6.5 X11 quirk: GDM's per-session `Xauthority` (only relevant without `--headless`)

If you need the internal debug `imshow` window (i.e. **not** using
`--headless`) and hit `"Authorization required, but no authorization
protocol specified"` after a reboot or a desktop logout/login cycle: the
container's `.Xauthority` is bind-mounted from `~/.Xauthority` at
container-*creation* time (see `docker_enter_jetson.sh`), but GDM manages
its own, separate, per-session auth file at
`/run/user/<uid>/gdm/Xauthority` (different for a logged-in desktop
session vs. the login greeter, and regenerated fresh each session) -- the
mounted file goes stale and the container has no way to see the new one.
Since it's a live bind-mounted **file** (not a directory), `docker
start`/`docker restart` doesn't refresh it even though the container
itself survives fine. Fix without recreating the container (which would
need an active desktop session + `$DISPLAY` anyway):
```
docker cp /run/user/<uid>/gdm/Xauthority <container>:/tmp/current_auth
docker exec -e DISPLAY=:<N> -e XAUTHORITY=/tmp/current_auth <container> <cmd>
```
(`<uid>` is usually `1000` for a normal logged-in user, or the `gdm`
system user's uid for the login greeter -- check `ps aux | grep Xorg` for
the exact `-auth` path and `vt`/`DISPLAY` currently in use; the auth file
is typically root-owned and needs `sudo cp`/`sudo chmod` to read from a
non-root shell.) Simplest long-term fix: just use `--headless` (§6.2/6.4)
and avoid all of this.

## 7. Known gaps / things to verify on real hardware

- `device: TENSORRT` (§5.7) needs `/usr/local/cuda-12.6` bind-mounted from
  the host at container run time to compile at all (`-v
  /usr/local/cuda-12.6:/usr/local/cuda-12.6:ro`) -- the Jetson apt repo has
  no package providing CUDA's own C headers. Not baked into the Dockerfile;
  see §5.7 for why and what a real fix would look like.
- Whenever GPU performance on Jetson looks disappointing, check
  `sudo jetson_clocks` has been run before assuming the code/model is the
  problem -- see §5.7, this alone was a ~3x difference for `device:
  TENSORRT`.
- `device: CUDA` (§5.5/§5.6) and `device: TENSORRT` (§5.7) only have a
  working ONNX export for `yolov5` — the model every shipped config
  actually uses. `yolo11`/`yolov8` don't have a CUDA or TensorRT path.
- Camera SDKs (HikRobot/MindVision) are vendored as prebuilt `.so` files with no
  visible build/version metadata in this repo — if the physical camera's
  firmware requires a newer SDK than what's bundled, you'll need to source an
  updated arm64 `.so` from the vendor.
- CAN bus setup (`can0`/`can1` bring-up via udev rule) assumes a
  SocketCAN-compatible interface is present — most Jetson carrier boards don't
  expose CAN natively (unlike the AGX Orin devkit's dev header, which requires
  extra hardware/DTS overlay setup), so you likely need a USB-CAN adapter.
- `cmake --build . --target <name>` only rebuilds `<name>` and *its own*
  dependencies — it does **not** rebuild sibling executables that also link
  a shared library you changed (`auto_aim`, `tools`, etc.) but weren't
  named. Bit this session: fixing `trt_engine.cpp` and only rebuilding
  `detector_video_test`/`mt_detector_video_test` left `auto_aim_test` (and
  `standard`/`mt_standard`/several others) as stale binaries that then
  failed loudly with a TensorRT plugin-registry error, unrelated-looking to
  the actual fix. After any change to shared/library code, prefer a full
  `cmake --build .` (no `--target`) unless you're certain which binaries
  are affected.
- If you ever edit a file **directly on the Jetson** (e.g. over SSH,
  without going through the assistant/normal local-edit flow) rather than
  in the local checkout that then gets synced over, that edit is invisible
  to anything that later syncs the local copy back — the next sync
  silently reverts it. Bit this session: a user's own on-Jetson fix to
  `mt_detector_video_test.cpp` (removing a misplaced-bounding-box `imshow`
  call) got clobbered by a later `scp` of an unrelated local change, which
  then reintroduced a crash under `--headless`. Pull on-device edits back
  into the local checkout (or make them there in the first place) before
  the next sync.
- **What was actually verified while writing this doc:** the `Dockerfile`'s
  x86_64 path was built end-to-end on a real Docker daemon and every non-ROS2
  target compiled and linked successfully. The arm64 OpenVINO archive URL was
  confirmed to serve a genuine, correctly-sized gzip archive (not an error
  page — an earlier draft of this Dockerfile silently downloaded an HTML 404
  page because the URL path/filename versioning is inconsistent on Intel's
  download server; the current Dockerfile avoids that trap).
- **Also verified:** `device: GPU` inference (§5.4) — YOLO detection running
  through the OpenVINO GPU plugin on an Intel iGPU inside the container, not
  just CPU-plugin builds.
- **Not yet verified:** an actual build *on* aarch64 hardware, and running any
  executable against a real camera/CAN/serial/C-board. Treat the Jetson
  compatibility table in §4 as a well-checked static analysis (CMake logic,
  package availability, architecture guards) rather than a confirmed
  on-device test.
