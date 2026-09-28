cmake_minimum_required(VERSION 3.16)

set(OpenVINO_DIR "/opt/intel/openvino_2024.6.0/runtime/cmake")

find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# Optional NVIDIA GPU backend (device: CUDA in the yaml configs) via ONNX
# Runtime's CUDA execution provider. OpenVINO's own GPU plugin is Intel-only,
# so this is a second, independent inference path for YOLOV5 -- see
# JETSON_ORIN.md for how the .onnx models were derived from the .xml/.bin
# OpenVINO IR already used by the OpenVINO path. Silently absent (falls back
# to OpenVINO-only) if onnxruntime isn't installed at /opt/onnxruntime.
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h PATHS /opt/onnxruntime/include)
find_library(ONNXRUNTIME_LIB onnxruntime PATHS /opt/onnxruntime/lib)

# Optional NVIDIA GPU backend #2 (device: TENSORRT) via TensorRT's own C++
# API directly -- not through ONNX Runtime's TensorRT execution provider.
# Substantially faster than device: CUDA above for the same model (TensorRT
# does FP16 + kernel-level autotuning for the exact GPU it's built on), at
# the cost of needing a one-time engine build per device (see yolov5.cpp
# and JETSON_ORIN.md). Silently absent if TensorRT's dev headers/libs
# (libnvinfer-dev, libnvonnxparsers-dev) and the CUDA runtime aren't
# installed. Explicit multiarch PATHS are needed here: apt installs these
# under Debian/Ubuntu's per-architecture-triplet layout
# (/usr/include/aarch64-linux-gnu, /usr/lib/aarch64-linux-gnu on Jetson;
# .../x86_64-linux-gnu on desktop) -- confirmed via `dpkg -L libnvinfer-dev`
# on a real Jetson device that plain find_path/find_library without a PATHS
# hint does not reliably search those on its own.
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu /usr/include)
find_library(TENSORRT_NVINFER_LIB nvinfer
    PATHS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib)
find_library(TENSORRT_NVONNXPARSER_LIB nvonnxparser
    PATHS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib)
# initLibNvInferPlugins() (needed to register EfficientNMS_TRT -- see
# scripts/onnx/fuse_efficient_nms.py -- with the plugin registry; trtexec
# does this automatically at startup, this project's own use of the
# TensorRT API doesn't) lives in libnvinfer_plugin, not libnvinfer.
find_library(TENSORRT_NVINFER_PLUGIN_LIB nvinfer_plugin
    PATHS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib)
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h PATHS /usr/local/cuda/include)
find_library(CUDA_CUDART_LIB cudart PATHS /usr/local/cuda/lib64 /usr/local/cuda/targets/aarch64-linux/lib /usr/local/cuda/targets/x86_64-linux/lib)

# device: TENSORRT's preprocessing (letterbox resize + BGR->RGB + normalize
# + HWC->CHW) runs as a single custom CUDA kernel on the GPU
# (yolos/preprocess_kernel.cu) instead of on the CPU -- needs an actual CUDA
# *compiler* (nvcc), not just the runtime libs found above. check_language
# probes for it without hard-failing configure if it's absent (e.g.
# TensorRT runtime libs installed but only the toolkit's runtime-only subset,
# no nvcc -- see the Dockerfile's aarch64 CUDA install comment). 75/86/89
# cover common desktop/laptop NVIDIA GPUs (Turing/Ampere/Ada); 87 is Jetson
# Orin's own compute capability -- covers both this project's supported
# platforms without needing CMAKE_CUDA_ARCHITECTURES=native (CMake 3.24+
# only; this repo targets 3.16).
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 87 89)
    enable_language(CUDA)
endif()

add_subdirectory(planner/tinympc)

add_library(auto_aim OBJECT 
    armor.cpp
    classifier.cpp 
    detector.cpp
    solver.cpp
    aimer.cpp
    target.cpp
    tracker.cpp
    voter.cpp
    shooter.cpp
    yolo.cpp
    yolos/yolov5.cpp
    yolos/yolov8.cpp
    yolos/yolo11.cpp
    yolos/trt_engine.cpp
    multithread/commandgener.cpp
    multithread/mt_detector.cpp
    planner/planner.cpp
)

target_link_libraries(auto_aim io openvino::runtime tinympcstatic)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
    message(STATUS "onnxruntime found -- device: CUDA (NVIDIA GPU) backend enabled for YOLOV5")
    target_compile_definitions(auto_aim PUBLIC HAVE_ONNXRUNTIME)
    target_include_directories(auto_aim PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(auto_aim ${ONNXRUNTIME_LIB})
else()
    message(STATUS "onnxruntime not found at /opt/onnxruntime -- device: CUDA will not be available (OpenVINO CPU/GPU only)")
endif()

if(TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER_LIB AND TENSORRT_NVONNXPARSER_LIB AND TENSORRT_NVINFER_PLUGIN_LIB
   AND CUDA_INCLUDE_DIR AND CUDA_CUDART_LIB AND CMAKE_CUDA_COMPILER)
    message(STATUS "TensorRT + CUDA compiler found -- device: TENSORRT (NVIDIA GPU, GPU-side preprocessing) backend enabled for YOLOV5")
    target_compile_definitions(auto_aim PUBLIC HAVE_TENSORRT)
    target_include_directories(auto_aim PUBLIC ${TENSORRT_INCLUDE_DIR} ${CUDA_INCLUDE_DIR})
    target_link_libraries(auto_aim ${TENSORRT_NVINFER_LIB} ${TENSORRT_NVONNXPARSER_LIB}
        ${TENSORRT_NVINFER_PLUGIN_LIB} ${CUDA_CUDART_LIB})
    target_sources(auto_aim PRIVATE yolos/preprocess_kernel.cu)
elseif(TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER_LIB AND TENSORRT_NVONNXPARSER_LIB AND TENSORRT_NVINFER_PLUGIN_LIB
       AND CUDA_INCLUDE_DIR AND CUDA_CUDART_LIB)
    message(STATUS "TensorRT found but no CUDA compiler (nvcc) -- device: TENSORRT will not be available (its preprocessing is a custom CUDA kernel, needs nvcc to build)")
else()
    message(STATUS "TensorRT not found -- device: TENSORRT will not be available")
endif()