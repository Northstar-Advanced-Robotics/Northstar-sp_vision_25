#ifndef AUTO_AIM__TRT_ENGINE_HPP
#define AUTO_AIM__TRT_ENGINE_HPP

#ifdef HAVE_TENSORRT
#include <NvInfer.h>

#include <cstdint>
#include <memory>
#include <string>

namespace auto_aim
{

// Shared by YOLOV5's single-threaded device: TENSORRT path (yolov5.hpp/.cpp)
// and MultiThreadDetector's async TENSORRT path (mt_detector.hpp/.cpp) --
// moved here (no logic changes) so this ~70 lines of TensorRT bring-up
// (builder/parser/config/FP16 flag/serialize-and-cache-to-disk) exists in
// exactly one place instead of drifting between two copies. Must be kept
// alive (as a member, not a stack local) for the lifetime of any
// runtime/engine/context built from it -- TensorRT keeps a reference to the
// ILogger internally.
class TRTLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override;
};

// Loads a cached .engine file at engine_path if present (fast path), else
// builds one from onnx_path (FP16 if the platform supports it) and caches it
// to engine_path -- a one-time, slow (can take several minutes) operation
// per device. Fills runtime/engine; both must be empty unique_ptrs on entry.
void trt_build_or_load_engine(
  TRTLogger & logger, std::unique_ptr<nvinfer1::IRuntime> & runtime,
  std::unique_ptr<nvinfer1::ICudaEngine> & engine, const std::string & onnx_path,
  const std::string & engine_path);

// Names baked into the fused-NMS ONNX graph by
// scripts/onnx/fuse_efficient_nms.py -- kept as named constants (not string
// literals scattered across yolov5.cpp/mt_detector.cpp) so there's exactly
// one place to update if that script's output names ever change.
//
// This uses TensorRT's own EfficientNMS_TRT plugin, not the ONNX-standard
// NonMaxSuppression op an earlier version of this fusion used (see git
// history for scripts/onnx/fuse_nms.py, superseded). That earlier approach
// produced a dynamically-shaped `selected_indices` output, which forced
// TensorRT's data-dependent-shape (DDS) machinery (IOutputAllocator) --
// and TensorRT 10.0-10.7 (this Jetson is pinned to 10.3.0 via JetPack
// 6.1/6.2, no in-place upgrade available) has a documented, NVIDIA-
// acknowledged performance regression there: enqueueV3() blocks until the
// whole network finishes instead of returning immediately, measured at
// ~6.5ms (essentially the full GPU compute time) instead of near-instant.
// EfficientNMS_TRT sidesteps this: its outputs are FIXED shape, padded to
// kMaxNmsOutputBoxes, so no IOutputAllocator/DDS is needed at all.
inline constexpr const char * kTrtOutputTensorName = "output/sink_port_0";
inline constexpr const char * kTrtNumDetectionsTensorName = "num_detections";
inline constexpr const char * kTrtDetectionBoxesTensorName = "detection_boxes";
inline constexpr const char * kTrtDetectionScoresTensorName = "detection_scores";
inline constexpr const char * kTrtDetectionClassesTensorName = "detection_classes";

// Must match max_output_boxes in scripts/onnx/fuse_efficient_nms.py exactly
// -- this is the fixed row count detection_boxes/detection_scores/
// detection_classes are padded to. No compiler links these two constants
// -- if you change one, change the other.
inline constexpr int64_t kMaxNmsOutputBoxes = 64;

struct TrtIONames
{
  std::string input;
  std::string output;             // kTrtOutputTensorName -- 1x25200x22 float, static shape
  std::string num_detections;     // kTrtNumDetectionsTensorName -- 1x1 int32, static shape
  std::string detection_boxes;    // kTrtDetectionBoxesTensorName -- 1x64x4 float, static shape (unused downstream -- EfficientNMS_TRT requires an address bound for it regardless)
  std::string detection_scores;   // kTrtDetectionScoresTensorName -- 1x64 float, static shape
  std::string detection_classes;  // kTrtDetectionClassesTensorName -- 1x64 int32, static shape (unused downstream -- only ever one "class")
};

// Discovers I/O tensor names by IOMode + exact name match. Replaces the
// naive "first kINPUT wins input, else output" loop this repo used before
// the engine had multiple outputs -- that loop silently broke once a
// second output existed (it just kept overwriting a single output-name
// variable with whichever output getIOTensorName() enumerated last). No
// backward-compat fallback for a pre-fusion single-output engine/onnx or
// the earlier DDS-based fusion (selected_indices): since assets/yolov5.onnx
// is permanently rewritten by fuse_efficient_nms.py going forward, a stale
// local .engine (gitignored, cached from before that change) or an
// accidentally-reverted .onnx should fail loudly here at construction
// time, not silently degrade.
TrtIONames trt_discover_io_names(nvinfer1::ICudaEngine & engine);

}  // namespace auto_aim

#endif  // HAVE_TENSORRT
#endif  // AUTO_AIM__TRT_ENGINE_HPP
