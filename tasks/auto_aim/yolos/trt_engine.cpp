#include "trt_engine.hpp"

#ifdef HAVE_TENSORRT
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include <fstream>
#include <vector>

#include "tools/logger.hpp"

namespace auto_aim
{

void TRTLogger::log(Severity severity, const char * msg) noexcept
{
  if (severity <= Severity::kWARNING) {
    tools::logger()->warn("[TensorRT] {}", msg);
  }
}

void trt_build_or_load_engine(
  TRTLogger & logger, std::unique_ptr<nvinfer1::IRuntime> & runtime,
  std::unique_ptr<nvinfer1::ICudaEngine> & engine, const std::string & onnx_path,
  const std::string & engine_path)
{
  // Registers TensorRT's built-in plugin creators (including
  // EfficientNMS_TRT, used by the fused-NMS graph -- see
  // scripts/onnx/fuse_efficient_nms.py) with the global plugin registry.
  // trtexec calls this automatically at startup, which is why a graph that
  // parses/builds/runs fine under trtexec can still fail here ("Plugin not
  // found, are the plugin name, version, and namespace correct?") without
  // it -- needed both for parsing the .onnx (builder path) and for
  // deserializing a cached .engine (its plugin layers look the creator up
  // by name too), so this must run before either path below, not just the
  // build path.
  if (!initLibNvInferPlugins(&logger, "")) {
    throw std::runtime_error("TensorRT: initLibNvInferPlugins failed");
  }

  runtime.reset(nvinfer1::createInferRuntime(logger));
  if (!runtime) throw std::runtime_error("TensorRT: failed to create runtime");

  std::ifstream engine_file(engine_path, std::ios::binary | std::ios::ate);
  if (engine_file.good()) {
    auto size = engine_file.tellg();
    engine_file.seekg(0);
    std::vector<char> engine_data(static_cast<size_t>(size));
    engine_file.read(engine_data.data(), size);
    engine.reset(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!engine) {
      throw std::runtime_error(
        "TensorRT: failed to deserialize cached engine at " + engine_path +
        " (likely built for a different GPU/TensorRT/CUDA version -- delete it to force a rebuild)");
    }
    tools::logger()->info("TensorRT: loaded cached engine from {}", engine_path);
    return;
  }

  // No cached engine: build one from the .onnx model. This is a one-time,
  // slow (can take several minutes) operation per device -- the resulting
  // .engine file is tied to the exact GPU/TensorRT/CUDA version it was built
  // on (not portable across devices, unlike the .onnx/.xml models) and is
  // cached to disk here so subsequent runs skip straight to the fast load
  // path above. See JETSON_ORIN.md for how to pre-build this ahead of time
  // instead of eating the delay on first run.
  tools::logger()->warn(
    "TensorRT: no cached engine at {}, building one from {} now "
    "(one-time, can take several minutes)...",
    engine_path, onnx_path);

  std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
  if (!builder) throw std::runtime_error("TensorRT: failed to create builder");

  std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0U));
  if (!network) throw std::runtime_error("TensorRT: failed to create network");

  std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
  if (!parser) throw std::runtime_error("TensorRT: failed to create ONNX parser");

  if (!parser->parseFromFile(
        onnx_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    throw std::runtime_error("TensorRT: failed to parse " + onnx_path);
  }

  std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
  if (!config) throw std::runtime_error("TensorRT: failed to create builder config");
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1ULL << 30);
  bool fp16_supported = builder->platformHasFastFp16();
  tools::logger()->info("TensorRT: platformHasFastFp16={}", fp16_supported);
  if (fp16_supported) config->setFlag(nvinfer1::BuilderFlag::kFP16);

  std::unique_ptr<nvinfer1::IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
  if (!serialized) throw std::runtime_error("TensorRT: engine build failed");

  std::ofstream out(engine_path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(serialized->data()), static_cast<std::streamsize>(serialized->size()));
  out.close();
  tools::logger()->info("TensorRT: built and cached engine at {}", engine_path);

  engine.reset(runtime->deserializeCudaEngine(serialized->data(), serialized->size()));
  if (!engine) throw std::runtime_error("TensorRT: failed to deserialize freshly-built engine");
}

TrtIONames trt_discover_io_names(nvinfer1::ICudaEngine & engine)
{
  TrtIONames names;
  for (int i = 0; i < engine.getNbIOTensors(); i++) {
    std::string name = engine.getIOTensorName(i);
    if (engine.getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT) {
      names.input = name;
    } else if (name == kTrtOutputTensorName) {
      names.output = name;
    } else if (name == kTrtNumDetectionsTensorName) {
      names.num_detections = name;
    } else if (name == kTrtDetectionBoxesTensorName) {
      names.detection_boxes = name;
    } else if (name == kTrtDetectionScoresTensorName) {
      names.detection_scores = name;
    } else if (name == kTrtDetectionClassesTensorName) {
      names.detection_classes = name;
    } else {
      throw std::runtime_error("TensorRT: engine has unexpected I/O tensor '" + name + "'");
    }
  }
  if (
    names.input.empty() || names.output.empty() || names.num_detections.empty() ||
    names.detection_boxes.empty() || names.detection_scores.empty() ||
    names.detection_classes.empty()) {
    throw std::runtime_error(
      "TensorRT: engine I/O tensors don't match the expected fused-NMS layout (found input='" +
      names.input + "' output='" + names.output + "' num_detections='" + names.num_detections +
      "' detection_boxes='" + names.detection_boxes + "' detection_scores='" + names.detection_scores +
      "' detection_classes='" + names.detection_classes +
      "' -- likely a stale .engine cached from before this fusion (or from the earlier DDS-based "
      "fusion), or an unfused .onnx; delete the local .engine and/or re-run "
      "scripts/onnx/fuse_efficient_nms.py)");
  }
  return names;
}

}  // namespace auto_aim

#endif  // HAVE_TENSORRT
