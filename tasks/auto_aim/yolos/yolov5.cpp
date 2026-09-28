#include "yolov5.hpp"

#include <fmt/chrono.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>

#include "tools/img_tools.hpp"
#include "tools/logger.hpp"

namespace auto_aim
{
YOLOV5::YOLOV5(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false)
{
  auto yaml = YAML::LoadFile(config_path);

  model_path_ = yaml["yolov5_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();
  binary_threshold_ = yaml["threshold"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  int x = 0, y = 0, width = 0, height = 0;
  x = yaml["roi"]["x"].as<int>();
  y = yaml["roi"]["y"].as<int>();
  width = yaml["roi"]["width"].as<int>();
  height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  use_traditional_ = yaml["use_traditional"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);

  if (device_ == "CUDA") {
#ifdef HAVE_ONNXRUNTIME
    use_cuda_ = true;

    // OpenVINO's GPU plugin only ever talks to an Intel GPU. NVIDIA GPU
    // acceleration goes through this separate ONNX Runtime CUDA backend
    // instead, using a .onnx sibling of the .xml/.bin IR model (same
    // weights, converted offline -- see JETSON_ORIN.md). The IR model does
    // BGR->RGB/u8->f32/scale(255) and NHWC->NCHW via the OpenVINO
    // preprocessor above; the ONNX export has no such preprocessor attached,
    // so that conversion is done manually below before feeding the tensor in.
    auto dot = model_path_.find_last_of('.');
    auto onnx_path = (dot == std::string::npos ? model_path_ : model_path_.substr(0, dot)) + ".onnx";

    ort_env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "yolov5");
    Ort::SessionOptions session_options;
    OrtCUDAProviderOptions cuda_options{};
    session_options.AppendExecutionProvider_CUDA(cuda_options);
    ort_session_ = std::make_unique<Ort::Session>(*ort_env_, onnx_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    ort_input_name_ = ort_session_->GetInputNameAllocated(0, allocator).get();
    ort_output_name_ = ort_session_->GetOutputNameAllocated(0, allocator).get();

    tools::logger()->info(
      "YOLOV5: using ONNX Runtime CUDA backend, model={}", onnx_path);
#else
    throw std::runtime_error(
      "device: CUDA requires building with ONNX Runtime support, but onnxruntime "
      "wasn't found at configure time (see JETSON_ORIN.md)");
#endif
  } else if (device_ == "TENSORRT") {
#ifdef HAVE_TENSORRT
    use_tensorrt_ = true;

    // Same .onnx model as device: CUDA above, but run through TensorRT's own
    // C++ API directly instead of ONNX Runtime -- substantially faster
    // (FP16 + kernel autotuning for the exact GPU), at the cost of a
    // one-time engine build per device. See JETSON_ORIN.md.
    auto dot = model_path_.find_last_of('.');
    auto onnx_path = (dot == std::string::npos ? model_path_ : model_path_.substr(0, dot)) + ".onnx";
    auto engine_path = (dot == std::string::npos ? model_path_ : model_path_.substr(0, dot)) + ".engine";

    trt_build_or_load_engine(trt_logger_, trt_runtime_, trt_engine_, onnx_path, engine_path);

    trt_context_.reset(trt_engine_->createExecutionContext());
    if (!trt_context_) throw std::runtime_error("YOLOV5: failed to create TensorRT execution context");

    if (cudaStreamCreate(&trt_stream_) != cudaSuccess)
      throw std::runtime_error("YOLOV5: cudaStreamCreate failed");

    // Fixed shapes: input 1x3x640x640, output 1x25200x22 -- same shapes
    // hardcoded throughout the device: CUDA path and parse() below, since
    // this .onnx model has no dynamic axes.
    if (cudaMalloc(&trt_input_device_, 1 * 3 * 640 * 640 * sizeof(float)) != cudaSuccess)
      throw std::runtime_error("YOLOV5: cudaMalloc (input) failed");
    if (cudaMalloc(&trt_output_device_, 1 * 25200 * 22 * sizeof(float)) != cudaSuccess)
      throw std::runtime_error("YOLOV5: cudaMalloc (output) failed");

    // Pinned host buffers, allocated once here rather than per-frame (see
    // the member comment in yolov5.hpp).
    if (cudaMallocHost(reinterpret_cast<void **>(&trt_input_host_), 1 * 3 * 640 * 640 * sizeof(float)) !=
        cudaSuccess)
      throw std::runtime_error("YOLOV5: cudaMallocHost (input) failed");
    if (cudaMallocHost(reinterpret_cast<void **>(&trt_output_host_), 1 * 25200 * 22 * sizeof(float)) !=
        cudaSuccess)
      throw std::runtime_error("YOLOV5: cudaMallocHost (output) failed");
    // EfficientNMS_TRT's four outputs are all FIXED shape (padded to
    // kMaxNmsOutputBoxes) -- no IOutputAllocator/data-dependent-shape
    // machinery needed, just plain device buffers + setTensorAddress like
    // input/output already use. detection_boxes/detection_classes are
    // allocated (every output tensor needs a bound address) but never read
    // back -- see the member comments in yolov5.hpp for why.
    if (cudaMalloc(&trt_num_detections_device_, sizeof(int32_t)) != cudaSuccess)
      throw std::runtime_error("YOLOV5: cudaMalloc (num_detections) failed");
    if (cudaMalloc(&trt_detection_boxes_device_, kMaxNmsOutputBoxes * 4 * sizeof(float)) != cudaSuccess)
      throw std::runtime_error("YOLOV5: cudaMalloc (detection_boxes) failed");
    if (cudaMalloc(&trt_detection_scores_device_, kMaxNmsOutputBoxes * sizeof(float)) != cudaSuccess)
      throw std::runtime_error("YOLOV5: cudaMalloc (detection_scores) failed");
    if (cudaMalloc(&trt_detection_classes_device_, kMaxNmsOutputBoxes * sizeof(int32_t)) != cudaSuccess)
      throw std::runtime_error("YOLOV5: cudaMalloc (detection_classes) failed");
    if (cudaMallocHost(reinterpret_cast<void **>(&trt_num_detections_host_), sizeof(int32_t)) != cudaSuccess)
      throw std::runtime_error("YOLOV5: cudaMallocHost (num_detections) failed");
    if (cudaMallocHost(
          reinterpret_cast<void **>(&trt_detection_scores_host_),
          kMaxNmsOutputBoxes * sizeof(float)) != cudaSuccess)
      throw std::runtime_error("YOLOV5: cudaMallocHost (detection_scores) failed");

    auto io_names = trt_discover_io_names(*trt_engine_);
    trt_input_name_ = io_names.input;
    trt_output_name_ = io_names.output;

    trt_context_->setTensorAddress(trt_input_name_.c_str(), trt_input_device_);
    trt_context_->setTensorAddress(trt_output_name_.c_str(), trt_output_device_);
    trt_context_->setTensorAddress(io_names.num_detections.c_str(), trt_num_detections_device_);
    trt_context_->setTensorAddress(io_names.detection_boxes.c_str(), trt_detection_boxes_device_);
    trt_context_->setTensorAddress(io_names.detection_scores.c_str(), trt_detection_scores_device_);
    trt_context_->setTensorAddress(io_names.detection_classes.c_str(), trt_detection_classes_device_);

    tools::logger()->info("YOLOV5: using TensorRT backend (fused NMS), engine={}", engine_path);
#else
    throw std::runtime_error(
      "device: TENSORRT requires building with TensorRT support, but TensorRT "
      "wasn't found at configure time (see JETSON_ORIN.md)");
#endif
  } else {
    auto model = core_.read_model(model_path_);
    ov::preprocess::PrePostProcessor ppp(model);
    auto & input = ppp.input();

    input.tensor()
      .set_element_type(ov::element::u8)
      .set_shape({1, 640, 640, 3})
      .set_layout("NHWC")
      .set_color_format(ov::preprocess::ColorFormat::BGR);

    input.model().set_layout("NCHW");

    input.preprocess()
      .convert_element_type(ov::element::f32)
      .convert_color(ov::preprocess::ColorFormat::RGB)
      .scale(255.0);

    // TODO: ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)
    model = ppp.build();
    compiled_model_ = core_.compile_model(
      model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  }
}

YOLOV5::~YOLOV5()
{
#ifdef HAVE_TENSORRT
  if (trt_input_device_) cudaFree(trt_input_device_);
  if (trt_output_device_) cudaFree(trt_output_device_);
  if (trt_input_host_) cudaFreeHost(trt_input_host_);
  if (trt_output_host_) cudaFreeHost(trt_output_host_);
  if (trt_raw_input_device_) cudaFree(trt_raw_input_device_);
  if (trt_raw_input_host_) cudaFreeHost(trt_raw_input_host_);
  if (trt_num_detections_device_) cudaFree(trt_num_detections_device_);
  if (trt_detection_boxes_device_) cudaFree(trt_detection_boxes_device_);
  if (trt_detection_scores_device_) cudaFree(trt_detection_scores_device_);
  if (trt_detection_classes_device_) cudaFree(trt_detection_classes_device_);
  if (trt_num_detections_host_) cudaFreeHost(trt_num_detections_host_);
  if (trt_detection_scores_host_) cudaFreeHost(trt_detection_scores_host_);
  if (trt_stream_) cudaStreamDestroy(trt_stream_);
#endif
}

std::list<Armor> YOLOV5::detect(const cv::Mat & raw_img, int frame_count)
{
  auto t_detect_start = std::chrono::steady_clock::now();

  if (raw_img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::list<Armor>();
  }

  cv::Mat bgr_img;
  if (use_roi_) {
    if (roi_.width == -1) {  // -1 表示该维度不裁切
      roi_.width = raw_img.cols;
    }
    if (roi_.height == -1) {  // -1 表示该维度不裁切
      roi_.height = raw_img.rows;
    }
    bgr_img = raw_img(roi_);
  } else {
    bgr_img = raw_img;
  }

  auto x_scale = static_cast<double>(640) / bgr_img.rows;
  auto y_scale = static_cast<double>(640) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(bgr_img.rows * scale);
  auto w = static_cast<int>(bgr_img.cols * scale);

  cv::Mat output;
#ifdef HAVE_TENSORRT
  if (use_tensorrt_) {
    // GPU-resident preprocessing (see infer_tensorrt_gpu_preprocess): the
    // CPU letterbox resize + blobFromImage chain below is skipped entirely
    // for this backend -- bgr_img (unresized, original resolution) is
    // handed straight to the GPU. NMS suppression also already happened on
    // the GPU (see trt_engine.hpp/scripts/onnx/fuse_efficient_nms.py), so
    // the TensorRT branch below calls parse_from_efficient_nms() instead of
    // parse() -- a tiny (0-64 row) postprocess over GPU-selected survivors
    // rather than a 25200-row CPU scan.
    auto trt_result = infer_tensorrt_gpu_preprocess(bgr_img, w, h, scale, frame_count);
    auto t_infer_done = std::chrono::steady_clock::now();
    auto result = parse_from_efficient_nms(
      scale, trt_result.raw_output, trt_result.detection_scores.data(), trt_result.num_detections,
      raw_img, frame_count);
    auto t_parse_done = std::chrono::steady_clock::now();
    auto parse_ms = std::chrono::duration<double, std::milli>(t_parse_done - t_infer_done).count();
    tools::logger()->info(
      "[PARSE-TIMING] parse={:.3f}ms num_detections={}", parse_ms, trt_result.num_detections);
    parse_stats_.add(parse_ms, frame_count);
    // This is the number that actually dictates achievable FPS for the
    // single-threaded path (1000/mean here is the sustainable ceiling) --
    // everything above (memcpy/dispatch/enqueueV3/gpu_wait_and_d2h/parse)
    // are its components, not the whole picture; this timer wraps the
    // entire detect() call, including the ROI/scale setup those brackets
    // don't cover.
    double detect_total_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_detect_start)
        .count();
    detect_total_stats_.add(detect_total_ms, frame_count);
    return result;
  }
#endif
  {
    // preproces
    //
    // letterbox_canvas_ is reused across frames instead of allocating +
    // zero-filling a fresh buffer every call: for a fixed camera/ROI, w/h
    // (and therefore the padding region) never change frame to frame, so the
    // zero-fill only actually needs to happen once. Only re-zeroed if the
    // computed size changed (first frame, or the source resolution changed).
    auto t_letterbox_start = std::chrono::steady_clock::now();
    if (letterbox_canvas_.empty() || letterbox_w_ != w || letterbox_h_ != h) {
      letterbox_canvas_ = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
      letterbox_w_ = w;
      letterbox_h_ = h;
    }
    auto roi = cv::Rect(0, 0, w, h);
    cv::resize(bgr_img, letterbox_canvas_(roi), {w, h});
    auto & input = letterbox_canvas_;
    auto t_letterbox_done = std::chrono::steady_clock::now();
    tools::logger()->info(
      "[LETTERBOX-TIMING] letterbox={:.2f}ms",
      std::chrono::duration<double, std::milli>(t_letterbox_done - t_letterbox_start).count());

#ifdef HAVE_ONNXRUNTIME
    if (use_cuda_) {
      output = infer_cuda(input);
    } else
#endif
    {
      output = infer_openvino(input);
    }
  }

  auto t_infer_done = std::chrono::steady_clock::now();
  auto result = parse(scale, output, raw_img, frame_count);
  auto t_parse_done = std::chrono::steady_clock::now();
  auto parse_ms = std::chrono::duration<double, std::milli>(t_parse_done - t_infer_done).count();
  tools::logger()->info("[PARSE-TIMING] parse={:.2f}ms", parse_ms);
  parse_stats_.add(parse_ms, frame_count);
  double detect_total_ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_detect_start).count();
  detect_total_stats_.add(detect_total_ms, frame_count);
  return result;
}

cv::Mat YOLOV5::infer_openvino(const cv::Mat & input)
{
  ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, input.data);

  auto infer_request = compiled_model_.create_infer_request();
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();

  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  // clone(): infer_request (and the buffer output_tensor views) is destroyed
  // when this function returns, so the caller needs its own copy.
  return cv::Mat(output_shape[1], output_shape[2], CV_32F, output_tensor.data()).clone();
}

#ifdef HAVE_ONNXRUNTIME
cv::Mat YOLOV5::infer_cuda(const cv::Mat & input)
{
  // The plain .onnx export has no baked-in preprocessor (unlike the IR model
  // above, which gets one from ov::preprocess::PrePostProcessor at load
  // time), so BGR->RGB, u8->f32/255, and HWC->CHW all happen here by hand.
  cv::Mat rgb, chw_input(3, 640 * 640, CV_32F);
  cv::cvtColor(input, rgb, cv::COLOR_BGR2RGB);
  rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

  std::vector<cv::Mat> channels(3);
  for (int c = 0; c < 3; c++) channels[c] = cv::Mat(640, 640, CV_32F, chw_input.ptr(c));
  cv::split(rgb, channels);

  std::array<int64_t, 4> input_shape{1, 3, 640, 640};
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    mem_info, reinterpret_cast<float *>(chw_input.data), chw_input.total(), input_shape.data(),
    input_shape.size());

  // Output is pre-allocated in plain host memory and bound explicitly via
  // IOBinding, rather than using Run()'s simple API (which lets the CUDA EP
  // auto-allocate the output value itself). On real Jetson hardware, the
  // auto-allocated output's GetTensorData<float>() reliably segfaults even
  // though Run() succeeds and the value's shape/metadata all come back
  // correct -- isolated down to a minimal standalone repro outside this
  // project, so it's a bug in this onnxruntime build's default output path,
  // not something fixable by changing preprocessing. Binding a caller-owned
  // CPU buffer sidesteps it entirely. 25200x22 matches this model's fixed
  // output shape (also hardcoded via the colRange(...) calls in parse()
  // below, so no new fragility here).
  cv::Mat output(25200, 22, CV_32F);
  std::array<int64_t, 3> output_shape{1, 25200, 22};
  Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
    mem_info, reinterpret_cast<float *>(output.data), output.total(), output_shape.data(),
    output_shape.size());

  Ort::IoBinding binding(*ort_session_);
  binding.BindInput(ort_input_name_.c_str(), input_tensor);
  binding.BindOutput(ort_output_name_.c_str(), output_tensor);
  ort_session_->Run(Ort::RunOptions{nullptr}, binding);

  return output;
}
#endif

#ifdef HAVE_TENSORRT
// GPU-resident preprocessing: skips the CPU letterbox + blobFromImage
// entirely. bgr_img is the raw, unresized source frame (or ROI crop) --
// copied into a pinned staging buffer, H2D'd to the device as-is, then
// letterbox-resized + BGR->RGB + normalized + transposed to CHW by a single
// CUDA kernel (see preprocess_kernel.cu) writing directly into
// trt_input_device_. No host sync between the H2D copy, the kernel launch,
// and enqueueV3 -- all three are enqueued on the same trt_stream_, and CUDA
// streams execute in FIFO order, so each is guaranteed to see the previous
// one's writes without an explicit intermediate cudaStreamSynchronize.
YOLOV5::TrtInferResult YOLOV5::infer_tensorrt_gpu_preprocess(
  const cv::Mat & bgr_img, int w, int h, double scale, int frame_count)
{
  auto t0 = std::chrono::steady_clock::now();

  // Lazily (re)allocate the raw-frame staging buffers -- only changes if
  // the source resolution changes (never, in practice, for a fixed
  // camera/ROI), mirroring letterbox_canvas_'s lazy-realloc pattern.
  if (trt_raw_w_ != bgr_img.cols || trt_raw_h_ != bgr_img.rows) {
    if (trt_raw_input_device_) cudaFree(trt_raw_input_device_);
    if (trt_raw_input_host_) cudaFreeHost(trt_raw_input_host_);
    size_t raw_size = static_cast<size_t>(bgr_img.cols) * bgr_img.rows * 3;
    if (cudaMalloc(&trt_raw_input_device_, raw_size) != cudaSuccess)
      throw std::runtime_error("YOLOV5: cudaMalloc (raw input) failed");
    if (cudaMallocHost(reinterpret_cast<void **>(&trt_raw_input_host_), raw_size) != cudaSuccess)
      throw std::runtime_error("YOLOV5: cudaMallocHost (raw input) failed");
    trt_raw_w_ = bgr_img.cols;
    trt_raw_h_ = bgr_img.rows;
  }

  // Copy into the tightly-packed pinned staging buffer. copyTo (not a flat
  // memcpy) handles both a plain contiguous frame (one fast memcpy) and a
  // non-contiguous ROI-cropped view (use_roi_: true; per-row memcpy
  // respecting bgr_img's own stride) correctly either way.
  cv::Mat host_view(bgr_img.rows, bgr_img.cols, CV_8UC3, trt_raw_input_host_);
  bgr_img.copyTo(host_view);
  auto t1 = std::chrono::steady_clock::now();

  size_t raw_size = static_cast<size_t>(bgr_img.cols) * bgr_img.rows * 3;
  cudaMemcpyAsync(
    trt_raw_input_device_, trt_raw_input_host_, raw_size, cudaMemcpyHostToDevice, trt_stream_);

  launch_letterbox_preprocess(
    static_cast<const uint8_t *>(trt_raw_input_device_), bgr_img.cols, bgr_img.rows, bgr_img.cols * 3,
    static_cast<float *>(trt_input_device_), w, h, static_cast<float>(scale), trt_stream_);
  auto t2 = std::chrono::steady_clock::now();

  // Unlike the earlier DDS-based fusion (see git history), EfficientNMS_TRT's
  // outputs are all FIXED shape, so enqueueV3() goes back to being a cheap
  // dispatch call -- no per-frame blocking wait baked into it, and no need
  // to know num_detections before deciding how much to copy back (always
  // just kMaxNmsOutputBoxes worth, truncated to num_detections after the
  // fact). t2b isolates exactly how long the enqueueV3() call itself takes,
  // for direct comparison against the ~6.5ms it cost under the old DDS
  // approach.
  if (!trt_context_->enqueueV3(trt_stream_)) {
    throw std::runtime_error("YOLOV5: TensorRT enqueueV3 failed");
  }
  auto t2b = std::chrono::steady_clock::now();

  cudaMemcpyAsync(
    trt_output_host_, trt_output_device_, 1 * 25200 * 22 * sizeof(float), cudaMemcpyDeviceToHost,
    trt_stream_);
  cudaMemcpyAsync(
    trt_num_detections_host_, trt_num_detections_device_, sizeof(int32_t), cudaMemcpyDeviceToHost,
    trt_stream_);
  cudaMemcpyAsync(
    trt_detection_scores_host_, trt_detection_scores_device_, kMaxNmsOutputBoxes * sizeof(float),
    cudaMemcpyDeviceToHost, trt_stream_);
  cudaStreamSynchronize(trt_stream_);
  auto t3 = std::chrono::steady_clock::now();

  int num_detections = std::min<int32_t>(trt_num_detections_host_[0], kMaxNmsOutputBoxes);

  auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
  double memcpy_ms = ms(t0, t1), dispatch_ms = ms(t1, t2), enqueue_ms = ms(t2, t2b),
         gpu_wait_d2h_ms = ms(t2b, t3);
  tools::logger()->info(
    "[TRT-TIMING] memcpy={:.2f}ms dispatch={:.2f}ms enqueueV3={:.2f}ms gpu_wait_and_d2h={:.2f}ms "
    "num_detections={}",
    memcpy_ms, dispatch_ms, enqueue_ms, gpu_wait_d2h_ms, num_detections);
  trt_memcpy_stats_.add(memcpy_ms, frame_count);
  trt_dispatch_stats_.add(dispatch_ms, frame_count);
  trt_enqueue_stats_.add(enqueue_ms, frame_count);
  trt_gpu_wait_d2h_stats_.add(gpu_wait_d2h_ms, frame_count);

  // clone()/copy: trt_output_host_/trt_detection_scores_host_ are
  // persistent buffers reused every call, so the caller needs its own copy,
  // not a view into them.
  return TrtInferResult{
    cv::Mat(25200, 22, CV_32F, trt_output_host_).clone(), num_detections,
    std::vector<float>(trt_detection_scores_host_, trt_detection_scores_host_ + num_detections)};
}
#endif

std::list<Armor> YOLOV5::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  // for each row: xywh + classess
  //
  // Rewritten to scan `output` via raw float pointers instead of
  // cv::Mat::at()/row().colRange()/cv::minMaxLoc(): those are all fine for
  // occasional use, but this loop runs across all 25200 candidate rows every
  // frame, and their per-call overhead (bounds checks, temporary cv::Mat
  // header construction, OpenCV API call overhead) adds up at that scale --
  // measured as a bigger share of total per-frame time than the actual GPU
  // inference itself on Jetson. Semantics are unchanged: same sigmoid
  // threshold, same argmax-over-range logic cv::minMaxLoc was doing (just
  // inlined as a manual scan over 4 / 9 elements), same keypoint/rect math.
  std::vector<int> color_ids, num_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;
  const float fscale = static_cast<float>(scale);
  // sigmoid is monotonic, so sigmoid(x) < score_threshold_ iff
  // x < logit(score_threshold_) -- comparing against this raw-logit
  // threshold first lets the (by far common) rejected rows skip the actual
  // sigmoid/exp() call entirely. All 25200 rows were calling exp() every
  // frame before this; only the handful that clear the threshold need it
  // now (still computed via sigmoid() below, for the real confidence value
  // NMS/downstream code needs).
  const float raw_score_threshold = std::log(score_threshold_ / (1.0f - score_threshold_));
  for (int r = 0; r < output.rows; r++) {
    const float * row = output.ptr<float>(r);
    if (row[8] < raw_score_threshold) continue;
    double score = sigmoid(row[8]);

    if (score < score_threshold_) continue;

    //颜色和类别独热向量 (argmax over cols 9-12 and 13-21 respectively)
    int _color_id = 0;
    float best_color = row[9];
    for (int i = 1; i < 4; i++) {
      if (row[9 + i] > best_color) {
        best_color = row[9 + i];
        _color_id = i;
      }
    }
    int _class_id = 0;
    float best_class = row[13];
    for (int i = 1; i < 9; i++) {
      if (row[13 + i] > best_class) {
        best_class = row[13 + i];
        _class_id = i;
      }
    }

    std::vector<cv::Point2f> armor_key_points{
      {row[0] / fscale, row[1] / fscale},
      {row[6] / fscale, row[7] / fscale},
      {row[4] / fscale, row[5] / fscale},
      {row[2] / fscale, row[3] / fscale},
    };

    float min_x = armor_key_points[0].x;
    float max_x = armor_key_points[0].x;
    float min_y = armor_key_points[0].y;
    float max_y = armor_key_points[0].y;

    for (int i = 1; i < armor_key_points.size(); i++) {
      if (armor_key_points[i].x < min_x) min_x = armor_key_points[i].x;
      if (armor_key_points[i].x > max_x) max_x = armor_key_points[i].x;
      if (armor_key_points[i].y < min_y) min_y = armor_key_points[i].y;
      if (armor_key_points[i].y > max_y) max_y = armor_key_points[i].y;
    }

    cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);

    color_ids.emplace_back(_color_id);
    num_ids.emplace_back(_class_id);
    boxes.emplace_back(rect);
    confidences.emplace_back(static_cast<float>(score));
    armors_key_points.emplace_back(std::move(armor_key_points));
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  std::list<Armor> armors;
  for (const auto & i : indices) {
    if (use_roi_) {
      armors.emplace_back(
        color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    } else {
      armors.emplace_back(color_ids[i], num_ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  finalize_armors(armors, bgr_img, frame_count);
  return armors;
}

void YOLOV5::finalize_armors(std::list<Armor> & armors, const cv::Mat & bgr_img, int frame_count)
{
  tmp_img_ = bgr_img;
  for (auto it = armors.begin(); it != armors.end();) {
    if (!check_name(*it)) {
      it = armors.erase(it);
      continue;
    }

    if (!check_type(*it)) {
      it = armors.erase(it);
      continue;
    }
    // 使用传统方法二次矫正角点
    if (use_traditional_) detector_.detect(*it, bgr_img);

    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);
}

#ifdef HAVE_TENSORRT
// TensorRT fused-NMS path: EfficientNMS_TRT's suppression already happened
// on the GPU (see scripts/onnx/fuse_efficient_nms.py) and returns each
// surviving detection's score, but -- unlike the earlier DDS-based fusion's
// selected_indices, which returned the original row index directly -- not
// which of the 25200 candidate rows in raw_output (otherwise byte-identical
// to what parse() scans; the graph surgery is purely additive) it came
// from. This model needs that row's actual 4 keypoints (not just an
// axis-aligned box) for the armor-corner PnP solve, so recover the row by
// matching against a single fresh pass of sigmoid(objectness) over all
// rows -- nearest-match, not exact equality, because the plugin's internal
// sigmoid (score_activation=true) isn't guaranteed bit-identical to this
// host-side one; ties among 25200 continuous float scores are practically
// impossible, so nearest-match is robust. Still cheap versus parse()'s full
// scan: one sigmoid pass over all rows (no per-row argmax/keypoint work)
// plus at most kMaxNmsOutputBoxes linear scans of that.
std::list<Armor> YOLOV5::parse_from_efficient_nms(
  double scale, const cv::Mat & raw_output, const float * detection_scores, int num_detections,
  const cv::Mat & bgr_img, int frame_count)
{
  std::list<Armor> armors;
  const float fscale = static_cast<float>(scale);

  if (num_detections == 0) {
    finalize_armors(armors, bgr_img, frame_count);
    return armors;
  }

  std::vector<float> objectness(static_cast<size_t>(raw_output.rows));
  for (int r = 0; r < raw_output.rows; r++) {
    objectness[static_cast<size_t>(r)] = static_cast<float>(sigmoid(raw_output.ptr<float>(r)[8]));
  }

  for (int i = 0; i < num_detections; i++) {
    float target = detection_scores[i];
    int best_row = 0;
    float best_diff = std::abs(objectness[0] - target);
    for (int r = 1; r < raw_output.rows; r++) {
      float diff = std::abs(objectness[static_cast<size_t>(r)] - target);
      if (diff < best_diff) {
        best_diff = diff;
        best_row = r;
      }
    }

    const float * row = raw_output.ptr<float>(best_row);
    double score = objectness[static_cast<size_t>(best_row)];

    int _color_id = 0;
    float best_color = row[9];
    for (int c = 1; c < 4; c++) {
      if (row[9 + c] > best_color) {
        best_color = row[9 + c];
        _color_id = c;
      }
    }
    int _class_id = 0;
    float best_class = row[13];
    for (int c = 1; c < 9; c++) {
      if (row[13 + c] > best_class) {
        best_class = row[13 + c];
        _class_id = c;
      }
    }

    std::vector<cv::Point2f> armor_key_points{
      {row[0] / fscale, row[1] / fscale},
      {row[6] / fscale, row[7] / fscale},
      {row[4] / fscale, row[5] / fscale},
      {row[2] / fscale, row[3] / fscale},
    };

    float min_x = armor_key_points[0].x;
    float max_x = armor_key_points[0].x;
    float min_y = armor_key_points[0].y;
    float max_y = armor_key_points[0].y;
    for (size_t k = 1; k < armor_key_points.size(); k++) {
      if (armor_key_points[k].x < min_x) min_x = armor_key_points[k].x;
      if (armor_key_points[k].x > max_x) max_x = armor_key_points[k].x;
      if (armor_key_points[k].y < min_y) min_y = armor_key_points[k].y;
      if (armor_key_points[k].y > max_y) max_y = armor_key_points[k].y;
    }
    cv::Rect rect(min_x, min_y, max_x - min_x, max_y - min_y);

    if (use_roi_) {
      armors.emplace_back(
        _color_id, _class_id, static_cast<float>(score), rect, armor_key_points, offset_);
    } else {
      armors.emplace_back(_color_id, _class_id, static_cast<float>(score), rect, armor_key_points);
    }
  }

  finalize_armors(armors, bgr_img, frame_count);
  return armors;
}

std::list<Armor> YOLOV5::postprocess_from_efficient_nms(
  double scale, cv::Mat & raw_output, const float * detection_scores, int num_detections,
  const cv::Mat & bgr_img, int frame_count)
{
  return parse_from_efficient_nms(scale, raw_output, detection_scores, num_detections, bgr_img, frame_count);
}
#endif

bool YOLOV5::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;

  // 保存不确定的图案，用于神经网络的迭代
  // if (name_ok && !confidence_ok) save(armor);

  return name_ok && confidence_ok;
}

bool YOLOV5::check_type(const Armor & armor) const
{
  auto name_ok = (armor.type == ArmorType::small)
                   ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                   : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                      armor.name != ArmorName::outpost);

  // 保存异常的图案，用于神经网络的迭代
  // if (!name_ok) save(armor);

  return name_ok;
}

cv::Point2f YOLOV5::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

void YOLOV5::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {} {}", armor.confidence, COLORS[armor.color], ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }

  if (use_roi_) {
    cv::Scalar green(0, 255, 0);
    cv::rectangle(detection, roi_, green, 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
  cv::imshow("detection", detection);
}

void YOLOV5::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, tmp_img_);
}

double YOLOV5::sigmoid(double x)
{
  if (x > 0)
    return 1.0 / (1.0 + exp(-x));
  else
    return exp(x) / (1.0 + exp(x));
}

std::list<Armor> YOLOV5::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim