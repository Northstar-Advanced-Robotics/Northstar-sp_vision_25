#ifndef AUTO_AIM__YOLOV5_HPP
#define AUTO_AIM__YOLOV5_HPP

#include <list>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>

#include <memory>
#endif

#ifdef HAVE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <memory>

#include "tasks/auto_aim/yolos/preprocess_kernel.hpp"
#include "tasks/auto_aim/yolos/trt_engine.hpp"
#endif

#include "tasks/auto_aim/armor.hpp"
#include "tasks/auto_aim/detector.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/stats.hpp"

namespace auto_aim
{
class YOLOV5 : public YOLOBase
{
public:
  YOLOV5(const std::string & config_path, bool debug);
  ~YOLOV5();

  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

#ifdef HAVE_TENSORRT
  // For device: TENSORRT's fused-NMS engine (see trt_engine.hpp/
  // scripts/onnx/fuse_efficient_nms.py): NMS suppression already happened
  // on the GPU via TensorRT's EfficientNMS_TRT plugin, which returns fixed-
  // shape (padded to kMaxNmsOutputBoxes) detection_scores -- the original
  // row each surviving detection came from is recovered by matching that
  // score against raw_output's objectness column (see
  // parse_from_efficient_nms in yolov5.cpp for why: this model needs the
  // matched row's 4 keypoints, not just an axis-aligned box, for the armor-
  // corner PnP solve, and EfficientNMS_TRT doesn't return the original row
  // index). Used by MultiThreadDetector (async path) instead of
  // postprocess() -- see YOLOBase::postprocess_from_efficient_nms for why
  // this is a virtual method with a default-throwing body rather than pure
  // virtual (YOLOV8/YOLO11 never support it).
  std::list<Armor> postprocess_from_efficient_nms(
    double scale, cv::Mat & raw_output, const float * detection_scores, int num_detections,
    const cv::Mat & bgr_img, int frame_count) override;
#endif

private:
  std::string device_, model_path_;
  std::string save_path_, debug_path_;
  bool debug_, use_roi_, use_traditional_;

  const int class_num_ = 13;
  const float nms_threshold_ = 0.3;
  const float score_threshold_ = 0.7;
  double min_confidence_, binary_threshold_;

  ov::Core core_;
  ov::CompiledModel compiled_model_;

  bool use_cuda_ = false;
#ifdef HAVE_ONNXRUNTIME
  std::unique_ptr<Ort::Env> ort_env_;
  std::unique_ptr<Ort::Session> ort_session_;
  std::string ort_input_name_, ort_output_name_;
  cv::Mat infer_cuda(const cv::Mat & input);
#endif

  bool use_tensorrt_ = false;
#ifdef HAVE_TENSORRT
  TRTLogger trt_logger_;
  std::unique_ptr<nvinfer1::IRuntime> trt_runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> trt_engine_;
  std::unique_ptr<nvinfer1::IExecutionContext> trt_context_;
  cudaStream_t trt_stream_ = nullptr;
  void * trt_input_device_ = nullptr;
  void * trt_output_device_ = nullptr;
  // Pinned (page-locked) host buffers, allocated once and reused every
  // frame -- pageable host memory (e.g. plain cv::Mat/new[]) forces the CUDA
  // driver to stage through an internal pinned bounce buffer on every
  // cudaMemcpyAsync, roughly doubling H2D/D2H transfer time. Preprocessing
  // writes directly into trt_input_host_ (no separate host buffer + copy).
  float * trt_input_host_ = nullptr;
  float * trt_output_host_ = nullptr;
  std::string trt_input_name_, trt_output_name_;

  // GPU-resident preprocessing: fuses letterbox resize + BGR->RGB +
  // normalize + HWC->CHW into one CUDA kernel (see preprocess_kernel.cu/
  // .hpp), replacing the CPU cv::resize + cv::dnn::blobFromImage chain
  // (~5ms combined, measured) with a raw-frame memcpy+H2D + kernel launch
  // on otherwise-idle GPU (tegrastats showed ~15-30% average utilization).
  // trt_raw_input_host_/device_ hold the *unprocessed* source frame (BGR
  // uint8, HWC), lazily (re)allocated -- like letterbox_canvas_ -- only
  // when the source resolution changes.
  void * trt_raw_input_device_ = nullptr;
  uint8_t * trt_raw_input_host_ = nullptr;  // pinned
  int trt_raw_w_ = -1, trt_raw_h_ = -1;

  // Fused-NMS engine's extra outputs (see trt_engine.hpp/
  // scripts/onnx/fuse_efficient_nms.py): all FIXED shape (padded to
  // kMaxNmsOutputBoxes), so -- unlike the earlier DDS-based fusion this
  // replaced -- no IOutputAllocator is needed, just plain device buffers
  // and setTensorAddress() like input/output already use.
  // detection_boxes/detection_classes are bound (TensorRT requires every
  // output tensor to have an address) but never read back to host: we
  // don't use EfficientNMS_TRT's box (recomputed from keypoints instead)
  // or class (always 0, single "class" = objectness) outputs.
  void * trt_num_detections_device_ = nullptr;
  void * trt_detection_boxes_device_ = nullptr;
  void * trt_detection_scores_device_ = nullptr;
  void * trt_detection_classes_device_ = nullptr;
  int32_t * trt_num_detections_host_ = nullptr;   // pinned, 1 int32
  float * trt_detection_scores_host_ = nullptr;   // pinned, kMaxNmsOutputBoxes float

  struct TrtInferResult
  {
    cv::Mat raw_output;  // 25200x22, cloned (unchanged from before fusion)
    int num_detections;
    std::vector<float> detection_scores;  // num_detections valid entries (of kMaxNmsOutputBoxes)
  };
  TrtInferResult infer_tensorrt_gpu_preprocess(
    const cv::Mat & bgr_img, int w, int h, double scale, int frame_count);

  std::list<Armor> parse_from_efficient_nms(
    double scale, const cv::Mat & raw_output, const float * detection_scores, int num_detections,
    const cv::Mat & bgr_img, int frame_count);

  // Per-bracket running mean/stddev/min/max + worst-N outlier frames for
  // the [TRT-TIMING]/[PARSE-TIMING] numbers below -- these vary noticeably
  // frame to frame (thermal throttling, OS scheduling jitter, occasional
  // slow video-decode frames), and eyeballing individual log lines doesn't
  // answer "how much" or "which frames". Auto-logs a [STATS] summary every
  // 200 frames and once more at destruction.
  tools::Stats trt_memcpy_stats_{"TRT memcpy"};
  tools::Stats trt_dispatch_stats_{"TRT dispatch"};
  tools::Stats trt_enqueue_stats_{"TRT enqueueV3"};
  tools::Stats trt_gpu_wait_d2h_stats_{"TRT gpu_wait_and_d2h"};
#endif
  tools::Stats parse_stats_{"parse"};
  // Wraps the ENTIRE detect() call (both TensorRT and OpenVINO/CUDA
  // backends) -- this is the number that actually dictates achievable FPS
  // in the single-threaded path (1000/mean = sustainable ceiling); the
  // per-bracket stats above are its components, not a substitute for it
  // (they don't cover ROI/scale setup, and for the non-TensorRT path,
  // detect() also has the [LETTERBOX-TIMING] cost this doesn't separately
  // track as a Stats object).
  tools::Stats detect_total_stats_{"detect total"};

  cv::Mat infer_openvino(const cv::Mat & input);

  cv::Rect roi_;
  cv::Point2f offset_;
  cv::Mat tmp_img_;

  // Persistent letterbox canvas, reused every frame instead of allocating +
  // zero-filling a fresh 640x640x3 buffer each call (measured as a real,
  // non-trivial per-frame cost on Jetson). Valid whenever letterbox_w_/
  // letterbox_h_ match the current frame's computed resize target; re-zeroed
  // if not (e.g. first frame, or source resolution changed).
  cv::Mat letterbox_canvas_;
  int letterbox_w_ = -1, letterbox_h_ = -1;

  Detector detector_;
  friend class MultiThreadDetector;

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;

  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  std::list<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  // Shared tail of parse() and (TensorRT-only) parse_from_efficient_nms():
  // name/type filtering, optional traditional-CV keypoint refinement,
  // center_norm, and debug drawing -- backend-independent per-detection
  // postprocessing that must behave identically regardless of which path
  // produced the (still unfiltered/unrefined) armors list.
  void finalize_armors(std::list<Armor> & armors, const cv::Mat & bgr_img, int frame_count);

  void save(const Armor & armor) const;
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;
  double sigmoid(double x);
};

}  // namespace auto_aim

#endif  //AUTO_AIM__YOLOV5_HPP