#ifndef AUTO_AIM__MT_DETECTOR_HPP
#define AUTO_AIM__MT_DETECTOR_HPP

#include <chrono>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <tuple>

#ifdef HAVE_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>

#include "tasks/auto_aim/yolos/preprocess_kernel.hpp"
#include "tasks/auto_aim/yolos/trt_engine.hpp"
#endif

#include "tasks/auto_aim/yolos/yolov5.hpp"
#include "tools/logger.hpp"
#include "tools/stats.hpp"
#include "tools/thread_safe_queue.hpp"

namespace auto_aim
{
namespace multithread
{

class MultiThreadDetector
{
public:
  MultiThreadDetector(const std::string & config_path, bool debug = false);
  ~MultiThreadDetector();

  void push(cv::Mat img, std::chrono::steady_clock::time_point t);

  std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> pop();  //暂时不支持yolov8

  std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point> debug_pop();

private:
  ov::Core core_;
  ov::CompiledModel compiled_model_;
  std::string device_;
  YOLO yolo_;

  // ---- OpenVINO async path (existing) ----
  tools::ThreadSafeQueue<
    std::tuple<cv::Mat, std::chrono::steady_clock::time_point, ov::InferRequest>>
    queue_{16, [] { tools::logger()->debug("[MultiThreadDetector] queue is full!"); }};

#ifdef HAVE_TENSORRT
  // ---- TensorRT async path ----
  //
  // YOLOV5's own device: TENSORRT path (yolov5.hpp/.cpp) is single-buffered
  // and fully synchronous (H2D copy, inference, and D2H copy each
  // cudaStreamSynchronize immediately), so the GPU and CPU take turns being
  // idle -- confirmed via tegrastats on real hardware (GPU averaging ~15%
  // utilization). This path instead ring-buffers every per-frame CUDA
  // resource across kTrtRingSize slots, so push() for frame N+1 can start
  // CPU preprocess and enqueue its own H2D/kernel/infer/D2H while frame N's
  // are still in flight on the stream -- the CPU only ever waits (via
  // cudaEventSynchronize on that *specific* slot's event, never a full
  // cudaStreamSynchronize which would also wait for later, unrelated
  // enqueued frames) when it's about to reuse a slot whose previous
  // occupant's GPU work hasn't finished yet. Preprocessing itself (letterbox
  // resize + BGR->RGB + normalize + HWC->CHW) runs as a CUDA kernel on the
  // GPU (see preprocess_kernel.cu and YOLOV5::infer_tensorrt_gpu_preprocess,
  // which this mirrors) rather than on the CPU.
  static constexpr int kTrtRingSize = 3;

  bool use_tensorrt_ = false;
  TRTLogger trt_logger_;
  std::unique_ptr<nvinfer1::IRuntime> trt_runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> trt_engine_;
  // Single execution context, reused sequentially across slots by
  // re-pointing addresses with setTensorAddress immediately before each
  // enqueueV3 -- confirmed against the installed TensorRT header
  // (NvInferRuntime.h): enqueueV3's contract only requires a slot's memory
  // not be modified/released before that specific enqueue's stream
  // synchronization, which the per-slot cudaEvent below guarantees. Safe
  // here because push() is only ever called from one producer thread (see
  // mt_standard.cpp), so there's no cross-thread race on the context.
  std::unique_ptr<nvinfer1::IExecutionContext> trt_context_;
  cudaStream_t trt_stream_ = nullptr;
  std::string trt_input_name_, trt_output_name_;
  std::string trt_num_detections_name_, trt_detection_boxes_name_, trt_detection_scores_name_,
    trt_detection_classes_name_;

  struct TrtSlot
  {
    void * input_device = nullptr;   // 640x640x3 float32 CHW -- TensorRT's input binding
    void * output_device = nullptr;
    float * input_host = nullptr;   // pinned, unused once GPU preprocessing writes input_device
                                     // directly, but TensorRT still needs an input binding for
                                     // setTensorAddress; kept for symmetry/debug readback.
    float * output_host = nullptr;  // pinned
    // Raw (unresized) source frame staging, BGR uint8 HWC -- own copy per
    // slot (not shared) because the CPU-side memcpy into raw_input_host
    // happens synchronously in push_tensorrt() the moment it's called, but
    // the GPU doesn't actually read it until its H2D copy executes later on
    // the stream; sharing one buffer across slots would let push() for
    // frame N+1 overwrite frame N's raw pixel data before the GPU has
    // copied it. raw_input_device only needs stream-ordering (safe to
    // share), but keeping it per-slot too avoids the asymmetry.
    void * raw_input_device = nullptr;
    uint8_t * raw_input_host = nullptr;  // pinned
    // Fused-NMS engine's extra outputs (see trt_engine.hpp/
    // scripts/onnx/fuse_efficient_nms.py) -- all FIXED shape (padded to
    // kMaxNmsOutputBoxes), so -- unlike the earlier DDS-based fusion this
    // replaced -- these are plain per-slot device buffers + setTensorAddress
    // like input/output already use, no IOutputAllocator needed.
    // detection_boxes/detection_classes are bound (every output tensor
    // needs an address) but never read back to host -- see the equivalent
    // comment in yolov5.hpp for why.
    void * num_detections_device = nullptr;
    void * detection_boxes_device = nullptr;
    void * detection_scores_device = nullptr;
    void * detection_classes_device = nullptr;
    int32_t * num_detections_host = nullptr;  // pinned, 1 int32
    float * detection_scores_host = nullptr;  // pinned, kMaxNmsOutputBoxes float
    cudaEvent_t d2h_done = nullptr;
    // True from the moment push_tensorrt() dispatches this slot's GPU work
    // until pop_tensorrt() has *fully read* its result (not merely until
    // the GPU finishes -- see trt_slot_mutex_/trt_slot_cv_ below for why
    // that distinction matters). Only ever cleared by pop_tensorrt().
    std::atomic<bool> in_flight{false};
  };
  std::array<TrtSlot, kTrtRingSize> trt_slots_;
  // Guards in_flight transitions and backs the wait/notify push_tensorrt()
  // uses when it's about to reuse a slot still in flight. GPU-completion
  // (cudaEventSynchronize on d2h_done) alone is NOT sufficient backpressure
  // here: with GPU-side preprocessing, push() got fast enough (~2ms) that
  // it can race many frames ahead of a slower consumer (postprocess +
  // imshow/waitKey), reusing a slot's buffers -- since GPU work for that
  // slot already finished -- before pop() has actually read the *previous*
  // occupant's data out of it, silently overwriting an unconsumed result
  // with a newer one. This actually happened: with kTrtRingSize=3 and no
  // proper backpressure, average pipeline_latency ballooned to 260ms+ from
  // queue backlog, and (worse) popped results could have been silently
  // wrong. push_tensorrt() must block until pop_tensorrt() -- not just the
  // GPU -- is done with a slot before reusing it.
  std::mutex trt_slot_mutex_;
  std::condition_variable trt_slot_cv_;
  int trt_next_slot_ = 0;  // producer-side ring cursor, advanced only in push_tensorrt()
  // Raw-frame buffer size (all slots share the same source resolution, so
  // one size suffices); (re)allocated across all slots together if it
  // changes (never, in practice, for a fixed camera/ROI).
  int trt_raw_w_ = -1, trt_raw_h_ = -1;

  // {img.clone(), t, slot_index, letterbox_scale, push_frame_index} -- FIFO
  // order matches slot-reuse order (the ring is strictly round-robin), so
  // no separate slot->queue-entry lookup is needed. Unlike the earlier
  // DDS-based fusion (where the survivor count was known the instant
  // enqueueV3() returned, via TensorRT's notifyShape() callback),
  // num_detections here is regular device data behind an async D2H copy --
  // not known until pop_tensorrt() waits on slot.d2h_done, so it's read
  // from the slot's pinned host buffer there instead of being threaded
  // through this queue. push_frame_index is a simple monotonic counter
  // (not the caller's actual frame_count -- push()/push_tensorrt() are
  // never given one), threaded through purely so the tools::Stats worst-N
  // outlier lists below can say *which* push/pop call was slow, not just
  // that some was.
  tools::ThreadSafeQueue<
    std::tuple<cv::Mat, std::chrono::steady_clock::time_point, int, double, int>>
    trt_queue_{16, [] { tools::logger()->debug("[MultiThreadDetector] TRT queue is full!"); }};

  void push_tensorrt(cv::Mat & img, std::chrono::steady_clock::time_point t);
  std::tuple<cv::Mat, std::chrono::steady_clock::time_point, std::list<Armor>> pop_tensorrt();

  // Per-bracket running mean/stddev/min/max + worst-N outlier frames for
  // the [MT-TRT-TIMING]/pipeline_latency numbers -- see the equivalent
  // tools::Stats members in yolov5.hpp for why (frame-to-frame variance is
  // real and worth quantifying, not just eyeballing individual log lines).
  tools::Stats trt_push_memcpy_stats_{"MT push memcpy"};
  tools::Stats trt_push_dispatch_stats_{"MT push dispatch"};
  tools::Stats trt_push_enqueue_stats_{"MT push enqueueV3"};
  // push total / pop total isolate each side's own ACTIVE cost -- push
  // total deliberately starts AFTER push_tensorrt()'s backpressure wait
  // (blocking for a free ring slot), so together with pop total these are
  // for diagnosing WHICH STAGE is doing the work, not for reading off
  // real-world achievable FPS directly (see [MT-THROUGHPUT] below, which is
  // a theoretical ceiling assuming infinite buffering -- it doesn't see
  // backpressure stalls or consumer-side imshow/waitKey cost either).
  // trt_push_wall_stats_ is the one that DOES answer "what FPS am I really
  // getting": it wraps the ENTIRE push() call, backpressure wait included
  // -- exactly what a caller (e.g. mt_standard.cpp's producer loop) actually
  // pays per frame, and because of that backpressure coupling it converges
  // to reflect whichever side is truly the bottleneck once the pipeline
  // reaches steady state, consumer-side cost included.
  tools::Stats trt_push_total_stats_{"MT push total (excl. backpressure)"};
  tools::Stats trt_push_wall_stats_{"MT push wall (incl. backpressure)"};
  tools::Stats trt_pop_total_stats_{"MT pop total"};
  tools::Stats trt_pipeline_latency_stats_{"MT pipeline_latency"};
  int trt_pop_count_ = 0;      // gates a periodic throughput-ceiling log, see pop_tensorrt()
  int trt_push_counter_ = 0;   // monotonic, see the trt_queue_ comment above
#endif
};

}  // namespace multithread

}  // namespace auto_aim

#endif  // AUTO_AIM__MT_DETECTOR_HPP
