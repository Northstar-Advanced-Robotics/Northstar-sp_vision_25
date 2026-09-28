#ifndef AUTO_AIM__PREPROCESS_KERNEL_HPP
#define AUTO_AIM__PREPROCESS_KERNEL_HPP

#ifdef HAVE_TENSORRT
#include <cstdint>
#include <cuda_runtime_api.h>

namespace auto_aim
{

// GPU-resident letterbox resize + BGR->RGB + normalize(/255) + HWC->CHW,
// fused into a single kernel launch (see preprocess_kernel.cu) -- replaces
// the CPU cv::resize + cv::dnn::blobFromImage chain (~5ms combined,
// measured on Jetson) with one GPU pass that runs on otherwise-idle
// hardware (tegrastats showed GPU averaging ~15-30% utilization). `src` is
// a device pointer to a raw BGR uint8 image (HWC layout, row pitch
// src_step bytes -- handles both contiguous frames and ROI-cropped views);
// `dst` is a device pointer to a 640x640x3 float32 CHW buffer (exactly
// TensorRT's expected input layout -- can be the engine's own input
// binding, written directly, no extra copy). Async on `stream`; caller is
// responsible for synchronizing before reading dst or reusing src.
void launch_letterbox_preprocess(
  const uint8_t * src, int src_w, int src_h, int src_step, float * dst, int active_w, int active_h,
  float scale, cudaStream_t stream);

}  // namespace auto_aim

#endif  // HAVE_TENSORRT
#endif  // AUTO_AIM__PREPROCESS_KERNEL_HPP
