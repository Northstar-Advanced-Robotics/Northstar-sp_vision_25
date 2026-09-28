#ifdef HAVE_TENSORRT
#include "preprocess_kernel.hpp"

namespace auto_aim
{

namespace
{

// One thread per output pixel of the fixed 640x640 canvas. Pixels inside
// [0,active_w)x[0,active_h) are bilinear-sampled from the source image
// (matching cv::resize's default INTER_LINEAR pixel-center convention:
// sample position = (out+0.5)*inv_scale - 0.5); pixels outside that (the
// letterbox padding) are left at 0, matching the CPU path's
// black-canvas-then-paste behavior.
__global__ void letterbox_preprocess_kernel(
  const uint8_t * src, int src_w, int src_h, int src_step, float * dst, int active_w, int active_h,
  float inv_scale)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= 640 || y >= 640) return;

  float r = 0.0f, g = 0.0f, b = 0.0f;
  if (x < active_w && y < active_h) {
    float sx = (x + 0.5f) * inv_scale - 0.5f;
    float sy = (y + 0.5f) * inv_scale - 0.5f;
    int x0 = static_cast<int>(floorf(sx));
    int y0 = static_cast<int>(floorf(sy));
    float fx = sx - x0;
    float fy = sy - y0;
    int x1 = min(max(x0 + 1, 0), src_w - 1);
    int y1 = min(max(y0 + 1, 0), src_h - 1);
    x0 = min(max(x0, 0), src_w - 1);
    y0 = min(max(y0, 0), src_h - 1);

    const uint8_t * p00 = src + static_cast<size_t>(y0) * src_step + x0 * 3;
    const uint8_t * p01 = src + static_cast<size_t>(y0) * src_step + x1 * 3;
    const uint8_t * p10 = src + static_cast<size_t>(y1) * src_step + x0 * 3;
    const uint8_t * p11 = src + static_cast<size_t>(y1) * src_step + x1 * 3;

#pragma unroll
    for (int c = 0; c < 3; c++) {
      float top = p00[c] * (1.0f - fx) + p01[c] * fx;
      float bot = p10[c] * (1.0f - fx) + p11[c] * fx;
      float val = (top * (1.0f - fy) + bot * fy) * (1.0f / 255.0f);
      // src is BGR (c=0 blue, 1 green, 2 red); write RGB order to match
      // blobFromImage's swapRB=true behavior the old CPU path relied on.
      if (c == 0)
        b = val;
      else if (c == 1)
        g = val;
      else
        r = val;
    }
  }

  // dst is CHW: channel 0 = R, channel 1 = G, channel 2 = B.
  int plane = 640 * 640;
  dst[0 * plane + y * 640 + x] = r;
  dst[1 * plane + y * 640 + x] = g;
  dst[2 * plane + y * 640 + x] = b;
}

}  // namespace

void launch_letterbox_preprocess(
  const uint8_t * src, int src_w, int src_h, int src_step, float * dst, int active_w, int active_h,
  float scale, cudaStream_t stream)
{
  dim3 block(32, 32);
  dim3 grid((640 + block.x - 1) / block.x, (640 + block.y - 1) / block.y);
  letterbox_preprocess_kernel<<<grid, block, 0, stream>>>(
    src, src_w, src_h, src_step, dst, active_w, active_h, 1.0f / scale);
}

}  // namespace auto_aim

#endif  // HAVE_TENSORRT
