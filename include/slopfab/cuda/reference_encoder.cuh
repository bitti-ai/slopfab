#pragma once
#include <cuda_fp16.h>

#include "slopfab/cuda/cublas_dispatch.h"
#include "slopfab/cuda/device.h"
#include "slopfab/cuda/profile.h"

namespace slopfab::cuda {
// FP32 contractions with bounded im2col scratch. Used only by reference
// encoders; existing single-image and audio decoder arithmetic is unchanged.
class ReferenceEncoderOps {
 public:
  explicit ReferenceEncoderOps(cudaStream_t stream);
  ~ReferenceEncoderOps();
  ReferenceEncoderOps(const ReferenceEncoderOps&) = delete;
  ReferenceEncoderOps& operator=(const ReferenceEncoderOps&) = delete;
  DeviceBuffer<float> conv3d(const float* x, const __half* weight,
                             const __half* bias, int cin, int cout, int frames,
                             int height, int width, int kernel,
                             int spatial_stride, int temporal_stride,
                             bool downsample);
  DeviceBuffer<float> conv1d(const float* x, const float* weight,
                             const float* bias, int batch, int cin, int cout,
                             int length, int kernel, int stride, int padding,
                             int dilation);
  DeviceBuffer<float> linear(const float* x, const float* weight,
                             const float* bias, int rows, int in, int out);
  DeviceBuffer<float> norm(const float* x, const float* weight,
                           const float* bias, int rows, int width);
  DeviceBuffer<float> transpose(const float* x, int batch, int rows,
                                int columns);
  DeviceBuffer<float> attention(const float* qkv, const float* qb,
                                const float* kb, const float* vb, int batch,
                                int length);
  void snake(float* x, const float* alpha, int batch, int channels, int length);
  void add(float* x, const float* branch, size_t count);
  void geglu(float* x, const float* gate, size_t count);
  void report_memory(const char* label) const { memory_.report(label); }

 private:
  cudaStream_t stream_;
  cublasHandle_t blas_ = nullptr;
  ReferenceMemoryProfiler memory_;
};
void reference_groupnorm(const float* x, const __half* weight,
                         const __half* bias, float* y, int channels, int frames,
                         int height, int width, cudaStream_t stream);
}  // namespace slopfab::cuda
