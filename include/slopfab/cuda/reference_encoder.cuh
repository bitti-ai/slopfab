#pragma once
#include <cuda_fp16.h>

#include "slopfab/cuda/cublas_dispatch.h"
#include "slopfab/cuda/device.h"
#include "slopfab/cuda/reference_buffer.cuh"
#include "slopfab/cuda/profile.h"

namespace slopfab::cuda {
// Bounded im2col contractions. Video supports FP16 storage/FP32 accumulation;
// audio and the reference FP32 authority retain their original arithmetic.
class ReferenceEncoderOps {
public:
  explicit ReferenceEncoderOps(cudaStream_t stream);
  ~ReferenceEncoderOps();
  ReferenceEncoderOps(const ReferenceEncoderOps&) = delete;
  ReferenceEncoderOps& operator=(const ReferenceEncoderOps&) = delete;
  ReferenceBuffer<float> conv3d(const float* x, const __half* weight, const __half* bias, int cin,
                                int cout, int frames, int height, int width, int kernel,
                                int spatial_stride, int temporal_stride, bool downsample);
  ReferenceBuffer<__half> conv3d(const __half* x, const __half* weight, const __half* bias, int cin,
                                 int cout, int frames, int height, int width, int kernel,
                                 int spatial_stride, int temporal_stride, bool downsample);
  ReferenceBuffer<float> conv1d(const float* x, const float* weight, const float* bias, int batch,
                                int cin, int cout, int length, int kernel, int stride, int padding,
                                int dilation);
  ReferenceBuffer<float> linear(const float* x, const float* weight, const float* bias, int rows,
                                int in, int out);
  ReferenceBuffer<float> norm(const float* x, const float* weight, const float* bias, int rows,
                              int width);
  ReferenceBuffer<float> transpose(const float* x, int batch, int rows, int columns);
  ReferenceBuffer<float> attention(const float* qkv, const float* qb, const float* kb,
                                   const float* vb, int batch, int length);
  void snake(float* x, const float* alpha, int batch, int channels, int length);
  void add(float* x, const float* branch, size_t count);
  void add(__half* x, const __half* branch, size_t count);
  void geglu(float* x, const float* gate, size_t count);

  template <typename T> ReferenceBuffer<T> allocate(size_t count) {
    return ReferenceBuffer<T>(count, activations_);
  }

  void report_memory(const char* label) const;

private:
  cudaStream_t stream_;
  cublasHandle_t blas_ = nullptr;
  ReferenceMemoryProfiler memory_;
  ReferenceBufferPool activations_, scratch_;
};

void reference_groupnorm(const float* x, const __half* weight, const __half* bias, float* y,
                         int channels, int frames, int height, int width, cudaStream_t stream);
void reference_groupnorm(const __half* x, const __half* weight, const __half* bias, __half* y,
                         int channels, int frames, int height, int width, cudaStream_t stream);
} // namespace slopfab::cuda
