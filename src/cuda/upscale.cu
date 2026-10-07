#include "../upscale/backend.h"
#include "slopfab/cuda/device.h"
#include "slopfab/cuda/cublas_dispatch.h"
#include "slopfab/cuda/reference_buffer.cuh"

namespace slopfab::upscale_detail {
namespace {
struct CudaTensor : Tensor {
  std::shared_ptr<cuda::ReferenceBufferPool> pool;
  cuda::ReferenceBuffer<float> lease;
  cuda::DeviceBuffer<float> data;

  explicit CudaTensor(size_t count) : data(count) {
  }

  CudaTensor(size_t count, std::shared_ptr<cuda::ReferenceBufferPool> owner)
      : pool(std::move(owner)), lease(count, *pool) {
  }
};

float* pointer(const Buffer& b) {
  auto& t = static_cast<CudaTensor&>(*b);
  return t.pool ? t.lease.get() : t.data.get();
}

__global__ void im2col(const float* x, float* col, Parameters p, unsigned offset, unsigned pixels) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned k = p.input * 9;
  if (i >= pixels * k)
    return;
  const unsigned pixel = offset + i / k, tap = i % k;
  const int y = int(pixel / p.width) + int(tap % 9 / 3) - 1;
  const int xx = int(pixel % p.width) + int(tap % 3) - 1;
  col[i] = y < 0 || xx < 0 || y >= int(p.height) || xx >= int(p.width)
               ? 0.0f
               : x[(y * p.width + xx) * p.input + tap / 9];
}

__global__ void pointwise(const float* x, const float* y, const float* bias, float* out,
                          Parameters p) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= p.count)
    return;
  const unsigned c = i % p.output, pixel = i / p.output;
  if (p.op == kConv) {
    const float v = out[i] + bias[c];
    out[i] = p.leaky && v < 0 ? v * 0.2f : v;
  } else if (p.op == kConcat) {
    out[i] = c < p.input ? x[pixel * p.input + c] : y[pixel * (p.output - p.input) + c - p.input];
  } else if (p.op == kResidual) {
    out[i] = x[i] + p.scale * y[i];
  } else {
    out[i] = x[((pixel / p.width / 2) * (p.width / 2) + pixel % p.width / 2) * p.output + c];
  }
}

void blas_check(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("Real-ESRGAN: cuBLAS error " + std::to_string(int(status)));
}

class CudaBackend final : public Backend {
  cublasHandle_t handle_ = nullptr;
  cuda::DeviceBuffer<float> columns_;
  std::shared_ptr<cuda::ReferenceBufferPool> pool_;

public:
  CudaBackend() {
    int device = 0;
    SLOPFAB_CUDA_CHECK(cudaGetDevice(&device));
    // Returned buffers keep the pool alive. Reuse is ordered on the default stream.
    pool_ = std::shared_ptr<cuda::ReferenceBufferPool>(new cuda::ReferenceBufferPool,
                                                       [device](cuda::ReferenceBufferPool* pool) {
                                                         int previous = 0;
                                                         cudaGetDevice(&previous);
                                                         cudaSetDevice(device);
                                                         cudaStreamSynchronize(nullptr);
                                                         delete pool;
                                                         cudaSetDevice(previous);
                                                       });
    blas_check(cuda::cublas_create(&handle_));
    // FP32 accumulation without TF32 truncation is important for deep RRDBs.
    const auto status = cuda::cublas_set_math_mode(handle_, CUBLAS_PEDANTIC_MATH);
    if (status != CUBLAS_STATUS_SUCCESS) {
      cuda::cublas_destroy(handle_);
      blas_check(status);
    }
  }

  ~CudaBackend() override {
    cuda::cublas_destroy(handle_);
  }

  Buffer allocate(size_t count) override {
    return std::make_shared<CudaTensor>(count, pool_);
  }

  Buffer upload(const std::vector<float>& v) override {
    // Persistent weights must not occupy slots in the activation reuse pool.
    auto t = std::make_shared<CudaTensor>(v.size());
    SLOPFAB_CUDA_CHECK(
        cudaMemcpy(pointer(t), v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
    return t;
  }

  std::vector<float> download(const Buffer& t, size_t count) override {
    std::vector<float> v(count);
    SLOPFAB_CUDA_CHECK(
        cudaMemcpy(v.data(), pointer(t), count * sizeof(float), cudaMemcpyDeviceToHost));
    return v;
  }

  void run(const Parameters& p, const Buffer& x, const Buffer& y, const Buffer& bias,
           const Buffer& out) override {
    if (p.op == kConv) {
      constexpr unsigned kChunkPixels = 32768;
      const unsigned pixels = p.height * p.width;
      const size_t count = size_t(std::min(pixels, kChunkPixels)) * p.input * 9;
      if (columns_.size() < count)
        columns_.allocate(count);
      const float alpha = 1, beta = 0;
      // Row-major [pixels,K] * [output,K]^T -> [pixels,output].
      for (unsigned offset = 0; offset < pixels; offset += kChunkPixels) {
        const unsigned rows = std::min(pixels - offset, kChunkPixels);
        const unsigned elements = rows * p.input * 9;
        im2col<<<(elements + 255) / 256, 256>>>(pointer(x), columns_.get(), p, offset, rows);
        SLOPFAB_CUDA_CHECK(cudaGetLastError());
        blas_check(cuda::cublas_sgemm(handle_, CUBLAS_OP_T, CUBLAS_OP_N, int(p.output), int(rows),
                                      int(p.input * 9), &alpha, pointer(y), int(p.input * 9),
                                      columns_.get(), int(p.input * 9), &beta,
                                      pointer(out) + size_t(offset) * p.output, int(p.output)));
      }
    }
    pointwise<<<(p.count + 255) / 256, 256>>>(pointer(x), pointer(y), pointer(bias), pointer(out),
                                              p);
    SLOPFAB_CUDA_CHECK(cudaGetLastError());
  }
};
} // namespace

std::unique_ptr<Backend> make_cuda_backend() {
  return std::make_unique<CudaBackend>();
}
} // namespace slopfab::upscale_detail
