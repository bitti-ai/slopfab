#include "slopfab/cuda/lora.cuh"
#include <algorithm>
#include <cstring>
#include "slopfab/cuda/cublas_dispatch.h"
#include "slopfab/cuda/deterministic_gemm.cuh"
#include "slopfab/cuda/gemm.cuh"
#include "slopfab/cuda/nn_kernels.cuh"
#include "slopfab/dtype.h"

namespace slopfab::cuda {
namespace {
constexpr int kRows = 256;
}

void LoraRunner::attach(const void* key, const std::vector<LoraFactors>& host, int offset, int out,
                        uint8_t* staged_host, uint8_t* staged_device, size_t* staged_offset) {
  if ((staged_host != nullptr || staged_device != nullptr || staged_offset != nullptr) &&
      !(staged_host && staged_device && staged_offset))
    throw std::invalid_argument("LoRA: incomplete weight staging buffers");
  auto upload = [&](DeviceBuffer<__nv_bfloat16>& device, const float* p,
                    size_t n) -> const __nv_bfloat16* {
    std::vector<uint16_t> bf(n);
    for (size_t i = 0; i < n; ++i)
      bf[i] = f32_to_bf16(p[i]);
    if (staged_host) {
      std::memcpy(staged_host + *staged_offset, bf.data(), n * 2);
      const auto* result = reinterpret_cast<const __nv_bfloat16*>(staged_device + *staged_offset);
      *staged_offset += (n * 2 + 255) / 256 * 256;
      return result;
    }
    device.allocate(n);
    SLOPFAB_CUDA_CHECK(cudaMemcpy(device.get(), bf.data(), n * 2, cudaMemcpyHostToDevice));
    return device.get();
  };
  for (const auto& h : host) {
    if (offset < 0 || out <= 0 || offset > h.out || out > h.out - offset)
      throw std::runtime_error("LoRA: invalid CUDA output slice");
    Factors f;
    f.rank = h.rank;
    f.in = h.in;
    f.out = out;
    f.a_ptr = upload(f.a, h.a.data(), h.a.size());
    f.b_ptr = upload(f.b, h.b.data() + static_cast<size_t>(offset) * h.rank,
                     static_cast<size_t>(out) * h.rank);
    if (hidden_.size() < static_cast<size_t>(kRows) * h.rank)
      hidden_.allocate(static_cast<size_t>(kRows) * h.rank);
    if (delta_.size() < static_cast<size_t>(kRows) * out)
      delta_.allocate(static_cast<size_t>(kRows) * out);
    factors_[key].push_back(std::move(f));
  }
}

size_t LoraRunner::weight_bytes() const {
  size_t result = 0;
  for (const auto& entry : factors_)
    for (const auto& f : entry.second)
      result += f.a.nbytes() + f.b.nbytes();
  return result;
}

void LoraRunner::apply(const void* key, const __nv_bfloat16* input, int rows, __nv_bfloat16* output,
                       cublasHandle_t blas, cudaStream_t stream, bool exact) {
  const auto found = factors_.find(key);
  if (found == factors_.end())
    return;
  auto gemm = [&](const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* y, int m, int n,
                  int k) {
    if (exact) {
      const int tiled = n % 16 == 0 && k % 16 == 0 ? m / 64 * 64 : 0;
      if (tiled)
        launch_deterministic_bf16_gemm_nt(x, w, nullptr, y, tiled, n, k, DenseGemmBias::kNone, 0, 0,
                                          stream);
      if (tiled != m)
        launch_deterministic_scalar_gemm_nt(x, w, nullptr, y, m - tiled, n, k,
                                            DenseGemmMode::kBFloat16, DenseGemmBias::kNone, tiled,
                                            tiled, stream);
    } else {
      const float one = 1.0f, zero = 0.0f;
      SLOPFAB_CUBLAS_CHECK(cublas_gemm_ex(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &one, w,
                                          CUDA_R_16BF, k, x, CUDA_R_16BF, k, &zero, y, CUDA_R_16BF,
                                          n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
  };
  for (const auto& f : found->second) {
    for (int row = 0; row < rows; row += kRows) {
      const int count = std::min(kRows, rows - row);
      gemm(input + static_cast<size_t>(row) * f.in, f.a_ptr, hidden_.get(), count, f.rank, f.in);
      gemm(hidden_.get(), f.b_ptr, delta_.get(), count, f.out, f.rank);
      launch_add_bf16(output + static_cast<size_t>(row) * f.out, delta_.get(),
                      static_cast<size_t>(count) * f.out, stream);
    }
  }
}
} // namespace slopfab::cuda
