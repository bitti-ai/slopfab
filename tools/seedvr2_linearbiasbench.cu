// Compare the original FP32 accumulator + bias kernel against a fused Lt bias.
#include "../src/seedvr2/lt_linear.cuh"
#include "slopfab/cuda/device.h"
#include "slopfab/cuda/gemm.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace slopfab;
using BFloat = __nv_bfloat16;

__global__ void fill_values(BFloat* values, size_t count, unsigned seed, float scale) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    unsigned v = unsigned(i) + seed;
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    v *= 0x846ca68bu;
    v ^= v >> 16;
    values[i] = __float2bfloat16((float(v & 65535u) / 32767.5f - 1) * scale);
  }
}

__global__ void add_bias(const float* input, const BFloat* bias, BFloat* output, size_t count,
                          int channels) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    output[i] = __float2bfloat16(input[i] + __bfloat162float(bias[i % channels]));
}

int main() {
  try {
    cuda::set_device(0);
    cublasHandle_t blas = nullptr;
    SLOPFAB_CUBLAS_CHECK(cuda::cublas_create(&blas));
    seedvr2::LtLinear fused;
    cudaEvent_t start, stop;
    SLOPFAB_CUDA_CHECK(cudaEventCreate(&start));
    SLOPFAB_CUDA_CHECK(cudaEventCreate(&stop));
    std::printf("cuBLAS major %d\n", cuda::cublas_loaded_major());
    for (int rows : {1, 64, 2048, 7200})
      for (int co : {2560, 6912, 7680}) {
        const int ci = 2560;
        const size_t count = size_t(rows) * co;
        cuda::DeviceBuffer<BFloat> x(size_t(rows) * ci), w(size_t(co) * ci), bias(co), old(count),
            actual(count);
        cuda::DeviceBuffer<float> accumulator(count);
        auto fill = [](cuda::DeviceBuffer<BFloat>& target, unsigned seed, float scale) {
          fill_values<<<int((target.size() + 255) / 256), 256>>>(target.get(), target.size(), seed,
                                                                 scale);
        };
        fill(x, 13, 0.2f);
        fill(w, 51, 0.02f);
        fill(bias, 17, 0.5f);
        auto reference = [&] {
          const float alpha = 1, beta = 0;
          SLOPFAB_CUBLAS_CHECK(cuda::cublas_gemm_ex(
              blas, CUBLAS_OP_T, CUBLAS_OP_N, co, rows, ci, &alpha, w.get(), CUDA_R_16BF, ci,
              x.get(), CUDA_R_16BF, ci, &beta, accumulator.get(), CUDA_R_32F, co,
              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
          add_bias<<<int((count + 255) / 256), 256>>>(accumulator.get(), bias.get(), old.get(),
                                                     count, co);
          SLOPFAB_CUDA_CHECK(cudaGetLastError());
        };
        auto optimized = [&] {
          return fused.forward(x.get(), w.get(), bias.get(), actual.get(), rows, ci, co);
        };
        reference();
        if (!optimized()) {
          std::printf("rows=%d co=%d: Lt unsupported\n", rows, co);
          continue;
        }
        SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
        auto measure = [&](auto run) {
          for (int repeat = 0; repeat < 3; ++repeat)
            run();
          SLOPFAB_CUDA_CHECK(cudaEventRecord(start));
          for (int repeat = 0; repeat < 20; ++repeat)
            run();
          SLOPFAB_CUDA_CHECK(cudaEventRecord(stop));
          SLOPFAB_CUDA_CHECK(cudaEventSynchronize(stop));
          float ms = 0;
          SLOPFAB_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
          return ms / 20;
        };
        const float before = measure(reference), after = measure(optimized);
        std::vector<BFloat> a(count), b(count);
        old.copy_to_host(a.data(), count);
        actual.copy_to_host(b.data(), count);
        size_t different = 0;
        double squared = 0;
        float max_error = 0;
        for (size_t i = 0; i < count; ++i) {
          float error = std::abs(__bfloat162float(a[i]) - __bfloat162float(b[i]));
          if (!std::isfinite(error))
            return 2;
          different += error != 0;
          squared += double(error) * error;
          max_error = std::max(max_error, error);
        }
        std::printf("rows=%d co=%d: old %.4f ms, Lt %.4f ms; max %.7g rms %.7g changed %zu/%zu; "
                    "avoids %.1f MiB accumulator\n", rows, co, before, after, max_error,
                    std::sqrt(squared / count), different, count, count * sizeof(float) / 1048576.0);
        if (max_error > 0.01f)
          return 3;
      }
    SLOPFAB_CUDA_CHECK(cudaEventDestroy(start));
    SLOPFAB_CUDA_CHECK(cudaEventDestroy(stop));
    SLOPFAB_CUBLAS_CHECK(cuda::cublas_destroy(blas));
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
