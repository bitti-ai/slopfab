// Standalone Sol-Attn microbenchmark. Defaults are representative and safe;
// pass --seq 37710 --heads 56 for the production H3 geometry.
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "vidfab/cuda/attention.cuh"
#include "vidfab/cuda/device.h"
#include "vidfab/cuda/workspace.cuh"

namespace {
__global__ void fill(__nv_bfloat16* p, size_t n, unsigned seed, float amplitude) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x) {
    unsigned x = unsigned(i) ^ seed;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    p[i] = __float2bfloat16_rn((float(x & 65535u) / 32767.5f - 1.0f) * amplitude);
  }
}

float time_backend(cublasHandle_t blas, const __nv_bfloat16* q, const __nv_bfloat16* k,
                   const __nv_bfloat16* v, __nv_bfloat16* out,
                   const vidfab::cuda::AttentionConfig& cfg,
                   vidfab::cuda::AttentionBackend backend, vidfab::cuda::Workspace& ws,
                   int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) {
    ws.clear();
    vidfab::cuda::attention_forward(blas, nullptr, q, k, v, out, cfg, backend, ws);
  }
  cudaEvent_t begin{}, end{};
  VIDFAB_CUDA_CHECK(cudaEventCreate(&begin));
  VIDFAB_CUDA_CHECK(cudaEventCreate(&end));
  VIDFAB_CUDA_CHECK(cudaEventRecord(begin));
  for (int i = 0; i < iterations; ++i) {
    ws.clear();
    vidfab::cuda::attention_forward(blas, nullptr, q, k, v, out, cfg, backend, ws);
  }
  VIDFAB_CUDA_CHECK(cudaEventRecord(end));
  VIDFAB_CUDA_CHECK(cudaEventSynchronize(end));
  float ms = 0;
  VIDFAB_CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
  cudaEventDestroy(begin); cudaEventDestroy(end);
  return ms / iterations;
}
}  // namespace

int main(int argc, char** argv) {
  int seq = 8192, heads = 8, iterations = 5, prefix = 951;
  float beta = 1.0f;
  for (int i = 1; i < argc; ++i) {
    auto next = [&]() { if (++i >= argc) std::exit(2); return std::atoi(argv[i]); };
    if (!std::strcmp(argv[i], "--seq")) seq = next();
    else if (!std::strcmp(argv[i], "--heads")) heads = next();
    else if (!std::strcmp(argv[i], "--iters")) iterations = next();
    else if (!std::strcmp(argv[i], "--prefix")) prefix = next();
    else if (!std::strcmp(argv[i], "--beta")) {
      if (++i >= argc) return 2;
      beta = std::strtof(argv[i], nullptr);
    }
    else { std::fprintf(stderr, "unknown option: %s\n", argv[i]); return 2; }
  }
  if (seq <= 0 || heads <= 0 || iterations <= 0) return 2;
  prefix = std::min(prefix, seq);
  const size_t values = size_t(seq) * heads * 128;
  vidfab::cuda::DeviceBuffer<__nv_bfloat16> q(values), k(values), v(values), out(values);
  fill<<<std::min<size_t>(65535, (values + 255) / 256), 256>>>(q.get(), values, 1, 0.3f);
  fill<<<std::min<size_t>(65535, (values + 255) / 256), 256>>>(k.get(), values, 2, 0.3f);
  fill<<<std::min<size_t>(65535, (values + 255) / 256), 256>>>(v.get(), values, 3, 1.0f);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
  cublasHandle_t blas{}; cublasCreate(&blas);
  vidfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq; cfg.num_heads = heads; cfg.head_dim = 128; cfg.exact_prefix = prefix;
  cfg.sol_beta = beta;
  vidfab::cuda::DeviceBuffer<unsigned long long> routes(2);
  VIDFAB_CUDA_CHECK(cudaMemset(routes.get(), 0, routes.nbytes()));
  cfg.sol_route_counts = routes.get();
  const size_t sol_bytes = vidfab::cuda::attention_workspace_bytes(
      cfg, vidfab::cuda::AttentionBackend::kSol);
  vidfab::cuda::Workspace ws; ws.reserve(sol_bytes);
  const float sol = time_backend(blas, q.get(), k.get(), v.get(), out.get(), cfg,
                                 vidfab::cuda::AttentionBackend::kSol, ws, 2, iterations);
  unsigned long long route_host[2]{};
  VIDFAB_CUDA_CHECK(cudaMemcpy(route_host, routes.get(), sizeof(route_host), cudaMemcpyDeviceToHost));
  cfg.sol_route_counts = nullptr;
  const float dense = time_backend(blas, q.get(), k.get(), v.get(), out.get(), cfg,
                                   vidfab::cuda::AttentionBackend::kFused, ws, 2, iterations);
  std::printf("seq=%d heads=%d prefix=%d beta=%.3g workspace=%.2f MiB\n", seq, heads, prefix, beta,
              sol_bytes / 1048576.0);
  std::printf("sol %.3f ms  dense %.3f ms  speedup %.3fx\n", sol, dense, dense / sol);
  const double route_total = double(route_host[0] + route_host[1]);
  std::printf("routes exact %.1f%%  approximate %.1f%%\n",
              100.0 * route_host[0] / route_total, 100.0 * route_host[1] / route_total);
  cublasDestroy(blas);
  return 0;
}
