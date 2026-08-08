// Standalone Sol-Attn microbenchmark. Defaults are representative and safe;
// pass --seq 37710 --heads 56 for the production H3 geometry.
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "vidfab/cuda/attention.cuh"
#include "vidfab/cuda/device.h"
#include "vidfab/cuda/workspace.cuh"
#include "vidfab/sol_capture.h"

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
  bool pipeline = false;
  std::string input;
  std::string save_input;
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
    else if (!std::strcmp(argv[i], "--pipeline")) pipeline = true;
    else if (!std::strcmp(argv[i], "--input")) {
      if (++i >= argc) return 2;
      input = argv[i];
    }
    else if (!std::strcmp(argv[i], "--save-input")) {
      if (++i >= argc) return 2;
      save_input = argv[i];
    }
    else { std::fprintf(stderr, "unknown option: %s\n", argv[i]); return 2; }
  }
  std::vector<uint16_t> captured;
  vidfab::SolCaptureHeader capture{};
  if (!input.empty()) {
    std::ifstream in(input, std::ios::binary);
    if (!in.read(reinterpret_cast<char*>(&capture), sizeof(capture)) ||
        std::memcmp(capture.magic, "VFSOLQKV", 8) != 0 || capture.version != 1 ||
        capture.header_bytes != sizeof(capture) || capture.head_dim != 128) {
      std::fprintf(stderr, "invalid or unsupported Sol capture: %s\n", input.c_str());
      return 2;
    }
    const uint64_t expected = uint64_t(capture.seq_len) * capture.num_heads * capture.head_dim;
    if (capture.tensor_elements != expected || expected > SIZE_MAX / 6) return 2;
    captured.resize(static_cast<size_t>(expected) * 3);
    if (!in.read(reinterpret_cast<char*>(captured.data()),
                 static_cast<std::streamsize>(captured.size() * sizeof(uint16_t))) ||
        in.peek() != std::ifstream::traits_type::eof()) {
      std::fprintf(stderr, "truncated or oversized Sol capture: %s\n", input.c_str());
      return 2;
    }
    seq = static_cast<int>(capture.seq_len);
    heads = static_cast<int>(capture.num_heads);
    prefix = static_cast<int>(capture.exact_prefix);
  }
  if (seq <= 0 || heads <= 0 || iterations <= 0) return 2;
  prefix = std::min(prefix, seq);
  const size_t values = size_t(seq) * heads * 128;
  vidfab::cuda::DeviceBuffer<__nv_bfloat16> q(values), k(values), v(values), out(values);
  if (captured.empty()) {
    fill<<<std::min<size_t>(65535, (values + 255) / 256), 256>>>(q.get(), values, 1, 0.3f);
    fill<<<std::min<size_t>(65535, (values + 255) / 256), 256>>>(k.get(), values, 2, 0.3f);
    fill<<<std::min<size_t>(65535, (values + 255) / 256), 256>>>(v.get(), values, 3, 1.0f);
  } else {
    VIDFAB_CUDA_CHECK(cudaMemcpy(q.get(), captured.data(), values * sizeof(uint16_t), cudaMemcpyHostToDevice));
    VIDFAB_CUDA_CHECK(cudaMemcpy(k.get(), captured.data() + values, values * sizeof(uint16_t), cudaMemcpyHostToDevice));
    VIDFAB_CUDA_CHECK(cudaMemcpy(v.get(), captured.data() + 2 * values, values * sizeof(uint16_t), cudaMemcpyHostToDevice));
  }
  VIDFAB_CUDA_CHECK(cudaGetLastError());
  if (!save_input.empty()) {
    if (!input.empty()) {
      std::fprintf(stderr, "--save-input cannot be combined with --input\n");
      return 2;
    }
    captured.resize(values * 3);
    VIDFAB_CUDA_CHECK(cudaMemcpy(captured.data(), q.get(), values * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    VIDFAB_CUDA_CHECK(cudaMemcpy(captured.data() + values, k.get(), values * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    VIDFAB_CUDA_CHECK(cudaMemcpy(captured.data() + 2 * values, v.get(), values * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    vidfab::SolCaptureHeader h{{'V','F','S','O','L','Q','K','V'}, 1,
        sizeof(vidfab::SolCaptureHeader), static_cast<uint32_t>(seq),
        static_cast<uint32_t>(heads), 128, static_cast<uint32_t>(prefix), -1, -1, values, {0, 0}};
    std::ofstream out_file(save_input, std::ios::binary | std::ios::trunc);
    out_file.write(reinterpret_cast<const char*>(&h), sizeof(h));
    out_file.write(reinterpret_cast<const char*>(captured.data()),
                   static_cast<std::streamsize>(captured.size() * sizeof(uint16_t)));
    if (!out_file) {
      std::fprintf(stderr, "failed writing synthetic Sol capture: %s\n", save_input.c_str());
      return 2;
    }
    captured.clear();
  }
  cublasHandle_t blas{}; cublasCreate(&blas);
  vidfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq; cfg.num_heads = heads; cfg.head_dim = 128; cfg.exact_prefix = prefix;
  cfg.sol_beta = beta;
  cfg.sol_pipeline = pipeline;
  vidfab::cuda::DeviceBuffer<unsigned long long> routes(2);
  const size_t sol_bytes = vidfab::cuda::attention_workspace_bytes(
      cfg, vidfab::cuda::AttentionBackend::kSol);
  vidfab::cuda::Workspace ws; ws.reserve(sol_bytes);
  const float sol = time_backend(blas, q.get(), k.get(), v.get(), out.get(), cfg,
                                 vidfab::cuda::AttentionBackend::kSol, ws, 2, iterations);
  // Diagnostics are deliberately outside the timed path.
  VIDFAB_CUDA_CHECK(cudaMemset(routes.get(), 0, routes.nbytes()));
  cfg.sol_route_counts = routes.get();
  ws.clear();
  vidfab::cuda::attention_forward(blas, nullptr, q.get(), k.get(), v.get(), out.get(), cfg,
                                  vidfab::cuda::AttentionBackend::kSol, ws);
  unsigned long long route_host[2]{};
  VIDFAB_CUDA_CHECK(cudaMemcpy(route_host, routes.get(), sizeof(route_host), cudaMemcpyDeviceToHost));
  cfg.sol_route_counts = nullptr;
  float phases[4]{};
  cfg.sol_phase_ms = phases;
  ws.clear();
  vidfab::cuda::attention_forward(blas, nullptr, q.get(), k.get(), v.get(), out.get(), cfg,
                                  vidfab::cuda::AttentionBackend::kSol, ws);
  cfg.sol_phase_ms = nullptr;
  const float dense = time_backend(blas, q.get(), k.get(), v.get(), out.get(), cfg,
                                   vidfab::cuda::AttentionBackend::kFused, ws, 2, iterations);
  std::printf("seq=%d heads=%d prefix=%d beta=%.3g pipeline=%d workspace=%.2f MiB\n", seq, heads, prefix, beta, int(pipeline),
              sol_bytes / 1048576.0);
  if (!input.empty()) std::printf("input=%s step=%d layer=%d\n", input.c_str(), capture.denoise_step, capture.layer);
  std::printf("sol %.3f ms  dense %.3f ms  speedup %.3fx\n", sol, dense, dense / sol);
  const double route_total = double(route_host[0] + route_host[1]);
  std::printf("routes exact %.1f%%  approximate %.1f%%\n",
              100.0 * route_host[0] / route_total, 100.0 * route_host[1] / route_total);
  std::printf("phases pool %.3f  stats %.3f  threshold %.3f  main %.3f ms\n",
              phases[0], phases[1], phases[2], phases[3]);
  cublasDestroy(blas);
  return 0;
}
