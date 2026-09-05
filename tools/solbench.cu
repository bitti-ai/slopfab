// Standalone Sol-Attn microbenchmark. Defaults are representative and safe;
// pass --seq 37710 --heads 56 for the production H3 geometry.
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include "slopfab/cuda/attention.cuh"
#include "slopfab/cuda/device.h"
#include "slopfab/cuda/gemm.cuh"
#include "slopfab/cuda/workspace.cuh"
#include "slopfab/sol_capture.h"

namespace {
__global__ void fill(__nv_bfloat16* p, size_t n, unsigned seed, float amplitude) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x) {
    unsigned x = unsigned(i) ^ seed;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    p[i] = __float2bfloat16_rn((float(x & 65535u) / 32767.5f - 1.0f) * amplitude);
  }
}

__global__ void scan_nonfinite(const __nv_bfloat16* p, size_t n,
                               unsigned long long* result) {
  for (size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;
       i+=size_t(gridDim.x)*blockDim.x) {
    if (!isfinite(__bfloat162float(p[i]))) {
      atomicAdd(result,1ull);
      atomicMin(result+1,static_cast<unsigned long long>(i));
    }
    const float a=fabsf(__bfloat162float(p[i]));
    if(isfinite(a)) atomicMax(reinterpret_cast<unsigned int*>(result+2),__float_as_uint(a));
  }
}

__global__ void compare_outputs(const __nv_bfloat16* sol,
                                const __nv_bfloat16* dense, size_t n,
                                double* sums, unsigned long long* bad,
                                unsigned* max_diff) {
  __shared__ double sd[256], ss[256], sr[256], sp[256];
  __shared__ unsigned long long bs[256], br[256];
  __shared__ float md[256];
  double d2=0,s2=0,r2=0,dot=0; unsigned long long sb=0,rb=0; float mx=0;
  for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;
      i+=size_t(gridDim.x)*blockDim.x) {
    const float s=__bfloat162float(sol[i]), r=__bfloat162float(dense[i]);
    const bool sf=isfinite(s),rf=isfinite(r); sb+=!sf;rb+=!rf;
    if(sf&&rf) { const double e=double(s)-r;d2+=e*e;s2+=double(s)*s;
      r2+=double(r)*r;dot+=double(s)*r;mx=fmaxf(mx,fabsf(s-r)); }
  }
  const int t=threadIdx.x;sd[t]=d2;ss[t]=s2;sr[t]=r2;sp[t]=dot;
  bs[t]=sb;br[t]=rb;md[t]=mx;__syncthreads();
  for(int k=128;k;k>>=1){if(t<k){sd[t]+=sd[t+k];ss[t]+=ss[t+k];
    sr[t]+=sr[t+k];sp[t]+=sp[t+k];bs[t]+=bs[t+k];br[t]+=br[t+k];
    md[t]=fmaxf(md[t],md[t+k]);}__syncthreads();}
  if(t==0){atomicAdd(sums,sd[0]);atomicAdd(sums+1,ss[0]);
    atomicAdd(sums+2,sr[0]);atomicAdd(sums+3,sp[0]);
    atomicAdd(bad,bs[0]);atomicAdd(bad+1,br[0]);
    atomicMax(max_diff,__float_as_uint(md[0]));}
}

float time_backend(cublasHandle_t blas, const __nv_bfloat16* q, const __nv_bfloat16* k,
                   const __nv_bfloat16* v, __nv_bfloat16* out,
                   const slopfab::cuda::AttentionConfig& cfg,
                   slopfab::cuda::AttentionBackend backend, slopfab::cuda::Workspace& ws,
                   int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) {
    ws.clear();
    slopfab::cuda::attention_forward(blas, nullptr, q, k, v, out, cfg, backend, ws);
  }
  cudaEvent_t begin{}, end{};
  SLOPFAB_CUDA_CHECK(cudaEventCreate(&begin));
  SLOPFAB_CUDA_CHECK(cudaEventCreate(&end));
  SLOPFAB_CUDA_CHECK(cudaEventRecord(begin));
  for (int i = 0; i < iterations; ++i) {
    ws.clear();
    slopfab::cuda::attention_forward(blas, nullptr, q, k, v, out, cfg, backend, ws);
  }
  SLOPFAB_CUDA_CHECK(cudaEventRecord(end));
  SLOPFAB_CUDA_CHECK(cudaEventSynchronize(end));
  float ms = 0;
  SLOPFAB_CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
  cudaEventDestroy(begin); cudaEventDestroy(end);
  return ms / iterations;
}
}  // namespace

int main(int argc, char** argv) {
  int seq = 8192, heads = 8, iterations = 5, prefix = 951;
  float beta = 1.0f;
  float error_k=0.0f,error_v=0.0f;
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
    else if (!std::strcmp(argv[i],"--error-k")){if(++i>=argc)return 2;error_k=std::strtof(argv[i],nullptr);}
    else if (!std::strcmp(argv[i],"--error-v")){if(++i>=argc)return 2;error_v=std::strtof(argv[i],nullptr);}
    else if (!std::strcmp(argv[i], "--experimental-pipeline")) pipeline = true;
    else if (!std::strcmp(argv[i], "--pipeline")) {
      std::fprintf(stderr,"--pipeline was renamed --experimental-pipeline\n");
      return 2;
    }
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
  slopfab::SolCaptureHeader capture{};
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
  int smem_per_sm = 0, regs_per_sm = 0;
  cudaDeviceGetAttribute(&smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
  cudaDeviceGetAttribute(&regs_per_sm, cudaDevAttrMaxRegistersPerMultiprocessor, 0);
  prefix = std::min(prefix, seq);
  const size_t values = size_t(seq) * heads * 128;
  slopfab::cuda::DeviceBuffer<__nv_bfloat16> q(values), k(values), v(values), out(values), dense_out(values);
  if (captured.empty()) {
    fill<<<std::min<size_t>(65535, (values + 255) / 256), 256>>>(q.get(), values, 1, 0.3f);
    fill<<<std::min<size_t>(65535, (values + 255) / 256), 256>>>(k.get(), values, 2, 0.3f);
    fill<<<std::min<size_t>(65535, (values + 255) / 256), 256>>>(v.get(), values, 3, 1.0f);
  } else {
    SLOPFAB_CUDA_CHECK(cudaMemcpy(q.get(), captured.data(), values * sizeof(uint16_t), cudaMemcpyHostToDevice));
    SLOPFAB_CUDA_CHECK(cudaMemcpy(k.get(), captured.data() + values, values * sizeof(uint16_t), cudaMemcpyHostToDevice));
    SLOPFAB_CUDA_CHECK(cudaMemcpy(v.get(), captured.data() + 2 * values, values * sizeof(uint16_t), cudaMemcpyHostToDevice));
  }
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
  if (!save_input.empty()) {
    if (!input.empty()) {
      std::fprintf(stderr, "--save-input cannot be combined with --input\n");
      return 2;
    }
    captured.resize(values * 3);
    SLOPFAB_CUDA_CHECK(cudaMemcpy(captured.data(), q.get(), values * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    SLOPFAB_CUDA_CHECK(cudaMemcpy(captured.data() + values, k.get(), values * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    SLOPFAB_CUDA_CHECK(cudaMemcpy(captured.data() + 2 * values, v.get(), values * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    slopfab::SolCaptureHeader h{{'V','F','S','O','L','Q','K','V'}, 1,
        sizeof(slopfab::SolCaptureHeader), static_cast<uint32_t>(seq),
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
  cublasHandle_t blas{}; slopfab::cuda::cublas_create(&blas);
  slopfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq; cfg.num_heads = heads; cfg.head_dim = 128; cfg.exact_prefix = prefix;
  cfg.sol_beta = beta;
  cfg.sol_error_k=error_k;cfg.sol_error_v=error_v;
  cfg.sol_pipeline = pipeline;
  slopfab::cuda::DeviceBuffer<unsigned long long> routes(2);
  const size_t sol_bytes = slopfab::cuda::attention_workspace_bytes(
      cfg, slopfab::cuda::AttentionBackend::kSol);
  slopfab::cuda::Workspace ws; ws.reserve(sol_bytes);
  const float sol = time_backend(blas, q.get(), k.get(), v.get(), out.get(), cfg,
                                 slopfab::cuda::AttentionBackend::kSol, ws, 2, iterations);
  // Diagnostics are deliberately outside the timed path.
  SLOPFAB_CUDA_CHECK(cudaMemset(routes.get(), 0, routes.nbytes()));
  cfg.sol_route_counts = routes.get();
  ws.clear();
  slopfab::cuda::attention_forward(blas, nullptr, q.get(), k.get(), v.get(), out.get(), cfg,
                                  slopfab::cuda::AttentionBackend::kSol, ws);
  unsigned long long route_host[2]{};
  SLOPFAB_CUDA_CHECK(cudaMemcpy(route_host, routes.get(), sizeof(route_host), cudaMemcpyDeviceToHost));
  cfg.sol_route_counts = nullptr;
  float phases[4]{};
  cfg.sol_phase_ms = phases;
  ws.clear();
  slopfab::cuda::attention_forward(blas, nullptr, q.get(), k.get(), v.get(), out.get(), cfg,
                                  slopfab::cuda::AttentionBackend::kSol, ws);
  cfg.sol_phase_ms = nullptr;
  slopfab::cuda::DeviceBuffer<unsigned long long> finite_diag(3);
  const unsigned long long finite_init[3]={0,~0ull,0};
  SLOPFAB_CUDA_CHECK(cudaMemcpy(finite_diag.get(),finite_init,sizeof(finite_init),
                               cudaMemcpyHostToDevice));
  scan_nonfinite<<<256,256>>>(out.get(),values,finite_diag.get());
  unsigned long long finite_host[3]{};
  SLOPFAB_CUDA_CHECK(cudaMemcpy(finite_host,finite_diag.get(),sizeof(finite_host),
                               cudaMemcpyDeviceToHost));
  const float dense = time_backend(blas, q.get(), k.get(), v.get(), dense_out.get(), cfg,
                                   slopfab::cuda::AttentionBackend::kFused, ws, 2, iterations);
  slopfab::cuda::DeviceBuffer<double> compare_sums(4);
  slopfab::cuda::DeviceBuffer<unsigned long long> compare_bad(2);
  slopfab::cuda::DeviceBuffer<unsigned> compare_max(1);
  SLOPFAB_CUDA_CHECK(cudaMemset(compare_sums.get(),0,compare_sums.nbytes()));
  SLOPFAB_CUDA_CHECK(cudaMemset(compare_bad.get(),0,compare_bad.nbytes()));
  SLOPFAB_CUDA_CHECK(cudaMemset(compare_max.get(),0,compare_max.nbytes()));
  compare_outputs<<<256,256>>>(out.get(),dense_out.get(),values,compare_sums.get(),
                               compare_bad.get(),compare_max.get());
  double cmp[4]{};unsigned long long cmp_bad[2]{};unsigned cmp_max_bits=0;
  SLOPFAB_CUDA_CHECK(cudaMemcpy(cmp,compare_sums.get(),sizeof(cmp),cudaMemcpyDeviceToHost));
  SLOPFAB_CUDA_CHECK(cudaMemcpy(cmp_bad,compare_bad.get(),sizeof(cmp_bad),cudaMemcpyDeviceToHost));
  SLOPFAB_CUDA_CHECK(cudaMemcpy(&cmp_max_bits,compare_max.get(),sizeof(cmp_max_bits),cudaMemcpyDeviceToHost));
  float cmp_max=0;std::memcpy(&cmp_max,&cmp_max_bits,sizeof(cmp_max));
  std::printf("seq=%d heads=%d prefix=%d beta=%.3g pipeline=%d workspace=%.2f MiB\n", seq, heads, prefix, beta, int(pipeline),
              sol_bytes / 1048576.0);
  std::printf("device shared/SM=%.1f KiB registers/SM=%d\n",
              smem_per_sm / 1024.0, regs_per_sm);
  if (!input.empty()) std::printf("input=%s step=%d layer=%d\n", input.c_str(), capture.denoise_step, capture.layer);
  std::printf("sol %.3f ms  dense %.3f ms  speedup %.3fx\n", sol, dense, dense / sol);
  std::printf("compare rel_l2=%.7g cosine=%.9g max_abs_diff=%.7g finite sol=%zu dense=%zu\n",
              std::sqrt(cmp[0]/cmp[2]),cmp[3]/std::sqrt(cmp[1]*cmp[2]),cmp_max,
              values-size_t(cmp_bad[0]),values-size_t(cmp_bad[1]));
  const double route_total = double(route_host[0] + route_host[1]);
  std::printf("routes exact %.1f%%  approximate %.1f%%\n",
              100.0 * route_host[0] / route_total, 100.0 * route_host[1] / route_total);
  std::printf("phases pool %.3f  stats %.3f  threshold %.3f  main %.3f ms\n",
              phases[0], phases[1], phases[2], phases[3]);
  if (finite_host[0]) {
    const size_t row=finite_host[1]/size_t(heads*128);
    const size_t rem=finite_host[1]%size_t(heads*128);
    std::printf("NONFINITE count=%llu first row=%zu head=%zu dim=%zu\n",
                finite_host[0],row,rem/128,rem%128);
  } else {
    const unsigned max_bits=static_cast<unsigned>(finite_host[2]);
    float max_abs=0; std::memcpy(&max_abs,&max_bits,sizeof(max_abs));
    std::printf("output finite max_abs=%.7g\n",max_abs);
  }
  slopfab::cuda::cublas_destroy(blas);
  return 0;
}
