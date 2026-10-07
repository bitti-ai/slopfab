#include "../src/seedvr2/vae_ops.cuh"
#include "slopfab/cuda/attention.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace slopfab;
using BFloat = __nv_bfloat16;

namespace {
__device__ float val(BFloat x) {
  return __bfloat162float(x);
}

__device__ BFloat bf(float x) {
  return __float2bfloat16_rn(x);
}

__device__ float rounded(float x) {
  return val(bf(x));
}

__device__ float sum_block(float v) {
  __shared__ float sums[256];
  sums[threadIdx.x] = v;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d /= 2) {
    if (threadIdx.x < d)
      sums[threadIdx.x] += sums[threadIdx.x + d];
    __syncthreads();
  }
  return sums[0];
}

__global__ void gn_kernel(const BFloat* x, BFloat* y, const BFloat* weight, const BFloat* bias,
                          int spatial, int c, bool silu) {
  int frame = blockIdx.x / 32, group = blockIdx.x % 32, gc = c / 32;
  int count = spatial * gc;
  float s = 0;
  for (int i = threadIdx.x; i < count; i += blockDim.x)
    s += val(x[(size_t(frame) * spatial + i / gc) * c + group * gc + i % gc]);
  float mean = sum_block(s) / count;
  __syncthreads();
  s = 0;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    float a = val(x[(size_t(frame) * spatial + i / gc) * c + group * gc + i % gc]) - mean;
    s += a * a;
  }
  float inv = rsqrtf(sum_block(s) / count + 1e-6f);
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    int ch = group * gc + i % gc;
    size_t p = (size_t(frame) * spatial + i / gc) * c + ch;
    float a = rounded((val(x[p]) - mean) * inv * val(weight[ch]) + val(bias[ch]));
    y[p] = bf(silu ? a / (1 + expf(-a)) : a);
  }
}

__global__ void col_kernel(const BFloat* x, BFloat* col, int t, int h, int w, int c, int kt, int kh,
                           int kw, int ss, int ts, int oh, int ow, bool down, int first, int rows) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int k = c * kt * kh * kw;
  if (i >= size_t(rows) * k)
    return;
  int r = int(i % k), p = int(i / k) + first;
  int kx = r % kw;
  r /= kw;
  int ky = r % kh;
  r /= kh;
  int kz = r % kt, ch = r / kt;
  int sx = (p % ow) * ss + kx - (down ? 0 : kw / 2);
  int sy = (p / ow % oh) * ss + ky - (down ? 0 : kh / 2);
  int st = max(0, p / (oh * ow) * ts + kz - (kt - 1));
  col[i] = sx < 0 || sx >= w || sy < 0 || sy >= h || st >= t
               ? bf(0)
               : x[((size_t(st) * h + sy) * w + sx) * c + ch];
}

__global__ void fill(BFloat* p, size_t n, float scale, float offset = 0) {
  size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
  if (i < n) {
    unsigned v = unsigned(i) * 747796405u + 2891336453u;
    v = ((v >> ((v >> 28) + 4)) ^ v) * 277803737u;
    v = (v >> 22) ^ v;
    p[i] = bf(offset + scale * (float(v & 65535) / 32768 - 1));
  }
}

struct Comparison {
  float maximum = 0;
  double squared = 0;
  size_t different = 0;
};

Comparison compare(const cuda::DeviceBuffer<BFloat>& a, const cuda::DeviceBuffer<BFloat>& b) {
  std::vector<BFloat> ah(a.size()), bh(b.size());
  a.copy_to_host(ah.data(), ah.size());
  b.copy_to_host(bh.data(), bh.size());
  Comparison result;
  for (size_t i = 0; i < ah.size(); ++i) {
    float d = std::abs(__bfloat162float(ah[i]) - __bfloat162float(bh[i]));
    if (!std::isfinite(d))
      throw std::runtime_error("non-finite output");
    result.maximum = std::max(result.maximum, d);
    result.squared += double(d) * d;
    result.different += d != 0;
  }
  result.squared = std::sqrt(result.squared / ah.size());
  return result;
}

template <class F> float time_ms(F fn, int iterations = 8) {
  fn();
  fn();
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t start, end;
  SLOPFAB_CUDA_CHECK(cudaEventCreate(&start));
  SLOPFAB_CUDA_CHECK(cudaEventCreate(&end));
  SLOPFAB_CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iterations; ++i)
    fn();
  SLOPFAB_CUDA_CHECK(cudaEventRecord(end));
  SLOPFAB_CUDA_CHECK(cudaEventSynchronize(end));
  float ms;
  SLOPFAB_CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  return ms / iterations;
}

void norm_case(int t, int h, int w, int c, float offset = 0, float scale = 1) {
  size_t n = size_t(t) * h * w * c;
  cuda::DeviceBuffer<BFloat> x(n), old(n), out(n), weight(c), bias(c);
  fill<<<int((n + 255) / 256), 256>>>(x.get(), n, scale, offset);
  fill<<<int((c + 255) / 256), 256>>>(weight.get(), c, .1f, 1);
  fill<<<int((c + 255) / 256), 256>>>(bias.get(), c, .1f);
  cuda::Workspace scratch;
  auto ref = [&] {
    gn_kernel<<<t * 32, 256>>>(x.get(), old.get(), weight.get(), bias.get(), h * w, c, true);
  };
  auto opt = [&] {
    seedvr2::vae_groupnorm(x.get(), out.get(), weight.get(), bias.get(), t, h * w, c, true,
                           scratch);
  };
  float before = time_ms(ref), after = time_ms(opt);
  Comparison diff = compare(old, out);
  std::printf(
      "norm %dx%dx%dx%d offset=%g scale=%g old_ms=%.4f new_ms=%.4f speedup=%.2f max=%g rmse=%g different=%.5f scratch=%zu\n",
      t, h, w, c, offset, scale, before, after, before / after, diff.maximum, diff.squared,
      double(diff.different) / n, scratch.capacity());
  if (diff.maximum > .032f || diff.squared > .0001)
    throw std::runtime_error("groupnorm parity failed");
}

void conv_case(cublasHandle_t blas, int t, int h, int w, int ci, int co, int kt, int kh, int kw,
               bool down = false, int ts = 1) {
  int oh = down ? h / 2 : h, ow = down ? w / 2 : w;
  int rows = ((t - 1) / ts + 1) * oh * ow, k = ci * kt * kh * kw;
  size_t n = size_t(t) * h * w * ci;
  cuda::DeviceBuffer<BFloat> x(n), old(size_t(rows) * co), out(size_t(rows) * co),
      weight(size_t(co) * k);
  cuda::DeviceBuffer<BFloat> col(size_t(std::min(2048, rows)) * k);
  fill<<<int((n + 255) / 256), 256>>>(x.get(), n, .25f);
  fill<<<int((weight.size() + 255) / 256), 256>>>(weight.get(), weight.size(), .02f);
  cuda::Workspace scratch;
  auto ref = [&] {
    float a = 1, b = 0;
    for (int start = 0; start < rows; start += 2048) {
      int count = std::min(2048, rows - start);
      col_kernel<<<int((size_t(count) * k + 255) / 256), 256>>>(x.get(), col.get(), t, h, w, ci, kt,
                                                                kh, kw, down ? 2 : 1, ts, oh, ow,
                                                                down, start, count);
      SLOPFAB_CUBLAS_CHECK(cuda::cublas_gemm_ex(
          blas, CUBLAS_OP_T, CUBLAS_OP_N, co, count, k, &a, weight.get(), CUDA_R_16BF, k, col.get(),
          CUDA_R_16BF, k, &b, old.get() + size_t(start) * co, CUDA_R_16BF, co, CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT));
    }
  };
  auto opt = [&] {
    seedvr2::vae_conv(blas, x.get(), weight.get(), out.get(), t, h, w, ci, co, kt, kh, kw, down, ts,
                      scratch);
  };
  float before = time_ms(ref), after = time_ms(opt);
  Comparison diff = compare(old, out);
  std::printf(
      "conv %dx%dx%dx%d co=%d k=%dx%dx%d down=%d ts=%d old_ms=%.4f new_ms=%.4f speedup=%.2f max=%g rmse=%g different=%.5f old_scratch=%zu new_scratch=%zu\n",
      t, h, w, ci, co, kt, kh, kw, down, ts, before, after, before / after, diff.maximum,
      diff.squared, double(diff.different) / out.size(), col.nbytes(), scratch.capacity());
  if (diff.maximum > .002f || diff.squared > .0001)
    throw std::runtime_error("convolution parity failed");
}

void attention_case(cublasHandle_t blas, int seq) {
  size_t n = size_t(seq) * 512;
  cuda::DeviceBuffer<BFloat> q(n), k(n), v(n), old(n), out(n);
  fill<<<int((n + 255) / 256), 256>>>(q.get(), n, .5f);
  fill<<<int((n + 255) / 256), 256>>>(k.get(), n, .4f);
  fill<<<int((n + 255) / 256), 256>>>(v.get(), n, .3f);
  cuda::AttentionConfig cfg;
  cfg.seq_len = seq;
  cfg.num_heads = 1;
  cfg.head_dim = 512;
  cuda::Workspace scratch;
  auto run = [&](BFloat* output) {
    scratch.reserve(cuda::attention_workspace_bytes(cfg, cuda::AttentionBackend::kBlocked));
    scratch.clear();
    cuda::attention_forward(blas, nullptr, q.get(), k.get(), v.get(), output, cfg,
                            cuda::AttentionBackend::kBlocked, scratch);
  };
  float baseline = time_ms([&] {
    run(old.get());
  });
  for (int block : {256, 512, 2048}) {
    cfg.query_block = block;
    float elapsed = time_ms([&] {
      run(out.get());
    });
    Comparison diff = compare(old, out);
    std::printf(
        "attention seq=%d dim=512 query_block=%d old_ms=%.4f new_ms=%.4f speedup=%.2f max=%g rmse=%g scratch=%zu\n",
        seq, block, baseline, elapsed, baseline / elapsed, diff.maximum, diff.squared,
        scratch.capacity());
    if (diff.maximum > .001f)
      throw std::runtime_error("attention parity failed");
  }
}
} // namespace

int main() {
  try {
    cublasHandle_t blas;
    SLOPFAB_CUBLAS_CHECK(cuda::cublas_create(&blas));
    norm_case(5, 256, 256, 128);
    norm_case(5, 128, 128, 256);
    norm_case(2, 32, 32, 512);
    norm_case(1, 360, 640, 128);
    norm_case(2, 32, 32, 128, 128, .75f);
    norm_case(2, 32, 32, 128, 128, 0);
    conv_case(blas, 5, 256, 256, 128, 256, 1, 1, 1);
    conv_case(blas, 2, 32, 32, 256, 512, 1, 1, 1);
    conv_case(blas, 5, 256, 256, 3, 128, 3, 3, 3);
    conv_case(blas, 5, 128, 128, 128, 128, 3, 3, 3);
    conv_case(blas, 2, 32, 32, 512, 512, 3, 3, 3);
    conv_case(blas, 5, 128, 128, 128, 256, 3, 3, 3, true, 2);
    conv_case(blas, 1, 360, 640, 3, 128, 3, 3, 3);
    attention_case(blas, 1024);
    attention_case(blas, 4096);
    attention_case(blas, 14400);
    SLOPFAB_CUBLAS_CHECK(cuda::cublas_destroy(blas));
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
  return 0;
}
