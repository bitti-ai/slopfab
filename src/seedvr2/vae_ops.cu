#include "vae_ops.cuh"
#include <algorithm>

namespace slopfab::seedvr2 {
namespace {
using BF = __nv_bfloat16;
constexpr int kNormChunk = 4096;

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

__global__ void gn_kernel(const BF* x, BF* y, const BF* weight, const BF* bias, int spatial, int c,
                          bool silu) {
  int frame = blockIdx.x / 32, group = blockIdx.x % 32, gc = c / 32;
  int count = spatial * gc;
  float s = 0;
  for (int i = threadIdx.x; i < count; i += blockDim.x)
    s += __bfloat162float(x[(size_t(frame) * spatial + i / gc) * c + group * gc + i % gc]);
  float mean = sum_block(s) / count;
  __syncthreads();
  s = 0;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    float a =
        __bfloat162float(x[(size_t(frame) * spatial + i / gc) * c + group * gc + i % gc]) - mean;
    s += a * a;
  }
  float inv = rsqrtf(sum_block(s) / count + 1e-6f);
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    int ch = group * gc + i % gc;
    size_t p = (size_t(frame) * spatial + i / gc) * c + ch;
    float a = __bfloat162float(
        __float2bfloat16_rn((__bfloat162float(x[p]) - mean) * inv * __bfloat162float(weight[ch]) +
                            __bfloat162float(bias[ch])));
    y[p] = __float2bfloat16_rn(silu ? a / (1 + expf(-a)) : a);
  }
}

struct Moments {
  float mean, m2;
  int count;
};

__device__ Moments merge(Moments a, Moments b) {
  if (!a.count)
    return b;
  if (!b.count)
    return a;
  float delta = b.mean - a.mean;
  int count = a.count + b.count;
  return {a.mean + delta * (float(b.count) / count),
          a.m2 + b.m2 + delta * delta * (float(a.count) * b.count / count), count};
}

__device__ Moments reduce(Moments v) {
  __shared__ Moments values[256];
  values[threadIdx.x] = v;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (threadIdx.x < d)
      values[threadIdx.x] = merge(values[threadIdx.x], values[threadIdx.x + d]);
    __syncthreads();
  }
  return values[0];
}

__global__ void norm_partials(const BF* x, Moments* partial, int spatial, int c, int chunks) {
  int group = blockIdx.x / chunks, chunk = blockIdx.x % chunks;
  int gc = c / 32, count = spatial * gc, end = min(count, (chunk + 1) * kNormChunk);
  Moments v{0, 0, 0};
  for (int i = chunk * kNormChunk + threadIdx.x; i < end; i += 256) {
    float a = __bfloat162float(
        x[(size_t(group / 32) * spatial + i / gc) * c + (group % 32) * gc + i % gc]);
    v.count++;
    float delta = a - v.mean;
    v.mean += delta / v.count;
    v.m2 += delta * (a - v.mean);
  }
  v = reduce(v);
  if (!threadIdx.x)
    partial[blockIdx.x] = v;
}

__global__ void norm_stats(const Moments* partial, float2* stats, int chunks) {
  Moments v{0, 0, 0};
  for (int i = threadIdx.x; i < chunks; i += 256)
    v = merge(v, partial[size_t(blockIdx.x) * chunks + i]);
  v = reduce(v);
  if (!threadIdx.x)
    stats[blockIdx.x] = make_float2(v.mean, rsqrtf(v.m2 / v.count + 1e-6f));
}

__global__ void norm_apply(const BF* x, BF* y, const BF* weight, const BF* bias,
                           const float2* stats, size_t n, int spatial, int c, bool silu) {
  size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
  if (i >= n)
    return;
  int ch = int(i % c), frame = int(i / (size_t(spatial) * c));
  float2 s = stats[frame * 32 + ch / (c / 32)];
  float a = __bfloat162float(
      __float2bfloat16_rn((__bfloat162float(x[i]) - s.x) * s.y * __bfloat162float(weight[ch]) +
                          __bfloat162float(bias[ch])));
  y[i] = __float2bfloat16_rn(silu ? a / (1 + expf(-a)) : a);
}

template <int KT = 0, int KH = 0, int KW = 0>
__global__ void columns(const BF* x, BF* col, int t, int h, int w, int c, int kt, int kh, int kw,
                        int ss, int ts, int oh, int ow, bool down, int first, int rows) {
  size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
  if constexpr (KT != 0) {
    kt = KT;
    kh = KH;
    kw = KW;
  }
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
               ? __float2bfloat16_rn(0)
               : x[((size_t(st) * h + sy) * w + sx) * c + ch];
}
} // namespace

void vae_groupnorm(const BF* x, BF* y, const BF* weight, const BF* bias, int frames, int spatial,
                   int channels, bool silu, cuda::Workspace& scratch) {
  if (spatial * (channels / 32) <= 4096) {
    gn_kernel<<<frames * 32, 256>>>(x, y, weight, bias, spatial, channels, silu);
    SLOPFAB_CUDA_CHECK(cudaGetLastError());
    return;
  }
  int chunks = (spatial * (channels / 32) + kNormChunk - 1) / kNormChunk;
  size_t count = size_t(frames) * 32 * chunks;
  scratch.reserve(count * sizeof(Moments) + size_t(frames) * 32 * sizeof(float2) + 512);
  scratch.clear();
  auto* partial = scratch.alloc_n<Moments>(count);
  auto* stats = scratch.alloc_n<float2>(size_t(frames) * 32);
  norm_partials<<<int(count), 256>>>(x, partial, spatial, channels, chunks);
  norm_stats<<<frames * 32, 256>>>(partial, stats, chunks);
  size_t n = size_t(frames) * spatial * channels;
  norm_apply<<<int((n + 255) / 256), 256>>>(x, y, weight, bias, stats, n, spatial, channels, silu);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void vae_conv(cublasHandle_t blas, const BF* x, const BF* weight, BF* y, int t, int h, int w,
              int ci, int co, int kt, int kh, int kw, bool down, int ts, cuda::Workspace& scratch) {
  int oh = down ? h / 2 : h, ow = down ? w / 2 : w;
  int rows = ((t - 1) / ts + 1) * oh * ow, k = ci * kt * kh * kw;
  const bool direct = kt == 1 && kh == 1 && kw == 1 && !down && ts == 1;
  // Pointwise convolution reads NHWC directly. General convolution bounds its
  // reusable arena at 32 MiB; align ordinary tiles to avoid ragged GEMMs.
  // Larger RGB tiles amortize launches. Larger hidden-channel tiles selected
  // a less accurate cuBLAS reduction in testing, so retain their 2048 ceiling.
  int budget_rows = std::max(1, (32 << 20) / (2 * k));
  if (budget_rows >= 256)
    budget_rows = budget_rows / 256 * 256;
  int tile = direct ? rows : std::min(rows, std::min(ci <= 3 ? 8192 : 2048, budget_rows));
  BF* col = nullptr;
  if (!direct) {
    scratch.reserve(size_t(tile) * k * sizeof(BF));
    scratch.clear();
    col = scratch.alloc_n<BF>(size_t(tile) * k);
  }
  float a = 1, b = 0;
  for (int start = 0; start < rows; start += tile) {
    int count = std::min(tile, rows - start);
    if (!direct) {
      if (kt == 3 && kh == 3 && kw == 3)
        columns<3, 3, 3><<<int((size_t(count) * k + 255) / 256), 256>>>(
            x, col, t, h, w, ci, kt, kh, kw, down ? 2 : 1, ts, oh, ow, down, start, count);
      else
        columns<><<<int((size_t(count) * k + 255) / 256), 256>>>(
            x, col, t, h, w, ci, kt, kh, kw, down ? 2 : 1, ts, oh, ow, down, start, count);
      SLOPFAB_CUDA_CHECK(cudaGetLastError());
    }
    SLOPFAB_CUBLAS_CHECK(cuda::cublas_gemm_ex(blas, CUBLAS_OP_T, CUBLAS_OP_N, co, count, k, &a,
                                              weight, CUDA_R_16BF, k, direct ? x : col, CUDA_R_16BF,
                                              k, &b, y + size_t(start) * co, CUDA_R_16BF, co,
                                              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }
}
} // namespace slopfab::seedvr2
