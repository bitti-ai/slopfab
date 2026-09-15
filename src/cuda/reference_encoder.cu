#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "slopfab/cuda/reference_encoder.cuh"

namespace slopfab::cuda {
namespace {
void blas_check(cublasStatus_t result) {
  if (result != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("reference encoder: cuBLAS failure");
}
__device__ int reflected(int i, int n) {
  if (n <= 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
  return i;
}
__global__ void convert(const __half* x, float* y, size_t n) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __half2float(x[i]);
}
template <typename T>
__global__ void col3(const T* x, T* col, int cin, int t, int h, int w,
                     int k, int ss, int ts, bool down, int oh, int ow,
                     int start, int columns) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(cin) * k * k * k * columns) return;
  int p = int(i % columns) + start, r = int(i / columns);
  int kx = r % k;
  r /= k;
  int ky = r % k;
  r /= k;
  int kt = r % k;
  int c = r / k;
  int sx = (p % ow) * ss + kx - (down ? 0 : k / 2);
  int sy = ((p / ow) % oh) * ss + ky - (down ? 0 : k / 2);
  int st = (p / (oh * ow)) * ts + kt - (k - 1);
  // Both the symmetric spatial pad and the downsampler's right/bottom pad
  // reflect. Temporal padding is causal zero, never replication.
  sx = reflected(sx, w);
  sy = reflected(sy, h);
  col[i] = st < 0 || st >= t ? T(0.0f) : x[((size_t(c) * t + st) * h + sy) * w + sx];
}
__global__ void col1(const float* x, float* col, int cin, int length, int k,
                     int stride, int pad, int dil, int start, int columns) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(cin) * k * columns) return;
  int r = int(i / columns), p = int(i % columns) + start,
      s = p * stride + (r % k) * dil - pad;
  col[i] = s < 0 || s >= length ? 0 : x[size_t(r / k) * length + s];
}
__global__ void bias_channels(float* x, const float* b, int n, int c) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size_t(n) * c) x[i] += b[i / n];
}
__global__ void half_bias(__half* x, const __half* b, int n, size_t count) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) x[i] = __float2half_rn(__half2float(x[i]) + __half2float(b[i / n]));
}
__global__ void half_add(__half* x, const __half* branch, size_t count) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) x[i] = __float2half_rn(__half2float(x[i]) + __half2float(branch[i]));
}
__global__ void bias_rows(float* x, const float* b, int n, int c) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size_t(n) * c) x[i] += b[i % c];
}
__global__ void transpose_kernel(const float* x, float* y, int batch, int rows,
                                 int cols) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= size_t(batch) * rows * cols) return;
  int c = int(i % cols), r = int(i / cols) % rows,
      b = int(i / (size_t(rows) * cols));
  y[(size_t(b) * cols + c) * rows + r] = x[i];
}
__global__ void element(float* x, const float* p, size_t n, int channels,
                        int length, int op) {
  size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (op == 0)
    x[i] += p[i];
  else if (op == 1) {
    float a = p[(i / length) % channels], s = sinf(a * x[i]);
    x[i] += s * s / (a + 1e-9f);
  } else {
    float v = x[i];
    x[i] = .5f * v *
           (1 + tanhf(.7978845608028654f * (v + .044715f * v * v * v))) * p[i];
  }
}
__global__ void norm_kernel(const float* x, const float* w, const float* b,
                            float* y, int width) {
  int row = blockIdx.x;
  __shared__ float a[256];
  float sum = 0;
  for (int c = threadIdx.x; c < width; c += 256)
    sum += x[size_t(row) * width + c];
  a[threadIdx.x] = sum;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (threadIdx.x < d) a[threadIdx.x] += a[threadIdx.x + d];
    __syncthreads();
  }
  float mean = a[0] / width;
  sum = 0;
  for (int c = threadIdx.x; c < width; c += 256) {
    float v = x[size_t(row) * width + c] - mean;
    sum += v * v;
  }
  __syncthreads();
  a[threadIdx.x] = sum;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (threadIdx.x < d) a[threadIdx.x] += a[threadIdx.x + d];
    __syncthreads();
  }
  float inv = rsqrtf(a[0] / width + 1e-5f);
  for (int c = threadIdx.x; c < width; c += 256)
    y[size_t(row) * width + c] =
        (x[size_t(row) * width + c] - mean) * inv * w[c] + b[c];
}
template <typename T>
__global__ void gn(const T* x, const __half* w, const __half* b, T* y,
                   int channels, int frames, int spatial) {
  int frame = blockIdx.x / 32, group = blockIdx.x % 32, cpg = channels / 32,
      count = cpg * spatial;
  __shared__ float a[256];
  float sum = 0;
  for (int j = threadIdx.x; j < count; j += 256) {
    int c = group * cpg + j / spatial;
    sum += static_cast<float>(x[(size_t(c) * frames + frame) * spatial + j % spatial]);
  }
  a[threadIdx.x] = sum;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (threadIdx.x < d) a[threadIdx.x] += a[threadIdx.x + d];
    __syncthreads();
  }
  float mean = a[0] / count;
  sum = 0;
  for (int j = threadIdx.x; j < count; j += 256) {
    int c = group * cpg + j / spatial;
    float v = static_cast<float>(x[(size_t(c) * frames + frame) * spatial + j % spatial]) - mean;
    sum += v * v;
  }
  __syncthreads();
  a[threadIdx.x] = sum;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (threadIdx.x < d) a[threadIdx.x] += a[threadIdx.x + d];
    __syncthreads();
  }
  float inv = rsqrtf(a[0] / count + 1e-6f);
  for (int j = threadIdx.x; j < count; j += 256) {
    int c = group * cpg + j / spatial;
    size_t at = (size_t(c) * frames + frame) * spatial + j % spatial;
    float v = (static_cast<float>(x[at]) - mean) * inv * __half2float(w[c]) + __half2float(b[c]);
    y[at] = v / (1 + expf(-v));
  }
}
__global__ void causal_attention(const float* qkv, const float* qb,
                                 const float* kb, const float* vb, float* y,
                                 int length) {
  int query = blockIdx.x % length, head = (blockIdx.x / length) % 8,
      batch = blockIdx.x / (length * 8);
  __shared__ float scores[608];
  __shared__ float reduction[256];
  float maximum = -CUDART_INF_F;
  for (int key = threadIdx.x; key <= query; key += 256) {
    float dot = 0;
    for (int d = 0; d < 256; ++d) {
      int c = head * 256 + d;
      dot = fmaf(qkv[(size_t(batch) * length + query) * 6144 + c] + qb[c],
                 qkv[(size_t(batch) * length + key) * 6144 + 2048 + c] + kb[c],
                 dot);
    }
    dot *= .0625f;
    scores[key] = dot;
    maximum = fmaxf(maximum, dot);
  }
  reduction[threadIdx.x] = maximum;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (threadIdx.x < d)
      reduction[threadIdx.x] =
          fmaxf(reduction[threadIdx.x], reduction[threadIdx.x + d]);
    __syncthreads();
  }
  maximum = reduction[0];
  float sum = 0;
  for (int key = threadIdx.x; key <= query; key += 256) {
    float v = expf(scores[key] - maximum);
    scores[key] = v;
    sum += v;
  }
  __syncthreads();
  reduction[threadIdx.x] = sum;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (threadIdx.x < d) reduction[threadIdx.x] += reduction[threadIdx.x + d];
    __syncthreads();
  }
  sum = reduction[0];
  int c = head * 256 + threadIdx.x;
  float value = 0;
  for (int key = 0; key <= query; ++key)
    value += scores[key] / sum *
             (qkv[(size_t(batch) * length + key) * 6144 + 4096 + c] + vb[c]);
  y[(size_t(batch) * length + query) * 2048 + c] = value;
}
__global__ void pool_heads(const float* x, float* y, int rows) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * 32) return;
  int row = i / 32, c = i % 32;
  float sum = 0;
  for (int d = 0; d < 8; ++d) {
    float heads = 0;
    for (int h = 0; h < 8; ++h)
      heads += x[size_t(row) * 2048 + h * 256 + c * 8 + d];
    sum += heads / 8;
  }
  y[i] = sum / 8;
}
int blocks(size_t count) {
  if (count > size_t(std::numeric_limits<int>::max()) * 256)
    throw std::overflow_error("reference encoder launch overflow");
  return int((count + 255) / 256);
}
}  // namespace
ReferenceEncoderOps::ReferenceEncoderOps(cudaStream_t s) : stream_(s) {
  blas_check(cublas_create(&blas_));
  try {
    blas_check(cublas_set_stream(blas_, s));
    blas_check(cublas_set_math_mode(blas_, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));
  } catch (...) {
    cublas_destroy(blas_);
    throw;
  }
}
ReferenceEncoderOps::~ReferenceEncoderOps() {
  cudaStreamSynchronize(stream_);  // Includes exception paths before pool teardown.
  if (blas_) cublas_destroy(blas_);
}
void ReferenceEncoderOps::report_memory(const char* label) const {
  memory_.report(label);
  if (!StepProfiler::instance().enabled()) return;
  constexpr double gib = 1024.0 * 1024 * 1024;
  std::printf("references  %s CUDA pools: activation peak %.3f GiB, scratch peak %.3f GiB, reserved %.3f GiB, %zu allocations\n",
      label, activations_.peak_bytes() / gib, scratch_.peak_bytes() / gib,
      (activations_.reserved_bytes() + scratch_.reserved_bytes()) / gib,
      activations_.allocations() + scratch_.allocations());
}
ReferenceBuffer<float> ReferenceEncoderOps::conv3d(const float* x, const __half* w,
                                                const __half* b, int ci, int co,
                                                int t, int h, int width, int k,
                                                int ss, int ts, bool down) {
  if (!x || !w || ci <= 0 || co <= 0 || t <= 0 || h <= 0 || width <= 0 ||
      (k != 1 && k != 3) || (ss != 1 && ss != 2) || (ts != 1 && ts != 2))
    throw std::invalid_argument("reference conv3d: invalid shape");
  int oh = h / ss, ow = width / ss, ot = (t - 1) / ts + 1, K = ci * k * k * k,
      N = ot * oh * ow;
  const int tile = std::min(N, 1024);
  ReferenceBuffer<float> col(size_t(K) * tile, scratch_), weight(size_t(co) * K, scratch_),
      bias(b ? co : 0, scratch_), out(size_t(co) * N, activations_);
  memory_.sample();
  convert<<<blocks(weight.size()), 256, 0, stream_>>>(w, weight.get(),
                                                      weight.size());
  if (b) convert<<<blocks(co), 256, 0, stream_>>>(b, bias.get(), co);
  float one = 1, zero = 0;
  for (int start = 0; start < N; start += tile) {
    int n = std::min(tile, N - start);
    col3<<<blocks(size_t(K) * n), 256, 0, stream_>>>(
        x, col.get(), ci, t, h, width, k, ss, ts, down, oh, ow, start, n);
    blas_check(cublas_sgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, n, co, K, &one,
                            col.get(), n, weight.get(), K, &zero,
                            out.get() + start, N));
  }
  if (b)
    bias_channels<<<blocks(out.size()), 256, 0, stream_>>>(out.get(),
                                                           bias.get(), N, co);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
  return out;
}
ReferenceBuffer<__half> ReferenceEncoderOps::conv3d(const __half* x, const __half* w,
    const __half* b, int ci, int co, int t, int h, int width, int k,
    int ss, int ts, bool down) {
  if (!x || !w || ci <= 0 || co <= 0 || t <= 0 || h <= 0 || width <= 0 ||
      (k != 1 && k != 3) || (ss != 1 && ss != 2) || (ts != 1 && ts != 2))
    throw std::invalid_argument("reference conv3d: invalid shape");
  const int oh = h / ss, ow = width / ss, ot = (t - 1) / ts + 1,
            K = ci * k * k * k, N = ot * oh * ow;
  const int tile = std::min(N, 2048);
  ReferenceBuffer<__half> col(size_t(K) * tile, scratch_);
  auto out = allocate<__half>(size_t(co) * N);
  memory_.sample();
  float one = 1, zero = 0;
  for (int start = 0; start < N; start += tile) {
    const int n = std::min(tile, N - start);
    col3<<<blocks(size_t(K) * n), 256, 0, stream_>>>(
        x, col.get(), ci, t, h, width, k, ss, ts, down, oh, ow, start, n);
    blas_check(cublas_gemm_ex(blas_, CUBLAS_OP_N, CUBLAS_OP_N, n, co, K,
        &one, col.get(), CUDA_R_16F, n, w, CUDA_R_16F, K, &zero,
        out.get() + start, CUDA_R_16F, N, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  if (b) half_bias<<<blocks(out.size()), 256, 0, stream_>>>(out.get(), b, N, out.size());
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
  return out;
}
ReferenceBuffer<float> ReferenceEncoderOps::conv1d(const float* x, const float* w,
                                                const float* b, int batch,
                                                int ci, int co, int len, int k,
                                                int stride, int pad, int dil) {
  int n = (len + 2 * pad - dil * (k - 1) - 1) / stride + 1;
  if (n <= 0 || batch <= 0 || ci <= 0 || co <= 0)
    throw std::invalid_argument("reference conv1d: invalid shape");
  auto out = allocate<float>(size_t(batch) * co * n);
  int K = ci * k, tile = std::min(n, 4096);
  ReferenceBuffer<float> col(size_t(K) * tile, scratch_);
  memory_.sample();
  float one = 1, zero = 0;
  for (int bch = 0; bch < batch; ++bch) {
    float* y = out.get() + size_t(bch) * co * n;
    for (int start = 0; start < n; start += tile) {
      int count = std::min(tile, n - start);
      col1<<<blocks(size_t(K) * count), 256, 0, stream_>>>(
          x + size_t(bch) * ci * len, col.get(), ci, len, k, stride, pad, dil,
          start, count);
      blas_check(cublas_sgemm(blas_, CUBLAS_OP_N, CUBLAS_OP_N, count, co, K,
                              &one, col.get(), count, w, K, &zero, y + start,
                              n));
    }
    if (b)
      bias_channels<<<blocks(size_t(n) * co), 256, 0, stream_>>>(y, b, n, co);
  }
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
  return out;
}
ReferenceBuffer<float> ReferenceEncoderOps::linear(const float* x, const float* w,
                                                const float* b, int rows,
                                                int in, int out) {
  auto y = allocate<float>(size_t(rows) * out);
  memory_.sample();
  float one = 1, zero = 0;
  blas_check(cublas_sgemm(blas_, CUBLAS_OP_T, CUBLAS_OP_N, out, rows, in, &one,
                          w, in, x, in, &zero, y.get(), out));
  if (b)
    bias_rows<<<blocks(y.size()), 256, 0, stream_>>>(y.get(), b, rows, out);
  return y;
}
ReferenceBuffer<float> ReferenceEncoderOps::norm(const float* x, const float* w,
                                              const float* b, int rows,
                                              int width) {
  auto y = allocate<float>(size_t(rows) * width);
  norm_kernel<<<rows, 256, 0, stream_>>>(x, w, b, y.get(), width);
  return y;
}
ReferenceBuffer<float> ReferenceEncoderOps::transpose(const float* x, int batch,
                                                   int rows, int cols) {
  auto y = allocate<float>(size_t(batch) * rows * cols);
  transpose_kernel<<<blocks(y.size()), 256, 0, stream_>>>(x, y.get(), batch,
                                                          rows, cols);
  return y;
}
ReferenceBuffer<float> ReferenceEncoderOps::attention(const float* x,
                                                   const float* qb,
                                                   const float* kb,
                                                   const float* vb, int batch,
                                                   int len) {
  if (len > 600 || len <= 0)
    throw std::invalid_argument(
        "reference audio: attention length must be 1..600");
  auto heads = allocate<float>(size_t(batch) * len * 2048);
  auto out = allocate<float>(size_t(batch) * len * 32);
  memory_.sample();
  causal_attention<<<batch * len * 8, 256, 0, stream_>>>(x, qb, kb, vb,
                                                         heads.get(), len);
  pool_heads<<<blocks(out.size()), 256, 0, stream_>>>(heads.get(), out.get(),
                                                      batch * len);
  return out;
}
void ReferenceEncoderOps::snake(float* x, const float* a, int b, int c,
                                int len) {
  size_t n = size_t(b) * c * len;
  element<<<blocks(n), 256, 0, stream_>>>(x, a, n, c, len, 1);
}
void ReferenceEncoderOps::add(float* x, const float* p, size_t n) {
  element<<<blocks(n), 256, 0, stream_>>>(x, p, n, 0, 0, 0);
}
void ReferenceEncoderOps::add(__half* x, const __half* p, size_t n) {
  half_add<<<blocks(n), 256, 0, stream_>>>(x, p, n);
}
void ReferenceEncoderOps::geglu(float* x, const float* p, size_t n) {
  element<<<blocks(n), 256, 0, stream_>>>(x, p, n, 0, 0, 2);
}
void reference_groupnorm(const float* x, const __half* w, const __half* b,
                         float* y, int c, int t, int h, int width,
                         cudaStream_t stream) {
  gn<<<32 * t, 256, 0, stream>>>(x, w, b, y, c, t, h * width);
}
void reference_groupnorm(const __half* x, const __half* w, const __half* b,
                         __half* y, int c, int t, int h, int width, cudaStream_t stream) {
  gn<<<32 * t, 256, 0, stream>>>(x, w, b, y, c, t, h * width);
}
}  // namespace slopfab::cuda
