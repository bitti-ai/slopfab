#include "runtime.cuh"
#include "slopfab/cuda/gemm.cuh"
#include <algorithm>
#include <cmath>

namespace slopfab::music {
__global__ void rms(const BFloat* x, const BFloat* w, BFloat* y, int width) {
  __shared__ float sums[512];
  const int t = threadIdx.x, row = blockIdx.x;
  float partial[4] = {};
  if (width > 128) {
    for (int first = t * 4; first < width; first += int(blockDim.x) * 4)
      for (int j = 0; j < 4; ++j) {
        float a = __bfloat162float(x[size_t(row) * width + first + j]);
        partial[j] = __fadd_rn(partial[j], __fmul_rn(a, a));
      }
  } else {
    for (int first = t; first < width; first += int(blockDim.x) * 4)
      for (int j = 0; j < 4 && first + j * int(blockDim.x) < width; ++j) {
        float a = __bfloat162float(x[size_t(row) * width + first + j * blockDim.x]);
        partial[j] = __fadd_rn(partial[j], __fmul_rn(a, a));
      }
  }
  float sum = ((partial[0] + partial[1]) + partial[2]) + partial[3];
  sums[t] = sum;
  __syncthreads();
  for (int d = int(blockDim.x) / 2; d >= 32; d /= 2) {
    if (t < d) {
      sum += sums[t + d];
      sums[t] = sum;
    }
    __syncthreads();
  }
  if (t < 32) {
    for (int d = 1; d < 32; d *= 2)
      sum += __shfl_down_sync(0xffffffff, sum, d);
    if (t == 0)
      sums[0] = sum;
  }
  __syncthreads();
  // YuE rounds the inverse RMS before multiplying, unlike Qwen3.
  const BFloat inv = __float2bfloat16(rsqrtf(__fadd_rn(__fmul_rn(sums[0], 1.f / width), 1e-6f)));
  for (int c = t; c < width; c += blockDim.x)
    y[size_t(row) * width + c] = (x[size_t(row) * width + c] * inv) * w[c];
}

__global__ void add_condition(BFloat* x, const BFloat* t, const BFloat* pe, int n, int width) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] = (x[i] + t[i % width]) + pe[i];
}

__global__ void causal_probabilities(const float* scores, BFloat* probabilities, int keys,
                                     int first) {
  __shared__ float reduction[256];
  int q = blockIdx.x, t = threadIdx.x, limit = first + q + 1;
  float maximum = -INFINITY;
  for (int k = t; k < limit; k += 256)
    maximum = fmaxf(maximum, scores[size_t(q) * keys + k]);
  reduction[t] = maximum;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (t < d)
      reduction[t] = fmaxf(reduction[t], reduction[t + d]);
    __syncthreads();
  }
  maximum = reduction[0];
  float sum = 0;
  for (int k = t; k < limit; k += 256)
    sum += expf(scores[size_t(q) * keys + k] - maximum);
  reduction[t] = sum;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (t < d)
      reduction[t] += reduction[t + d];
    __syncthreads();
  }
  sum = reduction[0];
  for (int k = t; k < keys; k += 256)
    probabilities[size_t(q) * keys + k] =
        __float2bfloat16(k < limit ? expf(scores[size_t(q) * keys + k] - maximum) / sum : 0);
}

Tensor gpu_causal_attention(Runtime& rt, const Tensor& q, const Tensor& k, const Tensor& v) {
  const int dim = 128, tile = 128;
  Tensor out(q.rows, q.cols);
  cuda::DeviceBuffer<float> scores(size_t(std::min(tile, q.rows)) * k.rows);
  cuda::DeviceBuffer<BFloat> probabilities(scores.size());
  const float scale = 1 / std::sqrt(float(dim)), one = 1, zero = 0;
  for (int head = 0; head < 16; ++head)
    for (int first = 0; first < q.rows; first += tile) {
      int count = std::min(tile, q.rows - first), kh = head / 2;
      SLOPFAB_CUBLAS_CHECK(cuda::cublas_gemm_ex(
          rt.blas, CUBLAS_OP_T, CUBLAS_OP_N, k.rows, count, dim, &scale, k.ptr() + kh * dim,
          CUDA_R_16BF, k.cols, q.ptr() + size_t(first) * q.cols + head * dim, CUDA_R_16BF, q.cols,
          &zero, scores.get(), CUDA_R_32F, k.rows, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      causal_probabilities<<<count, 256>>>(scores.get(), probabilities.get(), k.rows, first);
      SLOPFAB_CUBLAS_CHECK(cuda::cublas_gemm_ex(
          rt.blas, CUBLAS_OP_N, CUBLAS_OP_N, dim, count, k.rows, &one, v.ptr() + kh * dim,
          CUDA_R_16BF, v.cols, probabilities.get(), CUDA_R_16BF, k.rows, &zero,
          out.ptr() + size_t(first) * out.cols + head * dim, CUDA_R_16BF, out.cols,
          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
  return out;
}

__global__ void layernorm(const BFloat* x, const BFloat* gamma, const BFloat* beta, BFloat* y,
                          int width) {
  __shared__ float sums[256], squares[256];
  int t = threadIdx.x, r = blockIdx.x;
  float sum = 0, sq = 0;
  for (int c = t; c < width; c += 256) {
    float v = __bfloat162float(x[size_t(r) * width + c]);
    sum += v;
    sq += v * v;
  }
  sums[t] = sum;
  squares[t] = sq;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (t < d) {
      sums[t] += sums[t + d];
      squares[t] += squares[t + d];
    }
    __syncthreads();
  }
  float mean = sums[0] / width, inv = rsqrtf(fmaxf(0, squares[0] / width - mean * mean) + 1e-5f);
  for (int c = t; c < width; c += 256)
    y[size_t(r) * width + c] = __float2bfloat16(
        (__bfloat162float(x[size_t(r) * width + c]) - mean) * inv * __bfloat162float(gamma[c]) +
        __bfloat162float(beta[c]));
}

Tensor gpu_ln(Runtime& rt, const Tensor& x, const std::string& p) {
  Tensor y(x.rows, x.cols);
  layernorm<<<x.rows, 256>>>(x.ptr(), rt.weight(p + ".gamma"), rt.weight(p + ".beta"), y.ptr(),
                             x.cols);
  return y;
}

float gpu_bf(float x) {
  return __bfloat162float(__float2bfloat16(x));
}

void gpu_copy_rows(const Tensor& x, Tensor& y, int first, int dest, int count) {
  SLOPFAB_CUDA_CHECK(cudaMemcpy(y.ptr() + size_t(dest) * y.cols, x.ptr() + size_t(first) * x.cols,
                                size_t(count) * x.cols * 2, cudaMemcpyDeviceToDevice));
}

void gpu_copy_columns(const Tensor& x, Tensor& y, int offset) {
  SLOPFAB_CUDA_CHECK(cudaMemcpy2D(y.ptr() + offset, y.cols * 2, x.ptr(), x.cols * 2, x.cols * 2,
                                  x.rows, cudaMemcpyDeviceToDevice));
}

void gpu_zero(Tensor& x) {
  SLOPFAB_CUDA_CHECK(cudaMemset(x.ptr(), 0, x.data.nbytes()));
}

void gpu_condition(Runtime& rt, Tensor& x, const Tensor& t, const std::string& name) {
  int n = x.rows * x.cols;
  add_condition<<<(n + 255) / 256, 256>>>(x.ptr(), t.ptr(), rt.weight(name), n, x.cols);
}

Tensor gpu_yue_norm(Runtime& rt, const Tensor& x, const std::string& name, int width) {
  int rows = x.rows * x.cols / width, outputs = 1;
  while (outputs * 2 <= rows && outputs < 512)
    outputs *= 2;
  int threads = std::min(width > 128 ? width / 4 : width, 512 / std::min(outputs, 16));
  Tensor y(x.rows, x.cols);
  rms<<<rows, threads>>>(x.ptr(), rt.weight(name), y.ptr(), width);
  return y;
}
}
