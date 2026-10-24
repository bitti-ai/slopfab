#include "runtime.cuh"
#include <cmath>
#include <stdexcept>

namespace slopfab::fibo {
namespace {
__device__ float f(BFloat v) {
  return __bfloat162float(v);
}

__global__ void normalization(const BFloat* x, BFloat* out, const BFloat* weight,
                              const BFloat* bias, int width, bool rms, float eps) {
  __shared__ float sums[256], squares[256];
  int row = blockIdx.x, t = threadIdx.x;
  float sum = 0, sq = 0;
  for (int c = t; c < width; c += 256) {
    float a = f(x[size_t(row) * width + c]);
    sum += a;
    sq += a * a;
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
  float mean = rms ? 0 : sums[0] / width;
  float inv = rsqrtf(fmaxf(squares[0] / width - mean * mean, 0.f) + eps);
  for (int c = t; c < width; c += 256) {
    float a = (f(x[size_t(row) * width + c]) - mean) * inv;
    if (weight)
      a = a * f(weight[c]) + (bias ? f(bias[c]) : 0.f);
    out[size_t(row) * width + c] = __float2bfloat16(a);
  }
}

__global__ void gelu_kernel(const BFloat* x, BFloat* y, int n, bool approximate) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float a = f(x[i]);
  y[i] = __float2bfloat16(.5f * a *
                          (1 + (approximate ? tanhf(.7978845608f * (a + .044715f * a * a * a))
                                            : erff(a * .7071067812f))));
}

__global__ void rotary_kernel(BFloat* x, const float* cos, const float* sin, int n, int heads,
                              int dim, bool adjacent) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  int j = i % (dim / 2), r = i / (dim / 2) / heads;
  int offset = (i / (dim / 2)) * dim, a = adjacent ? 2 * j : j,
      b = adjacent ? 2 * j + 1 : j + dim / 2;
  float va = f(x[offset + a]), vb = f(x[offset + b]), c = cos[r * (dim / 2) + j],
        s = sin[r * (dim / 2) + j];
  x[offset + a] = __float2bfloat16(va * c - vb * s);
  x[offset + b] = __float2bfloat16(vb * c + va * s);
}

__global__ void scatter_kernel(BFloat* x, const BFloat* y, const int32_t* idx, int n, int width,
                               bool add) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  size_t dest = size_t(idx[i / width]) * width + i % width;
  x[dest] = add ? __float2bfloat16(f(x[dest]) + f(y[i])) : y[i];
}
}

Tensor rows(const Tensor& x, int start, int count) {
  if (start < 0 || count < 0 || start + count > x.rows)
    throw std::invalid_argument("Fibo row slice");
  Tensor y(count, x.cols);
  SLOPFAB_CUDA_CHECK(cudaMemcpy(y.ptr(), x.ptr() + size_t(start) * x.cols,
                                size_t(count) * x.cols * 2, cudaMemcpyDeviceToDevice));
  return y;
}

Tensor join(const Tensor& a, const Tensor& b) {
  if (a.cols != b.cols)
    throw std::invalid_argument("Fibo row concatenation");
  Tensor y(a.rows + b.rows, a.cols);
  SLOPFAB_CUDA_CHECK(cudaMemcpy(y.ptr(), a.ptr(), a.data.size() * 2, cudaMemcpyDeviceToDevice));
  SLOPFAB_CUDA_CHECK(
      cudaMemcpy(y.ptr() + a.data.size(), b.ptr(), b.data.size() * 2, cudaMemcpyDeviceToDevice));
  return y;
}

Tensor layer_norm(Runtime& rt, const Tensor& x, const std::string& p, bool rms, float eps) {
  Tensor y(x.rows, x.cols);
  normalization<<<x.rows, 256>>>(x.ptr(), y.ptr(), p.empty() ? nullptr : rt.weight(p + ".weight"),
                                 p.empty() || rms ? nullptr : rt.weight(p + ".bias"), x.cols, rms,
                                 eps);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
  return y;
}

Tensor gelu(const Tensor& x, bool approximate) {
  Tensor y(x.rows, x.cols);
  int n = x.rows * x.cols;
  gelu_kernel<<<(n + 255) / 256, 256>>>(x.ptr(), y.ptr(), n, approximate);
  return y;
}

void rope(Tensor& x, int heads, const std::vector<float>& cos, const std::vector<float>& sin,
          bool adjacent) {
  int dim = x.cols / heads;
  if (dim % 2 || cos.size() != size_t(x.rows) * dim / 2 || sin.size() != cos.size())
    throw std::invalid_argument("Fibo RoPE shape");
  cuda::DeviceBuffer<float> c(cos.size()), s(sin.size());
  c.copy_from_host(cos.data(), cos.size());
  s.copy_from_host(sin.data(), sin.size());
  int n = x.rows * x.cols / 2;
  rotary_kernel<<<(n + 255) / 256, 256>>>(x.ptr(), c.get(), s.get(), n, heads, dim, adjacent);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void scatter(Tensor& x, const Tensor& y, const std::vector<int32_t>& indices, bool add) {
  if (x.cols != y.cols || y.rows != int(indices.size()))
    throw std::invalid_argument("Fibo vision scatter shape");
  for (int i : indices)
    if (i < 0 || i >= x.rows)
      throw std::invalid_argument("Fibo vision scatter index");
  cuda::DeviceBuffer<int32_t> idx(indices.size());
  idx.copy_from_host(indices.data(), indices.size());
  int n = y.rows * y.cols;
  scatter_kernel<<<(n + 255) / 256, 256>>>(x.ptr(), y.ptr(), idx.get(), n, x.cols, add);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}
}
