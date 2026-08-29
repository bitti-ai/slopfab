#include "vidfab/cuda/keyframe_encoder.cuh"
#include "vidfab/cuda/deterministic_math.cuh"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace vidfab::cuda {
namespace {

__device__ int reflect_index(int i, int n) {
  if (n <= 1) return 0;
  while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i;
  return i;
}

__global__ void conv_kernel(const float* x, const __half* w, const __half* bias, float* y,
                            int cin, int cout, int ih, int iw, int oh, int ow, int k,
                            int stride, bool reflect, int pad_left, int pad_top) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = cout * oh * ow;
  if (index >= total) return;
  const int ox = index % ow;
  const int oy = (index / ow) % oh;
  const int oc = index / (oh * ow);
  float sum = bias ? __half2float(bias[oc]) : 0.0f;
  // Causal T=1: only kt=k-1 sees the sole frame. Its flattened weight offset
  // is therefore the last spatial kernel plane.
  for (int ic = 0; ic < cin; ++ic) {
    const size_t wb = (static_cast<size_t>(oc) * cin + ic) * k * k * k + (k - 1) * k * k;
    for (int ky = 0; ky < k; ++ky) {
      int sy = oy * stride + ky - pad_top;
      if (reflect) sy = reflect_index(sy, ih);
      else if (sy < 0 || sy >= ih) continue;
      for (int kx = 0; kx < k; ++kx) {
        int sx = ox * stride + kx - pad_left;
        if (reflect) sx = reflect_index(sx, iw);
        else if (sx < 0 || sx >= iw) continue;
        sum = fmaf(x[(static_cast<size_t>(ic) * ih + sy) * iw + sx],
                   __half2float(w[wb + ky * k + kx]), sum);
      }
    }
  }
  y[index] = sum;
}

__global__ void groupnorm_kernel(const float* x, const __half* weight, const __half* bias,
                                 float* y, int channels, int spatial, int groups, float eps) {
  const int group = blockIdx.x;
  const int cpg = channels / groups;
  const int count = cpg * spatial;
  __shared__ float scratch[256];
  __shared__ float group_mean;
  float sum = 0.0f, sq = 0.0f;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const int c = group * cpg + i / spatial;
    const float v = x[static_cast<size_t>(c) * spatial + i % spatial];
    sum += v;
    sq += v * v;
  }
  scratch[threadIdx.x] = sum;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d >>= 1) {
    if (threadIdx.x < d) scratch[threadIdx.x] += scratch[threadIdx.x + d];
    __syncthreads();
  }
  if (threadIdx.x == 0) group_mean = deterministic_divide(scratch[0], count);
  __syncthreads();
  const float mean = group_mean;
  scratch[threadIdx.x] = sq;
  __syncthreads();
  for (int d = blockDim.x / 2; d; d >>= 1) {
    if (threadIdx.x < d) scratch[threadIdx.x] += scratch[threadIdx.x + d];
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    const float second_moment = deterministic_divide(scratch[0], count);
    const float variance = fmaxf(0.0f, fmaf(-mean, mean, second_moment));
    const uint32_t base_bits = positive_float_add(__float_as_uint(variance),
                                                   __float_as_uint(eps));
    scratch[0] = deterministic_rsqrt(__uint_as_float(base_bits));
  }
  __syncthreads();
  const float inv = scratch[0];
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const int c = group * cpg + i / spatial;
    const size_t at = static_cast<size_t>(c) * spatial + i % spatial;
    const float normalized = (x[at] - mean) * inv;
    const float v = fmaf(normalized, __half2float(weight[c]), __half2float(bias[c]));
    y[at] = deterministic_silu(v);
  }
}

__global__ void add_kernel(const float* a, const float* b, float* y, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] + b[i];
}

}  // namespace

void launch_keyframe_conv3d(const float* x, const __half* weight, const __half* bias, float* y,
                            int cin, int cout, int ih, int iw, int k, int stride, bool reflect,
                            bool asymmetric, cudaStream_t stream) {
  if (!x || !weight || !y || cin <= 0 || cout <= 0 || ih <= 0 || iw <= 0 ||
      (k != 1 && k != 3) || (stride != 1 && stride != 2))
    throw std::runtime_error("keyframe conv3d: invalid arguments");
  const int oh = stride == 2 ? ih / 2 : ih;
  const int ow = stride == 2 ? iw / 2 : iw;
  // Downsample3D pads right/bottom only before a padding-zero convolution.
  // This is equivalent to a zero-pad convolution with left/top padding 0.
  const int pad = asymmetric ? 0 : k / 2;
  const int total = cout * oh * ow;
  conv_kernel<<<(total + 255) / 256, 256, 0, stream>>>(x, weight, bias, y, cin, cout, ih, iw,
                                                       oh, ow, k, stride, reflect,
                                                       pad, pad);
}

void launch_keyframe_groupnorm_silu(const float* x, const __half* weight, const __half* bias,
                                    float* y, int channels, int height, int width, int groups,
                                    float eps, cudaStream_t stream) {
  const uint64_t spatial = height > 0 && width > 0
      ? static_cast<uint64_t>(height) * static_cast<uint64_t>(width) : 0;
  const uint64_t total = channels > 0 ? static_cast<uint64_t>(channels) * spatial : 0;
  const uint64_t group_count = groups > 0 && channels > 0
      ? static_cast<uint64_t>(channels / groups) * spatial : 0;
  if (!x || !weight || !bias || !y || channels <= 0 || height <= 0 || width <= 0 ||
      groups <= 0 || channels % groups || !std::isnormal(eps) || eps <= 0.0f ||
      spatial > std::numeric_limits<int>::max() ||
      total > std::numeric_limits<uint32_t>::max() ||
      group_count > std::numeric_limits<uint32_t>::max())
    throw std::runtime_error("keyframe groupnorm: invalid arguments");
  groupnorm_kernel<<<groups, 256, 0, stream>>>(x, weight, bias, y, channels,
                                               static_cast<int>(spatial),
                                               groups, eps);
}

void launch_keyframe_add(const float* a, const float* b, float* y, size_t count,
                         cudaStream_t stream) {
  add_kernel<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(a, b, y, count);
}

}  // namespace vidfab::cuda
