#include "runtime.cuh"
#include "slopfab/tensor_convert.h"
#include "slopfab/cuda/gemm.cuh"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace slopfab::krea {
namespace {
struct Image {
  int h, w, c;
  cuda::DeviceBuffer<float> data;

  Image(int height, int width, int channels)
      : h(height), w(width), c(channels), data(size_t(h) * w * c) {
  }
};

__global__ void im2col(const float* x, float* out, int h, int w, int ci, int kernel, int first,
                       int count, int stride, int out_w) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int columns = ci * kernel * kernel;
  if (index >= count * columns)
    return;
  const int r = first + index / columns, col = index % columns;
  const int pad = stride == 1 ? kernel / 2 : 0;
  const int y = r / out_w * stride + col / kernel % kernel - pad,
            xx = r % out_w * stride + col % kernel - pad;
  out[index] = (y >= 0 && y < h && xx >= 0 && xx < w)
                   ? x[(size_t(y) * w + xx) * ci + col / (kernel * kernel)]
                   : 0.f;
}

__global__ void add_bias(float* x, const float* bias, int n, int c) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] += bias[i % c];
}

Image conv(Runtime& rt, const Image& x, const std::string& p) {
  const auto& tensor = rt.checkpoint.at(p + ".weight");
  const auto& shape = tensor.shape;
  if ((shape.size() != 4 && shape.size() != 5) || shape[1] != x.c ||
      shape[shape.size() - 2] != shape.back() || (shape.back() != 1 && shape.back() != 3))
    throw std::invalid_argument("Qwen-Image VAE convolution shape: " + p);
  const int co = int(shape[0]), kernel = int(shape.back()), columns = x.c * kernel * kernel;
  const int temporal = shape.size() == 5 ? int(shape[2]) : 1;
  auto all = to_f32(tensor);
  std::vector<float> host(size_t(co) * columns);
  // A still image is the first causal frame: all earlier temporal slices see zero.
  for (int o = 0; o < co; ++o)
    for (int i = 0; i < x.c; ++i)
      std::copy_n(all.data() + ((size_t(o) * x.c + i) * temporal + temporal - 1) * kernel * kernel,
                  kernel * kernel, host.data() + (size_t(o) * x.c + i) * kernel * kernel);
  cuda::DeviceBuffer<float> weight(host.size());
  SLOPFAB_CUDA_CHECK(
      cudaMemcpy(weight.get(), host.data(), host.size() * 4, cudaMemcpyHostToDevice));
  Image y(x.h, x.w, co);
  const int tile = 2048, pixels = x.h * x.w;
  cuda::DeviceBuffer<float> patches(size_t(std::min(tile, pixels)) * columns);
  for (int first = 0; first < pixels; first += tile) {
    int count = std::min(tile, pixels - first);
    const float* input = x.data.get() + size_t(first) * x.c;
    if (kernel != 1) {
      im2col<<<(count * columns + 255) / 256, 256>>>(x.data.get(), patches.get(), x.h, x.w, x.c,
                                                     kernel, first, count, 1, x.w);
      input = patches.get();
    }
    cuda::gemm_nt(rt.blas, input, weight.get(), y.data.get() + size_t(first) * co, count, co,
                  columns);
  }
  const int n = pixels * co;
  add_bias<<<(n + 255) / 256, 256>>>(y.data.get(), rt.float_weight(p + ".bias"), n, co);
  return y;
}

__global__ void rms_image(const float* x, const float* weight, float* out, int channels,
                          bool silu) {
  __shared__ float sums[256];
  int t = threadIdx.x;
  size_t offset = size_t(blockIdx.x) * channels;
  float sum = 0;
  for (int c = t; c < channels; c += 256)
    sum += x[offset + c] * x[offset + c];
  sums[t] = sum;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (t < d)
      sums[t] += sums[t + d];
    __syncthreads();
  }
  float scale = sqrtf(float(channels)) / fmaxf(sqrtf(sums[0]), 1e-12f);
  for (int c = t; c < channels; c += 256) {
    float v = x[offset + c] * scale * weight[c];
    out[offset + c] = silu ? v / (1 + expf(-v)) : v;
  }
}

Image normalize(Runtime& rt, const Image& x, const std::string& p, bool silu = true) {
  if (rt.checkpoint.at(p + ".gamma").numel() != x.c)
    throw std::invalid_argument("Qwen-Image VAE RMS shape: " + p);
  Image y(x.h, x.w, x.c);
  rms_image<<<x.h * x.w, 256>>>(x.data.get(), rt.float_weight(p + ".gamma"), y.data.get(), x.c,
                                silu);
  return y;
}

__global__ void add(float* x, const float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    x[i] += y[i];
}

Image resnet(Runtime& rt, Image x, const std::string& p) {
  auto y = conv(rt, normalize(rt, x, p + ".residual.0"), p + ".residual.2");
  y = conv(rt, normalize(rt, y, p + ".residual.3"), p + ".residual.6");
  if (rt.checkpoint.find(p + ".shortcut.weight"))
    x = conv(rt, x, p + ".shortcut");
  int n = x.h * x.w * x.c;
  add<<<(n + 255) / 256, 256>>>(y.data.get(), x.data.get(), n);
  return y;
}

__global__ void nearest(const float* x, float* y, int h, int w, int channels) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h * w * 4 * channels)
    return;
  int r = i / channels, c = i % channels;
  y[i] = x[(size_t(r / (2 * w) / 2) * w + (r % (2 * w)) / 2) * channels + c];
}

Image upsample(Runtime& rt, const Image& x, const std::string& p) {
  Image y(x.h * 2, x.w * 2, x.c);
  const int n = y.h * y.w * y.c;
  nearest<<<(n + 255) / 256, 256>>>(x.data.get(), y.data.get(), x.h, x.w, x.c);
  return conv(rt, y, p + ".resample.1");
}

__global__ void softmax(float* scores, int width, float scale) {
  __shared__ float reduce[256];
  float* row = scores + size_t(blockIdx.x) * width;
  float maximum = -3.402823466e38f;
  for (int i = threadIdx.x; i < width; i += 256)
    maximum = fmaxf(maximum, row[i] * scale);
  reduce[threadIdx.x] = maximum;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (threadIdx.x < d)
      reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + d]);
    __syncthreads();
  }
  maximum = reduce[0];
  __syncthreads();
  float total = 0;
  for (int i = threadIdx.x; i < width; i += 256) {
    row[i] = expf(row[i] * scale - maximum);
    total += row[i];
  }
  reduce[threadIdx.x] = total;
  __syncthreads();
  for (int d = 128; d; d /= 2) {
    if (threadIdx.x < d)
      reduce[threadIdx.x] += reduce[threadIdx.x + d];
    __syncthreads();
  }
  total = reduce[0];
  for (int i = threadIdx.x; i < width; i += 256)
    row[i] /= total;
}

Image attention(Runtime& rt, Image x, const std::string& p) {
  auto normalized = normalize(rt, x, p + ".norm", false);
  auto qkv = conv(rt, normalized, p + ".to_qkv");
  Image q(x.h, x.w, x.c), k(x.h, x.w, x.c), v(x.h, x.w, x.c);
  for (int i = 0; i < 3; ++i) {
    float* dst = i == 0 ? q.data.get() : (i == 1 ? k.data.get() : v.data.get());
    SLOPFAB_CUDA_CHECK(cudaMemcpy2D(dst, x.c * 4, qkv.data.get() + i * x.c, x.c * 3 * 4, x.c * 4,
                                    x.h * x.w, cudaMemcpyDeviceToDevice));
  }
  Image y(x.h, x.w, x.c);
  const int pixels = x.h * x.w, tile = 128;
  cuda::DeviceBuffer<float> scores(size_t(tile) * pixels);
  for (int first = 0; first < pixels; first += tile) {
    int count = std::min(tile, pixels - first);
    cuda::gemm_nt(rt.blas, q.data.get() + size_t(first) * x.c, k.data.get(), scores.get(), count,
                  pixels, x.c);
    softmax<<<count, 256>>>(scores.get(), pixels, 1.f / std::sqrt(float(x.c)));
    cuda::gemm_nn(rt.blas, scores.get(), v.data.get(), y.data.get() + size_t(first) * x.c, count,
                  x.c, pixels);
  }
  y = conv(rt, y, p + ".proj");
  int n = pixels * x.c;
  add<<<(n + 255) / 256, 256>>>(y.data.get(), x.data.get(), n);
  return y;
}
}

std::vector<float> decode_image(const std::string& path, const std::vector<float>& latent,
                                int height, int width) {
  if (height < 1 || width < 1 || latent.size() != size_t(height) * width * 64)
    throw std::invalid_argument("Krea 2 VAE requires [H/16,W/16,64] latents");
  const float mean[] = {-.7571f, -.7089f, -.9113f, .1075f,  -.1745f, .9653f,  -.1517f, 1.5508f,
                        .4134f,  -.0715f, .5517f,  -.3632f, -.1922f, -.9497f, .2503f,  -.2921f};
  const float stddev[] = {2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f, 2.0743f,
                          3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f};
  Runtime rt(path);
  std::vector<float> unpacked(latent.size());
  for (int y = 0; y < height; ++y)
    for (int xx = 0; xx < width; ++xx)
      for (int c = 0; c < 16; ++c)
        for (int dy = 0; dy < 2; ++dy)
          for (int dx = 0; dx < 2; ++dx)
            unpacked[(size_t(y * 2 + dy) * width * 2 + xx * 2 + dx) * 16 + c] =
                latent[(size_t(y) * width + xx) * 64 + c * 4 + dy * 2 + dx] * stddev[c] + mean[c];
  Image x(height * 2, width * 2, 16);
  SLOPFAB_CUDA_CHECK(
      cudaMemcpy(x.data.get(), unpacked.data(), unpacked.size() * 4, cudaMemcpyHostToDevice));
  x = conv(rt, x, "conv2");
  x = conv(rt, x, "decoder.conv1");
  x = resnet(rt, std::move(x), "decoder.middle.0");
  x = attention(rt, std::move(x), "decoder.middle.1");
  x = resnet(rt, std::move(x), "decoder.middle.2");
  for (int i = 0; i < 15; ++i) {
    const std::string p = "decoder.upsamples." + std::to_string(i);
    // Temporal upsampling is bypassed on the first frame, as in the upstream cache.
    if (i == 3 || i == 7 || i == 11)
      x = upsample(rt, x, p);
    else
      x = resnet(rt, std::move(x), p);
  }
  x = conv(rt, normalize(rt, x, "decoder.head.0"), "decoder.head.2");
  std::vector<float> pixels(x.data.size()), planar(x.data.size());
  SLOPFAB_CUDA_CHECK(
      cudaMemcpy(pixels.data(), x.data.get(), pixels.size() * 4, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < pixels.size() / 3; ++i)
    for (int c = 0; c < 3; ++c) {
      float value = pixels[i * 3 + c];
      if (!std::isfinite(value))
        throw std::runtime_error("non-finite Krea 2 VAE output");
      planar[size_t(c) * pixels.size() / 3 + i] = std::clamp(value * .5f + .5f, 0.f, 1.f);
    }
  return planar;
}
}
