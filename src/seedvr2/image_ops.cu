#include "image_ops.cuh"

namespace slopfab::seedvr2 {
namespace {
__global__ void prepare_tile(const float* input, BFloat* output, size_t count, int th, int tw,
                             int channels, int height, int width, int y, int x, bool image) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const int c = int(i % channels), xx = int(i / channels % tw), yy = int(i / channels / tw % th),
            z = int(i / channels / tw / th);
  const int sy = min(y + yy, height - 1), sx = min(x + xx, width - 1);
  float value = input[((size_t(z) * height + sy) * width + sx) * channels + c];
  value = image ? __fsub_rn(__fmul_rn(2.0f, value), 1.0f) : value / 0.9152f;
  output[i] = __float2bfloat16(value);
}

__device__ float feather_weight(int p, int lo, int hi, int extent, int overlap) {
  float value = 1.0f;
  if (lo > 0)
    value = fminf(value, float(p - lo + 1) / overlap);
  if (hi < extent)
    value = fminf(value, float(hi - p) / overlap);
  return value;
}

__global__ void accumulate_tile(const BFloat* tile, float* sum, float* coverage, size_t count,
                                int th, int tw, int channels, int height, int width,
                                int padded_height, int padded_width, int y, int x, int overlap) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count)
    return;
  const int c = int(i % channels), xx = int(i / channels % tw) + x,
            yy = int(i / channels / tw % th) + y, z = int(i / channels / tw / th);
  if (yy >= height || xx >= width)
    return;
  const float weight = __fmul_rn(feather_weight(yy, y, y + th, padded_height, overlap),
                                 feather_weight(xx, x, x + tw, padded_width, overlap));
  const size_t pos = size_t(yy) * width + xx;
  const size_t out = (size_t(z) * height * width + pos) * channels + c;
  // Tiles execute in the original order. Explicit rounded operations avoid an
  // FMA changing the CPU implementation's multiply-then-add rounding.
  sum[out] = __fadd_rn(sum[out], __fmul_rn(weight, __bfloat162float(tile[i])));
  if (z == 0 && c == 0)
    coverage[pos] = __fadd_rn(coverage[pos], weight);
}

__global__ void normalize_tiles(float* sum, const float* coverage, size_t count, size_t plane,
                                int channels, bool decoded) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    float value = sum[i] / coverage[(i / channels) % plane];
    sum[i] = decoded ? __fadd_rn(__fmul_rn(value, 0.5f), 0.5f) : value;
  }
}

int blocks(size_t count) {
  return int((count + 255) / 256);
}
}

void prepare_image_tile(const float* input, Tensor& tile, int height, int width, int y, int x) {
  prepare_tile<<<blocks(tile.size()), 256>>>(input, tile.data.get(), tile.size(), tile.h, tile.w,
                                             tile.c, height, width, y, x, true);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void prepare_latent_tile(const float* input, Tensor& tile, int height, int width, int y, int x) {
  prepare_tile<<<blocks(tile.size()), 256>>>(input, tile.data.get(), tile.size(), tile.h, tile.w,
                                             tile.c, height, width, y, x, false);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void accumulate_image_tile(const Tensor& tile, float* sum, float* coverage, int height, int width,
                           int padded_height, int padded_width, int y, int x, int overlap) {
  accumulate_tile<<<blocks(tile.size()), 256>>>(tile.data.get(), sum, coverage, tile.size(), tile.h,
                                                tile.w, tile.c, height, width, padded_height,
                                                padded_width, y, x, overlap);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void normalize_image_tiles(float* sum, const float* coverage, size_t count, int height, int width,
                           int channels, bool decoded) {
  normalize_tiles<<<blocks(count), 256>>>(sum, coverage, count, size_t(height) * width, channels,
                                          decoded);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}
}
