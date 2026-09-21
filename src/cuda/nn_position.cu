#include "nn_internal.cuh"

namespace slopfab::cuda {
using namespace nn_detail;

namespace {
// --- rotary -----------------------------------------------------------------
//
// MM-RoPE rotates the first 96 of 128 head channels, pairing j with j+48
// (spec 5.3). Channels 96..127 are copied untouched, which the kernel achieves
// by not writing them at all — the transform is in place.
//
// One warp per (row, head): 48 pairs, so each of 32 lanes handles one or two.

constexpr int kRopeHalf = 48;
constexpr int kRopeDim = 2 * kRopeHalf; // 96 rotary channels of 128

__device__ float rope_bf16_value(__nv_bfloat16 value) {
  const unsigned short bits = __bfloat16_as_ushort(value);
  return (bits & 0x7fffu) < 0x0080u ? __uint_as_float(static_cast<unsigned>(bits & 0x8000u) << 16u)
                                    : __bfloat162float(value);
}

__device__ float rope_float_value(float value) {
  const unsigned bits = __float_as_uint(value);
  return (bits & 0x7fffffffu) < 0x00800000u ? __uint_as_float(bits & 0x80000000u) : value;
}

__device__ __nv_bfloat16 rope_bf16_result(float value) {
  const __nv_bfloat16 rounded = __float2bfloat16(value);
  const unsigned short bits = __bfloat16_as_ushort(rounded);
  return (bits & 0x7fffu) < 0x0080u ? __ushort_as_bfloat16(bits & 0x8000u) : rounded;
}

// The portable RoPE domain canonicalizes a rounded subnormal multiply to
// signed zero before the high-half fused add. This makes the internal product
// independent of fp32 FTZ behavior while retaining the CUDA FMA sequence for
// every normal product. Exact zero operands still use FMA so IEEE signed-zero
// addition semantics are preserved.
__device__ float rope_fma_product(float a, float b, float base) {
  const float product = a * b;
  const unsigned magnitude = __float_as_uint(product) & 0x7fffffffu;
  const unsigned a_magnitude = __float_as_uint(a) & 0x7fffffffu;
  const unsigned b_magnitude = __float_as_uint(b) & 0x7fffffffu;
  if (magnitude < 0x00800000u && a_magnitude != 0u && b_magnitude != 0u) {
    return base;
  }
  return fmaf(a, b, base);
}

__global__ void rope_h3_kernel(__nv_bfloat16* __restrict__ x, const float* __restrict__ cos_tab,
                               const float* __restrict__ sin_tab, int rows, int heads,
                               int head_dim) {
  const int lane = threadIdx.x;
  const int head = blockIdx.y * blockDim.y + threadIdx.y;
  const int row = blockIdx.x;
  if (head >= heads || row >= rows)
    return;

  __nv_bfloat16* v = x + (static_cast<size_t>(row) * heads + head) * head_dim;
  const float* cos_row = cos_tab + static_cast<size_t>(row) * kRopeDim;
  const float* sin_row = sin_tab + static_cast<size_t>(row) * kRopeDim;

  for (int j = lane; j < kRopeHalf; j += blockDim.x) {
    const float c = rope_float_value(cos_row[j]);
    const float s = rope_float_value(sin_row[j]);
    const float lo = rope_bf16_value(v[j]);
    const float hi = rope_bf16_value(v[j + kRopeHalf]);
    const float left = rope_float_value(lo * c);
    const float right = rope_float_value(hi * s);
    const float base = rope_float_value(hi * c);
    v[j] = rope_bf16_result(left - right);
    v[j + kRopeHalf] = rope_bf16_result(rope_fma_product(lo, s, base));
  }
}

// GPT-NeoX rotary over the whole head dimension, pairing j with j + dim/2.
// cos/sin are `[rows, head_dim]`, i.e. the half-period duplicated, matching
// what the reference builds for the encoder.
__global__ void rope_neox_kernel(__nv_bfloat16* __restrict__ x, const float* __restrict__ cos_tab,
                                 const float* __restrict__ sin_tab, int rows, int heads,
                                 int head_dim) {
  const int lane = threadIdx.x;
  const int head = blockIdx.y * blockDim.y + threadIdx.y;
  const int row = blockIdx.x;
  if (head >= heads || row >= rows)
    return;

  const int half = head_dim / 2;
  __nv_bfloat16* v = x + (static_cast<size_t>(row) * heads + head) * head_dim;
  const float* cos_row = cos_tab + static_cast<size_t>(row) * head_dim;
  const float* sin_row = sin_tab + static_cast<size_t>(row) * head_dim;

  for (int j = lane; j < half; j += blockDim.x) {
    const float lo = rope_bf16_value(v[j]);
    const float hi = rope_bf16_value(v[j + half]);
    const float cos_lo = rope_float_value(cos_row[j]);
    const float sin_lo = rope_float_value(sin_row[j]);
    const float cos_hi = rope_float_value(cos_row[j + half]);
    const float sin_hi = rope_float_value(sin_row[j + half]);
    const float left = rope_float_value(lo * cos_lo);
    const float right = rope_float_value(hi * sin_lo);
    const float base = rope_float_value(hi * cos_hi);
    v[j] = rope_bf16_result(left - right);
    v[j + half] = rope_bf16_result(rope_fma_product(lo, sin_hi, base));
  }
}

} // namespace

void launch_rope_h3(__nv_bfloat16* x, const float* cos, const float* sin, int rows, int heads,
                    int head_dim, cudaStream_t stream) {
  require_positive(rows, heads, "launch_rope_h3");
  if (head_dim < kRopeDim) {
    throw std::runtime_error("launch_rope_h3: head_dim must be at least 96 rotary channels, got " +
                             std::to_string(head_dim));
  }
  // One warp per head, four heads per block.
  const dim3 block(kWarp, 4);
  const dim3 grid(rows, (heads + 3) / 4);
  rope_h3_kernel<<<grid, block, 0, stream>>>(x, cos, sin, rows, heads, head_dim);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_rope_neox(__nv_bfloat16* x, const float* cos, const float* sin, int rows, int heads,
                      int head_dim, cudaStream_t stream) {
  require_positive(rows, heads, "launch_rope_neox");
  if (head_dim <= 0 || head_dim % 2 != 0) {
    throw std::runtime_error("launch_rope_neox: head_dim must be even");
  }
  const dim3 block(kWarp, 4);
  const dim3 grid(rows, (heads + 3) / 4);
  rope_neox_kernel<<<grid, block, 0, stream>>>(x, cos, sin, rows, heads, head_dim);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

} // namespace slopfab::cuda
