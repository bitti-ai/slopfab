// Quantised linear layers. See the header for the contract; this file is the
// arithmetic and the reasons behind it.
//
// The whole file is organised around one decision: a stored weight is turned
// into a dense bf16 (or fp32) matrix in workspace and then multiplied by an
// ordinary cuBLAS GEMM. That costs one extra read+write of the weight per call
// and buys exact agreement with a reference that dequantises first (spec 8.2).
// Native fp8/int8 GEMM will slot in behind `set_native` later.

#include "vidfab/cuda/linear.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "vidfab/cuda/device.h"
#include "vidfab/cuda/gemm.cuh"
#include "vidfab/cuda/nvfp4_gemm.cuh"

namespace vidfab::cuda {
namespace {

constexpr int kThreads = 256;

inline int grid_1d(size_t n, int block) { return static_cast<int>((n + block - 1) / block); }

inline size_t align_up(size_t n) { return (n + 255) / 256 * 256; }

// --- float8 E4M3 ------------------------------------------------------------
//
// Transcribed from `vidfab::f8_e4m3_to_f32` in dtype.h, including the unsigned
// wraparound in the subnormal loop, because the two are compared on all 256
// bit patterns in test_nn_kernels.cu. Do not "simplify" one without the other.
__device__ inline float f8_e4m3_to_f32_dev(uint8_t v) {
  const uint32_t sign = static_cast<uint32_t>(v & 0x80u) << 24;
  const uint32_t exp = (v >> 3) & 0x0Fu;
  const uint32_t mant = v & 0x07u;

  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      uint32_t e = 1;
      uint32_t m = mant;
      while ((m & 0x08u) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x07u;
      bits = sign | ((e + 127u - 7u) << 23) | (m << 20);
    }
  } else if (exp == 0x0F && mant == 0x07) {
    bits = sign | 0x7FC00000u;  // e4m3fn has no infinities; 0x7F/0xFF are NaN
  } else {
    bits = sign | ((exp + 127u - 7u) << 23) | (mant << 20);
  }
  return __uint_as_float(bits);
}

// Round-to-nearest-even encode, saturating at ±448. `rintf` supplies the
// tie-to-even; the only subtlety is that the subnormal binade (exponent -6)
// encodes its mantissa without the implicit leading one, so the two cases
// differ by whether the rounded significand reached 8.
__device__ inline uint8_t f32_to_f8_e4m3_dev(float f) {
  const uint32_t bits = __float_as_uint(f);
  const uint8_t sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
  if (isnan(f)) return static_cast<uint8_t>(sign | 0x7Fu);

  float a = fabsf(f);
  if (a > 448.0f) a = 448.0f;
  if (a == 0.0f) return sign;

  int e = ilogbf(a);
  if (e < -6) e = -6;  // below the smallest normal the exponent stops moving
  int q = static_cast<int>(rintf(a * ldexpf(1.0f, 3 - e)));
  if (q >= 16) {  // 15.5 rounded up; roll into the next binade
    q >>= 1;
    ++e;
  }
  if (e > 8 || (e == 8 && q > 14)) {  // 480 would be the NaN encoding
    e = 8;
    q = 14;
  }
  if (q < 8) return static_cast<uint8_t>(sign | q);  // subnormal, exponent 0
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(e + 7) << 3) | (q - 8));
}

// --- float4 E2M1 ------------------------------------------------------------
//
// Sixteen representable values, which is why `f4_e2m1_to_f32` in dtype.h uses a
// table. A table is the wrong shape on the device: the index *is* the data, so
// a warp holding sixteen different nibbles serialises a __constant__ lookup
// into sixteen transactions on a kernel that is otherwise pure streaming.
// Building the float out of the bits is uniform and branch-light. The two are
// compared on all sixteen patterns in test_nn_kernels.cu.
//
// Exponent 0 holds only ±0 and ±0.5; above it the fp32 exponent is 127+e-1 and
// the single mantissa bit lands at the top of fp32's field.
__device__ inline float f4_e2m1_to_f32_dev(uint32_t nibble) {
  const uint32_t mag = nibble & 0x07u;
  uint32_t bits;
  if (mag == 0) {
    bits = 0u;
  } else if (mag == 1) {
    bits = 0x3F000000u;  // 0.5, the only subnormal
  } else {
    bits = ((126u + (mag >> 1)) << 23) | ((mag & 1u) << 22);
  }
  return __uint_as_float(bits | ((nibble & 0x08u) << 28));
}

// Where row `o`'s scale for contraction block `k` actually lives.
//
// The block scales are **not** row-major. Both shipped nvfp4 checkpoints write
// them in the 128x4 tile layout the block-scaled mma consumes:
//
//   tile = (o / 128) * (blocks_per_row / 4) + (k / 4)
//   idx  = tile * 512 + (o % 32) * 16 + ((o % 128) / 32) * 4 + (k % 4)
//
// Reading them row-major instead is silent: the output stays finite and
// correctly scaled, and its elementwise correlation against the fp8 build of
// the same model is 0.00003. docs/transformer_spec.md 8.6 records the
// measurement, the four alternative tilings that were ruled out, and the
// no-padding precondition this relies on.
//
// Tiles are row-major over `o`, which is the reason the qkv split in
// transformer.cpp may slice this array at a byte offset at all — and it only
// holds because each third is a whole number of 128-row tiles.
__device__ inline size_t nvfp4_scale_offset(int o, int k, int blocks_per_row) {
  const int tile = (o >> 7) * (blocks_per_row >> 2) + (k >> 2);
  return static_cast<size_t>(tile) * 512 +
         static_cast<size_t>((o & 31) * 16 + ((o & 127) >> 5) * 4 + (k & 3));
}

// --- packed bf16 store ------------------------------------------------------

__device__ inline void store8_bf16(__nv_bfloat16* p, const float* in) {
  uint4 raw;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(in[2 * i], in[2 * i + 1]);
  *reinterpret_cast<uint4*>(p) = raw;
}

// --- dequantisation kernels -------------------------------------------------

__global__ void dequant_f8_kernel(const uint8_t* __restrict__ src, const float* __restrict__ scale,
                                  __nv_bfloat16* __restrict__ dst, size_t packs) {
  const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= packs) return;
  const float s = *scale;
  const uint2 raw = reinterpret_cast<const uint2*>(src)[p];
  const uint8_t* b = reinterpret_cast<const uint8_t*>(&raw);
  float v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = f8_e4m3_to_f32_dev(b[i]) * s;
  store8_bf16(dst + p * 8, v);
}

__global__ void dequant_f8_scalar_kernel(const uint8_t* __restrict__ src,
                                         const float* __restrict__ scale,
                                         __nv_bfloat16* __restrict__ dst, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = __float2bfloat16(f8_e4m3_to_f32_dev(src[i]) * (*scale));
}

// Per-output-channel scale, despite the format tag reading "int8_tensorwise"
// (docs/convrot_notes.md). `in_features` divisible by 8 keeps a pack inside a
// single output channel.
__global__ void dequant_i8_kernel(const int8_t* __restrict__ src, const float* __restrict__ scale,
                                  __nv_bfloat16* __restrict__ dst, int in_features,
                                  size_t packs_total) {
  const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= packs_total) return;
  const int packs_per_row = in_features / 8;
  const float s = scale[p / packs_per_row];
  const uint2 raw = reinterpret_cast<const uint2*>(src)[p];
  const int8_t* b = reinterpret_cast<const int8_t*>(&raw);
  float v[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = static_cast<float>(b[i]) * s;
  store8_bf16(dst + p * 8, v);
}

__global__ void dequant_i8_scalar_kernel(const int8_t* __restrict__ src,
                                         const float* __restrict__ scale,
                                         __nv_bfloat16* __restrict__ dst, int in_features,
                                         size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = __float2bfloat16(static_cast<float>(src[i]) * scale[i / in_features]);
}

// Eight elements per thread: one 32-bit load of four packed bytes, one 16-byte
// store, and both nibbles of every pair sit inside the same 16-wide scale
// block, so the scale is fetched once per thread and its block index is a shift
// rather than a division. Rows ride grid.y with a stride loop so that no index
// here needs a 64-bit divide — the same reason add_bias_kernel takes its column
// off the grid.
//
// `global_scale` multiplies. That direction is not a convention we adopted: at
// every sampled tensor `6 * 448 * weight_scale_2` reproduces the amax of the
// fp8 build of the same model to four decimal places (qkv_proj block 0:
// 6 * 448 * 0.0013580322 = 3.6408 against a measured 3.64063), while dividing
// lands around 1e6.
__global__ void dequant_nvfp4_kernel(const uint8_t* __restrict__ src,
                                     const uint8_t* __restrict__ block_scale, float global_scale,
                                     __nv_bfloat16* __restrict__ dst, int out_features,
                                     int packs_per_row, int blocks_per_row) {
  const int pack = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (pack >= packs_per_row) return;
  const size_t row_bytes = static_cast<size_t>(packs_per_row) * 4;
  const size_t row_elems = static_cast<size_t>(packs_per_row) * 8;

  for (int o = static_cast<int>(blockIdx.y); o < out_features;
       o += static_cast<int>(gridDim.y)) {
    const uint32_t raw = *reinterpret_cast<const uint32_t*>(
        src + static_cast<size_t>(o) * row_bytes + static_cast<size_t>(pack) * 4);
    const float s =
        f8_e4m3_to_f32_dev(block_scale[nvfp4_scale_offset(o, pack >> 1, blocks_per_row)]) *
        global_scale;
    float v[8];
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const uint32_t byte = (raw >> (8 * b)) & 0xFFu;
      // The even-indexed element is the **high** nibble. Swapping these two
      // lines leaves the value histogram untouched and the output finite and
      // well scaled; it drops elementwise correlation against the fp8 build
      // from 0.995 to 0.00003 and nothing else moves (spec 8.6).
      v[2 * b] = f4_e2m1_to_f32_dev(byte >> 4) * s;
      v[2 * b + 1] = f4_e2m1_to_f32_dev(byte & 0x0Fu) * s;
    }
    store8_bf16(dst + static_cast<size_t>(o) * row_elems + static_cast<size_t>(pack) * 8, v);
  }
}

__global__ void dequant_nf4_kernel(const uint8_t* __restrict__ src,
                                   const uint8_t* __restrict__ absmax,
                                   const float* __restrict__ quant_map,
                                   const float* __restrict__ nested_quant_map,
                                   const float* __restrict__ nested_absmax, int block_size,
                                   int nested_block_size, float nested_offset,
                                   __nv_bfloat16* __restrict__ dst, size_t n) {
  const size_t byte = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t even = byte * 2;
  if (even >= n) return;
  const size_t scale_index = even / static_cast<size_t>(block_size);
  const float scale = nested_quant_map[absmax[scale_index]] *
                          nested_absmax[scale_index / static_cast<size_t>(nested_block_size)] +
                      nested_offset;
  const uint8_t packed = src[byte];
  dst[even] = __float2bfloat16(quant_map[packed >> 4] * scale);
  if (even + 1 < n) dst[even + 1] = __float2bfloat16(quant_map[packed & 0x0f] * scale);
}

__global__ void quantize_f8_kernel(const __nv_bfloat16* __restrict__ src, float inv_scale,
                                   uint8_t* __restrict__ dst, size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = f32_to_f8_e4m3_dev(__bfloat162float(src[i]) * inv_scale);
}

__global__ void widen_bf16_kernel(const __nv_bfloat16* __restrict__ src, float* __restrict__ dst,
                                  size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __bfloat162float(src[i]);
}

__global__ void narrow_bf16_kernel(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst,
                                   size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2bfloat16(src[i]);
}

// f16 -> bf16 goes through fp32; both are lossless-then-rounded, and the
// checkpoint's f16 tensors are small (AdaLN tables).
__global__ void f16_to_bf16_kernel(const __half* __restrict__ src, __nv_bfloat16* __restrict__ dst,
                                   size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2bfloat16(__half2float(src[i]));
}

__global__ void f32_to_bf16_kernel(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst,
                                   size_t n) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2bfloat16(src[i]);
}

// --- ConvRot ----------------------------------------------------------------
//
// H = kron(h4,h4,h4,h4)/16 with h4 the *regular* Hadamard matrix. Four radix-4
// butterfly stages at strides 1, 4, 16, 64 realise the Kronecker product, one
// stage per base-4 digit of the index; the single 1/sqrt(group) at the end is
// the normalisation that makes H orthogonal.
//
// A Walsh-Hadamard transform is the wrong matrix here and produces relative
// error 1.4, not a crash (docs/convrot_notes.md).
//
// Several groups per block, `group/4` threads each: a thread owns one butterfly
// and its group lives in shared memory across the four stages. One group per
// block would be 64 threads — two warps — for the 256-wide group the quantiser
// uses, which wastes most of a launch.
//
// **Shared memory is padded one word every four.** Stage 0 has thread t reading
// `buf[4t]`, so without padding 8 threads of every warp land in the same bank:
// an 8-way conflict on the first and heaviest stage. Mapping element i to
// `i + i/4` makes stage 0 and stage 1 conflict-free (5 and 32 are coprime) and
// leaves stages 2 and 3 at 2-way. The padding also has to be applied to the
// group base, hence `padded_group` rather than `group`.

template <typename T>
__device__ inline float load_as_f32(const T* p);
template <>
__device__ inline float load_as_f32<float>(const float* p) {
  return *p;
}
template <>
__device__ inline float load_as_f32<__nv_bfloat16>(const __nv_bfloat16* p) {
  return __bfloat162float(*p);
}

template <typename T>
__device__ inline void store_from_f32(T* p, float v);
template <>
__device__ inline void store_from_f32<float>(float* p, float v) {
  *p = v;
}
template <>
__device__ inline void store_from_f32<__nv_bfloat16>(__nv_bfloat16* p, float v) {
  *p = __float2bfloat16(v);
}

// Element i of a group sits at word `i + i/4` of that group's slice.
__device__ inline int convrot_pad(int i) { return i + (i >> 2); }

template <typename T>
__global__ void convrot_kernel(const T* __restrict__ in, T* __restrict__ out, int group,
                               int stages, float norm, int groups_per_block, size_t total_groups) {
  extern __shared__ float shared[];
  const int quarter = group / 4;
  const int local = static_cast<int>(threadIdx.x) / quarter;
  const int t = static_cast<int>(threadIdx.x) - local * quarter;
  const size_t g = static_cast<size_t>(blockIdx.x) * groups_per_block + local;

  float* buf = shared + static_cast<size_t>(local) * (group + group / 4);
  const size_t base = g * group;

  // A tail block has groups past the end. Those threads must not read or write
  // global memory, but they must still reach every barrier below, so they run
  // the butterfly over zeros in their own slice rather than returning.
  const bool live = g < total_groups;
  for (int i = t; i < group; i += quarter) {
    buf[convrot_pad(i)] = live ? load_as_f32(in + base + i) : 0.0f;
  }
  __syncthreads();

  // Exactly `quarter` threads per group, one butterfly each — the launcher
  // enforces it. Every barrier below is therefore hit by the whole block, which
  // a loop over butterflies would not guarantee.
  int stride = 1;
  for (int s = 0; s < stages; ++s) {
    const int i = (t / stride) * 4 * stride + (t % stride);
    const int i0 = convrot_pad(i);
    const int i1 = convrot_pad(i + stride);
    const int i2 = convrot_pad(i + 2 * stride);
    const int i3 = convrot_pad(i + 3 * stride);
    const float a = buf[i0];
    const float b = buf[i1];
    const float c = buf[i2];
    const float d = buf[i3];
    __syncthreads();  // every lane must have read before any write lands
    buf[i0] = a + b + c - d;
    buf[i1] = a + b - c + d;
    buf[i2] = a - b + c + d;
    buf[i3] = -a + b + c + d;
    __syncthreads();
    stride *= 4;
  }

  if (!live) return;
  for (int i = t; i < group; i += quarter) {
    store_from_f32(out + base + i, buf[convrot_pad(i)] * norm);
  }
}

// --- bias -------------------------------------------------------------------

// The column index comes off the grid, not out of a modulo. A flat `i % cols`
// on a `size_t` compiles to a 64-bit integer division — twenty-odd
// instructions — per element, on a kernel that is otherwise pure streaming, and
// qkv_proj runs it over 811M elements.
//
// Rows on grid.x and columns on grid.y, matching the row-wise kernels in
// nn_kernels.cu: gridDim.y is capped at 65535 and a 768p/10s request packs
// ~73.4k rows, so rows must not go on y.
template <typename YT, typename BT>
__global__ void add_bias_kernel(YT* __restrict__ y, const BT* __restrict__ bias, int cols) {
  const int col = static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x);
  if (col >= cols) return;
  const size_t i = static_cast<size_t>(blockIdx.x) * cols + col;
  const float b = load_as_f32(bias + col);
  store_from_f32(y + i, load_as_f32(y + i) + b);
}

// --- AWQ activation scale ---------------------------------------------------

// dst[r, i] = src[r, i] * scale[i]. Same grid shape and the same reason as
// add_bias_kernel: the column comes off the grid rather than a modulo, and rows
// go on x because a packed sequence runs to tens of thousands of them while
// gridDim.y stops at 65535.
template <typename T>
__global__ void pre_quant_scale_kernel(const T* __restrict__ src,
                                       const __nv_bfloat16* __restrict__ scale,
                                       T* __restrict__ dst, int dim) {
  const int col = static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x);
  if (col >= dim) return;
  const size_t i = static_cast<size_t>(blockIdx.x) * dim + col;
  store_from_f32(dst + i, load_as_f32(src + i) * __bfloat162float(scale[col]));
}

template <typename T>
void launch_pre_quant_scale_impl(const T* src, const __nv_bfloat16* scale, T* dst, int rows,
                                 int dim, cudaStream_t stream) {
  if (rows <= 0 || dim <= 0) return;
  if (scale == nullptr) throw std::runtime_error("pre_quant_scale: null scale");
  const dim3 grid(static_cast<unsigned>(rows),
                  static_cast<unsigned>(grid_1d(static_cast<size_t>(dim), kThreads)));
  pre_quant_scale_kernel<<<grid, kThreads, 0, stream>>>(src, scale, dst, dim);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

// --- helpers ----------------------------------------------------------------

int convrot_stages(int group) {
  int stages = 0;
  int n = 1;
  while (n < group) {
    n *= 4;
    ++stages;
  }
  if (n != group) {
    throw std::runtime_error("convrot: group size must be a power of four, got " +
                             std::to_string(group));
  }
  return stages;
}

bool convrot_applies(const QuantWeight& w) {
  return w.convrot && w.convrot_group > 0 && w.in_features % w.convrot_group == 0;
}

size_t element_bytes(QuantFormat f) {
  switch (f) {
    case QuantFormat::kF32:
      return 4;
    case QuantFormat::kF16:
    case QuantFormat::kBF16:
      return 2;
    case QuantFormat::kF8E4M3:
    case QuantFormat::kI8:
      return 1;
    case QuantFormat::kNVFP4:
      // Half a byte. Callers that need a size must go through `stored_bytes`,
      // which knows the packing; returning 0 or 1 here would be a silent
      // half-or-double on every offset computed from it.
      throw std::runtime_error("linear: nvfp4 has no whole-byte element size");
  }
  return 0;
}

void add_bias(void* y, bool y_is_f32, const QuantWeight& w, int rows, cudaStream_t stream) {
  if (!w.has_bias()) return;
  const int cols = w.out_features;
  const dim3 grid(static_cast<unsigned>(rows),
                  static_cast<unsigned>(grid_1d(static_cast<size_t>(cols), kThreads)));

  if (y_is_f32) {
    float* yf = static_cast<float*>(y);
    switch (w.bias_format) {
      case QuantFormat::kF32:
        add_bias_kernel<<<grid, kThreads, 0, stream>>>(yf, static_cast<const float*>(w.bias), cols);
        break;
      case QuantFormat::kBF16:
        add_bias_kernel<<<grid, kThreads, 0, stream>>>(
            yf, static_cast<const __nv_bfloat16*>(w.bias), cols);
        break;
      default:
        throw std::runtime_error("linear: unsupported bias format");
    }
  } else {
    __nv_bfloat16* yb = static_cast<__nv_bfloat16*>(y);
    switch (w.bias_format) {
      case QuantFormat::kF32:
        add_bias_kernel<<<grid, kThreads, 0, stream>>>(yb, static_cast<const float*>(w.bias), cols);
        break;
      case QuantFormat::kBF16:
        add_bias_kernel<<<grid, kThreads, 0, stream>>>(
            yb, static_cast<const __nv_bfloat16*>(w.bias), cols);
        break;
      default:
        throw std::runtime_error("linear: unsupported bias format");
    }
  }
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

// Returns a dense bf16 view of `w`, dequantising into `ws` when the stored
// format is not already bf16.
const __nv_bfloat16* materialise_bf16(const QuantWeight& w, Workspace& ws, cudaStream_t stream) {
  const size_t n = static_cast<size_t>(w.out_features) * w.in_features;
  if (w.format == QuantFormat::kBF16) {
    return static_cast<const __nv_bfloat16*>(w.data);
  }
  __nv_bfloat16* dst = ws.alloc_n<__nv_bfloat16>(n);
  switch (w.format) {
    case QuantFormat::kF8E4M3:
      if (w.weight_scale == nullptr) {
        throw std::runtime_error("linear: fp8 weight without weight_scale");
      }
      launch_dequant_f8e4m3(static_cast<const uint8_t*>(w.data), w.weight_scale, dst, n, stream);
      break;
    case QuantFormat::kNF4:
      if (w.nf4_absmax == nullptr || w.nf4_quant_map == nullptr ||
          w.nf4_nested_quant_map == nullptr || w.nf4_nested_absmax == nullptr) {
        throw std::runtime_error("linear: nf4 weight without complete double-quant state");
      }
      launch_dequant_nf4(static_cast<const uint8_t*>(w.data), w.nf4_absmax,
                         w.nf4_quant_map, w.nf4_nested_quant_map, w.nf4_nested_absmax,
                         w.nf4_block_size, w.nf4_nested_block_size, w.nf4_nested_offset, dst,
                         w.out_features, w.in_features, stream);
      break;
    case QuantFormat::kI8:
      if (w.weight_scale == nullptr) {
        throw std::runtime_error("linear: int8 weight without weight_scale");
      }
      launch_dequant_i8_per_channel(static_cast<const int8_t*>(w.data), w.weight_scale, dst,
                                    w.out_features, w.in_features, stream);
      break;
    case QuantFormat::kNVFP4:
      if (w.block_scale == nullptr) {
        throw std::runtime_error("linear: nvfp4 weight without block_scale");
      }
      launch_dequant_nvfp4(static_cast<const uint8_t*>(w.data), w.block_scale, w.global_scale, dst,
                           w.out_features, w.in_features, stream);
      break;
    case QuantFormat::kF16:
      f16_to_bf16_kernel<<<grid_1d(n, kThreads), kThreads, 0, stream>>>(
          static_cast<const __half*>(w.data), dst, n);
      VIDFAB_CUDA_CHECK(cudaGetLastError());
      break;
    case QuantFormat::kF32:
      f32_to_bf16_kernel<<<grid_1d(n, kThreads), kThreads, 0, stream>>>(
          static_cast<const float*>(w.data), dst, n);
      VIDFAB_CUDA_CHECK(cudaGetLastError());
      break;
    default:
      throw std::runtime_error("linear: unhandled weight format");
  }
  return dst;
}

}  // namespace

// --- QuantWeight ------------------------------------------------------------

size_t QuantWeight::stored_bytes() const {
  const size_t n = static_cast<size_t>(out_features) * in_features;
  // Two nibbles per byte. Both shipped 4-bit formats place the even element in
  // the high nibble; their dequantisers pin that convention independently.
  if (format == QuantFormat::kNVFP4 || format == QuantFormat::kNF4) return (n + 1) / 2;
  return element_bytes(format) * n;
}

size_t linear_workspace_bytes(const QuantWeight& w, int rows, ComputeType compute) {
  const size_t weights = static_cast<size_t>(w.out_features) * w.in_features;
  const size_t act = static_cast<size_t>(rows) * w.in_features;
  size_t total = 0;

  if (compute == ComputeType::kF32) {
    // fp32 compute needs an fp32 copy, and anything not already fp32 reaches it
    // through the bf16 dequantiser, so both buffers can be live at once.
    if (w.format != QuantFormat::kF32) {
      total += align_up(weights * sizeof(__nv_bfloat16));
      total += align_up(weights * sizeof(float));
    }
    // The AWQ scale writes a scaled copy of the activation, which ConvRot then
    // reads and rotates into a second copy, so on a layer with both the two
    // buffers are live at once and add rather than overlap.
    if (w.pre_quant_scale != nullptr) total += align_up(act * sizeof(float));
    if (convrot_applies(w)) total += align_up(act * sizeof(float));
  } else {
    if (w.format != QuantFormat::kBF16) total += align_up(weights * sizeof(__nv_bfloat16));
    if (w.pre_quant_scale != nullptr) total += align_up(act * sizeof(__nv_bfloat16));
    if (convrot_applies(w)) total += align_up(act * sizeof(__nv_bfloat16));
  }

  // The native nvfp4 GEMM carves a quantised copy of the activation instead of
  // a dequantised copy of the weight, so the two are alternatives and this is a
  // max rather than a sum. Sized unconditionally because this function is not
  // told whether `set_native` is on, and because the difference only ever
  // matters for a weight the native path could take at all.
  //
  // At every production shape the dequantised weight is the larger of the two,
  // so this changes nothing today. It is here because "the other buffer happens
  // to be bigger" is a coincidence, not an invariant, and a caller with many
  // rows and few output features would otherwise throw from Workspace::alloc.
  if (w.format == QuantFormat::kNVFP4 && compute == ComputeType::kBF16 &&
      nvfp4_gemm_supported(w.out_features, w.in_features)) {
    total = std::max(total, nvfp4_gemm_workspace_bytes(rows, w.in_features));
  }
  return total;
}

// --- LinearRunner -----------------------------------------------------------

void LinearRunner::init(cublasHandle_t handle, cudaStream_t stream) {
  handle_ = handle;
  stream_ = stream;
  VIDFAB_CUBLAS_CHECK(cublasSetStream(handle_, stream_));
}

void LinearRunner::forward(const QuantWeight& w, const __nv_bfloat16* x, int rows,
                           __nv_bfloat16* y, Workspace& ws) {
  if (handle_ == nullptr) throw std::runtime_error("LinearRunner::forward: init() not called");
  if (rows <= 0) return;

  // Native low-precision GEMM is not wired up yet — both settings dequantise.
  // The guard is written now so the invariant survives that change: a weight
  // the checkpoint flagged full_precision_matrix_mult, or an fp8 weight that
  // ships no input_scale (the same statement in the older files' vocabulary),
  // must never reach a low-precision GEMM (spec 8.2).
  const bool native_path = native_ && !w.full_precision &&
                           ((w.format == QuantFormat::kF8E4M3 && w.input_scale != 0.0f) ||
                            w.format == QuantFormat::kNVFP4);

  // Native nvfp4. The extra conditions are not a weakening of the guard above:
  // ConvRot never coincides with an nvfp4 transformer weight and the rotation
  // belongs to the activation, `pre_quant_scale` is an AWQ text-encoder field
  // that `full_precision` already excludes, and `nvfp4_gemm_supported` is the
  // block-scale swizzle's no-padding precondition. Each falls through to the
  // reference path, which is always correct, never to something approximate.
  if (native_path && w.format == QuantFormat::kNVFP4 && w.block_scale != nullptr &&
      !convrot_applies(w) && w.pre_quant_scale == nullptr &&
      nvfp4_gemm_supported(w.out_features, w.in_features)) {
    Workspace::Scope native_scope(ws);
    nvfp4_gemm_forward(x, static_cast<const uint8_t*>(w.data), w.block_scale, w.global_scale, y,
                       rows, w.out_features, w.in_features, ws, stream_);
    add_bias(y, /*y_is_f32=*/false, w, rows, stream_);
    return;
  }

  // The arena is only rewound, never freed, and everything below is issued on
  // one stream, so releasing the cursor at return cannot race the GEMM.
  Workspace::Scope scope(ws);

  const __nv_bfloat16* weight = materialise_bf16(w, ws, stream_);

  // AWQ scales the activation per input channel ahead of everything else. A
  // null pointer means the quantiser folded the scale into the preceding norm's
  // weight, so it is "already accounted for" rather than "unknown" — check the
  // tensor, never the layer's name. nvfp4 weights are never ConvRot, so the two
  // never actually compose; the order below is the one that would be right if
  // they ever did.
  const __nv_bfloat16* xin = x;
  if (w.pre_quant_scale != nullptr) {
    __nv_bfloat16* xs = ws.alloc_n<__nv_bfloat16>(static_cast<size_t>(rows) * w.in_features);
    launch_pre_quant_scale(xin, w.pre_quant_scale, xs, rows, w.in_features, stream_);
    xin = xs;
  }
  if (convrot_applies(w)) {
    __nv_bfloat16* xr = ws.alloc_n<__nv_bfloat16>(static_cast<size_t>(rows) * w.in_features);
    launch_convrot(xin, xr, rows, w.in_features, w.convrot_group, stream_);
    xin = xr;
  }

  // Row-major y[rows,out] = x[rows,in] @ W[out,in]^T. Column-major sees W as
  // [in,out] and x as [in,rows]; op_T on W then gives [out,in] * [in,rows].
  const float alpha = 1.0f;
  const float beta = 0.0f;
  VIDFAB_CUBLAS_CHECK(cublasGemmEx(handle_, CUBLAS_OP_T, CUBLAS_OP_N, w.out_features, rows,
                                   w.in_features, &alpha, weight, CUDA_R_16BF, w.in_features, xin,
                                   CUDA_R_16BF, w.in_features, &beta, y, CUDA_R_16BF,
                                   w.out_features, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

  add_bias(y, /*y_is_f32=*/false, w, rows, stream_);
}

void LinearRunner::forward_f32(const QuantWeight& w, const float* x, int rows, float* y,
                               Workspace& ws) {
  if (handle_ == nullptr) throw std::runtime_error("LinearRunner::forward_f32: init() not called");
  if (rows <= 0) return;

  Workspace::Scope scope(ws);
  const size_t n = static_cast<size_t>(w.out_features) * w.in_features;

  const float* weight = nullptr;
  if (w.format == QuantFormat::kF32) {
    weight = static_cast<const float*>(w.data);
  } else {
    const __nv_bfloat16* narrow = materialise_bf16(w, ws, stream_);
    float* wide = ws.alloc_n<float>(n);
    launch_widen_bf16(narrow, wide, n, stream_);
    weight = wide;
  }

  const float* xin = x;
  if (w.pre_quant_scale != nullptr) {
    float* xs = ws.alloc_n<float>(static_cast<size_t>(rows) * w.in_features);
    launch_pre_quant_scale_impl(xin, w.pre_quant_scale, xs, rows, w.in_features, stream_);
    xin = xs;
  }
  if (convrot_applies(w)) {
    float* xr = ws.alloc_n<float>(static_cast<size_t>(rows) * w.in_features);
    launch_convrot_f32(xin, xr, rows, w.in_features, w.convrot_group, stream_);
    xin = xr;
  }

  gemm_nt(handle_, xin, weight, y, rows, w.out_features, w.in_features);
  add_bias(y, /*y_is_f32=*/true, w, rows, stream_);
}

// --- ConvRot launchers ------------------------------------------------------

namespace {

template <typename T>
void launch_convrot_impl(const T* in, T* out, int rows, int dim, int group, cudaStream_t stream) {
  if (rows <= 0 || dim <= 0) return;
  if (dim % group != 0) {
    throw std::runtime_error("convrot: dim " + std::to_string(dim) + " is not a multiple of group " +
                             std::to_string(group));
  }
  const int stages = convrot_stages(group);
  if (group / 4 > 1024) {
    throw std::runtime_error("convrot: group " + std::to_string(group) +
                             " needs more than 1024 threads per block");
  }
  // sqrtf of a power of four is exact, so this is exactly 1/16 at group 256 —
  // which is why the quantiser insists on a power-of-four group size.
  const float norm = 1.0f / std::sqrt(static_cast<float>(group));
  const size_t groups = static_cast<size_t>(rows) * (dim / group);
  const int quarter = group / 4;
  // Aim for a 256-thread block; a group wider than 1024 butterflies has already
  // been rejected above.
  const int groups_per_block = std::max(1, 256 / quarter);
  const int threads = groups_per_block * quarter;
  const int shared =
      groups_per_block * (group + group / 4) * static_cast<int>(sizeof(float));
  const int blocks = static_cast<int>((groups + groups_per_block - 1) / groups_per_block);
  convrot_kernel<<<blocks, threads, shared, stream>>>(in, out, group, stages, norm,
                                                      groups_per_block, groups);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

}  // namespace

void launch_convrot(const __nv_bfloat16* in, __nv_bfloat16* out, int rows, int dim, int group,
                    cudaStream_t stream) {
  launch_convrot_impl(in, out, rows, dim, group, stream);
}

void launch_convrot_f32(const float* in, float* out, int rows, int dim, int group,
                        cudaStream_t stream) {
  launch_convrot_impl(in, out, rows, dim, group, stream);
}

// --- dequantisation launchers -----------------------------------------------

void launch_dequant_f8e4m3(const uint8_t* src, const float* scale, __nv_bfloat16* dst, size_t n,
                           cudaStream_t stream) {
  if (n == 0) return;
  if (n % 8 == 0) {
    const size_t packs = n / 8;
    dequant_f8_kernel<<<grid_1d(packs, kThreads), kThreads, 0, stream>>>(src, scale, dst, packs);
  } else {
    dequant_f8_scalar_kernel<<<grid_1d(n, kThreads), kThreads, 0, stream>>>(src, scale, dst, n);
  }
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_dequant_i8_per_channel(const int8_t* src, const float* scale, __nv_bfloat16* dst,
                                   int out_features, int in_features, cudaStream_t stream) {
  const size_t n = static_cast<size_t>(out_features) * in_features;
  if (n == 0) return;
  if (in_features % 8 == 0) {
    const size_t packs = n / 8;
    dequant_i8_kernel<<<grid_1d(packs, kThreads), kThreads, 0, stream>>>(src, scale, dst,
                                                                        in_features, packs);
  } else {
    dequant_i8_scalar_kernel<<<grid_1d(n, kThreads), kThreads, 0, stream>>>(src, scale, dst,
                                                                           in_features, n);
  }
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_dequant_nvfp4(const uint8_t* src, const uint8_t* block_scale, float global_scale,
                          __nv_bfloat16* dst, int out_features, int in_features,
                          cudaStream_t stream) {
  if (out_features <= 0 || in_features <= 0) return;
  if (block_scale == nullptr) {
    throw std::runtime_error("launch_dequant_nvfp4: block_scale is null");
  }
  // The 128x4 scale tiling has no padded case in either shipped checkpoint —
  // all 200 quantised linears of the transformer divide exactly, and so do the
  // encoder's — and a padded one would need a different address map rather than
  // a rounded-up bound. Refuse instead of computing a plausible wrong offset.
  if (out_features % 128 != 0 || in_features % (4 * kNVFP4BlockSize) != 0) {
    throw std::runtime_error(
        "launch_dequant_nvfp4: " + std::to_string(out_features) + "x" +
        std::to_string(in_features) +
        " does not tile the swizzled block-scale layout, which requires out_features % 128 == 0 "
        "and in_features % 64 == 0");
  }

  const int packs_per_row = in_features / 8;
  const int blocks_per_row = in_features / kNVFP4BlockSize;
  const dim3 grid(static_cast<unsigned>(grid_1d(static_cast<size_t>(packs_per_row), kThreads)),
                  static_cast<unsigned>(std::min(out_features, 65535)));
  dequant_nvfp4_kernel<<<grid, kThreads, 0, stream>>>(src, block_scale, global_scale, dst,
                                                      out_features, packs_per_row, blocks_per_row);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_dequant_nf4(const uint8_t* src, const uint8_t* absmax, const float* quant_map,
                        const float* nested_quant_map, const float* nested_absmax,
                        int block_size, int nested_block_size, float nested_offset,
                        __nv_bfloat16* dst, int out_features, int in_features,
                        cudaStream_t stream) {
  if (src == nullptr || absmax == nullptr || quant_map == nullptr || nested_quant_map == nullptr ||
      nested_absmax == nullptr || dst == nullptr) {
    throw std::runtime_error("launch_dequant_nf4: null pointer");
  }
  if (block_size <= 0 || nested_block_size <= 0) {
    throw std::runtime_error("launch_dequant_nf4: block sizes must be positive");
  }
  const size_t n = static_cast<size_t>(out_features) * in_features;
  dequant_nf4_kernel<<<grid_1d((n + 1) / 2, kThreads), kThreads, 0, stream>>>(
      src, absmax, quant_map, nested_quant_map, nested_absmax, block_size, nested_block_size,
      nested_offset, dst, n);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_pre_quant_scale(const __nv_bfloat16* src, const __nv_bfloat16* scale,
                            __nv_bfloat16* dst, int rows, int dim, cudaStream_t stream) {
  launch_pre_quant_scale_impl(src, scale, dst, rows, dim, stream);
}

void launch_quantize_f8e4m3(const __nv_bfloat16* src, float input_scale, uint8_t* dst, size_t n,
                            cudaStream_t stream) {
  if (n == 0) return;
  if (input_scale == 0.0f) {
    // A zero input_scale is the checkpoint's way of saying "this layer has no
    // activation calibration"; quantising anyway would divide by zero.
    throw std::runtime_error("launch_quantize_f8e4m3: input_scale is zero, which marks a layer "
                             "that must run at full precision");
  }
  quantize_f8_kernel<<<grid_1d(n, kThreads), kThreads, 0, stream>>>(src, 1.0f / input_scale, dst,
                                                                   n);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_widen_bf16(const __nv_bfloat16* src, float* dst, size_t n, cudaStream_t stream) {
  if (n == 0) return;
  widen_bf16_kernel<<<grid_1d(n, kThreads), kThreads, 0, stream>>>(src, dst, n);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_narrow_to_bf16(const float* src, __nv_bfloat16* dst, size_t n, cudaStream_t stream) {
  if (n == 0) return;
  narrow_bf16_kernel<<<grid_1d(n, kThreads), kThreads, 0, stream>>>(src, dst, n);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

}  // namespace vidfab::cuda
