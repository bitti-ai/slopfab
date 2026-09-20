#include "linear_internal.cuh"

namespace slopfab::cuda::linear_detail {
namespace {
// --- float8 E4M3 ------------------------------------------------------------
//
// Transcribed from `slopfab::f8_e4m3_to_f32` in dtype.h, including the unsigned
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

// The f16 counterpart, written as eight scalar `__float2half_rn` rather than
// four packed conversions on purpose: it replaces a kernel that used exactly
// that intrinsic, and the point of the rewrite is that the bits do not move.
__device__ inline void store8_half(__half* p, const float* in) {
  uint4 raw;
  __half* h = reinterpret_cast<__half*>(&raw);
#pragma unroll
  for (int i = 0; i < 8; ++i) h[i] = __float2half_rn(in[i]);
  *reinterpret_cast<uint4*>(p) = raw;
}

__device__ inline void store8(__nv_bfloat16* p, const float* in) { store8_bf16(p, in); }
__device__ inline void store8(__half* p, const float* in) { store8_half(p, in); }

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

// Eight elements per thread, matching `dequant_f8_kernel` and `dequant_i8_kernel`
// above: one 32-bit load of four packed bytes, one 16-byte store, one scale.
//
// The scalar form below moves two elements in two separate 2-byte stores and
// pays two runtime 64-bit integer divisions per thread to find its block. Both
// block sizes are powers of two in every checkpoint that ships (64 and 256), so
// the launcher hands down `log2` and the divisions become shifts; the shape and
// alignment preconditions that do not hold send the call to the scalar kernel
// instead of being papered over here.
//
// `quant_map` is staged in shared rather than `__constant__`, for the reason
// spelled out above `f4_e2m1_to_f32_dev`: the nibble *is* the index, so a warp
// holding sixteen distinct codes would serialise a constant-memory broadcast
// into sixteen transactions. Sixteen distinct shared words are sixteen distinct
// banks, which is one.
//
// `quant_map[i] * scale` keeps its order, so this is bit-identical to the
// scalar kernel element for element.
template <typename T>
__global__ void dequant_nf4_vec_kernel(const uint8_t* __restrict__ src,
                                       const uint8_t* __restrict__ absmax,
                                       const float* __restrict__ quant_map,
                                       const float* __restrict__ nested_quant_map,
                                       const float* __restrict__ nested_absmax, int block_shift,
                                       int nested_shift, float nested_offset,
                                       T* __restrict__ dst, size_t packs) {
  __shared__ float lut[16];
  if (threadIdx.x < 16) lut[threadIdx.x] = quant_map[threadIdx.x];
  __syncthreads();
  const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (p >= packs) return;
  const size_t even = p * 8;
  // One scale for all eight: `block_shift` is at least 3, so a pack never
  // straddles two blocks. The launcher checks that.
  const size_t scale_index = even >> block_shift;
  const float scale = nested_quant_map[absmax[scale_index]] *
                          nested_absmax[scale_index >> nested_shift] +
                      nested_offset;
  const uint32_t raw = reinterpret_cast<const uint32_t*>(src)[p];
  const uint8_t* b = reinterpret_cast<const uint8_t*>(&raw);
  float v[8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    v[2 * i] = lut[b[i] >> 4] * scale;
    v[2 * i + 1] = lut[b[i] & 0x0f] * scale;
  }
  store8(dst + even, v);
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

__global__ void dequant_nf4_f16_kernel(const uint8_t* __restrict__ src,
                                       const uint8_t* __restrict__ absmax,
                                       const float* __restrict__ quant_map,
                                       const float* __restrict__ nested_quant_map,
                                       const float* __restrict__ nested_absmax, int block_size,
                                       int nested_block_size, float nested_offset,
                                       __half* __restrict__ dst, size_t n) {
  const size_t byte = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const size_t even = byte * 2;
  if (even >= n) return;
  const size_t scale_index = even / static_cast<size_t>(block_size);
  const float scale = nested_quant_map[absmax[scale_index]] *
                          nested_absmax[scale_index / static_cast<size_t>(nested_block_size)] +
                      nested_offset;
  const uint8_t packed = src[byte];
  dst[even] = __float2half_rn(quant_map[packed >> 4] * scale);
  if (even + 1 < n) dst[even + 1] = __float2half_rn(quant_map[packed & 0x0f] * scale);
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
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
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

}  // namespace
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
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
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
      SLOPFAB_CUDA_CHECK(cudaGetLastError());
      break;
    case QuantFormat::kF32:
      f32_to_bf16_kernel<<<grid_1d(n, kThreads), kThreads, 0, stream>>>(
          static_cast<const float*>(w.data), dst, n);
      SLOPFAB_CUDA_CHECK(cudaGetLastError());
      break;
    default:
      throw std::runtime_error("linear: unhandled weight format");
  }
  return dst;
}

void pre_quant_scale_f32(const float* src, const __nv_bfloat16* scale, float* dst,
                          int rows, int dim, cudaStream_t stream) {
  launch_pre_quant_scale_impl(src, scale, dst, rows, dim, stream);
}
}  // namespace slopfab::cuda::linear_detail

namespace slopfab::cuda {
using namespace linear_detail;
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
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
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
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
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
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
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
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

namespace {

// log2 of a power of two, or -1. The vectorised NF4 kernel needs shifts rather
// than divisors and refuses to guess.
int exact_log2(int v) {
  if (v <= 0 || (v & (v - 1)) != 0) return -1;
  int s = 0;
  while ((1 << s) != v) ++s;
  return s;
}

// Whether the eight-wide NF4 kernel may run on this call.
//
// A pack is eight elements, so it must not straddle a block scale (block_size a
// multiple of eight, and a power of two so the index is a shift) and it must not
// run off the end (n a multiple of eight). The two loads it widens also need
// their natural alignment: `src` is sliced at a byte offset for the qkv thirds
// in transformer.cpp, and `dst` is whatever workspace the caller passed, so
// neither is assumed.
template <typename T>
bool nf4_vector_eligible(const uint8_t* src, const T* dst, size_t n, int block_shift,
                         int nested_shift) {
  return block_shift >= 3 && nested_shift >= 0 && n % 8 == 0 &&
         reinterpret_cast<uintptr_t>(src) % 4 == 0 && reinterpret_cast<uintptr_t>(dst) % 16 == 0;
}

}  // namespace

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
  const int block_shift = exact_log2(block_size);
  const int nested_shift = exact_log2(nested_block_size);
  if (nf4_vector_eligible(src, dst, n, block_shift, nested_shift)) {
    dequant_nf4_vec_kernel<<<grid_1d(n / 8, kThreads), kThreads, 0, stream>>>(
        src, absmax, quant_map, nested_quant_map, nested_absmax, block_shift, nested_shift,
        nested_offset, dst, n / 8);
  } else {
    dequant_nf4_kernel<<<grid_1d((n + 1) / 2, kThreads), kThreads, 0, stream>>>(
        src, absmax, quant_map, nested_quant_map, nested_absmax, block_size, nested_block_size,
        nested_offset, dst, n);
  }
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_dequant_nf4_f16(const uint8_t* src, const uint8_t* absmax, const float* quant_map,
                            const float* nested_quant_map, const float* nested_absmax,
                            int block_size, int nested_block_size, float nested_offset,
                            __half* dst, size_t n, cudaStream_t stream) {
  if (!src || !absmax || !quant_map || !nested_quant_map || !nested_absmax || !dst)
    throw std::runtime_error("launch_dequant_nf4_f16: null pointer");
  if (block_size <= 0 || nested_block_size <= 0)
    throw std::runtime_error("launch_dequant_nf4_f16: invalid block sizes");
  const int block_shift = exact_log2(block_size);
  const int nested_shift = exact_log2(nested_block_size);
  if (nf4_vector_eligible(src, dst, n, block_shift, nested_shift)) {
    dequant_nf4_vec_kernel<<<grid_1d(n / 8, kThreads), kThreads, 0, stream>>>(
        src, absmax, quant_map, nested_quant_map, nested_absmax, block_shift, nested_shift,
        nested_offset, dst, n / 8);
  } else {
    dequant_nf4_f16_kernel<<<grid_1d((n + 1) / 2, kThreads), kThreads, 0, stream>>>(
        src, absmax, quant_map, nested_quant_map, nested_absmax, block_size, nested_block_size,
        nested_offset, dst, n);
  }
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
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
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_widen_bf16(const __nv_bfloat16* src, float* dst, size_t n, cudaStream_t stream) {
  if (n == 0) return;
  widen_bf16_kernel<<<grid_1d(n, kThreads), kThreads, 0, stream>>>(src, dst, n);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

void launch_narrow_to_bf16(const float* src, __nv_bfloat16* dst, size_t n, cudaStream_t stream) {
  if (n == 0) return;
  narrow_bf16_kernel<<<grid_1d(n, kThreads), kThreads, 0, stream>>>(src, dst, n);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
}

const __nv_bfloat16* materialize_bf16_exact(const QuantWeight& weight,
                                            Workspace& workspace,
                                            cudaStream_t stream) {
  return materialise_bf16(weight, workspace, stream);
}

}  // namespace slopfab::cuda
