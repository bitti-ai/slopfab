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

#include <cmath>
#include <stdexcept>
#include <string>

#include "vidfab/cuda/device.h"
#include "vidfab/cuda/gemm.cuh"

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
// One block per 256-wide group, `group/4` threads: each thread owns one
// butterfly and the group lives in shared memory across the four stages.

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

template <typename T>
__global__ void convrot_kernel(const T* __restrict__ in, T* __restrict__ out, int group,
                               int stages, float norm) {
  extern __shared__ float buf[];
  const size_t base = static_cast<size_t>(blockIdx.x) * group;

  for (int i = threadIdx.x; i < group; i += blockDim.x) buf[i] = load_as_f32(in + base + i);
  __syncthreads();

  // Exactly `quarter` threads, one butterfly each — the launcher enforces it.
  // Every barrier below is therefore hit by the whole block, which a loop over
  // butterflies would not guarantee.
  const int t = threadIdx.x;
  int stride = 1;
  for (int s = 0; s < stages; ++s) {
    const int i = (t / stride) * 4 * stride + (t % stride);
    const float a = buf[i];
    const float b = buf[i + stride];
    const float c = buf[i + 2 * stride];
    const float d = buf[i + 3 * stride];
    __syncthreads();  // every lane must have read before any write lands
    buf[i] = a + b + c - d;
    buf[i + stride] = a + b - c + d;
    buf[i + 2 * stride] = a - b + c + d;
    buf[i + 3 * stride] = -a + b + c + d;
    __syncthreads();
    stride *= 4;
  }

  for (int i = threadIdx.x; i < group; i += blockDim.x) {
    store_from_f32(out + base + i, buf[i] * norm);
  }
}

// --- bias -------------------------------------------------------------------

template <typename YT, typename BT>
__global__ void add_bias_kernel(YT* __restrict__ y, const BT* __restrict__ bias, int cols,
                                size_t total) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const float b = load_as_f32(bias + (i % cols));
  store_from_f32(y + i, load_as_f32(y + i) + b);
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
  }
  return 0;
}

void add_bias(void* y, bool y_is_f32, const QuantWeight& w, int rows, cudaStream_t stream) {
  if (!w.has_bias()) return;
  const size_t total = static_cast<size_t>(rows) * w.out_features;
  const int grid = grid_1d(total, kThreads);
  const int cols = w.out_features;

  if (y_is_f32) {
    float* yf = static_cast<float*>(y);
    switch (w.bias_format) {
      case QuantFormat::kF32:
        add_bias_kernel<<<grid, kThreads, 0, stream>>>(yf, static_cast<const float*>(w.bias), cols,
                                                       total);
        break;
      case QuantFormat::kBF16:
        add_bias_kernel<<<grid, kThreads, 0, stream>>>(
            yf, static_cast<const __nv_bfloat16*>(w.bias), cols, total);
        break;
      default:
        throw std::runtime_error("linear: unsupported bias format");
    }
  } else {
    __nv_bfloat16* yb = static_cast<__nv_bfloat16*>(y);
    switch (w.bias_format) {
      case QuantFormat::kF32:
        add_bias_kernel<<<grid, kThreads, 0, stream>>>(yb, static_cast<const float*>(w.bias), cols,
                                                       total);
        break;
      case QuantFormat::kBF16:
        add_bias_kernel<<<grid, kThreads, 0, stream>>>(
            yb, static_cast<const __nv_bfloat16*>(w.bias), cols, total);
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
    case QuantFormat::kI8:
      if (w.weight_scale == nullptr) {
        throw std::runtime_error("linear: int8 weight without weight_scale");
      }
      launch_dequant_i8_per_channel(static_cast<const int8_t*>(w.data), w.weight_scale, dst,
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
  return element_bytes(format) * static_cast<size_t>(out_features) * in_features;
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
    if (convrot_applies(w)) total += align_up(act * sizeof(float));
  } else {
    if (w.format != QuantFormat::kBF16) total += align_up(weights * sizeof(__nv_bfloat16));
    if (convrot_applies(w)) total += align_up(act * sizeof(__nv_bfloat16));
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
  // that ships no input_scale is flagged full_precision_matrix_mult and must
  // never reach an fp8 GEMM (spec 8.2).
  const bool native_path =
      native_ && w.format == QuantFormat::kF8E4M3 && w.input_scale != 0.0f;
  (void)native_path;

  // The arena is only rewound, never freed, and everything below is issued on
  // one stream, so releasing the cursor at return cannot race the GEMM.
  Workspace::Scope scope(ws);

  const __nv_bfloat16* weight = materialise_bf16(w, ws, stream_);

  const __nv_bfloat16* xin = x;
  if (convrot_applies(w)) {
    __nv_bfloat16* xr = ws.alloc_n<__nv_bfloat16>(static_cast<size_t>(rows) * w.in_features);
    launch_convrot(x, xr, rows, w.in_features, w.convrot_group, stream_);
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
  if (convrot_applies(w)) {
    float* xr = ws.alloc_n<float>(static_cast<size_t>(rows) * w.in_features);
    launch_convrot_f32(x, xr, rows, w.in_features, w.convrot_group, stream_);
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
  const int threads = group / 4;
  const int shared = group * static_cast<int>(sizeof(float));
  convrot_kernel<<<static_cast<int>(groups), threads, shared, stream>>>(in, out, group, stages,
                                                                       norm);
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
