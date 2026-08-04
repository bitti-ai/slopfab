// Native block-scaled nvfp4 GEMM. See the header for the contract.
//
// Everything here rests on one instruction,
//
//   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
//       .f32.e2m1.e2m1.f32.ue4m3
//
// whose operand layout is pinned bit-exactly by
// tests/test_nn_kernels.cu::nvfp4_mma_operand_layout. Two facts from there are
// the whole reason this file can exist, and neither is inferable:
//
//   * **A-side scales.** Row r's four block scales are the four bytes of lane
//     `r < 8 ? 4r : 4(r-8)+1`. Lanes `4g+2` and `4g+3` supply nothing, so only
//     64 of the warp's 128 scale bytes are live.
//   * **B-side scales.** Column c's four block scales are the four bytes of
//     lane `4c`; lanes `4c+1..4c+3` supply nothing. Established the same way,
//     by doubling one (lane, byte) at a time and watching which output moved
//     -- 8 columns need 8 lanes, not 16, so it is *not* the A rule with the
//     row count halved.
//
// **Which operand is which.** `row.col` means A is M x K row-major and B is
// K x N column-major, i.e. N x K row-major. For `y = x W^T` that makes the
// *activation* the A operand and the *weight* the B operand, and B's N x K
// row-major form is precisely `[out_features, in_features]` as stored. No
// transpose, no repacking: the checkpoint's bytes are already B fragments.
//
// **Accumulation is fp32 and that is deliberate.** The reference path is a
// bf16 cuBLAS GEMM with CUBLAS_COMPUTE_32F, and the point of the tolerance
// check is to isolate the error of 4-bit *operands* from the error of the
// accumulator. Nothing here should be read as a precision compromise except
// the operand width itself.
//
// **What is not done.** No `cp.async` pipeline and no split-K. The k loop is a
// register-prefetch double buffer, which is enough to keep the tensor pipe fed
// at these shapes; splitting k would only pay at row counts far below what the
// transformer generates (it chunks at 8192).
//
// **What it costs numerically, measured rather than argued.** Against a
// reference that quantises the activation the same way, this GEMM agrees to
// 1.6e-3 rms relative -- bf16 output rounding, nothing more, on the shipped
// `blocks.0` weights as well as on synthetic ones. Against a *bf16*-activation
// reference it differs by about 9% rms, flat in K from 128 to 5376 and flat in
// output magnitude, with correlation 0.9955. That is the format, not the
// kernel: E2M1 has one mantissa bit, and a dot product cannot average the error
// away because signal and error both grow as sqrt(K).
// `nvfp4_activation_cost` measures it on every run and
// `nvfp4_gemm_exact_fp4_activations` is the control that separates the two.

#include "vidfab/cuda/nvfp4_gemm.cuh"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

#include "vidfab/cuda/device.h"

namespace vidfab::cuda {
namespace {

constexpr int kWarp = 32;

inline size_t align_up(size_t n) { return (n + 255) / 256 * 256; }

// --- activation quantisation -------------------------------------------------
//
// Per 16-element block along the contraction axis: `scale = amax / 6`, rounded
// to e4m3, and the elements divided by the *decoded* scale before rounding to
// e2m1. 6 is E2M1's largest magnitude, so amax/6 is the scale that uses the
// whole grid; dividing by the decoded scale rather than the fp32 one matters
// because the hardware will multiply by the decoded value, and using the other
// costs up to 6% of a block's range for nothing.
//
// Rounding the scale to nearest (rather than up) leaves the block maximum
// fractionally over 6 half the time; `cvt.rn.satfinite.e2m1x2.f32` saturates
// it to 6 rather than wrapping. Rounding the scale up instead would remove
// that clip at the cost of a systematic downward bias on every element of
// every block, which is the worse trade -- the clip touches one element per
// block and moves it by at most 3%.
//
// **No global activation scale.** NVIDIA's recipe divides the block scale by a
// per-tensor constant so the e4m3 scales sit mid-range. That needs a whole
// extra reduction pass over x, and it buys nothing here: these activations are
// post-RMSNorm and land at amax/6 ~ 0.1..1, against e4m3's [2^-9, 448]. A
// block small enough to flush the scale to zero is a block whose largest
// element is under 0.0117, which contributes nothing to a sum the row's other
// blocks dominate. Both ends are asserted by `nvfp4_activation_quant_range`.
//
// One lane per 8 elements, so a lane pair owns one block and the amax reduction
// is a single `shfl_xor` by 1. That pairing is what makes every access
// contiguous: 32 lanes read 512 consecutive bytes of bf16 and write 128
// consecutive bytes of nibbles.

__device__ inline float e4m3_decode(uint8_t v) {
  return __half2float(__nv_cvt_fp8_to_halfraw(v, __NV_E4M3));
}

__global__ __launch_bounds__(256) void quantize_act_kernel(const __nv_bfloat16* __restrict__ x,
                                                           uint8_t* __restrict__ packed,
                                                           uint8_t* __restrict__ scales,
                                                           size_t halves) {
  const size_t h = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  // Dead lanes still reach the shuffle below. `halves` is even and a block's
  // two halves are always both live or both dead, so a live lane never pairs
  // with a dead one and the zero it would contribute is never read anyway.
  const bool live = h < halves;

  float v[8];
  if (live) {
    const uint4 raw = *reinterpret_cast<const uint4*>(x + h * 8);
    const __nv_bfloat162* pair = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float2 f = __bfloat1622float2(pair[i]);
      v[i * 2] = f.x;
      v[i * 2 + 1] = f.y;
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = 0.0f;
  }

  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(v[i]));
  amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));

  const uint8_t s8 = __nv_cvt_float_to_fp8(amax * (1.0f / 6.0f), __NV_SATFINITE, __NV_E4M3);
  const float decoded = e4m3_decode(s8);
  // An all-zero block gives scale 0; 0 * 0 is the right answer, and the
  // reciprocal must not become an infinity that turns 0 into a NaN.
  const float inv = decoded > 0.0f ? 1.0f / decoded : 0.0f;

  uint32_t bytes = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const __nv_fp4x2_storage_t b = __nv_cvt_float2_to_fp4x2(
        make_float2(v[i * 2] * inv, v[i * 2 + 1] * inv), __NV_E2M1, cudaRoundNearest);
    bytes |= static_cast<uint32_t>(b) << (8 * i);
  }

  if (!live) return;
  *reinterpret_cast<uint32_t*>(packed + h * 4) = bytes;
  if ((threadIdx.x & 1) == 0) scales[h / 2] = s8;
}

// --- the GEMM ---------------------------------------------------------------
//
// Block tile 128 x 128 over (rows, out_features), 256 threads as eight warps in
// a 2 x 4 grid, so a warp owns 64 rows by 32 columns: four m-tiles by four
// n-tiles of the 16x8 accumulator, 16 mma per 64 contracted elements.
//
// **kBK is 128, two mma k-steps.** One step per staging round would halve the
// mma issued between barriers; four would push shared memory past the 48 KB
// that a static `__shared__` gets without an opt-in, for no further gain in
// arithmetic intensity.
//
// **Both operands stage into the same shared layout**, and that is the point:
// A and B differ only in how their bytes are *read* -- the weight's nibbles are
// swapped and its scales unswizzled on the way in -- so `fetch_tile` carries
// those two as template flags and everything downstream of the store is
// identical. The mma sees one layout, not two.
//
// **Measured, sm_120a, CUDA 13.0:** 126 registers, no spills, 45056 bytes of
// static shared, one barrier. Both limits land on the same number: 256 threads
// at 126 registers is 32256 of the SM's 65536, and 45056 bytes is under half of
// its 100 KB, so two blocks are resident -- 16 warps, four per scheduler.
// Occupancy is 25% and that is the intended operating point, as in
// attention.cu: what feeds a tensor pipe is independent instruction streams per
// scheduler, not warp slots filled.

constexpr int kBM = 128;
constexpr int kBN = 128;
constexpr int kBK = 128;
constexpr int kWarpsM = 2;
constexpr int kWarpsN = 4;
constexpr int kThreads = kWarpsM * kWarpsN * kWarp;  // 256
constexpr int kWM = kBM / kWarpsM;                   // 64 rows per warp
constexpr int kWN = kBN / kWarpsN;                   // 32 columns per warp
constexpr int kMTiles = kWM / 16;                    // 4
constexpr int kNTiles = kWN / 8;                     // 4
constexpr int kKSteps = kBK / 64;                    // 2 mma k-steps per tile

// A staged row is `kBK/2` = 64 bytes = 16 words of nibbles. The stride is
// padded to 20 rather than 16, and 20 rather than 17 or 18: a fragment load has
// lane `4g+t` reading word `row*S + t` with `row = base + g`, so the accesses
// are conflict-free exactly when `g*S mod 32` spreads the eight groups four
// banks apart, which happens for `S = 4u` with u odd. u=5 is the smallest that
// covers the 16 live words, and the four wasted words are 2 KB of the 45 KB
// budget -- cheaper than the register shuffling an XOR swizzle would need to
// keep the staging stores vectorised.
constexpr int kRowWords = 20;
constexpr int kUsedWords = kBK / 2 / 4;    // 16
constexpr int kScaleWords = kBK / 64;      // 2: one u32 of four block scales per k-step
constexpr int kU4PerRow = kUsedWords / 4;  // 4 uint4 loads per staged row

constexpr int kTileU4 = kBM * kU4PerRow;          // 512
constexpr int kU4PerThread = kTileU4 / kThreads;  // 2

constexpr int kStageWords = 2 * kBM * kRowWords + 2 * kBM * kScaleWords;  // 5632 -> 22 KB

static_assert(kBM == kBN, "the staging routines share one row count");
static_assert(kTileU4 % kThreads == 0, "the data staging must divide evenly across the block");
static_assert(kBM * kScaleWords == kThreads, "one scale word per thread, no loop");
static_assert(kRowWords % 4 == 0 && (kRowWords / 4) % 2 == 1,
              "conflict-free fragment loads need a stride of four times an odd number");

__device__ inline void mma_nvfp4(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
                                 uint32_t sa, uint32_t sb) {
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X"
      ".f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb));
}

// One staged operand's worth of global reads, held in registers so the loads
// issue before the mma they feed rather than after.
struct Prefetch {
  uint4 data[kU4PerThread];
  uint32_t scale;
};

// Turns a word of on-disk nibbles into a word of instruction-order nibbles.
// Three ops per eight elements, and only ever applied to the weight -- the
// activation is quantised by this file, so it is written the way the
// instruction reads it regardless of what the checkpoint does.
__device__ inline uint32_t swap_nibbles(uint32_t v) {
  return ((v & 0x0F0F0F0Fu) << 4) | ((v >> 4) & 0x0F0F0F0Fu);
}

// Byte offset of the four block scales `[j, j+4)` of row `m`. `j` is always a
// multiple of four, which is what lets the swizzle cost an address computation
// rather than a repack: the four bytes a `4X` operand wants stay contiguous and
// 4-byte aligned under it.
//
// `kSwizzled` distinguishes the weight from the activation, not one checkpoint
// from another. The activation is quantised by this file straight into
// row-major order; only the stored weight is swizzled.
template <bool kSwizzled>
__device__ inline size_t scale_offset(int m, int j, int k_blocks) {
  if (kSwizzled) {
    // `j & 3` is always zero on the kernel's own path -- it loads four blocks
    // at a time -- and is written out anyway so the same function serves a
    // host packer that addresses one block.
    return (static_cast<size_t>(m >> 7) * (k_blocks >> 2) + (j >> 2)) * 512 +
           static_cast<size_t>(m & 31) * 16 + static_cast<size_t>((m & 127) >> 5) * 4 +
           static_cast<size_t>(j & 3);
  }
  return static_cast<size_t>(m) * k_blocks + j;
}

// Rows past `n_rows` and k past `k_bytes` are zeroed rather than skipped: a
// zero nibble is E2M1 0.0 and a zero scale byte is E4M3 0.0, so a padded tile
// contributes exactly nothing to the accumulator. `in_features % 64 == 0`
// makes every one of these loads whole -- no partial uint4, no partial scale
// word -- which is the entire reason for that requirement.
template <bool kSwapNibbles, bool kSwizzledScale>
__device__ inline void fetch_tile(const uint8_t* __restrict__ src,
                                  const uint8_t* __restrict__ src_scale, int row0, int n_rows,
                                  int k0, int k_bytes, int k_blocks, int tid, Prefetch& p) {
#pragma unroll
  for (int i = 0; i < kU4PerThread; ++i) {
    const int idx = tid + i * kThreads;
    const int r = idx / kU4PerRow;
    const int c = idx % kU4PerRow;
    const int row = row0 + r;
    const int byte = k0 / 2 + c * 16;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (row < n_rows && byte + 16 <= k_bytes) {
      v = *reinterpret_cast<const uint4*>(src + static_cast<size_t>(row) * k_bytes + byte);
      if (kSwapNibbles) {
        v.x = swap_nibbles(v.x);
        v.y = swap_nibbles(v.y);
        v.z = swap_nibbles(v.z);
        v.w = swap_nibbles(v.w);
      }
    }
    p.data[i] = v;
  }
  const int sr = tid / kScaleWords;
  const int sh = tid % kScaleWords;
  const int srow = row0 + sr;
  const int sblock = k0 / 16 + sh * 4;
  p.scale = (srow < n_rows && sblock + 4 <= k_blocks)
                ? *reinterpret_cast<const uint32_t*>(
                      src_scale + scale_offset<kSwizzledScale>(srow, sblock, k_blocks))
                : 0u;
}

__device__ inline void store_tile(uint32_t* __restrict__ dst, uint32_t* __restrict__ dst_scale,
                                  int tid, const Prefetch& p) {
#pragma unroll
  for (int i = 0; i < kU4PerThread; ++i) {
    const int idx = tid + i * kThreads;
    const int r = idx / kU4PerRow;
    const int c = idx % kU4PerRow;
    *reinterpret_cast<uint4*>(dst + r * kRowWords + c * 4) = p.data[i];
  }
  dst_scale[tid] = p.scale;
}

// **The `, 2` is load-bearing and is not a hint about what would be nice.**
// Without a minimum-blocks argument ptxas optimises for one block and targets
// 255 registers; it happened to land on 126, one either side of the cliff at
// 128, so two blocks per SM was luck rather than a decision. Any change that
// moves register pressure -- and the shared-window fix below moves it to 130 --
// silently costs a block and about a quarter of the throughput.
//
// Two is also the ceiling, and both walls are already touched:
//   shared    45056 x 2 = 90112 of the SM's 102400 bytes
//   registers   128 x 512 = 65536, the whole register file
// So a third block is impossible at this tile size, and asking for one would
// force registers to 85 and spill. Anyone changing kBM/kBN/kBK is standing
// against those two numbers.
__global__ __launch_bounds__(kThreads, 2) void nvfp4_gemm_kernel(
    const uint8_t* __restrict__ aq, const uint8_t* __restrict__ as,
    const uint8_t* __restrict__ bq, const uint8_t* __restrict__ bs, __nv_bfloat16* __restrict__ y,
    int rows, int out_features, int k_bytes, int k_blocks, int k_tiles, float global_scale) {
  // 16-byte aligned so the staging stores can be `uint4`; a plain uint32 array
  // is only guaranteed 4.
  __shared__ __align__(16) uint32_t smem[2 * kStageWords];

  const int tid = static_cast<int>(threadIdx.x);
  const int warp = tid / kWarp;
  const int lane = tid % kWarp;
  const int gid = lane >> 2;
  const int tig = lane & 3;
  const int wm = warp / kWarpsN;
  const int wn = warp % kWarpsN;

  const int m0 = static_cast<int>(blockIdx.x) * kBM;
  const int n0 = static_cast<int>(blockIdx.y) * kBN;

  // **Every stage base is an integer offset from `smem`, never an element of a
  // pointer array.** `uint32_t* stage[2]` indexed by a runtime buffer number
  // is the obvious way to write this and it costs a quarter of the kernel:
  // ptxas can no longer prove the address is in the shared window, so every
  // access in the k loop below becomes a generic `LD.E`/`ST.E` instead of
  // `LDS`/`STS`, and the array itself lands in local memory and is reloaded
  // with an `LDL.64` each iteration. It does not show up as a spill -- a local
  // array is not a spill -- so the usual "0 bytes spill" check misses it
  // entirely. Arithmetic on the `__shared__` symbol keeps the window; an array
  // of pointers to it does not.
  Prefetch pa;
  Prefetch pb;
  fetch_tile<false, false>(aq, as, m0, rows, 0, k_bytes, k_blocks, tid, pa);
  fetch_tile<true, true>(bq, bs, n0, out_features, 0, k_bytes, k_blocks, tid, pb);
  store_tile(smem, smem + 2 * kBM * kRowWords, tid, pa);
  store_tile(smem + kBM * kRowWords, smem + 2 * kBM * kRowWords + kBM * kScaleWords, tid, pb);
  __syncthreads();

  float acc[kMTiles][kNTiles][4];
#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt)
#pragma unroll
    for (int nt = 0; nt < kNTiles; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[mt][nt][i] = 0.0f;

  // The A-scale row this lane must supply. Lanes `4g` and `4g+1` carry rows
  // `g` and `g+8`; `4g+2` and `4g+3` carry nothing, and pointing them at the
  // same row as `4g` keeps the read a broadcast instead of a fifth bank.
  const int a_row_lo = wm * kWM + gid;
  const int a_scale_off = (lane & 1) ? 8 : 0;

  for (int kt = 0; kt < k_tiles; ++kt) {
    const int cur = kt & 1;
    const int nxt = cur ^ 1;

    if (kt + 1 < k_tiles) {
      const int k0 = (kt + 1) * kBK;
      fetch_tile<false, false>(aq, as, m0, rows, k0, k_bytes, k_blocks, tid, pa);
      fetch_tile<true, true>(bq, bs, n0, out_features, k0, k_bytes, k_blocks, tid, pb);
    }

    const uint32_t* const sa = smem + cur * kStageWords;
    const uint32_t* const sb = sa + kBM * kRowWords;
    const uint32_t* const sas = sa + 2 * kBM * kRowWords;
    const uint32_t* const sbs = sas + kBM * kScaleWords;

#pragma unroll
    for (int s = 0; s < kKSteps; ++s) {
      const int word = s * 8 + tig;

      uint32_t af[kMTiles][4];
      uint32_t asc[kMTiles];
#pragma unroll
      for (int mt = 0; mt < kMTiles; ++mt) {
        const int lo = (a_row_lo + mt * 16) * kRowWords;
        const int hi = lo + 8 * kRowWords;
        af[mt][0] = sa[lo + word];
        af[mt][1] = sa[hi + word];
        af[mt][2] = sa[lo + word + 4];
        af[mt][3] = sa[hi + word + 4];
        asc[mt] = sas[(a_row_lo + mt * 16 + a_scale_off) * kScaleWords + s];
      }

      uint32_t bf[kNTiles][2];
      uint32_t bsc[kNTiles];
#pragma unroll
      for (int nt = 0; nt < kNTiles; ++nt) {
        const int col = wn * kWN + nt * 8 + gid;
        bf[nt][0] = sb[col * kRowWords + word];
        bf[nt][1] = sb[col * kRowWords + word + 4];
        bsc[nt] = sbs[col * kScaleWords + s];
      }

#pragma unroll
      for (int mt = 0; mt < kMTiles; ++mt)
#pragma unroll
        for (int nt = 0; nt < kNTiles; ++nt) mma_nvfp4(acc[mt][nt], af[mt], bf[nt], asc[mt],
                                                       bsc[nt]);
    }

    if (kt + 1 < k_tiles) {
      // Safe without a second barrier: buffer `nxt` was last read in iteration
      // kt-1, and every warp passed that iteration's barrier before reaching
      // this store.
      uint32_t* const dst = smem + nxt * kStageWords;
      store_tile(dst, dst + 2 * kBM * kRowWords, tid, pa);
      store_tile(dst + kBM * kRowWords, dst + 2 * kBM * kRowWords + kBM * kScaleWords, tid, pb);
      __syncthreads();
    }
  }

  // **The column never needs a guard, and there is no scalar fallback.**
  // `out_features % 128 == 0` is the block-scale swizzle's own precondition,
  // enforced by `nvfp4_gemm_supported` and thrown on by `nvfp4_gemm_forward`,
  // so `n0 + 128 <= out_features` and this thread's pair is always whole and
  // always 4-byte aligned. Carrying the fallback anyway cost 64 `STG.E.U16`
  // and 64 `F2FP` against 32 packed stores, plus 39 BSSY/BSYNC pairs of
  // reconvergence, to serve a shape the launcher refuses to accept.
  //
  // The *row* guard stays: `rows` is a batch size and is under no such
  // constraint.
#pragma unroll
  for (int mt = 0; mt < kMTiles; ++mt) {
    const int row_lo = m0 + wm * kWM + mt * 16 + gid;
#pragma unroll
    for (int nt = 0; nt < kNTiles; ++nt) {
      const int col = n0 + wn * kWN + nt * 8 + tig * 2;
      const float* c = acc[mt][nt];
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int row = row_lo + half * 8;
        if (row >= rows) continue;
        *reinterpret_cast<__nv_bfloat162*>(y + static_cast<size_t>(row) * out_features + col) =
            __floats2bfloat162_rn(c[half * 2] * global_scale, c[half * 2 + 1] * global_scale);
      }
    }
  }
}

}  // namespace

bool nvfp4_gemm_supported(int out_features, int in_features) {
  // `in % 64` is this kernel's own staging requirement -- it is what makes a
  // packed row 32-byte aligned and a scale row 4-byte aligned -- and it happens
  // to subsume the swizzle's `Kb % 4`. `out % 128` is the swizzle's.
  return in_features > 0 && out_features > 0 && in_features % 64 == 0 && out_features % 128 == 0;
}

size_t nvfp4_gemm_workspace_bytes(int rows, int in_features) {
  const size_t r = static_cast<size_t>(std::min(rows, kNVFP4RowChunk));
  return align_up(r * in_features / 2) + align_up(r * in_features / 16);
}

void launch_quantize_nvfp4_activations(const __nv_bfloat16* x, uint8_t* packed, uint8_t* scales,
                                       int rows, int dim, cudaStream_t stream) {
  if (rows <= 0 || dim <= 0) return;
  if (dim % 16 != 0) {
    throw std::runtime_error("nvfp4: activation dim must be a multiple of 16, got " +
                             std::to_string(dim));
  }
  const size_t halves = static_cast<size_t>(rows) * dim / 8;
  const int threads = 256;
  const int blocks = static_cast<int>((halves + threads - 1) / threads);
  quantize_act_kernel<<<blocks, threads, 0, stream>>>(x, packed, scales, halves);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void nvfp4_gemm_forward(const __nv_bfloat16* x, const uint8_t* w_packed, const uint8_t* w_scale,
                        float global_scale, __nv_bfloat16* y, int rows, int out_features,
                        int in_features, Workspace& ws, cudaStream_t stream) {
  if (rows <= 0 || out_features <= 0) return;
  // Throw rather than pad. `out % 128` and `Kb % 4` are the swizzle's
  // no-padding precondition; every quantised tensor in both checkpoints
  // satisfies both, so the padded layout has never been observed and guessing
  // its convention is exactly the unverified assumption this project keeps
  // being bitten by. A file that needs it should say so on the day it appears.
  if (!nvfp4_gemm_supported(out_features, in_features)) {
    throw std::runtime_error("nvfp4_gemm: " + std::to_string(out_features) + "x" +
                             std::to_string(in_features) +
                             " needs out % 128 == 0 and in % 64 == 0; the padded block-scale "
                             "layout has never been observed and is not guessed at");
  }

  const int k_bytes = in_features / 2;
  const int k_blocks = in_features / 16;
  const int k_tiles = (in_features + kBK - 1) / kBK;

  // Rows are processed in bounded passes so the activation buffer does not
  // scale with the batch. Every call the transformer makes is already inside
  // one pass; the loop exists so a larger caller degrades in workspace rather
  // than failing to allocate.
  for (int start = 0; start < rows; start += kNVFP4RowChunk) {
    const int n = std::min(kNVFP4RowChunk, rows - start);
    Workspace::Scope scope(ws);
    uint8_t* xq = ws.alloc_n<uint8_t>(static_cast<size_t>(n) * k_bytes);
    uint8_t* xs = ws.alloc_n<uint8_t>(static_cast<size_t>(n) * k_blocks);

    launch_quantize_nvfp4_activations(x + static_cast<size_t>(start) * in_features, xq, xs, n,
                                      in_features, stream);

    // Row blocks on x and column blocks on y, not the other way round: CUDA
    // dispatches x fastest, so the resident blocks share one 128-column slab of
    // the weight -- 344 KB at qkv_proj -- and sweep the activation together.
    // Swapping them would put the larger of the two streams in the reused
    // position and spill it out of L2.
    const dim3 grid(static_cast<unsigned>((n + kBM - 1) / kBM),
                    static_cast<unsigned>((out_features + kBN - 1) / kBN));
    nvfp4_gemm_kernel<<<grid, kThreads, 0, stream>>>(
        xq, xs, w_packed, w_scale, y + static_cast<size_t>(start) * out_features, n, out_features,
        k_bytes, k_blocks, k_tiles, global_scale);
    VIDFAB_CUDA_CHECK(cudaGetLastError());
  }
}

}  // namespace vidfab::cuda
