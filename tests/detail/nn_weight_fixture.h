#pragma once
#include "nn_kernels_fixture.h"

namespace {
// --- linear -----------------------------------------------------------------

// Hoisting the dequantisation out of a row-chunk loop must be BIT-identical to
// dequantising per chunk, not merely close: it is the whole justification for
// `prepare` + `forward_prepared`, and a tolerance-based check here would pass
// on a version that dequantised a stale or mis-sliced weight.

// The end-to-end ConvRot path: the stored weight is already rotated, so the
// activation must be rotated online or the result is `x H W^T`, which is
// well-scaled noise (docs/convrot_notes.md).

// --- nvfp4 storage ----------------------------------------------------------
//
// Two facts about how the shipped checkpoints store an nvfp4 weight are not
// inferable from the file and were measured against the fp8 build of the same
// model (docs/transformer_spec.md 8.6):
//
//   - the **high** nibble of each byte is the even-indexed element;
//   - block scales are written in a 128x4 tile layout, not row-major.
//
// Both are silent when wrong. They leave the value histogram intact and the
// output finite and correctly scaled, and they cost elementwise correlation
// against the reference — 0.995 becomes 0.00003 — while every summary statistic
// stays where it was. So each wrong form is constructed below and the kernel is
// required *not* to match it; a test that only checked the right answer would
// pass under all four combinations.

// The 128x4 tile map, written out as an explicit walk of the tile and its
// interior rather than as the kernel's packed shift expression, so that the two
// are genuinely independent statements of the same layout.
size_t nvfp4_scale_slot(int o, int k, int blocks_per_row) {
  const int tiles_per_row = blocks_per_row / 4;
  const size_t tile = size_t(o / 128) * tiles_per_row + size_t(k / 4);
  const int row_in_tile = o % 128;
  // The 128 rows of a tile are visited as four groups of 32, the group index
  // moving slower than the row inside it.
  const size_t inside =
      size_t(row_in_tile % 32) * 16 + size_t(row_in_tile / 32) * 4 + size_t(k % 4);
  return tile * 128 * 4 + inside;
}

// A weight in the checkpoint's storage form, plus everything needed to state
// what it should dequantise to.
struct Nvfp4Weight {
  int out_features = 0;
  int in_features = 0;
  float global = 0.0f;
  std::vector<uint8_t> codes;  // one E2M1 code per element, [out, in]
  std::vector<uint8_t> scales; // one e4m3 byte per 16 elements, unswizzled [out, in/16]
  std::vector<uint8_t> packed; // [out, in/2], even element in the high nibble
  std::vector<uint8_t> stored; // `scales` written through the 128x4 tile map
};

Nvfp4Weight make_nvfp4(int out_features, int in_features, float global, uint32_t seed) {
  Nvfp4Weight w;
  w.out_features = out_features;
  w.in_features = in_features;
  w.global = global;
  const int blocks_per_row = in_features / int(slopfab::cuda::kNVFP4BlockSize);

  uint32_t s = seed;
  auto next = [&]() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  };

  w.codes.resize(size_t(out_features) * in_features);
  for (size_t i = 0; i < w.codes.size(); ++i)
    w.codes[i] = uint8_t(next() & 0x0Fu);

  w.scales.resize(size_t(out_features) * blocks_per_row);
  for (size_t i = 0; i < w.scales.size(); ++i) {
    // Exponents 4..11 keep the scales well clear of the subnormals and of the
    // 0x7F/0xFF NaN encodings, and spread over three orders of magnitude so a
    // scale landing on the wrong block cannot go unnoticed.
    const uint32_t r = next();
    w.scales[i] = uint8_t(((4u + (r % 8u)) << 3) | (r >> 8 & 0x07u));
  }

  w.packed.assign(size_t(out_features) * in_features / 2, 0);
  for (int o = 0; o < out_features; ++o) {
    for (int i = 0; i < in_features; i += 2) {
      const size_t e = size_t(o) * in_features + i;
      w.packed[size_t(o) * (in_features / 2) + i / 2] = uint8_t((w.codes[e] << 4) | w.codes[e + 1]);
    }
  }

  w.stored.assign(w.scales.size(), 0);
  for (int o = 0; o < out_features; ++o) {
    for (int k = 0; k < blocks_per_row; ++k) {
      w.stored[nvfp4_scale_slot(o, k, blocks_per_row)] = w.scales[size_t(o) * blocks_per_row + k];
    }
  }
  return w;
}

// How the weight should come out. `block` and `swap_nibbles` exist so the same
// function can produce the wrong forms the kernel must be shown to reject.
std::vector<float> nvfp4_reference(const Nvfp4Weight& w, int block = 16,
                                   bool swap_nibbles = false) {
  const int blocks_per_row = w.in_features / int(slopfab::cuda::kNVFP4BlockSize);
  std::vector<float> out(w.codes.size());
  for (int o = 0; o < w.out_features; ++o) {
    for (int i = 0; i < w.in_features; ++i) {
      const int k = std::min(i / block, blocks_per_row - 1);
      // Same order of operations as the kernel, so bf16 rounding matches and
      // exact equality is the right bar.
      const float scale =
          slopfab::f8_e4m3_to_f32(w.scales[size_t(o) * blocks_per_row + k]) * w.global;
      const int src = swap_nibbles ? (i ^ 1) : i;
      const float v = slopfab::f4_e2m1_to_f32(w.codes[size_t(o) * w.in_features + src]) * scale;
      out[size_t(o) * w.in_features + i] = slopfab::bf16_to_f32(slopfab::f32_to_bf16(v));
    }
  }
  return out;
}

// `nn_dequant_nvfp4` pins the two layout facts on one 256x128 weight. This
// sweeps the tile counts instead, because the swizzle's address map is the part
// that varies with shape and the part a rewrite gets wrong: a tile count of one
// in a dimension hides a missing tile stride, and a count that is a power of two
// hides a missing multiply.
//
// Deliberately written against the launcher rather than any one kernel's
// indexing, so it survives a change of strategy inside `launch_dequant_nvfp4`.
// It was added alongside a tile-per-block dequant that was then reverted for
// being slower, and it passed unchanged across both — which is the property
// wanted from it.
//
// Exact equality is the bar, not a tolerance: the reference below is built from
// the independent tile walk in `nvfp4_scale_slot` and multiplies in the same
// order the kernel does.

struct Nf4Weight {
  int out = 0, in = 0;
  float offset = 0.0f;
  std::vector<uint8_t> packed, absmax;
  std::vector<float> map, nested_map, nested_absmax;
};

Nf4Weight make_nf4(int out, int in) {
  Nf4Weight w;
  w.out = out;
  w.in = in;
  w.offset = 0.21360844373703003f;
  w.map = {-1.0f,        -0.6961928f, -0.52507305f, -0.39491749f, -0.28444138f, -0.18477343f,
           -0.09105004f, 0.0f,        0.07958030f,  0.16093020f,  0.24611230f,  0.33791524f,
           0.44070983f,  0.56261700f, 0.72295684f,  1.0f};
  w.nested_map.resize(256);
  for (int i = 0; i < 256; ++i)
    w.nested_map[i] = (float(i) - 127.0f) / 128.0f;
  const size_t n = size_t(out) * in;
  const size_t blocks = (n + 63) / 64;
  w.nested_absmax.resize((blocks + 255) / 256);
  for (size_t i = 0; i < w.nested_absmax.size(); ++i)
    w.nested_absmax[i] = 0.75f + float(i) * 1.25f;
  w.absmax.resize(blocks);
  for (size_t i = 0; i < blocks; ++i)
    w.absmax[i] = uint8_t((i * 73 + 19) & 255);
  w.packed.resize((n + 1) / 2);
  for (size_t i = 0; i < w.packed.size(); ++i) {
    // Deliberately different nibbles; a swapped implementation cannot pass.
    w.packed[i] = uint8_t((((i * 5 + 3) & 15) << 4) | ((i * 11 + 9) & 15));
  }
  return w;
}

std::vector<float> nf4_reference(const Nf4Weight& w, bool swap = false) {
  const size_t n = size_t(w.out) * w.in;
  std::vector<float> result(n);
  for (size_t i = 0; i < n; ++i) {
    const size_t block = i / 64;
    const float scale = w.nested_map[w.absmax[block]] * w.nested_absmax[block / 256] + w.offset;
    const uint8_t byte = w.packed[i / 2];
    const bool high = ((i & 1) == 0) != swap;
    const uint8_t code = high ? byte >> 4 : byte & 15;
    result[i] = slopfab::bf16_to_f32(slopfab::f32_to_bf16(w.map[code] * scale));
  }
  return result;
}

// NF4 has two kernels now: an eight-wide one that loads four packed bytes at a
// time and a two-wide fallback for the shapes and alignments it cannot serve.
// Which one runs is decided entirely by `nf4_vector_eligible` in the launcher,
// so the two cases below are not guesses: 129x128 is 16512 elements out of two
// 256-byte-aligned DeviceBuffers with block sizes 64 and 256, which satisfies
// every clause, and offsetting the destination by one bf16 makes it 2-byte
// aligned, which fails the 16-byte clause and nothing else.
//
// Comparing the raw payloads is the point. The scalar kernel is the reference
// implementation this replaces; "same to within tolerance" would not establish
// anything, because the whole claim is that no bit moves.

// The AWQ per-input-channel activation scale. Only the text encoder's weights
// carry one; the transformer's are all null, which is a positive statement that
// the quantiser folded the scale into the preceding norm.

// Defined with the native-GEMM tests at the end of this file, where the rule
// they implement is written down. Declared here so `linear_nvfp4` can hold the
// native path to an fp4-activation reference without moving the definitions out
// of the block they belong to.
std::vector<float> host_quantise_act(const std::vector<float>& x, int rows, int dim);
double rms_rel(const std::vector<float>& want, const std::vector<float>& got);
double correlation(const std::vector<float>& a, const std::vector<float>& b);

// The whole path: a stored nvfp4 weight through LinearRunner against a host
// matmul of the dequantised reference.

// --- nvfp4 tensor core ------------------------------------------------------
//
// The shipped nvfp4 checkpoints are packed for
//
//   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale
//       .scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
//
// and every byte of that instruction's operand layout has to be right before a
// GEMM built on it can be trusted. The A/B packing is inferable from the 8-bit
// m16n8k32 layout; the block-scale operand is not, so it was established by
// experiment and is pinned here. Getting either wrong yields finite, plausibly
// scaled garbage -- the failure mode this whole project keeps running into.

__device__ __host__ inline float e2m1_ref(uint8_t n) {
  const float m[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float v = m[n & 7];
  return (n & 8) ? -v : v;
}

__global__ void nvfp4_mma_kernel(const uint32_t* a, const uint32_t* b, const uint32_t* sa,
                                 float* out) {
  const int lane = threadIdx.x;
  const uint32_t ra[4] = {a[lane * 4], a[lane * 4 + 1], a[lane * 4 + 2], a[lane * 4 + 3]};
  const uint32_t rb[2] = {b[lane * 2], b[lane * 2 + 1]};
  const uint32_t s_a = sa[lane];
  const uint32_t s_b = 0x38383838u; // four e4m3 1.0 scales
  float c[4] = {0, 0, 0, 0};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X"
               ".f32.e2m1.e2m1.f32.ue4m3 "
               "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]), "r"(s_a),
                 "r"(s_b));
#endif
  const int gid = lane >> 2, tig = lane & 3;
  out[gid * 8 + tig * 2] = c[0];
  out[gid * 8 + tig * 2 + 1] = c[1];
  out[(gid + 8) * 8 + tig * 2] = c[2];
  out[(gid + 8) * 8 + tig * 2 + 1] = c[3];
}

// The lane that carries row r's block scales. Rows 0-7 sit on lane 4r, rows
// 8-15 on lane 4(r-8)+1; lanes 4g+2 and 4g+3 carry nothing, which is why only
// 64 of the warp's 128 scale bytes are live.
int scale_lane_for_row(int r) {
  return r < 8 ? 4 * r : 4 * (r - 8) + 1;
}

// --- native nvfp4 GEMM ------------------------------------------------------
//
// Everything from here to the end of the file belongs to the native
// block-scaled nvfp4 GEMM in src/cuda/nvfp4_gemm.cu. It is one block so that
// the three tracks working on this file merge cleanly.
//
// Three separable things can be wrong and any two of them can cancel: the
// register operand layout (pinned by `nvfp4_mma_operand_layout`), the on-disk
// convention (pinned by `nn_dequant_nvfp4`), and the dynamic activation
// quantisation, which is this project's own choice and belongs to no
// checkpoint. Each is checked on its own here.
//
// The last of the three is not free and is not a bug. `nvfp4_activation_cost`
// measures it, and the reason it is measured rather than asserted is written
// out there.

// The e4m3 and e2m1 encoders, by brute force over the representable set. A
// reference exists to be obviously right; round-to-nearest-*even* on a
// four-bit exponent is easy to get subtly wrong by hand, and a reference that
// shares a bug with the thing it checks proves nothing.
// `nvfp4_rounding_reference` pins both against the hardware converters the
// kernel actually issues, before anything is built on them.
uint8_t host_e4m3(float v) {
  uint8_t best = 0;
  double bd = 1e300;
  for (int c = 0; c < 0x7F; ++c) { // non-negative only; 0x7F is NaN
    const double d = std::fabs(double(slopfab::f8_e4m3_to_f32(uint8_t(c))) - double(v));
    if (d < bd || (d == bd && (c & 1) == 0)) {
      bd = d;
      best = static_cast<uint8_t>(c);
    }
  }
  return best;
}

// The sign is carried, not searched for. Rounding over all sixteen codes at
// once makes +0.1 a tie between +0.0 and -0.0, and which of the two comes back
// is a question about the sign of zero rather than about rounding -- the
// hardware keeps the input's sign, so this does too. It is the only place the
// two ever disagreed, over 0 of 270 *values* and 8 of 270 codes.
uint8_t host_e2m1(float v) {
  uint8_t mag = 0;
  double bd = 1e300;
  for (int c = 0; c < 8; ++c) {
    const double d = std::fabs(double(slopfab::f4_e2m1_to_f32(uint8_t(c))) - std::fabs(double(v)));
    if (d < bd || (d == bd && (c & 1) == 0)) {
      bd = d;
      mag = static_cast<uint8_t>(c);
    }
  }
  return static_cast<uint8_t>(mag | (std::signbit(v) ? 8u : 0u));
}

// A dense matrix put into the checkpoint's storage form, plus exactly what
// those bytes mean. `make_nvfp4` above starts from random codes, which is the
// right shape for a layout test; this starts from real values, which is what a
// numerical one needs.
//
// `dense` is the reference the GEMM is judged against. It is not the input
// matrix: it is the input matrix after nvfp4 has had its way with it.
struct NvfpPacked {
  std::vector<uint8_t> data;
  std::vector<uint8_t> scale;
  std::vector<float> dense;
};

// `high_even` and `swizzled` select the on-disk convention. Both shipped
// checkpoints are (true, true); the other three exist so a test can show the
// kernel tells them apart rather than happening to agree on symmetric data.
NvfpPacked pack_nvfp4(const std::vector<float>& w, int rows, int cols, float global, bool high_even,
                      bool swizzled) {
  const int kb = cols / 16;
  NvfpPacked t;
  t.data.assign(size_t(rows) * cols / 2, 0);
  t.scale.assign(size_t(rows) * kb, 0);
  t.dense.assign(size_t(rows) * cols, 0.0f);
  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < kb; ++b) {
      float amax = 0.0f;
      for (int i = 0; i < 16; ++i) {
        amax = std::max(amax, std::fabs(w[size_t(r) * cols + b * 16 + i]));
      }
      const uint8_t s8 = host_e4m3(amax / 6.0f / global);
      const float sd = slopfab::f8_e4m3_to_f32(s8) * global;
      const float inv = sd > 0.0f ? 1.0f / sd : 0.0f;
      t.scale[swizzled ? nvfp4_scale_slot(r, b, kb) : size_t(r) * kb + b] = s8;
      for (int i = 0; i < 16; ++i) {
        const int col = b * 16 + i;
        const size_t flat = size_t(r) * cols + col;
        const uint8_t q = host_e2m1(w[flat] * inv);
        t.data[flat / 2] |= static_cast<uint8_t>(q << (((col % 2 == 0) == high_even) ? 4 : 0));
        t.dense[flat] = slopfab::f4_e2m1_to_f32(q) * sd;
      }
    }
  }
  return t;
}

// The kernel's activation rule restated: amax/6 rounded to e4m3, then the
// elements divided by the *decoded* scale and rounded to e2m1. In float rather
// than double so the ties fall the same way.
std::vector<float> host_quantise_act(const std::vector<float>& x, int rows, int dim) {
  std::vector<float> out(x.size(), 0.0f);
  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < dim / 16; ++b) {
      float amax = 0.0f;
      for (int i = 0; i < 16; ++i) {
        amax = std::max(amax, std::fabs(x[size_t(r) * dim + b * 16 + i]));
      }
      const uint8_t s8 = host_e4m3(amax * (1.0f / 6.0f));
      const float sd = slopfab::f8_e4m3_to_f32(s8);
      const float inv = sd > 0.0f ? 1.0f / sd : 0.0f;
      for (int i = 0; i < 16; ++i) {
        const size_t flat = size_t(r) * dim + b * 16 + i;
        out[flat] = slopfab::f4_e2m1_to_f32(host_e2m1(x[flat] * inv)) * sd;
      }
    }
  }
  return out;
}

// Box-Muller over the harness's own generator, because post-norm activations
// are Gaussian-ish and the harness's uniform is not.
//
// The reason is realism, not pessimism, and the naive argument for it is
// backwards. One would expect uniform data to flatter a block-scaled format --
// every element sits near its own block maximum, so a shared scale wastes
// nothing. Measured, uniform comes out *worse*: 0.101 against Gaussian's
// 0.095 (`nvfp4_activation_cost`). E2M1's grid is finer near zero, with a step
// of 0.5 below 2 and of 2 above 4, so a distribution that puts most of its
// mass well inside its own maximum is the one the format suits.
std::vector<float> make_gaussian(size_t n, uint32_t seed, float sigma) {
  const std::vector<float> u = make_data(2 * n, seed, 0.5f); // (-0.5, 0.5)
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    const float a = std::max(1e-7f, u[2 * i] + 0.5f);
    v[i] = sigma * std::sqrt(-2.0f * std::log(a)) * std::cos(6.2831853f * (u[2 * i + 1] + 0.5f));
  }
  return v;
}

double rms_rel(const std::vector<float>& want, const std::vector<float>& got) {
  double num = 0.0;
  double den = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double d = double(got[i]) - want[i];
    num += d * d;
    den += double(want[i]) * want[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : 0.0;
}

// Pearson correlation. The discriminator between quantisation noise and a
// layout bug: noise leaves this at 0.99-something, a misread operand collapses
// it towards zero while leaving every summary statistic looking healthy.
double correlation(const std::vector<float>& a, const std::vector<float>& b) {
  double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
  const double n = double(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    sa += a[i];
    sb += b[i];
    saa += double(a[i]) * a[i];
    sbb += double(b[i]) * b[i];
    sab += double(a[i]) * b[i];
  }
  const double cov = sab / n - (sa / n) * (sb / n);
  const double va = saa / n - (sa / n) * (sa / n);
  const double vb = sbb / n - (sb / n) * (sb / n);
  return (va > 0 && vb > 0) ? cov / std::sqrt(va * vb) : 0.0;
}

std::vector<float> run_native_nvfp4(const std::vector<float>& x, const NvfpPacked& w, int rows,
                                    int out_features, int in_features, float global) {
  BfBuf dx(x), dy(size_t(rows) * out_features);
  DeviceBuffer<uint8_t> dw(w.data.size()), dws(w.scale.size());
  dw.copy_from_host(w.data.data(), w.data.size());
  dws.copy_from_host(w.scale.data(), w.scale.size());
  Workspace ws;
  ws.reserve(slopfab::cuda::nvfp4_gemm_workspace_bytes(rows, in_features) + 4096);
  slopfab::cuda::nvfp4_gemm_forward(dx.p(), dw.get(), dws.get(), global, dy.p(), rows, out_features,
                                    in_features, ws, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  return dy.host();
}

// --- the B-side block-scale operand -----------------------------------------

__global__ void nvfp4_mma_bscale_kernel(const uint32_t* a, const uint32_t* b, const uint32_t* sa,
                                        const uint32_t* sb, float* out) {
  const int lane = threadIdx.x;
  const uint32_t ra[4] = {a[lane * 4], a[lane * 4 + 1], a[lane * 4 + 2], a[lane * 4 + 3]};
  const uint32_t rb[2] = {b[lane * 2], b[lane * 2 + 1]};
  const uint32_t s_a = sa[lane];
  const uint32_t s_b = sb[lane];
  float c[4] = {0, 0, 0, 0};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
  asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X"
               ".f32.e2m1.e2m1.f32.ue4m3 "
               "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]), "r"(s_a),
                 "r"(s_b));
#endif
  const int gid = lane >> 2, tig = lane & 3;
  out[gid * 8 + tig * 2] = c[0];
  out[gid * 8 + tig * 2 + 1] = c[1];
  out[(gid + 8) * 8 + tig * 2] = c[2];
  out[(gid + 8) * 8 + tig * 2 + 1] = c[3];
}

// `nvfp4_mma_operand_layout` pins the A side: row r's four block scales are the
// four bytes of lane `r < 8 ? 4r : 4(r-8)+1`. The B side needs its own
// experiment and does not follow by symmetry. A has sixteen rows and uses
// sixteen lanes; B has eight columns and uses **eight** — column c's scales are
// the four bytes of lane 4c, and lanes 4c+1..4c+3 carry nothing at all.
//
// Halving the A rule instead — putting columns 0-3 on lanes 4g and 4-7 on lanes
// 4g+1, which is the shape one reaches for — writes half the scales into lanes
// the instruction ignores. It does not crash and does not produce zeros. It
// produces a well-formed matrix with four of its eight columns scaled wrong.

// --- rounding ---------------------------------------------------------------

__global__ void nvfp4_cvt_probe_kernel(const float* in, uint8_t* e2m1, uint8_t* e4m3, int pairs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pairs)
    return;
  e2m1[i] = static_cast<uint8_t>(
      __nv_cvt_float2_to_fp4x2(make_float2(in[i * 2], in[i * 2 + 1]), __NV_E2M1, cudaRoundNearest));
  e4m3[i] = __nv_cvt_float_to_fp8(in[i * 2], __NV_SATFINITE, __NV_E4M3);
}

// The awkward inputs are the ties — 0.25, 0.75, 1.75, 3.5, 5.0 — where
// round-to-nearest-even and round-half-away-from-zero disagree, and the
// saturating end, where e2m1 must clamp to +-6 rather than wrap.

// --- activation quantisation ------------------------------------------------

// --- the GEMM against a CPU reference ---------------------------------------

// The load path from stored bytes to mma registers is not the identity, and
// every way of getting it wrong is silent. Each wrong form is constructed here
// and the kernel required not to match it.

// The global scale multiplies the whole tensor and is folded into the epilogue,
// so it has to appear exactly once. Twice, or not at all, still gives a
// well-formed matrix.

// --- what the operand path costs, separated from what the format costs -------

// The decisive experiment. Every activation here is already exactly on the fp4
// grid: each block is built from a power-of-two scale and the eight E2M1
// magnitudes, with at least one element at 6s so `amax/6` recovers `s` exactly.
// The kernel's dynamic quantiser is therefore the identity on this input, and
// what remains is the operand path alone.
//
// If this agrees and `nvfp4_activation_cost` does not, the activation
// quantisation is the entire story and the load path is correct. That is the
// one measurement that tells a numerical limit apart from a layout bug.

// The measurement, not an assertion.
//
// `set_native(true)` replaces bf16 activations with 4-bit ones, and that is a
// change to the arithmetic, not to the implementation of it. E2M1 has eight
// magnitudes; a 16-element block sharing one scale carries a per-element
// relative error of order 10%, and a dot product does not average it away —
// both the signal and the error grow as sqrt(K), so the output's relative error
// stays where the input's was. Nothing about the kernel changes that and no
// value of K rescues it, which is what the sweep below is for.
//
// So this reports rather than asserts. `nvfp4_gemm_exact_fp4_activations` is
// the assertion that the operand path is right; this is the cost of the format
// on top of it, and whether that cost is acceptable is a modelling decision.

// The same measurement on a real tensor. Synthetic weights cannot say whether
// the shipped block scales are benign; these are the bytes the model ships.

// --- timings ----------------------------------------------------------------

} // namespace
