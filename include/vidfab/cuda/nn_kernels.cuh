// Transformer primitives shared by the H3 DiT and the Qwen3-VL text encoder.
//
// Everything here is written from docs/transformer_spec.md and unit-tested
// against a CPU reference written from the same spec rather than from the
// kernel. The operator-order hazards in spec section 9.4 are the reason: a
// fused `n*scale + (n + shift)`, a RoPE applied before QK-norm, or a gate that
// multiplies the sum rather than the branch all produce finite, plausibly
// scaled output that is simply the wrong video.
//
// Convention: activations are row-major `[rows, dim]`, bf16 in the block
// stacks and fp32 where the spec calls for it. Every reduction accumulates in
// fp32 regardless of storage.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace vidfab::cuda {

// --- normalisation ----------------------------------------------------------

// out[r, i] = x[r, i] * rsqrt(mean_i(x[r, :]^2) + eps) * w[i]
//
// RMSNorm, not LayerNorm: no mean subtraction, no bias. `eps` is inside the
// square root, added to the mean of squares. Accumulates in fp32.
void launch_rmsnorm(const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* out, int rows,
                    int dim, float eps, cudaStream_t stream);
void launch_rmsnorm_f32(const float* x, const float* w, float* out, int rows, int dim, float eps,
                        cudaStream_t stream);

// ViT affine LayerNorm: subtract mean, divide by population standard
// deviation, then apply both learned weight and bias. All reductions are fp32.
void launch_layernorm_affine(const __nv_bfloat16* x, const __nv_bfloat16* w,
                             const __nv_bfloat16* bias, __nv_bfloat16* out,
                             int rows, int dim, float eps, cudaStream_t stream);

// RMSNorm followed by AdaLN modulation, fused:
//
//   n = rmsnorm(x[r], w, eps)
//   out[r, i] = n[i] * (1 + scale[a[r], i]) + shift[a[r], i]
//
// `a` is the per-row modulation index, `[rows]` int32. `scale` and `shift` are
// `[num_mod_rows, dim]` fp32 — they are computed once per denoising step from
// the rank-8 table and reused by every row, so they stay fp32 while the
// activation is bf16.
//
// The `1 +` and the order of operations are load-bearing: see spec section
// 9.4.1. Do not algebraically rearrange.
void launch_rmsnorm_modulate(const __nv_bfloat16* x, const __nv_bfloat16* w, const float* scale,
                             const float* shift, const int32_t* a, __nv_bfloat16* out, int rows,
                             int dim, float eps, cudaStream_t stream);

// As above but with a single modulation row shared by every token, for the
// final layer — which selects on `timestep_indices` alone and has no modality
// dependence (spec section 3.2). fp32 in and out.
void launch_rmsnorm_modulate_f32(const float* x, const __nv_bfloat16* w, const float* scale,
                                 const float* shift, const int32_t* a, float* out, int rows,
                                 int dim, float eps, cudaStream_t stream);

// --- residual and gating ----------------------------------------------------

// x[r, i] += gate[a[r], i] * branch[r, i]
//
// The gate multiplies the branch output only; the residual is ungated. In
// place on `x`.
void launch_add_gated(__nv_bfloat16* x, const __nv_bfloat16* branch, const float* gate,
                      const int32_t* a, int rows, int dim, cudaStream_t stream);

// --- activations ------------------------------------------------------------

// SwiGLU over a fused projection, gate first:
//
//   out[r, i] = silu(fused[r, i]) * fused[r, inner + i]
//
// The half order is the trap. Our checkpoints use the *original* key names
// (`mlp.fc1`), whose first half is the gate. A diffusers-converted
// `ff.net.0.proj` has the halves swapped. Spec section 4.4.
void launch_swiglu(const __nv_bfloat16* fused, __nv_bfloat16* out, int rows, int inner,
                   cudaStream_t stream);
// Backend-stable DiT exact-mode baseline paired with Vulkan. Normal attention
// modes retain the checkpoint's historical fast __expf implementation.
void launch_swiglu_exact(const __nv_bfloat16* fused, __nv_bfloat16* out,
                         int rows, int inner, cudaStream_t stream);

void launch_silu(const float* x, float* out, size_t n, cudaStream_t stream);
void launch_gelu_tanh(__nv_bfloat16* x, size_t n, cudaStream_t stream);
// Canonical exact-mode GELU paired with TensorBatch::vision_gelu_tanh_bf16.
// Unlike the shipped fast path above, this defines NaN/Inf/subnormal behavior.
void launch_gelu_tanh_exact(__nv_bfloat16* x, size_t n,
                            cudaStream_t stream);

// --- rotary -----------------------------------------------------------------

// MM-RoPE as used by the H3 transformer. `cos` and `sin` are `[rows, 96]` fp32
// shared across heads. Applied in place to `[rows, heads, head_dim]` with
// head_dim = 128:
//
//   out[j]       = x[j]    * cos[j] - x[j+48] * sin[j]     j in [0, 48)
//   out[j+48]    = x[j+48] * cos[j] + x[j]    * sin[j]
//   out[96..127] = x[96..127]                              passed through
//
// Note the pairing is j with j+48 over the 96 rotary channels, *not* j with
// j+64 over all 128. The 48-wide half-period is [T(16) | H(16) | W(16)].
// Spec section 5.3.
//
// The 96 is fixed by MM-RoPE's `3 axes x 16 frequencies x 2`, not derived from
// `head_dim`; `head_dim` only has to be at least that. Kept as a constant
// rather than a parameter because a caller that got it wrong would produce a
// plausible wrong video, not an error.
void launch_rope_h3(__nv_bfloat16* x, const float* cos, const float* sin, int rows, int heads,
                    int head_dim, cudaStream_t stream);

// Standard GPT-NeoX rotary over the full head dimension, pairing j with
// j + head_dim/2. For the Qwen3-VL encoder: its 3-D mrope degenerates to this
// for a pure-text sequence, because all three position axes carry the same
// value and the section split then has no effect.
//
// `cos` and `sin` are `[rows, head_dim]` — the half-period duplicated, as the
// reference builds it — not `[rows, head_dim/2]`.
void launch_rope_neox(__nv_bfloat16* x, const float* cos, const float* sin, int rows, int heads,
                      int head_dim, cudaStream_t stream);

// Builds the `[rows, 96]` cos/sin tables for MM-RoPE from float64 position
// coordinates. `pos` is `[rows, 3]` fp64 on the host — the reference builds it
// in float64 and casts to fp32 *inside* rope.forward, and the T coordinate
// reaches a few thousand, where the cast is the reference's own error floor
// (spec section 9.2). Reproduce the cast at the same point.
// RMSNorm over the head dimension, applied per head to `[rows, heads, dim]`.
// Used for q_norm/k_norm, which normalise over the 128-wide head dim and not
// over the 7168-wide concatenation. Applied **before** RoPE — reversing the
// two is a real, silent quality bug (spec section 4.3).
void launch_head_rmsnorm(__nv_bfloat16* x, const __nv_bfloat16* w, int rows, int heads, int dim,
                         float eps, cudaStream_t stream);

// --- row permutation --------------------------------------------------------

// dst[i, :] = src[index[i], :]
void launch_gather_rows(const __nv_bfloat16* src, const int32_t* index, __nv_bfloat16* dst,
                        int n, int dim, cudaStream_t stream);
void launch_gather_rows_f32(const float* src, const int32_t* index, float* dst, int n, int dim,
                            cudaStream_t stream);

// dst[index[i], :] = src[i, :]
void launch_scatter_rows(const __nv_bfloat16* src, const int32_t* index, __nv_bfloat16* dst,
                         int n, int dim, cudaStream_t stream);
void launch_scatter_rows_f32(const float* src, const int32_t* index, float* dst, int n, int dim,
                             cudaStream_t stream);

// dst[index[i], :] += src[i, :], used by Qwen DeepStack. Indices must be
// unique; accumulation is fp32 and rounded once to bf16.
void launch_scatter_add_rows(const __nv_bfloat16* src, const int32_t* index,
                             __nv_bfloat16* dst, int n, int dim, cudaStream_t stream);

// Concatenates each four consecutive merge-group-major patch rows. The host
// patchifier guarantees those rows describe a 2x2 spatial group.
void launch_merge_four_rows(const __nv_bfloat16* src, __nv_bfloat16* dst,
                            int groups, int dim, cudaStream_t stream);

// --- elementwise ------------------------------------------------------------

void launch_add(const float* a, const float* b, float* out, size_t n, cudaStream_t stream);
void launch_add_bf16(__nv_bfloat16* x, const __nv_bfloat16* branch, size_t n,
                     cudaStream_t stream);
// out = a - b. The block cache's delta capture (dit/block_cache.h): `a` is the
// residual stream after a span of blocks, `b` the state saved before it.
//
// **`out` may alias `b` exactly, and must not alias `a`.** Elementwise at a
// single index, so writing over `b` is safe and is what the block cache does —
// it parks the "before" state in the delta buffer and subtracts in place, which
// is what keeps the feature to one residual-stream-sized buffer instead of two.
// *Exactly*, not partially: `out == b + k` for nonzero `k` is a genuine
// cross-thread race, because one thread's write then lands on another's unread
// input. `a` is the live stream the rest of the stack runs on, is marked
// `__restrict__`, and aliasing it is undefined rather than merely wrong — so
// the launcher rejects `out == a` instead of trusting this paragraph.
//
// All three pointers must be 16-byte aligned to take the vectorised path; the
// launcher checks and falls back to a scalar kernel rather than faulting, so a
// caller passing a row offset gets a slower kernel and not a dead context.
void launch_sub_bf16(const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* out, size_t n,
                     cudaStream_t stream);
void launch_axpby(const float* x, float a, const float* y, float b, float* out, size_t n,
                  cudaStream_t stream);

}  // namespace vidfab::cuda
