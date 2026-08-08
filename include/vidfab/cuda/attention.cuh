// Full self-attention over one packed sequence.
//
// This is the dominant cost of the whole pipeline and the one kernel whose
// naive form does not fit in memory. For the default 124-frame request the
// packed sequence is ~37.7k rows; a materialised S x S score matrix is 2.8 GB
// **per head** at fp32, and there are 56 heads and 50 layers. The scores must
// therefore never exist in full: attention is computed in query blocks with an
// online (streaming) softmax, exactly as FlashAttention does.
//
// There is no mask, no causality and no cross-attention anywhere in MiniMax
// H3 (spec section 2.2). The t2va path never pads, so no attention mask is
// synthesised — and must not be, since an all-zero float mask hard-fails the
// flash backends the reference dispatches to.
//
// **Layout.** q, k, v are `[seq, heads * head_dim]` row-major bf16 — that is,
// exactly the shape `qkv_proj` produces after splitting on contiguous thirds,
// with no transpose. Per-head slices are strided views (`ld = heads*head_dim`,
// batch stride `head_dim`), which cuBLAS strided-batched GEMM consumes
// directly, so the blocked implementation needs no layout change.
#pragma once

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "vidfab/cuda/workspace.cuh"

namespace vidfab::cuda {

struct AttentionConfig {
  int seq_len = 0;
  int num_heads = 56;
  int head_dim = 128;

  // Defaults to 1/sqrt(head_dim); nothing in H3 overrides it.
  float scale = 0.0f;

  // Rows of queries processed per block. Sets the peak score-tile footprint:
  // `query_block * key_block * 2` bytes per concurrent head, with `key_block`
  // chosen to hold that inside a fixed budget. Tuning knob only — results must
  // not depend on it, and a unit test pins that.
  int query_block = 1024;

  // Keys processed per online-softmax step. 0 means "as large as the score tile
  // budget allows", which is what production wants and what every caller should
  // leave it at.
  //
  // It is settable only because the budget is 640 MiB: at any test-sized shape
  // the chosen key block covers the whole sequence in one step, so the running
  // max, the correction factor and the accumulator rescale — the part of this
  // algorithm that fails silently — would never execute in a unit test. Like
  // `query_block`, results must not depend on it, and a test pins that.
  int key_block = 0;

  // Frame-banded attention, kFused only. Device pointer to four int32 per query
  // tile -- `[lo0, hi0, lo1, hi1]`, two half-open key ranges -- or nullptr for
  // full attention, which is the default and what every existing caller gets.
  //
  // Two ranges rather than one because the packed sequence is
  // `[ text | conditions | audio | video ]`: text and audio sit at the front,
  // so a band around a late video frame would exclude the conditioning entirely
  // unless it is carried as its own range. Build it with
  // `vidfab::dit::build_banded_key_ranges`, passing `attention_fused_query_tile()`
  // and `attention_fused_key_align()` so the ranges match the kernel's own
  // tiling; the bounds must be multiples of the latter.
  //
  // This is a lossy approximation behind a default-off flag. It is not a tuning
  // knob like the two above: results *do* depend on it, by construction.
  const int32_t* band_ranges = nullptr;

  // Sol-Attn routing. Keys in [0, exact_prefix) are always evaluated exactly;
  // this keeps H3's text/condition/audio prefix out of the approximation.
  // sol_beta is the standardized proxy-score cutoff (1.28155 ~= 90% routed
  // away for a normal distribution).
  int exact_prefix = 0;
  float sol_beta = 1.0f;  // Official H3 diagonal-estimator cutoff.
  // Optional device counters [exact routes, approximate routes]. Null keeps
  // production routing free of diagnostic atomics.
  unsigned long long* sol_route_counts = nullptr;
  // Optional host output [pool, key stats, thresholds, main] in milliseconds.
  // Enabling this synchronizes the stream and is for solbench only.
  float* sol_phase_ms = nullptr;
  // Opt-in SM120 experimental exact mainloop. Unsupported/ragged shapes fall
  // through to the established Sol kernel.
  bool sol_pipeline = false;

  float effective_scale() const;
};

// The fused kernel's query tile and key-block granularity. Exposed so a caller
// can build `band_ranges` that line up with the kernel's own loop rather than
// hard-coding constants that live in the .cu.
int attention_fused_query_tile();
int attention_fused_key_align();

enum class AttentionBackend {
  // Two cuBLAS GEMMs per query block plus an online softmax. Correct, memory
  // bounded, and roughly half of peak because the score tile round-trips
  // through HBM. This is the reference implementation and the one the unit
  // tests pin.
  kBlocked,
  // Single fused kernel keeping the score tile in shared memory / registers.
  kFused,
  kSage2,
  kSol,
};

// The fastest backend that can run this configuration. kFused is instantiated
// for head_dim 64 and 128 and falls back to kBlocked for anything else.
//
// Callers must route both `attention_workspace_bytes` and `attention_forward`
// through this rather than choosing separately: the two backends have very
// different workspace requirements (kFused needs none), so a caller that sized
// for one and dispatched to the other would either waste a gigabyte or read
// past its arena.
AttentionBackend attention_preferred_backend(const AttentionConfig& cfg);

// Workspace required for a given configuration and backend.
size_t attention_workspace_bytes(const AttentionConfig& cfg, AttentionBackend backend);

// out[seq, heads*head_dim] = softmax(q k^T * scale) v, per head.
//
// `out` may not alias q, k or v.
//
// **Range limit.** The score tile is fp16 (see attention.cu for why that is
// more accurate than fp32-scores-plus-bf16-probabilities, not less). fp16
// saturates at 65504, and an infinity there becomes a NaN in the softmax rather
// than a large number. H3 scores land at ±10..30 because q_norm/k_norm fix the
// per-head RMS and RoPE preserves norm; a caller feeding unnormalised queries
// at a magnitude where `|q.k| * scale` could approach 6e4 needs a different
// kernel, not a bigger tolerance.
void attention_forward(cublasHandle_t handle, cudaStream_t stream, const __nv_bfloat16* q,
                       const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* out,
                       const AttentionConfig& cfg, AttentionBackend backend, Workspace& ws);

// Grouped-query variant for the Qwen3-VL encoder: 64 query heads share 8
// key/value heads, so `k` and `v` are `[seq, num_kv_heads * head_dim]` and
// query head `h` reads kv head `h / (num_heads / num_kv_heads)`.
void attention_forward_gqa(cublasHandle_t handle, cudaStream_t stream, const __nv_bfloat16* q,
                           const __nv_bfloat16* k, const __nv_bfloat16* v, __nv_bfloat16* out,
                           const AttentionConfig& cfg, int num_kv_heads,
                           AttentionBackend backend, Workspace& ws);

}  // namespace vidfab::cuda
