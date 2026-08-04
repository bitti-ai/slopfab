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
  // `query_block * seq_len * 4` bytes per concurrent head. Tuning knob only —
  // results must not depend on it, and a unit test pins that.
  int query_block = 1024;

  float effective_scale() const;
};

enum class AttentionBackend {
  // Two cuBLAS GEMMs per query block plus an online softmax. Correct, memory
  // bounded, and roughly half of peak because the score tile round-trips
  // through HBM. This is the reference implementation and the one the unit
  // tests pin.
  kBlocked,
  // Single fused kernel keeping the score tile in shared memory / registers.
  kFused,
};

// Workspace required for a given configuration and backend.
size_t attention_workspace_bytes(const AttentionConfig& cfg, AttentionBackend backend);

// out[seq, heads*head_dim] = softmax(q k^T * scale) v, per head.
//
// `out` may not alias q, k or v.
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
