#include "vidfab/cuda/qwen_vision.cuh"

#include <stdexcept>

#include "vidfab/cuda/attention.cuh"
#include "vidfab/cuda/device.h"
#include "vidfab/cuda/nn_kernels.cuh"

namespace vidfab::cuda {
namespace {
__global__ void split_qkv_kernel(const __nv_bfloat16* qkv, __nv_bfloat16* q,
                                 __nv_bfloat16* k, __nv_bfloat16* v, int hidden) {
  const int row = blockIdx.x;
  const int d = blockIdx.y * blockDim.x + threadIdx.x;
  if (d < hidden) {
    const size_t src = static_cast<size_t>(row) * 3 * hidden;
    const size_t dst = static_cast<size_t>(row) * hidden + d;
    q[dst] = qkv[src + d]; k[dst] = qkv[src + hidden + d];
    v[dst] = qkv[src + 2 * hidden + d];
  }
}

__global__ void add_positions_kernel(__nv_bfloat16* x, const __nv_bfloat16* table,
                                     const int32_t* index, int hidden) {
  const int r = blockIdx.x, d = blockIdx.y * blockDim.x + threadIdx.x;
  if (d < hidden) {
    const size_t at = static_cast<size_t>(r) * hidden + d;
    x[at] = __float2bfloat16(__bfloat162float(x[at]) +
                            __bfloat162float(table[static_cast<size_t>(index[r]) * hidden + d]));
  }
}
}  // namespace

void qwen_vision_attention(cublasHandle_t handle, cudaStream_t stream,
                           const __nv_bfloat16* qkv, const float* cos,
                           const float* sin, __nv_bfloat16* q,
                           __nv_bfloat16* k, __nv_bfloat16* v,
                           __nv_bfloat16* out, int rows, int heads,
                           int head_dim, Workspace& ws) {
  if (rows <= 0 || heads <= 0 || head_dim <= 0)
    throw std::runtime_error("qwen vision attention: invalid dimensions");
  const int hidden = heads * head_dim;
  split_qkv_kernel<<<dim3(rows, (hidden + 255) / 256), 256, 0, stream>>>(qkv, q, k, v, hidden);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
  launch_rope_neox(q, cos, sin, rows, heads, head_dim, stream);
  launch_rope_neox(k, cos, sin, rows, heads, head_dim, stream);
  AttentionConfig cfg;
  cfg.seq_len = rows; cfg.num_heads = heads; cfg.head_dim = head_dim;
  // head_dim=72 deliberately selects the numerically-pinned blocked backend.
  const auto backend = attention_preferred_backend(cfg);
  ws.reserve(attention_workspace_bytes(cfg, backend));
  attention_forward(handle, stream, q, k, v, out, cfg, backend, ws);
}

void qwen_vision_block_forward(cublasHandle_t handle, cudaStream_t stream,
                               LinearRunner& linear, const QwenVisionBlockWeights& w,
                               const float* cos, const float* sin, __nv_bfloat16* x,
                               int rows, QwenVisionBlockScratch s, Workspace& ws,
                               float eps) {
  constexpr int hidden = 1152, heads = 16, head_dim = 72, intermediate = 4304;
  if (!x || !s.normed || !s.qkv || !s.q || !s.k || !s.v || !s.branch || !s.mlp)
    throw std::runtime_error("qwen vision block: null activation scratch");
  launch_layernorm_affine(x, w.norm1_weight, w.norm1_bias, s.normed, rows, hidden, eps, stream);
  linear.forward(w.qkv, s.normed, rows, s.qkv, ws);
  qwen_vision_attention(handle, stream, s.qkv, cos, sin, s.q, s.k, s.v, s.branch,
                        rows, heads, head_dim, ws);
  linear.forward(w.attention_out, s.branch, rows, s.normed, ws);
  launch_add_bf16(x, s.normed, static_cast<size_t>(rows) * hidden, stream);
  launch_layernorm_affine(x, w.norm2_weight, w.norm2_bias, s.normed, rows, hidden, eps, stream);
  linear.forward(w.mlp_fc1, s.normed, rows, s.mlp, ws);
  launch_gelu_tanh(s.mlp, static_cast<size_t>(rows) * intermediate, stream);
  linear.forward(w.mlp_fc2, s.mlp, rows, s.normed, ws);
  launch_add_bf16(x, s.normed, static_cast<size_t>(rows) * hidden, stream);
}

void qwen_vision_patch_embed(LinearRunner& linear, const QuantWeight& projection,
                             const __nv_bfloat16* pixels, const __nv_bfloat16* pos,
                             const int32_t* index, __nv_bfloat16* x, int rows,
                             Workspace& ws, cudaStream_t stream) {
  constexpr int hidden = 1152;
  linear.forward(projection, pixels, rows, x, ws);
  add_positions_kernel<<<dim3(rows, (hidden + 255) / 256), 256, 0, stream>>>(x, pos, index, hidden);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void qwen_vision_merger_forward(cudaStream_t stream, LinearRunner& linear,
                                const QwenVisionMergerWeights& w, const __nv_bfloat16* x,
                                __nv_bfloat16* normed, __nv_bfloat16* merged,
                                __nv_bfloat16* hidden, __nv_bfloat16* output,
                                int rows, Workspace& ws, float eps) {
  constexpr int dim = 1152, merged_dim = 4608;
  if (rows <= 0 || rows % 4) throw std::runtime_error("qwen vision merger: rows not divisible by four");
  const int groups = rows / 4;
  if (w.norm_before_merge) {
    launch_layernorm_affine(x, w.norm_weight, w.norm_bias, normed, rows, dim, eps, stream);
    launch_merge_four_rows(normed, merged, groups, dim, stream);
  } else {
    launch_merge_four_rows(x, merged, groups, dim, stream);
    launch_layernorm_affine(merged, w.norm_weight, w.norm_bias, normed, groups, merged_dim,
                            eps, stream);
    merged = normed;
  }
  linear.forward(w.fc1, merged, groups, hidden, ws);
  launch_gelu_tanh(hidden, static_cast<size_t>(groups) * merged_dim, stream);
  linear.forward(w.fc2, hidden, groups, output, ws);
}

void qwen_vision_tower_forward(cublasHandle_t handle, cudaStream_t stream,
                               LinearRunner& linear, const QwenVisionBlockWeights* blocks,
                               const QwenVisionMergerWeights& main_merger,
                               const QwenVisionMergerWeights* deep,
                               const float* cos, const float* sin, __nv_bfloat16* x,
                               int rows, QwenVisionBlockScratch scratch,
                               __nv_bfloat16* merger_normed, __nv_bfloat16* merged,
                               __nv_bfloat16* merger_hidden, __nv_bfloat16* output,
                               __nv_bfloat16** deep_out, Workspace& ws) {
  int deep_index = 0;
  for (int i = 0; i < 27; ++i) {
    qwen_vision_block_forward(handle, stream, linear, blocks[i], cos, sin, x, rows, scratch, ws);
    if (i == 8 || i == 16 || i == 24) {
      qwen_vision_merger_forward(stream, linear, deep[deep_index], x, merger_normed, merged,
                                 merger_hidden, deep_out[deep_index], rows, ws);
      ++deep_index;
    }
  }
  qwen_vision_merger_forward(stream, linear, main_merger, x, merger_normed, merged,
                             merger_hidden, output, rows, ws);
}

}  // namespace vidfab::cuda
