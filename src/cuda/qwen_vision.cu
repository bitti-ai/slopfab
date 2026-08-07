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

}  // namespace vidfab::cuda
