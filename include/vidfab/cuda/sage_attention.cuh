#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace vidfab::cuda {

struct AttentionConfig;
class Workspace;

enum class Sage2KernelVariant {
  kUnsupported,
  kAmpereFp16,
  kBlackwellFp8,
};

// Pure architecture selection kept separate from the device query so adding
// SM89 later is one explicit dispatch case rather than another scattered gate.
Sage2KernelVariant sage2_variant_for_compute_capability(int capability) noexcept;

bool sage2_supported(const AttentionConfig& cfg, int device, const char** reason = nullptr);
size_t sage2_workspace_bytes(const AttentionConfig& cfg, int num_kv_heads);
void sage2_attention_forward(cudaStream_t stream, const __nv_bfloat16* q,
                             const __nv_bfloat16* k, const __nv_bfloat16* v,
                             __nv_bfloat16* out, const AttentionConfig& cfg,
                             int num_kv_heads, Workspace& ws);

}  // namespace vidfab::cuda
