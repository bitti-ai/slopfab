#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace slopfab::cuda {

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

// Dynamic shared memory required by the vendored Ampere FP16-P/V kernel. Q,
// K, and V occupy the arena concurrently; output later reuses it from byte 0.
// Keep this formula aligned with the pinned upstream launch wrapper.
constexpr size_t sage2_ampere_dynamic_smem_bytes(int head_dim) noexcept {
  const size_t qkv =
      static_cast<size_t>(128 + 64) * head_dim + static_cast<size_t>(64) * head_dim * 2;
  const size_t output = static_cast<size_t>(128) * head_dim * 2;
  return qkv > output ? qkv : output;
}

static_assert(sage2_ampere_dynamic_smem_bytes(64) == 20 * 1024);
static_assert(sage2_ampere_dynamic_smem_bytes(128) == 40 * 1024);

bool sage2_supported(const AttentionConfig& cfg, int device, const char** reason = nullptr);
size_t sage2_workspace_bytes(const AttentionConfig& cfg, int num_kv_heads);
void sage2_attention_forward(cudaStream_t stream, const __nv_bfloat16* q, const __nv_bfloat16* k,
                             const __nv_bfloat16* v, __nv_bfloat16* out, const AttentionConfig& cfg,
                             int num_kv_heads, Workspace& ws);

} // namespace slopfab::cuda
