#pragma once

#include "slopfab/vulkan/tensor.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tensor_validation.h"
#include "sage_selection.h"
#include "slopfab/attention.h"
#include "slopfab/text/limits.h"
#include "slopfab/vae/audio_primitives.h"
#include "slopfab/vulkan/compute.h"
#include "slopfab/vulkan/gemm.h"
#include "slopfab/vulkan/linear.h"
#include "slopfab/vulkan/vsa_attention.h"


namespace slopfab::vulkan {
namespace tensor_detail {

constexpr uint64_t kMaxExactNormDimension = 1ull << 24;
enum class VaePointwiseOperation : uint32_t { kResidual, kSwiglu, kDenorm };
constexpr uint32_t kH3AttentionLocalSize = 1024;
constexpr uint32_t kFastH3AttentionLocalSize = 256;
constexpr uint32_t kFastH3AttentionQueryTile = 32;
// The pinned module declares 99,328 bytes across phase-aliased workgroup
// arrays. NVIDIA 610.88 lowers those nonoverlapping phases under its reported
// 48 KiB core limit; pipeline creation remains the final module-resource gate.
constexpr uint32_t kH3AttentionMinReportedSharedBytes = 49152;

inline uint64_t checked_multiply(uint64_t a, uint64_t b, const char* operation) {
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
    throw std::overflow_error(std::string("vulkan tensor: ") + operation +
                              " shape overflow");
  }
  return a * b;
}

inline void write_zero_bytes(Buffer& buffer, uint64_t offset, uint64_t bytes) {
  static constexpr std::array<uint8_t, 256> zeros{};
  while (bytes != 0) {
    const uint64_t chunk = std::min<uint64_t>(bytes, zeros.size());
    buffer.write(offset, zeros.data(), chunk);
    offset += chunk;
    bytes -= chunk;
  }
}

inline uintptr_t next_context_identity() {
  static std::atomic<uintptr_t> next{1};
  uintptr_t result = next.load(std::memory_order_relaxed);
  for (;;) {
    if (result == 0 || result == std::numeric_limits<uintptr_t>::max()) {
      throw std::overflow_error("vulkan tensor: context identity space exhausted");
    }
    if (next.compare_exchange_weak(result, result + 1,
                                   std::memory_order_relaxed)) {
      return result;
    }
  }
}

inline bool known_exact_cooperative_gemm_device(const DeviceInfo& info) {
  static constexpr uint8_t kDriverUuid[16] = {
      0x86, 0x90, 0xf1, 0xc8, 0x0a, 0x3f, 0x54, 0x99,
      0x9b, 0xf6, 0xea, 0x2a, 0xee, 0x51, 0x56, 0x02};
  return info.vendor_id == 0x10de && info.device_id == 0x2b85 &&
      info.driver_version == 0x98960000 && info.subgroup_size == 32 &&
      std::memcmp(info.driver_uuid, kDriverUuid, sizeof(kDriverUuid)) == 0 &&
      info.cooperative_matrix_enabled && info.storage_buffer_16bit_enabled;
}

inline bool known_exact_cooperative_bf16_gemm_device(const DeviceInfo& info) {
  return known_exact_cooperative_gemm_device(info) &&
      info.shader_bfloat16_type && info.shader_bfloat16_cooperative_matrix &&
      info.cooperative_matrix_bf16_f32_16x16x16;
}

inline bool known_exact_cooperative_f16_gemm_device(const DeviceInfo& info) {
  return known_exact_cooperative_gemm_device(info) &&
      info.shader_float16_enabled &&
      info.cooperative_matrix_f16_f32_16x16x16;
}

inline bool known_exact_blocked_attention_device(const DeviceInfo& info) {
  return detail::known_exact_vae_norm_device(
             info.vendor_id, info.device_id, info.driver_version) &&
      info.fp32_signed_zero_inf_nan_preserve && info.shader_int64_enabled;
}

inline bool known_exact_causal_gqa_attention_device(const DeviceInfo& info) {
  return known_exact_blocked_attention_device(info);
}

inline bool known_exact_h3_attention_device(const DeviceInfo& info) {
  // Named separately because H3's direct-BF16/FP16-V contract and checked
  // shader artifacts can evolve independently of the prepared-FP16 plan.
  return known_exact_blocked_attention_device(info) &&
      known_exact_cooperative_gemm_device(info) && info.shader_float16_enabled &&
      info.shader_bfloat16_type && info.shader_bfloat16_cooperative_matrix &&
      info.cooperative_matrix_bf16_f32_16x16x16 &&
      info.cooperative_matrix_f16_f32_16x16x16 &&
      info.max_compute_workgroup_invocations >= kH3AttentionLocalSize &&
      info.max_compute_workgroup_size[0] >= kH3AttentionLocalSize &&
      info.max_compute_shared_memory_bytes >= kH3AttentionMinReportedSharedBytes;
}

inline bool fast_h3_attention_device(const DeviceInfo& info) {
  return info.cooperative_matrix_enabled && info.shader_float16_enabled &&
      info.storage_buffer_16bit_enabled && info.subgroup_size == 32 &&
      info.compute_subgroup_shuffle &&
      info.cooperative_matrix_f16_f32_16x16x16 &&
      info.max_compute_workgroup_invocations >= kFastH3AttentionLocalSize &&
      info.max_compute_workgroup_size[0] >= kFastH3AttentionLocalSize;
}

}  // namespace tensor_detail
using namespace tensor_detail;

struct DeviceTensor::Impl {
  Buffer buffer;
  TensorLayout layout;
  ScalarType type = ScalarType::kFloat32;
  uint64_t logical_bytes = 0;
  uintptr_t context = 0;
  uintptr_t identity = next_context_identity();
  bool has_access = false;
  BufferAccess access = BufferAccess::kTransferWrite;
};

struct LinearWeight::Impl {
  LinearWeightFormat format = LinearWeightFormat::kBFloat16;
  uint32_t out_features = 0;
  uint32_t in_features = 0;
  uint64_t stored_bytes = 0;
  uint64_t resident_bytes = 0;
  bool has_fp8_input_scale = false;
  float fp8_input_scale = 0.0f;
  bool full_precision_matrix_mult = false;
  float global_scale = 1.0f;
  float nf4_nested_offset = 0.0f;
  uint32_t nf4_block_size = 64;
  uint32_t nf4_nested_block_size = 256;
  bool convrot = false;
  uint32_t convrot_group = 256;
  DeviceTensor data;
  DeviceTensor weight_scale;
  DeviceTensor block_scale;
  DeviceTensor nf4_absmax;
  DeviceTensor nf4_quant_map;
  DeviceTensor nf4_nested_quant_map;
  DeviceTensor nf4_nested_absmax;
  DeviceTensor pre_quant_scale;
};

struct DenseGemmPlan::Impl {
  std::shared_ptr<TensorContext::Impl> owner;
  DenseGemmPlanDesc desc;
};

struct BlockedAttentionPlan::Impl {
  std::shared_ptr<TensorContext::Impl> owner;
  BlockedAttentionPlanDesc desc;
};

struct H3AttentionRanges::Impl {
  std::shared_ptr<TensorContext::Impl> owner;
  DeviceTensor tensor;
  uint32_t sequence = 0;
  uint32_t query_tiles = 0;
  uint64_t content_hash = 0;
  uintptr_t generation = next_context_identity();
};

struct H3AttentionPlan::Impl {
  std::shared_ptr<TensorContext::Impl> owner;
  H3AttentionPlanDesc desc;
  DeviceTensor quantized_query, quantized_key, sage_aux, prepared_value;
  SageAttentionConfiguration sage;
};

struct VsaAttentionPlan::Impl {
  std::shared_ptr<TensorContext::Impl> owner;
  uint32_t sequence = 0, heads = 0, dim = 0, tiles = 0, prefix = 0, padded = 0;
  DeviceTensor geometry, pooled, mask, compressed;
  uintptr_t batch_id = 0, output_id = 0;
};

struct CausalGQAAttentionPlan::Impl {
  std::shared_ptr<TensorContext::Impl> owner;
  CausalGQAAttentionPlanDesc desc;
};

struct PreparedAttentionInputs::Impl {
  std::shared_ptr<TensorContext::Impl> owner;
  DeviceTensor query;
  DeviceTensor key;
  DeviceTensor value;
  BlockedAttentionPlanDesc desc;
  uintptr_t batch_id = 0;
  uint64_t generation = 0;
  uint64_t reserved_bytes = 0;
  std::array<uintptr_t, 3> source_id{};
};

struct PreparedF16Activation::Impl {
  std::shared_ptr<TensorContext::Impl> owner;
  DeviceTensor tensor;
  uint32_t max_rows = 0;
  uint32_t in_features = 0;
  uint64_t generation = 0;
};

struct StreamedNVFP4WeightCache::Impl {
  std::shared_ptr<TensorContext::Impl> owner;
  DeviceTensor dense;
  uint64_t capacity_elements = 0;
  uint64_t generation = 0;
  uintptr_t batch_id = 0;
  uint32_t out_features = 0;
  uint32_t in_features = 0;
};

struct TensorWorkspace::Impl {
  BufferPool pool;
  Buffer buffer;
  uint64_t capacity = 0;
  uint64_t cursor = 0;
  uint64_t generation = 1;
  uintptr_t context = next_context_identity();

  Impl(const Device& input, uint64_t block_bytes)
      : pool(input, block_bytes) {}
};

}  // namespace slopfab::vulkan
