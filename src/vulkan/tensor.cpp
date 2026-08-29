#include "vidfab/vulkan/tensor.h"

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

#include "embedded_tensor_spv.h"
#include "tensor_validation.h"
#include "vidfab/attention.h"
#include "vidfab/vae/audio_primitives.h"
#include "vidfab/vulkan/compute.h"
#include "vidfab/vulkan/gemm.h"
#include "vidfab/vulkan/linear.h"

namespace vidfab::vulkan {
namespace {

constexpr uint64_t kMaxExactNormDimension = 1ull << 24;
enum class VaePointwiseOperation : uint32_t { kResidual, kSwiglu, kDenorm };
constexpr uint32_t kH3AttentionLocalSize = 1024;
// The pinned module declares 99,328 bytes across phase-aliased workgroup
// arrays. NVIDIA 610.88 lowers those nonoverlapping phases under its reported
// 48 KiB core limit; pipeline creation remains the final module-resource gate.
constexpr uint32_t kH3AttentionMinReportedSharedBytes = 49152;

uint64_t checked_multiply(uint64_t a, uint64_t b, const char* operation) {
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
    throw std::overflow_error(std::string("vulkan tensor: ") + operation +
                              " shape overflow");
  }
  return a * b;
}

uintptr_t next_context_identity() {
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

bool known_exact_cooperative_gemm_device(const DeviceInfo& info) {
  static constexpr uint8_t kDriverUuid[16] = {
      0x86, 0x90, 0xf1, 0xc8, 0x0a, 0x3f, 0x54, 0x99,
      0x9b, 0xf6, 0xea, 0x2a, 0xee, 0x51, 0x56, 0x02};
  return info.vendor_id == 0x10de && info.device_id == 0x2b85 &&
      info.driver_version == 0x98960000 && info.subgroup_size == 32 &&
      std::memcmp(info.driver_uuid, kDriverUuid, sizeof(kDriverUuid)) == 0 &&
      info.cooperative_matrix_enabled && info.storage_buffer_16bit_enabled;
}

bool known_exact_cooperative_bf16_gemm_device(const DeviceInfo& info) {
  return known_exact_cooperative_gemm_device(info) &&
      info.shader_bfloat16_type && info.shader_bfloat16_cooperative_matrix &&
      info.cooperative_matrix_bf16_f32_16x16x16;
}

bool known_exact_cooperative_f16_gemm_device(const DeviceInfo& info) {
  return known_exact_cooperative_gemm_device(info) &&
      info.shader_float16_enabled &&
      info.cooperative_matrix_f16_f32_16x16x16;
}

bool known_exact_blocked_attention_device(const DeviceInfo& info) {
  return detail::known_exact_vae_norm_device(
             info.vendor_id, info.device_id, info.driver_version) &&
      info.fp32_signed_zero_inf_nan_preserve && info.shader_int64_enabled;
}

bool known_exact_causal_gqa_attention_device(const DeviceInfo& info) {
  return known_exact_blocked_attention_device(info);
}

bool known_exact_h3_attention_device(const DeviceInfo& info) {
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

}  // namespace

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

struct TensorContext::Impl {
  struct Parameters {
    uint32_t op = 0;
    uint32_t count = 0;
    uint32_t p[6] = {};
  };
  struct NormParameters {
    uint32_t rows = 0;
    uint32_t dim = 0;
    uint32_t epsilon_bits = 0;
    uint32_t mod_rows = 0;
  };
  struct VaeRopeParameters {
    uint32_t sequence = 0;
    uint32_t heads = 0;
    uint32_t head_dim = 64;
    uint32_t rope_dim = 48;
    uint32_t num_patches = 0;
    uint32_t epsilon_bits = 0;
    uint32_t unused[2] = {};
  };
  struct AudioParameters {
    uint32_t op = 0;
    uint32_t batch = 0;
    uint32_t in_channels = 0;
    uint32_t out_channels = 0;
    uint32_t length_in = 0;
    uint32_t length_out = 0;
    uint32_t kernel = 0;
    uint32_t padding_or_stride = 0;
    uint32_t dilation_or_padding = 0;
    uint32_t count = 0;
    uint32_t groups_x = 0;
    uint32_t scalar_bits = 0;
  };
  static_assert(sizeof(AudioParameters) == 48);
  struct DitParameters {
    uint32_t op = 0;
    uint32_t rows = 0;
    uint32_t dim = 0;
    uint32_t mod_rows = 0;
    uint32_t num_t = 0;
    uint32_t num_modality = 0;
    uint32_t num_param = 0;
    uint32_t rank = 0;
    uint32_t count = 0;
    uint32_t groups_x = 0;
    uint32_t unused[2] = {};
  };
  static_assert(sizeof(DitParameters) == 48);
  struct WeightParameters {
    uint32_t op = 0;
    uint32_t count = 0;
    uint32_t out_features = 0;
    uint32_t in_features = 0;
    uint32_t block_size = 0;
    uint32_t nested_block_size = 0;
    uint32_t scalar_bits = 0;
    uint32_t group = 0;
  };
  struct GemmParameters {
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t in_features = 0;
    uint32_t input_row_offset = 0;
    uint32_t output_row_offset = 0;
    uint32_t mode = 0;
    uint32_t unused[2] = {};
  };
  struct GemmPrepareParameters {
    uint32_t rows = 0;
    uint32_t in_features = 0;
    uint32_t input_row_offset = 0;
    uint32_t groups_x = 0;
  };
  struct AttentionParameters {
    uint32_t sequence = 0;
    uint32_t heads = 0;
    uint32_t head_dim = 0;
    uint32_t scale_bits = 0;
    uint32_t query_row_offset = 0;
    uint32_t output_row_offset = 0;
    uint32_t rows = 0;
    uint32_t reserved = 0;
  };
  struct CausalGQAAttentionParameters {
    uint32_t sequence = 0;
    uint32_t query_heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    uint32_t scale_bits = 0;
    uint32_t query_row_offset = 0;
    uint32_t output_row_offset = 0;
    uint32_t rows = 0;
  };
  uint32_t max_batch_operators = 0;
  ComputeContext commands;
  BufferPool pool;
  TensorWorkspace scratch;
  ComputePipeline ops_pipeline;
  ComputePipeline rope_pipeline;
  ComputePipeline vae_rope_pipeline;
  ComputePipeline weight_pipeline;
  ComputePipeline gemm_pipeline;
  ComputePipeline gemm_coop_pipeline;
  ComputePipeline gemm_prepare_pipeline;
  ComputePipeline gemm_coop_f16_pipeline;
  ComputePipeline attention_blocked_pipeline;
  ComputePipeline attention_h3_pipeline;
  ComputePipeline attention_h3_banded_pipeline;
  ComputePipeline attention_prepare_pipeline;
  ComputePipeline attention_causal_gqa_pipeline;
  ComputePipeline rms_norm_pipeline;
  ComputePipeline layer_norm_pipeline;
  ComputePipeline bf16_rms_block_pipeline;
  ComputePipeline bf16_rms_narrow_pipeline;
  ComputePipeline bf16_layer_pipeline;
  ComputePipeline bf16_mod_pipeline;
  ComputePipeline fp32_mod_pipeline;
  ComputePipeline group_norm_pipeline;
  ComputePipeline vae_residual_pipeline;
  ComputePipeline vae_swiglu_pipeline;
  ComputePipeline vae_denorm_pipeline;
  ComputePipeline audio_pipeline;
  ComputePipeline dit_pipeline;
  Buffer upload_buffer;
  Buffer readback_buffer;
  uint64_t staging_capacity = 0;
  std::vector<StorageBinding> ops_bindings;
  std::vector<StorageBinding> vae_pointwise_bindings;
  std::vector<StorageBinding> norm_bindings;
  std::vector<StorageBinding> mod_bindings;
  std::vector<StorageBinding> vae_rope_bindings;
  std::vector<StorageBinding> audio_bindings;
  std::vector<StorageBinding> dit_bindings;
  std::vector<StorageBinding> weight_bindings;
  std::vector<StorageBinding> gemm_bindings;
  std::vector<StorageBinding> gemm_prepare_bindings;
  std::vector<StorageBinding> attention_bindings;
  std::vector<StorageBinding> attention_h3_bindings;
  std::vector<StorageBinding> attention_h3_banded_bindings;
  std::vector<StorageBinding> attention_prepare_bindings;
  std::vector<StorageBinding> attention_causal_gqa_bindings;
  bool full_arithmetic_exact = false;
  bool exact_vae_norm = false;
  bool exact_vae_pointwise = false;
  bool exact_audio = false;
  bool exact_dit_pointwise = false;
  bool exact_attention = false;
  bool exact_h3_attention = false;
  bool exact_causal_gqa_attention = false;
  bool cooperative_gemm = false;
  bool cooperative_f16_gemm = false;
  uint32_t max_dispatch_x = 0;
  uint32_t max_dispatch_y = 0;
  uint64_t max_storage_bytes = 0;
  uint64_t storage_binding_alignment = 1;
  std::atomic<bool> recorder_active{false};

  explicit Impl(const Device& input, const TensorContextOptions& tensor_options)
      : max_batch_operators(tensor_options.max_batch_operators),
        commands(input, [&] {
          if (tensor_options.max_batch_operators == 0 ||
              tensor_options.max_batch_operators > 4096) {
            throw std::invalid_argument(
                "vulkan tensor: max_batch_operators must be in [1,4096]");
          }
          ComputeContextOptions options;
          options.max_in_flight = tensor_options.max_in_flight;
          options.max_storage_bindings = 7;
          options.max_compute_binds_per_job = tensor_options.max_batch_operators * 2;
          return options;
        }()),
        pool(input, 4ull << 20),
        scratch(input),
        ops_bindings(3),
        vae_pointwise_bindings(4),
        norm_bindings(4),
        mod_bindings(6),
        vae_rope_bindings(7),
        audio_bindings(5),
        dit_bindings(5),
        weight_bindings(6),
        gemm_bindings(4),
        gemm_prepare_bindings(2),
        attention_bindings(4),
        attention_h3_bindings(4),
        attention_h3_banded_bindings(5),
        attention_prepare_bindings(6),
        attention_causal_gqa_bindings(4) {
    // Device is move-only; the opaque handle is sufficient for identity and
    // every owned Vulkan object already retains the shared device state.
    static_assert(sizeof(detail::kTensorOpsSpirv) % sizeof(uint32_t) == 0);
    if (!input.info().fp32_signed_zero_inf_nan_preserve ||
        !input.info().fp32_rounding_rte) {
      throw std::runtime_error(
          "vulkan tensor: fp32 arithmetic requires signed-zero/Inf/NaN preservation "
          "and round-to-nearest-even");
    }
    full_arithmetic_exact = input.info().fp32_denorm_preserve;
    exact_vae_norm = detail::known_exact_vae_norm_device(
                         input.info().vendor_id, input.info().device_id,
                         input.info().driver_version) &&
                     input.info().fp32_signed_zero_inf_nan_preserve &&
                     input.info().shader_int64_enabled;
    exact_vae_pointwise = detail::known_exact_vae_pointwise_device(
                              input.info().vendor_id, input.info().device_id,
                              input.info().driver_version) &&
                          input.info().fp32_signed_zero_inf_nan_preserve &&
                          input.info().fp32_rounding_rte &&
                          input.info().shader_int64_enabled;
    exact_audio = exact_vae_pointwise;
    exact_dit_pointwise = exact_vae_pointwise;
    exact_attention = known_exact_blocked_attention_device(input.info());
    exact_h3_attention = known_exact_h3_attention_device(input.info());
    exact_causal_gqa_attention =
        known_exact_causal_gqa_attention_device(input.info());
    max_dispatch_x = input.info().max_compute_workgroup_count[0];
    max_dispatch_y = input.info().max_compute_workgroup_count[1];
    max_storage_bytes = input.info().max_storage_buffer_bytes;
    storage_binding_alignment = input.info().min_storage_buffer_offset_alignment;
    const uint8_t* shader = full_arithmetic_exact ? detail::kTensorOpsDenormSpirv
                                           : detail::kTensorOpsSpirv;
    const size_t shader_bytes = full_arithmetic_exact ? sizeof(detail::kTensorOpsDenormSpirv)
                                               : sizeof(detail::kTensorOpsSpirv);
    std::vector<uint32_t> spirv(shader_bytes / sizeof(uint32_t));
    std::memcpy(spirv.data(), shader, shader_bytes);
    ComputePipelineOptions options;
    options.storage_binding_count = 3;
    options.push_constant_bytes = sizeof(Parameters);
    options.local_size[0] = 64;
    ops_pipeline = ComputePipeline::create(input, spirv, options);
    if (exact_vae_pointwise) {
      ComputePipelineOptions pointwise_options = options;
      pointwise_options.storage_binding_count = 4;
      auto make_pointwise = [&](const uint8_t* bytes, size_t byte_count) {
        std::vector<uint32_t> module(byte_count / sizeof(uint32_t));
        std::memcpy(module.data(), bytes, byte_count);
        return ComputePipeline::create(input, module, pointwise_options);
      };
      vae_residual_pipeline = make_pointwise(
          detail::kTensorVaeResidualSpirv,
          sizeof(detail::kTensorVaeResidualSpirv));
      vae_swiglu_pipeline = make_pointwise(
          detail::kTensorVaeSwigluSpirv,
          sizeof(detail::kTensorVaeSwigluSpirv));
      vae_denorm_pipeline = make_pointwise(
          detail::kTensorVaeDenormSpirv,
          sizeof(detail::kTensorVaeDenormSpirv));
    }
    if (exact_audio) {
      std::vector<uint32_t> audio_spirv(
          sizeof(detail::kTensorAudioSpirv) / sizeof(uint32_t));
      std::memcpy(audio_spirv.data(), detail::kTensorAudioSpirv,
                  sizeof(detail::kTensorAudioSpirv));
      ComputePipelineOptions audio_options;
      audio_options.storage_binding_count = 5;
      audio_options.push_constant_bytes = sizeof(AudioParameters);
      audio_options.local_size[0] = 64;
      audio_pipeline = ComputePipeline::create(input, audio_spirv, audio_options);
    }
    if (exact_dit_pointwise) {
      std::vector<uint32_t> dit_spirv(
          sizeof(detail::kTensorDitSpirv) / sizeof(uint32_t));
      std::memcpy(dit_spirv.data(), detail::kTensorDitSpirv,
                  sizeof(detail::kTensorDitSpirv));
      ComputePipelineOptions dit_options;
      dit_options.storage_binding_count = 5;
      dit_options.push_constant_bytes = sizeof(DitParameters);
      dit_options.local_size[0] = 64;
      dit_pipeline = ComputePipeline::create(input, dit_spirv, dit_options);
    }
    const uint8_t* rope_shader = full_arithmetic_exact
        ? detail::kTensorRopeDenormSpirv : detail::kTensorRopeSpirv;
    const size_t rope_shader_bytes = full_arithmetic_exact
        ? sizeof(detail::kTensorRopeDenormSpirv) : sizeof(detail::kTensorRopeSpirv);
    std::vector<uint32_t> rope_spirv(rope_shader_bytes / sizeof(uint32_t));
    std::memcpy(rope_spirv.data(), rope_shader, rope_shader_bytes);
    ComputePipelineOptions rope_options = options;
    rope_options.local_size[0] = 64;
    rope_pipeline = ComputePipeline::create(input, rope_spirv, rope_options);
    const uint8_t* weight_shader = full_arithmetic_exact
        ? detail::kTensorWeightDenormSpirv : detail::kTensorWeightSpirv;
    const size_t weight_shader_bytes = full_arithmetic_exact
        ? sizeof(detail::kTensorWeightDenormSpirv)
        : sizeof(detail::kTensorWeightSpirv);
    std::vector<uint32_t> weight_spirv(weight_shader_bytes / sizeof(uint32_t));
    std::memcpy(weight_spirv.data(), weight_shader, weight_shader_bytes);
    ComputePipelineOptions weight_options;
    weight_options.storage_binding_count = 6;
    weight_options.push_constant_bytes = sizeof(WeightParameters);
    weight_options.local_size[0] = 64;
    weight_pipeline = ComputePipeline::create(input, weight_spirv, weight_options);
    const uint8_t* gemm_shader = full_arithmetic_exact
        ? detail::kTensorGemmDenormSpirv : detail::kTensorGemmSpirv;
    const size_t gemm_shader_bytes = full_arithmetic_exact
        ? sizeof(detail::kTensorGemmDenormSpirv) : sizeof(detail::kTensorGemmSpirv);
    std::vector<uint32_t> gemm_spirv(gemm_shader_bytes / sizeof(uint32_t));
    std::memcpy(gemm_spirv.data(), gemm_shader, gemm_shader_bytes);
    ComputePipelineOptions gemm_options;
    gemm_options.storage_binding_count = 4;
    gemm_options.push_constant_bytes = sizeof(GemmParameters);
    gemm_options.local_size[0] = 8;
    gemm_options.local_size[1] = 8;
    gemm_pipeline = ComputePipeline::create(input, gemm_spirv, gemm_options);
    cooperative_gemm = known_exact_cooperative_bf16_gemm_device(input.info());
    if (cooperative_gemm) {
      const uint8_t* cooperative_shader = full_arithmetic_exact
          ? detail::kTensorGemmCoopDenormSpirv : detail::kTensorGemmCoopSpirv;
      const size_t cooperative_bytes = full_arithmetic_exact
          ? sizeof(detail::kTensorGemmCoopDenormSpirv)
          : sizeof(detail::kTensorGemmCoopSpirv);
      std::vector<uint32_t> cooperative_spirv(cooperative_bytes / sizeof(uint32_t));
      std::memcpy(cooperative_spirv.data(), cooperative_shader, cooperative_bytes);
      ComputePipelineOptions cooperative_options = gemm_options;
      cooperative_options.local_size[0] = 128;
      cooperative_options.local_size[1] = 1;
      gemm_coop_pipeline = ComputePipeline::create(
          input, cooperative_spirv, cooperative_options);
    }
    {
      std::vector<uint32_t> prepare_spirv(
          sizeof(detail::kTensorGemmPrepareSpirv) / sizeof(uint32_t));
      std::memcpy(prepare_spirv.data(), detail::kTensorGemmPrepareSpirv,
                  sizeof(detail::kTensorGemmPrepareSpirv));
      ComputePipelineOptions prepare_options;
      prepare_options.storage_binding_count = 2;
      prepare_options.push_constant_bytes = sizeof(GemmPrepareParameters);
      prepare_options.local_size[0] = 64;
      gemm_prepare_pipeline = ComputePipeline::create(
          input, prepare_spirv, prepare_options);
    }
    cooperative_f16_gemm = known_exact_cooperative_f16_gemm_device(input.info());
    if (cooperative_f16_gemm) {
      const uint8_t* cooperative_shader = full_arithmetic_exact
          ? detail::kTensorGemmCoopF16DenormSpirv
          : detail::kTensorGemmCoopF16Spirv;
      const size_t cooperative_bytes = full_arithmetic_exact
          ? sizeof(detail::kTensorGemmCoopF16DenormSpirv)
          : sizeof(detail::kTensorGemmCoopF16Spirv);
      std::vector<uint32_t> cooperative_spirv(cooperative_bytes / sizeof(uint32_t));
      std::memcpy(cooperative_spirv.data(), cooperative_shader, cooperative_bytes);
      ComputePipelineOptions cooperative_options = gemm_options;
      cooperative_options.local_size[0] = 128;
      cooperative_options.local_size[1] = 1;
      gemm_coop_f16_pipeline = ComputePipeline::create(
          input, cooperative_spirv, cooperative_options);
    }
    ComputePipelineOptions norm_options;
    norm_options.storage_binding_count = 4;
    norm_options.push_constant_bytes = sizeof(NormParameters);
    norm_options.local_size[0] = 256;
    auto make_norm_pipeline = [&](const uint8_t* shader, size_t shader_bytes,
                                  uint32_t bindings = 4, uint32_t local_x = 256,
                                  uint32_t local_y = 1,
                                  uint32_t push_bytes = sizeof(NormParameters)) {
      std::vector<uint32_t> module(shader_bytes / sizeof(uint32_t));
      std::memcpy(module.data(), shader, shader_bytes);
      ComputePipelineOptions selected = norm_options;
      selected.storage_binding_count = bindings;
      selected.local_size[0] = local_x;
      selected.local_size[1] = local_y;
      selected.push_constant_bytes = push_bytes;
      return ComputePipeline::create(input, module, selected);
    };
    if (exact_vae_norm) {
      rms_norm_pipeline = make_norm_pipeline(detail::kTensorRmsNormSpirv,
                                             sizeof(detail::kTensorRmsNormSpirv));
      layer_norm_pipeline = make_norm_pipeline(detail::kTensorLayerNormSpirv,
                                               sizeof(detail::kTensorLayerNormSpirv));
      bf16_rms_block_pipeline = make_norm_pipeline(
          detail::kTensorBf16RmsBlockSpirv,
          sizeof(detail::kTensorBf16RmsBlockSpirv), 3);
      bf16_rms_narrow_pipeline = make_norm_pipeline(
          detail::kTensorBf16RmsNarrowSpirv,
          sizeof(detail::kTensorBf16RmsNarrowSpirv), 3, 32, 8);
      bf16_layer_pipeline = make_norm_pipeline(
          detail::kTensorBf16LayerSpirv, sizeof(detail::kTensorBf16LayerSpirv));
      bf16_mod_pipeline = make_norm_pipeline(
          detail::kTensorBf16ModSpirv, sizeof(detail::kTensorBf16ModSpirv), 6);
      fp32_mod_pipeline = make_norm_pipeline(
          detail::kTensorFp32ModSpirv, sizeof(detail::kTensorFp32ModSpirv), 6);
      group_norm_pipeline = make_norm_pipeline(
          detail::kTensorGroupNormSpirv, sizeof(detail::kTensorGroupNormSpirv));
      vae_rope_pipeline = make_norm_pipeline(
          detail::kTensorVaeRopeSpirv, sizeof(detail::kTensorVaeRopeSpirv), 7,
          32, 1, sizeof(VaeRopeParameters));
    }
    if (exact_attention) {
      attention_blocked_pipeline = make_norm_pipeline(
          detail::kTensorAttentionBlockedSpirv,
          sizeof(detail::kTensorAttentionBlockedSpirv), 4, 128, 1,
          sizeof(AttentionParameters));
      attention_prepare_pipeline = make_norm_pipeline(
          detail::kTensorAttentionPrepareSpirv,
          sizeof(detail::kTensorAttentionPrepareSpirv), 6, 64, 1,
          sizeof(uint32_t));
    }
    if (exact_h3_attention) {
      attention_h3_pipeline = make_norm_pipeline(
          detail::kTensorAttentionH3Spirv,
          sizeof(detail::kTensorAttentionH3Spirv), 4,
          kH3AttentionLocalSize, 1,
          sizeof(AttentionParameters));
      attention_h3_banded_pipeline = make_norm_pipeline(
          detail::kTensorAttentionH3BandedSpirv,
          sizeof(detail::kTensorAttentionH3BandedSpirv), 5,
          kH3AttentionLocalSize, 1,
          sizeof(AttentionParameters));
    }
    if (exact_causal_gqa_attention) {
      attention_causal_gqa_pipeline = make_norm_pipeline(
          detail::kTensorAttentionCausalGqaSpirv,
          sizeof(detail::kTensorAttentionCausalGqaSpirv), 4, 128, 1,
          sizeof(CausalGQAAttentionParameters));
    }
    for (uint32_t i = 0; i < ops_bindings.size(); ++i) ops_bindings[i].binding = i;
    for (uint32_t i = 0; i < vae_pointwise_bindings.size(); ++i)
      vae_pointwise_bindings[i].binding = i;
    for (uint32_t i = 0; i < norm_bindings.size(); ++i) norm_bindings[i].binding = i;
    for (uint32_t i = 0; i < mod_bindings.size(); ++i) mod_bindings[i].binding = i;
    for (uint32_t i = 0; i < vae_rope_bindings.size(); ++i)
      vae_rope_bindings[i].binding = i;
    for (uint32_t i = 0; i < audio_bindings.size(); ++i)
      audio_bindings[i].binding = i;
    for (uint32_t i = 0; i < dit_bindings.size(); ++i)
      dit_bindings[i].binding = i;
    for (uint32_t i = 0; i < weight_bindings.size(); ++i)
      weight_bindings[i].binding = i;
    for (uint32_t i = 0; i < gemm_bindings.size(); ++i)
      gemm_bindings[i].binding = i;
    for (uint32_t i = 0; i < gemm_prepare_bindings.size(); ++i)
      gemm_prepare_bindings[i].binding = i;
    for (uint32_t i = 0; i < attention_bindings.size(); ++i)
      attention_bindings[i].binding = i;
    for (uint32_t i = 0; i < attention_h3_bindings.size(); ++i)
      attention_h3_bindings[i].binding = i;
    for (uint32_t i = 0; i < attention_h3_banded_bindings.size(); ++i)
      attention_h3_banded_bindings[i].binding = i;
    for (uint32_t i = 0; i < attention_prepare_bindings.size(); ++i)
      attention_prepare_bindings[i].binding = i;
    for (uint32_t i = 0; i < attention_causal_gqa_bindings.size(); ++i)
      attention_causal_gqa_bindings[i].binding = i;
  }

  uintptr_t context_id = next_context_identity();

  struct RecorderLease {
    Impl* owner = nullptr;
    RecorderLease() = default;
    explicit RecorderLease(Impl* value) : owner(value) {}
    RecorderLease(RecorderLease&& other) noexcept
        : owner(std::exchange(other.owner, nullptr)) {}
    RecorderLease& operator=(RecorderLease&& other) noexcept {
      if (this != &other) {
        release();
        owner = std::exchange(other.owner, nullptr);
      }
      return *this;
    }
    RecorderLease(const RecorderLease&) = delete;
    RecorderLease& operator=(const RecorderLease&) = delete;
    ~RecorderLease() { release(); }
    void release() noexcept {
      if (!owner) return;
      owner->recorder_active.store(false, std::memory_order_release);
      owner = nullptr;
    }
  };

  RecorderLease acquire_recorder() {
    bool expected = false;
    if (!recorder_active.compare_exchange_strong(
            expected, true, std::memory_order_acquire,
            std::memory_order_relaxed)) {
      throw std::logic_error(
          "vulkan tensor: another batch or boundary operation is active");
    }
    return RecorderLease(this);
  }

  void ensure_staging(uint64_t bytes) {
    if (bytes <= staging_capacity) return;
    Buffer new_upload = pool.allocate(bytes, BufferUsage::kTransferSource, MemoryUsage::kUpload);
    Buffer new_readback =
        pool.allocate(bytes, BufferUsage::kTransferDestination, MemoryUsage::kReadback);
    upload_buffer = std::move(new_upload);
    readback_buffer = std::move(new_readback);
    staging_capacity = bytes;
    pool.trim();
  }

  std::shared_ptr<DeviceTensor::Impl> require(DeviceTensor& tensor) const {
    if (!tensor.impl_ || tensor.impl_->context != context_id) {
      throw std::invalid_argument("vulkan tensor: tensor belongs to another context");
    }
    return tensor.impl_;
  }

  void complete(CommandList&& list) {
    Submission done = commands.submit(std::move(list));
    done.wait();
    commands.collect();
  }

  uint32_t validate_dispatch(uint64_t count) const {
    if (count == 0 || count > std::numeric_limits<uint32_t>::max()) {
      throw std::out_of_range("vulkan tensor: operation exceeds uint32 indexing limits");
    }
    const uint64_t groups = (count + 63ull) / 64ull;
    if (groups > max_dispatch_x) {
      throw std::out_of_range("vulkan tensor: operation exceeds dispatch limits");
    }
    return static_cast<uint32_t>(count);
  }
};

struct TensorBatch::Impl {
  struct AccessSnapshot {
    std::shared_ptr<DeviceTensor::Impl> tensor;
    bool has_access = false;
    BufferAccess access = BufferAccess::kTransferWrite;
  };

  std::shared_ptr<TensorContext::Impl> owner;
  uintptr_t batch_id = 0;
  CommandList commands;
  TensorContext::Impl::RecorderLease recording_lease;
  std::vector<AccessSnapshot> snapshots;
  uint32_t snapshot_count = 0;
  uint32_t operator_count = 0;
  bool submitted = false;
  bool poisoned = false;

  void transition(const std::shared_ptr<DeviceTensor::Impl>& tensor, BufferAccess next) {
    bool seen = false;
    for (uint32_t i = 0; i < snapshot_count; ++i) {
      if (snapshots[i].tensor.get() == tensor.get()) {
        seen = true;
        break;
      }
    }
    if (!seen) {
      if (snapshot_count == snapshots.size()) {
        throw std::logic_error("vulkan tensor: batch transition capacity exceeded");
      }
      snapshots[snapshot_count++] = {tensor, tensor->has_access, tensor->access};
    }
    const bool next_writes = next == BufferAccess::kTransferWrite ||
                             next == BufferAccess::kComputeWrite ||
                             next == BufferAccess::kComputeReadWrite;
    if (tensor->has_access && (tensor->access != next || next_writes)) {
      commands.barrier(tensor->buffer, tensor->access, next);
    }
    tensor->has_access = true;
    tensor->access = next;
  }

  void count_operator() {
    if (operator_count == owner->max_batch_operators) {
      throw std::logic_error("vulkan tensor: batch operator limit exceeded");
    }
    ++operator_count;
  }

  void dispatch(const TensorContext::Impl::Parameters& parameters,
                const std::shared_ptr<DeviceTensor::Impl>& a,
                const std::shared_ptr<DeviceTensor::Impl>& b,
                const std::shared_ptr<DeviceTensor::Impl>& output) {
    // Every caller validates this before it mutates access tracking. Widen the
    // addition anyway so the invariant remains safe if uint32 max is used.
    const uint32_t groups =
        static_cast<uint32_t>((static_cast<uint64_t>(parameters.count) + 63ull) / 64ull);
    owner->ops_bindings[0].buffer = &a->buffer;
    owner->ops_bindings[0].bytes = a->buffer.size();
    owner->ops_bindings[1].buffer = &b->buffer;
    owner->ops_bindings[1].bytes = b->buffer.size();
    owner->ops_bindings[2].buffer = &output->buffer;
    owner->ops_bindings[2].bytes = output->buffer.size();
    commands.bind_compute(owner->ops_pipeline, owner->ops_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups);
  }

  void dispatch_vae_pointwise(
      VaePointwiseOperation operation,
      const TensorContext::Impl::Parameters& parameters,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 4>& resources) {
    const uint32_t groups = static_cast<uint32_t>(
        (static_cast<uint64_t>(parameters.count) + 63ull) / 64ull);
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->vae_pointwise_bindings[i].buffer = &resources[i]->buffer;
      owner->vae_pointwise_bindings[i].bytes = resources[i]->buffer.size();
    }
    ComputePipeline* pipeline = nullptr;
    switch (operation) {
      case VaePointwiseOperation::kResidual: pipeline = &owner->vae_residual_pipeline; break;
      case VaePointwiseOperation::kSwiglu: pipeline = &owner->vae_swiglu_pipeline; break;
      case VaePointwiseOperation::kDenorm: pipeline = &owner->vae_denorm_pipeline; break;
    }
    if (pipeline == nullptr) throw std::logic_error("vulkan tensor: invalid VAE pointwise op");
    commands.bind_compute(*pipeline, owner->vae_pointwise_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups);
  }

  void dispatch_rope(const TensorContext::Impl::Parameters& parameters,
                     uint32_t groups,
                     const std::shared_ptr<DeviceTensor::Impl>& input,
                     const std::shared_ptr<DeviceTensor::Impl>& cosine,
                     const std::shared_ptr<DeviceTensor::Impl>& sine) {
    owner->ops_bindings[0].buffer = &input->buffer;
    owner->ops_bindings[0].bytes = input->buffer.size();
    owner->ops_bindings[1].buffer = &cosine->buffer;
    owner->ops_bindings[1].bytes = cosine->buffer.size();
    owner->ops_bindings[2].buffer = &sine->buffer;
    owner->ops_bindings[2].bytes = sine->buffer.size();
    commands.bind_compute(owner->rope_pipeline, owner->ops_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups);
  }

  void dispatch_norm(ComputePipeline& pipeline,
                     const TensorContext::Impl::NormParameters& parameters,
                     const std::shared_ptr<DeviceTensor::Impl>& input,
                     const std::shared_ptr<DeviceTensor::Impl>& weight,
                     const std::shared_ptr<DeviceTensor::Impl>& bias,
                     const std::shared_ptr<DeviceTensor::Impl>& output) {
    owner->norm_bindings[0].buffer = &input->buffer;
    owner->norm_bindings[0].bytes = input->buffer.size();
    owner->norm_bindings[1].buffer = &weight->buffer;
    owner->norm_bindings[1].bytes = weight->buffer.size();
    owner->norm_bindings[2].buffer = &bias->buffer;
    owner->norm_bindings[2].bytes = bias->buffer.size();
    owner->norm_bindings[3].buffer = &output->buffer;
    owner->norm_bindings[3].bytes = output->buffer.size();
    commands.bind_compute(pipeline, owner->norm_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(parameters.rows);
  }

  void dispatch_shared_rms(ComputePipeline& pipeline,
                           const TensorContext::Impl::NormParameters& parameters,
                           uint32_t groups,
                           const std::shared_ptr<DeviceTensor::Impl>& input,
                           const std::shared_ptr<DeviceTensor::Impl>& weight,
                           const std::shared_ptr<DeviceTensor::Impl>& output) {
    owner->ops_bindings[0].buffer = &input->buffer;
    owner->ops_bindings[0].bytes = input->buffer.size();
    owner->ops_bindings[1].buffer = &weight->buffer;
    owner->ops_bindings[1].bytes = weight->buffer.size();
    owner->ops_bindings[2].buffer = &output->buffer;
    owner->ops_bindings[2].bytes = output->buffer.size();
    commands.bind_compute(pipeline, owner->ops_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups);
  }

  void dispatch_shared_layer(ComputePipeline& pipeline,
                             const TensorContext::Impl::NormParameters& parameters,
                             const std::shared_ptr<DeviceTensor::Impl>& input,
                             const std::shared_ptr<DeviceTensor::Impl>& weight,
                             const std::shared_ptr<DeviceTensor::Impl>& bias,
                             const std::shared_ptr<DeviceTensor::Impl>& output) {
    owner->norm_bindings[0].buffer = &input->buffer;
    owner->norm_bindings[0].bytes = input->buffer.size();
    owner->norm_bindings[1].buffer = &weight->buffer;
    owner->norm_bindings[1].bytes = weight->buffer.size();
    owner->norm_bindings[2].buffer = &bias->buffer;
    owner->norm_bindings[2].bytes = bias->buffer.size();
    owner->norm_bindings[3].buffer = &output->buffer;
    owner->norm_bindings[3].bytes = output->buffer.size();
    commands.bind_compute(pipeline, owner->norm_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(parameters.rows);
  }

  void dispatch_group_norm(const TensorContext::Impl::NormParameters& parameters,
                           const std::shared_ptr<DeviceTensor::Impl>& input,
                           const std::shared_ptr<DeviceTensor::Impl>& weight,
                           const std::shared_ptr<DeviceTensor::Impl>& bias,
                           const std::shared_ptr<DeviceTensor::Impl>& output) {
    owner->norm_bindings[0].buffer = &input->buffer;
    owner->norm_bindings[0].bytes = input->buffer.size();
    owner->norm_bindings[1].buffer = &weight->buffer;
    owner->norm_bindings[1].bytes = weight->buffer.size();
    owner->norm_bindings[2].buffer = &bias->buffer;
    owner->norm_bindings[2].bytes = bias->buffer.size();
    owner->norm_bindings[3].buffer = &output->buffer;
    owner->norm_bindings[3].bytes = output->buffer.size();
    commands.bind_compute(owner->group_norm_pipeline, owner->norm_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(parameters.mod_rows);
  }

  void dispatch_shared_mod(ComputePipeline& pipeline,
                           const TensorContext::Impl::NormParameters& parameters,
                           const std::array<std::shared_ptr<DeviceTensor::Impl>, 6>& resources,
                           uint64_t scale_offset = 0,
                           uint64_t shift_offset = 0) {
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->mod_bindings[i].buffer = &resources[i]->buffer;
      owner->mod_bindings[i].offset = 0;
      owner->mod_bindings[i].bytes = resources[i]->buffer.size();
    }
    owner->mod_bindings[2].offset = scale_offset;
    owner->mod_bindings[2].bytes -= scale_offset;
    owner->mod_bindings[3].offset = shift_offset;
    owner->mod_bindings[3].bytes -= shift_offset;
    commands.bind_compute(pipeline, owner->mod_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(parameters.rows);
  }

  void dispatch_vae_rope(
      const TensorContext::Impl::VaeRopeParameters& parameters,
      uint32_t groups,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 7>& resources) {
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->vae_rope_bindings[i].buffer = &resources[i]->buffer;
      owner->vae_rope_bindings[i].bytes = resources[i]->buffer.size();
    }
    commands.bind_compute(owner->vae_rope_pipeline, owner->vae_rope_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups);
  }

  void dispatch_audio(
      TensorContext::Impl::AudioParameters parameters,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 5>& resources) {
    const uint64_t total_groups =
        (static_cast<uint64_t>(parameters.count) + 63ull) / 64ull;
    const uint32_t groups_x = static_cast<uint32_t>(
        std::min<uint64_t>(total_groups, owner->max_dispatch_x));
    if (groups_x == 0) {
      throw std::out_of_range("vulkan audio: device exposes no X dispatch capacity");
    }
    const uint64_t groups_y_wide =
        (total_groups + groups_x - 1ull) / groups_x;
    if (groups_y_wide > owner->max_dispatch_y) {
      throw std::out_of_range(
          "vulkan audio: primitive exceeds two-dimensional dispatch limits");
    }
    parameters.groups_x = groups_x;
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->audio_bindings[i].buffer = &resources[i]->buffer;
      owner->audio_bindings[i].bytes = resources[i]->buffer.size();
    }
    commands.bind_compute(owner->audio_pipeline, owner->audio_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups_x, static_cast<uint32_t>(groups_y_wide));
  }

  void dispatch_dit(
      TensorContext::Impl::DitParameters parameters,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 5>& resources,
      uint64_t tertiary_offset = 0) {
    const uint64_t total_groups =
        (static_cast<uint64_t>(parameters.count) + 63ull) / 64ull;
    const uint32_t groups_x = static_cast<uint32_t>(
        std::min<uint64_t>(total_groups, owner->max_dispatch_x));
    if (groups_x == 0) {
      throw std::out_of_range("vulkan DiT: device exposes no X dispatch capacity");
    }
    const uint64_t groups_y = (total_groups + groups_x - 1ull) / groups_x;
    if (groups_y > owner->max_dispatch_y) {
      throw std::out_of_range("vulkan DiT: operation exceeds dispatch limits");
    }
    parameters.groups_x = groups_x;
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->dit_bindings[i].buffer = &resources[i]->buffer;
      owner->dit_bindings[i].offset = 0;
      owner->dit_bindings[i].bytes = resources[i]->buffer.size();
    }
    owner->dit_bindings[2].offset = tertiary_offset;
    owner->dit_bindings[2].bytes -= tertiary_offset;
    commands.bind_compute(owner->dit_pipeline, owner->dit_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups_x, static_cast<uint32_t>(groups_y));
  }

  void dispatch_weight(
      const TensorContext::Impl::WeightParameters& parameters,
      uint32_t groups,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 6>& resources) {
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->weight_bindings[i].buffer = &resources[i]->buffer;
      owner->weight_bindings[i].bytes = resources[i]->buffer.size();
    }
    commands.bind_compute(owner->weight_pipeline, owner->weight_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups);
  }

  void dispatch_attention(
      const TensorContext::Impl::AttentionParameters& parameters,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 4>& resources) {
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->attention_bindings[i].buffer = &resources[i]->buffer;
      owner->attention_bindings[i].bytes = resources[i]->buffer.size();
    }
    commands.bind_compute(owner->attention_blocked_pipeline,
                          owner->attention_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(parameters.rows, parameters.heads);
  }

  void dispatch_h3_attention(
      const TensorContext::Impl::AttentionParameters& parameters,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 4>& resources) {
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->attention_h3_bindings[i].buffer = &resources[i]->buffer;
      owner->attention_h3_bindings[i].bytes = resources[i]->buffer.size();
    }
    commands.bind_compute(owner->attention_h3_pipeline,
                          owner->attention_h3_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    const uint32_t aligned_first = parameters.query_row_offset & ~63u;
    const uint32_t groups =
        (parameters.query_row_offset + parameters.rows - aligned_first + 63u) / 64u;
    commands.dispatch(groups, parameters.heads);
  }

  void dispatch_h3_banded_attention(
      const TensorContext::Impl::AttentionParameters& parameters,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 5>& resources) {
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->attention_h3_banded_bindings[i].buffer = &resources[i]->buffer;
      owner->attention_h3_banded_bindings[i].bytes = resources[i]->buffer.size();
    }
    commands.bind_compute(owner->attention_h3_banded_pipeline,
                          owner->attention_h3_banded_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    const uint32_t aligned_first = parameters.query_row_offset & ~63u;
    const uint32_t groups =
        (parameters.query_row_offset + parameters.rows - aligned_first + 63u) / 64u;
    commands.dispatch(groups, parameters.heads);
  }

  void dispatch_attention_prepare(
      uint32_t words,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 6>& resources) {
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->attention_prepare_bindings[i].buffer = &resources[i]->buffer;
      owner->attention_prepare_bindings[i].bytes = resources[i]->buffer.size();
    }
    commands.bind_compute(owner->attention_prepare_pipeline,
                          owner->attention_prepare_bindings);
    commands.push_constants(&words, sizeof(words));
    commands.dispatch(static_cast<uint32_t>((static_cast<uint64_t>(words) + 63) / 64));
  }

  void dispatch_causal_gqa_attention(
      const TensorContext::Impl::CausalGQAAttentionParameters& parameters,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 4>& resources) {
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->attention_causal_gqa_bindings[i].buffer = &resources[i]->buffer;
      owner->attention_causal_gqa_bindings[i].bytes = resources[i]->buffer.size();
    }
    commands.bind_compute(owner->attention_causal_gqa_pipeline,
                          owner->attention_causal_gqa_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(parameters.rows, parameters.query_heads);
  }

  void record_shared_mod(bool fp32, DeviceTensor& input, DeviceTensor& weight,
                         DeviceTensor& scale, DeviceTensor& shift,
                         DeviceTensor& selectors, DeviceTensor& output,
                         float epsilon) {
    if (!owner->exact_vae_norm) {
      throw std::runtime_error("vulkan tensor: exact shared RMSNorm modulation is unavailable");
    }
    auto src = owner->require(input);
    auto w = owner->require(weight);
    auto sc = owner->require(scale);
    auto sh = owner->require(shift);
    auto index = owner->require(selectors);
    auto dst = owner->require(output);
    const auto& shape = src->layout;
    const uint64_t rows = shape.extent[0];
    const uint64_t dim = shape.extent[1];
    const uint64_t mod_rows = sc->layout.extent[0];
    const bool aliases_parameters =
        src.get() == w.get() || src.get() == sc.get() || src.get() == sh.get() ||
        src.get() == index.get() || w.get() == sc.get() || w.get() == sh.get() ||
        w.get() == index.get() || sc.get() == sh.get() || sc.get() == index.get() ||
        sh.get() == index.get() || dst.get() == w.get() || dst.get() == sc.get() ||
        dst.get() == sh.get() || dst.get() == index.get();
    if (!std::isnormal(epsilon) || epsilon <= 0.0f || aliases_parameters ||
        shape.rank != 2 || w->layout.rank != 1 || sc->layout.rank != 2 ||
        sh->layout.rank != 2 || index->layout.rank != 1 || dst->layout.rank != 2 ||
        w->layout.extent[0] != dim || sc->layout.extent[1] != dim ||
        sh->layout.extent != sc->layout.extent || index->layout.extent[0] != rows ||
        dst->layout.extent != shape.extent || w->type != ScalarType::kBFloat16 ||
        sc->type != ScalarType::kFloat32 || sh->type != ScalarType::kFloat32 ||
        index->type != ScalarType::kInt32 ||
        src->type != (fp32 ? ScalarType::kFloat32 : ScalarType::kBFloat16) ||
        dst->type != src->type || !shape.is_contiguous() ||
        !w->layout.is_contiguous() || !sc->layout.is_contiguous() ||
        !sh->layout.is_contiguous() || !index->layout.is_contiguous() ||
        !dst->layout.is_contiguous() || rows > std::numeric_limits<uint32_t>::max() ||
        dim > kMaxExactNormDimension ||
        mod_rows > std::numeric_limits<uint32_t>::max() ||
        shape.elements() > std::numeric_limits<uint32_t>::max() ||
        sc->layout.elements() > std::numeric_limits<uint32_t>::max()) {
      throw std::invalid_argument("vulkan tensor: invalid shared RMSNorm modulation");
    }
    if (!detail::norm_dispatch_fits(rows, owner->max_dispatch_x)) {
      throw std::out_of_range("vulkan tensor: modulated RMSNorm rows exceed dispatch limits");
    }
    TensorContext::Impl::NormParameters p;
    p.rows = static_cast<uint32_t>(rows);
    p.dim = static_cast<uint32_t>(dim);
    p.mod_rows = static_cast<uint32_t>(mod_rows);
    std::memcpy(&p.epsilon_bits, &epsilon, sizeof(epsilon));
    try {
      count_operator();
      transition(src, src.get() == dst.get() ? BufferAccess::kComputeReadWrite
                                              : BufferAccess::kComputeRead);
      transition(w, BufferAccess::kComputeRead);
      transition(sc, BufferAccess::kComputeRead);
      transition(sh, BufferAccess::kComputeRead);
      transition(index, BufferAccess::kComputeRead);
      if (src.get() != dst.get()) transition(dst, BufferAccess::kComputeWrite);
      std::array<std::shared_ptr<DeviceTensor::Impl>, 6> resources{
          src, w, sc, sh, index, dst};
      dispatch_shared_mod(fp32 ? owner->fp32_mod_pipeline : owner->bf16_mod_pipeline,
                          p, resources);
    } catch (...) {
      poisoned = true;
      throw;
    }
  }

  void record_shared_mod_table(DeviceTensor& input, DeviceTensor& weight,
                               DeviceTensor& tables, uint32_t mod_rows,
                               uint32_t scale_table,
                               uint32_t shift_table, DeviceTensor& selectors,
                               DeviceTensor& output, float epsilon) {
    if (!owner->exact_vae_norm)
      throw std::runtime_error("vulkan tensor: exact shared RMSNorm modulation is unavailable");
    auto src=owner->require(input), w=owner->require(weight), table=owner->require(tables);
    auto index=owner->require(selectors), dst=owner->require(output);
    const auto& shape=src->layout; const uint64_t rows=shape.rank==2?shape.extent[0]:0;
    const uint64_t dim=shape.rank==2?shape.extent[1]:0;
    const uint64_t table_count=6;
    const uint64_t stride=table->layout.rank==1?table->layout.extent[0]/table_count:0;
    const uint64_t logical=checked_multiply(mod_rows,dim,"modulation table");
    if (!std::isnormal(epsilon)||epsilon<=0.0f||scale_table>=table_count||
        shift_table>=table_count||table->layout.extent[0]%table_count!=0||stride<logical||
        src.get()==w.get()||src.get()==table.get()||
        src.get()==index.get()||w.get()==table.get()||w.get()==index.get()||
        table.get()==index.get()||dst.get()==w.get()||dst.get()==table.get()||
        dst.get()==index.get()||shape.rank!=2||w->layout.rank!=1||
        table->layout.rank!=1||
        index->layout.rank!=1||index->layout.extent[0]!=rows||
        dst->layout.rank!=2||dst->layout.extent!=shape.extent||
        src->type!=ScalarType::kBFloat16||w->type!=ScalarType::kBFloat16||
        table->type!=ScalarType::kFloat32||index->type!=ScalarType::kInt32||
        dst->type!=ScalarType::kBFloat16||w->layout.extent[0]!=dim||
        !shape.is_contiguous()||!w->layout.is_contiguous()||
        !table->layout.is_contiguous()||!index->layout.is_contiguous()||
        !dst->layout.is_contiguous()||rows>UINT32_MAX||dim>kMaxExactNormDimension||
        mod_rows==0||shape.elements()>UINT32_MAX)
      throw std::invalid_argument("vulkan tensor: invalid shared RMSNorm table modulation");
    if (!detail::norm_dispatch_fits(rows, owner->max_dispatch_x))
      throw std::out_of_range("vulkan tensor: modulated RMSNorm rows exceed dispatch limits");
    const uint64_t table_bytes=stride*4;
    TensorContext::Impl::NormParameters p; p.rows=static_cast<uint32_t>(rows);
    p.dim=static_cast<uint32_t>(dim);p.mod_rows=static_cast<uint32_t>(mod_rows);
    std::memcpy(&p.epsilon_bits,&epsilon,sizeof(epsilon));
    try {
      count_operator();transition(src,src.get()==dst.get()?BufferAccess::kComputeReadWrite:BufferAccess::kComputeRead);
      transition(w,BufferAccess::kComputeRead);transition(table,BufferAccess::kComputeRead);
      transition(index,BufferAccess::kComputeRead);if(src.get()!=dst.get())transition(dst,BufferAccess::kComputeWrite);
      std::array<std::shared_ptr<DeviceTensor::Impl>,6> r{src,w,table,table,index,dst};
      dispatch_shared_mod(owner->bf16_mod_pipeline,p,r,scale_table*table_bytes,shift_table*table_bytes);
    } catch (...) { poisoned=true; throw; }
  }

  ~Impl() {
    if (submitted) return;
    // Discard Vulkan recording before making its speculative access state
    // invisible to the next batch.
    commands = CommandList{};
    for (uint32_t i = snapshot_count; i-- > 0;) {
      snapshots[i].tensor->has_access = snapshots[i].has_access;
      snapshots[i].tensor->access = snapshots[i].access;
    }
  }
};

DeviceTensor::DeviceTensor() = default;
DeviceTensor::~DeviceTensor() = default;
DeviceTensor::DeviceTensor(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
DeviceTensor::DeviceTensor(DeviceTensor&&) noexcept = default;
DeviceTensor& DeviceTensor::operator=(DeviceTensor&&) noexcept = default;
DeviceTensorView DeviceTensor::view() const {
  if (!impl_) throw std::logic_error("vulkan tensor: empty tensor");
  DeviceTensorView result;
  result.backend = DeviceBackend::kVulkan;
  result.type = impl_->type;
  result.layout = impl_->layout;
  result.context = impl_->context;
  result.resource = impl_->buffer.native_handle();
  result.byte_size = impl_->logical_bytes;
  return result;
}
const TensorLayout& DeviceTensor::layout() const {
  if (!impl_) throw std::logic_error("vulkan tensor: empty tensor");
  return impl_->layout;
}
ScalarType DeviceTensor::type() const {
  if (!impl_) throw std::logic_error("vulkan tensor: empty tensor");
  return impl_->type;
}
DeviceTensor::operator bool() const noexcept { return impl_ != nullptr; }

TensorWorkspace::TensorWorkspace(const Device& device, uint64_t block_bytes)
    : impl_(std::make_unique<Impl>(device, block_bytes)) {}
TensorWorkspace::~TensorWorkspace() = default;
TensorWorkspace::TensorWorkspace(TensorWorkspace&&) noexcept = default;
TensorWorkspace& TensorWorkspace::operator=(TensorWorkspace&&) noexcept = default;
DeviceBackend TensorWorkspace::backend() const noexcept { return DeviceBackend::kVulkan; }
void TensorWorkspace::reserve(uint64_t bytes) {
  if (!impl_) throw std::logic_error("vulkan workspace: moved-from object");
  if (bytes <= impl_->capacity) return;
  if (impl_->cursor != 0) throw std::logic_error("vulkan workspace: reset before growth");
  Buffer replacement = impl_->pool.allocate(
      bytes, BufferUsage::kStorage | BufferUsage::kTransferSource |
                 BufferUsage::kTransferDestination,
      MemoryUsage::kDevice);
  impl_->buffer = std::move(replacement);
  impl_->capacity = bytes;
  ++impl_->generation;
  impl_->pool.trim();
}
WorkspaceSpan TensorWorkspace::allocate(uint64_t bytes, uint64_t alignment) {
  if (!impl_) throw std::logic_error("vulkan workspace: moved-from object");
  if (bytes == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) {
    throw std::invalid_argument("vulkan workspace: size and power-of-two alignment required");
  }
  if (impl_->cursor > std::numeric_limits<uint64_t>::max() - (alignment - 1)) {
    throw std::overflow_error("vulkan workspace: offset overflow");
  }
  const uint64_t offset = (impl_->cursor + alignment - 1) & ~(alignment - 1);
  if (offset > impl_->capacity || bytes > impl_->capacity - offset) {
    throw std::out_of_range("vulkan workspace: reserved capacity exceeded");
  }
  impl_->cursor = offset + bytes;
  return {DeviceBackend::kVulkan, impl_->context, impl_->buffer.native_handle(), offset, bytes,
          impl_->generation};
}
void TensorWorkspace::reset() noexcept { if (impl_) impl_->cursor = 0; }
uint64_t TensorWorkspace::capacity() const noexcept { return impl_ ? impl_->capacity : 0; }
uint64_t TensorWorkspace::used() const noexcept { return impl_ ? impl_->cursor : 0; }
uint64_t TensorWorkspace::generation() const noexcept { return impl_ ? impl_->generation : 0; }
bool TensorWorkspace::valid(const WorkspaceSpan& span) const noexcept {
  return impl_ && span.backend == DeviceBackend::kVulkan && span.context == impl_->context &&
         span.resource == impl_->buffer.native_handle() &&
         span.generation == impl_->generation && span.byte_size != 0 &&
         span.byte_offset <= impl_->capacity &&
         span.byte_size <= impl_->capacity - span.byte_offset;
}
uint64_t TensorWorkspace::reserved_bytes() const noexcept {
  return impl_ ? impl_->pool.reserved_bytes() : 0;
}
uint64_t TensorWorkspace::pooled_used_bytes() const noexcept {
  return impl_ ? impl_->pool.used_bytes() : 0;
}

LinearWeight::LinearWeight() = default;
LinearWeight::~LinearWeight() = default;
LinearWeight::LinearWeight(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
LinearWeight::LinearWeight(LinearWeight&&) noexcept = default;
LinearWeight& LinearWeight::operator=(LinearWeight&&) noexcept = default;

LinearWeight LinearWeight::upload(TensorContext& context,
                                  const LinearWeightUpload& source) {
  if (!context.impl_) {
    throw std::logic_error("vulkan linear weight: moved-from tensor context");
  }
  if (source.out_features == 0 || source.in_features == 0 || source.data == nullptr) {
    throw std::invalid_argument("vulkan linear weight: nonzero shape and data required");
  }
  const uint64_t elements = checked_multiply(source.out_features,
                                              source.in_features, "linear weight");
  uint64_t expected_bytes = 0;
  ScalarType data_type = ScalarType::kUInt8;
  switch (source.format) {
    case LinearWeightFormat::kFloat32:
      expected_bytes = checked_multiply(elements, 4, "linear weight");
      data_type = ScalarType::kFloat32;
      break;
    case LinearWeightFormat::kFloat16:
      expected_bytes = checked_multiply(elements, 2, "linear weight");
      data_type = ScalarType::kFloat16;
      break;
    case LinearWeightFormat::kBFloat16:
      expected_bytes = checked_multiply(elements, 2, "linear weight");
      data_type = ScalarType::kBFloat16;
      break;
    case LinearWeightFormat::kFloat8E4M3:
    case LinearWeightFormat::kInt8:
      expected_bytes = elements;
      data_type = source.format == LinearWeightFormat::kInt8
          ? ScalarType::kInt8 : ScalarType::kUInt8;
      break;
    case LinearWeightFormat::kNVFloat4:
    case LinearWeightFormat::kNF4:
      expected_bytes = (elements + 1) / 2;
      break;
  }
  if (source.data_bytes != expected_bytes) {
    throw std::invalid_argument("vulkan linear weight: stored byte count mismatch");
  }
  const bool scale_pair_valid = (source.weight_scale == nullptr) ==
                                (source.weight_scale_count == 0);
  const bool block_pair_valid = (source.block_scale == nullptr) ==
                                (source.block_scale_count == 0);
  const bool absmax_pair_valid = (source.nf4_absmax == nullptr) ==
                                 (source.nf4_absmax_count == 0);
  const bool map_pair_valid = (source.nf4_quant_map == nullptr) ==
                              (source.nf4_quant_map_count == 0);
  const bool nested_map_pair_valid = (source.nf4_nested_quant_map == nullptr) ==
                                     (source.nf4_nested_quant_map_count == 0);
  const bool nested_absmax_pair_valid =
      (source.nf4_nested_absmax == nullptr) ==
      (source.nf4_nested_absmax_count == 0);
  const bool pre_scale_pair_valid =
      (source.pre_quant_scale_bf16 == nullptr) ==
      (source.pre_quant_scale_count == 0);
  if (!scale_pair_valid || !block_pair_valid || !absmax_pair_valid ||
      !map_pair_valid || !nested_map_pair_valid ||
      !nested_absmax_pair_valid || !pre_scale_pair_valid) {
    throw std::invalid_argument(
        "vulkan linear weight: auxiliary pointer/count mismatch");
  }
  if (source.has_fp8_input_scale &&
      (source.format != LinearWeightFormat::kFloat8E4M3 ||
       !std::isnormal(source.fp8_input_scale) || source.fp8_input_scale <= 0.0f)) {
    throw std::invalid_argument("vulkan linear weight: invalid FP8 input scale");
  }
  if (!source.has_fp8_input_scale && source.fp8_input_scale != 0.0f) {
    throw std::invalid_argument(
        "vulkan linear weight: absent FP8 input scale must retain zero value");
  }
  if (source.pre_quant_scale_count != 0 &&
      (source.pre_quant_scale_bf16 == nullptr ||
       source.pre_quant_scale_count != source.in_features)) {
    throw std::invalid_argument("vulkan linear weight: invalid AWQ pre-scale");
  }
  uint32_t power = 1;
  while (power < source.convrot_group && power <= UINT32_MAX / 4) power *= 4;
  if (source.convrot && (source.convrot_group < 4 || source.convrot_group > 256 ||
                         power != source.convrot_group ||
                         source.in_features % source.convrot_group != 0)) {
    throw std::invalid_argument("vulkan linear weight: invalid ConvRot group");
  }
  if (source.format == LinearWeightFormat::kFloat8E4M3 &&
      (source.weight_scale == nullptr || source.weight_scale_count != 1)) {
    throw std::invalid_argument("vulkan linear weight: FP8 scalar scale required");
  }
  if (source.format == LinearWeightFormat::kInt8 &&
      (source.weight_scale == nullptr ||
       source.weight_scale_count != source.out_features)) {
    throw std::invalid_argument("vulkan linear weight: INT8 per-row scales required");
  }
  if (source.format == LinearWeightFormat::kNVFloat4 &&
      (source.in_features % 64 != 0 || source.out_features % 128 != 0 ||
       source.block_scale == nullptr ||
       source.block_scale_count != elements / 16 ||
       !std::isfinite(source.global_scale))) {
    throw std::invalid_argument("vulkan linear weight: invalid NVFP4 metadata");
  }
  const uint64_t expected_absmax = source.nf4_block_size == 0
      ? 0 : 1 + (elements - 1) / source.nf4_block_size;
  const uint64_t expected_nested = source.nf4_nested_block_size == 0 ||
                                           expected_absmax == 0
      ? 0 : 1 + (expected_absmax - 1) / source.nf4_nested_block_size;
  if (source.format == LinearWeightFormat::kNF4 &&
      (source.nf4_block_size != 64 || source.nf4_nested_block_size != 256 ||
       source.nf4_absmax == nullptr || source.nf4_absmax_count != expected_absmax ||
       source.nf4_quant_map == nullptr || source.nf4_quant_map_count != 16 ||
       source.nf4_nested_quant_map == nullptr ||
       source.nf4_nested_quant_map_count != 256 ||
       source.nf4_nested_absmax == nullptr ||
       source.nf4_nested_absmax_count != expected_nested ||
       !std::isfinite(source.nf4_nested_offset))) {
    throw std::invalid_argument("vulkan linear weight: invalid NF4 metadata");
  }
  const bool has_nf4_aux = source.nf4_absmax_count != 0 ||
      source.nf4_quant_map_count != 0 ||
      source.nf4_nested_quant_map_count != 0 ||
      source.nf4_nested_absmax_count != 0;
  if ((source.format != LinearWeightFormat::kFloat8E4M3 &&
       source.format != LinearWeightFormat::kInt8 &&
       source.weight_scale_count != 0) ||
      (source.format != LinearWeightFormat::kNVFloat4 &&
       source.block_scale_count != 0) ||
      (source.format != LinearWeightFormat::kNF4 && has_nf4_aux)) {
    throw std::invalid_argument(
        "vulkan linear weight: auxiliary metadata does not match format");
  }

  auto result = std::make_unique<Impl>();
  result->format = source.format;
  result->out_features = source.out_features;
  result->in_features = source.in_features;
  result->stored_bytes = expected_bytes;
  result->has_fp8_input_scale = source.has_fp8_input_scale;
  result->fp8_input_scale = source.fp8_input_scale;
  result->full_precision_matrix_mult = source.full_precision_matrix_mult;
  result->global_scale = source.global_scale;
  result->nf4_nested_offset = source.nf4_nested_offset;
  result->nf4_block_size = source.nf4_block_size;
  result->nf4_nested_block_size = source.nf4_nested_block_size;
  result->convrot = source.convrot;
  result->convrot_group = source.convrot_group;
  struct PendingUpload {
    DeviceTensor* destination = nullptr;
    const void* values = nullptr;
    uint64_t logical_bytes = 0;
  };
  std::vector<PendingUpload> pending;
  pending.reserve(9);
  auto allocate = [&](DeviceTensor& destination, ScalarType type, uint64_t count,
                      const void* values) {
    TensorLayout layout = TensorLayout::contiguous(&count, 1);
    DeviceTensor replacement = context.allocate(layout, type);
    const uint64_t bytes = layout.bytes(type);
    if (result->resident_bytes > std::numeric_limits<uint64_t>::max() - bytes) {
      throw std::overflow_error("vulkan linear weight: resident byte count overflow");
    }
    result->resident_bytes += bytes;
    destination = std::move(replacement);
    pending.push_back({&destination, values, bytes});
  };
  const uint64_t data_count = data_type == ScalarType::kFloat32 ? elements
      : (data_type == ScalarType::kFloat16 || data_type == ScalarType::kBFloat16)
          ? elements : expected_bytes;
  allocate(result->data, data_type, data_count, source.data);
  if (source.weight_scale_count != 0)
    allocate(result->weight_scale, ScalarType::kFloat32,
             source.weight_scale_count, source.weight_scale);
  if (source.block_scale_count != 0)
    allocate(result->block_scale, ScalarType::kUInt8,
             source.block_scale_count, source.block_scale);
  if (source.format == LinearWeightFormat::kNF4) {
    allocate(result->nf4_absmax, ScalarType::kUInt8, source.nf4_absmax_count,
             source.nf4_absmax);
    allocate(result->nf4_quant_map, ScalarType::kFloat32,
             source.nf4_quant_map_count, source.nf4_quant_map);
    allocate(result->nf4_nested_quant_map, ScalarType::kFloat32,
             source.nf4_nested_quant_map_count, source.nf4_nested_quant_map);
    allocate(result->nf4_nested_absmax, ScalarType::kFloat32,
             source.nf4_nested_absmax_count, source.nf4_nested_absmax);
  }
  if (source.pre_quant_scale_count != 0)
    allocate(result->pre_quant_scale, ScalarType::kBFloat16,
             source.pre_quant_scale_count, source.pre_quant_scale_bf16);

  // All immutable components of one weight cross the queue in one job. Host-
  // visible staging is temporary and released immediately after the exact
  // submission completes; steady-state residency contains compressed bytes
  // and auxiliaries only.
  std::vector<Buffer> staging;
  staging.reserve(pending.size());
  for (const auto& item : pending) {
    const uint64_t physical = item.destination->impl_->buffer.size();
    Buffer buffer = context.impl_->pool.allocate(
        physical, BufferUsage::kTransferSource, MemoryUsage::kUpload);
    buffer.write(0, item.values, item.logical_bytes);
    if (physical != item.logical_bytes) {
      const uint32_t zero = 0;
      buffer.write(item.logical_bytes, &zero, physical - item.logical_bytes);
    }
    staging.push_back(std::move(buffer));
  }
  [[maybe_unused]] auto recording_lock = context.impl_->acquire_recorder();
  CommandList list = context.impl_->commands.begin();
  for (size_t i = 0; i < pending.size(); ++i) {
    DeviceTensor::Impl& destination = *pending[i].destination->impl_;
    const uint64_t physical = destination.buffer.size();
    list.barrier(staging[i], BufferAccess::kHostWrite,
                 BufferAccess::kTransferRead, 0, physical);
    list.copy_buffer(staging[i], destination.buffer, physical);
  }
  Submission uploaded = context.impl_->commands.submit(std::move(list));
  uploaded.wait();
  context.impl_->commands.collect();
  for (const auto& item : pending) {
    item.destination->impl_->has_access = true;
    item.destination->impl_->access = BufferAccess::kTransferWrite;
  }
  staging.clear();
  context.impl_->pool.trim();
  return LinearWeight(std::move(result));
}

LinearWeightFormat LinearWeight::format() const {
  if (!impl_) throw std::logic_error("vulkan linear weight: empty weight");
  return impl_->format;
}
uint32_t LinearWeight::out_features() const {
  if (!impl_) throw std::logic_error("vulkan linear weight: empty weight");
  return impl_->out_features;
}
uint32_t LinearWeight::in_features() const {
  if (!impl_) throw std::logic_error("vulkan linear weight: empty weight");
  return impl_->in_features;
}
uint64_t LinearWeight::stored_bytes() const noexcept {
  return impl_ ? impl_->stored_bytes : 0;
}
uint64_t LinearWeight::resident_bytes() const noexcept {
  return impl_ ? impl_->resident_bytes : 0;
}
bool LinearWeight::has_fp8_input_scale() const noexcept {
  return impl_ && impl_->has_fp8_input_scale;
}
float LinearWeight::fp8_input_scale() const {
  if (!impl_ || !impl_->has_fp8_input_scale) {
    throw std::logic_error("vulkan linear weight: FP8 input scale is absent");
  }
  return impl_->fp8_input_scale;
}
bool LinearWeight::full_precision_matrix_mult() const noexcept {
  return impl_ && impl_->full_precision_matrix_mult;
}
bool LinearWeight::has_pre_quant_scale() const noexcept {
  return impl_ && static_cast<bool>(impl_->pre_quant_scale);
}
bool LinearWeight::applies_convrot() const noexcept {
  return impl_ && impl_->convrot;
}
uint32_t LinearWeight::convrot_group() const noexcept {
  return impl_ ? impl_->convrot_group : 0;
}
void LinearWeight::materialize_bf16(TensorBatch& batch, DeviceTensor& dense) const {
  if (!impl_) throw std::logic_error("vulkan linear weight: empty weight");
  batch.materialize_linear_weight(*this, dense, false);
}
void LinearWeight::materialize_f16(TensorBatch& batch, DeviceTensor& dense) const {
  if (!impl_) throw std::logic_error("vulkan linear weight: empty weight");
  batch.materialize_linear_weight(*this, dense, true);
}
void LinearWeight::apply_pre_quant_scale(TensorBatch& batch, DeviceTensor& input,
                                         DeviceTensor& output) const {
  if (!impl_) throw std::logic_error("vulkan linear weight: empty weight");
  batch.transform_linear_activation(*this, input, output, false);
}
void LinearWeight::apply_convrot(TensorBatch& batch, DeviceTensor& input,
                                 DeviceTensor& output) const {
  if (!impl_) throw std::logic_error("vulkan linear weight: empty weight");
  batch.transform_linear_activation(*this, input, output, true);
}

TensorContext::TensorContext(const Device& device, const TensorContextOptions& options)
    : impl_(std::make_unique<Impl>(device, options)) {
  if (options.max_in_flight == 0) {
    throw std::invalid_argument("vulkan tensor: max_in_flight must be nonzero");
  }
}
TensorContext::~TensorContext() = default;
TensorContext::TensorContext(TensorContext&&) noexcept = default;
TensorContext& TensorContext::operator=(TensorContext&&) noexcept = default;

DeviceTensor TensorContext::allocate(const TensorLayout& layout, ScalarType type) {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  [[maybe_unused]] auto recording_lock = impl_->acquire_recorder();
  if (!layout.is_contiguous()) throw std::invalid_argument("vulkan tensor: contiguous layout required");
  const uint64_t logical_bytes = layout.bytes(type);
  if (logical_bytes > std::numeric_limits<uint64_t>::max() - 3) {
    throw std::overflow_error("vulkan tensor: allocation size overflow");
  }
  const uint64_t bytes = std::max<uint64_t>(4, (logical_bytes + 3) & ~uint64_t{3});
  if (bytes > impl_->max_storage_bytes) {
    throw std::out_of_range("vulkan tensor: tensor exceeds storage buffer limit");
  }
  auto tensor = std::make_unique<DeviceTensor::Impl>();
  tensor->buffer = impl_->pool.allocate(
      bytes, BufferUsage::kStorage | BufferUsage::kTransferSource |
                 BufferUsage::kTransferDestination,
      MemoryUsage::kDevice);
  tensor->layout = layout;
  tensor->type = type;
  tensor->logical_bytes = logical_bytes;
  tensor->context = impl_->context_id;
  return DeviceTensor(std::move(tensor));
}

TensorBatch TensorContext::begin_batch() {
  if (!impl_) throw std::logic_error("vulkan tensor: empty context");
  auto recording_lease = impl_->acquire_recorder();
  auto batch = std::make_unique<TensorBatch::Impl>();
  batch->owner = impl_;
  batch->snapshots.resize(static_cast<size_t>(impl_->max_batch_operators) * 7u);
  batch->batch_id = next_context_identity();
  batch->recording_lease = std::move(recording_lease);
  batch->commands = impl_->commands.begin();
  return TensorBatch(std::move(batch));
}

void TensorContext::upload(DeviceTensor& destination, const float* values, uint64_t count) {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  auto dst = impl_->require(destination);
  if (dst->type != ScalarType::kFloat32 || values == nullptr ||
      count != dst->layout.elements()) {
    throw std::invalid_argument("vulkan tensor: upload element count mismatch");
  }
  upload_bytes(destination, values, count * sizeof(float));
}

void TensorContext::upload_bytes(DeviceTensor& destination, const void* values,
                                 uint64_t bytes) {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  [[maybe_unused]] auto recording_lock = impl_->acquire_recorder();
  auto dst = impl_->require(destination);
  if (values == nullptr || bytes != dst->logical_bytes) {
    throw std::invalid_argument("vulkan tensor: upload byte count mismatch");
  }
  const uint64_t physical_bytes = dst->buffer.size();
  impl_->ensure_staging(physical_bytes);
  impl_->upload_buffer.write(0, values, bytes);
  if (physical_bytes != bytes) {
    const uint32_t zero = 0;
    impl_->upload_buffer.write(bytes, &zero, physical_bytes - bytes);
  }
  CommandList list = impl_->commands.begin();
  list.barrier(impl_->upload_buffer, BufferAccess::kHostWrite,
               BufferAccess::kTransferRead, 0, physical_bytes);
  const bool old_has_access = dst->has_access;
  const BufferAccess old_access = dst->access;
  try {
    if (dst->has_access) {
      list.barrier(dst->buffer, dst->access, BufferAccess::kTransferWrite);
    }
    dst->has_access = true;
    dst->access = BufferAccess::kTransferWrite;
    list.copy_buffer(impl_->upload_buffer, dst->buffer, physical_bytes);
    impl_->complete(std::move(list));
  } catch (...) {
    dst->has_access = old_has_access;
    dst->access = old_access;
    throw;
  }
}

void TensorContext::download(DeviceTensor& source, float* values, uint64_t count) {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  auto src = impl_->require(source);
  if (src->type != ScalarType::kFloat32 || values == nullptr ||
      count != src->layout.elements()) {
    throw std::invalid_argument("vulkan tensor: download element count mismatch");
  }
  download_bytes(source, values, count * sizeof(float));
}

void TensorContext::download_bytes(DeviceTensor& source, void* values,
                                   uint64_t bytes) {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  [[maybe_unused]] auto recording_lock = impl_->acquire_recorder();
  auto src = impl_->require(source);
  if (values == nullptr || bytes != src->logical_bytes) {
    throw std::invalid_argument("vulkan tensor: download byte count mismatch");
  }
  const uint64_t physical_bytes = src->buffer.size();
  impl_->ensure_staging(physical_bytes);
  CommandList list = impl_->commands.begin();
  const bool old_has_access = src->has_access;
  const BufferAccess old_access = src->access;
  try {
    if (src->has_access && src->access != BufferAccess::kTransferRead) {
      list.barrier(src->buffer, src->access, BufferAccess::kTransferRead);
    }
    src->has_access = true;
    src->access = BufferAccess::kTransferRead;
    list.copy_buffer(src->buffer, impl_->readback_buffer, physical_bytes);
    list.barrier(impl_->readback_buffer, BufferAccess::kTransferWrite,
                 BufferAccess::kHostRead, 0, physical_bytes);
    impl_->complete(std::move(list));
  } catch (...) {
    src->has_access = old_has_access;
    src->access = old_access;
    throw;
  }
  impl_->readback_buffer.read(0, values, bytes);
}

void TensorContext::copy(DeviceTensor& source, DeviceTensor& destination) {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  TensorBatch batch = begin_batch();
  batch.copy(source, destination);
  batch.submit().wait();
  impl_->commands.collect();
}

void TensorContext::add(DeviceTensor& a, DeviceTensor& b, DeviceTensor& output) {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  TensorBatch batch = begin_batch();
  batch.add(a, b, output);
  batch.submit().wait();
  impl_->commands.collect();
}

bool TensorContext::full_fp32_arithmetic_exactness() const noexcept {
  return impl_ && impl_->full_arithmetic_exact;
}
void TensorContext::require_full_fp32_arithmetic_exactness() const {
  if (!full_fp32_arithmetic_exactness()) {
    throw std::runtime_error(
          "vulkan tensor: device cannot preserve fp32 subnormal inputs/results; "
        "full CUDA-exact fp32 arithmetic is unavailable");
  }
}
bool TensorContext::exact_normalization() const noexcept {
  return impl_ && impl_->exact_vae_norm;
}
void TensorContext::require_exact_normalization() const {
  if (!exact_normalization()) {
    throw std::runtime_error(
        "vulkan tensor: exact normalization requires a compatible shaderInt64-enabled "
        "Vulkan device");
  }
}
bool TensorContext::exact_fp32_vae_normalization() const noexcept {
  return exact_normalization();
}
void TensorContext::require_exact_fp32_vae_normalization() const {
  require_exact_normalization();
}
bool TensorContext::exact_vae_pointwise() const noexcept {
  return impl_ && impl_->exact_vae_pointwise;
}
void TensorContext::require_exact_vae_pointwise() const {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  if (!impl_->exact_vae_pointwise) {
    throw std::runtime_error(
        "vulkan tensor: exact VAE pointwise operations are unavailable on this device/driver");
  }
}
bool TensorContext::exact_audio_vae_primitives() const noexcept {
  return impl_ && impl_->exact_audio;
}
void TensorContext::require_exact_audio_vae_primitives() const {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  if (!impl_->exact_audio) {
    throw std::runtime_error(
        "vulkan audio: exact primitives are unavailable on this device/driver");
  }
}
bool TensorContext::exact_blocked_attention() const noexcept {
  return impl_ && impl_->exact_attention;
}
void TensorContext::require_exact_blocked_attention() const {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  if (!impl_->exact_attention) {
    throw std::runtime_error(
        "vulkan attention: exact blocked attention is unavailable on this device/driver");
  }
}
bool TensorContext::exact_h3_attention() const noexcept {
  return impl_ && impl_->exact_h3_attention;
}
void TensorContext::require_exact_h3_attention() const {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  if (!impl_->exact_h3_attention) {
    throw std::runtime_error(
        "vulkan attention: exact H3 attention is unavailable on this device/driver");
  }
}
bool TensorContext::exact_causal_gqa_attention() const noexcept {
  return impl_ && impl_->exact_causal_gqa_attention;
}
void TensorContext::require_exact_causal_gqa_attention() const {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  if (!impl_->exact_causal_gqa_attention) {
    throw std::runtime_error(
        "vulkan attention: exact causal GQA attention is unavailable on this device/driver");
  }
}

bool TensorContext::native_nvfp4_gemm_available() const noexcept {
  return false;
}
void TensorContext::require_native_nvfp4_gemm() const {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  throw std::runtime_error(
      "vulkan tensor: native NVFP4 GEMM is unavailable; this device/runtime "
      "does not expose an implemented E2M1 block-scaled cooperative-matrix "
      "contract (use streamed NVFP4-to-BF16 execution)");
}

TensorWorkspace& TensorContext::workspace() {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  [[maybe_unused]] auto recording_lock = impl_->acquire_recorder();
  return impl_->scratch;
}
uint64_t TensorContext::reserved_bytes() const {
  if (!impl_) return 0;
  [[maybe_unused]] auto recording_lock = impl_->acquire_recorder();
  const uint64_t primary = impl_->pool.reserved_bytes();
  const uint64_t scratch = impl_->scratch.reserved_bytes();
  return primary > std::numeric_limits<uint64_t>::max() - scratch
             ? std::numeric_limits<uint64_t>::max()
             : primary + scratch;
}
uint64_t TensorContext::pooled_used_bytes() const {
  if (!impl_) return 0;
  [[maybe_unused]] auto recording_lock = impl_->acquire_recorder();
  const uint64_t primary = impl_->pool.used_bytes();
  const uint64_t scratch = impl_->scratch.pooled_used_bytes();
  return primary > std::numeric_limits<uint64_t>::max() - scratch
             ? std::numeric_limits<uint64_t>::max()
             : primary + scratch;
}
uint64_t TensorContext::staging_capacity_bytes() const noexcept {
  return impl_ ? impl_->staging_capacity : 0;
}
uint64_t TensorContext::descriptor_set_allocations() const noexcept {
  return impl_ ? impl_->commands.descriptor_set_allocations() : 0;
}
uint64_t TensorContext::storage_binding_alignment() const noexcept {
  return impl_ ? impl_->storage_binding_alignment : 1;
}

TensorBatch::TensorBatch() = default;
TensorBatch::~TensorBatch() = default;
TensorBatch::TensorBatch(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
TensorBatch::TensorBatch(TensorBatch&&) noexcept = default;
TensorBatch& TensorBatch::operator=(TensorBatch&&) noexcept = default;
TensorBatch::operator bool() const noexcept { return impl_ != nullptr; }

uint32_t TensorBatch::remaining_operator_capacity() const noexcept {
  if (!impl_ || !impl_->owner ||
      impl_->operator_count >= impl_->owner->max_batch_operators) return 0;
  return impl_->owner->max_batch_operators - impl_->operator_count;
}

void TensorBatch::copy(DeviceTensor& source, DeviceTensor& destination) {
  if (!impl_) throw std::logic_error("vulkan tensor: empty batch");
  if (impl_->poisoned) throw std::logic_error("vulkan tensor: batch is poisoned");
  auto src = impl_->owner->require(source);
  auto dst = impl_->owner->require(destination);
  const uint64_t bytes = src->buffer.size();
  if (src.get() == dst.get() || src->type != dst->type ||
      src->logical_bytes != dst->logical_bytes || bytes != dst->buffer.size()) {
    throw std::invalid_argument("vulkan tensor: copy needs distinct equal-sized tensors");
  }
  try {
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kTransferRead);
    impl_->transition(dst, BufferAccess::kTransferWrite);
    impl_->commands.copy_buffer(src->buffer, dst->buffer, bytes);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::add(DeviceTensor& a, DeviceTensor& b, DeviceTensor& output) {
  if (!impl_) throw std::logic_error("vulkan tensor: empty batch");
  if (impl_->poisoned) throw std::logic_error("vulkan tensor: batch is poisoned");
  auto av = impl_->owner->require(a);
  auto bv = impl_->owner->require(b);
  auto out = impl_->owner->require(output);
  const uint64_t count = av->layout.elements();
  if (out.get() == av.get() || out.get() == bv.get() ||
      av->type != ScalarType::kFloat32 || bv->type != ScalarType::kFloat32 ||
      out->type != ScalarType::kFloat32 ||
      count != bv->layout.elements() || count != out->layout.elements() ||
      count == 0 || count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: add needs distinct equal-sized fp32 tensors");
  }
  if (count > impl_->owner->max_storage_bytes / sizeof(float)) {
    throw std::out_of_range("vulkan tensor: add exceeds device dispatch limits");
  }
  const uint32_t dispatch_count = impl_->owner->validate_dispatch(count);
  try {
    impl_->count_operator();
    impl_->transition(av, BufferAccess::kComputeRead);
    impl_->transition(bv, BufferAccess::kComputeRead);
    impl_->transition(out, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters parameters;
    parameters.op = 0;
    parameters.count = dispatch_count;
    impl_->dispatch(parameters, av, bv, out);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::convert(DeviceTensor& source, DeviceTensor& destination) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(source);
  auto dst = impl_->owner->require(destination);
  const uint64_t elements = src->layout.elements();
  if (src.get() == dst.get() || elements != dst->layout.elements() ||
      elements > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: conversion shape mismatch");
  }
  uint32_t op = 0;
  bool narrowing = false;
  if (src->type == ScalarType::kFloat32 && dst->type == ScalarType::kBFloat16) {
    op = 1; narrowing = true;
  } else if (src->type == ScalarType::kBFloat16 && dst->type == ScalarType::kFloat32) {
    op = 2;
  } else if (src->type == ScalarType::kFloat32 && dst->type == ScalarType::kFloat16) {
    op = 3; narrowing = true;
  } else if (src->type == ScalarType::kFloat16 && dst->type == ScalarType::kFloat32) {
    op = 4;
  } else {
    throw std::invalid_argument("vulkan tensor: unsupported dtype conversion");
  }
  const uint64_t invocation_count = narrowing ? (elements + 1) / 2 : elements;
  const uint32_t dispatch_count = impl_->owner->validate_dispatch(invocation_count);
  try {
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters parameters;
    parameters.op = op;
    parameters.count = dispatch_count;
    parameters.p[0] = static_cast<uint32_t>(elements);
    impl_->dispatch(parameters, src, src, dst);
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::transpose_2d(DeviceTensor& source, DeviceTensor& destination) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(source);
  auto dst = impl_->owner->require(destination);
  const auto& a = src->layout;
  const auto& o = dst->layout;
  if (src.get() == dst.get() || src->type != ScalarType::kFloat32 ||
      dst->type != ScalarType::kFloat32 || a.rank != 2 || o.rank != 2 ||
      o.extent[0] != a.extent[1] || o.extent[1] != a.extent[0] ||
      a.elements() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid fp32 transpose");
  }
  if (a.extent[0] > std::numeric_limits<uint32_t>::max() ||
      a.extent[1] > std::numeric_limits<uint32_t>::max()) {
    throw std::out_of_range("vulkan tensor: transpose dimensions exceed uint32 limits");
  }
  const uint32_t dispatch_count = impl_->owner->validate_dispatch(a.elements());
  try {
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters p;
    p.op = 5; p.count = dispatch_count;
    p.p[0] = static_cast<uint32_t>(a.extent[0]);
    p.p[1] = static_cast<uint32_t>(a.extent[1]);
    impl_->dispatch(p, src, src, dst);
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::gather_rows(DeviceTensor& source, DeviceTensor& indices,
                              DeviceTensor& destination) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(source);
  auto idx = impl_->owner->require(indices);
  auto dst = impl_->owner->require(destination);
  const auto& a = src->layout; const auto& ix = idx->layout; const auto& o = dst->layout;
  if (src.get() == dst.get() || src.get() == idx.get() || dst.get() == idx.get() ||
      src->type != ScalarType::kFloat32 || dst->type != ScalarType::kFloat32 ||
      idx->type != ScalarType::kInt32 || a.rank != 2 || ix.rank != 1 || o.rank != 2 ||
      ix.extent[0] != o.extent[0] || a.extent[1] != o.extent[1] ||
      a.elements() > std::numeric_limits<uint32_t>::max() ||
      o.elements() > std::numeric_limits<uint32_t>::max() ||
      a.extent[0] > std::numeric_limits<uint32_t>::max() ||
      a.extent[1] > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid gather rows");
  }
  const uint32_t dispatch_count = impl_->owner->validate_dispatch(o.elements());
  try {
    impl_->count_operator(); impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(idx, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters p; p.op = 6;
    p.count = dispatch_count;
    p.p[0] = static_cast<uint32_t>(a.extent[0]); p.p[1] = static_cast<uint32_t>(a.extent[1]);
    impl_->dispatch(p, src, idx, dst);
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::scatter_rows(DeviceTensor& source, DeviceTensor& indices,
                               DeviceTensor& destination) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(source); auto idx = impl_->owner->require(indices);
  auto dst = impl_->owner->require(destination);
  const auto& a = src->layout; const auto& ix = idx->layout; const auto& o = dst->layout;
  if (src.get() == dst.get() || src.get() == idx.get() || dst.get() == idx.get() ||
      src->type != ScalarType::kFloat32 || dst->type != ScalarType::kFloat32 ||
      idx->type != ScalarType::kInt32 || a.rank != 2 || ix.rank != 1 || o.rank != 2 ||
      ix.extent[0] != a.extent[0] || a.extent[1] != o.extent[1] ||
      a.elements() > std::numeric_limits<uint32_t>::max() ||
      o.elements() > std::numeric_limits<uint32_t>::max() ||
      o.extent[0] > std::numeric_limits<uint32_t>::max() ||
      a.extent[1] > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid scatter rows");
  }
  const uint32_t dispatch_count = impl_->owner->validate_dispatch(a.elements());
  try {
    impl_->count_operator(); impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(idx, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters p; p.op = 7;
    p.count = dispatch_count; p.p[0] = static_cast<uint32_t>(o.extent[0]);
    p.p[1] = static_cast<uint32_t>(a.extent[1]); impl_->dispatch(p, src, idx, dst);
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::add_bias(DeviceTensor& input, DeviceTensor& bias, DeviceTensor& output) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(input); auto bv = impl_->owner->require(bias);
  auto dst = impl_->owner->require(output);
  const auto& a = src->layout; const auto& b = bv->layout; const auto& o = dst->layout;
  if (src.get() == bv.get() || dst.get() == bv.get() ||
      src->type != ScalarType::kFloat32 || bv->type != ScalarType::kFloat32 ||
      dst->type != ScalarType::kFloat32 || a.rank != 2 || b.rank != 1 || o.rank != 2 ||
      b.extent[0] != a.extent[1] || o.extent != a.extent ||
      a.elements() > std::numeric_limits<uint32_t>::max() ||
      a.extent[1] > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid add-bias");
  }
  const uint32_t dispatch_count = impl_->owner->validate_dispatch(a.elements());
  try {
    impl_->count_operator();
    impl_->transition(src, src.get() == dst.get() ? BufferAccess::kComputeReadWrite
                                                  : BufferAccess::kComputeRead);
    impl_->transition(bv, BufferAccess::kComputeRead);
    if (src.get() != dst.get()) impl_->transition(dst, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters p; p.op = 8;
    p.count = dispatch_count; p.p[1] = static_cast<uint32_t>(a.extent[1]);
    impl_->dispatch(p, src, bv, dst);
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::layer_scale_residual_f32(DeviceTensor& x, DeviceTensor& y,
                                           DeviceTensor& bias,
                                           DeviceTensor& scale) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_vae_pointwise) {
    throw std::runtime_error("vulkan tensor: exact VAE pointwise operations are unavailable");
  }
  auto xv = impl_->owner->require(x);
  auto yv = impl_->owner->require(y);
  auto bv = impl_->owner->require(bias);
  auto sv = impl_->owner->require(scale);
  const auto& shape = xv->layout;
  const uint64_t columns = shape.rank == 2 ? shape.extent[1] : 0;
  const uint64_t count = shape.elements();
  if (xv.get() == yv.get() || xv.get() == bv.get() || xv.get() == sv.get() ||
      yv.get() == bv.get() || yv.get() == sv.get() || bv.get() == sv.get() ||
      xv->type != ScalarType::kFloat32 || yv->type != ScalarType::kFloat32 ||
      bv->type != ScalarType::kFloat32 || sv->type != ScalarType::kFloat32 ||
      shape.rank != 2 || yv->layout.extent != shape.extent ||
      bv->layout.rank != 1 || sv->layout.rank != 1 ||
      bv->layout.extent[0] != columns || sv->layout.extent[0] != columns ||
      !shape.is_contiguous() || !yv->layout.is_contiguous() ||
      !bv->layout.is_contiguous() || !sv->layout.is_contiguous() ||
      shape.extent[0] == 0 || columns == 0 ||
      columns > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid fp32 layer-scale residual");
  }
  const uint32_t dispatch_count = impl_->owner->validate_dispatch(count);
  try {
    impl_->count_operator();
    impl_->transition(xv, BufferAccess::kComputeReadWrite);
    impl_->transition(yv, BufferAccess::kComputeRead);
    impl_->transition(bv, BufferAccess::kComputeRead);
    impl_->transition(sv, BufferAccess::kComputeRead);
    TensorContext::Impl::Parameters p;
    p.op = 0; p.count = dispatch_count; p.p[0] = static_cast<uint32_t>(columns);
    impl_->dispatch_vae_pointwise(VaePointwiseOperation::kResidual, p,
                                  {xv, yv, bv, sv});
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::swiglu_bias_f32(DeviceTensor& input, DeviceTensor& bias,
                                  DeviceTensor& output) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_vae_pointwise) {
    throw std::runtime_error("vulkan tensor: exact VAE pointwise operations are unavailable");
  }
  auto src = impl_->owner->require(input);
  auto bv = impl_->owner->require(bias);
  auto dst = impl_->owner->require(output);
  const auto& in_shape = src->layout;
  const auto& out_shape = dst->layout;
  const uint64_t inner = out_shape.rank == 2 ? out_shape.extent[1] : 0;
  const uint64_t count = out_shape.elements();
  const bool doubled_fits = inner <= std::numeric_limits<uint64_t>::max() / 2;
  if (src.get() == bv.get() || src.get() == dst.get() || bv.get() == dst.get() ||
      src->type != ScalarType::kFloat32 || bv->type != ScalarType::kFloat32 ||
      dst->type != ScalarType::kFloat32 || in_shape.rank != 2 ||
      out_shape.rank != 2 || bv->layout.rank != 1 || inner == 0 ||
      !doubled_fits || in_shape.extent[0] != out_shape.extent[0] ||
      in_shape.extent[1] != 2 * inner || bv->layout.extent[0] != 2 * inner ||
      !in_shape.is_contiguous() || !out_shape.is_contiguous() ||
      !bv->layout.is_contiguous() || out_shape.extent[0] == 0 ||
      inner > std::numeric_limits<uint32_t>::max() ||
      in_shape.elements() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid fp32 SwiGLU with bias");
  }
  const uint32_t dispatch_count = impl_->owner->validate_dispatch(count);
  try {
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(bv, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters p;
    p.op = 1; p.count = dispatch_count; p.p[0] = static_cast<uint32_t>(inner);
    impl_->dispatch_vae_pointwise(VaePointwiseOperation::kSwiglu, p,
                                  {src, bv, bv, dst});
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::latent_denorm_f32(DeviceTensor& input, DeviceTensor& mean,
                                    DeviceTensor& std_dev,
                                    DeviceTensor& output) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_vae_pointwise) {
    throw std::runtime_error("vulkan tensor: exact VAE pointwise operations are unavailable");
  }
  auto src = impl_->owner->require(input);
  auto mv = impl_->owner->require(mean);
  auto sv = impl_->owner->require(std_dev);
  auto dst = impl_->owner->require(output);
  const auto& shape = src->layout;
  const uint64_t channels = shape.rank == 2 ? shape.extent[0] : 0;
  const uint64_t voxels = shape.rank == 2 ? shape.extent[1] : 0;
  const uint64_t count = shape.elements();
  if (src.get() == mv.get() || src.get() == sv.get() || src.get() == dst.get() ||
      mv.get() == sv.get() || mv.get() == dst.get() || sv.get() == dst.get() ||
      src->type != ScalarType::kFloat32 || mv->type != ScalarType::kFloat32 ||
      sv->type != ScalarType::kFloat32 || dst->type != ScalarType::kFloat32 ||
      shape.rank != 2 || dst->layout.extent != shape.extent ||
      mv->layout.rank != 1 || sv->layout.rank != 1 ||
      mv->layout.extent[0] != channels || sv->layout.extent[0] != channels ||
      !shape.is_contiguous() || !dst->layout.is_contiguous() ||
      !mv->layout.is_contiguous() || !sv->layout.is_contiguous() ||
      channels == 0 || voxels == 0 ||
      voxels > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid fp32 latent denormalization");
  }
  const uint32_t dispatch_count = impl_->owner->validate_dispatch(count);
  try {
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(mv, BufferAccess::kComputeRead);
    impl_->transition(sv, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters p;
    p.op = 2; p.count = dispatch_count; p.p[0] = static_cast<uint32_t>(voxels);
    impl_->dispatch_vae_pointwise(VaePointwiseOperation::kDenorm, p,
                                  {src, mv, sv, dst});
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::heads_to_tokens_bf16(DeviceTensor& source, DeviceTensor& destination,
                                       uint32_t heads, uint32_t sequence,
                                       uint32_t head_dim) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(source); auto dst = impl_->owner->require(destination);
  const uint64_t elements = checked_multiply(
      checked_multiply(heads, sequence, "heads-to-tokens"), head_dim,
      "heads-to-tokens");
  if (heads == 0 || sequence == 0 || head_dim == 0 ||
      src.get() == dst.get() || src->type != ScalarType::kFloat32 ||
      dst->type != ScalarType::kBFloat16 ||
      src->layout.elements() != elements || dst->layout.elements() != elements ||
      elements > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid heads-to-tokens conversion");
  }
  const uint32_t dispatch_count = impl_->owner->validate_dispatch((elements + 1) / 2);
  try {
    impl_->count_operator(); impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters p; p.op = 9;
    p.count = dispatch_count;
    p.p[0] = heads; p.p[1] = head_dim; p.p[2] = sequence;
    p.p[3] = static_cast<uint32_t>(elements); impl_->dispatch(p, src, src, dst);
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::depth_to_space(DeviceTensor& source, DeviceTensor& destination,
                                 uint32_t time, uint32_t height, uint32_t width,
                                 uint32_t channels, uint32_t patch_time,
                                 uint32_t patch) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(source); auto dst = impl_->owner->require(destination);
  if (time == 0 || height == 0 || width == 0 || channels == 0 ||
      patch_time == 0 || patch == 0) {
    throw std::invalid_argument("vulkan tensor: depth-to-space dimensions must be nonzero");
  }
  const uint64_t tokens = checked_multiply(
      checked_multiply(time, height, "depth-to-space"), width, "depth-to-space");
  const uint64_t patch_volume = checked_multiply(
      checked_multiply(patch_time, patch, "depth-to-space"), patch,
      "depth-to-space");
  const uint64_t elements = checked_multiply(
      checked_multiply(tokens, channels, "depth-to-space"), patch_volume,
      "depth-to-space");
  if (patch_volume > std::numeric_limits<uint32_t>::max() ||
      src.get() == dst.get() || src->type != ScalarType::kFloat32 ||
      dst->type != ScalarType::kFloat32 ||
      src->layout.elements() != elements || dst->layout.elements() != elements ||
      elements > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid depth-to-space tensors");
  }
  const uint32_t dispatch_count = impl_->owner->validate_dispatch(elements);
  try {
    impl_->count_operator(); impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters p; p.op = 10; p.count = dispatch_count;
    p.p[0] = time; p.p[1] = height; p.p[2] = width; p.p[3] = patch_time;
    p.p[4] = patch; p.p[5] = channels; impl_->dispatch(p, src, src, dst);
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::rope_h3_bf16(DeviceTensor& input, DeviceTensor& cosine,
                                DeviceTensor& sine) {
  rope_bf16(input, cosine, sine, 0);
}

void TensorBatch::rope_neox_bf16(DeviceTensor& input, DeviceTensor& cosine,
                                  DeviceTensor& sine) {
  rope_bf16(input, cosine, sine, 1);
}

void TensorBatch::rope_bf16(DeviceTensor& input, DeviceTensor& cosine,
                             DeviceTensor& sine, uint32_t mode) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  auto data = impl_->owner->require(input);
  auto cos_table = impl_->owner->require(cosine);
  auto sin_table = impl_->owner->require(sine);
  const auto& shape = data->layout;
  const uint64_t rows = shape.extent[0];
  const uint64_t heads = shape.extent[1];
  const uint64_t head_dim = shape.extent[2];
  const uint64_t table_width = mode == 0 ? 96 : head_dim;
  const bool groups_overflow = rows != 0 && heads > UINT64_MAX / rows;
  const uint64_t groups = groups_overflow ? 0 : rows * heads;
  if (mode > 1 || shape.rank != 3 || rows == 0 || heads == 0 ||
      head_dim == 0 || (head_dim & 1u) != 0 || head_dim > 256 ||
      (mode == 0 && head_dim < 96) || groups_overflow ||
      groups > std::numeric_limits<uint32_t>::max() ||
      data.get() == cos_table.get() || data.get() == sin_table.get() ||
      cos_table.get() == sin_table.get() ||
      data->type != ScalarType::kBFloat16 ||
      cos_table->type != ScalarType::kFloat32 ||
      sin_table->type != ScalarType::kFloat32 ||
      cos_table->layout.rank != 2 || sin_table->layout.rank != 2 ||
      cos_table->layout.extent[0] != rows ||
      sin_table->layout.extent[0] != rows ||
      cos_table->layout.extent[1] != table_width ||
      sin_table->layout.extent[1] != table_width ||
      !shape.is_contiguous() || !cos_table->layout.is_contiguous() ||
      !sin_table->layout.is_contiguous()) {
    throw std::invalid_argument("vulkan tensor: invalid BF16 RoPE");
  }
  if (!detail::norm_dispatch_fits(groups, impl_->owner->max_dispatch_x)) {
    throw std::out_of_range("vulkan tensor: RoPE row-head count exceeds dispatch limits");
  }
  TensorContext::Impl::Parameters parameters;
  parameters.op = mode;
  parameters.p[0] = static_cast<uint32_t>(rows);
  parameters.p[1] = static_cast<uint32_t>(heads);
  parameters.p[2] = static_cast<uint32_t>(head_dim);
  try {
    impl_->count_operator();
    impl_->transition(data, BufferAccess::kComputeReadWrite);
    impl_->transition(cos_table, BufferAccess::kComputeRead);
    impl_->transition(sin_table, BufferAccess::kComputeRead);
    impl_->dispatch_rope(parameters, static_cast<uint32_t>(groups), data,
                         cos_table, sin_table);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::split_qkv_norm_rope_f32(
    DeviceTensor& qkv, DeviceTensor& bias, DeviceTensor& cosine,
    DeviceTensor& sine, DeviceTensor& q, DeviceTensor& k, DeviceTensor& v,
    uint32_t num_patches, float epsilon) {
  if (!impl_ || impl_->poisoned) {
    throw std::logic_error("vulkan tensor: invalid batch");
  }
  if (!impl_->owner->exact_vae_norm) {
    throw std::runtime_error(
        "vulkan tensor: exact video-VAE fused RoPE is unavailable");
  }
  std::array<std::shared_ptr<DeviceTensor::Impl>, 7> resources{
      impl_->owner->require(qkv), impl_->owner->require(bias),
      impl_->owner->require(cosine), impl_->owner->require(sine),
      impl_->owner->require(q), impl_->owner->require(k),
      impl_->owner->require(v)};
  bool aliases = false;
  for (size_t i = 0; i < resources.size(); ++i) {
    for (size_t j = i + 1; j < resources.size(); ++j) {
      aliases = aliases || resources[i].get() == resources[j].get();
    }
  }
  const auto& input_shape = resources[0]->layout;
  const uint64_t sequence = input_shape.extent[0];
  const bool flattened = input_shape.rank == 2;
  const uint64_t heads = flattened
      ? resources[4]->layout.extent[0] : input_shape.extent[1];
  const bool flattened_width_overflow =
      flattened && heads > std::numeric_limits<uint64_t>::max() / 192u;
  const bool groups_overflow = sequence != 0 && heads > UINT64_MAX / sequence;
  const uint64_t groups = groups_overflow ? 0 : sequence * heads;
  if (aliases || !std::isnormal(epsilon) || epsilon <= 0.0f ||
      (input_shape.rank != 3 && input_shape.rank != 2) || sequence == 0 || heads == 0 ||
      flattened_width_overflow ||
      (flattened ? input_shape.extent[1] != heads * 192
                 : input_shape.extent[2] != 192) || num_patches > sequence ||
      groups_overflow || groups > std::numeric_limits<uint32_t>::max() ||
      (flattened ? (resources[1]->layout.rank != 1 ||
                    resources[1]->layout.extent[0] != heads * 192)
                 : (resources[1]->layout.rank != 2 ||
                    resources[1]->layout.extent[0] != heads ||
                    resources[1]->layout.extent[1] != 192)) ||
      resources[2]->layout.rank != 2 || resources[3]->layout.rank != 2 ||
      resources[2]->layout.extent[0] != sequence ||
      resources[3]->layout.extent[0] != sequence ||
      resources[2]->layout.extent[1] != 48 ||
      resources[3]->layout.extent[1] != 48) {
    throw std::invalid_argument("vulkan tensor: invalid video-VAE fused RoPE");
  }
  for (size_t i = 0; i < resources.size(); ++i) {
    if (resources[i]->type != ScalarType::kFloat32 ||
        !resources[i]->layout.is_contiguous()) {
      throw std::invalid_argument("vulkan tensor: invalid video-VAE fused RoPE");
    }
    if (i >= 4 &&
        (resources[i]->layout.rank != 3 ||
         resources[i]->layout.extent[0] != heads ||
         resources[i]->layout.extent[1] != sequence ||
         resources[i]->layout.extent[2] != 64)) {
      throw std::invalid_argument("vulkan tensor: invalid video-VAE fused RoPE");
    }
  }
  if (!detail::norm_dispatch_fits(groups, impl_->owner->max_dispatch_x)) {
    throw std::out_of_range(
        "vulkan tensor: video-VAE fused RoPE row-head count exceeds dispatch limits");
  }
  TensorContext::Impl::VaeRopeParameters parameters;
  parameters.sequence = static_cast<uint32_t>(sequence);
  parameters.heads = static_cast<uint32_t>(heads);
  parameters.num_patches = num_patches;
  std::memcpy(&parameters.epsilon_bits, &epsilon, sizeof(epsilon));
  try {
    impl_->count_operator();
    for (size_t i = 0; i < 4; ++i) {
      impl_->transition(resources[i], BufferAccess::kComputeRead);
    }
    for (size_t i = 4; i < resources.size(); ++i) {
      impl_->transition(resources[i], BufferAccess::kComputeWrite);
    }
    impl_->dispatch_vae_rope(parameters, static_cast<uint32_t>(groups),
                             resources);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::rms_norm(DeviceTensor& input, DeviceTensor& weight,
                           DeviceTensor& output, float epsilon) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_vae_norm) {
    throw std::runtime_error("vulkan tensor: exact fp32 VAE RMSNorm is unavailable");
  }
  auto src = impl_->owner->require(input);
  auto w = impl_->owner->require(weight);
  auto dst = impl_->owner->require(output);
  const auto& shape = src->layout;
  if (!std::isnormal(epsilon) || epsilon <= 0.0f || src.get() == w.get() ||
      dst.get() == w.get() || src->type != ScalarType::kFloat32 ||
      w->type != ScalarType::kFloat32 || dst->type != ScalarType::kFloat32 ||
      shape.rank != 2 || w->layout.rank != 1 || dst->layout.extent != shape.extent ||
      dst->layout.rank != 2 || w->layout.extent[0] != shape.extent[1] ||
      !shape.is_contiguous() || !w->layout.is_contiguous() ||
      !dst->layout.is_contiguous() ||
      shape.extent[0] > std::numeric_limits<uint32_t>::max() ||
      shape.extent[1] > kMaxExactNormDimension ||
      shape.elements() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid fp32 RMSNorm");
  }
  if (!detail::norm_dispatch_fits(shape.extent[0], impl_->owner->max_dispatch_x)) {
    throw std::out_of_range("vulkan tensor: RMSNorm row count exceeds dispatch limits");
  }
  TensorContext::Impl::NormParameters p;
  p.rows = static_cast<uint32_t>(shape.extent[0]);
  p.dim = static_cast<uint32_t>(shape.extent[1]);
  std::memcpy(&p.epsilon_bits, &epsilon, sizeof(epsilon));
  try {
    impl_->count_operator();
    impl_->transition(src, src.get() == dst.get() ? BufferAccess::kComputeReadWrite
                                                   : BufferAccess::kComputeRead);
    impl_->transition(w, BufferAccess::kComputeRead);
    if (src.get() != dst.get()) impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_norm(impl_->owner->rms_norm_pipeline, p, src, w, w, dst);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::layer_norm(DeviceTensor& input, DeviceTensor& weight,
                             DeviceTensor& bias, DeviceTensor& output,
                             float epsilon) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_vae_norm) {
    throw std::runtime_error("vulkan tensor: exact fp32 VAE LayerNorm is unavailable");
  }
  auto src = impl_->owner->require(input);
  auto w = impl_->owner->require(weight);
  auto b = impl_->owner->require(bias);
  auto dst = impl_->owner->require(output);
  const auto& shape = src->layout;
  if (!std::isnormal(epsilon) || epsilon <= 0.0f || src.get() == w.get() ||
      src.get() == b.get() || w.get() == b.get() || dst.get() == w.get() ||
      dst.get() == b.get() || src->type != ScalarType::kFloat32 ||
      w->type != ScalarType::kFloat32 || b->type != ScalarType::kFloat32 ||
      dst->type != ScalarType::kFloat32 || shape.rank != 2 ||
      w->layout.rank != 1 || b->layout.rank != 1 || dst->layout.rank != 2 ||
      dst->layout.extent != shape.extent || w->layout.extent[0] != shape.extent[1] ||
      b->layout.extent[0] != shape.extent[1] ||
      !shape.is_contiguous() || !w->layout.is_contiguous() ||
      !b->layout.is_contiguous() || !dst->layout.is_contiguous() ||
      shape.extent[0] > std::numeric_limits<uint32_t>::max() ||
      shape.extent[1] > kMaxExactNormDimension ||
      shape.elements() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid fp32 LayerNorm");
  }
  if (!detail::norm_dispatch_fits(shape.extent[0], impl_->owner->max_dispatch_x)) {
    throw std::out_of_range("vulkan tensor: LayerNorm row count exceeds dispatch limits");
  }
  TensorContext::Impl::NormParameters p;
  p.rows = static_cast<uint32_t>(shape.extent[0]);
  p.dim = static_cast<uint32_t>(shape.extent[1]);
  std::memcpy(&p.epsilon_bits, &epsilon, sizeof(epsilon));
  try {
    impl_->count_operator();
    impl_->transition(src, src.get() == dst.get() ? BufferAccess::kComputeReadWrite
                                                   : BufferAccess::kComputeRead);
    impl_->transition(w, BufferAccess::kComputeRead);
    impl_->transition(b, BufferAccess::kComputeRead);
    if (src.get() != dst.get()) impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_norm(impl_->owner->layer_norm_pipeline, p, src, w, b, dst);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::rms_norm_bf16(DeviceTensor& input, DeviceTensor& weight,
                                DeviceTensor& output, float epsilon) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_vae_norm) {
    throw std::runtime_error("vulkan tensor: exact BF16 RMSNorm is unavailable");
  }
  auto src = impl_->owner->require(input);
  auto w = impl_->owner->require(weight);
  auto dst = impl_->owner->require(output);
  const auto& shape = src->layout;
  const uint64_t rows = shape.extent[0];
  const uint64_t dim = shape.extent[1];
  if (!std::isnormal(epsilon) || epsilon <= 0.0f || src.get() == w.get() ||
      dst.get() == w.get() || src->type != ScalarType::kBFloat16 ||
      w->type != ScalarType::kBFloat16 || dst->type != ScalarType::kBFloat16 ||
      shape.rank != 2 || w->layout.rank != 1 || dst->layout.rank != 2 ||
      dst->layout.extent != shape.extent || w->layout.extent[0] != dim ||
      !shape.is_contiguous() || !w->layout.is_contiguous() ||
      !dst->layout.is_contiguous() || rows > std::numeric_limits<uint32_t>::max() ||
      dim > kMaxExactNormDimension ||
      shape.elements() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid BF16 RMSNorm");
  }
  const uint64_t vec = (dim & 7u) == 0 ? 8u : 1u;
  const bool narrow = dim / vec <= 32u;
  const uint64_t groups = narrow ? (rows + 7u) / 8u : rows;
  if (!detail::norm_dispatch_fits(groups, impl_->owner->max_dispatch_x)) {
    throw std::out_of_range("vulkan tensor: BF16 RMSNorm rows exceed dispatch limits");
  }
  TensorContext::Impl::NormParameters p;
  p.rows = static_cast<uint32_t>(rows);
  p.dim = static_cast<uint32_t>(dim);
  std::memcpy(&p.epsilon_bits, &epsilon, sizeof(epsilon));
  try {
    impl_->count_operator();
    impl_->transition(src, src.get() == dst.get() ? BufferAccess::kComputeReadWrite
                                                   : BufferAccess::kComputeRead);
    impl_->transition(w, BufferAccess::kComputeRead);
    if (src.get() != dst.get()) impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_shared_rms(narrow ? impl_->owner->bf16_rms_narrow_pipeline
                                      : impl_->owner->bf16_rms_block_pipeline,
                               p, static_cast<uint32_t>(groups), src, w, dst);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::rms_norm_heads_bf16(DeviceTensor& input,
                                      DeviceTensor& weight,
                                      DeviceTensor& output, uint32_t heads,
                                      uint32_t head_dim, float epsilon) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_vae_norm) {
    throw std::runtime_error("vulkan tensor: exact BF16 head RMSNorm is unavailable");
  }
  auto src = impl_->owner->require(input); auto w = impl_->owner->require(weight);
  auto dst = impl_->owner->require(output);
  const auto& shape = src->layout;
  const bool rank2 = shape.rank == 2;
  const bool rank3 = shape.rank == 3;
  const uint64_t token_rows = (rank2 || rank3) ? shape.extent[0] : 0;
  const uint64_t rows = checked_multiply(token_rows, heads, "BF16 head RMSNorm");
  const uint64_t width = checked_multiply(heads, head_dim, "BF16 head RMSNorm");
  if (!std::isnormal(epsilon) || epsilon <= 0.0f || heads == 0 || head_dim == 0 ||
      src.get() == w.get() || dst.get() == w.get() ||
      src->type != ScalarType::kBFloat16 || w->type != ScalarType::kBFloat16 ||
      dst->type != ScalarType::kBFloat16 || (!rank2 && !rank3) ||
      (rank2 ? shape.extent[1] != width
             : (shape.extent[1] != heads || shape.extent[2] != head_dim)) ||
      w->layout.rank != 1 ||
      w->layout.extent[0] != head_dim || dst->layout.rank != shape.rank ||
      dst->layout.extent != shape.extent || !shape.is_contiguous() ||
      !w->layout.is_contiguous() || !dst->layout.is_contiguous() ||
      rows > std::numeric_limits<uint32_t>::max() ||
      head_dim > kMaxExactNormDimension ||
      shape.elements() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid BF16 head RMSNorm");
  }
  const uint64_t vec = (head_dim & 7u) == 0 ? 8u : 1u;
  const bool narrow = head_dim / vec <= 32u;
  const uint64_t groups = narrow ? (rows + 7u) / 8u : rows;
  if (!detail::norm_dispatch_fits(groups, impl_->owner->max_dispatch_x)) {
    throw std::out_of_range("vulkan tensor: BF16 head RMSNorm rows exceed dispatch limits");
  }
  TensorContext::Impl::NormParameters p;
  p.rows = static_cast<uint32_t>(rows); p.dim = head_dim;
  std::memcpy(&p.epsilon_bits, &epsilon, sizeof(epsilon));
  try {
    impl_->count_operator();
    impl_->transition(src, src.get() == dst.get() ? BufferAccess::kComputeReadWrite
                                                   : BufferAccess::kComputeRead);
    impl_->transition(w, BufferAccess::kComputeRead);
    if (src.get() != dst.get()) impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_shared_rms(narrow ? impl_->owner->bf16_rms_narrow_pipeline
                                      : impl_->owner->bf16_rms_block_pipeline,
                               p, static_cast<uint32_t>(groups), src, w, dst);
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::layer_norm_bf16(DeviceTensor& input, DeviceTensor& weight,
                                  DeviceTensor& bias, DeviceTensor& output,
                                  float epsilon) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_vae_norm) {
    throw std::runtime_error("vulkan tensor: exact BF16 LayerNorm is unavailable");
  }
  auto src = impl_->owner->require(input);
  auto w = impl_->owner->require(weight);
  auto b = impl_->owner->require(bias);
  auto dst = impl_->owner->require(output);
  const auto& shape = src->layout;
  const uint64_t rows = shape.extent[0];
  const uint64_t dim = shape.extent[1];
  if (!std::isnormal(epsilon) || epsilon <= 0.0f || src.get() == w.get() ||
      src.get() == b.get() || w.get() == b.get() || dst.get() == w.get() ||
      dst.get() == b.get() || src->type != ScalarType::kBFloat16 ||
      w->type != ScalarType::kBFloat16 || b->type != ScalarType::kBFloat16 ||
      dst->type != ScalarType::kBFloat16 || shape.rank != 2 ||
      w->layout.rank != 1 || b->layout.rank != 1 || dst->layout.rank != 2 ||
      dst->layout.extent != shape.extent || w->layout.extent[0] != dim ||
      b->layout.extent[0] != dim || !shape.is_contiguous() ||
      !w->layout.is_contiguous() || !b->layout.is_contiguous() ||
      !dst->layout.is_contiguous() || rows > std::numeric_limits<uint32_t>::max() ||
      dim > kMaxExactNormDimension ||
      shape.elements() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid BF16 LayerNorm");
  }
  if (!detail::norm_dispatch_fits(rows, impl_->owner->max_dispatch_x)) {
    throw std::out_of_range("vulkan tensor: BF16 LayerNorm rows exceed dispatch limits");
  }
  TensorContext::Impl::NormParameters p;
  p.rows = static_cast<uint32_t>(rows);
  p.dim = static_cast<uint32_t>(dim);
  std::memcpy(&p.epsilon_bits, &epsilon, sizeof(epsilon));
  try {
    impl_->count_operator();
    impl_->transition(src, src.get() == dst.get() ? BufferAccess::kComputeReadWrite
                                                   : BufferAccess::kComputeRead);
    impl_->transition(w, BufferAccess::kComputeRead);
    impl_->transition(b, BufferAccess::kComputeRead);
    if (src.get() != dst.get()) impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_shared_layer(impl_->owner->bf16_layer_pipeline, p, src, w, b, dst);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::rms_norm_modulate_bf16(DeviceTensor& input, DeviceTensor& weight,
                                         DeviceTensor& scale, DeviceTensor& shift,
                                         DeviceTensor& selectors, DeviceTensor& output,
                                         float epsilon) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  impl_->record_shared_mod(false, input, weight, scale, shift, selectors, output,
                           epsilon);
}

void TensorBatch::rms_norm_modulate_f32(DeviceTensor& input, DeviceTensor& weight,
                                        DeviceTensor& scale, DeviceTensor& shift,
                                        DeviceTensor& selectors, DeviceTensor& output,
                                        float epsilon) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  impl_->record_shared_mod(true, input, weight, scale, shift, selectors, output,
                           epsilon);
}

void TensorBatch::rms_norm_modulate_bf16_table(
    DeviceTensor& input, DeviceTensor& weight, DeviceTensor& tables,
    uint32_t mod_rows, uint32_t scale_table, uint32_t shift_table, DeviceTensor& selectors,
    DeviceTensor& output, float epsilon) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  impl_->record_shared_mod_table(input, weight, tables, mod_rows, scale_table,
                                 shift_table, selectors, output, epsilon);
}

void TensorBatch::dit_add_gated_bf16(DeviceTensor& residual,
                                     DeviceTensor& branch,
                                     DeviceTensor& gate,
                                     DeviceTensor& selectors) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_dit_pointwise) {
    throw std::runtime_error("vulkan DiT: exact pointwise operations are unavailable");
  }
  auto x = impl_->owner->require(residual);
  auto b = impl_->owner->require(branch);
  auto g = impl_->owner->require(gate);
  auto a = impl_->owner->require(selectors);
  const auto& shape = x->layout;
  const uint64_t rows = shape.rank == 2 ? shape.extent[0] : 0;
  const uint64_t dim = shape.rank == 2 ? shape.extent[1] : 0;
  const uint64_t mod_rows = g->layout.rank == 2 ? g->layout.extent[0] : 0;
  const uint64_t count = shape.elements();
  if (x.get() == b.get() || x.get() == g.get() || x.get() == a.get() ||
      b.get() == g.get() || b.get() == a.get() || g.get() == a.get() ||
      shape.rank != 2 || b->layout.rank != 2 || b->layout.extent != shape.extent ||
      g->layout.rank != 2 || g->layout.extent[1] != dim ||
      a->layout.rank != 1 || a->layout.extent[0] != rows ||
      x->type != ScalarType::kBFloat16 || b->type != ScalarType::kBFloat16 ||
      g->type != ScalarType::kFloat32 || a->type != ScalarType::kInt32 ||
      !shape.is_contiguous() || !b->layout.is_contiguous() ||
      !g->layout.is_contiguous() || !a->layout.is_contiguous() ||
      rows == 0 || dim == 0 || mod_rows == 0 ||
      rows > std::numeric_limits<uint32_t>::max() ||
      dim > std::numeric_limits<uint32_t>::max() ||
      mod_rows > std::numeric_limits<uint32_t>::max() ||
      count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan DiT: invalid gated residual tensors");
  }
  TensorContext::Impl::DitParameters p;
  p.op = 0; p.rows = static_cast<uint32_t>(rows);
  p.dim = static_cast<uint32_t>(dim);
  p.mod_rows = static_cast<uint32_t>(mod_rows);
  p.count = static_cast<uint32_t>(count);
  try {
    impl_->count_operator();
    impl_->transition(x, BufferAccess::kComputeReadWrite);
    impl_->transition(b, BufferAccess::kComputeRead);
    impl_->transition(g, BufferAccess::kComputeRead);
    impl_->transition(a, BufferAccess::kComputeRead);
    impl_->dispatch_dit(p, {x, b, g, a, x});
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::dit_add_gated_bf16_table(
    DeviceTensor& residual, DeviceTensor& branch, DeviceTensor& tables,
    uint32_t mod_rows, uint32_t gate_table, DeviceTensor& selectors) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_dit_pointwise)
    throw std::runtime_error("vulkan DiT: exact pointwise operations are unavailable");
  auto x=impl_->owner->require(residual), b=impl_->owner->require(branch);
  auto t=impl_->owner->require(tables), a=impl_->owner->require(selectors);
  const auto& shape=x->layout;const uint64_t rows=shape.rank==2?shape.extent[0]:0;
  const uint64_t dim=shape.rank==2?shape.extent[1]:0;
  const uint64_t table_count=6;
  const uint64_t stride=t->layout.rank==1?t->layout.extent[0]/table_count:0;
  const uint64_t logical=checked_multiply(mod_rows,dim,"DiT gate table");
  const uint64_t count=shape.elements();
  if(gate_table>=table_count||x.get()==b.get()||x.get()==t.get()||x.get()==a.get()||
     b.get()==t.get()||b.get()==a.get()||t.get()==a.get()||shape.rank!=2||
     b->layout.rank!=2||b->layout.extent!=shape.extent||t->layout.rank!=1||
     t->layout.extent[0]%table_count!=0||stride<logical||a->layout.rank!=1||a->layout.extent[0]!=rows||
     x->type!=ScalarType::kBFloat16||b->type!=ScalarType::kBFloat16||
     t->type!=ScalarType::kFloat32||a->type!=ScalarType::kInt32||
     !shape.is_contiguous()||!b->layout.is_contiguous()||!t->layout.is_contiguous()||
     !a->layout.is_contiguous()||rows==0||dim==0||mod_rows==0||rows>UINT32_MAX||
     dim>UINT32_MAX||count>UINT32_MAX)
    throw std::invalid_argument("vulkan DiT: invalid gated residual table tensors");
  TensorContext::Impl::DitParameters p;p.op=0;p.rows=static_cast<uint32_t>(rows);
  p.dim=static_cast<uint32_t>(dim);p.mod_rows=static_cast<uint32_t>(mod_rows);
  p.count=static_cast<uint32_t>(count);
  const uint64_t offset=stride*4*gate_table;
  try {impl_->count_operator();impl_->transition(x,BufferAccess::kComputeReadWrite);
    impl_->transition(b,BufferAccess::kComputeRead);impl_->transition(t,BufferAccess::kComputeRead);
    impl_->transition(a,BufferAccess::kComputeRead);impl_->dispatch_dit(p,{x,b,t,a,x},offset);
  } catch (...) {impl_->poisoned=true;throw;}
}

void TensorBatch::dit_swiglu_bf16(DeviceTensor& fused, DeviceTensor& output) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_dit_pointwise) {
    throw std::runtime_error("vulkan DiT: exact pointwise operations are unavailable");
  }
  auto src = impl_->owner->require(fused);
  auto dst = impl_->owner->require(output);
  const uint64_t rows = src->layout.rank == 2 ? src->layout.extent[0] : 0;
  const uint64_t doubled = src->layout.rank == 2 ? src->layout.extent[1] : 0;
  const uint64_t inner = doubled / 2;
  const uint64_t count = checked_multiply(rows, inner, "DiT SwiGLU");
  if (src.get() == dst.get() || src->layout.rank != 2 ||
      (doubled & 1u) != 0 || rows == 0 || inner == 0 ||
      dst->layout.rank != 2 || dst->layout.extent[0] != rows ||
      dst->layout.extent[1] != inner ||
      src->type != ScalarType::kBFloat16 || dst->type != ScalarType::kBFloat16 ||
      !src->layout.is_contiguous() || !dst->layout.is_contiguous() ||
      rows > std::numeric_limits<uint32_t>::max() ||
      inner > std::numeric_limits<uint32_t>::max() ||
      count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan DiT: invalid SwiGLU tensors");
  }
  TensorContext::Impl::DitParameters p;
  p.op = 1; p.rows = static_cast<uint32_t>(rows);
  p.dim = static_cast<uint32_t>(inner);
  p.count = static_cast<uint32_t>(count);
  try {
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_dit(p, {src, src, src, src, dst});
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::dit_expand_adaln(DeviceTensor& weight, DeviceTensor& bias,
                                   DeviceTensor& code, DeviceTensor& output,
                                   uint32_t num_modality, uint32_t num_param,
                                   uint32_t channels) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_dit_pointwise) {
    throw std::runtime_error("vulkan DiT: exact pointwise operations are unavailable");
  }
  auto w = impl_->owner->require(weight); auto b = impl_->owner->require(bias);
  auto c = impl_->owner->require(code); auto dst = impl_->owner->require(output);
  const uint64_t timesteps = c->layout.rank == 2 ? c->layout.extent[0] : 0;
  const uint64_t rank = c->layout.rank == 2 ? c->layout.extent[1] : 0;
  const uint64_t features = checked_multiply(
      checked_multiply(num_modality, num_param, "DiT AdaLN"), channels,
      "DiT AdaLN");
  const uint64_t count = checked_multiply(timesteps, features, "DiT AdaLN");
  const uint64_t table_elements = checked_multiply(
      checked_multiply(timesteps, num_modality, "DiT AdaLN table"), channels,
      "DiT AdaLN table");
  const bool canonical_output = dst->layout.rank == 3 &&
      dst->layout.extent[0] == num_param &&
      dst->layout.extent[1] == timesteps * num_modality &&
      dst->layout.extent[2] == channels;
  const uint64_t flat_stride = dst->layout.rank == 1 && num_param != 0 &&
      dst->layout.extent[0] % num_param == 0
          ? dst->layout.extent[0] / num_param : 0;
  const bool flat_output = flat_stride >= table_elements &&
      (flat_stride * sizeof(float)) % impl_->owner->storage_binding_alignment == 0;
  if (w.get() == b.get() || w.get() == c.get() || w.get() == dst.get() ||
      b.get() == c.get() || b.get() == dst.get() || c.get() == dst.get() ||
      num_modality == 0 || num_param == 0 || channels == 0 ||
      w->layout.rank != 2 || w->layout.extent[0] != features ||
      w->layout.extent[1] != rank || b->layout.rank != 1 ||
      b->layout.extent[0] != features || c->layout.rank != 2 ||
      timesteps == 0 || rank == 0 || (!canonical_output && !flat_output) ||
      w->type != ScalarType::kFloat32 || b->type != ScalarType::kFloat32 ||
      c->type != ScalarType::kFloat32 || dst->type != ScalarType::kFloat32 ||
      !w->layout.is_contiguous() || !b->layout.is_contiguous() ||
      !c->layout.is_contiguous() || !dst->layout.is_contiguous() ||
      timesteps > std::numeric_limits<uint32_t>::max() ||
      rank > std::numeric_limits<uint32_t>::max() ||
      count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan DiT: invalid AdaLN tensors");
  }
  TensorContext::Impl::DitParameters p;
  p.op = 2; p.rows = static_cast<uint32_t>(timesteps); p.dim = channels;
  p.num_t = static_cast<uint32_t>(timesteps); p.num_modality = num_modality;
  p.num_param = num_param; p.rank = static_cast<uint32_t>(rank);
  p.count = static_cast<uint32_t>(count);
  p.unused[0] = flat_output ? static_cast<uint32_t>(flat_stride) : 0;
  try {
    impl_->count_operator(); impl_->transition(w, BufferAccess::kComputeRead);
    impl_->transition(b, BufferAccess::kComputeRead);
    impl_->transition(c, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_dit(p, {w, b, c, c, dst});
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::group_norm_silu_f16_affine(DeviceTensor& input,
                                              DeviceTensor& weight,
                                              DeviceTensor& bias,
                                              DeviceTensor& output,
                                              uint32_t groups, float epsilon) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_vae_norm) {
    throw std::runtime_error("vulkan tensor: exact VAE GroupNorm+SiLU is unavailable");
  }
  auto src = impl_->owner->require(input);
  auto w = impl_->owner->require(weight);
  auto b = impl_->owner->require(bias);
  auto dst = impl_->owner->require(output);
  const auto& shape = src->layout;
  const uint64_t channels = shape.extent[0];
  uint64_t spatial = 0;
  if (shape.rank == 3 &&
      shape.extent[1] <= std::numeric_limits<uint64_t>::max() / shape.extent[2]) {
    spatial = shape.extent[1] * shape.extent[2];
  }
  const uint64_t channels_per_group = groups == 0 ? 0 : channels / groups;
  const bool group_count_overflows = spatial != 0 &&
      channels_per_group > std::numeric_limits<uint64_t>::max() / spatial;
  const uint64_t group_count = group_count_overflows ? 0 : channels_per_group * spatial;
  if (!std::isnormal(epsilon) || epsilon <= 0.0f || groups == 0 ||
      shape.rank != 3 || spatial == 0 || channels == 0 || channels % groups != 0 ||
      group_count_overflows || channels > std::numeric_limits<uint32_t>::max() ||
      spatial > std::numeric_limits<uint32_t>::max() ||
      group_count > kMaxExactNormDimension ||
      shape.elements() > std::numeric_limits<uint32_t>::max() ||
      src.get() == w.get() || src.get() == b.get() || w.get() == b.get() ||
      dst.get() == w.get() || dst.get() == b.get() ||
      src->type != ScalarType::kFloat32 || dst->type != ScalarType::kFloat32 ||
      w->type != ScalarType::kFloat16 || b->type != ScalarType::kFloat16 ||
      w->layout.rank != 1 || b->layout.rank != 1 || dst->layout.rank != 3 ||
      w->layout.extent[0] != channels || b->layout.extent[0] != channels ||
      dst->layout.extent != shape.extent || !shape.is_contiguous() ||
      !w->layout.is_contiguous() || !b->layout.is_contiguous() ||
      !dst->layout.is_contiguous()) {
    throw std::invalid_argument("vulkan tensor: invalid VAE GroupNorm+SiLU");
  }
  if (!detail::norm_dispatch_fits(groups, impl_->owner->max_dispatch_x)) {
    throw std::out_of_range("vulkan tensor: GroupNorm group count exceeds dispatch limits");
  }
  TensorContext::Impl::NormParameters p;
  p.rows = static_cast<uint32_t>(channels);
  p.dim = static_cast<uint32_t>(spatial);
  p.mod_rows = groups;
  std::memcpy(&p.epsilon_bits, &epsilon, sizeof(epsilon));
  try {
    impl_->count_operator();
    impl_->transition(src, src.get() == dst.get() ? BufferAccess::kComputeReadWrite
                                                   : BufferAccess::kComputeRead);
    impl_->transition(w, BufferAccess::kComputeRead);
    impl_->transition(b, BufferAccess::kComputeRead);
    if (src.get() != dst.get()) impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_group_norm(p, src, w, b, dst);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::audio_conv1d(DeviceTensor& input, DeviceTensor& weight,
                               DeviceTensor* bias, DeviceTensor& output,
                               const vae::AudioConv1DDesc& desc) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_audio) {
    throw std::runtime_error("vulkan audio: exact primitives are unavailable");
  }
  desc.validate();
  auto src = impl_->owner->require(input);
  auto w = impl_->owner->require(weight);
  auto dst = impl_->owner->require(output);
  auto bv = bias != nullptr ? impl_->owner->require(*bias) : w;
  const bool valid_bias = bias == nullptr ||
      (bv->type == ScalarType::kFloat32 && bv->layout.rank == 1 &&
       bv->layout.extent[0] == desc.out_channels &&
       bv->layout.is_contiguous());
  const bool valid_input =
      (src->layout.rank == 3 && src->layout.extent[0] == desc.batch &&
       src->layout.extent[1] == desc.in_channels &&
       src->layout.extent[2] == desc.length_in) ||
      (src->layout.rank == 1 &&
       src->layout.elements() >= desc.input_elements());
  const bool valid_output =
      (dst->layout.rank == 3 && dst->layout.extent[0] == desc.batch &&
       dst->layout.extent[1] == desc.out_channels &&
       dst->layout.extent[2] == desc.length_out) ||
      (dst->layout.rank == 1 &&
       dst->layout.elements() >= desc.output_elements());
  if (src.get() == w.get() || src.get() == dst.get() || w.get() == dst.get() ||
      (bias != nullptr && (bv.get() == src.get() || bv.get() == w.get() ||
                           bv.get() == dst.get())) ||
      src->type != ScalarType::kFloat32 || w->type != ScalarType::kFloat32 ||
      dst->type != ScalarType::kFloat32 || !valid_bias || !valid_input ||
      w->layout.rank != 3 || w->layout.extent[0] != desc.out_channels ||
      w->layout.extent[1] != desc.in_channels ||
      w->layout.extent[2] != desc.kernel || !valid_output ||
      !src->layout.is_contiguous() || !w->layout.is_contiguous() ||
      !dst->layout.is_contiguous() ||
      desc.input_elements() > std::numeric_limits<uint32_t>::max() ||
      desc.output_elements() > std::numeric_limits<uint32_t>::max() ||
      desc.weight_elements() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan audio: invalid Conv1D tensors");
  }
  TensorContext::Impl::AudioParameters p;
  p.op = 0; p.batch = desc.batch; p.in_channels = desc.in_channels;
  p.out_channels = desc.out_channels; p.length_in = desc.length_in;
  p.length_out = desc.length_out; p.kernel = desc.kernel;
  p.padding_or_stride = desc.padding;
  p.dilation_or_padding = desc.dilation;
  p.count = static_cast<uint32_t>(desc.output_elements());
  p.scalar_bits = bias != nullptr ? 1u : 0u;
  try {
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(w, BufferAccess::kComputeRead);
    if (bias != nullptr) impl_->transition(bv, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_audio(p, {src, w, bv, bv, dst});
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::audio_conv_transpose1d(
    DeviceTensor& input, DeviceTensor& weight, DeviceTensor* bias,
    DeviceTensor& output, const vae::AudioConvTranspose1DDesc& desc) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_audio) {
    throw std::runtime_error("vulkan audio: exact primitives are unavailable");
  }
  desc.validate();
  auto src = impl_->owner->require(input);
  auto w = impl_->owner->require(weight);
  auto dst = impl_->owner->require(output);
  auto bv = bias != nullptr ? impl_->owner->require(*bias) : w;
  const bool valid_bias = bias == nullptr ||
      (bv->type == ScalarType::kFloat32 && bv->layout.rank == 1 &&
       bv->layout.extent[0] == desc.out_channels &&
       bv->layout.is_contiguous());
  const bool valid_input =
      (src->layout.rank == 3 && src->layout.extent[0] == desc.batch &&
       src->layout.extent[1] == desc.in_channels &&
       src->layout.extent[2] == desc.length_in) ||
      (src->layout.rank == 1 &&
       src->layout.elements() >= desc.input_elements());
  const bool valid_output =
      (dst->layout.rank == 3 && dst->layout.extent[0] == desc.batch &&
       dst->layout.extent[1] == desc.out_channels &&
       dst->layout.extent[2] == desc.length_out) ||
      (dst->layout.rank == 1 &&
       dst->layout.elements() >= desc.output_elements());
  if (src.get() == w.get() || src.get() == dst.get() || w.get() == dst.get() ||
      (bias != nullptr && (bv.get() == src.get() || bv.get() == w.get() ||
                           bv.get() == dst.get())) ||
      src->type != ScalarType::kFloat32 || w->type != ScalarType::kFloat32 ||
      dst->type != ScalarType::kFloat32 || !valid_bias || !valid_input ||
      w->layout.rank != 3 ||
      w->layout.extent[0] != desc.in_channels ||
      w->layout.extent[1] != desc.out_channels ||
      w->layout.extent[2] != desc.kernel || !valid_output ||
      !src->layout.is_contiguous() || !w->layout.is_contiguous() ||
      !dst->layout.is_contiguous() ||
      desc.input_elements() > std::numeric_limits<uint32_t>::max() ||
      desc.output_elements() > std::numeric_limits<uint32_t>::max() ||
      desc.weight_elements() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan audio: invalid ConvTranspose1D tensors");
  }
  TensorContext::Impl::AudioParameters p;
  p.op = 1; p.batch = desc.batch; p.in_channels = desc.in_channels;
  p.out_channels = desc.out_channels; p.length_in = desc.length_in;
  p.length_out = desc.length_out; p.kernel = desc.kernel;
  p.padding_or_stride = desc.stride;
  p.dilation_or_padding = desc.padding;
  p.count = static_cast<uint32_t>(desc.output_elements());
  p.scalar_bits = bias != nullptr ? 1u : 0u;
  try {
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(w, BufferAccess::kComputeRead);
    if (bias != nullptr) impl_->transition(bv, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_audio(p, {src, w, bv, bv, dst});
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::audio_add_inplace(DeviceTensor& input_output,
                                    DeviceTensor& branch, uint64_t live_count) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_audio) throw std::runtime_error("vulkan audio: exact primitives are unavailable");
  auto x = impl_->owner->require(input_output);
  auto y = impl_->owner->require(branch);
  const uint64_t capacity = x->layout.elements();
  const uint64_t count = live_count == 0 ? capacity : live_count;
  if (x.get() == y.get() || x->type != ScalarType::kFloat32 ||
      y->type != ScalarType::kFloat32 ||
      (live_count == 0 && (x->layout.extent != y->layout.extent ||
                           x->layout.rank != y->layout.rank)) ||
      capacity < count || y->layout.elements() < count ||
      !x->layout.is_contiguous() || !y->layout.is_contiguous() || count == 0 ||
      count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan audio: invalid in-place add tensors");
  }
  TensorContext::Impl::AudioParameters p; p.op = 2;
  p.count = static_cast<uint32_t>(count);
  try {
    impl_->count_operator(); impl_->transition(x, BufferAccess::kComputeReadWrite);
    impl_->transition(y, BufferAccess::kComputeRead);
    impl_->dispatch_audio(p, {x, y, y, y, x});
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::audio_copy_prefix(DeviceTensor& source,
                                    DeviceTensor& destination,
                                    uint64_t count) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_audio)
    throw std::runtime_error("vulkan audio: exact primitives are unavailable");
  auto src = impl_->owner->require(source);
  auto dst = impl_->owner->require(destination);
  const uint64_t bytes = checked_multiply(count, sizeof(float),
                                           "audio copy prefix");
  if (src.get() == dst.get() || count == 0 ||
      src->type != ScalarType::kFloat32 || dst->type != ScalarType::kFloat32 ||
      !src->layout.is_contiguous() || !dst->layout.is_contiguous() ||
      src->logical_bytes < bytes || dst->logical_bytes < bytes) {
    throw std::invalid_argument("vulkan audio: invalid prefix copy");
  }
  try {
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kTransferRead);
    impl_->transition(dst, BufferAccess::kTransferWrite);
    impl_->commands.copy_buffer(src->buffer, dst->buffer, bytes);
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::audio_scale_inplace(DeviceTensor& input_output, float scale,
                                      uint64_t live_count) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_audio) throw std::runtime_error("vulkan audio: exact primitives are unavailable");
  auto x = impl_->owner->require(input_output);
  const uint64_t capacity = x->layout.elements();
  const uint64_t count = live_count == 0 ? capacity : live_count;
  if (x->type != ScalarType::kFloat32 || !x->layout.is_contiguous() ||
      count == 0 || capacity < count ||
      count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan audio: invalid in-place scale tensor");
  }
  TensorContext::Impl::AudioParameters p; p.op = 3;
  p.count = static_cast<uint32_t>(count);
  std::memcpy(&p.scalar_bits, &scale, sizeof(scale));
  try {
    impl_->count_operator(); impl_->transition(x, BufferAccess::kComputeReadWrite);
    impl_->dispatch_audio(p, {x, x, x, x, x});
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::audio_clamp_inplace(DeviceTensor& input_output, float lower,
                                      float upper, uint64_t live_count) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_audio) throw std::runtime_error("vulkan audio: exact primitives are unavailable");
  auto x = impl_->owner->require(input_output);
  const uint64_t capacity = x->layout.elements();
  const uint64_t count = live_count == 0 ? capacity : live_count;
  if (!std::isfinite(lower) || !std::isfinite(upper) || lower > upper ||
      x->type != ScalarType::kFloat32 || !x->layout.is_contiguous() ||
      count == 0 || capacity < count ||
      count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan audio: invalid in-place clamp");
  }
  TensorContext::Impl::AudioParameters p; p.op = 4;
  p.count = static_cast<uint32_t>(count);
  std::memcpy(&p.kernel, &lower, sizeof(lower));
  std::memcpy(&p.padding_or_stride, &upper, sizeof(upper));
  try {
    impl_->count_operator(); impl_->transition(x, BufferAccess::kComputeReadWrite);
    impl_->dispatch_audio(p, {x, x, x, x, x});
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::audio_interleave(DeviceTensor& planar,
                                   DeviceTensor& interleaved, uint32_t batch,
                                   uint32_t frames) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_audio) throw std::runtime_error("vulkan audio: exact primitives are unavailable");
  auto src = impl_->owner->require(planar);
  auto dst = impl_->owner->require(interleaved);
  const uint64_t count = checked_multiply(batch, frames, "audio interleave");
  const bool valid_source =
      (src->layout.rank == 3 && src->layout.extent[0] == batch &&
       src->layout.extent[1] == 1 && src->layout.extent[2] == frames) ||
      (src->layout.rank == 1 && src->layout.elements() >= count);
  const bool valid_destination =
      (dst->layout.rank == 2 && dst->layout.extent[0] == frames &&
       dst->layout.extent[1] == batch) ||
      (dst->layout.rank == 1 && dst->layout.elements() >= count);
  if (batch == 0 || frames == 0 || src.get() == dst.get() ||
      src->type != ScalarType::kFloat32 || dst->type != ScalarType::kFloat32 ||
      !valid_source || !valid_destination || !src->layout.is_contiguous() ||
      !dst->layout.is_contiguous() ||
      count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan audio: invalid planar interleave tensors");
  }
  TensorContext::Impl::AudioParameters p; p.op = 5; p.batch = batch;
  p.length_in = frames; p.count = static_cast<uint32_t>(count);
  try {
    impl_->count_operator(); impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_audio(p, {src, src, src, src, dst});
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::audio_snake_beta_inplace(
    DeviceTensor& input_output, DeviceTensor& log_alpha,
    DeviceTensor& log_beta, uint32_t batch, uint32_t channels,
    uint32_t length) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_audio) throw std::runtime_error("vulkan audio: exact primitives are unavailable");
  auto x = impl_->owner->require(input_output);
  auto alpha = impl_->owner->require(log_alpha);
  auto beta = impl_->owner->require(log_beta);
  const uint64_t count = checked_multiply(
      checked_multiply(batch, channels, "audio Snake"), length, "audio Snake");
  const bool valid_x =
      (x->layout.rank == 3 && x->layout.extent[0] == batch &&
       x->layout.extent[1] == channels && x->layout.extent[2] == length) ||
      (x->layout.rank == 1 && x->layout.elements() >= count);
  if (batch == 0 || channels == 0 || length == 0 || x.get() == alpha.get() ||
      x.get() == beta.get() || alpha.get() == beta.get() ||
      x->type != ScalarType::kFloat32 || alpha->type != ScalarType::kFloat32 ||
      beta->type != ScalarType::kFloat32 || !valid_x || alpha->layout.rank != 1 ||
      beta->layout.rank != 1 || alpha->layout.extent[0] != channels ||
      beta->layout.extent[0] != channels || !x->layout.is_contiguous() ||
      !alpha->layout.is_contiguous() || !beta->layout.is_contiguous() ||
      count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan audio: invalid SnakeBeta tensors");
  }
  TensorContext::Impl::AudioParameters p; p.op = 6; p.batch = batch;
  p.out_channels = channels; p.length_in = length;
  p.count = static_cast<uint32_t>(count);
  try {
    impl_->count_operator(); impl_->transition(x, BufferAccess::kComputeReadWrite);
    impl_->transition(alpha, BufferAccess::kComputeRead);
    impl_->transition(beta, BufferAccess::kComputeRead);
    impl_->dispatch_audio(p, {x, x, alpha, beta, x});
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::audio_aa_upsample_snake(
    DeviceTensor& input, DeviceTensor& filter, DeviceTensor& log_alpha,
    DeviceTensor& log_beta, DeviceTensor& output, uint32_t batch,
    uint32_t channels, uint32_t length_in) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_audio) throw std::runtime_error("vulkan audio: exact primitives are unavailable");
  if (length_in > std::numeric_limits<uint32_t>::max() / 2u) {
    throw std::out_of_range("vulkan audio: AA upsample length overflow");
  }
  const uint32_t length_out = length_in * 2u;
  auto src = impl_->owner->require(input); auto f = impl_->owner->require(filter);
  auto alpha = impl_->owner->require(log_alpha);
  auto beta = impl_->owner->require(log_beta); auto dst = impl_->owner->require(output);
  const uint64_t count = checked_multiply(
      checked_multiply(batch, channels, "audio AA upsample"), length_out,
      "audio AA upsample");
  const uint64_t input_count = checked_multiply(
      checked_multiply(batch, channels, "audio AA upsample input"), length_in,
      "audio AA upsample input");
  const bool valid_source =
      (src->layout.rank == 3 && src->layout.extent[0] == batch &&
       src->layout.extent[1] == channels && src->layout.extent[2] == length_in) ||
      (src->layout.rank == 1 && src->layout.elements() >= input_count);
  const bool valid_destination =
      (dst->layout.rank == 3 && dst->layout.extent[0] == batch &&
       dst->layout.extent[1] == channels && dst->layout.extent[2] == length_out) ||
      (dst->layout.rank == 1 && dst->layout.elements() >= count);
  if (batch == 0 || channels == 0 || length_in == 0 || src.get() == f.get() ||
      src.get() == alpha.get() || src.get() == beta.get() || src.get() == dst.get() ||
      f.get() == alpha.get() || f.get() == beta.get() || f.get() == dst.get() ||
      alpha.get() == beta.get() || alpha.get() == dst.get() || beta.get() == dst.get() ||
      src->type != ScalarType::kFloat32 || f->type != ScalarType::kFloat32 ||
      alpha->type != ScalarType::kFloat32 || beta->type != ScalarType::kFloat32 ||
      dst->type != ScalarType::kFloat32 || !valid_source || f->layout.rank != 1 ||
      f->layout.extent[0] != 12 || alpha->layout.rank != 1 ||
      beta->layout.rank != 1 || alpha->layout.extent[0] != channels ||
      beta->layout.extent[0] != channels || !valid_destination ||
      !src->layout.is_contiguous() ||
      !f->layout.is_contiguous() || !alpha->layout.is_contiguous() ||
      !beta->layout.is_contiguous() || !dst->layout.is_contiguous() ||
      count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan audio: invalid AA upsample tensors");
  }
  TensorContext::Impl::AudioParameters p; p.op = 7; p.batch = batch;
  p.out_channels = channels; p.length_in = length_in;
  p.length_out = length_out; p.count = static_cast<uint32_t>(count);
  try {
    impl_->count_operator(); impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(f, BufferAccess::kComputeRead);
    impl_->transition(alpha, BufferAccess::kComputeRead);
    impl_->transition(beta, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_audio(p, {src, f, alpha, beta, dst});
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::audio_aa_downsample(
    DeviceTensor& input, DeviceTensor& filter, DeviceTensor& output,
    uint32_t batch, uint32_t channels, uint32_t length_in,
    uint32_t length_out) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_audio) throw std::runtime_error("vulkan audio: exact primitives are unavailable");
  auto src = impl_->owner->require(input); auto f = impl_->owner->require(filter);
  auto dst = impl_->owner->require(output);
  const uint64_t count = checked_multiply(
      checked_multiply(batch, channels, "audio AA downsample"), length_out,
      "audio AA downsample");
  const uint64_t input_count = checked_multiply(
      checked_multiply(batch, channels, "audio AA downsample input"), length_in,
      "audio AA downsample input");
  const bool valid_source =
      (src->layout.rank == 3 && src->layout.extent[0] == batch &&
       src->layout.extent[1] == channels && src->layout.extent[2] == length_in) ||
      (src->layout.rank == 1 && src->layout.elements() >= input_count);
  const bool valid_destination =
      (dst->layout.rank == 3 && dst->layout.extent[0] == batch &&
       dst->layout.extent[1] == channels && dst->layout.extent[2] == length_out) ||
      (dst->layout.rank == 1 && dst->layout.elements() >= count);
  if (batch == 0 || channels == 0 || length_in == 0 ||
      length_out != (length_in - 1u) / 2u + 1u || src.get() == f.get() ||
      src.get() == dst.get() || f.get() == dst.get() ||
      src->type != ScalarType::kFloat32 || f->type != ScalarType::kFloat32 ||
      dst->type != ScalarType::kFloat32 || !valid_source || f->layout.rank != 1 ||
      f->layout.extent[0] != 12 || !valid_destination ||
      !src->layout.is_contiguous() ||
      !f->layout.is_contiguous() || !dst->layout.is_contiguous() ||
      count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan audio: invalid AA downsample tensors");
  }
  TensorContext::Impl::AudioParameters p; p.op = 8; p.batch = batch;
  p.out_channels = channels; p.length_in = length_in;
  p.length_out = length_out; p.count = static_cast<uint32_t>(count);
  try {
    impl_->count_operator(); impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(f, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_audio(p, {src, f, f, f, dst});
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::materialize_linear_weight(const LinearWeight& weight,
                                             DeviceTensor& dense, bool fp16) {
  if (!impl_ || impl_->poisoned) {
    throw std::logic_error("vulkan tensor: invalid batch");
  }
  if (!weight.impl_) {
    throw std::logic_error("vulkan linear weight: empty weight");
  }
  auto output = impl_->owner->require(dense);
  auto data = impl_->owner->require(weight.impl_->data);
  const uint64_t elements = checked_multiply(weight.impl_->out_features,
                                              weight.impl_->in_features,
                                              "linear materialization");
  const ScalarType output_type = fp16 ? ScalarType::kFloat16
                                      : ScalarType::kBFloat16;
  if (dense.layout().rank != 2 ||
      dense.layout().extent[0] != weight.impl_->out_features ||
      dense.layout().extent[1] != weight.impl_->in_features ||
      !dense.layout().is_contiguous() || output->type != output_type ||
      elements > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(
        "vulkan linear weight: dense target must be contiguous [out,in]");
  }

  std::array<std::shared_ptr<DeviceTensor::Impl>, 6> resources;
  resources.fill(data);
  resources[5] = output;
  uint32_t op = 0;
  switch (weight.impl_->format) {
    case LinearWeightFormat::kFloat32: op = 0; break;
    case LinearWeightFormat::kFloat16: op = 1; break;
    case LinearWeightFormat::kBFloat16: op = 2; break;
    case LinearWeightFormat::kFloat8E4M3:
      op = 3;
      resources[1] = impl_->owner->require(weight.impl_->weight_scale);
      break;
    case LinearWeightFormat::kInt8:
      op = 4;
      resources[1] = impl_->owner->require(weight.impl_->weight_scale);
      break;
    case LinearWeightFormat::kNVFloat4:
      op = 5;
      resources[1] = impl_->owner->require(weight.impl_->block_scale);
      break;
    case LinearWeightFormat::kNF4:
      op = 6;
      resources[1] = impl_->owner->require(weight.impl_->nf4_absmax);
      resources[2] = impl_->owner->require(weight.impl_->nf4_quant_map);
      resources[3] = impl_->owner->require(weight.impl_->nf4_nested_quant_map);
      resources[4] = impl_->owner->require(weight.impl_->nf4_nested_absmax);
      break;
  }
  for (size_t i = 0; i + 1 < resources.size(); ++i) {
    if (resources[i].get() == output.get()) {
      throw std::invalid_argument(
          "vulkan linear weight: dense target aliases persistent storage");
    }
  }
  const uint64_t invocations = 1 + (elements - 1) / 2;
  impl_->owner->validate_dispatch(invocations);

  TensorContext::Impl::WeightParameters parameters;
  parameters.op = op;
  parameters.count = static_cast<uint32_t>(elements);
  parameters.out_features = fp16 ? 1u : 0u;
  parameters.in_features = weight.impl_->in_features;
  parameters.block_size = weight.impl_->nf4_block_size;
  parameters.nested_block_size = weight.impl_->nf4_nested_block_size;
  const float scalar = op == 5 ? weight.impl_->global_scale
                               : weight.impl_->nf4_nested_offset;
  std::memcpy(&parameters.scalar_bits, &scalar, sizeof(scalar));
  const uint32_t groups = static_cast<uint32_t>((invocations + 63) / 64);
  try {
    impl_->count_operator();
    for (size_t i = 0; i + 1 < resources.size(); ++i) {
      bool duplicate = false;
      for (size_t j = 0; j < i; ++j) {
        duplicate = duplicate || resources[j].get() == resources[i].get();
      }
      if (!duplicate) impl_->transition(resources[i], BufferAccess::kComputeRead);
    }
    impl_->transition(output, BufferAccess::kComputeWrite);
    impl_->dispatch_weight(parameters, groups, resources);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::transform_linear_activation(const LinearWeight& weight,
                                               DeviceTensor& input,
                                               DeviceTensor& output,
                                               bool convrot) {
  if (!impl_ || impl_->poisoned) {
    throw std::logic_error("vulkan tensor: invalid batch");
  }
  if (!weight.impl_) {
    throw std::logic_error("vulkan linear weight: empty weight");
  }
  auto source = impl_->owner->require(input);
  auto destination = impl_->owner->require(output);
  const auto& shape = source->layout;
  const uint64_t elements = shape.elements();
  if (source.get() == destination.get() || shape.rank != 2 ||
      shape.extent[1] != weight.impl_->in_features ||
      destination->layout.rank != 2 ||
      destination->layout.extent != shape.extent ||
      !shape.is_contiguous() || !destination->layout.is_contiguous() ||
      source->type != destination->type ||
      (source->type != ScalarType::kBFloat16 &&
       source->type != ScalarType::kFloat32) ||
      elements > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(
        "vulkan linear weight: transform needs distinct contiguous [rows,in] tensors");
  }

  std::array<std::shared_ptr<DeviceTensor::Impl>, 6> resources;
  resources.fill(source);
  resources[5] = destination;
  TensorContext::Impl::WeightParameters parameters;
  parameters.count = static_cast<uint32_t>(elements);
  parameters.in_features = weight.impl_->in_features;
  uint32_t groups = 0;
  if (convrot) {
    if (!weight.impl_->convrot) {
      throw std::invalid_argument("vulkan linear weight: ConvRot is not enabled");
    }
    parameters.op = source->type == ScalarType::kFloat32 ? 10u : 9u;
    parameters.group = weight.impl_->convrot_group;
    const uint64_t group_count = elements / weight.impl_->convrot_group;
    if (group_count == 0 || group_count > impl_->owner->max_dispatch_x) {
      throw std::out_of_range(
          "vulkan linear weight: ConvRot exceeds dispatch limits");
    }
    groups = static_cast<uint32_t>(group_count);
    const float normalization =
        1.0f / std::sqrt(static_cast<float>(weight.impl_->convrot_group));
    std::memcpy(&parameters.scalar_bits, &normalization, sizeof(normalization));
  } else {
    if (!weight.impl_->pre_quant_scale) {
      throw std::invalid_argument(
          "vulkan linear weight: AWQ pre-scale is not present");
    }
    resources[1] = impl_->owner->require(weight.impl_->pre_quant_scale);
    parameters.op = source->type == ScalarType::kFloat32 ? 8u : 7u;
    const uint64_t invocations = source->type == ScalarType::kFloat32
        ? elements : 1 + (elements - 1) / 2;
    impl_->owner->validate_dispatch(invocations);
    groups = static_cast<uint32_t>((invocations + 63) / 64);
  }
  try {
    impl_->count_operator();
    impl_->transition(source, BufferAccess::kComputeRead);
    if (!convrot) {
      impl_->transition(resources[1], BufferAccess::kComputeRead);
    }
    impl_->transition(destination, BufferAccess::kComputeWrite);
    impl_->dispatch_weight(parameters, groups, resources);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

PreparedF16Activation::PreparedF16Activation() = default;
PreparedF16Activation::~PreparedF16Activation() = default;
PreparedF16Activation::PreparedF16Activation(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
PreparedF16Activation::PreparedF16Activation(PreparedF16Activation&&) noexcept = default;
PreparedF16Activation& PreparedF16Activation::operator=(
    PreparedF16Activation&&) noexcept = default;
PreparedF16Activation::operator bool() const noexcept { return impl_ != nullptr; }
uint64_t PreparedF16Activation::reserved_bytes() const noexcept {
  return impl_ ? impl_->tensor.layout().bytes(impl_->tensor.type()) : 0;
}

PreparedF16Activation PreparedF16Activation::create(
    TensorContext& context, uint32_t max_rows, uint32_t in_features) {
  if (!context.impl_) {
    throw std::logic_error("vulkan gemm: moved-from tensor context");
  }
  if (max_rows == 0 || in_features == 0) {
    throw std::invalid_argument(
        "vulkan gemm: fp16 activation dimensions must be nonzero");
  }
  auto result = std::make_shared<Impl>();
  result->owner = context.impl_;
  result->max_rows = max_rows;
  result->in_features = in_features;
  const uint64_t shape[] = {max_rows, in_features};
  result->tensor = context.allocate(
      TensorLayout::contiguous(shape, 2), ScalarType::kFloat16);
  return PreparedF16Activation(std::move(result));
}

PreparedF16ActivationView PreparedF16Activation::prepare(
    TensorBatch& batch, DeviceTensor& input, uint32_t rows,
    uint32_t input_row_offset) {
  if (!impl_) throw std::logic_error("vulkan gemm: empty fp16 activation slot");
  if (!batch.impl_) throw std::logic_error("vulkan gemm: empty batch");
  if (batch.impl_->poisoned) throw std::logic_error("vulkan gemm: batch is poisoned");
  if (batch.impl_->owner.get() != impl_->owner.get()) {
    throw std::invalid_argument(
        "vulkan gemm: fp16 activation slot and batch contexts differ");
  }
  auto source = impl_->owner->require(input);
  auto destination = impl_->owner->require(impl_->tensor);
  const bool valid_range = rows != 0 && rows <= impl_->max_rows &&
      source->layout.rank == 2 && source->layout.extent[1] == impl_->in_features &&
      input_row_offset <= source->layout.extent[0] &&
      rows <= source->layout.extent[0] - input_row_offset;
  if (!valid_range || source->type != ScalarType::kFloat32 ||
      !source->layout.is_contiguous() || source.get() == destination.get()) {
    throw std::invalid_argument(
        "vulkan gemm: fp16 activation source does not match the slot");
  }
  const uint64_t elements = checked_multiply(rows, impl_->in_features,
                                             "gemm fp16 prepare");
  const uint64_t pairs = (elements + 1) / 2;
  const uint64_t groups = (pairs + 63) / 64;
  if (groups == 0 || impl_->owner->max_dispatch_x == 0) {
    throw std::out_of_range("vulkan gemm: fp16 preparation dispatch is invalid");
  }
  const uint64_t gx = std::min<uint64_t>(groups, impl_->owner->max_dispatch_x);
  const uint64_t gy = (groups + gx - 1) / gx;
  if (gy > impl_->owner->max_dispatch_y) {
    throw std::out_of_range(
        "vulkan gemm: fp16 preparation exceeds dispatch limits");
  }
  if (impl_->generation == std::numeric_limits<uint64_t>::max()) {
    throw std::overflow_error("vulkan gemm: fp16 activation generation exhausted");
  }
  TensorContext::Impl::GemmPrepareParameters parameters;
  parameters.rows = rows;
  parameters.in_features = impl_->in_features;
  parameters.input_row_offset = input_row_offset;
  parameters.groups_x = static_cast<uint32_t>(gx);
  PreparedF16ActivationView view(
      impl_, batch.impl_->batch_id, rows, impl_->generation + 1);
  try {
    batch.impl_->count_operator();
    batch.impl_->transition(source, BufferAccess::kComputeRead);
    batch.impl_->transition(destination, BufferAccess::kComputeWrite);
    auto& bindings = impl_->owner->gemm_prepare_bindings;
    bindings[0].buffer = &source->buffer;
    bindings[0].bytes = source->buffer.size();
    bindings[1].buffer = &destination->buffer;
    bindings[1].bytes = destination->buffer.size();
    batch.impl_->commands.bind_compute(
        impl_->owner->gemm_prepare_pipeline, bindings);
    batch.impl_->commands.push_constants(&parameters, sizeof(parameters));
    batch.impl_->commands.dispatch(static_cast<uint32_t>(gx),
                                   static_cast<uint32_t>(gy));
  } catch (...) {
    batch.impl_->poisoned = true;
    throw;
  }
  ++impl_->generation;
  return view;
}

PreparedF16ActivationView::PreparedF16ActivationView() = default;
PreparedF16ActivationView::~PreparedF16ActivationView() = default;
PreparedF16ActivationView::PreparedF16ActivationView(
    std::shared_ptr<void> slot, uintptr_t batch_id, uint32_t rows,
    uint64_t generation) noexcept
    : slot_(std::move(slot)), batch_id_(batch_id), rows_(rows),
      generation_(generation) {}
PreparedF16ActivationView::PreparedF16ActivationView(
    PreparedF16ActivationView&&) noexcept = default;
PreparedF16ActivationView& PreparedF16ActivationView::operator=(
    PreparedF16ActivationView&&) noexcept = default;
uint32_t PreparedF16ActivationView::rows() const noexcept {
  return slot_ ? rows_ : 0;
}
PreparedF16ActivationView::operator bool() const noexcept { return slot_ != nullptr; }

StreamedNVFP4WeightCache::StreamedNVFP4WeightCache() = default;
StreamedNVFP4WeightCache::~StreamedNVFP4WeightCache() = default;
StreamedNVFP4WeightCache::StreamedNVFP4WeightCache(
    std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
StreamedNVFP4WeightCache::StreamedNVFP4WeightCache(
    StreamedNVFP4WeightCache&&) noexcept = default;
StreamedNVFP4WeightCache& StreamedNVFP4WeightCache::operator=(
    StreamedNVFP4WeightCache&&) noexcept = default;
StreamedNVFP4WeightCache::operator bool() const noexcept {
  return impl_ != nullptr;
}

StreamedNVFP4WeightCache StreamedNVFP4WeightCache::create(
    TensorContext& context, uint64_t max_weight_elements) {
  if (!context.impl_) {
    throw std::logic_error("vulkan nvfp4 stream: moved-from tensor context");
  }
  if (max_weight_elements == 0 ||
      max_weight_elements > std::numeric_limits<uint64_t>::max() / 2) {
    throw std::invalid_argument(
        "vulkan nvfp4 stream: cache capacity is invalid");
  }
  auto result = std::make_shared<Impl>();
  result->owner = context.impl_;
  result->capacity_elements = max_weight_elements;
  const uint64_t shape[] = {max_weight_elements};
  result->dense = context.allocate(
      TensorLayout::contiguous(shape, 1), ScalarType::kBFloat16);
  return StreamedNVFP4WeightCache(std::move(result));
}

PreparedNVFP4WeightView StreamedNVFP4WeightCache::prepare(
    TensorBatch& batch, const LinearWeight& weight,
    const DenseGemmPlan& plan) {
  if (!impl_) throw std::logic_error("vulkan nvfp4 stream: empty cache");
  if (!batch.impl_) throw std::logic_error("vulkan nvfp4 stream: empty batch");
  if (!weight.impl_) throw std::logic_error("vulkan nvfp4 stream: empty weight");
  if (!plan.impl_) throw std::logic_error("vulkan nvfp4 stream: empty GEMM plan");
  if (batch.impl_->poisoned) {
    throw std::logic_error("vulkan nvfp4 stream: batch is poisoned");
  }
  if (batch.impl_->owner.get() != impl_->owner.get() ||
      plan.impl_->owner.get() != impl_->owner.get()) {
    throw std::invalid_argument(
        "vulkan nvfp4 stream: cache, batch and plan contexts differ");
  }
  const DenseGemmPlanDesc& desc = plan.impl_->desc;
  const uint64_t elements = checked_multiply(
      weight.impl_->out_features, weight.impl_->in_features,
      "nvfp4 streamed weight");
  if (weight.impl_->format != LinearWeightFormat::kNVFloat4 ||
      desc.mode != DenseGemmMode::kBFloat16 ||
      desc.out_features != weight.impl_->out_features ||
      desc.in_features != weight.impl_->in_features ||
      elements > impl_->capacity_elements ||
      elements > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(
        "vulkan nvfp4 stream: weight does not match the BF16 plan/cache");
  }
  if (batch.impl_->operator_count == batch.impl_->owner->max_batch_operators) {
    throw std::logic_error("vulkan tensor: batch operator limit exceeded");
  }
  if (impl_->generation == std::numeric_limits<uint64_t>::max()) {
    throw std::overflow_error("vulkan nvfp4 stream: generation exhausted");
  }
  impl_->owner->validate_dispatch(1 + (elements - 1) / 2);
  auto data = impl_->owner->require(weight.impl_->data);
  auto block_scale = impl_->owner->require(weight.impl_->block_scale);
  auto dense = impl_->owner->require(impl_->dense);
  if (data.get() == dense.get() || block_scale.get() == dense.get()) {
    throw std::invalid_argument(
        "vulkan nvfp4 stream: cache aliases persistent weight storage");
  }
  const uint64_t shape[] = {weight.impl_->out_features,
                            weight.impl_->in_features};
  const TensorLayout layout = TensorLayout::contiguous(shape, 2);
  const uint64_t logical_bytes = checked_multiply(
      elements, 2, "nvfp4 streamed BF16 bytes");
  if (logical_bytes > dense->buffer.size()) {
    throw std::out_of_range(
        "vulkan nvfp4 stream: weight exceeds the cache allocation");
  }
  PreparedNVFP4WeightView view(
      impl_, batch.impl_->batch_id, impl_->generation + 1,
      weight.impl_->out_features, weight.impl_->in_features,
      weight.impl_->full_precision_matrix_mult);
  dense->layout = layout;
  dense->logical_bytes = logical_bytes;
  weight.materialize_bf16(batch, impl_->dense);
  impl_->batch_id = batch.impl_->batch_id;
  impl_->out_features = weight.impl_->out_features;
  impl_->in_features = weight.impl_->in_features;
  ++impl_->generation;
  return view;
}

uint64_t StreamedNVFP4WeightCache::capacity_elements() const noexcept {
  return impl_ ? impl_->capacity_elements : 0;
}
uint64_t StreamedNVFP4WeightCache::dense_bytes() const noexcept {
  return impl_ ? impl_->capacity_elements * 2 : 0;
}

PreparedNVFP4WeightView::PreparedNVFP4WeightView() = default;
PreparedNVFP4WeightView::~PreparedNVFP4WeightView() = default;
PreparedNVFP4WeightView::PreparedNVFP4WeightView(
    std::shared_ptr<void> cache, uintptr_t batch_id, uint64_t generation,
    uint32_t out_features, uint32_t in_features,
    bool full_precision) noexcept
    : cache_(std::move(cache)), batch_id_(batch_id), generation_(generation),
      out_features_(out_features), in_features_(in_features),
      full_precision_(full_precision) {}
PreparedNVFP4WeightView::PreparedNVFP4WeightView(
    PreparedNVFP4WeightView&&) noexcept = default;
PreparedNVFP4WeightView& PreparedNVFP4WeightView::operator=(
    PreparedNVFP4WeightView&&) noexcept = default;
uint32_t PreparedNVFP4WeightView::out_features() const noexcept {
  return cache_ ? out_features_ : 0;
}
uint32_t PreparedNVFP4WeightView::in_features() const noexcept {
  return cache_ ? in_features_ : 0;
}
bool PreparedNVFP4WeightView::full_precision_matrix_mult() const noexcept {
  return cache_ && full_precision_;
}
PreparedNVFP4WeightView::operator bool() const noexcept {
  return cache_ != nullptr;
}

DenseGemmPlan::DenseGemmPlan() = default;
DenseGemmPlan::~DenseGemmPlan() = default;
DenseGemmPlan::DenseGemmPlan(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
DenseGemmPlan::DenseGemmPlan(DenseGemmPlan&&) noexcept = default;
DenseGemmPlan& DenseGemmPlan::operator=(DenseGemmPlan&&) noexcept = default;
DenseGemmPlan::operator bool() const noexcept { return impl_ != nullptr; }

DenseGemmPlan DenseGemmPlan::create(TensorContext& context,
                                    const DenseGemmPlanDesc& desc) {
  if (!context.impl_) throw std::logic_error("vulkan gemm: moved-from context");
  switch (desc.mode) {
    case DenseGemmMode::kBFloat16:
    case DenseGemmMode::kFloat16Vae:
    case DenseGemmMode::kFloat32:
      break;
    default:
      throw std::invalid_argument("vulkan gemm: invalid execution mode");
  }
  if (desc.max_rows == 0 || desc.out_features == 0 || desc.in_features == 0) {
    throw std::invalid_argument("vulkan gemm: dimensions must be nonzero");
  }
  detail::GemmDispatchGeometry plan_geometry;
  if (!detail::gemm_dispatch_geometry(
          desc.max_rows, desc.out_features, 16, 16,
          context.impl_->max_dispatch_x, context.impl_->max_dispatch_y,
          &plan_geometry)) {
    throw std::out_of_range("vulkan gemm: plan exceeds device dispatch limits");
  }
  const uint64_t weight_elements = checked_multiply(
      desc.out_features, desc.in_features, "gemm weight");
  const uint64_t input_elements = checked_multiply(
      desc.max_rows, desc.in_features, "gemm input");
  const uint64_t output_elements = checked_multiply(
      desc.max_rows, desc.out_features, "gemm output");
  const uint64_t input_bytes = checked_multiply(
      input_elements, desc.mode == DenseGemmMode::kBFloat16 ? 2u : 4u,
      "gemm input bytes");
  const uint64_t weight_bytes = checked_multiply(
      weight_elements, desc.mode == DenseGemmMode::kFloat32 ? 4u : 2u,
      "gemm weight bytes");
  const uint64_t output_bytes = checked_multiply(
      output_elements, desc.mode == DenseGemmMode::kBFloat16 ? 2u : 4u,
      "gemm output bytes");
  if (input_bytes > context.impl_->max_storage_bytes ||
      weight_bytes > context.impl_->max_storage_bytes ||
      output_bytes > context.impl_->max_storage_bytes) {
    throw std::out_of_range("vulkan gemm: matrix exceeds storage-buffer limits");
  }
  const bool valid_bias =
      desc.bias == DenseGemmBias::kNone ||
      (desc.mode == DenseGemmMode::kBFloat16 &&
       (desc.bias == DenseGemmBias::kFloat32 ||
        desc.bias == DenseGemmBias::kBFloat16)) ||
      (desc.mode == DenseGemmMode::kFloat32 &&
       desc.bias == DenseGemmBias::kFloat32);
  if (!valid_bias) {
    throw std::invalid_argument("vulkan gemm: bias is invalid for this mode");
  }
  auto result = std::make_shared<Impl>();
  result->owner = context.impl_;
  result->desc = desc;
  return DenseGemmPlan(std::move(result));
}

const DenseGemmPlanDesc& DenseGemmPlan::description() const {
  if (!impl_) throw std::logic_error("vulkan gemm: empty plan");
  return impl_->desc;
}

void DenseGemmPlan::record(TensorBatch& batch, DeviceTensor& input,
                           DeviceTensor& prepared_weight, DeviceTensor& output,
                           uint32_t rows, uint32_t input_row_offset,
                           uint32_t output_row_offset, DeviceTensor* bias) const {
  record_impl(batch, input, prepared_weight, output, rows, input_row_offset,
              output_row_offset, bias, false);
}

void DenseGemmPlan::record(TensorBatch& batch, PreparedF16ActivationView& input,
                           DeviceTensor& prepared_weight, DeviceTensor& output,
                           uint32_t output_row_offset, DeviceTensor* bias) const {
  if (!impl_) throw std::logic_error("vulkan gemm: empty plan");
  if (!input.slot_) throw std::logic_error("vulkan gemm: empty prepared fp16 view");
  if (!batch.impl_ || input.batch_id_ != batch.impl_->batch_id) {
    throw std::invalid_argument(
        "vulkan gemm: prepared fp16 view belongs to another batch");
  }
  auto slot = std::static_pointer_cast<PreparedF16Activation::Impl>(input.slot_);
  if (input.generation_ != slot->generation) {
    throw std::invalid_argument("vulkan gemm: prepared fp16 view was superseded");
  }
  if (impl_->desc.mode != DenseGemmMode::kFloat16Vae ||
      slot->owner.get() != impl_->owner.get() ||
      slot->in_features != impl_->desc.in_features) {
    throw std::invalid_argument(
        "vulkan gemm: prepared fp16 view does not match the plan");
  }
  record_impl(batch, slot->tensor, prepared_weight, output,
              input.rows_, 0, output_row_offset, bias, true);
}

void DenseGemmPlan::record(
    TensorBatch& batch, DeviceTensor& input,
    PreparedNVFP4WeightView& prepared_weight, DeviceTensor& output,
    uint32_t rows, uint32_t input_row_offset, uint32_t output_row_offset,
    DeviceTensor* bias) const {
  if (!impl_) throw std::logic_error("vulkan gemm: empty plan");
  if (!prepared_weight.cache_) {
    throw std::logic_error("vulkan gemm: empty streamed NVFP4 view");
  }
  if (!batch.impl_ || prepared_weight.batch_id_ != batch.impl_->batch_id) {
    throw std::invalid_argument(
        "vulkan gemm: streamed NVFP4 view belongs to another batch");
  }
  auto cache = std::static_pointer_cast<StreamedNVFP4WeightCache::Impl>(
      prepared_weight.cache_);
  if (prepared_weight.generation_ != cache->generation ||
      prepared_weight.batch_id_ != cache->batch_id) {
    throw std::invalid_argument(
        "vulkan gemm: streamed NVFP4 view was superseded");
  }
  if (cache->owner.get() != impl_->owner.get() ||
      impl_->desc.mode != DenseGemmMode::kBFloat16 ||
      prepared_weight.out_features_ != impl_->desc.out_features ||
      prepared_weight.in_features_ != impl_->desc.in_features) {
    throw std::invalid_argument(
        "vulkan gemm: streamed NVFP4 view does not match the plan");
  }
  record_impl(batch, input, cache->dense, output, rows, input_row_offset,
              output_row_offset, bias, false);
}

void DenseGemmPlan::record_impl(
    TensorBatch& batch, DeviceTensor& input, DeviceTensor& prepared_weight,
    DeviceTensor& output, uint32_t rows, uint32_t input_row_offset,
    uint32_t output_row_offset, DeviceTensor* bias,
    bool input_is_prepared_f16) const {
  if (!impl_) throw std::logic_error("vulkan gemm: empty plan");
  if (!batch.impl_) throw std::logic_error("vulkan gemm: empty batch");
  if (batch.impl_->poisoned) throw std::logic_error("vulkan gemm: batch is poisoned");
  if (batch.impl_->owner.get() != impl_->owner.get()) {
    throw std::invalid_argument("vulkan gemm: plan and batch contexts differ");
  }
  const DenseGemmPlanDesc& desc = impl_->desc;
  if (desc.mode == DenseGemmMode::kFloat16Vae && !input_is_prepared_f16) {
    throw std::invalid_argument(
        "vulkan gemm: fp16 VAE mode requires a prepared activation view");
  }
  if (rows == 0 || rows > desc.max_rows) {
    throw std::invalid_argument("vulkan gemm: row count is outside the plan");
  }
  auto src = impl_->owner->require(input);
  auto weight = impl_->owner->require(prepared_weight);
  auto dst = impl_->owner->require(output);
  std::shared_ptr<DeviceTensor::Impl> bias_tensor;
  if (bias != nullptr) bias_tensor = impl_->owner->require(*bias);
  if ((desc.bias == DenseGemmBias::kNone) != (bias == nullptr)) {
    throw std::invalid_argument("vulkan gemm: bias presence differs from plan");
  }
  const ScalarType input_type = input_is_prepared_f16
      ? ScalarType::kFloat16
      : desc.mode == DenseGemmMode::kBFloat16
          ? ScalarType::kBFloat16 : ScalarType::kFloat32;
  const ScalarType weight_type = desc.mode == DenseGemmMode::kBFloat16
      ? ScalarType::kBFloat16 : desc.mode == DenseGemmMode::kFloat16Vae
          ? ScalarType::kFloat16 : ScalarType::kFloat32;
  const ScalarType output_type = desc.mode == DenseGemmMode::kBFloat16
      ? ScalarType::kBFloat16 : ScalarType::kFloat32;
  const ScalarType bias_type = desc.bias == DenseGemmBias::kBFloat16
      ? ScalarType::kBFloat16 : ScalarType::kFloat32;
  const bool aliases = src.get() == weight.get() || src.get() == dst.get() ||
      weight.get() == dst.get() ||
      (bias_tensor && (bias_tensor.get() == src.get() ||
                       bias_tensor.get() == weight.get() ||
                       bias_tensor.get() == dst.get()));
  const bool output_matrix = dst->layout.rank == 2;
  const bool output_heads = dst->layout.rank == 3 &&
      dst->layout.extent[1] <= std::numeric_limits<uint64_t>::max() /
                                   dst->layout.extent[2] &&
      dst->layout.extent[1] * dst->layout.extent[2] == desc.out_features;
  const bool row_ranges_valid =
      src->layout.rank == 2 && (output_matrix || output_heads) &&
      input_row_offset <= src->layout.extent[0] &&
      rows <= src->layout.extent[0] - input_row_offset &&
      output_row_offset <= dst->layout.extent[0] &&
      rows <= dst->layout.extent[0] - output_row_offset;
  const bool bias_valid = !bias_tensor ||
      (bias_tensor->layout.rank == 1 &&
       bias_tensor->layout.extent[0] == desc.out_features &&
       bias_tensor->layout.is_contiguous() && bias_tensor->type == bias_type);
  if (aliases || !row_ranges_valid || !bias_valid ||
      src->layout.extent[1] != desc.in_features ||
      weight->layout.rank != 2 ||
      weight->layout.extent[0] != desc.out_features ||
      weight->layout.extent[1] != desc.in_features ||
      (output_matrix && dst->layout.extent[1] != desc.out_features) ||
      !src->layout.is_contiguous() || !weight->layout.is_contiguous() ||
      !dst->layout.is_contiguous() || src->type != input_type ||
      weight->type != weight_type || dst->type != output_type) {
    throw std::invalid_argument("vulkan gemm: tensors do not match the plan");
  }
  detail::GemmDispatchGeometry geometry;
  if (!detail::gemm_dispatch_geometry(
          rows, desc.out_features, 16, 16, impl_->owner->max_dispatch_x,
          impl_->owner->max_dispatch_y, &geometry)) {
    throw std::out_of_range("vulkan gemm: dispatch exceeds device limits");
  }
  uint32_t groups_x = geometry.x;
  uint32_t groups_y = geometry.y;
  TensorContext::Impl::GemmParameters parameters;
  parameters.rows = rows;
  parameters.out_features = desc.out_features;
  parameters.in_features = desc.in_features;
  parameters.input_row_offset = input_row_offset;
  parameters.output_row_offset = output_row_offset;
  if (desc.mode == DenseGemmMode::kBFloat16) {
    parameters.mode = desc.bias == DenseGemmBias::kNone ? 0u
        : desc.bias == DenseGemmBias::kFloat32 ? 1u : 2u;
  } else if (desc.mode == DenseGemmMode::kFloat16Vae) {
    parameters.mode = 3u;
    parameters.input_row_offset = 0;
  } else {
    parameters.mode = desc.bias == DenseGemmBias::kNone ? 4u : 5u;
  }
  const bool cooperative_shape = (rows % 64u) == 0u &&
      (desc.out_features % 16u) == 0u && (desc.in_features % 16u) == 0u;
  const bool use_cooperative = !desc.force_scalar_order && cooperative_shape &&
      ((desc.mode == DenseGemmMode::kBFloat16 && impl_->owner->cooperative_gemm) ||
       (desc.mode == DenseGemmMode::kFloat16Vae &&
        impl_->owner->cooperative_f16_gemm));
  if (use_cooperative) {
    if (!detail::gemm_dispatch_geometry(
            rows, desc.out_features, 64, 16, impl_->owner->max_dispatch_x,
            impl_->owner->max_dispatch_y, &geometry)) {
      throw std::out_of_range("vulkan gemm: cooperative dispatch exceeds device limits");
    }
    groups_x = geometry.x;
    groups_y = geometry.y;
  }
  try {
    batch.impl_->count_operator();
    batch.impl_->transition(src, BufferAccess::kComputeRead);
    batch.impl_->transition(weight, BufferAccess::kComputeRead);
    if (bias_tensor) batch.impl_->transition(bias_tensor, BufferAccess::kComputeRead);
    batch.impl_->transition(dst, BufferAccess::kComputeWrite);
    auto& bindings = impl_->owner->gemm_bindings;
    bindings[0].buffer = &src->buffer;
    bindings[0].bytes = src->buffer.size();
    bindings[1].buffer = &weight->buffer;
    bindings[1].bytes = weight->buffer.size();
    bindings[2].buffer = bias_tensor ? &bias_tensor->buffer : &src->buffer;
    bindings[2].bytes = bias_tensor ? bias_tensor->buffer.size()
                                    : src->buffer.size();
    bindings[3].buffer = &dst->buffer;
    bindings[3].bytes = dst->buffer.size();
    batch.impl_->commands.bind_compute(
        use_cooperative
            ? (desc.mode == DenseGemmMode::kFloat16Vae
                   ? impl_->owner->gemm_coop_f16_pipeline
                   : impl_->owner->gemm_coop_pipeline)
            : impl_->owner->gemm_pipeline,
        bindings);
    batch.impl_->commands.push_constants(&parameters, sizeof(parameters));
    batch.impl_->commands.dispatch(groups_x, groups_y);
  } catch (...) {
    batch.impl_->poisoned = true;
    throw;
  }
}

BlockedAttentionPlan::BlockedAttentionPlan() = default;
BlockedAttentionPlan::~BlockedAttentionPlan() = default;
BlockedAttentionPlan::BlockedAttentionPlan(BlockedAttentionPlan&&) noexcept = default;
BlockedAttentionPlan& BlockedAttentionPlan::operator=(BlockedAttentionPlan&&) noexcept = default;
BlockedAttentionPlan::BlockedAttentionPlan(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
BlockedAttentionPlan::operator bool() const noexcept { return impl_ != nullptr; }

BlockedAttentionPlan BlockedAttentionPlan::create(
    TensorContext& context, const BlockedAttentionPlanDesc& desc) {
  if (!context.impl_) throw std::invalid_argument("vulkan attention: empty context");
  if (!context.impl_->exact_attention) {
    throw std::runtime_error(
        "vulkan attention: exact blocked attention is unavailable on this device/driver");
  }
  if (desc.sequence == 0 || desc.heads == 0 ||
      (desc.head_dim != 64 && desc.head_dim != 72 && desc.head_dim != 128) ||
      !is_exact_attention_scale(desc.head_dim, desc.scale)) {
    throw std::invalid_argument("vulkan attention: invalid exact blocked plan");
  }
  uint64_t elements = checked_multiply(desc.sequence, desc.heads, "attention");
  elements = checked_multiply(elements, desc.head_dim, "attention");
  if (elements > std::numeric_limits<uint32_t>::max() ||
      desc.sequence > context.impl_->max_dispatch_x ||
      desc.heads > context.impl_->max_dispatch_y ||
      checked_multiply(elements, 2, "attention") > context.impl_->max_storage_bytes) {
    throw std::out_of_range("vulkan attention: plan exceeds device/index limits");
  }
  auto result = std::make_shared<Impl>();
  result->owner = context.impl_;
  result->desc = desc;
  return BlockedAttentionPlan(std::move(result));
}

const BlockedAttentionPlanDesc& BlockedAttentionPlan::description() const {
  if (!impl_) throw std::logic_error("vulkan attention: empty plan");
  return impl_->desc;
}

PreparedAttentionInputs::PreparedAttentionInputs() = default;
PreparedAttentionInputs::~PreparedAttentionInputs() = default;
PreparedAttentionInputs::PreparedAttentionInputs(PreparedAttentionInputs&&) noexcept = default;
PreparedAttentionInputs& PreparedAttentionInputs::operator=(
    PreparedAttentionInputs&&) noexcept = default;
PreparedAttentionInputs::PreparedAttentionInputs(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
PreparedAttentionInputs::operator bool() const noexcept { return impl_ != nullptr; }
uint64_t PreparedAttentionInputs::reserved_bytes() const noexcept {
  return impl_ ? impl_->reserved_bytes : 0;
}

PreparedAttentionInputs PreparedAttentionInputs::create(
    TensorContext& context, const BlockedAttentionPlanDesc& desc) {
  // Reuse the plan's complete capability/shape validation before allocating.
  (void)BlockedAttentionPlan::create(context, desc);
  const uint64_t shape[] = {desc.sequence, desc.heads, desc.head_dim};
  DeviceTensor query = context.allocate(
      TensorLayout::contiguous(shape, 3), ScalarType::kFloat16);
  DeviceTensor key = context.allocate(
      TensorLayout::contiguous(shape, 3), ScalarType::kFloat16);
  DeviceTensor value = context.allocate(
      TensorLayout::contiguous(shape, 3), ScalarType::kFloat16);
  auto result = std::make_shared<Impl>();
  result->owner = context.impl_;
  result->desc = desc;
  result->query = std::move(query);
  result->key = std::move(key);
  result->value = std::move(value);
  result->reserved_bytes = checked_multiply(
      checked_multiply(checked_multiply(desc.sequence, desc.heads,
                                        "attention preparation"),
                       desc.head_dim, "attention preparation"),
      3 * sizeof(uint16_t), "attention preparation");
  return PreparedAttentionInputs(std::move(result));
}

PreparedAttentionView PreparedAttentionInputs::prepare(
    TensorBatch& batch, DeviceTensor& query, DeviceTensor& key,
    DeviceTensor& value) {
  if (!impl_ || !batch.impl_ || batch.impl_->poisoned) {
    throw std::logic_error("vulkan attention: empty/poisoned preparation");
  }
  if (batch.impl_->owner != impl_->owner) {
    throw std::invalid_argument("vulkan attention: preparation belongs to another context");
  }
  auto q = impl_->owner->require(query);
  auto k = impl_->owner->require(key);
  auto v = impl_->owner->require(value);
  auto q16 = impl_->owner->require(impl_->query);
  auto k16 = impl_->owner->require(impl_->key);
  auto v16 = impl_->owner->require(impl_->value);
  const auto& desc = impl_->desc;
  auto valid_input = [&](const std::shared_ptr<DeviceTensor::Impl>& tensor) {
    return tensor->type == ScalarType::kBFloat16 && tensor->layout.rank == 3 &&
        tensor->layout.is_contiguous() && tensor->layout.extent[0] == desc.sequence &&
        tensor->layout.extent[1] == desc.heads &&
        tensor->layout.extent[2] == desc.head_dim;
  };
  if (!valid_input(q) || !valid_input(k) || !valid_input(v) ||
      q.get() == k.get() || q.get() == v.get() || k.get() == v.get()) {
    throw std::invalid_argument("vulkan attention: invalid preparation inputs");
  }
  const uint64_t elements = checked_multiply(
      checked_multiply(desc.sequence, desc.heads, "attention prepare"),
      desc.head_dim, "attention prepare");
  const uint64_t word_count = (elements + 1) / 2;
  if (word_count == 0 || word_count > std::numeric_limits<uint32_t>::max()) {
    throw std::out_of_range("vulkan attention: preparation indexing overflow");
  }
  impl_->owner->validate_dispatch(word_count);
  if (impl_->generation == std::numeric_limits<uint64_t>::max()) {
    throw std::overflow_error("vulkan attention: preparation generation exhausted");
  }
  PreparedAttentionView result(
      impl_, batch.impl_->batch_id, impl_->generation + 1);
  try {
    batch.impl_->count_operator();
    batch.impl_->transition(q, BufferAccess::kComputeRead);
    batch.impl_->transition(k, BufferAccess::kComputeRead);
    batch.impl_->transition(v, BufferAccess::kComputeRead);
    batch.impl_->transition(q16, BufferAccess::kComputeWrite);
    batch.impl_->transition(k16, BufferAccess::kComputeWrite);
    batch.impl_->transition(v16, BufferAccess::kComputeWrite);
    std::array<std::shared_ptr<DeviceTensor::Impl>, 6> resources{
        q, k, v, q16, k16, v16};
    batch.impl_->dispatch_attention_prepare(
        static_cast<uint32_t>(word_count), resources);
  } catch (...) {
    batch.impl_->poisoned = true;
    throw;
  }
  impl_->batch_id = batch.impl_->batch_id;
  impl_->source_id = {q->identity, k->identity, v->identity};
  ++impl_->generation;
  return result;
}

PreparedAttentionView::PreparedAttentionView() = default;
PreparedAttentionView::~PreparedAttentionView() = default;
PreparedAttentionView::PreparedAttentionView(
    std::shared_ptr<void> slot, uintptr_t batch_id,
    uint64_t generation) noexcept
    : slot_(std::move(slot)), batch_id_(batch_id), generation_(generation) {}
PreparedAttentionView::PreparedAttentionView(PreparedAttentionView&&) noexcept = default;
PreparedAttentionView& PreparedAttentionView::operator=(
    PreparedAttentionView&&) noexcept = default;
PreparedAttentionView::operator bool() const noexcept { return slot_ != nullptr; }

void BlockedAttentionPlan::record(
    TensorBatch& batch, PreparedAttentionView& inputs,
    DeviceTensor& output, uint32_t query_row_offset,
    uint32_t rows, uint32_t output_row_offset) const {
  if (!impl_ || !batch.impl_ || batch.impl_->poisoned) {
    throw std::logic_error("vulkan attention: empty plan or batch");
  }
  if (batch.impl_->owner != impl_->owner) {
    throw std::invalid_argument("vulkan attention: plan belongs to another context");
  }
  auto prepared = std::static_pointer_cast<PreparedAttentionInputs::Impl>(inputs.slot_);
  if (!prepared || prepared->owner != impl_->owner ||
      inputs.batch_id_ != batch.impl_->batch_id ||
      inputs.batch_id_ != prepared->batch_id ||
      inputs.generation_ != prepared->generation ||
      prepared->desc.sequence != impl_->desc.sequence ||
      prepared->desc.heads != impl_->desc.heads ||
      prepared->desc.head_dim != impl_->desc.head_dim) {
    throw std::invalid_argument("vulkan attention: stale/incompatible prepared inputs");
  }
  auto q = impl_->owner->require(prepared->query);
  auto k = impl_->owner->require(prepared->key);
  auto v = impl_->owner->require(prepared->value);
  auto out = impl_->owner->require(output);
  const auto& desc = impl_->desc;
  if (query_row_offset > desc.sequence) {
    throw std::invalid_argument("vulkan attention: query row offset is out of range");
  }
  const uint32_t selected_rows = rows == 0 ? desc.sequence - query_row_offset : rows;
  const std::array<uint64_t, 4> expected{
      desc.sequence, desc.heads, desc.head_dim, 0};
  auto valid_layout = [&](const std::shared_ptr<DeviceTensor::Impl>& tensor) {
    return tensor->type == ScalarType::kFloat16 &&
        tensor->layout.rank == 3 && tensor->layout.is_contiguous() &&
        tensor->layout.extent[0] == expected[0] &&
        tensor->layout.extent[1] == expected[1] &&
        tensor->layout.extent[2] == expected[2];
  };
  const uint64_t query_end = static_cast<uint64_t>(query_row_offset) + selected_rows;
  const uint64_t output_end = static_cast<uint64_t>(output_row_offset) + selected_rows;
  if (selected_rows == 0 || query_end > desc.sequence || output_end > desc.sequence ||
      selected_rows > impl_->owner->max_dispatch_x ||
      desc.heads > impl_->owner->max_dispatch_y || !valid_layout(q) ||
      !valid_layout(k) || !valid_layout(v) ||
      out->type != ScalarType::kBFloat16 || out->layout.rank != 3 ||
      !out->layout.is_contiguous() || out->layout.extent[0] != expected[0] ||
      out->layout.extent[1] != expected[1] || out->layout.extent[2] != expected[2] ||
      q.get() == k.get() || q.get() == v.get() || k.get() == v.get() ||
      out->identity == prepared->source_id[0] ||
      out->identity == prepared->source_id[1] ||
      out->identity == prepared->source_id[2]) {
    throw std::invalid_argument("vulkan attention: invalid tensor/range/alias");
  }
  TensorContext::Impl::AttentionParameters parameters;
  parameters.sequence = desc.sequence;
  parameters.heads = desc.heads;
  parameters.head_dim = desc.head_dim;
  std::memcpy(&parameters.scale_bits, &desc.scale, sizeof(desc.scale));
  parameters.query_row_offset = query_row_offset;
  parameters.output_row_offset = output_row_offset;
  parameters.rows = selected_rows;
  try {
    batch.impl_->count_operator();
    batch.impl_->transition(q, BufferAccess::kComputeRead);
    batch.impl_->transition(k, BufferAccess::kComputeRead);
    batch.impl_->transition(v, BufferAccess::kComputeRead);
    batch.impl_->transition(out, BufferAccess::kComputeWrite);
    std::array<std::shared_ptr<DeviceTensor::Impl>, 4> resources{q, k, v, out};
    batch.impl_->dispatch_attention(parameters, resources);
  } catch (...) {
    batch.impl_->poisoned = true;
    throw;
  }
}

H3AttentionRanges::H3AttentionRanges() = default;
H3AttentionRanges::~H3AttentionRanges() = default;
H3AttentionRanges::H3AttentionRanges(H3AttentionRanges&&) noexcept = default;
H3AttentionRanges& H3AttentionRanges::operator=(H3AttentionRanges&&) noexcept = default;
H3AttentionRanges::H3AttentionRanges(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
H3AttentionRanges::operator bool() const noexcept { return impl_ != nullptr; }

H3AttentionRanges H3AttentionRanges::create(
    TensorContext& context, uint32_t sequence, const int32_t* values,
    uint32_t value_count) {
  if (!context.impl_) throw std::invalid_argument("vulkan H3 attention: empty context");
  if (!context.impl_->exact_h3_attention) {
    throw std::runtime_error(
        "vulkan H3 attention: exact mode is unavailable on this device/driver");
  }
  if (sequence == 0 || !values) {
    throw std::invalid_argument("vulkan H3 attention: invalid range table");
  }
  const uint64_t tiles64 = (static_cast<uint64_t>(sequence) + 127) / 128;
  const uint64_t count64 = checked_multiply(tiles64, 4, "H3 range table");
  const uint64_t aligned_end64 =
      ((static_cast<uint64_t>(sequence) + 63) / 64) * 64;
  if (tiles64 > std::numeric_limits<uint32_t>::max() ||
      count64 != value_count ||
      aligned_end64 > static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
    throw std::invalid_argument("vulkan H3 attention: invalid range table size");
  }
  const int32_t aligned_end = static_cast<int32_t>(aligned_end64);
  std::vector<int32_t> canonical(values, values + value_count);
  for (uint32_t tile = 0; tile < static_cast<uint32_t>(tiles64); ++tile) {
    int32_t& lo0 = canonical[static_cast<size_t>(tile) * 4];
    int32_t& hi0 = canonical[static_cast<size_t>(tile) * 4 + 1];
    int32_t& lo1 = canonical[static_cast<size_t>(tile) * 4 + 2];
    int32_t& hi1 = canonical[static_cast<size_t>(tile) * 4 + 3];
    auto valid_endpoint = [&](int32_t endpoint) {
      return endpoint >= 0 && endpoint <= aligned_end && endpoint % 64 == 0;
    };
    if (!valid_endpoint(lo0) || !valid_endpoint(hi0) ||
        !valid_endpoint(lo1) || !valid_endpoint(hi1) || lo0 >= hi0 ||
        lo0 >= static_cast<int32_t>(sequence) ||
        std::min<uint32_t>(static_cast<uint32_t>(hi0), sequence) <=
            static_cast<uint32_t>(lo0)) {
      throw std::invalid_argument("vulkan H3 attention: invalid primary range");
    }
    if (lo1 == 0 && hi1 == 0) continue;
    if (lo1 >= hi1 || lo1 < lo0 || lo1 >= static_cast<int32_t>(sequence) ||
        !valid_endpoint(lo1) || !valid_endpoint(hi1) ||
        std::min<uint32_t>(static_cast<uint32_t>(hi1), sequence) <=
            static_cast<uint32_t>(lo1)) {
      throw std::invalid_argument("vulkan H3 attention: invalid secondary range");
    }
    if (lo1 <= hi0) {
      hi0 = std::max(hi0, hi1);
      lo1 = 0;
      hi1 = 0;
    }
  }
  uint64_t hash = 1469598103934665603ull;
  for (int32_t value : canonical) {
    uint32_t word = 0;
    std::memcpy(&word, &value, sizeof(word));
    for (uint32_t byte = 0; byte < 4; ++byte) {
      hash ^= (word >> (byte * 8)) & 0xffu;
      hash *= 1099511628211ull;
    }
  }
  const uint64_t shape[] = {tiles64, 4};
  DeviceTensor tensor = context.allocate(
      TensorLayout::contiguous(shape, 2), ScalarType::kInt32);
  context.upload_bytes(tensor, canonical.data(), count64 * sizeof(int32_t));
  auto result = std::make_shared<Impl>();
  result->owner = context.impl_;
  result->tensor = std::move(tensor);
  result->sequence = sequence;
  result->query_tiles = static_cast<uint32_t>(tiles64);
  result->content_hash = hash;
  return H3AttentionRanges(std::move(result));
}

uint32_t H3AttentionRanges::sequence() const {
  if (!impl_) throw std::logic_error("vulkan H3 attention: empty range table");
  return impl_->sequence;
}
uint32_t H3AttentionRanges::query_tiles() const {
  if (!impl_) throw std::logic_error("vulkan H3 attention: empty range table");
  return impl_->query_tiles;
}
uint64_t H3AttentionRanges::content_hash() const {
  if (!impl_) throw std::logic_error("vulkan H3 attention: empty range table");
  return impl_->content_hash;
}

H3AttentionPlan::H3AttentionPlan() = default;
H3AttentionPlan::~H3AttentionPlan() = default;
H3AttentionPlan::H3AttentionPlan(H3AttentionPlan&&) noexcept = default;
H3AttentionPlan& H3AttentionPlan::operator=(H3AttentionPlan&&) noexcept = default;
H3AttentionPlan::H3AttentionPlan(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
H3AttentionPlan::operator bool() const noexcept { return impl_ != nullptr; }

H3AttentionPlan H3AttentionPlan::create(
    TensorContext& context, const H3AttentionPlanDesc& desc) {
  if (!context.impl_) throw std::invalid_argument("vulkan H3 attention: empty context");
  if (!context.impl_->exact_h3_attention) {
    throw std::runtime_error(
        "vulkan H3 attention: exact mode is unavailable on this device/driver");
  }
  if (desc.sequence == 0 || desc.heads == 0 ||
      (desc.head_dim != 64 && desc.head_dim != 128) ||
      !is_exact_attention_scale(desc.head_dim, desc.scale)) {
    throw std::invalid_argument("vulkan H3 attention: invalid exact plan");
  }
  uint64_t elements = checked_multiply(desc.sequence, desc.heads, "H3 attention");
  elements = checked_multiply(elements, desc.head_dim, "H3 attention");
  if (elements > std::numeric_limits<uint32_t>::max() ||
      desc.sequence > context.impl_->max_dispatch_x ||
      desc.heads > context.impl_->max_dispatch_y ||
      checked_multiply(elements, 2, "H3 attention") > context.impl_->max_storage_bytes) {
    throw std::out_of_range("vulkan H3 attention: plan exceeds device/index limits");
  }
  auto result = std::make_shared<Impl>();
  result->owner = context.impl_;
  result->desc = desc;
  return H3AttentionPlan(std::move(result));
}

const H3AttentionPlanDesc& H3AttentionPlan::description() const {
  if (!impl_) throw std::logic_error("vulkan H3 attention: empty plan");
  return impl_->desc;
}

void H3AttentionPlan::record(
    TensorBatch& batch, DeviceTensor& query, DeviceTensor& key,
    DeviceTensor& value, DeviceTensor& output, const H3AttentionRanges* ranges,
    uint32_t query_row_offset, uint32_t rows,
    uint32_t output_row_offset) const {
  if (!impl_ || !batch.impl_ || batch.impl_->poisoned) {
    throw std::logic_error("vulkan H3 attention: empty plan or batch");
  }
  if (batch.impl_->owner != impl_->owner) {
    throw std::invalid_argument("vulkan H3 attention: plan belongs to another context");
  }
  auto q = impl_->owner->require(query);
  auto k = impl_->owner->require(key);
  auto v = impl_->owner->require(value);
  auto out = impl_->owner->require(output);
  std::shared_ptr<DeviceTensor::Impl> range_tensor;
  if (ranges) {
    if (!ranges->impl_ || ranges->impl_->owner != impl_->owner ||
        ranges->impl_->sequence != impl_->desc.sequence ||
        ranges->impl_->query_tiles !=
            (static_cast<uint64_t>(impl_->desc.sequence) + 127) / 128) {
      throw std::invalid_argument("vulkan H3 attention: incompatible range table");
    }
    range_tensor = impl_->owner->require(ranges->impl_->tensor);
  }
  const auto& desc = impl_->desc;
  const uint32_t selected_rows = rows == 0 && query_row_offset <= desc.sequence
      ? desc.sequence - query_row_offset : rows;
  auto valid_layout = [&](const std::shared_ptr<DeviceTensor::Impl>& tensor) {
    const bool shaped = tensor->layout.rank == 3 &&
        tensor->layout.extent[1] == desc.heads &&
        tensor->layout.extent[2] == desc.head_dim;
    const bool flat = tensor->layout.rank == 2 &&
        tensor->layout.extent[1] == static_cast<uint64_t>(desc.heads) * desc.head_dim;
    return tensor->type == ScalarType::kBFloat16 && (shaped || flat) &&
        tensor->layout.is_contiguous() && tensor->layout.extent[0] == desc.sequence;
  };
  const uint64_t query_end = static_cast<uint64_t>(query_row_offset) + selected_rows;
  const uint64_t output_end = static_cast<uint64_t>(output_row_offset) + selected_rows;
  if (query_row_offset > desc.sequence || selected_rows == 0 ||
      query_end > desc.sequence || output_end > desc.sequence ||
      selected_rows > impl_->owner->max_dispatch_x ||
      desc.heads > impl_->owner->max_dispatch_y || !valid_layout(q) ||
      !valid_layout(k) || !valid_layout(v) || !valid_layout(out) ||
      q.get() == k.get() || q.get() == v.get() || k.get() == v.get() ||
      out.get() == q.get() || out.get() == k.get() || out.get() == v.get()) {
    throw std::invalid_argument("vulkan H3 attention: invalid tensor/range/alias");
  }
  TensorContext::Impl::AttentionParameters parameters;
  parameters.sequence = desc.sequence;
  parameters.heads = desc.heads;
  parameters.head_dim = desc.head_dim;
  std::memcpy(&parameters.scale_bits, &desc.scale, sizeof(desc.scale));
  parameters.query_row_offset = query_row_offset;
  parameters.output_row_offset = output_row_offset;
  parameters.rows = selected_rows;
  try {
    batch.impl_->count_operator();
    batch.impl_->transition(q, BufferAccess::kComputeRead);
    batch.impl_->transition(k, BufferAccess::kComputeRead);
    batch.impl_->transition(v, BufferAccess::kComputeRead);
    batch.impl_->transition(out, BufferAccess::kComputeWrite);
    if (range_tensor) {
      batch.impl_->transition(range_tensor, BufferAccess::kComputeRead);
      std::array<std::shared_ptr<DeviceTensor::Impl>, 5> resources{
          q, k, v, out, range_tensor};
      batch.impl_->dispatch_h3_banded_attention(parameters, resources);
    } else {
      std::array<std::shared_ptr<DeviceTensor::Impl>, 4> resources{q, k, v, out};
      batch.impl_->dispatch_h3_attention(parameters, resources);
    }
  } catch (...) {
    batch.impl_->poisoned = true;
    throw;
  }
}

CausalGQAAttentionPlan::CausalGQAAttentionPlan() = default;
CausalGQAAttentionPlan::~CausalGQAAttentionPlan() = default;
CausalGQAAttentionPlan::CausalGQAAttentionPlan(
    CausalGQAAttentionPlan&&) noexcept = default;
CausalGQAAttentionPlan& CausalGQAAttentionPlan::operator=(
    CausalGQAAttentionPlan&&) noexcept = default;
CausalGQAAttentionPlan::CausalGQAAttentionPlan(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
CausalGQAAttentionPlan::operator bool() const noexcept {
  return impl_ != nullptr;
}

CausalGQAAttentionPlan CausalGQAAttentionPlan::create(
    TensorContext& context, const CausalGQAAttentionPlanDesc& desc) {
  if (!context.impl_) {
    throw std::invalid_argument("vulkan causal GQA attention: empty context");
  }
  if (!context.impl_->exact_causal_gqa_attention) {
    throw std::runtime_error(
        "vulkan causal GQA attention: exact mode is unavailable on this device/driver");
  }
  // This plan names the shipped Qwen3-VL text contract rather than advertising
  // an unverified generic GQA family.
  if (desc.sequence == 0 || desc.sequence > 8192 ||
      desc.query_heads != 64 || desc.kv_heads != 8 || desc.head_dim != 128 ||
      !is_exact_attention_scale(desc.head_dim, desc.scale)) {
    throw std::invalid_argument(
        "vulkan causal GQA attention: invalid Qwen text plan");
  }
  uint64_t query_elements =
      checked_multiply(desc.sequence, desc.query_heads, "causal GQA attention");
  query_elements = checked_multiply(query_elements, desc.head_dim,
                                    "causal GQA attention");
  uint64_t kv_elements =
      checked_multiply(desc.sequence, desc.kv_heads, "causal GQA attention");
  kv_elements = checked_multiply(kv_elements, desc.head_dim,
                                 "causal GQA attention");
  if (query_elements > std::numeric_limits<uint32_t>::max() ||
      kv_elements > std::numeric_limits<uint32_t>::max() ||
      desc.sequence > context.impl_->max_dispatch_x ||
      desc.query_heads > context.impl_->max_dispatch_y ||
      checked_multiply(query_elements, sizeof(uint16_t),
                       "causal GQA attention") > context.impl_->max_storage_bytes ||
      checked_multiply(kv_elements, sizeof(uint16_t),
                       "causal GQA attention") > context.impl_->max_storage_bytes) {
    throw std::out_of_range(
        "vulkan causal GQA attention: plan exceeds device/index limits");
  }
  auto result = std::make_shared<Impl>();
  result->owner = context.impl_;
  result->desc = desc;
  return CausalGQAAttentionPlan(std::move(result));
}

const CausalGQAAttentionPlanDesc& CausalGQAAttentionPlan::description() const {
  if (!impl_) throw std::logic_error("vulkan causal GQA attention: empty plan");
  return impl_->desc;
}

void CausalGQAAttentionPlan::record(
    TensorBatch& batch, DeviceTensor& query, DeviceTensor& key,
    DeviceTensor& value, DeviceTensor& output, uint32_t query_row_offset,
    uint32_t rows, uint32_t output_row_offset) const {
  if (!impl_ || !batch.impl_ || batch.impl_->poisoned) {
    throw std::logic_error("vulkan causal GQA attention: empty plan or batch");
  }
  if (batch.impl_->owner != impl_->owner) {
    throw std::invalid_argument(
        "vulkan causal GQA attention: plan belongs to another context");
  }
  auto q = impl_->owner->require(query);
  auto k = impl_->owner->require(key);
  auto v = impl_->owner->require(value);
  auto out = impl_->owner->require(output);
  const auto& desc = impl_->desc;
  if (query_row_offset > desc.sequence) {
    throw std::invalid_argument(
        "vulkan causal GQA attention: query row offset is out of range");
  }
  const uint32_t selected_rows =
      rows == 0 ? desc.sequence - query_row_offset : rows;
  const uint64_t query_end =
      static_cast<uint64_t>(query_row_offset) + selected_rows;
  const uint64_t output_end =
      static_cast<uint64_t>(output_row_offset) + selected_rows;
  auto valid_layout = [&](const std::shared_ptr<DeviceTensor::Impl>& tensor,
                          uint32_t heads) {
    return tensor->type == ScalarType::kBFloat16 &&
        tensor->layout.rank == 3 && tensor->layout.is_contiguous() &&
        tensor->layout.extent[0] == desc.sequence &&
        tensor->layout.extent[1] == heads &&
        tensor->layout.extent[2] == desc.head_dim;
  };
  if (selected_rows == 0 || query_end > desc.sequence ||
      output_end > desc.sequence || selected_rows > impl_->owner->max_dispatch_x ||
      desc.query_heads > impl_->owner->max_dispatch_y ||
      !valid_layout(q, desc.query_heads) || !valid_layout(k, desc.kv_heads) ||
      !valid_layout(v, desc.kv_heads) || !valid_layout(out, desc.query_heads) ||
      q->identity == k->identity || q->identity == v->identity ||
      q->identity == out->identity || k->identity == v->identity ||
      k->identity == out->identity || v->identity == out->identity) {
    throw std::invalid_argument(
        "vulkan causal GQA attention: invalid tensor/range/alias");
  }
  TensorContext::Impl::CausalGQAAttentionParameters parameters;
  parameters.sequence = desc.sequence;
  parameters.query_heads = desc.query_heads;
  parameters.kv_heads = desc.kv_heads;
  parameters.head_dim = desc.head_dim;
  std::memcpy(&parameters.scale_bits, &desc.scale, sizeof(desc.scale));
  parameters.query_row_offset = query_row_offset;
  parameters.output_row_offset = output_row_offset;
  parameters.rows = selected_rows;
  try {
    batch.impl_->count_operator();
    batch.impl_->transition(q, BufferAccess::kComputeRead);
    batch.impl_->transition(k, BufferAccess::kComputeRead);
    batch.impl_->transition(v, BufferAccess::kComputeRead);
    batch.impl_->transition(out, BufferAccess::kComputeWrite);
    std::array<std::shared_ptr<DeviceTensor::Impl>, 4> resources{q, k, v, out};
    batch.impl_->dispatch_causal_gqa_attention(parameters, resources);
  } catch (...) {
    batch.impl_->poisoned = true;
    throw;
  }
}

Submission TensorBatch::submit() {
  if (!impl_ || impl_->operator_count == 0) {
    throw std::logic_error("vulkan tensor: cannot submit an empty batch");
  }
  if (impl_->poisoned) {
    throw std::logic_error("vulkan tensor: cannot submit a poisoned batch");
  }
  Submission result = impl_->owner->commands.submit(std::move(impl_->commands));
  impl_->submitted = true;
  impl_->recording_lease.release();
  impl_.reset();
  return result;
}

}  // namespace vidfab::vulkan
