#include "tensor_context.h"
#include "embedded_tensor_spv.h"

namespace slopfab::vulkan {
TensorContext::Impl::Impl(const Device& input, const TensorContextOptions& tensor_options)
      : max_batch_operators(tensor_options.max_batch_operators),
        commands(input, [&] {
          if (tensor_options.max_batch_operators == 0 ||
              tensor_options.max_batch_operators > 4096) {
            throw std::invalid_argument(
                "vulkan tensor: max_batch_operators must be in [1,4096]");
          }
          ComputeContextOptions options;
          options.max_in_flight = tensor_options.max_in_flight;
          options.max_storage_bindings = 8;
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
        keyframe_bindings(4),
        dit_bindings(5),
        weight_bindings(6),
        gemm_bindings(4),
        gemm_prepare_bindings(2),
        attention_bindings(4),
        attention_h3_bindings(4),
        attention_h3_banded_bindings(5),
        attention_sage_bindings(6),
        attention_sage_prepare_bindings(7),
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
    flash_attention = fast_h3_attention_device(input.info()) &&
        input.info().max_compute_shared_memory_bytes >= 41344 &&
        input.info().shader_bfloat16_type &&
        input.info().shader_bfloat16_cooperative_matrix &&
        input.info().cooperative_matrix_bf16_f32_16x16x16;
    sage_device_info = input.info();
    sage_extra_workspace_bytes = tensor_options.sage_extra_workspace_bytes;
    sage_attention = input.info().cooperative_matrix_enabled &&
        input.info().shader_float16_enabled && input.info().storage_buffer_16bit_enabled &&
        input.info().compute_subgroup_shuffle &&
        input.info().cooperative_matrix_f16_f32_16x16x16 &&
        detail::sage_kernel_fits(input.info(), 1) &&
        input.info().compute_subgroup_arithmetic && input.info().shader_int8_enabled &&
        input.info().cooperative_matrix_i8_i32_16x16x32;
    exact_causal_gqa_attention =
        known_exact_causal_gqa_attention_device(input.info());
    max_dispatch_x = input.info().max_compute_workgroup_count[0];
    max_dispatch_y = input.info().max_compute_workgroup_count[1];
    max_storage_bytes = input.info().max_storage_buffer_bytes;
    storage_binding_alignment = input.info().min_storage_buffer_offset_alignment;
    device_identity = input.native_handle();
    prepare_pipelines(input, tensor_options.pipeline_sets |
        (tensor_options.enable_reference_encoder ? TensorPipelineSet::kReference
                                                : TensorPipelineSet::kCore));
    for (uint32_t i = 0; i < ops_bindings.size(); ++i) ops_bindings[i].binding = i;
    for (uint32_t i = 0; i < vae_pointwise_bindings.size(); ++i)
      vae_pointwise_bindings[i].binding = i;
    for (uint32_t i = 0; i < norm_bindings.size(); ++i) norm_bindings[i].binding = i;
    for (uint32_t i = 0; i < mod_bindings.size(); ++i) mod_bindings[i].binding = i;
    for (uint32_t i = 0; i < vae_rope_bindings.size(); ++i)
      vae_rope_bindings[i].binding = i;
    for (uint32_t i = 0; i < audio_bindings.size(); ++i)
      audio_bindings[i].binding = i;
    for (uint32_t i = 0; i < keyframe_bindings.size(); ++i)
      keyframe_bindings[i].binding = i;
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
    for (uint32_t i = 0; i < attention_sage_bindings.size(); ++i)
      attention_sage_bindings[i].binding = i;
    for (uint32_t i = 0; i < attention_sage_prepare_bindings.size(); ++i)
      attention_sage_prepare_bindings[i].binding = i;
    for (uint32_t i = 0; i < attention_prepare_bindings.size(); ++i)
      attention_prepare_bindings[i].binding = i;
    for (uint32_t i = 0; i < attention_causal_gqa_bindings.size(); ++i)
      attention_causal_gqa_bindings[i].binding = i;
  }

void TensorContext::Impl::prepare_pipelines(const Device& input, TensorPipelineSet sets) {
  const uint32_t requested = static_cast<uint32_t>(sets) |
      static_cast<uint32_t>(TensorPipelineSet::kCore);
  if (requested & ~static_cast<uint32_t>(TensorPipelineSet::kAll))
    throw std::invalid_argument("vulkan tensor: unknown pipeline set");
  const uint32_t missing = requested & ~prepared_sets;
  if (!missing) return;
  auto has = [&](TensorPipelineSet set) {
    return (missing & static_cast<uint32_t>(set)) != 0;
  };
    ComputePipelineOptions options;
    options.storage_binding_count = 3;
    options.push_constant_bytes = sizeof(Parameters);
    options.local_size[0] = 64;
    if (has(TensorPipelineSet::kCore)) {
    const uint8_t* shader = full_arithmetic_exact ? detail::kTensorOpsDenormSpirv
                                           : detail::kTensorOpsSpirv;
    const size_t shader_bytes = full_arithmetic_exact ? sizeof(detail::kTensorOpsDenormSpirv)
                                               : sizeof(detail::kTensorOpsSpirv);
    std::vector<uint32_t> spirv(shader_bytes / sizeof(uint32_t));
    std::memcpy(spirv.data(), shader, shader_bytes);
    ops_pipeline = ComputePipeline::create(input, spirv, options);
    }
    if (has(TensorPipelineSet::kVideo) && exact_vae_pointwise) {
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
    if (has(TensorPipelineSet::kReference)) {
      if (input.info().max_compute_workgroup_invocations < 256 ||
          input.info().max_compute_workgroup_size[0] < 16 ||
          input.info().max_compute_workgroup_size[1] < 16)
        throw std::runtime_error("Vulkan reference: device requires 16x16 compute workgroups");
      std::vector<uint32_t> module(sizeof(detail::kTensorReferenceSpirv) / 4);
      std::memcpy(module.data(), detail::kTensorReferenceSpirv, sizeof(detail::kTensorReferenceSpirv));
      ComputePipelineOptions reference_options;
      reference_options.storage_binding_count = 6;
      reference_options.push_constant_bytes = 64;
      reference_options.local_size[0] = reference_options.local_size[1] = 16;
      reference_pipeline = ComputePipeline::create(input, module, reference_options);
    }
    if (has(TensorPipelineSet::kAudio) && exact_audio) {
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
    if (has(TensorPipelineSet::kVideo) && exact_vae_pointwise) {
      std::vector<uint32_t> keyframe_spirv(
          sizeof(detail::kTensorKeyframeSpirv) / sizeof(uint32_t));
      std::memcpy(keyframe_spirv.data(), detail::kTensorKeyframeSpirv,
                  sizeof(detail::kTensorKeyframeSpirv));
      ComputePipelineOptions keyframe_options;
      keyframe_options.storage_binding_count = 4;
      keyframe_options.push_constant_bytes = sizeof(KeyframeParameters);
      keyframe_options.local_size[0] = 64;
      keyframe_pipeline =
          ComputePipeline::create(input, keyframe_spirv, keyframe_options);
    }
    if (has(TensorPipelineSet::kDit) && exact_dit_pointwise) {
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
    if (has(TensorPipelineSet::kCore)) {
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
    }
    ComputePipelineOptions norm_options;
    norm_options.storage_binding_count = 4;
    norm_options.push_constant_bytes = sizeof(NormParameters);
    norm_options.local_size[0] = 256;
    auto make_norm_pipeline = [&](const uint8_t* shader, size_t shader_bytes,
                                  uint32_t bindings = 4, uint32_t local_x = 256,
                                  uint32_t local_y = 1,
                                  uint32_t push_bytes = sizeof(NormParameters),
                                  std::vector<SpecializationConstant> constants = {}) {
      std::vector<uint32_t> module(shader_bytes / sizeof(uint32_t));
      std::memcpy(module.data(), shader, shader_bytes);
      ComputePipelineOptions selected = norm_options;
      selected.storage_binding_count = bindings;
      selected.local_size[0] = local_x;
      selected.local_size[1] = local_y;
      selected.push_constant_bytes = push_bytes;
      selected.specialization_constants = std::move(constants);
      return ComputePipeline::create(input, module, selected);
    };
    if (has(TensorPipelineSet::kCore) && exact_vae_norm) {
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
    if (has(TensorPipelineSet::kBlockedAttention) && exact_attention) {
      attention_blocked_pipeline = make_norm_pipeline(
          detail::kTensorAttentionBlockedSpirv,
          sizeof(detail::kTensorAttentionBlockedSpirv), 4, 128, 1,
          sizeof(AttentionParameters));
      attention_prepare_pipeline = make_norm_pipeline(
          detail::kTensorAttentionPrepareSpirv,
          sizeof(detail::kTensorAttentionPrepareSpirv), 6, 64, 1,
          sizeof(uint32_t));
    }
    if (has(TensorPipelineSet::kExactH3Attention) && exact_h3_attention) {
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
    if (has(TensorPipelineSet::kFlashAttention) && flash_attention) {
      attention_vsa_pipeline = make_norm_pipeline(detail::kTensorAttention_VSA,
          sizeof(detail::kTensorAttention_VSA), 6, 256, 1, 32);
      attention_vsa_prepare_pipeline = make_norm_pipeline(detail::kTensorAttention_VSA_PREPARE,
          sizeof(detail::kTensorAttention_VSA_PREPARE), 8, 256, 1, 32);
      attention_flash_pipeline = make_norm_pipeline(detail::kTensorAttentionFlashSpirv,
          sizeof(detail::kTensorAttentionFlashSpirv), 4, kFastH3AttentionLocalSize, 1, sizeof(AttentionParameters));
      attention_flash_banded_pipeline = make_norm_pipeline(detail::kTensorAttentionFlashBandedSpirv,
          sizeof(detail::kTensorAttentionFlashBandedSpirv), 5, kFastH3AttentionLocalSize, 1, sizeof(AttentionParameters));
    }
    if (has(TensorPipelineSet::kSageAttention) && sage_attention) {
      auto make_sage = [&](uint32_t kernel, const uint8_t* full, size_t full_bytes,
                           const uint8_t* banded, size_t banded_bytes) {
        if (!detail::sage_kernel_fits(input.info(), kernel)) return;
        const auto c = detail::sage_kernel_configuration(kernel, input.info().subgroup_size);
        for (uint32_t variant = 0; variant < 4; ++variant) {
          const std::vector<SpecializationConstant> constants = {{0, variant & 1u}, {1, variant < 2 ? 64u : 128u}};
          attention_sage_pipelines[kernel-1][variant] = make_norm_pipeline(full, full_bytes, 6, c.local_size, 1, sizeof(AttentionParameters), constants);
          attention_sage_banded_pipelines[kernel-1][variant] = make_norm_pipeline(banded, banded_bytes, 6, c.local_size, 1, sizeof(AttentionParameters), constants);
        }
      };
#define SAGE_PIPELINE(K, F, B) make_sage(K, detail::F, sizeof(detail::F), detail::B, sizeof(detail::B))
      if (input.info().subgroup_size == 32) {
        SAGE_PIPELINE(1, kTensorAttentionSageSpirv, kTensorAttentionSageBandedSpirv);
        SAGE_PIPELINE(2, kTensorAttention_SAGE_FULL, kTensorAttention_SAGE_FULL_BANDED);
        SAGE_PIPELINE(3, kTensorAttention_SAGE_WIDE, kTensorAttention_SAGE_WIDE_BANDED);
        SAGE_PIPELINE(4, kTensorAttention_SAGE_WIDE_FULL, kTensorAttention_SAGE_WIDE_FULL_BANDED);
      } else {
        SAGE_PIPELINE(1, kTensorAttention_SAGE_SG64, kTensorAttention_SAGE_SG64_BANDED);
        SAGE_PIPELINE(2, kTensorAttention_SAGE_FULL_SG64, kTensorAttention_SAGE_FULL_SG64_BANDED);
        SAGE_PIPELINE(3, kTensorAttention_SAGE_WIDE_SG64, kTensorAttention_SAGE_WIDE_SG64_BANDED);
        SAGE_PIPELINE(4, kTensorAttention_SAGE_WIDE_FULL_SG64, kTensorAttention_SAGE_WIDE_FULL_SG64_BANDED);
      }
#undef SAGE_PIPELINE
      attention_sage_prepare_pipeline = make_norm_pipeline(detail::kTensorAttentionSagePrepareSpirv,
          sizeof(detail::kTensorAttentionSagePrepareSpirv), 7, 128, 1, sizeof(AttentionParameters));
    }
    if (has(TensorPipelineSet::kTextAttention) && exact_causal_gqa_attention) {
      attention_causal_gqa_pipeline = make_norm_pipeline(
          detail::kTensorAttentionCausalGqaSpirv,
          sizeof(detail::kTensorAttentionCausalGqaSpirv), 4, 128, 1,
          sizeof(CausalGQAAttentionParameters));
    }
  prepared_sets |= requested;
}

void TensorContext::prepare_pipeline_sets(const Device& device, TensorPipelineSet sets) {
  if (!impl_ || !device || impl_->device_identity != device.native_handle())
    throw std::invalid_argument("vulkan tensor: pipeline preparation requires the context device");
  [[maybe_unused]] auto recording_lock = impl_->acquire_recorder();
  impl_->prepare_pipelines(device, sets);
}

TensorPipelineSet TensorContext::prepared_pipeline_sets() const noexcept {
  return static_cast<TensorPipelineSet>(impl_ ? impl_->prepared_sets : 0);
}
}  // namespace slopfab::vulkan
