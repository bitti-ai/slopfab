#include "tensor_recording.h"

namespace slopfab::vulkan {

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
  const bool input_matrix = src->layout.rank == 2;
  const bool input_heads = src->layout.rank == 3 &&
      src->layout.extent[1] <= std::numeric_limits<uint64_t>::max() /
                                   src->layout.extent[2] &&
      src->layout.extent[1] * src->layout.extent[2] == desc.in_features;
  const bool output_matrix = dst->layout.rank == 2;
  const bool output_heads = dst->layout.rank == 3 &&
      dst->layout.extent[1] <= std::numeric_limits<uint64_t>::max() /
                                   dst->layout.extent[2] &&
      dst->layout.extent[1] * dst->layout.extent[2] == desc.out_features;
  const bool row_ranges_valid =
      (input_matrix || input_heads) && (output_matrix || output_heads) &&
      input_row_offset <= src->layout.extent[0] &&
      rows <= src->layout.extent[0] - input_row_offset &&
      output_row_offset <= dst->layout.extent[0] &&
      rows <= dst->layout.extent[0] - output_row_offset;
  const bool bias_valid = !bias_tensor ||
      (bias_tensor->layout.rank == 1 &&
       bias_tensor->layout.extent[0] == desc.out_features &&
       bias_tensor->layout.is_contiguous() && bias_tensor->type == bias_type);
  if (aliases || !row_ranges_valid || !bias_valid ||
      (input_matrix && src->layout.extent[1] != desc.in_features) ||
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


}  // namespace slopfab::vulkan
