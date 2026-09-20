#include "tensor_recording.h"

namespace slopfab::vulkan {

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
    if (physical != item.logical_bytes)
      write_zero_bytes(buffer, item.logical_bytes,
                       physical - item.logical_bytes);
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
  const bool matrix = shape.rank == 2 &&
      shape.extent[1] == weight.impl_->in_features;
  const bool heads = shape.rank == 3 &&
      shape.extent[1] <= std::numeric_limits<uint64_t>::max() /
          shape.extent[2] &&
      shape.extent[1] * shape.extent[2] == weight.impl_->in_features;
  if (source.get() == destination.get() || (!matrix && !heads) ||
      destination->layout.rank != shape.rank ||
      destination->layout.extent != shape.extent ||
      !shape.is_contiguous() || !destination->layout.is_contiguous() ||
      source->type != destination->type ||
      (source->type != ScalarType::kBFloat16 &&
       source->type != ScalarType::kFloat32) ||
      elements > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(
        "vulkan linear weight: transform needs distinct contiguous row tensors");
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
  // Validate the entire materialization transaction before changing the
  // shared dense tensor's logical shape or cache generation. In particular,
  // require every format-specific resource through this cache's owner now;
  // materialize_bf16() must not discover a foreign weight after an older
  // prepared view has already been made stale.
  std::array<std::shared_ptr<DeviceTensor::Impl>, 5> sources{};
  sources[0] = impl_->owner->require(weight.impl_->data);
  switch (weight.impl_->format) {
    case LinearWeightFormat::kFloat32:
    case LinearWeightFormat::kFloat16:
    case LinearWeightFormat::kBFloat16:
      break;
    case LinearWeightFormat::kFloat8E4M3:
    case LinearWeightFormat::kInt8:
      sources[1] = impl_->owner->require(weight.impl_->weight_scale);
      break;
    case LinearWeightFormat::kNVFloat4:
      sources[1] = impl_->owner->require(weight.impl_->block_scale);
      break;
    case LinearWeightFormat::kNF4:
      sources[1] = impl_->owner->require(weight.impl_->nf4_absmax);
      sources[2] = impl_->owner->require(weight.impl_->nf4_quant_map);
      sources[3] = impl_->owner->require(weight.impl_->nf4_nested_quant_map);
      sources[4] = impl_->owner->require(weight.impl_->nf4_nested_absmax);
      break;
    default:
      throw std::invalid_argument(
          "vulkan linear stream: unsupported weight format");
  }
  const uint64_t elements = checked_multiply(
      weight.impl_->out_features, weight.impl_->in_features,
      "nvfp4 streamed weight");
  if (desc.mode != DenseGemmMode::kBFloat16 ||
      desc.out_features != weight.impl_->out_features ||
      desc.in_features != weight.impl_->in_features ||
      elements > impl_->capacity_elements ||
      elements > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(
        "vulkan linear stream: weight does not match the BF16 plan/cache");
  }
  if (batch.impl_->operator_count == batch.impl_->owner->max_batch_operators) {
    throw std::logic_error("vulkan tensor: batch operator limit exceeded");
  }
  if (impl_->generation == std::numeric_limits<uint64_t>::max()) {
    throw std::overflow_error("vulkan nvfp4 stream: generation exhausted");
  }
  impl_->owner->validate_dispatch(1 + (elements - 1) / 2);
  auto dense = impl_->owner->require(impl_->dense);
  for (const auto& source : sources) {
    if (source && source.get() == dense.get()) {
      throw std::invalid_argument(
          "vulkan linear stream: cache aliases persistent weight storage");
    }
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


}  // namespace slopfab::vulkan
