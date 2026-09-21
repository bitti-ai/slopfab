#include "tensor_recording.h"
#include "slopfab/attention_descriptor.h"

namespace slopfab::vulkan {

BlockedAttentionPlan::BlockedAttentionPlan() = default;
BlockedAttentionPlan::~BlockedAttentionPlan() = default;
BlockedAttentionPlan::BlockedAttentionPlan(BlockedAttentionPlan&&) noexcept = default;
BlockedAttentionPlan& BlockedAttentionPlan::operator=(BlockedAttentionPlan&&) noexcept = default;

BlockedAttentionPlan::BlockedAttentionPlan(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {
}

BlockedAttentionPlan::operator bool() const noexcept {
  return impl_ != nullptr;
}

BlockedAttentionPlan BlockedAttentionPlan::create(TensorContext& context,
                                                  const BlockedAttentionPlanDesc& desc) {
  if (!context.impl_)
    throw std::invalid_argument("vulkan attention: empty context");
  context.impl_->require_pipeline_set(TensorPipelineSet::kBlockedAttention);
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
      desc.sequence > context.impl_->max_dispatch_x || desc.heads > context.impl_->max_dispatch_y ||
      checked_multiply(elements, 2, "attention") > context.impl_->max_storage_bytes) {
    throw std::out_of_range("vulkan attention: plan exceeds device/index limits");
  }
  auto result = std::make_shared<Impl>();
  result->owner = context.impl_;
  result->desc = desc;
  return BlockedAttentionPlan(std::move(result));
}

const BlockedAttentionPlanDesc& BlockedAttentionPlan::description() const {
  if (!impl_)
    throw std::logic_error("vulkan attention: empty plan");
  return impl_->desc;
}

PreparedAttentionInputs::PreparedAttentionInputs() = default;
PreparedAttentionInputs::~PreparedAttentionInputs() = default;
PreparedAttentionInputs::PreparedAttentionInputs(PreparedAttentionInputs&&) noexcept = default;
PreparedAttentionInputs&
PreparedAttentionInputs::operator=(PreparedAttentionInputs&&) noexcept = default;

PreparedAttentionInputs::PreparedAttentionInputs(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {
}

PreparedAttentionInputs::operator bool() const noexcept {
  return impl_ != nullptr;
}

uint64_t PreparedAttentionInputs::reserved_bytes() const noexcept {
  return impl_ ? impl_->reserved_bytes : 0;
}

PreparedAttentionInputs PreparedAttentionInputs::create(TensorContext& context,
                                                        const BlockedAttentionPlanDesc& desc) {
  // Reuse the plan's complete capability/shape validation before allocating.
  (void)BlockedAttentionPlan::create(context, desc);
  const uint64_t shape[] = {desc.sequence, desc.heads, desc.head_dim};
  DeviceTensor query = context.allocate(TensorLayout::contiguous(shape, 3), ScalarType::kFloat16);
  DeviceTensor key = context.allocate(TensorLayout::contiguous(shape, 3), ScalarType::kFloat16);
  DeviceTensor value = context.allocate(TensorLayout::contiguous(shape, 3), ScalarType::kFloat16);
  auto result = std::make_shared<Impl>();
  result->owner = context.impl_;
  result->desc = desc;
  result->query = std::move(query);
  result->key = std::move(key);
  result->value = std::move(value);
  result->reserved_bytes = checked_multiply(
      checked_multiply(checked_multiply(desc.sequence, desc.heads, "attention preparation"),
                       desc.head_dim, "attention preparation"),
      3 * sizeof(uint16_t), "attention preparation");
  return PreparedAttentionInputs(std::move(result));
}

PreparedAttentionView PreparedAttentionInputs::prepare(TensorBatch& batch, DeviceTensor& query,
                                                       DeviceTensor& key, DeviceTensor& value) {
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
           tensor->layout.extent[1] == desc.heads && tensor->layout.extent[2] == desc.head_dim;
  };
  if (!valid_input(q) || !valid_input(k) || !valid_input(v) || q.get() == k.get() ||
      q.get() == v.get() || k.get() == v.get()) {
    throw std::invalid_argument("vulkan attention: invalid preparation inputs");
  }
  const uint64_t elements =
      checked_multiply(checked_multiply(desc.sequence, desc.heads, "attention prepare"),
                       desc.head_dim, "attention prepare");
  const uint64_t word_count = (elements + 1) / 2;
  if (word_count == 0 || word_count > std::numeric_limits<uint32_t>::max()) {
    throw std::out_of_range("vulkan attention: preparation indexing overflow");
  }
  impl_->owner->validate_dispatch(word_count);
  if (impl_->generation == std::numeric_limits<uint64_t>::max()) {
    throw std::overflow_error("vulkan attention: preparation generation exhausted");
  }
  PreparedAttentionView result(impl_, batch.impl_->batch_id, impl_->generation + 1);
  try {
    batch.impl_->count_operator();
    batch.impl_->transition(q, BufferAccess::kComputeRead);
    batch.impl_->transition(k, BufferAccess::kComputeRead);
    batch.impl_->transition(v, BufferAccess::kComputeRead);
    batch.impl_->transition(q16, BufferAccess::kComputeWrite);
    batch.impl_->transition(k16, BufferAccess::kComputeWrite);
    batch.impl_->transition(v16, BufferAccess::kComputeWrite);
    std::array<std::shared_ptr<DeviceTensor::Impl>, 6> resources{q, k, v, q16, k16, v16};
    batch.impl_->dispatch_attention_prepare(static_cast<uint32_t>(word_count), resources);
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

PreparedAttentionView::PreparedAttentionView(std::shared_ptr<void> slot, uintptr_t batch_id,
                                             uint64_t generation) noexcept
    : slot_(std::move(slot)), batch_id_(batch_id), generation_(generation) {
}

PreparedAttentionView::PreparedAttentionView(PreparedAttentionView&&) noexcept = default;
PreparedAttentionView& PreparedAttentionView::operator=(PreparedAttentionView&&) noexcept = default;

PreparedAttentionView::operator bool() const noexcept {
  return slot_ != nullptr;
}

void BlockedAttentionPlan::record(TensorBatch& batch, PreparedAttentionView& inputs,
                                  DeviceTensor& output, uint32_t query_row_offset, uint32_t rows,
                                  uint32_t output_row_offset) const {
  if (!impl_ || !batch.impl_ || batch.impl_->poisoned) {
    throw std::logic_error("vulkan attention: empty plan or batch");
  }
  if (batch.impl_->owner != impl_->owner) {
    throw std::invalid_argument("vulkan attention: plan belongs to another context");
  }
  auto prepared = std::static_pointer_cast<PreparedAttentionInputs::Impl>(inputs.slot_);
  if (!prepared || prepared->owner != impl_->owner || inputs.batch_id_ != batch.impl_->batch_id ||
      inputs.batch_id_ != prepared->batch_id || inputs.generation_ != prepared->generation ||
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
  const std::array<uint64_t, 4> expected{desc.sequence, desc.heads, desc.head_dim, 0};
  auto valid_layout = [&](const std::shared_ptr<DeviceTensor::Impl>& tensor) {
    return tensor->type == ScalarType::kFloat16 && tensor->layout.rank == 3 &&
           tensor->layout.is_contiguous() && tensor->layout.extent[0] == expected[0] &&
           tensor->layout.extent[1] == expected[1] && tensor->layout.extent[2] == expected[2];
  };
  const uint64_t query_end = static_cast<uint64_t>(query_row_offset) + selected_rows;
  const uint64_t output_end = static_cast<uint64_t>(output_row_offset) + selected_rows;
  if (selected_rows == 0 || query_end > desc.sequence || output_end > desc.sequence ||
      selected_rows > impl_->owner->max_dispatch_x || desc.heads > impl_->owner->max_dispatch_y ||
      !valid_layout(q) || !valid_layout(k) || !valid_layout(v) ||
      out->type != ScalarType::kBFloat16 || out->layout.rank != 3 || !out->layout.is_contiguous() ||
      out->layout.extent[0] != expected[0] || out->layout.extent[1] != expected[1] ||
      out->layout.extent[2] != expected[2] || q.get() == k.get() || q.get() == v.get() ||
      k.get() == v.get() || out->identity == prepared->source_id[0] ||
      out->identity == prepared->source_id[1] || out->identity == prepared->source_id[2]) {
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

H3AttentionRanges::H3AttentionRanges(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {
}

H3AttentionRanges::operator bool() const noexcept {
  return impl_ != nullptr;
}

H3AttentionRanges H3AttentionRanges::create(TensorContext& context, uint32_t sequence,
                                            const int32_t* values, uint32_t value_count) {
  if (!context.impl_)
    throw std::invalid_argument("vulkan H3 attention: empty context");
  if (!context.impl_->exact_h3_attention && !context.impl_->flash_attention &&
      !context.impl_->sage_attention) {
    throw std::runtime_error(
        "vulkan H3 attention: no supported attention mode on this device/driver");
  }
  if (sequence == 0 || !values) {
    throw std::invalid_argument("vulkan H3 attention: invalid range table");
  }
  const uint64_t tiles64 = (static_cast<uint64_t>(sequence) + 127) / 128;
  const uint64_t count64 = checked_multiply(tiles64, 4, "H3 range table");
  const uint64_t aligned_end64 = ((static_cast<uint64_t>(sequence) + 63) / 64) * 64;
  if (tiles64 > std::numeric_limits<uint32_t>::max() || count64 != value_count ||
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
    if (!valid_endpoint(lo0) || !valid_endpoint(hi0) || !valid_endpoint(lo1) ||
        !valid_endpoint(hi1) || lo0 >= hi0 || lo0 >= static_cast<int32_t>(sequence) ||
        std::min<uint32_t>(static_cast<uint32_t>(hi0), sequence) <= static_cast<uint32_t>(lo0)) {
      throw std::invalid_argument("vulkan H3 attention: invalid primary range");
    }
    if (lo1 == 0 && hi1 == 0)
      continue;
    if (lo1 >= hi1 || lo1 < lo0 || lo1 >= static_cast<int32_t>(sequence) || !valid_endpoint(lo1) ||
        !valid_endpoint(hi1) ||
        std::min<uint32_t>(static_cast<uint32_t>(hi1), sequence) <= static_cast<uint32_t>(lo1)) {
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
  DeviceTensor tensor = context.allocate(TensorLayout::contiguous(shape, 2), ScalarType::kInt32);
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
  if (!impl_)
    throw std::logic_error("vulkan H3 attention: empty range table");
  return impl_->sequence;
}

uint32_t H3AttentionRanges::query_tiles() const {
  if (!impl_)
    throw std::logic_error("vulkan H3 attention: empty range table");
  return impl_->query_tiles;
}

uint64_t H3AttentionRanges::resident_bytes() const noexcept {
  return impl_ && impl_->tensor ? impl_->tensor.layout().bytes(impl_->tensor.type()) : 0;
}

uint64_t H3AttentionRanges::content_hash() const {
  if (!impl_)
    throw std::logic_error("vulkan H3 attention: empty range table");
  return impl_->content_hash;
}

bool H3AttentionRanges::belongs_to(const TensorContext& context) const noexcept {
  return impl_ && context.impl_ && impl_->owner == context.impl_;
}

H3AttentionPlan::H3AttentionPlan() = default;
H3AttentionPlan::~H3AttentionPlan() = default;
H3AttentionPlan::H3AttentionPlan(H3AttentionPlan&&) noexcept = default;
H3AttentionPlan& H3AttentionPlan::operator=(H3AttentionPlan&&) noexcept = default;

H3AttentionPlan::H3AttentionPlan(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {
}

H3AttentionPlan::operator bool() const noexcept {
  return impl_ != nullptr;
}

H3AttentionPlan H3AttentionPlan::create(TensorContext& context, const H3AttentionPlanDesc& desc) {
  if (!context.impl_)
    throw std::invalid_argument("vulkan H3 attention: empty context");
  context.require_h3_attention(desc.mode);
  if (desc.sequence == 0 || desc.heads == 0 || (desc.head_dim != 64 && desc.head_dim != 128) ||
      !std::isfinite(desc.scale) || desc.scale <= 0.0f ||
      (desc.mode == AttentionMode::kExact &&
       !is_exact_attention_scale(desc.head_dim, desc.scale))) {
    throw std::invalid_argument("vulkan H3 attention: invalid plan");
  }
  uint64_t elements = checked_multiply(desc.sequence, desc.heads, "H3 attention");
  elements = checked_multiply(elements, desc.head_dim, "H3 attention");
  if (elements > std::numeric_limits<uint32_t>::max() ||
      desc.sequence > context.impl_->max_dispatch_x || desc.heads > context.impl_->max_dispatch_y ||
      checked_multiply(elements, 2, "H3 attention") > context.impl_->max_storage_bytes) {
    throw std::out_of_range("vulkan H3 attention: plan exceeds device/index limits");
  }
  auto result = std::make_shared<Impl>();
  result->owner = context.impl_;
  result->desc = desc;
  if (desc.mode == AttentionMode::kSage2) {
    result->sage = detail::select_sage_configuration(context.impl_->sage_device_info, desc,
                                                     context.impl_->sage_extra_workspace_bytes);
    uint64_t aux_elements =
        uint64_t(desc.heads) * (desc.head_dim + 2ull * ((uint64_t(desc.sequence) + 15) / 16));
    if (result->sage.parallel_mean)
      aux_elements +=
          uint64_t(desc.heads) * desc.head_dim * ((uint64_t(desc.sequence) + 255) / 256);
    const uint64_t shape[] = {elements};
    const uint64_t aux_shape[] = {aux_elements};
    result->quantized_query =
        context.allocate(TensorLayout::contiguous(shape, 1), ScalarType::kInt8);
    result->quantized_key = context.allocate(TensorLayout::contiguous(shape, 1), ScalarType::kInt8);
    result->sage_aux =
        context.allocate(TensorLayout::contiguous(aux_shape, 1), ScalarType::kFloat32);
    if (result->sage.prepared_value)
      result->prepared_value =
          context.allocate(TensorLayout::contiguous(shape, 1), ScalarType::kFloat16);
  }
  return H3AttentionPlan(std::move(result));
}

uint64_t H3AttentionPlan::workspace_bytes() const noexcept {
  return impl_ && impl_->desc.mode == AttentionMode::kSage2
             ? impl_->sage.required_workspace_bytes + impl_->sage.extra_workspace_bytes
             : 0;
}

SageAttentionConfiguration H3AttentionPlan::sage_configuration() const {
  if (!impl_ || impl_->desc.mode != AttentionMode::kSage2)
    throw std::logic_error("vulkan Sage: configuration requires a Sage plan");
  return impl_->sage;
}

const H3AttentionPlanDesc& H3AttentionPlan::description() const {
  if (!impl_)
    throw std::logic_error("vulkan H3 attention: empty plan");
  return impl_->desc;
}

void H3AttentionPlan::record(TensorBatch& batch, DeviceTensor& query, DeviceTensor& key,
                             DeviceTensor& value, DeviceTensor& output,
                             const H3AttentionRanges* ranges, uint32_t query_row_offset,
                             uint32_t rows, uint32_t output_row_offset,
                             TimestampQuery* sage_timestamps) const {
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
        ranges->impl_->query_tiles != (static_cast<uint64_t>(impl_->desc.sequence) + 127) / 128) {
      throw std::invalid_argument("vulkan H3 attention: incompatible range table");
    }
    range_tensor = impl_->owner->require(ranges->impl_->tensor);
  }
  const auto& desc = impl_->desc;
  if (sage_timestamps && (desc.mode != AttentionMode::kSage2 || sage_timestamps->count() < 4))
    throw std::invalid_argument("vulkan Sage: timing requires four timestamp queries");
  const uint32_t selected_rows =
      rows == 0 && query_row_offset <= desc.sequence ? desc.sequence - query_row_offset : rows;
  auto valid_layout = [&](const std::shared_ptr<DeviceTensor::Impl>& tensor) {
    const bool shaped = tensor->layout.rank == 3 && tensor->layout.extent[1] == desc.heads &&
                        tensor->layout.extent[2] == desc.head_dim;
    const bool flat = tensor->layout.rank == 2 &&
                      tensor->layout.extent[1] == static_cast<uint64_t>(desc.heads) * desc.head_dim;
    return tensor->type == ScalarType::kBFloat16 && (shaped || flat) &&
           tensor->layout.is_contiguous() && tensor->layout.extent[0] == desc.sequence;
  };
  const uint64_t query_end = static_cast<uint64_t>(query_row_offset) + selected_rows;
  const uint64_t output_end = static_cast<uint64_t>(output_row_offset) + selected_rows;
  if (query_row_offset > desc.sequence || selected_rows == 0 || query_end > desc.sequence ||
      output_end > desc.sequence || selected_rows > impl_->owner->max_dispatch_x ||
      desc.heads > impl_->owner->max_dispatch_y || !valid_layout(q) || !valid_layout(k) ||
      !valid_layout(v) || !valid_layout(out) || q.get() == k.get() || q.get() == v.get() ||
      k.get() == v.get() || out.get() == q.get() || out.get() == k.get() || out.get() == v.get()) {
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
    if (desc.mode == AttentionMode::kSage2) {
      auto q8 = impl_->owner->require(impl_->quantized_query);
      auto k8 = impl_->owner->require(impl_->quantized_key);
      auto aux = impl_->owner->require(impl_->sage_aux);
      auto pv = impl_->sage.prepared_value ? impl_->owner->require(impl_->prepared_value) : v;
      auto& commands = batch.impl_->commands;
      if (sage_timestamps) {
        commands.reset_timestamps(*sage_timestamps);
        commands.write_timestamp(*sage_timestamps, 0);
      }
      auto& bindings = impl_->owner->attention_sage_prepare_bindings;
      std::array<std::shared_ptr<DeviceTensor::Impl>, 7> prep{
          q, k, q8, k8, aux, v, impl_->sage.prepared_value ? pv : q8};
      for (size_t i = 0; i < prep.size(); ++i) {
        bindings[i].buffer = &prep[i]->buffer;
        bindings[i].bytes = prep[i]->buffer.size();
      }
      batch.impl_->transition(q8, BufferAccess::kComputeWrite);
      batch.impl_->transition(k8, BufferAccess::kComputeWrite);
      batch.impl_->transition(aux, BufferAccess::kComputeWrite);
      if (impl_->sage.prepared_value)
        batch.impl_->transition(pv, BufferAccess::kComputeWrite);
      commands.bind_compute(impl_->owner->attention_sage_prepare_pipeline, bindings);
      parameters.reserved = impl_->sage.parallel_mean ? 2 : 0;
      commands.push_constants(&parameters, sizeof(parameters));
      commands.dispatch(impl_->sage.parallel_mean ? (desc.sequence + 255u) / 256u : 1u, desc.heads);
      if (impl_->sage.parallel_mean) {
        batch.impl_->transition(aux, BufferAccess::kComputeReadWrite);
        parameters.reserved = 3;
        commands.push_constants(&parameters, sizeof(parameters));
        commands.dispatch(1, desc.heads);
      }
      batch.impl_->transition(aux, BufferAccess::kComputeReadWrite);
      if (sage_timestamps)
        commands.write_timestamp(*sage_timestamps, 1);
      parameters.reserved = 1 | (impl_->sage.prepared_value ? 8u : 0u);
      commands.push_constants(&parameters, sizeof(parameters));
      commands.dispatch((desc.sequence + 15u) / 16u, desc.heads);
      if (sage_timestamps)
        commands.write_timestamp(*sage_timestamps, 2);
      batch.impl_->transition(q8, BufferAccess::kComputeRead);
      batch.impl_->transition(k8, BufferAccess::kComputeRead);
      batch.impl_->transition(aux, BufferAccess::kComputeRead);
      if (impl_->sage.prepared_value)
        batch.impl_->transition(pv, BufferAccess::kComputeRead);
      if (range_tensor)
        batch.impl_->transition(range_tensor, BufferAccess::kComputeRead);
      std::array<std::shared_ptr<DeviceTensor::Impl>, 6> resources{
          q8, k8, pv, out, range_tensor ? range_tensor : aux, aux};
      auto& sage_bindings = impl_->owner->attention_sage_bindings;
      for (size_t i = 0; i < resources.size(); ++i) {
        sage_bindings[i].buffer = &resources[i]->buffer;
        sage_bindings[i].bytes = resources[i]->buffer.size();
      }
      const uint32_t kernel = impl_->sage.kernel - 1;
      const uint32_t variant =
          (desc.head_dim == 128 ? 2u : 0u) + (impl_->sage.prepared_value ? 1u : 0u);
      commands.bind_compute(range_tensor
                                ? impl_->owner->attention_sage_banded_pipelines[kernel][variant]
                                : impl_->owner->attention_sage_pipelines[kernel][variant],
                            sage_bindings);
      parameters.reserved = impl_->sage.prepared_value ? 1 : 0;
      commands.push_constants(&parameters, sizeof(parameters));
      const uint32_t tile = impl_->sage.query_rows;
      commands.dispatch((query_row_offset % tile + selected_rows + tile - 1u) / tile, desc.heads);
      if (sage_timestamps)
        commands.write_timestamp(*sage_timestamps, 3);
    } else if (range_tensor) {
      batch.impl_->transition(range_tensor, BufferAccess::kComputeRead);
      std::array<std::shared_ptr<DeviceTensor::Impl>, 5> resources{q, k, v, out, range_tensor};
      batch.impl_->dispatch_h3_banded_attention(parameters, resources, desc.mode);
    } else {
      std::array<std::shared_ptr<DeviceTensor::Impl>, 4> resources{q, k, v, out};
      batch.impl_->dispatch_h3_attention(parameters, resources, desc.mode);
    }
  } catch (...) {
    batch.impl_->poisoned = true;
    throw;
  }
}

VsaAttentionPlan::VsaAttentionPlan() = default;
VsaAttentionPlan::~VsaAttentionPlan() = default;
VsaAttentionPlan::VsaAttentionPlan(VsaAttentionPlan&&) noexcept = default;
VsaAttentionPlan& VsaAttentionPlan::operator=(VsaAttentionPlan&&) noexcept = default;

VsaAttentionPlan::VsaAttentionPlan(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {
}

VsaAttentionPlan VsaAttentionPlan::create(TensorContext& context, const dit::VsaTiles& tiles,
                                          uint32_t heads, uint32_t head_dim) {
  context.require_h3_attention(AttentionMode::kFlash2);
  const size_t n = tiles.sizes.size(), sequence = tiles.row_tiles.size();
  if (n == 0 || n > 4096 || sequence == 0 || sequence > UINT32_MAX || tiles.rows.size() != n * 64 ||
      tiles.prefix_tiles < 0 || static_cast<size_t>(tiles.prefix_tiles) >= n || heads == 0 ||
      (head_dim != 64 && head_dim != 128))
    throw std::invalid_argument("Vulkan VSA: invalid tile geometry or head shape");
  const uint64_t elements = uint64_t(sequence) * heads * head_dim;
  if (elements > UINT32_MAX || elements * 2 > context.impl_->max_storage_bytes ||
      n * 2 > context.impl_->max_dispatch_x || heads > context.impl_->max_dispatch_y ||
      (elements + 511) / 512 >
          uint64_t(context.impl_->max_dispatch_x) * context.impl_->max_dispatch_y)
    throw std::out_of_range("Vulkan VSA: shape exceeds device/index limits");
  std::vector<bool> seen(sequence, false);
  for (size_t tile = 0; tile < n; ++tile) {
    if (tiles.sizes[tile] < 1 || tiles.sizes[tile] > 64)
      throw std::invalid_argument("Vulkan VSA: tile size must be in [1,64]");
    for (int i = 0; i < 64; ++i) {
      const int row = tiles.rows[tile * 64 + i];
      if (i >= tiles.sizes[tile]) {
        if (row != -1)
          throw std::invalid_argument("Vulkan VSA: invalid padding");
      } else {
        if (row < 0 || static_cast<size_t>(row) >= sequence || seen[row] ||
            tiles.row_tiles[row] != static_cast<int>(tile))
          throw std::invalid_argument("Vulkan VSA: invalid row mapping");
        seen[row] = true;
      }
    }
  }
  if (std::find(seen.begin(), seen.end(), false) != seen.end())
    throw std::invalid_argument("Vulkan VSA: incomplete row mapping");
  auto result = std::make_shared<Impl>();
  result->owner = context.impl_;
  result->sequence = static_cast<uint32_t>(sequence);
  result->heads = heads;
  result->dim = head_dim;
  result->tiles = static_cast<uint32_t>(n);
  result->prefix = tiles.prefix_tiles;
  result->padded = 1;
  while (result->padded < n - tiles.prefix_tiles)
    result->padded *= 2;
  auto allocate = [&](uint64_t count, ScalarType type) {
    return context.allocate(TensorLayout::contiguous(&count, 1), type);
  };
  std::vector<int32_t> geometry = tiles.rows;
  geometry.insert(geometry.end(), tiles.sizes.begin(), tiles.sizes.end());
  geometry.insert(geometry.end(), tiles.row_tiles.begin(), tiles.row_tiles.end());
  result->geometry = allocate(geometry.size(), ScalarType::kInt32);
  result->pooled = allocate(3ull * n * heads * head_dim, ScalarType::kFloat32);
  result->mask = allocate(uint64_t(heads) * n * ((n + 31) / 32), ScalarType::kInt32);
  result->compressed = allocate(n * heads * head_dim, ScalarType::kBFloat16);
  context.upload_transient_bytes(result->geometry, geometry.data(), geometry.size() * 4);
  return VsaAttentionPlan(std::move(result));
}

uint64_t VsaAttentionPlan::workspace_bytes() const noexcept {
  if (!impl_)
    return 0;
  uint64_t count = 0;
  for (const DeviceTensor* tensor :
       {&impl_->geometry, &impl_->pooled, &impl_->mask, &impl_->compressed})
    count += tensor->layout().bytes(tensor->type());
  return count;
}

void VsaAttentionPlan::record(TensorBatch& batch, DeviceTensor& query, DeviceTensor& key,
                              DeviceTensor& value, DeviceTensor& output) const {
  if (!impl_ || !batch.impl_ || batch.impl_->poisoned)
    throw std::logic_error("Vulkan VSA: empty plan or batch");
  auto& s = *impl_;
  if (batch.impl_->owner != s.owner)
    throw std::invalid_argument("Vulkan VSA: incompatible context");
  auto q = s.owner->require(query), k = s.owner->require(key);
  auto v = s.owner->require(value), out = s.owner->require(output);
  auto valid = [&](const auto& tensor) {
    const auto& l = tensor->layout;
    return tensor->type == ScalarType::kBFloat16 && l.is_contiguous() &&
           l.extent[0] == s.sequence &&
           ((l.rank == 3 && l.extent[1] == s.heads && l.extent[2] == s.dim) ||
            (l.rank == 2 && l.extent[1] == uint64_t(s.heads) * s.dim));
  };
  if (!valid(q) || !valid(k) || !valid(v) || !valid(out) || q == k || q == v || k == v ||
      out == q || out == k || out == v)
    throw std::invalid_argument("Vulkan VSA: invalid tensor shape or alias");
  if (batch.remaining_operator_capacity() < 3)
    throw std::logic_error("Vulkan VSA: insufficient batch capacity");
  auto geometry = s.owner->require(s.geometry), pooled = s.owner->require(s.pooled);
  auto mask = s.owner->require(s.mask), compressed = s.owner->require(s.compressed);
  uint32_t parameters[] = {
      s.sequence, s.heads, s.dim, s.tiles, s.prefix, s.padded, (s.tiles - s.prefix + 4) / 5, 0};
  try {
    for (auto& tensor : {q, k, v, geometry})
      batch.impl_->transition(tensor, BufferAccess::kComputeRead);
    batch.impl_->transition(pooled, BufferAccess::kComputeWrite);
    batch.impl_->transition(mask, BufferAccess::kComputeWrite);
    batch.impl_->transition(compressed, BufferAccess::kComputeWrite);
    batch.impl_->transition(out, BufferAccess::kComputeWrite);
    const std::array<std::shared_ptr<DeviceTensor::Impl>, 8> resources{
        q, k, v, geometry, pooled, mask, compressed, out};
    std::vector<StorageBinding> bindings(8);
    for (uint32_t i = 0; i < 8; ++i) {
      bindings[i].binding = i;
      bindings[i].buffer = &resources[i]->buffer;
      bindings[i].bytes = resources[i]->buffer.size();
    }
    auto& commands = batch.impl_->commands;
    batch.impl_->count_operator();
    commands.bind_compute(s.owner->attention_vsa_prepare_pipeline, bindings);
    commands.push_constants(parameters, sizeof(parameters));
    commands.dispatch(s.tiles, s.heads);
    batch.impl_->transition(pooled, BufferAccess::kComputeRead);
    parameters[7] = 1;
    batch.impl_->count_operator();
    commands.push_constants(parameters, sizeof(parameters));
    commands.dispatch(s.tiles, s.heads);
    batch.impl_->transition(mask, BufferAccess::kComputeRead);
    std::vector<StorageBinding> sparse(bindings.begin(), bindings.begin() + 4);
    sparse[3] = bindings[7];
    sparse[3].binding = 3;
    sparse.push_back(bindings[3]);
    sparse[4].binding = 4;
    sparse.push_back(bindings[5]);
    sparse[5].binding = 5;
    TensorContext::Impl::AttentionParameters attention;
    attention.sequence = s.sequence;
    attention.heads = s.heads;
    attention.head_dim = s.dim;
    const float scale = 1.0f / std::sqrt(float(s.dim));
    std::memcpy(&attention.scale_bits, &scale, sizeof(scale));
    attention.rows = s.sequence;
    attention.reserved = s.tiles;
    batch.impl_->count_operator();
    commands.bind_compute(s.owner->attention_vsa_pipeline, sparse);
    commands.push_constants(&attention, sizeof(attention));
    commands.dispatch(s.tiles * 2, s.heads);
    s.batch_id = batch.impl_->batch_id;
    s.output_id = out->identity;
  } catch (...) {
    batch.impl_->poisoned = true;
    throw;
  }
}

void VsaAttentionPlan::add_compression(TensorBatch& batch, DeviceTensor& gate,
                                       DeviceTensor& output) const {
  if (!impl_ || !batch.impl_ || batch.impl_->poisoned)
    throw std::logic_error("Vulkan VSA: empty plan or batch");
  auto& s = *impl_;
  if (batch.impl_->owner != s.owner || s.batch_id != batch.impl_->batch_id)
    throw std::invalid_argument("Vulkan VSA: compression must follow attention in the same batch");
  auto g = s.owner->require(gate), out = s.owner->require(output);
  if (g->type != ScalarType::kBFloat16 || g->layout.rank != 2 ||
      g->layout.extent[0] != s.sequence || g->layout.extent[1] != uint64_t(s.heads) * s.dim ||
      !g->layout.is_contiguous() || g == out || out->identity != s.output_id)
    throw std::invalid_argument("Vulkan VSA: invalid gate or output");
  if (batch.remaining_operator_capacity() < 1)
    throw std::logic_error("Vulkan VSA: insufficient batch capacity");
  auto geometry = s.owner->require(s.geometry), compressed = s.owner->require(s.compressed);
  const uint64_t groups = (uint64_t(s.sequence) * s.heads * s.dim + 511) / 512;
  const uint32_t groups_x =
      static_cast<uint32_t>(std::min<uint64_t>(groups, s.owner->max_dispatch_x));
  uint32_t parameters[] = {s.sequence, s.heads, s.dim, s.tiles, s.prefix, s.padded, groups_x, 2};
  try {
    batch.impl_->count_operator();
    batch.impl_->transition(g, BufferAccess::kComputeRead);
    batch.impl_->transition(geometry, BufferAccess::kComputeRead);
    batch.impl_->transition(compressed, BufferAccess::kComputeRead);
    batch.impl_->transition(out, BufferAccess::kComputeReadWrite);
    const std::array<std::shared_ptr<DeviceTensor::Impl>, 8> resources{g, g, g,          geometry,
                                                                       g, g, compressed, out};
    std::vector<StorageBinding> bindings(8);
    for (uint32_t i = 0; i < 8; ++i) {
      bindings[i].binding = i;
      bindings[i].buffer = &resources[i]->buffer;
      bindings[i].bytes = resources[i]->buffer.size();
    }
    batch.impl_->commands.bind_compute(s.owner->attention_vsa_prepare_pipeline, bindings);
    batch.impl_->commands.push_constants(parameters, sizeof(parameters));
    batch.impl_->commands.dispatch(groups_x,
                                   static_cast<uint32_t>((groups + groups_x - 1) / groups_x));
  } catch (...) {
    batch.impl_->poisoned = true;
    throw;
  }
}

CausalGQAAttentionPlan::CausalGQAAttentionPlan() = default;
CausalGQAAttentionPlan::~CausalGQAAttentionPlan() = default;
CausalGQAAttentionPlan::CausalGQAAttentionPlan(CausalGQAAttentionPlan&&) noexcept = default;
CausalGQAAttentionPlan&
CausalGQAAttentionPlan::operator=(CausalGQAAttentionPlan&&) noexcept = default;

CausalGQAAttentionPlan::CausalGQAAttentionPlan(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {
}

CausalGQAAttentionPlan::operator bool() const noexcept {
  return impl_ != nullptr;
}

CausalGQAAttentionPlan CausalGQAAttentionPlan::create(TensorContext& context,
                                                      const CausalGQAAttentionPlanDesc& desc) {
  if (!context.impl_) {
    throw std::invalid_argument("vulkan causal GQA attention: empty context");
  }
  context.impl_->require_pipeline_set(TensorPipelineSet::kTextAttention);
  if (!context.impl_->exact_causal_gqa_attention) {
    throw std::runtime_error(
        "vulkan causal GQA attention: exact mode is unavailable on this device/driver");
  }
  // This plan names the shipped Qwen3-VL text contract rather than advertising
  // an unverified generic GQA family.
  const AttentionDescriptor descriptor{desc.sequence,
                                       desc.sequence,
                                       desc.query_heads,
                                       desc.kv_heads,
                                       desc.head_dim,
                                       AttentionLayout::kTokensHeadsChannels,
                                       AttentionMask::kCausal,
                                       AttentionArithmetic::kExact,
                                       desc.scale};
  if (!supports_exact_text_attention(descriptor, text::kMaxPromptTokens)) {
    throw std::invalid_argument("vulkan causal GQA attention: invalid Qwen text plan");
  }
  uint64_t query_elements =
      checked_multiply(desc.sequence, desc.query_heads, "causal GQA attention");
  query_elements = checked_multiply(query_elements, desc.head_dim, "causal GQA attention");
  uint64_t kv_elements = checked_multiply(desc.sequence, desc.kv_heads, "causal GQA attention");
  kv_elements = checked_multiply(kv_elements, desc.head_dim, "causal GQA attention");
  if (query_elements > std::numeric_limits<uint32_t>::max() ||
      kv_elements > std::numeric_limits<uint32_t>::max() ||
      desc.sequence > context.impl_->max_dispatch_x ||
      desc.query_heads > context.impl_->max_dispatch_y ||
      checked_multiply(query_elements, sizeof(uint16_t), "causal GQA attention") >
          context.impl_->max_storage_bytes ||
      checked_multiply(kv_elements, sizeof(uint16_t), "causal GQA attention") >
          context.impl_->max_storage_bytes) {
    throw std::out_of_range("vulkan causal GQA attention: plan exceeds device/index limits");
  }
  auto result = std::make_shared<Impl>();
  result->owner = context.impl_;
  result->desc = desc;
  return CausalGQAAttentionPlan(std::move(result));
}

const CausalGQAAttentionPlanDesc& CausalGQAAttentionPlan::description() const {
  if (!impl_)
    throw std::logic_error("vulkan causal GQA attention: empty plan");
  return impl_->desc;
}

void CausalGQAAttentionPlan::record(TensorBatch& batch, DeviceTensor& query, DeviceTensor& key,
                                    DeviceTensor& value, DeviceTensor& output,
                                    uint32_t query_row_offset, uint32_t rows,
                                    uint32_t output_row_offset) const {
  if (!impl_ || !batch.impl_ || batch.impl_->poisoned) {
    throw std::logic_error("vulkan causal GQA attention: empty plan or batch");
  }
  if (batch.impl_->owner != impl_->owner) {
    throw std::invalid_argument("vulkan causal GQA attention: plan belongs to another context");
  }
  auto q = impl_->owner->require(query);
  auto k = impl_->owner->require(key);
  auto v = impl_->owner->require(value);
  auto out = impl_->owner->require(output);
  const auto& desc = impl_->desc;
  if (query_row_offset > desc.sequence) {
    throw std::invalid_argument("vulkan causal GQA attention: query row offset is out of range");
  }
  const uint32_t selected_rows = rows == 0 ? desc.sequence - query_row_offset : rows;
  const uint64_t query_end = static_cast<uint64_t>(query_row_offset) + selected_rows;
  const uint64_t output_end = static_cast<uint64_t>(output_row_offset) + selected_rows;
  auto valid_layout = [&](const std::shared_ptr<DeviceTensor::Impl>& tensor, uint32_t heads) {
    return tensor->type == ScalarType::kBFloat16 && tensor->layout.rank == 3 &&
           tensor->layout.is_contiguous() && tensor->layout.extent[0] == desc.sequence &&
           tensor->layout.extent[1] == heads && tensor->layout.extent[2] == desc.head_dim;
  };
  if (selected_rows == 0 || query_end > desc.sequence || output_end > desc.sequence ||
      selected_rows > impl_->owner->max_dispatch_x ||
      desc.query_heads > impl_->owner->max_dispatch_y || !valid_layout(q, desc.query_heads) ||
      !valid_layout(k, desc.kv_heads) || !valid_layout(v, desc.kv_heads) ||
      !valid_layout(out, desc.query_heads) || q->identity == k->identity ||
      q->identity == v->identity || q->identity == out->identity || k->identity == v->identity ||
      k->identity == out->identity || v->identity == out->identity) {
    throw std::invalid_argument("vulkan causal GQA attention: invalid tensor/range/alias");
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

} // namespace slopfab::vulkan
