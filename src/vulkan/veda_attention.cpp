#include "tensor_recording.h"
#include "slopfab/vulkan/veda_attention.h"

namespace slopfab::vulkan {
struct VedaAttentionPlan::Impl {
  std::shared_ptr<TensorContext::Impl> owner;
  std::shared_ptr<dit::VedaRequest> request;
  std::vector<DeviceTensor> geometry;
  DeviceTensor features, projected, mask;
};

VedaAttentionPlan VedaAttentionPlan::create(TensorContext& context,
                                            std::shared_ptr<dit::VedaRequest> request) {
  context.require_h3_attention(AttentionMode::kFlash2);
  if (!request || request->tilings.empty())
    throw std::invalid_argument("Vulkan VEDA: missing request");
  const auto& b = request->bundle;
  const uint64_t elements = uint64_t(request->layout.total_rows()) * b.heads * b.head_dim;
  if (elements > UINT32_MAX || elements * 2 > context.impl_->max_storage_bytes)
    throw std::out_of_range("Vulkan VEDA: sequence exceeds device/index limits");
  VedaAttentionPlan result;
  result.impl_ = std::make_shared<Impl>();
  auto& s = *result.impl_;
  s.owner = context.impl_;
  s.request = std::move(request);
  auto allocate = [&](uint64_t n, ScalarType type) {
    return context.allocate(TensorLayout::contiguous(&n, 1), type);
  };
  size_t max_video = 0, max_tiles = 0;
  for (size_t i = 0; i < s.request->tilings.size(); ++i) {
    const auto& t = s.request->tilings[i];
    if (t.sizes.size() * 4 > context.impl_->max_dispatch_x)
      throw std::out_of_range("Vulkan VEDA: tile count exceeds dispatch limits");
    std::vector<int32_t> data = t.rows;
    data.insert(data.end(), t.sizes.begin(), t.sizes.end());
    data.insert(data.end(), s.request->keeps[i].begin(), s.request->keeps[i].end());
    auto tensor = allocate(data.size(), ScalarType::kInt32);
    context.upload_transient_bytes(tensor, data.data(), data.size() * 4);
    s.geometry.push_back(std::move(tensor));
    max_video = std::max(max_video, size_t(t.video_tiles));
    max_tiles = std::max(max_tiles, t.sizes.size());
  }
  s.features = allocate(6 * max_video * b.head_dim, ScalarType::kFloat32);
  s.projected = allocate(2 * max_video * b.head_dim, ScalarType::kFloat32);
  s.mask = allocate(max_tiles * ((max_tiles + 31) / 32), ScalarType::kInt32);
  return result;
}

uint64_t VedaAttentionPlan::workspace_bytes() const noexcept {
  if (!impl_)
    return 0;
  uint64_t result = 0;
  for (const auto* t : {&impl_->features, &impl_->projected, &impl_->mask})
    result += t->layout().bytes(t->type());
  for (const auto& t : impl_->geometry)
    result += t.layout().bytes(t.type());
  return result;
}

DeviceTensor VedaAttentionPlan::upload_projection(TensorContext& context,
                                                  const dit::VedaBundle& bundle, int layer) {
  auto data = bundle.projection(layer, true);
  const auto key = bundle.projection(layer, false);
  data.insert(data.end(), key.begin(), key.end());
  const uint64_t n = data.size();
  auto result = context.allocate(TensorLayout::contiguous(&n, 1), ScalarType::kBFloat16);
  context.upload_transient_bytes(result, data.data(), data.size() * 2);
  return result;
}

void VedaAttentionPlan::record(TensorBatch& batch, int layer, DeviceTensor& projection,
                               DeviceTensor& query, DeviceTensor& key, DeviceTensor& value,
                               DeviceTensor& output) const {
  if (!impl_ || !batch.impl_ || batch.impl_->poisoned)
    throw std::logic_error("Vulkan VEDA: empty plan or batch");
  auto& s = *impl_;
  const auto& b = s.request->bundle;
  if (batch.impl_->owner != s.owner || layer < 0 || layer >= b.layers)
    throw std::invalid_argument("Vulkan VEDA: incompatible context or layer");
  auto q = s.owner->require(query), k = s.owner->require(key), v = s.owner->require(value),
       out = s.owner->require(output);
  auto weights = s.owner->require(projection), features = s.owner->require(s.features);
  auto projected = s.owner->require(s.projected), mask = s.owner->require(s.mask);
  auto valid = [&](const auto& t) {
    const auto& l = t->layout;
    return t->type == ScalarType::kBFloat16 && l.is_contiguous() &&
           l.extent[0] == uint64_t(s.request->layout.total_rows()) &&
           ((l.rank == 3 && l.extent[1] == uint64_t(b.heads) &&
             l.extent[2] == uint64_t(b.head_dim)) ||
            (l.rank == 2 && l.extent[1] == uint64_t(b.heads) * b.head_dim));
  };
  if (!valid(q) || !valid(k) || !valid(v) || !valid(out) || q == k || q == v || k == v ||
      out == q || out == k || out == v || weights->type != ScalarType::kBFloat16 ||
      weights->layout.rank != 1 || !weights->layout.is_contiguous() ||
      weights->layout.extent[0] != uint64_t(6) * b.heads * b.head_dim * b.head_dim)
    throw std::invalid_argument("Vulkan VEDA: invalid tensor shapes or aliases");
  if (batch.remaining_operator_capacity() < uint32_t(b.heads))
    throw std::logic_error("Vulkan VEDA: insufficient batch capacity");
  try {
    for (const auto& t : {q, k, v, weights})
      batch.impl_->transition(t, BufferAccess::kComputeRead);
    auto& commands = batch.impl_->commands;
    auto bind = [&](ComputePipeline& pipeline,
                    const std::vector<std::shared_ptr<DeviceTensor::Impl>>& resources) {
      std::vector<StorageBinding> bindings(resources.size());
      for (size_t i = 0; i < resources.size(); ++i) {
        bindings[i].binding = uint32_t(i);
        bindings[i].buffer = &resources[i]->buffer;
        bindings[i].bytes = resources[i]->buffer.size();
      }
      commands.bind_compute(pipeline, bindings);
    };
    for (int head = 0; head < b.heads; ++head) {
      const int index = s.request->plan.head_shape[layer][head];
      const auto& t = s.request->tilings[index];
      auto geometry = s.owner->require(s.geometry[index]);
      const uint32_t tiles = uint32_t(t.sizes.size()), video = t.video_tiles;
      uint32_t padded = 1;
      while (padded < video)
        padded *= 2;
      uint32_t parameters[] = {
          uint32_t(b.heads), uint32_t(b.head_dim), tiles, video, uint32_t(head), padded, 0, 0};
      batch.impl_->transition(geometry, BufferAccess::kComputeRead);
      batch.impl_->transition(features, BufferAccess::kComputeWrite);
      bind(s.owner->attention_veda_prepare_pipeline,
           {q, k, geometry, features, projected, weights, mask});
      batch.impl_->count_operator();
      commands.push_constants(parameters, sizeof(parameters));
      commands.dispatch(video);
      batch.impl_->transition(features, BufferAccess::kComputeRead);
      batch.impl_->transition(projected, BufferAccess::kComputeWrite);
      parameters[6] = 1;
      commands.push_constants(parameters, sizeof(parameters));
      commands.dispatch(video);
      batch.impl_->transition(projected, BufferAccess::kComputeRead);
      batch.impl_->transition(mask, BufferAccess::kComputeWrite);
      parameters[6] = 2;
      commands.push_constants(parameters, sizeof(parameters));
      commands.dispatch(tiles);
      batch.impl_->transition(mask, BufferAccess::kComputeRead);
      batch.impl_->transition(out, BufferAccess::kComputeWrite);
      TensorContext::Impl::AttentionParameters attention;
      attention.sequence = s.request->layout.total_rows();
      attention.heads = b.heads;
      attention.head_dim = b.head_dim;
      const float scale = 1.0f / std::sqrt(float(b.head_dim));
      std::memcpy(&attention.scale_bits, &scale, 4);
      attention.rows = attention.sequence;
      attention.reserved = tiles;
      attention.output_row_offset = head;
      bind(s.owner->attention_veda_pipeline, {q, k, v, out, geometry, mask});
      commands.push_constants(&attention, sizeof(attention));
      commands.dispatch(tiles * 4);
    }
  } catch (...) {
    batch.impl_->poisoned = true;
    throw;
  }
}
} // namespace slopfab::vulkan
