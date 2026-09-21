#include "tensor_recording.h"

namespace slopfab::vulkan {

void TensorBatch::rope_h3_bf16(DeviceTensor& input, DeviceTensor& cosine, DeviceTensor& sine) {
  rope_bf16(input, cosine, sine, 0);
}

void TensorBatch::rope_neox_bf16(DeviceTensor& input, DeviceTensor& cosine, DeviceTensor& sine) {
  rope_bf16(input, cosine, sine, 1);
}

void TensorBatch::rope_bf16(DeviceTensor& input, DeviceTensor& cosine, DeviceTensor& sine,
                            uint32_t mode) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
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
  if (mode > 1 || shape.rank != 3 || rows == 0 || heads == 0 || head_dim == 0 ||
      (head_dim & 1u) != 0 || head_dim > 256 || (mode == 0 && head_dim < 96) || groups_overflow ||
      groups > std::numeric_limits<uint32_t>::max() || data.get() == cos_table.get() ||
      data.get() == sin_table.get() || cos_table.get() == sin_table.get() ||
      data->type != ScalarType::kBFloat16 || cos_table->type != ScalarType::kFloat32 ||
      sin_table->type != ScalarType::kFloat32 || cos_table->layout.rank != 2 ||
      sin_table->layout.rank != 2 || cos_table->layout.extent[0] != rows ||
      sin_table->layout.extent[0] != rows || cos_table->layout.extent[1] != table_width ||
      sin_table->layout.extent[1] != table_width || !shape.is_contiguous() ||
      !cos_table->layout.is_contiguous() || !sin_table->layout.is_contiguous()) {
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
    impl_->dispatch_rope(parameters, static_cast<uint32_t>(groups), data, cos_table, sin_table);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::split_qkv_norm_rope_f32(DeviceTensor& qkv, DeviceTensor& bias,
                                          DeviceTensor& cosine, DeviceTensor& sine, DeviceTensor& q,
                                          DeviceTensor& k, DeviceTensor& v, uint32_t num_patches,
                                          float epsilon) {
  if (!impl_ || impl_->poisoned) {
    throw std::logic_error("vulkan tensor: invalid batch");
  }
  if (!impl_->owner->exact_vae_norm) {
    throw std::runtime_error("vulkan tensor: exact video-VAE fused RoPE is unavailable");
  }
  std::array<std::shared_ptr<DeviceTensor::Impl>, 7> resources{
      impl_->owner->require(qkv),  impl_->owner->require(bias), impl_->owner->require(cosine),
      impl_->owner->require(sine), impl_->owner->require(q),    impl_->owner->require(k),
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
  const uint64_t heads = flattened ? resources[4]->layout.extent[0] : input_shape.extent[1];
  const bool flattened_width_overflow =
      flattened && heads > std::numeric_limits<uint64_t>::max() / 192u;
  const bool groups_overflow = sequence != 0 && heads > UINT64_MAX / sequence;
  const uint64_t groups = groups_overflow ? 0 : sequence * heads;
  if (aliases || !std::isnormal(epsilon) || epsilon <= 0.0f ||
      (input_shape.rank != 3 && input_shape.rank != 2) || sequence == 0 || heads == 0 ||
      flattened_width_overflow ||
      (flattened ? input_shape.extent[1] != heads * 192 : input_shape.extent[2] != 192) ||
      num_patches > sequence || groups_overflow || groups > std::numeric_limits<uint32_t>::max() ||
      (flattened ? (resources[1]->layout.rank != 1 || resources[1]->layout.extent[0] != heads * 192)
                 : (resources[1]->layout.rank != 2 || resources[1]->layout.extent[0] != heads ||
                    resources[1]->layout.extent[1] != 192)) ||
      resources[2]->layout.rank != 2 || resources[3]->layout.rank != 2 ||
      resources[2]->layout.extent[0] != sequence || resources[3]->layout.extent[0] != sequence ||
      resources[2]->layout.extent[1] != 48 || resources[3]->layout.extent[1] != 48) {
    throw std::invalid_argument("vulkan tensor: invalid video-VAE fused RoPE");
  }
  for (size_t i = 0; i < resources.size(); ++i) {
    if (resources[i]->type != ScalarType::kFloat32 || !resources[i]->layout.is_contiguous()) {
      throw std::invalid_argument("vulkan tensor: invalid video-VAE fused RoPE");
    }
    if (i >= 4 &&
        (resources[i]->layout.rank != 3 || resources[i]->layout.extent[0] != heads ||
         resources[i]->layout.extent[1] != sequence || resources[i]->layout.extent[2] != 64)) {
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
    impl_->dispatch_vae_rope(parameters, static_cast<uint32_t>(groups), resources);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::rms_norm(DeviceTensor& input, DeviceTensor& weight, DeviceTensor& output,
                           float epsilon) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_vae_norm) {
    throw std::runtime_error("vulkan tensor: exact fp32 VAE RMSNorm is unavailable");
  }
  auto src = impl_->owner->require(input);
  auto w = impl_->owner->require(weight);
  auto dst = impl_->owner->require(output);
  const auto& shape = src->layout;
  if (!std::isnormal(epsilon) || epsilon <= 0.0f || src.get() == w.get() || dst.get() == w.get() ||
      src->type != ScalarType::kFloat32 || w->type != ScalarType::kFloat32 ||
      dst->type != ScalarType::kFloat32 || shape.rank != 2 || w->layout.rank != 1 ||
      dst->layout.extent != shape.extent || dst->layout.rank != 2 ||
      w->layout.extent[0] != shape.extent[1] || !shape.is_contiguous() ||
      !w->layout.is_contiguous() || !dst->layout.is_contiguous() ||
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
    if (src.get() != dst.get())
      impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_norm(impl_->owner->rms_norm_pipeline, p, src, w, w, dst);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::layer_norm(DeviceTensor& input, DeviceTensor& weight, DeviceTensor& bias,
                             DeviceTensor& output, float epsilon) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_vae_norm) {
    throw std::runtime_error("vulkan tensor: exact fp32 VAE LayerNorm is unavailable");
  }
  auto src = impl_->owner->require(input);
  auto w = impl_->owner->require(weight);
  auto b = impl_->owner->require(bias);
  auto dst = impl_->owner->require(output);
  const auto& shape = src->layout;
  if (!std::isnormal(epsilon) || epsilon <= 0.0f || src.get() == w.get() || src.get() == b.get() ||
      w.get() == b.get() || dst.get() == w.get() || dst.get() == b.get() ||
      src->type != ScalarType::kFloat32 || w->type != ScalarType::kFloat32 ||
      b->type != ScalarType::kFloat32 || dst->type != ScalarType::kFloat32 || shape.rank != 2 ||
      w->layout.rank != 1 || b->layout.rank != 1 || dst->layout.rank != 2 ||
      dst->layout.extent != shape.extent || w->layout.extent[0] != shape.extent[1] ||
      b->layout.extent[0] != shape.extent[1] || !shape.is_contiguous() ||
      !w->layout.is_contiguous() || !b->layout.is_contiguous() || !dst->layout.is_contiguous() ||
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
    if (src.get() != dst.get())
      impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_norm(impl_->owner->layer_norm_pipeline, p, src, w, b, dst);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::rms_norm_bf16(DeviceTensor& input, DeviceTensor& weight, DeviceTensor& output,
                                float epsilon) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_vae_norm) {
    throw std::runtime_error("vulkan tensor: exact BF16 RMSNorm is unavailable");
  }
  auto src = impl_->owner->require(input);
  auto w = impl_->owner->require(weight);
  auto dst = impl_->owner->require(output);
  const auto& shape = src->layout;
  const uint64_t rows = shape.extent[0];
  const uint64_t dim = shape.extent[1];
  if (!std::isnormal(epsilon) || epsilon <= 0.0f || src.get() == w.get() || dst.get() == w.get() ||
      src->type != ScalarType::kBFloat16 || w->type != ScalarType::kBFloat16 ||
      dst->type != ScalarType::kBFloat16 || shape.rank != 2 || w->layout.rank != 1 ||
      dst->layout.rank != 2 || dst->layout.extent != shape.extent || w->layout.extent[0] != dim ||
      !shape.is_contiguous() || !w->layout.is_contiguous() || !dst->layout.is_contiguous() ||
      rows > std::numeric_limits<uint32_t>::max() || dim > kMaxExactNormDimension ||
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
    if (src.get() != dst.get())
      impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_shared_rms(narrow ? impl_->owner->bf16_rms_narrow_pipeline
                                      : impl_->owner->bf16_rms_block_pipeline,
                               p, static_cast<uint32_t>(groups), src, w, dst);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::rms_norm_heads_bf16(DeviceTensor& input, DeviceTensor& weight,
                                      DeviceTensor& output, uint32_t heads, uint32_t head_dim,
                                      float epsilon) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_vae_norm) {
    throw std::runtime_error("vulkan tensor: exact BF16 head RMSNorm is unavailable");
  }
  auto src = impl_->owner->require(input);
  auto w = impl_->owner->require(weight);
  auto dst = impl_->owner->require(output);
  const auto& shape = src->layout;
  const bool rank2 = shape.rank == 2;
  const bool rank3 = shape.rank == 3;
  const uint64_t token_rows = (rank2 || rank3) ? shape.extent[0] : 0;
  const uint64_t rows = checked_multiply(token_rows, heads, "BF16 head RMSNorm");
  const uint64_t width = checked_multiply(heads, head_dim, "BF16 head RMSNorm");
  if (!std::isnormal(epsilon) || epsilon <= 0.0f || heads == 0 || head_dim == 0 ||
      src.get() == w.get() || dst.get() == w.get() || src->type != ScalarType::kBFloat16 ||
      w->type != ScalarType::kBFloat16 || dst->type != ScalarType::kBFloat16 ||
      (!rank2 && !rank3) ||
      (rank2 ? shape.extent[1] != width
             : (shape.extent[1] != heads || shape.extent[2] != head_dim)) ||
      w->layout.rank != 1 || w->layout.extent[0] != head_dim || dst->layout.rank != shape.rank ||
      dst->layout.extent != shape.extent || !shape.is_contiguous() || !w->layout.is_contiguous() ||
      !dst->layout.is_contiguous() || rows > std::numeric_limits<uint32_t>::max() ||
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
  p.rows = static_cast<uint32_t>(rows);
  p.dim = head_dim;
  std::memcpy(&p.epsilon_bits, &epsilon, sizeof(epsilon));
  try {
    impl_->count_operator();
    impl_->transition(src, src.get() == dst.get() ? BufferAccess::kComputeReadWrite
                                                  : BufferAccess::kComputeRead);
    impl_->transition(w, BufferAccess::kComputeRead);
    if (src.get() != dst.get())
      impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_shared_rms(narrow ? impl_->owner->bf16_rms_narrow_pipeline
                                      : impl_->owner->bf16_rms_block_pipeline,
                               p, static_cast<uint32_t>(groups), src, w, dst);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::layer_norm_bf16(DeviceTensor& input, DeviceTensor& weight, DeviceTensor& bias,
                                  DeviceTensor& output, float epsilon) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
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
  if (!std::isnormal(epsilon) || epsilon <= 0.0f || src.get() == w.get() || src.get() == b.get() ||
      w.get() == b.get() || dst.get() == w.get() || dst.get() == b.get() ||
      src->type != ScalarType::kBFloat16 || w->type != ScalarType::kBFloat16 ||
      b->type != ScalarType::kBFloat16 || dst->type != ScalarType::kBFloat16 || shape.rank != 2 ||
      w->layout.rank != 1 || b->layout.rank != 1 || dst->layout.rank != 2 ||
      dst->layout.extent != shape.extent || w->layout.extent[0] != dim ||
      b->layout.extent[0] != dim || !shape.is_contiguous() || !w->layout.is_contiguous() ||
      !b->layout.is_contiguous() || !dst->layout.is_contiguous() ||
      rows > std::numeric_limits<uint32_t>::max() || dim > kMaxExactNormDimension ||
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
    if (src.get() != dst.get())
      impl_->transition(dst, BufferAccess::kComputeWrite);
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
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
  impl_->record_shared_mod(false, input, weight, scale, shift, selectors, output, epsilon);
}

void TensorBatch::rms_norm_modulate_f32(DeviceTensor& input, DeviceTensor& weight,
                                        DeviceTensor& scale, DeviceTensor& shift,
                                        DeviceTensor& selectors, DeviceTensor& output,
                                        float epsilon) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
  impl_->record_shared_mod(true, input, weight, scale, shift, selectors, output, epsilon);
}

void TensorBatch::rms_norm_modulate_bf16_table(DeviceTensor& input, DeviceTensor& weight,
                                               DeviceTensor& tables, uint32_t mod_rows,
                                               uint32_t scale_table, uint32_t shift_table,
                                               DeviceTensor& selectors, DeviceTensor& output,
                                               float epsilon) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
  impl_->record_shared_mod_table(input, weight, tables, mod_rows, scale_table, shift_table,
                                 selectors, output, epsilon);
}

} // namespace slopfab::vulkan
