#include "tensor_recording.h"

namespace slopfab::vulkan {

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


}  // namespace slopfab::vulkan
