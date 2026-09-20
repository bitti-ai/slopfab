#include "tensor_recording.h"

namespace slopfab::vulkan {

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

void TensorBatch::reference_operation(DeviceTensor& input, DeviceTensor& weight,
    DeviceTensor& bias, DeviceTensor& output, const uint32_t* parameters,
    uint32_t groups, uint32_t batches, DeviceTensor* previous, DeviceTensor* earliest) {
  if (!impl_ || impl_->poisoned || !parameters || !groups || !batches)
    throw std::invalid_argument("Vulkan reference: invalid dispatch");
  if (!impl_->owner->reference_pipeline)
    throw std::logic_error("Vulkan reference: encoder shaders are not enabled in this context");
  std::array<std::shared_ptr<DeviceTensor::Impl>, 6> tensors = {
    impl_->owner->require(input), impl_->owner->require(weight),
    impl_->owner->require(bias), impl_->owner->require(output),
    impl_->owner->require(previous ? *previous : input),
    impl_->owner->require(earliest ? *earliest : input)};
  for (size_t i = 0; i < tensors.size(); ++i) {
    if (tensors[i]->type != ScalarType::kFloat32 || !tensors[i]->layout.is_contiguous() ||
        (i != 3 && tensors[i] == tensors[3]))
      throw std::invalid_argument("Vulkan reference: invalid tensor or output alias");
  }
  uint32_t p[16]; std::copy_n(parameters, 16, p);
  p[15] = groups;
  p[13] = std::min(groups, impl_->owner->max_dispatch_x);
  const uint32_t gy = (groups + p[13] - 1) / p[13];
  if (gy > impl_->owner->max_dispatch_y || batches > 2)
    throw std::out_of_range("Vulkan reference: dispatch capacity exceeded");
  try {
    impl_->count_operator();
    std::vector<StorageBinding> bindings;
    for (uint32_t i = 0; i < tensors.size(); ++i) {
      impl_->transition(tensors[i], i == 3 ? BufferAccess::kComputeWrite : BufferAccess::kComputeRead);
      bindings.push_back({i, &tensors[i]->buffer, 0, tensors[i]->buffer.size()});
    }
    impl_->commands.bind_compute(impl_->owner->reference_pipeline, bindings);
    impl_->commands.push_constants(p, sizeof(p));
    impl_->commands.dispatch(p[13], gy, batches);
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::keyframe_conv3d_f16(
    DeviceTensor& input, DeviceTensor& weight, DeviceTensor& bias,
    DeviceTensor& output, uint32_t in_channels, uint32_t out_channels,
    uint32_t input_height, uint32_t input_width, uint32_t kernel,
    uint32_t stride, bool reflect_padding, bool asymmetric_padding) {
  if (!impl_ || impl_->poisoned) {
    throw std::logic_error("vulkan tensor: invalid batch");
  }
  if (!impl_->owner->exact_vae_pointwise) {
    throw std::runtime_error(
        "vulkan keyframe: exact Conv3D is unavailable");
  }
  auto src = impl_->owner->require(input);
  auto w = impl_->owner->require(weight);
  auto b = impl_->owner->require(bias);
  auto dst = impl_->owner->require(output);
  const uint32_t output_height = stride == 2 ? input_height / 2 : input_height;
  const uint32_t output_width = stride == 2 ? input_width / 2 : input_width;
  uint64_t input_count = 0, output_count = 0, weight_count = 0;
  try {
    input_count = checked_multiply(
        in_channels, checked_multiply(input_height, input_width, "keyframe input"),
        "keyframe input");
    output_count = checked_multiply(
        out_channels,
        checked_multiply(output_height, output_width, "keyframe output"),
        "keyframe output");
    weight_count = checked_multiply(
        checked_multiply(out_channels, in_channels, "keyframe weight"),
        checked_multiply(kernel, checked_multiply(kernel, kernel, "keyframe weight"),
                         "keyframe weight"),
        "keyframe weight");
  } catch (const std::overflow_error&) {
    throw std::invalid_argument("vulkan keyframe: convolution shape overflow");
  }
  const bool shaped_input = src->layout.rank == 3 &&
      src->layout.extent[0] == in_channels &&
      src->layout.extent[1] == input_height &&
      src->layout.extent[2] == input_width;
  const bool valid_input = shaped_input ||
      (src->layout.rank == 1 && src->layout.elements() >= input_count);
  const bool valid_weight = w->layout.rank == 5 &&
      w->layout.extent[0] == out_channels &&
      w->layout.extent[1] == in_channels &&
      w->layout.extent[2] == kernel && w->layout.extent[3] == kernel &&
      w->layout.extent[4] == kernel;
  const bool shaped_output = dst->layout.rank == 3 &&
      dst->layout.extent[0] == out_channels &&
      dst->layout.extent[1] == output_height &&
      dst->layout.extent[2] == output_width;
  const bool valid_output = shaped_output ||
      (dst->layout.rank == 1 && dst->layout.elements() >= output_count);
  if (in_channels == 0 || out_channels == 0 || input_height == 0 ||
      input_width == 0 || (kernel != 1 && kernel != 3) ||
      (stride != 1 && stride != 2) ||
      (asymmetric_padding && (stride != 2 || reflect_padding)) ||
      output_height == 0 || output_width == 0 ||
      output_count > std::numeric_limits<uint32_t>::max() ||
      src.get() == w.get() || src.get() == b.get() || src.get() == dst.get() ||
      w.get() == b.get() || w.get() == dst.get() || b.get() == dst.get() ||
      src->type != ScalarType::kFloat32 || dst->type != ScalarType::kFloat32 ||
      w->type != ScalarType::kFloat16 || b->type != ScalarType::kFloat16 ||
      !valid_input || !valid_weight || !valid_output ||
      b->layout.rank != 1 || b->layout.extent[0] != out_channels ||
      src->layout.elements() < input_count ||
      w->layout.elements() != weight_count ||
      dst->layout.elements() < output_count ||
      !src->layout.is_contiguous() || !w->layout.is_contiguous() ||
      !b->layout.is_contiguous() || !dst->layout.is_contiguous()) {
    throw std::invalid_argument("vulkan keyframe: invalid Conv3D tensors");
  }
  TensorContext::Impl::KeyframeParameters p;
  p.in_channels = in_channels;
  p.out_channels = out_channels;
  p.input_height = input_height;
  p.input_width = input_width;
  p.output_height = output_height;
  p.output_width = output_width;
  p.kernel = kernel;
  p.stride = stride;
  p.reflect_padding = reflect_padding ? 1u : 0u;
  p.asymmetric_padding = asymmetric_padding ? 1u : 0u;
  p.count = static_cast<uint32_t>(output_count);
  try {
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(w, BufferAccess::kComputeRead);
    impl_->transition(b, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_keyframe(p, {src, w, b, dst});
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::keyframe_group_norm_silu_f16_affine(
    DeviceTensor& input, DeviceTensor& weight, DeviceTensor& bias,
    DeviceTensor& output, uint32_t channels, uint32_t height,
    uint32_t width, uint32_t groups, float epsilon) {
  if (!impl_ || impl_->poisoned) {
    throw std::logic_error("vulkan tensor: invalid batch");
  }
  if (!impl_->owner->exact_vae_norm) {
    throw std::runtime_error(
        "vulkan keyframe: exact GroupNorm+SiLU is unavailable");
  }
  auto src = impl_->owner->require(input);
  auto w = impl_->owner->require(weight);
  auto b = impl_->owner->require(bias);
  auto dst = impl_->owner->require(output);
  const uint64_t spatial = static_cast<uint64_t>(height) * width;
  const uint64_t count = static_cast<uint64_t>(channels) * spatial;
  const uint64_t group_count = groups == 0 ? 0 :
      static_cast<uint64_t>(channels / groups) * spatial;
  if (!std::isnormal(epsilon) || epsilon <= 0.0f || channels == 0 ||
      height == 0 || width == 0 || groups == 0 || channels % groups != 0 ||
      spatial > std::numeric_limits<uint32_t>::max() ||
      count > std::numeric_limits<uint32_t>::max() ||
      group_count > std::numeric_limits<uint32_t>::max() ||
      src.get() == dst.get() ||
      src.get() == w.get() || src.get() == b.get() || dst.get() == w.get() ||
      dst.get() == b.get() || w.get() == b.get() ||
      src->type != ScalarType::kFloat32 || dst->type != ScalarType::kFloat32 ||
      w->type != ScalarType::kFloat16 || b->type != ScalarType::kFloat16 ||
      src->layout.rank != 1 || dst->layout.rank != 1 ||
      src->layout.elements() < count || dst->layout.elements() < count ||
      w->layout.rank != 1 || b->layout.rank != 1 ||
      w->layout.extent[0] != channels || b->layout.extent[0] != channels ||
      !src->layout.is_contiguous() || !dst->layout.is_contiguous() ||
      !w->layout.is_contiguous() || !b->layout.is_contiguous()) {
    throw std::invalid_argument("vulkan keyframe: invalid flat GroupNorm+SiLU");
  }
  if (!detail::norm_dispatch_fits(groups, impl_->owner->max_dispatch_x)) {
    throw std::out_of_range(
        "vulkan keyframe: GroupNorm group count exceeds dispatch limits");
  }
  TensorContext::Impl::NormParameters p;
  p.rows = channels;
  p.dim = static_cast<uint32_t>(spatial);
  p.mod_rows = groups;
  std::memcpy(&p.epsilon_bits, &epsilon, sizeof(epsilon));
  try {
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(w, BufferAccess::kComputeRead);
    impl_->transition(b, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_group_norm(p, src, w, b, dst);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::keyframe_add_f32(DeviceTensor& a, DeviceTensor& b,
                                    DeviceTensor& output, uint64_t count) {
  if (!impl_ || impl_->poisoned) {
    throw std::logic_error("vulkan tensor: invalid batch");
  }
  auto av = impl_->owner->require(a);
  auto bv = impl_->owner->require(b);
  auto out = impl_->owner->require(output);
  if (count == 0 || count > std::numeric_limits<uint32_t>::max() ||
      av.get() == bv.get() || av.get() == out.get() || bv.get() == out.get() ||
      av->type != ScalarType::kFloat32 || bv->type != ScalarType::kFloat32 ||
      out->type != ScalarType::kFloat32 || av->layout.rank != 1 ||
      bv->layout.rank != 1 || out->layout.rank != 1 ||
      av->layout.elements() < count || bv->layout.elements() < count ||
      out->layout.elements() < count || !av->layout.is_contiguous() ||
      !bv->layout.is_contiguous() || !out->layout.is_contiguous()) {
    throw std::invalid_argument("vulkan keyframe: invalid flat residual add");
  }
  TensorContext::Impl::Parameters p;
  p.op = 0;
  p.count = impl_->owner->validate_dispatch(count);
  try {
    impl_->count_operator();
    impl_->transition(av, BufferAccess::kComputeRead);
    impl_->transition(bv, BufferAccess::kComputeRead);
    impl_->transition(out, BufferAccess::kComputeWrite);
    impl_->dispatch(p, av, bv, out);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}


}  // namespace slopfab::vulkan
