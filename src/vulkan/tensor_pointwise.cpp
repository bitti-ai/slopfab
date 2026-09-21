#include "tensor_recording.h"

namespace slopfab::vulkan {

void TensorBatch::copy(DeviceTensor& source, DeviceTensor& destination) {
  if (!impl_)
    throw std::logic_error("vulkan tensor: empty batch");
  if (impl_->poisoned)
    throw std::logic_error("vulkan tensor: batch is poisoned");
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

void TensorBatch::copy_rows(DeviceTensor& source, DeviceTensor& destination, uint32_t source_row,
                            uint32_t destination_row, uint32_t rows) {
  if (!impl_)
    throw std::logic_error("vulkan tensor: empty batch");
  if (impl_->poisoned)
    throw std::logic_error("vulkan tensor: batch is poisoned");
  auto src = impl_->owner->require(source);
  auto dst = impl_->owner->require(destination);
  const auto& a = src->layout;
  const auto& o = dst->layout;
  const uint64_t scalar_bytes =
      src->type == ScalarType::kFloat32 || src->type == ScalarType::kInt32      ? 4u
      : src->type == ScalarType::kFloat16 || src->type == ScalarType::kBFloat16 ? 2u
                                                                                : 1u;
  const uint64_t row_bytes = a.rank == 2 ? a.extent[1] * scalar_bytes : 0;
  if (src.get() == dst.get() || src->type != dst->type || a.rank != 2 || o.rank != 2 ||
      !a.is_contiguous() || !o.is_contiguous() || a.extent[1] != o.extent[1] || rows == 0 ||
      source_row > a.extent[0] || rows > a.extent[0] - source_row ||
      destination_row > o.extent[0] || rows > o.extent[0] - destination_row || row_bytes == 0 ||
      (row_bytes & 3u) != 0 || row_bytes > std::numeric_limits<uint64_t>::max() / rows) {
    throw std::invalid_argument("vulkan tensor: invalid row-range copy");
  }
  const uint64_t copy_bytes = row_bytes * rows;
  try {
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kTransferRead);
    impl_->transition(dst, BufferAccess::kTransferWrite);
    impl_->commands.copy_buffer(src->buffer, dst->buffer, copy_bytes, row_bytes * source_row,
                                row_bytes * destination_row);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::add(DeviceTensor& a, DeviceTensor& b, DeviceTensor& output) {
  if (!impl_)
    throw std::logic_error("vulkan tensor: empty batch");
  if (impl_->poisoned)
    throw std::logic_error("vulkan tensor: batch is poisoned");
  auto av = impl_->owner->require(a);
  auto bv = impl_->owner->require(b);
  auto out = impl_->owner->require(output);
  const uint64_t count = av->layout.elements();
  if (out.get() == av.get() || out.get() == bv.get() || av->type != ScalarType::kFloat32 ||
      bv->type != ScalarType::kFloat32 || out->type != ScalarType::kFloat32 ||
      count != bv->layout.elements() || count != out->layout.elements() || count == 0 ||
      count > std::numeric_limits<uint32_t>::max()) {
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
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
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
    op = 1;
    narrowing = true;
  } else if (src->type == ScalarType::kBFloat16 && dst->type == ScalarType::kFloat32) {
    op = 2;
  } else if (src->type == ScalarType::kFloat32 && dst->type == ScalarType::kFloat16) {
    op = 3;
    narrowing = true;
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
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::transpose_2d(DeviceTensor& source, DeviceTensor& destination) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
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
    p.op = 5;
    p.count = dispatch_count;
    p.p[0] = static_cast<uint32_t>(a.extent[0]);
    p.p[1] = static_cast<uint32_t>(a.extent[1]);
    impl_->dispatch(p, src, src, dst);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::gather_rows(DeviceTensor& source, DeviceTensor& indices,
                              DeviceTensor& destination) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(source);
  auto idx = impl_->owner->require(indices);
  auto dst = impl_->owner->require(destination);
  const auto& a = src->layout;
  const auto& ix = idx->layout;
  const auto& o = dst->layout;
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
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(idx, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters p;
    p.op = 6;
    p.count = dispatch_count;
    p.p[0] = static_cast<uint32_t>(a.extent[0]);
    p.p[1] = static_cast<uint32_t>(a.extent[1]);
    impl_->dispatch(p, src, idx, dst);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::scatter_rows(DeviceTensor& source, DeviceTensor& indices,
                               DeviceTensor& destination) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(source);
  auto idx = impl_->owner->require(indices);
  auto dst = impl_->owner->require(destination);
  const auto& a = src->layout;
  const auto& ix = idx->layout;
  const auto& o = dst->layout;
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
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(idx, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters p;
    p.op = 7;
    p.count = dispatch_count;
    p.p[0] = static_cast<uint32_t>(o.extent[0]);
    p.p[1] = static_cast<uint32_t>(a.extent[1]);
    impl_->dispatch(p, src, idx, dst);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::add_bias(DeviceTensor& input, DeviceTensor& bias, DeviceTensor& output) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(input);
  auto bv = impl_->owner->require(bias);
  auto dst = impl_->owner->require(output);
  const auto& a = src->layout;
  const auto& b = bv->layout;
  const auto& o = dst->layout;
  if (src.get() == bv.get() || dst.get() == bv.get() || src->type != ScalarType::kFloat32 ||
      bv->type != ScalarType::kFloat32 || dst->type != ScalarType::kFloat32 || a.rank != 2 ||
      b.rank != 1 || o.rank != 2 || b.extent[0] != a.extent[1] || o.extent != a.extent ||
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
    if (src.get() != dst.get())
      impl_->transition(dst, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters p;
    p.op = 8;
    p.count = dispatch_count;
    p.p[1] = static_cast<uint32_t>(a.extent[1]);
    impl_->dispatch(p, src, bv, dst);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::layer_scale_residual_f32(DeviceTensor& x, DeviceTensor& y, DeviceTensor& bias,
                                           DeviceTensor& scale) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
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
      bv->type != ScalarType::kFloat32 || sv->type != ScalarType::kFloat32 || shape.rank != 2 ||
      yv->layout.extent != shape.extent || bv->layout.rank != 1 || sv->layout.rank != 1 ||
      bv->layout.extent[0] != columns || sv->layout.extent[0] != columns ||
      !shape.is_contiguous() || !yv->layout.is_contiguous() || !bv->layout.is_contiguous() ||
      !sv->layout.is_contiguous() || shape.extent[0] == 0 || columns == 0 ||
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
    p.op = 0;
    p.count = dispatch_count;
    p.p[0] = static_cast<uint32_t>(columns);
    impl_->dispatch_vae_pointwise(VaePointwiseOperation::kResidual, p, {xv, yv, bv, sv});
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::swiglu_bias_f32(DeviceTensor& input, DeviceTensor& bias, DeviceTensor& output) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
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
      dst->type != ScalarType::kFloat32 || in_shape.rank != 2 || out_shape.rank != 2 ||
      bv->layout.rank != 1 || inner == 0 || !doubled_fits ||
      in_shape.extent[0] != out_shape.extent[0] || in_shape.extent[1] != 2 * inner ||
      bv->layout.extent[0] != 2 * inner || !in_shape.is_contiguous() ||
      !out_shape.is_contiguous() || !bv->layout.is_contiguous() || out_shape.extent[0] == 0 ||
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
    p.op = 1;
    p.count = dispatch_count;
    p.p[0] = static_cast<uint32_t>(inner);
    impl_->dispatch_vae_pointwise(VaePointwiseOperation::kSwiglu, p, {src, bv, bv, dst});
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::latent_denorm_f32(DeviceTensor& input, DeviceTensor& mean, DeviceTensor& std_dev,
                                    DeviceTensor& output) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
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
      sv->type != ScalarType::kFloat32 || dst->type != ScalarType::kFloat32 || shape.rank != 2 ||
      dst->layout.extent != shape.extent || mv->layout.rank != 1 || sv->layout.rank != 1 ||
      mv->layout.extent[0] != channels || sv->layout.extent[0] != channels ||
      !shape.is_contiguous() || !dst->layout.is_contiguous() || !mv->layout.is_contiguous() ||
      !sv->layout.is_contiguous() || channels == 0 || voxels == 0 ||
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
    p.op = 2;
    p.count = dispatch_count;
    p.p[0] = static_cast<uint32_t>(voxels);
    impl_->dispatch_vae_pointwise(VaePointwiseOperation::kDenorm, p, {src, mv, sv, dst});
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::heads_to_tokens_bf16(DeviceTensor& source, DeviceTensor& destination,
                                       uint32_t heads, uint32_t sequence, uint32_t head_dim) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(source);
  auto dst = impl_->owner->require(destination);
  const uint64_t elements = checked_multiply(checked_multiply(heads, sequence, "heads-to-tokens"),
                                             head_dim, "heads-to-tokens");
  if (heads == 0 || sequence == 0 || head_dim == 0 || src.get() == dst.get() ||
      src->type != ScalarType::kFloat32 || dst->type != ScalarType::kBFloat16 ||
      src->layout.elements() != elements || dst->layout.elements() != elements ||
      elements > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid heads-to-tokens conversion");
  }
  const uint32_t dispatch_count = impl_->owner->validate_dispatch((elements + 1) / 2);
  try {
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters p;
    p.op = 9;
    p.count = dispatch_count;
    p.p[0] = heads;
    p.p[1] = head_dim;
    p.p[2] = sequence;
    p.p[3] = static_cast<uint32_t>(elements);
    impl_->dispatch(p, src, src, dst);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::depth_to_space(DeviceTensor& source, DeviceTensor& destination, uint32_t time,
                                 uint32_t height, uint32_t width, uint32_t channels,
                                 uint32_t patch_time, uint32_t patch) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(source);
  auto dst = impl_->owner->require(destination);
  if (time == 0 || height == 0 || width == 0 || channels == 0 || patch_time == 0 || patch == 0) {
    throw std::invalid_argument("vulkan tensor: depth-to-space dimensions must be nonzero");
  }
  const uint64_t tokens =
      checked_multiply(checked_multiply(time, height, "depth-to-space"), width, "depth-to-space");
  const uint64_t patch_volume = checked_multiply(
      checked_multiply(patch_time, patch, "depth-to-space"), patch, "depth-to-space");
  const uint64_t elements = checked_multiply(checked_multiply(tokens, channels, "depth-to-space"),
                                             patch_volume, "depth-to-space");
  if (patch_volume > std::numeric_limits<uint32_t>::max() || src.get() == dst.get() ||
      src->type != ScalarType::kFloat32 || dst->type != ScalarType::kFloat32 ||
      src->layout.elements() != elements || dst->layout.elements() != elements ||
      elements > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid depth-to-space tensors");
  }
  const uint32_t dispatch_count = impl_->owner->validate_dispatch(elements);
  try {
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters p;
    p.op = 10;
    p.count = dispatch_count;
    p.p[0] = time;
    p.p[1] = height;
    p.p[2] = width;
    p.p[3] = patch_time;
    p.p[4] = patch;
    p.p[5] = channels;
    impl_->dispatch(p, src, src, dst);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

} // namespace slopfab::vulkan
