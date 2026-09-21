#include "tensor_recording.h"

namespace slopfab::vulkan {

void TensorBatch::text_add_residual_bf16(DeviceTensor& residual, DeviceTensor& branch) {
  if (!impl_ || impl_->poisoned) {
    throw std::logic_error("vulkan tensor: invalid batch");
  }
  if (!impl_->owner->exact_dit_pointwise) {
    throw std::runtime_error("vulkan text: exact pointwise operations are unavailable");
  }
  auto x = impl_->owner->require(residual);
  auto b = impl_->owner->require(branch);
  const uint64_t rows = x->layout.rank == 2 || x->layout.rank == 3 ? x->layout.extent[0] : 0;
  const uint64_t dim =
      x->layout.rank == 2 ? x->layout.extent[1]
      : x->layout.rank == 3
          ? checked_multiply(x->layout.extent[1], x->layout.extent[2], "residual heads")
          : 0;
  const uint64_t count = checked_multiply(rows, dim, "text residual");
  const uint64_t packed = count == 0 ? 0 : 1 + (count - 1) / 2;
  if (x.get() == b.get() || rows == 0 || dim == 0 || (b->layout.rank != 2 && b->layout.rank != 3) ||
      b->layout.extent[0] != rows || b->layout.elements() != count ||
      x->type != ScalarType::kBFloat16 || b->type != ScalarType::kBFloat16 ||
      !x->layout.is_contiguous() || !b->layout.is_contiguous() ||
      rows > std::numeric_limits<uint32_t>::max() || dim > std::numeric_limits<uint32_t>::max() ||
      count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan text: invalid residual tensors");
  }
  TensorContext::Impl::DitParameters p;
  p.op = 4;
  p.rows = static_cast<uint32_t>(rows);
  p.dim = static_cast<uint32_t>(dim);
  p.count = static_cast<uint32_t>(packed);
  try {
    impl_->count_operator();
    impl_->transition(x, BufferAccess::kComputeReadWrite);
    impl_->transition(b, BufferAccess::kComputeRead);
    impl_->dispatch_dit(p, {x, b, b, b, x});
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::text_swiglu_split_bf16(DeviceTensor& gate, DeviceTensor& up,
                                         DeviceTensor& output) {
  if (!impl_ || impl_->poisoned) {
    throw std::logic_error("vulkan tensor: invalid batch");
  }
  if (!impl_->owner->exact_dit_pointwise) {
    throw std::runtime_error("vulkan text: exact pointwise operations are unavailable");
  }
  auto g = impl_->owner->require(gate);
  auto u = impl_->owner->require(up);
  auto dst = impl_->owner->require(output);
  const uint64_t rows = g->layout.rank == 2 ? g->layout.extent[0] : 0;
  const uint64_t dim = g->layout.rank == 2 ? g->layout.extent[1] : 0;
  const uint64_t count = checked_multiply(rows, dim, "text SwiGLU");
  const uint64_t packed = count == 0 ? 0 : 1 + (count - 1) / 2;
  if (g.get() == u.get() || g.get() == dst.get() || u.get() == dst.get() || g->layout.rank != 2 ||
      rows == 0 || dim == 0 || u->layout.rank != 2 || u->layout.extent != g->layout.extent ||
      dst->layout.rank != 2 || dst->layout.extent != g->layout.extent ||
      g->type != ScalarType::kBFloat16 || u->type != ScalarType::kBFloat16 ||
      dst->type != ScalarType::kBFloat16 || !g->layout.is_contiguous() ||
      !u->layout.is_contiguous() || !dst->layout.is_contiguous() ||
      rows > std::numeric_limits<uint32_t>::max() || dim > std::numeric_limits<uint32_t>::max() ||
      count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan text: invalid split SwiGLU tensors");
  }
  TensorContext::Impl::DitParameters p;
  p.op = 5;
  p.rows = static_cast<uint32_t>(rows);
  p.dim = static_cast<uint32_t>(dim);
  p.count = static_cast<uint32_t>(packed);
  try {
    impl_->count_operator();
    impl_->transition(g, BufferAccess::kComputeRead);
    impl_->transition(u, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_dit(p, {g, u, u, u, dst});
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::vision_gelu_tanh_bf16(DeviceTensor& activation) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_dit_pointwise)
    throw std::runtime_error("vulkan vision: exact GELU is unavailable");
  auto x = impl_->owner->require(activation);
  const uint64_t rows = x->layout.rank == 2 ? x->layout.extent[0] : 0;
  const uint64_t dim = x->layout.rank == 2 ? x->layout.extent[1] : 0;
  const uint64_t count = checked_multiply(rows, dim, "vision GELU");
  const uint64_t packed = count == 0 ? 0 : 1 + (count - 1) / 2;
  if (x->layout.rank != 2 || rows == 0 || dim == 0 || x->type != ScalarType::kBFloat16 ||
      !x->layout.is_contiguous() || rows > std::numeric_limits<uint32_t>::max() ||
      dim > std::numeric_limits<uint32_t>::max() || count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan vision: invalid GELU tensor");
  }
  TensorContext::Impl::DitParameters p;
  p.op = 6;
  p.rows = static_cast<uint32_t>(rows);
  p.dim = static_cast<uint32_t>(dim);
  p.count = static_cast<uint32_t>(packed);
  try {
    impl_->count_operator();
    impl_->transition(x, BufferAccess::kComputeReadWrite);
    impl_->dispatch_dit(p, {x, x, x, x, x});
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::vision_add_positions_bf16(DeviceTensor& activation, DeviceTensor& position_table,
                                            DeviceTensor& position_index) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
  auto x = impl_->owner->require(activation);
  auto table = impl_->owner->require(position_table);
  auto index = impl_->owner->require(position_index);
  const uint64_t rows = x->layout.rank == 2 ? x->layout.extent[0] : 0;
  const uint64_t dim = x->layout.rank == 2 ? x->layout.extent[1] : 0;
  const uint64_t count = checked_multiply(rows, dim, "vision position add");
  const uint64_t packed = count == 0 ? 0 : 1 + (count - 1) / 2;
  if (x.get() == table.get() || x.get() == index.get() || table.get() == index.get() ||
      x->layout.rank != 2 || rows == 0 || dim == 0 || table->layout.rank != 2 ||
      table->layout.extent[1] != dim || table->layout.extent[0] == 0 || index->layout.rank != 1 ||
      index->layout.extent[0] != rows || x->type != ScalarType::kBFloat16 ||
      table->type != ScalarType::kBFloat16 || index->type != ScalarType::kInt32 ||
      !x->layout.is_contiguous() || !table->layout.is_contiguous() ||
      !index->layout.is_contiguous() || rows > UINT32_MAX || dim > UINT32_MAX ||
      table->layout.extent[0] > UINT32_MAX || count > UINT32_MAX)
    throw std::invalid_argument("vulkan vision: invalid position tensors");
  TensorContext::Impl::DitParameters p;
  p.op = 7;
  p.rows = static_cast<uint32_t>(rows);
  p.dim = static_cast<uint32_t>(dim);
  p.mod_rows = static_cast<uint32_t>(table->layout.extent[0]);
  p.count = static_cast<uint32_t>(packed);
  try {
    impl_->count_operator();
    impl_->transition(x, BufferAccess::kComputeReadWrite);
    impl_->transition(table, BufferAccess::kComputeRead);
    impl_->transition(index, BufferAccess::kComputeRead);
    impl_->dispatch_dit(p, {x, table, index, table, x});
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::vision_split_qkv_bf16(DeviceTensor& fused, DeviceTensor& query, DeviceTensor& key,
                                        DeviceTensor& value) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(fused);
  auto q = impl_->owner->require(query), k = impl_->owner->require(key);
  auto v = impl_->owner->require(value);
  const uint64_t rows = src->layout.rank == 2 ? src->layout.extent[0] : 0;
  const uint64_t fused_dim = src->layout.rank == 2 ? src->layout.extent[1] : 0;
  const uint64_t dim = fused_dim / 3;
  const uint64_t count = checked_multiply(rows, dim, "vision QKV split");
  const uint64_t packed = count == 0 ? 0 : 1 + (count - 1) / 2;
  const auto output_valid = [&](const std::shared_ptr<DeviceTensor::Impl>& out) {
    const bool flat =
        out->layout.rank == 2 && out->layout.extent[0] == rows && out->layout.extent[1] == dim;
    const bool headed =
        out->layout.rank == 3 && out->layout.extent[0] == rows && out->layout.extent[1] != 0 &&
        out->layout.extent[2] == dim / out->layout.extent[1] && dim % out->layout.extent[1] == 0;
    return (flat || headed) && out->type == ScalarType::kBFloat16 && out->layout.is_contiguous();
  };
  if (src.get() == q.get() || src.get() == k.get() || src.get() == v.get() || q.get() == k.get() ||
      q.get() == v.get() || k.get() == v.get() || src->layout.rank != 2 || rows == 0 || dim == 0 ||
      fused_dim % 3 != 0 || src->type != ScalarType::kBFloat16 || !src->layout.is_contiguous() ||
      !output_valid(q) || !output_valid(k) || !output_valid(v) || rows > UINT32_MAX ||
      dim > UINT32_MAX || count > UINT32_MAX || impl_->owner->max_batch_operators < 3 ||
      impl_->operator_count > impl_->owner->max_batch_operators - 3)
    throw std::invalid_argument("vulkan vision: invalid QKV split tensors");
  TensorContext::Impl::DitParameters p;
  p.op = 8;
  p.rows = static_cast<uint32_t>(rows);
  p.dim = static_cast<uint32_t>(dim);
  p.count = static_cast<uint32_t>(packed);
  try {
    impl_->transition(src, BufferAccess::kComputeRead);
    std::array<std::shared_ptr<DeviceTensor::Impl>, 3> outputs = {q, k, v};
    for (uint32_t part = 0; part < 3; ++part) {
      p.unused[0] = part;
      impl_->count_operator();
      impl_->transition(outputs[part], BufferAccess::kComputeWrite);
      impl_->dispatch_dit(p, {src, src, src, src, outputs[part]});
    }
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::vision_merge_four_bf16(DeviceTensor& input, DeviceTensor& output) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(input), dst = impl_->owner->require(output);
  const uint64_t rows = src->layout.rank == 2 ? src->layout.extent[0] : 0;
  const uint64_t dim = src->layout.rank == 2 ? src->layout.extent[1] : 0;
  const uint64_t count = checked_multiply(rows, dim, "vision merge four");
  const bool merged_dim_overflows = dim > std::numeric_limits<uint64_t>::max() / 4;
  if (src.get() == dst.get() || src->layout.rank != 2 || rows == 0 || (rows & 3u) != 0 ||
      dim == 0 || dst->layout.rank != 2 || merged_dim_overflows ||
      dst->layout.extent[0] != rows / 4 || dst->layout.extent[1] != dim * 4 ||
      src->type != ScalarType::kBFloat16 || dst->type != ScalarType::kBFloat16 ||
      !src->layout.is_contiguous() || !dst->layout.is_contiguous() || rows > UINT32_MAX ||
      dim > UINT32_MAX || count > UINT32_MAX)
    throw std::invalid_argument("vulkan vision: invalid merge-four tensors");
  TensorContext::Impl::DitParameters p;
  p.op = 9;
  p.rows = static_cast<uint32_t>(rows);
  p.dim = static_cast<uint32_t>(dim);
  p.count = static_cast<uint32_t>(count / 2);
  try {
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_dit(p, {src, src, src, src, dst});
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::vision_scatter_add_bf16(DeviceTensor& source, DeviceTensor& destination,
                                          DeviceTensor& row_index) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(source), dst = impl_->owner->require(destination);
  auto index = impl_->owner->require(row_index);
  const uint64_t rows = src->layout.rank == 2 ? src->layout.extent[0] : 0;
  const uint64_t dim = src->layout.rank == 2 ? src->layout.extent[1] : 0;
  const uint64_t count = checked_multiply(rows, dim, "vision scatter add");
  if (src.get() == dst.get() || src.get() == index.get() || dst.get() == index.get() ||
      src->layout.rank != 2 || rows == 0 || dim == 0 || dst->layout.rank != 2 ||
      dst->layout.extent[0] < rows || dst->layout.extent[1] != dim || index->layout.rank != 1 ||
      index->layout.extent[0] != rows || src->type != ScalarType::kBFloat16 ||
      dst->type != ScalarType::kBFloat16 || index->type != ScalarType::kInt32 ||
      !src->layout.is_contiguous() || !dst->layout.is_contiguous() ||
      !index->layout.is_contiguous() || rows > UINT32_MAX || dim > UINT32_MAX ||
      dst->layout.extent[0] > UINT32_MAX || count > UINT32_MAX)
    throw std::invalid_argument("vulkan vision: invalid scatter-add tensors");
  TensorContext::Impl::DitParameters p;
  p.op = 10;
  p.rows = static_cast<uint32_t>(rows);
  p.dim = static_cast<uint32_t>(dim);
  p.mod_rows = static_cast<uint32_t>(dst->layout.extent[0]);
  p.count = static_cast<uint32_t>(count);
  try {
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeReadWrite);
    impl_->transition(index, BufferAccess::kComputeRead);
    impl_->dispatch_dit(p, {dst, src, index, src, dst});
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::vision_scatter_bf16(DeviceTensor& source, DeviceTensor& destination,
                                      DeviceTensor& row_index) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(source), dst = impl_->owner->require(destination);
  auto index = impl_->owner->require(row_index);
  const uint64_t rows = src->layout.rank == 2 ? src->layout.extent[0] : 0;
  const uint64_t dim = src->layout.rank == 2 ? src->layout.extent[1] : 0;
  const uint64_t count = checked_multiply(rows, dim, "vision scatter");
  if (src.get() == dst.get() || src.get() == index.get() || dst.get() == index.get() ||
      src->layout.rank != 2 || rows == 0 || dim == 0 || dst->layout.rank != 2 ||
      dst->layout.extent[0] < rows || dst->layout.extent[1] != dim || index->layout.rank != 1 ||
      index->layout.extent[0] != rows || src->type != ScalarType::kBFloat16 ||
      dst->type != ScalarType::kBFloat16 || index->type != ScalarType::kInt32 ||
      !src->layout.is_contiguous() || !dst->layout.is_contiguous() ||
      !index->layout.is_contiguous() || rows > UINT32_MAX || dim > UINT32_MAX ||
      dst->layout.extent[0] > UINT32_MAX || count > UINT32_MAX)
    throw std::invalid_argument("vulkan vision: invalid scatter tensors");
  TensorContext::Impl::DitParameters p;
  p.op = 11;
  p.rows = static_cast<uint32_t>(rows);
  p.dim = static_cast<uint32_t>(dim);
  p.mod_rows = static_cast<uint32_t>(dst->layout.extent[0]);
  p.count = static_cast<uint32_t>(count);
  try {
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->transition(index, BufferAccess::kComputeRead);
    impl_->dispatch_dit(p, {dst, src, index, src, dst});
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

} // namespace slopfab::vulkan
