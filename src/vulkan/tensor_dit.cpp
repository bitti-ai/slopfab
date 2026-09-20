#include "tensor_recording.h"

namespace slopfab::vulkan {

void TensorBatch::dit_add_gated_bf16(DeviceTensor& residual,
                                     DeviceTensor& branch,
                                     DeviceTensor& gate,
                                     DeviceTensor& selectors) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_dit_pointwise) {
    throw std::runtime_error("vulkan DiT: exact pointwise operations are unavailable");
  }
  auto x = impl_->owner->require(residual);
  auto b = impl_->owner->require(branch);
  auto g = impl_->owner->require(gate);
  auto a = impl_->owner->require(selectors);
  const auto& shape = x->layout;
  const uint64_t rows = shape.rank == 2 ? shape.extent[0] : 0;
  const uint64_t dim = shape.rank == 2 ? shape.extent[1] : 0;
  const uint64_t mod_rows = g->layout.rank == 2 ? g->layout.extent[0] : 0;
  const uint64_t count = shape.elements();
  if (x.get() == b.get() || x.get() == g.get() || x.get() == a.get() ||
      b.get() == g.get() || b.get() == a.get() || g.get() == a.get() ||
      shape.rank != 2 || b->layout.rank != 2 || b->layout.extent != shape.extent ||
      g->layout.rank != 2 || g->layout.extent[1] != dim ||
      a->layout.rank != 1 || a->layout.extent[0] != rows ||
      x->type != ScalarType::kBFloat16 || b->type != ScalarType::kBFloat16 ||
      g->type != ScalarType::kFloat32 || a->type != ScalarType::kInt32 ||
      !shape.is_contiguous() || !b->layout.is_contiguous() ||
      !g->layout.is_contiguous() || !a->layout.is_contiguous() ||
      rows == 0 || dim == 0 || mod_rows == 0 ||
      rows > std::numeric_limits<uint32_t>::max() ||
      dim > std::numeric_limits<uint32_t>::max() ||
      mod_rows > std::numeric_limits<uint32_t>::max() ||
      count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan DiT: invalid gated residual tensors");
  }
  TensorContext::Impl::DitParameters p;
  p.op = 0; p.rows = static_cast<uint32_t>(rows);
  p.dim = static_cast<uint32_t>(dim);
  p.mod_rows = static_cast<uint32_t>(mod_rows);
  p.count = static_cast<uint32_t>(count);
  try {
    impl_->count_operator();
    impl_->transition(x, BufferAccess::kComputeReadWrite);
    impl_->transition(b, BufferAccess::kComputeRead);
    impl_->transition(g, BufferAccess::kComputeRead);
    impl_->transition(a, BufferAccess::kComputeRead);
    impl_->dispatch_dit(p, {x, b, g, a, x});
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::dit_add_gated_bf16_table(
    DeviceTensor& residual, DeviceTensor& branch, DeviceTensor& tables,
    uint32_t mod_rows, uint32_t gate_table, DeviceTensor& selectors) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_dit_pointwise)
    throw std::runtime_error("vulkan DiT: exact pointwise operations are unavailable");
  auto x=impl_->owner->require(residual), b=impl_->owner->require(branch);
  auto t=impl_->owner->require(tables), a=impl_->owner->require(selectors);
  const auto& shape=x->layout;const uint64_t rows=shape.rank==2?shape.extent[0]:0;
  const uint64_t dim=shape.rank==2?shape.extent[1]:0;
  const uint64_t table_count=6;
  const uint64_t stride=t->layout.rank==1?t->layout.extent[0]/table_count:0;
  const uint64_t logical=checked_multiply(mod_rows,dim,"DiT gate table");
  const uint64_t count=shape.elements();
  if(gate_table>=table_count||x.get()==b.get()||x.get()==t.get()||x.get()==a.get()||
     b.get()==t.get()||b.get()==a.get()||t.get()==a.get()||shape.rank!=2||
     b->layout.rank!=2||b->layout.extent!=shape.extent||t->layout.rank!=1||
     t->layout.extent[0]%table_count!=0||stride<logical||a->layout.rank!=1||a->layout.extent[0]!=rows||
     x->type!=ScalarType::kBFloat16||b->type!=ScalarType::kBFloat16||
     t->type!=ScalarType::kFloat32||a->type!=ScalarType::kInt32||
     !shape.is_contiguous()||!b->layout.is_contiguous()||!t->layout.is_contiguous()||
     !a->layout.is_contiguous()||rows==0||dim==0||mod_rows==0||rows>UINT32_MAX||
     dim>UINT32_MAX||count>UINT32_MAX)
    throw std::invalid_argument("vulkan DiT: invalid gated residual table tensors");
  TensorContext::Impl::DitParameters p;p.op=0;p.rows=static_cast<uint32_t>(rows);
  p.dim=static_cast<uint32_t>(dim);p.mod_rows=static_cast<uint32_t>(mod_rows);
  p.count=static_cast<uint32_t>(count);
  const uint64_t offset=stride*4*gate_table;
  try {impl_->count_operator();impl_->transition(x,BufferAccess::kComputeReadWrite);
    impl_->transition(b,BufferAccess::kComputeRead);impl_->transition(t,BufferAccess::kComputeRead);
    impl_->transition(a,BufferAccess::kComputeRead);impl_->dispatch_dit(p,{x,b,t,a,x},offset);
  } catch (...) {impl_->poisoned=true;throw;}
}

void TensorBatch::dit_swiglu_bf16(DeviceTensor& fused, DeviceTensor& output) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_dit_pointwise) {
    throw std::runtime_error("vulkan DiT: exact pointwise operations are unavailable");
  }
  auto src = impl_->owner->require(fused);
  auto dst = impl_->owner->require(output);
  const uint64_t rows = src->layout.rank == 2 ? src->layout.extent[0] : 0;
  const uint64_t doubled = src->layout.rank == 2 ? src->layout.extent[1] : 0;
  const uint64_t inner = doubled / 2;
  const uint64_t count = checked_multiply(rows, inner, "DiT SwiGLU");
  if (src.get() == dst.get() || src->layout.rank != 2 ||
      (doubled & 1u) != 0 || rows == 0 || inner == 0 ||
      dst->layout.rank != 2 || dst->layout.extent[0] != rows ||
      dst->layout.extent[1] != inner ||
      src->type != ScalarType::kBFloat16 || dst->type != ScalarType::kBFloat16 ||
      !src->layout.is_contiguous() || !dst->layout.is_contiguous() ||
      rows > std::numeric_limits<uint32_t>::max() ||
      inner > std::numeric_limits<uint32_t>::max() ||
      count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan DiT: invalid SwiGLU tensors");
  }
  TensorContext::Impl::DitParameters p;
  p.op = 1; p.rows = static_cast<uint32_t>(rows);
  p.dim = static_cast<uint32_t>(inner);
  p.count = static_cast<uint32_t>(count);
  try {
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_dit(p, {src, src, src, src, dst});
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::dit_expand_adaln(DeviceTensor& weight, DeviceTensor& bias,
                                   DeviceTensor& code, DeviceTensor& output,
                                   uint32_t num_modality, uint32_t num_param,
                                   uint32_t channels) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_dit_pointwise) {
    throw std::runtime_error("vulkan DiT: exact pointwise operations are unavailable");
  }
  auto w = impl_->owner->require(weight); auto b = impl_->owner->require(bias);
  auto c = impl_->owner->require(code); auto dst = impl_->owner->require(output);
  const uint64_t timesteps = c->layout.rank == 2 ? c->layout.extent[0] : 0;
  const uint64_t rank = c->layout.rank == 2 ? c->layout.extent[1] : 0;
  const uint64_t features = checked_multiply(
      checked_multiply(num_modality, num_param, "DiT AdaLN"), channels,
      "DiT AdaLN");
  const uint64_t count = checked_multiply(timesteps, features, "DiT AdaLN");
  const uint64_t table_elements = checked_multiply(
      checked_multiply(timesteps, num_modality, "DiT AdaLN table"), channels,
      "DiT AdaLN table");
  const bool canonical_output = dst->layout.rank == 3 &&
      dst->layout.extent[0] == num_param &&
      dst->layout.extent[1] == timesteps * num_modality &&
      dst->layout.extent[2] == channels;
  const bool single_table_output = num_param == 1 && dst->layout.rank == 2 &&
      dst->layout.extent[0] == timesteps * num_modality &&
      dst->layout.extent[1] == channels;
  const uint64_t flat_stride = dst->layout.rank == 1 && num_param != 0 &&
      dst->layout.extent[0] % num_param == 0
          ? dst->layout.extent[0] / num_param : 0;
  const bool flat_output = flat_stride >= table_elements &&
      (flat_stride * sizeof(float)) % impl_->owner->storage_binding_alignment == 0;
  if (w.get() == b.get() || w.get() == c.get() || w.get() == dst.get() ||
      b.get() == c.get() || b.get() == dst.get() || c.get() == dst.get() ||
      num_modality == 0 || num_param == 0 || channels == 0 ||
      w->layout.rank != 2 || w->layout.extent[0] != features ||
      w->layout.extent[1] != rank || b->layout.rank != 1 ||
      b->layout.extent[0] != features || c->layout.rank != 2 ||
      timesteps == 0 || rank == 0 ||
      (!canonical_output && !single_table_output && !flat_output) ||
      w->type != ScalarType::kFloat32 || b->type != ScalarType::kFloat32 ||
      c->type != ScalarType::kFloat32 || dst->type != ScalarType::kFloat32 ||
      !w->layout.is_contiguous() || !b->layout.is_contiguous() ||
      !c->layout.is_contiguous() || !dst->layout.is_contiguous() ||
      timesteps > std::numeric_limits<uint32_t>::max() ||
      rank > std::numeric_limits<uint32_t>::max() ||
      count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan DiT: invalid AdaLN tensors");
  }
  TensorContext::Impl::DitParameters p;
  p.op = 2; p.rows = static_cast<uint32_t>(timesteps); p.dim = channels;
  p.num_t = static_cast<uint32_t>(timesteps); p.num_modality = num_modality;
  p.num_param = num_param; p.rank = static_cast<uint32_t>(rank);
  p.count = static_cast<uint32_t>(count);
  p.unused[0] = flat_output ? static_cast<uint32_t>(flat_stride) : 0;
  try {
    impl_->count_operator(); impl_->transition(w, BufferAccess::kComputeRead);
    impl_->transition(b, BufferAccess::kComputeRead);
    impl_->transition(c, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_dit(p, {w, b, c, c, dst});
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::dit_euler_step_f32(DeviceTensor& sample,
                                     DeviceTensor& velocity,
                                     float sigma_from_timestep, float ratio) {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_dit_pointwise)
    throw std::runtime_error("vulkan tensor: exact DiT Euler is unavailable");
  auto x = impl_->owner->require(sample);
  auto v = impl_->owner->require(velocity);
  const auto& layout = x->layout;
  if (x.get() == v.get() || x->type != ScalarType::kFloat32 ||
      v->type != ScalarType::kFloat32 || v->layout.extent != layout.extent ||
      v->layout.rank != layout.rank || !layout.is_contiguous() ||
      !v->layout.is_contiguous() || layout.elements() == 0 ||
      layout.elements() > UINT32_MAX || !std::isfinite(sigma_from_timestep) ||
      sigma_from_timestep < 0.0f || sigma_from_timestep > 1.0f ||
      !std::isfinite(ratio) || ratio < 0.0f || ratio > 1.0f)
    throw std::invalid_argument("vulkan tensor: invalid exact DiT Euler inputs");
  TensorContext::Impl::DitParameters p;
  p.op = 3;
  p.count = static_cast<uint32_t>(layout.elements());
  std::memcpy(&p.unused[0], &sigma_from_timestep, sizeof(float));
  std::memcpy(&p.unused[1], &ratio, sizeof(float));
  try {
    impl_->count_operator();
    impl_->transition(x, BufferAccess::kComputeReadWrite);
    impl_->transition(v, BufferAccess::kComputeRead);
    impl_->dispatch_dit(p, {x, v, x, x, x});
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}


}  // namespace slopfab::vulkan
