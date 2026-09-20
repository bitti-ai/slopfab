#pragma once
#include "tensor_context.h"

// Private transactional recording interface shared by operator domains.
namespace slopfab::vulkan {
struct TensorBatch::Impl {
  struct AccessSnapshot {
    std::shared_ptr<DeviceTensor::Impl> tensor;
    bool has_access = false;
    BufferAccess access = BufferAccess::kTransferWrite;
  };

  std::shared_ptr<TensorContext::Impl> owner;
  uintptr_t batch_id = 0;
  CommandList commands;
  TensorContext::Impl::RecorderLease recording_lease;
  std::vector<AccessSnapshot> snapshots;
  uint32_t snapshot_count = 0;
  uint32_t operator_count = 0;
  bool submitted = false;
  bool poisoned = false;

  void transition(const std::shared_ptr<DeviceTensor::Impl>& tensor, BufferAccess next) {
    bool seen = false;
    for (uint32_t i = 0; i < snapshot_count; ++i) {
      if (snapshots[i].tensor.get() == tensor.get()) {
        seen = true;
        break;
      }
    }
    if (!seen) {
      if (snapshot_count == snapshots.size()) {
        throw std::logic_error("vulkan tensor: batch transition capacity exceeded");
      }
      snapshots[snapshot_count++] = {tensor, tensor->has_access, tensor->access};
    }
    const bool next_writes = next == BufferAccess::kTransferWrite ||
                             next == BufferAccess::kComputeWrite ||
                             next == BufferAccess::kComputeReadWrite;
    if (tensor->has_access && (tensor->access != next || next_writes)) {
      commands.barrier(tensor->buffer, tensor->access, next);
    }
    tensor->has_access = true;
    tensor->access = next;
  }

  void count_operator() {
    if (operator_count == owner->max_batch_operators) {
      throw std::logic_error("vulkan tensor: batch operator limit exceeded");
    }
    ++operator_count;
  }

  void dispatch(const TensorContext::Impl::Parameters& parameters,
                const std::shared_ptr<DeviceTensor::Impl>& a,
                const std::shared_ptr<DeviceTensor::Impl>& b,
                const std::shared_ptr<DeviceTensor::Impl>& output) {
    // Every caller validates this before it mutates access tracking. Widen the
    // addition anyway so the invariant remains safe if uint32 max is used.
    const uint32_t groups =
        static_cast<uint32_t>((static_cast<uint64_t>(parameters.count) + 63ull) / 64ull);
    owner->ops_bindings[0].buffer = &a->buffer;
    owner->ops_bindings[0].bytes = a->buffer.size();
    owner->ops_bindings[1].buffer = &b->buffer;
    owner->ops_bindings[1].bytes = b->buffer.size();
    owner->ops_bindings[2].buffer = &output->buffer;
    owner->ops_bindings[2].bytes = output->buffer.size();
    commands.bind_compute(owner->ops_pipeline, owner->ops_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups);
  }

  void dispatch_vae_pointwise(
      VaePointwiseOperation operation,
      const TensorContext::Impl::Parameters& parameters,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 4>& resources) {
    const uint32_t groups = static_cast<uint32_t>(
        (static_cast<uint64_t>(parameters.count) + 63ull) / 64ull);
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->vae_pointwise_bindings[i].buffer = &resources[i]->buffer;
      owner->vae_pointwise_bindings[i].bytes = resources[i]->buffer.size();
    }
    ComputePipeline* pipeline = nullptr;
    switch (operation) {
      case VaePointwiseOperation::kResidual: pipeline = &owner->vae_residual_pipeline; break;
      case VaePointwiseOperation::kSwiglu: pipeline = &owner->vae_swiglu_pipeline; break;
      case VaePointwiseOperation::kDenorm: pipeline = &owner->vae_denorm_pipeline; break;
    }
    if (pipeline == nullptr) throw std::logic_error("vulkan tensor: invalid VAE pointwise op");
    commands.bind_compute(*pipeline, owner->vae_pointwise_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups);
  }

  void dispatch_rope(const TensorContext::Impl::Parameters& parameters,
                     uint32_t groups,
                     const std::shared_ptr<DeviceTensor::Impl>& input,
                     const std::shared_ptr<DeviceTensor::Impl>& cosine,
                     const std::shared_ptr<DeviceTensor::Impl>& sine) {
    owner->ops_bindings[0].buffer = &input->buffer;
    owner->ops_bindings[0].bytes = input->buffer.size();
    owner->ops_bindings[1].buffer = &cosine->buffer;
    owner->ops_bindings[1].bytes = cosine->buffer.size();
    owner->ops_bindings[2].buffer = &sine->buffer;
    owner->ops_bindings[2].bytes = sine->buffer.size();
    commands.bind_compute(owner->rope_pipeline, owner->ops_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups);
  }

  void dispatch_norm(ComputePipeline& pipeline,
                     const TensorContext::Impl::NormParameters& parameters,
                     const std::shared_ptr<DeviceTensor::Impl>& input,
                     const std::shared_ptr<DeviceTensor::Impl>& weight,
                     const std::shared_ptr<DeviceTensor::Impl>& bias,
                     const std::shared_ptr<DeviceTensor::Impl>& output) {
    owner->norm_bindings[0].buffer = &input->buffer;
    owner->norm_bindings[0].bytes = input->buffer.size();
    owner->norm_bindings[1].buffer = &weight->buffer;
    owner->norm_bindings[1].bytes = weight->buffer.size();
    owner->norm_bindings[2].buffer = &bias->buffer;
    owner->norm_bindings[2].bytes = bias->buffer.size();
    owner->norm_bindings[3].buffer = &output->buffer;
    owner->norm_bindings[3].bytes = output->buffer.size();
    commands.bind_compute(pipeline, owner->norm_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(parameters.rows);
  }

  void dispatch_shared_rms(ComputePipeline& pipeline,
                           const TensorContext::Impl::NormParameters& parameters,
                           uint32_t groups,
                           const std::shared_ptr<DeviceTensor::Impl>& input,
                           const std::shared_ptr<DeviceTensor::Impl>& weight,
                           const std::shared_ptr<DeviceTensor::Impl>& output) {
    owner->ops_bindings[0].buffer = &input->buffer;
    owner->ops_bindings[0].bytes = input->buffer.size();
    owner->ops_bindings[1].buffer = &weight->buffer;
    owner->ops_bindings[1].bytes = weight->buffer.size();
    owner->ops_bindings[2].buffer = &output->buffer;
    owner->ops_bindings[2].bytes = output->buffer.size();
    commands.bind_compute(pipeline, owner->ops_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups);
  }

  void dispatch_shared_layer(ComputePipeline& pipeline,
                             const TensorContext::Impl::NormParameters& parameters,
                             const std::shared_ptr<DeviceTensor::Impl>& input,
                             const std::shared_ptr<DeviceTensor::Impl>& weight,
                             const std::shared_ptr<DeviceTensor::Impl>& bias,
                             const std::shared_ptr<DeviceTensor::Impl>& output) {
    owner->norm_bindings[0].buffer = &input->buffer;
    owner->norm_bindings[0].bytes = input->buffer.size();
    owner->norm_bindings[1].buffer = &weight->buffer;
    owner->norm_bindings[1].bytes = weight->buffer.size();
    owner->norm_bindings[2].buffer = &bias->buffer;
    owner->norm_bindings[2].bytes = bias->buffer.size();
    owner->norm_bindings[3].buffer = &output->buffer;
    owner->norm_bindings[3].bytes = output->buffer.size();
    commands.bind_compute(pipeline, owner->norm_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(parameters.rows);
  }

  void dispatch_group_norm(const TensorContext::Impl::NormParameters& parameters,
                           const std::shared_ptr<DeviceTensor::Impl>& input,
                           const std::shared_ptr<DeviceTensor::Impl>& weight,
                           const std::shared_ptr<DeviceTensor::Impl>& bias,
                           const std::shared_ptr<DeviceTensor::Impl>& output) {
    owner->norm_bindings[0].buffer = &input->buffer;
    owner->norm_bindings[0].bytes = input->buffer.size();
    owner->norm_bindings[1].buffer = &weight->buffer;
    owner->norm_bindings[1].bytes = weight->buffer.size();
    owner->norm_bindings[2].buffer = &bias->buffer;
    owner->norm_bindings[2].bytes = bias->buffer.size();
    owner->norm_bindings[3].buffer = &output->buffer;
    owner->norm_bindings[3].bytes = output->buffer.size();
    commands.bind_compute(owner->group_norm_pipeline, owner->norm_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(parameters.mod_rows);
  }

  void dispatch_shared_mod(ComputePipeline& pipeline,
                           const TensorContext::Impl::NormParameters& parameters,
                           const std::array<std::shared_ptr<DeviceTensor::Impl>, 6>& resources,
                           uint64_t scale_offset = 0,
                           uint64_t shift_offset = 0) {
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->mod_bindings[i].buffer = &resources[i]->buffer;
      owner->mod_bindings[i].offset = 0;
      owner->mod_bindings[i].bytes = resources[i]->buffer.size();
    }
    owner->mod_bindings[2].offset = scale_offset;
    owner->mod_bindings[2].bytes -= scale_offset;
    owner->mod_bindings[3].offset = shift_offset;
    owner->mod_bindings[3].bytes -= shift_offset;
    commands.bind_compute(pipeline, owner->mod_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(parameters.rows);
  }

  void dispatch_vae_rope(
      const TensorContext::Impl::VaeRopeParameters& parameters,
      uint32_t groups,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 7>& resources) {
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->vae_rope_bindings[i].buffer = &resources[i]->buffer;
      owner->vae_rope_bindings[i].bytes = resources[i]->buffer.size();
    }
    commands.bind_compute(owner->vae_rope_pipeline, owner->vae_rope_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups);
  }

  void dispatch_audio(
      TensorContext::Impl::AudioParameters parameters,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 5>& resources) {
    const uint64_t total_groups =
        (static_cast<uint64_t>(parameters.count) + 63ull) / 64ull;
    const uint32_t groups_x = static_cast<uint32_t>(
        std::min<uint64_t>(total_groups, owner->max_dispatch_x));
    if (groups_x == 0) {
      throw std::out_of_range("vulkan audio: device exposes no X dispatch capacity");
    }
    const uint64_t groups_y_wide =
        (total_groups + groups_x - 1ull) / groups_x;
    if (groups_y_wide > owner->max_dispatch_y) {
      throw std::out_of_range(
          "vulkan audio: primitive exceeds two-dimensional dispatch limits");
    }
    parameters.groups_x = groups_x;
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->audio_bindings[i].buffer = &resources[i]->buffer;
      owner->audio_bindings[i].bytes = resources[i]->buffer.size();
    }
    commands.bind_compute(owner->audio_pipeline, owner->audio_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups_x, static_cast<uint32_t>(groups_y_wide));
  }

  void dispatch_keyframe(
      TensorContext::Impl::KeyframeParameters parameters,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 4>& resources) {
    const uint64_t total_groups =
        (static_cast<uint64_t>(parameters.count) + 63ull) / 64ull;
    const uint32_t groups_x = static_cast<uint32_t>(
        std::min<uint64_t>(total_groups, owner->max_dispatch_x));
    if (groups_x == 0) {
      throw std::out_of_range(
          "vulkan keyframe: device exposes no X dispatch capacity");
    }
    const uint64_t groups_y_wide =
        (total_groups + groups_x - 1ull) / groups_x;
    if (groups_y_wide > owner->max_dispatch_y) {
      throw std::out_of_range(
          "vulkan keyframe: convolution exceeds two-dimensional dispatch limits");
    }
    parameters.groups_x = groups_x;
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->keyframe_bindings[i].buffer = &resources[i]->buffer;
      owner->keyframe_bindings[i].bytes = resources[i]->buffer.size();
    }
    commands.bind_compute(owner->keyframe_pipeline, owner->keyframe_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups_x, static_cast<uint32_t>(groups_y_wide));
  }

  void dispatch_dit(
      TensorContext::Impl::DitParameters parameters,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 5>& resources,
      uint64_t tertiary_offset = 0) {
    const uint64_t total_groups =
        (static_cast<uint64_t>(parameters.count) + 63ull) / 64ull;
    const uint32_t groups_x = static_cast<uint32_t>(
        std::min<uint64_t>(total_groups, owner->max_dispatch_x));
    if (groups_x == 0) {
      throw std::out_of_range("vulkan DiT: device exposes no X dispatch capacity");
    }
    const uint64_t groups_y = (total_groups + groups_x - 1ull) / groups_x;
    if (groups_y > owner->max_dispatch_y) {
      throw std::out_of_range("vulkan DiT: operation exceeds dispatch limits");
    }
    parameters.groups_x = groups_x;
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->dit_bindings[i].buffer = &resources[i]->buffer;
      owner->dit_bindings[i].offset = 0;
      owner->dit_bindings[i].bytes = resources[i]->buffer.size();
    }
    owner->dit_bindings[2].offset = tertiary_offset;
    owner->dit_bindings[2].bytes -= tertiary_offset;
    commands.bind_compute(owner->dit_pipeline, owner->dit_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups_x, static_cast<uint32_t>(groups_y));
  }

  void dispatch_weight(
      const TensorContext::Impl::WeightParameters& parameters,
      uint32_t groups,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 6>& resources) {
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->weight_bindings[i].buffer = &resources[i]->buffer;
      owner->weight_bindings[i].bytes = resources[i]->buffer.size();
    }
    commands.bind_compute(owner->weight_pipeline, owner->weight_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups);
  }

  void dispatch_attention(
      const TensorContext::Impl::AttentionParameters& parameters,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 4>& resources) {
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->attention_bindings[i].buffer = &resources[i]->buffer;
      owner->attention_bindings[i].bytes = resources[i]->buffer.size();
    }
    commands.bind_compute(owner->attention_blocked_pipeline,
                          owner->attention_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(parameters.rows, parameters.heads);
  }

  void dispatch_h3_attention(
      const TensorContext::Impl::AttentionParameters& parameters,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 4>& resources,
      AttentionMode mode) {
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->attention_h3_bindings[i].buffer = &resources[i]->buffer;
      owner->attention_h3_bindings[i].bytes = resources[i]->buffer.size();
    }
    commands.bind_compute(mode == AttentionMode::kExact ? owner->attention_h3_pipeline
                                                       : owner->attention_flash_pipeline,
                          owner->attention_h3_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    const uint32_t tile = mode == AttentionMode::kExact ? 64u : kFastH3AttentionQueryTile;
    const uint32_t aligned_first = parameters.query_row_offset & ~(tile - 1u);
    const uint32_t groups =
        (parameters.query_row_offset + parameters.rows - aligned_first + tile - 1u) / tile;
    commands.dispatch(groups, parameters.heads);
  }

  void dispatch_h3_banded_attention(
      const TensorContext::Impl::AttentionParameters& parameters,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 5>& resources,
      AttentionMode mode) {
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->attention_h3_banded_bindings[i].buffer = &resources[i]->buffer;
      owner->attention_h3_banded_bindings[i].bytes = resources[i]->buffer.size();
    }
    commands.bind_compute(mode == AttentionMode::kExact ? owner->attention_h3_banded_pipeline
                                                       : owner->attention_flash_banded_pipeline,
                          owner->attention_h3_banded_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    const uint32_t tile = mode == AttentionMode::kExact ? 64u : kFastH3AttentionQueryTile;
    const uint32_t aligned_first = parameters.query_row_offset & ~(tile - 1u);
    const uint32_t groups =
        (parameters.query_row_offset + parameters.rows - aligned_first + tile - 1u) / tile;
    commands.dispatch(groups, parameters.heads);
  }

  void dispatch_attention_prepare(
      uint32_t words,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 6>& resources) {
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->attention_prepare_bindings[i].buffer = &resources[i]->buffer;
      owner->attention_prepare_bindings[i].bytes = resources[i]->buffer.size();
    }
    commands.bind_compute(owner->attention_prepare_pipeline,
                          owner->attention_prepare_bindings);
    commands.push_constants(&words, sizeof(words));
    commands.dispatch(static_cast<uint32_t>((static_cast<uint64_t>(words) + 63) / 64));
  }

  void dispatch_causal_gqa_attention(
      const TensorContext::Impl::CausalGQAAttentionParameters& parameters,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 4>& resources) {
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->attention_causal_gqa_bindings[i].buffer = &resources[i]->buffer;
      owner->attention_causal_gqa_bindings[i].bytes = resources[i]->buffer.size();
    }
    commands.bind_compute(owner->attention_causal_gqa_pipeline,
                          owner->attention_causal_gqa_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(parameters.rows, parameters.query_heads);
  }

  void record_shared_mod(bool fp32, DeviceTensor& input, DeviceTensor& weight,
                         DeviceTensor& scale, DeviceTensor& shift,
                         DeviceTensor& selectors, DeviceTensor& output,
                         float epsilon) {
    if (!owner->exact_vae_norm) {
      throw std::runtime_error("vulkan tensor: exact shared RMSNorm modulation is unavailable");
    }
    auto src = owner->require(input);
    auto w = owner->require(weight);
    auto sc = owner->require(scale);
    auto sh = owner->require(shift);
    auto index = owner->require(selectors);
    auto dst = owner->require(output);
    const auto& shape = src->layout;
    const uint64_t rows = shape.extent[0];
    const uint64_t dim = shape.extent[1];
    const uint64_t mod_rows = sc->layout.extent[0];
    const bool aliases_parameters =
        src.get() == w.get() || src.get() == sc.get() || src.get() == sh.get() ||
        src.get() == index.get() || w.get() == sc.get() || w.get() == sh.get() ||
        w.get() == index.get() || sc.get() == sh.get() || sc.get() == index.get() ||
        sh.get() == index.get() || dst.get() == w.get() || dst.get() == sc.get() ||
        dst.get() == sh.get() || dst.get() == index.get();
    if (!std::isnormal(epsilon) || epsilon <= 0.0f || aliases_parameters ||
        shape.rank != 2 || w->layout.rank != 1 || sc->layout.rank != 2 ||
        sh->layout.rank != 2 || index->layout.rank != 1 || dst->layout.rank != 2 ||
        w->layout.extent[0] != dim || sc->layout.extent[1] != dim ||
        sh->layout.extent != sc->layout.extent || index->layout.extent[0] != rows ||
        dst->layout.extent != shape.extent || w->type != ScalarType::kBFloat16 ||
        sc->type != ScalarType::kFloat32 || sh->type != ScalarType::kFloat32 ||
        index->type != ScalarType::kInt32 ||
        src->type != (fp32 ? ScalarType::kFloat32 : ScalarType::kBFloat16) ||
        dst->type != src->type || !shape.is_contiguous() ||
        !w->layout.is_contiguous() || !sc->layout.is_contiguous() ||
        !sh->layout.is_contiguous() || !index->layout.is_contiguous() ||
        !dst->layout.is_contiguous() || rows > std::numeric_limits<uint32_t>::max() ||
        dim > kMaxExactNormDimension ||
        mod_rows > std::numeric_limits<uint32_t>::max() ||
        shape.elements() > std::numeric_limits<uint32_t>::max() ||
        sc->layout.elements() > std::numeric_limits<uint32_t>::max()) {
      throw std::invalid_argument("vulkan tensor: invalid shared RMSNorm modulation");
    }
    if (!detail::norm_dispatch_fits(rows, owner->max_dispatch_x)) {
      throw std::out_of_range("vulkan tensor: modulated RMSNorm rows exceed dispatch limits");
    }
    TensorContext::Impl::NormParameters p;
    p.rows = static_cast<uint32_t>(rows);
    p.dim = static_cast<uint32_t>(dim);
    p.mod_rows = static_cast<uint32_t>(mod_rows);
    std::memcpy(&p.epsilon_bits, &epsilon, sizeof(epsilon));
    try {
      count_operator();
      transition(src, src.get() == dst.get() ? BufferAccess::kComputeReadWrite
                                              : BufferAccess::kComputeRead);
      transition(w, BufferAccess::kComputeRead);
      transition(sc, BufferAccess::kComputeRead);
      transition(sh, BufferAccess::kComputeRead);
      transition(index, BufferAccess::kComputeRead);
      if (src.get() != dst.get()) transition(dst, BufferAccess::kComputeWrite);
      std::array<std::shared_ptr<DeviceTensor::Impl>, 6> resources{
          src, w, sc, sh, index, dst};
      dispatch_shared_mod(fp32 ? owner->fp32_mod_pipeline : owner->bf16_mod_pipeline,
                          p, resources);
    } catch (...) {
      poisoned = true;
      throw;
    }
  }

  void record_shared_mod_table(DeviceTensor& input, DeviceTensor& weight,
                               DeviceTensor& tables, uint32_t mod_rows,
                               uint32_t scale_table,
                               uint32_t shift_table, DeviceTensor& selectors,
                               DeviceTensor& output, float epsilon) {
    if (!owner->exact_vae_norm)
      throw std::runtime_error("vulkan tensor: exact shared RMSNorm modulation is unavailable");
    auto src=owner->require(input), w=owner->require(weight), table=owner->require(tables);
    auto index=owner->require(selectors), dst=owner->require(output);
    const auto& shape=src->layout; const uint64_t rows=shape.rank==2?shape.extent[0]:0;
    const uint64_t dim=shape.rank==2?shape.extent[1]:0;
    const uint64_t table_count=6;
    const uint64_t stride=table->layout.rank==1?table->layout.extent[0]/table_count:0;
    const uint64_t logical=checked_multiply(mod_rows,dim,"modulation table");
    if (!std::isnormal(epsilon)||epsilon<=0.0f||scale_table>=table_count||
        shift_table>=table_count||table->layout.extent[0]%table_count!=0||stride<logical||
        src.get()==w.get()||src.get()==table.get()||
        src.get()==index.get()||w.get()==table.get()||w.get()==index.get()||
        table.get()==index.get()||dst.get()==w.get()||dst.get()==table.get()||
        dst.get()==index.get()||shape.rank!=2||w->layout.rank!=1||
        table->layout.rank!=1||
        index->layout.rank!=1||index->layout.extent[0]!=rows||
        dst->layout.rank!=2||dst->layout.extent!=shape.extent||
        src->type!=ScalarType::kBFloat16||w->type!=ScalarType::kBFloat16||
        table->type!=ScalarType::kFloat32||index->type!=ScalarType::kInt32||
        dst->type!=ScalarType::kBFloat16||w->layout.extent[0]!=dim||
        !shape.is_contiguous()||!w->layout.is_contiguous()||
        !table->layout.is_contiguous()||!index->layout.is_contiguous()||
        !dst->layout.is_contiguous()||rows>UINT32_MAX||dim>kMaxExactNormDimension||
        mod_rows==0||shape.elements()>UINT32_MAX)
      throw std::invalid_argument("vulkan tensor: invalid shared RMSNorm table modulation");
    if (!detail::norm_dispatch_fits(rows, owner->max_dispatch_x))
      throw std::out_of_range("vulkan tensor: modulated RMSNorm rows exceed dispatch limits");
    const uint64_t table_bytes=stride*4;
    TensorContext::Impl::NormParameters p; p.rows=static_cast<uint32_t>(rows);
    p.dim=static_cast<uint32_t>(dim);p.mod_rows=static_cast<uint32_t>(mod_rows);
    std::memcpy(&p.epsilon_bits,&epsilon,sizeof(epsilon));
    try {
      count_operator();transition(src,src.get()==dst.get()?BufferAccess::kComputeReadWrite:BufferAccess::kComputeRead);
      transition(w,BufferAccess::kComputeRead);transition(table,BufferAccess::kComputeRead);
      transition(index,BufferAccess::kComputeRead);if(src.get()!=dst.get())transition(dst,BufferAccess::kComputeWrite);
      std::array<std::shared_ptr<DeviceTensor::Impl>,6> r{src,w,table,table,index,dst};
      dispatch_shared_mod(owner->bf16_mod_pipeline,p,r,scale_table*table_bytes,shift_table*table_bytes);
    } catch (...) { poisoned=true; throw; }
  }

  ~Impl() {
    if (submitted) return;
    // Discard Vulkan recording before making its speculative access state
    // invisible to the next batch.
    commands = CommandList{};
    for (uint32_t i = snapshot_count; i-- > 0;) {
      snapshots[i].tensor->has_access = snapshots[i].has_access;
      snapshots[i].tensor->access = snapshots[i].access;
    }
  }
};

}  // namespace slopfab::vulkan
