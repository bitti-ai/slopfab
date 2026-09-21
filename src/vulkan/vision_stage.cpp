#include "slopfab/vulkan/vision_stage.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "slopfab/attention.h"
#include "slopfab/tensor_convert.h"

namespace slopfab::vulkan {
namespace {

uint64_t nonstaging_used_bytes(const TensorContext& context) {
  const uint64_t capacity = context.staging_capacity_bytes();
  const uint64_t staging = capacity > std::numeric_limits<uint64_t>::max() / 2
                               ? std::numeric_limits<uint64_t>::max()
                               : capacity * 2;
  const uint64_t used = context.pooled_used_bytes();
  return used >= staging ? used - staging : 0;
}

TensorLayout matrix(uint64_t rows, uint64_t columns) {
  const uint64_t shape[] = {rows, columns};
  return TensorLayout::contiguous(shape, 2);
}

TensorLayout three(uint64_t rows, uint64_t heads, uint64_t dim) {
  const uint64_t shape[] = {rows, heads, dim};
  return TensorLayout::contiguous(shape, 3);
}

TensorLayout vector(uint64_t count) {
  return TensorLayout::contiguous(&count, 1);
}

uint64_t bytes(const DeviceTensor& tensor) {
  return tensor ? tensor.layout().bytes(tensor.type()) : 0;
}

void validate_config(const QwenVisionStageConfig& c) {
  const auto& v = c.vision;
  if (c.sequence == 0 || c.sequence > 16384 || (c.sequence & 3u) != 0 || v.hidden_size != 1152 ||
      v.intermediate_size != 4304 || v.num_heads != 16 || v.depth != 27 || v.position_side != 48 ||
      v.merge_size != 2 || v.output_size != 5120) {
    throw std::invalid_argument("Vulkan Qwen vision: invalid production configuration");
  }
}

bool same_config(const QwenVisionStageConfig& a, const QwenVisionStageConfig& b) {
  return a.sequence == b.sequence && a.vision.hidden_size == b.vision.hidden_size &&
         a.vision.intermediate_size == b.vision.intermediate_size &&
         a.vision.num_heads == b.vision.num_heads && a.vision.depth == b.vision.depth;
}

uint32_t row_gemm_operators(uint32_t rows) {
  return (rows >= 64 ? 1u : 0u) + (rows % 64 != 0 ? 1u : 0u);
}

DeviceTensor upload_bf16(TensorContext& context, const TensorView& view, const TensorLayout& layout,
                         const char* what) {
  if (view.dtype != DType::kBF16 || view.nbytes != layout.bytes(ScalarType::kBFloat16))
    throw std::runtime_error(std::string("Vulkan Qwen vision: invalid ") + what);
  DeviceTensor result = context.allocate(layout, ScalarType::kBFloat16);
  context.upload_transient_bytes(result, view.data, view.nbytes);
  return result;
}

} // namespace

struct ExactQwenVisionScratch::Impl {
  TensorContext* context = nullptr;
  QwenVisionStageConfig config;
  DeviceTensor normed, qkv, query, key, value, attention, branch, mlp;
  DeviceTensor merged, merger_normed, merger_hidden;
  PreparedAttentionInputs prepared_attention;
  BlockedAttentionPlan attention_plan;
  DenseGemmPlan qkv_plan, projection_plan, fc1_plan, fc2_plan;
  DenseGemmPlan patch_plan, merger_fc1_plan, merger_fc2_plan;

  Impl(TensorContext& owner, const QwenVisionStageConfig& c) : context(&owner), config(c) {
    const uint32_t rows = c.sequence;
    const uint32_t hidden = static_cast<uint32_t>(c.vision.hidden_size);
    const uint32_t heads = static_cast<uint32_t>(c.vision.num_heads);
    const uint32_t head_dim = hidden / heads;
    const uint32_t intermediate = static_cast<uint32_t>(c.vision.intermediate_size);
    normed = owner.allocate(matrix(rows, hidden), ScalarType::kBFloat16);
    qkv = owner.allocate(matrix(rows, 3u * hidden), ScalarType::kBFloat16);
    query = owner.allocate(three(rows, heads, head_dim), ScalarType::kBFloat16);
    key = owner.allocate(three(rows, heads, head_dim), ScalarType::kBFloat16);
    value = owner.allocate(three(rows, heads, head_dim), ScalarType::kBFloat16);
    attention = owner.allocate(three(rows, heads, head_dim), ScalarType::kBFloat16);
    branch = owner.allocate(matrix(rows, hidden), ScalarType::kBFloat16);
    mlp = owner.allocate(matrix(rows, intermediate), ScalarType::kBFloat16);
    const uint32_t groups = rows / 4u;
    constexpr uint32_t merged_width = 4608;
    merged = owner.allocate(matrix(groups, merged_width), ScalarType::kBFloat16);
    merger_normed = owner.allocate(matrix(groups, merged_width), ScalarType::kBFloat16);
    merger_hidden = owner.allocate(matrix(groups, merged_width), ScalarType::kBFloat16);
    const BlockedAttentionPlanDesc attention_desc{rows, heads, head_dim,
                                                  exact_attention_scale(head_dim)};
    prepared_attention = PreparedAttentionInputs::create(owner, attention_desc);
    attention_plan = BlockedAttentionPlan::create(owner, attention_desc);
    auto plan = [&](uint32_t out, uint32_t in) {
      return DenseGemmPlan::create(
          owner, {rows, out, in, DenseGemmMode::kBFloat16, DenseGemmBias::kBFloat16, false});
    };
    qkv_plan = plan(3u * hidden, hidden);
    projection_plan = plan(hidden, hidden);
    fc1_plan = plan(intermediate, hidden);
    fc2_plan = plan(hidden, intermediate);
    patch_plan = DenseGemmPlan::create(
        owner, {rows, hidden, 1536, DenseGemmMode::kBFloat16, DenseGemmBias::kBFloat16, false});
    auto merger_plan = [&](uint32_t out, uint32_t in) {
      return DenseGemmPlan::create(
          owner, {groups, out, in, DenseGemmMode::kBFloat16, DenseGemmBias::kBFloat16, false});
    };
    merger_fc1_plan = merger_plan(merged_width, merged_width);
    merger_fc2_plan = merger_plan(5120, merged_width);
  }

  uint64_t reserved() const noexcept {
    return bytes(normed) + bytes(qkv) + bytes(query) + bytes(key) + bytes(value) +
           bytes(attention) + bytes(branch) + bytes(mlp) + bytes(merged) + bytes(merger_normed) +
           bytes(merger_hidden) + prepared_attention.reserved_bytes();
  }
};

struct ExactQwenVisionPatchStage::Impl {
  TensorContext* context = nullptr;
  QwenVisionStageConfig config;
  DeviceTensor weight, bias, position_table;

  Impl(TensorContext& owner, const QwenVisionStageConfig& c) : context(&owner), config(c) {
  }

  uint64_t resident() const noexcept {
    return bytes(weight) + bytes(bias) + bytes(position_table);
  }
};

ExactQwenVisionPatchStage::ExactQwenVisionPatchStage() = default;
ExactQwenVisionPatchStage::~ExactQwenVisionPatchStage() = default;

ExactQwenVisionPatchStage::ExactQwenVisionPatchStage(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {
}

ExactQwenVisionPatchStage::ExactQwenVisionPatchStage(ExactQwenVisionPatchStage&&) noexcept =
    default;
ExactQwenVisionPatchStage&
ExactQwenVisionPatchStage::operator=(ExactQwenVisionPatchStage&&) noexcept = default;

ExactQwenVisionPatchStage ExactQwenVisionPatchStage::create(TensorContext& context,
                                                            const QwenVisionStageConfig& config) {
  validate_config(config);
  context.require_exact_vae_pointwise();
  return ExactQwenVisionPatchStage(std::make_shared<Impl>(context, config));
}

void ExactQwenVisionPatchStage::load(const text::QwenVisionCheckpoint& checkpoint) {
  if (!impl_ || !checkpoint.checkpoint ||
      checkpoint.config.hidden_size != impl_->config.vision.hidden_size)
    throw std::invalid_argument("Vulkan Qwen vision patch: invalid archive");
  const SafeTensors& archive = *checkpoint.checkpoint;
  const std::string& p = checkpoint.prefix;
  Impl next(*impl_->context, impl_->config);
  next.weight = upload_bf16(*impl_->context, archive.at(p + "patch_embed.proj.weight"),
                            matrix(1152, 1536), "patch weight");
  next.bias = upload_bf16(*impl_->context, archive.at(p + "patch_embed.proj.bias"), vector(1152),
                          "patch bias");
  next.position_table = upload_bf16(*impl_->context, archive.at(p + "pos_embed.weight"),
                                    matrix(2304, 1152), "position table");
  impl_->weight = std::move(next.weight);
  impl_->bias = std::move(next.bias);
  impl_->position_table = std::move(next.position_table);
}

void ExactQwenVisionPatchStage::unload() noexcept {
  if (impl_) {
    impl_->weight = DeviceTensor();
    impl_->bias = DeviceTensor();
    impl_->position_table = DeviceTensor();
  }
}

bool ExactQwenVisionPatchStage::loaded() const noexcept {
  return impl_ && impl_->weight && impl_->bias && impl_->position_table;
}

uint32_t ExactQwenVisionPatchStage::required_operators() const noexcept {
  return impl_ ? row_gemm_operators(impl_->config.sequence) + 1u : 0u;
}

uint64_t ExactQwenVisionPatchStage::persistent_bytes() const noexcept {
  return loaded() ? impl_->resident() : 0;
}

void ExactQwenVisionPatchStage::record(TensorBatch& batch, DeviceTensor& pixel_rows,
                                       DeviceTensor& learned_index, DeviceTensor& output,
                                       ExactQwenVisionScratch& scratch) const {
  if (!loaded())
    throw std::logic_error("Vulkan Qwen vision patch: not loaded");
  if (!scratch.impl_ || scratch.impl_->context != impl_->context ||
      !same_config(scratch.impl_->config, impl_->config) || !batch.belongs_to(*impl_->context))
    throw std::invalid_argument("Vulkan Qwen vision patch: incompatible scratch");
  const uint32_t rows = impl_->config.sequence;
  const DeviceTensorView pixels = pixel_rows.view(), index = learned_index.view();
  const DeviceTensorView out = output.view();
  if (!impl_->context->owns(pixel_rows) || !impl_->context->owns(learned_index) ||
      !impl_->context->owns(output) || pixels.type != ScalarType::kBFloat16 ||
      pixels.layout.rank != 2 || pixels.layout.extent[0] != rows ||
      pixels.layout.extent[1] != 1536 || !pixels.layout.is_contiguous() ||
      index.type != ScalarType::kInt32 || index.layout.rank != 1 ||
      index.layout.extent[0] != rows || !index.layout.is_contiguous() ||
      out.type != ScalarType::kBFloat16 || out.layout.rank != 2 || out.layout.extent[0] != rows ||
      out.layout.extent[1] != 1152 || !out.layout.is_contiguous() ||
      pixels.resource == index.resource || pixels.resource == out.resource ||
      index.resource == out.resource)
    throw std::invalid_argument("Vulkan Qwen vision patch: invalid inputs");
  batch.require_operator_capacity(required_operators());
  const uint32_t tiled = rows / 64u * 64u;
  if (tiled != 0)
    scratch.impl_->patch_plan.record(batch, pixel_rows, impl_->weight, output, tiled, 0, 0,
                                     &impl_->bias);
  if (tiled != rows)
    scratch.impl_->patch_plan.record(batch, pixel_rows, impl_->weight, output, rows - tiled, tiled,
                                     tiled, &impl_->bias);
  batch.vision_add_positions_bf16(output, impl_->position_table, learned_index);
}

struct ExactQwenVisionBlockStage::Impl {
  struct Weights {
    uint32_t block = 0;
    DeviceTensor norm1_weight, norm1_bias, norm2_weight, norm2_bias;
    DeviceTensor qkv_weight, qkv_bias, projection_weight, projection_bias;
    DeviceTensor fc1_weight, fc1_bias, fc2_weight, fc2_bias;

    uint64_t resident() const noexcept {
      return bytes(norm1_weight) + bytes(norm1_bias) + bytes(norm2_weight) + bytes(norm2_bias) +
             bytes(qkv_weight) + bytes(qkv_bias) + bytes(projection_weight) +
             bytes(projection_bias) + bytes(fc1_weight) + bytes(fc1_bias) + bytes(fc2_weight) +
             bytes(fc2_bias);
    }
  };

  TensorContext* context = nullptr;
  QwenVisionStageConfig config;
  std::unique_ptr<Weights> weights;

  Impl(TensorContext& owner, const QwenVisionStageConfig& c) : context(&owner), config(c) {
  }
};

struct ExactQwenVisionMergerStage::Impl {
  struct Weights {
    int slot = -2;
    DeviceTensor norm_weight, norm_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias;

    uint64_t resident() const noexcept {
      return bytes(norm_weight) + bytes(norm_bias) + bytes(fc1_weight) + bytes(fc1_bias) +
             bytes(fc2_weight) + bytes(fc2_bias);
    }
  };

  TensorContext* context = nullptr;
  QwenVisionStageConfig config;
  std::unique_ptr<Weights> weights;

  Impl(TensorContext& owner, const QwenVisionStageConfig& c) : context(&owner), config(c) {
  }
};

ExactQwenVisionScratch::ExactQwenVisionScratch() = default;
ExactQwenVisionScratch::~ExactQwenVisionScratch() = default;

ExactQwenVisionScratch::ExactQwenVisionScratch(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {
}

ExactQwenVisionScratch::ExactQwenVisionScratch(ExactQwenVisionScratch&&) noexcept = default;
ExactQwenVisionScratch&
ExactQwenVisionScratch::operator=(ExactQwenVisionScratch&&) noexcept = default;

ExactQwenVisionScratch ExactQwenVisionScratch::create(TensorContext& context,
                                                      const QwenVisionStageConfig& config) {
  validate_config(config);
  context.require_exact_fp32_vae_normalization();
  context.require_exact_vae_pointwise();
  context.require_exact_blocked_attention();
  return ExactQwenVisionScratch(std::make_shared<Impl>(context, config));
}

uint64_t ExactQwenVisionScratch::reserved_bytes() const noexcept {
  return impl_ ? impl_->reserved() : 0;
}

ExactQwenVisionBlockStage::ExactQwenVisionBlockStage() = default;
ExactQwenVisionBlockStage::~ExactQwenVisionBlockStage() = default;

ExactQwenVisionBlockStage::ExactQwenVisionBlockStage(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {
}

ExactQwenVisionBlockStage::ExactQwenVisionBlockStage(ExactQwenVisionBlockStage&&) noexcept =
    default;
ExactQwenVisionBlockStage&
ExactQwenVisionBlockStage::operator=(ExactQwenVisionBlockStage&&) noexcept = default;

ExactQwenVisionBlockStage ExactQwenVisionBlockStage::create(TensorContext& context,
                                                            const QwenVisionStageConfig& config) {
  validate_config(config);
  context.require_exact_fp32_vae_normalization();
  context.require_exact_vae_pointwise();
  context.require_exact_blocked_attention();
  return ExactQwenVisionBlockStage(std::make_shared<Impl>(context, config));
}

void ExactQwenVisionBlockStage::load(const text::QwenVisionCheckpoint& checkpoint,
                                     uint32_t block_index) {
  if (!impl_)
    throw std::logic_error("Vulkan Qwen vision: empty stage");
  if (!checkpoint.checkpoint || block_index >= 27 ||
      checkpoint.config.hidden_size != impl_->config.vision.hidden_size ||
      checkpoint.config.intermediate_size != impl_->config.vision.intermediate_size) {
    throw std::invalid_argument("Vulkan Qwen vision: invalid validated archive");
  }
  const SafeTensors& archive = *checkpoint.checkpoint;
  const std::string prefix = checkpoint.prefix + "blocks." + std::to_string(block_index) + ".";
  const uint32_t hidden = static_cast<uint32_t>(impl_->config.vision.hidden_size);
  const uint32_t intermediate = static_cast<uint32_t>(impl_->config.vision.intermediate_size);
  auto next = std::make_unique<Impl::Weights>();
  next->block = block_index;
  next->norm1_weight = upload_bf16(*impl_->context, archive.at(prefix + "norm1.weight"),
                                   vector(hidden), "norm1 weight");
  next->norm1_bias =
      upload_bf16(*impl_->context, archive.at(prefix + "norm1.bias"), vector(hidden), "norm1 bias");
  next->norm2_weight = upload_bf16(*impl_->context, archive.at(prefix + "norm2.weight"),
                                   vector(hidden), "norm2 weight");
  next->norm2_bias =
      upload_bf16(*impl_->context, archive.at(prefix + "norm2.bias"), vector(hidden), "norm2 bias");
  next->qkv_weight = upload_bf16(*impl_->context, archive.at(prefix + "attn.qkv.weight"),
                                 matrix(3u * hidden, hidden), "QKV weight");
  next->qkv_bias = upload_bf16(*impl_->context, archive.at(prefix + "attn.qkv.bias"),
                               vector(3u * hidden), "QKV bias");
  next->projection_weight = upload_bf16(*impl_->context, archive.at(prefix + "attn.proj.weight"),
                                        matrix(hidden, hidden), "attention projection weight");
  next->projection_bias = upload_bf16(*impl_->context, archive.at(prefix + "attn.proj.bias"),
                                      vector(hidden), "attention projection bias");
  next->fc1_weight = upload_bf16(*impl_->context, archive.at(prefix + "mlp.linear_fc1.weight"),
                                 matrix(intermediate, hidden), "MLP fc1 weight");
  next->fc1_bias = upload_bf16(*impl_->context, archive.at(prefix + "mlp.linear_fc1.bias"),
                               vector(intermediate), "MLP fc1 bias");
  next->fc2_weight = upload_bf16(*impl_->context, archive.at(prefix + "mlp.linear_fc2.weight"),
                                 matrix(hidden, intermediate), "MLP fc2 weight");
  next->fc2_bias = upload_bf16(*impl_->context, archive.at(prefix + "mlp.linear_fc2.bias"),
                               vector(hidden), "MLP fc2 bias");
  impl_->weights = std::move(next);
}

void ExactQwenVisionBlockStage::unload() noexcept {
  if (impl_)
    impl_->weights.reset();
}

bool ExactQwenVisionBlockStage::loaded() const noexcept {
  return impl_ && impl_->weights;
}

uint32_t ExactQwenVisionBlockStage::block() const {
  if (!loaded())
    throw std::logic_error("Vulkan Qwen vision: stage not loaded");
  return impl_->weights->block;
}

uint32_t ExactQwenVisionBlockStage::required_operators() const noexcept {
  return impl_ ? 12u + 4u * row_gemm_operators(impl_->config.sequence) : 0u;
}

uint64_t ExactQwenVisionBlockStage::persistent_bytes() const noexcept {
  return loaded() ? impl_->weights->resident() : 0;
}

void ExactQwenVisionBlockStage::record(TensorBatch& batch, DeviceTensor& residual,
                                       DeviceTensor& cosine, DeviceTensor& sine,
                                       ExactQwenVisionScratch& scratch) const {
  if (!loaded())
    throw std::logic_error("Vulkan Qwen vision: stage not loaded");
  if (!scratch.impl_ || scratch.impl_->context != impl_->context ||
      !same_config(scratch.impl_->config, impl_->config) || !batch.belongs_to(*impl_->context)) {
    throw std::invalid_argument("Vulkan Qwen vision: incompatible scratch/batch");
  }
  const uint32_t rows = impl_->config.sequence;
  const uint32_t hidden = static_cast<uint32_t>(impl_->config.vision.hidden_size);
  const DeviceTensorView x = residual.view(), c = cosine.view(), si = sine.view();
  if (!impl_->context->owns(residual) || !impl_->context->owns(cosine) ||
      !impl_->context->owns(sine) || x.type != ScalarType::kBFloat16 || x.layout.rank != 2 ||
      x.layout.extent[0] != rows || x.layout.extent[1] != hidden || !x.layout.is_contiguous() ||
      c.type != ScalarType::kFloat32 || si.type != ScalarType::kFloat32 || c.layout.rank != 2 ||
      si.layout.rank != 2 || c.layout.extent[0] != rows || si.layout.extent[0] != rows ||
      c.layout.extent[1] != 72 || si.layout.extent[1] != 72 || !c.layout.is_contiguous() ||
      !si.layout.is_contiguous() || x.resource == c.resource || x.resource == si.resource ||
      c.resource == si.resource) {
    throw std::invalid_argument("Vulkan Qwen vision: invalid block inputs");
  }
  batch.require_operator_capacity(required_operators());
  auto& s = *scratch.impl_;
  auto& w = *impl_->weights;
  auto gemm = [&](const DenseGemmPlan& plan, DeviceTensor& input, DeviceTensor& weight,
                  DeviceTensor& bias, DeviceTensor& output) {
    const uint32_t tiled = rows / 64u * 64u;
    if (tiled != 0)
      plan.record(batch, input, weight, output, tiled, 0, 0, &bias);
    if (tiled != rows)
      plan.record(batch, input, weight, output, rows - tiled, tiled, tiled, &bias);
  };
  batch.layer_norm_bf16(residual, w.norm1_weight, w.norm1_bias, s.normed, 1.0e-6f);
  gemm(s.qkv_plan, s.normed, w.qkv_weight, w.qkv_bias, s.qkv);
  batch.vision_split_qkv_bf16(s.qkv, s.query, s.key, s.value);
  batch.rope_neox_bf16(s.query, cosine, sine);
  batch.rope_neox_bf16(s.key, cosine, sine);
  PreparedAttentionView prepared = s.prepared_attention.prepare(batch, s.query, s.key, s.value);
  s.attention_plan.record(batch, prepared, s.attention);
  gemm(s.projection_plan, s.attention, w.projection_weight, w.projection_bias, s.branch);
  batch.text_add_residual_bf16(residual, s.branch);
  batch.layer_norm_bf16(residual, w.norm2_weight, w.norm2_bias, s.normed, 1.0e-6f);
  gemm(s.fc1_plan, s.normed, w.fc1_weight, w.fc1_bias, s.mlp);
  batch.vision_gelu_tanh_bf16(s.mlp);
  gemm(s.fc2_plan, s.mlp, w.fc2_weight, w.fc2_bias, s.branch);
  batch.text_add_residual_bf16(residual, s.branch);
}

ExactQwenVisionMergerStage::ExactQwenVisionMergerStage() = default;
ExactQwenVisionMergerStage::~ExactQwenVisionMergerStage() = default;

ExactQwenVisionMergerStage::ExactQwenVisionMergerStage(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {
}

ExactQwenVisionMergerStage::ExactQwenVisionMergerStage(ExactQwenVisionMergerStage&&) noexcept =
    default;
ExactQwenVisionMergerStage&
ExactQwenVisionMergerStage::operator=(ExactQwenVisionMergerStage&&) noexcept = default;

ExactQwenVisionMergerStage ExactQwenVisionMergerStage::create(TensorContext& context,
                                                              const QwenVisionStageConfig& config) {
  validate_config(config);
  context.require_exact_fp32_vae_normalization();
  context.require_exact_vae_pointwise();
  return ExactQwenVisionMergerStage(std::make_shared<Impl>(context, config));
}

void ExactQwenVisionMergerStage::load(const text::QwenVisionCheckpoint& checkpoint,
                                      int merger_slot) {
  if (!impl_ || !checkpoint.checkpoint || merger_slot < -1 || merger_slot > 2 ||
      checkpoint.config.hidden_size != impl_->config.vision.hidden_size ||
      checkpoint.config.output_size != impl_->config.vision.output_size)
    throw std::invalid_argument("Vulkan Qwen vision merger: invalid archive/slot");
  const SafeTensors& archive = *checkpoint.checkpoint;
  const std::string p = merger_slot < 0 ? checkpoint.prefix + "merger."
                                        : checkpoint.prefix + "deepstack_merger_list." +
                                              std::to_string(merger_slot) + ".";
  const uint32_t norm_width = merger_slot < 0 ? 1152u : 4608u;
  auto next = std::make_unique<Impl::Weights>();
  next->slot = merger_slot;
  next->norm_weight = upload_bf16(*impl_->context, archive.at(p + "norm.weight"),
                                  vector(norm_width), "merger norm weight");
  next->norm_bias = upload_bf16(*impl_->context, archive.at(p + "norm.bias"), vector(norm_width),
                                "merger norm bias");
  next->fc1_weight = upload_bf16(*impl_->context, archive.at(p + "linear_fc1.weight"),
                                 matrix(4608, 4608), "merger fc1 weight");
  next->fc1_bias = upload_bf16(*impl_->context, archive.at(p + "linear_fc1.bias"), vector(4608),
                               "merger fc1 bias");
  next->fc2_weight = upload_bf16(*impl_->context, archive.at(p + "linear_fc2.weight"),
                                 matrix(5120, 4608), "merger fc2 weight");
  next->fc2_bias = upload_bf16(*impl_->context, archive.at(p + "linear_fc2.bias"), vector(5120),
                               "merger fc2 bias");
  impl_->weights = std::move(next);
}

void ExactQwenVisionMergerStage::unload() noexcept {
  if (impl_)
    impl_->weights.reset();
}

bool ExactQwenVisionMergerStage::loaded() const noexcept {
  return impl_ && impl_->weights;
}

int ExactQwenVisionMergerStage::slot() const {
  if (!loaded())
    throw std::logic_error("Vulkan Qwen vision merger: not loaded");
  return impl_->weights->slot;
}

uint32_t ExactQwenVisionMergerStage::required_operators() const noexcept {
  return impl_ ? 3u + 2u * row_gemm_operators(impl_->config.sequence / 4u) : 0u;
}

uint64_t ExactQwenVisionMergerStage::persistent_bytes() const noexcept {
  return loaded() ? impl_->weights->resident() : 0;
}

void ExactQwenVisionMergerStage::record(TensorBatch& batch, DeviceTensor& visual_residual,
                                        DeviceTensor& output,
                                        ExactQwenVisionScratch& scratch) const {
  if (!loaded())
    throw std::logic_error("Vulkan Qwen vision merger: not loaded");
  if (!scratch.impl_ || scratch.impl_->context != impl_->context ||
      !same_config(scratch.impl_->config, impl_->config) || !batch.belongs_to(*impl_->context))
    throw std::invalid_argument("Vulkan Qwen vision merger: incompatible scratch");
  const uint32_t rows = impl_->config.sequence, groups = rows / 4u;
  const DeviceTensorView input = visual_residual.view(), out = output.view();
  if (!impl_->context->owns(visual_residual) || !impl_->context->owns(output) ||
      input.type != ScalarType::kBFloat16 || input.layout.rank != 2 ||
      input.layout.extent[0] != rows || input.layout.extent[1] != 1152 ||
      !input.layout.is_contiguous() || out.type != ScalarType::kBFloat16 || out.layout.rank != 2 ||
      out.layout.extent[0] != groups || out.layout.extent[1] != 5120 ||
      !out.layout.is_contiguous() || input.resource == out.resource)
    throw std::invalid_argument("Vulkan Qwen vision merger: invalid inputs");
  batch.require_operator_capacity(required_operators());
  auto& s = *scratch.impl_;
  auto& w = *impl_->weights;
  DeviceTensor* merged_input = &s.merged;
  if (w.slot < 0) {
    batch.layer_norm_bf16(visual_residual, w.norm_weight, w.norm_bias, s.normed, 1.0e-6f);
    batch.vision_merge_four_bf16(s.normed, s.merged);
  } else {
    batch.vision_merge_four_bf16(visual_residual, s.merged);
    batch.layer_norm_bf16(s.merged, w.norm_weight, w.norm_bias, s.merger_normed, 1.0e-6f);
    merged_input = &s.merger_normed;
  }
  const uint32_t tiled = groups / 64u * 64u;
  auto gemm = [&](const DenseGemmPlan& plan, DeviceTensor& source, DeviceTensor& weight,
                  DeviceTensor& bias, DeviceTensor& destination) {
    if (tiled != 0)
      plan.record(batch, source, weight, destination, tiled, 0, 0, &bias);
    if (tiled != groups)
      plan.record(batch, source, weight, destination, groups - tiled, tiled, tiled, &bias);
  };
  gemm(s.merger_fc1_plan, *merged_input, w.fc1_weight, w.fc1_bias, s.merger_hidden);
  batch.vision_gelu_tanh_bf16(s.merger_hidden);
  gemm(s.merger_fc2_plan, s.merger_hidden, w.fc2_weight, w.fc2_bias, output);
}

struct ExactQwenVisionEncoder::Impl {
  struct Shape {
    QwenVisionStageConfig config;
    ExactQwenVisionScratch scratch;
    ExactQwenVisionPatchStage patch;
    ExactQwenVisionBlockStage block;
    ExactQwenVisionMergerStage merger;
    DeviceTensor pixels, learned_index, cosine, sine, residual, main, deep[3];

    Shape(TensorContext& context, uint32_t rows) {
      config.sequence = rows;
      scratch = ExactQwenVisionScratch::create(context, config);
      patch = ExactQwenVisionPatchStage::create(context, config);
      block = ExactQwenVisionBlockStage::create(context, config);
      merger = ExactQwenVisionMergerStage::create(context, config);
      pixels = context.allocate(matrix(rows, 1536), ScalarType::kBFloat16);
      learned_index = context.allocate(vector(rows), ScalarType::kInt32);
      cosine = context.allocate(matrix(rows, 72), ScalarType::kFloat32);
      sine = context.allocate(matrix(rows, 72), ScalarType::kFloat32);
      residual = context.allocate(matrix(rows, 1152), ScalarType::kBFloat16);
      main = context.allocate(matrix(rows / 4u, 5120), ScalarType::kBFloat16);
      for (auto& tensor : deep)
        tensor = context.allocate(matrix(rows / 4u, 5120), ScalarType::kBFloat16);
    }

    uint64_t activation_bytes() const noexcept {
      uint64_t result = bytes(pixels) + bytes(learned_index) + bytes(cosine) + bytes(sine) +
                        bytes(residual) + bytes(main);
      for (const auto& tensor : deep)
        result += bytes(tensor);
      return result;
    }
  };

  TensorContext* context = nullptr;
  const SafeTensors* checkpoint = nullptr;
  text::QwenVisionCheckpoint validated;
  std::unique_ptr<Shape> shape;
  ExactQwenVisionStats stats;
  uint32_t output_tokens = 0;

  explicit Impl(TensorContext& owner) : context(&owner) {
  }
};

ExactQwenVisionEncoder::ExactQwenVisionEncoder() = default;
ExactQwenVisionEncoder::~ExactQwenVisionEncoder() = default;

ExactQwenVisionEncoder::ExactQwenVisionEncoder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {
}

ExactQwenVisionEncoder::ExactQwenVisionEncoder(ExactQwenVisionEncoder&&) noexcept = default;
ExactQwenVisionEncoder&
ExactQwenVisionEncoder::operator=(ExactQwenVisionEncoder&&) noexcept = default;

ExactQwenVisionEncoder ExactQwenVisionEncoder::create(TensorContext& context) {
  context.require_exact_fp32_vae_normalization();
  context.require_exact_vae_pointwise();
  context.require_exact_blocked_attention();
  return ExactQwenVisionEncoder(std::make_unique<Impl>(context));
}

void ExactQwenVisionEncoder::load(const SafeTensors& checkpoint) {
  if (!impl_)
    throw std::logic_error("Vulkan Qwen vision encoder: empty");
  const text::QwenVisionCheckpoint validated = text::load_qwen3vl_vision_checkpoint(checkpoint);
  impl_->checkpoint = &checkpoint;
  impl_->validated = validated;
  impl_->stats = ExactQwenVisionStats{};
  impl_->output_tokens = 0;
}

void ExactQwenVisionEncoder::unload() noexcept {
  if (!impl_)
    return;
  impl_->shape.reset();
  try {
    impl_->context->collect();
  } catch (...) {
  }
  impl_->checkpoint = nullptr;
  impl_->validated = {};
  impl_->stats = {};
  impl_->output_tokens = 0;
}

bool ExactQwenVisionEncoder::loaded() const noexcept {
  return impl_ && impl_->checkpoint != nullptr;
}

void ExactQwenVisionEncoder::encode(const text::QwenPixelValues& image,
                                    text::QwenVisionTrace* trace) {
  if (!loaded())
    throw std::logic_error("Vulkan Qwen vision encoder: not loaded");
  const size_t patch_count = image.grid.patch_count();
  if (image.grid.temporal <= 0 || image.grid.height <= 0 || image.grid.width <= 0 ||
      (image.grid.height & 1) != 0 || (image.grid.width & 1) != 0 || patch_count == 0 ||
      patch_count > 16384 || patch_count % 4 != 0 || image.rows.size() != patch_count * 1536)
    throw std::invalid_argument("Vulkan Qwen vision encoder: invalid image rows");
  const uint32_t rows = static_cast<uint32_t>(patch_count);
  const uint32_t max_operators = std::max(12u + 4u * row_gemm_operators(rows) + (trace ? 1u : 0u),
                                          3u + 2u * row_gemm_operators(rows / 4u));
  {
    TensorBatch preflight = impl_->context->begin_batch();
    preflight.require_operator_capacity(max_operators);
  }
  if (!impl_->shape || impl_->shape->config.sequence != rows)
    impl_->shape = std::make_unique<Impl::Shape>(*impl_->context, rows);
  auto& s = *impl_->shape;
  DeviceTensor trace_device;
  if (trace) {
    *trace = {};
    trace_device = impl_->context->allocate(matrix(27u * rows, 1152), ScalarType::kBFloat16);
  }
  std::vector<uint16_t> pixels(image.rows.size());
  for (size_t i = 0; i < pixels.size(); ++i)
    pixels[i] = f32_to_bf16(image.rows[i]);
  const text::QwenVisionPositions positions = text::qwen3vl_vision_positions(image.grid);
  std::vector<float> cosine, sine;
  text::qwen3vl_vision_rope_tables(positions, cosine, sine);
  const TensorUpload uploads[] = {
      {&s.pixels, pixels.data(), pixels.size() * sizeof(uint16_t)},
      {&s.learned_index, positions.learned.data(), positions.learned.size() * sizeof(int32_t)},
      {&s.cosine, cosine.data(), cosine.size() * sizeof(float)},
      {&s.sine, sine.data(), sine.size() * sizeof(float)}};
  impl_->context->upload_batch(uploads, 4);
  const auto begin = std::chrono::steady_clock::now();
  uint64_t peak_used = impl_->context->pooled_used_bytes();
  uint64_t peak_nonstaging = nonstaging_used_bytes(*impl_->context);
  uint64_t max_weight = 0;
  try {
    s.patch.load(impl_->validated);
    max_weight = std::max(max_weight, s.patch.persistent_bytes());
    peak_used = std::max(peak_used, impl_->context->pooled_used_bytes());
    peak_nonstaging = std::max(peak_nonstaging, nonstaging_used_bytes(*impl_->context));
    {
      TensorBatch batch = impl_->context->begin_batch();
      s.patch.record(batch, s.pixels, s.learned_index, s.residual, s.scratch);
      batch.submit().wait();
    }
    s.patch.unload();
    for (uint32_t layer = 0; layer < 27; ++layer) {
      s.block.load(impl_->validated, layer);
      max_weight = std::max(max_weight, s.block.persistent_bytes());
      peak_used = std::max(peak_used, impl_->context->pooled_used_bytes());
      peak_nonstaging = std::max(peak_nonstaging, nonstaging_used_bytes(*impl_->context));
      {
        TensorBatch batch = impl_->context->begin_batch();
        s.block.record(batch, s.residual, s.cosine, s.sine, s.scratch);
        if (trace)
          batch.copy_rows(s.residual, trace_device, 0, layer * rows, rows);
        batch.submit().wait();
      }
      s.block.unload();
      int deep_slot = layer == 8 ? 0 : layer == 16 ? 1 : layer == 24 ? 2 : -1;
      if (deep_slot >= 0) {
        s.merger.load(impl_->validated, deep_slot);
        max_weight = std::max(max_weight, s.merger.persistent_bytes());
        peak_used = std::max(peak_used, impl_->context->pooled_used_bytes());
        peak_nonstaging = std::max(peak_nonstaging, nonstaging_used_bytes(*impl_->context));
        TensorBatch batch = impl_->context->begin_batch();
        s.merger.record(batch, s.residual, s.deep[deep_slot], s.scratch);
        batch.submit().wait();
        s.merger.unload();
      }
    }
    s.merger.load(impl_->validated, -1);
    max_weight = std::max(max_weight, s.merger.persistent_bytes());
    peak_used = std::max(peak_used, impl_->context->pooled_used_bytes());
    peak_nonstaging = std::max(peak_nonstaging, nonstaging_used_bytes(*impl_->context));
    {
      TensorBatch batch = impl_->context->begin_batch();
      s.merger.record(batch, s.residual, s.main, s.scratch);
      batch.submit().wait();
    }
    s.merger.unload();
    if (trace) {
      trace->tokens = static_cast<int>(rows);
      trace->block_residuals.resize(static_cast<size_t>(27) * rows * 1152u);
      impl_->context->download_bytes(trace_device, trace->block_residuals.data(),
                                     trace->block_residuals.size() * sizeof(uint16_t));
    }
  } catch (...) {
    s.patch.unload();
    s.block.unload();
    s.merger.unload();
    throw;
  }
  impl_->output_tokens = rows / 4u;
  impl_->stats.last_encode_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  impl_->stats.patch_rows = rows;
  impl_->stats.scratch_bytes = s.scratch.reserved_bytes();
  impl_->stats.activation_bytes = s.activation_bytes();
  impl_->stats.max_streamed_weight_bytes = max_weight;
  impl_->stats.allocator_peak_used_bytes = peak_used;
  impl_->stats.allocator_peak_nonstaging_bytes = peak_nonstaging;
  impl_->stats.allocator_used_bytes = impl_->context->pooled_used_bytes();
  impl_->stats.allocator_reserved_bytes = impl_->context->reserved_bytes();
  impl_->stats.descriptor_set_allocations = impl_->context->descriptor_set_allocations();
}

DeviceTensor& ExactQwenVisionEncoder::main_output() {
  if (!impl_ || !impl_->shape || impl_->output_tokens == 0)
    throw std::logic_error("Vulkan Qwen vision encoder: no output");
  return impl_->shape->main;
}

DeviceTensor& ExactQwenVisionEncoder::deepstack_output(uint32_t slot_index) {
  if (!impl_ || !impl_->shape || impl_->output_tokens == 0 || slot_index >= 3)
    throw std::logic_error("Vulkan Qwen vision encoder: no DeepStack output");
  return impl_->shape->deep[slot_index];
}

uint32_t ExactQwenVisionEncoder::output_tokens() const noexcept {
  return impl_ ? impl_->output_tokens : 0;
}

const ExactQwenVisionStats& ExactQwenVisionEncoder::stats() const noexcept {
  static const ExactQwenVisionStats empty{};
  return impl_ ? impl_->stats : empty;
}

} // namespace slopfab::vulkan
