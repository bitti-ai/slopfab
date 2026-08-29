#include "vidfab/vulkan/vision_stage.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "vidfab/attention.h"

namespace vidfab::vulkan {
namespace {

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
  if (c.sequence == 0 || c.sequence > 16384 || (c.sequence & 3u) != 0 ||
      v.hidden_size != 1152 || v.intermediate_size != 4304 ||
      v.num_heads != 16 || v.depth != 27 || v.position_side != 48 ||
      v.merge_size != 2 || v.output_size != 5120) {
    throw std::invalid_argument(
        "Vulkan Qwen vision: invalid production configuration");
  }
}
bool same_config(const QwenVisionStageConfig& a,
                 const QwenVisionStageConfig& b) {
  return a.sequence == b.sequence &&
      a.vision.hidden_size == b.vision.hidden_size &&
      a.vision.intermediate_size == b.vision.intermediate_size &&
      a.vision.num_heads == b.vision.num_heads &&
      a.vision.depth == b.vision.depth;
}
uint32_t row_gemm_operators(uint32_t rows) {
  return (rows >= 64 ? 1u : 0u) + (rows % 64 != 0 ? 1u : 0u);
}

DeviceTensor upload_bf16(TensorContext& context, const TensorView& view,
                         const TensorLayout& layout, const char* what) {
  if (view.dtype != DType::kBF16 || view.nbytes != layout.bytes(ScalarType::kBFloat16))
    throw std::runtime_error(std::string("Vulkan Qwen vision: invalid ") + what);
  DeviceTensor result = context.allocate(layout, ScalarType::kBFloat16);
  context.upload_transient_bytes(result, view.data, view.nbytes);
  return result;
}

}  // namespace

struct ExactQwenVisionScratch::Impl {
  TensorContext* context = nullptr;
  QwenVisionStageConfig config;
  DeviceTensor normed, qkv, query, key, value, attention, branch, mlp;
  PreparedAttentionInputs prepared_attention;
  BlockedAttentionPlan attention_plan;
  DenseGemmPlan qkv_plan, projection_plan, fc1_plan, fc2_plan;

  Impl(TensorContext& owner, const QwenVisionStageConfig& c)
      : context(&owner), config(c) {
    const uint32_t rows = c.sequence;
    const uint32_t hidden = static_cast<uint32_t>(c.vision.hidden_size);
    const uint32_t heads = static_cast<uint32_t>(c.vision.num_heads);
    const uint32_t head_dim = hidden / heads;
    const uint32_t intermediate =
        static_cast<uint32_t>(c.vision.intermediate_size);
    normed = owner.allocate(matrix(rows, hidden), ScalarType::kBFloat16);
    qkv = owner.allocate(matrix(rows, 3u * hidden), ScalarType::kBFloat16);
    query = owner.allocate(three(rows, heads, head_dim), ScalarType::kBFloat16);
    key = owner.allocate(three(rows, heads, head_dim), ScalarType::kBFloat16);
    value = owner.allocate(three(rows, heads, head_dim), ScalarType::kBFloat16);
    attention = owner.allocate(three(rows, heads, head_dim), ScalarType::kBFloat16);
    branch = owner.allocate(matrix(rows, hidden), ScalarType::kBFloat16);
    mlp = owner.allocate(matrix(rows, intermediate), ScalarType::kBFloat16);
    const BlockedAttentionPlanDesc attention_desc{
        rows, heads, head_dim, exact_attention_scale(head_dim)};
    prepared_attention = PreparedAttentionInputs::create(owner, attention_desc);
    attention_plan = BlockedAttentionPlan::create(owner, attention_desc);
    auto plan = [&](uint32_t out, uint32_t in) {
      return DenseGemmPlan::create(owner, {rows, out, in,
          DenseGemmMode::kBFloat16, DenseGemmBias::kBFloat16, false});
    };
    qkv_plan = plan(3u * hidden, hidden);
    projection_plan = plan(hidden, hidden);
    fc1_plan = plan(intermediate, hidden);
    fc2_plan = plan(hidden, intermediate);
  }

  uint64_t reserved() const noexcept {
    return bytes(normed) + bytes(qkv) + bytes(query) + bytes(key) +
        bytes(value) + bytes(attention) + bytes(branch) + bytes(mlp) +
        prepared_attention.reserved_bytes();
  }
};

struct ExactQwenVisionBlockStage::Impl {
  struct Weights {
    uint32_t block = 0;
    DeviceTensor norm1_weight, norm1_bias, norm2_weight, norm2_bias;
    DeviceTensor qkv_weight, qkv_bias, projection_weight, projection_bias;
    DeviceTensor fc1_weight, fc1_bias, fc2_weight, fc2_bias;
    uint64_t resident() const noexcept {
      return bytes(norm1_weight) + bytes(norm1_bias) + bytes(norm2_weight) +
          bytes(norm2_bias) + bytes(qkv_weight) + bytes(qkv_bias) +
          bytes(projection_weight) + bytes(projection_bias) +
          bytes(fc1_weight) + bytes(fc1_bias) + bytes(fc2_weight) +
          bytes(fc2_bias);
    }
  };
  TensorContext* context = nullptr;
  QwenVisionStageConfig config;
  std::unique_ptr<Weights> weights;
  Impl(TensorContext& owner, const QwenVisionStageConfig& c)
      : context(&owner), config(c) {}
};

ExactQwenVisionScratch::ExactQwenVisionScratch() = default;
ExactQwenVisionScratch::~ExactQwenVisionScratch() = default;
ExactQwenVisionScratch::ExactQwenVisionScratch(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ExactQwenVisionScratch::ExactQwenVisionScratch(
    ExactQwenVisionScratch&&) noexcept = default;
ExactQwenVisionScratch& ExactQwenVisionScratch::operator=(
    ExactQwenVisionScratch&&) noexcept = default;
ExactQwenVisionScratch ExactQwenVisionScratch::create(
    TensorContext& context, const QwenVisionStageConfig& config) {
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
ExactQwenVisionBlockStage::ExactQwenVisionBlockStage(
    std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
ExactQwenVisionBlockStage::ExactQwenVisionBlockStage(
    ExactQwenVisionBlockStage&&) noexcept = default;
ExactQwenVisionBlockStage& ExactQwenVisionBlockStage::operator=(
    ExactQwenVisionBlockStage&&) noexcept = default;
ExactQwenVisionBlockStage ExactQwenVisionBlockStage::create(
    TensorContext& context, const QwenVisionStageConfig& config) {
  validate_config(config);
  context.require_exact_fp32_vae_normalization();
  context.require_exact_vae_pointwise();
  context.require_exact_blocked_attention();
  return ExactQwenVisionBlockStage(std::make_shared<Impl>(context, config));
}

void ExactQwenVisionBlockStage::load(
    const text::QwenVisionCheckpoint& checkpoint, uint32_t block_index) {
  if (!impl_) throw std::logic_error("Vulkan Qwen vision: empty stage");
  if (!checkpoint.checkpoint || block_index >= 27 ||
      checkpoint.config.hidden_size != impl_->config.vision.hidden_size ||
      checkpoint.config.intermediate_size !=
          impl_->config.vision.intermediate_size) {
    throw std::invalid_argument("Vulkan Qwen vision: invalid validated archive");
  }
  const SafeTensors& archive = *checkpoint.checkpoint;
  const std::string prefix = checkpoint.prefix + "blocks." +
      std::to_string(block_index) + ".";
  const uint32_t hidden = static_cast<uint32_t>(impl_->config.vision.hidden_size);
  const uint32_t intermediate =
      static_cast<uint32_t>(impl_->config.vision.intermediate_size);
  auto next = std::make_unique<Impl::Weights>();
  next->block = block_index;
  next->norm1_weight = upload_bf16(*impl_->context,
      archive.at(prefix + "norm1.weight"), vector(hidden), "norm1 weight");
  next->norm1_bias = upload_bf16(*impl_->context,
      archive.at(prefix + "norm1.bias"), vector(hidden), "norm1 bias");
  next->norm2_weight = upload_bf16(*impl_->context,
      archive.at(prefix + "norm2.weight"), vector(hidden), "norm2 weight");
  next->norm2_bias = upload_bf16(*impl_->context,
      archive.at(prefix + "norm2.bias"), vector(hidden), "norm2 bias");
  next->qkv_weight = upload_bf16(*impl_->context,
      archive.at(prefix + "attn.qkv.weight"), matrix(3u * hidden, hidden),
      "QKV weight");
  next->qkv_bias = upload_bf16(*impl_->context,
      archive.at(prefix + "attn.qkv.bias"), vector(3u * hidden), "QKV bias");
  next->projection_weight = upload_bf16(*impl_->context,
      archive.at(prefix + "attn.proj.weight"), matrix(hidden, hidden),
      "attention projection weight");
  next->projection_bias = upload_bf16(*impl_->context,
      archive.at(prefix + "attn.proj.bias"), vector(hidden),
      "attention projection bias");
  next->fc1_weight = upload_bf16(*impl_->context,
      archive.at(prefix + "mlp.linear_fc1.weight"),
      matrix(intermediate, hidden), "MLP fc1 weight");
  next->fc1_bias = upload_bf16(*impl_->context,
      archive.at(prefix + "mlp.linear_fc1.bias"), vector(intermediate),
      "MLP fc1 bias");
  next->fc2_weight = upload_bf16(*impl_->context,
      archive.at(prefix + "mlp.linear_fc2.weight"),
      matrix(hidden, intermediate), "MLP fc2 weight");
  next->fc2_bias = upload_bf16(*impl_->context,
      archive.at(prefix + "mlp.linear_fc2.bias"), vector(hidden),
      "MLP fc2 bias");
  impl_->weights = std::move(next);
}

void ExactQwenVisionBlockStage::unload() noexcept {
  if (impl_) impl_->weights.reset();
}
bool ExactQwenVisionBlockStage::loaded() const noexcept {
  return impl_ && impl_->weights;
}
uint32_t ExactQwenVisionBlockStage::block() const {
  if (!loaded()) throw std::logic_error("Vulkan Qwen vision: stage not loaded");
  return impl_->weights->block;
}
uint32_t ExactQwenVisionBlockStage::required_operators() const noexcept {
  return impl_ ? 12u + 4u * row_gemm_operators(impl_->config.sequence) : 0u;
}
uint64_t ExactQwenVisionBlockStage::persistent_bytes() const noexcept {
  return loaded() ? impl_->weights->resident() : 0;
}

void ExactQwenVisionBlockStage::record(
    TensorBatch& batch, DeviceTensor& residual, DeviceTensor& cosine,
    DeviceTensor& sine, ExactQwenVisionScratch& scratch) const {
  if (!loaded()) throw std::logic_error("Vulkan Qwen vision: stage not loaded");
  if (!scratch.impl_ || scratch.impl_->context != impl_->context ||
      !same_config(scratch.impl_->config, impl_->config) ||
      !batch.belongs_to(*impl_->context)) {
    throw std::invalid_argument("Vulkan Qwen vision: incompatible scratch/batch");
  }
  const uint32_t rows = impl_->config.sequence;
  const uint32_t hidden = static_cast<uint32_t>(impl_->config.vision.hidden_size);
  const DeviceTensorView x = residual.view(), c = cosine.view(), si = sine.view();
  if (!impl_->context->owns(residual) || !impl_->context->owns(cosine) ||
      !impl_->context->owns(sine) || x.type != ScalarType::kBFloat16 ||
      x.layout.rank != 2 || x.layout.extent[0] != rows ||
      x.layout.extent[1] != hidden || !x.layout.is_contiguous() ||
      c.type != ScalarType::kFloat32 || si.type != ScalarType::kFloat32 ||
      c.layout.rank != 2 || si.layout.rank != 2 ||
      c.layout.extent[0] != rows || si.layout.extent[0] != rows ||
      c.layout.extent[1] != 72 || si.layout.extent[1] != 72 ||
      !c.layout.is_contiguous() || !si.layout.is_contiguous() ||
      x.resource == c.resource || x.resource == si.resource ||
      c.resource == si.resource) {
    throw std::invalid_argument("Vulkan Qwen vision: invalid block inputs");
  }
  batch.require_operator_capacity(required_operators());
  auto& s = *scratch.impl_;
  auto& w = *impl_->weights;
  auto gemm = [&](const DenseGemmPlan& plan, DeviceTensor& input,
                  DeviceTensor& weight, DeviceTensor& bias,
                  DeviceTensor& output) {
    const uint32_t tiled = rows / 64u * 64u;
    if (tiled != 0) plan.record(batch, input, weight, output, tiled, 0, 0, &bias);
    if (tiled != rows)
      plan.record(batch, input, weight, output, rows - tiled, tiled, tiled, &bias);
  };
  batch.layer_norm_bf16(residual, w.norm1_weight, w.norm1_bias, s.normed,
                        1.0e-6f);
  gemm(s.qkv_plan, s.normed, w.qkv_weight, w.qkv_bias, s.qkv);
  batch.vision_split_qkv_bf16(s.qkv, s.query, s.key, s.value);
  batch.rope_neox_bf16(s.query, cosine, sine);
  batch.rope_neox_bf16(s.key, cosine, sine);
  PreparedAttentionView prepared =
      s.prepared_attention.prepare(batch, s.query, s.key, s.value);
  s.attention_plan.record(batch, prepared, s.attention);
  gemm(s.projection_plan, s.attention, w.projection_weight,
       w.projection_bias, s.branch);
  batch.text_add_residual_bf16(residual, s.branch);
  batch.layer_norm_bf16(residual, w.norm2_weight, w.norm2_bias, s.normed,
                        1.0e-6f);
  gemm(s.fc1_plan, s.normed, w.fc1_weight, w.fc1_bias, s.mlp);
  batch.vision_gelu_tanh_bf16(s.mlp);
  gemm(s.fc2_plan, s.mlp, w.fc2_weight, w.fc2_bias, s.branch);
  batch.text_add_residual_bf16(residual, s.branch);
}

}  // namespace vidfab::vulkan
