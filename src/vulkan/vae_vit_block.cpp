#include "vidfab/vulkan/vae_vit_block.h"

#include <cmath>
#include <stdexcept>
#include <utility>

#include "vidfab/gemm.h"
#include "vidfab/vulkan/gemm.h"

namespace vidfab::vulkan {
namespace {

using vae::ViTBlockConfig;
using vae::ViTBlockWeightsView;

void validate_config(const ViTBlockConfig& c) {
  if (c.sequence == 0 || c.num_patches > c.sequence || c.dim == 0 ||
      c.heads == 0 || c.head_dim != 64 || c.heads * c.head_dim != c.dim ||
      c.ffn_inner == 0 || c.rope_dim == 0 || c.rope_dim > c.head_dim ||
      (c.rope_dim & 1u) != 0 || !std::isfinite(c.epsilon) || c.epsilon <= 0.0f) {
    throw std::invalid_argument("exact Vulkan VAE ViT block: invalid configuration");
  }
}

void validate_weights(const ViTBlockWeightsView& w) {
  if (!w.norm1 || !w.norm2 || !w.scale1 || !w.scale2 || !w.qkv_weight ||
      !w.qkv_bias || !w.out_weight || !w.out_bias || !w.w1_weight ||
      !w.w1_bias || !w.w2_weight || !w.w2_bias) {
    throw std::invalid_argument("exact Vulkan VAE ViT block: null weight view");
  }
}

TensorLayout matrix(uint64_t rows, uint64_t columns) {
  const uint64_t shape[] = {rows, columns};
  return TensorLayout::contiguous(shape, 2);
}

TensorLayout three(uint64_t a, uint64_t b, uint64_t c) {
  const uint64_t shape[] = {a, b, c};
  return TensorLayout::contiguous(shape, 3);
}

TensorLayout vector(uint64_t count) {
  return TensorLayout::contiguous(&count, 1);
}

uint64_t bytes(const DeviceTensor& tensor) {
  return tensor ? tensor.layout().bytes(tensor.type()) : 0;
}

}  // namespace

struct ExactViTBlockScratch::Impl {
  TensorContext* context = nullptr;
  ViTBlockConfig config;
  DeviceTensor normed, qkv, q, k, v, q_bf16, k_bf16, v_bf16,
      attention_bf16, projected, ffn, activation;
  PreparedF16Activation prepared_dim, prepared_inner;
  PreparedAttentionInputs prepared_attention;
  DenseGemmPlan qkv_plan, out_plan, w1_plan, w2_plan;
  BlockedAttentionPlan attention_plan;

  Impl(TensorContext& owner, const ViTBlockConfig& c)
      : context(&owner), config(c) {
    const uint64_t rows = c.sequence, d = c.dim, inner = c.ffn_inner;
    normed = owner.allocate(matrix(rows, d));
    qkv = owner.allocate(matrix(rows, 3 * d));
    q = owner.allocate(three(c.heads, rows, c.head_dim));
    k = owner.allocate(three(c.heads, rows, c.head_dim));
    v = owner.allocate(three(c.heads, rows, c.head_dim));
    q_bf16 = owner.allocate(three(rows, c.heads, c.head_dim), ScalarType::kBFloat16);
    k_bf16 = owner.allocate(three(rows, c.heads, c.head_dim), ScalarType::kBFloat16);
    v_bf16 = owner.allocate(three(rows, c.heads, c.head_dim), ScalarType::kBFloat16);
    attention_bf16 = owner.allocate(three(rows, c.heads, c.head_dim),
                                    ScalarType::kBFloat16);
    projected = owner.allocate(matrix(rows, d));
    ffn = owner.allocate(matrix(rows, 2 * inner));
    activation = owner.allocate(matrix(rows, inner));
    prepared_dim = PreparedF16Activation::create(owner, c.sequence, c.dim);
    prepared_inner = PreparedF16Activation::create(owner, c.sequence, c.ffn_inner);
    BlockedAttentionPlanDesc attention_desc{c.sequence, c.heads, c.head_dim, 0.125f};
    prepared_attention = PreparedAttentionInputs::create(owner, attention_desc);
    attention_plan = BlockedAttentionPlan::create(owner, attention_desc);
    qkv_plan = DenseGemmPlan::create(owner, DenseGemmPlanDesc{
        c.sequence, 3 * c.dim, c.dim, DenseGemmMode::kFloat16Vae,
        DenseGemmBias::kNone, true});
    out_plan = DenseGemmPlan::create(owner, DenseGemmPlanDesc{
        c.sequence, c.dim, c.dim, DenseGemmMode::kFloat16Vae,
        DenseGemmBias::kNone, true});
    w1_plan = DenseGemmPlan::create(owner, DenseGemmPlanDesc{
        c.sequence, 2 * c.ffn_inner, c.dim, DenseGemmMode::kFloat16Vae,
        DenseGemmBias::kNone, true});
    w2_plan = DenseGemmPlan::create(owner, DenseGemmPlanDesc{
        c.sequence, c.dim, c.ffn_inner, DenseGemmMode::kFloat16Vae,
        DenseGemmBias::kNone, true});
  }

  uint64_t direct_bytes() const noexcept {
    return bytes(normed) + bytes(qkv) + bytes(q) + bytes(k) + bytes(v) +
        bytes(q_bf16) + bytes(k_bf16) + bytes(v_bf16) + bytes(attention_bf16) +
        bytes(projected) + bytes(ffn) + bytes(activation) +
        static_cast<uint64_t>(config.sequence) * config.dim * 2u +
        static_cast<uint64_t>(config.sequence) * config.ffn_inner * 2u +
        prepared_attention.reserved_bytes();
  }
};

struct ExactViTBlockStage::Impl {
  TensorContext* context = nullptr;
  ViTBlockConfig config;
  bool loaded = false;
  DeviceTensor norm1, norm2, scale1, scale2, qkv_weight, qkv_bias,
      out_weight, out_bias, w1_weight, w1_bias, w2_weight, w2_bias;

  struct HostState {
    DeviceTensor tokens, cosine, sine;
    ExactViTBlockScratch scratch;
  };
  std::unique_ptr<HostState> host;

  Impl(TensorContext& owner, const ViTBlockConfig& c) : context(&owner), config(c) {}

  uint64_t weight_bytes() const noexcept {
    return bytes(norm1) + bytes(norm2) + bytes(scale1) + bytes(scale2) +
        bytes(qkv_weight) + bytes(qkv_bias) + bytes(out_weight) + bytes(out_bias) +
        bytes(w1_weight) + bytes(w1_bias) + bytes(w2_weight) + bytes(w2_bias);
  }
};

ExactViTBlockScratch::ExactViTBlockScratch() = default;
ExactViTBlockScratch::~ExactViTBlockScratch() = default;
ExactViTBlockScratch::ExactViTBlockScratch(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ExactViTBlockScratch::ExactViTBlockScratch(ExactViTBlockScratch&&) noexcept = default;
ExactViTBlockScratch& ExactViTBlockScratch::operator=(ExactViTBlockScratch&&) noexcept = default;

ExactViTBlockScratch ExactViTBlockScratch::create(
    TensorContext& context, const ViTBlockConfig& config) {
  validate_config(config);
  context.require_exact_fp32_vae_normalization();
  context.require_exact_vae_pointwise();
  context.require_exact_blocked_attention();
  return ExactViTBlockScratch(std::make_shared<Impl>(context, config));
}

uint64_t ExactViTBlockScratch::reserved_bytes() const noexcept {
  return impl_ ? impl_->direct_bytes() : 0;
}

ExactViTBlockStage::ExactViTBlockStage() = default;
ExactViTBlockStage::~ExactViTBlockStage() = default;
ExactViTBlockStage::ExactViTBlockStage(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ExactViTBlockStage::ExactViTBlockStage(ExactViTBlockStage&&) noexcept = default;
ExactViTBlockStage& ExactViTBlockStage::operator=(ExactViTBlockStage&&) noexcept = default;

ExactViTBlockStage ExactViTBlockStage::create(TensorContext& context,
                                              const ViTBlockConfig& config) {
  validate_config(config);
  context.require_exact_fp32_vae_normalization();
  context.require_exact_vae_pointwise();
  context.require_exact_blocked_attention();
  return ExactViTBlockStage(std::make_shared<Impl>(context, config));
}

DeviceBackend ExactViTBlockStage::backend() const noexcept {
  return DeviceBackend::kVulkan;
}

const ViTBlockConfig& ExactViTBlockStage::config() const noexcept {
  static const ViTBlockConfig empty{};
  return impl_ ? impl_->config : empty;
}

void ExactViTBlockStage::load(const ViTBlockWeightsView& w) {
  if (!impl_) throw std::logic_error("exact Vulkan VAE ViT block: empty stage");
  validate_weights(w);
  Impl& s = *impl_;
  s.loaded = false;
  TensorContext& context = *s.context;
  const uint64_t d = s.config.dim, inner = s.config.ffn_inner;
  auto require_canonical_half = [](const uint16_t* values, uint64_t count) {
    for (uint64_t i = 0; i < count; ++i) {
      if ((values[i] & 0x7c00u) == 0 && (values[i] & 0x03ffu) != 0)
        throw std::invalid_argument(
            "exact Vulkan VAE ViT block: fp16 subnormal weight is not canonicalized");
    }
  };
  require_canonical_half(w.qkv_weight, 3 * d * d);
  require_canonical_half(w.out_weight, d * d);
  require_canonical_half(w.w1_weight, 2 * inner * d);
  require_canonical_half(w.w2_weight, d * inner);
  s.norm1 = context.allocate(vector(d)); s.norm2 = context.allocate(vector(d));
  s.scale1 = context.allocate(vector(d)); s.scale2 = context.allocate(vector(d));
  s.qkv_weight = context.allocate(matrix(3 * d, d), ScalarType::kFloat16);
  s.qkv_bias = context.allocate(vector(3 * d));
  s.out_weight = context.allocate(matrix(d, d), ScalarType::kFloat16);
  s.out_bias = context.allocate(vector(d));
  s.w1_weight = context.allocate(matrix(2 * inner, d), ScalarType::kFloat16);
  s.w1_bias = context.allocate(vector(2 * inner));
  s.w2_weight = context.allocate(matrix(d, inner), ScalarType::kFloat16);
  s.w2_bias = context.allocate(vector(d));
  context.upload(s.norm1, w.norm1, d); context.upload(s.norm2, w.norm2, d);
  context.upload(s.scale1, w.scale1, d); context.upload(s.scale2, w.scale2, d);
  context.upload_bytes(s.qkv_weight, w.qkv_weight, 3 * d * d * 2);
  context.upload(s.qkv_bias, w.qkv_bias, 3 * d);
  context.upload_bytes(s.out_weight, w.out_weight, d * d * 2);
  context.upload(s.out_bias, w.out_bias, d);
  context.upload_bytes(s.w1_weight, w.w1_weight, 2 * inner * d * 2);
  context.upload(s.w1_bias, w.w1_bias, 2 * inner);
  context.upload_bytes(s.w2_weight, w.w2_weight, d * inner * 2);
  context.upload(s.w2_bias, w.w2_bias, d);
  s.loaded = true;
}

namespace {

void record_gemm_chunks(TensorBatch& batch, DeviceTensor& input,
                        PreparedF16Activation& prepared,
                        const DenseGemmPlan& plan, DeviceTensor& weight,
                        DeviceTensor& output, uint32_t rows) {
  if (plan.description().force_scalar_order) {
    PreparedF16ActivationView view = prepared.prepare(batch, input, rows, 0);
    plan.record(batch, view, weight, output, 0);
    return;
  }
  const uint32_t tiled = rows / 64u * 64u;
  if (tiled != 0) {
    PreparedF16ActivationView view = prepared.prepare(batch, input, tiled, 0);
    plan.record(batch, view, weight, output, 0);
  }
  if (tiled != rows) {
    PreparedF16ActivationView view = prepared.prepare(batch, input, rows - tiled,
                                                       tiled);
    plan.record(batch, view, weight, output, tiled);
  }
}

}  // namespace

void ExactViTBlockStage::record(TensorBatch& batch, DeviceTensor& tokens,
                                DeviceTensor& cosine, DeviceTensor& sine,
                                ExactViTBlockScratch& scratch) const {
  if (!impl_ || !impl_->loaded)
    throw std::logic_error("exact Vulkan VAE ViT block: weights not loaded");
  if (!scratch.impl_ || scratch.impl_->context != impl_->context ||
      scratch.impl_->config.sequence != impl_->config.sequence ||
      scratch.impl_->config.dim != impl_->config.dim ||
      scratch.impl_->config.heads != impl_->config.heads ||
      scratch.impl_->config.head_dim != impl_->config.head_dim ||
      scratch.impl_->config.ffn_inner != impl_->config.ffn_inner ||
      scratch.impl_->config.rope_dim != impl_->config.rope_dim) {
    throw std::invalid_argument("exact Vulkan VAE ViT block: incompatible scratch");
  }
  const ViTBlockConfig& c = impl_->config;
  const DeviceTensorView token_view = tokens.view();
  const DeviceTensorView cosine_view = cosine.view();
  const DeviceTensorView sine_view = sine.view();
  const bool token_shape = token_view.type == ScalarType::kFloat32 &&
      token_view.layout.rank == 2 && token_view.layout.extent[0] == c.sequence &&
      token_view.layout.extent[1] == c.dim && token_view.layout.is_contiguous();
  const bool cosine_shape = cosine_view.type == ScalarType::kFloat32 &&
      cosine_view.layout.rank == 2 && cosine_view.layout.extent[0] == c.sequence &&
      cosine_view.layout.extent[1] == c.rope_dim &&
      cosine_view.layout.is_contiguous();
  const bool sine_shape = sine_view.type == ScalarType::kFloat32 &&
      sine_view.layout.rank == 2 && sine_view.layout.extent[0] == c.sequence &&
      sine_view.layout.extent[1] == c.rope_dim && sine_view.layout.is_contiguous();
  if (!token_shape || !cosine_shape || !sine_shape ||
      token_view.backend != DeviceBackend::kVulkan ||
      cosine_view.backend != DeviceBackend::kVulkan ||
      sine_view.backend != DeviceBackend::kVulkan ||
      token_view.context != cosine_view.context ||
      token_view.context != sine_view.context ||
      token_view.resource == cosine_view.resource ||
      token_view.resource == sine_view.resource ||
      cosine_view.resource == sine_view.resource) {
    throw std::invalid_argument(
        "exact Vulkan VAE ViT block: invalid activation tensors");
  }
  constexpr uint32_t kRecordedOperators = 20;
  if (batch.remaining_operator_capacity() < kRecordedOperators) {
    throw std::logic_error(
        "exact Vulkan VAE ViT block: insufficient batch capacity");
  }
  Impl& w = *impl_;
  ExactViTBlockScratch::Impl& s = *scratch.impl_;
  batch.rms_norm(tokens, w.norm1, s.normed, c.epsilon);
  record_gemm_chunks(batch, s.normed, s.prepared_dim, s.qkv_plan,
                     w.qkv_weight, s.qkv, c.sequence);
  batch.split_qkv_norm_rope_f32(s.qkv, w.qkv_bias, cosine, sine, s.q, s.k, s.v,
                                c.num_patches, c.epsilon);
  batch.heads_to_tokens_bf16(s.q, s.q_bf16, c.heads, c.sequence, c.head_dim);
  batch.heads_to_tokens_bf16(s.k, s.k_bf16, c.heads, c.sequence, c.head_dim);
  batch.heads_to_tokens_bf16(s.v, s.v_bf16, c.heads, c.sequence, c.head_dim);
  PreparedAttentionView attention_inputs = s.prepared_attention.prepare(
      batch, s.q_bf16, s.k_bf16, s.v_bf16);
  s.attention_plan.record(batch, attention_inputs, s.attention_bf16);
  batch.convert(s.attention_bf16, s.normed);
  record_gemm_chunks(batch, s.normed, s.prepared_dim, s.out_plan,
                     w.out_weight, s.projected, c.sequence);
  batch.layer_scale_residual_f32(tokens, s.projected, w.out_bias, w.scale1);
  batch.rms_norm(tokens, w.norm2, s.normed, c.epsilon);
  record_gemm_chunks(batch, s.normed, s.prepared_dim, s.w1_plan,
                     w.w1_weight, s.ffn, c.sequence);
  batch.swiglu_bias_f32(s.ffn, w.w1_bias, s.activation);
  record_gemm_chunks(batch, s.activation, s.prepared_inner, s.w2_plan,
                     w.w2_weight, s.projected, c.sequence);
  batch.layer_scale_residual_f32(tokens, s.projected, w.w2_bias, w.scale2);
}

void ExactViTBlockStage::forward(const float* tokens, const float* cosine,
                                 const float* sine, float* output) {
  if (!impl_ || !impl_->loaded)
    throw std::logic_error("exact Vulkan VAE ViT block: weights not loaded");
  if (!tokens || !cosine || !sine || !output)
    throw std::invalid_argument("exact Vulkan VAE ViT block: null activation");
  Impl& s = *impl_;
  if (!s.host) {
    auto host = std::make_unique<Impl::HostState>();
    host->tokens = s.context->allocate(matrix(s.config.sequence, s.config.dim));
    host->cosine = s.context->allocate(matrix(s.config.sequence, s.config.rope_dim));
    host->sine = s.context->allocate(matrix(s.config.sequence, s.config.rope_dim));
    host->scratch = ExactViTBlockScratch::create(*s.context, s.config);
    s.host = std::move(host);
  }
  const uint64_t token_count = static_cast<uint64_t>(s.config.sequence) * s.config.dim;
  const uint64_t rope_count = static_cast<uint64_t>(s.config.sequence) * s.config.rope_dim;
  s.context->upload(s.host->tokens, tokens, token_count);
  s.context->upload(s.host->cosine, cosine, rope_count);
  s.context->upload(s.host->sine, sine, rope_count);
  TensorBatch batch = s.context->begin_batch();
  record(batch, s.host->tokens, s.host->cosine, s.host->sine, s.host->scratch);
  batch.submit().wait();
  s.context->download(s.host->tokens, output, token_count);
}

uint64_t ExactViTBlockStage::persistent_bytes() const noexcept {
  return impl_ ? impl_->weight_bytes() : 0;
}

uint64_t ExactViTBlockStage::peak_device_bytes() const noexcept {
  if (!impl_) return 0;
  if (impl_->host) {
    return impl_->weight_bytes() + impl_->host->scratch.reserved_bytes() +
        bytes(impl_->host->tokens) + bytes(impl_->host->cosine) + bytes(impl_->host->sine);
  }
  return impl_->weight_bytes();
}

}  // namespace vidfab::vulkan
