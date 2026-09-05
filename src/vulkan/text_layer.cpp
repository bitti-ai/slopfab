#include "slopfab/vulkan/text_layer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "slopfab/attention.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/vulkan/linear.h"

namespace slopfab::vulkan {
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
uint64_t tensor_bytes(const DeviceTensor& tensor) {
  return tensor ? tensor.layout().bytes(tensor.type()) : 0;
}
uint64_t checked_product(uint64_t left, uint64_t right, const char* what) {
  if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
    throw std::overflow_error(std::string("Vulkan Qwen layer: ") + what +
                              " overflow");
  }
  return left * right;
}
bool same_model_shape(const QwenTextLayerConfig& left,
                      const QwenTextLayerConfig& right) {
  const auto& a = left.encoder;
  const auto& b = right.encoder;
  return left.sequence == right.sequence &&
      a.hidden_size == b.hidden_size &&
      a.num_attention_heads == b.num_attention_heads &&
      a.num_key_value_heads == b.num_key_value_heads &&
      a.head_dim == b.head_dim &&
      a.intermediate_size == b.intermediate_size &&
      a.rms_norm_eps == b.rms_norm_eps;
}
void validate_config(const QwenTextLayerConfig& config) {
  const text::EncoderConfig& c = config.encoder;
  if (config.sequence == 0 || config.sequence > 8192 ||
      c.hidden_size != 5120 || c.num_attention_heads != 64 ||
      c.num_key_value_heads != 8 || c.head_dim != 128 ||
      c.intermediate_size <= 0 || c.num_layers <= 0 ||
      c.rms_norm_eps != 1.0e-6f ||
      c.intermediate_size % 256 != 0) {
    throw std::invalid_argument("Vulkan Qwen layer: invalid production configuration");
  }
}
text::EncoderConfig resolved_encoder(const SafeTensors& checkpoint,
                                     const QwenTextLayerConfig& config) {
  text::EncoderConfig result = config.encoder;
  const text::WeightFormat detected = text::detect_weight_format(checkpoint);
  if (result.format != text::WeightFormat::kAuto && result.format != detected) {
    throw std::runtime_error(
        "Vulkan Qwen layer: requested format differs from checkpoint metadata");
  }
  result.format = detected;
  return result;
}

void validate_target_layer(const SafeTensors& checkpoint,
                           const text::EncoderConfig& config,
                           uint32_t layer,
                           bool verify_manifest) {
  const std::string prefix =
      "model.layers." + std::to_string(layer) + ".";
  size_t expected_count = 0;
  for (int i = 0; i < text::kLayerTensorCount; ++i) {
    const text::TensorSpec spec = text::layer_tensor_spec(
        config, static_cast<text::LayerTensor>(i));
    if (!spec.present()) continue;
    ++expected_count;
    const std::string name = prefix + spec.suffix;
    const TensorView& view = checkpoint.at(name);
    const std::vector<int64_t> expected = spec.dim1 == 0
        ? std::vector<int64_t>{spec.dim0}
        : std::vector<int64_t>{spec.dim0, spec.dim1};
    const uint64_t elements = spec.dim1 == 0
        ? static_cast<uint64_t>(spec.dim0)
        : checked_product(static_cast<uint64_t>(spec.dim0),
                          static_cast<uint64_t>(spec.dim1), "tensor elements");
    const uint64_t expected_bytes = checked_product(
        elements, dtype_size(spec.dtype), "tensor bytes");
    if (view.dtype != spec.dtype || view.shape != expected ||
        view.nbytes != expected_bytes) {
      throw std::runtime_error("Vulkan Qwen layer: invalid tensor '" + name + "'");
    }
  }
  static constexpr const char* kLinears[] = {
      "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
      "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
      "mlp.down_proj"};
  const bool nvfp4 = config.format == text::WeightFormat::kNVFP4Awq;
  const char* canonical_metadata = nvfp4
      ? "{\"format\": \"nvfp4\", \"full_precision_matrix_mult\": true}"
      : "{\"format\": \"int8_tensorwise\", \"convrot\": true, "
        "\"convrot_groupsize\": 256}";
  const size_t canonical_metadata_bytes = std::strlen(canonical_metadata);
  for (const char* linear : kLinears) {
    const std::string name = prefix + linear;
    const TensorView& metadata = checkpoint.at(name + ".comfy_quant");
    ++expected_count;
    if (metadata.dtype != DType::kU8 ||
        metadata.shape != std::vector<int64_t>{
            static_cast<int64_t>(canonical_metadata_bytes)} ||
        metadata.nbytes != canonical_metadata_bytes ||
        std::memcmp(metadata.data, canonical_metadata,
                    canonical_metadata_bytes) != 0) {
      throw std::runtime_error(
          "Vulkan Qwen layer: non-canonical comfy_quant for '" + name + "'");
    }
    if (nvfp4) ++expected_count;
  }
  if (nvfp4) {
    (void)text::read_global_scales(checkpoint, config,
                                   static_cast<int>(layer));
  }

  if (!verify_manifest) return;
  size_t actual_count = 0;
  for (const auto& entry : checkpoint.tensors()) {
    if (entry.first.rfind(prefix, 0) != 0) continue;
    ++actual_count;
    const std::string suffix = entry.first.substr(prefix.size());
    bool expected = false;
    for (int i = 0; i < text::kLayerTensorCount && !expected; ++i) {
      const text::TensorSpec spec = text::layer_tensor_spec(
          config, static_cast<text::LayerTensor>(i));
      expected = spec.present() && suffix == spec.suffix;
    }
    for (const char* linear : kLinears) {
      expected = expected || suffix == std::string(linear) + ".comfy_quant";
      expected = expected || (nvfp4 &&
          suffix == std::string(linear) + ".weight_scale_2");
    }
    if (!expected) {
      throw std::runtime_error(
          "Vulkan Qwen layer: unexpected target tensor '" + entry.first + "'");
    }
  }
  if (actual_count != expected_count) {
    throw std::runtime_error(
        "Vulkan Qwen layer: incomplete target-layer tensor manifest");
  }
}

void validate_embedding_manifest(const SafeTensors& checkpoint,
                                 const text::EncoderConfig& config) {
  const TensorView& weight = checkpoint.at("model.embed_tokens.weight");
  const uint64_t elements = checked_product(
      static_cast<uint64_t>(config.vocab_size),
      static_cast<uint64_t>(config.hidden_size), "embedding elements");
  const bool nvfp4 = config.format == text::WeightFormat::kNVFP4Awq;
  const DType expected_dtype = nvfp4 ? DType::kI8 : DType::kBF16;
  const uint64_t expected_bytes = checked_product(
      elements, dtype_size(expected_dtype), "embedding bytes");
  if (weight.dtype != expected_dtype ||
      weight.shape != std::vector<int64_t>{config.vocab_size,
                                           config.hidden_size} ||
      weight.nbytes != expected_bytes) {
    throw std::runtime_error(
        "Vulkan Qwen encoder: invalid model.embed_tokens.weight");
  }
  if (!nvfp4) return;

  const TensorView& scale = checkpoint.at("model.embed_tokens.weight_scale");
  const uint64_t scale_bytes = checked_product(
      static_cast<uint64_t>(config.vocab_size), sizeof(float),
      "embedding scale bytes");
  if (scale.dtype != DType::kF32 ||
      scale.shape != std::vector<int64_t>{config.vocab_size, 1} ||
      scale.nbytes != scale_bytes) {
    throw std::runtime_error(
        "Vulkan Qwen encoder: invalid model.embed_tokens.weight_scale");
  }
  const auto* values = static_cast<const float*>(scale.data);
  for (int64_t row = 0; row < config.vocab_size; ++row) {
    if (!std::isfinite(values[row]) || values[row] <= 0.0f) {
      throw std::runtime_error(
          "Vulkan Qwen encoder: non-positive embedding weight_scale");
    }
  }

  static constexpr char kEmbeddingMetadata[] =
      "{\"format\": \"int8_tensorwise\"}";
  const TensorView& metadata =
      checkpoint.at("model.embed_tokens.comfy_quant");
  constexpr size_t kMetadataBytes = sizeof(kEmbeddingMetadata) - 1;
  if (metadata.dtype != DType::kU8 ||
      metadata.shape != std::vector<int64_t>{
          static_cast<int64_t>(kMetadataBytes)} ||
      metadata.nbytes != kMetadataBytes ||
      std::memcmp(metadata.data, kEmbeddingMetadata, kMetadataBytes) != 0) {
    throw std::runtime_error(
        "Vulkan Qwen encoder: non-canonical embedding comfy_quant");
  }
}

struct Projection {
  LinearWeight weight;
  uint32_t out = 0;
  uint32_t in = 0;
  uint64_t bytes() const noexcept { return weight.resident_bytes(); }
};

Projection load_projection(TensorContext& context, const SafeTensors& checkpoint,
                           const text::EncoderConfig& config, uint32_t layer,
                           text::LayerTensor weight_kind,
                           text::LayerTensor scale_kind,
                           text::LayerTensor pre_scale_kind,
                           uint32_t global_index, uint32_t out, uint32_t in,
                           const text::LayerGlobalScales& globals) {
  const std::string prefix = "model.layers." + std::to_string(layer) + ".";
  const text::TensorSpec weight_spec =
      text::layer_tensor_spec(config, weight_kind);
  const text::TensorSpec scale_spec =
      text::layer_tensor_spec(config, scale_kind);
  const TensorView& weight_view = checkpoint.at(prefix + weight_spec.suffix);
  const TensorView& scale_view = checkpoint.at(prefix + scale_spec.suffix);
  LinearWeightUpload upload;
  upload.out_features = out;
  upload.in_features = in;
  upload.data = weight_view.data;
  upload.data_bytes = weight_view.nbytes;
  std::vector<float> scales;
  if (config.format == text::WeightFormat::kI8ConvRot) {
    scales = to_f32(scale_view);
    if (scales.size() != out) {
      throw std::runtime_error("Vulkan Qwen layer: invalid I8 scale count");
    }
    upload.format = LinearWeightFormat::kInt8;
    upload.weight_scale = scales.data();
    upload.weight_scale_count = scales.size();
    upload.convrot = true;
    upload.convrot_group = 256;
  } else {
    upload.format = LinearWeightFormat::kNVFloat4;
    upload.block_scale = static_cast<const uint8_t*>(scale_view.data);
    upload.block_scale_count = scale_view.nbytes;
    upload.global_scale = globals.value[global_index];
    upload.full_precision_matrix_mult = true;
    if (pre_scale_kind != text::LayerTensor::kCount) {
      const text::TensorSpec pre_spec =
          text::layer_tensor_spec(config, pre_scale_kind);
      const TensorView& pre = checkpoint.at(prefix + pre_spec.suffix);
      upload.pre_quant_scale_bf16 =
          static_cast<const uint16_t*>(pre.data);
      upload.pre_quant_scale_count = pre.nbytes / sizeof(uint16_t);
    }
  }
  Projection result;
  result.out = out;
  result.in = in;
  result.weight = LinearWeight::upload(context, upload);
  return result;
}

DeviceTensor upload_norm(TensorContext& context, const SafeTensors& checkpoint,
                         const std::string& name, uint32_t count) {
  const TensorView& view = checkpoint.at(name);
  if (view.dtype != DType::kBF16 || view.nbytes != uint64_t(count) * 2) {
    throw std::runtime_error("Vulkan Qwen layer: invalid norm '" + name + "'");
  }
  DeviceTensor result = context.allocate(vector(count), ScalarType::kBFloat16);
  context.upload_transient_bytes(result, view.data, view.nbytes);
  return result;
}

uint32_t row_gemm_operators(uint32_t rows) {
  return (rows >= 64 ? 1u : 0u) + (rows % 64 != 0 ? 1u : 0u);
}
uint32_t projection_operators(const Projection& projection, uint32_t rows) {
  return 1u + row_gemm_operators(rows) +
      (projection.weight.has_pre_quant_scale() ? 1u : 0u) +
      (projection.weight.applies_convrot() ? 1u : 0u);
}

void validate_tap(TensorContext& context, DeviceTensor* tensor,
                  ScalarType type, const TensorLayout& layout,
                  const char* name,
                  std::array<DeviceTensorView, 14>& views,
                  size_t& view_count) {
  if (!tensor) return;
  if (!context.owns(*tensor)) {
    throw std::invalid_argument(std::string("Vulkan Qwen layer: foreign ") +
                                name + " tap");
  }
  const DeviceTensorView view = tensor->view();
  if (view.type != type || view.layout.rank != layout.rank ||
      view.layout.extent != layout.extent || !view.layout.is_contiguous()) {
    throw std::invalid_argument(std::string("Vulkan Qwen layer: invalid ") +
                                name + " tap");
  }
  for (size_t i = 0; i < view_count; ++i) {
    if (views[i].resource == view.resource) {
      throw std::invalid_argument("Vulkan Qwen layer: aliased replay tap");
    }
  }
  if (view_count == views.size()) {
    throw std::logic_error("Vulkan Qwen layer: tap validation overflow");
  }
  views[view_count++] = view;
}

}  // namespace

struct ExactQwenTextLayerScratch::Impl {
  TensorContext* context = nullptr;
  QwenTextLayerConfig config;
  DeviceTensor normed, query, key, value, attention, branch;
  DeviceTensor gate, up, activation;
  DeviceTensor hidden_transform, query_transform, ffn_transform;
  StreamedNVFP4WeightCache cache;
  DenseGemmPlan q_plan, k_plan, v_plan, o_plan;
  DenseGemmPlan gate_plan, up_plan, down_plan;
  CausalGQAAttentionPlan attention_plan;

  Impl(TensorContext& owner, const QwenTextLayerConfig& c)
      : context(&owner), config(c) {
    const uint32_t rows = c.sequence;
    const uint32_t hidden = static_cast<uint32_t>(c.encoder.hidden_size);
    const uint32_t q_width = static_cast<uint32_t>(
        c.encoder.num_attention_heads * c.encoder.head_dim);
    const uint32_t kv_width = static_cast<uint32_t>(
        c.encoder.num_key_value_heads * c.encoder.head_dim);
    const uint32_t ffn = static_cast<uint32_t>(c.encoder.intermediate_size);
    normed = owner.allocate(matrix(rows, hidden), ScalarType::kBFloat16);
    query = owner.allocate(three(rows, c.encoder.num_attention_heads,
                                 c.encoder.head_dim), ScalarType::kBFloat16);
    key = owner.allocate(three(rows, c.encoder.num_key_value_heads,
                               c.encoder.head_dim), ScalarType::kBFloat16);
    value = owner.allocate(three(rows, c.encoder.num_key_value_heads,
                                 c.encoder.head_dim), ScalarType::kBFloat16);
    attention = owner.allocate(three(rows, c.encoder.num_attention_heads,
                                     c.encoder.head_dim), ScalarType::kBFloat16);
    branch = owner.allocate(matrix(rows, hidden), ScalarType::kBFloat16);
    gate = owner.allocate(matrix(rows, ffn), ScalarType::kBFloat16);
    up = owner.allocate(matrix(rows, ffn), ScalarType::kBFloat16);
    activation = owner.allocate(matrix(rows, ffn), ScalarType::kBFloat16);
    hidden_transform = owner.allocate(matrix(rows, hidden), ScalarType::kBFloat16);
    query_transform = owner.allocate(
        three(rows, c.encoder.num_attention_heads, c.encoder.head_dim),
        ScalarType::kBFloat16);
    ffn_transform = owner.allocate(matrix(rows, ffn), ScalarType::kBFloat16);
    const uint64_t largest = std::max({
        checked_product(q_width, hidden, "Q matrix"),
        checked_product(kv_width, hidden, "K matrix"),
        checked_product(hidden, q_width, "O matrix"),
        checked_product(ffn, hidden, "gate matrix"),
        checked_product(hidden, ffn, "down matrix")});
    cache = StreamedNVFP4WeightCache::create(owner, largest);
    auto plan = [&](uint32_t out, uint32_t in) {
      return DenseGemmPlan::create(owner, {rows, out, in,
          DenseGemmMode::kBFloat16, DenseGemmBias::kNone, false});
    };
    q_plan = plan(q_width, hidden);
    k_plan = plan(kv_width, hidden);
    v_plan = plan(kv_width, hidden);
    o_plan = plan(hidden, q_width);
    gate_plan = plan(ffn, hidden);
    up_plan = plan(ffn, hidden);
    down_plan = plan(hidden, ffn);
    attention_plan = CausalGQAAttentionPlan::create(owner, {
        rows, static_cast<uint32_t>(c.encoder.num_attention_heads),
        static_cast<uint32_t>(c.encoder.num_key_value_heads),
        static_cast<uint32_t>(c.encoder.head_dim),
        exact_attention_scale(c.encoder.head_dim)});
  }

  uint64_t reserved() const noexcept {
    return tensor_bytes(normed) + tensor_bytes(query) + tensor_bytes(key) +
        tensor_bytes(value) + tensor_bytes(attention) + tensor_bytes(branch) +
        tensor_bytes(gate) + tensor_bytes(up) + tensor_bytes(activation) +
        tensor_bytes(hidden_transform) + tensor_bytes(query_transform) +
        tensor_bytes(ffn_transform) + cache.dense_bytes();
  }
};

struct ExactQwenTextLayerStage::Impl {
  struct Weights {
    text::WeightFormat format = text::WeightFormat::kAuto;
    DeviceTensor input_norm, post_norm, q_norm, k_norm;
    Projection q, k, v, o, gate, up, down;
    uint64_t bytes() const noexcept {
      return tensor_bytes(input_norm) + tensor_bytes(post_norm) +
          tensor_bytes(q_norm) + tensor_bytes(k_norm) + q.bytes() + k.bytes() +
          v.bytes() + o.bytes() + gate.bytes() + up.bytes() + down.bytes();
    }
  };
  TensorContext* context = nullptr;
  QwenTextLayerConfig config;
  std::unique_ptr<Weights> weights;
  Impl(TensorContext& owner, const QwenTextLayerConfig& c)
      : context(&owner), config(c) {}
};

ExactQwenTextLayerScratch::ExactQwenTextLayerScratch() = default;
ExactQwenTextLayerScratch::~ExactQwenTextLayerScratch() = default;
ExactQwenTextLayerScratch::ExactQwenTextLayerScratch(
    std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
ExactQwenTextLayerScratch::ExactQwenTextLayerScratch(
    ExactQwenTextLayerScratch&&) noexcept = default;
ExactQwenTextLayerScratch& ExactQwenTextLayerScratch::operator=(
    ExactQwenTextLayerScratch&&) noexcept = default;
ExactQwenTextLayerScratch ExactQwenTextLayerScratch::create(
    TensorContext& context, const QwenTextLayerConfig& config) {
  validate_config(config);
  context.require_exact_fp32_vae_normalization();
  context.require_exact_vae_pointwise();
  context.require_exact_causal_gqa_attention();
  return ExactQwenTextLayerScratch(
      std::make_shared<Impl>(context, config));
}
uint64_t ExactQwenTextLayerScratch::reserved_bytes() const noexcept {
  return impl_ ? impl_->reserved() : 0;
}
uint64_t ExactQwenTextLayerScratch::dense_cache_bytes() const noexcept {
  return impl_ ? impl_->cache.dense_bytes() : 0;
}

ExactQwenTextLayerStage::ExactQwenTextLayerStage() = default;
ExactQwenTextLayerStage::~ExactQwenTextLayerStage() = default;
ExactQwenTextLayerStage::ExactQwenTextLayerStage(
    std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
ExactQwenTextLayerStage::ExactQwenTextLayerStage(
    ExactQwenTextLayerStage&&) noexcept = default;
ExactQwenTextLayerStage& ExactQwenTextLayerStage::operator=(
    ExactQwenTextLayerStage&&) noexcept = default;
ExactQwenTextLayerStage ExactQwenTextLayerStage::create(
    TensorContext& context, const QwenTextLayerConfig& config) {
  validate_config(config);
  context.require_exact_fp32_vae_normalization();
  context.require_exact_vae_pointwise();
  context.require_exact_causal_gqa_attention();
  return ExactQwenTextLayerStage(std::make_shared<Impl>(context, config));
}

void ExactQwenTextLayerStage::validate_checkpoint(
    const SafeTensors& checkpoint, uint32_t layer,
    const QwenTextLayerConfig& config) {
  validate_config(config);
  if (layer >= static_cast<uint32_t>(config.encoder.num_layers)) {
    throw std::out_of_range("Vulkan Qwen layer: layer is out of range");
  }
  text::EncoderConfig resolved = resolved_encoder(checkpoint, config);
  validate_target_layer(checkpoint, resolved, layer, true);
}

void ExactQwenTextLayerStage::validate_archive(
    const SafeTensors& checkpoint, const QwenTextLayerConfig& config) {
  validate_config(config);
  text::EncoderConfig resolved = resolved_encoder(checkpoint, config);
  text::validate_checkpoint(checkpoint, resolved);
  validate_embedding_manifest(checkpoint, resolved);
  // validate_checkpoint() has already made the aggregate layer manifest exact,
  // so these strict canonical checks need only keyed lookups. In particular,
  // they do not rescan a vision-heavy archive 50 times.
  for (uint32_t layer = 0;
       layer < static_cast<uint32_t>(resolved.num_layers); ++layer) {
    validate_target_layer(checkpoint, resolved, layer, false);
  }
}

void ExactQwenTextLayerStage::load(const SafeTensors& checkpoint,
                                   uint32_t layer) {
  if (!impl_) throw std::logic_error("Vulkan Qwen layer: empty stage");
  validate_checkpoint(checkpoint, layer, impl_->config);
  text::EncoderConfig resolved = resolved_encoder(checkpoint, impl_->config);
  const text::LayerGlobalScales globals =
      text::read_global_scales(checkpoint, resolved, static_cast<int>(layer));
  const uint32_t hidden = static_cast<uint32_t>(resolved.hidden_size);
  const uint32_t q_width = static_cast<uint32_t>(
      resolved.num_attention_heads * resolved.head_dim);
  const uint32_t kv_width = static_cast<uint32_t>(
      resolved.num_key_value_heads * resolved.head_dim);
  const uint32_t ffn = static_cast<uint32_t>(resolved.intermediate_size);
  const std::string prefix =
      "model.layers." + std::to_string(layer) + ".";

  // All host archive metadata was checked above. Device construction is held
  // in a temporary owner so an upload/allocation failure preserves the active
  // layer and releases every partial resource.
  auto next = std::make_unique<Impl::Weights>();
  next->format = resolved.format;
  next->input_norm = upload_norm(*impl_->context, checkpoint,
      prefix + "input_layernorm.weight", hidden);
  next->post_norm = upload_norm(*impl_->context, checkpoint,
      prefix + "post_attention_layernorm.weight", hidden);
  next->q_norm = upload_norm(*impl_->context, checkpoint,
      prefix + "self_attn.q_norm.weight", resolved.head_dim);
  next->k_norm = upload_norm(*impl_->context, checkpoint,
      prefix + "self_attn.k_norm.weight", resolved.head_dim);
  next->q = load_projection(*impl_->context, checkpoint, resolved, layer,
      text::LayerTensor::kQWeight, text::LayerTensor::kQScale,
      text::LayerTensor::kCount, 0, q_width, hidden, globals);
  next->k = load_projection(*impl_->context, checkpoint, resolved, layer,
      text::LayerTensor::kKWeight, text::LayerTensor::kKScale,
      text::LayerTensor::kCount, 1, kv_width, hidden, globals);
  next->v = load_projection(*impl_->context, checkpoint, resolved, layer,
      text::LayerTensor::kVWeight, text::LayerTensor::kVScale,
      text::LayerTensor::kCount, 2, kv_width, hidden, globals);
  next->o = load_projection(*impl_->context, checkpoint, resolved, layer,
      text::LayerTensor::kOWeight, text::LayerTensor::kOScale,
      text::LayerTensor::kOPreQuantScale, 3, hidden, q_width, globals);
  next->gate = load_projection(*impl_->context, checkpoint, resolved, layer,
      text::LayerTensor::kGateWeight, text::LayerTensor::kGateScale,
      text::LayerTensor::kCount, 4, ffn, hidden, globals);
  next->up = load_projection(*impl_->context, checkpoint, resolved, layer,
      text::LayerTensor::kUpWeight, text::LayerTensor::kUpScale,
      text::LayerTensor::kCount, 5, ffn, hidden, globals);
  next->down = load_projection(*impl_->context, checkpoint, resolved, layer,
      text::LayerTensor::kDownWeight, text::LayerTensor::kDownScale,
      text::LayerTensor::kDownPreQuantScale, 6, hidden, ffn, globals);
  impl_->weights = std::move(next);
}

namespace {

template <typename Scratch>
DeviceTensor& transform_for(Scratch& scratch, uint32_t width) {
  const uint32_t hidden = static_cast<uint32_t>(scratch.config.encoder.hidden_size);
  const uint32_t query = static_cast<uint32_t>(
      scratch.config.encoder.num_attention_heads *
      scratch.config.encoder.head_dim);
  const uint32_t ffn = static_cast<uint32_t>(
      scratch.config.encoder.intermediate_size);
  if (width == hidden) return scratch.hidden_transform;
  if (width == query) return scratch.query_transform;
  if (width == ffn) return scratch.ffn_transform;
  throw std::logic_error("Vulkan Qwen layer: missing activation transform arena");
}

template <typename Scratch>
void record_projection(TensorBatch& batch, Projection& projection,
                       const DenseGemmPlan& plan, Scratch& scratch,
                       DeviceTensor& input, DeviceTensor& output,
                       uint32_t rows,
                       DeviceTensor* shared_transformed_input = nullptr) {
  DeviceTensor* source = shared_transformed_input
      ? shared_transformed_input : &input;
  if (projection.weight.has_pre_quant_scale() &&
      projection.weight.applies_convrot()) {
    throw std::logic_error("Vulkan Qwen layer: unsupported combined transform");
  }
  if (shared_transformed_input) {
    if (!projection.weight.applies_convrot() ||
        projection.weight.has_pre_quant_scale()) {
      throw std::logic_error(
          "Vulkan Qwen layer: invalid shared projection transform");
    }
  } else if (projection.weight.has_pre_quant_scale()) {
    DeviceTensor& transformed = transform_for(scratch, projection.in);
    projection.weight.apply_pre_quant_scale(batch, input, transformed);
    source = &transformed;
  } else if (projection.weight.applies_convrot()) {
    DeviceTensor& transformed = transform_for(scratch, projection.in);
    projection.weight.apply_convrot(batch, input, transformed);
    source = &transformed;
  }
  PreparedNVFP4WeightView prepared =
      scratch.cache.prepare(batch, projection.weight, plan);
  const uint32_t tiled = rows / 64 * 64;
  if (tiled != 0) plan.record(batch, *source, prepared, output, tiled);
  if (tiled != rows) {
    plan.record(batch, *source, prepared, output, rows - tiled, tiled, tiled);
  }
}

}  // namespace

void ExactQwenTextLayerStage::record(
    TensorBatch& batch, DeviceTensor& tokens, DeviceTensor& cosine,
    DeviceTensor& sine, ExactQwenTextLayerScratch& scratch,
    const QwenTextLayerTaps* taps) const {
  if (!impl_ || !impl_->weights) {
    throw std::logic_error("Vulkan Qwen layer: stage is not loaded");
  }
  if (!scratch.impl_ || scratch.impl_->context != impl_->context ||
      !same_model_shape(scratch.impl_->config, impl_->config)) {
    throw std::invalid_argument("Vulkan Qwen layer: incompatible scratch");
  }
  const auto token_view = tokens.view();
  const auto cosine_view = cosine.view();
  const auto sine_view = sine.view();
  const uint32_t rows = impl_->config.sequence;
  const uint32_t hidden = static_cast<uint32_t>(impl_->config.encoder.hidden_size);
  if (!impl_->context->owns(tokens) || !impl_->context->owns(cosine) ||
      !impl_->context->owns(sine) ||
      token_view.type != ScalarType::kBFloat16 ||
      token_view.layout.rank != 2 || token_view.layout.extent[0] != rows ||
      token_view.layout.extent[1] != hidden ||
      !token_view.layout.is_contiguous() ||
      cosine_view.type != ScalarType::kFloat32 ||
      sine_view.type != ScalarType::kFloat32 ||
      cosine_view.layout.rank != 2 || sine_view.layout.rank != 2 ||
      cosine_view.layout.extent[0] != rows ||
      sine_view.layout.extent[0] != rows ||
      cosine_view.layout.extent[1] != 128 ||
      sine_view.layout.extent[1] != 128 ||
      !cosine_view.layout.is_contiguous() ||
      !sine_view.layout.is_contiguous() ||
      token_view.resource == cosine_view.resource ||
      token_view.resource == sine_view.resource ||
      cosine_view.resource == sine_view.resource) {
    throw std::invalid_argument("Vulkan Qwen layer: invalid activation tensors");
  }
  auto& s = *scratch.impl_;
  const auto& c = impl_->config.encoder;
  const uint32_t ffn = static_cast<uint32_t>(c.intermediate_size);
  std::array<DeviceTensorView, 14> tap_views{};
  tap_views[0] = token_view;
  tap_views[1] = cosine_view;
  tap_views[2] = sine_view;
  size_t tap_view_count = 3;
  if (taps) {
    validate_tap(*impl_->context, taps->input_norm, ScalarType::kBFloat16,
                 matrix(rows, hidden), "input norm", tap_views, tap_view_count);
    validate_tap(*impl_->context, taps->query, ScalarType::kBFloat16,
                 three(rows, c.num_attention_heads, c.head_dim), "query",
                 tap_views, tap_view_count);
    validate_tap(*impl_->context, taps->key, ScalarType::kBFloat16,
                 three(rows, c.num_key_value_heads, c.head_dim), "key",
                 tap_views, tap_view_count);
    validate_tap(*impl_->context, taps->value, ScalarType::kBFloat16,
                 three(rows, c.num_key_value_heads, c.head_dim), "value",
                 tap_views, tap_view_count);
    validate_tap(*impl_->context, taps->attention, ScalarType::kBFloat16,
                 three(rows, c.num_attention_heads, c.head_dim), "attention",
                 tap_views, tap_view_count);
    validate_tap(*impl_->context, taps->attention_residual, ScalarType::kBFloat16,
                 matrix(rows, hidden), "attention residual", tap_views,
                 tap_view_count);
    validate_tap(*impl_->context, taps->post_attention_norm, ScalarType::kBFloat16,
                 matrix(rows, hidden), "post-attention norm", tap_views,
                 tap_view_count);
    validate_tap(*impl_->context, taps->gate, ScalarType::kBFloat16,
                 matrix(rows, ffn), "gate", tap_views, tap_view_count);
    validate_tap(*impl_->context, taps->up, ScalarType::kBFloat16,
                 matrix(rows, ffn), "up", tap_views, tap_view_count);
    validate_tap(*impl_->context, taps->activation, ScalarType::kBFloat16,
                 matrix(rows, ffn), "activation", tap_views, tap_view_count);
    validate_tap(*impl_->context, taps->final_residual, ScalarType::kBFloat16,
                 matrix(rows, hidden), "final residual", tap_views,
                 tap_view_count);
  }
  if (!batch.belongs_to(*impl_->context)) {
    throw std::invalid_argument("Vulkan Qwen layer: batch belongs to another context");
  }
  const uint32_t needed = required_operators(taps);
  batch.require_operator_capacity(needed);

  auto& w = *impl_->weights;
  batch.rms_norm_bf16(tokens, w.input_norm, s.normed, c.rms_norm_eps);
  if (taps && taps->input_norm) batch.copy(s.normed, *taps->input_norm);
  DeviceTensor* shared_qkv = nullptr;
  if (w.q.weight.applies_convrot() && w.k.weight.applies_convrot() &&
      w.v.weight.applies_convrot()) {
    w.q.weight.apply_convrot(batch, s.normed, s.hidden_transform);
    shared_qkv = &s.hidden_transform;
  }
  record_projection(batch, w.q, s.q_plan, s, s.normed, s.query, rows,
                    shared_qkv);
  record_projection(batch, w.k, s.k_plan, s, s.normed, s.key, rows,
                    shared_qkv);
  record_projection(batch, w.v, s.v_plan, s, s.normed, s.value, rows,
                    shared_qkv);
  batch.rms_norm_heads_bf16(s.query, w.q_norm, s.query,
      c.num_attention_heads, c.head_dim, c.rms_norm_eps);
  batch.rms_norm_heads_bf16(s.key, w.k_norm, s.key,
      c.num_key_value_heads, c.head_dim, c.rms_norm_eps);
  batch.rope_neox_bf16(s.query, cosine, sine);
  batch.rope_neox_bf16(s.key, cosine, sine);
  if (taps) {
    if (taps->query) batch.copy(s.query, *taps->query);
    if (taps->key) batch.copy(s.key, *taps->key);
    if (taps->value) batch.copy(s.value, *taps->value);
  }
  s.attention_plan.record(batch, s.query, s.key, s.value, s.attention);
  if (taps && taps->attention) batch.copy(s.attention, *taps->attention);
  record_projection(batch, w.o, s.o_plan, s, s.attention, s.branch, rows);
  batch.text_add_residual_bf16(tokens, s.branch);
  if (taps && taps->attention_residual)
    batch.copy(tokens, *taps->attention_residual);
  batch.rms_norm_bf16(tokens, w.post_norm, s.normed, c.rms_norm_eps);
  if (taps && taps->post_attention_norm)
    batch.copy(s.normed, *taps->post_attention_norm);
  DeviceTensor* shared_gate_up = nullptr;
  if (w.gate.weight.applies_convrot() && w.up.weight.applies_convrot()) {
    w.gate.weight.apply_convrot(batch, s.normed, s.hidden_transform);
    shared_gate_up = &s.hidden_transform;
  }
  record_projection(batch, w.gate, s.gate_plan, s, s.normed, s.gate, rows,
                    shared_gate_up);
  record_projection(batch, w.up, s.up_plan, s, s.normed, s.up, rows,
                    shared_gate_up);
  if (taps) {
    if (taps->gate) batch.copy(s.gate, *taps->gate);
    if (taps->up) batch.copy(s.up, *taps->up);
  }
  batch.text_swiglu_split_bf16(s.gate, s.up, s.activation);
  if (taps && taps->activation) batch.copy(s.activation, *taps->activation);
  record_projection(batch, w.down, s.down_plan, s, s.activation, s.branch, rows);
  batch.text_add_residual_bf16(tokens, s.branch);
  if (taps && taps->final_residual)
    batch.copy(tokens, *taps->final_residual);
}

uint32_t ExactQwenTextLayerStage::required_operators(
    const QwenTextLayerTaps* taps) const {
  if (!impl_ || !impl_->weights) {
    throw std::logic_error("Vulkan Qwen layer: stage is not loaded");
  }
  const auto& w = *impl_->weights;
  uint32_t count = 10u +
      projection_operators(w.q, impl_->config.sequence) +
      projection_operators(w.k, impl_->config.sequence) +
      projection_operators(w.v, impl_->config.sequence) +
      projection_operators(w.o, impl_->config.sequence) +
      projection_operators(w.gate, impl_->config.sequence) +
      projection_operators(w.up, impl_->config.sequence) +
      projection_operators(w.down, impl_->config.sequence);
  if (w.q.weight.applies_convrot() && w.k.weight.applies_convrot() &&
      w.v.weight.applies_convrot()) {
    count -= 2;  // one shared activation transform instead of three
  }
  if (w.gate.weight.applies_convrot() && w.up.weight.applies_convrot()) {
    count -= 1;  // one shared activation transform instead of two
  }
  if (taps) {
    count += taps->input_norm != nullptr;
    count += taps->query != nullptr;
    count += taps->key != nullptr;
    count += taps->value != nullptr;
    count += taps->attention != nullptr;
    count += taps->attention_residual != nullptr;
    count += taps->post_attention_norm != nullptr;
    count += taps->gate != nullptr;
    count += taps->up != nullptr;
    count += taps->activation != nullptr;
    count += taps->final_residual != nullptr;
  }
  return count;
}

void ExactQwenTextLayerStage::unload() noexcept {
  if (impl_) impl_->weights.reset();
}
bool ExactQwenTextLayerStage::loaded() const noexcept {
  return impl_ && impl_->weights;
}
const QwenTextLayerConfig& ExactQwenTextLayerStage::config() const noexcept {
  return impl_->config;
}
text::WeightFormat ExactQwenTextLayerStage::format() const noexcept {
  return impl_ && impl_->weights ? impl_->weights->format
                                : text::WeightFormat::kAuto;
}
uint64_t ExactQwenTextLayerStage::persistent_bytes() const noexcept {
  return impl_ && impl_->weights ? impl_->weights->bytes() : 0;
}
uint64_t ExactQwenTextLayerStage::peak_device_bytes(
    const ExactQwenTextLayerScratch& scratch) const noexcept {
  return persistent_bytes() + scratch.reserved_bytes();
}

}  // namespace slopfab::vulkan
