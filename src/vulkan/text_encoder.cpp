#include "vidfab/vulkan/text_encoder.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

#include "vidfab/vulkan/text_layer.h"

namespace vidfab::vulkan {
namespace {

using Clock = std::chrono::steady_clock;

TensorLayout matrix(uint64_t rows, uint64_t columns) {
  const uint64_t shape[] = {rows, columns};
  return TensorLayout::contiguous(shape, 2);
}

text::EncoderConfig resolved_config(const SafeTensors& checkpoint,
                                    const text::EncoderConfig& requested) {
  text::EncoderConfig result = requested;
  const text::WeightFormat detected = text::detect_weight_format(checkpoint);
  if (result.format != text::WeightFormat::kAuto && result.format != detected) {
    throw std::runtime_error(
        "Vulkan Qwen encoder: requested format differs from checkpoint");
  }
  result.format = detected;
  result.residency = text::Residency::kStreaming;
  result.arithmetic = text::EncoderArithmetic::kExact;
  if (result.num_layers != 50 || result.hidden_size != 5120 ||
      result.num_attention_heads != 64 || result.num_key_value_heads != 8 ||
      result.head_dim != 128 || result.intermediate_size != 25600 ||
      result.rms_norm_eps != 1.0e-6f || result.rope_theta != 5.0e6f ||
      result.max_prompt_tokens <= 0 || result.max_prompt_tokens > 8192) {
    throw std::invalid_argument(
        "Vulkan Qwen encoder: invalid production configuration");
  }
  return result;
}

uint32_t layer_operators(text::WeightFormat format, uint32_t rows) {
  const uint32_t row_gemm = (rows >= 64 ? 1u : 0u) +
      (rows % 64 != 0 ? 1u : 0u);
  if (format == text::WeightFormat::kI8ConvRot)
    return 10u + 7u * (2u + row_gemm) - 3u;
  if (format == text::WeightFormat::kNVFP4Awq)
    return 10u + 5u * (1u + row_gemm) + 2u * (2u + row_gemm);
  throw std::logic_error("Vulkan Qwen encoder: unresolved format");
}

}  // namespace

struct ExactQwenTextEncoder::Impl {
  struct ShapeState {
    QwenTextLayerConfig layer_config;
    ExactQwenTextLayerStage stage;
    ExactQwenTextLayerScratch scratch;
    DeviceTensor tokens, cosine, sine, output;

    ShapeState(TensorContext& context, const text::EncoderConfig& config,
               uint32_t sequence) {
      layer_config.sequence = sequence;
      layer_config.encoder = config;
      stage = ExactQwenTextLayerStage::create(context, layer_config);
      scratch = ExactQwenTextLayerScratch::create(context, layer_config);
      tokens = context.allocate(matrix(sequence, config.hidden_size),
                                ScalarType::kBFloat16);
      cosine = context.allocate(matrix(sequence, config.head_dim),
                                ScalarType::kFloat32);
      sine = context.allocate(matrix(sequence, config.head_dim),
                              ScalarType::kFloat32);
      output = context.allocate(matrix(sequence, config.hidden_size),
                                ScalarType::kFloat32);
    }

    uint64_t activation_bytes() const noexcept {
      return tokens.layout().bytes(tokens.type()) +
          cosine.layout().bytes(cosine.type()) +
          sine.layout().bytes(sine.type()) +
          output.layout().bytes(output.type());
    }
  };

  TensorContext* context = nullptr;
  const SafeTensors* checkpoint = nullptr;
  const TensorView* embedding = nullptr;
  const TensorView* embedding_scale = nullptr;
  text::EncoderConfig config;
  std::unique_ptr<ShapeState> shape;
  ExactQwenTextEncoderStats stats;
  uint64_t allocator_baseline = 0;

  explicit Impl(TensorContext& owner)
      : context(&owner), allocator_baseline(owner.pooled_used_bytes()) {}

  void reset() noexcept {
    shape.reset();
    // Submitted work is synchronously waited by encode(), but command flight
    // slots may still retain shared tensor owners until collection. Retire
    // them here so unload releases the complete model-owned working set.
    try { context->collect(); } catch (...) {}
    checkpoint = nullptr;
    embedding = nullptr;
    embedding_scale = nullptr;
    config = text::EncoderConfig{};
    stats = ExactQwenTextEncoderStats{};
  }

  ShapeState& ensure_shape(uint32_t sequence) {
    if (shape && shape->layer_config.sequence == sequence &&
        shape->layer_config.encoder.format == config.format) return *shape;
    auto next = std::make_unique<ShapeState>(*context, config, sequence);
    shape = std::move(next);
    return *shape;
  }
};

ExactQwenTextEncoder::ExactQwenTextEncoder() = default;
ExactQwenTextEncoder::~ExactQwenTextEncoder() = default;
ExactQwenTextEncoder::ExactQwenTextEncoder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ExactQwenTextEncoder::ExactQwenTextEncoder(ExactQwenTextEncoder&&) noexcept =
    default;
ExactQwenTextEncoder& ExactQwenTextEncoder::operator=(
    ExactQwenTextEncoder&&) noexcept = default;

ExactQwenTextEncoder ExactQwenTextEncoder::create(TensorContext& context) {
  context.require_exact_fp32_vae_normalization();
  context.require_exact_vae_pointwise();
  context.require_exact_causal_gqa_attention();
  return ExactQwenTextEncoder(std::make_unique<Impl>(context));
}

void ExactQwenTextEncoder::load(const SafeTensors& checkpoint,
                                const text::EncoderConfig& requested) {
  if (!impl_) throw std::logic_error("Vulkan Qwen encoder: empty encoder");
  const Clock::time_point begin = Clock::now();
  text::EncoderConfig next = resolved_config(checkpoint, requested);
  // Validate the complete archive before changing the active model.
  text::validate_checkpoint(checkpoint, next);
  const TensorView* embedding = &checkpoint.at("model.embed_tokens.weight");
  const TensorView* embedding_scale =
      checkpoint.find("model.embed_tokens.weight_scale");

  const bool shape_compatible = impl_->shape &&
      impl_->config.format == next.format;
  impl_->checkpoint = &checkpoint;
  impl_->embedding = embedding;
  impl_->embedding_scale = embedding_scale;
  impl_->config = next;
  if (!shape_compatible) impl_->shape.reset();
  impl_->stats = ExactQwenTextEncoderStats{};
  impl_->stats.load_seconds =
      std::chrono::duration<double>(Clock::now() - begin).count();
}

void ExactQwenTextEncoder::unload() noexcept {
  if (impl_) impl_->reset();
}
bool ExactQwenTextEncoder::loaded() const noexcept {
  return impl_ && impl_->checkpoint != nullptr;
}
const text::EncoderConfig& ExactQwenTextEncoder::config() const {
  if (!loaded()) throw std::logic_error("Vulkan Qwen encoder: not loaded");
  return impl_->config;
}
text::WeightFormat ExactQwenTextEncoder::format() const noexcept {
  return loaded() ? impl_->config.format : text::WeightFormat::kAuto;
}

text::PromptEmbedding ExactQwenTextEncoder::encode(
    const std::vector<int32_t>& token_ids, text::EncoderTrace* trace) {
  if (!loaded()) throw std::logic_error("Vulkan Qwen encoder: not loaded");
  if (token_ids.empty() ||
      token_ids.size() > static_cast<size_t>(impl_->config.max_prompt_tokens)) {
    throw std::invalid_argument("Vulkan Qwen encoder: invalid prompt length");
  }
  const Clock::time_point begin = Clock::now();
  const uint32_t sequence = static_cast<uint32_t>(token_ids.size());
  // Validate the largest layer transaction before shape allocation, uploads,
  // or residual mutation. The final layer additionally widens the output and
  // an enabled trace adds its device-only boundary copy.
  {
    TensorBatch capacity = impl_->context->begin_batch();
    capacity.require_operator_capacity(layer_operators(impl_->config.format,
        sequence) + 1u + (trace != nullptr ? 1u : 0u));
  }
  Impl::ShapeState& state = impl_->ensure_shape(sequence);
  DeviceTensor trace_device;
  if (trace != nullptr) {
    trace_device = impl_->context->allocate(
        matrix(uint64_t(sequence) * 50, impl_->config.hidden_size),
        ScalarType::kBFloat16);
  }
  std::vector<uint16_t> embedding;
  text::gather_embedding_rows(*impl_->embedding, impl_->embedding_scale,
                              token_ids, embedding);
  std::vector<float> cosine, sine;
  text::build_rope_tables(sequence,
      text::rope_inv_freq(impl_->config.head_dim, impl_->config.rope_theta),
      cosine, sine);
  const TensorUpload uploads[] = {
      {&state.tokens, embedding.data(), embedding.size() * sizeof(uint16_t)},
      {&state.cosine, cosine.data(), cosine.size() * sizeof(float)},
      {&state.sine, sine.data(), sine.size() * sizeof(float)}};
  impl_->context->upload_batch(uploads, 3);

  uint64_t max_weight = 0;
  uint64_t peak_used = impl_->context->pooled_used_bytes();
  try {
    for (uint32_t layer = 0; layer < 50; ++layer) {
      state.stage.unload();
      state.stage.load(*impl_->checkpoint, layer);
      max_weight = std::max(max_weight, state.stage.persistent_bytes());
      peak_used = std::max(peak_used, impl_->context->pooled_used_bytes());
      TensorBatch batch = impl_->context->begin_batch();
      batch.require_operator_capacity(state.stage.required_operators() +
          (trace != nullptr ? 1u : 0u) + (layer == 49 ? 1u : 0u));
      state.stage.record(batch, state.tokens, state.cosine, state.sine,
                         state.scratch);
      if (trace != nullptr)
        batch.copy_rows(state.tokens, trace_device, 0, layer * sequence,
                        sequence);
      if (layer == 49) batch.convert(state.tokens, state.output);
      batch.submit().wait();
    }
  } catch (...) {
    state.stage.unload();
    throw;
  }
  state.stage.unload();

  text::PromptEmbedding result;
  result.num_tokens = static_cast<int>(sequence);
  result.hidden_size = impl_->config.hidden_size;
  result.data.resize(static_cast<size_t>(sequence) * impl_->config.hidden_size);
  result.modality_tags.assign(sequence, 1);
  impl_->context->download(state.output, result.data.data(), result.data.size());
  const uint64_t trace_bytes = trace_device
      ? trace_device.layout().bytes(trace_device.type()) : 0;
  if (trace != nullptr) {
    std::vector<uint16_t> boundaries(
        static_cast<size_t>(50) * sequence * impl_->config.hidden_size);
    impl_->context->download_bytes(trace_device, boundaries.data(),
                                   boundaries.size() * sizeof(uint16_t));
    trace->num_tokens = static_cast<int>(sequence);
    trace->hidden_size = impl_->config.hidden_size;
    trace->layer_residual_bf16 = std::move(boundaries);
  }
  trace_device = DeviceTensor();

  impl_->stats.last_num_tokens = sequence;
  impl_->stats.max_layer_weight_bytes = max_weight;
  impl_->stats.scratch_bytes = state.scratch.reserved_bytes();
  impl_->stats.activation_bytes = state.activation_bytes();
  impl_->stats.peak_device_bytes = max_weight + state.scratch.reserved_bytes() +
      state.activation_bytes() + trace_bytes;
  impl_->stats.allocator_baseline_bytes = impl_->allocator_baseline;
  impl_->stats.allocator_peak_used_bytes = peak_used;
  impl_->stats.allocator_used_bytes = impl_->context->pooled_used_bytes();
  impl_->stats.allocator_reserved_bytes = impl_->context->reserved_bytes();
  impl_->stats.descriptor_set_allocations =
      impl_->context->descriptor_set_allocations();
  impl_->stats.last_encode_seconds =
      std::chrono::duration<double>(Clock::now() - begin).count();
  return result;
}

text::PromptEmbedding ExactQwenTextEncoder::encode(
    const text::Tokenizer& tokenizer, const std::string& prompt) {
  if (prompt.empty()) throw std::invalid_argument("Vulkan Qwen encoder: empty prompt");
  return encode(tokenizer.encode(prompt));
}

const ExactQwenTextEncoderStats& ExactQwenTextEncoder::stats() const noexcept {
  static const ExactQwenTextEncoderStats empty{};
  return impl_ ? impl_->stats : empty;
}

}  // namespace vidfab::vulkan
