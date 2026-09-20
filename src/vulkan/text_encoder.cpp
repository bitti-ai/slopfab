#include "slopfab/vulkan/text_encoder.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>

#include "slopfab/vulkan/text_layer.h"
#include "slopfab/vulkan/vision_stage.h"
#include "slopfab/text/backend_capabilities.h"

namespace slopfab::vulkan {
namespace {

using Clock = std::chrono::steady_clock;

TensorLayout matrix(uint64_t rows, uint64_t columns) {
  const uint64_t shape[] = {rows, columns};
  return TensorLayout::contiguous(shape, 2);
}
uint64_t tensor_bytes(const DeviceTensor& tensor) {
  return tensor ? tensor.layout().bytes(tensor.type()) : 0;
}
uint64_t nonstaging_used_bytes(const TensorContext& context) {
  const uint64_t capacity = context.staging_capacity_bytes();
  const uint64_t staging = capacity > std::numeric_limits<uint64_t>::max() / 2
      ? std::numeric_limits<uint64_t>::max() : capacity * 2;
  const uint64_t used = context.pooled_used_bytes();
  return used >= staging ? used - staging : 0;
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
  if (!text::supports_exact_text_layer({result.max_prompt_tokens, result.hidden_size,
          result.num_attention_heads, result.num_key_value_heads, result.head_dim,
          result.intermediate_size, result.rms_norm_eps}) ||
      result.num_layers != 50 || result.intermediate_size != 25600 ||
      result.rope_theta != 5.0e6f ||
      result.max_prompt_tokens <= 0 || result.max_prompt_tokens > text::kMaxPromptTokens) {
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
  ExactQwenVisionEncoder vision;

  explicit Impl(TensorContext& owner)
      : context(&owner), vision(ExactQwenVisionEncoder::create(owner)) {}

  // Return the allocator bytes not owned by this encoder at an encode
  // boundary. This excludes context staging and allocations owned by callers,
  // while subtracting a retained shape so a text->multimodal shape switch does
  // not hide model memory in the baseline. Exact logical accounting below
  // remains the authority for tensor bytes when pool alignment differs.
  uint64_t external_used_bytes() const noexcept {
    const uint64_t used = nonstaging_used_bytes(*context);
    const uint64_t owned = shape
        ? shape->scratch.reserved_bytes() + shape->activation_bytes() : 0;
    return used >= owned ? used - owned : 0;
  }

  void reset() noexcept {
    shape.reset();
    vision.unload();
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
  // Strictly validate the complete archive before changing the active model.
  // This includes every canonical quant descriptor/global scalar and the
  // embedding contract, in one aggregate manifest scan.
  QwenTextLayerConfig validation_config;
  validation_config.sequence = 1;
  validation_config.encoder = next;
  ExactQwenTextLayerStage::validate_archive(checkpoint, validation_config);
  // The same production archives contain the complete visual tower. Validate
  // its exact 351-tensor manifest in the same pre-commit transaction, even for
  // a text-only encode, so a later multimodal request cannot discover a corrupt
  // layer after allocating its activation graph.
  (void)text::load_qwen3vl_vision_checkpoint(checkpoint);
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
  const uint64_t allocator_baseline = impl_->external_used_bytes();
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
  uint64_t peak_nonstaging = nonstaging_used_bytes(*impl_->context);
  try {
    for (uint32_t layer = 0; layer < 50; ++layer) {
      state.stage.unload();
      state.stage.load(*impl_->checkpoint, layer);
      max_weight = std::max(max_weight, state.stage.persistent_bytes());
      peak_used = std::max(peak_used, impl_->context->pooled_used_bytes());
      peak_nonstaging = std::max(
          peak_nonstaging, nonstaging_used_bytes(*impl_->context));
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
  const uint64_t observed_peak = peak_nonstaging >= allocator_baseline
      ? peak_nonstaging - allocator_baseline : peak_nonstaging;
  impl_->stats.peak_device_bytes = std::max(
      max_weight + state.scratch.reserved_bytes() + state.activation_bytes() +
          trace_bytes,
      observed_peak);
  impl_->stats.allocator_baseline_bytes = allocator_baseline;
  impl_->stats.allocator_peak_used_bytes = peak_used;
  impl_->stats.allocator_peak_nonstaging_bytes = peak_nonstaging;
  impl_->stats.allocator_used_bytes = impl_->context->pooled_used_bytes();
  impl_->stats.allocator_reserved_bytes = impl_->context->reserved_bytes();
  impl_->stats.descriptor_set_allocations =
      impl_->context->descriptor_set_allocations();
  impl_->stats.last_encode_seconds =
      std::chrono::duration<double>(Clock::now() - begin).count();
  return result;
}

text::PromptEmbedding ExactQwenTextEncoder::encode(
    const std::vector<int32_t>& token_ids,
    const std::vector<text::QwenPixelValues>& images,
    text::EncoderTrace* trace) {
  if (!loaded()) throw std::logic_error("Vulkan Qwen encoder: not loaded");
  if (images.empty()) return encode(token_ids, trace);
  if (token_ids.empty() ||
      token_ids.size() > static_cast<size_t>(impl_->config.max_prompt_tokens))
    throw std::invalid_argument("Vulkan Qwen encoder: invalid multimodal prompt length");
  const Clock::time_point begin = Clock::now();
  const uint32_t sequence = static_cast<uint32_t>(token_ids.size());
  const uint64_t allocator_baseline = impl_->external_used_bytes();
  std::vector<text::QwenImageGrid> grids;
  grids.reserve(images.size());
  uint64_t visual_tokens_wide = 0;
  for (const auto& image : images) {
    const size_t patches = image.grid.patch_count();
    if (image.grid.temporal <= 0 || image.grid.height <= 0 ||
        image.grid.width <= 0 || (image.grid.height & 1) != 0 ||
        (image.grid.width & 1) != 0 || patches == 0 || patches > 16384 ||
        patches % 4 != 0 ||
        image.rows.size() != patches * 1536)
      throw std::invalid_argument("Vulkan Qwen encoder: invalid visual rows");
    grids.push_back(image.grid);
    visual_tokens_wide += image.grid.merged_token_count();
  }
  const text::QwenMultimodalPlan multimodal =
      text::qwen3vl_multimodal_plan(token_ids, grids);
  if (visual_tokens_wide == 0 ||
      visual_tokens_wide != multimodal.image_rows.size() ||
      visual_tokens_wide > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("Vulkan Qwen encoder: visual token mismatch");
  const uint32_t visual_tokens = static_cast<uint32_t>(visual_tokens_wide);
  {
    TensorBatch capacity = impl_->context->begin_batch();
    capacity.require_operator_capacity(layer_operators(impl_->config.format,
        sequence) + 2u + (trace != nullptr ? 1u : 0u));
  }
  Impl::ShapeState& state = impl_->ensure_shape(sequence);
  DeviceTensor visual_main = impl_->context->allocate(
      matrix(visual_tokens, impl_->config.hidden_size), ScalarType::kBFloat16);
  DeviceTensor visual_deep[3];
  for (auto& tensor : visual_deep)
    tensor = impl_->context->allocate(
        matrix(visual_tokens, impl_->config.hidden_size), ScalarType::kBFloat16);
  const uint64_t image_row_count = visual_tokens;
  DeviceTensor image_rows = impl_->context->allocate(
      TensorLayout::contiguous(&image_row_count, 1), ScalarType::kInt32);
  impl_->context->upload_bytes(image_rows, multimodal.image_rows.data(),
      multimodal.image_rows.size() * sizeof(int32_t));
  uint32_t visual_offset = 0;
  uint64_t vision_peak_used = 0;
  uint64_t vision_peak_nonstaging = 0;
  uint64_t vision_phase_bytes = 0;
  try {
    impl_->vision.load(*impl_->checkpoint);
    for (const auto& image : images) {
      impl_->vision.encode(image);
      const ExactQwenVisionStats& vision_stats = impl_->vision.stats();
      vision_peak_used = std::max(
          vision_peak_used, vision_stats.allocator_peak_used_bytes);
      vision_peak_nonstaging = std::max(vision_peak_nonstaging,
          vision_stats.allocator_peak_nonstaging_bytes);
      vision_phase_bytes = std::max(
          vision_phase_bytes, vision_stats.scratch_bytes +
              vision_stats.activation_bytes +
              vision_stats.max_streamed_weight_bytes);
      const uint32_t count = impl_->vision.output_tokens();
      TensorBatch copy = impl_->context->begin_batch();
      copy.require_operator_capacity(4);
      copy.copy_rows(impl_->vision.main_output(), visual_main, 0,
                     visual_offset, count);
      for (uint32_t slot = 0; slot < 3; ++slot)
        copy.copy_rows(impl_->vision.deepstack_output(slot), visual_deep[slot],
                       0, visual_offset, count);
      copy.submit().wait();
      visual_offset += count;
    }
    impl_->vision.unload();
  } catch (...) {
    impl_->vision.unload();
    throw;
  }
  if (visual_offset != visual_tokens)
    throw std::logic_error("Vulkan Qwen encoder: incomplete visual output");

  std::vector<uint16_t> embedding;
  text::gather_embedding_rows(*impl_->embedding, impl_->embedding_scale,
                              token_ids, embedding);
  std::vector<float> cosine, sine;
  text::qwen3vl_decoder_rope_tables(multimodal, sequence, cosine, sine,
      impl_->config.head_dim, impl_->config.rope_theta);
  const TensorUpload uploads[] = {
      {&state.tokens, embedding.data(), embedding.size() * sizeof(uint16_t)},
      {&state.cosine, cosine.data(), cosine.size() * sizeof(float)},
      {&state.sine, sine.data(), sine.size() * sizeof(float)}};
  impl_->context->upload_batch(uploads, 3);
  { TensorBatch scatter = impl_->context->begin_batch();
    scatter.vision_scatter_bf16(visual_main, state.tokens, image_rows);
    scatter.submit().wait(); }

  DeviceTensor trace_device;
  if (trace != nullptr)
    trace_device = impl_->context->allocate(
        matrix(uint64_t(sequence) * 50, impl_->config.hidden_size),
        ScalarType::kBFloat16);
  uint64_t max_weight = 0;
  uint64_t peak_used = std::max(
      vision_peak_used, impl_->context->pooled_used_bytes());
  uint64_t peak_nonstaging = std::max(
      vision_peak_nonstaging, nonstaging_used_bytes(*impl_->context));
  try {
    for (uint32_t layer = 0; layer < 50; ++layer) {
      state.stage.unload();
      state.stage.load(*impl_->checkpoint, layer);
      max_weight = std::max(max_weight, state.stage.persistent_bytes());
      peak_used = std::max(peak_used, impl_->context->pooled_used_bytes());
      peak_nonstaging = std::max(
          peak_nonstaging, nonstaging_used_bytes(*impl_->context));
      TensorBatch batch = impl_->context->begin_batch();
      const int deep_slot = text::qwen3vl_deepstack_slot(layer);
      batch.require_operator_capacity(state.stage.required_operators() +
          (deep_slot >= 0 ? 1u : 0u) + (trace != nullptr ? 1u : 0u) +
          (layer == 49 ? 1u : 0u));
      state.stage.record(batch, state.tokens, state.cosine, state.sine,
                         state.scratch);
      if (deep_slot >= 0)
        batch.vision_scatter_add_bf16(visual_deep[deep_slot], state.tokens,
                                      image_rows);
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
  size_t image_at = 0;
  for (const auto& image : images) {
    const size_t count = image.grid.merged_token_count();
    const int first = multimodal.image_rows[image_at];
    for (size_t row = 0; row < count + 2; ++row)
      result.modality_tags[static_cast<size_t>(first - 1) + row] = 0;
    image_at += count;
  }
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
  const uint64_t visual_bytes = tensor_bytes(visual_main) +
      tensor_bytes(visual_deep[0]) + tensor_bytes(visual_deep[1]) +
      tensor_bytes(visual_deep[2]) + tensor_bytes(image_rows);
  impl_->stats.last_num_tokens = sequence;
  impl_->stats.max_layer_weight_bytes = max_weight;
  impl_->stats.scratch_bytes = state.scratch.reserved_bytes();
  impl_->stats.activation_bytes = state.activation_bytes() + visual_bytes;
  const uint64_t decoder_live_bytes = state.scratch.reserved_bytes() +
      state.activation_bytes();
  const uint64_t observed_peak = peak_nonstaging >= allocator_baseline
      ? peak_nonstaging - allocator_baseline : peak_nonstaging;
  impl_->stats.peak_device_bytes = std::max({
      max_weight + decoder_live_bytes + visual_bytes + trace_bytes,
      decoder_live_bytes + vision_phase_bytes + visual_bytes,
      observed_peak});
  impl_->stats.allocator_baseline_bytes = allocator_baseline;
  impl_->stats.allocator_peak_used_bytes = peak_used;
  impl_->stats.allocator_peak_nonstaging_bytes = peak_nonstaging;
  impl_->stats.allocator_used_bytes = impl_->context->pooled_used_bytes();
  impl_->stats.allocator_reserved_bytes = impl_->context->reserved_bytes();
  impl_->stats.descriptor_set_allocations =
      impl_->context->descriptor_set_allocations();
  impl_->stats.last_encode_seconds = std::chrono::duration<double>(
      Clock::now() - begin).count();
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

}  // namespace slopfab::vulkan
