#include "vidfab/vulkan/dit_transformer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vidfab/dtype.h"
#include "vidfab/tensor_convert.h"
#include "vidfab/vulkan/gemm.h"

namespace vidfab::vulkan {
namespace {

TensorLayout matrix(uint64_t rows, uint64_t columns) {
  const uint64_t shape[] = {rows, columns};
  return TensorLayout::contiguous(shape, 2);
}
TensorLayout vector(uint64_t count) {
  return TensorLayout::contiguous(&count, 1);
}
uint64_t bytes(const DeviceTensor& tensor) {
  return tensor ? tensor.layout().bytes(tensor.type()) : 0;
}
uint64_t add_saturating(uint64_t a, uint64_t b) {
  return b > std::numeric_limits<uint64_t>::max() - a
      ? std::numeric_limits<uint64_t>::max() : a + b;
}
void require_shape(const TensorView& view,
                   std::initializer_list<int64_t> shape,
                   const std::string& name) {
  if (view.shape != std::vector<int64_t>(shape))
    throw std::runtime_error("Vulkan H3 transformer: '" + name +
                             "' has wrong shape");
}
void validate_config(const ExactH3TransformerConfig& c) {
  const H3BlockConfig& b = c.main.block;
  const uint64_t packed = uint64_t(c.text_rows) + c.video_rows + c.audio_rows;
  if (c.main.layers == 0 || c.main.layers > 50 ||
      c.refiner_layers == 0 || c.refiner_layers > 2 || c.text_rows == 0 ||
      c.video_rows == 0 || c.audio_rows == 0 || packed != b.sequence ||
      c.text_dim == 0 || c.video_dim == 0 || c.audio_dim == 0 ||
      b.hidden == 0 || b.heads == 0 || b.head_dim != 128 ||
      uint64_t(b.heads) * b.head_dim > UINT32_MAX || b.ffn == 0 ||
      b.adaln_rank == 0 || b.modalities != 3 ||
      b.timesteps == 0 || !std::isnormal(b.epsilon) || b.epsilon <= 0.0f)
    throw std::invalid_argument("Vulkan H3 transformer: invalid exact configuration");
}
void validate_endpoint_archive(const SafeTensors& st,
                               const ExactH3TransformerConfig& c) {
  const uint32_t h = c.main.block.hidden;
  const uint32_t r = c.main.block.adaln_rank;
  require_shape(st.at("condition_proj.weight"), {h, c.text_dim},
                "condition_proj.weight");
  require_shape(st.at("condition_proj.bias"), {h}, "condition_proj.bias");
  require_shape(st.at("video_patch_proj.weight"), {h, c.video_dim},
                "video_patch_proj.weight");
  require_shape(st.at("video_patch_proj.bias"), {h},
                "video_patch_proj.bias");
  require_shape(st.at("audio_patch_proj.weight"), {h, c.audio_dim},
                "audio_patch_proj.weight");
  require_shape(st.at("audio_patch_proj.bias"), {h},
                "audio_patch_proj.bias");
  require_shape(st.at("token_refiner.final_norm.weight"), {h},
                "token_refiner.final_norm.weight");
  require_shape(st.at("final_layer.norm.weight"), {h},
                "final_layer.norm.weight");
  require_shape(st.at("final_layer.adaln_proj.linear.weight"), {2 * h, r},
                "final_layer.adaln_proj.linear.weight");
  require_shape(st.at("final_layer.adaln_proj.linear.bias"), {2 * h},
                "final_layer.adaln_proj.linear.bias");
  require_shape(st.at("final_layer.video_out.weight"), {c.video_dim, h},
                "final_layer.video_out.weight");
  require_shape(st.at("final_layer.video_out.bias"), {c.video_dim},
                "final_layer.video_out.bias");
  require_shape(st.at("final_layer.audio_out.weight"), {c.audio_dim, h},
                "final_layer.audio_out.weight");
  require_shape(st.at("final_layer.audio_out.bias"), {c.audio_dim},
                "final_layer.audio_out.bias");
  // Decode now, before a Vulkan allocation, so malformed dtypes/non-finite
  // conversion paths are part of the host-only initial-load transaction.
  (void)to_f32(st.at("condition_proj.weight"));
  (void)to_f32(st.at("condition_proj.bias"));
  (void)to_f32(st.at("video_patch_proj.weight"));
  (void)to_f32(st.at("video_patch_proj.bias"));
  (void)to_f32(st.at("audio_patch_proj.weight"));
  (void)to_f32(st.at("audio_patch_proj.bias"));
  (void)to_f32(st.at("token_refiner.final_norm.weight"));
  (void)to_f32(st.at("final_layer.norm.weight"));
  (void)to_f32(st.at("final_layer.adaln_proj.linear.weight"));
  (void)to_f32(st.at("final_layer.adaln_proj.linear.bias"));
  (void)to_f32(st.at("final_layer.video_out.weight"));
  (void)to_f32(st.at("final_layer.video_out.bias"));
  (void)to_f32(st.at("final_layer.audio_out.weight"));
  (void)to_f32(st.at("final_layer.audio_out.bias"));
}
DeviceTensor upload_f32(TensorContext& context, const TensorView& view,
                        const TensorLayout& layout) {
  std::vector<float> values = to_f32(view);
  if (values.size() != layout.elements())
    throw std::runtime_error("Vulkan H3 transformer: fp32 upload size mismatch");
  DeviceTensor result = context.allocate(layout);
  context.upload_transient(result, values.data(), values.size());
  return result;
}
DeviceTensor upload_bf16(TensorContext& context, const TensorView& view,
                         const TensorLayout& layout) {
  std::vector<float> wide = to_f32(view);
  if (wide.size() != layout.elements())
    throw std::runtime_error("Vulkan H3 transformer: BF16 upload size mismatch");
  std::vector<uint16_t> values(wide.size());
  for (size_t i = 0; i < wide.size(); ++i) values[i] = f32_to_bf16(wide[i]);
  DeviceTensor result = context.allocate(layout, ScalarType::kBFloat16);
  context.upload_transient_bytes(result, values.data(), values.size() * 2);
  return result;
}
bool tensor_is(const TensorContext& context, const DeviceTensor& tensor,
               ScalarType type, uint64_t rows, uint64_t columns = 0) {
  if (!context.owns(tensor)) return false;
  const DeviceTensorView view = tensor.view();
  return view.type == type && view.layout.is_contiguous() &&
      ((columns == 0 && view.layout.rank == 1 &&
        view.layout.extent[0] == rows) ||
       (columns != 0 && view.layout.rank == 2 &&
        view.layout.extent[0] == rows && view.layout.extent[1] == columns));
}

}  // namespace

struct ExactH3Transformer::Impl {
  struct State {
    ExactH3MainGraph main;
    ExactH3BlockScratch refiner_scratch;
    std::vector<ExactH3BlockStage> refiner;
    DeviceTensor condition_weight, condition_bias;
    DeviceTensor video_in_weight, video_in_bias;
    DeviceTensor audio_in_weight, audio_in_bias;
    DeviceTensor refiner_norm, final_norm;
    DeviceTensor final_shift_weight, final_shift_bias;
    DeviceTensor final_scale_weight, final_scale_bias;
    DeviceTensor video_out_weight, video_out_bias;
    DeviceTensor audio_out_weight, audio_out_bias;
    DenseGemmPlan condition_plan, video_in_plan, audio_in_plan;
    DenseGemmPlan video_out_plan, audio_out_plan;
    DeviceTensor text_input, text_cache;
    DeviceTensor refiner_selectors, refiner_code, refiner_cosine, refiner_sine;
    DeviceTensor hidden;
    DeviceTensor video_projected, video_projected_bf16;
    DeviceTensor audio_projected, audio_projected_bf16;
    DeviceTensor final_shift, final_scale;
    DeviceTensor video_gather_bf16, video_gather, video_normed;
    DeviceTensor audio_gather_bf16, audio_gather, audio_normed;
  };
  TensorContext* context = nullptr;
  ExactH3TransformerConfig config;
  std::unique_ptr<State> state;
  bool text_ready = false;

  Impl(TensorContext& owner, const ExactH3TransformerConfig& value)
      : context(&owner), config(value) {}
};

ExactH3Transformer::ExactH3Transformer() = default;
ExactH3Transformer::~ExactH3Transformer() = default;
ExactH3Transformer::ExactH3Transformer(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ExactH3Transformer::ExactH3Transformer(ExactH3Transformer&&) noexcept = default;
ExactH3Transformer& ExactH3Transformer::operator=(
    ExactH3Transformer&&) noexcept = default;

ExactH3Transformer ExactH3Transformer::create(
    TensorContext& context, const ExactH3TransformerConfig& config) {
  validate_config(config);
  return ExactH3Transformer(std::make_shared<Impl>(context, config));
}

void ExactH3Transformer::load(const SafeTensors& checkpoint) {
  if (!impl_) throw std::logic_error("Vulkan H3 transformer: empty object");
  if (loaded())
    throw std::logic_error("Vulkan H3 transformer: unload before load");
  const ExactH3TransformerConfig& c = impl_->config;
  validate_endpoint_archive(checkpoint, c);
  for (uint32_t layer = 0; layer < c.refiner_layers; ++layer)
    ExactH3BlockStage::validate_refiner_checkpoint(
        checkpoint, layer, H3BlockConfig{c.text_rows, c.main.block.hidden,
          c.main.block.heads, c.main.block.head_dim, c.main.block.ffn,
          1, 1, 1, c.main.block.epsilon});
  for (uint32_t layer = 0; layer < c.main.layers; ++layer)
    ExactH3BlockStage::validate_checkpoint(checkpoint, layer, c.main.block);

  auto next = std::make_unique<Impl::State>();
  TensorContext& context = *impl_->context;
  const uint32_t h = c.main.block.hidden;
  const uint32_t rank = c.main.block.adaln_rank;
  const uint32_t timesteps = c.main.block.timesteps;
  H3BlockConfig ref_cfg = c.main.block;
  ref_cfg.sequence = c.text_rows;
  ref_cfg.timesteps = 1;
  ref_cfg.modalities = 1;
  ref_cfg.adaln_rank = 1;
  next->refiner_scratch = ExactH3BlockScratch::create(context, ref_cfg);
  next->refiner.reserve(c.refiner_layers);
  for (uint32_t layer = 0; layer < c.refiner_layers; ++layer) {
    ExactH3BlockStage stage = ExactH3BlockStage::create(context, ref_cfg);
    stage.load_refiner(checkpoint, layer);
    stage.prepare(next->refiner_scratch);
    next->refiner.push_back(std::move(stage));
  }
  next->main = ExactH3MainGraph::create(context, c.main);
  next->main.load(checkpoint);

  next->condition_weight = upload_bf16(context,
      checkpoint.at("condition_proj.weight"), matrix(h, c.text_dim));
  next->condition_bias = upload_f32(context,
      checkpoint.at("condition_proj.bias"), vector(h));
  next->video_in_weight = upload_f32(context,
      checkpoint.at("video_patch_proj.weight"), matrix(h, c.video_dim));
  next->video_in_bias = upload_f32(context,
      checkpoint.at("video_patch_proj.bias"), vector(h));
  next->audio_in_weight = upload_f32(context,
      checkpoint.at("audio_patch_proj.weight"), matrix(h, c.audio_dim));
  next->audio_in_bias = upload_f32(context,
      checkpoint.at("audio_patch_proj.bias"), vector(h));
  next->refiner_norm = upload_bf16(context,
      checkpoint.at("token_refiner.final_norm.weight"), vector(h));
  next->final_norm = upload_bf16(context,
      checkpoint.at("final_layer.norm.weight"), vector(h));

  const std::vector<float> adaln_w = to_f32(
      checkpoint.at("final_layer.adaln_proj.linear.weight"));
  const std::vector<float> adaln_b = to_f32(
      checkpoint.at("final_layer.adaln_proj.linear.bias"));
  auto upload_slice = [&](const float* values, const TensorLayout& layout) {
    DeviceTensor result = context.allocate(layout);
    context.upload_transient(result, values, layout.elements());
    return result;
  };
  const uint64_t one_weight = uint64_t(h) * rank;
  next->final_shift_weight = upload_slice(adaln_w.data(), matrix(h, rank));
  next->final_scale_weight = upload_slice(adaln_w.data() + one_weight,
                                          matrix(h, rank));
  next->final_shift_bias = upload_slice(adaln_b.data(), vector(h));
  next->final_scale_bias = upload_slice(adaln_b.data() + h, vector(h));
  next->video_out_weight = upload_f32(context,
      checkpoint.at("final_layer.video_out.weight"), matrix(c.video_dim, h));
  next->video_out_bias = upload_f32(context,
      checkpoint.at("final_layer.video_out.bias"), vector(c.video_dim));
  next->audio_out_weight = upload_f32(context,
      checkpoint.at("final_layer.audio_out.weight"), matrix(c.audio_dim, h));
  next->audio_out_bias = upload_f32(context,
      checkpoint.at("final_layer.audio_out.bias"), vector(c.audio_dim));

  next->condition_plan = DenseGemmPlan::create(context,
      {c.text_rows, h, c.text_dim, DenseGemmMode::kBFloat16,
       DenseGemmBias::kFloat32, false});
  next->video_in_plan = DenseGemmPlan::create(context,
      {c.video_rows, h, c.video_dim, DenseGemmMode::kFloat32,
       DenseGemmBias::kFloat32, false});
  next->audio_in_plan = DenseGemmPlan::create(context,
      {c.audio_rows, h, c.audio_dim, DenseGemmMode::kFloat32,
       DenseGemmBias::kFloat32, false});
  next->video_out_plan = DenseGemmPlan::create(context,
      {c.video_rows, c.video_dim, h, DenseGemmMode::kFloat32,
       DenseGemmBias::kFloat32, false});
  next->audio_out_plan = DenseGemmPlan::create(context,
      {c.audio_rows, c.audio_dim, h, DenseGemmMode::kFloat32,
       DenseGemmBias::kFloat32, false});

  next->text_input = context.allocate(matrix(c.text_rows, c.text_dim),
                                      ScalarType::kBFloat16);
  next->text_cache = context.allocate(matrix(c.text_rows, h),
                                      ScalarType::kBFloat16);
  next->refiner_selectors = context.allocate(vector(c.text_rows),
                                              ScalarType::kInt32);
  next->refiner_code = context.allocate(matrix(1, 1));
  next->refiner_cosine = context.allocate(matrix(c.text_rows, 96));
  next->refiner_sine = context.allocate(matrix(c.text_rows, 96));
  std::vector<int32_t> zero_selectors(c.text_rows, 0);
  const float zero = 0.0f;
  std::vector<float> ones(uint64_t(c.text_rows) * 96, 1.0f);
  std::vector<float> zeros(ones.size(), 0.0f);
  context.upload_transient_bytes(next->refiner_selectors,
      zero_selectors.data(), zero_selectors.size() * sizeof(int32_t));
  context.upload_transient(next->refiner_code, &zero, 1);
  context.upload_transient(next->refiner_cosine, ones.data(), ones.size());
  context.upload_transient(next->refiner_sine, zeros.data(), zeros.size());

  next->hidden = context.allocate(matrix(c.main.block.sequence, h),
                                  ScalarType::kBFloat16);
  next->video_projected = context.allocate(matrix(c.video_rows, h));
  next->video_projected_bf16 = context.allocate(
      matrix(c.video_rows, h), ScalarType::kBFloat16);
  next->audio_projected = context.allocate(matrix(c.audio_rows, h));
  next->audio_projected_bf16 = context.allocate(
      matrix(c.audio_rows, h), ScalarType::kBFloat16);
  next->final_shift = context.allocate(matrix(timesteps, h));
  next->final_scale = context.allocate(matrix(timesteps, h));
  next->video_gather_bf16 = context.allocate(
      matrix(c.video_rows, h), ScalarType::kBFloat16);
  next->video_gather = context.allocate(matrix(c.video_rows, h));
  next->video_normed = context.allocate(matrix(c.video_rows, h));
  next->audio_gather_bf16 = context.allocate(
      matrix(c.audio_rows, h), ScalarType::kBFloat16);
  next->audio_gather = context.allocate(matrix(c.audio_rows, h));
  next->audio_normed = context.allocate(matrix(c.audio_rows, h));
  impl_->state = std::move(next);
  impl_->text_ready = false;
}

void ExactH3Transformer::unload() noexcept {
  if (impl_) {
    impl_->state.reset();
    impl_->text_ready = false;
  }
}
bool ExactH3Transformer::loaded() const noexcept {
  return impl_ && impl_->state && impl_->state->main.loaded();
}
bool ExactH3Transformer::text_prepared() const noexcept {
  return loaded() && impl_->text_ready;
}
const ExactH3TransformerConfig& ExactH3Transformer::config() const noexcept {
  return impl_->config;
}

uint32_t ExactH3Transformer::required_prepare_text_operators(
    const H3TransformerTextReplayTaps* taps) const {
  if (!loaded()) throw std::logic_error("Vulkan H3 transformer: not loaded");
  if (taps && (taps->count != 6 || !taps->boundaries))
    throw std::invalid_argument("Vulkan H3 transformer: invalid text taps");
  uint64_t count = 3;  // narrow, condition projection, final norm
  for (const ExactH3BlockStage& stage : impl_->state->refiner)
    count += stage.required_operators();
  if (taps) count += 6;
  if (count > UINT32_MAX) throw std::overflow_error(
      "Vulkan H3 transformer: prepare operator overflow");
  return static_cast<uint32_t>(count);
}

void ExactH3Transformer::prepare_text(
    DeviceTensor& prompt, const H3TransformerTextReplayTaps* taps) {
  if (!loaded()) throw std::logic_error("Vulkan H3 transformer: not loaded");
  const auto& c = impl_->config; auto& s = *impl_->state;
  if (!tensor_is(*impl_->context, prompt, ScalarType::kFloat32,
                 c.text_rows, c.text_dim))
    throw std::invalid_argument("Vulkan H3 transformer: invalid prompt tensor");
  if (taps) {
    for (uint32_t i = 0; i < taps->count; ++i) {
      if (!tensor_is(*impl_->context, taps->boundaries[i],
                     ScalarType::kBFloat16, c.text_rows,
                     c.main.block.hidden))
        throw std::invalid_argument("Vulkan H3 transformer: invalid text tap");
      for (uint32_t j = 0; j < i; ++j)
        if (taps->boundaries[i].view().resource ==
            taps->boundaries[j].view().resource)
          throw std::invalid_argument("Vulkan H3 transformer: aliased text taps");
    }
  }
  const uint32_t need = required_prepare_text_operators(taps);
  TensorBatch batch = impl_->context->begin_batch();
  if (batch.remaining_operator_capacity() < need)
    throw std::logic_error("Vulkan H3 transformer: insufficient prepare capacity");
  batch.convert(prompt, s.text_input);
  s.condition_plan.record(batch, s.text_input, s.condition_weight,
                          s.text_cache, c.text_rows, 0, 0,
                          &s.condition_bias);
  if (taps) batch.copy(s.text_cache, taps->boundaries[0]);
  for (uint32_t layer = 0; layer < s.refiner.size(); ++layer) {
    H3BlockReplayTaps block_taps;
    if (taps) {
      block_taps.attention_residual = &taps->boundaries[1 + layer * 2];
      block_taps.final_residual = &taps->boundaries[2 + layer * 2];
    }
    ExactH3BlockStage& stage = s.refiner[layer];
    stage.record(batch, s.text_cache, s.refiner_selectors, s.refiner_code,
                 s.refiner_cosine, s.refiner_sine, s.refiner_scratch, nullptr,
                 taps ? &block_taps : nullptr);
  }
  batch.rms_norm_bf16(s.text_cache, s.refiner_norm, s.text_cache,
                      c.main.block.epsilon);
  if (taps) batch.copy(s.text_cache, taps->boundaries[5]);
  batch.submit().wait();
  impl_->text_ready = true;
}

uint32_t ExactH3Transformer::required_forward_operators(
    const H3TransformerForwardReplayTaps* taps) const {
  if (!loaded()) throw std::logic_error("Vulkan H3 transformer: not loaded");
  const H3MainGraphReplayTaps* main_taps = taps ? taps->main_boundaries : nullptr;
  const uint64_t count = uint64_t(17) +
      impl_->state->main.required_operators(main_taps) +
      (taps && taps->packed_input ? 1u : 0u) +
      (taps && taps->main_final ? 1u : 0u);
  if (count > UINT32_MAX) throw std::overflow_error(
      "Vulkan H3 transformer: forward operator overflow");
  return static_cast<uint32_t>(count);
}

void ExactH3Transformer::record_forward(
    TensorBatch& batch, DeviceTensor& video_latents,
    DeviceTensor& audio_latents, DeviceTensor& main_selectors,
    DeviceTensor& code,
    DeviceTensor& cosine, DeviceTensor& sine,
    DeviceTensor& video_timestep_indices,
    DeviceTensor& audio_timestep_indices,
    DeviceTensor& video_velocity, DeviceTensor& audio_velocity,
    const H3AttentionRanges* ranges,
    const H3TransformerForwardReplayTaps* taps) {
  if (!text_prepared())
    throw std::logic_error("Vulkan H3 transformer: text is not prepared");
  const auto& c = impl_->config; auto& s = *impl_->state;
  const H3BlockConfig& b = c.main.block;
  const bool valid =
      tensor_is(*impl_->context, video_latents, ScalarType::kFloat32,
                c.video_rows, c.video_dim) &&
      tensor_is(*impl_->context, audio_latents, ScalarType::kFloat32,
                c.audio_rows, c.audio_dim) &&
      tensor_is(*impl_->context, main_selectors, ScalarType::kInt32,
                b.sequence) &&
      tensor_is(*impl_->context, code, ScalarType::kFloat32,
                b.timesteps, b.adaln_rank) &&
      tensor_is(*impl_->context, cosine, ScalarType::kFloat32,
                b.sequence, 96) &&
      tensor_is(*impl_->context, sine, ScalarType::kFloat32,
                b.sequence, 96) &&
      tensor_is(*impl_->context, video_timestep_indices, ScalarType::kInt32,
                c.video_rows) &&
      tensor_is(*impl_->context, audio_timestep_indices, ScalarType::kInt32,
                c.audio_rows) &&
      tensor_is(*impl_->context, video_velocity, ScalarType::kFloat32,
                c.video_rows, c.video_dim) &&
      tensor_is(*impl_->context, audio_velocity, ScalarType::kFloat32,
                c.audio_rows, c.audio_dim);
  if (!valid || video_velocity.view().resource == video_latents.view().resource ||
      audio_velocity.view().resource == audio_latents.view().resource)
    throw std::invalid_argument("Vulkan H3 transformer: invalid forward tensors");
  if (taps) {
    for (DeviceTensor* tap : {taps->packed_input, taps->main_final}) {
      if (tap && !tensor_is(*impl_->context, *tap, ScalarType::kBFloat16,
                            b.sequence, b.hidden))
        throw std::invalid_argument("Vulkan H3 transformer: invalid forward tap");
    }
    if (taps->packed_input && taps->main_final &&
        taps->packed_input->view().resource == taps->main_final->view().resource)
      throw std::invalid_argument("Vulkan H3 transformer: aliased forward taps");
  }
  const uint32_t need = required_forward_operators(taps);
  if (batch.remaining_operator_capacity() < need)
    throw std::logic_error("Vulkan H3 transformer: insufficient forward capacity");

  s.video_in_plan.record(batch, video_latents, s.video_in_weight,
                         s.video_projected, c.video_rows, 0, 0,
                         &s.video_in_bias);
  batch.convert(s.video_projected, s.video_projected_bf16);
  batch.copy_rows(s.video_projected_bf16, s.hidden, 0,
                  c.text_rows + c.audio_rows, c.video_rows);
  s.audio_in_plan.record(batch, audio_latents, s.audio_in_weight,
                         s.audio_projected, c.audio_rows, 0, 0,
                         &s.audio_in_bias);
  batch.convert(s.audio_projected, s.audio_projected_bf16);
  batch.copy_rows(s.audio_projected_bf16, s.hidden, 0,
                  c.text_rows, c.audio_rows);
  batch.copy_rows(s.text_cache, s.hidden, 0, 0, c.text_rows);
  if (taps && taps->packed_input) batch.copy(s.hidden, *taps->packed_input);
  s.main.record(batch, s.hidden, main_selectors, code, cosine, sine,
                ranges, taps ? taps->main_boundaries : nullptr);
  if (taps && taps->main_final) batch.copy(s.hidden, *taps->main_final);

  batch.dit_expand_adaln(s.final_shift_weight, s.final_shift_bias, code,
                         s.final_shift, 1, 1, b.hidden);
  batch.dit_expand_adaln(s.final_scale_weight, s.final_scale_bias, code,
                         s.final_scale, 1, 1, b.hidden);
  batch.copy_rows(s.hidden, s.video_gather_bf16,
                  c.text_rows + c.audio_rows, 0, c.video_rows);
  batch.convert(s.video_gather_bf16, s.video_gather);
  batch.rms_norm_modulate_f32(s.video_gather, s.final_norm,
      s.final_scale, s.final_shift, video_timestep_indices,
      s.video_normed, b.epsilon);
  s.video_out_plan.record(batch, s.video_normed, s.video_out_weight,
                          video_velocity, c.video_rows, 0, 0,
                          &s.video_out_bias);
  batch.copy_rows(s.hidden, s.audio_gather_bf16,
                  c.text_rows, 0, c.audio_rows);
  batch.convert(s.audio_gather_bf16, s.audio_gather);
  batch.rms_norm_modulate_f32(s.audio_gather, s.final_norm,
      s.final_scale, s.final_shift, audio_timestep_indices,
      s.audio_normed, b.epsilon);
  s.audio_out_plan.record(batch, s.audio_normed, s.audio_out_weight,
                          audio_velocity, c.audio_rows, 0, 0,
                          &s.audio_out_bias);
}

uint64_t ExactH3Transformer::persistent_bytes() const noexcept {
  if (!loaded()) return 0;
  const auto& s = *impl_->state;
  uint64_t total = s.main.persistent_bytes();
  for (const auto& stage : s.refiner)
    total = add_saturating(total, stage.persistent_bytes());
  for (const DeviceTensor* tensor : {&s.condition_weight, &s.condition_bias,
      &s.video_in_weight, &s.video_in_bias, &s.audio_in_weight, &s.audio_in_bias,
      &s.refiner_norm, &s.final_norm, &s.final_shift_weight,
      &s.final_shift_bias, &s.final_scale_weight, &s.final_scale_bias,
      &s.video_out_weight, &s.video_out_bias, &s.audio_out_weight,
      &s.audio_out_bias}) total = add_saturating(total, bytes(*tensor));
  return total;
}
uint64_t ExactH3Transformer::scratch_bytes() const noexcept {
  if (!loaded()) return 0;
  const auto& s = *impl_->state;
  uint64_t total = add_saturating(s.main.scratch_bytes(),
                                  s.refiner_scratch.reserved_bytes());
  for (const DeviceTensor* tensor : {&s.text_input, &s.text_cache,
      &s.refiner_selectors, &s.refiner_code, &s.refiner_cosine,
      &s.refiner_sine, &s.hidden, &s.video_projected,
      &s.video_projected_bf16, &s.audio_projected,
      &s.audio_projected_bf16, &s.final_shift, &s.final_scale,
      &s.video_gather_bf16, &s.video_gather, &s.video_normed,
      &s.audio_gather_bf16, &s.audio_gather, &s.audio_normed})
    total = add_saturating(total, bytes(*tensor));
  return total;
}
uint64_t ExactH3Transformer::peak_device_bytes() const noexcept {
  return add_saturating(persistent_bytes(), scratch_bytes());
}

}  // namespace vidfab::vulkan
