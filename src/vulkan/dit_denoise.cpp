#include "slopfab/vulkan/dit_denoise.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "slopfab/dit/adaln.h"
#include "slopfab/dit/rope.h"

namespace slopfab::vulkan {
namespace {

TensorLayout matrix(uint64_t rows, uint64_t columns) {
  const uint64_t shape[] = {rows, columns};
  return TensorLayout::contiguous(shape, 2);
}
TensorLayout vector(uint64_t count) {
  return TensorLayout::contiguous(&count, 1);
}
uint64_t tensor_bytes(const DeviceTensor& tensor) {
  return tensor ? tensor.layout().bytes(tensor.type()) : 0;
}
bool finite_span(const float* values, uint64_t count) {
  if (count == 0) return true;
  if (!values) return false;
  for (uint64_t i = 0; i < count; ++i)
    if (!std::isfinite(values[i])) return false;
  return true;
}

void validate_config(const ExactH3DenoiseConfig& c) {
  const dit::SequenceLayout& l = c.layout;
  const uint32_t sequence = c.transformer.main.block.sequence;
  const uint32_t condition_video = l.num_condition_video < 0 ? 0u :
      static_cast<uint32_t>(l.num_condition_video);
  const uint32_t condition_audio = l.num_condition_audio < 0 ? 0u :
      static_cast<uint32_t>(l.num_condition_audio);
  const uint32_t target_video = l.num_video_rows < 0 ? 0u :
      static_cast<uint32_t>(l.num_video_rows);
  const uint32_t target_audio = l.num_audio_rows < 0 ? 0u :
      static_cast<uint32_t>(l.num_audio_rows);
  const uint64_t total_video_wide = uint64_t(condition_video) + target_video;
  const uint64_t total_audio_wide = uint64_t(condition_audio) + target_audio;
  const uint64_t audio_start_wide = uint64_t(l.num_text > 0 ? l.num_text : 0) +
      condition_video + condition_audio;
  const uint64_t video_start_wide = audio_start_wide + target_audio;
  const uint64_t sequence_wide = video_start_wide + target_video;
  const uint32_t total_video =
      total_video_wide > std::numeric_limits<uint32_t>::max()
      ? 0u : static_cast<uint32_t>(total_video_wide);
  const uint32_t total_audio =
      total_audio_wide > std::numeric_limits<uint32_t>::max()
      ? 0u : static_cast<uint32_t>(total_audio_wide);
  const bool conditioned = condition_video != 0 || condition_audio != 0;
  if (l.num_condition_video < 0 || l.num_condition_audio < 0 ||
      total_video_wide > std::numeric_limits<uint32_t>::max() ||
      total_audio_wide > std::numeric_limits<uint32_t>::max() ||
      sequence_wide > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()) ||
      (conditioned && !l.condition_audio_is_explicit) || l.num_text <= 0 ||
      l.num_video_rows <= 0 || l.num_audio_rows < 0 ||
      sequence_wide != sequence ||
      c.transformer.text_rows != static_cast<uint32_t>(l.num_text) ||
      c.transformer.video_rows != total_video ||
      c.transformer.audio_rows != total_audio ||
      c.transformer.main.block.timesteps != (conditioned ? 4u : 2u) ||
      c.transformer.main.block.modalities != 3 ||
      c.transformer.main.block.adaln_rank != dit::AdaLNTable::kRank ||
      c.transformer.video_dim == 0 || c.transformer.audio_dim == 0 ||
      c.attention_band < 0 ||
      (c.attention_band != 0 && !c.attention_ranges.empty()) ||
      c.position_ids.size() != static_cast<size_t>(sequence) * 3 ||
      c.indices.tags.size() != sequence ||
      c.indices.text.size() != c.transformer.text_rows ||
      c.indices.audio.size() != total_audio ||
      c.indices.video.size() != total_video ||
      (conditioned &&
       (c.transformer.video_output_rows != target_video ||
        c.transformer.audio_output_rows != target_audio ||
        c.transformer.video_output_start != static_cast<uint32_t>(video_start_wide) ||
        c.transformer.audio_output_start != static_cast<uint32_t>(audio_start_wide)))) {
    throw std::invalid_argument("Vulkan H3 denoise: unsupported sequence config");
  }
  const uint64_t range_values =
      uint64_t((sequence + 127u) / 128u) * 4u;
  if (!c.attention_ranges.empty() &&
      c.attention_ranges.size() != range_values)
    throw std::invalid_argument("Vulkan H3 denoise: invalid captured ranges");
  std::vector<uint8_t> covered(sequence, 0);
  auto require_indices = [&](const std::vector<int32_t>& indices,
                             int32_t expected_tag) {
    for (int32_t index : indices) {
      const int32_t actual_tag = index >= 0 &&
          static_cast<uint32_t>(index) < sequence
          ? c.indices.tags[static_cast<uint32_t>(index)] : -1;
      if (index < 0 || static_cast<uint32_t>(index) >= sequence ||
          covered[static_cast<uint32_t>(index)] != 0 ||
          (expected_tag >= 0 ? actual_tag != expected_tag
                             : actual_tag < dit::kTagVideo ||
                                   actual_tag > dit::kTagAudio))
        throw std::invalid_argument(
            "Vulkan H3 denoise: indices are not a unique tagged cover");
      covered[static_cast<uint32_t>(index)] = 1;
    }
  };
  // Multimodal Qwen residual rows still use the text input projection, while
  // image-pad rows retain kTagVideo for AdaLN selection.
  require_indices(c.indices.text, -1);
  require_indices(c.indices.audio, dit::kTagAudio);
  require_indices(c.indices.video, dit::kTagVideo);
  if (std::find(covered.begin(), covered.end(), uint8_t{0}) != covered.end())
    throw std::invalid_argument("Vulkan H3 denoise: indices do not cover sequence");
  const int32_t condition_begin = l.num_text;
  const int32_t condition_end = static_cast<int32_t>(audio_start_wide);
  for (uint32_t i = 0; i < condition_video; ++i)
    if (c.indices.video[i] < condition_begin ||
        c.indices.video[i] >= condition_end)
      throw std::invalid_argument("Vulkan H3 denoise: video anchor leaves condition interval");
  for (uint32_t i = 0; i < condition_audio; ++i)
    if (c.indices.audio[i] < condition_begin ||
        c.indices.audio[i] >= condition_end)
      throw std::invalid_argument("Vulkan H3 denoise: audio anchor leaves condition interval");
  for (uint32_t i = 0; i < target_audio; ++i)
    if (c.indices.audio[condition_audio + i] !=
        static_cast<int32_t>(audio_start_wide + i))
      throw std::invalid_argument("Vulkan H3 denoise: generated audio is not contiguous");
  for (uint32_t i = 0; i < target_video; ++i)
    if (c.indices.video[condition_video + i] !=
        static_cast<int32_t>(video_start_wide + i))
      throw std::invalid_argument("Vulkan H3 denoise: generated video is not contiguous");
  for (double position : c.position_ids)
    if (!std::isfinite(position))
      throw std::invalid_argument("Vulkan H3 denoise: non-finite position");
}

}  // namespace

struct ExactH3Denoiser::Impl {
  struct State {
    dit::AdaLNTable table;
    ExactH3Transformer transformer;
    DeviceTensor video;
    DeviceTensor audio;
    DeviceTensor video_velocity;
    DeviceTensor audio_velocity;
    DeviceTensor video_result;
    DeviceTensor audio_result;
    DeviceTensor selectors;
    DeviceTensor code;
    DeviceTensor cosine;
    DeviceTensor sine;
    DeviceTensor video_timestep_indices;
    DeviceTensor audio_timestep_indices;
    DeviceTensor video_row_indices;
    DeviceTensor audio_row_indices;
    H3AttentionRanges ranges;
  };

  TensorContext* context = nullptr;
  ExactH3DenoiseConfig config;
  std::unique_ptr<State> state;
  bool ready = false;
  bool running = false;

  Impl(TensorContext& owner, ExactH3DenoiseConfig value)
      : context(&owner), config(std::move(value)) {}
};

ExactH3Denoiser::ExactH3Denoiser() = default;
ExactH3Denoiser::~ExactH3Denoiser() = default;
ExactH3Denoiser::ExactH3Denoiser(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ExactH3Denoiser::ExactH3Denoiser(ExactH3Denoiser&&) noexcept = default;
ExactH3Denoiser& ExactH3Denoiser::operator=(ExactH3Denoiser&&) noexcept = default;

ExactH3Denoiser ExactH3Denoiser::create(
    TensorContext& context, const ExactH3DenoiseConfig& config) {
  validate_config(config);
  return ExactH3Denoiser(std::make_shared<Impl>(context, config));
}

void ExactH3Denoiser::load(const SafeTensors& checkpoint) {
  if (!impl_) throw std::logic_error("Vulkan H3 denoise: empty object");
  if (loaded()) throw std::logic_error("Vulkan H3 denoise: unload before load");
  auto next = std::make_unique<Impl::State>();
  next->table.load(checkpoint);
  next->transformer = ExactH3Transformer::create(
      *impl_->context, impl_->config.transformer);
  next->transformer.load(checkpoint);
  try {
    const auto& c = impl_->config;
    TensorContext& context = *impl_->context;
    next->video = context.allocate(
        matrix(c.transformer.video_rows, c.transformer.video_dim));
    const bool has_audio = c.transformer.audio_rows != 0;
    if (has_audio) {
      next->audio = context.allocate(
          matrix(c.transformer.audio_rows, c.transformer.audio_dim));
    }
    const uint32_t video_output = c.transformer.video_output_rows
        ? c.transformer.video_output_rows : c.transformer.video_rows;
    const uint32_t audio_output = c.transformer.audio_output_rows
        ? c.transformer.audio_output_rows : c.transformer.audio_rows;
    next->video_velocity = context.allocate(
        matrix(video_output, c.transformer.video_dim));
    if (has_audio) {
      next->audio_velocity = context.allocate(
          matrix(audio_output, c.transformer.audio_dim));
    }
    const bool conditioned = c.layout.num_condition_video != 0 ||
                             c.layout.num_condition_audio != 0;
    if (conditioned) {
      next->video_result = context.allocate(
          matrix(video_output, c.transformer.video_dim));
      if (has_audio) {
        next->audio_result = context.allocate(
            matrix(audio_output, c.transformer.audio_dim));
      }
    }
    next->selectors = context.allocate(
        vector(c.transformer.main.block.sequence), ScalarType::kInt32);
    next->code = context.allocate(matrix(
        c.transformer.main.block.timesteps, dit::AdaLNTable::kRank));
    const dit::H3RopeTables rope = dit::build_h3_rope_tables(
        c.position_ids, 10000.0f, 16);
    if (rope.rows != c.transformer.main.block.sequence ||
        rope.cosine.size() != static_cast<size_t>(rope.rows) * 96 ||
        rope.sine.size() != rope.cosine.size())
      throw std::runtime_error("Vulkan H3 denoise: invalid canonical RoPE table");
    next->cosine = context.allocate(matrix(rope.rows, 96));
    next->sine = context.allocate(matrix(rope.rows, 96));
    next->video_timestep_indices = context.allocate(
        vector(video_output), ScalarType::kInt32);
    if (has_audio) {
      next->audio_timestep_indices = context.allocate(
          vector(audio_output), ScalarType::kInt32);
    }
    if (conditioned) {
      next->video_row_indices = context.allocate(
          vector(c.transformer.video_rows), ScalarType::kInt32);
      if (has_audio) {
        next->audio_row_indices = context.allocate(
            vector(c.transformer.audio_rows), ScalarType::kInt32);
      }
      context.upload_transient_bytes(next->video_row_indices,
          c.indices.video.data(), c.indices.video.size() * sizeof(int32_t));
      if (has_audio) {
        context.upload_transient_bytes(next->audio_row_indices,
            c.indices.audio.data(), c.indices.audio.size() * sizeof(int32_t));
      }
    }
    context.upload_transient(next->cosine, rope.cosine.data(), rope.cosine.size());
    context.upload_transient(next->sine, rope.sine.data(), rope.sine.size());
    if (c.attention_band > 0 || !c.attention_ranges.empty()) {
      std::vector<int32_t> range_values = c.attention_ranges;
      if (range_values.empty()) {
        const dit::BandedKeyRanges band = dit::build_banded_key_ranges(
            c.layout, c.attention_band, 128, 64);
        range_values = band.ranges;
      }
      next->ranges = H3AttentionRanges::create(
          context, c.transformer.main.block.sequence, range_values.data(),
          static_cast<uint32_t>(range_values.size()));
    }
  } catch (...) {
    next->transformer.unload();
    throw;
  }
  impl_->state = std::move(next);
  impl_->ready = false;
}

void ExactH3Denoiser::unload() noexcept {
  if (!impl_ || impl_->running) return;
  impl_->state.reset();
  try { impl_->context->collect(); } catch (...) {}
  impl_->ready = false;
}
bool ExactH3Denoiser::loaded() const noexcept {
  return impl_ && impl_->state && impl_->state->transformer.loaded();
}
bool ExactH3Denoiser::prepared() const noexcept {
  return loaded() && impl_->ready && impl_->state->transformer.text_prepared();
}
const ExactH3DenoiseConfig& ExactH3Denoiser::config() const noexcept {
  return impl_->config;
}

void ExactH3Denoiser::prepare(
    const float* prompt, uint64_t prompt_elements,
    const float* video_rows, uint64_t video_elements,
    const float* audio_rows, uint64_t audio_elements,
    const H3TransformerTextReplayTaps* taps) {
  if (!loaded()) throw std::logic_error("Vulkan H3 denoise: not loaded");
  if (impl_->running) throw std::logic_error("Vulkan H3 denoise: run is active");
  const auto& c = impl_->config.transformer;
  const uint64_t expected_prompt = uint64_t(c.text_rows) * c.text_dim;
  const uint64_t expected_video = uint64_t(c.video_rows) * c.video_dim;
  const uint64_t expected_audio = uint64_t(c.audio_rows) * c.audio_dim;
  if (prompt_elements != expected_prompt || video_elements != expected_video ||
      audio_elements != expected_audio || !finite_span(prompt, prompt_elements) ||
      !finite_span(video_rows, video_elements) ||
      !finite_span(audio_rows, audio_elements))
    throw std::invalid_argument("Vulkan H3 denoise: invalid prepare inputs");
  Impl::State& s = *impl_->state;
  const uint32_t condition_video =
      static_cast<uint32_t>(impl_->config.layout.num_condition_video);
  const uint32_t condition_audio =
      static_cast<uint32_t>(impl_->config.layout.num_condition_audio);
  const uint32_t video_output = c.video_output_rows
      ? c.video_output_rows : c.video_rows;
  const uint32_t audio_output = c.audio_output_rows
      ? c.audio_output_rows : c.audio_rows;
  DeviceTensor prompt_tensor = impl_->context->allocate(
      matrix(c.text_rows, c.text_dim));
  std::vector<TensorUpload> base_uploads{
      {&prompt_tensor, prompt, prompt_elements * sizeof(float)},
      {&s.video, video_rows, video_elements * sizeof(float)}};
  if (expected_audio != 0) {
    base_uploads.push_back(
        {&s.audio, audio_rows, audio_elements * sizeof(float)});
  }
  impl_->context->upload_batch(base_uploads.data(),
                               static_cast<uint32_t>(base_uploads.size()));
  if (condition_video != 0 || condition_audio != 0) {
    std::vector<TensorUpload> result_uploads{
        {&s.video_result,
         video_rows + uint64_t(condition_video) * c.video_dim,
         uint64_t(video_output) * c.video_dim * sizeof(float)}};
    if (expected_audio != 0) {
      result_uploads.push_back(
          {&s.audio_result,
           audio_rows + uint64_t(condition_audio) * c.audio_dim,
           uint64_t(audio_output) * c.audio_dim * sizeof(float)});
    }
    impl_->context->upload_batch(
        result_uploads.data(), static_cast<uint32_t>(result_uploads.size()));
  }
  s.transformer.prepare_text(prompt_tensor, taps);
  impl_->ready = true;
}

ExactH3DenoiseResult ExactH3Denoiser::run(
    const sampler::FlowScheduler& video,
    const sampler::FlowScheduler& audio,
    const ExactH3DenoiseProgress& progress,
    const ExactH3DenoiseBoundary& boundary,
    const H3TransformerForwardReplayTaps* taps) {
  if (!prepared()) throw std::logic_error("Vulkan H3 denoise: not prepared");
  if (impl_->running) throw std::logic_error("Vulkan H3 denoise: run is active");
  if (video.sampler() != sampler::SamplerKind::kEuler ||
      audio.sampler() != sampler::SamplerKind::kEuler)
    throw std::invalid_argument("Vulkan H3 denoise: only exact Euler is supported");
  if (video.num_steps() == 0 || video.num_steps() != audio.num_steps() ||
      video.timesteps().size() != video.sigmas().size() - 1 ||
      audio.timesteps().size() != audio.sigmas().size() - 1)
    throw std::invalid_argument("Vulkan H3 denoise: incompatible schedules");
  struct RunGuard {
    bool& value;
    ~RunGuard() { value = false; }
  } guard{impl_->running};
  impl_->running = true;

  Impl::State& s = *impl_->state;
  const auto& c = impl_->config;
  const uint32_t code_rows = c.transformer.main.block.timesteps;
  const uint32_t condition_video = static_cast<uint32_t>(c.layout.num_condition_video);
  const uint32_t condition_audio = static_cast<uint32_t>(c.layout.num_condition_audio);
  const uint32_t video_output = c.transformer.video_output_rows
      ? c.transformer.video_output_rows : c.transformer.video_rows;
  const uint32_t audio_output = c.transformer.audio_output_rows
      ? c.transformer.audio_output_rows : c.transformer.audio_rows;
  const bool has_audio = c.transformer.audio_rows != 0;
  if (c.pin_target_audio && c.layout.num_audio_rows == 0)
    throw std::invalid_argument("Vulkan H3 denoise: pinned target audio needs target rows");
  std::vector<float> code(uint64_t(code_rows) * dit::AdaLNTable::kRank);
  std::vector<int32_t> video_indices(video_output);
  std::vector<int32_t> audio_indices(audio_output);
  ExactH3DenoiseResult result;
  const uint32_t steps = static_cast<uint32_t>(video.num_steps());
  const bool conditioned = condition_video != 0 || condition_audio != 0;
  for (uint32_t step = 0; step < steps; ++step) {
    const float video_t = video.timesteps()[step];
    const float audio_t = c.pin_target_audio ? 1.0f : audio.timesteps()[step];
    const dit::RowTimesteps row = conditioned
        ? dit::build_row_timesteps(c.layout, c.indices, video_t, audio_t,
                                   std::max(video_t, 0.999f), 1.0f)
        : dit::build_row_timesteps(c.layout, c.indices, video_t, audio_t);
    if (row.unique.empty() || row.unique.size() > code_rows ||
        row.adaln.size() != c.transformer.main.block.sequence)
      throw std::runtime_error("Vulkan H3 denoise: invalid row timestep plan");
    for (uint32_t t = 0; t < code_rows; ++t) {
      const float value = row.unique[std::min<size_t>(t, row.unique.size() - 1)];
      const auto selected = s.table.lookup(value);
      std::copy(selected.begin(), selected.end(),
                code.begin() + uint64_t(t) * dit::AdaLNTable::kRank);
    }
    for (uint32_t i = 0; i < video_output; ++i)
      video_indices[i] = row.indices[static_cast<size_t>(
          c.indices.video[condition_video + i])];
    for (uint32_t i = 0; i < audio_output; ++i)
      audio_indices[i] = row.indices[static_cast<size_t>(
          c.indices.audio[condition_audio + i])];
    std::vector<TensorUpload> controls{
        {&s.selectors, row.adaln.data(), row.adaln.size() * sizeof(int32_t)},
        {&s.code, code.data(), code.size() * sizeof(float)},
        {&s.video_timestep_indices, video_indices.data(),
         video_indices.size() * sizeof(int32_t)}};
    if (has_audio) {
      controls.push_back(
          {&s.audio_timestep_indices, audio_indices.data(),
           audio_indices.size() * sizeof(int32_t)});
    }
    impl_->context->upload_batch(controls.data(),
                                 static_cast<uint32_t>(controls.size()));

    const float video_sigma = 1.0f - video_t;
    const float audio_sigma = 1.0f - audio_t;
    const float video_ratio = video.sigmas()[step + 1] / video.sigmas()[step];
    const float audio_ratio = audio.sigmas()[step + 1] / audio.sigmas()[step];
    TensorBatch batch = impl_->context->begin_batch();
    batch.require_operator_capacity(required_step_operators(taps));
    if (conditioned) {
      batch.copy_rows(s.video_result, s.video, 0, condition_video, video_output);
      if (has_audio) {
        batch.copy_rows(s.audio_result, s.audio, 0, condition_audio, audio_output);
      }
    }
    s.transformer.record_forward(
        batch, s.video, s.audio, s.selectors, s.code, s.cosine, s.sine,
        s.video_timestep_indices, s.audio_timestep_indices,
        s.video_velocity, s.audio_velocity,
        s.ranges ? &s.ranges : nullptr, taps,
        conditioned ? &s.video_row_indices : nullptr,
        conditioned && has_audio ? &s.audio_row_indices : nullptr);
    DeviceTensor& video_state = conditioned ? s.video_result : s.video;
    batch.dit_euler_step_f32(video_state, s.video_velocity,
                             video_sigma, video_ratio);
    if (has_audio && !c.pin_target_audio) {
      DeviceTensor& audio_state = conditioned ? s.audio_result : s.audio;
      batch.dit_euler_step_f32(audio_state, s.audio_velocity,
                               audio_sigma, audio_ratio);
    }
    batch.submit().wait();
    result.steps_completed = step + 1;
    if (boundary) {
      result.video_rows.resize(uint64_t(video_output) *
                               c.transformer.video_dim);
      result.audio_rows.resize(uint64_t(audio_output) *
                               c.transformer.audio_dim);
      impl_->context->download(video_state, result.video_rows.data(),
                               result.video_rows.size());
      if (has_audio) {
        DeviceTensor& audio_state = conditioned ? s.audio_result : s.audio;
        impl_->context->download(audio_state, result.audio_rows.data(),
                                 result.audio_rows.size());
      }
      boundary(step, result.video_rows, result.audio_rows);
    }
    if (progress && !progress(step, steps)) {
      result.cancelled = true;
      break;
    }
  }
  result.video_rows.resize(uint64_t(video_output) *
                           c.transformer.video_dim);
  result.audio_rows.resize(uint64_t(audio_output) *
                           c.transformer.audio_dim);
  DeviceTensor& final_video = conditioned ? s.video_result : s.video;
  impl_->context->download(final_video, result.video_rows.data(),
                           result.video_rows.size());
  if (has_audio) {
    DeviceTensor& final_audio = conditioned ? s.audio_result : s.audio;
    impl_->context->download(final_audio, result.audio_rows.data(),
                             result.audio_rows.size());
  }
  return result;
}

uint64_t ExactH3Denoiser::persistent_bytes() const noexcept {
  return loaded() ? impl_->state->transformer.persistent_bytes() : 0;
}
uint64_t ExactH3Denoiser::scratch_bytes() const noexcept {
  if (!loaded()) return 0;
  const Impl::State& s = *impl_->state;
  uint64_t total = s.transformer.scratch_bytes();
  for (const DeviceTensor* tensor : {
           &s.video, &s.audio, &s.video_velocity, &s.audio_velocity,
           &s.video_result, &s.audio_result,
           &s.selectors, &s.code, &s.cosine, &s.sine,
           &s.video_timestep_indices, &s.audio_timestep_indices,
           &s.video_row_indices, &s.audio_row_indices}) {
    const uint64_t bytes = tensor_bytes(*tensor);
    if (bytes > std::numeric_limits<uint64_t>::max() - total)
      return std::numeric_limits<uint64_t>::max();
    total += bytes;
  }
  const uint64_t range_bytes = s.ranges.resident_bytes();
  if (range_bytes > std::numeric_limits<uint64_t>::max() - total)
    return std::numeric_limits<uint64_t>::max();
  total += range_bytes;
  return total;
}
uint64_t ExactH3Denoiser::peak_device_bytes() const noexcept {
  const uint64_t persistent = persistent_bytes();
  const uint64_t scratch = scratch_bytes();
  return scratch > std::numeric_limits<uint64_t>::max() - persistent
      ? std::numeric_limits<uint64_t>::max() : persistent + scratch;
}

uint32_t ExactH3Denoiser::required_step_operators(
    const H3TransformerForwardReplayTaps* taps) const {
  if (!loaded()) throw std::logic_error("Vulkan H3 denoise: not loaded");
  const uint32_t forward =
      impl_->state->transformer.required_forward_operators(taps);
  const bool conditioned = impl_->config.layout.num_condition_video != 0 ||
                           impl_->config.layout.num_condition_audio != 0;
  const bool has_audio = impl_->config.transformer.audio_rows != 0;
  const uint32_t tail = (conditioned ? 2u : 1u) * (has_audio ? 2u : 1u) -
      (has_audio && impl_->config.pin_target_audio ? 1u : 0u);
  if (forward > UINT32_MAX - tail)
    throw std::overflow_error("Vulkan H3 denoise: step operator overflow");
  return forward + tail;
}

}  // namespace slopfab::vulkan
