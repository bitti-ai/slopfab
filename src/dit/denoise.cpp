// The t2va denoising loop (spec 1.6).
//
// One transformer call per iteration serving both modalities, then two
// independent Euler steps — the video scheduler on its shift-12 grid at
// `video_timesteps[i]`, the audio scheduler on its shift-3 grid at
// `audio_timesteps[i]`. There is no guider and no second forward pass: the
// released checkpoints are CFG-distilled.

#include "vidfab/dit/denoise.h"

#include <stdexcept>
#include <string>

#include "vidfab/sampler/noise.h"

namespace vidfab::dit {
namespace {

void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(std::string("denoise: ") + message);
}

}  // namespace

DenoiseOutputs denoise(Transformer& transformer, const DenoiseInputs& inputs,
                       const ProgressFn& progress) {
  require(inputs.layout != nullptr, "layout is null");
  require(inputs.indices != nullptr, "indices is null");
  require(inputs.video_timesteps != nullptr && inputs.audio_timesteps != nullptr,
          "both timestep schedules are required");
  require(inputs.video_scheduler != nullptr && inputs.audio_scheduler != nullptr,
          "both schedulers are required");

  const SequenceLayout& layout = *inputs.layout;
  const PackedIndices& indices = *inputs.indices;
  const std::vector<float>& video_t = *inputs.video_timesteps;
  const std::vector<float>& audio_t = *inputs.audio_timesteps;

  // The reference builds its row-timestep plan by zipping the two schedules
  // while iterating the video one, so a length mismatch truncates silently
  // (spec 9.5). resolve_plan already checks this; checking again is free.
  require(video_t.size() == audio_t.size(), "video and audio schedules differ in length");
  require(!video_t.empty(), "schedule has no model evaluations");
  // t2va has no conditioning rows; fl2va's keyframe path is explicitly out of
  // scope (spec 0.1) and would need the anchor rows re-imposed by construction.
  require(layout.num_condition_video == 0, "conditioning rows are an fl2va feature, not t2va");

  const int patch = transformer.config().video_patch_dim();
  const int audio_dim = transformer.config().audio_in_channels;
  const size_t video_rows = indices.video.size();
  const size_t audio_rows = indices.audio.size();

  DenoiseOutputs out;
  out.video_rows.assign(video_rows * patch, 0.0f);
  out.audio_rows.assign(audio_rows * audio_dim, 0.0f);

  // Draw order matters for reproducibility even though our generator is not
  // torch's: video first, in `(24, F, Hl, Wl)` layout and then patchified, then
  // audio drawn directly in row layout `(2A, 32)` (spec 1.3).
  {
    const std::vector<float> noise = sampler::video_noise(
        inputs.seed, layout.num_latent_frames, layout.latent_height, layout.latent_width,
        transformer.config().in_channels);
    patchify_video(noise.data(), layout, out.video_rows.data());
  }
  {
    const std::vector<float> noise =
        sampler::audio_noise(inputs.seed, layout.num_audio_latents, audio_dim);
    require(noise.size() == out.audio_rows.size(), "audio noise shape disagrees with the layout");
    out.audio_rows = noise;
  }

  std::vector<float> video_velocity(out.video_rows.size(), 0.0f);
  std::vector<float> audio_velocity(out.audio_rows.size(), 0.0f);

  const int steps = static_cast<int>(video_t.size());
  for (int i = 0; i < steps; ++i) {
    const RowTimesteps row_timesteps = build_row_timesteps(layout, indices, video_t[static_cast<size_t>(i)],
                                                           audio_t[static_cast<size_t>(i)]);
    if (inputs.velocity) {
      inputs.velocity(i, row_timesteps, out.video_rows.data(), out.audio_rows.data(),
                      video_velocity.data(), audio_velocity.data());
    } else {
      transformer.forward(out.video_rows.data(), out.audio_rows.data(), row_timesteps,
                          video_velocity.data(), audio_velocity.data());
    }

    // In place: FlowScheduler::step permits `out` to alias `sample`.
    inputs.video_scheduler->step(i, out.video_rows.data(), video_velocity.data(),
                                 out.video_rows.size(), out.video_rows.data());
    inputs.audio_scheduler->step(i, out.audio_rows.data(), audio_velocity.data(),
                                 out.audio_rows.size(), out.audio_rows.data());

    if (progress && !progress(i, steps)) break;
  }

  return out;
}

}  // namespace vidfab::dit
