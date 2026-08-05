// The t2va denoising loop (spec 1.6).
//
// One transformer call per iteration serving both modalities, then two
// independent Euler steps — the video scheduler on its shift-12 grid at
// `video_timesteps[i]`, the audio scheduler on its shift-3 grid at
// `audio_timesteps[i]`. There is no guider and no second forward pass: the
// released checkpoints are CFG-distilled.

#include "vidfab/dit/denoise.h"

#include <array>
#include <stdexcept>
#include <string>

#include "vidfab/cuda/profile.h"
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
  //
  // Supplied initial latents replace the draw entirely rather than perturbing
  // it, and both modalities are all-or-nothing per modality so a caller cannot
  // half-substitute one and silently get seeded noise for the rest.
  if (inputs.init_video_rows != nullptr) {
    require(inputs.init_video_rows->size() == out.video_rows.size(),
            "the supplied initial video latents disagree with the layout");
    out.video_rows = *inputs.init_video_rows;
  } else {
    const std::vector<float> noise = sampler::video_noise(
        inputs.seed, layout.num_latent_frames, layout.latent_height, layout.latent_width,
        transformer.config().in_channels);
    patchify_video(noise.data(), layout, out.video_rows.data());
  }
  if (inputs.init_audio_rows != nullptr) {
    require(inputs.init_audio_rows->size() == out.audio_rows.size(),
            "the supplied initial audio latents disagree with the layout");
    out.audio_rows = *inputs.init_audio_rows;
  } else {
    const std::vector<float> noise =
        sampler::audio_noise(inputs.seed, layout.num_audio_latents, audio_dim);
    require(noise.size() == out.audio_rows.size(), "audio noise shape disagrees with the layout");
    out.audio_rows = noise;
  }

  // These persist across iterations and are the whole of the cache's storage:
  // a skipped step simply does not overwrite them, and the two schedulers
  // consume the velocities still sitting here. 14.3 MB at the default geometry,
  // already allocated, so the feature costs no memory at all.
  std::vector<float> video_velocity(out.video_rows.size(), 0.0f);
  std::vector<float> audio_velocity(out.audio_rows.size(), 0.0f);

  const int steps = static_cast<int>(video_t.size());
  StepCache cache(inputs.cache, steps);

  // The signature of a step, `c(t_v)` then `c(t_a)`. Built only when the cache
  // is on, so a default run never touches the AdaLN table here.
  std::vector<float> signature;
  if (cache.enabled()) signature.resize(2 * static_cast<size_t>(AdaLNTable::kRank));

  for (int i = 0; i < steps; ++i) {
    bool compute = true;
    if (cache.enabled()) {
      cuda::HostSpan span("step_cache");
      // Two distinct timesteps per step for t2va (spec 7.5), on grids of
      // different shift that move at different rates, so both go into the
      // distance. `adaln_code` honours the configured lookup mode rather than
      // hardcoding one — the grid semantics are unresolved and the mode is
      // deliberately a knob (spec 3.5).
      const std::array<float, AdaLNTable::kRank> cv =
          inputs.code ? inputs.code(video_t[static_cast<size_t>(i)])
                      : transformer.adaln_code(video_t[static_cast<size_t>(i)]);
      const std::array<float, AdaLNTable::kRank> ca =
          inputs.code ? inputs.code(audio_t[static_cast<size_t>(i)])
                      : transformer.adaln_code(audio_t[static_cast<size_t>(i)]);
      for (int k = 0; k < AdaLNTable::kRank; ++k) {
        signature[static_cast<size_t>(k)] = cv[static_cast<size_t>(k)];
        signature[static_cast<size_t>(AdaLNTable::kRank + k)] = ca[static_cast<size_t>(k)];
      }
      compute = cache.should_compute(i, signature.data(), static_cast<int>(signature.size()));
    } else {
      compute = cache.should_compute(i, nullptr, 0);
    }

    // The skip. Not calling `forward` is the entire mechanism; the velocity
    // buffers below still hold the last computed prediction.
    //
    // NOTE for whoever lands the AB2 sampler: if `--sampler ab2` and a nonzero
    // `--cache-threshold` are ever enabled together, AB2's velocity history is
    // built partly from *reused* velocities rather than from fresh evaluations,
    // so its two-point extrapolation is extrapolating a constant over the
    // skipped interval. That is recorded, not solved, and the two speedups are
    // not multiplicative either.
    if (compute) {
      RowTimesteps row_timesteps;
      {
        // Rebuilt every step because `torch.unique(sorted=True)` reorders the
        // two timesteps as the schedules cross, so this is not cacheable. It is
        // pure host work over `seq` rows and it is on the critical path — and
        // it is the *only* per-step work a skipped step also avoids, which is
        // why it sits inside this branch rather than above it.
        cuda::HostSpan span("build_row_timesteps");
        row_timesteps = build_row_timesteps(layout, indices, video_t[static_cast<size_t>(i)],
                                            audio_t[static_cast<size_t>(i)]);
      }
      if (inputs.velocity) {
        inputs.velocity(i, row_timesteps, out.video_rows.data(), out.audio_rows.data(),
                        video_velocity.data(), audio_velocity.data());
      } else {
        transformer.forward(out.video_rows.data(), out.audio_rows.data(), row_timesteps,
                            video_velocity.data(), audio_velocity.data());
      }
    }

    {
      // In place: FlowScheduler::step permits `out` to alias `sample`.
      cuda::HostSpan span("scheduler_step");
      inputs.video_scheduler->step(i, out.video_rows.data(), video_velocity.data(),
                                   out.video_rows.size(), out.video_rows.data());
      inputs.audio_scheduler->step(i, out.audio_rows.data(), audio_velocity.data(),
                                   out.audio_rows.size(), out.audio_rows.data());
    }

    if (progress && !progress(i, steps)) break;
  }

  out.steps_computed = cache.computed();
  out.steps_skipped = cache.skipped();
  return out;
}

}  // namespace vidfab::dit
