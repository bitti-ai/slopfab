// The t2va denoising loop (spec 1.6).
//
// One transformer call per iteration serving both modalities, then two
// independent Euler steps — the video scheduler on its shift-12 grid at
// `video_timesteps[i]`, the audio scheduler on its shift-3 grid at
// `audio_timesteps[i]`. There is no guider and no second forward pass: the
// released checkpoints are CFG-distilled.

#include "vidfab/dit/denoise.h"

#include <algorithm>
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
  const int patch = transformer.config().video_patch_dim();
  const int audio_dim = transformer.config().audio_in_channels;
  const size_t video_rows = indices.video.size();
  const size_t audio_rows = indices.audio.size();

  DenoiseOutputs out;
  const size_t cv = static_cast<size_t>(layout.num_condition_video) * patch;
  const size_t ca = static_cast<size_t>(layout.num_condition_audio) * audio_dim;
  require((cv == 0) == (inputs.condition_video_rows == nullptr),
          "condition video rows are missing or unexpected");
  require((ca == 0) == (inputs.condition_audio_rows == nullptr),
          "condition audio rows are missing or unexpected");
  if (cv) require(inputs.condition_video_rows->size() == cv, "condition video shape disagrees with layout");
  if (ca) require(inputs.condition_audio_rows->size() == ca, "condition audio shape disagrees with layout");
  std::vector<float> all_video(video_rows * patch, 0.0f);
  const size_t audio_values = audio_rows * static_cast<size_t>(audio_dim);
  // Keep a valid address even for the video-only still path. The transformer
  // and optional capture hooks receive the pointer alongside a logical row
  // count of zero; a one-float sentinel avoids null-pointer arithmetic in
  // instrumentation without introducing an audio token or output sample.
  std::vector<float> all_audio(std::max<size_t>(audio_values, 1), 0.0f);
  if (cv) std::copy(inputs.condition_video_rows->begin(), inputs.condition_video_rows->end(), all_video.begin());
  if (ca) std::copy(inputs.condition_audio_rows->begin(), inputs.condition_audio_rows->end(), all_audio.begin());
  out.video_rows.assign(all_video.size() - cv, 0.0f);
  out.audio_rows.assign(audio_values - ca, 0.0f);

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
  } else if (!out.audio_rows.empty()) {
    const std::vector<float> noise =
        sampler::audio_noise(inputs.seed, layout.num_audio_latents, audio_dim);
    require(noise.size() == out.audio_rows.size(), "audio noise shape disagrees with the layout");
    out.audio_rows = noise;
  }
  std::copy(out.video_rows.begin(), out.video_rows.end(), all_video.begin() + cv);
  std::copy(out.audio_rows.begin(), out.audio_rows.end(), all_audio.begin() + ca);

  // These persist across iterations and are the whole of the cache's storage:
  // a skipped step simply does not overwrite them, and the two schedulers
  // consume the velocities still sitting here. 14.3 MB at the default geometry,
  // already allocated, so the feature costs no memory at all.
  std::vector<float> video_velocity(all_video.size(), 0.0f);
  std::vector<float> audio_velocity(std::max<size_t>(audio_values, 1), 0.0f);

  const int steps = static_cast<int>(video_t.size());
  StepCache cache(inputs.cache, steps);

  // The signature of a step, `c(t_v)` then `c(t_a)`. Built only when the cache
  // is on, so a default run never touches the AdaLN table here.
  //
  // `adaln_code` honours the configured lookup mode rather than hardcoding one
  // — the grid semantics are unresolved and the mode is deliberately a knob
  // (spec 3.5). Two distinct timesteps per step for t2va (spec 7.5), on grids
  // of different shift that move at different rates, so both go into it.
  const CodeFn code = inputs.code ? inputs.code : CodeFn([&transformer](float t) {
    return transformer.adaln_code(t);
  });
  std::vector<float> signature;

  out.decisions.reserve(static_cast<size_t>(std::max(0, steps)));
  for (int i = 0; i < steps; ++i) {
    bool compute = true;
    if (cache.enabled()) {
      cuda::HostSpan span("step_cache");
      // Shared with `plan_step_cache`, so the loop and the planner cannot
      // disagree about what this step's signature is — only about what to do
      // with it, which is the thing under test.
      build_signature(code, video_t[static_cast<size_t>(i)], audio_t[static_cast<size_t>(i)],
                      signature);
      compute = cache.should_compute(i, signature.data(), static_cast<int>(signature.size()));
    } else {
      compute = cache.should_compute(i, nullptr, 0);
    }
    // Recorded here, from the variable that gates the call below, before
    // anything can act on it.
    out.decisions.push_back(compute ? 1u : 0u);

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
        const float vt = video_t[static_cast<size_t>(i)];
        const float at = audio_t[static_cast<size_t>(i)];
        row_timesteps = layout.condition_audio_is_explicit
                            ? build_row_timesteps(layout, indices, vt, at,
                                                  std::max(vt, 0.999f), 1.0f)
                            : build_row_timesteps(layout, indices, vt, at);
      }
      if (inputs.velocity) {
        inputs.velocity(i, row_timesteps, all_video.data(), all_audio.data(),
                        video_velocity.data(), audio_velocity.data());
      } else {
        transformer.set_denoise_step(i);
        transformer.forward(all_video.data(), all_audio.data(), row_timesteps,
                            video_velocity.data(), audio_velocity.data());
      }
    }

    {
      // In place: FlowScheduler::step permits `out` to alias `sample`.
      cuda::HostSpan span("scheduler_step");
      inputs.video_scheduler->step(i, all_video.data() + cv, video_velocity.data() + cv,
                                   out.video_rows.size(), out.video_rows.data());
      inputs.audio_scheduler->step(i, all_audio.data() + ca, audio_velocity.data() + ca,
                                   out.audio_rows.size(), out.audio_rows.data());
      std::copy(out.video_rows.begin(), out.video_rows.end(), all_video.begin() + cv);
      std::copy(out.audio_rows.begin(), out.audio_rows.end(), all_audio.begin() + ca);
    }

    if (inputs.boundary) inputs.boundary(i, out.video_rows, out.audio_rows);

    if (progress && !progress(i, steps)) break;
  }

  out.steps_computed = cache.computed();
  out.steps_skipped = cache.skipped();
  return out;
}

}  // namespace vidfab::dit
