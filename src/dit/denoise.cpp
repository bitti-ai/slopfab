// The t2va denoising loop (spec 1.6).
//
// One transformer call per iteration serving both modalities, then two
// independent scheduler updates. With unequal counts, update boundaries are
// interleaved and each modality keeps its own step index and noise level.
// There is no guider and no second forward pass: the
// released checkpoints are CFG-distilled.

#include "slopfab/dit/denoise.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

#include "slopfab/cuda/profile.h"
#include "slopfab/sampler/noise.h"
#include "slopfab/sampler/joint_schedule.h"

namespace slopfab::dit {
namespace {

void require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(std::string("denoise: ") + message);
}

} // namespace

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

  require(video_t == inputs.video_scheduler->timesteps() &&
              audio_t == inputs.audio_scheduler->timesteps(),
          "timestep lists disagree with schedulers");
  const auto schedule = sampler::joint_schedule(video_t.size(), audio_t.size());
  const bool independent = video_t.size() != audio_t.size();
  require(!inputs.continuation ||
              (!inputs.inpaint && !inputs.pin_target_audio && !inputs.cache.enabled() &&
               !inputs.motion_cache.active() && !transformer.block_cache_config().enabled()),
          "locked continuation overlap is incompatible with image editing, pinned audio or caches");
  require(!independent || (!inputs.pin_target_audio && layout.num_audio_rows > 0),
          "independent audio steps require generated target audio");
  require(!independent || (!inputs.cache.enabled() && !inputs.motion_cache.active() &&
                           !transformer.block_cache_config().enabled()),
          "independent audio steps do not support approximate caches");
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
  if (cv)
    require(inputs.condition_video_rows->size() == cv,
            "condition video shape disagrees with layout");
  if (ca)
    require(inputs.condition_audio_rows->size() == ca,
            "condition audio shape disagrees with layout");
  std::vector<float> all_video(video_rows * patch, 0.0f);
  const size_t audio_values = audio_rows * static_cast<size_t>(audio_dim);
  // Keep a valid address even for the video-only still path. The transformer
  // and optional capture hooks receive the pointer alongside a logical row
  // count of zero; a one-float sentinel avoids null-pointer arithmetic in
  // instrumentation without introducing an audio token or output sample.
  std::vector<float> all_audio(std::max<size_t>(audio_values, 1), 0.0f);
  if (cv)
    std::copy(inputs.condition_video_rows->begin(), inputs.condition_video_rows->end(),
              all_video.begin());
  if (ca)
    std::copy(inputs.condition_audio_rows->begin(), inputs.condition_audio_rows->end(),
              all_audio.begin());
  out.video_rows.assign(all_video.size() - cv, 0.0f);
  out.audio_rows.assign(audio_values - ca, 0.0f);
  require(!inputs.pin_target_audio || (inputs.init_audio_rows && !out.audio_rows.empty()),
          "pinned target audio requires nonempty initial audio rows");

  // Draw order matters for reproducibility even though our generator is not
  // torch's: video first, in `(24, F, Hl, Wl)` layout and then patchified, then
  // audio drawn directly in row layout `(2A, 32)` (spec 1.3).
  //
  // Supplied initial latents replace the draw entirely rather than perturbing
  // it, and both modalities are all-or-nothing per modality so a caller cannot
  // half-substitute one and silently get seeded noise for the rest.
  if (inputs.inpaint) {
    require(inputs.init_video_rows == nullptr, "inpainting replaces initial video latents");
    inputs.inpaint->validate(out.video_rows.size());
    out.video_rows = inputs.inpaint->initial(inputs.video_scheduler->sigmas().front());
  } else if (inputs.init_video_rows != nullptr) {
    require(inputs.init_video_rows->size() == out.video_rows.size(),
            "the supplied initial video latents disagree with the layout");
    out.video_rows = *inputs.init_video_rows;
  } else {
    const std::vector<float> noise =
        sampler::video_noise(inputs.seed, layout.num_latent_frames, layout.latent_height,
                             layout.latent_width, transformer.config().in_channels);
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
  ContinuationConstraint overlap;
  if (inputs.continuation) {
    overlap = *inputs.continuation;
    overlap.video.capture_noise(out.video_rows.data(), out.video_rows.size());
    overlap.audio.capture_noise(out.audio_rows.data(), out.audio_rows.size());
    overlap.video.apply(out.video_rows.data(), out.video_rows.size(),
                        inputs.video_scheduler->sigmas().front());
    overlap.audio.apply(out.audio_rows.data(), out.audio_rows.size(),
                        inputs.audio_scheduler->sigmas().front());
  }
  std::copy(out.video_rows.begin(), out.video_rows.end(), all_video.begin() + cv);
  std::copy(out.audio_rows.begin(), out.audio_rows.end(), all_audio.begin() + ca);

  // These persist across iterations and are the whole of the cache's storage:
  // a skipped step simply does not overwrite them, and the two schedulers
  // consume the velocities still sitting here. 14.3 MB at the default geometry,
  // already allocated, so the feature costs no memory at all.
  std::vector<float> video_velocity(all_video.size(), 0.0f);
  std::vector<float> audio_velocity(std::max<size_t>(audio_values, 1), 0.0f);

  const int steps = static_cast<int>(schedule.size());
  StepCache cache(inputs.cache, steps);
  MotionCache motion(inputs.motion_cache, layout, patch, audio_dim, steps,
                     inputs.video_scheduler->shift(), inputs.pin_target_audio);
  const bool renoise = inputs.video_scheduler->sampler() == sampler::SamplerKind::kRenoise;
  require(renoise == (inputs.audio_scheduler->sampler() == sampler::SamplerKind::kRenoise),
          "re-noising must be selected for both modalities");
  require(!renoise || (!cache.enabled() && !motion.enabled() &&
                       !transformer.block_cache_config().enabled()),
          "re-noising does not support approximate caches");
  std::vector<float> video_noise(renoise ? out.video_rows.size() : 0);
  std::vector<float> audio_noise(renoise ? out.audio_rows.size() : 0);
  require(!motion.enabled() || (!cache.enabled() && !transformer.block_cache_config().enabled() &&
                                inputs.video_scheduler->sampler() == sampler::SamplerKind::kEuler &&
                                inputs.audio_scheduler->sampler() == sampler::SamplerKind::kEuler),
          "MotionCache requires Euler without step or block caching");

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
    const auto& update = schedule[static_cast<size_t>(i)];
    const float vt = video_t[update.video];
    const float at = inputs.pin_target_audio ? 1.0f : audio_t[update.audio];
    bool compute = true;
    if (motion.enabled()) {
      compute = motion.should_compute(i, 1.0f - vt, all_video.data() + cv, all_audio.data() + ca);
    } else if (cache.enabled()) {
      cuda::HostSpan span("step_cache");
      // Shared with `plan_step_cache`, so the loop and the planner cannot
      // disagree about what this step's signature is — only about what to do
      // with it, which is the thing under test.
      build_signature(code, vt, at, signature);
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
        row_timesteps =
            layout.condition_audio_is_explicit
                ? build_row_timesteps(layout, indices, vt, at, std::max(vt, 0.999f), 1.0f)
                : build_row_timesteps(layout, indices, vt, at);
      }
      if (inputs.velocity) {
        inputs.velocity(i, row_timesteps, all_video.data(), all_audio.data(), video_velocity.data(),
                        audio_velocity.data());
      } else {
        transformer.set_denoise_step(i);
        transformer.forward(all_video.data(), all_audio.data(), row_timesteps,
                            video_velocity.data(), audio_velocity.data());
      }
      motion.update(1.0f - vt, all_video.data() + cv, all_audio.data() + ca,
                    video_velocity.data() + cv, audio_velocity.data() + ca);
    } else if (motion.enabled()) {
      motion.reuse(all_video.data() + cv, all_audio.data() + ca, video_velocity.data() + cv,
                   audio_velocity.data() + ca);
    }

    {
      // In place: FlowScheduler::step permits `out` to alias `sample`.
      cuda::HostSpan span("scheduler_step");
      if (update.advance_video) {
        if (renoise && update.video + 1 < video_t.size())
          sampler::fill_renoise_normal(inputs.seed, static_cast<int>(update.video),
                                       sampler::NoiseStream::kVideoLatents, video_noise.data(),
                                       video_noise.size());
        inputs.video_scheduler->step(static_cast<int>(update.video), all_video.data() + cv,
                                     video_velocity.data() + cv, out.video_rows.size(),
                                     out.video_rows.data(), video_noise.data());
        if (inputs.inpaint)
          inputs.inpaint->apply(out.video_rows.data(), out.video_rows.size(),
                                inputs.video_scheduler->sigmas()[update.video + 1],
                                renoise ? video_noise.data() : nullptr);
        if (inputs.continuation)
          overlap.video.apply(out.video_rows.data(), out.video_rows.size(),
                              inputs.video_scheduler->sigmas()[update.video + 1]);
      }
      if (update.advance_audio && !inputs.pin_target_audio) {
        if (renoise && update.audio + 1 < audio_t.size())
          sampler::fill_renoise_normal(inputs.seed, static_cast<int>(update.audio),
                                       sampler::NoiseStream::kAudioLatents, audio_noise.data(),
                                       audio_noise.size());
        inputs.audio_scheduler->step(static_cast<int>(update.audio), all_audio.data() + ca,
                                     audio_velocity.data() + ca, out.audio_rows.size(),
                                     out.audio_rows.data(), audio_noise.data());
        if (inputs.continuation)
          overlap.audio.apply(out.audio_rows.data(), out.audio_rows.size(),
                              inputs.audio_scheduler->sigmas()[update.audio + 1]);
      }
      std::copy(out.video_rows.begin(), out.video_rows.end(), all_video.begin() + cv);
      std::copy(out.audio_rows.begin(), out.audio_rows.end(), all_audio.begin() + ca);
    }

    if (inputs.boundary)
      inputs.boundary(i, out.video_rows, out.audio_rows);

    if (progress && !progress(i, steps))
      break;
  }

  out.steps_computed = motion.enabled() ? motion.computed() : cache.computed();
  out.steps_skipped = motion.enabled() ? motion.skipped() : cache.skipped();
  return out;
}

} // namespace slopfab::dit
