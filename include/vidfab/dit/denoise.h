// The t2va denoising loop.
//
// One transformer evaluation per step — the released checkpoints are
// CFG-distilled, so there is no guider, no negative prompt and no second
// forward pass. Two schedulers run inside one loop, stepped independently with
// their own timestep and their own sigma grid: shift 12.0 for video, 3.0 for
// audio.
#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include "vidfab/dit/packing.h"
#include "vidfab/dit/step_cache.h"
#include "vidfab/dit/transformer.h"
#include "vidfab/sampler/scheduler.h"

namespace vidfab::dit {

// Produces the two velocity predictions for one step, in place of the
// transformer. See `DenoiseInputs::velocity`.
using VelocityFn = std::function<void(int step, const RowTimesteps& row_timesteps,
                                      const float* video_rows, const float* audio_rows,
                                      float* video_velocity, float* audio_velocity)>;

struct DenoiseInputs {
  const SequenceLayout* layout = nullptr;
  const PackedIndices* indices = nullptr;

  // Substitutes the transformer. Null in production, in which case
  // `Transformer::forward` runs.
  //
  // This is here because the loop's arithmetic is worth pinning on its own and
  // is otherwise reachable only through a 19.5 GiB checkpoint and forty minutes
  // of GPU time. Three things in it are easy to get wrong and invisible in the
  // output shape: the plus in `x + sigma*v`, the two schedulers advancing on
  // their own sigma grids inside one iteration, and the terminal ratio of zero
  // that makes the last step return the denoised estimate outright.
  VelocityFn velocity;

  // Step caching. Disabled by default (`threshold == 0`, `skip_every == 0`),
  // in which case the loop below is the loop it was, plus one host branch per
  // step that is always taken.
  StepCacheConfig cache;

  // Substitutes the transformer's AdaLN lookup when building a step's
  // conditioning signature, for the same reason `velocity` substitutes the
  // forward pass. Null in production, where `Transformer::adaln_code` runs and
  // honours the configured lookup mode.
  CodeFn code;

  // Schedules, already validated to be the same length by resolve_plan.
  const std::vector<float>* video_timesteps = nullptr;
  const std::vector<float>* audio_timesteps = nullptr;
  // Non-const because a second-order sampler keeps v_{n-1} inside the
  // scheduler, one buffer per modality. The loop below is unaware of it.
  sampler::FlowScheduler* video_scheduler = nullptr;
  sampler::FlowScheduler* audio_scheduler = nullptr;

  uint64_t seed = 0;

  // Initial latents, replacing the seeded draw. Null in production — the loop
  // then draws from `seed` exactly as the reference does.
  //
  // This exists for the frame-banding quality probe, which runs one request as
  // several overlapping shorter ones and needs each of them to start from its
  // *slice of the full request's noise field* rather than from an independent
  // chunk-sized draw. A chunk-sized draw would be a different field entirely
  // (the generator is indexed by flat position, so F changes the stride), and
  // the probe would then be measuring three unrelated samples instead of the
  // loss of cross-chunk attention. Shapes are checked against the layout.
  const std::vector<float>* init_video_rows = nullptr;
  const std::vector<float>* init_audio_rows = nullptr;
};

struct DenoiseOutputs {
  std::vector<float> video_rows;  // [V, 96] fp32
  std::vector<float> audio_rows;  // [Sa, 32] fp32

  // How the loop actually spent its evaluations. Reported rather than inferred:
  // a cache threshold whose skip count is not printed cannot be reasoned about
  // at all, since the same flag value skips a different number of steps at
  // every geometry and schedule length.
  int steps_computed = 0;
  int steps_skipped = 0;
};

// Called after each step with (step_index, total_steps). Return false to abort.
using ProgressFn = std::function<bool(int, int)>;

// Runs the loop. `transformer` must already have had `prepare_text` and
// `prepare_sequence` called.
DenoiseOutputs denoise(Transformer& transformer, const DenoiseInputs& inputs,
                       const ProgressFn& progress = {});

}  // namespace vidfab::dit
