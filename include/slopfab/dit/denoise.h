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

#include "slopfab/dit/packing.h"
#include "slopfab/dit/step_cache.h"
#include "slopfab/dit/motion_cache.h"
#include "slopfab/dit/transformer.h"
#include "slopfab/sampler/scheduler.h"

namespace slopfab::dit {

// Produces the two velocity predictions for one step, in place of the
// transformer. See `DenoiseInputs::velocity`.
using VelocityFn = std::function<void(int step, const RowTimesteps& row_timesteps,
                                      const float* video_rows, const float* audio_rows,
                                      float* video_velocity, float* audio_velocity)>;
using DenoiseBoundaryFn = std::function<void(
    int step, const std::vector<float>& video_rows,
    const std::vector<float>& audio_rows)>;

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
  MotionCacheConfig motion_cache;

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
  // Supplied target audio stays clean (t=1) and is never scheduler-updated.
  // It remains in the target slice so the normal output decoder returns it.
  bool pin_target_audio = false;

  // Fixed Ref2VA anchors, in the same condition-first order as PackedIndices.
  // They are projected by every transformer evaluation but never stepped.
  const std::vector<float>* condition_video_rows = nullptr;
  const std::vector<float>* condition_audio_rows = nullptr;

  // Optional verification hook after both modality updates. Production leaves
  // this null; it never changes the trajectory and exists to pin every durable
  // scheduler boundary without duplicating the canonical loop in a test.
  DenoiseBoundaryFn boundary;
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

  // The compute/skip decision the loop actually took at each step: 1 = the
  // transformer ran, 0 = the previous velocity was reused. Length is the number
  // of model evaluations in the schedule.
  //
  // This exists because the counts above are a *weak* check. They catch
  // "skipped 14 when the plan said 12" and miss "skipped 12, two of them the
  // wrong steps" — and a swap is the more likely failure, since an off-by-one
  // in the warmup or terminal guard moves *which* steps rather than how many.
  // The by-skip-position analysis is the one that survives the trajectory noise
  // floor, so it is the one carrying the result, and it would be describing a
  // schedule that did not happen.
  //
  // The bug direction here is "faster", which is the direction nobody
  // interrogates: skipping more steps, or different ones, looks like the
  // feature working well and beats whatever was pre-registered. A
  // pre-registered number is no protection against that; only a reference is,
  // and this is the reference.
  //
  // Recorded from the same variable that gates the forward call, not
  // re-derived from the config — a re-derivation would agree with the planner
  // by construction and check nothing.
  std::vector<uint8_t> decisions;
};

// Called after each step with (step_index, total_steps). Return false to abort.
using ProgressFn = std::function<bool(int, int)>;

// Runs the loop. `transformer` must already have had `prepare_text` and
// `prepare_sequence` called.
DenoiseOutputs denoise(Transformer& transformer, const DenoiseInputs& inputs,
                       const ProgressFn& progress = {});

}  // namespace slopfab::dit
