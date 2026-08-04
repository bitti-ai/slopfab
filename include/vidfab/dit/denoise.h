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

  // Schedules, already validated to be the same length by resolve_plan.
  const std::vector<float>* video_timesteps = nullptr;
  const std::vector<float>* audio_timesteps = nullptr;
  const sampler::FlowScheduler* video_scheduler = nullptr;
  const sampler::FlowScheduler* audio_scheduler = nullptr;

  uint64_t seed = 0;
};

struct DenoiseOutputs {
  std::vector<float> video_rows;  // [V, 96] fp32
  std::vector<float> audio_rows;  // [Sa, 32] fp32
};

// Called after each step with (step_index, total_steps). Return false to abort.
using ProgressFn = std::function<bool(int, int)>;

// Runs the loop. `transformer` must already have had `prepare_text` and
// `prepare_sequence` called.
DenoiseOutputs denoise(Transformer& transformer, const DenoiseInputs& inputs,
                       const ProgressFn& progress = {});

}  // namespace vidfab::dit
