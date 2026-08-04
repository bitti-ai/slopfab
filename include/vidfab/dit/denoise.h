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

struct DenoiseInputs {
  const SequenceLayout* layout = nullptr;
  const PackedIndices* indices = nullptr;

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
