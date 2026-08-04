// Running a resolved request.
//
// `pipeline.h` turns a request into a plan without touching a weight file;
// this runs the plan. It lives apart from `pipeline.h` because it needs the
// decoders, and the dependency in this project runs one way — cuda depends on
// core, never back — so the orchestration that touches both belongs on the
// cuda side.
//
// The stages run strictly in sequence because they do not fit together: the
// int8 conditioner is 24.4 GB resident and the fp8 transformer 19.3 GB, on a
// 32 GB card. Each stage frees its weights before the next loads.
#pragma once

#include <string>
#include <vector>

#include "vidfab/pipeline.h"

namespace vidfab {

// Where the latents come from. The stages land one at a time, so the runner
// has to be able to say precisely which one is missing rather than failing
// somewhere deep in a decoder.
enum class LatentSource {
  // Encode the prompt, load the transformer, run the denoising loop.
  kDenoise,
  // Skip conditioning and denoising and feed the decoders seeded noise. Not a
  // useful video, but it exercises unpatchify, both VAEs, the colour
  // transform and the muxer against real weights — which is most of the
  // failure surface of the output half, and none of it needs 44 GB of
  // conditioner and transformer to test.
  kSyntheticNoise,
};

struct RunOptions {
  LatentSource source = LatentSource::kDenoise;
  bool verbose = true;
};

struct RunResult {
  bool ok = false;
  std::string message;

  // Paths actually written. More than one when muxing was unavailable and the
  // run fell back to .y4m + .wav.
  std::vector<std::string> outputs;

  double seconds_conditioning = 0.0;
  double seconds_denoise = 0.0;
  double seconds_video_decode = 0.0;
  double seconds_audio_decode = 0.0;
  double seconds_output = 0.0;
};

RunResult run_generate(const GenerateRequest& request, const GeneratePlan& plan,
                       const RunOptions& options = {});

}  // namespace vidfab
