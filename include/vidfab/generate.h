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

#include "vidfab/attention_mode.h"
#include "vidfab/pipeline.h"
#include "vidfab/sampler/scheduler.h"

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

  // Counted CLI runs share prompt conditioning. Transformer residency cannot
  // cross the VAE phase: those weights together exceed practical VRAM on
  // supported cards and trigger paging instead of a speedup.
  bool reuse_models = false;
  bool release_reused_models = false;

  // Which integrator the two schedulers use. Both cost one forward pass per
  // step; the reason to change it is to be able to lower `num_inference_steps`
  // for the same quality, not to make a step cheaper. Defaults to the
  // reference's Euler, and nothing about the default path changes.
  sampler::SamplerKind sampler = sampler::SamplerKind::kEuler;

  // If set, the denoiser's own output — the packed video and audio rows,
  // fp32 — is written here as safetensors before either VAE sees it.
  //
  // This is the diff point for a change to the transformer. `decode --dump`
  // compares pixels, which works but puts 9 GB of VAE between the change and
  // the comparison and costs 400 MB a side; these are the bytes the change
  // actually moves, and 7.5 MB of them at the default geometry. Two runs of
  // the same seed and geometry must produce byte-identical files, so
  // `vidfab compare a b --abs-tol 0` is the whole test.
  std::string dump_latents_path;

  // Frame-banded attention: a video row attends to +/- this many latent frames
  // rather than the whole packed sequence. 0 is off and is the default.
  //
  // Lossy by construction and therefore not a tuning knob: it changes the
  // sample. Text and audio rows keep global attention and every video row keeps
  // the text/audio prefix, so the conditioning path is unaffected; what it drops
  // is distant video-to-video attention. Its cost scales as the band's share of
  // the sequence, so it saves more the longer the request.
  int attention_band = 0;
  // Attention implementation. Flash2 preserves the former default.
  AttentionMode attention_mode = AttentionMode::kFlash2;
  SolSchedule sol_schedule;
  // If set, the fp32 latent rows in this file replace the seeded noise draw.
  // Off by default; nothing about a normal run reads it.
  //
  // It does two jobs, and both belong to the frame-banding quality probe:
  //
  //   - with the denoiser, it starts a chunk from its slice of the *full*
  //     request's noise field instead of an independent chunk-sized draw, so
  //     the probe measures the loss of cross-chunk attention rather than three
  //     unrelated samples;
  //   - with `--synthetic-latents`, it skips conditioning and denoising
  //     altogether and sends the file's own rows through unpatchify, both VAEs
  //     and the muxer — which is how a cross-faded latent becomes an MP4 with
  //     its audio, without a second decode path existing to drift.
  //
  // The file is the same shape `--dump-latents` writes: `video_rows` [V, 96]
  // and `audio_rows` [Sa, 32], both fp32, both checked against the layout.
  std::string init_latents_path;
};

struct RunResult {
  bool ok = false;
  std::string message;

  // Paths actually written. More than one when muxing was unavailable and the
  // run fell back to .y4m + .wav.
  std::vector<std::string> outputs;

  double seconds_conditioning = 0.0;
  // Opening the checkpoint and filling the device arena. Part of
  // `seconds_denoise`, broken out because it is fixed cost and the loop is not.
  double seconds_transformer_load = 0.0;
  // `prepare_text` (the token refiner, once per request) plus
  // `prepare_sequence` (rotary tables, index uploads, workspace reserve).
  double seconds_prepare = 0.0;
  // The denoising loop alone: no load, no prepare.
  double seconds_denoise_loop = 0.0;
  // Evaluations actually run, and evaluations served from the previous
  // velocity. `steps_skipped` is zero unless step caching was turned on.
  int steps_computed = 0;
  int steps_skipped = 0;
  double seconds_denoise = 0.0;
  double seconds_video_decode = 0.0;
  double seconds_audio_decode = 0.0;
  double seconds_output = 0.0;
};

RunResult run_generate(const GenerateRequest& request, const GeneratePlan& plan,
                       const RunOptions& options = {});

}  // namespace vidfab
