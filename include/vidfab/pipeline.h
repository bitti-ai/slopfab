// End-to-end text-to-video-and-audio generation.
//
// One request produces one packed sequence, one denoising loop and one output
// file. The stages run strictly in order, and the order is forced by memory
// rather than by taste: the int8 Qwen3-VL conditioner is ~24 GB resident and
// the fp8 transformer is ~19 GB, so on a 32 GB card they cannot both be loaded.
// The pipeline encodes the prompt, frees the conditioner, loads the
// transformer, denoises, frees it, then decodes with the two VAEs.
//
// The nvfp4 pair is the one combination that *would* fit — 12.8 GB and 11.7 GB
// against ~30 GB free — but the sequencing stays, because the other three
// combinations still do not and holding both buys a single 0.12 s encode. The
// ordering is therefore not conditional on the checkpoint formats, and nothing
// below inspects them to decide it.
//
// `resolve_plan` does every piece of geometry and schedule arithmetic up front
// and without touching a weight file, so a request can be validated — and its
// memory footprint reported — before 20 GB of I/O happens.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vidfab/dit/packing.h"

namespace vidfab {

struct GenerateRequest {
  std::string prompt;
  // Empty means the CLI will choose output/video-<generation timestamp>.mp4.
  // Library callers should set this explicitly before running the pipeline.
  std::string out_path;

  // Only the ratio matters; the short edge is fixed at 768 and the area capped
  // at 768*1344, per the released model. Ignored when the canvas is named
  // outright below.
  int aspect_w = 16;
  int aspect_h = 9;

  // An explicit canvas, which takes precedence over the aspect ratio when both
  // axes are set. Zero means "derive it from the aspect", so the default is
  // whatever 16:9 resolves to and adding this field changed no existing run.
  // Unlike the aspect path this is not scaled down to the trained area — a
  // caller naming a canvas gets that canvas, and is warned if it is larger than
  // the model was trained for.
  int canvas_width = 0;
  int canvas_height = 0;

  bool has_explicit_canvas() const { return canvas_width > 0 && canvas_height > 0; }

  // Snapped up to the next 17*k + 5 the video VAE can encode.
  int num_frames = 124;

  // Sigma grid points *including* the terminal zero, so the model runs
  // `num_inference_steps - 1` times.
  int num_inference_steps = 50;

  uint64_t seed = 0;

  std::string transformer_path;
  std::string text_encoder_path;
  std::string tokenizer_path;
  std::string video_vae_path;
  std::string audio_vae_path;

  // Ordered subject/style/scene references. Presence selects the Ref2VA task;
  // the same order labels images in the multimodal prompt and packed sequence.
  std::vector<std::string> reference_image_paths;

  // Write .y4m + .wav instead of muxing an MP4. Also the automatic fallback
  // when ffmpeg cannot be loaded.
  bool raw_output = false;

  // Step caching (see dit/step_cache.h). Off by default: `threshold == 0` and
  // `skip_every == 0` mean every step is evaluated, which is the shipped
  // behaviour and must stay bit-identical to a build without any of this.
  float cache_threshold = 0.0f;
  int cache_warmup = 3;
  int skip_every = 0;
};

// Everything derivable from a request without reading a checkpoint. `num_text`
// in the layout is filled in only after tokenisation, so it is zero here and
// `sequence_length_without_text` is what can be known in advance.
struct GeneratePlan {
  int canvas_height = 0;
  int canvas_width = 0;
  int aligned_frames = 0;
  double duration_seconds = 0.0;

  dit::SequenceLayout layout;  // layout.num_text is 0 until the prompt is tokenised

  // Two independent schedules stepped inside one loop: shift 12.0 for video,
  // 3.0 for audio. Both are asserted to be the same length — the reference
  // zips them, and a length mismatch after `unique_consecutive` would silently
  // truncate the run (spec section 9.5).
  std::vector<float> video_sigmas;
  std::vector<float> audio_sigmas;
  std::vector<float> video_timesteps;
  std::vector<float> audio_timesteps;

  int num_model_evaluations() const { return static_cast<int>(video_timesteps.size()); }
  int sequence_length_without_text() const { return layout.total_rows(); }
};

// Throws std::runtime_error with a specific message for an unsupported aspect
// ratio, a non-positive frame count, or a schedule shorter than one step.
GeneratePlan resolve_plan(const GenerateRequest& request);

// Human-readable summary of a resolved plan, for `--dry-run` and for the
// header a real run prints before it starts.
std::string describe_plan(const GenerateRequest& request, const GeneratePlan& plan);

}  // namespace vidfab
