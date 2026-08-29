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
#include "vidfab/video/y4m.h"
#include "vidfab/pixel_buffer.h"
#include "vidfab/pipeline.h"
#include "vidfab/sampler/scheduler.h"

namespace vidfab {

// Where a run is, for `RunOptions::on_progress`. Ordered, and a run may skip
// several: no references, no audio VAE, synthetic latents. The values are
// mirrored by VIDFAB_STAGE_* in capi.h and must not be renumbered.
enum class RunStage {
  kStarting = 0,
  kReferences = 1,
  kConditioning = 2,
  kTransformerLoad = 3,
  kDenoising = 4,
  kVideoDecode = 5,
  kAudioDecode = 6,
  kDelivering = 7,
  kFinished = 8,
};

// The decoded run, borrowed by `RunOptions::on_samples` for the duration of
// the call.
//
// The two buffers are non-const so a caller that wants to keep them can move
// out of them rather than copy: at the default geometry the video plane alone
// is 2.3 GB of float. Returning true from the hook means exactly that the
// caller took them, and `run_generate` then writes no file and never reads
// them again.
struct RunSamples {
  int channels = 3;
  int frames = 0;
  int height = 0;
  int width = 0;
  PixelBuffer* video = nullptr;  // [channels][frames][height][width], fp32 in [0,1]

  int audio_channels = 0;
  int audio_sample_rate = 0;
  std::vector<float>* audio = nullptr;  // interleaved, may be null or empty
};

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
  // Neural decoder backend. CUDA remains the default. Vulkan currently owns
  // the complete video/audio VAE vertical slice; conditioning and denoising
  // remain fail-closed until their Vulkan graphs are implemented.
  DeviceBackend inference_backend = DeviceBackend::kCuda;
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

  // --- host hooks -----------------------------------------------------------
  //
  // Plain function pointers rather than std::function, because the one caller
  // that needs them is the C ABI in capi.h and these have to survive the trip
  // through it. All three default to null, and a run that leaves them null is
  // byte-for-byte the run it was.
  //
  // Called at each stage boundary and after each denoising step, with `step`
  // -1 outside the loop. Returning false cancels: the run stops at that point
  // and returns `ok == false`, `cancelled == true`. Cancellation is only as
  // fine-grained as the checkpoints — a cancel during a multi-gigabyte
  // checkpoint read is not seen until that read finishes.
  bool (*on_progress)(RunStage stage, int step, int steps, void* userdata) = nullptr;

  // Called once, after both VAEs and before anything is written. Returning
  // true means the caller has taken the samples and no output file is
  // produced — which is how a host gets frames and PCM without the muxer, and
  // therefore without FFmpeg being loaded at all. Returning false leaves the
  // normal output stage to run.
  bool (*on_samples)(RunSamples& samples, void* userdata) = nullptr;

  void* hook_userdata = nullptr;
  // Output-only acceleration hook, independent of `inference_backend`.
  video::FrameConverter* output_frame_converter = nullptr;
};

inline bool generation_backend_supported(DeviceBackend backend,
                                         LatentSource source) noexcept {
  switch (backend) {
    case DeviceBackend::kCuda:
      return true;
    case DeviceBackend::kVulkan:
      return source == LatentSource::kSyntheticNoise;
  }
  return false;
}

struct RunResult {
  bool ok = false;
  // Set when `RunOptions::on_progress` returned false. Distinct from a plain
  // failure: nothing went wrong and `message` says only where it stopped.
  bool cancelled = false;
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
