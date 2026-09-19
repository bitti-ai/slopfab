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
// without loading weights, so a request can be validated — and its
// memory footprint reported — before 20 GB of I/O happens. An available
// transformer header is inspected to select the model's sigma shift.
#pragma once

#include "slopfab/dit/motion_cache.h"

#include <cstdint>
#include <string>
#include <vector>

#include "slopfab/dit/packing.h"
#include "slopfab/reference_media.h"
#include "slopfab/refmod.h"
#include "slopfab/continuation.h"
#include "slopfab/lora.h"
#include "slopfab/sampler/scheduler.h"

namespace slopfab {

// ref/FL2VA/model_index.json -> _minimax_h3.sigma_shift_scales, consumed by
// convert.py:630-653.
//
// Named here rather than in pipeline.cpp because the denoise loop has to
// integrate on the same grid `resolve_plan` built and `describe_plan` printed.
// These were previously two private constants and two literals in the runner,
// which is one value written in four places: changing the shift in one of them
// would have left the loop stepping a schedule nothing else agreed with, and
// nothing would have said so.
constexpr float kVideoSigmaShift = 12.0f;
constexpr float kAudioSigmaShift = 3.0f;
// Viggle-Animate's distilled Euler recipe uses the lower video flow shift.
constexpr float kViggleVideoSigmaShift = 3.0f;

struct GenerateRequest {
  std::string prompt;
  // The CLI replaces this default with output/video-<timestamp>.mp4; the
  // concrete fallback remains useful to callers of the library API.
  std::string out_path = "video.mp4";

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

  // Generate one still image instead of a temporal video. This is deliberately
  // opt-in: it denoises one video latent frame, omits target audio rows, and
  // repeats that latent across seven VAE temporal positions and retains phase
  // 3 of the first position. It is a different sampling mode, not a faster
  // way to reproduce frame zero of a normal video request.
  bool still_image = false;

  // Sigma grid points *including* the terminal zero, so the model runs
  // `num_inference_steps - 1` times.
  int num_inference_steps = 50;
  sampler::ScheduleKind schedule = sampler::ScheduleKind::kDefault;
  std::vector<LoraSpec> loras;

  // Viggle's frozen-conditioning recipe: driving video then repainted frame,
  // target-sized references, and no reference soundtrack. The fixed embedding
  // is supplied through RunOptions::prompt_embedding_path.
  bool animate = false;
  bool preserve_driving_audio = false;

  uint64_t seed = 0;

  std::string transformer_path;
  std::string text_encoder_path;
  std::string tokenizer_path;
  std::string video_vae_path;
  std::string audio_vae_path;

  // Ordered subject/style/scene references. Presence selects the Ref2VA task;
  // the same order labels images in the multimodal prompt and packed sequence.
  std::vector<std::string> reference_image_paths;

  // Decoded video/audio references, in insertion order, following the legacy
  // image list. Attached payloads are immutable and shared by request copies.
  // CUDA and Vulkan support video/audio conditioning.
  std::vector<std::shared_ptr<const ReferenceMedia>> reference_media;

  // Pre-encoded references follow native media; they do not add Qwen tokens.
  std::vector<RefModReference> refmods;

  // Owning immutable snapshot. With continuation, num_frames means NEW frames
  // (rounded up to a multiple of 17); the output includes the source clip.
  std::shared_ptr<const LatentClip> continuation;
  int continuation_overlap_frames = 22;

  bool has_native_references() const {
    return !reference_image_paths.empty() || !reference_media.empty();
  }
  bool has_refmods() const {
    for (const auto& ref : refmods) if (ref.enabled()) return true;
    return false;
  }
  bool has_references() const { return has_native_references() || has_refmods(); }

  // Write .y4m + .wav instead of muxing an MP4. Also the automatic fallback
  // when ffmpeg cannot be loaded.
  bool raw_output = false;

  // Step caching (see dit/step_cache.h). Off by default: `threshold == 0` and
  // `skip_every == 0` mean every step is evaluated, which is the shipped
  // behaviour and must stay bit-identical to a build without any of this.
  float cache_threshold = 0.0f;
  dit::MotionCacheConfig motion_cache;
  int cache_warmup = 3;
  int skip_every = 0;

  // Block-span residual caching (see dit/block_cache.h). Independent of the
  // step cache above and off by default for the same reason: `span == 0` must
  // leave the forward pass bit-identical to a build without any of this.
  int block_cache_span = 0;
  int block_cache_start = -1;  // negative centres the span in the stack
  int block_cache_interval = 2;
  int block_cache_warmup = 3;
};

// Everything derivable from a request and an available checkpoint header. `num_text`
// in the layout is filled in only after tokenisation, so it is zero here and
// `sequence_length_without_text` is what can be known in advance.
struct GeneratePlan {
  bool fasth3_v2 = false;
  int canvas_height = 0;
  int canvas_width = 0;
  int aligned_frames = 0;
  double duration_seconds = 0.0;

  // aligned_frames/duration describe delivered output; layout describes only
  // the bounded sampling window. Equal for ordinary generations.
  int sampling_frames = 0;
  ContinuationPlan continuation;

  dit::SequenceLayout layout;  // layout.num_text is 0 until the prompt is tokenised

  // Two independent schedules stepped inside one loop: video shift 12.0 for
  // H3 or 3.0 for Viggle-Animate, audio shift 3.0. Both have the same length — the reference
  // zips them, and a length mismatch after `unique_consecutive` would silently
  // truncate the run (spec section 9.5).
  std::vector<float> video_sigmas;
  std::vector<float> audio_sigmas;
  std::vector<float> video_timesteps;
  std::vector<float> audio_timesteps;

  // The shifts the four lists above were built with, carried so the runner can
  // reconstruct the identical schedulers instead of naming the numbers again.
  float video_sigma_shift = kVideoSigmaShift;
  float audio_sigma_shift = kAudioSigmaShift;
  // Grid points requested, so a scheduler rebuilt from this plan gets the same
  // `set_timesteps` argument without consulting the request.
  int num_inference_steps = 0;

  int num_model_evaluations() const { return static_cast<int>(video_timesteps.size()); }
  int sequence_length_without_text() const { return layout.total_rows(); }
};

// Throws std::runtime_error with a specific message for an unsupported aspect
// ratio, a non-positive frame count, or a schedule shorter than one step.
GeneratePlan resolve_plan(const GenerateRequest& request);

// --- reuse keys -------------------------------------------------------------
//
// `--count` and `--reuse-models` keep expensive per-request work alive across
// generations in one process. What makes that safe is the key: two requests
// share a cache entry only if every input the cached value was computed from is
// identical. A path is *not* such an input — overwriting `ref.png` in place
// between two generations leaves the path equal and the content different, and
// keying on the path alone silently reuses the previous image.
//
// The two kinds of file a request names get two different identities, because
// the cost of being sure is not the same for both.
//
// Checkpoints are multi-gigabyte and get (size, last-write time). Hashing them
// was measured at ~25 s across the 34 GB a run touches, which is far more than
// any of the work being cached; stat'ing them is ~0.025 ms each. The blind spot
// is real — a same-size replacement inside one filesystem timestamp tick is
// invisible — but nobody edits a 12 GB checkpoint in place without moving one
// of the two, and the alternative is not affordable.
//
// Reference images get their contents hashed, because for them that reasoning
// inverts. They are small, and mtime is *routinely* preserved by exactly the
// operations that replace them: `copy`, `robocopy /COPY:T`, `xcopy /K`, most
// image tooling writing through a temp file, rsync with `--times`. exFAT keeps
// timestamps to 2 s and SMB inherits the server's clock. A same-size
// mtime-preserving overwrite of `ref.png` is an ordinary event, not a contrived
// one, and it is the precise case this whole key exists to catch. Hashing 4 MB
// was measured at 5.57 ms warm — under 1% of the decode, Lanczos resize and
// keyframe encode it guards.

// `path`, then its size and last-write time, appended to `key` with NUL
// separators. A file that cannot be stat'ed contributes a distinct marker
// rather than being silently treated as unchanged. For checkpoints.
void append_file_identity(std::string& key, const std::string& path);

// `path`, then its byte count and a 64-bit FNV-1a hash of its contents. Detects
// any change to the bytes regardless of what the filesystem metadata says. For
// reference images, which are small enough to afford it.
void append_file_content_identity(std::string& key, const std::string& path);

// The content identity of each reference image, in request order.
//
// Hashing is the expensive part of both keys below and the only part they
// share, so a caller that wants both should compute this once and hand it to
// each. Beyond saving the second pass, it makes the two keys one consistent
// snapshot: read separately, they could straddle a write and disagree about
// which image the run used.
std::vector<std::string> reference_image_identities(const GenerateRequest& request);

// The conditioner arithmetic is part of the cached value's identity. Exact
// CUDA and Vulkan intentionally have the same bytes today, but keeping their
// authorities distinct prevents a backend run from being silently satisfied
// by another backend (and makes that invariant survive future rebaselines).
enum class ConditionerAuthority : uint8_t {
  kCudaShipped = 0,
  kCudaExact = 1,
  kVulkanExact = 2,
};

// Key for a request's prompt conditioning: the encoder and tokenizer files by
// stat identity, the prompt text, and every reference image by content.
// Decoded media contributes content digests, dimensions, timing and PCM format.
//
// The single-argument form hashes the references itself. Prefer the other one
// wherever both keys are needed; passing a list that does not match
// `request.reference_image_paths` in length falls back to hashing rather than
// keying off a stale snapshot.
std::string conditioning_cache_key(const GenerateRequest& request);
std::string conditioning_cache_key(const GenerateRequest& request,
                                   const std::vector<std::string>& reference_identities);
std::string conditioning_cache_key_for_authority(
    const GenerateRequest& request, ConditionerAuthority authority);
std::string conditioning_cache_key_for_authority(
    const GenerateRequest& request,
    const std::vector<std::string>& reference_identities,
    ConditionerAuthority authority);

// Key for the seed-independent reference-image work: decode, Lanczos resize and
// the VAE keyframe encode. Deliberately narrower than the conditioning key,
// because none of that work reads the prompt or the text encoder — but wider in
// one place, because all of it reads the video VAE.
// With decoded media, the audio VAE identity also participates.
std::string reference_cache_key(const GenerateRequest& request);
enum class ReferenceEncoderAuthority : uint8_t { kCudaFp32, kCudaFp16, kVulkanFp32 };
std::string media_encoding_cache_key(const GenerateRequest& request,
                                     ReferenceEncoderAuthority authority);
std::string reference_cache_key(const GenerateRequest& request,
                                const std::vector<std::string>& reference_identities);

// Key for a loaded tokenizer. Empty `tokenizer_path` means the embedded copy,
// which is part of the binary and so cannot go stale.
std::string tokenizer_cache_key(const GenerateRequest& request);

// Human-readable summary of a resolved plan, for `--dry-run` and for the
// header a real run prints before it starts.
std::string describe_plan(const GenerateRequest& request, const GeneratePlan& plan);

}  // namespace slopfab
