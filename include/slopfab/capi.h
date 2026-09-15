/* Stable C ABI for slopfab.
 *
 * The C++ headers beside this one are the library's real API, and they are not
 * callable from anything but C++ built by the same compiler with the same CRT:
 * `std::string` and `std::vector` cross nearly every signature and exceptions
 * propagate out of them. This header exists so that Rust, C#, Python, Go and
 * plain C can drive the pipeline over an interface that survives a toolchain
 * mismatch.
 *
 * **This API produces pixels, not files.** A generation hands back the decoded
 * frames as planar float RGB and the audio as interleaved float PCM, and
 * writes no media files. Optional latent archives can be saved for continuation.
 * Encoding, muxing and playback belong to the host,
 * which is why the DLL needs no FFmpeg: the one part of this project that
 * loads it is the muxer, and nothing here calls it. A host that wants an MP4
 * feeds these buffers to its own encoder; `slopfab.exe` is the reference
 * consumer that does exactly that.
 *
 * The shape of the interface, and what it costs:
 *
 *   - **Opaque handles.** A caller never sees the layout of a request or a
 *     generation, so adding a field to either is not an ABI break. The three
 *     plain structs that do cross the boundary — `slopfab_plan`,
 *     `slopfab_progress` and `slopfab_output` — have their layout fixed here,
 *     and changing any of them bumps the minor version.
 *
 *   - **No exceptions escape.** Every function is noexcept in effect: what the
 *     C++ side throws becomes a status code plus a message on
 *     `slopfab_last_error()`. Unwinding through a C frame is undefined
 *     behaviour, and on MSVC it kills the host process without a diagnostic.
 *
 *   - **Status codes are plain `int`, not an enum.** Deliberate: a caller
 *     linked against an older header can receive a status this header does not
 *     name and hold it faithfully, where an enum-typed return would be an
 *     out-of-range value with undefined behaviour behind it.
 *
 *   - **One allocation contract.** The only pointer the caller ever frees is a
 *     `char*` from `slopfab_describe_plan`, and it is freed by
 *     `slopfab_free_string` — in this DLL, on the heap that allocated it.
 *     Calling the host's own `free` on it would cross CRTs, which on Windows
 *     is a crash often enough to matter. Everything else is owned by a handle
 *     and documented against it.
 *
 * Generation is **asynchronous**: `slopfab_generation_start` returns as soon as
 * the request is known to be satisfiable and the work proceeds on a worker
 * thread, so a UI stays responsive across a run that takes minutes. Progress
 * arrives on that worker thread, not the caller's.
 */
#ifndef SLOPFAB_CAPI_H
#define SLOPFAB_CAPI_H

#include <stddef.h>
#include <stdint.h>

/* Linkage. Consumers of the DLL need nothing: the default is the import side.
 * Define SLOPFAB_C_STATIC when linking these entry points into a binary rather
 * than importing them, and SLOPFAB_C_BUILD only when building the library. */
#if defined(_WIN32)
#  if defined(SLOPFAB_C_BUILD)
#    define SLOPFAB_C_API __declspec(dllexport)
#  elif defined(SLOPFAB_C_STATIC)
#    define SLOPFAB_C_API
#  else
#    define SLOPFAB_C_API __declspec(dllimport)
#  endif
/* Named rather than left to the compiler default, because a consumer built
 * with /Gz would otherwise disagree with this DLL about the calling convention
 * on 32-bit x86 and corrupt the stack. On x64 there is one convention and this
 * expands to nothing that matters. */
#  define SLOPFAB_CALL __cdecl
#else
#  if defined(SLOPFAB_C_BUILD)
#    define SLOPFAB_C_API __attribute__((visibility("default")))
#  else
#    define SLOPFAB_C_API
#  endif
#  define SLOPFAB_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* --- version ---------------------------------------------------------------
 *
 * MAJOR changes when something already here changes meaning or disappears;
 * MINOR when a function or a struct field is added. Adding a function is not a
 * break: an unresolved import fails at load time, which is loud, rather than
 * misreading memory, which is not.
 *
 * A binding should compare `slopfab_capi_version()` against the value it was
 * compiled with and refuse a different MAJOR. */
#define SLOPFAB_CAPI_VERSION_MAJOR 1
#define SLOPFAB_CAPI_VERSION_MINOR 9
#define SLOPFAB_CAPI_VERSION_PATCH 0

/* Packed as (major << 24) | (minor << 12) | patch.
 *
 * The 12-bit minor field is not generosity: minor bumps whenever functions
 * are added, an 8-bit field would have run out at 255 revisions, and widening
 * the packing after the fact is itself the ABI break this number exists to
 * report. */
SLOPFAB_C_API uint32_t SLOPFAB_CALL slopfab_capi_version(void);

/* The same three numbers as a dotted string, built from the macros above so
 * the two cannot disagree. Owned by the library and valid forever. */
SLOPFAB_C_API const char* SLOPFAB_CALL slopfab_capi_version_string(void);

/* --- status ----------------------------------------------------------------
 *
 * Returned as `int` by every fallible function. The distinctions that carry
 * weight are INVALID_REQUEST versus RUNTIME — the first means the request
 * could never work and no file was opened to learn that, the second means it
 * was plausible and something failed doing it — and OUT_OF_MEMORY, which is
 * separated because the answer to it is a smaller geometry rather than a bug
 * report, and because it is the ordinary failure on a card that is also
 * driving a desktop. */
#define SLOPFAB_OK 0
/* A null pointer, an out-of-range index, or an id this build does not know. */
#define SLOPFAB_ERR_INVALID_ARGUMENT (-1)
/* The request cannot be resolved: an aspect outside 1:4..4:1, a canvas axis
 * that is not a multiple of 32, too few frames, a schedule shorter than one
 * step, or a tenth reference image. Nothing was read to determine it. */
#define SLOPFAB_ERR_INVALID_REQUEST (-2)
/* A path the caller named does not exist or cannot be read. */
#define SLOPFAB_ERR_NOT_FOUND (-3)
/* Host or device allocation failed. */
#define SLOPFAB_ERR_OUT_OF_MEMORY (-4)
/* The run started and failed: a malformed checkpoint, a driver error, a
 * decoder fault. */
#define SLOPFAB_ERR_RUNTIME (-5)
/* An exception with no better classification reached the boundary. */
#define SLOPFAB_ERR_UNKNOWN (-6)
/* The generation is still running, so what was asked for does not exist yet.
 * Also the status a generation holds until it finishes. */
#define SLOPFAB_ERR_NOT_READY (-7)
/* Stopped by `slopfab_generation_cancel`. Not a failure: nothing went wrong and
 * the message says only where it stopped. */
#define SLOPFAB_ERR_CANCELLED (-8)
/* A generation is already running in this process. See
 * `slopfab_generation_start`. */
#define SLOPFAB_ERR_BUSY (-9)
/* The requested feature has an input contract but no execution backend yet. */
#define SLOPFAB_ERR_UNSUPPORTED (-10)

/* The message belonging to the most recent failure **on the calling thread**,
 * or "" when the last call on this thread succeeded. Never null.
 *
 * Thread-local, so two threads cannot overwrite each other's diagnosis — and
 * therefore *not* where a run's failure lands, because the run fails on a
 * worker thread the caller never enters. Use `slopfab_generation_error` for
 * that. The pointer is valid until the next slopfab call on this thread. */
SLOPFAB_C_API const char* SLOPFAB_CALL slopfab_last_error(void);

/* Frees a string this library allocated — only ever the `char*` out-parameter
 * of `slopfab_describe_plan`. Null is a no-op. */
SLOPFAB_C_API void SLOPFAB_CALL slopfab_free_string(char* text);

/* --- CUDA runtime selection ------------------------------------------------
 *
 * Windows builds contain one CUDA 12.8 static-runtime fat binary and resolve
 * cuBLAS from an installed CUDA 13 or CUDA 12 toolkit on first use. The
 * default is "auto" (13, then 12); SLOPFAB_CUDA_VERSION provides the same
 * process-wide setting for hosts that prefer environment configuration.
 *
 * This setter is the DLL equivalent of slopfab.exe's --cuda-version option.
 * Call it before the first CUDA/cuBLAS operation. `version` is "auto", "13"
 * or "12". Selection is immutable after initialization, including if another
 * thread initialized it first.
 */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_cuda_set_version(const char* version);

/* Initializes cuBLAS if necessary and writes the selected toolkit major (12
 * or 13). This is also a cheap host-side installation check before starting a
 * generation. `out_major` must not be null. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_cuda_loaded_major(int32_t* out_major);

/* --- ids -------------------------------------------------------------------
 *
 * Plain integers rather than enums, for the reason given for the status codes:
 * they cross the ABI as values, and a caller must be able to hold one this
 * header does not name. */

/* Which of a request's five checkpoints a path names. An empty tokenizer path
 * selects the copy embedded in the library module. */
#define SLOPFAB_MODEL_TRANSFORMER 0
#define SLOPFAB_MODEL_TEXT_ENCODER 1
#define SLOPFAB_MODEL_TOKENIZER 2
#define SLOPFAB_MODEL_VIDEO_VAE 3
#define SLOPFAB_MODEL_AUDIO_VAE 4

/* Neural backend. Vulkan denoising accepts exact, flash2 and sage2, supports native
 * text-only conditioning, and never calls the CUDA conditioner. */
#define SLOPFAB_INFERENCE_CUDA 0
#define SLOPFAB_INFERENCE_VULKAN 1

/* Where a run is, in `slopfab_progress::stage`. Ordered, and a run may skip
 * several of them: no references, no audio VAE, synthetic latents. These
 * mirror slopfab::RunStage and must not be renumbered. */
#define SLOPFAB_STAGE_STARTING 0
#define SLOPFAB_STAGE_REFERENCES 1
#define SLOPFAB_STAGE_CONDITIONING 2
#define SLOPFAB_STAGE_TRANSFORMER_LOAD 3
#define SLOPFAB_STAGE_DENOISING 4
#define SLOPFAB_STAGE_VIDEO_DECODE 5
#define SLOPFAB_STAGE_AUDIO_DECODE 6
#define SLOPFAB_STAGE_DELIVERING 7
#define SLOPFAB_STAGE_FINISHED 8

/* --- handles --------------------------------------------------------------- */

typedef struct slopfab_request slopfab_request;
typedef struct slopfab_generation slopfab_generation;
typedef struct slopfab_reference_video slopfab_reference_video;

/* Decoded video/audio reference ingestion (no FFmpeg).
 *
 * Generation supports CUDA and Vulkan with a Ref2VA transformer. Video uses
 * the video VAE encoder; attached/standalone audio also needs an audio VAE
 * with floating-point encoder weights. Vulkan video references also require
 * floating-point encoder weights. Host frames and PCM never require FFmpeg.
 *
 * Video duration is 2..15 seconds. Frame times are relative to the clip start:
 * first time zero, subsequent times strictly increasing and below duration.
 * Dimensions stay constant. Row stride is positive, in bytes; only visible
 * pixels are read and RGBA alpha is ignored. Every append copies its input.
 *
 * Soundtracks are interleaved float PCM in [-1,1], mono or stereo, at a positive
 * native sample rate. float_count counts floats, not sample frames or bytes.
 * set_audio replaces the soundtrack and copies before returning. The start
 * offset is clip-relative; the soundtrack must end within the video duration.
 *
 * Attaching retains an immutable snapshot: the video handle may then be edited
 * or destroyed and input buffers reused. A failed setter leaves the handle
 * unchanged. Serialize access to each mutable video or request handle.
 *
 * Images precede video/audio inputs; video/audio inputs retain insertion order.
 * At most 9 images, 3 videos, 3 standalone audios, and 12 total references.
 * Videos and standalone audios each have a 15 second aggregate duration limit.
 * A video's soundtrack does not consume a standalone audio-reference slot.
 * Standalone audio references require at least one image or video in the
 * completed request; this is checked by slopfab_resolve_plan.
 */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_reference_video_create(
    double duration_seconds, slopfab_reference_video** out_video);
SLOPFAB_C_API void SLOPFAB_CALL slopfab_reference_video_destroy(slopfab_reference_video* video);
SLOPFAB_C_API int SLOPFAB_CALL slopfab_reference_video_append_rgb24(
    slopfab_reference_video* video, const uint8_t* pixels, size_t buffer_bytes,
    int32_t width, int32_t height, size_t row_stride_bytes, double timestamp_seconds);
SLOPFAB_C_API int SLOPFAB_CALL slopfab_reference_video_append_rgba8(
    slopfab_reference_video* video, const uint8_t* pixels, size_t buffer_bytes,
    int32_t width, int32_t height, size_t row_stride_bytes, double timestamp_seconds);
SLOPFAB_C_API int SLOPFAB_CALL slopfab_reference_video_set_audio_f32(
    slopfab_reference_video* video, const float* samples, size_t float_count,
    int32_t channels, int32_t sample_rate, double start_seconds);
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_add_reference_video(
    slopfab_request* request, const slopfab_reference_video* video);
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_add_reference_audio_f32(
    slopfab_request* request, const float* samples, size_t float_count,
    int32_t channels, int32_t sample_rate);

/* --- plan ------------------------------------------------------------------
 *
 * Everything a request implies that can be known without opening a weight
 * file. Filled by `slopfab_resolve_plan`, which is instant — so a host can
 * validate a request, show its geometry and estimate its cost before 20 GB of
 * I/O happens. */
typedef struct slopfab_plan {
  int32_t canvas_width;
  int32_t canvas_height;
  /* The frame count snapped up to the next 17*k + 5 the video VAE can encode,
   * which is what will actually be produced; exactly 1 in still-image mode. */
  int32_t aligned_frames;
  double duration_seconds;
  /* Forward passes the loop will run: one fewer than the grid points, because
   * the terminal sigma gets none. */
  int32_t num_model_evaluations;
  /* Packed rows excluding text, which is not known until the prompt is
   * tokenised. Attention cost grows with the square of this. */
  int32_t sequence_rows_without_text;
  int32_t latent_frames;
  int32_t latent_height;
  int32_t latent_width;
  int32_t num_video_rows;
  int32_t num_audio_rows;
  int32_t num_audio_latents;
} slopfab_plan;

/* --- progress --------------------------------------------------------------
 *
 * Delivered on the worker thread, not the caller's: a callback that touches
 * host UI state must marshal to its own thread. `step` is -1 outside the
 * denoising loop and on entry before its first step completes; `total_steps`
 * is already set on denoising entry and is 0 where it means nothing. */
typedef struct slopfab_progress {
  int32_t stage; /* one of SLOPFAB_STAGE_* */
  int32_t step;
  int32_t total_steps;
  double elapsed_seconds;
} slopfab_progress;

typedef void(SLOPFAB_CALL* slopfab_progress_fn)(const slopfab_progress* progress, void* userdata);

/* --- output ----------------------------------------------------------------
 *
 * The decoded run, borrowed from the generation. **Every pointer here is owned
 * by the generation handle and dies with `slopfab_generation_destroy`** — at
 * the default geometry the video plane alone is over 2 GB, so it is handed
 * over by pointer rather than copied, and a host that wants to keep it past
 * the handle must copy it out.
 *
 * Video is planar float RGB in [0,1], shaped [channels][frames][height][width]
 * — channel-major, so the red plane for every frame precedes the green. That
 * is the decoder's own layout, kept rather than interleaved because converting
 * gigabytes to please a consumer that may want something else again is not the
 * library's call. `slopfab_generation_frame_rgba8` does the common conversion
 * for one frame at a time.
 *
 * Audio is interleaved float in [-1,1], `audio_channels` per frame. */
typedef struct slopfab_output {
  const float* video;
  size_t video_float_count;
  int32_t channels;
  int32_t frames;
  int32_t width;
  int32_t height;

  const float* audio; /* null when the run had no audio VAE */
  size_t audio_float_count;
  int32_t audio_channels;
  int32_t audio_sample_rate;
  int64_t audio_frames; /* per channel */

  /* Frames per second the request resolved to, for a host assembling a
   * container or scheduling playback. */
  double fps;

  double seconds_conditioning;
  double seconds_denoise;
  double seconds_video_decode;
  double seconds_audio_decode;
  double seconds_total;
  /* Evaluations actually run, and evaluations served from the previous
   * velocity. `steps_skipped` is zero unless a cache was turned on. */
  int32_t steps_computed;
  int32_t steps_skipped;
} slopfab_output;

/* --- request ---------------------------------------------------------------
 *
 * Created with the library defaults — 16:9, 124 frames, 50 steps, seed 0,
 * Euler, Flash2 — except that no output path is set and raw output is on,
 * because this API delivers pixels and writes nothing.
 *
 * Returns null only if allocation failed, with the reason on
 * `slopfab_last_error()`. Every setter copies what it is given, so the caller's
 * strings need not outlive the call, and every setter rejects a null handle
 * with SLOPFAB_ERR_INVALID_ARGUMENT rather than faulting. */
SLOPFAB_C_API slopfab_request* SLOPFAB_CALL slopfab_request_create(void);
SLOPFAB_C_API void SLOPFAB_CALL slopfab_request_destroy(slopfab_request* request);

/* UTF-8, and on Windows genuinely UTF-8 rather than the active code page. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_prompt(slopfab_request* request, const char* utf8);

/* Reads the prompt out of a UTF-8 file, applying the same normalisation
 * `slopfab generate --prompt-file` does: a BOM, CR characters and surrounding
 * blank space are stripped. Long Context-IR prompts do not belong in a string
 * literal. SLOPFAB_ERR_NOT_FOUND if the file cannot be read. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_prompt_file(slopfab_request* request,
                                                            const char* path);

/* Only the ratio matters: the short edge is fixed at 768 and the area capped
 * at the trained 768*1344. Setting an aspect clears any explicit resolution. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_aspect(slopfab_request* request, int32_t width,
                                                       int32_t height);

/* A canvas named outright, which wins over the aspect. Both axes must be a
 * multiple of 32 and the ratio must stay inside 1:4..4:1, checked when the
 * plan is resolved. Unlike the aspect path the area is *not* capped: a caller
 * naming 1920x1088 gets it, and pays attention cost with the square of the
 * area. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_resolution(slopfab_request* request, int32_t width,
                                                           int32_t height);

/* Snapped up to the next 17*k + 5 the video VAE can encode; `slopfab_plan`
 * reports what it became. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_frames(slopfab_request* request, int32_t frames);

/* Selects the dedicated still-image path. When enabled, `frames` is ignored:
 * the plan contains one video latent frame, no target audio rows, and one
 * decoded output frame. The VAE repeats that latent across seven temporal
 * positions and retains phase 3 of the first position. This is a distinct
 * sampling mode and is not bit-equivalent to frame zero of a video request.
 * Disabling it restores the request's previous frame count. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_still_image(slopfab_request* request,
                                                            int32_t enable);

/* Sigma grid points *including* the terminal zero, so the model runs
 * `steps - 1` times. At least 2. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_steps(slopfab_request* request, int32_t steps);

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_seed(slopfab_request* request, uint64_t seed);

/* `which` is one of SLOPFAB_MODEL_*. There is no discovery here: unlike the
 * CLI, which looks beside its own executable, a host names every checkpoint it
 * wants used. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_model_path(slopfab_request* request, int32_t which,
                                                           const char* path);

/* Optional safetensors containing F32 `prompt_embedding` [L,5120]. Accepted by
 * either backend to compare captured conditioning without recomputation. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_prompt_embedding_path(
    slopfab_request* request, const char* path);

/* H3 attention/MLP LoRA adapters, combined by summing their updates. Paths are copied.
 * A finite strength may be zero (disabled) or negative. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_add_lora(
    slopfab_request* request, const char* path, float strength);
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_clear_loras(slopfab_request* request);

/* Standalone ComfyUI H3 refmod safetensors (image/video/audio). Loads and owns
 * the latents immediately; later file changes do not affect queued requests.
 * strength: 0..1 (0 disables), copies: 1..10. Requires a Ref2VA transformer.
 * Refmods follow native references and do not alter the text prompt. Since 1.8. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_add_refmod(
    slopfab_request* request, const char* path, float strength, int32_t copies);
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_clear_refmods(slopfab_request* request);
#define SLOPFAB_SCHEDULE_DEFAULT 0
#define SLOPFAB_SCHEDULE_TAOMATE_3STEP 1
/* TaoMate uses three evaluations and overrides the ordinary step count.
 * Requires an enabled TaoMate adapter, Euler and no step/block caches. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_schedule(
    slopfab_request* request, int32_t schedule);

/* Ordered subject/style/scene references; presence selects the Ref2VA task and
 * this order labels the images in the packed sequence. At most nine, and the
 * tenth is refused here rather than at run time. Requires a Ref2VA transformer
 * checkpoint — FL2VA weights are not compatible.
 *
 * Decoding is by path because that is what the pipeline consumes. Which
 * formats are readable depends on the build: PPM always, and everything the
 * platform image APIs or a runtime-loaded FFmpeg can open otherwise. A host
 * that already holds pixels should write a PPM, which is a 15-byte header and
 * the bytes. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_add_reference_image(slopfab_request* request,
                                                                const char* path);

/* "none", "flash2" (the default), "sage2", "sol", "sol-experimental" or
 * "exact". Exact selects the pinned deterministic cooperative H3 arithmetic;
 * it never remaps to another implementation. Named by string rather than by id
 * so that an attention implementation can be added without a new constant in
 * this header. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_attention(slopfab_request* request,
                                                           const char* mode);

/* Selects the neural inference backend. Default is SLOPFAB_INFERENCE_CUDA. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_inference_backend(
    slopfab_request* request, int32_t backend);

/* Skip conditioning and denoising and feed the decoders seeded noise. Not a
 * useful video, but it exercises both VAEs and the colour transform against
 * real weights without loading 44 GB of conditioner and transformer — which
 * makes it the cheap way for a host to prove its own plumbing. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_synthetic_latents(slopfab_request* request,
                                                                  int32_t enable);

/* Progress prose on the process's stdout. On by default, because the library
 * default is on; a GUI host wants it off and the progress callback instead. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_verbose(slopfab_request* request, int32_t enable);

/* Retains reusable tokenizer, conditioning and reference preparation between
 * serial generations. The host controls the lifetime with
 * `slopfab_reused_models_clear`; overlapping generations remain forbidden. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_reuse_models(slopfab_request* request,
                                                             int32_t enable);

/* --- saved latents and continuation ----------------------------------------
 * All functions are optional. Existing generations keep their old memory and
 * file behavior unless saving or retention is requested. Archives contain
 * normalized FP32 video AND audio, geometry, and the cumulative timeline.
 */

/* Save the completed (joined, when continuing) latents before VAE decode.
 * Null or empty path disables saving. A later decode failure does not remove
 * the saved archive. Does not require retention. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_save_latents(
    slopfab_request* request, const char* path);

/* Keep a shared immutable latent snapshot on the generation handle. Default
 * off. Required for generation_save_latents and set_continuation_generation. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_retain_latents(
    slopfab_request* request, int32_t enable);

/* Load an owning snapshot now, so deleting/replacing the file later is safe.
 * overlap_frames must be 17*k+5, at least 5, and fit in the source. With
 * continuation set, request frames means NEW frames (rounded up to a multiple
 * of 17). The source canvas is inherited unless an explicit matching canvas
 * is set. Output pixels and saved latents contain the full extended clip.
 * Synthetic and still-image generation cannot continue. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_continuation_file(
    slopfab_request* request, const char* path, int32_t overlap_frames);

/* Share retained latents from a successful generation; source may be destroyed
 * after this returns. No disk I/O and no latent buffer copy. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_continuation_generation(
    slopfab_request* request, const slopfab_generation* source, int32_t overlap_frames);
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_clear_continuation(slopfab_request* request);

/* Save retained latents after a successful generation. NOT_READY while
 * running; INVALID_REQUEST when retention was not enabled. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_generation_save_latents(
    const slopfab_generation* generation, const char* path);

/* --- plan ------------------------------------------------------------------ */

/* Resolves `request` into `out_plan`. Reads no weights. With continuation,
 * aligned_frames/duration describe the full output; latent sizes and row
 * counts describe the bounded sampling window, including hidden overlap.
 * SLOPFAB_ERR_INVALID_REQUEST, with the reason on `slopfab_last_error()`, if the
 * request cannot be satisfied. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_resolve_plan(const slopfab_request* request,
                                                 slopfab_plan* out_plan);

/* The human-readable summary the CLI prints for `--dry-run`. On SLOPFAB_OK,
 * `*out_text` holds a NUL-terminated string the caller must release with
 * `slopfab_free_string`; on failure it is null. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_describe_plan(const slopfab_request* request, char** out_text);

/* --- generation ------------------------------------------------------------
 *
 * Starts the run on a worker thread and returns immediately. The request is
 * copied, so the caller may destroy or reuse theirs as soon as this returns —
 * the only sane contract when the run outlives the call by minutes.
 *
 * The request is resolved before the thread starts, so an unsatisfiable one
 * comes back as SLOPFAB_ERR_INVALID_REQUEST here rather than as a handle that
 * fails a millisecond later. `callback` may be null.
 *
 * **One at a time.** The model-reuse cache, the CUDA context and the device
 * arena are process-global, so a second start while one is still running
 * returns SLOPFAB_ERR_BUSY rather than corrupting the first. The run holds that
 * claim until it finishes, not until its handle is destroyed — so a caller may
 * start the next generation while still reading the previous one's pixels.
 * Serialise in the host if you want a queue; this only refuses to overlap. */

/* Releases everything retained by requests with reuse enabled. Returns
 * SLOPFAB_ERR_BUSY if a generation is active, so the cache can never be freed
 * while its worker is reading it. Finished generation pixel buffers are
 * independent and remain valid. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_reused_models_clear(void);

SLOPFAB_C_API int SLOPFAB_CALL slopfab_generation_start(const slopfab_request* request,
                                                     slopfab_progress_fn callback, void* userdata,
                                                     slopfab_generation** out_generation);

/* Asks the run to stop at its next checkpoint; returns without waiting. The
 * generation then finishes with SLOPFAB_ERR_CANCELLED. Cancellation is only as
 * fine-grained as the checkpoints: one issued during a multi-gigabyte
 * checkpoint read is not seen until that read completes. Null is a no-op. */
SLOPFAB_C_API void SLOPFAB_CALL slopfab_generation_cancel(slopfab_generation* generation);

/* SLOPFAB_ERR_NOT_READY while it runs, then SLOPFAB_OK or a failure code. Safe
 * to poll from any thread. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_generation_status(const slopfab_generation* generation);

/* Blocks until the run finishes or `timeout_ms` elapses, and returns the
 * status either way — so a timeout is reported as SLOPFAB_ERR_NOT_READY rather
 * than as a distinct code. Negative waits forever. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_generation_wait(slopfab_generation* generation,
                                                    int32_t timeout_ms);

/* Why this generation failed, or "" if it has not — including while it is
 * still running, which is a defined answer rather than an accident: the string
 * is not readable until the run has finished writing it. Owned by the
 * generation.
 *
 * This is where a run's message lives; `slopfab_last_error` cannot carry it,
 * because the run failed on a thread the caller never entered. */
SLOPFAB_C_API const char* SLOPFAB_CALL slopfab_generation_error(const slopfab_generation* generation);

/* Fills `out_output` with pointers into the finished generation's buffers.
 * SLOPFAB_ERR_NOT_READY while it is still running, or the failure code if it
 * failed. The pointers are valid until `slopfab_generation_destroy`. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_generation_output(const slopfab_generation* generation,
                                                      slopfab_output* out_output);

/* One frame converted to tightly packed 8-bit RGBA, alpha 255 — the layout a
 * texture upload or a screenshot wants, and the one conversion common enough
 * to be worth doing here rather than in every binding. `dst` must hold at
 * least `width * height * 4` bytes; `dst_bytes` is checked. Values are clamped
 * to [0,255] and rounded. */
SLOPFAB_C_API int SLOPFAB_CALL slopfab_generation_frame_rgba8(const slopfab_generation* generation,
                                                           int32_t frame_index, uint8_t* dst,
                                                           size_t dst_bytes);

/* Cancels if still running, waits for the worker to drain, and frees
 * everything including the pixel buffers. Every pointer from
 * `slopfab_generation_output` dangles after this. Null is a no-op.
 *
 * It blocks: the worker is joined rather than detached, because a detached one
 * would outlive the buffers it is about to write and the caller has no way to
 * know when the last CUDA call has drained. A cancel can take as long as the
 * checkpoint read it lands in.
 *
 * Not a way to interrupt a `slopfab_generation_wait` from another thread: this
 * frees the handle those waiters are parked on. Cancel, let the waits return,
 * then destroy. */
SLOPFAB_C_API void SLOPFAB_CALL slopfab_generation_destroy(slopfab_generation* generation);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SLOPFAB_CAPI_H */
