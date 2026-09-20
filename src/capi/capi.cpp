// The flat C ABI declared in include/slopfab/capi.h.
//
// Everything in this file exists to hold one line of that header true: no C++
// exception may cross the boundary. An exception unwinding into a caller that
// was not compiled by this compiler is undefined behaviour, and on MSVC it
// usually means an immediate process death with no diagnostic — in a host
// application, the user's unsaved work goes with it. So every entry point is
// wrapped, every throw becomes a status code, and the message goes into
// thread-local storage for `slopfab_last_error`.
//
// This is the only translation unit in slopfab_c, the target that builds
// slopfab.dll and carries the C ABI, and it is compiled into that target
// *alone*. It links slopfab_cuda,
// because `run_generate` is declared in slopfab/generate.h and implemented on
// the CUDA side — so a consumer that only wants the weight-free plan
// resolution still pulls the CUDA half in. That is a real cost, and the
// alternative, splitting the C surface across two DLLs, is worse.
//
// Building it into slopfab_cuda as well was tried and is wrong: every entry
// point would be compiled twice, once with capi.h seen as dllimport and once
// as dllexport, and the linker quietly picks one. See the note in CMakeLists.
#include "slopfab/capi.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "slopfab/attention_mode.h"
#include "slopfab/cuda/cublas_dispatch.h"
#include "slopfab/generate.h"
#include "slopfab/pipeline.h"
#include "slopfab/pixel_buffer.h"

namespace {

using slopfab::GeneratePlan;
using slopfab::GenerateRequest;
using slopfab::PixelBuffer;
using slopfab::RunOptions;
using slopfab::RunResult;
using slopfab::RunSamples;
using slopfab::RunStage;

// Thread-local so two threads driving two generations cannot overwrite each
// other's diagnosis. It is also why a *run's* failure message is not here —
// the run fails on a worker thread the caller never touches — and why
// `slopfab_generation_error` exists separately.
thread_local std::string g_last_error;

// Two overloads, and the split is the point.
//
// Most failures here are a string literal, and those call sites are the
// argument checks that run *before* `guarded` opens its try block — the null
// checks at the top of nearly every entry point. Passing a literal to the
// `std::string` overload would construct the temporary at the call site,
// outside any handler, so a host-side allocation failure would unwind straight
// through the C boundary. That is the one thing this file exists to prevent,
// and it would happen in the code written to report an error.
//
// So the literal form allocates nothing the caller can see and cannot throw:
// the only allocation is inside, and it is caught. Assigning a `const char*`
// to a std::string can throw; failing to record the message is survivable,
// returning the code is not optional.
int fail(int code, const char* message) noexcept {
  try {
    g_last_error = message;
  } catch (...) {
    // The diagnosis is lost but the status still gets back to the caller,
    // which is the half that matters. `clear()` is noexcept.
    g_last_error.clear();
  }
  return code;
}

// The dynamic form, for messages that name what was wrong. Every one of its
// call sites is inside `guarded`, so the construction of the argument is
// covered; moving into `g_last_error` is itself noexcept.
int fail(int code, std::string message) {
  g_last_error = std::move(message);
  return code;
}

bool contains_ci(const std::string& haystack, const char* needle) {
  const std::string lowered = [&haystack] {
    std::string s = haystack;
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
  }();
  return lowered.find(needle) != std::string::npos;
}

// Memory exhaustion is separated from every other runtime failure because the
// caller's response to it is different — a smaller geometry, not a bug report
// — and because it is the routine failure on a card that is also driving a
// desktop. It arrives as a message rather than a type: `cuda::check` throws
// std::runtime_error carrying cudaGetErrorString, which for an allocation
// failure is "out of memory".
int classify(const std::string& message) {
  if (contains_ci(message, "out of memory") || contains_ci(message, "bad allocation") ||
      contains_ci(message, "cudaerrormemoryallocation")) {
    return SLOPFAB_ERR_OUT_OF_MEMORY;
  }
  return SLOPFAB_ERR_RUNTIME;
}

// The wrapper every entry point uses. A lambda returning int keeps the
// try/catch in one place instead of in thirty.
template <typename Fn>
int guarded(Fn&& body) {
  try {
    const int status = body();
    if (status == SLOPFAB_OK) {
      // clear() is noexcept and releases no storage, so success cannot turn
      // into an allocation failure while retiring the previous diagnosis.
      // Centralising this here keeps every guarded C entry point consistent
      // with capi.h: last_error describes the most recent call, not an older
      // failure on the same thread.
      g_last_error.clear();
    }
    return status;
  } catch (const std::bad_alloc&) {
    // First, and it allocates nothing: the handler for running out of memory
    // must not be the thing that needs memory.
    return fail(SLOPFAB_ERR_OUT_OF_MEMORY, "out of memory");
  } catch (const std::exception& e) {
    // Copying the message and classifying it both allocate, so this handler
    // can itself throw — a double fault, needing a non-bad_alloc exception and
    // simultaneous host exhaustion, but the one remaining way an exception
    // could leave this function and cross the C boundary. The inner catch is
    // the difference between "no exception escapes" being true and being
    // true except when it matters most.
    try {
      const std::string message = e.what();
      return fail(classify(message), message);
    } catch (...) {
      return fail(SLOPFAB_ERR_RUNTIME,
                  "an exception reached the C ABI boundary and its message could not be recorded");
    }
  } catch (...) {
    return fail(SLOPFAB_ERR_UNKNOWN, "unknown exception at the C ABI boundary");
  }
}

// Allocated with the DLL's own allocator and freed by `slopfab_free_string`,
// which is in this DLL too. That pairing is the point: a caller's `free` on a
// pointer from another CRT's heap is undefined, and on Windows it is a crash
// often enough to matter.
char* dup_string(const std::string& text) {
  char* out = static_cast<char*>(std::malloc(text.size() + 1));
  if (out == nullptr) return nullptr;
  std::memcpy(out, text.c_str(), text.size() + 1);
  return out;
}

}  // namespace

// The handle types are declared at namespace scope because the header names
// them, and defined here so nothing about their contents is visible to a
// caller.
struct slopfab_request {
  GenerateRequest request;
  RunOptions options;
};

struct slopfab_reference_video {
  slopfab::ReferenceMedia media;
};

namespace {

template <typename Fn>
int reference_input_guarded(Fn&& body) {
  return guarded([&] {
    try { body(); }
    catch (const std::invalid_argument& e) {
      return fail(SLOPFAB_ERR_INVALID_ARGUMENT, e.what());
    }
    return SLOPFAB_OK;
  });
}

void attach_reference(slopfab_request* request, const slopfab::ReferenceMedia& media) {
  auto references = request->request.reference_media;
  references.push_back(std::make_shared<const slopfab::ReferenceMedia>(media));
  slopfab::validate_reference_media(request->request.reference_image_paths.size(), references);
  request->request.reference_media.swap(references);
}

}  // namespace

SLOPFAB_C_API int SLOPFAB_CALL slopfab_reference_video_create(
    double duration_seconds, slopfab_reference_video** out_video) {
  if (!out_video) return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "reference video: null output pointer");
  *out_video = nullptr;
  return reference_input_guarded([&] {
    auto video = std::make_unique<slopfab_reference_video>();
    video->media = slopfab::ReferenceMedia::video(duration_seconds);
    *out_video = video.release();
  });
}

SLOPFAB_C_API void SLOPFAB_CALL slopfab_reference_video_destroy(slopfab_reference_video* video) {
  delete video;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_reference_video_append_rgb24(
    slopfab_reference_video* video, const uint8_t* pixels, size_t buffer_bytes,
    int32_t width, int32_t height, size_t row_stride_bytes, double timestamp_seconds) {
  if (!video) return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "reference video: null handle");
  return reference_input_guarded([&] {
    video->media.append_frame(pixels, buffer_bytes, width, height, row_stride_bytes, 3, timestamp_seconds);
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_reference_video_append_rgba8(
    slopfab_reference_video* video, const uint8_t* pixels, size_t buffer_bytes,
    int32_t width, int32_t height, size_t row_stride_bytes, double timestamp_seconds) {
  if (!video) return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "reference video: null handle");
  return reference_input_guarded([&] {
    video->media.append_frame(pixels, buffer_bytes, width, height, row_stride_bytes, 4, timestamp_seconds);
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_reference_video_set_audio_f32(
    slopfab_reference_video* video, const float* samples, size_t float_count,
    int32_t channels, int32_t sample_rate, double start_seconds) {
  if (!video) return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "reference video: null handle");
  return reference_input_guarded([&] {
    video->media.set_audio(samples, float_count, channels, sample_rate, start_seconds);
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_add_reference_video(
    slopfab_request* request, const slopfab_reference_video* video) {
  if (!request || !video) return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "reference video: null handle");
  return reference_input_guarded([&] { attach_reference(request, video->media); });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_add_reference_audio_f32(
    slopfab_request* request, const float* samples, size_t float_count,
    int32_t channels, int32_t sample_rate) {
  if (!request) return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "reference audio: null request");
  return reference_input_guarded([&] {
    attach_reference(request, slopfab::ReferenceMedia::audio(samples, float_count, channels, sample_rate));
  });
}

struct slopfab_generation {
  GenerateRequest request;
  RunOptions options;

  slopfab_progress_fn callback = nullptr;
  void* userdata = nullptr;

  std::thread worker;
  std::atomic<bool> cancel{false};
  std::atomic<int> status{SLOPFAB_ERR_NOT_READY};
  std::mutex mutex;
  std::condition_variable done_cv;
  bool done = false;

  std::string error;
  // Written and read only by the worker, including when no callback is set.
  const char* stage_name = "starting";
  std::chrono::steady_clock::time_point started;

  // Moved out of the decoder's own buffers by `on_samples`, so the pixels are
  // never copied: they are decoded once and handed to the caller by pointer.
  PixelBuffer video;
  std::shared_ptr<const slopfab::LatentClip> latents;
  std::vector<float> audio;
  int channels = 0;
  int frames = 0;
  int width = 0;
  int height = 0;
  int audio_channels = 0;
  int audio_sample_rate = 0;
  double fps = 24.0;

  double seconds_conditioning = 0.0;
  double seconds_denoise = 0.0;
  double seconds_video_decode = 0.0;
  double seconds_audio_decode = 0.0;
  double seconds_total = 0.0;
  int steps_computed = 0;
  int steps_skipped = 0;

  double elapsed() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
  }

  void finish(int code, std::string message) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      error = std::move(message);
      done = true;
    }
    status.store(code);
    done_cv.notify_all();
  }
};

namespace {

// Claimed for the duration of a run, and released when the worker finishes
// rather than when its handle is destroyed — so a caller may start the next
// generation while still reading the previous one's pixels.
//
// This is not defensive programming. `run_generate` reaches a model-reuse
// cache that is a function-local static, a CUDA context, a device arena and
// two profiler singletons, none of which is per-call: two overlapping runs
// would share a transformer one of them is freeing. Refusing the second is the
// only answer this layer can give, and it has to be given here because a host
// driving the DLL from a UI thread has no way to know the API is serial.
std::atomic<bool> g_generation_active{false};

// SLOPFAB_OK when the run finished successfully, and otherwise the code and
// message describing how it ended.
//
// Shared by the two accessors a caller can reach before a run has finished.
// Each used to carry its own copy of this, which is exactly how two accessors
// on one handle drift into telling a caller different stories about it — they
// had already done so once, one reporting a blanket "not ready" for a run that
// had actually failed.
//
// The failure path copies the run's message, so it allocates and is guarded;
// the two paths above it cannot throw at all.
int report_terminal_status(const slopfab_generation* generation) noexcept {
  const int status = generation->status.load();
  if (status == SLOPFAB_OK) return SLOPFAB_OK;
  if (status == SLOPFAB_ERR_NOT_READY) {
    return fail(SLOPFAB_ERR_NOT_READY, "the generation is still running");
  }
  return guarded([&] {
    return fail(status, generation->error.empty() ? "the generation failed" : generation->error);
  });
}

// Both hooks are plain functions because `RunOptions` holds function pointers
// — see the note there — and both are noexcept in effect: a throw here would
// unwind through `run_generate` and out of the worker thread.
bool progress_hook(RunStage stage, int step, int steps, void* userdata) {
  auto* gen = static_cast<slopfab_generation*>(userdata);
  switch (stage) {
    case RunStage::kStarting: gen->stage_name = "starting"; break;
    case RunStage::kReferences: gen->stage_name = "reference encoding"; break;
    case RunStage::kConditioning: gen->stage_name = "prompt conditioning"; break;
    case RunStage::kTransformerLoad: gen->stage_name = "transformer loading"; break;
    case RunStage::kDenoising: gen->stage_name = "denoising"; break;
    case RunStage::kVideoDecode: gen->stage_name = "video VAE decoding"; break;
    case RunStage::kAudioDecode: gen->stage_name = "audio VAE decoding"; break;
    case RunStage::kDelivering: gen->stage_name = "output delivery"; break;
    case RunStage::kFinished: gen->stage_name = "finishing"; break;
  }
  if (gen->callback != nullptr) {
    slopfab_progress progress;
    progress.stage = static_cast<int32_t>(stage);
    progress.step = step;
    progress.total_steps = steps;
    progress.elapsed_seconds = gen->elapsed();
    gen->callback(&progress, gen->userdata);
  }
  return !gen->cancel.load();
}

bool samples_hook(RunSamples& samples, void* userdata) {
  auto* gen = static_cast<slopfab_generation*>(userdata);
  gen->channels = samples.channels;
  gen->frames = samples.frames;
  gen->width = samples.width;
  gen->height = samples.height;
  // A move, not a copy. At 248 frames of 1344x768 the video plane is 2.3 GB;
  // copying it here would double the peak host footprint of every run for no
  // reason at all.
  if (samples.video != nullptr) gen->video = std::move(*samples.video);
  if (samples.audio != nullptr) gen->audio = std::move(*samples.audio);
  gen->audio_channels = samples.audio_channels;
  gen->audio_sample_rate = samples.audio_sample_rate;
  return true;
}

void latents_hook(const std::shared_ptr<const slopfab::LatentClip>& latents, void* userdata) {
  static_cast<slopfab_generation*>(userdata)->latents = latents;
}

void run_worker(slopfab_generation* gen) {
  int code = SLOPFAB_ERR_UNKNOWN;
  std::string message;
  try {
    const GeneratePlan plan = slopfab::resolve_plan(gen->request);
    if (plan.duration_seconds > 0.0) {
      gen->fps = static_cast<double>(plan.aligned_frames) / plan.duration_seconds;
    }
    RunOptions options = gen->options;
    options.on_progress = &progress_hook;
    options.on_samples = &samples_hook;
    options.hook_userdata = gen;
    const RunResult result = slopfab::run_generate(gen->request, plan, options);
    gen->seconds_conditioning = result.seconds_conditioning;
    gen->seconds_denoise = result.seconds_denoise;
    gen->seconds_video_decode = result.seconds_video_decode;
    gen->seconds_audio_decode = result.seconds_audio_decode;
    gen->steps_computed = result.steps_computed;
    gen->steps_skipped = result.steps_skipped;
    gen->seconds_total = gen->elapsed();
    if (result.cancelled) {
      code = SLOPFAB_ERR_CANCELLED;
      message = result.message;
    } else if (!result.ok) {
      code = classify(result.message);
      message = std::string(gen->stage_name) + ": " + result.message;
    } else {
      code = SLOPFAB_OK;
    }
  } catch (const std::bad_alloc&) {
    code = SLOPFAB_ERR_OUT_OF_MEMORY;
    message = "out of memory";
  } catch (const std::exception& e) {
    message = std::string(gen->stage_name) + ": " + e.what();
    code = classify(message);
  } catch (...) {
    code = SLOPFAB_ERR_UNKNOWN;
    message = "unknown exception in the generation worker";
  }
  // Released before `finish`, so a caller woken by `slopfab_generation_wait`
  // can start the next run immediately rather than racing this thread's
  // remaining bookkeeping.
  g_generation_active.store(false);
  gen->finish(code, std::move(message));
}

}  // namespace

extern "C" {

SLOPFAB_C_API uint32_t SLOPFAB_CALL slopfab_capi_version(void) {
  return (static_cast<uint32_t>(SLOPFAB_CAPI_VERSION_MAJOR) << 24) |
         (static_cast<uint32_t>(SLOPFAB_CAPI_VERSION_MINOR) << 12) |
         static_cast<uint32_t>(SLOPFAB_CAPI_VERSION_PATCH);
}

// Built from the same macros rather than written out, so the string and the
// packed number cannot disagree after someone bumps one of them.
#define SLOPFAB_CAPI_STRINGIFY_(x) #x
#define SLOPFAB_CAPI_STRINGIFY(x) SLOPFAB_CAPI_STRINGIFY_(x)

SLOPFAB_C_API const char* SLOPFAB_CALL slopfab_capi_version_string(void) {
  return SLOPFAB_CAPI_STRINGIFY(SLOPFAB_CAPI_VERSION_MAJOR) "." SLOPFAB_CAPI_STRINGIFY(
      SLOPFAB_CAPI_VERSION_MINOR) "." SLOPFAB_CAPI_STRINGIFY(SLOPFAB_CAPI_VERSION_PATCH);
}

SLOPFAB_C_API const char* SLOPFAB_CALL slopfab_last_error(void) { return g_last_error.c_str(); }

SLOPFAB_C_API void SLOPFAB_CALL slopfab_free_string(char* text) { std::free(text); }

SLOPFAB_C_API int SLOPFAB_CALL slopfab_cuda_set_version(const char* version) {
  if (version == nullptr)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                "slopfab_cuda_set_version: null version");
  if (std::strcmp(version, "auto") != 0 && std::strcmp(version, "13") != 0 &&
      std::strcmp(version, "12") != 0)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                "slopfab_cuda_set_version: expected auto, 13, or 12");
  return guarded([&] {
    slopfab::cuda::set_cublas_version_request(version);
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_cuda_loaded_major(int32_t* out_major) {
  if (out_major == nullptr)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                "slopfab_cuda_loaded_major: null out_major");
  *out_major = 0;
  return guarded([&] {
    *out_major = slopfab::cuda::cublas_loaded_major();
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API slopfab_request* SLOPFAB_CALL slopfab_request_create(void) {
  try {
    auto* request = new slopfab_request();
    // The library default is an MP4 path, which a C-API caller never uses:
    // `on_samples` takes the pixels and nothing is written. Cleared so that a
    // future code path which does consult it cannot quietly write video.mp4
    // into whatever the host's working directory happens to be.
    request->request.out_path.clear();
    request->request.raw_output = true;
    return request;
  } catch (...) {
    g_last_error = "out of memory allocating a request";
    return nullptr;
  }
}

SLOPFAB_C_API void SLOPFAB_CALL slopfab_request_destroy(slopfab_request* request) { delete request; }

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_prompt(slopfab_request* request, const char* utf8) {
  if (request == nullptr || utf8 == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_request_set_prompt: null argument");
  }
  return guarded([&] {
    request->request.prompt = utf8;
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_prompt_file(slopfab_request* request, const char* path) {
  if (request == nullptr || path == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_request_set_prompt_file: null argument");
  }
  return guarded([&] {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      return fail(SLOPFAB_ERR_NOT_FOUND,
                  std::string("cannot read prompt file '") + path + "'");
    }
    std::ostringstream contents;
    contents << in.rdbuf();
    std::string text = contents.str();
    // The same normalisation `slopfab generate --prompt-file` applies, so a
    // file that works on the CLI conditions identically here: a UTF-8 BOM,
    // CRLF line endings and surrounding blank space are all things an editor
    // adds and no prompt wants in its token stream.
    if (text.rfind("\xEF\xBB\xBF", 0) == 0) text.erase(0, 3);
    text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
    const size_t first = text.find_first_not_of(" \t\n");
    if (first == std::string::npos) {
      return fail(SLOPFAB_ERR_INVALID_REQUEST,
                  std::string("prompt file '") + path + "' has no prompt in it");
    }
    const size_t last = text.find_last_not_of(" \t\n");
    request->request.prompt = text.substr(first, last - first + 1);
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_aspect(slopfab_request* request, int32_t width, int32_t height) {
  if (request == nullptr || width <= 0 || height <= 0) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_request_set_aspect: needs positive extents");
  }
  request->request.aspect_w = width;
  request->request.aspect_h = height;
  request->request.canvas_width = 0;
  request->request.canvas_height = 0;
  return SLOPFAB_OK;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_resolution(slopfab_request* request, int32_t width, int32_t height) {
  if (request == nullptr || width <= 0 || height <= 0) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                "slopfab_request_set_resolution: needs positive extents");
  }
  request->request.canvas_width = width;
  request->request.canvas_height = height;
  return SLOPFAB_OK;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_frames(slopfab_request* request, int32_t frames) {
  if (request == nullptr || frames <= 0) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_request_set_frames: needs a positive count");
  }
  request->request.num_frames = frames;
  return SLOPFAB_OK;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_still_image(slopfab_request* request,
                                                            int32_t enable) {
  if (request == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                "slopfab_request_set_still_image: null request");
  }
  request->request.still_image = enable != 0;
  return SLOPFAB_OK;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_steps(slopfab_request* request, int32_t steps) {
  if (request == nullptr || steps < 2) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                "slopfab_request_set_steps: the grid includes a terminal zero, so it needs at "
                "least 2 points");
  }
  request->request.num_inference_steps = steps;
  return SLOPFAB_OK;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_seed(slopfab_request* request, uint64_t seed) {
  if (request == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_request_set_seed: null request");
  }
  request->request.seed = seed;
  return SLOPFAB_OK;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_model_path(slopfab_request* request, int32_t which, const char* path) {
  if (request == nullptr || path == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_request_set_model_path: null argument");
  }
  return guarded([&] {
    switch (which) {
      case SLOPFAB_MODEL_TRANSFORMER: request->request.transformer_path = path; break;
      case SLOPFAB_MODEL_TEXT_ENCODER: request->request.text_encoder_path = path; break;
      case SLOPFAB_MODEL_TOKENIZER: request->request.tokenizer_path = path; break;
      case SLOPFAB_MODEL_VIDEO_VAE: request->request.video_vae_path = path; break;
      case SLOPFAB_MODEL_AUDIO_VAE: request->request.audio_vae_path = path; break;
      default:
        return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                    "slopfab_request_set_model_path: unknown model id " + std::to_string(which));
    }
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_prompt_embedding_path(
    slopfab_request* request, const char* path) {
  if (request == nullptr || path == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                "slopfab_request_set_prompt_embedding_path: null argument");
  }
  return guarded([&] {
    request->options.prompt_embedding_path = path;
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_animate(
    slopfab_request* request, int32_t enable, int32_t preserve_driving_audio) {
  if (!request) return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "set_animate: null request");
  request->request.animate = enable != 0;
  request->request.preserve_driving_audio = enable != 0 && preserve_driving_audio != 0;
  if (enable) {
    request->request.num_inference_steps = 4;
    request->request.schedule = slopfab::sampler::ScheduleKind::kDefault;
    request->options.sampler = slopfab::sampler::SamplerKind::kEuler;
  }
  return SLOPFAB_OK;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_add_lora(
    slopfab_request* request, const char* path, float strength) {
  if (!request || !path || !*path || !std::isfinite(strength))
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "LoRA needs a request, nonempty path and finite strength");
  return guarded([&] {
    request->request.loras.push_back({path, strength});
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_clear_loras(slopfab_request* request) {
  if (!request) return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "clear_loras: null request");
  return guarded([&] {
    request->request.loras.clear();
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_schedule(
    slopfab_request* request, int32_t schedule) {
  if (!request || (schedule != SLOPFAB_SCHEDULE_DEFAULT && schedule != SLOPFAB_SCHEDULE_TAOMATE_3STEP))
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "unknown denoising schedule or null request");
  return guarded([&] {
    request->request.schedule = schedule == SLOPFAB_SCHEDULE_DEFAULT
        ? slopfab::sampler::ScheduleKind::kDefault : slopfab::sampler::ScheduleKind::kTaoMate3Step;
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_sampling_settings(
    slopfab_request* request, const char* json) {
  if (!request) return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "sampling settings: null request");
  return guarded([&] {
    try {
      auto settings = json ? slopfab::parse_sampling_settings(json) : slopfab::SamplingSettings{};
      request->request.sampling = std::move(settings);
    } catch (const std::exception& e) {
      return fail(SLOPFAB_ERR_INVALID_ARGUMENT, e.what());
    }
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_motion_cache(
    slopfab_request* request, int32_t enabled, float reuse_threshold,
    float motion_strength, int32_t warmup_steps, int32_t max_consecutive_skips,
    float start_percent, float end_percent, int32_t subsample_factor, int32_t verbose) {
  if (!request || (enabled != 0 && enabled != 1) || (verbose != 0 && verbose != 1))
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "MotionCache: null request or invalid boolean");
  return guarded([&] {
    slopfab::dit::MotionCacheConfig config;
    config.enabled = enabled != 0;
    config.reuse_threshold = reuse_threshold;
    config.motion_strength = motion_strength;
    config.warmup_steps = warmup_steps;
    config.max_consecutive_skips = max_consecutive_skips;
    config.start_percent = start_percent;
    config.end_percent = end_percent;
    config.subsample_factor = subsample_factor;
    config.verbose = verbose != 0;
    try { config.validate(); }
    catch (const std::invalid_argument& e) { return fail(SLOPFAB_ERR_INVALID_ARGUMENT, e.what()); }
    request->request.motion_cache = config;
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_add_refmod(
    slopfab_request* request, const char* path, float strength, int32_t copies) {
  if (!request || !path || !*path || !std::isfinite(strength) || strength < 0 || strength > 1 || copies < 1 || copies > 10)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "refmod needs a request, path, strength 0..1 and copies 1..10");
  return guarded([&] {
    auto refs = request->request.refmods;
    refs.push_back({slopfab::RefMod::load(path), strength, copies});
    slopfab::validate_refmods(refs);
    request->request.refmods.swap(refs);
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_save_latents(
    slopfab_request* request, const char* path) {
  if (!request) return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "save latents: null request");
  return guarded([&] {
    request->options.save_latents_path = path ? path : "";
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_retain_latents(
    slopfab_request* request, int32_t enable) {
  if (!request) return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "retain latents: null request");
  return guarded([&] {
    request->options.on_latents = enable ? &latents_hook : nullptr;
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_continuation_file(
    slopfab_request* request, const char* path, int32_t overlap_frames) {
  if (!request || !path || !*path || overlap_frames < 5 || overlap_frames % 17 != 5)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "continuation needs a request, archive path and 17*k+5 overlap >=5");
  return guarded([&] {
    auto clip = slopfab::LatentClip::load(path);
    try { (void)slopfab::plan_continuation(*clip, overlap_frames, 17); }
    catch (const std::exception& e) { return fail(SLOPFAB_ERR_INVALID_REQUEST, e.what()); }
    request->request.continuation = std::move(clip);
    request->request.continuation_overlap_frames = overlap_frames;
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_continuation_generation(
    slopfab_request* request, const slopfab_generation* source, int32_t overlap_frames) {
  if (!request || !source || overlap_frames < 5 || overlap_frames % 17 != 5)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "continuation needs a request, source generation and 17*k+5 overlap >=5");
  const int terminal = report_terminal_status(source);
  if (terminal != SLOPFAB_OK) return terminal;
  return guarded([&] {
    if (!source->latents)
      return fail(SLOPFAB_ERR_INVALID_REQUEST, "enable latent retention before starting the source generation");
    try { (void)slopfab::plan_continuation(*source->latents, overlap_frames, 17); }
    catch (const std::exception& e) { return fail(SLOPFAB_ERR_INVALID_REQUEST, e.what()); }
    request->request.continuation = source->latents;
    request->request.continuation_overlap_frames = overlap_frames;
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_clear_continuation(slopfab_request* request) {
  if (!request) return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "clear continuation: null request");
  return guarded([&] {
    request->request.continuation.reset();
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_generation_save_latents(
    const slopfab_generation* generation, const char* path) {
  if (!generation || !path || !*path)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "save latents needs a generation and path");
  const int terminal = report_terminal_status(generation);
  if (terminal != SLOPFAB_OK) return terminal;
  return guarded([&] {
    if (!generation->latents)
      return fail(SLOPFAB_ERR_INVALID_REQUEST, "enable latent retention before starting the generation");
    generation->latents->save(path);
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_clear_refmods(slopfab_request* request) {
  if (!request) return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "clear_refmods: null request");
  return guarded([&] {
    request->request.refmods.clear();
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_add_reference_image(slopfab_request* request, const char* path) {
  if (request == nullptr || path == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_request_add_reference_image: null argument");
  }
  return guarded([&] {
    if (request->request.reference_image_paths.size() >= 9) {
      return fail(SLOPFAB_ERR_INVALID_REQUEST,
                  "MiniMax-H3 Ref2VA accepts at most 9 reference images");
    }
    if (request->request.reference_image_paths.size() + request->request.reference_media.size() >= 12) {
      return fail(SLOPFAB_ERR_INVALID_REQUEST, "MiniMax-H3 Ref2VA accepts at most 12 references in total");
    }
    request->request.reference_image_paths.emplace_back(path);
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_attention(slopfab_request* request, const char* mode) {
  if (request == nullptr || mode == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_request_set_attention: null argument");
  }
  // Guarded because the failure message quotes the caller's string, so the
  // reporting path allocates.
  return guarded([&] {
    const std::string name = mode;
    if (!slopfab::parse_attention_mode(name, &request->options.attention_mode)) {
      return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                  "unknown attention mode '" + name +
                      "'; expected none, flash2, sage2, sol, sol-experimental or exact");
    }
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_inference_backend(
    slopfab_request* request, int32_t backend) {
  if (request == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                "slopfab_request_set_inference_backend: null request");
  }
  switch (backend) {
    case SLOPFAB_INFERENCE_CUDA:
      request->options.inference_backend = slopfab::DeviceBackend::kCuda;
      return SLOPFAB_OK;
    case SLOPFAB_INFERENCE_VULKAN:
      request->options.inference_backend = slopfab::DeviceBackend::kVulkan;
      return SLOPFAB_OK;
    default:
      return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                  "slopfab_request_set_inference_backend: unknown backend " +
                      std::to_string(backend));
  }
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_synthetic_latents(slopfab_request* request, int32_t enable) {
  if (request == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_request_set_synthetic_latents: null request");
  }
  request->options.source =
      enable != 0 ? slopfab::LatentSource::kSyntheticNoise : slopfab::LatentSource::kDenoise;
  return SLOPFAB_OK;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_verbose(slopfab_request* request, int32_t enable) {
  if (request == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_request_set_verbose: null request");
  }
  request->options.verbose = enable != 0;
  return SLOPFAB_OK;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_reuse_models(slopfab_request* request,
                                                             int32_t enable) {
  if (request == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                "slopfab_request_set_reuse_models: null request");
  }
  request->options.reuse_models = enable != 0;
  return SLOPFAB_OK;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_resolve_plan(const slopfab_request* request, slopfab_plan* out_plan) {
  if (request == nullptr || out_plan == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_resolve_plan: null argument");
  }
  return guarded([&] {
    GeneratePlan plan;
    try {
      if (request->request.continuation && request->options.source != slopfab::LatentSource::kDenoise)
        throw std::invalid_argument("continuation requires denoising");
      plan = slopfab::resolve_plan(request->request);
      slopfab::validate_sampling_sampler(plan, request->options.sampler);
    } catch (const std::exception& e) {
      // Every throw out of resolve_plan is the request being unsatisfiable —
      // an aspect outside 1:4..4:1, too few frames, a schedule shorter than
      // one step — and none of it is a runtime fault. Reported as such so a
      // caller can tell "you asked for the impossible" from "the machine
      // failed", which are not the same message to a user.
      return fail(SLOPFAB_ERR_INVALID_REQUEST, e.what());
    }
    out_plan->canvas_width = plan.canvas_width;
    out_plan->canvas_height = plan.canvas_height;
    out_plan->aligned_frames = plan.aligned_frames;
    out_plan->duration_seconds = plan.duration_seconds;
    out_plan->num_model_evaluations = plan.num_model_evaluations();
    out_plan->sequence_rows_without_text = plan.sequence_length_without_text();
    out_plan->latent_frames = plan.layout.num_latent_frames;
    out_plan->latent_height = plan.layout.latent_height;
    out_plan->latent_width = plan.layout.latent_width;
    out_plan->num_video_rows = plan.layout.num_video_rows;
    out_plan->num_audio_rows = plan.layout.num_audio_rows;
    out_plan->num_audio_latents = plan.layout.num_audio_latents;
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_describe_plan(const slopfab_request* request, char** out_text) {
  if (request == nullptr || out_text == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_describe_plan: null argument");
  }
  *out_text = nullptr;
  return guarded([&] {
    GeneratePlan plan;
    try {
      if (request->request.continuation && request->options.source != slopfab::LatentSource::kDenoise)
        throw std::invalid_argument("continuation requires denoising");
      plan = slopfab::resolve_plan(request->request);
      slopfab::validate_sampling_sampler(plan, request->options.sampler);
    } catch (const std::exception& e) {
      return fail(SLOPFAB_ERR_INVALID_REQUEST, e.what());
    }
    char* text = dup_string(slopfab::describe_plan(request->request, plan));
    if (text == nullptr) return fail(SLOPFAB_ERR_OUT_OF_MEMORY, "out of memory");
    *out_text = text;
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_reused_models_clear(void) {
  bool expected = false;
  if (!g_generation_active.compare_exchange_strong(expected, true)) {
    return fail(SLOPFAB_ERR_BUSY,
                "cannot clear reused models while a generation is running");
  }
  struct ActiveClaim {
    ~ActiveClaim() { g_generation_active.store(false); }
  } claim;
  return guarded([] {
    slopfab::clear_reused_generation_models();
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_generation_start(const slopfab_request* request, slopfab_progress_fn callback,
                            void* userdata, slopfab_generation** out_generation) {
  if (request == nullptr || out_generation == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_generation_start: null argument");
  }
  *out_generation = nullptr;
  return guarded([&] {
    // Resolved before the thread exists so an unsatisfiable request is
    // reported synchronously, as a return code, rather than as a handle that
    // fails a millisecond later.
    try {
      if (request->request.continuation && request->options.source != slopfab::LatentSource::kDenoise)
        throw std::invalid_argument("continuation requires denoising");
      const auto plan = slopfab::resolve_plan(request->request);
      slopfab::validate_sampling_sampler(plan, request->options.sampler);
    } catch (const std::exception& e) {
      return fail(SLOPFAB_ERR_INVALID_REQUEST, e.what());
    }
    // Claimed before anything is allocated, so a refused start costs nothing.
    bool expected = false;
    if (!g_generation_active.compare_exchange_strong(expected, true)) {
      return fail(SLOPFAB_ERR_BUSY, "a generation is already running in this process");
    }

    // Released by the destructor on every path out of here that is not a
    // successfully launched worker — and there are several, because the
    // allocation and the six string copies below can all throw straight past
    // this frame into `guarded`. Getting that wrong does not fail the call
    // that threw; it strands the flag at true and makes *every* later start
    // return SLOPFAB_ERR_BUSY for the life of the process, waiting on a run
    // that does not exist. Once the thread is running, the worker owns the
    // claim and releases it in `run_worker`.
    struct ActiveClaim {
      bool held = true;
      ~ActiveClaim() {
        if (held) g_generation_active.store(false);
      }
      void release() { held = false; }
    } claim;

    // Owning, so that a throw anywhere below frees it. The string copies on
    // the next lines can throw, and `std::thread`'s constructor is only
    // *documented* to throw std::system_error — catching that one type and
    // deleting by hand left the handle leaked for anything else.
    std::unique_ptr<slopfab_generation> gen(new slopfab_generation());
    // The request is copied, so the caller may destroy or reuse theirs the
    // moment this returns. It is the only sane contract across an ABI: the
    // run outlives the call by minutes.
    gen->request = request->request;
    gen->options = request->options;
    gen->callback = callback;
    gen->userdata = userdata;
    gen->started = std::chrono::steady_clock::now();
    try {
      gen->worker = std::thread(&run_worker, gen.get());
    } catch (const std::system_error& e) {
      // `claim` and `gen` both still hold, so unwinding releases the flag and
      // frees the handle.
      return fail(SLOPFAB_ERR_RUNTIME, std::string("cannot start a worker thread: ") + e.what());
    }
    // The worker is running: it now owns the claim, and the caller owns the
    // handle. Nothing below here may throw.
    claim.release();
    *out_generation = gen.release();
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API void SLOPFAB_CALL slopfab_generation_cancel(slopfab_generation* generation) {
  if (generation == nullptr) return;
  generation->cancel.store(true);
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_generation_status(const slopfab_generation* generation) {
  if (generation == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_generation_status: null generation");
  }
  return generation->status.load();
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_generation_wait(slopfab_generation* generation, int32_t timeout_ms) {
  if (generation == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_generation_wait: null generation");
  }
  std::unique_lock<std::mutex> lock(generation->mutex);
  if (timeout_ms < 0) {
    generation->done_cv.wait(lock, [generation] { return generation->done; });
  } else {
    generation->done_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                                 [generation] { return generation->done; });
  }
  return generation->status.load();
}

SLOPFAB_C_API const char* SLOPFAB_CALL slopfab_generation_error(const slopfab_generation* generation) {
  if (generation == nullptr) return "";
  // The status load is what makes the read below safe, so it is a guard and
  // not an optimisation. `finish` assigns `error` and *then* stores a terminal
  // status; observing that status here synchronises with the assignment, and
  // after it nothing writes `error` again.
  //
  // Reading unconditionally is what the obvious version does, and it is a
  // crash rather than a torn read: a host polling this beside
  // `slopfab_generation_status` in a UI loop would be calling `c_str()` while
  // the worker move-assigns the string underneath it, and the SSO-to-heap
  // transition hands back a pointer into a buffer being freed.
  if (generation->status.load() == SLOPFAB_ERR_NOT_READY) return "";
  return generation->error.c_str();
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_generation_output(const slopfab_generation* generation, slopfab_output* out_output) {
  if (generation == nullptr || out_output == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_generation_output: null argument");
  }
  const int status = report_terminal_status(generation);
  if (status != SLOPFAB_OK) return status;
  out_output->video = generation->video.empty() ? nullptr : generation->video.data();
  out_output->video_float_count = generation->video.size();
  out_output->channels = generation->channels;
  out_output->frames = generation->frames;
  out_output->width = generation->width;
  out_output->height = generation->height;
  out_output->audio = generation->audio.empty() ? nullptr : generation->audio.data();
  out_output->audio_float_count = generation->audio.size();
  out_output->audio_channels = generation->audio_channels;
  out_output->audio_sample_rate = generation->audio_sample_rate;
  out_output->audio_frames =
      generation->audio_channels > 0
          ? static_cast<int64_t>(generation->audio.size()) / generation->audio_channels
          : 0;
  out_output->fps = generation->fps;
  out_output->seconds_conditioning = generation->seconds_conditioning;
  out_output->seconds_denoise = generation->seconds_denoise;
  out_output->seconds_video_decode = generation->seconds_video_decode;
  out_output->seconds_audio_decode = generation->seconds_audio_decode;
  out_output->seconds_total = generation->seconds_total;
  out_output->steps_computed = generation->steps_computed;
  out_output->steps_skipped = generation->steps_skipped;
  return SLOPFAB_OK;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_generation_frame_rgba8(const slopfab_generation* generation, int32_t frame_index,
                                  uint8_t* dst, size_t dst_bytes) {
  if (generation == nullptr || dst == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_generation_frame_rgba8: null argument");
  }
  // Reports what actually happened rather than a blanket "not ready", so that
  // this and `slopfab_generation_output` cannot tell a caller two different
  // stories about the same handle.
  const int status = report_terminal_status(generation);
  if (status != SLOPFAB_OK) return status;

  // Guarded from here on: every check below names the offending number in its
  // message, so the reporting path allocates.
  return guarded([&] {
    if (frame_index < 0 || frame_index >= generation->frames) {
      return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                  "frame " + std::to_string(frame_index) + " is outside 0.." +
                      std::to_string(generation->frames - 1));
    }
    const size_t pixels = static_cast<size_t>(generation->width) * generation->height;
    if (dst_bytes < pixels * 4) {
      return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "the destination holds " +
                                                   std::to_string(dst_bytes) +
                                                   " bytes; this frame needs " +
                                                   std::to_string(pixels * 4));
    }
    // Every other argument is validated; these are the invariants that govern
    // the actual memory access, and the loop below hardcodes three planes. The
    // decoder yields three today, so this is the check that keeps a future
    // fourth channel — or a short buffer from a partial decode — from being
    // read out of bounds instead of reported.
    if (generation->channels != 3) {
      return fail(SLOPFAB_ERR_RUNTIME,
                  "frame_rgba8 needs 3 channels, got " + std::to_string(generation->channels));
    }
    if (generation->video.size() < 3 * pixels * static_cast<size_t>(generation->frames)) {
      return fail(SLOPFAB_ERR_RUNTIME, "the decoded video buffer is shorter than its own geometry");
    }
    const float* video = generation->video.data();
    const size_t plane = pixels * static_cast<size_t>(generation->frames);
    const size_t base = static_cast<size_t>(frame_index) * pixels;
    for (size_t i = 0; i < pixels; ++i) {
      for (int c = 0; c < 3; ++c) {
        const float v = video[c * plane + base + i] * 255.0f;
        const float clamped = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
        dst[i * 4 + static_cast<size_t>(c)] = static_cast<uint8_t>(clamped + 0.5f);
      }
      dst[i * 4 + 3] = 255;
    }
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API void SLOPFAB_CALL slopfab_generation_destroy(slopfab_generation* generation) {
  if (generation == nullptr) return;
  generation->cancel.store(true);
  // Joined rather than detached. A detached worker would outlive the pixels
  // it is about to write into, and the caller has no way to know when the
  // last CUDA call has drained.
  if (generation->worker.joinable()) generation->worker.join();
  delete generation;
}

}  // extern "C"
