// The flat C ABI declared in include/vidfab/capi.h.
//
// Everything in this file exists to hold one line of that header true: no C++
// exception may cross the boundary. An exception unwinding into a caller that
// was not compiled by this compiler is undefined behaviour, and on MSVC it
// usually means an immediate process death with no diagnostic — in a host
// application, the user's unsaved work goes with it. So every entry point is
// wrapped, every throw becomes a status code, and the message goes into
// thread-local storage for `vidfab_last_error`.
//
// This is the only translation unit in vidfab_c, the DLL that carries the C
// ABI, and it is compiled into that target *alone*. It links vidfab_cuda,
// because `run_generate` is declared in vidfab/generate.h and implemented on
// the CUDA side — so a consumer that only wants the weight-free plan
// resolution still pulls the CUDA half in. That is a real cost, and the
// alternative, splitting the C surface across two DLLs, is worse.
//
// Building it into vidfab_cuda as well was tried and is wrong: every entry
// point would be compiled twice, once with capi.h seen as dllimport and once
// as dllexport, and the linker quietly picks one. See the note in CMakeLists.
#include "vidfab/capi.h"

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

#include "vidfab/attention_mode.h"
#include "vidfab/generate.h"
#include "vidfab/pipeline.h"
#include "vidfab/pixel_buffer.h"

namespace {

using vidfab::GeneratePlan;
using vidfab::GenerateRequest;
using vidfab::PixelBuffer;
using vidfab::RunOptions;
using vidfab::RunResult;
using vidfab::RunSamples;
using vidfab::RunStage;

// Thread-local so two threads driving two generations cannot overwrite each
// other's diagnosis. It is also why a *run's* failure message is not here —
// the run fails on a worker thread the caller never touches — and why
// `vidfab_generation_error` exists separately.
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
    return VIDFAB_ERR_OUT_OF_MEMORY;
  }
  return VIDFAB_ERR_RUNTIME;
}

// The wrapper every entry point uses. A lambda returning int keeps the
// try/catch in one place instead of in thirty.
template <typename Fn>
int guarded(Fn&& body) {
  try {
    return body();
  } catch (const std::bad_alloc&) {
    // First, and it allocates nothing: the handler for running out of memory
    // must not be the thing that needs memory.
    return fail(VIDFAB_ERR_OUT_OF_MEMORY, "out of memory");
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
      return fail(VIDFAB_ERR_RUNTIME,
                  "an exception reached the C ABI boundary and its message could not be recorded");
    }
  } catch (...) {
    return fail(VIDFAB_ERR_UNKNOWN, "unknown exception at the C ABI boundary");
  }
}

// Allocated with the DLL's own allocator and freed by `vidfab_free_string`,
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
struct vidfab_request {
  GenerateRequest request;
  RunOptions options;
};

struct vidfab_generation {
  GenerateRequest request;
  RunOptions options;

  vidfab_progress_fn callback = nullptr;
  void* userdata = nullptr;

  std::thread worker;
  std::atomic<bool> cancel{false};
  std::atomic<int> status{VIDFAB_ERR_NOT_READY};
  std::mutex mutex;
  std::condition_variable done_cv;
  bool done = false;

  std::string error;
  std::chrono::steady_clock::time_point started;

  // Moved out of the decoder's own buffers by `on_samples`, so the pixels are
  // never copied: they are decoded once and handed to the caller by pointer.
  PixelBuffer video;
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

// VIDFAB_OK when the run finished successfully, and otherwise the code and
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
int report_terminal_status(const vidfab_generation* generation) noexcept {
  const int status = generation->status.load();
  if (status == VIDFAB_OK) return VIDFAB_OK;
  if (status == VIDFAB_ERR_NOT_READY) {
    return fail(VIDFAB_ERR_NOT_READY, "the generation is still running");
  }
  return guarded([&] {
    return fail(status, generation->error.empty() ? "the generation failed" : generation->error);
  });
}

// Both hooks are plain functions because `RunOptions` holds function pointers
// — see the note there — and both are noexcept in effect: a throw here would
// unwind through `run_generate` and out of the worker thread.
bool progress_hook(RunStage stage, int step, int steps, void* userdata) {
  auto* gen = static_cast<vidfab_generation*>(userdata);
  if (gen->callback != nullptr) {
    vidfab_progress progress;
    progress.stage = static_cast<int32_t>(stage);
    progress.step = step;
    progress.total_steps = steps;
    progress.elapsed_seconds = gen->elapsed();
    gen->callback(&progress, gen->userdata);
  }
  return !gen->cancel.load();
}

bool samples_hook(RunSamples& samples, void* userdata) {
  auto* gen = static_cast<vidfab_generation*>(userdata);
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

void run_worker(vidfab_generation* gen) {
  int code = VIDFAB_ERR_UNKNOWN;
  std::string message;
  try {
    const GeneratePlan plan = vidfab::resolve_plan(gen->request);
    if (plan.duration_seconds > 0.0) {
      gen->fps = static_cast<double>(plan.aligned_frames) / plan.duration_seconds;
    }
    RunOptions options = gen->options;
    options.on_progress = &progress_hook;
    options.on_samples = &samples_hook;
    options.hook_userdata = gen;
    const RunResult result = vidfab::run_generate(gen->request, plan, options);
    gen->seconds_conditioning = result.seconds_conditioning;
    gen->seconds_denoise = result.seconds_denoise;
    gen->seconds_video_decode = result.seconds_video_decode;
    gen->seconds_audio_decode = result.seconds_audio_decode;
    gen->steps_computed = result.steps_computed;
    gen->steps_skipped = result.steps_skipped;
    gen->seconds_total = gen->elapsed();
    if (result.cancelled) {
      code = VIDFAB_ERR_CANCELLED;
      message = result.message;
    } else if (!result.ok) {
      code = classify(result.message);
      message = result.message;
    } else {
      code = VIDFAB_OK;
    }
  } catch (const std::bad_alloc&) {
    code = VIDFAB_ERR_OUT_OF_MEMORY;
    message = "out of memory";
  } catch (const std::exception& e) {
    message = e.what();
    code = classify(message);
  } catch (...) {
    code = VIDFAB_ERR_UNKNOWN;
    message = "unknown exception in the generation worker";
  }
  // Released before `finish`, so a caller woken by `vidfab_generation_wait`
  // can start the next run immediately rather than racing this thread's
  // remaining bookkeeping.
  g_generation_active.store(false);
  gen->finish(code, std::move(message));
}

}  // namespace

extern "C" {

VIDFAB_C_API uint32_t VIDFAB_CALL vidfab_capi_version(void) {
  return (static_cast<uint32_t>(VIDFAB_CAPI_VERSION_MAJOR) << 24) |
         (static_cast<uint32_t>(VIDFAB_CAPI_VERSION_MINOR) << 12) |
         static_cast<uint32_t>(VIDFAB_CAPI_VERSION_PATCH);
}

// Built from the same macros rather than written out, so the string and the
// packed number cannot disagree after someone bumps one of them.
#define VIDFAB_CAPI_STRINGIFY_(x) #x
#define VIDFAB_CAPI_STRINGIFY(x) VIDFAB_CAPI_STRINGIFY_(x)

VIDFAB_C_API const char* VIDFAB_CALL vidfab_capi_version_string(void) {
  return VIDFAB_CAPI_STRINGIFY(VIDFAB_CAPI_VERSION_MAJOR) "." VIDFAB_CAPI_STRINGIFY(
      VIDFAB_CAPI_VERSION_MINOR) "." VIDFAB_CAPI_STRINGIFY(VIDFAB_CAPI_VERSION_PATCH);
}

VIDFAB_C_API const char* VIDFAB_CALL vidfab_last_error(void) { return g_last_error.c_str(); }

VIDFAB_C_API void VIDFAB_CALL vidfab_free_string(char* text) { std::free(text); }

VIDFAB_C_API vidfab_request* VIDFAB_CALL vidfab_request_create(void) {
  try {
    auto* request = new vidfab_request();
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

VIDFAB_C_API void VIDFAB_CALL vidfab_request_destroy(vidfab_request* request) { delete request; }

VIDFAB_C_API int VIDFAB_CALL vidfab_request_set_prompt(vidfab_request* request, const char* utf8) {
  if (request == nullptr || utf8 == nullptr) {
    return fail(VIDFAB_ERR_INVALID_ARGUMENT, "vidfab_request_set_prompt: null argument");
  }
  return guarded([&] {
    request->request.prompt = utf8;
    return VIDFAB_OK;
  });
}

VIDFAB_C_API int VIDFAB_CALL vidfab_request_set_prompt_file(vidfab_request* request, const char* path) {
  if (request == nullptr || path == nullptr) {
    return fail(VIDFAB_ERR_INVALID_ARGUMENT, "vidfab_request_set_prompt_file: null argument");
  }
  return guarded([&] {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      return fail(VIDFAB_ERR_NOT_FOUND,
                  std::string("cannot read prompt file '") + path + "'");
    }
    std::ostringstream contents;
    contents << in.rdbuf();
    std::string text = contents.str();
    // The same normalisation `vidfab generate --prompt-file` applies, so a
    // file that works on the CLI conditions identically here: a UTF-8 BOM,
    // CRLF line endings and surrounding blank space are all things an editor
    // adds and no prompt wants in its token stream.
    if (text.rfind("\xEF\xBB\xBF", 0) == 0) text.erase(0, 3);
    text.erase(std::remove(text.begin(), text.end(), '\r'), text.end());
    const size_t first = text.find_first_not_of(" \t\n");
    if (first == std::string::npos) {
      return fail(VIDFAB_ERR_INVALID_REQUEST,
                  std::string("prompt file '") + path + "' has no prompt in it");
    }
    const size_t last = text.find_last_not_of(" \t\n");
    request->request.prompt = text.substr(first, last - first + 1);
    return VIDFAB_OK;
  });
}

VIDFAB_C_API int VIDFAB_CALL vidfab_request_set_aspect(vidfab_request* request, int32_t width, int32_t height) {
  if (request == nullptr || width <= 0 || height <= 0) {
    return fail(VIDFAB_ERR_INVALID_ARGUMENT, "vidfab_request_set_aspect: needs positive extents");
  }
  request->request.aspect_w = width;
  request->request.aspect_h = height;
  request->request.canvas_width = 0;
  request->request.canvas_height = 0;
  return VIDFAB_OK;
}

VIDFAB_C_API int VIDFAB_CALL vidfab_request_set_resolution(vidfab_request* request, int32_t width, int32_t height) {
  if (request == nullptr || width <= 0 || height <= 0) {
    return fail(VIDFAB_ERR_INVALID_ARGUMENT,
                "vidfab_request_set_resolution: needs positive extents");
  }
  request->request.canvas_width = width;
  request->request.canvas_height = height;
  return VIDFAB_OK;
}

VIDFAB_C_API int VIDFAB_CALL vidfab_request_set_frames(vidfab_request* request, int32_t frames) {
  if (request == nullptr || frames <= 0) {
    return fail(VIDFAB_ERR_INVALID_ARGUMENT, "vidfab_request_set_frames: needs a positive count");
  }
  request->request.num_frames = frames;
  return VIDFAB_OK;
}

VIDFAB_C_API int VIDFAB_CALL vidfab_request_set_steps(vidfab_request* request, int32_t steps) {
  if (request == nullptr || steps < 2) {
    return fail(VIDFAB_ERR_INVALID_ARGUMENT,
                "vidfab_request_set_steps: the grid includes a terminal zero, so it needs at "
                "least 2 points");
  }
  request->request.num_inference_steps = steps;
  return VIDFAB_OK;
}

VIDFAB_C_API int VIDFAB_CALL vidfab_request_set_seed(vidfab_request* request, uint64_t seed) {
  if (request == nullptr) {
    return fail(VIDFAB_ERR_INVALID_ARGUMENT, "vidfab_request_set_seed: null request");
  }
  request->request.seed = seed;
  return VIDFAB_OK;
}

VIDFAB_C_API int VIDFAB_CALL vidfab_request_set_model_path(vidfab_request* request, int32_t which, const char* path) {
  if (request == nullptr || path == nullptr) {
    return fail(VIDFAB_ERR_INVALID_ARGUMENT, "vidfab_request_set_model_path: null argument");
  }
  return guarded([&] {
    switch (which) {
      case VIDFAB_MODEL_TRANSFORMER: request->request.transformer_path = path; break;
      case VIDFAB_MODEL_TEXT_ENCODER: request->request.text_encoder_path = path; break;
      case VIDFAB_MODEL_TOKENIZER: request->request.tokenizer_path = path; break;
      case VIDFAB_MODEL_VIDEO_VAE: request->request.video_vae_path = path; break;
      case VIDFAB_MODEL_AUDIO_VAE: request->request.audio_vae_path = path; break;
      default:
        return fail(VIDFAB_ERR_INVALID_ARGUMENT,
                    "vidfab_request_set_model_path: unknown model id " + std::to_string(which));
    }
    return VIDFAB_OK;
  });
}

VIDFAB_C_API int VIDFAB_CALL vidfab_request_add_reference_image(vidfab_request* request, const char* path) {
  if (request == nullptr || path == nullptr) {
    return fail(VIDFAB_ERR_INVALID_ARGUMENT, "vidfab_request_add_reference_image: null argument");
  }
  return guarded([&] {
    if (request->request.reference_image_paths.size() >= 9) {
      return fail(VIDFAB_ERR_INVALID_REQUEST,
                  "MiniMax-H3 Ref2VA accepts at most 9 reference images");
    }
    request->request.reference_image_paths.emplace_back(path);
    return VIDFAB_OK;
  });
}

VIDFAB_C_API int VIDFAB_CALL vidfab_request_set_attention(vidfab_request* request, const char* mode) {
  if (request == nullptr || mode == nullptr) {
    return fail(VIDFAB_ERR_INVALID_ARGUMENT, "vidfab_request_set_attention: null argument");
  }
  // Guarded because the failure message quotes the caller's string, so the
  // reporting path allocates.
  return guarded([&] {
    const std::string name = mode;
    if (name == "none") request->options.attention_mode = vidfab::AttentionMode::kNone;
    else if (name == "flash2") request->options.attention_mode = vidfab::AttentionMode::kFlash2;
    else if (name == "sage2") request->options.attention_mode = vidfab::AttentionMode::kSage2;
    else if (name == "sol") request->options.attention_mode = vidfab::AttentionMode::kSol;
    else if (name == "sol-experimental")
      request->options.attention_mode = vidfab::AttentionMode::kSolExperimental;
    else {
      return fail(VIDFAB_ERR_INVALID_ARGUMENT,
                  "unknown attention mode '" + name +
                      "'; expected none, flash2, sage2, sol or sol-experimental");
    }
    return VIDFAB_OK;
  });
}

VIDFAB_C_API int VIDFAB_CALL vidfab_request_set_synthetic_latents(vidfab_request* request, int32_t enable) {
  if (request == nullptr) {
    return fail(VIDFAB_ERR_INVALID_ARGUMENT, "vidfab_request_set_synthetic_latents: null request");
  }
  request->options.source =
      enable != 0 ? vidfab::LatentSource::kSyntheticNoise : vidfab::LatentSource::kDenoise;
  return VIDFAB_OK;
}

VIDFAB_C_API int VIDFAB_CALL vidfab_request_set_verbose(vidfab_request* request, int32_t enable) {
  if (request == nullptr) {
    return fail(VIDFAB_ERR_INVALID_ARGUMENT, "vidfab_request_set_verbose: null request");
  }
  request->options.verbose = enable != 0;
  return VIDFAB_OK;
}

VIDFAB_C_API int VIDFAB_CALL vidfab_resolve_plan(const vidfab_request* request, vidfab_plan* out_plan) {
  if (request == nullptr || out_plan == nullptr) {
    return fail(VIDFAB_ERR_INVALID_ARGUMENT, "vidfab_resolve_plan: null argument");
  }
  return guarded([&] {
    GeneratePlan plan;
    try {
      plan = vidfab::resolve_plan(request->request);
    } catch (const std::exception& e) {
      // Every throw out of resolve_plan is the request being unsatisfiable —
      // an aspect outside 1:4..4:1, too few frames, a schedule shorter than
      // one step — and none of it is a runtime fault. Reported as such so a
      // caller can tell "you asked for the impossible" from "the machine
      // failed", which are not the same message to a user.
      return fail(VIDFAB_ERR_INVALID_REQUEST, e.what());
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
    return VIDFAB_OK;
  });
}

VIDFAB_C_API int VIDFAB_CALL vidfab_describe_plan(const vidfab_request* request, char** out_text) {
  if (request == nullptr || out_text == nullptr) {
    return fail(VIDFAB_ERR_INVALID_ARGUMENT, "vidfab_describe_plan: null argument");
  }
  *out_text = nullptr;
  return guarded([&] {
    GeneratePlan plan;
    try {
      plan = vidfab::resolve_plan(request->request);
    } catch (const std::exception& e) {
      return fail(VIDFAB_ERR_INVALID_REQUEST, e.what());
    }
    char* text = dup_string(vidfab::describe_plan(request->request, plan));
    if (text == nullptr) return fail(VIDFAB_ERR_OUT_OF_MEMORY, "out of memory");
    *out_text = text;
    return VIDFAB_OK;
  });
}

VIDFAB_C_API int VIDFAB_CALL vidfab_generation_start(const vidfab_request* request, vidfab_progress_fn callback,
                            void* userdata, vidfab_generation** out_generation) {
  if (request == nullptr || out_generation == nullptr) {
    return fail(VIDFAB_ERR_INVALID_ARGUMENT, "vidfab_generation_start: null argument");
  }
  *out_generation = nullptr;
  return guarded([&] {
    // Resolved before the thread exists so an unsatisfiable request is
    // reported synchronously, as a return code, rather than as a handle that
    // fails a millisecond later.
    try {
      (void)vidfab::resolve_plan(request->request);
    } catch (const std::exception& e) {
      return fail(VIDFAB_ERR_INVALID_REQUEST, e.what());
    }
    // Claimed before anything is allocated, so a refused start costs nothing.
    bool expected = false;
    if (!g_generation_active.compare_exchange_strong(expected, true)) {
      return fail(VIDFAB_ERR_BUSY, "a generation is already running in this process");
    }

    // Released by the destructor on every path out of here that is not a
    // successfully launched worker — and there are several, because the
    // allocation and the six string copies below can all throw straight past
    // this frame into `guarded`. Getting that wrong does not fail the call
    // that threw; it strands the flag at true and makes *every* later start
    // return VIDFAB_ERR_BUSY for the life of the process, waiting on a run
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
    std::unique_ptr<vidfab_generation> gen(new vidfab_generation());
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
      return fail(VIDFAB_ERR_RUNTIME, std::string("cannot start a worker thread: ") + e.what());
    }
    // The worker is running: it now owns the claim, and the caller owns the
    // handle. Nothing below here may throw.
    claim.release();
    *out_generation = gen.release();
    return VIDFAB_OK;
  });
}

VIDFAB_C_API void VIDFAB_CALL vidfab_generation_cancel(vidfab_generation* generation) {
  if (generation == nullptr) return;
  generation->cancel.store(true);
}

VIDFAB_C_API int VIDFAB_CALL vidfab_generation_status(const vidfab_generation* generation) {
  if (generation == nullptr) {
    return fail(VIDFAB_ERR_INVALID_ARGUMENT, "vidfab_generation_status: null generation");
  }
  return generation->status.load();
}

VIDFAB_C_API int VIDFAB_CALL vidfab_generation_wait(vidfab_generation* generation, int32_t timeout_ms) {
  if (generation == nullptr) {
    return fail(VIDFAB_ERR_INVALID_ARGUMENT, "vidfab_generation_wait: null generation");
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

VIDFAB_C_API const char* VIDFAB_CALL vidfab_generation_error(const vidfab_generation* generation) {
  if (generation == nullptr) return "";
  // The status load is what makes the read below safe, so it is a guard and
  // not an optimisation. `finish` assigns `error` and *then* stores a terminal
  // status; observing that status here synchronises with the assignment, and
  // after it nothing writes `error` again.
  //
  // Reading unconditionally is what the obvious version does, and it is a
  // crash rather than a torn read: a host polling this beside
  // `vidfab_generation_status` in a UI loop would be calling `c_str()` while
  // the worker move-assigns the string underneath it, and the SSO-to-heap
  // transition hands back a pointer into a buffer being freed.
  if (generation->status.load() == VIDFAB_ERR_NOT_READY) return "";
  return generation->error.c_str();
}

VIDFAB_C_API int VIDFAB_CALL vidfab_generation_output(const vidfab_generation* generation, vidfab_output* out_output) {
  if (generation == nullptr || out_output == nullptr) {
    return fail(VIDFAB_ERR_INVALID_ARGUMENT, "vidfab_generation_output: null argument");
  }
  const int status = report_terminal_status(generation);
  if (status != VIDFAB_OK) return status;
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
  return VIDFAB_OK;
}

VIDFAB_C_API int VIDFAB_CALL vidfab_generation_frame_rgba8(const vidfab_generation* generation, int32_t frame_index,
                                  uint8_t* dst, size_t dst_bytes) {
  if (generation == nullptr || dst == nullptr) {
    return fail(VIDFAB_ERR_INVALID_ARGUMENT, "vidfab_generation_frame_rgba8: null argument");
  }
  // Reports what actually happened rather than a blanket "not ready", so that
  // this and `vidfab_generation_output` cannot tell a caller two different
  // stories about the same handle.
  const int status = report_terminal_status(generation);
  if (status != VIDFAB_OK) return status;

  // Guarded from here on: every check below names the offending number in its
  // message, so the reporting path allocates.
  return guarded([&] {
    if (frame_index < 0 || frame_index >= generation->frames) {
      return fail(VIDFAB_ERR_INVALID_ARGUMENT,
                  "frame " + std::to_string(frame_index) + " is outside 0.." +
                      std::to_string(generation->frames - 1));
    }
    const size_t pixels = static_cast<size_t>(generation->width) * generation->height;
    if (dst_bytes < pixels * 4) {
      return fail(VIDFAB_ERR_INVALID_ARGUMENT, "the destination holds " +
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
      return fail(VIDFAB_ERR_RUNTIME,
                  "frame_rgba8 needs 3 channels, got " + std::to_string(generation->channels));
    }
    if (generation->video.size() < 3 * pixels * static_cast<size_t>(generation->frames)) {
      return fail(VIDFAB_ERR_RUNTIME, "the decoded video buffer is shorter than its own geometry");
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
    return VIDFAB_OK;
  });
}

VIDFAB_C_API void VIDFAB_CALL vidfab_generation_destroy(vidfab_generation* generation) {
  if (generation == nullptr) return;
  generation->cancel.store(true);
  // Joined rather than detached. A detached worker would outlive the pixels
  // it is about to write into, and the caller has no way to know when the
  // last CUDA call has drained.
  if (generation->worker.joinable()) generation->worker.join();
  delete generation;
}

}  // extern "C"
