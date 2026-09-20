#pragma once
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

namespace slopfab::capi {

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
inline thread_local std::string g_last_error;

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
inline int fail(int code, const char* message) noexcept {
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
inline int fail(int code, std::string message) {
  g_last_error = std::move(message);
  return code;
}

inline bool contains_ci(const std::string& haystack, const char* needle) {
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
inline int classify(const std::string& message) {
  if (contains_ci(message, "out of memory") || contains_ci(message, "bad allocation") ||
      contains_ci(message, "cudaerrormemoryallocation")) {
    return SLOPFAB_ERR_OUT_OF_MEMORY;
  }
  return SLOPFAB_ERR_RUNTIME;
}

// The wrapper every entry point uses. A lambda returning int keeps the
// try/catch in one place instead of in thirty.
template <typename Fn>
inline int guarded(Fn&& body) {
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
inline char* dup_string(const std::string& text) {
  char* out = static_cast<char*>(std::malloc(text.size() + 1));
  if (out == nullptr) return nullptr;
  std::memcpy(out, text.c_str(), text.size() + 1);
  return out;
}

}  // namespace slopfab::capi
using namespace slopfab::capi;

// The handle types are declared at namespace scope because the header names
// them, and defined here so nothing about their contents is visible to a
// caller.
struct slopfab_request {
  std::shared_ptr<slopfab::GenerationSession> session;
  GenerateRequest request;
  RunOptions options;
};

struct slopfab_reference_video {
  slopfab::ReferenceMedia media;
};
struct slopfab_session {
  std::shared_ptr<slopfab::GenerationSession> value;
};

namespace slopfab::capi {

template <typename Fn>
inline int reference_input_guarded(Fn&& body) {
  return guarded([&] {
    try { body(); }
    catch (const std::invalid_argument& e) {
      return fail(SLOPFAB_ERR_INVALID_ARGUMENT, e.what());
    }
    return SLOPFAB_OK;
  });
}

inline void attach_reference(slopfab_request* request, const slopfab::ReferenceMedia& media) {
  auto references = request->request.reference_media;
  references.push_back(std::make_shared<const slopfab::ReferenceMedia>(media));
  slopfab::validate_reference_media(request->request.reference_image_paths.size(), references);
  request->request.reference_media.swap(references);
}

}  // namespace slopfab::capi
using namespace slopfab::capi;

struct slopfab_generation {
  std::shared_ptr<slopfab::GenerationSession> session;
  GenerateRequest request;
  GeneratePlan plan;
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

namespace slopfab::capi {

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
inline std::atomic<bool> g_generation_active{false};

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
inline int report_terminal_status(const slopfab_generation* generation) noexcept {
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
inline bool progress_hook(RunStage stage, int step, int steps, void* userdata) {
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

inline bool samples_hook(RunSamples& samples, void* userdata) {
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

inline void latents_hook(const std::shared_ptr<const slopfab::LatentClip>& latents, void* userdata) {
  static_cast<slopfab_generation*>(userdata)->latents = latents;
}

inline void run_worker(slopfab_generation* gen) {
  int code = SLOPFAB_ERR_UNKNOWN;
  std::string message;
  try {
    const GeneratePlan& plan = gen->plan;
    if (plan.duration_seconds > 0.0) {
      gen->fps = static_cast<double>(plan.aligned_frames) / plan.duration_seconds;
    }
    RunOptions options = gen->options;
    options.on_progress = &progress_hook;
    options.on_samples = &samples_hook;
    options.hook_userdata = gen;
    const RunResult result = gen->session
        ? slopfab::run_generate(*gen->session, gen->request, plan, options)
        : slopfab::run_generate(gen->request, plan, options);
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

}  // namespace slopfab::capi
using namespace slopfab::capi;

