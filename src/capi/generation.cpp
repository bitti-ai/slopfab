#include "internal.h"
extern "C" {
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
    GeneratePlan resolved_plan;
    try {
      if (request->request.continuation && request->options.source != slopfab::LatentSource::kDenoise)
        throw std::invalid_argument("continuation requires denoising");
      resolved_plan = slopfab::resolve_plan(request->request);
      slopfab::validate_generation_options(request->request, resolved_plan, request->options);
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
    gen->plan = std::move(resolved_plan);
    gen->session = request->session;
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


}
