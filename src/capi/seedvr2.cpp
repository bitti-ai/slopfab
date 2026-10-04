#include "internal.h"
#include "slopfab/seedvr2.h"

extern "C" SLOPFAB_C_API int SLOPFAB_CALL slopfab_seedvr2_upscale(
    const slopfab_seedvr2_options* options, slopfab_seedvr2_read_fn read_frame,
    slopfab_seedvr2_write_fn write_frame, slopfab_seedvr2_cancel_fn cancelled,
    slopfab_seedvr2_progress_fn progress, void* user, uint64_t* frames_written) {
  if (frames_written)
    *frames_written = 0;
  if (!options || options->struct_size != sizeof(slopfab_seedvr2_options) || !read_frame ||
      !write_frame || !frames_written || !options->transformer_path ||
      !*options->transformer_path || !options->vae_path || !*options->vae_path ||
      (options->color_match != 0 && options->color_match != 1))
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "seedvr2_upscale: invalid options or callbacks");
  return guarded([&] {
    slopfab::seedvr2::Options o;
    o.transformer = options->transformer_path;
    o.vae = options->vae_path;
    o.width = options->width;
    o.height = options->height;
    o.segment_frames = options->segment_frames;
    o.vae_tile = options->vae_tile;
    o.device = options->device;
    o.color_match = options->color_match != 0;
    o.seed = options->seed;
    try {
      slopfab::seedvr2::validate(o);
    } catch (const std::invalid_argument& e) {
      return fail(SLOPFAB_ERR_INVALID_ARGUMENT, e.what());
    }
#if SLOPFAB_WITH_CUDA
    bool stopped = false;
    auto cancel = [&] {
      stopped = stopped || (cancelled && cancelled(user));
      return stopped;
    };
    if (cancel())
      return fail(SLOPFAB_ERR_CANCELLED, "SeedVR2: cancelled");
    slopfab::seedvr2::Restorer model(o);
    model.cancelled = cancel;
    if (progress)
      model.progress = [&](const std::string& s) {
        progress(user, s.c_str());
      };
    try {
      slopfab::seedvr2::stream(
          o,
          [&](slopfab::seedvr2::Frame& frame) {
            if (cancel())
              throw std::runtime_error("SeedVR2: cancelled");
            frame.resize(size_t(o.width) * o.height * 3);
            int result = read_frame(user, frame.data(), frame.size());
            if (result != 0 && result != 1)
              throw std::runtime_error("SeedVR2: input callback failed");
            return result == 1;
          },
          [&](const slopfab::seedvr2::Frame& frame) {
            if (cancel())
              throw std::runtime_error("SeedVR2: cancelled");
            if (write_frame(user, frame.data(), frame.size()))
              throw std::runtime_error("SeedVR2: output callback failed");
            ++*frames_written;
          },
          [&](const std::vector<slopfab::seedvr2::Frame>& frames, uint64_t first) {
            return model.restore(frames, first);
          });
    } catch (...) {
      if (stopped)
        return fail(SLOPFAB_ERR_CANCELLED, "SeedVR2: cancelled");
      throw;
    }
    return SLOPFAB_OK;
#else
    (void)cancelled;
    (void)progress;
    (void)user;
    return fail(SLOPFAB_ERR_RUNTIME, "SeedVR2 requires a CUDA-enabled build");
#endif
  });
}
