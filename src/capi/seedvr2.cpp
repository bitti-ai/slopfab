#include "internal.h"
#include "slopfab/seedvr2.h"

extern "C" SLOPFAB_C_API int SLOPFAB_CALL slopfab_realesrgan_upscale(
    const char* model_path, int32_t width, int32_t height, int32_t backend,
    int32_t tile_size, slopfab_seedvr2_read_fn read_frame,
    slopfab_seedvr2_write_fn write_frame, slopfab_seedvr2_cancel_fn cancelled,
    void* user, uint64_t* frames_written) {
  if (frames_written) *frames_written = 0;
  if (!model_path || !*model_path || width < 1 || height < 1 || width > 4096 ||
      height > 4096 || (backend != 0 && backend != 1) || tile_size < 0 ||
      !read_frame || !write_frame || !frames_written)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "realesrgan_upscale: invalid arguments");
  return guarded([&] {
    const auto stop = [&] { return cancelled && cancelled(user); };
    if (stop()) return fail(SLOPFAB_ERR_CANCELLED, "Real-ESRGAN: cancelled");
    slopfab::RealEsrgan model(model_path, backend == 0 ? slopfab::DeviceBackend::kCuda
                                                    : slopfab::DeviceBackend::kVulkan);
    slopfab::UpscaleOptions options;
    options.tile_size = tile_size;
    const size_t pixels = size_t(width) * height;
    std::vector<float> packed(pixels * 3);
    slopfab::PixelBuffer planar(pixels * 3);
    for (;;) {
      if (stop()) return fail(SLOPFAB_ERR_CANCELLED, "Real-ESRGAN: cancelled");
      const int read = read_frame(user, packed.data(), packed.size());
      if (read == 0) return SLOPFAB_OK;
      if (read != 1) return fail(SLOPFAB_ERR_RUNTIME, "Real-ESRGAN: input callback failed");
      for (size_t p = 0; p < pixels; ++p)
        for (size_t c = 0; c < 3; ++c) planar[c * pixels + p] = packed[p * 3 + c];
      slopfab::PixelBuffer restored;
      try {
        restored = model.upscale(planar, 1, height, width, options,
                                 [&](int, int) { return !stop(); });
      } catch (const slopfab::UpscaleCancelled&) {
        return fail(SLOPFAB_ERR_CANCELLED, "Real-ESRGAN: cancelled");
      }
      const size_t output_pixels = pixels * 16;
      std::vector<float> output(output_pixels * 3);
      for (size_t p = 0; p < output_pixels; ++p)
        for (size_t c = 0; c < 3; ++c) output[p * 3 + c] = restored[c * output_pixels + p];
      if (stop()) return fail(SLOPFAB_ERR_CANCELLED, "Real-ESRGAN: cancelled");
      if (write_frame(user, output.data(), output.size()))
        return fail(SLOPFAB_ERR_RUNTIME, "Real-ESRGAN: output callback failed");
      ++*frames_written;
    }
  });
}

extern "C" SLOPFAB_C_API int SLOPFAB_CALL
slopfab_request_set_seedvr2_options(slopfab_request* request, const slopfab_seedvr2_options* o) {
  if (!request || !o || o->struct_size != sizeof(*o) || !o->transformer_path ||
      !*o->transformer_path || !o->vae_path || !*o->vae_path ||
      (o->color_match != 0 && o->color_match != 1) || o->width < 16 || o->height < 16)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "set_seedvr2_options: invalid arguments");
  return guarded([&] {
    slopfab::UpscaleOptions options;
    options.width = o->width;
    options.height = o->height;
    options.tile_size = o->vae_tile;
    options.segment_frames = o->segment_frames;
    options.vae_path = o->vae_path;
    options.seed = o->seed;
    options.device = o->device;
    options.color_match = o->color_match != 0;
    try {
      slopfab::validate_upscale_options(options, slopfab::UpscaleMethod::kSeedVr2);
    } catch (const std::invalid_argument& e) {
      return fail(SLOPFAB_ERR_INVALID_ARGUMENT, e.what());
    }
    request->options.upscale_model_path = o->transformer_path;
    request->options.upscale_method = slopfab::UpscaleMethod::kSeedVr2;
    request->options.upscale = std::move(options);
    return SLOPFAB_OK;
  });
}

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
