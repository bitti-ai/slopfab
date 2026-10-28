#include "internal.h"

namespace {
int validate(const char* checkpoint, float scale, int chunking, int backend,
             slopfab_generation** output) {
  if (output)
    *output = nullptr;
  if (!output || !checkpoint || !*checkpoint || !std::isfinite(scale) || scale < 1 || scale > 4 ||
      (chunking != 0 && chunking != 1) ||
      (backend != SLOPFAB_INFERENCE_CUDA && backend != SLOPFAB_INFERENCE_VULKAN))
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "latent upscale: invalid output, checkpoint, scale, chunking or backend");
#if !SLOPFAB_WITH_CUDA
  if (backend == SLOPFAB_INFERENCE_CUDA)
    return fail(SLOPFAB_ERR_UNSUPPORTED, "CUDA backend not compiled in");
#endif
#if !SLOPFAB_WITH_VULKAN
  if (backend == SLOPFAB_INFERENCE_VULKAN)
    return fail(SLOPFAB_ERR_UNSUPPORTED, "Vulkan backend not compiled in");
#endif
  return SLOPFAB_OK;
}

int launch(std::shared_ptr<const slopfab::LatentClip> source, const char* checkpoint,
           float scale, int chunking, int backend, slopfab_progress_fn callback,
           void* userdata, slopfab_generation** output,
           const char* video_vae = nullptr, const char* audio_vae = nullptr) {
  if (!source || !source->sampled)
    return fail(SLOPFAB_ERR_INVALID_REQUEST, "latent upscale requires retained completed sampled latents");
  source->validate();
  const auto dims = slopfab::latent_upscale_dimensions(source->height / 16, source->width / 16,
                                                      {scale, chunking != 0});
  slopfab::LatentClip shape;
  shape.width = dims.second * 16;
  shape.height = dims.first * 16;
  shape.frames = source->frames;
  shape.geometry = source->geometry;
  (void)shape.layout();
  bool expected = false;
  if (!g_generation_active.compare_exchange_strong(expected, true))
    return fail(SLOPFAB_ERR_BUSY, "a generation or latent upscale is already running");
  struct Claim {
    bool held = true;
    ~Claim() { if (held) g_generation_active.store(false); }
  } claim;
  auto gen = std::make_unique<slopfab_generation>();
  gen->fps = 24.0;
  gen->request.still_image = source->frames == 1;
  gen->latent_upscale_source = std::move(source);
  gen->latents_only = !video_vae;
  if (video_vae) gen->request.video_vae_path = video_vae;
  if (audio_vae) gen->request.audio_vae_path = audio_vae;
  gen->options.latent_upscale_model_path = checkpoint;
  gen->options.latent_upscale = {scale, chunking != 0};
  gen->options.inference_backend = static_cast<slopfab::DeviceBackend>(backend);
  gen->callback = callback;
  gen->userdata = userdata;
  gen->started = std::chrono::steady_clock::now();
  gen->worker = std::thread(&run_worker, gen.get());
  claim.held = false;
  *output = gen.release();
  return SLOPFAB_OK;
}
}

extern "C" {
SLOPFAB_C_API int SLOPFAB_CALL slopfab_latent_upscale_decode_file_start(
    const char* source_path, const char* checkpoint, float scale,
    const char* video_vae, const char* audio_vae, int32_t backend,
    slopfab_progress_fn callback, void* userdata, slopfab_generation** output) {
  const int valid = validate(checkpoint, scale, 1, backend, output);
  if (valid != SLOPFAB_OK) return valid;
  if (!source_path || !*source_path || !video_vae || !*video_vae)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "latent upscale decode: source and video VAE are required");
  return guarded([&] {
    return launch(slopfab::LatentClip::load(source_path), checkpoint, scale, 1,
                  backend, callback, userdata, output, video_vae, audio_vae);
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_latent_upscale_file_start(
    const char* source_path, const char* checkpoint, float scale, int32_t chunking,
    int32_t backend, slopfab_progress_fn callback, void* userdata,
    slopfab_generation** output) {
  const int valid = validate(checkpoint, scale, chunking, backend, output);
  if (valid != SLOPFAB_OK)
    return valid;
  if (!source_path || !*source_path)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "latent upscale: empty source path");
  return guarded([&] {
    return launch(slopfab::LatentClip::load(source_path), checkpoint, scale, chunking,
                  backend, callback, userdata, output);
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_latent_upscale_generation_start(
    const slopfab_generation* source, const char* checkpoint, float scale,
    int32_t chunking, int32_t backend, slopfab_progress_fn callback,
    void* userdata, slopfab_generation** output) {
  const int valid = validate(checkpoint, scale, chunking, backend, output);
  if (valid != SLOPFAB_OK)
    return valid;
  if (!source)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "latent upscale: null source generation");
  const int status = report_terminal_status(source);
  if (status != SLOPFAB_OK)
    return status;
  return guarded([&] {
    return launch(source->latents, checkpoint, scale, chunking, backend, callback, userdata, output);
  });
}
}
