#include "internal.h"
extern "C" {
SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_image_edit_path(slopfab_request* request,
                                                                   const char* path, int32_t x,
                                                                   int32_t y, int32_t width,
                                                                   int32_t height, float strength,
                                                                   int32_t feather) {
  if (!request || !path || !*path)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "image edit: null request or empty path");
  return reference_input_guarded([&] {
    slopfab::ImageEdit edit{
        std::make_shared<const slopfab::RGBImage>(slopfab::load_reference_image(path)),
        x,
        y,
        width,
        height,
        strength,
        feather};
    edit.validate();
    request->request.image_edit = std::move(edit);
    request->request.still_image = true;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_image_edit_rgb24(
    slopfab_request* request, const uint8_t* pixels, size_t buffer_bytes, int32_t image_width,
    int32_t image_height, size_t row_stride_bytes, int32_t x, int32_t y, int32_t width,
    int32_t height, float strength, int32_t feather) {
  if (!request)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "image edit: null request");
  return reference_input_guarded([&] {
    auto media = slopfab::ReferenceMedia::video(2);
    media.append_frame(pixels, buffer_bytes, image_width, image_height, row_stride_bytes, 3, 0);
    slopfab::ImageEdit edit{
        std::make_shared<const slopfab::RGBImage>(media.frames().front()->image),
        x,
        y,
        width,
        height,
        strength,
        feather};
    edit.validate();
    request->request.image_edit = std::move(edit);
    request->request.still_image = true;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_clear_image_edit(slopfab_request* request) {
  if (!request)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "image edit: null request");
  request->request.image_edit = {};
  return SLOPFAB_OK;
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

SLOPFAB_C_API void SLOPFAB_CALL slopfab_request_destroy(slopfab_request* request) {
  delete request;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_prompt(slopfab_request* request,
                                                          const char* utf8) {
  if (request == nullptr || utf8 == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_request_set_prompt: null argument");
  }
  return guarded([&] {
    request->request.prompt = utf8;
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_prompt_file(slopfab_request* request,
                                                               const char* path) {
  if (request == nullptr || path == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_request_set_prompt_file: null argument");
  }
  return guarded([&] {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
      return fail(SLOPFAB_ERR_NOT_FOUND, std::string("cannot read prompt file '") + path + "'");
    }
    std::ostringstream contents;
    contents << in.rdbuf();
    std::string text = contents.str();
    // The same normalisation `slopfab generate --prompt-file` applies, so a
    // file that works on the CLI conditions identically here: a UTF-8 BOM,
    // CRLF line endings and surrounding blank space are all things an editor
    // adds and no prompt wants in its token stream.
    if (text.rfind("\xEF\xBB\xBF", 0) == 0)
      text.erase(0, 3);
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

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_aspect(slopfab_request* request, int32_t width,
                                                          int32_t height) {
  if (request == nullptr || width <= 0 || height <= 0) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_request_set_aspect: needs positive extents");
  }
  request->request.aspect_w = width;
  request->request.aspect_h = height;
  request->request.canvas_width = 0;
  request->request.canvas_height = 0;
  return SLOPFAB_OK;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_resolution(slopfab_request* request,
                                                              int32_t width, int32_t height) {
  if (request == nullptr || width <= 0 || height <= 0) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                "slopfab_request_set_resolution: needs positive extents");
  }
  request->request.canvas_width = width;
  request->request.canvas_height = height;
  return SLOPFAB_OK;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_frames(slopfab_request* request,
                                                          int32_t frames) {
  if (request == nullptr || frames <= 0) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_request_set_frames: needs a positive count");
  }
  request->request.num_frames = frames;
  return SLOPFAB_OK;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_still_image(slopfab_request* request,
                                                               int32_t enable) {
  if (request == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_request_set_still_image: null request");
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

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_model_path(slopfab_request* request,
                                                              int32_t which, const char* path) {
  if (request == nullptr || path == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_request_set_model_path: null argument");
  }
  return guarded([&] {
    switch (which) {
    case SLOPFAB_MODEL_TRANSFORMER:
      request->request.transformer_path = path;
      break;
    case SLOPFAB_MODEL_TEXT_ENCODER:
      request->request.text_encoder_path = path;
      break;
    case SLOPFAB_MODEL_TOKENIZER:
      request->request.tokenizer_path = path;
      break;
    case SLOPFAB_MODEL_VIDEO_VAE:
      request->request.video_vae_path = path;
      break;
    case SLOPFAB_MODEL_AUDIO_VAE:
      request->request.audio_vae_path = path;
      break;
    default:
      return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                  "slopfab_request_set_model_path: unknown model id " + std::to_string(which));
    }
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_prompt_embedding_path(slopfab_request* request,
                                                                         const char* path) {
  if (request == nullptr || path == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                "slopfab_request_set_prompt_embedding_path: null argument");
  }
  return guarded([&] {
    request->options.prompt_embedding_path = path;
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_animate(slopfab_request* request, int32_t enable,
                                                           int32_t preserve_driving_audio) {
  if (!request)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "set_animate: null request");
  request->request.animate = enable != 0;
  request->request.preserve_driving_audio = enable != 0 && preserve_driving_audio != 0;
  return SLOPFAB_OK;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_add_lora(slopfab_request* request, const char* path,
                                                        float strength) {
  if (!request || !path || !*path || !std::isfinite(strength))
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                "LoRA needs a request, nonempty path and finite strength");
  return guarded([&] {
    request->request.loras.push_back({path, strength});
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_clear_loras(slopfab_request* request) {
  if (!request)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "clear_loras: null request");
  return guarded([&] {
    request->request.loras.clear();
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_schedule(slopfab_request* request,
                                                            int32_t schedule) {
  if (!request ||
      (schedule != SLOPFAB_SCHEDULE_DEFAULT && schedule != SLOPFAB_SCHEDULE_TAOMATE_3STEP))
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "unknown denoising schedule or null request");
  return guarded([&] {
    request->request.schedule = schedule == SLOPFAB_SCHEDULE_DEFAULT
                                    ? slopfab::sampler::ScheduleKind::kDefault
                                    : slopfab::sampler::ScheduleKind::kTaoMate3Step;
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_sampling_settings(slopfab_request* request,
                                                                     const char* json) {
  if (!request)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "sampling settings: null request");
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

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_conditioning_settings(slopfab_request* request,
                                                                         const char* json) {
  if (!request)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "conditioning settings: null request");
  return guarded([&] {
    try {
      auto settings =
          json ? slopfab::parse_conditioning_settings(json) : slopfab::ConditioningSettings{};
      request->request.conditioning = std::move(settings);
    } catch (const std::exception& e) {
      return fail(SLOPFAB_ERR_INVALID_ARGUMENT, e.what());
    }
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_motion_cache(
    slopfab_request* request, int32_t enabled, float reuse_threshold, float motion_strength,
    int32_t warmup_steps, int32_t max_consecutive_skips, float start_percent, float end_percent,
    int32_t subsample_factor, int32_t verbose) {
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
    try {
      config.validate();
    } catch (const std::invalid_argument& e) {
      return fail(SLOPFAB_ERR_INVALID_ARGUMENT, e.what());
    }
    request->request.motion_cache = config;
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_add_refmod(slopfab_request* request,
                                                          const char* path, float strength,
                                                          int32_t copies) {
  if (!request || !path || !*path || !std::isfinite(strength) || strength < 0 || strength > 1 ||
      copies < 1 || copies > 10)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                "refmod needs a request, path, strength 0..1 and copies 1..10");
  return guarded([&] {
    auto refs = request->request.refmods;
    refs.push_back({slopfab::RefMod::load(path), strength, copies});
    slopfab::validate_refmods(refs);
    request->request.refmods.swap(refs);
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_save_latents(slopfab_request* request,
                                                                const char* path) {
  if (!request)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "save latents: null request");
  return guarded([&] {
    request->options.save_latents_path = path ? path : "";
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_retain_latents(slopfab_request* request,
                                                                  int32_t enable) {
  if (!request)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "retain latents: null request");
  return guarded([&] {
    request->options.on_latents = enable ? &latents_hook : nullptr;
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_continuation_file(slopfab_request* request,
                                                                     const char* path,
                                                                     int32_t overlap_frames) {
  if (!request || !path || !*path || overlap_frames < 5 || overlap_frames % 17 != 5)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                "continuation needs a request, archive path and 17*k+5 overlap >=5");
  return guarded([&] {
    auto clip = slopfab::LatentClip::load(path);
    try {
      (void)slopfab::plan_continuation(*clip, overlap_frames, 17);
    } catch (const std::exception& e) {
      return fail(SLOPFAB_ERR_INVALID_REQUEST, e.what());
    }
    request->request.continuation = std::move(clip);
    request->request.continuation_overlap_frames = overlap_frames;
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_continuation_generation(
    slopfab_request* request, const slopfab_generation* source, int32_t overlap_frames) {
  if (!request || !source || overlap_frames < 5 || overlap_frames % 17 != 5)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                "continuation needs a request, source generation and 17*k+5 overlap >=5");
  const int terminal = report_terminal_status(source);
  if (terminal != SLOPFAB_OK)
    return terminal;
  return guarded([&] {
    if (!source->latents)
      return fail(SLOPFAB_ERR_INVALID_REQUEST,
                  "enable latent retention before starting the source generation");
    try {
      (void)slopfab::plan_continuation(*source->latents, overlap_frames, 17);
    } catch (const std::exception& e) {
      return fail(SLOPFAB_ERR_INVALID_REQUEST, e.what());
    }
    request->request.continuation = source->latents;
    request->request.continuation_overlap_frames = overlap_frames;
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_clear_continuation(slopfab_request* request) {
  if (!request)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "clear continuation: null request");
  return guarded([&] {
    request->request.continuation.reset();
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_video_transition(slopfab_request* request,
                                                                    int32_t mode) {
  if (!request || mode < 0 || mode > 2)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                "video transition requires a request and mode 0, 1 or 2");
  request->request.video_transition = mode;
  return SLOPFAB_OK;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_generation_save_latents(const slopfab_generation* generation,
                                                               const char* path) {
  if (!generation || !path || !*path)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "save latents needs a generation and path");
  const int terminal = report_terminal_status(generation);
  if (terminal != SLOPFAB_OK)
    return terminal;
  return guarded([&] {
    if (!generation->latents)
      return fail(SLOPFAB_ERR_INVALID_REQUEST,
                  "enable latent retention before starting the generation");
    generation->latents->save(path);
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_clear_refmods(slopfab_request* request) {
  if (!request)
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "clear_refmods: null request");
  return guarded([&] {
    request->request.refmods.clear();
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_add_reference_image(slopfab_request* request,
                                                                   const char* path) {
  if (request == nullptr || path == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_request_add_reference_image: null argument");
  }
  return guarded([&] {
    if (request->request.reference_image_paths.size() >= 9) {
      return fail(SLOPFAB_ERR_INVALID_REQUEST,
                  "MiniMax-H3 Ref2VA accepts at most 9 reference images");
    }
    if (request->request.reference_image_paths.size() + request->request.reference_media.size() >=
        12) {
      return fail(SLOPFAB_ERR_INVALID_REQUEST,
                  "MiniMax-H3 Ref2VA accepts at most 12 references in total");
    }
    request->request.reference_image_paths.emplace_back(path);
    return SLOPFAB_OK;
  });
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_attention(slopfab_request* request,
                                                             const char* mode) {
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

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_inference_backend(slopfab_request* request,
                                                                     int32_t backend) {
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

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_synthetic_latents(slopfab_request* request,
                                                                     int32_t enable) {
  if (request == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT,
                "slopfab_request_set_synthetic_latents: null request");
  }
  request->options.source =
      enable != 0 ? slopfab::LatentSource::kSyntheticNoise : slopfab::LatentSource::kDenoise;
  return SLOPFAB_OK;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_verbose(slopfab_request* request,
                                                           int32_t enable) {
  if (request == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_request_set_verbose: null request");
  }
  request->options.verbose = enable != 0;
  return SLOPFAB_OK;
}

SLOPFAB_C_API int SLOPFAB_CALL slopfab_request_set_reuse_models(slopfab_request* request,
                                                                int32_t enable) {
  if (request == nullptr) {
    return fail(SLOPFAB_ERR_INVALID_ARGUMENT, "slopfab_request_set_reuse_models: null request");
  }
  request->options.reuse_models = enable != 0;
  return SLOPFAB_OK;
}
}
