#include "internal.h"
extern "C" {
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
      slopfab::validate_generation_options(request->request, plan, request->options);
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
      slopfab::validate_generation_options(request->request, plan, request->options);
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


}
