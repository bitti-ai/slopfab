#include "harness.h"
#include "slopfab/generate.h"

#include <memory>
#include <string>

namespace {
slopfab::GenerateRequest small_request() {
  slopfab::GenerateRequest request;
  request.prompt = "backend smoke test";
  request.canvas_width = 256;
  request.canvas_height = 256;
  request.num_frames = 22;
  request.num_inference_steps = 4;
  return request;
}

struct Capture {
  std::shared_ptr<const slopfab::LatentClip> latents;
  bool started = false;
  bool reached_decode = false;
};
}

SLOPFAB_TEST(generation_compiled_backend_default) {
  const slopfab::RunOptions options;
#if SLOPFAB_WITH_CUDA
  CHECK(options.inference_backend == slopfab::DeviceBackend::kCuda);
  CHECK(options.attention_mode == slopfab::AttentionMode::kFlash2);
#else
  CHECK(options.inference_backend == slopfab::DeviceBackend::kVulkan);
  CHECK(options.attention_mode == slopfab::AttentionMode::kExact);
#endif
}

// This executes the shared session, sampler and latent handoff on each compiled
// backend. Cancellation before VAE loading needs neither a GPU nor model files.
SLOPFAB_TEST(generation_synthetic_pipeline_without_gpu) {
  using namespace slopfab;
  const auto request = small_request();
  const auto plan = resolve_plan(request);
  GenerationSession session;
  for (auto backend : {DeviceBackend::kCuda, DeviceBackend::kVulkan}) {
#if !SLOPFAB_WITH_CUDA
    if (backend == DeviceBackend::kCuda)
      continue;
#endif
#if !SLOPFAB_WITH_VULKAN
    if (backend == DeviceBackend::kVulkan)
      continue;
#endif
    Capture capture;
    RunOptions options;
    options.inference_backend = backend;
    options.attention_mode = AttentionMode::kExact;
    options.source = LatentSource::kSyntheticNoise;
    options.verbose = false;
    options.hook_userdata = &capture;
    options.on_latents = [](const std::shared_ptr<const LatentClip>& clip, void* userdata) {
      static_cast<Capture*>(userdata)->latents = clip;
    };
    options.on_progress = [](RunStage stage, int, int, void* userdata) {
      auto& capture = *static_cast<Capture*>(userdata);
      capture.started |= stage == RunStage::kStarting;
      capture.reached_decode |= stage == RunStage::kVideoDecode;
      return stage != RunStage::kVideoDecode;
    };
    const auto result = run_generate(session, request, plan, options);
    CHECK(!result.ok && result.cancelled);
    CHECK(result.message == "cancelled during video decode");
    CHECK(capture.started && capture.reached_decode);
    CHECK(capture.latents != nullptr);
    CHECK(capture.latents->width == 256 && capture.latents->height == 256);
    CHECK(!capture.latents->sampled);
    CHECK(!capture.latents->video_rows.empty());
    CHECK(!capture.latents->audio_rows.empty());
    session.clear();
  }
}

SLOPFAB_TEST(generation_unavailable_backend_rejected_before_start) {
  using namespace slopfab;
  const auto request = small_request();
  const auto plan = resolve_plan(request);
  for (auto backend : {DeviceBackend::kCuda, DeviceBackend::kVulkan}) {
#if SLOPFAB_WITH_CUDA
    if (backend == DeviceBackend::kCuda)
      continue;
#endif
#if SLOPFAB_WITH_VULKAN
    if (backend == DeviceBackend::kVulkan)
      continue;
#endif
    bool called = false;
    RunOptions options;
    options.inference_backend = backend;
    options.attention_mode = AttentionMode::kExact;
    options.source = LatentSource::kSyntheticNoise;
    options.verbose = false;
    options.hook_userdata = &called;
    options.on_progress = [](RunStage, int, int, void* userdata) {
      *static_cast<bool*>(userdata) = true;
      return true;
    };
    const auto result = run_generate(request, plan, options);
    CHECK(!result.ok && !result.cancelled);
    CHECK(result.message.find("this build disabled") != std::string::npos);
    CHECK(!called);
  }
}
