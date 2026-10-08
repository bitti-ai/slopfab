#include "harness.h"
#include "slopfab/latent_upscale.h"
#include "slopfab/generate.h"
#include "slopfab/upscale.h"
#include "slopfab/tensor_convert.h"
#include <cstdlib>
#include <filesystem>
#include <limits>

using namespace slopfab;

namespace {
template <class F> bool rejects(F f) {
  try {
    f();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}
}

SLOPFAB_TEST(latent_upscale_validation) {
  CHECK(latent_upscale_dimensions(16, 32) == std::make_pair(32, 64));
  CHECK(latent_upscale_dimensions(2, 6, {2.5f}) == std::make_pair(4, 16));
  for (float s : {0.0f, .9f, 4.1f, std::numeric_limits<float>::infinity(),
                  std::numeric_limits<float>::quiet_NaN()})
    CHECK(rejects([&] {
      latent_upscale_dimensions(16, 16, {s});
    }));
  CHECK(rejects([] {
    latent_upscale_dimensions(0, 2);
  }));
  CHECK(rejects([] {
    latent_upscale_dimensions(INT32_MAX, 2);
  }));
  std::vector<float> x(24 * 2 * 4, .2f);
  CHECK(upscale_latents(x, 1, 2, 4, "missing", DeviceBackend::kCuda, {1}) == x);
  CHECK(rejects([&] {
    upscale_latents(x, 2, 2, 4, "missing", DeviceBackend::kCuda);
  }));
  bool cancelled = false;
  try {
    upscale_latents(x, 1, 2, 4, "missing", DeviceBackend::kCuda, {}, [](int, int) {
      return false;
    });
  } catch (const UpscaleCancelled&) {
    cancelled = true;
  }
  CHECK(cancelled);
  x[0] = std::numeric_limits<float>::quiet_NaN();
  CHECK(rejects([&] {
    upscale_latents(x, 1, 2, 4, "missing", DeviceBackend::kCuda);
  }));
  SafeTensors empty;
  CHECK(rejects([&] {
    validate_latent_upscale_checkpoint(empty);
  }));
}

SLOPFAB_TEST_CATEGORY(latent_upscale_reference, "integration") {
  const char* weights = std::getenv("SLOPFAB_LATENT_UPSCALE_MODEL");
  const char* golden = std::getenv("SLOPFAB_LATENT_UPSCALE_GOLDEN");
  if (!weights || !golden || !std::filesystem::exists(weights) ||
      !std::filesystem::exists(golden)) {
    SKIP_MISSING_FIXTURE("set SLOPFAB_LATENT_UPSCALE_MODEL and SLOPFAB_LATENT_UPSCALE_GOLDEN");
    return;
  }
  SafeTensors reference, model;
  reference.open(golden);
  model.open(weights);
  validate_latent_upscale_checkpoint(model);
  std::vector<DeviceBackend> backends;
#if SLOPFAB_WITH_CUDA
  backends.push_back(DeviceBackend::kCuda);
#endif
#if SLOPFAB_WITH_VULKAN
  backends.push_back(DeviceBackend::kVulkan);
#endif
  for (const char* name : {"still", "video", "chunked"}) {
    const std::string prefix(name);
    const auto* in = reference.find(prefix + ".input");
    if (!in)
      continue;
    auto input = to_f32(*in), expected = to_f32(reference.at(prefix + ".output"));
    for (auto backend : backends) {
      const float scale = to_f32(reference.at(prefix + ".scale"))[0];
      int previous = -1, total = 0;
      auto actual = upscale_latents(input, int(in->shape[1]), int(in->shape[2]), int(in->shape[3]),
                                    weights, backend, {scale}, [&](int done, int steps) {
                                      CHECK(done >= previous);
                                      previous = done;
                                      total = steps;
                                      return true;
                                    });
      CHECK(previous == total);
      CHECK_CLOSE_REL(expected, actual, 2e-4, 2e-4, name);
    }
  }
}

SLOPFAB_TEST_CATEGORY(latent_upscale_generation_handoff, "integration") {
  const char* weights = std::getenv("SLOPFAB_LATENT_UPSCALE_MODEL");
  if (!weights || !std::filesystem::exists(weights)) {
    SKIP_MISSING_FIXTURE("set SLOPFAB_LATENT_UPSCALE_MODEL");
    return;
  }
  GenerateRequest request;
  request.canvas_width = 64;
  request.canvas_height = 32;
  request.num_frames = 22;
  request.video_vae_path = "unused.safetensors";
  const auto plan = resolve_plan(request);
  struct Capture {
    std::shared_ptr<const LatentClip> latents;
    bool upscaled = false;
  } capture;
  RunOptions options;
  options.source = LatentSource::kSyntheticNoise;
  options.verbose = false;
  options.attention_mode = AttentionMode::kExact;
  options.latent_upscale_model_path = weights;
  options.hook_userdata = &capture;
  options.on_latents = [](const std::shared_ptr<const LatentClip>& clip, void* data) {
    static_cast<Capture*>(data)->latents = clip;
  };
  options.on_progress = [](RunStage stage, int done, int total, void* data) {
    auto& c = *static_cast<Capture*>(data);
    if (stage == RunStage::kUpscaling && total > 0 && done == total) c.upscaled = true;
    return stage != RunStage::kVideoDecode;
  };
  const auto result = run_generate(request, plan, options);
  CHECK(result.cancelled && !result.ok);
  CHECK(result.message == "cancelled during video decode");
  CHECK(capture.upscaled);
  CHECK(capture.latents != nullptr);
  if (capture.latents) {
    CHECK(capture.latents->width == 64 && capture.latents->height == 32);
    capture.latents->validate();
  }
}
