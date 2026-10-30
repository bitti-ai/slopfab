#include "harness.h"
#include "slopfab/krea.h"
#include "slopfab/model_profile.h"
#include "slopfab/generate.h"
#include <cmath>
#include <limits>

SLOPFAB_TEST(krea_turbo_plan) {
  slopfab::GenerateRequest r;
  r.model_family = "krea2";
  auto p = slopfab::resolve_plan(r);
  CHECK(p.model.family == "krea2");
  CHECK(p.canvas_width == 1024 && p.canvas_height == 1024);
  CHECK(p.num_model_evaluations() == 8);
  CHECK(p.geometry.video_patch_dim() == 64);
  CHECK(p.layout.num_audio_rows == 0);
  CHECK(p.video_sigmas.size() == 9);
  CHECK(p.video_sigmas.front() == 1 && p.video_sigmas.back() == 0);
  CHECK_NEAR(p.video_sigmas[4], 1 / (1 + std::exp(-1.15)), 1e-7);
  CHECK(slopfab::krea_guidance(r) == 0);
  r.guidance_scale_set = true;
  r.guidance_scale = 3.5f;
  CHECK(slopfab::krea_guidance(r) == 3.5f);
  r.num_inference_steps = 52;
  r.canvas_width = 768;
  r.canvas_height = 1024;
  r.sampling.video_sigma_shift = 1.f;
  p = slopfab::resolve_plan(r);
  CHECK(p.num_model_evaluations() == 52);
  CHECK(p.layout.num_video_rows == 48 * 64);
  CHECK_NEAR(p.video_sigmas[26], .5, 1e-7);
  CHECK(slopfab::model_profile("krea2").conditioner_width == 30720);
}

SLOPFAB_TEST(krea_rejects_incompatible_requests) {
  slopfab::GenerateRequest base;
  base.model_family = "krea2";
  auto fails = [](const auto& r) {
    try {
      slopfab::resolve_plan(r);
      return false;
    } catch (const std::exception&) {
      return true;
    }
  };
  auto r = base;
  r.canvas_width = 513;
  r.canvas_height = 512;
  CHECK(fails(r));
  r = base;
  r.canvas_width = 512;
  CHECK(fails(r));
  r = base;
  r.num_inference_steps = -1;
  CHECK(fails(r));
  r = base;
  r.sampling.video_sigma_shift = std::numeric_limits<float>::infinity();
  CHECK(fails(r));
  r = base;
  r.reference_image_paths = {"unused.png"};
  CHECK(fails(r));
  r = base;
  r.sampling.audio_steps = 2;
  CHECK(fails(r));
  r = base;
  r.guidance_scale_set = true;
  r.guidance_scale = -1;
  CHECK(fails(r));
  auto p = slopfab::resolve_plan(base);
  slopfab::RunOptions o;
  o.inference_backend = slopfab::DeviceBackend::kVulkan;
  o.vulkan_portable_arithmetic = false;
  o.attention_mode = slopfab::AttentionMode::kFlash2;
  bool rejected = false;
  try {
    slopfab::validate_generation_options(base, p, o);
  } catch (const std::exception&) {
    rejected = true;
  }
  CHECK(rejected);
  o.vulkan_portable_arithmetic = true;
  slopfab::validate_generation_options(base, p, o);
}
