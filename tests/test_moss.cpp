#include "harness.h"
#include "slopfab/model_profile.h"
#include "slopfab/pipeline.h"
#include "slopfab/generate.h"
#include <limits>

SLOPFAB_TEST(moss_plan_and_schedule) {
  slopfab::GenerateRequest r;
  r.model_family = "moss";
  const auto p = slopfab::resolve_plan(r);
  CHECK(p.canvas_width == 1024 && p.canvas_height == 1024);
  CHECK(p.model.family == "moss");
  CHECK(p.layout.num_audio_rows == 0);
  CHECK(p.layout.num_video_rows == 4096);
  CHECK(p.geometry.video_patch_dim() == 128);
  CHECK(p.video_sigmas.size() == 51);
  CHECK(p.num_model_evaluations() == 50);
  CHECK_NEAR(p.video_sigmas.front(), 1., 1e-7);
  CHECK_NEAR(p.video_sigmas[49], .024, 1e-7);
  CHECK(p.video_sigmas.back() == 0);
  for (size_t i = 1; i < p.video_sigmas.size(); ++i)
    CHECK(p.video_sigmas[i] < p.video_sigmas[i - 1]);
  CHECK(slopfab::moss_sigma_schedule(1, 5) == std::vector<float>({1, 0}));
}

SLOPFAB_TEST(moss_rejects_incompatible_requests) {
  const auto rejected = [](slopfab::GenerateRequest r) {
    try {
      slopfab::resolve_plan(r);
      return false;
    } catch (const std::exception&) {
      return true;
    }
  };
  slopfab::GenerateRequest r;
  r.model_family = "moss";
  r.canvas_width = 513;
  CHECK(rejected(r));
  r.canvas_width = 512;
  r.canvas_height = 512;
  r.sampling.audio_steps = 8;
  CHECK(rejected(r));
  r.sampling.audio_steps.reset();
  r.uncond_skip_layers = {42};
  CHECK(rejected(r));
  r.uncond_skip_layers = {22};
  r.guidance_scale = std::numeric_limits<float>::quiet_NaN();
  CHECK(rejected(r));
  r.guidance_scale = 3;
  r.reference_image_paths = {"image.png"};
  CHECK(rejected(r));
}
