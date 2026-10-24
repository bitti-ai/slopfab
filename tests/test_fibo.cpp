#include "harness.h"
#include "slopfab/fibo.h"
#include "slopfab/model_profile.h"
#include "slopfab/pipeline.h"
#include "slopfab/generate.h"
#include <cmath>

SLOPFAB_TEST(fibo_plan_and_schedule) {
  slopfab::GenerateRequest r;
  r.model_family = "fibo2";
  r.reference_image_paths = {"source.png"};
  auto p = slopfab::resolve_plan(r);
  CHECK(p.model.family == "fibo2");
  CHECK(p.canvas_width == 1024);
  CHECK(p.num_model_evaluations() == 4);
  CHECK(p.geometry.video_patch_dim() == 128);
  CHECK(p.layout.num_audio_rows == 0);
  CHECK(p.video_sigmas.front() == 1);
  CHECK(p.video_sigmas.back() == 0);
  CHECK_NEAR(p.video_sigmas[2], 1 / (1 + std::exp(-1.15)), 1e-7);
  auto small = slopfab::fibo_sigma_schedule(4, 256);
  CHECK_NEAR(small[2], 1 / (1 + std::exp(-.5)), 1e-7);
  CHECK(slopfab::fibo_sigma_schedule(1, 256) == std::vector<float>({1, 0}));
  CHECK(slopfab::fibo::gist_count(1) == 256);
  CHECK(slopfab::fibo::gist_count(257) == 320);
  CHECK(slopfab::fibo::gist_count(4096) == 512);
}

SLOPFAB_TEST(fibo_mask_plan_and_guards) {
  slopfab::GenerateRequest r;
  r.model_family = "fibo2";
  auto fails = [](const auto& r) {
    try {
      slopfab::resolve_plan(r);
      return false;
    } catch (const std::exception&) {
      return true;
    }
  };
  CHECK(fails(r));
  r.image_edit.image = std::make_shared<slopfab::RGBImage>(
      slopfab::RGBImage{63, 45, std::vector<uint8_t>(63 * 45 * 3, 120)});
  r.image_edit.mask = std::make_shared<std::vector<uint8_t>>(63 * 45, 255);
  r.image_edit.strength = .5;
  auto p = slopfab::resolve_plan(r);
  CHECK(p.canvas_width == 64);
  CHECK(p.canvas_height == 48);
  CHECK(p.num_inference_steps == 2);
  r.reference_image_paths = {"other.png"};
  CHECK(fails(r));
  r.reference_image_paths.clear();
  r.sampling.video_sigma_shift = 2;
  CHECK(fails(r));
  r.sampling.video_sigma_shift.reset();
  slopfab::RunOptions o;
  o.inference_backend = slopfab::DeviceBackend::kVulkan;
  bool rejected = false;
  try {
    slopfab::validate_generation_options(r, p, o);
  } catch (const std::exception&) {
    rejected = true;
  }
  CHECK(rejected);
}

SLOPFAB_TEST(fibo_prompt_images_and_glyphs) {
  std::vector<std::string> glyphs;
  auto plain = slopfab::fibo::edit_prompt("Make it blue", 1, glyphs);
  CHECK(plain.find("<|vision_start|><|image_pad|><|vision_end|>") != std::string::npos);
  CHECK(plain.find("\"edit_instruction\":\"Make it blue\"") != std::string::npos);
  CHECK(glyphs.empty());
  auto structured = slopfab::fibo::edit_prompt(
      R"({"edit_instruction":"Add a sign","structured_caption":{"text_render":[{"text":"Hello"}],"aesthetic_score":4,"empty":""}})",
      2, glyphs);
  CHECK(glyphs == std::vector<std::string>{"Hello"});
  CHECK(structured.find("image_2") != std::string::npos);
  CHECK(structured.find("aesthetic_score") == std::string::npos);
  CHECK(structured.find("empty") == std::string::npos);
}
