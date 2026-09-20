#include "harness.h"
#include "slopfab/generate.h"
#include "slopfab/safetensors_write.h"
#include <filesystem>

namespace {
template <typename F> bool rejected(F f) {
  try { f(); } catch (const std::exception&) { return true; }
  return false;
}
}

SLOPFAB_TEST(conditioning_settings_schema_and_cache_dependencies) {
  using namespace slopfab;
  for (const char* text : {"{}", R"({"version":2})", R"({"version":1,"unknown":0})",
      R"({"version":1,"image_short_edge":0})", R"({"version":1,"video_first":1})",
      R"({"version":1,"max_frames":1.5})"})
    CHECK(rejected([&] { parse_conditioning_settings(text); }));
  GenerateRequest r;
  r.reference_image_paths = {"unavailable.png"};
  const auto original = reference_cache_key(r);
  r.conditioning = parse_conditioning_settings(R"({"version":1,"image_short_edge":512})");
  CHECK(reference_cache_key(r) != original);
  const auto resized = reference_cache_key(r);
  r.sampling.video_sigma_shift = 6.0f;
  CHECK(reference_cache_key(r) == resized);
  r.conditioning.require_prompt_embedding = true;
  r.conditioning.fixed_prompt_tokens = 17;
  const auto plan = resolve_plan(r);
  CHECK(plan.conditioning.fixed_prompt_tokens == 17);
  RunOptions options;
  CHECK(rejected([&] { validate_generation_options(r, plan, options); }));
  options.prompt_embedding_path = "embedding.safetensors";
  validate_generation_options(r, plan, options);
}

SLOPFAB_TEST(conditioning_settings_presets_and_explicit_steps) {
  using namespace slopfab;
  GenerateRequest r;
  CHECK(resolve_plan(r).num_inference_steps == 50);
  r.sampling.default_steps = 8;
  CHECK(resolve_plan(r).num_inference_steps == 8);
  r.num_inference_steps = 6;
  CHECK(resolve_plan(r).num_inference_steps == 6);
  r.animate = true;
  const auto preset = resolve_conditioning_settings(r);
  CHECK(preset.video_first && preset.fixed_prompt_tokens == 362);
  CHECK(preset.require_euler && !preset.allow_caches && preset.max_frames == 360);
  r.conditioning.image_short_edge = 512;
  r.conditioning.max_frames = 720;
  const auto override = resolve_conditioning_settings(r);
  CHECK(override.max_frames == 720 && override.image_short_edge == 512);
  r.animate = false;
  r.conditioning = parse_conditioning_settings(R"({"version":1,"require_euler":true,"allow_caches":false})");
  auto plan = resolve_plan(r);
  RunOptions options;
  options.sampler = sampler::SamplerKind::kAb2;
  CHECK(rejected([&] { validate_generation_options(r, plan, options); }));
  options.sampler = sampler::SamplerKind::kEuler;
  r.skip_every = 2;
  CHECK(rejected([&] { validate_generation_options(r, plan, options); }));
}

SLOPFAB_TEST(conditioning_settings_metadata_conflicts_and_disabled_adapters) {
  using namespace slopfab;
  const auto first = std::filesystem::temp_directory_path() / "slopfab_conditioning_first.safetensors";
  const auto second = std::filesystem::temp_directory_path() / "slopfab_conditioning_second.safetensors";
  struct Cleanup {
    std::filesystem::path a,b;
    ~Cleanup() { std::error_code ec; std::filesystem::remove(a,ec); std::filesystem::remove(b,ec); }
  } cleanup{first,second};
  write_safetensors(first.string(), {{"x",{1},{0}}},
      {{"slopfab.conditioning", R"({"version":1,"image_short_edge":512})"}});
  write_safetensors(second.string(), {{"x",{1},{0}}},
      {{"slopfab.conditioning", R"({"version":1,"image_short_edge":768})"}});
  GenerateRequest r;
  r.loras = {{first.string(),1}, {second.string(),0}};
  CHECK(resolve_conditioning_settings(r).image_short_edge == 512);
  r.loras.back().strength = 1;
  CHECK(rejected([&] { resolve_conditioning_settings(r); }));
  r.conditioning.image_short_edge = 1024;
  CHECK(resolve_conditioning_settings(r).image_short_edge == 1024);
}

SLOPFAB_TEST(generation_options_shared_validation) {
  using namespace slopfab;
  GenerateRequest request;
  auto plan = resolve_plan(request);
  RunOptions options;
  validate_generation_options(request, plan, options);
  request.cache_threshold = 0.1f;
  request.skip_every = 2;
  CHECK(rejected([&] { validate_generation_options(request, plan, options); }));
  request.cache_threshold = 0;
  request.skip_every = 0;
  options.attention_band = -1;
  CHECK(rejected([&] { validate_generation_options(request, plan, options); }));
  options.attention_band = 0;
  options.sol_schedule.step_every = 0;
  CHECK(rejected([&] { validate_generation_options(request, plan, options); }));
}

SLOPFAB_TEST(conditioning_cache_tracks_continuation_canvas) {
  using namespace slopfab;
  GenerateRequest request;
  request.conditioning.references_at_target_canvas = true;
  request.reference_image_paths = {"unavailable.png"};
  auto clip = std::make_shared<LatentClip>();
  clip->width = 256;
  clip->height = 256;
  request.continuation = clip;
  const auto reference = reference_cache_key(request);
  const auto prompt = conditioning_cache_key(request);
  auto larger = std::make_shared<LatentClip>(*clip);
  larger->width = 512;
  larger->height = 512;
  request.continuation = larger;
  CHECK(reference_cache_key(request) != reference);
  CHECK(conditioning_cache_key(request) != prompt);
}
