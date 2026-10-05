#include "harness.h"
#include "slopfab/generate.h"
#include "slopfab/safetensors_write.h"

#include <algorithm>
#include <filesystem>
#include <limits>

namespace {
struct SamplingFixture {
  std::filesystem::path path;

  SamplingFixture(const char* name, const char* settings, bool fast = false)
      : path(std::filesystem::temp_directory_path() / name) {
    std::vector<slopfab::TensorWrite> tensors = {{"adaln_t_table", {1}, {0}},
                                                 {"blocks.0.adaln_proj.linear.weight", {1}, {0}}};
    if (fast)
      tensors.push_back({"blocks.0.attn.to_gate_compress.weight", {1}, {0}});
    slopfab::write_safetensors(path.string(), tensors, {{"slopfab.sampling", settings}});
  }

  ~SamplingFixture() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

template <typename F

> bool rejects(F fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}
}

SLOPFAB_TEST(sampling_plan_metadata_precedence_and_renaming) {
  using namespace slopfab;
  SamplingFixture model(
      "slopfab_sampling_model.safetensors",
      R"({"version":1,"video_sigma_shift":6,"audio_sigma_shift":2,"base_sigmas":[1,0.5,0]})");
  SamplingFixture adapter("slopfab_sampling_adapter.safetensors",
                          R"({"version":1,"video_sigma_shift":4})");
  GenerateRequest request;
  request.transformer_path = model.path.string();
  request.num_inference_steps = 1; // fixed-grid length replaces the step count
  auto plan = resolve_plan(request);
  CHECK(plan.video_sigma_shift == 6 && plan.audio_sigma_shift == 2);
  CHECK(plan.fixed_sampling_grid && plan.num_model_evaluations() == 2);
  CHECK_NEAR(plan.video_sigmas[1], 6.0f / 7.0f, 0);
  request.loras.push_back({adapter.path.string(), 0});
  CHECK(resolve_plan(request).video_sigma_shift == 6);
  request.loras.back().strength = -0.5f;
  CHECK(resolve_plan(request).video_sigma_shift == 4);
  request.sampling.video_sigma_shift = 8.0f;
  plan = resolve_plan(request);
  CHECK(plan.video_sigma_shift == 8 && plan.audio_sigma_shift == 2);
  CHECK(plan.num_inference_steps == 3);
  CHECK(describe_plan(request, plan).find("shift 8") != std::string::npos);
  CHECK(describe_plan(request, plan).find("LoRA: ") != std::string::npos);
  CHECK(describe_plan(request, plan).find("request overrides") != std::string::npos);
  const auto previous = model.path;
  model.path = model.path.parent_path() / "slopfab_sampling_renamed.safetensors";
  std::filesystem::rename(previous, model.path);
  request.transformer_path = model.path.string();
  CHECK(resolve_plan(request).video_sigmas == plan.video_sigmas);
  CHECK(resolve_plan(request).audio_sigmas == plan.audio_sigmas);
}

SLOPFAB_TEST(sampling_plan_adapter_conflicts_need_explicit_override) {
  using namespace slopfab;
  SamplingFixture first("slopfab_sampling_first.safetensors",
                        R"({"version":1,"video_sigma_shift":4,"base_sigmas":[1,0.5,0]})");
  SamplingFixture second("slopfab_sampling_second.safetensors",
                         R"({"version":1,"video_sigma_shift":8,"base_sigmas":[1,0.25,0]})");
  GenerateRequest request;
  request.loras = {{first.path.string(), 1}, {second.path.string(), 1}};
  CHECK(rejects([&] {
    resolve_plan(request);
  }));
  request.sampling.video_sigma_shift = 3.0f;
  CHECK(rejects([&] {
    resolve_plan(request);
  }));
  request.sampling.base_sigmas = std::vector<float>{1, 0.75f, 0};
  const auto plan = resolve_plan(request);
  std::reverse(request.loras.begin(), request.loras.end());
  CHECK(resolve_plan(request).video_sigmas == plan.video_sigmas);
  CHECK(plan.video_sigma_shift == 3);
}

SLOPFAB_TEST(sampling_plan_explicit_steps_resolve_adapter_defaults) {
  using namespace slopfab;
  SamplingFixture first("slopfab_steps_first.safetensors", R"({"version":1,"default_steps":4})");
  SamplingFixture second("slopfab_steps_second.safetensors", R"({"version":1,"default_steps":8})");
  GenerateRequest request;
  request.loras = {{first.path.string(), 1}, {second.path.string(), 1}};
  CHECK(rejects([&] {
    resolve_plan(request);
  }));
  request.num_inference_steps = 6;
  CHECK(resolve_plan(request).num_inference_steps == 6);
  request.num_inference_steps = 0;
  request.sampling.default_steps = 10;
  CHECK(resolve_plan(request).num_inference_steps == 10);
}

SLOPFAB_TEST(sampling_plan_rejects_invalid_metadata_and_unsafe_execution) {
  using namespace slopfab;
  SamplingFixture bad("slopfab_sampling_bad.safetensors", R"({"version":1,"typo":2})");
  GenerateRequest request;
  request.loras = {{bad.path.string(), 0}};
  CHECK(resolve_plan(request).num_model_evaluations() == 49);
  request.loras.front().strength = 1;
  CHECK(rejects([&] {
    resolve_plan(request);
  }));
  request.loras.clear();
  request.sampling.base_sigmas = std::vector<float>{1, 0.5f, 0};
  const auto plan = resolve_plan(request);
  CHECK(rejects([&] {
    validate_sampling_sampler(plan, sampler::SamplerKind::kAb2);
  }));
  validate_sampling_sampler(plan, sampler::SamplerKind::kEuler);
  request.skip_every = 2;
  CHECK(rejects([&] {
    resolve_plan(request);
  }));
  request.skip_every = 0;
  request.motion_cache.enabled = true;
  CHECK(rejects([&] {
    resolve_plan(request);
  }));
  request.motion_cache.enabled = false;
  request.sampling.base_sigmas.reset();
  request.sampling.video_sigma_shift = std::numeric_limits<float>::min();
  CHECK(rejects([&] {
    resolve_plan(request);
  }));
}

SLOPFAB_TEST(sampling_plan_preserves_fasth3_requirements) {
  using namespace slopfab;
  SamplingFixture model("slopfab_sampling_fasth3_8step_v2.safetensors", R"({"version":1})", true);
  GenerateRequest request;
  request.transformer_path = model.path.string();
  CHECK(resolve_plan(request).num_model_evaluations() == 8);
  request.sampling.audio_steps = 25;
  CHECK(rejects([&] {
    resolve_plan(request);
  }));
  request.sampling.audio_steps.reset();
  request.sampling.video_sigma_shift = 12.0f;
  CHECK(rejects([&] {
    resolve_plan(request);
  }));
  request.sampling.video_sigma_shift = 10.0f;
  CHECK(resolve_plan(request).num_model_evaluations() == 8);
  request.sampling.base_sigmas = std::vector<float>{1, 0.5f, 0};
  CHECK(rejects([&] {
    resolve_plan(request);
  }));
}

SLOPFAB_TEST(sampling_plan_dmad_settings_match_compatibility_alias) {
  using namespace slopfab;
  GenerateRequest named;
  named.schedule = sampler::ScheduleKind::kDmad4Step;
  const auto legacy = resolve_plan(named);
  GenerateRequest request;
  request.sampling = parse_sampling_settings(
      R"({"version":1,"sampler":"renoise","video_sigma_shift":12,"audio_sigma_shift":2,"base_sigmas":[1,0.75,0.5,0.25,0]})");
  const auto plan = resolve_plan(request);
  CHECK(plan.num_model_evaluations() == 4);
  CHECK(plan.video_sigmas == legacy.video_sigmas && plan.audio_sigmas == legacy.audio_sigmas);
  CHECK(plan.video_timesteps == legacy.video_timesteps &&
        plan.audio_timesteps == legacy.audio_timesteps);
  CHECK(plan.sampler == legacy.sampler && plan.sampler == sampler::SamplerKind::kRenoise);
  CHECK(resolve_sampling_sampler(plan) == sampler::SamplerKind::kRenoise);
  CHECK(resolve_sampling_sampler(plan, sampler::SamplerKind::kEuler) ==
        sampler::SamplerKind::kEuler);
  CHECK(describe_plan(request, plan).find("sampler             renoise") != std::string::npos);
  RunOptions options;
  validate_generation_options(request, plan, options);
  options.inference_backend = DeviceBackend::kVulkan;
  validate_generation_options(request, plan, options);
  options.sampler = sampler::SamplerKind::kEuler;
  validate_generation_options(request, plan, options);
  options.sampler = sampler::SamplerKind::kAb2;
  CHECK(rejects([&] {
    validate_generation_options(request, plan, options);
  }));
  options.sampler.reset();
  request.block_cache_span = 1;
  CHECK(rejects([&] {
    validate_generation_options(request, plan, options);
  }));
  request.block_cache_span = 0;
  request.num_inference_steps = 12; // A fixed grid determines its own length.
  CHECK(resolve_plan(request).num_model_evaluations() == 4);
  request.sampling.audio_sigma_shift = 3;
  CHECK(resolve_plan(request).audio_sigma_shift == 3); // Settings are generic data.
}

SLOPFAB_TEST(sampling_plan_sampler_metadata_precedence_and_conflicts) {
  using namespace slopfab;
  SamplingFixture model("slopfab_sampler_model.safetensors", R"({"version":1,"sampler":"ab2"})");
  SamplingFixture first("slopfab_sampler_first.safetensors",
                        R"({"version":1,"sampler":"renoise"})");
  SamplingFixture second("slopfab_sampler_second.safetensors",
                         R"({"version":1,"sampler":"euler"})");
  GenerateRequest request;
  CHECK(resolve_plan(request).sampler == sampler::SamplerKind::kEuler);
  request.transformer_path = model.path.string();
  CHECK(resolve_plan(request).sampler == sampler::SamplerKind::kAb2);
  request.loras.push_back({first.path.string(), 0});
  CHECK(resolve_plan(request).sampler == sampler::SamplerKind::kAb2);
  request.loras.front().strength = 1;
  CHECK(resolve_plan(request).sampler == sampler::SamplerKind::kRenoise);
  request.loras.push_back({second.path.string(), 1});
  CHECK(rejects([&] {
    resolve_plan(request);
  }));
  request.sampling.sampler = sampler::SamplerKind::kEuler;
  CHECK(resolve_plan(request).sampler == sampler::SamplerKind::kEuler);
  request.sampling.sampler.reset();
  request.schedule = sampler::ScheduleKind::kDmad4Step;
  CHECK(resolve_plan(request).sampler == sampler::SamplerKind::kRenoise);
  request.sampling.sampler = sampler::SamplerKind::kEuler;
  CHECK(resolve_plan(request).sampler == sampler::SamplerKind::kEuler);
}
