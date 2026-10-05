#include "harness.h"
#include "slopfab/pipeline.h"
#include "slopfab/safetensors_write.h"

#include <algorithm>
#include <filesystem>

namespace {
template <typename F

> bool rejects(F fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}
} // namespace

SLOPFAB_TEST(audio_steps_joint_schedule) {
  using namespace slopfab::sampler;
  CHECK(rejects([] {
    joint_schedule(0, 3);
  }));
  CHECK(rejects([] {
    joint_schedule(3, 0);
  }));
  // Unequal, non-divisible counts: audio, video, audio, both.
  const auto steps = joint_schedule(2, 3);
  CHECK(steps.size() == 4);
  CHECK(steps[0].video == 0 && steps[0].audio == 0 && !steps[0].advance_video &&
        steps[0].advance_audio);
  CHECK(steps[1].video == 0 && steps[1].audio == 1 && steps[1].advance_video &&
        !steps[1].advance_audio);
  CHECK(steps[2].video == 1 && steps[2].audio == 1 && !steps[2].advance_video &&
        steps[2].advance_audio);
  CHECK(steps[3].video == 1 && steps[3].audio == 2 && steps[3].advance_video &&
        steps[3].advance_audio);
  for (size_t nv = 1; nv <= 25; ++nv) {
    for (size_t na = 1; na <= 25; ++na) {
      size_t v = 0, a = 0;
      const auto schedule = joint_schedule(nv, na);
      CHECK(schedule.size() == joint_step_count(nv, na));
      for (const auto& s : schedule) {
        CHECK(s.video == v && s.audio == a && v < nv && a < na);
        CHECK(s.advance_video || s.advance_audio);
        if (nv == na)
          CHECK(s.advance_video && s.advance_audio && v == a);
        v += s.advance_video;
        a += s.advance_audio;
      }
      CHECK(v == nv && a == na);
      CHECK(schedule.back().advance_video && schedule.back().advance_audio);
    }
  }
}

SLOPFAB_TEST(audio_steps_settings_and_plan) {
  using namespace slopfab;
  auto settings = parse_sampling_settings(R"({"version":1,"audio_steps":25})");
  CHECK(settings.audio_steps == 25);
  overlay_sampling_settings(settings,
                            parse_sampling_settings(R"({"version":1,"default_steps":4})"));
  CHECK(settings.audio_steps == 25 && settings.default_steps == 4);
  for (const char* value : {"null", "true", "\"25\"", "0", "1", "-1", "2.5", "1000001"})
    CHECK(rejects([&] {
      parse_sampling_settings(std::string("{\"version\":1,\"audio_steps\":") + value + "}");
    }));
  for (int value : {-1, 0, 1, 1000001}) {
    settings.audio_steps = value;
    CHECK(rejects([&] {
      validate_sampling_settings(settings);
    }));
  }
  GenerateRequest request;
  request.num_inference_steps = 4;
  const auto original = resolve_plan(request);
  request.sampling.audio_steps = 25;
  auto plan = resolve_plan(request);
  CHECK(plan.video_sigmas == original.video_sigmas);
  CHECK(plan.audio_sigmas.size() == 25 && plan.num_model_evaluations() == 24);
  CHECK(describe_plan(request, plan).find("video 3, audio 24") != std::string::npos);
  request.sampling.audio_steps = 6;
  CHECK(resolve_plan(request).num_model_evaluations() == 7); // 3 + 5 - gcd(3,5)
  request.sampling.base_sigmas = std::vector<float>{1, .7f, .1f, 0};
  plan = resolve_plan(request);
  sampler::FlowScheduler video(12), audio(3);
  video.set_base_sigmas(*request.sampling.base_sigmas);
  audio.set_timesteps(6);
  CHECK(plan.video_sigmas == video.sigmas() && plan.audio_sigmas == audio.sigmas());
  CHECK(plan.fixed_sampling_grid && plan.num_model_evaluations() == 7);
  request.sampling.base_sigmas.reset();
  request.sampling.audio_steps = 4;
  CHECK(resolve_plan(request).audio_sigmas == original.audio_sigmas);
  request.sampling.audio_steps = 6;
  request.still_image = true;
  CHECK(rejects([&] {
    resolve_plan(request);
  }));
  request.still_image = false;
  request.skip_every = 2;
  CHECK(rejects([&] {
    resolve_plan(request);
  }));
  request.skip_every = 0;
  request.block_cache_span = 1;
  CHECK(rejects([&] {
    resolve_plan(request);
  }));
  request.block_cache_span = 0;
  request.motion_cache.enabled = true;
  CHECK(rejects([&] {
    resolve_plan(request);
  }));
}

SLOPFAB_TEST(audio_steps_metadata_precedence) {
  using namespace slopfab;
  const auto root = std::filesystem::temp_directory_path();
  const auto first = root / "slopfab_audio_steps_first.safetensors";
  const auto second = root / "slopfab_audio_steps_second.safetensors";

  struct Cleanup {
    std::filesystem::path a, b;

    ~Cleanup() {
      std::error_code ec;
      std::filesystem::remove(a, ec);
      std::filesystem::remove(b, ec);
    }
  } cleanup{first, second};

  const std::vector<TensorWrite> tensors = {{"adaln_t_table", {1}, {0}},
                                            {"blocks.0.adaln_proj.linear.weight", {1}, {0}}};
  write_safetensors(first.string(), tensors,
                    {{"slopfab.sampling", R"({"version":1,"audio_steps":6})"}});
  write_safetensors(second.string(), tensors,
                    {{"slopfab.sampling", R"({"version":1,"audio_steps":8})"}});
  GenerateRequest r;
  r.num_inference_steps = 4;
  r.transformer_path = first.string();
  CHECK(resolve_plan(r).audio_sigmas.size() == 6);
  r.loras = {{second.string(), -1}};
  CHECK(resolve_plan(r).audio_sigmas.size() == 8);
  r.loras.push_back({first.string(), 1});
  CHECK(rejects([&] {
    resolve_plan(r);
  }));
  r.sampling.audio_steps = 10;
  CHECK(resolve_plan(r).audio_sigmas.size() == 10);
  std::reverse(r.loras.begin(), r.loras.end());
  CHECK(resolve_plan(r).audio_sigmas.size() == 10);
  r.sampling.audio_steps.reset();
  r.loras[0].strength = 0;
  CHECK(resolve_plan(r).audio_sigmas.size() == 8);
}
