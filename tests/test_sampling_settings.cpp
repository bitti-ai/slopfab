#include "harness.h"
#include "slopfab/sampling_settings.h"

#include <cmath>
#include <limits>

namespace {
using namespace slopfab;
using sampler::FlowScheduler;
using sampler::ScheduleKind;

template <class Function> bool rejects(Function function) {
  try {
    function();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}
} // namespace

SLOPFAB_TEST(sampling_settings_parse_and_overlay) {
  SamplingSettings settings = parse_sampling_settings(
      R"({"version":1,"video_sigma_shift":12,"audio_sigma_shift":3,"base_sigmas":[1,0.5,0]})");
  CHECK(settings.video_sigma_shift == 12.0f);
  CHECK(settings.audio_sigma_shift == 3.0f);
  CHECK(*settings.base_sigmas == std::vector<float>({1.0f, 0.5f, 0.0f}));
  overlay_sampling_settings(settings,
                            parse_sampling_settings(R"({"version":1,"video_sigma_shift":5})"));
  CHECK(settings.video_sigma_shift == 5.0f);
  CHECK(settings.audio_sigma_shift == 3.0f);
  CHECK(settings.base_sigmas->size() == 3);
  const SamplingSettings absent = sampling_settings_from_metadata({{"unrelated", "ignored"}});
  CHECK(!absent.video_sigma_shift && !absent.audio_sigma_shift && !absent.base_sigmas);
  const auto embedded = sampling_settings_from_metadata(
      {{"slopfab.sampling", R"({"version":1,"audio_sigma_shift":2})"}});
  CHECK(embedded.audio_sigma_shift == 2.0f);
  CHECK(!embedded.video_sigma_shift && !embedded.base_sigmas);
  CHECK(rejects([] {
    sampling_settings_from_metadata({{"slopfab.sampling", ""}});
  }));
}

SLOPFAB_TEST(sampling_settings_reject_malformed_schema_and_values) {
  for (const char* invalid : {"",
                              "{",
                              "[]",
                              "null",
                              "{}",
                              "{\"version\":1} trailing",
                              R"({"version":2})",
                              R"({"version":1.5})",
                              R"({"version":"1"})",
                              R"({"version":true})",
                              R"({"version":1,"unknown":3})",
                              R"({"version":1,"video_sigma_shift":"3"})",
                              R"({"version":1,"video_sigma_shift":null})",
                              R"({"version":1,"audio_sigma_shift":false})",
                              R"({"version":1,"audio_sigma_shift":0})",
                              R"({"version":1,"video_sigma_shift":-1})",
                              R"({"version":1,"video_sigma_shift":1e999})",
                              R"({"version":1,"video_sigma_shift":1e40})",
                              R"({"version":1,"video_sigma_shift":1e-100})",
                              R"({"version":1,"base_sigmas":null})",
                              R"({"version":1,"base_sigmas":[]})",
                              R"({"version":1,"base_sigmas":[1]})",
                              R"({"version":1,"base_sigmas":[1,0.5]})",
                              R"({"version":1,"base_sigmas":[0,0]})",
                              R"({"version":1,"base_sigmas":[1,1,0]})",
                              R"({"version":1,"base_sigmas":[0.5,0.8,0]})",
                              R"({"version":1,"base_sigmas":[1,-0.1,0]})",
                              R"({"version":1,"base_sigmas":[1.01,0]})",
                              R"({"version":1,"base_sigmas":[1,"0"]})",
                              R"({"version":1,"base_sigmas":[1,1e999,0]})"}) {
    CHECK_MSG(rejects([&] {
                parse_sampling_settings(invalid);
              }),
              "accepted invalid settings: %s", invalid);
  }
  for (float shift : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()}) {
    SamplingSettings invalid;
    invalid.video_sigma_shift = shift;
    CHECK(rejects([&] {
      validate_sampling_settings(invalid);
    }));
    CHECK(rejects([&] {
      FlowScheduler scheduler(shift);
    }));
    SamplingSettings destination;
    destination.audio_sigma_shift = 3.0f;
    CHECK(rejects([&] {
      overlay_sampling_settings(destination, invalid);
    }));
    CHECK(!destination.video_sigma_shift && destination.audio_sigma_shift == 3.0f);
  }
}

SLOPFAB_TEST(sampling_settings_preserve_named_schedule_arithmetic) {
  const auto defaults = sampling_schedule_defaults(ScheduleKind::kDefault);
  CHECK(!defaults.video_sigma_shift && !defaults.audio_sigma_shift && !defaults.base_sigmas);
  const auto fast = sampling_schedule_defaults(ScheduleKind::kFastH3V2);
  CHECK(fast.video_sigma_shift == 10.0f && fast.audio_sigma_shift == 3.0f);
  for (float shift : {1.0f, 3.0f, 10.0f, 12.0f}) {
    for (auto kind : {ScheduleKind::kTaoMate3Step, ScheduleKind::kFastH3V2}) {
      std::vector<float> legacy;
      if (kind == ScheduleKind::kTaoMate3Step) {
        for (int i : {0, 16, 33, 49}) {
          const float base = static_cast<float>(49 - i) / 49.0f;
          legacy.push_back(shift * base / (1.0f + (shift - 1.0f) * base));
        }
      } else {
        for (int rung : {999, 874, 749, 624, 500, 375, 250, 125, 0}) {
          const float base = static_cast<float>(rung) / 1000.0f;
          legacy.push_back(shift * base / (1.0f + (shift - 1.0f) * base));
        }
      }
      FlowScheduler named(shift), generic(shift);
      named.set_timesteps(50, kind);
      generic.set_base_sigmas(*sampling_schedule_defaults(kind).base_sigmas);
      CHECK(named.sigmas() == legacy);
      CHECK(generic.sigmas() == legacy);
      CHECK(named.timesteps() == generic.timesteps());
    }
  }
  CHECK(rejects([] {
    sampling_schedule_defaults(static_cast<ScheduleKind>(99));
  }));
}

SLOPFAB_TEST(sampling_settings_reject_invalid_or_collapsed_grids_atomically) {
  FlowScheduler scheduler(12.0f);
  scheduler.set_base_sigmas({1.0f, 0.5f, 0.0f});
  const auto original = scheduler.sigmas();
  const auto timesteps = scheduler.timesteps();
  for (const auto& invalid :
       std::vector<std::vector<float>>{{},
                                       {1.0f},
                                       {1.0f, 0.1f},
                                       {1.0f, 0.5f, 0.5f, 0.0f},
                                       {1.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f},
                                       {1.0f, std::nextafter(1.0f, 0.0f), 0.0f}}) {
    CHECK(rejects([&] {
      scheduler.set_base_sigmas(invalid);
    }));
    CHECK(scheduler.sigmas() == original);
    CHECK(scheduler.timesteps() == timesteps);
  }
}
