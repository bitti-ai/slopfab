#include "harness.h"
#include "slopfab/lcs.h"
#include "slopfab/continuation.h"
#include "latent_fixture.h"
#include "../src/core/lcs_fingerprint_cache.h"
#include <fstream>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>

namespace {
slopfab::LcsData fixture() {
  std::vector<float> colors(8 * 96, 0), anchors(8 * 96, 0);
  const float rgb[8][3] = {{1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {1, 0, 1},
                           {0, 1, 1}, {1, 1, 0}, {0, 0, 0}, {1, 1, 1}};
  for (int i = 0; i < 8; ++i)
    for (int k = 0; k < 3; ++k)
      colors[i * 96 + k] = anchors[i * 96 + k] = rgb[i][k];
  auto d = slopfab::fit_lcs(colors, anchors);
  d.vae_sha256 = std::string(64, 'a');
  return d;
}

template <class F> bool rejects(F fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}
}

SLOPFAB_TEST(lcs_calibration_roundtrip_and_validation) {
  auto data = fixture();
  CHECK_NEAR(data.explained_variance, 1, 1e-6);
  auto bad = data;
  bad.basis[0] = 10;
  CHECK(rejects([&] {
    bad.validate();
  }));
  std::vector<float> sharp(96 * 8, 0);
  for (int r = 0; r < 8; ++r) {
    sharp[r * 96 + 5] = float(r + 1);
    sharp[r * 96 + 6] = -float(r + 1);
  }
  slopfab::fit_lcs_sharpness(data, sharp);
  CHECK_NEAR(data.sharpness[5], 1 / std::sqrt(2.f), 1e-5);
  CHECK_NEAR(data.sharpness[6], -1 / std::sqrt(2.f), 1e-5);
  const auto path =
      (std::filesystem::temp_directory_path() / "slopfab-lcs-test.safetensors").string();
  data.save(path);
  const auto loaded = slopfab::LcsData::load(path);
  CHECK(data.basis == loaded->basis && data.sharpness == loaded->sharpness);
  CHECK(data.vae_sha256 == loaded->vae_sha256);
  std::filesystem::remove(path);
  CHECK(rejects([] {
    slopfab::fit_lcs(std::vector<float>(96 * 8, 0), std::vector<float>(96 * 8, 0));
  }));
}

SLOPFAB_TEST(lcs_color_tone_mask_and_structure) {
  auto d = std::make_shared<slopfab::LcsData>(fixture());
  d->latent_std.assign(24, 2);
  d->latent_mean.assign(24, .25f);
  slopfab::SamplingStep step;
  step.frames = 1;
  step.height = 2;
  step.width = 4;
  step.row_width = 96;
  step.count = 192;
  step.sigma = .5f;
  std::vector<float> prediction(192, .125f);
  step.denoised = prediction.data();
  const auto original = prediction;
  slopfab::LcsSettings neutral;
  CHECK(!slopfab::make_lcs_effect(d, neutral));
  auto tone = neutral;
  tone.brightness = .2f;
  auto effect = slopfab::make_lcs_effect(d, tone, {0, 1});
  CHECK(effect(step));
  CHECK(std::equal(prediction.begin(), prediction.begin() + 96, original.begin()));
  for (int i = 0; i < 96; ++i)
    CHECK_NEAR(prediction[96 + i], original[96 + i] + (i < 3 ? .1f : 0.f), 1e-6);
  for (int mode = 0; mode < 3; ++mode) {
    auto settings = neutral;
    settings.mode = mode;
    settings.strength = .6f;
    settings.color = {1, 0, 0};
    prediction = original;
    effect = slopfab::make_lcs_effect(d, settings);
    CHECK(effect(step));
    CHECK(prediction != original);
    for (int r = 0; r < 2; ++r)
      for (int i = 3; i < 96; ++i)
        CHECK(prediction[r * 96 + i] == original[r * 96 + i]);
    prediction = original;
    step.step = 3;
    settings.end_step = 2;
    effect = slopfab::make_lcs_effect(d, settings);
    CHECK(effect(step));
    CHECK(prediction == original);
    step.step = 0;
  }
  prediction = original;
  step.modality = 1;
  CHECK(effect(step));
  CHECK(prediction == original);
  step.modality = 0;
  effect = slopfab::make_lcs_effect(d, tone, {1, 1, 1});
  CHECK(rejects([&] {
    effect(step);
  }));
  CHECK(rejects([&] {
    slopfab::make_lcs_effect(d, tone, {-1});
  }));
}

SLOPFAB_TEST(lcs_settings_and_anchor_isolation) {
  CHECK(rejects([] {
    slopfab::parse_lcs_settings("{\"typo\":1}");
  }));
  CHECK(rejects([] {
    slopfab::parse_lcs_settings("{\"start_step\":1.5}");
  }));
  CHECK(rejects([] {
    slopfab::parse_lcs_settings("{\"strength\":-1}");
  }));
  CHECK(rejects([] {
    slopfab::parse_lcs_settings("{\"color\":[1,0]}");
  }));
  auto s = slopfab::parse_lcs_settings(
      "{\"mode\":\"type_i\",\"anchor\":\"self\",\"anchor_strength\":1}");
  auto d = std::make_shared<slopfab::LcsData>(fixture());
  auto first = slopfab::make_lcs_effect(d, s), second = slopfab::make_lcs_effect(d, s);
  std::vector<float> values(96, .2f);
  slopfab::SamplingStep step;
  step.frames = 1;
  step.height = step.width = 2;
  step.row_width = 96;
  step.count = 96;
  step.denoised = values.data();
  CHECK(first(step));
  std::fill(values.begin(), values.end(), .8f);
  step.step = 1;
  CHECK(first(step));
  for (int i = 0; i < 3; ++i)
    CHECK_NEAR(values[i], .2f, 1e-5);
  for (int i = 3; i < 96; ++i)
    CHECK(values[i] == .8f);
  std::fill(values.begin(), values.end(), .8f);
  CHECK(second(step));
  for (float x : values)
    CHECK(x == .8f);
}

SLOPFAB_TEST(lcs_single_shot_tone_grades_body_and_ending_once_preserving_source) {
  using namespace slopfab;
  auto data = std::make_shared<LcsData>(fixture());
  auto tone = parse_lcs_settings(R"({"tone_after_denoise":true,"contrast":0.9,"brightness":0.1,"saturation":0.95,"temperature":0.05,"start_step":2,"end_step":4})");
  CHECK(!lcs_sampling_settings(tone).active());
  auto mixed = tone;
  mixed.sharpness = .5f;
  auto sampling = lcs_sampling_settings(mixed);
  CHECK(sampling.sharpness == .5f && sampling.start_step == 2 && sampling.end_step == 4);
  CHECK(sampling.contrast == 1 && sampling.brightness == 0 && sampling.saturation == 1 && sampling.temperature == 0);
  CHECK(rejects([] { parse_lcs_settings(R"({"tone_after_denoise":1})"); }));

  LatentFixture file;
  file.write(22);
  auto ending = *LatentClip::load(file.path.string());
  // Identical neutral patches either side of the join must receive exactly
  // the same grade, with no dependence on the ending pass's length or steps.
  std::fill(ending.video_rows.begin(), ending.video_rows.end(), .2f);
  const auto original = ending;
  for (bool continued : {false, true}) {
    for (const std::vector<float> mask : {std::vector<float>{}, std::vector<float>{0, 1}, std::vector<float>{1, 0}}) {
      const LatentClip* source = continued ? &original : nullptr;
      const auto plan = plan_latent_end_anchor(ending, 22, 34, source);
      auto layout = ending.layout();
      const size_t row = size_t(layout.rows_per_frame()) * 96;
      const int window_latents = plan.prefix_video_latents + plan.generated_video_latents + plan.suffix_video_latents;
      std::vector<float> video(size_t(window_latents) * row, .2f);
      std::vector<float> audio(size_t(plan.window_audio_latents) * 64, 0);
      auto joined = join_latent_end_anchor(ending, source, plan, video, audio);
      const auto before = joined;
      const auto output = joined.layout();
      const int preserved = continued ? original.layout().num_latent_frames : 0;
      SamplingStep step;
      step.frames = output.num_latent_frames; step.height = output.latent_height; step.width = output.latent_width;
      step.row_width = 96; step.count = joined.video_rows.size(); step.denoised = joined.video_rows.data();
      apply_lcs_output_tone(data, tone, mask, step, preserved);
      CHECK(joined.audio_rows == before.audio_rows);
      CHECK(ending.video_rows == original.video_rows);
      CHECK(std::equal(joined.video_rows.begin(), joined.video_rows.begin() + preserved * row, before.video_rows.begin()));
      auto expected = std::vector<float>(row, .2f);
      step.frames = 1; step.count = row; step.denoised = expected.data(); step.step = 0;
      auto once = tone; once.tone_after_denoise = false; once.start_step = 0;
      CHECK(make_lcs_effect(data, once, mask)(step));
      CHECK(expected != std::vector<float>(row, .2f));
      for (int f = preserved; f < output.num_latent_frames; ++f)
        for (size_t i = 0; i < row; ++i)
          CHECK_NEAR(joined.video_rows[size_t(f) * row + i], expected[i], 1e-6);
    }
  }
}

SLOPFAB_TEST(lcs_fingerprint_reuses_unchanged_files_and_invalidates_edits) {
  namespace fs = std::filesystem;
  const auto folder = fs::temp_directory_path() / ("lcs-fingerprint-" +
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directory(folder);
  struct Cleanup { fs::path folder; ~Cleanup() { std::error_code ec; fs::remove_all(folder, ec); } } cleanup{folder};
  const auto path = folder / "vae";
  const auto write = [](const fs::path& p, const char* content) { std::ofstream(p, std::ios::binary) << content; };
  write(path, "first");
  slopfab::detail::LcsFingerprintCache cache;
  int reads = 0;
  const auto hash = [&](const std::string& p) { ++reads; return slopfab::lcs_vae_fingerprint(p); };
  const auto first = cache.get(path.string(), hash);
  CHECK(first.size() == 64);
  CHECK(cache.get(path.string(), hash) == first);
  CHECK(reads == 1);
  const auto modified = fs::last_write_time(path);
  // Windows can coalesce ChangeTime updates inside one system-clock tick.
  // This test checks invalidation once the filesystem reports a new change time.
#ifdef _WIN32
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
#endif
  write(path, "other");
  fs::last_write_time(path, modified);
  const auto second = cache.get(path.string(), hash);
  CHECK(second != first);
  CHECK(reads == 2);
  const auto replacement = folder / "replacement";
  write(replacement, "third");
  fs::last_write_time(replacement, modified);
  fs::remove(path); fs::rename(replacement, path);
  CHECK(cache.get(path.string(), hash) != second);
  CHECK(reads == 3);
  fs::remove(path);
  CHECK(rejects([&] { cache.get(path.string(), hash); }));
  write(path, "first");
  CHECK(rejects([&] { cache.get(path.string(), [](const std::string&) -> std::string { throw std::runtime_error("read failed"); }); }));
  CHECK(rejects([&] { cache.get(path.string(), [&](const std::string&) { write(path, "changed during hash"); return "bad"; }); }));
  CHECK(cache.get(path.string(), hash) != "bad");
  CHECK(reads == 4);
}
