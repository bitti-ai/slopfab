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

SLOPFAB_TEST(lcs_drift_bounds_normalization_and_preservation) {
  auto d = std::make_shared<slopfab::LcsData>(fixture());
  d->latent_mean.assign(24, .25f);
  d->latent_std.assign(24, 2);
  d->drift_direction.assign(96, 0);
  d->drift_direction[4] = 1;
  d->drift_reference = {.5f, .25f};
  auto settings = slopfab::parse_lcs_settings(
      "{\"drift_strength\":0.5,\"drift_max\":1,\"start_step\":2,\"end_step\":3}");
  CHECK(settings.active());
  slopfab::SamplingStep step;
  step.frames = 3;
  step.height = 2;
  step.width = 4;
  step.row_width = 96;
  step.count = 6 * 96;
  step.step = 2;
  std::vector<float> values(step.count, .125f);
  // Frame 0 is below threshold, frame 1 exceeds it by .5 raw units,
  // frame 2 exceeds the cap. One masked patch in each frame is untouched.
  for (int r = 0; r < 6; ++r)
    values[r * 96 + 4] = r < 2 ? .125f : r < 4 ? .5f : 2.f;
  const auto original = values;
  step.denoised = values.data();
  auto effect = slopfab::make_lcs_effect(d, settings, {0, 1});
  CHECK(effect(step));
  for (int r = 0; r < 6; ++r)
    for (int i = 0; i < 96; ++i) {
      const float delta = i == 4 && r % 2 ? (r == 3 ? -.125f : r == 5 ? -.25f : 0.f) : 0.f;
      CHECK_NEAR(values[r * 96 + i], original[r * 96 + i] + delta, 1e-7);
    }
  // Color and every orthogonal coordinate are unchanged, including nonunit
  // model normalization. Audio and inactive steps must be exact no-ops.
  for (int index : {1, 4}) {
    values = original;
    step.step = index;
    CHECK(effect(step));
    CHECK(values == original);
  }
  step.step = 2;
  step.modality = 1;
  CHECK(effect(step));
  CHECK(values == original);
  const auto path = (std::filesystem::temp_directory_path() / "slopfab-lcs-drift-test.safetensors").string();
  d->save(path);
  const auto loaded = slopfab::LcsData::load(path);
  CHECK(loaded->drift_direction == d->drift_direction);
  CHECK(loaded->drift_reference == d->drift_reference);
  std::filesystem::remove(path);
  CHECK(rejects([&] {
    slopfab::make_lcs_effect(std::make_shared<slopfab::LcsData>(fixture()), settings);
  }));
  d->drift_direction[4] = 2;
  CHECK(rejects([&] { d->validate(); }));
  d->drift_direction[4] = 0;
  d->drift_direction[0] = 1;
  CHECK(rejects([&] { d->validate(); }));
  CHECK(rejects([] { slopfab::parse_lcs_settings("{\"drift_strength\":1.1}"); }));
  CHECK(rejects([] { slopfab::parse_lcs_settings("{\"drift_max\":-1}"); }));
}

SLOPFAB_TEST(lcs_highlight_shoulder_preserves_shadows_chroma_and_residual) {
  auto d = std::make_shared<slopfab::LcsData>(fixture());
  auto settings = slopfab::parse_lcs_settings("{\"highlight_strength\":0.5,\"highlight_knee\":0.8}");
  slopfab::SamplingStep step;
  step.frames = 1;
  step.height = 2;
  step.width = 6;
  step.row_width = 96;
  step.count = 288;
  std::vector<float> values(step.count, .123f);
  for (int r = 0; r < 3; ++r)
    for (int i = 0; i < 3; ++i)
      values[r * 96 + i] = r == 0 ? .4f : 1.2f + (i - 1) * .1f;
  const auto original = values;
  step.denoised = values.data();
  auto effect = slopfab::make_lcs_effect(d, settings, {1, 1, 0});
  CHECK(effect(step));
  for (int r = 0; r < 3; ++r)
    for (int i = 0; i < 96; ++i)
      CHECK_NEAR(values[r * 96 + i], original[r * 96 + i] - (r == 1 && i < 3 ? .2f : 0.f), 1e-6);
  CHECK(rejects([] { slopfab::parse_lcs_settings("{\"highlight_strength\":1.1}"); }));
  CHECK(rejects([] { slopfab::parse_lcs_settings("{\"highlight_knee\":-0.1}"); }));
}

SLOPFAB_TEST(lcs_output_stability_preserves_prefix_lightness_residual_and_masks) {
  auto d = std::make_shared<slopfab::LcsData>(fixture());
  d->latent_mean.assign(24, .25f);
  d->latent_std.assign(24, 2.f);
  d->color_reference.assign(30, 0);
  d->edge_reference.assign(5, .1f);
  auto s = slopfab::parse_lcs_settings("{\"color_stability\":0.75,\"color_max_shift\":0.1}");
  CHECK(!slopfab::lcs_sampling_settings(s).active());
  slopfab::SamplingStep step;
  step.frames = 12; step.height = step.width = 6; step.row_width = 96;
  step.count = size_t(step.frames) * 9 * 96;
  std::vector<float> values(step.count, .123f);
  for (size_t r = 0; r < step.count / 96; ++r) {
    values[r * 96] = .1f; values[r * 96 + 1] = .3f; values[r * 96 + 2] = .5f;
  }
  const auto original = values;
  step.denoised = values.data();
  std::vector<float> mask(9, 1); mask[0] = 0;
  slopfab::apply_lcs_output_stability(d, s, mask, step, 7);
  CHECK(values != original);
  for (size_t r = 0; r < step.count / 96; ++r) {
    const bool held = r < 7 * 9 || r % 9 == 0;
    float light_delta = 0;
    for (int i = 0; i < 96; ++i) {
      if (held || i >= 3)
        CHECK(values[r * 96 + i] == original[r * 96 + i]);
      if (i < 3) {
        light_delta += values[r * 96 + i] - original[r * 96 + i];
        CHECK(std::abs(values[r * 96 + i] - original[r * 96 + i]) < .1f);
      }
    }
    CHECK_NEAR(light_delta, 0.f, 1e-6);
  }
  // A positive luminance spike is bounded, not a blanket blur or a chroma edit.
  s.color_stability = 0; s.edge_stability = 1;
  values = original;
  const size_t spike = (11 * 9 + 4) * 96;
  for (int i = 0; i < 3; ++i)
    values[spike + i] += .5f;
  const auto before_edge = values;
  slopfab::apply_lcs_output_stability(d, s, {}, step, 7);
  for (int i = 0; i < 3; ++i)
    CHECK_NEAR(values[spike + i], before_edge[spike + i] - .05f, 1e-6);
  for (size_t i = 0; i < values.size(); ++i)
    if (i < spike || i >= spike + 3)
      CHECK_NEAR(values[i], before_edge[i], 1e-6);
  CHECK(rejects([&] { slopfab::apply_lcs_output_stability(d, s, {1, 1}, step, 7); }));
  CHECK(rejects([&] { slopfab::apply_lcs_output_stability(d, s, {}, step, 13); }));
  auto bad = std::make_shared<slopfab::LcsData>(fixture());
  CHECK(rejects([&] { slopfab::apply_lcs_output_stability(bad, s, {}, step); }));
  const auto path = (std::filesystem::temp_directory_path() / "slopfab-lcs-stability-test.safetensors").string();
  d->save(path);
  auto loaded = slopfab::LcsData::load(path);
  CHECK(loaded->color_reference == d->color_reference && loaded->edge_reference == d->edge_reference);
  std::filesystem::remove(path);
  d->color_reference[0] = 1;
  CHECK(rejects([&] { d->validate(); }));
  CHECK(rejects([] { slopfab::parse_lcs_settings("{\"color_stability\":1.1}"); }));
  CHECK(rejects([] { slopfab::parse_lcs_settings("{\"edge_stability\":-1}"); }));
}

SLOPFAB_TEST(lcs_output_color_reference_uses_global_temporal_phase) {
  auto d = std::make_shared<slopfab::LcsData>(fixture());
  d->basis.assign(288, 0); d->mean.assign(96, 0);
  for (int i = 0; i < 3; ++i)
    d->basis[i * 3 + i] = 1;
  d->anchors = {1,0,0, 0,0,1, 0,1,0, 1,0,1, 0,1,1, 1,1,0, 0,0,0, 1,1,1};
  d->color_reference.assign(30, 0);
  for (int phase = 0; phase < 5; ++phase)
    for (int q = 0; q < 3; ++q)
      d->color_reference[phase * 6 + q] = .04f * phase / std::sqrt(18.f);
  slopfab::SamplingStep step;
  step.frames = 12; step.height = step.width = 2; step.row_width = 96; step.count = 12 * 96;
  std::vector<float> values(step.count, 0);
  for (int f = 0; f < 12; ++f) {
    values[f * 96] = .5f + .02f * (f % 5);
    values[f * 96 + 1] = values[f * 96 + 2] = .5f;
  }
  const auto original = values;
  step.denoised = values.data();
  slopfab::LcsSettings settings; settings.color_stability = 1;
  // A nonmultiple-of-five prefix catches resetting the phase at the new window.
  slopfab::apply_lcs_output_stability(d, settings, {}, step, 7);
  for (size_t i = 0; i < values.size(); ++i)
    CHECK_NEAR(values[i], original[i], 1e-6);
}

SLOPFAB_TEST(lcs_stability_covers_42_saved_keystone_continuations) {
  using namespace slopfab;
  auto data = std::make_shared<LcsData>(fixture());
  data->color_reference.assign(30, 0);
  data->edge_reference.assign(5, 0);
  // Output correction must not stop at a sampling-step limit.
  auto settings = parse_lcs_settings(R"({"color_stability":0.75,"edge_stability":0.75,"color_max_shift":0.1,"start_step":100,"end_step":101})");
  CHECK(!lcs_sampling_settings(settings).active());
  LatentFixture original_file, saved_file;
  original_file.write(73);
  const auto original = LatentClip::load(original_file.path.string());
  auto previous = original;
  const auto original_video = original->video_rows;
  Keystone keystone{original, true, 4};
  const size_t slice = size_t(original->layout().rows_per_frame()) * 96;
  int corrected_slices = 0;
  for (int shot = 0; shot < 42; ++shot) {
    const auto plan = plan_continuation(*previous, 22, 68);
    auto constraint = make_keystone_constraint(*previous, plan, keystone);
    CHECK(constraint.capstone.size() == slice && constraint.capstone_start_step == 4);
    CHECK(std::equal(original_video.begin(), original_video.begin() + slice,
                     constraint.video.original.begin()));
    // Remaining locked motion context must come from the saved corrected source.
    const auto tail = previous->video_rows.end() - size_t(plan.overlap_video_latents) * slice;
    CHECK(std::equal(tail + slice, tail + size_t(plan.held_video_latents) * slice,
                     constraint.video.original.begin() + slice));
    std::vector<float> video(constraint.video.target_values_per_channel, 0);
    for (size_t r = 0; r < video.size() / 96; ++r) {
      const float light = r % 2 ? .4f : 0.f;
      video[r * 96] = .1f + light;
      video[r * 96 + 1] = .3f + light;
      video[r * 96 + 2] = .5f + light;
      video[r * 96 + 7] = float(shot + 1); // Structure outside the color subspace.
    }
    constraint.video.capture_noise(video.data(), video.size());
    constraint.video.apply(video.data(), video.size(), 0, 24);
    constraint.restore_video(video);
    std::vector<float> audio(size_t(plan.window_audio_latents) * 64, float(shot));
    auto joined = join_continuation(*previous, plan, video, audio);
    const auto before = joined;
    const auto layout = joined.layout();
    const int preserved = previous->layout().num_latent_frames - plan.overlap_video_latents + plan.held_video_latents;
    CHECK(preserved == previous->layout().num_latent_frames);
    CHECK(layout.num_latent_frames - preserved == 20);
    SamplingStep step;
    step.frames = layout.num_latent_frames; step.height = layout.latent_height; step.width = layout.latent_width;
    step.row_width = 96; step.count = joined.video_rows.size(); step.denoised = joined.video_rows.data();
    apply_lcs_output_stability(data, settings, {}, step, preserved);
    CHECK(std::equal(previous->video_rows.begin(), previous->video_rows.end(), joined.video_rows.begin()));
    CHECK(joined.audio_rows == before.audio_rows);
    for (int f = preserved; f < layout.num_latent_frames; ++f) {
      const size_t start = size_t(f) * slice;
      // Every newly generated slice, including the ramp and the very last slice, is corrected.
      CHECK(!std::equal(before.video_rows.begin() + start, before.video_rows.begin() + start + slice,
                        joined.video_rows.begin() + start));
      for (size_t p = 0; p < slice / 96; ++p)
        CHECK(joined.video_rows[start + p * 96 + 7] == before.video_rows[start + p * 96 + 7]);
      ++corrected_slices;
    }
    joined.save(saved_file.path.string());
    previous = LatentClip::load(saved_file.path.string());
    CHECK(previous->video_rows == joined.video_rows && previous->audio_rows == joined.audio_rows);
    CHECK(previous->frames == 73 + (shot + 1) * 68);
  }
  CHECK(corrected_slices == 840 && previous->frames == 2929);
  CHECK(original->video_rows == original_video);
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
