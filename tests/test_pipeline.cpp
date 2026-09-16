// Request-to-plan resolution.
//
// Everything here is arithmetic a user can trigger from the command line
// before any weight is read, so it is the first thing that should refuse a bad
// request — and the last thing that should silently produce a subtly wrong
// geometry.

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "harness.h"
#include "slopfab/pipeline.h"
#include "slopfab/generate.h"
#include "slopfab/attention_mode.h"
#include "slopfab/cuda/cuda_toolkit.h"
#include "slopfab/sampler/scheduler.h"
#include "slopfab/safetensors_write.h"

namespace {

std::filesystem::path scratch_path(const char* name) {
  return std::filesystem::temp_directory_path() / name;
}

// Writes `content` and stamps the file with a last-write time `age_seconds`
// before now, so a rewrite is distinguishable from its predecessor even on a
// filesystem whose timestamp granularity is coarser than the test.
void write_file(const std::filesystem::path& path, const std::string& content,
                int age_seconds) {
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
  }
  std::error_code ec;
  std::filesystem::last_write_time(
      path, std::filesystem::file_time_type::clock::now() - std::chrono::seconds(age_seconds), ec);
}

// `throws` takes a plain function pointer, so a capturing lambda will not do.
bool rejects_frame_count(int frames) {
  try {
    slopfab::GenerateRequest r;
    r.num_frames = frames;
    slopfab::resolve_plan(r);
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

slopfab::GenerateRequest base_request() {
  slopfab::GenerateRequest r;
  r.prompt = "a test";
  return r;
}

SLOPFAB_TEST(pipeline_plan_default) {
  const slopfab::GenerateRequest r = base_request();
  const slopfab::GeneratePlan p = slopfab::resolve_plan(r);

  CHECK(p.canvas_height == 768);
  CHECK(p.canvas_width == 1344);
  CHECK(p.aligned_frames == 124);
  CHECK(p.layout.latent_height == 48);
  CHECK(p.layout.latent_width == 84);
  CHECK(p.layout.num_latent_frames == 37);
  CHECK(p.layout.num_video_rows == 37296);
  CHECK(p.layout.num_audio_rows == 414);

  // The prompt has not been tokenised yet, so the layout carries no text rows
  // and the sequence length is the media part alone.
  CHECK(p.layout.num_text == 0);
  CHECK(p.sequence_length_without_text() == 37710);

  // 50 grid points, 49 model evaluations: the terminal sigma gets none.
  CHECK(p.video_sigmas.size() == 50);
  CHECK(p.video_timesteps.size() == 49);
  CHECK(p.num_model_evaluations() == 49);
  CHECK(p.audio_timesteps.size() == p.video_timesteps.size());

  // The shift maps 0 to 0 and 1 to 1 exactly, at both shifts.
  CHECK_NEAR(p.video_sigmas.front(), 1.0, 0.0);
  CHECK_NEAR(p.video_sigmas.back(), 0.0, 0.0);
  CHECK_NEAR(p.audio_sigmas.front(), 1.0, 0.0);
  CHECK_NEAR(p.audio_sigmas.back(), 0.0, 0.0);

  // t = 1 - sigma, so the first timestep is 0 (fully noisy) and t = 1 is clean
  // — the opposite direction from diffusers' flow-match schedulers.
  CHECK_NEAR(p.video_timesteps.front(), 0.0, 1e-7);
  CHECK(p.video_timesteps.back() < 1.0);

  // Both grids are strictly decreasing in sigma.
  bool decreasing = true;
  for (size_t i = 1; i < p.video_sigmas.size(); ++i) {
    decreasing = decreasing && p.video_sigmas[i] < p.video_sigmas[i - 1];
  }
  CHECK(decreasing);

  // The video shift compresses the grid far harder than the audio one, so the
  // last non-zero video sigma sits well above the audio one. If these ever
  // matched, the two schedulers would have been built with the same shift.
  const float last_video = p.video_sigmas[p.video_sigmas.size() - 2];
  const float last_audio = p.audio_sigmas[p.audio_sigmas.size() - 2];
  CHECK_NEAR(last_video, 0.2, 1e-6);
  CHECK_NEAR(last_audio, 1.0 / 17.0, 1e-6);
  CHECK(last_video > last_audio);
}

SLOPFAB_TEST(attention_mode_parse_name_and_backend_contract) {
  using slopfab::AttentionMode;
  using slopfab::DeviceBackend;
  const AttentionMode modes[] = {
      AttentionMode::kNone, AttentionMode::kFlash2, AttentionMode::kSage2,
      AttentionMode::kSol, AttentionMode::kSolExperimental, AttentionMode::kExact};
  for (AttentionMode mode : modes) {
    AttentionMode parsed = AttentionMode::kNone;
    CHECK(slopfab::parse_attention_mode(slopfab::attention_mode_name(mode), &parsed));
    CHECK(parsed == mode);
    CHECK(slopfab::attention_mode_supported(DeviceBackend::kCuda, mode));
    CHECK(slopfab::attention_mode_supported(DeviceBackend::kVulkan, mode) ==
          (mode == AttentionMode::kExact || mode == AttentionMode::kFlash2 ||
           mode == AttentionMode::kSage2));
    CHECK(slopfab::generation_backend_supported(DeviceBackend::kVulkan,
          slopfab::LatentSource::kDenoise, mode) ==
          slopfab::attention_mode_supported(DeviceBackend::kVulkan, mode));
  }

  AttentionMode unchanged = AttentionMode::kSol;
  CHECK(!slopfab::parse_attention_mode("flash3", &unchanged));
  CHECK(unchanged == AttentionMode::kSol);
  CHECK(!slopfab::parse_attention_mode("exact ", &unchanged));
  CHECK(!slopfab::parse_attention_mode("exact", nullptr));
  CHECK(std::string(slopfab::attention_mode_name(static_cast<AttentionMode>(999))) == "unknown");
  CHECK(!slopfab::attention_mode_supported(DeviceBackend::kVulkan,
                                           static_cast<AttentionMode>(999)));
}

SLOPFAB_TEST(generation_backend_contract) {
  using slopfab::DeviceBackend;
  using slopfab::LatentSource;
  using slopfab::AttentionMode;
  CHECK(slopfab::generation_backend_supported(DeviceBackend::kCuda,
                                              LatentSource::kDenoise,
                                              AttentionMode::kFlash2));
  CHECK(slopfab::generation_backend_supported(DeviceBackend::kCuda,
                                              LatentSource::kSyntheticNoise,
                                              AttentionMode::kSage2));
  CHECK(slopfab::generation_backend_supported(DeviceBackend::kVulkan,
                                              LatentSource::kDenoise,
                                              AttentionMode::kExact));
  CHECK(slopfab::generation_backend_supported(DeviceBackend::kVulkan,
                                              LatentSource::kSyntheticNoise,
                                              AttentionMode::kExact));
  CHECK(slopfab::generation_backend_supported(DeviceBackend::kVulkan,
                                               LatentSource::kSyntheticNoise,
                                               AttentionMode::kFlash2));
  CHECK(slopfab::generation_backend_supported(DeviceBackend::kVulkan,
                                               LatentSource::kSyntheticNoise,
                                               AttentionMode::kSage2));
  slopfab::RunOptions defaults;
  CHECK(defaults.inference_backend == DeviceBackend::kCuda);
}

SLOPFAB_TEST(pipeline_plan_aspect_and_frames) {
  slopfab::GenerateRequest r = base_request();
  r.aspect_w = 9;
  r.aspect_h = 16;
  r.num_frames = 240;
  const slopfab::GeneratePlan p = slopfab::resolve_plan(r);

  // The portrait canvas is the landscape one transposed, and the 10-second
  // request is the worked example in the spec.
  CHECK(p.canvas_height == 1344);
  CHECK(p.canvas_width == 768);
  CHECK(p.aligned_frames == 243);
  CHECK(p.layout.num_latent_frames == 72);
  CHECK(p.layout.rows_per_frame() == 1008);
  CHECK(p.layout.num_video_rows == 72576);
  CHECK(p.layout.num_audio_latents == 405);
  CHECK(p.sequence_length_without_text() == 73386);
  CHECK_NEAR(p.duration_seconds, 243.0 / 24.0, 1e-12);

  // A square request stays square and keeps the same row count per frame.
  r.aspect_w = 1;
  r.aspect_h = 1;
  const slopfab::GeneratePlan sq = slopfab::resolve_plan(r);
  CHECK(sq.canvas_height == 768 && sq.canvas_width == 768);
  CHECK(sq.layout.latent_height == 48 && sq.layout.latent_width == 48);
  CHECK(sq.layout.rows_per_frame() == 24 * 24);
}

SLOPFAB_TEST(pipeline_plan_still_image) {
  slopfab::GenerateRequest r = base_request();
  r.aspect_w = 1;
  r.aspect_h = 1;
  r.num_frames = 1;  // Invalid for video, deliberately irrelevant for a still.
  r.still_image = true;
  const slopfab::GeneratePlan p = slopfab::resolve_plan(r);

  CHECK(p.aligned_frames == 1);
  CHECK_NEAR(p.duration_seconds, 1.0 / 24.0, 1e-12);
  CHECK(p.layout.num_latent_frames == 1);
  CHECK(p.layout.latent_height == 48);
  CHECK(p.layout.latent_width == 48);
  CHECK(p.layout.rows_per_frame() == 576);
  CHECK(p.layout.num_video_rows == 576);
  CHECK(p.layout.num_audio_latents == 0);
  CHECK(p.layout.num_audio_rows == 0);
  CHECK(p.sequence_length_without_text() == 576);

  const std::string description = slopfab::describe_plan(r, p);
  CHECK(description.find("still image") != std::string::npos);
  CHECK(description.find("1 output") != std::string::npos);

  // Turning the mode off restores ordinary validation; the request field was
  // not silently rewritten by resolving the still plan.
  r.still_image = false;
  CHECK(rejects_frame_count(r.num_frames));
}

SLOPFAB_TEST(pipeline_plan_explicit_resolution) {
  // The property the whole flag rests on: naming the canvas the default
  // already produces must reproduce the default plan exactly. If this drifts,
  // adding the option moved the default for everyone who never passed it.
  const slopfab::GeneratePlan derived = slopfab::resolve_plan(base_request());

  slopfab::GenerateRequest named = base_request();
  named.canvas_width = 1344;
  named.canvas_height = 768;
  const slopfab::GeneratePlan explicit_plan = slopfab::resolve_plan(named);

  CHECK(explicit_plan.canvas_width == derived.canvas_width);
  CHECK(explicit_plan.canvas_height == derived.canvas_height);
  CHECK(explicit_plan.layout.latent_width == derived.layout.latent_width);
  CHECK(explicit_plan.layout.latent_height == derived.layout.latent_height);
  CHECK(explicit_plan.layout.total_rows() == derived.layout.total_rows());

  // An explicit canvas wins over the aspect rather than being reconciled with
  // it — the two disagree here on purpose, and the canvas is what survives.
  slopfab::GenerateRequest both = base_request();
  both.aspect_w = 1;
  both.aspect_h = 1;
  both.canvas_width = 1024;
  both.canvas_height = 512;
  const slopfab::GeneratePlan p = slopfab::resolve_plan(both);
  CHECK(p.canvas_width == 1024 && p.canvas_height == 512);
  CHECK(p.layout.latent_width == 64 && p.layout.latent_height == 32);

  // One axis alone is not an explicit canvas, so it falls back to the aspect
  // rather than silently generating a 1344x0 plan.
  slopfab::GenerateRequest half = base_request();
  half.canvas_width = 1024;
  const slopfab::GeneratePlan fell_back = slopfab::resolve_plan(half);
  CHECK(fell_back.canvas_width == 1344 && fell_back.canvas_height == 768);

  // Over the trained area is allowed here and refused nowhere — the warning
  // lives in the CLI, and the plan resolves so a caller can see the cost.
  slopfab::GenerateRequest big = base_request();
  big.canvas_width = 1920;
  big.canvas_height = 1088;
  const slopfab::GeneratePlan large = slopfab::resolve_plan(big);
  CHECK(large.layout.total_rows() > derived.layout.total_rows());

  // A canvas off the 32-grid is rejected, not rounded.
  CHECK(::slopfab::test::throws([] {
    slopfab::GenerateRequest r;
    r.canvas_width = 1350;
    r.canvas_height = 768;
    slopfab::resolve_plan(r);
  }));
}

SLOPFAB_TEST(pipeline_plan_rejects_bad_requests) {
  // A one-point grid has no model evaluation at all.
  CHECK(::slopfab::test::throws([] {
    slopfab::GenerateRequest r;
    r.num_inference_steps = 1;
    slopfab::resolve_plan(r);
  }));

  // Outside 1:4 .. 4:1.
  CHECK(::slopfab::test::throws([] {
    slopfab::GenerateRequest r;
    r.aspect_w = 5;
    r.aspect_h = 1;
    slopfab::resolve_plan(r);
  }));

  CHECK(::slopfab::test::throws([] {
    slopfab::GenerateRequest r;
    r.num_frames = 0;
    slopfab::resolve_plan(r);
  }));

  // Too few frames to decode. `F = 5k + 2` and the video decoder needs 7-token
  // temporal windows, so `F >= 7` means at least 22 aligned frames. Alignment
  // rounds *up*, so only requests that align to 5 — that is, 1 through 5 — are
  // actually unsatisfiable; 6 already aligns to 22. Rejected up front rather
  // than after 9 GB of weights have been uploaded.
  for (int frames : {1, 2, 3, 4, 5}) {
    CHECK_MSG(rejects_frame_count(frames),
              "num_frames=%d should be rejected: it yields fewer than 7 latent frames", frames);
  }

  // 6 is the smallest request that resolves, and it aligns to exactly 7 latent
  // frames.
  CHECK(!rejects_frame_count(6));
  CHECK(!rejects_frame_count(21));
  CHECK(!rejects_frame_count(22));
  slopfab::GenerateRequest ok;
  ok.num_frames = 6;
  const slopfab::GeneratePlan p = slopfab::resolve_plan(ok);
  CHECK(p.aligned_frames == 22);
  CHECK(p.layout.num_latent_frames == 7);
}

SLOPFAB_TEST(pipeline_schedules_stay_paired) {
  // The denoise loop zips the video and audio timestep lists while iterating
  // the video one, so a length divergence caused by unique_consecutive
  // collapsing the two shifted grids differently would silently truncate the
  // run. resolve_plan asserts they match; check it holds across the practical
  // range rather than only at the default.
  for (int steps = 2; steps <= 120; ++steps) {
    slopfab::GenerateRequest r = base_request();
    r.num_inference_steps = steps;
    const slopfab::GeneratePlan p = slopfab::resolve_plan(r);
    CHECK_MSG(p.video_timesteps.size() == p.audio_timesteps.size(),
              "steps=%d: video %zu vs audio %zu timesteps", steps, p.video_timesteps.size(),
              p.audio_timesteps.size());
    CHECK_MSG(p.num_model_evaluations() >= 1, "steps=%d collapsed to no evaluations", steps);
  }
}

// The runner used to rebuild both schedulers from its own copies of 12.0 and
// 3.0. It now rebuilds them from the plan, so this checks the plan carries
// everything that reconstruction needs and that doing so reproduces the plan's
// own grids exactly — not nearly, exactly, because the sample is bit-identical
// only if the loop steps the identical sigmas.
SLOPFAB_TEST(pipeline_plan_carries_its_schedule_inputs) {
  for (int steps : {2, 7, 29, 50, 120}) {
    slopfab::GenerateRequest r = base_request();
    r.num_inference_steps = steps;
    const slopfab::GeneratePlan p = slopfab::resolve_plan(r);

    CHECK(p.num_inference_steps == steps);
    CHECK_NEAR(p.video_sigma_shift, 12.0, 0.0);
    CHECK_NEAR(p.audio_sigma_shift, 3.0, 0.0);

    slopfab::sampler::FlowScheduler video(p.video_sigma_shift);
    slopfab::sampler::FlowScheduler audio(p.audio_sigma_shift);
    video.set_timesteps(p.num_inference_steps);
    audio.set_timesteps(p.num_inference_steps);

    CHECK(video.sigmas() == p.video_sigmas);
    CHECK(audio.sigmas() == p.audio_sigmas);
    CHECK(video.timesteps() == p.video_timesteps);
    CHECK(audio.timesteps() == p.audio_timesteps);
  }

  // A plan that never went through resolve_plan still names the shipped shifts
  // rather than zero, so the defaults cannot quietly become shift-free.
  const slopfab::GeneratePlan fresh;
  CHECK_NEAR(fresh.video_sigma_shift, slopfab::kVideoSigmaShift, 0.0);
  CHECK_NEAR(fresh.audio_sigma_shift, slopfab::kAudioSigmaShift, 0.0);
}

SLOPFAB_TEST(pipeline_viggle_schedule_uses_model_identity) {
  const std::vector<slopfab::TensorWrite> tensors = {
      {"adaln_t_table", {1, 8}, std::vector<float>(8)},
      {"blocks.0.adaln_proj.linear.weight", {1, 8}, std::vector<float>(8)}};
  const std::vector<float> expected = {1.0f, 6.0f / 7.0f, 0.6f, 0.0f};
  for (const char* filename : {"slopfab_plan_renamed.safetensors",
                               "slopfab_plan_Viggle-Animate.safetensors"}) {
    const auto path = scratch_path(filename);
    struct Cleanup {
      std::filesystem::path path;
      ~Cleanup() { std::error_code ec; std::filesystem::remove(path, ec); }
    } cleanup{path};
    slopfab::GenerateRequest r = base_request();
    r.transformer_path = path.string();
    r.num_inference_steps = 4;
    const bool renamed = std::string(filename).find("renamed") != std::string::npos;
    slopfab::write_safetensors(r.transformer_path, tensors,
        renamed ? std::map<std::string, std::string>{{"source", "Viggle/Viggle-Animate"}}
                : std::map<std::string, std::string>{});

    const auto p = slopfab::resolve_plan(r);
    CHECK_NEAR(p.video_sigma_shift, 3.0, 0.0);
    CHECK_NEAR(p.audio_sigma_shift, 3.0, 0.0);
    CHECK(p.num_inference_steps == 4);
    CHECK(p.num_model_evaluations() == 3);
    CHECK_CLOSE(expected, p.video_sigmas, 1e-6, "Viggle three-pass sigma grid");
    CHECK(p.video_sigmas == p.audio_sigmas);
    for (size_t i = 0; i < p.video_timesteps.size(); ++i)
      CHECK_NEAR(p.video_timesteps[i], 1.0f - expected[i], 1e-6);

    // Both GPU runners reconstruct from these fields, so they must step the
    // exact grid the planner reports, also with other user-selected counts.
    for (int steps : {4, 10}) {
      r.num_inference_steps = steps;
      const auto plan = slopfab::resolve_plan(r);
      slopfab::sampler::FlowScheduler video(plan.video_sigma_shift);
      video.set_timesteps(plan.num_inference_steps, r.schedule);
      CHECK(video.sigmas() == plan.video_sigmas);
      CHECK(video.timesteps() == plan.video_timesteps);
      CHECK_NEAR(plan.video_sigma_shift, 3.0, 0.0);
      CHECK(plan.num_model_evaluations() == steps - 1);
    }

    // Explicit non-Viggle provenance takes priority over a suggestive name.
    // Replacing a file at the same path must not leave a cached Viggle shift.
    slopfab::write_safetensors(r.transformer_path, tensors, {{"source", "MiniMax/H3"}});
    r.num_inference_steps = 4;
    const auto h3 = slopfab::resolve_plan(r);
    CHECK_NEAR(h3.video_sigma_shift, 12.0, 0.0);
    CHECK_CLOSE((std::vector<float>{1.0f, 0.96f, 6.0f / 7.0f, 0.0f}),
                h3.video_sigmas, 1e-6, "H3 four-boundary grid is unchanged");
  }
}

SLOPFAB_TEST(pipeline_describe_plan) {
  const slopfab::GenerateRequest r = base_request();
  const slopfab::GeneratePlan p = slopfab::resolve_plan(r);
  const std::string text = slopfab::describe_plan(r, p);
  CHECK(text.find("768 x 1344") != std::string::npos);
  CHECK(text.find("37710") != std::string::npos);
  CHECK(text.find("49 model evaluations") != std::string::npos);

  // Every number in the summary comes from the plan, not from the request that
  // produced it. The grid the header prints and the grid the loop integrates
  // have to be the same object, and a request field read here is the one way
  // back to two of them.
  slopfab::GeneratePlan renumbered = p;
  renumbered.num_inference_steps = 7;
  const std::string retold = slopfab::describe_plan(r, renumbered);
  CHECK(retold.find("7 grid points") != std::string::npos);
  CHECK(retold.find("50 grid points") == std::string::npos);
}

SLOPFAB_TEST(pipeline_reference_image_limit) {
  slopfab::GenerateRequest r = base_request();
  r.reference_image_paths = {"subject.png", "style.png", "scene.png"};
  const slopfab::GeneratePlan p = slopfab::resolve_plan(r);
  CHECK(slopfab::describe_plan(r, p).find("3 (Ref2VA, ordered)") != std::string::npos);
  CHECK(r.reference_image_paths[0] == "subject.png");
  CHECK(r.reference_image_paths[1] == "style.png");
  CHECK(r.reference_image_paths[2] == "scene.png");

  // Nine is accepted; the tenth is rejected before any image I/O or GPU work.
  r.reference_image_paths.assign(9, "image.png");
  CHECK(slopfab::describe_plan(r, slopfab::resolve_plan(r)).find("9 (Ref2VA, ordered)") !=
        std::string::npos);
  r.reference_image_paths.push_back("too-many.png");
  bool rejected = false;
  try {
    slopfab::resolve_plan(r);
  } catch (const std::exception&) {
    rejected = true;
  }
  CHECK(rejected);
}

// The whole point of keying on identity rather than on name: a reference image
// overwritten in place between two `--reuse-models` generations must not be
// served from the first image's cache entry. Before this, both runs produced
// the same key and the second silently rendered the first image's conditioning.
SLOPFAB_TEST(conditioning_cache_key_detects_overwritten_reference_image) {
  const std::filesystem::path ref = scratch_path("slopfab_cachekey_ref.ppm");

  slopfab::GenerateRequest r = base_request();
  r.text_encoder_path = "encoder.safetensors";
  r.tokenizer_path = "tokenizer.json";
  r.reference_image_paths = {ref.string()};

  write_file(ref, "first image bytes", 120);
  const std::string before = slopfab::conditioning_cache_key(r);

  // Same path, same request, different content. Different size *and* a
  // different mtime, which is what any real overwrite produces.
  write_file(ref, "second image bytes, a different length entirely", 0);
  const std::string after = slopfab::conditioning_cache_key(r);
  CHECK(before != after);

  // Rewriting the identical bytes at a later mtime is the *same* image, and a
  // content-keyed entry says so rather than throwing away a valid encode. This
  // is the direction stat keying gets wrong in the harmless way.
  write_file(ref, "second image bytes, a different length entirely", 60);
  const std::string rewritten = slopfab::conditioning_cache_key(r);
  CHECK(rewritten == after);

  // And nothing else moved: asking twice with the file untouched is a hit.
  CHECK(slopfab::conditioning_cache_key(r) == rewritten);

  // The reference key sees the same overwrite, because the VAE keyframe encode
  // is cached against it.
  r.video_vae_path = "video_vae.safetensors";
  const std::string ref_key = slopfab::reference_cache_key(r);
  write_file(ref, "third", 0);
  CHECK(slopfab::reference_cache_key(r) != ref_key);

  // Deleting the file is a change too, rather than "unchanged since last time".
  const std::string present = slopfab::conditioning_cache_key(r);
  std::filesystem::remove(ref);
  CHECK(slopfab::conditioning_cache_key(r) != present);

  // Checkpoints keep the cheaper stat identity, and it is live: a rebuilt
  // encoder at the same path invalidates on mtime alone. Hashing 27 GB here
  // would cost more than everything the key protects.
  const std::filesystem::path ckpt = scratch_path("slopfab_cachekey_ckpt.bin");
  r.reference_image_paths.clear();
  r.text_encoder_path = ckpt.string();
  write_file(ckpt, "weights", 300);
  const std::string old_ckpt = slopfab::conditioning_cache_key(r);
  write_file(ckpt, "weights", 0);
  CHECK(slopfab::conditioning_cache_key(r) != old_ckpt);
  std::filesystem::remove(ckpt);
}

// The one the previous test cannot prove. `write_file` above manufactures its
// mtime difference, so it shows the key reads mtime — not that a real in-place
// overwrite is caught. Here the replacement is same-size and the timestamp is
// put back exactly, which is what `copy`, `robocopy /COPY:T`, `xcopy /K`,
// rsync --times and a good deal of image tooling actually do. Only hashing the
// contents distinguishes these two files.
SLOPFAB_TEST(reference_key_detects_overwrite_that_preserves_size_and_mtime) {
  const std::filesystem::path ref = scratch_path("slopfab_cachekey_samestamp.ppm");

  slopfab::GenerateRequest r = base_request();
  r.video_vae_path = "video_vae.safetensors";
  r.text_encoder_path = "encoder.safetensors";
  r.reference_image_paths = {ref.string()};

  write_file(ref, "PPM-payload-version-one", 90);
  const std::filesystem::file_time_type stamp = std::filesystem::last_write_time(ref);
  const std::string before = slopfab::conditioning_cache_key(r);
  const std::string before_ref = slopfab::reference_cache_key(r);

  // Same byte count, different bytes, and the timestamp restored to the exact
  // value it had — so (size, mtime) is identical across the overwrite.
  {
    std::ofstream out(ref, std::ios::binary | std::ios::trunc);
    const std::string replacement = "PPM-payload-version-two";
    CHECK(replacement.size() == std::string("PPM-payload-version-one").size());
    out.write(replacement.data(), static_cast<std::streamsize>(replacement.size()));
  }
  std::filesystem::last_write_time(ref, stamp);

  // The metadata really is unchanged: this is the case that used to slip past.
  CHECK(std::filesystem::last_write_time(ref) == stamp);
  CHECK(std::filesystem::file_size(ref) == std::string("PPM-payload-version-one").size());

  CHECK(slopfab::conditioning_cache_key(r) != before);
  CHECK(slopfab::reference_cache_key(r) != before_ref);

  // And it is still stable when nothing moves at all, so the hash has not just
  // made every lookup a miss.
  const std::string settled = slopfab::conditioning_cache_key(r);
  CHECK(slopfab::conditioning_cache_key(r) == settled);
  CHECK(slopfab::reference_cache_key(r) == slopfab::reference_cache_key(r));

  // A byte-for-byte identical rewrite at a different mtime is the same image,
  // and content keying says so — which stat keying could not.
  write_file(ref, "PPM-payload-version-two", 5);
  CHECK(slopfab::conditioning_cache_key(r) == settled);

  std::filesystem::remove(ref);
}

// Sharing one hash between the two keys has to be a pure saving: the keys it
// produces must be the ones the self-hashing form produces, or a run that took
// the fast path would miss a cache the slow path would have hit.
SLOPFAB_TEST(shared_reference_identities_agree_with_hashing_twice) {
  const std::filesystem::path one = scratch_path("slopfab_cachekey_share1.ppm");
  const std::filesystem::path two = scratch_path("slopfab_cachekey_share2.ppm");
  write_file(one, "first reference payload", 40);
  write_file(two, "second reference payload, longer", 40);

  slopfab::GenerateRequest r = base_request();
  r.text_encoder_path = "encoder.safetensors";
  r.video_vae_path = "video_vae.safetensors";
  r.reference_image_paths = {one.string(), two.string()};

  const std::vector<std::string> identities = slopfab::reference_image_identities(r);
  CHECK(identities.size() == 2);
  CHECK(identities[0] != identities[1]);

  CHECK(slopfab::conditioning_cache_key(r, identities) == slopfab::conditioning_cache_key(r));
  CHECK(slopfab::reference_cache_key(r, identities) == slopfab::reference_cache_key(r));

  // A list that does not match the request falls back to hashing rather than
  // keying off a stale snapshot — the failure mode that would otherwise reuse
  // the wrong image silently, which is the whole bug class this file guards.
  const std::vector<std::string> truncated{identities[0]};
  CHECK(slopfab::conditioning_cache_key(r, truncated) == slopfab::conditioning_cache_key(r));
  CHECK(slopfab::reference_cache_key(r, truncated) == slopfab::reference_cache_key(r));
  CHECK(slopfab::conditioning_cache_key(r, {}) == slopfab::conditioning_cache_key(r));

  // The empty case is not a special case: no references, and both forms agree.
  slopfab::GenerateRequest text_only = base_request();
  text_only.text_encoder_path = "encoder.safetensors";
  CHECK(slopfab::reference_image_identities(text_only).empty());
  CHECK(slopfab::conditioning_cache_key(text_only, {}) ==
        slopfab::conditioning_cache_key(text_only));

  // And the shared snapshot still tracks content: rehashing after an overwrite
  // gives different identities and therefore different keys.
  const std::string before = slopfab::conditioning_cache_key(r, identities);
  write_file(one, "first reference payload, edited", 40);
  const std::vector<std::string> rehashed = slopfab::reference_image_identities(r);
  CHECK(slopfab::conditioning_cache_key(r, rehashed) != before);

  std::filesystem::remove(one);
  std::filesystem::remove(two);
}

SLOPFAB_TEST(cache_keys_separate_their_inputs) {
  const std::filesystem::path a = scratch_path("slopfab_cachekey_a.bin");
  const std::filesystem::path b = scratch_path("slopfab_cachekey_b.bin");
  write_file(a, "aaaa", 60);
  write_file(b, "aaaa", 60);

  slopfab::GenerateRequest r = base_request();
  r.text_encoder_path = a.string();
  r.tokenizer_path = a.string();
  r.video_vae_path = a.string();
  r.reference_image_paths = {a.string()};

  // Two files with identical size and mtime still differ, because the path is
  // part of the identity.
  slopfab::GenerateRequest other = r;
  other.reference_image_paths = {b.string()};
  CHECK(slopfab::conditioning_cache_key(r) != slopfab::conditioning_cache_key(other));

  // The prompt drives conditioning and nothing else. A prompt sweep must not
  // invalidate the reference encode or the tokenizer.
  slopfab::GenerateRequest reworded = r;
  reworded.prompt = r.prompt + " at night";
  CHECK(slopfab::conditioning_cache_key(reworded) != slopfab::conditioning_cache_key(r));
  CHECK(slopfab::reference_cache_key(reworded) == slopfab::reference_cache_key(r));
  CHECK(slopfab::tokenizer_cache_key(reworded) == slopfab::tokenizer_cache_key(r));

  // The video VAE is an input to the reference encode but not to conditioning.
  slopfab::GenerateRequest other_vae = r;
  other_vae.video_vae_path = b.string();
  CHECK(slopfab::reference_cache_key(other_vae) != slopfab::reference_cache_key(r));
  CHECK(slopfab::conditioning_cache_key(other_vae) == slopfab::conditioning_cache_key(r));

  // A cached embedding never crosses conditioner implementation or arithmetic
  // authority, even when the exact implementations currently agree bytewise.
  const auto cuda_shipped = slopfab::ConditionerAuthority::kCudaShipped;
  const auto cuda_exact = slopfab::ConditionerAuthority::kCudaExact;
  const auto vulkan_exact = slopfab::ConditionerAuthority::kVulkanExact;
  CHECK(slopfab::conditioning_cache_key_for_authority(r, cuda_shipped) ==
        slopfab::conditioning_cache_key_for_authority(r, cuda_shipped));
  CHECK(slopfab::conditioning_cache_key_for_authority(r, cuda_shipped) !=
        slopfab::conditioning_cache_key_for_authority(r, cuda_exact));
  CHECK(slopfab::conditioning_cache_key_for_authority(r, cuda_exact) !=
        slopfab::conditioning_cache_key_for_authority(r, vulkan_exact));
  CHECK(slopfab::conditioning_cache_key_for_authority(r, cuda_shipped) !=
        slopfab::conditioning_cache_key_for_authority(r, vulkan_exact));

  // Reference count is part of both: dropping one must not leave a prefix that
  // compares equal to the longer list.
  slopfab::GenerateRequest two = r;
  two.reference_image_paths = {a.string(), b.string()};
  CHECK(slopfab::reference_cache_key(two) != slopfab::reference_cache_key(r));
  CHECK(slopfab::conditioning_cache_key(two) != slopfab::conditioning_cache_key(r));

  // An empty tokenizer path is the embedded tokenizer, which cannot go stale
  // and must key stably rather than looking like a missing file each time.
  slopfab::GenerateRequest embedded = r;
  embedded.tokenizer_path.clear();
  CHECK(slopfab::tokenizer_cache_key(embedded) == slopfab::tokenizer_cache_key(embedded));
  CHECK(slopfab::tokenizer_cache_key(embedded) != slopfab::tokenizer_cache_key(r));

  std::filesystem::remove(a);
  std::filesystem::remove(b);
}

SLOPFAB_TEST(sol_schedule_ranges_and_cadence) {
  slopfab::SolSchedule s;
  CHECK(!s.active(9,2));
  CHECK(!s.active(10,1));
  CHECK(s.active(10,2));
  CHECK(s.active(11,49));
  s.step_end=16;s.step_every=2;s.layer_end=10;s.layer_every=3;
  CHECK(s.active(10,2));
  CHECK(s.active(12,5));
  CHECK(s.active(16,8));
  CHECK(!s.active(11,2));
  CHECK(!s.active(12,3));
  CHECK(!s.active(18,8));
  CHECK(!s.active(12,11));
}

SLOPFAB_TEST(cuda_toolkit_selection) {
  const std::vector<slopfab::cuda::CudaToolkitCandidate> both = {
      {13, L"C:\\CUDA\\v13\\bin\\x64"},
      {12, L"C:\\CUDA\\v12\\bin"},
  };
  const auto* preferred = slopfab::cuda::select_cuda_toolkit(L"auto", both);
  CHECK(preferred != nullptr && preferred->major == 13);

  std::vector<slopfab::cuda::CudaToolkitCandidate> fallback = both;
  fallback[0].bin.clear();
  const auto* selected_fallback = slopfab::cuda::select_cuda_toolkit(L"auto", fallback);
  CHECK(selected_fallback != nullptr && selected_fallback->major == 12);

  CHECK(slopfab::cuda::cuda_version_request(L"", L"12") == L"12");
  CHECK(slopfab::cuda::cuda_version_request(L"13", L"12") == L"13");
  const auto* environment_override = slopfab::cuda::select_cuda_toolkit(
      slopfab::cuda::cuda_version_request(L"", L"12"), both);
  CHECK(environment_override != nullptr && environment_override->major == 12);

  CHECK(slopfab::cuda::cuda_version_matches_linked_toolkit(L"auto", 12));
  CHECK(slopfab::cuda::cuda_version_matches_linked_toolkit(L"12", 12));
  CHECK(!slopfab::cuda::cuda_version_matches_linked_toolkit(L"13", 12));

  CHECK(!slopfab::cuda::cuda_driver_supports_toolkit(12999, 13));
  CHECK(slopfab::cuda::cuda_driver_supports_toolkit(12999, 12));
  CHECK(slopfab::cuda::cuda_driver_supports_toolkit(13000, 13));
  const auto* driver_fallback = slopfab::cuda::select_cuda_toolkit_for_driver(
      L"auto", both, 12999);
  CHECK(driver_fallback != nullptr && driver_fallback->major == 12);
  CHECK(slopfab::cuda::select_cuda_toolkit_for_driver(L"13", both, 12999) ==
        nullptr);
}

}  // namespace
