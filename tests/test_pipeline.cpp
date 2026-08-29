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
#include "vidfab/pipeline.h"
#include "vidfab/attention_mode.h"
#include "vidfab/sampler/scheduler.h"

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
    vidfab::GenerateRequest r;
    r.num_frames = frames;
    vidfab::resolve_plan(r);
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

vidfab::GenerateRequest base_request() {
  vidfab::GenerateRequest r;
  r.prompt = "a test";
  return r;
}

VIDFAB_TEST(pipeline_plan_default) {
  const vidfab::GenerateRequest r = base_request();
  const vidfab::GeneratePlan p = vidfab::resolve_plan(r);

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

VIDFAB_TEST(attention_mode_parse_name_and_backend_contract) {
  using vidfab::AttentionMode;
  using vidfab::DeviceBackend;
  const AttentionMode modes[] = {
      AttentionMode::kNone, AttentionMode::kFlash2, AttentionMode::kSage2,
      AttentionMode::kSol, AttentionMode::kSolExperimental, AttentionMode::kExact};
  for (AttentionMode mode : modes) {
    AttentionMode parsed = AttentionMode::kNone;
    CHECK(vidfab::parse_attention_mode(vidfab::attention_mode_name(mode), &parsed));
    CHECK(parsed == mode);
    CHECK(vidfab::attention_mode_supported(DeviceBackend::kCuda, mode));
    CHECK(vidfab::attention_mode_supported(DeviceBackend::kVulkan, mode) ==
          (mode == AttentionMode::kExact));
  }

  AttentionMode unchanged = AttentionMode::kSol;
  CHECK(!vidfab::parse_attention_mode("flash3", &unchanged));
  CHECK(unchanged == AttentionMode::kSol);
  CHECK(!vidfab::parse_attention_mode("exact ", &unchanged));
  CHECK(!vidfab::parse_attention_mode("exact", nullptr));
  CHECK(std::string(vidfab::attention_mode_name(static_cast<AttentionMode>(999))) == "unknown");
  CHECK(!vidfab::attention_mode_supported(DeviceBackend::kVulkan,
                                           static_cast<AttentionMode>(999)));
}

VIDFAB_TEST(pipeline_plan_aspect_and_frames) {
  vidfab::GenerateRequest r = base_request();
  r.aspect_w = 9;
  r.aspect_h = 16;
  r.num_frames = 240;
  const vidfab::GeneratePlan p = vidfab::resolve_plan(r);

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
  const vidfab::GeneratePlan sq = vidfab::resolve_plan(r);
  CHECK(sq.canvas_height == 768 && sq.canvas_width == 768);
  CHECK(sq.layout.latent_height == 48 && sq.layout.latent_width == 48);
  CHECK(sq.layout.rows_per_frame() == 24 * 24);
}

VIDFAB_TEST(pipeline_plan_explicit_resolution) {
  // The property the whole flag rests on: naming the canvas the default
  // already produces must reproduce the default plan exactly. If this drifts,
  // adding the option moved the default for everyone who never passed it.
  const vidfab::GeneratePlan derived = vidfab::resolve_plan(base_request());

  vidfab::GenerateRequest named = base_request();
  named.canvas_width = 1344;
  named.canvas_height = 768;
  const vidfab::GeneratePlan explicit_plan = vidfab::resolve_plan(named);

  CHECK(explicit_plan.canvas_width == derived.canvas_width);
  CHECK(explicit_plan.canvas_height == derived.canvas_height);
  CHECK(explicit_plan.layout.latent_width == derived.layout.latent_width);
  CHECK(explicit_plan.layout.latent_height == derived.layout.latent_height);
  CHECK(explicit_plan.layout.total_rows() == derived.layout.total_rows());

  // An explicit canvas wins over the aspect rather than being reconciled with
  // it — the two disagree here on purpose, and the canvas is what survives.
  vidfab::GenerateRequest both = base_request();
  both.aspect_w = 1;
  both.aspect_h = 1;
  both.canvas_width = 1024;
  both.canvas_height = 512;
  const vidfab::GeneratePlan p = vidfab::resolve_plan(both);
  CHECK(p.canvas_width == 1024 && p.canvas_height == 512);
  CHECK(p.layout.latent_width == 64 && p.layout.latent_height == 32);

  // One axis alone is not an explicit canvas, so it falls back to the aspect
  // rather than silently generating a 1344x0 plan.
  vidfab::GenerateRequest half = base_request();
  half.canvas_width = 1024;
  const vidfab::GeneratePlan fell_back = vidfab::resolve_plan(half);
  CHECK(fell_back.canvas_width == 1344 && fell_back.canvas_height == 768);

  // Over the trained area is allowed here and refused nowhere — the warning
  // lives in the CLI, and the plan resolves so a caller can see the cost.
  vidfab::GenerateRequest big = base_request();
  big.canvas_width = 1920;
  big.canvas_height = 1088;
  const vidfab::GeneratePlan large = vidfab::resolve_plan(big);
  CHECK(large.layout.total_rows() > derived.layout.total_rows());

  // A canvas off the 32-grid is rejected, not rounded.
  CHECK(::vidfab::test::throws([] {
    vidfab::GenerateRequest r;
    r.canvas_width = 1350;
    r.canvas_height = 768;
    vidfab::resolve_plan(r);
  }));
}

VIDFAB_TEST(pipeline_plan_rejects_bad_requests) {
  // A one-point grid has no model evaluation at all.
  CHECK(::vidfab::test::throws([] {
    vidfab::GenerateRequest r;
    r.num_inference_steps = 1;
    vidfab::resolve_plan(r);
  }));

  // Outside 1:4 .. 4:1.
  CHECK(::vidfab::test::throws([] {
    vidfab::GenerateRequest r;
    r.aspect_w = 5;
    r.aspect_h = 1;
    vidfab::resolve_plan(r);
  }));

  CHECK(::vidfab::test::throws([] {
    vidfab::GenerateRequest r;
    r.num_frames = 0;
    vidfab::resolve_plan(r);
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
  vidfab::GenerateRequest ok;
  ok.num_frames = 6;
  const vidfab::GeneratePlan p = vidfab::resolve_plan(ok);
  CHECK(p.aligned_frames == 22);
  CHECK(p.layout.num_latent_frames == 7);
}

VIDFAB_TEST(pipeline_schedules_stay_paired) {
  // The denoise loop zips the video and audio timestep lists while iterating
  // the video one, so a length divergence caused by unique_consecutive
  // collapsing the two shifted grids differently would silently truncate the
  // run. resolve_plan asserts they match; check it holds across the practical
  // range rather than only at the default.
  for (int steps = 2; steps <= 120; ++steps) {
    vidfab::GenerateRequest r = base_request();
    r.num_inference_steps = steps;
    const vidfab::GeneratePlan p = vidfab::resolve_plan(r);
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
VIDFAB_TEST(pipeline_plan_carries_its_schedule_inputs) {
  for (int steps : {2, 7, 29, 50, 120}) {
    vidfab::GenerateRequest r = base_request();
    r.num_inference_steps = steps;
    const vidfab::GeneratePlan p = vidfab::resolve_plan(r);

    CHECK(p.num_inference_steps == steps);
    CHECK_NEAR(p.video_sigma_shift, 12.0, 0.0);
    CHECK_NEAR(p.audio_sigma_shift, 3.0, 0.0);

    vidfab::sampler::FlowScheduler video(p.video_sigma_shift);
    vidfab::sampler::FlowScheduler audio(p.audio_sigma_shift);
    video.set_timesteps(p.num_inference_steps);
    audio.set_timesteps(p.num_inference_steps);

    CHECK(video.sigmas() == p.video_sigmas);
    CHECK(audio.sigmas() == p.audio_sigmas);
    CHECK(video.timesteps() == p.video_timesteps);
    CHECK(audio.timesteps() == p.audio_timesteps);
  }

  // A plan that never went through resolve_plan still names the shipped shifts
  // rather than zero, so the defaults cannot quietly become shift-free.
  const vidfab::GeneratePlan fresh;
  CHECK_NEAR(fresh.video_sigma_shift, vidfab::kVideoSigmaShift, 0.0);
  CHECK_NEAR(fresh.audio_sigma_shift, vidfab::kAudioSigmaShift, 0.0);
}

VIDFAB_TEST(pipeline_describe_plan) {
  const vidfab::GenerateRequest r = base_request();
  const vidfab::GeneratePlan p = vidfab::resolve_plan(r);
  const std::string text = vidfab::describe_plan(r, p);
  CHECK(text.find("768 x 1344") != std::string::npos);
  CHECK(text.find("37710") != std::string::npos);
  CHECK(text.find("49 model evaluations") != std::string::npos);

  // Every number in the summary comes from the plan, not from the request that
  // produced it. The grid the header prints and the grid the loop integrates
  // have to be the same object, and a request field read here is the one way
  // back to two of them.
  vidfab::GeneratePlan renumbered = p;
  renumbered.num_inference_steps = 7;
  const std::string retold = vidfab::describe_plan(r, renumbered);
  CHECK(retold.find("7 grid points") != std::string::npos);
  CHECK(retold.find("50 grid points") == std::string::npos);
}

VIDFAB_TEST(pipeline_reference_image_limit) {
  vidfab::GenerateRequest r = base_request();
  r.reference_image_paths = {"subject.png", "style.png", "scene.png"};
  const vidfab::GeneratePlan p = vidfab::resolve_plan(r);
  CHECK(vidfab::describe_plan(r, p).find("3 (Ref2VA, ordered)") != std::string::npos);
  CHECK(r.reference_image_paths[0] == "subject.png");
  CHECK(r.reference_image_paths[1] == "style.png");
  CHECK(r.reference_image_paths[2] == "scene.png");

  // Nine is accepted; the tenth is rejected before any image I/O or GPU work.
  r.reference_image_paths.assign(9, "image.png");
  CHECK(vidfab::describe_plan(r, vidfab::resolve_plan(r)).find("9 (Ref2VA, ordered)") !=
        std::string::npos);
  r.reference_image_paths.push_back("too-many.png");
  bool rejected = false;
  try {
    vidfab::resolve_plan(r);
  } catch (const std::exception&) {
    rejected = true;
  }
  CHECK(rejected);
}

// The whole point of keying on identity rather than on name: a reference image
// overwritten in place between two `--reuse-models` generations must not be
// served from the first image's cache entry. Before this, both runs produced
// the same key and the second silently rendered the first image's conditioning.
VIDFAB_TEST(conditioning_cache_key_detects_overwritten_reference_image) {
  const std::filesystem::path ref = scratch_path("vidfab_cachekey_ref.ppm");

  vidfab::GenerateRequest r = base_request();
  r.text_encoder_path = "encoder.safetensors";
  r.tokenizer_path = "tokenizer.json";
  r.reference_image_paths = {ref.string()};

  write_file(ref, "first image bytes", 120);
  const std::string before = vidfab::conditioning_cache_key(r);

  // Same path, same request, different content. Different size *and* a
  // different mtime, which is what any real overwrite produces.
  write_file(ref, "second image bytes, a different length entirely", 0);
  const std::string after = vidfab::conditioning_cache_key(r);
  CHECK(before != after);

  // Rewriting the identical bytes at a later mtime is the *same* image, and a
  // content-keyed entry says so rather than throwing away a valid encode. This
  // is the direction stat keying gets wrong in the harmless way.
  write_file(ref, "second image bytes, a different length entirely", 60);
  const std::string rewritten = vidfab::conditioning_cache_key(r);
  CHECK(rewritten == after);

  // And nothing else moved: asking twice with the file untouched is a hit.
  CHECK(vidfab::conditioning_cache_key(r) == rewritten);

  // The reference key sees the same overwrite, because the VAE keyframe encode
  // is cached against it.
  r.video_vae_path = "video_vae.safetensors";
  const std::string ref_key = vidfab::reference_cache_key(r);
  write_file(ref, "third", 0);
  CHECK(vidfab::reference_cache_key(r) != ref_key);

  // Deleting the file is a change too, rather than "unchanged since last time".
  const std::string present = vidfab::conditioning_cache_key(r);
  std::filesystem::remove(ref);
  CHECK(vidfab::conditioning_cache_key(r) != present);

  // Checkpoints keep the cheaper stat identity, and it is live: a rebuilt
  // encoder at the same path invalidates on mtime alone. Hashing 27 GB here
  // would cost more than everything the key protects.
  const std::filesystem::path ckpt = scratch_path("vidfab_cachekey_ckpt.bin");
  r.reference_image_paths.clear();
  r.text_encoder_path = ckpt.string();
  write_file(ckpt, "weights", 300);
  const std::string old_ckpt = vidfab::conditioning_cache_key(r);
  write_file(ckpt, "weights", 0);
  CHECK(vidfab::conditioning_cache_key(r) != old_ckpt);
  std::filesystem::remove(ckpt);
}

// The one the previous test cannot prove. `write_file` above manufactures its
// mtime difference, so it shows the key reads mtime — not that a real in-place
// overwrite is caught. Here the replacement is same-size and the timestamp is
// put back exactly, which is what `copy`, `robocopy /COPY:T`, `xcopy /K`,
// rsync --times and a good deal of image tooling actually do. Only hashing the
// contents distinguishes these two files.
VIDFAB_TEST(reference_key_detects_overwrite_that_preserves_size_and_mtime) {
  const std::filesystem::path ref = scratch_path("vidfab_cachekey_samestamp.ppm");

  vidfab::GenerateRequest r = base_request();
  r.video_vae_path = "video_vae.safetensors";
  r.text_encoder_path = "encoder.safetensors";
  r.reference_image_paths = {ref.string()};

  write_file(ref, "PPM-payload-version-one", 90);
  const std::filesystem::file_time_type stamp = std::filesystem::last_write_time(ref);
  const std::string before = vidfab::conditioning_cache_key(r);
  const std::string before_ref = vidfab::reference_cache_key(r);

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

  CHECK(vidfab::conditioning_cache_key(r) != before);
  CHECK(vidfab::reference_cache_key(r) != before_ref);

  // And it is still stable when nothing moves at all, so the hash has not just
  // made every lookup a miss.
  const std::string settled = vidfab::conditioning_cache_key(r);
  CHECK(vidfab::conditioning_cache_key(r) == settled);
  CHECK(vidfab::reference_cache_key(r) == vidfab::reference_cache_key(r));

  // A byte-for-byte identical rewrite at a different mtime is the same image,
  // and content keying says so — which stat keying could not.
  write_file(ref, "PPM-payload-version-two", 5);
  CHECK(vidfab::conditioning_cache_key(r) == settled);

  std::filesystem::remove(ref);
}

// Sharing one hash between the two keys has to be a pure saving: the keys it
// produces must be the ones the self-hashing form produces, or a run that took
// the fast path would miss a cache the slow path would have hit.
VIDFAB_TEST(shared_reference_identities_agree_with_hashing_twice) {
  const std::filesystem::path one = scratch_path("vidfab_cachekey_share1.ppm");
  const std::filesystem::path two = scratch_path("vidfab_cachekey_share2.ppm");
  write_file(one, "first reference payload", 40);
  write_file(two, "second reference payload, longer", 40);

  vidfab::GenerateRequest r = base_request();
  r.text_encoder_path = "encoder.safetensors";
  r.video_vae_path = "video_vae.safetensors";
  r.reference_image_paths = {one.string(), two.string()};

  const std::vector<std::string> identities = vidfab::reference_image_identities(r);
  CHECK(identities.size() == 2);
  CHECK(identities[0] != identities[1]);

  CHECK(vidfab::conditioning_cache_key(r, identities) == vidfab::conditioning_cache_key(r));
  CHECK(vidfab::reference_cache_key(r, identities) == vidfab::reference_cache_key(r));

  // A list that does not match the request falls back to hashing rather than
  // keying off a stale snapshot — the failure mode that would otherwise reuse
  // the wrong image silently, which is the whole bug class this file guards.
  const std::vector<std::string> truncated{identities[0]};
  CHECK(vidfab::conditioning_cache_key(r, truncated) == vidfab::conditioning_cache_key(r));
  CHECK(vidfab::reference_cache_key(r, truncated) == vidfab::reference_cache_key(r));
  CHECK(vidfab::conditioning_cache_key(r, {}) == vidfab::conditioning_cache_key(r));

  // The empty case is not a special case: no references, and both forms agree.
  vidfab::GenerateRequest text_only = base_request();
  text_only.text_encoder_path = "encoder.safetensors";
  CHECK(vidfab::reference_image_identities(text_only).empty());
  CHECK(vidfab::conditioning_cache_key(text_only, {}) ==
        vidfab::conditioning_cache_key(text_only));

  // And the shared snapshot still tracks content: rehashing after an overwrite
  // gives different identities and therefore different keys.
  const std::string before = vidfab::conditioning_cache_key(r, identities);
  write_file(one, "first reference payload, edited", 40);
  const std::vector<std::string> rehashed = vidfab::reference_image_identities(r);
  CHECK(vidfab::conditioning_cache_key(r, rehashed) != before);

  std::filesystem::remove(one);
  std::filesystem::remove(two);
}

VIDFAB_TEST(cache_keys_separate_their_inputs) {
  const std::filesystem::path a = scratch_path("vidfab_cachekey_a.bin");
  const std::filesystem::path b = scratch_path("vidfab_cachekey_b.bin");
  write_file(a, "aaaa", 60);
  write_file(b, "aaaa", 60);

  vidfab::GenerateRequest r = base_request();
  r.text_encoder_path = a.string();
  r.tokenizer_path = a.string();
  r.video_vae_path = a.string();
  r.reference_image_paths = {a.string()};

  // Two files with identical size and mtime still differ, because the path is
  // part of the identity.
  vidfab::GenerateRequest other = r;
  other.reference_image_paths = {b.string()};
  CHECK(vidfab::conditioning_cache_key(r) != vidfab::conditioning_cache_key(other));

  // The prompt drives conditioning and nothing else. A prompt sweep must not
  // invalidate the reference encode or the tokenizer.
  vidfab::GenerateRequest reworded = r;
  reworded.prompt = r.prompt + " at night";
  CHECK(vidfab::conditioning_cache_key(reworded) != vidfab::conditioning_cache_key(r));
  CHECK(vidfab::reference_cache_key(reworded) == vidfab::reference_cache_key(r));
  CHECK(vidfab::tokenizer_cache_key(reworded) == vidfab::tokenizer_cache_key(r));

  // The video VAE is an input to the reference encode but not to conditioning.
  vidfab::GenerateRequest other_vae = r;
  other_vae.video_vae_path = b.string();
  CHECK(vidfab::reference_cache_key(other_vae) != vidfab::reference_cache_key(r));
  CHECK(vidfab::conditioning_cache_key(other_vae) == vidfab::conditioning_cache_key(r));

  // Reference count is part of both: dropping one must not leave a prefix that
  // compares equal to the longer list.
  vidfab::GenerateRequest two = r;
  two.reference_image_paths = {a.string(), b.string()};
  CHECK(vidfab::reference_cache_key(two) != vidfab::reference_cache_key(r));
  CHECK(vidfab::conditioning_cache_key(two) != vidfab::conditioning_cache_key(r));

  // An empty tokenizer path is the embedded tokenizer, which cannot go stale
  // and must key stably rather than looking like a missing file each time.
  vidfab::GenerateRequest embedded = r;
  embedded.tokenizer_path.clear();
  CHECK(vidfab::tokenizer_cache_key(embedded) == vidfab::tokenizer_cache_key(embedded));
  CHECK(vidfab::tokenizer_cache_key(embedded) != vidfab::tokenizer_cache_key(r));

  std::filesystem::remove(a);
  std::filesystem::remove(b);
}

VIDFAB_TEST(sol_schedule_ranges_and_cadence) {
  vidfab::SolSchedule s;
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

}  // namespace
