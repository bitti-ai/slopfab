// Request-to-plan resolution.
//
// Everything here is arithmetic a user can trigger from the command line
// before any weight is read, so it is the first thing that should refuse a bad
// request — and the last thing that should silently produce a subtly wrong
// geometry.

#include <cmath>
#include <stdexcept>
#include <string>

#include "harness.h"
#include "vidfab/pipeline.h"
#include "vidfab/attention_mode.h"

namespace {

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

VIDFAB_TEST(pipeline_describe_plan) {
  const vidfab::GenerateRequest r = base_request();
  const vidfab::GeneratePlan p = vidfab::resolve_plan(r);
  const std::string text = vidfab::describe_plan(r, p);
  CHECK(text.find("768 x 1344") != std::string::npos);
  CHECK(text.find("37710") != std::string::npos);
  CHECK(text.find("49 model evaluations") != std::string::npos);
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
