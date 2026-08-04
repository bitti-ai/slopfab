// Request-to-plan resolution.
//
// Everything here is arithmetic a user can trigger from the command line
// before any weight is read, so it is the first thing that should refuse a bad
// request — and the last thing that should silently produce a subtly wrong
// geometry.

#include <cmath>
#include <string>

#include "harness.h"
#include "vidfab/pipeline.h"

namespace {

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

}  // namespace
