#include "harness.h"
#include "slopfab/inpaint.h"
#include "slopfab/generate.h"
#include "slopfab/dit/packing.h"
#include <limits>

namespace {
slopfab::ImageEdit edit_fixture() {
  auto image = std::make_shared<slopfab::RGBImage>();
  image->width = 35;
  image->height = 33;
  image->pixels.resize(35 * 33 * 3);
  for (size_t i = 0; i < image->pixels.size(); ++i)
    image->pixels[i] = static_cast<uint8_t>(i);
  return {image, 15, 16, 2, 2, .5f, 0};
}
}

SLOPFAB_TEST(inpaint_packing_preserves_subpatch_coordinates) {
  const auto edit = edit_fixture();
  const auto rows = slopfab::edit_mask_rows(edit, 64, 64);
  slopfab::dit::SequenceLayout layout;
  layout.num_latent_frames = 1;
  layout.latent_width = layout.latent_height = 4;
  std::vector<float> mask(rows.size());
  slopfab::dit::unpatchify_video(rows.data(), layout, mask.data());
  for (int c = 0; c < 24; ++c)
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x)
        CHECK(mask[c * 16 + y * 4 + x] == (y == 1 && x < 2 ? 1.0f : 0.0f));
  const auto padded = slopfab::pad_edit_image(edit, 64, 64);
  for (int c = 0; c < 3; ++c)
    CHECK(padded.pixels[(63 * 64 + 63) * 3 + c] == edit.image->pixels[(32 * 35 + 34) * 3 + c]);
}

SLOPFAB_TEST(inpaint_composite_preserves_every_outside_pixel) {
  auto edit = edit_fixture();
  slopfab::PixelBuffer generated(3 * 64 * 64, .9f);
  auto output = slopfab::composite_image_edit(edit, generated, 64, 64);
  CHECK(output.size() == 35 * 33 * 3);
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < 33; ++y)
      for (int x = 0; x < 35; ++x) {
        const int p = y * 35 + x;
        const bool inside = x >= 15 && x < 17 && y >= 16 && y < 18;
        CHECK(output[c * 35 * 33 + p] == (inside ? .9f : edit.image->pixels[p * 3 + c] / 255.0f));
      }
  edit.feather = 1;
  output = slopfab::composite_image_edit(edit, generated, 64, 64);
  const int p = 16 * 35 + 15;
  CHECK_NEAR(output[p], .45f + .5f * edit.image->pixels[p * 3] / 255, 1e-7);
  CHECK(output[0] == edit.image->pixels[0] / 255.0f);
}

SLOPFAB_TEST(inpaint_constraint_uses_resulting_sigma_and_fixed_noise) {
  slopfab::InpaintConstraint c{{2, 4, 6}, {10, 20, 30}, {0, 1, 0}};
  auto rows = c.initial(.5f);
  CHECK(rows == std::vector<float>({6, 12, 18}));
  rows = {-1, -2, -3};
  c.apply(rows.data(), rows.size(), .25f);
  CHECK(rows == std::vector<float>({4, -2, 12}));
  c.apply(rows.data(), rows.size(), 0);
  CHECK(rows == std::vector<float>({2, -2, 6}));
  CHECK(slopfab::test::throws([] {
    slopfab::InpaintConstraint{{1}, {}, {0}}.initial(.5f);
  }));
}

SLOPFAB_TEST(inpaint_plan_strength_padding_and_invalid_requests) {
  slopfab::GenerateRequest request;
  request.still_image = true;
  request.out_path = "edited.ppm";
  request.image_edit = edit_fixture();
  request.num_inference_steps = 11;
  const auto plan = slopfab::resolve_plan(request);
  CHECK(plan.canvas_width == 64 && plan.canvas_height == 64);
  CHECK(plan.aligned_frames == 1 && plan.layout.num_audio_rows == 0);
  CHECK(plan.num_model_evaluations() == 5);
  request.image_edit.strength = 1;
  const auto full = slopfab::resolve_plan(request);
  CHECK(plan.video_sigmas.front() == full.video_sigmas[5]);
  CHECK(plan.video_sigmas.back() == 0);
  slopfab::RunOptions options;
  slopfab::validate_generation_options(request, full, options);
  CHECK(slopfab::test::throws([] {
    auto edit = edit_fixture();
    edit.x = std::numeric_limits<int>::max();
    edit.validate();
  }));
  CHECK(slopfab::test::throws([] {
    auto edit = edit_fixture();
    edit.strength = std::numeric_limits<float>::quiet_NaN();
    edit.validate();
  }));
  CHECK(slopfab::test::throws([] {
    slopfab::GenerateRequest r;
    r.image_edit = edit_fixture();
    slopfab::resolve_plan(r);
  }));
  CHECK(slopfab::test::throws([] {
    slopfab::GenerateRequest r;
    r.still_image = true;
    r.image_edit = edit_fixture();
    r.canvas_width = r.canvas_height = 32;
    slopfab::resolve_plan(r);
  }));
  CHECK(slopfab::test::throws([] {
    slopfab::GenerateRequest r;
    r.still_image = true;
    r.image_edit = edit_fixture();
    auto p = slopfab::resolve_plan(r);
    slopfab::RunOptions o;
    o.source = slopfab::LatentSource::kSyntheticNoise;
    slopfab::validate_generation_options(r, p, o);
  }));
}
