#include "harness.h"
#include "slopfab/inpaint.h"
#include "slopfab/generate.h"
#include "slopfab/dit/packing.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

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

SLOPFAB_TEST(inpaint_constraint_uses_fresh_noise_without_mutating_initial_noise) {
  slopfab::InpaintConstraint c{{2, 4, 6}, {10, 20, 30}, {0, 1, 0}};
  std::vector<float> rows{-1, -2, -3}, fresh{-10, -20, -30};
  c.apply(rows.data(), rows.size(), .25f, fresh.data());
  CHECK(rows == std::vector<float>({-1, -2, -3}));
  CHECK(c.noise == std::vector<float>({10, 20, 30}));
  std::fill(fresh.begin(), fresh.end(), std::numeric_limits<float>::quiet_NaN());
  c.apply(rows.data(), rows.size(), 0, fresh.data());
  CHECK(rows == std::vector<float>({2, -2, 6}));
}

SLOPFAB_TEST(inpaint_renoise_validation_preserves_other_restrictions) {
  slopfab::GenerateRequest request;
  request.still_image = true;
  request.out_path = "edited.ppm";
  request.image_edit = edit_fixture();
  slopfab::RunOptions options;
  options.sampler = slopfab::sampler::SamplerKind::kRenoise;
  const auto plan = slopfab::resolve_plan(request);
  slopfab::validate_generation_options(request, plan, options);
  const auto rejected = [&] {
    try {
      slopfab::validate_generation_options(request, plan, options);
      return false;
    } catch (const std::invalid_argument&) {
      return true;
    }
  };
  options.sampler = slopfab::sampler::SamplerKind::kAb2;
  CHECK(rejected());
  options.sampler = slopfab::sampler::SamplerKind::kRenoise;
  options.init_latents_path = "initial.safetensors";
  CHECK(rejected());
  options.init_latents_path.clear();
  request.skip_every = 2;
  CHECK(rejected());
}

SLOPFAB_TEST(outpaint_preserves_original_context_and_generates_the_whole_surround) {
  auto image = std::make_shared<slopfab::RGBImage>();
  image->width = image->height = 96;
  image->pixels.resize(96 * 96 * 3, 123);
  // Exercise aligned, unaligned and corner-anchored originals.
  for (const int offset : {0, 17, 32}) {
    slopfab::ImageEdit edit{image, offset, offset, 48, 48, 1, 0, true};
    edit.blend_overlap = 1; // explicit legacy hard-edge preservation
    const auto source = slopfab::outpaint_source_image(edit);
    CHECK(source.width == 48 && source.height == 48);
    CHECK(source.pixels == std::vector<uint8_t>(48 * 48 * 3, 123));
    const auto packed = slopfab::edit_mask_rows(edit, 96, 96);
    slopfab::dit::SequenceLayout layout;
    layout.num_latent_frames = 1;
    layout.latent_width = layout.latent_height = 6;
    std::vector<float> mask(packed.size());
    slopfab::dit::unpatchify_video(packed.data(), layout, mask.data());
    for (int c = 0; c < 24; ++c)
      for (int y = 0; y < 6; ++y)
        for (int x = 0; x < 6; ++x) {
          const bool preserved = x * 16 >= offset && (x + 1) * 16 <= offset + 48 &&
                                 y * 16 >= offset && (y + 1) * 16 <= offset + 48;
          CHECK(mask[c * 36 + y * 6 + x] == (preserved ? 0.0f : 1.0f));
        }
    slopfab::InpaintConstraint constraint{std::vector<float>(packed.size(), .4f),
                                         std::vector<float>(packed.size(), .8f), packed};
    for (float sigma : {.9f, .5f, 0.0f}) {
      std::vector<float> rows(packed.size(), -.25f);
      constraint.apply(rows.data(), rows.size(), sigma);
      for (size_t i = 0; i < rows.size(); ++i)
        CHECK_NEAR(rows[i], packed[i] == 0 ? (1 - sigma) * .4f + sigma * .8f : -.25f, 1e-6);
    }
    slopfab::PixelBuffer generated(3 * 96 * 96, .9f);
    // Preserved pixels do not depend on the model's reconstruction, even NaNs.
    for (int c = 0; c < 3; ++c)
      for (int y = offset; y < offset + 48; ++y)
        for (int x = offset; x < offset + 48; ++x)
          generated[(c * 96 + y) * 96 + x] = std::numeric_limits<float>::quiet_NaN();
    const auto output = slopfab::composite_image_edit(edit, generated, 96, 96);
    for (int c = 0; c < 3; ++c)
      for (int y = 0; y < 96; ++y)
        for (int x = 0; x < 96; ++x) {
          const bool preserved = x >= offset && x < offset + 48 && y >= offset && y < offset + 48;
          CHECK(output[(c * 96 + y) * 96 + x] == (preserved ? 123 / 255.0f : .9f));
        }
  }
}

SLOPFAB_TEST(outpaint_semantic_source_excludes_padding_and_keeps_crop_coordinates) {
  auto edit = edit_fixture();
  edit.width = 20;
  edit.height = 17;
  edit.invert_mask = true;
  const auto source = slopfab::outpaint_source_image(edit);
  CHECK(source.width == 20 && source.height == 17);
  for (int y = 0; y < source.height; ++y)
    for (int x = 0; x < source.width; ++x)
      for (int c = 0; c < 3; ++c)
        CHECK(source.pixels[(y * source.width + x) * 3 + c] ==
              edit.image->pixels[((y + edit.y) * edit.image->width + x + edit.x) * 3 + c]);
}

SLOPFAB_TEST(outpaint_blend_matches_dense_max_pool_and_gaussian) {
  auto image = std::make_shared<slopfab::RGBImage>();
  image->width = 71;
  image->height = 67;
  image->pixels.resize(71 * 67 * 3, 51);
  // A dense reference exercises corners, small preserved interiors, touching
  // canvas edges and decoded alignment padding independently of the fast path.
  for (const int offset : {0, 17}) {
    slopfab::ImageEdit edit{image, offset, offset, 48, 48, 1, 0, true};
    CHECK(edit.blend_overlap == 9);
    for (int k : {1, 9, 51}) {
      edit.blend_overlap = k;
      const int r = k / 2, w = image->width, h = image->height;
      std::vector<float> expanded(w * h);
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
          for (int dy = -r; dy <= r; ++dy)
            for (int dx = -r; dx <= r; ++dx) {
              const int qx = std::clamp(x + dx, 0, w - 1);
              const int qy = std::clamp(y + dy, 0, h - 1);
              if (qx < offset || qy < offset || qx >= offset + 48 || qy >= offset + 48)
                expanded[y * w + x] = 1;
            }
      slopfab::PixelBuffer generated(3 * 96 * 96, .8f);
      const auto output = slopfab::composite_image_edit(edit, generated, 96, 96);
      double total = 0;
      std::vector<double> kernel(k * k);
      for (int dy = -r; dy <= r; ++dy)
        for (int dx = -r; dx <= r; ++dx) {
          const double sigma = (k - 1) / 4.0;
          const double weight = r ? std::exp(-(dx * dx + dy * dy) / (2 * sigma * sigma)) : 1;
          kernel[(dy + r) * k + dx + r] = weight;
          total += weight;
        }
      for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
          double alpha = 0;
          for (int dy = -r; dy <= r; ++dy)
            for (int dx = -r; dx <= r; ++dx)
              alpha += kernel[(dy + r) * k + dx + r] *
                  expanded[std::clamp(y + dy, 0, h - 1) * w + std::clamp(x + dx, 0, w - 1)];
          alpha /= total;
          for (int c = 0; c < 3; ++c)
            CHECK_NEAR(output[(c * h + y) * w + x], .2 * (1 - alpha) + .8 * alpha, 2e-7);
          if (x < offset || y < offset || x >= offset + 48 || y >= offset + 48)
            CHECK(output[y * w + x] == .8f);
        }
    }
  }
}

SLOPFAB_TEST(outpaint_blend_validates_kernel_and_preserves_deep_interior) {
  auto edit = edit_fixture();
  edit.x = edit.y = 0;
  edit.width = edit.height = 32;
  edit.invert_mask = true;
  for (int k : {0, 2, 50, 53}) {
    edit.blend_overlap = k;
    bool rejected = false;
    try { edit.validate(); } catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
  }
  edit.blend_overlap = 9;
  slopfab::PixelBuffer generated(3 * 64 * 64, std::numeric_limits<float>::quiet_NaN());
  const auto out = slopfab::composite_image_edit(edit, generated, 64, 64);
  CHECK(out[0] == edit.image->pixels[0] / 255.0f);
  CHECK(out[16 * 35 + 16] == edit.image->pixels[(16 * 35 + 16) * 3] / 255.0f);
}

SLOPFAB_TEST(outpaint_keyframes_reuse_only_pinned_patches_at_target_coordinates) {
  auto image = std::make_shared<slopfab::RGBImage>();
  image->width = 224; image->height = 160;
  image->pixels.resize(224 * 160 * 3, 127);
  slopfab::ImageEdit edit{image, 37, 19, 150, 115, 1, 0, true};
  std::vector<float> original(7 * 5 * 96);
  for (size_t i = 0; i < original.size(); ++i) original[i] = float(i);
  std::vector<slopfab::dit::ReferenceGeometry> geometry;
  std::vector<float> rows;
  slopfab::append_outpaint_keyframe(edit, 224, 160, original, geometry, rows);
  CHECK(geometry.size() == 1);
  const auto& g = geometry.front();
  CHECK(g.target_aligned && g.target_latent_x == 4 && g.target_latent_y == 2);
  CHECK(g.latent_width == 6 && g.latent_height == 6);
  CHECK(rows.size() == 9 * 96);
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 3; ++x)
      for (int c = 0; c < 96; ++c)
        CHECK(rows[(y * 3 + x) * 96 + c] == original[((y + 1) * 7 + x + 2) * 96 + c]);
  auto packed = slopfab::dit::build_ref2va_packed_sequence({0, 0}, geometry, 1, 10, 14, 0);
  CHECK(packed.layout.num_condition_video == 9);
  for (int y = 0; y < 3; ++y)
    for (int x = 0; x < 3; ++x)
      for (int c = 0; c < 3; ++c)
        CHECK(packed.position_ids[(2 + y * 3 + x) * 3 + c] ==
              packed.position_ids[(2 + 9 + (y + 1) * 7 + x + 2) * 3 + c]);
  const auto rejects = [](auto action) {
    try { action(); return false; } catch (const std::invalid_argument&) { return true; }
  };
  geometry[0].target_latent_x = 3;
  CHECK(rejects([&] { slopfab::dit::build_ref2va_packed_sequence({}, geometry, 1, 10, 14, 0); }));
  geometry[0].target_latent_x = 12;
  CHECK(rejects([&] { slopfab::dit::build_ref2va_packed_sequence({}, geometry, 1, 10, 14, 0); }));
  CHECK(rejects([&] { slopfab::append_outpaint_keyframe(edit, 192, 160, original, geometry, rows); }));
}

SLOPFAB_TEST(outpaint_rejects_missing_context_and_empty_extension) {
  CHECK(slopfab::test::throws([] {
    auto edit = edit_fixture();
    edit.invert_mask = true;
    edit.validate(); // sub-cell context
  }));
  CHECK(slopfab::test::throws([] {
    auto edit = edit_fixture();
    edit.invert_mask = true;
    edit.x = edit.y = 0;
    edit.width = edit.height = 32;
    edit.validate();
    edit.feather = 1;
    edit.validate();
  }));
  CHECK(slopfab::test::throws([] {
    auto edit = edit_fixture();
    edit.invert_mask = true;
    edit.x = edit.y = 0;
    edit.width = 35;
    edit.height = 33;
    edit.validate();
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
