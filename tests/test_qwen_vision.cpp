#include "harness.h"
#include "vidfab/text/qwen_vision.h"

using namespace vidfab::text;

VIDFAB_TEST(qwen_vision_smart_resize_and_merge) {
  auto square = qwen3vl_image_grid(256, 256);
  CHECK(square.temporal == 1);
  CHECK(square.height == 16);
  CHECK(square.width == 16);
  CHECK(square.patch_count() == 256);
  CHECK(square.merged_token_count() == 64);

  // A small image is enlarged to at least 65,536 pixels on the factor-32 grid.
  auto small = qwen3vl_image_grid(64, 32);
  CHECK(small.height == 12);
  CHECK(small.width == 24);
  CHECK(small.merged_token_count() == 72);
  CHECK(::vidfab::test::throws([] { (void)qwen3vl_image_grid(201, 1); }));
}

VIDFAB_TEST(qwen_vision_minimax_presentation) {
  const auto ids = qwen3vl_image_block({10, 11}, 3);
  CHECK(ids.size() == 7);
  CHECK(ids[0] == 10);
  CHECK(ids[1] == 11);
  CHECK(ids[2] == 151652);
  CHECK(ids[3] == 151655);
  CHECK(ids[5] == 151655);
  CHECK(ids[6] == 151653);
}

VIDFAB_TEST(qwen_vision_pixel_patch_order) {
  std::vector<uint8_t> rgb(32 * 32 * 3, 128);
  // Distinguish the first pixel/channel and the first pixel of the patch to
  // its right. Merge-group ordering must make those patches consecutive rows.
  rgb[0] = 255;
  rgb[(16 * 3)] = 0;
  const auto pixels = qwen3vl_patchify_resized_rgb(rgb, 32, 32);
  CHECK(pixels.grid.patch_count() == 4);
  CHECK(pixels.rows.size() == 4 * 1536);
  CHECK_NEAR(pixels.rows[0], 1.0, 1e-6);
  CHECK_NEAR(pixels.rows[1536], -1.0, 1e-6);
  // Temporal duplication is inside a row, after channel and before y/x.
  CHECK_NEAR(pixels.rows[256], 1.0, 1e-6);
  CHECK(::vidfab::test::throws([] {
    (void)qwen3vl_patchify_resized_rgb(std::vector<uint8_t>(31 * 32 * 3), 31, 32);
  }));
}

VIDFAB_TEST(qwen_vision_position_order_and_interpolation) {
  const auto p = qwen3vl_vision_positions({1, 2, 4});
  CHECK(p.learned.size() == 8);
  CHECK(p.learned[0] == 0);
  CHECK(p.learned[1] == 15);
  CHECK(p.learned[2] == 47 * 48);
  CHECK(p.learned[3] == 47 * 48 + 15);
  CHECK(p.learned[4] == 31);
  CHECK(p.learned[7] == 48 * 48 - 1);
  CHECK(p.rotary_thw[0] == 0 && p.rotary_thw[1] == 0 && p.rotary_thw[2] == 0);
  CHECK(p.rotary_thw[3] == 0 && p.rotary_thw[4] == 0 && p.rotary_thw[5] == 1);
}

VIDFAB_TEST(qwen_vision_decoder_mrope_and_scatter_rows) {
  const std::vector<int32_t> ids = {7, 151652, 151655, 151655, 151655, 151655, 151653, 8};
  const auto p = qwen3vl_multimodal_plan(ids, {{1, 4, 4}});
  CHECK(p.image_rows == std::vector<int32_t>({2, 3, 4, 5}));
  const size_t L = ids.size();
  CHECK(p.position_ids[0] == 0 && p.position_ids[1] == 1);
  CHECK(p.position_ids[2] == 2 && p.position_ids[L + 2] == 2 && p.position_ids[2 * L + 2] == 2);
  CHECK(p.position_ids[3] == 2 && p.position_ids[L + 3] == 2 && p.position_ids[2 * L + 3] == 3);
  CHECK(p.position_ids[4] == 2 && p.position_ids[L + 4] == 3 && p.position_ids[2 * L + 4] == 2);
  CHECK(p.position_ids[6] == 4 && p.position_ids[7] == 5);
  CHECK(::vidfab::test::throws([] {
    (void)qwen3vl_multimodal_plan(
        {7, 151652, 151655, 151655, 151655, 151655, 151653, 8}, {{1, 2, 2}});
  }));
}

VIDFAB_TEST(qwen_vision_two_axis_rope) {
  const auto p = qwen3vl_vision_positions({1, 2, 2});
  std::vector<float> c, s;
  qwen3vl_vision_rope_tables(p, c, s);
  CHECK(c.size() == 4 * 72 && s.size() == c.size());
  CHECK_NEAR(c[0], 1.0, 1e-6);
  CHECK_NEAR(s[0], 0.0, 1e-6);
  // Row one is (h=0,w=1): height frequencies remain identity, width changes.
  CHECK_NEAR(c[72], 1.0, 1e-6);
  CHECK_NEAR(c[72 + 18], std::cos(1.0), 1e-6);
  CHECK_NEAR(c[72 + 18 + 36], c[72 + 18], 1e-6);
}
