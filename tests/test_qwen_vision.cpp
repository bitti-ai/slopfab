#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "harness.h"
#include "vidfab/text/qwen_vision.h"

using namespace vidfab::text;

VIDFAB_TEST(qwen_vision_deepstack_targets_first_text_layers) {
  CHECK(qwen3vl_deepstack_slot(-1) == -1);
  CHECK(qwen3vl_deepstack_slot(0) == 0);
  CHECK(qwen3vl_deepstack_slot(1) == 1);
  CHECK(qwen3vl_deepstack_slot(2) == 2);
  CHECK(qwen3vl_deepstack_slot(8) == -1);
  CHECK(qwen3vl_deepstack_slot(16) == -1);
  CHECK(qwen3vl_deepstack_slot(24) == -1);
}

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

// The two RoPE tables used to evaluate `std::pow` once per (row, axis, j),
// for an answer that depends only on `j`. Hoisting it out of the loops was
// meant to change nothing at all, and "nothing at all" here means every one of
// the ~590000 floats in the vision table is the same bit pattern, not close to
// it: these tables feed attention on every image, and a table that drifted in
// the last mantissa bit would produce a plausible video conditioned on
// slightly the wrong geometry.
//
// So the pre-hoist expression is written out again below and compared exactly.
// `docs/text_encoder_spec.md` 6.4 pins the fp64-then-round evaluation, which
// is why this compares against `std::pow(double, double)` rather than against
// an `exp2` or reciprocal formulation that would be faster and different.
namespace {

bool same_bits(float a, float b) {
  uint32_t x = 0;
  uint32_t y = 0;
  std::memcpy(&x, &a, sizeof(x));
  std::memcpy(&y, &b, sizeof(y));
  return x == y;
}

}  // namespace

VIDFAB_TEST(qwen_vision_rope_hoisting_is_bit_identical) {
  // A grid big enough that the hoist actually matters: 32x32 patches is 1024
  // rows of 72 channels.
  const auto p = qwen3vl_vision_positions({1, 32, 32});
  std::vector<float> cos_got, sin_got;
  qwen3vl_vision_rope_tables(p, cos_got, sin_got);

  const int head_dim = 72;
  const int axis_half = head_dim / 4;
  const float theta = 10000.0f;
  const size_t rows = p.rotary_thw.size() / 3;
  CHECK(rows == 1024);
  CHECK(cos_got.size() == rows * head_dim);

  std::vector<float> cos_want(rows * head_dim), sin_want(rows * head_dim);
  for (size_t r = 0; r < rows; ++r) {
    const int coords[2] = {p.rotary_thw[r * 3 + 1], p.rotary_thw[r * 3 + 2]};
    for (int a = 0; a < 2; ++a) {
      for (int j = 0; j < axis_half; ++j) {
        // Verbatim the pre-hoist expression, `pow` inside the inner loop.
        const double inv = std::pow(static_cast<double>(theta), -2.0 * j / (head_dim / 2));
        const float angle = static_cast<float>(coords[a] * inv);
        const int k = a * axis_half + j;
        cos_want[r * head_dim + k] = cos_want[r * head_dim + k + head_dim / 2] = std::cos(angle);
        sin_want[r * head_dim + k] = sin_want[r * head_dim + k + head_dim / 2] = std::sin(angle);
      }
    }
  }

  size_t diffs = 0;
  size_t first = 0;
  for (size_t i = 0; i < cos_want.size(); ++i) {
    if (!same_bits(cos_want[i], cos_got[i]) || !same_bits(sin_want[i], sin_got[i])) {
      if (diffs == 0) first = i;
      ++diffs;
    }
  }
  CHECK_MSG(diffs == 0, "vision rope: %zu of %zu entries differ, first at %zu (%.9g vs %.9g)",
            diffs, cos_want.size(), first, static_cast<double>(cos_want[first]),
            static_cast<double>(cos_got[first]));
}

VIDFAB_TEST(qwen_decoder_mrope_hoisting_is_bit_identical) {
  // 512 text tokens with a 4x4 image in the middle, so all three axes carry
  // non-trivial positions rather than the identity a pure-text plan gives.
  std::vector<int32_t> ids;
  for (int i = 0; i < 200; ++i) ids.push_back(1000 + i);
  ids.push_back(151652);
  for (int i = 0; i < 16; ++i) ids.push_back(151655);
  ids.push_back(151653);
  for (int i = 0; i < 200; ++i) ids.push_back(2000 + i);
  const auto plan = qwen3vl_multimodal_plan(ids, {{1, 8, 8}});
  const int tokens = static_cast<int>(ids.size());

  std::vector<float> cos_got, sin_got;
  qwen3vl_decoder_rope_tables(plan, tokens, cos_got, sin_got);

  const int head_dim = 128;
  const float theta = 5.0e6f;
  std::vector<float> cos_want(static_cast<size_t>(tokens) * head_dim), sin_want(cos_want.size());
  for (int r = 0; r < tokens; ++r) {
    for (int j = 0; j < head_dim / 2; ++j) {
      const int axis = j < 60 ? j % 3 : 0;
      // Verbatim the pre-hoist expression.
      const double inv = std::pow(static_cast<double>(theta), -2.0 * j / head_dim);
      const float a =
          static_cast<float>(plan.position_ids[static_cast<size_t>(axis) * tokens + r] * inv);
      cos_want[static_cast<size_t>(r) * head_dim + j] =
          cos_want[static_cast<size_t>(r) * head_dim + j + 64] = std::cos(a);
      sin_want[static_cast<size_t>(r) * head_dim + j] =
          sin_want[static_cast<size_t>(r) * head_dim + j + 64] = std::sin(a);
    }
  }

  size_t diffs = 0;
  size_t first = 0;
  for (size_t i = 0; i < cos_want.size(); ++i) {
    if (!same_bits(cos_want[i], cos_got[i]) || !same_bits(sin_want[i], sin_got[i])) {
      if (diffs == 0) first = i;
      ++diffs;
    }
  }
  CHECK_MSG(diffs == 0, "decoder mrope: %zu of %zu entries differ, first at %zu (%.9g vs %.9g)",
            diffs, cos_want.size(), first, static_cast<double>(cos_want[first]),
            static_cast<double>(cos_got[first]));
}
