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
