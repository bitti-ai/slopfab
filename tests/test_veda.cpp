#include "harness.h"
#include "slopfab/dit/veda.h"
#include "slopfab/safetensors_write.h"
#include <algorithm>
#include <filesystem>

using namespace slopfab;
using namespace slopfab::dit;

SLOPFAB_TEST(veda_tile_order_and_padding) {
  SequenceLayout l;
  l.num_text = 3;
  l.num_condition_video = 65;
  l.num_audio_rows = 67;
  l.num_latent_frames = 5;
  l.latent_height = 10;
  l.latent_width = 18;
  l.num_video_rows = 5 * 5 * 9;
  const auto tiles = build_veda_tiles(l, {4, 4, 8});
  CHECK(tiles.video_tiles == 8);
  CHECK(tiles.sizes.size() == 10);
  CHECK(tiles.sizes[0] == 128);
  CHECK(tiles.sizes[1] == 32);
  CHECK(tiles.sizes[7] == 1);
  CHECK(tiles.rows[0] == 135);
  CHECK(tiles.rows[8] == 144);
  CHECK(tiles.rows[32] == 180);
  CHECK(tiles.rows[128] == 315);
  CHECK(tiles.rows[256] == 143);
  CHECK(tiles.rows[1024] == 0);
  CHECK(tiles.sizes.back() == 7);
  std::vector<int> seen(l.total_rows());
  for (size_t tile = 0; tile < tiles.sizes.size(); ++tile)
    for (int i = 0; i < 128; ++i) {
      const int row = tiles.rows[tile * 128 + i];
      if (i < tiles.sizes[tile]) ++seen.at(row);
      else CHECK(row == -1);
    }
  CHECK(std::all_of(seen.begin(), seen.end(), [](int n) { return n == 1; }));
  CHECK(test::throws([] { build_veda_tiles({}, {4,4,4}); }));
}

SLOPFAB_TEST(veda_equal_cost_budget) {
  const auto budget = parse_veda_budget("90%");
  CHECK_NEAR(budget.value, .1, 1e-12);
  // 10 ideal tiles, padded to 12: keep .1*10*10/12, clamped to one.
  CHECK(budget.keep(1280, 12, 0) == 1);
  const auto fractional = parse_veda_budget("2.3");
  int total = 0;
  for (int row = 0; row < 10; ++row) total += fractional.keep(1280, 10, row);
  CHECK(total == 23);
  CHECK(parse_veda_budget("0%").keep(129, 4, 0) == 4);
  CHECK(parse_veda_budget("24").keep(129, 4, 0) == 4);
  CHECK(test::throws([] { parse_veda_budget("nan%"); }));
  CHECK(test::throws([] { parse_veda_budget("100%"); }));
  CHECK(test::throws([] { parse_veda_budget("0"); }));
  CHECK(test::throws([] { parse_veda_budget("90%junk"); }));
}

SLOPFAB_TEST(veda_plan_selection) {
  VedaPlan p{"landscape", {37,24,42}, {{4,4,8}}, {{0}}};
  auto selected = select_veda_plan({p}, {37,42,24});
  CHECK(selected.name == "landscape_T");
  CHECK((selected.shapes[0] == VedaShape{4,8,4}));
  auto longer = p;
  longer.name = "long";
  longer.grid[0] = 72;
  CHECK(select_veda_plan({p,longer}, {70,24,42}).name == "long");
}

SLOPFAB_TEST(veda_bundle_validation_and_rounding) {
  const auto path = (std::filesystem::temp_directory_path() / "slopfab_veda_host.safetensors").string();
  std::map<std::string,std::string> metadata{
    {"format","miowtion-veda-predictor-v1"}, {"num_layers","1"}, {"num_heads","1"},
    {"head_dim","64"}, {"keep_ratio","0.1"},
    {"plans", R"({"test":{"geometry":"test","grid":[4,4,8],"shapes":["4x4x8"],"head_shape":[[0]]}})"}};
  std::vector<TensorWrite> weights;
  for (const auto* name : {"layers.0.proj_q", "layers.0.proj_k"})
    weights.push_back({name, {1,192,64}, std::vector<float>(192*64, .123456f)});
  write_safetensors(path, weights, metadata);
  {
    VedaBundle bundle;
    bundle.load(path);
    CHECK(bundle.layers == 1 && bundle.heads == 1 && bundle.head_dim == 64);
    CHECK(bundle.projection(0, true)[0] == f32_to_bf16(.123456f));
  }
  metadata["plans"] = R"({"test":{"geometry":"test","grid":[4,4,8],"shapes":["4x4x8"],"head_shape":[[1]]}})";
  write_safetensors(path, weights, metadata);
  bool rejected = false;
  try { VedaBundle bundle; bundle.load(path); } catch (const std::exception&) { rejected = true; }
  CHECK(rejected);
  std::filesystem::remove(path);
}
