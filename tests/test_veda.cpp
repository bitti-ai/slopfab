#include "harness.h"
#include "slopfab/dit/veda.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/generate.h"
#include "slopfab/json.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>

using namespace slopfab;
using namespace slopfab::dit;

SLOPFAB_TEST(veda_generation_validation) {
  GenerateRequest request;
  auto plan = resolve_plan(request);
  RunOptions options;
  options.attention_mode = AttentionMode::kVeda;
  auto rejected = [&] {
    try {
      validate_generation_options(request, plan, options);
      return false;
    } catch (const std::exception&) {
      return true;
    }
  };
  CHECK(rejected());
  options.veda.predictor_path = "predictor.safetensors";
  for (auto backend : {DeviceBackend::kCuda, DeviceBackend::kVulkan}) {
    options.inference_backend = backend;
    CHECK(!rejected());
  }
  options.attention_band = 1;
  CHECK(rejected());
  options.attention_band = 0;
  plan.model.compressed_attention = true;
  CHECK(rejected());
  plan.model.compressed_attention = false;
  options.veda.generated_sparsity = "100%";
  CHECK(rejected());
  options.veda.generated_sparsity = "trained";
  options.attention_mode = AttentionMode::kFlash2;
  CHECK(rejected());
  options.veda.predictor_path.clear();
  CHECK(!rejected());
}

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
      if (i < tiles.sizes[tile])
        ++seen.at(row);
      else
        CHECK(row == -1);
    }
  CHECK(std::all_of(seen.begin(), seen.end(), [](int n) {
    return n == 1;
  }));
  CHECK(test::throws([] {
    build_veda_tiles({}, {4, 4, 4});
  }));
}

SLOPFAB_TEST(veda_equal_cost_budget) {
  const auto budget = parse_veda_budget("90%");
  CHECK_NEAR(budget.value, .1, 1e-12);
  // 10 ideal tiles, padded to 12: keep .1*10*10/12, clamped to one.
  CHECK(budget.keep(1280, 12, 0) == 1);
  const auto fractional = parse_veda_budget("2.3");
  int total = 0;
  for (int row = 0; row < 10; ++row)
    total += fractional.keep(1280, 10, row);
  CHECK(total == 23);
  CHECK(parse_veda_budget("0%").keep(129, 4, 0) == 4);
  CHECK(parse_veda_budget("24").keep(129, 4, 0) == 4);
  CHECK(test::throws([] {
    parse_veda_budget("nan%");
  }));
  CHECK(test::throws([] {
    parse_veda_budget("100%");
  }));
  CHECK(test::throws([] {
    parse_veda_budget("0");
  }));
  CHECK(test::throws([] {
    parse_veda_budget("90%junk");
  }));
}

SLOPFAB_TEST(veda_plan_selection) {
  VedaPlan p{"landscape", {37, 24, 42}, {{4, 4, 8}}, {{0}}};
  auto selected = select_veda_plan({p}, {37, 42, 24});
  CHECK(selected.name == "landscape_T");
  CHECK((selected.shapes[0] == VedaShape{4, 8, 4}));
  auto longer = p;
  longer.name = "long";
  longer.grid[0] = 72;
  CHECK(select_veda_plan({p, longer}, {70, 24, 42}).name == "long");
}

SLOPFAB_TEST(veda_bundle_validation_and_rounding) {
  const auto path =
      (std::filesystem::temp_directory_path() / "slopfab_veda_host.safetensors").string();
  std::map<std::string, std::string> metadata{
      {"format", "miowtion-veda-predictor-v1"},
      {"num_layers", "1"},
      {"num_heads", "1"},
      {"head_dim", "64"},
      {"keep_ratio", "0.1"},
      {"plans",
       R"({"test":{"geometry":"test","grid":[4,4,8],"shapes":["4x4x8"],"head_shape":[[0]]}})"}};
  std::vector<TensorWrite> weights;
  for (const auto* name : {"layers.0.proj_q", "layers.0.proj_k"})
    weights.push_back({name, {1, 192, 64}, std::vector<float>(192 * 64, .123456f)});
  write_safetensors(path, weights, metadata);
  {
    VedaBundle bundle;
    bundle.load(path);
    CHECK(bundle.layers == 1 && bundle.heads == 1 && bundle.head_dim == 64);
    CHECK(bundle.projection(0, true)[0] == f32_to_bf16(.123456f));
  }
  metadata["plans"] =
      R"({"test":{"geometry":"test","grid":[4,4,8],"shapes":["4x4x8"],"head_shape":[[1]]}})";
  write_safetensors(path, weights, metadata);
  bool rejected = false;
  try {
    VedaBundle bundle;
    bundle.load(path);
  } catch (const std::exception&) {
    rejected = true;
  }
  CHECK(rejected);
  std::filesystem::remove(path);
}

SLOPFAB_TEST(veda_fp8_per_head_scales) {
  const auto path =
      (std::filesystem::temp_directory_path() / "slopfab_veda_fp8.safetensors").string();
  const size_t per_head = 192 * 64, n = 2 * per_head;
  std::vector<uint8_t> data(2 * n + 16, 0x38); // E4M3 1.0
  const float qscale[] = {0.25f, 2.0f}, kscale[] = {0.5f, 1.0f};
  std::memcpy(data.data() + 2 * n, qscale, 8);
  std::memcpy(data.data() + 2 * n + 8, kscale, 8);
  json::Object metadata;
  for (
      const auto& entry : std::map<std::string, std::string>{
          {"format", "miowtion-veda-predictor-v1"},
          {"num_layers", "1"},
          {"num_heads", "2"},
          {"head_dim", "64"},
          {"target_budget_kind", "tiles"},
          {"target_budget_value", "32"},
          {"dtype", "float8_e4m3fn"},
          {"plans",
           R"({"test":{"geometry":"test","grid":[4,4,8],"shapes":["4x4x8"],"head_shape":[[0,0]]}})"}})
    metadata.emplace(entry.first, json::Value(entry.second));
  json::Object header{{"__metadata__", json::Value(metadata)}};
  auto tensor = [&](const char* name, const char* dtype, const std::vector<int>& dims, size_t begin,
                    size_t end) {
    json::Array shape;
    for (int dim : dims)
      shape.emplace_back(double(dim));
    header.emplace(name,
                   json::Value(json::Object{
                       {"dtype", json::Value(std::string(dtype))},
                       {"shape", json::Value(shape)},
                       {"data_offsets", json::Value(json::Array{json::Value(double(begin)),
                                                                json::Value(double(end))})}}));
  };
  tensor("layers.0.proj_q", "F8_E4M3", {2, 192, 64}, 0, n);
  tensor("layers.0.proj_k", "F8_E4M3", {2, 192, 64}, n, 2 * n);
  tensor("layers.0.proj_q.__scale", "F32", {2}, 2 * n, 2 * n + 8);
  tensor("layers.0.proj_k.__scale", "F32", {2}, 2 * n + 8, 2 * n + 16);
  auto write = [&] {
    const auto text = json::stringify(json::Value(header));
    const uint64_t size = text.size();
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(&size), 8);
    file.write(text.data(), text.size());
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
  };
  write();
  {
    VedaBundle bundle;
    bundle.load(path);
    CHECK(bundle.generated.tiles && bundle.generated.value == 32);
    const auto q = bundle.projection(0, true), k = bundle.projection(0, false);
    CHECK(q[0] == f32_to_bf16(.25f));
    CHECK(q[per_head] == f32_to_bf16(2.0f));
    CHECK(k[0] == f32_to_bf16(.5f));
    CHECK(k[per_head] == f32_to_bf16(1.0f));
  }
  data[0] = 0x7f; // FP8 NaN must never reach routing.
  write();
  bool rejected = false;
  try {
    VedaBundle bundle;
    bundle.load(path);
    bundle.projection(0, true);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
  std::filesystem::remove(path);
}
