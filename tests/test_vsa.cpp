#include "harness.h"
#include "slopfab/dit/checkpoint.h"
#include "slopfab/dit/vsa.h"
#include "slopfab/dit/pdd.h"
#include "slopfab/pipeline.h"
#include "slopfab/safetensors_write.h"

#include <algorithm>
#include <filesystem>

SLOPFAB_TEST(omniref_fused_heads_match_fine_grid_integration) {
  using namespace slopfab;
  // The archive stores head zero followed by residuals relative to head zero.
  std::vector<float> values(32 * 6);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = float(int(i % 29) - 14) / 17;
  TensorView tensor{"head", DType::kF32, {64, 3}, values.data(), values.size() * 4};
  for (float shift : {3.f, 12.f}) {
    const auto fused = dit::fuse_omniref_heads(tensor, shift);
    const auto grid = dit::omniref_sigmas(shift);
    auto sigma = [shift](int j) {
      double u = 1 - j / 32.;
      return .999 * shift * u / (1 + (shift - 1) * u);
    };
    CHECK(grid.size() == 9);
    for (int step = 0; step < 8; ++step) {
      CHECK_NEAR(grid[step], sigma(step * 4), 6e-8);
      for (int i = 0; i < 6; ++i) {
        double integrated = 0;
        for (int j = step * 4; j < step * 4 + 4; ++j)
          integrated += (sigma(j) - sigma(j + 1)) * (values[i] + (j ? values[j * 6 + i] : 0));
        CHECK_NEAR(fused[step * 6 + i], integrated / (sigma(step * 4) - sigma(step * 4 + 4)), 3e-7);
      }
    }
  }
  const auto grid = dit::omniref_sigmas(12);
  for (int i = 0; i < 8; ++i)
    CHECK(dit::omniref_step(1 - grid[i]) == i);
  bool rejected = false;
  try {
    (void)dit::omniref_step(.5f);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  CHECK(rejected);
}

SLOPFAB_TEST(omniref_reference_tile_geometry) {
  slopfab::dit::SequenceLayout l;
  l.num_text = 3;
  l.num_condition_audio = 5;
  l.num_condition_video = 9 + 5 * 5 * 9;
  l.reference_segments = {{3, 9}, {12, 5}, {17, 225, 5, 5, 9}};
  l.num_audio_rows = 7;
  l.num_latent_frames = 5;
  l.latent_height = 10;
  l.latent_width = 18;
  l.num_video_rows = 225;
  auto tiles = slopfab::dit::build_vsa_tiles(l, 128, 10);
  CHECK(tiles.prefix_tiles == 4);
  CHECK(tiles.region_ends == std::vector<int32_t>({12, 20}));
  CHECK(tiles.sizes[4] == 128);
  CHECK(tiles.rows[4 * 128] == 17);
  std::vector<int> seen(l.total_rows());
  for (size_t tile = 0; tile < tiles.sizes.size(); ++tile)
    for (int i = 0; i < 128; ++i) {
      int row = tiles.rows[tile * 128 + i];
      if (i >= tiles.sizes[tile]) {
        CHECK(row == -1);
        continue;
      }
      CHECK(tiles.row_tiles.at(row) == tile);
      ++seen.at(row);
    }
  CHECK(std::all_of(seen.begin(), seen.end(), [](int n) {
    return n == 1;
  }));
}

SLOPFAB_TEST(vsa_tile_geometry) {
  slopfab::dit::SequenceLayout layout;
  layout.num_text = 65;
  layout.num_audio_rows = 67;
  layout.num_latent_frames = 5;
  layout.latent_height = 10;
  layout.latent_width = 14;
  layout.num_video_rows = 5 * 5 * 7;
  const auto tiles = slopfab::dit::build_vsa_tiles(layout);
  CHECK(tiles.prefix_tiles == 4);
  CHECK(tiles.sizes.size() == 12);
  CHECK(tiles.sizes[0] == 64 && tiles.sizes[1] == 1);
  CHECK(tiles.sizes[2] == 64 && tiles.sizes[3] == 3);
  CHECK(tiles.sizes[4] == 64 && tiles.sizes[5] == 48);
  CHECK(tiles.sizes.back() == 3);
  CHECK(tiles.rows[4 * 64] == 132);
  CHECK(tiles.rows[4 * 64 + 4] == 139);
  CHECK(tiles.rows[4 * 64 + 16] == 167);
  std::vector<int> seen(layout.total_rows());
  for (size_t t = 0; t < tiles.sizes.size(); ++t) {
    for (int i = 0; i < 64; ++i) {
      const int row = tiles.rows[t * 64 + i];
      if (i >= tiles.sizes[t]) {
        CHECK(row == -1);
        continue;
      }
      CHECK(row >= 0 && row < layout.total_rows());
      ++seen.at(row);
      CHECK(tiles.row_tiles.at(row) == t);
    }
  }
  CHECK(std::all_of(seen.begin(), seen.end(), [](int n) {
    return n == 1;
  }));
  layout.num_condition_audio = 1;
  bool rejected = false;
  try {
    (void)slopfab::dit::build_vsa_tiles(layout);
  } catch (...) {
    rejected = true;
  }
  CHECK(rejected);
}

SLOPFAB_TEST(vsa_checkpoint_and_schedule) {
  using namespace slopfab;
  const auto path =
      std::filesystem::temp_directory_path() / "slopfab_fasth3_8step_v2_test.safetensors";
  const std::vector<TensorWrite> tensors = {{"adaln_t_table", {1}, {0}},
                                            {"blocks.0.adaln_proj.linear.weight", {1}, {0}},
                                            {"blocks.0.attn.to_gate_compress.weight", {1}, {0}}};
  write_safetensors(path.string(), tensors);
  {
    SafeTensors st;
    st.open(path.string());
    CHECK(dit::detect_transformer_architecture(st) ==
          dit::TransformerArchitecture::kFastH3V2PrunedTable);
  }
  GenerateRequest request;
  request.prompt = "A forest";
  request.transformer_path = path.string();
  auto plan = resolve_plan(request);
  CHECK(plan.fasth3_v2);
  CHECK(plan.num_model_evaluations() == 8);
  CHECK(plan.video_sigma_shift == 10 && plan.audio_sigma_shift == 3);
  const int rungs[] = {999, 874, 749, 624, 500, 375, 250, 125, 0};
  for (int i = 0; i < 9; ++i) {
    const double base = rungs[i] / 1000.0;
    CHECK_NEAR(plan.video_sigmas[i], 10 * base / (1 + 9 * base), 2e-7);
    CHECK_NEAR(plan.audio_sigmas[i], 3 * base / (1 + 2 * base), 2e-7);
  }
  CHECK(plan.video_sigmas.front() < 1.0f);
  sampler::FlowScheduler replay(10);
  replay.set_sigmas(plan.video_sigmas);
  CHECK(replay.timesteps() == plan.video_timesteps);
  request.reference_image_paths.push_back("reference.png");
  bool rejected = false;
  try {
    (void)resolve_plan(request);
  } catch (...) {
    rejected = true;
  }
  CHECK(rejected);
  request.reference_image_paths.clear();
  request.schedule = sampler::ScheduleKind::kTaoMate3Step;
  request.loras.push_back({"taomate.safetensors", 1.0f});
  rejected = false;
  try {
    (void)resolve_plan(request);
  } catch (...) {
    rejected = true;
  }
  CHECK(rejected);
  std::filesystem::remove(path);
  const auto renamed = std::filesystem::temp_directory_path() / "slopfab_vsa_renamed.safetensors";
  write_safetensors(renamed.string(), tensors,
                    {{"model_id", "FastVideo/FastVideo-FastH3-8-Step-V2"}});
  {
    SafeTensors st;
    st.open(renamed.string());
    CHECK(dit::detect_transformer_architecture(st) ==
          dit::TransformerArchitecture::kFastH3V2PrunedTable);
  }
  std::filesystem::remove(renamed);
  write_safetensors(renamed.string(), tensors, {{"model_id", "FastVideo/FastH3-V1"}});
  {
    SafeTensors st;
    st.open(renamed.string());
    CHECK(dit::detect_transformer_architecture(st) == dit::TransformerArchitecture::kUnknown);
  }
  std::filesystem::remove(renamed);
}
