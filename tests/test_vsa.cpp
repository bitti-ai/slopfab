#include "harness.h"
#include "slopfab/dit/checkpoint.h"
#include "slopfab/dit/vsa.h"
#include "slopfab/pipeline.h"
#include "slopfab/safetensors_write.h"

#include <algorithm>
#include <filesystem>

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
      if (i >= tiles.sizes[t]) { CHECK(row == -1); continue; }
      CHECK(row >= 0 && row < layout.total_rows());
      ++seen.at(row);
      CHECK(tiles.row_tiles.at(row) == t);
    }
  }
  CHECK(std::all_of(seen.begin(), seen.end(), [](int n) { return n == 1; }));
  layout.num_condition_audio = 1;
  bool rejected = false;
  try { (void)slopfab::dit::build_vsa_tiles(layout); } catch (...) { rejected = true; }
  CHECK(rejected);
}

SLOPFAB_TEST(vsa_checkpoint_and_schedule) {
  using namespace slopfab;
  const auto path = std::filesystem::temp_directory_path() / "slopfab_fasth3_8step_v2_test.safetensors";
  const std::vector<TensorWrite> tensors = {
      {"adaln_t_table", {1}, {0}},
      {"blocks.0.adaln_proj.linear.weight", {1}, {0}},
      {"blocks.0.attn.to_gate_compress.weight", {1}, {0}}};
  write_safetensors(path.string(), tensors);
  {
    SafeTensors st; st.open(path.string());
    CHECK(dit::detect_transformer_architecture(st) == dit::TransformerArchitecture::kFastH3V2PrunedTable);
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
  std::filesystem::remove(path);
  const auto renamed = std::filesystem::temp_directory_path() / "slopfab_vsa_renamed.safetensors";
  write_safetensors(renamed.string(), tensors, {{"model_id", "FastVideo/FastVideo-FastH3-8-Step-V2"}});
  {
    SafeTensors st; st.open(renamed.string());
    CHECK(dit::detect_transformer_architecture(st) == dit::TransformerArchitecture::kFastH3V2PrunedTable);
  }
  std::filesystem::remove(renamed);
}
