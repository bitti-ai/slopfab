#include "harness.h"
#include <filesystem>
#include <limits>
#include "slopfab/lora.h"
#include "slopfab/pipeline.h"
#include "slopfab/safetensors_write.h"

namespace {
using namespace slopfab;
struct Fixture {
  std::filesystem::path dir = std::filesystem::temp_directory_path() / "slopfab_lora_test";
  SafeTensors base;
  std::string path;
  Fixture() {
    std::filesystem::create_directories(dir);
    const auto base_path = (dir / "base.safetensors").string();
    write_safetensors(base_path, {{"blocks.0.attn.qkv_proj.weight", {6, 3}, std::vector<float>(18)},
        {"token_refiner.blocks.1.mlp.fc2.weight", {6, 3}, std::vector<float>(18)}});
    base.open(base_path);
    path = (dir / "adapter.safetensors").string();
  }
  ~Fixture() { base.close(); std::filesystem::remove_all(dir); }
  std::vector<TensorWrite> tensors(std::string prefix = "diffusion_model.blocks.0.attn.qkv_proj",
                                  std::string a = ".lora_A.weight", std::string b = ".lora_B.weight") {
    return {{prefix + a, {2, 3}, {1, 2, 3, 4, 5, 6}, DType::kBF16},
            {prefix + b, {6, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, DType::kF16},
            {prefix + ".alpha", {}, {4}}};
  }
};
}

SLOPFAB_TEST(lora_formats_strength_stacking_and_disable) {
  Fixture f;
  for (int format = 0; format < 3; ++format) {
    const std::string target = format == 2 ? "token_refiner.blocks.1.mlp.fc2" : "blocks.0.attn.qkv_proj";
    const std::string prefix = format == 1 ? "base_model.model." : "diffusion_model.";
    const auto tensors = f.tensors(prefix + target,
        format == 1 ? ".lora_A.default.weight" : format == 2 ? ".lora_down.weight" : ".lora_A.weight",
        format == 1 ? ".lora_B.default.weight" : format == 2 ? ".lora_up.weight" : ".lora_B.weight");
    write_safetensors(f.path, tensors);
    LoraAdapters loras;
    loras.load({{f.path, -0.25f}}, f.base);
    CHECK(loras.projection_count() == 1);
    const auto& pair = loras.find(target)->front();
    CHECK(pair.rank == 2 && pair.in == 3 && pair.out == 6);
    CHECK(pair.a == tensors[0].data);
    for (size_t i = 0; i < pair.b.size(); ++i) CHECK_NEAR(pair.b[i], -0.5f * tensors[1].data[i], 0);
    loras.load({{f.path, 0.5f}, {f.path, -0.25f}}, f.base);
    const auto& combined = loras.find(target)->front();
    CHECK(combined.rank == 4);
    // Check BA numerically: verifies row-wise B concatenation, rather than
    // merely comparing the implementation's chosen storage layout.
    for (int row = 0; row < 6; ++row) for (int col = 0; col < 3; ++col) {
      float actual = 0, expected = 0;
      for (int r = 0; r < 4; ++r) actual += combined.b[row*4+r] * combined.a[r*3+col];
      for (int r = 0; r < 2; ++r) expected += 0.5f*tensors[1].data[row*2+r]*tensors[0].data[r*3+col];
      CHECK_NEAR(actual, expected, 0);
    }
    loras.load({{f.path, 0}}, f.base);
    CHECK(loras.projection_count() == 0);
  }
}

SLOPFAB_TEST(lora_rejects_malformed_or_incompatible_adapters_transactionally) {
  Fixture f;
  const auto valid = f.tensors();
  write_safetensors(f.path, valid);
  LoraAdapters loras;
  loras.load({{f.path, 1}}, f.base);
  for (int mutation = 0; mutation < 8; ++mutation) {
    auto bad = valid;
    if (mutation == 0) bad.erase(bad.begin());
    if (mutation == 1) { bad[0].shape = {3, 2}; }
    if (mutation == 2) bad[0].data[0] = std::numeric_limits<float>::infinity();
    if (mutation == 3) bad[2].data[0] = std::numeric_limits<float>::quiet_NaN();
    if (mutation == 4) bad[2] = {bad[2].name, {2}, {1, 2}};
    if (mutation == 5) bad[0].name = "diffusion_model.condition_proj.lora_A.weight";
    if (mutation == 6) bad.push_back({"unknown", {}, {1}});
    if (mutation == 7) { bad[0].shape = {2, 4}; bad[0].data.resize(8); }
    write_safetensors(f.path, bad);
    bool rejected = false;
    try { loras.load({{f.path, 1}}, f.base); } catch (const std::exception&) { rejected = true; }
    CHECK(rejected);
    CHECK(loras.projection_count() == 1);
    CHECK(loras.find("blocks.0.attn.qkv_proj")->front().rank == 2);
  }
  auto plain = valid; plain.pop_back();
  write_safetensors(f.path, plain);
  loras.load({{f.path, 1}}, f.base);
  CHECK(loras.find("blocks.0.attn.qkv_proj")->front().b == plain[1].data);
}

SLOPFAB_TEST(lora_viggle_diffusers_mapping_and_endpoint_merge) {
  Fixture f;
  f.base.close();
  const auto base_path = (f.dir / "base.safetensors").string();
  write_safetensors(base_path, {
      {"blocks.0.attn.qkv_proj.weight", {6, 3}, std::vector<float>(18)},
      {"blocks.0.attn.out_proj.weight", {6, 2}, std::vector<float>(12)},
      {"blocks.0.mlp.fc1.weight", {8, 3}, std::vector<float>(24)},
      {"blocks.0.mlp.fc2.weight", {6, 4}, std::vector<float>(24)},
      {"video_patch_proj.weight", {6, 3}, std::vector<float>(18, 1)},
      {"final_layer.video_out.weight", {3, 6}, std::vector<float>(18, -1)}},
      {{"source", "Viggle/Viggle-Animate"}, {"qkv_layout", "interleaved"}});
  f.base.open(base_path);
  std::vector<TensorWrite> adapter;
  auto add = [&](const std::string& name, int in, int out) {
    std::vector<float> a(in), b(out);
    for (int i = 0; i < in; ++i) a[i] = float(i + 1);
    for (int i = 0; i < out; ++i) b[i] = float(i + 2);
    adapter.push_back({name + ".lora_A.weight", {1, in}, a, DType::kBF16});
    adapter.push_back({name + ".lora_B.weight", {out, 1}, b, DType::kBF16});
  };
  add("proj_in", 3, 6);
  add("proj_out", 6, 3);
  add("transformer_blocks.0.attn.to_q", 3, 2);
  add("transformer_blocks.0.attn.to_k", 3, 2);
  add("transformer_blocks.0.attn.to_v", 3, 2);
  add("transformer_blocks.0.attn.to_out.0", 2, 6);
  add("transformer_blocks.0.ff.net.0.proj", 3, 8);
  add("transformer_blocks.0.ff.net.2", 4, 6);
  const std::map<std::string, std::string> metadata = {
      {"lora_adapter_metadata", R"({"lora_alpha":2,"r":1,"use_dora":false,"alpha_pattern":{}})"}};
  write_safetensors(f.path, adapter, metadata);
  LoraAdapters loras;
  loras.load({{f.path, .25f}}, f.base);
  CHECK(loras.projection_count() == 8);
  CHECK(loras.find("blocks.0.attn.qkv_proj") == nullptr);
  for (const char* part : {"to_q", "to_k", "to_v"}) {
    const auto& q = loras.find(std::string("blocks.0.attn.") + part)->front();
    CHECK(q.out == 2 && q.in == 3 && q.rank == 1);
    CHECK(q.b == std::vector<float>({1, 1.5f})); // No base-QKV interleave applied to split factors.
  }
  CHECK(loras.find("blocks.0.attn.out_proj")->front().in == 2);
  CHECK(loras.find("blocks.0.mlp.fc2")->front().in == 4);
  // Swapped [value; gate] -> [gate; value], with alpha/rank and strength.
  CHECK(loras.find("blocks.0.mlp.fc1")->front().b ==
        std::vector<float>({3, 3.5f, 4, 4.5f, 1, 1.5f, 2, 2.5f}));
  for (const char* name : {"video_patch_proj", "final_layer.video_out"}) {
    const auto& shape = f.base.at(std::string(name) + ".weight").shape;
    const auto merged = loras.merged_endpoint_weight(f.base, name);
    const float initial = std::string(name) == "video_patch_proj" ? 1.0f : -1.0f;
    for (int row = 0; row < shape[0]; ++row) for (int col = 0; col < shape[1]; ++col)
      CHECK_NEAR(merged[row * shape[1] + col], initial + .5f * (row + 2) * (col + 1), 0);
  }
  // Stacked opposite strengths cancel for endpoint and block factors alike.
  loras.load({{f.path, .25f}, {f.path, -.25f}}, f.base);
  CHECK(loras.merged_endpoint_weight(f.base, "video_patch_proj") == std::vector<float>(18, 1));
  auto bad = adapter;
  bad[5].shape = {3, 1}; bad[5].data.resize(3); // Q output must be one third of the base QKV rows.
  write_safetensors(f.path, bad, metadata);
  bool rejected = false;
  try { loras.load({{f.path, 1}}, f.base); } catch (const std::exception&) { rejected = true; }
  CHECK(rejected && loras.projection_count() == 8);
  for (const char* setting : {R"({"use_dora":true})", R"({"use_rslora":true})",
                              R"({"alpha_pattern":{"proj_in":4}})"}) {
    write_safetensors(f.path, adapter, {{"lora_adapter_metadata", setting}});
    rejected = false;
    try { loras.load({{f.path, 1}}, f.base); } catch (const std::exception&) { rejected = true; }
    CHECK(rejected && loras.projection_count() == 8);
  }
}

SLOPFAB_TEST(lora_viggle_released_adapter_compatibility) {
  const std::string adapter = "weights/loras/viggle_animate_distillation_bf16.safetensors";
  const std::string checkpoint = "weights/transformer/Viggle-Animate-pruned_rank8_int8_convrot.safetensors";
  if (!std::filesystem::exists(adapter) || !std::filesystem::exists(checkpoint)) {
    SKIP_MISSING_FIXTURE("Viggle adapter or checkpoint unavailable");
    return;
  }
  SafeTensors base;
  base.open(checkpoint);
  LoraAdapters loras;
  loras.load({{adapter, 1}}, base);
  CHECK(loras.projection_count() == 302);
  for (int layer = 0; layer < 50; ++layer) {
    const std::string p = "blocks." + std::to_string(layer) + ".";
    for (const char* name : {"attn.to_q", "attn.to_k", "attn.to_v", "attn.out_proj", "mlp.fc1", "mlp.fc2"}) {
      const auto* factors = loras.find(p + name);
      CHECK(factors && factors->size() == 1 && factors->front().rank == 128);
    }
  }
  for (const char* name : {"video_patch_proj", "final_layer.video_out"}) {
    CHECK(loras.find(name)->front().rank == 128);
    const auto merged = loras.merged_endpoint_weight(base, name);
    CHECK(merged.size() == size_t(base.at(std::string(name) + ".weight").numel()));
  }
}

SLOPFAB_TEST(lora_taomate_three_step_schedule_and_plan) {
  using namespace slopfab::sampler;
  for (float shift : {12.0f, 3.0f}) {
    FlowScheduler s(shift);
    s.set_timesteps(4, ScheduleKind::kTaoMate3Step);
    CHECK(s.num_steps() == 3);
    CHECK(s.sigmas().size() == 4 && s.sigmas()[0] == 1 && s.sigmas()[3] == 0);
    CHECK_NEAR(s.sigmas()[1], shift * 33.0 / (16.0 + shift * 33.0), 1e-7);
    CHECK_NEAR(s.sigmas()[2], shift * 16.0 / (33.0 + shift * 16.0), 1e-7);
    FlowScheduler replay(shift);
    replay.set_sigmas(s.sigmas());
    CHECK(replay.timesteps() == s.timesteps());
    float x = 2, velocity = 3;
    for (int i = 0; i < 3; ++i) replay.step(i, &x, &velocity, 1, &x);
    CHECK_NEAR(x, 5, 1e-6);
    s.set_timesteps(4);
    CHECK(s.sigmas() != replay.sigmas());
  }
  GenerateRequest request;
  request.prompt = "test";
  request.schedule = ScheduleKind::kTaoMate3Step;
  bool rejected = false;
  try { (void)resolve_plan(request); } catch (const std::exception&) { rejected = true; }
  CHECK(rejected);
  request.loras.push_back({"TaoMate.safetensors", 1});
  const auto plan = resolve_plan(request);
  CHECK(plan.num_inference_steps == 4);
  CHECK(plan.num_model_evaluations() == 3);
  FlowScheduler s;
  for (const auto& grid : std::vector<std::vector<float>>{{1, 1, 0}, {1, .2f, .3f, 0}, {1, .1f},
      {1, std::numeric_limits<float>::quiet_NaN(), 0}}) {
    bool invalid = false;
    try { s.set_sigmas(grid); } catch (const std::exception&) { invalid = true; }
    CHECK(invalid);
  }
}

SLOPFAB_TEST(lora_taomate_released_adapter_compatibility) {
  const std::string adapter = "weights/loras/TaoMate-H3-3step-ComfyUI.safetensors";
  const std::string checkpoint = "weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors";
  if (!std::filesystem::exists(adapter) || !std::filesystem::exists(checkpoint)) {
    SKIP_MISSING_FIXTURE("TaoMate adapter or NVFP4 checkpoint unavailable");
    return;
  }
  SafeTensors base;
  base.open(checkpoint);
  LoraAdapters loras;
  loras.load({{adapter, 1}}, base);
  CHECK(loras.projection_count() == 208);
  CHECK(loras.find("token_refiner.blocks.1.mlp.fc2")->front().rank == 128);
  CHECK(loras.find("blocks.49.attn.qkv_proj")->front().out == 21504);
}
