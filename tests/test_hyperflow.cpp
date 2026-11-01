#include "harness.h"
#include "slopfab/dit/hyperflow.h"
#include "slopfab/generate.h"
#include "slopfab/lora.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/sha256.h"
#include <filesystem>

namespace {
using namespace slopfab;
std::map<std::string, std::string> metadata() {
  return {{"hyperflow", "true"}, {"hyperflow_version", "1.0"},
          {"base_model", "MiniMaxAI/MiniMax-H3"}, {"base_model_revision", "fixture"},
          {"hyperflow_gate", "0.25"}, {"lora_rank", "1"}, {"lora_alpha", "2"},
          {"hyperflow_video_shift", "12"}, {"hyperflow_audio_shift", "3"},
          {"hyperflow_sigmas", "[1,0.931506,0.839236,0.703462,0.5,0.296538,0.160764,0.068494,0]"}};
}
template<class F> bool rejects(F fn) {
  try { fn(); } catch (const std::exception&) { return true; }
  return false;
}
struct Fixture {
  std::filesystem::path dir = std::filesystem::temp_directory_path() / "slopfab_hyperflow_test";
  std::string adapter, base_path, companion;
  std::vector<float> pairs;
  Fixture() {
    std::filesystem::create_directories(dir);
    adapter = (dir / "adapter.safetensors").string();
    base_path = (dir / "base.safetensors").string();
    companion = (dir / "adapter.hyperflow-fl2va.safetensors").string();
    std::vector<TensorWrite> tensors = {
        {"transformer.transformer_blocks.0.attn.to_q.lora_A.weight", {1,2}, {1,2}},
        {"transformer.transformer_blocks.0.attn.to_q.lora_B.weight", {2,1}, {3,4}}};
    for (const char* module : {"time_embedder", "endpoint_time_embedder"})
      for (const char* layer : {"linear_1", "linear_2"})
        for (const char* factor : {"A", "B"})
          tensors.push_back({"transformer." + std::string(module) + "." + layer + ".lora_" +
              factor + ".weight", {1,1}, {1}});
    write_safetensors(adapter, tensors, metadata());
    write_safetensors(base_path, {
        {"adaln_t_table", {1025,8}, std::vector<float>(1025*8)},
        {"blocks.0.attn.qkv_proj.weight", {6,2}, std::vector<float>(12)},
        {"blocks.0.adaln_proj.linear.weight", {36,8}, std::vector<float>(36*8)},
        {"blocks.0.adaln_proj.linear.bias", {36}, std::vector<float>(36)},
        {"final_layer.adaln_proj.linear.weight", {4,8}, std::vector<float>(32)},
        {"final_layer.adaln_proj.linear.bias", {4}, std::vector<float>(4)}});
    const auto hf = dit::hyperflow_metadata(metadata());
    for (float shift : {12.f, 3.f}) {
      sampler::FlowScheduler scheduler(shift);
      scheduler.set_base_sigmas(*hf->sampling.base_sigmas);
      for (int i = 0; i < 8; ++i) {
        pairs.push_back(scheduler.timesteps()[i]);
        pairs.push_back(1.f - scheduler.sigmas()[i+1]);
      }
    }
    pairs.insert(pairs.end(), {.999f, .999f, 1, 1});
    write_companion();
  }
  void write_companion(float strength = 1, std::string hash = {}) {
    if (hash.empty())
      for (uint8_t b : sha256_file(adapter)) {
        hash += "0123456789abcdef"[b >> 4]; hash += "0123456789abcdef"[b & 15];
      }
    std::vector<float> w(36*18);
    for (size_t i = 0; i < w.size(); ++i) w[i] = float(i);
    write_safetensors(companion, {{"pairs", {18,2}, pairs},
        {"blocks.0.adaln_proj.linear", {36,18}, w},
        {"final_layer.adaln_proj.linear", {4,18}, std::vector<float>(4*18, 7)}},
        {{"slopfab.hyperflow", "1"}, {"variant", "fl2va"},
         {"base_model_revision", "fixture"}, {"strength", std::to_string(strength)},
         {"adapter_sha256", hash}});
  }
  ~Fixture() { std::error_code ec; std::filesystem::remove_all(dir, ec); }
};
}

SLOPFAB_TEST(hyperflow_metadata_and_sampling_contract) {
  auto m = metadata();
  const auto hf = dit::hyperflow_metadata(m);
  CHECK(hf && hf->gate == .25f && hf->alpha == 2 && hf->rank == 1);
  CHECK(sampling_settings_from_metadata(m).base_sigmas->size() == 9);
  CHECK(!dit::hyperflow_metadata({}));
  for (const auto& change : std::vector<std::pair<std::string, std::string>>{
      {"hyperflow", "false"}, {"hyperflow_gate", "1e100"}, {"lora_rank", "1.5"},
      {"hyperflow_sigmas", "[1,0]"}, {"hyperflow_audio_shift", "0"}}) {
    auto bad = m; bad[change.first] = change.second;
    CHECK(rejects([&] { dit::hyperflow_metadata(bad); }));
  }
  Fixture f;
  GenerateRequest request;
  request.loras = {{f.adapter, 1}};
  auto plan = resolve_plan(request);
  CHECK(plan.hyperflow && plan.num_model_evaluations() == 8);
  CHECK(plan.video_sigma_shift == 12 && plan.audio_sigma_shift == 3);
  CHECK(rejects([&] { validate_sampling_sampler(plan, sampler::SamplerKind::kRenoise); }));
  request.num_inference_steps = 8;
  CHECK(resolve_plan(request).num_inference_steps == 8);
  request.num_inference_steps = 9;
  CHECK(rejects([&] { resolve_plan(request); }));
  request.num_inference_steps = 0;
  request.sampling.audio_steps = 5;
  CHECK(rejects([&] { resolve_plan(request); }));
  request.sampling = {};
  request.loras[0].strength = 0;
  CHECK(!resolve_plan(request).hyperflow);
}

SLOPFAB_TEST(hyperflow_distinguishes_equal_start_times_and_pins_conditions) {
  dit::SequenceLayout layout;
  layout.num_text = 1; layout.num_video_rows = 1; layout.num_audio_latents = 1; layout.num_audio_rows = 2;
  layout.num_condition_video = 1; layout.num_condition_audio = 1;
  layout.condition_audio_is_explicit = true;
  const auto indices = dit::build_indices(layout);
  const auto times = dit::build_hyperflow_timesteps(layout, indices, 0, .1f, 0, .2f);
  CHECK(times.unique.size() == 4);
  const auto video = times.indices[indices.video.back()];
  const auto audio = times.indices[indices.audio.back()];
  CHECK(video != audio && times.unique[video] == 0 && times.unique[audio] == 0);
  CHECK(times.endpoints[video] == .1f && times.endpoints[audio] == .2f);
  const auto cv = times.indices[indices.video.front()], ca = times.indices[indices.audio.front()];
  CHECK(times.unique[cv] == .999f && times.endpoints[cv] == .999f);
  CHECK(times.unique[ca] == 1 && times.endpoints[ca] == 1);
  CHECK(times.indices[0] == video);
}

SLOPFAB_TEST(hyperflow_companion_identity_code_and_lora_loading) {
  Fixture f;
  SafeTensors base; base.open(f.base_path);
  {
    LoraAdapters adapters;
    adapters.load({{f.adapter, 1}}, base);
    CHECK(adapters.projection_count() == 1 && adapters.hyperflow()->rank() == 18);
    CHECK(adapters.find("blocks.0.attn.to_q")->front().b[0] == 6); // alpha/rank
    dit::RowTimesteps times;
    times.unique = {f.pairs[0], f.pairs[16]};
    times.endpoints = {f.pairs[1], f.pairs[17]};
    const auto code = adapters.hyperflow()->code(times, 4);
    CHECK(code.size() == 72 && code[0] == 1 && code[18+8] == 1 && code[54+8] == 1);
    const auto w = adapters.merged_adaln_weight(base, "blocks.0.adaln_proj.linear");
    CHECK(w.size() == 36*18 && w[18+8] == 26);
    CHECK(adapters.merged_adaln_bias(base, "blocks.0.adaln_proj.linear")[0] == 0);
    times.endpoints.clear();
    CHECK(rejects([&] { adapters.hyperflow()->code(times); }));
    CHECK(rejects([&] { adapters.load({{f.adapter, .5f}}, base); }));
    CHECK(bool(adapters.hyperflow())); // failed reload is transactional
  }
  f.write_companion(1, "wrong");
  LoraAdapters adapters;
  CHECK(rejects([&] { adapters.load({{f.adapter, 1}}, base); }));
  adapters.load({{f.adapter, 0}}, base);
  CHECK(!adapters.hyperflow() && adapters.projection_count() == 0);
  f.write_companion(-1);
  adapters.load({{f.adapter, -1}}, base);
  CHECK(adapters.find("blocks.0.attn.to_q")->front().b[0] == -6);
}
