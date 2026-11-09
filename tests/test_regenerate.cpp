#include "harness.h"
#include "slopfab/generate.h"
#include "slopfab/safetensors_write.h"
#include <filesystem>
#include <cmath>

using namespace slopfab;
namespace {
template<class F> bool rejects(F f) {
  try { f(); } catch (const std::exception&) { return true; }
  return false;
}
struct Fixture {
  std::filesystem::path dir = std::filesystem::temp_directory_path() / "slopfab_regenerate_test";
  GenerateRequest request;
  Fixture() {
    std::filesystem::create_directories(dir);
    const auto ic = (dir / "ic.safetensors").string();
    std::vector<TensorWrite> tensors{{"transformer.token_refiner.refiner_blocks.0.attn.to_k.lora_A.weight",
                                     {384, 5376}, std::vector<float>(384*5376)}};
    for (int i = 1; i < 624; ++i) tensors.push_back({"fixture." + std::to_string(i), {1}, {0}});
    write_safetensors(ic, tensors, {{"target", "MiniMaxAI/MiniMax-H3 transformer_ref (Ref2VA transformer)"},
                                    {"training_step", "6000"}});
    const auto hf = (dir / "hf.safetensors").string();
    write_safetensors(hf, {{"unused", {1}, {0}}},
      {{"hyperflow", "true"}, {"hyperflow_version", "1.0"}, {"base_model", "MiniMaxAI/MiniMax-H3"},
       {"base_model_revision", "fixture"}, {"hyperflow_gate", "0.25"}, {"lora_rank", "1"}, {"lora_alpha", "2"},
       {"hyperflow_video_shift", "12"}, {"hyperflow_audio_shift", "3"},
       {"hyperflow_sigmas", "[1,0.931506,0.839236,0.703462,0.5,0.296538,0.160764,0.068494,0]"}});
    auto video = ReferenceMedia::video(124.0 / 24);
    std::vector<uint8_t> rgb(1344*768*3, 127);
    video.append_frame(rgb.data(), rgb.size(), 1344, 768, 1344*3, 3, 0);
    request.reference_media = {std::make_shared<const ReferenceMedia>(std::move(video))};
    request.regenerate.mode = "up3";
    request.loras = {{ic, 1}, {hf, 1}};
  }
  ~Fixture() { std::error_code ec; std::filesystem::remove_all(dir, ec); }
};
}

SLOPFAB_TEST(regenerate_schedule_tails_and_conditioning) {
  Fixture f;
  auto& r = f.request;
  const auto key = conditioning_cache_key(r);
  for (const char* mode : {"up3", "up4", "hf8"}) {
    r.regenerate.mode = mode;
    const auto p = resolve_plan(r);
    const int steps = regenerate_steps(mode);
    CHECK(p.hyperflow && p.num_inference_steps == steps);
    CHECK(p.canvas_width == 2560 && p.canvas_height == 1440);
    CHECK(p.layout.num_latent_frames == 37 && p.layout.num_audio_latents == 207);
    CHECK(p.conditioning.pin_target_audio && !p.conditioning.include_reference_audio);
    CHECK(!p.conditioning.references_at_target_canvas && p.conditioning.media_short_edge == 768);
    CHECK(!p.conditioning.video_first && p.layout.num_condition_audio == 0);
    sampler::FlowScheduler full(12), audio(3);
    const std::vector<float> base{1,.931506f,.839236f,.703462f,.5f,.296538f,.160764f,.068494f,0};
    full.set_base_sigmas(base); audio.set_base_sigmas(base);
    CHECK(p.video_sigmas == std::vector<float>(full.sigmas().end()-steps-1, full.sigmas().end()));
    CHECK(p.audio_sigmas == std::vector<float>(audio.sigmas().end()-steps-1, audio.sigmas().end()));
    r.num_inference_steps = steps;
    CHECK(resolve_plan(r).num_inference_steps == steps);
    r.num_inference_steps = 0;
  }
  CHECK(conditioning_cache_key(r) != key);
  r.num_inference_steps = 3;
  CHECK(rejects([&] { resolve_plan(r); }));
  r.num_inference_steps = 0;
  r.guidance_scale_set = true;
  CHECK(rejects([&] { resolve_plan(r); }));
  r.guidance_scale_set = false;
  r.conditioning.include_reference_audio = true;
  CHECK(rejects([&] { resolve_plan(r); }));
  r.conditioning = {};
  r.num_frames = 90;
  CHECK(rejects([&] { resolve_plan(r); }));
  r.num_frames = 124;
  r.loras.pop_back();
  CHECK(rejects([&] { resolve_plan(r); }));
  r.loras.clear();
  CHECK(rejects([&] { resolve_plan(r); }));
}

SLOPFAB_TEST(regenerate_silence_transpose_and_exact_target) {
  Fixture f;
  const auto path = (f.dir / "silence.safetensors").string();
  std::vector<float> values(2*32*207);
  for (size_t i = 0; i < values.size(); ++i) values[i] = float(i);
  write_safetensors(path, {{"audio_latents", {2,32,207}, values}});
  const auto rows = regenerate_silence(path);
  CHECK(rows[32+5] == values[5*207+1]);
  CHECK(rows[207*32+31] == values[(32+31)*207]);
  CHECK(rows.back() == values.back());
  write_safetensors(path, {{"audio_latents", {1}, {0}}});
  CHECK(rejects([&] { regenerate_silence(path); }));
  LatentUpscaleOptions o;
  o.target_width = 2560; o.target_height = 1440;
  CHECK(latent_upscale_dimensions(48,84,o) == std::make_pair(90,160));
  o.target_height = 1080;
  CHECK(rejects([&] { latent_upscale_dimensions(48,84,o); }));
  CHECK(regenerate_prompt("original caption").find("Scene description: original caption") != std::string::npos);
}

SLOPFAB_TEST(regenerate_released_lora_ref2va_compatibility) {
  const std::string adapter = "weights/open_h3_regenerate_2k_ic_lora.safetensors";
  const std::string checkpoint = "weights/transformer/minimax_h3_ref2va_pruned_nvfp4.safetensors";
  if (!std::filesystem::exists(adapter) || !std::filesystem::exists(checkpoint)) {
    SKIP_MISSING_FIXTURE("Regenerate IC adapter or Ref2VA unavailable");
    return;
  }
  SafeTensors base, ic;
  base.open(checkpoint); ic.open(adapter);
  CHECK(is_regenerate_lora(ic));
  LoraAdapters loras;
  loras.load({{adapter, 1}}, base);
  CHECK(loras.projection_count() == 312);
  for (const auto& prefix : {std::string("blocks.49."), std::string("token_refiner.blocks.1.")}) {
    CHECK(loras.find(prefix + "attn.to_q")->front().rank == 384);
    CHECK(loras.find(prefix + "attn.to_k")->front().out == 7168);
    CHECK(loras.find(prefix + "mlp.fc1")->front().out == 28672);
  }
}
