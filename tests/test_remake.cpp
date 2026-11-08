#include "harness.h"
#include "slopfab/generate.h"
#include "slopfab/remake.h"
#include "slopfab/safetensors_write.h"
#include <filesystem>
#include <algorithm>
#include <cmath>

using namespace slopfab;
namespace {
template<class F> bool rejects(F f) {
  try { f(); } catch (const std::exception&) { return true; }
  return false;
}
GenerateRequest remake_request() {
  GenerateRequest r;
  r.remake_mode = "swap";
  r.prompt = "subject_definitions: mascot\nsummary: dancing\nretention_analysis: same motion\n"
             "detailed_description: dancing\noverall_soundscape: source\nnon_diegetic_music: source";
  r.num_frames = 124;
  auto video = ReferenceMedia::video(124.0 / 24);
  std::vector<uint8_t> rgb(64 * 96 * 3, 127);
  video.append_frame(rgb.data(), rgb.size(), 64, 96, 64 * 3, 3, 0);
  r.reference_media = {std::make_shared<const ReferenceMedia>(std::move(video))};
  r.reference_image_paths = {"painted1.png", "painted2.png"};
  return r;
}
}

SLOPFAB_TEST(remake_schedules_conditioning_and_validation) {
  auto r = remake_request();
  const auto p = resolve_plan(r);
  CHECK(p.num_inference_steps == 6);
  CHECK(p.video_sigma_shift == 12 && p.audio_sigma_shift == 3);
  const std::vector<float> expected{1, .990476f, .965243f, .923077f, .8f, .631579f, 0};
  CHECK_CLOSE(p.video_sigmas, expected, 1e-6, "upstream Remake swap sigma grid");
  CHECK(p.conditioning.pin_target_audio && p.conditioning.video_first);
  CHECK(!p.conditioning.include_reference_audio && !p.conditioning.require_prompt_embedding);
  CHECK(p.conditioning.fixed_prompt_tokens == 0 && !p.conditioning.allow_caches);
  CHECK(p.canvas_width == 64 && p.canvas_height == 96);
  CHECK(p.layout.num_condition_audio == 0);
  RunOptions options;
  validate_generation_options(r, p, options);
  r.remake_mode = "edit";
  const std::vector<float> edit{1, .983607f, .96f, .923077f, .857143f, .705882f, 0};
  CHECK_CLOSE(resolve_plan(r).video_sigmas, edit, 1e-6, "upstream Remake edit sigma grid");
  r.num_inference_steps = 4;
  CHECK(rejects([&] { resolve_plan(r); }));
  r.num_inference_steps = 0;
  r.sampling.video_sigma_shift = 3;
  CHECK(rejects([&] { resolve_plan(r); }));
  r.sampling = {};
  r.num_frames = 125;
  CHECK(rejects([&] { resolve_plan(r); }));
  r.num_frames = 124;
  r.reference_image_paths.resize(5, "extra.png");
  CHECK(rejects([&] { resolve_plan(r); }));
  r = remake_request();
  r.animate = true;
  CHECK(rejects([&] { resolve_plan(r); }));
  r = remake_request();
  r.conditioning.include_reference_audio = true;
  CHECK(rejects([&] { resolve_plan(r); }));
  r = remake_request();
  r.prompt.clear();
  CHECK(rejects([&] { resolve_plan(r); }));
}

SLOPFAB_TEST(remake_text_labels_order_and_silent_soundtrack) {
  const auto segments = remake_prompt_segments(124, 2, "prompt");
  const std::vector<std::string> expected{"<Video 1>: ", "<0.2 seconds>", "<1.2 seconds>",
      "<2.2 seconds>", "<3.2 seconds>", "<4.2 seconds>", "<5.0 seconds>",
      "<Picture 1>: ", "<Picture 2>: ", "prompt"};
  CHECK(segments == expected);
  auto r = remake_request();
  const auto key = conditioning_cache_key(r);
  r.remake_mode.clear();
  CHECK(conditioning_cache_key(r) != key);
  std::vector<dit::ReferenceGeometry> geometry = {
      {dit::ReferenceKind::kImage, 1, 4, 4, 0},
      {dit::ReferenceKind::kImage, 1, 4, 4, 0},
      {dit::ReferenceKind::kVideo, 7, 4, 4, 0}};
  std::vector<float> rows(4*96, 1);
  rows.insert(rows.end(), 4*96, 2);
  rows.insert(rows.end(), 28*96, 3);
  order_animate_references(geometry, rows);
  CHECK(geometry.front().kind == dit::ReferenceKind::kVideo);
  CHECK(rows.front() == 3 && rows[28*96] == 1 && rows[32*96] == 2);
  const auto silent = prepare_target_audio(ReferenceAudio{{}, 2, 32000, 0}, 124);
  CHECK(silent.size() == 2 * size_t(std::floor(125.0 / 24 * 32000 + .5)));
  CHECK(std::all_of(silent.begin(), silent.end(), [](float v) { return v == 0; }));
}

SLOPFAB_TEST(remake_checkpoint_identity_and_double_adapter_guard) {
  const auto path = std::filesystem::temp_directory_path() / "viggle_remake_swap_pruned_int8_convrot.safetensors";
  struct Cleanup { std::filesystem::path p; ~Cleanup() { std::filesystem::remove(p); } } cleanup{path};
  const std::vector<TensorWrite> tensors = {{"adaln_t_table", {1}, {0}},
                                           {"blocks.0.adaln_proj.linear.weight", {1}, {0}}};
  write_safetensors(path.string(), tensors);
  auto r = remake_request();
  r.transformer_path = path.string();
  CHECK(resolve_plan(r).model.supports_references);
  r.loras = {{"viggle_remake_swap_lora_r64.safetensors", 1}};
  CHECK(rejects([&] { resolve_plan(r); }));
  r.loras.clear();
  r.remake_mode = "edit";
  CHECK(rejects([&] { resolve_plan(r); }));
  write_safetensors(path.string(), tensors, {{"source", "Viggle/Viggle-Remake"}, {"mode", "edit"}});
  CHECK(resolve_plan(r).num_inference_steps == 6);
  write_safetensors(path.string(), tensors, {{"source", "MiniMaxAI/MiniMax-H3"}});
  CHECK(rejects([&] { resolve_plan(r); }));
  r.loras = {{"viggle_remake_edit_lora_r64.safetensors", 1}};
  CHECK(resolve_plan(r).num_inference_steps == 6);
}
