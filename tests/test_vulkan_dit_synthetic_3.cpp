#include "detail/vulkan_fixture.h"

SLOPFAB_TEST_CATEGORY(vulkan_animate_pinned_audio_boundaries, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  if (!Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("Vulkan unavailable");
    return;
  }
  auto instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) {
    SKIP_UNSUPPORTED_HARDWARE("No Vulkan device");
    return;
  }
  const auto& info = physical.front().info();
  if (!info.cooperative_matrix_bf16_f32_16x16x16) {
    SKIP_UNSUPPORTED_HARDWARE("Exact attention unavailable");
    return;
  }
  DeviceOptions opts;
  opts.enable_timeline_semaphore = opts.enable_shader_int64 = true;
  opts.enable_shader_float16 = opts.enable_storage_buffer_16bit = opts.enable_cooperative_matrix =
      true;
  auto device = physical.front().create_device(opts);
  TensorContextOptions context_options;
  context_options.max_batch_operators = 128;
  TensorContext context(device, context_options);
  std::vector<TensorWrite> tensors;
  auto add = [&](std::string name, std::vector<int64_t> shape, float value = 0,
                 DType dtype = DType::kF32) {
    size_t count = 1;
    for (auto n : shape)
      count *= n;
    tensors.push_back({name, shape, std::vector<float>(count, value), dtype});
  };
  for (const std::string prefix : {"blocks.0.", "token_refiner.blocks.0."}) {
    for (const char* norm :
         {"norm1.weight", "norm2.weight", "attn.q_norm.weight", "attn.k_norm.weight"})
      add(prefix + norm, {128}, 1);
    add(prefix + "attn.qkv_proj.weight", {384, 128});
    add(prefix + "attn.out_proj.weight", {128, 128});
    add(prefix + "mlp.fc1.weight", {256, 128});
    add(prefix + "mlp.fc2.weight", {128, 128});
  }
  add("blocks.0.adaln_proj.linear.weight", {3 * 6 * 128, 8});
  add("blocks.0.adaln_proj.linear.bias", {3 * 6 * 128});
  add("adaln_t_table", {1025, 8});
  add("condition_proj.weight", {128, 16}, 0, DType::kBF16);
  add("condition_proj.bias", {128}, 0, DType::kBF16);
  add("video_patch_proj.weight", {128, 4});
  add("video_patch_proj.bias", {128});
  add("audio_patch_proj.weight", {128, 2});
  add("audio_patch_proj.bias", {128});
  add("token_refiner.final_norm.weight", {128}, 1, DType::kBF16);
  add("final_layer.norm.weight", {128}, 1, DType::kBF16);
  add("final_layer.adaln_proj.linear.weight", {256, 8}, 0, DType::kF16);
  add("final_layer.adaln_proj.linear.bias", {256}, 0, DType::kF16);
  add("final_layer.video_out.weight", {4, 128});
  add("final_layer.video_out.bias", {4}, .25f);
  add("final_layer.audio_out.weight", {2, 128});
  add("final_layer.audio_out.bias", {2}, 10);
  const auto path = std::filesystem::temp_directory_path() / "slopfab_pinned_audio.safetensors";
  write_safetensors(path.string(), tensors);
  SafeTensors checkpoint;
  checkpoint.open(path.string());
  const auto packed = dit::build_ref2va_packed_sequence(
      {1, 1, 1},
      {{dit::ReferenceKind::kVideo, 7, 4, 4, 0}, {dit::ReferenceKind::kImage, 1, 4, 4, 0}}, 1, 4, 4,
      1);
  ExactH3DenoiseConfig config;
  config.layout = packed.layout;
  config.indices = packed.indices;
  config.position_ids = packed.position_ids;
  config.pin_target_audio = true;
  auto& t = config.transformer;
  t.main.layers = 1;
  t.main.block.hidden = 128;
  t.main.block.heads = 1;
  t.main.block.head_dim = 128;
  t.main.block.ffn = 128;
  t.main.block.sequence = packed.layout.total_rows();
  t.main.block.timesteps = 4;
  t.text_rows = 3;
  t.text_dim = 16;
  t.video_dim = 4;
  t.audio_dim = 2;
  t.refiner_layers = 1;
  t.video_rows = static_cast<uint32_t>(packed.indices.video.size());
  t.audio_rows = 2;
  t.video_output_rows = packed.layout.num_video_rows;
  t.audio_output_rows = 2;
  t.video_output_start = packed.layout.video_start();
  t.audio_output_start = packed.layout.audio_start();
  auto model = ExactH3Denoiser::create(context, config);
  model.load(checkpoint);
  const std::vector<float> prompt(3 * 16, .5f), video(t.video_rows * 4, .5f),
      audio{.1f, .2f, .3f, .4f};
  model.prepare(prompt.data(), prompt.size(), video.data(), video.size(), audio.data(),
                audio.size());
  sampler::FlowScheduler vs(3), as(3);
  vs.set_timesteps(4);
  as.set_timesteps(4);
  int boundaries = 0;
  const auto result =
      model.run(vs, as, {}, [&](uint32_t, const std::vector<float>&, const std::vector<float>& a) {
        ++boundaries;
        CHECK(a == audio);
      });
  CHECK(boundaries == 3 && result.steps_completed == 3);
  CHECK(result.audio_rows == audio);
  CHECK(result.video_rows != std::vector<float>(t.video_output_rows * 4, .5f));
  model.unload();
  checkpoint.close();
  std::filesystem::remove(path);
}
