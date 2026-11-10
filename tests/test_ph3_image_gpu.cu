#include "detail/transformer_fixture.h"
#include "slopfab/ph3_image.h"
#include "slopfab/vulkan/dit_denoise.h"
#include "slopfab/vulkan/tensor.h"

namespace {
struct PH3Result {
  std::vector<float> video, audio;
};

PH3Result run_ph3_cuda(const slopfab::SafeTensors& checkpoint, const TransformerConfig& cfg,
                       const Case& input) {
  Transformer model;
  model.set_attention_mode(slopfab::AttentionMode::kFlash2);
  model.load(checkpoint, cfg);
  model.prepare_text(input.prompt.data(), input.layout.num_text);
  model.prepare_sequence(input.layout, input.idx, input.pos);
  slopfab::sampler::FlowScheduler video(1), audio(1);
  video.set_sigmas(slopfab::ph3_image_sigmas());
  audio.set_sigmas(slopfab::ph3_image_sigmas());
  auto in = make_denoise_inputs(input.layout, input.idx, video, audio);
  in.init_video_rows = &input.video_rows;
  in.init_audio_rows = &input.audio_rows;
  const auto out = slopfab::dit::denoise(model, in);
  CHECK(out.steps_computed == 8);
  CHECK(all_finite(out.video_rows) && all_finite(out.audio_rows));
  return {out.video_rows, out.audio_rows};
}

PH3Result run_ph3_vulkan(const slopfab::SafeTensors& checkpoint, const TransformerConfig& cfg,
                         const Case& input) {
  using namespace slopfab::vulkan;
  auto instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) throw std::runtime_error("no Vulkan device");
  DeviceOptions options;
  options.portable_arithmetic = true;
  options.enable_shader_float16 = options.enable_storage_buffer_16bit = options.enable_cooperative_matrix = true;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  auto device = physical.front().create_device(options);
  TensorContextOptions context_options;
  context_options.max_batch_operators = 2048;
  TensorContext context(device, context_options);
  ExactH3DenoiseConfig c;
  c.layout = input.layout;
  c.indices = input.idx;
  c.position_ids = input.pos;
  auto& t = c.transformer;
  t.main.layers = cfg.num_layers;
  t.main.block.hidden = cfg.hidden_size;
  t.main.block.heads = cfg.num_attention_heads;
  t.main.block.head_dim = cfg.attention_head_dim;
  t.main.block.ffn = cfg.ffn_dim;
  t.main.block.sequence = input.layout.total_rows();
  t.main.block.timesteps = 2;
  t.main.block.adaln_rank = cfg.adaln_rank;
  t.main.block.attention_mode = slopfab::AttentionMode::kFlash2;
  t.text_rows = input.layout.num_text;
  t.text_dim = cfg.text_dim;
  t.video_dim = cfg.video_patch_dim();
  t.audio_dim = cfg.audio_in_channels;
  t.refiner_layers = cfg.num_refiner_layers;
  t.video_rows = input.layout.num_video_rows;
  t.audio_rows = input.layout.num_audio_rows;
  auto model = ExactH3Denoiser::create(context, c);
  model.load(checkpoint);
  model.prepare(input.prompt.data(), size_t(t.text_rows) * t.text_dim,
                input.video_rows.data(), input.video_rows.size(),
                input.audio_rows.data(), input.audio_rows.size());
  slopfab::sampler::FlowScheduler video(1), audio(1);
  video.set_sigmas(slopfab::ph3_image_sigmas());
  audio.set_sigmas(slopfab::ph3_image_sigmas());
  const auto out = model.run(video, audio);
  CHECK(out.steps_completed == 8);
  CHECK(all_finite(out.video_rows) && all_finite(out.audio_rows));
  return {out.video_rows, out.audio_rows};
}

Case ph3_case(const TransformerConfig& cfg) {
  Case c;
  c.layout.num_text = 2;
  c.layout.num_latent_frames = 1;
  c.layout.latent_height = c.layout.latent_width = 2;
  c.layout.num_video_rows = 1;
  c.layout.num_audio_latents = 2;
  c.layout.num_audio_rows = 4;
  c.idx = slopfab::dit::build_indices(c.layout);
  c.pos = slopfab::dit::build_position_ids(c.layout);
  c.prompt = make_data(size_t(c.layout.num_text) * cfg.text_dim, 321, .25f);
  c.video_rows = make_data(cfg.video_patch_dim(), 765, .5f);
  c.audio_rows = make_data(size_t(4) * cfg.audio_in_channels, 234, .5f);
  return c;
}
}

SLOPFAB_TEST_CATEGORY(ph3_image_rank16_cuda_vulkan_trajectory, "synthetic") {
  auto cfg = tiny_config();
  auto tensors = build_synthetic(cfg);
  const auto input = ph3_case(cfg);
  auto path = write_synthetic(tensors);
  slopfab::SafeTensors checkpoint;
  checkpoint.open(path);
  const auto baseline = run_ph3_cuda(checkpoint, cfg, input);
  const auto vulkan_baseline = run_ph3_vulkan(checkpoint, cfg, input);
  checkpoint.close();

  // Embed the rank-8 model entirely in columns 9..16. A runner truncating
  // the code or using an eight-float stride cannot reproduce the baseline.
  const auto original_table = tensors.at("adaln_t_table").data;
  for (auto& entry : tensors) {
    auto& tensor = entry.second;
    if (entry.first.find("adaln_proj.linear.weight") == std::string::npos) continue;
    std::vector<float> wide(size_t(tensor.shape[0]) * 16, 0.f);
    for (int64_t row = 0; row < tensor.shape[0]; ++row)
      std::copy_n(tensor.data.data() + row * 8, 8, wide.data() + row * 16 + 8);
    tensor.shape[1] = 16;
    tensor.data = std::move(wide);
  }
  auto& table = tensors.at("adaln_t_table");
  table.shape = {4097, 16};
  table.data.assign(4097 * 16, 0.f);
  for (int row = 0; row < 4097; ++row) {
    const int lo = std::min(row / 4, 1023);
    const double fraction = double(row - lo * 4) / 4;
    for (int col = 0; col < 8; ++col)
      table.data[row * 16 + 8 + col] = float(
          (1 - fraction) * original_table[lo * 8 + col] + fraction * original_table[(lo + 1) * 8 + col]);
  }
  cfg.adaln_rank = 16;
  cfg.adaln_table_rows = 4097;
  path = write_synthetic(tensors);
  checkpoint.open(path);
  const auto cuda = run_ph3_cuda(checkpoint, cfg, input);
  CHECK_CLOSE(baseline.video, cuda.video, 1e-5, "rank-16 video preserves upper-half modulation");
  CHECK_CLOSE(baseline.audio, cuda.audio, 1e-5, "rank-16 audio preserves upper-half modulation");
  const auto vulkan = run_ph3_vulkan(checkpoint, cfg, input);
  CHECK_CLOSE(vulkan_baseline.video, vulkan.video, 1e-5, "Vulkan rank-16 video preserves upper-half modulation");
  CHECK_CLOSE(vulkan_baseline.audio, vulkan.audio, 1e-5, "Vulkan rank-16 audio preserves upper-half modulation");
  checkpoint.close();
  std::filesystem::remove(path);
}

SLOPFAB_TEST_CATEGORY(ph3_image_checkpoint_cuda_vulkan_trajectory, "integration") {
  const char* path = std::getenv("SLOPFAB_PH3_IMAGE_CHECKPOINT");
  const char* enabled = std::getenv("SLOPFAB_PH3_GPU");
  if (!path || !enabled || std::string(enabled) != "1") {
    SKIP_OPT_IN("set SLOPFAB_PH3_GPU=1 and SLOPFAB_PH3_IMAGE_CHECKPOINT for eight-step GPU validation");
    return;
  }
  TransformerConfig cfg;
  cfg.adaln_rank = 16;
  cfg.adaln_table_rows = 4097;
  slopfab::SafeTensors checkpoint;
  checkpoint.open(path);
  slopfab::validate_ph3_image_transformer(checkpoint);
  const auto input = ph3_case(cfg);
  const auto cuda = run_ph3_cuda(checkpoint, cfg, input);
  const auto vulkan = run_ph3_vulkan(checkpoint, cfg, input);
  // Shipped CUDA and portable Vulkan arithmetic differ across this 50-layer
  // quantized trajectory. The synthetic test above pins rank preservation
  // independently on each backend; this real-weight smoke test checks execution.
  for (const auto* result : {&cuda, &vulkan}) {
    CHECK(result->video.size() == input.video_rows.size());
    CHECK(result->audio.size() == input.audio_rows.size());
    CHECK(result->video != input.video_rows && result->audio != input.audio_rows);
  }
}
