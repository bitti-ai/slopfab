#include "harness.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include "slopfab/cuda/lora.cuh"
#include "slopfab/cuda/gemm.cuh"
#include "slopfab/cuda/deterministic_attention.cuh"
#include "slopfab/dtype.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/vulkan/lora.h"
#include "slopfab/vulkan/dit_denoise.h"
#include "slopfab/dit/transformer.h"

namespace {
using namespace slopfab;
float bf(float x) { return bf16_to_f32(f32_to_bf16(x)); }
TensorLayout layout(std::initializer_list<uint64_t> shape) {
  return TensorLayout::contiguous(shape.begin(), static_cast<uint32_t>(shape.size()));
}

// Preserve the real compressed bytes and metadata tensors while reducing the
// main stack to one block. CUDA's loader correctly refuses unclaimed tensors.
struct ReducedCheckpoint {
  std::filesystem::path path = std::filesystem::temp_directory_path() / "slopfab_lora_reduced.safetensors";
  SafeTensors file;
  explicit ReducedCheckpoint(const SafeTensors& source) {
    std::string header = "{";
    uint64_t offset = 0;
    std::vector<const TensorView*> tensors;
    for (const auto& item : source.tensors()) {
      if (item.first.compare(0, 7, "blocks.") == 0 && item.first.compare(0, 9, "blocks.0.") != 0) continue;
      const auto& v = item.second;
      if (!tensors.empty()) header += ",";
      header += "\"" + item.first + "\":{\"dtype\":\"" + dtype_name(v.dtype) + "\",\"shape\":[";
      for (size_t i = 0; i < v.shape.size(); ++i) {
        if (i) header += ",";
        header += std::to_string(v.shape[i]);
      }
      header += "],\"data_offsets\":[" + std::to_string(offset) + "," + std::to_string(offset + v.nbytes) + "]}";
      offset += v.nbytes;
      tensors.push_back(&v);
    }
    header += "}";
    while (header.size() % 8) header += " ";
    std::ofstream out(path, std::ios::binary);
    const uint64_t n = header.size();
    out.write(reinterpret_cast<const char*>(&n), 8);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const auto* t : tensors)
      out.write(static_cast<const char*>(t->data), static_cast<std::streamsize>(t->nbytes));
    out.close();
    if (!out) throw std::runtime_error("failed to write reduced LoRA test checkpoint");
    file.open(path.string());
  }
  ~ReducedCheckpoint() {
    file.close();
    std::error_code error;
    std::filesystem::remove(path, error);
  }
};

void compare_projection(vulkan::TensorContext& vk, const LoraFactors& f,
                         int rows, int offset, int out) {
  const auto input = test::make_data(static_cast<size_t>(rows) * f.in, 754, .5f);
  std::vector<uint16_t> x(input.size()), initial(static_cast<size_t>(rows) * out);
  for (size_t i = 0; i < x.size(); ++i) x[i] = f32_to_bf16(input[i]);
  for (size_t i = 0; i < initial.size(); ++i) initial[i] = f32_to_bf16(static_cast<float>(i % 13) / 32);
  std::vector<float> expected(initial.size());
  std::vector<float> mid(f.rank);
  for (int row = 0; row < rows; ++row) {
    for (int r = 0; r < f.rank; ++r) {
      float sum = 0;
      for (int i = 0; i < f.in; ++i)
        sum = std::fma(bf16_to_f32(x[static_cast<size_t>(row)*f.in+i]), bf(f.a[r*f.in+i]), sum);
      mid[r] = bf(sum);
    }
    for (int o = 0; o < out; ++o) {
      float sum = 0;
      for (int r = 0; r < f.rank; ++r)
        sum = std::fma(mid[r], bf(f.b[static_cast<size_t>(o+offset)*f.rank+r]), sum);
      const size_t i = static_cast<size_t>(row)*out+o;
      expected[i] = bf(bf16_to_f32(initial[i]) + bf(sum));
    }
  }
  cuda::DeviceBuffer<__nv_bfloat16> cx(x.size()), cy(initial.size());
  SLOPFAB_CUDA_CHECK(cudaMemcpy(cx.get(), x.data(), x.size()*2, cudaMemcpyHostToDevice));
  cuda::LoraRunner runner;
  runner.attach(&f, {f}, offset, out);
  cublasHandle_t blas = nullptr;
  SLOPFAB_CUBLAS_CHECK(cuda::cublas_create(&blas));
  std::vector<uint16_t> exact;
  for (bool deterministic : {false, true}) {
    SLOPFAB_CUDA_CHECK(cudaMemcpy(cy.get(), initial.data(), initial.size()*2, cudaMemcpyHostToDevice));
    runner.apply(&f, cx.get(), rows, cy.get(), blas, nullptr, deterministic);
    std::vector<uint16_t> actual(initial.size());
    SLOPFAB_CUDA_CHECK(cudaMemcpy(actual.data(), cy.get(), actual.size()*2, cudaMemcpyDeviceToHost));
    std::vector<float> wide(actual.size());
    for (size_t i = 0; i < wide.size(); ++i) wide[i] = bf16_to_f32(actual[i]);
    CHECK_CLOSE(expected, wide, deterministic ? 0 : .00390625, "LoRA CUDA vs CPU");
    if (deterministic) exact = actual;
  }
  SLOPFAB_CUBLAS_CHECK(cuda::cublas_destroy(blas));
  vulkan::LoraProjection projection;
  projection.load(vk, {f}, offset, out, rows);
  vulkan::LoraScratch scratch;
  projection.prepare(vk, scratch);
  auto vx = vk.allocate(layout({static_cast<uint64_t>(rows), static_cast<uint64_t>(f.in)}), ScalarType::kBFloat16);
  // Exercise Q/K/V's actual three-dimensional output layout.
  auto vy = vk.allocate(layout({static_cast<uint64_t>(rows), 1, static_cast<uint64_t>(out)}), ScalarType::kBFloat16);
  vk.upload_bytes(vx, x.data(), x.size()*2);
  const auto reserved = scratch.reserved_bytes();
  for (int repeat = 0; repeat < 2; ++repeat) {
    vk.upload_bytes(vy, initial.data(), initial.size()*2);
    auto batch = vk.begin_batch();
    projection.record(batch, vx, vy, rows, scratch);
    batch.submit().wait();
    std::vector<uint16_t> actual(initial.size());
    vk.download_bytes(vy, actual.data(), actual.size()*2);
    CHECK(actual == exact);
    CHECK(scratch.reserved_bytes() == reserved);
  }
}
}

SLOPFAB_TEST(lora_cuda_vulkan_projection_numerics) {
  using namespace slopfab::vulkan;
  if (!Instance::available() || cuda::device_count() == 0) {
    SKIP_UNSUPPORTED_HARDWARE("CUDA and Vulkan required"); return;
  }
  auto instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) { SKIP_UNSUPPORTED_HARDWARE("No Vulkan device"); return; }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  options.enable_shader_float16 = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_cooperative_matrix = true;
  auto device = physical.front().create_device(options);
  TensorContext vk(device);
  for (int rank : {3, 16}) for (int rows : {5, 64, 259}) {
    LoraFactors f;
    f.rank = rank; f.in = 32; f.out = 48;
    f.a = test::make_data(static_cast<size_t>(rank)*f.in, 681, .125f);
    f.b = test::make_data(static_cast<size_t>(rank)*f.out, 923, .125f);
    for (int offset : {0, 16, 32}) compare_projection(vk, f, rows, offset, 16);
  }
  const std::string path = "weights/loras/TaoMate-H3-3step-ComfyUI.safetensors";
  if (std::filesystem::exists(path)) {
    SafeTensors adapter;
    adapter.open(path);
    const std::string name = "diffusion_model.blocks.0.attn.qkv_proj";
    LoraFactors f;
    f.rank = 128; f.in = 5376; f.out = 21504;
    f.a = to_f32(adapter.at(name + ".lora_A.weight"));
    f.b = to_f32(adapter.at(name + ".lora_B.weight"));
    const float scale = to_f32(adapter.at(name + ".alpha"))[0] / 128.0f;
    for (float& value : f.b) value *= scale;
    compare_projection(vk, f, 3, 7168, 7168);
  } else {
    SKIP_MISSING_FIXTURE("Real TaoMate QKV projection unavailable");
  }
}

SLOPFAB_TEST(lora_taomate_real_transformer_three_step_cuda_vulkan) {
  using namespace slopfab::vulkan;
  const std::string adapter = "weights/loras/TaoMate-H3-3step-ComfyUI.safetensors";
  const std::string path = "weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors";
  if (!std::filesystem::exists(path) || !std::filesystem::exists(adapter)) {
    SKIP_MISSING_FIXTURE("TaoMate and NVFP4 checkpoints required"); return;
  }
  if (!Instance::available() || cuda::device_count() == 0) {
    SKIP_UNSUPPORTED_HARDWARE("CUDA and Vulkan required"); return;
  }
  if (cuda::query_device(0).free_memory < (4ull << 30)) {
    SKIP_INSUFFICIENT_VRAM("Reduced real transformer test needs 4 GiB free"); return;
  }
  SafeTensors base;
  base.open(path);
  LoraAdapters loras;
  loras.load({{adapter, 1}}, base);
  ReducedCheckpoint reduced(base);
  dit::SequenceLayout sequence;
  sequence.num_text = 1;
  sequence.num_condition_video = 0;
  sequence.num_condition_audio = 0;
  sequence.num_audio_latents = 0;
  sequence.num_audio_rows = 0;
  sequence.num_latent_frames = 1;
  sequence.latent_height = 2;
  sequence.latent_width = 2;
  sequence.num_video_rows = 1;
  const auto indices = dit::build_indices(sequence);
  const auto positions = dit::build_position_ids(sequence);
  const auto prompt = test::make_data(5120, 284, .125f);
  const auto noise = test::make_data(96, 381, .5f);
  sampler::FlowScheduler video(12), audio(3);
  video.set_timesteps(4, sampler::ScheduleKind::kTaoMate3Step);
  audio.set_timesteps(4, sampler::ScheduleKind::kTaoMate3Step);
  std::vector<float> expected = noise;
  std::vector<std::vector<float>> boundaries;
  const bool exact_attention = cuda::deterministic_h3_attention_available();
  const auto attention = exact_attention ? AttentionMode::kExact : AttentionMode::kFlash2;
  std::printf("  real reduced transformer attention: %s\n", exact_attention ? "exact" : "flash2");
  {
    dit::TransformerConfig config;
    config.num_layers = 1;  // Both real text refiners plus one real main block.
    dit::Transformer model;
    model.load(reduced.file, config);
    model.set_attention_mode(attention);
    model.prepare_text(prompt.data(), 1);
    model.prepare_sequence(sequence, indices, positions);
    std::vector<float> baseline(96);
    auto initial_times = dit::build_row_timesteps(sequence, indices,
        video.timesteps()[0], audio.timesteps()[0]);
    model.forward(noise.data(), nullptr, initial_times, baseline.data(), nullptr);
    model.load(reduced.file, config, &loras);
    model.set_attention_mode(attention);
    model.prepare_text(prompt.data(), 1);
    model.prepare_sequence(sequence, indices, positions);
    std::vector<float> velocity(96);
    for (int step = 0; step < 3; ++step) {
      auto times = dit::build_row_timesteps(sequence, indices,
          video.timesteps()[step], audio.timesteps()[step]);
      model.forward(expected.data(), nullptr, times, velocity.data(), nullptr);
      if (step == 0) CHECK(velocity != baseline);
      video.step(step, expected.data(), velocity.data(), expected.size(), expected.data());
      boundaries.push_back(expected);
    }
  }
  auto instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  CHECK(!physical.empty());
  if (physical.empty()) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  options.enable_shader_float16 = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_cooperative_matrix = true;
  auto device = physical.front().create_device(options);
  TensorContextOptions context_options;
  context_options.max_batch_operators = 256;
  TensorContext vk(device, context_options);
  ExactH3DenoiseConfig config;
  config.layout = sequence;
  config.indices = indices;
  config.position_ids = positions;
  config.transformer.main.layers = 1;
  config.transformer.main.block.sequence = sequence.total_rows();
  config.transformer.main.block.attention_mode = attention;
  config.transformer.main.block.timesteps = 2;
  config.transformer.main.block.loras = &loras;
  config.transformer.text_rows = 1;
  config.transformer.video_rows = 1;
  config.transformer.audio_rows = 0;
  auto without_lora_config = config;
  without_lora_config.transformer.main.block.loras = nullptr;
  std::vector<float> without_lora;
  {
    auto baseline = ExactH3Denoiser::create(vk, without_lora_config);
    baseline.load(reduced.file);
    baseline.prepare(prompt.data(), prompt.size(), noise.data(), noise.size(), nullptr, 0);
    without_lora = baseline.run(video, audio).video_rows;
  }
  auto model = ExactH3Denoiser::create(vk, config);
  model.load(reduced.file);
  model.prepare(prompt.data(), prompt.size(), noise.data(), noise.size(), nullptr, 0);
  auto actual = model.run(video, audio, {}, [&](uint32_t step, const std::vector<float>& v,
                                               const std::vector<float>&) {
    if (exact_attention) {
      CHECK_CLOSE(boundaries.at(step), v, 1e-5, "TaoMate real reduced transformer boundary");
    } else {
      // Flash2 is not the pinned CUDA/Vulkan numerical contract. The adapter
      // arithmetic is compared independently above; report end-to-end drift
      // here without inventing a quality tolerance for different operators.
      const auto stats = slopfab::compare(boundaries.at(step), v);
      std::printf("  Flash2 step %u CUDA/Vulkan relative L2 %.6f, max abs %.6f\n",
                   step + 1, stats.rel_l2, stats.max_abs_err);
    }
  });
  CHECK(actual.steps_completed == 3);
  CHECK(actual.video_rows != without_lora);
  CHECK(std::all_of(actual.video_rows.begin(), actual.video_rows.end(), [](float v) { return std::isfinite(v); }));
  model.prepare(prompt.data(), prompt.size(), noise.data(), noise.size(), nullptr, 0);
  const auto repeated = model.run(video, audio);
  CHECK(actual.video_rows == repeated.video_rows);
}
