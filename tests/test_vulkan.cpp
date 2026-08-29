#include "harness.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "vidfab/vulkan/runtime.h"
#include "vidfab/vulkan/compute.h"
#include "vidfab/vulkan/gemm.h"
#include "vidfab/vulkan/linear.h"
#include "vidfab/vulkan/tensor.h"
#include "vidfab/vulkan/audio_decoder.h"
#include "vidfab/vulkan/dit_block.h"
#include "vidfab/vulkan/dit_denoise.h"
#include "vidfab/vulkan/dit_graph.h"
#include "vidfab/vulkan/dit_transformer.h"
#include "vidfab/vulkan/text_layer.h"
#include "vidfab/vulkan/text_encoder.h"
#include "vidfab/vulkan/vae_decoder.h"
#include "vidfab/vulkan/yuv_converter.h"
#include "vidfab/attention.h"
#include "vidfab/safetensors_write.h"
#include "vidfab/sampler/scheduler.h"
#include "vidfab/video/y4m.h"
#include "vidfab/dtype.h"
#include "vidfab/safetensors.h"
#include "vidfab/sha256.h"
#include "vidfab/text/layer_capture.h"
#include "../src/vulkan/tensor_validation.h"

namespace {

std::vector<uint32_t> load_spirv(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error(std::string("cannot open SPIR-V: ") + path);
  const std::streamoff length = file.tellg();
  if (length <= 0 || (length % 4) != 0) throw std::runtime_error("invalid SPIR-V byte size");
  file.seekg(0);
  std::vector<uint32_t> words(static_cast<size_t>(length) / 4);
  file.read(reinterpret_cast<char*>(words.data()), length);
  if (!file) throw std::runtime_error("cannot read SPIR-V");
  return words;
}

std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
}

uint32_t float_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float float_from_bits(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint64_t fnv64_floats(const std::vector<float>& values) {
  uint64_t hash = 1469598103934665603ull;
  for (float value : values) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (unsigned byte = 0; byte < 4; ++byte) {
      hash ^= (bits >> (8u * byte)) & 0xffu;
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

uint64_t fnv64_bytes(const void* data, size_t bytes) {
  uint64_t hash = 1469598103934665603ull;
  const auto* cursor = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < bytes; ++i) {
    hash ^= cursor[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

uint16_t reference_bf16(float value) {
  uint32_t bits = float_bits(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return 0x7fffu;
  }
  return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

VIDFAB_TEST(vulkan_qwen_layer0_real_l132_capture_replay) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  constexpr std::array<uint64_t, 11> expected_hashes{
      0x6ca9b8c5e16917b5ull, 0xf9bf6e554a1e84e4ull,
      0x9c2c264a60b4b8b1ull, 0xec21312eea810e06ull,
      0x73901fb1cb7f2cfbull, 0x5ab1cc9e26345fe2ull,
      0x9e072ab6646a2a11ull, 0x74d47f5ed650356dull,
      0x8292d03af91a2025ull, 0x48f99e7549238eceull,
      0xfb3966de636ac098ull};
  constexpr std::array<uint8_t, 32> checkpoint_sha{
      0xbc,0x2c,0xed,0x0f,0xbe,0xa6,0x47,0x57,
      0xfa,0x9a,0xcd,0xdc,0xcf,0xc0,0xb3,0xf4,
      0x81,0x9d,0x1d,0xcf,0x1d,0xa6,0xc1,0x24,
      0xd6,0x90,0xd3,0x68,0xbe,0x28,0x39,0x23};
  constexpr std::array<uint8_t, 32> tokenizer_sha{
      0xa5,0xd8,0x5b,0x6d,0xcc,0x53,0x5e,0x6b,
      0x93,0x11,0x5a,0x9e,0xf2,0x87,0xe6,0x13,
      0x2f,0xdb,0xf3,0x02,0x70,0xda,0x62,0x18,
      0x19,0x4b,0xa7,0x42,0x26,0x11,0x73,0xc7};
  const std::filesystem::path source(VIDFAB_TEST_SOURCE_DIR);
  const std::filesystem::path checkpoint_path = source /
      "weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors";
  const std::filesystem::path capture_path = source /
      "tests/data/qwen_layer0_l132.vfqw";
  const std::filesystem::path tokenizer_path = source /
      "ref/text_encoder/tokenizer.json";
  if (!std::filesystem::exists(checkpoint_path) ||
      !std::filesystem::exists(capture_path) ||
      !std::filesystem::exists(tokenizer_path) || !Instance::available()) return;

#if !defined(VIDFAB_WITH_CUDA) || !VIDFAB_WITH_CUDA
  constexpr Sha256Digest capture_sha{
      0xec,0x13,0xad,0x62,0xa7,0xe2,0x53,0xd5,
      0x88,0xbf,0xac,0x51,0x85,0x0b,0x92,0x48,
      0x7b,0x7c,0xb8,0x8b,0xa7,0x3e,0x78,0x69,
      0xa2,0xb0,0x2c,0xba,0x79,0x11,0x04,0xb3};
  const auto provenance_begin = std::chrono::steady_clock::now();
  CHECK(sha256_file(checkpoint_path.string()) == checkpoint_sha);
  CHECK(sha256_file(tokenizer_path.string()) == tokenizer_sha);
  CHECK(sha256_file(capture_path.string()) == capture_sha);
  const double provenance_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - provenance_begin).count();
  std::printf("  portable Qwen checkpoint/tokenizer/capture SHA-256 %.1f ms\n",
              provenance_ms);
#endif

  const text::QwenLayerCapture capture =
      text::read_qwen_layer_capture(capture_path.string());
  CHECK(capture.header.sequence == 132);
  CHECK(capture.header.hidden == 5120);
  CHECK(capture.header.query_heads == 64);
  CHECK(capture.header.kv_heads == 8);
  CHECK(capture.header.head_dim == 128);
  CHECK(capture.header.intermediate == 25600);
  CHECK(capture.header.checkpoint_sha256 == checkpoint_sha);
  CHECK(capture.header.tokenizer_sha256 == tokenizer_sha);
  CHECK(capture.header.input_fnv64 == 0x617329501f3c87a1ull);
  CHECK(capture.header.rope_fnv64 == 0x693c23a9886dd147ull);
  CHECK(capture.header.boundary_fnv64 == expected_hashes);
  CHECK(fnv64_bytes(capture.input_bf16.data(),
                    capture.input_bf16.size() * sizeof(uint16_t)) ==
        capture.header.input_fnv64);
  uint64_t rope_hash = fnv64_bytes(
      capture.cosine.data(), capture.cosine.size() * sizeof(float));
  const auto* sine_bytes = reinterpret_cast<const uint8_t*>(capture.sine.data());
  for (size_t i = 0; i < capture.sine.size() * sizeof(float); ++i) {
    rope_hash ^= sine_bytes[i];
    rope_hash *= 1099511628211ull;
  }
  CHECK(rope_hash == capture.header.rope_fnv64);

  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) return;
  const DeviceInfo& info = physical.front().info();
  if (!info.timeline_semaphore || !info.shader_int64) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  options.enable_shader_float16 = info.shader_float16;
  options.enable_storage_buffer_16bit = info.storage_buffer_16bit;
  options.enable_cooperative_matrix =
      info.cooperative_matrix_bf16_f32_16x16x16;
  Device device = physical.front().create_device(options);
  TensorContextOptions context_options;
  context_options.max_batch_operators = 46;
  TensorContext context(device, context_options);
  if (!context.exact_causal_gqa_attention() ||
      !context.exact_fp32_vae_normalization() ||
      !context.exact_vae_pointwise()) return;

  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
  QwenTextLayerConfig config;
  config.sequence = capture.header.sequence;
  ExactQwenTextLayerStage stage =
      ExactQwenTextLayerStage::create(context, config);
  stage.load(checkpoint, 0);
  ExactQwenTextLayerScratch scratch =
      ExactQwenTextLayerScratch::create(context, config);
  CHECK(stage.format() == text::WeightFormat::kI8ConvRot);
  CHECK(stage.persistent_bytes() < 500ull * 1024 * 1024);
  CHECK(scratch.dense_cache_bytes() == 250ull * 1024 * 1024);
  CHECK(stage.peak_device_bytes(scratch) ==
        stage.persistent_bytes() + scratch.reserved_bytes());
  CHECK(stage.peak_device_bytes(scratch) < 800ull * 1024 * 1024);

  auto matrix = [](uint64_t rows, uint64_t cols) {
    const uint64_t shape[] = {rows, cols};
    return TensorLayout::contiguous(shape, 2);
  };
  auto heads = [](uint64_t rows, uint64_t count, uint64_t dim) {
    const uint64_t shape[] = {rows, count, dim};
    return TensorLayout::contiguous(shape, 3);
  };
  const uint32_t rows = capture.header.sequence;
  DeviceTensor tokens = context.allocate(
      matrix(rows, 5120), ScalarType::kBFloat16);
  DeviceTensor cosine = context.allocate(matrix(rows, 128));
  DeviceTensor sine = context.allocate(matrix(rows, 128));
  DeviceTensor norm = context.allocate(matrix(rows, 5120), ScalarType::kBFloat16);
  DeviceTensor query = context.allocate(heads(rows, 64, 128), ScalarType::kBFloat16);
  DeviceTensor key = context.allocate(heads(rows, 8, 128), ScalarType::kBFloat16);
  DeviceTensor value = context.allocate(heads(rows, 8, 128), ScalarType::kBFloat16);
  DeviceTensor attention = context.allocate(heads(rows, 64, 128), ScalarType::kBFloat16);
  DeviceTensor attention_residual = context.allocate(
      matrix(rows, 5120), ScalarType::kBFloat16);
  DeviceTensor post_norm = context.allocate(
      matrix(rows, 5120), ScalarType::kBFloat16);
  DeviceTensor gate = context.allocate(
      matrix(rows, 25600), ScalarType::kBFloat16);
  DeviceTensor up = context.allocate(
      matrix(rows, 25600), ScalarType::kBFloat16);
  DeviceTensor activation = context.allocate(
      matrix(rows, 25600), ScalarType::kBFloat16);
  DeviceTensor final = context.allocate(
      matrix(rows, 5120), ScalarType::kBFloat16);
  context.upload(cosine, capture.cosine.data(), capture.cosine.size());
  context.upload(sine, capture.sine.data(), capture.sine.size());
  QwenTextLayerTaps taps{&norm, &query, &key, &value, &attention,
      &attention_residual, &post_norm, &gate, &up, &activation, &final};
  CHECK(stage.required_operators(&taps) == 46);

  std::array<DeviceTensor*, 11> outputs{&norm, &query, &key, &value,
      &attention, &attention_residual, &post_norm, &gate, &up, &activation,
      &final};
  const std::array<size_t, 11> counts{
      size_t(rows) * 5120, size_t(rows) * 64 * 128,
      size_t(rows) * 8 * 128, size_t(rows) * 8 * 128,
      size_t(rows) * 64 * 128, size_t(rows) * 5120,
      size_t(rows) * 5120, size_t(rows) * 25600,
      size_t(rows) * 25600, size_t(rows) * 25600,
      size_t(rows) * 5120};
  auto run = [&] {
    context.upload_bytes(tokens, capture.input_bf16.data(),
                         capture.input_bf16.size() * sizeof(uint16_t));
    TensorBatch batch = context.begin_batch();
    stage.record(batch, tokens, cosine, sine, scratch, &taps);
    CHECK(batch.remaining_operator_capacity() == 0);
    batch.submit().wait();
    std::array<uint64_t, 11> hashes{};
    for (size_t i = 0; i < outputs.size(); ++i) {
      std::vector<uint16_t> values(counts[i]);
      context.download_bytes(*outputs[i], values.data(),
                             values.size() * sizeof(uint16_t));
      hashes[i] = fnv64_bytes(values.data(), values.size() * sizeof(uint16_t));
    }
    return hashes;
  };
  const auto first = run();
  CHECK(first == expected_hashes);
  const uint64_t stable_descriptors = context.descriptor_set_allocations();
  const auto repeat = run();
  CHECK(repeat == first);
  CHECK(context.descriptor_set_allocations() == stable_descriptors);
}

#if !defined(VIDFAB_WITH_CUDA) || !VIDFAB_WITH_CUDA
VIDFAB_TEST(vulkan_qwen_full50_real_l132_replay) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  const std::filesystem::path source(VIDFAB_TEST_SOURCE_DIR);
  const std::filesystem::path checkpoint_path = source /
      "weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors";
  const std::filesystem::path capture_path = source /
      "tests/data/qwen_layer0_l132.vfqw";
  if (!std::filesystem::exists(checkpoint_path) ||
      !std::filesystem::exists(capture_path) || !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) return;
  const DeviceInfo& info = physical.front().info();
  if (!info.timeline_semaphore || !info.shader_int64 ||
      !info.shader_float16 || !info.storage_buffer_16bit ||
      !info.cooperative_matrix_bf16_f32_16x16x16) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  options.enable_shader_float16 = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(options);
  TensorContextOptions context_options;
  context_options.max_batch_operators = 37;
  TensorContext context(device, context_options);
  const uint64_t cold_baseline = context.pooled_used_bytes();

  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
  const text::QwenLayerCapture capture =
      text::read_qwen_layer_capture(capture_path.string());
  ExactQwenTextEncoder encoder = ExactQwenTextEncoder::create(context);
  encoder.load(checkpoint);
  text::EncoderTrace trace;
  const auto begin = std::chrono::steady_clock::now();
  const text::PromptEmbedding output = encoder.encode(capture.token_ids, &trace);
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - begin).count();
  constexpr std::array<uint64_t, 50> expected{
      0xfb3966de636ac098ull,0x16b0e53a0265959dull,
      0x7e2377403c49f851ull,0xdacca747d7db7ccaull,
      0xb3de1cdb9248573cull,0xcb7ef7852ef4ce79ull,
      0xfe09924dff391f53ull,0x0ac372b850d058d8ull,
      0xe818f084c10d99d5ull,0xeed04e74ca44abcfull,
      0xe3e63048102b3e14ull,0x6e5647916a5d3586ull,
      0x1b9a96cd7cb703c2ull,0xe60df16cc8c0390dull,
      0xcf624f216f397d84ull,0x2e680a56b7030b0aull,
      0x8b8e97be1f24e88eull,0x61e2e50339318a1eull,
      0x410a6b1ab0a9c0e4ull,0xaf70d3523ce3a525ull,
      0xfb6b26f66fc1eb31ull,0x732c1c3ed4d7e6a6ull,
      0x1b81b2dd36a55f5aull,0x01f09dd599e38e09ull,
      0xc93c6edb4d56f691ull,0xcb9bd628a2587064ull,
      0x30c8bbdab894f8cfull,0xd0c82e71b41c4e74ull,
      0xef88fb29b4c38602ull,0x3b865ed94b23284eull,
      0x160d4d0750485e84ull,0xa3128750a0466a21ull,
      0xc1c6ba2884daa0e1ull,0xbf2a42d54c6d336eull,
      0x7ae8060855d02ab2ull,0x2fe95298685c12c3ull,
      0x17610f06aabb0ce5ull,0xd09a61dcf57388c9ull,
      0xc363bc14f4fabe3bull,0xc857cd797d07a823ull,
      0xb5ba41c7ec13df0bull,0x153186eecfcb34e2ull,
      0x89792cd842ae3215ull,0x03cbb8ab112884f9ull,
      0x0d9c0669be4b1818ull,0xc45536c76bfb5268ull,
      0x465a47fdc0f38a3bull,0xc6c70427de251f9dull,
      0x2082d9a03b0f2c88ull,0x141e4954a3b02693ull};
  std::array<uint64_t, 50> actual{};
  const size_t layer_elements = size_t(132) * 5120;
  CHECK(trace.layer_residual_bf16.size() == 50 * layer_elements);
  for (size_t layer = 0; layer < actual.size(); ++layer)
    actual[layer] = fnv64_bytes(
        trace.layer_residual_bf16.data() + layer * layer_elements,
        layer_elements * sizeof(uint16_t));
  CHECK(actual == expected);
  CHECK(fnv64_floats(output.data) == 0x579170f52abfc8dbull);
  const ExactQwenTextEncoderStats stats = encoder.stats();
  CHECK(stats.peak_device_bytes < 900ull * 1024 * 1024);
  CHECK(stats.descriptor_set_allocations == 36);
  encoder.unload();
  context.collect();
  CHECK(context.pooled_used_bytes() >= cold_baseline);
  CHECK(context.pooled_used_bytes() <=
        cold_baseline + 2 * context.staging_capacity_bytes());
  std::printf(
      "  CUDA-off real exact Qwen full50 L132 %.2f s final %016llx peak/reserved %.1f/%.1f MiB descriptors %llu\n",
      seconds,
      static_cast<unsigned long long>(fnv64_floats(output.data)),
      double(stats.peak_device_bytes) / 1048576.0,
      double(stats.allocator_reserved_bytes) / 1048576.0,
      static_cast<unsigned long long>(stats.descriptor_set_allocations));
}
#endif

VIDFAB_TEST(vulkan_exact_dit_euler_matches_host_scheduler) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions device_options;
  device_options.enable_timeline_semaphore = true;
  device_options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(device_options);
  TensorContextOptions context_options;
  context_options.max_batch_operators = 4;
  TensorContext context(device, context_options);
  if (!context.exact_vae_pointwise()) return;

  constexpr uint64_t count = 257;
  const TensorLayout layout = TensorLayout::contiguous(&count, 1);
  DeviceTensor sample = context.allocate(layout);
  DeviceTensor velocity = context.allocate(layout);
  std::vector<float> host_sample(count), host_velocity(count), got(count);
  for (uint64_t i = 0; i < count; ++i) {
    host_sample[i] = float(int((i * 37) % 257) - 128) / 64.0f;
    host_velocity[i] = float(int((i * 53) % 193) - 96) / 128.0f;
  }
  context.upload(sample, host_sample.data(), count);
  context.upload(velocity, host_velocity.data(), count);

  sampler::FlowScheduler schedule(12.0f);
  schedule.set_timesteps(6);
  for (size_t step = 0; step < schedule.num_steps(); ++step) {
    schedule.step(static_cast<int>(step), host_sample.data(),
                  host_velocity.data(), count, host_sample.data());
    const float sigma_from_timestep = 1.0f - schedule.timesteps()[step];
    const float ratio = schedule.sigmas()[step + 1] / schedule.sigmas()[step];
    TensorBatch batch = context.begin_batch();
    batch.dit_euler_step_f32(sample, velocity, sigma_from_timestep, ratio);
    CHECK(batch.remaining_operator_capacity() == 3u);
    batch.submit().wait();
    context.download(sample, got.data(), count);
    CHECK(std::memcmp(host_sample.data(), got.data(), count * sizeof(float)) == 0);
  }

  // Validation is transactional and the same batch remains usable.
  TensorBatch transactional = context.begin_batch();
  bool alias_rejected = false;
  try { transactional.dit_euler_step_f32(sample, sample, 0.5f, 0.5f); }
  catch (const std::invalid_argument&) { alias_rejected = true; }
  CHECK(alias_rejected && transactional.remaining_operator_capacity() == 4u);
  transactional.dit_euler_step_f32(sample, velocity, 0.5f, 0.0f);
  transactional.submit().wait();

  // Total-domain parity, including a one-element dispatch and a 64-thread
  // tail. Exceptional source values are canonicalized on device, never read
  // back for graph-owned validation.
  const std::vector<float> exceptional_samples{
      0.0f, -0.0f, float_from_bits(0x00000001u),
      float_from_bits(0x80000001u), float_from_bits(0x7fc12345u),
      float_from_bits(0xffdabcdeu), float_from_bits(0x7f800000u),
      float_from_bits(0xff800000u), std::numeric_limits<float>::max(),
      -std::numeric_limits<float>::max(), 1.0f, -1.0f};
  auto run_exceptional = [&](uint64_t elements, float sigma, float ratio) {
    const TensorLayout test_layout = TensorLayout::contiguous(&elements, 1);
    DeviceTensor test_sample = context.allocate(test_layout);
    DeviceTensor test_velocity = context.allocate(test_layout);
    std::vector<float> x(elements), v(elements), expected(elements), actual(elements);
    for (uint64_t i = 0; i < elements; ++i) {
      x[i] = exceptional_samples[i % exceptional_samples.size()];
      v[i] = exceptional_samples[(i * 5u + 1u) % exceptional_samples.size()];
      expected[i] = sampler::exact_euler_value(x[i], v[i], sigma, ratio);
    }
    context.upload(test_sample, x.data(), elements);
    context.upload(test_velocity, v.data(), elements);
    TensorBatch batch = context.begin_batch();
    batch.dit_euler_step_f32(test_sample, test_velocity, sigma, ratio);
    batch.submit().wait();
    context.download(test_sample, actual.data(), elements);
    CHECK(std::memcmp(expected.data(), actual.data(), elements * sizeof(float)) == 0);
  };
  run_exceptional(1u, 1.0f, 1.0f);
  for (const auto controls : {std::array<float, 2>{0.0f, 0.0f},
                              std::array<float, 2>{1.0f, 0.0f},
                              std::array<float, 2>{0.0f, 1.0f},
                              std::array<float, 2>{1.0f, 1.0f},
                              std::array<float, 2>{float_from_bits(1u),
                                                   float_from_bits(1u)},
                              std::array<float, 2>{0.5f, 0.5f}})
    run_exceptional(65u, controls[0], controls[1]);
}

VIDFAB_TEST(vulkan_exact_blocked_attention_single_key) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore ||
      !physical.front().info().shader_int64) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  Device device = physical.front().create_device(options);
  TensorContext context(device);
  if (!context.exact_normalization()) return;

  constexpr uint32_t sequence = 1, heads = 2, dim = 64;
  const uint64_t extent[] = {sequence, heads, dim};
  const TensorLayout layout = TensorLayout::contiguous(extent, 3);
  DeviceTensor q = context.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor k = context.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor v = context.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor out = context.allocate(layout, ScalarType::kBFloat16);
  std::vector<uint16_t> zeros(heads * dim, 0);
  std::vector<uint16_t> values(heads * dim);
  for (size_t i = 0; i < values.size(); ++i) {
    const float value = static_cast<float>(static_cast<int>(i % 17) - 8) / 8.0f;
    values[i] = reference_bf16(value);
  }
  const uint16_t conversion_edges[] = {
      0x0000u, 0x8000u, 0x0001u, 0x007fu, 0x0080u, 0x387fu,
      0x3880u, 0x7f7fu, 0x7f80u, 0xff80u, 0x3f80u, 0x3f81u};
  std::copy(std::begin(conversion_edges), std::end(conversion_edges),
            values.begin());
  std::vector<uint16_t> expected(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    expected[i] = f32_to_bf16(f16_to_f32(f32_to_f16(bf16_to_f32(values[i]))));
  }
  // The fixed PV FMA starts from +0, so (-0 * 1) + +0 is +0.
  expected[1] = 0;
  context.upload_bytes(q, zeros.data(), zeros.size() * sizeof(uint16_t));
  context.upload_bytes(k, zeros.data(), zeros.size() * sizeof(uint16_t));
  context.upload_bytes(v, values.data(), values.size() * sizeof(uint16_t));
  BlockedAttentionPlanDesc desc;
  desc.sequence = sequence;
  desc.heads = heads;
  desc.head_dim = dim;
  desc.scale = 0.125f;
  BlockedAttentionPlan plan = BlockedAttentionPlan::create(context, desc);
  PreparedAttentionInputs prepared = PreparedAttentionInputs::create(context, desc);
  TensorBatch batch = context.begin_batch();
  PreparedAttentionView inputs = prepared.prepare(batch, q, k, v);
  plan.record(batch, inputs, out);
  batch.submit().wait();
  std::vector<uint16_t> actual(values.size());
  context.download_bytes(out, actual.data(), actual.size() * sizeof(uint16_t));
  size_t edge_mismatch = expected.size();
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != actual[i]) { edge_mismatch = i; break; }
  }
  CHECK_MSG(edge_mismatch == expected.size(),
            "attention conversion edge %zu: %04x != %04x",
            edge_mismatch,
            edge_mismatch == expected.size() ? 0u : expected[edge_mismatch],
            edge_mismatch == expected.size() ? 0u : actual[edge_mismatch]);

  CHECK(prepared.reserved_bytes() == values.size() * sizeof(uint16_t) * 3);
  DeviceTensor out2 = context.allocate(layout, ScalarType::kBFloat16);
  PreparedAttentionInputs prepared2 = PreparedAttentionInputs::create(context, desc);
  TensorBatch first = context.begin_batch();
  PreparedAttentionView first_view = prepared.prepare(first, q, k, v);
  plan.record(first, first_view, out);
  Submission first_token = first.submit();
  TensorBatch second = context.begin_batch();
  PreparedAttentionView second_view = prepared2.prepare(second, q, k, v);
  plan.record(second, second_view, out2);
  Submission second_token = second.submit();
  TensorBatch third = context.begin_batch();
  PreparedAttentionView third_view = prepared.prepare(third, q, k, v);
  plan.record(third, third_view, out);
  Submission third_token = third.submit();
  CHECK(first_token.value() < second_token.value());
  CHECK(second_token.value() < third_token.value());
  first_token.wait(); second_token.wait(); third_token.wait();
  const uint64_t stable_reserved = context.reserved_bytes();
  const uint64_t stable_descriptors = context.descriptor_set_allocations();
  for (int repeat = 0; repeat < 3; ++repeat) {
    TensorBatch stable = context.begin_batch();
    PreparedAttentionView stable_view = prepared.prepare(stable, q, k, v);
    plan.record(stable, stable_view, out);
    stable.submit().wait();
    CHECK(context.reserved_bytes() == stable_reserved);
    CHECK(context.descriptor_set_allocations() == stable_descriptors);
  }

  TensorBatch after_submit = context.begin_batch();
  bool stale_rejected = false;
  try { plan.record(after_submit, third_view, out); }
  catch (const std::invalid_argument&) { stale_rejected = true; }
  CHECK(stale_rejected);
  PreparedAttentionView fresh = prepared.prepare(after_submit, q, k, v);
  plan.record(after_submit, fresh, out);
  after_submit.submit().wait();

  {
    TensorBatch superseded = context.begin_batch();
    PreparedAttentionView old = prepared.prepare(superseded, q, k, v);
    PreparedAttentionView newest = prepared.prepare(superseded, q, k, v);
    bool old_rejected = false;
    try { plan.record(superseded, old, out); }
    catch (const std::invalid_argument&) { old_rejected = true; }
    CHECK(old_rejected);
    plan.record(superseded, newest, out);
    superseded.submit().wait();
  }
  PreparedAttentionView discarded;
  {
    TensorBatch abandoned = context.begin_batch();
    discarded = prepared.prepare(abandoned, q, k, v);
  }
  TensorBatch recovery = context.begin_batch();
  bool discarded_rejected = false;
  try { plan.record(recovery, discarded, out); }
  catch (const std::invalid_argument&) { discarded_rejected = true; }
  CHECK(discarded_rejected);
  PreparedAttentionView recovery_view = prepared.prepare(recovery, q, k, v);
  bool alias_rejected = false;
  try { plan.record(recovery, recovery_view, q); }
  catch (const std::invalid_argument&) { alias_rejected = true; }
  CHECK(alias_rejected);
  plan.record(recovery, recovery_view, out);
  recovery.submit().wait();

  bool thirty_third_rejected = false;
  {
    TensorBatch bounded = context.begin_batch();
    PreparedAttentionView bounded_view = prepared.prepare(bounded, q, k, v);
    for (int i = 0; i < 31; ++i) plan.record(bounded, bounded_view, out);
    try { plan.record(bounded, bounded_view, out); }
    catch (const std::logic_error&) { thirty_third_rejected = true; }
  }
  CHECK(thirty_third_rejected);

  // Qwen image grids vary between requests. Recreating shape-specific plans
  // and slots must reuse the pool and the bounded descriptor arenas rather
  // than accumulating one allocation set per observed grid.
  auto run_shape = [&](uint32_t variable_sequence) {
    const uint64_t variable_extent[] = {variable_sequence, 1, dim};
    const TensorLayout variable_layout =
        TensorLayout::contiguous(variable_extent, 3);
    const size_t variable_count = size_t(variable_sequence) * dim;
    std::vector<uint16_t> data(variable_count, 0);
    DeviceTensor variable_q = context.allocate(variable_layout, ScalarType::kBFloat16);
    DeviceTensor variable_k = context.allocate(variable_layout, ScalarType::kBFloat16);
    DeviceTensor variable_v = context.allocate(variable_layout, ScalarType::kBFloat16);
    DeviceTensor variable_out = context.allocate(variable_layout, ScalarType::kBFloat16);
    context.upload_bytes(variable_q, data.data(), data.size() * 2);
    context.upload_bytes(variable_k, data.data(), data.size() * 2);
    context.upload_bytes(variable_v, data.data(), data.size() * 2);
    BlockedAttentionPlanDesc variable_desc{
        variable_sequence, 1, dim, exact_attention_scale(dim)};
    BlockedAttentionPlan variable_plan =
        BlockedAttentionPlan::create(context, variable_desc);
    PreparedAttentionInputs variable_prepared =
        PreparedAttentionInputs::create(context, variable_desc);
    TensorBatch variable_batch = context.begin_batch();
    PreparedAttentionView variable_inputs = variable_prepared.prepare(
        variable_batch, variable_q, variable_k, variable_v);
    variable_plan.record(variable_batch, variable_inputs, variable_out);
    variable_batch.submit().wait();
  };
  for (uint32_t variable_sequence : {3u, 17u, 5u, 33u}) run_shape(variable_sequence);
  context.upload_bytes(q, zeros.data(), zeros.size() * sizeof(uint16_t));
  const uint64_t varied_reserved = context.reserved_bytes();
  const uint64_t varied_descriptors = context.descriptor_set_allocations();
  for (uint32_t variable_sequence : {33u, 5u, 17u, 3u}) run_shape(variable_sequence);
  context.upload_bytes(q, zeros.data(), zeros.size() * sizeof(uint16_t));
  CHECK(context.reserved_bytes() == varied_reserved);
  CHECK(context.descriptor_set_allocations() == varied_descriptors);

  const uint64_t attention_live_baseline = context.pooled_used_bytes();
  Submission dropped_wrappers;
  {
    constexpr uint32_t drop_sequence = 3;
    const uint64_t drop_extent[] = {drop_sequence, 1, dim};
    const TensorLayout drop_layout = TensorLayout::contiguous(drop_extent, 3);
    DeviceTensor drop_q = context.allocate(drop_layout, ScalarType::kBFloat16);
    DeviceTensor drop_k = context.allocate(drop_layout, ScalarType::kBFloat16);
    DeviceTensor drop_v = context.allocate(drop_layout, ScalarType::kBFloat16);
    DeviceTensor drop_out = context.allocate(drop_layout, ScalarType::kBFloat16);
    BlockedAttentionPlanDesc drop_desc{
        drop_sequence, 1, dim, exact_attention_scale(dim)};
    PreparedAttentionInputs drop_prepared =
        PreparedAttentionInputs::create(context, drop_desc);
    BlockedAttentionPlan drop_plan = BlockedAttentionPlan::create(context, drop_desc);
    TensorBatch drop_batch = context.begin_batch();
    PreparedAttentionView drop_inputs = drop_prepared.prepare(
        drop_batch, drop_q, drop_k, drop_v);
    drop_plan.record(drop_batch, drop_inputs, drop_out);
    dropped_wrappers = drop_batch.submit();
  }
  CHECK(context.pooled_used_bytes() > attention_live_baseline);
  dropped_wrappers.wait();
  context.upload_bytes(q, zeros.data(), zeros.size() * sizeof(uint16_t));
  CHECK(context.pooled_used_bytes() == attention_live_baseline);
}

VIDFAB_TEST(vulkan_exact_h3_attention_single_key) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) return;
  const DeviceInfo& info = physical.front().info();
  DeviceOptions disabled_options;
  disabled_options.enable_timeline_semaphore = info.timeline_semaphore;
  disabled_options.enable_shader_int64 = info.shader_int64;
  disabled_options.enable_shader_float16 = info.shader_float16;
  disabled_options.enable_storage_buffer_16bit = info.storage_buffer_16bit;
  Device disabled_device = physical.front().create_device(disabled_options);
  TensorContext disabled(disabled_device);
  CHECK(!disabled.exact_h3_attention());
  bool unavailable_rejected = false;
  try {
    (void)H3AttentionPlan::create(
        disabled, {1, 1, 64, exact_attention_scale(64)});
  } catch (const std::runtime_error&) { unavailable_rejected = true; }
  CHECK(unavailable_rejected);
  if (!info.timeline_semaphore || !info.shader_int64 || !info.shader_float16 ||
      !info.storage_buffer_16bit || !info.cooperative_matrix) return;
  DeviceOptions options = disabled_options;
  options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(options);
  TensorContext context(device);
  if (!context.exact_h3_attention()) return;
  constexpr uint32_t dim = 64;
  const uint64_t shape[] = {1, 1, dim};
  const TensorLayout layout = TensorLayout::contiguous(shape, 3);
  DeviceTensor q = context.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor k = context.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor v = context.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor out = context.allocate(layout, ScalarType::kBFloat16);
  std::vector<uint16_t> zeros(dim, 0), values(dim);
  for (uint32_t i = 0; i < dim; ++i)
    values[i] = reference_bf16(float(int(i % 15) - 7) / 8.0f);
  context.upload_bytes(q, zeros.data(), zeros.size() * 2);
  context.upload_bytes(k, zeros.data(), zeros.size() * 2);
  context.upload_bytes(v, values.data(), values.size() * 2);
  H3AttentionPlan plan = H3AttentionPlan::create(
      context, {1, 1, dim, exact_attention_scale(dim)});
  TensorBatch batch = context.begin_batch();
  plan.record(batch, q, k, v, out);
  batch.submit().wait();
  std::vector<uint16_t> actual(dim);
  context.download_bytes(out, actual.data(), actual.size() * 2);
  CHECK(actual == values);
}

VIDFAB_TEST(vulkan_attention_prepare_exhaustive_bf16) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  Device device = physical.front().create_device(options);
  ComputeContext context(device, {2, 6, 1});
  ComputePipelineOptions pipeline_options;
  pipeline_options.storage_binding_count = 6;
  pipeline_options.push_constant_bytes = sizeof(uint32_t);
  pipeline_options.local_size[0] = 64;
  ComputePipeline pipeline = ComputePipeline::create(
      device, load_spirv(VIDFAB_TEST_ATTENTION_PREPARE_SPV_PATH),
      pipeline_options);

  constexpr uint32_t patterns = 1u << 16;
  constexpr uint64_t bytes = uint64_t{patterns} * sizeof(uint16_t);
  constexpr uint32_t words = patterns / 2;
  BufferPool pool(device, 1024 * 1024);
  Buffer upload = pool.allocate(bytes, BufferUsage::kTransferSource,
                                MemoryUsage::kUpload);
  Buffer source = pool.allocate(bytes, BufferUsage::kTransferDestination |
                                         BufferUsage::kStorage,
                                MemoryUsage::kDevice);
  Buffer output0 = pool.allocate(bytes, BufferUsage::kStorage |
                                          BufferUsage::kTransferSource,
                                 MemoryUsage::kDevice);
  Buffer output1 = pool.allocate(bytes, BufferUsage::kStorage,
                                 MemoryUsage::kDevice);
  Buffer output2 = pool.allocate(bytes, BufferUsage::kStorage,
                                 MemoryUsage::kDevice);
  Buffer readback = pool.allocate(bytes, BufferUsage::kTransferDestination,
                                  MemoryUsage::kReadback);
  std::vector<uint16_t> input(patterns), actual(patterns);
  for (uint32_t i = 0; i < patterns; ++i) input[i] = static_cast<uint16_t>(i);
  upload.write(0, input.data(), bytes);
  CommandList commands = context.begin();
  commands.barrier(upload, BufferAccess::kHostWrite, BufferAccess::kTransferRead);
  commands.copy_buffer(upload, source, bytes);
  commands.barrier(source, BufferAccess::kTransferWrite, BufferAccess::kComputeRead);
  commands.bind_compute(pipeline, {{0, &source, 0, bytes},
                                   {1, &source, 0, bytes},
                                   {2, &source, 0, bytes},
                                   {3, &output0, 0, bytes},
                                   {4, &output1, 0, bytes},
                                   {5, &output2, 0, bytes}});
  commands.push_constants(&words, sizeof(words));
  commands.dispatch((words + 63) / 64);
  commands.barrier(output0, BufferAccess::kComputeWrite,
                   BufferAccess::kTransferRead);
  commands.copy_buffer(output0, readback, bytes);
  commands.barrier(readback, BufferAccess::kTransferWrite,
                   BufferAccess::kHostRead);
  context.submit(std::move(commands)).wait();
  readback.read(0, actual.data(), bytes);
  for (uint32_t i = 0; i < patterns; ++i) {
    const uint16_t expected = (input[i] & 0x7fffu) > 0x7f80u
        ? 0x7fffu : f32_to_f16(bf16_to_f32(input[i]));
    CHECK(actual[i] == expected);
  }
}

VIDFAB_TEST(vulkan_linear_weight_cpu_reference) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext context(device);
  auto run = [&](const char* label, const LinearWeightUpload& upload,
                 const std::vector<uint16_t>& expected) {
    LinearWeight weight = LinearWeight::upload(context, upload);
    const uint64_t shape[] = {upload.out_features, upload.in_features};
    DeviceTensor output = context.allocate(TensorLayout::contiguous(shape, 2),
                                           ScalarType::kBFloat16);
    TensorBatch batch = context.begin_batch();
    weight.materialize_bf16(batch, output);
    batch.submit().wait();
    std::vector<uint16_t> actual(expected.size());
    context.download_bytes(output, actual.data(), actual.size() * 2);
    size_t mismatch = expected.size();
    for (size_t i = 0; i < expected.size(); ++i)
      if (actual[i] != expected[i]) { mismatch = i; break; }
    CHECK_MSG(mismatch == expected.size(), "%s mismatch at %zu: %04x != %04x",
              label, mismatch, mismatch == expected.size() ? 0u : expected[mismatch],
              mismatch == expected.size() ? 0u : actual[mismatch]);
  };

  const std::vector<float> dense_f32 = {-3.5f, -0.0f, 0.125f, 1.0f, 17.25f};
  std::vector<uint16_t> dense_expected(dense_f32.size());
  for (size_t i = 0; i < dense_f32.size(); ++i)
    dense_expected[i] = reference_bf16(dense_f32[i]);
  LinearWeightUpload dense;
  dense.format = LinearWeightFormat::kFloat32;
  dense.out_features = 1;
  dense.in_features = static_cast<uint32_t>(dense_f32.size());
  dense.data = dense_f32.data();
  dense.data_bytes = dense_f32.size() * 4;
  run("f32", dense, dense_expected);
  const std::vector<uint16_t> raw_bf16 = {0x0000, 0x8000, 0x0001, 0x7f80, 0x7fc1};
  dense.format = LinearWeightFormat::kBFloat16;
  dense.data = raw_bf16.data();
  dense.data_bytes = raw_bf16.size() * 2;
  run("bf16", dense, raw_bf16);

  std::vector<uint8_t> f8(256);
  std::vector<uint16_t> f8_expected(256);
  const float f8_scale = 0.75f;
  for (uint32_t i = 0; i < 256; ++i) {
    f8[i] = static_cast<uint8_t>(i);
    f8_expected[i] = reference_bf16(f8_e4m3_to_f32(f8[i]) * f8_scale);
  }
  LinearWeightUpload f8_upload;
  f8_upload.format = LinearWeightFormat::kFloat8E4M3;
  f8_upload.out_features = 1;
  f8_upload.in_features = 256;
  f8_upload.data = f8.data();
  f8_upload.data_bytes = f8.size();
  f8_upload.weight_scale = &f8_scale;
  f8_upload.weight_scale_count = 1;
  run("f8", f8_upload, f8_expected);

  constexpr uint32_t i8_out = 2, i8_in = 67;
  std::vector<int8_t> i8(i8_out * i8_in);
  const float i8_scales[] = {-0.25f, 1.5f};
  std::vector<uint16_t> i8_expected(i8.size());
  for (size_t i = 0; i < i8.size(); ++i) {
    i8[i] = static_cast<int8_t>(i * 71u);
    i8_expected[i] = reference_bf16(static_cast<float>(i8[i]) *
                                  i8_scales[i / i8_in]);
  }
  LinearWeightUpload i8_upload;
  i8_upload.format = LinearWeightFormat::kInt8;
  i8_upload.out_features = i8_out;
  i8_upload.in_features = i8_in;
  i8_upload.data = i8.data();
  i8_upload.data_bytes = i8.size();
  i8_upload.weight_scale = i8_scales;
  i8_upload.weight_scale_count = i8_out;
  run("i8", i8_upload, i8_expected);

  constexpr uint32_t nv_out = 128, nv_in = 64;
  const size_t nv_count = static_cast<size_t>(nv_out) * nv_in;
  std::vector<uint8_t> nv_codes(nv_count / 2), nv_scales(nv_count / 16, 0x38);
  std::vector<uint16_t> nv_expected(nv_count);
  for (size_t byte = 0; byte < nv_codes.size(); ++byte)
    nv_codes[byte] = static_cast<uint8_t>(((2 * byte & 15) << 4) |
                                          ((2 * byte + 1) & 15));
  const float nv_global = 0.25f;
  auto scale_slot = [&](uint32_t row, uint32_t block) {
    const uint32_t blocks_per_row = nv_in / 16;
    const uint32_t tile = (row >> 7) * (blocks_per_row >> 2) + (block >> 2);
    return static_cast<size_t>(tile) * 512 + (row & 31) * 16 +
           ((row & 127) >> 5) * 4 + (block & 3);
  };
  for (size_t i = 0; i < nv_count; ++i) {
    const uint8_t packed = nv_codes[i / 2];
    const uint8_t code = (i & 1) == 0 ? packed >> 4 : packed & 15;
    const uint32_t row = static_cast<uint32_t>(i / nv_in);
    const float scale_value = f8_e4m3_to_f32(
        nv_scales[scale_slot(row, static_cast<uint32_t>((i % nv_in) / 16))]) *
        nv_global;
    nv_expected[i] = reference_bf16(f4_e2m1_to_f32(code) * scale_value);
  }
  LinearWeightUpload nv_upload;
  nv_upload.format = LinearWeightFormat::kNVFloat4;
  nv_upload.out_features = nv_out;
  nv_upload.in_features = nv_in;
  nv_upload.data = nv_codes.data();
  nv_upload.data_bytes = nv_codes.size();
  nv_upload.block_scale = nv_scales.data();
  nv_upload.block_scale_count = nv_scales.size();
  nv_upload.global_scale = nv_global;
  run("nvfp4", nv_upload, nv_expected);

  constexpr uint32_t nf_out = 3, nf_in = 45;
  const size_t nf_count = static_cast<size_t>(nf_out) * nf_in;
  std::vector<uint8_t> nf_codes((nf_count + 1) / 2), nf_absmax(3);
  for (size_t i = 0; i < nf_codes.size(); ++i)
    nf_codes[i] = static_cast<uint8_t>((((i + 3) & 15) << 4) | ((i + 9) & 15));
  for (size_t i = 0; i < nf_absmax.size(); ++i) nf_absmax[i] = static_cast<uint8_t>(i * 97);
  std::array<float, 16> nf_map{};
  std::array<float, 256> nested_map{};
  for (size_t i = 0; i < nf_map.size(); ++i) nf_map[i] = (float(i) - 7.0f) / 8.0f;
  for (size_t i = 0; i < nested_map.size(); ++i)
    nested_map[i] = (float(i) - 127.0f) / 128.0f;
  const float nested_absmax[] = {0.75f};
  const float offset = 0.125f;
  std::vector<uint16_t> nf_expected(nf_count);
  for (size_t i = 0; i < nf_count; ++i) {
    const size_t scale_index = i / 64;
    const float scale_value = nested_map[nf_absmax[scale_index]] *
                                  nested_absmax[scale_index / 256] + offset;
    const uint8_t packed = nf_codes[i / 2];
    const uint8_t code = (i & 1) == 0 ? packed >> 4 : packed & 15;
    nf_expected[i] = reference_bf16(nf_map[code] * scale_value);
  }
  LinearWeightUpload nf_upload;
  nf_upload.format = LinearWeightFormat::kNF4;
  nf_upload.out_features = nf_out;
  nf_upload.in_features = nf_in;
  nf_upload.data = nf_codes.data();
  nf_upload.data_bytes = nf_codes.size();
  nf_upload.nf4_absmax = nf_absmax.data();
  nf_upload.nf4_absmax_count = nf_absmax.size();
  nf_upload.nf4_quant_map = nf_map.data();
  nf_upload.nf4_quant_map_count = nf_map.size();
  nf_upload.nf4_nested_quant_map = nested_map.data();
  nf_upload.nf4_nested_quant_map_count = nested_map.size();
  nf_upload.nf4_nested_absmax = nested_absmax;
  nf_upload.nf4_nested_absmax_count = 1;
  nf_upload.nf4_nested_offset = offset;
  run("nf4", nf_upload, nf_expected);
}

VIDFAB_TEST(vulkan_streamed_nvfp4_gemm_cache) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_cooperative_matrix = physical.front().info().cooperative_matrix;
  options.enable_storage_buffer_16bit = options.enable_cooperative_matrix;
  Device device = physical.front().create_device(options);
  TensorContext context(device);
  CHECK(!context.native_nvfp4_gemm_available());
  bool native_threw = false;
  try { context.require_native_nvfp4_gemm(); }
  catch (const std::runtime_error&) { native_threw = true; }
  CHECK(native_threw);

  constexpr uint32_t rows = 6, n = 128, k = 64;
  const size_t weight_elements = static_cast<size_t>(n) * k;
  std::vector<uint8_t> positive(weight_elements / 2, 0x22);
  std::vector<uint8_t> negative(weight_elements / 2, 0xaa);
  std::vector<uint8_t> scales(weight_elements / 16, 0x38);
  LinearWeightUpload upload;
  upload.format = LinearWeightFormat::kNVFloat4;
  upload.out_features = n;
  upload.in_features = k;
  upload.data_bytes = positive.size();
  upload.block_scale = scales.data();
  upload.block_scale_count = scales.size();
  upload.global_scale = 1.0f;
  upload.data = positive.data();
  LinearWeight w_positive = LinearWeight::upload(context, upload);
  upload.data = negative.data();
  upload.full_precision_matrix_mult = true;
  LinearWeight w_negative = LinearWeight::upload(context, upload);
  std::vector<uint16_t> pre_scale(k, reference_bf16(1.0f));
  upload.data = positive.data();
  upload.pre_quant_scale_bf16 = pre_scale.data();
  upload.pre_quant_scale_count = pre_scale.size();
  LinearWeight awq_weight = LinearWeight::upload(context, upload);

  const uint64_t input_shape[] = {rows, k};
  const uint64_t output_shape[] = {rows, n};
  DeviceTensor input = context.allocate(
      TensorLayout::contiguous(input_shape, 2), ScalarType::kBFloat16);
  DeviceTensor output = context.allocate(
      TensorLayout::contiguous(output_shape, 2), ScalarType::kBFloat16);
  std::vector<uint16_t> input_bits(static_cast<size_t>(rows) * k,
                                   reference_bf16(1.0f));
  std::vector<uint16_t> sentinel(static_cast<size_t>(rows) * n, 0x7fc1);
  context.upload_bytes(input, input_bits.data(), input_bits.size() * 2);
  context.upload_bytes(output, sentinel.data(), sentinel.size() * 2);
  DenseGemmPlan plan = DenseGemmPlan::create(
      context, {rows, n, k, DenseGemmMode::kBFloat16,
                DenseGemmBias::kNone});
  StreamedNVFP4WeightCache cache =
      StreamedNVFP4WeightCache::create(context, weight_elements * 2);
  CHECK(cache.capacity_elements() == weight_elements * 2);
  CHECK(cache.dense_bytes() == weight_elements * 4);

  // A foreign weight with a different valid shape must fail before it changes
  // the shared slot metadata or invalidates the current W1 generation.
  TensorContext foreign_context(device);
  std::vector<uint8_t> foreign_codes(weight_elements, 0x22);
  std::vector<uint8_t> foreign_scales(weight_elements / 8, 0x38);
  LinearWeightUpload foreign_upload = upload;
  foreign_upload.in_features = 128;
  foreign_upload.pre_quant_scale_bf16 = nullptr;
  foreign_upload.pre_quant_scale_count = 0;
  foreign_upload.data = foreign_codes.data();
  foreign_upload.data_bytes = foreign_codes.size();
  foreign_upload.block_scale = foreign_scales.data();
  foreign_upload.block_scale_count = foreign_scales.size();
  LinearWeight foreign_weight = LinearWeight::upload(foreign_context, foreign_upload);
  DenseGemmPlan foreign_shape_plan = DenseGemmPlan::create(
      context, {rows, n, 128, DenseGemmMode::kBFloat16,
                DenseGemmBias::kNone});

  TensorBatch first = context.begin_batch();
  PreparedNVFP4WeightView p = cache.prepare(first, w_positive, plan);
  const uint32_t capacity_after_prepare = first.remaining_operator_capacity();
  bool foreign_threw = false;
  try { (void)cache.prepare(first, foreign_weight, foreign_shape_plan); }
  catch (const std::invalid_argument&) { foreign_threw = true; }
  CHECK(foreign_threw);
  CHECK(first.remaining_operator_capacity() == capacity_after_prepare);
  bool shape_threw = false;
  try { (void)cache.prepare(first, w_positive, foreign_shape_plan); }
  catch (const std::invalid_argument&) { shape_threw = true; }
  CHECK(shape_threw);
  CHECK(first.remaining_operator_capacity() == capacity_after_prepare);
  // Both failures leave the prior generation, dense layout and access state
  // intact: it remains immediately recordable in this same batch.
  plan.record(first, input, p, output, 2, 0, 0);
  // AWQ transforms the activation before GEMM; it does not alter NVFP4
  // materialization. Preparing such a weight is therefore valid and, like
  // every successful prepare, supersedes the preceding cache generation.
  (void)cache.prepare(first, awq_weight, plan);
  bool awq_superseded_p = false;
  try { plan.record(first, input, p, output, 1, 0, 0); }
  catch (const std::invalid_argument&) { awq_superseded_p = true; }
  CHECK(awq_superseded_p);
  p = cache.prepare(first, w_positive, plan);
  plan.record(first, input, p, output, 2, 2, 2);
  PreparedNVFP4WeightView m = cache.prepare(first, w_negative, plan);
  CHECK(m.full_precision_matrix_mult());
  bool stale_threw = false;
  try { plan.record(first, input, p, output, 1, 0, 0); }
  catch (const std::invalid_argument&) { stale_threw = true; }
  CHECK(stale_threw);
  plan.record(first, input, m, output, 2, 4, 4);
  Submission first_token = first.submit();

  // Overwrite the same cache in a second queued submission. The queue-ordered
  // R->W barrier protects the first job without a CPU/device-wide wait.
  DeviceTensor second_output = context.allocate(
      TensorLayout::contiguous(output_shape, 2), ScalarType::kBFloat16);
  context.upload_bytes(second_output, sentinel.data(), sentinel.size() * 2);
  TensorBatch second = context.begin_batch();
  PreparedNVFP4WeightView again = cache.prepare(second, w_positive, plan);
  plan.record(second, input, again, second_output, rows);
  Submission second_token = second.submit();
  first_token.wait();
  second_token.wait();

  std::vector<uint16_t> got(sentinel.size()), got_second(sentinel.size());
  context.download_bytes(output, got.data(), got.size() * 2);
  context.download_bytes(second_output, got_second.data(), got_second.size() * 2);
  const uint16_t plus = reference_bf16(64.0f);
  const uint16_t minus = reference_bf16(-64.0f);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t col = 0; col < n; ++col) {
      CHECK(got[static_cast<size_t>(row) * n + col] ==
            (row < 4 ? plus : minus));
      CHECK(got_second[static_cast<size_t>(row) * n + col] == plus);
    }
  }

  // A materialize plus 31 chunk/fanout reads exactly fills the bounded
  // 32-operation schedule. The 33rd operation is rejected, and discarding the
  // poisoned recording leaves the following batch usable.
  Submission warm_flights[2];
  for (int flight = 0; flight < 2; ++flight) {
    TensorBatch full = context.begin_batch();
    PreparedNVFP4WeightView prepared = cache.prepare(full, w_positive, plan);
    for (int i = 0; i < 31; ++i)
      plan.record(full, input, prepared, output, rows);
    warm_flights[flight] = full.submit();
  }
  warm_flights[0].wait(); warm_flights[1].wait();
  const uint64_t high_reserved = context.reserved_bytes();
  const uint64_t high_descriptors = context.descriptor_set_allocations();
  {
    TensorBatch overflow = context.begin_batch();
    PreparedNVFP4WeightView prepared = cache.prepare(overflow, w_positive, plan);
    for (int i = 0; i < 31; ++i)
      plan.record(overflow, input, prepared, output, rows);
    bool threw = false;
    try { plan.record(overflow, input, prepared, output, rows); }
    catch (const std::logic_error&) { threw = true; }
    CHECK(threw);
  }
  Submission tail;
  for (int i = 0; i < 50; ++i) {
    TensorBatch repeat = context.begin_batch();
    PreparedNVFP4WeightView prepared = cache.prepare(repeat, w_positive, plan);
    plan.record(repeat, input, prepared, output, rows);
    tail = repeat.submit();
  }
  tail.wait();
  CHECK(context.reserved_bytes() == high_reserved);
  CHECK(context.descriptor_set_allocations() == high_descriptors);
}

VIDFAB_TEST(vulkan_streamed_nvfp4_wrapper_drop) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_cooperative_matrix = physical.front().info().cooperative_matrix;
  options.enable_storage_buffer_16bit = options.enable_cooperative_matrix;
  Device device = physical.front().create_device(options);
  TensorContext context(device);
  {
    const uint64_t warm_shape[] = {2, 64};
    DeviceTensor warm = context.allocate(
        TensorLayout::contiguous(warm_shape, 2), ScalarType::kBFloat16);
    std::vector<uint16_t> zeros(128);
    context.upload_bytes(warm, zeros.data(), zeros.size() * 2);
  }
  { TensorBatch collect = context.begin_batch(); }
  const uint64_t baseline = context.pooled_used_bytes();
  Submission token;
  {
    constexpr uint32_t rows = 2, n = 128, k = 64;
    const size_t elements = size_t(n) * k;
    std::vector<uint8_t> codes(elements / 2, 0x22);
    std::vector<uint8_t> scales(elements / 16, 0x38);
    LinearWeightUpload upload;
    upload.format = LinearWeightFormat::kNVFloat4;
    upload.out_features = n; upload.in_features = k;
    upload.data = codes.data(); upload.data_bytes = codes.size();
    upload.block_scale = scales.data();
    upload.block_scale_count = scales.size();
    LinearWeight weight = LinearWeight::upload(context, upload);
    const uint64_t is[] = {rows, k}, os[] = {rows, n};
    DeviceTensor input = context.allocate(TensorLayout::contiguous(is, 2),
                                          ScalarType::kBFloat16);
    DeviceTensor output = context.allocate(TensorLayout::contiguous(os, 2),
                                           ScalarType::kBFloat16);
    std::vector<uint16_t> bits(size_t(rows) * k, reference_bf16(1.0f));
    context.upload_bytes(input, bits.data(), bits.size() * 2);
    DenseGemmPlan plan = DenseGemmPlan::create(
        context, {rows, n, k, DenseGemmMode::kBFloat16,
                  DenseGemmBias::kNone});
    StreamedNVFP4WeightCache cache =
        StreamedNVFP4WeightCache::create(context, elements);
    TensorBatch batch = context.begin_batch();
    PreparedNVFP4WeightView prepared = cache.prepare(batch, weight, plan);
    plan.record(batch, input, prepared, output, rows);
    token = batch.submit();
  }
  CHECK(context.pooled_used_bytes() > baseline);
  token.wait();
  token = Submission{};
  { TensorBatch collect = context.begin_batch(); }
  CHECK_MSG(context.pooled_used_bytes() == baseline,
            "streamed wrapper drop retained %llu bytes (baseline %llu)",
            static_cast<unsigned long long>(context.pooled_used_bytes()),
            static_cast<unsigned long long>(baseline));
}

VIDFAB_TEST(vulkan_dense_gemm_tail_reference) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_cooperative_matrix = physical.front().info().cooperative_matrix;
  options.enable_storage_buffer_16bit = options.enable_cooperative_matrix;
  options.enable_shader_float16 = options.enable_cooperative_matrix &&
      physical.front().info().shader_float16;
  Device device = physical.front().create_device(options);
  TensorContext context(device);
  constexpr uint32_t rows = 3, total_input_rows = 5, total_output_rows = 6;
  constexpr uint32_t n = 11, k = 19;
  const uint64_t input_shape[] = {total_input_rows, k};
  const uint64_t weight_shape[] = {n, k};
  const uint64_t output_shape[] = {total_output_rows, n};
  const uint64_t bias_shape[] = {n};
  DeviceTensor input = context.allocate(TensorLayout::contiguous(input_shape, 2),
                                        ScalarType::kBFloat16);
  DeviceTensor weight = context.allocate(TensorLayout::contiguous(weight_shape, 2),
                                         ScalarType::kBFloat16);
  DeviceTensor output = context.allocate(TensorLayout::contiguous(output_shape, 2),
                                         ScalarType::kBFloat16);
  DeviceTensor bias = context.allocate(TensorLayout::contiguous(bias_shape, 1),
                                       ScalarType::kFloat32);
  std::vector<uint16_t> input_bits(total_input_rows * k);
  std::vector<uint16_t> weight_bits(n * k);
  std::vector<uint16_t> output_bits(total_output_rows * n, 0x3e80u);
  std::vector<float> bias_values(n);
  for (size_t i = 0; i < input_bits.size(); ++i)
    input_bits[i] = reference_bf16(static_cast<float>(static_cast<int>(i % 17) - 8) / 16.0f);
  for (size_t i = 0; i < weight_bits.size(); ++i)
    weight_bits[i] = reference_bf16(static_cast<float>(static_cast<int>(i % 13) - 6) / 8.0f);
  for (uint32_t i = 0; i < n; ++i)
    bias_values[i] = static_cast<float>(static_cast<int>(i) - 5) / 32.0f;
  context.upload_bytes(input, input_bits.data(), input_bits.size() * 2);
  context.upload_bytes(weight, weight_bits.data(), weight_bits.size() * 2);
  context.upload_bytes(output, output_bits.data(), output_bits.size() * 2);
  context.upload(bias, bias_values.data(), bias_values.size());

  DenseGemmPlanDesc desc;
  desc.max_rows = 4;
  desc.out_features = n;
  desc.in_features = k;
  desc.mode = DenseGemmMode::kBFloat16;
  desc.bias = DenseGemmBias::kFloat32;
  DenseGemmPlan plan = DenseGemmPlan::create(context, desc);
  TensorBatch batch = context.begin_batch();
  plan.record(batch, input, weight, output, rows, 1, 2, &bias);
  batch.submit().wait();
  std::vector<uint16_t> actual(output_bits.size());
  context.download_bytes(output, actual.data(), actual.size() * 2);
  for (uint32_t row = 0; row < rows; ++row) {
    for (uint32_t column = 0; column < n; ++column) {
      float sum = 0.0f;
      for (uint32_t inner = 0; inner < k; ++inner) {
        float a = 0.0f, w = 0.0f;
        const uint32_t ab = uint32_t(input_bits[(row + 1) * k + inner]) << 16;
        const uint32_t wb = uint32_t(weight_bits[column * k + inner]) << 16;
        std::memcpy(&a, &ab, 4); std::memcpy(&w, &wb, 4);
        sum = std::fma(a, w, sum);
      }
      const uint16_t rounded = reference_bf16(sum);
      const uint32_t rounded_bits = uint32_t(rounded) << 16;
      std::memcpy(&sum, &rounded_bits, 4);
      const uint16_t expected = reference_bf16(sum + bias_values[column]);
      const size_t index = size_t(row + 2) * n + column;
      CHECK_MSG(actual[index] == expected,
                "gemm tail [%u,%u]: %04x != %04x", row, column,
                actual[index], expected);
    }
  }
  for (size_t i = 0; i < actual.size(); ++i) {
    const size_t row = i / n;
    if (row < 2 || row >= 5) CHECK(actual[i] == 0x3e80u);
  }

  auto run_float_mode = [&](DenseGemmMode mode, DenseGemmBias bias_mode) {
    const uint64_t fs[] = {rows, k}, fws[] = {n, k}, fos[] = {rows, n};
    DeviceTensor fi = context.allocate(TensorLayout::contiguous(fs, 2),
                                       ScalarType::kFloat32);
    DeviceTensor fw = context.allocate(
        TensorLayout::contiguous(fws, 2),
        mode == DenseGemmMode::kFloat16Vae ? ScalarType::kFloat16
                                           : ScalarType::kFloat32);
    DeviceTensor fo = context.allocate(TensorLayout::contiguous(fos, 2),
                                       ScalarType::kFloat32);
    std::vector<float> host_input(rows * k), host_output(rows * n);
    std::vector<float> host_weight_f32;
    std::vector<uint16_t> host_weight_f16;
    for (size_t i = 0; i < host_input.size(); ++i)
      host_input[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 17.0f;
    context.upload(fi, host_input.data(), host_input.size());
    if (mode == DenseGemmMode::kFloat16Vae) {
      host_weight_f16.resize(n * k);
      for (size_t i = 0; i < host_weight_f16.size(); ++i)
        host_weight_f16[i] = f32_to_f16(
            static_cast<float>(static_cast<int>(i % 23) - 11) / 19.0f);
      context.upload_bytes(fw, host_weight_f16.data(), host_weight_f16.size() * 2);
    } else {
      host_weight_f32.resize(n * k);
      for (size_t i = 0; i < host_weight_f32.size(); ++i)
        host_weight_f32[i] =
            static_cast<float>(static_cast<int>(i % 23) - 11) / 19.0f;
      context.upload(fw, host_weight_f32.data(), host_weight_f32.size());
    }
    DenseGemmPlanDesc float_desc{rows, n, k, mode, bias_mode};
    DenseGemmPlan float_plan = DenseGemmPlan::create(context, float_desc);
    PreparedF16Activation slot;
    if (mode == DenseGemmMode::kFloat16Vae)
      slot = PreparedF16Activation::create(context, rows, k);
    TensorBatch float_batch = context.begin_batch();
    if (mode == DenseGemmMode::kFloat16Vae) {
      PreparedF16ActivationView prepared = slot.prepare(float_batch, fi, rows);
      float_plan.record(float_batch, prepared, fw, fo, 0,
                        bias_mode == DenseGemmBias::kNone ? nullptr : &bias);
      // A single conversion is shared by multiple projections in the same
      // chunk; recording a second consumer must not re-run preparation.
      float_plan.record(float_batch, prepared, fw, fo, 0,
                        bias_mode == DenseGemmBias::kNone ? nullptr : &bias);
    } else {
      float_plan.record(float_batch, fi, fw, fo, rows, 0, 0,
                        bias_mode == DenseGemmBias::kNone ? nullptr : &bias);
    }
    float_batch.submit().wait();
    context.download(fo, host_output.data(), host_output.size());
    for (uint32_t row = 0; row < rows; ++row) {
      for (uint32_t column = 0; column < n; ++column) {
        float expected = 0.0f;
        for (uint32_t inner = 0; inner < k; ++inner) {
          const float a = mode == DenseGemmMode::kFloat16Vae
              ? f16_to_f32(f32_to_f16(host_input[row * k + inner]))
              : host_input[row * k + inner];
          const float w = mode == DenseGemmMode::kFloat16Vae
              ? f16_to_f32(host_weight_f16[column * k + inner])
              : host_weight_f32[column * k + inner];
          expected = std::fma(a, w, expected);
        }
        if (bias_mode == DenseGemmBias::kFloat32)
          expected += bias_values[column];
        CHECK_MSG(float_bits(host_output[row * n + column]) == float_bits(expected),
                  "gemm float mode %u [%u,%u] differs", unsigned(mode), row,
                  column);
      }
    }
  };
  run_float_mode(DenseGemmMode::kFloat16Vae, DenseGemmBias::kNone);
  run_float_mode(DenseGemmMode::kFloat32, DenseGemmBias::kFloat32);

  // Explicit fp32->fp16 boundary values exercise ties, signed zero,
  // subnormal-half results, carry into infinity, and both signs.
  {
    const std::vector<float> edge = {
        0.0f, -0.0f, std::ldexp(1.0f, -24), std::ldexp(1.0f, -25),
        std::nextafter(std::ldexp(1.0f, -25), 1.0f), 65504.0f,
        65520.0f, -65520.0f};
    const uint64_t es[] = {edge.size(), 1}, ews[] = {1, 1};
    DeviceTensor ei = context.allocate(TensorLayout::contiguous(es, 2));
    DeviceTensor ew = context.allocate(TensorLayout::contiguous(ews, 2),
                                       ScalarType::kFloat16);
    DeviceTensor eo = context.allocate(TensorLayout::contiguous(es, 2));
    const uint16_t one = f32_to_f16(1.0f);
    context.upload(ei, edge.data(), edge.size());
    context.upload_bytes(ew, &one, sizeof(one));
    PreparedF16Activation slot = PreparedF16Activation::create(
        context, static_cast<uint32_t>(edge.size()), 1);
    DenseGemmPlan edge_plan = DenseGemmPlan::create(
        context, {static_cast<uint32_t>(edge.size()), 1, 1,
                  DenseGemmMode::kFloat16Vae, DenseGemmBias::kNone});
    TensorBatch edge_batch = context.begin_batch();
    PreparedF16ActivationView prepared = slot.prepare(
        edge_batch, ei, static_cast<uint32_t>(edge.size()));
    edge_plan.record(edge_batch, prepared, ew, eo);
    edge_batch.submit().wait();
    std::vector<float> got(edge.size());
    context.download(eo, got.data(), got.size());
    for (size_t i = 0; i < edge.size(); ++i) {
      const float narrowed = f16_to_f32(f32_to_f16(edge[i]));
      const float expected = std::fma(narrowed, 1.0f, 0.0f);
      CHECK(std::memcmp(&got[i], &expected, sizeof(float)) == 0);
    }
  }

  bool invalid_mode_rejected = false;
  try {
    DenseGemmPlanDesc invalid{rows, n, k,
        static_cast<DenseGemmMode>(0xffffffffu), DenseGemmBias::kNone};
    (void)DenseGemmPlan::create(context, invalid);
  } catch (const std::invalid_argument&) {
    invalid_mode_rejected = true;
  }
  CHECK(invalid_mode_rejected);
  // Invalid plan construction is entirely pre-record and cannot poison a
  // subsequent valid batch.
  run_float_mode(DenseGemmMode::kFloat32, DenseGemmBias::kNone);
  bool invalid_bias_rejected = false;
  try {
    (void)DenseGemmPlan::create(
        context, {rows, n, k, DenseGemmMode::kFloat16Vae,
                  DenseGemmBias::kFloat32});
  } catch (const std::invalid_argument&) {
    invalid_bias_rejected = true;
  }
  CHECK(invalid_bias_rejected);
  TensorBatch validation_batch = context.begin_batch();
  bool alias_rejected = false, range_rejected = false, missing_bias_rejected = false;
  try {
    plan.record(validation_batch, input, weight, input, rows, 1, 0, &bias);
  } catch (const std::invalid_argument&) {
    alias_rejected = true;
  }
  try {
    plan.record(validation_batch, input, weight, output, desc.max_rows + 1,
                0, 0, &bias);
  } catch (const std::invalid_argument&) {
    range_rejected = true;
  }
  try {
    plan.record(validation_batch, input, weight, output, rows, 1, 0, nullptr);
  } catch (const std::invalid_argument&) {
    missing_bias_rejected = true;
  }
  CHECK(alias_rejected); CHECK(range_rejected); CHECK(missing_bias_rejected);
  plan.record(validation_batch, input, weight, output, rows, 1, 0, &bias);
  validation_batch.submit().wait();

  // Prepared fp16 activations are batch-scoped, shared by distinct
  // projections, and retained exactly through their submission token.
  const uint64_t staging_warm_shape[] = {64, 32};
  {
    DeviceTensor staging_warm = context.allocate(
        TensorLayout::contiguous(staging_warm_shape, 2));
    std::vector<float> staging_warm_values(64 * 32, 0.0f);
    context.upload(staging_warm, staging_warm_values.data(),
                   staging_warm_values.size());
  }
  context.upload_bytes(input, input_bits.data(), input_bits.size() * 2);
  const uint64_t gemm_lifetime_baseline = context.pooled_used_bytes();
  {
    constexpr uint32_t lm = 64, lk = 32, ln0 = 16, ln1 = 32;
    const uint64_t ais[] = {lm, lk}, w0s[] = {ln0, lk}, w1s[] = {ln1, lk};
    const uint64_t o0s[] = {lm, ln0}, o1s[] = {lm, ln1};
    DeviceTensor ai = context.allocate(TensorLayout::contiguous(ais, 2));
    DeviceTensor w0 = context.allocate(TensorLayout::contiguous(w0s, 2),
                                       ScalarType::kFloat16);
    DeviceTensor w1 = context.allocate(TensorLayout::contiguous(w1s, 2),
                                       ScalarType::kFloat16);
    DeviceTensor o0 = context.allocate(TensorLayout::contiguous(o0s, 2));
    DeviceTensor o1 = context.allocate(TensorLayout::contiguous(o1s, 2));
    std::vector<float> ah(size_t(lm) * lk, 0.25f);
    std::vector<uint16_t> w0h(size_t(ln0) * lk, f32_to_f16(0.5f));
    std::vector<uint16_t> w1h(size_t(ln1) * lk, f32_to_f16(-0.25f));
    context.upload(ai, ah.data(), ah.size());
    context.upload_bytes(w0, w0h.data(), w0h.size() * 2);
    context.upload_bytes(w1, w1h.data(), w1h.size() * 2);
    PreparedF16Activation slot0 = PreparedF16Activation::create(context, lm, lk);
    PreparedF16Activation slot1 = PreparedF16Activation::create(context, lm, lk);
    DenseGemmPlan p0 = DenseGemmPlan::create(
        context, {lm, ln0, lk, DenseGemmMode::kFloat16Vae,
                  DenseGemmBias::kNone});
    DenseGemmPlan p1 = DenseGemmPlan::create(
        context, {lm, ln1, lk, DenseGemmMode::kFloat16Vae,
                  DenseGemmBias::kNone});

    TensorBatch first = context.begin_batch();
    PreparedF16ActivationView first_view = slot0.prepare(first, ai, lm);
    TensorBatch moved = std::move(first);
    p0.record(moved, first_view, w0, o0);
    p1.record(moved, first_view, w1, o1);
    Submission first_token = moved.submit();
    TensorBatch second = context.begin_batch();
    PreparedF16ActivationView second_view = slot1.prepare(second, ai, lm);
    p0.record(second, second_view, w0, o0);
    p1.record(second, second_view, w1, o1);
    Submission second_token = second.submit();
    CHECK(second_token.value() > first_token.value());
    // The third begin waits for/reuses the oldest bounded command slot before
    // slot0 is overwritten; it does not allocate a third command arena.
    TensorBatch third = context.begin_batch();
    PreparedF16ActivationView third_view = slot0.prepare(third, ai, lm);
    p0.record(third, third_view, w0, o0);
    Submission third_token = third.submit();
    CHECK(third_token.value() > second_token.value());
    first_token.wait(); second_token.wait(); third_token.wait();
    std::vector<float> got0(size_t(lm) * ln0), got1(size_t(lm) * ln1);
    context.download(o0, got0.data(), got0.size());
    context.download(o1, got1.data(), got1.size());
    for (float value : got0) CHECK(value == 4.0f);
    for (float value : got1) CHECK(value == -2.0f);
    const uint64_t warm_reserved = context.reserved_bytes();
    const uint64_t warm_descriptors = context.descriptor_set_allocations();

    // A submitted view is permanently stale, even if allocator addresses are
    // reused. Its rejection is pre-record, so a fresh view continues in the
    // same batch.
    TensorBatch after_submit = context.begin_batch();
    bool submitted_stale_rejected = false;
    try {
      p0.record(after_submit, third_view, w0, o0);
    } catch (const std::invalid_argument&) {
      submitted_stale_rejected = true;
    }
    CHECK(submitted_stale_rejected);
    PreparedF16ActivationView fresh = slot1.prepare(after_submit, ai, lm);
    p0.record(after_submit, fresh, w0, o0);
    after_submit.submit().wait();

    for (int repeat = 0; repeat < 4; ++repeat) {
      TensorBatch stable = context.begin_batch();
      PreparedF16ActivationView stable_view = slot0.prepare(stable, ai, lm);
      p0.record(stable, stable_view, w0, o0);
      p1.record(stable, stable_view, w1, o1);
      stable.submit().wait();
      CHECK(context.reserved_bytes() == warm_reserved);
      CHECK(context.descriptor_set_allocations() == warm_descriptors);
    }

    // Preparation is one logical operator: 31 consumers reach the exact
    // 32-op bound and the 32nd consumer poisons/rejects submission.
    bool thirty_third_gemm_rejected = false;
    {
      TensorBatch bounded = context.begin_batch();
      PreparedF16ActivationView bounded_view = slot0.prepare(bounded, ai, lm);
      for (int i = 0; i < 31; ++i) p0.record(bounded, bounded_view, w0, o0);
      try {
        p0.record(bounded, bounded_view, w0, o0);
      } catch (const std::logic_error&) {
        thirty_third_gemm_rejected = true;
      }
    }
    CHECK(thirty_third_gemm_rejected);
  }
  // A boundary operation collects completed jobs. All wrappers above were
  // dropped while the context stayed alive; no GEMM-owned device allocation
  // remains pinned.
  context.upload_bytes(input, input_bits.data(), input_bits.size() * 2);
  CHECK(context.pooled_used_bytes() == gemm_lifetime_baseline);

  // Drop every caller wrapper immediately after submit. The job retains all
  // four tensors, the prepared slot and plan context until its exact token is
  // collected; afterward the pool returns to the pre-job live-byte baseline.
  Submission wrapper_drop_token;
  {
    constexpr uint32_t dm = 64, dn = 16, dk = 32;
    const uint64_t das[] = {dm, dk}, dws[] = {dn, dk}, dos[] = {dm, dn};
    DeviceTensor da = context.allocate(TensorLayout::contiguous(das, 2));
    DeviceTensor dw = context.allocate(TensorLayout::contiguous(dws, 2),
                                       ScalarType::kFloat16);
    DeviceTensor dout = context.allocate(TensorLayout::contiguous(dos, 2));
    PreparedF16Activation dslot =
        PreparedF16Activation::create(context, dm, dk);
    DenseGemmPlan dplan = DenseGemmPlan::create(
        context, {dm, dn, dk, DenseGemmMode::kFloat16Vae,
                  DenseGemmBias::kNone});
    TensorBatch drop_batch = context.begin_batch();
    PreparedF16ActivationView dview = dslot.prepare(drop_batch, da, dm);
    dplan.record(drop_batch, dview, dw, dout);
    wrapper_drop_token = drop_batch.submit();
  }
  CHECK(context.pooled_used_bytes() > gemm_lifetime_baseline);
  wrapper_drop_token.wait();
  context.upload_bytes(input, input_bits.data(), input_bits.size() * 2);
  CHECK(context.pooled_used_bytes() == gemm_lifetime_baseline);

  auto check_isolated_view_contract = [&](bool supersession) {
    TensorContext isolated(device);
    constexpr uint32_t im = 1, in = 1, ik = 1;
    const uint64_t is[] = {im, ik}, ws[] = {in, ik}, os[] = {im, in};
    DeviceTensor ii = isolated.allocate(TensorLayout::contiguous(is, 2));
    DeviceTensor iw = isolated.allocate(TensorLayout::contiguous(ws, 2),
                                        ScalarType::kFloat16);
    DeviceTensor io = isolated.allocate(TensorLayout::contiguous(os, 2));
    PreparedF16Activation slot =
        PreparedF16Activation::create(isolated, im, ik);
    DenseGemmPlan plan = DenseGemmPlan::create(
        isolated, {im, in, ik, DenseGemmMode::kFloat16Vae,
                   DenseGemmBias::kNone});
    if (supersession) {
      TensorBatch batch = isolated.begin_batch();
      PreparedF16ActivationView old = slot.prepare(batch, ii, im);
      PreparedF16ActivationView newest = slot.prepare(batch, ii, im);
      bool rejected = false;
      try {
        plan.record(batch, old, iw, io);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      CHECK(rejected);
      plan.record(batch, newest, iw, io);
      batch.submit().wait();
    } else {
      PreparedF16ActivationView discarded;
      {
        TensorBatch abandoned = isolated.begin_batch();
        discarded = slot.prepare(abandoned, ii, im);
      }
      TensorBatch next = isolated.begin_batch();
      bool rejected = false;
      try {
        plan.record(next, discarded, iw, io);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      CHECK(rejected);
      PreparedF16ActivationView fresh = slot.prepare(next, ii, im);
      plan.record(next, fresh, iw, io);
      next.submit().wait();
    }
  };
  check_isolated_view_contract(true);
  check_isolated_view_contract(false);

  // Temporary development measurement; retained as a visible performance
  // guard until the production-shape suite supplies the same metric.
  {
    constexpr uint32_t bm = 64, bn = 5376, bk = 5376;
    const uint64_t as[] = {bm, bk}, ws[] = {bn, bk}, os[] = {bm, bn};
    DeviceTensor ai = context.allocate(TensorLayout::contiguous(as, 2),
                                       ScalarType::kBFloat16);
    DeviceTensor wi = context.allocate(TensorLayout::contiguous(ws, 2),
                                       ScalarType::kBFloat16);
    DeviceTensor oi = context.allocate(TensorLayout::contiguous(os, 2),
                                       ScalarType::kBFloat16);
    std::vector<uint16_t> az(size_t(bm) * bk, reference_bf16(0.25f));
    std::vector<uint16_t> wz(size_t(bn) * bk, reference_bf16(0.001f));
    context.upload_bytes(ai, az.data(), az.size() * 2);
    context.upload_bytes(wi, wz.data(), wz.size() * 2);
    DenseGemmPlanDesc bd{bm, bn, bk, DenseGemmMode::kBFloat16,
                         DenseGemmBias::kNone};
    DenseGemmPlan bp = DenseGemmPlan::create(context, bd);
    for (int iteration = -1; iteration < 3; ++iteration) {
      const auto start = std::chrono::steady_clock::now();
      TensorBatch b = context.begin_batch();
      bp.record(b, ai, wi, oi, bm);
      b.submit().wait();
      if (iteration >= 0) {
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        std::printf("  deterministic Vulkan BF16 GEMM 64x5376x5376: %.3f ms\n", ms);
      }
    }
    const auto batched_start = std::chrono::steady_clock::now();
    TensorBatch repeated = context.begin_batch();
    for (int i = 0; i < 16; ++i) bp.record(repeated, ai, wi, oi, bm);
    repeated.submit().wait();
    const double batched_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - batched_start).count() / 16.0;
    std::printf("  deterministic Vulkan BF16 GEMM batched device time proxy: %.3f ms\n",
                batched_ms);
  }
}

VIDFAB_TEST(vulkan_runtime_and_pool) {
  using namespace vidfab::vulkan;

  std::string diagnostic;
  if (!Instance::available(&diagnostic)) {
    // Absence is a supported runtime state. The important contract is a useful
    // diagnostic rather than a load-time process failure.
    CHECK(!diagnostic.empty());
    return;
  }

  bool old_api_rejected = false;
  try {
    InstanceOptions old_api;
    old_api.api_version = (1u << 22) | (1u << 12);  // Vulkan 1.1.0
    (void)Instance::create(old_api);
  } catch (const std::runtime_error&) {
    old_api_rejected = true;
  }
  CHECK(old_api_rejected);

  Instance instance = Instance::create();
  CHECK(static_cast<bool>(instance));
  CHECK(instance.loader_version().major >= 1);
  CHECK(instance.api_version().major > 1 || instance.api_version().minor >= 2);
  const std::vector<PhysicalDevice> physical = instance.enumerate_devices();
  if (physical.empty()) {
    CHECK(physical.empty());
    return;
  }

  const DeviceInfo& info = physical.front().info();
  CHECK(!info.name.empty());
  CHECK(info.api_version.major >= 1);
  CHECK(info.compute_queue_count > 0);
  CHECK(info.max_compute_workgroup_invocations > 0);
  CHECK(info.max_compute_shared_memory_bytes > 0);
  CHECK(!info.memory_heaps.empty());

  DeviceOptions feature_options;
  feature_options.enable_shader_float16 = info.shader_float16;
  feature_options.enable_shader_int8 = info.shader_int8;
  feature_options.enable_storage_buffer_16bit = info.storage_buffer_16bit;
  feature_options.enable_storage_buffer_8bit = info.storage_buffer_8bit;
  feature_options.enable_timeline_semaphore = info.timeline_semaphore;
  feature_options.enable_buffer_device_address = info.buffer_device_address;
  feature_options.enable_descriptor_indexing = info.descriptor_indexing;
  Device feature_device = physical.front().create_device(feature_options);
  CHECK(static_cast<bool>(feature_device));
  if (info.buffer_device_address) {
    BufferPool address_pool(feature_device, 64 * 1024);
    {
      Buffer addressable = address_pool.allocate(
          257, BufferUsage::kStorage | BufferUsage::kDeviceAddress,
          MemoryUsage::kDevice);
      CHECK(static_cast<bool>(addressable));
      CHECK(address_pool.used_bytes() >= addressable.size());
    }
    CHECK(address_pool.used_bytes() == 0);
    address_pool.trim();
    CHECK(address_pool.reserved_bytes() == 0);
  }
  feature_device.wait_idle();
  feature_device = Device();

  Device device = physical.front().create_device();
  CHECK(static_cast<bool>(device));
  CHECK(device.native_handle() != nullptr);
  Queue queue = device.compute_queue();
  CHECK(static_cast<bool>(queue));
  CHECK(queue.native_handle() != nullptr);
  CHECK(queue.family_index() == info.compute_queue_family);

  // A deliberately small block makes reuse and trimming observable without
  // reserving meaningful VRAM in the unit suite.
  BufferPool pool(device, 64 * 1024);
  bool disabled_address_rejected = false;
  try {
    (void)pool.allocate(256, BufferUsage::kStorage | BufferUsage::kDeviceAddress,
                        MemoryUsage::kDevice);
  } catch (const std::logic_error&) {
    disabled_address_rejected = true;
  }
  CHECK(disabled_address_rejected);
  const BufferUsage transfer = BufferUsage::kTransferSource |
                               BufferUsage::kTransferDestination;
  uintptr_t tiny_native = 0;
  uint64_t tiny_offset = 0;
  {
    Buffer tiny = pool.allocate(1, transfer, MemoryUsage::kUpload);
    Buffer unaligned = pool.allocate(3, transfer, MemoryUsage::kUpload);
    tiny_native = tiny.native_handle();
    tiny_offset = tiny.memory_offset();
    CHECK(unaligned.memory_offset() >= tiny.memory_offset() +
          info.non_coherent_atom_bytes);
    CHECK(pool.used_bytes() >= 4);
  }
  CHECK(pool.used_bytes() == 0);
  Buffer reused = pool.allocate(1, transfer, MemoryUsage::kUpload);
  CHECK(reused.memory_offset() == tiny_offset);
  // Vulkan buffer objects themselves need not be recycled; their pooled
  // memory spans are. Keep this check merely to ensure a real handle exists.
  CHECK(tiny_native != 0);
  reused.reset_after_idle(queue);
  CHECK(!static_cast<bool>(reused));
  CHECK(pool.used_bytes() == 0);

  Buffer first = pool.allocate(4096, transfer, MemoryUsage::kUpload);
  Buffer second = pool.allocate(4096, transfer, MemoryUsage::kUpload);
  CHECK(first.mapped_data() != nullptr);
  CHECK(second.mapped_data() != nullptr);
  CHECK(pool.used_bytes() >= 8192);
  CHECK(pool.reserved_bytes() >= pool.used_bytes());

  std::vector<uint32_t> source(1024);
  for (uint32_t i = 0; i < source.size(); ++i) source[i] = i * 2654435761u;
  first.write(0, source.data(), source.size() * sizeof(uint32_t));
  std::vector<uint32_t> copy(source.size());
  first.read(0, copy.data(), copy.size() * sizeof(uint32_t));
  CHECK(std::memcmp(source.data(), copy.data(), source.size() * sizeof(uint32_t)) == 0);

  first = Buffer();
  second = Buffer();
  CHECK(pool.used_bytes() == 0);
  pool.trim();
  CHECK(pool.reserved_bytes() == 0);
  queue.wait_idle();
}

VIDFAB_TEST(vulkan_compute_submission) {
  using namespace vidfab::vulkan;
  if (!Instance::available()) return;
  Instance instance = Instance::create();
  std::vector<PhysicalDevice> physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;

  DeviceOptions device_options;
  device_options.enable_timeline_semaphore = true;
  Device device = physical.front().create_device(device_options);
  ComputeContextOptions context_options;
  context_options.max_in_flight = 2;
  context_options.max_storage_bindings = 2;
  ComputeContext context(device, context_options);
  ComputeContext other_context(device, context_options);

  const std::vector<uint32_t> spirv = load_spirv(VIDFAB_TEST_AFFINE_SPV_PATH);
  ComputePipelineOptions pipeline_options;
  pipeline_options.storage_binding_count = 2;
  pipeline_options.push_constant_bytes = 12;
  pipeline_options.local_size[0] = 64;
  ComputePipeline pipeline = ComputePipeline::create(device, spirv, pipeline_options);

  bool local_size_rejected = false;
  try {
    ComputePipelineOptions wrong = pipeline_options;
    wrong.local_size[0] = 32;
    (void)ComputePipeline::create(device, spirv, wrong);
  } catch (const std::invalid_argument&) {
    local_size_rejected = true;
  }
  CHECK(local_size_rejected);

  // A list is context-owned and cannot be submitted through a different
  // timeline. Its destructor safely returns the still-recording slot.
  bool foreign_rejected = false;
  {
    CommandList foreign = other_context.begin();
    try {
      (void)context.submit(std::move(foreign));
    } catch (const std::invalid_argument&) {
      foreign_rejected = true;
    }
  }
  CHECK(foreign_rejected);
  CHECK(other_context.in_flight() == 0);

  constexpr uint32_t count = 1003;  // deliberately not a multiple of local_size_x
  constexpr uint64_t bytes = static_cast<uint64_t>(count) * sizeof(float);
  BufferPool pool(device, 64 * 1024);
  const BufferUsage upload_usage = BufferUsage::kTransferSource;
  const BufferUsage input_usage = BufferUsage::kTransferDestination | BufferUsage::kStorage;
  const BufferUsage output_usage = BufferUsage::kStorage | BufferUsage::kTransferSource;
  const BufferUsage readback_usage = BufferUsage::kTransferDestination;

  // Range and usage failures happen before any Vulkan command is emitted and
  // the abandoned list remains recyclable.
  {
    Buffer tiny_src = pool.allocate(16, upload_usage, MemoryUsage::kUpload);
    Buffer tiny_dst = pool.allocate(16, input_usage, MemoryUsage::kDevice);
    Buffer tiny_out = pool.allocate(16, output_usage, MemoryUsage::kDevice);
    CommandList invalid = context.begin();
    bool range_rejected = false;
    try {
      invalid.copy_buffer(tiny_src, tiny_dst, 17);
    } catch (const std::out_of_range&) {
      range_rejected = true;
    }
    CHECK(range_rejected);
    bool descriptor_range_rejected = false;
    try {
      invalid.bind_compute(pipeline, {{0, &tiny_dst, 0, 17},
                                      {1, &tiny_out, 0, 16}});
    } catch (const std::out_of_range&) {
      descriptor_range_rejected = true;
    }
    CHECK(descriptor_range_rejected);
    if (physical.front().info().min_storage_buffer_offset_alignment > 1) {
      bool descriptor_alignment_rejected = false;
      try {
        invalid.bind_compute(pipeline, {{0, &tiny_dst, 1, 4},
                                        {1, &tiny_out, 0, 16}});
      } catch (const std::invalid_argument&) {
        descriptor_alignment_rejected = true;
      }
      CHECK(descriptor_alignment_rejected);
    }
    Buffer self_copy = pool.allocate(
        32, BufferUsage::kTransferSource | BufferUsage::kTransferDestination,
        MemoryUsage::kUpload);
    bool overlap_rejected = false;
    try {
      invalid.copy_buffer(self_copy, self_copy, 16, 0, 8);
    } catch (const std::invalid_argument&) {
      overlap_rejected = true;
    }
    CHECK(overlap_rejected);
    // Non-overlapping regions in one VkBuffer are legal.
    invalid.copy_buffer(self_copy, self_copy, 16, 0, 16);
  }
  CHECK(context.in_flight() == 0);
  CHECK(pool.used_bytes() == 0);

  struct Parameters { float scale; float bias; uint32_t count; };
  struct Job {
    Submission completion;
    Buffer readback;
    float scale = 0;
    float bias = 0;
  };
  std::vector<float> input(count);
  for (uint32_t i = 0; i < count; ++i) input[i] = static_cast<float>(i) * 0.125f - 7.0f;
  std::vector<Job> jobs;

  for (uint32_t iteration = 0; iteration < 5; ++iteration) {
    Buffer upload = pool.allocate(bytes, upload_usage, MemoryUsage::kUpload);
    Buffer device_input = pool.allocate(bytes, input_usage, MemoryUsage::kDevice);
    Buffer device_output = pool.allocate(bytes, output_usage, MemoryUsage::kDevice);
    Buffer readback = pool.allocate(bytes, readback_usage, MemoryUsage::kReadback);
    upload.write(0, input.data(), bytes);

    const Parameters parameters{1.25f + iteration * 0.5f,
                                -3.0f + static_cast<float>(iteration), count};
    CommandList commands = context.begin();
    commands.barrier(upload, BufferAccess::kHostWrite, BufferAccess::kTransferRead);
    commands.copy_buffer(upload, device_input, bytes);
    commands.barrier(device_input, BufferAccess::kTransferWrite, BufferAccess::kComputeRead);
    commands.bind_compute(pipeline, {{0, &device_input, 0, bytes},
                                     {1, &device_output, 0, bytes}});
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch((count + 63) / 64);
    commands.barrier(device_output, BufferAccess::kComputeWrite,
                     BufferAccess::kTransferRead);
    commands.copy_buffer(device_output, readback, bytes);
    commands.barrier(readback, BufferAccess::kTransferWrite, BufferAccess::kHostRead);
    Submission completion = context.submit(std::move(commands));
    CHECK(completion.value() != 0);
    CHECK(context.in_flight() <= 2);
    jobs.push_back({std::move(completion), std::move(readback),
                    parameters.scale, parameters.bias});
    // upload/device_input/device_output wrappers die here. The submitted slot
    // retains their allocations until its exact timeline value completes.
  }
  pipeline = ComputePipeline();  // in-flight jobs retain the pipeline too

  uint64_t previous_value = 0;
  for (Job& job : jobs) {
    CHECK(job.completion.value() > previous_value);
    previous_value = job.completion.value();
    job.completion.wait();
    context.collect();
    std::vector<float> output(count);
    job.readback.read(0, output.data(), bytes);
    for (uint32_t i = 0; i < count; ++i) {
      CHECK_NEAR(output[i], input[i] * job.scale + job.bias, 1e-5);
    }
  }
  CHECK(context.in_flight() == 0);
  jobs.clear();
  context.collect();
  CHECK(pool.used_bytes() == 0);
  pool.trim();
  CHECK(pool.reserved_bytes() == 0);
}

VIDFAB_TEST(vulkan_tensor_batch_and_workspace) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!Instance::available()) return;
  Instance instance = Instance::create();
  const auto devices = instance.enumerate_devices();
  if (devices.empty() || !devices.front().info().timeline_semaphore) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  Device device = devices.front().create_device(options);
  TensorContext tensors(device);
  TensorContext other(device);
  CHECK(tensors.full_fp32_arithmetic_exactness() ==
        devices.front().info().fp32_denorm_preserve);
  bool exact_gate_rejected = false;
  try {
    tensors.require_full_fp32_arithmetic_exactness();
  } catch (const std::runtime_error&) {
    exact_gate_rejected = true;
  }
  CHECK(exact_gate_rejected == !devices.front().info().fp32_denorm_preserve);
  const uint64_t extent = 257;
  const TensorLayout layout = TensorLayout::contiguous(&extent, 1);
  DeviceTensor a = tensors.allocate(layout);
  DeviceTensor b = tensors.allocate(layout);
  DeviceTensor sum = tensors.allocate(layout);
  DeviceTensor copied = tensors.allocate(layout);
  DeviceTensor tail = tensors.allocate(layout);

  std::vector<float> host_a(extent), host_b(extent);
  for (size_t i = 0; i < host_a.size(); ++i) {
    host_a[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 16.0f;
    host_b[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 32.0f;
  }
  const uint32_t special_a[] = {
      0x00000000u, 0x80000000u, 0x00000001u, 0x00800000u,
      0x3f800000u, 0x3f800000u, 0x7f800000u, 0xff800000u,
      0x7fc12345u, 0x7fa54321u};
  const uint32_t special_b[] = {
      0x80000000u, 0x80000000u, 0x00000001u, 0x807fffffu,
      0x33800000u, 0x33800001u, 0x3f800000u, 0xbf800000u,
      0x00000000u, 0x00000000u};
  static_assert(sizeof(float) == sizeof(uint32_t));
  std::memcpy(host_a.data(), special_a, sizeof(special_a));
  std::memcpy(host_b.data(), special_b, sizeof(special_b));
  tensors.upload(a, host_a.data(), extent);
  tensors.upload(b, host_b.data(), extent);

  // Access state is speculative during recording, so a context admits exactly
  // one open recorder/boundary operation. Discard releases the lease and rolls
  // state back; successful submit releases it before GPU completion.
  {
    TensorBatch open = tensors.begin_batch();
    open.add(a, b, sum);
    bool second_begin_rejected = false;
    try {
      (void)tensors.begin_batch();
    } catch (const std::logic_error&) {
      second_begin_rejected = true;
    }
    CHECK(second_begin_rejected);
    bool upload_rejected = false;
    try {
      tensors.upload(a, host_a.data(), extent);
    } catch (const std::logic_error&) {
      upload_rejected = true;
    }
    CHECK(upload_rejected);
    bool download_rejected = false;
    try {
      tensors.download(a, host_a.data(), extent);
    } catch (const std::logic_error&) {
      download_rejected = true;
    }
    CHECK(download_rejected);
  }
  tensors.upload(a, host_a.data(), extent);
  TensorBatch releasing = tensors.begin_batch();
  releasing.add(a, b, sum);
  Submission releasing_token = releasing.submit();
  TensorBatch after_submit = tensors.begin_batch();
  after_submit.add(a, b, copied);
  Submission after_submit_token = after_submit.submit();
  releasing_token.wait();
  after_submit_token.wait();

  // Four operators, one command buffer, one timeline submission. The first
  // copy covers payload NaNs, signed zero, subnormal and infinities exactly;
  // arithmetic comparisons below use the ordinary finite tail, while the
  // CUDA/Vulkan test covers special-value arithmetic on both devices.
  TensorBatch batch = tensors.begin_batch();
  batch.copy(a, copied);
  batch.add(a, b, sum);
  batch.copy(sum, tail);
  batch.add(sum, b, tail);
  Submission done = batch.submit();
  CHECK(done.value() != 0);
  done.wait();

  std::vector<float> got_sum(extent), got_copy(extent);
  tensors.download(sum, got_sum.data(), extent);
  tensors.download(copied, got_copy.data(), extent);
  CHECK(std::memcmp(host_a.data(), got_copy.data(), extent * sizeof(float)) == 0);
  for (size_t i = 0; i < host_a.size(); ++i) {
    const uint32_t input_bits = [&] {
      uint32_t bits = 0;
      std::memcpy(&bits, &host_a[i], sizeof(bits));
      return bits;
    }();
    if ((input_bits & 0x7f800000u) == 0x7f800000u &&
        (input_bits & 0x007fffffu) != 0) continue;
    if (!tensors.full_fp32_arithmetic_exactness() && (i == 2 || i == 3)) continue;
    const float expected = host_a[i] + host_b[i];
    uint32_t expected_bits = 0, actual_bits = 0;
    std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
    std::memcpy(&actual_bits, &got_sum[i], sizeof(actual_bits));
    CHECK_MSG(expected_bits == actual_bits,
              "fp32 add bit mismatch at %zu: %08x != %08x", i,
              expected_bits, actual_bits);
  }

  // An abandoned recording restores speculative access tracking and releases
  // its command slot; the same tensors remain immediately usable.
  {
    TensorBatch abandoned = tensors.begin_batch();
    abandoned.add(a, b, sum);
  }
  tensors.copy(a, copied);
  tensors.download(copied, got_copy.data(), extent);
  CHECK(std::memcmp(got_copy.data(), host_a.data(), extent * sizeof(float)) == 0);

  bool cross_context_rejected = false;
  try {
    DeviceTensor foreign = other.allocate(layout);
    TensorBatch invalid = tensors.begin_batch();
    invalid.copy(a, foreign);
  } catch (const std::invalid_argument&) {
    cross_context_rejected = true;
  }
  CHECK(cross_context_rejected);
  CHECK(a.view().context != other.allocate(layout).view().context);

  const uint64_t tail_extent = 8;
  const TensorLayout tail_layout = TensorLayout::contiguous(&tail_extent, 1);
  DeviceTensorView tail_view = a.view().slice(16, tail_layout, 16);
  CHECK(tail_view.byte_offset == 16);
  bool overflow_rejected = false;
  try {
    (void)tail_view.slice(std::numeric_limits<uint64_t>::max(), tail_layout, 4);
  } catch (const std::overflow_error&) {
    overflow_rejected = true;
  }
  CHECK(overflow_rejected);

  TensorWorkspace& workspace = tensors.workspace();
  const uint64_t before_workspace = tensors.reserved_bytes();
  workspace.reserve(256);
  WorkspaceSpan old_span = workspace.allocate(64, 32);
  CHECK(workspace.valid(old_span));
  TensorWorkspace foreign_workspace(device, 1024);
  foreign_workspace.reserve(256);
  CHECK(!foreign_workspace.valid(old_span));
  workspace.reset();
  workspace.reserve(512);
  CHECK(!workspace.valid(old_span));
  CHECK(workspace.generation() != old_span.generation);
  CHECK(tensors.reserved_bytes() > before_workspace);

  // Descriptor sets and pooled buffers reach a fixed high-water after warmup.
  tensors.add(a, b, sum);
  const uint64_t warm_reserved = tensors.reserved_bytes();
  const uint64_t warm_descriptors = tensors.descriptor_set_allocations();
  for (int repeat = 0; repeat < 8; ++repeat) {
    TensorBatch repeated = tensors.begin_batch();
    repeated.add(a, b, sum);
    repeated.copy(sum, copied);
    repeated.submit().wait();
    CHECK(tensors.reserved_bytes() == warm_reserved);
    CHECK(tensors.descriptor_set_allocations() == warm_descriptors);
  }

  // The advertised 32-op bound is exact. A rejected 33rd record causes the
  // whole scope to be abandoned; rollback leaves the next batch usable.
  bool thirty_third_rejected = false;
  {
    TensorBatch bounded = tensors.begin_batch();
    for (int i = 0; i < 32; ++i) bounded.add(a, b, sum);
    try {
      bounded.add(a, b, sum);
    } catch (const std::logic_error&) {
      thirty_third_rejected = true;
    }
    bool poisoned_submit_rejected = false;
    try {
      (void)bounded.submit();
    } catch (const std::logic_error&) {
      poisoned_submit_rejected = true;
    }
    CHECK(poisoned_submit_rejected);
  }
  CHECK(thirty_third_rejected);
  tensors.add(a, b, sum);

  // Two full graph chunks can be outstanding on separate bounded flight
  // slots. The second begin neither waits for nor resets the first slot.
  const uint64_t before_chunks = tensors.reserved_bytes();
  TensorBatch first_chunk = tensors.begin_batch();
  for (int i = 0; i < 32; ++i) first_chunk.add(a, b, sum);
  Submission first_token = first_chunk.submit();
  TensorBatch second_chunk = tensors.begin_batch();
  for (int i = 0; i < 32; ++i) second_chunk.add(a, b, copied);
  Submission second_token = second_chunk.submit();
  CHECK(second_token.value() > first_token.value());
  // A third begin applies oldest-slot timeline backpressure, then reuses that
  // slot and its descriptor cache rather than growing resources.
  TensorBatch third_chunk = tensors.begin_batch();
  third_chunk.add(a, b, tail);
  Submission third_token = third_chunk.submit();
  first_token.wait();
  second_token.wait();
  third_token.wait();
  CHECK(third_token.value() > second_token.value());
  CHECK(tensors.reserved_bytes() == before_chunks);
  const uint64_t descriptor_high_water = tensors.descriptor_set_allocations();
  TensorBatch reuse_first = tensors.begin_batch();
  for (int i = 0; i < 32; ++i) reuse_first.add(a, b, sum);
  Submission reuse_first_token = reuse_first.submit();
  TensorBatch reuse_second = tensors.begin_batch();
  for (int i = 0; i < 32; ++i) reuse_second.add(a, b, copied);
  Submission reuse_second_token = reuse_second.submit();
  reuse_first_token.wait();
  reuse_second_token.wait();
  CHECK(tensors.descriptor_set_allocations() == descriptor_high_water);

  // Numeric identities never recycle even after owners and native handles are
  // destroyed. This covers both tensor contexts and scratch workspaces.
  uintptr_t old_tensor_context = 0;
  {
    TensorContext temporary(device);
    DeviceTensor old = temporary.allocate(layout);
    old_tensor_context = old.view().context;
  }
  TensorContext replacement_context(device);
  CHECK(replacement_context.allocate(layout).view().context != old_tensor_context);
  uintptr_t old_workspace_context = 0;
  {
    TensorWorkspace old_workspace(device, 1024);
    old_workspace.reserve(64);
    old_workspace_context = old_workspace.allocate(16, 4).context;
  }
  TensorWorkspace replacement_workspace(device, 1024);
  replacement_workspace.reserve(64);
  CHECK(replacement_workspace.allocate(16, 4).context != old_workspace_context);

  // Submitted jobs own their tensors and Vulkan resources through completion,
  // even when every caller wrapper and the TensorContext are dropped.
  Submission retained;
  {
    TensorContext temporary(device);
    DeviceTensor left = temporary.allocate(layout);
    DeviceTensor right = temporary.allocate(layout);
    DeviceTensor output = temporary.allocate(layout);
    temporary.upload(left, host_a.data(), extent);
    temporary.upload(right, host_b.data(), extent);
    TensorBatch live = temporary.begin_batch();
    live.add(left, right, output);
    retained = live.submit();
  }
  retained.wait();

  TensorWorkspace movable_workspace(device, 1024);
  TensorWorkspace moved_workspace = std::move(movable_workspace);
  CHECK(moved_workspace.capacity() == 0);
  CHECK(movable_workspace.capacity() == 0);
  CHECK(movable_workspace.used() == 0);
  CHECK(movable_workspace.generation() == 0);
  CHECK(movable_workspace.reserved_bytes() == 0);
  movable_workspace.reset();
  bool moved_workspace_rejected = false;
  try {
    movable_workspace.reserve(16);
  } catch (const std::logic_error&) {
    moved_workspace_rejected = true;
  }
  CHECK(moved_workspace_rejected);

  TensorContext movable_context(device);
  TensorContext moved_context = std::move(movable_context);
  CHECK(moved_context.descriptor_set_allocations() == 0);
  CHECK(movable_context.reserved_bytes() == 0);
  CHECK(!movable_context.full_fp32_arithmetic_exactness());
  bool moved_context_rejected = false;
  try {
    (void)movable_context.allocate(layout);
  } catch (const std::logic_error&) {
    moved_context_rejected = true;
  }
  CHECK(moved_context_rejected);
}

VIDFAB_TEST(vulkan_tensor_layout_and_conversion_ops) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore ||
      !physical.front().info().fp32_signed_zero_inf_nan_preserve ||
      !physical.front().info().fp32_rounding_rte) return;
  DeviceOptions options; options.enable_timeline_semaphore = true;
  Device device = physical.front().create_device(options);
  TensorContext tensors(device);

  const uint64_t shape_data[] = {3, 5};
  const uint64_t transposed_data[] = {5, 3};
  const TensorLayout shape = TensorLayout::contiguous(shape_data, 2);
  const TensorLayout transposed_shape = TensorLayout::contiguous(transposed_data, 2);
  TensorLayout padded_shape = shape;
  padded_shape.stride[0] = 6;
  bool noncontiguous_rejected = false;
  try {
    (void)tensors.allocate(padded_shape);
  } catch (const std::invalid_argument&) {
    noncontiguous_rejected = true;
  }
  CHECK(noncontiguous_rejected);
  DeviceTensor input = tensors.allocate(shape);
  DeviceTensor bf16 = tensors.allocate(shape, ScalarType::kBFloat16);
  DeviceTensor fp16 = tensors.allocate(shape, ScalarType::kFloat16);
  DeviceTensor bf16_back = tensors.allocate(shape);
  DeviceTensor fp16_back = tensors.allocate(shape);
  DeviceTensor transposed = tensors.allocate(transposed_shape);
  DeviceTensor biased = tensors.allocate(shape);
  const uint64_t bias_extent = 5;
  DeviceTensor bias = tensors.allocate(TensorLayout::contiguous(&bias_extent, 1));

  const uint32_t patterns[] = {
      0x00000000u, 0x80000000u, 0x3f800000u, 0xc0000000u, 0x477fe000u,
      0x7f800000u, 0xff800000u, 0x33800000u, 0x33800001u, 0x00000001u,
      0x7fc12345u, 0x7fa54321u, 0x38800000u, 0x3f000000u, 0xbf000000u};
  std::vector<float> host(std::size(patterns));
  std::memcpy(host.data(), patterns, sizeof(patterns));
  const float host_bias[] = {1.0f, -1.0f, 0.25f, -0.25f, 2.0f};
  tensors.upload(input, host.data(), host.size());
  tensors.upload(bias, host_bias, std::size(host_bias));

  TensorBatch batch = tensors.begin_batch();
  batch.convert(input, bf16);
  batch.convert(bf16, bf16_back);
  batch.convert(input, fp16);
  batch.convert(fp16, fp16_back);
  batch.transpose_2d(input, transposed);
  batch.add_bias(input, bias, biased);
  Submission done = batch.submit();
  done.wait();

  std::vector<uint16_t> got_bf16(host.size()), got_fp16(host.size());
  std::vector<float> got_bf16_back(host.size()), got_fp16_back(host.size());
  std::vector<float> got_transposed(host.size()), got_biased(host.size());
  tensors.download_bytes(bf16, got_bf16.data(), got_bf16.size() * sizeof(uint16_t));
  tensors.download_bytes(fp16, got_fp16.data(), got_fp16.size() * sizeof(uint16_t));
  tensors.download(bf16_back, got_bf16_back.data(), got_bf16_back.size());
  tensors.download(fp16_back, got_fp16_back.data(), got_fp16_back.size());
  tensors.download(transposed, got_transposed.data(), got_transposed.size());
  tensors.download(biased, got_biased.data(), got_biased.size());
  for (size_t i = 0; i < host.size(); ++i) {
    CHECK(got_bf16[i] == reference_bf16(host[i]));
    CHECK(float_bits(got_bf16_back[i]) == (static_cast<uint32_t>(got_bf16[i]) << 16));
  }
  const uint16_t expected_half[] = {
      0x0000u, 0x8000u, 0x3c00u, 0xc000u, 0x7bffu, 0x7c00u, 0xfc00u,
      0x0001u, 0x0001u, 0x0000u, 0x7fffu, 0x7fffu, 0x0400u, 0x3800u, 0xb800u};
  for (size_t i = 0; i < host.size(); ++i) {
    CHECK_MSG(got_fp16[i] == expected_half[i], "fp16 mismatch %zu: %04x != %04x",
              i, got_fp16[i], expected_half[i]);
  }
  for (size_t row = 0; row < 3; ++row) {
    for (size_t col = 0; col < 5; ++col) {
      CHECK(float_bits(got_transposed[col * 3 + row]) == float_bits(host[row * 5 + col]));
      if (std::isfinite(host[row * 5 + col])) {
        CHECK(float_bits(got_biased[row * 5 + col]) ==
              float_bits(host[row * 5 + col] + host_bias[col]));
      }
    }
  }

  const uint64_t matrix_shape_data[] = {3, 4};
  const uint64_t selected_shape_data[] = {2, 4};
  const TensorLayout matrix_shape = TensorLayout::contiguous(matrix_shape_data, 2);
  const TensorLayout selected_shape = TensorLayout::contiguous(selected_shape_data, 2);
  const uint64_t index_count = 2;
  DeviceTensor matrix = tensors.allocate(matrix_shape);
  DeviceTensor indices = tensors.allocate(TensorLayout::contiguous(&index_count, 1),
                                          ScalarType::kInt32);
  DeviceTensor gathered = tensors.allocate(selected_shape);
  DeviceTensor scattered = tensors.allocate(matrix_shape);
  std::vector<float> matrix_host(12), scatter_initial(12, -99.0f);
  for (size_t i = 0; i < matrix_host.size(); ++i) matrix_host[i] = static_cast<float>(i + 1);
  const int32_t host_indices[] = {2, 0};
  tensors.upload(matrix, matrix_host.data(), matrix_host.size());
  tensors.upload_bytes(indices, host_indices, sizeof(host_indices));
  tensors.upload(scattered, scatter_initial.data(), scatter_initial.size());
  TensorBatch indexed = tensors.begin_batch();
  indexed.gather_rows(matrix, indices, gathered);
  indexed.scatter_rows(gathered, indices, scattered);
  indexed.submit().wait();
  std::vector<float> gathered_host(8), scattered_host(12);
  tensors.download(gathered, gathered_host.data(), gathered_host.size());
  tensors.download(scattered, scattered_host.data(), scattered_host.size());
  for (size_t col = 0; col < 4; ++col) {
    CHECK(gathered_host[col] == matrix_host[8 + col]);
    CHECK(gathered_host[4 + col] == matrix_host[col]);
    CHECK(scattered_host[8 + col] == matrix_host[8 + col]);
    CHECK(scattered_host[col] == matrix_host[col]);
    CHECK(scattered_host[4 + col] == -99.0f);
  }

  const uint64_t heads_elements = 24;
  DeviceTensor heads_input = tensors.allocate(TensorLayout::contiguous(&heads_elements, 1));
  DeviceTensor token_bf16 = tensors.allocate(TensorLayout::contiguous(&heads_elements, 1),
                                             ScalarType::kBFloat16);
  std::vector<float> heads_host(heads_elements);
  for (size_t i = 0; i < heads_host.size(); ++i) heads_host[i] = static_cast<float>(i) / 8.0f;
  tensors.upload(heads_input, heads_host.data(), heads_host.size());
  TensorBatch layout_batch = tensors.begin_batch();
  layout_batch.heads_to_tokens_bf16(heads_input, token_bf16, 2, 3, 4);
  layout_batch.submit().wait();
  std::vector<uint16_t> token_host(heads_elements);
  tensors.download_bytes(token_bf16, token_host.data(), token_host.size() * 2);
  for (uint32_t s = 0; s < 3; ++s) for (uint32_t h = 0; h < 2; ++h)
    for (uint32_t d = 0; d < 4; ++d) {
      const size_t out = (s * 2 + h) * 4 + d;
      const size_t in = (h * 3 + s) * 4 + d;
      CHECK(token_host[out] == reference_bf16(heads_host[in]));
    }

  const uint64_t depth_elements = 8;
  DeviceTensor depth_in = tensors.allocate(TensorLayout::contiguous(&depth_elements, 1));
  DeviceTensor depth_out = tensors.allocate(TensorLayout::contiguous(&depth_elements, 1));
  const float depth_values[] = {0, 1, 2, 3, 4, 5, 6, 7};
  tensors.upload(depth_in, depth_values, depth_elements);
  TensorBatch depth_batch = tensors.begin_batch();
  depth_batch.depth_to_space(depth_in, depth_out, 1, 1, 2, 1, 1, 2);
  depth_batch.submit().wait();
  std::vector<float> depth_result(depth_elements);
  tensors.download(depth_out, depth_result.data(), depth_elements);
  const float depth_expected[] = {0, 1, 4, 5, 2, 3, 6, 7};
  CHECK(std::memcmp(depth_result.data(), depth_expected, sizeof(depth_expected)) == 0);

  // Deterministic validation happens before access state or command recording,
  // so a caller may catch it and still submit the earlier valid operation.
  bool alias_rejected = false;
  TensorBatch recoverable = tensors.begin_batch();
  recoverable.convert(input, bf16);
  try {
    recoverable.transpose_2d(input, input);
  } catch (const std::invalid_argument&) {
    alias_rejected = true;
  }
  CHECK(alias_rejected);
  recoverable.submit().wait();

  TensorContext other(device);
  DeviceTensor foreign = other.allocate(shape);
  bool context_rejected = false;
  try {
    TensorBatch invalid = tensors.begin_batch();
    invalid.convert(foreign, bf16);
  } catch (const std::invalid_argument&) {
    context_rejected = true;
  }
  CHECK(context_rejected);

  bool overflow_rejected = false;
  try {
    TensorBatch invalid = tensors.begin_batch();
    invalid.heads_to_tokens_bf16(input, bf16,
                                 std::numeric_limits<uint32_t>::max(),
                                 std::numeric_limits<uint32_t>::max(),
                                 std::numeric_limits<uint32_t>::max());
  } catch (const std::overflow_error&) {
    overflow_rejected = true;
  }
  CHECK(overflow_rejected);

  // Invalid trusted device indices are still memory-safe: gather writes zero
  // and scatter preserves the destination row. Producers remain responsible
  // for satisfying the documented range/uniqueness contract.
  const int32_t unsafe_indices[] = {2, -1};
  tensors.upload_bytes(indices, unsafe_indices, sizeof(unsafe_indices));
  tensors.upload(scattered, scatter_initial.data(), scatter_initial.size());
  TensorBatch safe_indexing = tensors.begin_batch();
  safe_indexing.gather_rows(matrix, indices, gathered);
  safe_indexing.scatter_rows(gathered, indices, scattered);
  safe_indexing.submit().wait();
  tensors.download(gathered, gathered_host.data(), gathered_host.size());
  tensors.download(scattered, scattered_host.data(), scattered_host.size());
  for (size_t col = 0; col < 4; ++col) {
    CHECK(gathered_host[col] == matrix_host[8 + col]);
    CHECK(float_bits(gathered_host[4 + col]) == 0u);
    CHECK(scattered_host[8 + col] == matrix_host[8 + col]);
    CHECK(scattered_host[col] == -99.0f);
    CHECK(scattered_host[4 + col] == -99.0f);
  }

  // Warm both bounded flight slots, then prove that repeated conversion and
  // layout batches reuse descriptor sets and pooled allocations exactly.
  for (int warm = 0; warm < 2; ++warm) {
    TensorBatch repeated = tensors.begin_batch();
    repeated.convert(input, bf16);
    repeated.convert(bf16, bf16_back);
    repeated.transpose_2d(input, transposed);
    repeated.add_bias(input, bias, biased);
    repeated.submit().wait();
  }
  const uint64_t warm_reserved = tensors.reserved_bytes();
  const uint64_t warm_descriptors = tensors.descriptor_set_allocations();
  for (int repeat = 0; repeat < 8; ++repeat) {
    TensorBatch repeated = tensors.begin_batch();
    repeated.convert(input, bf16);
    repeated.convert(bf16, bf16_back);
    repeated.transpose_2d(input, transposed);
    repeated.add_bias(input, bias, biased);
    repeated.submit().wait();
    CHECK(tensors.reserved_bytes() == warm_reserved);
    CHECK(tensors.descriptor_set_allocations() == warm_descriptors);
  }
}

VIDFAB_TEST(vulkan_tensor_exact_vae_norms) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  const DeviceInfo& info = physical.front().info();
  CHECK(!detail::norm_dispatch_fits(0, info.max_compute_workgroup_count[0]));
  CHECK(detail::norm_dispatch_fits(info.max_compute_workgroup_count[0],
                                   info.max_compute_workgroup_count[0]));
  CHECK(!detail::norm_dispatch_fits(
      static_cast<uint64_t>(info.max_compute_workgroup_count[0]) + 1,
      info.max_compute_workgroup_count[0]));
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  Device disabled_device = physical.front().create_device(options);
  TensorContext disabled_tensors(disabled_device);
  CHECK(!disabled_tensors.exact_normalization());
  CHECK(!disabled_tensors.exact_vae_pointwise());
  CHECK(disabled_tensors.exact_fp32_vae_normalization() ==
        disabled_tensors.exact_normalization());
  bool disabled_rejected = false;
  try { disabled_tensors.require_exact_normalization(); }
  catch (const std::runtime_error&) { disabled_rejected = true; }
  CHECK(disabled_rejected);
  options.enable_shader_int64 = info.shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext tensors(device);
  const bool expected_capability = detail::known_exact_vae_norm_device(
                                       info.vendor_id, info.device_id,
                                       info.driver_version) &&
                                   info.fp32_signed_zero_inf_nan_preserve &&
                                   info.shader_int64;
  CHECK(!detail::known_exact_vae_norm_device(0x10deu, 0x2b85u, 0x98960001u));
  CHECK(!detail::known_exact_vae_norm_device(0x10deu, 0x2b86u, 0x98960000u));
  const bool expected_pointwise = detail::known_exact_vae_pointwise_device(
                                      info.vendor_id, info.device_id,
                                      info.driver_version) &&
                                  info.fp32_signed_zero_inf_nan_preserve &&
                                  info.fp32_rounding_rte && info.shader_int64;
  CHECK(!detail::known_exact_vae_pointwise_device(
      0x10deu, 0x2b85u, 0x98960001u));
  CHECK(!detail::known_exact_vae_pointwise_device(
      0x10deu, 0x2b86u, 0x98960000u));
  CHECK(tensors.exact_vae_pointwise() == expected_pointwise);
  CHECK(tensors.exact_normalization() == expected_capability);
  CHECK(tensors.exact_fp32_vae_normalization() == tensors.exact_normalization());
  if (!expected_capability) {
    bool rejected = false;
    try { tensors.require_exact_normalization(); }
    catch (const std::runtime_error&) { rejected = true; }
    CHECK(rejected);
    return;
  }
  tensors.require_exact_normalization();
  tensors.require_exact_fp32_vae_normalization();

  const uint64_t extents[] = {2, 8};
  const uint64_t features = 8;
  const TensorLayout matrix = TensorLayout::contiguous(extents, 2);
  const TensorLayout vector = TensorLayout::contiguous(&features, 1);
  DeviceTensor input = tensors.allocate(matrix);
  DeviceTensor weight = tensors.allocate(vector);
  DeviceTensor bias = tensors.allocate(vector);
  DeviceTensor rms = tensors.allocate(matrix);
  DeviceTensor layer = tensors.allocate(matrix);
  std::vector<float> values(16), weights(8, 1.0f), biases(8, 0.0f);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = (i & 1) ? -0.5f : 0.5f;
  tensors.upload(input, values.data(), values.size());
  tensors.upload(weight, weights.data(), weights.size());
  tensors.upload(bias, biases.data(), biases.size());
  TensorBatch exact = tensors.begin_batch();
  // mean(x^2)+eps = .25+.75 = 1 and mean(x)=0, so both expected
  // results are exactly the input without relying on host sqrt behavior.
  exact.rms_norm(input, weight, rms, 0.75f);
  exact.layer_norm(input, weight, bias, layer, 0.75f);
  exact.submit().wait();
  std::vector<float> rms_host(values.size()), layer_host(values.size());
  tensors.download(rms, rms_host.data(), rms_host.size());
  tensors.download(layer, layer_host.data(), layer_host.size());
  CHECK(std::memcmp(rms_host.data(), values.data(), values.size() * sizeof(float)) == 0);
  CHECK(std::memcmp(layer_host.data(), values.data(), values.size() * sizeof(float)) == 0);

  // RMSNorm preserves signed zero when sqrt(eps) is exactly one.
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = (i & 1) ? -0.0f : 0.0f;
  tensors.upload(input, values.data(), values.size());
  TensorBatch signed_zero = tensors.begin_batch();
  signed_zero.rms_norm(input, weight, rms, 1.0f);
  signed_zero.submit().wait();
  tensors.download(rms, rms_host.data(), rms_host.size());
  CHECK(std::memcmp(rms_host.data(), values.data(), values.size() * sizeof(float)) == 0);

  // A zero-variance LayerNorm row maps exactly to its affine bias.
  std::fill(values.begin(), values.end(), 2.0f);
  for (size_t i = 0; i < biases.size(); ++i)
    biases[i] = static_cast<float>(static_cast<int>(i) - 4) * 0.25f;
  tensors.upload(input, values.data(), values.size());
  tensors.upload(bias, biases.data(), biases.size());
  TensorBatch constant = tensors.begin_batch();
  constant.layer_norm(input, weight, bias, layer, 1.0e-6f);
  constant.submit().wait();
  tensors.download(layer, layer_host.data(), layer_host.size());
  for (size_t row = 0; row < 2; ++row) {
    CHECK(std::memcmp(layer_host.data() + row * 8, biases.data(),
                      biases.size() * sizeof(float)) == 0);
  }

  bool subnormal_epsilon_rejected = false;
  try {
    TensorBatch invalid = tensors.begin_batch();
    invalid.rms_norm(input, weight, rms, std::numeric_limits<float>::denorm_min());
  } catch (const std::invalid_argument&) {
    subnormal_epsilon_rejected = true;
  }
  CHECK(subnormal_epsilon_rejected);
}

VIDFAB_TEST(vulkan_yuv420_output) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!Instance::available()) return;
  Instance probe = Instance::create();
  const auto devices = probe.enumerate_devices();
  if (devices.empty() || !devices.front().info().timeline_semaphore) return;

  Yuv420Converter converter;
  bool unavailable_device_rejected = false;
  try {
    Yuv420Converter unavailable(std::numeric_limits<uint32_t>::max());
  } catch (const std::runtime_error& error) {
    unavailable_device_rejected = std::string(error.what()).find("unavailable") != std::string::npos;
  }
  CHECK(unavailable_device_rejected);
  struct Extent { int width; int height; };
  const Extent extents[] = {{2, 2}, {10, 6}, {128, 66}, {1280, 768}};
  uint64_t previous_high_water = 0;
  for (const Extent extent : extents) {
    const size_t pixels = static_cast<size_t>(extent.width) * extent.height;
    std::vector<float> r(pixels), g(pixels), b(pixels);
    for (size_t i = 0; i < pixels; ++i) {
      r[i] = static_cast<float>((i * 17) % 113) / 97.0f - 0.08f;
      g[i] = static_cast<float>((i * 29 + 3) % 127) / 109.0f;
      b[i] = static_cast<float>((i * 43 + 11) % 139) / 101.0f - 0.12f;
    }
    size_t boundary_block = 0;
    for (int by = 0; by < extent.height && boundary_block < 96; by += 2) {
      for (int bx = 0; bx < extent.width && boundary_block < 96; bx += 2, ++boundary_block) {
        const float boundary =
            (static_cast<float>(96 + (boundary_block % 64)) + 0.5f - 128.0f) / 112.0f;
        const int direction = static_cast<int>(boundary_block % 3) - 1;
        const float value = direction < 0 ? std::nextafter(boundary, -INFINITY)
                            : direction > 0 ? std::nextafter(boundary, INFINITY)
                                            : boundary;
        for (int dy = 0; dy < 2; ++dy) {
          for (int dx = 0; dx < 2; ++dx) {
            const size_t i = static_cast<size_t>(by + dy) * extent.width + bx + dx;
            r[i] = (boundary_block & 1) != 0 ? value : 0.0f;
            g[i] = 0.0f;
            b[i] = (boundary_block & 1) == 0 ? value : 0.0f;
          }
        }
      }
    }
    // Keep luma half-step fixtures at the opposite end of larger frames so
    // the chroma-boundary blocks above cannot overwrite them.
    if (pixels >= 512) {
      for (size_t fixture = 0; fixture < 192; ++fixture) {
        const size_t i = pixels - 192 + fixture;
        const float boundary =
            (static_cast<float>(16 + (fixture % 220)) + 0.5f - 16.0f) / 219.0f;
        const int direction = static_cast<int>(fixture % 3) - 1;
        const float value = direction < 0 ? std::nextafter(boundary, -INFINITY)
                            : direction > 0 ? std::nextafter(boundary, INFINITY)
                                            : boundary;
        r[i] = value;
        g[i] = value;
        b[i] = value;
      }
    }
    const int ys = extent.width + 13;
    const int cs = extent.width / 2 + 7;
    std::vector<uint8_t> cpu_y(static_cast<size_t>(ys) * extent.height, 0xa5);
    std::vector<uint8_t> cpu_u(static_cast<size_t>(cs) * (extent.height / 2), 0xa5);
    std::vector<uint8_t> cpu_v(cpu_u.size(), 0xa5);
    std::vector<uint8_t> vk_y(cpu_y.size(), 0xa5), vk_u(cpu_u.size(), 0xa5), vk_v(cpu_v.size(), 0xa5);
    video::rgb_frame_to_yuv420(r.data(), g.data(), b.data(), extent.height, extent.width,
                               cpu_y.data(), ys, cpu_u.data(), cs, cpu_v.data(), cs);
    converter.convert(r.data(), g.data(), b.data(), extent.height, extent.width,
                      vk_y.data(), ys, vk_u.data(), cs, vk_v.data(), cs);

    int differences = 0;
    int max_difference = 0;
    auto compare_plane = [&](const std::vector<uint8_t>& expected,
                             const std::vector<uint8_t>& actual, int rows, int columns,
                             int stride) {
      for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < columns; ++x) {
          const int delta = std::abs(static_cast<int>(expected[static_cast<size_t>(y) * stride + x]) -
                                     static_cast<int>(actual[static_cast<size_t>(y) * stride + x]));
          if (delta != 0) ++differences;
          max_difference = std::max(max_difference, delta);
        }
        for (int x = columns; x < stride; ++x) {
          CHECK(actual[static_cast<size_t>(y) * stride + x] == 0xa5);
        }
      }
    };
    compare_plane(cpu_y, vk_y, extent.height, extent.width, ys);
    compare_plane(cpu_u, vk_u, extent.height / 2, extent.width / 2, cs);
    compare_plane(cpu_v, vk_v, extent.height / 2, extent.width / 2, cs);
    CHECK_MSG(differences == 0,
              "Vulkan YUV differs by %d (changed samples %d) at %dx%d",
              max_difference, differences, extent.width, extent.height);
    CHECK(converter.capacity_pixels() >= pixels);
    CHECK(converter.high_water_bytes() >= previous_high_water);
    previous_high_water = converter.high_water_bytes();

    const uint64_t reserved = converter.reserved_bytes();
    converter.convert(r.data(), g.data(), b.data(), extent.height, extent.width,
                      vk_y.data(), ys, vk_u.data(), cs, vk_v.data(), cs);
    CHECK(converter.reserved_bytes() == reserved);
  }
  CHECK_MSG(converter.high_water_bytes() <= (48ull << 20),
            "Vulkan YUV persistent high-water is %llu bytes",
            static_cast<unsigned long long>(converter.high_water_bytes()));

  const uint64_t capacity_before_reject = converter.capacity_pixels();
  const uint64_t reserved_before_reject = converter.reserved_bytes();
  bool index_overflow_rejected = false;
  uint8_t reject_byte = 0;
  float reject_sample = 0.0f;
  try {
    converter.convert(&reject_sample, &reject_sample, &reject_sample, 32768, 65536,
                      &reject_byte, 65536, &reject_byte, 32768, &reject_byte, 32768);
  } catch (const std::overflow_error&) {
    index_overflow_rejected = true;
  }
  CHECK(index_overflow_rejected);
  CHECK(converter.capacity_pixels() == capacity_before_reject);
  CHECK(converter.reserved_bytes() == reserved_before_reject);

  bool odd_rejected = false;
  uint8_t byte = 0;
  float sample = 0;
  try {
    converter.convert(&sample, &sample, &sample, 2, 3, &byte, 3, &byte, 1, &byte, 1);
  } catch (const std::invalid_argument&) {
    odd_rejected = true;
  }
  CHECK(odd_rejected);

  bool null_rejected = false;
  try {
    converter.convert(nullptr, &sample, &sample, 2, 2, &byte, 2, &byte, 1, &byte, 1);
  } catch (const std::invalid_argument&) {
    null_rejected = true;
  }
  CHECK(null_rejected);

  bool stride_rejected = false;
  try {
    converter.convert(&sample, &sample, &sample, 2, 2, &byte, 1, &byte, 1, &byte, 1);
  } catch (const std::invalid_argument&) {
    stride_rejected = true;
  }
  CHECK(stride_rejected);

  const int frames = 3;
  const int y4m_width = 10;
  const int y4m_height = 6;
  const size_t frame_pixels = static_cast<size_t>(y4m_width) * y4m_height;
  PixelBuffer clip(static_cast<size_t>(3) * frames * frame_pixels);
  const size_t plane = static_cast<size_t>(frames) * frame_pixels;
  for (int frame = 0; frame < frames; ++frame) {
    for (size_t i = 0; i < frame_pixels; ++i) {
      const size_t offset = static_cast<size_t>(frame) * frame_pixels + i;
      clip[offset] = static_cast<float>((i * 11 + frame * 7) % 101) / 100.0f;
      clip[plane + offset] = static_cast<float>((i * 23 + frame * 5) % 103) / 102.0f;
      clip[2 * plane + offset] = static_cast<float>((i * 37 + frame * 3) % 107) / 106.0f;
    }
  }
  const auto cpu_path = std::filesystem::temp_directory_path() / "vidfab_vulkan_cpu.y4m";
  const auto vk_path = std::filesystem::temp_directory_path() / "vidfab_vulkan_output.y4m";
  std::filesystem::remove(cpu_path);
  std::filesystem::remove(vk_path);
  video::write_y4m(cpu_path.string(), clip, frames, y4m_height, y4m_width);
  const uint64_t before_y4m = converter.reserved_bytes();
  video::write_y4m(vk_path.string(), clip, frames, y4m_height, y4m_width, {}, &converter);
  CHECK(read_bytes(cpu_path) == read_bytes(vk_path));
  CHECK(converter.reserved_bytes() == before_y4m);
  std::filesystem::remove(cpu_path);
  std::filesystem::remove(vk_path);
}

}  // namespace

VIDFAB_TEST(vulkan_video_vae_decoder_contract) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);

  vae::ViTConfig shipped;
  bool shipped_rejected = false;
  try {
    (void)VideoVaeDecoder::create(device, shipped);
  } catch (const std::invalid_argument&) {
    shipped_rejected = true;
  }
  CHECK(shipped_rejected);

  vae::ViTConfig too_many;
  too_many.transformer_mode = vae::ViTTransformerMode::kExact;
  too_many.num_layers = 205;  // 15 + 20*205 > the 4096-op transaction.
  bool capacity_rejected = false;
  try {
    (void)VideoVaeDecoder::create(device, too_many);
  } catch (const std::invalid_argument&) {
    capacity_rejected = true;
  }
  CHECK(capacity_rejected);

  vae::ViTConfig exact;
  exact.transformer_mode = vae::ViTTransformerMode::kExact;
  VideoVaeDecoder decoder;
  try {
    decoder = VideoVaeDecoder::create(device, exact);
  } catch (const std::runtime_error&) {
    // The exact arithmetic allow-list is intentionally narrower than Vulkan
    // availability. The two validation checks above are device-independent.
    return;
  }
  CHECK(decoder.operators_per_document() == 735);

  std::vector<float> normalized(size_t(exact.in_channels) * 7, 0.0f);
  bool unloaded_decode_rejected = false;
  try {
    (void)decoder.decode(normalized.data(), 7, 1, 1,
                         vae::default_video_latents_mean(),
                         vae::default_video_latents_std());
  } catch (const std::logic_error&) {
    unloaded_decode_rejected = true;
  }
  CHECK(unloaded_decode_rejected);

  vae::ViTConfig one_layer = exact;
  one_layer.num_layers = 1;
  VideoVaeDecoder minimum = VideoVaeDecoder::create(device, one_layer);
  CHECK(minimum.operators_per_document() == 35);
  vae::ViTConfig boundary = exact;
  boundary.num_layers = 204;
  VideoVaeDecoder maximum = VideoVaeDecoder::create(device, boundary);
  CHECK(maximum.operators_per_document() == 4095);
}

VIDFAB_TEST(vulkan_audio_vae_decoder_cuda_off_contract) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore ||
      !physical.front().info().shader_int64) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  Device device = physical.front().create_device(options);

  AudioDecoder decoder = AudioDecoder::create(device);
  CHECK(decoder.recorded_operators() == 497u);
  CHECK(decoder.weight_bytes() == 0u);
  CHECK(decoder.peak_device_bytes() == 0u);
  std::vector<float> latent(64, 0.0f);
  bool unloaded_decode_rejected = false;
  try {
    (void)decoder.decode(latent.data(), 1);
  } catch (const std::logic_error&) {
    unloaded_decode_rejected = true;
  }
  CHECK(unloaded_decode_rejected);
  decoder.unload();
  CHECK(decoder.weight_bytes() == 0u);

  if (!std::getenv("VIDFAB_AUDIO_DECODER_REAL")) return;
  const char* configured_path = std::getenv("VIDFAB_AUDIO_VAE_PATH");
  const std::filesystem::path checkpoint_path = configured_path != nullptr
      ? configured_path
      : "weights/vae/minimax_h3_audio_vae_fp32.safetensors";
  if (!std::filesystem::exists(checkpoint_path)) return;
  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
  CHECK(checkpoint.file_size() == 605254808u);
  CHECK(checkpoint.tensor_count() == 917u);
  decoder.load(checkpoint);
  CHECK(decoder.recorded_operators() == 497u);
  CHECK(decoder.weight_bytes() == 259672032u);
  constexpr int latent_length = 3;
  std::vector<float> real_latent(size_t(2) * 32 * latent_length);
  for (size_t i = 0; i < real_latent.size(); ++i)
    real_latent[i] = float(int((i * 67) % 607) - 303) / 128.0f;
  const vae::DecodedAudio first = decoder.decode(real_latent.data(),
                                                  latent_length);
  const uint64_t first_digest = fnv64_floats(first.samples);
  CHECK(first_digest == 0x528f17a83d5ef7eeull);
  const uint64_t stable_reserved = decoder.allocator_reserved_bytes();
  const uint64_t stable_descriptors = decoder.descriptor_set_allocations();
  const vae::DecodedAudio repeat = decoder.decode(real_latent.data(),
                                                   latent_length);
  CHECK(first.samples == repeat.samples);
  CHECK(fnv64_floats(repeat.samples) == first_digest);
  CHECK(decoder.allocator_reserved_bytes() == stable_reserved);
  CHECK(decoder.descriptor_set_allocations() == stable_descriptors);
  std::printf("  CUDA-off real Vulkan audio A3 FNV64 %016llx\n",
              static_cast<unsigned long long>(first_digest));
  decoder.unload();
  CHECK(decoder.weight_bytes() == 0u);
  CHECK(decoder.peak_device_bytes() == 0u);
  unloaded_decode_rejected = false;
  try {
    (void)decoder.decode(real_latent.data(), latent_length);
  } catch (const std::logic_error&) {
    unloaded_decode_rejected = true;
  }
  CHECK(unloaded_decode_rejected);
}

VIDFAB_TEST(vulkan_h3_loaded_stage_cuda_off_contract) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) return;
  const DeviceInfo& info = physical.front().info();
  if (!info.timeline_semaphore || !info.shader_int64 || !info.shader_float16 ||
      !info.storage_buffer_16bit ||
      !info.cooperative_matrix_bf16_f32_16x16x16) return;
  DeviceOptions device_options;
  device_options.enable_timeline_semaphore = true;
  device_options.enable_shader_int64 = true;
  device_options.enable_shader_float16 = true;
  device_options.enable_storage_buffer_16bit = true;
  device_options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(device_options);
  TensorContextOptions context_options;
  context_options.max_batch_operators = 64;
  TensorContext context(device, context_options);
  if (!context.exact_h3_attention() || !context.exact_vae_pointwise() ||
      !context.exact_fp32_vae_normalization()) return;

  H3BlockConfig config;
  config.sequence = 65;
  config.hidden = 128;
  config.heads = 1;
  config.head_dim = 128;
  config.ffn = 128;
  config.timesteps = 1;
  config.modalities = 3;
  config.adaln_rank = 8;
  auto values = [](size_t count, uint32_t salt, float scale) {
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i)
      result[i] = float(int((i * 37 + salt) % 127) - 63) * scale;
    return result;
  };
  auto metadata = [](const std::string& text) {
    std::vector<float> result((text.size() + 3) / 4, 0.0f);
    std::memcpy(result.data(), text.data(), text.size());
    return result;
  };
  auto fixture = [&](uint32_t corruption) {
    const int64_t h = config.hidden, inner = config.heads * config.head_dim;
    const int64_t f = config.ffn;
    const int64_t adaln = int64_t(config.modalities) * 6 * h;
    std::vector<TensorWrite> tensors{
      {"blocks.0.norm1.weight", {h}, std::vector<float>(h, 1.0f)},
      {"blocks.0.norm2.weight", {h}, std::vector<float>(h, 1.0f)},
      {"blocks.0.attn.q_norm.weight", {config.head_dim},
       std::vector<float>(config.head_dim, 1.0f)},
      {"blocks.0.attn.k_norm.weight", {config.head_dim},
       std::vector<float>(config.head_dim, 1.0f)},
      {"blocks.0.attn.qkv_proj.weight", {3 * inner, h},
       values(size_t(3 * inner * h), 3, 1.0f / 4096.0f)},
      {"blocks.0.attn.out_proj.weight", {h, inner},
       values(size_t(h * inner), 5, 1.0f / 4096.0f)},
      // Exactly one AWQ activation pre-scale fixture.
      {"blocks.0.attn.out_proj.pre_quant_scale", {inner},
       values(size_t(inner), 71, 1.0f / 256.0f)},
      {"blocks.0.mlp.fc1.weight", {2 * f, h},
       values(size_t(2 * f * h), 7, 1.0f / 4096.0f)},
      {"blocks.0.mlp.fc2.weight", {h, f},
       values(size_t(h * f), 11, 1.0f / 4096.0f)},
      // Exactly one regular-H4 ConvRot metadata fixture.
      {"blocks.0.mlp.fc2.comfy_quant",
       {static_cast<int64_t>(metadata(
          corruption == 2 ? "{bad fc2 metadata" :
          "{\"convrot\":true,\"convrot_groupsize\":16}").size())},
       metadata(corruption == 2 ? "{bad fc2 metadata" :
                "{\"convrot\":true,\"convrot_groupsize\":16}")},
      {"blocks.0.adaln_proj.linear.weight", {adaln, config.adaln_rank},
       values(size_t(adaln * config.adaln_rank), 13, 1.0f / 8192.0f)},
      {"blocks.0.adaln_proj.linear.bias",
       {corruption == 1 ? adaln - 1 : adaln},
       values(size_t(corruption == 1 ? adaln - 1 : adaln), 17, 1.0f / 64.0f)}};
    return tensors;
  };
  const auto base = std::filesystem::temp_directory_path() /
      ("vidfab_h3_cuda_off_" + std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(base);
  const auto valid_path = base / "vidfab_h3_cuda_off_valid.safetensors";
  const auto corrupt_path = base / "vidfab_h3_cuda_off_corrupt.safetensors";
  const auto corrupt_fc2_path =
      base / "vidfab_h3_cuda_off_corrupt_fc2.safetensors";
  write_safetensors(valid_path.string(), fixture(0));
  write_safetensors(corrupt_path.string(), fixture(1));
  write_safetensors(corrupt_fc2_path.string(), fixture(2));
  SafeTensors valid, corrupt, corrupt_fc2;
  valid.open(valid_path.string());
  corrupt.open(corrupt_path.string());
  corrupt_fc2.open(corrupt_fc2_path.string());

  auto graph_fixture = [&](bool corrupt_last) {
    std::vector<TensorWrite> all;
    for (uint32_t layer = 0; layer < 3; ++layer) {
      std::vector<TensorWrite> one = fixture(corrupt_last && layer == 2 ? 2 : 0);
      for (TensorWrite& tensor : one) {
        tensor.name.replace(0, std::strlen("blocks.0"),
                            "blocks." + std::to_string(layer));
        all.push_back(std::move(tensor));
      }
    }
    return all;
  };
  const auto graph_path = base / "vidfab_h3_cuda_off_graph.safetensors";
  const auto corrupt_graph_path =
      base / "vidfab_h3_cuda_off_corrupt_graph.safetensors";
  write_safetensors(graph_path.string(), graph_fixture(false));
  write_safetensors(corrupt_graph_path.string(), graph_fixture(true));
  SafeTensors graph_checkpoint, corrupt_graph;
  graph_checkpoint.open(graph_path.string());
  corrupt_graph.open(corrupt_graph_path.string());

  auto transformer_fixture = [&](uint32_t corruption) {
    std::vector<TensorWrite> all = fixture(0);
    std::vector<TensorWrite> refiner = fixture(0);
    for (TensorWrite& tensor : refiner) {
      if (tensor.name.find(".adaln_proj.") != std::string::npos) continue;
      tensor.name.replace(0, std::strlen("blocks.0"),
                          "token_refiner.blocks.0");
      all.push_back(std::move(tensor));
    }
    const int64_t h = config.hidden;
    constexpr int64_t text_dim = 16, video_dim = 4, audio_dim = 2;
    const int64_t final_adaln = 2 * h;
    all.insert(all.end(), {
      {"adaln_t_table", {1025, 8}, values(1025 * 8, 97, 1.0f / 4096.0f)},
      {"condition_proj.weight", {h, text_dim},
       values(size_t(h * text_dim), 101, 1.0f / 1024.0f),
       corruption == 2 ? DType::kF32 : DType::kBF16},
      {"condition_proj.bias", {h}, values(h, 103, 1.0f / 128.0f),
       DType::kBF16},
      {"video_patch_proj.weight", {h, video_dim},
       values(size_t(h * video_dim), 107, 1.0f / 1024.0f)},
      {"video_patch_proj.bias", {h}, values(h, 109, 1.0f / 128.0f)},
      {"audio_patch_proj.weight", {h, audio_dim},
       values(size_t(h * audio_dim), 113, 1.0f / 1024.0f)},
      {"audio_patch_proj.bias", {h}, values(h, 127, 1.0f / 128.0f)},
      {"token_refiner.final_norm.weight", {h}, std::vector<float>(h, 1.0f),
       DType::kBF16},
      {"final_layer.norm.weight", {h}, std::vector<float>(h, 1.0f),
       DType::kBF16},
      {"final_layer.adaln_proj.linear.weight",
       {final_adaln, config.adaln_rank},
       values(size_t(final_adaln * config.adaln_rank), 131, 1.0f / 8192.0f),
       DType::kF16},
      {"final_layer.adaln_proj.linear.bias", {final_adaln},
       values(final_adaln, 137, 1.0f / 128.0f), DType::kF16},
      {"final_layer.video_out.weight", {video_dim, h},
       values(size_t(video_dim * h), 139, 1.0f / 1024.0f)},
      {"final_layer.video_out.bias", {video_dim},
       values(video_dim, 149, 1.0f / 128.0f)},
      {"final_layer.audio_out.weight", {audio_dim, h},
       values(size_t(audio_dim * h), 151, 1.0f / 1024.0f)},
      {"final_layer.audio_out.bias",
       {corruption == 1 ? audio_dim - 1 : audio_dim},
       values(size_t(corruption == 1 ? audio_dim - 1 : audio_dim),
              157, 1.0f / 128.0f)}
    });
    const char* metadata_suffix = nullptr;
    if (corruption == 3) metadata_suffix = ".pre_quant_scale";
    if (corruption == 4) metadata_suffix = ".weight_scale";
    if (corruption == 5) metadata_suffix = ".input_scale";
    if (corruption == 6) metadata_suffix = ".comfy_quant";
    if (metadata_suffix) {
      all.push_back({std::string("final_layer.adaln_proj.linear") +
                         metadata_suffix,
                     {1}, {1.0f}});
    }
    return all;
  };
  const auto transformer_path =
      base / "vidfab_h3_cuda_off_transformer.safetensors";
  const auto corrupt_transformer_path =
      base / "vidfab_h3_cuda_off_corrupt_transformer.safetensors";
  const auto corrupt_transformer_dtype_path =
      base / "vidfab_h3_cuda_off_corrupt_transformer_dtype.safetensors";
  const std::array<std::filesystem::path, 4> corrupt_transformer_metadata_paths{
      base / "vidfab_h3_cuda_off_corrupt_transformer_prequant.safetensors",
      base / "vidfab_h3_cuda_off_corrupt_transformer_weightscale.safetensors",
      base / "vidfab_h3_cuda_off_corrupt_transformer_inputscale.safetensors",
      base / "vidfab_h3_cuda_off_corrupt_transformer_comfy.safetensors"};
  write_safetensors(transformer_path.string(), transformer_fixture(0));
  write_safetensors(corrupt_transformer_path.string(),
                    transformer_fixture(1));
  write_safetensors(corrupt_transformer_dtype_path.string(),
                    transformer_fixture(2));
  for (uint32_t i = 0; i < corrupt_transformer_metadata_paths.size(); ++i)
    write_safetensors(corrupt_transformer_metadata_paths[i].string(),
                      transformer_fixture(3 + i));
  SafeTensors transformer_checkpoint, corrupt_transformer,
      corrupt_transformer_dtype;
  std::array<SafeTensors, 4> corrupt_transformer_metadata;
  transformer_checkpoint.open(transformer_path.string());
  corrupt_transformer.open(corrupt_transformer_path.string());
  corrupt_transformer_dtype.open(corrupt_transformer_dtype_path.string());
  for (uint32_t i = 0; i < corrupt_transformer_metadata.size(); ++i)
    corrupt_transformer_metadata[i].open(
        corrupt_transformer_metadata_paths[i].string());

  ExactH3BlockStage first = ExactH3BlockStage::create(context, config);
  ExactH3BlockStage second = ExactH3BlockStage::create(context, config);
  first.load(valid, 0);
  second.load(valid, 0);
  CHECK(first.required_operators() == 25u);
  CHECK(second.required_operators() == 25u);
  ExactH3BlockScratch scratch = ExactH3BlockScratch::create(context, config);
  first.prepare(scratch);
  second.prepare(scratch);
  CHECK(scratch.reserved_bytes() != 0u);
  const uint64_t token_shape[] = {config.sequence, config.hidden};
  const uint64_t selector_shape[] = {config.sequence};
  const uint64_t code_shape[] = {config.timesteps, config.adaln_rank};
  const uint64_t rope_shape[] = {config.sequence, 96};
  DeviceTensor tokens = context.allocate(
      TensorLayout::contiguous(token_shape, 2), ScalarType::kBFloat16);
  DeviceTensor selectors = context.allocate(
      TensorLayout::contiguous(selector_shape, 1), ScalarType::kInt32);
  DeviceTensor code = context.allocate(TensorLayout::contiguous(code_shape, 2));
  DeviceTensor cosine = context.allocate(TensorLayout::contiguous(rope_shape, 2));
  DeviceTensor sine = context.allocate(TensorLayout::contiguous(rope_shape, 2));
  CHECK(static_cast<bool>(sine));
  std::vector<uint16_t> input(size_t(config.sequence) * config.hidden);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = f32_to_bf16(float(int(i % 47) - 23) / 32.0f);
  std::vector<int32_t> host_selectors(config.sequence);
  for (uint32_t i = 0; i < config.sequence; ++i)
    host_selectors[i] = static_cast<int32_t>(i % config.modalities);
  std::vector<float> host_code(config.adaln_rank);
  for (uint32_t i = 0; i < config.adaln_rank; ++i)
    host_code[i] = float(int(i) - 3) / 16.0f;
  std::vector<float> host_cos(size_t(config.sequence) * 96);
  std::vector<float> host_sin(host_cos.size());
  for (size_t i = 0; i < host_cos.size(); ++i) {
    const float angle = float((i * 13) % 101) / 997.0f;
    host_cos[i] = std::cos(angle);
    host_sin[i] = std::sin(angle);
  }
  context.upload_bytes(selectors, host_selectors.data(), host_selectors.size() * 4);
  context.upload(code, host_code.data(), host_code.size());
  context.upload(cosine, host_cos.data(), host_cos.size());
  context.upload(sine, host_sin.data(), host_sin.size());
  DeviceTensor dummy = context.allocate(
      TensorLayout::contiguous(token_shape, 2), ScalarType::kBFloat16);
  {
    TensorBatch exact_capacity = context.begin_batch();
    for (uint32_t i = first.required_operators(); i < 64; ++i)
      exact_capacity.copy(dummy, tokens);
    first.record(exact_capacity, tokens, selectors, code, cosine, sine, scratch);
    CHECK(exact_capacity.remaining_operator_capacity() == 0u);
    exact_capacity.submit().wait();
  }
  {
    TensorBatch short_capacity = context.begin_batch();
    for (uint32_t i = first.required_operators(); i <= 64; ++i)
      short_capacity.copy(dummy, tokens);
    bool rejected = false;
    try {
      first.record(short_capacity, tokens, selectors, code, cosine, sine, scratch);
    } catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected && short_capacity.remaining_operator_capacity() == 24u);
  }
  TensorLayout bad_token_layout = TensorLayout::contiguous(token_shape, 2);
  bad_token_layout.stride[0] = config.hidden + 1;
  TensorLayout bad_selector_layout = TensorLayout::contiguous(selector_shape, 1);
  bad_selector_layout.stride[0] = 2;
  TensorLayout bad_code_layout = TensorLayout::contiguous(code_shape, 2);
  bad_code_layout.stride[0] = config.adaln_rank + 1;
  const uint64_t token_backing_shape[] = {config.sequence + 1, config.hidden};
  const uint64_t selector_backing_shape[] = {config.sequence * 2};
  const uint64_t code_backing_shape[] = {config.timesteps,
                                         config.adaln_rank + 1};
  DeviceTensor bad_tokens = context.allocate(
      TensorLayout::contiguous(token_backing_shape, 2), ScalarType::kBFloat16);
  DeviceTensor bad_selectors = context.allocate(
      TensorLayout::contiguous(selector_backing_shape, 1), ScalarType::kInt32);
  DeviceTensor bad_code = context.allocate(
      TensorLayout::contiguous(code_backing_shape, 2));
  // DeviceTensor intentionally exposes only validated contiguous allocation;
  // mutate test-only metadata on oversized backing allocations to inject the
  // otherwise-unconstructible adversarial views into stage preflight.
  const_cast<TensorLayout&>(bad_tokens.layout()) = bad_token_layout;
  const_cast<TensorLayout&>(bad_selectors.layout()) = bad_selector_layout;
  const_cast<TensorLayout&>(bad_code.layout()) = bad_code_layout;
  TensorContext foreign_context(device, context_options);
  const int32_t full_range[] = {0, 128, 0, 0};
  H3AttentionRanges foreign_ranges = H3AttentionRanges::create(
      foreign_context, config.sequence, full_range, 4);
  {
    TensorBatch invalid = context.begin_batch();
    auto rejected = [&](DeviceTensor& token_arg, DeviceTensor& selector_arg,
                        DeviceTensor& code_arg,
                        const H3AttentionRanges* ranges = nullptr) {
      bool threw = false;
      try {
        first.record(invalid, token_arg, selector_arg, code_arg, cosine, sine,
                     scratch, ranges);
      } catch (const std::invalid_argument&) { threw = true; }
      CHECK(threw && invalid.remaining_operator_capacity() == 64u);
    };
    rejected(bad_tokens, selectors, code);
    rejected(tokens, bad_selectors, code);
    rejected(tokens, selectors, bad_code);
    rejected(tokens, selectors, code, &foreign_ranges);
    // None of the rejected calls poisoned access tracking or consumed an op.
    first.record(invalid, tokens, selectors, code, cosine, sine, scratch);
    CHECK(invalid.remaining_operator_capacity() == 39u);
    invalid.submit().wait();
  }
  auto run_chain = [&] {
    context.upload_bytes(tokens, input.data(), input.size() * 2);
    TensorBatch batch = context.begin_batch();
    first.record(batch, tokens, selectors, code, cosine, sine, scratch);
    second.record(batch, tokens, selectors, code, cosine, sine, scratch);
    batch.submit().wait();
    std::vector<uint16_t> output(input.size());
    context.download_bytes(tokens, output.data(), output.size() * 2);
    return output;
  };
  const std::vector<uint16_t> output = run_chain();
  uint64_t digest = 1469598103934665603ull;
  for (uint16_t bits : output) {
    digest ^= bits & 0xffu; digest *= 1099511628211ull;
    digest ^= bits >> 8; digest *= 1099511628211ull;
  }
  CHECK(digest == 0x70e1eaf01723020bull);
  const uint64_t stable_used = context.pooled_used_bytes();
  const uint64_t stable_reserved = context.reserved_bytes();
  const uint64_t stable_descriptors = context.descriptor_set_allocations();
  CHECK(run_chain() == output);
  CHECK(context.pooled_used_bytes() == stable_used);
  CHECK(context.reserved_bytes() == stable_reserved);
  CHECK(context.descriptor_set_allocations() == stable_descriptors);

  // Corruption in the sixth and final logical projection is decoded before
  // the first Vulkan allocation. The active stage and allocator HWM remain
  // exactly unchanged.
  bool fc2_rejected = false;
  try { first.load(corrupt_fc2, 0); }
  catch (const std::exception&) { fc2_rejected = true; }
  CHECK(fc2_rejected && first.loaded());
  CHECK(context.pooled_used_bytes() == stable_used);
  CHECK(context.reserved_bytes() == stable_reserved);
  CHECK(context.descriptor_set_allocations() == stable_descriptors);
  CHECK(run_chain() == output);

  bool corrupt_rejected = false;
  try { first.load(corrupt, 0); }
  catch (const std::exception&) { corrupt_rejected = true; }
  CHECK(corrupt_rejected && first.loaded());
  CHECK(context.pooled_used_bytes() == stable_used);
  CHECK(context.reserved_bytes() == stable_reserved);
  CHECK(context.descriptor_set_allocations() == stable_descriptors);
  CHECK(run_chain() == output);
  const uint64_t persistent = first.persistent_bytes();
  first.unload();
  second.unload();

  // Multi-layer graph: one batch, one shared transformed-activation scratch,
  // exact capacity boundary, every device-only block boundary, transactional
  // corruption in the last projection of the last layer, and repeat reuse.
  TensorContextOptions graph_options;
  graph_options.max_batch_operators = 128;
  TensorContext graph_context(device, graph_options);
  H3MainGraphConfig graph_config;
  graph_config.block = config;
  graph_config.layers = 3;
  const uint64_t graph_create_used = graph_context.pooled_used_bytes();
  const uint64_t graph_create_reserved = graph_context.reserved_bytes();
  const uint64_t graph_create_descriptors =
      graph_context.descriptor_set_allocations();
  ExactH3MainGraph graph = ExactH3MainGraph::create(
      graph_context, graph_config);
  CHECK(!graph.loaded() && graph.layers() == 3u);
  CHECK(graph.persistent_bytes() == 0u && graph.scratch_bytes() == 0u &&
        graph.peak_device_bytes() == 0u);
  CHECK(graph_context.pooled_used_bytes() == graph_create_used);
  CHECK(graph_context.reserved_bytes() == graph_create_reserved);
  CHECK(graph_context.descriptor_set_allocations() == graph_create_descriptors);
  // Complete validation, including the last layer's fc2 metadata, is host-only
  // and precedes scratch or weight allocation on an initial load.
  bool initial_corrupt_rejected = false;
  try { graph.load(corrupt_graph); }
  catch (const std::exception&) { initial_corrupt_rejected = true; }
  CHECK(initial_corrupt_rejected && !graph.loaded());
  CHECK(graph.persistent_bytes() == 0u && graph.scratch_bytes() == 0u &&
        graph.peak_device_bytes() == 0u);
  CHECK(graph_context.pooled_used_bytes() == graph_create_used);
  CHECK(graph_context.reserved_bytes() == graph_create_reserved);
  CHECK(graph_context.descriptor_set_allocations() == graph_create_descriptors);
  const uint64_t graph_token_shape[] = {config.sequence, config.hidden};
  const uint64_t graph_selector_shape[] = {config.sequence};
  const uint64_t graph_code_shape[] = {config.timesteps, config.adaln_rank};
  const uint64_t graph_rope_shape[] = {config.sequence, 96};
  DeviceTensor graph_tokens = graph_context.allocate(
      TensorLayout::contiguous(graph_token_shape, 2), ScalarType::kBFloat16);
  DeviceTensor graph_selectors = graph_context.allocate(
      TensorLayout::contiguous(graph_selector_shape, 1), ScalarType::kInt32);
  DeviceTensor graph_code = graph_context.allocate(
      TensorLayout::contiguous(graph_code_shape, 2));
  DeviceTensor graph_cosine = graph_context.allocate(
      TensorLayout::contiguous(graph_rope_shape, 2));
  DeviceTensor graph_sine = graph_context.allocate(
      TensorLayout::contiguous(graph_rope_shape, 2));
  graph_context.upload_bytes(graph_selectors, host_selectors.data(),
                             host_selectors.size() * 4);
  graph_context.upload(graph_code, host_code.data(), host_code.size());
  graph_context.upload(graph_cosine, host_cos.data(), host_cos.size());
  graph_context.upload(graph_sine, host_sin.data(), host_sin.size());
  std::vector<DeviceTensor> boundaries;
  boundaries.reserve(graph_config.layers);
  for (uint32_t layer = 0; layer < graph_config.layers; ++layer)
    boundaries.push_back(graph_context.allocate(
        TensorLayout::contiguous(graph_token_shape, 2), ScalarType::kBFloat16));
  H3MainGraphReplayTaps graph_taps{boundaries.data(),
                                   static_cast<uint32_t>(boundaries.size())};
  DeviceTensor graph_dummy = graph_context.allocate(
      TensorLayout::contiguous(graph_token_shape, 2), ScalarType::kBFloat16);
  DeviceTensor graph_dummy_out = graph_context.allocate(
      TensorLayout::contiguous(graph_token_shape, 2), ScalarType::kBFloat16);
  const uint64_t graph_preload_used = graph_context.pooled_used_bytes();
  graph.load(graph_checkpoint);
  CHECK(graph.loaded() && graph.required_operators() == 75u);
  CHECK(graph.required_operators(&graph_taps) == 78u);
  {
    TensorBatch short_batch = graph_context.begin_batch();
    for (uint32_t i = graph.required_operators(&graph_taps); i <= 128; ++i)
      short_batch.copy(graph_dummy, graph_dummy_out);
    bool rejected = false;
    try {
      graph.record(short_batch, graph_tokens, graph_selectors, graph_code,
                   graph_cosine, graph_sine, nullptr, &graph_taps);
    } catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected && short_batch.remaining_operator_capacity() == 77u);
  }
  auto run_graph = [&] {
    graph_context.upload_bytes(graph_tokens, input.data(), input.size() * 2);
    TensorBatch batch = graph_context.begin_batch();
    for (uint32_t i = graph.required_operators(&graph_taps); i < 128; ++i)
      batch.copy(graph_dummy, graph_dummy_out);
    graph.record(batch, graph_tokens, graph_selectors, graph_code,
                 graph_cosine, graph_sine, nullptr, &graph_taps);
    CHECK(batch.remaining_operator_capacity() == 0u);
    batch.submit().wait();
    std::vector<uint16_t> result(input.size());
    graph_context.download_bytes(graph_tokens, result.data(), result.size() * 2);
    return result;
  };
  const std::vector<uint16_t> graph_output = run_graph();
  auto digest_bf16 = [](const std::vector<uint16_t>& values) {
    uint64_t hash = 1469598103934665603ull;
    for (uint16_t bits : values) {
      hash ^= bits & 0xffu; hash *= 1099511628211ull;
      hash ^= bits >> 8; hash *= 1099511628211ull;
    }
    return hash;
  };
  std::vector<uint64_t> boundary_hashes;
  for (DeviceTensor& boundary : boundaries) {
    std::vector<uint16_t> boundary_values(input.size());
    graph_context.download_bytes(boundary, boundary_values.data(),
                                 boundary_values.size() * 2);
    boundary_hashes.push_back(digest_bf16(boundary_values));
  }
  CHECK(boundary_hashes == std::vector<uint64_t>({
      0xc40d66ec8f2b7f26ull, 0x70e1eaf01723020bull,
      0xe9b03bc5e1718291ull}));
  CHECK(digest_bf16(graph_output) == boundary_hashes.back());
  CHECK(graph.required_operators(0, 1, &graph_taps) == 26u);
  CHECK(graph.required_operators(1, 2, &graph_taps) == 52u);
  graph_context.upload_bytes(graph_tokens, input.data(), input.size() * 2);
  {
    TensorBatch split = graph_context.begin_batch();
    graph.record_layers(split, graph_tokens, graph_selectors, graph_code,
                        graph_cosine, graph_sine, 0, 1, nullptr, &graph_taps);
    graph.record_layers(split, graph_tokens, graph_selectors, graph_code,
                        graph_cosine, graph_sine, 1, 2, nullptr, &graph_taps);
    CHECK(split.remaining_operator_capacity() == 50u);
    split.submit().wait();
  }
  std::vector<uint16_t> split_output(input.size());
  graph_context.download_bytes(graph_tokens, split_output.data(),
                               split_output.size() * 2);
  CHECK(split_output == graph_output);
  {
    TensorBatch invalid_span = graph_context.begin_batch();
    bool rejected = false;
    try {
      graph.record_layers(invalid_span, graph_tokens, graph_selectors,
                          graph_code, graph_cosine, graph_sine, 2, 2);
    } catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected && invalid_span.remaining_operator_capacity() == 128u);
  }
  const uint64_t graph_used = graph_context.pooled_used_bytes();
  const uint64_t graph_reserved = graph_context.reserved_bytes();
  const uint64_t graph_descriptors = graph_context.descriptor_set_allocations();
  CHECK(run_graph() == graph_output);
  CHECK(graph_context.pooled_used_bytes() == graph_used);
  CHECK(graph_context.reserved_bytes() == graph_reserved);
  CHECK(graph_context.descriptor_set_allocations() == graph_descriptors);
  const uint64_t graph_peak = graph.peak_device_bytes();
  bool active_reload_rejected = false;
  try { graph.load(graph_checkpoint); }
  catch (const std::logic_error&) { active_reload_rejected = true; }
  CHECK(active_reload_rejected && graph.loaded());
  CHECK(graph_context.pooled_used_bytes() == graph_used);
  CHECK(graph_context.reserved_bytes() == graph_reserved);
  CHECK(graph_context.descriptor_set_allocations() == graph_descriptors);
  CHECK(graph.peak_device_bytes() == graph_peak);
  CHECK(run_graph() == graph_output);
  const uint64_t graph_persistent = graph.persistent_bytes();
  const uint64_t graph_scratch = graph.scratch_bytes();
  graph.unload();
  CHECK(!graph.loaded() && graph.persistent_bytes() == 0u &&
        graph.scratch_bytes() == 0u && graph.peak_device_bytes() == 0u);
  CHECK_MSG(graph_context.pooled_used_bytes() == graph_preload_used,
            "H3 graph unload used %llu, baseline %llu",
            static_cast<unsigned long long>(graph_context.pooled_used_bytes()),
            static_cast<unsigned long long>(graph_preload_used));
  graph.load(graph_checkpoint);
  CHECK(graph.persistent_bytes() == graph_persistent &&
        graph.scratch_bytes() == graph_scratch &&
        graph.peak_device_bytes() == graph_peak);
  CHECK(run_graph() == graph_output);
  std::printf(
      "  CUDA-off H3 3-layer graph FNV64 %016llx, ops %u, persistent/scratch %.2f/%.2f MiB\n",
      static_cast<unsigned long long>(digest_bf16(graph_output)),
      graph.required_operators(), double(graph.persistent_bytes()) / 1048576.0,
      double(graph.scratch_bytes()) / 1048576.0);
  graph.unload();

  // Complete small/tail transformer evaluation: refiner cache, endpoint
  // projections/scatters, main graph and final modality heads.
  ExactH3TransformerConfig transformer_config;
  transformer_config.main.block = config;
  transformer_config.main.layers = 1;
  transformer_config.text_rows = 3;
  transformer_config.video_rows = 60;
  transformer_config.audio_rows = 2;
  transformer_config.text_dim = 16;
  transformer_config.video_dim = 4;
  transformer_config.audio_dim = 2;
  transformer_config.refiner_layers = 1;
  const uint64_t prompt_shape[] = {3, 16};
  const uint64_t video_input_shape[] = {60, 4};
  const uint64_t audio_input_shape[] = {2, 2};
  DeviceTensor prompt = graph_context.allocate(
      TensorLayout::contiguous(prompt_shape, 2));
  DeviceTensor transformer_video = graph_context.allocate(
      TensorLayout::contiguous(video_input_shape, 2));
  DeviceTensor transformer_audio = graph_context.allocate(
      TensorLayout::contiguous(audio_input_shape, 2));
  DeviceTensor transformer_video_out = graph_context.allocate(
      TensorLayout::contiguous(video_input_shape, 2));
  DeviceTensor transformer_audio_out = graph_context.allocate(
      TensorLayout::contiguous(audio_input_shape, 2));
  DeviceTensor transformer_text_idx = graph_context.allocate(
      TensorLayout::contiguous(prompt_shape, 1), ScalarType::kInt32);
  const uint64_t video_index_shape[] = {60};
  const uint64_t audio_index_shape[] = {2};
  DeviceTensor transformer_video_idx = graph_context.allocate(
      TensorLayout::contiguous(video_index_shape, 1), ScalarType::kInt32);
  DeviceTensor transformer_audio_idx = graph_context.allocate(
      TensorLayout::contiguous(audio_index_shape, 1), ScalarType::kInt32);
  DeviceTensor transformer_selectors = graph_context.allocate(
      TensorLayout::contiguous(selector_shape, 1), ScalarType::kInt32);
  DeviceTensor transformer_code = graph_context.allocate(
      TensorLayout::contiguous(code_shape, 2));
  DeviceTensor transformer_cosine = graph_context.allocate(
      TensorLayout::contiguous(rope_shape, 2));
  DeviceTensor transformer_sine = graph_context.allocate(
      TensorLayout::contiguous(rope_shape, 2));
  DeviceTensor transformer_video_ts = graph_context.allocate(
      TensorLayout::contiguous(video_index_shape, 1), ScalarType::kInt32);
  DeviceTensor transformer_audio_ts = graph_context.allocate(
      TensorLayout::contiguous(audio_index_shape, 1), ScalarType::kInt32);
  const std::vector<float> prompt_values = values(3 * 16, 163, 1.0f / 32.0f);
  const std::vector<float> video_values = values(60 * 4, 167, 1.0f / 32.0f);
  const std::vector<float> audio_values = values(2 * 2, 173, 1.0f / 32.0f);
  std::vector<int32_t> text_idx{0, 1, 2}, audio_idx{3, 4}, video_idx(60);
  for (int32_t i = 0; i < 60; ++i) video_idx[i] = i + 5;
  std::vector<int32_t> zero_video_ts(60, 0), zero_audio_ts(2, 0);
  graph_context.upload(prompt, prompt_values.data(), prompt_values.size());
  graph_context.upload(transformer_video, video_values.data(), video_values.size());
  graph_context.upload(transformer_audio, audio_values.data(), audio_values.size());
  graph_context.upload_bytes(transformer_text_idx, text_idx.data(), text_idx.size() * 4);
  graph_context.upload_bytes(transformer_video_idx, video_idx.data(), video_idx.size() * 4);
  graph_context.upload_bytes(transformer_audio_idx, audio_idx.data(), audio_idx.size() * 4);
  graph_context.upload_bytes(transformer_selectors, host_selectors.data(),
                             host_selectors.size() * 4);
  graph_context.upload(transformer_code, host_code.data(), host_code.size());
  graph_context.upload(transformer_cosine, host_cos.data(), host_cos.size());
  graph_context.upload(transformer_sine, host_sin.data(), host_sin.size());
  graph_context.upload_bytes(transformer_video_ts, zero_video_ts.data(),
                             zero_video_ts.size() * 4);
  graph_context.upload_bytes(transformer_audio_ts, zero_audio_ts.data(),
                             zero_audio_ts.size() * 4);
  DeviceTensor shared_forward_tap = graph_context.allocate(
      TensorLayout::contiguous(token_shape, 2), ScalarType::kBFloat16);
  const uint64_t transformer_baseline = graph_context.pooled_used_bytes();
  const uint64_t transformer_baseline_reserved = graph_context.reserved_bytes();
  const uint64_t transformer_baseline_descriptors =
      graph_context.descriptor_set_allocations();
  ExactH3Transformer transformer = ExactH3Transformer::create(
      graph_context, transformer_config);
  CHECK(!transformer.loaded() && transformer.persistent_bytes() == 0u &&
        transformer.scratch_bytes() == 0u);
  bool transformer_corrupt_rejected = false;
  try { transformer.load(corrupt_transformer); }
  catch (const std::exception&) { transformer_corrupt_rejected = true; }
  CHECK(transformer_corrupt_rejected && !transformer.loaded());
  CHECK(graph_context.pooled_used_bytes() == transformer_baseline);
  std::vector<SafeTensors*> invalid_endpoint_archives{
      &corrupt_transformer_dtype};
  for (SafeTensors& invalid : corrupt_transformer_metadata)
    invalid_endpoint_archives.push_back(&invalid);
  for (SafeTensors* invalid : invalid_endpoint_archives) {
    bool rejected = false;
    try { transformer.load(*invalid); }
    catch (const std::exception&) { rejected = true; }
    CHECK(rejected && !transformer.loaded());
    CHECK(graph_context.pooled_used_bytes() == transformer_baseline);
    CHECK(graph_context.reserved_bytes() == transformer_baseline_reserved);
    CHECK(graph_context.descriptor_set_allocations() ==
          transformer_baseline_descriptors);
  }
  transformer.load(transformer_checkpoint);
  CHECK(transformer.loaded() && !transformer.text_prepared());
  H3TransformerTextReplayTaps null_text_taps{nullptr, 6};
  H3TransformerTextReplayTaps short_text_taps{&prompt, 5};
  for (const H3TransformerTextReplayTaps* invalid :
       {&null_text_taps, &short_text_taps}) {
    bool rejected = false;
    try { transformer.prepare_text(prompt, invalid); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected && !transformer.text_prepared());
  }
  transformer.prepare_text(prompt);
  CHECK(transformer.text_prepared());
  CHECK(transformer.required_forward_operators() == 42u);
  H3MainGraphReplayTaps null_main_taps{nullptr, 1};
  H3TransformerForwardReplayTaps null_forward_taps{
      nullptr, nullptr, &null_main_taps};
  H3MainGraphReplayTaps aliased_main_taps{&shared_forward_tap, 1};
  H3TransformerForwardReplayTaps aliased_forward_taps{
      &shared_forward_tap, nullptr, &aliased_main_taps};
  {
    TensorBatch transactional = graph_context.begin_batch();
    auto reject_forward = [&](const H3AttentionRanges* test_ranges,
                              const H3TransformerForwardReplayTaps* test_taps) {
      bool rejected = false;
      try {
        transformer.record_forward(
            transactional, transformer_video, transformer_audio,
            transformer_selectors, transformer_code, transformer_cosine,
            transformer_sine, transformer_video_ts, transformer_audio_ts,
            transformer_video_out, transformer_audio_out, test_ranges,
            test_taps);
      } catch (const std::invalid_argument&) { rejected = true; }
      CHECK(rejected && transactional.remaining_operator_capacity() == 128u);
    };
    reject_forward(&foreign_ranges, nullptr);
    reject_forward(nullptr, &null_forward_taps);
    reject_forward(nullptr, &aliased_forward_taps);
    transformer.record_forward(
        transactional, transformer_video, transformer_audio,
        transformer_selectors, transformer_code, transformer_cosine,
        transformer_sine, transformer_video_ts, transformer_audio_ts,
        transformer_video_out, transformer_audio_out);
    CHECK(transactional.remaining_operator_capacity() == 86u);
    transactional.submit().wait();
  }
  auto run_transformer = [&] {
    TensorBatch batch = graph_context.begin_batch();
    transformer.record_forward(
        batch, transformer_video, transformer_audio, transformer_selectors,
        transformer_code, transformer_cosine, transformer_sine,
        transformer_video_ts, transformer_audio_ts, transformer_video_out,
        transformer_audio_out);
    CHECK(batch.remaining_operator_capacity() == 86u);
    batch.submit().wait();
    std::vector<float> video_result(60 * 4), audio_result(2 * 2);
    graph_context.download(transformer_video_out, video_result.data(),
                           video_result.size());
    graph_context.download(transformer_audio_out, audio_result.data(),
                           audio_result.size());
    video_result.insert(video_result.end(), audio_result.begin(), audio_result.end());
    return video_result;
  };
  const std::vector<float> transformer_output = run_transformer();
  const uint64_t transformer_digest = fnv64_floats(transformer_output);
  CHECK(transformer_digest == 0x30dec4598d352f88ull);
  std::printf("  CUDA-off H3 transformer S65 FNV64 %016llx, ops %u, persistent/scratch %.2f/%.2f MiB\n",
      static_cast<unsigned long long>(transformer_digest),
      transformer.required_forward_operators(),
      double(transformer.persistent_bytes()) / 1048576.0,
      double(transformer.scratch_bytes()) / 1048576.0);
  CHECK(run_transformer() == transformer_output);
  const uint64_t transformer_persistent = transformer.persistent_bytes();
  const uint64_t transformer_scratch = transformer.scratch_bytes();
  const uint64_t transformer_live_used = graph_context.pooled_used_bytes();
  const uint64_t transformer_reserved = graph_context.reserved_bytes();
  const uint64_t transformer_descriptors =
      graph_context.descriptor_set_allocations();
  bool transformer_reload_rejected = false;
  try { transformer.load(corrupt_transformer_metadata.back()); }
  catch (const std::logic_error&) { transformer_reload_rejected = true; }
  CHECK(transformer_reload_rejected && transformer.text_prepared());
  CHECK(graph_context.pooled_used_bytes() == transformer_live_used);
  CHECK(graph_context.reserved_bytes() == transformer_reserved);
  CHECK(graph_context.descriptor_set_allocations() == transformer_descriptors);
  CHECK(run_transformer() == transformer_output);
  transformer.unload();
  CHECK(!transformer.loaded() && transformer.persistent_bytes() == 0u &&
        transformer.scratch_bytes() == 0u &&
        transformer.peak_device_bytes() == 0u);
  CHECK(graph_context.pooled_used_bytes() == transformer_baseline);
  transformer.load(transformer_checkpoint);
  CHECK(transformer.persistent_bytes() == transformer_persistent &&
        transformer.scratch_bytes() == transformer_scratch);
  transformer.prepare_text(prompt);
  CHECK(run_transformer() == transformer_output);
  transformer.unload();

  // Complete CUDA-off T2VA denoise trajectory. Modality rows remain on the
  // device across every transformer evaluation and Euler update; only final
  // rows cross back to the host.
  TensorContextOptions denoise_options;
  denoise_options.max_batch_operators = 64;
  TensorContext denoise_context(device, denoise_options);
  ExactH3DenoiseConfig denoise_config;
  denoise_config.transformer = transformer_config;
  denoise_config.transformer.main.block.timesteps = 2;
  denoise_config.layout.num_text = 3;
  denoise_config.layout.num_audio_rows = 2;
  denoise_config.layout.num_video_rows = 60;
  denoise_config.layout.num_audio_latents = 1;
  denoise_config.layout.num_latent_frames = 15;
  denoise_config.layout.latent_height = 4;
  denoise_config.layout.latent_width = 4;
  denoise_config.indices = dit::build_indices(denoise_config.layout);
  denoise_config.position_ids = dit::build_position_ids(denoise_config.layout);
  denoise_config.attention_band = 1;
  const uint64_t denoise_baseline = denoise_context.pooled_used_bytes();
  const uint64_t denoise_baseline_reserved = denoise_context.reserved_bytes();
  const uint64_t denoise_baseline_descriptors =
      denoise_context.descriptor_set_allocations();
  ExactH3Denoiser denoiser = ExactH3Denoiser::create(
      denoise_context, denoise_config);
  bool denoise_corrupt_rejected = false;
  try { denoiser.load(corrupt_transformer_metadata.front()); }
  catch (const std::exception&) { denoise_corrupt_rejected = true; }
  CHECK(denoise_corrupt_rejected && !denoiser.loaded());
  CHECK_MSG(denoise_context.pooled_used_bytes() == denoise_baseline,
            "corrupt denoise load used %llu baseline %llu",
            static_cast<unsigned long long>(denoise_context.pooled_used_bytes()),
            static_cast<unsigned long long>(denoise_baseline));
  CHECK(denoise_context.reserved_bytes() == denoise_baseline_reserved);
  CHECK(denoise_context.descriptor_set_allocations() ==
        denoise_baseline_descriptors);
  denoiser.load(transformer_checkpoint);
  CHECK(denoiser.loaded() && !denoiser.prepared());
  CHECK(denoiser.required_step_operators() == 44u);
  const uint64_t capacity_shape[] = {1};
  DeviceTensor capacity_source = denoise_context.allocate(
      TensorLayout::contiguous(capacity_shape, 1));
  DeviceTensor capacity_destination = denoise_context.allocate(
      TensorLayout::contiguous(capacity_shape, 1));
  {
    // The complete transformer + two Euler updates fits exactly. Preflight is
    // observational: the same recording can still accept a smaller graph.
    TensorBatch exact_capacity = denoise_context.begin_batch();
    for (uint32_t i = denoiser.required_step_operators(); i < 64u; ++i)
      exact_capacity.copy(capacity_source, capacity_destination);
    exact_capacity.require_operator_capacity(denoiser.required_step_operators());
    CHECK(exact_capacity.remaining_operator_capacity() == 44u);
    exact_capacity.copy(capacity_source, capacity_destination);
    exact_capacity.submit().wait();
  }
  {
    TensorBatch short_capacity = denoise_context.begin_batch();
    for (uint32_t i = denoiser.required_step_operators(); i <= 64u; ++i)
      short_capacity.copy(capacity_source, capacity_destination);
    bool rejected = false;
    try {
      short_capacity.require_operator_capacity(
          denoiser.required_step_operators());
    } catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected && short_capacity.remaining_operator_capacity() == 43u);
    short_capacity.copy(capacity_source, capacity_destination);
    short_capacity.submit().wait();
  }
  sampler::FlowScheduler denoise_video(12.0f), denoise_audio(3.0f);
  denoise_video.set_timesteps(4);
  denoise_audio.set_timesteps(4);
  denoiser.prepare(prompt_values.data(), prompt_values.size(),
                   video_values.data(), video_values.size(),
                   audio_values.data(), audio_values.size());
  CHECK(denoiser.prepared());
  const ExactH3DenoiseResult denoise_output = denoiser.run(
      denoise_video, denoise_audio);
  CHECK(!denoise_output.cancelled && denoise_output.steps_completed == 3u);
  std::vector<float> denoise_joined = denoise_output.video_rows;
  denoise_joined.insert(denoise_joined.end(), denoise_output.audio_rows.begin(),
                        denoise_output.audio_rows.end());
  const uint64_t denoise_digest = fnv64_floats(denoise_joined);
  CHECK(denoise_digest == 0xad06feae77d1c494ull);
  const uint64_t denoise_used = denoise_context.pooled_used_bytes();
  const uint64_t denoise_reserved = denoise_context.reserved_bytes();
  const uint64_t denoise_descriptors =
      denoise_context.descriptor_set_allocations();
  denoiser.prepare(prompt_values.data(), prompt_values.size(),
                   video_values.data(), video_values.size(),
                   audio_values.data(), audio_values.size());
  ExactH3DenoiseResult denoise_repeat = denoiser.run(
      denoise_video, denoise_audio);
  CHECK(denoise_repeat.video_rows == denoise_output.video_rows &&
        denoise_repeat.audio_rows == denoise_output.audio_rows);
  CHECK(denoise_context.pooled_used_bytes() == denoise_used);
  CHECK(denoise_context.reserved_bytes() == denoise_reserved);
  CHECK(denoise_context.descriptor_set_allocations() == denoise_descriptors);
  denoiser.prepare(prompt_values.data(), prompt_values.size(),
                   video_values.data(), video_values.size(),
                   audio_values.data(), audio_values.size());
  ExactH3DenoiseResult cancelled = denoiser.run(
      denoise_video, denoise_audio,
      [](uint32_t step, uint32_t) { return step != 0; });
  CHECK(cancelled.cancelled && cancelled.steps_completed == 1u);
  sampler::FlowScheduler unsupported_audio = denoise_audio;
  unsupported_audio.set_sampler(sampler::SamplerKind::kAb2);
  bool ab2_rejected = false;
  try { (void)denoiser.run(denoise_video, unsupported_audio); }
  catch (const std::invalid_argument&) { ab2_rejected = true; }
  CHECK(ab2_rejected && denoiser.prepared());
  const uint64_t denoise_persistent = denoiser.persistent_bytes();
  const uint64_t denoise_scratch = denoiser.scratch_bytes();
  CHECK(denoise_persistent != 0 && denoise_scratch != 0 &&
        denoiser.peak_device_bytes() == denoise_persistent + denoise_scratch);
  denoiser.unload();
  CHECK(!denoiser.loaded() && !denoiser.prepared() &&
        denoiser.persistent_bytes() == 0u && denoiser.scratch_bytes() == 0u &&
        denoiser.peak_device_bytes() == 0u);
  // The context intentionally keeps its paired upload/readback buffers. Pool
  // usage includes driver memory-requirement rounding, so use the observed
  // unloaded watermark and prove it is bounded and stable across reload.
  const uint64_t denoise_staging_used = denoise_context.pooled_used_bytes();
  CHECK(denoise_context.staging_capacity_bytes() != 0 &&
        denoise_staging_used > denoise_baseline &&
        denoise_staging_used - denoise_baseline < denoise_persistent);
  denoiser.load(transformer_checkpoint);
  denoiser.prepare(prompt_values.data(), prompt_values.size(),
                   video_values.data(), video_values.size(),
                   audio_values.data(), audio_values.size());
  ExactH3DenoiseResult after_reload = denoiser.run(
      denoise_video, denoise_audio);
  CHECK(after_reload.video_rows == denoise_output.video_rows &&
        after_reload.audio_rows == denoise_output.audio_rows);
  CHECK(denoiser.persistent_bytes() == denoise_persistent &&
        denoiser.scratch_bytes() == denoise_scratch);
  std::printf(
      "  CUDA-off H3 denoise S65 x3 FNV64 %016llx, persistent/scratch %.2f/%.2f MiB\n",
      static_cast<unsigned long long>(denoise_digest),
      double(denoise_persistent) / 1048576.0,
      double(denoise_scratch) / 1048576.0);
  denoiser.unload();
  CHECK(denoise_context.pooled_used_bytes() == denoise_staging_used);

  CHECK(first.persistent_bytes() == 0u && second.persistent_bytes() == 0u);
  CHECK(context.pooled_used_bytes() + 2 * persistent <= stable_used);
  first.load(valid, 0);
  second.load(valid, 0);
  CHECK(run_chain() == output);
  CHECK(first.persistent_bytes() == persistent);
  std::printf(
      "  CUDA-off H3 S65 two-stage FNV64 %016llx, persistent/scratch %.2f/%.2f MiB\n",
      static_cast<unsigned long long>(digest),
      double(first.persistent_bytes() + second.persistent_bytes()) / 1048576.0,
      double(scratch.reserved_bytes()) / 1048576.0);
  first.unload();
  second.unload();

  // This executable links no CUDA code.  When the shipped checkpoint is
  // present, additionally load its real NVFP4 block 0 and pin the canonical
  // S65 result used by the CUDA-enabled replay suite.
  const std::filesystem::path real_path = std::filesystem::path(
      VIDFAB_TEST_SOURCE_DIR) /
      "weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors";
  if (std::filesystem::exists(real_path)) {
    H3BlockConfig real_config;
    real_config.sequence = 65;
    SafeTensors real_checkpoint;
    real_checkpoint.open(real_path.string());
    ExactH3BlockStage real_stage = ExactH3BlockStage::create(context, real_config);
    real_stage.load(real_checkpoint, 0);
    ExactH3BlockScratch real_scratch =
        ExactH3BlockScratch::create(context, real_config);
    real_stage.prepare(real_scratch);
    const uint64_t real_token_shape[] = {real_config.sequence, real_config.hidden};
    const uint64_t real_selector_shape[] = {real_config.sequence};
    const uint64_t real_code_shape[] = {real_config.timesteps,
                                        real_config.adaln_rank};
    const uint64_t real_rope_shape[] = {real_config.sequence, 96};
    DeviceTensor real_tokens = context.allocate(
        TensorLayout::contiguous(real_token_shape, 2), ScalarType::kBFloat16);
    DeviceTensor real_selectors = context.allocate(
        TensorLayout::contiguous(real_selector_shape, 1), ScalarType::kInt32);
    DeviceTensor real_code = context.allocate(
        TensorLayout::contiguous(real_code_shape, 2));
    DeviceTensor real_cosine = context.allocate(
        TensorLayout::contiguous(real_rope_shape, 2));
    DeviceTensor real_sine = context.allocate(
        TensorLayout::contiguous(real_rope_shape, 2));
    std::vector<uint16_t> real_input(
        size_t(real_config.sequence) * real_config.hidden);
    for (size_t i = 0; i < real_input.size(); ++i)
      real_input[i] = f32_to_bf16(float(int(i % 61) - 30) / 64.0f);
    std::vector<int32_t> real_selector_values(real_config.sequence);
    for (uint32_t i = 0; i < real_config.sequence; ++i)
      real_selector_values[i] = static_cast<int32_t>(i % real_config.modalities);
    std::vector<float> real_code_values(
        size_t(real_config.timesteps) * real_config.adaln_rank);
    for (size_t i = 0; i < real_code_values.size(); ++i)
      real_code_values[i] = float(int(i % 7) - 3) / 16.0f;
    std::vector<float> real_cosine_values(size_t(real_config.sequence) * 96, 1.0f);
    std::vector<float> real_sine_values(real_cosine_values.size(), 0.0f);
    context.upload_bytes(real_selectors, real_selector_values.data(),
                         real_selector_values.size() * 4);
    context.upload(real_code, real_code_values.data(), real_code_values.size());
    context.upload(real_cosine, real_cosine_values.data(), real_cosine_values.size());
    context.upload(real_sine, real_sine_values.data(), real_sine_values.size());
    auto run_real = [&] {
      context.upload_bytes(real_tokens, real_input.data(), real_input.size() * 2);
      TensorBatch batch = context.begin_batch();
      real_stage.record(batch, real_tokens, real_selectors, real_code,
                        real_cosine, real_sine, real_scratch);
      batch.submit().wait();
      std::vector<uint16_t> result(real_input.size());
      context.download_bytes(real_tokens, result.data(), result.size() * 2);
      return result;
    };
    const std::vector<uint16_t> real_output = run_real();
    uint64_t real_digest = 1469598103934665603ull;
    for (uint16_t bits : real_output) {
      real_digest ^= bits & 0xffu; real_digest *= 1099511628211ull;
      real_digest ^= bits >> 8; real_digest *= 1099511628211ull;
    }
    CHECK(real_digest == 0x191929c14480e873ull);
    const uint64_t real_reserved = context.reserved_bytes();
    const uint64_t real_descriptors = context.descriptor_set_allocations();
    CHECK(run_real() == real_output);
    CHECK(context.reserved_bytes() == real_reserved);
    CHECK(context.descriptor_set_allocations() == real_descriptors);
    const uint64_t real_used = context.pooled_used_bytes();
    real_stage.unload();
    CHECK(context.pooled_used_bytes() < real_used);
    std::printf("  CUDA-off real H3 block0 S65 FNV64 %016llx\n",
                static_cast<unsigned long long>(real_digest));

    H3MainGraphConfig real_graph_config;
    real_graph_config.block = real_config;
    real_graph_config.layers = 2;
    const uint64_t real_graph_preload = context.pooled_used_bytes();
    ExactH3MainGraph real_graph = ExactH3MainGraph::create(
        context, real_graph_config);
    CHECK(context.pooled_used_bytes() == real_graph_preload);
    real_graph.load(real_checkpoint);
    CHECK(real_graph.required_operators() == 58u);
    auto run_real_graph = [&] {
      context.upload_bytes(real_tokens, real_input.data(), real_input.size() * 2);
      TensorBatch batch = context.begin_batch();
      real_graph.record(batch, real_tokens, real_selectors, real_code,
                        real_cosine, real_sine);
      batch.submit().wait();
      std::vector<uint16_t> result(real_input.size());
      context.download_bytes(real_tokens, result.data(), result.size() * 2);
      return result;
    };
    const std::vector<uint16_t> real_graph_output = run_real_graph();
    uint64_t real_graph_digest = 1469598103934665603ull;
    for (uint16_t bits : real_graph_output) {
      real_graph_digest ^= bits & 0xffu; real_graph_digest *= 1099511628211ull;
      real_graph_digest ^= bits >> 8; real_graph_digest *= 1099511628211ull;
    }
    CHECK(real_graph_digest == 0x7f940c81104e7471ull);
    const uint64_t real_graph_reserved = context.reserved_bytes();
    const uint64_t real_graph_descriptors = context.descriptor_set_allocations();
    CHECK(run_real_graph() == real_graph_output);
    CHECK(context.reserved_bytes() == real_graph_reserved);
    CHECK(context.descriptor_set_allocations() == real_graph_descriptors);
    const uint64_t real_graph_used = context.pooled_used_bytes();
    real_graph.unload();
    CHECK(context.pooled_used_bytes() == real_graph_preload);
    CHECK(real_graph_used > real_graph_preload);
    CHECK(real_graph.persistent_bytes() == 0u &&
          real_graph.scratch_bytes() == 0u &&
          real_graph.peak_device_bytes() == 0u);
    std::printf("  CUDA-off real H3 main2 S65 FNV64 %016llx\n",
                static_cast<unsigned long long>(real_graph_digest));

    {
    TensorContextOptions real_transformer_options;
    real_transformer_options.max_batch_operators = 2048;
    TensorContext context(device, real_transformer_options);
    ExactH3TransformerConfig real_transformer_config;
    real_transformer_config.main.block = real_config;
    real_transformer_config.main.layers = 50;
    real_transformer_config.text_rows = 3;
    real_transformer_config.video_rows = 60;
    real_transformer_config.audio_rows = 2;
    real_transformer_config.refiner_layers = 2;
    const uint64_t real_prompt_shape[] = {3, 5120};
    const uint64_t real_video_shape[] = {60, 96};
    const uint64_t real_audio_shape[] = {2, 32};
    const uint64_t real_video_index_shape[] = {60};
    const uint64_t real_audio_index_shape[] = {2};
    DeviceTensor real_prompt = context.allocate(
        TensorLayout::contiguous(real_prompt_shape, 2));
    DeviceTensor real_video = context.allocate(
        TensorLayout::contiguous(real_video_shape, 2));
    DeviceTensor real_audio = context.allocate(
        TensorLayout::contiguous(real_audio_shape, 2));
    DeviceTensor real_video_out = context.allocate(
        TensorLayout::contiguous(real_video_shape, 2));
    DeviceTensor real_audio_out = context.allocate(
        TensorLayout::contiguous(real_audio_shape, 2));
    DeviceTensor real_transformer_selectors = context.allocate(
        TensorLayout::contiguous(real_selector_shape, 1), ScalarType::kInt32);
    DeviceTensor real_transformer_code = context.allocate(
        TensorLayout::contiguous(real_code_shape, 2));
    DeviceTensor real_transformer_cosine = context.allocate(
        TensorLayout::contiguous(real_rope_shape, 2));
    DeviceTensor real_transformer_sine = context.allocate(
        TensorLayout::contiguous(real_rope_shape, 2));
    DeviceTensor real_video_ts = context.allocate(
        TensorLayout::contiguous(real_video_index_shape, 1), ScalarType::kInt32);
    DeviceTensor real_audio_ts = context.allocate(
        TensorLayout::contiguous(real_audio_index_shape, 1), ScalarType::kInt32);
    std::vector<float> real_prompt_values(3 * 5120);
    std::vector<float> real_video_values(60 * 96);
    std::vector<float> real_audio_values(2 * 32);
    for (size_t i = 0; i < real_prompt_values.size(); ++i)
      real_prompt_values[i] = float(int(i % 251) - 125) / 128.0f;
    for (size_t i = 0; i < real_video_values.size(); ++i)
      real_video_values[i] = float(int(i % 127) - 63) / 64.0f;
    for (size_t i = 0; i < real_audio_values.size(); ++i)
      real_audio_values[i] = float(int(i % 61) - 30) / 32.0f;
    std::vector<int32_t> real_video_ts_values(60, 0),
        real_audio_ts_values(2, 0);
    context.upload(real_prompt, real_prompt_values.data(), real_prompt_values.size());
    context.upload(real_video, real_video_values.data(), real_video_values.size());
    context.upload(real_audio, real_audio_values.data(), real_audio_values.size());
    context.upload_bytes(real_transformer_selectors, real_selector_values.data(),
                         real_selector_values.size() * 4);
    context.upload(real_transformer_code, real_code_values.data(),
                   real_code_values.size());
    context.upload(real_transformer_cosine, real_cosine_values.data(),
                   real_cosine_values.size());
    context.upload(real_transformer_sine, real_sine_values.data(),
                   real_sine_values.size());
    context.upload_bytes(real_video_ts, real_video_ts_values.data(),
                         real_video_ts_values.size() * 4);
    context.upload_bytes(real_audio_ts, real_audio_ts_values.data(),
                         real_audio_ts_values.size() * 4);
    const uint64_t real_transformer_baseline = context.pooled_used_bytes();
    ExactH3Transformer real_transformer = ExactH3Transformer::create(
        context, real_transformer_config);
    real_transformer.load(real_checkpoint);
    real_transformer.prepare_text(real_prompt);
    auto run_real_transformer = [&] {
      TensorBatch batch = context.begin_batch();
      real_transformer.record_forward(
          batch, real_video, real_audio, real_transformer_selectors,
          real_transformer_code, real_transformer_cosine,
          real_transformer_sine, real_video_ts, real_audio_ts,
          real_video_out, real_audio_out);
      batch.submit().wait();
      std::vector<float> result(real_video_values.size() + real_audio_values.size());
      context.download(real_video_out, result.data(), real_video_values.size());
      context.download(real_audio_out, result.data() + real_video_values.size(),
                       real_audio_values.size());
      return result;
    };
    const std::vector<float> real_transformer_output = run_real_transformer();
    const uint64_t real_transformer_digest =
        fnv64_floats(real_transformer_output);
    CHECK(real_transformer_digest == 0x42764ebbb3850be4ull);
    const uint64_t real_transformer_reserved = context.reserved_bytes();
    const uint64_t real_transformer_descriptors =
        context.descriptor_set_allocations();
    CHECK(run_real_transformer() == real_transformer_output);
    CHECK(context.reserved_bytes() == real_transformer_reserved);
    CHECK(context.descriptor_set_allocations() == real_transformer_descriptors);
    real_transformer.unload();
    CHECK(context.pooled_used_bytes() == real_transformer_baseline);
    CHECK(real_transformer.persistent_bytes() == 0u &&
          real_transformer.scratch_bytes() == 0u);
    std::printf("  CUDA-off real H3 transformer S65 FNV64 %016llx\n",
                static_cast<unsigned long long>(real_transformer_digest));
    }
  }
  std::error_code ignored;
  std::filesystem::remove(valid_path, ignored);
  std::filesystem::remove(corrupt_path, ignored);
  std::filesystem::remove(corrupt_fc2_path, ignored);
  std::filesystem::remove(graph_path, ignored);
  std::filesystem::remove(corrupt_graph_path, ignored);
  std::filesystem::remove(transformer_path, ignored);
  std::filesystem::remove(corrupt_transformer_path, ignored);
  std::filesystem::remove(corrupt_transformer_dtype_path, ignored);
  for (const auto& path : corrupt_transformer_metadata_paths)
    std::filesystem::remove(path, ignored);
  std::filesystem::remove(base, ignored);
}

VIDFAB_TEST(vulkan_gemm_dispatch_geometry) {
  using vidfab::vulkan::detail::GemmDispatchGeometry;
  using vidfab::vulkan::detail::gemm_dispatch_geometry;
  GemmDispatchGeometry geometry{99, 99};
  CHECK(!gemm_dispatch_geometry(0, 1, 16, 16, 8, 8, &geometry));
  CHECK(!gemm_dispatch_geometry(1, 0, 16, 16, 8, 8, &geometry));
  CHECK(gemm_dispatch_geometry(1, 1, 16, 16, 8, 8, &geometry));
  CHECK(geometry.x == 1 && geometry.y == 1);
  CHECK(gemm_dispatch_geometry(15, 15, 16, 16, 8, 8, &geometry));
  CHECK(geometry.x == 1 && geometry.y == 1);
  CHECK(gemm_dispatch_geometry(16, 16, 16, 16, 8, 8, &geometry));
  CHECK(geometry.x == 1 && geometry.y == 1);
  CHECK(gemm_dispatch_geometry(17, 17, 16, 16, 8, 8, &geometry));
  CHECK(geometry.x == 2 && geometry.y == 2);
  CHECK(gemm_dispatch_geometry(128, 128, 16, 16, 8, 8, &geometry));
  CHECK(geometry.x == 8 && geometry.y == 8);
  CHECK(!gemm_dispatch_geometry(129, 128, 16, 16, 8, 8, &geometry));
  CHECK(!gemm_dispatch_geometry(128, 129, 16, 16, 8, 8, &geometry));
  CHECK(!gemm_dispatch_geometry(UINT64_MAX, UINT64_MAX, 16, 16,
                                UINT32_MAX, UINT32_MAX, &geometry));
  CHECK(!gemm_dispatch_geometry(1, 1, 0, 16, 8, 8, &geometry));
  CHECK(!gemm_dispatch_geometry(1, 1, 16, 16, 8, 8, nullptr));
}

int main() { return ::vidfab::test::run_all(); }
