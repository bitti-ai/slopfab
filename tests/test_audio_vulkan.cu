#include "harness.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <bcrypt.h>
#endif

#include "vidfab/cuda/audio_vae_kernels.cuh"
#include "vidfab/cuda/device.h"
#include "vidfab/audio/wav.h"
#include "vidfab/safetensors.h"
#include "vidfab/vae/audio_decoder.h"
#include "vidfab/vae/audio_primitives.h"
#include "vidfab/vulkan/audio_decoder.h"
#include "vidfab/vulkan/tensor.h"

namespace {

vidfab::TensorLayout layout(std::initializer_list<uint64_t> extents) {
  std::vector<uint64_t> shape(extents);
  return vidfab::TensorLayout::contiguous(shape.data(),
                                          static_cast<uint32_t>(shape.size()));
}

void check_exact(const std::vector<float>& cuda_values,
                 const std::vector<float>& vulkan_values,
                 const char* operation) {
  CHECK(cuda_values.size() == vulkan_values.size());
  for (size_t i = 0; i < cuda_values.size(); ++i) {
    uint32_t cuda_bits = 0, vulkan_bits = 0;
    std::memcpy(&cuda_bits, &cuda_values[i], sizeof(cuda_bits));
    std::memcpy(&vulkan_bits, &vulkan_values[i], sizeof(vulkan_bits));
    CHECK_MSG(cuda_bits == vulkan_bits,
              "%s CUDA/Vulkan mismatch at %zu: %08x != %08x", operation,
              i, cuda_bits, vulkan_bits);
  }
}

std::vector<float> values(size_t count, uint32_t multiplier, uint32_t modulus,
                          float scale) {
  std::vector<float> result(count);
  for (size_t i = 0; i < count; ++i)
    result[i] = float(int((i * multiplier) % modulus) - int(modulus / 2)) * scale;
  return result;
}

size_t count_subnormals(const std::vector<float>& values) {
  size_t count = 0;
  for (float value : values) {
    uint32_t bits = 0; std::memcpy(&bits, &value, sizeof(bits));
    count += (bits & 0x7f800000u) == 0u && (bits & 0x007fffffu) != 0u;
  }
  return count;
}

uint64_t fnv64(const std::vector<std::vector<float>>& tensors) {
  uint64_t hash = 1469598103934665603ull;
  for (const auto& tensor : tensors) {
    for (float value : tensor) {
      uint32_t bits = 0; std::memcpy(&bits, &value, sizeof(bits));
      for (unsigned byte = 0; byte < 4; ++byte) {
        hash ^= (bits >> (8u * byte)) & 0xffu;
        hash *= 1099511628211ull;
      }
    }
  }
  return hash;
}

#ifdef _WIN32
std::array<uint8_t, 32> mapping_sha256(const void* mapping, size_t bytes) {
  BCRYPT_ALG_HANDLE algorithm = nullptr;
  BCRYPT_HASH_HANDLE hash = nullptr;
  std::array<uint8_t, 32> digest{};
  auto fail = [&] {
    if (hash) BCryptDestroyHash(hash);
    if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
    throw std::runtime_error("audio primitive checkpoint SHA-256 failed");
  };
  if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM,
                                  nullptr, 0) < 0 ||
      BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) < 0) fail();
  const auto* cursor = static_cast<const uint8_t*>(mapping);
  while (bytes != 0) {
    const ULONG chunk = static_cast<ULONG>(std::min<size_t>(bytes, 64ull << 20));
    if (BCryptHashData(hash, const_cast<PUCHAR>(cursor), chunk, 0) < 0) fail();
    cursor += chunk; bytes -= chunk;
  }
  if (BCryptFinishHash(hash, digest.data(),
                       static_cast<ULONG>(digest.size()), 0) < 0) fail();
  BCryptDestroyHash(hash); BCryptCloseAlgorithmProvider(algorithm, 0);
  return digest;
}
#endif

}  // namespace

VIDFAB_TEST(cuda_vulkan_exact_audio_primitives) {
  using namespace vidfab;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !vulkan::Instance::available()) return;
  vulkan::Instance instance = vulkan::Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore ||
      !physical.front().info().shader_int64) return;
  vulkan::DeviceOptions device_options;
  device_options.enable_timeline_semaphore = true;
  device_options.enable_shader_int64 = true;
  vulkan::Device device = physical.front().create_device(device_options);
  vulkan::TensorContextOptions tensor_options;
  tensor_options.max_batch_operators = 32;
  vulkan::TensorContext vk(device, tensor_options);
  if (!vk.exact_audio_vae_primitives()) return;
  vk.require_exact_audio_vae_primitives();

  vae::AudioConv1DDesc conv{2, 3, 5, 19, 19, 5, 4, 2};
  conv.validate();
  std::vector<float> conv_x = values(conv.input_elements(), 17, 101, 1.0f / 64.0f);
  std::vector<float> conv_w = values(conv.weight_elements(), 29, 67, 1.0f / 128.0f);
  std::vector<float> conv_b = values(conv.out_channels, 7, 19, 1.0f / 32.0f);
  const uint32_t exceptional[] = {0x00000001u, 0x807fffffu, 0x7fc12345u};
  std::memcpy(conv_x.data() + 2, exceptional, sizeof(exceptional));
  std::memcpy(conv_w.data() + 7, exceptional, sizeof(exceptional));
  cuda::DeviceBuffer<float> c_conv_x(conv_x.size()), c_conv_w(conv_w.size()),
      c_conv_b(conv_b.size()), c_conv_y(conv.output_elements());
  c_conv_x.copy_from_host(conv_x.data(), conv_x.size());
  c_conv_w.copy_from_host(conv_w.data(), conv_w.size());
  c_conv_b.copy_from_host(conv_b.data(), conv_b.size());
  cuda::launch_conv1d(c_conv_x.get(), c_conv_w.get(), c_conv_b.get(),
                      c_conv_y.get(), conv.batch, conv.in_channels,
                      conv.out_channels, conv.length_in, conv.length_out,
                      conv.kernel, conv.padding, conv.dilation, nullptr);

  vulkan::DeviceTensor v_conv_x = vk.allocate(layout({2, 3, 19}));
  vulkan::DeviceTensor v_conv_w = vk.allocate(layout({5, 3, 5}));
  vulkan::DeviceTensor v_conv_b = vk.allocate(layout({5}));
  vulkan::DeviceTensor v_conv_y = vk.allocate(layout({2, 5, 19}));
  vk.upload(v_conv_x, conv_x.data(), conv_x.size());
  vk.upload(v_conv_w, conv_w.data(), conv_w.size());
  vk.upload(v_conv_b, conv_b.data(), conv_b.size());
  {
    vulkan::TensorBatch batch = vk.begin_batch();
    batch.audio_conv1d(v_conv_x, v_conv_w, &v_conv_b, v_conv_y, conv);
    batch.submit().wait();
  }
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> cuda_conv(conv.output_elements()),
      vk_conv(conv.output_elements());
  c_conv_y.copy_to_host(cuda_conv.data(), cuda_conv.size());
  vk.download(v_conv_y, vk_conv.data(), vk_conv.size());
  check_exact(cuda_conv, vk_conv, "Conv1D");

  vae::AudioConvTranspose1DDesc transpose{1, 4, 3, 11, 55, 9, 5, 2};
  transpose.validate();
  std::vector<float> trans_x = values(transpose.input_elements(), 13, 71, 1.0f / 64.0f);
  std::vector<float> trans_w = values(transpose.weight_elements(), 23, 83, 1.0f / 128.0f);
  std::vector<float> trans_b = values(transpose.out_channels, 3, 11, 1.0f / 32.0f);
  cuda::DeviceBuffer<float> c_trans_x(trans_x.size()), c_trans_w(trans_w.size()),
      c_trans_b(trans_b.size()), c_trans_y(transpose.output_elements());
  c_trans_x.copy_from_host(trans_x.data(), trans_x.size());
  c_trans_w.copy_from_host(trans_w.data(), trans_w.size());
  c_trans_b.copy_from_host(trans_b.data(), trans_b.size());
  cuda::launch_conv_transpose1d(
      c_trans_x.get(), c_trans_w.get(), c_trans_b.get(), c_trans_y.get(),
      transpose.batch, transpose.in_channels, transpose.out_channels,
      transpose.length_in, transpose.length_out, transpose.kernel,
      transpose.stride, transpose.padding, nullptr);
  vulkan::DeviceTensor v_trans_x = vk.allocate(layout({1, 4, 11}));
  vulkan::DeviceTensor v_trans_w = vk.allocate(layout({4, 3, 9}));
  vulkan::DeviceTensor v_trans_b = vk.allocate(layout({3}));
  vulkan::DeviceTensor v_trans_y = vk.allocate(layout({1, 3, 55}));
  vk.upload(v_trans_x, trans_x.data(), trans_x.size());
  vk.upload(v_trans_w, trans_w.data(), trans_w.size());
  vk.upload(v_trans_b, trans_b.data(), trans_b.size());
  {
    vulkan::TensorBatch batch = vk.begin_batch();
    batch.audio_conv_transpose1d(v_trans_x, v_trans_w, &v_trans_b,
                                 v_trans_y, transpose);
    batch.submit().wait();
  }
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> cuda_trans(transpose.output_elements()),
      vk_trans(transpose.output_elements());
  c_trans_y.copy_to_host(cuda_trans.data(), cuda_trans.size());
  vk.download(v_trans_y, vk_trans.data(), vk_trans.size());
  check_exact(cuda_trans, vk_trans, "ConvTranspose1D");

  constexpr uint32_t batch_count = 2, channels = 3, length = 67;
  const size_t snake_count = size_t(batch_count) * channels * length;
  std::vector<float> snake_x = values(snake_count, 31, 127, 1.0f / 48.0f);
  std::vector<float> log_alpha{-0.75f, 0.0f, 0.625f};
  std::vector<float> log_beta{-0.5f, 0.25f, 0.875f};
  cuda::DeviceBuffer<float> c_snake(snake_count), c_alpha(channels), c_beta(channels);
  c_snake.copy_from_host(snake_x.data(), snake_count);
  c_alpha.copy_from_host(log_alpha.data(), channels);
  c_beta.copy_from_host(log_beta.data(), channels);
  cuda::launch_snake_beta(c_snake.get(), c_alpha.get(), c_beta.get(),
                          batch_count, channels, length, nullptr);
  vulkan::DeviceTensor v_snake = vk.allocate(layout({batch_count, channels, length}));
  vulkan::DeviceTensor v_alpha = vk.allocate(layout({channels}));
  vulkan::DeviceTensor v_beta = vk.allocate(layout({channels}));
  vk.upload(v_snake, snake_x.data(), snake_count);
  vk.upload(v_alpha, log_alpha.data(), channels);
  vk.upload(v_beta, log_beta.data(), channels);
  {
    vulkan::TensorBatch batch = vk.begin_batch();
    batch.audio_snake_beta_inplace(v_snake, v_alpha, v_beta, batch_count,
                                   channels, length);
    batch.submit().wait();
  }
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> cuda_snake(snake_count), vk_snake(snake_count);
  c_snake.copy_to_host(cuda_snake.data(), snake_count);
  vk.download(v_snake, vk_snake.data(), snake_count);
  check_exact(cuda_snake, vk_snake, "SnakeBeta");

  // Totality probes: no exceptional value may reach an undefined float-to-int
  // conversion in exp/sin range reduction, and all NaNs use one payload.
  constexpr uint32_t exceptional_channels = 10, exceptional_length = 6;
  std::vector<float> exceptional_x(exceptional_channels * exceptional_length);
  for (size_t i = 0; i < exceptional_x.size(); ++i)
    exceptional_x[i] = float(int(i % 13) - 6) / 8.0f;
  const uint32_t x_bits[] = {
      0x00000001u, 0x807fffffu, 0x7fc12345u,
      0x7f800000u, 0xff800000u, 0x60ad78ecu};
  std::memcpy(exceptional_x.data(), x_bits, sizeof(x_bits));
  const uint32_t alpha_bits[] = {
      0x00000001u, 0x7fa54321u, 0x00000000u, 0x7f800000u,
      0xff800000u, 0x60ad78ecu, 0xe0ad78ecu, 0x3e800000u,
      0x3e800000u, 0x3e800000u};
  const uint32_t beta_bits[] = {
      0x80000001u, 0x3e800000u, 0x7fcabcdeu, 0x3e800000u,
      0x3e800000u, 0x3e800000u, 0x3e800000u, 0x7f800000u,
      0xff800000u, 0x60ad78ecu};
  std::vector<float> exceptional_alpha(exceptional_channels),
      exceptional_beta(exceptional_channels);
  std::memcpy(exceptional_alpha.data(), alpha_bits, sizeof(alpha_bits));
  std::memcpy(exceptional_beta.data(), beta_bits, sizeof(beta_bits));
  cuda::DeviceBuffer<float> c_exceptional(exceptional_x.size()),
      c_exceptional_alpha(exceptional_channels),
      c_exceptional_beta(exceptional_channels);
  c_exceptional.copy_from_host(exceptional_x.data(), exceptional_x.size());
  c_exceptional_alpha.copy_from_host(exceptional_alpha.data(), exceptional_channels);
  c_exceptional_beta.copy_from_host(exceptional_beta.data(), exceptional_channels);
  cuda::launch_snake_beta(
      c_exceptional.get(), c_exceptional_alpha.get(), c_exceptional_beta.get(),
      1, exceptional_channels, exceptional_length, nullptr);
  vulkan::DeviceTensor v_exceptional =
      vk.allocate(layout({1, exceptional_channels, exceptional_length}));
  vulkan::DeviceTensor v_exceptional_alpha =
      vk.allocate(layout({exceptional_channels}));
  vulkan::DeviceTensor v_exceptional_beta =
      vk.allocate(layout({exceptional_channels}));
  vk.upload(v_exceptional, exceptional_x.data(), exceptional_x.size());
  vk.upload(v_exceptional_alpha, exceptional_alpha.data(), exceptional_channels);
  vk.upload(v_exceptional_beta, exceptional_beta.data(), exceptional_channels);
  {
    vulkan::TensorBatch batch = vk.begin_batch();
    batch.audio_snake_beta_inplace(
        v_exceptional, v_exceptional_alpha, v_exceptional_beta, 1,
        exceptional_channels, exceptional_length);
    batch.submit().wait();
  }
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> cuda_exceptional(exceptional_x.size()),
      vk_exceptional(exceptional_x.size());
  c_exceptional.copy_to_host(cuda_exceptional.data(), cuda_exceptional.size());
  vk.download(v_exceptional, vk_exceptional.data(), vk_exceptional.size());
  check_exact(cuda_exceptional, vk_exceptional, "exceptional SnakeBeta");
  for (float output : vk_exceptional) {
    uint32_t bits = 0; std::memcpy(&bits, &output, sizeof(bits));
    if ((bits & 0x7fffffffu) > 0x7f800000u) CHECK(bits == 0x7fc00000u);
  }
  for (uint32_t index : {2u, 3u, 4u, 5u}) {
    uint32_t bits = 0;
    std::memcpy(&bits, &vk_exceptional[index], sizeof(bits));
    CHECK(bits == 0x7fc00000u);
  }

  constexpr uint32_t aa_batch = 1, aa_channels = 3, aa_length = 19;
  const size_t aa_input_count = size_t(aa_batch) * aa_channels * aa_length;
  const size_t aa_up_count = aa_input_count * 2;
  std::vector<float> aa_x = values(aa_input_count, 11, 97, 1.0f / 64.0f);
  std::vector<float> filter{
      -0.0012f, -0.0045f, 0.0123f, 0.0521f, 0.1432f, 0.2981f,
       0.2981f,  0.1432f, 0.0521f, 0.0123f, -0.0045f, -0.0012f};
  cuda::DeviceBuffer<float> c_aa_x(aa_input_count), c_filter(12),
      c_aa_up(aa_up_count), c_aa_down(aa_input_count);
  c_aa_x.copy_from_host(aa_x.data(), aa_input_count);
  c_filter.copy_from_host(filter.data(), filter.size());
  cuda::launch_aa_upsample_snake(
      c_aa_x.get(), c_filter.get(), c_alpha.get(), c_beta.get(), c_aa_up.get(),
      aa_batch, aa_channels, aa_length, nullptr);
  cuda::launch_aa_downsample(c_aa_up.get(), c_filter.get(), c_aa_down.get(),
                             aa_batch, aa_channels, aa_length * 2, aa_length,
                             nullptr);
  vulkan::DeviceTensor v_aa_x = vk.allocate(layout({aa_batch, aa_channels, aa_length}));
  vulkan::DeviceTensor v_filter = vk.allocate(layout({12}));
  vulkan::DeviceTensor v_aa_up = vk.allocate(layout({aa_batch, aa_channels, aa_length * 2}));
  vulkan::DeviceTensor v_aa_down = vk.allocate(layout({aa_batch, aa_channels, aa_length}));
  vk.upload(v_aa_x, aa_x.data(), aa_input_count);
  vk.upload(v_filter, filter.data(), filter.size());
  {
    vulkan::TensorBatch batch = vk.begin_batch();
    batch.audio_aa_upsample_snake(v_aa_x, v_filter, v_alpha, v_beta,
                                  v_aa_up, aa_batch, aa_channels, aa_length);
    batch.audio_aa_downsample(v_aa_up, v_filter, v_aa_down, aa_batch,
                              aa_channels, aa_length * 2, aa_length);
    batch.submit().wait();
  }
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> cuda_up(aa_up_count), vk_up(aa_up_count),
      cuda_down(aa_input_count), vk_down(aa_input_count);
  c_aa_up.copy_to_host(cuda_up.data(), aa_up_count);
  c_aa_down.copy_to_host(cuda_down.data(), aa_input_count);
  vk.download(v_aa_up, vk_up.data(), aa_up_count);
  vk.download(v_aa_down, vk_down.data(), aa_input_count);
  check_exact(cuda_up, vk_up, "AA upsample + SnakeBeta");
  check_exact(cuda_down, vk_down, "AA downsample");

  constexpr uint32_t frames = 259;
  std::vector<float> element_x = values(size_t(2) * frames, 19, 113, 1.0f / 32.0f);
  std::vector<float> element_y = values(size_t(2) * frames, 37, 109, 1.0f / 64.0f);
  const uint32_t element_exceptions[] = {
      0x00000001u, 0x807fffffu, 0x7fc12345u, 0x7f800000u, 0xff800000u};
  std::memcpy(element_x.data(), element_exceptions,
              sizeof(element_exceptions));
  cuda::DeviceBuffer<float> c_element_x(element_x.size()), c_element_y(element_y.size()),
      c_interleaved(element_x.size());
  c_element_x.copy_from_host(element_x.data(), element_x.size());
  c_element_y.copy_from_host(element_y.data(), element_y.size());
  cuda::launch_add_inplace(c_element_x.get(), c_element_y.get(), element_x.size(), nullptr);
  cuda::launch_scale_inplace(c_element_x.get(), 1.0f / 3.0f, element_x.size(), nullptr);
  cuda::launch_clamp_inplace(c_element_x.get(), -1.0f, 1.0f, element_x.size(), nullptr);
  cuda::launch_interleave(c_element_x.get(), c_interleaved.get(), 2, frames, nullptr);
  vulkan::DeviceTensor v_element_x = vk.allocate(layout({2, 1, frames}));
  vulkan::DeviceTensor v_element_y = vk.allocate(layout({2, 1, frames}));
  vulkan::DeviceTensor v_interleaved = vk.allocate(layout({frames, 2}));
  vk.upload(v_element_x, element_x.data(), element_x.size());
  vk.upload(v_element_y, element_y.data(), element_y.size());
  {
    vulkan::TensorBatch invalid = vk.begin_batch();
    const uint32_t remaining = invalid.remaining_operator_capacity();
    bool alias_rejected = false;
    try { invalid.audio_add_inplace(v_element_x, v_element_x); }
    catch (const std::invalid_argument&) { alias_rejected = true; }
    CHECK(alias_rejected);
    CHECK(invalid.remaining_operator_capacity() == remaining);
    invalid.audio_add_inplace(v_element_x, v_element_y);
    invalid.audio_scale_inplace(v_element_x, 1.0f / 3.0f);
    invalid.audio_clamp_inplace(v_element_x, -1.0f, 1.0f);
    invalid.audio_interleave(v_element_x, v_interleaved, 2, frames);
    invalid.submit().wait();
  }
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> cuda_element(element_x.size()), vk_element(element_x.size()),
      cuda_interleaved(element_x.size()), vk_interleaved(element_x.size());
  c_element_x.copy_to_host(cuda_element.data(), element_x.size());
  c_interleaved.copy_to_host(cuda_interleaved.data(), element_x.size());
  vk.download(v_element_x, vk_element.data(), element_x.size());
  vk.download(v_interleaved, vk_interleaved.data(), element_x.size());
  check_exact(cuda_element, vk_element, "add/scale/clamp");
  check_exact(cuda_interleaved, vk_interleaved, "interleave");

  const uint64_t warm_reserved = vk.reserved_bytes();
  const uint64_t warm_descriptors = vk.descriptor_set_allocations();
  vk.upload(v_element_x, element_x.data(), element_x.size());
  {
    vulkan::TensorBatch repeat = vk.begin_batch();
    repeat.audio_add_inplace(v_element_x, v_element_y);
    repeat.audio_scale_inplace(v_element_x, 1.0f / 3.0f);
    repeat.audio_clamp_inplace(v_element_x, -1.0f, 1.0f);
    repeat.audio_interleave(v_element_x, v_interleaved, 2, frames);
    repeat.submit().wait();
  }
  CHECK(vk.reserved_bytes() == warm_reserved);
  CHECK(vk.descriptor_set_allocations() == warm_descriptors);
  std::printf("  exact audio primitives: persistent/reserved %.1f MiB, descriptors %llu\n",
              double(vk.pooled_used_bytes()) / 1048576.0,
              static_cast<unsigned long long>(vk.descriptor_set_allocations()));
}

VIDFAB_TEST(cuda_vulkan_exact_audio_real_checkpoint_primitives) {
  using namespace vidfab;
  if (!std::getenv("VIDFAB_AUDIO_VAE_REAL")) return;
  const std::filesystem::path path =
      "weights/vae/minimax_h3_audio_vae_fp32.safetensors";
  if (!std::filesystem::exists(path)) return;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !vulkan::Instance::available()) return;
  vulkan::Instance instance = vulkan::Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore ||
      !physical.front().info().shader_int64) return;

  SafeTensors checkpoint;
  checkpoint.open(path.string());
  CHECK(checkpoint.file_size() == 605254808u);
  CHECK(checkpoint.tensor_count() == 917u);
#ifdef _WIN32
  const std::array<uint8_t, 32> expected_sha{
      0x8e, 0x50, 0x5d, 0x95, 0xdd, 0x15, 0x61, 0xd4,
      0x7a, 0xbd, 0x43, 0xd4, 0x23, 0x8f, 0xd4, 0x0d,
      0x9b, 0xb1, 0xae, 0x9e, 0x14, 0x7e, 0xd0, 0xa4,
      0xcb, 0xa7, 0x78, 0xd7, 0x6a, 0xe4, 0xdb, 0x48};
  CHECK(mapping_sha256(checkpoint.mapping_base(), checkpoint.file_size()) ==
        expected_sha);
#endif

  vae::AudioConvWeights conv_weights = vae::load_audio_conv_weights(
      checkpoint, "dec_in_proj", {2048, 32, 1}, 2048, true);
  vae::AudioConvWeights transpose_weights = vae::load_audio_conv_weights(
      checkpoint, "decoder.ups.0.0", {1024, 512, 9}, 512, true);
  vae::AudioConvWeights dilated_weights = vae::load_audio_conv_weights(
      checkpoint, "decoder.resblocks.2.convs1.2", {512, 512, 11}, 512,
      true);
  CHECK(!conv_weights.folded_weight_norm);
  CHECK(!transpose_weights.folded_weight_norm);
  CHECK(!dilated_weights.folded_weight_norm);
  std::vector<float> log_alpha = vae::load_audio_f32_tensor(
      checkpoint, "decoder.activation_post.act.alpha", {8});
  std::vector<float> log_beta = vae::load_audio_f32_tensor(
      checkpoint, "decoder.activation_post.act.beta", {8});
  std::vector<float> up_filter = vae::load_audio_f32_tensor(
      checkpoint, "decoder.activation_post.upsample.filter", {1, 1, 12});
  std::vector<float> down_filter = vae::load_audio_f32_tensor(
      checkpoint, "decoder.activation_post.downsample.lowpass.filter",
      {1, 1, 12});
  const size_t raw_subnormals =
      count_subnormals(conv_weights.weight) +
      count_subnormals(conv_weights.bias) +
      count_subnormals(transpose_weights.weight) +
      count_subnormals(transpose_weights.bias) +
      count_subnormals(dilated_weights.weight) +
      count_subnormals(dilated_weights.bias) + count_subnormals(log_alpha) +
      count_subnormals(log_beta) + count_subnormals(up_filter) +
      count_subnormals(down_filter);
  CHECK(raw_subnormals == 0u);

  vulkan::DeviceOptions device_options;
  device_options.enable_timeline_semaphore = true;
  device_options.enable_shader_int64 = true;
  vulkan::Device device = physical.front().create_device(device_options);
  vulkan::TensorContext vk(device);
  if (!vk.exact_audio_vae_primitives()) return;

  auto upload = [&](const TensorLayout& tensor_layout,
                    const std::vector<float>& host) {
    vulkan::DeviceTensor tensor = vk.allocate(tensor_layout);
    vk.upload(tensor, host.data(), host.size());
    return tensor;
  };

  const vae::AudioConv1DDesc conv{2, 32, 2048, 3, 3, 1, 0, 1};
  std::vector<float> conv_input = values(conv.input_elements(), 37, 509,
                                         1.0f / 256.0f);
  cuda::DeviceBuffer<float> c_conv_input(conv.input_elements()),
      c_conv_weight(conv.weight_elements()), c_conv_bias(2048),
      c_conv_output(conv.output_elements());
  c_conv_input.copy_from_host(conv_input.data(), conv_input.size());
  c_conv_weight.copy_from_host(conv_weights.weight.data(),
                               conv_weights.weight.size());
  c_conv_bias.copy_from_host(conv_weights.bias.data(), conv_weights.bias.size());
  cuda::launch_conv1d(
      c_conv_input.get(), c_conv_weight.get(), c_conv_bias.get(),
      c_conv_output.get(), conv.batch, conv.in_channels, conv.out_channels,
      conv.length_in, conv.length_out, conv.kernel, conv.padding,
      conv.dilation, nullptr);
  vulkan::DeviceTensor v_conv_input = upload(layout({2, 32, 3}), conv_input);
  vulkan::DeviceTensor v_conv_weight =
      upload(layout({2048, 32, 1}), conv_weights.weight);
  vulkan::DeviceTensor v_conv_bias = upload(layout({2048}), conv_weights.bias);
  vulkan::DeviceTensor v_conv_output = vk.allocate(layout({2, 2048, 3}));
  {
    vulkan::TensorBatch batch = vk.begin_batch();
    batch.audio_conv1d(v_conv_input, v_conv_weight, &v_conv_bias,
                       v_conv_output, conv);
    batch.submit().wait();
  }

  const vae::AudioConvTranspose1DDesc transpose{
      1, 1024, 512, 3, 15, 9, 5, 2};
  std::vector<float> transpose_input = values(
      transpose.input_elements(), 41, 257, 1.0f / 256.0f);
  cuda::DeviceBuffer<float> c_transpose_input(transpose.input_elements()),
      c_transpose_weight(transpose.weight_elements()), c_transpose_bias(512),
      c_transpose_output(transpose.output_elements());
  c_transpose_input.copy_from_host(transpose_input.data(),
                                   transpose_input.size());
  c_transpose_weight.copy_from_host(transpose_weights.weight.data(),
                                    transpose_weights.weight.size());
  c_transpose_bias.copy_from_host(transpose_weights.bias.data(),
                                  transpose_weights.bias.size());
  cuda::launch_conv_transpose1d(
      c_transpose_input.get(), c_transpose_weight.get(), c_transpose_bias.get(),
      c_transpose_output.get(), transpose.batch, transpose.in_channels,
      transpose.out_channels, transpose.length_in, transpose.length_out,
      transpose.kernel, transpose.stride, transpose.padding, nullptr);
  vulkan::DeviceTensor v_transpose_input =
      upload(layout({1, 1024, 3}), transpose_input);
  vulkan::DeviceTensor v_transpose_weight =
      upload(layout({1024, 512, 9}), transpose_weights.weight);
  vulkan::DeviceTensor v_transpose_bias =
      upload(layout({512}), transpose_weights.bias);
  vulkan::DeviceTensor v_transpose_output =
      vk.allocate(layout({1, 512, 15}));
  const auto transpose_begin = std::chrono::steady_clock::now();
  {
    vulkan::TensorBatch batch = vk.begin_batch();
    batch.audio_conv_transpose1d(
        v_transpose_input, v_transpose_weight, &v_transpose_bias,
        v_transpose_output, transpose);
    batch.submit().wait();
  }
  const double transpose_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - transpose_begin).count();

  const vae::AudioConv1DDesc dilated{1, 512, 512, 15, 15, 11, 25, 5};
  std::vector<float> dilated_input = values(
      dilated.input_elements(), 47, 263, 1.0f / 256.0f);
  cuda::DeviceBuffer<float> c_dilated_input(dilated.input_elements()),
      c_dilated_weight(dilated.weight_elements()), c_dilated_bias(512),
      c_dilated_output(dilated.output_elements());
  c_dilated_input.copy_from_host(dilated_input.data(), dilated_input.size());
  c_dilated_weight.copy_from_host(dilated_weights.weight.data(),
                                  dilated_weights.weight.size());
  c_dilated_bias.copy_from_host(dilated_weights.bias.data(),
                                dilated_weights.bias.size());
  cuda::launch_conv1d(
      c_dilated_input.get(), c_dilated_weight.get(), c_dilated_bias.get(),
      c_dilated_output.get(), dilated.batch, dilated.in_channels,
      dilated.out_channels, dilated.length_in, dilated.length_out,
      dilated.kernel, dilated.padding, dilated.dilation, nullptr);
  vulkan::DeviceTensor v_dilated_input =
      upload(layout({1, 512, 15}), dilated_input);
  vulkan::DeviceTensor v_dilated_weight =
      upload(layout({512, 512, 11}), dilated_weights.weight);
  vulkan::DeviceTensor v_dilated_bias =
      upload(layout({512}), dilated_weights.bias);
  vulkan::DeviceTensor v_dilated_output =
      vk.allocate(layout({1, 512, 15}));
  const auto dilated_begin = std::chrono::steady_clock::now();
  {
    vulkan::TensorBatch batch = vk.begin_batch();
    batch.audio_conv1d(v_dilated_input, v_dilated_weight, &v_dilated_bias,
                       v_dilated_output, dilated);
    batch.submit().wait();
  }
  const double dilated_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - dilated_begin).count();

  constexpr uint32_t aa_batch = 2, aa_channels = 8, aa_length = 259;
  const size_t aa_count = size_t(aa_batch) * aa_channels * aa_length;
  std::vector<float> aa_input = values(aa_count, 43, 311, 1.0f / 128.0f);
  cuda::DeviceBuffer<float> c_aa_input(aa_count), c_up_filter(12),
      c_down_filter(12), c_alpha(8), c_beta(8), c_aa_up(aa_count * 2),
      c_aa_down(aa_count);
  c_aa_input.copy_from_host(aa_input.data(), aa_input.size());
  c_up_filter.copy_from_host(up_filter.data(), up_filter.size());
  c_down_filter.copy_from_host(down_filter.data(), down_filter.size());
  c_alpha.copy_from_host(log_alpha.data(), log_alpha.size());
  c_beta.copy_from_host(log_beta.data(), log_beta.size());
  cuda::launch_aa_upsample_snake(
      c_aa_input.get(), c_up_filter.get(), c_alpha.get(), c_beta.get(),
      c_aa_up.get(), aa_batch, aa_channels, aa_length, nullptr);
  cuda::launch_aa_downsample(c_aa_up.get(), c_down_filter.get(),
                             c_aa_down.get(), aa_batch, aa_channels,
                             aa_length * 2, aa_length, nullptr);
  vulkan::DeviceTensor v_aa_input =
      upload(layout({aa_batch, aa_channels, aa_length}), aa_input);
  vulkan::DeviceTensor v_up_filter = upload(layout({12}), up_filter);
  vulkan::DeviceTensor v_down_filter = upload(layout({12}), down_filter);
  vulkan::DeviceTensor v_alpha = upload(layout({8}), log_alpha);
  vulkan::DeviceTensor v_beta = upload(layout({8}), log_beta);
  vulkan::DeviceTensor v_aa_up =
      vk.allocate(layout({aa_batch, aa_channels, aa_length * 2}));
  vulkan::DeviceTensor v_aa_down =
      vk.allocate(layout({aa_batch, aa_channels, aa_length}));
  {
    vulkan::TensorBatch batch = vk.begin_batch();
    batch.audio_aa_upsample_snake(v_aa_input, v_up_filter, v_alpha, v_beta,
                                  v_aa_up, aa_batch, aa_channels, aa_length);
    batch.audio_aa_downsample(v_aa_up, v_down_filter, v_aa_down, aa_batch,
                              aa_channels, aa_length * 2, aa_length);
    batch.submit().wait();
  }
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> cuda_conv(conv.output_elements()),
      vk_conv(conv.output_elements()),
      cuda_transpose(transpose.output_elements()),
      vk_transpose(transpose.output_elements()),
      cuda_dilated(dilated.output_elements()),
      vk_dilated(dilated.output_elements()), cuda_up(aa_count * 2),
      vk_up(aa_count * 2), cuda_down(aa_count), vk_down(aa_count);
  c_conv_output.copy_to_host(cuda_conv.data(), cuda_conv.size());
  c_transpose_output.copy_to_host(cuda_transpose.data(), cuda_transpose.size());
  c_dilated_output.copy_to_host(cuda_dilated.data(), cuda_dilated.size());
  c_aa_up.copy_to_host(cuda_up.data(), cuda_up.size());
  c_aa_down.copy_to_host(cuda_down.data(), cuda_down.size());
  vk.download(v_conv_output, vk_conv.data(), vk_conv.size());
  vk.download(v_transpose_output, vk_transpose.data(), vk_transpose.size());
  vk.download(v_dilated_output, vk_dilated.data(), vk_dilated.size());
  vk.download(v_aa_up, vk_up.data(), vk_up.size());
  vk.download(v_aa_down, vk_down.data(), vk_down.size());
  check_exact(cuda_conv, vk_conv, "real dec_in_proj");
  check_exact(cuda_transpose, vk_transpose, "real upsample transpose");
  check_exact(cuda_dilated, vk_dilated, "real dilated Conv1D");
  check_exact(cuda_up, vk_up, "real activation up+SnakeBeta");
  check_exact(cuda_down, vk_down, "real activation downsample");
  const uint64_t digest = fnv64(
      {vk_conv, vk_transpose, vk_dilated, vk_up, vk_down});
  CHECK(digest == 0x2f78225c9577a73bull);
  const uint64_t logical_bytes = sizeof(float) * (
      conv.input_elements() + conv.weight_elements() + conv.out_channels +
      conv.output_elements() + transpose.input_elements() +
      transpose.weight_elements() + transpose.out_channels +
      transpose.output_elements() + dilated.input_elements() +
      dilated.weight_elements() + dilated.out_channels +
      dilated.output_elements() + aa_count + 12u + 12u + 8u + 8u +
      aa_count * 2u + aa_count);
  CHECK(vk.pooled_used_bytes() >= logical_bytes);
  CHECK(vk.pooled_used_bytes() <= logical_bytes + (40ull << 20));
  CHECK(vk.reserved_bytes() <= (80ull << 20));
  const uint64_t stable_reserved = vk.reserved_bytes();
  const uint64_t stable_descriptors = vk.descriptor_set_allocations();
  {
    vulkan::TensorBatch batch = vk.begin_batch();
    batch.audio_conv_transpose1d(
        v_transpose_input, v_transpose_weight, &v_transpose_bias,
        v_transpose_output, transpose);
    batch.submit().wait();
  }
  CHECK(vk.reserved_bytes() == stable_reserved);
  CHECK(vk.descriptor_set_allocations() == stable_descriptors);

  // Full production temporal regime used for a 10.125-second decode. The
  // transpose and residual convolution share one device-resident batch.
  constexpr uint32_t production_length = 405;
  const vae::AudioConv1DDesc production_conv{
      2, 32, 2048, production_length, production_length, 1, 0, 1};
  const vae::AudioConvTranspose1DDesc production_transpose{
      2, 1024, 512, production_length, production_length * 5, 9, 5, 2};
  const vae::AudioConv1DDesc production_residual{
      2, 512, 512, production_length * 5, production_length * 5,
      11, 25, 5};
  std::vector<float> production_conv_input = values(
      production_conv.input_elements(), 53, 509, 1.0f / 256.0f);
  std::vector<float> production_transpose_input = values(
      production_transpose.input_elements(), 59, 521, 1.0f / 256.0f);
  cuda::DeviceBuffer<float> c_production_conv_input(
      production_conv.input_elements());
  cuda::DeviceBuffer<float> c_production_conv_output(
      production_conv.output_elements());
  cuda::DeviceBuffer<float> c_production_transpose_input(
      production_transpose.input_elements());
  cuda::DeviceBuffer<float> c_production_transpose_output(
      production_transpose.output_elements());
  cuda::DeviceBuffer<float> c_production_residual_output(
      production_residual.output_elements());
  c_production_conv_input.copy_from_host(production_conv_input.data(),
                                         production_conv_input.size());
  c_production_transpose_input.copy_from_host(
      production_transpose_input.data(), production_transpose_input.size());
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const auto cuda_conv_begin = std::chrono::steady_clock::now();
  cuda::launch_conv1d(
      c_production_conv_input.get(), c_conv_weight.get(), c_conv_bias.get(),
      c_production_conv_output.get(), production_conv.batch,
      production_conv.in_channels, production_conv.out_channels,
      production_conv.length_in, production_conv.length_out,
      production_conv.kernel, production_conv.padding,
      production_conv.dilation, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const double cuda_conv_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - cuda_conv_begin).count();
  const auto cuda_stage_begin = std::chrono::steady_clock::now();
  cuda::launch_conv_transpose1d(
      c_production_transpose_input.get(), c_transpose_weight.get(),
      c_transpose_bias.get(), c_production_transpose_output.get(),
      production_transpose.batch, production_transpose.in_channels,
      production_transpose.out_channels, production_transpose.length_in,
      production_transpose.length_out, production_transpose.kernel,
      production_transpose.stride, production_transpose.padding, nullptr);
  cuda::launch_conv1d(
      c_production_transpose_output.get(), c_dilated_weight.get(),
      c_dilated_bias.get(), c_production_residual_output.get(),
      production_residual.batch, production_residual.in_channels,
      production_residual.out_channels, production_residual.length_in,
      production_residual.length_out, production_residual.kernel,
      production_residual.padding, production_residual.dilation, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const double cuda_stage_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - cuda_stage_begin).count();

  const uint64_t before_production_used = vk.pooled_used_bytes();
  const uint64_t before_production_reserved = vk.reserved_bytes();
  vulkan::DeviceTensor v_production_conv_input = upload(
      layout({2, 32, production_length}), production_conv_input);
  vulkan::DeviceTensor v_production_conv_output = vk.allocate(
      layout({2, 2048, production_length}));
  vulkan::DeviceTensor v_production_transpose_input = upload(
      layout({2, 1024, production_length}), production_transpose_input);
  vulkan::DeviceTensor v_production_transpose_output = vk.allocate(
      layout({2, 512, production_length * 5}));
  vulkan::DeviceTensor v_production_residual_output = vk.allocate(
      layout({2, 512, production_length * 5}));
  const auto vk_conv_begin = std::chrono::steady_clock::now();
  {
    vulkan::TensorBatch batch = vk.begin_batch();
    batch.audio_conv1d(v_production_conv_input, v_conv_weight, &v_conv_bias,
                       v_production_conv_output, production_conv);
    batch.submit().wait();
  }
  const double vk_conv_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - vk_conv_begin).count();
  const auto vk_stage_begin = std::chrono::steady_clock::now();
  {
    vulkan::TensorBatch batch = vk.begin_batch();
    batch.audio_conv_transpose1d(
        v_production_transpose_input, v_transpose_weight, &v_transpose_bias,
        v_production_transpose_output, production_transpose);
    batch.audio_conv1d(
        v_production_transpose_output, v_dilated_weight, &v_dilated_bias,
        v_production_residual_output, production_residual);
    batch.submit().wait();
  }
  const double vk_stage_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - vk_stage_begin).count();
  std::vector<float> cuda_production_conv(production_conv.output_elements()),
      vk_production_conv(production_conv.output_elements()),
      cuda_production_transpose(production_transpose.output_elements()),
      vk_production_transpose(production_transpose.output_elements()),
      cuda_production_residual(production_residual.output_elements()),
      vk_production_residual(production_residual.output_elements());
  c_production_conv_output.copy_to_host(cuda_production_conv.data(),
                                        cuda_production_conv.size());
  c_production_transpose_output.copy_to_host(cuda_production_transpose.data(),
                                             cuda_production_transpose.size());
  c_production_residual_output.copy_to_host(cuda_production_residual.data(),
                                            cuda_production_residual.size());
  vk.download(v_production_conv_output, vk_production_conv.data(),
              vk_production_conv.size());
  vk.download(v_production_transpose_output, vk_production_transpose.data(),
              vk_production_transpose.size());
  vk.download(v_production_residual_output, vk_production_residual.data(),
              vk_production_residual.size());
  check_exact(cuda_production_conv, vk_production_conv,
              "A405 input projection");
  check_exact(cuda_production_transpose, vk_production_transpose,
              "A405 transpose 405->2025");
  check_exact(cuda_production_residual, vk_production_residual,
              "A405 stereo residual");
  const uint64_t production_digest = fnv64(
      {vk_production_conv, vk_production_transpose, vk_production_residual});
  CHECK(production_digest == 0x2a3eeba122112082ull);
  const uint64_t production_logical_bytes = sizeof(float) * (
      production_conv.input_elements() + production_conv.output_elements() +
      production_transpose.input_elements() +
      production_transpose.output_elements() +
      production_residual.output_elements());
  const uint64_t production_used_delta =
      vk.pooled_used_bytes() - before_production_used;
  CHECK(production_used_delta >= production_logical_bytes);
  CHECK(production_used_delta <= production_logical_bytes + (8ull << 20));
  CHECK(vk.reserved_bytes() <= before_production_reserved + (32ull << 20));
  const uint64_t production_stable_reserved = vk.reserved_bytes();
  const uint64_t production_stable_descriptors =
      vk.descriptor_set_allocations();
  {
    vulkan::TensorBatch batch = vk.begin_batch();
    batch.audio_conv_transpose1d(
        v_production_transpose_input, v_transpose_weight, &v_transpose_bias,
        v_production_transpose_output, production_transpose);
    batch.audio_conv1d(
        v_production_transpose_output, v_dilated_weight, &v_dilated_bias,
        v_production_residual_output, production_residual);
    batch.submit().wait();
  }
  CHECK(vk.reserved_bytes() == production_stable_reserved);
  CHECK(vk.descriptor_set_allocations() == production_stable_descriptors);
  std::printf(
      "  real audio primitives: raw subnormals %zu, FNV64 %016llx, first transpose %.2f ms, dilated k11/d5 %.2f ms, pool %.1f/%.1f MiB, descriptors %llu\n"
      "  A405 stereo: FNV64 %016llx, input projection CUDA/Vulkan %.2f/%.2f ms, transpose+residual CUDA/Vulkan %.2f/%.2f ms, production delta %.1f MiB\n",
      raw_subnormals, static_cast<unsigned long long>(digest), transpose_ms,
      dilated_ms,
      double(vk.pooled_used_bytes()) / 1048576.0,
      double(vk.reserved_bytes()) / 1048576.0,
      static_cast<unsigned long long>(vk.descriptor_set_allocations()),
      static_cast<unsigned long long>(production_digest), cuda_conv_ms,
      vk_conv_ms, cuda_stage_ms, vk_stage_ms,
      double(production_used_delta) / 1048576.0);
}

VIDFAB_TEST(cuda_vulkan_exact_audio_decoder_graph) {
  using namespace vidfab;
  if (!std::getenv("VIDFAB_AUDIO_DECODER_REAL")) return;
  const std::filesystem::path checkpoint_path =
      "weights/vae/minimax_h3_audio_vae_fp32.safetensors";
  if (!std::filesystem::exists(checkpoint_path)) return;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !vulkan::Instance::available()) return;
  vulkan::Instance instance = vulkan::Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore ||
      !physical.front().info().shader_int64) return;
  vulkan::DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  vulkan::Device device = physical.front().create_device(options);

  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
  vae::AudioDecoder cuda_decoder;
  cuda_decoder.load(checkpoint);
  vulkan::AudioDecoder vk_decoder = vulkan::AudioDecoder::create(device);
  const auto load_begin = std::chrono::steady_clock::now();
  vk_decoder.load(checkpoint);
  const double load_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - load_begin).count();
  CHECK(cuda_decoder.weight_bytes() == vk_decoder.weight_bytes());
  CHECK(vk_decoder.weight_bytes() == 259672032u);
  CHECK(vk_decoder.recorded_operators() == 497u);
  CHECK(cuda_decoder.latents_mean() == vk_decoder.latents_mean());
  CHECK(cuda_decoder.latents_std() == vk_decoder.latents_std());

  constexpr int latent_length = 3;
  std::vector<float> latent = values(size_t(2) * 32 * latent_length,
                                     67, 607, 1.0f / 128.0f);
  const vae::DecodedAudio cuda_audio = cuda_decoder.decode(
      latent.data(), latent_length);
  const auto forward_begin = std::chrono::steady_clock::now();
  const vae::DecodedAudio vk_audio = vk_decoder.decode(
      latent.data(), latent_length);
  const double forward_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - forward_begin).count();
  CHECK(cuda_audio.channels == vk_audio.channels);
  CHECK(cuda_audio.sample_rate == vk_audio.sample_rate);
  check_exact(cuda_audio.samples, vk_audio.samples, "complete audio decoder");

  const auto cuda_wav = std::filesystem::temp_directory_path() /
      "vidfab_cuda_audio_exact.wav";
  const auto vk_wav = std::filesystem::temp_directory_path() /
      "vidfab_vulkan_audio_exact.wav";
  audio::write_wav(cuda_wav.string(), cuda_audio.samples, cuda_audio.channels,
                   cuda_audio.sample_rate, audio::SampleFormat::kPcm16);
  audio::write_wav(vk_wav.string(), vk_audio.samples, vk_audio.channels,
                   vk_audio.sample_rate, audio::SampleFormat::kPcm16);
  auto read_file = [](const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(stream), {});
  };
  CHECK(read_file(cuda_wav) == read_file(vk_wav));
  std::filesystem::remove(cuda_wav);
  std::filesystem::remove(vk_wav);

  constexpr int production_length = 405;
  std::vector<float> production = values(
      size_t(2) * 32 * production_length, 71, 613, 1.0f / 256.0f);
  const auto cuda_begin = std::chrono::steady_clock::now();
  const vae::DecodedAudio cuda_production = cuda_decoder.decode(
      production.data(), production_length);
  const double cuda_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - cuda_begin).count();
  const vae::DecodedAudio vk_warm = vk_decoder.decode(
      production.data(), production_length);
  check_exact(cuda_production.samples, vk_warm.samples,
              "complete A405 audio decoder");
  const uint64_t stable_reserved = vk_decoder.allocator_reserved_bytes();
  const uint64_t stable_descriptors = vk_decoder.descriptor_set_allocations();
  const auto vk_begin = std::chrono::steady_clock::now();
  const vae::DecodedAudio vk_production = vk_decoder.decode(
      production.data(), production_length);
  const double vk_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - vk_begin).count();
  check_exact(cuda_production.samples, vk_production.samples,
              "repeated A405 audio decoder");
  CHECK(vk_decoder.allocator_reserved_bytes() == stable_reserved);
  CHECK(vk_decoder.descriptor_set_allocations() == stable_descriptors);
  CHECK(vk_decoder.allocator_used_bytes() >= vk_decoder.peak_device_bytes());
  CHECK(vk_decoder.allocator_used_bytes() <=
        vk_decoder.peak_device_bytes() + (128ull << 20));
  const uint64_t production_digest = fnv64({vk_production.samples});
  std::printf(
      "  exact Vulkan audio decoder: load %.1f ms, A3 forward %.1f ms, A405 CUDA/Vulkan %.1f/%.1f ms, FNV64 %016llx, weights/peak %.1f/%.1f MiB, pool %.1f/%.1f MiB, descriptors %llu\n",
      load_ms, forward_ms, cuda_ms, vk_ms,
      static_cast<unsigned long long>(production_digest),
      double(vk_decoder.weight_bytes()) / 1048576.0,
      double(vk_decoder.peak_device_bytes()) / 1048576.0,
      double(vk_decoder.allocator_used_bytes()) / 1048576.0,
      double(vk_decoder.allocator_reserved_bytes()) / 1048576.0,
      static_cast<unsigned long long>(vk_decoder.descriptor_set_allocations()));
}
