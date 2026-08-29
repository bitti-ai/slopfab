#include "harness.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "vidfab/cuda/audio_vae_kernels.cuh"
#include "vidfab/cuda/device.h"
#include "vidfab/vae/audio_primitives.h"
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
