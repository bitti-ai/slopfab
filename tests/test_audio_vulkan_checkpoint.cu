#include "detail/audio_vulkan_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_exact_audio_real_checkpoint_primitives, "checkpoint") {
  using namespace slopfab;
  if (!std::getenv("SLOPFAB_AUDIO_VAE_REAL")) {
    SKIP_OPT_IN("unavailable prerequisite: !std::getenv(\"SLOPFAB_AUDIO_VAE_REAL\")");
    return;
  }
  const std::filesystem::path path = "weights/vae/minimax_h3_audio_vae_fp32.safetensors";
  if (!std::filesystem::exists(path)) {
    SKIP_MISSING_FIXTURE("unavailable prerequisite: !std::filesystem::exists(path)");
    return;
  }
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !vulkan::Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !vulkan::Instance::available()");
    return;
  }
  vulkan::Instance instance = vulkan::Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore ||
      !physical.front().info().shader_int64) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore || !physical.front().info().shader_int64");
    return;
  }

  SafeTensors checkpoint;
  checkpoint.open(path.string());
  CHECK(checkpoint.file_size() == 605254808u);
  CHECK(checkpoint.tensor_count() == 917u);
#ifdef _WIN32
  const std::array<uint8_t, 32> expected_sha{0x8e, 0x50, 0x5d, 0x95, 0xdd, 0x15, 0x61, 0xd4,
                                             0x7a, 0xbd, 0x43, 0xd4, 0x23, 0x8f, 0xd4, 0x0d,
                                             0x9b, 0xb1, 0xae, 0x9e, 0x14, 0x7e, 0xd0, 0xa4,
                                             0xcb, 0xa7, 0x78, 0xd7, 0x6a, 0xe4, 0xdb, 0x48};
  CHECK(mapping_sha256(checkpoint.mapping_base(), checkpoint.file_size()) == expected_sha);
#endif

  vae::AudioConvWeights conv_weights =
      vae::load_audio_conv_weights(checkpoint, "dec_in_proj", {2048, 32, 1}, 2048, true);
  vae::AudioConvWeights transpose_weights =
      vae::load_audio_conv_weights(checkpoint, "decoder.ups.0.0", {1024, 512, 9}, 512, true);
  vae::AudioConvWeights dilated_weights = vae::load_audio_conv_weights(
      checkpoint, "decoder.resblocks.2.convs1.2", {512, 512, 11}, 512, true);
  CHECK(!conv_weights.folded_weight_norm);
  CHECK(!transpose_weights.folded_weight_norm);
  CHECK(!dilated_weights.folded_weight_norm);
  std::vector<float> log_alpha =
      vae::load_audio_f32_tensor(checkpoint, "decoder.activation_post.act.alpha", {8});
  std::vector<float> log_beta =
      vae::load_audio_f32_tensor(checkpoint, "decoder.activation_post.act.beta", {8});
  std::vector<float> up_filter =
      vae::load_audio_f32_tensor(checkpoint, "decoder.activation_post.upsample.filter", {1, 1, 12});
  std::vector<float> down_filter = vae::load_audio_f32_tensor(
      checkpoint, "decoder.activation_post.downsample.lowpass.filter", {1, 1, 12});
  const size_t raw_subnormals =
      count_subnormals(conv_weights.weight) + count_subnormals(conv_weights.bias) +
      count_subnormals(transpose_weights.weight) + count_subnormals(transpose_weights.bias) +
      count_subnormals(dilated_weights.weight) + count_subnormals(dilated_weights.bias) +
      count_subnormals(log_alpha) + count_subnormals(log_beta) + count_subnormals(up_filter) +
      count_subnormals(down_filter);
  CHECK(raw_subnormals == 0u);

  vulkan::DeviceOptions device_options;
  device_options.enable_timeline_semaphore = true;
  device_options.enable_shader_int64 = true;
  vulkan::Device device = physical.front().create_device(device_options);
  vulkan::TensorContext vk(device);
  if (!vk.exact_audio_vae_primitives()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !vk.exact_audio_vae_primitives()");
    return;
  }

  auto upload = [&](const TensorLayout& tensor_layout, const std::vector<float>& host) {
    vulkan::DeviceTensor tensor = vk.allocate(tensor_layout);
    vk.upload(tensor, host.data(), host.size());
    return tensor;
  };

  const vae::AudioConv1DDesc conv{2, 32, 2048, 3, 3, 1, 0, 1};
  std::vector<float> conv_input = values(conv.input_elements(), 37, 509, 1.0f / 256.0f);
  cuda::DeviceBuffer<float> c_conv_input(conv.input_elements()),
      c_conv_weight(conv.weight_elements()), c_conv_bias(2048),
      c_conv_output(conv.output_elements());
  c_conv_input.copy_from_host(conv_input.data(), conv_input.size());
  c_conv_weight.copy_from_host(conv_weights.weight.data(), conv_weights.weight.size());
  c_conv_bias.copy_from_host(conv_weights.bias.data(), conv_weights.bias.size());
  cuda::launch_conv1d(c_conv_input.get(), c_conv_weight.get(), c_conv_bias.get(),
                      c_conv_output.get(), conv.batch, conv.in_channels, conv.out_channels,
                      conv.length_in, conv.length_out, conv.kernel, conv.padding, conv.dilation,
                      nullptr);
  vulkan::DeviceTensor v_conv_input = upload(layout({2, 32, 3}), conv_input);
  vulkan::DeviceTensor v_conv_weight = upload(layout({2048, 32, 1}), conv_weights.weight);
  vulkan::DeviceTensor v_conv_bias = upload(layout({2048}), conv_weights.bias);
  vulkan::DeviceTensor v_conv_output = vk.allocate(layout({2, 2048, 3}));
  {
    vulkan::TensorBatch batch = vk.begin_batch();
    batch.audio_conv1d(v_conv_input, v_conv_weight, &v_conv_bias, v_conv_output, conv);
    batch.submit().wait();
  }

  const vae::AudioConvTranspose1DDesc transpose{1, 1024, 512, 3, 15, 9, 5, 2};
  std::vector<float> transpose_input = values(transpose.input_elements(), 41, 257, 1.0f / 256.0f);
  cuda::DeviceBuffer<float> c_transpose_input(transpose.input_elements()),
      c_transpose_weight(transpose.weight_elements()), c_transpose_bias(512),
      c_transpose_output(transpose.output_elements());
  c_transpose_input.copy_from_host(transpose_input.data(), transpose_input.size());
  c_transpose_weight.copy_from_host(transpose_weights.weight.data(),
                                    transpose_weights.weight.size());
  c_transpose_bias.copy_from_host(transpose_weights.bias.data(), transpose_weights.bias.size());
  cuda::launch_conv_transpose1d(c_transpose_input.get(), c_transpose_weight.get(),
                                c_transpose_bias.get(), c_transpose_output.get(), transpose.batch,
                                transpose.in_channels, transpose.out_channels, transpose.length_in,
                                transpose.length_out, transpose.kernel, transpose.stride,
                                transpose.padding, nullptr);
  vulkan::DeviceTensor v_transpose_input = upload(layout({1, 1024, 3}), transpose_input);
  vulkan::DeviceTensor v_transpose_weight =
      upload(layout({1024, 512, 9}), transpose_weights.weight);
  vulkan::DeviceTensor v_transpose_bias = upload(layout({512}), transpose_weights.bias);
  vulkan::DeviceTensor v_transpose_output = vk.allocate(layout({1, 512, 15}));
  const auto transpose_begin = std::chrono::steady_clock::now();
  {
    vulkan::TensorBatch batch = vk.begin_batch();
    batch.audio_conv_transpose1d(v_transpose_input, v_transpose_weight, &v_transpose_bias,
                                 v_transpose_output, transpose);
    batch.submit().wait();
  }
  const double transpose_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - transpose_begin)
          .count();

  const vae::AudioConv1DDesc dilated{1, 512, 512, 15, 15, 11, 25, 5};
  std::vector<float> dilated_input = values(dilated.input_elements(), 47, 263, 1.0f / 256.0f);
  cuda::DeviceBuffer<float> c_dilated_input(dilated.input_elements()),
      c_dilated_weight(dilated.weight_elements()), c_dilated_bias(512),
      c_dilated_output(dilated.output_elements());
  c_dilated_input.copy_from_host(dilated_input.data(), dilated_input.size());
  c_dilated_weight.copy_from_host(dilated_weights.weight.data(), dilated_weights.weight.size());
  c_dilated_bias.copy_from_host(dilated_weights.bias.data(), dilated_weights.bias.size());
  cuda::launch_conv1d(c_dilated_input.get(), c_dilated_weight.get(), c_dilated_bias.get(),
                      c_dilated_output.get(), dilated.batch, dilated.in_channels,
                      dilated.out_channels, dilated.length_in, dilated.length_out, dilated.kernel,
                      dilated.padding, dilated.dilation, nullptr);
  vulkan::DeviceTensor v_dilated_input = upload(layout({1, 512, 15}), dilated_input);
  vulkan::DeviceTensor v_dilated_weight = upload(layout({512, 512, 11}), dilated_weights.weight);
  vulkan::DeviceTensor v_dilated_bias = upload(layout({512}), dilated_weights.bias);
  vulkan::DeviceTensor v_dilated_output = vk.allocate(layout({1, 512, 15}));
  const auto dilated_begin = std::chrono::steady_clock::now();
  {
    vulkan::TensorBatch batch = vk.begin_batch();
    batch.audio_conv1d(v_dilated_input, v_dilated_weight, &v_dilated_bias, v_dilated_output,
                       dilated);
    batch.submit().wait();
  }
  const double dilated_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - dilated_begin)
          .count();

  constexpr uint32_t aa_batch = 2, aa_channels = 8, aa_length = 259;
  const size_t aa_count = size_t(aa_batch) * aa_channels * aa_length;
  std::vector<float> aa_input = values(aa_count, 43, 311, 1.0f / 128.0f);
  cuda::DeviceBuffer<float> c_aa_input(aa_count), c_up_filter(12), c_down_filter(12), c_alpha(8),
      c_beta(8), c_aa_up(aa_count * 2), c_aa_down(aa_count);
  c_aa_input.copy_from_host(aa_input.data(), aa_input.size());
  c_up_filter.copy_from_host(up_filter.data(), up_filter.size());
  c_down_filter.copy_from_host(down_filter.data(), down_filter.size());
  c_alpha.copy_from_host(log_alpha.data(), log_alpha.size());
  c_beta.copy_from_host(log_beta.data(), log_beta.size());
  cuda::launch_aa_upsample_snake(c_aa_input.get(), c_up_filter.get(), c_alpha.get(), c_beta.get(),
                                 c_aa_up.get(), aa_batch, aa_channels, aa_length, nullptr);
  cuda::launch_aa_downsample(c_aa_up.get(), c_down_filter.get(), c_aa_down.get(), aa_batch,
                             aa_channels, aa_length * 2, aa_length, nullptr);
  vulkan::DeviceTensor v_aa_input = upload(layout({aa_batch, aa_channels, aa_length}), aa_input);
  vulkan::DeviceTensor v_up_filter = upload(layout({12}), up_filter);
  vulkan::DeviceTensor v_down_filter = upload(layout({12}), down_filter);
  vulkan::DeviceTensor v_alpha = upload(layout({8}), log_alpha);
  vulkan::DeviceTensor v_beta = upload(layout({8}), log_beta);
  vulkan::DeviceTensor v_aa_up = vk.allocate(layout({aa_batch, aa_channels, aa_length * 2}));
  vulkan::DeviceTensor v_aa_down = vk.allocate(layout({aa_batch, aa_channels, aa_length}));
  {
    vulkan::TensorBatch batch = vk.begin_batch();
    batch.audio_aa_upsample_snake(v_aa_input, v_up_filter, v_alpha, v_beta, v_aa_up, aa_batch,
                                  aa_channels, aa_length);
    batch.audio_aa_downsample(v_aa_up, v_down_filter, v_aa_down, aa_batch, aa_channels,
                              aa_length * 2, aa_length);
    batch.submit().wait();
  }
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> cuda_conv(conv.output_elements()), vk_conv(conv.output_elements()),
      cuda_transpose(transpose.output_elements()), vk_transpose(transpose.output_elements()),
      cuda_dilated(dilated.output_elements()), vk_dilated(dilated.output_elements()),
      cuda_up(aa_count * 2), vk_up(aa_count * 2), cuda_down(aa_count), vk_down(aa_count);
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
  const uint64_t digest = fnv64({vk_conv, vk_transpose, vk_dilated, vk_up, vk_down});
  CHECK(digest == 0x2f78225c9577a73bull);
  const uint64_t logical_bytes =
      sizeof(float) *
      (conv.input_elements() + conv.weight_elements() + conv.out_channels + conv.output_elements() +
       transpose.input_elements() + transpose.weight_elements() + transpose.out_channels +
       transpose.output_elements() + dilated.input_elements() + dilated.weight_elements() +
       dilated.out_channels + dilated.output_elements() + aa_count + 12u + 12u + 8u + 8u +
       aa_count * 2u + aa_count);
  CHECK(vk.pooled_used_bytes() >= logical_bytes);
  CHECK(vk.pooled_used_bytes() <= logical_bytes + (40ull << 20));
  CHECK(vk.reserved_bytes() <= (80ull << 20));
  const uint64_t stable_reserved = vk.reserved_bytes();
  const uint64_t stable_descriptors = vk.descriptor_set_allocations();
  {
    vulkan::TensorBatch batch = vk.begin_batch();
    batch.audio_conv_transpose1d(v_transpose_input, v_transpose_weight, &v_transpose_bias,
                                 v_transpose_output, transpose);
    batch.submit().wait();
  }
  CHECK(vk.reserved_bytes() == stable_reserved);
  CHECK(vk.descriptor_set_allocations() == stable_descriptors);

  // Full production temporal regime used for a 10.125-second decode. The
  // transpose and residual convolution share one device-resident batch.
  constexpr uint32_t production_length = 405;
  const vae::AudioConv1DDesc production_conv{2, 32, 2048, production_length, production_length,
                                             1, 0,  1};
  const vae::AudioConvTranspose1DDesc production_transpose{
      2, 1024, 512, production_length, production_length * 5, 9, 5, 2};
  const vae::AudioConv1DDesc production_residual{
      2, 512, 512, production_length * 5, production_length * 5, 11, 25, 5};
  std::vector<float> production_conv_input =
      values(production_conv.input_elements(), 53, 509, 1.0f / 256.0f);
  std::vector<float> production_transpose_input =
      values(production_transpose.input_elements(), 59, 521, 1.0f / 256.0f);
  cuda::DeviceBuffer<float> c_production_conv_input(production_conv.input_elements());
  cuda::DeviceBuffer<float> c_production_conv_output(production_conv.output_elements());
  cuda::DeviceBuffer<float> c_production_transpose_input(production_transpose.input_elements());
  cuda::DeviceBuffer<float> c_production_transpose_output(production_transpose.output_elements());
  cuda::DeviceBuffer<float> c_production_residual_output(production_residual.output_elements());
  c_production_conv_input.copy_from_host(production_conv_input.data(),
                                         production_conv_input.size());
  c_production_transpose_input.copy_from_host(production_transpose_input.data(),
                                              production_transpose_input.size());
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const auto cuda_conv_begin = std::chrono::steady_clock::now();
  cuda::launch_conv1d(c_production_conv_input.get(), c_conv_weight.get(), c_conv_bias.get(),
                      c_production_conv_output.get(), production_conv.batch,
                      production_conv.in_channels, production_conv.out_channels,
                      production_conv.length_in, production_conv.length_out, production_conv.kernel,
                      production_conv.padding, production_conv.dilation, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const double cuda_conv_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cuda_conv_begin)
          .count();
  const auto cuda_stage_begin = std::chrono::steady_clock::now();
  cuda::launch_conv_transpose1d(c_production_transpose_input.get(), c_transpose_weight.get(),
                                c_transpose_bias.get(), c_production_transpose_output.get(),
                                production_transpose.batch, production_transpose.in_channels,
                                production_transpose.out_channels, production_transpose.length_in,
                                production_transpose.length_out, production_transpose.kernel,
                                production_transpose.stride, production_transpose.padding, nullptr);
  cuda::launch_conv1d(c_production_transpose_output.get(), c_dilated_weight.get(),
                      c_dilated_bias.get(), c_production_residual_output.get(),
                      production_residual.batch, production_residual.in_channels,
                      production_residual.out_channels, production_residual.length_in,
                      production_residual.length_out, production_residual.kernel,
                      production_residual.padding, production_residual.dilation, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const double cuda_stage_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cuda_stage_begin)
          .count();

  const uint64_t before_production_used = vk.pooled_used_bytes();
  const uint64_t before_production_reserved = vk.reserved_bytes();
  vulkan::DeviceTensor v_production_conv_input =
      upload(layout({2, 32, production_length}), production_conv_input);
  vulkan::DeviceTensor v_production_conv_output = vk.allocate(layout({2, 2048, production_length}));
  vulkan::DeviceTensor v_production_transpose_input =
      upload(layout({2, 1024, production_length}), production_transpose_input);
  vulkan::DeviceTensor v_production_transpose_output =
      vk.allocate(layout({2, 512, production_length * 5}));
  vulkan::DeviceTensor v_production_residual_output =
      vk.allocate(layout({2, 512, production_length * 5}));
  const auto vk_conv_begin = std::chrono::steady_clock::now();
  {
    vulkan::TensorBatch batch = vk.begin_batch();
    batch.audio_conv1d(v_production_conv_input, v_conv_weight, &v_conv_bias,
                       v_production_conv_output, production_conv);
    batch.submit().wait();
  }
  const double vk_conv_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - vk_conv_begin)
          .count();
  const auto vk_stage_begin = std::chrono::steady_clock::now();
  {
    vulkan::TensorBatch batch = vk.begin_batch();
    batch.audio_conv_transpose1d(v_production_transpose_input, v_transpose_weight,
                                 &v_transpose_bias, v_production_transpose_output,
                                 production_transpose);
    batch.audio_conv1d(v_production_transpose_output, v_dilated_weight, &v_dilated_bias,
                       v_production_residual_output, production_residual);
    batch.submit().wait();
  }
  const double vk_stage_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - vk_stage_begin)
          .count();
  std::vector<float> cuda_production_conv(production_conv.output_elements()),
      vk_production_conv(production_conv.output_elements()),
      cuda_production_transpose(production_transpose.output_elements()),
      vk_production_transpose(production_transpose.output_elements()),
      cuda_production_residual(production_residual.output_elements()),
      vk_production_residual(production_residual.output_elements());
  c_production_conv_output.copy_to_host(cuda_production_conv.data(), cuda_production_conv.size());
  c_production_transpose_output.copy_to_host(cuda_production_transpose.data(),
                                             cuda_production_transpose.size());
  c_production_residual_output.copy_to_host(cuda_production_residual.data(),
                                            cuda_production_residual.size());
  vk.download(v_production_conv_output, vk_production_conv.data(), vk_production_conv.size());
  vk.download(v_production_transpose_output, vk_production_transpose.data(),
              vk_production_transpose.size());
  vk.download(v_production_residual_output, vk_production_residual.data(),
              vk_production_residual.size());
  check_exact(cuda_production_conv, vk_production_conv, "A405 input projection");
  check_exact(cuda_production_transpose, vk_production_transpose, "A405 transpose 405->2025");
  check_exact(cuda_production_residual, vk_production_residual, "A405 stereo residual");
  const uint64_t production_digest =
      fnv64({vk_production_conv, vk_production_transpose, vk_production_residual});
  CHECK(production_digest == 0x2a3eeba122112082ull);
  const uint64_t production_logical_bytes =
      sizeof(float) *
      (production_conv.input_elements() + production_conv.output_elements() +
       production_transpose.input_elements() + production_transpose.output_elements() +
       production_residual.output_elements());
  const uint64_t production_used_delta = vk.pooled_used_bytes() - before_production_used;
  CHECK(production_used_delta >= production_logical_bytes);
  CHECK(production_used_delta <= production_logical_bytes + (8ull << 20));
  CHECK(vk.reserved_bytes() <= before_production_reserved + (32ull << 20));
  const uint64_t production_stable_reserved = vk.reserved_bytes();
  const uint64_t production_stable_descriptors = vk.descriptor_set_allocations();
  {
    vulkan::TensorBatch batch = vk.begin_batch();
    batch.audio_conv_transpose1d(v_production_transpose_input, v_transpose_weight,
                                 &v_transpose_bias, v_production_transpose_output,
                                 production_transpose);
    batch.audio_conv1d(v_production_transpose_output, v_dilated_weight, &v_dilated_bias,
                       v_production_residual_output, production_residual);
    batch.submit().wait();
  }
  CHECK(vk.reserved_bytes() == production_stable_reserved);
  CHECK(vk.descriptor_set_allocations() == production_stable_descriptors);
  std::printf(
      "  real audio primitives: raw subnormals %zu, FNV64 %016llx, first transpose %.2f ms, dilated k11/d5 %.2f ms, pool %.1f/%.1f MiB, descriptors %llu\n"
      "  A405 stereo: FNV64 %016llx, input projection CUDA/Vulkan %.2f/%.2f ms, transpose+residual CUDA/Vulkan %.2f/%.2f ms, production delta %.1f MiB\n",
      raw_subnormals, static_cast<unsigned long long>(digest), transpose_ms, dilated_ms,
      double(vk.pooled_used_bytes()) / 1048576.0, double(vk.reserved_bytes()) / 1048576.0,
      static_cast<unsigned long long>(vk.descriptor_set_allocations()),
      static_cast<unsigned long long>(production_digest), cuda_conv_ms, vk_conv_ms, cuda_stage_ms,
      vk_stage_ms, double(production_used_delta) / 1048576.0);
}
