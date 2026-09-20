#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_h3_real_timing, "benchmark") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  if (!std::getenv("SLOPFAB_H3_ATTENTION_REAL_BENCH")) {
    SKIP_OPT_IN("unavailable prerequisite: !std::getenv(\"SLOPFAB_H3_ATTENTION_REAL_BENCH\")");
    return;
  }
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !Instance::available()");
    return;
  }
  constexpr uint32_t sequence = 37727, heads = 56, dim = 128;
  const size_t count = size_t(sequence) * heads * dim;
  dit::SequenceLayout layout;
  layout.num_text = 17;
  layout.num_audio_rows = 414;
  layout.num_latent_frames = 37;
  layout.latent_height = 48;
  layout.latent_width = 84;
  layout.num_video_rows = layout.num_latent_frames * layout.rows_per_frame();
  const dit::BandedKeyRanges band =
      dit::build_banded_key_ranges(layout, 9, 128, 64);
  const dit::BandedKeyRanges wide =
      dit::build_banded_key_ranges(layout, 64, 128, 64);
  std::vector<uint16_t> host(count);
  for (size_t i = 0; i < count; ++i)
    host[i] = f32_to_bf16(float(int(i % 31) - 15) / 64.0f);
  std::vector<uint16_t> expected_full(count), expected_band(count);
  float cuda_full_ms = 0.0f, cuda_band_ms = 0.0f;
  float shipped_full_ms = 0.0f, shipped_band_ms = 0.0f;
  size_t shipped_differences = 0;
  float shipped_max_abs = 0.0f;
  {
    cuda::DeviceBuffer<uint16_t> q(count), k(count), v(count), out(count);
    cuda::DeviceBuffer<int32_t> ranges(band.ranges.size());
    q.copy_from_host(host.data(), count);
    k.copy_from_host(host.data(), count);
    v.copy_from_host(host.data(), count);
    ranges.copy_from_host(band.ranges.data(), band.ranges.size());
    auto timed = [&](auto&& launch) {
      cudaEvent_t begin{}, end{};
      SLOPFAB_CUDA_CHECK(cudaEventCreate(&begin));
      SLOPFAB_CUDA_CHECK(cudaEventCreate(&end));
      SLOPFAB_CUDA_CHECK(cudaEventRecord(begin));
      launch();
      SLOPFAB_CUDA_CHECK(cudaEventRecord(end));
      SLOPFAB_CUDA_CHECK(cudaEventSynchronize(end));
      float ms = 0.0f;
      SLOPFAB_CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
      cudaEventDestroy(begin); cudaEventDestroy(end);
      return ms;
    };
    cuda_full_ms = timed([&] {
      cuda::launch_deterministic_h3_attention(
          nullptr, reinterpret_cast<const __nv_bfloat16*>(q.get()),
          reinterpret_cast<const __nv_bfloat16*>(k.get()),
          reinterpret_cast<const __nv_bfloat16*>(v.get()),
          reinterpret_cast<__nv_bfloat16*>(out.get()), nullptr,
          sequence, heads, dim, exact_attention_scale(dim));
    });
    out.copy_to_host(expected_full.data(), count);
    cuda_band_ms = timed([&] {
      cuda::launch_deterministic_h3_attention(
          nullptr, reinterpret_cast<const __nv_bfloat16*>(q.get()),
          reinterpret_cast<const __nv_bfloat16*>(k.get()),
          reinterpret_cast<const __nv_bfloat16*>(v.get()),
          reinterpret_cast<__nv_bfloat16*>(out.get()), ranges.get(),
          sequence, heads, dim, exact_attention_scale(dim));
    });
    out.copy_to_host(expected_band.data(), count);
    cublasHandle_t handle = nullptr;
    SLOPFAB_CUBLAS_CHECK(slopfab::cuda::cublas_create(&handle));
    cuda::Workspace workspace;
    cuda::AttentionConfig config;
    config.seq_len = sequence;
    config.num_heads = heads;
    config.head_dim = dim;
    config.scale = exact_attention_scale(dim);
    config.band_ranges = nullptr;
    shipped_full_ms = timed([&] {
      cuda::attention_forward(
          handle, nullptr, reinterpret_cast<const __nv_bfloat16*>(q.get()),
          reinterpret_cast<const __nv_bfloat16*>(k.get()),
          reinterpret_cast<const __nv_bfloat16*>(v.get()),
          reinterpret_cast<__nv_bfloat16*>(out.get()), config,
          cuda::AttentionBackend::kFused, workspace);
    });
    std::vector<uint16_t> shipped(count);
    out.copy_to_host(shipped.data(), count);
    for (size_t i = 0; i < count; ++i) {
      if (shipped[i] != expected_full[i]) ++shipped_differences;
      shipped_max_abs = std::max(
          shipped_max_abs,
          std::abs(bf16_to_f32(shipped[i]) - bf16_to_f32(expected_full[i])));
    }
    config.band_ranges = ranges.get();
    shipped_band_ms = timed([&] {
      cuda::attention_forward(
          handle, nullptr, reinterpret_cast<const __nv_bfloat16*>(q.get()),
          reinterpret_cast<const __nv_bfloat16*>(k.get()),
          reinterpret_cast<const __nv_bfloat16*>(v.get()),
          reinterpret_cast<__nv_bfloat16*>(out.get()), config,
          cuda::AttentionBackend::kFused, workspace);
    });
    slopfab::cuda::cublas_destroy(handle);
  }

  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  CHECK(!physical.empty());
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  options.enable_shader_float16 = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  CHECK(vk.exact_h3_attention());
  const uint64_t shape[] = {sequence, heads, dim};
  const TensorLayout tensor_layout = TensorLayout::contiguous(shape, 3);
  DeviceTensor q = vk.allocate(tensor_layout, ScalarType::kBFloat16);
  DeviceTensor k = vk.allocate(tensor_layout, ScalarType::kBFloat16);
  DeviceTensor v = vk.allocate(tensor_layout, ScalarType::kBFloat16);
  DeviceTensor out = vk.allocate(tensor_layout, ScalarType::kBFloat16);
  vk.upload_bytes(q, host.data(), count * sizeof(uint16_t));
  vk.upload_bytes(k, host.data(), count * sizeof(uint16_t));
  vk.upload_bytes(v, host.data(), count * sizeof(uint16_t));
  H3AttentionPlan plan = H3AttentionPlan::create(
      vk, {sequence, heads, dim, exact_attention_scale(dim)});
  H3AttentionRanges band_table = H3AttentionRanges::create(
      vk, sequence, band.ranges.data(),
      static_cast<uint32_t>(band.ranges.size()));
  H3AttentionRanges wide_table = H3AttentionRanges::create(
      vk, sequence, wide.ranges.data(),
      static_cast<uint32_t>(wide.ranges.size()));
  auto timed_vk = [&](const H3AttentionRanges* selected) {
    const auto begin = std::chrono::steady_clock::now();
    TensorBatch batch = vk.begin_batch();
    plan.record(batch, q, k, v, out, selected);
    batch.submit().wait();
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
  };
  const double vulkan_full_ms = timed_vk(nullptr);
  std::vector<uint16_t> got(count);
  vk.download_bytes(out, got.data(), count * sizeof(uint16_t));
  CHECK(got == expected_full);
  const double vulkan_band_ms = timed_vk(&band_table);
  vk.download_bytes(out, got.data(), count * sizeof(uint16_t));
  CHECK(got == expected_band);
  const double vulkan_wide_ms = timed_vk(&wide_table);
  vk.download_bytes(out, got.data(), count * sizeof(uint16_t));
  CHECK(got == expected_full);
  const double direct_mib = double(count * sizeof(uint16_t) * 4) / 1048576.0;
  std::printf(
      "  H3 real S37727 H56 D128: exact CUDA full %.3f ms/band %.3f ms; Vulkan full %.3f ms/band %.3f ms/wide %.3f ms; shipped fused full %.3f ms/band %.3f ms; shipped drift %zu/%zu maxabs %.7g; direct QKV/out %.2f MiB, range %zu bytes, scratch 0\n",
      cuda_full_ms, cuda_band_ms, vulkan_full_ms, vulkan_band_ms,
      vulkan_wide_ms, shipped_full_ms, shipped_band_ms,
      shipped_differences, count, shipped_max_abs, direct_mib,
      band.ranges.size() * sizeof(int32_t));
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_causal_gqa_real_timing, "benchmark") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  if (!std::getenv("SLOPFAB_CAUSAL_GQA_BENCH")) {
    SKIP_OPT_IN("unavailable prerequisite: !std::getenv(\"SLOPFAB_CAUSAL_GQA_BENCH\")");
    return;
  }
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: physical.empty()");
    return;
  }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  if (!vk.exact_causal_gqa_attention()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !vk.exact_causal_gqa_attention()");
    return;
  }

  for (uint32_t sequence : {132u, 8192u}) {
    constexpr uint32_t query_heads = 64, kv_heads = 8, dim = 128;
    const size_t q_count = size_t(sequence) * query_heads * dim;
    const size_t kv_count = size_t(sequence) * kv_heads * dim;
    std::vector<uint16_t> hq(q_count, f32_to_bf16(0.03125f));
    std::vector<uint16_t> hk(kv_count, f32_to_bf16(-0.015625f));
    std::vector<uint16_t> hv(kv_count, f32_to_bf16(0.0625f));
    cuda::DeviceBuffer<uint16_t> cq(q_count), ck(kv_count), cv(kv_count),
        co(q_count);
    cq.copy_from_host(hq.data(), hq.size());
    ck.copy_from_host(hk.data(), hk.size());
    cv.copy_from_host(hv.data(), hv.size());
    const uint64_t q_shape[] = {sequence, query_heads, dim};
    const uint64_t kv_shape[] = {sequence, kv_heads, dim};
    DeviceTensor q = vk.allocate(TensorLayout::contiguous(q_shape, 3),
                                 ScalarType::kBFloat16);
    DeviceTensor k = vk.allocate(TensorLayout::contiguous(kv_shape, 3),
                                 ScalarType::kBFloat16);
    DeviceTensor v = vk.allocate(TensorLayout::contiguous(kv_shape, 3),
                                 ScalarType::kBFloat16);
    DeviceTensor out = vk.allocate(TensorLayout::contiguous(q_shape, 3),
                                   ScalarType::kBFloat16);
    vk.upload_bytes(q, hq.data(), hq.size() * 2);
    vk.upload_bytes(k, hk.data(), hk.size() * 2);
    vk.upload_bytes(v, hv.data(), hv.size() * 2);
    CausalGQAAttentionPlan plan = CausalGQAAttentionPlan::create(
        vk, {sequence, query_heads, kv_heads, dim, exact_attention_scale(dim)});
    auto submit_vulkan = [&] {
      TensorBatch batch = vk.begin_batch();
      plan.record(batch, q, k, v, out);
      return batch.submit();
    };
    auto launch_cuda = [&] {
      cuda::launch_deterministic_causal_gqa_attention(
          nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
          reinterpret_cast<const __nv_bfloat16*>(ck.get()),
          reinterpret_cast<const __nv_bfloat16*>(cv.get()),
          reinterpret_cast<__nv_bfloat16*>(co.get()), sequence, query_heads,
          kv_heads, dim, exact_attention_scale(dim));
    };
    launch_cuda();
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    submit_vulkan().wait();
    cudaEvent_t start = nullptr, stop = nullptr;
    SLOPFAB_CUDA_CHECK(cudaEventCreate(&start));
    SLOPFAB_CUDA_CHECK(cudaEventCreate(&stop));
    SLOPFAB_CUDA_CHECK(cudaEventRecord(start));
    launch_cuda();
    SLOPFAB_CUDA_CHECK(cudaEventRecord(stop));
    SLOPFAB_CUDA_CHECK(cudaEventSynchronize(stop));
    float cuda_ms = 0.0f;
    SLOPFAB_CUDA_CHECK(cudaEventElapsedTime(&cuda_ms, start, stop));
    SLOPFAB_CUDA_CHECK(cudaEventDestroy(start));
    SLOPFAB_CUDA_CHECK(cudaEventDestroy(stop));
    const auto begin = std::chrono::steady_clock::now();
    submit_vulkan().wait();
    const double vulkan_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    std::vector<uint16_t> cuda_out(q_count), vulkan_out(q_count);
    co.copy_to_host(cuda_out.data(), cuda_out.size());
    vk.download_bytes(out, vulkan_out.data(), vulkan_out.size() * 2);
    CHECK(cuda_out == vulkan_out);
    const uint64_t direct_bytes =
        (uint64_t(q_count) * 2 + uint64_t(kv_count) * 2) * sizeof(uint16_t);
    std::printf("  causal GQA L%u H64/KV8/D128: CUDA %.3f ms, Vulkan %.3f ms, "
                "direct Q/K/V/out %.2f MiB, pool used %.2f MiB, context reserved %.2f MiB, descriptors %llu\n",
                sequence, cuda_ms, vulkan_ms, direct_bytes / (1024.0 * 1024.0),
                vk.pooled_used_bytes() / (1024.0 * 1024.0),
                vk.reserved_bytes() / (1024.0 * 1024.0),
                static_cast<unsigned long long>(vk.descriptor_set_allocations()));
  }
}
