#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_exact_blocked_attention, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore ||
      !physical.front().info().shader_int64) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore || !physical.front().info().shader_int64");
    return;
  }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  if (!vk.exact_blocked_attention()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !vk.exact_blocked_attention()");
    return;
  }

  auto run = [&](uint32_t sequence, uint32_t heads, uint32_t dim,
                 const std::vector<uint16_t>* custom_q = nullptr,
                 const std::vector<uint16_t>* custom_k = nullptr,
                 const std::vector<uint16_t>* custom_v = nullptr) {
    const size_t count = static_cast<size_t>(sequence) * heads * dim;
    std::vector<uint16_t> q(count), k(count), v(count);
    for (size_t i = 0; i < count; ++i) {
      q[i] = f32_to_bf16(static_cast<float>(static_cast<int>(i * 17 % 41) - 20) / 32.0f);
      k[i] = f32_to_bf16(static_cast<float>(static_cast<int>(i * 13 % 37) - 18) / 32.0f);
      v[i] = f32_to_bf16(static_cast<float>(static_cast<int>(i * 19 % 43) - 21) / 16.0f);
    }
    if (count >= 8) {
      q[0] = 0x8000u; q[1] = f32_to_bf16(1.0f);
      k[0] = f32_to_bf16(1.0f); k[1] = f32_to_bf16(-1.0f);
      v[0] = 0x8000u; v[1] = f32_to_bf16(1.0f);
    }
    if (custom_q && custom_k && custom_v) {
      CHECK(custom_q->size() == count);
      CHECK(custom_k->size() == count);
      CHECK(custom_v->size() == count);
      q = *custom_q; k = *custom_k; v = *custom_v;
    }
    cuda::DeviceBuffer<uint16_t> cq(count), ck(count), cv(count), co(count),
        co_repeat(sequence == 129 ? count : 0);
    cuda::DeviceBuffer<uint16_t> cq16(count), ck16(count), cv16(count);
    cq.copy_from_host(q.data(), count); ck.copy_from_host(k.data(), count);
    cv.copy_from_host(v.data(), count);
    const float scale = dim == 64 ? 0.125f : dim == 128 ? 0.0883883461356163f
                                                         : 0.11785113019775793f;
    cuda::launch_prepare_deterministic_attention_inputs(
        nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
        reinterpret_cast<const __nv_bfloat16*>(ck.get()),
        reinterpret_cast<const __nv_bfloat16*>(cv.get()),
        reinterpret_cast<__half*>(cq16.get()), reinterpret_cast<__half*>(ck16.get()),
        reinterpret_cast<__half*>(cv16.get()), count);
    if (sequence == 129) {
      cuda::launch_deterministic_blocked_attention_f16(
          nullptr, reinterpret_cast<const __half*>(cq16.get()),
          reinterpret_cast<const __half*>(ck16.get()),
          reinterpret_cast<const __half*>(cv16.get()),
          reinterpret_cast<__nv_bfloat16*>(co.get()), sequence, heads, dim,
          scale, 0, 65, 0);
      cuda::launch_deterministic_blocked_attention_f16(
          nullptr, reinterpret_cast<const __half*>(cq16.get()),
          reinterpret_cast<const __half*>(ck16.get()),
          reinterpret_cast<const __half*>(cv16.get()),
          reinterpret_cast<__nv_bfloat16*>(co.get()), sequence, heads, dim,
          scale, 65, 64, 65);
    } else {
      cuda::launch_deterministic_blocked_attention_f16(
          nullptr, reinterpret_cast<const __half*>(cq16.get()),
          reinterpret_cast<const __half*>(ck16.get()),
          reinterpret_cast<const __half*>(cv16.get()),
          reinterpret_cast<__nv_bfloat16*>(co.get()), sequence, heads, dim, scale);
    }
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> cuda_output(count);
    co.copy_to_host(cuda_output.data(), count);
    if (sequence == 129) {
      cuda::launch_deterministic_blocked_attention_f16(
          nullptr, reinterpret_cast<const __half*>(cq16.get()),
          reinterpret_cast<const __half*>(ck16.get()),
          reinterpret_cast<const __half*>(cv16.get()),
          reinterpret_cast<__nv_bfloat16*>(co_repeat.get()), sequence, heads,
          dim, scale);
      SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<uint16_t> cuda_repeat(count);
      co_repeat.copy_to_host(cuda_repeat.data(), count);
      CHECK(cuda_repeat == cuda_output);
    }

    const uint64_t shape[] = {sequence, heads, dim};
    const TensorLayout layout = TensorLayout::contiguous(shape, 3);
    DeviceTensor vq = vk.allocate(layout, ScalarType::kBFloat16);
    DeviceTensor vk_tensor = vk.allocate(layout, ScalarType::kBFloat16);
    DeviceTensor vv = vk.allocate(layout, ScalarType::kBFloat16);
    DeviceTensor vo = vk.allocate(layout, ScalarType::kBFloat16);
    vk.upload_bytes(vq, q.data(), count * 2);
    vk.upload_bytes(vk_tensor, k.data(), count * 2);
    vk.upload_bytes(vv, v.data(), count * 2);
    BlockedAttentionPlanDesc desc{sequence, heads, dim, scale};
    BlockedAttentionPlan plan = BlockedAttentionPlan::create(vk, desc);
    PreparedAttentionInputs prepared = PreparedAttentionInputs::create(vk, desc);
    TensorBatch batch = vk.begin_batch();
    PreparedAttentionView inputs = prepared.prepare(batch, vq, vk_tensor, vv);
    if (sequence == 129) {
      plan.record(batch, inputs, vo, 0, 65, 0);
      plan.record(batch, inputs, vo, 65, 64, 65);
    } else {
      plan.record(batch, inputs, vo);
    }
    batch.submit().wait();
    std::vector<uint16_t> vulkan_output(count);
    vk.download_bytes(vo, vulkan_output.data(), count * 2);
    if (sequence == 129) {
      TensorBatch repeat = vk.begin_batch();
      PreparedAttentionView repeat_inputs =
          prepared.prepare(repeat, vq, vk_tensor, vv);
      plan.record(repeat, repeat_inputs, vo);
      repeat.submit().wait();
      std::vector<uint16_t> vulkan_repeat(count);
      vk.download_bytes(vo, vulkan_repeat.data(), count * 2);
      CHECK(vulkan_repeat == vulkan_output);
    }
    size_t mismatch = count;
    for (size_t i = 0; i < count; ++i) {
      if (cuda_output[i] != vulkan_output[i]) { mismatch = i; break; }
    }
    CHECK_MSG(mismatch == count,
              "CUDA/Vulkan exact attention S%u H%u D%u mismatch at %zu: %04x != %04x",
              sequence, heads, dim, mismatch,
              mismatch == count ? 0u : cuda_output[mismatch],
              mismatch == count ? 0u : vulkan_output[mismatch]);
    return vulkan_output;
  };
  CHECK(cuda::deterministic_attention_grid_fits(1, 1, 1, 1));
  CHECK(cuda::deterministic_attention_grid_fits(65535, 65535, 65535, 65535));
  CHECK(!cuda::deterministic_attention_grid_fits(0, 1, 65535, 65535));
  CHECK(!cuda::deterministic_attention_grid_fits(65536, 1, 65535, 65535));
  CHECK(!cuda::deterministic_attention_grid_fits(1, 65536, 65535, 65535));

  // CUDA hardware and the canonical host conversion agree for every BF16
  // bit pattern. Vulkan's word-owned conversion is checked separately by the
  // Vulkan shader test, so a duplicated conversion bug cannot hide in final
  // attention equality.
  {
    constexpr size_t patterns = 1u << 16;
    std::vector<uint16_t> bits(patterns), got(patterns);
    for (size_t i = 0; i < patterns; ++i) bits[i] = static_cast<uint16_t>(i);
    cuda::DeviceBuffer<uint16_t> source0(patterns), source1(patterns),
        source2(patterns), prepared0(patterns), prepared1(patterns),
        prepared2(patterns);
    source0.copy_from_host(bits.data(), patterns);
    source1.copy_from_host(bits.data(), patterns);
    source2.copy_from_host(bits.data(), patterns);
    bool prepare_alias_rejected = false;
    try {
      cuda::launch_prepare_deterministic_attention_inputs(
          nullptr, reinterpret_cast<const __nv_bfloat16*>(source0.get()),
          reinterpret_cast<const __nv_bfloat16*>(source0.get()),
          reinterpret_cast<const __nv_bfloat16*>(source2.get()),
          reinterpret_cast<__half*>(prepared0.get()),
          reinterpret_cast<__half*>(prepared1.get()),
          reinterpret_cast<__half*>(prepared2.get()), patterns);
    } catch (const std::invalid_argument&) {
      prepare_alias_rejected = true;
    }
    CHECK(prepare_alias_rejected);
    cuda::launch_prepare_deterministic_attention_inputs(
        nullptr, reinterpret_cast<const __nv_bfloat16*>(source0.get()),
        reinterpret_cast<const __nv_bfloat16*>(source1.get()),
        reinterpret_cast<const __nv_bfloat16*>(source2.get()),
        reinterpret_cast<__half*>(prepared0.get()),
        reinterpret_cast<__half*>(prepared1.get()),
        reinterpret_cast<__half*>(prepared2.get()), patterns);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    prepared0.copy_to_host(got.data(), patterns);
    for (size_t i = 0; i < patterns; ++i) {
      const uint16_t expected = (bits[i] & 0x7fffu) > 0x7f80u
          ? 0x7fffu : f32_to_f16(bf16_to_f32(bits[i]));
      CHECK(got[i] == expected);
    }
    bool output_alias_rejected = false;
    try {
      cuda::launch_deterministic_blocked_attention_f16(
          nullptr, reinterpret_cast<const __half*>(prepared0.get()),
          reinterpret_cast<const __half*>(prepared1.get()),
          reinterpret_cast<const __half*>(prepared2.get()),
          reinterpret_cast<__nv_bfloat16*>(prepared0.get()), 1, 1, 64,
          exact_attention_scale(64));
    } catch (const std::invalid_argument&) {
      output_alias_rejected = true;
    }
    CHECK(output_alias_rejected);
  }
  run(1, 2, 64);
  run(17, 3, 72);
  run(129, 2, 128);
  {
    constexpr uint32_t sequence = 129, heads = 1, dim = 64;
    constexpr size_t count = size_t(sequence) * heads * dim;
    std::vector<uint16_t> q(count, f32_to_bf16(1.0f));
    std::vector<uint16_t> k(count, f32_to_bf16(0.0f));
    std::vector<uint16_t> v(count, f32_to_bf16(0.0f));
    for (uint32_t d = 0; d < dim; ++d) {
      k[(sequence - 1) * dim + d] = f32_to_bf16(10.75f);
    }
    for (uint32_t row = 0; row + 1 < sequence; ++row) {
      v[size_t(row) * dim] = f32_to_bf16(0x1p-24f);
    }
    // The first tile accumulates 128*2^-24=2^-17. The next tile raises the
    // score maximum by exactly 86, so its correction makes that old value an
    // fp32 subnormal. Both backends deliberately flush it to signed zero.
    CHECK(std::ldexp(1.0, -17) * std::exp(-86.0) <
          std::numeric_limits<float>::min());
    const std::vector<uint16_t> flushed =
        run(sequence, heads, dim, &q, &k, &v);
    for (uint16_t bits : flushed) CHECK(bits == 0x0000u);
  }

  // A Qwen-vision-shaped D72 tail: preparation is once, then two query-row
  // consumers share it. Uploads/downloads are outside both timings.
  {
    const bool real_shape = std::getenv("SLOPFAB_ATTENTION_REAL_BENCH") != nullptr;
    const uint32_t sequence = real_shape ? 16384u : 257u;
    constexpr uint32_t heads = 16, dim = 72;
    const uint32_t first_rows = (sequence + 1) / 2;
    const uint32_t second_rows = sequence - first_rows;
    const size_t count = size_t(sequence) * heads * dim;
    const int warmups = real_shape ? 0 : 2;
    const int samples = real_shape ? 2 : 5;
    std::vector<uint16_t> host(count, f32_to_bf16(0.03125f));
    cuda::DeviceBuffer<uint16_t> cq(count), ck(count), cv(count), co(count);
    cuda::DeviceBuffer<uint16_t> cq16(count), ck16(count), cv16(count);
    cq.copy_from_host(host.data(), count); ck.copy_from_host(host.data(), count);
    cv.copy_from_host(host.data(), count);
    cudaEvent_t begin{}, end{};
    SLOPFAB_CUDA_CHECK(cudaEventCreate(&begin));
    SLOPFAB_CUDA_CHECK(cudaEventCreate(&end));
    float cuda_prepare_ms = 0.0f, cuda_attention_ms = 0.0f;
    for (int iteration = -warmups; iteration < samples; ++iteration) {
      SLOPFAB_CUDA_CHECK(cudaEventRecord(begin));
      cuda::launch_prepare_deterministic_attention_inputs(
          nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
          reinterpret_cast<const __nv_bfloat16*>(ck.get()),
          reinterpret_cast<const __nv_bfloat16*>(cv.get()),
          reinterpret_cast<__half*>(cq16.get()),
          reinterpret_cast<__half*>(ck16.get()),
          reinterpret_cast<__half*>(cv16.get()), count);
      SLOPFAB_CUDA_CHECK(cudaEventRecord(end));
      SLOPFAB_CUDA_CHECK(cudaEventSynchronize(end));
      float prepare_ms = 0.0f;
      SLOPFAB_CUDA_CHECK(cudaEventElapsedTime(&prepare_ms, begin, end));
      SLOPFAB_CUDA_CHECK(cudaEventRecord(begin));
      cuda::launch_deterministic_blocked_attention_f16(
          nullptr, reinterpret_cast<const __half*>(cq16.get()),
          reinterpret_cast<const __half*>(ck16.get()),
          reinterpret_cast<const __half*>(cv16.get()),
          reinterpret_cast<__nv_bfloat16*>(co.get()), sequence, heads, dim,
          exact_attention_scale(dim), 0, first_rows, 0);
      cuda::launch_deterministic_blocked_attention_f16(
          nullptr, reinterpret_cast<const __half*>(cq16.get()),
          reinterpret_cast<const __half*>(ck16.get()),
          reinterpret_cast<const __half*>(cv16.get()),
          reinterpret_cast<__nv_bfloat16*>(co.get()), sequence, heads, dim,
          exact_attention_scale(dim), first_rows, second_rows, first_rows);
      SLOPFAB_CUDA_CHECK(cudaEventRecord(end));
      SLOPFAB_CUDA_CHECK(cudaEventSynchronize(end));
      float attention_ms = 0.0f;
      SLOPFAB_CUDA_CHECK(cudaEventElapsedTime(&attention_ms, begin, end));
      if (iteration >= 0) {
        cuda_prepare_ms += prepare_ms;
        cuda_attention_ms += attention_ms;
      }
    }
    SLOPFAB_CUDA_CHECK(cudaEventDestroy(begin));
    SLOPFAB_CUDA_CHECK(cudaEventDestroy(end));

    const uint64_t shape[] = {sequence, heads, dim};
    const TensorLayout layout = TensorLayout::contiguous(shape, 3);
    DeviceTensor q = vk.allocate(layout, ScalarType::kBFloat16);
    DeviceTensor k = vk.allocate(layout, ScalarType::kBFloat16);
    DeviceTensor v = vk.allocate(layout, ScalarType::kBFloat16);
    DeviceTensor out = vk.allocate(layout, ScalarType::kBFloat16);
    vk.upload_bytes(q, host.data(), count * 2);
    vk.upload_bytes(k, host.data(), count * 2);
    vk.upload_bytes(v, host.data(), count * 2);
    BlockedAttentionPlanDesc desc{sequence, heads, dim,
                                  exact_attention_scale(dim)};
    BlockedAttentionPlan plan = BlockedAttentionPlan::create(vk, desc);
    PreparedAttentionInputs prepared = PreparedAttentionInputs::create(vk, desc);
    double vulkan_total_ms = 0.0;
    for (int iteration = -warmups; iteration < samples; ++iteration) {
      const auto start = std::chrono::steady_clock::now();
      TensorBatch batch = vk.begin_batch();
      PreparedAttentionView inputs = prepared.prepare(batch, q, k, v);
      plan.record(batch, inputs, out, 0, first_rows, 0);
      plan.record(batch, inputs, out, first_rows, second_rows, first_rows);
      batch.submit().wait();
      if (iteration >= 0) {
        vulkan_total_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
      }
    }
    std::printf("  exact D72 attention S%u H16, 2 query chunks: CUDA prepare %.3f ms + attention %.3f ms, Vulkan total %.3f ms, FP16 slot %.2f MiB\n",
                sequence, cuda_prepare_ms / samples,
                cuda_attention_ms / samples, vulkan_total_ms / samples,
        prepared.reserved_bytes() / (1024.0 * 1024.0));
    if (real_shape) {
      std::vector<uint16_t> cuda_real(count), vulkan_real(count);
      co.copy_to_host(cuda_real.data(), count);
      vk.download_bytes(out, vulkan_real.data(), count * sizeof(uint16_t));
      CHECK(cuda_real == vulkan_real);
    }
  }
}

SLOPFAB_TEST_CATEGORY(cuda_vulkan_exact_h3_attention, "synthetic") {
  if (slopfab::cuda::current_device_compute_capability() != 120) {
    SKIP_UNSUPPORTED_HARDWARE("exact H3 attention requires the shipped SM120 image");
    return;
  }
  if (!slopfab::cuda::deterministic_h3_attention_available()) {
    SKIP_UNSUPPORTED_HARDWARE("exact H3 CUDA driver/runtime tuple is not qualified");
    return;
  }
  using namespace slopfab;
  using namespace slopfab::vulkan;
  const unsigned char board_a[16] = {};
  const unsigned char board_b[16] = {1, 2, 3, 4};
  CHECK(cuda::deterministic_h3_cuda_tuple_fits(
      12, 0, "NVIDIA GeForce RTX 5090", 13010, 13000, 1024, 99328,
      board_a));
  CHECK(cuda::deterministic_h3_cuda_tuple_fits(
      12, 0, "NVIDIA GeForce RTX 5090", 13010, 13000, 1024, 99328,
      board_b));
  CHECK(!cuda::deterministic_h3_cuda_tuple_fits(
      12, 0, "NVIDIA GeForce RTX 5080", 13010, 13000, 1024, 99328,
      board_a));
  CHECK(!cuda::deterministic_h3_cuda_tuple_fits(
      10, 0, "NVIDIA GeForce RTX 5090", 13010, 13000, 1024, 99328,
      board_a));
  CHECK(!cuda::deterministic_h3_cuda_tuple_fits(
      12, 0, "NVIDIA GeForce RTX 5090", 13010, 12090, 1024, 99328,
      board_a));
  CHECK(!cuda::deterministic_h3_cuda_tuple_fits(
      12, 0, "NVIDIA GeForce RTX 5090", 13010, 13000, 512, 99328,
      board_a));
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore");
    return;
  }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  options.enable_shader_float16 = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  if (!vk.exact_h3_attention()) {
    SKIP_UNSUPPORTED_HARDWARE("unavailable prerequisite: !vk.exact_h3_attention()");
    return;
  }

  // The production 124-frame table is built by the same backend-neutral host
  // path used by the transformer. Boundary probes independently derive each
  // tile's selected prefix/frame band rather than trusting the flattened table.
  {
    dit::SequenceLayout real;
    real.num_text = 17;
    real.num_audio_rows = 414;
    real.num_latent_frames = 37;
    real.latent_height = 48;
    real.latent_width = 84;
    real.num_video_rows = real.num_latent_frames * real.rows_per_frame();
    CHECK(real.total_rows() == 37727);
    CHECK(real.video_start() == 431);
    CHECK(real.rows_per_frame() == 1008);
    const dit::BandedKeyRanges table =
        dit::build_banded_key_ranges(real, 9, 128, 64);
    CHECK(table.query_tile == 128 && table.num_query_tiles == 295);
    auto round_down = [](int value) { return value / 64 * 64; };
    auto round_up = [](int value) { return (value + 63) / 64 * 64; };
    const int seq_end = round_up(real.total_rows());
    for (int tile = 0; tile < table.num_query_tiles; ++tile) {
      const int q0 = tile * 128;
      const int qlast = std::min(q0 + 128, real.total_rows()) - 1;
      int expected_lo0 = 0, expected_hi0 = seq_end;
      int expected_lo1 = 0, expected_hi1 = 0;
      if (q0 >= real.video_start()) {
        const int first = (q0 - real.video_start()) / real.rows_per_frame();
        const int last = (qlast - real.video_start()) / real.rows_per_frame();
        expected_hi0 = round_up(real.video_start());
        expected_lo1 = round_down(real.video_start() +
                                  std::max(0, first - 9) * real.rows_per_frame());
        expected_hi1 = std::min(seq_end, round_up(
            real.video_start() + std::min(real.num_latent_frames, last + 10) *
                                     real.rows_per_frame()));
        if (expected_lo1 <= expected_hi0) {
          expected_hi0 = std::max(expected_hi0, expected_hi1);
          expected_lo1 = expected_hi1 = 0;
        }
      }
      const int32_t* got = table.ranges.data() + size_t(tile) * 4;
      CHECK(got[0] == expected_lo0 && got[1] == expected_hi0 &&
            got[2] == expected_lo1 && got[3] == expected_hi1);
      const int probes[] = {0, real.video_start() - 1, real.video_start(),
                            expected_lo1 - 1, expected_lo1,
                            expected_hi1 - 1, expected_hi1,
                            real.total_rows() - 1, real.total_rows()};
      for (int row : probes) {
        if (row < 0 || row >= real.total_rows()) continue;
        const bool selected =
            (row >= got[0] && row < got[1]) ||
            (row >= got[2] && row < got[3]);
        const bool expected =
            (row >= expected_lo0 && row < expected_hi0) ||
            (row >= expected_lo1 && row < expected_hi1);
        CHECK(selected == expected);
      }
    }
    H3AttentionRanges uploaded = H3AttentionRanges::create(
        vk, real.total_rows(), table.ranges.data(),
        static_cast<uint32_t>(table.ranges.size()));
    std::printf("  H3 production +/-9 range table: %u tiles, %zu bytes, FNV64 %016llx\n",
                uploaded.query_tiles(), table.ranges.size() * sizeof(int32_t),
                static_cast<unsigned long long>(uploaded.content_hash()));
    CHECK(uploaded.content_hash() == 0x32b19bc0895faa6aull);
  }

  constexpr uint32_t sequence = 129, heads = 2, dim = 64;
  const size_t count = size_t(sequence) * heads * dim;
  std::vector<uint16_t> hq(count), hk(count), hv(count);
  for (size_t i = 0; i < count; ++i) {
    hq[i] = f32_to_bf16(float(int(i % 29) - 14) / 32.0f);
    hk[i] = f32_to_bf16(float(int(i % 31) - 15) / 32.0f);
    hv[i] = f32_to_bf16(float(int(i % 37) - 18) / 16.0f);
  }
  hq[0] = 0x0001u; hq[1] = 0x807fu;
  hk[0] = 0x007fu; hk[1] = 0x8001u;
  hv[0] = 0x0001u; hv[1] = 0x807fu;
  const std::vector<int32_t> band{
      0, 64, 128, 192,
      0, 128, 0, 0};
  const std::vector<int32_t> wide{
      0, 192, 0, 0,
      0, 192, 0, 0};
  cuda::DeviceBuffer<uint16_t> cq(count), ck(count), cv(count),
      co_full(count), co_band(count), co_wide(count);
  cuda::DeviceBuffer<int32_t> cband(band.size()), cwide(wide.size());
  cq.copy_from_host(hq.data(), hq.size());
  ck.copy_from_host(hk.data(), hk.size());
  cv.copy_from_host(hv.data(), hv.size());
  cband.copy_from_host(band.data(), band.size());
  cwide.copy_from_host(wide.data(), wide.size());
  const float scale = exact_attention_scale(dim);
  cuda::launch_deterministic_h3_attention(
      nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
      reinterpret_cast<const __nv_bfloat16*>(ck.get()),
      reinterpret_cast<const __nv_bfloat16*>(cv.get()),
      reinterpret_cast<__nv_bfloat16*>(co_full.get()), nullptr,
      sequence, heads, dim, scale);
  cuda::launch_deterministic_h3_attention(
      nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
      reinterpret_cast<const __nv_bfloat16*>(ck.get()),
      reinterpret_cast<const __nv_bfloat16*>(cv.get()),
      reinterpret_cast<__nv_bfloat16*>(co_band.get()), cband.get(),
      sequence, heads, dim, scale, 0, 65, 0);
  cuda::launch_deterministic_h3_attention(
      nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
      reinterpret_cast<const __nv_bfloat16*>(ck.get()),
      reinterpret_cast<const __nv_bfloat16*>(cv.get()),
      reinterpret_cast<__nv_bfloat16*>(co_band.get()), cband.get(),
      sequence, heads, dim, scale, 65, 64, 65);
  cuda::launch_deterministic_h3_attention(
      nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
      reinterpret_cast<const __nv_bfloat16*>(ck.get()),
      reinterpret_cast<const __nv_bfloat16*>(cv.get()),
      reinterpret_cast<__nv_bfloat16*>(co_wide.get()), cwide.get(),
      sequence, heads, dim, scale);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> expected_full(count), expected_band(count), expected_wide(count);
  co_full.copy_to_host(expected_full.data(), count);
  co_band.copy_to_host(expected_band.data(), count);
  co_wide.copy_to_host(expected_wide.data(), count);
  CHECK(expected_wide == expected_full);

  const uint64_t shape[] = {sequence, heads, dim};
  const TensorLayout layout = TensorLayout::contiguous(shape, 3);
  DeviceTensor q = vk.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor k = vk.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor v = vk.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor out_full = vk.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor out_band = vk.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor out_wide = vk.allocate(layout, ScalarType::kBFloat16);
  vk.upload_bytes(q, hq.data(), count * 2);
  vk.upload_bytes(k, hk.data(), count * 2);
  vk.upload_bytes(v, hv.data(), count * 2);
  H3AttentionPlan plan = H3AttentionPlan::create(
      vk, {sequence, heads, dim, scale});
  H3AttentionRanges bands = H3AttentionRanges::create(
      vk, sequence, band.data(), static_cast<uint32_t>(band.size()));
  H3AttentionRanges wide_ranges = H3AttentionRanges::create(
      vk, sequence, wide.data(), static_cast<uint32_t>(wide.size()));
  // Touching ranges canonicalize to the same immutable table as full coverage.
  const std::vector<int32_t> touching{
      0, 64, 64, 192,
      0, 128, 128, 192};
  H3AttentionRanges touching_ranges = H3AttentionRanges::create(
      vk, sequence, touching.data(), static_cast<uint32_t>(touching.size()));
  CHECK(touching_ranges.content_hash() == wide_ranges.content_hash());
  TensorBatch batch = vk.begin_batch();
  plan.record(batch, q, k, v, out_full);
  plan.record(batch, q, k, v, out_band, &bands, 0, 65, 0);
  plan.record(batch, q, k, v, out_band, &bands, 65, 64, 65);
  plan.record(batch, q, k, v, out_wide, &wide_ranges);
  batch.submit().wait();
  std::vector<uint16_t> got_full(count), got_band(count), got_wide(count);
  vk.download_bytes(out_full, got_full.data(), count * 2);
  vk.download_bytes(out_band, got_band.data(), count * 2);
  vk.download_bytes(out_wide, got_wide.data(), count * 2);
  CHECK(got_full == expected_full);
  CHECK(got_band == expected_band);
  CHECK(got_wide == expected_wide);
  CHECK(got_wide == got_full);

  // Q/K are zero, so selected values average exactly. Excluded sentinels must
  // not affect the banded result, including across the padded final block.
  std::fill(hq.begin(), hq.end(), f32_to_bf16(0.0f));
  std::fill(hk.begin(), hk.end(), f32_to_bf16(0.0f));
  std::fill(hv.begin(), hv.end(), f32_to_bf16(16.0f));
  for (uint32_t row = 0; row < 64; ++row)
    for (uint32_t h = 0; h < heads; ++h)
      for (uint32_t d = 0; d < dim; ++d)
        hv[(size_t(row) * heads + h) * dim + d] = f32_to_bf16(1.0f);
  for (uint32_t h = 0; h < heads; ++h)
    for (uint32_t d = 0; d < dim; ++d)
      hv[(size_t(128) * heads + h) * dim + d] = f32_to_bf16(1.0f);
  vk.upload_bytes(q, hq.data(), count * 2);
  vk.upload_bytes(k, hk.data(), count * 2);
  vk.upload_bytes(v, hv.data(), count * 2);
  TensorBatch sentinel = vk.begin_batch();
  plan.record(sentinel, q, k, v, out_band, &bands, 0, 1, 0);
  sentinel.submit().wait();
  vk.download_bytes(out_band, got_band.data(), count * 2);
  for (uint32_t i = 0; i < heads * dim; ++i)
    CHECK(got_band[i] == f32_to_bf16(1.0f));
  const std::vector<int32_t> single_values{
      128, 192, 0, 0, 128, 192, 0, 0};
  H3AttentionRanges single = H3AttentionRanges::create(
      vk, sequence, single_values.data(),
      static_cast<uint32_t>(single_values.size()));
  std::fill(hv.begin(), hv.end(), f32_to_bf16(-7.0f));
  for (uint32_t h = 0; h < heads; ++h)
    for (uint32_t d = 0; d < dim; ++d)
      hv[(size_t(128) * heads + h) * dim + d] = f32_to_bf16(1.5f);
  vk.upload_bytes(v, hv.data(), count * 2);
  TensorBatch one_key = vk.begin_batch();
  plan.record(one_key, q, k, v, out_band, &single, 0, 1, 0);
  one_key.submit().wait();
  vk.download_bytes(out_band, got_band.data(), count * 2);
  for (uint32_t i = 0; i < heads * dim; ++i)
    CHECK(got_band[i] == f32_to_bf16(1.5f));

  const std::vector<int32_t> uniform_values{
      0, 64, 0, 0, 0, 64, 0, 0};
  H3AttentionRanges uniform = H3AttentionRanges::create(
      vk, sequence, uniform_values.data(),
      static_cast<uint32_t>(uniform_values.size()));
  for (uint32_t row = 0; row < 64; ++row)
    for (uint32_t h = 0; h < heads; ++h)
      for (uint32_t d = 0; d < dim; ++d)
        hv[(size_t(row) * heads + h) * dim + d] =
            f32_to_bf16(row < 32 ? 1.0f : 3.0f);
  vk.upload_bytes(v, hv.data(), count * 2);
  TensorBatch average = vk.begin_batch();
  plan.record(average, q, k, v, out_band, &uniform, 0, 1, 0);
  average.submit().wait();
  vk.download_bytes(out_band, got_band.data(), count * 2);
  for (uint32_t i = 0; i < heads * dim; ++i)
    CHECK(got_band[i] == f32_to_bf16(2.0f));

  // D128, three global query tiles, every tile/64-key boundary, row chunks,
  // and a nonzero output offset. The CUDA full launch is the exact oracle;
  // Vulkan consumes the same immutable table across all chunk records.
  {
    constexpr uint32_t s = 257, h = 1, d = 128;
    const size_t n = size_t(s) * h * d;
    std::vector<uint16_t> qh(n), kh(n), vh(n);
    for (size_t i = 0; i < n; ++i) {
      qh[i] = f32_to_bf16(float(int(i % 23) - 11) / 32.0f);
      kh[i] = f32_to_bf16(float(int(i % 27) - 13) / 32.0f);
      vh[i] = f32_to_bf16(float(int(i % 33) - 16) / 16.0f);
    }
    const std::vector<int32_t> r{
        0, 64, 128, 192,
        0, 128, 192, 256,
        0, 320, 0, 0};
    cuda::DeviceBuffer<uint16_t> dq(n), dk(n), dv(n), dout(n);
    cuda::DeviceBuffer<int32_t> dr(r.size());
    dq.copy_from_host(qh.data(), n); dk.copy_from_host(kh.data(), n);
    dv.copy_from_host(vh.data(), n); dr.copy_from_host(r.data(), r.size());
    cuda::launch_deterministic_h3_attention(
        nullptr, reinterpret_cast<const __nv_bfloat16*>(dq.get()),
        reinterpret_cast<const __nv_bfloat16*>(dk.get()),
        reinterpret_cast<const __nv_bfloat16*>(dv.get()),
        reinterpret_cast<__nv_bfloat16*>(dout.get()), dr.get(), s, h, d,
        exact_attention_scale(d));
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> expected(n);
    dout.copy_to_host(expected.data(), n);
    const uint64_t sshape[] = {s, h, d};
    TensorLayout slayout = TensorLayout::contiguous(sshape, 3);
    DeviceTensor vq = vk.allocate(slayout, ScalarType::kBFloat16);
    DeviceTensor vkey = vk.allocate(slayout, ScalarType::kBFloat16);
    DeviceTensor vv = vk.allocate(slayout, ScalarType::kBFloat16);
    DeviceTensor vo = vk.allocate(slayout, ScalarType::kBFloat16);
    vk.upload_bytes(vq, qh.data(), n * 2);
    vk.upload_bytes(vkey, kh.data(), n * 2);
    vk.upload_bytes(vv, vh.data(), n * 2);
    H3AttentionPlan p = H3AttentionPlan::create(
        vk, {s, h, d, exact_attention_scale(d)});
    H3AttentionRanges vr = H3AttentionRanges::create(
        vk, s, r.data(), static_cast<uint32_t>(r.size()));
    TensorBatch chunks = vk.begin_batch();
    const uint32_t starts[] = {0, 1, 15, 16, 63, 64, 65, 127, 128, 129};
    const uint32_t lengths[] = {1, 14, 1, 47, 1, 1, 62, 1, 1, 128};
    for (size_t i = 0; i < std::size(starts); ++i)
      p.record(chunks, vq, vkey, vv, vo, &vr, starts[i], lengths[i], starts[i]);
    chunks.submit().wait();
    std::vector<uint16_t> got(n);
    vk.download_bytes(vo, got.data(), n * 2);
    CHECK(got == expected);
    const uint16_t sentinel_bits = f32_to_bf16(-123.0f);
    std::fill(got.begin(), got.end(), sentinel_bits);
    vk.upload_bytes(vo, got.data(), n * 2);
    TensorBatch offset = vk.begin_batch();
    p.record(offset, vq, vkey, vv, vo, &vr, 128, 1, 0);
    offset.submit().wait();
    vk.download_bytes(vo, got.data(), n * 2);
    CHECK(std::memcmp(got.data(), expected.data() + size_t(128) * d,
                      d * sizeof(uint16_t)) == 0);
    for (size_t i = d; i < n; ++i) CHECK(got[i] == sentinel_bits);

    // Independent two-range/two-block recurrence anchor. The second range's
    // score is >87 above the first, so the specified exp cutoff makes the old
    // block correction exactly zero and every output is exactly V=3.
    std::fill(qh.begin(), qh.end(), f32_to_bf16(1.0f));
    std::fill(kh.begin(), kh.end(), f32_to_bf16(0.0f));
    std::fill(vh.begin(), vh.end(), f32_to_bf16(1.0f));
    for (uint32_t column = 0; column < d; ++column) {
      kh[size_t(128) * d + column] = f32_to_bf16(8.0f);
      vh[size_t(128) * d + column] = f32_to_bf16(3.0f);
    }
    vk.upload_bytes(vq, qh.data(), n * 2);
    vk.upload_bytes(vkey, kh.data(), n * 2);
    vk.upload_bytes(vv, vh.data(), n * 2);
    TensorBatch seam = vk.begin_batch();
    p.record(seam, vq, vkey, vv, vo, &vr, 0, 1, 0);
    seam.submit().wait();
    vk.download_bytes(vo, got.data(), n * 2);
    for (uint32_t column = 0; column < d; ++column)
      CHECK(got[column] == f32_to_bf16(3.0f));

    // Three ordered blocks with successively larger finite maxima force two
    // strictly-between-zero-and-one online corrections. This stresses a
    // nonzero cooperative C tile rather than only the first-block/zero-cutoff
    // cases; CUDA and Vulkan must repeat exactly and the independent bound
    // excludes either dropped-old-state or reset-at-range-seam outcomes.
    const std::vector<int32_t> correction_values{
        0, 128, 192, 256, 0, 128, 192, 256, 0, 256, 0, 0};
    cuda::DeviceBuffer<int32_t> correction_device(correction_values.size());
    correction_device.copy_from_host(correction_values.data(),
                                     correction_values.size());
    H3AttentionRanges correction_ranges = H3AttentionRanges::create(
        vk, s, correction_values.data(),
        static_cast<uint32_t>(correction_values.size()));
    std::fill(qh.begin(), qh.end(), f32_to_bf16(1.0f));
    std::fill(kh.begin(), kh.end(), f32_to_bf16(0.0f));
    std::fill(vh.begin(), vh.end(), f32_to_bf16(1.0f));
    for (uint32_t row = 64; row < 128; ++row) {
      for (uint32_t column = 0; column < d; ++column) {
        kh[size_t(row) * d + column] = f32_to_bf16(1.0f / 64.0f);
        vh[size_t(row) * d + column] = f32_to_bf16(2.0f);
      }
    }
    for (uint32_t row = 192; row < 256; ++row) {
      for (uint32_t column = 0; column < d; ++column) {
        kh[size_t(row) * d + column] = f32_to_bf16(1.0f / 32.0f);
        vh[size_t(row) * d + column] = f32_to_bf16(3.0f);
      }
    }
    dq.copy_from_host(qh.data(), n); dk.copy_from_host(kh.data(), n);
    dv.copy_from_host(vh.data(), n);
    cuda::launch_deterministic_h3_attention(
        nullptr, reinterpret_cast<const __nv_bfloat16*>(dq.get()),
        reinterpret_cast<const __nv_bfloat16*>(dk.get()),
        reinterpret_cast<const __nv_bfloat16*>(dv.get()),
        reinterpret_cast<__nv_bfloat16*>(dout.get()), correction_device.get(),
        s, h, d, exact_attention_scale(d), 0, 1, 0);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    dout.copy_to_host(expected.data(), n);
    vk.upload_bytes(vq, qh.data(), n * 2);
    vk.upload_bytes(vkey, kh.data(), n * 2);
    vk.upload_bytes(vv, vh.data(), n * 2);
    TensorBatch correction_batch = vk.begin_batch();
    p.record(correction_batch, vq, vkey, vv, vo, &correction_ranges, 0, 1, 0);
    correction_batch.submit().wait();
    vk.download_bytes(vo, got.data(), n * 2);
    for (uint32_t column = 0; column < d; ++column) {
      CHECK(got[column] == expected[column]);
      CHECK(bf16_to_f32(got[column]) > 1.0f &&
            bf16_to_f32(got[column]) < 3.0f);
    }
  }

  // Cooperative-MMA tuple corpus: signed zeros, the minimum normal boundary,
  // mixed exponents, cancellation and one-ULP-neighbor operands exercise QK
  // and PV rounding for both supported head widths. Repeated independent
  // submissions pin the empirically qualified WMMA/KHR internal semantics.
  for (uint32_t adversarial_dim : {64u, 128u}) {
    constexpr uint32_t adversarial_sequence = 65;
    const size_t adversarial_count =
        size_t(adversarial_sequence) * adversarial_dim;
    const uint16_t patterns[] = {
        0x0000u, 0x8000u, 0x0080u, 0x8080u, 0x3f80u, 0xbf80u,
        0x3f81u, 0xbf81u, 0x3f00u, 0xbf00u, 0x4000u, 0xc000u,
        0x3c00u, 0xbc00u, 0x3eabu, 0xbeabu};
    std::vector<uint16_t> aq(adversarial_count), ak(adversarial_count),
        av(adversarial_count);
    for (size_t i = 0; i < adversarial_count; ++i) {
      aq[i] = patterns[i % std::size(patterns)];
      ak[i] = patterns[(i * 5 + (i / adversarial_dim)) % std::size(patterns)];
      av[i] = patterns[(i * 7 + 3) % std::size(patterns)];
    }
    cuda::DeviceBuffer<uint16_t> daq(adversarial_count), dak(adversarial_count),
        dav(adversarial_count), dao(adversarial_count), dar(adversarial_count);
    daq.copy_from_host(aq.data(), adversarial_count);
    dak.copy_from_host(ak.data(), adversarial_count);
    dav.copy_from_host(av.data(), adversarial_count);
    auto launch_adversarial = [&](cuda::DeviceBuffer<uint16_t>& selected) {
      cuda::launch_deterministic_h3_attention(
          nullptr, reinterpret_cast<const __nv_bfloat16*>(daq.get()),
          reinterpret_cast<const __nv_bfloat16*>(dak.get()),
          reinterpret_cast<const __nv_bfloat16*>(dav.get()),
          reinterpret_cast<__nv_bfloat16*>(selected.get()), nullptr,
          adversarial_sequence, 1, adversarial_dim,
          exact_attention_scale(adversarial_dim));
    };
    launch_adversarial(dao); launch_adversarial(dar);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> adversarial_expected(adversarial_count),
        adversarial_repeat(adversarial_count);
    dao.copy_to_host(adversarial_expected.data(), adversarial_count);
    dar.copy_to_host(adversarial_repeat.data(), adversarial_count);
    CHECK(adversarial_repeat == adversarial_expected);
    const uint64_t ashape[] = {adversarial_sequence, 1, adversarial_dim};
    const TensorLayout alayout = TensorLayout::contiguous(ashape, 3);
    DeviceTensor vaq = vk.allocate(alayout, ScalarType::kBFloat16);
    DeviceTensor vak = vk.allocate(alayout, ScalarType::kBFloat16);
    DeviceTensor vav = vk.allocate(alayout, ScalarType::kBFloat16);
    DeviceTensor vao = vk.allocate(alayout, ScalarType::kBFloat16);
    DeviceTensor var = vk.allocate(alayout, ScalarType::kBFloat16);
    vk.upload_bytes(vaq, aq.data(), adversarial_count * 2);
    vk.upload_bytes(vak, ak.data(), adversarial_count * 2);
    vk.upload_bytes(vav, av.data(), adversarial_count * 2);
    H3AttentionPlan adversarial_plan = H3AttentionPlan::create(
        vk, {adversarial_sequence, 1, adversarial_dim,
             exact_attention_scale(adversarial_dim)});
    TensorBatch adversarial_batch = vk.begin_batch();
    adversarial_plan.record(adversarial_batch, vaq, vak, vav, vao);
    adversarial_plan.record(adversarial_batch, vaq, vak, vav, var);
    adversarial_batch.submit().wait();
    std::vector<uint16_t> adversarial_got(adversarial_count);
    vk.download_bytes(vao, adversarial_got.data(), adversarial_count * 2);
    CHECK(adversarial_got == adversarial_expected);
    vk.download_bytes(var, adversarial_got.data(), adversarial_count * 2);
    CHECK(adversarial_got == adversarial_expected);
  }

  // Range validation is a setup boundary and cannot mutate a recorder. Empty,
  // misaligned, reversed, padding-only and wrong-count tables fail closed.
  auto range_rejected = [&](const std::vector<int32_t>& bad) {
    bool rejected = false;
    try {
      (void)H3AttentionRanges::create(
          vk, sequence, bad.data(), static_cast<uint32_t>(bad.size()));
    } catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
  };
  range_rejected({0, 0, 0, 0, 0, 128, 0, 0});
  range_rejected({1, 64, 0, 0, 0, 128, 0, 0});
  range_rejected({64, 0, 0, 0, 0, 128, 0, 0});
  range_rejected({192, 192, 0, 0, 0, 128, 0, 0});
  range_rejected({0, 64, 0, 0});

  // Rejected record calls leave the batch usable; operator overflow poisons
  // the recording. Repeated two-flight/oldest-slot reuse stays bounded.
  DeviceTensor wrong = vk.allocate(layout, ScalarType::kFloat32);
  {
    TensorBatch recover = vk.begin_batch();
    bool rejected = false;
    try { plan.record(recover, q, k, v, wrong, &bands); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    rejected = false;
    try { plan.record(recover, q, k, v, q, &bands); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    rejected = false;
    try { plan.record(recover, q, k, v, out_full, &bands, sequence, 1, 0); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    plan.record(recover, q, k, v, out_full, &bands);
    TensorBatch moved = std::move(recover);
    moved.submit().wait();
  }
  DeviceTensor second = vk.allocate(layout, ScalarType::kBFloat16);
  auto submit = [&](DeviceTensor& selected,
                    const H3AttentionRanges* selected_ranges) {
    TensorBatch selected_batch = vk.begin_batch();
    plan.record(selected_batch, q, k, v, selected, selected_ranges);
    return selected_batch.submit();
  };
  Submission first = submit(out_full, &bands),
             second_job = submit(second, &bands),
             third = submit(out_full, &bands);
  CHECK(second_job.value() > first.value() && third.value() > second_job.value());
  first.wait(); second_job.wait(); third.wait();
  std::vector<uint16_t> first_output(count), second_output(count);
  vk.download_bytes(out_full, first_output.data(), count * 2);
  vk.download_bytes(second, second_output.data(), count * 2);
  CHECK(first_output == second_output);
  Submission full_a = submit(out_full, nullptr),
             full_b = submit(second, nullptr),
             full_c = submit(out_full, nullptr);
  full_a.wait(); full_b.wait(); full_c.wait();
  const uint64_t stable_reserved = vk.reserved_bytes();
  const uint64_t stable_descriptors = vk.descriptor_set_allocations();
  for (int repeat = 0; repeat < 50; ++repeat) {
    const H3AttentionRanges* selected = (repeat & 1) ? &bands : nullptr;
    Submission a = submit(out_full, selected), b = submit(second, selected),
               c = submit(out_full, selected);
    a.wait(); b.wait(); c.wait();
    CHECK(vk.reserved_bytes() == stable_reserved);
    CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  }
  {
    TensorBatch full = vk.begin_batch();
    for (int op = 0; op < 32; ++op)
      plan.record(full, q, k, v, out_full, &bands, 0, 1, 0);
    full.submit().wait();
  }
  {
    TensorBatch overflow = vk.begin_batch();
    for (int op = 0; op < 32; ++op)
      plan.record(overflow, q, k, v, out_full, &bands, 0, 1, 0);
    bool rejected = false;
    try { plan.record(overflow, q, k, v, out_full, &bands, 0, 1, 0); }
    catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected);
    bool submit_rejected = false;
    try { (void)overflow.submit(); }
    catch (const std::logic_error&) { submit_rejected = true; }
    CHECK(submit_rejected);
  }

  // A discarded recorder releases speculative resources. A submitted job
  // retains the immutable table, plan and all four tensors after every public
  // wrapper drops, then releases them after its exact token is collected.
  const uint64_t used_before_drop = vk.pooled_used_bytes();
  {
    DeviceTensor temporary = vk.allocate(layout, ScalarType::kBFloat16);
    TensorBatch discarded = vk.begin_batch();
    plan.record(discarded, q, k, v, temporary, &bands, 0, 1, 0);
  }
  CHECK(vk.pooled_used_bytes() == used_before_drop);
  Submission retained;
  {
    DeviceTensor tq = vk.allocate(layout, ScalarType::kBFloat16);
    DeviceTensor tk = vk.allocate(layout, ScalarType::kBFloat16);
    DeviceTensor tv = vk.allocate(layout, ScalarType::kBFloat16);
    DeviceTensor tout = vk.allocate(layout, ScalarType::kBFloat16);
    vk.upload_bytes(tq, hq.data(), count * 2);
    vk.upload_bytes(tk, hk.data(), count * 2);
    vk.upload_bytes(tv, hv.data(), count * 2);
    H3AttentionPlan temporary_plan = H3AttentionPlan::create(
        vk, {sequence, heads, dim, scale});
    H3AttentionRanges temporary_ranges = H3AttentionRanges::create(
        vk, sequence, band.data(), static_cast<uint32_t>(band.size()));
    TensorBatch keep = vk.begin_batch();
    temporary_plan.record(keep, tq, tk, tv, tout, &temporary_ranges, 0, 1, 0);
    retained = keep.submit();
  }
  CHECK(vk.pooled_used_bytes() > used_before_drop);
  retained.wait();
  retained = Submission{};
  { TensorBatch collect = vk.begin_batch(); }
  CHECK(vk.pooled_used_bytes() == used_before_drop);
}
