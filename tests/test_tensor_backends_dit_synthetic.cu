#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_vulkan_dit_exact_pointwise, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 || !Instance::available()");
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: physical.empty() || !physical.front().info().timeline_semaphore");
    return;
  }
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContextOptions context_options;
  context_options.max_batch_operators = 16;
  TensorContext vk(device, context_options);
  if (!vk.exact_vae_pointwise() || !vk.exact_fp32_vae_normalization()) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: !vk.exact_vae_pointwise() || !vk.exact_fp32_vae_normalization()");
    return;
  }

  constexpr uint32_t rows = 5, dim = 13, timesteps = 2, modalities = 3, params = 6, rank = 3;
  constexpr uint32_t mod_rows = timesteps * modalities, features = modalities * params * dim;
  constexpr uint32_t inner = 11;
  std::vector<float> weight(size_t(features) * rank), bias(features), code(timesteps * rank);
  for (size_t i = 0; i < weight.size(); ++i)
    weight[i] = float(int(i % 17) - 8) / 64.0f;
  for (size_t i = 0; i < bias.size(); ++i)
    bias[i] = float(int(i % 11) - 5) / 32.0f;
  for (size_t i = 0; i < code.size(); ++i)
    code[i] = float(int(i % 7) - 3) / 8.0f;
  std::vector<uint16_t> residual(size_t(rows) * dim), branch(residual.size());
  std::vector<uint16_t> norm_input(residual.size()), norm_weight(dim);
  std::vector<int32_t> selectors{0, 5, 2, 3, 1};
  for (size_t i = 0; i < residual.size(); ++i) {
    residual[i] = f32_to_bf16(float(int(i % 23) - 11) / 16.0f);
    branch[i] = f32_to_bf16(float(int(i % 19) - 9) / 32.0f);
    norm_input[i] = f32_to_bf16(float(int(i % 29) - 14) / 16.0f);
  }
  for (uint32_t i = 0; i < dim; ++i)
    norm_weight[i] = f32_to_bf16(0.75f + float(i % 5) / 16.0f);
  std::vector<uint16_t> fused(size_t(rows) * 2 * inner);
  for (size_t i = 0; i < fused.size(); ++i)
    fused[i] = f32_to_bf16(float(int(i % 31) - 15) / 8.0f);
  std::vector<uint16_t> text_residual(size_t(rows) * dim), text_branch(text_residual.size());
  std::vector<uint16_t> text_gate(size_t(rows) * inner), text_up(text_gate.size());
  for (size_t i = 0; i < text_residual.size(); ++i) {
    text_residual[i] = f32_to_bf16(float(int(i % 37) - 18) / 16.0f);
    text_branch[i] = f32_to_bf16(float(int(i % 29) - 14) / 32.0f);
  }
  for (size_t i = 0; i < text_gate.size(); ++i) {
    text_gate[i] = f32_to_bf16(float(int(i % 31) - 15) / 8.0f);
    text_up[i] = f32_to_bf16(float(int(i % 23) - 11) / 16.0f);
  }
  const std::array<uint16_t, 8> exceptional{0x0001u, 0x8001u, 0x0000u, 0x8000u,
                                            0x7f80u, 0xff80u, 0x7fc1u, 0x7f7fu};
  for (size_t i = 0; i < exceptional.size(); ++i) {
    text_residual[i] = exceptional[i];
    text_branch[i] = exceptional[exceptional.size() - 1 - i];
    text_gate[i] = exceptional[i];
    text_up[i] = exceptional[exceptional.size() - 1 - i];
  }

  cuda::DeviceBuffer<float> cw(weight.size()), cb(bias.size()), cc(code.size()),
      cm(size_t(params) * mod_rows * dim);
  cuda::DeviceBuffer<uint16_t> cx(residual.size()), cbranch(branch.size()),
      cnorm(norm_input.size()), cnorm_w(norm_weight.size()), cnorm_out(norm_input.size()),
      cfused(fused.size()), cswiglu(size_t(rows) * inner);
  cuda::DeviceBuffer<uint16_t> ctext_x(text_residual.size()), ctext_branch(text_branch.size()),
      ctext_gate(text_gate.size()), ctext_up(text_up.size()), ctext_out(text_gate.size());
  cuda::DeviceBuffer<int32_t> cselectors(selectors.size());
  cw.copy_from_host(weight.data(), weight.size());
  cb.copy_from_host(bias.data(), bias.size());
  cc.copy_from_host(code.data(), code.size());
  cx.copy_from_host(residual.data(), residual.size());
  cbranch.copy_from_host(branch.data(), branch.size());
  cnorm.copy_from_host(norm_input.data(), norm_input.size());
  cnorm_w.copy_from_host(norm_weight.data(), norm_weight.size());
  cselectors.copy_from_host(selectors.data(), selectors.size());
  cfused.copy_from_host(fused.data(), fused.size());
  ctext_x.copy_from_host(text_residual.data(), text_residual.size());
  ctext_branch.copy_from_host(text_branch.data(), text_branch.size());
  ctext_gate.copy_from_host(text_gate.data(), text_gate.size());
  ctext_up.copy_from_host(text_up.data(), text_up.size());
  cuda::launch_adaln_expand(cw.get(), cb.get(), cc.get(), cm.get(), timesteps, modalities, params,
                            dim, rank, nullptr);
  cuda::launch_add_gated(reinterpret_cast<__nv_bfloat16*>(cx.get()),
                         reinterpret_cast<const __nv_bfloat16*>(cbranch.get()),
                         cm.get() + size_t(2) * mod_rows * dim, cselectors.get(), rows, dim,
                         nullptr);
  cuda::launch_rmsnorm_modulate(
      reinterpret_cast<const __nv_bfloat16*>(cnorm.get()),
      reinterpret_cast<const __nv_bfloat16*>(cnorm_w.get()), cm.get() + size_t(4) * mod_rows * dim,
      cm.get() + size_t(3) * mod_rows * dim, cselectors.get(),
      reinterpret_cast<__nv_bfloat16*>(cnorm_out.get()), rows, dim, 1e-6f, nullptr);
  cuda::launch_swiglu_exact(reinterpret_cast<const __nv_bfloat16*>(cfused.get()),
                            reinterpret_cast<__nv_bfloat16*>(cswiglu.get()), rows, inner, nullptr);
  text::launch_residual_add_exact(reinterpret_cast<__nv_bfloat16*>(ctext_x.get()),
                                  reinterpret_cast<const __nv_bfloat16*>(ctext_branch.get()),
                                  text_residual.size(), nullptr);
  text::launch_swiglu_split_exact(reinterpret_cast<const __nv_bfloat16*>(ctext_gate.get()),
                                  reinterpret_cast<const __nv_bfloat16*>(ctext_up.get()),
                                  reinterpret_cast<__nv_bfloat16*>(ctext_out.get()),
                                  text_gate.size(), nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());

  auto mat = [&](uint64_t a, uint64_t b) {
    const uint64_t s[] = {a, b};
    return TensorLayout::contiguous(s, 2);
  };
  auto vec = [&](uint64_t n) {
    return TensorLayout::contiguous(&n, 1);
  };
  DeviceTensor vw = vk.allocate(mat(features, rank)), vb = vk.allocate(vec(features)),
               vc = vk.allocate(mat(timesteps, rank));
  const uint64_t logical_table = uint64_t(mod_rows) * dim;
  const uint64_t align_elements = std::max<uint64_t>(1, vk.storage_binding_alignment() / 4);
  const uint64_t table_stride =
      (logical_table + align_elements - 1) / align_elements * align_elements;
  DeviceTensor vm = vk.allocate(vec(params * table_stride));
  DeviceTensor vx = vk.allocate(mat(rows, dim), ScalarType::kBFloat16),
               vbranch = vk.allocate(mat(rows, dim), ScalarType::kBFloat16);
  DeviceTensor vn = vk.allocate(mat(rows, dim), ScalarType::kBFloat16),
               vnw = vk.allocate(vec(dim), ScalarType::kBFloat16),
               vno = vk.allocate(mat(rows, dim), ScalarType::kBFloat16);
  DeviceTensor va = vk.allocate(vec(rows), ScalarType::kInt32),
               vf = vk.allocate(mat(rows, 2 * inner), ScalarType::kBFloat16),
               vso = vk.allocate(mat(rows, inner), ScalarType::kBFloat16);
  DeviceTensor vtext_x = vk.allocate(mat(rows, dim), ScalarType::kBFloat16),
               vtext_branch = vk.allocate(mat(rows, dim), ScalarType::kBFloat16),
               vtext_gate = vk.allocate(mat(rows, inner), ScalarType::kBFloat16),
               vtext_up = vk.allocate(mat(rows, inner), ScalarType::kBFloat16),
               vtext_out = vk.allocate(mat(rows, inner), ScalarType::kBFloat16);
  vk.upload(vw, weight.data(), weight.size());
  vk.upload(vb, bias.data(), bias.size());
  vk.upload(vc, code.data(), code.size());
  vk.upload_bytes(vx, residual.data(), residual.size() * 2);
  vk.upload_bytes(vbranch, branch.data(), branch.size() * 2);
  vk.upload_bytes(vn, norm_input.data(), norm_input.size() * 2);
  vk.upload_bytes(vnw, norm_weight.data(), norm_weight.size() * 2);
  vk.upload_bytes(va, selectors.data(), selectors.size() * 4);
  vk.upload_bytes(vf, fused.data(), fused.size() * 2);
  vk.upload_bytes(vtext_x, text_residual.data(), text_residual.size() * 2);
  vk.upload_bytes(vtext_branch, text_branch.data(), text_branch.size() * 2);
  vk.upload_bytes(vtext_gate, text_gate.data(), text_gate.size() * 2);
  vk.upload_bytes(vtext_up, text_up.data(), text_up.size() * 2);
  TensorBatch batch = vk.begin_batch();
  batch.dit_expand_adaln(vw, vb, vc, vm, modalities, params, dim);
  batch.dit_add_gated_bf16_table(vx, vbranch, vm, mod_rows, 2, va);
  batch.rms_norm_modulate_bf16_table(vn, vnw, vm, mod_rows, 4, 3, va, vno, 1e-6f);
  batch.dit_swiglu_bf16(vf, vso);
  batch.text_add_residual_bf16(vtext_x, vtext_branch);
  batch.text_swiglu_split_bf16(vtext_gate, vtext_up, vtext_out);
  batch.submit().wait();
  std::vector<float> cuda_mod(size_t(params) * mod_rows * dim), vk_mod(cuda_mod.size()),
      vk_arena(params * table_stride);
  std::vector<uint16_t> cuda_x(residual.size()), vk_x(residual.size()), cuda_n(norm_input.size()),
      vk_n(norm_input.size()), cuda_s(size_t(rows) * inner), vk_s(cuda_s.size());
  std::vector<uint16_t> cuda_text_x(text_residual.size()), vk_text_x(text_residual.size()),
      cuda_text_out(text_gate.size()), vk_text_out(text_gate.size());
  cm.copy_to_host(cuda_mod.data(), cuda_mod.size());
  cx.copy_to_host(cuda_x.data(), cuda_x.size());
  cnorm_out.copy_to_host(cuda_n.data(), cuda_n.size());
  cswiglu.copy_to_host(cuda_s.data(), cuda_s.size());
  ctext_x.copy_to_host(cuda_text_x.data(), cuda_text_x.size());
  ctext_out.copy_to_host(cuda_text_out.data(), cuda_text_out.size());
  vk.download(vm, vk_arena.data(), vk_arena.size());
  for (uint32_t table = 0; table < params; ++table)
    std::memcpy(vk_mod.data() + table * logical_table, vk_arena.data() + table * table_stride,
                logical_table * 4);
  vk.download_bytes(vx, vk_x.data(), vk_x.size() * 2);
  vk.download_bytes(vno, vk_n.data(), vk_n.size() * 2);
  vk.download_bytes(vso, vk_s.data(), vk_s.size() * 2);
  vk.download_bytes(vtext_x, vk_text_x.data(), vk_text_x.size() * 2);
  vk.download_bytes(vtext_out, vk_text_out.data(), vk_text_out.size() * 2);
  CHECK(std::memcmp(cuda_mod.data(), vk_mod.data(), cuda_mod.size() * 4) == 0);
  CHECK(std::memcmp(cuda_x.data(), vk_x.data(), cuda_x.size() * 2) == 0);
  CHECK(std::memcmp(cuda_n.data(), vk_n.data(), cuda_n.size() * 2) == 0);
  CHECK(std::memcmp(cuda_s.data(), vk_s.data(), cuda_s.size() * 2) == 0);
  CHECK(std::memcmp(cuda_text_x.data(), vk_text_x.data(), cuda_text_x.size() * 2) == 0);
  CHECK(std::memcmp(cuda_text_out.data(), vk_text_out.data(), cuda_text_out.size() * 2) == 0);

  // The text layer reuses one dense BF16 slot for every compressed matrix.
  // Exercise I8 here as well as the existing real-NVFP4 streaming test so a
  // format-specific scale lookup cannot regress the shared cache.
  constexpr uint32_t stream_rows = 3, stream_in = 17, stream_out = 13;
  std::vector<int8_t> i8(size_t(stream_out) * stream_in);
  std::vector<float> i8_scale(stream_out);
  std::vector<uint16_t> stream_input(size_t(stream_rows) * stream_in);
  for (size_t i = 0; i < i8.size(); ++i)
    i8[i] = static_cast<int8_t>(int(i % 17) - 8);
  for (uint32_t i = 0; i < stream_out; ++i)
    i8_scale[i] = float(i + 1) / 64.0f;
  for (size_t i = 0; i < stream_input.size(); ++i)
    stream_input[i] = f32_to_bf16(float(int(i % 19) - 9) / 16.0f);
  LinearWeightUpload i8_upload;
  i8_upload.format = LinearWeightFormat::kInt8;
  i8_upload.out_features = stream_out;
  i8_upload.in_features = stream_in;
  i8_upload.data = i8.data();
  i8_upload.data_bytes = i8.size();
  i8_upload.weight_scale = i8_scale.data();
  i8_upload.weight_scale_count = i8_scale.size();
  LinearWeight i8_weight = LinearWeight::upload(vk, i8_upload);
  DeviceTensor stream_i = vk.allocate(mat(stream_rows, stream_in), ScalarType::kBFloat16);
  DeviceTensor stream_dense = vk.allocate(mat(stream_out, stream_in), ScalarType::kBFloat16);
  DeviceTensor stream_expected = vk.allocate(mat(stream_rows, stream_out), ScalarType::kBFloat16);
  DeviceTensor stream_actual = vk.allocate(mat(stream_rows, stream_out), ScalarType::kBFloat16);
  vk.upload_bytes(stream_i, stream_input.data(), stream_input.size() * 2);
  DenseGemmPlan stream_plan = DenseGemmPlan::create(
      vk, {stream_rows, stream_out, stream_in, DenseGemmMode::kBFloat16, DenseGemmBias::kNone});
  {
    TensorBatch direct = vk.begin_batch();
    i8_weight.materialize_bf16(direct, stream_dense);
    stream_plan.record(direct, stream_i, stream_dense, stream_expected, stream_rows);
    direct.submit().wait();
  }
  StreamedNVFP4WeightCache stream_cache =
      StreamedNVFP4WeightCache::create(vk, uint64_t(stream_out) * stream_in);
  TensorContext foreign_vk(device);
  LinearWeight foreign_i8_weight = LinearWeight::upload(foreign_vk, i8_upload);
  DenseGemmPlan wrong_stream_plan = DenseGemmPlan::create(
      vk, {stream_rows, stream_out, stream_in + 1, DenseGemmMode::kBFloat16, DenseGemmBias::kNone});
  {
    TensorBatch streamed = vk.begin_batch();
    PreparedNVFP4WeightView prepared = stream_cache.prepare(streamed, i8_weight, stream_plan);
    const uint32_t capacity = streamed.remaining_operator_capacity();
    bool foreign_rejected = false;
    try {
      (void)stream_cache.prepare(streamed, foreign_i8_weight, stream_plan);
    } catch (const std::invalid_argument&) {
      foreign_rejected = true;
    }
    CHECK(foreign_rejected);
    CHECK(streamed.remaining_operator_capacity() == capacity);
    bool shape_rejected = false;
    try {
      (void)stream_cache.prepare(streamed, i8_weight, wrong_stream_plan);
    } catch (const std::invalid_argument&) {
      shape_rejected = true;
    }
    CHECK(shape_rejected);
    CHECK(streamed.remaining_operator_capacity() == capacity);
    // Rejections preserve the exact generation/layout/access state.
    stream_plan.record(streamed, stream_i, prepared, stream_actual, stream_rows);
    streamed.submit().wait();
  }
  std::vector<uint16_t> stream_expected_bits(size_t(stream_rows) * stream_out),
      stream_actual_bits(stream_expected_bits.size());
  vk.download_bytes(stream_expected, stream_expected_bits.data(), stream_expected_bits.size() * 2);
  vk.download_bytes(stream_actual, stream_actual_bits.data(), stream_actual_bits.size() * 2);
  CHECK(stream_expected_bits == stream_actual_bits);
}
