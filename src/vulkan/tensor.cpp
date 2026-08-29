#include "vidfab/vulkan/tensor.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "embedded_tensor_spv.h"
#include "tensor_validation.h"
#include "vidfab/vulkan/compute.h"
#include "vidfab/vulkan/gemm.h"
#include "vidfab/vulkan/linear.h"

namespace vidfab::vulkan {
namespace {

constexpr uint64_t kMaxExactNormDimension = 1ull << 24;

uint64_t checked_multiply(uint64_t a, uint64_t b, const char* operation) {
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) {
    throw std::overflow_error(std::string("vulkan tensor: ") + operation +
                              " shape overflow");
  }
  return a * b;
}

uintptr_t next_context_identity() {
  static std::atomic<uintptr_t> next{1};
  uintptr_t result = next.load(std::memory_order_relaxed);
  for (;;) {
    if (result == 0 || result == std::numeric_limits<uintptr_t>::max()) {
      throw std::overflow_error("vulkan tensor: context identity space exhausted");
    }
    if (next.compare_exchange_weak(result, result + 1,
                                   std::memory_order_relaxed)) {
      return result;
    }
  }
}

bool known_exact_cooperative_gemm_device(const DeviceInfo& info) {
  static constexpr uint8_t kDriverUuid[16] = {
      0x86, 0x90, 0xf1, 0xc8, 0x0a, 0x3f, 0x54, 0x99,
      0x9b, 0xf6, 0xea, 0x2a, 0xee, 0x51, 0x56, 0x02};
  return info.vendor_id == 0x10de && info.device_id == 0x2b85 &&
      info.driver_version == 0x98960000 && info.subgroup_size == 32 &&
      std::memcmp(info.driver_uuid, kDriverUuid, sizeof(kDriverUuid)) == 0 &&
      info.cooperative_matrix_enabled && info.storage_buffer_16bit_enabled &&
      info.shader_bfloat16_type && info.shader_bfloat16_cooperative_matrix &&
      info.cooperative_matrix_bf16_f32_16x16x16;
}

bool known_exact_cooperative_f16_gemm_device(const DeviceInfo& info) {
  return known_exact_cooperative_gemm_device(info) &&
      info.shader_float16_enabled &&
      info.cooperative_matrix_f16_f32_16x16x16;
}

}  // namespace

struct DeviceTensor::Impl {
  Buffer buffer;
  TensorLayout layout;
  ScalarType type = ScalarType::kFloat32;
  uint64_t logical_bytes = 0;
  uintptr_t context = 0;
  bool has_access = false;
  BufferAccess access = BufferAccess::kTransferWrite;
};

struct LinearWeight::Impl {
  LinearWeightFormat format = LinearWeightFormat::kBFloat16;
  uint32_t out_features = 0;
  uint32_t in_features = 0;
  uint64_t stored_bytes = 0;
  uint64_t resident_bytes = 0;
  bool has_fp8_input_scale = false;
  float fp8_input_scale = 0.0f;
  bool full_precision_matrix_mult = false;
  float global_scale = 1.0f;
  float nf4_nested_offset = 0.0f;
  uint32_t nf4_block_size = 64;
  uint32_t nf4_nested_block_size = 256;
  bool convrot = false;
  uint32_t convrot_group = 256;
  DeviceTensor data;
  DeviceTensor weight_scale;
  DeviceTensor block_scale;
  DeviceTensor nf4_absmax;
  DeviceTensor nf4_quant_map;
  DeviceTensor nf4_nested_quant_map;
  DeviceTensor nf4_nested_absmax;
  DeviceTensor pre_quant_scale;
};

struct DenseGemmPlan::Impl {
  std::shared_ptr<TensorContext::Impl> owner;
  DenseGemmPlanDesc desc;
};

struct PreparedF16Activation::Impl {
  std::shared_ptr<TensorContext::Impl> owner;
  DeviceTensor tensor;
  uint32_t max_rows = 0;
  uint32_t in_features = 0;
  uint64_t generation = 0;
};

struct PreparedF16ActivationView::Impl {
  std::shared_ptr<PreparedF16Activation::Impl> slot;
  const void* batch_identity = nullptr;
  uint32_t prepared_rows = 0;
  uint64_t generation = 0;
};

struct TensorWorkspace::Impl {
  BufferPool pool;
  Buffer buffer;
  uint64_t capacity = 0;
  uint64_t cursor = 0;
  uint64_t generation = 1;
  uintptr_t context = next_context_identity();

  Impl(const Device& input, uint64_t block_bytes)
      : pool(input, block_bytes) {}
};

struct TensorContext::Impl {
  struct Parameters {
    uint32_t op = 0;
    uint32_t count = 0;
    uint32_t p[6] = {};
  };
  struct NormParameters {
    uint32_t rows = 0;
    uint32_t dim = 0;
    uint32_t epsilon_bits = 0;
    uint32_t mod_rows = 0;
  };
  struct VaeRopeParameters {
    uint32_t sequence = 0;
    uint32_t heads = 0;
    uint32_t head_dim = 64;
    uint32_t rope_dim = 48;
    uint32_t num_patches = 0;
    uint32_t epsilon_bits = 0;
    uint32_t unused[2] = {};
  };
  struct WeightParameters {
    uint32_t op = 0;
    uint32_t count = 0;
    uint32_t out_features = 0;
    uint32_t in_features = 0;
    uint32_t block_size = 0;
    uint32_t nested_block_size = 0;
    uint32_t scalar_bits = 0;
    uint32_t group = 0;
  };
  struct GemmParameters {
    uint32_t rows = 0;
    uint32_t out_features = 0;
    uint32_t in_features = 0;
    uint32_t input_row_offset = 0;
    uint32_t output_row_offset = 0;
    uint32_t mode = 0;
    uint32_t unused[2] = {};
  };
  struct GemmPrepareParameters {
    uint32_t rows = 0;
    uint32_t in_features = 0;
    uint32_t input_row_offset = 0;
    uint32_t groups_x = 0;
  };
  static constexpr uint32_t kMaxBatchOperators = 32;

  ComputeContext commands;
  BufferPool pool;
  TensorWorkspace scratch;
  ComputePipeline ops_pipeline;
  ComputePipeline rope_pipeline;
  ComputePipeline vae_rope_pipeline;
  ComputePipeline weight_pipeline;
  ComputePipeline gemm_pipeline;
  ComputePipeline gemm_coop_pipeline;
  ComputePipeline gemm_prepare_pipeline;
  ComputePipeline gemm_coop_f16_pipeline;
  ComputePipeline rms_norm_pipeline;
  ComputePipeline layer_norm_pipeline;
  ComputePipeline bf16_rms_block_pipeline;
  ComputePipeline bf16_rms_narrow_pipeline;
  ComputePipeline bf16_layer_pipeline;
  ComputePipeline bf16_mod_pipeline;
  ComputePipeline fp32_mod_pipeline;
  ComputePipeline group_norm_pipeline;
  Buffer upload_buffer;
  Buffer readback_buffer;
  uint64_t staging_capacity = 0;
  std::vector<StorageBinding> ops_bindings;
  std::vector<StorageBinding> norm_bindings;
  std::vector<StorageBinding> mod_bindings;
  std::vector<StorageBinding> vae_rope_bindings;
  std::vector<StorageBinding> weight_bindings;
  std::vector<StorageBinding> gemm_bindings;
  std::vector<StorageBinding> gemm_prepare_bindings;
  bool full_arithmetic_exact = false;
  bool exact_vae_norm = false;
  bool cooperative_gemm = false;
  bool cooperative_f16_gemm = false;
  uint32_t max_dispatch_x = 0;
  uint32_t max_dispatch_y = 0;
  uint64_t max_storage_bytes = 0;
  std::atomic<bool> recorder_active{false};

  explicit Impl(const Device& input, const TensorContextOptions& tensor_options)
      : commands(input, [&] {
          ComputeContextOptions options;
          options.max_in_flight = tensor_options.max_in_flight;
          options.max_storage_bindings = 7;
          options.max_compute_binds_per_job = kMaxBatchOperators * 2;
          return options;
        }()),
        pool(input, 4ull << 20),
        scratch(input),
        ops_bindings(3),
        norm_bindings(4),
        mod_bindings(6),
        vae_rope_bindings(7),
        weight_bindings(6),
        gemm_bindings(4),
        gemm_prepare_bindings(2) {
    // Device is move-only; the opaque handle is sufficient for identity and
    // every owned Vulkan object already retains the shared device state.
    static_assert(sizeof(detail::kTensorOpsSpirv) % sizeof(uint32_t) == 0);
    if (!input.info().fp32_signed_zero_inf_nan_preserve ||
        !input.info().fp32_rounding_rte) {
      throw std::runtime_error(
          "vulkan tensor: fp32 arithmetic requires signed-zero/Inf/NaN preservation "
          "and round-to-nearest-even");
    }
    full_arithmetic_exact = input.info().fp32_denorm_preserve;
    exact_vae_norm = detail::known_exact_vae_norm_device(
                         input.info().vendor_id, input.info().device_id,
                         input.info().driver_version) &&
                     input.info().fp32_signed_zero_inf_nan_preserve &&
                     input.info().shader_int64_enabled;
    max_dispatch_x = input.info().max_compute_workgroup_count[0];
    max_dispatch_y = input.info().max_compute_workgroup_count[1];
    max_storage_bytes = input.info().max_storage_buffer_bytes;
    const uint8_t* shader = full_arithmetic_exact ? detail::kTensorOpsDenormSpirv
                                           : detail::kTensorOpsSpirv;
    const size_t shader_bytes = full_arithmetic_exact ? sizeof(detail::kTensorOpsDenormSpirv)
                                               : sizeof(detail::kTensorOpsSpirv);
    std::vector<uint32_t> spirv(shader_bytes / sizeof(uint32_t));
    std::memcpy(spirv.data(), shader, shader_bytes);
    ComputePipelineOptions options;
    options.storage_binding_count = 3;
    options.push_constant_bytes = sizeof(Parameters);
    options.local_size[0] = 64;
    ops_pipeline = ComputePipeline::create(input, spirv, options);
    const uint8_t* rope_shader = full_arithmetic_exact
        ? detail::kTensorRopeDenormSpirv : detail::kTensorRopeSpirv;
    const size_t rope_shader_bytes = full_arithmetic_exact
        ? sizeof(detail::kTensorRopeDenormSpirv) : sizeof(detail::kTensorRopeSpirv);
    std::vector<uint32_t> rope_spirv(rope_shader_bytes / sizeof(uint32_t));
    std::memcpy(rope_spirv.data(), rope_shader, rope_shader_bytes);
    ComputePipelineOptions rope_options = options;
    rope_options.local_size[0] = 64;
    rope_pipeline = ComputePipeline::create(input, rope_spirv, rope_options);
    const uint8_t* weight_shader = full_arithmetic_exact
        ? detail::kTensorWeightDenormSpirv : detail::kTensorWeightSpirv;
    const size_t weight_shader_bytes = full_arithmetic_exact
        ? sizeof(detail::kTensorWeightDenormSpirv)
        : sizeof(detail::kTensorWeightSpirv);
    std::vector<uint32_t> weight_spirv(weight_shader_bytes / sizeof(uint32_t));
    std::memcpy(weight_spirv.data(), weight_shader, weight_shader_bytes);
    ComputePipelineOptions weight_options;
    weight_options.storage_binding_count = 6;
    weight_options.push_constant_bytes = sizeof(WeightParameters);
    weight_options.local_size[0] = 64;
    weight_pipeline = ComputePipeline::create(input, weight_spirv, weight_options);
    const uint8_t* gemm_shader = full_arithmetic_exact
        ? detail::kTensorGemmDenormSpirv : detail::kTensorGemmSpirv;
    const size_t gemm_shader_bytes = full_arithmetic_exact
        ? sizeof(detail::kTensorGemmDenormSpirv) : sizeof(detail::kTensorGemmSpirv);
    std::vector<uint32_t> gemm_spirv(gemm_shader_bytes / sizeof(uint32_t));
    std::memcpy(gemm_spirv.data(), gemm_shader, gemm_shader_bytes);
    ComputePipelineOptions gemm_options;
    gemm_options.storage_binding_count = 4;
    gemm_options.push_constant_bytes = sizeof(GemmParameters);
    gemm_options.local_size[0] = 8;
    gemm_options.local_size[1] = 8;
    gemm_pipeline = ComputePipeline::create(input, gemm_spirv, gemm_options);
    cooperative_gemm = known_exact_cooperative_gemm_device(input.info());
    if (cooperative_gemm) {
      const uint8_t* cooperative_shader = full_arithmetic_exact
          ? detail::kTensorGemmCoopDenormSpirv : detail::kTensorGemmCoopSpirv;
      const size_t cooperative_bytes = full_arithmetic_exact
          ? sizeof(detail::kTensorGemmCoopDenormSpirv)
          : sizeof(detail::kTensorGemmCoopSpirv);
      std::vector<uint32_t> cooperative_spirv(cooperative_bytes / sizeof(uint32_t));
      std::memcpy(cooperative_spirv.data(), cooperative_shader, cooperative_bytes);
      ComputePipelineOptions cooperative_options = gemm_options;
      cooperative_options.local_size[0] = 128;
      cooperative_options.local_size[1] = 1;
      gemm_coop_pipeline = ComputePipeline::create(
          input, cooperative_spirv, cooperative_options);
    }
    {
      std::vector<uint32_t> prepare_spirv(
          sizeof(detail::kTensorGemmPrepareSpirv) / sizeof(uint32_t));
      std::memcpy(prepare_spirv.data(), detail::kTensorGemmPrepareSpirv,
                  sizeof(detail::kTensorGemmPrepareSpirv));
      ComputePipelineOptions prepare_options;
      prepare_options.storage_binding_count = 2;
      prepare_options.push_constant_bytes = sizeof(GemmPrepareParameters);
      prepare_options.local_size[0] = 64;
      gemm_prepare_pipeline = ComputePipeline::create(
          input, prepare_spirv, prepare_options);
    }
    cooperative_f16_gemm = known_exact_cooperative_f16_gemm_device(input.info());
    if (cooperative_f16_gemm) {
      const uint8_t* cooperative_shader = full_arithmetic_exact
          ? detail::kTensorGemmCoopF16DenormSpirv
          : detail::kTensorGemmCoopF16Spirv;
      const size_t cooperative_bytes = full_arithmetic_exact
          ? sizeof(detail::kTensorGemmCoopF16DenormSpirv)
          : sizeof(detail::kTensorGemmCoopF16Spirv);
      std::vector<uint32_t> cooperative_spirv(cooperative_bytes / sizeof(uint32_t));
      std::memcpy(cooperative_spirv.data(), cooperative_shader, cooperative_bytes);
      ComputePipelineOptions cooperative_options = gemm_options;
      cooperative_options.local_size[0] = 128;
      cooperative_options.local_size[1] = 1;
      gemm_coop_f16_pipeline = ComputePipeline::create(
          input, cooperative_spirv, cooperative_options);
    }
    ComputePipelineOptions norm_options;
    norm_options.storage_binding_count = 4;
    norm_options.push_constant_bytes = sizeof(NormParameters);
    norm_options.local_size[0] = 256;
    auto make_norm_pipeline = [&](const uint8_t* shader, size_t shader_bytes,
                                  uint32_t bindings = 4, uint32_t local_x = 256,
                                  uint32_t local_y = 1,
                                  uint32_t push_bytes = sizeof(NormParameters)) {
      std::vector<uint32_t> module(shader_bytes / sizeof(uint32_t));
      std::memcpy(module.data(), shader, shader_bytes);
      ComputePipelineOptions selected = norm_options;
      selected.storage_binding_count = bindings;
      selected.local_size[0] = local_x;
      selected.local_size[1] = local_y;
      selected.push_constant_bytes = push_bytes;
      return ComputePipeline::create(input, module, selected);
    };
    if (exact_vae_norm) {
      rms_norm_pipeline = make_norm_pipeline(detail::kTensorRmsNormSpirv,
                                             sizeof(detail::kTensorRmsNormSpirv));
      layer_norm_pipeline = make_norm_pipeline(detail::kTensorLayerNormSpirv,
                                               sizeof(detail::kTensorLayerNormSpirv));
      bf16_rms_block_pipeline = make_norm_pipeline(
          detail::kTensorBf16RmsBlockSpirv,
          sizeof(detail::kTensorBf16RmsBlockSpirv), 3);
      bf16_rms_narrow_pipeline = make_norm_pipeline(
          detail::kTensorBf16RmsNarrowSpirv,
          sizeof(detail::kTensorBf16RmsNarrowSpirv), 3, 32, 8);
      bf16_layer_pipeline = make_norm_pipeline(
          detail::kTensorBf16LayerSpirv, sizeof(detail::kTensorBf16LayerSpirv));
      bf16_mod_pipeline = make_norm_pipeline(
          detail::kTensorBf16ModSpirv, sizeof(detail::kTensorBf16ModSpirv), 6);
      fp32_mod_pipeline = make_norm_pipeline(
          detail::kTensorFp32ModSpirv, sizeof(detail::kTensorFp32ModSpirv), 6);
      group_norm_pipeline = make_norm_pipeline(
          detail::kTensorGroupNormSpirv, sizeof(detail::kTensorGroupNormSpirv));
      vae_rope_pipeline = make_norm_pipeline(
          detail::kTensorVaeRopeSpirv, sizeof(detail::kTensorVaeRopeSpirv), 7,
          32, 1, sizeof(VaeRopeParameters));
    }
    for (uint32_t i = 0; i < ops_bindings.size(); ++i) ops_bindings[i].binding = i;
    for (uint32_t i = 0; i < norm_bindings.size(); ++i) norm_bindings[i].binding = i;
    for (uint32_t i = 0; i < mod_bindings.size(); ++i) mod_bindings[i].binding = i;
    for (uint32_t i = 0; i < vae_rope_bindings.size(); ++i)
      vae_rope_bindings[i].binding = i;
    for (uint32_t i = 0; i < weight_bindings.size(); ++i)
      weight_bindings[i].binding = i;
    for (uint32_t i = 0; i < gemm_bindings.size(); ++i)
      gemm_bindings[i].binding = i;
    for (uint32_t i = 0; i < gemm_prepare_bindings.size(); ++i)
      gemm_prepare_bindings[i].binding = i;
  }

  uintptr_t context_id = next_context_identity();

  struct RecorderLease {
    Impl* owner = nullptr;
    RecorderLease() = default;
    explicit RecorderLease(Impl* value) : owner(value) {}
    RecorderLease(RecorderLease&& other) noexcept
        : owner(std::exchange(other.owner, nullptr)) {}
    RecorderLease& operator=(RecorderLease&& other) noexcept {
      if (this != &other) {
        release();
        owner = std::exchange(other.owner, nullptr);
      }
      return *this;
    }
    RecorderLease(const RecorderLease&) = delete;
    RecorderLease& operator=(const RecorderLease&) = delete;
    ~RecorderLease() { release(); }
    void release() noexcept {
      if (!owner) return;
      owner->recorder_active.store(false, std::memory_order_release);
      owner = nullptr;
    }
  };

  RecorderLease acquire_recorder() {
    bool expected = false;
    if (!recorder_active.compare_exchange_strong(
            expected, true, std::memory_order_acquire,
            std::memory_order_relaxed)) {
      throw std::logic_error(
          "vulkan tensor: another batch or boundary operation is active");
    }
    return RecorderLease(this);
  }

  void ensure_staging(uint64_t bytes) {
    if (bytes <= staging_capacity) return;
    Buffer new_upload = pool.allocate(bytes, BufferUsage::kTransferSource, MemoryUsage::kUpload);
    Buffer new_readback =
        pool.allocate(bytes, BufferUsage::kTransferDestination, MemoryUsage::kReadback);
    upload_buffer = std::move(new_upload);
    readback_buffer = std::move(new_readback);
    staging_capacity = bytes;
    pool.trim();
  }

  std::shared_ptr<DeviceTensor::Impl> require(DeviceTensor& tensor) const {
    if (!tensor.impl_ || tensor.impl_->context != context_id) {
      throw std::invalid_argument("vulkan tensor: tensor belongs to another context");
    }
    return tensor.impl_;
  }

  void complete(CommandList&& list) {
    Submission done = commands.submit(std::move(list));
    done.wait();
    commands.collect();
  }

  uint32_t validate_dispatch(uint64_t count) const {
    if (count == 0 || count > std::numeric_limits<uint32_t>::max()) {
      throw std::out_of_range("vulkan tensor: operation exceeds uint32 indexing limits");
    }
    const uint64_t groups = (count + 63ull) / 64ull;
    if (groups > max_dispatch_x) {
      throw std::out_of_range("vulkan tensor: operation exceeds dispatch limits");
    }
    return static_cast<uint32_t>(count);
  }
};

struct TensorBatch::Impl {
  struct AccessSnapshot {
    std::shared_ptr<DeviceTensor::Impl> tensor;
    bool has_access = false;
    BufferAccess access = BufferAccess::kTransferWrite;
  };

  std::shared_ptr<TensorContext::Impl> owner;
  CommandList commands;
  TensorContext::Impl::RecorderLease recording_lease;
  std::array<AccessSnapshot, TensorContext::Impl::kMaxBatchOperators * 7> snapshots{};
  uint32_t snapshot_count = 0;
  uint32_t operator_count = 0;
  bool submitted = false;
  bool poisoned = false;

  void transition(const std::shared_ptr<DeviceTensor::Impl>& tensor, BufferAccess next) {
    bool seen = false;
    for (uint32_t i = 0; i < snapshot_count; ++i) {
      if (snapshots[i].tensor.get() == tensor.get()) {
        seen = true;
        break;
      }
    }
    if (!seen) {
      if (snapshot_count == snapshots.size()) {
        throw std::logic_error("vulkan tensor: batch transition capacity exceeded");
      }
      snapshots[snapshot_count++] = {tensor, tensor->has_access, tensor->access};
    }
    const bool next_writes = next == BufferAccess::kTransferWrite ||
                             next == BufferAccess::kComputeWrite ||
                             next == BufferAccess::kComputeReadWrite;
    if (tensor->has_access && (tensor->access != next || next_writes)) {
      commands.barrier(tensor->buffer, tensor->access, next);
    }
    tensor->has_access = true;
    tensor->access = next;
  }

  void count_operator() {
    if (operator_count == TensorContext::Impl::kMaxBatchOperators) {
      throw std::logic_error("vulkan tensor: batch operator limit exceeded");
    }
    ++operator_count;
  }

  void dispatch(const TensorContext::Impl::Parameters& parameters,
                const std::shared_ptr<DeviceTensor::Impl>& a,
                const std::shared_ptr<DeviceTensor::Impl>& b,
                const std::shared_ptr<DeviceTensor::Impl>& output) {
    // Every caller validates this before it mutates access tracking. Widen the
    // addition anyway so the invariant remains safe if uint32 max is used.
    const uint32_t groups =
        static_cast<uint32_t>((static_cast<uint64_t>(parameters.count) + 63ull) / 64ull);
    owner->ops_bindings[0].buffer = &a->buffer;
    owner->ops_bindings[0].bytes = a->buffer.size();
    owner->ops_bindings[1].buffer = &b->buffer;
    owner->ops_bindings[1].bytes = b->buffer.size();
    owner->ops_bindings[2].buffer = &output->buffer;
    owner->ops_bindings[2].bytes = output->buffer.size();
    commands.bind_compute(owner->ops_pipeline, owner->ops_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups);
  }

  void dispatch_rope(const TensorContext::Impl::Parameters& parameters,
                     uint32_t groups,
                     const std::shared_ptr<DeviceTensor::Impl>& input,
                     const std::shared_ptr<DeviceTensor::Impl>& cosine,
                     const std::shared_ptr<DeviceTensor::Impl>& sine) {
    owner->ops_bindings[0].buffer = &input->buffer;
    owner->ops_bindings[0].bytes = input->buffer.size();
    owner->ops_bindings[1].buffer = &cosine->buffer;
    owner->ops_bindings[1].bytes = cosine->buffer.size();
    owner->ops_bindings[2].buffer = &sine->buffer;
    owner->ops_bindings[2].bytes = sine->buffer.size();
    commands.bind_compute(owner->rope_pipeline, owner->ops_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups);
  }

  void dispatch_norm(ComputePipeline& pipeline,
                     const TensorContext::Impl::NormParameters& parameters,
                     const std::shared_ptr<DeviceTensor::Impl>& input,
                     const std::shared_ptr<DeviceTensor::Impl>& weight,
                     const std::shared_ptr<DeviceTensor::Impl>& bias,
                     const std::shared_ptr<DeviceTensor::Impl>& output) {
    owner->norm_bindings[0].buffer = &input->buffer;
    owner->norm_bindings[0].bytes = input->buffer.size();
    owner->norm_bindings[1].buffer = &weight->buffer;
    owner->norm_bindings[1].bytes = weight->buffer.size();
    owner->norm_bindings[2].buffer = &bias->buffer;
    owner->norm_bindings[2].bytes = bias->buffer.size();
    owner->norm_bindings[3].buffer = &output->buffer;
    owner->norm_bindings[3].bytes = output->buffer.size();
    commands.bind_compute(pipeline, owner->norm_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(parameters.rows);
  }

  void dispatch_shared_rms(ComputePipeline& pipeline,
                           const TensorContext::Impl::NormParameters& parameters,
                           uint32_t groups,
                           const std::shared_ptr<DeviceTensor::Impl>& input,
                           const std::shared_ptr<DeviceTensor::Impl>& weight,
                           const std::shared_ptr<DeviceTensor::Impl>& output) {
    owner->ops_bindings[0].buffer = &input->buffer;
    owner->ops_bindings[0].bytes = input->buffer.size();
    owner->ops_bindings[1].buffer = &weight->buffer;
    owner->ops_bindings[1].bytes = weight->buffer.size();
    owner->ops_bindings[2].buffer = &output->buffer;
    owner->ops_bindings[2].bytes = output->buffer.size();
    commands.bind_compute(pipeline, owner->ops_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups);
  }

  void dispatch_shared_layer(ComputePipeline& pipeline,
                             const TensorContext::Impl::NormParameters& parameters,
                             const std::shared_ptr<DeviceTensor::Impl>& input,
                             const std::shared_ptr<DeviceTensor::Impl>& weight,
                             const std::shared_ptr<DeviceTensor::Impl>& bias,
                             const std::shared_ptr<DeviceTensor::Impl>& output) {
    owner->norm_bindings[0].buffer = &input->buffer;
    owner->norm_bindings[0].bytes = input->buffer.size();
    owner->norm_bindings[1].buffer = &weight->buffer;
    owner->norm_bindings[1].bytes = weight->buffer.size();
    owner->norm_bindings[2].buffer = &bias->buffer;
    owner->norm_bindings[2].bytes = bias->buffer.size();
    owner->norm_bindings[3].buffer = &output->buffer;
    owner->norm_bindings[3].bytes = output->buffer.size();
    commands.bind_compute(pipeline, owner->norm_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(parameters.rows);
  }

  void dispatch_group_norm(const TensorContext::Impl::NormParameters& parameters,
                           const std::shared_ptr<DeviceTensor::Impl>& input,
                           const std::shared_ptr<DeviceTensor::Impl>& weight,
                           const std::shared_ptr<DeviceTensor::Impl>& bias,
                           const std::shared_ptr<DeviceTensor::Impl>& output) {
    owner->norm_bindings[0].buffer = &input->buffer;
    owner->norm_bindings[0].bytes = input->buffer.size();
    owner->norm_bindings[1].buffer = &weight->buffer;
    owner->norm_bindings[1].bytes = weight->buffer.size();
    owner->norm_bindings[2].buffer = &bias->buffer;
    owner->norm_bindings[2].bytes = bias->buffer.size();
    owner->norm_bindings[3].buffer = &output->buffer;
    owner->norm_bindings[3].bytes = output->buffer.size();
    commands.bind_compute(owner->group_norm_pipeline, owner->norm_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(parameters.mod_rows);
  }

  void dispatch_shared_mod(ComputePipeline& pipeline,
                           const TensorContext::Impl::NormParameters& parameters,
                           const std::array<std::shared_ptr<DeviceTensor::Impl>, 6>& resources) {
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->mod_bindings[i].buffer = &resources[i]->buffer;
      owner->mod_bindings[i].bytes = resources[i]->buffer.size();
    }
    commands.bind_compute(pipeline, owner->mod_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(parameters.rows);
  }

  void dispatch_vae_rope(
      const TensorContext::Impl::VaeRopeParameters& parameters,
      uint32_t groups,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 7>& resources) {
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->vae_rope_bindings[i].buffer = &resources[i]->buffer;
      owner->vae_rope_bindings[i].bytes = resources[i]->buffer.size();
    }
    commands.bind_compute(owner->vae_rope_pipeline, owner->vae_rope_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups);
  }

  void dispatch_weight(
      const TensorContext::Impl::WeightParameters& parameters,
      uint32_t groups,
      const std::array<std::shared_ptr<DeviceTensor::Impl>, 6>& resources) {
    for (size_t i = 0; i < resources.size(); ++i) {
      owner->weight_bindings[i].buffer = &resources[i]->buffer;
      owner->weight_bindings[i].bytes = resources[i]->buffer.size();
    }
    commands.bind_compute(owner->weight_pipeline, owner->weight_bindings);
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch(groups);
  }

  void record_shared_mod(bool fp32, DeviceTensor& input, DeviceTensor& weight,
                         DeviceTensor& scale, DeviceTensor& shift,
                         DeviceTensor& selectors, DeviceTensor& output,
                         float epsilon) {
    if (!owner->exact_vae_norm) {
      throw std::runtime_error("vulkan tensor: exact shared RMSNorm modulation is unavailable");
    }
    auto src = owner->require(input);
    auto w = owner->require(weight);
    auto sc = owner->require(scale);
    auto sh = owner->require(shift);
    auto index = owner->require(selectors);
    auto dst = owner->require(output);
    const auto& shape = src->layout;
    const uint64_t rows = shape.extent[0];
    const uint64_t dim = shape.extent[1];
    const uint64_t mod_rows = sc->layout.extent[0];
    const bool aliases_parameters =
        src.get() == w.get() || src.get() == sc.get() || src.get() == sh.get() ||
        src.get() == index.get() || w.get() == sc.get() || w.get() == sh.get() ||
        w.get() == index.get() || sc.get() == sh.get() || sc.get() == index.get() ||
        sh.get() == index.get() || dst.get() == w.get() || dst.get() == sc.get() ||
        dst.get() == sh.get() || dst.get() == index.get();
    if (!std::isnormal(epsilon) || epsilon <= 0.0f || aliases_parameters ||
        shape.rank != 2 || w->layout.rank != 1 || sc->layout.rank != 2 ||
        sh->layout.rank != 2 || index->layout.rank != 1 || dst->layout.rank != 2 ||
        w->layout.extent[0] != dim || sc->layout.extent[1] != dim ||
        sh->layout.extent != sc->layout.extent || index->layout.extent[0] != rows ||
        dst->layout.extent != shape.extent || w->type != ScalarType::kBFloat16 ||
        sc->type != ScalarType::kFloat32 || sh->type != ScalarType::kFloat32 ||
        index->type != ScalarType::kInt32 ||
        src->type != (fp32 ? ScalarType::kFloat32 : ScalarType::kBFloat16) ||
        dst->type != src->type || !shape.is_contiguous() ||
        !w->layout.is_contiguous() || !sc->layout.is_contiguous() ||
        !sh->layout.is_contiguous() || !index->layout.is_contiguous() ||
        !dst->layout.is_contiguous() || rows > std::numeric_limits<uint32_t>::max() ||
        dim > kMaxExactNormDimension ||
        mod_rows > std::numeric_limits<uint32_t>::max() ||
        shape.elements() > std::numeric_limits<uint32_t>::max() ||
        sc->layout.elements() > std::numeric_limits<uint32_t>::max()) {
      throw std::invalid_argument("vulkan tensor: invalid shared RMSNorm modulation");
    }
    if (!detail::norm_dispatch_fits(rows, owner->max_dispatch_x)) {
      throw std::out_of_range("vulkan tensor: modulated RMSNorm rows exceed dispatch limits");
    }
    TensorContext::Impl::NormParameters p;
    p.rows = static_cast<uint32_t>(rows);
    p.dim = static_cast<uint32_t>(dim);
    p.mod_rows = static_cast<uint32_t>(mod_rows);
    std::memcpy(&p.epsilon_bits, &epsilon, sizeof(epsilon));
    try {
      count_operator();
      transition(src, src.get() == dst.get() ? BufferAccess::kComputeReadWrite
                                              : BufferAccess::kComputeRead);
      transition(w, BufferAccess::kComputeRead);
      transition(sc, BufferAccess::kComputeRead);
      transition(sh, BufferAccess::kComputeRead);
      transition(index, BufferAccess::kComputeRead);
      if (src.get() != dst.get()) transition(dst, BufferAccess::kComputeWrite);
      std::array<std::shared_ptr<DeviceTensor::Impl>, 6> resources{
          src, w, sc, sh, index, dst};
      dispatch_shared_mod(fp32 ? owner->fp32_mod_pipeline : owner->bf16_mod_pipeline,
                          p, resources);
    } catch (...) {
      poisoned = true;
      throw;
    }
  }

  ~Impl() {
    if (submitted) return;
    // Discard Vulkan recording before making its speculative access state
    // invisible to the next batch.
    commands = CommandList{};
    for (uint32_t i = snapshot_count; i-- > 0;) {
      snapshots[i].tensor->has_access = snapshots[i].has_access;
      snapshots[i].tensor->access = snapshots[i].access;
    }
  }
};

DeviceTensor::DeviceTensor() = default;
DeviceTensor::~DeviceTensor() = default;
DeviceTensor::DeviceTensor(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
DeviceTensor::DeviceTensor(DeviceTensor&&) noexcept = default;
DeviceTensor& DeviceTensor::operator=(DeviceTensor&&) noexcept = default;
DeviceTensorView DeviceTensor::view() const {
  if (!impl_) throw std::logic_error("vulkan tensor: empty tensor");
  DeviceTensorView result;
  result.backend = DeviceBackend::kVulkan;
  result.type = impl_->type;
  result.layout = impl_->layout;
  result.context = impl_->context;
  result.resource = impl_->buffer.native_handle();
  result.byte_size = impl_->logical_bytes;
  return result;
}
const TensorLayout& DeviceTensor::layout() const {
  if (!impl_) throw std::logic_error("vulkan tensor: empty tensor");
  return impl_->layout;
}
ScalarType DeviceTensor::type() const {
  if (!impl_) throw std::logic_error("vulkan tensor: empty tensor");
  return impl_->type;
}
DeviceTensor::operator bool() const noexcept { return impl_ != nullptr; }

TensorWorkspace::TensorWorkspace(const Device& device, uint64_t block_bytes)
    : impl_(std::make_unique<Impl>(device, block_bytes)) {}
TensorWorkspace::~TensorWorkspace() = default;
TensorWorkspace::TensorWorkspace(TensorWorkspace&&) noexcept = default;
TensorWorkspace& TensorWorkspace::operator=(TensorWorkspace&&) noexcept = default;
DeviceBackend TensorWorkspace::backend() const noexcept { return DeviceBackend::kVulkan; }
void TensorWorkspace::reserve(uint64_t bytes) {
  if (!impl_) throw std::logic_error("vulkan workspace: moved-from object");
  if (bytes <= impl_->capacity) return;
  if (impl_->cursor != 0) throw std::logic_error("vulkan workspace: reset before growth");
  Buffer replacement = impl_->pool.allocate(
      bytes, BufferUsage::kStorage | BufferUsage::kTransferSource |
                 BufferUsage::kTransferDestination,
      MemoryUsage::kDevice);
  impl_->buffer = std::move(replacement);
  impl_->capacity = bytes;
  ++impl_->generation;
  impl_->pool.trim();
}
WorkspaceSpan TensorWorkspace::allocate(uint64_t bytes, uint64_t alignment) {
  if (!impl_) throw std::logic_error("vulkan workspace: moved-from object");
  if (bytes == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) {
    throw std::invalid_argument("vulkan workspace: size and power-of-two alignment required");
  }
  if (impl_->cursor > std::numeric_limits<uint64_t>::max() - (alignment - 1)) {
    throw std::overflow_error("vulkan workspace: offset overflow");
  }
  const uint64_t offset = (impl_->cursor + alignment - 1) & ~(alignment - 1);
  if (offset > impl_->capacity || bytes > impl_->capacity - offset) {
    throw std::out_of_range("vulkan workspace: reserved capacity exceeded");
  }
  impl_->cursor = offset + bytes;
  return {DeviceBackend::kVulkan, impl_->context, impl_->buffer.native_handle(), offset, bytes,
          impl_->generation};
}
void TensorWorkspace::reset() noexcept { if (impl_) impl_->cursor = 0; }
uint64_t TensorWorkspace::capacity() const noexcept { return impl_ ? impl_->capacity : 0; }
uint64_t TensorWorkspace::used() const noexcept { return impl_ ? impl_->cursor : 0; }
uint64_t TensorWorkspace::generation() const noexcept { return impl_ ? impl_->generation : 0; }
bool TensorWorkspace::valid(const WorkspaceSpan& span) const noexcept {
  return impl_ && span.backend == DeviceBackend::kVulkan && span.context == impl_->context &&
         span.resource == impl_->buffer.native_handle() &&
         span.generation == impl_->generation && span.byte_size != 0 &&
         span.byte_offset <= impl_->capacity &&
         span.byte_size <= impl_->capacity - span.byte_offset;
}
uint64_t TensorWorkspace::reserved_bytes() const noexcept {
  return impl_ ? impl_->pool.reserved_bytes() : 0;
}
uint64_t TensorWorkspace::pooled_used_bytes() const noexcept {
  return impl_ ? impl_->pool.used_bytes() : 0;
}

LinearWeight::LinearWeight() = default;
LinearWeight::~LinearWeight() = default;
LinearWeight::LinearWeight(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
LinearWeight::LinearWeight(LinearWeight&&) noexcept = default;
LinearWeight& LinearWeight::operator=(LinearWeight&&) noexcept = default;

LinearWeight LinearWeight::upload(TensorContext& context,
                                  const LinearWeightUpload& source) {
  if (!context.impl_) {
    throw std::logic_error("vulkan linear weight: moved-from tensor context");
  }
  if (source.out_features == 0 || source.in_features == 0 || source.data == nullptr) {
    throw std::invalid_argument("vulkan linear weight: nonzero shape and data required");
  }
  const uint64_t elements = checked_multiply(source.out_features,
                                              source.in_features, "linear weight");
  uint64_t expected_bytes = 0;
  ScalarType data_type = ScalarType::kUInt8;
  switch (source.format) {
    case LinearWeightFormat::kFloat32:
      expected_bytes = checked_multiply(elements, 4, "linear weight");
      data_type = ScalarType::kFloat32;
      break;
    case LinearWeightFormat::kFloat16:
      expected_bytes = checked_multiply(elements, 2, "linear weight");
      data_type = ScalarType::kFloat16;
      break;
    case LinearWeightFormat::kBFloat16:
      expected_bytes = checked_multiply(elements, 2, "linear weight");
      data_type = ScalarType::kBFloat16;
      break;
    case LinearWeightFormat::kFloat8E4M3:
    case LinearWeightFormat::kInt8:
      expected_bytes = elements;
      data_type = source.format == LinearWeightFormat::kInt8
          ? ScalarType::kInt8 : ScalarType::kUInt8;
      break;
    case LinearWeightFormat::kNVFloat4:
    case LinearWeightFormat::kNF4:
      expected_bytes = (elements + 1) / 2;
      break;
  }
  if (source.data_bytes != expected_bytes) {
    throw std::invalid_argument("vulkan linear weight: stored byte count mismatch");
  }
  const bool scale_pair_valid = (source.weight_scale == nullptr) ==
                                (source.weight_scale_count == 0);
  const bool block_pair_valid = (source.block_scale == nullptr) ==
                                (source.block_scale_count == 0);
  const bool absmax_pair_valid = (source.nf4_absmax == nullptr) ==
                                 (source.nf4_absmax_count == 0);
  const bool map_pair_valid = (source.nf4_quant_map == nullptr) ==
                              (source.nf4_quant_map_count == 0);
  const bool nested_map_pair_valid = (source.nf4_nested_quant_map == nullptr) ==
                                     (source.nf4_nested_quant_map_count == 0);
  const bool nested_absmax_pair_valid =
      (source.nf4_nested_absmax == nullptr) ==
      (source.nf4_nested_absmax_count == 0);
  const bool pre_scale_pair_valid =
      (source.pre_quant_scale_bf16 == nullptr) ==
      (source.pre_quant_scale_count == 0);
  if (!scale_pair_valid || !block_pair_valid || !absmax_pair_valid ||
      !map_pair_valid || !nested_map_pair_valid ||
      !nested_absmax_pair_valid || !pre_scale_pair_valid) {
    throw std::invalid_argument(
        "vulkan linear weight: auxiliary pointer/count mismatch");
  }
  if (source.has_fp8_input_scale &&
      (source.format != LinearWeightFormat::kFloat8E4M3 ||
       !std::isnormal(source.fp8_input_scale) || source.fp8_input_scale <= 0.0f)) {
    throw std::invalid_argument("vulkan linear weight: invalid FP8 input scale");
  }
  if (!source.has_fp8_input_scale && source.fp8_input_scale != 0.0f) {
    throw std::invalid_argument(
        "vulkan linear weight: absent FP8 input scale must retain zero value");
  }
  if (source.pre_quant_scale_count != 0 &&
      (source.pre_quant_scale_bf16 == nullptr ||
       source.pre_quant_scale_count != source.in_features)) {
    throw std::invalid_argument("vulkan linear weight: invalid AWQ pre-scale");
  }
  uint32_t power = 1;
  while (power < source.convrot_group && power <= UINT32_MAX / 4) power *= 4;
  if (source.convrot && (source.convrot_group < 4 || source.convrot_group > 256 ||
                         power != source.convrot_group ||
                         source.in_features % source.convrot_group != 0)) {
    throw std::invalid_argument("vulkan linear weight: invalid ConvRot group");
  }
  if (source.format == LinearWeightFormat::kFloat8E4M3 &&
      (source.weight_scale == nullptr || source.weight_scale_count != 1)) {
    throw std::invalid_argument("vulkan linear weight: FP8 scalar scale required");
  }
  if (source.format == LinearWeightFormat::kInt8 &&
      (source.weight_scale == nullptr ||
       source.weight_scale_count != source.out_features)) {
    throw std::invalid_argument("vulkan linear weight: INT8 per-row scales required");
  }
  if (source.format == LinearWeightFormat::kNVFloat4 &&
      (source.in_features % 64 != 0 || source.out_features % 128 != 0 ||
       source.block_scale == nullptr ||
       source.block_scale_count != elements / 16 ||
       !std::isfinite(source.global_scale))) {
    throw std::invalid_argument("vulkan linear weight: invalid NVFP4 metadata");
  }
  const uint64_t expected_absmax = source.nf4_block_size == 0
      ? 0 : 1 + (elements - 1) / source.nf4_block_size;
  const uint64_t expected_nested = source.nf4_nested_block_size == 0 ||
                                           expected_absmax == 0
      ? 0 : 1 + (expected_absmax - 1) / source.nf4_nested_block_size;
  if (source.format == LinearWeightFormat::kNF4 &&
      (source.nf4_block_size != 64 || source.nf4_nested_block_size != 256 ||
       source.nf4_absmax == nullptr || source.nf4_absmax_count != expected_absmax ||
       source.nf4_quant_map == nullptr || source.nf4_quant_map_count != 16 ||
       source.nf4_nested_quant_map == nullptr ||
       source.nf4_nested_quant_map_count != 256 ||
       source.nf4_nested_absmax == nullptr ||
       source.nf4_nested_absmax_count != expected_nested ||
       !std::isfinite(source.nf4_nested_offset))) {
    throw std::invalid_argument("vulkan linear weight: invalid NF4 metadata");
  }
  const bool has_nf4_aux = source.nf4_absmax_count != 0 ||
      source.nf4_quant_map_count != 0 ||
      source.nf4_nested_quant_map_count != 0 ||
      source.nf4_nested_absmax_count != 0;
  if ((source.format != LinearWeightFormat::kFloat8E4M3 &&
       source.format != LinearWeightFormat::kInt8 &&
       source.weight_scale_count != 0) ||
      (source.format != LinearWeightFormat::kNVFloat4 &&
       source.block_scale_count != 0) ||
      (source.format != LinearWeightFormat::kNF4 && has_nf4_aux)) {
    throw std::invalid_argument(
        "vulkan linear weight: auxiliary metadata does not match format");
  }

  auto result = std::make_unique<Impl>();
  result->format = source.format;
  result->out_features = source.out_features;
  result->in_features = source.in_features;
  result->stored_bytes = expected_bytes;
  result->has_fp8_input_scale = source.has_fp8_input_scale;
  result->fp8_input_scale = source.fp8_input_scale;
  result->full_precision_matrix_mult = source.full_precision_matrix_mult;
  result->global_scale = source.global_scale;
  result->nf4_nested_offset = source.nf4_nested_offset;
  result->nf4_block_size = source.nf4_block_size;
  result->nf4_nested_block_size = source.nf4_nested_block_size;
  result->convrot = source.convrot;
  result->convrot_group = source.convrot_group;
  struct PendingUpload {
    DeviceTensor* destination = nullptr;
    const void* values = nullptr;
    uint64_t logical_bytes = 0;
  };
  std::vector<PendingUpload> pending;
  pending.reserve(9);
  auto allocate = [&](DeviceTensor& destination, ScalarType type, uint64_t count,
                      const void* values) {
    TensorLayout layout = TensorLayout::contiguous(&count, 1);
    DeviceTensor replacement = context.allocate(layout, type);
    const uint64_t bytes = layout.bytes(type);
    if (result->resident_bytes > std::numeric_limits<uint64_t>::max() - bytes) {
      throw std::overflow_error("vulkan linear weight: resident byte count overflow");
    }
    result->resident_bytes += bytes;
    destination = std::move(replacement);
    pending.push_back({&destination, values, bytes});
  };
  const uint64_t data_count = data_type == ScalarType::kFloat32 ? elements
      : (data_type == ScalarType::kFloat16 || data_type == ScalarType::kBFloat16)
          ? elements : expected_bytes;
  allocate(result->data, data_type, data_count, source.data);
  if (source.weight_scale_count != 0)
    allocate(result->weight_scale, ScalarType::kFloat32,
             source.weight_scale_count, source.weight_scale);
  if (source.block_scale_count != 0)
    allocate(result->block_scale, ScalarType::kUInt8,
             source.block_scale_count, source.block_scale);
  if (source.format == LinearWeightFormat::kNF4) {
    allocate(result->nf4_absmax, ScalarType::kUInt8, source.nf4_absmax_count,
             source.nf4_absmax);
    allocate(result->nf4_quant_map, ScalarType::kFloat32,
             source.nf4_quant_map_count, source.nf4_quant_map);
    allocate(result->nf4_nested_quant_map, ScalarType::kFloat32,
             source.nf4_nested_quant_map_count, source.nf4_nested_quant_map);
    allocate(result->nf4_nested_absmax, ScalarType::kFloat32,
             source.nf4_nested_absmax_count, source.nf4_nested_absmax);
  }
  if (source.pre_quant_scale_count != 0)
    allocate(result->pre_quant_scale, ScalarType::kBFloat16,
             source.pre_quant_scale_count, source.pre_quant_scale_bf16);

  // All immutable components of one weight cross the queue in one job. Host-
  // visible staging is temporary and released immediately after the exact
  // submission completes; steady-state residency contains compressed bytes
  // and auxiliaries only.
  std::vector<Buffer> staging;
  staging.reserve(pending.size());
  for (const auto& item : pending) {
    const uint64_t physical = item.destination->impl_->buffer.size();
    Buffer buffer = context.impl_->pool.allocate(
        physical, BufferUsage::kTransferSource, MemoryUsage::kUpload);
    buffer.write(0, item.values, item.logical_bytes);
    if (physical != item.logical_bytes) {
      const uint32_t zero = 0;
      buffer.write(item.logical_bytes, &zero, physical - item.logical_bytes);
    }
    staging.push_back(std::move(buffer));
  }
  [[maybe_unused]] auto recording_lock = context.impl_->acquire_recorder();
  CommandList list = context.impl_->commands.begin();
  for (size_t i = 0; i < pending.size(); ++i) {
    DeviceTensor::Impl& destination = *pending[i].destination->impl_;
    const uint64_t physical = destination.buffer.size();
    list.barrier(staging[i], BufferAccess::kHostWrite,
                 BufferAccess::kTransferRead, 0, physical);
    list.copy_buffer(staging[i], destination.buffer, physical);
  }
  Submission uploaded = context.impl_->commands.submit(std::move(list));
  uploaded.wait();
  context.impl_->commands.collect();
  for (const auto& item : pending) {
    item.destination->impl_->has_access = true;
    item.destination->impl_->access = BufferAccess::kTransferWrite;
  }
  staging.clear();
  context.impl_->pool.trim();
  return LinearWeight(std::move(result));
}

LinearWeightFormat LinearWeight::format() const {
  if (!impl_) throw std::logic_error("vulkan linear weight: empty weight");
  return impl_->format;
}
uint32_t LinearWeight::out_features() const {
  if (!impl_) throw std::logic_error("vulkan linear weight: empty weight");
  return impl_->out_features;
}
uint32_t LinearWeight::in_features() const {
  if (!impl_) throw std::logic_error("vulkan linear weight: empty weight");
  return impl_->in_features;
}
uint64_t LinearWeight::stored_bytes() const noexcept {
  return impl_ ? impl_->stored_bytes : 0;
}
uint64_t LinearWeight::resident_bytes() const noexcept {
  return impl_ ? impl_->resident_bytes : 0;
}
bool LinearWeight::has_fp8_input_scale() const noexcept {
  return impl_ && impl_->has_fp8_input_scale;
}
float LinearWeight::fp8_input_scale() const {
  if (!impl_ || !impl_->has_fp8_input_scale) {
    throw std::logic_error("vulkan linear weight: FP8 input scale is absent");
  }
  return impl_->fp8_input_scale;
}
bool LinearWeight::full_precision_matrix_mult() const noexcept {
  return impl_ && impl_->full_precision_matrix_mult;
}
bool LinearWeight::has_pre_quant_scale() const noexcept {
  return impl_ && static_cast<bool>(impl_->pre_quant_scale);
}
bool LinearWeight::applies_convrot() const noexcept {
  return impl_ && impl_->convrot;
}
uint32_t LinearWeight::convrot_group() const noexcept {
  return impl_ ? impl_->convrot_group : 0;
}
void LinearWeight::materialize_bf16(TensorBatch& batch, DeviceTensor& dense) const {
  if (!impl_) throw std::logic_error("vulkan linear weight: empty weight");
  batch.materialize_linear_weight(*this, dense, false);
}
void LinearWeight::materialize_f16(TensorBatch& batch, DeviceTensor& dense) const {
  if (!impl_) throw std::logic_error("vulkan linear weight: empty weight");
  batch.materialize_linear_weight(*this, dense, true);
}
void LinearWeight::apply_pre_quant_scale(TensorBatch& batch, DeviceTensor& input,
                                         DeviceTensor& output) const {
  if (!impl_) throw std::logic_error("vulkan linear weight: empty weight");
  batch.transform_linear_activation(*this, input, output, false);
}
void LinearWeight::apply_convrot(TensorBatch& batch, DeviceTensor& input,
                                 DeviceTensor& output) const {
  if (!impl_) throw std::logic_error("vulkan linear weight: empty weight");
  batch.transform_linear_activation(*this, input, output, true);
}

TensorContext::TensorContext(const Device& device, const TensorContextOptions& options)
    : impl_(std::make_unique<Impl>(device, options)) {
  if (options.max_in_flight == 0) {
    throw std::invalid_argument("vulkan tensor: max_in_flight must be nonzero");
  }
}
TensorContext::~TensorContext() = default;
TensorContext::TensorContext(TensorContext&&) noexcept = default;
TensorContext& TensorContext::operator=(TensorContext&&) noexcept = default;

DeviceTensor TensorContext::allocate(const TensorLayout& layout, ScalarType type) {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  [[maybe_unused]] auto recording_lock = impl_->acquire_recorder();
  if (!layout.is_contiguous()) throw std::invalid_argument("vulkan tensor: contiguous layout required");
  const uint64_t logical_bytes = layout.bytes(type);
  if (logical_bytes > std::numeric_limits<uint64_t>::max() - 3) {
    throw std::overflow_error("vulkan tensor: allocation size overflow");
  }
  const uint64_t bytes = std::max<uint64_t>(4, (logical_bytes + 3) & ~uint64_t{3});
  if (bytes > impl_->max_storage_bytes) {
    throw std::out_of_range("vulkan tensor: tensor exceeds storage buffer limit");
  }
  auto tensor = std::make_unique<DeviceTensor::Impl>();
  tensor->buffer = impl_->pool.allocate(
      bytes, BufferUsage::kStorage | BufferUsage::kTransferSource |
                 BufferUsage::kTransferDestination,
      MemoryUsage::kDevice);
  tensor->layout = layout;
  tensor->type = type;
  tensor->logical_bytes = logical_bytes;
  tensor->context = impl_->context_id;
  return DeviceTensor(std::move(tensor));
}

TensorBatch TensorContext::begin_batch() {
  if (!impl_) throw std::logic_error("vulkan tensor: empty context");
  auto recording_lease = impl_->acquire_recorder();
  auto batch = std::make_unique<TensorBatch::Impl>();
  batch->owner = impl_;
  batch->recording_lease = std::move(recording_lease);
  batch->commands = impl_->commands.begin();
  return TensorBatch(std::move(batch));
}

void TensorContext::upload(DeviceTensor& destination, const float* values, uint64_t count) {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  auto dst = impl_->require(destination);
  if (dst->type != ScalarType::kFloat32 || values == nullptr ||
      count != dst->layout.elements()) {
    throw std::invalid_argument("vulkan tensor: upload element count mismatch");
  }
  upload_bytes(destination, values, count * sizeof(float));
}

void TensorContext::upload_bytes(DeviceTensor& destination, const void* values,
                                 uint64_t bytes) {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  [[maybe_unused]] auto recording_lock = impl_->acquire_recorder();
  auto dst = impl_->require(destination);
  if (values == nullptr || bytes != dst->logical_bytes) {
    throw std::invalid_argument("vulkan tensor: upload byte count mismatch");
  }
  const uint64_t physical_bytes = dst->buffer.size();
  impl_->ensure_staging(physical_bytes);
  impl_->upload_buffer.write(0, values, bytes);
  if (physical_bytes != bytes) {
    const uint32_t zero = 0;
    impl_->upload_buffer.write(bytes, &zero, physical_bytes - bytes);
  }
  CommandList list = impl_->commands.begin();
  list.barrier(impl_->upload_buffer, BufferAccess::kHostWrite,
               BufferAccess::kTransferRead, 0, physical_bytes);
  const bool old_has_access = dst->has_access;
  const BufferAccess old_access = dst->access;
  try {
    if (dst->has_access) {
      list.barrier(dst->buffer, dst->access, BufferAccess::kTransferWrite);
    }
    dst->has_access = true;
    dst->access = BufferAccess::kTransferWrite;
    list.copy_buffer(impl_->upload_buffer, dst->buffer, physical_bytes);
    impl_->complete(std::move(list));
  } catch (...) {
    dst->has_access = old_has_access;
    dst->access = old_access;
    throw;
  }
}

void TensorContext::download(DeviceTensor& source, float* values, uint64_t count) {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  auto src = impl_->require(source);
  if (src->type != ScalarType::kFloat32 || values == nullptr ||
      count != src->layout.elements()) {
    throw std::invalid_argument("vulkan tensor: download element count mismatch");
  }
  download_bytes(source, values, count * sizeof(float));
}

void TensorContext::download_bytes(DeviceTensor& source, void* values,
                                   uint64_t bytes) {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  [[maybe_unused]] auto recording_lock = impl_->acquire_recorder();
  auto src = impl_->require(source);
  if (values == nullptr || bytes != src->logical_bytes) {
    throw std::invalid_argument("vulkan tensor: download byte count mismatch");
  }
  const uint64_t physical_bytes = src->buffer.size();
  impl_->ensure_staging(physical_bytes);
  CommandList list = impl_->commands.begin();
  const bool old_has_access = src->has_access;
  const BufferAccess old_access = src->access;
  try {
    if (src->has_access && src->access != BufferAccess::kTransferRead) {
      list.barrier(src->buffer, src->access, BufferAccess::kTransferRead);
    }
    src->has_access = true;
    src->access = BufferAccess::kTransferRead;
    list.copy_buffer(src->buffer, impl_->readback_buffer, physical_bytes);
    list.barrier(impl_->readback_buffer, BufferAccess::kTransferWrite,
                 BufferAccess::kHostRead, 0, physical_bytes);
    impl_->complete(std::move(list));
  } catch (...) {
    src->has_access = old_has_access;
    src->access = old_access;
    throw;
  }
  impl_->readback_buffer.read(0, values, bytes);
}

void TensorContext::copy(DeviceTensor& source, DeviceTensor& destination) {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  TensorBatch batch = begin_batch();
  batch.copy(source, destination);
  batch.submit().wait();
  impl_->commands.collect();
}

void TensorContext::add(DeviceTensor& a, DeviceTensor& b, DeviceTensor& output) {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  TensorBatch batch = begin_batch();
  batch.add(a, b, output);
  batch.submit().wait();
  impl_->commands.collect();
}

bool TensorContext::full_fp32_arithmetic_exactness() const noexcept {
  return impl_ && impl_->full_arithmetic_exact;
}
void TensorContext::require_full_fp32_arithmetic_exactness() const {
  if (!full_fp32_arithmetic_exactness()) {
    throw std::runtime_error(
          "vulkan tensor: device cannot preserve fp32 subnormal inputs/results; "
        "full CUDA-exact fp32 arithmetic is unavailable");
  }
}
bool TensorContext::exact_normalization() const noexcept {
  return impl_ && impl_->exact_vae_norm;
}
void TensorContext::require_exact_normalization() const {
  if (!exact_normalization()) {
    throw std::runtime_error(
        "vulkan tensor: exact normalization requires a compatible shaderInt64-enabled "
        "Vulkan device");
  }
}
bool TensorContext::exact_fp32_vae_normalization() const noexcept {
  return exact_normalization();
}
void TensorContext::require_exact_fp32_vae_normalization() const {
  require_exact_normalization();
}

TensorWorkspace& TensorContext::workspace() {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  [[maybe_unused]] auto recording_lock = impl_->acquire_recorder();
  return impl_->scratch;
}
uint64_t TensorContext::reserved_bytes() const {
  if (!impl_) return 0;
  [[maybe_unused]] auto recording_lock = impl_->acquire_recorder();
  const uint64_t primary = impl_->pool.reserved_bytes();
  const uint64_t scratch = impl_->scratch.reserved_bytes();
  return primary > std::numeric_limits<uint64_t>::max() - scratch
             ? std::numeric_limits<uint64_t>::max()
             : primary + scratch;
}
uint64_t TensorContext::pooled_used_bytes() const {
  if (!impl_) return 0;
  [[maybe_unused]] auto recording_lock = impl_->acquire_recorder();
  const uint64_t primary = impl_->pool.used_bytes();
  const uint64_t scratch = impl_->scratch.pooled_used_bytes();
  return primary > std::numeric_limits<uint64_t>::max() - scratch
             ? std::numeric_limits<uint64_t>::max()
             : primary + scratch;
}
uint64_t TensorContext::descriptor_set_allocations() const noexcept {
  return impl_ ? impl_->commands.descriptor_set_allocations() : 0;
}

TensorBatch::TensorBatch() = default;
TensorBatch::~TensorBatch() = default;
TensorBatch::TensorBatch(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
TensorBatch::TensorBatch(TensorBatch&&) noexcept = default;
TensorBatch& TensorBatch::operator=(TensorBatch&&) noexcept = default;
TensorBatch::operator bool() const noexcept { return impl_ != nullptr; }

void TensorBatch::copy(DeviceTensor& source, DeviceTensor& destination) {
  if (!impl_) throw std::logic_error("vulkan tensor: empty batch");
  if (impl_->poisoned) throw std::logic_error("vulkan tensor: batch is poisoned");
  auto src = impl_->owner->require(source);
  auto dst = impl_->owner->require(destination);
  const uint64_t bytes = src->buffer.size();
  if (src.get() == dst.get() || src->type != dst->type ||
      src->logical_bytes != dst->logical_bytes || bytes != dst->buffer.size()) {
    throw std::invalid_argument("vulkan tensor: copy needs distinct equal-sized tensors");
  }
  try {
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kTransferRead);
    impl_->transition(dst, BufferAccess::kTransferWrite);
    impl_->commands.copy_buffer(src->buffer, dst->buffer, bytes);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::add(DeviceTensor& a, DeviceTensor& b, DeviceTensor& output) {
  if (!impl_) throw std::logic_error("vulkan tensor: empty batch");
  if (impl_->poisoned) throw std::logic_error("vulkan tensor: batch is poisoned");
  auto av = impl_->owner->require(a);
  auto bv = impl_->owner->require(b);
  auto out = impl_->owner->require(output);
  const uint64_t count = av->layout.elements();
  if (out.get() == av.get() || out.get() == bv.get() ||
      av->type != ScalarType::kFloat32 || bv->type != ScalarType::kFloat32 ||
      out->type != ScalarType::kFloat32 ||
      count != bv->layout.elements() || count != out->layout.elements() ||
      count == 0 || count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: add needs distinct equal-sized fp32 tensors");
  }
  if (count > impl_->owner->max_storage_bytes / sizeof(float)) {
    throw std::out_of_range("vulkan tensor: add exceeds device dispatch limits");
  }
  const uint32_t dispatch_count = impl_->owner->validate_dispatch(count);
  try {
    impl_->count_operator();
    impl_->transition(av, BufferAccess::kComputeRead);
    impl_->transition(bv, BufferAccess::kComputeRead);
    impl_->transition(out, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters parameters;
    parameters.op = 0;
    parameters.count = dispatch_count;
    impl_->dispatch(parameters, av, bv, out);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::convert(DeviceTensor& source, DeviceTensor& destination) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(source);
  auto dst = impl_->owner->require(destination);
  const uint64_t elements = src->layout.elements();
  if (src.get() == dst.get() || elements != dst->layout.elements() ||
      elements > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: conversion shape mismatch");
  }
  uint32_t op = 0;
  bool narrowing = false;
  if (src->type == ScalarType::kFloat32 && dst->type == ScalarType::kBFloat16) {
    op = 1; narrowing = true;
  } else if (src->type == ScalarType::kBFloat16 && dst->type == ScalarType::kFloat32) {
    op = 2;
  } else if (src->type == ScalarType::kFloat32 && dst->type == ScalarType::kFloat16) {
    op = 3; narrowing = true;
  } else if (src->type == ScalarType::kFloat16 && dst->type == ScalarType::kFloat32) {
    op = 4;
  } else {
    throw std::invalid_argument("vulkan tensor: unsupported dtype conversion");
  }
  const uint64_t invocation_count = narrowing ? (elements + 1) / 2 : elements;
  const uint32_t dispatch_count = impl_->owner->validate_dispatch(invocation_count);
  try {
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters parameters;
    parameters.op = op;
    parameters.count = dispatch_count;
    parameters.p[0] = static_cast<uint32_t>(elements);
    impl_->dispatch(parameters, src, src, dst);
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::transpose_2d(DeviceTensor& source, DeviceTensor& destination) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(source);
  auto dst = impl_->owner->require(destination);
  const auto& a = src->layout;
  const auto& o = dst->layout;
  if (src.get() == dst.get() || src->type != ScalarType::kFloat32 ||
      dst->type != ScalarType::kFloat32 || a.rank != 2 || o.rank != 2 ||
      o.extent[0] != a.extent[1] || o.extent[1] != a.extent[0] ||
      a.elements() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid fp32 transpose");
  }
  if (a.extent[0] > std::numeric_limits<uint32_t>::max() ||
      a.extent[1] > std::numeric_limits<uint32_t>::max()) {
    throw std::out_of_range("vulkan tensor: transpose dimensions exceed uint32 limits");
  }
  const uint32_t dispatch_count = impl_->owner->validate_dispatch(a.elements());
  try {
    impl_->count_operator();
    impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters p;
    p.op = 5; p.count = dispatch_count;
    p.p[0] = static_cast<uint32_t>(a.extent[0]);
    p.p[1] = static_cast<uint32_t>(a.extent[1]);
    impl_->dispatch(p, src, src, dst);
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::gather_rows(DeviceTensor& source, DeviceTensor& indices,
                              DeviceTensor& destination) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(source);
  auto idx = impl_->owner->require(indices);
  auto dst = impl_->owner->require(destination);
  const auto& a = src->layout; const auto& ix = idx->layout; const auto& o = dst->layout;
  if (src.get() == dst.get() || src.get() == idx.get() || dst.get() == idx.get() ||
      src->type != ScalarType::kFloat32 || dst->type != ScalarType::kFloat32 ||
      idx->type != ScalarType::kInt32 || a.rank != 2 || ix.rank != 1 || o.rank != 2 ||
      ix.extent[0] != o.extent[0] || a.extent[1] != o.extent[1] ||
      a.elements() > std::numeric_limits<uint32_t>::max() ||
      o.elements() > std::numeric_limits<uint32_t>::max() ||
      a.extent[0] > std::numeric_limits<uint32_t>::max() ||
      a.extent[1] > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid gather rows");
  }
  const uint32_t dispatch_count = impl_->owner->validate_dispatch(o.elements());
  try {
    impl_->count_operator(); impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(idx, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters p; p.op = 6;
    p.count = dispatch_count;
    p.p[0] = static_cast<uint32_t>(a.extent[0]); p.p[1] = static_cast<uint32_t>(a.extent[1]);
    impl_->dispatch(p, src, idx, dst);
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::scatter_rows(DeviceTensor& source, DeviceTensor& indices,
                               DeviceTensor& destination) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(source); auto idx = impl_->owner->require(indices);
  auto dst = impl_->owner->require(destination);
  const auto& a = src->layout; const auto& ix = idx->layout; const auto& o = dst->layout;
  if (src.get() == dst.get() || src.get() == idx.get() || dst.get() == idx.get() ||
      src->type != ScalarType::kFloat32 || dst->type != ScalarType::kFloat32 ||
      idx->type != ScalarType::kInt32 || a.rank != 2 || ix.rank != 1 || o.rank != 2 ||
      ix.extent[0] != a.extent[0] || a.extent[1] != o.extent[1] ||
      a.elements() > std::numeric_limits<uint32_t>::max() ||
      o.elements() > std::numeric_limits<uint32_t>::max() ||
      o.extent[0] > std::numeric_limits<uint32_t>::max() ||
      a.extent[1] > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid scatter rows");
  }
  const uint32_t dispatch_count = impl_->owner->validate_dispatch(a.elements());
  try {
    impl_->count_operator(); impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(idx, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters p; p.op = 7;
    p.count = dispatch_count; p.p[0] = static_cast<uint32_t>(o.extent[0]);
    p.p[1] = static_cast<uint32_t>(a.extent[1]); impl_->dispatch(p, src, idx, dst);
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::add_bias(DeviceTensor& input, DeviceTensor& bias, DeviceTensor& output) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(input); auto bv = impl_->owner->require(bias);
  auto dst = impl_->owner->require(output);
  const auto& a = src->layout; const auto& b = bv->layout; const auto& o = dst->layout;
  if (src.get() == bv.get() || dst.get() == bv.get() ||
      src->type != ScalarType::kFloat32 || bv->type != ScalarType::kFloat32 ||
      dst->type != ScalarType::kFloat32 || a.rank != 2 || b.rank != 1 || o.rank != 2 ||
      b.extent[0] != a.extent[1] || o.extent != a.extent ||
      a.elements() > std::numeric_limits<uint32_t>::max() ||
      a.extent[1] > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid add-bias");
  }
  const uint32_t dispatch_count = impl_->owner->validate_dispatch(a.elements());
  try {
    impl_->count_operator();
    impl_->transition(src, src.get() == dst.get() ? BufferAccess::kComputeReadWrite
                                                  : BufferAccess::kComputeRead);
    impl_->transition(bv, BufferAccess::kComputeRead);
    if (src.get() != dst.get()) impl_->transition(dst, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters p; p.op = 8;
    p.count = dispatch_count; p.p[1] = static_cast<uint32_t>(a.extent[1]);
    impl_->dispatch(p, src, bv, dst);
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::heads_to_tokens_bf16(DeviceTensor& source, DeviceTensor& destination,
                                       uint32_t heads, uint32_t sequence,
                                       uint32_t head_dim) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(source); auto dst = impl_->owner->require(destination);
  const uint64_t elements = checked_multiply(
      checked_multiply(heads, sequence, "heads-to-tokens"), head_dim,
      "heads-to-tokens");
  if (heads == 0 || sequence == 0 || head_dim == 0 ||
      src.get() == dst.get() || src->type != ScalarType::kFloat32 ||
      dst->type != ScalarType::kBFloat16 ||
      src->layout.elements() != elements || dst->layout.elements() != elements ||
      elements > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid heads-to-tokens conversion");
  }
  const uint32_t dispatch_count = impl_->owner->validate_dispatch((elements + 1) / 2);
  try {
    impl_->count_operator(); impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters p; p.op = 9;
    p.count = dispatch_count;
    p.p[0] = heads; p.p[1] = head_dim; p.p[2] = sequence;
    p.p[3] = static_cast<uint32_t>(elements); impl_->dispatch(p, src, src, dst);
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::depth_to_space(DeviceTensor& source, DeviceTensor& destination,
                                 uint32_t time, uint32_t height, uint32_t width,
                                 uint32_t channels, uint32_t patch_time,
                                 uint32_t patch) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  auto src = impl_->owner->require(source); auto dst = impl_->owner->require(destination);
  if (time == 0 || height == 0 || width == 0 || channels == 0 ||
      patch_time == 0 || patch == 0) {
    throw std::invalid_argument("vulkan tensor: depth-to-space dimensions must be nonzero");
  }
  const uint64_t tokens = checked_multiply(
      checked_multiply(time, height, "depth-to-space"), width, "depth-to-space");
  const uint64_t patch_volume = checked_multiply(
      checked_multiply(patch_time, patch, "depth-to-space"), patch,
      "depth-to-space");
  const uint64_t elements = checked_multiply(
      checked_multiply(tokens, channels, "depth-to-space"), patch_volume,
      "depth-to-space");
  if (patch_volume > std::numeric_limits<uint32_t>::max() ||
      src.get() == dst.get() || src->type != ScalarType::kFloat32 ||
      dst->type != ScalarType::kFloat32 ||
      src->layout.elements() != elements || dst->layout.elements() != elements ||
      elements > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid depth-to-space tensors");
  }
  const uint32_t dispatch_count = impl_->owner->validate_dispatch(elements);
  try {
    impl_->count_operator(); impl_->transition(src, BufferAccess::kComputeRead);
    impl_->transition(dst, BufferAccess::kComputeWrite);
    TensorContext::Impl::Parameters p; p.op = 10; p.count = dispatch_count;
    p.p[0] = time; p.p[1] = height; p.p[2] = width; p.p[3] = patch_time;
    p.p[4] = patch; p.p[5] = channels; impl_->dispatch(p, src, src, dst);
  } catch (...) { impl_->poisoned = true; throw; }
}

void TensorBatch::rope_h3_bf16(DeviceTensor& input, DeviceTensor& cosine,
                                DeviceTensor& sine) {
  rope_bf16(input, cosine, sine, 0);
}

void TensorBatch::rope_neox_bf16(DeviceTensor& input, DeviceTensor& cosine,
                                  DeviceTensor& sine) {
  rope_bf16(input, cosine, sine, 1);
}

void TensorBatch::rope_bf16(DeviceTensor& input, DeviceTensor& cosine,
                             DeviceTensor& sine, uint32_t mode) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  auto data = impl_->owner->require(input);
  auto cos_table = impl_->owner->require(cosine);
  auto sin_table = impl_->owner->require(sine);
  const auto& shape = data->layout;
  const uint64_t rows = shape.extent[0];
  const uint64_t heads = shape.extent[1];
  const uint64_t head_dim = shape.extent[2];
  const uint64_t table_width = mode == 0 ? 96 : head_dim;
  const bool groups_overflow = rows != 0 && heads > UINT64_MAX / rows;
  const uint64_t groups = groups_overflow ? 0 : rows * heads;
  if (mode > 1 || shape.rank != 3 || rows == 0 || heads == 0 ||
      head_dim == 0 || (head_dim & 1u) != 0 || head_dim > 256 ||
      (mode == 0 && head_dim < 96) || groups_overflow ||
      groups > std::numeric_limits<uint32_t>::max() ||
      data.get() == cos_table.get() || data.get() == sin_table.get() ||
      cos_table.get() == sin_table.get() ||
      data->type != ScalarType::kBFloat16 ||
      cos_table->type != ScalarType::kFloat32 ||
      sin_table->type != ScalarType::kFloat32 ||
      cos_table->layout.rank != 2 || sin_table->layout.rank != 2 ||
      cos_table->layout.extent[0] != rows ||
      sin_table->layout.extent[0] != rows ||
      cos_table->layout.extent[1] != table_width ||
      sin_table->layout.extent[1] != table_width ||
      !shape.is_contiguous() || !cos_table->layout.is_contiguous() ||
      !sin_table->layout.is_contiguous()) {
    throw std::invalid_argument("vulkan tensor: invalid BF16 RoPE");
  }
  if (!detail::norm_dispatch_fits(groups, impl_->owner->max_dispatch_x)) {
    throw std::out_of_range("vulkan tensor: RoPE row-head count exceeds dispatch limits");
  }
  TensorContext::Impl::Parameters parameters;
  parameters.op = mode;
  parameters.p[0] = static_cast<uint32_t>(rows);
  parameters.p[1] = static_cast<uint32_t>(heads);
  parameters.p[2] = static_cast<uint32_t>(head_dim);
  try {
    impl_->count_operator();
    impl_->transition(data, BufferAccess::kComputeReadWrite);
    impl_->transition(cos_table, BufferAccess::kComputeRead);
    impl_->transition(sin_table, BufferAccess::kComputeRead);
    impl_->dispatch_rope(parameters, static_cast<uint32_t>(groups), data,
                         cos_table, sin_table);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::split_qkv_norm_rope_f32(
    DeviceTensor& qkv, DeviceTensor& bias, DeviceTensor& cosine,
    DeviceTensor& sine, DeviceTensor& q, DeviceTensor& k, DeviceTensor& v,
    uint32_t num_patches, float epsilon) {
  if (!impl_ || impl_->poisoned) {
    throw std::logic_error("vulkan tensor: invalid batch");
  }
  if (!impl_->owner->exact_vae_norm) {
    throw std::runtime_error(
        "vulkan tensor: exact video-VAE fused RoPE is unavailable");
  }
  std::array<std::shared_ptr<DeviceTensor::Impl>, 7> resources{
      impl_->owner->require(qkv), impl_->owner->require(bias),
      impl_->owner->require(cosine), impl_->owner->require(sine),
      impl_->owner->require(q), impl_->owner->require(k),
      impl_->owner->require(v)};
  bool aliases = false;
  for (size_t i = 0; i < resources.size(); ++i) {
    for (size_t j = i + 1; j < resources.size(); ++j) {
      aliases = aliases || resources[i].get() == resources[j].get();
    }
  }
  const auto& input_shape = resources[0]->layout;
  const uint64_t sequence = input_shape.extent[0];
  const uint64_t heads = input_shape.extent[1];
  const bool groups_overflow = sequence != 0 && heads > UINT64_MAX / sequence;
  const uint64_t groups = groups_overflow ? 0 : sequence * heads;
  if (aliases || !std::isnormal(epsilon) || epsilon <= 0.0f ||
      input_shape.rank != 3 || sequence == 0 || heads == 0 ||
      input_shape.extent[2] != 192 || num_patches > sequence ||
      groups_overflow || groups > std::numeric_limits<uint32_t>::max() ||
      resources[1]->layout.rank != 2 ||
      resources[1]->layout.extent[0] != heads ||
      resources[1]->layout.extent[1] != 192 ||
      resources[2]->layout.rank != 2 || resources[3]->layout.rank != 2 ||
      resources[2]->layout.extent[0] != sequence ||
      resources[3]->layout.extent[0] != sequence ||
      resources[2]->layout.extent[1] != 48 ||
      resources[3]->layout.extent[1] != 48) {
    throw std::invalid_argument("vulkan tensor: invalid video-VAE fused RoPE");
  }
  for (size_t i = 0; i < resources.size(); ++i) {
    if (resources[i]->type != ScalarType::kFloat32 ||
        !resources[i]->layout.is_contiguous()) {
      throw std::invalid_argument("vulkan tensor: invalid video-VAE fused RoPE");
    }
    if (i >= 4 &&
        (resources[i]->layout.rank != 3 ||
         resources[i]->layout.extent[0] != heads ||
         resources[i]->layout.extent[1] != sequence ||
         resources[i]->layout.extent[2] != 64)) {
      throw std::invalid_argument("vulkan tensor: invalid video-VAE fused RoPE");
    }
  }
  if (!detail::norm_dispatch_fits(groups, impl_->owner->max_dispatch_x)) {
    throw std::out_of_range(
        "vulkan tensor: video-VAE fused RoPE row-head count exceeds dispatch limits");
  }
  TensorContext::Impl::VaeRopeParameters parameters;
  parameters.sequence = static_cast<uint32_t>(sequence);
  parameters.heads = static_cast<uint32_t>(heads);
  parameters.num_patches = num_patches;
  std::memcpy(&parameters.epsilon_bits, &epsilon, sizeof(epsilon));
  try {
    impl_->count_operator();
    for (size_t i = 0; i < 4; ++i) {
      impl_->transition(resources[i], BufferAccess::kComputeRead);
    }
    for (size_t i = 4; i < resources.size(); ++i) {
      impl_->transition(resources[i], BufferAccess::kComputeWrite);
    }
    impl_->dispatch_vae_rope(parameters, static_cast<uint32_t>(groups),
                             resources);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::rms_norm(DeviceTensor& input, DeviceTensor& weight,
                           DeviceTensor& output, float epsilon) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_vae_norm) {
    throw std::runtime_error("vulkan tensor: exact fp32 VAE RMSNorm is unavailable");
  }
  auto src = impl_->owner->require(input);
  auto w = impl_->owner->require(weight);
  auto dst = impl_->owner->require(output);
  const auto& shape = src->layout;
  if (!std::isnormal(epsilon) || epsilon <= 0.0f || src.get() == w.get() ||
      dst.get() == w.get() || src->type != ScalarType::kFloat32 ||
      w->type != ScalarType::kFloat32 || dst->type != ScalarType::kFloat32 ||
      shape.rank != 2 || w->layout.rank != 1 || dst->layout.extent != shape.extent ||
      dst->layout.rank != 2 || w->layout.extent[0] != shape.extent[1] ||
      !shape.is_contiguous() || !w->layout.is_contiguous() ||
      !dst->layout.is_contiguous() ||
      shape.extent[0] > std::numeric_limits<uint32_t>::max() ||
      shape.extent[1] > kMaxExactNormDimension ||
      shape.elements() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid fp32 RMSNorm");
  }
  if (!detail::norm_dispatch_fits(shape.extent[0], impl_->owner->max_dispatch_x)) {
    throw std::out_of_range("vulkan tensor: RMSNorm row count exceeds dispatch limits");
  }
  TensorContext::Impl::NormParameters p;
  p.rows = static_cast<uint32_t>(shape.extent[0]);
  p.dim = static_cast<uint32_t>(shape.extent[1]);
  std::memcpy(&p.epsilon_bits, &epsilon, sizeof(epsilon));
  try {
    impl_->count_operator();
    impl_->transition(src, src.get() == dst.get() ? BufferAccess::kComputeReadWrite
                                                   : BufferAccess::kComputeRead);
    impl_->transition(w, BufferAccess::kComputeRead);
    if (src.get() != dst.get()) impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_norm(impl_->owner->rms_norm_pipeline, p, src, w, w, dst);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::layer_norm(DeviceTensor& input, DeviceTensor& weight,
                             DeviceTensor& bias, DeviceTensor& output,
                             float epsilon) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_vae_norm) {
    throw std::runtime_error("vulkan tensor: exact fp32 VAE LayerNorm is unavailable");
  }
  auto src = impl_->owner->require(input);
  auto w = impl_->owner->require(weight);
  auto b = impl_->owner->require(bias);
  auto dst = impl_->owner->require(output);
  const auto& shape = src->layout;
  if (!std::isnormal(epsilon) || epsilon <= 0.0f || src.get() == w.get() ||
      src.get() == b.get() || w.get() == b.get() || dst.get() == w.get() ||
      dst.get() == b.get() || src->type != ScalarType::kFloat32 ||
      w->type != ScalarType::kFloat32 || b->type != ScalarType::kFloat32 ||
      dst->type != ScalarType::kFloat32 || shape.rank != 2 ||
      w->layout.rank != 1 || b->layout.rank != 1 || dst->layout.rank != 2 ||
      dst->layout.extent != shape.extent || w->layout.extent[0] != shape.extent[1] ||
      b->layout.extent[0] != shape.extent[1] ||
      !shape.is_contiguous() || !w->layout.is_contiguous() ||
      !b->layout.is_contiguous() || !dst->layout.is_contiguous() ||
      shape.extent[0] > std::numeric_limits<uint32_t>::max() ||
      shape.extent[1] > kMaxExactNormDimension ||
      shape.elements() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid fp32 LayerNorm");
  }
  if (!detail::norm_dispatch_fits(shape.extent[0], impl_->owner->max_dispatch_x)) {
    throw std::out_of_range("vulkan tensor: LayerNorm row count exceeds dispatch limits");
  }
  TensorContext::Impl::NormParameters p;
  p.rows = static_cast<uint32_t>(shape.extent[0]);
  p.dim = static_cast<uint32_t>(shape.extent[1]);
  std::memcpy(&p.epsilon_bits, &epsilon, sizeof(epsilon));
  try {
    impl_->count_operator();
    impl_->transition(src, src.get() == dst.get() ? BufferAccess::kComputeReadWrite
                                                   : BufferAccess::kComputeRead);
    impl_->transition(w, BufferAccess::kComputeRead);
    impl_->transition(b, BufferAccess::kComputeRead);
    if (src.get() != dst.get()) impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_norm(impl_->owner->layer_norm_pipeline, p, src, w, b, dst);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::rms_norm_bf16(DeviceTensor& input, DeviceTensor& weight,
                                DeviceTensor& output, float epsilon) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_vae_norm) {
    throw std::runtime_error("vulkan tensor: exact BF16 RMSNorm is unavailable");
  }
  auto src = impl_->owner->require(input);
  auto w = impl_->owner->require(weight);
  auto dst = impl_->owner->require(output);
  const auto& shape = src->layout;
  const uint64_t rows = shape.extent[0];
  const uint64_t dim = shape.extent[1];
  if (!std::isnormal(epsilon) || epsilon <= 0.0f || src.get() == w.get() ||
      dst.get() == w.get() || src->type != ScalarType::kBFloat16 ||
      w->type != ScalarType::kBFloat16 || dst->type != ScalarType::kBFloat16 ||
      shape.rank != 2 || w->layout.rank != 1 || dst->layout.rank != 2 ||
      dst->layout.extent != shape.extent || w->layout.extent[0] != dim ||
      !shape.is_contiguous() || !w->layout.is_contiguous() ||
      !dst->layout.is_contiguous() || rows > std::numeric_limits<uint32_t>::max() ||
      dim > kMaxExactNormDimension ||
      shape.elements() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid BF16 RMSNorm");
  }
  const uint64_t vec = (dim & 7u) == 0 ? 8u : 1u;
  const bool narrow = dim / vec <= 32u;
  const uint64_t groups = narrow ? (rows + 7u) / 8u : rows;
  if (!detail::norm_dispatch_fits(groups, impl_->owner->max_dispatch_x)) {
    throw std::out_of_range("vulkan tensor: BF16 RMSNorm rows exceed dispatch limits");
  }
  TensorContext::Impl::NormParameters p;
  p.rows = static_cast<uint32_t>(rows);
  p.dim = static_cast<uint32_t>(dim);
  std::memcpy(&p.epsilon_bits, &epsilon, sizeof(epsilon));
  try {
    impl_->count_operator();
    impl_->transition(src, src.get() == dst.get() ? BufferAccess::kComputeReadWrite
                                                   : BufferAccess::kComputeRead);
    impl_->transition(w, BufferAccess::kComputeRead);
    if (src.get() != dst.get()) impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_shared_rms(narrow ? impl_->owner->bf16_rms_narrow_pipeline
                                      : impl_->owner->bf16_rms_block_pipeline,
                               p, static_cast<uint32_t>(groups), src, w, dst);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::layer_norm_bf16(DeviceTensor& input, DeviceTensor& weight,
                                  DeviceTensor& bias, DeviceTensor& output,
                                  float epsilon) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_vae_norm) {
    throw std::runtime_error("vulkan tensor: exact BF16 LayerNorm is unavailable");
  }
  auto src = impl_->owner->require(input);
  auto w = impl_->owner->require(weight);
  auto b = impl_->owner->require(bias);
  auto dst = impl_->owner->require(output);
  const auto& shape = src->layout;
  const uint64_t rows = shape.extent[0];
  const uint64_t dim = shape.extent[1];
  if (!std::isnormal(epsilon) || epsilon <= 0.0f || src.get() == w.get() ||
      src.get() == b.get() || w.get() == b.get() || dst.get() == w.get() ||
      dst.get() == b.get() || src->type != ScalarType::kBFloat16 ||
      w->type != ScalarType::kBFloat16 || b->type != ScalarType::kBFloat16 ||
      dst->type != ScalarType::kBFloat16 || shape.rank != 2 ||
      w->layout.rank != 1 || b->layout.rank != 1 || dst->layout.rank != 2 ||
      dst->layout.extent != shape.extent || w->layout.extent[0] != dim ||
      b->layout.extent[0] != dim || !shape.is_contiguous() ||
      !w->layout.is_contiguous() || !b->layout.is_contiguous() ||
      !dst->layout.is_contiguous() || rows > std::numeric_limits<uint32_t>::max() ||
      dim > kMaxExactNormDimension ||
      shape.elements() > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: invalid BF16 LayerNorm");
  }
  if (!detail::norm_dispatch_fits(rows, impl_->owner->max_dispatch_x)) {
    throw std::out_of_range("vulkan tensor: BF16 LayerNorm rows exceed dispatch limits");
  }
  TensorContext::Impl::NormParameters p;
  p.rows = static_cast<uint32_t>(rows);
  p.dim = static_cast<uint32_t>(dim);
  std::memcpy(&p.epsilon_bits, &epsilon, sizeof(epsilon));
  try {
    impl_->count_operator();
    impl_->transition(src, src.get() == dst.get() ? BufferAccess::kComputeReadWrite
                                                   : BufferAccess::kComputeRead);
    impl_->transition(w, BufferAccess::kComputeRead);
    impl_->transition(b, BufferAccess::kComputeRead);
    if (src.get() != dst.get()) impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_shared_layer(impl_->owner->bf16_layer_pipeline, p, src, w, b, dst);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::rms_norm_modulate_bf16(DeviceTensor& input, DeviceTensor& weight,
                                         DeviceTensor& scale, DeviceTensor& shift,
                                         DeviceTensor& selectors, DeviceTensor& output,
                                         float epsilon) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  impl_->record_shared_mod(false, input, weight, scale, shift, selectors, output,
                           epsilon);
}

void TensorBatch::rms_norm_modulate_f32(DeviceTensor& input, DeviceTensor& weight,
                                        DeviceTensor& scale, DeviceTensor& shift,
                                        DeviceTensor& selectors, DeviceTensor& output,
                                        float epsilon) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  impl_->record_shared_mod(true, input, weight, scale, shift, selectors, output,
                           epsilon);
}

void TensorBatch::group_norm_silu_f16_affine(DeviceTensor& input,
                                              DeviceTensor& weight,
                                              DeviceTensor& bias,
                                              DeviceTensor& output,
                                              uint32_t groups, float epsilon) {
  if (!impl_ || impl_->poisoned) throw std::logic_error("vulkan tensor: invalid batch");
  if (!impl_->owner->exact_vae_norm) {
    throw std::runtime_error("vulkan tensor: exact VAE GroupNorm+SiLU is unavailable");
  }
  auto src = impl_->owner->require(input);
  auto w = impl_->owner->require(weight);
  auto b = impl_->owner->require(bias);
  auto dst = impl_->owner->require(output);
  const auto& shape = src->layout;
  const uint64_t channels = shape.extent[0];
  uint64_t spatial = 0;
  if (shape.rank == 3 &&
      shape.extent[1] <= std::numeric_limits<uint64_t>::max() / shape.extent[2]) {
    spatial = shape.extent[1] * shape.extent[2];
  }
  const uint64_t channels_per_group = groups == 0 ? 0 : channels / groups;
  const bool group_count_overflows = spatial != 0 &&
      channels_per_group > std::numeric_limits<uint64_t>::max() / spatial;
  const uint64_t group_count = group_count_overflows ? 0 : channels_per_group * spatial;
  if (!std::isnormal(epsilon) || epsilon <= 0.0f || groups == 0 ||
      shape.rank != 3 || spatial == 0 || channels == 0 || channels % groups != 0 ||
      group_count_overflows || channels > std::numeric_limits<uint32_t>::max() ||
      spatial > std::numeric_limits<uint32_t>::max() ||
      group_count > kMaxExactNormDimension ||
      shape.elements() > std::numeric_limits<uint32_t>::max() ||
      src.get() == w.get() || src.get() == b.get() || w.get() == b.get() ||
      dst.get() == w.get() || dst.get() == b.get() ||
      src->type != ScalarType::kFloat32 || dst->type != ScalarType::kFloat32 ||
      w->type != ScalarType::kFloat16 || b->type != ScalarType::kFloat16 ||
      w->layout.rank != 1 || b->layout.rank != 1 || dst->layout.rank != 3 ||
      w->layout.extent[0] != channels || b->layout.extent[0] != channels ||
      dst->layout.extent != shape.extent || !shape.is_contiguous() ||
      !w->layout.is_contiguous() || !b->layout.is_contiguous() ||
      !dst->layout.is_contiguous()) {
    throw std::invalid_argument("vulkan tensor: invalid VAE GroupNorm+SiLU");
  }
  if (!detail::norm_dispatch_fits(groups, impl_->owner->max_dispatch_x)) {
    throw std::out_of_range("vulkan tensor: GroupNorm group count exceeds dispatch limits");
  }
  TensorContext::Impl::NormParameters p;
  p.rows = static_cast<uint32_t>(channels);
  p.dim = static_cast<uint32_t>(spatial);
  p.mod_rows = groups;
  std::memcpy(&p.epsilon_bits, &epsilon, sizeof(epsilon));
  try {
    impl_->count_operator();
    impl_->transition(src, src.get() == dst.get() ? BufferAccess::kComputeReadWrite
                                                   : BufferAccess::kComputeRead);
    impl_->transition(w, BufferAccess::kComputeRead);
    impl_->transition(b, BufferAccess::kComputeRead);
    if (src.get() != dst.get()) impl_->transition(dst, BufferAccess::kComputeWrite);
    impl_->dispatch_group_norm(p, src, w, b, dst);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::materialize_linear_weight(const LinearWeight& weight,
                                             DeviceTensor& dense, bool fp16) {
  if (!impl_ || impl_->poisoned) {
    throw std::logic_error("vulkan tensor: invalid batch");
  }
  if (!weight.impl_) {
    throw std::logic_error("vulkan linear weight: empty weight");
  }
  auto output = impl_->owner->require(dense);
  auto data = impl_->owner->require(weight.impl_->data);
  const uint64_t elements = checked_multiply(weight.impl_->out_features,
                                              weight.impl_->in_features,
                                              "linear materialization");
  const ScalarType output_type = fp16 ? ScalarType::kFloat16
                                      : ScalarType::kBFloat16;
  if (dense.layout().rank != 2 ||
      dense.layout().extent[0] != weight.impl_->out_features ||
      dense.layout().extent[1] != weight.impl_->in_features ||
      !dense.layout().is_contiguous() || output->type != output_type ||
      elements > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(
        "vulkan linear weight: dense target must be contiguous [out,in]");
  }

  std::array<std::shared_ptr<DeviceTensor::Impl>, 6> resources;
  resources.fill(data);
  resources[5] = output;
  uint32_t op = 0;
  switch (weight.impl_->format) {
    case LinearWeightFormat::kFloat32: op = 0; break;
    case LinearWeightFormat::kFloat16: op = 1; break;
    case LinearWeightFormat::kBFloat16: op = 2; break;
    case LinearWeightFormat::kFloat8E4M3:
      op = 3;
      resources[1] = impl_->owner->require(weight.impl_->weight_scale);
      break;
    case LinearWeightFormat::kInt8:
      op = 4;
      resources[1] = impl_->owner->require(weight.impl_->weight_scale);
      break;
    case LinearWeightFormat::kNVFloat4:
      op = 5;
      resources[1] = impl_->owner->require(weight.impl_->block_scale);
      break;
    case LinearWeightFormat::kNF4:
      op = 6;
      resources[1] = impl_->owner->require(weight.impl_->nf4_absmax);
      resources[2] = impl_->owner->require(weight.impl_->nf4_quant_map);
      resources[3] = impl_->owner->require(weight.impl_->nf4_nested_quant_map);
      resources[4] = impl_->owner->require(weight.impl_->nf4_nested_absmax);
      break;
  }
  for (size_t i = 0; i + 1 < resources.size(); ++i) {
    if (resources[i].get() == output.get()) {
      throw std::invalid_argument(
          "vulkan linear weight: dense target aliases persistent storage");
    }
  }
  const uint64_t invocations = 1 + (elements - 1) / 2;
  impl_->owner->validate_dispatch(invocations);

  TensorContext::Impl::WeightParameters parameters;
  parameters.op = op;
  parameters.count = static_cast<uint32_t>(elements);
  parameters.out_features = fp16 ? 1u : 0u;
  parameters.in_features = weight.impl_->in_features;
  parameters.block_size = weight.impl_->nf4_block_size;
  parameters.nested_block_size = weight.impl_->nf4_nested_block_size;
  const float scalar = op == 5 ? weight.impl_->global_scale
                               : weight.impl_->nf4_nested_offset;
  std::memcpy(&parameters.scalar_bits, &scalar, sizeof(scalar));
  const uint32_t groups = static_cast<uint32_t>((invocations + 63) / 64);
  try {
    impl_->count_operator();
    for (size_t i = 0; i + 1 < resources.size(); ++i) {
      bool duplicate = false;
      for (size_t j = 0; j < i; ++j) {
        duplicate = duplicate || resources[j].get() == resources[i].get();
      }
      if (!duplicate) impl_->transition(resources[i], BufferAccess::kComputeRead);
    }
    impl_->transition(output, BufferAccess::kComputeWrite);
    impl_->dispatch_weight(parameters, groups, resources);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

void TensorBatch::transform_linear_activation(const LinearWeight& weight,
                                               DeviceTensor& input,
                                               DeviceTensor& output,
                                               bool convrot) {
  if (!impl_ || impl_->poisoned) {
    throw std::logic_error("vulkan tensor: invalid batch");
  }
  if (!weight.impl_) {
    throw std::logic_error("vulkan linear weight: empty weight");
  }
  auto source = impl_->owner->require(input);
  auto destination = impl_->owner->require(output);
  const auto& shape = source->layout;
  const uint64_t elements = shape.elements();
  if (source.get() == destination.get() || shape.rank != 2 ||
      shape.extent[1] != weight.impl_->in_features ||
      destination->layout.rank != 2 ||
      destination->layout.extent != shape.extent ||
      !shape.is_contiguous() || !destination->layout.is_contiguous() ||
      source->type != destination->type ||
      (source->type != ScalarType::kBFloat16 &&
       source->type != ScalarType::kFloat32) ||
      elements > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument(
        "vulkan linear weight: transform needs distinct contiguous [rows,in] tensors");
  }

  std::array<std::shared_ptr<DeviceTensor::Impl>, 6> resources;
  resources.fill(source);
  resources[5] = destination;
  TensorContext::Impl::WeightParameters parameters;
  parameters.count = static_cast<uint32_t>(elements);
  parameters.in_features = weight.impl_->in_features;
  uint32_t groups = 0;
  if (convrot) {
    if (!weight.impl_->convrot) {
      throw std::invalid_argument("vulkan linear weight: ConvRot is not enabled");
    }
    parameters.op = source->type == ScalarType::kFloat32 ? 10u : 9u;
    parameters.group = weight.impl_->convrot_group;
    const uint64_t group_count = elements / weight.impl_->convrot_group;
    if (group_count == 0 || group_count > impl_->owner->max_dispatch_x) {
      throw std::out_of_range(
          "vulkan linear weight: ConvRot exceeds dispatch limits");
    }
    groups = static_cast<uint32_t>(group_count);
    const float normalization =
        1.0f / std::sqrt(static_cast<float>(weight.impl_->convrot_group));
    std::memcpy(&parameters.scalar_bits, &normalization, sizeof(normalization));
  } else {
    if (!weight.impl_->pre_quant_scale) {
      throw std::invalid_argument(
          "vulkan linear weight: AWQ pre-scale is not present");
    }
    resources[1] = impl_->owner->require(weight.impl_->pre_quant_scale);
    parameters.op = source->type == ScalarType::kFloat32 ? 8u : 7u;
    const uint64_t invocations = source->type == ScalarType::kFloat32
        ? elements : 1 + (elements - 1) / 2;
    impl_->owner->validate_dispatch(invocations);
    groups = static_cast<uint32_t>((invocations + 63) / 64);
  }
  try {
    impl_->count_operator();
    impl_->transition(source, BufferAccess::kComputeRead);
    if (!convrot) {
      impl_->transition(resources[1], BufferAccess::kComputeRead);
    }
    impl_->transition(destination, BufferAccess::kComputeWrite);
    impl_->dispatch_weight(parameters, groups, resources);
  } catch (...) {
    impl_->poisoned = true;
    throw;
  }
}

PreparedF16Activation::PreparedF16Activation() = default;
PreparedF16Activation::~PreparedF16Activation() = default;
PreparedF16Activation::PreparedF16Activation(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
PreparedF16Activation::PreparedF16Activation(PreparedF16Activation&&) noexcept = default;
PreparedF16Activation& PreparedF16Activation::operator=(
    PreparedF16Activation&&) noexcept = default;
PreparedF16Activation::operator bool() const noexcept { return impl_ != nullptr; }

PreparedF16Activation PreparedF16Activation::create(
    TensorContext& context, uint32_t max_rows, uint32_t in_features) {
  if (!context.impl_) {
    throw std::logic_error("vulkan gemm: moved-from tensor context");
  }
  if (max_rows == 0 || in_features == 0) {
    throw std::invalid_argument(
        "vulkan gemm: fp16 activation dimensions must be nonzero");
  }
  auto result = std::make_shared<Impl>();
  result->owner = context.impl_;
  result->max_rows = max_rows;
  result->in_features = in_features;
  const uint64_t shape[] = {max_rows, in_features};
  result->tensor = context.allocate(
      TensorLayout::contiguous(shape, 2), ScalarType::kFloat16);
  return PreparedF16Activation(std::move(result));
}

PreparedF16ActivationView PreparedF16Activation::prepare(
    TensorBatch& batch, DeviceTensor& input, uint32_t rows,
    uint32_t input_row_offset) {
  if (!impl_) throw std::logic_error("vulkan gemm: empty fp16 activation slot");
  if (!batch.impl_) throw std::logic_error("vulkan gemm: empty batch");
  if (batch.impl_->poisoned) throw std::logic_error("vulkan gemm: batch is poisoned");
  if (batch.impl_->owner.get() != impl_->owner.get()) {
    throw std::invalid_argument(
        "vulkan gemm: fp16 activation slot and batch contexts differ");
  }
  auto source = impl_->owner->require(input);
  auto destination = impl_->owner->require(impl_->tensor);
  const bool valid_range = rows != 0 && rows <= impl_->max_rows &&
      source->layout.rank == 2 && source->layout.extent[1] == impl_->in_features &&
      input_row_offset <= source->layout.extent[0] &&
      rows <= source->layout.extent[0] - input_row_offset;
  if (!valid_range || source->type != ScalarType::kFloat32 ||
      !source->layout.is_contiguous() || source.get() == destination.get()) {
    throw std::invalid_argument(
        "vulkan gemm: fp16 activation source does not match the slot");
  }
  const uint64_t elements = checked_multiply(rows, impl_->in_features,
                                             "gemm fp16 prepare");
  const uint64_t pairs = (elements + 1) / 2;
  const uint64_t groups = (pairs + 63) / 64;
  if (groups == 0 || impl_->owner->max_dispatch_x == 0) {
    throw std::out_of_range("vulkan gemm: fp16 preparation dispatch is invalid");
  }
  const uint64_t gx = std::min<uint64_t>(groups, impl_->owner->max_dispatch_x);
  const uint64_t gy = (groups + gx - 1) / gx;
  if (gy > impl_->owner->max_dispatch_y) {
    throw std::out_of_range(
        "vulkan gemm: fp16 preparation exceeds dispatch limits");
  }
  if (impl_->generation == std::numeric_limits<uint64_t>::max()) {
    throw std::overflow_error("vulkan gemm: fp16 activation generation exhausted");
  }
  TensorContext::Impl::GemmPrepareParameters parameters;
  parameters.rows = rows;
  parameters.in_features = impl_->in_features;
  parameters.input_row_offset = input_row_offset;
  parameters.groups_x = static_cast<uint32_t>(gx);
  try {
    batch.impl_->count_operator();
    batch.impl_->transition(source, BufferAccess::kComputeRead);
    batch.impl_->transition(destination, BufferAccess::kComputeWrite);
    auto& bindings = impl_->owner->gemm_prepare_bindings;
    bindings[0].buffer = &source->buffer;
    bindings[0].bytes = source->buffer.size();
    bindings[1].buffer = &destination->buffer;
    bindings[1].bytes = destination->buffer.size();
    batch.impl_->commands.bind_compute(
        impl_->owner->gemm_prepare_pipeline, bindings);
    batch.impl_->commands.push_constants(&parameters, sizeof(parameters));
    batch.impl_->commands.dispatch(static_cast<uint32_t>(gx),
                                   static_cast<uint32_t>(gy));
  } catch (...) {
    batch.impl_->poisoned = true;
    throw;
  }
  auto view = std::make_unique<PreparedF16ActivationView::Impl>();
  view->slot = impl_;
  view->batch_identity = batch.impl_.get();
  view->prepared_rows = rows;
  view->generation = ++impl_->generation;
  return PreparedF16ActivationView(std::move(view));
}

PreparedF16ActivationView::PreparedF16ActivationView() = default;
PreparedF16ActivationView::~PreparedF16ActivationView() = default;
PreparedF16ActivationView::PreparedF16ActivationView(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
PreparedF16ActivationView::PreparedF16ActivationView(
    PreparedF16ActivationView&&) noexcept = default;
PreparedF16ActivationView& PreparedF16ActivationView::operator=(
    PreparedF16ActivationView&&) noexcept = default;
uint32_t PreparedF16ActivationView::rows() const noexcept {
  return impl_ ? impl_->prepared_rows : 0;
}
PreparedF16ActivationView::operator bool() const noexcept { return impl_ != nullptr; }

DenseGemmPlan::DenseGemmPlan() = default;
DenseGemmPlan::~DenseGemmPlan() = default;
DenseGemmPlan::DenseGemmPlan(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
DenseGemmPlan::DenseGemmPlan(DenseGemmPlan&&) noexcept = default;
DenseGemmPlan& DenseGemmPlan::operator=(DenseGemmPlan&&) noexcept = default;
DenseGemmPlan::operator bool() const noexcept { return impl_ != nullptr; }

DenseGemmPlan DenseGemmPlan::create(TensorContext& context,
                                    const DenseGemmPlanDesc& desc) {
  if (!context.impl_) throw std::logic_error("vulkan gemm: moved-from context");
  switch (desc.mode) {
    case DenseGemmMode::kBFloat16:
    case DenseGemmMode::kFloat16Vae:
    case DenseGemmMode::kFloat32:
      break;
    default:
      throw std::invalid_argument("vulkan gemm: invalid execution mode");
  }
  if (desc.max_rows == 0 || desc.out_features == 0 || desc.in_features == 0) {
    throw std::invalid_argument("vulkan gemm: dimensions must be nonzero");
  }
  detail::GemmDispatchGeometry plan_geometry;
  if (!detail::gemm_dispatch_geometry(
          desc.max_rows, desc.out_features, 16, 16,
          context.impl_->max_dispatch_x, context.impl_->max_dispatch_y,
          &plan_geometry)) {
    throw std::out_of_range("vulkan gemm: plan exceeds device dispatch limits");
  }
  const uint64_t weight_elements = checked_multiply(
      desc.out_features, desc.in_features, "gemm weight");
  const uint64_t input_elements = checked_multiply(
      desc.max_rows, desc.in_features, "gemm input");
  const uint64_t output_elements = checked_multiply(
      desc.max_rows, desc.out_features, "gemm output");
  const uint64_t input_bytes = checked_multiply(
      input_elements, desc.mode == DenseGemmMode::kBFloat16 ? 2u : 4u,
      "gemm input bytes");
  const uint64_t weight_bytes = checked_multiply(
      weight_elements, desc.mode == DenseGemmMode::kFloat32 ? 4u : 2u,
      "gemm weight bytes");
  const uint64_t output_bytes = checked_multiply(
      output_elements, desc.mode == DenseGemmMode::kBFloat16 ? 2u : 4u,
      "gemm output bytes");
  if (input_bytes > context.impl_->max_storage_bytes ||
      weight_bytes > context.impl_->max_storage_bytes ||
      output_bytes > context.impl_->max_storage_bytes) {
    throw std::out_of_range("vulkan gemm: matrix exceeds storage-buffer limits");
  }
  const bool valid_bias =
      desc.bias == DenseGemmBias::kNone ||
      (desc.mode == DenseGemmMode::kBFloat16 &&
       (desc.bias == DenseGemmBias::kFloat32 ||
        desc.bias == DenseGemmBias::kBFloat16)) ||
      (desc.mode == DenseGemmMode::kFloat32 &&
       desc.bias == DenseGemmBias::kFloat32);
  if (!valid_bias) {
    throw std::invalid_argument("vulkan gemm: bias is invalid for this mode");
  }
  auto result = std::make_shared<Impl>();
  result->owner = context.impl_;
  result->desc = desc;
  return DenseGemmPlan(std::move(result));
}

const DenseGemmPlanDesc& DenseGemmPlan::description() const {
  if (!impl_) throw std::logic_error("vulkan gemm: empty plan");
  return impl_->desc;
}

void DenseGemmPlan::record(TensorBatch& batch, DeviceTensor& input,
                           DeviceTensor& prepared_weight, DeviceTensor& output,
                           uint32_t rows, uint32_t input_row_offset,
                           uint32_t output_row_offset, DeviceTensor* bias) const {
  record_impl(batch, input, prepared_weight, output, rows, input_row_offset,
              output_row_offset, bias, false);
}

void DenseGemmPlan::record(TensorBatch& batch, PreparedF16ActivationView& input,
                           DeviceTensor& prepared_weight, DeviceTensor& output,
                           uint32_t output_row_offset, DeviceTensor* bias) const {
  if (!impl_) throw std::logic_error("vulkan gemm: empty plan");
  if (!input.impl_) throw std::logic_error("vulkan gemm: empty prepared fp16 view");
  if (!batch.impl_ || input.impl_->batch_identity != batch.impl_.get()) {
    throw std::invalid_argument(
        "vulkan gemm: prepared fp16 view belongs to another batch");
  }
  if (input.impl_->generation != input.impl_->slot->generation) {
    throw std::invalid_argument("vulkan gemm: prepared fp16 view was superseded");
  }
  if (impl_->desc.mode != DenseGemmMode::kFloat16Vae ||
      input.impl_->slot->owner.get() != impl_->owner.get() ||
      input.impl_->slot->in_features != impl_->desc.in_features) {
    throw std::invalid_argument(
        "vulkan gemm: prepared fp16 view does not match the plan");
  }
  record_impl(batch, input.impl_->slot->tensor, prepared_weight, output,
              input.impl_->prepared_rows, 0, output_row_offset, bias, true);
}

void DenseGemmPlan::record_impl(
    TensorBatch& batch, DeviceTensor& input, DeviceTensor& prepared_weight,
    DeviceTensor& output, uint32_t rows, uint32_t input_row_offset,
    uint32_t output_row_offset, DeviceTensor* bias,
    bool input_is_prepared_f16) const {
  if (!impl_) throw std::logic_error("vulkan gemm: empty plan");
  if (!batch.impl_) throw std::logic_error("vulkan gemm: empty batch");
  if (batch.impl_->poisoned) throw std::logic_error("vulkan gemm: batch is poisoned");
  if (batch.impl_->owner.get() != impl_->owner.get()) {
    throw std::invalid_argument("vulkan gemm: plan and batch contexts differ");
  }
  const DenseGemmPlanDesc& desc = impl_->desc;
  if (desc.mode == DenseGemmMode::kFloat16Vae && !input_is_prepared_f16) {
    throw std::invalid_argument(
        "vulkan gemm: fp16 VAE mode requires a prepared activation view");
  }
  if (rows == 0 || rows > desc.max_rows) {
    throw std::invalid_argument("vulkan gemm: row count is outside the plan");
  }
  auto src = impl_->owner->require(input);
  auto weight = impl_->owner->require(prepared_weight);
  auto dst = impl_->owner->require(output);
  std::shared_ptr<DeviceTensor::Impl> bias_tensor;
  if (bias != nullptr) bias_tensor = impl_->owner->require(*bias);
  if ((desc.bias == DenseGemmBias::kNone) != (bias == nullptr)) {
    throw std::invalid_argument("vulkan gemm: bias presence differs from plan");
  }
  const ScalarType input_type = input_is_prepared_f16
      ? ScalarType::kFloat16
      : desc.mode == DenseGemmMode::kBFloat16
          ? ScalarType::kBFloat16 : ScalarType::kFloat32;
  const ScalarType weight_type = desc.mode == DenseGemmMode::kBFloat16
      ? ScalarType::kBFloat16 : desc.mode == DenseGemmMode::kFloat16Vae
          ? ScalarType::kFloat16 : ScalarType::kFloat32;
  const ScalarType output_type = desc.mode == DenseGemmMode::kBFloat16
      ? ScalarType::kBFloat16 : ScalarType::kFloat32;
  const ScalarType bias_type = desc.bias == DenseGemmBias::kBFloat16
      ? ScalarType::kBFloat16 : ScalarType::kFloat32;
  const bool aliases = src.get() == weight.get() || src.get() == dst.get() ||
      weight.get() == dst.get() ||
      (bias_tensor && (bias_tensor.get() == src.get() ||
                       bias_tensor.get() == weight.get() ||
                       bias_tensor.get() == dst.get()));
  const bool row_ranges_valid =
      src->layout.rank == 2 && dst->layout.rank == 2 &&
      input_row_offset <= src->layout.extent[0] &&
      rows <= src->layout.extent[0] - input_row_offset &&
      output_row_offset <= dst->layout.extent[0] &&
      rows <= dst->layout.extent[0] - output_row_offset;
  const bool bias_valid = !bias_tensor ||
      (bias_tensor->layout.rank == 1 &&
       bias_tensor->layout.extent[0] == desc.out_features &&
       bias_tensor->layout.is_contiguous() && bias_tensor->type == bias_type);
  if (aliases || !row_ranges_valid || !bias_valid ||
      src->layout.extent[1] != desc.in_features ||
      weight->layout.rank != 2 ||
      weight->layout.extent[0] != desc.out_features ||
      weight->layout.extent[1] != desc.in_features ||
      dst->layout.extent[1] != desc.out_features ||
      !src->layout.is_contiguous() || !weight->layout.is_contiguous() ||
      !dst->layout.is_contiguous() || src->type != input_type ||
      weight->type != weight_type || dst->type != output_type) {
    throw std::invalid_argument("vulkan gemm: tensors do not match the plan");
  }
  detail::GemmDispatchGeometry geometry;
  if (!detail::gemm_dispatch_geometry(
          rows, desc.out_features, 16, 16, impl_->owner->max_dispatch_x,
          impl_->owner->max_dispatch_y, &geometry)) {
    throw std::out_of_range("vulkan gemm: dispatch exceeds device limits");
  }
  uint32_t groups_x = geometry.x;
  uint32_t groups_y = geometry.y;
  TensorContext::Impl::GemmParameters parameters;
  parameters.rows = rows;
  parameters.out_features = desc.out_features;
  parameters.in_features = desc.in_features;
  parameters.input_row_offset = input_row_offset;
  parameters.output_row_offset = output_row_offset;
  if (desc.mode == DenseGemmMode::kBFloat16) {
    parameters.mode = desc.bias == DenseGemmBias::kNone ? 0u
        : desc.bias == DenseGemmBias::kFloat32 ? 1u : 2u;
  } else if (desc.mode == DenseGemmMode::kFloat16Vae) {
    parameters.mode = 3u;
    parameters.input_row_offset = 0;
  } else {
    parameters.mode = desc.bias == DenseGemmBias::kNone ? 4u : 5u;
  }
  const bool cooperative_shape = (rows % 64u) == 0u &&
      (desc.out_features % 16u) == 0u && (desc.in_features % 16u) == 0u;
  const bool use_cooperative = cooperative_shape &&
      ((desc.mode == DenseGemmMode::kBFloat16 && impl_->owner->cooperative_gemm) ||
       (desc.mode == DenseGemmMode::kFloat16Vae &&
        impl_->owner->cooperative_f16_gemm));
  if (use_cooperative) {
    if (!detail::gemm_dispatch_geometry(
            rows, desc.out_features, 64, 16, impl_->owner->max_dispatch_x,
            impl_->owner->max_dispatch_y, &geometry)) {
      throw std::out_of_range("vulkan gemm: cooperative dispatch exceeds device limits");
    }
    groups_x = geometry.x;
    groups_y = geometry.y;
  }
  try {
    batch.impl_->count_operator();
    batch.impl_->transition(src, BufferAccess::kComputeRead);
    batch.impl_->transition(weight, BufferAccess::kComputeRead);
    if (bias_tensor) batch.impl_->transition(bias_tensor, BufferAccess::kComputeRead);
    batch.impl_->transition(dst, BufferAccess::kComputeWrite);
    auto& bindings = impl_->owner->gemm_bindings;
    bindings[0].buffer = &src->buffer;
    bindings[0].bytes = src->buffer.size();
    bindings[1].buffer = &weight->buffer;
    bindings[1].bytes = weight->buffer.size();
    bindings[2].buffer = bias_tensor ? &bias_tensor->buffer : &src->buffer;
    bindings[2].bytes = bias_tensor ? bias_tensor->buffer.size()
                                    : src->buffer.size();
    bindings[3].buffer = &dst->buffer;
    bindings[3].bytes = dst->buffer.size();
    batch.impl_->commands.bind_compute(
        use_cooperative
            ? (desc.mode == DenseGemmMode::kFloat16Vae
                   ? impl_->owner->gemm_coop_f16_pipeline
                   : impl_->owner->gemm_coop_pipeline)
            : impl_->owner->gemm_pipeline,
        bindings);
    batch.impl_->commands.push_constants(&parameters, sizeof(parameters));
    batch.impl_->commands.dispatch(groups_x, groups_y);
  } catch (...) {
    batch.impl_->poisoned = true;
    throw;
  }
}

Submission TensorBatch::submit() {
  if (!impl_ || impl_->operator_count == 0) {
    throw std::logic_error("vulkan tensor: cannot submit an empty batch");
  }
  if (impl_->poisoned) {
    throw std::logic_error("vulkan tensor: cannot submit a poisoned batch");
  }
  Submission result = impl_->owner->commands.submit(std::move(impl_->commands));
  impl_->submitted = true;
  impl_->recording_lease.release();
  impl_.reset();
  return result;
}

}  // namespace vidfab::vulkan
