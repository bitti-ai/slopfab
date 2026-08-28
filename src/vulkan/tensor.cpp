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

namespace vidfab::vulkan {
namespace {

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
  static constexpr uint32_t kMaxBatchOperators = 32;

  ComputeContext commands;
  BufferPool pool;
  TensorWorkspace scratch;
  ComputePipeline ops_pipeline;
  ComputePipeline rms_norm_pipeline;
  ComputePipeline layer_norm_pipeline;
  ComputePipeline bf16_rms_block_pipeline;
  ComputePipeline bf16_rms_narrow_pipeline;
  ComputePipeline bf16_layer_pipeline;
  ComputePipeline bf16_mod_pipeline;
  ComputePipeline fp32_mod_pipeline;
  Buffer upload_buffer;
  Buffer readback_buffer;
  uint64_t staging_capacity = 0;
  std::vector<StorageBinding> ops_bindings;
  std::vector<StorageBinding> norm_bindings;
  std::vector<StorageBinding> mod_bindings;
  bool full_arithmetic_exact = false;
  bool exact_vae_norm = false;
  uint32_t max_dispatch_x = 0;
  uint64_t max_storage_bytes = 0;
  std::atomic<bool> recorder_active{false};

  explicit Impl(const Device& input, const TensorContextOptions& tensor_options)
      : commands(input, [&] {
          ComputeContextOptions options;
          options.max_in_flight = tensor_options.max_in_flight;
          options.max_storage_bindings = 6;
          options.max_compute_binds_per_job = kMaxBatchOperators;
          return options;
        }()),
        pool(input, 4ull << 20),
        scratch(input),
        ops_bindings(3),
        norm_bindings(4),
        mod_bindings(6) {
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
                     input.info().fp32_signed_zero_inf_nan_preserve;
    max_dispatch_x = input.info().max_compute_workgroup_count[0];
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
    ComputePipelineOptions norm_options;
    norm_options.storage_binding_count = 4;
    norm_options.push_constant_bytes = sizeof(NormParameters);
    norm_options.local_size[0] = 256;
    auto make_norm_pipeline = [&](const uint8_t* shader, size_t shader_bytes,
                                  uint32_t bindings = 4, uint32_t local_x = 256,
                                  uint32_t local_y = 1) {
      std::vector<uint32_t> module(shader_bytes / sizeof(uint32_t));
      std::memcpy(module.data(), shader, shader_bytes);
      ComputePipelineOptions selected = norm_options;
      selected.storage_binding_count = bindings;
      selected.local_size[0] = local_x;
      selected.local_size[1] = local_y;
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
    }
    for (uint32_t i = 0; i < ops_bindings.size(); ++i) ops_bindings[i].binding = i;
    for (uint32_t i = 0; i < norm_bindings.size(); ++i) norm_bindings[i].binding = i;
    for (uint32_t i = 0; i < mod_bindings.size(); ++i) mod_bindings[i].binding = i;
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
  std::array<AccessSnapshot, TensorContext::Impl::kMaxBatchOperators * 6> snapshots{};
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
        dim > std::numeric_limits<uint32_t>::max() ||
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
bool TensorContext::exact_fp32_vae_normalization() const noexcept {
  return impl_ && impl_->exact_vae_norm;
}
void TensorContext::require_exact_fp32_vae_normalization() const {
  if (!exact_fp32_vae_normalization()) {
    throw std::runtime_error(
        "vulkan tensor: exact fp32 VAE normalization requires a compatible "
        "NVIDIA Vulkan device");
  }
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
      shape.extent[1] > std::numeric_limits<uint32_t>::max() ||
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
      shape.extent[1] > std::numeric_limits<uint32_t>::max() ||
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
      dim > std::numeric_limits<uint32_t>::max() ||
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
      dim > std::numeric_limits<uint32_t>::max() ||
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
