#include "vidfab/vulkan/tensor.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "embedded_tensor_spv.h"
#include "vidfab/vulkan/compute.h"

namespace vidfab::vulkan {
namespace {

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
  struct Parameters { uint32_t count; };
  static constexpr uint32_t kMaxBatchOperators = 32;

  ComputeContext commands;
  BufferPool pool;
  TensorWorkspace scratch;
  ComputePipeline add_pipeline;
  Buffer upload_buffer;
  Buffer readback_buffer;
  uint64_t staging_capacity = 0;
  std::vector<StorageBinding> bindings;
  bool full_add_exact = false;
  uint32_t max_dispatch_x = 0;
  uint64_t max_storage_bytes = 0;

  explicit Impl(const Device& input, const TensorContextOptions& tensor_options)
      : commands(input, [&] {
          ComputeContextOptions options;
          options.max_in_flight = tensor_options.max_in_flight;
          options.max_storage_bindings = 3;
          options.max_compute_binds_per_job = kMaxBatchOperators;
          return options;
        }()),
        pool(input, 4ull << 20),
        scratch(input),
        bindings(3) {
    // Device is move-only; the opaque handle is sufficient for identity and
    // every owned Vulkan object already retains the shared device state.
    static_assert(sizeof(detail::kTensorAddSpirv) % sizeof(uint32_t) == 0);
    if (!input.info().fp32_signed_zero_inf_nan_preserve ||
        !input.info().fp32_rounding_rte) {
      throw std::runtime_error(
          "vulkan tensor: fp32 add requires signed-zero/Inf/NaN preservation "
          "and round-to-nearest-even");
    }
    full_add_exact = input.info().fp32_denorm_preserve;
    max_dispatch_x = input.info().max_compute_workgroup_count[0];
    max_storage_bytes = input.info().max_storage_buffer_bytes;
    const uint8_t* shader = full_add_exact ? detail::kTensorAddDenormSpirv
                                           : detail::kTensorAddSpirv;
    const size_t shader_bytes = full_add_exact ? sizeof(detail::kTensorAddDenormSpirv)
                                               : sizeof(detail::kTensorAddSpirv);
    std::vector<uint32_t> spirv(shader_bytes / sizeof(uint32_t));
    std::memcpy(spirv.data(), shader, shader_bytes);
    ComputePipelineOptions options;
    options.storage_binding_count = 3;
    options.push_constant_bytes = sizeof(Parameters);
    options.local_size[0] = 64;
    add_pipeline = ComputePipeline::create(input, spirv, options);
    for (uint32_t i = 0; i < bindings.size(); ++i) bindings[i].binding = i;
  }

  uintptr_t context_id = next_context_identity();

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
};

struct TensorBatch::Impl {
  struct AccessSnapshot {
    std::shared_ptr<DeviceTensor::Impl> tensor;
    bool has_access = false;
    BufferAccess access = BufferAccess::kTransferWrite;
  };

  std::shared_ptr<TensorContext::Impl> owner;
  CommandList commands;
  std::array<AccessSnapshot, TensorContext::Impl::kMaxBatchOperators * 3> snapshots{};
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
  result.type = ScalarType::kFloat32;
  result.layout = impl_->layout;
  result.context = impl_->context;
  result.resource = impl_->buffer.native_handle();
  result.byte_size = impl_->buffer.size();
  return result;
}
const TensorLayout& DeviceTensor::layout() const {
  if (!impl_) throw std::logic_error("vulkan tensor: empty tensor");
  return impl_->layout;
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

DeviceTensor TensorContext::allocate(const TensorLayout& layout) {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  if (!layout.is_contiguous()) throw std::invalid_argument("vulkan tensor: contiguous layout required");
  const uint64_t bytes = layout.bytes(ScalarType::kFloat32);
  if (bytes > impl_->max_storage_bytes) {
    throw std::out_of_range("vulkan tensor: tensor exceeds storage buffer limit");
  }
  auto tensor = std::make_unique<DeviceTensor::Impl>();
  tensor->buffer = impl_->pool.allocate(
      bytes, BufferUsage::kStorage | BufferUsage::kTransferSource |
                 BufferUsage::kTransferDestination,
      MemoryUsage::kDevice);
  tensor->layout = layout;
  tensor->context = impl_->context_id;
  return DeviceTensor(std::move(tensor));
}

TensorBatch TensorContext::begin_batch() {
  if (!impl_) throw std::logic_error("vulkan tensor: empty context");
  auto batch = std::make_unique<TensorBatch::Impl>();
  batch->owner = impl_;
  batch->commands = impl_->commands.begin();
  return TensorBatch(std::move(batch));
}

void TensorContext::upload(DeviceTensor& destination, const float* values, uint64_t count) {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  auto dst = impl_->require(destination);
  if (values == nullptr || count != dst->layout.elements()) {
    throw std::invalid_argument("vulkan tensor: upload element count mismatch");
  }
  const uint64_t bytes = count * sizeof(float);
  impl_->ensure_staging(bytes);
  impl_->upload_buffer.write(0, values, bytes);
  CommandList list = impl_->commands.begin();
  list.barrier(impl_->upload_buffer, BufferAccess::kHostWrite, BufferAccess::kTransferRead, 0, bytes);
  const bool old_has_access = dst->has_access;
  const BufferAccess old_access = dst->access;
  try {
    if (dst->has_access) {
      list.barrier(dst->buffer, dst->access, BufferAccess::kTransferWrite);
    }
    dst->has_access = true;
    dst->access = BufferAccess::kTransferWrite;
    list.copy_buffer(impl_->upload_buffer, dst->buffer, bytes);
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
  if (values == nullptr || count != src->layout.elements()) {
    throw std::invalid_argument("vulkan tensor: download element count mismatch");
  }
  const uint64_t bytes = count * sizeof(float);
  impl_->ensure_staging(bytes);
  CommandList list = impl_->commands.begin();
  const bool old_has_access = src->has_access;
  const BufferAccess old_access = src->access;
  try {
    if (src->has_access && src->access != BufferAccess::kTransferRead) {
      list.barrier(src->buffer, src->access, BufferAccess::kTransferRead);
    }
    src->has_access = true;
    src->access = BufferAccess::kTransferRead;
    list.copy_buffer(src->buffer, impl_->readback_buffer, bytes);
    list.barrier(impl_->readback_buffer, BufferAccess::kTransferWrite,
                 BufferAccess::kHostRead, 0, bytes);
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

bool TensorContext::full_fp32_add_exactness() const noexcept {
  return impl_ && impl_->full_add_exact;
}
void TensorContext::require_full_fp32_add_exactness() const {
  if (!full_fp32_add_exactness()) {
    throw std::runtime_error(
        "vulkan tensor: device cannot preserve fp32 subnormal inputs/results; "
        "full CUDA-exact fp32 add is unavailable");
  }
}

TensorWorkspace& TensorContext::workspace() {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  return impl_->scratch;
}
uint64_t TensorContext::reserved_bytes() const {
  if (!impl_) return 0;
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
  const uint64_t bytes = src->layout.bytes(ScalarType::kFloat32);
  if (src.get() == dst.get() || bytes != dst->layout.bytes(ScalarType::kFloat32)) {
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
      count != bv->layout.elements() || count != out->layout.elements() ||
      count == 0 || count > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan tensor: add needs distinct equal-sized fp32 tensors");
  }
  const uint64_t bytes = count * sizeof(float);
  const uint32_t groups = static_cast<uint32_t>((count + 63) / 64);
  if (bytes > impl_->owner->max_storage_bytes || groups == 0 ||
      groups > impl_->owner->max_dispatch_x) {
    throw std::out_of_range("vulkan tensor: add exceeds device dispatch limits");
  }
  try {
    impl_->count_operator();
    impl_->transition(av, BufferAccess::kComputeRead);
    impl_->transition(bv, BufferAccess::kComputeRead);
    impl_->transition(out, BufferAccess::kComputeWrite);
    impl_->owner->bindings[0].buffer = &av->buffer;
    impl_->owner->bindings[1].buffer = &bv->buffer;
    impl_->owner->bindings[2].buffer = &out->buffer;
    for (StorageBinding& binding : impl_->owner->bindings) binding.bytes = bytes;
    impl_->commands.bind_compute(impl_->owner->add_pipeline, impl_->owner->bindings);
    const TensorContext::Impl::Parameters parameters{static_cast<uint32_t>(count)};
    impl_->commands.push_constants(&parameters, sizeof(parameters));
    impl_->commands.dispatch(groups);
  } catch (...) {
    impl_->poisoned = true;
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
  impl_.reset();
  return result;
}

}  // namespace vidfab::vulkan
