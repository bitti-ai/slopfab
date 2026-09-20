#include "tensor_recording.h"

namespace slopfab::vulkan {

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
  batch->snapshots.resize(static_cast<size_t>(impl_->max_batch_operators) * 9u);
  batch->batch_id = next_context_identity();
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

void TensorContext::upload_transient(DeviceTensor& destination,
                                     const float* values, uint64_t count) {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  auto dst = impl_->require(destination);
  if (dst->type != ScalarType::kFloat32 || values == nullptr ||
      count != dst->layout.elements()) {
    throw std::invalid_argument("vulkan tensor: upload element count mismatch");
  }
  upload_transient_bytes(destination, values, count * sizeof(float));
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
  if (physical_bytes != bytes)
    write_zero_bytes(impl_->upload_buffer, bytes, physical_bytes - bytes);
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

void TensorContext::upload_transient_bytes(DeviceTensor& destination,
                                           const void* values,
                                           uint64_t bytes) {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  [[maybe_unused]] auto recording_lock = impl_->acquire_recorder();
  auto dst = impl_->require(destination);
  if (values == nullptr || bytes != dst->logical_bytes) {
    throw std::invalid_argument("vulkan tensor: upload byte count mismatch");
  }
  const uint64_t physical_bytes = dst->buffer.size();
  Buffer staging = impl_->pool.allocate(
      physical_bytes, BufferUsage::kTransferSource, MemoryUsage::kUpload);
  staging.write(0, values, bytes);
  if (physical_bytes != bytes)
    write_zero_bytes(staging, bytes, physical_bytes - bytes);
  CommandList list = impl_->commands.begin();
  list.barrier(staging, BufferAccess::kHostWrite,
               BufferAccess::kTransferRead, 0, physical_bytes);
  const bool old_has_access = dst->has_access;
  const BufferAccess old_access = dst->access;
  try {
    if (dst->has_access)
      list.barrier(dst->buffer, dst->access, BufferAccess::kTransferWrite);
    dst->has_access = true;
    dst->access = BufferAccess::kTransferWrite;
    list.copy_buffer(staging, dst->buffer, physical_bytes);
    impl_->complete(std::move(list));
  } catch (...) {
    dst->has_access = old_has_access;
    dst->access = old_access;
    throw;
  }
  staging = Buffer();
  impl_->pool.trim();
}

void TensorContext::upload_batch(const TensorUpload* uploads, uint32_t count) {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  if (!uploads || count == 0)
    throw std::invalid_argument("vulkan tensor: invalid upload batch");
  struct Pending {
    std::shared_ptr<DeviceTensor::Impl> tensor;
    const void* values = nullptr;
    uint64_t bytes = 0;
    uint64_t physical = 0;
    uint64_t offset = 0;
    bool old_has_access = false;
    BufferAccess old_access = BufferAccess::kTransferWrite;
  };
  std::vector<Pending> pending;
  pending.reserve(count);
  uint64_t total = 0;
  for (uint32_t i = 0; i < count; ++i) {
    if (!uploads[i].destination || !uploads[i].values)
      throw std::invalid_argument("vulkan tensor: invalid upload batch item");
    auto tensor = impl_->require(*uploads[i].destination);
    for (const Pending& prior : pending)
      if (prior.tensor.get() == tensor.get())
        throw std::invalid_argument("vulkan tensor: duplicate upload destination");
    if (uploads[i].bytes != tensor->logical_bytes)
      throw std::invalid_argument("vulkan tensor: upload batch byte count mismatch");
    const uint64_t physical = tensor->buffer.size();
    if (total > std::numeric_limits<uint64_t>::max() - physical)
      throw std::overflow_error("vulkan tensor: upload batch size overflow");
    pending.push_back({tensor, uploads[i].values, uploads[i].bytes, physical,
                       total, tensor->has_access, tensor->access});
    total += physical;
  }
  [[maybe_unused]] auto recording_lock = impl_->acquire_recorder();
  impl_->ensure_staging(total);
  for (const Pending& item : pending) {
    impl_->upload_buffer.write(item.offset, item.values, item.bytes);
    if (item.physical != item.bytes)
      write_zero_bytes(impl_->upload_buffer, item.offset + item.bytes,
                       item.physical - item.bytes);
  }
  CommandList list = impl_->commands.begin();
  list.barrier(impl_->upload_buffer, BufferAccess::kHostWrite,
               BufferAccess::kTransferRead, 0, total);
  try {
    for (Pending& item : pending) {
      if (item.tensor->has_access)
        list.barrier(item.tensor->buffer, item.tensor->access,
                     BufferAccess::kTransferWrite);
      item.tensor->has_access = true;
      item.tensor->access = BufferAccess::kTransferWrite;
      list.copy_buffer(impl_->upload_buffer, item.tensor->buffer,
                       item.physical, item.offset, 0);
    }
    impl_->complete(std::move(list));
  } catch (...) {
    for (Pending& item : pending) {
      item.tensor->has_access = item.old_has_access;
      item.tensor->access = item.old_access;
    }
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
bool TensorContext::exact_vae_pointwise() const noexcept {
  return impl_ && impl_->exact_vae_pointwise;
}
void TensorContext::require_exact_vae_pointwise() const {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  if (!impl_->exact_vae_pointwise) {
    throw std::runtime_error(
        "vulkan tensor: exact VAE pointwise operations are unavailable on this device/driver");
  }
}
bool TensorContext::exact_audio_vae_primitives() const noexcept {
  return impl_ && impl_->exact_audio;
}
void TensorContext::require_exact_audio_vae_primitives() const {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  if (!impl_->exact_audio) {
    throw std::runtime_error(
        "vulkan audio: exact primitives are unavailable on this device/driver");
  }
}
bool TensorContext::exact_blocked_attention() const noexcept {
  return impl_ && impl_->exact_attention;
}
void TensorContext::require_exact_blocked_attention() const {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  if (!impl_->exact_attention) {
    throw std::runtime_error(
        "vulkan attention: exact blocked attention is unavailable on this device/driver");
  }
}
bool TensorContext::exact_h3_attention() const noexcept {
  return impl_ && impl_->exact_h3_attention;
}
bool TensorContext::h3_attention_supported(AttentionMode mode) const noexcept {
  if (!impl_) return false;
  switch (mode) {
    case AttentionMode::kExact: return impl_->exact_h3_attention;
    case AttentionMode::kFlash2: return impl_->flash_attention;
    case AttentionMode::kSage2: return impl_->sage_attention;
    default: return false;
  }
}
void TensorContext::require_h3_attention(AttentionMode mode) const {
  if (!h3_attention_supported(mode))
    throw std::runtime_error(std::string("vulkan H3 attention: mode '") +
        attention_mode_name(mode) + "' is unavailable; check enabled cooperative matrix, "
        "subgroup and arithmetic features");
}
void TensorContext::require_exact_h3_attention() const {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  if (!impl_->exact_h3_attention) {
    throw std::runtime_error(
        "vulkan attention: exact H3 attention is unavailable on this device/driver");
  }
}
bool TensorContext::exact_causal_gqa_attention() const noexcept {
  return impl_ && impl_->exact_causal_gqa_attention;
}
void TensorContext::require_exact_causal_gqa_attention() const {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  if (!impl_->exact_causal_gqa_attention) {
    throw std::runtime_error(
        "vulkan attention: exact causal GQA attention is unavailable on this device/driver");
  }
}

bool TensorContext::native_nvfp4_gemm_available() const noexcept {
  return false;
}
void TensorContext::require_native_nvfp4_gemm() const {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  throw std::runtime_error(
      "vulkan tensor: native NVFP4 GEMM is unavailable; this device/runtime "
      "does not expose an implemented E2M1 block-scaled cooperative-matrix "
      "contract (use streamed NVFP4-to-BF16 execution)");
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
bool TensorContext::owns(const DeviceTensor& tensor) const noexcept {
  return impl_ && tensor.impl_ && tensor.impl_->context == impl_->context_id;
}
uint64_t TensorContext::staging_capacity_bytes() const noexcept {
  return impl_ ? impl_->staging_capacity : 0;
}
uint64_t TensorContext::descriptor_set_allocations() const noexcept {
  return impl_ ? impl_->commands.descriptor_set_allocations() : 0;
}

void TensorContext::collect() {
  if (!impl_) throw std::logic_error("vulkan tensor: moved-from context");
  impl_->commands.collect();
}
uint64_t TensorContext::storage_binding_alignment() const noexcept {
  return impl_ ? impl_->storage_binding_alignment : 1;
}

TensorBatch::TensorBatch() = default;
TensorBatch::~TensorBatch() = default;
TensorBatch::TensorBatch(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
TensorBatch::TensorBatch(TensorBatch&&) noexcept = default;
TensorBatch& TensorBatch::operator=(TensorBatch&&) noexcept = default;
TensorBatch::operator bool() const noexcept { return impl_ != nullptr; }

uint32_t TensorBatch::remaining_operator_capacity() const noexcept {
  if (!impl_ || !impl_->owner ||
      impl_->operator_count >= impl_->owner->max_batch_operators) return 0;
  return impl_->owner->max_batch_operators - impl_->operator_count;
}

void TensorBatch::require_operator_capacity(uint32_t operators) const {
  if (!impl_ || impl_->poisoned)
    throw std::logic_error("vulkan tensor: invalid batch");
  if (operators == 0 || remaining_operator_capacity() < operators)
    throw std::logic_error("vulkan tensor: insufficient operator capacity");
}

bool TensorBatch::belongs_to(const TensorContext& context) const noexcept {
  return impl_ && context.impl_ && impl_->owner == context.impl_;
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


}  // namespace slopfab::vulkan
