#pragma once

#include <cstdint>
#include <memory>

#include "vidfab/device_tensor.h"
#include "vidfab/vulkan/compute.h"
#include "vidfab/vulkan/runtime.h"

namespace vidfab::vulkan {

class TensorContext;
class TensorBatch;

class DeviceTensor {
 public:
  DeviceTensor();
  ~DeviceTensor();
  DeviceTensor(DeviceTensor&&) noexcept;
  DeviceTensor& operator=(DeviceTensor&&) noexcept;
  DeviceTensor(const DeviceTensor&) = delete;
  DeviceTensor& operator=(const DeviceTensor&) = delete;

  DeviceTensorView view() const;
  const TensorLayout& layout() const;
  explicit operator bool() const noexcept;

 private:
  struct Impl;
  explicit DeviceTensor(std::unique_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
  friend class TensorContext;
  friend class TensorBatch;
};

class TensorWorkspace final : public DeviceWorkspace {
 public:
  explicit TensorWorkspace(const Device& device, uint64_t block_bytes = 4ull << 20);
  ~TensorWorkspace() override;
  TensorWorkspace(TensorWorkspace&&) noexcept;
  TensorWorkspace& operator=(TensorWorkspace&&) noexcept;
  TensorWorkspace(const TensorWorkspace&) = delete;
  TensorWorkspace& operator=(const TensorWorkspace&) = delete;

  DeviceBackend backend() const noexcept override;
  void reserve(uint64_t bytes) override;
  WorkspaceSpan allocate(uint64_t bytes, uint64_t alignment = 256) override;
  void reset() noexcept override;
  uint64_t capacity() const noexcept override;
  uint64_t used() const noexcept override;
  uint64_t generation() const noexcept override;
  bool valid(const WorkspaceSpan& span) const noexcept override;
  uint64_t reserved_bytes() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// A bounded device recording scope. Multiple operators record into one Vulkan
// command buffer and `submit` produces one exact timeline token. Tensor
// wrappers passed to it must remain alive until submit; submitted buffers are
// retained internally through completion. Dropping an unsubmitted batch
// discards commands and restores tensor access tracking.
class TensorBatch {
 public:
  TensorBatch();
  ~TensorBatch();
  TensorBatch(TensorBatch&&) noexcept;
  TensorBatch& operator=(TensorBatch&&) noexcept;
  TensorBatch(const TensorBatch&) = delete;
  TensorBatch& operator=(const TensorBatch&) = delete;

  void copy(DeviceTensor& source, DeviceTensor& destination);
  void add(DeviceTensor& a, DeviceTensor& b, DeviceTensor& output);
  Submission submit();
  explicit operator bool() const noexcept;

 private:
  struct Impl;
  explicit TensorBatch(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  friend class TensorContext;
};

// Persistent Vulkan tensor allocation, staging and elementary operators.
// Upload/download are explicit boundaries; copy/add operate device-to-device
// and do not stage through host memory.
class TensorContext {
 public:
  explicit TensorContext(const Device& device);
  ~TensorContext();
  TensorContext(TensorContext&&) noexcept;
  TensorContext& operator=(TensorContext&&) noexcept;
  TensorContext(const TensorContext&) = delete;
  TensorContext& operator=(const TensorContext&) = delete;

  DeviceTensor allocate(const TensorLayout& layout);
  TensorBatch begin_batch();
  void upload(DeviceTensor& destination, const float* values, uint64_t count);
  void download(DeviceTensor& source, float* values, uint64_t count);
  // Exact self-copy and partial aliasing are rejected.
  void copy(DeviceTensor& source, DeviceTensor& destination);
  // Inputs may alias each other; output must be a distinct allocation.
  void add(DeviceTensor& a, DeviceTensor& b, DeviceTensor& output);
  TensorWorkspace& workspace();
  uint64_t reserved_bytes() const;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
  friend class TensorBatch;
};

}  // namespace vidfab::vulkan
