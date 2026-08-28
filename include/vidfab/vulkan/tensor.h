#pragma once

#include <cstdint>
#include <memory>

#include "vidfab/device_tensor.h"
#include "vidfab/vulkan/compute.h"
#include "vidfab/vulkan/runtime.h"

namespace vidfab::vulkan {

class TensorContext;
class TensorBatch;

struct TensorContextOptions {
  // Two slots let a producer record the next bounded graph chunk while the
  // previous timeline submission is still executing.
  uint32_t max_in_flight = 2;
};

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
  ScalarType type() const;
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
// allocations are retained as they are recorded and then through completion.
// Dropping an unsubmitted batch discards commands and restores tensor access
// tracking.
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
  void convert(DeviceTensor& source, DeviceTensor& destination);
  void transpose_2d(DeviceTensor& source, DeviceTensor& destination);
  // `indices` is a trusted device tensor: every int32 value must be in
  // [0, source.rows). The shader bounds-checks and writes zero for an invalid
  // value to prevent device memory access, but validation belongs at the
  // producer/callsite because checking device values here would add a host
  // synchronization boundary.
  void gather_rows(DeviceTensor& source, DeviceTensor& indices,
                   DeviceTensor& destination);
  // `indices` is trusted: values must be unique and in [0, destination.rows).
  // The shader bounds-checks invalid values and leaves those rows untouched;
  // uniqueness must be guaranteed by the producer. Untouched destination rows
  // are preserved.
  void scatter_rows(DeviceTensor& source, DeviceTensor& indices,
                    DeviceTensor& destination);
  void add_bias(DeviceTensor& input, DeviceTensor& bias, DeviceTensor& output);
  void heads_to_tokens_bf16(DeviceTensor& source, DeviceTensor& destination,
                            uint32_t heads, uint32_t sequence, uint32_t head_dim);
  void depth_to_space(DeviceTensor& source, DeviceTensor& destination,
                      uint32_t time, uint32_t height, uint32_t width,
                      uint32_t channels, uint32_t patch_time, uint32_t patch);
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
// and do not stage through host memory. One TensorContext instance permits one
// active CPU recorder/boundary operation at a time; calls from different host
// threads require external synchronization. Submitted jobs use the configured
// bounded flight slots independently of that recorder lease.
class TensorContext {
 public:
  explicit TensorContext(const Device& device,
                         const TensorContextOptions& options = {});
  ~TensorContext();
  TensorContext(TensorContext&&) noexcept;
  TensorContext& operator=(TensorContext&&) noexcept;
  TensorContext(const TensorContext&) = delete;
  TensorContext& operator=(const TensorContext&) = delete;

  DeviceTensor allocate(const TensorLayout& layout,
                        ScalarType type = ScalarType::kFloat32);
  TensorBatch begin_batch();
  void upload(DeviceTensor& destination, const float* values, uint64_t count);
  void download(DeviceTensor& source, float* values, uint64_t count);
  void upload_bytes(DeviceTensor& destination, const void* values, uint64_t bytes);
  void download_bytes(DeviceTensor& source, void* values, uint64_t bytes);
  // Exact self-copy and partial aliasing are rejected.
  void copy(DeviceTensor& source, DeviceTensor& destination);
  // Inputs may alias each other; output must be a distinct allocation.
  // Results are CUDA-bit-exact when inputs and the correctly rounded result
  // are zero, normal, or infinity. NaN payload arithmetic is not promised.
  void add(DeviceTensor& a, DeviceTensor& b, DeviceTensor& output);
  // True only when add additionally covers subnormal inputs/results.
  bool full_fp32_add_exactness() const noexcept;
  void require_full_fp32_add_exactness() const;
  TensorWorkspace& workspace();
  uint64_t reserved_bytes() const;
  uint64_t descriptor_set_allocations() const noexcept;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
  friend class TensorBatch;
};

}  // namespace vidfab::vulkan
