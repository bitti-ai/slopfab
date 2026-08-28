// Reusable, timeline-backed Vulkan compute submission.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vidfab/vulkan/runtime.h"

namespace vidfab::vulkan {

struct ComputePipelineOptions {
  uint32_t storage_binding_count = 0;
  uint32_t push_constant_bytes = 0;
  uint32_t local_size[3] = {1, 1, 1};
  std::string entry_point = "main";
};

class ComputePipeline {
 public:
  ComputePipeline();
  ~ComputePipeline();
  ComputePipeline(ComputePipeline&&) noexcept;
  ComputePipeline& operator=(ComputePipeline&&) noexcept;
  ComputePipeline(const ComputePipeline&) = delete;
  ComputePipeline& operator=(const ComputePipeline&) = delete;

  static ComputePipeline create(const Device& device,
                                const std::vector<uint32_t>& spirv,
                                const ComputePipelineOptions& options);
  uint32_t storage_binding_count() const noexcept;
  uint32_t push_constant_bytes() const noexcept;
  explicit operator bool() const noexcept;

 private:
  struct Impl;
  explicit ComputePipeline(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
  friend class CommandList;
};

enum class BufferAccess {
  kHostWrite,
  kHostRead,
  kTransferRead,
  kTransferWrite,
  kComputeRead,
  kComputeWrite,
  kComputeReadWrite,
};

struct StorageBinding {
  uint32_t binding = 0;
  Buffer* buffer = nullptr;
  uint64_t offset = 0;
  uint64_t bytes = ~uint64_t{0};
};

struct ComputeContextOptions {
  uint32_t max_in_flight = 4;
  uint32_t max_storage_bindings = 8;
  uint32_t max_compute_binds_per_job = 1;
};

class Submission {
 public:
  Submission();
  uint64_t value() const noexcept;
  bool ready() const;
  void wait() const;
  explicit operator bool() const noexcept;

 private:
  struct Impl;
  explicit Submission(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
  friend class ComputeContext;
};

class CommandList {
 public:
  CommandList();
  ~CommandList();
  CommandList(CommandList&&) noexcept;
  CommandList& operator=(CommandList&&) noexcept;
  CommandList(const CommandList&) = delete;
  CommandList& operator=(const CommandList&) = delete;

  void copy_buffer(Buffer& source, Buffer& destination, uint64_t bytes,
                   uint64_t source_offset = 0, uint64_t destination_offset = 0);
  void barrier(Buffer& buffer, BufferAccess before, BufferAccess after,
               uint64_t offset = 0, uint64_t bytes = ~uint64_t{0});
  void bind_compute(ComputePipeline& pipeline,
                    const std::vector<StorageBinding>& bindings);
  void push_constants(const void* data, uint32_t bytes);
  void dispatch(uint32_t groups_x, uint32_t groups_y = 1, uint32_t groups_z = 1);
  explicit operator bool() const noexcept;

 private:
  struct Impl;
  explicit CommandList(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  friend class ComputeContext;
};

class ComputeContext {
 public:
  explicit ComputeContext(const Device& device, const ComputeContextOptions& options = {});
  ~ComputeContext();
  ComputeContext(ComputeContext&&) noexcept;
  ComputeContext& operator=(ComputeContext&&) noexcept;
  ComputeContext(const ComputeContext&) = delete;
  ComputeContext& operator=(const ComputeContext&) = delete;

  CommandList begin();
  Submission submit(CommandList&& commands);
  // Reclaims completed slots without waiting. Called automatically by begin.
  void collect();
  uint32_t in_flight() const;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace vidfab::vulkan
