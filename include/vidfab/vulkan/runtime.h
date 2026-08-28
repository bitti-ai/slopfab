// Standalone Vulkan runtime foundation.
//
// This public header deliberately contains no Vulkan SDK types. vidfab loads
// the system Vulkan loader at runtime, and the implementation uses a pinned
// copy of the Khronos ABI header. Consumers therefore need neither an SDK nor
// Vulkan headers in their own build.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vidfab::vulkan {

struct Version {
  uint32_t major = 0;
  uint32_t minor = 0;
  uint32_t patch = 0;
};

struct MemoryHeapInfo {
  uint64_t bytes = 0;
  bool device_local = false;
};

struct DeviceInfo {
  std::string name;
  uint32_t vendor_id = 0;
  uint32_t device_id = 0;
  uint32_t driver_version = 0;
  Version api_version;
  bool discrete = false;
  uint32_t compute_queue_family = 0;
  uint32_t compute_queue_count = 0;
  uint32_t max_compute_workgroup_invocations = 0;
  uint32_t max_compute_workgroup_count[3] = {};
  uint32_t max_compute_workgroup_size[3] = {};
  uint32_t max_push_constant_bytes = 0;
  uint64_t max_storage_buffer_bytes = 0;
  uint64_t max_allocation_bytes = 0;
  uint64_t non_coherent_atom_bytes = 1;
  bool shader_float16 = false;
  bool shader_int8 = false;
  bool storage_buffer_16bit = false;
  bool storage_buffer_8bit = false;
  bool timeline_semaphore = false;
  bool buffer_device_address = false;
  bool descriptor_indexing = false;
  bool runtime_descriptor_array = false;
  bool descriptor_binding_partially_bound = false;
  bool descriptor_binding_variable_count = false;
  bool storage_buffer_non_uniform_indexing = false;
  std::vector<MemoryHeapInfo> memory_heaps;
  std::vector<std::string> extensions;

  bool supports_extension(const std::string& name) const;
  uint64_t device_local_bytes() const;
};

struct InstanceOptions {
  std::string application_name = "vidfab";
  // Zero asks for the highest API understood by both this implementation and
  // the installed loader (currently capped at Vulkan 1.3).
  uint32_t api_version = 0;
};

struct DeviceOptions {
  std::vector<std::string> extensions;
  bool enable_shader_float16 = false;
  bool enable_shader_int8 = false;
  bool enable_storage_buffer_16bit = false;
  bool enable_storage_buffer_8bit = false;
  bool enable_timeline_semaphore = false;
  bool enable_buffer_device_address = false;
  bool enable_descriptor_indexing = false;
};

class Device;

class PhysicalDevice {
 public:
  PhysicalDevice();
  const DeviceInfo& info() const;
  Device create_device(const DeviceOptions& options = {}) const;
  explicit operator bool() const noexcept;

 private:
  struct Impl;
  explicit PhysicalDevice(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
  friend class Instance;
};

class Instance {
 public:
  Instance();
  ~Instance();
  Instance(Instance&&) noexcept;
  Instance& operator=(Instance&&) noexcept;
  Instance(const Instance&) = delete;
  Instance& operator=(const Instance&) = delete;

  static Instance create(const InstanceOptions& options = {});
  // A missing loader is an ordinary deployment state, so availability can be
  // probed without exceptions. diagnostic receives the loader error if given.
  static bool available(std::string* diagnostic = nullptr) noexcept;

  Version loader_version() const;
  Version api_version() const;
  std::vector<PhysicalDevice> enumerate_devices() const;
  explicit operator bool() const noexcept;

 private:
  struct Impl;
  explicit Instance(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
};

class Queue {
 public:
  Queue();
  uint32_t family_index() const;
  void wait_idle() const;
  // Dispatchable Vulkan handles are pointer-shaped. This escape hatch is for
  // the later command-submission layer without leaking SDK types here. Calls
  // made through it are externally synchronized with Queue methods.
  void* native_handle() const noexcept;
  explicit operator bool() const noexcept;

 private:
  struct Impl;
  explicit Queue(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
  friend class Device;
  friend class Buffer;
};

class Device {
 public:
  Device();
  ~Device();
  Device(Device&&) noexcept;
  Device& operator=(Device&&) noexcept;
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  const DeviceInfo& info() const;
  Queue compute_queue() const;
  void wait_idle() const;
  void* native_handle() const noexcept;
  explicit operator bool() const noexcept;

 private:
  struct Impl;
  explicit Device(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
  friend class PhysicalDevice;
  friend class BufferPool;
  friend class ComputePipeline;
  friend class ComputeContext;
};

enum class BufferUsage : uint32_t {
  kStorage = 1u << 0,
  kUniform = 1u << 1,
  kTransferSource = 1u << 2,
  kTransferDestination = 1u << 3,
  kDeviceAddress = 1u << 4,
};

constexpr BufferUsage operator|(BufferUsage lhs, BufferUsage rhs) {
  return static_cast<BufferUsage>(static_cast<uint32_t>(lhs) |
                                  static_cast<uint32_t>(rhs));
}

enum class MemoryUsage {
  kDevice,    // device-local; normally not host-visible
  kUpload,    // persistently mapped, coherent preferred
  kReadback,  // persistently mapped, cached and coherent preferred
};

class Buffer {
 public:
  Buffer();
  ~Buffer();
  Buffer(Buffer&&) noexcept;
  Buffer& operator=(Buffer&&) noexcept;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  uint64_t size() const noexcept;
  BufferUsage usage() const noexcept;
  MemoryUsage memory_usage() const noexcept;
  // Raw mapped access is externally synchronized. Prefer read/write, which
  // serialize per memory block and perform non-coherent cache maintenance.
  void* mapped_data() noexcept;
  const void* mapped_data() const noexcept;
  void write(uint64_t offset, const void* data, uint64_t bytes);
  void read(uint64_t offset, void* data, uint64_t bytes) const;
  void flush(uint64_t offset = 0, uint64_t bytes = ~uint64_t{0});
  void invalidate(uint64_t offset = 0, uint64_t bytes = ~uint64_t{0}) const;
  // Waits for the device queue, then releases this buffer. Use this after a
  // raw native handle has participated in submitted work. Plain destruction
  // is intentionally wait-free and is valid only once the caller already
  // knows all GPU use is complete (Vulkan's external-synchronization rule).
  void reset_after_idle(const Queue& queue);
  uintptr_t native_handle() const noexcept;
  uint64_t memory_offset() const noexcept;
  explicit operator bool() const noexcept;

 private:
  struct Impl;
  explicit Buffer(std::shared_ptr<Impl> impl);
  std::shared_ptr<Impl> impl_;
  friend class BufferPool;
  friend class CommandList;
  friend class ComputePipeline;
};

class BufferPool {
 public:
  // 256 MiB amortises vkAllocateMemory overhead for model tensors without
  // committing a block until the first allocation. Small heaps are clamped.
  explicit BufferPool(const Device& device, uint64_t block_bytes = 256ull << 20);
  ~BufferPool();
  BufferPool(BufferPool&&) noexcept;
  BufferPool& operator=(BufferPool&&) noexcept;
  BufferPool(const BufferPool&) = delete;
  BufferPool& operator=(const BufferPool&) = delete;

  Buffer allocate(uint64_t bytes, BufferUsage usage, MemoryUsage memory);
  // Releases completely unused pooled blocks. Live buffers remain valid.
  void trim();
  uint64_t reserved_bytes() const;
  uint64_t used_bytes() const;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;
  friend class Buffer;
};

}  // namespace vidfab::vulkan
