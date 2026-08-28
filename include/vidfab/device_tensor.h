#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace vidfab {

enum class DeviceBackend { kCuda, kVulkan };
enum class ScalarType { kFloat32 };

struct TensorLayout {
  uint32_t rank = 0;
  std::array<uint64_t, 6> extent{};
  std::array<uint64_t, 6> stride{};  // elements, not bytes

  static TensorLayout contiguous(const uint64_t* extents, uint32_t rank);
  uint64_t elements() const;
  uint64_t bytes(ScalarType type) const;
  bool is_contiguous() const;
};

// Backend-neutral, non-owning device view. `resource` is an opaque backend
// allocation identity; operator implementations validate backend and context
// before interpreting it. No CUDA or Vulkan ABI type crosses this interface.
struct DeviceTensorView {
  DeviceBackend backend = DeviceBackend::kCuda;
  ScalarType type = ScalarType::kFloat32;
  TensorLayout layout;
  uintptr_t resource = 0;
  uint64_t byte_offset = 0;
  uint64_t byte_size = 0;
};

struct WorkspaceSpan {
  DeviceBackend backend = DeviceBackend::kCuda;
  uintptr_t resource = 0;
  uint64_t byte_offset = 0;
  uint64_t byte_size = 0;
};

// Stack-like scratch contract shared by backend stages. A stage reserves once,
// carves aligned spans during recording, and resets the cursor after the job;
// growth may invalidate old spans and therefore happens before command build.
class DeviceWorkspace {
 public:
  virtual ~DeviceWorkspace() = default;
  virtual DeviceBackend backend() const noexcept = 0;
  virtual void reserve(uint64_t bytes) = 0;
  virtual WorkspaceSpan allocate(uint64_t bytes, uint64_t alignment = 256) = 0;
  virtual void reset() noexcept = 0;
  virtual uint64_t capacity() const noexcept = 0;
  virtual uint64_t used() const noexcept = 0;
};

}  // namespace vidfab
