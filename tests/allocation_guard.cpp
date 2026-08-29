#include "allocation_guard.h"

#include <cstddef>
#include <cstdlib>
#include <new>

#ifdef _WIN32
#include <malloc.h>
#endif

namespace {
thread_local unsigned allocation_guard_depth = 0;

void* allocate_unaligned(std::size_t bytes) {
  if (allocation_guard_depth != 0) throw std::bad_alloc();
  if (void* result = std::malloc(bytes == 0 ? 1 : bytes)) return result;
  throw std::bad_alloc();
}

void* allocate_aligned(std::size_t bytes, std::size_t alignment) {
  if (allocation_guard_depth != 0) throw std::bad_alloc();
#ifdef _WIN32
  if (void* result = _aligned_malloc(bytes == 0 ? 1 : bytes, alignment))
    return result;
#else
  const std::size_t rounded =
      ((bytes == 0 ? 1 : bytes) + alignment - 1) / alignment * alignment;
  if (void* result = std::aligned_alloc(alignment, rounded)) return result;
#endif
  throw std::bad_alloc();
}
}  // namespace

void* operator new(std::size_t bytes) { return allocate_unaligned(bytes); }
void* operator new[](std::size_t bytes) { return allocate_unaligned(bytes); }
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete[](void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
void operator delete[](void* pointer, std::size_t) noexcept { std::free(pointer); }

void* operator new(std::size_t bytes, std::align_val_t alignment) {
  return allocate_aligned(bytes, static_cast<std::size_t>(alignment));
}
void* operator new[](std::size_t bytes, std::align_val_t alignment) {
  return allocate_aligned(bytes, static_cast<std::size_t>(alignment));
}
void operator delete(void* pointer, std::align_val_t) noexcept {
#ifdef _WIN32
  _aligned_free(pointer);
#else
  std::free(pointer);
#endif
}
void operator delete[](void* pointer, std::align_val_t alignment) noexcept {
  ::operator delete(pointer, alignment);
}
void operator delete(void* pointer, std::size_t,
                     std::align_val_t alignment) noexcept {
  ::operator delete(pointer, alignment);
}
void operator delete[](void* pointer, std::size_t,
                       std::align_val_t alignment) noexcept {
  ::operator delete(pointer, alignment);
}

namespace vidfab::test {
HostAllocationGuard::HostAllocationGuard() noexcept { ++allocation_guard_depth; }
HostAllocationGuard::~HostAllocationGuard() { --allocation_guard_depth; }
}  // namespace vidfab::test
