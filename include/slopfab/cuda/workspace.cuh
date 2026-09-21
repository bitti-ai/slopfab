// A single growable device scratch arena.
//
// Every stage in this pipeline needs transient device memory whose size
// depends on the request (sequence length, tile count), and on a 32 GB card
// with a 19 GB weight set the difference between "allocate per call" and "one
// arena" is whether the run fits. The arena grows to the high-water mark and
// never shrinks, so after the first denoising step there are no further
// allocations in the loop.
//
// It is deliberately not a general allocator: allocations are stack-like
// within one call site. Take a `Scope`, carve what you need, let it go.
#pragma once

#include <cstddef>
#include <cstdint>

#include "slopfab/cuda/device.h"

namespace slopfab::cuda {

class Workspace {
public:
  Workspace() = default;

  // Ensures at least `bytes` are available, reallocating if not. Any pointer
  // handed out earlier is invalidated by a grow, so reserve before carving.
  void reserve(size_t bytes);

  // Set capacity to the aligned size, including shrinking. Invalidates every
  // carved pointer. Only use between stages after their stream has completed.
  void resize(size_t bytes);

  // Bumps the cursor and returns a 256-byte-aligned pointer. Throws if the
  // arena is exhausted — that is a sizing bug, not a runtime condition.
  void* alloc(size_t bytes);

  template <typename T> T* alloc_n(size_t count) {
    return static_cast<T*>(alloc(count * sizeof(T)));
  }

  // Resets the cursor to zero without freeing.
  void clear() {
    cursor_ = 0;
  }

  size_t capacity() const {
    return buffer_.nbytes();
  }

  size_t used() const {
    return cursor_;
  }

  // Restores the cursor on destruction, so nested helpers can carve scratch
  // without leaking it for the rest of the call.
  class Scope {
  public:
    explicit Scope(Workspace& w) : ws_(w), mark_(w.cursor_) {
    }

    ~Scope() {
      ws_.cursor_ = mark_;
    }

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

  private:
    Workspace& ws_;
    size_t mark_;
  };

private:
  DeviceBuffer<uint8_t> buffer_;
  size_t cursor_ = 0;
};

} // namespace slopfab::cuda
