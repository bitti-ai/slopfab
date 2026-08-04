// The device scratch arena.
//
// Bump allocation with a 256-byte grain. 256 rather than 128 because that is
// the alignment cuBLAS wants for its operands and the width of a full memory
// transaction; over-aligning costs at most a few kilobytes per call site.

#include "vidfab/cuda/workspace.cuh"

#include <stdexcept>
#include <string>

namespace vidfab::cuda {
namespace {

constexpr size_t kAlign = 256;

size_t align_up(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

}  // namespace

void Workspace::reserve(size_t bytes) {
  if (bytes <= buffer_.nbytes()) return;
  // A grow frees the old allocation, so every pointer handed out so far is
  // dangling afterwards. Callers size the arena once at load time from the
  // per-layer maxima; growing mid-forward is a bug, not a fallback.
  buffer_.allocate(align_up(bytes));
  cursor_ = 0;
}

void* Workspace::alloc(size_t bytes) {
  const size_t offset = align_up(cursor_);
  const size_t end = offset + bytes;
  if (end > buffer_.nbytes()) {
    throw std::runtime_error("Workspace::alloc: need " + std::to_string(end) + " bytes, have " +
                             std::to_string(buffer_.nbytes()) +
                             "; reserve() the high-water mark at load time");
  }
  cursor_ = end;
  return buffer_.get() + offset;
}

}  // namespace vidfab::cuda
