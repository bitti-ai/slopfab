#pragma once

#include <algorithm>
#include <limits>
#include <vector>
#include "slopfab/cuda/device.h"

namespace slopfab::cuda {

// Leases are used exclusively on the encoder's one stream. A returned block
// can be reused immediately on that stream; the owner drains it before the
// pool is destroyed. Separate activation/scratch pools avoid a tiny scratch
// request pinning a multi-GiB activation block.
class ReferenceBufferPool {
public:
  ReferenceBufferPool() = default;
  ReferenceBufferPool(const ReferenceBufferPool&) = delete;
  ReferenceBufferPool& operator=(const ReferenceBufferPool&) = delete;
  ReferenceBufferPool(ReferenceBufferPool&&) = delete;
  ReferenceBufferPool& operator=(ReferenceBufferPool&&) = delete;

  struct Lease {
    void* pointer;
    size_t slot;
  };

  Lease acquire(size_t bytes) {
    size_t best = blocks_.size();
    for (size_t i = 0; i < blocks_.size(); ++i)
      if (!blocks_[i].used && blocks_[i].data.size() >= bytes &&
          (best == blocks_.size() || blocks_[i].data.size() < blocks_[best].data.size()))
        best = i;
    if (best == blocks_.size()) {
      // Discard undersized idle blocks before growing, bounding retention by
      // the live graph rather than every shape encountered over a whole clip.
      for (size_t i = 0; i < blocks_.size(); ++i)
        if (!blocks_[i].used) {
          reserved_ -= blocks_[i].data.size();
          blocks_[i].data.reset();
          best = i;
        }
      if (best == blocks_.size())
        blocks_.emplace_back();
      blocks_[best].data.allocate(bytes);
      reserved_ += bytes;
      ++allocations_;
    }
    blocks_[best].used = bytes;
    live_ += bytes;
    peak_ = std::max(peak_, live_);
    return {blocks_[best].data.get(), best};
  }

  void release(size_t slot) noexcept {
    live_ -= blocks_[slot].used;
    blocks_[slot].used = 0;
  }

  size_t peak_bytes() const {
    return peak_;
  }

  size_t reserved_bytes() const {
    return reserved_;
  }

  size_t allocations() const {
    return allocations_;
  }

private:
  struct Block {
    DeviceBuffer<uint8_t> data;
    size_t used = 0;
  };

  std::vector<Block> blocks_;
  size_t live_ = 0, peak_ = 0, reserved_ = 0, allocations_ = 0;
};

template <typename T> class ReferenceBuffer {
public:
  ReferenceBuffer() = default;

  ReferenceBuffer(size_t count, ReferenceBufferPool& pool) : count_(count) {
    if (!count)
      return;
    if (count > std::numeric_limits<size_t>::max() / sizeof(T))
      throw std::overflow_error("reference buffer size overflow");
    auto lease = pool.acquire(count * sizeof(T));
    pointer_ = static_cast<T*>(lease.pointer);
    slot_ = lease.slot;
    pool_ = &pool;
  }

  ~ReferenceBuffer() {
    reset();
  }

  ReferenceBuffer(const ReferenceBuffer&) = delete;
  ReferenceBuffer& operator=(const ReferenceBuffer&) = delete;

  ReferenceBuffer(ReferenceBuffer&& other) noexcept {
    swap(other);
  }

  ReferenceBuffer& operator=(ReferenceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      swap(other);
    }
    return *this;
  }

  void reset() noexcept {
    if (pool_)
      pool_->release(slot_);
    pointer_ = nullptr;
    pool_ = nullptr;
    count_ = 0;
  }

  T* get() const {
    return pointer_;
  }

  size_t size() const {
    return count_;
  }

  void copy_from_host(const T* src, size_t count, cudaStream_t stream) {
    if (count > count_)
      throw std::out_of_range("reference upload size");
    SLOPFAB_CUDA_CHECK(
        cudaMemcpyAsync(pointer_, src, count * sizeof(T), cudaMemcpyHostToDevice, stream));
  }

  void copy_to_host(T* dst, size_t count, cudaStream_t stream) const {
    if (count > count_)
      throw std::out_of_range("reference download size");
    SLOPFAB_CUDA_CHECK(
        cudaMemcpyAsync(dst, pointer_, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
  }

private:
  void swap(ReferenceBuffer& other) noexcept {
    std::swap(pointer_, other.pointer_);
    std::swap(pool_, other.pool_);
    std::swap(slot_, other.slot_);
    std::swap(count_, other.count_);
  }

  T* pointer_ = nullptr;
  ReferenceBufferPool* pool_ = nullptr;
  size_t slot_ = 0, count_ = 0;
};
} // namespace slopfab::cuda
