#pragma once
#include "slopfab/cuda/reference_buffer.cuh"
#include <memory>
#include <utility>

namespace slopfab::moss {
enum class TemporaryBufferKind { kActivation, kScratch };

// Opt-in reuse for operations issued exclusively on the default CUDA stream
// of one device. A scope belongs to its creating thread and must not migrate
// between devices. Nested scopes restore the previous pools on exit.
class TemporaryBufferScope {
public:
  TemporaryBufferScope();
  ~TemporaryBufferScope();
  TemporaryBufferScope(const TemporaryBufferScope&) = delete;
  TemporaryBufferScope& operator=(const TemporaryBufferScope&) = delete;

  const std::shared_ptr<cuda::ReferenceBufferPool>& activations() const {
    return activations_;
  }

  const std::shared_ptr<cuda::ReferenceBufferPool>& temporaries() const {
    return temporaries_;
  }

private:
  std::shared_ptr<cuda::ReferenceBufferPool> activations_, temporaries_;
  std::shared_ptr<cuda::ReferenceBufferPool> previous_activations_, previous_temporaries_;
};

std::shared_ptr<cuda::ReferenceBufferPool> temporary_buffer_pool(TemporaryBufferKind kind);

// Every lease retains its pool, so a returned tensor may outlive the scope.
// Returning a lease does not wait: subsequent uses of the block are ordered
// after its previous uses on the same default stream. Final pool destruction
// uses DeviceBuffer's synchronizing frees. Checkpoint weights never use this.
template <typename T> class TemporaryBuffer {
public:
  TemporaryBuffer() = default;

  explicit TemporaryBuffer(size_t count,
                           TemporaryBufferKind kind = TemporaryBufferKind::kActivation)
      : pool_(temporary_buffer_pool(kind)), count_(count) {
    if (pool_) {
      if (count > (std::numeric_limits<size_t>::max() - 4095) / sizeof(T))
        throw std::overflow_error("temporary buffer size overflow");
      // KV attention scratch grows a few bytes per token; page-sized leases
      // avoid a fresh allocation at every autoregressive step.
      const size_t rounded = ((count * sizeof(T) + 4095) / 4096) * 4096;
      pooled_ = cuda::ReferenceBuffer<T>(rounded / sizeof(T), *pool_);
    } else
      plain_.allocate(count);
  }

  TemporaryBuffer(TemporaryBuffer&& other) noexcept {
    *this = std::move(other);
  }

  TemporaryBuffer& operator=(TemporaryBuffer&& other) noexcept {
    if (this != &other) {
      // Release the old lease before its final pool owner, including when
      // assigning from a buffer whose pool differs from ours.
      pooled_.reset();
      plain_.reset();
      pool_ = std::move(other.pool_);
      pooled_ = std::move(other.pooled_);
      plain_ = std::move(other.plain_);
      count_ = std::exchange(other.count_, 0);
    }
    return *this;
  }

  T* get() {
    return pool_ ? pooled_.get() : plain_.get();
  }

  const T* get() const {
    return pool_ ? pooled_.get() : plain_.get();
  }

  size_t size() const {
    return count_;
  }

  size_t nbytes() const {
    return count_ * sizeof(T);
  }

  bool empty() const {
    return count_ == 0;
  }

private:
  // Destruction order releases pooled_ before pool_.
  std::shared_ptr<cuda::ReferenceBufferPool> pool_;
  cuda::ReferenceBuffer<T> pooled_;
  cuda::DeviceBuffer<T> plain_;
  size_t count_ = 0;
};
}
