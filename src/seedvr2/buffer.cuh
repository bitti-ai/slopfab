#pragma once
#include "slopfab/cuda/reference_buffer.cuh"
#include <memory>

namespace slopfab::seedvr2 {
// Each lease keeps its pool alive, including tensors returned to a caller.
// All accesses and reuse are ordered on the runtime's default CUDA stream.
template<class T> class Buffer {
public:
  Buffer() = default;
  explicit Buffer(size_t n, std::shared_ptr<cuda::ReferenceBufferPool> pool = {})
      : pool_(std::move(pool)) {
    if (pool_) pooled_ = cuda::ReferenceBuffer<T>(n, *pool_);
    else plain_.allocate(n);
  }
  Buffer(Buffer&& other) noexcept { *this = std::move(other); }
  Buffer& operator=(Buffer&& other) noexcept {
    if (this != &other) {
      pooled_.reset(); plain_.reset();
      pool_ = std::move(other.pool_);
      pooled_ = std::move(other.pooled_);
      plain_ = std::move(other.plain_);
      view_ = std::exchange(other.view_, nullptr);
      view_size_ = std::exchange(other.view_size_, 0);
    }
    return *this;
  }
  // Borrowed views are for nested operations; the owning tensor must outlive them.
  static Buffer view(T* pointer, size_t count) {
    Buffer result; result.view_ = pointer; result.view_size_ = count; return result;
  }
  T* get() const { return view_ ? view_ : pool_ ? pooled_.get() : const_cast<T*>(plain_.get()); }
  size_t size() const { return view_ ? view_size_ : pool_ ? pooled_.size() : plain_.size(); }
  void copy_from_host(const T* p, size_t n) {
    if (n > size()) throw std::out_of_range("SeedVR2 upload size");
    if (n) SLOPFAB_CUDA_CHECK(cudaMemcpy(get(), p, n * sizeof(T), cudaMemcpyHostToDevice));
  }
  void copy_to_host(T* p, size_t n) const {
    if (n > size()) throw std::out_of_range("SeedVR2 download size");
    if (n) SLOPFAB_CUDA_CHECK(cudaMemcpy(p, get(), n * sizeof(T), cudaMemcpyDeviceToHost));
  }
  void zero() {
    if (size()) SLOPFAB_CUDA_CHECK(cudaMemsetAsync(get(), 0, size() * sizeof(T)));
  }
private:
  std::shared_ptr<cuda::ReferenceBufferPool> pool_;
  cuda::ReferenceBuffer<T> pooled_;
  cuda::DeviceBuffer<T> plain_;
  T* view_ = nullptr;
  size_t view_size_ = 0;
};
}
