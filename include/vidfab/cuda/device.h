// Thin RAII layer over the CUDA runtime.
//
// Scope is deliberately small: an error-checking macro, owning device/host
// buffers, and a stream wrapper. Everything above this uses raw pointers, so
// kernels stay ordinary CUDA and nothing here appears in a hot loop.
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace vidfab::cuda {

// Throws std::runtime_error naming the call site. Every runtime call goes
// through this: a silently ignored async error surfaces later as a wrong
// result, which is far more expensive to diagnose than a throw.
void check(cudaError_t status, const char* expr, const char* file, int line);

#define VIDFAB_CUDA_CHECK(expr) ::vidfab::cuda::check((expr), #expr, __FILE__, __LINE__)

struct DeviceInfo {
  int index = 0;
  std::string name;
  int major = 0;
  int minor = 0;
  size_t total_memory = 0;
  size_t free_memory = 0;
  int multiprocessors = 0;
  int max_threads_per_block = 0;
  size_t shared_memory_per_block = 0;
  bool supports_bf16 = false;    // sm_80+
  bool supports_fp8 = false;     // sm_89+
  bool supports_fp4 = false;     // sm_100+ (Blackwell)

  int compute_capability() const { return major * 10 + minor; }
};

int device_count();
DeviceInfo query_device(int index);
void set_device(int index);

// Owning device allocation. Move-only; freeing is best-effort in the
// destructor because throwing from one would terminate.
template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t count) { allocate(count); }

  ~DeviceBuffer() { reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept
      : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  void allocate(size_t count) {
    reset();
    if (count == 0) return;
    void* raw = nullptr;
    VIDFAB_CUDA_CHECK(cudaMalloc(&raw, count * sizeof(T)));
    ptr_ = static_cast<T*>(raw);
    count_ = count;
  }

  void reset() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
    count_ = 0;
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  size_t size() const { return count_; }
  size_t nbytes() const { return count_ * sizeof(T); }
  bool empty() const { return count_ == 0; }

  void copy_from_host(const T* src, size_t count, cudaStream_t stream = nullptr);
  void copy_to_host(T* dst, size_t count, cudaStream_t stream = nullptr) const;
  void zero(cudaStream_t stream = nullptr);

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

// Page-locked host allocation, so uploads can overlap compute. Worth the
// slower allocation for weight staging, which happens once per stage.
template <typename T>
class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  explicit PinnedBuffer(size_t count) { allocate(count); }
  ~PinnedBuffer() { reset(); }

  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  PinnedBuffer(PinnedBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  void allocate(size_t count) {
    reset();
    if (count == 0) return;
    void* raw = nullptr;
    VIDFAB_CUDA_CHECK(cudaHostAlloc(&raw, count * sizeof(T), cudaHostAllocDefault));
    ptr_ = static_cast<T*>(raw);
    count_ = count;
  }

  void reset() {
    if (ptr_ != nullptr) {
      cudaFreeHost(ptr_);
      ptr_ = nullptr;
    }
    count_ = 0;
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  size_t size() const { return count_; }
  size_t nbytes() const { return count_ * sizeof(T); }

 private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

class Stream {
 public:
  Stream();
  ~Stream();

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  Stream(Stream&& other) noexcept : stream_(other.stream_) { other.stream_ = nullptr; }
  Stream& operator=(Stream&& other) noexcept {
    if (this != &other) {
      destroy();
      stream_ = other.stream_;
      other.stream_ = nullptr;
    }
    return *this;
  }

  cudaStream_t get() const { return stream_; }
  operator cudaStream_t() const { return stream_; }
  void synchronize() const;

 private:
  cudaStream_t stream_ = nullptr;
  void destroy();
};

// --- template definitions ---------------------------------------------------

template <typename T>
void DeviceBuffer<T>::copy_from_host(const T* src, size_t count, cudaStream_t stream) {
  if (count == 0) return;
  if (count > count_) {
    throw std::runtime_error("DeviceBuffer::copy_from_host: source larger than allocation");
  }
  if (stream != nullptr) {
    VIDFAB_CUDA_CHECK(
        cudaMemcpyAsync(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice, stream));
  } else {
    VIDFAB_CUDA_CHECK(cudaMemcpy(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice));
  }
}

template <typename T>
void DeviceBuffer<T>::copy_to_host(T* dst, size_t count, cudaStream_t stream) const {
  if (count == 0) return;
  if (count > count_) {
    throw std::runtime_error("DeviceBuffer::copy_to_host: request larger than allocation");
  }
  if (stream != nullptr) {
    VIDFAB_CUDA_CHECK(
        cudaMemcpyAsync(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
  } else {
    VIDFAB_CUDA_CHECK(cudaMemcpy(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
  }
}

template <typename T>
void DeviceBuffer<T>::zero(cudaStream_t stream) {
  if (count_ == 0) return;
  if (stream != nullptr) {
    VIDFAB_CUDA_CHECK(cudaMemsetAsync(ptr_, 0, count_ * sizeof(T), stream));
  } else {
    VIDFAB_CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T)));
  }
}

}  // namespace vidfab::cuda
