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

namespace slopfab::cuda {

// Throws std::runtime_error naming the call site. Every runtime call goes
// through this: a silently ignored async error surfaces later as a wrong
// result, which is far more expensive to diagnose than a throw.
void check(cudaError_t status, const char* expr, const char* file, int line);

#define SLOPFAB_CUDA_CHECK(expr) ::slopfab::cuda::check((expr), #expr, __FILE__, __LINE__)

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
  bool supports_bf16 = false; // sm_80+
  bool supports_fp8 = false;  // sm_89+
  bool supports_fp4 = false;  // sm_100+ (Blackwell)

  int compute_capability() const {
    return major * 10 + minor;
  }
};

int device_count();
DeviceInfo query_device(int index);
void set_device(int index);

// Cached immutable hardware capability. The cache is indexed by CUDA device,
// so callers may switch devices without reusing another GPU's answer.
int device_compute_capability(int index);
int current_device_compute_capability();

// Owning device allocation. Move-only; freeing is best-effort in the
// destructor because throwing from one would terminate.
template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t count) {
    allocate(count);
  }

  ~DeviceBuffer() {
    reset();
  }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
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
    if (count == 0)
      return;
    void* raw = nullptr;
    SLOPFAB_CUDA_CHECK(cudaMalloc(&raw, count * sizeof(T)));
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

  T* get() {
    return ptr_;
  }

  const T* get() const {
    return ptr_;
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

  void copy_from_host(const T* src, size_t count, cudaStream_t stream = nullptr);
  void copy_to_host(T* dst, size_t count, cudaStream_t stream = nullptr) const;
  void zero(cudaStream_t stream = nullptr);

private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

// Page-locked host allocation, so uploads can overlap compute. Worth the
// slower allocation for weight staging, which happens once per stage.
template <typename T> class PinnedBuffer {
public:
  PinnedBuffer() = default;

  explicit PinnedBuffer(size_t count) {
    allocate(count);
  }

  ~PinnedBuffer() {
    reset();
  }

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
    if (count == 0)
      return;
    void* raw = nullptr;
    SLOPFAB_CUDA_CHECK(cudaHostAlloc(&raw, count * sizeof(T), cudaHostAllocDefault));
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

  T* get() {
    return ptr_;
  }

  const T* get() const {
    return ptr_;
  }

  size_t size() const {
    return count_;
  }

  size_t nbytes() const {
    return count_ * sizeof(T);
  }

private:
  T* ptr_ = nullptr;
  size_t count_ = 0;
};

// Best-effort page-lock of an already-mapped host range — in practice a whole
// checkpoint file mapping — so the DMA engine can read weights straight out of
// it instead of staging every tensor through a host copy first. Measured on
// the text encoder, that is the difference between 8.7 GB/s pageable and
// ~42 GB/s, and most of what it removes is not memcpy bandwidth but soft page
// faults on a multi-gigabyte mapping.
//
// Best-effort by design: locking tens of gigabytes can fail on a machine short
// of physical memory or lockable pages, and that is not a reason to refuse to
// run. `contains` then simply answers false and callers take their staged
// path, which is correct either way and only slower.
class RegisteredMapping {
public:
  RegisteredMapping() = default;

  // `bytes` is the usable length; the registration itself is rounded up to a
  // whole page, which stays inside a file mapping.
  RegisteredMapping(const void* base, size_t bytes) {
    if (base == nullptr || bytes == 0)
      return;
    constexpr size_t kPage = 4096;
    const size_t locked = (bytes + kPage - 1) / kPage * kPage;
    if (cudaHostRegister(const_cast<void*>(base), locked, cudaHostRegisterReadOnly) ==
        cudaSuccess) {
      base_ = base;
      bytes_ = bytes;
    } else {
      // Clear the sticky error so the next real call is not misattributed.
      cudaGetLastError();
    }
  }

  ~RegisteredMapping() {
    reset();
  }

  RegisteredMapping(const RegisteredMapping&) = delete;
  RegisteredMapping& operator=(const RegisteredMapping&) = delete;

  void reset() {
    if (base_ != nullptr) {
      cudaHostUnregister(const_cast<void*>(base_));
      // Symmetric with the constructor: do not leave a sticky error behind for
      // the next unrelated call to be blamed for.
      cudaGetLastError();
    }
    base_ = nullptr;
    bytes_ = 0;
  }

  bool registered() const {
    return base_ != nullptr;
  }

  // Whether `[p, p + n)` lies inside the page-locked range.
  //
  // This is a safety net, not a provenance test: a caller must already know
  // that `p` points into the mapping, because an unrelated heap block can sit
  // anywhere relative to it. Compared as integers rather than pointers — a
  // pointer subtraction between unrelated objects is undefined behaviour, and
  // the obvious `n <= (b + bytes_) - q` spelling also wraps to a huge size_t
  // for any `q` past the end, which made this answer true for exactly the
  // pointers it exists to reject.
  bool contains(const void* p, size_t n) const {
    if (base_ == nullptr)
      return false;
    const auto b = reinterpret_cast<uintptr_t>(base_);
    const auto q = reinterpret_cast<uintptr_t>(p);
    return q >= b && q <= b + bytes_ && n <= (b + bytes_) - q;
  }

private:
  const void* base_ = nullptr;
  size_t bytes_ = 0;
};

class Stream {
public:
  Stream();
  ~Stream();

  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  Stream(Stream&& other) noexcept : stream_(other.stream_) {
    other.stream_ = nullptr;
  }

  Stream& operator=(Stream&& other) noexcept {
    if (this != &other) {
      destroy();
      stream_ = other.stream_;
      other.stream_ = nullptr;
    }
    return *this;
  }

  cudaStream_t get() const {
    return stream_;
  }

  operator cudaStream_t() const {
    return stream_;
  }

  void synchronize() const;

private:
  cudaStream_t stream_ = nullptr;
  void destroy();
};

// --- template definitions ---------------------------------------------------

template <typename T>
void DeviceBuffer<T>::copy_from_host(const T* src, size_t count, cudaStream_t stream) {
  if (count == 0)
    return;
  if (count > count_) {
    throw std::runtime_error("DeviceBuffer::copy_from_host: source larger than allocation");
  }
  if (stream != nullptr) {
    SLOPFAB_CUDA_CHECK(
        cudaMemcpyAsync(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice, stream));
  } else {
    SLOPFAB_CUDA_CHECK(cudaMemcpy(ptr_, src, count * sizeof(T), cudaMemcpyHostToDevice));
  }
}

template <typename T>
void DeviceBuffer<T>::copy_to_host(T* dst, size_t count, cudaStream_t stream) const {
  if (count == 0)
    return;
  if (count > count_) {
    throw std::runtime_error("DeviceBuffer::copy_to_host: request larger than allocation");
  }
  if (stream != nullptr) {
    SLOPFAB_CUDA_CHECK(
        cudaMemcpyAsync(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
  } else {
    SLOPFAB_CUDA_CHECK(cudaMemcpy(dst, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
  }
}

template <typename T> void DeviceBuffer<T>::zero(cudaStream_t stream) {
  if (count_ == 0)
    return;
  if (stream != nullptr) {
    SLOPFAB_CUDA_CHECK(cudaMemsetAsync(ptr_, 0, count_ * sizeof(T), stream));
  } else {
    SLOPFAB_CUDA_CHECK(cudaMemset(ptr_, 0, count_ * sizeof(T)));
  }
}

} // namespace slopfab::cuda
