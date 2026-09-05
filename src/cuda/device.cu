#include "slopfab/cuda/device.h"

#include <stdexcept>
#include <string>
#include <mutex>

namespace slopfab::cuda {

void check(cudaError_t status, const char* expr, const char* file, int line) {
  if (status == cudaSuccess) return;
  // Strip the directory prefix so messages stay readable.
  const char* base = file;
  for (const char* p = file; *p != '\0'; ++p) {
    if (*p == '/' || *p == '\\') base = p + 1;
  }
  throw std::runtime_error("cuda: " + std::string(cudaGetErrorName(status)) + ": " +
                           cudaGetErrorString(status) + "\n  at " + base + ":" +
                           std::to_string(line) + "\n  in " + expr);
}

int device_count() {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  // No driver or no device is a legitimate state to report, not an error to
  // throw from: the CLI needs to print a helpful message instead.
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) return 0;
  SLOPFAB_CUDA_CHECK(status);
  return count;
}

DeviceInfo query_device(int index) {
  cudaDeviceProp props{};
  SLOPFAB_CUDA_CHECK(cudaGetDeviceProperties(&props, index));

  DeviceInfo info;
  info.index = index;
  info.name = props.name;
  info.major = props.major;
  info.minor = props.minor;
  info.total_memory = props.totalGlobalMem;
  info.multiprocessors = props.multiProcessorCount;
  info.max_threads_per_block = props.maxThreadsPerBlock;
  info.shared_memory_per_block = props.sharedMemPerBlock;

  const int cc = info.compute_capability();
  info.supports_bf16 = cc >= 80;
  info.supports_fp8 = cc >= 89;
  info.supports_fp4 = cc >= 100;

  // Free memory is only meaningful for the current device, so query it after
  // switching; restore the previous device to avoid surprising the caller.
  int previous = 0;
  SLOPFAB_CUDA_CHECK(cudaGetDevice(&previous));
  if (cudaSetDevice(index) == cudaSuccess) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
      info.free_memory = free_bytes;
    }
    SLOPFAB_CUDA_CHECK(cudaSetDevice(previous));
  }
  return info;
}

void set_device(int index) { SLOPFAB_CUDA_CHECK(cudaSetDevice(index)); }

int device_compute_capability(int index) {
  constexpr int kCachedDevices = 64;
  static std::once_flag once[kCachedDevices];
  static int capabilities[kCachedDevices]{};
  if (index < 0) throw std::out_of_range("CUDA device index must be non-negative");
  if (index >= kCachedDevices) {
    cudaDeviceProp properties{};
    SLOPFAB_CUDA_CHECK(cudaGetDeviceProperties(&properties, index));
    return properties.major * 10 + properties.minor;
  }
  std::call_once(once[index], [index] {
    cudaDeviceProp properties{};
    SLOPFAB_CUDA_CHECK(cudaGetDeviceProperties(&properties, index));
    capabilities[index] = properties.major * 10 + properties.minor;
  });
  return capabilities[index];
}

int current_device_compute_capability() {
  int device = 0;
  SLOPFAB_CUDA_CHECK(cudaGetDevice(&device));
  return device_compute_capability(device);
}

Stream::Stream() {
  // Non-blocking: a default stream would implicitly synchronise against the
  // legacy NULL stream, silently serialising any future concurrent work.
  SLOPFAB_CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

Stream::~Stream() { destroy(); }

void Stream::destroy() {
  if (stream_ != nullptr) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

void Stream::synchronize() const {
  if (stream_ != nullptr) SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace slopfab::cuda
