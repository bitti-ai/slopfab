#include "weight_stream.cuh"
#include "slopfab/cuda/device.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <array>
#include <climits>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <thread>

namespace slopfab::seedvr2 {
namespace {
constexpr size_t kChunk = size_t(16) << 20;
using BF = __nv_bfloat16;

__global__ void convert(const void* raw, BF* out, size_t n, DType dtype) {
  size_t i = size_t(blockIdx.x) * 256 + threadIdx.x;
  if (i >= n)
    return;
  float value;
  if (dtype == DType::kF16)
    value = __half2float(static_cast<const __half*>(raw)[i]);
  else if (dtype == DType::kF32)
    value = static_cast<const float*>(raw)[i];
  else {
    int bits = static_cast<const uint8_t*>(raw)[i], exponent = (bits >> 3) & 15;
    int mantissa = bits & 7;
    value = exponent ? ldexpf(1.0f + mantissa * .125f, exponent - 7) : ldexpf(float(mantissa), -9);
    if (exponent == 15 && mantissa == 7)
      value = nanf("");
    if (bits & 128)
      value = -value;
  }
  out[i] = __float2bfloat16_rn(value);
}

size_t scalar_bytes(DType type) {
  if (type == DType::kF16 || type == DType::kBF16)
    return 2;
  if (type == DType::kF32)
    return 4;
  if (type == DType::kF8E4M3)
    return 1;
  return 0;
}
} // namespace

struct WeightStream::Impl {
  struct Entry {
    const TensorView* tensor;
    size_t offset;
  };

  struct Slot {
    cuda::DeviceBuffer<uint8_t> data;
    cudaEvent_t ready = nullptr, released = nullptr;
    int block = -1;
    bool release_recorded = false;
  };

  int device, active = -1, pending = -1;
  bool busy = false, stopping = false, started = false;
  size_t bytes = 0;
  std::vector<std::map<std::string, Entry>> plans;
  std::array<Slot, 2> slots;
  cuda::DeviceBuffer<uint8_t> raw;
  std::array<cuda::PinnedBuffer<uint8_t>, 2> staging;
  std::array<cudaEvent_t, 2> staged{};
  std::array<bool, 2> staged_recorded{};
  cudaStream_t transfer = nullptr;
  std::thread worker;
  mutable std::mutex mutex;
  std::condition_variable changed;
  std::exception_ptr failure;

  explicit Impl(int d) : device(d) {
  }

  ~Impl() {
    int previous = device;
    cudaGetDevice(&previous);
    cudaSetDevice(device);
    // Default-stream work can reference the current block's borrowed views.
    // Complete it before freeing or joining a worker waiting on release events.
    cudaStreamSynchronize(nullptr);
    {
      std::lock_guard<std::mutex> lock(mutex);
      stopping = true;
      pending = -1;
    }
    changed.notify_all();
    if (worker.joinable())
      worker.join();
    if (transfer)
      cudaStreamSynchronize(transfer);
    for (auto& slot : slots) {
      slot.data.reset();
      if (slot.ready)
        cudaEventDestroy(slot.ready);
      if (slot.released)
        cudaEventDestroy(slot.released);
    }
    for (auto& host : staging)
      host.reset();
    for (auto event : staged)
      if (event)
        cudaEventDestroy(event);
    raw.reset();
    if (transfer)
      cudaStreamDestroy(transfer);
    cudaSetDevice(previous);
  }

  void upload(int block) {
    Slot& slot = slots[block % 2];
    if (slot.release_recorded)
      SLOPFAB_CUDA_CHECK(cudaStreamWaitEvent(transfer, slot.released, 0));
    size_t chunk_index = 0;
    for (const auto& item : plans[block]) {
      const Entry& entry = item.second;
      const TensorView& tensor = *entry.tensor;
      size_t width = scalar_bytes(tensor.dtype), elements = size_t(tensor.numel());
      BF* destination = reinterpret_cast<BF*>(slot.data.get() + entry.offset);
      for (size_t first = 0; first < elements;) {
        int host_index = int(chunk_index++ % 2);
        if (staged_recorded[host_index])
          SLOPFAB_CUDA_CHECK(cudaEventSynchronize(staged[host_index]));
        size_t count = std::min(elements - first, kChunk / width);
        size_t nbytes = count * width;
        std::memcpy(staging[host_index].get(),
                    static_cast<const uint8_t*>(tensor.data) + first * width, nbytes);
        void* target = tensor.dtype == DType::kBF16 ? static_cast<void*>(destination + first)
                                                    : static_cast<void*>(raw.get());
        SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(target, staging[host_index].get(), nbytes,
                                           cudaMemcpyHostToDevice, transfer));
        // Host staging may be reused after the copy; raw reuse remains ordered
        // behind conversion because both operations share the transfer stream.
        SLOPFAB_CUDA_CHECK(cudaEventRecord(staged[host_index], transfer));
        staged_recorded[host_index] = true;
        if (tensor.dtype != DType::kBF16) {
          convert<<<int((count + 255) / 256), 256, 0, transfer>>>(raw.get(), destination + first,
                                                                  count, tensor.dtype);
          SLOPFAB_CUDA_CHECK(cudaGetLastError());
        }
        first += count;
      }
    }
    SLOPFAB_CUDA_CHECK(cudaEventRecord(slot.ready, transfer));
  }

  void run() noexcept {
    try {
      SLOPFAB_CUDA_CHECK(cudaSetDevice(device));
      for (;;) {
        int block;
        {
          std::unique_lock<std::mutex> lock(mutex);
          changed.wait(lock, [&] {
            return stopping || pending >= 0;
          });
          if (stopping)
            return;
          block = pending;
          pending = -1;
          busy = true;
        }
        upload(block);
        {
          std::lock_guard<std::mutex> lock(mutex);
          slots[block % 2].block = block;
          busy = false;
        }
        changed.notify_all();
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex);
      failure = std::current_exception();
      busy = false;
      pending = -1;
      changed.notify_all();
    }
  }
};

WeightStream::WeightStream(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
}

WeightStream::~WeightStream() = default;

std::unique_ptr<WeightStream> WeightStream::create(SafeTensors& file, int device, size_t budget) {
  if (const char* value = std::getenv("SLOPFAB_SEEDVR2_PREFETCH"))
    if (std::string(value) == "0")
      return nullptr;
  auto p = std::make_unique<Impl>(device);
  p->plans.resize(32);
  std::array<size_t, 32> sizes{};
  for (const auto& item : file.tensors()) {
    std::string name = item.first;
    const std::string wrapper = "model.diffusion_model.";
    if (name.compare(0, wrapper.size(), wrapper) == 0)
      name.erase(0, wrapper.size());
    if (name.compare(0, 7, "blocks.") != 0)
      continue;
    size_t end = name.find('.', 7);
    if (end == std::string::npos)
      continue;
    int block = std::stoi(name.substr(7, end - 7));
    if (block < 0 || block >= 32)
      return nullptr;
    const TensorView& tensor = item.second;
    if (!scalar_bytes(tensor.dtype) || tensor.numel() < 1 || tensor.numel() > INT_MAX)
      return nullptr;
    size_t count = size_t(tensor.numel());
    if (tensor.nbytes != count * scalar_bytes(tensor.dtype))
      return nullptr;
    size_t& offset = sizes[block];
    p->plans[block].emplace(name, Impl::Entry{&tensor, offset});
    offset += (count * sizeof(BF) + 255) / 256 * 256;
  }
  if (std::any_of(sizes.begin(), sizes.end(), [](size_t n) {
        return n == 0;
      }))
    return nullptr;
  size_t largest = *std::max_element(sizes.begin(), sizes.end());
  p->bytes = largest * 2 + kChunk;
  size_t available = 0, total = 0;
  SLOPFAB_CUDA_CHECK(cudaMemGetInfo(&available, &total));
  if (p->bytes > budget || p->bytes > available / 4)
    return nullptr;
  // Allocation failure is an optional-optimization fallback. Worker/runtime
  // failures after construction are surfaced to the caller by begin/finish.
  try {
    SLOPFAB_CUDA_CHECK(cudaStreamCreateWithFlags(&p->transfer, cudaStreamNonBlocking));
    p->raw.allocate(kChunk);
    for (auto& slot : p->slots) {
      slot.data.allocate(largest);
      SLOPFAB_CUDA_CHECK(cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming));
      SLOPFAB_CUDA_CHECK(cudaEventCreateWithFlags(&slot.released, cudaEventDisableTiming));
    }
    for (int i = 0; i < 2; ++i) {
      p->staging[i].allocate(kChunk);
      SLOPFAB_CUDA_CHECK(cudaEventCreateWithFlags(&p->staged[i], cudaEventDisableTiming));
    }
    p->worker = std::thread([raw = p.get()] {
      raw->run();
    });
  } catch (...) {
    cudaGetLastError();
    return nullptr;
  }
  return std::unique_ptr<WeightStream>(new WeightStream(std::move(p)));
}

void WeightStream::begin() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->failure)
    std::rethrow_exception(impl_->failure);
  if (impl_->started || impl_->busy || impl_->pending >= 0)
    throw std::logic_error("SeedVR2: weight stream already active");
  impl_->active = -1;
  for (auto& slot : impl_->slots)
    slot.block = -1;
  impl_->started = true;
  impl_->pending = 0;
  impl_->changed.notify_all();
}

void WeightStream::begin_block(int block) {
  std::unique_lock<std::mutex> lock(impl_->mutex);
  if (!impl_->started || block != impl_->active + 1 || block >= int(impl_->plans.size()))
    throw std::logic_error("SeedVR2: weight stream block order");
  if (impl_->active >= 0) {
    auto& previous = impl_->slots[impl_->active % 2];
    SLOPFAB_CUDA_CHECK(cudaEventRecord(previous.released, nullptr));
    previous.release_recorded = true;
  }
  auto& current = impl_->slots[block % 2];
  impl_->changed.wait(lock, [&] {
    return impl_->failure || current.block == block;
  });
  if (impl_->failure)
    std::rethrow_exception(impl_->failure);
  SLOPFAB_CUDA_CHECK(cudaStreamWaitEvent(nullptr, current.ready, 0));
  impl_->active = block;
  if (block + 1 < int(impl_->plans.size())) {
    impl_->pending = block + 1;
    impl_->changed.notify_all();
  }
}

void WeightStream::finish() {
  std::unique_lock<std::mutex> lock(impl_->mutex);
  if (!impl_->started)
    return;
  if (impl_->active >= 0) {
    auto& slot = impl_->slots[impl_->active % 2];
    SLOPFAB_CUDA_CHECK(cudaEventRecord(slot.released, nullptr));
    slot.release_recorded = true;
  }
  impl_->changed.wait(lock, [&] {
    return impl_->failure || (!impl_->busy && impl_->pending < 0);
  });
  if (impl_->failure)
    std::rethrow_exception(impl_->failure);
  lock.unlock();
  SLOPFAB_CUDA_CHECK(cudaStreamSynchronize(impl_->transfer));
  // Slots can remain allocated across segments; their reuse waits release events.
  lock.lock();
  impl_->started = false;
  impl_->active = -1;
}

WeightStream::View WeightStream::find(const std::string& name) const {
  if (impl_->active < 0)
    return {};
  const auto& plan = impl_->plans[impl_->active];
  auto found = plan.find(name);
  if (found == plan.end())
    return {};
  const auto& entry = found->second;
  return {reinterpret_cast<BF*>(impl_->slots[impl_->active % 2].data.get() + entry.offset),
          size_t(entry.tensor->numel())};
}

size_t WeightStream::device_bytes() const {
  return impl_->bytes;
}
} // namespace slopfab::seedvr2
