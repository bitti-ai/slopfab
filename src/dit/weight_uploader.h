#pragma once

#include "slopfab/cuda/device.h"

#include <algorithm>
#include <cstring>

namespace slopfab::dit {

// Bounded, double-buffered staging for transformed tensors and checkpoints
// whose mapping could not be page-locked. Small tensors share a slot: assigning
// an entire slot to each norm/bias forces the CPU to wait for preceding large
// direct transfers every two tensors.
class Uploader {
public:
  Uploader(cudaStream_t stream, const cuda::RegisteredMapping* lock,
           size_t stage_bytes = 32u << 20)
      : stream_(stream), lock_(lock), stage_bytes_(stage_bytes) {
    if (stage_bytes == 0)
      throw std::invalid_argument("transformer upload: empty staging slot");
  }

  ~Uploader() {
    // Scratch and the registered mapping must outlive this synchronization.
    // Best-effort here; load() checks stream completion before using weights.
    cudaStreamSynchronize(stream_);
    for (auto event : event_)
      if (event)
        cudaEventDestroy(event);
  }

  Uploader(const Uploader&) = delete;
  Uploader& operator=(const Uploader&) = delete;

  void copy(void* dst, const void* src, size_t bytes, bool from_mapping) {
    // Provenance is required in addition to the range check: temporary host
    // conversion buffers may be overwritten as soon as this call returns.
    if (from_mapping && lock_ != nullptr && lock_->contains(src, bytes)) {
      SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream_));
      return;
    }

    const auto* source = static_cast<const uint8_t*>(src);
    auto* target = static_cast<uint8_t*>(dst);
    while (bytes > 0) {
      if (used_ == stage_bytes_) {
        // Record once per full slot, after its last read, rather than once per
        // tensor. Reusing the other slot waits only when it has prior readers.
        SLOPFAB_CUDA_CHECK(cudaEventRecord(event_[current_], stream_));
        current_ ^= 1;
        used_ = 0;
        if (event_[current_])
          SLOPFAB_CUDA_CHECK(cudaEventSynchronize(event_[current_]));
      }
      if (!event_[current_]) {
        // Most loads need only one slot for converted norms. Allocate the
        // second only for a large conversion or an unregistered checkpoint.
        slot_[current_].allocate(stage_bytes_);
        SLOPFAB_CUDA_CHECK(cudaEventCreateWithFlags(&event_[current_], cudaEventDisableTiming));
      }
      const size_t count = std::min(bytes, stage_bytes_ - used_);
      auto* staged = slot_[current_].get() + used_;
      std::memcpy(staged, source, count);
      SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(target, staged, count, cudaMemcpyHostToDevice, stream_));
      used_ += count;
      source += count;
      target += count;
      bytes -= count;
    }
  }

private:
  cudaStream_t stream_;
  const cuda::RegisteredMapping* lock_;
  size_t stage_bytes_, used_ = 0;
  cuda::PinnedBuffer<uint8_t> slot_[2];
  cudaEvent_t event_[2] = {nullptr, nullptr};
  int current_ = 0;
};

} // namespace slopfab::dit
