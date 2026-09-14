// Read-only safetensors reader backed by a memory mapping.
//
// The file is mapped rather than read so that a 20 GB checkpoint costs no
// resident memory until tensors are actually touched, and so upload to the
// device can stream straight from the mapping.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "slopfab/dtype.h"

namespace slopfab {

struct TensorView {
  std::string name;
  DType dtype = DType::kUnknown;
  std::vector<int64_t> shape;
  const void* data = nullptr;  // into the mapping; valid while the file lives
  size_t nbytes = 0;

  // Element count implied by `shape`. For packed 4-bit tensors this is the
  // count of *stored bytes* worth of elements, not logical values, because the
  // trailing dimension is already halved on disk.
  int64_t numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
  }
  bool is_scalar() const { return shape.empty(); }
};

class SafeTensors {
 public:
  SafeTensors() = default;
  ~SafeTensors();

  SafeTensors(const SafeTensors&) = delete;
  SafeTensors& operator=(const SafeTensors&) = delete;
  SafeTensors(SafeTensors&& other) noexcept;
  SafeTensors& operator=(SafeTensors&& other) noexcept;

  // Maps `path` and parses the header. Throws std::runtime_error on a missing
  // file, malformed header, or any tensor whose byte range escapes the file.
  void open(const std::string& path);
  void close();

  bool is_open() const { return base_ != nullptr; }
  const std::string& path() const { return path_; }
  size_t file_size() const { return size_; }

  // Base of the mapping. Exposed so a caller can page-lock the whole range
  // with `cudaHostRegister` and then DMA tensors straight out of it, which
  // avoids staging every weight through a host copy first — worth 5x on the
  // text encoder's streaming path, where the memcpy was 96% of the time.
  // Null when closed.
  const void* mapping_base() const { return base_; }

  // Asks the OS to read the whole mapping in, asynchronously, instead of
  // waiting for it to be demanded a fault at a time. Best-effort in exactly the
  // sense `cudaHostRegister` is on the conditioner's mapping: it is a hint, it
  // can fail, and every caller is correct without it — just slower.
  //
  // This is worth 3x on a cold load and it is not the drive. The same 12.5 GB
  // nvfp4 checkpoint, same session, same access order, cache evicted before
  // each sample: 20.8 s demand-faulted against 7.3 s prefetched, where an
  // unbuffered sequential read of the same file is 5.8-7.1 s. Demand faulting a
  // mapping is a synchronous, one-outstanding-request-at-a-time walk; the drive
  // is an NVMe SSD that needs depth to reach its rate and never gets any.
  //
  // Call it only where the whole file is about to be consumed. `inspect` and
  // `compare` read the header and a few tensors, and would pay 12 GB of I/O for
  // nothing.
  //
  // Returns whether the hint was accepted, for logging; ignoring it is fine.
  bool prefetch() const;

  // The same hint over one byte range of the mapping, for a consumer that reads
  // a contiguous slice rather than the whole file. The Qwen vision tower is
  // 1.19 GB of a 27 GB conditioner: whole-file prefetch would pull 22x the
  // bytes it needs, while demand faulting it costs ~290k serialised 4 KB
  // faults.
  //
  // `begin` must point into the mapping. The range is clamped to the mapping
  // and an empty or out-of-range one is a no-op returning false, because this
  // is a hint and refusing to guess is better than hinting at someone else's
  // memory. Both back ends already take a range — Win32
  // `PrefetchVirtualMemory` a `WIN32_MEMORY_RANGE_ENTRY`, POSIX `madvise` an
  // address and a length — so this is the general form and `prefetch()` is the
  // whole-file case of it.
  bool prefetch_range(const void* begin, size_t bytes) const;

  // Byte extent of every tensor whose name starts with `prefix`, as
  // `[begin, begin + bytes)` into the mapping. `bytes` is zero when nothing
  // matches. Written for `prefetch_range`: it is a bounding extent, not a
  // promise that the range holds only matching tensors.
  void prefix_extent(std::string_view prefix, const void** begin, size_t* bytes) const;

  // Free-form key/value block stored under "__metadata__". Absent in most
  // checkpoints; ComfyUI writes provenance here.
  const std::map<std::string, std::string>& metadata() const { return metadata_; }

  const std::map<std::string, TensorView>& tensors() const { return tensors_; }
  size_t tensor_count() const { return tensors_.size(); }

  // Returns nullptr when absent. Archives uniformly wrapped in ComfyUI's
  // `model.diffusion_model.` namespace also accept unprefixed lookup names.
  // TensorView::name and tensors() retain the original names from disk.
  const TensorView* find(std::string_view name) const;

  // Throws when absent, naming the tensor. Use where a missing weight is a
  // structural error rather than an optional feature.
  const TensorView& at(std::string_view name) const;

 private:
  std::string path_;
  void* base_ = nullptr;  // start of the mapping
  size_t size_ = 0;
  std::map<std::string, TensorView> tensors_;
  std::map<std::string, std::string> metadata_;
  std::string lookup_prefix_;

#ifdef _WIN32
  void* file_handle_ = nullptr;
  void* mapping_handle_ = nullptr;
#else
  int fd_ = -1;
#endif

  void parse_header();
};

}  // namespace slopfab
