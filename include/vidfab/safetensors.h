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

#include "vidfab/dtype.h"

namespace vidfab {

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

  // Free-form key/value block stored under "__metadata__". Absent in most
  // checkpoints; ComfyUI writes provenance here.
  const std::map<std::string, std::string>& metadata() const { return metadata_; }

  const std::map<std::string, TensorView>& tensors() const { return tensors_; }
  size_t tensor_count() const { return tensors_.size(); }

  // Returns nullptr when absent.
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

#ifdef _WIN32
  void* file_handle_ = nullptr;
  void* mapping_handle_ = nullptr;
#else
  int fd_ = -1;
#endif

  void parse_header();
};

}  // namespace vidfab
