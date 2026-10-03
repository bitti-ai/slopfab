// Safetensors writer for activations and portable conditioning archives.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "slopfab/dtype.h"

namespace slopfab {

struct TensorWrite {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<float> data;
  DType dtype = DType::kF32;
  // Used instead of data when dtype is I32 (e.g. modality tags).
  std::vector<int32_t> integers = {};
};

// Supports F32, F16, BF16 and I32. Validates all tensors before opening output.
void write_safetensors(const std::string& path, const std::vector<TensorWrite>& tensors,
                       const std::map<std::string, std::string>& metadata = {});

// Writes beside the destination then atomically replaces it on success.
void write_safetensors_atomic(const std::string& path, const std::vector<TensorWrite>& tensors,
                              const std::map<std::string, std::string>& metadata = {});

} // namespace slopfab
