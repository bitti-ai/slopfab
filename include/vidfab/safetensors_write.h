// Minimal fp32 safetensors writer, for dumping activations to compare.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vidfab {

struct TensorWrite {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<float> data;
};

// Writes `tensors` as an fp32 safetensors archive. Throws if any tensor's
// data length disagrees with its declared shape.
void write_safetensors(const std::string& path, const std::vector<TensorWrite>& tensors);

}  // namespace vidfab
