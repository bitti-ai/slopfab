// Minimal safetensors writer.
//
// Exists so intermediate activations can be dumped and fed to `vidfab compare`
// — both for regression-checking our own optimisations and, later, for diffing
// against tensors dumped from the reference implementation.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vidfab/safetensors_write.h"

namespace vidfab {
namespace {

std::string shape_to_json(const std::vector<int64_t>& shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) out += ",";
    out += std::to_string(shape[i]);
  }
  out += "]";
  return out;
}

}  // namespace

void write_safetensors(const std::string& path, const std::vector<TensorWrite>& tensors) {
  // Header first: offsets are relative to the start of the data block, so the
  // whole layout is known before anything is written.
  std::string header = "{";
  size_t offset = 0;
  bool first = true;
  for (const TensorWrite& t : tensors) {
    int64_t elems = 1;
    for (int64_t d : t.shape) elems *= d;
    if (t.dtype != DType::kF32 && t.dtype != DType::kBF16 &&
        t.dtype != DType::kF16)
      throw std::runtime_error(
          "safetensors write: only F32, F16 and BF16 fixtures are supported");
    const size_t bytes = static_cast<size_t>(elems) * dtype_size(t.dtype);
    if (t.data.size() != static_cast<size_t>(elems)) {
      throw std::runtime_error("safetensors write: tensor '" + t.name + "' has " +
                               std::to_string(t.data.size()) + " values but shape implies " +
                               std::to_string(elems));
    }
    if (!first) header += ",";
    first = false;
    header += "\"" + t.name + "\":{\"dtype\":\"" + dtype_name(t.dtype) +
              "\",\"shape\":" + shape_to_json(t.shape) +
              ",\"data_offsets\":[" + std::to_string(offset) + "," +
              std::to_string(offset + bytes) + "]}";
    offset += bytes;
  }
  header += "}";

  // The spec requires the data block to start 8-byte aligned; pad the header
  // with spaces, which JSON ignores.
  while ((8 + header.size()) % 8 != 0) header += " ";

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("safetensors write: cannot open " + path);

  const uint64_t header_len = header.size();
  out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  for (const TensorWrite& t : tensors) {
    if (t.dtype == DType::kF32) {
      out.write(reinterpret_cast<const char*>(t.data.data()),
                static_cast<std::streamsize>(t.data.size() * sizeof(float)));
    } else {
      std::vector<uint16_t> bits(t.data.size());
      for (size_t i = 0; i < bits.size(); ++i)
        bits[i] = t.dtype == DType::kBF16 ? f32_to_bf16(t.data[i])
                                          : f32_to_f16(t.data[i]);
      out.write(reinterpret_cast<const char*>(bits.data()),
                static_cast<std::streamsize>(bits.size() * sizeof(uint16_t)));
    }
  }
  if (!out) throw std::runtime_error("safetensors write: failed writing " + path);
}

}  // namespace vidfab
