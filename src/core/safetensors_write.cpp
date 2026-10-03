// Minimal safetensors writer.
//
// Exists so intermediate activations can be dumped and fed to `slopfab compare`
// — both for regression-checking our own optimisations and, later, for diffing
// against tensors dumped from the reference implementation.

#include <cstdint>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <limits>
#include <set>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "slopfab/safetensors_write.h"

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace slopfab {
namespace {

std::string quote_json(const std::string& value) {
  const char* hex = "0123456789abcdef";
  std::string result = "\"";
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') {
      result += '\\';
      result += static_cast<char>(c);
    } else if (c < 32) {
      result += "\\u00";
      result += hex[c >> 4];
      result += hex[c & 15];
    } else
      result += static_cast<char>(c);
  }
  return result + '"';
}

std::string shape_to_json(const std::vector<int64_t>& shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0)
      out += ",";
    out += std::to_string(shape[i]);
  }
  out += "]";
  return out;
}

} // namespace

void write_safetensors(const std::string& path, const std::vector<TensorWrite>& tensors,
                       const std::map<std::string, std::string>& metadata) {
  // Header first: offsets are relative to the start of the data block, so the
  // whole layout is known before anything is written.
  std::string header = "{";
  size_t offset = 0;
  bool first = true;
  if (!metadata.empty()) {
    header += "\"__metadata__\":{";
    for (const auto& entry : metadata) {
      if (!first)
        header += ',';
      first = false;
      header += quote_json(entry.first) + ':' + quote_json(entry.second);
    }
    header += '}';
  }
  std::set<std::string> names;
  for (const TensorWrite& t : tensors) {
    if (t.name == "__metadata__" || !names.insert(t.name).second)
      throw std::invalid_argument("safetensors write: reserved or duplicate tensor name");
    size_t elems = 1;
    for (int64_t d : t.shape) {
      if (d < 0 || (d && elems > std::numeric_limits<size_t>::max() / uint64_t(d)))
        throw std::invalid_argument("safetensors write: invalid or excessive shape");
      elems *= static_cast<size_t>(d);
    }
    if (t.dtype != DType::kF32 && t.dtype != DType::kBF16 && t.dtype != DType::kF16 &&
        t.dtype != DType::kI32)
      throw std::runtime_error("safetensors write: only F32, F16, BF16 and I32 are supported");
    if (elems > (std::numeric_limits<size_t>::max() - offset) / dtype_size(t.dtype))
      throw std::overflow_error("safetensors write: archive too large");
    const size_t bytes = elems * dtype_size(t.dtype);
    const bool integer = t.dtype == DType::kI32;
    if ((integer ? t.integers.size() : t.data.size()) != elems ||
        !(integer ? t.data.empty() : t.integers.empty())) {
      throw std::runtime_error("safetensors write: tensor '" + t.name + "' has " +
                               std::to_string(integer ? t.integers.size() : t.data.size()) +
                               " values but shape implies " + std::to_string(elems));
    }
    if (!first)
      header += ",";
    first = false;
    header += quote_json(t.name) + ":{\"dtype\":\"" + dtype_name(t.dtype) +
              "\",\"shape\":" + shape_to_json(t.shape) + ",\"data_offsets\":[" +
              std::to_string(offset) + "," + std::to_string(offset + bytes) + "]}";
    offset += bytes;
  }
  header += "}";

  // The spec requires the data block to start 8-byte aligned; pad the header
  // with spaces, which JSON ignores.
  while ((8 + header.size()) % 8 != 0)
    header += " ";

  std::ofstream out(std::filesystem::u8path(path), std::ios::binary | std::ios::trunc);
  if (!out)
    throw std::runtime_error("safetensors write: cannot open " + path);

  const uint64_t header_len = header.size();
  out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  for (const TensorWrite& t : tensors) {
    if (t.dtype == DType::kI32) {
      out.write(reinterpret_cast<const char*>(t.integers.data()),
                static_cast<std::streamsize>(t.integers.size() * sizeof(int32_t)));
    } else if (t.dtype == DType::kF32) {
      out.write(reinterpret_cast<const char*>(t.data.data()),
                static_cast<std::streamsize>(t.data.size() * sizeof(float)));
    } else {
      std::vector<uint16_t> bits(t.data.size());
      for (size_t i = 0; i < bits.size(); ++i)
        bits[i] = t.dtype == DType::kBF16 ? f32_to_bf16(t.data[i]) : f32_to_f16(t.data[i]);
      out.write(reinterpret_cast<const char*>(bits.data()),
                static_cast<std::streamsize>(bits.size() * sizeof(uint16_t)));
    }
  }
  out.flush();
  if (!out)
    throw std::runtime_error("safetensors write: failed writing " + path);
  out.close();
  if (!out)
    throw std::runtime_error("safetensors write: failed closing " + path);
}

void write_safetensors_atomic(const std::string& path, const std::vector<TensorWrite>& tensors,
                              const std::map<std::string, std::string>& metadata) {
  if (path.empty())
    throw std::invalid_argument("safetensors write: empty output path");
  static std::atomic<uint64_t> serial{0};
  const auto target = std::filesystem::u8path(path);
  auto temporary = target;
#ifdef _WIN32
  const auto process = GetCurrentProcessId();
#else
  const auto process = getpid();
#endif
  temporary += ".pending-" + std::to_string(process) + "-" +
               std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
               std::to_string(serial.fetch_add(1));

  struct Cleanup {
    std::filesystem::path path;

    ~Cleanup() {
      std::error_code ec;
      std::filesystem::remove(path, ec);
    }
  } cleanup{temporary};

  write_safetensors(temporary.u8string(), tensors, metadata);
#ifdef _WIN32
  if (!MoveFileExW(temporary.c_str(), target.c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
    throw std::runtime_error("safetensors write: cannot replace archive (Windows error " +
                             std::to_string(GetLastError()) + ")");
#else
  std::filesystem::rename(temporary, target);
#endif
}

} // namespace slopfab
