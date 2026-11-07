#include "network.h"
#include "slopfab/safetensors.h"
#include "slopfab/sha256.h"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace slopfab::syncnet {
namespace {
struct Entry {
  const char* name;
  size_t offset;
  std::vector<int> shape;
};
#include "syncnet_manifest.inc"
#include "s3fd_manifest.inc"
std::string hex(const Sha256Digest& d) {
  std::string result;
  for (auto v : d) {
    result += "0123456789abcdef"[v >> 4];
    result += "0123456789abcdef"[v & 15];
  }
  return result;
}
}

Weights load_weights(const std::string& path, bool detector) {
  const Entry* entries = detector ? s3fd_entries : syncnet_entries;
  const size_t count = detector ? std::size(s3fd_entries) : std::size(syncnet_entries);
  const size_t expected_size = detector ? s3fd_size : syncnet_size;
  const char* expected_hash = detector ? s3fd_hash : syncnet_hash;
  std::vector<char> bytes;
  SafeTensors safe;
  if (detector && std::filesystem::u8path(path).extension() == ".safetensors") {
    safe.open(path);
    if (safe.tensor_count() != count || safe.file_size() > 128 * 1024 * 1024)
      throw std::runtime_error("S3FD: unexpected safetensors architecture or size");
  } else {
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != std::streamoff(expected_size))
      throw std::runtime_error(std::string(detector ? "S3FD" : "SyncNet") +
                               ": checkpoint must match SHA256 " + expected_hash);
    bytes.resize(expected_size);
    file.seekg(0);
    file.read(bytes.data(), std::streamsize(bytes.size()));
    if (!file || hex(sha256_bytes(bytes.data(), bytes.size())) != expected_hash)
      throw std::runtime_error(std::string(detector ? "S3FD" : "SyncNet") +
                               ": checkpoint must match SHA256 " + expected_hash);
  }
  const uint16_t endian = 1;
  if (*reinterpret_cast<const uint8_t*>(&endian) != 1)
    throw std::runtime_error("SyncNet requires a little-endian host");
  Weights result;
  for (size_t i = 0; i < count; ++i) {
    const auto& e = entries[i];
    Tensor t;
    t.shape = e.shape;
    size_t n = 1;
    for (int d : e.shape)
      n *= size_t(d);
    t.data.resize(n);
    if (safe.is_open()) {
      const auto& source = safe.at(e.name);
      if (source.dtype != DType::kF32 ||
          source.shape != std::vector<int64_t>(e.shape.begin(), e.shape.end()) ||
          source.nbytes != n * sizeof(float))
        throw std::runtime_error("S3FD: expected FP32 tensor with standard shape: " +
                                 std::string(e.name));
      std::memcpy(t.data.data(), source.data, n * sizeof(float));
    } else {
      if (e.offset > bytes.size() || n * sizeof(float) > bytes.size() - e.offset)
        throw std::logic_error("SyncNet: invalid compiled tensor range");
      std::memcpy(t.data.data(), bytes.data() + e.offset, n * sizeof(float));
    }
    for (float v : t.data)
      if (!std::isfinite(v))
        throw std::runtime_error("S3FD: non-finite weight: " + std::string(e.name));
    result.emplace(e.name, std::move(t));
  }
  return result;
}
} // namespace slopfab::syncnet
