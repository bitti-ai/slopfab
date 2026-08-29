#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace vidfab {

using Sha256Digest = std::array<uint8_t, 32>;

// Portable SHA-256 used for durable capture/checkpoint provenance. The file
// variant streams bounded chunks and never maps or duplicates the whole file.
Sha256Digest sha256_bytes(const void* data, size_t bytes);
Sha256Digest sha256_file(const std::string& path);

}  // namespace vidfab
