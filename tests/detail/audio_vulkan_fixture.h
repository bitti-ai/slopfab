#pragma once

// Private shared fixtures for the audio_vulkan suites.
#include "../harness.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <bcrypt.h>
#endif

#include "slopfab/cuda/audio_vae_kernels.cuh"
#include "slopfab/cuda/device.h"
#include "slopfab/audio/wav.h"
#include "slopfab/generate.h"
#include "slopfab/safetensors.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/vae/audio_decoder.h"
#include "slopfab/vae/audio_primitives.h"
#include "slopfab/video/y4m.h"
#include "slopfab/vulkan/audio_decoder.h"
#include "slopfab/vulkan/tensor.h"


namespace {
namespace {

slopfab::TensorLayout layout(std::initializer_list<uint64_t> extents) {
  std::vector<uint64_t> shape(extents);
  return slopfab::TensorLayout::contiguous(shape.data(),
                                          static_cast<uint32_t>(shape.size()));
}

void check_exact(const std::vector<float>& cuda_values,
                 const std::vector<float>& vulkan_values,
                 const char* operation) {
  CHECK(cuda_values.size() == vulkan_values.size());
  for (size_t i = 0; i < cuda_values.size(); ++i) {
    uint32_t cuda_bits = 0, vulkan_bits = 0;
    std::memcpy(&cuda_bits, &cuda_values[i], sizeof(cuda_bits));
    std::memcpy(&vulkan_bits, &vulkan_values[i], sizeof(vulkan_bits));
    CHECK_MSG(cuda_bits == vulkan_bits,
              "%s CUDA/Vulkan mismatch at %zu: %08x != %08x", operation,
              i, cuda_bits, vulkan_bits);
  }
}

std::vector<float> values(size_t count, uint32_t multiplier, uint32_t modulus,
                          float scale) {
  std::vector<float> result(count);
  for (size_t i = 0; i < count; ++i)
    result[i] = float(int((i * multiplier) % modulus) - int(modulus / 2)) * scale;
  return result;
}

size_t count_subnormals(const std::vector<float>& values) {
  size_t count = 0;
  for (float value : values) {
    uint32_t bits = 0; std::memcpy(&bits, &value, sizeof(bits));
    count += (bits & 0x7f800000u) == 0u && (bits & 0x007fffffu) != 0u;
  }
  return count;
}

uint64_t fnv64(const std::vector<std::vector<float>>& tensors) {
  uint64_t hash = 1469598103934665603ull;
  for (const auto& tensor : tensors) {
    for (float value : tensor) {
      uint32_t bits = 0; std::memcpy(&bits, &value, sizeof(bits));
      for (unsigned byte = 0; byte < 4; ++byte) {
        hash ^= (bits >> (8u * byte)) & 0xffu;
        hash *= 1099511628211ull;
      }
    }
  }
  return hash;
}

uint64_t fnv64_floats(const float* data, size_t count) {
  uint64_t hash = 1469598103934665603ull;
  for (size_t i = 0; i < count; ++i) {
    uint32_t bits = 0;
    std::memcpy(&bits, &data[i], sizeof(bits));
    for (unsigned byte = 0; byte < 4; ++byte) {
      hash ^= (bits >> (8u * byte)) & 0xffu;
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

uint64_t fnv64_bytes(const std::vector<uint8_t>& data) {
  uint64_t hash = 1469598103934665603ull;
  for (uint8_t byte : data) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  return hash;
}

struct CapturedGeneration {
  int channels = 0;
  int frames = 0;
  int height = 0;
  int width = 0;
  int audio_channels = 0;
  int audio_sample_rate = 0;
  slopfab::PixelBuffer video;
  std::vector<float> audio;
};

bool capture_generation(slopfab::RunSamples& samples, void* userdata) {
  auto& captured = *static_cast<CapturedGeneration*>(userdata);
  captured.channels = samples.channels;
  captured.frames = samples.frames;
  captured.height = samples.height;
  captured.width = samples.width;
  captured.audio_channels = samples.audio_channels;
  captured.audio_sample_rate = samples.audio_sample_rate;
  captured.video = std::move(*samples.video);
  if (samples.audio != nullptr) captured.audio = std::move(*samples.audio);
  return true;
}

std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(stream), {});
}

#ifdef _WIN32
std::array<uint8_t, 32> mapping_sha256(const void* mapping, size_t bytes) {
  BCRYPT_ALG_HANDLE algorithm = nullptr;
  BCRYPT_HASH_HANDLE hash = nullptr;
  std::array<uint8_t, 32> digest{};
  auto fail = [&] {
    if (hash) BCryptDestroyHash(hash);
    if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
    throw std::runtime_error("audio primitive checkpoint SHA-256 failed");
  };
  if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM,
                                  nullptr, 0) < 0 ||
      BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) < 0) fail();
  const auto* cursor = static_cast<const uint8_t*>(mapping);
  while (bytes != 0) {
    const ULONG chunk = static_cast<ULONG>(std::min<size_t>(bytes, 64ull << 20));
    if (BCryptHashData(hash, const_cast<PUCHAR>(cursor), chunk, 0) < 0) fail();
    cursor += chunk; bytes -= chunk;
  }
  if (BCryptFinishHash(hash, digest.data(),
                       static_cast<ULONG>(digest.size()), 0) < 0) fail();
  BCryptDestroyHash(hash); BCryptCloseAlgorithmProvider(algorithm, 0);
  return digest;
}
#endif

}  // namespace

}  // namespace
