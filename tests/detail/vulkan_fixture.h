#pragma once

// Private shared fixtures for the vulkan suites.
#include "../harness.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "slopfab/vulkan/runtime.h"
#include "slopfab/vulkan/compute.h"
#include "slopfab/vulkan/gemm.h"
#include "slopfab/vulkan/linear.h"
#include "slopfab/vulkan/tensor.h"
#include "slopfab/vulkan/audio_decoder.h"
#include "slopfab/vulkan/dit_block.h"
#include "slopfab/vulkan/dit_denoise.h"
#include "slopfab/vulkan/dit_graph.h"
#include "slopfab/vulkan/dit_transformer.h"
#include "slopfab/vulkan/text_layer.h"
#include "slopfab/vulkan/text_encoder.h"
#include "slopfab/vulkan/vae_decoder.h"
#include "slopfab/vulkan/yuv_converter.h"
#include "slopfab/attention.h"
#include "slopfab/dit/ref2va.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/sampler/scheduler.h"
#include "slopfab/video/y4m.h"
#include "slopfab/dtype.h"
#include "slopfab/safetensors.h"
#include "slopfab/sha256.h"
#include "slopfab/text/layer_capture.h"
#include "../../src/vulkan/tensor_validation.h"
#include "../../src/vulkan/sage_selection.h"


namespace {

std::vector<uint32_t> load_spirv(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error(std::string("cannot open SPIR-V: ") + path);
  const std::streamoff length = file.tellg();
  if (length <= 0 || (length % 4) != 0) throw std::runtime_error("invalid SPIR-V byte size");
  file.seekg(0);
  std::vector<uint32_t> words(static_cast<size_t>(length) / 4);
  file.read(reinterpret_cast<char*>(words.data()), length);
  if (!file) throw std::runtime_error("cannot read SPIR-V");
  return words;
}

std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
}

uint32_t float_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float float_from_bits(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint64_t fnv64_floats(const std::vector<float>& values) {
  uint64_t hash = 1469598103934665603ull;
  for (float value : values) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    for (unsigned byte = 0; byte < 4; ++byte) {
      hash ^= (bits >> (8u * byte)) & 0xffu;
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

uint64_t fnv64_bytes(const void* data, size_t bytes) {
  uint64_t hash = 1469598103934665603ull;
  const auto* cursor = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < bytes; ++i) {
    hash ^= cursor[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

uint16_t reference_bf16(float value) {
  uint32_t bits = float_bits(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return 0x7fffu;
  }
  return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

#if !defined(SLOPFAB_WITH_CUDA) || !SLOPFAB_WITH_CUDA

#endif

}  // namespace
