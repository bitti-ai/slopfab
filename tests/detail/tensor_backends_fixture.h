#pragma once

// Private shared fixtures for the tensor_backends suites.
#include "../harness.h"
#include "../allocation_guard.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winioctl.h>
#include <bcrypt.h>
#endif

#include "slopfab/cuda/device.h"
#include "slopfab/cuda/attention.cuh"
#include "slopfab/cuda/deterministic_math.cuh"
#include "slopfab/cuda/deterministic_gemm.cuh"
#include "slopfab/cuda/deterministic_attention.cuh"
#include "slopfab/cuda/gemm.cuh"
#include "slopfab/cuda/linear.cuh"
#include "slopfab/cuda/keyframe_encoder.cuh"
#include "slopfab/cuda/nn_kernels.cuh"
#include "slopfab/cuda/qwen_vision.cuh"
#include "slopfab/cuda/vae_kernels.cuh"
#include "slopfab/cuda/vae_vit_block.h"
#include "slopfab/attention.h"
#include "slopfab/dit/rope.h"
#include "slopfab/dit/denoise.h"
#include "slopfab/dit/transformer.h"
#include "slopfab/dit/block_capture.h"
#include "slopfab/dit/graph_capture.h"
#include "slopfab/dit/packing.h"
#include "slopfab/dit/ref2va.h"
#include "slopfab/dtype.h"
#include "slopfab/generate.h"
#include "slopfab/nf4.h"
#include "slopfab/safetensors.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/sha256.h"
#include "slopfab/sol_capture.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/text/encoder.h"
#include "slopfab/text/layer_capture.h"
#include "slopfab/vulkan/linear.h"
#include "slopfab/vulkan/keyframe_encoder.h"
#include "slopfab/vulkan/dit_block.h"
#include "slopfab/vulkan/dit_graph.h"
#include "slopfab/vulkan/dit_transformer.h"
#include "slopfab/vulkan/dit_denoise.h"
#include "slopfab/vulkan/gemm.h"
#include "slopfab/vulkan/tensor.h"
#include "slopfab/vulkan/text_layer.h"
#include "slopfab/vulkan/text_encoder.h"
#include "slopfab/vulkan/vision_stage.h"
#include "slopfab/vulkan/vae_vit_block.h"
#include "slopfab/vulkan/vae_decoder.h"
#include "slopfab/vulkan/yuv_converter.h"
#include "slopfab/vae/vit_decoder.h"
#include "slopfab/vae/keyframe_encoder.h"
#include "slopfab/video/y4m.h"
#include "slopfab/video/y4m_compare.h"

namespace slopfab::cuda {
void launch_adaln_expand(const float*, const float*, const float*, float*,
                         int, int, int, int, int, cudaStream_t);
}

namespace {
#ifdef _WIN32
namespace {

std::array<uint8_t, 32> sha256_mapping(const void* data, size_t bytes) {
  BCRYPT_ALG_HANDLE algorithm = nullptr;
  BCRYPT_HASH_HANDLE hash = nullptr;
  std::array<uint8_t, 32> digest{};
  auto fail = [&] {
    if (hash) BCryptDestroyHash(hash);
    if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
    throw std::runtime_error("CNG SHA-256 failed");
  };
  if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM,
                                  nullptr, 0) < 0 ||
      BCryptCreateHash(algorithm, &hash, nullptr, 0, nullptr, 0, 0) < 0) {
    fail();
  }
  const auto* cursor = static_cast<const uint8_t*>(data);
  while (bytes != 0) {
    const ULONG chunk = static_cast<ULONG>(
        std::min<size_t>(bytes, 64ull << 20));
    if (BCryptHashData(hash, const_cast<PUCHAR>(cursor), chunk, 0) < 0) fail();
    cursor += chunk;
    bytes -= chunk;
  }
  if (BCryptFinishHash(hash, digest.data(),
                       static_cast<ULONG>(digest.size()), 0) < 0) fail();
  BCryptDestroyHash(hash);
  BCryptCloseAlgorithmProvider(algorithm, 0);
  return digest;
}

std::filesystem::path make_sparse_qwen_metadata_corruption(
    const slopfab::SafeTensors& source, slopfab::text::WeightFormat format,
    const std::string& corrupt_name, bool rank_one = false,
    bool zero_scalar = false, bool visual_shape = false) {
  static std::atomic<uint32_t> serial{0};
  const std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("slopfab_qwen_corrupt_" + std::to_string(GetCurrentProcessId()) + "_" +
       std::to_string(serial.fetch_add(1)) + ".safetensors");
  HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr,
                            CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE)
    throw std::runtime_error("cannot create sparse Qwen corruption fixture");
  auto close_and_fail = [&](const char* message) {
    CloseHandle(file);
    std::filesystem::remove(path);
    throw std::runtime_error(message);
  };
  DWORD ignored = 0;
  if (!DeviceIoControl(file, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0,
                       &ignored, nullptr)) {
    close_and_fail("cannot mark Qwen corruption fixture sparse");
  }
  LARGE_INTEGER end{};
  end.QuadPart = static_cast<LONGLONG>(source.file_size());
  if (!SetFilePointerEx(file, end, nullptr, FILE_BEGIN) || !SetEndOfFile(file))
    close_and_fail("cannot size Qwen corruption fixture");
  auto write_at = [&](uint64_t offset, const void* data, uint64_t bytes) {
    LARGE_INTEGER position{};
    position.QuadPart = static_cast<LONGLONG>(offset);
    if (!SetFilePointerEx(file, position, nullptr, FILE_BEGIN))
      close_and_fail("cannot seek Qwen corruption fixture");
    const auto* cursor = static_cast<const uint8_t*>(data);
    while (bytes != 0) {
      const DWORD chunk = static_cast<DWORD>(std::min<uint64_t>(bytes, 1u << 20));
      DWORD written = 0;
      if (!WriteFile(file, cursor, chunk, &written, nullptr) || written != chunk)
        close_and_fail("cannot write Qwen corruption fixture");
      cursor += chunk;
      bytes -= chunk;
    }
  };
  uint64_t json_bytes = 0;
  std::memcpy(&json_bytes, source.mapping_base(), sizeof(json_bytes));
  const uint64_t header_bytes = json_bytes + sizeof(json_bytes);
  std::vector<uint8_t> header(static_cast<size_t>(header_bytes));
  std::memcpy(header.data(), source.mapping_base(), header.size());
  if (rank_one || visual_shape) {
    std::string json(reinterpret_cast<const char*>(header.data() + 8),
                     static_cast<size_t>(json_bytes));
    const size_t tensor = json.find("\"" + corrupt_name + "\"");
    const char* source_shape = visual_shape
        ? "\"shape\":[1152,4304]" : "\"shape\":[]";
    const char* replacement = visual_shape
        ? "\"shape\":[4304,1152]" : "\"shape\":[1]";
    const size_t shape = tensor == std::string::npos
        ? std::string::npos : json.find(source_shape, tensor);
    if (shape == std::string::npos || json.empty() ||
        (!visual_shape && json.back() != ' '))
      close_and_fail("cannot mutate Qwen tensor shape in header");
    json.replace(shape, std::strlen(source_shape), replacement);
    if (!visual_shape) json.pop_back();
    if (json.size() != json_bytes)
      close_and_fail("Qwen scalar rank mutation changed header size");
    std::memcpy(header.data() + 8, json.data(), json.size());
  }
  write_at(0, header.data(), header.size());
  const auto* base = static_cast<const uint8_t*>(source.mapping_base());
  auto ends_with = [](const std::string& value, const char* suffix) {
    const size_t n = std::strlen(suffix);
    return value.size() >= n && value.compare(value.size() - n, n, suffix) == 0;
  };
  for (const auto& entry : source.tensors()) {
    const std::string& name = entry.first;
    const bool copy = ends_with(name, ".comfy_quant") ||
        (format == slopfab::text::WeightFormat::kNVFP4Awq &&
         (ends_with(name, ".weight_scale_2") ||
          name == "model.embed_tokens.weight_scale"));
    if (!copy) continue;
    const slopfab::TensorView& view = entry.second;
    const uint64_t offset = static_cast<const uint8_t*>(view.data) - base;
    if (name == corrupt_name && zero_scalar) {
      const float zero = 0.0f;
      write_at(offset, &zero, sizeof(zero));
    } else if (name == corrupt_name && !rank_one) {
      std::vector<uint8_t> corrupted(view.nbytes);
      std::memcpy(corrupted.data(), view.data, view.nbytes);
      corrupted.back() ^= 1u;
      write_at(offset, corrupted.data(), corrupted.size());
    } else {
      write_at(offset, view.data, view.nbytes);
    }
  }
  if (!CloseHandle(file)) {
    std::filesystem::remove(path);
    throw std::runtime_error("cannot close Qwen corruption fixture");
  }
  return path;
}

}  // namespace
#endif

__global__ void deterministic_rsqrt_probe(const float* input, float* stable,
                                           float* native, int count) {
  int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < count) {
    stable[index] = slopfab::cuda::deterministic_rsqrt(input[index]);
    native[index] = rsqrtf(input[index]);
  }
}

__global__ void deterministic_divide_add_probe(const uint32_t* input_bits,
                                                const uint32_t* divisors,
                                                const uint32_t* epsilon_bits,
                                                uint32_t* positive_divided,
                                                uint32_t* signed_divided,
                                                uint32_t* added, int count) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= count) return;
  const uint32_t magnitude = input_bits[index] & 0x7fffffffu;
  positive_divided[index] =
      slopfab::cuda::positive_float_div_uint(magnitude, divisors[index]);
  const float signed_value = __uint_as_float(input_bits[index]);
  signed_divided[index] = __float_as_uint(
      slopfab::cuda::deterministic_divide(signed_value, divisors[index]));
  added[index] = slopfab::cuda::positive_float_add(positive_divided[index],
                                                   epsilon_bits[index]);
}

__global__ void deterministic_silu_probe(const float* input, float* output,
                                         float* pointwise_output, int count) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < count) {
    output[index] = slopfab::cuda::deterministic_silu(input[index]);
    pointwise_output[index] = slopfab::cuda::deterministic_pointwise_silu(input[index]);
  }
}

// Test-only copy of the pre-rebaseline Video-VAE SwiGLU. This preserves the
// shipped __expf arithmetic solely for an apples-to-apples performance and
// output-drift measurement; production never calls this kernel.
__global__ void legacy_vae_swiglu_probe(const float* input, const float* bias,
                                        float* output, int inner) {
  const int column = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (column >= inner) return;
  const size_t row = blockIdx.y;
  const float* values = input + row * 2 * inner;
  const float gate = values[column] + bias[column];
  const float value = values[inner + column] + bias[inner + column];
  output[row * inner + column] =
      (gate / (1.0f + __expf(-gate))) * value;
}

__global__ void fp32_pre_quant_scale_probe(const float* input,
                                            const __nv_bfloat16* scale,
                                            float* output, int count, int dim) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < count) output[index] = input[index] * __bfloat162float(scale[index % dim]);
}

__global__ void dense_weight_convert_probe(const void* input, uint16_t* output,
                                            int count, int source_type,
                                            bool output_fp16) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= count) return;
  float value = source_type == 0 ? static_cast<const float*>(input)[index]
      : source_type == 1 ? __half2float(reinterpret_cast<const __half*>(input)[index])
                         : __bfloat162float(
                               reinterpret_cast<const __nv_bfloat16*>(input)[index]);
  if (output_fp16) {
    reinterpret_cast<__half*>(output)[index] = __float2half_rn(value);
  } else {
    reinterpret_cast<__nv_bfloat16*>(output)[index] = __float2bfloat16_rn(value);
  }
}

}  // namespace
