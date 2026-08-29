#include "harness.h"
#include "allocation_guard.h"

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

#include "vidfab/cuda/device.h"
#include "vidfab/cuda/attention.cuh"
#include "vidfab/cuda/deterministic_math.cuh"
#include "vidfab/cuda/deterministic_gemm.cuh"
#include "vidfab/cuda/deterministic_attention.cuh"
#include "vidfab/cuda/gemm.cuh"
#include "vidfab/cuda/linear.cuh"
#include "vidfab/cuda/keyframe_encoder.cuh"
#include "vidfab/cuda/nn_kernels.cuh"
#include "vidfab/cuda/vae_kernels.cuh"
#include "vidfab/cuda/vae_vit_block.h"
#include "vidfab/attention.h"
#include "vidfab/dit/rope.h"
#include "vidfab/dit/denoise.h"
#include "vidfab/dit/transformer.h"
#include "vidfab/dit/block_capture.h"
#include "vidfab/dit/graph_capture.h"
#include "vidfab/dit/packing.h"
#include "vidfab/dtype.h"
#include "vidfab/generate.h"
#include "vidfab/nf4.h"
#include "vidfab/safetensors.h"
#include "vidfab/safetensors_write.h"
#include "vidfab/sha256.h"
#include "vidfab/sol_capture.h"
#include "vidfab/tensor_convert.h"
#include "vidfab/text/encoder.h"
#include "vidfab/text/layer_capture.h"
#include "vidfab/vulkan/linear.h"
#include "vidfab/vulkan/dit_block.h"
#include "vidfab/vulkan/dit_graph.h"
#include "vidfab/vulkan/dit_transformer.h"
#include "vidfab/vulkan/dit_denoise.h"
#include "vidfab/vulkan/gemm.h"
#include "vidfab/vulkan/tensor.h"
#include "vidfab/vulkan/text_layer.h"
#include "vidfab/vulkan/text_encoder.h"
#include "vidfab/vulkan/vae_vit_block.h"
#include "vidfab/vulkan/vae_decoder.h"
#include "vidfab/vulkan/yuv_converter.h"
#include "vidfab/vae/vit_decoder.h"
#include "vidfab/video/y4m.h"
#include "vidfab/video/y4m_compare.h"

namespace vidfab::cuda {
void launch_adaln_expand(const float*, const float*, const float*, float*,
                         int, int, int, int, int, cudaStream_t);
}

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
    const vidfab::SafeTensors& source, vidfab::text::WeightFormat format,
    const std::string& corrupt_name, bool rank_one = false,
    bool zero_scalar = false) {
  static std::atomic<uint32_t> serial{0};
  const std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("vidfab_qwen_corrupt_" + std::to_string(GetCurrentProcessId()) + "_" +
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
  if (rank_one) {
    std::string json(reinterpret_cast<const char*>(header.data() + 8),
                     static_cast<size_t>(json_bytes));
    const size_t tensor = json.find("\"" + corrupt_name + "\"");
    const size_t shape = tensor == std::string::npos
        ? std::string::npos : json.find("\"shape\":[]", tensor);
    if (shape == std::string::npos || json.empty() || json.back() != ' ')
      close_and_fail("cannot mutate Qwen scalar rank in header");
    json.replace(shape, std::strlen("\"shape\":[]"), "\"shape\":[1]");
    json.pop_back();
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
        (format == vidfab::text::WeightFormat::kNVFP4Awq &&
         (ends_with(name, ".weight_scale_2") ||
          name == "model.embed_tokens.weight_scale"));
    if (!copy) continue;
    const vidfab::TensorView& view = entry.second;
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
    stable[index] = vidfab::cuda::deterministic_rsqrt(input[index]);
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
      vidfab::cuda::positive_float_div_uint(magnitude, divisors[index]);
  const float signed_value = __uint_as_float(input_bits[index]);
  signed_divided[index] = __float_as_uint(
      vidfab::cuda::deterministic_divide(signed_value, divisors[index]));
  added[index] = vidfab::cuda::positive_float_add(positive_divided[index],
                                                   epsilon_bits[index]);
}

__global__ void deterministic_silu_probe(const float* input, float* output,
                                         float* pointwise_output, int count) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < count) {
    output[index] = vidfab::cuda::deterministic_silu(input[index]);
    pointwise_output[index] = vidfab::cuda::deterministic_pointwise_silu(input[index]);
  }
}

VIDFAB_TEST(cuda_deterministic_rsqrt_dense_reference) {
  using namespace vidfab;
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return;
  std::vector<uint32_t> bits = {
      0x00000000u, 0x80000000u, 0x00000001u, 0x007fffffu,
      0x00800000u, 0x3f800000u, 0x7f7fffffu, 0x7f800000u,
      0xff800000u, 0xbf800000u, 0x7fc12345u};
  uint32_t state = 0x12345678u;
  for (uint32_t exponent = 1; exponent < 255; ++exponent) {
    const uint32_t boundary[] = {0u, 1u, 0x003fffffu, 0x00400000u, 0x007ffffeu,
                                 0x007fffffu};
    for (uint32_t mantissa : boundary) bits.push_back(exponent << 23u | mantissa);
    for (int sample = 0; sample < 32; ++sample) {
      state = state * 1664525u + 1013904223u;
      bits.push_back(exponent << 23u | (state & 0x007fffffu));
    }
  }
  std::vector<float> input(bits.size()), stable(bits.size()), native(bits.size());
  std::memcpy(input.data(), bits.data(), bits.size() * sizeof(uint32_t));
  cuda::DeviceBuffer<float> d_input(input.size()), d_stable(input.size()),
      d_native(input.size());
  d_input.copy_from_host(input.data(), input.size());
  deterministic_rsqrt_probe<<<static_cast<unsigned>((input.size() + 255) / 256), 256>>>(
      d_input.get(), d_stable.get(), d_native.get(), static_cast<int>(input.size()));
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  d_stable.copy_to_host(stable.data(), stable.size());
  d_native.copy_to_host(native.data(), native.size());
  const struct { size_t index; uint32_t expected; } exceptional[] = {
      {0, 0x7f800000u}, {1, 0xff800000u}, {7, 0x00000000u},
      {8, 0x7fc00000u}, {9, 0x7fc00000u}, {10, 0x7fc00000u}};
  for (const auto& item : exceptional) {
    uint32_t actual = 0;
    std::memcpy(&actual, &stable[item.index], sizeof(actual));
    CHECK_MSG(actual == item.expected, "deterministic rsqrt exceptional %zu: %08x",
              item.index, actual);
  }
  uint32_t max_reference_ulp = 0;
  uint32_t max_native_ulp = 0;
  double max_relative_error = 0.0;
  for (size_t i = 2; i < bits.size(); ++i) {
    if ((bits[i] & 0x80000000u) != 0u || (bits[i] & 0x7f800000u) == 0x7f800000u)
      continue;
    const float reference = static_cast<float>(1.0 / std::sqrt(static_cast<double>(input[i])));
    uint32_t reference_bits = 0, stable_bits = 0, native_bits = 0;
    std::memcpy(&reference_bits, &reference, 4);
    std::memcpy(&stable_bits, &stable[i], 4);
    std::memcpy(&native_bits, &native[i], 4);
    max_reference_ulp = std::max(max_reference_ulp,
        reference_bits > stable_bits ? reference_bits - stable_bits : stable_bits - reference_bits);
    max_native_ulp = std::max(max_native_ulp,
        native_bits > stable_bits ? native_bits - stable_bits : stable_bits - native_bits);
    const double exact = 1.0 / std::sqrt(static_cast<double>(input[i]));
    max_relative_error = std::max(max_relative_error,
                                  std::abs(static_cast<double>(stable[i]) - exact) / exact);
  }
  CHECK_MSG(max_reference_ulp <= 1u, "deterministic rsqrt max reference error %u ULP",
            max_reference_ulp);
  CHECK_MSG(max_native_ulp <= 2u, "deterministic/native rsqrt max difference %u ULP",
            max_native_ulp);
  std::printf("  deterministic rsqrt: reference max %u ULP, native max delta %u ULP, "
              "relative %.3e\n", max_reference_ulp, max_native_ulp,
              max_relative_error);

  // Independent IEEE-RNE references for the integer division and restricted
  // norm-base addition. The curated cross product hits sign, tie/carry,
  // subnormal quotient, mantissa and exponent boundaries; randomized values
  // sample across the full legal divisor range through 2^24.
  const uint32_t curated_values[] = {
      0x00000001u, 0x007fffffu, 0x00800000u, 0x00800001u,
      0x3effffffu, 0x3f000000u, 0x3f000001u, 0x3f7fffffu,
      0x3f800000u, 0x3f800001u, 0x4b7fffffu, 0x7f7fffffu};
  const uint32_t curated_divisors[] = {
      1u, 2u, 3u, 5u, 7u, 9u, 127u, 255u, 257u, 65535u,
      0x00ffffffu, 0x01000000u};
  const uint32_t curated_epsilons[] = {
      0x00800000u, 0x33800000u, 0x358637bdu, 0x3f000000u, 0x3f800000u};
  std::vector<uint32_t> divide_inputs, divide_divisors, divide_epsilons;
  for (uint32_t value : curated_values) {
    for (uint32_t divisor : curated_divisors) {
      const size_t index = divide_inputs.size();
      divide_inputs.push_back(value | ((index & 1u) ? 0x80000000u : 0u));
      divide_divisors.push_back(divisor);
      divide_epsilons.push_back(curated_epsilons[index % std::size(curated_epsilons)]);
    }
  }
  state = 0x9e3779b9u;
  for (int sample = 0; sample < 8192; ++sample) {
    state = state * 1664525u + 1013904223u;
    const uint32_t exponent = 1u + state % 254u;
    const uint32_t value = exponent << 23u | (state & 0x007fffffu);
    state = state * 1664525u + 1013904223u;
    divide_inputs.push_back(value | ((state & 1u) << 31u));
    divide_divisors.push_back(1u + state % 0x01000000u);
    divide_epsilons.push_back(curated_epsilons[
        static_cast<size_t>(state) % std::size(curated_epsilons)]);
  }
  const size_t arithmetic_count = divide_inputs.size();
  cuda::DeviceBuffer<uint32_t> d_divide_inputs(arithmetic_count),
      d_divisors(arithmetic_count), d_epsilons(arithmetic_count),
      d_positive(arithmetic_count), d_signed(arithmetic_count), d_added(arithmetic_count);
  d_divide_inputs.copy_from_host(divide_inputs.data(), arithmetic_count);
  d_divisors.copy_from_host(divide_divisors.data(), arithmetic_count);
  d_epsilons.copy_from_host(divide_epsilons.data(), arithmetic_count);
  deterministic_divide_add_probe<<<static_cast<unsigned>((arithmetic_count + 255) / 256), 256>>>(
      d_divide_inputs.get(), d_divisors.get(), d_epsilons.get(), d_positive.get(),
      d_signed.get(), d_added.get(), static_cast<int>(arithmetic_count));
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint32_t> positive(arithmetic_count), signed_result(arithmetic_count),
      add_result(arithmetic_count);
  d_positive.copy_to_host(positive.data(), arithmetic_count);
  d_signed.copy_to_host(signed_result.data(), arithmetic_count);
  d_added.copy_to_host(add_result.data(), arithmetic_count);
  for (size_t i = 0; i < arithmetic_count; ++i) {
    const uint32_t magnitude_bits = divide_inputs[i] & 0x7fffffffu;
    float magnitude = 0.0f, signed_value = 0.0f, epsilon = 0.0f;
    std::memcpy(&magnitude, &magnitude_bits, sizeof(magnitude));
    std::memcpy(&signed_value, &divide_inputs[i], sizeof(signed_value));
    std::memcpy(&epsilon, &divide_epsilons[i], sizeof(epsilon));
    const float expected_positive = static_cast<float>(
        static_cast<double>(magnitude) / static_cast<double>(divide_divisors[i]));
    const float expected_signed = static_cast<float>(
        static_cast<double>(signed_value) / static_cast<double>(divide_divisors[i]));
    const float expected_add = static_cast<float>(
        static_cast<double>(expected_positive) + static_cast<double>(epsilon));
    uint32_t expected_positive_bits = 0, expected_signed_bits = 0, expected_add_bits = 0;
    std::memcpy(&expected_positive_bits, &expected_positive, 4);
    std::memcpy(&expected_signed_bits, &expected_signed, 4);
    std::memcpy(&expected_add_bits, &expected_add, 4);
    CHECK_MSG(positive[i] == expected_positive_bits,
              "RNE positive divide mismatch %zu: %08x != %08x", i,
              positive[i], expected_positive_bits);
    CHECK_MSG(signed_result[i] == expected_signed_bits,
              "RNE signed divide mismatch %zu: %08x != %08x", i,
              signed_result[i], expected_signed_bits);
    CHECK_MSG(add_result[i] == expected_add_bits,
              "RNE norm-base add mismatch %zu: %08x != %08x", i,
              add_result[i], expected_add_bits);
  }

}

VIDFAB_TEST(cuda_deterministic_silu_dense_reference) {
  using namespace vidfab;
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return;
  std::vector<float> input;
  auto push_bits = [&](uint32_t bits) {
    float value = 0.0f; std::memcpy(&value, &bits, 4); input.push_back(value);
  };
  const uint32_t special[] = {0x00000000u, 0x80000000u, 0x7f800000u,
                              0xff800000u, 0x7fc12345u};
  for (uint32_t bits : special) push_bits(bits);
  for (uint32_t bits : {0x00800000u, 0x80800000u, 0x00ffffffu,
                        0x80ffffffu, 0x01000000u, 0x81000000u,
                        0x01000001u, 0x81000001u}) push_bits(bits);
  for (float anchor : {-87.0f, 87.0f, -16.0f, 16.0f}) {
    input.push_back(std::nextafter(anchor, -std::numeric_limits<float>::infinity()));
    input.push_back(anchor);
    input.push_back(std::nextafter(anchor, std::numeric_limits<float>::infinity()));
  }
  constexpr double ln2 = 0.693147180559945309417232121458176568;
  for (int k = -125; k <= 125; ++k) {
    const float anchor = static_cast<float>(k * ln2);
    if (anchor < -87.0f || anchor > 87.0f) continue;
    input.push_back(std::nextafter(anchor, -std::numeric_limits<float>::infinity()));
    input.push_back(anchor);
    input.push_back(std::nextafter(anchor, std::numeric_limits<float>::infinity()));
  }
  uint32_t state = 0x31415926u;
  for (int i = 0; i < 65536; ++i) {
    state = state * 1664525u + 1013904223u;
    const double unit = static_cast<double>(state) / 4294967295.0;
    input.push_back(static_cast<float>(-87.0 + unit * 174.0));
  }
  cuda::DeviceBuffer<float> d_input(input.size()), d_output(input.size()),
      d_pointwise_output(input.size());
  d_input.copy_from_host(input.data(), input.size());
  deterministic_silu_probe<<<static_cast<unsigned>((input.size() + 255) / 256), 256>>>(
      d_input.get(), d_output.get(), d_pointwise_output.get(),
      static_cast<int>(input.size()));
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> output(input.size()), pointwise_output(input.size());
  d_output.copy_to_host(output.data(), output.size());
  d_pointwise_output.copy_to_host(pointwise_output.data(), pointwise_output.size());
  const uint32_t expected_special[] = {0x00000000u, 0x80000000u, 0x7f800000u,
                                       0x80000000u, 0x7fc00000u};
  for (size_t i = 0; i < std::size(expected_special); ++i) {
    uint32_t actual = 0; std::memcpy(&actual, &output[i], 4);
    CHECK_MSG(actual == expected_special[i], "deterministic SiLU special %zu: %08x", i,
              actual);
    uint32_t pointwise_actual = 0;
    std::memcpy(&pointwise_actual, &pointwise_output[i], 4);
    CHECK_MSG(pointwise_actual == expected_special[i],
              "pointwise SiLU special %zu: %08x", i, pointwise_actual);
  }
  auto ordered = [](float value) {
    uint32_t bits = 0; std::memcpy(&bits, &value, 4);
    return (bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u;
  };
  uint32_t max_ulp = 0, pointwise_max_ulp = 0;
  double max_absolute = 0.0, max_relative = 0.0;
  double pointwise_max_absolute = 0.0, pointwise_max_relative = 0.0;
  float worst_relative_input = 0.0f, worst_relative_output = 0.0f;
  double worst_relative_reference = 0.0;
  for (size_t i = std::size(expected_special); i < input.size(); ++i) {
    const double x = input[i];
    const double reference_double = x < 0.0
        ? x * std::exp(x) / (1.0 + std::exp(x))
        : x / (1.0 + std::exp(-x));
    if (x <= -87.0) {
      uint32_t actual = 0; std::memcpy(&actual, &output[i], 4);
      CHECK(actual == 0x80000000u);
      std::memcpy(&actual, &pointwise_output[i], 4);
      CHECK(actual == 0x80000000u);
      continue;
    }
    const float rounded_reference = static_cast<float>(reference_double);
    uint32_t rounded_reference_bits = 0;
    std::memcpy(&rounded_reference_bits, &rounded_reference, 4);
    if ((rounded_reference_bits & 0x7f800000u) == 0u &&
        (rounded_reference_bits & 0x007fffffu) != 0u) {
      uint32_t actual = 0, input_bits = 0;
      std::memcpy(&actual, &output[i], 4); std::memcpy(&input_bits, &input[i], 4);
      CHECK(actual == (input_bits & 0x80000000u));
      std::memcpy(&actual, &pointwise_output[i], 4);
      CHECK(actual == (input_bits & 0x80000000u));
      continue;
    }
    uint32_t input_bits = 0; std::memcpy(&input_bits, &input[i], 4);
    if ((input_bits & 0x7fffffffu) < 0x00800000u) {
      uint32_t actual = 0; std::memcpy(&actual, &output[i], 4);
      CHECK(actual == (input_bits & 0x80000000u));
      std::memcpy(&actual, &pointwise_output[i], 4);
      CHECK(actual == (input_bits & 0x80000000u));
      continue;
    }
    const float reference = rounded_reference;
    const uint32_t a = ordered(output[i]), b = ordered(reference);
    max_ulp = std::max(max_ulp, a > b ? a - b : b - a);
    const uint32_t pa = ordered(pointwise_output[i]);
    pointwise_max_ulp = std::max(pointwise_max_ulp,
                                 pa > b ? pa - b : b - pa);
    const double absolute = std::abs(static_cast<double>(output[i]) - reference_double);
    max_absolute = std::max(max_absolute, absolute);
    const double pointwise_absolute =
        std::abs(static_cast<double>(pointwise_output[i]) - reference_double);
    pointwise_max_absolute = std::max(pointwise_max_absolute, pointwise_absolute);
    if (reference_double != 0.0) {
      const double relative = absolute / std::abs(reference_double);
      const double pointwise_relative = pointwise_absolute / std::abs(reference_double);
      pointwise_max_relative = std::max(pointwise_max_relative, pointwise_relative);
      if (relative > max_relative) {
        max_relative = relative;
        worst_relative_input = input[i];
        worst_relative_output = output[i];
        worst_relative_reference = reference_double;
      }
    }
  }
  CHECK_MSG(max_ulp <= 3u, "deterministic SiLU max reference error %u ULP", max_ulp);
  CHECK(max_relative < 2.1e-7);
  CHECK_MSG(pointwise_max_ulp <= 3u,
            "pointwise deterministic SiLU max reference error %u ULP",
            pointwise_max_ulp);
  CHECK(pointwise_max_relative < 2.1e-7);
  const double cutoff_error = 87.0 * std::exp(-87.0) / (1.0 + std::exp(-87.0));
  CHECK(cutoff_error < 1.5e-36);
  std::printf("  deterministic SiLU: max %u ULP, abs %.3e, relative %.3e; "
              "-87 cutoff %.3e; worst relative x=%g out=%.9g ref=%.9g\n",
              max_ulp, max_absolute, max_relative, cutoff_error,
              worst_relative_input, worst_relative_output, worst_relative_reference);
  std::printf("  pointwise SiLU: max %u ULP, abs %.3e, relative %.3e\n",
              pointwise_max_ulp, pointwise_max_absolute,
              pointwise_max_relative);
}

VIDFAB_TEST(cuda_vulkan_tensor_exact_copy_and_add) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) {
    return;
  }
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  bool full_exact_rejected = false;
  try {
    vk.require_full_fp32_arithmetic_exactness();
  } catch (const std::runtime_error&) {
    full_exact_rejected = true;
  }
  CHECK(full_exact_rejected == !physical.front().info().fp32_denorm_preserve);
  CHECK(vk.full_fp32_arithmetic_exactness() ==
        physical.front().info().fp32_denorm_preserve);
  CHECK(vk.full_fp32_add_exactness() == vk.full_fp32_arithmetic_exactness());

  constexpr uint64_t count = 259;
  const TensorLayout layout = TensorLayout::contiguous(&count, 1);
  std::vector<float> a(count), b(count);
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = static_cast<float>(static_cast<int>(i % 41) - 20) / 32.0f;
    b[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 64.0f;
  }
  // Copy must preserve every bit pattern. Arithmetic includes signed zero,
  // subnormal and infinity inputs whose selected sums have stable exact bits
  // on both APIs. NaN payload preservation is covered by copy: GLSL fp32 add
  // is permitted to canonicalise a NaN and is therefore not advertised as an
  // exact-payload arithmetic operation.
  const uint32_t copy_special[] = {
      0x00000000u, 0x80000000u, 0x00000001u, 0x00800000u,
      0x3f800000u, 0x3f800000u, 0x7f800000u, 0xff800000u,
      0x7fc12345u, 0x7fa54321u};
  const uint32_t add_rhs[] = {
      0x80000000u, 0x80000000u, 0x00000001u, 0x807fffffu,
      0x33800000u, 0x33800001u, 0x3f800000u, 0xbf800000u,
      0x00000000u, 0x00000000u};
  std::memcpy(a.data(), copy_special, sizeof(copy_special));
  std::memcpy(b.data(), add_rhs, sizeof(add_rhs));

  cuda::DeviceBuffer<float> cuda_a(count), cuda_b(count), cuda_copy(count), cuda_sum(count);
  cuda_a.copy_from_host(a.data(), count);
  cuda_b.copy_from_host(b.data(), count);
  VIDFAB_CUDA_CHECK(cudaMemcpy(cuda_copy.get(), cuda_a.get(), count * sizeof(float),
                               cudaMemcpyDeviceToDevice));
  cuda::launch_add(cuda_a.get(), cuda_b.get(), cuda_sum.get(), count, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> cuda_copy_host(count), cuda_sum_host(count);
  cuda_copy.copy_to_host(cuda_copy_host.data(), count);
  cuda_sum.copy_to_host(cuda_sum_host.data(), count);

  DeviceTensor vk_a = vk.allocate(layout);
  DeviceTensor vk_b = vk.allocate(layout);
  DeviceTensor vk_copy = vk.allocate(layout);
  DeviceTensor vk_sum = vk.allocate(layout);
  vk.upload(vk_a, a.data(), count);
  vk.upload(vk_b, b.data(), count);
  TensorBatch batch = vk.begin_batch();
  batch.copy(vk_a, vk_copy);
  batch.add(vk_a, vk_b, vk_sum);
  batch.submit().wait();
  std::vector<float> vk_copy_host(count), vk_sum_host(count);
  vk.download(vk_copy, vk_copy_host.data(), count);
  vk.download(vk_sum, vk_sum_host.data(), count);

  CHECK(std::memcmp(cuda_copy_host.data(), vk_copy_host.data(),
                    count * sizeof(float)) == 0);
  // NaN results are excluded from the arithmetic exactness claim; every
  // finite/infinite result, including signed-zero/subnormal inputs, is exact.
  for (size_t i = 0; i < count; ++i) {
    uint32_t bits = 0;
    std::memcpy(&bits, &a[i], sizeof(bits));
    if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0) continue;
    // Indices 2/3 require denormal-preserving arithmetic. On devices without
    // that Vulkan mode, the explicit capability gate above fails rather than
    // claiming those results are CUDA-exact.
    if (!vk.full_fp32_arithmetic_exactness() && (i == 2 || i == 3)) continue;
    uint32_t cuda_bits = 0, vk_bits = 0;
    std::memcpy(&cuda_bits, &cuda_sum_host[i], sizeof(cuda_bits));
    std::memcpy(&vk_bits, &vk_sum_host[i], sizeof(vk_bits));
    CHECK_MSG(cuda_bits == vk_bits,
              "CUDA/Vulkan fp32 add mismatch at %zu: %08x != %08x", i,
              cuda_bits, vk_bits);
  }
}

VIDFAB_TEST(cuda_vulkan_exact_blocked_attention) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore ||
      !physical.front().info().shader_int64) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  if (!vk.exact_blocked_attention()) return;

  auto run = [&](uint32_t sequence, uint32_t heads, uint32_t dim,
                 const std::vector<uint16_t>* custom_q = nullptr,
                 const std::vector<uint16_t>* custom_k = nullptr,
                 const std::vector<uint16_t>* custom_v = nullptr) {
    const size_t count = static_cast<size_t>(sequence) * heads * dim;
    std::vector<uint16_t> q(count), k(count), v(count);
    for (size_t i = 0; i < count; ++i) {
      q[i] = f32_to_bf16(static_cast<float>(static_cast<int>(i * 17 % 41) - 20) / 32.0f);
      k[i] = f32_to_bf16(static_cast<float>(static_cast<int>(i * 13 % 37) - 18) / 32.0f);
      v[i] = f32_to_bf16(static_cast<float>(static_cast<int>(i * 19 % 43) - 21) / 16.0f);
    }
    if (count >= 8) {
      q[0] = 0x8000u; q[1] = f32_to_bf16(1.0f);
      k[0] = f32_to_bf16(1.0f); k[1] = f32_to_bf16(-1.0f);
      v[0] = 0x8000u; v[1] = f32_to_bf16(1.0f);
    }
    if (custom_q && custom_k && custom_v) {
      CHECK(custom_q->size() == count);
      CHECK(custom_k->size() == count);
      CHECK(custom_v->size() == count);
      q = *custom_q; k = *custom_k; v = *custom_v;
    }
    cuda::DeviceBuffer<uint16_t> cq(count), ck(count), cv(count), co(count),
        co_repeat(sequence == 129 ? count : 0);
    cuda::DeviceBuffer<uint16_t> cq16(count), ck16(count), cv16(count);
    cq.copy_from_host(q.data(), count); ck.copy_from_host(k.data(), count);
    cv.copy_from_host(v.data(), count);
    const float scale = dim == 64 ? 0.125f : dim == 128 ? 0.0883883461356163f
                                                         : 0.11785113019775793f;
    cuda::launch_prepare_deterministic_attention_inputs(
        nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
        reinterpret_cast<const __nv_bfloat16*>(ck.get()),
        reinterpret_cast<const __nv_bfloat16*>(cv.get()),
        reinterpret_cast<__half*>(cq16.get()), reinterpret_cast<__half*>(ck16.get()),
        reinterpret_cast<__half*>(cv16.get()), count);
    if (sequence == 129) {
      cuda::launch_deterministic_blocked_attention_f16(
          nullptr, reinterpret_cast<const __half*>(cq16.get()),
          reinterpret_cast<const __half*>(ck16.get()),
          reinterpret_cast<const __half*>(cv16.get()),
          reinterpret_cast<__nv_bfloat16*>(co.get()), sequence, heads, dim,
          scale, 0, 65, 0);
      cuda::launch_deterministic_blocked_attention_f16(
          nullptr, reinterpret_cast<const __half*>(cq16.get()),
          reinterpret_cast<const __half*>(ck16.get()),
          reinterpret_cast<const __half*>(cv16.get()),
          reinterpret_cast<__nv_bfloat16*>(co.get()), sequence, heads, dim,
          scale, 65, 64, 65);
    } else {
      cuda::launch_deterministic_blocked_attention_f16(
          nullptr, reinterpret_cast<const __half*>(cq16.get()),
          reinterpret_cast<const __half*>(ck16.get()),
          reinterpret_cast<const __half*>(cv16.get()),
          reinterpret_cast<__nv_bfloat16*>(co.get()), sequence, heads, dim, scale);
    }
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> cuda_output(count);
    co.copy_to_host(cuda_output.data(), count);
    if (sequence == 129) {
      cuda::launch_deterministic_blocked_attention_f16(
          nullptr, reinterpret_cast<const __half*>(cq16.get()),
          reinterpret_cast<const __half*>(ck16.get()),
          reinterpret_cast<const __half*>(cv16.get()),
          reinterpret_cast<__nv_bfloat16*>(co_repeat.get()), sequence, heads,
          dim, scale);
      VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<uint16_t> cuda_repeat(count);
      co_repeat.copy_to_host(cuda_repeat.data(), count);
      CHECK(cuda_repeat == cuda_output);
    }

    const uint64_t shape[] = {sequence, heads, dim};
    const TensorLayout layout = TensorLayout::contiguous(shape, 3);
    DeviceTensor vq = vk.allocate(layout, ScalarType::kBFloat16);
    DeviceTensor vk_tensor = vk.allocate(layout, ScalarType::kBFloat16);
    DeviceTensor vv = vk.allocate(layout, ScalarType::kBFloat16);
    DeviceTensor vo = vk.allocate(layout, ScalarType::kBFloat16);
    vk.upload_bytes(vq, q.data(), count * 2);
    vk.upload_bytes(vk_tensor, k.data(), count * 2);
    vk.upload_bytes(vv, v.data(), count * 2);
    BlockedAttentionPlanDesc desc{sequence, heads, dim, scale};
    BlockedAttentionPlan plan = BlockedAttentionPlan::create(vk, desc);
    PreparedAttentionInputs prepared = PreparedAttentionInputs::create(vk, desc);
    TensorBatch batch = vk.begin_batch();
    PreparedAttentionView inputs = prepared.prepare(batch, vq, vk_tensor, vv);
    if (sequence == 129) {
      plan.record(batch, inputs, vo, 0, 65, 0);
      plan.record(batch, inputs, vo, 65, 64, 65);
    } else {
      plan.record(batch, inputs, vo);
    }
    batch.submit().wait();
    std::vector<uint16_t> vulkan_output(count);
    vk.download_bytes(vo, vulkan_output.data(), count * 2);
    if (sequence == 129) {
      TensorBatch repeat = vk.begin_batch();
      PreparedAttentionView repeat_inputs =
          prepared.prepare(repeat, vq, vk_tensor, vv);
      plan.record(repeat, repeat_inputs, vo);
      repeat.submit().wait();
      std::vector<uint16_t> vulkan_repeat(count);
      vk.download_bytes(vo, vulkan_repeat.data(), count * 2);
      CHECK(vulkan_repeat == vulkan_output);
    }
    size_t mismatch = count;
    for (size_t i = 0; i < count; ++i) {
      if (cuda_output[i] != vulkan_output[i]) { mismatch = i; break; }
    }
    CHECK_MSG(mismatch == count,
              "CUDA/Vulkan exact attention S%u H%u D%u mismatch at %zu: %04x != %04x",
              sequence, heads, dim, mismatch,
              mismatch == count ? 0u : cuda_output[mismatch],
              mismatch == count ? 0u : vulkan_output[mismatch]);
    return vulkan_output;
  };
  CHECK(cuda::deterministic_attention_grid_fits(1, 1, 1, 1));
  CHECK(cuda::deterministic_attention_grid_fits(65535, 65535, 65535, 65535));
  CHECK(!cuda::deterministic_attention_grid_fits(0, 1, 65535, 65535));
  CHECK(!cuda::deterministic_attention_grid_fits(65536, 1, 65535, 65535));
  CHECK(!cuda::deterministic_attention_grid_fits(1, 65536, 65535, 65535));

  // CUDA hardware and the canonical host conversion agree for every BF16
  // bit pattern. Vulkan's word-owned conversion is checked separately by the
  // Vulkan shader test, so a duplicated conversion bug cannot hide in final
  // attention equality.
  {
    constexpr size_t patterns = 1u << 16;
    std::vector<uint16_t> bits(patterns), got(patterns);
    for (size_t i = 0; i < patterns; ++i) bits[i] = static_cast<uint16_t>(i);
    cuda::DeviceBuffer<uint16_t> source0(patterns), source1(patterns),
        source2(patterns), prepared0(patterns), prepared1(patterns),
        prepared2(patterns);
    source0.copy_from_host(bits.data(), patterns);
    source1.copy_from_host(bits.data(), patterns);
    source2.copy_from_host(bits.data(), patterns);
    bool prepare_alias_rejected = false;
    try {
      cuda::launch_prepare_deterministic_attention_inputs(
          nullptr, reinterpret_cast<const __nv_bfloat16*>(source0.get()),
          reinterpret_cast<const __nv_bfloat16*>(source0.get()),
          reinterpret_cast<const __nv_bfloat16*>(source2.get()),
          reinterpret_cast<__half*>(prepared0.get()),
          reinterpret_cast<__half*>(prepared1.get()),
          reinterpret_cast<__half*>(prepared2.get()), patterns);
    } catch (const std::invalid_argument&) {
      prepare_alias_rejected = true;
    }
    CHECK(prepare_alias_rejected);
    cuda::launch_prepare_deterministic_attention_inputs(
        nullptr, reinterpret_cast<const __nv_bfloat16*>(source0.get()),
        reinterpret_cast<const __nv_bfloat16*>(source1.get()),
        reinterpret_cast<const __nv_bfloat16*>(source2.get()),
        reinterpret_cast<__half*>(prepared0.get()),
        reinterpret_cast<__half*>(prepared1.get()),
        reinterpret_cast<__half*>(prepared2.get()), patterns);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    prepared0.copy_to_host(got.data(), patterns);
    for (size_t i = 0; i < patterns; ++i) {
      const uint16_t expected = (bits[i] & 0x7fffu) > 0x7f80u
          ? 0x7fffu : f32_to_f16(bf16_to_f32(bits[i]));
      CHECK(got[i] == expected);
    }
    bool output_alias_rejected = false;
    try {
      cuda::launch_deterministic_blocked_attention_f16(
          nullptr, reinterpret_cast<const __half*>(prepared0.get()),
          reinterpret_cast<const __half*>(prepared1.get()),
          reinterpret_cast<const __half*>(prepared2.get()),
          reinterpret_cast<__nv_bfloat16*>(prepared0.get()), 1, 1, 64,
          exact_attention_scale(64));
    } catch (const std::invalid_argument&) {
      output_alias_rejected = true;
    }
    CHECK(output_alias_rejected);
  }
  run(1, 2, 64);
  run(17, 3, 72);
  run(129, 2, 128);
  {
    constexpr uint32_t sequence = 129, heads = 1, dim = 64;
    constexpr size_t count = size_t(sequence) * heads * dim;
    std::vector<uint16_t> q(count, f32_to_bf16(1.0f));
    std::vector<uint16_t> k(count, f32_to_bf16(0.0f));
    std::vector<uint16_t> v(count, f32_to_bf16(0.0f));
    for (uint32_t d = 0; d < dim; ++d) {
      k[(sequence - 1) * dim + d] = f32_to_bf16(10.75f);
    }
    for (uint32_t row = 0; row + 1 < sequence; ++row) {
      v[size_t(row) * dim] = f32_to_bf16(0x1p-24f);
    }
    // The first tile accumulates 128*2^-24=2^-17. The next tile raises the
    // score maximum by exactly 86, so its correction makes that old value an
    // fp32 subnormal. Both backends deliberately flush it to signed zero.
    CHECK(std::ldexp(1.0, -17) * std::exp(-86.0) <
          std::numeric_limits<float>::min());
    const std::vector<uint16_t> flushed =
        run(sequence, heads, dim, &q, &k, &v);
    for (uint16_t bits : flushed) CHECK(bits == 0x0000u);
  }

  // A Qwen-vision-shaped D72 tail: preparation is once, then two query-row
  // consumers share it. Uploads/downloads are outside both timings.
  {
    const bool real_shape = std::getenv("VIDFAB_ATTENTION_REAL_BENCH") != nullptr;
    const uint32_t sequence = real_shape ? 16384u : 257u;
    constexpr uint32_t heads = 16, dim = 72;
    const uint32_t first_rows = (sequence + 1) / 2;
    const uint32_t second_rows = sequence - first_rows;
    const size_t count = size_t(sequence) * heads * dim;
    const int warmups = real_shape ? 0 : 2;
    const int samples = real_shape ? 2 : 5;
    std::vector<uint16_t> host(count, f32_to_bf16(0.03125f));
    cuda::DeviceBuffer<uint16_t> cq(count), ck(count), cv(count), co(count);
    cuda::DeviceBuffer<uint16_t> cq16(count), ck16(count), cv16(count);
    cq.copy_from_host(host.data(), count); ck.copy_from_host(host.data(), count);
    cv.copy_from_host(host.data(), count);
    cudaEvent_t begin{}, end{};
    VIDFAB_CUDA_CHECK(cudaEventCreate(&begin));
    VIDFAB_CUDA_CHECK(cudaEventCreate(&end));
    float cuda_prepare_ms = 0.0f, cuda_attention_ms = 0.0f;
    for (int iteration = -warmups; iteration < samples; ++iteration) {
      VIDFAB_CUDA_CHECK(cudaEventRecord(begin));
      cuda::launch_prepare_deterministic_attention_inputs(
          nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
          reinterpret_cast<const __nv_bfloat16*>(ck.get()),
          reinterpret_cast<const __nv_bfloat16*>(cv.get()),
          reinterpret_cast<__half*>(cq16.get()),
          reinterpret_cast<__half*>(ck16.get()),
          reinterpret_cast<__half*>(cv16.get()), count);
      VIDFAB_CUDA_CHECK(cudaEventRecord(end));
      VIDFAB_CUDA_CHECK(cudaEventSynchronize(end));
      float prepare_ms = 0.0f;
      VIDFAB_CUDA_CHECK(cudaEventElapsedTime(&prepare_ms, begin, end));
      VIDFAB_CUDA_CHECK(cudaEventRecord(begin));
      cuda::launch_deterministic_blocked_attention_f16(
          nullptr, reinterpret_cast<const __half*>(cq16.get()),
          reinterpret_cast<const __half*>(ck16.get()),
          reinterpret_cast<const __half*>(cv16.get()),
          reinterpret_cast<__nv_bfloat16*>(co.get()), sequence, heads, dim,
          exact_attention_scale(dim), 0, first_rows, 0);
      cuda::launch_deterministic_blocked_attention_f16(
          nullptr, reinterpret_cast<const __half*>(cq16.get()),
          reinterpret_cast<const __half*>(ck16.get()),
          reinterpret_cast<const __half*>(cv16.get()),
          reinterpret_cast<__nv_bfloat16*>(co.get()), sequence, heads, dim,
          exact_attention_scale(dim), first_rows, second_rows, first_rows);
      VIDFAB_CUDA_CHECK(cudaEventRecord(end));
      VIDFAB_CUDA_CHECK(cudaEventSynchronize(end));
      float attention_ms = 0.0f;
      VIDFAB_CUDA_CHECK(cudaEventElapsedTime(&attention_ms, begin, end));
      if (iteration >= 0) {
        cuda_prepare_ms += prepare_ms;
        cuda_attention_ms += attention_ms;
      }
    }
    VIDFAB_CUDA_CHECK(cudaEventDestroy(begin));
    VIDFAB_CUDA_CHECK(cudaEventDestroy(end));

    const uint64_t shape[] = {sequence, heads, dim};
    const TensorLayout layout = TensorLayout::contiguous(shape, 3);
    DeviceTensor q = vk.allocate(layout, ScalarType::kBFloat16);
    DeviceTensor k = vk.allocate(layout, ScalarType::kBFloat16);
    DeviceTensor v = vk.allocate(layout, ScalarType::kBFloat16);
    DeviceTensor out = vk.allocate(layout, ScalarType::kBFloat16);
    vk.upload_bytes(q, host.data(), count * 2);
    vk.upload_bytes(k, host.data(), count * 2);
    vk.upload_bytes(v, host.data(), count * 2);
    BlockedAttentionPlanDesc desc{sequence, heads, dim,
                                  exact_attention_scale(dim)};
    BlockedAttentionPlan plan = BlockedAttentionPlan::create(vk, desc);
    PreparedAttentionInputs prepared = PreparedAttentionInputs::create(vk, desc);
    double vulkan_total_ms = 0.0;
    for (int iteration = -warmups; iteration < samples; ++iteration) {
      const auto start = std::chrono::steady_clock::now();
      TensorBatch batch = vk.begin_batch();
      PreparedAttentionView inputs = prepared.prepare(batch, q, k, v);
      plan.record(batch, inputs, out, 0, first_rows, 0);
      plan.record(batch, inputs, out, first_rows, second_rows, first_rows);
      batch.submit().wait();
      if (iteration >= 0) {
        vulkan_total_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
      }
    }
    std::printf("  exact D72 attention S%u H16, 2 query chunks: CUDA prepare %.3f ms + attention %.3f ms, Vulkan total %.3f ms, FP16 slot %.2f MiB\n",
                sequence, cuda_prepare_ms / samples,
                cuda_attention_ms / samples, vulkan_total_ms / samples,
        prepared.reserved_bytes() / (1024.0 * 1024.0));
    if (real_shape) {
      std::vector<uint16_t> cuda_real(count), vulkan_real(count);
      co.copy_to_host(cuda_real.data(), count);
      vk.download_bytes(out, vulkan_real.data(), count * sizeof(uint16_t));
      CHECK(cuda_real == vulkan_real);
    }
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

VIDFAB_TEST(cuda_vulkan_exact_h3_attention) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  const unsigned char board_a[16] = {};
  const unsigned char board_b[16] = {1, 2, 3, 4};
  CHECK(cuda::deterministic_h3_cuda_tuple_fits(
      12, 0, "NVIDIA GeForce RTX 5090", 13010, 13000, 1024, 99328,
      board_a));
  CHECK(cuda::deterministic_h3_cuda_tuple_fits(
      12, 0, "NVIDIA GeForce RTX 5090", 13010, 13000, 1024, 99328,
      board_b));
  CHECK(!cuda::deterministic_h3_cuda_tuple_fits(
      12, 0, "NVIDIA GeForce RTX 5080", 13010, 13000, 1024, 99328,
      board_a));
  CHECK(!cuda::deterministic_h3_cuda_tuple_fits(
      10, 0, "NVIDIA GeForce RTX 5090", 13010, 13000, 1024, 99328,
      board_a));
  CHECK(!cuda::deterministic_h3_cuda_tuple_fits(
      12, 0, "NVIDIA GeForce RTX 5090", 13010, 12090, 1024, 99328,
      board_a));
  CHECK(!cuda::deterministic_h3_cuda_tuple_fits(
      12, 0, "NVIDIA GeForce RTX 5090", 13010, 13000, 512, 99328,
      board_a));
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  options.enable_shader_float16 = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  if (!vk.exact_h3_attention()) return;

  // The production 124-frame table is built by the same backend-neutral host
  // path used by the transformer. Boundary probes independently derive each
  // tile's selected prefix/frame band rather than trusting the flattened table.
  {
    dit::SequenceLayout real;
    real.num_text = 17;
    real.num_audio_rows = 414;
    real.num_latent_frames = 37;
    real.latent_height = 48;
    real.latent_width = 84;
    real.num_video_rows = real.num_latent_frames * real.rows_per_frame();
    CHECK(real.total_rows() == 37727);
    CHECK(real.video_start() == 431);
    CHECK(real.rows_per_frame() == 1008);
    const dit::BandedKeyRanges table =
        dit::build_banded_key_ranges(real, 9, 128, 64);
    CHECK(table.query_tile == 128 && table.num_query_tiles == 295);
    auto round_down = [](int value) { return value / 64 * 64; };
    auto round_up = [](int value) { return (value + 63) / 64 * 64; };
    const int seq_end = round_up(real.total_rows());
    for (int tile = 0; tile < table.num_query_tiles; ++tile) {
      const int q0 = tile * 128;
      const int qlast = std::min(q0 + 128, real.total_rows()) - 1;
      int expected_lo0 = 0, expected_hi0 = seq_end;
      int expected_lo1 = 0, expected_hi1 = 0;
      if (q0 >= real.video_start()) {
        const int first = (q0 - real.video_start()) / real.rows_per_frame();
        const int last = (qlast - real.video_start()) / real.rows_per_frame();
        expected_hi0 = round_up(real.video_start());
        expected_lo1 = round_down(real.video_start() +
                                  std::max(0, first - 9) * real.rows_per_frame());
        expected_hi1 = std::min(seq_end, round_up(
            real.video_start() + std::min(real.num_latent_frames, last + 10) *
                                     real.rows_per_frame()));
        if (expected_lo1 <= expected_hi0) {
          expected_hi0 = std::max(expected_hi0, expected_hi1);
          expected_lo1 = expected_hi1 = 0;
        }
      }
      const int32_t* got = table.ranges.data() + size_t(tile) * 4;
      CHECK(got[0] == expected_lo0 && got[1] == expected_hi0 &&
            got[2] == expected_lo1 && got[3] == expected_hi1);
      const int probes[] = {0, real.video_start() - 1, real.video_start(),
                            expected_lo1 - 1, expected_lo1,
                            expected_hi1 - 1, expected_hi1,
                            real.total_rows() - 1, real.total_rows()};
      for (int row : probes) {
        if (row < 0 || row >= real.total_rows()) continue;
        const bool selected =
            (row >= got[0] && row < got[1]) ||
            (row >= got[2] && row < got[3]);
        const bool expected =
            (row >= expected_lo0 && row < expected_hi0) ||
            (row >= expected_lo1 && row < expected_hi1);
        CHECK(selected == expected);
      }
    }
    H3AttentionRanges uploaded = H3AttentionRanges::create(
        vk, real.total_rows(), table.ranges.data(),
        static_cast<uint32_t>(table.ranges.size()));
    std::printf("  H3 production +/-9 range table: %u tiles, %zu bytes, FNV64 %016llx\n",
                uploaded.query_tiles(), table.ranges.size() * sizeof(int32_t),
                static_cast<unsigned long long>(uploaded.content_hash()));
    CHECK(uploaded.content_hash() == 0x32b19bc0895faa6aull);
  }

  constexpr uint32_t sequence = 129, heads = 2, dim = 64;
  const size_t count = size_t(sequence) * heads * dim;
  std::vector<uint16_t> hq(count), hk(count), hv(count);
  for (size_t i = 0; i < count; ++i) {
    hq[i] = f32_to_bf16(float(int(i % 29) - 14) / 32.0f);
    hk[i] = f32_to_bf16(float(int(i % 31) - 15) / 32.0f);
    hv[i] = f32_to_bf16(float(int(i % 37) - 18) / 16.0f);
  }
  hq[0] = 0x0001u; hq[1] = 0x807fu;
  hk[0] = 0x007fu; hk[1] = 0x8001u;
  hv[0] = 0x0001u; hv[1] = 0x807fu;
  const std::vector<int32_t> band{
      0, 64, 128, 192,
      0, 128, 0, 0};
  const std::vector<int32_t> wide{
      0, 192, 0, 0,
      0, 192, 0, 0};
  cuda::DeviceBuffer<uint16_t> cq(count), ck(count), cv(count),
      co_full(count), co_band(count), co_wide(count);
  cuda::DeviceBuffer<int32_t> cband(band.size()), cwide(wide.size());
  cq.copy_from_host(hq.data(), hq.size());
  ck.copy_from_host(hk.data(), hk.size());
  cv.copy_from_host(hv.data(), hv.size());
  cband.copy_from_host(band.data(), band.size());
  cwide.copy_from_host(wide.data(), wide.size());
  const float scale = exact_attention_scale(dim);
  cuda::launch_deterministic_h3_attention(
      nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
      reinterpret_cast<const __nv_bfloat16*>(ck.get()),
      reinterpret_cast<const __nv_bfloat16*>(cv.get()),
      reinterpret_cast<__nv_bfloat16*>(co_full.get()), nullptr,
      sequence, heads, dim, scale);
  cuda::launch_deterministic_h3_attention(
      nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
      reinterpret_cast<const __nv_bfloat16*>(ck.get()),
      reinterpret_cast<const __nv_bfloat16*>(cv.get()),
      reinterpret_cast<__nv_bfloat16*>(co_band.get()), cband.get(),
      sequence, heads, dim, scale, 0, 65, 0);
  cuda::launch_deterministic_h3_attention(
      nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
      reinterpret_cast<const __nv_bfloat16*>(ck.get()),
      reinterpret_cast<const __nv_bfloat16*>(cv.get()),
      reinterpret_cast<__nv_bfloat16*>(co_band.get()), cband.get(),
      sequence, heads, dim, scale, 65, 64, 65);
  cuda::launch_deterministic_h3_attention(
      nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
      reinterpret_cast<const __nv_bfloat16*>(ck.get()),
      reinterpret_cast<const __nv_bfloat16*>(cv.get()),
      reinterpret_cast<__nv_bfloat16*>(co_wide.get()), cwide.get(),
      sequence, heads, dim, scale);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> expected_full(count), expected_band(count), expected_wide(count);
  co_full.copy_to_host(expected_full.data(), count);
  co_band.copy_to_host(expected_band.data(), count);
  co_wide.copy_to_host(expected_wide.data(), count);
  CHECK(expected_wide == expected_full);

  const uint64_t shape[] = {sequence, heads, dim};
  const TensorLayout layout = TensorLayout::contiguous(shape, 3);
  DeviceTensor q = vk.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor k = vk.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor v = vk.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor out_full = vk.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor out_band = vk.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor out_wide = vk.allocate(layout, ScalarType::kBFloat16);
  vk.upload_bytes(q, hq.data(), count * 2);
  vk.upload_bytes(k, hk.data(), count * 2);
  vk.upload_bytes(v, hv.data(), count * 2);
  H3AttentionPlan plan = H3AttentionPlan::create(
      vk, {sequence, heads, dim, scale});
  H3AttentionRanges bands = H3AttentionRanges::create(
      vk, sequence, band.data(), static_cast<uint32_t>(band.size()));
  H3AttentionRanges wide_ranges = H3AttentionRanges::create(
      vk, sequence, wide.data(), static_cast<uint32_t>(wide.size()));
  // Touching ranges canonicalize to the same immutable table as full coverage.
  const std::vector<int32_t> touching{
      0, 64, 64, 192,
      0, 128, 128, 192};
  H3AttentionRanges touching_ranges = H3AttentionRanges::create(
      vk, sequence, touching.data(), static_cast<uint32_t>(touching.size()));
  CHECK(touching_ranges.content_hash() == wide_ranges.content_hash());
  TensorBatch batch = vk.begin_batch();
  plan.record(batch, q, k, v, out_full);
  plan.record(batch, q, k, v, out_band, &bands, 0, 65, 0);
  plan.record(batch, q, k, v, out_band, &bands, 65, 64, 65);
  plan.record(batch, q, k, v, out_wide, &wide_ranges);
  batch.submit().wait();
  std::vector<uint16_t> got_full(count), got_band(count), got_wide(count);
  vk.download_bytes(out_full, got_full.data(), count * 2);
  vk.download_bytes(out_band, got_band.data(), count * 2);
  vk.download_bytes(out_wide, got_wide.data(), count * 2);
  CHECK(got_full == expected_full);
  CHECK(got_band == expected_band);
  CHECK(got_wide == expected_wide);
  CHECK(got_wide == got_full);

  // Q/K are zero, so selected values average exactly. Excluded sentinels must
  // not affect the banded result, including across the padded final block.
  std::fill(hq.begin(), hq.end(), f32_to_bf16(0.0f));
  std::fill(hk.begin(), hk.end(), f32_to_bf16(0.0f));
  std::fill(hv.begin(), hv.end(), f32_to_bf16(16.0f));
  for (uint32_t row = 0; row < 64; ++row)
    for (uint32_t h = 0; h < heads; ++h)
      for (uint32_t d = 0; d < dim; ++d)
        hv[(size_t(row) * heads + h) * dim + d] = f32_to_bf16(1.0f);
  for (uint32_t h = 0; h < heads; ++h)
    for (uint32_t d = 0; d < dim; ++d)
      hv[(size_t(128) * heads + h) * dim + d] = f32_to_bf16(1.0f);
  vk.upload_bytes(q, hq.data(), count * 2);
  vk.upload_bytes(k, hk.data(), count * 2);
  vk.upload_bytes(v, hv.data(), count * 2);
  TensorBatch sentinel = vk.begin_batch();
  plan.record(sentinel, q, k, v, out_band, &bands, 0, 1, 0);
  sentinel.submit().wait();
  vk.download_bytes(out_band, got_band.data(), count * 2);
  for (uint32_t i = 0; i < heads * dim; ++i)
    CHECK(got_band[i] == f32_to_bf16(1.0f));
  const std::vector<int32_t> single_values{
      128, 192, 0, 0, 128, 192, 0, 0};
  H3AttentionRanges single = H3AttentionRanges::create(
      vk, sequence, single_values.data(),
      static_cast<uint32_t>(single_values.size()));
  std::fill(hv.begin(), hv.end(), f32_to_bf16(-7.0f));
  for (uint32_t h = 0; h < heads; ++h)
    for (uint32_t d = 0; d < dim; ++d)
      hv[(size_t(128) * heads + h) * dim + d] = f32_to_bf16(1.5f);
  vk.upload_bytes(v, hv.data(), count * 2);
  TensorBatch one_key = vk.begin_batch();
  plan.record(one_key, q, k, v, out_band, &single, 0, 1, 0);
  one_key.submit().wait();
  vk.download_bytes(out_band, got_band.data(), count * 2);
  for (uint32_t i = 0; i < heads * dim; ++i)
    CHECK(got_band[i] == f32_to_bf16(1.5f));

  const std::vector<int32_t> uniform_values{
      0, 64, 0, 0, 0, 64, 0, 0};
  H3AttentionRanges uniform = H3AttentionRanges::create(
      vk, sequence, uniform_values.data(),
      static_cast<uint32_t>(uniform_values.size()));
  for (uint32_t row = 0; row < 64; ++row)
    for (uint32_t h = 0; h < heads; ++h)
      for (uint32_t d = 0; d < dim; ++d)
        hv[(size_t(row) * heads + h) * dim + d] =
            f32_to_bf16(row < 32 ? 1.0f : 3.0f);
  vk.upload_bytes(v, hv.data(), count * 2);
  TensorBatch average = vk.begin_batch();
  plan.record(average, q, k, v, out_band, &uniform, 0, 1, 0);
  average.submit().wait();
  vk.download_bytes(out_band, got_band.data(), count * 2);
  for (uint32_t i = 0; i < heads * dim; ++i)
    CHECK(got_band[i] == f32_to_bf16(2.0f));

  // D128, three global query tiles, every tile/64-key boundary, row chunks,
  // and a nonzero output offset. The CUDA full launch is the exact oracle;
  // Vulkan consumes the same immutable table across all chunk records.
  {
    constexpr uint32_t s = 257, h = 1, d = 128;
    const size_t n = size_t(s) * h * d;
    std::vector<uint16_t> qh(n), kh(n), vh(n);
    for (size_t i = 0; i < n; ++i) {
      qh[i] = f32_to_bf16(float(int(i % 23) - 11) / 32.0f);
      kh[i] = f32_to_bf16(float(int(i % 27) - 13) / 32.0f);
      vh[i] = f32_to_bf16(float(int(i % 33) - 16) / 16.0f);
    }
    const std::vector<int32_t> r{
        0, 64, 128, 192,
        0, 128, 192, 256,
        0, 320, 0, 0};
    cuda::DeviceBuffer<uint16_t> dq(n), dk(n), dv(n), dout(n);
    cuda::DeviceBuffer<int32_t> dr(r.size());
    dq.copy_from_host(qh.data(), n); dk.copy_from_host(kh.data(), n);
    dv.copy_from_host(vh.data(), n); dr.copy_from_host(r.data(), r.size());
    cuda::launch_deterministic_h3_attention(
        nullptr, reinterpret_cast<const __nv_bfloat16*>(dq.get()),
        reinterpret_cast<const __nv_bfloat16*>(dk.get()),
        reinterpret_cast<const __nv_bfloat16*>(dv.get()),
        reinterpret_cast<__nv_bfloat16*>(dout.get()), dr.get(), s, h, d,
        exact_attention_scale(d));
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> expected(n);
    dout.copy_to_host(expected.data(), n);
    const uint64_t sshape[] = {s, h, d};
    TensorLayout slayout = TensorLayout::contiguous(sshape, 3);
    DeviceTensor vq = vk.allocate(slayout, ScalarType::kBFloat16);
    DeviceTensor vkey = vk.allocate(slayout, ScalarType::kBFloat16);
    DeviceTensor vv = vk.allocate(slayout, ScalarType::kBFloat16);
    DeviceTensor vo = vk.allocate(slayout, ScalarType::kBFloat16);
    vk.upload_bytes(vq, qh.data(), n * 2);
    vk.upload_bytes(vkey, kh.data(), n * 2);
    vk.upload_bytes(vv, vh.data(), n * 2);
    H3AttentionPlan p = H3AttentionPlan::create(
        vk, {s, h, d, exact_attention_scale(d)});
    H3AttentionRanges vr = H3AttentionRanges::create(
        vk, s, r.data(), static_cast<uint32_t>(r.size()));
    TensorBatch chunks = vk.begin_batch();
    const uint32_t starts[] = {0, 1, 15, 16, 63, 64, 65, 127, 128, 129};
    const uint32_t lengths[] = {1, 14, 1, 47, 1, 1, 62, 1, 1, 128};
    for (size_t i = 0; i < std::size(starts); ++i)
      p.record(chunks, vq, vkey, vv, vo, &vr, starts[i], lengths[i], starts[i]);
    chunks.submit().wait();
    std::vector<uint16_t> got(n);
    vk.download_bytes(vo, got.data(), n * 2);
    CHECK(got == expected);
    const uint16_t sentinel_bits = f32_to_bf16(-123.0f);
    std::fill(got.begin(), got.end(), sentinel_bits);
    vk.upload_bytes(vo, got.data(), n * 2);
    TensorBatch offset = vk.begin_batch();
    p.record(offset, vq, vkey, vv, vo, &vr, 128, 1, 0);
    offset.submit().wait();
    vk.download_bytes(vo, got.data(), n * 2);
    CHECK(std::memcmp(got.data(), expected.data() + size_t(128) * d,
                      d * sizeof(uint16_t)) == 0);
    for (size_t i = d; i < n; ++i) CHECK(got[i] == sentinel_bits);

    // Independent two-range/two-block recurrence anchor. The second range's
    // score is >87 above the first, so the specified exp cutoff makes the old
    // block correction exactly zero and every output is exactly V=3.
    std::fill(qh.begin(), qh.end(), f32_to_bf16(1.0f));
    std::fill(kh.begin(), kh.end(), f32_to_bf16(0.0f));
    std::fill(vh.begin(), vh.end(), f32_to_bf16(1.0f));
    for (uint32_t column = 0; column < d; ++column) {
      kh[size_t(128) * d + column] = f32_to_bf16(8.0f);
      vh[size_t(128) * d + column] = f32_to_bf16(3.0f);
    }
    vk.upload_bytes(vq, qh.data(), n * 2);
    vk.upload_bytes(vkey, kh.data(), n * 2);
    vk.upload_bytes(vv, vh.data(), n * 2);
    TensorBatch seam = vk.begin_batch();
    p.record(seam, vq, vkey, vv, vo, &vr, 0, 1, 0);
    seam.submit().wait();
    vk.download_bytes(vo, got.data(), n * 2);
    for (uint32_t column = 0; column < d; ++column)
      CHECK(got[column] == f32_to_bf16(3.0f));

    // Three ordered blocks with successively larger finite maxima force two
    // strictly-between-zero-and-one online corrections. This stresses a
    // nonzero cooperative C tile rather than only the first-block/zero-cutoff
    // cases; CUDA and Vulkan must repeat exactly and the independent bound
    // excludes either dropped-old-state or reset-at-range-seam outcomes.
    const std::vector<int32_t> correction_values{
        0, 128, 192, 256, 0, 128, 192, 256, 0, 256, 0, 0};
    cuda::DeviceBuffer<int32_t> correction_device(correction_values.size());
    correction_device.copy_from_host(correction_values.data(),
                                     correction_values.size());
    H3AttentionRanges correction_ranges = H3AttentionRanges::create(
        vk, s, correction_values.data(),
        static_cast<uint32_t>(correction_values.size()));
    std::fill(qh.begin(), qh.end(), f32_to_bf16(1.0f));
    std::fill(kh.begin(), kh.end(), f32_to_bf16(0.0f));
    std::fill(vh.begin(), vh.end(), f32_to_bf16(1.0f));
    for (uint32_t row = 64; row < 128; ++row) {
      for (uint32_t column = 0; column < d; ++column) {
        kh[size_t(row) * d + column] = f32_to_bf16(1.0f / 64.0f);
        vh[size_t(row) * d + column] = f32_to_bf16(2.0f);
      }
    }
    for (uint32_t row = 192; row < 256; ++row) {
      for (uint32_t column = 0; column < d; ++column) {
        kh[size_t(row) * d + column] = f32_to_bf16(1.0f / 32.0f);
        vh[size_t(row) * d + column] = f32_to_bf16(3.0f);
      }
    }
    dq.copy_from_host(qh.data(), n); dk.copy_from_host(kh.data(), n);
    dv.copy_from_host(vh.data(), n);
    cuda::launch_deterministic_h3_attention(
        nullptr, reinterpret_cast<const __nv_bfloat16*>(dq.get()),
        reinterpret_cast<const __nv_bfloat16*>(dk.get()),
        reinterpret_cast<const __nv_bfloat16*>(dv.get()),
        reinterpret_cast<__nv_bfloat16*>(dout.get()), correction_device.get(),
        s, h, d, exact_attention_scale(d), 0, 1, 0);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    dout.copy_to_host(expected.data(), n);
    vk.upload_bytes(vq, qh.data(), n * 2);
    vk.upload_bytes(vkey, kh.data(), n * 2);
    vk.upload_bytes(vv, vh.data(), n * 2);
    TensorBatch correction_batch = vk.begin_batch();
    p.record(correction_batch, vq, vkey, vv, vo, &correction_ranges, 0, 1, 0);
    correction_batch.submit().wait();
    vk.download_bytes(vo, got.data(), n * 2);
    for (uint32_t column = 0; column < d; ++column) {
      CHECK(got[column] == expected[column]);
      CHECK(bf16_to_f32(got[column]) > 1.0f &&
            bf16_to_f32(got[column]) < 3.0f);
    }
  }

  // Cooperative-MMA tuple corpus: signed zeros, the minimum normal boundary,
  // mixed exponents, cancellation and one-ULP-neighbor operands exercise QK
  // and PV rounding for both supported head widths. Repeated independent
  // submissions pin the empirically qualified WMMA/KHR internal semantics.
  for (uint32_t adversarial_dim : {64u, 128u}) {
    constexpr uint32_t adversarial_sequence = 65;
    const size_t adversarial_count =
        size_t(adversarial_sequence) * adversarial_dim;
    const uint16_t patterns[] = {
        0x0000u, 0x8000u, 0x0080u, 0x8080u, 0x3f80u, 0xbf80u,
        0x3f81u, 0xbf81u, 0x3f00u, 0xbf00u, 0x4000u, 0xc000u,
        0x3c00u, 0xbc00u, 0x3eabu, 0xbeabu};
    std::vector<uint16_t> aq(adversarial_count), ak(adversarial_count),
        av(adversarial_count);
    for (size_t i = 0; i < adversarial_count; ++i) {
      aq[i] = patterns[i % std::size(patterns)];
      ak[i] = patterns[(i * 5 + (i / adversarial_dim)) % std::size(patterns)];
      av[i] = patterns[(i * 7 + 3) % std::size(patterns)];
    }
    cuda::DeviceBuffer<uint16_t> daq(adversarial_count), dak(adversarial_count),
        dav(adversarial_count), dao(adversarial_count), dar(adversarial_count);
    daq.copy_from_host(aq.data(), adversarial_count);
    dak.copy_from_host(ak.data(), adversarial_count);
    dav.copy_from_host(av.data(), adversarial_count);
    auto launch_adversarial = [&](cuda::DeviceBuffer<uint16_t>& selected) {
      cuda::launch_deterministic_h3_attention(
          nullptr, reinterpret_cast<const __nv_bfloat16*>(daq.get()),
          reinterpret_cast<const __nv_bfloat16*>(dak.get()),
          reinterpret_cast<const __nv_bfloat16*>(dav.get()),
          reinterpret_cast<__nv_bfloat16*>(selected.get()), nullptr,
          adversarial_sequence, 1, adversarial_dim,
          exact_attention_scale(adversarial_dim));
    };
    launch_adversarial(dao); launch_adversarial(dar);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> adversarial_expected(adversarial_count),
        adversarial_repeat(adversarial_count);
    dao.copy_to_host(adversarial_expected.data(), adversarial_count);
    dar.copy_to_host(adversarial_repeat.data(), adversarial_count);
    CHECK(adversarial_repeat == adversarial_expected);
    const uint64_t ashape[] = {adversarial_sequence, 1, adversarial_dim};
    const TensorLayout alayout = TensorLayout::contiguous(ashape, 3);
    DeviceTensor vaq = vk.allocate(alayout, ScalarType::kBFloat16);
    DeviceTensor vak = vk.allocate(alayout, ScalarType::kBFloat16);
    DeviceTensor vav = vk.allocate(alayout, ScalarType::kBFloat16);
    DeviceTensor vao = vk.allocate(alayout, ScalarType::kBFloat16);
    DeviceTensor var = vk.allocate(alayout, ScalarType::kBFloat16);
    vk.upload_bytes(vaq, aq.data(), adversarial_count * 2);
    vk.upload_bytes(vak, ak.data(), adversarial_count * 2);
    vk.upload_bytes(vav, av.data(), adversarial_count * 2);
    H3AttentionPlan adversarial_plan = H3AttentionPlan::create(
        vk, {adversarial_sequence, 1, adversarial_dim,
             exact_attention_scale(adversarial_dim)});
    TensorBatch adversarial_batch = vk.begin_batch();
    adversarial_plan.record(adversarial_batch, vaq, vak, vav, vao);
    adversarial_plan.record(adversarial_batch, vaq, vak, vav, var);
    adversarial_batch.submit().wait();
    std::vector<uint16_t> adversarial_got(adversarial_count);
    vk.download_bytes(vao, adversarial_got.data(), adversarial_count * 2);
    CHECK(adversarial_got == adversarial_expected);
    vk.download_bytes(var, adversarial_got.data(), adversarial_count * 2);
    CHECK(adversarial_got == adversarial_expected);
  }

  // Range validation is a setup boundary and cannot mutate a recorder. Empty,
  // misaligned, reversed, padding-only and wrong-count tables fail closed.
  auto range_rejected = [&](const std::vector<int32_t>& bad) {
    bool rejected = false;
    try {
      (void)H3AttentionRanges::create(
          vk, sequence, bad.data(), static_cast<uint32_t>(bad.size()));
    } catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
  };
  range_rejected({0, 0, 0, 0, 0, 128, 0, 0});
  range_rejected({1, 64, 0, 0, 0, 128, 0, 0});
  range_rejected({64, 0, 0, 0, 0, 128, 0, 0});
  range_rejected({192, 192, 0, 0, 0, 128, 0, 0});
  range_rejected({0, 64, 0, 0});

  // Rejected record calls leave the batch usable; operator overflow poisons
  // the recording. Repeated two-flight/oldest-slot reuse stays bounded.
  DeviceTensor wrong = vk.allocate(layout, ScalarType::kFloat32);
  {
    TensorBatch recover = vk.begin_batch();
    bool rejected = false;
    try { plan.record(recover, q, k, v, wrong, &bands); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    rejected = false;
    try { plan.record(recover, q, k, v, q, &bands); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    rejected = false;
    try { plan.record(recover, q, k, v, out_full, &bands, sequence, 1, 0); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    plan.record(recover, q, k, v, out_full, &bands);
    TensorBatch moved = std::move(recover);
    moved.submit().wait();
  }
  DeviceTensor second = vk.allocate(layout, ScalarType::kBFloat16);
  auto submit = [&](DeviceTensor& selected,
                    const H3AttentionRanges* selected_ranges) {
    TensorBatch selected_batch = vk.begin_batch();
    plan.record(selected_batch, q, k, v, selected, selected_ranges);
    return selected_batch.submit();
  };
  Submission first = submit(out_full, &bands),
             second_job = submit(second, &bands),
             third = submit(out_full, &bands);
  CHECK(second_job.value() > first.value() && third.value() > second_job.value());
  first.wait(); second_job.wait(); third.wait();
  std::vector<uint16_t> first_output(count), second_output(count);
  vk.download_bytes(out_full, first_output.data(), count * 2);
  vk.download_bytes(second, second_output.data(), count * 2);
  CHECK(first_output == second_output);
  Submission full_a = submit(out_full, nullptr),
             full_b = submit(second, nullptr),
             full_c = submit(out_full, nullptr);
  full_a.wait(); full_b.wait(); full_c.wait();
  const uint64_t stable_reserved = vk.reserved_bytes();
  const uint64_t stable_descriptors = vk.descriptor_set_allocations();
  for (int repeat = 0; repeat < 50; ++repeat) {
    const H3AttentionRanges* selected = (repeat & 1) ? &bands : nullptr;
    Submission a = submit(out_full, selected), b = submit(second, selected),
               c = submit(out_full, selected);
    a.wait(); b.wait(); c.wait();
    CHECK(vk.reserved_bytes() == stable_reserved);
    CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  }
  {
    TensorBatch full = vk.begin_batch();
    for (int op = 0; op < 32; ++op)
      plan.record(full, q, k, v, out_full, &bands, 0, 1, 0);
    full.submit().wait();
  }
  {
    TensorBatch overflow = vk.begin_batch();
    for (int op = 0; op < 32; ++op)
      plan.record(overflow, q, k, v, out_full, &bands, 0, 1, 0);
    bool rejected = false;
    try { plan.record(overflow, q, k, v, out_full, &bands, 0, 1, 0); }
    catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected);
    bool submit_rejected = false;
    try { (void)overflow.submit(); }
    catch (const std::logic_error&) { submit_rejected = true; }
    CHECK(submit_rejected);
  }

  // A discarded recorder releases speculative resources. A submitted job
  // retains the immutable table, plan and all four tensors after every public
  // wrapper drops, then releases them after its exact token is collected.
  const uint64_t used_before_drop = vk.pooled_used_bytes();
  {
    DeviceTensor temporary = vk.allocate(layout, ScalarType::kBFloat16);
    TensorBatch discarded = vk.begin_batch();
    plan.record(discarded, q, k, v, temporary, &bands, 0, 1, 0);
  }
  CHECK(vk.pooled_used_bytes() == used_before_drop);
  Submission retained;
  {
    DeviceTensor tq = vk.allocate(layout, ScalarType::kBFloat16);
    DeviceTensor tk = vk.allocate(layout, ScalarType::kBFloat16);
    DeviceTensor tv = vk.allocate(layout, ScalarType::kBFloat16);
    DeviceTensor tout = vk.allocate(layout, ScalarType::kBFloat16);
    vk.upload_bytes(tq, hq.data(), count * 2);
    vk.upload_bytes(tk, hk.data(), count * 2);
    vk.upload_bytes(tv, hv.data(), count * 2);
    H3AttentionPlan temporary_plan = H3AttentionPlan::create(
        vk, {sequence, heads, dim, scale});
    H3AttentionRanges temporary_ranges = H3AttentionRanges::create(
        vk, sequence, band.data(), static_cast<uint32_t>(band.size()));
    TensorBatch keep = vk.begin_batch();
    temporary_plan.record(keep, tq, tk, tv, tout, &temporary_ranges, 0, 1, 0);
    retained = keep.submit();
  }
  CHECK(vk.pooled_used_bytes() > used_before_drop);
  retained.wait();
  retained = Submission{};
  { TensorBatch collect = vk.begin_batch(); }
  CHECK(vk.pooled_used_bytes() == used_before_drop);
}

VIDFAB_TEST(cuda_vulkan_h3_real_timing) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!std::getenv("VIDFAB_H3_ATTENTION_REAL_BENCH")) return;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  constexpr uint32_t sequence = 37727, heads = 56, dim = 128;
  const size_t count = size_t(sequence) * heads * dim;
  dit::SequenceLayout layout;
  layout.num_text = 17;
  layout.num_audio_rows = 414;
  layout.num_latent_frames = 37;
  layout.latent_height = 48;
  layout.latent_width = 84;
  layout.num_video_rows = layout.num_latent_frames * layout.rows_per_frame();
  const dit::BandedKeyRanges band =
      dit::build_banded_key_ranges(layout, 9, 128, 64);
  const dit::BandedKeyRanges wide =
      dit::build_banded_key_ranges(layout, 64, 128, 64);
  std::vector<uint16_t> host(count);
  for (size_t i = 0; i < count; ++i)
    host[i] = f32_to_bf16(float(int(i % 31) - 15) / 64.0f);
  std::vector<uint16_t> expected_full(count), expected_band(count);
  float cuda_full_ms = 0.0f, cuda_band_ms = 0.0f;
  float shipped_full_ms = 0.0f, shipped_band_ms = 0.0f;
  size_t shipped_differences = 0;
  float shipped_max_abs = 0.0f;
  {
    cuda::DeviceBuffer<uint16_t> q(count), k(count), v(count), out(count);
    cuda::DeviceBuffer<int32_t> ranges(band.ranges.size());
    q.copy_from_host(host.data(), count);
    k.copy_from_host(host.data(), count);
    v.copy_from_host(host.data(), count);
    ranges.copy_from_host(band.ranges.data(), band.ranges.size());
    auto timed = [&](auto&& launch) {
      cudaEvent_t begin{}, end{};
      VIDFAB_CUDA_CHECK(cudaEventCreate(&begin));
      VIDFAB_CUDA_CHECK(cudaEventCreate(&end));
      VIDFAB_CUDA_CHECK(cudaEventRecord(begin));
      launch();
      VIDFAB_CUDA_CHECK(cudaEventRecord(end));
      VIDFAB_CUDA_CHECK(cudaEventSynchronize(end));
      float ms = 0.0f;
      VIDFAB_CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
      cudaEventDestroy(begin); cudaEventDestroy(end);
      return ms;
    };
    cuda_full_ms = timed([&] {
      cuda::launch_deterministic_h3_attention(
          nullptr, reinterpret_cast<const __nv_bfloat16*>(q.get()),
          reinterpret_cast<const __nv_bfloat16*>(k.get()),
          reinterpret_cast<const __nv_bfloat16*>(v.get()),
          reinterpret_cast<__nv_bfloat16*>(out.get()), nullptr,
          sequence, heads, dim, exact_attention_scale(dim));
    });
    out.copy_to_host(expected_full.data(), count);
    cuda_band_ms = timed([&] {
      cuda::launch_deterministic_h3_attention(
          nullptr, reinterpret_cast<const __nv_bfloat16*>(q.get()),
          reinterpret_cast<const __nv_bfloat16*>(k.get()),
          reinterpret_cast<const __nv_bfloat16*>(v.get()),
          reinterpret_cast<__nv_bfloat16*>(out.get()), ranges.get(),
          sequence, heads, dim, exact_attention_scale(dim));
    });
    out.copy_to_host(expected_band.data(), count);
    cublasHandle_t handle = nullptr;
    VIDFAB_CUBLAS_CHECK(cublasCreate(&handle));
    cuda::Workspace workspace;
    cuda::AttentionConfig config;
    config.seq_len = sequence;
    config.num_heads = heads;
    config.head_dim = dim;
    config.scale = exact_attention_scale(dim);
    config.band_ranges = nullptr;
    shipped_full_ms = timed([&] {
      cuda::attention_forward(
          handle, nullptr, reinterpret_cast<const __nv_bfloat16*>(q.get()),
          reinterpret_cast<const __nv_bfloat16*>(k.get()),
          reinterpret_cast<const __nv_bfloat16*>(v.get()),
          reinterpret_cast<__nv_bfloat16*>(out.get()), config,
          cuda::AttentionBackend::kFused, workspace);
    });
    std::vector<uint16_t> shipped(count);
    out.copy_to_host(shipped.data(), count);
    for (size_t i = 0; i < count; ++i) {
      if (shipped[i] != expected_full[i]) ++shipped_differences;
      shipped_max_abs = std::max(
          shipped_max_abs,
          std::abs(bf16_to_f32(shipped[i]) - bf16_to_f32(expected_full[i])));
    }
    config.band_ranges = ranges.get();
    shipped_band_ms = timed([&] {
      cuda::attention_forward(
          handle, nullptr, reinterpret_cast<const __nv_bfloat16*>(q.get()),
          reinterpret_cast<const __nv_bfloat16*>(k.get()),
          reinterpret_cast<const __nv_bfloat16*>(v.get()),
          reinterpret_cast<__nv_bfloat16*>(out.get()), config,
          cuda::AttentionBackend::kFused, workspace);
    });
    cublasDestroy(handle);
  }

  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  CHECK(!physical.empty());
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  options.enable_shader_float16 = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  CHECK(vk.exact_h3_attention());
  const uint64_t shape[] = {sequence, heads, dim};
  const TensorLayout tensor_layout = TensorLayout::contiguous(shape, 3);
  DeviceTensor q = vk.allocate(tensor_layout, ScalarType::kBFloat16);
  DeviceTensor k = vk.allocate(tensor_layout, ScalarType::kBFloat16);
  DeviceTensor v = vk.allocate(tensor_layout, ScalarType::kBFloat16);
  DeviceTensor out = vk.allocate(tensor_layout, ScalarType::kBFloat16);
  vk.upload_bytes(q, host.data(), count * sizeof(uint16_t));
  vk.upload_bytes(k, host.data(), count * sizeof(uint16_t));
  vk.upload_bytes(v, host.data(), count * sizeof(uint16_t));
  H3AttentionPlan plan = H3AttentionPlan::create(
      vk, {sequence, heads, dim, exact_attention_scale(dim)});
  H3AttentionRanges band_table = H3AttentionRanges::create(
      vk, sequence, band.ranges.data(),
      static_cast<uint32_t>(band.ranges.size()));
  H3AttentionRanges wide_table = H3AttentionRanges::create(
      vk, sequence, wide.ranges.data(),
      static_cast<uint32_t>(wide.ranges.size()));
  auto timed_vk = [&](const H3AttentionRanges* selected) {
    const auto begin = std::chrono::steady_clock::now();
    TensorBatch batch = vk.begin_batch();
    plan.record(batch, q, k, v, out, selected);
    batch.submit().wait();
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
  };
  const double vulkan_full_ms = timed_vk(nullptr);
  std::vector<uint16_t> got(count);
  vk.download_bytes(out, got.data(), count * sizeof(uint16_t));
  CHECK(got == expected_full);
  const double vulkan_band_ms = timed_vk(&band_table);
  vk.download_bytes(out, got.data(), count * sizeof(uint16_t));
  CHECK(got == expected_band);
  const double vulkan_wide_ms = timed_vk(&wide_table);
  vk.download_bytes(out, got.data(), count * sizeof(uint16_t));
  CHECK(got == expected_full);
  const double direct_mib = double(count * sizeof(uint16_t) * 4) / 1048576.0;
  std::printf(
      "  H3 real S37727 H56 D128: exact CUDA full %.3f ms/band %.3f ms; Vulkan full %.3f ms/band %.3f ms/wide %.3f ms; shipped fused full %.3f ms/band %.3f ms; shipped drift %zu/%zu maxabs %.7g; direct QKV/out %.2f MiB, range %zu bytes, scratch 0\n",
      cuda_full_ms, cuda_band_ms, vulkan_full_ms, vulkan_band_ms,
      vulkan_wide_ms, shipped_full_ms, shipped_band_ms,
      shipped_differences, count, shipped_max_abs, direct_mib,
      band.ranges.size() * sizeof(int32_t));
}

VIDFAB_TEST(cuda_vulkan_h3_capture_replay) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  const char* path = std::getenv("VIDFAB_H3_ATTENTION_CAPTURE");
  if (!path || !*path) return;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;

  SolCaptureHeader header{};
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(&header), sizeof(header)) ||
      std::memcmp(header.magic, "VFSOLQKV", 8) != 0 || header.version != 1 ||
      header.header_bytes != sizeof(header) || header.head_dim != 128) {
    throw std::runtime_error("invalid H3 capture replay header");
  }
  const uint64_t elements64 = uint64_t(header.seq_len) * header.num_heads *
                              header.head_dim;
  if (header.tensor_elements != elements64 || elements64 > SIZE_MAX / 6)
    throw std::runtime_error("invalid H3 capture replay shape");
  const size_t count = static_cast<size_t>(elements64);
  std::vector<uint16_t> captured(count * 3);
  if (!input.read(reinterpret_cast<char*>(captured.data()),
                  static_cast<std::streamsize>(captured.size() * 2)) ||
      input.peek() != std::ifstream::traits_type::eof()) {
    throw std::runtime_error("truncated H3 capture replay");
  }
  uint64_t input_hash = 1469598103934665603ull;
  uint64_t subnormal[3]{}, nonfinite[3]{};
  float maximum[3]{};
  for (size_t i = 0; i < captured.size(); ++i) {
    const uint16_t bits = captured[i];
    input_hash ^= bits & 0xffu; input_hash *= 1099511628211ull;
    input_hash ^= bits >> 8; input_hash *= 1099511628211ull;
    const size_t tensor = i / count;
    const uint16_t exponent = bits & 0x7f80u;
    subnormal[tensor] += exponent == 0 && (bits & 0x007fu) != 0;
    nonfinite[tensor] += exponent == 0x7f80u;
    if (exponent != 0x7f80u)
      maximum[tensor] = std::max(maximum[tensor], std::abs(bf16_to_f32(bits)));
  }

  cuda::DeviceBuffer<uint16_t> cq(count), ck(count), cv(count),
      exact_output(count), shipped_output(count);
  cq.copy_from_host(captured.data(), count);
  ck.copy_from_host(captured.data() + count, count);
  cv.copy_from_host(captured.data() + count * 2, count);
  auto cuda_time = [&](auto&& launch) {
    cudaEvent_t begin{}, end{};
    VIDFAB_CUDA_CHECK(cudaEventCreate(&begin));
    VIDFAB_CUDA_CHECK(cudaEventCreate(&end));
    VIDFAB_CUDA_CHECK(cudaEventRecord(begin));
    launch();
    VIDFAB_CUDA_CHECK(cudaEventRecord(end));
    VIDFAB_CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds = 0;
    VIDFAB_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
    cudaEventDestroy(begin); cudaEventDestroy(end);
    return milliseconds;
  };
  const float scale = exact_attention_scale(header.head_dim);
  const float exact_cuda_ms = cuda_time([&] {
    cuda::launch_deterministic_h3_attention(
        nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
        reinterpret_cast<const __nv_bfloat16*>(ck.get()),
        reinterpret_cast<const __nv_bfloat16*>(cv.get()),
        reinterpret_cast<__nv_bfloat16*>(exact_output.get()), nullptr,
        header.seq_len, header.num_heads, header.head_dim, scale);
  });
  cublasHandle_t blas{};
  VIDFAB_CUBLAS_CHECK(cublasCreate(&blas));
  cuda::Workspace workspace;
  cuda::AttentionConfig config;
  config.seq_len = header.seq_len;
  config.num_heads = header.num_heads;
  config.head_dim = header.head_dim;
  config.scale = scale;
  const float shipped_ms = cuda_time([&] {
    cuda::attention_forward(
        blas, nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
        reinterpret_cast<const __nv_bfloat16*>(ck.get()),
        reinterpret_cast<const __nv_bfloat16*>(cv.get()),
        reinterpret_cast<__nv_bfloat16*>(shipped_output.get()), config,
        cuda::AttentionBackend::kFused, workspace);
  });
  cublasDestroy(blas);
  std::vector<uint16_t> expected(count), shipped(count);
  exact_output.copy_to_host(expected.data(), count);
  shipped_output.copy_to_host(shipped.data(), count);
  size_t differences = 0;
  double error2 = 0, reference2 = 0;
  float max_abs = 0;
  uint64_t output_hash = 1469598103934665603ull;
  for (size_t i = 0; i < count; ++i) {
    output_hash ^= expected[i] & 0xffu; output_hash *= 1099511628211ull;
    output_hash ^= expected[i] >> 8; output_hash *= 1099511628211ull;
    differences += expected[i] != shipped[i];
    const double exact = bf16_to_f32(expected[i]);
    const double approximate = bf16_to_f32(shipped[i]);
    const double error = exact - approximate;
    error2 += error * error;
    reference2 += exact * exact;
    max_abs = std::max(max_abs, static_cast<float>(std::abs(error)));
  }

  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  options.enable_shader_float16 = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  CHECK(vk.exact_h3_attention());
  const uint64_t shape[] = {header.seq_len, header.num_heads, header.head_dim};
  const TensorLayout layout = TensorLayout::contiguous(shape, 3);
  DeviceTensor q = vk.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor k = vk.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor v = vk.allocate(layout, ScalarType::kBFloat16);
  DeviceTensor output = vk.allocate(layout, ScalarType::kBFloat16);
  vk.upload_bytes(q, captured.data(), count * 2);
  vk.upload_bytes(k, captured.data() + count, count * 2);
  vk.upload_bytes(v, captured.data() + count * 2, count * 2);
  H3AttentionPlan plan = H3AttentionPlan::create(
      vk, {header.seq_len, header.num_heads, header.head_dim, scale});
  const auto begin = std::chrono::steady_clock::now();
  TensorBatch batch = vk.begin_batch();
  plan.record(batch, q, k, v, output);
  batch.submit().wait();
  const double vulkan_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - begin).count();
  std::vector<uint16_t> got(count);
  vk.download_bytes(output, got.data(), count * 2);
  CHECK(got == expected);
  std::printf(
      "  H3 capture S%u H%u D%u step%d layer%d: input FNV64 %016llx, exact output %016llx; max Q/K/V %.7g/%.7g/%.7g, subnormal %llu/%llu/%llu, nonfinite %llu/%llu/%llu; exact CUDA %.3f ms Vulkan %.3f ms shipped %.3f ms, shipped drift %zu/%zu relL2 %.7g maxabs %.7g\n",
      header.seq_len, header.num_heads, header.head_dim, header.denoise_step,
      header.layer, static_cast<unsigned long long>(input_hash),
      static_cast<unsigned long long>(output_hash), maximum[0], maximum[1],
      maximum[2], static_cast<unsigned long long>(subnormal[0]),
      static_cast<unsigned long long>(subnormal[1]),
      static_cast<unsigned long long>(subnormal[2]),
      static_cast<unsigned long long>(nonfinite[0]),
      static_cast<unsigned long long>(nonfinite[1]),
      static_cast<unsigned long long>(nonfinite[2]), exact_cuda_ms, vulkan_ms,
      shipped_ms, differences, count,
      reference2 == 0 ? 0 : std::sqrt(error2 / reference2), max_abs);
}

VIDFAB_TEST(cuda_vulkan_exact_causal_gqa_attention) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  if (!vk.exact_causal_gqa_attention()) return;

  constexpr uint32_t sequence = 129;
  constexpr uint32_t query_heads = 64;
  constexpr uint32_t kv_heads = 8;
  constexpr uint32_t dim = 128;
  const size_t q_count = size_t(sequence) * query_heads * dim;
  const size_t kv_count = size_t(sequence) * kv_heads * dim;
  std::vector<uint16_t> hq(q_count), hk(kv_count), hv(kv_count);
  for (size_t i = 0; i < q_count; ++i)
    hq[i] = f32_to_bf16(float(int(i % 29) - 14) / 32.0f);
  for (size_t i = 0; i < kv_count; ++i) {
    hk[i] = f32_to_bf16(float(int(i % 31) - 15) / 32.0f);
    hv[i] = f32_to_bf16(float(int(i % 37) - 18) / 16.0f);
  }
  hq[0] = 0x0001u;
  hq[1] = 0x807fu;
  hk[0] = 0x007fu;
  hk[1] = 0x8001u;
  hv[0] = 0x0001u;
  hv[1] = 0x807fu;
  cuda::DeviceBuffer<uint16_t> cq(q_count), ck(kv_count), cv(kv_count),
      co(q_count), co_repeat(q_count);
  cq.copy_from_host(hq.data(), q_count);
  ck.copy_from_host(hk.data(), kv_count);
  cv.copy_from_host(hv.data(), kv_count);
  cuda::launch_deterministic_causal_gqa_attention(
      nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
      reinterpret_cast<const __nv_bfloat16*>(ck.get()),
      reinterpret_cast<const __nv_bfloat16*>(cv.get()),
      reinterpret_cast<__nv_bfloat16*>(co.get()), sequence, query_heads,
      kv_heads, dim, exact_attention_scale(dim));
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> expected(q_count);
  co.copy_to_host(expected.data(), q_count);
  cuda::launch_deterministic_causal_gqa_attention(
      nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
      reinterpret_cast<const __nv_bfloat16*>(ck.get()),
      reinterpret_cast<const __nv_bfloat16*>(cv.get()),
      reinterpret_cast<__nv_bfloat16*>(co_repeat.get()), sequence, query_heads,
      kv_heads, dim, exact_attention_scale(dim));
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> expected_repeat(q_count);
  co_repeat.copy_to_host(expected_repeat.data(), q_count);
  CHECK(expected_repeat == expected);

  const uint64_t q_shape[] = {sequence, query_heads, dim};
  const uint64_t kv_shape[] = {sequence, kv_heads, dim};
  DeviceTensor q = vk.allocate(TensorLayout::contiguous(q_shape, 3),
                               ScalarType::kBFloat16);
  DeviceTensor k = vk.allocate(TensorLayout::contiguous(kv_shape, 3),
                               ScalarType::kBFloat16);
  DeviceTensor v = vk.allocate(TensorLayout::contiguous(kv_shape, 3),
                               ScalarType::kBFloat16);
  DeviceTensor out = vk.allocate(TensorLayout::contiguous(q_shape, 3),
                                 ScalarType::kBFloat16);
  vk.upload_bytes(q, hq.data(), q_count * sizeof(uint16_t));
  vk.upload_bytes(k, hk.data(), kv_count * sizeof(uint16_t));
  vk.upload_bytes(v, hv.data(), kv_count * sizeof(uint16_t));
  CausalGQAAttentionPlanDesc desc{sequence, query_heads, kv_heads, dim,
                                  exact_attention_scale(dim)};
  CausalGQAAttentionPlan plan = CausalGQAAttentionPlan::create(vk, desc);
  TensorBatch batch = vk.begin_batch();
  plan.record(batch, q, k, v, out, 0, 65, 0);
  plan.record(batch, q, k, v, out, 65, sequence - 65, 65);
  batch.submit().wait();
  std::vector<uint16_t> got(q_count);
  vk.download_bytes(out, got.data(), got.size() * sizeof(uint16_t));
  if (got != expected) {
    for (size_t i = 0; i < got.size(); ++i) {
      if (got[i] != expected[i]) {
        std::printf("  causal GQA mismatch %zu: CUDA %04x Vulkan %04x\n",
                    i, expected[i], got[i]);
        break;
      }
    }
  }
  CHECK(got == expected);
  CHECK(got[0] == 0x0000u);
  // The negative subnormal is first canonicalized to -0; adding that product
  // to the +0 accumulator has the specified round-to-nearest result +0.
  CHECK(got[1] == 0x0000u);
  // Causal row zero is exactly V row zero from the mapped KV head.
  for (uint32_t h = 0; h < query_heads; ++h) {
    const uint32_t kv = h / (query_heads / kv_heads);
    for (uint32_t d = 0; d < dim; ++d) {
      uint16_t expected_value = hv[size_t(kv) * dim + d];
      if ((expected_value & 0x7f80u) == 0u)
        expected_value = 0u;
      CHECK(got[size_t(h) * dim + d] == expected_value);
    }
  }

  // Exercise both sides of the 128-key recurrence boundary. Future K/V rows
  // carry large sentinels; row zero must still be exactly V[0] for each mapped
  // KV head, proving that the causal kernel never reads the upper triangle.
  auto run_boundary = [&](uint32_t boundary_sequence, uint32_t first_rows) {
    const size_t boundary_q_count =
        size_t(boundary_sequence) * query_heads * dim;
    const size_t boundary_kv_count =
        size_t(boundary_sequence) * kv_heads * dim;
    std::vector<uint16_t> bq(boundary_q_count, f32_to_bf16(0.0f));
    std::vector<uint16_t> bk(boundary_kv_count);
    std::vector<uint16_t> bv(boundary_kv_count);
    for (uint32_t row = 0; row < boundary_sequence; ++row) {
      for (uint32_t head = 0; head < kv_heads; ++head) {
        for (uint32_t d = 0; d < dim; ++d) {
          const size_t index =
              (size_t(row) * kv_heads + head) * dim + d;
          bk[index] = f32_to_bf16(row == 0 ? 0.0f : 31.0f);
          bv[index] = f32_to_bf16(
              row == 0 ? float(int(head) - 4) / 8.0f
                       : float(int((row + head + d) % 15) - 7) / 4.0f);
        }
      }
    }
    cuda::DeviceBuffer<uint16_t> dcq(boundary_q_count), dck(boundary_kv_count),
        dcv(boundary_kv_count), dco(boundary_q_count);
    dcq.copy_from_host(bq.data(), bq.size());
    dck.copy_from_host(bk.data(), bk.size());
    dcv.copy_from_host(bv.data(), bv.size());
    cuda::launch_deterministic_causal_gqa_attention(
        nullptr, reinterpret_cast<const __nv_bfloat16*>(dcq.get()),
        reinterpret_cast<const __nv_bfloat16*>(dck.get()),
        reinterpret_cast<const __nv_bfloat16*>(dcv.get()),
        reinterpret_cast<__nv_bfloat16*>(dco.get()), boundary_sequence,
        query_heads, kv_heads, dim, exact_attention_scale(dim));
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> boundary_expected(boundary_q_count);
    dco.copy_to_host(boundary_expected.data(), boundary_expected.size());

    const uint64_t boundary_q_shape[] = {boundary_sequence, query_heads, dim};
    const uint64_t boundary_kv_shape[] = {boundary_sequence, kv_heads, dim};
    DeviceTensor vq_boundary = vk.allocate(
        TensorLayout::contiguous(boundary_q_shape, 3), ScalarType::kBFloat16);
    DeviceTensor vk_boundary = vk.allocate(
        TensorLayout::contiguous(boundary_kv_shape, 3), ScalarType::kBFloat16);
    DeviceTensor vv_boundary = vk.allocate(
        TensorLayout::contiguous(boundary_kv_shape, 3), ScalarType::kBFloat16);
    DeviceTensor vo_boundary = vk.allocate(
        TensorLayout::contiguous(boundary_q_shape, 3), ScalarType::kBFloat16);
    vk.upload_bytes(vq_boundary, bq.data(), bq.size() * sizeof(uint16_t));
    vk.upload_bytes(vk_boundary, bk.data(), bk.size() * sizeof(uint16_t));
    vk.upload_bytes(vv_boundary, bv.data(), bv.size() * sizeof(uint16_t));
    CausalGQAAttentionPlan boundary_plan = CausalGQAAttentionPlan::create(
        vk, {boundary_sequence, query_heads, kv_heads, dim,
             exact_attention_scale(dim)});
    TensorBatch boundary_batch = vk.begin_batch();
    boundary_plan.record(boundary_batch, vq_boundary, vk_boundary, vv_boundary,
                         vo_boundary, 0, first_rows, 0);
    if (first_rows < boundary_sequence)
      boundary_plan.record(boundary_batch, vq_boundary, vk_boundary,
                           vv_boundary, vo_boundary, first_rows,
                           boundary_sequence - first_rows, first_rows);
    boundary_batch.submit().wait();
    std::vector<uint16_t> boundary_got(boundary_q_count);
    vk.download_bytes(vo_boundary, boundary_got.data(),
                      boundary_got.size() * sizeof(uint16_t));
    CHECK(boundary_got == boundary_expected);
    for (uint32_t head = 0; head < query_heads; ++head) {
      const uint32_t mapped = head / (query_heads / kv_heads);
      for (uint32_t d = 0; d < dim; ++d)
        CHECK(boundary_got[size_t(head) * dim + d] ==
              bv[size_t(mapped) * dim + d]);
    }
  };
  run_boundary(1, 1);
  run_boundary(127, 63);
  run_boundary(128, 64);
  run_boundary(257, 129);

  DeviceTensor out_second = vk.allocate(TensorLayout::contiguous(q_shape, 3),
                                        ScalarType::kBFloat16);
  auto submit_full = [&](DeviceTensor& selected_output) {
    TensorBatch selected = vk.begin_batch();
    plan.record(selected, q, k, v, selected_output);
    return selected.submit();
  };
  Submission first_job = submit_full(out);
  Submission second_job = submit_full(out_second);
  CHECK(second_job.value() > first_job.value());
  Submission third_job = submit_full(out);
  CHECK(third_job.value() > second_job.value());
  first_job.wait();
  second_job.wait();
  third_job.wait();
  std::vector<uint16_t> second_got(q_count);
  vk.download_bytes(out_second, second_got.data(), second_got.size() * 2);
  CHECK(second_got == expected);
  vk.download_bytes(out, got.data(), got.size() * 2);
  CHECK(got == expected);
  const uint64_t stable_reserved = vk.reserved_bytes();
  const uint64_t stable_descriptors = vk.descriptor_set_allocations();
  for (int repeat = 0; repeat < 4; ++repeat) {
    Submission a = submit_full(out), b = submit_full(out_second),
               c = submit_full(out);
    CHECK(b.value() > a.value() && c.value() > b.value());
    a.wait(); b.wait(); c.wait();
    CHECK(vk.reserved_bytes() == stable_reserved);
    CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  }

  // Validation happens before recording and a rejected call leaves the batch
  // usable. By contrast, exceeding the bounded 32-op command list poisons it.
  DeviceTensor wrong_type =
      vk.allocate(TensorLayout::contiguous(q_shape, 3), ScalarType::kFloat32);
  {
    TensorBatch recover = vk.begin_batch();
    bool rejected = false;
    try { plan.record(recover, q, k, v, wrong_type); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    rejected = false;
    try { plan.record(recover, q, k, v, q); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    plan.record(recover, q, k, v, out);
    recover.submit().wait();
  }
  {
    TensorBatch full = vk.begin_batch();
    for (int op = 0; op < 32; ++op)
      plan.record(full, q, k, v, out, 0, 1, 0);
    full.submit().wait();
  }
  const uint64_t saturated_reserved = vk.reserved_bytes();
  const uint64_t saturated_descriptors = vk.descriptor_set_allocations();
  {
    TensorBatch overflow = vk.begin_batch();
    for (int op = 0; op < 32; ++op)
      plan.record(overflow, q, k, v, out, 0, 1, 0);
    bool rejected = false;
    try { plan.record(overflow, q, k, v, out, 0, 1, 0); }
    catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected);
    bool submit_rejected = false;
    try { (void)overflow.submit(); }
    catch (const std::logic_error&) { submit_rejected = true; }
    CHECK(submit_rejected);
  }
  CHECK(vk.reserved_bytes() == saturated_reserved);
  CHECK(vk.descriptor_set_allocations() == saturated_descriptors);
  {
    TensorBatch full = vk.begin_batch();
    for (int op = 0; op < 32; ++op)
      plan.record(full, q, k, v, out, 0, 1, 0);
    full.submit().wait();
  }
  CHECK(vk.reserved_bytes() == saturated_reserved);
  CHECK(vk.descriptor_set_allocations() == saturated_descriptors);

  // The submitted job retains input allocations after all caller wrappers are
  // dropped. The kept output remains readable after token completion.
  Submission retained = submit_full(out);
  q = DeviceTensor();
  k = DeviceTensor();
  v = DeviceTensor();
  retained.wait();
  vk.download_bytes(out, got.data(), got.size() * 2);
  CHECK(got == expected);
}

VIDFAB_TEST(cuda_vulkan_causal_gqa_real_timing) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!std::getenv("VIDFAB_CAUSAL_GQA_BENCH")) return;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  if (!vk.exact_causal_gqa_attention()) return;

  for (uint32_t sequence : {132u, 8192u}) {
    constexpr uint32_t query_heads = 64, kv_heads = 8, dim = 128;
    const size_t q_count = size_t(sequence) * query_heads * dim;
    const size_t kv_count = size_t(sequence) * kv_heads * dim;
    std::vector<uint16_t> hq(q_count, f32_to_bf16(0.03125f));
    std::vector<uint16_t> hk(kv_count, f32_to_bf16(-0.015625f));
    std::vector<uint16_t> hv(kv_count, f32_to_bf16(0.0625f));
    cuda::DeviceBuffer<uint16_t> cq(q_count), ck(kv_count), cv(kv_count),
        co(q_count);
    cq.copy_from_host(hq.data(), hq.size());
    ck.copy_from_host(hk.data(), hk.size());
    cv.copy_from_host(hv.data(), hv.size());
    const uint64_t q_shape[] = {sequence, query_heads, dim};
    const uint64_t kv_shape[] = {sequence, kv_heads, dim};
    DeviceTensor q = vk.allocate(TensorLayout::contiguous(q_shape, 3),
                                 ScalarType::kBFloat16);
    DeviceTensor k = vk.allocate(TensorLayout::contiguous(kv_shape, 3),
                                 ScalarType::kBFloat16);
    DeviceTensor v = vk.allocate(TensorLayout::contiguous(kv_shape, 3),
                                 ScalarType::kBFloat16);
    DeviceTensor out = vk.allocate(TensorLayout::contiguous(q_shape, 3),
                                   ScalarType::kBFloat16);
    vk.upload_bytes(q, hq.data(), hq.size() * 2);
    vk.upload_bytes(k, hk.data(), hk.size() * 2);
    vk.upload_bytes(v, hv.data(), hv.size() * 2);
    CausalGQAAttentionPlan plan = CausalGQAAttentionPlan::create(
        vk, {sequence, query_heads, kv_heads, dim, exact_attention_scale(dim)});
    auto submit_vulkan = [&] {
      TensorBatch batch = vk.begin_batch();
      plan.record(batch, q, k, v, out);
      return batch.submit();
    };
    auto launch_cuda = [&] {
      cuda::launch_deterministic_causal_gqa_attention(
          nullptr, reinterpret_cast<const __nv_bfloat16*>(cq.get()),
          reinterpret_cast<const __nv_bfloat16*>(ck.get()),
          reinterpret_cast<const __nv_bfloat16*>(cv.get()),
          reinterpret_cast<__nv_bfloat16*>(co.get()), sequence, query_heads,
          kv_heads, dim, exact_attention_scale(dim));
    };
    launch_cuda();
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    submit_vulkan().wait();
    cudaEvent_t start = nullptr, stop = nullptr;
    VIDFAB_CUDA_CHECK(cudaEventCreate(&start));
    VIDFAB_CUDA_CHECK(cudaEventCreate(&stop));
    VIDFAB_CUDA_CHECK(cudaEventRecord(start));
    launch_cuda();
    VIDFAB_CUDA_CHECK(cudaEventRecord(stop));
    VIDFAB_CUDA_CHECK(cudaEventSynchronize(stop));
    float cuda_ms = 0.0f;
    VIDFAB_CUDA_CHECK(cudaEventElapsedTime(&cuda_ms, start, stop));
    VIDFAB_CUDA_CHECK(cudaEventDestroy(start));
    VIDFAB_CUDA_CHECK(cudaEventDestroy(stop));
    const auto begin = std::chrono::steady_clock::now();
    submit_vulkan().wait();
    const double vulkan_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    std::vector<uint16_t> cuda_out(q_count), vulkan_out(q_count);
    co.copy_to_host(cuda_out.data(), cuda_out.size());
    vk.download_bytes(out, vulkan_out.data(), vulkan_out.size() * 2);
    CHECK(cuda_out == vulkan_out);
    const uint64_t direct_bytes =
        (uint64_t(q_count) * 2 + uint64_t(kv_count) * 2) * sizeof(uint16_t);
    std::printf("  causal GQA L%u H64/KV8/D128: CUDA %.3f ms, Vulkan %.3f ms, "
                "direct Q/K/V/out %.2f MiB, pool used %.2f MiB, context reserved %.2f MiB, descriptors %llu\n",
                sequence, cuda_ms, vulkan_ms, direct_bytes / (1024.0 * 1024.0),
                vk.pooled_used_bytes() / (1024.0 * 1024.0),
                vk.reserved_bytes() / (1024.0 * 1024.0),
                static_cast<unsigned long long>(vk.descriptor_set_allocations()));
  }
}

VIDFAB_TEST(cuda_vulkan_tensor_exact_conversion_and_layout_ops) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);

  // 335 values exercise a non-workgroup tail; the prefix pins the CUDA NaN,
  // Inf, signed-zero, subnormal and half-way conversion policy exactly.
  constexpr int rows = 5;
  constexpr int cols = 67;
  constexpr size_t count = static_cast<size_t>(rows) * cols;
  std::vector<float> input(count), bias(cols);
  for (size_t i = 0; i < count; ++i) {
    input[i] = static_cast<float>(static_cast<int>(i % 101) - 50) / 64.0f;
  }
  const uint32_t special[] = {
      0x00000000u, 0x80000000u, 0x00000001u, 0x007fffffu,
      0x00800000u, 0x33800000u, 0x33800001u, 0x387fffffu,
      0x38800000u, 0x3f800000u, 0x477fe000u, 0x7f800000u,
      0xff800000u, 0x7fc12345u, 0x7fa54321u, 0xffc12345u,
      0x33000000u, 0x33000001u, 0x3f808000u, 0x3f808001u,
      0x00800000u};
  std::memcpy(input.data(), special, sizeof(special));
  for (int i = 0; i < cols; ++i) bias[i] = static_cast<float>((i % 13) - 6) / 32.0f;
  for (int i = 0; i < 4; ++i) bias[i] = 0.0f;
  // Two normal operands whose exact sum is the smallest subnormal.
  const uint32_t normal_above_min = 0x00800001u;
  const uint32_t negative_min_normal = 0x80800000u;
  std::memcpy(&input[4], &normal_above_min, sizeof(float));
  std::memcpy(&bias[4], &negative_min_normal, sizeof(float));

  cuda::DeviceBuffer<float> c_input(count), c_bf16_back(count), c_f16_back(count);
  cuda::DeviceBuffer<uint16_t> c_bf16(count), c_f16(count);
  c_input.copy_from_host(input.data(), count);
  cuda::launch_narrow_to_bf16(
      c_input.get(), reinterpret_cast<__nv_bfloat16*>(c_bf16.get()), count, nullptr);
  cuda::launch_widen_bf16(reinterpret_cast<const __nv_bfloat16*>(c_bf16.get()),
                          c_bf16_back.get(), count, nullptr);
  cuda::launch_narrow_f16(c_input.get(), c_f16.get(), count, nullptr);
  cuda::launch_widen_f16(c_f16.get(), c_f16_back.get(), count, nullptr);

  cuda::DeviceBuffer<float> c_transpose(count), c_biased(count);
  cuda::DeviceBuffer<float> c_bias(cols);
  c_bias.copy_from_host(bias.data(), bias.size());
  VIDFAB_CUDA_CHECK(cudaMemcpy(c_biased.get(), c_input.get(), count * sizeof(float),
                               cudaMemcpyDeviceToDevice));
  cuda::launch_transpose_cn_to_nc(c_input.get(), c_transpose.get(), rows, cols, nullptr);
  cuda::launch_add_bias(c_biased.get(), c_bias.get(), rows, cols, nullptr);

  constexpr int matrix_rows = 7;
  constexpr int selected_rows = 5;
  constexpr int matrix_cols = 67;
  constexpr size_t matrix_count = static_cast<size_t>(matrix_rows) * matrix_cols;
  constexpr size_t selected_count = static_cast<size_t>(selected_rows) * matrix_cols;
  std::vector<float> matrix(matrix_count), scatter_initial(matrix_count, -99.0f);
  for (size_t i = 0; i < matrix.size(); ++i)
    matrix[i] = static_cast<float>(static_cast<int>(i) - 170) / 16.0f;
  const int32_t indices_host[selected_rows] = {6, 0, 4, 1, 3};
  cuda::DeviceBuffer<float> c_matrix(matrix_count), c_gathered(selected_count),
      c_scattered(matrix_count);
  cuda::DeviceBuffer<int32_t> c_indices(selected_rows);
  c_matrix.copy_from_host(matrix.data(), matrix.size());
  c_indices.copy_from_host(indices_host, selected_rows);
  c_scattered.copy_from_host(scatter_initial.data(), scatter_initial.size());
  cuda::launch_gather_rows_f32(c_matrix.get(), c_indices.get(), c_gathered.get(),
                               selected_rows, matrix_cols, nullptr);
  cuda::launch_scatter_rows_f32(c_gathered.get(), c_indices.get(), c_scattered.get(),
                                selected_rows, matrix_cols, nullptr);

  constexpr int heads = 3, sequence = 5, head_dim = 7;
  constexpr size_t heads_count = static_cast<size_t>(heads) * sequence * head_dim;
  std::vector<float> heads_input(heads_count);
  for (size_t i = 0; i < heads_count; ++i) heads_input[i] = input[i];
  cuda::DeviceBuffer<float> c_heads(heads_count);
  cuda::DeviceBuffer<uint16_t> c_tokens(heads_count);
  c_heads.copy_from_host(heads_input.data(), heads_input.size());
  cuda::launch_heads_to_tokens_bf16(
      c_heads.get(), reinterpret_cast<__nv_bfloat16*>(c_tokens.get()), sequence,
      heads, head_dim, nullptr);

  // Every axis is nontrivial, including temporal patching. 288 output values
  // leave a 32-invocation workgroup tail and catch an output-plane expression
  // that accidentally omits patch_t.
  constexpr int depth_t = 2, depth_h = 2, depth_w = 3, depth_channels = 3;
  constexpr int patch_t = 2, patch = 2;
  constexpr size_t depth_count = static_cast<size_t>(depth_t) * depth_h * depth_w *
                                 depth_channels * patch_t * patch * patch;
  std::vector<float> depth_input(depth_count);
  for (size_t i = 0; i < depth_input.size(); ++i)
    depth_input[i] = static_cast<float>(i) + 0.25f;
  cuda::DeviceBuffer<float> c_depth_input(depth_count), c_depth_output(depth_count);
  c_depth_input.copy_from_host(depth_input.data(), depth_input.size());
  cuda::launch_depth_to_space(c_depth_input.get(), c_depth_output.get(), depth_t,
                              depth_h, depth_w, depth_channels, patch_t, patch, nullptr);

  // Widening also has an explicit CUDA NaN policy for arbitrary checkpoint
  // half bits, independently of values produced by the narrowing kernel.
  constexpr size_t raw_half_count = 1u << 16;
  std::vector<uint16_t> raw_half_host(raw_half_count);
  for (size_t i = 0; i < raw_half_count; ++i)
    raw_half_host[i] = static_cast<uint16_t>(i);
  cuda::DeviceBuffer<uint16_t> c_raw_half(raw_half_count);
  cuda::DeviceBuffer<float> c_raw_half_wide(raw_half_count);
  c_raw_half.copy_from_host(raw_half_host.data(), raw_half_count);
  cuda::launch_widen_f16(c_raw_half.get(), c_raw_half_wide.get(), raw_half_count,
                         nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());

  const uint64_t shape_extents[] = {rows, cols};
  const uint64_t transpose_extents[] = {cols, rows};
  const TensorLayout shape = TensorLayout::contiguous(shape_extents, 2);
  DeviceTensor v_input = vk.allocate(shape);
  DeviceTensor v_bf16 = vk.allocate(shape, ScalarType::kBFloat16);
  DeviceTensor v_f16 = vk.allocate(shape, ScalarType::kFloat16);
  DeviceTensor v_bf16_back = vk.allocate(shape);
  DeviceTensor v_f16_back = vk.allocate(shape);
  DeviceTensor v_transpose = vk.allocate(TensorLayout::contiguous(transpose_extents, 2));
  DeviceTensor v_bias = vk.allocate(TensorLayout::contiguous(&transpose_extents[0], 1));
  DeviceTensor v_biased = vk.allocate(shape);
  vk.upload(v_input, input.data(), count);
  vk.upload(v_bias, bias.data(), bias.size());

  const uint64_t matrix_extents[] = {matrix_rows, matrix_cols};
  const uint64_t selected_extents[] = {selected_rows, matrix_cols};
  const uint64_t index_extent = selected_rows;
  DeviceTensor v_matrix = vk.allocate(TensorLayout::contiguous(matrix_extents, 2));
  DeviceTensor v_indices = vk.allocate(TensorLayout::contiguous(&index_extent, 1),
                                       ScalarType::kInt32);
  DeviceTensor v_gathered = vk.allocate(TensorLayout::contiguous(selected_extents, 2));
  DeviceTensor v_scattered = vk.allocate(TensorLayout::contiguous(matrix_extents, 2));
  vk.upload(v_matrix, matrix.data(), matrix.size());
  vk.upload_bytes(v_indices, indices_host, sizeof(indices_host));
  vk.upload(v_scattered, scatter_initial.data(), scatter_initial.size());

  const uint64_t heads_extent = heads_count;
  DeviceTensor v_heads = vk.allocate(TensorLayout::contiguous(&heads_extent, 1));
  DeviceTensor v_tokens = vk.allocate(TensorLayout::contiguous(&heads_extent, 1),
                                      ScalarType::kBFloat16);
  vk.upload(v_heads, heads_input.data(), heads_input.size());
  const uint64_t depth_extent = depth_count;
  DeviceTensor v_depth_input = vk.allocate(TensorLayout::contiguous(&depth_extent, 1));
  DeviceTensor v_depth_output = vk.allocate(TensorLayout::contiguous(&depth_extent, 1));
  vk.upload(v_depth_input, depth_input.data(), depth_input.size());
  const uint64_t raw_half_extent = raw_half_count;
  DeviceTensor v_raw_half = vk.allocate(TensorLayout::contiguous(&raw_half_extent, 1),
                                        ScalarType::kFloat16);
  DeviceTensor v_raw_half_wide = vk.allocate(TensorLayout::contiguous(&raw_half_extent, 1));
  vk.upload_bytes(v_raw_half, raw_half_host.data(),
                  raw_half_host.size() * sizeof(uint16_t));

  // All operators are one device-only Vulkan batch: there is no host boundary
  // between conversion, indexed movement, elementwise, and layout work.
  TensorBatch batch = vk.begin_batch();
  batch.convert(v_input, v_bf16);
  batch.convert(v_bf16, v_bf16_back);
  batch.convert(v_input, v_f16);
  batch.convert(v_f16, v_f16_back);
  batch.transpose_2d(v_input, v_transpose);
  batch.add_bias(v_input, v_bias, v_biased);
  batch.gather_rows(v_matrix, v_indices, v_gathered);
  batch.scatter_rows(v_gathered, v_indices, v_scattered);
  batch.heads_to_tokens_bf16(v_heads, v_tokens, heads, sequence, head_dim);
  batch.depth_to_space(v_depth_input, v_depth_output, depth_t, depth_h,
                       depth_w, depth_channels, patch_t, patch);
  batch.convert(v_raw_half, v_raw_half_wide);
  batch.submit().wait();

  auto compare_bytes = [&](auto& cuda_buffer, DeviceTensor& vulkan_tensor,
                           size_t elements, const char* label) {
    using Value = std::remove_pointer_t<decltype(cuda_buffer.get())>;
    std::vector<Value> cuda_host(elements), vulkan_host(elements);
    cuda_buffer.copy_to_host(cuda_host.data(), elements);
    vk.download_bytes(vulkan_tensor, vulkan_host.data(), elements * sizeof(Value));
    size_t mismatch = elements;
    for (size_t i = 0; i < elements; ++i) {
      if (std::memcmp(&cuda_host[i], &vulkan_host[i], sizeof(Value)) != 0) {
        mismatch = i;
        break;
      }
    }
    uint64_t cuda_bits = 0, vulkan_bits = 0;
    if (mismatch != elements) {
      std::memcpy(&cuda_bits, &cuda_host[mismatch], sizeof(Value));
      std::memcpy(&vulkan_bits, &vulkan_host[mismatch], sizeof(Value));
    }
    CHECK_MSG(mismatch == elements,
              "CUDA/Vulkan %s mismatch at %zu: %llx != %llx", label,
              mismatch, static_cast<unsigned long long>(cuda_bits),
              static_cast<unsigned long long>(vulkan_bits));
  };
  compare_bytes(c_bf16, v_bf16, count, "fp32-to-bf16");
  compare_bytes(c_bf16_back, v_bf16_back, count, "bf16-to-fp32");
  compare_bytes(c_f16, v_f16, count, "fp32-to-fp16");
  compare_bytes(c_f16_back, v_f16_back, count, "fp16-to-fp32");
  compare_bytes(c_transpose, v_transpose, count, "transpose");
  compare_bytes(c_gathered, v_gathered, selected_count, "gather");
  compare_bytes(c_scattered, v_scattered, matrix_count, "scatter");
  compare_bytes(c_tokens, v_tokens, heads_count, "heads-to-tokens");
  compare_bytes(c_depth_output, v_depth_output, depth_count, "depth-to-space");
  compare_bytes(c_raw_half_wide, v_raw_half_wide, raw_half_count, "arbitrary-fp16-widen");

  bool full_arithmetic_gate_rejected = false;
  try {
    vk.require_full_fp32_arithmetic_exactness();
  } catch (const std::runtime_error&) {
    full_arithmetic_gate_rejected = true;
  }
  CHECK(full_arithmetic_gate_rejected == !vk.full_fp32_arithmetic_exactness());
  std::vector<float> biased_cuda(count), biased_vulkan(count);
  c_biased.copy_to_host(biased_cuda.data(), biased_cuda.size());
  vk.download(v_biased, biased_vulkan.data(), biased_vulkan.size());
  size_t subnormal_cases = 0;
  auto bits_of = [](float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  };
  auto is_nan = [](uint32_t bits) {
    return (bits & 0x7f800000u) == 0x7f800000u &&
           (bits & 0x007fffffu) != 0;
  };
  auto is_subnormal = [](uint32_t bits) {
    return (bits & 0x7f800000u) == 0 && (bits & 0x007fffffu) != 0;
  };
  for (size_t i = 0; i < count; ++i) {
    const uint32_t input_bits = bits_of(input[i]);
    const uint32_t bias_bits = bits_of(bias[i % cols]);
    const uint32_t cuda_bits = bits_of(biased_cuda[i]);
    if (is_nan(input_bits) || is_nan(bias_bits) || is_nan(cuda_bits)) continue;
    const bool uses_subnormal = is_subnormal(input_bits) || is_subnormal(bias_bits) ||
                                is_subnormal(cuda_bits);
    if (uses_subnormal) ++subnormal_cases;
    if (uses_subnormal && !vk.full_fp32_arithmetic_exactness()) continue;
    CHECK_MSG(cuda_bits == bits_of(biased_vulkan[i]),
              "CUDA/Vulkan add-bias mismatch at %zu: %08x != %08x", i,
              cuda_bits, bits_of(biased_vulkan[i]));
  }
  CHECK(subnormal_cases >= 3);

  std::vector<float> depth_vulkan(depth_count);
  vk.download(v_depth_output, depth_vulkan.data(), depth_vulkan.size());
  const int out_t = depth_t * patch_t;
  const int out_h = depth_h * patch;
  const int out_w = depth_w * patch;
  const int patch_volume = patch_t * patch * patch;
  for (int channel = 0; channel < depth_channels; ++channel) {
    for (int ot = 0; ot < out_t; ++ot) {
      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          const size_t output_index =
              ((static_cast<size_t>(channel) * out_t + ot) * out_h + oh) * out_w + ow;
          const int token = ((ot / patch_t) * depth_h + oh / patch) * depth_w + ow / patch;
          const int feature = channel * patch_volume +
              ((ot % patch_t) * patch + oh % patch) * patch + ow % patch;
          const size_t source_index = static_cast<size_t>(token) *
                                          (depth_channels * patch_volume) +
                                      feature;
          CHECK(depth_vulkan[output_index] == depth_input[source_index]);
        }
      }
    }
  }
}

VIDFAB_TEST(cuda_vulkan_tensor_exact_vae_pointwise) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  if (!vk.exact_vae_pointwise()) {
    bool rejected = false;
    try { vk.require_exact_vae_pointwise(); }
    catch (const std::runtime_error&) { rejected = true; }
    CHECK(rejected);
    return;
  }
  vk.require_exact_vae_pointwise();

  auto from_bits = [](uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  };
  constexpr int rows = 3, columns = 67, inner = 67;
  constexpr int channels = 3, voxels = 67;
  const size_t matrix_count = static_cast<size_t>(rows) * columns;
  const size_t swiglu_input_count = static_cast<size_t>(rows) * 2 * inner;
  const size_t swiglu_output_count = static_cast<size_t>(rows) * inner;
  const size_t latent_count = static_cast<size_t>(channels) * voxels;
  std::vector<float> x(matrix_count), y(matrix_count), residual_bias(columns),
      scale(columns), swiglu_input(swiglu_input_count), swiglu_bias(2 * inner),
      latent(latent_count), mean(channels), std_dev(channels);
  for (size_t i = 0; i < matrix_count; ++i) {
    x[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 16.0f;
    y[i] = static_cast<float>(static_cast<int>(i % 37) - 18) / 32.0f;
  }
  for (int i = 0; i < columns; ++i) {
    residual_bias[i] = static_cast<float>((i % 11) - 5) / 64.0f;
    scale[i] = 0.25f + static_cast<float>(i % 7) / 16.0f;
  }
  for (size_t i = 0; i < swiglu_input.size(); ++i)
    swiglu_input[i] = static_cast<float>(static_cast<int>(i % 101) - 50) / 16.0f;
  for (size_t i = 0; i < swiglu_bias.size(); ++i)
    swiglu_bias[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 32.0f;
  for (size_t i = 0; i < latent.size(); ++i)
    latent[i] = static_cast<float>(static_cast<int>(i % 47) - 23) / 16.0f;
  mean = {-0.25f, 0.0f, 0.375f};
  std_dev = {0.5f, -0.75f, 1.25f};

  // Each operation has an input-subnormal case, a normal cancellation whose
  // correctly rounded intermediate is subnormal, an output-underflow case,
  // signed zeros and exceptional values. The policy is asserted below rather
  // than merely comparing two backends with the same bug.
  x[0] = 0.0f; y[0] = from_bits(0x00800001u);
  residual_bias[0] = from_bits(0x80800000u); scale[0] = from_bits(0x7e800000u);
  x[1] = -0.0f; y[1] = from_bits(0x00800000u);
  residual_bias[1] = 0.0f; scale[1] = 0.5f;
  x[2] = from_bits(0x00000001u); y[2] = -0.0f;
  residual_bias[2] = 0.0f; scale[2] = 1.0f;
  x[3] = 1.0f; y[3] = from_bits(0x7fc12345u);
  residual_bias[3] = 0.0f; scale[3] = 1.0f;
  x[4] = -0.0f; y[4] = -0.0f; residual_bias[4] = -0.0f; scale[4] = 1.0f;
  x[5] = std::numeric_limits<float>::infinity(); y[5] = 1.0f;
  residual_bias[5] = 0.0f; scale[5] = 1.0f;
  x[6] = std::numeric_limits<float>::infinity();
  y[6] = -std::numeric_limits<float>::infinity();
  residual_bias[6] = 0.0f; scale[6] = 1.0f;

  swiglu_input[0] = from_bits(0x00800001u);
  swiglu_bias[0] = from_bits(0x80800000u);
  swiglu_input[inner] = from_bits(0x7e800000u);
  swiglu_bias[inner] = 0.0f;
  swiglu_input[1] = 1.0f; swiglu_bias[1] = 0.0f;
  swiglu_input[inner + 1] = from_bits(0x00800000u);
  swiglu_bias[inner + 1] = 0.0f;
  swiglu_input[2] = from_bits(0x7fc12345u); swiglu_bias[2] = 0.0f;
  swiglu_input[inner + 2] = 1.0f; swiglu_bias[inner + 2] = 0.0f;
  swiglu_input[3] = -0.0f; swiglu_bias[3] = -0.0f;
  swiglu_input[inner + 3] = 2.0f; swiglu_bias[inner + 3] = 0.0f;
  swiglu_input[4] = std::numeric_limits<float>::infinity();
  swiglu_input[inner + 4] = 1.0f;
  swiglu_input[5] = -std::numeric_limits<float>::infinity();
  swiglu_input[inner + 5] = 1.0f;
  swiglu_input[6] = std::numeric_limits<float>::infinity();
  swiglu_input[inner + 6] = 0.0f;
  swiglu_input[7] = -std::numeric_limits<float>::infinity();
  swiglu_input[inner + 7] = std::numeric_limits<float>::infinity();
  for (int index = 4; index <= 10; ++index) {
    swiglu_bias[index] = 0.0f;
    swiglu_bias[inner + index] = 0.0f;
  }
  swiglu_input[8] = std::nextafter(-87.0f,
                                   -std::numeric_limits<float>::infinity());
  swiglu_input[9] = -87.0f;
  swiglu_input[10] = std::nextafter(-87.0f,
                                    std::numeric_limits<float>::infinity());
  swiglu_input[inner + 8] = 1.0f;
  swiglu_input[inner + 9] = 1.0f;
  swiglu_input[inner + 10] = 1.0f;

  latent[0] = from_bits(0x00800000u); std_dev[0] = 0.5f; mean[0] = 0.0f;
  latent[voxels] = from_bits(0x00000001u); std_dev[1] = 1.0f; mean[1] = -0.0f;
  latent[5] = std::numeric_limits<float>::infinity();
  latent[6] = -std::numeric_limits<float>::infinity();
  latent[2 * voxels] = from_bits(0x7fc01234u);
  latent[2 * voxels + 1] = std::numeric_limits<float>::infinity();
  std_dev[2] = 0.0f; mean[2] = 1.0f;

  cuda::DeviceBuffer<float> cx(matrix_count), cy(matrix_count),
      crb(columns), cs(columns), csi(swiglu_input_count), csb(2 * inner),
      cso(swiglu_output_count), cl(latent_count), cm(channels), csd(channels),
      clo(latent_count);
  cx.copy_from_host(x.data(), x.size()); cy.copy_from_host(y.data(), y.size());
  crb.copy_from_host(residual_bias.data(), residual_bias.size());
  cs.copy_from_host(scale.data(), scale.size());
  csi.copy_from_host(swiglu_input.data(), swiglu_input.size());
  csb.copy_from_host(swiglu_bias.data(), swiglu_bias.size());
  cl.copy_from_host(latent.data(), latent.size()); cm.copy_from_host(mean.data(), mean.size());
  csd.copy_from_host(std_dev.data(), std_dev.size());
  cuda::launch_layerscale_residual(cx.get(), cy.get(), crb.get(), cs.get(),
                                   rows, columns, nullptr);
  cuda::launch_swiglu(csi.get(), csb.get(), cso.get(), rows, inner, nullptr);
  cuda::launch_latent_denorm(cl.get(), cm.get(), csd.get(), clo.get(),
                             channels, voxels, nullptr);

  // The legacy CUDA ABI permits nullable biases. Its null branch must remain
  // exactly equivalent to a present all-zero bias after the semantic rebase.
  cuda::DeviceBuffer<float> c_zero_residual_bias(columns),
      c_null_residual(matrix_count), c_zero_residual(matrix_count),
      c_nullable_y(matrix_count), c_zero_swiglu_bias(2 * inner),
      c_nullable_swiglu_input(swiglu_input_count),
      c_null_swiglu(swiglu_output_count), c_zero_swiglu(swiglu_output_count);
  std::vector<float> zero_residual_bias(columns, 0.0f),
      zero_swiglu_bias(2 * inner, 0.0f), nullable_x(matrix_count),
      nullable_y(matrix_count), nullable_swiglu_input(swiglu_input_count);
  for (size_t index = 0; index < matrix_count; ++index) {
    nullable_x[index] = 0.25f + static_cast<float>(index % 7) / 16.0f;
    nullable_y[index] = -0.5f + static_cast<float>(index % 11) / 32.0f;
  }
  for (size_t index = 0; index < swiglu_input_count; ++index)
    nullable_swiglu_input[index] =
        -1.0f + static_cast<float>(index % 23) / 16.0f;
  c_zero_residual_bias.copy_from_host(zero_residual_bias.data(), columns);
  c_zero_swiglu_bias.copy_from_host(zero_swiglu_bias.data(), 2 * inner);
  c_null_residual.copy_from_host(nullable_x.data(), nullable_x.size());
  c_zero_residual.copy_from_host(nullable_x.data(), nullable_x.size());
  c_nullable_y.copy_from_host(nullable_y.data(), nullable_y.size());
  c_nullable_swiglu_input.copy_from_host(nullable_swiglu_input.data(),
                                         nullable_swiglu_input.size());
  cuda::launch_layerscale_residual(c_null_residual.get(), c_nullable_y.get(), nullptr,
                                   cs.get(), rows, columns, nullptr);
  cuda::launch_layerscale_residual(c_zero_residual.get(), c_nullable_y.get(),
                                   c_zero_residual_bias.get(), cs.get(), rows,
                                   columns, nullptr);
  cuda::launch_swiglu(c_nullable_swiglu_input.get(), nullptr,
                      c_null_swiglu.get(), rows, inner, nullptr);
  cuda::launch_swiglu(c_nullable_swiglu_input.get(),
                      c_zero_swiglu_bias.get(), c_zero_swiglu.get(), rows,
                      inner, nullptr);

  const uint64_t matrix_shape[] = {rows, columns};
  const uint64_t column_shape = columns;
  const uint64_t swiglu_in_shape[] = {rows, 2 * inner};
  const uint64_t swiglu_out_shape[] = {rows, inner};
  const uint64_t swiglu_bias_shape = 2 * inner;
  const uint64_t latent_shape[] = {channels, voxels};
  const uint64_t channel_shape = channels;
  DeviceTensor vx = vk.allocate(TensorLayout::contiguous(matrix_shape, 2));
  DeviceTensor vy = vk.allocate(TensorLayout::contiguous(matrix_shape, 2));
  DeviceTensor vrb = vk.allocate(TensorLayout::contiguous(&column_shape, 1));
  DeviceTensor vs = vk.allocate(TensorLayout::contiguous(&column_shape, 1));
  DeviceTensor vsi = vk.allocate(TensorLayout::contiguous(swiglu_in_shape, 2));
  DeviceTensor vsb = vk.allocate(TensorLayout::contiguous(&swiglu_bias_shape, 1));
  DeviceTensor vso = vk.allocate(TensorLayout::contiguous(swiglu_out_shape, 2));
  DeviceTensor vl = vk.allocate(TensorLayout::contiguous(latent_shape, 2));
  DeviceTensor vm = vk.allocate(TensorLayout::contiguous(&channel_shape, 1));
  DeviceTensor vsd = vk.allocate(TensorLayout::contiguous(&channel_shape, 1));
  DeviceTensor vlo = vk.allocate(TensorLayout::contiguous(latent_shape, 2));
  vk.upload(vx, x.data(), x.size()); vk.upload(vy, y.data(), y.size());
  vk.upload(vrb, residual_bias.data(), residual_bias.size());
  vk.upload(vs, scale.data(), scale.size());
  vk.upload(vsi, swiglu_input.data(), swiglu_input.size());
  vk.upload(vsb, swiglu_bias.data(), swiglu_bias.size());
  vk.upload(vl, latent.data(), latent.size()); vk.upload(vm, mean.data(), mean.size());
  vk.upload(vsd, std_dev.data(), std_dev.size());
  TensorBatch batch = vk.begin_batch();
  batch.layer_scale_residual_f32(vx, vy, vrb, vs);
  batch.swiglu_bias_f32(vsi, vsb, vso);
  batch.latent_denorm_f32(vl, vm, vsd, vlo);
  batch.submit().wait();

  auto compare = [&](auto& cuda_buffer, DeviceTensor& vulkan_tensor, size_t count,
                     const char* label) {
    std::vector<float> cuda_host(count), vulkan_host(count);
    cuda_buffer.copy_to_host(cuda_host.data(), count);
    vk.download(vulkan_tensor, vulkan_host.data(), count);
    size_t mismatch = count;
    for (size_t i = 0; i < count; ++i) {
      if (std::memcmp(&cuda_host[i], &vulkan_host[i], sizeof(float)) != 0) {
        mismatch = i; break;
      }
    }
    uint32_t cb = 0, vb = 0;
    if (mismatch != count) {
      std::memcpy(&cb, &cuda_host[mismatch], 4);
      std::memcpy(&vb, &vulkan_host[mismatch], 4);
    }
    CHECK_MSG(mismatch == count, "%s mismatch at %zu: %08x != %08x",
              label, mismatch, cb, vb);
    return cuda_host;
  };
  const auto residual = compare(cx, vx, matrix_count, "VAE residual");
  const auto swiglu = compare(cso, vso, swiglu_output_count, "VAE SwiGLU");
  const auto denorm = compare(clo, vlo, latent_count, "VAE latent denorm");
  auto bits_of = [](float value) {
    uint32_t bits = 0; std::memcpy(&bits, &value, 4); return bits;
  };
  CHECK(bits_of(residual[0]) == 0x00000000u);
  CHECK((bits_of(residual[1]) & 0x7fffffffu) == 0u);
  CHECK((bits_of(residual[2]) & 0x7fffffffu) == 0u);
  CHECK(bits_of(residual[3]) == 0x7fc00000u);
  CHECK(bits_of(residual[4]) == 0x80000000u);
  CHECK(bits_of(residual[5]) == 0x7f800000u);
  CHECK(bits_of(residual[6]) == 0x7fc00000u);
  CHECK(bits_of(swiglu[0]) == 0x00000000u);
  CHECK((bits_of(swiglu[1]) & 0x7fffffffu) == 0u);
  CHECK(bits_of(swiglu[2]) == 0x7fc00000u);
  CHECK(bits_of(swiglu[3]) == 0x80000000u);
  CHECK(bits_of(swiglu[4]) == 0x7f800000u);
  CHECK(bits_of(swiglu[5]) == 0x80000000u);
  CHECK(bits_of(swiglu[6]) == 0x7fc00000u);
  CHECK(bits_of(swiglu[7]) == 0x7fc00000u);
  CHECK(bits_of(swiglu[8]) == 0x80000000u);
  CHECK(bits_of(swiglu[9]) == 0x80000000u);
  CHECK((bits_of(swiglu[10]) & 0x7fffffffu) != 0u);
  CHECK((bits_of(denorm[0]) & 0x7fffffffu) == 0u);
  CHECK(bits_of(denorm[5]) == 0x7f800000u);
  CHECK(bits_of(denorm[6]) == 0xff800000u);
  CHECK(bits_of(denorm[voxels]) == 0x00000000u);
  CHECK(bits_of(denorm[2 * voxels]) == 0x7fc00000u);
  CHECK(bits_of(denorm[2 * voxels + 1]) == 0x7fc00000u);

  std::vector<float> null_residual(matrix_count), zero_residual(matrix_count),
      null_swiglu(swiglu_output_count), zero_swiglu(swiglu_output_count);
  c_null_residual.copy_to_host(null_residual.data(), null_residual.size());
  c_zero_residual.copy_to_host(zero_residual.data(), zero_residual.size());
  c_null_swiglu.copy_to_host(null_swiglu.data(), null_swiglu.size());
  c_zero_swiglu.copy_to_host(zero_swiglu.data(), zero_swiglu.size());
  CHECK(std::memcmp(null_residual.data(), zero_residual.data(),
                    null_residual.size() * sizeof(float)) == 0);
  CHECK(std::memcmp(null_swiglu.data(), zero_swiglu.data(),
                    null_swiglu.size() * sizeof(float)) == 0);

  // Every rejection below happens before access tracking/command mutation, so
  // the same batch remains usable and proves transactional validation.
  const uint64_t wrong_output_shape[] = {rows, inner + 1};
  DeviceTensor wrong_output = vk.allocate(
      TensorLayout::contiguous(wrong_output_shape, 2));
  {
    TensorBatch valid_after_rejection = vk.begin_batch();
    bool alias_rejected = false, shape_rejected = false;
    try { valid_after_rejection.layer_scale_residual_f32(vx, vx, vrb, vs); }
    catch (const std::invalid_argument&) { alias_rejected = true; }
    try { valid_after_rejection.swiglu_bias_f32(vsi, vsb, wrong_output); }
    catch (const std::invalid_argument&) { shape_rejected = true; }
    CHECK(alias_rejected && shape_rejected);
    valid_after_rejection.latent_denorm_f32(vl, vm, vsd, vlo);
    valid_after_rejection.submit().wait();
  }

  auto record_mixed = [&] {
    TensorBatch mixed = vk.begin_batch();
    for (int operation = 0; operation < 32; ++operation) {
      switch (operation % 3) {
        case 0: mixed.layer_scale_residual_f32(vx, vy, vrb, vs); break;
        case 1: mixed.swiglu_bias_f32(vsi, vsb, vso); break;
        default: mixed.latent_denorm_f32(vl, vm, vsd, vlo); break;
      }
    }
    return mixed.submit();
  };
  Submission warm_first = record_mixed(), warm_second = record_mixed();
  warm_first.wait(); warm_second.wait();
  const uint64_t stable_reserved = vk.reserved_bytes();
  const uint64_t stable_descriptors = vk.descriptor_set_allocations();
  for (int repeat = 0; repeat < 6; ++repeat) {
    Submission first = record_mixed(), second = record_mixed();
    Submission third = record_mixed();
    CHECK(first.value() < second.value() && second.value() < third.value());
    first.wait(); second.wait(); third.wait();
    CHECK(vk.reserved_bytes() == stable_reserved);
    CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  }
  {
    TensorBatch too_many = vk.begin_batch();
    bool rejected = false;
    try {
      for (int operation = 0; operation < 33; ++operation)
        too_many.latent_denorm_f32(vl, vm, vsd, vlo);
    } catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected);
    bool poisoned_submit_rejected = false;
    try { (void)too_many.submit(); }
    catch (const std::logic_error&) { poisoned_submit_rejected = true; }
    CHECK(poisoned_submit_rejected);
  }

  // Submitted jobs retain all four resources even when every public wrapper
  // is dropped before completion, and release them after the token is done.
  const uint64_t used_before_drop = vk.pooled_used_bytes();
  Submission dropped;
  {
    DeviceTensor ti = vk.allocate(TensorLayout::contiguous(latent_shape, 2));
    DeviceTensor tm = vk.allocate(TensorLayout::contiguous(&channel_shape, 1));
    DeviceTensor ts = vk.allocate(TensorLayout::contiguous(&channel_shape, 1));
    DeviceTensor to = vk.allocate(TensorLayout::contiguous(latent_shape, 2));
    vk.upload(ti, latent.data(), latent.size());
    vk.upload(tm, mean.data(), mean.size());
    vk.upload(ts, std_dev.data(), std_dev.size());
    TensorBatch retained = vk.begin_batch();
    retained.latent_denorm_f32(ti, tm, ts, to);
    dropped = retained.submit();
  }
  dropped.wait();
  dropped = Submission{};
  Submission collected = record_mixed();
  collected.wait();
  CHECK(vk.pooled_used_bytes() == used_before_drop);
}

VIDFAB_TEST(cuda_vulkan_dit_exact_pointwise) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options; options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContextOptions context_options; context_options.max_batch_operators = 16;
  TensorContext vk(device, context_options);
  if (!vk.exact_vae_pointwise() || !vk.exact_fp32_vae_normalization()) return;

  constexpr uint32_t rows=5, dim=13, timesteps=2, modalities=3, params=6, rank=3;
  constexpr uint32_t mod_rows=timesteps*modalities, features=modalities*params*dim;
  constexpr uint32_t inner=11;
  std::vector<float> weight(size_t(features)*rank), bias(features), code(timesteps*rank);
  for(size_t i=0;i<weight.size();++i) weight[i]=float(int(i%17)-8)/64.0f;
  for(size_t i=0;i<bias.size();++i) bias[i]=float(int(i%11)-5)/32.0f;
  for(size_t i=0;i<code.size();++i) code[i]=float(int(i%7)-3)/8.0f;
  std::vector<uint16_t> residual(size_t(rows)*dim), branch(residual.size());
  std::vector<uint16_t> norm_input(residual.size()), norm_weight(dim);
  std::vector<int32_t> selectors{0,5,2,3,1};
  for(size_t i=0;i<residual.size();++i){
    residual[i]=f32_to_bf16(float(int(i%23)-11)/16.0f);
    branch[i]=f32_to_bf16(float(int(i%19)-9)/32.0f);
    norm_input[i]=f32_to_bf16(float(int(i%29)-14)/16.0f);
  }
  for(uint32_t i=0;i<dim;++i) norm_weight[i]=f32_to_bf16(0.75f+float(i%5)/16.0f);
  std::vector<uint16_t> fused(size_t(rows)*2*inner);
  for(size_t i=0;i<fused.size();++i) fused[i]=f32_to_bf16(float(int(i%31)-15)/8.0f);
  std::vector<uint16_t> text_residual(size_t(rows)*dim), text_branch(text_residual.size());
  std::vector<uint16_t> text_gate(size_t(rows)*inner), text_up(text_gate.size());
  for(size_t i=0;i<text_residual.size();++i){
    text_residual[i]=f32_to_bf16(float(int(i%37)-18)/16.0f);
    text_branch[i]=f32_to_bf16(float(int(i%29)-14)/32.0f);
  }
  for(size_t i=0;i<text_gate.size();++i){
    text_gate[i]=f32_to_bf16(float(int(i%31)-15)/8.0f);
    text_up[i]=f32_to_bf16(float(int(i%23)-11)/16.0f);
  }
  const std::array<uint16_t,8> exceptional{
      0x0001u,0x8001u,0x0000u,0x8000u,0x7f80u,0xff80u,0x7fc1u,0x7f7fu};
  for(size_t i=0;i<exceptional.size();++i){
    text_residual[i]=exceptional[i];
    text_branch[i]=exceptional[exceptional.size()-1-i];
    text_gate[i]=exceptional[i];
    text_up[i]=exceptional[exceptional.size()-1-i];
  }

  cuda::DeviceBuffer<float> cw(weight.size()), cb(bias.size()), cc(code.size()),
      cm(size_t(params)*mod_rows*dim);
  cuda::DeviceBuffer<uint16_t> cx(residual.size()), cbranch(branch.size()),
      cnorm(norm_input.size()), cnorm_w(norm_weight.size()), cnorm_out(norm_input.size()),
      cfused(fused.size()), cswiglu(size_t(rows)*inner);
  cuda::DeviceBuffer<uint16_t> ctext_x(text_residual.size()),
      ctext_branch(text_branch.size()),ctext_gate(text_gate.size()),
      ctext_up(text_up.size()),ctext_out(text_gate.size());
  cuda::DeviceBuffer<int32_t> cselectors(selectors.size());
  cw.copy_from_host(weight.data(),weight.size());cb.copy_from_host(bias.data(),bias.size());
  cc.copy_from_host(code.data(),code.size());cx.copy_from_host(residual.data(),residual.size());
  cbranch.copy_from_host(branch.data(),branch.size());cnorm.copy_from_host(norm_input.data(),norm_input.size());
  cnorm_w.copy_from_host(norm_weight.data(),norm_weight.size());
  cselectors.copy_from_host(selectors.data(),selectors.size());cfused.copy_from_host(fused.data(),fused.size());
  ctext_x.copy_from_host(text_residual.data(),text_residual.size());
  ctext_branch.copy_from_host(text_branch.data(),text_branch.size());
  ctext_gate.copy_from_host(text_gate.data(),text_gate.size());
  ctext_up.copy_from_host(text_up.data(),text_up.size());
  cuda::launch_adaln_expand(cw.get(),cb.get(),cc.get(),cm.get(),timesteps,modalities,params,dim,rank,nullptr);
  cuda::launch_add_gated(reinterpret_cast<__nv_bfloat16*>(cx.get()),
      reinterpret_cast<const __nv_bfloat16*>(cbranch.get()),
      cm.get()+size_t(2)*mod_rows*dim,cselectors.get(),rows,dim,nullptr);
  cuda::launch_rmsnorm_modulate(reinterpret_cast<const __nv_bfloat16*>(cnorm.get()),
      reinterpret_cast<const __nv_bfloat16*>(cnorm_w.get()),
      cm.get()+size_t(4)*mod_rows*dim,cm.get()+size_t(3)*mod_rows*dim,
      cselectors.get(),reinterpret_cast<__nv_bfloat16*>(cnorm_out.get()),rows,dim,1e-6f,nullptr);
  cuda::launch_swiglu_exact(reinterpret_cast<const __nv_bfloat16*>(cfused.get()),
      reinterpret_cast<__nv_bfloat16*>(cswiglu.get()),rows,inner,nullptr);
  text::launch_residual_add_exact(reinterpret_cast<__nv_bfloat16*>(ctext_x.get()),
      reinterpret_cast<const __nv_bfloat16*>(ctext_branch.get()),text_residual.size(),nullptr);
  text::launch_swiglu_split_exact(reinterpret_cast<const __nv_bfloat16*>(ctext_gate.get()),
      reinterpret_cast<const __nv_bfloat16*>(ctext_up.get()),
      reinterpret_cast<__nv_bfloat16*>(ctext_out.get()),text_gate.size(),nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());

  auto mat=[&](uint64_t a,uint64_t b){const uint64_t s[]={a,b};return TensorLayout::contiguous(s,2);};
  auto vec=[&](uint64_t n){return TensorLayout::contiguous(&n,1);};
  DeviceTensor vw=vk.allocate(mat(features,rank)),vb=vk.allocate(vec(features)),vc=vk.allocate(mat(timesteps,rank));
  const uint64_t logical_table=uint64_t(mod_rows)*dim;
  const uint64_t align_elements=std::max<uint64_t>(1,vk.storage_binding_alignment()/4);
  const uint64_t table_stride=(logical_table+align_elements-1)/align_elements*align_elements;
  DeviceTensor vm=vk.allocate(vec(params*table_stride));
  DeviceTensor vx=vk.allocate(mat(rows,dim),ScalarType::kBFloat16),vbranch=vk.allocate(mat(rows,dim),ScalarType::kBFloat16);
  DeviceTensor vn=vk.allocate(mat(rows,dim),ScalarType::kBFloat16),vnw=vk.allocate(vec(dim),ScalarType::kBFloat16),vno=vk.allocate(mat(rows,dim),ScalarType::kBFloat16);
  DeviceTensor va=vk.allocate(vec(rows),ScalarType::kInt32),vf=vk.allocate(mat(rows,2*inner),ScalarType::kBFloat16),vso=vk.allocate(mat(rows,inner),ScalarType::kBFloat16);
  DeviceTensor vtext_x=vk.allocate(mat(rows,dim),ScalarType::kBFloat16),
      vtext_branch=vk.allocate(mat(rows,dim),ScalarType::kBFloat16),
      vtext_gate=vk.allocate(mat(rows,inner),ScalarType::kBFloat16),
      vtext_up=vk.allocate(mat(rows,inner),ScalarType::kBFloat16),
      vtext_out=vk.allocate(mat(rows,inner),ScalarType::kBFloat16);
  vk.upload(vw,weight.data(),weight.size());vk.upload(vb,bias.data(),bias.size());vk.upload(vc,code.data(),code.size());
  vk.upload_bytes(vx,residual.data(),residual.size()*2);vk.upload_bytes(vbranch,branch.data(),branch.size()*2);
  vk.upload_bytes(vn,norm_input.data(),norm_input.size()*2);vk.upload_bytes(vnw,norm_weight.data(),norm_weight.size()*2);
  vk.upload_bytes(va,selectors.data(),selectors.size()*4);vk.upload_bytes(vf,fused.data(),fused.size()*2);
  vk.upload_bytes(vtext_x,text_residual.data(),text_residual.size()*2);
  vk.upload_bytes(vtext_branch,text_branch.data(),text_branch.size()*2);
  vk.upload_bytes(vtext_gate,text_gate.data(),text_gate.size()*2);
  vk.upload_bytes(vtext_up,text_up.data(),text_up.size()*2);
  TensorBatch batch=vk.begin_batch();batch.dit_expand_adaln(vw,vb,vc,vm,modalities,params,dim);
  batch.dit_add_gated_bf16_table(vx,vbranch,vm,mod_rows,2,va);
  batch.rms_norm_modulate_bf16_table(vn,vnw,vm,mod_rows,4,3,va,vno,1e-6f);
  batch.dit_swiglu_bf16(vf,vso);
  batch.text_add_residual_bf16(vtext_x,vtext_branch);
  batch.text_swiglu_split_bf16(vtext_gate,vtext_up,vtext_out);
  batch.submit().wait();
  std::vector<float> cuda_mod(size_t(params)*mod_rows*dim),vk_mod(cuda_mod.size()),vk_arena(params*table_stride);
  std::vector<uint16_t> cuda_x(residual.size()),vk_x(residual.size()),cuda_n(norm_input.size()),vk_n(norm_input.size()),cuda_s(size_t(rows)*inner),vk_s(cuda_s.size());
  std::vector<uint16_t> cuda_text_x(text_residual.size()),vk_text_x(text_residual.size()),
      cuda_text_out(text_gate.size()),vk_text_out(text_gate.size());
  cm.copy_to_host(cuda_mod.data(),cuda_mod.size());cx.copy_to_host(cuda_x.data(),cuda_x.size());
  cnorm_out.copy_to_host(cuda_n.data(),cuda_n.size());cswiglu.copy_to_host(cuda_s.data(),cuda_s.size());
  ctext_x.copy_to_host(cuda_text_x.data(),cuda_text_x.size());
  ctext_out.copy_to_host(cuda_text_out.data(),cuda_text_out.size());
  vk.download(vm,vk_arena.data(),vk_arena.size());
  for(uint32_t table=0;table<params;++table)
    std::memcpy(vk_mod.data()+table*logical_table,
                vk_arena.data()+table*table_stride,logical_table*4);
  vk.download_bytes(vx,vk_x.data(),vk_x.size()*2);
  vk.download_bytes(vno,vk_n.data(),vk_n.size()*2);vk.download_bytes(vso,vk_s.data(),vk_s.size()*2);
  vk.download_bytes(vtext_x,vk_text_x.data(),vk_text_x.size()*2);
  vk.download_bytes(vtext_out,vk_text_out.data(),vk_text_out.size()*2);
  CHECK(std::memcmp(cuda_mod.data(),vk_mod.data(),cuda_mod.size()*4)==0);
  CHECK(std::memcmp(cuda_x.data(),vk_x.data(),cuda_x.size()*2)==0);
  CHECK(std::memcmp(cuda_n.data(),vk_n.data(),cuda_n.size()*2)==0);
  CHECK(std::memcmp(cuda_s.data(),vk_s.data(),cuda_s.size()*2)==0);
  CHECK(std::memcmp(cuda_text_x.data(),vk_text_x.data(),cuda_text_x.size()*2)==0);
  CHECK(std::memcmp(cuda_text_out.data(),vk_text_out.data(),cuda_text_out.size()*2)==0);

  // The text layer reuses one dense BF16 slot for every compressed matrix.
  // Exercise I8 here as well as the existing real-NVFP4 streaming test so a
  // format-specific scale lookup cannot regress the shared cache.
  constexpr uint32_t stream_rows=3,stream_in=17,stream_out=13;
  std::vector<int8_t> i8(size_t(stream_out)*stream_in);
  std::vector<float> i8_scale(stream_out);
  std::vector<uint16_t> stream_input(size_t(stream_rows)*stream_in);
  for(size_t i=0;i<i8.size();++i)i8[i]=static_cast<int8_t>(int(i%17)-8);
  for(uint32_t i=0;i<stream_out;++i)i8_scale[i]=float(i+1)/64.0f;
  for(size_t i=0;i<stream_input.size();++i)
    stream_input[i]=f32_to_bf16(float(int(i%19)-9)/16.0f);
  LinearWeightUpload i8_upload;i8_upload.format=LinearWeightFormat::kInt8;
  i8_upload.out_features=stream_out;i8_upload.in_features=stream_in;
  i8_upload.data=i8.data();i8_upload.data_bytes=i8.size();
  i8_upload.weight_scale=i8_scale.data();i8_upload.weight_scale_count=i8_scale.size();
  LinearWeight i8_weight=LinearWeight::upload(vk,i8_upload);
  DeviceTensor stream_i=vk.allocate(mat(stream_rows,stream_in),ScalarType::kBFloat16);
  DeviceTensor stream_dense=vk.allocate(mat(stream_out,stream_in),ScalarType::kBFloat16);
  DeviceTensor stream_expected=vk.allocate(mat(stream_rows,stream_out),ScalarType::kBFloat16);
  DeviceTensor stream_actual=vk.allocate(mat(stream_rows,stream_out),ScalarType::kBFloat16);
  vk.upload_bytes(stream_i,stream_input.data(),stream_input.size()*2);
  DenseGemmPlan stream_plan=DenseGemmPlan::create(vk,{stream_rows,stream_out,stream_in,
      DenseGemmMode::kBFloat16,DenseGemmBias::kNone});
  {TensorBatch direct=vk.begin_batch();i8_weight.materialize_bf16(direct,stream_dense);
    stream_plan.record(direct,stream_i,stream_dense,stream_expected,stream_rows);
    direct.submit().wait();}
  StreamedNVFP4WeightCache stream_cache=StreamedNVFP4WeightCache::create(
      vk,uint64_t(stream_out)*stream_in);
  TensorContext foreign_vk(device);
  LinearWeight foreign_i8_weight=LinearWeight::upload(foreign_vk,i8_upload);
  DenseGemmPlan wrong_stream_plan=DenseGemmPlan::create(vk,
      {stream_rows,stream_out,stream_in+1,DenseGemmMode::kBFloat16,
       DenseGemmBias::kNone});
  {TensorBatch streamed=vk.begin_batch();
    PreparedNVFP4WeightView prepared=stream_cache.prepare(streamed,i8_weight,stream_plan);
    const uint32_t capacity=streamed.remaining_operator_capacity();
    bool foreign_rejected=false;
    try{(void)stream_cache.prepare(streamed,foreign_i8_weight,stream_plan);}
    catch(const std::invalid_argument&){foreign_rejected=true;}
    CHECK(foreign_rejected);
    CHECK(streamed.remaining_operator_capacity()==capacity);
    bool shape_rejected=false;
    try{(void)stream_cache.prepare(streamed,i8_weight,wrong_stream_plan);}
    catch(const std::invalid_argument&){shape_rejected=true;}
    CHECK(shape_rejected);
    CHECK(streamed.remaining_operator_capacity()==capacity);
    // Rejections preserve the exact generation/layout/access state.
    stream_plan.record(streamed,stream_i,prepared,stream_actual,stream_rows);
    streamed.submit().wait();}
  std::vector<uint16_t> stream_expected_bits(size_t(stream_rows)*stream_out),
      stream_actual_bits(stream_expected_bits.size());
  vk.download_bytes(stream_expected,stream_expected_bits.data(),stream_expected_bits.size()*2);
  vk.download_bytes(stream_actual,stream_actual_bits.data(),stream_actual_bits.size()*2);
  CHECK(stream_expected_bits==stream_actual_bits);
}

VIDFAB_TEST(vulkan_qwen_real_layer0_synthetic_activation) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  const std::filesystem::path checkpoint_path =
      "weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors";
  if (!std::filesystem::exists(checkpoint_path) || !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) return;
  const DeviceInfo& info = physical.front().info();
  if (!info.timeline_semaphore || !info.shader_int64) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  options.enable_shader_float16 = info.shader_float16;
  options.enable_storage_buffer_16bit = info.storage_buffer_16bit;
  options.enable_cooperative_matrix =
      info.cooperative_matrix_bf16_f32_16x16x16;
  Device device = physical.front().create_device(options);
  TensorContextOptions context_options;
  // S3 I8: q/k/v and gate/up share their identical ConvRot activation, saving
  // three operators, plus eleven requested diagnostic copies.
  context_options.max_batch_operators = 39;
  TensorContext vk(device, context_options);
  if (!vk.exact_causal_gqa_attention() ||
      !vk.exact_fp32_vae_normalization() || !vk.exact_vae_pointwise()) return;

  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
  QwenTextLayerConfig config;
  config.sequence = 3;
  const auto load_begin = std::chrono::steady_clock::now();
  ExactQwenTextLayerStage stage = ExactQwenTextLayerStage::create(vk, config);
  stage.load(checkpoint, 0);
  const double load_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - load_begin).count();
  ExactQwenTextLayerScratch scratch =
      ExactQwenTextLayerScratch::create(vk, config);
  CHECK(stage.loaded());
  CHECK(stage.format() == text::WeightFormat::kI8ConvRot);
  CHECK(scratch.dense_cache_bytes() == 250ull * 1024 * 1024);

  constexpr uint32_t rows = 3, hidden = 5120, q_heads = 64, kv_heads = 8,
                     head_dim = 128, ffn = 25600;
  std::vector<uint16_t> input(size_t(rows) * hidden);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = f32_to_bf16(float(int((i * 19) % 101) - 50) / 64.0f);
  std::vector<float> cosine, sine;
  text::build_rope_tables(rows, text::rope_inv_freq(head_dim, 5.0e6f),
                          cosine, sine);
  const uint64_t token_shape[] = {rows, hidden};
  const uint64_t rope_shape[] = {rows, head_dim};
  DeviceTensor tokens = vk.allocate(
      TensorLayout::contiguous(token_shape, 2), ScalarType::kBFloat16);
  DeviceTensor cos_tensor = vk.allocate(
      TensorLayout::contiguous(rope_shape, 2), ScalarType::kFloat32);
  DeviceTensor sin_tensor = vk.allocate(
      TensorLayout::contiguous(rope_shape, 2), ScalarType::kFloat32);
  vk.upload(cos_tensor, cosine.data(), cosine.size());
  vk.upload(sin_tensor, sine.data(), sine.size());

  auto mat = [&](uint64_t a, uint64_t b) {
    const uint64_t shape[] = {a, b}; return TensorLayout::contiguous(shape, 2);
  };
  auto three = [&](uint64_t a, uint64_t b, uint64_t c) {
    const uint64_t shape[] = {a, b, c}; return TensorLayout::contiguous(shape, 3);
  };
  DeviceTensor tap_norm = vk.allocate(mat(rows, hidden), ScalarType::kBFloat16);
  DeviceTensor tap_q = vk.allocate(three(rows, q_heads, head_dim), ScalarType::kBFloat16);
  DeviceTensor tap_k = vk.allocate(three(rows, kv_heads, head_dim), ScalarType::kBFloat16);
  DeviceTensor tap_v = vk.allocate(three(rows, kv_heads, head_dim), ScalarType::kBFloat16);
  DeviceTensor tap_attention = vk.allocate(
      three(rows, q_heads, head_dim), ScalarType::kBFloat16);
  DeviceTensor tap_attention_residual = vk.allocate(
      mat(rows, hidden), ScalarType::kBFloat16);
  DeviceTensor tap_post_norm = vk.allocate(
      mat(rows, hidden), ScalarType::kBFloat16);
  DeviceTensor tap_gate = vk.allocate(mat(rows, ffn), ScalarType::kBFloat16);
  DeviceTensor tap_up = vk.allocate(mat(rows, ffn), ScalarType::kBFloat16);
  DeviceTensor tap_activation = vk.allocate(
      mat(rows, ffn), ScalarType::kBFloat16);
  DeviceTensor tap_final = vk.allocate(mat(rows, hidden), ScalarType::kBFloat16);
  QwenTextLayerTaps taps{&tap_norm, &tap_q, &tap_k, &tap_v,
      &tap_attention, &tap_attention_residual, &tap_post_norm, &tap_gate,
      &tap_up, &tap_activation, &tap_final};
  CHECK(stage.required_operators(&taps) == 39);

  auto run = [&] {
    vk.upload_bytes(tokens, input.data(), input.size() * 2);
    const auto begin = std::chrono::steady_clock::now();
    TensorBatch batch = vk.begin_batch();
    {
      test::HostAllocationGuard allocation_guard;
      stage.record(batch, tokens, cos_tensor, sin_tensor, scratch, &taps);
    }
    CHECK(batch.remaining_operator_capacity() ==
          context_options.max_batch_operators - stage.required_operators(&taps));
    batch.submit().wait();
    const double elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    std::vector<uint16_t> result(input.size());
    vk.download_bytes(tokens, result.data(), result.size() * 2);
    std::vector<uint16_t> tapped(result.size());
    vk.download_bytes(tap_final, tapped.data(), tapped.size() * 2);
    CHECK(result == tapped);
    return std::pair<std::vector<uint16_t>, double>(std::move(result), elapsed);
  };

  // One-short capacity is rejected before the first stage operator. The
  // caller can still submit its preceding copy and the tokens remain intact.
  const uint64_t one_shape[] = {1};
  DeviceTensor dummy_a = vk.allocate(
      TensorLayout::contiguous(one_shape, 1), ScalarType::kBFloat16);
  DeviceTensor dummy_b = vk.allocate(
      TensorLayout::contiguous(one_shape, 1), ScalarType::kBFloat16);
  const uint16_t dummy = f32_to_bf16(1.0f);
  vk.upload_bytes(dummy_a, &dummy, sizeof(dummy));
  vk.upload_bytes(tokens, input.data(), input.size() * 2);
  TensorBatch short_batch = vk.begin_batch();
  short_batch.copy(dummy_a, dummy_b);
  bool short_rejected = false;
  try {
    stage.record(short_batch, tokens, cos_tensor, sin_tensor, scratch, &taps);
  } catch (const std::logic_error&) {
    short_rejected = true;
  }
  CHECK(short_rejected);
  CHECK(short_batch.remaining_operator_capacity() == 38);
  short_batch.submit().wait();
  std::vector<uint16_t> unchanged(input.size());
  vk.download_bytes(tokens, unchanged.data(), unchanged.size() * 2);
  CHECK(unchanged == input);

  // Establish the exact non-weight pool baseline with caller scratch,
  // activations and taps retained. Every unload below must return here.
  stage.unload();
  const uint64_t unloaded_pool_baseline = vk.pooled_used_bytes();
  const uint64_t unloaded_descriptor_baseline =
      vk.descriptor_set_allocations();
  stage.load(checkpoint, 0);

  const auto first = run();
  const uint64_t stable_descriptors = vk.descriptor_set_allocations();
  const auto repeat = run();
  CHECK(first.first == repeat.first);
  CHECK(vk.descriptor_set_allocations() == stable_descriptors);

  uint64_t digest = 1469598103934665603ull;
  for (uint16_t value : first.first) {
    digest ^= value & 0xffu; digest *= 1099511628211ull;
    digest ^= value >> 8u; digest *= 1099511628211ull;
  }
  CHECK(digest == 0x42a875f728d707f3ull);
  const uint64_t persistent = stage.persistent_bytes();
  const uint64_t scratch_bytes = scratch.reserved_bytes();
  CHECK(stage.peak_device_bytes(scratch) == persistent + scratch_bytes);
  CHECK(persistent < 500ull * 1024 * 1024);
  CHECK(scratch_bytes < 270ull * 1024 * 1024);

  // A malformed replacement fails during host validation, preserving active
  // weights, pool high-water and the executable output.
  const std::filesystem::path bad_path =
      std::filesystem::temp_directory_path() /
      ("vidfab_qwen_layer_bad_reload_" + std::to_string(
          std::chrono::high_resolution_clock::now().time_since_epoch().count()) +
       ".safetensors");
  write_safetensors(bad_path.string(),
      {{"unrelated", {1}, {0.0f}, DType::kF32}});
  const uint64_t before_bad_persistent = stage.persistent_bytes();
  const uint64_t before_bad_reserved = vk.reserved_bytes();
  bool bad_rejected = false;
  {
    SafeTensors bad;
    bad.open(bad_path.string());
    try { stage.load(bad, 0); }
    catch (const std::exception&) { bad_rejected = true; }
  }
  CHECK(bad_rejected);
  CHECK(stage.persistent_bytes() == before_bad_persistent);
  CHECK(vk.reserved_bytes() == before_bad_reserved);
  const auto after_bad = run();
  CHECK(after_bad.first == first.first);
  std::filesystem::remove(bad_path);

  // A valid full manifest whose final projection descriptor is corrupted is
  // rejected just as transactionally. The sparse fixture retains the real
  // archive header/offset contract without duplicating a 27 GiB checkpoint.
  const std::filesystem::path late_i8_path =
      make_sparse_qwen_metadata_corruption(
          checkpoint, text::WeightFormat::kI8ConvRot,
          "model.layers.0.mlp.down_proj.comfy_quant");
  const uint64_t before_late_i8_used = vk.pooled_used_bytes();
  const uint64_t before_late_i8_reserved = vk.reserved_bytes();
  const uint64_t before_late_i8_descriptors = vk.descriptor_set_allocations();
  bool late_i8_rejected = false;
  {
    SafeTensors bad;
    bad.open(late_i8_path.string());
    try { stage.load(bad, 0); }
    catch (const std::exception&) { late_i8_rejected = true; }
  }
  CHECK(late_i8_rejected);
  CHECK(vk.pooled_used_bytes() == before_late_i8_used);
  CHECK(vk.reserved_bytes() == before_late_i8_reserved);
  CHECK(vk.descriptor_set_allocations() == before_late_i8_descriptors);
  CHECK(run().first == first.first);
  std::filesystem::remove(late_i8_path);

  const uint64_t i8_high_water = vk.reserved_bytes();
  stage.unload();
  CHECK(!stage.loaded());
  CHECK(stage.persistent_bytes() == 0);
  CHECK(vk.pooled_used_bytes() == unloaded_pool_baseline);
  CHECK(vk.descriptor_set_allocations() >= unloaded_descriptor_baseline);
  stage.load(checkpoint, 0);
  const auto reloaded = run();
  CHECK(reloaded.first == first.first);
  CHECK(vk.reserved_bytes() == i8_high_water);

  // The same stage/scratch executes the shipped NVFP4+AWQ contract. Only the
  // two declared AWQ transforms record; all seven matrices still use the one
  // shared dense slot.
  const std::filesystem::path nvfp4_path =
      "weights/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors";
  SafeTensors nvfp4;
  nvfp4.open(nvfp4_path.string());
  stage.unload();
  CHECK(vk.pooled_used_bytes() == unloaded_pool_baseline);
  CHECK(vk.reserved_bytes() == i8_high_water);
  const auto nv_load_begin = std::chrono::steady_clock::now();
  stage.load(nvfp4, 0);
  const double nv_load_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - nv_load_begin).count();
  CHECK(stage.format() == text::WeightFormat::kNVFP4Awq);
  CHECK(stage.required_operators(&taps) == 37);
  const auto nv_first = run();
  const uint64_t nv_stable_descriptors = vk.descriptor_set_allocations();
  const auto nv_repeat = run();
  CHECK(nv_first.first == nv_repeat.first);
  CHECK(vk.descriptor_set_allocations() == nv_stable_descriptors);
  const std::filesystem::path late_nv_path =
      make_sparse_qwen_metadata_corruption(
          nvfp4, text::WeightFormat::kNVFP4Awq,
          "model.layers.0.mlp.down_proj.comfy_quant");
  const uint64_t before_late_nv_used = vk.pooled_used_bytes();
  const uint64_t before_late_nv_reserved = vk.reserved_bytes();
  const uint64_t before_late_nv_descriptors = vk.descriptor_set_allocations();
  bool late_nv_rejected = false;
  {
    SafeTensors bad;
    bad.open(late_nv_path.string());
    try { stage.load(bad, 0); }
    catch (const std::exception&) { late_nv_rejected = true; }
  }
  CHECK(late_nv_rejected);
  CHECK(vk.pooled_used_bytes() == before_late_nv_used);
  CHECK(vk.reserved_bytes() == before_late_nv_reserved);
  CHECK(vk.descriptor_set_allocations() == before_late_nv_descriptors);
  CHECK(run().first == nv_first.first);
  std::filesystem::remove(late_nv_path);
  uint64_t nv_digest = 1469598103934665603ull;
  for (uint16_t value : nv_first.first) {
    nv_digest ^= value & 0xffu; nv_digest *= 1099511628211ull;
    nv_digest ^= value >> 8u; nv_digest *= 1099511628211ull;
  }
  CHECK(nv_digest == 0x935104aeb9432d6aull);
  const uint64_t nv_persistent = stage.persistent_bytes();
  CHECK(nv_persistent < 270ull * 1024 * 1024);
  const uint64_t nv_high_water = vk.reserved_bytes();
  stage.unload();
  CHECK(vk.pooled_used_bytes() == unloaded_pool_baseline);
  CHECK(vk.reserved_bytes() == nv_high_water);
  stage.load(nvfp4, 0);
  CHECK(run().first == nv_first.first);
  CHECK(vk.reserved_bytes() == nv_high_water);
  CHECK(vk.descriptor_set_allocations() == nv_stable_descriptors);
  stage.unload();
  CHECK(stage.persistent_bytes() == 0);
  CHECK(vk.pooled_used_bytes() == unloaded_pool_baseline);
  CHECK(vk.reserved_bytes() == nv_high_water);
  std::printf(
      "  exact Vulkan Qwen layer0 S3: I8 load/first/repeat %.1f/%.1f/%.1f ms FNV64 %016llx persistent %.1f MiB; NVFP4 load/first/repeat %.1f/%.1f/%.1f ms FNV64 %016llx persistent %.1f MiB; shared scratch %.1f MiB, unloaded pool baseline/reserved HWM %.1f/%.1f MiB, descriptors %llu\n",
      load_ms, first.second, repeat.second,
      static_cast<unsigned long long>(digest), double(persistent) / 1048576.0,
      nv_load_ms, nv_first.second, nv_repeat.second,
      static_cast<unsigned long long>(nv_digest),
      double(nv_persistent) / 1048576.0,
      double(scratch_bytes) / 1048576.0,
      double(unloaded_pool_baseline) / 1048576.0,
      double(vk.reserved_bytes()) / 1048576.0,
      static_cast<unsigned long long>(stable_descriptors));
}

VIDFAB_TEST(cuda_vulkan_qwen_layer0_real_l132) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  constexpr uint32_t rows=132,hidden=5120,q_heads=64,kv_heads=8,
                     head_dim=128,ffn=25600;
  const std::filesystem::path source(VIDFAB_TEST_SOURCE_DIR);
  const std::filesystem::path checkpoint_path=source/
      "weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors";
  const std::filesystem::path tokenizer_path=source/
      "ref/text_encoder/tokenizer.json";
  int cuda_devices=0;
  if(!std::filesystem::exists(checkpoint_path)||
     !std::filesystem::exists(tokenizer_path)||
     cudaGetDeviceCount(&cuda_devices)!=cudaSuccess||cuda_devices==0||
     !Instance::available()) return;
  SafeTensors checkpoint;checkpoint.open(checkpoint_path.string());
  text::EncoderConfig encoder_config;
  encoder_config.format=text::detect_weight_format(checkpoint);
  CHECK(encoder_config.format==text::WeightFormat::kI8ConvRot);
  text::Tokenizer tokenizer;tokenizer.load(tokenizer_path.string());
  std::string prompt;
  const std::string sentence=
      "A cinematic tracking shot follows a copper airship over snowy forests, "
      "with warm sunrise reflections and natural motion. ";
  while(tokenizer.encode(prompt).size()<rows) prompt+=sentence;
  std::vector<int32_t> token_ids=tokenizer.encode(prompt);
  token_ids.resize(rows);
  std::vector<uint16_t> input;
  text::gather_embedding_rows(checkpoint.at("model.embed_tokens.weight"),
                              nullptr,token_ids,input);
  CHECK(input.size()==size_t(rows)*hidden);
  std::vector<float> cosine,sine;
  text::build_rope_tables(rows,text::rope_inv_freq(head_dim,5.0e6f),cosine,sine);

  // Upload the checkpoint-native packed layer once and invoke the reusable
  // canonical exact CUDA path. This is the authority, not shipped cuBLAS.
  const text::LayerLayout layer_layout=text::make_layer_layout(encoder_config);
  cuda::DeviceBuffer<uint8_t> cuda_layer(layer_layout.total_bytes);
  text::upload_layer_direct(checkpoint,encoder_config,0,layer_layout,
                            cuda_layer.get(),nullptr);
  const text::LayerGlobalScales globals=
      text::read_global_scales(checkpoint,encoder_config,0);
  const text::LayerWeights cuda_weights=text::layer_weights_from_blob(
      cuda_layer.get(),layer_layout,encoder_config,globals);
  text::LayerDims dims;dims.format=encoder_config.format;dims.num_tokens=rows;
  dims.hidden=hidden;dims.num_heads=q_heads;dims.num_kv_heads=kv_heads;
  dims.head_dim=head_dim;dims.intermediate=ffn;dims.rms_norm_eps=1.0e-6f;
  cuda::Workspace cuda_workspace;
  cuda_workspace.reserve(text::exact_layer_workspace_bytes(cuda_weights,dims));
  cuda::DeviceBuffer<uint16_t> cuda_tokens(input.size());
  cuda::DeviceBuffer<float> cuda_cosine(cosine.size()),cuda_sine(sine.size());
  cuda_tokens.copy_from_host(input.data(),input.size());
  cuda_cosine.copy_from_host(cosine.data(),cosine.size());
  cuda_sine.copy_from_host(sine.data(),sine.size());
  cuda::DeviceBuffer<uint16_t> c_norm(size_t(rows)*hidden),
      c_q(size_t(rows)*q_heads*head_dim),c_k(size_t(rows)*kv_heads*head_dim),
      c_v(size_t(rows)*kv_heads*head_dim),
      c_attention(size_t(rows)*q_heads*head_dim),
      c_attention_residual(size_t(rows)*hidden),c_post_norm(size_t(rows)*hidden),
      c_gate(size_t(rows)*ffn),c_up(size_t(rows)*ffn),
      c_activation(size_t(rows)*ffn),c_final(size_t(rows)*hidden);
  text::ExactLayerTaps cuda_taps{
      reinterpret_cast<__nv_bfloat16*>(c_norm.get()),
      reinterpret_cast<__nv_bfloat16*>(c_q.get()),
      reinterpret_cast<__nv_bfloat16*>(c_k.get()),
      reinterpret_cast<__nv_bfloat16*>(c_v.get()),
      reinterpret_cast<__nv_bfloat16*>(c_attention.get()),
      reinterpret_cast<__nv_bfloat16*>(c_attention_residual.get()),
      reinterpret_cast<__nv_bfloat16*>(c_post_norm.get()),
      reinterpret_cast<__nv_bfloat16*>(c_gate.get()),
      reinterpret_cast<__nv_bfloat16*>(c_up.get()),
      reinterpret_cast<__nv_bfloat16*>(c_activation.get()),
      reinterpret_cast<__nv_bfloat16*>(c_final.get())};
  const auto cuda_begin=std::chrono::steady_clock::now();
  text::encoder_layer_forward_exact(nullptr,cuda_weights,dims,
      cuda_cosine.get(),cuda_sine.get(),
      reinterpret_cast<__nv_bfloat16*>(cuda_tokens.get()),cuda_workspace,
      &cuda_taps);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const double cuda_ms=std::chrono::duration<double,std::milli>(
      std::chrono::steady_clock::now()-cuda_begin).count();

  Instance instance=Instance::create();const auto physical=instance.enumerate_devices();
  if(physical.empty())return;const DeviceInfo& info=physical.front().info();
  if(!info.timeline_semaphore||!info.shader_int64)return;
  DeviceOptions options;options.enable_timeline_semaphore=true;
  options.enable_shader_int64=true;options.enable_shader_float16=info.shader_float16;
  options.enable_storage_buffer_16bit=info.storage_buffer_16bit;
  options.enable_cooperative_matrix=info.cooperative_matrix_bf16_f32_16x16x16;
  Device device=physical.front().create_device(options);
  TensorContextOptions context_options;context_options.max_batch_operators=46;
  TensorContext vk(device,context_options);
  if(!vk.exact_causal_gqa_attention()||!vk.exact_fp32_vae_normalization()||
     !vk.exact_vae_pointwise())return;
  QwenTextLayerConfig config;config.sequence=rows;
  ExactQwenTextLayerStage stage=ExactQwenTextLayerStage::create(vk,config);
  const auto load_begin=std::chrono::steady_clock::now();stage.load(checkpoint,0);
  const double load_ms=std::chrono::duration<double,std::milli>(
      std::chrono::steady_clock::now()-load_begin).count();
  ExactQwenTextLayerScratch scratch=ExactQwenTextLayerScratch::create(vk,config);
  auto mat=[&](uint64_t a,uint64_t b){const uint64_t s[]={a,b};return TensorLayout::contiguous(s,2);};
  auto three=[&](uint64_t a,uint64_t b,uint64_t c){const uint64_t s[]={a,b,c};return TensorLayout::contiguous(s,3);};
  DeviceTensor v_tokens=vk.allocate(mat(rows,hidden),ScalarType::kBFloat16),
      v_cos=vk.allocate(mat(rows,head_dim)),v_sin=vk.allocate(mat(rows,head_dim)),
      v_norm=vk.allocate(mat(rows,hidden),ScalarType::kBFloat16),
      v_q=vk.allocate(three(rows,q_heads,head_dim),ScalarType::kBFloat16),
      v_k=vk.allocate(three(rows,kv_heads,head_dim),ScalarType::kBFloat16),
      v_v=vk.allocate(three(rows,kv_heads,head_dim),ScalarType::kBFloat16),
      v_attention=vk.allocate(three(rows,q_heads,head_dim),ScalarType::kBFloat16),
      v_attention_residual=vk.allocate(mat(rows,hidden),ScalarType::kBFloat16),
      v_post_norm=vk.allocate(mat(rows,hidden),ScalarType::kBFloat16),
      v_gate=vk.allocate(mat(rows,ffn),ScalarType::kBFloat16),
      v_up=vk.allocate(mat(rows,ffn),ScalarType::kBFloat16),
      v_activation=vk.allocate(mat(rows,ffn),ScalarType::kBFloat16),
      v_final=vk.allocate(mat(rows,hidden),ScalarType::kBFloat16);
  vk.upload_bytes(v_tokens,input.data(),input.size()*2);vk.upload(v_cos,cosine.data(),cosine.size());
  vk.upload(v_sin,sine.data(),sine.size());
  QwenTextLayerTaps vk_taps{&v_norm,&v_q,&v_k,&v_v,&v_attention,
      &v_attention_residual,&v_post_norm,&v_gate,&v_up,&v_activation,&v_final};
  CHECK(stage.required_operators(&vk_taps)==46);
  const auto vk_begin=std::chrono::steady_clock::now();TensorBatch batch=vk.begin_batch();
  stage.record(batch,v_tokens,v_cos,v_sin,scratch,&vk_taps);CHECK(batch.remaining_operator_capacity()==0);
  batch.submit().wait();const double vk_ms=std::chrono::duration<double,std::milli>(
      std::chrono::steady_clock::now()-vk_begin).count();

  auto compare=[&](const char* name,cuda::DeviceBuffer<uint16_t>& expected_buffer,
                   DeviceTensor& actual_tensor,size_t count){
    std::vector<uint16_t> expected(count),actual(count);
    expected_buffer.copy_to_host(expected.data(),count);
    vk.download_bytes(actual_tensor,actual.data(),count*2);
    size_t mismatch=count;for(size_t i=0;i<count;++i)if(expected[i]!=actual[i]){mismatch=i;break;}
    CHECK_MSG(mismatch==count,"Qwen L132 %s mismatch at %zu: CUDA %04x Vulkan %04x",
        name,mismatch,mismatch==count?0u:expected[mismatch],mismatch==count?0u:actual[mismatch]);
    uint64_t hash=1469598103934665603ull;for(uint16_t value:actual){
      hash^=value&0xffu;hash*=1099511628211ull;hash^=value>>8u;hash*=1099511628211ull;}
    return hash;
  };
  std::array<uint64_t,11> hashes{
      compare("input norm",c_norm,v_norm,size_t(rows)*hidden),
      compare("query",c_q,v_q,size_t(rows)*q_heads*head_dim),
      compare("key",c_k,v_k,size_t(rows)*kv_heads*head_dim),
      compare("value",c_v,v_v,size_t(rows)*kv_heads*head_dim),
      compare("attention",c_attention,v_attention,size_t(rows)*q_heads*head_dim),
      compare("attention residual",c_attention_residual,v_attention_residual,size_t(rows)*hidden),
      compare("post norm",c_post_norm,v_post_norm,size_t(rows)*hidden),
      compare("gate",c_gate,v_gate,size_t(rows)*ffn),
      compare("up",c_up,v_up,size_t(rows)*ffn),
      compare("activation",c_activation,v_activation,size_t(rows)*ffn),
      compare("final",c_final,v_final,size_t(rows)*hidden)};
  constexpr std::array<uint64_t,11> expected_hashes{
      0x6ca9b8c5e16917b5ull,0xf9bf6e554a1e84e4ull,
      0x9c2c264a60b4b8b1ull,0xec21312eea810e06ull,
      0x73901fb1cb7f2cfbull,0x5ab1cc9e26345fe2ull,
      0x9e072ab6646a2a11ull,0x74d47f5ed650356dull,
      0x8292d03af91a2025ull,0x48f99e7549238eceull,
      0xfb3966de636ac098ull};
  CHECK(hashes==expected_hashes);

  auto fnv_bytes=[](const void* data,size_t bytes,uint64_t hash=1469598103934665603ull){
    const auto* cursor=static_cast<const uint8_t*>(data);
    for(size_t i=0;i<bytes;++i){hash^=cursor[i];hash*=1099511628211ull;}
    return hash;
  };
  const uint64_t input_hash=fnv_bytes(input.data(),input.size()*sizeof(uint16_t));
  uint64_t rope_hash=fnv_bytes(cosine.data(),cosine.size()*sizeof(float));
  rope_hash=fnv_bytes(sine.data(),sine.size()*sizeof(float),rope_hash);
  CHECK(input_hash==0x617329501f3c87a1ull);
  CHECK(rope_hash==0x693c23a9886dd147ull);
  constexpr std::array<uint8_t,32> checkpoint_sha{
      0xbc,0x2c,0xed,0x0f,0xbe,0xa6,0x47,0x57,
      0xfa,0x9a,0xcd,0xdc,0xcf,0xc0,0xb3,0xf4,
      0x81,0x9d,0x1d,0xcf,0x1d,0xa6,0xc1,0x24,
      0xd6,0x90,0xd3,0x68,0xbe,0x28,0x39,0x23};
  constexpr std::array<uint8_t,32> tokenizer_sha{
      0xa5,0xd8,0x5b,0x6d,0xcc,0x53,0x5e,0x6b,
      0x93,0x11,0x5a,0x9e,0xf2,0x87,0xe6,0x13,
      0x2f,0xdb,0xf3,0x02,0x70,0xda,0x62,0x18,
      0x19,0x4b,0xa7,0x42,0x26,0x11,0x73,0xc7};
  text::QwenLayerCapture generated;
  generated.header.sequence=rows;generated.header.hidden=hidden;
  generated.header.query_heads=q_heads;generated.header.kv_heads=kv_heads;
  generated.header.head_dim=head_dim;generated.header.intermediate=ffn;
  generated.header.checkpoint_sha256=checkpoint_sha;
  generated.header.tokenizer_sha256=tokenizer_sha;
  generated.header.input_fnv64=input_hash;generated.header.rope_fnv64=rope_hash;
  generated.header.boundary_fnv64=hashes;generated.token_ids=token_ids;
  generated.input_bf16=input;generated.cosine=cosine;generated.sine=sine;
  const std::filesystem::path capture_path=
      std::filesystem::path(VIDFAB_TEST_SOURCE_DIR)/
      "tests/data/qwen_layer0_l132.vfqw";
  if(const char* write=std::getenv("VIDFAB_WRITE_QWEN_LAYER_CAPTURE");
     write&&std::strcmp(write,"1")==0){
    text::write_qwen_layer_capture(capture_path.string(),generated);
  }
  CHECK(std::filesystem::exists(capture_path));
#ifdef _WIN32
  std::ifstream capture_stream(capture_path,std::ios::binary);
  const std::vector<uint8_t> capture_bytes{
      std::istreambuf_iterator<char>(capture_stream),
      std::istreambuf_iterator<char>()};
  constexpr std::array<uint8_t,32> capture_sha{
      0xec,0x13,0xad,0x62,0xa7,0xe2,0x53,0xd5,
      0x88,0xbf,0xac,0x51,0x85,0x0b,0x92,0x48,
      0x7b,0x7c,0xb8,0x8b,0xa7,0x3e,0x78,0x69,
      0xa2,0xb0,0x2c,0xba,0x79,0x11,0x04,0xb3};
  CHECK(sha256_mapping(capture_bytes.data(),capture_bytes.size())==capture_sha);
#endif
  const text::QwenLayerCapture captured=
      text::read_qwen_layer_capture(capture_path.string());
  CHECK(captured.header.sequence==rows);
  CHECK(captured.header.hidden==hidden);
  CHECK(captured.header.query_heads==q_heads);
  CHECK(captured.header.kv_heads==kv_heads);
  CHECK(captured.header.head_dim==head_dim);
  CHECK(captured.header.intermediate==ffn);
  CHECK(captured.header.checkpoint_sha256==checkpoint_sha);
  CHECK(captured.header.tokenizer_sha256==tokenizer_sha);
  CHECK(captured.header.input_fnv64==input_hash);
  CHECK(captured.header.rope_fnv64==rope_hash);
  CHECK(captured.header.boundary_fnv64==expected_hashes);
  CHECK(captured.token_ids==token_ids);
  CHECK(captured.input_bf16==input);
  CHECK(captured.cosine==cosine);
  CHECK(captured.sine==sine);

  // Run the identical captured activation through the shipped NVFP4+AWQ
  // layer. L132 exercises the 128-row cooperative tile plus a four-row GEMM
  // tail, while o/down execute their checkpoint-declared AWQ pre-scales.
  const std::filesystem::path nv_path=source/
      "weights/text_encoder/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors";
  const uint64_t i8_persistent_bytes=stage.persistent_bytes();
  CHECK(std::filesystem::exists(nv_path));
  constexpr Sha256Digest nv_checkpoint_sha{
      0x33,0xe6,0x9e,0x3e,0xda,0xb8,0x46,0xd5,
      0x29,0x49,0xba,0xfd,0xb0,0x03,0x78,0xbd,
      0x3f,0x5a,0x93,0xf7,0x81,0x24,0xfc,0x83,
      0xd5,0xef,0x10,0x9d,0xc4,0xa1,0xfc,0xbb};
  CHECK(sha256_file(nv_path.string())==nv_checkpoint_sha);
  SafeTensors nv_checkpoint;nv_checkpoint.open(nv_path.string());
  text::EncoderConfig nv_config;
  nv_config.format=text::detect_weight_format(nv_checkpoint);
  CHECK(nv_config.format==text::WeightFormat::kNVFP4Awq);
  const text::LayerLayout nv_layout=text::make_layer_layout(nv_config);
  cuda::DeviceBuffer<uint8_t> nv_cuda_layer(nv_layout.total_bytes);
  text::upload_layer_direct(nv_checkpoint,nv_config,0,nv_layout,
                            nv_cuda_layer.get(),nullptr);
  const text::LayerGlobalScales nv_globals=
      text::read_global_scales(nv_checkpoint,nv_config,0);
  const text::LayerWeights nv_cuda_weights=text::layer_weights_from_blob(
      nv_cuda_layer.get(),nv_layout,nv_config,nv_globals);
  dims.format=nv_config.format;
  cuda_tokens.copy_from_host(input.data(),input.size());
  const auto nv_cuda_begin=std::chrono::steady_clock::now();
  text::encoder_layer_forward_exact(nullptr,nv_cuda_weights,dims,
      cuda_cosine.get(),cuda_sine.get(),
      reinterpret_cast<__nv_bfloat16*>(cuda_tokens.get()),cuda_workspace,
      &cuda_taps);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const double nv_cuda_ms=std::chrono::duration<double,std::milli>(
      std::chrono::steady_clock::now()-nv_cuda_begin).count();

  stage.load(nv_checkpoint,0);
  CHECK(stage.format()==text::WeightFormat::kNVFP4Awq);
  CHECK(stage.required_operators(&vk_taps)==44);
  vk.upload_bytes(v_tokens,input.data(),input.size()*sizeof(uint16_t));
  const auto nv_vk_begin=std::chrono::steady_clock::now();
  TensorBatch nv_batch=vk.begin_batch();
  {
    test::HostAllocationGuard allocation_guard;
    stage.record(nv_batch,v_tokens,v_cos,v_sin,scratch,&vk_taps);
  }
  CHECK(nv_batch.remaining_operator_capacity()==2);
  nv_batch.submit().wait();
  const double nv_vk_ms=std::chrono::duration<double,std::milli>(
      std::chrono::steady_clock::now()-nv_vk_begin).count();
  std::array<uint64_t,11> nv_hashes{
      compare("NV input norm",c_norm,v_norm,size_t(rows)*hidden),
      compare("NV query",c_q,v_q,size_t(rows)*q_heads*head_dim),
      compare("NV key",c_k,v_k,size_t(rows)*kv_heads*head_dim),
      compare("NV value",c_v,v_v,size_t(rows)*kv_heads*head_dim),
      compare("NV attention",c_attention,v_attention,size_t(rows)*q_heads*head_dim),
      compare("NV attention residual",c_attention_residual,v_attention_residual,size_t(rows)*hidden),
      compare("NV post norm",c_post_norm,v_post_norm,size_t(rows)*hidden),
      compare("NV gate",c_gate,v_gate,size_t(rows)*ffn),
      compare("NV up",c_up,v_up,size_t(rows)*ffn),
      compare("NV activation",c_activation,v_activation,size_t(rows)*ffn),
      compare("NV final",c_final,v_final,size_t(rows)*hidden)};
  constexpr std::array<uint64_t,11> expected_nv_hashes{
      0x395ca0928bc2fe96ull,0xbe792d7f50dffcd8ull,
      0xabf4d02a691c93feull,0xa7c505fccf740093ull,
      0x3c771911cbcbba19ull,0xafcdcae26c9edd26ull,
      0x503531dfaee1b283ull,0x241abe4e317283c4ull,
      0xb3f789dffb6d190cull,0xdd56acc48480841eull,
      0xa389ed1f9e8067e8ull};
  CHECK(nv_hashes==expected_nv_hashes);
  std::printf("  real Qwen layer0 L132 exact CUDA %.1f ms Vulkan %.1f ms load %.1f ms, persistent/scratch %.1f/%.1f MiB, boundary FNV64:",cuda_ms,vk_ms,load_ms,
      double(i8_persistent_bytes)/1048576.0,double(scratch.reserved_bytes())/1048576.0);
  for(uint64_t hash:hashes)std::printf(" %016llx",static_cast<unsigned long long>(hash));
  std::printf("; input/rope %016llx/%016llx\n",
      static_cast<unsigned long long>(input_hash),
      static_cast<unsigned long long>(rope_hash));
  std::printf("  real NVFP4+AWQ Qwen layer0 L132 exact CUDA %.1f ms Vulkan %.1f ms, boundary FNV64:",
      nv_cuda_ms,nv_vk_ms);
  for(uint64_t hash:nv_hashes)
    std::printf(" %016llx",static_cast<unsigned long long>(hash));
  std::printf("\n");
}

VIDFAB_TEST(cuda_vulkan_qwen_full50_real_l132) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  const std::filesystem::path source(VIDFAB_TEST_SOURCE_DIR);
  const std::filesystem::path checkpoint_path = source /
      "weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors";
  const std::filesystem::path capture_path = source /
      "tests/data/qwen_layer0_l132.vfqw";
  int cuda_devices = 0;
  if (!std::filesystem::exists(checkpoint_path) ||
      !std::filesystem::exists(capture_path) ||
      cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;

  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) return;
  const DeviceInfo& info = physical.front().info();
  if (!info.timeline_semaphore || !info.shader_int64 ||
      !info.shader_float16 || !info.storage_buffer_16bit ||
      !info.cooperative_matrix_bf16_f32_16x16x16) return;
  DeviceOptions device_options;
  device_options.enable_timeline_semaphore = true;
  device_options.enable_shader_int64 = true;
  device_options.enable_shader_float16 = true;
  device_options.enable_storage_buffer_16bit = true;
  device_options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(device_options);

  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
  const text::QwenLayerCapture capture =
      text::read_qwen_layer_capture(capture_path.string());
  CHECK(capture.header.sequence == 132);
  CHECK(capture.token_ids.size() == 132);

  text::PromptEmbedding cuda_output;
  text::EncoderTrace cuda_trace;
  double cuda_seconds = 0.0;
  {
    text::Encoder encoder;
    text::EncoderConfig config;
    config.residency = text::Residency::kStreaming;
    config.arithmetic = text::EncoderArithmetic::kExact;
    encoder.load(checkpoint, config);
    const auto begin = std::chrono::steady_clock::now();
    cuda_output = encoder.encode(capture.token_ids, &cuda_trace);
    cuda_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
    CHECK(encoder.format() == text::WeightFormat::kI8ConvRot);
    CHECK(encoder.residency() == text::Residency::kStreaming);
    encoder.unload();
  }

  TensorContextOptions context_options;
  // I8 L132 records 35 layer operators, one trace copy and the final widen.
  context_options.max_batch_operators = 37;
  TensorContext vk(device, context_options);
  const uint64_t unloaded_baseline = vk.pooled_used_bytes();
  ExactQwenTextEncoder encoder = ExactQwenTextEncoder::create(vk);
  encoder.load(checkpoint);
  text::EncoderTrace vk_trace;
  const auto vk_begin = std::chrono::steady_clock::now();
  const text::PromptEmbedding vk_output =
      encoder.encode(capture.token_ids, &vk_trace);
  const double vk_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - vk_begin).count();

  CHECK(cuda_output.num_tokens == vk_output.num_tokens);
  CHECK(cuda_output.hidden_size == vk_output.hidden_size);
  CHECK(cuda_output.modality_tags == vk_output.modality_tags);
  CHECK(cuda_output.data == vk_output.data);
  CHECK(cuda_trace.num_tokens == vk_trace.num_tokens);
  CHECK(cuda_trace.hidden_size == vk_trace.hidden_size);
  CHECK(cuda_trace.layer_residual_bf16 == vk_trace.layer_residual_bf16);
  CHECK(vk_trace.layer_residual_bf16.size() ==
        size_t(50) * 132 * 5120);

  auto fnv64 = [](const void* values, size_t bytes) {
    const auto* data = static_cast<const uint8_t*>(values);
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; ++i) {
      hash ^= data[i]; hash *= 1099511628211ull;
    }
    return hash;
  };
  std::array<uint64_t, 50> hashes{};
  const size_t layer_elements = size_t(132) * 5120;
  for (size_t layer = 0; layer < hashes.size(); ++layer) {
    hashes[layer] = fnv64(
        vk_trace.layer_residual_bf16.data() + layer * layer_elements,
        layer_elements * sizeof(uint16_t));
  }
  const uint64_t final_f32 =
      fnv64(vk_output.data.data(), vk_output.data.size() * sizeof(float));
  constexpr std::array<uint64_t, 50> expected_hashes{
      0xfb3966de636ac098ull,0x16b0e53a0265959dull,
      0x7e2377403c49f851ull,0xdacca747d7db7ccaull,
      0xb3de1cdb9248573cull,0xcb7ef7852ef4ce79ull,
      0xfe09924dff391f53ull,0x0ac372b850d058d8ull,
      0xe818f084c10d99d5ull,0xeed04e74ca44abcfull,
      0xe3e63048102b3e14ull,0x6e5647916a5d3586ull,
      0x1b9a96cd7cb703c2ull,0xe60df16cc8c0390dull,
      0xcf624f216f397d84ull,0x2e680a56b7030b0aull,
      0x8b8e97be1f24e88eull,0x61e2e50339318a1eull,
      0x410a6b1ab0a9c0e4ull,0xaf70d3523ce3a525ull,
      0xfb6b26f66fc1eb31ull,0x732c1c3ed4d7e6a6ull,
      0x1b81b2dd36a55f5aull,0x01f09dd599e38e09ull,
      0xc93c6edb4d56f691ull,0xcb9bd628a2587064ull,
      0x30c8bbdab894f8cfull,0xd0c82e71b41c4e74ull,
      0xef88fb29b4c38602ull,0x3b865ed94b23284eull,
      0x160d4d0750485e84ull,0xa3128750a0466a21ull,
      0xc1c6ba2884daa0e1ull,0xbf2a42d54c6d336eull,
      0x7ae8060855d02ab2ull,0x2fe95298685c12c3ull,
      0x17610f06aabb0ce5ull,0xd09a61dcf57388c9ull,
      0xc363bc14f4fabe3bull,0xc857cd797d07a823ull,
      0xb5ba41c7ec13df0bull,0x153186eecfcb34e2ull,
      0x89792cd842ae3215ull,0x03cbb8ab112884f9ull,
      0x0d9c0669be4b1818ull,0xc45536c76bfb5268ull,
      0x465a47fdc0f38a3bull,0xc6c70427de251f9dull,
      0x2082d9a03b0f2c88ull,0x141e4954a3b02693ull};
  CHECK(hashes == expected_hashes);
  CHECK(final_f32 == 0x579170f52abfc8dbull);
  const ExactQwenTextEncoderStats stats = encoder.stats();
  CHECK(stats.last_num_tokens == 132);
  CHECK(stats.max_layer_weight_bytes < 500ull * 1024 * 1024);
  CHECK(stats.peak_device_bytes < 900ull * 1024 * 1024);
  CHECK(stats.allocator_peak_used_bytes >= stats.allocator_baseline_bytes);
  CHECK(stats.descriptor_set_allocations <= 40);
  const uint64_t stable_reserved = stats.allocator_reserved_bytes;
  const uint64_t stable_descriptors = stats.descriptor_set_allocations;
  encoder.unload();
  const uint64_t warm_unloaded_baseline = vk.pooled_used_bytes();
  CHECK(warm_unloaded_baseline >= unloaded_baseline);
  CHECK(warm_unloaded_baseline <=
        unloaded_baseline + 2 * vk.staging_capacity_bytes());
  CHECK(vk.reserved_bytes() == stable_reserved);
  CHECK(vk.descriptor_set_allocations() == stable_descriptors);

  // A complete unload/reload preserves exact output and returns to the same
  // warmed context-only staging baseline. No descriptor or pool growth is
  // permitted on the second trajectory.
  encoder.load(checkpoint);
  text::EncoderTrace reload_trace;
  const text::PromptEmbedding reloaded =
      encoder.encode(capture.token_ids, &reload_trace);
  CHECK(reloaded.data == cuda_output.data);
  CHECK(reload_trace.layer_residual_bf16 == cuda_trace.layer_residual_bf16);
  const ExactQwenTextEncoderStats reload_stats = encoder.stats();
  CHECK(reload_stats.allocator_reserved_bytes >= stable_reserved);
  CHECK(reload_stats.allocator_reserved_bytes < 1536ull * 1024 * 1024);
  CHECK(reload_stats.peak_device_bytes < 900ull * 1024 * 1024);
  CHECK(reload_stats.descriptor_set_allocations == stable_descriptors);
  encoder.unload();
  CHECK(vk.pooled_used_bytes() == warm_unloaded_baseline);
  CHECK(vk.reserved_bytes() == reload_stats.allocator_reserved_bytes);
  CHECK(vk.descriptor_set_allocations() == stable_descriptors);

  // The first cold->warm transition may reserve an additional pool block due
  // to a different free-list order after complete shape destruction. A third
  // identical lifecycle must reuse that warmed high-water exactly.
  encoder.load(checkpoint);
  text::EncoderTrace third_trace;
  const text::PromptEmbedding third =
      encoder.encode(capture.token_ids, &third_trace);
  CHECK(third.data == cuda_output.data);
  CHECK(third_trace.layer_residual_bf16 == cuda_trace.layer_residual_bf16);
  CHECK(encoder.stats().allocator_reserved_bytes ==
        reload_stats.allocator_reserved_bytes);
  CHECK(encoder.stats().descriptor_set_allocations == stable_descriptors);
#ifdef _WIN32
  // Aggregate load validation reaches the actual last layer before changing
  // the active archive or allocating. A corrupt canonical descriptor must
  // leave the loaded I8 model immediately usable and every allocator metric
  // unchanged.
  const std::filesystem::path corrupt_i8_path =
      make_sparse_qwen_metadata_corruption(
          checkpoint, text::WeightFormat::kI8ConvRot,
          "model.layers.49.mlp.down_proj.comfy_quant");
  const uint64_t rollback_used = vk.pooled_used_bytes();
  const uint64_t rollback_reserved = vk.reserved_bytes();
  const uint64_t rollback_descriptors = vk.descriptor_set_allocations();
  bool corrupt_i8_rejected = false;
  {
    SafeTensors corrupt_i8;
    corrupt_i8.open(corrupt_i8_path.string());
    try { encoder.load(corrupt_i8); }
    catch (const std::runtime_error&) { corrupt_i8_rejected = true; }
  }
  CHECK(corrupt_i8_rejected && encoder.loaded());
  CHECK(vk.pooled_used_bytes() == rollback_used);
  CHECK(vk.reserved_bytes() == rollback_reserved);
  CHECK(vk.descriptor_set_allocations() == rollback_descriptors);
  text::EncoderTrace rollback_trace;
  const text::PromptEmbedding rollback_output =
      encoder.encode(capture.token_ids, &rollback_trace);
  CHECK(rollback_output.data == cuda_output.data);
  CHECK(rollback_trace.layer_residual_bf16 == cuda_trace.layer_residual_bf16);
  std::filesystem::remove(corrupt_i8_path);
#endif
  encoder.unload();
  CHECK(vk.pooled_used_bytes() == warm_unloaded_baseline);
  CHECK(vk.reserved_bytes() == reload_stats.allocator_reserved_bytes);

  // Exact capacity is preflighted before activation allocation/upload. L132
  // I8 with a boundary trace requires 35+copy+final-widen = 37 operators.
  TensorContextOptions short_options;
  short_options.max_batch_operators = 36;
  TensorContext short_vk(device, short_options);
  const uint64_t short_used = short_vk.pooled_used_bytes();
  const uint64_t short_reserved = short_vk.reserved_bytes();
  ExactQwenTextEncoder short_encoder =
      ExactQwenTextEncoder::create(short_vk);
  short_encoder.load(checkpoint);
  text::EncoderTrace rejected_trace;
  bool short_rejected = false;
  try { (void)short_encoder.encode(capture.token_ids, &rejected_trace); }
  catch (const std::logic_error&) { short_rejected = true; }
  CHECK(short_rejected);
  CHECK(rejected_trace.layer_residual_bf16.empty());
  CHECK(short_vk.pooled_used_bytes() == short_used);
  CHECK(short_vk.reserved_bytes() == short_reserved);
  CHECK(short_vk.descriptor_set_allocations() == 0);

  std::printf(
      "  real exact Qwen full50 L132 CUDA/Vulkan %.2f/%.2f s final FNV64 %016llx, peak/used/reserved %.1f/%.1f/%.1f MiB descriptors %llu, boundaries:",
      cuda_seconds, vk_seconds,
      static_cast<unsigned long long>(final_f32),
      double(stats.peak_device_bytes) / 1048576.0,
      double(stats.allocator_used_bytes) / 1048576.0,
      double(stats.allocator_reserved_bytes) / 1048576.0,
      static_cast<unsigned long long>(stats.descriptor_set_allocations));
  for (uint64_t hash : hashes)
    std::printf(" %016llx", static_cast<unsigned long long>(hash));
  std::printf("\n");
}

VIDFAB_TEST(cuda_vulkan_dit_real_block0_replay) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  const std::filesystem::path path =
      "weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors";
  if (!std::filesystem::exists(path) || !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) return;
  const DeviceInfo& info = physical.front().info();
  if (!info.timeline_semaphore || !info.shader_int64 ||
      !info.shader_float16 || !info.storage_buffer_16bit ||
      !info.cooperative_matrix_bf16_f32_16x16x16) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  options.enable_shader_float16 = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(options);
  TensorContextOptions context_options;
  context_options.max_batch_operators = 64;
  TensorContext vk(device, context_options);
  if (!vk.exact_h3_attention() || !vk.exact_vae_pointwise() ||
      !vk.exact_fp32_vae_normalization()) return;

  uint32_t sequence = 65;
  if (const char* requested = std::getenv("VIDFAB_DIT_BLOCK_SEQUENCE")) {
    const unsigned long parsed = std::strtoul(requested, nullptr, 10);
    if (parsed == 0 || parsed > UINT32_MAX)
      throw std::invalid_argument("VIDFAB_DIT_BLOCK_SEQUENCE is invalid");
    sequence = static_cast<uint32_t>(parsed);
  }
  H3BlockConfig config;
  config.sequence = sequence;
  SafeTensors checkpoint;
  checkpoint.open(path.string());
  const auto load_begin = std::chrono::steady_clock::now();
  ExactH3BlockStage stage = ExactH3BlockStage::create(vk, config);
  stage.load(checkpoint, 0);
  const double load_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - load_begin).count();
  ExactH3BlockScratch scratch = ExactH3BlockScratch::create(vk, config);

  std::vector<uint16_t> token_bits(size_t(sequence) * config.hidden);
  for (size_t i = 0; i < token_bits.size(); ++i)
    token_bits[i] = f32_to_bf16(float(int(i % 61) - 30) / 64.0f);
  std::vector<int32_t> selectors(sequence);
  for (uint32_t i = 0; i < sequence; ++i)
    selectors[i] = static_cast<int32_t>(i % config.modalities);
  std::vector<float> code(size_t(config.timesteps) * config.adaln_rank);
  for (size_t i = 0; i < code.size(); ++i)
    code[i] = float(int(i % 7) - 3) / 16.0f;
  std::vector<float> cosine(size_t(sequence) * 96, 1.0f);
  std::vector<float> sine(size_t(sequence) * 96, 0.0f);
  const uint64_t token_shape[] = {sequence, config.hidden};
  const uint64_t selector_shape[] = {sequence};
  const uint64_t code_shape[] = {config.timesteps, config.adaln_rank};
  const uint64_t rope_shape[] = {sequence, 96};
  DeviceTensor tokens = vk.allocate(
      TensorLayout::contiguous(token_shape, 2), ScalarType::kBFloat16);
  DeviceTensor selector_tensor = vk.allocate(
      TensorLayout::contiguous(selector_shape, 1), ScalarType::kInt32);
  DeviceTensor code_tensor = vk.allocate(
      TensorLayout::contiguous(code_shape, 2));
  DeviceTensor cosine_tensor = vk.allocate(
      TensorLayout::contiguous(rope_shape, 2));
  DeviceTensor sine_tensor = vk.allocate(
      TensorLayout::contiguous(rope_shape, 2));
  vk.upload_bytes(tokens, token_bits.data(), token_bits.size() * 2);
  vk.upload_bytes(selector_tensor, selectors.data(), selectors.size() * 4);
  vk.upload(code_tensor, code.data(), code.size());
  vk.upload(cosine_tensor, cosine.data(), cosine.size());
  vk.upload(sine_tensor, sine.data(), sine.size());

  auto run = [&] {
    const auto begin = std::chrono::steady_clock::now();
    TensorBatch batch = vk.begin_batch();
    stage.record(batch, tokens, selector_tensor, code_tensor, cosine_tensor,
                 sine_tensor, scratch);
    batch.submit().wait();
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
  };
  const double first_ms = run();
  std::vector<uint16_t> first(token_bits.size()), repeated(token_bits.size());
  vk.download_bytes(tokens, first.data(), first.size() * 2);
  vk.upload_bytes(tokens, token_bits.data(), token_bits.size() * 2);
  const uint64_t descriptors = vk.descriptor_set_allocations();
  const uint64_t reserved = vk.reserved_bytes();
  const double repeat_ms = run();
  vk.download_bytes(tokens, repeated.data(), repeated.size() * 2);
  CHECK(first == repeated);
  CHECK(vk.descriptor_set_allocations() == descriptors);
  CHECK(vk.reserved_bytes() == reserved);

  if (!cuda::deterministic_h3_attention_available()) return;
  const uint32_t hidden = config.hidden;
  const uint32_t inner = config.heads * config.head_dim;
  const uint32_t ffn = config.ffn;
  const uint32_t modulation_rows = config.timesteps * config.modalities;
  auto bf16_host = [&](const std::string& name, uint32_t count) {
    std::vector<float> wide = to_f32(checkpoint.at(name));
    CHECK(wide.size() == count);
    std::vector<uint16_t> bits(count);
    for (uint32_t i = 0; i < count; ++i) bits[i] = f32_to_bf16(wide[i]);
    return bits;
  };
  const std::vector<uint16_t> norm1 = bf16_host("blocks.0.norm1.weight", hidden);
  const std::vector<uint16_t> norm2 = bf16_host("blocks.0.norm2.weight", hidden);
  const std::vector<uint16_t> q_norm = bf16_host("blocks.0.attn.q_norm.weight", config.head_dim);
  const std::vector<uint16_t> k_norm = bf16_host("blocks.0.attn.k_norm.weight", config.head_dim);
  std::vector<float> adaln_w = to_f32(checkpoint.at("blocks.0.adaln_proj.linear.weight"));
  std::vector<float> adaln_b = to_f32(checkpoint.at("blocks.0.adaln_proj.linear.bias"));
  const size_t largest_weight = std::max({
      size_t(inner) * hidden, size_t(hidden) * inner,
      size_t(2) * ffn * hidden, size_t(hidden) * ffn});
  const uint32_t largest_input = std::max({hidden, inner, ffn});
  cuda::DeviceBuffer<uint8_t> cuda_stored(largest_weight / 2),
      cuda_scale(largest_weight / 16);
  cuda::DeviceBuffer<uint16_t> cuda_dense(largest_weight),
      cuda_pre(largest_input), cuda_transform(size_t(sequence) * largest_input);
  cuda::DeviceBuffer<uint16_t> cuda_tokens(token_bits.size()),
      cuda_normed(size_t(sequence) * hidden), cuda_q(size_t(sequence) * inner),
      cuda_k(size_t(sequence) * inner), cuda_v(size_t(sequence) * inner),
      cuda_attention(size_t(sequence) * inner), cuda_branch(size_t(sequence) * hidden),
      cuda_fused(size_t(sequence) * 2 * ffn),
      cuda_activation(size_t(sequence) * ffn), cuda_norm1(hidden),
      cuda_norm2(hidden), cuda_q_norm(config.head_dim), cuda_k_norm(config.head_dim);
  cuda::DeviceBuffer<float> cuda_adaln_w(adaln_w.size()),
      cuda_adaln_b(adaln_b.size()), cuda_code(code.size()),
      cuda_modulation(size_t(6) * modulation_rows * hidden),
      cuda_cosine(cosine.size()), cuda_sine(sine.size());
  cuda::DeviceBuffer<int32_t> cuda_selectors(selectors.size());
  cuda_tokens.copy_from_host(token_bits.data(), token_bits.size());
  cuda_norm1.copy_from_host(norm1.data(), norm1.size());
  cuda_norm2.copy_from_host(norm2.data(), norm2.size());
  cuda_q_norm.copy_from_host(q_norm.data(), q_norm.size());
  cuda_k_norm.copy_from_host(k_norm.data(), k_norm.size());
  cuda_adaln_w.copy_from_host(adaln_w.data(), adaln_w.size());
  cuda_adaln_b.copy_from_host(adaln_b.data(), adaln_b.size());
  cuda_code.copy_from_host(code.data(), code.size());
  cuda_cosine.copy_from_host(cosine.data(), cosine.size());
  cuda_sine.copy_from_host(sine.data(), sine.size());
  cuda_selectors.copy_from_host(selectors.data(), selectors.size());

  auto cuda_projection = [&](const std::string& name, uint32_t out,
                             uint32_t in, uint32_t source_out,
                             uint32_t row_offset, const uint16_t* input,
                             uint16_t* output) {
    const TensorView& weight = checkpoint.at(name + ".weight");
    const TensorView& scale = checkpoint.at(name + ".weight_scale");
    CHECK(weight.dtype == DType::kU8 && scale.dtype == DType::kF8E4M3);
    CHECK(weight.shape == std::vector<int64_t>({source_out, in / 2}));
    CHECK(scale.shape == std::vector<int64_t>({source_out, in / 16}));
    const size_t elements = size_t(out) * in;
    const size_t element_offset = size_t(row_offset) * in;
    cuda_stored.copy_from_host(
        static_cast<const uint8_t*>(weight.data) + element_offset / 2,
        elements / 2);
    cuda_scale.copy_from_host(
        static_cast<const uint8_t*>(scale.data) + element_offset / 16,
        elements / 16);
    const std::vector<float> global_values =
        to_f32(checkpoint.at(name + ".weight_scale_2"));
    CHECK(global_values.size() == 1);
    const uint16_t* source = input;
    if (const TensorView* pre = checkpoint.find(name + ".pre_quant_scale")) {
      std::vector<float> wide = to_f32(*pre);
      CHECK(wide.size() == in);
      std::vector<uint16_t> bits(in);
      for (uint32_t i = 0; i < in; ++i) bits[i] = f32_to_bf16(wide[i]);
      cuda_pre.copy_from_host(bits.data(), bits.size());
      cuda::launch_pre_quant_scale(
          reinterpret_cast<const __nv_bfloat16*>(input),
          reinterpret_cast<const __nv_bfloat16*>(cuda_pre.get()),
          reinterpret_cast<__nv_bfloat16*>(cuda_transform.get()),
          sequence, in, nullptr);
      source = cuda_transform.get();
    }
    cuda::launch_dequant_nvfp4(
        cuda_stored.get(), cuda_scale.get(), global_values.front(),
        reinterpret_cast<__nv_bfloat16*>(cuda_dense.get()), out, in, nullptr);
    const uint32_t tiled_rows = sequence / 64 * 64;
    if (tiled_rows != 0) {
      cuda::launch_deterministic_bf16_gemm_nt(
          reinterpret_cast<const __nv_bfloat16*>(source),
          reinterpret_cast<const __nv_bfloat16*>(cuda_dense.get()), nullptr,
          reinterpret_cast<__nv_bfloat16*>(output), tiled_rows, out, in,
          DenseGemmBias::kNone);
    }
    if (tiled_rows != sequence) {
      cuda::launch_deterministic_scalar_gemm_nt(
          source, cuda_dense.get(), nullptr, output,
          sequence - tiled_rows, out, in, DenseGemmMode::kBFloat16,
          DenseGemmBias::kNone, tiled_rows, tiled_rows);
    }
  };
  const auto cuda_begin = std::chrono::steady_clock::now();
  cuda::launch_adaln_expand(
      cuda_adaln_w.get(), cuda_adaln_b.get(), cuda_code.get(),
      cuda_modulation.get(), config.timesteps, config.modalities, 6, hidden,
      config.adaln_rank, nullptr);
  const size_t table = size_t(modulation_rows) * hidden;
  cuda::launch_rmsnorm_modulate(
      reinterpret_cast<const __nv_bfloat16*>(cuda_tokens.get()),
      reinterpret_cast<const __nv_bfloat16*>(cuda_norm1.get()),
      cuda_modulation.get() + table, cuda_modulation.get(),
      cuda_selectors.get(),
      reinterpret_cast<__nv_bfloat16*>(cuda_normed.get()), sequence, hidden,
      config.epsilon, nullptr);
  const std::string qkv = "blocks.0.attn.qkv_proj";
  cuda_projection(qkv, inner, hidden, 3 * inner, 0, cuda_normed.get(), cuda_q.get());
  cuda_projection(qkv, inner, hidden, 3 * inner, inner, cuda_normed.get(), cuda_k.get());
  cuda_projection(qkv, inner, hidden, 3 * inner, 2 * inner, cuda_normed.get(), cuda_v.get());
  cuda::launch_head_rmsnorm(
      reinterpret_cast<__nv_bfloat16*>(cuda_q.get()),
      reinterpret_cast<const __nv_bfloat16*>(cuda_q_norm.get()), sequence,
      config.heads, config.head_dim, config.epsilon, nullptr);
  cuda::launch_head_rmsnorm(
      reinterpret_cast<__nv_bfloat16*>(cuda_k.get()),
      reinterpret_cast<const __nv_bfloat16*>(cuda_k_norm.get()), sequence,
      config.heads, config.head_dim, config.epsilon, nullptr);
  cuda::launch_rope_h3(
      reinterpret_cast<__nv_bfloat16*>(cuda_q.get()), cuda_cosine.get(),
      cuda_sine.get(), sequence, config.heads, config.head_dim, nullptr);
  cuda::launch_rope_h3(
      reinterpret_cast<__nv_bfloat16*>(cuda_k.get()), cuda_cosine.get(),
      cuda_sine.get(), sequence, config.heads, config.head_dim, nullptr);
  cuda::launch_deterministic_h3_attention(
      nullptr, reinterpret_cast<const __nv_bfloat16*>(cuda_q.get()),
      reinterpret_cast<const __nv_bfloat16*>(cuda_k.get()),
      reinterpret_cast<const __nv_bfloat16*>(cuda_v.get()),
      reinterpret_cast<__nv_bfloat16*>(cuda_attention.get()), nullptr,
      sequence, config.heads, config.head_dim,
      exact_attention_scale(config.head_dim));
  cuda_projection("blocks.0.attn.out_proj", hidden, inner, hidden, 0,
                  cuda_attention.get(), cuda_branch.get());
  cuda::launch_add_gated(
      reinterpret_cast<__nv_bfloat16*>(cuda_tokens.get()),
      reinterpret_cast<const __nv_bfloat16*>(cuda_branch.get()),
      cuda_modulation.get() + 2 * table, cuda_selectors.get(), sequence,
      hidden, nullptr);
  cuda::launch_rmsnorm_modulate(
      reinterpret_cast<const __nv_bfloat16*>(cuda_tokens.get()),
      reinterpret_cast<const __nv_bfloat16*>(cuda_norm2.get()),
      cuda_modulation.get() + 4 * table,
      cuda_modulation.get() + 3 * table, cuda_selectors.get(),
      reinterpret_cast<__nv_bfloat16*>(cuda_normed.get()), sequence, hidden,
      config.epsilon, nullptr);
  cuda_projection("blocks.0.mlp.fc1", 2 * ffn, hidden, 2 * ffn, 0,
                  cuda_normed.get(), cuda_fused.get());
  cuda::launch_swiglu_exact(
      reinterpret_cast<const __nv_bfloat16*>(cuda_fused.get()),
      reinterpret_cast<__nv_bfloat16*>(cuda_activation.get()), sequence, ffn,
      nullptr);
  cuda_projection("blocks.0.mlp.fc2", hidden, ffn, hidden, 0,
                  cuda_activation.get(), cuda_branch.get());
  cuda::launch_add_gated(
      reinterpret_cast<__nv_bfloat16*>(cuda_tokens.get()),
      reinterpret_cast<const __nv_bfloat16*>(cuda_branch.get()),
      cuda_modulation.get() + 5 * table, cuda_selectors.get(), sequence,
      hidden, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const double cuda_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - cuda_begin).count();
  std::vector<uint16_t> cuda_output(token_bits.size());
  cuda_tokens.copy_to_host(cuda_output.data(), cuda_output.size());
  size_t mismatch = first.size();
  for (size_t i = 0; i < first.size(); ++i) {
    if (first[i] != cuda_output[i]) { mismatch = i; break; }
  }
  CHECK_MSG(mismatch == first.size(),
            "real H3 block0 CUDA/Vulkan mismatch at %zu: %04x != %04x",
            mismatch, mismatch == first.size() ? 0 : cuda_output[mismatch],
            mismatch == first.size() ? 0 : first[mismatch]);
  uint64_t digest = 1469598103934665603ull;
  for (uint16_t bits : first) {
    digest ^= bits & 0xffu; digest *= 1099511628211ull;
    digest ^= bits >> 8; digest *= 1099511628211ull;
  }
  if (sequence == 65) CHECK(digest == 0x191929c14480e873ull);
  const uint64_t persistent = stage.persistent_bytes();
  bool failed_reload = false;
  try { stage.load(checkpoint, 50); }
  catch (const std::exception&) { failed_reload = true; }
  CHECK(failed_reload && stage.loaded());
  CHECK(stage.persistent_bytes() == persistent);
  vk.upload_bytes(tokens, token_bits.data(), token_bits.size() * 2);
  (void)run();
  std::vector<uint16_t> after_failure(first.size());
  vk.download_bytes(tokens, after_failure.data(), after_failure.size() * 2);
  CHECK(after_failure == first);
  stage.unload();
  CHECK(!stage.loaded() && stage.persistent_bytes() == 0);
  stage.load(checkpoint, 0);
  CHECK(stage.loaded() && stage.persistent_bytes() == persistent);
  vk.upload_bytes(tokens, token_bits.data(), token_bits.size() * 2);
  (void)run();
  std::vector<uint16_t> after_reload(first.size());
  vk.download_bytes(tokens, after_reload.data(), after_reload.size() * 2);
  CHECK(after_reload == first);
  std::printf(
      "  real H3 block0 S%u: load %.3f ms, CUDA %.3f ms, Vulkan first/repeat %.3f/%.3f ms, FNV64 %016llx, persistent/scratch/peak %.2f/%.2f/%.2f MiB, pool used/reserved %.2f/%.2f MiB, descriptors %llu\n",
      sequence, load_ms, cuda_ms, first_ms, repeat_ms,
      static_cast<unsigned long long>(digest),
      double(stage.persistent_bytes()) / 1048576.0,
      double(scratch.reserved_bytes()) / 1048576.0,
      double(stage.peak_device_bytes(scratch)) / 1048576.0,
      double(vk.pooled_used_bytes()) / 1048576.0,
      double(vk.reserved_bytes()) / 1048576.0,
      static_cast<unsigned long long>(vk.descriptor_set_allocations()));

  if (const char* production = std::getenv("VIDFAB_DIT_GRAPH_PRODUCTION");
      production && production[0] == '1') {
    TensorContextOptions prod_context_options;
    prod_context_options.max_batch_operators = 2048;
    TensorContext prod_vk(device, prod_context_options);
    H3MainGraphConfig prod_config;
    prod_config.layers = 50;
    prod_config.block.sequence = 9864;
    ExactH3MainGraph prod = ExactH3MainGraph::create(prod_vk, prod_config);
    const auto prod_load_begin = std::chrono::steady_clock::now();
    prod.load(checkpoint);
    const double prod_load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - prod_load_begin).count();
    const uint64_t prod_token_shape[] = {prod_config.block.sequence,
                                         prod_config.block.hidden};
    const uint64_t prod_selector_shape[] = {prod_config.block.sequence};
    const uint64_t prod_code_shape[] = {prod_config.block.timesteps,
                                        prod_config.block.adaln_rank};
    const uint64_t prod_rope_shape[] = {prod_config.block.sequence, 96};
    DeviceTensor prod_tokens = prod_vk.allocate(
        TensorLayout::contiguous(prod_token_shape, 2), ScalarType::kBFloat16);
    DeviceTensor prod_selectors = prod_vk.allocate(
        TensorLayout::contiguous(prod_selector_shape, 1), ScalarType::kInt32);
    DeviceTensor prod_code = prod_vk.allocate(
        TensorLayout::contiguous(prod_code_shape, 2));
    DeviceTensor prod_cosine = prod_vk.allocate(
        TensorLayout::contiguous(prod_rope_shape, 2));
    DeviceTensor prod_sine = prod_vk.allocate(
        TensorLayout::contiguous(prod_rope_shape, 2));
    std::vector<uint16_t> prod_input(
        size_t(prod_config.block.sequence) * prod_config.block.hidden);
    for (size_t i = 0; i < prod_input.size(); ++i)
      prod_input[i] = f32_to_bf16(float(int(i % 61) - 30) / 64.0f);
    std::vector<int32_t> prod_selector_values(prod_config.block.sequence);
    for (uint32_t i = 0; i < prod_config.block.sequence; ++i)
      prod_selector_values[i] = static_cast<int32_t>(i % prod_config.block.modalities);
    std::vector<float> prod_code_values(
        size_t(prod_config.block.timesteps) * prod_config.block.adaln_rank);
    for (size_t i = 0; i < prod_code_values.size(); ++i)
      prod_code_values[i] = float(int(i % 7) - 3) / 16.0f;
    std::vector<float> prod_cosine_values(
        size_t(prod_config.block.sequence) * 96, 1.0f);
    std::vector<float> prod_sine_values(prod_cosine_values.size(), 0.0f);
    prod_vk.upload_bytes(prod_selectors, prod_selector_values.data(),
                    prod_selector_values.size() * 4);
    prod_vk.upload(prod_code, prod_code_values.data(), prod_code_values.size());
    prod_vk.upload(prod_cosine, prod_cosine_values.data(), prod_cosine_values.size());
    prod_vk.upload(prod_sine, prod_sine_values.data(), prod_sine_values.size());
    auto run_prod = [&] {
      prod_vk.upload_bytes(prod_tokens, prod_input.data(), prod_input.size() * 2);
      const auto begin = std::chrono::steady_clock::now();
      TensorBatch prod_batch = prod_vk.begin_batch();
      prod.record(prod_batch, prod_tokens, prod_selectors, prod_code,
                  prod_cosine, prod_sine);
      CHECK(prod_batch.remaining_operator_capacity() == 598u);
      prod_batch.submit().wait();
      const double elapsed = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - begin).count();
      std::vector<uint16_t> output(prod_input.size());
      prod_vk.download_bytes(prod_tokens, output.data(), output.size() * 2);
      return std::pair<std::vector<uint16_t>, double>(std::move(output), elapsed);
    };
    auto prod_first = run_prod();
    uint64_t prod_digest = 1469598103934665603ull;
    for (uint16_t bits : prod_first.first) {
      prod_digest ^= bits & 0xffu; prod_digest *= 1099511628211ull;
      prod_digest ^= bits >> 8; prod_digest *= 1099511628211ull;
    }
    // This durable pin follows the separately asserted CUDA/Vulkan block-0
    // S9864 boundary `51e414a3b2556e88`; the full CUDA 50-block capture at
    // S526 remains the cross-backend authority for every graph boundary.
    CHECK(prod_digest == 0x3d59d01afa11ba77ull);
    const uint64_t prod_used = prod_vk.pooled_used_bytes();
    const uint64_t prod_reserved = prod_vk.reserved_bytes();
    const uint64_t prod_descriptors = prod_vk.descriptor_set_allocations();
    auto prod_repeat = run_prod();
    CHECK(prod_repeat.first == prod_first.first);
    CHECK(prod_vk.pooled_used_bytes() == prod_used);
    CHECK(prod_vk.reserved_bytes() == prod_reserved);
    CHECK(prod_vk.descriptor_set_allocations() == prod_descriptors);
    std::printf(
        "  production H3 main50 S9864: load %.3f ms, first/repeat %.3f/%.3f ms, final %016llx, persistent/scratch/peak %.2f/%.2f/%.2f MiB, pool %.2f/%.2f MiB, descriptors %llu\n",
        prod_load_ms, prod_first.second, prod_repeat.second,
        static_cast<unsigned long long>(prod_digest),
        double(prod.persistent_bytes()) / 1048576.0,
        double(prod.scratch_bytes()) / 1048576.0,
        double(prod.peak_device_bytes()) / 1048576.0,
        double(prod_vk.pooled_used_bytes()) / 1048576.0,
        double(prod_vk.reserved_bytes()) / 1048576.0,
        static_cast<unsigned long long>(prod_vk.descriptor_set_allocations()));
  }
}

VIDFAB_TEST(cuda_vulkan_dit_real_capture_replay) {
  using namespace vidfab;
  using namespace vidfab::dit;
  using namespace vidfab::vulkan;
  const std::filesystem::path checkpoint_path =
      "weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors";
  const std::filesystem::path capture_path =
      "tests/data/h3_block0_step0_seed424242_256.vfh3";
  if (!std::filesystem::exists(checkpoint_path) ||
      !std::filesystem::exists(capture_path) || !Instance::available()) return;
  std::ifstream stream(capture_path, std::ios::binary | std::ios::ate);
  const std::streamsize capture_bytes = stream.tellg();
  stream.seekg(0);
  std::vector<uint8_t> capture(static_cast<size_t>(capture_bytes));
  CHECK(static_cast<bool>(stream.read(
      reinterpret_cast<char*>(capture.data()), capture_bytes)));
#ifdef _WIN32
  const std::array<uint8_t, 32> capture_sha =
      sha256_mapping(capture.data(), capture.size());
  const std::array<uint8_t, 32> expected_capture_sha{
      0xe0,0x9e,0x29,0x7b,0x8d,0xc7,0x30,0x06,0xf6,0x75,0x7b,0x0a,0x1a,0x5b,0xf0,0xa6,
      0x05,0x53,0xf0,0x05,0x92,0xea,0xf2,0xa2,0xf0,0x48,0x4d,0x2e,0x11,0x03,0x82,0x75};
  CHECK(capture_sha == expected_capture_sha);
#endif
  if (capture.size() < sizeof(H3BlockCaptureHeader))
    throw std::runtime_error("truncated H3 block capture");
  H3BlockCaptureHeader header{};
  std::memcpy(&header, capture.data(), sizeof(header));
  CHECK(std::memcmp(header.magic, "VFH3BLK\0", 8) == 0);
  CHECK(header.version == 1 && header.header_bytes == sizeof(header));
  CHECK(header.sequence == 538 && header.hidden == 5376 &&
        header.heads == 56 && header.head_dim == 128 && header.ffn == 14336);
  CHECK(header.timesteps == 1 && header.modalities == 3 &&
        header.adaln_rank == 8 && header.denoise_step == 0 &&
        header.layer == 0 && header.range_values == 20);
  CHECK(header.input_fnv64 == 0x7c9f5a55cc5266ebull);
  CHECK(header.qkv_fnv64 == 0x0ce5a1f4d191bdd1ull);
  CHECK(header.attention_fnv64 == 0x550f1253844cd657ull);
  CHECK(header.attention_residual_fnv64 == 0xe8a9ee4dfec51636ull);
  CHECK(header.final_fnv64 == 0x2fd91fe15c281f00ull);
  size_t cursor = sizeof(header);
  auto take = [&](auto& values, size_t count) {
    using Value = typename std::decay_t<decltype(values)>::value_type;
    if (count > (capture.size() - cursor) / sizeof(Value))
      throw std::runtime_error("truncated H3 block capture payload");
    values.resize(count);
    std::memcpy(values.data(), capture.data() + cursor, count * sizeof(Value));
    cursor += count * sizeof(Value);
  };
  std::vector<uint16_t> input, expected_final;
  std::vector<int32_t> selectors, range_values;
  std::vector<float> code, cosine, sine;
  take(input, static_cast<size_t>(header.residual_elements));
  take(selectors, header.sequence);
  take(code, size_t(header.timesteps) * header.adaln_rank);
  take(cosine, static_cast<size_t>(header.rope_elements));
  take(sine, static_cast<size_t>(header.rope_elements));
  take(range_values, header.range_values);
  take(expected_final, static_cast<size_t>(header.residual_elements));
  CHECK(cursor == capture.size());
  CHECK(std::any_of(cosine.begin(), cosine.end(), [](float x) { return x != 1.0f; }));
  CHECK(std::any_of(sine.begin(), sine.end(), [](float x) { return x != 0.0f; }));
  CHECK(std::all_of(selectors.begin(), selectors.end(), [](int32_t x) {
    return x >= 0 && x < 3;
  }));

  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
#ifdef _WIN32
  const std::array<uint8_t, 32> checkpoint_sha = sha256_mapping(
      checkpoint.mapping_base(), checkpoint.file_size());
  const std::array<uint8_t, 32> expected_checkpoint_sha{
      0x6a,0xb7,0xf0,0xc4,0x81,0x41,0xe7,0x91,0x9b,0x32,0xf9,0x25,0xca,0x3d,0xef,0x22,
      0xe0,0x6a,0x6a,0xeb,0xeb,0x9e,0x0b,0x6f,0x5a,0x0b,0xe0,0xfe,0x84,0x09,0x97,0x6f};
  CHECK(checkpoint_sha == expected_checkpoint_sha);
#endif
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) return;
  const DeviceInfo& info = physical.front().info();
  if (!info.timeline_semaphore || !info.shader_int64 || !info.shader_float16 ||
      !info.storage_buffer_16bit ||
      !info.cooperative_matrix_bf16_f32_16x16x16) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  options.enable_shader_float16 = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(options);
  TensorContextOptions context_options; context_options.max_batch_operators = 64;
  TensorContext vk(device, context_options);
  if (!vk.exact_h3_attention()) return;
  H3BlockConfig config;
  config.sequence = header.sequence; config.hidden = header.hidden;
  config.heads = header.heads; config.head_dim = header.head_dim;
  config.ffn = header.ffn; config.timesteps = header.timesteps;
  config.modalities = header.modalities; config.adaln_rank = header.adaln_rank;
  ExactH3BlockStage stage = ExactH3BlockStage::create(vk, config);
  stage.load(checkpoint, header.layer);
  ExactH3BlockScratch scratch = ExactH3BlockScratch::create(vk, config);
  const uint64_t residual_shape[] = {header.sequence, header.hidden};
  const uint64_t selector_shape[] = {header.sequence};
  const uint64_t code_shape[] = {header.timesteps, header.adaln_rank};
  const uint64_t rope_shape[] = {header.sequence, 96};
  const uint64_t qkv_shape[] = {header.sequence, header.heads, header.head_dim};
  const uint64_t attention_shape[] = {header.sequence, header.heads * header.head_dim};
  auto bf = [&](const uint64_t* shape, uint32_t rank) {
    return vk.allocate(TensorLayout::contiguous(shape, rank), ScalarType::kBFloat16);
  };
  DeviceTensor tokens = bf(residual_shape, 2);
  DeviceTensor selector_tensor = vk.allocate(
      TensorLayout::contiguous(selector_shape, 1), ScalarType::kInt32);
  DeviceTensor code_tensor = vk.allocate(TensorLayout::contiguous(code_shape, 2));
  DeviceTensor cosine_tensor = vk.allocate(TensorLayout::contiguous(rope_shape, 2));
  DeviceTensor sine_tensor = vk.allocate(TensorLayout::contiguous(rope_shape, 2));
  DeviceTensor q_tap = bf(qkv_shape, 3), k_tap = bf(qkv_shape, 3),
      v_tap = bf(qkv_shape, 3), attention_tap = bf(attention_shape, 2),
      attention_residual_tap = bf(residual_shape, 2), final_tap = bf(residual_shape, 2);
  vk.upload_bytes(tokens, input.data(), input.size() * 2);
  vk.upload_bytes(selector_tensor, selectors.data(), selectors.size() * 4);
  vk.upload(code_tensor, code.data(), code.size());
  vk.upload(cosine_tensor, cosine.data(), cosine.size());
  vk.upload(sine_tensor, sine.data(), sine.size());
  H3AttentionRanges ranges = H3AttentionRanges::create(
      vk, header.sequence, range_values.data(), header.range_values);
  H3BlockReplayTaps taps{&q_tap, &k_tap, &v_tap, &attention_tap,
                         &attention_residual_tap, &final_tap};
  CHECK(stage.required_operators(&taps) == 35);
  // Fill exactly the spare capacity. A one-operation larger prefix below is
  // rejected before the stage changes tokens or the batch's access state.
  DeviceTensor dummy_a = bf(residual_shape, 2), dummy_b = bf(residual_shape, 2);
  TensorBatch exact = vk.begin_batch();
  for (uint32_t i = stage.required_operators(&taps); i < 64; ++i)
    exact.copy(dummy_a, dummy_b);
  stage.record(exact, tokens, selector_tensor, code_tensor, cosine_tensor,
               sine_tensor, scratch, &ranges, &taps);
  CHECK(exact.remaining_operator_capacity() == 0);
  exact.submit().wait();
  auto fnv = [](const std::vector<uint16_t>& values,
                uint64_t hash = 1469598103934665603ull) {
    for (uint16_t bits : values) {
      hash ^= bits & 0xffu; hash *= 1099511628211ull;
      hash ^= bits >> 8; hash *= 1099511628211ull;
    }
    return hash;
  };
  std::vector<uint16_t> q(header.qkv_elements), k(header.qkv_elements),
      v(header.qkv_elements), attention(header.qkv_elements),
      attention_residual(header.residual_elements), final(header.residual_elements),
      tokens_final(header.residual_elements);
  vk.download_bytes(q_tap, q.data(), q.size() * 2);
  vk.download_bytes(k_tap, k.data(), k.size() * 2);
  vk.download_bytes(v_tap, v.data(), v.size() * 2);
  vk.download_bytes(attention_tap, attention.data(), attention.size() * 2);
  vk.download_bytes(attention_residual_tap, attention_residual.data(), attention_residual.size() * 2);
  vk.download_bytes(final_tap, final.data(), final.size() * 2);
  vk.download_bytes(tokens, tokens_final.data(), tokens_final.size() * 2);
  uint64_t qkv_hash = fnv(q); qkv_hash = fnv(k, qkv_hash); qkv_hash = fnv(v, qkv_hash);
  CHECK(qkv_hash == header.qkv_fnv64);
  CHECK(fnv(attention) == header.attention_fnv64);
  CHECK(fnv(attention_residual) == header.attention_residual_fnv64);
  CHECK(fnv(final) == header.final_fnv64);
  CHECK(final == expected_final && tokens_final == expected_final);

  // Late-shape and range faults are transactional and consume no operator.
  const uint64_t bad_rope_shape[] = {header.sequence, 95};
  DeviceTensor bad_sine = vk.allocate(TensorLayout::contiguous(bad_rope_shape, 2));
  std::vector<int32_t> wrong_values(4 * ((header.sequence + 127) / 128), 0);
  for (size_t i = 0; i < wrong_values.size(); i += 4)
    wrong_values[i + 1] = 576;
  H3AttentionRanges wrong_ranges = H3AttentionRanges::create(
      vk, header.sequence + 1, wrong_values.data(),
      static_cast<uint32_t>(wrong_values.size()));
  {
  TensorBatch invalid = vk.begin_batch();
  bool bad_rope_threw = false;
  try { stage.record(invalid, tokens, selector_tensor, code_tensor, cosine_tensor,
                     bad_sine, scratch, &ranges); }
  catch (const std::invalid_argument&) { bad_rope_threw = true; }
  CHECK(bad_rope_threw && invalid.remaining_operator_capacity() == 64);
  bool bad_ranges_threw = false;
  try { stage.record(invalid, tokens, selector_tensor, code_tensor, cosine_tensor,
                     sine_tensor, scratch, &wrong_ranges); }
  catch (const std::invalid_argument&) { bad_ranges_threw = true; }
  CHECK(bad_ranges_threw && invalid.remaining_operator_capacity() == 64);
  }
  {
  TensorBatch short_capacity = vk.begin_batch();
  for (uint32_t i = stage.required_operators(); i <= 64; ++i)
    short_capacity.copy(dummy_a, dummy_b);
  bool capacity_threw = false;
  try { stage.record(short_capacity, tokens, selector_tensor, code_tensor,
                     cosine_tensor, sine_tensor, scratch, &ranges); }
  catch (const std::logic_error&) { capacity_threw = true; }
  CHECK(capacity_threw && short_capacity.remaining_operator_capacity() ==
        stage.required_operators() - 1);
  }

  bool overflow_threw = false;
  try {
    H3BlockConfig overflow = config; overflow.heads = UINT32_MAX;
    (void)ExactH3BlockStage::create(vk, overflow);
  } catch (const std::invalid_argument&) { overflow_threw = true; }
  CHECK(overflow_threw);
}

VIDFAB_TEST(cuda_vulkan_dit_real_main50_capture_replay) {
  using namespace vidfab;
  using namespace vidfab::dit;
  using namespace vidfab::vulkan;
  const std::filesystem::path checkpoint_path =
      "weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors";
  const std::filesystem::path capture_path =
      "tests/data/h3_main50_step0_seed424242_256.vfh3g";
  if (!std::filesystem::exists(checkpoint_path) ||
      !std::filesystem::exists(capture_path) || !Instance::available()) return;
  std::ifstream stream(capture_path, std::ios::binary | std::ios::ate);
  const std::streamsize capture_bytes = stream.tellg();
  stream.seekg(0);
  std::vector<uint8_t> capture(static_cast<size_t>(capture_bytes));
  CHECK(static_cast<bool>(stream.read(
      reinterpret_cast<char*>(capture.data()), capture_bytes)));
#ifdef _WIN32
  const std::array<uint8_t, 32> expected_capture_sha{
      0xbb,0x15,0x68,0x92,0x83,0x66,0x94,0x96,0xc5,0x61,0x4e,0xc3,0x0f,0xaa,0xf0,0x88,
      0x12,0x9d,0x8c,0xb8,0x97,0xcb,0x34,0x0a,0xe5,0x36,0x37,0x3a,0xd4,0x2c,0xb3,0x62};
  CHECK(sha256_mapping(capture.data(), capture.size()) == expected_capture_sha);
#endif
  if (capture.size() < sizeof(H3MainGraphCaptureHeader))
    throw std::runtime_error("truncated H3 main graph capture");
  H3MainGraphCaptureHeader header{};
  std::memcpy(&header, capture.data(), sizeof(header));
  CHECK(std::memcmp(header.magic, "VFH3GRF\0", 8) == 0);
  CHECK(header.version == 1 && header.header_bytes == sizeof(header));
  CHECK(header.sequence == 526 && header.hidden == 5376 &&
        header.heads == 56 && header.head_dim == 128 && header.ffn == 14336);
  CHECK(header.timesteps == 1 && header.modalities == 3 &&
        header.adaln_rank == 8 && header.layers == 50 &&
        header.range_values == 20 && header.denoise_step == 0);
  CHECK(header.input_fnv64 == 0x8e130a074619290full);
  CHECK(header.final_fnv64 == 0x94d7dfcfcef6f4ceull);
  const std::array<uint64_t, 50> expected_boundaries{
      0xf7d651766756f015ull,0x09afbb821887020bull,0x779869da69c31d75ull,
      0x8b76a7261ab91253ull,0x1df1afd29841879cull,0x61db215dee7be558ull,
      0x93bdf2ff13a3ba62ull,0xa8a94eae981cddb7ull,0x23e0dd2de693a165ull,
      0xdb0e0f8aa5a49972ull,0x1ee2ee965c134dc1ull,0x4bc93b42a31b9a63ull,
      0xed1f848054d1213cull,0xcd0c9af21969802cull,0x1c2dbc71eddb6bbaull,
      0x4e5096ae2df1ef5full,0xddca1935d085d1d1ull,0xaa8ed6a379181141ull,
      0xa1ae4a76ac4cf58aull,0xddfb16645d9e3ddaull,0x8d7e09064314ab3cull,
      0x50f0367cff5988b9ull,0x28be37c299d8d34dull,0xef16ad91e90c119aull,
      0x47edfc041ca17670ull,0x0d42e4191058937dull,0x53ffb9a8f3a2809eull,
      0x6bf28610fbf92805ull,0x71b964652480244dull,0xbd860e69ea9f1bfeull,
      0x3f1fe901496195cfull,0x5b04fa6007303ccfull,0xb97828cd5a0c0ddfull,
      0x9be29ebd17324ad8ull,0x9593307ba6dca3faull,0x25b1d163a816d621ull,
      0xff23adde8d9a9e50ull,0xc779a185f7d8aa8aull,0xf1951f02921680a6ull,
      0x18b7758650a17769ull,0x7ffd3c6465091afbull,0x9b99975035dcf76dull,
      0x2977798f27e1a904ull,0xdace59bec11d235aull,0x6a30ef6d8415272full,
      0xbdb70b87dcc70527ull,0xabd55e9b1b59d6caull,0x2220fd508ca8a60cull,
      0xd8f6dc81376879f7ull,0x94d7dfcfcef6f4ceull};
  for (uint32_t layer = 0; layer < 50; ++layer)
    CHECK(header.boundary_fnv64[layer] == expected_boundaries[layer]);

  size_t cursor = sizeof(header);
  auto take = [&](auto& values, size_t count) {
    using Value = typename std::decay_t<decltype(values)>::value_type;
    if (count > (capture.size() - cursor) / sizeof(Value))
      throw std::runtime_error("truncated H3 graph capture payload");
    values.resize(count);
    std::memcpy(values.data(), capture.data() + cursor, count * sizeof(Value));
    cursor += count * sizeof(Value);
  };
  std::vector<uint16_t> input, expected_final;
  std::vector<int32_t> selectors, range_values;
  std::vector<float> code, cosine, sine;
  take(input, static_cast<size_t>(header.residual_elements));
  take(selectors, header.sequence);
  take(code, size_t(header.timesteps) * header.adaln_rank);
  take(cosine, static_cast<size_t>(header.rope_elements));
  take(sine, static_cast<size_t>(header.rope_elements));
  take(range_values, header.range_values);
  take(expected_final, static_cast<size_t>(header.residual_elements));
  CHECK(cursor == capture.size());
  CHECK(std::any_of(cosine.begin(), cosine.end(), [](float x) { return x != 1.0f; }));
  CHECK(std::any_of(sine.begin(), sine.end(), [](float x) { return x != 0.0f; }));

  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
#ifdef _WIN32
  const std::array<uint8_t, 32> expected_checkpoint_sha{
      0x6a,0xb7,0xf0,0xc4,0x81,0x41,0xe7,0x91,0x9b,0x32,0xf9,0x25,0xca,0x3d,0xef,0x22,
      0xe0,0x6a,0x6a,0xeb,0xeb,0x9e,0x0b,0x6f,0x5a,0x0b,0xe0,0xfe,0x84,0x09,0x97,0x6f};
  CHECK(sha256_mapping(checkpoint.mapping_base(), checkpoint.file_size()) ==
        expected_checkpoint_sha);
#endif
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) return;
  const DeviceInfo& info = physical.front().info();
  if (!info.timeline_semaphore || !info.shader_int64 || !info.shader_float16 ||
      !info.storage_buffer_16bit ||
      !info.cooperative_matrix_bf16_f32_16x16x16) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  options.enable_shader_float16 = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(options);
  TensorContextOptions context_options;
  context_options.max_batch_operators = 2048;
  TensorContext vk(device, context_options);
  if (!vk.exact_h3_attention()) return;
  H3MainGraphConfig graph_config;
  graph_config.layers = header.layers;
  graph_config.block.sequence = header.sequence;
  graph_config.block.hidden = header.hidden;
  graph_config.block.heads = header.heads;
  graph_config.block.head_dim = header.head_dim;
  graph_config.block.ffn = header.ffn;
  graph_config.block.timesteps = header.timesteps;
  graph_config.block.modalities = header.modalities;
  graph_config.block.adaln_rank = header.adaln_rank;
  const auto load_begin = std::chrono::steady_clock::now();
  ExactH3MainGraph graph = ExactH3MainGraph::create(vk, graph_config);
  graph.load(checkpoint);
  const double load_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - load_begin).count();
  CHECK(graph.layers() == 50u && graph.loaded());
  CHECK(graph.required_operators() == 1450u);
  const uint64_t residual_shape[] = {header.sequence, header.hidden};
  const uint64_t selector_shape[] = {header.sequence};
  const uint64_t code_shape[] = {header.timesteps, header.adaln_rank};
  const uint64_t rope_shape[] = {header.sequence, 96};
  auto bf = [&](const uint64_t* shape, uint32_t rank) {
    return vk.allocate(TensorLayout::contiguous(shape, rank), ScalarType::kBFloat16);
  };
  DeviceTensor tokens = bf(residual_shape, 2);
  DeviceTensor selector_tensor = vk.allocate(
      TensorLayout::contiguous(selector_shape, 1), ScalarType::kInt32);
  DeviceTensor code_tensor = vk.allocate(TensorLayout::contiguous(code_shape, 2));
  DeviceTensor cosine_tensor = vk.allocate(TensorLayout::contiguous(rope_shape, 2));
  DeviceTensor sine_tensor = vk.allocate(TensorLayout::contiguous(rope_shape, 2));
  vk.upload_bytes(selector_tensor, selectors.data(), selectors.size() * 4);
  vk.upload(code_tensor, code.data(), code.size());
  vk.upload(cosine_tensor, cosine.data(), cosine.size());
  vk.upload(sine_tensor, sine.data(), sine.size());
  H3AttentionRanges ranges = H3AttentionRanges::create(
      vk, header.sequence, range_values.data(), header.range_values);
  std::vector<DeviceTensor> boundary_tensors;
  boundary_tensors.reserve(header.layers);
  for (uint32_t layer = 0; layer < header.layers; ++layer)
    boundary_tensors.push_back(bf(residual_shape, 2));
  H3MainGraphReplayTaps taps{boundary_tensors.data(), header.layers};
  CHECK(graph.required_operators(&taps) == 1500u);
  vk.upload_bytes(tokens, input.data(), input.size() * 2);
  const auto first_begin = std::chrono::steady_clock::now();
  TensorBatch batch = vk.begin_batch();
  graph.record(batch, tokens, selector_tensor, code_tensor, cosine_tensor,
               sine_tensor, &ranges, &taps);
  CHECK(batch.remaining_operator_capacity() == 548u);
  batch.submit().wait();
  const double first_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - first_begin).count();
  auto fnv = [](const std::vector<uint16_t>& values) {
    uint64_t hash = 1469598103934665603ull;
    for (uint16_t bits : values) {
      hash ^= bits & 0xffu; hash *= 1099511628211ull;
      hash ^= bits >> 8; hash *= 1099511628211ull;
    }
    return hash;
  };
  for (uint32_t layer = 0; layer < header.layers; ++layer) {
    std::vector<uint16_t> boundary(header.residual_elements);
    vk.download_bytes(boundary_tensors[layer], boundary.data(), boundary.size() * 2);
    CHECK(fnv(boundary) == expected_boundaries[layer]);
  }
  std::vector<uint16_t> final(header.residual_elements);
  vk.download_bytes(tokens, final.data(), final.size() * 2);
  CHECK(final == expected_final);
  CHECK(fnv(final) == header.final_fnv64);
  const uint64_t stable_used = vk.pooled_used_bytes();
  const uint64_t stable_reserved = vk.reserved_bytes();
  const uint64_t stable_descriptors = vk.descriptor_set_allocations();
  auto repeat = [&] {
    vk.upload_bytes(tokens, input.data(), input.size() * 2);
    const auto begin = std::chrono::steady_clock::now();
    TensorBatch next = vk.begin_batch();
    graph.record(next, tokens, selector_tensor, code_tensor, cosine_tensor,
                 sine_tensor, &ranges);
    next.submit().wait();
    const double elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - begin).count();
    std::vector<uint16_t> result(header.residual_elements);
    vk.download_bytes(tokens, result.data(), result.size() * 2);
    CHECK(result == expected_final);
    return elapsed;
  };
  const double repeat_ms = repeat();
  CHECK(vk.pooled_used_bytes() == stable_used);
  CHECK(vk.reserved_bytes() == stable_reserved);
  CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  (void)repeat();
  CHECK(vk.pooled_used_bytes() == stable_used);
  CHECK(vk.reserved_bytes() == stable_reserved);
  CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  std::printf(
      "  real H3 main50 S%u: load %.3f ms, Vulkan taps/repeat %.3f/%.3f ms, final %016llx, persistent/scratch/peak %.2f/%.2f/%.2f MiB, pool %.2f/%.2f MiB, descriptors %llu\n",
      header.sequence, load_ms, first_ms, repeat_ms,
      static_cast<unsigned long long>(header.final_fnv64),
      double(graph.persistent_bytes()) / 1048576.0,
      double(graph.scratch_bytes()) / 1048576.0,
      double(graph.peak_device_bytes()) / 1048576.0,
      double(vk.pooled_used_bytes()) / 1048576.0,
      double(vk.reserved_bytes()) / 1048576.0,
      static_cast<unsigned long long>(vk.descriptor_set_allocations()));

}

VIDFAB_TEST(cuda_vulkan_dit_real_transformer_capture_replay) {
  using namespace vidfab;
  using namespace vidfab::dit;
  using namespace vidfab::vulkan;
  const std::filesystem::path checkpoint_path =
      "weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors";
  const std::filesystem::path capture_path =
      "tests/data/h3_transformer_step0_seed424242_256.vfh3f";
  if (!std::filesystem::exists(checkpoint_path) ||
      !std::filesystem::exists(capture_path) || !Instance::available()) return;
  std::ifstream stream(capture_path, std::ios::binary | std::ios::ate);
  const std::streamsize capture_bytes = stream.tellg();
  stream.seekg(0);
  std::vector<uint8_t> capture(static_cast<size_t>(capture_bytes));
  CHECK(static_cast<bool>(stream.read(
      reinterpret_cast<char*>(capture.data()), capture_bytes)));
#ifdef _WIN32
  const std::array<uint8_t, 32> expected_capture_sha{
      0x3e,0x34,0x76,0xe3,0x97,0xfc,0xee,0x20,0x37,0x37,0x33,0x2d,0x17,0x1d,0x46,0x50,
      0xf5,0x54,0x33,0x47,0x12,0x31,0xa7,0xfd,0x4f,0xf2,0x4f,0x8e,0x0f,0x84,0xf8,0xe7};
  CHECK(sha256_mapping(capture.data(), capture.size()) == expected_capture_sha);
#endif
  if (capture.size() < sizeof(H3TransformerCaptureHeader))
    throw std::runtime_error("truncated H3 transformer capture");
  H3TransformerCaptureHeader header{};
  std::memcpy(&header, capture.data(), sizeof(header));
  CHECK(std::memcmp(header.magic, "VFH3FWD\0", 8) == 0);
  CHECK(header.version == 1 && header.header_bytes == sizeof(header));
  CHECK(header.sequence == 526 && header.hidden == 5376 &&
        header.heads == 56 && header.head_dim == 128 && header.ffn == 14336);
  CHECK(header.timesteps == 1 && header.modalities == 3 &&
        header.adaln_rank == 8 && header.layers == 50 &&
        header.text_rows == 4 && header.video_rows == 448 &&
        header.audio_rows == 74 && header.text_dim == 5120 &&
        header.video_dim == 96 && header.audio_dim == 32 &&
        header.refiner_layers == 2 && header.range_values == 20 &&
        header.denoise_step == 0);
  const std::array<uint64_t, 6> expected_text_hashes{
      0x6d891a14ee2a38bdull, 0xcc91d0e61a56bd7bull,
      0xd5ddff8656b1d581ull, 0xaf5733d914839cf2ull,
      0xd510022f4c0e9032ull, 0x1e4af4a0c48fffc7ull};
  for (uint32_t stage = 0; stage < expected_text_hashes.size(); ++stage)
    CHECK(header.text_boundary_fnv64[stage] == expected_text_hashes[stage]);
  CHECK(header.packed_input_fnv64 == 0x54c4e5ce3af6d0deull);
  CHECK(header.main_final_fnv64 == 0xda1038eb60eea19full);
  CHECK(header.video_output_fnv64 == 0x7d7af2480929ae03ull);
  CHECK(header.audio_output_fnv64 == 0x58197cbc23da3ab3ull);

  size_t cursor = sizeof(header);
  auto take = [&](auto& values, size_t count) {
    using Value = typename std::decay_t<decltype(values)>::value_type;
    if (count > (capture.size() - cursor) / sizeof(Value))
      throw std::runtime_error("truncated H3 transformer capture payload");
    values.resize(count);
    std::memcpy(values.data(), capture.data() + cursor, count * sizeof(Value));
    cursor += count * sizeof(Value);
  };
  std::vector<float> prompt, video, audio, code, cosine, sine,
      expected_video, expected_audio;
  std::vector<int32_t> selectors, ranges_data, video_ts, audio_ts;
  std::vector<uint16_t> expected_text, expected_packed, expected_main;
  take(prompt, header.prompt_elements);
  take(video, header.video_elements);
  take(audio, header.audio_elements);
  take(selectors, header.sequence);
  take(code, size_t(header.timesteps) * header.adaln_rank);
  take(cosine, header.rope_elements);
  take(sine, header.rope_elements);
  take(ranges_data, header.range_values);
  take(video_ts, header.video_rows);
  take(audio_ts, header.audio_rows);
  take(expected_text, header.text_elements);
  take(expected_packed, header.packed_elements);
  take(expected_main, header.packed_elements);
  take(expected_video, header.video_elements);
  take(expected_audio, header.audio_elements);
  CHECK(cursor == capture.size());
  CHECK(std::any_of(cosine.begin(), cosine.end(), [](float x) { return x != 1.0f; }));
  CHECK(std::any_of(sine.begin(), sine.end(), [](float x) { return x != 0.0f; }));
  auto fnv_bytes = [](const void* data, size_t bytes) {
    uint64_t hash = 1469598103934665603ull;
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < bytes; ++i) { hash ^= p[i]; hash *= 1099511628211ull; }
    return hash;
  };
  CHECK(fnv_bytes(expected_text.data(), expected_text.size() * 2) ==
        header.text_boundary_fnv64[5]);
  CHECK(fnv_bytes(expected_packed.data(), expected_packed.size() * 2) ==
        header.packed_input_fnv64);
  CHECK(fnv_bytes(expected_main.data(), expected_main.size() * 2) ==
        header.main_final_fnv64);
  CHECK(fnv_bytes(expected_video.data(), expected_video.size() * 4) ==
        header.video_output_fnv64);
  CHECK(fnv_bytes(expected_audio.data(), expected_audio.size() * 4) ==
        header.audio_output_fnv64);

  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
#ifdef _WIN32
  const std::array<uint8_t, 32> expected_checkpoint_sha{
      0x6a,0xb7,0xf0,0xc4,0x81,0x41,0xe7,0x91,0x9b,0x32,0xf9,0x25,0xca,0x3d,0xef,0x22,
      0xe0,0x6a,0x6a,0xeb,0xeb,0x9e,0x0b,0x6f,0x5a,0x0b,0xe0,0xfe,0x84,0x09,0x97,0x6f};
  CHECK(sha256_mapping(checkpoint.mapping_base(), checkpoint.file_size()) ==
        expected_checkpoint_sha);
#endif
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) return;
  const DeviceInfo& info = physical.front().info();
  if (!info.timeline_semaphore || !info.shader_int64 || !info.shader_float16 ||
      !info.storage_buffer_16bit ||
      !info.cooperative_matrix_bf16_f32_16x16x16) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  options.enable_shader_float16 = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_cooperative_matrix = true;
  Device device = physical.front().create_device(options);
  TensorContextOptions context_options;
  context_options.max_batch_operators = 2048;
  TensorContext vk(device, context_options);
  if (!vk.exact_h3_attention()) return;
  ExactH3TransformerConfig config;
  config.main.layers = header.layers;
  config.main.block.sequence = header.sequence;
  config.main.block.hidden = header.hidden;
  config.main.block.heads = header.heads;
  config.main.block.head_dim = header.head_dim;
  config.main.block.ffn = header.ffn;
  config.main.block.timesteps = header.timesteps;
  config.main.block.modalities = header.modalities;
  config.main.block.adaln_rank = header.adaln_rank;
  config.text_rows = header.text_rows;
  config.video_rows = header.video_rows;
  config.audio_rows = header.audio_rows;
  config.text_dim = header.text_dim;
  config.video_dim = header.video_dim;
  config.audio_dim = header.audio_dim;
  config.refiner_layers = header.refiner_layers;

  auto tensor2 = [&](uint64_t rows, uint64_t columns,
                     ScalarType type = ScalarType::kFloat32) {
    const uint64_t shape[] = {rows, columns};
    return vk.allocate(TensorLayout::contiguous(shape, 2), type);
  };
  auto tensor1 = [&](uint64_t rows, ScalarType type) {
    const uint64_t shape[] = {rows};
    return vk.allocate(TensorLayout::contiguous(shape, 1), type);
  };
  DeviceTensor prompt_tensor = tensor2(header.text_rows, header.text_dim);
  DeviceTensor video_tensor = tensor2(header.video_rows, header.video_dim);
  DeviceTensor audio_tensor = tensor2(header.audio_rows, header.audio_dim);
  DeviceTensor selector_tensor = tensor1(header.sequence, ScalarType::kInt32);
  DeviceTensor code_tensor = tensor2(header.timesteps, header.adaln_rank);
  DeviceTensor cosine_tensor = tensor2(header.sequence, 96);
  DeviceTensor sine_tensor = tensor2(header.sequence, 96);
  DeviceTensor video_ts_tensor = tensor1(header.video_rows, ScalarType::kInt32);
  DeviceTensor audio_ts_tensor = tensor1(header.audio_rows, ScalarType::kInt32);
  DeviceTensor video_output = tensor2(header.video_rows, header.video_dim);
  DeviceTensor audio_output = tensor2(header.audio_rows, header.audio_dim);
  vk.upload(prompt_tensor, prompt.data(), prompt.size());
  vk.upload(video_tensor, video.data(), video.size());
  vk.upload(audio_tensor, audio.data(), audio.size());
  vk.upload_bytes(selector_tensor, selectors.data(), selectors.size() * 4);
  vk.upload(code_tensor, code.data(), code.size());
  vk.upload(cosine_tensor, cosine.data(), cosine.size());
  vk.upload(sine_tensor, sine.data(), sine.size());
  vk.upload_bytes(video_ts_tensor, video_ts.data(), video_ts.size() * 4);
  vk.upload_bytes(audio_ts_tensor, audio_ts.data(), audio_ts.size() * 4);
  H3AttentionRanges ranges = H3AttentionRanges::create(
      vk, header.sequence, ranges_data.data(), header.range_values);

  std::vector<DeviceTensor> text_boundaries;
  for (uint32_t i = 0; i < 6; ++i)
    text_boundaries.push_back(tensor2(header.text_rows, header.hidden,
                                     ScalarType::kBFloat16));
  DeviceTensor packed_tap = tensor2(header.sequence, header.hidden,
                                    ScalarType::kBFloat16);
  DeviceTensor main_tap = tensor2(header.sequence, header.hidden,
                                  ScalarType::kBFloat16);
  H3TransformerTextReplayTaps text_taps{text_boundaries.data(), 6};
  H3TransformerForwardReplayTaps forward_taps{&packed_tap, &main_tap, nullptr};
  const auto load_begin = std::chrono::steady_clock::now();
  ExactH3Transformer transformer = ExactH3Transformer::create(vk, config);
  transformer.load(checkpoint);
  const double load_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - load_begin).count();
  const uint32_t text_ops =
      transformer.required_prepare_text_operators(&text_taps);
  CHECK_MSG(text_ops == 43u, "real H3 transformer text ops %u != 43", text_ops);
  const auto text_begin = std::chrono::steady_clock::now();
  transformer.prepare_text(prompt_tensor, &text_taps);
  const double text_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - text_begin).count();
  for (uint32_t stage = 0; stage < 6; ++stage) {
    std::vector<uint16_t> actual(header.text_elements);
    vk.download_bytes(text_boundaries[stage], actual.data(), actual.size() * 2);
    CHECK_MSG(fnv_bytes(actual.data(), actual.size() * 2) ==
                  header.text_boundary_fnv64[stage],
              "real H3 transformer text boundary %u mismatch: %016llx != %016llx",
              stage, static_cast<unsigned long long>(
                  fnv_bytes(actual.data(), actual.size() * 2)),
              static_cast<unsigned long long>(header.text_boundary_fnv64[stage]));
    if (stage == 5) CHECK(actual == expected_text);
  }
  CHECK(transformer.required_forward_operators(&forward_taps) == 1469u);
  const auto first_begin = std::chrono::steady_clock::now();
  TensorBatch first = vk.begin_batch();
  transformer.record_forward(first, video_tensor, audio_tensor,
      selector_tensor, code_tensor, cosine_tensor, sine_tensor,
      video_ts_tensor, audio_ts_tensor, video_output, audio_output,
      &ranges, &forward_taps);
  CHECK(first.remaining_operator_capacity() == 579u);
  first.submit().wait();
  const double first_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - first_begin).count();
  std::vector<uint16_t> actual_packed(header.packed_elements),
      actual_main(header.packed_elements);
  std::vector<float> actual_video(header.video_elements),
      actual_audio(header.audio_elements);
  vk.download_bytes(packed_tap, actual_packed.data(), actual_packed.size() * 2);
  vk.download_bytes(main_tap, actual_main.data(), actual_main.size() * 2);
  vk.download(video_output, actual_video.data(), actual_video.size());
  vk.download(audio_output, actual_audio.data(), actual_audio.size());
  CHECK(actual_packed == expected_packed);
  CHECK(actual_main == expected_main);
  CHECK(std::memcmp(actual_video.data(), expected_video.data(),
                    actual_video.size() * 4) == 0);
  CHECK(std::memcmp(actual_audio.data(), expected_audio.data(),
                    actual_audio.size() * 4) == 0);
  const uint64_t stable_used = vk.pooled_used_bytes();
  const uint64_t stable_reserved = vk.reserved_bytes();
  const uint64_t stable_descriptors = vk.descriptor_set_allocations();
  const auto repeat_begin = std::chrono::steady_clock::now();
  TensorBatch repeat = vk.begin_batch();
  transformer.record_forward(repeat, video_tensor, audio_tensor,
      selector_tensor, code_tensor, cosine_tensor, sine_tensor,
      video_ts_tensor, audio_ts_tensor, video_output, audio_output, &ranges);
  repeat.submit().wait();
  const double repeat_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - repeat_begin).count();
  vk.download(video_output, actual_video.data(), actual_video.size());
  vk.download(audio_output, actual_audio.data(), actual_audio.size());
  CHECK(std::memcmp(actual_video.data(), expected_video.data(),
                    actual_video.size() * 4) == 0);
  CHECK(std::memcmp(actual_audio.data(), expected_audio.data(),
                    actual_audio.size() * 4) == 0);
  CHECK(vk.pooled_used_bytes() == stable_used);
  CHECK(vk.reserved_bytes() == stable_reserved);
  CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  std::printf(
      "  real H3 transformer S%u: load/text %.3f/%.3f ms, Vulkan taps/repeat %.3f/%.3f ms, packed/main/video/audio %016llx/%016llx/%016llx/%016llx, persistent/scratch/peak %.2f/%.2f/%.2f MiB, pool %.2f/%.2f MiB, descriptors %llu\n",
      header.sequence, load_ms, text_ms, first_ms, repeat_ms,
      static_cast<unsigned long long>(header.packed_input_fnv64),
      static_cast<unsigned long long>(header.main_final_fnv64),
      static_cast<unsigned long long>(header.video_output_fnv64),
      static_cast<unsigned long long>(header.audio_output_fnv64),
      double(transformer.persistent_bytes()) / 1048576.0,
      double(transformer.scratch_bytes()) / 1048576.0,
      double(transformer.peak_device_bytes()) / 1048576.0,
      double(vk.pooled_used_bytes()) / 1048576.0,
      double(vk.reserved_bytes()) / 1048576.0,
      static_cast<unsigned long long>(vk.descriptor_set_allocations()));

  const uint64_t loaded_used = vk.pooled_used_bytes();
  transformer.unload();
  CHECK(!transformer.loaded() && transformer.persistent_bytes() == 0u &&
        transformer.scratch_bytes() == 0u &&
        transformer.peak_device_bytes() == 0u);
  CHECK(vk.pooled_used_bytes() < loaded_used);
  transformer.load(checkpoint);
  transformer.prepare_text(prompt_tensor);
  TensorBatch reloaded = vk.begin_batch();
  transformer.record_forward(reloaded, video_tensor, audio_tensor,
      selector_tensor, code_tensor, cosine_tensor, sine_tensor,
      video_ts_tensor, audio_ts_tensor, video_output, audio_output, &ranges);
  reloaded.submit().wait();
  vk.download(video_output, actual_video.data(), actual_video.size());
  vk.download(audio_output, actual_audio.data(), actual_audio.size());
  CHECK(std::memcmp(actual_video.data(), expected_video.data(),
                    actual_video.size() * 4) == 0);
  CHECK(std::memcmp(actual_audio.data(), expected_audio.data(),
                    actual_audio.size() * 4) == 0);
  transformer.unload();

  if (const char* real_denoise = std::getenv("VIDFAB_DIT_DENOISE_REAL");
      real_denoise && real_denoise[0] == '1') {
    SequenceLayout denoise_layout;
    denoise_layout.num_text = static_cast<int>(header.text_rows);
    denoise_layout.num_audio_rows = static_cast<int>(header.audio_rows);
    denoise_layout.num_video_rows = static_cast<int>(header.video_rows);
    denoise_layout.num_audio_latents = static_cast<int>(header.audio_rows / 2);
    denoise_layout.num_latent_frames = 7;
    denoise_layout.latent_height = 16;
    denoise_layout.latent_width = 16;
    CHECK(denoise_layout.total_rows() == static_cast<int>(header.sequence) &&
          denoise_layout.num_latent_frames * denoise_layout.rows_per_frame() ==
              static_cast<int>(header.video_rows));
    const PackedIndices denoise_indices = build_indices(denoise_layout);
    const std::vector<double> denoise_positions =
        build_position_ids(denoise_layout);

    auto joined_hash = [&](const std::vector<float>& video_rows,
                           const std::vector<float>& audio_rows) {
      uint64_t hash = 1469598103934665603ull;
      auto append = [&](const std::vector<float>& rows) {
        const auto* bytes = reinterpret_cast<const uint8_t*>(rows.data());
        for (size_t i = 0; i < rows.size() * sizeof(float); ++i) {
          hash ^= bytes[i];
          hash *= 1099511628211ull;
        }
      };
      append(video_rows);
      append(audio_rows);
      return hash;
    };
    struct Boundary {
      std::vector<float> video;
      std::vector<float> audio;
    };
    std::vector<Boundary> cuda_boundaries;

    dit::Transformer cuda_model;
    const auto cuda_load_begin = std::chrono::steady_clock::now();
    cuda_model.load(checkpoint);
    cuda_model.set_attention_mode(AttentionMode::kExact);
    cuda_model.set_attention_band(0);
    cuda_model.prepare_text(prompt.data(), static_cast<int>(header.text_rows));
    cuda_model.prepare_sequence(
        denoise_layout, denoise_indices, denoise_positions);
    const double cuda_load_prepare_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - cuda_load_begin).count();
    sampler::FlowScheduler cuda_video_schedule(12.0f);
    sampler::FlowScheduler cuda_audio_schedule(3.0f);
    cuda_video_schedule.set_timesteps(4);
    cuda_audio_schedule.set_timesteps(4);
    DenoiseInputs cuda_inputs;
    cuda_inputs.layout = &denoise_layout;
    cuda_inputs.indices = &denoise_indices;
    cuda_inputs.video_timesteps = &cuda_video_schedule.timesteps();
    cuda_inputs.audio_timesteps = &cuda_audio_schedule.timesteps();
    cuda_inputs.video_scheduler = &cuda_video_schedule;
    cuda_inputs.audio_scheduler = &cuda_audio_schedule;
    cuda_inputs.init_video_rows = &video;
    cuda_inputs.init_audio_rows = &audio;
    cuda_inputs.boundary = [&](int, const std::vector<float>& video_rows,
                               const std::vector<float>& audio_rows) {
      cuda_boundaries.push_back({video_rows, audio_rows});
    };
    const auto cuda_run_begin = std::chrono::steady_clock::now();
    const DenoiseOutputs cuda_result = denoise(cuda_model, cuda_inputs);
    const double cuda_run_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - cuda_run_begin).count();
    CHECK(cuda_result.steps_computed == 3 &&
          cuda_result.steps_skipped == 0 && cuda_boundaries.size() == 3u);
    cuda_model.unload();
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());

    ExactH3DenoiseConfig denoise_config;
    denoise_config.transformer = config;
    denoise_config.transformer.main.block.timesteps = 2;
    denoise_config.layout = denoise_layout;
    denoise_config.indices = denoise_indices;
    denoise_config.position_ids = denoise_positions;
    denoise_config.attention_ranges = ranges_data;
    ExactH3Denoiser vk_denoiser = ExactH3Denoiser::create(vk, denoise_config);
    const auto vk_load_begin = std::chrono::steady_clock::now();
    vk_denoiser.load(checkpoint);
    vk_denoiser.prepare(prompt.data(), prompt.size(), video.data(), video.size(),
                        audio.data(), audio.size());
    const double vk_load_prepare_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - vk_load_begin).count();
    sampler::FlowScheduler vk_video_schedule(12.0f);
    sampler::FlowScheduler vk_audio_schedule(3.0f);
    vk_video_schedule.set_timesteps(4);
    vk_audio_schedule.set_timesteps(4);
    std::vector<uint64_t> boundary_hashes;
    size_t boundary_index = 0;
    const auto vk_run_begin = std::chrono::steady_clock::now();
    const ExactH3DenoiseResult vk_result = vk_denoiser.run(
        vk_video_schedule, vk_audio_schedule, {},
        [&](uint32_t step, const std::vector<float>& video_rows,
            const std::vector<float>& audio_rows) {
          CHECK(step == boundary_index && boundary_index < cuda_boundaries.size());
          if (boundary_index < cuda_boundaries.size()) {
            CHECK(video_rows == cuda_boundaries[boundary_index].video);
            CHECK(audio_rows == cuda_boundaries[boundary_index].audio);
          }
          boundary_hashes.push_back(joined_hash(video_rows, audio_rows));
          ++boundary_index;
        });
    const double vk_run_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - vk_run_begin).count();
    CHECK(!vk_result.cancelled && vk_result.steps_completed == 3u &&
          boundary_index == cuda_boundaries.size());
    CHECK(vk_result.video_rows == cuda_result.video_rows &&
          vk_result.audio_rows == cuda_result.audio_rows);
    CHECK(boundary_hashes.size() == 3u);
    const std::array<uint64_t, 3> expected_denoise_hashes{
        0x2472491d7573a692ull, 0x3343aa4828944315ull,
        0xbfc3aec499e7b836ull};
    CHECK(std::equal(boundary_hashes.begin(), boundary_hashes.end(),
                     expected_denoise_hashes.begin()));
    std::printf(
        "  real exact denoise S%u x3: CUDA load+prep/run %.3f/%.3f ms, Vulkan %.3f/%.3f ms, boundaries %016llx/%016llx/%016llx, persistent/scratch/peak %.2f/%.2f/%.2f MiB\n",
        header.sequence, cuda_load_prepare_ms, cuda_run_ms,
        vk_load_prepare_ms, vk_run_ms,
        static_cast<unsigned long long>(boundary_hashes[0]),
        static_cast<unsigned long long>(boundary_hashes[1]),
        static_cast<unsigned long long>(boundary_hashes[2]),
        double(vk_denoiser.persistent_bytes()) / 1048576.0,
        double(vk_denoiser.scratch_bytes()) / 1048576.0,
        double(vk_denoiser.peak_device_bytes()) / 1048576.0);
    vk_denoiser.unload();
  }

  const char* captured_vertical = std::getenv("VIDFAB_DIT_VERTICAL_REAL");
  const char* qwen_vertical = std::getenv("VIDFAB_QWEN_VERTICAL_REAL");
  if ((captured_vertical && captured_vertical[0] == '1') ||
      (qwen_vertical && qwen_vertical[0] == '1')) {
    const bool normal_prompt = qwen_vertical && qwen_vertical[0] == '1';
    const std::filesystem::path video_vae_path =
        "weights/vae/minimax_h3_video_vae_fp16.safetensors";
    const std::filesystem::path audio_vae_path =
        "weights/vae/minimax_h3_audio_vae_fp32.safetensors";
    CHECK(std::filesystem::exists(video_vae_path) &&
          std::filesystem::exists(audio_vae_path));
    const std::string unique = std::to_string(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const std::filesystem::path temp = std::filesystem::temp_directory_path();
    const std::filesystem::path prompt_path =
        temp / ("vidfab-g7d-prompt-" + unique + ".safetensors");
    const std::filesystem::path init_path =
        temp / ("vidfab-g7d-init-" + unique + ".safetensors");
    const std::filesystem::path cuda_out =
        temp / ("vidfab-g7d-cuda-" + unique + ".raw");
    const std::filesystem::path vulkan_out =
        temp / ("vidfab-g7d-vulkan-" + unique + ".raw");
    write_safetensors(prompt_path.string(),
                      {{"prompt_embedding",
                        {static_cast<int64_t>(header.text_rows),
                         static_cast<int64_t>(header.text_dim)}, prompt}});
    write_safetensors(init_path.string(),
                      {{"video_rows",
                        {static_cast<int64_t>(header.video_rows),
                         static_cast<int64_t>(header.video_dim)}, video},
                       {"audio_rows",
                        {static_cast<int64_t>(header.audio_rows),
                         static_cast<int64_t>(header.audio_dim)}, audio}});
    GenerateRequest request;
    request.canvas_width = 256;
    request.canvas_height = 256;
    request.num_frames = 22;
    request.num_inference_steps = 4;
    request.seed = 424242;
    if (normal_prompt) {
      request.prompt = "A copper airship glides above a snowy forest at sunrise.";
      request.text_encoder_path =
          "weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors";
      request.tokenizer_path = "ref/text_encoder/tokenizer.json";
    }
    request.transformer_path = checkpoint_path.string();
    request.video_vae_path = video_vae_path.string();
    request.audio_vae_path = audio_vae_path.string();
    request.raw_output = true;
    const GeneratePlan vertical_plan = resolve_plan(request);
    CHECK(vertical_plan.layout.num_video_rows ==
              static_cast<int>(header.video_rows) &&
          vertical_plan.layout.num_audio_rows ==
              static_cast<int>(header.audio_rows) &&
          vertical_plan.num_model_evaluations() == 3);
    if (normal_prompt) {
      // A conditioner cache hit is allowed only within one explicit execution
      // authority. Cancel at the next stage so this exercises the public
      // run_generate cache without loading the transformer or either VAE.
      auto stop_after_conditioning = +[](RunStage stage, int, int, void*) {
        return stage != RunStage::kTransformerLoad;
      };
      GenerateRequest cache_request = request;
      cache_request.prompt += " cache-authority-" + unique;
      const GeneratePlan cache_plan = resolve_plan(cache_request);
      auto conditioning_only = [&](DeviceBackend backend,
                                   AttentionMode arithmetic,
                                   bool release,
                                   bool captured = false) {
        RunOptions options;
        options.inference_backend = backend;
        options.attention_mode = arithmetic;
        options.reuse_models = true;
        options.release_reused_models = release;
        if (captured) options.prompt_embedding_path = prompt_path.string();
        options.verbose = false;
        options.on_progress = stop_after_conditioning;
        const RunResult result = run_generate(cache_request, cache_plan, options);
        CHECK(result.cancelled && !result.ok);
        return result.conditioner_executed;
      };
      CHECK(!conditioning_only(DeviceBackend::kVulkan,
                               AttentionMode::kExact, false, true));
      CHECK(conditioning_only(DeviceBackend::kVulkan,
                              AttentionMode::kExact, false));
      CHECK(!conditioning_only(DeviceBackend::kVulkan,
                               AttentionMode::kExact, true));

      CHECK(conditioning_only(DeviceBackend::kCuda,
                              AttentionMode::kFlash2, false));
      CHECK(conditioning_only(DeviceBackend::kVulkan,
                              AttentionMode::kExact, false));
      CHECK(!conditioning_only(DeviceBackend::kVulkan,
                               AttentionMode::kExact, true));

      CHECK(conditioning_only(DeviceBackend::kCuda,
                              AttentionMode::kExact, false));
      CHECK(conditioning_only(DeviceBackend::kVulkan,
                              AttentionMode::kExact, false));
      CHECK(!conditioning_only(DeviceBackend::kVulkan,
                               AttentionMode::kExact, true));
    }
    struct CapturedSamples {
      PixelBuffer video;
      std::vector<float> audio;
      int channels = 0;
      int frames = 0;
      int height = 0;
      int width = 0;
      int audio_channels = 0;
      int sample_rate = 0;
    } cuda_samples, vulkan_samples;
    auto capture_samples = +[](RunSamples& samples, void* userdata) {
      auto* captured = static_cast<CapturedSamples*>(userdata);
      captured->channels = samples.channels;
      captured->frames = samples.frames;
      captured->height = samples.height;
      captured->width = samples.width;
      captured->audio_channels = samples.audio_channels;
      captured->sample_rate = samples.audio_sample_rate;
      if (samples.video) captured->video = *samples.video;
      if (samples.audio) captured->audio = *samples.audio;
      return false;
    };
    auto run_vertical = [&](DeviceBackend backend,
                            const std::filesystem::path& output,
                            CapturedSamples& samples) {
      request.out_path = output.string();
      RunOptions options;
      options.inference_backend = backend;
      options.attention_mode = AttentionMode::kExact;
      if (!normal_prompt) options.prompt_embedding_path = prompt_path.string();
      options.init_latents_path = init_path.string();
      options.verbose = false;
      options.on_samples = capture_samples;
      options.hook_userdata = &samples;
      const auto begin = std::chrono::steady_clock::now();
      const RunResult result = run_generate(request, vertical_plan, options);
      const double elapsed = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - begin).count();
      CHECK_MSG(result.ok, "real vertical %s failed: %s",
                backend == DeviceBackend::kCuda ? "CUDA" : "Vulkan",
                result.message.c_str());
      CHECK(result.conditioner_executed == normal_prompt);
      CHECK(result.steps_computed == 3 && result.steps_skipped == 0 &&
            result.outputs.size() == 2u);
      return std::pair<RunResult, double>{result, elapsed};
    };
    const auto cuda_vertical =
        run_vertical(DeviceBackend::kCuda, cuda_out, cuda_samples);
    const auto vulkan_vertical =
        run_vertical(DeviceBackend::kVulkan, vulkan_out, vulkan_samples);
    CHECK(cuda_samples.channels == vulkan_samples.channels &&
          cuda_samples.frames == vulkan_samples.frames &&
          cuda_samples.height == vulkan_samples.height &&
          cuda_samples.width == vulkan_samples.width &&
          cuda_samples.audio_channels == vulkan_samples.audio_channels &&
          cuda_samples.sample_rate == vulkan_samples.sample_rate);
    CHECK(cuda_samples.video == vulkan_samples.video);
    CHECK(cuda_samples.audio == vulkan_samples.audio);
    auto read_file = [](const std::string& path) {
      std::ifstream input(path, std::ios::binary | std::ios::ate);
      if (!input) throw std::runtime_error("cannot open vertical output " + path);
      const std::streamsize bytes = input.tellg();
      input.seekg(0);
      std::vector<uint8_t> result(static_cast<size_t>(bytes));
      if (!input.read(reinterpret_cast<char*>(result.data()), bytes))
        throw std::runtime_error("cannot read vertical output " + path);
      return result;
    };
    const std::vector<uint8_t> cuda_y4m =
        read_file(cuda_vertical.first.outputs[0]);
    const std::vector<uint8_t> cuda_wav =
        read_file(cuda_vertical.first.outputs[1]);
    const std::vector<uint8_t> vulkan_y4m =
        read_file(vulkan_vertical.first.outputs[0]);
    const std::vector<uint8_t> vulkan_wav =
        read_file(vulkan_vertical.first.outputs[1]);
    CHECK(cuda_y4m == vulkan_y4m && cuda_wav == vulkan_wav);
    const uint64_t pixel_hash = fnv_bytes(
        cuda_samples.video.data(), cuda_samples.video.size() * sizeof(float));
    const uint64_t pcm_hash = fnv_bytes(
        cuda_samples.audio.data(), cuda_samples.audio.size() * sizeof(float));
    const uint64_t y4m_hash = fnv_bytes(cuda_y4m.data(), cuda_y4m.size());
    const uint64_t wav_hash = fnv_bytes(cuda_wav.data(), cuda_wav.size());
    if (!normal_prompt) {
      CHECK(pixel_hash == 0x714a67162495817eull);
      CHECK(pcm_hash == 0x671f1519e5d0cea1ull);
      CHECK(y4m_hash == 0xbb480c4fd04ac34aull);
      CHECK(wav_hash == 0xa447eb6d02620637ull);
    } else {
      CHECK(pixel_hash == 0x52f6148fa46959f8ull);
      CHECK(pcm_hash == 0xb2e09a49fc952e5eull);
      CHECK(y4m_hash == 0x2f595da467a8ac60ull);
      CHECK(wav_hash == 0xe0d84106a3018c29ull);
    }
    std::printf(
        "  real exact %s vertical S%u x3: CUDA/Vulkan %.3f/%.3f ms, pixels/pcm/y4m/wav %016llx/%016llx/%016llx/%016llx\n",
        normal_prompt ? "normal-prompt" : "captured-prompt",
        header.sequence, cuda_vertical.second, vulkan_vertical.second,
        static_cast<unsigned long long>(pixel_hash),
        static_cast<unsigned long long>(pcm_hash),
        static_cast<unsigned long long>(y4m_hash),
        static_cast<unsigned long long>(wav_hash));
    std::error_code ignored;
    for (const std::filesystem::path& path : {
             prompt_path, init_path,
             std::filesystem::path(cuda_vertical.first.outputs[0]),
             std::filesystem::path(cuda_vertical.first.outputs[1]),
             std::filesystem::path(vulkan_vertical.first.outputs[0]),
             std::filesystem::path(vulkan_vertical.first.outputs[1])})
      std::filesystem::remove(path, ignored);
  }

  if (const char* production = std::getenv("VIDFAB_DIT_TRANSFORMER_PRODUCTION");
      production && production[0] == '1') {
    constexpr uint32_t prod_sequence = 9864;
    constexpr uint32_t prod_audio_rows = 74;
    constexpr uint32_t prod_video_rows = prod_sequence - 4 - prod_audio_rows;
    ExactH3TransformerConfig prod_config = config;
    prod_config.main.block.sequence = prod_sequence;
    prod_config.video_rows = prod_video_rows;
    prod_config.audio_rows = prod_audio_rows;
    ExactH3Transformer prod = ExactH3Transformer::create(vk, prod_config);
    const auto prod_load_begin = std::chrono::steady_clock::now();
    prod.load(checkpoint);
    const double prod_load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - prod_load_begin).count();
    DeviceTensor prod_prompt = tensor2(4, 5120);
    DeviceTensor prod_video = tensor2(prod_video_rows, 96);
    DeviceTensor prod_audio = tensor2(prod_audio_rows, 32);
    DeviceTensor prod_selectors = tensor1(prod_sequence, ScalarType::kInt32);
    DeviceTensor prod_code = tensor2(1, 8);
    DeviceTensor prod_cosine = tensor2(prod_sequence, 96);
    DeviceTensor prod_sine = tensor2(prod_sequence, 96);
    DeviceTensor prod_video_ts = tensor1(prod_video_rows, ScalarType::kInt32);
    DeviceTensor prod_audio_ts = tensor1(prod_audio_rows, ScalarType::kInt32);
    DeviceTensor prod_video_out = tensor2(prod_video_rows, 96);
    DeviceTensor prod_audio_out = tensor2(prod_audio_rows, 32);
    std::vector<float> prod_video_values(size_t(prod_video_rows) * 96);
    std::vector<float> prod_audio_values(size_t(prod_audio_rows) * 32);
    for (size_t i = 0; i < prod_video_values.size(); ++i)
      prod_video_values[i] = float(int(i % 251) - 125) / 128.0f;
    for (size_t i = 0; i < prod_audio_values.size(); ++i)
      prod_audio_values[i] = float(int(i % 127) - 63) / 64.0f;
    std::vector<int32_t> prod_selector_values(prod_sequence, 0);
    std::vector<int32_t> prod_video_ts_values(prod_video_rows, 0);
    std::vector<int32_t> prod_audio_ts_values(prod_audio_rows, 0);
    std::vector<float> prod_cos_values(size_t(prod_sequence) * 96, 1.0f);
    std::vector<float> prod_sin_values(prod_cos_values.size(), 0.0f);
    vk.upload(prod_prompt, prompt.data(), prompt.size());
    vk.upload(prod_video, prod_video_values.data(), prod_video_values.size());
    vk.upload(prod_audio, prod_audio_values.data(), prod_audio_values.size());
    vk.upload_bytes(prod_selectors, prod_selector_values.data(),
                    prod_selector_values.size() * 4);
    vk.upload(prod_code, code.data(), code.size());
    vk.upload(prod_cosine, prod_cos_values.data(), prod_cos_values.size());
    vk.upload(prod_sine, prod_sin_values.data(), prod_sin_values.size());
    vk.upload_bytes(prod_video_ts, prod_video_ts_values.data(),
                    prod_video_ts_values.size() * 4);
    vk.upload_bytes(prod_audio_ts, prod_audio_ts_values.data(),
                    prod_audio_ts_values.size() * 4);
    prod.prepare_text(prod_prompt);
    auto run_prod = [&] {
      const auto begin = std::chrono::steady_clock::now();
      TensorBatch batch = vk.begin_batch();
      prod.record_forward(batch, prod_video, prod_audio, prod_selectors,
          prod_code, prod_cosine, prod_sine, prod_video_ts, prod_audio_ts,
          prod_video_out, prod_audio_out);
      CHECK(batch.remaining_operator_capacity() == 581u);
      batch.submit().wait();
      return std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - begin).count();
    };
    const double prod_first_ms = run_prod();
    std::vector<float> prod_video_result(prod_video_values.size());
    std::vector<float> prod_audio_result(prod_audio_values.size());
    vk.download(prod_video_out, prod_video_result.data(), prod_video_result.size());
    vk.download(prod_audio_out, prod_audio_result.data(), prod_audio_result.size());
    uint64_t prod_hash = 1469598103934665603ull;
    auto append_hash = [&](const void* data, size_t bytes) {
      const auto* p = static_cast<const uint8_t*>(data);
      for (size_t i = 0; i < bytes; ++i) {
        prod_hash ^= p[i]; prod_hash *= 1099511628211ull;
      }
    };
    append_hash(prod_video_result.data(), prod_video_result.size() * 4);
    append_hash(prod_audio_result.data(), prod_audio_result.size() * 4);
    CHECK(prod_hash == 0xa998bb5ff7a03383ull);
    const uint64_t prod_used = vk.pooled_used_bytes();
    const uint64_t prod_reserved = vk.reserved_bytes();
    const uint64_t prod_descriptors = vk.descriptor_set_allocations();
    const double prod_repeat_ms = run_prod();
    std::vector<float> repeated_video(prod_video_values.size());
    std::vector<float> repeated_audio(prod_audio_values.size());
    vk.download(prod_video_out, repeated_video.data(), repeated_video.size());
    vk.download(prod_audio_out, repeated_audio.data(), repeated_audio.size());
    CHECK(repeated_video == prod_video_result && repeated_audio == prod_audio_result);
    CHECK(vk.pooled_used_bytes() == prod_used &&
          vk.reserved_bytes() == prod_reserved &&
          vk.descriptor_set_allocations() == prod_descriptors);
    std::printf(
        "  production H3 transformer S9864: load %.3f ms, Vulkan first/repeat %.3f/%.3f ms, final %016llx, persistent/scratch/peak %.2f/%.2f/%.2f MiB, pool %.2f/%.2f MiB, descriptors %llu\n",
        prod_load_ms, prod_first_ms, prod_repeat_ms,
        static_cast<unsigned long long>(prod_hash),
        double(prod.persistent_bytes()) / 1048576.0,
        double(prod.scratch_bytes()) / 1048576.0,
        double(prod.peak_device_bytes()) / 1048576.0,
        double(vk.pooled_used_bytes()) / 1048576.0,
        double(vk.reserved_bytes()) / 1048576.0,
        static_cast<unsigned long long>(vk.descriptor_set_allocations()));
  }
}

VIDFAB_TEST(cuda_vulkan_vae_pointwise_real_timing) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!std::getenv("VIDFAB_VAE_POINTWISE_BENCH")) return;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;

  constexpr int rows = 1797;
  constexpr int columns = 2048;
  constexpr int inner = 8192;
  constexpr int channels = 24;
  constexpr int voxels = 1792;
  constexpr int repeats = 8;
  const size_t matrix_count = size_t(rows) * columns;
  const size_t swiglu_input_count = size_t(rows) * 2 * inner;
  const size_t swiglu_output_count = size_t(rows) * inner;
  const size_t latent_count = size_t(channels) * voxels;
  std::vector<float> x(matrix_count), y(matrix_count), bias(columns),
      scale(columns), swiglu_input(swiglu_input_count),
      swiglu_bias(2 * inner), latent(latent_count), mean(channels),
      std_dev(channels);
  for (size_t index = 0; index < matrix_count; ++index) {
    x[index] = float(int(index % 127) - 63) / 128.0f;
    y[index] = float(int(index % 109) - 54) / 128.0f;
  }
  for (int index = 0; index < columns; ++index) {
    bias[index] = float((index % 31) - 15) / 256.0f;
    scale[index] = 0.01f + float(index % 13) / 1024.0f;
  }
  for (size_t index = 0; index < swiglu_input_count; ++index)
    swiglu_input[index] = float(int(index % 251) - 125) / 64.0f;
  for (size_t index = 0; index < swiglu_bias.size(); ++index)
    swiglu_bias[index] = float(int(index % 37) - 18) / 256.0f;
  for (size_t index = 0; index < latent_count; ++index)
    latent[index] = float(int(index % 97) - 48) / 32.0f;
  for (int index = 0; index < channels; ++index) {
    mean[index] = float(index - 12) / 64.0f;
    std_dev[index] = 0.5f + float(index % 7) / 16.0f;
  }

  cuda::DeviceBuffer<float> cx(matrix_count), cy(matrix_count), cbias(columns),
      cscale(columns), cinput(swiglu_input_count), csbias(2 * inner),
      clegacy(swiglu_output_count), cexact(swiglu_output_count),
      clatent(latent_count), cmean(channels), cstd(channels),
      cout(latent_count);
  cx.copy_from_host(x.data(), x.size());
  cy.copy_from_host(y.data(), y.size());
  cbias.copy_from_host(bias.data(), bias.size());
  cscale.copy_from_host(scale.data(), scale.size());
  cinput.copy_from_host(swiglu_input.data(), swiglu_input.size());
  csbias.copy_from_host(swiglu_bias.data(), swiglu_bias.size());
  clatent.copy_from_host(latent.data(), latent.size());
  cmean.copy_from_host(mean.data(), mean.size());
  cstd.copy_from_host(std_dev.data(), std_dev.size());

  auto time_cuda = [&](auto&& launch) {
    launch();
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t begin{}, end{};
    VIDFAB_CUDA_CHECK(cudaEventCreate(&begin));
    VIDFAB_CUDA_CHECK(cudaEventCreate(&end));
    VIDFAB_CUDA_CHECK(cudaEventRecord(begin));
    for (int repeat = 0; repeat < repeats; ++repeat) launch();
    VIDFAB_CUDA_CHECK(cudaEventRecord(end));
    VIDFAB_CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds = 0.0f;
    VIDFAB_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return milliseconds / repeats;
  };
  const dim3 swiglu_grid((inner + 255) / 256, rows);
  const float legacy_swiglu_ms = time_cuda([&] {
    legacy_vae_swiglu_probe<<<swiglu_grid, 256>>>(
        cinput.get(), csbias.get(), clegacy.get(), inner);
    VIDFAB_CUDA_CHECK(cudaGetLastError());
  });
  const float exact_swiglu_ms = time_cuda([&] {
    cuda::launch_swiglu(cinput.get(), csbias.get(), cexact.get(), rows, inner,
                        nullptr);
  });
  cx.copy_from_host(x.data(), x.size());
  const float residual_ms = time_cuda([&] {
    cuda::launch_layerscale_residual(cx.get(), cy.get(), cbias.get(),
                                     cscale.get(), rows, columns, nullptr);
  });
  const float denorm_ms = time_cuda([&] {
    cuda::launch_latent_denorm(clatent.get(), cmean.get(), cstd.get(),
                               cout.get(), channels, voxels, nullptr);
  });

  std::vector<float> legacy(swiglu_output_count), exact(swiglu_output_count);
  clegacy.copy_to_host(legacy.data(), legacy.size());
  cexact.copy_to_host(exact.data(), exact.size());
  size_t legacy_differences = 0;
  float legacy_max_absolute = 0.0f;
  for (size_t index = 0; index < exact.size(); ++index) {
    if (std::memcmp(&legacy[index], &exact[index], sizeof(float)) != 0)
      ++legacy_differences;
    legacy_max_absolute = std::max(
        legacy_max_absolute, std::abs(legacy[index] - exact[index]));
  }

  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  CHECK(!physical.empty());
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  CHECK(vk.exact_vae_pointwise());
  const uint64_t matrix_shape[] = {rows, columns};
  const uint64_t swiglu_input_shape[] = {rows, 2 * inner};
  const uint64_t swiglu_output_shape[] = {rows, inner};
  const uint64_t latent_shape[] = {channels, voxels};
  const uint64_t column_shape = columns, swiglu_bias_shape = 2 * inner,
                 channel_shape = channels;
  DeviceTensor vx = vk.allocate(TensorLayout::contiguous(matrix_shape, 2));
  DeviceTensor vy = vk.allocate(TensorLayout::contiguous(matrix_shape, 2));
  DeviceTensor vb = vk.allocate(TensorLayout::contiguous(&column_shape, 1));
  DeviceTensor vs = vk.allocate(TensorLayout::contiguous(&column_shape, 1));
  DeviceTensor vi = vk.allocate(TensorLayout::contiguous(swiglu_input_shape, 2));
  DeviceTensor vsb = vk.allocate(TensorLayout::contiguous(&swiglu_bias_shape, 1));
  DeviceTensor vo = vk.allocate(TensorLayout::contiguous(swiglu_output_shape, 2));
  DeviceTensor vl = vk.allocate(TensorLayout::contiguous(latent_shape, 2));
  DeviceTensor vm = vk.allocate(TensorLayout::contiguous(&channel_shape, 1));
  DeviceTensor vsd = vk.allocate(TensorLayout::contiguous(&channel_shape, 1));
  DeviceTensor vlo = vk.allocate(TensorLayout::contiguous(latent_shape, 2));
  vk.upload(vx, x.data(), x.size());
  vk.upload(vy, y.data(), y.size());
  vk.upload(vb, bias.data(), bias.size());
  vk.upload(vs, scale.data(), scale.size());
  vk.upload(vi, swiglu_input.data(), swiglu_input.size());
  vk.upload(vsb, swiglu_bias.data(), swiglu_bias.size());
  vk.upload(vl, latent.data(), latent.size());
  vk.upload(vm, mean.data(), mean.size());
  vk.upload(vsd, std_dev.data(), std_dev.size());
  auto time_vulkan = [&](auto&& record) {
    { TensorBatch warm = vk.begin_batch(); record(warm); warm.submit().wait(); }
    const auto begin = std::chrono::steady_clock::now();
    TensorBatch batch = vk.begin_batch();
    for (int repeat = 0; repeat < repeats; ++repeat) record(batch);
    batch.submit().wait();
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - begin).count() /
           repeats;
  };
  const double vk_swiglu_ms = time_vulkan(
      [&](TensorBatch& batch) { batch.swiglu_bias_f32(vi, vsb, vo); });
  const double vk_residual_ms = time_vulkan([&](TensorBatch& batch) {
    batch.layer_scale_residual_f32(vx, vy, vb, vs);
  });
  const double vk_denorm_ms = time_vulkan([&](TensorBatch& batch) {
    batch.latent_denorm_f32(vl, vm, vsd, vlo);
  });
  std::vector<float> vulkan_swiglu(swiglu_output_count);
  vk.download(vo, vulkan_swiglu.data(), vulkan_swiglu.size());
  CHECK(std::memcmp(vulkan_swiglu.data(), exact.data(),
                    exact.size() * sizeof(float)) == 0);
  const double direct_mib =
      double((matrix_count * 2 + columns * 2 + swiglu_input_count +
              swiglu_output_count + swiglu_bias.size() + latent_count * 2 +
              channels * 2) *
             sizeof(float)) /
      1048576.0;
  std::printf(
      "  VAE pointwise R%d C%d I%d: SwiGLU old CUDA %.3f ms, exact CUDA %.3f ms, Vulkan %.3f ms (x36 %.1f/%.1f/%.1f ms); residual CUDA %.3f/Vulkan %.3f ms; denorm CUDA %.3f/Vulkan %.3f ms; old/exact drift %zu/%zu maxabs %.7g; direct %.1f MiB, scratch 0, reserved %.1f MiB\n",
      rows, columns, inner, legacy_swiglu_ms, exact_swiglu_ms, vk_swiglu_ms,
      legacy_swiglu_ms * 36.0f, exact_swiglu_ms * 36.0f,
      vk_swiglu_ms * 36.0, residual_ms, vk_residual_ms, denorm_ms,
      vk_denorm_ms, legacy_differences, exact.size(), legacy_max_absolute,
      direct_mib, double(vk.reserved_bytes()) / 1048576.0);
}

VIDFAB_TEST(cuda_vulkan_tensor_exact_vae_norms) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  if (!vk.exact_normalization()) {
    bool rejected = false;
    try { vk.require_exact_normalization(); }
    catch (const std::runtime_error&) { rejected = true; }
    CHECK(rejected);
    return;
  }
  vk.require_exact_normalization();

  auto run_case = [&](int rows, int dim, float epsilon, int pattern) {
    const size_t count = static_cast<size_t>(rows) * dim;
    std::vector<float> input(count), weight(dim), bias(dim);
    for (size_t i = 0; i < input.size(); ++i)
      input[i] = static_cast<float>(static_cast<int>((i * 17) % 251) - 125) / 32.0f;
    for (int i = 0; i < dim; ++i) {
      weight[i] = 0.5f + static_cast<float>(i % 29) / 32.0f;
      bias[i] = static_cast<float>((i % 17) - 8) / 64.0f;
    }
    if (pattern == 1) {
      // Constant/zero-variance row and signed-zero values exercise degenerate
      // LayerNorm and sign preservation without leaving the advertised domain.
      for (int col = 0; col < dim; ++col) input[col] = 2.0f;
      for (int col = 0; col < dim; ++col)
        input[dim + col] = (col & 1) ? -0.0f : 0.0f;
    } else if (pattern == 2) {
      // Large values whose squares and reduction totals remain finite.
      for (size_t i = 0; i < input.size(); ++i)
        input[i] = (i & 1) ? -1.0e16f : 1.0e16f;
    } else if (pattern == 3) {
      std::fill(input.begin(), input.end(), 0.0f);
    } else if (pattern == 4) {
      for (size_t i = 0; i < input.size(); ++i) {
        const uint32_t exponent = 97u + static_cast<uint32_t>(i % 61u);
        const uint32_t mantissa = static_cast<uint32_t>(i * 2654435761u) & 0x007fffffu;
        const uint32_t value_bits = exponent << 23u | mantissa;
        std::memcpy(&input[i], &value_bits, sizeof(value_bits));
      }
    }

    cuda::DeviceBuffer<float> c_input(count), c_weight(dim), c_bias(dim),
        c_rms(count), c_layer(count), c_rms_repeat(count),
        c_layer_repeat(count);
    c_input.copy_from_host(input.data(), input.size());
    c_weight.copy_from_host(weight.data(), weight.size());
    c_bias.copy_from_host(bias.data(), bias.size());
    cuda::launch_rmsnorm(c_input.get(), c_weight.get(), c_rms.get(), rows, dim,
                         epsilon, nullptr);
    cuda::launch_layernorm(c_input.get(), c_weight.get(), c_bias.get(), c_layer.get(),
                           rows, dim, epsilon, nullptr);
    cuda::launch_rmsnorm(c_input.get(), c_weight.get(), c_rms_repeat.get(), rows,
                         dim, epsilon, nullptr);
    cuda::launch_layernorm(c_input.get(), c_weight.get(), c_bias.get(),
                           c_layer_repeat.get(), rows, dim, epsilon, nullptr);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> rms_once(count), rms_twice(count), layer_once(count),
        layer_twice(count);
    c_rms.copy_to_host(rms_once.data(), count);
    c_rms_repeat.copy_to_host(rms_twice.data(), count);
    c_layer.copy_to_host(layer_once.data(), count);
    c_layer_repeat.copy_to_host(layer_twice.data(), count);
    CHECK(std::memcmp(rms_once.data(), rms_twice.data(), count * sizeof(float)) == 0);
    CHECK(std::memcmp(layer_once.data(), layer_twice.data(), count * sizeof(float)) == 0);

    const uint64_t shape_extents[] = {static_cast<uint64_t>(rows),
                                      static_cast<uint64_t>(dim)};
    const uint64_t feature_extent = static_cast<uint64_t>(dim);
    DeviceTensor v_input = vk.allocate(TensorLayout::contiguous(shape_extents, 2));
    DeviceTensor v_weight = vk.allocate(TensorLayout::contiguous(&feature_extent, 1));
    DeviceTensor v_bias = vk.allocate(TensorLayout::contiguous(&feature_extent, 1));
    DeviceTensor v_rms = vk.allocate(TensorLayout::contiguous(shape_extents, 2));
    DeviceTensor v_layer = vk.allocate(TensorLayout::contiguous(shape_extents, 2));
    vk.upload(v_input, input.data(), input.size());
    vk.upload(v_weight, weight.data(), weight.size());
    vk.upload(v_bias, bias.data(), bias.size());
    TensorBatch batch = vk.begin_batch();
    batch.rms_norm(v_input, v_weight, v_rms, epsilon);
    batch.layer_norm(v_input, v_weight, v_bias, v_layer, epsilon);
    batch.submit().wait();

    auto compare = [&](auto& cuda_buffer, DeviceTensor& vulkan_tensor,
                       const char* label) {
      std::vector<float> cuda_host(count), vulkan_host(count);
      cuda_buffer.copy_to_host(cuda_host.data(), cuda_host.size());
      vk.download(vulkan_tensor, vulkan_host.data(), vulkan_host.size());
      size_t mismatch = count;
      for (size_t i = 0; i < count; ++i) {
        if (std::memcmp(&cuda_host[i], &vulkan_host[i], sizeof(float)) != 0) {
          mismatch = i;
          break;
        }
      }
      uint32_t cuda_bits = 0, vulkan_bits = 0;
      if (mismatch != count) {
        std::memcpy(&cuda_bits, &cuda_host[mismatch], sizeof(cuda_bits));
        std::memcpy(&vulkan_bits, &vulkan_host[mismatch], sizeof(vulkan_bits));
      }
      CHECK_MSG(mismatch == count,
                "CUDA/Vulkan %s %dx%d mismatch at %zu: %08x != %08x", label,
                rows, dim, mismatch, cuda_bits, vulkan_bits);
    };
    compare(c_rms, v_rms, "VAE RMSNorm");
    compare(c_layer, v_layer, "VAE LayerNorm");
  };

  run_case(3, 513, 1.0e-6f, 0);   // odd scalar and workgroup tail
  run_case(2, 128, 1.0e-5f, 1);   // test-model width, signed zero/variance zero
  run_case(2, 2048, 1.0e-6f, 2);  // shipped video-VAE width
  run_case(2, 7, 1.0e-6f, 1);     // dimension below a warp
  run_case(1, 9, std::numeric_limits<float>::min(), 3);
  run_case(4099, 1, 1.0e-6f, 4);  // dense exponent/mantissa and dispatch tail

  // Host-known values outside the exact domain fail before recording and do
  // not poison an otherwise valid batch.
  const uint64_t extents[] = {2, 7};
  const uint64_t feature = 7;
  DeviceTensor input = vk.allocate(TensorLayout::contiguous(extents, 2));
  DeviceTensor weight = vk.allocate(TensorLayout::contiguous(&feature, 1));
  DeviceTensor bias = vk.allocate(TensorLayout::contiguous(&feature, 1));
  DeviceTensor rms_a = vk.allocate(TensorLayout::contiguous(extents, 2));
  DeviceTensor rms_b = vk.allocate(TensorLayout::contiguous(extents, 2));
  DeviceTensor layer_a = vk.allocate(TensorLayout::contiguous(extents, 2));
  DeviceTensor layer_b = vk.allocate(TensorLayout::contiguous(extents, 2));
  std::vector<float> values(14, 1.0f), affine(7, 1.0f), offsets(7, 0.25f);
  vk.upload(input, values.data(), values.size());
  vk.upload(weight, affine.data(), affine.size());
  vk.upload(bias, offsets.data(), offsets.size());
  TensorBatch recoverable = vk.begin_batch();
  bool subnormal_epsilon_rejected = false;
  try {
    recoverable.rms_norm(input, weight, rms_a,
                         std::numeric_limits<float>::denorm_min());
  } catch (const std::invalid_argument&) {
    subnormal_epsilon_rejected = true;
  }
  CHECK(subnormal_epsilon_rejected);
  recoverable.rms_norm(input, weight, rms_a, 1.0e-6f);
  recoverable.submit().wait();

  // Warm both flight slots with the same two-pipeline sequence. Two jobs are
  // submitted before either token is waited, then a third begin exercises
  // bounded oldest-slot backpressure without queue/device idle.
  auto submit_pair = [&](DeviceTensor& rms, DeviceTensor& layer) {
    TensorBatch batch = vk.begin_batch();
    batch.rms_norm(input, weight, rms, 1.0e-6f);
    batch.layer_norm(input, weight, bias, layer, 1.0e-6f);
    return batch.submit();
  };
  Submission first = submit_pair(rms_a, layer_a);
  Submission second = submit_pair(rms_b, layer_b);
  CHECK(first.value() != 0 && second.value() > first.value());
  Submission third = submit_pair(rms_a, layer_a);
  third.wait();
  second.wait();
  const uint64_t warm_reserved = vk.reserved_bytes();
  const uint64_t warm_descriptors = vk.descriptor_set_allocations();
  for (int repeat = 0; repeat < 8; ++repeat) {
    Submission a = submit_pair(rms_a, layer_a);
    Submission b = submit_pair(rms_b, layer_b);
    a.wait();
    b.wait();
    CHECK(vk.reserved_bytes() == warm_reserved);
    CHECK(vk.descriptor_set_allocations() == warm_descriptors);
  }
}

VIDFAB_TEST(cuda_vulkan_tensor_exact_shared_norms) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options; options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  if (!vk.exact_normalization()) return;
  auto exact_bf16 = [](float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
  };

  auto compare_bf16 = [&](const std::vector<uint16_t>& expected,
                          const std::vector<uint16_t>& actual, const char* label,
                          int rows, int dim) {
    size_t mismatch = expected.size();
    for (size_t i = 0; i < expected.size(); ++i) {
      if (expected[i] != actual[i]) { mismatch = i; break; }
    }
    CHECK_MSG(mismatch == expected.size(),
              "CUDA/Vulkan %s %dx%d mismatch at %zu: %04x != %04x", label,
              rows, dim, mismatch,
              mismatch == expected.size() ? 0 : expected[mismatch],
              mismatch == expected.size() ? 0 : actual[mismatch]);
  };

  auto make_bf16_data = [&](size_t count, int pattern) {
    std::vector<uint16_t> values(count);
    for (size_t i = 0; i < count; ++i) {
      float value = static_cast<float>(static_cast<int>((i * 17) % 61) - 30) / 16.0f;
      if (pattern == 1 && i < 16) value = (i & 1) ? -0.0f : 0.0f;
      if (pattern == 2) value = (i & 1) ? -1.0e16f : 1.0e16f;
      values[i] = exact_bf16(value);
    }
    return values;
  };

  auto run_rms = [&](int rows, int dim, int pattern, bool in_place) {
    const size_t count = static_cast<size_t>(rows) * dim;
    auto input = make_bf16_data(count, pattern);
    std::vector<uint16_t> weight(dim);
    for (int i = 0; i < dim; ++i)
      weight[i] = exact_bf16(0.5f + static_cast<float>(i % 8) / 8.0f);
    cuda::DeviceBuffer<uint16_t> c_input(count), c_weight(dim), c_output(count),
        c_repeat(count);
    c_input.copy_from_host(input.data(), count);
    c_weight.copy_from_host(weight.data(), dim);
    cuda::launch_rmsnorm(reinterpret_cast<const __nv_bfloat16*>(c_input.get()),
                         reinterpret_cast<const __nv_bfloat16*>(c_weight.get()),
                         reinterpret_cast<__nv_bfloat16*>(c_output.get()), rows, dim,
                         1.0e-5f, nullptr);
    cuda::launch_rmsnorm(reinterpret_cast<const __nv_bfloat16*>(c_input.get()),
                         reinterpret_cast<const __nv_bfloat16*>(c_weight.get()),
                         reinterpret_cast<__nv_bfloat16*>(c_repeat.get()), rows,
                         dim, 1.0e-5f, nullptr);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> expected(count), actual(count);
    c_output.copy_to_host(expected.data(), count);
    std::vector<uint16_t> repeated(count);
    c_repeat.copy_to_host(repeated.data(), count);
    CHECK(repeated == expected);
    const uint64_t extents[] = {static_cast<uint64_t>(rows), static_cast<uint64_t>(dim)};
    const uint64_t feature = dim;
    DeviceTensor v_input = vk.allocate(TensorLayout::contiguous(extents, 2),
                                       ScalarType::kBFloat16);
    DeviceTensor v_weight = vk.allocate(TensorLayout::contiguous(&feature, 1),
                                        ScalarType::kBFloat16);
    DeviceTensor v_output = in_place
                                ? DeviceTensor()
                                : vk.allocate(TensorLayout::contiguous(extents, 2),
                                              ScalarType::kBFloat16);
    vk.upload_bytes(v_input, input.data(), input.size() * sizeof(uint16_t));
    vk.upload_bytes(v_weight, weight.data(), weight.size() * sizeof(uint16_t));
    TensorBatch batch = vk.begin_batch();
    batch.rms_norm_bf16(v_input, v_weight, in_place ? v_input : v_output, 1.0e-5f);
    batch.submit().wait();
    DeviceTensor& result = in_place ? v_input : v_output;
    vk.download_bytes(result, actual.data(), actual.size() * sizeof(uint16_t));
    compare_bf16(expected, actual, "BF16 RMSNorm", rows, dim);
  };

  run_rms(9, 128, 1, true);   // head path, rows%8 and in-place
  run_rms(2, 5120, 0, false); // Qwen text width
  run_rms(3, 5376, 0, false); // DiT width and pack/workgroup tail
  run_rms(3, 31, 1, false);   // scalar narrow and odd packed rows
  run_rms(2, 513, 0, false);  // scalar block and odd packed rows

  auto run_layer = [&](int rows, int dim, bool constant_row, bool in_place = false) {
    const size_t count = static_cast<size_t>(rows) * dim;
    auto input = make_bf16_data(count, 0);
    if (constant_row) std::fill(input.begin(), input.begin() + dim, exact_bf16(2.0f));
    std::vector<uint16_t> weight(dim), bias(dim);
    for (int i = 0; i < dim; ++i) {
      weight[i] = exact_bf16(0.5f + static_cast<float>(i % 4) / 4.0f);
      bias[i] = exact_bf16(static_cast<float>((i % 9) - 4) / 16.0f);
    }
    cuda::DeviceBuffer<uint16_t> c_input(count), c_weight(dim), c_bias(dim),
        c_output(count), c_repeat(count);
    c_input.copy_from_host(input.data(), count);
    c_weight.copy_from_host(weight.data(), dim);
    c_bias.copy_from_host(bias.data(), dim);
    cuda::launch_layernorm_affine(
        reinterpret_cast<const __nv_bfloat16*>(c_input.get()),
        reinterpret_cast<const __nv_bfloat16*>(c_weight.get()),
        reinterpret_cast<const __nv_bfloat16*>(c_bias.get()),
        reinterpret_cast<__nv_bfloat16*>(c_output.get()), rows, dim, 1.0e-6f, nullptr);
    cuda::launch_layernorm_affine(
        reinterpret_cast<const __nv_bfloat16*>(c_input.get()),
        reinterpret_cast<const __nv_bfloat16*>(c_weight.get()),
        reinterpret_cast<const __nv_bfloat16*>(c_bias.get()),
        reinterpret_cast<__nv_bfloat16*>(c_repeat.get()), rows, dim, 1.0e-6f,
        nullptr);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> expected(count), actual(count);
    c_output.copy_to_host(expected.data(), count);
    std::vector<uint16_t> repeated(count);
    c_repeat.copy_to_host(repeated.data(), count);
    CHECK(repeated == expected);
    const uint64_t extents[] = {static_cast<uint64_t>(rows), static_cast<uint64_t>(dim)};
    const uint64_t feature = dim;
    DeviceTensor v_input = vk.allocate(TensorLayout::contiguous(extents, 2), ScalarType::kBFloat16);
    DeviceTensor v_weight = vk.allocate(TensorLayout::contiguous(&feature, 1), ScalarType::kBFloat16);
    DeviceTensor v_bias = vk.allocate(TensorLayout::contiguous(&feature, 1), ScalarType::kBFloat16);
    DeviceTensor v_output = in_place
                                ? DeviceTensor()
                                : vk.allocate(TensorLayout::contiguous(extents, 2),
                                              ScalarType::kBFloat16);
    vk.upload_bytes(v_input, input.data(), input.size() * sizeof(uint16_t));
    vk.upload_bytes(v_weight, weight.data(), weight.size() * sizeof(uint16_t));
    vk.upload_bytes(v_bias, bias.data(), bias.size() * sizeof(uint16_t));
    TensorBatch batch = vk.begin_batch();
    batch.layer_norm_bf16(v_input, v_weight, v_bias,
                          in_place ? v_input : v_output, 1.0e-6f);
    batch.submit().wait();
    DeviceTensor& result = in_place ? v_input : v_output;
    vk.download_bytes(result, actual.data(), actual.size() * sizeof(uint16_t));
    compare_bf16(expected, actual, "BF16 LayerNorm", rows, dim);
  };
  run_layer(2, 1152, true);
  run_layer(1, 4608, false);
  run_layer(3, 129, true, true);

  auto run_mod = [&](bool fp32, bool invalid_selectors = false,
                     bool contraction_fixture = false, int rows = 3,
                     int dim = 5376, bool in_place = false) {
    constexpr int mod_rows = 4;
    const size_t count = static_cast<size_t>(rows) * dim;
    auto bf_input = make_bf16_data(count, 0);
    std::vector<float> f_input(count);
    for (size_t i = 0; i < count; ++i) {
      uint32_t bits = static_cast<uint32_t>(bf_input[i]) << 16u;
      std::memcpy(&f_input[i], &bits, sizeof(bits));
    }
    std::vector<uint16_t> weight(dim);
    std::vector<float> scale(static_cast<size_t>(mod_rows) * dim);
    std::vector<float> shift(scale.size());
    std::vector<int32_t> valid_selectors(rows), invalid_selector_values(rows);
    for (int row = 0; row < rows; ++row) {
      valid_selectors[row] = (row * 3 + 1) % mod_rows;
      invalid_selector_values[row] = (row & 1) ? mod_rows + row : -1 - row;
    }
    const int32_t* selectors = invalid_selectors ? invalid_selector_values.data()
                                                 : valid_selectors.data();
    for (int i = 0; i < dim; ++i)
      weight[i] = exact_bf16(0.5f + static_cast<float>(i % 8) / 8.0f);
    for (size_t i = 0; i < scale.size(); ++i) {
      scale[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 32.0f;
      shift[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 64.0f;
    }
    if (contraction_fixture) {
      std::fill(f_input.begin(), f_input.begin() + dim, 0.5f);
      weight[0] = exact_bf16(1.5f);
      const uint32_t scale_bits = 0xbebf60eeu;
      const uint32_t shift_bits = 0x3714c343u;
      std::memcpy(&scale[static_cast<size_t>(selectors[0]) * dim], &scale_bits, 4);
      std::memcpy(&shift[static_cast<size_t>(selectors[0]) * dim], &shift_bits, 4);
    }
    const float epsilon = contraction_fixture ? 0.75f : 1.0e-5f;
    cuda::DeviceBuffer<uint16_t> c_bf_input(count), c_weight(dim),
        c_bf_output(count), c_bf_repeat(count);
    cuda::DeviceBuffer<float> c_f_input(count), c_scale(scale.size()), c_shift(shift.size()),
        c_f_output(count), c_f_repeat(count);
    cuda::DeviceBuffer<int32_t> c_selectors(rows);
    c_bf_input.copy_from_host(bf_input.data(), count);
    c_f_input.copy_from_host(f_input.data(), count);
    c_weight.copy_from_host(weight.data(), dim);
    c_scale.copy_from_host(scale.data(), scale.size());
    c_shift.copy_from_host(shift.data(), shift.size());
    c_selectors.copy_from_host(selectors, rows);
    if (!invalid_selectors && fp32) {
      cuda::launch_rmsnorm_modulate_f32(
          c_f_input.get(), reinterpret_cast<const __nv_bfloat16*>(c_weight.get()),
          c_scale.get(), c_shift.get(), c_selectors.get(), c_f_output.get(), rows, dim,
          epsilon, nullptr);
      cuda::launch_rmsnorm_modulate_f32(
          c_f_input.get(), reinterpret_cast<const __nv_bfloat16*>(c_weight.get()),
          c_scale.get(), c_shift.get(), c_selectors.get(), c_f_repeat.get(), rows,
          dim, epsilon, nullptr);
    } else if (!invalid_selectors) {
      cuda::launch_rmsnorm_modulate(
          reinterpret_cast<const __nv_bfloat16*>(c_bf_input.get()),
          reinterpret_cast<const __nv_bfloat16*>(c_weight.get()), c_scale.get(),
          c_shift.get(), c_selectors.get(),
          reinterpret_cast<__nv_bfloat16*>(c_bf_output.get()), rows, dim, 1.0e-5f,
          nullptr);
      cuda::launch_rmsnorm_modulate(
          reinterpret_cast<const __nv_bfloat16*>(c_bf_input.get()),
          reinterpret_cast<const __nv_bfloat16*>(c_weight.get()), c_scale.get(),
          c_shift.get(), c_selectors.get(),
          reinterpret_cast<__nv_bfloat16*>(c_bf_repeat.get()), rows, dim,
          1.0e-5f, nullptr);
    }
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    if (!invalid_selectors) {
      if (fp32) {
        std::vector<float> once(count), twice(count);
        c_f_output.copy_to_host(once.data(), count);
        c_f_repeat.copy_to_host(twice.data(), count);
        CHECK(std::memcmp(once.data(), twice.data(), count * sizeof(float)) == 0);
      } else {
        std::vector<uint16_t> once(count), twice(count);
        c_bf_output.copy_to_host(once.data(), count);
        c_bf_repeat.copy_to_host(twice.data(), count);
        CHECK(once == twice);
      }
    }
    const uint64_t extents[] = {static_cast<uint64_t>(rows),
                                static_cast<uint64_t>(dim)};
    const uint64_t feature = dim;
    const uint64_t mod_extents[] = {mod_rows, static_cast<uint64_t>(dim)};
    const uint64_t selector_extent = rows;
    DeviceTensor v_input = vk.allocate(TensorLayout::contiguous(extents, 2),
                                       fp32 ? ScalarType::kFloat32 : ScalarType::kBFloat16);
    DeviceTensor v_weight = vk.allocate(TensorLayout::contiguous(&feature, 1), ScalarType::kBFloat16);
    DeviceTensor v_scale = vk.allocate(TensorLayout::contiguous(mod_extents, 2));
    DeviceTensor v_shift = vk.allocate(TensorLayout::contiguous(mod_extents, 2));
    DeviceTensor v_selectors = vk.allocate(TensorLayout::contiguous(&selector_extent, 1),
                                           ScalarType::kInt32);
    DeviceTensor v_output = in_place
                                ? DeviceTensor()
                                : vk.allocate(TensorLayout::contiguous(extents, 2),
                                              fp32 ? ScalarType::kFloat32
                                                   : ScalarType::kBFloat16);
    if (fp32) vk.upload(v_input, f_input.data(), f_input.size());
    else vk.upload_bytes(v_input, bf_input.data(), bf_input.size() * sizeof(uint16_t));
    vk.upload_bytes(v_weight, weight.data(), weight.size() * sizeof(uint16_t));
    vk.upload(v_scale, scale.data(), scale.size());
    vk.upload(v_shift, shift.data(), shift.size());
    vk.upload_bytes(v_selectors, selectors, rows * sizeof(int32_t));
    TensorBatch batch = vk.begin_batch();
    if (fp32) batch.rms_norm_modulate_f32(v_input, v_weight, v_scale, v_shift,
                                          v_selectors, in_place ? v_input : v_output,
                                          epsilon);
    else batch.rms_norm_modulate_bf16(v_input, v_weight, v_scale, v_shift,
                                      v_selectors, in_place ? v_input : v_output,
                                      epsilon);
    batch.submit().wait();
    DeviceTensor& result = in_place ? v_input : v_output;
    if (fp32) {
      std::vector<float> expected(count), actual(count);
      if (!invalid_selectors) c_f_output.copy_to_host(expected.data(), count);
      vk.download(result, actual.data(), count);
      if (contraction_fixture) {
        uint32_t fused_bits = 0;
        std::memcpy(&fused_bits, &expected[0], 4);
        CHECK_MSG(fused_bits == 0x3ef07877u,
                  "CUDA AdaLN must select fused result, got %08x", fused_bits);
      }
      size_t mismatch = count;
      for (size_t i = 0; i < count; ++i) {
        if (std::memcmp(&expected[i], &actual[i], sizeof(float)) != 0) {
          mismatch = i; break;
        }
      }
      uint32_t expected_bits = 0, actual_bits = 0;
      if (mismatch != count) {
        std::memcpy(&expected_bits, &expected[mismatch], sizeof(expected_bits));
        std::memcpy(&actual_bits, &actual[mismatch], sizeof(actual_bits));
      }
      CHECK_MSG(mismatch == count,
                "CUDA/Vulkan fp32 AdaLN mismatch at %zu: %08x != %08x", mismatch,
                expected_bits, actual_bits);
    } else {
      std::vector<uint16_t> expected(count), actual(count);
      if (!invalid_selectors) c_bf_output.copy_to_host(expected.data(), count);
      vk.download_bytes(result, actual.data(), actual.size() * sizeof(uint16_t));
      compare_bf16(expected, actual, "BF16 AdaLN", rows, dim);
    }
  };
  run_mod(false);
  run_mod(true);
  run_mod(false, true);
  run_mod(true, true);
  run_mod(true, false, true);
  run_mod(false, false, false, 5, 31);
  run_mod(false, false, false, 3, 513, true);
  run_mod(true, false, false, 5, 31, true);
  run_mod(true, false, false, 3, 513);

  const uint64_t used_before_mixed = vk.pooled_used_bytes();
  {
  // All five shared-normalization pipelines share the same bounded two-slot
  // submission context. Keep two mixed jobs outstanding, force oldest-slot
  // reuse with a third, and prove descriptor/pool high-water stays stable.
  const uint64_t narrow_shape[] = {9, 128};
  const uint64_t wide_shape[] = {2, 513};
  const uint64_t layer_shape[] = {2, 129};
  const uint64_t mod_shape[] = {3, 31};
  const uint64_t narrow_feature = 128, wide_feature = 513, layer_feature = 129;
  const uint64_t mod_feature = 31, mod_rows = 4, selector_rows = 3;
  const uint64_t mod_parameter_shape[] = {mod_rows, mod_feature};
  DeviceTensor narrow_input = vk.allocate(TensorLayout::contiguous(narrow_shape, 2),
                                           ScalarType::kBFloat16);
  DeviceTensor narrow_weight = vk.allocate(TensorLayout::contiguous(&narrow_feature, 1),
                                            ScalarType::kBFloat16);
  DeviceTensor wide_input = vk.allocate(TensorLayout::contiguous(wide_shape, 2),
                                         ScalarType::kBFloat16);
  DeviceTensor wide_weight = vk.allocate(TensorLayout::contiguous(&wide_feature, 1),
                                          ScalarType::kBFloat16);
  DeviceTensor layer_input = vk.allocate(TensorLayout::contiguous(layer_shape, 2),
                                          ScalarType::kBFloat16);
  DeviceTensor layer_weight = vk.allocate(TensorLayout::contiguous(&layer_feature, 1),
                                           ScalarType::kBFloat16);
  DeviceTensor layer_bias = vk.allocate(TensorLayout::contiguous(&layer_feature, 1),
                                         ScalarType::kBFloat16);
  DeviceTensor mod_bf_input = vk.allocate(TensorLayout::contiguous(mod_shape, 2),
                                           ScalarType::kBFloat16);
  DeviceTensor mod_f_input = vk.allocate(TensorLayout::contiguous(mod_shape, 2));
  DeviceTensor mod_weight = vk.allocate(TensorLayout::contiguous(&mod_feature, 1),
                                         ScalarType::kBFloat16);
  DeviceTensor mod_scale = vk.allocate(TensorLayout::contiguous(mod_parameter_shape, 2));
  DeviceTensor mod_shift = vk.allocate(TensorLayout::contiguous(mod_parameter_shape, 2));
  DeviceTensor mod_selectors = vk.allocate(TensorLayout::contiguous(&selector_rows, 1),
                                            ScalarType::kInt32);
  std::array<std::array<DeviceTensor, 5>, 2> mixed_outputs;
  for (auto& outputs : mixed_outputs) {
    outputs[0] = vk.allocate(TensorLayout::contiguous(narrow_shape, 2), ScalarType::kBFloat16);
    outputs[1] = vk.allocate(TensorLayout::contiguous(wide_shape, 2), ScalarType::kBFloat16);
    outputs[2] = vk.allocate(TensorLayout::contiguous(layer_shape, 2), ScalarType::kBFloat16);
    outputs[3] = vk.allocate(TensorLayout::contiguous(mod_shape, 2), ScalarType::kBFloat16);
    outputs[4] = vk.allocate(TensorLayout::contiguous(mod_shape, 2));
  }
  std::vector<uint16_t> narrow_zero(9 * 128), narrow_one(128, exact_bf16(1.0f));
  std::vector<uint16_t> wide_zero(2 * 513), wide_one(513, exact_bf16(1.0f));
  std::vector<uint16_t> layer_zero(2 * 129), layer_one(129, exact_bf16(1.0f));
  std::vector<uint16_t> layer_zero_bias(129), mod_bf_zero(3 * 31), mod_one(31, exact_bf16(1.0f));
  std::vector<float> mod_f_zero(3 * 31), mod_parameter_zero(4 * 31);
  const int32_t mixed_selectors[] = {3, 1, 0};
  vk.upload_bytes(narrow_input, narrow_zero.data(), narrow_zero.size() * 2);
  vk.upload_bytes(narrow_weight, narrow_one.data(), narrow_one.size() * 2);
  vk.upload_bytes(wide_input, wide_zero.data(), wide_zero.size() * 2);
  vk.upload_bytes(wide_weight, wide_one.data(), wide_one.size() * 2);
  vk.upload_bytes(layer_input, layer_zero.data(), layer_zero.size() * 2);
  vk.upload_bytes(layer_weight, layer_one.data(), layer_one.size() * 2);
  vk.upload_bytes(layer_bias, layer_zero_bias.data(), layer_zero_bias.size() * 2);
  vk.upload_bytes(mod_bf_input, mod_bf_zero.data(), mod_bf_zero.size() * 2);
  vk.upload(mod_f_input, mod_f_zero.data(), mod_f_zero.size());
  vk.upload_bytes(mod_weight, mod_one.data(), mod_one.size() * 2);
  vk.upload(mod_scale, mod_parameter_zero.data(), mod_parameter_zero.size());
  vk.upload(mod_shift, mod_parameter_zero.data(), mod_parameter_zero.size());
  vk.upload_bytes(mod_selectors, mixed_selectors, sizeof(mixed_selectors));
  auto submit_mixed = [&](std::array<DeviceTensor, 5>& outputs) {
    TensorBatch batch = vk.begin_batch();
    batch.rms_norm_bf16(narrow_input, narrow_weight, outputs[0], 1.0e-5f);
    batch.rms_norm_bf16(wide_input, wide_weight, outputs[1], 1.0e-5f);
    batch.layer_norm_bf16(layer_input, layer_weight, layer_bias, outputs[2], 1.0e-6f);
    batch.rms_norm_modulate_bf16(mod_bf_input, mod_weight, mod_scale, mod_shift,
                                 mod_selectors, outputs[3], 1.0e-5f);
    batch.rms_norm_modulate_f32(mod_f_input, mod_weight, mod_scale, mod_shift,
                                mod_selectors, outputs[4], 1.0e-5f);
    return batch.submit();
  };
  {
    Submission first = submit_mixed(mixed_outputs[0]);
    Submission second = submit_mixed(mixed_outputs[1]);
    CHECK(first.value() != 0 && second.value() > first.value());
    first.wait(); second.wait();
  }
  const uint64_t mixed_reserved = vk.reserved_bytes();
  const uint64_t mixed_descriptors = vk.descriptor_set_allocations();
  for (int iteration = 0; iteration < 8; ++iteration) {
    Submission first = submit_mixed(mixed_outputs[0]);
    Submission second = submit_mixed(mixed_outputs[1]);
    Submission third = submit_mixed(mixed_outputs[0]);
    CHECK(second.value() > first.value() && third.value() > second.value());
    first.wait(); second.wait(); third.wait();
    CHECK(vk.reserved_bytes() == mixed_reserved);
    CHECK(vk.descriptor_set_allocations() == mixed_descriptors);
  }

  // Every new API rejects invalid metadata before recording. The same batch
  // remains usable after each rejection, proving no partial descriptor/barrier
  // mutation was committed.
  DeviceTensor wrong_type = vk.allocate(TensorLayout::contiguous(&narrow_feature, 1));
  const uint64_t short_shape[] = {9, 127};
  DeviceTensor wrong_shape = vk.allocate(TensorLayout::contiguous(short_shape, 2),
                                          ScalarType::kBFloat16);
  auto valid_after_rejection = [&](auto&& invalid) {
    TensorBatch batch = vk.begin_batch();
    bool rejected = false;
    try { invalid(batch); } catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    batch.rms_norm_bf16(narrow_input, narrow_weight, mixed_outputs[0][0], 1.0e-5f);
    batch.submit().wait();
  };
  valid_after_rejection([&](TensorBatch& batch) {
    batch.rms_norm_bf16(narrow_input, wrong_type, mixed_outputs[0][0], 1.0e-5f);
  });
  valid_after_rejection([&](TensorBatch& batch) {
    batch.rms_norm_bf16(narrow_input, narrow_weight, wrong_shape, 1.0e-5f);
  });
  valid_after_rejection([&](TensorBatch& batch) {
    batch.layer_norm_bf16(layer_input, layer_weight, layer_input, mixed_outputs[0][2],
                          1.0e-6f);
  });
  valid_after_rejection([&](TensorBatch& batch) {
    batch.rms_norm_bf16(narrow_input, narrow_weight, mixed_outputs[0][0], 0.0f);
  });
  }
  // Submission records retain every referenced tensor until its exact token
  // completes. Once jobs and caller wrappers are gone, all pooled spans are
  // returned while the context itself remains alive.
  { TensorBatch collect_completed_slots = vk.begin_batch(); }
  CHECK_MSG(vk.pooled_used_bytes() == used_before_mixed,
            "mixed pooled bytes not released: %llu != %llu",
            static_cast<unsigned long long>(vk.pooled_used_bytes()),
            static_cast<unsigned long long>(used_before_mixed));
}

VIDFAB_TEST(cuda_vulkan_tensor_exact_group_norm_silu) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options; options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  if (!vk.exact_normalization()) return;

  auto run_case = [&](int channels, int height, int width, int groups,
                      int pattern, bool in_place) {
    const size_t count = static_cast<size_t>(channels) * height * width;
    std::vector<float> input(count);
    for (size_t i = 0; i < count; ++i)
      input[i] = static_cast<float>(static_cast<int>((i * 29) % 103) - 51) / 16.0f;
    if (pattern == 1) std::fill(input.begin(), input.end(), 2.0f);
    if (pattern == 2) {
      for (size_t i = 0; i < count; ++i)
        input[i] = 4096.0f + static_cast<float>(static_cast<int>(i % 7) - 3) / 8.0f;
    }
    if (pattern == 3) {
      for (size_t i = 0; i < count; ++i) input[i] = (i & 1) ? -0.0f : 0.0f;
    }
    std::vector<__half> weight(channels), bias(channels);
    std::vector<uint16_t> weight_bits(channels), bias_bits(channels);
    for (int c = 0; c < channels; ++c) {
      weight[c] = __float2half(0.5f + static_cast<float>(c % 11) / 16.0f);
      bias[c] = __float2half(static_cast<float>((c % 13) - 6) / 32.0f);
      if (pattern == 3 && c < 4) {
        const float edge_weight[] = {-0.0f, 0.0f, 65504.0f, -65504.0f};
        const float edge_bias[] = {-0.0f, 0.0f, 1.0f, -1.0f};
        weight[c] = __float2half(edge_weight[c]); bias[c] = __float2half(edge_bias[c]);
      }
      std::memcpy(&weight_bits[c], &weight[c], 2);
      std::memcpy(&bias_bits[c], &bias[c], 2);
    }
    cuda::DeviceBuffer<float> c_input(count), c_output(count);
    cuda::DeviceBuffer<__half> c_weight(channels), c_bias(channels);
    c_input.copy_from_host(input.data(), count);
    c_weight.copy_from_host(weight.data(), channels);
    c_bias.copy_from_host(bias.data(), channels);
    cuda::launch_keyframe_groupnorm_silu(c_input.get(), c_weight.get(), c_bias.get(),
                                         c_output.get(), channels, height, width,
                                         groups, 1.0e-6f, nullptr);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> expected(count), actual(count);
    c_output.copy_to_host(expected.data(), count);

    const uint64_t shape[] = {static_cast<uint64_t>(channels),
                              static_cast<uint64_t>(height),
                              static_cast<uint64_t>(width)};
    const uint64_t feature = channels;
    DeviceTensor v_input = vk.allocate(TensorLayout::contiguous(shape, 3));
    DeviceTensor v_weight = vk.allocate(TensorLayout::contiguous(&feature, 1),
                                         ScalarType::kFloat16);
    DeviceTensor v_bias = vk.allocate(TensorLayout::contiguous(&feature, 1),
                                       ScalarType::kFloat16);
    DeviceTensor v_output = in_place ? DeviceTensor() :
        vk.allocate(TensorLayout::contiguous(shape, 3));
    vk.upload(v_input, input.data(), count);
    vk.upload_bytes(v_weight, weight_bits.data(), weight_bits.size() * 2);
    vk.upload_bytes(v_bias, bias_bits.data(), bias_bits.size() * 2);
    TensorBatch batch = vk.begin_batch();
    batch.group_norm_silu_f16_affine(v_input, v_weight, v_bias,
                                     in_place ? v_input : v_output,
                                     static_cast<uint32_t>(groups), 1.0e-6f);
    batch.submit().wait();
    DeviceTensor& result = in_place ? v_input : v_output;
    vk.download(result, actual.data(), count);
    size_t mismatch = count;
    for (size_t i = 0; i < count; ++i) {
      if (std::memcmp(&expected[i], &actual[i], 4) != 0) { mismatch = i; break; }
    }
    uint32_t expected_bits = 0, actual_bits = 0;
    if (mismatch != count) {
      std::memcpy(&expected_bits, &expected[mismatch], 4);
      std::memcpy(&actual_bits, &actual[mismatch], 4);
    }
    CHECK_MSG(mismatch == count,
              "CUDA/Vulkan GroupNorm+SiLU %dx%dx%d g%d mismatch at %zu: %08x != %08x",
              channels, height, width, groups, mismatch, expected_bits, actual_bits);
  };
  run_case(128, 17, 19, 32, 0, false);
  run_case(96, 5, 7, 32, 1, true);
  run_case(128, 3, 11, 16, 2, false);
  run_case(128, 4, 13, 32, 3, true);

  bool cuda_limit_rejected = false;
  try {
    cuda::launch_keyframe_groupnorm_silu(
        reinterpret_cast<const float*>(uintptr_t{1}),
        reinterpret_cast<const __half*>(uintptr_t{1}),
        reinterpret_cast<const __half*>(uintptr_t{1}),
        reinterpret_cast<float*>(uintptr_t{1}), 128, 2048, 2049, 32,
        1.0e-6f, nullptr);
  } catch (const std::runtime_error&) { cuda_limit_rejected = true; }
  CHECK(cuda_limit_rejected);

  const uint64_t used_before_reuse = vk.pooled_used_bytes();
  {
    const uint64_t shape[] = {128, 3, 5}, feature = 128;
    DeviceTensor input = vk.allocate(TensorLayout::contiguous(shape, 3));
    DeviceTensor weight = vk.allocate(TensorLayout::contiguous(&feature, 1),
                                       ScalarType::kFloat16);
    DeviceTensor bias = vk.allocate(TensorLayout::contiguous(&feature, 1),
                                     ScalarType::kFloat16);
    std::array<DeviceTensor, 2> outputs{
        vk.allocate(TensorLayout::contiguous(shape, 3)),
        vk.allocate(TensorLayout::contiguous(shape, 3))};
    std::vector<float> zeros(128 * 3 * 5);
    std::vector<uint16_t> affine(128, 0x3c00u), offsets(128);
    vk.upload(input, zeros.data(), zeros.size());
    vk.upload_bytes(weight, affine.data(), affine.size() * 2);
    vk.upload_bytes(bias, offsets.data(), offsets.size() * 2);
    auto submit = [&](DeviceTensor& output) {
      TensorBatch batch = vk.begin_batch();
      batch.group_norm_silu_f16_affine(input, weight, bias, output, 32, 1.0e-6f);
      return batch.submit();
    };
    Submission warm_a = submit(outputs[0]), warm_b = submit(outputs[1]);
    warm_a.wait(); warm_b.wait();
    const uint64_t stable_reserved = vk.reserved_bytes();
    const uint64_t stable_descriptors = vk.descriptor_set_allocations();
    for (int repeat = 0; repeat < 8; ++repeat) {
      Submission first = submit(outputs[0]);
      Submission second = submit(outputs[1]);
      Submission third = submit(outputs[0]);
      CHECK(second.value() > first.value() && third.value() > second.value());
      first.wait(); second.wait(); third.wait();
      CHECK(vk.reserved_bytes() == stable_reserved);
      CHECK_MSG(vk.descriptor_set_allocations() == stable_descriptors,
                "GroupNorm descriptors grew: %llu != %llu",
                static_cast<unsigned long long>(vk.descriptor_set_allocations()),
                static_cast<unsigned long long>(stable_descriptors));
    }
    DeviceTensor wrong_type = vk.allocate(TensorLayout::contiguous(&feature, 1));
    const uint64_t short_shape[] = {128, 3, 4};
    DeviceTensor wrong_shape = vk.allocate(TensorLayout::contiguous(short_shape, 3));
    auto valid_after_rejection = [&](auto&& invalid) {
      TensorBatch batch = vk.begin_batch(); bool rejected = false;
      try { invalid(batch); } catch (const std::invalid_argument&) { rejected = true; }
      CHECK(rejected);
      batch.group_norm_silu_f16_affine(input, weight, bias, outputs[0], 32, 1.0e-6f);
      batch.submit().wait();
    };
    valid_after_rejection([&](TensorBatch& batch) {
      batch.group_norm_silu_f16_affine(input, wrong_type, bias, outputs[0], 32, 1.0e-6f);
    });
    valid_after_rejection([&](TensorBatch& batch) {
      batch.group_norm_silu_f16_affine(input, weight, bias, wrong_shape, 32, 1.0e-6f);
    });
    valid_after_rejection([&](TensorBatch& batch) {
      batch.group_norm_silu_f16_affine(input, weight, bias, outputs[0], 31, 1.0e-6f);
    });
    valid_after_rejection([&](TensorBatch& batch) {
      batch.group_norm_silu_f16_affine(input, weight, weight, outputs[0], 32, 1.0e-6f);
    });
    valid_after_rejection([&](TensorBatch& batch) {
      batch.group_norm_silu_f16_affine(input, weight, bias, outputs[0], 32, 0.0f);
    });
  }
  { TensorBatch collect_completed_slots = vk.begin_batch(); }
  CHECK(vk.pooled_used_bytes() == used_before_reuse);
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

VIDFAB_TEST(cuda_vulkan_tensor_exact_bf16_rope) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);

  auto run = [&](uint32_t mode, uint32_t rows, uint32_t heads,
                 uint32_t head_dim) {
    const size_t count = static_cast<size_t>(rows) * heads * head_dim;
    std::vector<uint16_t> input(count);
    for (size_t i = 0; i < count; ++i) {
      if (i < 127) input[i] = static_cast<uint16_t>(i + 1);
      else if (i < 254) input[i] = static_cast<uint16_t>(0x8000u | (i - 126));
      else input[i] = static_cast<uint16_t>(
          ((i * 977u + 0x3c00u) & 0x7fffu) |
          ((i & 7u) == 0u ? 0x8000u : 0u));
    }
    const uint16_t boundaries[] = {0x0000u, 0x8000u, 0x007fu, 0x807fu,
                                   0x0080u, 0x8080u, 0x0081u, 0x8081u,
                                   0x7f80u, 0xff80u, 0x7fc1u};
    for (size_t i = 0; i < std::size(boundaries); ++i) input[254 + i] = boundaries[i];
    const uint32_t table_width = mode == 0 ? 96u : head_dim;
    std::vector<float> cosine, sine;
    if (mode == 0) {
      std::vector<double> positions(static_cast<size_t>(rows) * 3);
      for (size_t i = 0; i < positions.size(); ++i)
        positions[i] = static_cast<double>(static_cast<int>(i * 17u % 101u) - 50) / 8.0;
      auto tables = dit::build_h3_rope_tables(positions, 10000.0f, 16);
      cosine = std::move(tables.cosine);
      sine = std::move(tables.sine);
    } else {
      cosine.resize(static_cast<size_t>(rows) * head_dim);
      sine.resize(cosine.size());
      const uint32_t half = head_dim / 2;
      for (uint32_t row = 0; row < rows; ++row) {
        for (uint32_t j = 0; j < half; ++j) {
          const float angle = static_cast<float>((row + 1) * (j + 1)) / 37.0f;
          const float c = std::cos(angle), s = std::sin(angle);
          cosine[static_cast<size_t>(row) * head_dim + j] =
              cosine[static_cast<size_t>(row) * head_dim + j + half] = c;
          sine[static_cast<size_t>(row) * head_dim + j] =
              sine[static_cast<size_t>(row) * head_dim + j + half] = s;
        }
      }
      // Exact cancellation and an intermediate that underflows fp32 are part
      // of the backend-stable signed-zero policy.
      input[0] = input[head_dim / 2] = 0x3f80u;
      cosine[0] = cosine[head_dim / 2] = 0.5f;
      sine[0] = sine[head_dim / 2] = 0.5f;
      if (rows > 1) {
        const size_t data_row = static_cast<size_t>(heads) * head_dim;
        input[data_row + 1] = 0x0080u;
        input[data_row + 1 + head_dim / 2] = 0x0000u;
        const size_t table_row = head_dim;
        cosine[table_row + 1] = cosine[table_row + 1 + head_dim / 2] =
            std::numeric_limits<float>::min();
        sine[table_row + 1] = sine[table_row + 1 + head_dim / 2] = 0.0f;
        // The high-half FMA has a nonzero product that rounds into the fp32
        // subnormal range. Both backends canonicalize that internal product
        // before it can contribute to an otherwise normal result.
        input[data_row + 2] = 0x0080u;
        input[data_row + 2 + head_dim / 2] = 0x3f80u;
        cosine[table_row + 2 + head_dim / 2] = 0.75f;
        sine[table_row + 2 + head_dim / 2] =
            std::numeric_limits<float>::min();
      }
    }
    cuda::DeviceBuffer<__nv_bfloat16> cuda_data(count);
    cuda::DeviceBuffer<float> cuda_cos(cosine.size()), cuda_sin(sine.size());
    cuda_data.copy_from_host(reinterpret_cast<const __nv_bfloat16*>(input.data()), count);
    cuda_cos.copy_from_host(cosine.data(), cosine.size());
    cuda_sin.copy_from_host(sine.data(), sine.size());
    if (mode == 0)
      cuda::launch_rope_h3(cuda_data.get(), cuda_cos.get(), cuda_sin.get(), rows,
                           heads, head_dim, nullptr);
    else
      cuda::launch_rope_neox(cuda_data.get(), cuda_cos.get(), cuda_sin.get(), rows,
                             heads, head_dim, nullptr);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> expected(count), actual(count);
    cuda_data.copy_to_host(reinterpret_cast<__nv_bfloat16*>(expected.data()), count);

    const uint64_t data_shape[] = {rows, heads, head_dim};
    const uint64_t table_shape[] = {rows, table_width};
    DeviceTensor vk_data = vk.allocate(
        TensorLayout::contiguous(data_shape, 3), ScalarType::kBFloat16);
    DeviceTensor vk_cos = vk.allocate(TensorLayout::contiguous(table_shape, 2));
    DeviceTensor vk_sin = vk.allocate(TensorLayout::contiguous(table_shape, 2));
    vk.upload_bytes(vk_data, input.data(), input.size() * sizeof(uint16_t));
    vk.upload(vk_cos, cosine.data(), cosine.size());
    vk.upload(vk_sin, sine.data(), sine.size());
    TensorBatch batch = vk.begin_batch();
    if (mode == 0) batch.rope_h3_bf16(vk_data, vk_cos, vk_sin);
    else batch.rope_neox_bf16(vk_data, vk_cos, vk_sin);
    batch.submit().wait();
    vk.download_bytes(vk_data, actual.data(), actual.size() * sizeof(uint16_t));
    size_t mismatch = count;
    for (size_t i = 0; i < count; ++i) {
      if (expected[i] != actual[i]) { mismatch = i; break; }
    }
    CHECK_MSG(mismatch == count,
              "CUDA/Vulkan BF16 RoPE mode%u %ux%ux%u mismatch at %zu: %04x != %04x",
              mode, rows, heads, head_dim, mismatch,
              mismatch == count ? 0 : expected[mismatch],
              mismatch == count ? 0 : actual[mismatch]);
    if (mode == 0) {
      bool tail_preserved = true;
      for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t head = 0; head < heads; ++head)
          for (uint32_t d = 96; d < head_dim; ++d) {
            const size_t at = (static_cast<size_t>(row) * heads + head) * head_dim + d;
            tail_preserved &= actual[at] == input[at];
          }
      CHECK(tail_preserved);
    } else {
      CHECK(actual[0] == 0x0000u);
      CHECK(actual[static_cast<size_t>(heads) * head_dim + 1] == 0x0000u);
      CHECK(actual[static_cast<size_t>(heads) * head_dim + 2 + head_dim / 2] ==
            0x3f40u);
    }
  };
  run(0, 5, 7, 128);
  run(1, 5, 3, 72);
  run(1, 3, 5, 128);

  const uint64_t used_before_reuse = vk.pooled_used_bytes();
  {
    const uint64_t data_shape[] = {2, 3, 128}, table_shape[] = {2, 96};
    DeviceTensor first = vk.allocate(
        TensorLayout::contiguous(data_shape, 3), ScalarType::kBFloat16);
    DeviceTensor second = vk.allocate(
        TensorLayout::contiguous(data_shape, 3), ScalarType::kBFloat16);
    DeviceTensor cosine = vk.allocate(TensorLayout::contiguous(table_shape, 2));
    DeviceTensor sine = vk.allocate(TensorLayout::contiguous(table_shape, 2));
    std::vector<uint16_t> zeros(2 * 3 * 128);
    std::vector<float> ones(2 * 96, 1.0f), table_zeros(2 * 96);
    vk.upload_bytes(first, zeros.data(), zeros.size() * 2);
    vk.upload_bytes(second, zeros.data(), zeros.size() * 2);
    vk.upload(cosine, ones.data(), ones.size());
    vk.upload(sine, table_zeros.data(), table_zeros.size());
    auto submit = [&](DeviceTensor& data) {
      TensorBatch batch = vk.begin_batch();
      batch.rope_h3_bf16(data, cosine, sine);
      return batch.submit();
    };
    Submission warm_a = submit(first), warm_b = submit(second);
    warm_a.wait(); warm_b.wait();
    const uint64_t stable_reserved = vk.reserved_bytes();
    const uint64_t stable_descriptors = vk.descriptor_set_allocations();
    for (int repeat = 0; repeat < 4; ++repeat) {
      Submission a = submit(first), b = submit(second), c = submit(first);
      CHECK(b.value() > a.value() && c.value() > b.value());
      a.wait(); b.wait(); c.wait();
      CHECK(vk.reserved_bytes() == stable_reserved);
      CHECK(vk.descriptor_set_allocations() == stable_descriptors);
    }
    const uint64_t odd_shape[] = {2, 3, 127};
    const uint64_t short_table_shape[] = {2, 95};
    DeviceTensor odd = vk.allocate(
        TensorLayout::contiguous(odd_shape, 3), ScalarType::kBFloat16);
    DeviceTensor short_table = vk.allocate(
        TensorLayout::contiguous(short_table_shape, 2));
    DeviceTensor wrong_type = vk.allocate(
        TensorLayout::contiguous(table_shape, 2), ScalarType::kBFloat16);
    auto valid_after_rejection = [&](auto&& invalid) {
      TensorBatch batch = vk.begin_batch();
      bool rejected = false;
      try { invalid(batch); } catch (const std::invalid_argument&) { rejected = true; }
      CHECK(rejected);
      batch.rope_h3_bf16(first, cosine, sine);
      batch.submit().wait();
    };
    valid_after_rejection([&](TensorBatch& batch) {
      batch.rope_h3_bf16(odd, cosine, sine);
    });
    valid_after_rejection([&](TensorBatch& batch) {
      batch.rope_h3_bf16(first, short_table, sine);
    });
    valid_after_rejection([&](TensorBatch& batch) {
      batch.rope_h3_bf16(first, wrong_type, sine);
    });
    valid_after_rejection([&](TensorBatch& batch) {
      batch.rope_h3_bf16(first, cosine, cosine);
    });
  }
  { TensorBatch collect_completed_slots = vk.begin_batch(); }
  CHECK(vk.pooled_used_bytes() == used_before_reuse);
}

VIDFAB_TEST(cuda_vulkan_tensor_exact_vae_fused_rope) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore ||
      !physical.front().info().shader_int64) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = true;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  if (!vk.exact_normalization()) return;

  constexpr uint32_t sequence = 5, heads = 3, head_dim = 64,
                     rope_dim = 48, num_patches = 3;
  constexpr float epsilon = 1.0e-6f;
  const size_t qkv_count = static_cast<size_t>(sequence) * heads * 3 * head_dim;
  const size_t bias_count = static_cast<size_t>(heads) * 3 * head_dim;
  const size_t output_count = static_cast<size_t>(sequence) * heads * head_dim;
  std::vector<float> input = test::make_data(qkv_count, 0x13579u, 1.25f);
  std::vector<float> bias_values = test::make_data(bias_count, 0x24680u, 0.3f);
  std::vector<float> cosine(static_cast<size_t>(sequence) * rope_dim);
  std::vector<float> sine(cosine.size());
  for (uint32_t row = 0; row < sequence; ++row) {
    for (uint32_t d = 0; d < rope_dim / 2; ++d) {
      const float angle = static_cast<float>((row + 1) * (d + 3)) / 41.0f;
      const float c = std::cos(angle), s = std::sin(angle);
      cosine[static_cast<size_t>(row) * rope_dim + d] =
          cosine[static_cast<size_t>(row) * rope_dim + d + rope_dim / 2] = c;
      sine[static_cast<size_t>(row) * rope_dim + d] =
          sine[static_cast<size_t>(row) * rope_dim + d + rope_dim / 2] = s;
    }
  }

  cuda::DeviceBuffer<float> c_input(qkv_count), c_bias(bias_count),
      c_cos(cosine.size()), c_sin(sine.size()), c_q(output_count),
      c_k(output_count), c_v(output_count);
  c_input.copy_from_host(input.data(), input.size());
  c_bias.copy_from_host(bias_values.data(), bias_values.size());
  c_cos.copy_from_host(cosine.data(), cosine.size());
  c_sin.copy_from_host(sine.data(), sine.size());
  cuda::launch_split_qkv_norm_rope(
      c_input.get(), c_bias.get(), c_cos.get(), c_sin.get(), c_q.get(),
      c_k.get(), c_v.get(), sequence, heads, head_dim, rope_dim, num_patches,
      epsilon, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::array<std::vector<float>, 3> expected{
      std::vector<float>(output_count), std::vector<float>(output_count),
      std::vector<float>(output_count)};
  c_q.copy_to_host(expected[0].data(), output_count);
  c_k.copy_to_host(expected[1].data(), output_count);
  c_v.copy_to_host(expected[2].data(), output_count);

  const uint64_t qkv_shape[] = {sequence, heads, 3 * head_dim};
  const uint64_t bias_shape[] = {heads, 3 * head_dim};
  const uint64_t table_shape[] = {sequence, rope_dim};
  const uint64_t output_shape[] = {heads, sequence, head_dim};
  DeviceTensor v_input = vk.allocate(TensorLayout::contiguous(qkv_shape, 3));
  DeviceTensor v_bias = vk.allocate(TensorLayout::contiguous(bias_shape, 2));
  DeviceTensor v_cos = vk.allocate(TensorLayout::contiguous(table_shape, 2));
  DeviceTensor v_sin = vk.allocate(TensorLayout::contiguous(table_shape, 2));
  std::array<DeviceTensor, 3> outputs{
      vk.allocate(TensorLayout::contiguous(output_shape, 3)),
      vk.allocate(TensorLayout::contiguous(output_shape, 3)),
      vk.allocate(TensorLayout::contiguous(output_shape, 3))};
  vk.upload(v_input, input.data(), input.size());
  vk.upload(v_bias, bias_values.data(), bias_values.size());
  vk.upload(v_cos, cosine.data(), cosine.size());
  vk.upload(v_sin, sine.data(), sine.size());
  TensorBatch batch = vk.begin_batch();
  batch.split_qkv_norm_rope_f32(v_input, v_bias, v_cos, v_sin, outputs[0],
                                outputs[1], outputs[2], num_patches, epsilon);
  batch.submit().wait();
  std::array<std::vector<float>, 3> actual_outputs{
      std::vector<float>(output_count), std::vector<float>(output_count),
      std::vector<float>(output_count)};
  for (size_t output = 0; output < outputs.size(); ++output) {
    auto& actual = actual_outputs[output];
    vk.download(outputs[output], actual.data(), actual.size());
    size_t mismatch = output_count;
    for (size_t i = 0; i < output_count; ++i) {
      uint32_t want = 0, got = 0;
      std::memcpy(&want, &expected[output][i], 4);
      std::memcpy(&got, &actual[i], 4);
      if (want != got) { mismatch = i; break; }
    }
    CHECK_MSG(mismatch == output_count,
              "CUDA/Vulkan fused VAE RoPE output%zu mismatch at %zu: %08x != %08x",
              output, mismatch,
              mismatch == output_count ? 0u : [&] { uint32_t x; std::memcpy(&x, &expected[output][mismatch], 4); return x; }(),
              mismatch == output_count ? 0u : [&] { uint32_t x; std::memcpy(&x, &actual[mismatch], 4); return x; }());
  }
  // Independent head-major V layout reference: V is bias-only and bypasses
  // both normalization and rotation.
  for (uint32_t token = 0; token < sequence; ++token) {
    for (uint32_t head = 0; head < heads; ++head) {
      for (uint32_t d = 0; d < head_dim; ++d) {
        const size_t source = (static_cast<size_t>(token) * heads + head) *
                                  (3 * head_dim) +
                              2 * head_dim + d;
        const size_t destination =
            (static_cast<size_t>(head) * sequence + token) * head_dim + d;
        const float reference = input[source] +
            bias_values[static_cast<size_t>(head) * 3 * head_dim + 2 * head_dim + d];
        uint32_t want = 0, got = 0;
        std::memcpy(&want, &reference, 4);
        std::memcpy(&got, &actual_outputs[2][destination], 4);
        CHECK_MSG(want == got, "fused V layout mismatch token%u head%u dim%u", token,
                  head, d);
      }
    }
  }

  std::array<DeviceTensor, 3> second_outputs{
      vk.allocate(TensorLayout::contiguous(output_shape, 3)),
      vk.allocate(TensorLayout::contiguous(output_shape, 3)),
      vk.allocate(TensorLayout::contiguous(output_shape, 3))};
  auto submit = [&](std::array<DeviceTensor, 3>& selected) {
    TensorBatch next = vk.begin_batch();
    next.split_qkv_norm_rope_f32(v_input, v_bias, v_cos, v_sin, selected[0],
                                 selected[1], selected[2], num_patches, epsilon);
    return next.submit();
  };
  std::vector<float> suffix_cosine = cosine, suffix_sine = sine;
  for (uint32_t token = num_patches; token < sequence; ++token) {
    for (uint32_t d = 0; d < rope_dim; ++d) {
      suffix_cosine[static_cast<size_t>(token) * rope_dim + d] =
          static_cast<float>(17 + token + d);
      suffix_sine[static_cast<size_t>(token) * rope_dim + d] =
          static_cast<float>(-31 - static_cast<int>(token) - static_cast<int>(d));
    }
  }
  vk.upload(v_cos, suffix_cosine.data(), suffix_cosine.size());
  vk.upload(v_sin, suffix_sine.data(), suffix_sine.size());
  submit(second_outputs).wait();
  for (size_t output = 0; output < 2; ++output) {
    std::vector<float> suffix_actual(output_count);
    vk.download(second_outputs[output], suffix_actual.data(), suffix_actual.size());
    for (uint32_t head = 0; head < heads; ++head) {
      for (uint32_t token = num_patches; token < sequence; ++token) {
        const size_t begin = (static_cast<size_t>(head) * sequence + token) * head_dim;
        CHECK(std::memcmp(suffix_actual.data() + begin,
                          actual_outputs[output].data() + begin,
                          head_dim * sizeof(float)) == 0);
      }
    }
  }
  vk.upload(v_cos, cosine.data(), cosine.size());
  vk.upload(v_sin, sine.data(), sine.size());
  Submission warm_a = submit(outputs), warm_b = submit(second_outputs);
  warm_a.wait(); warm_b.wait();
  const uint64_t stable_reserved = vk.reserved_bytes();
  const uint64_t stable_descriptors = vk.descriptor_set_allocations();
  for (int repeat = 0; repeat < 4; ++repeat) {
    Submission a = submit(outputs), b = submit(second_outputs), c = submit(outputs);
    CHECK(b.value() > a.value() && c.value() > b.value());
    a.wait(); b.wait(); c.wait();
    CHECK(vk.reserved_bytes() == stable_reserved);
    CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  }

  const uint64_t short_table_shape[] = {sequence, rope_dim - 1};
  DeviceTensor short_table = vk.allocate(TensorLayout::contiguous(short_table_shape, 2));
  DeviceTensor wrong_type = vk.allocate(TensorLayout::contiguous(table_shape, 2),
                                        ScalarType::kBFloat16);
  auto valid_after_rejection = [&](auto&& invalid) {
    TensorBatch next = vk.begin_batch();
    bool rejected = false;
    try { invalid(next); } catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    next.split_qkv_norm_rope_f32(v_input, v_bias, v_cos, v_sin, outputs[0],
                                 outputs[1], outputs[2], num_patches, epsilon);
    next.submit().wait();
  };
  valid_after_rejection([&](TensorBatch& next) {
    next.split_qkv_norm_rope_f32(v_input, v_bias, short_table, v_sin,
                                 outputs[0], outputs[1], outputs[2],
                                 num_patches, epsilon);
  });
  valid_after_rejection([&](TensorBatch& next) {
    next.split_qkv_norm_rope_f32(v_input, v_bias, wrong_type, v_sin,
                                 outputs[0], outputs[1], outputs[2],
                                 num_patches, epsilon);
  });
  valid_after_rejection([&](TensorBatch& next) {
    next.split_qkv_norm_rope_f32(v_input, v_bias, v_cos, v_sin,
                                 outputs[0], outputs[0], outputs[2],
                                 num_patches, epsilon);
  });
  valid_after_rejection([&](TensorBatch& next) {
    next.split_qkv_norm_rope_f32(v_input, v_bias, v_cos, v_sin,
                                 outputs[0], outputs[1], outputs[2],
                                 sequence + 1, epsilon);
  });
  valid_after_rejection([&](TensorBatch& next) {
    next.split_qkv_norm_rope_f32(v_input, v_bias, v_cos, v_sin,
                                 outputs[0], outputs[1], outputs[2],
                                 num_patches, 0.0f);
  });

  // A submitted job owns all seven buffers after every public wrapper drops.
  // Once the exact token completes and the slot is collected, no hidden
  // descriptor/scratch reference may keep any of them alive.
  const uint64_t used_before_drop = vk.pooled_used_bytes();
  Submission retained;
  {
    const uint64_t tiny_qkv_shape[] = {2, 1, 192};
    const uint64_t tiny_bias_shape[] = {1, 192};
    const uint64_t tiny_table_shape[] = {2, 48};
    const uint64_t tiny_output_shape[] = {1, 2, 64};
    DeviceTensor in = vk.allocate(TensorLayout::contiguous(tiny_qkv_shape, 3));
    DeviceTensor bv = vk.allocate(TensorLayout::contiguous(tiny_bias_shape, 2));
    DeviceTensor cv = vk.allocate(TensorLayout::contiguous(tiny_table_shape, 2));
    DeviceTensor sv = vk.allocate(TensorLayout::contiguous(tiny_table_shape, 2));
    DeviceTensor qv = vk.allocate(TensorLayout::contiguous(tiny_output_shape, 3));
    DeviceTensor kv = vk.allocate(TensorLayout::contiguous(tiny_output_shape, 3));
    DeviceTensor vv = vk.allocate(TensorLayout::contiguous(tiny_output_shape, 3));
    TensorBatch next = vk.begin_batch();
    next.split_qkv_norm_rope_f32(in, bv, cv, sv, qv, kv, vv, 1, epsilon);
    retained = next.submit();
  }
  CHECK(vk.pooled_used_bytes() > used_before_drop);
  retained.wait();
  retained = Submission{};
  { TensorBatch collect_completed_slot = vk.begin_batch(); }
  CHECK(vk.pooled_used_bytes() == used_before_drop);

}

VIDFAB_TEST(cuda_vulkan_linear_weight_f8_i8_exact) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);

  auto exact_case = [&](const LinearWeightUpload& upload,
                        const std::vector<uint16_t>& expected) {
    LinearWeight weight = LinearWeight::upload(vk, upload);
    const uint64_t shape[] = {upload.out_features, upload.in_features};
    DeviceTensor dense = vk.allocate(TensorLayout::contiguous(shape, 2),
                                     ScalarType::kBFloat16);
    TensorBatch batch = vk.begin_batch();
    weight.materialize_bf16(batch, dense);
    batch.submit().wait();
    std::vector<uint16_t> actual(expected.size());
    vk.download_bytes(dense, actual.data(), actual.size() * sizeof(uint16_t));
    size_t mismatch = expected.size();
    for (size_t i = 0; i < expected.size(); ++i) {
      if (expected[i] != actual[i]) { mismatch = i; break; }
    }
    CHECK_MSG(mismatch == expected.size(),
              "CUDA/Vulkan linear weight mismatch at %zu: %04x != %04x",
              mismatch, mismatch == expected.size() ? 0u : expected[mismatch],
              mismatch == expected.size() ? 0u : actual[mismatch]);
  };

  std::vector<uint8_t> f8(256);
  for (uint32_t i = 0; i < 256; ++i) f8[i] = static_cast<uint8_t>(i);
  const float f8_scale = 0.75f;
  cuda::DeviceBuffer<uint8_t> cuda_f8(f8.size());
  cuda::DeviceBuffer<float> cuda_f8_scale(1);
  cuda::DeviceBuffer<__nv_bfloat16> cuda_f8_output(f8.size());
  cuda_f8.copy_from_host(f8.data(), f8.size());
  cuda_f8_scale.copy_from_host(&f8_scale, 1);
  cuda::launch_dequant_f8e4m3(cuda_f8.get(), cuda_f8_scale.get(),
                              cuda_f8_output.get(), f8.size(), nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> f8_expected(f8.size());
  VIDFAB_CUDA_CHECK(cudaMemcpy(f8_expected.data(), cuda_f8_output.get(),
                              f8_expected.size() * sizeof(uint16_t),
                              cudaMemcpyDeviceToHost));
  LinearWeightUpload f8_upload;
  f8_upload.format = LinearWeightFormat::kFloat8E4M3;
  f8_upload.out_features = 1;
  f8_upload.in_features = static_cast<uint32_t>(f8.size());
  f8_upload.data = f8.data();
  f8_upload.data_bytes = f8.size();
  f8_upload.weight_scale = &f8_scale;
  f8_upload.weight_scale_count = 1;
  f8_upload.has_fp8_input_scale = true;
  f8_upload.fp8_input_scale = 0.125f;
  f8_upload.full_precision_matrix_mult = true;
  exact_case(f8_upload, f8_expected);
  {
    LinearWeight metadata = LinearWeight::upload(vk, f8_upload);
    CHECK(metadata.has_fp8_input_scale());
    CHECK(metadata.fp8_input_scale() == f8_upload.fp8_input_scale);
    CHECK(metadata.full_precision_matrix_mult());
  }

  constexpr uint32_t i8_rows = 3, i8_columns = 131;
  std::vector<int8_t> i8(static_cast<size_t>(i8_rows) * i8_columns);
  for (size_t i = 0; i < i8.size(); ++i) {
    i8[i] = static_cast<int8_t>((i * 73u + 128u) & 0xffu);
  }
  const float i8_scale[] = {0.5f, -0.25f, 1.5f};
  cuda::DeviceBuffer<int8_t> cuda_i8(i8.size());
  cuda::DeviceBuffer<float> cuda_i8_scale(i8_rows);
  cuda::DeviceBuffer<__nv_bfloat16> cuda_i8_output(i8.size());
  cuda_i8.copy_from_host(i8.data(), i8.size());
  cuda_i8_scale.copy_from_host(i8_scale, i8_rows);
  cuda::launch_dequant_i8_per_channel(cuda_i8.get(), cuda_i8_scale.get(),
                                      cuda_i8_output.get(), i8_rows, i8_columns,
                                      nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> i8_expected(i8.size());
  VIDFAB_CUDA_CHECK(cudaMemcpy(i8_expected.data(), cuda_i8_output.get(),
                              i8_expected.size() * sizeof(uint16_t),
                              cudaMemcpyDeviceToHost));
  LinearWeightUpload i8_upload;
  i8_upload.format = LinearWeightFormat::kInt8;
  i8_upload.out_features = i8_rows;
  i8_upload.in_features = i8_columns;
  i8_upload.data = i8.data();
  i8_upload.data_bytes = i8.size();
  i8_upload.weight_scale = i8_scale;
  i8_upload.weight_scale_count = i8_rows;
  exact_case(i8_upload, i8_expected);

  // Native dense formats are immutable checkpoint payloads. Every possible
  // 16-bit pattern, including all NaN payloads and subnormals, must survive a
  // same-format materialization without a floating-point round trip.
  auto raw_dense_case = [&](LinearWeightFormat format, ScalarType type) {
    std::vector<uint16_t> patterns(65536);
    for (uint32_t i = 0; i < patterns.size(); ++i)
      patterns[i] = static_cast<uint16_t>(i);
    LinearWeightUpload upload;
    upload.format = format;
    upload.out_features = 256;
    upload.in_features = 256;
    upload.data = patterns.data();
    upload.data_bytes = patterns.size() * sizeof(uint16_t);
    LinearWeight weight = LinearWeight::upload(vk, upload);
    const uint64_t shape[] = {256, 256};
    DeviceTensor dense = vk.allocate(TensorLayout::contiguous(shape, 2), type);
    TensorBatch batch = vk.begin_batch();
    if (type == ScalarType::kFloat16) weight.materialize_f16(batch, dense);
    else weight.materialize_bf16(batch, dense);
    batch.submit().wait();
    std::vector<uint16_t> actual(patterns.size());
    vk.download_bytes(dense, actual.data(), actual.size() * sizeof(uint16_t));
    CHECK(std::memcmp(patterns.data(), actual.data(),
                      patterns.size() * sizeof(uint16_t)) == 0);
  };
  raw_dense_case(LinearWeightFormat::kFloat16, ScalarType::kFloat16);
  raw_dense_case(LinearWeightFormat::kBFloat16, ScalarType::kBFloat16);

  auto cross_dense_case = [&](LinearWeightFormat format, ScalarType source_type,
                              const void* input, uint64_t count,
                              bool output_fp16) {
    const uint64_t input_bytes = count *
        (source_type == ScalarType::kFloat32 ? 4ull : 2ull);
    cuda::DeviceBuffer<uint8_t> cuda_input(input_bytes);
    cuda::DeviceBuffer<uint16_t> cuda_output(count);
    cuda_input.copy_from_host(static_cast<const uint8_t*>(input), input_bytes);
    const int source_op = source_type == ScalarType::kFloat32 ? 0
        : source_type == ScalarType::kFloat16 ? 1 : 2;
    dense_weight_convert_probe<<<static_cast<unsigned>((count + 255) / 256), 256>>>(
        cuda_input.get(), cuda_output.get(), static_cast<int>(count), source_op,
        output_fp16);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint16_t> expected(count), actual(count);
    cuda_output.copy_to_host(expected.data(), count);
    LinearWeightUpload upload;
    upload.format = format;
    upload.out_features = 1;
    upload.in_features = static_cast<uint32_t>(count);
    upload.data = input;
    upload.data_bytes = input_bytes;
    LinearWeight weight = LinearWeight::upload(vk, upload);
    const uint64_t shape[] = {1, count};
    DeviceTensor dense = vk.allocate(TensorLayout::contiguous(shape, 2),
        output_fp16 ? ScalarType::kFloat16 : ScalarType::kBFloat16);
    TensorBatch batch = vk.begin_batch();
    if (output_fp16) weight.materialize_f16(batch, dense);
    else weight.materialize_bf16(batch, dense);
    batch.submit().wait();
    vk.download_bytes(dense, actual.data(), count * sizeof(uint16_t));
    CHECK(std::memcmp(expected.data(), actual.data(), count * 2) == 0);
  };
  std::vector<uint16_t> all_half(65536), all_bf16(65536);
  for (uint32_t i = 0; i < 65536; ++i) {
    all_half[i] = static_cast<uint16_t>(i);
    all_bf16[i] = static_cast<uint16_t>(i);
  }
  cross_dense_case(LinearWeightFormat::kFloat16, ScalarType::kFloat16,
                   all_half.data(), all_half.size(), false);
  cross_dense_case(LinearWeightFormat::kBFloat16, ScalarType::kBFloat16,
                   all_bf16.data(), all_bf16.size(), true);
  std::vector<uint32_t> f32_bits(4099);
  uint32_t state = 0x31415926u;
  for (size_t i = 0; i < f32_bits.size(); ++i) {
    state = state * 1664525u + 1013904223u;
    f32_bits[i] = state;
  }
  const uint32_t special_f32[] = {0u, 0x80000000u, 1u, 0x007fffffu,
      0x00800000u, 0x7f7fffffu, 0x7f800000u, 0xff800000u, 0x7fc12345u};
  std::copy(std::begin(special_f32), std::end(special_f32), f32_bits.begin());
  cross_dense_case(LinearWeightFormat::kFloat32, ScalarType::kFloat32,
                   f32_bits.data(), f32_bits.size(), false);
  cross_dense_case(LinearWeightFormat::kFloat32, ScalarType::kFloat32,
                   f32_bits.data(), f32_bits.size(), true);
}

VIDFAB_TEST(cuda_vulkan_linear_weight_nvfp4_nf4_exact) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);

  auto materialize = [&](const LinearWeightUpload& upload, bool fp16) {
    LinearWeight weight = LinearWeight::upload(vk, upload);
    const uint64_t shape[] = {upload.out_features, upload.in_features};
    DeviceTensor dense = vk.allocate(TensorLayout::contiguous(shape, 2),
        fp16 ? ScalarType::kFloat16 : ScalarType::kBFloat16);
    TensorBatch batch = vk.begin_batch();
    if (fp16) weight.materialize_f16(batch, dense);
    else weight.materialize_bf16(batch, dense);
    batch.submit().wait();
    std::vector<uint16_t> result(
        static_cast<size_t>(upload.out_features) * upload.in_features);
    vk.download_bytes(dense, result.data(), result.size() * sizeof(uint16_t));
    return result;
  };

  // Three row tiles by five contraction tiles catches both physical scale-tile
  // strides; the previous one-tile shape could not distinguish a flat layout.
  constexpr uint32_t nv_out = 384, nv_in = 320;
  constexpr size_t nv_count = static_cast<size_t>(nv_out) * nv_in;
  std::vector<uint8_t> nv_packed((nv_count + 1) / 2);
  for (size_t i = 0; i < nv_packed.size(); ++i) {
    nv_packed[i] = static_cast<uint8_t>(((2 * i & 15u) << 4u) |
                                        ((2 * i + 1u) & 15u));
  }
  std::vector<uint8_t> nv_scales(nv_count / 16);
  for (size_t i = 0; i < nv_scales.size(); ++i) {
    // Covers every E4M3 bit pattern twice in the checkpoint's already-swizzled
    // physical scale array.
    nv_scales[i] = static_cast<uint8_t>(i);
  }
  const float nv_global = 1.3580322e-3f;
  cuda::DeviceBuffer<uint8_t> d_nv(nv_packed.size()), d_nv_scales(nv_scales.size());
  cuda::DeviceBuffer<__nv_bfloat16> d_nv_output(nv_count);
  d_nv.copy_from_host(nv_packed.data(), nv_packed.size());
  d_nv_scales.copy_from_host(nv_scales.data(), nv_scales.size());
  cuda::launch_dequant_nvfp4(d_nv.get(), d_nv_scales.get(), nv_global,
                             d_nv_output.get(), nv_out, nv_in, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> nv_expected(nv_count);
  VIDFAB_CUDA_CHECK(cudaMemcpy(nv_expected.data(), d_nv_output.get(),
                              nv_count * sizeof(uint16_t), cudaMemcpyDeviceToHost));
  LinearWeightUpload nv_upload;
  nv_upload.format = LinearWeightFormat::kNVFloat4;
  nv_upload.out_features = nv_out;
  nv_upload.in_features = nv_in;
  nv_upload.data = nv_packed.data();
  nv_upload.data_bytes = nv_packed.size();
  nv_upload.block_scale = nv_scales.data();
  nv_upload.block_scale_count = nv_scales.size();
  nv_upload.global_scale = nv_global;
  const auto nv_actual = materialize(nv_upload, false);
  CHECK(std::memcmp(nv_expected.data(), nv_actual.data(),
                    nv_count * sizeof(uint16_t)) == 0);

  constexpr uint32_t nf_out = 129, nf_in = 129;
  constexpr size_t nf_count = static_cast<size_t>(nf_out) * nf_in;
  constexpr uint32_t block = 64, nested_block = 256;
  const size_t nf_blocks = 1 + (nf_count - 1) / block;
  const size_t nf_nested_blocks = 1 + (nf_blocks - 1) / nested_block;
  std::vector<uint8_t> nf_packed((nf_count + 1) / 2), nf_absmax(nf_blocks);
  for (size_t i = 0; i < nf_packed.size(); ++i) {
    nf_packed[i] = static_cast<uint8_t>((((i * 5 + 3) & 15) << 4) |
                                        ((i * 11 + 9) & 15));
  }
  for (size_t i = 0; i < nf_absmax.size(); ++i)
    nf_absmax[i] = static_cast<uint8_t>((i * 73 + 19) & 255);
  const std::array<float, 16> nf_map = {
      -1.0f, -0.6961928f, -0.52507305f, -0.39491749f,
      -0.28444138f, -0.18477343f, -0.09105004f, 0.0f,
       0.07958030f, 0.16093020f, 0.24611230f, 0.33791524f,
       0.44070983f, 0.56261700f, 0.72295684f, 1.0f};
  std::array<float, 256> nf_nested_map{};
  for (size_t i = 0; i < nf_nested_map.size(); ++i)
    nf_nested_map[i] = (static_cast<float>(i) - 127.0f) / 128.0f;
  std::vector<float> nf_nested_absmax(nf_nested_blocks);
  for (size_t i = 0; i < nf_nested_absmax.size(); ++i)
    nf_nested_absmax[i] = 0.75f + static_cast<float>(i) * 1.25f;
  const float nf_offset = 0.21360844373703003f;
  cuda::DeviceBuffer<uint8_t> d_nf(nf_packed.size()), d_nf_absmax(nf_absmax.size());
  cuda::DeviceBuffer<float> d_nf_map(nf_map.size()),
      d_nf_nested_map(nf_nested_map.size()),
      d_nf_nested_absmax(nf_nested_absmax.size());
  cuda::DeviceBuffer<uint16_t> d_nf_bf16(nf_count), d_nf_f16(nf_count);
  d_nf.copy_from_host(nf_packed.data(), nf_packed.size());
  d_nf_absmax.copy_from_host(nf_absmax.data(), nf_absmax.size());
  d_nf_map.copy_from_host(nf_map.data(), nf_map.size());
  d_nf_nested_map.copy_from_host(nf_nested_map.data(), nf_nested_map.size());
  d_nf_nested_absmax.copy_from_host(nf_nested_absmax.data(), nf_nested_absmax.size());
  cuda::launch_dequant_nf4(d_nf.get(), d_nf_absmax.get(), d_nf_map.get(),
      d_nf_nested_map.get(), d_nf_nested_absmax.get(), block, nested_block,
      nf_offset, reinterpret_cast<__nv_bfloat16*>(d_nf_bf16.get()), nf_out,
      nf_in, nullptr);
  cuda::launch_dequant_nf4_f16(d_nf.get(), d_nf_absmax.get(), d_nf_map.get(),
      d_nf_nested_map.get(), d_nf_nested_absmax.get(), block, nested_block,
      nf_offset, reinterpret_cast<__half*>(d_nf_f16.get()), nf_count, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> nf_bf16_expected(nf_count), nf_f16_expected(nf_count);
  d_nf_bf16.copy_to_host(nf_bf16_expected.data(), nf_count);
  d_nf_f16.copy_to_host(nf_f16_expected.data(), nf_count);
  LinearWeightUpload nf_upload;
  nf_upload.format = LinearWeightFormat::kNF4;
  nf_upload.out_features = nf_out;
  nf_upload.in_features = nf_in;
  nf_upload.data = nf_packed.data();
  nf_upload.data_bytes = nf_packed.size();
  nf_upload.nf4_absmax = nf_absmax.data();
  nf_upload.nf4_absmax_count = nf_absmax.size();
  nf_upload.nf4_quant_map = nf_map.data();
  nf_upload.nf4_quant_map_count = nf_map.size();
  nf_upload.nf4_nested_quant_map = nf_nested_map.data();
  nf_upload.nf4_nested_quant_map_count = nf_nested_map.size();
  nf_upload.nf4_nested_absmax = nf_nested_absmax.data();
  nf_upload.nf4_nested_absmax_count = nf_nested_absmax.size();
  nf_upload.nf4_block_size = block;
  nf_upload.nf4_nested_block_size = nested_block;
  nf_upload.nf4_nested_offset = nf_offset;
  const uint64_t before_invalid = vk.pooled_used_bytes();
  {
    LinearWeightUpload invalid = nf_upload;
    invalid.nf4_quant_map_count = 15;
    bool rejected = false;
    try { (void)LinearWeight::upload(vk, invalid); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    CHECK(vk.pooled_used_bytes() == before_invalid);
  }
  {
    LinearWeightUpload invalid = nf_upload;
    invalid.nf4_block_size = 63;
    bool rejected = false;
    try { (void)LinearWeight::upload(vk, invalid); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    CHECK(vk.pooled_used_bytes() == before_invalid);
  }
  {
    LinearWeightUpload invalid = nf_upload;
    invalid.nf4_nested_quant_map_count = 255;
    bool rejected = false;
    try { (void)LinearWeight::upload(vk, invalid); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    CHECK(vk.pooled_used_bytes() == before_invalid);
  }
  const auto nf_bf16_actual = materialize(nf_upload, false);
  const auto nf_f16_actual = materialize(nf_upload, true);
  CHECK(std::memcmp(nf_bf16_expected.data(), nf_bf16_actual.data(),
                    nf_count * sizeof(uint16_t)) == 0);
  CHECK(std::memcmp(nf_f16_expected.data(), nf_f16_actual.data(),
                    nf_count * sizeof(uint16_t)) == 0);
}

VIDFAB_TEST(cuda_vulkan_linear_weight_activation_transforms_exact) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);

  constexpr uint32_t rows = 3, dim = 256;
  constexpr size_t count = static_cast<size_t>(rows) * dim;
  std::vector<float> values(count), scales(dim);
  for (size_t i = 0; i < count; ++i)
    values[i] = static_cast<float>(static_cast<int>(i * 37 % 257) - 128) / 64.0f;
  for (size_t i = 0; i < dim; ++i)
    scales[i] = static_cast<float>(static_cast<int>(i * 19 % 61) - 30) / 32.0f;
  std::vector<uint16_t> bf_values(count), bf_scales(dim), zero_weight(dim);
  for (size_t i = 0; i < count; ++i) bf_values[i] = f32_to_bf16(values[i]);
  for (size_t i = 0; i < dim; ++i) bf_scales[i] = f32_to_bf16(scales[i]);

  LinearWeightUpload upload;
  upload.format = LinearWeightFormat::kBFloat16;
  upload.out_features = 1;
  upload.in_features = dim;
  upload.data = zero_weight.data();
  upload.data_bytes = zero_weight.size() * sizeof(uint16_t);
  upload.pre_quant_scale_bf16 = bf_scales.data();
  upload.pre_quant_scale_count = bf_scales.size();
  upload.convrot = true;
  upload.convrot_group = dim;
  LinearWeight weight = LinearWeight::upload(vk, upload);

  cuda::DeviceBuffer<uint16_t> d_bf_input(count), d_scale(dim),
      d_bf_scaled(count), d_bf_rotated(count);
  cuda::DeviceBuffer<float> d_f32_input(count), d_f32_scaled(count),
      d_f32_rotated(count);
  d_bf_input.copy_from_host(bf_values.data(), count);
  d_scale.copy_from_host(bf_scales.data(), dim);
  d_f32_input.copy_from_host(values.data(), count);
  cuda::launch_pre_quant_scale(
      reinterpret_cast<const __nv_bfloat16*>(d_bf_input.get()),
      reinterpret_cast<const __nv_bfloat16*>(d_scale.get()),
      reinterpret_cast<__nv_bfloat16*>(d_bf_scaled.get()), rows, dim, nullptr);
  fp32_pre_quant_scale_probe<<<static_cast<unsigned>((count + 255) / 256), 256>>>(
      d_f32_input.get(), reinterpret_cast<const __nv_bfloat16*>(d_scale.get()),
      d_f32_scaled.get(), static_cast<int>(count), dim);
  cuda::launch_convrot(
      reinterpret_cast<const __nv_bfloat16*>(d_bf_input.get()),
      reinterpret_cast<__nv_bfloat16*>(d_bf_rotated.get()), rows, dim, dim,
      nullptr);
  cuda::launch_convrot_f32(d_f32_input.get(), d_f32_rotated.get(), rows, dim,
                           dim, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> bf_scaled_expected(count), bf_rotated_expected(count);
  std::vector<float> f32_scaled_expected(count), f32_rotated_expected(count);
  d_bf_scaled.copy_to_host(bf_scaled_expected.data(), count);
  d_bf_rotated.copy_to_host(bf_rotated_expected.data(), count);
  d_f32_scaled.copy_to_host(f32_scaled_expected.data(), count);
  d_f32_rotated.copy_to_host(f32_rotated_expected.data(), count);

  const uint64_t shape[] = {rows, dim};
  DeviceTensor bf_input = vk.allocate(TensorLayout::contiguous(shape, 2),
                                      ScalarType::kBFloat16);
  DeviceTensor bf_scaled = vk.allocate(TensorLayout::contiguous(shape, 2),
                                       ScalarType::kBFloat16);
  DeviceTensor bf_rotated = vk.allocate(TensorLayout::contiguous(shape, 2),
                                        ScalarType::kBFloat16);
  DeviceTensor f32_input = vk.allocate(TensorLayout::contiguous(shape, 2));
  DeviceTensor f32_scaled = vk.allocate(TensorLayout::contiguous(shape, 2));
  DeviceTensor f32_rotated = vk.allocate(TensorLayout::contiguous(shape, 2));
  vk.upload_bytes(bf_input, bf_values.data(), bf_values.size() * 2);
  vk.upload(f32_input, values.data(), values.size());
  TensorBatch batch = vk.begin_batch();
  weight.apply_pre_quant_scale(batch, bf_input, bf_scaled);
  weight.apply_pre_quant_scale(batch, f32_input, f32_scaled);
  weight.apply_convrot(batch, bf_input, bf_rotated);
  weight.apply_convrot(batch, f32_input, f32_rotated);
  batch.submit().wait();
  std::vector<uint16_t> bf_scaled_actual(count), bf_rotated_actual(count);
  std::vector<float> f32_scaled_actual(count), f32_rotated_actual(count);
  vk.download_bytes(bf_scaled, bf_scaled_actual.data(), count * 2);
  vk.download_bytes(bf_rotated, bf_rotated_actual.data(), count * 2);
  vk.download(f32_scaled, f32_scaled_actual.data(), count);
  vk.download(f32_rotated, f32_rotated_actual.data(), count);
  CHECK(std::memcmp(bf_scaled_expected.data(), bf_scaled_actual.data(), count * 2) == 0);
  CHECK(std::memcmp(bf_rotated_expected.data(), bf_rotated_actual.data(), count * 2) == 0);
  CHECK(std::memcmp(f32_scaled_expected.data(), f32_scaled_actual.data(), count * 4) == 0);
  CHECK(std::memcmp(f32_rotated_expected.data(), f32_rotated_actual.data(), count * 4) == 0);
}

VIDFAB_TEST(vulkan_linear_weight_bounded_reuse_and_lifetime) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  constexpr uint32_t out = 3, in = 131;
  constexpr size_t count = static_cast<size_t>(out) * in;
  std::vector<int8_t> codes(count);
  for (size_t i = 0; i < count; ++i) codes[i] = static_cast<int8_t>(i * 29u);
  const float scales[] = {0.5f, -0.25f, 1.5f};
  LinearWeightUpload upload;
  upload.format = LinearWeightFormat::kInt8;
  upload.out_features = out;
  upload.in_features = in;
  upload.data = codes.data();
  upload.data_bytes = codes.size();
  upload.weight_scale = scales;
  upload.weight_scale_count = out;
  const uint64_t shape[] = {out, in};

  LinearWeight weight = LinearWeight::upload(vk, upload);
  DeviceTensor first = vk.allocate(TensorLayout::contiguous(shape, 2),
                                   ScalarType::kBFloat16);
  DeviceTensor second = vk.allocate(TensorLayout::contiguous(shape, 2),
                                    ScalarType::kBFloat16);
  auto submit = [&](DeviceTensor& output) {
    TensorBatch batch = vk.begin_batch();
    weight.materialize_bf16(batch, output);
    return batch.submit();
  };
  Submission warm_a = submit(first), warm_b = submit(second);
  warm_a.wait(); warm_b.wait();
  { TensorBatch collect = vk.begin_batch(); }
  const uint64_t stable_reserved = vk.reserved_bytes();
  const uint64_t stable_descriptors = vk.descriptor_set_allocations();
  for (int repeat = 0; repeat < 6; ++repeat) {
    Submission a = submit(first), b = submit(second), c = submit(first);
    CHECK(b.value() > a.value() && c.value() > b.value());
    a.wait(); b.wait(); c.wait();
    CHECK(vk.reserved_bytes() == stable_reserved);
    CHECK(vk.descriptor_set_allocations() == stable_descriptors);
  }
  {
    TensorBatch full = vk.begin_batch();
    for (int op = 0; op < 32; ++op) weight.materialize_bf16(full, first);
    full.submit().wait();
  }
  {
    TensorBatch overflow = vk.begin_batch();
    for (int op = 0; op < 32; ++op) weight.materialize_bf16(overflow, first);
    bool rejected = false;
    try { weight.materialize_bf16(overflow, first); }
    catch (const std::logic_error&) { rejected = true; }
    CHECK(rejected);
    bool submit_rejected = false;
    try { (void)overflow.submit(); }
    catch (const std::logic_error&) { submit_rejected = true; }
    CHECK(submit_rejected);
  }
  CHECK(vk.reserved_bytes() == stable_reserved);
  const uint64_t saturated_descriptors = vk.descriptor_set_allocations();
  CHECK(saturated_descriptors >= stable_descriptors);
  for (int repeat = 0; repeat < 2; ++repeat) {
    TensorBatch full = vk.begin_batch();
    for (int op = 0; op < 32; ++op) weight.materialize_bf16(full, first);
    full.submit().wait();
    CHECK(vk.reserved_bytes() == stable_reserved);
    CHECK(vk.descriptor_set_allocations() == saturated_descriptors);
  }

  auto valid_after_rejection = [&](auto&& invalid) {
    TensorBatch batch = vk.begin_batch();
    bool rejected = false;
    try { invalid(batch); } catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    weight.materialize_bf16(batch, first);
    batch.submit().wait();
  };
  DeviceTensor wrong_type = vk.allocate(TensorLayout::contiguous(shape, 2),
                                        ScalarType::kFloat32);
  const uint64_t short_shape[] = {out, in - 1};
  DeviceTensor wrong_shape = vk.allocate(TensorLayout::contiguous(short_shape, 2),
                                         ScalarType::kBFloat16);
  valid_after_rejection([&](TensorBatch& batch) {
    weight.materialize_bf16(batch, wrong_type);
  });
  valid_after_rejection([&](TensorBatch& batch) {
    weight.materialize_bf16(batch, wrong_shape);
  });
  Device other_device = physical.front().create_device(options);
  TensorContext other(other_device);
  DeviceTensor foreign = other.allocate(TensorLayout::contiguous(shape, 2),
                                        ScalarType::kBFloat16);
  valid_after_rejection([&](TensorBatch& batch) {
    weight.materialize_bf16(batch, foreign);
  });

  // Discarded recording releases its speculative references immediately.
  const uint64_t used_before_discard = vk.pooled_used_bytes();
  {
    LinearWeight temporary = LinearWeight::upload(vk, upload);
    DeviceTensor output = vk.allocate(TensorLayout::contiguous(shape, 2),
                                      ScalarType::kBFloat16);
    { TensorBatch discarded = vk.begin_batch();
      temporary.materialize_bf16(discarded, output); }
  }
  CHECK(vk.pooled_used_bytes() == used_before_discard);

  // A submitted job retains compressed storage, scales, and dense output even
  // after every public wrapper drops, then releases them after its exact token
  // and slot collection.
  const uint64_t used_before_submit = vk.pooled_used_bytes();
  Submission retained;
  {
    LinearWeight temporary = LinearWeight::upload(vk, upload);
    DeviceTensor output = vk.allocate(TensorLayout::contiguous(shape, 2),
                                      ScalarType::kBFloat16);
    TensorBatch batch = vk.begin_batch();
    temporary.materialize_bf16(batch, output);
    retained = batch.submit();
  }
  CHECK(vk.pooled_used_bytes() > used_before_submit);
  retained.wait();
  retained = Submission{};
  { TensorBatch collect = vk.begin_batch(); }
  CHECK(vk.pooled_used_bytes() == used_before_submit);
}

VIDFAB_TEST(cuda_vulkan_linear_weight_real_nvfp4_slab) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  const std::filesystem::path path =
      "weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors";
  int cuda_devices = 0;
  if (!std::filesystem::exists(path) ||
      cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  SafeTensors checkpoint;
  checkpoint.open(path.string());
  const std::string prefix = "blocks.0.attn.qkv_proj";
  const TensorView& stored = checkpoint.at(prefix + ".weight");
  const TensorView& scale = checkpoint.at(prefix + ".weight_scale");
  const TensorView& global_view = checkpoint.at(prefix + ".weight_scale_2");
  CHECK(stored.shape.size() == 2);
  constexpr uint32_t slab_out = 384;
  const uint32_t full_out = static_cast<uint32_t>(stored.shape[0]);
  const uint32_t in = static_cast<uint32_t>(stored.shape[1] * 2);
  CHECK(full_out >= slab_out && full_out % 128 == 0 && in % 64 == 0);
  const size_t elements = static_cast<size_t>(slab_out) * in;
  const size_t stored_bytes = elements / 2;
  const size_t scale_bytes = elements / 16;
  CHECK(stored.nbytes >= stored_bytes && scale.nbytes >= scale_bytes);
  float global = 0.0f;
  std::memcpy(&global, global_view.data, sizeof(global));

  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  LinearWeightUpload upload;
  upload.format = LinearWeightFormat::kNVFloat4;
  upload.out_features = slab_out;
  upload.in_features = in;
  upload.data = stored.data;
  upload.data_bytes = stored_bytes;
  upload.block_scale = static_cast<const uint8_t*>(scale.data);
  upload.block_scale_count = scale_bytes;
  upload.global_scale = global;
  const auto upload_begin = std::chrono::steady_clock::now();
  LinearWeight weight = LinearWeight::upload(vk, upload);
  const double upload_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - upload_begin).count();
  CHECK(weight.stored_bytes() == stored_bytes);
  CHECK(weight.resident_bytes() == stored_bytes + scale_bytes);
  const uint64_t shape[] = {slab_out, in};
  DeviceTensor dense = vk.allocate(TensorLayout::contiguous(shape, 2),
                                   ScalarType::kBFloat16);

  cuda::DeviceBuffer<uint8_t> cuda_stored(stored_bytes), cuda_scale(scale_bytes);
  cuda::DeviceBuffer<uint16_t> cuda_dense(elements);
  cuda_stored.copy_from_host(static_cast<const uint8_t*>(stored.data), stored_bytes);
  cuda_scale.copy_from_host(static_cast<const uint8_t*>(scale.data), scale_bytes);
  cuda::launch_dequant_nvfp4(cuda_stored.get(), cuda_scale.get(), global,
      reinterpret_cast<__nv_bfloat16*>(cuda_dense.get()), slab_out, in, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  {
    TensorBatch batch = vk.begin_batch();
    weight.materialize_bf16(batch, dense);
    batch.submit().wait();
  }
  std::vector<uint16_t> cuda_bits(elements), vulkan_bits(elements);
  cuda_dense.copy_to_host(cuda_bits.data(), elements);
  vk.download_bytes(dense, vulkan_bits.data(), elements * sizeof(uint16_t));
  CHECK(std::memcmp(cuda_bits.data(), vulkan_bits.data(),
                    elements * sizeof(uint16_t)) == 0);

  constexpr int iterations = 20;
  auto cuda_begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    cuda::launch_dequant_nvfp4(cuda_stored.get(), cuda_scale.get(), global,
        reinterpret_cast<__nv_bfloat16*>(cuda_dense.get()), slab_out, in, nullptr);
  }
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const double cuda_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - cuda_begin).count() / iterations;
  auto vulkan_begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    TensorBatch batch = vk.begin_batch();
    weight.materialize_bf16(batch, dense);
    batch.submit().wait();
  }
  const double vulkan_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - vulkan_begin).count() / iterations;
  std::printf("  real NVFP4 slab %ux%u: upload %.3f ms, CUDA %.3f ms, "
              "Vulkan %.3f ms, persistent %.2f MiB, dense %.2f MiB\n",
              slab_out, in, upload_ms, cuda_ms, vulkan_ms,
              static_cast<double>(weight.resident_bytes()) / 1048576.0,
              static_cast<double>(elements * 2) / 1048576.0);
}

VIDFAB_TEST(cuda_vulkan_streamed_nvfp4_gemm_real_slab) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  const std::filesystem::path path =
      "weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors";
  int cuda_devices = 0;
  if (!std::filesystem::exists(path) ||
      cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore ||
      !physical.front().info().cooperative_matrix_bf16_f32_16x16x16) return;
  SafeTensors checkpoint;
  checkpoint.open(path.string());
  const std::string prefix = "blocks.0.attn.qkv_proj";
  const TensorView& stored = checkpoint.at(prefix + ".weight");
  const TensorView& scale = checkpoint.at(prefix + ".weight_scale");
  const TensorView& global_view = checkpoint.at(prefix + ".weight_scale_2");
  constexpr uint32_t n = 384, k = 5376, rows = 66, output_rows = 68;
  const size_t weight_elements = size_t(n) * k;
  const size_t stored_bytes = weight_elements / 2;
  const size_t scale_bytes = weight_elements / 16;
  CHECK(stored.nbytes >= stored_bytes && scale.nbytes >= scale_bytes);
  float global = 0.0f;
  std::memcpy(&global, global_view.data, sizeof(global));

  std::vector<uint16_t> input(size_t(rows) * k), bias(n);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = f32_to_bf16(float(int(i % 29) - 14) / 32.0f);
  for (uint32_t i = 0; i < n; ++i)
    bias[i] = f32_to_bf16(float(int(i % 17) - 8) / 64.0f);
  constexpr uint16_t sentinel = 0x7fc1;
  std::vector<uint16_t> initial(size_t(output_rows) * n, sentinel);

  cuda::DeviceBuffer<uint8_t> cw(stored_bytes), cs(scale_bytes);
  cuda::DeviceBuffer<uint16_t> cdense(weight_elements), ci(input.size()),
      cbias(bias.size()), co(initial.size());
  cw.copy_from_host(static_cast<const uint8_t*>(stored.data), stored_bytes);
  cs.copy_from_host(static_cast<const uint8_t*>(scale.data), scale_bytes);
  ci.copy_from_host(input.data(), input.size());
  cbias.copy_from_host(bias.data(), bias.size());
  co.copy_from_host(initial.data(), initial.size());
  auto cuda_run = [&] {
    cuda::launch_dequant_nvfp4(
        cw.get(), cs.get(), global,
        reinterpret_cast<__nv_bfloat16*>(cdense.get()), n, k, nullptr);
    cuda::launch_deterministic_bf16_gemm_nt(
        reinterpret_cast<const __nv_bfloat16*>(ci.get()),
        reinterpret_cast<const __nv_bfloat16*>(cdense.get()), cbias.get(),
        reinterpret_cast<__nv_bfloat16*>(co.get()), 64, n, k,
        DenseGemmBias::kBFloat16);
    cuda::launch_deterministic_scalar_gemm_nt(
        ci.get(), cdense.get(), cbias.get(), co.get(), 2, n, k,
        DenseGemmMode::kBFloat16, DenseGemmBias::kBFloat16, 64, 64);
  };
  cuda_run();
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> cuda_output(initial.size());
  co.copy_to_host(cuda_output.data(), cuda_output.size());

  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_cooperative_matrix = true;
  options.enable_storage_buffer_16bit = true;
  Device device = physical.front().create_device(options);
  TensorContext context(device);
  LinearWeightUpload upload;
  upload.format = LinearWeightFormat::kNVFloat4;
  upload.out_features = n;
  upload.in_features = k;
  upload.data = stored.data;
  upload.data_bytes = stored_bytes;
  upload.block_scale = static_cast<const uint8_t*>(scale.data);
  upload.block_scale_count = scale_bytes;
  upload.global_scale = global;
  LinearWeight weight = LinearWeight::upload(context, upload);
  const uint64_t input_shape[] = {rows, k}, output_shape[] = {output_rows, n};
  const uint64_t bias_shape[] = {n};
  DeviceTensor vi = context.allocate(TensorLayout::contiguous(input_shape, 2),
                                     ScalarType::kBFloat16);
  DeviceTensor vo = context.allocate(TensorLayout::contiguous(output_shape, 2),
                                     ScalarType::kBFloat16);
  DeviceTensor vb = context.allocate(TensorLayout::contiguous(bias_shape, 1),
                                     ScalarType::kBFloat16);
  context.upload_bytes(vi, input.data(), input.size() * 2);
  context.upload_bytes(vo, initial.data(), initial.size() * 2);
  context.upload_bytes(vb, bias.data(), bias.size() * 2);
  DenseGemmPlan plan = DenseGemmPlan::create(
      context, {64, n, k, DenseGemmMode::kBFloat16,
                DenseGemmBias::kBFloat16});
  StreamedNVFP4WeightCache cache =
      StreamedNVFP4WeightCache::create(context, weight_elements);
  auto vulkan_run = [&] {
    TensorBatch batch = context.begin_batch();
    PreparedNVFP4WeightView prepared = cache.prepare(batch, weight, plan);
    plan.record(batch, vi, prepared, vo, 64, 0, 0, &vb);
    plan.record(batch, vi, prepared, vo, 2, 64, 64, &vb);
    return batch.submit();
  };
  vulkan_run().wait();
  std::vector<uint16_t> vulkan_output(initial.size());
  context.download_bytes(vo, vulkan_output.data(), vulkan_output.size() * 2);
  size_t mismatch = vulkan_output.size();
  for (size_t i = 0; i < vulkan_output.size(); ++i) {
    if (cuda_output[i] != vulkan_output[i]) { mismatch = i; break; }
  }
  CHECK_MSG(mismatch == vulkan_output.size(),
            "streamed real NVFP4 GEMM mismatch at %zu: %04x != %04x",
            mismatch, mismatch == vulkan_output.size() ? 0u : cuda_output[mismatch],
            mismatch == vulkan_output.size() ? 0u : vulkan_output[mismatch]);

  constexpr int iterations = 10;
  cudaEvent_t begin = nullptr, end = nullptr;
  VIDFAB_CUDA_CHECK(cudaEventCreate(&begin));
  VIDFAB_CUDA_CHECK(cudaEventCreate(&end));
  VIDFAB_CUDA_CHECK(cudaEventRecord(begin));
  for (int i = 0; i < iterations; ++i) cuda_run();
  VIDFAB_CUDA_CHECK(cudaEventRecord(end));
  VIDFAB_CUDA_CHECK(cudaEventSynchronize(end));
  float cuda_elapsed = 0.0f;
  VIDFAB_CUDA_CHECK(cudaEventElapsedTime(&cuda_elapsed, begin, end));
  cudaEventDestroy(begin); cudaEventDestroy(end);
  const auto vk_begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) vulkan_run().wait();
  const double vulkan_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - vk_begin).count() / iterations;
  std::printf("  streamed real NVFP4 dequant+GEMM %ux%ux%u: CUDA %.3f ms, "
              "Vulkan %.3f ms, compressed %.2f MiB, one dense slot %.2f MiB\n",
              rows, n, k, cuda_elapsed / iterations, vulkan_ms,
              double(weight.resident_bytes()) / 1048576.0,
              double(cache.dense_bytes()) / 1048576.0);
}

VIDFAB_TEST(cuda_vulkan_linear_weight_real_nf4_conv) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  const std::filesystem::path path = "weights/vae/video_vae_nf4.safetensors";
  int cuda_devices = 0;
  if (!std::filesystem::exists(path) ||
      cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  SafeTensors checkpoint;
  checkpoint.open(path.string());
  constexpr std::string_view state_suffix =
      ".quant_state.bitsandbytes__nf4";
  std::string name;
  uint64_t largest_elements = 0;
  for (const auto& entry : checkpoint.tensors()) {
    const std::string& candidate = entry.first;
    if (candidate.size() <= state_suffix.size() ||
        candidate.compare(candidate.size() - state_suffix.size(),
                          state_suffix.size(), state_suffix) != 0) continue;
    const std::string weight_name =
        candidate.substr(0, candidate.size() - state_suffix.size());
    const NF4State candidate_state =
        read_nf4_state(checkpoint, weight_name, "Vulkan weight test");
    uint64_t candidate_elements = 1;
    for (int64_t extent : candidate_state.shape)
      candidate_elements *= static_cast<uint64_t>(extent);
    if (candidate_elements > largest_elements) {
      largest_elements = candidate_elements;
      name = weight_name;
    }
  }
  CHECK(!name.empty());
  if (name.empty()) return;
  const TensorView& stored = checkpoint.at(name);
  const TensorView& absmax = checkpoint.at(name + ".absmax");
  const TensorView& map = checkpoint.at(name + ".quant_map");
  const TensorView& nested_map = checkpoint.at(name + ".nested_quant_map");
  const TensorView& nested_absmax = checkpoint.at(name + ".nested_absmax");
  const NF4State state = read_nf4_state(checkpoint, name, "Vulkan weight test");
  CHECK(state.shape.size() >= 2 && state.shape[0] > 0);
  uint64_t elements = 1;
  for (int64_t extent : state.shape) elements *= static_cast<uint64_t>(extent);
  const uint32_t out = static_cast<uint32_t>(state.shape[0]);
  const uint32_t in = static_cast<uint32_t>(elements / out);
  CHECK(elements == static_cast<uint64_t>(out) * in && stored.nbytes * 2 == elements);

  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_shader_int64 = physical.front().info().shader_int64;
  Device device = physical.front().create_device(options);
  TensorContext vk(device);
  LinearWeightUpload upload;
  upload.format = LinearWeightFormat::kNF4;
  upload.out_features = out;
  upload.in_features = in;
  upload.data = stored.data;
  upload.data_bytes = stored.nbytes;
  upload.nf4_absmax = static_cast<const uint8_t*>(absmax.data);
  upload.nf4_absmax_count = absmax.nbytes;
  upload.nf4_quant_map = static_cast<const float*>(map.data);
  upload.nf4_quant_map_count = static_cast<uint64_t>(map.numel());
  upload.nf4_nested_quant_map = static_cast<const float*>(nested_map.data);
  upload.nf4_nested_quant_map_count = static_cast<uint64_t>(nested_map.numel());
  upload.nf4_nested_absmax = static_cast<const float*>(nested_absmax.data);
  upload.nf4_nested_absmax_count = static_cast<uint64_t>(nested_absmax.numel());
  upload.nf4_block_size = state.block_size;
  upload.nf4_nested_block_size = state.nested_block_size;
  upload.nf4_nested_offset = state.nested_offset;
  const auto upload_begin = std::chrono::steady_clock::now();
  LinearWeight weight = LinearWeight::upload(vk, upload);
  const double upload_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - upload_begin).count();
  const uint64_t shape[] = {out, in};
  DeviceTensor dense = vk.allocate(TensorLayout::contiguous(shape, 2),
                                   ScalarType::kFloat16);

  cuda::DeviceBuffer<uint8_t> d_stored(stored.nbytes), d_absmax(absmax.nbytes);
  cuda::DeviceBuffer<float> d_map(map.numel()), d_nested_map(nested_map.numel()),
      d_nested_absmax(nested_absmax.numel());
  cuda::DeviceBuffer<uint16_t> d_dense(elements);
  d_stored.copy_from_host(static_cast<const uint8_t*>(stored.data), stored.nbytes);
  d_absmax.copy_from_host(static_cast<const uint8_t*>(absmax.data), absmax.nbytes);
  d_map.copy_from_host(static_cast<const float*>(map.data), map.numel());
  d_nested_map.copy_from_host(static_cast<const float*>(nested_map.data), nested_map.numel());
  d_nested_absmax.copy_from_host(static_cast<const float*>(nested_absmax.data),
                                 nested_absmax.numel());
  cuda::launch_dequant_nf4_f16(d_stored.get(), d_absmax.get(), d_map.get(),
      d_nested_map.get(), d_nested_absmax.get(), state.block_size,
      state.nested_block_size, state.nested_offset,
      reinterpret_cast<__half*>(d_dense.get()), elements, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  {
    TensorBatch batch = vk.begin_batch();
    weight.materialize_f16(batch, dense);
    batch.submit().wait();
  }
  std::vector<uint16_t> cuda_bits(elements), vulkan_bits(elements);
  d_dense.copy_to_host(cuda_bits.data(), elements);
  vk.download_bytes(dense, vulkan_bits.data(), elements * sizeof(uint16_t));
  CHECK(std::memcmp(cuda_bits.data(), vulkan_bits.data(), elements * 2) == 0);

  constexpr int iterations = 10;
  auto cuda_begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    cuda::launch_dequant_nf4_f16(d_stored.get(), d_absmax.get(), d_map.get(),
        d_nested_map.get(), d_nested_absmax.get(), state.block_size,
        state.nested_block_size, state.nested_offset,
        reinterpret_cast<__half*>(d_dense.get()), elements, nullptr);
  }
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const double cuda_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - cuda_begin).count() / iterations;
  auto vulkan_begin = std::chrono::steady_clock::now();
  for (int i = 0; i < iterations; ++i) {
    TensorBatch batch = vk.begin_batch();
    weight.materialize_f16(batch, dense);
    batch.submit().wait();
  }
  const double vulkan_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - vulkan_begin).count() / iterations;
  std::printf("  real NF4 conv %ux%u: upload %.3f ms, CUDA %.3f ms, "
              "Vulkan %.3f ms, persistent %.2f MiB, dense %.2f MiB\n",
              out, in, upload_ms, cuda_ms, vulkan_ms,
              static_cast<double>(weight.resident_bytes()) / 1048576.0,
              static_cast<double>(elements * 2) / 1048576.0);
}

VIDFAB_TEST(cuda_bf16_gemm_5376_baseline) {
  constexpr int m = 64, n = 5376, k = 5376;
  vidfab::cuda::DeviceBuffer<__nv_bfloat16> a(size_t(m) * k), w(size_t(n) * k), c(size_t(m) * n);
  VIDFAB_CUDA_CHECK(cudaMemset(a.get(), 0, size_t(m) * k * sizeof(__nv_bfloat16)));
  VIDFAB_CUDA_CHECK(cudaMemset(w.get(), 0, size_t(n) * k * sizeof(__nv_bfloat16)));
  cublasHandle_t handle = nullptr;
  VIDFAB_CUBLAS_CHECK(cublasCreate(&handle));
  const float alpha = 1.0f, beta = 0.0f;
  auto launch = [&] {
    VIDFAB_CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
                                     &alpha, w.get(), CUDA_R_16BF, k,
                                     a.get(), CUDA_R_16BF, k, &beta,
                                     c.get(), CUDA_R_16BF, n,
                                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  };
  launch();
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t begin = nullptr, end = nullptr;
  VIDFAB_CUDA_CHECK(cudaEventCreate(&begin)); VIDFAB_CUDA_CHECK(cudaEventCreate(&end));
  VIDFAB_CUDA_CHECK(cudaEventRecord(begin));
  for (int i = 0; i < 20; ++i) launch();
  VIDFAB_CUDA_CHECK(cudaEventRecord(end)); VIDFAB_CUDA_CHECK(cudaEventSynchronize(end));
  float elapsed = 0.0f; VIDFAB_CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
  std::printf("  cuBLAS BF16 GEMM 64x5376x5376: %.3f ms\n", elapsed / 20.0f);
  cudaEventDestroy(begin); cudaEventDestroy(end); cublasDestroy(handle);
}

VIDFAB_TEST(cuda_vulkan_cooperative_bf16_gemm_exact) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  constexpr uint32_t m = 64, n = 16, k = 5376;
  constexpr uint32_t input_rows = 128, output_rows = 128;
  constexpr uint32_t input_offset = 64, output_offset = 17;
  std::vector<uint16_t> input(size_t(input_rows) * k), weight(size_t(n) * k);
  std::vector<float> bias(n);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = f32_to_bf16(float(int(i % 31) - 15) / 32.0f);
  for (size_t i = 0; i < weight.size(); ++i)
    weight[i] = f32_to_bf16(float(int(i % 23) - 11) / 16.0f);
  for (uint32_t i = 0; i < n; ++i) bias[i] = float(int(i) - 7) / 64.0f;
  // Long-K cancellation, signed zero, and infinity rows stay inside the
  // documented non-NaN arithmetic domain.
  for (uint32_t inner = 0; inner < k; ++inner) {
    input[size_t(input_offset + 0) * k + inner] =
        f32_to_bf16((inner & 1) ? -1.0f : 1.0f);
    input[size_t(input_offset + 1) * k + inner] = 0x8000u;
    input[size_t(input_offset + 2) * k + inner] = 0;
  }
  input[size_t(input_offset + 2) * k] = 0x7f80u;
  for (uint32_t column = 0; column < n; ++column)
    weight[size_t(column) * k] = f32_to_bf16(1.0f);

  cuda::DeviceBuffer<uint16_t> ci(input.size()), cw(weight.size()),
      co(size_t(output_rows) * n);
  cuda::DeviceBuffer<float> cb(bias.size());
  ci.copy_from_host(input.data(), input.size()); cw.copy_from_host(weight.data(), weight.size());
  cb.copy_from_host(bias.data(), bias.size());
  cuda::launch_deterministic_bf16_gemm_nt(
      reinterpret_cast<const __nv_bfloat16*>(ci.get()) +
          size_t(input_offset) * k,
      reinterpret_cast<const __nv_bfloat16*>(cw.get()), cb.get(),
      reinterpret_cast<__nv_bfloat16*>(co.get()), m, n, k,
      DenseGemmBias::kFloat32, 0, output_offset);
  std::vector<uint16_t> cuda_output(size_t(output_rows) * n);
  co.copy_to_host(cuda_output.data(), cuda_output.size());

  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  CHECK(!physical.empty());
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_cooperative_matrix = true;
  options.enable_storage_buffer_16bit = true;
  Device device = physical.front().create_device(options);
  TensorContext context(device);
  const uint64_t is[] = {input_rows, k}, ws[] = {n, k};
  const uint64_t os[] = {output_rows, n}, bs[] = {n};
  DeviceTensor vi = context.allocate(TensorLayout::contiguous(is, 2), ScalarType::kBFloat16);
  DeviceTensor vw = context.allocate(TensorLayout::contiguous(ws, 2), ScalarType::kBFloat16);
  DeviceTensor vo = context.allocate(TensorLayout::contiguous(os, 2), ScalarType::kBFloat16);
  DeviceTensor vb = context.allocate(TensorLayout::contiguous(bs, 1), ScalarType::kFloat32);
  context.upload_bytes(vi, input.data(), input.size() * 2);
  context.upload_bytes(vw, weight.data(), weight.size() * 2);
  context.upload(vb, bias.data(), bias.size());
  DenseGemmPlanDesc desc{m, n, k, DenseGemmMode::kBFloat16,
                         DenseGemmBias::kFloat32};
  DenseGemmPlan plan = DenseGemmPlan::create(context, desc);
  TensorBatch batch = context.begin_batch();
  plan.record(batch, vi, vw, vo, m, input_offset, output_offset, &vb);
  batch.submit().wait();
  std::vector<uint16_t> vulkan_output(cuda_output.size());
  context.download_bytes(vo, vulkan_output.data(), vulkan_output.size() * 2);
  size_t differences = 0;
  for (uint32_t row = 0; row < m; ++row) {
    for (uint32_t column = 0; column < n; ++column) {
      const size_t i = size_t(output_offset + row) * n + column;
      differences += cuda_output[i] != vulkan_output[i];
    }
  }
  CHECK_MSG(differences == 0, "cooperative BF16 GEMM differs in %zu/%zu values",
            differences, cuda_output.size());
}

VIDFAB_TEST(cuda_vulkan_cooperative_f16_gemm_exact) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  constexpr uint32_t m = 64, n = 64, k = 64;
  constexpr uint32_t input_rows = 128, output_rows = 128;
  constexpr uint32_t input_offset = 64, output_offset = 17;
  std::vector<float> input_f32(size_t(input_rows) * k);
  std::vector<uint16_t> input_f16(input_f32.size()), weight(size_t(n) * k);
  for (size_t i = 0; i < input_f32.size(); ++i) {
    input_f32[i] = float(int(i % 37) - 18) / 29.0f;
    input_f16[i] = f32_to_f16(input_f32[i]);
  }
  for (size_t i = 0; i < weight.size(); ++i)
    weight[i] = f32_to_f16(float(int(i % 29) - 14) / 23.0f);

  cuda::DeviceBuffer<uint16_t> ci(input_f16.size()), cw(weight.size());
  cuda::DeviceBuffer<float> co(size_t(output_rows) * n);
  ci.copy_from_host(input_f16.data(), input_f16.size());
  cw.copy_from_host(weight.data(), weight.size());
  // Point the CUDA reference at the same prepared source-row range.
  cuda::launch_deterministic_f16_gemm_nt(
      reinterpret_cast<const __half*>(ci.get()) + size_t(input_offset) * k,
      reinterpret_cast<const __half*>(cw.get()), co.get(), m, n, k,
      output_offset);
  std::vector<float> cuda_output(size_t(output_rows) * n);
  co.copy_to_host(cuda_output.data(), cuda_output.size());

  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  CHECK(!physical.empty());
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_cooperative_matrix = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_shader_float16 = true;
  Device device = physical.front().create_device(options);
  TensorContext context(device);
  const uint64_t is[] = {input_rows, k}, ws[] = {n, k};
  const uint64_t os[] = {output_rows, n};
  DeviceTensor vi = context.allocate(TensorLayout::contiguous(is, 2),
                                     ScalarType::kFloat32);
  DeviceTensor vw = context.allocate(TensorLayout::contiguous(ws, 2),
                                     ScalarType::kFloat16);
  DeviceTensor vo = context.allocate(TensorLayout::contiguous(os, 2),
                                     ScalarType::kFloat32);
  context.upload(vi, input_f32.data(), input_f32.size());
  context.upload_bytes(vw, weight.data(), weight.size() * 2);
  PreparedF16Activation slot = PreparedF16Activation::create(context, m, k);
  DenseGemmPlanDesc desc{m, n, k, DenseGemmMode::kFloat16Vae,
                         DenseGemmBias::kNone};
  DenseGemmPlan plan = DenseGemmPlan::create(context, desc);
  TensorBatch batch = context.begin_batch();
  PreparedF16ActivationView prepared = slot.prepare(
      batch, vi, m, input_offset);
  plan.record(batch, prepared, vw, vo, output_offset);
  // A second projection consumes the same batch-scoped conversion.
  plan.record(batch, prepared, vw, vo, output_offset);
  batch.submit().wait();
  std::vector<float> vulkan_output(cuda_output.size());
  context.download(vo, vulkan_output.data(), vulkan_output.size());
  size_t differences = 0;
  for (uint32_t row = 0; row < m; ++row) {
    for (uint32_t column = 0; column < n; ++column) {
      const size_t i = size_t(output_offset + row) * n + column;
      differences += std::memcmp(&cuda_output[i], &vulkan_output[i],
                                 sizeof(float)) != 0;
    }
  }
  CHECK_MSG(differences == 0,
            "cooperative F16 GEMM differs in %zu/%zu values", differences,
            cuda_output.size());
}

VIDFAB_TEST(cuda_vulkan_scalar_gemm_modes_exact) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  constexpr uint32_t m = 3, n = 11, k = 19;
  constexpr uint32_t input_rows = 5, output_rows = 6;
  constexpr uint32_t input_offset = 1, output_offset = 2;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  CHECK(!physical.empty());
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_cooperative_matrix = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_shader_float16 = true;
  Device device = physical.front().create_device(options);
  TensorContext context(device);
  const uint64_t as[] = {input_rows, k}, ws[] = {n, k};
  const uint64_t os[] = {output_rows, n}, bs[] = {n};

  auto run = [&](DenseGemmMode mode, DenseGemmBias bias_type) {
    std::vector<float> af(size_t(input_rows) * k), wf(size_t(n) * k);
    std::vector<float> biasf(n);
    for (size_t i = 0; i < af.size(); ++i)
      af[i] = float(int(i % 31) - 15) / 23.0f;
    for (size_t i = 0; i < wf.size(); ++i)
      wf[i] = float(int(i % 27) - 13) / 19.0f;
    for (uint32_t i = 0; i < n; ++i)
      biasf[i] = float(int(i) - 5) / 37.0f;

    const ScalarType at = mode == DenseGemmMode::kBFloat16
        ? ScalarType::kBFloat16 : ScalarType::kFloat32;
    const ScalarType wt = mode == DenseGemmMode::kBFloat16
        ? ScalarType::kBFloat16 : mode == DenseGemmMode::kFloat16Vae
            ? ScalarType::kFloat16 : ScalarType::kFloat32;
    const ScalarType ot = mode == DenseGemmMode::kBFloat16
        ? ScalarType::kBFloat16 : ScalarType::kFloat32;
    DeviceTensor va = context.allocate(TensorLayout::contiguous(as, 2), at);
    DeviceTensor vw = context.allocate(TensorLayout::contiguous(ws, 2), wt);
    DeviceTensor vo = context.allocate(TensorLayout::contiguous(os, 2), ot);
    DeviceTensor vb;
    if (bias_type != DenseGemmBias::kNone) {
      vb = context.allocate(TensorLayout::contiguous(bs, 1),
          bias_type == DenseGemmBias::kBFloat16 ? ScalarType::kBFloat16
                                                : ScalarType::kFloat32);
    }

    DenseGemmPlan plan = DenseGemmPlan::create(
        context, {m, n, k, mode, bias_type});
    if (mode == DenseGemmMode::kBFloat16) {
      std::vector<uint16_t> ah(af.size()), wh(wf.size()),
          biash(n), sentinel(size_t(output_rows) * n, 0x3f00u);
      for (size_t i = 0; i < ah.size(); ++i) ah[i] = f32_to_bf16(af[i]);
      for (size_t i = 0; i < wh.size(); ++i) wh[i] = f32_to_bf16(wf[i]);
      for (uint32_t i = 0; i < n; ++i) biash[i] = f32_to_bf16(biasf[i]);
      cuda::DeviceBuffer<uint16_t> ca(ah.size()), cw(wh.size()),
          cbh(biash.size()), co(sentinel.size());
      cuda::DeviceBuffer<float> cbf(biasf.size());
      ca.copy_from_host(ah.data(), ah.size());
      cw.copy_from_host(wh.data(), wh.size());
      co.copy_from_host(sentinel.data(), sentinel.size());
      const void* cb = nullptr;
      if (bias_type == DenseGemmBias::kFloat32) {
        cbf.copy_from_host(biasf.data(), biasf.size()); cb = cbf.get();
      } else if (bias_type == DenseGemmBias::kBFloat16) {
        cbh.copy_from_host(biash.data(), biash.size()); cb = cbh.get();
      }
      cuda::launch_deterministic_scalar_gemm_nt(
          ca.get(), cw.get(), cb, co.get(), m, n, k, mode, bias_type,
          input_offset, output_offset);
      std::vector<uint16_t> cuda_out(sentinel.size()), vk_out(sentinel.size());
      co.copy_to_host(cuda_out.data(), cuda_out.size());
      context.upload_bytes(va, ah.data(), ah.size() * 2);
      context.upload_bytes(vw, wh.data(), wh.size() * 2);
      context.upload_bytes(vo, sentinel.data(), sentinel.size() * 2);
      if (bias_type == DenseGemmBias::kFloat32)
        context.upload(vb, biasf.data(), biasf.size());
      else if (bias_type == DenseGemmBias::kBFloat16)
        context.upload_bytes(vb, biash.data(), biash.size() * 2);
      TensorBatch batch = context.begin_batch();
      plan.record(batch, va, vw, vo, m, input_offset, output_offset,
                  bias_type == DenseGemmBias::kNone ? nullptr : &vb);
      batch.submit().wait();
      context.download_bytes(vo, vk_out.data(), vk_out.size() * 2);
      CHECK(std::memcmp(cuda_out.data(), vk_out.data(), vk_out.size() * 2) == 0);
    } else {
      std::vector<float> sentinel(size_t(output_rows) * n, 0.375f);
      std::vector<uint16_t> ah16, wh16;
      cuda::DeviceBuffer<float> caf(af.size()), cwf(wf.size()),
          cbf(biasf.size()), co(sentinel.size());
      cuda::DeviceBuffer<uint16_t> cah, cwh;
      const void* ca = nullptr;
      const void* cw = nullptr;
      if (mode == DenseGemmMode::kFloat16Vae) {
        ah16.resize(af.size()); wh16.resize(wf.size());
        for (size_t i = 0; i < af.size(); ++i) ah16[i] = f32_to_f16(af[i]);
        for (size_t i = 0; i < wf.size(); ++i) wh16[i] = f32_to_f16(wf[i]);
        cah.allocate(ah16.size()); cwh.allocate(wh16.size());
        cah.copy_from_host(ah16.data(), ah16.size());
        cwh.copy_from_host(wh16.data(), wh16.size());
        ca = cah.get(); cw = cwh.get();
        context.upload(va, af.data(), af.size());
        context.upload_bytes(vw, wh16.data(), wh16.size() * 2);
      } else {
        caf.copy_from_host(af.data(), af.size());
        cwf.copy_from_host(wf.data(), wf.size());
        ca = caf.get(); cw = cwf.get();
        context.upload(va, af.data(), af.size());
        context.upload(vw, wf.data(), wf.size());
      }
      if (bias_type == DenseGemmBias::kFloat32) {
        cbf.copy_from_host(biasf.data(), biasf.size());
        context.upload(vb, biasf.data(), biasf.size());
      }
      co.copy_from_host(sentinel.data(), sentinel.size());
      cuda::launch_deterministic_scalar_gemm_nt(
          ca, cw, bias_type == DenseGemmBias::kNone ? nullptr : cbf.get(),
          co.get(), m, n, k, mode, bias_type, input_offset, output_offset);
      context.upload(vo, sentinel.data(), sentinel.size());
      PreparedF16Activation slot;
      if (mode == DenseGemmMode::kFloat16Vae)
        slot = PreparedF16Activation::create(context, m, k);
      TensorBatch batch = context.begin_batch();
      if (mode == DenseGemmMode::kFloat16Vae) {
        PreparedF16ActivationView prepared = slot.prepare(
            batch, va, m, input_offset);
        plan.record(batch, prepared, vw, vo, output_offset);
      } else {
        plan.record(batch, va, vw, vo, m, input_offset, output_offset,
                    bias_type == DenseGemmBias::kNone ? nullptr : &vb);
      }
      batch.submit().wait();
      std::vector<float> cuda_out(sentinel.size()), vk_out(sentinel.size());
      co.copy_to_host(cuda_out.data(), cuda_out.size());
      context.download(vo, vk_out.data(), vk_out.size());
      CHECK(std::memcmp(cuda_out.data(), vk_out.data(), vk_out.size() * 4) == 0);
    }
  };
  run(DenseGemmMode::kBFloat16, DenseGemmBias::kNone);
  run(DenseGemmMode::kBFloat16, DenseGemmBias::kFloat32);
  run(DenseGemmMode::kBFloat16, DenseGemmBias::kBFloat16);
  run(DenseGemmMode::kFloat16Vae, DenseGemmBias::kNone);
  run(DenseGemmMode::kFloat32, DenseGemmBias::kNone);
  run(DenseGemmMode::kFloat32, DenseGemmBias::kFloat32);
}

VIDFAB_TEST(cuda_vulkan_dense_gemm_production_timing) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  CHECK(!physical.empty());
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  options.enable_cooperative_matrix = true;
  options.enable_storage_buffer_16bit = true;
  options.enable_shader_float16 = true;
  Device device = physical.front().create_device(options);
  TensorContext context(device);

  cublasHandle_t handle = nullptr;
  VIDFAB_CUBLAS_CHECK(cublasCreate(&handle));
  cudaEvent_t begin = nullptr, end = nullptr;
  VIDFAB_CUDA_CHECK(cudaEventCreate(&begin));
  VIDFAB_CUDA_CHECK(cudaEventCreate(&end));
  const float alpha = 1.0f, beta = 0.0f;

  {
    constexpr uint32_t m = 64, n = 6144, k = 2048;
    cuda::DeviceBuffer<float> caf(size_t(m) * k);
    cuda::DeviceBuffer<__half> ca(size_t(m) * k), cw(size_t(n) * k);
    cuda::DeviceBuffer<float> co(size_t(m) * n);
    VIDFAB_CUDA_CHECK(cudaMemset(caf.get(), 0, size_t(m) * k * 4));
    VIDFAB_CUDA_CHECK(cudaMemset(ca.get(), 0, size_t(m) * k * 2));
    VIDFAB_CUDA_CHECK(cudaMemset(cw.get(), 0, size_t(n) * k * 2));
    auto cuda_launch = [&] {
      VIDFAB_CUBLAS_CHECK(cublasGemmEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, cw.get(),
          CUDA_R_16F, k, ca.get(), CUDA_R_16F, k, &beta, co.get(),
          CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    };
    cuda_launch();
    VIDFAB_CUDA_CHECK(cudaEventRecord(begin));
    for (int i = 0; i < 20; ++i) cuda_launch();
    VIDFAB_CUDA_CHECK(cudaEventRecord(end));
    VIDFAB_CUDA_CHECK(cudaEventSynchronize(end));
    float cuda_ms = 0;
    VIDFAB_CUDA_CHECK(cudaEventElapsedTime(&cuda_ms, begin, end));
    cuda_ms /= 20.0f;
    auto cuda_total_launch = [&] {
      cuda::launch_narrow_f16(caf.get(), ca.get(), size_t(m) * k, nullptr);
      cuda_launch();
    };
    cuda_total_launch();
    VIDFAB_CUDA_CHECK(cudaEventRecord(begin));
    for (int i = 0; i < 20; ++i) cuda_total_launch();
    VIDFAB_CUDA_CHECK(cudaEventRecord(end));
    VIDFAB_CUDA_CHECK(cudaEventSynchronize(end));
    float cuda_total_ms = 0;
    VIDFAB_CUDA_CHECK(cudaEventElapsedTime(&cuda_total_ms, begin, end));
    cuda_total_ms /= 20.0f;

    const uint64_t as[] = {m, k}, ws[] = {n, k}, os[] = {m, n};
    DeviceTensor a = context.allocate(TensorLayout::contiguous(as, 2),
                                      ScalarType::kFloat32);
    DeviceTensor w = context.allocate(TensorLayout::contiguous(ws, 2),
                                      ScalarType::kFloat16);
    DeviceTensor o = context.allocate(TensorLayout::contiguous(os, 2),
                                      ScalarType::kFloat32);
    std::vector<float> ah(size_t(m) * k, 0.0f);
    std::vector<uint16_t> wh(size_t(n) * k, 0);
    context.upload(a, ah.data(), ah.size());
    context.upload_bytes(w, wh.data(), wh.size() * 2);
    PreparedF16Activation slot = PreparedF16Activation::create(context, m, k);
    DenseGemmPlan plan = DenseGemmPlan::create(
        context, {m, n, k, DenseGemmMode::kFloat16Vae,
                  DenseGemmBias::kNone});
    double prepare_ms = 0.0;
    for (int i = -2; i < 10; ++i) {
      const auto start = std::chrono::steady_clock::now();
      TensorBatch batch = context.begin_batch();
      (void)slot.prepare(batch, a, m);
      batch.submit().wait();
      if (i >= 0)
        prepare_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    }
    prepare_ms /= 10.0;
    double vulkan_total_ms = 0.0;
    for (int i = -2; i < 10; ++i) {
      const auto total_start = std::chrono::steady_clock::now();
      TensorBatch total_batch = context.begin_batch();
      PreparedF16ActivationView total_prepared =
          slot.prepare(total_batch, a, m);
      plan.record(total_batch, total_prepared, w, o);
      total_batch.submit().wait();
      if (i >= 0)
        vulkan_total_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - total_start).count();
    }
    vulkan_total_ms /= 10.0;
    const auto start = std::chrono::steady_clock::now();
    TensorBatch batch = context.begin_batch();
    PreparedF16ActivationView prepared = slot.prepare(batch, a, m);
    for (int i = 0; i < 16; ++i) plan.record(batch, prepared, w, o);
    batch.submit().wait();
    const double vulkan_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count() / 16.0;
    std::printf("  F16 VAE M64 N6144 K2048: GEMM CUDA %.3f ms/Vulkan %.3f ms, "
                "narrow+GEMM CUDA %.3f ms/Vulkan %.3f ms, prepare %.3f ms\n",
                cuda_ms, vulkan_ms, cuda_total_ms, vulkan_total_ms, prepare_ms);
  }

  {
    constexpr uint32_t m = 64, n = 2048, k = 2048;
    cuda::DeviceBuffer<float> ca(size_t(m) * k), cw(size_t(n) * k),
        co(size_t(m) * n);
    VIDFAB_CUDA_CHECK(cudaMemset(ca.get(), 0, size_t(m) * k * 4));
    VIDFAB_CUDA_CHECK(cudaMemset(cw.get(), 0, size_t(n) * k * 4));
    auto cuda_launch = [&] {
      VIDFAB_CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                      n, m, k, &alpha, cw.get(), k,
                                      ca.get(), k, &beta, co.get(), n));
    };
    cuda_launch();
    VIDFAB_CUDA_CHECK(cudaEventRecord(begin));
    for (int i = 0; i < 20; ++i) cuda_launch();
    VIDFAB_CUDA_CHECK(cudaEventRecord(end));
    VIDFAB_CUDA_CHECK(cudaEventSynchronize(end));
    float cuda_ms = 0;
    VIDFAB_CUDA_CHECK(cudaEventElapsedTime(&cuda_ms, begin, end));
    cuda_ms /= 20.0f;

    const uint64_t as[] = {m, k}, ws[] = {n, k}, os[] = {m, n};
    DeviceTensor a = context.allocate(TensorLayout::contiguous(as, 2));
    DeviceTensor w = context.allocate(TensorLayout::contiguous(ws, 2));
    DeviceTensor o = context.allocate(TensorLayout::contiguous(os, 2));
    std::vector<float> ah(size_t(m) * k, 0.0f), wh(size_t(n) * k, 0.0f);
    context.upload(a, ah.data(), ah.size());
    context.upload(w, wh.data(), wh.size());
    DenseGemmPlan plan = DenseGemmPlan::create(
        context, {m, n, k, DenseGemmMode::kFloat32, DenseGemmBias::kNone});
    const auto start = std::chrono::steady_clock::now();
    TensorBatch batch = context.begin_batch();
    for (int i = 0; i < 8; ++i) plan.record(batch, a, w, o, m);
    batch.submit().wait();
    const double vulkan_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count() / 8.0;
    std::printf("  F32 SGEMM M64 N2048 K2048: CUDA %.3f ms, Vulkan %.3f ms\n",
                cuda_ms, vulkan_ms);
  }
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  cublasDestroy(handle);
}

VIDFAB_TEST(cuda_vulkan_exact_vae_vit_block_stage) {
  using namespace vidfab;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0 ||
      !vulkan::Instance::available()) return;
  vulkan::Instance instance = vulkan::Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;
  vulkan::DeviceOptions device_options;
  device_options.enable_timeline_semaphore = true;
  device_options.enable_shader_int64 = physical.front().info().shader_int64;
  vulkan::Device device = physical.front().create_device(device_options);
  vulkan::TensorContextOptions context_options;
  context_options.max_batch_operators = 64;
  vulkan::TensorContext context(device, context_options);
  if (!context.exact_fp32_vae_normalization() || !context.exact_vae_pointwise() ||
      !context.exact_blocked_attention()) return;

  vae::ViTBlockConfig config;
  config.sequence = 69;
  config.num_patches = 64;
  config.dim = 64;
  config.heads = 1;
  config.head_dim = 64;
  config.ffn_inner = 128;
  config.rope_dim = 48;
  vae::ViTBlockWeights weights;
  weights.norm1.resize(config.dim); weights.norm2.resize(config.dim);
  weights.scale1.resize(config.dim); weights.scale2.resize(config.dim);
  weights.qkv_weight.resize(size_t(3) * config.dim * config.dim);
  weights.qkv_bias.resize(3 * config.dim);
  weights.out_weight.resize(size_t(config.dim) * config.dim);
  weights.out_bias.resize(config.dim);
  weights.w1_weight.resize(size_t(2) * config.ffn_inner * config.dim);
  weights.w1_bias.resize(2 * config.ffn_inner);
  weights.w2_weight.resize(size_t(config.dim) * config.ffn_inner);
  weights.w2_bias.resize(config.dim);
  for (uint32_t i = 0; i < config.dim; ++i) {
    weights.norm1[i] = 0.75f + float(i % 11) / 32.0f;
    weights.norm2[i] = 0.875f + float(i % 7) / 32.0f;
    weights.scale1[i] = 0.01f + float(i % 5) / 1024.0f;
    weights.scale2[i] = 0.0125f + float(i % 3) / 1024.0f;
    weights.out_bias[i] = float(int(i % 13) - 6) / 512.0f;
    weights.w2_bias[i] = float(int(i % 17) - 8) / 512.0f;
  }
  auto fill_half = [](std::vector<uint16_t>& values, uint32_t multiplier) {
    for (size_t i = 0; i < values.size(); ++i) {
      const float value = float(int((i * multiplier) % 31) - 15) / 512.0f;
      values[i] = f32_to_f16(value);
    }
  };
  fill_half(weights.qkv_weight, 7); fill_half(weights.out_weight, 11);
  fill_half(weights.w1_weight, 13); fill_half(weights.w2_weight, 17);
  for (size_t i = 0; i < weights.qkv_bias.size(); ++i)
    weights.qkv_bias[i] = float(int(i % 19) - 9) / 512.0f;
  for (size_t i = 0; i < weights.w1_bias.size(); ++i)
    weights.w1_bias[i] = float(int(i % 23) - 11) / 512.0f;

  const size_t token_count = size_t(config.sequence) * config.dim;
  const size_t rope_count = size_t(config.sequence) * config.rope_dim;
  std::vector<float> input(token_count), cosine(rope_count, 1.0f),
      sine(rope_count, 0.0f), cuda_once(token_count), vulkan_once(token_count),
      cuda_twice(token_count), vulkan_twice(token_count);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = float(int((i * 29) % 251) - 125) / 128.0f;

  auto cuda_stage = cuda::create_exact_vae_vit_block_stage(config);
  cuda_stage->load(weights.view());
  vulkan::ExactViTBlockStage vk_stage =
      vulkan::ExactViTBlockStage::create(context, config);
  vk_stage.load(weights.view());
  CHECK(cuda_stage->backend() == DeviceBackend::kCuda);
  CHECK(vk_stage.backend() == DeviceBackend::kVulkan);
  CHECK(cuda_stage->persistent_bytes() == weights.bytes());
  CHECK(vk_stage.persistent_bytes() == weights.bytes());
  {
    vae::ViTBlockWeights noncanonical = weights;
    noncanonical.qkv_weight[0] = 0x0001u;
    bool cuda_rejected = false, vulkan_rejected = false;
    try {
      auto bad = cuda::create_exact_vae_vit_block_stage(config);
      bad->load(noncanonical.view());
    } catch (const std::invalid_argument&) { cuda_rejected = true; }
    try {
      vulkan::ExactViTBlockStage bad =
          vulkan::ExactViTBlockStage::create(context, config);
      bad.load(noncanonical.view());
    } catch (const std::invalid_argument&) { vulkan_rejected = true; }
    CHECK(cuda_rejected && vulkan_rejected);
  }
  cuda_stage->forward(input.data(), cosine.data(), sine.data(), cuda_once.data());
  vk_stage.forward(input.data(), cosine.data(), sine.data(), vulkan_once.data());
  CHECK(std::memcmp(cuda_once.data(), vulkan_once.data(), token_count * 4) == 0);

  // The production seam chains two block records through one device tensor,
  // one command buffer and one shared scratch arena. No host boundary occurs.
  const uint64_t token_shape[] = {config.sequence, config.dim};
  const uint64_t rope_shape[] = {config.sequence, config.rope_dim};
  vulkan::DeviceTensor tokens = context.allocate(
      TensorLayout::contiguous(token_shape, 2));
  vulkan::DeviceTensor vk_cosine = context.allocate(
      TensorLayout::contiguous(rope_shape, 2));
  vulkan::DeviceTensor vk_sine = context.allocate(
      TensorLayout::contiguous(rope_shape, 2));
  context.upload(tokens, input.data(), input.size());
  context.upload(vk_cosine, cosine.data(), cosine.size());
  context.upload(vk_sine, sine.data(), sine.size());
  vulkan::ExactViTBlockScratch scratch =
      vulkan::ExactViTBlockScratch::create(context, config);
  const uint64_t wrong_rope_shape[] = {config.sequence, config.rope_dim - 1};
  vulkan::DeviceTensor wrong_cosine = context.allocate(
      TensorLayout::contiguous(wrong_rope_shape, 2));
  vulkan::TensorBatch chained = context.begin_batch();
  const uint32_t capacity_before_rejection =
      chained.remaining_operator_capacity();
  bool shape_rejected = false;
  try { vk_stage.record(chained, tokens, wrong_cosine, vk_sine, scratch); }
  catch (const std::invalid_argument&) { shape_rejected = true; }
  CHECK(shape_rejected);
  CHECK(chained.remaining_operator_capacity() == capacity_before_rejection);
  vk_stage.record(chained, tokens, vk_cosine, vk_sine, scratch);
  vk_stage.record(chained, tokens, vk_cosine, vk_sine, scratch);
  chained.submit().wait();
  context.download(tokens, vulkan_twice.data(), vulkan_twice.size());
  cuda_stage->forward(cuda_once.data(), cosine.data(), sine.data(), cuda_twice.data());
  CHECK(std::memcmp(cuda_twice.data(), vulkan_twice.data(), token_count * 4) == 0);

  const uint64_t stable_reserved = context.reserved_bytes();
  const uint64_t stable_descriptors = context.descriptor_set_allocations();
  vk_stage.forward(input.data(), cosine.data(), sine.data(), vulkan_twice.data());
  CHECK(std::memcmp(cuda_once.data(), vulkan_twice.data(), token_count * 4) == 0);
  CHECK(context.reserved_bytes() == stable_reserved);
  CHECK(context.descriptor_set_allocations() == stable_descriptors);

  // Three real graph nodes share one scratch arena. Vulkan records all sixty
  // operators into a single submission and exactly matches the CUDA graph.
  cuda::ExactViTBlockGraph cuda_graph =
      cuda::ExactViTBlockGraph::create(config, 3);
  vulkan::ExactViTBlockGraph vk_graph =
      vulkan::ExactViTBlockGraph::create(context, config, 3);
  for (uint32_t layer = 0; layer < 3; ++layer) {
    cuda_graph.load_layer(layer, weights.view());
    vk_graph.load_layer(layer, weights.view());
  }
  std::vector<float> cuda_graph_output(token_count),
      vk_graph_output(token_count);
  cuda_graph.forward(input.data(), cosine.data(), sine.data(),
                     cuda_graph_output.data());
  vk_graph.forward(input.data(), cosine.data(), sine.data(),
                   vk_graph_output.data());
  CHECK(std::memcmp(cuda_graph_output.data(), vk_graph_output.data(),
                    token_count * sizeof(float)) == 0);
  CHECK(cuda_graph.layers() == 3 && vk_graph.layers() == 3);
  CHECK(cuda_graph.persistent_bytes() == 3 * weights.bytes());
  CHECK(vk_graph.persistent_bytes() == 3 * weights.bytes());
  CHECK(cuda_graph.peak_device_bytes() < 3 * cuda_stage->peak_device_bytes());
  CHECK(vk_graph.peak_device_bytes() < 3 * vk_stage.peak_device_bytes());
  const uint64_t graph_reserved = context.reserved_bytes();
  const uint64_t graph_descriptors = context.descriptor_set_allocations();
  vk_graph.forward(input.data(), cosine.data(), sine.data(),
                   vk_graph_output.data());
  CHECK(context.reserved_bytes() == graph_reserved);
  CHECK(context.descriptor_set_allocations() == graph_descriptors);

  // A full graph is transactional with respect to load state: neither backend
  // may execute its loaded prefix when a later layer is missing.
  cuda::ExactViTBlockGraph partial_cuda =
      cuda::ExactViTBlockGraph::create(config, 3);
  vulkan::ExactViTBlockGraph partial_vk =
      vulkan::ExactViTBlockGraph::create(context, config, 3);
  partial_cuda.load_layer(0, weights.view());
  partial_vk.load_layer(0, weights.view());
  cuda::DeviceBuffer<float> partial_tokens(token_count),
      partial_cosine(rope_count), partial_sine(rope_count);
  cuda::Stream partial_stream;
  partial_tokens.copy_from_host(input.data(), input.size(), partial_stream.get());
  partial_cosine.copy_from_host(cosine.data(), cosine.size(), partial_stream.get());
  partial_sine.copy_from_host(sine.data(), sine.size(), partial_stream.get());
  partial_stream.synchronize();
  bool partial_cuda_rejected = false;
  try {
    partial_cuda.forward_device(partial_tokens.get(), partial_cosine.get(),
                                partial_sine.get(), partial_stream.get());
  } catch (const std::logic_error&) {
    partial_cuda_rejected = true;
  }
  CHECK(partial_cuda_rejected);
  CHECK(cudaStreamQuery(partial_stream.get()) == cudaSuccess);
  std::vector<float> partial_after(token_count);
  partial_tokens.copy_to_host(partial_after.data(), partial_after.size(),
                              partial_stream.get());
  partial_stream.synchronize();
  CHECK(std::memcmp(input.data(), partial_after.data(), token_count * 4) == 0);
  {
    vulkan::TensorBatch partial_batch = context.begin_batch();
    const uint32_t partial_capacity =
        partial_batch.remaining_operator_capacity();
    bool partial_vk_rejected = false;
    try {
      partial_vk.record(partial_batch, tokens, vk_cosine, vk_sine);
    } catch (const std::logic_error&) {
      partial_vk_rejected = true;
    }
    CHECK(partial_vk_rejected);
    CHECK(partial_batch.remaining_operator_capacity() == partial_capacity);
  }

  // The CUDA ragged-shape cache is hard-bounded to two arenas. The largest
  // shape below supplies a monotonic upper bound for every transition.
  cuda::ExactViTBlockGraph shape_graph =
      cuda::ExactViTBlockGraph::create(config, 3);
  for (uint32_t layer = 0; layer < 3; ++layer)
    shape_graph.load_layer(layer, weights.view());
  const uint64_t shape_persistent = shape_graph.persistent_bytes();
  const uint64_t base_peak = shape_graph.peak_device_bytes();
  const uint64_t base_scratch = base_peak - shape_persistent;
  CHECK(shape_graph.cached_scratch_shapes() == 1);
  shape_graph.prepare_shape(101, 96);
  const uint64_t full_peak = shape_graph.peak_device_bytes();
  const uint64_t full_scratch = full_peak - shape_persistent - base_scratch;
  CHECK(shape_graph.cached_scratch_shapes() == 2);
  const uint64_t two_full_bound = shape_persistent + 2 * full_scratch;
  CHECK(full_peak <= two_full_bound);
  shape_graph.prepare_shape(85, 80);
  CHECK(shape_graph.cached_scratch_shapes() == 2);
  CHECK(shape_graph.peak_device_bytes() <= two_full_bound);
  shape_graph.prepare_shape(93, 88);
  CHECK(shape_graph.cached_scratch_shapes() == 2);
  CHECK(shape_graph.peak_device_bytes() <= two_full_bound);
  vulkan::ExactViTBlockGraph vk_shape_graph =
      vulkan::ExactViTBlockGraph::create(context, config, 3);
  for (uint32_t layer = 0; layer < 3; ++layer)
    vk_shape_graph.load_layer(layer, weights.view());
  const uint64_t vk_shape_persistent = vk_shape_graph.persistent_bytes();
  const uint64_t vk_base_scratch =
      vk_shape_graph.peak_device_bytes() - vk_shape_persistent;
  CHECK(vk_shape_graph.cached_scratch_shapes() == 1);
  vk_shape_graph.prepare_shape(101, 96);
  const uint64_t vk_full_scratch = vk_shape_graph.peak_device_bytes() -
      vk_shape_persistent - vk_base_scratch;
  const uint64_t vk_two_full_bound =
      vk_shape_persistent + 2 * vk_full_scratch;
  CHECK(vk_shape_graph.cached_scratch_shapes() == 2);
  CHECK(vk_shape_graph.peak_device_bytes() <= vk_two_full_bound);
  vk_shape_graph.prepare_shape(85, 80);
  CHECK(vk_shape_graph.cached_scratch_shapes() == 2);
  CHECK(vk_shape_graph.peak_device_bytes() <= vk_two_full_bound);
  vk_shape_graph.prepare_shape(93, 88);
  CHECK(vk_shape_graph.cached_scratch_shapes() == 2);
  CHECK(vk_shape_graph.peak_device_bytes() <= vk_two_full_bound);

  // Queue work against the current arena, switch twice so that arena is the
  // eviction victim, and then queue another shape on a second stream. Slot
  // completion events make both reuse and eviction safe without a device-wide
  // synchronization in the execution path.
  auto make_cuda_activation = [&](uint32_t sequence) {
    return cuda::DeviceBuffer<float>(size_t(sequence) * config.dim);
  };
  auto make_cuda_rope = [&](uint32_t sequence) {
    return cuda::DeviceBuffer<float>(size_t(sequence) * config.rope_dim);
  };
  cuda::DeviceBuffer<float> queued_tokens = make_cuda_activation(93);
  cuda::DeviceBuffer<float> queued_cosine = make_cuda_rope(93);
  cuda::DeviceBuffer<float> queued_sine = make_cuda_rope(93);
  cuda::DeviceBuffer<float> base_tokens = make_cuda_activation(69);
  cuda::DeviceBuffer<float> base_cosine = make_cuda_rope(69);
  cuda::DeviceBuffer<float> base_sine = make_cuda_rope(69);
  cuda::Stream queued_stream, switched_stream;
  VIDFAB_CUDA_CHECK(cudaMemsetAsync(queued_tokens.get(), 0,
      queued_tokens.nbytes(), queued_stream.get()));
  VIDFAB_CUDA_CHECK(cudaMemsetAsync(queued_cosine.get(), 0,
      queued_cosine.nbytes(), queued_stream.get()));
  VIDFAB_CUDA_CHECK(cudaMemsetAsync(queued_sine.get(), 0,
      queued_sine.nbytes(), queued_stream.get()));
  shape_graph.forward_device(queued_tokens.get(), queued_cosine.get(),
                             queued_sine.get(), queued_stream.get());
  shape_graph.prepare_shape(69, 64);
  VIDFAB_CUDA_CHECK(cudaMemsetAsync(base_tokens.get(), 0, base_tokens.nbytes(),
                                    switched_stream.get()));
  VIDFAB_CUDA_CHECK(cudaMemsetAsync(base_cosine.get(), 0, base_cosine.nbytes(),
                                    switched_stream.get()));
  VIDFAB_CUDA_CHECK(cudaMemsetAsync(base_sine.get(), 0, base_sine.nbytes(),
                                    switched_stream.get()));
  shape_graph.forward_device(base_tokens.get(), base_cosine.get(),
                             base_sine.get(), switched_stream.get());
  shape_graph.prepare_shape(101, 96);  // evicts and fences queued R93
  queued_stream.synchronize();
  switched_stream.synchronize();
  CHECK(shape_graph.cached_scratch_shapes() == 2);
  CHECK(shape_graph.peak_device_bytes() <= two_full_bound);

  {
    vulkan::ExactViTBlockGraph oversized_graph =
        vulkan::ExactViTBlockGraph::create(context, config, 4);
    vulkan::TensorBatch insufficient = context.begin_batch();
    const uint32_t insufficient_capacity =
        insufficient.remaining_operator_capacity();
    bool capacity_rejected = false;
    try {
      oversized_graph.record(insufficient, tokens, vk_cosine, vk_sine);
    } catch (const std::logic_error&) {
      capacity_rejected = true;
    }
    CHECK(capacity_rejected);
    CHECK(insufficient.remaining_operator_capacity() == insufficient_capacity);
  }

  if (!std::getenv("VIDFAB_VAE_VIT_BLOCK_REAL")) return;
  const std::filesystem::path checkpoint_path =
      "weights/vae/minimax_h3_video_vae_fp16.safetensors";
  if (!std::filesystem::exists(checkpoint_path)) return;
  vae::ViTBlockConfig real_config;
  real_config.sequence = 1797;
  real_config.num_patches = 1792;
  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
#ifdef _WIN32
  const std::array<uint8_t, 32> expected_checkpoint_sha{
      0x7c, 0x1f, 0x13, 0x14, 0x92, 0xe7, 0xed, 0xda,
      0xca, 0xac, 0x90, 0x69, 0xa6, 0x1b, 0x81, 0xbd,
      0xd3, 0x9d, 0xe5, 0xcc, 0x96, 0x56, 0x1e, 0x67,
      0x7c, 0x5e, 0xab, 0x1c, 0xdc, 0xe5, 0xe5, 0x22};
  CHECK(sha256_mapping(checkpoint.mapping_base(), checkpoint.file_size()) ==
        expected_checkpoint_sha);
#endif
  const std::array<const char*, 4> real_matrix_names{
      "decoder.transformer_blocks.0.attn.to_qkv.weight",
      "decoder.transformer_blocks.0.attn.to_out.weight",
      "decoder.transformer_blocks.0.ff.w1.weight",
      "decoder.transformer_blocks.0.ff.w2.weight"};
  size_t raw_fp16_subnormals = 0;
  for (const char* name : real_matrix_names) {
    const TensorView& tensor = checkpoint.at(name);
    CHECK(tensor.dtype == DType::kF16);
    const auto* words = static_cast<const uint16_t*>(tensor.data);
    for (size_t i = 0; i < tensor.nbytes / sizeof(uint16_t); ++i)
      raw_fp16_subnormals += (words[i] & 0x7c00u) == 0 &&
                             (words[i] & 0x03ffu) != 0;
  }
  CHECK(raw_fp16_subnormals == 330659u);
  vae::ViTBlockWeights real_weights =
      vae::load_vit_block_weights(checkpoint, 0, real_config);
  CHECK(real_weights.bytes() == 134356992ull);
  size_t fp32_subnormals = 0, fp16_subnormals = 0;
  auto scan_float = [&](const std::vector<float>& values) {
    for (float value : values) {
      uint32_t bits = 0; std::memcpy(&bits, &value, 4);
      fp32_subnormals += (bits & 0x7f800000u) == 0 &&
                         (bits & 0x007fffffu) != 0;
    }
  };
  auto scan_half = [&](const std::vector<uint16_t>& values) {
    for (uint16_t bits : values)
      fp16_subnormals += (bits & 0x7c00u) == 0 && (bits & 0x03ffu) != 0;
  };
  scan_float(real_weights.norm1); scan_float(real_weights.norm2);
  scan_float(real_weights.scale1); scan_float(real_weights.scale2);
  scan_float(real_weights.qkv_bias); scan_float(real_weights.out_bias);
  scan_float(real_weights.w1_bias); scan_float(real_weights.w2_bias);
  scan_half(real_weights.qkv_weight); scan_half(real_weights.out_weight);
  scan_half(real_weights.w1_weight); scan_half(real_weights.w2_weight);
  CHECK(fp32_subnormals == 0);
  CHECK(fp16_subnormals == 0);
  std::printf("  real block weight subnormals fp32=%zu fp16=%zu\n",
              fp32_subnormals, fp16_subnormals);
  std::vector<float> real_input(size_t(real_config.sequence) * real_config.dim),
      real_cosine(size_t(real_config.sequence) * real_config.rope_dim, 1.0f),
      real_sine(real_cosine.size(), 0.0f), real_cuda(real_input.size()),
      real_vulkan(real_input.size());
  for (size_t i = 0; i < real_input.size(); ++i)
    real_input[i] = float(int((i * 29) % 509) - 254) / 512.0f;
  auto real_cuda_stage = cuda::create_exact_vae_vit_block_stage(real_config);
  real_cuda_stage->load(real_weights.view());
  vulkan::ExactViTBlockStage real_vk_stage =
      vulkan::ExactViTBlockStage::create(context, real_config);
  real_vk_stage.load(real_weights.view());
  const auto cuda_begin = std::chrono::steady_clock::now();
  real_cuda_stage->forward(real_input.data(), real_cosine.data(), real_sine.data(),
                           real_cuda.data());
  const double cuda_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - cuda_begin).count();
  const uint64_t real_token_shape[] = {real_config.sequence, real_config.dim};
  const uint64_t real_rope_shape[] = {real_config.sequence, real_config.rope_dim};
  vulkan::DeviceTensor real_tokens = context.allocate(
      TensorLayout::contiguous(real_token_shape, 2));
  vulkan::DeviceTensor real_vk_cosine = context.allocate(
      TensorLayout::contiguous(real_rope_shape, 2));
  vulkan::DeviceTensor real_vk_sine = context.allocate(
      TensorLayout::contiguous(real_rope_shape, 2));
  vulkan::ExactViTBlockScratch real_scratch =
      vulkan::ExactViTBlockScratch::create(context, real_config);
  context.upload(real_tokens, real_input.data(), real_input.size());
  context.upload(real_vk_cosine, real_cosine.data(), real_cosine.size());
  context.upload(real_vk_sine, real_sine.data(), real_sine.size());
  const auto vk_begin = std::chrono::steady_clock::now();
  vulkan::TensorBatch real_batch = context.begin_batch();
  real_vk_stage.record(real_batch, real_tokens, real_vk_cosine, real_vk_sine,
                       real_scratch);
  real_batch.submit().wait();
  const double vk_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - vk_begin).count();
  context.download(real_tokens, real_vulkan.data(), real_vulkan.size());
  size_t mismatch = real_input.size();
  for (size_t i = 0; i < real_input.size(); ++i) {
    if (std::memcmp(&real_cuda[i], &real_vulkan[i], 4) != 0) {
      mismatch = i;
      break;
    }
  }
  if (mismatch != real_input.size()) {
    uint32_t cb = 0, vb = 0;
    std::memcpy(&cb, &real_cuda[mismatch], 4);
    std::memcpy(&vb, &real_vulkan[mismatch], 4);
    std::printf("  first real block mismatch %zu: CUDA %08x Vulkan %08x\n",
                mismatch, cb, vb);
  }
  CHECK_MSG(mismatch == real_input.size(),
            "real VAE ViT block mismatch at %zu/%zu", mismatch,
            real_input.size());
  uint64_t fnv = 1469598103934665603ull;
  for (float value : real_vulkan) {
    uint32_t bits = 0; std::memcpy(&bits, &value, 4);
    for (int byte = 0; byte < 4; ++byte) {
      fnv ^= (bits >> (byte * 8)) & 0xffu;
      fnv *= 1099511628211ull;
    }
  }
  CHECK(fnv == 0xc8a7ac3241effbb7ull);
  std::printf(
      "  real VAE ViT block0 R1797/D2048/I8192: CUDA %.3f ms, Vulkan %.3f ms, exact %zu words, FNV64 %016llx, weights %.1f MiB, Vulkan peak %.1f MiB\n",
      cuda_ms, vk_ms, real_vulkan.size(),
      static_cast<unsigned long long>(fnv),
      double(real_weights.bytes()) / 1048576.0,
      double(real_vk_stage.persistent_bytes() + real_scratch.reserved_bytes() +
             real_input.size() * sizeof(float) +
             2 * real_cosine.size() * sizeof(float)) / 1048576.0);

  if (!std::getenv("VIDFAB_VAE_VIT_GRAPH_REAL")) return;
  vulkan::TensorContextOptions graph_options;
  graph_options.max_batch_operators = 1024;
  vulkan::TensorContext graph_context(device, graph_options);
  if (!graph_context.exact_fp32_vae_normalization() ||
      !graph_context.exact_vae_pointwise() ||
      !graph_context.exact_blocked_attention()) return;
  constexpr uint32_t kGraphLayers = 36;
  cuda::ExactViTBlockGraph real_cuda_graph =
      cuda::ExactViTBlockGraph::create(real_config, kGraphLayers);
  vulkan::ExactViTBlockGraph real_vk_graph =
      vulkan::ExactViTBlockGraph::create(graph_context, real_config,
                                         kGraphLayers);
  size_t graph_raw_subnormals = 0, graph_loaded_subnormals = 0,
      graph_loaded_fp32_subnormals = 0;
  const auto graph_load_begin = std::chrono::steady_clock::now();
  for (uint32_t layer = 0; layer < kGraphLayers; ++layer) {
    const std::array<std::string, 4> names{
        "decoder.transformer_blocks." + std::to_string(layer) +
            ".attn.to_qkv.weight",
        "decoder.transformer_blocks." + std::to_string(layer) +
            ".attn.to_out.weight",
        "decoder.transformer_blocks." + std::to_string(layer) +
            ".ff.w1.weight",
        "decoder.transformer_blocks." + std::to_string(layer) +
            ".ff.w2.weight"};
    for (const std::string& name : names) {
      const TensorView& tensor = checkpoint.at(name);
      CHECK(tensor.dtype == DType::kF16);
      const auto* words = static_cast<const uint16_t*>(tensor.data);
      for (size_t i = 0; i < tensor.nbytes / sizeof(uint16_t); ++i)
        graph_raw_subnormals += (words[i] & 0x7c00u) == 0 &&
                                (words[i] & 0x03ffu) != 0;
    }
    vae::ViTBlockWeights layer_weights =
        vae::load_vit_block_weights(checkpoint, layer, real_config);
    auto count_loaded = [&](const std::vector<uint16_t>& values) {
      for (uint16_t word : values)
        graph_loaded_subnormals += (word & 0x7c00u) == 0 &&
                                   (word & 0x03ffu) != 0;
    };
    count_loaded(layer_weights.qkv_weight);
    count_loaded(layer_weights.out_weight);
    count_loaded(layer_weights.w1_weight);
    count_loaded(layer_weights.w2_weight);
    auto count_loaded_float = [&](const std::vector<float>& values) {
      for (float value : values) {
        uint32_t bits = 0; std::memcpy(&bits, &value, sizeof(bits));
        graph_loaded_fp32_subnormals += (bits & 0x7f800000u) == 0 &&
                                        (bits & 0x007fffffu) != 0;
      }
    };
    count_loaded_float(layer_weights.norm1);
    count_loaded_float(layer_weights.norm2);
    count_loaded_float(layer_weights.scale1);
    count_loaded_float(layer_weights.scale2);
    count_loaded_float(layer_weights.qkv_bias);
    count_loaded_float(layer_weights.out_bias);
    count_loaded_float(layer_weights.w1_bias);
    count_loaded_float(layer_weights.w2_bias);
    real_cuda_graph.load_layer(layer, layer_weights.view());
    real_vk_graph.load_layer(layer, layer_weights.view());
  }
  const double graph_load_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - graph_load_begin).count();
  CHECK(graph_raw_subnormals == 8495330u);
  CHECK(graph_loaded_subnormals == 0);
  CHECK(graph_loaded_fp32_subnormals == 0);
  CHECK(real_cuda_graph.persistent_bytes() ==
        uint64_t(kGraphLayers) * real_weights.bytes());
  CHECK(real_vk_graph.persistent_bytes() ==
        uint64_t(kGraphLayers) * real_weights.bytes());

  cuda::Stream graph_cuda_stream;
  cuda::DeviceBuffer<float> graph_cuda_tokens(real_input.size()),
      graph_cuda_cosine(real_cosine.size()), graph_cuda_sine(real_sine.size());
  graph_cuda_tokens.copy_from_host(real_input.data(), real_input.size(),
                                   graph_cuda_stream.get());
  graph_cuda_cosine.copy_from_host(real_cosine.data(), real_cosine.size(),
                                   graph_cuda_stream.get());
  graph_cuda_sine.copy_from_host(real_sine.data(), real_sine.size(),
                                 graph_cuda_stream.get());
  const uint64_t graph_token_shape[] = {real_config.sequence, real_config.dim};
  const uint64_t graph_rope_shape[] = {real_config.sequence,
                                       real_config.rope_dim};
  vulkan::DeviceTensor graph_vk_tokens = graph_context.allocate(
      TensorLayout::contiguous(graph_token_shape, 2));
  vulkan::DeviceTensor graph_vk_cosine = graph_context.allocate(
      TensorLayout::contiguous(graph_rope_shape, 2));
  vulkan::DeviceTensor graph_vk_sine = graph_context.allocate(
      TensorLayout::contiguous(graph_rope_shape, 2));
  graph_context.upload(graph_vk_tokens, real_input.data(), real_input.size());
  graph_context.upload(graph_vk_cosine, real_cosine.data(), real_cosine.size());
  graph_context.upload(graph_vk_sine, real_sine.data(), real_sine.size());
  std::vector<float> graph_cuda_boundary(real_input.size()),
      graph_vk_boundary(real_input.size());
  std::array<uint64_t, kGraphLayers> boundary_fnv{};
  auto fnv_words = [](const std::vector<float>& values) {
    uint64_t digest = 1469598103934665603ull;
    for (float value : values) {
      uint32_t bits = 0; std::memcpy(&bits, &value, 4);
      for (int byte = 0; byte < 4; ++byte) {
        digest ^= (bits >> (byte * 8)) & 0xffu;
        digest *= 1099511628211ull;
      }
    }
    return digest;
  };
  const auto boundary_begin = std::chrono::steady_clock::now();
  for (uint32_t layer = 0; layer < kGraphLayers; ++layer) {
    real_cuda_graph.forward_layer_device(
        layer, graph_cuda_tokens.get(), graph_cuda_cosine.get(),
        graph_cuda_sine.get(), graph_cuda_stream.get());
    graph_cuda_tokens.copy_to_host(graph_cuda_boundary.data(),
                                   graph_cuda_boundary.size(),
                                   graph_cuda_stream.get());
    graph_cuda_stream.synchronize();
    vulkan::TensorBatch boundary_batch = graph_context.begin_batch();
    real_vk_graph.record_layer(layer, boundary_batch, graph_vk_tokens,
                               graph_vk_cosine, graph_vk_sine);
    boundary_batch.submit().wait();
    graph_context.download(graph_vk_tokens, graph_vk_boundary.data(),
                           graph_vk_boundary.size());
    CHECK_MSG(std::memcmp(graph_cuda_boundary.data(), graph_vk_boundary.data(),
                          graph_vk_boundary.size() * sizeof(float)) == 0,
              "real VAE ViT graph boundary %u mismatch", layer);
    boundary_fnv[layer] = fnv_words(graph_vk_boundary);
  }
  constexpr std::array<uint64_t, kGraphLayers> kExpectedBoundaryFnv{
      0xc8a7ac3241effbb7ull, 0xe6bdd67a8d48bff8ull,
      0xfe02227922d69136ull, 0x4a4e37dbdc31d38eull,
      0x904f376dd2c88994ull, 0x38f3a254dcaa4b58ull,
      0x1adf972aa1b5ad86ull, 0xcf07b32f966af880ull,
      0x41216b0be919cf32ull, 0xde0123fcd7230526ull,
      0x768a973a476b9f3dull, 0x39f8eec6518d09ecull,
      0xd96137acfd15c70bull, 0x258870835106c183ull,
      0x2b9269a616840b60ull, 0x2f80754cca866626ull,
      0xfa59d4e2bb5827fcull, 0x9fce240021df0759ull,
      0x569c14bf07d8f02full, 0x72148c477fbd0555ull,
      0xc2e6327af72e9180ull, 0x866d0f23de08c438ull,
      0xa4af356bb581aa36ull, 0x9115b305a680b5a3ull,
      0x5d2f5f5b25608da1ull, 0x7dba3acd266c90d6ull,
      0x1bf5ac0ceb451f2aull, 0xf3f0972ba2242d72ull,
      0xa00df63f64401882ull, 0x44c2c7fbe814ef3full,
      0xf408320a5e9c4dc4ull, 0xbb6a8a106ede9fdbull,
      0x2b0694c284a43fdfull, 0x5ece5abf0e1457b4ull,
      0x5cfe899e8a5da408ull, 0x50d92f167ac90922ull};
  CHECK(boundary_fnv == kExpectedBoundaryFnv);
  const double boundary_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - boundary_begin).count();

  graph_cuda_tokens.copy_from_host(real_input.data(), real_input.size(),
                                   graph_cuda_stream.get());
  graph_context.upload(graph_vk_tokens, real_input.data(), real_input.size());
  const auto cuda_graph_begin = std::chrono::steady_clock::now();
  real_cuda_graph.forward_device(
      graph_cuda_tokens.get(), graph_cuda_cosine.get(), graph_cuda_sine.get(),
      graph_cuda_stream.get());
  graph_cuda_tokens.copy_to_host(graph_cuda_boundary.data(),
                                 graph_cuda_boundary.size(),
                                 graph_cuda_stream.get());
  graph_cuda_stream.synchronize();
  const double cuda_graph_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - cuda_graph_begin).count();
  const auto vk_graph_begin = std::chrono::steady_clock::now();
  vulkan::TensorBatch graph_batch = graph_context.begin_batch();
  real_vk_graph.record(graph_batch, graph_vk_tokens, graph_vk_cosine,
                       graph_vk_sine);
  graph_batch.submit().wait();
  const double vk_graph_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - vk_graph_begin).count();
  graph_context.download(graph_vk_tokens, graph_vk_boundary.data(),
                         graph_vk_boundary.size());
  CHECK(std::memcmp(graph_cuda_boundary.data(), graph_vk_boundary.data(),
                    graph_vk_boundary.size() * sizeof(float)) == 0);
  CHECK(fnv_words(graph_vk_boundary) == boundary_fnv.back());

  const uint64_t full_graph_reserved = graph_context.reserved_bytes();
  const uint64_t full_graph_used = graph_context.pooled_used_bytes();
  const uint64_t full_graph_descriptors =
      graph_context.descriptor_set_allocations();
  graph_context.upload(graph_vk_tokens, real_input.data(), real_input.size());
  vulkan::TensorBatch repeat_batch = graph_context.begin_batch();
  real_vk_graph.record(repeat_batch, graph_vk_tokens, graph_vk_cosine,
                       graph_vk_sine);
  repeat_batch.submit().wait();
  CHECK(graph_context.reserved_bytes() == full_graph_reserved);
  CHECK(graph_context.pooled_used_bytes() == full_graph_used);
  CHECK(graph_context.descriptor_set_allocations() == full_graph_descriptors);
  std::printf(
      "  real 36-block graph: load %.1f ms, boundary replay %.1f ms, CUDA %.1f ms, Vulkan %.1f ms, raw subnormals %zu, final FNV64 %016llx, persistent %.1f MiB, Vulkan peak %.1f MiB, pool used/reserved %.1f/%.1f MiB, descriptors %llu\n",
      graph_load_ms, boundary_ms, cuda_graph_ms, vk_graph_ms,
      graph_raw_subnormals,
      static_cast<unsigned long long>(boundary_fnv.back()),
      double(real_vk_graph.persistent_bytes()) / 1048576.0,
      double(real_vk_graph.peak_device_bytes()) / 1048576.0,
      double(full_graph_used) / 1048576.0,
      double(full_graph_reserved) / 1048576.0,
      static_cast<unsigned long long>(full_graph_descriptors));
  std::printf("  graph boundary FNV64:");
  for (uint64_t digest : boundary_fnv)
    std::printf(" %016llx", static_cast<unsigned long long>(digest));
  std::printf("\n");
}

VIDFAB_TEST(cuda_exact_vae_vit_decoder_integration) {
  using namespace vidfab;
  vae::ViTConfig default_config;
  CHECK(default_config.transformer_mode == vae::ViTTransformerMode::kShipped);
  if (!std::getenv("VIDFAB_VAE_VIT_DECODER_REAL")) return;
  int cuda_devices = 0;
  if (cudaGetDeviceCount(&cuda_devices) != cudaSuccess || cuda_devices == 0)
    return;
  const std::filesystem::path checkpoint_path =
      "weights/vae/minimax_h3_video_vae_fp16.safetensors";
  if (!std::filesystem::exists(checkpoint_path)) return;
  SafeTensors checkpoint;
  checkpoint.open(checkpoint_path.string());
#ifdef _WIN32
  const std::array<uint8_t, 32> expected_checkpoint_sha{
      0x7c, 0x1f, 0x13, 0x14, 0x92, 0xe7, 0xed, 0xda,
      0xca, 0xac, 0x90, 0x69, 0xa6, 0x1b, 0x81, 0xbd,
      0xd3, 0x9d, 0xe5, 0xcc, 0x96, 0x56, 0x1e, 0x67,
      0x7c, 0x5e, 0xab, 0x1c, 0xdc, 0xe5, 0xe5, 0x22};
  CHECK(sha256_mapping(checkpoint.mapping_base(), checkpoint.file_size()) ==
        expected_checkpoint_sha);
#endif
  vae::ViTConfig config;
  config.transformer_mode = vae::ViTTransformerMode::kExact;
  vae::ViTDecoder decoder;
  const auto load_begin = std::chrono::steady_clock::now();
  decoder.load(checkpoint, config);
  const double load_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - load_begin).count();
  CHECK(decoder.config().transformer_mode == vae::ViTTransformerMode::kExact);
  std::vector<float> latent(size_t(config.in_channels) * 7 * 16 * 16);
  for (size_t i = 0; i < latent.size(); ++i)
    latent[i] = float(int((i * 37) % 509) - 254) / 512.0f;
  std::vector<float> first, repeat;
  const auto forward_begin = std::chrono::steady_clock::now();
  decoder.forward_window(latent.data(), 7, 16, 16, first);
  const double forward_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - forward_begin).count();
  decoder.forward_window(latent.data(), 7, 16, 16, repeat);
  CHECK(first == repeat);
  const size_t loaded_weight_bytes = decoder.weight_bytes();
  std::vector<float> ragged_latent(size_t(config.in_channels) * 7 * 8 * 16);
  for (size_t i = 0; i < ragged_latent.size(); ++i)
    ragged_latent[i] = float(int((i * 41) % 509) - 254) / 512.0f;
  std::vector<float> ragged, first_after_ragged;
  decoder.forward_window(ragged_latent.data(), 7, 8, 16, ragged);
  decoder.forward_window(latent.data(), 7, 16, 16, first_after_ragged);
  CHECK(first == first_after_ragged);
  CHECK(decoder.weight_bytes() == loaded_weight_bytes);
  uint64_t digest = 1469598103934665603ull;
  for (float value : first) {
    uint32_t bits = 0; std::memcpy(&bits, &value, sizeof(bits));
    for (int byte = 0; byte < 4; ++byte) {
      digest ^= (bits >> (8 * byte)) & 0xffu;
      digest *= 1099511628211ull;
    }
  }
  CHECK(first.size() == 5505024u);
  CHECK(ragged.size() == 2752512u);
  CHECK(digest == 0x4d84e832e07db0a8ull);
  if (vulkan::Instance::available()) {
    vulkan::Instance instance = vulkan::Instance::create();
    const auto physical = instance.enumerate_devices();
    if (!physical.empty() && physical.front().info().timeline_semaphore) {
      vulkan::DeviceOptions options;
      options.enable_timeline_semaphore = true;
      options.enable_shader_int64 = physical.front().info().shader_int64;
      vulkan::Device device = physical.front().create_device(options);
      vulkan::VideoVaeDecoder vk_decoder =
          vulkan::VideoVaeDecoder::create(device, config);
      const auto vk_load_begin = std::chrono::steady_clock::now();
      vk_decoder.load(checkpoint);
      const double vk_load_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - vk_load_begin).count();
      const uint64_t loaded_accounted = vk_decoder.peak_device_bytes();
      const uint64_t loaded_used = vk_decoder.allocator_used_bytes();
      std::vector<float> vk_first, vk_ragged, vk_first_again;
      const auto vk_forward_begin = std::chrono::steady_clock::now();
      vk_decoder.forward_window(latent.data(), 7, 16, 16, vk_first);
      const double vk_forward_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - vk_forward_begin).count();
      const uint64_t one_shape_accounted = vk_decoder.peak_device_bytes();
      const uint64_t one_shape_used = vk_decoder.allocator_used_bytes();
      vk_decoder.forward_window(ragged_latent.data(), 7, 8, 16, vk_ragged);
      const uint64_t two_shape_accounted = vk_decoder.peak_device_bytes();
      const uint64_t two_shape_used = vk_decoder.allocator_used_bytes();
      const uint64_t one_accounted_delta = one_shape_accounted - loaded_accounted;
      const uint64_t one_used_delta = one_shape_used - loaded_used;
      const uint64_t two_accounted_delta = two_shape_accounted - loaded_accounted;
      const uint64_t two_used_delta = two_shape_used - loaded_used;
      CHECK(one_accounted_delta <= one_used_delta);
      CHECK(two_accounted_delta <= two_used_delta);
      CHECK_MSG(one_used_delta <= one_accounted_delta + (32ull << 20),
                "one-shape allocator delta exceeds accounting by %.1f MiB",
                double(one_used_delta) / 1048576.0 -
                    double(one_accounted_delta) / 1048576.0);
      CHECK_MSG(two_used_delta <= two_accounted_delta + (32ull << 20),
                "two-shape allocator delta exceeds accounting by %.1f MiB",
                double(two_used_delta) / 1048576.0 -
                    double(two_accounted_delta) / 1048576.0);
      vk_decoder.forward_window(latent.data(), 7, 16, 16, vk_first_again);
      size_t window_mismatch = first.size();
      for (size_t i = 0; i < first.size(); ++i) {
        if (std::memcmp(&first[i], &vk_first[i], sizeof(float)) != 0) {
          window_mismatch = i;
          break;
        }
      }
      if (window_mismatch != first.size()) {
        uint32_t cb = 0, vb = 0;
        std::memcpy(&cb, &first[window_mismatch], 4);
        std::memcpy(&vb, &vk_first[window_mismatch], 4);
        std::printf("  window mismatch %zu CUDA=%08x Vulkan=%08x\n",
                    window_mismatch, cb, vb);
      }
      CHECK(std::memcmp(first.data(), vk_first.data(), first.size() * 4) == 0);
      CHECK(std::memcmp(ragged.data(), vk_ragged.data(), ragged.size() * 4) == 0);
      CHECK(vk_first == vk_first_again);
      CHECK(vk_decoder.cached_shapes() == 2);
      const uint64_t stable_reserved = vk_decoder.allocator_reserved_bytes();
      const uint64_t stable_descriptors =
          vk_decoder.descriptor_set_allocations();
      vk_decoder.forward_window(latent.data(), 7, 16, 16, vk_first_again);
      CHECK(vk_decoder.allocator_reserved_bytes() == stable_reserved);
      CHECK(vk_decoder.descriptor_set_allocations() == stable_descriptors);

      // Six equal-shape documents exceed one 4096-op transaction (735 ops per
      // document), so this exercises the production 5+1 split and nontrivial
      // output-slot mapping against the exact CUDA implementation.
      constexpr int kSplitBatch = 6;
      std::vector<float> split_latent(size_t(kSplitBatch) * config.in_channels * 7);
      for (size_t i = 0; i < split_latent.size(); ++i)
        split_latent[i] = float(int((i * 53) % 251) - 125) / 256.0f;
      const std::array<size_t, kSplitBatch> split_slots{5, 0, 4, 1, 3, 2};
      std::vector<std::vector<float>> cuda_split(kSplitBatch), vk_split(kSplitBatch);
      const uint64_t before_bad_peak = vk_decoder.peak_device_bytes();
      const uint64_t before_bad_reserved = vk_decoder.allocator_reserved_bytes();
      const uint64_t before_bad_descriptors =
          vk_decoder.descriptor_set_allocations();
      const uint32_t before_bad_shapes = vk_decoder.cached_shapes();
      const std::array<size_t, 2> bad_slots{0, kSplitBatch};
      bool rejected_bad_slots = false;
      try {
        vk_decoder.forward_windows(split_latent.data(), 2, 7, 2, 2,
                                   vk_split, bad_slots.data());
      } catch (const std::out_of_range&) {
        rejected_bad_slots = true;
      }
      CHECK(rejected_bad_slots);
      CHECK(vk_decoder.peak_device_bytes() == before_bad_peak);
      CHECK(vk_decoder.allocator_reserved_bytes() == before_bad_reserved);
      CHECK(vk_decoder.descriptor_set_allocations() == before_bad_descriptors);
      CHECK(vk_decoder.cached_shapes() == before_bad_shapes);
      decoder.forward_windows(split_latent.data(), kSplitBatch, 7, 1, 1,
                              cuda_split, split_slots.data());
      decoder.release_host_registrations();
      vk_decoder.forward_windows(split_latent.data(), kSplitBatch, 7, 1, 1,
                                 vk_split, split_slots.data());
      for (size_t slot = 0; slot < kSplitBatch; ++slot) {
        CHECK(cuda_split[slot].size() == vk_split[slot].size());
        CHECK(std::memcmp(cuda_split[slot].data(), vk_split[slot].data(),
                          cuda_split[slot].size() * sizeof(float)) == 0);
      }
      CHECK(vk_decoder.cached_shapes() == 2);

      // Exercise the shared temporal scheduler, device latent de-normalize,
      // stitch/pixel de-normalize, and the exact Y4M output boundary.
      std::vector<float> normalized(size_t(config.in_channels) * 7);
      for (size_t i = 0; i < normalized.size(); ++i)
        normalized[i] = float(int((i * 29) % 127) - 63) / 128.0f;
      vae::DecodeSchedule decode_schedule;
      decode_schedule.tiling_enabled = false;
      const vae::DecodedVideo cuda_video = decoder.decode(
          normalized.data(), 7, 1, 1, vae::default_video_latents_mean(),
          vae::default_video_latents_std(), decode_schedule);
      const vae::DecodedVideo vk_video = vk_decoder.decode(
          normalized.data(), 7, 1, 1, vae::default_video_latents_mean(),
          vae::default_video_latents_std(), decode_schedule);
      CHECK(cuda_video.channels == vk_video.channels);
      CHECK(cuda_video.frames == vk_video.frames);
      CHECK(cuda_video.height == vk_video.height);
      CHECK(cuda_video.width == vk_video.width);
      CHECK(cuda_video.data.size() == vk_video.data.size());
      CHECK(std::memcmp(cuda_video.data.data(), vk_video.data.data(),
                        cuda_video.data.size() * sizeof(float)) == 0);
      uint64_t pixel_digest = 1469598103934665603ull;
      for (float value : vk_video.data) {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        for (int byte = 0; byte < 4; ++byte) {
          pixel_digest ^= (bits >> (8 * byte)) & 0xffu;
          pixel_digest *= 1099511628211ull;
        }
      }
      CHECK(pixel_digest == 0xf455f77e718d9c21ull);
      const std::filesystem::path cuda_y4m =
          std::filesystem::temp_directory_path() / "vidfab_exact_vae_cuda.y4m";
      const std::filesystem::path vk_y4m =
          std::filesystem::temp_directory_path() / "vidfab_exact_vae_vulkan.y4m";
      std::filesystem::remove(cuda_y4m);
      std::filesystem::remove(vk_y4m);
      video::write_y4m(cuda_y4m.string(), cuda_video.data, cuda_video.frames,
                       cuda_video.height, cuda_video.width);
      vulkan::Yuv420Converter yuv_converter;
      video::write_y4m(vk_y4m.string(), vk_video.data, vk_video.frames,
                       vk_video.height, vk_video.width, {}, &yuv_converter);
      const video::ExactY4mComparison y4m_comparison =
          video::compare_y4m_exact(cuda_y4m.string(), vk_y4m.string());
      CHECK(y4m_comparison.equal());
      std::filesystem::remove(cuda_y4m);
      std::filesystem::remove(vk_y4m);
      const uint64_t final_peak = vk_decoder.peak_device_bytes();
      const uint64_t final_reserved = vk_decoder.allocator_reserved_bytes();
      const uint64_t final_descriptors = vk_decoder.descriptor_set_allocations();
      CHECK(vk_decoder.cached_shapes() == 2);
      CHECK(final_peak <= 5700ull * 1024 * 1024);
      std::printf(
          "  exact Vulkan ViTDecoder: load %.1f ms, forward %.1f ms, persistent/peak %.1f/%.1f MiB, pool used/reserved %.1f/%.1f MiB, descriptors %llu, final %dx%dx%d FNV64 %016llx and Y4M exact\n",
          vk_load_ms, vk_forward_ms,
          double(vk_decoder.persistent_bytes()) / 1048576.0,
          double(final_peak) / 1048576.0,
          double(vk_decoder.allocator_used_bytes()) / 1048576.0,
          double(final_reserved) / 1048576.0,
          static_cast<unsigned long long>(final_descriptors),
          vk_video.frames, vk_video.height, vk_video.width,
          static_cast<unsigned long long>(pixel_digest));
    }
  }
  std::printf(
      "  exact ViTDecoder: load %.1f ms, forward %.1f ms, weights %.1f MiB, output %zu words, ragged output %zu words, FNV64 %016llx\n",
      load_ms, forward_ms, double(decoder.weight_bytes()) / 1048576.0,
      first.size(), ragged.size(), static_cast<unsigned long long>(digest));
}

int main() { return ::vidfab::test::run_all(); }
