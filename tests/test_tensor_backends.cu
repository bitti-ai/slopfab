#include "harness.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <filesystem>
#include <string_view>
#include <type_traits>
#include <vector>

#include "vidfab/cuda/device.h"
#include "vidfab/cuda/deterministic_math.cuh"
#include "vidfab/cuda/deterministic_gemm.cuh"
#include "vidfab/cuda/deterministic_attention.cuh"
#include "vidfab/cuda/gemm.cuh"
#include "vidfab/cuda/linear.cuh"
#include "vidfab/cuda/keyframe_encoder.cuh"
#include "vidfab/cuda/nn_kernels.cuh"
#include "vidfab/cuda/vae_kernels.cuh"
#include "vidfab/attention.h"
#include "vidfab/dit/rope.h"
#include "vidfab/dtype.h"
#include "vidfab/nf4.h"
#include "vidfab/safetensors.h"
#include "vidfab/vulkan/linear.h"
#include "vidfab/vulkan/gemm.h"
#include "vidfab/vulkan/tensor.h"

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
                                         int count) {
  const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index < count) output[index] = vidfab::cuda::deterministic_silu(input[index]);
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
  cuda::DeviceBuffer<float> d_input(input.size()), d_output(input.size());
  d_input.copy_from_host(input.data(), input.size());
  deterministic_silu_probe<<<static_cast<unsigned>((input.size() + 255) / 256), 256>>>(
      d_input.get(), d_output.get(), static_cast<int>(input.size()));
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> output(input.size()); d_output.copy_to_host(output.data(), output.size());
  const uint32_t expected_special[] = {0x00000000u, 0x80000000u, 0x7f800000u,
                                       0x80000000u, 0x7fc00000u};
  for (size_t i = 0; i < std::size(expected_special); ++i) {
    uint32_t actual = 0; std::memcpy(&actual, &output[i], 4);
    CHECK_MSG(actual == expected_special[i], "deterministic SiLU special %zu: %08x", i,
              actual);
  }
  auto ordered = [](float value) {
    uint32_t bits = 0; std::memcpy(&bits, &value, 4);
    return (bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u;
  };
  uint32_t max_ulp = 0;
  double max_absolute = 0.0, max_relative = 0.0;
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
      continue;
    }
    uint32_t input_bits = 0; std::memcpy(&input_bits, &input[i], 4);
    if ((input_bits & 0x7fffffffu) < 0x00800000u) {
      uint32_t actual = 0; std::memcpy(&actual, &output[i], 4);
      CHECK(actual == (input_bits & 0x80000000u));
      continue;
    }
    const float reference = rounded_reference;
    const uint32_t a = ordered(output[i]), b = ordered(reference);
    max_ulp = std::max(max_ulp, a > b ? a - b : b - a);
    const double absolute = std::abs(static_cast<double>(output[i]) - reference_double);
    max_absolute = std::max(max_absolute, absolute);
    if (reference_double != 0.0) {
      const double relative = absolute / std::abs(reference_double);
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
  const double cutoff_error = 87.0 * std::exp(-87.0) / (1.0 + std::exp(-87.0));
  CHECK(cutoff_error < 1.5e-36);
  std::printf("  deterministic SiLU: max %u ULP, abs %.3e, relative %.3e; "
              "-87 cutoff %.3e; worst relative x=%g out=%.9g ref=%.9g\n",
              max_ulp, max_absolute, max_relative, cutoff_error,
              worst_relative_input, worst_relative_output, worst_relative_reference);
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

VIDFAB_TEST(cuda_vulkan_exact_h3_attention) {
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
  if (!vk.exact_h3_attention()) return;

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
      CHECK(vk.descriptor_set_allocations() == stable_descriptors);
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

int main() { return ::vidfab::test::run_all(); }
