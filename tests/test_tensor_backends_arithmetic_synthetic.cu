#include "detail/tensor_backends_fixture.h"

SLOPFAB_TEST_CATEGORY(cuda_deterministic_rsqrt_dense_reference, "synthetic") {
  using namespace slopfab;
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0");
    return;
  }
  std::vector<uint32_t> bits = {0x00000000u, 0x80000000u, 0x00000001u, 0x007fffffu,
                                0x00800000u, 0x3f800000u, 0x7f7fffffu, 0x7f800000u,
                                0xff800000u, 0xbf800000u, 0x7fc12345u};
  uint32_t state = 0x12345678u;
  for (uint32_t exponent = 1; exponent < 255; ++exponent) {
    const uint32_t boundary[] = {0u, 1u, 0x003fffffu, 0x00400000u, 0x007ffffeu, 0x007fffffu};
    for (uint32_t mantissa : boundary)
      bits.push_back(exponent << 23u | mantissa);
    for (int sample = 0; sample < 32; ++sample) {
      state = state * 1664525u + 1013904223u;
      bits.push_back(exponent << 23u | (state & 0x007fffffu));
    }
  }
  std::vector<float> input(bits.size()), stable(bits.size()), native(bits.size());
  std::memcpy(input.data(), bits.data(), bits.size() * sizeof(uint32_t));
  cuda::DeviceBuffer<float> d_input(input.size()), d_stable(input.size()), d_native(input.size());
  d_input.copy_from_host(input.data(), input.size());
  deterministic_rsqrt_probe<<<static_cast<unsigned>((input.size() + 255) / 256), 256>>>(
      d_input.get(), d_stable.get(), d_native.get(), static_cast<int>(input.size()));
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  d_stable.copy_to_host(stable.data(), stable.size());
  d_native.copy_to_host(native.data(), native.size());

  const struct {
    size_t index;
    uint32_t expected;
  } exceptional[] = {{0, 0x7f800000u}, {1, 0xff800000u}, {7, 0x00000000u},
                     {8, 0x7fc00000u}, {9, 0x7fc00000u}, {10, 0x7fc00000u}};

  for (const auto& item : exceptional) {
    uint32_t actual = 0;
    std::memcpy(&actual, &stable[item.index], sizeof(actual));
    CHECK_MSG(actual == item.expected, "deterministic rsqrt exceptional %zu: %08x", item.index,
              actual);
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
    max_reference_ulp =
        std::max(max_reference_ulp, reference_bits > stable_bits ? reference_bits - stable_bits
                                                                 : stable_bits - reference_bits);
    max_native_ulp =
        std::max(max_native_ulp,
                 native_bits > stable_bits ? native_bits - stable_bits : stable_bits - native_bits);
    const double exact = 1.0 / std::sqrt(static_cast<double>(input[i]));
    max_relative_error =
        std::max(max_relative_error, std::abs(static_cast<double>(stable[i]) - exact) / exact);
  }
  CHECK_MSG(max_reference_ulp <= 1u, "deterministic rsqrt max reference error %u ULP",
            max_reference_ulp);
  CHECK_MSG(max_native_ulp <= 2u, "deterministic/native rsqrt max difference %u ULP",
            max_native_ulp);
  std::printf("  deterministic rsqrt: reference max %u ULP, native max delta %u ULP, "
              "relative %.3e\n",
              max_reference_ulp, max_native_ulp, max_relative_error);

  // Independent IEEE-RNE references for the integer division and restricted
  // norm-base addition. The curated cross product hits sign, tie/carry,
  // subnormal quotient, mantissa and exponent boundaries; randomized values
  // sample across the full legal divisor range through 2^24.
  const uint32_t curated_values[] = {0x00000001u, 0x007fffffu, 0x00800000u, 0x00800001u,
                                     0x3effffffu, 0x3f000000u, 0x3f000001u, 0x3f7fffffu,
                                     0x3f800000u, 0x3f800001u, 0x4b7fffffu, 0x7f7fffffu};
  const uint32_t curated_divisors[] = {1u,   2u,   3u,   5u,     7u,          9u,
                                       127u, 255u, 257u, 65535u, 0x00ffffffu, 0x01000000u};
  const uint32_t curated_epsilons[] = {0x00800000u, 0x33800000u, 0x358637bdu, 0x3f000000u,
                                       0x3f800000u};
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
    divide_epsilons.push_back(
        curated_epsilons[static_cast<size_t>(state) % std::size(curated_epsilons)]);
  }
  const size_t arithmetic_count = divide_inputs.size();
  cuda::DeviceBuffer<uint32_t> d_divide_inputs(arithmetic_count), d_divisors(arithmetic_count),
      d_epsilons(arithmetic_count), d_positive(arithmetic_count), d_signed(arithmetic_count),
      d_added(arithmetic_count);
  d_divide_inputs.copy_from_host(divide_inputs.data(), arithmetic_count);
  d_divisors.copy_from_host(divide_divisors.data(), arithmetic_count);
  d_epsilons.copy_from_host(divide_epsilons.data(), arithmetic_count);
  deterministic_divide_add_probe<<<static_cast<unsigned>((arithmetic_count + 255) / 256), 256>>>(
      d_divide_inputs.get(), d_divisors.get(), d_epsilons.get(), d_positive.get(), d_signed.get(),
      d_added.get(), static_cast<int>(arithmetic_count));
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
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
    const float expected_positive = static_cast<float>(static_cast<double>(magnitude) /
                                                       static_cast<double>(divide_divisors[i]));
    const float expected_signed = static_cast<float>(static_cast<double>(signed_value) /
                                                     static_cast<double>(divide_divisors[i]));
    const float expected_add =
        static_cast<float>(static_cast<double>(expected_positive) + static_cast<double>(epsilon));
    uint32_t expected_positive_bits = 0, expected_signed_bits = 0, expected_add_bits = 0;
    std::memcpy(&expected_positive_bits, &expected_positive, 4);
    std::memcpy(&expected_signed_bits, &expected_signed, 4);
    std::memcpy(&expected_add_bits, &expected_add, 4);
    CHECK_MSG(positive[i] == expected_positive_bits,
              "RNE positive divide mismatch %zu: %08x != %08x", i, positive[i],
              expected_positive_bits);
    CHECK_MSG(signed_result[i] == expected_signed_bits,
              "RNE signed divide mismatch %zu: %08x != %08x", i, signed_result[i],
              expected_signed_bits);
    CHECK_MSG(add_result[i] == expected_add_bits, "RNE norm-base add mismatch %zu: %08x != %08x", i,
              add_result[i], expected_add_bits);
  }
}

SLOPFAB_TEST_CATEGORY(cuda_deterministic_silu_dense_reference, "synthetic") {
  using namespace slopfab;
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    SKIP_UNSUPPORTED_HARDWARE(
        "unavailable prerequisite: cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0");
    return;
  }
  std::vector<float> input;
  auto push_bits = [&](uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, 4);
    input.push_back(value);
  };
  const uint32_t special[] = {0x00000000u, 0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc12345u};
  for (uint32_t bits : special)
    push_bits(bits);
  for (uint32_t bits : {0x00800000u, 0x80800000u, 0x00ffffffu, 0x80ffffffu, 0x01000000u,
                        0x81000000u, 0x01000001u, 0x81000001u})
    push_bits(bits);
  for (float anchor : {-87.0f, 87.0f, -16.0f, 16.0f}) {
    input.push_back(std::nextafter(anchor, -std::numeric_limits<float>::infinity()));
    input.push_back(anchor);
    input.push_back(std::nextafter(anchor, std::numeric_limits<float>::infinity()));
  }
  constexpr double ln2 = 0.693147180559945309417232121458176568;
  for (int k = -125; k <= 125; ++k) {
    const float anchor = static_cast<float>(k * ln2);
    if (anchor < -87.0f || anchor > 87.0f)
      continue;
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
      d_input.get(), d_output.get(), d_pointwise_output.get(), static_cast<int>(input.size()));
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> output(input.size()), pointwise_output(input.size());
  d_output.copy_to_host(output.data(), output.size());
  d_pointwise_output.copy_to_host(pointwise_output.data(), pointwise_output.size());
  const uint32_t expected_special[] = {0x00000000u, 0x80000000u, 0x7f800000u, 0x80000000u,
                                       0x7fc00000u};
  for (size_t i = 0; i < std::size(expected_special); ++i) {
    uint32_t actual = 0;
    std::memcpy(&actual, &output[i], 4);
    CHECK_MSG(actual == expected_special[i], "deterministic SiLU special %zu: %08x", i, actual);
    uint32_t pointwise_actual = 0;
    std::memcpy(&pointwise_actual, &pointwise_output[i], 4);
    CHECK_MSG(pointwise_actual == expected_special[i], "pointwise SiLU special %zu: %08x", i,
              pointwise_actual);
  }
  auto ordered = [](float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, 4);
    return (bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u;
  };
  uint32_t max_ulp = 0, pointwise_max_ulp = 0;
  double max_absolute = 0.0, max_relative = 0.0;
  double pointwise_max_absolute = 0.0, pointwise_max_relative = 0.0;
  float worst_relative_input = 0.0f, worst_relative_output = 0.0f;
  double worst_relative_reference = 0.0;
  for (size_t i = std::size(expected_special); i < input.size(); ++i) {
    const double x = input[i];
    const double reference_double =
        x < 0.0 ? x * std::exp(x) / (1.0 + std::exp(x)) : x / (1.0 + std::exp(-x));
    if (x <= -87.0) {
      uint32_t actual = 0;
      std::memcpy(&actual, &output[i], 4);
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
      std::memcpy(&actual, &output[i], 4);
      std::memcpy(&input_bits, &input[i], 4);
      CHECK(actual == (input_bits & 0x80000000u));
      std::memcpy(&actual, &pointwise_output[i], 4);
      CHECK(actual == (input_bits & 0x80000000u));
      continue;
    }
    uint32_t input_bits = 0;
    std::memcpy(&input_bits, &input[i], 4);
    if ((input_bits & 0x7fffffffu) < 0x00800000u) {
      uint32_t actual = 0;
      std::memcpy(&actual, &output[i], 4);
      CHECK(actual == (input_bits & 0x80000000u));
      std::memcpy(&actual, &pointwise_output[i], 4);
      CHECK(actual == (input_bits & 0x80000000u));
      continue;
    }
    const float reference = rounded_reference;
    const uint32_t a = ordered(output[i]), b = ordered(reference);
    max_ulp = std::max(max_ulp, a > b ? a - b : b - a);
    const uint32_t pa = ordered(pointwise_output[i]);
    pointwise_max_ulp = std::max(pointwise_max_ulp, pa > b ? pa - b : b - pa);
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
  CHECK_MSG(pointwise_max_ulp <= 3u, "pointwise deterministic SiLU max reference error %u ULP",
            pointwise_max_ulp);
  CHECK(pointwise_max_relative < 2.1e-7);
  const double cutoff_error = 87.0 * std::exp(-87.0) / (1.0 + std::exp(-87.0));
  CHECK(cutoff_error < 1.5e-36);
  std::printf("  deterministic SiLU: max %u ULP, abs %.3e, relative %.3e; "
              "-87 cutoff %.3e; worst relative x=%g out=%.9g ref=%.9g\n",
              max_ulp, max_absolute, max_relative, cutoff_error, worst_relative_input,
              worst_relative_output, worst_relative_reference);
  std::printf("  pointwise SiLU: max %u ULP, abs %.3e, relative %.3e\n", pointwise_max_ulp,
              pointwise_max_absolute, pointwise_max_relative);
}
