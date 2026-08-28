#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace vidfab::cuda {

__device__ inline uint64_t round_shift_even(uint64_t value, unsigned shift) {
  const uint64_t quotient = value >> shift;
  const uint64_t remainder = value & ((uint64_t{1} << shift) - 1u);
  const uint64_t halfway = uint64_t{1} << (shift - 1u);
  return quotient + static_cast<uint64_t>(remainder > halfway ||
                                           (remainder == halfway && (quotient & 1u)));
}

__device__ inline int highest_bit(uint32_t value) {
  int result = 0;
  for (uint32_t scan = value; (scan >>= 1u) != 0u;) ++result;
  return result;
}

__device__ inline uint32_t positive_float_div_uint(uint32_t bits, uint32_t divisor) {
  if ((bits & 0x7fffffffu) == 0u) return 0u;
  if ((bits & 0x7f800000u) == 0x7f800000u) return bits;
  uint32_t significand = bits & 0x007fffffu;
  int exponent;
  if ((bits & 0x7f800000u) == 0u) {
    const int top = highest_bit(significand);
    exponent = top - 149;
    significand <<= 23 - top;
  } else {
    exponent = static_cast<int>((bits >> 23u) & 0xffu) - 127;
    significand |= 0x00800000u;
  }
  int ratio_exponent = highest_bit(significand) - highest_bit(divisor);
  const bool below = ratio_exponent >= 0
      ? static_cast<uint64_t>(significand) < (static_cast<uint64_t>(divisor) << ratio_exponent)
      : (static_cast<uint64_t>(significand) << -ratio_exponent) < divisor;
  if (below) --ratio_exponent;
  int output_exponent = exponent - 23 + ratio_exponent;
  if (output_exponent >= -126) {
    const unsigned shift = static_cast<unsigned>(23 - ratio_exponent);
    const uint64_t numerator = static_cast<uint64_t>(significand) << shift;
    uint64_t rounded = numerator / divisor;
    const uint64_t remainder = numerator % divisor;
    rounded += static_cast<uint64_t>(remainder > divisor / 2u ||
        (remainder * 2u == divisor && (rounded & 1u)));
    if (rounded == (1ull << 24)) { rounded >>= 1u; ++output_exponent; }
    return static_cast<uint32_t>(output_exponent + 127) << 23u |
           static_cast<uint32_t>(rounded) & 0x007fffffu;
  }
  const int subnormal_shift = exponent - 23 + 149;
  uint64_t numerator = significand;
  uint64_t denominator = divisor;
  if (subnormal_shift >= 0) numerator <<= subnormal_shift;
  else if (-subnormal_shift < 32) denominator <<= -subnormal_shift;
  else return 0u;
  uint64_t rounded = numerator / denominator;
  const uint64_t remainder = numerator % denominator;
  rounded += static_cast<uint64_t>(remainder > denominator / 2u ||
      (remainder * 2u == denominator && (rounded & 1u)));
  return static_cast<uint32_t>(rounded);
}

__device__ inline float deterministic_divide(float value, uint32_t divisor) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t divided = positive_float_div_uint(bits & 0x7fffffffu, divisor);
  return __uint_as_float(divided | (bits & 0x80000000u));
}

// Private norm-base add: operands are nonnegative, b is a normal positive
// epsilon, and the finite result is normal. This is deliberately not a general
// IEEE-754 addition routine (two-subnormal and overflow cases are out of scope).
__device__ inline uint32_t positive_float_add(uint32_t a_bits, uint32_t b_bits) {
  if ((a_bits & 0x7fffffffu) == 0u) return b_bits;
  if ((b_bits & 0x7fffffffu) == 0u) return a_bits;
  auto decode = [](uint32_t bits, uint32_t* significand, int* exponent) {
    *significand = bits & 0x007fffffu;
    if ((bits & 0x7f800000u) == 0u) {
      const int top = highest_bit(*significand);
      *exponent = top - 149;
      *significand <<= 23 - top;
    } else {
      *exponent = static_cast<int>((bits >> 23u) & 0xffu) - 127;
      *significand |= 0x00800000u;
    }
  };
  uint32_t a, b;
  int ae, be;
  decode(a_bits, &a, &ae); decode(b_bits, &b, &be);
  if (ae < be || (ae == be && a < b)) { uint32_t t=a; a=b; b=t; int te=ae; ae=be; be=te; }
  const unsigned difference = static_cast<unsigned>(ae - be);
  uint64_t a_ext = static_cast<uint64_t>(a) << 3u;
  uint64_t b_ext = static_cast<uint64_t>(b) << 3u;
  if (difference != 0u) {
    if (difference >= 63u) b_ext = 1u;
    else {
      const uint64_t lost = b_ext & ((uint64_t{1} << difference) - 1u);
      b_ext = (b_ext >> difference) | static_cast<uint64_t>(lost != 0u);
    }
  }
  uint64_t sum = a_ext + b_ext;
  if (sum >= (1ull << 27)) { sum = (sum >> 1u) | (sum & 1u); ++ae; }
  uint64_t rounded = sum >> 3u;
  const uint64_t remainder = sum & 7u;
  rounded += static_cast<uint64_t>(remainder > 4u ||
                                   (remainder == 4u && (rounded & 1u)));
  if (rounded == (1ull << 24)) { rounded >>= 1u; ++ae; }
  return static_cast<uint32_t>(ae + 127) << 23u |
         static_cast<uint32_t>(rounded) & 0x007fffffu;
}

// Backend-stable reciprocal square root for normalization. The finite path is
// entirely unsigned-integer Q30 arithmetic: normalized m is in [1, 4), y is
// in (0, 1], and every product is below 2^62. Six fixed Newton steps use RNE
// shifts. Do not replace this with rsqrtf, whose approximation differs between
// CUDA and Vulkan even on the same NVIDIA device.
__device__ inline float deterministic_rsqrt(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude == 0u) return __uint_as_float((bits & 0x80000000u) | 0x7f800000u);
  if ((bits & 0x80000000u) != 0u || magnitude > 0x7f800000u)
    return __uint_as_float(0x7fc00000u);
  if (magnitude == 0x7f800000u) return 0.0f;

  uint32_t significand = magnitude & 0x007fffffu;
  int exponent = 0;
  if (magnitude < 0x00800000u) {
    int leading = 0;
    for (uint32_t scan = significand; (scan >>= 1u) != 0u;) ++leading;
    exponent = leading - 149;
    significand <<= static_cast<unsigned>(23 - leading);
  } else {
    exponent = static_cast<int>((magnitude >> 23u) & 0xffu) - 127;
    significand |= 0x00800000u;
  }
  const int parity = exponent & 1;
  const uint64_t mantissa_q30 = static_cast<uint64_t>(significand) << (7 + parity);
  const int scale_exponent = -(exponent - parity) / 2;
  uint64_t estimate_q30 = parity ? 637534208ull : 872415232ull;
#pragma unroll
  for (int iteration = 0; iteration < 6; ++iteration) {
    const uint64_t square_q30 = round_shift_even(estimate_q30 * estimate_q30, 30);
    const uint64_t product_q30 = round_shift_even(mantissa_q30 * square_q30, 30);
    const uint64_t correction_q30 = 3ull * (1ull << 30) - product_q30;
    estimate_q30 = round_shift_even(estimate_q30 * correction_q30, 31);
  }
  int top = 0;
  for (uint64_t scan = estimate_q30; (scan >>= 1u) != 0u;) ++top;
  const unsigned shift = static_cast<unsigned>(top - 23);
  uint64_t rounded = round_shift_even(estimate_q30, shift);
  if (rounded == (1ull << 24)) {
    rounded >>= 1u;
    ++top;
  }
  const int output_exponent = top - 30 + scale_exponent;
  const uint32_t output_bits =
      static_cast<uint32_t>(output_exponent + 127) << 23u |
      static_cast<uint32_t>(rounded) & 0x007fffffu;
  return __uint_as_float(output_bits);
}

__device__ inline float deterministic_norm_rsqrt(float sum, uint32_t divisor,
                                                  float epsilon) {
  const uint32_t quotient = positive_float_div_uint(__float_as_uint(sum), divisor);
  const uint32_t base = positive_float_add(quotient, __float_as_uint(epsilon));
  return deterministic_rsqrt(__uint_as_float(base));
}

}  // namespace vidfab::cuda
