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

// Backend-stable SiLU for normalization fusions. The exponential argument is
// always non-positive: range reduction produces r in [0,ln(2)), a fixed
// degree-10 Horner polynomial uses explicit FMAs, and 2^n is assembled by
// integer bits. Values below -87 map to signed zero, keeping the advertised
// finite-normal/zero result domain independent of denormal modes.
__device__ inline float deterministic_exp_nonpositive(float value) {
  if (value <= -87.0f) return 0.0f;
  const float scaled = value * 1.4426950408889634f;
  int exponent = static_cast<int>(scaled);
  if (static_cast<float>(exponent) > scaled) --exponent;
  float remainder = fmaf(-static_cast<float>(exponent), 0.693145751953125f, value);
  remainder = fmaf(-static_cast<float>(exponent), 1.428606765330187e-6f, remainder);
  float polynomial = 2.7557319223985893e-7f;
  polynomial = fmaf(polynomial, remainder, 2.755731922398589e-6f);
  polynomial = fmaf(polynomial, remainder, 2.48015873015873e-5f);
  polynomial = fmaf(polynomial, remainder, 1.984126984126984e-4f);
  polynomial = fmaf(polynomial, remainder, 1.388888888888889e-3f);
  polynomial = fmaf(polynomial, remainder, 8.333333333333333e-3f);
  polynomial = fmaf(polynomial, remainder, 4.166666666666667e-2f);
  polynomial = fmaf(polynomial, remainder, 1.666666666666667e-1f);
  polynomial = fmaf(polynomial, remainder, 0.5f);
  polynomial = fmaf(polynomial, remainder, 1.0f);
  polynomial = fmaf(polynomial, remainder, 1.0f);
  const uint32_t scale_bits = static_cast<uint32_t>(exponent + 127) << 23u;
  return polynomial * __uint_as_float(scale_bits);
}

__device__ inline float deterministic_float_divide(float numerator_value,
                                                    float denominator_value);
__device__ inline float canonicalize_pointwise_float(float value);

// General finite-range exponential for audio Snake parameters. Positive
// arguments are the exactly-rounded reciprocal of the shared negative-domain
// polynomial, so CUDA and Vulkan never call vendor exp implementations.
__device__ inline float deterministic_exp(float value) {
  if (value <= 0.0f) return deterministic_exp_nonpositive(value);
  if (value >= 87.0f) return __uint_as_float(0x7f800000u);
  return deterministic_float_divide(1.0f,
                                    deterministic_exp_nonpositive(-value));
}

// Backend-stable sine for the finite Snake input domain. Range reduction uses
// nearest multiples of pi and a fixed odd degree-11 Horner polynomial. The
// split pi constant and every contraction are explicit FMAs.
__device__ inline float deterministic_sin(float value) {
  value = canonicalize_pointwise_float(value);
  if (!isfinite(value)) return __uint_as_float(0x7fc00000u);
  const float scaled = value * 0.3183098861837907f;
  const int quadrant = scaled >= 0.0f
      ? static_cast<int>(scaled + 0.5f)
      : static_cast<int>(scaled - 0.5f);
  float reduced = fmaf(-static_cast<float>(quadrant),
                       3.141592502593994140625f, value);
  reduced = fmaf(-static_cast<float>(quadrant),
                 1.5099579909783764e-7f, reduced);
  const float square = __fmul_rn(reduced, reduced);
  float polynomial = -2.505210838544172e-8f;
  polynomial = fmaf(polynomial, square, 2.7557319223985893e-6f);
  polynomial = fmaf(polynomial, square, -1.9841269841269841e-4f);
  polynomial = fmaf(polynomial, square, 8.3333333333333332e-3f);
  polynomial = fmaf(polynomial, square, -1.6666666666666666e-1f);
  polynomial = fmaf(polynomial, square, 1.0f);
  float result = __fmul_rn(reduced, polynomial);
  if ((quadrant & 1) != 0) result = -result;
  return canonicalize_pointwise_float(result);
}

__device__ inline float deterministic_snake(float value, float log_alpha,
                                            float log_beta) {
  value = canonicalize_pointwise_float(value);
  const float alpha = deterministic_exp(
      canonicalize_pointwise_float(log_alpha));
  const float beta = deterministic_exp(
      canonicalize_pointwise_float(log_beta));
  const float angle = canonicalize_pointwise_float(__fmul_rn(alpha, value));
  const float sine = deterministic_sin(angle);
  const float square = canonicalize_pointwise_float(__fmul_rn(sine, sine));
  const float denominator = __uint_as_float(positive_float_add(
      __float_as_uint(beta), __float_as_uint(1.0e-9f)));
  const float periodic = deterministic_float_divide(square, denominator);
  return canonicalize_pointwise_float(value + periodic);
}

__device__ inline uint64_t deterministic_round_quotient_even(uint64_t numerator,
                                                              uint64_t denominator) {
  uint64_t quotient = numerator / denominator;
  const uint64_t remainder = numerator - quotient * denominator;
  const uint64_t complement = denominator - remainder;
  if (remainder > complement ||
      (remainder == complement && (quotient & 1ull) != 0ull)) {
    ++quotient;
  }
  return quotient;
}

// Backend-stable IEEE round-to-nearest-even division for the finite-normal
// denominator domain used by exact attention normalization. Zero numerators
// preserve their sign; overflow and subnormal outputs are constructed by bits.
__device__ inline float deterministic_float_divide(float numerator_value,
                                                    float denominator_value) {
  const uint32_t numerator_bits = __float_as_uint(numerator_value);
  const uint32_t denominator_bits = __float_as_uint(denominator_value);
  const uint32_t sign = (numerator_bits ^ denominator_bits) & 0x80000000u;
  const uint32_t numerator_magnitude = numerator_bits & 0x7fffffffu;
  const uint32_t denominator_magnitude = denominator_bits & 0x7fffffffu;
  // Any fp32-subnormal numerator remains below the minimum BF16 result after
  // division by attention's running sum (which is >= 1). Canonicalize it to a
  // signed zero before decoding the implicit significand bit.
  if (numerator_magnitude < 0x00800000u) return __uint_as_float(sign);

  const uint32_t a = (numerator_magnitude & 0x007fffffu) | 0x00800000u;
  const uint32_t b = (denominator_magnitude & 0x007fffffu) | 0x00800000u;
  int exponent = static_cast<int>(numerator_magnitude >> 23u) -
                 static_cast<int>(denominator_magnitude >> 23u) + 127;
  unsigned shift = 23;
  if (a < b) { --exponent; shift = 24; }
  uint64_t significand = deterministic_round_quotient_even(
      static_cast<uint64_t>(a) << shift, static_cast<uint64_t>(b));
  if (significand == (1ull << 24u)) { significand >>= 1u; ++exponent; }
  if (exponent >= 255) return __uint_as_float(sign | 0x7f800000u);
  if (exponent <= 0) {
    const unsigned sub_shift = static_cast<unsigned>(1 - exponent);
    if (sub_shift >= 64) return __uint_as_float(sign);
    uint64_t base = significand >> sub_shift;
    const uint64_t lost_mask = (1ull << sub_shift) - 1ull;
    const uint64_t lost = significand & lost_mask;
    const uint64_t halfway = 1ull << (sub_shift - 1u);
    if (lost > halfway || (lost == halfway && (base & 1ull) != 0ull)) ++base;
    return __uint_as_float(sign | static_cast<uint32_t>(base));
  }
  return __uint_as_float(sign | (static_cast<uint32_t>(exponent) << 23u) |
                         (static_cast<uint32_t>(significand) & 0x007fffffu));
}

__device__ inline float deterministic_silu(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude < 0x00800000u) return __uint_as_float(bits & 0x80000000u);
  if (magnitude > 0x7f800000u) return __uint_as_float(0x7fc00000u);
  if (magnitude == 0x7f800000u)
    return (bits & 0x80000000u) != 0u ? __uint_as_float(0x80000000u) : value;
  float result;
  if (value < 0.0f) {
    const float exponential = deterministic_exp_nonpositive(value);
    result = (value * exponential) / (1.0f + exponential);
  } else {
    const float exponential = deterministic_exp_nonpositive(-value);
    result = value / (1.0f + exponential);
  }
  const uint32_t result_bits = __float_as_uint(result);
  return (result_bits & 0x7fffffffu) < 0x00800000u
             ? __uint_as_float(result_bits & 0x80000000u) : result;
}

// Exact pointwise VAE operations deliberately make the device's denormal and
// NaN behavior irrelevant. Subnormal operands/results become signed zero and
// every NaN becomes the same quiet-NaN payload; finite normals, infinities and
// signed zeros retain their bits. Call at each documented arithmetic boundary.
__device__ inline float canonicalize_pointwise_float(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude < 0x00800000u) return __uint_as_float(bits & 0x80000000u);
  if (magnitude > 0x7f800000u) return __uint_as_float(0x7fc00000u);
  return value;
}

// Pointwise SwiGLU uses the accepted deterministic exponential polynomial but
// defines the sigmoid division with the integer IEEE-RNE helper. This avoids a
// backend-native divide while retaining a single fused tensor pass.
__device__ inline float deterministic_pointwise_silu(float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t magnitude = bits & 0x7fffffffu;
  if (magnitude < 0x00800000u) return __uint_as_float(bits & 0x80000000u);
  if (magnitude > 0x7f800000u) return __uint_as_float(0x7fc00000u);
  if (magnitude == 0x7f800000u)
    return (bits & 0x80000000u) != 0u ? __uint_as_float(0x80000000u) : value;
  float result;
  if (value < 0.0f) {
    const float exponential = deterministic_exp_nonpositive(value);
    result = deterministic_float_divide(
        __fmul_rn(value, exponential),
        __uint_as_float(positive_float_add(__float_as_uint(exponential), 0x3f800000u)));
  } else {
    const float exponential = deterministic_exp_nonpositive(-value);
    result = deterministic_float_divide(
        value,
        __uint_as_float(positive_float_add(__float_as_uint(exponential), 0x3f800000u)));
  }
  return canonicalize_pointwise_float(result);
}

}  // namespace vidfab::cuda
