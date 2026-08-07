// Tensor element types and host-side conversions.
//
// The checkpoints we consume mix five storage formats in a single file: bf16
// norms, f16 AdaLN tables, f32 scales, and either f8_e4m3, int8 or packed fp4
// for the large projections. Everything here is host-side reference conversion
// used by the loader and the verification harness; device kernels carry their
// own implementations.
#pragma once

#include <cstdint>
#include <cstring>
#include <string_view>

namespace vidfab {

enum class DType {
  kUnknown,
  kBool,
  kU8,
  kI8,
  kI16,
  kI32,
  kI64,
  kF8E4M3,
  kF8E5M2,
  kF16,
  kBF16,
  kF32,
  kF64,
};

// Maps a safetensors dtype string ("BF16", "F8_E4M3", ...) to the enum.
// Returns kUnknown for anything unrecognised so the loader can report the
// offending name rather than silently misreading bytes.
DType dtype_from_string(std::string_view name);

const char* dtype_name(DType dt);

// Storage size of one element. Packed 4-bit formats have no single-element
// size and are stored as kU8 with a halved trailing dimension, so this always
// reflects the on-disk element.
size_t dtype_size(DType dt);

// --- Host-side numeric conversions -----------------------------------------

inline float bf16_to_f32(uint16_t v) {
  // bfloat16 shares f32's exponent layout: widening is a shift.
  const uint32_t bits = static_cast<uint32_t>(v) << 16;
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

inline uint16_t f32_to_bf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  // Round-to-nearest-even, matching PyTorch's .bfloat16() cast.
  const uint32_t rounding = 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + rounding) >> 16);
}

inline float f16_to_f32(uint16_t v) {
  const uint32_t sign = static_cast<uint32_t>(v & 0x8000u) << 16;
  uint32_t exp = (v >> 10) & 0x1Fu;
  uint32_t mant = v & 0x3FFu;

  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;  // signed zero
    } else {
      // Subnormal: renormalise into f32's exponent range.
      exp = 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FFu;
      bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (mant << 13);  // inf / nan
  } else {
    bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

inline uint16_t f32_to_f16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const uint32_t abs = bits & 0x7FFFFFFFu;
  if (abs >= 0x7F800000u)
    return static_cast<uint16_t>(sign | 0x7C00u | ((abs & 0x7FFFFFu) ? 0x0200u : 0u));
  int exp = static_cast<int>((abs >> 23) & 0xFFu) - 127 + 15;
  uint32_t mant = abs & 0x7FFFFFu;
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant |= 0x800000u;
    const int shift = 14 - exp;
    const uint32_t half_mant = mant >> shift;
    const uint32_t remainder = mant & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1);
    return static_cast<uint16_t>(sign | half_mant +
        (remainder > halfway || (remainder == halfway && (half_mant & 1u))));
  }
  uint32_t half = sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
  const uint32_t remainder = mant & 0x1FFFu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) ++half;
  return static_cast<uint16_t>(half);
}

// float8 E4M3 in the OCP/CUDA variant used by both ComfyUI's fp8 checkpoints
// and NVFP4 block scales: 4 exponent bits, 3 mantissa bits, bias 7, no
// infinities, and 0xFF/0x7F reserved for NaN.
inline float f8_e4m3_to_f32(uint8_t v) {
  const uint32_t sign = static_cast<uint32_t>(v & 0x80u) << 24;
  const uint32_t exp = (v >> 3) & 0x0Fu;
  const uint32_t mant = v & 0x07u;

  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      // Subnormal: value is mant * 2^-9.
      uint32_t e = 1;
      uint32_t m = mant;
      while ((m & 0x08u) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x07u;
      bits = sign | ((e + 127u - 7u) << 23) | (m << 20);
    }
  } else if (exp == 0x0F && mant == 0x07) {
    bits = sign | 0x7FC00000u;  // NaN
  } else {
    bits = sign | ((exp + 127u - 7u) << 23) | (mant << 20);
  }
  float out;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

// float4 E2M1 as used by NVFP4: 1 sign, 2 exponent, 1 mantissa bit. Only
// sixteen representable values, so a table is both simplest and fastest.
inline float f4_e2m1_to_f32(uint8_t nibble) {
  static constexpr float kTable[16] = {
      0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
      -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
  };
  return kTable[nibble & 0x0Fu];
}

// NVFP4 packs two values per byte, and the **high** nibble is the even-indexed
// element. These functions name the nibble's position rather than its index so
// that the mapping is stated once, here, instead of being re-assumed at every
// call site.
//
// This comment used to assert the opposite, and the way that survived is worth
// recording. A swapped nibble order leaves the value histogram almost
// untouched, so every aggregate statistic stays healthy and the error is
// visible only elementwise — which is why it was found by scoring candidate
// layouts by elementwise correlation against the int8 build of the *same*
// model. Right separated from wrong by roughly a thousandfold.
//
// Measured independently on both checkpoints, which have different quantiser
// provenance (a 19-byte `comfy_quant` with no `pre_quant_scale` on the
// transformer, 55 bytes with one on the text encoder), and they agree. So this
// is a constant, not a per-file property, and a per-file switch would be a
// distinction with no evidence behind it. A 332-tensor bitwise-identical
// control confirms the fp8 and nvfp4 transformers really are one model, which
// is what makes that cross-checkpoint comparison mean anything.
inline float f4_lo(uint8_t byte) { return f4_e2m1_to_f32(byte & 0x0Fu); }
inline float f4_hi(uint8_t byte) { return f4_e2m1_to_f32(byte >> 4); }

}  // namespace vidfab
