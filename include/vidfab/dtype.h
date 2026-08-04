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

// NVFP4 packs two values per byte. Low nibble is the even-indexed element.
inline float f4_lo(uint8_t byte) { return f4_e2m1_to_f32(byte & 0x0Fu); }
inline float f4_hi(uint8_t byte) { return f4_e2m1_to_f32(byte >> 4); }

}  // namespace vidfab
