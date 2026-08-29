// Exact audio-VAE primitives. One invocation owns one output element; every
// convolution reduction visits channels then taps in ascending order with an
// explicit FMA, matching the CUDA exact baseline.

struct Parameters {
  uint op;
  uint batch;
  uint in_channels;
  uint out_channels;
  uint length_in;
  uint length_out;
  uint kernel;
  uint padding_or_stride;
  uint dilation_or_padding;
  uint count;
  uint groups_x;
  uint scalar_bits;
};

[[vk::push_constant]] ConstantBuffer<Parameters> p;
[[vk::binding(0, 0)]] ByteAddressBuffer primary;
[[vk::binding(1, 0)]] ByteAddressBuffer secondary;
[[vk::binding(2, 0)]] ByteAddressBuffer tertiary;
[[vk::binding(3, 0)]] ByteAddressBuffer quaternary;
[[vk::binding(4, 0)]] RWByteAddressBuffer output_data;

float load_f32(ByteAddressBuffer data, uint index) {
  return asfloat(data.Load(index * 4));
}
void store_f32(uint index, float value) {
  output_data.Store(index * 4, asuint(value));
}
float canonical(float value) {
  uint bits = asuint(value);
  uint magnitude = bits & 0x7fffffffu;
  if (magnitude < 0x00800000u) return asfloat(bits & 0x80000000u);
  if (magnitude > 0x7f800000u) return asfloat(0x7fc00000u);
  return value;
}
float exact_mad(float a, float b, float c) {
  a = canonical(a); b = canonical(b); c = canonical(c);
  if ((asuint(a) & 0x7fffffffu) > 0x7f800000u ||
      (asuint(b) & 0x7fffffffu) > 0x7f800000u ||
      (asuint(c) & 0x7fffffffu) > 0x7f800000u)
    return asfloat(0x7fc00000u);
  return canonical(mad(a, b, c));
}

uint64_t round_quotient_even(uint64_t numerator, uint64_t denominator) {
  uint64_t quotient = numerator / denominator;
  uint64_t remainder = numerator - quotient * denominator;
  uint64_t complement = denominator - remainder;
  if (remainder > complement ||
      (remainder == complement && (quotient & uint64_t(1)) != uint64_t(0)))
    ++quotient;
  return quotient;
}

float exact_divide(float numerator_value, float denominator_value) {
  uint numerator_bits = asuint(numerator_value);
  uint denominator_bits = asuint(denominator_value);
  uint sign = (numerator_bits ^ denominator_bits) & 0x80000000u;
  uint numerator_magnitude = numerator_bits & 0x7fffffffu;
  uint denominator_magnitude = denominator_bits & 0x7fffffffu;
  if (numerator_magnitude < 0x00800000u) return asfloat(sign);
  uint a = (numerator_magnitude & 0x007fffffu) | 0x00800000u;
  uint b = (denominator_magnitude & 0x007fffffu) | 0x00800000u;
  int exponent = int(numerator_magnitude >> 23u) -
                 int(denominator_magnitude >> 23u) + 127;
  uint shift = 23u;
  if (a < b) { --exponent; shift = 24u; }
  uint64_t significand = round_quotient_even(uint64_t(a) << shift, uint64_t(b));
  if (significand == (uint64_t(1) << 24u)) { significand >>= 1u; ++exponent; }
  if (exponent >= 255) return asfloat(sign | 0x7f800000u);
  if (exponent <= 0) {
    uint sub_shift = uint(1 - exponent);
    if (sub_shift >= 64u) return asfloat(sign);
    uint64_t base = significand >> sub_shift;
    uint64_t lost = significand & ((uint64_t(1) << sub_shift) - uint64_t(1));
    uint64_t halfway = uint64_t(1) << (sub_shift - 1u);
    if (lost > halfway ||
        (lost == halfway && (base & uint64_t(1)) != uint64_t(0))) ++base;
    return asfloat(sign | uint(base));
  }
  return asfloat(sign | (uint(exponent) << 23u) |
                 (uint(significand) & 0x007fffffu));
}

uint positive_add(uint a_bits, uint b_bits) {
  if ((a_bits & 0x7fffffffu) == 0u) return b_bits;
  if ((b_bits & 0x7fffffffu) == 0u) return a_bits;
  uint a = (a_bits & 0x007fffffu) | 0x00800000u;
  uint b = (b_bits & 0x007fffffu) | 0x00800000u;
  int ae = int((a_bits >> 23u) & 0xffu) - 127;
  int be = int((b_bits >> 23u) & 0xffu) - 127;
  if (ae < be || (ae == be && a < b)) {
    uint t = a; a = b; b = t; int te = ae; ae = be; be = te;
  }
  uint difference = uint(ae - be);
  uint64_t a_ext = uint64_t(a) << 3u;
  uint64_t b_ext = uint64_t(b) << 3u;
  if (difference != 0u) {
    if (difference >= 63u) b_ext = uint64_t(1);
    else {
      uint64_t lost = b_ext & ((uint64_t(1) << difference) - uint64_t(1));
      b_ext = (b_ext >> difference) | uint64_t(lost != uint64_t(0));
    }
  }
  uint64_t sum = a_ext + b_ext;
  if (sum >= (uint64_t(1) << 27u)) {
    sum = (sum >> 1u) | (sum & uint64_t(1)); ++ae;
  }
  uint64_t rounded = sum >> 3u;
  uint64_t remainder = sum & uint64_t(7);
  rounded += uint64_t(remainder > uint64_t(4) ||
      (remainder == uint64_t(4) && (rounded & uint64_t(1)) != uint64_t(0)));
  if (rounded == (uint64_t(1) << 24u)) { rounded >>= 1u; ++ae; }
  return (uint(ae + 127) << 23u) | (uint(rounded) & 0x007fffffu);
}

float exp_nonpositive(float value) {
  if (value <= -87.0f) return 0.0f;
  float scaled = value * 1.4426950408889634f;
  int exponent = int(scaled);
  if (float(exponent) > scaled) --exponent;
  float remainder = mad(-float(exponent), 0.693145751953125f, value);
  remainder = mad(-float(exponent), 1.428606765330187e-6f, remainder);
  float polynomial = 2.7557319223985893e-7f;
  polynomial = mad(polynomial, remainder, 2.755731922398589e-6f);
  polynomial = mad(polynomial, remainder, 2.48015873015873e-5f);
  polynomial = mad(polynomial, remainder, 1.984126984126984e-4f);
  polynomial = mad(polynomial, remainder, 1.388888888888889e-3f);
  polynomial = mad(polynomial, remainder, 8.333333333333333e-3f);
  polynomial = mad(polynomial, remainder, 4.166666666666667e-2f);
  polynomial = mad(polynomial, remainder, 1.666666666666667e-1f);
  polynomial = mad(polynomial, remainder, 0.5f);
  polynomial = mad(polynomial, remainder, 1.0f);
  polynomial = mad(polynomial, remainder, 1.0f);
  return polynomial * asfloat(uint(exponent + 127) << 23u);
}
float exact_exp(float value) {
  uint bits = asuint(value);
  uint magnitude = bits & 0x7fffffffu;
  if (magnitude < 0x00800000u) value = asfloat(bits & 0x80000000u);
  if (magnitude > 0x7f800000u) return asfloat(0x7fc00000u);
  if (magnitude == 0x7f800000u)
    return (bits & 0x80000000u) != 0u ? 0.0f : value;
  if (value <= 0.0f) return exp_nonpositive(value);
  if (value >= 87.0f) return asfloat(0x7f800000u);
  return exact_divide(1.0f, exp_nonpositive(-value));
}
float exact_sin(float value) {
  value = canonical(value);
  if ((asuint(value) & 0x7fffffffu) >= 0x7f800000u)
    return asfloat(0x7fc00000u);
  if ((asuint(value) & 0x7fffffffu) >= 0x4f000000u)
    return asfloat(0x7fc00000u);
  float scaled = value * 0.3183098861837907f;
  int quadrant = scaled >= 0.0f ? int(scaled + 0.5f) : int(scaled - 0.5f);
  float reduced = mad(-float(quadrant), 3.141592502593994140625f, value);
  reduced = mad(-float(quadrant), 1.5099579909783764e-7f, reduced);
  float square = reduced * reduced;
  float polynomial = -2.505210838544172e-8f;
  polynomial = mad(polynomial, square, 2.7557319223985893e-6f);
  polynomial = mad(polynomial, square, -1.9841269841269841e-4f);
  polynomial = mad(polynomial, square, 8.3333333333333332e-3f);
  polynomial = mad(polynomial, square, -1.6666666666666666e-1f);
  polynomial = mad(polynomial, square, 1.0f);
  float result = reduced * polynomial;
  if ((quadrant & 1) != 0) result = -result;
  return canonical(result);
}
float snake(float value, uint channel) {
  value = canonical(value);
  if ((asuint(value) & 0x7fffffffu) > 0x7f800000u)
    return asfloat(0x7fc00000u);
  float alpha = exact_exp(canonical(load_f32(tertiary, channel)));
  float beta = exact_exp(canonical(load_f32(quaternary, channel)));
  if ((asuint(alpha) & 0x7fffffffu) > 0x7f800000u ||
      (asuint(beta) & 0x7fffffffu) > 0x7f800000u)
    return asfloat(0x7fc00000u);
  float sine = exact_sin(canonical(alpha * value));
  if ((asuint(sine) & 0x7fffffffu) > 0x7f800000u)
    return asfloat(0x7fc00000u);
  float square = canonical(sine * sine);
  float denominator = asfloat(positive_add(asuint(beta), asuint(1.0e-9f)));
  float periodic = (asuint(beta) & 0x7fffffffu) == 0x7f800000u
      ? 0.0f : exact_divide(square, denominator);
  return canonical(value + periodic);
}

[numthreads(64, 1, 1)]
void main(uint3 local_id : SV_GroupThreadID, uint3 group_id : SV_GroupID) {
  uint i = (group_id.y * p.groups_x + group_id.x) * 64u + local_id.x;
  if (i >= p.count) return;

  if (p.op == 0u) { // Conv1D, W [Cout,Cin,K]
    uint n = i % p.length_out;
    uint row = i / p.length_out;
    uint co = row % p.out_channels;
    uint b = row / p.out_channels;
    float accumulator = p.scalar_bits != 0u
        ? canonical(load_f32(tertiary, co)) : 0.0f;
    for (uint ci = 0; ci < p.in_channels; ++ci) {
      for (uint k = 0; k < p.kernel; ++k) {
        int t = int(n + k * p.dilation_or_padding) - int(p.padding_or_stride);
        float x = 0.0f;
        if (t >= 0 && t < int(p.length_in))
          x = canonical(load_f32(primary,
              (b * p.in_channels + ci) * p.length_in + uint(t)));
        float w = canonical(load_f32(secondary,
            (co * p.in_channels + ci) * p.kernel + k));
        accumulator = exact_mad(x, w, accumulator);
      }
    }
    store_f32(i, accumulator);
  } else if (p.op == 1u) { // ConvTranspose1D, W [Cin,Cout,K]
    uint n = i % p.length_out;
    uint row = i / p.length_out;
    uint co = row % p.out_channels;
    uint b = row / p.out_channels;
    float accumulator = p.scalar_bits != 0u
        ? canonical(load_f32(tertiary, co)) : 0.0f;
    uint stride = p.padding_or_stride;
    uint padding = p.dilation_or_padding;
    uint m = n + padding;
    uint phase = m % stride;
    int j0 = int(m / stride);
    for (uint ci = 0; ci < p.in_channels; ++ci) {
      uint tap_index = 0;
      for (uint k = phase; k < p.kernel; k += stride, ++tap_index) {
        int j = j0 - int(tap_index);
        if (j < 0) break;
        if (j >= int(p.length_in)) continue;
        float x = canonical(load_f32(primary,
            (b * p.in_channels + ci) * p.length_in + uint(j)));
        float w = canonical(load_f32(secondary,
            (ci * p.out_channels + co) * p.kernel + k));
        accumulator = exact_mad(x, w, accumulator);
      }
    }
    store_f32(i, accumulator);
  } else if (p.op == 2u) { // add
    store_f32(i, canonical(canonical(load_f32(primary, i)) +
                           canonical(load_f32(secondary, i))));
  } else if (p.op == 3u) { // scale
    store_f32(i, canonical(canonical(load_f32(primary, i)) *
                           canonical(asfloat(p.scalar_bits))));
  } else if (p.op == 4u) { // clamp; finite exact domain
    float value = canonical(load_f32(primary, i));
    store_f32(i, min(max(value, asfloat(p.kernel)),
                     asfloat(p.padding_or_stride)));
  } else if (p.op == 5u) { // [B,1,T] -> [T,B]
    uint b = i / p.length_in;
    uint t = i - b * p.length_in;
    store_f32(t * p.batch + b, canonical(load_f32(primary, i)));
  } else if (p.op == 6u) { // SnakeBeta
    uint channel = (i / p.length_in) % p.out_channels;
    store_f32(i, snake(load_f32(primary, i), channel));
  } else if (p.op == 7u) { // 2x AA upsample + SnakeBeta
    uint n = i % p.length_out;
    uint row = i / p.length_out;
    uint channel = row % p.out_channels;
    uint b = row / p.out_channels;
    uint m = n + 15u;
    uint parity = m & 1u;
    int j = int(m >> 1u) - 5;
    float accumulator = 0.0f;
    for (uint tap = 0; tap < 6u; ++tap) {
      int source = j - int(tap);
      source = max(0, min(source, int(p.length_in) - 1));
      float x = canonical(load_f32(primary,
          (b * p.out_channels + channel) * p.length_in + uint(source)));
      float w = canonical(load_f32(secondary, parity + 2u * tap));
      accumulator = exact_mad(x, w, accumulator);
    }
    store_f32(i, snake(canonical(accumulator * 2.0f), channel));
  } else { // 12-tap AA downsample
    uint n = i % p.length_out;
    uint row = i / p.length_out;
    uint channel = row % p.out_channels;
    uint b = row / p.out_channels;
    float accumulator = 0.0f;
    for (uint k = 0; k < 12u; ++k) {
      int source = int(2u * n + k) - 5;
      source = max(0, min(source, int(p.length_in) - 1));
      float x = canonical(load_f32(primary,
          (b * p.out_channels + channel) * p.length_in + uint(source)));
      float w = canonical(load_f32(secondary, k));
      accumulator = exact_mad(x, w, accumulator);
    }
    store_f32(i, accumulator);
  }
}
