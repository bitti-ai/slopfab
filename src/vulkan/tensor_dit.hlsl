// Exact H3 DiT pointwise operations. These kernels deliberately share the
// CUDA exact-mode arithmetic order and keep every activation device-resident.

struct Parameters {
  uint op;
  uint rows;
  uint dim;
  uint mod_rows;
  uint num_t;
  uint num_modality;
  uint num_param;
  uint rank;
  uint count;
  uint groups_x;
  uint unused0;
  uint unused1;
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

float load_bf16(ByteAddressBuffer data, uint index) {
  const uint packed = data.Load((index >> 1) * 4);
  return asfloat(((packed >> ((index & 1) * 16)) & 0xffff) << 16);
}

uint bf16_rte(float value) {
  const uint bits = asuint(value);
  const uint magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) return 0x7fffu;
  return (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
}

void store_bf16(uint index, uint bits) {
  const uint byte_offset = (index >> 1) * 4;
  uint ignored;
  if ((index & 1) == 0) {
    output_data.InterlockedAnd(byte_offset, 0xffff0000u, ignored);
    output_data.InterlockedOr(byte_offset, bits, ignored);
  } else {
    output_data.InterlockedAnd(byte_offset, 0x0000ffffu, ignored);
    output_data.InterlockedOr(byte_offset, bits << 16, ignored);
  }
}

uint64_t round_quotient_even(uint64_t numerator, uint64_t denominator) {
  uint64_t quotient = numerator / denominator;
  const uint64_t remainder = numerator - quotient * denominator;
  const uint64_t complement = denominator - remainder;
  if (remainder > complement ||
      (remainder == complement && (quotient & uint64_t(1)) != uint64_t(0)))
    ++quotient;
  return quotient;
}

float exact_divide(float numerator_value, float denominator_value) {
  const uint numerator_bits = asuint(numerator_value);
  const uint denominator_bits = asuint(denominator_value);
  const uint sign = (numerator_bits ^ denominator_bits) & 0x80000000u;
  const uint numerator_magnitude = numerator_bits & 0x7fffffffu;
  const uint denominator_magnitude = denominator_bits & 0x7fffffffu;
  if (numerator_magnitude < 0x00800000u) return asfloat(sign);
  uint a = (numerator_magnitude & 0x007fffffu) | 0x00800000u;
  const uint b = (denominator_magnitude & 0x007fffffu) | 0x00800000u;
  int exponent = int(numerator_magnitude >> 23u) -
                 int(denominator_magnitude >> 23u) + 127;
  uint shift = 23u;
  if (a < b) { --exponent; shift = 24u; }
  uint64_t significand = round_quotient_even(uint64_t(a) << shift, uint64_t(b));
  if (significand == (uint64_t(1) << 24u)) { significand >>= 1u; ++exponent; }
  if (exponent >= 255) return asfloat(sign | 0x7f800000u);
  if (exponent <= 0) {
    const uint sub_shift = uint(1 - exponent);
    if (sub_shift >= 64u) return asfloat(sign);
    uint64_t base = significand >> sub_shift;
    const uint64_t lost = significand & ((uint64_t(1) << sub_shift) - uint64_t(1));
    const uint64_t halfway = uint64_t(1) << (sub_shift - 1u);
    if (lost > halfway ||
        (lost == halfway && (base & uint64_t(1)) != uint64_t(0))) ++base;
    return asfloat(sign | uint(base));
  }
  return asfloat(sign | (uint(exponent) << 23u) |
                 (uint(significand) & 0x007fffffu));
}

float exp_nonpositive(float value) {
  if (value <= -87.0f) return 0.0f;
  const float scaled = value * 1.4426950408889634f;
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
  const uint bits = asuint(value);
  const uint magnitude = bits & 0x7fffffffu;
  if (magnitude < 0x00800000u) value = asfloat(bits & 0x80000000u);
  if (magnitude > 0x7f800000u) return asfloat(0x7fc00000u);
  if (magnitude == 0x7f800000u)
    return (bits & 0x80000000u) != 0u ? 0.0f : value;
  if (value <= 0.0f) return exp_nonpositive(value);
  if (value >= 87.0f) return asfloat(0x7f800000u);
  return exact_divide(1.0f, exp_nonpositive(-value));
}

[numthreads(64, 1, 1)]
void main(uint3 local_id : SV_GroupThreadID, uint3 group_id : SV_GroupID) {
  const uint group = group_id.y * p.groups_x + group_id.x;
  const uint index = group * 64u + local_id.x;
  if (index >= p.count) return;

  if (p.op == 0u) { // x = fma(gate[selector], branch, x), BF16 RNE
    const uint row = index / p.dim;
    const uint column = index - row * p.dim;
    const uint selector = quaternary.Load(row * 4);
    if (selector >= p.mod_rows) return;
    const float x = load_bf16(primary, index);
    const float branch = load_bf16(secondary, index);
    const float gate = load_f32(tertiary, selector * p.dim + column);
    store_bf16(index, bf16_rte(mad(gate, branch, x)));
  } else if (p.op == 1u) { // deterministic exact SwiGLU
    const uint row = index / p.dim;
    const uint column = index - row * p.dim;
    const uint base = row * (2u * p.dim) + column;
    const float gate = load_bf16(primary, base);
    const float value = load_bf16(primary, base + p.dim);
    const float silu = exact_divide(gate, 1.0f + exact_exp(-gate));
    store_bf16(index, bf16_rte(silu * value));
  } else { // rank-R AdaLN: [out,R] x [T,R] -> [P,T*M,C]
    const uint out_features = p.num_modality * p.num_param * p.dim;
    const uint ti = index / out_features;
    const uint feature = index - ti * out_features;
    float acc = load_f32(secondary, feature);
    [loop]
    for (uint k = 0u; k < p.rank; ++k)
      acc = mad(load_f32(primary, feature * p.rank + k),
                load_f32(tertiary, ti * p.rank + k), acc);
    const uint channel = feature % p.dim;
    const uint rest = feature / p.dim;
    const uint param = rest % p.num_param;
    const uint modality = rest / p.num_param;
    const uint destination =
        ((param * p.num_t + ti) * p.num_modality + modality) * p.dim + channel;
    output_data.Store(destination * 4, asuint(acc));
  }
}
