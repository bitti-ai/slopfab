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

float canonical_bf16(float value) {
  const uint bits = asuint(value);
  const uint magnitude = bits & 0x7fffffffu;
  if (magnitude < 0x00800000u) return asfloat(bits & 0x80000000u);
  if (magnitude > 0x7f800000u) return asfloat(0x7fc00000u);
  return value;
}

float exact_divide(float numerator_value, float denominator_value);
float exact_exp(float value);
float exp_nonpositive(float value);

float exact_silu(float value) {
  value = canonical_bf16(value);
  const uint bits = asuint(value);
  const uint magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) return asfloat(0x7fc00000u);
  if (magnitude == 0x7f800000u)
    return (bits & 0x80000000u) != 0u ? asfloat(0x80000000u) : value;
  return exact_divide(value, 1.0f + exact_exp(-value));
}

float exact_gelu_tanh(float value) {
  value = canonical_bf16(value);
  const uint bits = asuint(value);
  const uint magnitude = bits & 0x7fffffffu;
  if (magnitude > 0x7f800000u) return asfloat(0x7fc00000u);
  if (magnitude == 0x7f800000u)
    return (bits & 0x80000000u) != 0u ? asfloat(0x80000000u) : value;
  precise float square = canonical_bf16(value * value);
  precise float cubic = canonical_bf16(square * value);
  precise float inner = canonical_bf16(value + 0.044715f * cubic);
  precise float angle = canonical_bf16(0.7978845608028654f * inner);
  const uint angle_bits = asuint(angle);
  const uint angle_magnitude = angle_bits & 0x7fffffffu;
  float tanh_value;
  if (angle_magnitude == 0x7f800000u) {
    tanh_value = (angle_bits & 0x80000000u) != 0u ? -1.0f : 1.0f;
  } else {
    precise float exponential = exp_nonpositive(-2.0f * abs(angle));
    precise float numerator = canonical_bf16(1.0f - exponential);
    precise float denominator = canonical_bf16(1.0f + exponential);
    tanh_value = exact_divide(numerator, denominator);
    if ((angle_bits & 0x80000000u) != 0u) tanh_value = -tanh_value;
  }
  precise float half_value = canonical_bf16(0.5f * value);
  precise float one_plus_tanh = canonical_bf16(1.0f + tanh_value);
  return canonical_bf16(half_value * one_plus_tanh);
}

uint exact_residual_bf16(uint index) {
  const float left = canonical_bf16(load_bf16(primary, index));
  const float right = canonical_bf16(load_bf16(secondary, index));
  precise float sum = left + right;
  return bf16_rte(canonical_bf16(sum));
}

uint exact_swiglu_bf16(uint index) {
  const float gate = canonical_bf16(load_bf16(primary, index));
  const float up = canonical_bf16(load_bf16(secondary, index));
  precise float result = exact_silu(gate) * up;
  return bf16_rte(canonical_bf16(result));
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

float euler_canonical(float value) {
  const uint bits = asuint(value);
  const uint magnitude = bits & 0x7fffffffu;
  if (magnitude < 0x00800000u) return asfloat(bits & 0x80000000u);
  if (magnitude >= 0x7f800000u) return asfloat(0x7fc00000u);
  return value;
}

float euler_multiply(float left, float right) {
  left = euler_canonical(left);
  right = euler_canonical(right);
  if (asuint(left) == 0x7fc00000u || asuint(right) == 0x7fc00000u)
    return asfloat(0x7fc00000u);
  precise float product = left * right;
  return euler_canonical(product);
}

float euler_add(float left, float right) {
  left = euler_canonical(left);
  right = euler_canonical(right);
  if (asuint(left) == 0x7fc00000u || asuint(right) == 0x7fc00000u)
    return asfloat(0x7fc00000u);
  precise float sum = left + right;
  return euler_canonical(sum);
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
  } else if (p.op == 2u) { // rank-R AdaLN: [out,R] x [T,R] -> [P,T*M,C]
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
    const uint table_stride = p.unused0 != 0u
        ? p.unused0 : p.num_t * p.num_modality * p.dim;
    const uint destination = param * table_stride +
        (ti * p.num_modality + modality) * p.dim + channel;
    output_data.Store(destination * 4, asuint(acc));
  } else if (p.op == 3u) { // exact rectified-flow Euler, fp32 in place
    // Keep the three reference source expressions separate. In particular,
    // sigma_from_timestep is not reconstructed from the ratio's sigma grid.
    const float sample = euler_canonical(load_f32(primary, index));
    const float velocity = euler_canonical(load_f32(secondary, index));
    const float sigma_from_timestep = euler_canonical(asfloat(p.unused0));
    const float ratio = euler_canonical(asfloat(p.unused1));
    const float scaled_velocity = euler_multiply(sigma_from_timestep, velocity);
    const float denoised = euler_add(sample, scaled_velocity);
    const float retained = euler_multiply(ratio, sample);
    const float one_minus_ratio = euler_add(1.0f, -ratio);
    const float incoming = euler_multiply(one_minus_ratio, denoised);
    const float next = euler_add(retained, incoming);
    output_data.Store(index * 4u, asuint(next));
  } else if (p.op == 4u) { // exact in-place BF16 residual, packed pairs
    const uint first = index * 2u;
    const uint live_count = p.rows * p.dim;
    const uint low = exact_residual_bf16(first);
    const uint high = first + 1u < live_count
        ? exact_residual_bf16(first + 1u) : 0u;
    output_data.Store(index * 4u, low | (high << 16u));
  } else if (p.op == 5u) { // exact split-input BF16 SwiGLU, packed pairs
    const uint first = index * 2u;
    const uint live_count = p.rows * p.dim;
    const uint low = exact_swiglu_bf16(first);
    const uint high = first + 1u < live_count
        ? exact_swiglu_bf16(first + 1u) : 0u;
    output_data.Store(index * 4u, low | (high << 16u));
  } else if (p.op == 6u) { // exact Qwen vision GELU-tanh, packed pairs in place
    const uint first = index * 2u;
    const uint live_count = p.rows * p.dim;
    const uint low = bf16_rte(exact_gelu_tanh(load_bf16(primary, first)));
    const uint high = first + 1u < live_count
        ? bf16_rte(exact_gelu_tanh(load_bf16(primary, first + 1u))) : 0u;
    output_data.Store(index * 4u, low | (high << 16u));
  } else if (p.op == 7u) { // exact learned-position add, packed pairs
    const uint first = index * 2u;
    const uint live_count = p.rows * p.dim;
    uint packed = 0u;
    [unroll]
    for (uint lane = 0u; lane < 2u; ++lane) {
      const uint at = first + lane;
      if (at < live_count) {
        const uint row = at / p.dim;
        const uint column = at - row * p.dim;
        const uint position = tertiary.Load(row * 4u);
        if (position < p.mod_rows) {
          precise float sum = canonical_bf16(load_bf16(primary, at)) +
                              canonical_bf16(load_bf16(
                                  secondary, position * p.dim + column));
          packed |= bf16_rte(canonical_bf16(sum)) << (lane * 16u);
        }
      }
    }
    output_data.Store(index * 4u, packed);
  } else if (p.op == 8u) { // raw fused-QKV part copy, packed pairs
    const uint first = index * 2u;
    const uint live_count = p.rows * p.dim;
    uint packed = 0u;
    [unroll]
    for (uint lane = 0u; lane < 2u; ++lane) {
      const uint at = first + lane;
      if (at < live_count) {
        const uint row = at / p.dim;
        const uint column = at - row * p.dim;
        const uint source = row * (3u * p.dim) + p.unused0 * p.dim + column;
        packed |= (primary.Load((source >> 1u) * 4u) >>
                   ((source & 1u) * 16u) & 0xffffu) << (lane * 16u);
      }
    }
    output_data.Store(index * 4u, packed);
  } else if (p.op == 9u) { // raw merge-four reshape-copy, packed words
    output_data.Store(index * 4u, primary.Load(index * 4u));
  } else { // exact DeepStack scatter-add; row indices must be unique
    const uint row = index / p.dim;
    const uint column = index - row * p.dim;
    const uint destination_row = tertiary.Load(row * 4u);
    if (destination_row >= p.mod_rows) return;
    const uint destination = destination_row * p.dim + column;
    precise float sum = canonical_bf16(load_bf16(primary, destination)) +
                        canonical_bf16(load_bf16(secondary, index));
    store_bf16(destination, bf16_rte(canonical_bf16(sum)));
  }
}
