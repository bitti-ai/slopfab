// Deterministic dense row-major NT GEMM.
//
// One lane owns a 2x2 output quad. Every output consumes K in strictly
// increasing order through an explicit fused multiply-add. The 16x16 tile is
// only a load-sharing device: it never changes an output's reduction tree.

struct Parameters {
  uint rows;
  uint out_features;
  uint in_features;
  uint input_row_offset;
  uint output_row_offset;
  uint mode;
  uint unused0;
  uint unused1;
};

[[vk::push_constant]] ConstantBuffer<Parameters> p;
[[vk::binding(0, 0)]] ByteAddressBuffer input_data;
[[vk::binding(1, 0)]] ByteAddressBuffer weight_data;
[[vk::binding(2, 0)]] ByteAddressBuffer bias_data;
[[vk::binding(3, 0)]] RWByteAddressBuffer output_data;

groupshared float input_tile[256];
groupshared float weight_tile[256];

float load_f32(ByteAddressBuffer data, uint index) {
  return asfloat(data.Load(index * 4));
}

float load_bf16(ByteAddressBuffer data, uint index) {
  const uint packed = data.Load((index >> 1) * 4);
  return asfloat(((packed >> ((index & 1) * 16)) & 0xffff) << 16);
}

float load_f16(ByteAddressBuffer data, uint index) {
  const uint packed = data.Load((index >> 1) * 4);
  return f16tof32((packed >> ((index & 1) * 16)) & 0xffff);
}

uint bf16_rte(float value) {
  uint bits = asuint(value);
  const uint magnitude = bits & 0x7fffffff;
  if (magnitude > 0x7f800000) return 0x7fff;
  return (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
}

float round_f16(float value) {
  return f16tof32(f32tof16(value));
}

float load_input(uint index) {
  if (p.mode <= 2) return load_bf16(input_data, index);
  const float value = load_f32(input_data, index);
  return p.mode == 3 ? round_f16(value) : value;
}

float load_weight(uint index) {
  if (p.mode <= 2) return load_bf16(weight_data, index);
  if (p.mode == 3) return load_f16(weight_data, index);
  return load_f32(weight_data, index);
}

float apply_bias(float value, uint column) {
  if (p.mode == 1) return value + load_f32(bias_data, column);
  if (p.mode == 2) return value + load_bf16(bias_data, column);
  if (p.mode == 5) return value + load_f32(bias_data, column);
  return value;
}

void store_bf16(uint index, uint bits) {
  const uint byte_offset = (index >> 1) * 4;
  if ((p.out_features & 1) == 0 && (index & 1) == 0) {
    // Even row widths let one lane own both halves; caller uses store_pair.
    return;
  }
  uint ignored;
  if ((index & 1) == 0) {
    output_data.InterlockedAnd(byte_offset, 0xffff0000, ignored);
    output_data.InterlockedOr(byte_offset, bits, ignored);
  } else {
    output_data.InterlockedAnd(byte_offset, 0x0000ffff, ignored);
    output_data.InterlockedOr(byte_offset, bits << 16, ignored);
  }
}

void store_pair(uint row, uint column, float lo, float hi) {
  const uint first = row * p.out_features + column;
  const uint lo_bits = bf16_rte(apply_bias(asfloat(bf16_rte(lo) << 16), column));
  if (column + 1 < p.out_features) {
    const uint hi_bits = bf16_rte(
        apply_bias(asfloat(bf16_rte(hi) << 16), column + 1));
    if ((p.out_features & 1) == 0) {
      output_data.Store((first >> 1) * 4, lo_bits | (hi_bits << 16));
    } else {
      store_bf16(first, lo_bits);
      store_bf16(first + 1, hi_bits);
    }
  } else {
    store_bf16(first, lo_bits);
  }
}

[numthreads(8, 8, 1)]
void main(uint3 local_id : SV_GroupThreadID, uint3 group_id : SV_GroupID) {
  const uint lane = local_id.y * 8 + local_id.x;
  const uint row0 = group_id.y * 16 + local_id.y;
  const uint row1 = row0 + 8;
  const uint col0 = group_id.x * 16 + local_id.x * 2;
  const uint col1 = col0 + 1;

  precise float sum00 = 0.0;
  precise float sum01 = 0.0;
  precise float sum10 = 0.0;
  precise float sum11 = 0.0;

  for (uint base = 0; base < p.in_features; base += 16) {
    [unroll]
    for (uint load = 0; load < 4; ++load) {
      const uint index = lane + load * 64;
      const uint tile_row = index >> 4;
      const uint tile_k = index & 15;
      const uint k = base + tile_k;
      const uint input_row = group_id.y * 16 + tile_row;
      const uint output_col = group_id.x * 16 + tile_row;
      input_tile[index] = input_row < p.rows && k < p.in_features
          ? load_input((p.input_row_offset + input_row) * p.in_features + k)
          : 0.0;
      weight_tile[index] = output_col < p.out_features && k < p.in_features
          ? load_weight(output_col * p.in_features + k)
          : 0.0;
    }
    GroupMemoryBarrierWithGroupSync();
    const uint remaining = min(16, p.in_features - base);
    [loop]
    for (uint k = 0; k < remaining; ++k) {
      const float a0 = input_tile[local_id.y * 16 + k];
      const float a1 = input_tile[(local_id.y + 8) * 16 + k];
      const float b0 = weight_tile[(local_id.x * 2) * 16 + k];
      const float b1 = weight_tile[(local_id.x * 2 + 1) * 16 + k];
      sum00 = mad(a0, b0, sum00);
      sum01 = mad(a0, b1, sum01);
      sum10 = mad(a1, b0, sum10);
      sum11 = mad(a1, b1, sum11);
    }
    GroupMemoryBarrierWithGroupSync();
  }

  const bool bf16_output = p.mode <= 2;
  if (row0 < p.rows && col0 < p.out_features) {
    const uint output_row = p.output_row_offset + row0;
    if (bf16_output) store_pair(output_row, col0, sum00, sum01);
    else {
      output_data.Store((output_row * p.out_features + col0) * 4,
                        asuint(apply_bias(sum00, col0)));
      if (col1 < p.out_features)
        output_data.Store((output_row * p.out_features + col1) * 4,
                          asuint(apply_bias(sum01, col1)));
    }
  }
  if (row1 < p.rows && col0 < p.out_features) {
    const uint output_row = p.output_row_offset + row1;
    if (bf16_output) store_pair(output_row, col0, sum10, sum11);
    else {
      output_data.Store((output_row * p.out_features + col0) * 4,
                        asuint(apply_bias(sum10, col0)));
      if (col1 < p.out_features)
        output_data.Store((output_row * p.out_features + col1) * 4,
                          asuint(apply_bias(sum11, col1)));
    }
  }
}
