// Exact single-frame H3 keyframe-encoder Conv3D. One invocation owns one
// output element. Only the final temporal weight plane contributes at T=1;
// the spatial reduction order matches cuda/keyframe_encoder.cu.

struct Parameters {
  uint in_channels;
  uint out_channels;
  uint input_height;
  uint input_width;
  uint output_height;
  uint output_width;
  uint kernel;
  uint stride;
  uint reflect_padding;
  uint asymmetric_padding;
  uint count;
  uint groups_x;
};

[[vk::push_constant]] ConstantBuffer<Parameters> p;
[[vk::binding(0, 0)]] ByteAddressBuffer input_data;
[[vk::binding(1, 0)]] ByteAddressBuffer weight_data;
[[vk::binding(2, 0)]] ByteAddressBuffer bias_data;
[[vk::binding(3, 0)]] RWByteAddressBuffer output_data;

float load_f32(ByteAddressBuffer data, uint index) {
  return asfloat(data.Load(index * 4u));
}

float load_f16(ByteAddressBuffer data, uint index) {
  const uint pair = data.Load((index >> 1u) * 4u);
  return f16tof32((pair >> ((index & 1u) * 16u)) & 0xffffu);
}

int reflect_index(int value, int extent) {
  if (extent <= 1) return 0;
  while (value < 0 || value >= extent)
    value = value < 0 ? -value : 2 * extent - 2 - value;
  return value;
}

[numthreads(64, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint lane : SV_GroupIndex) {
  const uint index = (group_id.y * p.groups_x + group_id.x) * 64u + lane;
  if (index >= p.count) return;

  const uint ox = index % p.output_width;
  const uint oy = (index / p.output_width) % p.output_height;
  const uint oc = index / (p.output_height * p.output_width);
  float sum = load_f16(bias_data, oc);
  const int pad = p.asymmetric_padding != 0u ? 0 : int(p.kernel / 2u);
  for (uint ic = 0; ic < p.in_channels; ++ic) {
    const uint weight_base =
        ((oc * p.in_channels + ic) * p.kernel * p.kernel * p.kernel) +
        (p.kernel - 1u) * p.kernel * p.kernel;
    for (uint ky = 0; ky < p.kernel; ++ky) {
      int sy = int(oy * p.stride + ky) - pad;
      if (p.reflect_padding != 0u) sy = reflect_index(sy, int(p.input_height));
      else if (sy < 0 || sy >= int(p.input_height)) continue;
      for (uint kx = 0; kx < p.kernel; ++kx) {
        int sx = int(ox * p.stride + kx) - pad;
        if (p.reflect_padding != 0u) sx = reflect_index(sx, int(p.input_width));
        else if (sx < 0 || sx >= int(p.input_width)) continue;
        const uint input_at =
            (ic * p.input_height + uint(sy)) * p.input_width + uint(sx);
        sum = mad(load_f32(input_data, input_at),
                  load_f16(weight_data, weight_base + ky * p.kernel + kx), sum);
      }
    }
  }
  output_data.Store(index * 4u, asuint(sum));
}
