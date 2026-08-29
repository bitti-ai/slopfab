// Persistent fp16 activation-slot preparation for VAE/ViT GEMM. One
// invocation owns a complete packed fp16 word, so odd tails never race.
struct Parameters {
  uint rows;
  uint in_features;
  uint input_row_offset;
  uint groups_x;
};

[[vk::push_constant]] ConstantBuffer<Parameters> p;
[[vk::binding(0, 0)]] ByteAddressBuffer input_data;
[[vk::binding(1, 0)]] RWByteAddressBuffer output_data;

[numthreads(64, 1, 1)]
void main(uint3 local_id : SV_GroupThreadID, uint3 group_id : SV_GroupID) {
  const uint group = group_id.y * p.groups_x + group_id.x;
  const uint pair = group * 64 + local_id.x;
  const uint count = p.rows * p.in_features;
  const uint first = pair * 2;
  if (first >= count) return;
  const uint source = p.input_row_offset * p.in_features + first;
  const uint lo = f32tof16(asfloat(input_data.Load(source * 4)));
  uint hi = 0;
  if (first + 1 < count)
    hi = f32tof16(asfloat(input_data.Load((source + 1) * 4)));
  output_data.Store(pair * 4, lo | (hi << 16));
}
