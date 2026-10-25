#include "slopfab/vulkan/linear.h"
#include "slopfab/vulkan/runtime.h"
#include "slopfab/w4a8.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/safetensors_write.h"
#include <cstdio>

int main(int argc, char** argv) {
  try {
    if (argc != 5) throw std::invalid_argument("vulkan_w4a8probe MODEL PROJECTION INPUT OUTPUT");
    using namespace slopfab;
    using namespace slopfab::vulkan;
    SafeTensors st, input;
    st.open(argv[1]); input.open(argv[3]);
    const std::string name = argv[2];
    const auto& tensor = st.at(name + ".weight");
    const auto state = read_packed_int8_state(st, name + ".weight", "probe");
    const uint32_t out = uint32_t(tensor.shape.at(0)),
                   in = uint32_t(tensor.shape.at(1)) * 8 / state.bits;
    validate_packed_int8_weight(st, name + ".weight", out, in);
    auto channel = to_f32(st.at(name + ".weight_s_channel"));
    auto book = state.bits == 4 ? to_f32(st.at(name + ".weight_codebook")) : std::vector<float>{};
    const auto& group = st.at(name + ".weight_s_rel");
    const auto& t = input.at("input");
    if (t.shape.size() != 2 || t.shape[1] != in) throw std::invalid_argument("input shape");
    const uint32_t rows = uint32_t(t.shape[0]);
    auto instance = Instance::create();
    auto physical = instance.enumerate_devices();
    if (physical.empty()) throw std::runtime_error("No Vulkan device");
    DeviceOptions options; options.enable_timeline_semaphore = true;
    auto device = physical.front().create_device(options);
    TensorContext context(device);
    LinearWeightUpload u;
    u.format = state.bits == 6 ? LinearWeightFormat::kW6A8 : LinearWeightFormat::kW4A8;
    u.packed_group_size = state.group_size;
    u.out_features = out;
    u.in_features = in;
    u.data = tensor.data; u.data_bytes = tensor.nbytes;
    u.block_scale = static_cast<const uint8_t*>(group.data); u.block_scale_count = group.nbytes;
    u.weight_scale = channel.data(); u.weight_scale_count = channel.size();
    u.w4_codebook = book.empty() ? nullptr : book.data();
    u.w4_codebook_count = book.size();
    u.convrot = true; u.convrot_group = 256;
    auto weight = LinearWeight::upload(context, u);
    auto matrix = [&](uint32_t columns, ScalarType type) {
      const uint64_t shape[] = {rows, columns};
      return context.allocate(TensorLayout::contiguous(shape, 2), type);
    };
    auto x = matrix(in, ScalarType::kBFloat16), rotated = matrix(in, ScalarType::kBFloat16);
    auto scratch = matrix(in + 1, ScalarType::kFloat32), y = matrix(out, ScalarType::kBFloat16);
    auto values = to_f32(t);
    std::vector<uint16_t> bits(values.size());
    for (size_t i = 0; i < bits.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    context.upload_bytes(x, bits.data(), bits.size() * 2);
    auto batch = context.begin_batch();
    weight.apply_convrot(batch, x, rotated);
    weight.forward_w4a8(batch, rotated, scratch, y);
    batch.submit().wait();
    bits.resize(size_t(rows) * out); values.resize(bits.size());
    context.download_bytes(y, bits.data(), bits.size() * 2);
    for (size_t i = 0; i < bits.size(); ++i) values[i] = bf16_to_f32(bits[i]);
    write_safetensors(argv[4], {{"output", {rows, out}, std::move(values)}});
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
