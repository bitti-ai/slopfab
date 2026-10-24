#include "../src/moss/runtime.cuh"
#include "slopfab/w4a8.h"
#include "slopfab/cuda/cublas_dispatch.h"
#include "slopfab/cuda/gemm.cuh"
#include "slopfab/tensor_convert.h"
#include "slopfab/safetensors_write.h"
#include <cstdio>

int main(int argc, char** argv) {
  try {
    if (argc != 5)
      throw std::invalid_argument("w4a8probe MODEL PROJECTION INPUT_OUTPUT_PREFIX");
    using namespace slopfab;
    SafeTensors st, input;
    st.open(argv[1]);
    std::string name = argv[2];
    input.open(argv[3]);
    const auto& tensor = st.at(name + ".weight");
    int out = int(tensor.shape.at(0)), in = int(tensor.shape.at(1)) * 2;
    validate_w4a8_weight(st, name + ".weight", out, in);
    const auto& group = st.at(name + ".weight_s_rel");
    auto channel = to_f32(st.at(name + ".weight_s_channel")),
         book = to_f32(st.at(name + ".weight_codebook"));
    cuda::DeviceBuffer<uint8_t> packed(tensor.nbytes), scales(group.nbytes);
    cuda::DeviceBuffer<float> channels(channel.size()), codebook(book.size());
    packed.copy_from_host(static_cast<const uint8_t*>(tensor.data), tensor.nbytes);
    scales.copy_from_host(static_cast<const uint8_t*>(group.data), group.nbytes);
    channels.copy_from_host(channel.data(), channel.size());
    codebook.copy_from_host(book.data(), book.size());
    const auto& t = input.at("input");
    if (t.shape.size() != 2 || t.shape[1] != in)
      throw std::invalid_argument("input shape");
    auto x = moss::from_host(to_f32(t), int(t.shape[0]), in);
    moss::Tensor y(x.rows, out);
    cuda::QuantWeight w;
    w.format = cuda::QuantFormat::kW4A8;
    w.in_features = in;
    w.out_features = out;
    w.data = packed.get();
    w.block_scale = scales.get();
    w.weight_scale = channels.get();
    w.w4_codebook = codebook.get();
    w.convrot = true;
    w.convrot_group = 256;
    cublasHandle_t handle;
    SLOPFAB_CUBLAS_CHECK(cuda::cublas_create(&handle));
    cuda::LinearRunner runner;
    runner.init(handle, nullptr);
    cuda::Workspace workspace;
    workspace.reserve(cuda::linear_workspace_bytes(w, x.rows, cuda::ComputeType::kBF16));
    runner.forward(w, x.ptr(), x.rows, y.ptr(), workspace);
    write_safetensors(argv[4], {{"output", {x.rows, out}, moss::to_host(y)}});
    cuda::cublas_destroy(handle);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
