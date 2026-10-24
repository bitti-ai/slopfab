#include "../src/fibo/runtime.cuh"
#include "slopfab/tensor_convert.h"
#include "slopfab/safetensors_write.h"
#include <cstdio>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    if (argc != 5)
      throw std::invalid_argument(
          "fiboprobe perceiver|transformer|encode|vae|text MODEL INPUT OUTPUT");
    using namespace slopfab;
    using namespace fibo;
    std::string mode = argv[1];
    std::vector<float> output;
    std::vector<int64_t> shape;
    if (mode == "text") {
      auto context =
          encode_prompt(argv[2], "", "Make the object blue", {load_reference_image(argv[3])}, [] {
          });
      Tensor packed(context[0].rows, 46080);
      for (int i = 0; i < 6; ++i)
        SLOPFAB_CUDA_CHECK(cudaMemcpy2D(packed.ptr() + i * 7680, 46080 * 2, context[i].ptr(),
                                        7680 * 2, 7680 * 2, packed.rows, cudaMemcpyDeviceToDevice));
      output = to_host(packed);
      shape = {packed.rows, packed.cols};
    } else if (mode == "encode") {
      auto image = load_reference_image(argv[3]);
      output = moss::encode_flux_image(argv[2], image.pixels, image.height, image.width);
      shape = {image.height / 16, image.width / 16, 128};
    } else {
      SafeTensors input;
      input.open(argv[3]);
      if (mode == "vae") {
        const auto& t = input.at("image_latents");
        int h = int(t.shape[0]), w = int(t.shape[1]);
        output = moss::decode_image(argv[2], to_f32(t), h, w);
        shape = {3, h * 16, w * 16};
      } else {
        Runtime rt(argv[2]);
        const auto& t = input.at("prompt_embedding");
        auto packed = from_host(to_f32(t), int(t.shape[0]), int(t.shape[1]));
        auto gist = perceiver(rt, moss::slice(packed, 0, 7680), true);
        if (mode == "perceiver") {
          output = to_host(gist);
          shape = {gist.rows, gist.cols};
        } else if (mode == "transformer") {
          std::vector<Tensor> context;
          for (int i = 0; i < 6; ++i)
            context.push_back(moss::slice(packed, i * 7680, 7680));
          const auto& x = input.at("image_latents");
          int h = int(x.shape[0]), w = int(x.shape[1]);
          const auto& ref = input.at("reference_latents");
          auto y = predict(rt, from_host(to_f32(x), h * w, 128), h, w,
                           {{int(ref.shape[0]), int(ref.shape[1]), to_f32(ref)}}, context, gist,
                           to_f32(input.at("sigma"))[0], true, [] {
                           });
          output = to_host(y);
          shape = {h, w, 128};
        } else
          throw std::invalid_argument("unknown probe mode");
      }
    }
    write_safetensors(argv[4], {{"output", shape, output}});
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
