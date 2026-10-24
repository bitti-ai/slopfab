// Component-level comparison against independent PyTorch reference outputs.
#include "../src/moss/runtime.cuh"
#include "slopfab/text/tokenizer.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/safetensors_write.h"
#include <cstdio>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    if (argc != 5)
      throw std::invalid_argument("mossprobe text|transformer|vae MODEL INPUT OUTPUT");
    const std::string mode = argv[1];
    slopfab::SafeTensors input;
    input.open(argv[3]);
    std::vector<float> result;
    std::vector<int64_t> shape;
    if (mode == "text") {
      const auto ids = slopfab::to_f32(input.at("tokens"));
      std::vector<int32_t> tokens(ids.begin(), ids.end());
      slopfab::moss::Runtime runtime(argv[2]);
      result = slopfab::moss::to_host(slopfab::moss::encode_text(runtime, tokens));
      shape = {int64_t(tokens.size()), 4096};
    } else {
      const auto& tensor = input.at("image_latents");
      const auto latent = slopfab::to_f32(tensor);
      const int h = int(tensor.shape.at(0)), w = int(tensor.shape.at(1));
      if (mode == "vae") {
        result = slopfab::moss::decode_image(argv[2], latent, h, w);
        shape = {3, h * 16, w * 16};
      } else if (mode == "transformer") {
        const auto& text = input.at("prompt_embedding");
        std::vector<int> skip;
        if (input.find("skip_layers")) {
          const auto values = slopfab::to_f32(input.at("skip_layers"));
          skip.assign(values.begin(), values.end());
        }
        slopfab::moss::Runtime runtime(argv[2]);
        auto prediction = slopfab::moss::image_prediction(
            runtime, slopfab::moss::from_host(latent, h * w, 128),
            slopfab::moss::from_host(slopfab::to_f32(text), int(text.shape[0]), 4096), h, w,
            slopfab::to_f32(input.at("timestep")).at(0), skip);
        result = slopfab::moss::to_host(prediction);
        shape = {h, w, 128};
      } else
        throw std::invalid_argument("unknown probe mode");
    }
    slopfab::write_safetensors(argv[4], {{"output", shape, result}});
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
