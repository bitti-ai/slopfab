#include "../src/krea/runtime.cuh"
#include "slopfab/tensor_convert.h"
#include "slopfab/safetensors_write.h"
#include <cstdio>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    if (argc != 5)
      throw std::invalid_argument("kreaprobe text|fusion|transformer|vae MODEL INPUT OUTPUT");
    using namespace slopfab;
    using namespace SLOPFAB_KREA_NS;
    std::string mode = argv[1];
#ifdef SLOPFAB_KREA_VULKAN
    Runtime scope(argv[2]);
#endif
    std::vector<float> output;
    std::vector<int64_t> shape;
    if (mode == "text") {
      auto x = encode_prompt(argv[2], "", argv[3], [] {
      });
      output = to_host(x);
      shape = {x.rows / 12, 12, 2560};
    } else {
      SafeTensors input;
      input.open(argv[3]);
      if (mode == "vae") {
        const auto& x = input.at("image_latents");
        int h = int(x.shape.at(0)), w = int(x.shape.at(1));
        output = decode_image(argv[2], to_f32(x), h, w);
        shape = {3, h * 16, w * 16};
      } else {
        Runtime rt(argv[2]);
        const auto& c = input.at("prompt_embedding");
        auto text = fuse_text(rt, from_host(to_f32(c), int(c.shape.at(0)) * 12, 2560));
        if (mode == "fusion") {
          output = to_host(text);
          shape = {text.rows, text.cols};
        } else if (mode == "transformer") {
          const auto& x = input.at("image_latents");
          int h = int(x.shape.at(0)), w = int(x.shape.at(1));
          output = to_host(predict(rt, from_host(to_f32(x), h * w, 64), text, h, w,
                                   to_f32(input.at("sigma"))[0], true, [] {
                                   }));
          shape = {h, w, 64};
        } else
          throw std::invalid_argument("unknown Krea probe mode");
      }
    }
    write_safetensors(argv[4], {{"output", shape, output}});
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
