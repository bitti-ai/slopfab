#include "../src/music/runtime.cuh"
#include "slopfab/safetensors_write.h"
#include "slopfab/tensor_convert.h"
#include <cstdio>
#include <fstream>
#include "slopfab/text/tokenizer.h"

int main(int argc, char** argv) {
  try {
    if (argc != 5 && argc != 6)
      throw std::invalid_argument("musicprobe ar|flow|vae MODEL INPUT OUTPUT [TRACE_DIRECTORY]");
    using namespace slopfab;
    music::Runtime rt(argv[2]);
    if (std::string(argv[1]) == "tokenize") {
      text::Tokenizer tokenizer;
      music::load_yue_tokenizer(tokenizer, rt.checkpoint);
      std::ifstream stream(argv[3], std::ios::binary);
      std::string text((std::istreambuf_iterator<char>(stream)), {});
      auto ids = tokenizer.encode(text, false);
      std::vector<float> result(ids.begin(), ids.end());
      write_safetensors(argv[4], {{"output", {int64_t(ids.size())}, result}});
      return 0;
    }
    SafeTensors input;
    input.open(argv[3]);
    std::string mode = argv[1];
    std::vector<float> result;
    std::vector<int64_t> shape;
    if (mode == "vae") {
      result = music::decode_yue(rt, to_f32(input.at("audio_latents")));
      shape = {int64_t(result.size() / 2), 2};
    } else {
      std::vector<int32_t> ids;
      for (float id : to_f32(input.at("tokens")))
        ids.push_back(int32_t(id));
      auto out = music::probe_yue(
          rt, ids, mode == "flow" ? to_f32(input.at("audio_latents")) : std::vector<float>{},
          argc == 6 ? argv[5] : "");
      result = moss::to_host(out);
      shape = {out.rows, out.cols};
    }
    write_safetensors(argv[4], {{"output", shape, result}});
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
