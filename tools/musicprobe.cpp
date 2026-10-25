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
    const std::string requested=argv[1];
    if(requested=="mm-prompt") {
      auto ids=music::minimax_prompt(rt.checkpoint,"# Warm **piano**\n<|BPM 90|>","[Verse] Hello\n [Chorus] World");
      std::vector<float> values(ids.begin(),ids.end());write_safetensors(argv[4],{{"output",{int64_t(ids.size())},values}});return 0;
    }
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
    if(mode.rfind("mm-",0)==0) {
      music::Tensor out;
      if(mode=="mm-vae") {
        result=music::decode_minimax(rt,to_f32(input.at("audio_latents")));shape={int64_t(result.size()/2),2};
      } else {
        if(mode=="mm-global") {
          std::vector<int32_t> ids;for(float id:to_f32(input.at("tokens")))ids.push_back(int32_t(id));
          out=music::probe_minimax_global(rt,ids);
        } else if(mode=="mm-depth") {
          const auto& t=input.at("sequence");out=music::probe_minimax_depth(rt,moss::from_host(to_f32(t),int(t.shape[0]),int(t.shape[1])));
        } else if(mode=="mm-condition")out=music::minimax_condition(rt,to_f32(input.at("hidden")));
        else if(mode=="mm-flow") {
          auto condition=music::minimax_condition(rt,to_f32(input.at("hidden")));
          const auto& state=input.at("audio_latents");out=music::minimax_prediction(rt,moss::from_host(to_f32(state),int(state.shape[0]),128),condition,.5f);
        } else throw std::invalid_argument("unknown MiniMax probe mode");
        result=moss::to_host(out);shape={out.rows,out.cols};
      }
    } else if (mode == "vae") {
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
