#include "../src/music/runtime.cuh"
#include "slopfab/safetensors_write.h"
#include "slopfab/tensor_convert.h"
#include <cstdio>
#include <fstream>
#include "slopfab/text/tokenizer.h"
#include "slopfab/sampler/noise.h"
#include <filesystem>

namespace mg = slopfab::SLOPFAB_MUSIC_NS;
#ifdef SLOPFAB_MUSIC_VULKAN
namespace tensor_ops = slopfab::music_gpu;
#else
namespace tensor_ops = slopfab::moss;
#endif
int main(int argc, char** argv) {
  try {
    if (argc != 5 && argc != 6 && !(argc == 7 && std::string(argv[1]) == "mm-hidden"))
      throw std::invalid_argument("musicprobe ar|flow|vae MODEL INPUT OUTPUT [TRACE_DIRECTORY]");
    using namespace slopfab;
    mg::Runtime rt(argv[2]);
    rt.cache_dense_weights = true;
    const std::string requested = argv[1];
    if (requested == "mm-prompt") {
      auto ids = mg::minimax_prompt(rt.checkpoint, "# Warm **piano**\n<|BPM 90|>",
                                       "[Verse] Hello\n [Chorus] World");
      std::vector<float> values(ids.begin(), ids.end());
      write_safetensors(argv[4], {{"output", {int64_t(ids.size())}, values}});
      return 0;
    }
    if (requested == "mm-hidden") {
      rt.cache_dense_weights = true;
      std::ifstream stream(argv[3], std::ios::binary);
      std::string caption((std::istreambuf_iterator<char>(stream)), {});
      auto ids = mg::minimax_prompt(rt.checkpoint, caption, "");
      auto hidden =
          mg::minimax_hidden(rt, ids, MusicSettings{}, argc == 7 ? std::stoi(argv[6]) : 300,
                                42, {}, argc >= 6 ? argv[5] : "");
      write_safetensors(argv[4], {{"hidden", {int64_t(hidden.size() / 32768), 32768}, hidden}});
      return 0;
    }
    if (std::string(argv[1]) == "tokenize") {
      text::Tokenizer tokenizer;
      mg::load_yue_tokenizer(tokenizer, rt.checkpoint);
      std::ifstream stream(argv[3], std::ios::binary);
      std::string text((std::istreambuf_iterator<char>(stream)), {});
      auto ids = tokenizer.encode(text, false);
      std::vector<float> result(ids.begin(), ids.end());
      write_safetensors(argv[4], {{"output", {int64_t(ids.size())}, result}});
      return 0;
    }
    SafeTensors input;
    input.open(argv[3]);
    if (requested == "mm-render") {
      if (argc != 6) throw std::invalid_argument("mm-render DIT HIDDEN OUTPUT.wav VAE");
      GenerateRequest request;
      request.model_family = "minimax_music3";
      request.transformer_path = argv[2];
      request.audio_vae_path = argv[5];
      request.out_path = argv[4];
      request.seed = 42;
      auto hidden = to_f32(input.at("hidden"));
      request.music.duration = double(hidden.size() / 32768) / 25;
      auto plan = resolve_music_plan(request);
      RunOptions options;
#ifdef SLOPFAB_MUSIC_VULKAN
      options.inference_backend = DeviceBackend::kVulkan;
#endif
      options.dump_latents_path = std::filesystem::path(argv[4]).replace_extension(".safetensors").string();
      auto result = mg::run_minimax_from_hidden(request, plan, options, std::move(hidden));
      if (!result.ok) throw std::runtime_error(result.message);
      return 0;
    }
    std::string mode = argv[1];
    std::vector<float> result;
    std::vector<int64_t> shape;
    if (mode.rfind("mm-", 0) == 0) {
      mg::Tensor out;
      if (mode == "mm-sample") {
        rt.cache_dense_weights = true;
        auto condition = mg::minimax_condition(rt, to_f32(input.at("hidden")));
        auto state = to_f32(input.at("audio_latents"));
        auto round = [](float v) {
          return mg::gpu_bf(v);
        };
        for (auto& v : state)
          v = round(v);
        mg::Tensor zero(condition.rows, condition.cols);
        mg::gpu_zero(zero);
        for (int step = 0; step < 30; ++step) {
          auto x = tensor_ops::from_host(state, condition.rows, 128);
          auto c = tensor_ops::to_host(mg::minimax_prediction(rt, x, condition, step / 30.f));
          auto u = tensor_ops::to_host(mg::minimax_prediction(rt, x, zero, step / 30.f));
          for (size_t i = 0; i < state.size(); ++i)
            state[i] = round(state[i] + round(u[i] + round(round(c[i] - u[i]) * 1.7f)) / 30.f);
        }
        result = std::move(state);
        shape = {condition.rows, 128};
      } else if (mode == "mm-noise") {
        const auto& t = input.at("hidden");
        int length = mg::minimax_latent_length(int(t.shape[0]));
        result.resize(size_t(length) * 128);
        sampler::fill_normal(42, sampler::NoiseStream::kAudioLatents, result.data(), result.size());
        shape = {length, 128};
      } else if (mode == "mm-vae") {
        result = mg::decode_minimax(rt, to_f32(input.at("audio_latents")));
        shape = {int64_t(result.size() / 2), 2};
      } else {
        if (mode == "mm-global") {
          std::vector<int32_t> ids;
          for (float id : to_f32(input.at("tokens")))
            ids.push_back(int32_t(id));
          out = mg::probe_minimax_global(rt, ids);
        } else if (mode == "mm-depth") {
          const auto& t = input.at("sequence");
          out = mg::probe_minimax_depth(
              rt, tensor_ops::from_host(to_f32(t), int(t.shape[0]), int(t.shape[1])));
        } else if (mode == "mm-condition")
          out = mg::minimax_condition(rt, to_f32(input.at("hidden")));
        else if (mode == "mm-flow") {
          auto condition = mg::minimax_condition(rt, to_f32(input.at("hidden")));
          const auto& state = input.at("audio_latents");
          out = mg::minimax_prediction(
              rt, tensor_ops::from_host(to_f32(state), int(state.shape[0]), 128), condition,
              input.find("time") ? to_f32(input.at("time"))[0] : .5f, {}, argc == 6 ? argv[5] : "");
        } else
          throw std::invalid_argument("unknown MiniMax probe mode");
        result = tensor_ops::to_host(out);
        shape = {out.rows, out.cols};
      }
    } else if (mode == "vae") {
      result = mg::decode_yue(rt, to_f32(input.at("audio_latents")));
      shape = {int64_t(result.size() / 2), 2};
    } else {
      std::vector<int32_t> ids;
      for (float id : to_f32(input.at("tokens")))
        ids.push_back(int32_t(id));
      auto out = mg::probe_yue(
          rt, ids, mode == "flow" ? to_f32(input.at("audio_latents")) : std::vector<float>{},
          argc == 6 ? argv[5] : "");
      result = tensor_ops::to_host(out);
      shape = {out.rows, out.cols};
    }
    write_safetensors(argv[4], {{"output", shape, result}});
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
