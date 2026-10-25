#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace slopfab {
class SafeTensors;

namespace text {
class Tokenizer;
}
struct GenerateRequest;
struct GeneratePlan;

// Duration is a generation ceiling. EOS may finish a song sooner.
struct MusicSettings {
  std::string lyrics;
  std::string planning = "full"; // YuE2: full, melody, off
  std::string abc;
  double duration = 30;
  float temperature = 1.f;
  float top_p = .95f;
  int top_k = 0; // zero selects 100 for YuE2, 50 for MiniMax Music3
  float repetition_penalty = 1.2f;
  float semantic_guidance = 0.f; // zero selects the model's recipe
  float flow_guidance = 0.f; // MiniMax Music3: zero selects 1.7
  int abc_max_tokens = 4096;
};

MusicSettings parse_music_settings(const std::string& json);
void validate_music_settings(const MusicSettings& settings);
GeneratePlan resolve_music_plan(const GenerateRequest& request);

namespace music {
void load_yue_tokenizer(text::Tokenizer&, const SafeTensors&);
std::vector<int32_t> minimax_prompt(const SafeTensors&, const std::string& caption, const std::string& lyrics);
int minimax_latent_length(int frames);
constexpr int eod = 151643, abc_start = 151847, abc_end = 151848;
constexpr int music_start = 151851, music_end = 151852;
constexpr int codec_offset = 151853, codec_size = 32768, context = 24576;
std::string yue_instruction(const std::string& planning);
}
}
