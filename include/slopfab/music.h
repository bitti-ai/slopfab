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
  int top_k = 100;
  float repetition_penalty = 1.2f;
  float semantic_guidance = 0.f; // zero selects the model's recipe
  int abc_max_tokens = 4096;
};

MusicSettings parse_music_settings(const std::string& json);
void validate_music_settings(const MusicSettings& settings);
GeneratePlan resolve_music_plan(const GenerateRequest& request);

namespace music {
void load_yue_tokenizer(text::Tokenizer&, const SafeTensors&);
constexpr int eod = 151643, abc_start = 151847, abc_end = 151848;
constexpr int music_start = 151851, music_end = 151852;
constexpr int codec_offset = 151853, codec_size = 32768, context = 24576;
std::string yue_instruction(const std::string& planning);
}
}
