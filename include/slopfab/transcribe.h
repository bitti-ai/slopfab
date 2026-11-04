#pragma once
#include <functional>
#include <string>
#include <vector>

namespace slopfab::whisper {
struct Options {
  std::string model;            // Hugging Face model.safetensors, without sidecar files.
  std::string variant = "auto"; // Explicit large-v1/v2 when shapes are ambiguous.
  std::string backend = "auto";
  std::string language = "auto";
  std::string prompt;
  bool word_timestamps = false;
  bool condition_on_previous_text = true;
  int beam_size = 1;
  // Called between layers/tokens/windows. Return false to cancel.
  std::function<bool(double, const std::string&)> progress;
};

struct Word {
  double start = 0, end = 0;
  std::string text;
};

struct Segment {
  double start = 0, end = 0;
  std::string text;
  double average_logprob = 0, no_speech_probability = 0;
  std::vector<Word> words;
};

struct Result {
  std::string model, backend, language, text;
  double duration = 0;
  std::vector<Segment> segments;
};

// Interleaved finite PCM at its native sample rate. The implementation owns
// downmixing/resampling. No Python, tokenizer files, or network access at runtime.
Result transcribe(const std::vector<float>& pcm, int sample_rate, int channels,
                  const Options& options);
std::string subtitles(const Result&, bool webvtt = false);
}
