#pragma once

#include <functional>
#include <string>
#include <vector>

namespace slopfab::demucs {
struct Options {
  std::string model;
  // HTDemucs was trained on 7.8-second segments. Shorter segments trade quality
  // for memory/latency. No random shifts: repeated runs are deterministic.
  double segment = 7.8;
  double overlap = 0.25;
  std::function<bool(double, const std::string&)> progress;
};

struct Result {
  int sample_rate = 44100;
  int channels = 2;
  std::vector<std::string> sources;
  // Each stem is interleaved stereo, with exactly the input frame count.
  std::vector<std::vector<float>> stems;
};

// Native FP32 CPU HTDemucs inference. Input must be finite, interleaved stereo
// at 44100 Hz. Accepts upstream state_dict tensor names in F32/F16/BF16.
// Four-stem HTDemucs only; incompatible architectures fail before inference.
Result separate(const std::vector<float>& stereo, const Options& options);
void validate_options(const Options& options);
} // namespace slopfab::demucs
