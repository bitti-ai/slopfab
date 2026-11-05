#include "../src/demucs/internal.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/tensor_convert.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace slopfab;
using namespace slopfab::demucs;
void require(bool ok, const char* text) {
  if (!ok) throw std::runtime_error(text);
}
int main(int argc, char** argv) {
  try {
    if (argc == 6 && std::string(argv[1]) == "probe") {
      Model model(argv[2]);
      SafeTensors input;
      input.open(argv[3]);
      const auto& pcm = input.at("pcm");
      Tensor wave(2, 1, int(pcm.shape.at(1)));
      wave.v = to_f32(pcm);
      auto spec = spectrogram(wave);
      auto inverse = inverse_spectrogram(spec, wave.t);
      auto out = model.forward(wave, [](const std::string&) {});
      Options options;
      options.model = argv[2];
      options.segment = std::stod(argv[5]);
      auto mixed = to_f32(input.at("mixed"));
      auto separated = separate(mixed, options);
      std::vector<float> stems;
      for (const auto& stem : separated.stems) stems.insert(stems.end(), stem.begin(), stem.end());
      write_safetensors(argv[4], {{"spec", {spec.c, spec.f, spec.t}, spec.v},
                                  {"inverse", {inverse.c, inverse.t}, inverse.v},
                                  {"output", {4, 2, wave.t}, out.v},
                                  {"separated", {4, int64_t(mixed.size() / 2), 2}, stems}});
      return 0;
    }
    for (int length : {1, 15, 2048, 4097, 8192}) {
      Tensor wave(2, 1, length);
      for (int c = 0; c < 2; ++c)
        for (int t = 0; t < length; ++t)
          wave.at(c, 0, t) = float(std::sin(t * 0.03 + c) * 0.4);
      auto spec = spectrogram(wave);
      require(spec.c == 4 && spec.f == 2048 && spec.t == (length + 1023) / 1024,
              "spectrogram shape");
      auto inverse = inverse_spectrogram(spec, length);
      for (float v : inverse.v) require(std::isfinite(v), "inverse finite");
      if (length == 8192) {
        // Dropped Nyquist and zero exterior STFT frames affect the edges.
        for (int t = 2048; t < length - 2048; ++t)
          require(std::abs(wave.at(0, 0, t) - inverse.at(0, 0, t)) < 2e-4,
                  "STFT interior round trip");
      }
    }
    Options options;
    options.model = "missing.safetensors";
    validate_options(options);
    for (double segment : {0., -1., 8., double(INFINITY), double(NAN)}) {
      options.segment = segment;
      bool rejected = false;
      try { validate_options(options); } catch (const std::invalid_argument&) { rejected = true; }
      require(rejected, "invalid segment accepted");
    }
    options.segment = 7.8;
    for (double overlap : {-0.1, 1., double(INFINITY), double(NAN)}) {
      options.overlap = overlap;
      bool rejected = false;
      try { validate_options(options); } catch (const std::invalid_argument&) { rejected = true; }
      require(rejected, "invalid overlap accepted");
    }
    options.overlap = 0.25;
    for (const std::vector<float>& input : {std::vector<float>{}, {1}, {NAN, 0}}) {
      bool rejected = false;
      try { separate(input, options); } catch (const std::invalid_argument&) { rejected = true; }
      require(rejected, "invalid audio accepted");
    }
    std::cout << "Demucs spectral and validation checks passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
