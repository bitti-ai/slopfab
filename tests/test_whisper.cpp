#include "../src/whisper/internal.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/safetensors_write.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace slopfab;
using namespace slopfab::whisper;

void require(bool ok, const char* text) {
  if (!ok)
    throw std::runtime_error(text);
}

void close(const std::vector<float>& a, const std::vector<float>& b, float tolerance = 0.0002f) {
  require(a.size() == b.size(), "shape mismatch");
  for (size_t i = 0; i < a.size(); ++i)
    if (!std::isfinite(b[i]) || std::abs(a[i] - b[i]) > tolerance * (1 + std::abs(a[i])))
      throw std::runtime_error("numeric mismatch at " + std::to_string(i) + ": " +
                               std::to_string(a[i]) + " vs " + std::to_string(b[i]));
}

int main(int argc, char** argv) {
  try {
    if (argc == 6 && std::string(argv[1]) == "probe") {
      Options o;
      o.backend = argv[2];
      o.model = argv[3];
      Model m(o);
      SafeTensors input;
      input.open(argv[4]);
      auto pcm = to_f32(input.at("pcm"));
      auto mel = log_mel(pcm, m.config.mels);
      auto window = std::vector<float>(mel.begin(), mel.begin() + 3000 * m.config.mels);
      m.encode(window);
      auto encoder = m.backend->download(m.audio);
      std::vector<int> tokens;
      for (float v : to_f32(input.at("tokens")))
        tokens.push_back(int(v));
      auto logits = m.decode(tokens);
      auto next = m.decode({123});
      m.reset();
      tokens.push_back(123);
      auto full = m.decode(tokens);
      close(full, next, 0.002f);
      write_safetensors(argv[5],
                        {{"mel", {int64_t(mel.size() / m.config.mels), m.config.mels}, mel},
                         {"encoder", {1500, m.config.width}, encoder},
                         {"logits", {m.config.vocab}, logits},
                         {"cached_logits", {m.config.vocab}, next}});
      return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "tokenize") {
      Config c;
      c.multilingual = true;
      c.vocab = 51866;
      Tokenizer t(c);
      for (auto id : t.encode(argv[2]))
        std::cout << id << " ";
      std::cout << "\n";
      return 0;
    }
    auto reference = cpu_backend();
    std::unique_ptr<Backend> backend;
    std::string name = argc > 1 ? argv[1] : "cpu";
    if (name == "cpu")
      backend = cpu_backend();
#if SLOPFAB_WITH_CUDA
    if (name == "cuda")
      backend = cuda_backend();
#endif
#if SLOPFAB_WITH_VULKAN
    if (name == "vulkan")
      backend = vulkan_backend();
#endif
    require(bool(backend), "backend unavailable");
    std::vector<float> x(7 * 64), w(80 * 64), bias(80), scale(64), shift(64);
    for (size_t i = 0; i < x.size(); ++i)
      x[i] = std::sin(float(i) * 0.73f);
    for (size_t i = 0; i < w.size(); ++i)
      w[i] = std::cos(float(i) * 0.13f) * 0.1f;
    for (int i = 0; i < 80; ++i)
      bias[i] = float(i) * .01f;
    for (int i = 0; i < 64; ++i) {
      scale[i] = .5f + i * .01f;
      shift[i] = -.1f + i * .002f;
    }
    auto a = reference->upload(x, 7, 64), b = backend->upload(x, 7, 64);
    auto wa = reference->upload(w, 80, 64), wb = backend->upload(w, 80, 64);
    auto ba = reference->upload(bias, 1, 80), bb = backend->upload(bias, 1, 80);
    close(reference->download(reference->linear(a, wa, &ba)),
          backend->download(backend->linear(b, wb, &bb)));
    auto sa = reference->upload(scale, 1, 64), sb = backend->upload(scale, 1, 64),
         ta = reference->upload(shift, 1, 64), tb = backend->upload(shift, 1, 64);
    close(reference->download(reference->norm(a, sa, ta)),
          backend->download(backend->norm(b, sb, tb)));
    close(reference->download(reference->gelu(a)), backend->download(backend->gelu(b)));
    for (int stride : {1, 2})
      close(reference->download(reference->im2col(a, stride)),
            backend->download(backend->im2col(b, stride)), 0);
    for (bool causal : {false, true}) {
      std::vector<float> pa, pb;
      close(reference->download(reference->attention(a, a, a, 4, causal, &pa)),
            backend->download(backend->attention(b, b, b, 4, causal, &pb)));
      close(pa, pb);
    }
    auto ar = reference->rows(a, 2, 1), br = backend->rows(b, 2, 1);
    close(reference->download(reference->attention(ar, a, a, 4, false)),
          backend->download(backend->attention(br, b, b, 4, false)));
    close(reference->download(reference->concat(a, ar)), backend->download(backend->concat(b, br)),
          0);
    Config c;
    c.multilingual = true;
    c.vocab = 51866;
    Tokenizer tok(c);
    for (const std::string s :
         {"Hello world!", "I'm singing 12345.\n\n A song.", "Hyvää päivää! 日本語 中文 العربية"})
      require(tok.decode(tok.encode(s)) == s, "tokenizer roundtrip");
    require(tok.encode("Hello world!") == std::vector<int>({15947, 1002, 0}),
            "known tokenizer IDs");
    require(tok.language_id("fi") == 50277 && tok.timestamp_begin == 50365, "v3 special IDs");
    auto silence = log_mel(std::vector<float>(400, 0), 80);
    for (float v : silence)
      require(v == -1.5f, "silence mel");
    auto mono = mono_16k({1, -1, .5f, .5f}, 16000, 2);
    require(mono == std::vector<float>({0, .5f}), "downmix");
    Result r;
    r.segments.push_back({1.234, 2.567, "Hello"});
    require(subtitles(r).find("00:00:01,234 --> 00:00:02,567") != std::string::npos, "SRT timing");
    require(subtitles(r, true).find("WEBVTT\n\n00:00:01.234") != std::string::npos, "VTT timing");
    std::cout << "PASS: Whisper " << name
              << " primitives, tokenizer, features and subtitle timing\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
