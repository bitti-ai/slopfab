#pragma once
#include "slopfab/transcribe.h"
#include "slopfab/safetensors.h"
#include "slopfab/json.h"
#include <map>
#include <memory>
#include <unordered_map>

namespace slopfab::whisper {
std::string embedded_asset(const std::string& name);
const json::Value& assets();
struct Config {
  std::string name;
  int width = 0, heads = 0, encoder_layers = 0, decoder_layers = 0, mels = 0, vocab = 0;
  int audio_context = 1500, text_context = 448;
  bool multilingual = true;
  std::vector<std::pair<int,int>> alignment_heads;
};
Config inspect(const SafeTensors&, const std::string& variant);
struct Tokenizer {
  explicit Tokenizer(const Config&);
  std::vector<int> encode(const std::string&) const;
  std::string decode(const std::vector<int>&) const;
  int language_id(const std::string&) const;
  std::string language_code(int) const;
  int eot, sot, translate, transcribe, previous, no_speech, no_timestamps, timestamp_begin;
  int language_count;
  std::vector<int> suppress;
  std::vector<std::string> vocabulary;
  std::unordered_map<std::string,int> ranks;
};
std::vector<float> mono_16k(const std::vector<float>&, int rate, int channels);
// Entire recording + 30 seconds of zero padding, in [frames,mels] layout.
std::vector<float> log_mel(const std::vector<float>&, int mels);
struct Tensor {
  int rows = 0, cols = 0;
  std::shared_ptr<void> data;
};
class Backend {
public:
  virtual ~Backend() = default;
  virtual Tensor upload(const std::vector<float>&, int rows, int cols) = 0;
  virtual std::vector<float> download(const Tensor&) = 0;
  virtual Tensor linear(const Tensor&, const Tensor&, const Tensor* bias = nullptr) = 0;
  virtual Tensor norm(const Tensor&, const Tensor& weight, const Tensor& bias) = 0;
  virtual Tensor gelu(const Tensor&) = 0;
  virtual Tensor add(const Tensor&, const Tensor&) = 0;
  virtual Tensor rows(const Tensor&, int first, int count) = 0;
  virtual Tensor concat(const Tensor&, const Tensor&) = 0;
  virtual Tensor im2col(const Tensor&, int stride) = 0;
  // Optional probabilities: [heads,queries,keys], before any rounding.
  virtual Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v,
                           int heads, bool causal, std::vector<float>* probabilities = nullptr) = 0;
};
std::unique_ptr<Backend> cpu_backend();
std::unique_ptr<Backend> cuda_backend();
std::unique_ptr<Backend> vulkan_backend();
struct Cache { Tensor k, v; };
class Model {
public:
  Model(const Options&);
  SafeTensors checkpoint;
  Config config;
  std::string backend_name;
  std::unique_ptr<Backend> backend;
  std::map<std::string,Tensor> weights;
  std::vector<Cache> self_cache, cross_cache;
  Tensor audio;
  std::function<void(const std::string&)> check;
  const Tensor& weight(const std::string&);
  Tensor projection(const Tensor&, const std::string&);
  Tensor normalization(const Tensor&, const std::string&);
  void encode(const std::vector<float>& mel);
  void reset();
  std::vector<float> decode(const std::vector<int>& ids,
                            std::vector<std::vector<float>>* alignment = nullptr);
};
std::vector<Word> align_words(Model&, const Tokenizer&, const std::vector<int>& prefix,
                              const std::vector<int>& text, double offset, double duration);
}
