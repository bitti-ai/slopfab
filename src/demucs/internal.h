#pragma once
#include "slopfab/separate.h"
#include "slopfab/safetensors.h"
#include <map>
#include <set>

namespace slopfab::demucs {
// All activations are channel-major [channels, frequencies, time]. The
// transformer uses [channels, 1, sequence], ordered time then frequency.
struct Tensor {
  int c, f, t;
  std::vector<float> v;
  Tensor(int channels, int frequencies, int time);
  float& at(int ch, int fr, int ti) { return v[(size_t(ch) * f + fr) * t + ti]; }
  float at(int ch, int fr, int ti) const { return v[(size_t(ch) * f + fr) * t + ti]; }
};
Tensor spectrogram(const Tensor& wave);
Tensor inverse_spectrogram(const Tensor& spec, int length);
std::pair<float, float> normalize(Tensor& x);

struct Weight {
  std::vector<int64_t> shape;
  std::vector<float> v;
};
class Model {
public:
  explicit Model(const std::string& path);
  Tensor forward(const Tensor& wave, const std::function<void(const std::string&)>& progress);
private:
  std::map<std::string, Weight> weights;
  int channels = 48, bottom = 512, layers = 5;
  const Weight& w(const std::string& name) const;
  void expect(const std::string& name, std::vector<int64_t> shape);
  std::set<std::string> checked;
  Tensor conv(const Tensor& x, const std::string& name, int sf=1, int st=1,
              int pf=0, int pt=0, int dilation=1, bool transpose=false,
              int output_time=0) const;
  Tensor norm(Tensor x, const std::string& name, bool per_token=false,
              bool per_frequency=false) const;
  Tensor dconv(Tensor x, const std::string& name) const;
  Tensor encode(Tensor x, const std::string& name, bool frequency) const;
  Tensor decode(Tensor x, const Tensor& skip, const std::string& name,
                bool frequency, int length, bool last) const;
  Tensor linear(const Tensor& x, const std::string& name, int part=-1) const;
  Tensor attention(const Tensor& q, const Tensor& kv, const std::string& name) const;
  Tensor transformer(Tensor x, const Tensor& other, const std::string& name, bool cross) const;
};
} // namespace slopfab::demucs
