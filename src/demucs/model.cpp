#include "internal.h"
#include "parallel.h"
#include "slopfab/tensor_convert.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace slopfab::demucs {
namespace {
void gelu(Tensor& x) {
  for (auto& v : x.v)
    v = 0.5f * v * (1 + std::erf(v * 0.7071067811865475f));
}

Tensor glu(const Tensor& x) {
  Tensor y(x.c / 2, x.f, x.t);
  for (size_t i = 0; i < y.v.size(); ++i)
    y.v[i] = x.v[i] / (1 + std::exp(-x.v[i + y.v.size()]));
  return y;
}

void add(Tensor& x, const Tensor& y, const std::vector<float>* scale = nullptr) {
  if (x.c != y.c || x.f != y.f || x.t != y.t)
    throw std::runtime_error("Demucs: residual shape mismatch");
  for (int c = 0; c < x.c; ++c) {
    float s = scale ? (*scale)[c] : 1;
    for (int i = 0; i < x.f * x.t; ++i)
      x.v[size_t(c) * x.f * x.t + i] += s * y.v[size_t(c) * x.f * x.t + i];
  }
}

Tensor tokens(const Tensor& x) {
  Tensor y(x.c, 1, x.f * x.t);
  for (int c = 0; c < x.c; ++c)
    for (int f = 0; f < x.f; ++f)
      for (int t = 0; t < x.t; ++t)
        y.at(c, 0, t * x.f + f) = x.at(c, f, t);
  return y;
}

Tensor frequencies(const Tensor& x, int f) {
  Tensor y(x.c, f, x.t / f);
  for (int c = 0; c < x.c; ++c)
    for (int fr = 0; fr < f; ++fr)
      for (int t = 0; t < y.t; ++t)
        y.at(c, fr, t) = x.at(c, 0, t * f + fr);
  return y;
}
} // namespace

const Weight& Model::w(const std::string& name) const {
  auto it = weights.find(name);
  if (it == weights.end())
    throw std::invalid_argument("Demucs: missing tensor " + name);
  return it->second;
}

void Model::expect(const std::string& name, std::vector<int64_t> shape) {
  if (w(name).shape != shape)
    throw std::invalid_argument("Demucs: incompatible shape for " + name);
  checked.insert(name);
}

Model::Model(const std::string& path) {
  SafeTensors file;
  file.open(path);
  if (file.file_size() > (512ull << 20))
    throw std::invalid_argument("Demucs: checkpoint exceeds 512 MiB");
  for (const auto& expected :
       std::map<std::string, std::string>{{"architecture", "htdemucs"},
                                          {"sources", "drums,bass,other,vocals"},
                                          {"samplerate", "44100"}}) {
    auto it = file.metadata().find(expected.first);
    if (it != file.metadata().end() && it->second != expected.second)
      throw std::invalid_argument("Demucs: incompatible metadata " + expected.first);
  }
  // All non-tensor architecture choices follow the released 955717e8 HTDemucs
  // configuration. Reject unfamiliar tensors rather than silently ignoring them.
  size_t expanded_bytes = 0;
  for (const auto& entry : file.tensors()) {
    const auto& t = entry.second;
    if (t.dtype != DType::kF32 && t.dtype != DType::kF16 && t.dtype != DType::kBF16)
      throw std::invalid_argument("Demucs: expected F32/F16/BF16 tensor " + entry.first);
    if (t.nbytes > (256ull << 20))
      throw std::invalid_argument("Demucs: excessive tensor size");
    expanded_bytes += size_t(t.numel()) * sizeof(float);
    if (expanded_bytes > (512ull << 20))
      throw std::invalid_argument("Demucs: expanded checkpoint exceeds 512 MiB");
    auto data = to_f32(t);
    for (float value : data)
      if (!std::isfinite(value))
        throw std::invalid_argument("Demucs: non-finite tensor " + entry.first);
    weights.emplace(entry.first, Weight{t.shape, std::move(data)});
  }
  const auto& first = w("encoder.0.conv.weight").shape;
  if (first.size() != 4 || first[0] < 8 || first[0] > 128 || first[0] % 8)
    throw std::invalid_argument("Demucs: expected standard HTDemucs weights");
  channels = int(first[0]);
  const auto& up = w("channel_upsampler.weight").shape;
  if (up.size() != 3 || up[0] < 8 || up[0] > 1024 || up[0] % 8)
    throw std::invalid_argument("Demucs: invalid transformer width");
  bottom = int(up[0]);
  layers = 0;
  while (weights.count("crosstransformer.layers." + std::to_string(layers) + ".linear1.weight")) {
    if (++layers > 12)
      throw std::invalid_argument("Demucs: too many transformer layers");
  }
  if (!layers)
    throw std::invalid_argument("Demucs: missing transformer");
  auto norm_weights = [&](const std::string& p, int c) {
    expect(p + ".weight", {c});
    expect(p + ".bias", {c});
  };
  auto conv_weights = [&](const std::string& p, int out, int in, int kf, int kt, bool freq) {
    expect(p + ".weight",
           freq ? std::vector<int64_t>{out, in, kf, kt} : std::vector<int64_t>{out, in, kt});
    expect(p + ".bias", {out});
  };
  for (bool freq : {true, false}) {
    for (int i = 0; i < 4; ++i) {
      int c = channels << i, in = i ? c / 2 : (freq ? 4 : 2);
      std::string enc = (freq ? "encoder." : "tencoder.") + std::to_string(i);
      conv_weights(enc + ".conv", c, in, freq ? 8 : 1, freq ? 1 : 8, freq);
      conv_weights(enc + ".rewrite", 2 * c, c, 1, 1, freq);
      std::string dec = (freq ? "decoder." : "tdecoder.") + std::to_string(3 - i);
      int out = i ? c / 2 : (freq ? 16 : 8);
      // Transpose convolutions store [in, out, kernel...].
      expect(dec + ".conv_tr.weight",
             freq ? std::vector<int64_t>{c, out, 8, 1} : std::vector<int64_t>{c, out, 8});
      expect(dec + ".conv_tr.bias", {out});
      conv_weights(dec + ".rewrite", 2 * c, c, freq ? 3 : 1, 3, freq);
      for (const auto& prefix : {enc, dec})
        for (int d = 0; d < 2; ++d) {
          auto p = prefix + ".dconv.layers." + std::to_string(d);
          conv_weights(p + ".0", c / 8, c, 1, 3, false);
          norm_weights(p + ".1", c / 8);
          conv_weights(p + ".3", 2 * c, c / 8, 1, 1, false);
          norm_weights(p + ".4", 2 * c);
          expect(p + ".6.scale", {c});
        }
    }
  }
  expect("freq_emb.embedding.weight", {512, channels});
  for (const std::string suffix : {"", "_t"}) {
    conv_weights("channel_upsampler" + suffix, bottom, channels * 8, 1, 1, false);
    conv_weights("channel_downsampler" + suffix, channels * 8, bottom, 1, 1, false);
    norm_weights("crosstransformer.norm_in" + suffix, bottom);
    for (int i = 0; i < layers; ++i) {
      auto p = "crosstransformer.layers" + suffix + "." + std::to_string(i);
      auto a = p + (i % 2 ? ".cross_attn" : ".self_attn");
      expect(a + ".in_proj_weight", {bottom * 3, bottom});
      expect(a + ".in_proj_bias", {bottom * 3});
      expect(a + ".out_proj.weight", {bottom, bottom});
      expect(a + ".out_proj.bias", {bottom});
      expect(p + ".linear1.weight", {bottom * 4, bottom});
      expect(p + ".linear1.bias", {bottom * 4});
      expect(p + ".linear2.weight", {bottom, bottom * 4});
      expect(p + ".linear2.bias", {bottom});
      for (const auto& n : {".norm1", ".norm2", ".norm_out"})
        norm_weights(p + n, bottom);
      if (i % 2)
        norm_weights(p + ".norm3", bottom);
      expect(p + ".gamma_1.scale", {bottom});
      expect(p + ".gamma_2.scale", {bottom});
    }
  }
  for (const auto& item : weights)
    if (!checked.count(item.first))
      throw std::invalid_argument("Demucs: unsupported tensor " + item.first);
}

Tensor Model::conv(const Tensor& x, const std::string& name, int sf, int st, int pf, int pt,
                   int dilation, bool transpose, int output_time) const {
  const auto& weight = w(name + ".weight");
  const auto& bias = w(name + ".bias").v;
  bool freq = weight.shape.size() == 4;
  int kf = freq ? int(weight.shape[2]) : 1;
  int kt = int(weight.shape.back());
  int oc = int(bias.size());
  int of = transpose ? (x.f - 1) * sf + kf - 2 * pf : (x.f + 2 * pf - kf) / sf + 1;
  int ot =
      transpose ? (x.t - 1) * st + kt - 2 * pt : (x.t + 2 * pt - dilation * (kt - 1) - 1) / st + 1;
  if (output_time)
    ot = output_time;
  Tensor y(oc, of, ot);
  // Vectorize over contiguous time samples; each worker owns an output plane.
  parallel_for(oc * of, [&](int plane) {
    int o = plane / of, f = plane % of;
    auto* dst = y.v.data() + size_t(plane) * ot;
    std::fill_n(dst, ot, bias[o]);
    for (int c = 0; c < x.c; ++c)
      for (int a = 0; a < kf; ++a) {
        int fr = transpose ? f + pf - a : f * sf + a - pf;
        if (transpose && (fr < 0 || fr % sf))
          continue;
        if (transpose)
          fr /= sf;
        if (fr < 0 || fr >= x.f)
          continue;
        const float* src = x.v.data() + (size_t(c) * x.f + fr) * x.t;
        for (int b = 0; b < kt; ++b) {
          size_t wi = transpose ? ((size_t(c) * oc + o) * kf + a) * kt + b
                                : ((size_t(o) * x.c + c) * kf + a) * kt + b;
          float value = weight.v[wi];
          if (transpose) {
            int first = std::max(0, (pt - b + st - 1) / st);
            int last = std::min(x.t, (ot - 1 + pt - b) / st + 1);
            for (int t = first; t < last; ++t)
              dst[t * st + b - pt] += value * src[t];
          } else {
            int shift = b * dilation - pt;
            int first = std::max(0, (-shift + st - 1) / st);
            int last = std::min(ot, (x.t - 1 - shift) / st + 1);
            for (int t = first; t < last; ++t)
              dst[t] += value * src[t * st + shift];
          }
        }
      }
  });
  return y;
}

Tensor Model::norm(Tensor x, const std::string& name, bool per_token, bool per_frequency) const {
  const auto& scale = w(name + ".weight").v;
  const auto& bias = w(name + ".bias").v;
  int groups = per_token ? x.t : (per_frequency ? x.f : 1);
  parallel_for(groups, [&](int group) {
    double mean = 0, variance = 0;
    int count = per_token ? x.c : (per_frequency ? x.c * x.t : int(x.v.size()));
    auto index = [&](int j) -> size_t {
      if (per_token)
        return size_t(j) * x.t + group;
      if (per_frequency)
        return (size_t(j / x.t) * x.f + group) * x.t + j % x.t;
      return size_t(j);
    };
    for (int j = 0; j < count; ++j)
      mean += x.v[index(j)];
    mean /= count;
    for (int j = 0; j < count; ++j) {
      double d = x.v[index(j)] - mean;
      variance += d * d;
    }
    float inv = float(1 / std::sqrt(variance / count + 1e-5));
    for (int j = 0; j < count; ++j) {
      auto at = index(j);
      int c = int(at / (x.f * x.t));
      x.v[at] = (x.v[at] - float(mean)) * inv * scale[c] + bias[c];
    }
  });
  return x;
}

Tensor Model::dconv(Tensor x, const std::string& name) const {
  for (int d = 0; d < 2; ++d) {
    auto p = name + ".dconv.layers." + std::to_string(d);
    auto y = conv(x, p + ".0", 1, 1, 0, 1 << d, 1 << d);
    y = norm(std::move(y), p + ".1", false, true);
    gelu(y);
    y = conv(y, p + ".3");
    y = glu(norm(std::move(y), p + ".4", false, true));
    add(x, y, &w(p + ".6.scale").v);
  }
  return x;
}

Tensor Model::encode(Tensor x, const std::string& name, bool frequency) const {
  int length = (x.t + 3) / 4;
  x = conv(x, name + ".conv", frequency ? 4 : 1, frequency ? 1 : 4, frequency ? 2 : 0,
           frequency ? 0 : 2, 1, false, frequency ? 0 : length);
  gelu(x);
  x = dconv(std::move(x), name);
  return glu(conv(x, name + ".rewrite"));
}

Tensor Model::decode(Tensor x, const Tensor& skip, const std::string& name, bool frequency,
                     int length, bool last) const {
  add(x, skip);
  x = glu(conv(x, name + ".rewrite", 1, 1, frequency ? 1 : 0, 1));
  x = dconv(std::move(x), name);
  x = conv(x, name + ".conv_tr", frequency ? 4 : 1, frequency ? 1 : 4, frequency ? 2 : 0,
           frequency ? 0 : 2, 1, true, frequency ? 0 : length);
  if (!last)
    gelu(x);
  return x;
}

Tensor Model::linear(const Tensor& x, const std::string& name, int part) const {
  const auto& weight = w(name + (part < 0 ? ".weight" : "_weight"));
  const auto& bias = w(name + (part < 0 ? ".bias" : "_bias")).v;
  int out = int(weight.shape[0]) / (part < 0 ? 1 : 3);
  int offset = part < 0 ? 0 : part * out;
  Tensor y(out, 1, x.t);
  parallel_for(out, [&](int o) {
    float* dst = y.v.data() + size_t(o) * x.t;
    std::fill_n(dst, x.t, bias[offset + o]);
    for (int c = 0; c < x.c; ++c) {
      float value = weight.v[size_t(offset + o) * x.c + c];
      const float* src = x.v.data() + size_t(c) * x.t;
      for (int t = 0; t < x.t; ++t)
        dst[t] += value * src[t];
    }
  });
  return y;
}

Tensor Model::attention(const Tensor& query, const Tensor& kv, const std::string& name) const {
  auto q = linear(query, name + ".in_proj", 0);
  auto k = linear(kv, name + ".in_proj", 1);
  auto v = linear(kv, name + ".in_proj", 2);
  Tensor out(q.c, 1, q.t);
  const int heads = 8, dim = q.c / heads;
  // Only one row of scores per worker; never allocate a quadratic score matrix.
  parallel_for(heads * q.t, [&](int row) {
    int h = row / q.t, t = row % q.t;
    std::vector<float> scores(k.t);
    for (int d = 0; d < dim; ++d) {
      float a = q.at(h * dim + d, 0, t) / std::sqrt(float(dim));
      const float* b = k.v.data() + size_t(h * dim + d) * k.t;
      for (int j = 0; j < k.t; ++j)
        scores[j] += a * b[j];
    }
    float maximum = *std::max_element(scores.begin(), scores.end()), sum = 0;
    for (auto& s : scores) {
      s = std::exp(s - maximum);
      sum += s;
    }
    for (auto& s : scores)
      s /= sum;
    for (int d = 0; d < dim; ++d) {
      float value = 0;
      const float* b = v.v.data() + size_t(h * dim + d) * k.t;
      for (int j = 0; j < k.t; ++j)
        value += scores[j] * b[j];
      out.at(h * dim + d, 0, t) = value;
    }
  });
  return linear(out, name + ".out_proj");
}

Tensor Model::transformer(Tensor x, const Tensor& other, const std::string& name,
                          bool cross) const {
  auto query = norm(x, name + ".norm1", true);
  auto kv = cross ? norm(other, name + ".norm2", true) : query;
  auto y = attention(query, kv, name + (cross ? ".cross_attn" : ".self_attn"));
  add(x, y, &w(name + ".gamma_1.scale").v);
  y = linear(norm(x, name + (cross ? ".norm3" : ".norm2"), true), name + ".linear1");
  gelu(y);
  y = linear(y, name + ".linear2");
  add(x, y, &w(name + ".gamma_2.scale").v);
  return norm(std::move(x), name + ".norm_out");
}

Tensor Model::forward(const Tensor& wave, const std::function<void(const std::string&)>& progress) {
  auto x = spectrogram(wave), xt = wave;
  auto stats = normalize(x), statst = normalize(xt);
  std::vector<Tensor> saved, saved_t;
  std::vector<int> lengths;
  for (int i = 0; i < 4; ++i) {
    progress("encoder " + std::to_string(i + 1));
    lengths.push_back(xt.t);
    xt = encode(std::move(xt), "tencoder." + std::to_string(i), false);
    x = encode(std::move(x), "encoder." + std::to_string(i), true);
    if (!i) {
      const auto& embedding = w("freq_emb.embedding.weight").v;
      for (int c = 0; c < x.c; ++c)
        for (int f = 0; f < x.f; ++f)
          for (int t = 0; t < x.t; ++t)
            x.at(c, f, t) += 2 * embedding[size_t(f) * x.c + c];
    }
    saved.push_back(x);
    saved_t.push_back(xt);
  }
  int frequencies_count = x.f;
  // 1x1 channel projections commute with this permutation.
  x = conv(tokens(x), "channel_upsampler");
  xt = conv(xt, "channel_upsampler_t");
  x = norm(std::move(x), "crosstransformer.norm_in", true);
  xt = norm(std::move(xt), "crosstransformer.norm_in_t", true);
  for (int c = 0; c < bottom; ++c) {
    int half = bottom / 2;
    for (int t = 0; t < x.t; ++t) {
      int position = c < half ? t / frequencies_count : t % frequencies_count;
      double phase = position * std::pow(10000., -double((c % half) / 2 * 2) / half);
      x.at(c, 0, t) += float(c % 2 ? std::cos(phase) : std::sin(phase));
    }
    for (int t = 0; t < xt.t; ++t) {
      double phase = t / std::pow(10000., double(c % half) / (half - 1));
      xt.at(c, 0, t) += float(c < half ? std::cos(phase) : std::sin(phase));
    }
  }
  for (int i = 0; i < layers; ++i) {
    progress("transformer " + std::to_string(i + 1));
    auto old_x = x;
    x = transformer(std::move(x), xt, "crosstransformer.layers." + std::to_string(i), i % 2 != 0);
    xt = transformer(std::move(xt), old_x, "crosstransformer.layers_t." + std::to_string(i),
                     i % 2 != 0);
  }
  x = frequencies(conv(x, "channel_downsampler"), frequencies_count);
  xt = conv(xt, "channel_downsampler_t");
  for (int i = 0; i < 4; ++i) {
    progress("decoder " + std::to_string(i + 1));
    x = decode(std::move(x), saved.back(), "decoder." + std::to_string(i), true, 0, i == 3);
    xt = decode(std::move(xt), saved_t.back(), "tdecoder." + std::to_string(i), false,
                lengths.back(), i == 3);
    saved.pop_back();
    saved_t.pop_back();
    lengths.pop_back();
  }
  for (auto& value : x.v)
    value = value * stats.second + stats.first;
  for (auto& value : xt.v)
    value = value * statst.second + statst.first;
  progress("inverse STFT");
  x = inverse_spectrogram(x, wave.t);
  add(x, xt);
  return x;
}
} // namespace slopfab::demucs
