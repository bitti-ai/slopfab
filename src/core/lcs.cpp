#include "slopfab/lcs.h"
#include "slopfab/json.h"
#include "slopfab/safetensors.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/sha256.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace slopfab {
namespace {
constexpr int D = LcsData::dim;
constexpr float pi = 3.14159265358979323846f;
using V = std::array<float, 3>;

float dot(V a, V b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

V add(V a, V b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

V sub(V a, V b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

V mul(V a, float b) {
  return {a[0] * b, a[1] * b, a[2] * b};
}

float norm(V a) {
  return std::sqrt(dot(a, a));
}

float unit(float x) {
  return std::clamp(x, 0.f, 1.f);
}

float wrap(float x) {
  return x - std::floor(x);
}

float hue_delta(float x) {
  return x - std::floor(x + .5f);
}

void require(bool ok, const char* message) {
  if (!ok)
    throw std::invalid_argument(std::string("LCS: ") + message);
}

void finite(const std::vector<float>& v, size_t n) {
  require(v.size() == n, "invalid calibration tensor shape");
  for (float x : v)
    require(std::isfinite(x), "non-finite calibration");
}

V project(const LcsData& d, const float* raw) {
  V p{};
  for (int i = 0; i < D; ++i)
    for (int k = 0; k < 3; ++k)
      p[k] += (raw[i] - d.mean[i]) * d.basis[i * 3 + k];
  return p;
}

// Symmetric Jacobi PCA in double precision; calibration is tiny (96 columns).
std::vector<double> eigen(std::vector<double> a, std::vector<double>& values) {
  std::vector<double> v(D * D, 0);
  for (int i = 0; i < D; ++i)
    v[i * D + i] = 1;
  for (int sweep = 0; sweep < 80; ++sweep) {
    double largest = 0, scale = 0;
    for (int i = 0; i < D; ++i)
      scale = std::max(scale, std::abs(a[i * D + i]));
    for (int p = 0; p < D; ++p)
      for (int q = p + 1; q < D; ++q) {
        const double apq = a[p * D + q];
        largest = std::max(largest, std::abs(apq));
        if (std::abs(apq) <= std::max(1e-20, scale * 1e-13))
          continue;
        const double tau = (a[q * D + q] - a[p * D + p]) / (2 * apq);
        const double t = std::copysign(1., tau) / (std::abs(tau) + std::hypot(1., tau));
        const double c = 1 / std::sqrt(1 + t * t), s = t * c;
        a[p * D + p] -= t * apq;
        a[q * D + q] += t * apq;
        a[p * D + q] = a[q * D + p] = 0;
        for (int k = 0; k < D; ++k) {
          if (k != p && k != q) {
            const double x = a[k * D + p], y = a[k * D + q];
            a[k * D + p] = a[p * D + k] = c * x - s * y;
            a[k * D + q] = a[q * D + k] = s * x + c * y;
          }
          const double x = v[k * D + p], y = v[k * D + q];
          v[k * D + p] = c * x - s * y;
          v[k * D + q] = s * x + c * y;
        }
      }
    if (largest <= std::max(1e-20, scale * 1e-12))
      break;
  }
  values.resize(D);
  for (int i = 0; i < D; ++i)
    values[i] = a[i * D + i];
  return v;
}

std::vector<double> covariance(const std::vector<float>& samples, const std::vector<float>& mean) {
  std::vector<double> cov(D * D, 0);
  for (size_t r = 0; r < samples.size(); r += D)
    for (int i = 0; i < D; ++i)
      for (int j = 0; j <= i; ++j)
        cov[i * D + j] += double(samples[r + i] - mean[i]) * (samples[r + j] - mean[j]);
  for (int i = 0; i < D; ++i)
    for (int j = 0; j < i; ++j)
      cov[j * D + i] = cov[i * D + j];
  return cov;
}

// Scalar bicone color geometry, anchored to this VAE's encoded colors.
struct ColorSpace {
  V black, axis, e1, e2, warm;
  std::array<V, 6> chroma;
  std::array<float, 6> angles;
  std::array<int, 6> sorted{0, 1, 2, 3, 4, 5};
  const std::array<float, 6> hues{0, 4.f / 6, 2.f / 6, 5.f / 6, 3.f / 6, 1.f / 6};

  explicit ColorSpace(const LcsData& d) {
    auto anchor = [&](int i) -> V {
      return {d.anchors[i * 3], d.anchors[i * 3 + 1], d.anchors[i * 3 + 2]};
    };
    black = anchor(6);
    axis = sub(anchor(7), black);
    const V n = mul(axis, 1 / norm(axis));
    const V arb = std::abs(n[0]) > .9f ? V{0, 1, 0} : V{1, 0, 0};
    e1 = sub(arb, mul(n, dot(arb, n)));
    e1 = mul(e1, 1 / norm(e1));
    e2 = {n[1] * e1[2] - n[2] * e1[1], n[2] * e1[0] - n[0] * e1[2], n[0] * e1[1] - n[1] * e1[0]};
    float radius = 0;
    for (int i = 0; i < 6; ++i) {
      V v = sub(anchor(i), black);
      chroma[i] = sub(v, mul(n, dot(v, n)));
      angles[i] = wrap(std::atan2(dot(chroma[i], e2), dot(chroma[i], e1)) / (2 * pi)) * 2 * pi;
      radius += norm(chroma[i]) / 6;
    }
    std::sort(sorted.begin(), sorted.end(), [&](int a, int b) {
      return angles[a] < angles[b];
    });
    warm = sub(add(chroma[0], chroma[5]), add(chroma[1], chroma[4]));
    warm = mul(warm, radius / std::max(norm(warm), 1e-8f));
  }

  V boundary(float h) const {
    constexpr int order[]{0, 5, 2, 4, 1, 3};
    const float x = wrap(h) * 6;
    const int i = std::min(5, int(x));
    const float t = x - i;
    const int a = order[i], b = order[(i + 1) % 6];
    const float angle = angles[a] + t * hue_delta((angles[b] - angles[a]) / (2 * pi)) * 2 * pi;
    const float radius = (1 - t) * norm(chroma[a]) + t * norm(chroma[b]);
    return mul(add(mul(e1, std::cos(angle)), mul(e2, std::sin(angle))), radius);
  }

  V encode(V hsl) const {
    return add(add(black, mul(axis, hsl[2])),
               mul(boundary(hsl[0]), hsl[1] * (1 - std::abs(2 * hsl[2] - 1))));
  }

  V decode(V c) const {
    const float l = unit(dot(sub(c, black), axis) / dot(axis, axis));
    const V chromatic = sub(c, add(black, mul(axis, l)));
    float angle = wrap(std::atan2(dot(chromatic, e2), dot(chromatic, e1)) / (2 * pi)) * 2 * pi;
    float h = 0;
    for (int i = 0; i < 6; ++i) {
      const int a = sorted[i], b = sorted[(i + 1) % 6];
      float lo = angles[a], hi = angles[b];
      if (i == 5)
        hi += 2 * pi;
      float x = angle;
      if (x < lo)
        x += 2 * pi;
      if (x >= lo && x <= hi && hi - lo > 1e-8f) {
        h = wrap(hues[a] + (x - lo) / (hi - lo) * hue_delta(hues[b] - hues[a]));
        break;
      }
    }
    return {h,
            unit(norm(chromatic) / std::max(1e-8f, norm(boundary(h)) * (1 - std::abs(2 * l - 1)))),
            l};
  }
};

V rgb_hsl(V rgb) {
  const float hi = *std::max_element(rgb.begin(), rgb.end()),
              lo = *std::min_element(rgb.begin(), rgb.end());
  const float d = hi - lo, l = (hi + lo) / 2;
  if (d < 1e-8f)
    return {0, 0, l};
  float h = hi == rgb[0]   ? (rgb[1] - rgb[2]) / d
            : hi == rgb[1] ? (rgb[2] - rgb[0]) / d + 2
                           : (rgb[0] - rgb[1]) / d + 4;
  return {wrap(h / 6), d / std::max(1e-8f, 1 - std::abs(2 * l - 1)), l};
}
} // namespace

std::string lcs_vae_fingerprint(const std::string& path) {
  const auto digest = sha256_file(path);
  const char* hex = "0123456789abcdef";
  std::string s;
  for (uint8_t b : digest) {
    s += hex[b >> 4];
    s += hex[b & 15];
  }
  return s;
}

void LcsData::validate() const {
  finite(basis, D * 3);
  finite(mean, D);
  finite(anchors, 24);
  finite(latent_mean, 24);
  finite(latent_std, 24);
  for (float s : latent_std)
    require(s > 1e-8f, "invalid latent standard deviation");
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      double v = 0;
      for (int k = 0; k < D; ++k)
        v += double(basis[k * 3 + i]) * basis[k * 3 + j];
      require(std::abs(v - (i == j ? 1 : 0)) < 1e-4, "color basis is not orthonormal");
    }
  V axis{anchors[21] - anchors[18], anchors[22] - anchors[19], anchors[23] - anchors[20]};
  require(std::isfinite(norm(axis)) && norm(axis) > 1e-5f, "invalid black-to-white anchor axis");
  if (!sharpness.empty()) {
    finite(sharpness, D);
    double n = 0, sum = 0;
    V p{};
    for (int i = 0; i < D; ++i) {
      n += double(sharpness[i]) * sharpness[i];
      sum += sharpness[i];
      for (int k = 0; k < 3; ++k)
        p[k] += sharpness[i] * basis[i * 3 + k];
    }
    require(std::abs(n - 1) < 1e-4 && std::abs(sum) < 1e-3 && norm(p) < 1e-4,
            "sharpness direction must be unit, color-orthogonal and DC-free");
  }
  require(std::isfinite(explained_variance) && explained_variance >= 0 &&
              explained_variance <= 1.00001f,
          "invalid explained variance");
}

void LcsData::save(const std::string& path) const {
  validate();
  std::vector<TensorWrite> t{
      {"basis", {D, 3}, basis},         {"mean", {D}, mean},
      {"anchor_lcs", {8, 3}, anchors},  {"latent_mean", {24}, latent_mean},
      {"latent_std", {24}, latent_std}, {"explained_variance", {1}, {explained_variance}}};
  if (!sharpness.empty())
    t.push_back({"sharpness", {D}, sharpness});
  write_safetensors_atomic(path, t, {{"slopfab_lcs", "1"}, {"vae_sha256", vae_sha256}});
}

std::shared_ptr<const LcsData> LcsData::load(const std::string& path) {
  SafeTensors st;
  st.open(path);
  const auto version = st.metadata().find("slopfab_lcs");
  require(version != st.metadata().end() && version->second == "1",
          "unsupported calibration version");
  auto d = std::make_shared<LcsData>();
  const auto read = [&](const char* name, std::vector<int64_t> shape) {
    const auto& t = st.at(name);
    require(t.dtype == DType::kF32 && t.shape == shape, "invalid calibration tensor");
    std::vector<float> v(size_t(t.numel()));
    std::memcpy(v.data(), t.data, t.nbytes);
    return v;
  };
  d->basis = read("basis", {D, 3});
  d->mean = read("mean", {D});
  d->anchors = read("anchor_lcs", {8, 3});
  d->latent_mean = read("latent_mean", {24});
  d->latent_std = read("latent_std", {24});
  d->explained_variance = read("explained_variance", {1})[0];
  if (st.find("sharpness"))
    d->sharpness = read("sharpness", {D});
  const auto hash = st.metadata().find("vae_sha256");
  require(hash != st.metadata().end() && hash->second.size() == 64, "missing VAE fingerprint");
  d->vae_sha256 = hash->second;
  d->validate();
  return d;
}

LcsData fit_lcs(const std::vector<float>& colors, const std::vector<float>& anchors) {
  require(colors.size() >= D * 8 && colors.size() % D == 0,
          "at least eight color samples are required");
  finite(colors, colors.size());
  finite(anchors, 8 * D);
  LcsData d;
  d.mean.assign(D, 0);
  d.basis.resize(D * 3);
  d.latent_mean.assign(24, 0);
  d.latent_std.assign(24, 1);
  for (int i = 0; i < D; ++i) {
    double sum = 0;
    for (size_t r = 0; r < colors.size(); r += D)
      sum += colors[r + i];
    d.mean[i] = float(sum / double(colors.size() / D));
  }
  std::vector<double> values;
  const auto vectors = eigen(covariance(colors, d.mean), values);
  std::array<int, D> order;
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    return values[a] > values[b];
  });
  require(values[order[2]] > 1e-12, "color calibration is rank deficient");
  double total = 0, top = 0;
  for (double x : values)
    total += std::max(0., x);
  for (int k = 0; k < 3; ++k) {
    top += values[order[k]];
    for (int i = 0; i < D; ++i)
      d.basis[i * 3 + k] = float(vectors[i * D + order[k]]);
  }
  d.explained_variance = float(top / total);
  for (int i = 0; i < 8; ++i) {
    V p = project(d, anchors.data() + i * D);
    d.anchors.insert(d.anchors.end(), p.begin(), p.end());
  }
  d.validate();
  return d;
}

void fit_lcs_sharpness(LcsData& d, const std::vector<float>& differences) {
  d.validate();
  require(!differences.empty() && differences.size() % D == 0, "invalid sharpness samples");
  finite(differences, differences.size());
  auto samples = differences;
  // Add the DC vector to the orthonormal color constraints.
  std::vector<float> dc(D, 1);
  for (int k = 0; k < 3; ++k) {
    double p = 0;
    for (int i = 0; i < D; ++i)
      p += dc[i] * d.basis[i * 3 + k];
    for (int i = 0; i < D; ++i)
      dc[i] -= float(p) * d.basis[i * 3 + k];
  }
  double n = 0;
  for (float x : dc)
    n += double(x) * x;
  if (n > 1e-12)
    for (float& x : dc)
      x /= float(std::sqrt(n));
  else
    std::fill(dc.begin(), dc.end(), 0);
  for (size_t r = 0; r < samples.size(); r += D) {
    for (int k = 0; k < 3; ++k) {
      double p = 0;
      for (int i = 0; i < D; ++i)
        p += samples[r + i] * d.basis[i * 3 + k];
      for (int i = 0; i < D; ++i)
        samples[r + i] -= float(p) * d.basis[i * 3 + k];
    }
    double p = 0;
    for (int i = 0; i < D; ++i)
      p += samples[r + i] * dc[i];
    for (int i = 0; i < D; ++i)
      samples[r + i] -= float(p) * dc[i];
  }
  std::vector<double> values;
  auto vectors = eigen(covariance(samples, std::vector<float>(D, 0)), values);
  const int k = int(std::max_element(values.begin(), values.end()) - values.begin());
  require(values[k] > 1e-12, "sharpness calibration is rank deficient");
  d.sharpness.resize(D);
  double alignment = 0;
  for (int i = 0; i < D; ++i) {
    d.sharpness[i] = float(vectors[i * D + k]);
    for (size_t r = 0; r < samples.size(); r += D)
      alignment += d.sharpness[i] * samples[r + i];
  }
  if (alignment < 0)
    for (float& x : d.sharpness)
      x = -x;
  d.validate();
}

bool LcsSettings::active() const {
  return strength != 0 || contrast != 1 || brightness != 0 || saturation != 1 || temperature != 0 ||
         sharpness != 0 || (anchor != 0 && anchor_strength != 0);
}

void LcsSettings::validate() const {
  require(mode >= 0 && mode <= 2 && anchor >= 0 && anchor <= 2, "invalid effect mode");
  require(start_step >= 0 && end_step >= start_step, "invalid step range");
  for (float x : color)
    require(std::isfinite(x) && x >= 0 && x <= 1, "color must be RGB in [0,1]");
  const auto range = [](float x, float lo, float hi) {
    return std::isfinite(x) && x >= lo && x <= hi;
  };
  require(range(strength, 0, 2) && range(contrast, 0, 4) && range(brightness, -1, 1) &&
              range(saturation, 0, 4) && range(temperature, -1, 1) && range(sharpness, -5, 5) &&
              range(anchor_strength, 0, 1),
          "invalid effect strength or tone range");
}

LcsSettings parse_lcs_settings(const std::string& text) try {
  LcsSettings s;
  const auto root = json::parse(text.empty() ? "{}" : text);
  require(root.is_object(), "settings must be an object");
  for (const auto& item : root.as_object()) {
    const auto& k = item.first;
    const auto& v = item.second;
    const auto number = [&] {
      double x = v.as_number();
      require(std::isfinite(x) && std::abs(x) <= 2147483647., "invalid number");
      return float(x);
    };
    const auto integer = [&] {
      double x = v.as_number();
      require(std::isfinite(x) && x >= 0 && x <= 2147483647. && std::floor(x) == x, "invalid step");
      return int(x);
    };
    if (k == "color") {
      const auto& a = v.as_array();
      require(a.size() == 3, "color needs three channels");
      for (int i = 0; i < 3; ++i)
        s.color[i] = float(a[i].as_number());
    } else if (k == "mode") {
      const auto& m = v.as_string();
      require(m == "interpolated" || m == "type_i" || m == "type_ii", "invalid color mode");
      s.mode = m == "type_i" ? 1 : m == "type_ii" ? 2 : 0;
    } else if (k == "anchor") {
      const auto& m = v.as_string();
      require(m == "none" || m == "self" || m == "smooth", "invalid anchor mode");
      s.anchor = m == "self" ? 1 : m == "smooth" ? 2 : 0;
    } else if (k == "strength")
      s.strength = number();
    else if (k == "contrast")
      s.contrast = number();
    else if (k == "brightness")
      s.brightness = number();
    else if (k == "saturation")
      s.saturation = number();
    else if (k == "temperature")
      s.temperature = number();
    else if (k == "sharpness")
      s.sharpness = number();
    else if (k == "anchor_strength")
      s.anchor_strength = number();
    else if (k == "start_step")
      s.start_step = integer();
    else if (k == "end_step")
      s.end_step = integer();
    else
      throw std::invalid_argument("LCS: unknown setting " + k);
  }
  s.validate();
  return s;
} catch (const std::runtime_error& e) {

  throw std::invalid_argument(std::string("LCS settings: ") + e.what());
}

SamplingEffect make_lcs_effect(std::shared_ptr<const LcsData> data, LcsSettings s,
                               std::vector<float> mask) {
  require(bool(data), "missing calibration");
  data->validate();
  s.validate();
  require(s.sharpness == 0 || !data->sharpness.empty(), "calibration has no sharpness direction");
  for (float x : mask)
    require(std::isfinite(x) && x >= 0 && x <= 1, "mask must be in [0,1]");
  if (!s.active())
    return {};
  return [data, s, mask = std::move(mask), space = ColorSpace(*data),
          previous = std::vector<V>{}](const SamplingStep& step) mutable {
    if (step.modality != 0 || step.step < s.start_step || step.step > s.end_step)
      return true;
    const size_t plane = size_t(step.height / 2) * size_t(step.width / 2), rows = step.count / D;
    require(step.row_width == D && step.height > 0 && step.width > 0 && step.height % 2 == 0 &&
                step.width % 2 == 0 && step.frames > 0 && rows == plane * size_t(step.frames) &&
                step.count % D == 0,
            "invalid H3 patch geometry");
    require(mask.empty() || mask.size() == plane || mask.size() == rows,
            "mask does not match latent patch grid");
    const auto& d = *data;
    std::vector<V> colors(rows), means(step.frames, V{});
    for (size_t r = 0; r < rows; ++r) {
      float raw[D];
      for (int i = 0; i < D; ++i)
        raw[i] = step.denoised[r * D + i] * d.latent_std[i / 4] + d.latent_mean[i / 4];
      colors[r] = project(d, raw);
      means[r / plane] = add(means[r / plane], mul(colors[r], 1.f / float(plane)));
    }
    const V target_hsl = rgb_hsl(s.color), target = space.encode(target_hsl);
    const bool tone =
        s.contrast != 1 || s.brightness != 0 || s.saturation != 1 || s.temperature != 0;
    for (size_t r = 0; r < rows; ++r) {
      const float weight = mask.empty() ? 1 : mask[r % mask.size()];
      if (weight == 0)
        continue;
      V c = colors[r];
      if (s.strength != 0) {
        V one = add(c, mul(sub(target, means[r / plane]), s.strength));
        V hsl = space.decode(c);
        hsl = {wrap(hsl[0] + s.strength * hue_delta(target_hsl[0] - hsl[0])),
               unit(hsl[1] + s.strength * (target_hsl[1] - hsl[1])),
               unit(hsl[2] + s.strength * (target_hsl[2] - hsl[2]))};
        V two = space.encode(hsl);
        const float t = s.mode == 1 ? 1 : s.mode == 2 ? 0 : unit(step.sigma);
        c = add(mul(one, t), mul(two, 1 - t));
      }
      if (tone) {
        const float l = dot(sub(c, space.black), space.axis) / dot(space.axis, space.axis);
        V chroma = sub(c, add(space.black, mul(space.axis, l)));
        c = add(add(space.black, mul(space.axis, (l - .5f) * s.contrast + .5f + s.brightness)),
                add(mul(chroma, s.saturation), mul(space.warm, s.temperature)));
      }
      if (s.anchor && s.anchor_strength > 0) {
        V anchor = c;
        if (s.anchor == 1 && previous.size() == rows)
          anchor = previous[r];
        if (s.anchor == 2) {
          const size_t y = (r % plane) / size_t(step.width / 2), x = r % size_t(step.width / 2);
          V sum = colors[r];
          float count = 1;
          if (x > 0) {
            sum = add(sum, colors[r - 1]);
            ++count;
          }
          if (x + 1 < size_t(step.width / 2)) {
            sum = add(sum, colors[r + 1]);
            ++count;
          }
          if (y > 0) {
            sum = add(sum, colors[r - step.width / 2]);
            ++count;
          }
          if (y + 1 < size_t(step.height / 2)) {
            sum = add(sum, colors[r + step.width / 2]);
            ++count;
          }
          anchor = mul(sum, 1 / count);
        }
        c = add(c, mul(sub(anchor, c), s.anchor_strength));
      }
      const V delta = mul(sub(c, colors[r]), weight);
      for (int i = 0; i < D; ++i) {
        float change = 0;
        for (int k = 0; k < 3; ++k)
          change += d.basis[i * 3 + k] * delta[k];
        if (s.sharpness != 0)
          change += s.sharpness * weight * d.sharpness[i];
        if (change != 0)
          step.denoised[r * D + i] += change / d.latent_std[i / 4];
      }
    }
    previous = std::move(colors);
    return true;
  };
}
} // namespace slopfab
