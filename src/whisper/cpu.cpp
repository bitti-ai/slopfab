#include "internal.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace slopfab::whisper {
namespace {
using Buffer = std::vector<float>;

const Buffer& data(const Tensor& t) {
  return *std::static_pointer_cast<Buffer>(t.data);
}

Tensor make(int r, int c) {
  return {r, c, std::make_shared<Buffer>(size_t(r) * c)};
}

Buffer& writable(Tensor& t) {
  return *std::static_pointer_cast<Buffer>(t.data);
}

class CPU final : public Backend {
public:
  Tensor upload(const Buffer& v, int r, int c) override {
    if (v.size() != size_t(r) * c)
      throw std::invalid_argument("Whisper: upload shape");
    return {r, c, std::make_shared<Buffer>(v)};
  }

  Buffer download(const Tensor& t) override {
    return data(t);
  }

  Tensor linear(const Tensor& x, const Tensor& w, const Tensor* b) override {
    auto y = make(x.rows, w.rows);
    auto& out = writable(y);
    for (int r = 0; r < x.rows; ++r)
      for (int o = 0; o < w.rows; ++o) {
        float sum = 0;
        for (int k = 0; k < x.cols; ++k)
          sum += data(x)[size_t(r) * x.cols + k] * data(w)[size_t(o) * x.cols + k];
        out[size_t(r) * y.cols + o] = sum + (b ? data(*b)[o] : 0);
      }
    return y;
  }

  Tensor norm(const Tensor& x, const Tensor& w, const Tensor& b) override {
    auto y = make(x.rows, x.cols);
    auto& out = writable(y);
    for (int r = 0; r < x.rows; ++r) {
      double mean = 0, var = 0;
      for (int c = 0; c < x.cols; ++c)
        mean += data(x)[size_t(r) * x.cols + c];
      mean /= x.cols;
      for (int c = 0; c < x.cols; ++c) {
        double d = data(x)[size_t(r) * x.cols + c] - mean;
        var += d * d;
      }
      float inv = float(1 / std::sqrt(var / x.cols + 1e-5));
      for (int c = 0; c < x.cols; ++c)
        out[size_t(r) * x.cols + c] =
            (data(x)[size_t(r) * x.cols + c] - float(mean)) * inv * data(w)[c] + data(b)[c];
    }
    return y;
  }

  Tensor gelu(const Tensor& x) override {
    auto y = upload(data(x), x.rows, x.cols);
    for (auto& v : writable(y))
      v = 0.5f * v * (1 + std::erf(v * 0.7071067811865475f));
    return y;
  }

  Tensor add(const Tensor& a, const Tensor& b) override {
    auto y = upload(data(a), a.rows, a.cols);
    for (size_t i = 0; i < data(a).size(); ++i)
      writable(y)[i] += data(b)[i];
    return y;
  }

  Tensor rows(const Tensor& x, int first, int count) override {
    return upload(Buffer(data(x).begin() + size_t(first) * x.cols,
                         data(x).begin() + size_t(first + count) * x.cols),
                  count, x.cols);
  }

  Tensor concat(const Tensor& a, const Tensor& b) override {
    if (!a.data)
      return b;
    auto v = data(a);
    v.insert(v.end(), data(b).begin(), data(b).end());
    return upload(v, a.rows + b.rows, a.cols);
  }

  Tensor im2col(const Tensor& x, int stride) override {
    auto y = make((x.rows + stride - 1) / stride, x.cols * 3);
    for (int r = 0; r < y.rows; ++r)
      for (int c = 0; c < x.cols; ++c)
        for (int k = 0; k < 3; ++k) {
          int at = r * stride + k - 1;
          writable(y)[size_t(r) * y.cols + c * 3 + k] =
              at >= 0 && at < x.rows ? data(x)[size_t(at) * x.cols + c] : 0;
        }
    return y;
  }

  Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, int heads, bool causal,
                   Buffer* probabilities) override {
    auto out = make(q.rows, q.cols);
    int dim = q.cols / heads;
    Buffer scores(k.rows);
    if (probabilities)
      probabilities->resize(size_t(heads) * q.rows * k.rows);
    for (int h = 0; h < heads; ++h)
      for (int r = 0; r < q.rows; ++r) {
        float maximum = -INFINITY;
        for (int j = 0; j < k.rows; ++j) {
          float sum = 0;
          for (int c = 0; c < dim; ++c)
            sum += data(q)[size_t(r) * q.cols + h * dim + c] *
                   data(k)[size_t(j) * k.cols + h * dim + c];
          scores[j] = causal && j > r ? -INFINITY : sum / std::sqrt(float(dim));
          maximum = std::max(maximum, scores[j]);
        }
        float sum = 0;
        for (auto& s : scores) {
          s = std::exp(s - maximum);
          sum += s;
        }
        for (auto& s : scores)
          s /= sum;
        if (probabilities)
          std::copy(scores.begin(), scores.end(),
                    probabilities->begin() + (size_t(h) * q.rows + r) * k.rows);
        for (int c = 0; c < dim; ++c) {
          float z = 0;
          for (int j = 0; j < k.rows; ++j)
            z += scores[j] * data(v)[size_t(j) * v.cols + h * dim + c];
          writable(out)[size_t(r) * out.cols + h * dim + c] = z;
        }
      }
    return out;
  }
};
}

std::unique_ptr<Backend> cpu_backend() {
  return std::make_unique<CPU>();
}
}
