#include "slopfab/text/clip_projection.h"
#include "slopfab/int8_weight.h"
#include "slopfab/tensor_convert.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>
#include <thread>

namespace slopfab::text {
namespace {
void require(bool condition, const std::string& message) {
  if (!condition)
    throw std::invalid_argument("ClipProj: " + message);
}

bool floating(DType type) {
  return type == DType::kF32 || type == DType::kF16 || type == DType::kBF16;
}

float round_type(float value, DType type) {
  if (type == DType::kF16)
    return f16_to_f32(f32_to_f16(value));
  if (type == DType::kBF16)
    return bf16_to_f32(f32_to_bf16(value));
  return value;
}

std::vector<float> vector(const SafeTensors& file, const std::string& name, int width) {
  const auto& view = file.at(name);
  require(view.shape == std::vector<int64_t>{width} && floating(view.dtype),
          name + " has invalid shape or dtype");
  auto values = to_f32(view);
  for (float value : values)
    require(std::isfinite(value), name + " contains nonfinite values");
  return values;
}

int integer(const std::string& text) {
  size_t used = 0;
  int value = std::stoi(text, &used);
  require(used == text.size(), "invalid integer metadata: " + text);
  return value;
}

// Cache-sized output tiles avoid expanding a 500 MB fp16 MLP into a full
// transposed fp32 copy. Workers own separate token rows and share one tile.
std::vector<float> multiply(const std::vector<float>& x, int rows, int in, int out,
                            const TensorView& w, bool transposed, const std::vector<float>& bias,
                            DType compute, const std::function<void()>& cancel) {
  std::vector<float> result(size_t(rows) * out);
  for (int start = 0; start < out; start += 256) {
    if (cancel)
      cancel();
    const int count = std::min(256, out - start);
    std::vector<float> tile(size_t(in) * count);
    if (transposed) {
      TensorView slice = w;
      slice.shape = {count, in};
      slice.nbytes = size_t(count) * in * dtype_size(w.dtype);
      slice.data = static_cast<const uint8_t*>(w.data) + size_t(start) * in * dtype_size(w.dtype);
      auto values = to_f32(slice);
      for (int j = 0; j < count; ++j)
        for (int k = 0; k < in; ++k)
          tile[size_t(k) * count + j] = round_type(values[size_t(j) * in + k], compute);
    } else {
      for (int k = 0; k < in; ++k) {
        TensorView slice = w;
        slice.shape = {count};
        slice.nbytes = size_t(count) * dtype_size(w.dtype);
        slice.data =
            static_cast<const uint8_t*>(w.data) + (size_t(k) * out + start) * dtype_size(w.dtype);
        auto values = to_f32(slice);
        std::copy(values.begin(), values.end(), tile.begin() + size_t(k) * count);
      }
    }
    for (float v : tile)
      require(std::isfinite(v), "projection weights contain nonfinite values");
    const unsigned workers = std::min<unsigned>(
        unsigned(rows), std::max(1u, std::min(8u, std::thread::hardware_concurrency())));
    auto work = [&](unsigned worker) {
      for (int row = int(worker); row < rows; row += int(workers)) {
        float* y = result.data() + size_t(row) * out + start;
        for (int k = 0; k < in; ++k) {
          const float a = x[size_t(row) * in + k];
          const float* b = tile.data() + size_t(k) * count;
          for (int j = 0; j < count; ++j)
            y[j] += a * b[j];
        }
        for (int j = 0; j < count; ++j)
          y[j] = round_type(y[j] + (bias.empty() ? 0.f : round_type(bias[start + j], compute)),
                            compute);
      }
    };
    std::vector<std::thread> threads;
    try {
      for (unsigned i = 1; i < workers; ++i)
        threads.emplace_back(work, i);
      work(0);
    } catch (...) {
      for (auto& t : threads)
        t.join();
      throw;
    }
    for (auto& t : threads)
      t.join();
  }
  return result;
}
}

ClipProjection::ClipProjection(const std::string& path) {
  file_.open(path);
  const auto& input = file_.at("mean_in");
  const auto& output = file_.at("mean_out");
  require(input.shape.size() == 1 && output.shape.size() == 1, "means must be vectors");
  require(input.shape[0] > 0 && input.shape[0] <= 32768 && output.shape[0] > 0 &&
              output.shape[0] <= 32768,
          "invalid projection dimensions");
  input_ = int(input.shape[0]);
  output_ = int(output.shape[0]);
  auto tap = file_.metadata().find("tap");
  require(tap != file_.metadata().end(), "missing tap metadata");
  tap_ = integer(tap->second);
  require(tap_ >= 0 && tap_ <= 35, "tap must select an unnormalized Qwen hidden state (0..35)");
  for (auto pair : {std::make_pair("d_in", input_), std::make_pair("d_out", output_)}) {
    auto it = file_.metadata().find(pair.first);
    if (it != file_.metadata().end())
      require(integer(it->second) == pair.second, "dimension metadata disagrees with tensors");
  }
  mean_in_ = vector(file_, "mean_in", input_);
  std_in_ = vector(file_, "std_in", input_);
  mean_out_ = vector(file_, "mean_out", output_);
  std_out_ = vector(file_, "std_out", output_);
  for (float s : std_in_)
    require(s > 0, "std_in must be positive");
  for (float s : std_out_)
    require(s >= 0, "std_out must be nonnegative");
  if (file_.find("sink_out"))
    sink_ = vector(file_, "sink_out", output_);
  if (const auto* w = file_.find("W"))
    require(w->shape == std::vector<int64_t>{input_, output_} && floating(w->dtype),
            "W must have shape [d_in,d_out] and floating dtype");
  std::set<int> indices;
  for (const auto& t : file_.tensors()) {
    if (t.first.rfind("mlp.", 0) != 0)
      continue;
    auto dot = t.first.find('.', 4);
    require(dot != std::string::npos, "invalid MLP tensor name");
    const int index = integer(t.first.substr(4, dot - 4));
    require(index >= 0 && index <= 32 &&
                (t.first.substr(dot) == ".weight" || t.first.substr(dot) == ".bias"),
            "invalid MLP tensor name");
    indices.insert(index);
  }
  int width = input_;
  DType dtype = DType::kUnknown;
  for (int index : indices) {
    const auto name = "mlp." + std::to_string(index);
    const auto& w = file_.at(name + ".weight");
    require(w.shape.size() == 2 && w.shape[1] == width && w.shape[0] > 0 && w.shape[0] <= 65536 &&
                floating(w.dtype),
            "invalid MLP shape or dtype: " + name);
    if (dtype == DType::kUnknown)
      dtype = w.dtype;
    require(w.dtype == dtype, "MLP layers must share a dtype");
    width = int(w.shape[0]);
    if (file_.find(name + ".bias"))
      (void)vector(file_, name + ".bias", width);
    layers_.push_back(index);
  }
  require(file_.find("W") || !layers_.empty(), "checkpoint has neither W nor an MLP");
  require(layers_.empty() || width == output_, "MLP output width does not match mean_out");
}

int ClipProjection::input_width() const {
  return input_;
}

int ClipProjection::output_width() const {
  return output_;
}

int ClipProjection::tap() const {
  return tap_;
}

PromptEmbedding ClipProjection::apply(const PromptEmbedding& hidden,
                                      const std::function<void()>& cancel) const {
  require(hidden.hidden_size == input_ && hidden.num_tokens > 0 &&
              hidden.num_tokens <= kMaxPromptTokens &&
              hidden.data.size() == size_t(hidden.num_tokens) * input_,
          "student hidden state shape mismatch");
  std::vector<float> x(hidden.data.size());
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = (hidden.data[i] - mean_in_[i % input_]) / std_in_[i % input_];
    require(std::isfinite(x[i]), "student hidden state contains nonfinite values");
  }
  PromptEmbedding result;
  result.num_tokens = hidden.num_tokens;
  result.hidden_size = output_;
  result.modality_tags = hidden.modality_tags;
  if (const auto* w = file_.find("W"))
    result.data =
        multiply(x, hidden.num_tokens, input_, output_, *w, false, {}, DType::kF32, cancel);
  else
    result.data.resize(size_t(hidden.num_tokens) * output_);
  if (!layers_.empty()) {
    int width = input_;
    auto residual = x;
    const auto dtype = file_.at("mlp." + std::to_string(layers_[0]) + ".weight").dtype;
    for (float& v : residual)
      v = round_type(v, dtype);
    for (size_t i = 0; i < layers_.size(); ++i) {
      const auto name = "mlp." + std::to_string(layers_[i]);
      const auto& w = file_.at(name + ".weight");
      std::vector<float> bias;
      if (file_.find(name + ".bias"))
        bias = vector(file_, name + ".bias", int(w.shape[0]));
      residual = multiply(residual, hidden.num_tokens, width, int(w.shape[0]), w, true, bias, dtype,
                          cancel);
      width = int(w.shape[0]);
      if (i + 1 < layers_.size())
        for (float& v : residual)
          v = round_type(.5f * v * (1.f + std::erf(v * .7071067811865475f)), dtype);
    }
    for (size_t i = 0; i < residual.size(); ++i)
      result.data[i] += residual[i];
  }
  for (size_t i = 0; i < result.data.size(); ++i) {
    result.data[i] = result.data[i] * std_out_[i % output_] + mean_out_[i % output_];
    require(std::isfinite(result.data[i]), "projection produced nonfinite conditioning");
  }
  if (!sink_.empty())
    std::copy(sink_.begin(), sink_.end(), result.data.begin());
  return result;
}

std::vector<float> clip_dense_values(const SafeTensors& file, const std::string& name) {
  const auto& w = file.at(name);
  require(floating(w.dtype) || w.dtype == DType::kF8E4M3,
          "unsupported dense student dtype: " + name);
  auto values = to_f32(w);
  if (w.dtype == DType::kF8E4M3) {
    auto prefix = name.substr(0, name.size() - 7);
    const auto* scale = file.find(prefix + ".weight_scale");
    if (!scale)
      scale = file.find(prefix + ".scale_weight");
    require(scale && scale->numel() == 1 && floating(scale->dtype),
            "FP8 student requires a scalar weight scale: " + name);
    const auto s = to_f32(*scale)[0];
    require(std::isfinite(s) && s > 0, "invalid FP8 scale");
    for (float& v : values)
      v *= s;
  }
  return values;
}

ClipStudentConfig validate_clip_student(const SafeTensors& file, const ClipProjection& projection) {
  require(projection.output_width() == 5120, "H3 requires 5120 output features");
  ClipStudentConfig c;
  c.prefix =
      file.find("model.language_model.embed_tokens.weight") ? "model.language_model." : "model.";
  const auto& w = file.at(c.prefix + "embed_tokens.weight");
  require(w.shape.size() == 2 && w.shape[0] == 151936 && (w.shape[1] == 2560 || w.shape[1] == 4096),
          "use a ComfyUI Qwen3-VL-4B or 8B encoder");
  c.width = int(w.shape[1]);
  c.layers = projection.tap();
  require(c.width == projection.input_width(),
          "encoder width does not match projection; pair 4B with 4B or 8B with 8B");
  require(floating(w.dtype) || w.dtype == DType::kI8,
          "student embedding requires BF16, FP16, FP32 or INT8 ConvRot (INT4 is unsupported)");
  if (w.dtype == DType::kI8)
    (void)read_int8_weight(file, w.name, "ClipProj embedding");
  const int inner = c.width == 2560 ? 9728 : 12288;
  for (int i = 0; i < c.layers; ++i) {
    const auto p = c.prefix + "layers." + std::to_string(i) + ".";
    for (const auto& pair : {std::make_pair("input_layernorm.weight", c.width),
                             {"post_attention_layernorm.weight", c.width},
                             {"self_attn.q_norm.weight", 128},
                             {"self_attn.k_norm.weight", 128}}) {
      const auto& t = file.at(p + pair.first);
      require(t.shape == std::vector<int64_t>{pair.second} && floating(t.dtype),
              "student norm shape/dtype mismatch: " + t.name);
    }

    struct Shape {
      const char* name;
      int out, in;
    };

    for (const auto& shape : {Shape{"self_attn.q_proj", 4096, c.width},
                              {"self_attn.k_proj", 1024, c.width},
                              {"self_attn.v_proj", 1024, c.width},
                              {"self_attn.o_proj", c.width, 4096},
                              {"mlp.gate_proj", inner, c.width},
                              {"mlp.up_proj", inner, c.width},
                              {"mlp.down_proj", c.width, inner}}) {
      const auto& t = file.at(p + shape.name + ".weight");
      require(t.shape == std::vector<int64_t>{shape.out, shape.in},
              "student linear shape mismatch: " + t.name);
      require(floating(t.dtype) || t.dtype == DType::kI8 || t.dtype == DType::kF8E4M3,
              "unsupported student weight dtype: " + t.name);
      if (t.dtype == DType::kI8)
        (void)read_int8_weight(file, t.name, "ClipProj");
      if (t.dtype == DType::kF8E4M3) {
        const auto base = p + shape.name;
        const auto* scale = file.find(base + ".weight_scale");
        if (!scale)
          scale = file.find(base + ".scale_weight");
        require(scale && scale->numel() == 1 && floating(scale->dtype),
                "FP8 weight is missing scalar scale: " + t.name);
      }
    }
  }
  return c;
}

std::vector<float> clip_student_embeddings(const SafeTensors& file, const ClipStudentConfig& c,
                                           const std::vector<int32_t>& ids) {
  require(!ids.empty() && ids.size() <= kMaxPromptTokens, "prompt token count outside capacity");
  const auto& w = file.at(c.prefix + "embed_tokens.weight");
  Int8WeightState q;
  if (w.dtype == DType::kI8)
    q = read_int8_weight(file, w.name, "ClipProj embedding");
  std::vector<float> values(ids.size() * c.width);
  for (size_t r = 0; r < ids.size(); ++r) {
    require(ids[r] >= 0 && ids[r] < 151936, "token outside Qwen vocabulary");
    float* row = values.data() + r * c.width;
    if (w.dtype != DType::kI8) {
      auto view = w;
      view.shape = {c.width};
      view.nbytes = c.width * dtype_size(w.dtype);
      view.data = static_cast<const uint8_t*>(w.data) + size_t(ids[r]) * view.nbytes;
      auto v = to_f32(view);
      std::copy(v.begin(), v.end(), row);
    } else {
      const int group = q.rotation_group;
      for (int base = 0; base < c.width; base += group) {
        float* v = row + base;
        for (int i = 0; i < group; ++i)
          v[i] = q.codes[size_t(ids[r]) * c.width + base + i];
        for (int stride = 1; stride < group; stride *= 4)
          for (int start = 0; start < group; start += 4 * stride)
            for (int j = start; j < start + stride; ++j) {
              float a = v[j], b = v[j + stride], cc = v[j + 2 * stride], d = v[j + 3 * stride];
              v[j] = a + b + cc - d;
              v[j + stride] = a + b - cc + d;
              v[j + 2 * stride] = a - b + cc + d;
              v[j + 3 * stride] = -a + b + cc + d;
            }
        for (int i = 0; i < group; ++i)
          v[i] *= q.scales[ids[r]] / std::sqrt(float(group));
      }
    }
  }
  return values;
}
}
