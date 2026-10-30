#include "vulkan_runtime.h"
#include "slopfab/int8_weight.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/text/tokenizer.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace slopfab::krea_vulkan {
namespace {
Tensor embedding(Runtime& rt, const std::vector<int32_t>& ids) {
  const std::string name = "model.language_model.embed_tokens.weight";
  const auto& w = rt.checkpoint.at(name);
  if (w.shape != std::vector<int64_t>{151936, 2560})
    throw std::invalid_argument(
        "Krea 2 requires Qwen3-VL-4B INT8 ConvRot or floating-point weights (INT4 is unsupported)");
  Int8WeightState q;
  if (w.dtype == DType::kI8)
    q = read_int8_weight(rt.checkpoint, name, "Krea 2 Qwen embedding");
  std::vector<float> values(ids.size() * 2560);
  for (size_t r = 0; r < ids.size(); ++r) {
    if (ids[r] < 0 || ids[r] >= 151936)
      throw std::invalid_argument("Krea 2 token outside vocabulary");
    if (w.dtype != DType::kI8) {
      TensorView row = w;
      row.shape = {2560};
      row.nbytes = 2560 * dtype_size(w.dtype);
      row.data = static_cast<const uint8_t*>(w.data) + size_t(ids[r]) * row.nbytes;
      auto v = to_f32(row);
      std::copy(v.begin(), v.end(), values.begin() + r * 2560);
      continue;
    }
    const int group = q.rotation_group;
    for (int c = 0; c < 2560; c += group) {
      float* v = values.data() + r * 2560 + c;
      for (int i = 0; i < group; ++i)
        v[i] = q.codes[size_t(ids[r]) * 2560 + c + i];
      for (int stride = 1; stride < group; stride *= 4)
        for (int base = 0; base < group; base += 4 * stride)
          for (int i = 0; i < stride; ++i) {
            int j = base + i;
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
  return from_host(values, int(ids.size()), 2560);
}

}

Tensor encode_prompt(const std::string& path, const std::string& tokenizer_path,
                     const std::string& prompt, const std::function<void()>& check_cancel) {
  text::Tokenizer tokenizer;
  if (tokenizer_path.empty())
    tokenizer.load_embedded();
  else
    tokenizer.load(tokenizer_path);
  if (tokenizer.token_to_id("<|im_start|>") != 151644)
    throw std::invalid_argument("Krea 2 requires the Qwen3-VL tokenizer");
  const std::string prefix =
      "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n";
  auto ids = tokenizer.encode(prefix + prompt);
  if (tokenizer.encode(prefix).size() != 34 || ids.size() < 34)
    throw std::invalid_argument("Krea 2 tokenizer prefix mismatch");
  ids.resize(std::min(size_t(541), ids.size()));
  const auto suffix = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n");
  if (suffix.size() != 5)
    throw std::invalid_argument("Krea 2 tokenizer suffix mismatch");
  ids.insert(ids.end(), suffix.begin(), suffix.end());
  Runtime rt(path);
  auto x = embedding(rt, ids);
  std::vector<float> c, s;
  for (int r = 0; r < int(ids.size()); ++r)
    for (int j = 0; j < 64; ++j) {
      float angle = float(r * std::pow(5000000., -double(j) / 64));
      // Qwen computes trig in float32 and casts tables to the activation dtype.
      c.push_back(round_bf(std::cos(angle)));
      s.push_back(round_bf(std::sin(angle)));
    }
  const int tokens = int(ids.size()) - 34;
  Tensor result(tokens * 12, 2560);
  int tap = 0;
  for (int i = 0; i < 35; ++i) {
    check_cancel();
    const std::string p = "model.language_model.layers." + std::to_string(i) + ".";
    auto y = rt.norm(x, p + "input_layernorm.weight");
    auto q = rt.norm(rt.project(y, p + "self_attn.q_proj"), p + "self_attn.q_norm.weight", 128);
    auto k = rt.norm(rt.project(y, p + "self_attn.k_proj"), p + "self_attn.k_norm.weight", 128);
    auto v = rt.project(y, p + "self_attn.v_proj");
    ops::rope(q, 32, c, s, false);
    ops::rope(k, 8, c, s, false);
    auto update = rt.project(rt.attend(q, k, v, 32, 8, true), p + "self_attn.o_proj");
    ops::residual(x, update);
    y = rt.norm(x, p + "post_attention_layernorm.weight");
    auto gate = rt.project(y, p + "mlp.gate_proj"), up = rt.project(y, p + "mlp.up_proj");
    update = rt.project(ops::activate(up, &gate), p + "mlp.down_proj");
    ops::residual(x, update);
    if ((i + 1) % 3 == 2) {
      uint32_t n = tokens * 2560;
      dispatch(*x.engine, x.data, x.data, x.data, result.data,
               {CaptureText, n, 2560, uint32_t(tap)}, (n + 255) / 256);
      ++tap;
    }
    rt.release_prefix(p);
  }
  return result;
}

}
