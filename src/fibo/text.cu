#include "runtime.cuh"
#include "slopfab/int8_weight.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/text/tokenizer.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

namespace slopfab::fibo {
namespace {
float bf(float x) {
  return __bfloat162float(__float2bfloat16(x));
}

Tensor merge(Runtime& rt, const Tensor& x, const std::string& p, bool deep) {
  Tensor y = deep ? rows(x, 0, x.rows) : layer_norm(rt, x, p + ".norm");
  y.rows /= 4;
  y.cols *= 4;
  if (deep)
    y = layer_norm(rt, y, p + ".norm");
  return rt.project(gelu(rt.project(y, p + ".linear_fc1"), false), p + ".linear_fc2");
}

Tensor embedding(Runtime& rt, const std::vector<int32_t>& ids) {
  const std::string name = "model.language_model.embed_tokens.weight";
  const auto& w = rt.checkpoint.at(name);
  if (w.shape != std::vector<int64_t>{151936, 2560})
    throw std::invalid_argument(
        "Fibo requires Qwen3-VL-4B INT8 ConvRot or floating-point weights (INT4 is unsupported)");
  Int8WeightState q;
  if (w.dtype == DType::kI8)
    q = read_int8_weight(rt.checkpoint, name, "Fibo Qwen embedding");
  std::vector<float> values(ids.size() * 2560);
  for (size_t r = 0; r < ids.size(); ++r) {
    if (ids[r] < 0 || ids[r] >= 151936)
      throw std::invalid_argument("Fibo token outside vocabulary");
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

text::QwenPixelValues pixels(const RGBImage& source) {
  double scale = std::min(1., 768. / std::max(source.width, source.height));
  int h = std::max(1, int(std::round(source.height * scale))),
      w = std::max(1, int(std::round(source.width * scale)));
  int hh = std::max(32, int(std::nearbyint(h / 32.)) * 32),
      ww = std::max(32, int(std::nearbyint(w / 32.)) * 32);
  if (hh * ww < 3136) {
    double b = std::sqrt(3136. / (h * w));
    hh = int(std::ceil(h * b / 32)) * 32;
    ww = int(std::ceil(w * b / 32)) * 32;
  }
  auto image = resize_vision_image(resize_vision_image(source, w, h), ww, hh);
  return text::qwen3vl_patchify_resized_rgb(image.pixels, ww, hh);
}
}

Tensor vision(Runtime& rt, const text::QwenPixelValues& image, std::vector<Tensor>& deep) {
  const auto& g = image.grid;
  const int n = int(g.patch_count());
  const std::string p = "model.visual.";
  if (rt.checkpoint.at(p + "patch_embed.proj.weight").shape !=
      std::vector<int64_t>{1024, 3, 2, 16, 16})
    throw std::invalid_argument("Fibo requires the 1024-wide Qwen3-VL-4B visual tower");
  auto x = rt.project(from_host(image.rows, n, 1536), p + "patch_embed.proj");
  auto pos = text::qwen3vl_vision_positions(g);
  const auto table = to_f32(rt.checkpoint.at(p + "pos_embed.weight"));
  std::vector<float> position(size_t(n) * 1024), c, s;
  for (int r = 0; r < n; ++r) {
    int y = pos.rotary_thw[r * 3 + 1], xx = pos.rotary_thw[r * 3 + 2];
    float yy = g.height == 1 ? 0 : float(y) * 47 / (g.height - 1),
          fx = g.width == 1 ? 0 : float(xx) * 47 / (g.width - 1);
    int y0 = int(yy), x0 = int(fx), y1 = std::min(47, y0 + 1), x1 = std::min(47, x0 + 1);
    float dy = yy - y0, dx = fx - x0;
    float weights[] = {bf((1 - dy) * (1 - dx)), bf((1 - dy) * dx), bf(dy * (1 - dx)), bf(dy * dx)};
    int idx[] = {y0 * 48 + x0, y0 * 48 + x1, y1 * 48 + x0, y1 * 48 + x1};
    for (int j = 0; j < 1024; ++j) {
      float a = bf(table[idx[0] * 1024 + j] * weights[0]);
      for (int k = 1; k < 4; ++k)
        a = bf(a + bf(table[idx[k] * 1024 + j] * weights[k]));
      position[r * 1024 + j] = a;
    }
    for (int axis = 0; axis < 2; ++axis)
      for (int j = 0; j < 16; ++j) {
        float a = float((axis ? xx : y) * std::pow(10000., -double(j) / 16));
        c.push_back(std::cos(a));
        s.push_back(std::sin(a));
      }
  }
  moss::residual(x, from_host(position, n, 1024));
  deep.clear();
  for (int i = 0; i < 24; ++i) {
    std::string b = p + "blocks." + std::to_string(i);
    auto qkv = rt.project(layer_norm(rt, x, b + ".norm1"), b + ".attn.qkv");
    auto q = moss::slice(qkv, 0, 1024), k = moss::slice(qkv, 1024, 1024),
         v = moss::slice(qkv, 2048, 1024);
    rope(q, 16, c, s, false);
    rope(k, 16, c, s, false);
    auto y = rt.project(rt.attend(q, k, v, 16, 16), b + ".attn.proj");
    moss::residual(x, y);
    y = rt.project(gelu(rt.project(layer_norm(rt, x, b + ".norm2"), b + ".mlp.linear_fc1"), true),
                   b + ".mlp.linear_fc2");
    moss::residual(x, y);
    if (i == 5 || i == 11 || i == 17)
      deep.push_back(
          merge(rt, x, p + "deepstack_merger_list." + std::to_string(deep.size()), true));
    rt.release_prefix(b + ".");
  }
  auto output = merge(rt, x, p + "merger", false);
  rt.release_prefix(p);
  return output;
}

std::vector<Tensor> encode_prompt(const std::string& path, const std::string& tokenizer_path,
                                  const std::string& prompt, const std::vector<RGBImage>& images,
                                  const std::function<void()>& check_cancel) {
  text::Tokenizer tokenizer;
  if (tokenizer_path.empty())
    tokenizer.load_embedded();
  else
    tokenizer.load(tokenizer_path);
  if (tokenizer.token_to_id("<|image_pad|>") != 151655 ||
      tokenizer.token_to_id("<|im_start|>") != 151644)
    throw std::invalid_argument("Fibo requires a Qwen3-VL tokenizer");
  Runtime rt(path);
  std::vector<std::string> glyphs;
  const auto formatted = edit_prompt(prompt, int(images.size()), glyphs);
  std::vector<text::QwenPixelValues> prepared;
  std::vector<text::QwenImageGrid> grids;
  for (const auto& i : images)
    prepared.push_back(pixels(i));
  for (const auto& t : glyphs)
    prepared.push_back(pixels(render_text(t)));
  for (const auto& i : prepared)
    grids.push_back(i.grid);
  auto basic = tokenizer.encode(formatted);
  std::vector<int32_t> ids;
  size_t image = 0;
  for (auto id : basic) {
    if (id == 151655) {
      if (image >= grids.size())
        throw std::invalid_argument("Fibo vision marker mismatch");
      ids.insert(ids.end(), grids[image++].merged_token_count(), id);
    } else
      ids.push_back(id);
  }
  if (image != grids.size() || ids.empty() || ids.size() > 4096)
    throw std::invalid_argument(
        "Fibo prompt exceeds 4096 multimodal tokens or has unmatched image markers");
  auto plan = text::qwen3vl_multimodal_plan(ids, grids);
  auto x = embedding(rt, ids);
  std::vector<Tensor> all_deep;
  size_t row = 0;
  for (const auto& input : prepared) {
    check_cancel();
    std::vector<Tensor> deep;
    auto visual = vision(rt, input, deep);
    std::vector<int32_t> indices(plan.image_rows.begin() + row,
                                 plan.image_rows.begin() + row + visual.rows);
    scatter(x, visual, indices, false);
    row += visual.rows;
    if (all_deep.empty())
      all_deep = std::move(deep);
    else
      for (int i = 0; i < 3; ++i)
        all_deep[i] = join(all_deep[i], deep[i]);
  }
  std::vector<float> full_c, full_s, c, s;
  text::qwen3vl_decoder_rope_tables(plan, int(ids.size()), full_c, full_s);
  for (size_t r = 0; r < ids.size(); ++r) {
    c.insert(c.end(), full_c.begin() + r * 128, full_c.begin() + r * 128 + 64);
    s.insert(s.end(), full_s.begin() + r * 128, full_s.begin() + r * 128 + 64);
  }
  const int bundles[6][3] = {{9, 20, 31}, {5, 16, 27},  {7, 18, 29},
                             {9, 20, 31}, {11, 22, 33}, {13, 24, 35}};
  std::set<int> taps;
  for (const auto& b : bundles)
    for (int i : b)
      taps.insert(i);
  std::map<int, Tensor> captured;
  for (int i = 0; i < 35; ++i) {
    check_cancel();
    const std::string p = "model.language_model.layers." + std::to_string(i) + ".";
    auto y = rt.norm(x, p + "input_layernorm.weight");
    auto q = rt.norm(rt.project(y, p + "self_attn.q_proj"), p + "self_attn.q_norm.weight", 128);
    auto k = rt.norm(rt.project(y, p + "self_attn.k_proj"), p + "self_attn.k_norm.weight", 128),
         v = rt.project(y, p + "self_attn.v_proj");
    rope(q, 32, c, s, false);
    rope(k, 8, c, s, false);
    auto update = rt.project(rt.attend(q, k, v, 32, 8, true), p + "self_attn.o_proj");
    moss::residual(x, update);
    y = rt.norm(x, p + "post_attention_layernorm.weight");
    auto gate = rt.project(y, p + "mlp.gate_proj"), up = rt.project(y, p + "mlp.up_proj");
    update = rt.project(moss::activate(up, &gate), p + "mlp.down_proj");
    moss::residual(x, update);
    if (i < 3 && !all_deep.empty())
      scatter(x, all_deep[i], plan.image_rows, true);
    if (taps.count(i + 1))
      captured.emplace(i + 1, layer_norm(rt, x, "", true));
    rt.release_prefix(p);
  }
  std::vector<Tensor> result;
  for (const auto& b : bundles) {
    Tensor bundle(x.rows, 7680);
    for (int j = 0; j < 3; ++j)
      SLOPFAB_CUDA_CHECK(cudaMemcpy2D(bundle.ptr() + j * 2560, 7680 * 2, captured.at(b[j]).ptr(),
                                      2560 * 2, 2560 * 2, x.rows, cudaMemcpyDeviceToDevice));
    result.push_back(std::move(bundle));
  }
  return result;
}
}
