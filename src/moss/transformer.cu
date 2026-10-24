#include "runtime.cuh"
#include <algorithm>
#include <cmath>

namespace slopfab::moss {
namespace {
Tensor qkv_attention(Runtime& rt, const Tensor& y, const std::string& projection,
                     const std::string& norms, const Tensor& rope) {
  Tensor q, k, v;
  if (rt.checkpoint.find(projection + "qkv.weight")) {
    auto qkv = rt.project(y, projection + "qkv");
    q = slice(qkv, 0, 3072);
    k = slice(qkv, 3072, 3072);
    v = slice(qkv, 6144, 3072);
  } else {
    q = rt.project(y, projection + "q");
    k = rt.project(y, projection + "k");
    v = rt.project(y, projection + "v");
  }
  q = rt.norm(q, norms + "q_norm.weight", 128);
  k = rt.norm(k, norms + "k_norm.weight", 128);
  rotary(q, 24, rope, 128);
  rotary(k, 24, rope, 128);
  return rt.attend(q, k, v, 24, 24);
}

Tensor refine_text(Runtime& rt, const Tensor& text) {
  auto x = rt.project(text, "text_embedder.proj");
  const auto rope = image_rope(0, 0, text.rows, true);
  for (int i = 0; i < 2; ++i) {
    const std::string p = "text_embedder.blocks." + std::to_string(i) + ".";
    auto y = rt.norm(x, p + "norm1.weight");
    auto update = rt.project(qkv_attention(rt, y, p, p, rope), p + "proj");
    residual(x, update);
    y = rt.norm(x, p + "norm2.weight");
    auto gate = rt.project(y, p + "ffn_gate"), up = rt.project(y, p + "ffn_up");
    update = rt.project(activate(up, &gate, true), p + "ffn_down");
    residual(x, update);
  }
  return x;
}
}

Tensor image_prediction(Runtime& rt, const Tensor& latent, const Tensor& text, int height,
                        int width, float timestep, const std::vector<int>& skip) {
  std::vector<float> frequencies(256);
  for (int i = 0; i < 128; ++i) {
    const double phase = double(timestep) * std::pow(10000., -double(i) / 128.);
    frequencies[i] = float(std::cos(phase));
    frequencies[128 + i] = float(std::sin(phase));
  }
  auto time = rt.project(activate(rt.project(from_host(frequencies, 1, 256), "time_embedding.0")),
                         "time_embedding.2");
  auto mod = rt.project(time, "shared_block_modulation");
  auto head = rt.project(time, "video_head_modulation");
  auto x = concatenate(rt.project(latent, "video_embedder"), refine_text(rt, text));
  const auto rope = image_rope(height, width, text.rows);
  for (int i = 0; i < 42; ++i) {
    if (std::find(skip.begin(), skip.end(), i) != skip.end())
      continue;
    const std::string p = "layers." + std::to_string(i) + ".";
    auto y = rt.norm(x, p + "norm1.weight", 0, false, &mod, 0);
    auto update = rt.project(qkv_attention(rt, y, p + "proj.", p, rope), p + "proj.o");
    residual(x, update, &mod, 2);
    y = rt.norm(x, p + "norm2.weight", 0, false, &mod, 3);
    auto both = rt.project(y, p + "proj.ffn_up_gate");
    auto up = slice(both, 0, 8192), gate = slice(both, 8192, 8192);
    update = rt.project(activate(up, &gate, true), p + "proj.ffn_down");
    residual(x, update, &mod, 5);
  }
  // The image prefix is contiguous; shortening rows retains ownership of the allocation.
  x.rows = height * width;
  return rt.project(rt.norm(x, "final_norm_video.weight", 0, false, &head, 0),
                    "final_linear_video");
}
} // namespace slopfab::moss
