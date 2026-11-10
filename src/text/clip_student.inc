// Backend adapters provide the existing BF16 Qwen operations. Do not apply
// Krea's chat template. P-H3 captures five taps and normalizes only the last one.
#include "../generation/clip_student.h"
#include <cmath>
#include <algorithm>

namespace slopfab::text {
PromptEmbedding CLIP_STUDENT_FUNCTION(const std::string& path, const ClipProjection& projection,
                                      const std::vector<int32_t>& ids,
                                      const std::function<void()>& check_cancel) {
  CLIP_RUNTIME rt(path);
  const auto config = validate_clip_student(rt.checkpoint, projection);
  rt.allow_clip_fp8 = true;
  auto x = CLIP_FROM_HOST(clip_student_embeddings(rt.checkpoint, config, ids), int(ids.size()),
                          config.width);
  std::vector<float> c, s;
  for (size_t r = 0; r < ids.size(); ++r)
    for (int j = 0; j < 64; ++j) {
      const float angle = float(r * std::pow(5000000., -double(j) / 64));
      c.push_back(bf16_to_f32(f32_to_bf16(std::cos(angle))));
      s.push_back(bf16_to_f32(f32_to_bf16(std::sin(angle))));
    }
  PromptEmbedding result;
  result.num_tokens = int(ids.size());
  result.hidden_size = projection.input_width();
  result.modality_tags.assign(ids.size(), 1);
  if (projection.ph3_image()) result.data.resize(ids.size() * result.hidden_size);
  auto capture = [&](int layer, const auto& state) {
    const auto& taps = projection.hidden_taps();
    const auto found = std::find(taps.begin(), taps.end(), layer);
    if (found == taps.end()) return;
    const auto hidden = CLIP_TO_HOST(state);
    const size_t slot = size_t(found - taps.begin());
    for (size_t row = 0; row < ids.size(); ++row)
      std::copy_n(hidden.data() + row * config.width, config.width,
                  result.data.data() + row * result.hidden_size + slot * config.width);
  };
  // Comfy's list-valued tap captures BEFORE layer i: tap 24 runs layers 0..23.
  for (int i = 0; i < config.layers; ++i) {
    if (check_cancel)
      check_cancel();
    if (projection.ph3_image()) capture(i, x);
    const auto p = config.prefix + "layers." + std::to_string(i) + ".";
    auto y = rt.norm(x, p + "input_layernorm.weight");
    auto q = rt.norm(rt.project(y, p + "self_attn.q_proj"), p + "self_attn.q_norm.weight", 128);
    auto k = rt.norm(rt.project(y, p + "self_attn.k_proj"), p + "self_attn.k_norm.weight", 128);
    auto v = rt.project(y, p + "self_attn.v_proj");
    CLIP_ROPE(q, 32, c, s, false);
    CLIP_ROPE(k, 8, c, s, false);
    auto update = rt.project(rt.attend(q, k, v, 32, 8, true), p + "self_attn.o_proj");
    CLIP_RESIDUAL(x, update);
    y = rt.norm(x, p + "post_attention_layernorm.weight");
    auto gate = rt.project(y, p + "mlp.gate_proj"), up = rt.project(y, p + "mlp.up_proj");
    update = rt.project(CLIP_ACTIVATE(up, &gate), p + "mlp.down_proj");
    CLIP_RESIDUAL(x, update);
    rt.release_prefix(p);
  }
  if (projection.ph3_image())
    capture(config.layers, rt.norm(x, config.prefix + "norm.weight"));
  else
    result.data = CLIP_TO_HOST(x);
  return result;
}
}
