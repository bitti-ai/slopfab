#include "slopfab/text/qwen_vision.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace slopfab::text {

int qwen3vl_deepstack_slot(int text_decoder_layer) {
  return text_decoder_layer >= 0 && text_decoder_layer < 3 ? text_decoder_layer : -1;
}
namespace {
constexpr int kFactor = 32;
constexpr double kMinPixels = 65536.0;
constexpr double kMaxPixels = 16777216.0;
constexpr double kConditioningMaxPixels = 4194304.0;
constexpr size_t kConditioningMaxPatches = 16384;

int round_factor(double value) {
  return std::max(kFactor, static_cast<int>(std::round(value / kFactor)) * kFactor);
}

QwenImageGrid smart_grid(int width, int height, double max_pixels) {
  if (width <= 0 || height <= 0)
    throw std::runtime_error("Qwen image: invalid size");
  const double ratio = std::max(width, height) /
      static_cast<double>(std::min(width, height));
  if (ratio > 200.0)
    throw std::runtime_error("Qwen image: aspect ratio exceeds 200:1");

  int resized_h = round_factor(height);
  int resized_w = round_factor(width);
  const double source_pixels = static_cast<double>(height) * width;
  const double rounded_pixels = static_cast<double>(resized_h) * resized_w;
  if (rounded_pixels > max_pixels) {
    const double beta = std::sqrt(source_pixels / max_pixels);
    resized_h = std::max(kFactor,
        static_cast<int>(std::floor(height / beta / kFactor)) * kFactor);
    resized_w = std::max(kFactor,
        static_cast<int>(std::floor(width / beta / kFactor)) * kFactor);
  } else if (rounded_pixels < kMinPixels) {
    const double beta = std::sqrt(kMinPixels / source_pixels);
    resized_h = static_cast<int>(std::ceil(height * beta / kFactor)) * kFactor;
    resized_w = static_cast<int>(std::ceil(width * beta / kFactor)) * kFactor;
  }
  return {1, resized_h / 16, resized_w / 16};
}

void check(const SafeTensors& st, const std::string& name, const std::vector<int64_t>& shape) {
  const TensorView* t = st.find(name);
  if (!t) throw std::runtime_error("Qwen vision: missing " + name);
  uint64_t elements = 1;
  for (const int64_t extent : shape) {
    if (extent <= 0 || elements >
            std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(extent))
      throw std::logic_error("Qwen vision: invalid expected manifest shape");
    elements *= static_cast<uint64_t>(extent);
  }
  const uint64_t bytes = elements * sizeof(uint16_t);
  if (t->dtype != DType::kBF16 || t->shape != shape || t->nbytes != bytes ||
      t->data == nullptr)
    throw std::runtime_error("Qwen vision: incompatible tensor " + name);
}
}  // namespace

QwenVisionCheckpoint load_qwen3vl_vision_checkpoint(const SafeTensors& st) {
  std::string p;
  const bool flat = st.find("visual.patch_embed.proj.weight") != nullptr;
  const bool nested = st.find("model.visual.patch_embed.proj.weight") != nullptr;
  if (flat && nested)
    throw std::runtime_error("Qwen vision: contradictory visual prefixes");
  if (flat) p = "visual.";
  else if (nested) p = "model.visual.";
  else throw std::runtime_error("Qwen vision: visual patch embedding is absent");

  check(st, p + "patch_embed.proj.weight", {1152, 3, 2, 16, 16});
  check(st, p + "patch_embed.proj.bias", {1152});
  check(st, p + "pos_embed.weight", {2304, 1152});
  for (int i = 0; i < 27; ++i) {
    const std::string b = p + "blocks." + std::to_string(i) + ".";
    check(st, b + "norm1.weight", {1152}); check(st, b + "norm1.bias", {1152});
    check(st, b + "norm2.weight", {1152}); check(st, b + "norm2.bias", {1152});
    check(st, b + "attn.qkv.weight", {3456, 1152}); check(st, b + "attn.qkv.bias", {3456});
    check(st, b + "attn.proj.weight", {1152, 1152}); check(st, b + "attn.proj.bias", {1152});
    check(st, b + "mlp.linear_fc1.weight", {4304, 1152}); check(st, b + "mlp.linear_fc1.bias", {4304});
    check(st, b + "mlp.linear_fc2.weight", {1152, 4304}); check(st, b + "mlp.linear_fc2.bias", {1152});
  }
  auto merger = [&](const std::string& base, bool main) {
    check(st, base + "norm.weight", {main ? 1152 : 4608});
    check(st, base + "norm.bias", {main ? 1152 : 4608});
    check(st, base + "linear_fc1.weight", {4608, 4608});
    check(st, base + "linear_fc1.bias", {4608});
    check(st, base + "linear_fc2.weight", {5120, 4608});
    check(st, base + "linear_fc2.bias", {5120});
  };
  merger(p + "merger.", true);
  for (int i = 0; i < 3; ++i)
    merger(p + "deepstack_merger_list." + std::to_string(i) + ".", false);

  size_t count = 0;
  for (const auto& kv : st.tensors()) if (kv.first.rfind(p, 0) == 0) ++count;
  if (count != 351) throw std::runtime_error("Qwen vision: expected 351 tensors, found " + std::to_string(count));
  return {&st, p, {}};
}

QwenVisionPositions qwen3vl_vision_positions(const QwenImageGrid& g, int side, int merge) {
  const size_t patches = g.patch_count();
  if (g.temporal <= 0 || g.height <= 0 || g.width <= 0 || side <= 0 || merge <= 0 ||
      g.height % merge || g.width % merge || patches == 0 ||
      patches > std::numeric_limits<size_t>::max() / 3)
    throw std::runtime_error("Qwen vision: invalid position grid");
  QwenVisionPositions out;
  out.learned.reserve(patches); out.rotary_thw.reserve(patches * 3);
  auto bucket = [side](int x, int extent) {
    // torch.linspace(0, side-1, extent).long(): conversion truncates.
    return extent == 1 ? 0 : static_cast<int>((static_cast<int64_t>(x) * (side - 1)) / (extent - 1));
  };
  for (int t = 0; t < g.temporal; ++t)
    for (int by = 0; by < g.height / merge; ++by)
      for (int bx = 0; bx < g.width / merge; ++bx)
        for (int my = 0; my < merge; ++my)
          for (int mx = 0; mx < merge; ++mx) {
            const int y = by * merge + my, x = bx * merge + mx;
            out.learned.push_back(bucket(y, g.height) * side + bucket(x, g.width));
            out.rotary_thw.insert(out.rotary_thw.end(), {t, y, x});
          }
  return out;
}

QwenMultimodalPlan qwen3vl_multimodal_plan(const std::vector<int32_t>& ids,
                                           const std::vector<QwenImageGrid>& grids,
                                           int32_t vs, int32_t pad, int32_t ve) {
  QwenMultimodalPlan out;
  const size_t L = ids.size();
  if (L > std::numeric_limits<size_t>::max() / 3)
    throw std::runtime_error("Qwen vision: multimodal prompt is too large");
  for (const QwenImageGrid& grid : grids)
    if (grid.merged_token_count() == 0)
      throw std::runtime_error("Qwen vision: invalid multimodal grid");
  out.position_ids.resize(3 * L);
  size_t cursor = 0, image = 0;
  int32_t next = 0;
  auto scalar = [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i, ++next)
      for (int a = 0; a < 3; ++a) out.position_ids[static_cast<size_t>(a) * L + i] = next;
  };
  while (cursor < L) {
    auto it = std::find(ids.begin() + static_cast<std::ptrdiff_t>(cursor), ids.end(), vs);
    if (it == ids.end()) { scalar(cursor, L); cursor = L; break; }
    const size_t start = static_cast<size_t>(it - ids.begin());
    scalar(cursor, start + 1); // text prefix and vision_start are ordinary 1-D positions
    if (image >= grids.size()) throw std::runtime_error("Qwen vision: more vision blocks than grids");
    const auto& g = grids[image++];
    const size_t merged = g.merged_token_count();
    if (start + 1 + merged >= L) throw std::runtime_error("Qwen vision: truncated image-pad run");
    const int mh = g.height / 2, mw = g.width / 2;
    const int32_t block_pad = ids[start + 1];
    if (block_pad != pad && block_pad != 151656)
      throw std::runtime_error("Qwen vision: expected image or video pad");
    for (size_t j = 0; j < merged; ++j) {
      const size_t row = start + 1 + j;
      if (ids[row] != block_pad) throw std::runtime_error("Qwen vision: pad count disagrees with grid");
      const int t = static_cast<int>(j / static_cast<size_t>(mh * mw));
      const int rem = static_cast<int>(j % static_cast<size_t>(mh * mw));
      out.position_ids[row] = next + t;
      out.position_ids[L + row] = next + rem / mw;
      out.position_ids[2 * L + row] = next + rem % mw;
      out.image_rows.push_back(static_cast<int32_t>(row));
    }
    const size_t end = start + 1 + merged;
    if (ids[end] != ve) throw std::runtime_error("Qwen vision: image-pad run lacks vision_end");
    next += std::max({g.temporal, mh, mw});
    cursor = end; // vision_end is consumed by the next scalar run
  }
  if (image != grids.size()) throw std::runtime_error("Qwen vision: fewer vision blocks than grids");
  return out;
}

void qwen3vl_vision_rope_tables(const QwenVisionPositions& p, std::vector<float>& cos,
                               std::vector<float>& sin, int head_dim, float theta) {
  if (head_dim <= 0 || head_dim % 4 || theta <= 0 || p.rotary_thw.size() % 3)
    throw std::runtime_error("Qwen vision: invalid rotary shape");
  const size_t rows = p.rotary_thw.size() / 3;
  const int axis_half = head_dim / 4; // 18 frequencies for a 72-wide head
  cos.resize(rows * head_dim); sin.resize(rows * head_dim);
  // `inv` depends only on `j` — 18 distinct values for a 72-wide head — but was
  // evaluated once per (row, axis, j), which is 2 * rows * 18 calls to
  // `std::pow` for 18 answers. `ViTDecoder::build_rope` and
  // `text::rope_inv_freq` already hoist theirs.
  //
  // Bit-identical, and deliberately so: the same `std::pow(double, double)`
  // for the same `j`, just evaluated once. `docs/text_encoder_spec.md` 6.4
  // pins the fp64-then-round evaluation, so this must not become `exp2` or a
  // running reciprocal — those are faster and give different last bits.
  std::vector<double> inv_freq(static_cast<size_t>(axis_half));
  for (int j = 0; j < axis_half; ++j) {
    inv_freq[static_cast<size_t>(j)] =
        std::pow(static_cast<double>(theta), -2.0 * j / (head_dim / 2));
  }
  for (size_t r = 0; r < rows; ++r) {
    const int coords[2] = {p.rotary_thw[r * 3 + 1], p.rotary_thw[r * 3 + 2]};
    for (int a = 0; a < 2; ++a)
      for (int j = 0; j < axis_half; ++j) {
        const double inv = inv_freq[static_cast<size_t>(j)];
        const float angle = static_cast<float>(coords[a] * inv);
        const int k = a * axis_half + j;
        cos[r * head_dim + k] = cos[r * head_dim + k + head_dim / 2] = std::cos(angle);
        sin[r * head_dim + k] = sin[r * head_dim + k + head_dim / 2] = std::sin(angle);
      }
  }
}

void qwen3vl_decoder_rope_tables(const QwenMultimodalPlan& p, int tokens,
                                 std::vector<float>& cos, std::vector<float>& sin,
                                 int head_dim, float theta) {
  if (tokens <= 0 || head_dim != 128 || p.position_ids.size() != static_cast<size_t>(3 * tokens))
    throw std::runtime_error("Qwen vision: invalid decoder rotary plan");
  cos.resize(static_cast<size_t>(tokens) * head_dim); sin.resize(cos.size());
  // As above: 64 distinct values, previously recomputed for every one of the
  // `tokens` rows. Same `std::pow` call, same argument, evaluated once.
  std::vector<double> inv_freq(static_cast<size_t>(head_dim / 2));
  for (int j = 0; j < head_dim / 2; ++j) {
    inv_freq[static_cast<size_t>(j)] =
        std::pow(static_cast<double>(theta), -2.0 * j / head_dim);
  }
  for (int r = 0; r < tokens; ++r) for (int j = 0; j < head_dim / 2; ++j) {
    // Qwen3-VL interleaves THW for 20 cycles, then assigns four trailing
    // frequencies to T: section counts [24,20,20].
    const int axis = j < 60 ? j % 3 : 0;
    const double inv = inv_freq[static_cast<size_t>(j)];
    const float a = static_cast<float>(p.position_ids[static_cast<size_t>(axis) * tokens + r] * inv);
    cos[static_cast<size_t>(r)*head_dim+j]=cos[static_cast<size_t>(r)*head_dim+j+64]=std::cos(a);
    sin[static_cast<size_t>(r)*head_dim+j]=sin[static_cast<size_t>(r)*head_dim+j+64]=std::sin(a);
  }
}

size_t QwenImageGrid::patch_count() const {
  if (temporal <= 0 || height <= 0 || width <= 0) return 0;
  const size_t t = static_cast<size_t>(temporal);
  const size_t h = static_cast<size_t>(height);
  const size_t w = static_cast<size_t>(width);
  if (t > std::numeric_limits<size_t>::max() / h) return 0;
  const size_t th = t * h;
  if (th > std::numeric_limits<size_t>::max() / w) return 0;
  return th * w;
}

size_t QwenImageGrid::merged_token_count() const {
  if (height <= 0 || width <= 0 || (height & 1) != 0 || (width & 1) != 0)
    return 0;
  const size_t patches = patch_count();
  return patches == 0 ? 0 : patches / 4;
}

QwenImageGrid qwen3vl_image_grid(int width, int height) {
  return smart_grid(width, height, kMaxPixels);
}

QwenImageGrid qwen3vl_conditioning_grid(int width, int height) {
  QwenImageGrid grid = smart_grid(width, height, kConditioningMaxPixels);
  if (grid.patch_count() == 0 ||
      grid.patch_count() > kConditioningMaxPatches ||
      grid.merged_token_count() == 0) {
    throw std::runtime_error("Qwen image: conditioning grid exceeds exact capacity");
  }
  return grid;
}

size_t qwen3vl_conditioning_token_count(
    const std::vector<QwenImageGrid>& grids, size_t nonvision_tokens,
    size_t max_prompt_tokens) {
  if (max_prompt_tokens == 0 || nonvision_tokens > max_prompt_tokens)
    throw std::runtime_error("Qwen image: conditioning exceeds max prompt tokens");
  size_t total = nonvision_tokens;
  for (const QwenImageGrid& grid : grids) {
    const size_t patches = grid.patch_count();
    const size_t merged = grid.merged_token_count();
    if (patches == 0 || patches > kConditioningMaxPatches || merged == 0)
      throw std::runtime_error("Qwen image: invalid conditioning grid");
    if (merged > max_prompt_tokens - total ||
        size_t(2) > max_prompt_tokens - total - merged)
      throw std::runtime_error("Qwen image: conditioning exceeds max prompt tokens");
    total += merged + 2;
  }
  return total;
}

QwenPixelValues qwen3vl_patchify_resized_rgb(const std::vector<uint8_t>& rgb,
                                             int width, int height) {
  return qwen3vl_patchify_resized_rgb_pair(rgb, rgb, width, height);
}

QwenPixelValues qwen3vl_patchify_resized_rgb_pair(const std::vector<uint8_t>& rgb,
    const std::vector<uint8_t>& second, int width, int height) {
  if (width <= 0 || height <= 0 || width % kFactor || height % kFactor)
    throw std::runtime_error("Qwen image: resized dimensions must be positive multiples of 32");
  if (rgb.size() != static_cast<size_t>(width) * height * 3 || second.size() != rgb.size())
    throw std::runtime_error("Qwen image: RGB byte count does not match dimensions");

  QwenPixelValues out;
  out.grid = {1, height / 16, width / 16};
  constexpr int patch = 16, merge = 2, temporal = 2, channels = 3;
  constexpr int row_width = channels * temporal * patch * patch;
  out.rows.resize(out.grid.patch_count() * row_width);

  // Equivalent to reshape(t,tp,c,h/m,m,p,w/m,m,p), then transpose
  // (t,h/m,w/m,m,m,c,tp,p,p). This keeps each 2x2 merge group contiguous.
  size_t dst = 0;
  for (int tile_y = 0; tile_y < height / (merge * patch); ++tile_y)
    for (int tile_x = 0; tile_x < width / (merge * patch); ++tile_x)
      for (int merge_y = 0; merge_y < merge; ++merge_y)
        for (int merge_x = 0; merge_x < merge; ++merge_x)
          for (int channel = 0; channel < channels; ++channel)
            for (int time = 0; time < temporal; ++time)
              for (int py = 0; py < patch; ++py)
                for (int px = 0; px < patch; ++px) {
                  const auto& pixels = time == 0 ? rgb : second;
                  const int y = (tile_y * merge + merge_y) * patch + py;
                  const int x = (tile_x * merge + merge_x) * patch + px;
                  out.rows[dst++] = pixels[(static_cast<size_t>(y) * width + x) * 3 + channel] /
                                           127.5f -
                                       1.0f;
                }
  return out;
}

std::vector<int32_t> qwen3vl_image_block(const std::vector<int32_t>& label_ids,
                                         size_t merged_tokens, int32_t vision_start_id,
                                         int32_t image_pad_id, int32_t vision_end_id) {
  if (merged_tokens == 0) throw std::runtime_error("Qwen image: empty vision block");
  std::vector<int32_t> ids;
  ids.reserve(label_ids.size() + merged_tokens + 2);
  ids.insert(ids.end(), label_ids.begin(), label_ids.end());
  ids.push_back(vision_start_id);
  ids.insert(ids.end(), merged_tokens, image_pad_id);
  ids.push_back(vision_end_id);
  return ids;
}

}  // namespace slopfab::text
