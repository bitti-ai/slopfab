#include "slopfab/vae/sol_h3.h"

#include <algorithm>
#include <cmath>
#include <climits>
#include <stdexcept>

namespace slopfab::vae {
bool is_sol_h3_vae(const SafeTensors& checkpoint) {
  return checkpoint.find("blocks.0.attn.to_qkv.weight") &&
         checkpoint.find("x_embedder.weight") && checkpoint.find("register_tokens");
}

void validate_sol_h3_vae(const SafeTensors& checkpoint) {
  auto require = [&](const std::string& name, std::vector<int64_t> shape) {
    const auto& t = checkpoint.at(name);
    if (t.shape != shape || (t.dtype != DType::kF32 && t.dtype != DType::kF16 &&
                            t.dtype != DType::kBF16))
      throw std::invalid_argument("Sol-H3-VAE: invalid shape or dtype for " + name);
  };
  require("post_quant_conv.weight", {24, 24, 1, 1, 1});
  require("post_quant_conv.bias", {24});
  require("x_embedder.weight", {2048, 24});
  require("x_embedder.bias", {2048});
  require("register_tokens", {1, 4, 2048});
  require("norm_out.weight", {2048});
  require("norm_out.bias", {2048});
  require("proj_out.weight", {3072, 2048});
  require("proj_out.bias", {3072});
  for (int i = 0; i < 24; ++i) {
    const auto p = "blocks." + std::to_string(i) + ".";
    for (const auto* name : {"norm1.weight", "norm2.weight", "scale1", "scale2"})
      require(p + name, {2048});
    for (const auto& spec : std::vector<std::pair<std::string, std::pair<int, int>>>{
             {"attn.to_qkv", {6144, 2048}}, {"attn.to_out", {2048, 2048}},
             {"ff.w1", {16384, 2048}}, {"ff.w2", {2048, 8192}}}) {
      require(p + spec.first + ".weight", {spec.second.first, spec.second.second});
      require(p + spec.first + ".bias", {spec.second.first});
    }
  }
  if (checkpoint.find("blocks.24.attn.to_qkv.weight"))
    throw std::invalid_argument("Sol-H3-VAE: expected the released 24-layer decoder");
}

int sol_h3_frames(int time) {
  if (time <= 0 || time > INT_MAX / 4)
    throw std::invalid_argument("Sol-H3-VAE: invalid latent time");
  return time * 4 - 3 * ((time + 4) / 5);
}

SolH3Geometry build_sol_h3_geometry(int T, int H, int W) {
  sol_h3_frames(T);
  if (H <= 0 || W <= 0 || H > INT_MAX / 16 || W > INT_MAX / 16 ||
      T > INT_MAX / (5 * 2048) || int64_t(T) > (INT_MAX / 2) / int64_t(H) / W)
    throw std::invalid_argument("Sol-H3-VAE: invalid or excessive latent dimensions");
  SolH3Geometry g;
  g.patches = T * H * W;
  const int nb = (T + 4) / 5;
  g.rows = g.patches + nb * 5;
  auto globals = [&](SolH3Window& win, int b) {
    for (int j = 0; j < 5; ++j) win.rows.push_back(g.patches + b * 5 + j);
  };
  for (int b = 0; b < nb; ++b) {
    SolH3Window win;
    for (int i = b * 5 * H * W; i < std::min(T, (b + 1) * 5) * H * W; ++i)
      win.rows.push_back(i);
    win.patches = static_cast<int>(win.rows.size());
    globals(win, b);
    g.blocks.push_back(std::move(win));
  }
  for (int parity = 0; parity < 2; ++parity) {
    const int st = parity * 2, s = parity * 8;
    for (int t0 = -st; t0 < T; t0 += 5)
      for (int y0 = -s; y0 < H; y0 += 16)
        for (int x0 = -s; x0 < W; x0 += 16) {
          SolH3Window win;
          const int lo = std::max(t0, 0), hi = std::min(t0 + 5, T);
          for (int t = lo; t < hi; ++t)
            for (int y = std::max(y0, 0); y < std::min(y0 + 16, H); ++y)
              for (int x = std::max(x0, 0); x < std::min(x0 + 16, W); ++x)
                win.rows.push_back((t * H + y) * W + x);
          win.patches = static_cast<int>(win.rows.size());
          if (!win.patches) continue;
          globals(win, lo / 5);
          if ((hi - 1) / 5 != lo / 5) globals(win, (hi - 1) / 5);
          g.windows[parity].push_back(std::move(win));
        }
    auto& r = g.rope[parity];
    r.cosine.assign(size_t(g.rows) * 48, 1);
    r.sine.assign(size_t(g.rows) * 48, 0);
    for (int t = 0; t < T; ++t)
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
          const float coords[] = {2 * (((t + st) % 5 + .5f) / 5) - 1,
                                  2 * (((y + s) % 16 + .5f) / 16) - 1,
                                  2 * (((x + s) % 16 + .5f) / 16) - 1};
          const size_t row = size_t((t * H + y) * W + x) * 48;
          for (int a = 0; a < 3; ++a)
            for (int f = 0; f < 8; ++f) {
              const float angle = 6.283185307179586f * coords[a] / std::pow(100.f, f / 8.f);
              const int j = a * 8 + f;
              r.cosine[row + j] = r.cosine[row + j + 24] = std::cos(angle);
              r.sine[row + j] = r.sine[row + j + 24] = std::sin(angle);
            }
        }
  }
  return g;
}
} // namespace slopfab::vae
