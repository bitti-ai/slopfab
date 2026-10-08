#include "slopfab/latent_upscale.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace slopfab {
void validate_latent_upscale_options(const LatentUpscaleOptions& o) {
  if (!std::isfinite(o.scale) || o.scale < 1 || o.scale > 4)
    throw std::invalid_argument("latent upscale: scale must be finite and in [1,4]");
}

std::pair<int, int> latent_upscale_dimensions(int h, int w, const LatentUpscaleOptions& o) {
  validate_latent_upscale_options(o);
  auto axis = [&](int n) {
    if (n <= 0 || double(n) * o.scale > std::numeric_limits<int>::max() / 16.0 - 2)
      throw std::invalid_argument("latent upscale: invalid or excessive dimensions");
    const double value = double(n) * o.scale / 2;
    const double base = std::floor(value), fraction = value - base;
    const int rounded = int(base) + (fraction > .5 || (fraction == .5 && int(base) % 2));
    return std::max(2, 2 * rounded);
  };
  return {axis(h), axis(w)};
}

void validate_latent_upscale_checkpoint(const SafeTensors& f) {
  size_t expected = 0;
  auto require = [&](const std::string& name, std::vector<int64_t> shape) {
    ++expected;
    const auto* t = f.find(name);
    if (!t || t->shape != shape ||
        (t->dtype != DType::kF16 && t->dtype != DType::kBF16 && t->dtype != DType::kF32))
      throw std::invalid_argument("H3 latent upscaler v1: missing or incompatible tensor " + name);
  };
  auto affine = [&](const std::string& n, int c) {
    require(n + ".weight", {c});
    require(n + ".bias", {c});
  };
  auto conv = [&](const std::string& n, int in, int out, int k, int spatial) {
    require(n + ".weight", {out, in, k, spatial, spatial});
    require(n + ".bias", {out});
  };
  conv("conv_in", 24, 512, 3, 3);
  conv("conv_out", 512, 24, 3, 3);
  affine("norm_out", 512);
  require("embed.0.weight", {64, 1});
  require("embed.0.bias", {64});
  require("embed.2.weight", {64, 64});
  require("embed.2.bias", {64});
  for (const char* stage : {"in_blocks.", "out_blocks."}) {
    for (int i = 0; i < 18; ++i) {
      const std::string p = std::string(stage) + std::to_string(i);
      if (i % 3 == 1) {
        affine(p + ".norm", 512);
        conv(p + ".dwconv", 1, 512, 5, 1);
        conv(p + ".pwconv", 512, 512, 1, 1);
      } else {
        affine(p + ".in_layers.0", 512);
        conv(p + ".in_layers.2", 512, 512, 3, 3);
        require(p + ".emb_layers.1.weight", {1024, 64});
        require(p + ".emb_layers.1.bias", {1024});
        affine(p + ".out_norm", 512);
        conv(p + ".out_layers.2", 512, 512, 3, 3);
      }
    }
  }
  if (f.tensors().size() != expected)
    throw std::invalid_argument("H3 latent upscaler: expected exactly the 3D conv v1 architecture");
}
} // namespace slopfab
