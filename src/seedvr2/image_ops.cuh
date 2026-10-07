#pragma once
#include "runtime.cuh"

namespace slopfab::seedvr2 {
// Tile conversions keep their input in float until the final BF16 rounding.
void prepare_image_tile(const float* input, Tensor& tile, int height, int width, int y, int x);
void prepare_latent_tile(const float* input, Tensor& tile, int height, int width, int y, int x);
void accumulate_image_tile(const Tensor& tile, float* sum, float* coverage, int height, int width,
                           int padded_height, int padded_width, int y, int x, int overlap);
void normalize_image_tiles(float* sum, const float* coverage, size_t count, int height, int width,
                           int channels, bool decoded);
}
