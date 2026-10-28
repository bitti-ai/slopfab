#pragma once

#include "slopfab/vae/vit_block.h"

namespace slopfab::vae {
bool is_sol_h3_vae(const SafeTensors& checkpoint);
void validate_sol_h3_vae(const SafeTensors& checkpoint);
int sol_h3_frames(int time);

// Patch rows are [T,H,W], followed by five global rows per native block.
// A window contains patch keys first, then the globals of the blocks it overlaps.
struct SolH3Window {
  std::vector<int32_t> rows;
  int patches = 0;
};
struct SolH3Geometry {
  int patches = 0, rows = 0;
  std::vector<SolH3Window> windows[2];
  std::vector<SolH3Window> blocks;
  ViTRopeTables rope[2];
};
SolH3Geometry build_sol_h3_geometry(int time, int height, int width);
} // namespace slopfab::vae
