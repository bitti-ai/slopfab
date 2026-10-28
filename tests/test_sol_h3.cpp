#include "harness.h"
#include "slopfab/vae/sol_h3.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

SLOPFAB_TEST(sol_h3_window_geometry) {
  using namespace slopfab::vae;
  CHECK(sol_h3_frames(1) == 1);
  CHECK(sol_h3_frames(5) == 17);
  CHECK(sol_h3_frames(6) == 18);
  CHECK(sol_h3_frames(10) == 34);
  const auto g = build_sol_h3_geometry(6, 17, 19);
  CHECK(g.patches == 6 * 17 * 19);
  CHECK(g.rows == g.patches + 10);
  CHECK(g.blocks.size() == 2);
  CHECK(g.blocks[0].patches == 5 * 17 * 19);
  CHECK(g.blocks[1].patches == 17 * 19);
  for (int parity = 0; parity < 2; ++parity) {
    std::vector<int> seen(g.patches, 0);
    bool crossed = false;
    for (const auto& w : g.windows[parity]) {
      CHECK(w.patches > 0 && w.patches <= 5 * 16 * 16);
      CHECK(w.rows.size() == size_t(w.patches + 5) || w.rows.size() == size_t(w.patches + 10));
      const int first_block = w.rows[0] / (5 * 17 * 19);
      const int last_block = w.rows[w.patches - 1] / (5 * 17 * 19);
      crossed |= first_block != last_block;
      for (int i = 0; i < w.patches; ++i) {
        CHECK(w.rows[i] >= 0 && w.rows[i] < g.patches);
        ++seen[w.rows[i]];
      }
      for (size_t i = w.patches; i < w.rows.size(); ++i) {
        CHECK(w.rows[i] >= g.patches && w.rows[i] < g.rows);
        const int b = (w.rows[i] - g.patches) / 5;
        CHECK(b == first_block || b == last_block);
      }
    }
    CHECK(std::all_of(seen.begin(), seen.end(), [](int n) { return n == 1; }));
    CHECK(crossed == (parity == 1));
    for (size_t i = size_t(g.patches) * 48; i < g.rope[parity].cosine.size(); ++i) {
      CHECK(g.rope[parity].cosine[i] == 1);
      CHECK(g.rope[parity].sine[i] == 0);
    }
  }
  // First temporal coordinate is -0.8 (not normalized by the clip length).
  CHECK_NEAR(g.rope[0].cosine[0], std::cos(-.8 * 6.283185307179586), 1e-6);
  // Spatial windows restart their local positions at x=16.
  CHECK(g.rope[0].cosine[16] == g.rope[0].cosine[16 * 48 + 16]);
  CHECK(g.rope[0].cosine != g.rope[1].cosine);
}

SLOPFAB_TEST(sol_h3_geometry_rejects_invalid_dimensions) {
  for (const auto shape : std::vector<std::vector<int>>{{0, 2, 2}, {1, 0, 2}, {1, 2, -1},
                                                       {1000000, 1000000, 1000000}}) {
    bool threw = false;
    try { slopfab::vae::build_sol_h3_geometry(shape[0], shape[1], shape[2]); }
    catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
  }
}
