// Rank-8 AdaLN table lookup.
//
// The whole timestep conditioning of a 33B model passes through this 8-vector,
// and every one of the 51 consumers reads the same one. A lookup that is off by
// a row, or that reads the grid backwards, conditions all 50 blocks on the
// wrong noise level at every step — which is not a crash and not even
// obviously wrong output, just a worse video.
//
// Golden values below were read directly from
// weights/transformer/fl2va_pruned_fp8_scaled.safetensors. That file is
// licence-restricted and not committed, so those tests skip when it is absent;
// the interpolation arithmetic is tested against a synthetic table and always
// runs.

#include <array>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "harness.h"
#include "vidfab/dit/adaln.h"
#include "vidfab/safetensors.h"
#include "vidfab/safetensors_write.h"

namespace {

using vidfab::dit::AdaLNLookup;
using vidfab::dit::AdaLNTable;
using vidfab::dit::FullAdaLNTimestepEmbedding;

std::string find_checkpoint() {
  for (const char* prefix : {"", "../", "../../"}) {
    const std::string p =
        std::string(prefix) + "weights/transformer/fl2va_pruned_fp8_scaled.safetensors";
    if (std::filesystem::exists(p)) return p;
  }
  return {};
}

// A synthetic table whose rows are an exactly-known linear ramp, so that
// interpolation error is attributable to the lookup and nothing else.
std::string write_synthetic_table() {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "vidfab_adaln_test";
  std::filesystem::create_directories(dir);
  const std::string path = (dir / "table.safetensors").string();

  std::vector<float> data(static_cast<size_t>(AdaLNTable::kRows) * AdaLNTable::kRank);
  for (int j = 0; j < AdaLNTable::kRows; ++j) {
    for (int c = 0; c < AdaLNTable::kRank; ++c) {
      // Column c is the row index scaled by (c + 1): linear in j, so linear
      // interpolation must be exact everywhere, not merely close.
      data[static_cast<size_t>(j) * AdaLNTable::kRank + c] =
          static_cast<float>(j) * static_cast<float>(c + 1);
    }
  }

  std::vector<vidfab::TensorWrite> tensors;
  tensors.push_back({"adaln_t_table", {AdaLNTable::kRows, AdaLNTable::kRank}, std::move(data)});
  vidfab::write_safetensors(path, tensors);
  return path;
}

VIDFAB_TEST(adaln_interpolation_arithmetic) {
  const std::string path = write_synthetic_table();
  vidfab::SafeTensors st;
  st.open(path);

  AdaLNTable table;
  table.load(st);
  CHECK(table.loaded());

  // On a linear table, linear interpolation is exact at every point, not just
  // on grid points. Sweep a range that lands mid-interval as well as on them.
  for (int k = 0; k <= 4096; ++k) {
    const float t = static_cast<float>(k) / 4096.0f;
    const std::array<float, 8> c = table.lookup(t, AdaLNLookup::kLinear);
    const double u = static_cast<double>(t) * 1024.0;
    for (int col = 0; col < 8; ++col) {
      const double want = u * (col + 1);
      if (std::fabs(c[col] - want) > 1e-2) {
        CHECK_MSG(false, "t=%.6f col=%d: got %.6f want %.6f", t, col, c[col], want);
        return;
      }
    }
  }
  CHECK(true);  // the sweep above ran clean

  // Endpoints land exactly on rows 0 and 1024.
  const std::array<float, 8> at0 = table.lookup(0.0f);
  const std::array<float, 8> at1 = table.lookup(1.0f);
  CHECK_NEAR(at0[0], 0.0, 0.0);
  CHECK_NEAR(at1[0], 1024.0, 0.0);
  CHECK_NEAR(at1[7], 1024.0 * 8.0, 0.0);

  // Grid points agree exactly between linear and nearest — that identity is
  // what makes interpolation the safe default.
  for (int j = 0; j <= 1024; j += 97) {
    const float t = static_cast<float>(j) / 1024.0f;
    const std::array<float, 8> lin = table.lookup(t, AdaLNLookup::kLinear);
    const std::array<float, 8> nea = table.lookup(t, AdaLNLookup::kNearest);
    for (int col = 0; col < 8; ++col) CHECK_NEAR(lin[col], nea[col], 1e-3);
  }

  // Mid-interval is where the two schemes must diverge, and by half a step.
  const float mid = 0.5f / 1024.0f;
  CHECK_NEAR(table.lookup(mid, AdaLNLookup::kLinear)[0], 0.5, 1e-4);
  CHECK_NEAR(table.lookup(mid, AdaLNLookup::kNearest)[0], 1.0, 1e-4);

  // Reversed indexes by 1 - t, so it mirrors the table.
  CHECK_NEAR(table.lookup(0.0f, AdaLNLookup::kLinearReversed)[0], 1024.0, 0.0);
  CHECK_NEAR(table.lookup(1.0f, AdaLNLookup::kLinearReversed)[0], 0.0, 0.0);
  CHECK_NEAR(table.lookup(0.25f, AdaLNLookup::kLinearReversed)[0], 768.0, 1e-2);

  // Out of range clamps instead of extrapolating.
  CHECK_NEAR(table.lookup(-0.5f)[0], 0.0, 0.0);
  CHECK_NEAR(table.lookup(2.0f)[0], 1024.0, 0.0);

  // The reader holds a mapping, and Windows refuses to unlink a mapped file.
  st.close();
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

VIDFAB_TEST(adaln_real_table) {
  const std::string path = find_checkpoint();
  if (path.empty()) {
    std::printf("  transformer checkpoint not present; skipping\n");
    return;
  }

  vidfab::SafeTensors st;
  st.open(path);
  AdaLNTable table;
  table.load(st);

  // Golden rows read straight out of the checkpoint.
  const float* r0 = table.row(0);
  CHECK_NEAR(r0[0], 0.46788, 1e-4);
  CHECK_NEAR(r0[1], 0.24823, 1e-4);
  CHECK_NEAR(r0[2], -0.09163, 1e-4);
  const float* r1024 = table.row(1024);
  CHECK_NEAR(r1024[0], -0.40221, 1e-4);
  CHECK_NEAR(r1024[1], 0.08635, 1e-4);

  // Endpoint lookups hit those rows exactly.
  CHECK_NEAR(table.lookup(0.0f)[0], r0[0], 0.0);
  CHECK_NEAR(table.lookup(1.0f)[0], r1024[0], 0.0);

  // The table is smooth: the second difference is a small fraction of each
  // column's range. This is the measurement that settles nearest vs
  // interpolated, so it is worth pinning rather than trusting a note.
  for (int col = 0; col < 8; ++col) {
    float lo = table.row(0)[col];
    float hi = lo;
    for (int j = 0; j < 1025; ++j) {
      const float v = table.row(j)[col];
      lo = std::min(lo, v);
      hi = std::max(hi, v);
    }
    const double range = static_cast<double>(hi) - lo;
    CHECK_MSG(range > 0.0, "column %d is constant", col);

    double worst_second_diff = 0.0;
    for (int j = 1; j < 1024; ++j) {
      const double d2 = static_cast<double>(table.row(j - 1)[col]) -
                        2.0 * table.row(j)[col] + table.row(j + 1)[col];
      worst_second_diff = std::max(worst_second_diff, std::fabs(d2));
    }
    // Measured worst case across all eight columns is 4.1e-3 of range.
    CHECK_MSG(worst_second_diff / range < 1e-2,
              "column %d second difference %.3e is %.4f of range %.6f — the table is not "
              "smooth, so linear interpolation is not the right lookup",
              col, worst_second_diff, worst_second_diff / range, range);
  }

  // Halving the grid and interpolating back recovers the omitted rows. This is
  // the direct evidence that the intended lookup is interpolated: measured
  // worst error is 2.4e-5 absolute.
  double worst = 0.0;
  for (int j = 1; j < 1024; j += 2) {
    for (int col = 0; col < 8; ++col) {
      const double mid = 0.5 * (static_cast<double>(table.row(j - 1)[col]) + table.row(j + 1)[col]);
      worst = std::max(worst, std::fabs(mid - table.row(j)[col]));
    }
  }
  CHECK_MSG(worst < 1e-4, "half-resolution interpolation error %.3e exceeds 1e-4", worst);

  // The per-column ranges decay geometrically, which is what identifies the
  // factorisation as a truncated SVD rather than an arbitrary basis. If a
  // future checkpoint broke that ordering, the components would no longer be
  // sorted by importance and the rank-8 assumption would need revisiting.
  double prev_range = 1e30;
  for (int col = 0; col < 8; ++col) {
    float lo = table.row(0)[col];
    float hi = lo;
    for (int j = 0; j < 1025; ++j) {
      lo = std::min(lo, table.row(j)[col]);
      hi = std::max(hi, table.row(j)[col]);
    }
    const double range = static_cast<double>(hi) - lo;
    CHECK_MSG(range < prev_range, "column %d range %.6f is not below column %d's %.6f", col,
              range, col - 1, prev_range);
    prev_range = range;
  }
}

}  // namespace

VIDFAB_TEST(full_adaln_timestep_sinusoid_layout) {
  const std::vector<float> at_zero = vidfab::dit::minimax_h3_timestep_sinusoid(0.0f, 4);
  CHECK(at_zero.size() == 4);
  CHECK_NEAR(at_zero[0], 1.0, 0.0);
  CHECK_NEAR(at_zero[1], 1.0, 0.0);
  CHECK_NEAR(at_zero[2], 0.0, 0.0);
  CHECK_NEAR(at_zero[3], 0.0, 0.0);

  const std::vector<float> at_one = vidfab::dit::minimax_h3_timestep_sinusoid(1.0f, 4);
  CHECK_NEAR(at_one[0], std::cos(1.0), 1e-7);
  CHECK_NEAR(at_one[1], std::cos(0.01), 1e-7);
  CHECK_NEAR(at_one[2], std::sin(1.0), 1e-7);
  CHECK_NEAR(at_one[3], std::sin(0.01), 1e-7);
  CHECK(::vidfab::test::throws([] { vidfab::dit::minimax_h3_timestep_sinusoid(0.5f, 3); }));
}

VIDFAB_TEST(full_adaln_timestep_mlp_contract_and_math) {
  const auto path = std::filesystem::temp_directory_path() / "vidfab_full_adaln_time.safetensors";
  vidfab::write_safetensors(
      path.string(),
      {{"time_embedder.proj_in.weight", {2, 4}, {1, 0, 0, 0, 0, 1, 0, 0}},
       {"time_embedder.proj_in.bias", {2}, {0.25f, -0.5f}},
       {"time_embedder.proj_out.weight", {3, 2}, {1, 0, 0, 1, 2, -1}},
       {"time_embedder.proj_out.bias", {3}, {0.1f, 0.2f, 0.3f}}});
  vidfab::SafeTensors checkpoint;
  checkpoint.open(path.string());
  FullAdaLNTimestepEmbedding embedding;
  embedding.load(checkpoint, 4, 2, 3);
  const std::vector<float> got = embedding.forward(0.0f);
  const float h0 = 1.25f / (1.0f + std::exp(-1.25f));
  const float h1 = 0.5f / (1.0f + std::exp(-0.5f));
  CHECK_NEAR(got[0], h0 + 0.1f, 1e-6);
  CHECK_NEAR(got[1], h1 + 0.2f, 1e-6);
  CHECK_NEAR(got[2], 2.0f * h0 - h1 + 0.3f, 1e-6);

  const std::vector<float> batch = embedding.forward({0.0f, 1.0f});
  CHECK(batch.size() == 6);
  CHECK_NEAR(batch[0], got[0], 0.0);
  CHECK_NEAR(batch[1], got[1], 0.0);
  CHECK_NEAR(batch[2], got[2], 0.0);

  bool rejected = false;
  try {
    FullAdaLNTimestepEmbedding bad;
    bad.load(checkpoint, 6, 2, 3);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  CHECK(rejected);
}
