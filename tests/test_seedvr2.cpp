#include "harness.h"
#include "slopfab/seedvr2.h"
#include <algorithm>
#include <stdexcept>
using namespace slopfab::seedvr2;

SLOPFAB_TEST(seedvr2_windows_partition) {
  for (auto dims : std::vector<std::vector<int>>{{1, 1, 1}, {2, 45, 80}, {9, 68, 120}, {33, 7, 11}})
    for (bool shifted : {false, true}) {
      int t = dims[0], h = dims[1], w = dims[2];
      std::vector<int> seen(t * h * w, 0);
      for (auto win : attention_windows(t, h, w, shifted))
        for (int z = win.t0; z < win.t1; ++z)
          for (int y = win.y0; y < win.y1; ++y)
            for (int x = win.x0; x < win.x1; ++x)
              ++seen[(z * h + y) * w + x];
      CHECK(std::all_of(seen.begin(), seen.end(), [](int n) {
        return n == 1;
      }));
    }
  auto regular = attention_windows(9, 45, 80, false);
  CHECK(regular.size() == 27);
  CHECK(regular[0].t1 == 3);
  CHECK(regular[0].y1 == 15);
  CHECK(regular[0].x1 == 27);
  auto shifted = attention_windows(9, 45, 80, true);
  CHECK(shifted[0].t1 == 1);
  CHECK(shifted[0].y1 == 7);
  CHECK(shifted[0].x1 == 13);
}

SLOPFAB_TEST(seedvr2_stream_exact_duration) {
  for (int segment : {1, 5, 9, 17})
    for (int count : {0, 1, 2, 4, 5, 6, 8, 9, 17, 360, 1001}) {
      Options o;
      o.width = o.height = 16;
      o.segment_frames = segment;
      int read = 0, written = 0, calls = 0;
      uint64_t frames = stream(
          o,
          [&](Frame& f) {
            if (read == count)
              return false;
            f.assign(16 * 16 * 3, float(read++));
            return true;
          },
          [&](const Frame& f) {
            CHECK(f[0] == float(written++));
          },
          [&](const std::vector<Frame>& f, uint64_t first) {
            ++calls;
            CHECK(f.size() <= size_t(segment));
            CHECK((f.size() - 1) % 4 == 0);
            CHECK(f[0][0] == float(first));
            return f;
          });
      CHECK(frames == uint64_t(count));
      CHECK(written == count);
      CHECK(calls <= (segment == 1 ? count : (count + segment - 2) / (segment - 1)));
    }
}

SLOPFAB_TEST(seedvr2_stream_blends_overlap) {
  Options o;
  o.width = o.height = 16;
  int read = 0;
  std::vector<float> out;
  stream(
      o,
      [&](Frame& f) {
        if (read++ == 9)
          return false;
        f.assign(768, 0);
        return true;
      },
      [&](const Frame& f) {
        out.push_back(f[0]);
      },
      [](const std::vector<Frame>& f, uint64_t first) {
        return std::vector<Frame>(f.size(), Frame(768, float(first)));
      });
  CHECK(out.size() == 9);
  CHECK(out[3] == 0);
  CHECK(out[4] == 2);
  CHECK(out[5] == 4);
  CHECK(out[8] == 4);
}

SLOPFAB_TEST(seedvr2_validation) {
  Options o;
  validate(o);
  for (int n : {0, 2, 4, 6, 130}) {
    o.segment_frames = n;
    bool failed = false;
    try {
      validate(o);
    } catch (const std::invalid_argument&) {
      failed = true;
    }
    CHECK(failed);
  }
  o.segment_frames = 5;
  o.vae_tile = 64;
  bool failed = false;
  try {
    validate(o);
  } catch (const std::invalid_argument&) {
    failed = true;
  }
  CHECK(failed);
  o.vae_tile = 256;
  o.width = o.height = 8192;
  o.segment_frames = 129;
  failed = false;
  try {
    validate(o);
  } catch (const std::invalid_argument&) {
    failed = true;
  }
  CHECK(failed);
}
