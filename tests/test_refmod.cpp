#include "harness.h"
#include "refmod_fixture.h"
#include "slopfab/refmod.h"
#include "slopfab/pipeline.h"
#include <cmath>
#include <limits>

namespace {
using namespace slopfab;

template <class F> bool rejects(F f) {
  try {
    f();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}
}

SLOPFAB_TEST(refmod_visual_packing_and_dtype) {
  RefModFixture f;
  std::vector<float> z(24 * 2 * 4 * 6);
  for (size_t i = 0; i < z.size(); ++i)
    z[i] = float(i) / 16;
  for (auto dtype : {DType::kF32, DType::kF16, DType::kBF16}) {
    f.write(R"({"kind":"video","latent_h":4,"latent_w":6,"latent_t":2})", {1, 24, 2, 4, 6}, z,
            dtype);
    const auto mod = RefMod::load(f.path.string());
    CHECK(mod->token_count() == 12);
    const auto rows = mod->rows();
    for (int t = 0; t < 2; ++t)
      for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 6; ++x)
          for (int c = 0; c < 24; ++c) {
            float expected = z[((c * 2 + t) * 4 + y) * 6 + x];
            if (dtype == DType::kBF16)
              expected = bf16_to_f32(f32_to_bf16(expected));
            const int row = (t * 2 + y / 2) * 3 + x / 2;
            CHECK_NEAR(rows[row * 96 + c * 4 + (y % 2) * 2 + x % 2], expected, 0);
          }
  }
}

SLOPFAB_TEST(refmod_strength_blurs_detail_preserving_frame_and_channel_means) {
  RefModFixture f;
  std::vector<float> z(24 * 2 * 4 * 4);
  for (size_t i = 0; i < z.size(); ++i)
    z[i] = float(i);
  f.write(R"({"kind":"video"})", {1, 24, 2, 4, 4}, z);
  const auto mod = RefMod::load(f.path.string());
  const auto rows = mod->rows(.25f);
  for (int t = 0; t < 2; ++t)
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x)
        for (int c = 0; c < 24; ++c) {
          const int offset = (c * 2 + t) * 16;
          const float expected = .25f * z[offset + y * 4 + x] + .75f * (offset + 7.5f);
          CHECK_NEAR(rows[((t * 2 + y / 2) * 2 + x / 2) * 96 + c * 4 + y % 2 * 2 + x % 2], expected,
                     0);
        }
  CHECK(mod->rows(0).empty());
  CHECK(rejects([&] {
    mod->rows(-.1f);
  }));
  CHECK(rejects([&] {
    mod->rows(NAN);
  }));
}

SLOPFAB_TEST(refmod_audio_packing_and_linear_blur) {
  RefModFixture f;
  std::vector<float> z(32 * 2 * 16);
  for (size_t i = 0; i < z.size(); ++i)
    z[i] = float(i);
  f.write(R"({"kind":"audio","latent_t":16,"sample_rate":32000})", {1, 32, 2, 16}, z);
  const auto mod = RefMod::load(f.path.string());
  CHECK(mod->token_count() == 32);
  auto rows = mod->rows(.5f);
  for (int ch = 0; ch < 2; ++ch)
    for (int t = 0; t < 16; ++t)
      for (int c = 0; c < 32; ++c) {
        const float origin = float((c * 2 + ch) * 16);
        const float blurred = t < 4 ? 3.5f : t > 11 ? 11.5f : float(t);
        CHECK_NEAR(rows[(ch * 16 + t) * 32 + c], origin + .5f * (t + blurred), 1e-4);
      }
}

SLOPFAB_TEST(refmod_blur_uneven_pool_cells_and_bilinear_interior) {
  RefModFixture f;
  std::vector<float> z(24 * 18 * 26);
  for (int c = 0; c < 24; ++c)
    for (int y = 0; y < 18; ++y)
      for (int x = 0; x < 26; ++x)
        z[(c * 18 + y) * 26 + x] = float(y * y + 2 * x);
  f.write(R"({"kind":"image"})", {1, 24, 1, 18, 26}, z);
  const auto rows = RefMod::load(f.path.string())->rows(.5f);
  const auto at = [&](int y, int x) {
    return rows[(y / 2 * 13 + x / 2) * 96 + y % 2 * 2 + x % 2];
  };
  // Analytic pool means: Y^2 -> {68/3,527/3}; 2X -> {8,25,42}.
  // The middle horizontal pooling cell overlaps both neighbours (26/3).
  CHECK_NEAR(at(0, 0), (68.0 / 3 + 8) / 2, 1e-5);
  CHECK_NEAR(at(17, 25), (339 + 527.0 / 3 + 42) / 2, 3e-5);
  CHECK_NEAR(at(8, 12), (88 + 68.0 / 3 + 153 * 4.0 / 9 + 8 + 17 * 49.0 / 52) / 2, 2e-5);
  CHECK_NEAR(at(9, 13), (107 + 68.0 / 3 + 153 * 5.0 / 9 + 25 + 17 * 3.0 / 52) / 2, 2e-5);
}

SLOPFAB_TEST(refmod_metadata_validation_and_owned_snapshot) {
  RefModFixture f;
  f.write();
  auto loaded = RefMod::load(f.path.string());
  auto rows = loaded->rows();
  for (const char* meta : {R"({"kind":"bundle"})", R"({"kind":"image","latent_h":8})",
                           R"({"kind":"image","latent_t":1.5})", "[]", "{"}) {
    f.write(meta);
    CHECK(rejects([&] {
      RefMod::load(f.path.string());
    }));
  }
  f.write("", {1, 24, 1, 4, 4});
  CHECK(rejects([&] {
    RefMod::load(f.path.string());
  }));
  auto sidecar = f.path;
  sidecar.replace_extension(".json");
  {
    std::ofstream out(sidecar);
    out << R"({"kind":"image","name":"legacy"})";
  }
  CHECK(RefMod::load(f.path.string())->name() == "legacy");
  f.write(); // Embedded metadata takes precedence over a stale sidecar.
  CHECK(RefMod::load(f.path.string())->name() == "test");
  f.write(R"({"kind":"image"})", {1, 24, 1, 3, 4});
  CHECK(rejects([&] {
    RefMod::load(f.path.string());
  }));
  f.write(R"({"kind":"image"})", {1, 24, 2, 4, 4});
  CHECK(rejects([&] {
    RefMod::load(f.path.string());
  }));
  f.write(R"({"kind":"audio"})", {1, 24, 1, 4, 4});
  CHECK(rejects([&] {
    RefMod::load(f.path.string());
  }));
  f.write(R"({"kind":"image"})", {1, 24, 1, 4, 4}, std::vector<float>(384, NAN));
  CHECK(rejects([&] {
    RefMod::load(f.path.string());
  }));
  std::filesystem::remove(f.path);
  CHECK(loaded->rows() == rows);
}

SLOPFAB_TEST(refmod_mixed_conditions_copies_positions_and_seed) {
  RefModFixture visual, audio;
  visual.write();
  audio.write(R"({"kind":"audio"})", {1, 32, 2, 3});
  const std::vector<RefModReference> refs = {{RefMod::load(visual.path.string()), 1, 2},
                                             {RefMod::load(audio.path.string()), 1, 1}};
  std::vector<dit::ReferenceGeometry> geometry;
  std::vector<float> v, a;
  append_refmod_conditions(refs, 42, geometry, v, a);
  CHECK(geometry.size() == 3 && v.size() == 8 * 96 && a.size() == 6 * 32);
  CHECK(a == refs[1].mod->rows());
  auto packed = dit::build_ref2va_packed_sequence({dit::kTagText}, geometry, 1, 4, 4, 2);
  CHECK(packed.layout.num_condition_video == 8);
  CHECK(packed.layout.num_condition_audio == 6);
  CHECK_NEAR(packed.position_ids[3], 1, 0);      // first image after text
  CHECK_NEAR(packed.position_ids[5 * 3], 2, 0);  // second copy
  CHECK_NEAR(packed.position_ids[9 * 3], 3, 0);  // audio reference
  CHECK_NEAR(packed.position_ids[15 * 3], 6, 0); // target audio after reference
  std::vector<dit::ReferenceGeometry> g2;
  std::vector<float> v2, a2;
  append_refmod_conditions(refs, 42, g2, v2, a2);
  CHECK(v == v2 && a == a2);
  g2.clear();
  v2.clear();
  a2.clear();
  append_refmod_conditions(refs, 43, g2, v2, a2);
  CHECK(v != v2 && a == a2);
}

SLOPFAB_TEST(refmod_plan_disabled_and_prompt_cache) {
  RefModFixture f;
  f.write();
  GenerateRequest request;
  request.prompt = "a person walking";
  const auto base = resolve_plan(request);
  const auto key = conditioning_cache_key(request);
  request.refmods.push_back({RefMod::load(f.path.string()), 0, 2});
  CHECK(!request.has_references());
  CHECK(resolve_plan(request).sequence_length_without_text() ==
        base.sequence_length_without_text());
  request.refmods[0].strength = 1;
  CHECK(request.has_references() && !request.has_native_references());
  CHECK(resolve_plan(request).sequence_length_without_text() ==
        base.sequence_length_without_text() + 8);
  CHECK(conditioning_cache_key(request) == key);
  CHECK(describe_plan(request, resolve_plan(request)).find("copies 2, tokens 8") !=
        std::string::npos);
  request.refmods[0].copies = 11;
  CHECK(rejects([&] {
    resolve_plan(request);
  }));
  request.refmods[0] = {nullptr, 1, 1};
  CHECK(rejects([&] {
    resolve_plan(request);
  }));
}
