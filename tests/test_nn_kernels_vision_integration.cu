#include "detail/nn_kernels_fixture.h"

SLOPFAB_TEST_CATEGORY(qwen_vision_encode_reuses_arena_across_images, "integration") {
  // Five levels, because the natural place to run the exe is
  // build/Release, which is three below the tree root, and a ctest run from
  // build/ is two. A search that stops short resolves nothing, and a skip that
  // only printf's is indistinguishable from a pass -- this campaign has already
  // been bitten once by a fixture-dependent case that quietly skipped for
  // months. SKIP_MISSING_FIXTURE reports the unavailable coverage, and
  // the resolved path is printed so "found" is never taken on trust either.
  std::string path;
  for (const char* prefix : {"", "../", "../../", "../../../", "../../../../"}) {
    const std::string p =
        std::string(prefix) + "weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors";
    if (std::filesystem::exists(p)) {
      path = p;
      break;
    }
  }
  if (path.empty()) {
    SKIP_MISSING_FIXTURE(
        "qwen vision: no text encoder checkpoint under any of ./ .. ../.. ../../.. "
        "../../../.. -- the whole-tower carve is NOT being exercised");
    return;
  }
  std::printf("  qwen vision: using %s\n", path.c_str());

  std::vector<uint8_t> rgb(256 * 256 * 3);
  for (int y = 0; y < 256; ++y) {
    for (int x = 0; x < 256; ++x) {
      const size_t i = (size_t(y) * 256 + x) * 3;
      rgb[i] = uint8_t((x * 17 + y * 3) & 255);
      rgb[i + 1] = uint8_t((x * 5 + y * 11) & 255);
      rgb[i + 2] = uint8_t((x ^ y) & 255);
    }
  }
  const auto pixels = slopfab::text::qwen3vl_patchify_resized_rgb(rgb, 256, 256);

  slopfab::SafeTensors checkpoint;
  checkpoint.open(path);
  slopfab::text::QwenVisionEncoder encoder;
  encoder.load(checkpoint);
  const slopfab::text::QwenVisionEmbedding out = encoder.encode({pixels, pixels});

  CHECK(out.tokens == 128);
  CHECK(out.hidden == 5120);
  CHECK(out.main.size() == 128ull * 5120);
  const size_t half = out.main.size() / 2;
  size_t bad = 0;
  for (size_t i = 0; i < half; ++i)
    bad += out.main[i] != out.main[half + i];
  CHECK_MSG(bad == 0, "second image's main embedding differs in %zu of %zu bf16", bad, half);
  for (int d = 0; d < 3; ++d) {
    CHECK(out.deepstack[d].size() == out.main.size());
    size_t dbad = 0;
    for (size_t i = 0; i < half; ++i)
      dbad += out.deepstack[d][i] != out.deepstack[d][half + i];
    CHECK_MSG(dbad == 0, "deepstack %d differs between identical images in %zu of %zu bf16", d,
              dbad, half);
  }

  // Every value finite, and the rms printed so a future carve bug that shifts
  // an activation shows up as a number rather than as silence. These are the
  // same four figures qwenvisionprobe reports.
  size_t nonfinite = 0;
  auto rms = [&](const std::vector<uint16_t>& v) {
    long double acc = 0;
    for (uint16_t b : v) {
      const float x = slopfab::bf16_to_f32(b);
      nonfinite += !std::isfinite(x);
      acc += static_cast<long double>(x) * x;
    }
    return std::sqrt(double(acc / v.size()));
  };
  std::printf("  qwen vision main rms %.9g  deepstack %.9g %.9g %.9g\n", rms(out.main),
              rms(out.deepstack[0]), rms(out.deepstack[1]), rms(out.deepstack[2]));
  CHECK_MSG(nonfinite == 0, "qwen vision produced %zu nonfinite values", nonfinite);
}
