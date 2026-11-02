#include "harness.h"
#include "slopfab/text/clip_projection.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/pipeline.h"
#include <filesystem>
#include <chrono>
#include <limits>
#include <stdexcept>

namespace {
using namespace slopfab;

struct Fixture {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("slopfab-clipproj-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
       ".safetensors");

  ~Fixture() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }

  std::vector<TensorWrite> tensors = {{"mean_in", {2}, {1, 2}},
                                      {"std_in", {2}, {2, 4}},
                                      {"mean_out", {3}, {10, 20, 30}},
                                      {"std_out", {3}, {2, 3, 4}}};

  void save(std::string tap = "24") {
    write_safetensors(path.u8string(), tensors, {{"tap", tap}});
  }
};

text::PromptEmbedding hidden() {
  text::PromptEmbedding h;
  h.num_tokens = 2;
  h.hidden_size = 2;
  h.data = {3, 6, 5, 14};
  h.modality_tags = {1, 1};
  return h;
}

template <class F> bool rejects(F f) {
  try {
    f();
    return false;
  } catch (const std::exception&) {
    return true;
  }
}
}

SLOPFAB_TEST(clipproj_ridge_orientation_standardization_and_sink) {
  Fixture f;
  f.tensors.push_back({"W", {2, 3}, {1, 2, 3, 4, 5, 6}, DType::kF16});
  f.tensors.push_back({"sink_out", {3}, {7, 8, 9}, DType::kF16});
  f.save();
  slopfab::text::ClipProjection projection(f.path.u8string());
  CHECK(projection.input_width() == 2 && projection.output_width() == 3 && projection.tap() == 24);
  auto y = projection.apply(hidden());
  CHECK_CLOSE((std::vector<float>{7, 8, 9, 38, 77, 126}), y.data, 0, "ClipProj ridge and sink");
  CHECK(y.modality_tags == hidden().modality_tags);
  auto invalid = hidden();
  invalid.hidden_size = 3;
  CHECK(rejects([&] {
    projection.apply(invalid);
  }));
  CHECK(rejects([&] {
    projection.apply(hidden(), [] {
      throw std::runtime_error("cancelled");
    });
  }));
}

SLOPFAB_TEST(clipproj_residual_only_and_ridge_plus_mlp) {
  Fixture f;
  f.tensors.push_back({"mlp.0.weight", {2, 2}, {1, 0, 0, -1}, DType::kF16});
  f.tensors.push_back({"mlp.0.bias", {2}, {0, 0}, DType::kF16});
  f.tensors.push_back({"mlp.2.weight", {3, 2}, {1, 0, 0, 1, 1, 1}, DType::kF16});
  f.tensors.push_back({"mlp.2.bias", {3}, {0, 0, 1}, DType::kF16});
  f.save();
  auto y = slopfab::text::ClipProjection(f.path.u8string()).apply(hidden());
  // Independently calculated fp16 GELU(1), GELU(-1), including linear-output rounding.
  CHECK_NEAR(y.data[0], 11.6826171875, 1e-6);
  CHECK_NEAR(y.data[1], 19.52392578125, 1e-6);
  CHECK_NEAR(y.data[2], 36.73046875, 1e-6);
  auto ridge = f.tensors;
  f.tensors.push_back({"W", {2, 3}, {1, 2, 3, 4, 5, 6}});
  f.save();
  auto both = slopfab::text::ClipProjection(f.path.u8string()).apply(hidden());
  CHECK_NEAR(both.data[0] - y.data[0], 10, 1e-6);
  CHECK_NEAR(both.data[1] - y.data[1], 21, 1e-6);
  CHECK_NEAR(both.data[2] - y.data[2], 36, 1e-6);
}

SLOPFAB_TEST(clipproj_rejects_incomplete_or_incompatible_files) {
  Fixture f;
  f.save();
  CHECK(rejects([&] {
    slopfab::text::ClipProjection p(f.path.u8string());
  }));
  f.tensors.push_back({"W", {2, 3}, {1, 2, 3, 4, 5, 6}});
  f.save("36");
  CHECK(rejects([&] {
    slopfab::text::ClipProjection p(f.path.u8string());
  }));
  f.save("24junk");
  CHECK(rejects([&] {
    slopfab::text::ClipProjection p(f.path.u8string());
  }));
  f.tensors[1].data[0] = 0;
  f.save();
  CHECK(rejects([&] {
    slopfab::text::ClipProjection p(f.path.u8string());
  }));
  f.tensors[1].data[0] = 2;
  f.tensors.push_back({"mlp.0.weight", {3, 3}, std::vector<float>(9)});
  f.save();
  CHECK(rejects([&] {
    slopfab::text::ClipProjection p(f.path.u8string());
  }));
}

SLOPFAB_TEST(clipproj_changes_conditioning_cache_and_rejects_other_families) {
  slopfab::GenerateRequest r;
  r.prompt = "red ball";
  const auto original = slopfab::conditioning_cache_key(r);
  r.clip_projection_path = "projection-a.safetensors";
  const auto a = slopfab::conditioning_cache_key(r);
  r.clip_projection_path = "projection-b.safetensors";
  CHECK(a != original && a != slopfab::conditioning_cache_key(r));
  r.model_family = "krea2";
  CHECK(rejects([&] {
    slopfab::resolve_plan(r);
  }));
  r.model_family = "h3";
  r.reference_image_paths = {"photo.png"};
  CHECK(rejects([&] {
    slopfab::resolve_plan(r);
  }));
}
