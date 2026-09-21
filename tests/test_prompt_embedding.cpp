#include "harness.h"
#include "slopfab/text/prompt_embedding.h"
#include "slopfab/safetensors_write.h"
#include <filesystem>
#include <fstream>
#include <limits>

namespace {
struct EmbeddingFixture {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() / "slopfab_fixed_prompt.safetensors";

  ~EmbeddingFixture() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }

  void write(const std::vector<int32_t>& tags, float value = 0.25f) {
    const std::vector<float> data(3 * 5120, value);
    const size_t bytes = data.size() * sizeof(float);
    std::string header =
        "{\"prompt_embedding\":{\"dtype\":\"F32\",\"shape\":[3,5120],\"data_offsets\":[0," +
        std::to_string(bytes) + "]},\"text_token_tags\":{\"dtype\":\"I32\",\"shape\":[" +
        std::to_string(tags.size()) + "],\"data_offsets\":[" + std::to_string(bytes) + "," +
        std::to_string(bytes + tags.size() * sizeof(int32_t)) + "]}}";
    while (header.size() % 8)
      header += ' ';
    const uint64_t length = header.size();
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&length), 8);
    out.write(header.data(), header.size());
    out.write(reinterpret_cast<const char*>(data.data()), bytes);
    out.write(reinterpret_cast<const char*>(tags.data()), tags.size() * sizeof(int32_t));
  }

  bool rejected() {
    try {
      (void)slopfab::text::read_prompt_embedding(path.string(), true);
    } catch (const std::exception&) {
      return true;
    }
    return false;
  }
};
}

SLOPFAB_TEST(fixed_prompt_preserves_values_and_tags) {
  EmbeddingFixture f;
  f.write({1, 0, 2});
  const auto prompt = slopfab::text::read_prompt_embedding(f.path.string(), true);
  CHECK(prompt.num_tokens == 3);
  CHECK(prompt.hidden_size == 5120);
  CHECK(prompt.data == std::vector<float>(3 * 5120, .25f));
  CHECK(prompt.modality_tags == (std::vector<int32_t>{1, 0, 2}));
  f.write({1, 1});
  CHECK(f.rejected());
  f.write({1, -1, 2});
  CHECK(f.rejected());
  f.write({1, 3, 2});
  CHECK(f.rejected());
  f.write({1, 1, 1}, std::numeric_limits<float>::quiet_NaN());
  CHECK(f.rejected());
}

SLOPFAB_TEST(fixed_prompt_requires_tags_for_references) {
  EmbeddingFixture f;
  slopfab::write_safetensors(f.path.string(),
                             {{"prompt_embedding", {3, 5120}, std::vector<float>(3 * 5120)}});
  CHECK(f.rejected());
  const auto legacy = slopfab::text::read_prompt_embedding(f.path.string());
  CHECK(legacy.modality_tags == std::vector<int32_t>(3, 1));
  slopfab::write_safetensors(f.path.string(),
                             {{"prompt_embedding", {3, 32}, std::vector<float>(96)}});
  CHECK(f.rejected());
}
