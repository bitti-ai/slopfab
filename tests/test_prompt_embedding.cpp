#include "harness.h"
#include "slopfab/text/prompt_embedding.h"
#include "slopfab/safetensors_write.h"
#include <filesystem>
#include <fstream>
#include <limits>
#include "slopfab/json.h"

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

SLOPFAB_TEST(fixed_prompt_export_roundtrip_and_transactional_failure) {
  using namespace slopfab;
  EmbeddingFixture f;
  text::PromptEmbedding prompt;
  prompt.num_tokens = 3;
  prompt.data.resize(3 * 5120);
  for (size_t i = 0; i < prompt.data.size(); ++i)
    prompt.data[i] = float(int(i % 57) - 20) / 8;
  prompt.modality_tags = {1, 0, 2};
  const std::string source = "A \"quoted\" prompt\nwith a backslash \\";
  text::write_prompt_embedding(f.path.string(), prompt, {{"prompt", source}});
  CHECK(text::read_prompt_embedding(f.path.string(), true).data == prompt.data);
  CHECK(text::read_prompt_embedding(f.path.string(), true).modality_tags == prompt.modality_tags);
  {
    SafeTensors file;
    file.open(f.path.string());
    CHECK(file.metadata().at("prompt") == source);
    CHECK(file.at("text_token_tags").dtype == DType::kI32);
  }
  const auto original = prompt.data;
  for (int bad = 0; bad < 4; ++bad) {
    auto invalid = prompt;
    if (bad == 0) invalid.data[0] = std::numeric_limits<float>::infinity();
    if (bad == 1) invalid.modality_tags = {1, 1};
    if (bad == 2) invalid.modality_tags[0] = 5;
    if (bad == 3) invalid.hidden_size = 32;
    bool rejected = false;
    try { text::write_prompt_embedding(f.path.string(), invalid); }
    catch (const std::exception&) { rejected = true; }
    CHECK(rejected);
    CHECK(text::read_prompt_embedding(f.path.string(), true).data == original);
  }
  prompt.modality_tags.clear();
  text::write_prompt_embedding(f.path.string(), prompt);
  CHECK(text::read_prompt_embedding(f.path.string(), true).modality_tags ==
        std::vector<int32_t>(3, 1));
}

SLOPFAB_TEST(conditioning_metadata_json_roundtrip) {
  using namespace slopfab;
  const auto value = json::parse(R"({"text":"a\n\"b\\c","nested":[true,false,null,1.25,{"x":"voice"}]})");
  const auto loaded = json::parse(json::stringify(value));
  CHECK(loaded.find("text")->as_string() == "a\n\"b\\c");
  const auto& nested = loaded.find("nested")->as_array();
  CHECK(nested[0].as_bool());
  CHECK(!nested[1].as_bool());
  CHECK(nested[2].is_null());
  CHECK(nested[3].as_number() == 1.25);
  CHECK(nested[4].find("x")->as_string() == "voice");
}
