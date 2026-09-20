#include "harness.h"
#include "slopfab/dit/checkpoint.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/text/tokenizer.h"
#include <filesystem>
#include <stdexcept>

namespace {
using namespace slopfab;
struct ModelFixture {
  std::filesystem::path path = std::filesystem::temp_directory_path() / "slopfab_model_contract.safetensors";
  SafeTensors file;
  ~ModelFixture() { file.close(); std::filesystem::remove(path); }
  void write(const std::string& metadata, bool gates = false) {
    file.close();
    std::vector<TensorWrite> tensors = {{"adaln_t_table", {1025,8}, std::vector<float>(1025*8)},
        {"blocks.0.adaln_proj.linear.weight", {18,8}, std::vector<float>(18*8)}};
    if (gates) tensors.push_back({"blocks.0.attn.to_gate_compress.weight", {8,8}, std::vector<float>(64)});
    write_safetensors(path.string(), tensors, {{"slopfab.model", metadata}});
    file.open(path.string());
  }
};
const std::string table = R"({"version":1,"family":"h3","modulation":"table","supports_references":true,"compressed_attention":false,"qkv_layout":"contiguous"})";
template<class Fn> bool fails(Fn fn) { try { fn(); return false; } catch (const std::exception&) { return true; } }
}
SLOPFAB_TEST(model_descriptor_explicit_identity_survives_rename) {
  ModelFixture fixture;
  fixture.write(table);
  auto model = dit::resolve_model_descriptor(fixture.file);
  CHECK(model.explicit_metadata && model.supports_references);
  CHECK(model.compatibility_architecture == dit::TransformerArchitecture::kRef2VAPrunedTable);
  fixture.file.close();
  auto renamed = fixture.path.parent_path() / "Viggle-Animate-FastH3-8Step-V2.safetensors";
  std::filesystem::rename(fixture.path, renamed);
  fixture.path = renamed;
  fixture.file.open(renamed.string());
  model = dit::resolve_model_descriptor(fixture.file);
  CHECK(model.compatibility_architecture == dit::TransformerArchitecture::kRef2VAPrunedTable);
  CHECK(!model.compressed_attention);
}
SLOPFAB_TEST(model_descriptor_rejects_unsupported_contracts) {
  ModelFixture fixture;
  for (const auto& bad : {std::string("{}"), std::string(R"({"version":2})"),
      table.substr(0, table.size()-1) + R"(,"unknown":true})"}) {
    fixture.write(bad);
    CHECK(fails([&] { dit::resolve_model_descriptor(fixture.file); }));
  }
  fixture.write(table, true);
  CHECK(fails([&] { dit::resolve_model_descriptor(fixture.file); }));
  CHECK(fails([] { dit::validate_adaln_table_config(16, 1025); }));
  CHECK(fails([] { dit::validate_adaln_table_config(8, 1024); }));
  dit::validate_adaln_table_config(8,1025);
}
SLOPFAB_TEST(model_descriptor_compression_is_independent_of_release) {
  ModelFixture fixture;
  fixture.write(R"({"version":1,"family":"h3","modulation":"table","supports_references":false,"compressed_attention":true,"qkv_layout":"interleaved"})", true);
  const auto model = dit::resolve_model_descriptor(fixture.file);
  CHECK(model.compressed_attention && model.qkv_interleaved);
  CHECK(model.compatibility_architecture == dit::TransformerArchitecture::kPrunedTable);
}
SLOPFAB_TEST(tokenizer_rejects_incompatible_algorithms) {
  text::Tokenizer tokenizer;
  for (const char* document : {
      R"({"model":{"type":"Unigram","vocab":{"a":0},"merges":[]}})",
      R"({"model":{"type":"BPE","dropout":0.2,"vocab":{"a":0},"merges":[]}})",
      R"({"normalizer":{"type":"Lowercase"},"model":{"vocab":{"a":0}}})",
      R"({"pre_tokenizer":{"type":"Whitespace"},"model":{"vocab":{"a":0}}})",
      R"({"decoder":{"type":"WordPiece"},"model":{"vocab":{"a":0}}})"})
    CHECK(fails([&] { tokenizer.load_json(document); }));
  tokenizer.load_json(R"({"model":{"type":"BPE","dropout":null,"vocab":{"a":0},"merges":[]}})");
  CHECK(tokenizer.loaded());
}
