#include "harness.h"
#include "slopfab/dit/checkpoint.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/text/tokenizer.h"
#include "slopfab/text/encoder.h"
#include "slopfab/nf4.h"
#include <fstream>
#include <filesystem>
#include <stdexcept>

namespace {
using namespace slopfab;

struct ModelFixture {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() / "slopfab_model_contract.safetensors";
  SafeTensors file;

  ~ModelFixture() {
    file.close();
    std::filesystem::remove(path);
  }

  void write(const std::string& metadata, bool gates = false) {
    file.close();
    std::vector<TensorWrite> tensors = {
        {"adaln_t_table", {1025, 8}, std::vector<float>(1025 * 8)},
        {"blocks.0.adaln_proj.linear.weight", {18, 8}, std::vector<float>(18 * 8)}};
    if (gates)
      tensors.push_back({"blocks.0.attn.to_gate_compress.weight", {8, 8}, std::vector<float>(64)});
    write_safetensors(path.string(), tensors, {{"slopfab.model", metadata}});
    file.open(path.string());
  }
};

const std::string table =
    R"({"version":1,"family":"h3","modulation":"table","supports_references":true,"compressed_attention":false,"qkv_layout":"contiguous"})";

template <class Fn> bool fails(Fn fn) {
  try {
    fn();
    return false;
  } catch (const std::exception&) {
    return true;
  }
}
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
                          table.substr(0, table.size() - 1) + R"(,"unknown":true})"}) {
    fixture.write(bad);
    CHECK(fails([&] {
      dit::resolve_model_descriptor(fixture.file);
    }));
  }
  fixture.write(table, true);
  CHECK(fails([&] {
    dit::resolve_model_descriptor(fixture.file);
  }));
  CHECK(fails([] {
    dit::validate_adaln_table_config(16, 1025);
  }));
  CHECK(fails([] {
    dit::validate_adaln_table_config(8, 1024);
  }));
  dit::validate_adaln_table_config(8, 1025);
}

SLOPFAB_TEST(model_descriptor_compression_is_independent_of_release) {
  ModelFixture fixture;
  fixture.write(
      R"({"version":1,"family":"h3","modulation":"table","supports_references":false,"compressed_attention":true,"qkv_layout":"interleaved"})",
      true);
  const auto model = dit::resolve_model_descriptor(fixture.file);
  CHECK(model.compressed_attention && model.qkv_interleaved);
  CHECK(model.compatibility_architecture == dit::TransformerArchitecture::kPrunedTable);
}

SLOPFAB_TEST(tokenizer_rejects_incompatible_algorithms) {
  text::Tokenizer tokenizer;
  for (const char* document :
       {R"({"model":{"type":"Unigram","vocab":{"a":0},"merges":[]}})",
        R"({"model":{"type":"BPE","dropout":0.2,"vocab":{"a":0},"merges":[]}})",
        R"({"normalizer":{"type":"Lowercase"},"model":{"vocab":{"a":0}}})",
        R"({"pre_tokenizer":{"type":"Whitespace"},"model":{"vocab":{"a":0}}})",
        R"({"decoder":{"type":"WordPiece"},"model":{"vocab":{"a":0}}})"})
    CHECK(fails([&] {
      tokenizer.load_json(document);
    }));
  tokenizer.load_json(R"({"model":{"type":"BPE","dropout":null,"vocab":{"a":0},"merges":[]}})");
  CHECK(tokenizer.loaded());
}

SLOPFAB_TEST(conditioner_descriptor_checks_semantics_and_fingerprint) {
  ModelFixture fixture;
  const std::string contract =
      R"({"version":1,"family":"qwen3_vl","tokenizer":"qwen_byte_bpe","output_width":5120,"output_layer":49,"final_normalization":false,"vision":false})";
  write_safetensors(fixture.path.string(), {{"stub", {1}, {0}}},
                    {{"slopfab.conditioner", contract}});
  fixture.file.open(fixture.path.string());
  const auto descriptor = text::resolve_conditioner_descriptor(fixture.file);
  CHECK(descriptor.explicit_metadata && !descriptor.vision);
  CHECK(descriptor.output_width == 5120 && descriptor.output_layer == 49);
  text::EncoderConfig smaller;
  smaller.hidden_size = 4096;
  CHECK(fails([&] {
    text::resolve_conditioner_descriptor(fixture.file, smaller);
  }));
  auto changed = descriptor;
  changed.output_layer = 47;
  CHECK(descriptor.fingerprint() != changed.fingerprint());
  fixture.file.close();
  auto normalized = contract;
  normalized.replace(normalized.find("false"), 5, "true");
  write_safetensors(fixture.path.string(), {{"stub", {1}, {0}}},
                    {{"slopfab.conditioner", normalized}});
  fixture.file.open(fixture.path.string());
  CHECK(fails([&] {
    text::resolve_conditioner_descriptor(fixture.file);
  }));
}

SLOPFAB_TEST(nf4_shared_parser_preserves_transformer_source_contract) {
  ModelFixture fixture;
  auto write = [&](const std::string& dtype, const std::string& blocksize) {
    fixture.file.close();
    const std::string payload = "{\"quant_type\":\"nf4\",\"dtype\":\"" + dtype +
                                "\",\"nested_dtype\":\"float32\",\"blocksize\":" + blocksize +
                                ",\"nested_blocksize\":256,\"nested_offset\":0,\"shape\":[8,8]}";
    std::string header =
        "{\"linear.weight.quant_state.bitsandbytes__nf4\":{\"dtype\":\"U8\",\"shape\":[" +
        std::to_string(payload.size()) + "],\"data_offsets\":[0," + std::to_string(payload.size()) +
        "]}}";
    while (header.size() % 8)
      header += ' ';
    std::ofstream out(fixture.path, std::ios::binary);
    const uint64_t bytes = header.size();
    out.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
    out << header << payload;
    out.close();
    fixture.file.open(fixture.path.string());
  };
  write("bfloat16", "64");
  CHECK(read_nf4_state(fixture.file, "linear.weight", "test", true).block_size == 64);
  write("float16", "64");
  CHECK(read_nf4_state(fixture.file, "linear.weight", "test").source_dtype == "float16");
  CHECK(fails([&] {
    read_nf4_state(fixture.file, "linear.weight", "test", true);
  }));
  write("bfloat16", "64.5");
  CHECK(fails([&] {
    read_nf4_state(fixture.file, "linear.weight", "test");
  }));
}
