#include "harness.h"
#include "slopfab/dit/checkpoint.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/text/tokenizer.h"
#include "slopfab/text/encoder.h"
#include "slopfab/nf4.h"
#include "slopfab/pipeline.h"
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

SLOPFAB_TEST(model_descriptor_prism_release_is_ref2va) {
  const std::vector<TensorWrite> tensors = {
      {"adaln_t_table", {1025, 8}, std::vector<float>(1025 * 8)},
      {"blocks.0.adaln_proj.linear.weight", {18, 8}, std::vector<float>(18 * 8)}};
  for (const char* filename : {"Minimax-h3_Singularity_x_10Eros-Max_PRISM_w6a8.safetensors",
                               "Minimax-h3_Singularity_x_10Eros-Max_PRISM_int8.safetensors",
                               "Minimax-h3_Singularity_x_10Eros-Max_PRISM_fp16.safetensors",
                               "unrelated_w6a8.safetensors"}) {
    ModelFixture fixture;
    fixture.path = std::filesystem::temp_directory_path() / filename;
    write_safetensors(fixture.path.string(), tensors, {{"format", "pt"}, {"pruned", "curve_rank8"}});
    fixture.file.open(fixture.path.string());
    const auto model = dit::resolve_model_descriptor(fixture.file);
    const bool prism = std::string(filename).find("PRISM") != std::string::npos;
    CHECK(model.supports_references == prism);
    CHECK(model.compatibility_architecture == (prism ? dit::TransformerArchitecture::kRef2VAPrunedTable
                                                     : dit::TransformerArchitecture::kPrunedTable));
    CHECK(!model.compressed_attention && !model.qkv_interleaved);
    // Explicit capability metadata still takes precedence over release aliases.
    fixture.write(R"({"version":1,"family":"h3","modulation":"table","supports_references":false,"compressed_attention":false,"qkv_layout":"contiguous"})");
    CHECK(!dit::resolve_model_descriptor(fixture.file).supports_references);
  }
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

SLOPFAB_TEST(model_descriptor_resolves_omniref_without_legacy_aliases) {
  ModelFixture fixture;
  // Keep the output dimensions from the released OmniRef export, using a
  // small hidden dimension so this header-only contract needs no checkpoint.
  const std::vector<TensorWrite> tensors = {
      {"adaln_t_table", {1025, 8}, std::vector<float>(1025 * 8)},
      {"blocks.0.adaln_proj.linear.weight", {18, 8}, std::vector<float>(18 * 8)},
      {"blocks.0.attn.to_gate_compress.weight", {8, 8}, std::vector<float>(64)},
      {"final_layer.video_out.weight", {3072, 8}, std::vector<float>(3072 * 8)},
      {"final_layer.audio_out.weight", {1024, 8}, std::vector<float>(1024 * 8)}};
  for (int variant = 0; variant < 3; ++variant) {
    fixture.file.close();
    std::map<std::string, std::string> metadata;
    if (variant == 0)
      metadata = {{"source", "transformer_ref"},
                  {"config", R"({"transformer":{"pdd_steps":32}})"}};
    if (variant == 1)
      metadata = {{"model_id", "FastVideo/FastVideo-FastH3-8-Step-V2"}};
    if (variant == 2)
      metadata = {{"slopfab.model",
          R"({"version":1,"family":"h3","modulation":"table","supports_references":true,"compressed_attention":true,"qkv_layout":"contiguous"})"}};
    write_safetensors(fixture.path.string(), tensors, metadata);
    fixture.file.open(fixture.path.string());
    GenerateRequest request;
    request.transformer_path = fixture.path.string();
    if (variant == 2) {
      CHECK(fails([&] { dit::resolve_model_descriptor(fixture.file); }));
      continue;
    }
    const auto model = dit::resolve_model_descriptor(fixture.file);
    CHECK(model.pdd_steps == 32 && model.supports_references && model.compressed_attention);
    CHECK(model.compatibility_architecture == dit::TransformerArchitecture::kOmniRefPrunedTable);
    const auto plan = resolve_plan(request);
    CHECK(plan.num_model_evaluations() == 8 && !plan.fasth3_v2);
    CHECK(plan.video_sigma_shift == 12 && plan.audio_sigma_shift == 3);
    CHECK(plan.video_sigmas.front() == .999f && plan.audio_sigmas.front() == .999f);
    request.sampling.sampler = sampler::SamplerKind::kRenoise;
    CHECK(fails([&] { resolve_plan(request); }));
    request.sampling.sampler.reset();
    request.sampling.base_sigmas = std::vector<float>{1, .5f, 0};
    CHECK(fails([&] { resolve_plan(request); }));
    request.sampling.base_sigmas.reset();
    request.cache_threshold = .1f;
    CHECK(fails([&] { resolve_plan(request); }));
  }
  // Ordinary FastH3 V2 heads must retain their existing architecture/schedule.
  fixture.file.close();
  auto ordinary = tensors;
  ordinary[3] = {"final_layer.video_out.weight", {96, 8}, std::vector<float>(96 * 8)};
  ordinary[4] = {"final_layer.audio_out.weight", {32, 8}, std::vector<float>(32 * 8)};
  write_safetensors(fixture.path.string(), ordinary,
                    {{"model_id", "FastVideo/FastVideo-FastH3-8-Step-V2"}});
  fixture.file.open(fixture.path.string());
  CHECK(dit::detect_transformer_architecture(fixture.file) ==
        dit::TransformerArchitecture::kFastH3V2PrunedTable);
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
