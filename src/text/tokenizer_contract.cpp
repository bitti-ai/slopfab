#include "tokenizer_contract.h"
#include "slopfab/json.h"
#include <stdexcept>

namespace slopfab::text::detail {
namespace {
void require(bool ok, const std::string& name) {
  if (!ok)
    throw std::runtime_error("tokenizer: unsupported Qwen byte-BPE contract: " + name);
}

const json::Value& field(const json::Value& value, const char* name) {
  const auto* result = value.find(name);
  if (!result)
    throw std::runtime_error(std::string("tokenizer: missing contract field ") + name);
  return *result;
}

void byte_level(const json::Value& value, const std::string& name) {
  require(field(value, "type").as_string() == "ByteLevel", name);
  for (const char* key : {"add_prefix_space", "use_regex", "trim_offsets"})
    if (const auto* v = value.find(key))
      require(!v->as_bool(), name + "." + key);
}
}

void validate_tokenizer_component(const std::string& name, std::string_view text) {
  const auto value = json::parse(text);
  if (value.is_null())
    return; // Historical minimal vocabulary fixtures.
  if (name == "normalizer") {
    require(field(value, "type").as_string() == "NFC", name);
    return;
  }
  if (name == "decoder" || name == "post_processor") {
    byte_level(value, name);
    return;
  }
  require(field(value, "type").as_string() == "Sequence", name);
  const auto& sequence = field(value, "pretokenizers").as_array();
  require(sequence.size() == 2, name + ".pretokenizers");
  const auto& split = sequence[0];
  require(field(split, "type").as_string() == "Split", name + ".Split");
  require(field(split, "behavior").as_string() == "Isolated", name + ".behavior");
  require(!field(split, "invert").as_bool(), name + ".invert");
  constexpr const char* pattern =
      R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
  require(field(field(split, "pattern"), "Regex").as_string() == pattern, name + ".pattern");
  byte_level(sequence[1], name + ".ByteLevel");
}

void validate_tokenizer_model_field(const std::string& name, std::string_view text) {
  const auto value = json::parse(text);
  if (name == "type")
    require(value.as_string() == "BPE", "model.type");
  else if (name == "dropout" || name == "unk_token")
    require(value.is_null(), "model." + name);
  else if (name == "continuing_subword_prefix" || name == "end_of_word_suffix")
    require(value.is_null() || value.as_string().empty(), "model." + name);
  else if (name == "byte_fallback" || name == "fuse_unk" || name == "ignore_merges")
    require(!value.as_bool(), "model." + name);
}
}
