#include "slopfab/music.h"
#include "slopfab/text/tokenizer.h"
#include "slopfab/safetensors.h"
#include "slopfab/json.h"

namespace slopfab::music {
void load_yue_tokenizer(text::Tokenizer& tokenizer, const SafeTensors& checkpoint) {
  auto tokenizer_blob = checkpoint.at("text_encoders.yue2_tokenizer_json");
  // The bundle exporter double-escaped its Qwen pattern. Restore the upstream
  // pattern before validating the tokenizer; vocab and merges stay untouched.
  std::string source(static_cast<const char*>(tokenizer_blob.data), tokenizer_blob.nbytes);
  auto root = json::parse(source);
  auto object = root.as_object();
  auto pre = object.at("pre_tokenizer").as_object();
  auto seq = pre.at("pretokenizers").as_array();
  auto split = seq[0].as_object();
  auto pat = split.at("pattern").as_object();
  auto regex = pat.at("Regex").as_string();
  for (size_t at = 0; (at = regex.find("\\\\", at)) != std::string::npos; ++at)
    regex.erase(at, 1);
  pat["Regex"] = json::Value(regex);
  split["pattern"] = json::Value(pat);
  seq[0] = json::Value(split);
  pre["pretokenizers"] = json::Value(seq);
  object["pre_tokenizer"] = json::Value(pre);
  tokenizer.load_json(json::stringify(json::Value(object)));
}
}
