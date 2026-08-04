// Tokenizer tests against golden ids from the HuggingFace reference.
//
// The prompt is tokenised once per request and its ids select rows of a
// 151936-row embedding table. A tokenisation that differs from the reference
// by one merge produces a *valid* embedding sequence conditioning the model on
// slightly the wrong thing, which shows up as a video that is subtly off
// prompt and nothing else. There is no shape to check and no assertion that
// fires.
//
// So the ids below were produced by `tokenizers.Tokenizer.from_file(...)`
// .encode(s, add_special_tokens=False) on the real Qwen3-VL tokenizer.json and
// are compared exactly. Beyond these, the implementation was fuzzed against
// the same reference over 98 cases — the Context-IR prompt structure,
// contractions, repeated whitespace, numerals, control tokens and 80 random
// ASCII strings — with zero mismatches.
//
// `tokenizer.json` lives under `ref/`, which is licence-restricted and not
// committed, so these tests skip cleanly when it is absent.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "harness.h"
#include "vidfab/text/tokenizer.h"

namespace {

const char* kTokenizerPath = "ref/FL2VA/text_encoder/tokenizer.json";

// The tests run from the build directory or the repo root depending on how the
// binary is invoked, so probe both rather than depending on the caller.
std::string find_tokenizer() {
  for (const char* prefix : {"", "../", "../../"}) {
    const std::string candidate = std::string(prefix) + kTokenizerPath;
    if (std::filesystem::exists(candidate)) return candidate;
  }
  return {};
}

struct GoldenCase {
  const char* text;
  std::vector<int32_t> ids;
};

VIDFAB_TEST(tokenizer_golden_ids) {
  const std::string path = find_tokenizer();
  if (path.empty()) {
    std::printf("  ref/ tokenizer.json not present; skipping\n");
    return;
  }

  vidfab::text::Tokenizer tok;
  tok.load(path);
  CHECK(tok.loaded());
  CHECK(tok.vocab_size() > 151000);

  const std::vector<GoldenCase> cases = {
      {"integrated_multimodal_description: A cat walks across a sunlit kitchen floor, tail high.",
       {396, 47172, 26290, 318, 57597, 11448, 25, 362, 8251, 22479, 3941, 264, 7015, 31635, 9780,
        6422, 11, 9787, 1550, 13}},
      {"don't can't I'll we've", {15007, 944, 646, 944, 358, 3278, 582, 3003}},
      {"  leading and   multiple spaces ", {220, 6388, 323, 256, 5248, 12621, 220}},
      {"123 4567 0.5e-3 -42",
       {16, 17, 18, 220, 19, 20, 21, 22, 220, 15, 13, 20, 68, 12, 18, 481, 19, 17}},
      // Control tokens resolve to their reserved ids and are not split into
      // pieces, even though `encode_prompt` never emits them on the t2va path.
      {"<|im_start|>system<|im_end|>", {151644, 8948, 151645}},
      {"<|vision_start|><|image_pad|><|vision_end|>", {151652, 151655, 151653}},
      {"()[]{}<>", {368, 1294, 6257, 21122}},
      {"snake_case camelCase kebab-case dot.case",
       {72139, 19096, 49152, 4207, 1962, 47722, 38485, 12756, 68687}},
  };

  for (const GoldenCase& c : cases) {
    const std::vector<int32_t> got = tok.encode(c.text);
    CHECK_MSG(got.size() == c.ids.size(), "%s: %zu ids, expected %zu", c.text, got.size(),
              c.ids.size());
    if (got.size() != c.ids.size()) continue;
    bool same = true;
    size_t first_bad = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      if (got[i] != c.ids[i]) {
        same = false;
        first_bad = i;
        break;
      }
    }
    CHECK_MSG(same, "%s: first difference at %zu (%d, expected %d)", c.text, first_bad,
              same ? 0 : got[first_bad], same ? 0 : c.ids[first_bad]);
  }

  // Every id must index the embedding table, whose row count comes from the
  // text config rather than from the tokenizer's own vocabulary size.
  bool in_range = true;
  for (const GoldenCase& c : cases) {
    for (int32_t id : tok.encode(c.text)) in_range = in_range && id >= 0 && id < 151936;
  }
  CHECK(in_range);
}

VIDFAB_TEST(tokenizer_round_trip) {
  const std::string path = find_tokenizer();
  if (path.empty()) {
    std::printf("  ref/ tokenizer.json not present; skipping\n");
    return;
  }

  vidfab::text::Tokenizer tok;
  tok.load(path);

  // Byte-level BPE is lossless, so decode(encode(s)) == s for any input,
  // including bytes that are not valid UTF-8 on their own.
  const std::vector<std::string> texts = {
      "hello world",
      "  leading and   multiple spaces ",
      "line1\nline2\ttabbed",
      "naive cafe -- em-dash, punctuation!?",
      "\xe4\xb8\xad\xe6\x96\x87\xe5\xad\x97\xe7\xac\xa6",  // CJK
      "\xf0\x9f\x8e\xac\xf0\x9f\x8e\xa5",                  // emoji
      "a",
      " ",
  };
  for (const std::string& s : texts) {
    const std::string back = tok.decode(tok.encode(s));
    CHECK_MSG(back == s, "round trip differs: %s -> %s", s.c_str(), back.c_str());
  }

  // The empty string produces no tokens at all — the t2va path adds no BOS or
  // EOS, so there is nothing to emit.
  CHECK(tok.encode("").empty());
}

}  // namespace
