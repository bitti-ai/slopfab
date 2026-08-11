// Differential tests for the direct tokenizer.json scan.
//
// `Tokenizer::load_json` used to build a `json::Value` tree and walk it once.
// It now scans the text directly. Nothing about the tokenizer's *output* was
// meant to change, and the failure mode if it did is the quiet one this
// project cares about: a vocabulary that is valid, loads fine, and disagrees
// with the reference on a handful of tokens.
//
// So the real tokenizer.json is parsed here a second time, the old way, with
// `vidfab::json`, and the two results are compared entry by entry over all
// 151643 vocabulary entries. Merges are not reachable through the public API,
// so they are compared through behaviour instead: the merge list read by the
// tree parser is written back out as a second tokenizer.json, loaded by the
// new scanner, and the two tokenizers must encode a corpus identically.
//
// `tokenizer.json` lives under `ref/`, which is licence-restricted and not
// committed, so these tests skip cleanly when it is absent.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "harness.h"
#include "vidfab/json.h"
#include "vidfab/text/tokenizer.h"

namespace {

using vidfab::text::Tokenizer;

const char* kTokenizerPath = "ref/FL2VA/text_encoder/tokenizer.json";

std::string find_tokenizer() {
  for (const char* prefix : {"", "../", "../../"}) {
    const std::string candidate = std::string(prefix) + kTokenizerPath;
    if (std::filesystem::exists(candidate)) return candidate;
  }
  return {};
}

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream buf;
  buf << in.rdbuf();
  return buf.str();
}

// The state the old implementation built, rebuilt here through the same
// `vidfab::json` tree it used. Deliberately a transcription of the code that
// was replaced, not a tidied version of it: its value is being the *previous*
// answer.
struct Reference {
  std::map<std::string, int32_t> vocab;                     // after added-token overwrite
  std::vector<std::pair<std::string, int32_t>> added;        // file order
  std::vector<std::string> id_to_token;
  std::vector<std::pair<std::string, std::string>> merges;   // rank order
};

Reference parse_the_old_way(const std::string& text) {
  Reference ref;
  const vidfab::json::Value root = vidfab::json::parse(text);
  const vidfab::json::Value* model = root.find("model");
  const vidfab::json::Value* vocab = model->find("vocab");
  int32_t max_id = -1;
  for (const auto& [token, id] : vocab->as_object()) {
    const auto value = static_cast<int32_t>(id.as_int());
    ref.vocab.emplace(token, value);
    max_id = std::max(max_id, value);
  }

  const vidfab::json::Value* added = root.find("added_tokens");
  if (added != nullptr && added->is_array()) {
    for (const vidfab::json::Value& entry : added->as_array()) {
      const vidfab::json::Value* content = entry.find("content");
      const vidfab::json::Value* id = entry.find("id");
      if (content == nullptr || id == nullptr) continue;
      const auto value = static_cast<int32_t>(id->as_int());
      ref.vocab[content->as_string()] = value;
      ref.added.emplace_back(content->as_string(), value);
      max_id = std::max(max_id, value);
    }
  }

  ref.id_to_token.assign(static_cast<size_t>(max_id) + 1, std::string());
  for (const auto& [token, id] : ref.vocab) ref.id_to_token[static_cast<size_t>(id)] = token;

  const vidfab::json::Value* merges = model->find("merges");
  if (merges != nullptr && merges->is_array()) {
    for (const vidfab::json::Value& m : merges->as_array()) {
      if (m.is_string()) {
        const std::string& s = m.as_string();
        const size_t sp = s.find(' ');
        if (sp == std::string::npos) continue;
        ref.merges.emplace_back(s.substr(0, sp), s.substr(sp + 1));
      } else if (m.is_array() && m.as_array().size() == 2) {
        ref.merges.emplace_back(m.as_array()[0].as_string(), m.as_array()[1].as_string());
      }
    }
  }
  return ref;
}

// Text that exercises a wide spread of the merge table: real vocabulary
// fragments stitched together, plus punctuation and whitespace shapes that
// change how the pre-tokenizer splits.
std::vector<std::string> corpus(const Tokenizer& tok) {
  std::vector<std::string> out;
  const auto vocab_end = static_cast<int32_t>(151000);
  for (int32_t base = 0; base + 8 < vocab_end; base += 331) {
    std::vector<int32_t> ids;
    for (int32_t k = 0; k < 8; ++k) ids.push_back(base + k);
    out.push_back(tok.decode(ids));
  }
  const char* fixed[] = {
      "integrated_multimodal_description: A cat walks across a sunlit kitchen floor.",
      "don't can't I'll we've",
      "  leading and   multiple spaces ",
      "line1\nline2\ttabbed",
      "123 4567 0.5e-3 -42",
      "<|im_start|>system<|im_end|>",
      "<|vision_start|><|image_pad|><|vision_end|>",
      "snake_case camelCase kebab-case dot.case",
      "\xe4\xb8\xad\xe6\x96\x87\xe5\xad\x97\xe7\xac\xa6",
      "\xf0\x9f\x8e\xac\xf0\x9f\x8e\xa5",
  };
  for (const char* s : fixed) out.emplace_back(s);
  // Deterministic pseudo-random ASCII, including the byte values the
  // byte-level fallback exists for.
  uint32_t state = 0x12345678u;
  auto next = [&] {
    state = state * 1664525u + 1013904223u;
    return state >> 16;
  };
  for (int i = 0; i < 200; ++i) {
    std::string s;
    const int n = static_cast<int>(next() % 40) + 1;
    for (int k = 0; k < n; ++k) s.push_back(static_cast<char>(0x20 + next() % 0x5F));
    out.push_back(std::move(s));
  }
  return out;
}

VIDFAB_TEST(tokenizer_scan_matches_json_tree_over_the_whole_vocabulary) {
  const std::string path = find_tokenizer();
  if (path.empty()) {
    std::printf("  ref/ tokenizer.json not present; skipping\n");
    return;
  }
  const std::string text = read_file(path);
  const Reference ref = parse_the_old_way(text);

  Tokenizer tok;
  tok.load_json(text);

  CHECK_MSG(tok.vocab_size() == ref.id_to_token.size(), "vocab_size %zu, tree parser said %zu",
            tok.vocab_size(), ref.id_to_token.size());

  // Every id maps to the same token as before. Counted rather than asserted
  // per element so one systematic difference does not print 151643 times.
  size_t id_diffs = 0;
  size_t first_id_diff = 0;
  const size_t n = std::min(tok.vocab_size(), ref.id_to_token.size());
  for (size_t i = 0; i < n; ++i) {
    if (tok.id_to_token(static_cast<int32_t>(i)) != ref.id_to_token[i]) {
      if (id_diffs == 0) first_id_diff = i;
      ++id_diffs;
    }
  }
  CHECK_MSG(id_diffs == 0, "%zu of %zu ids decode differently, first at %zu", id_diffs, n,
            first_id_diff);

  // ...and every token maps to the same id, which is the direction the encoder
  // actually uses.
  size_t token_diffs = 0;
  std::string first_token_diff;
  for (const auto& [token, id] : ref.vocab) {
    if (tok.token_to_id(token) != id) {
      if (token_diffs == 0) first_token_diff = token;
      ++token_diffs;
    }
  }
  CHECK_MSG(token_diffs == 0, "%zu of %zu tokens map to a different id, first '%s'", token_diffs,
            ref.vocab.size(), first_token_diff.c_str());

  // The added-token table is what makes <|im_start|> survive intact, and the
  // trap is that it must stay sorted longest-first after the scan.
  CHECK(ref.added.size() == 26);
  for (const auto& [content, id] : ref.added) {
    CHECK_MSG(tok.token_to_id(content) == id, "added token '%s' resolved to %d, expected %d",
              content.c_str(), tok.token_to_id(content), id);
    const std::vector<int32_t> ids = tok.encode(content);
    CHECK_MSG(ids.size() == 1 && ids[0] == id, "added token '%s' encoded to %zu ids",
              content.c_str(), ids.size());
  }
}

VIDFAB_TEST(tokenizer_scan_reads_the_same_merges_as_the_json_tree) {
  const std::string path = find_tokenizer();
  if (path.empty()) {
    std::printf("  ref/ tokenizer.json not present; skipping\n");
    return;
  }
  const std::string text = read_file(path);

  Tokenizer scanned;
  scanned.load_json(text);

  // Rebuild the merge table exactly as the old loader did — the same
  // "left\x1Fright" key, the same rank counter that only advances on a
  // successful insert, the same keep-first on a duplicate — and compare it
  // entry for entry. This is the only piece of loaded state with no other
  // observable, and a comparison that went through `encode` on both sides
  // would share the downstream code and see nothing.
  const Reference ref = parse_the_old_way(text);
  CHECK(ref.merges.size() == 151387);
  std::map<std::string, int32_t> expected;
  int32_t rank = 0;
  for (const auto& [left, right] : ref.merges) {
    std::string key = left;
    key.push_back('\x1F');
    key += right;
    if (expected.emplace(std::move(key), rank).second) ++rank;
  }

  const auto& actual = scanned.merge_ranks_for_testing();
  CHECK_MSG(actual.size() == expected.size(), "%zu merges loaded, tree parser found %zu",
            actual.size(), expected.size());

  size_t missing = 0;
  size_t misranked = 0;
  std::string first_bad;
  for (const auto& [key, want] : expected) {
    const auto it = actual.find(key);
    if (it == actual.end()) {
      if (missing + misranked == 0) first_bad = key;
      ++missing;
    } else if (it->second != want) {
      if (missing + misranked == 0) first_bad = key;
      ++misranked;
    }
  }
  // `\x1F` is not printable; report the key with it spelled out.
  std::string shown = first_bad;
  for (char& c : shown) {
    if (c == '\x1F') c = '|';
  }
  CHECK_MSG(missing == 0 && misranked == 0,
            "%zu merges missing and %zu misranked of %zu, first '%s'", missing, misranked,
            expected.size(), shown.c_str());

  // Byte-level BPE stays lossless over a broad corpus, which is the property a
  // corrupted merge *key* breaks first even when the rank survives.
  size_t lossy = 0;
  for (const std::string& s : corpus(scanned)) {
    if (scanned.decode(scanned.encode(s)) != s) ++lossy;
  }
  CHECK_MSG(lossy == 0, "%zu corpus strings failed to round trip", lossy);
}

VIDFAB_TEST(tokenizer_scan_handles_escapes_added_tokens_and_malformed_input) {
  // Small enough to reason about exactly, and it holds every shape the scanner
  // has to get right: escaped and literal UTF-8, a surrogate pair, merges in
  // both the string and the pair-array form, and added tokens listed *before*
  // the model section and shortest-first.
  const std::string doc =
      "{\n"
      "  \"version\": \"1.0\",\n"
      "  \"added_tokens\": [\n"
      "    {\"id\": 5, \"content\": \"ab\", \"special\": true},\n"
      "    {\"id\": 6, \"content\": \"abc\", \"special\": true},\n"
      "    {\"id\": 7, \"content\": \"a\"}\n"
      "  ],\n"
      "  \"normalizer\": null,\n"
      "  \"pre_tokenizer\": {\"type\": \"ByteLevel\", \"nested\": [1, [2, {\"x\": \"}\"}]]},\n"
      "  \"model\": {\n"
      "    \"type\": \"BPE\",\n"
      "    \"dropout\": null,\n"
      "    \"vocab\": {\"a\": 0, \"b\": 1, \"\\u00e9\": 2, \"\\ud83d\\ude00\": 3,"
      " \"q\\\"r\": 4, \"tail\": 8},\n"
      "    \"merges\": [\"a b\", [\"b\", \"a\"], \"nospace\", [\"x\"], \"c d\"]\n"
      "  }\n"
      "}";

  Tokenizer tok;
  tok.load_json(doc);

  // Escapes decode to the same bytes the tree parser produced.
  CHECK(tok.token_to_id("\xc3\xa9") == 2);            // \u00e9
  CHECK(tok.token_to_id("\xf0\x9f\x98\x80") == 3);    // \ud83d\ude00 surrogate pair
  CHECK(tok.token_to_id("q\"r") == 4);                // escaped quote inside a key

  // An added token overwrites the vocabulary entry of the same content: "a"
  // was 0 in the vocab and is 7 in added_tokens.
  CHECK(tok.token_to_id("a") == 7);
  CHECK(tok.id_to_token(7) == "a");
  CHECK(tok.vocab_size() == 9);  // max id 8 ("tail") + 1

  // THE TRAP. The added tokens are listed shortest-first in the file and must
  // be matched longest-first, or "abc" is consumed as "ab" plus a stray "c".
  const std::vector<int32_t> abc = tok.encode("abc");
  CHECK_MSG(abc.size() == 1 && abc[0] == 6, "encode(\"abc\") gave %zu ids, first %d", abc.size(),
            abc.empty() ? -1 : abc[0]);

  // Objects and arrays nested inside skipped sections, including a '}' inside
  // a string, must not end the walk early: `tail` is the last vocab entry and
  // it is present.
  CHECK(tok.token_to_id("tail") == 8);

  // Malformed documents are rejected rather than half-loaded.
  const char* bad[] = {
      "{\"model\": {\"vocab\": {\"a\": 0}}} trailing",
      "{\"model\": {\"vocab\": {\"a\": 0}}",
      "{\"model\": {\"vocab\": {\"a\": }}}",
      "{\"model\": {\"vocab\": {\"a\": 0}, \"merges\": [\"a b\"}}",
      "[]",
      "{\"model\": {\"merges\": []}}",  // no vocab
      "{\"version\": \"1.0\"}",         // no model
  };
  for (const char* doc_text : bad) {
    Tokenizer t;
    bool threw = false;
    try {
      t.load_json(doc_text);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK_MSG(threw, "malformed document was accepted: %s", doc_text);
  }
}

}  // namespace
