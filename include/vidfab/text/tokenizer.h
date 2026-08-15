// Byte-level BPE tokenizer for Qwen3-VL (Qwen2TokenizerFast).
//
// The pipeline is the usual three stages:
//   1. split the text with the GPT-4 style pattern
//        (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}
//        | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//   2. map each byte of a piece through the reversible byte-to-unicode table
//   3. apply BPE merges within each piece
//
// std::regex has no \p{L} / \p{N} support, so the split is implemented
// directly against a Unicode category table rather than by a regex engine.
// This also avoids a regex dependency, which matters here.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vidfab::text {

class Tokenizer {
 public:
  // Loads vocab and merges from a HuggingFace tokenizer.json. Added/special
  // tokens are read from the same file so that <|im_start|> and friends
  // resolve to their reserved ids.
  void load(const std::string& tokenizer_json_path);

  // Loads the same tokenizer directly from JSON bytes. Used by the CLI's
  // embedded tokenizer resource; keeping parsing here makes file overrides
  // and the built-in data follow exactly the same code path.
  void load_json(std::string_view tokenizer_json);

  // Loads tokenizer.json compiled into whichever module holds this code — the
  // vidfab executable in a static build, vidfab_core.dll in a shared one.
  // Windows only; elsewhere it throws and the caller must pass a file.
  void load_embedded();

  bool loaded() const { return !vocab_.empty(); }
  size_t vocab_size() const { return id_to_token_.size(); }

  // Encodes text. Special-token strings present in the added-tokens table are
  // matched before the byte-level split, so a chat template's control tokens
  // survive intact instead of being broken into pieces.
  std::vector<int32_t> encode(const std::string& text) const;

  // Inverse of encode, including the byte-level unmapping.
  std::string decode(const std::vector<int32_t>& ids) const;

  // Returns -1 when absent.
  int32_t token_to_id(const std::string& token) const;
  const std::string& id_to_token(int32_t id) const;

  // Splits `text` with the pre-tokenizer pattern. Exposed for testing: it is
  // the stage most likely to diverge from the reference.
  std::vector<std::string> pre_tokenize(const std::string& text) const;

  // Exposed for testing. The merge table is the one piece of loaded state with
  // no other observable — a dropped or misranked merge only shows up as a
  // differently but still plausibly tokenised prompt — and the load path has
  // an old-versus-new differential test that has to compare it exactly.
  const std::unordered_map<std::string, int32_t>& merge_ranks_for_testing() const {
    return merge_ranks_;
  }

 private:
  std::unordered_map<std::string, int32_t> vocab_;
  std::vector<std::string> id_to_token_;
  // Merge rank keyed by "left\x1Fright"; lower rank merges first.
  std::unordered_map<std::string, int32_t> merge_ranks_;
  // Added tokens matched verbatim before splitting, longest first.
  std::vector<std::pair<std::string, int32_t>> added_tokens_;

  std::vector<std::string> bpe(const std::string& piece) const;
};

// --- exposed for testing ----------------------------------------------------

// Decodes one UTF-8 code point starting at `pos`, advancing it. Returns
// U+FFFD and advances one byte on malformed input.
uint32_t utf8_next(const std::string& s, size_t& pos);

// Appends `cp` as UTF-8.
void utf8_append(std::string& out, uint32_t cp);

// Unicode general-category tests used by the pre-tokenizer pattern.
bool is_letter(uint32_t cp);  // \p{L}
bool is_number(uint32_t cp);  // \p{N}
bool is_space(uint32_t cp);   // \s

// The reversible byte <-> code point map GPT-2 style tokenizers use so that
// every byte is printable and no byte maps to a whitespace the splitter would
// then re-split.
const std::vector<uint32_t>& byte_to_unicode();

}  // namespace vidfab::text
