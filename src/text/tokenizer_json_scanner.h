#pragma once
#include "slopfab/text/tokenizer.h"
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace slopfab::text::detail {
// --- tokenizer.json scanning -------------------------------------------------
//
// A structural walk of the tokenizer file that materialises only the strings
// the tokenizer keeps, instead of building a `json::Value` tree first.
//
// `src/core/json.cpp` is deliberately untouched and still parses everything
// else. It is the right shape for a safetensors header — 100 KB, read once,
// arbitrary structure. It is the wrong shape for this file: 151643 vocab
// entries and 151387 merges become ~303000 tree nodes of 80 bytes each, every
// one owning a `std::string`, about 40 MB of allocation built to be walked
// once and thrown away. Measured at 147-166 ms against 35-44 ms for this.
//
// String decoding below — the escapes, the `\uXXXX` surrogate pairing and the
// lone-surrogate to U+FFFD fallback — is character for character the same as
// `json.cpp`'s. A tokenizer that decoded one escape differently would build a
// perfectly valid vocabulary that silently disagreed with the reference on the
// tokens containing it.
class TokenizerScanner {
 public:
  explicit TokenizerScanner(std::string_view text) : s_(text) {}

  bool at_object() { return peek_or_null() == '{'; }
  bool at_array() { return peek_or_null() == '['; }
  bool at_string() { return peek_or_null() == '"'; }

  // Calls `on_key(key)` for each member of the object at the cursor, with the
  // cursor left on that member's value. The callback must consume exactly one
  // value; `skip_value()` is how it declines one.
  //
  // `key` is one buffer reused across this object's own members, so a callback
  // that needs it past its own return must copy. Nested objects get their own.
  template <typename Fn>
  void object(Fn&& on_key) {
    expect('{');
    ws();
    if (peek() == '}') {
      ++pos_;
      return;
    }
    std::string key;
    for (;;) {
      ws();
      string(key);
      ws();
      expect(':');
      on_key(key);
      ws();
      const char c = peek();
      ++pos_;
      if (c == '}') break;
      if (c != ',') fail("expected ',' or '}' in object");
    }
  }

  // As `object`, for arrays: `on_element()` consumes exactly one value.
  template <typename Fn>
  void array(Fn&& on_element) {
    expect('[');
    ws();
    if (peek() == ']') {
      ++pos_;
      return;
    }
    for (;;) {
      ws();
      on_element();
      ws();
      const char c = peek();
      ++pos_;
      if (c == ']') break;
      if (c != ',') fail("expected ',' or ']' in array");
    }
  }

  // Decodes the string at the cursor into `out`, which is cleared first and
  // keeps its capacity. Unescaped runs are appended in one go, which is nearly
  // all of this file.
  void string(std::string& out) {
    out.clear();
    ws();
    if (pos_ >= s_.size() || s_[pos_] != '"') fail("expected a string");
    ++pos_;
    size_t run = pos_;
    for (;;) {
      if (pos_ >= s_.size()) fail("unterminated string");
      const char c = s_[pos_];
      if (c == '"') {
        out.append(s_.data() + run, pos_ - run);
        ++pos_;
        return;
      }
      if (c != '\\') {
        ++pos_;
        continue;
      }
      out.append(s_.data() + run, pos_ - run);
      ++pos_;
      if (pos_ >= s_.size()) fail("unterminated escape");
      switch (s_[pos_++]) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          uint32_t cp = hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (pos_ + 1 < s_.size() && s_[pos_] == '\\' && s_[pos_ + 1] == 'u') {
              const size_t save = pos_;
              pos_ += 2;
              const uint32_t lo = hex4();
              if (lo >= 0xDC00 && lo <= 0xDFFF) {
                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
              } else {
                pos_ = save;
                cp = 0xFFFD;
              }
            } else {
              cp = 0xFFFD;
            }
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            cp = 0xFFFD;
          }
          utf8_append(out, cp);
          break;
        }
        default: fail("unrecognised escape sequence");
      }
      run = pos_;
    }
  }

  // The number at the cursor, truncated toward zero. `json::Value::as_int` is
  // `static_cast<int64_t>` over a `strtod` result, so the plain-integer fast
  // path below has to agree with that: it does, because every value it accepts
  // is exactly representable as a double and the cast then truncates nothing.
  int64_t integer() {
    ws();
    const size_t start = pos_;
    if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+')) ++pos_;
    while (pos_ < s_.size()) {
      const char c = s_[pos_];
      const bool numeric = (c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
                           c == '+' || c == '-';
      if (!numeric) break;
      ++pos_;
    }
    if (pos_ == start) fail("expected a number");
    const std::string_view token = s_.substr(start, pos_ - start);

    size_t i = 0;
    bool negative = false;
    if (token[0] == '-') {
      negative = true;
      i = 1;
    } else if (token[0] == '+') {
      i = 1;
    }
    bool plain = i < token.size();
    int64_t value = 0;
    for (; plain && i < token.size(); ++i) {
      if (token[i] < '0' || token[i] > '9') {
        plain = false;
        break;
      }
      value = value * 10 + (token[i] - '0');
      // Past 2^53 a double stops representing every integer, so hand those to
      // strtod rather than quietly disagreeing with the tree parser.
      if (value > (int64_t{1} << 53)) {
        plain = false;
        break;
      }
    }
    if (plain) return negative ? -value : value;

    const std::string text(token);
    char* end = nullptr;
    const double d = std::strtod(text.c_str(), &end);
    if (end != text.c_str() + text.size()) {
      pos_ = start;
      fail("malformed number");
    }
    return static_cast<int64_t>(d);
  }

  // Consumes one value of any shape without materialising it.
  void skip_value(int depth = 0) {
    if (depth > kMaxDepth) fail("maximum nesting depth exceeded");
    ws();
    switch (peek()) {
      case '{':
        ++pos_;
        ws();
        if (peek() == '}') {
          ++pos_;
          return;
        }
        for (;;) {
          ws();
          skip_string();
          ws();
          expect(':');
          skip_value(depth + 1);
          ws();
          {
            const char c = peek();
            ++pos_;
            if (c == '}') return;
            if (c != ',') fail("expected ',' or '}' in object");
          }
        }
      case '[':
        ++pos_;
        ws();
        if (peek() == ']') {
          ++pos_;
          return;
        }
        for (;;) {
          skip_value(depth + 1);
          ws();
          {
            const char c = peek();
            ++pos_;
            if (c == ']') return;
            if (c != ',') fail("expected ',' or ']' in array");
          }
        }
      case '"': skip_string(); return;
      case 't':
        if (!literal("true")) fail("invalid literal");
        return;
      case 'f':
        if (!literal("false")) fail("invalid literal");
        return;
      case 'n':
        if (!literal("null")) fail("invalid literal");
        return;
      default: integer(); return;
    }
  }

  std::string_view value_text() {
    ws();
    const size_t start = pos_;
    skip_value();
    return s_.substr(start, pos_ - start);
  }

  // Same trailing-content check the tree parser performs, so a truncated or
  // concatenated file fails loudly here too rather than yielding a half-built
  // vocabulary.
  void finish() {
    ws();
    if (pos_ != s_.size()) fail("trailing content after JSON document");
  }

 private:
  static constexpr int kMaxDepth = 64;

  [[noreturn]] void fail(const char* what) const {
    throw std::runtime_error("tokenizer: json: " + std::string(what) + " at byte " +
                             std::to_string(pos_));
  }

  void ws() {
    while (pos_ < s_.size()) {
      const char c = s_[pos_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++pos_;
      } else {
        break;
      }
    }
  }

  char peek() {
    if (pos_ >= s_.size()) fail("unexpected end of input");
    return s_[pos_];
  }

  // For the `at_*` probes, where end of input is simply "not that".
  char peek_or_null() {
    ws();
    return pos_ < s_.size() ? s_[pos_] : '\0';
  }

  void expect(char c) {
    ws();
    if (pos_ >= s_.size() || s_[pos_] != c) fail("expected character");
    ++pos_;
  }

  bool literal(std::string_view lit) {
    if (s_.substr(pos_, lit.size()) != lit) return false;
    pos_ += lit.size();
    return true;
  }

  uint32_t hex4() {
    if (pos_ + 4 > s_.size()) fail("truncated \\u escape");
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = s_[pos_++];
      v <<= 4;
      if (c >= '0' && c <= '9') {
        v |= static_cast<uint32_t>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        v |= static_cast<uint32_t>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        v |= static_cast<uint32_t>(c - 'A' + 10);
      } else {
        fail("invalid hex digit in \\u escape");
      }
    }
    return v;
  }

  // Walks a string without decoding it, for the parts of the file the
  // tokenizer does not read.
  void skip_string() {
    ws();
    if (pos_ >= s_.size() || s_[pos_] != '"') fail("expected a string");
    ++pos_;
    for (;;) {
      if (pos_ >= s_.size()) fail("unterminated string");
      const char c = s_[pos_++];
      if (c == '"') return;
      if (c == '\\') {
        if (pos_ >= s_.size()) fail("unterminated escape");
        ++pos_;
      }
    }
  }

  std::string_view s_;
  size_t pos_ = 0;
};

}
