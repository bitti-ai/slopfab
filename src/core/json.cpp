#include "slopfab/json.h"

#include <cmath>
#include <cstdlib>

namespace slopfab::json {
namespace {

class Parser {
 public:
  explicit Parser(std::string_view text) : s_(text) {}

  Value parse_document() {
    skip_ws();
    Value v = parse_value(0);
    skip_ws();
    if (pos_ != s_.size()) fail("trailing content after JSON document");
    return v;
  }

 private:
  // Guards against stack exhaustion on adversarial or corrupt input. Real
  // safetensors headers nest three or four levels deep.
  static constexpr int kMaxDepth = 64;

  std::string_view s_;
  size_t pos_ = 0;

  [[noreturn]] void fail(const char* what) const {
    throw std::runtime_error("json: " + std::string(what) + " at byte " + std::to_string(pos_));
  }

  bool eof() const { return pos_ >= s_.size(); }
  char peek() const {
    if (eof()) fail("unexpected end of input");
    return s_[pos_];
  }

  void skip_ws() {
    while (!eof()) {
      const char c = s_[pos_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++pos_;
      } else {
        break;
      }
    }
  }

  void expect(char c) {
    if (eof() || s_[pos_] != c) fail("expected character");
    ++pos_;
  }

  bool consume_literal(std::string_view lit) {
    if (s_.substr(pos_, lit.size()) != lit) return false;
    pos_ += lit.size();
    return true;
  }

  Value parse_value(int depth) {
    if (depth > kMaxDepth) fail("maximum nesting depth exceeded");
    switch (peek()) {
      case '{': return parse_object(depth);
      case '[': return parse_array(depth);
      case '"': return Value(parse_string());
      case 't':
        if (!consume_literal("true")) fail("invalid literal");
        return Value(true);
      case 'f':
        if (!consume_literal("false")) fail("invalid literal");
        return Value(false);
      case 'n':
        if (!consume_literal("null")) fail("invalid literal");
        return Value();
      default: return parse_number();
    }
  }

  Value parse_object(int depth) {
    expect('{');
    Object obj;
    skip_ws();
    if (peek() == '}') {
      ++pos_;
      return Value(std::move(obj));
    }
    for (;;) {
      skip_ws();
      std::string key = parse_string();
      skip_ws();
      expect(':');
      skip_ws();
      obj.emplace(std::move(key), parse_value(depth + 1));
      skip_ws();
      const char c = peek();
      ++pos_;
      if (c == '}') break;
      if (c != ',') fail("expected ',' or '}' in object");
    }
    return Value(std::move(obj));
  }

  Value parse_array(int depth) {
    expect('[');
    Array arr;
    skip_ws();
    if (peek() == ']') {
      ++pos_;
      return Value(std::move(arr));
    }
    for (;;) {
      skip_ws();
      arr.push_back(parse_value(depth + 1));
      skip_ws();
      const char c = peek();
      ++pos_;
      if (c == ']') break;
      if (c != ',') fail("expected ',' or ']' in array");
    }
    return Value(std::move(arr));
  }

  // Encodes a code point as UTF-8. Tensor names are ASCII in practice, but
  // metadata fields carry arbitrary text.
  static void append_utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7F) {
      out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
      out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
      out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
  }

  uint32_t parse_hex4() {
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

  std::string parse_string() {
    expect('"');
    std::string out;
    for (;;) {
      if (eof()) fail("unterminated string");
      const char c = s_[pos_++];
      if (c == '"') break;
      if (c != '\\') {
        out.push_back(c);
        continue;
      }
      if (eof()) fail("unterminated escape");
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
          uint32_t cp = parse_hex4();
          // Combine surrogate pairs; a lone surrogate is passed through as the
          // replacement character rather than aborting the parse.
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (pos_ + 1 < s_.size() && s_[pos_] == '\\' && s_[pos_ + 1] == 'u') {
              const size_t save = pos_;
              pos_ += 2;
              const uint32_t lo = parse_hex4();
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
          append_utf8(out, cp);
          break;
        }
        default: fail("unrecognised escape sequence");
      }
    }
    return out;
  }

  Value parse_number() {
    const size_t start = pos_;
    if (!eof() && (s_[pos_] == '-' || s_[pos_] == '+')) ++pos_;
    while (!eof()) {
      const char c = s_[pos_];
      const bool numeric = (c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' ||
                           c == '+' || c == '-';
      if (!numeric) break;
      ++pos_;
    }
    if (pos_ == start) fail("expected a value");

    const std::string text(s_.substr(start, pos_ - start));
    char* end = nullptr;
    const double d = std::strtod(text.c_str(), &end);
    if (end != text.c_str() + text.size()) {
      pos_ = start;
      fail("malformed number");
    }
    return Value(d);
  }
};

}  // namespace

Value parse(std::string_view text) { return Parser(text).parse_document(); }

}  // namespace slopfab::json
