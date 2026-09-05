#include "slopfab/text/tokenizer.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

namespace slopfab::text {
namespace {

struct Range {
  uint32_t lo;
  uint32_t hi;
};

// Unicode general category L (letters). Covers the scripts a prompt is
// realistically written in: Latin (incl. all supplements and extended blocks),
// Greek, Cyrillic, Armenian, Hebrew, Arabic, Syriac, Thaana, N'Ko, Devanagari
// through Sinhala, Thai, Lao, Tibetan, Myanmar, Georgian, Hangul, Ethiopic,
// Cherokee, Khmer, Mongolian, CJK (incl. extensions A-B), Hiragana, Katakana,
// Bopomofo, Yi, and the Latin/Greek maths alphanumerics.
//
// It is a table, not the full UCD: a code point in an unlisted historic or
// rare script is treated as a symbol rather than a letter, which changes only
// how that run is split, never whether text round-trips. Byte-level BPE below
// still encodes it losslessly.
constexpr Range kLetterRanges[] = {
    {0x0041, 0x005A}, {0x0061, 0x007A}, {0x00AA, 0x00AA}, {0x00B5, 0x00B5},
    {0x00BA, 0x00BA}, {0x00C0, 0x00D6}, {0x00D8, 0x00F6}, {0x00F8, 0x02C1},
    {0x02C6, 0x02D1}, {0x02E0, 0x02E4}, {0x0370, 0x0374}, {0x0376, 0x0377},
    {0x037A, 0x037D}, {0x037F, 0x037F}, {0x0386, 0x0386}, {0x0388, 0x03FF},
    {0x0400, 0x0481}, {0x048A, 0x052F}, {0x0531, 0x0556}, {0x0561, 0x0587},
    {0x05D0, 0x05EA}, {0x05EF, 0x05F2}, {0x0620, 0x064A}, {0x066E, 0x066F},
    {0x0671, 0x06D3}, {0x06D5, 0x06D5}, {0x06E5, 0x06E6}, {0x06EE, 0x06EF},
    {0x06FA, 0x06FC}, {0x0712, 0x072F}, {0x074D, 0x07A5}, {0x07B1, 0x07B1},
    {0x07CA, 0x07EA}, {0x0904, 0x0939}, {0x093D, 0x093D}, {0x0950, 0x0950},
    {0x0958, 0x0961}, {0x0971, 0x0980}, {0x0985, 0x09B9}, {0x09DC, 0x09E1},
    {0x0A05, 0x0A39}, {0x0A85, 0x0AB9}, {0x0B05, 0x0B39}, {0x0B85, 0x0BB9},
    {0x0C05, 0x0C39}, {0x0C85, 0x0CB9}, {0x0D05, 0x0D3A}, {0x0D85, 0x0DC6},
    {0x0E01, 0x0E30}, {0x0E32, 0x0E33}, {0x0E40, 0x0E46}, {0x0E81, 0x0EB0},
    {0x0EC0, 0x0EC6}, {0x0F40, 0x0F6C}, {0x1000, 0x102A}, {0x103F, 0x103F},
    {0x1050, 0x1055}, {0x10A0, 0x10FA}, {0x10FC, 0x1248}, {0x124A, 0x1360},
    {0x13A0, 0x13F5}, {0x1401, 0x166C}, {0x1681, 0x169A}, {0x16A0, 0x16EA},
    {0x1700, 0x17D7}, {0x1820, 0x18AA}, {0x1900, 0x191E}, {0x1A00, 0x1A16},
    {0x1B05, 0x1B33}, {0x1C00, 0x1C23}, {0x1E00, 0x1FBC}, {0x2071, 0x2071},
    {0x207F, 0x207F}, {0x2090, 0x209C}, {0x2102, 0x2102}, {0x2107, 0x2107},
    {0x210A, 0x2113}, {0x2115, 0x2115}, {0x2119, 0x211D}, {0x2124, 0x2124},
    {0x2126, 0x2126}, {0x2128, 0x2128}, {0x212A, 0x212D}, {0x212F, 0x2139},
    {0x213C, 0x213F}, {0x2145, 0x2149}, {0x214E, 0x214E}, {0x2183, 0x2184},
    {0x2C00, 0x2CE4}, {0x2D00, 0x2D2D}, {0x2D30, 0x2D67}, {0x2D80, 0x2DDE},
    {0x3005, 0x3006}, {0x3031, 0x3035}, {0x303B, 0x303C}, {0x3041, 0x3096},
    {0x309D, 0x309F}, {0x30A1, 0x30FA}, {0x30FC, 0x30FF}, {0x3105, 0x312F},
    {0x3131, 0x318E}, {0x31A0, 0x31BF}, {0x31F0, 0x31FF}, {0x3400, 0x4DBF},
    {0x4E00, 0x9FFF}, {0xA000, 0xA48C}, {0xA4D0, 0xA4FD}, {0xA500, 0xA60C},
    {0xA610, 0xA61F}, {0xA62A, 0xA62B}, {0xA640, 0xA66E}, {0xA67F, 0xA69D},
    {0xA717, 0xA71F}, {0xA722, 0xA788}, {0xA78B, 0xA7CA}, {0xA800, 0xA801},
    {0xAC00, 0xD7A3}, {0xF900, 0xFA6D}, {0xFB00, 0xFB17}, {0xFB1D, 0xFBB1},
    {0xFC00, 0xFD3D}, {0xFE70, 0xFEFC}, {0xFF21, 0xFF3A}, {0xFF41, 0xFF5A},
    {0xFF66, 0xFFDC}, {0x1D400, 0x1D7CB}, {0x20000, 0x2A6DF}, {0x2A700, 0x2EBE0},
};

// Unicode general category N (numbers): ASCII digits, the fullwidth forms, and
// the digit blocks of the scripts above.
constexpr Range kNumberRanges[] = {
    {0x0030, 0x0039}, {0x00B2, 0x00B3}, {0x00B9, 0x00B9}, {0x00BC, 0x00BE},
    {0x0660, 0x0669}, {0x06F0, 0x06F9}, {0x07C0, 0x07C9}, {0x0966, 0x096F},
    {0x09E6, 0x09EF}, {0x0A66, 0x0A6F}, {0x0AE6, 0x0AEF}, {0x0B66, 0x0B6F},
    {0x0BE6, 0x0BF2}, {0x0C66, 0x0C6F}, {0x0CE6, 0x0CEF}, {0x0D66, 0x0D75},
    {0x0DE6, 0x0DEF}, {0x0E50, 0x0E59}, {0x0ED0, 0x0ED9}, {0x0F20, 0x0F33},
    {0x1040, 0x1049}, {0x1090, 0x1099}, {0x17E0, 0x17E9}, {0x1810, 0x1819},
    {0x1946, 0x194F}, {0x19D0, 0x19DA}, {0x1A80, 0x1A99}, {0x1B50, 0x1B59},
    {0x1BB0, 0x1BB9}, {0x1C40, 0x1C49}, {0x1C50, 0x1C59}, {0x2070, 0x2070},
    {0x2074, 0x2079}, {0x2080, 0x2089}, {0x2150, 0x2182}, {0x2185, 0x2189},
    {0x2460, 0x249B}, {0x24EA, 0x24FF}, {0x2776, 0x2793}, {0x3007, 0x3007},
    {0x3021, 0x3029}, {0x3038, 0x303A}, {0x3192, 0x3195}, {0x3220, 0x3229},
    {0x3248, 0x324F}, {0x3251, 0x325F}, {0x3280, 0x3289}, {0x32B1, 0x32BF},
    {0xA620, 0xA629}, {0xA8D0, 0xA8D9}, {0xA900, 0xA909}, {0xA9D0, 0xA9D9},
    {0xAA50, 0xAA59}, {0xABF0, 0xABF9}, {0xFF10, 0xFF19}, {0x1D7CE, 0x1D7FF},
};

template <size_t N>
bool in_ranges(uint32_t cp, const Range (&ranges)[N]) {
  size_t lo = 0;
  size_t hi = N;
  while (lo < hi) {
    const size_t mid = (lo + hi) / 2;
    if (cp < ranges[mid].lo) {
      hi = mid;
    } else if (cp > ranges[mid].hi) {
      lo = mid + 1;
    } else {
      return true;
    }
  }
  return false;
}

}  // namespace

bool is_letter(uint32_t cp) { return in_ranges(cp, kLetterRanges); }
bool is_number(uint32_t cp) { return in_ranges(cp, kNumberRanges); }

bool is_space(uint32_t cp) {
  // \s in the Rust regex crate: ASCII whitespace plus the Unicode White_Space
  // property. The pattern only ever tests membership, never a specific class.
  switch (cp) {
    case 0x09: case 0x0A: case 0x0B: case 0x0C: case 0x0D: case 0x20:
    case 0x85: case 0xA0: case 0x1680: case 0x2028: case 0x2029:
    case 0x202F: case 0x205F: case 0x3000:
      return true;
    default:
      return cp >= 0x2000 && cp <= 0x200A;
  }
}

uint32_t utf8_next(const std::string& s, size_t& pos) {
  if (pos >= s.size()) return 0;
  const auto b0 = static_cast<uint8_t>(s[pos]);
  if (b0 < 0x80) {
    ++pos;
    return b0;
  }
  auto cont = [&](size_t i) -> bool {
    return pos + i < s.size() && (static_cast<uint8_t>(s[pos + i]) & 0xC0) == 0x80;
  };
  if ((b0 & 0xE0) == 0xC0 && cont(1)) {
    const uint32_t cp = ((b0 & 0x1Fu) << 6) | (static_cast<uint8_t>(s[pos + 1]) & 0x3Fu);
    pos += 2;
    return cp;
  }
  if ((b0 & 0xF0) == 0xE0 && cont(1) && cont(2)) {
    const uint32_t cp = ((b0 & 0x0Fu) << 12) |
                        ((static_cast<uint8_t>(s[pos + 1]) & 0x3Fu) << 6) |
                        (static_cast<uint8_t>(s[pos + 2]) & 0x3Fu);
    pos += 3;
    return cp;
  }
  if ((b0 & 0xF8) == 0xF0 && cont(1) && cont(2) && cont(3)) {
    const uint32_t cp = ((b0 & 0x07u) << 18) |
                        ((static_cast<uint8_t>(s[pos + 1]) & 0x3Fu) << 12) |
                        ((static_cast<uint8_t>(s[pos + 2]) & 0x3Fu) << 6) |
                        (static_cast<uint8_t>(s[pos + 3]) & 0x3Fu);
    pos += 4;
    return cp;
  }
  ++pos;
  return 0xFFFD;
}

void utf8_append(std::string& out, uint32_t cp) {
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

const std::vector<uint32_t>& byte_to_unicode() {
  // Printable ASCII, Latin-1 punctuation and Latin-1 letters keep their own
  // code point; every other byte is shifted into the private-use-adjacent
  // range starting at U+0100, in byte order.
  static const std::vector<uint32_t> table = [] {
    std::vector<uint32_t> t(256, 0);
    std::vector<bool> direct(256, false);
    for (int b = '!'; b <= '~'; ++b) direct[static_cast<size_t>(b)] = true;
    for (int b = 0xA1; b <= 0xAC; ++b) direct[static_cast<size_t>(b)] = true;
    for (int b = 0xAE; b <= 0xFF; ++b) direct[static_cast<size_t>(b)] = true;
    uint32_t next = 0x100;
    for (int b = 0; b < 256; ++b) {
      if (direct[static_cast<size_t>(b)]) {
        t[static_cast<size_t>(b)] = static_cast<uint32_t>(b);
      } else {
        t[static_cast<size_t>(b)] = next++;
      }
    }
    return t;
  }();
  return table;
}

namespace {

const std::unordered_map<uint32_t, uint8_t>& unicode_to_byte() {
  static const std::unordered_map<uint32_t, uint8_t> inverse = [] {
    std::unordered_map<uint32_t, uint8_t> m;
    const std::vector<uint32_t>& fwd = byte_to_unicode();
    for (size_t b = 0; b < fwd.size(); ++b) m.emplace(fwd[b], static_cast<uint8_t>(b));
    return m;
  }();
  return inverse;
}

// Case-insensitive match of one of the contraction suffixes at `pos`.
// Returns its byte length, or 0.
size_t match_contraction(const std::string& s, size_t pos) {
  if (pos >= s.size() || s[pos] != '\'') return 0;
  static const char* kOne[] = {"s", "t", "m", "d"};
  static const char* kTwo[] = {"re", "ve", "ll"};
  auto lower = [](char c) { return static_cast<char>(c | 0x20); };
  if (pos + 1 < s.size()) {
    for (const char* c : kOne) {
      if (lower(s[pos + 1]) == c[0]) return 2;
    }
  }
  if (pos + 2 < s.size()) {
    for (const char* c : kTwo) {
      if (lower(s[pos + 1]) == c[0] && lower(s[pos + 2]) == c[1]) return 3;
    }
  }
  return 0;
}

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

// Sizing hints for the two big tables. Wrong in either direction costs one
// rehash, never correctness, so they are the real counts rather than a bound.
constexpr size_t kVocabHint = 151643;
constexpr size_t kMergeHint = 151387;

}  // namespace

std::vector<std::string> Tokenizer::pre_tokenize(const std::string& text) const {
  std::vector<std::string> out;
  size_t i = 0;
  const size_t n = text.size();

  while (i < n) {
    // Alternative 1: contractions.
    if (const size_t len = match_contraction(text, i); len != 0) {
      out.emplace_back(text, i, len);
      i += len;
      continue;
    }

    size_t probe = i;
    const uint32_t cp = utf8_next(text, probe);
    const size_t cp_len = probe - i;

    // Alternative 2: an optional non-letter/non-digit/non-newline lead
    // character, then a run of letters.
    {
      size_t j = i;
      if (!is_letter(cp) && !is_number(cp) && cp != '\r' && cp != '\n') {
        size_t after = probe;
        size_t k = after;
        const uint32_t next = utf8_next(text, k);
        if (after < n && is_letter(next)) j = after;
      }
      if (j == i ? is_letter(cp) : true) {
        size_t k = j;
        size_t end = j;
        while (k < n) {
          size_t save = k;
          const uint32_t c = utf8_next(text, k);
          if (!is_letter(c)) {
            k = save;
            break;
          }
          end = k;
        }
        if (end > j) {
          out.emplace_back(text, i, end - i);
          i = end;
          continue;
        }
      }
    }

    // Alternative 3: a single digit. The pattern has no quantifier here, so
    // numbers are split one code point at a time.
    if (is_number(cp)) {
      out.emplace_back(text, i, cp_len);
      i = probe;
      continue;
    }

    // Alternative 4: an optional single space, then a run of symbols, then
    // any trailing newlines.
    {
      size_t j = i;
      if (cp == ' ') j = probe;
      size_t k = j;
      size_t end = j;
      while (k < n) {
        size_t save = k;
        const uint32_t c = utf8_next(text, k);
        if (is_space(c) || is_letter(c) || is_number(c)) {
          k = save;
          break;
        }
        end = k;
      }
      if (end > j) {
        while (end < n && (text[end] == '\r' || text[end] == '\n')) ++end;
        out.emplace_back(text, i, end - i);
        i = end;
        continue;
      }
    }

    // Alternative 5: whitespace runs ending in newlines.
    {
      size_t k = i;
      size_t last_newline_end = std::string::npos;
      while (k < n) {
        size_t save = k;
        const uint32_t c = utf8_next(text, k);
        if (!is_space(c)) {
          k = save;
          break;
        }
        if (c == '\r' || c == '\n') last_newline_end = k;
      }
      if (last_newline_end != std::string::npos) {
        out.emplace_back(text, i, last_newline_end - i);
        i = last_newline_end;
        continue;
      }
    }

    // Alternatives 6 and 7: whitespace not followed by a non-space, else any
    // whitespace run.
    {
      size_t k = i;
      size_t end = i;
      while (k < n) {
        size_t save = k;
        const uint32_t c = utf8_next(text, k);
        if (!is_space(c)) {
          k = save;
          break;
        }
        end = k;
      }
      if (end > i) {
        // \s+(?!\S): keep the final whitespace for the next piece if more
        // text follows, matching the lookahead.
        size_t stop = end;
        if (end < n) {
          size_t back = end;
          while (back > i) {
            size_t probe2 = i;
            size_t prev = i;
            while (probe2 < back) {
              prev = probe2;
              utf8_next(text, probe2);
            }
            back = prev;
            break;
          }
          if (back > i) stop = back;
        }
        out.emplace_back(text, i, stop - i);
        i = stop;
        continue;
      }
    }

    // Nothing matched (should be unreachable); consume one code point so the
    // loop always terminates.
    out.emplace_back(text, i, cp_len);
    i = probe;
  }

  return out;
}

std::vector<std::string> Tokenizer::bpe(const std::string& piece) const {
  // Start from single code points of the byte-mapped string.
  std::vector<std::string> symbols;
  size_t pos = 0;
  while (pos < piece.size()) {
    const size_t start = pos;
    utf8_next(piece, pos);
    symbols.emplace_back(piece, start, pos - start);
  }
  if (symbols.size() < 2) return symbols;

  for (;;) {
    int32_t best_rank = std::numeric_limits<int32_t>::max();
    size_t best_i = 0;
    bool found = false;
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      std::string key = symbols[i];
      key.push_back('\x1F');
      key += symbols[i + 1];
      auto it = merge_ranks_.find(key);
      if (it != merge_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = i;
        found = true;
      }
    }
    if (!found) break;
    symbols[best_i] += symbols[best_i + 1];
    symbols.erase(symbols.begin() + static_cast<long long>(best_i) + 1);
    if (symbols.size() < 2) break;
  }
  return symbols;
}

void Tokenizer::load(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("tokenizer: cannot open " + path);
  std::ostringstream buf;
  buf << in.rdbuf();
  const std::string text = buf.str();
  load_json(text);
}

void Tokenizer::load_json(std::string_view tokenizer_json) {
  vocab_.clear();
  id_to_token_.clear();
  merge_ranks_.clear();
  added_tokens_.clear();

  // The four things below are collected by one forward scan of the text, but
  // they are *assembled* in exactly the order the json-tree version assembled
  // them — base vocab, then added tokens over the top, then the longest-first
  // sort, then the id table — because that order is what makes <|im_start|>
  // win over a shorter prefix and what decides which token owns an id. The
  // scan therefore parks the added tokens rather than applying them where it
  // finds them: `added_tokens` precedes `model` in the file, and applying them
  // first would invert the overwrite.
  std::vector<std::pair<std::string, int32_t>> added;
  int32_t max_id = -1;
  bool saw_model = false;
  bool saw_vocab = false;

  TokenizerScanner scan(tokenizer_json);
  if (!scan.at_object()) throw std::runtime_error("tokenizer: no \"model\" section");
  scan.object([&](const std::string& key) {
    if (key == "added_tokens") {
      if (!scan.at_array()) {
        scan.skip_value();
        return;
      }
      std::string content;
      scan.array([&] {
        if (!scan.at_object()) {
          scan.skip_value();
          return;
        }
        bool have_content = false;
        bool have_id = false;
        int32_t id = 0;
        content.clear();
        scan.object([&](const std::string& field) {
          if (field == "content") {
            scan.string(content);
            have_content = true;
          } else if (field == "id") {
            id = static_cast<int32_t>(scan.integer());
            have_id = true;
          } else {
            scan.skip_value();
          }
        });
        if (!have_content || !have_id) return;
        added.emplace_back(content, id);
      });
      return;
    }
    if (key != "model") {
      scan.skip_value();
      return;
    }
    saw_model = true;
    if (!scan.at_object()) {
      scan.skip_value();
      return;
    }
    scan.object([&](const std::string& model_key) {
      if (model_key == "vocab") {
        saw_vocab = true;
        if (!scan.at_object()) throw std::runtime_error("tokenizer: vocab is not an object");
        // 151643 entries. Reserving turns the rehash chain into one allocation.
        vocab_.reserve(kVocabHint);
        scan.object([&](const std::string& token) {
          const auto value = static_cast<int32_t>(scan.integer());
          vocab_.emplace(token, value);
          max_id = std::max(max_id, value);
        });
        return;
      }
      if (model_key != "merges") {
        scan.skip_value();
        return;
      }
      if (!scan.at_array()) {
        scan.skip_value();
        return;
      }
      merge_ranks_.reserve(kMergeHint);
      int32_t rank = 0;
      std::string entry;
      std::string key_buf;
      scan.array([&] {
        if (scan.at_string()) {
          // "left right"
          scan.string(entry);
          const size_t sp = entry.find(' ');
          if (sp == std::string::npos) return;
          key_buf.assign(entry, 0, sp);
          key_buf.push_back('\x1F');
          key_buf.append(entry, sp + 1, std::string::npos);
        } else if (scan.at_array()) {
          // ["left", "right"], the pair form. An array of any other length is
          // skipped without consuming a rank, exactly as before; a two-element
          // array holding something other than strings still throws, because
          // that is a merge table this cannot read rather than one it can
          // ignore.
          int seen = 0;
          bool strings = true;
          key_buf.clear();
          scan.array([&] {
            if (seen < 2 && scan.at_string()) {
              scan.string(entry);
              if (seen == 1) key_buf.push_back('\x1F');
              key_buf += entry;
            } else {
              if (seen < 2) strings = false;
              scan.skip_value();
            }
            ++seen;
          });
          if (seen != 2) return;
          if (!strings) throw std::runtime_error("tokenizer: merge pair is not two strings");
        } else {
          scan.skip_value();
          return;
        }
        merge_ranks_.emplace(key_buf, rank++);
      });
    });
  });
  scan.finish();

  if (!saw_model) throw std::runtime_error("tokenizer: no \"model\" section");
  if (!saw_vocab) throw std::runtime_error("tokenizer: no vocab");

  for (const auto& [content, value] : added) {
    vocab_[content] = value;
    added_tokens_.emplace_back(content, value);
    max_id = std::max(max_id, value);
  }
  // Longest first so <|im_start|> wins over any shorter prefix.
  std::sort(added_tokens_.begin(), added_tokens_.end(),
            [](const auto& a, const auto& b) { return a.first.size() > b.first.size(); });

  // Every id has to index `id_to_token_` below, and `static_cast<size_t>` on a
  // negative int32 does not fail, it produces ~1.8e19 and writes there. A
  // single flipped byte in the file is enough: turning `"ĠEXTI": 85018` into
  // `"ĠEXTI":-85018` is one character and segfaults the unguarded assembly.
  //
  // The upper bound is separate from the sign check and not redundant with it:
  // a plausible-looking id of 2e9 asks for a 2-billion-entry vector of
  // std::string, which is a 64 GB allocation rather than a wrong answer. Four
  // times the vocabulary is far above any real gap between the base vocab and
  // the reserved control-token block and far below anything that allocates.
  const auto vocab_entries = static_cast<int64_t>(vocab_.size());
  const int64_t id_limit = 4 * vocab_entries + 1024;
  for (const auto& [token, id] : vocab_) {
    if (id < 0) {
      throw std::runtime_error("tokenizer: token '" + token + "' has a negative id " +
                               std::to_string(id));
    }
    if (static_cast<int64_t>(id) > id_limit) {
      throw std::runtime_error("tokenizer: token '" + token + "' has id " + std::to_string(id) +
                               ", beyond the plausible bound " + std::to_string(id_limit) +
                               " for a vocabulary of " + std::to_string(vocab_entries));
    }
  }

  // `max_id` is taken over every id the file declared, including one that an
  // added token later overwrote and that therefore no longer appears in
  // `vocab_`. It sizes the vector, so it is bounded separately rather than
  // inferred from the survivors.
  if (max_id > id_limit) {
    throw std::runtime_error("tokenizer: largest id " + std::to_string(max_id) +
                             " is beyond the plausible bound " + std::to_string(id_limit) +
                             " for a vocabulary of " + std::to_string(vocab_entries));
  }

  id_to_token_.assign(static_cast<size_t>(max_id) + 1, std::string());
  for (const auto& [token, id] : vocab_) {
    id_to_token_[static_cast<size_t>(id)] = token;
  }
}

#if defined(_WIN32)
namespace {

// The module this translation unit was linked into, which is where the
// tokenizer resource lives: either slopfab.exe or slopfab.dll.
//
// `FindResourceW(nullptr, ...)` asks for the *process* module instead, i.e.
// always the executable. That is correct for the CLI but wrong when an
// application loads slopfab.dll and carries no resource 101 at all — or, worse,
// carries an unrelated RCDATA 101 of its own, which would be handed to
// `load_json` as a
// tokenizer. Anchoring on an address inside this module is correct for both
// outputs.
HMODULE containing_module() {
  HMODULE module = nullptr;
  GetModuleHandleExW(
      GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
      reinterpret_cast<LPCWSTR>(&containing_module), &module);
  return module;
}

}  // namespace
#endif

void Tokenizer::load_embedded() {
#if defined(_WIN32)
  const HMODULE module = containing_module();
  if (module == nullptr) {
    throw std::runtime_error("tokenizer: cannot identify the module holding the embedded resource");
  }
  HRSRC resource = FindResourceW(module, MAKEINTRESOURCEW(101), MAKEINTRESOURCEW(10));
  if (resource == nullptr) throw std::runtime_error("tokenizer: embedded resource is missing");
  HGLOBAL loaded = LoadResource(module, resource);
  const DWORD size = SizeofResource(module, resource);
  const void* bytes = loaded == nullptr ? nullptr : LockResource(loaded);
  if (bytes == nullptr || size == 0) {
    throw std::runtime_error("tokenizer: cannot read embedded resource");
  }
  load_json(std::string_view(static_cast<const char*>(bytes), size));
#else
  throw std::runtime_error(
      "tokenizer: this build has no embedded tokenizer; pass --tokenizer <file>");
#endif
}

int32_t Tokenizer::token_to_id(const std::string& token) const {
  auto it = vocab_.find(token);
  return it == vocab_.end() ? -1 : it->second;
}

const std::string& Tokenizer::id_to_token(int32_t id) const {
  static const std::string empty;
  if (id < 0 || static_cast<size_t>(id) >= id_to_token_.size()) return empty;
  return id_to_token_[static_cast<size_t>(id)];
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
  if (!loaded()) throw std::runtime_error("tokenizer: not loaded");
  std::vector<int32_t> ids;

  const std::vector<uint32_t>& b2u = byte_to_unicode();

  // Walk the input, peeling off added tokens verbatim wherever they occur so
  // chat-template control tokens are never split.
  size_t cursor = 0;
  while (cursor <= text.size()) {
    size_t hit = std::string::npos;
    size_t hit_len = 0;
    int32_t hit_id = -1;
    for (const auto& [token, id] : added_tokens_) {
      const size_t at = text.find(token, cursor);
      if (at != std::string::npos && (hit == std::string::npos || at < hit)) {
        hit = at;
        hit_len = token.size();
        hit_id = id;
      }
    }

    const size_t chunk_end = (hit == std::string::npos) ? text.size() : hit;
    if (chunk_end > cursor) {
      const std::string chunk = text.substr(cursor, chunk_end - cursor);
      for (const std::string& piece : pre_tokenize(chunk)) {
        std::string mapped;
        for (char c : piece) {
          utf8_append(mapped, b2u[static_cast<uint8_t>(c)]);
        }
        for (const std::string& sym : bpe(mapped)) {
          auto it = vocab_.find(sym);
          if (it != vocab_.end()) ids.push_back(it->second);
        }
      }
    }

    if (hit == std::string::npos) break;
    ids.push_back(hit_id);
    cursor = hit + hit_len;
  }

  return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
  const std::unordered_map<uint32_t, uint8_t>& u2b = unicode_to_byte();
  std::string out;
  for (int32_t id : ids) {
    const std::string& token = id_to_token(id);
    // Added tokens are stored verbatim rather than byte-mapped.
    bool is_added = false;
    for (const auto& [content, added_id] : added_tokens_) {
      if (added_id == id) {
        out += content;
        is_added = true;
        break;
      }
    }
    if (is_added) continue;

    size_t pos = 0;
    while (pos < token.size()) {
      const uint32_t cp = utf8_next(token, pos);
      auto it = u2b.find(cp);
      if (it != u2b.end()) out.push_back(static_cast<char>(it->second));
    }
  }
  return out;
}

}  // namespace slopfab::text
