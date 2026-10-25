#include "nfc.h"
#include "slopfab/text/tokenizer.h"
#include <algorithm>
#include <iterator>
#include <vector>

namespace slopfab::text {
namespace {
#include "nfc_tables.inc"

int combining(uint32_t cp) {
  auto it =
      std::lower_bound(std::begin(classes), std::end(classes), cp, [](const auto& a, uint32_t b) {
        return a[0] < b;
      });
  return it != std::end(classes) && (*it)[0] == cp ? int((*it)[1]) : 0;
}

void decompose(uint32_t cp, std::vector<uint32_t>& out) {
  if (cp >= 0xac00 && cp < 0xac00 + 11172) {
    auto s = cp - 0xac00;
    out.push_back(0x1100 + s / 588);
    out.push_back(0x1161 + (s % 588) / 28);
    if (s % 28)
      out.push_back(0x11a7 + s % 28);
    return;
  }
  auto it = std::lower_bound(std::begin(decompositions), std::end(decompositions), cp,
                             [](const auto& a, uint32_t b) {
                               return a[0] < b;
                             });
  if (it == std::end(decompositions) || (*it)[0] != cp)
    out.push_back(cp);
  else {
    decompose((*it)[1], out);
    if ((*it)[2])
      decompose((*it)[2], out);
  }
}

uint32_t compose(uint32_t a, uint32_t b) {
  if (a >= 0x1100 && a < 0x1113 && b >= 0x1161 && b < 0x1176)
    return 0xac00 + (a - 0x1100) * 588 + (b - 0x1161) * 28;
  if (a >= 0xac00 && a < 0xac00 + 11172 && (a - 0xac00) % 28 == 0 && b > 0x11a7 && b < 0x11c3)
    return a + b - 0x11a7;
  auto key = std::make_pair(a, b);
  auto it = std::lower_bound(std::begin(compositions), std::end(compositions), key,
                             [](const auto& row, auto k) {
                               return std::make_pair(row[0], row[1]) < k;
                             });
  return it != std::end(compositions) && (*it)[0] == a && (*it)[1] == b ? (*it)[2] : 0;
}
}

std::string normalize_nfc(const std::string& source) {
  std::vector<uint32_t> d;
  for (size_t pos = 0; pos < source.size();)
    decompose(utf8_next(source, pos), d);
  for (size_t i = 1; i < d.size(); ++i) {
    const int cc = combining(d[i]);
    if (!cc)
      continue;
    size_t j = i;
    while (j && combining(d[j - 1]) > cc) {
      std::swap(d[j], d[j - 1]);
      --j;
    }
  }
  std::vector<uint32_t> out;
  size_t starter = 0;
  int previous = 0;
  for (uint32_t cp : d) {
    int cc = combining(cp);
    uint32_t combined = out.empty() ? 0 : compose(out[starter], cp);
    if (combined && (previous < cc || previous == 0))
      out[starter] = combined;
    else {
      if (!cc)
        starter = out.size();
      out.push_back(cp);
      previous = cc;
    }
  }
  std::string result;
  for (uint32_t cp : out)
    utf8_append(result, cp);
  return result;
}
}
