#include "vidfab/video/y4m_compare.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace vidfab::video {
namespace {

uint64_t stream_size(std::ifstream& stream, const std::string& path) {
  stream.seekg(0, std::ios::end);
  const std::streamoff size = stream.tellg();
  if (size < 0) throw std::runtime_error("compare-y4m: cannot size " + path);
  stream.seekg(0, std::ios::beg);
  return static_cast<uint64_t>(size);
}

std::string read_header(std::ifstream& stream, const std::string& path) {
  std::string header;
  for (size_t i = 0; i < 4096; ++i) {
    const int ch = stream.get();
    if (ch == EOF) break;
    if (ch == '\n') break;
    header.push_back(static_cast<char>(ch));
  }
  if (header.rfind("YUV4MPEG2 ", 0) != 0) {
    throw std::runtime_error("compare-y4m: " + path + " has no YUV4MPEG2 header");
  }
  stream.clear();
  stream.seekg(0, std::ios::beg);
  return header;
}

}  // namespace

ExactY4mComparison compare_y4m_exact(const std::string& expected_path,
                                     const std::string& actual_path) {
  std::error_code equivalent_error;
  if (std::filesystem::equivalent(expected_path, actual_path, equivalent_error)) {
    throw std::invalid_argument("compare-y4m: expected and actual refer to the same file");
  }
  std::ifstream expected(expected_path, std::ios::binary);
  if (!expected) throw std::runtime_error("compare-y4m: cannot open " + expected_path);
  std::ifstream actual(actual_path, std::ios::binary);
  if (!actual) throw std::runtime_error("compare-y4m: cannot open " + actual_path);

  ExactY4mComparison result;
  result.expected_size = stream_size(expected, expected_path);
  result.actual_size = stream_size(actual, actual_path);
  result.expected_header = read_header(expected, expected_path);
  result.actual_header = read_header(actual, actual_path);

  std::array<unsigned char, 64 * 1024> expected_bytes{};
  std::array<unsigned char, 64 * 1024> actual_bytes{};
  uint64_t offset = 0;
  const uint64_t common = std::min(result.expected_size, result.actual_size);
  while (offset < common) {
    const size_t count = static_cast<size_t>(std::min<uint64_t>(expected_bytes.size(),
                                                                common - offset));
    expected.read(reinterpret_cast<char*>(expected_bytes.data()),
                  static_cast<std::streamsize>(count));
    actual.read(reinterpret_cast<char*>(actual_bytes.data()), static_cast<std::streamsize>(count));
    if (expected.gcount() != static_cast<std::streamsize>(count) ||
        actual.gcount() != static_cast<std::streamsize>(count)) {
      throw std::runtime_error("compare-y4m: file changed or became unreadable during comparison");
    }
    for (size_t i = 0; i < count; ++i) {
      if (expected_bytes[i] != actual_bytes[i]) {
        result.first_difference = offset + i;
        result.expected_byte = expected_bytes[i];
        result.actual_byte = actual_bytes[i];
        return result;
      }
    }
    offset += count;
  }
  if (result.expected_size != result.actual_size) {
    result.first_difference = common;
    if (common < result.expected_size) {
      char next = 0;
      expected.get(next);
      result.expected_byte = static_cast<unsigned char>(next);
    }
    if (common < result.actual_size) {
      char next = 0;
      actual.get(next);
      result.actual_byte = static_cast<unsigned char>(next);
    }
  }
  return result;
}

}  // namespace vidfab::video
