#pragma once

#include <cstdint>
#include <string>

namespace vidfab::video {

struct ExactY4mComparison {
  uint64_t expected_size = 0;
  uint64_t actual_size = 0;
  uint64_t first_difference = UINT64_MAX;
  int expected_byte = -1;
  int actual_byte = -1;
  std::string expected_header;
  std::string actual_header;

  bool equal() const noexcept { return first_difference == UINT64_MAX; }
};

// Byte-exact, streaming comparison for deterministic raw-video runs. Throws
// for unreadable files or inputs without a YUV4MPEG2 header.
ExactY4mComparison compare_y4m_exact(const std::string& expected_path,
                                     const std::string& actual_path);

}  // namespace vidfab::video
