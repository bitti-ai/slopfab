#pragma once

#include "slopfab/json.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

namespace slopfab::cli {

inline constexpr const char* kUpscaleVideoEntries =
    "stream=width,height,sample_aspect_ratio,avg_frame_rate,r_frame_rate,start_time:"
    "stream_side_data=rotation,displaymatrix:stream_tags=rotate";

struct UpscaleVideoGeometry {
  int width, height;
  int sar_num = 1, sar_den = 1;

  std::string sar() const {
    return std::to_string(sar_num) + "/" + std::to_string(sar_den);
  }
};

// Predict only the canvas/SAR produced by FFmpeg's autorotate. FFmpeg remains
// responsible for the transform itself, including direction and mirroring.
inline UpscaleVideoGeometry upscale_video_geometry(const json::Value& stream) {
  auto dimension = [&](const char* name) {
    const auto* value = stream.find(name);
    const double n = value && value->is_number() ? value->as_number() : 0;
    if (!std::isfinite(n) || n < 1 || n > std::numeric_limits<int>::max() || std::floor(n) != n)
      throw std::runtime_error("invalid input video dimensions");
    return int(n);
  };
  UpscaleVideoGeometry result{dimension("width"), dimension("height")};
  if (const auto* sar = stream.find("sample_aspect_ratio")) {
    const auto& text = sar->as_string();
    if (text != "N/A" && text != "0:1" && text != "0/1" && text != "0:0") {
      const auto split = text.find_first_of(":/");
      auto positive = [](const std::string& value) {
        size_t end = 0;
        const int n = std::stoi(value, &end);
        if (end != value.size() || n < 1)
          throw std::runtime_error("invalid input video pixel aspect ratio");
        return n;
      };
      if (split == std::string::npos)
        throw std::runtime_error("invalid input video pixel aspect ratio");
      result.sar_num = positive(text.substr(0, split));
      result.sar_den = positive(text.substr(split + 1));
      const int divisor = std::gcd(result.sar_num, result.sar_den);
      result.sar_num /= divisor;
      result.sar_den /= divisor;
    }
  }
  double rotation = 0;
  bool found = false;
  if (const auto* side_data = stream.find("side_data_list"))
    for (const auto& entry : side_data->as_array()) {
      if (const auto* matrix = entry.find("displaymatrix")) {
        // ffprobe truncates its rotation field to whole degrees. Recover the
        // angle from the matrix so e.g. 89.5 degrees still matches autorotate.
        std::istringstream lines(matrix->as_string());
        std::string line;
        double values[9];
        int count = 0;
        while (std::getline(lines, line)) {
          if (line.empty())
            continue;
          const auto colon = line.find(':');
          if (colon == std::string::npos || count >= 9)
            throw std::runtime_error("invalid input video display matrix");
          std::istringstream row(line.substr(colon + 1));
          for (int i = 0; i < 3; ++i) {
            int64_t value;
            if (!(row >> value) || value < INT32_MIN || value > INT32_MAX)
              throw std::runtime_error("invalid input video display matrix");
            values[count++] = double(value);
          }
          if (row >> line)
            throw std::runtime_error("invalid input video display matrix");
        }
        if (count != 9)
          throw std::runtime_error("invalid input video display matrix");
        const double sx = std::hypot(values[0], values[3]);
        const double sy = std::hypot(values[1], values[4]);
        if (sx == 0 || sy == 0)
          throw std::runtime_error("invalid input video display matrix");
        rotation = std::atan2(values[1] / sy, values[0] / sx) * 180.0 / std::acos(-1.0);
        found = true;
        break;
      }
      if (const auto* angle = entry.find("rotation")) {
        rotation = angle->as_number();
        found = true;
        break;
      }
    }
  if (!found)
    if (const auto* tags = stream.find("tags"))
      if (const auto* angle = tags->find("rotate")) {
        size_t end = 0;
        rotation = std::stod(angle->as_string(), &end);
        if (end != angle->as_string().size())
          throw std::runtime_error("invalid input video rotation");
      }
  if (!std::isfinite(rotation))
    throw std::runtime_error("invalid input video rotation");
  // FFmpeg transposes at quarter turns within one degree. Its general-angle
  // rotate filter preserves the input canvas, so arbitrary angles do not swap.
  if (std::abs(std::abs(std::remainder(rotation, 360.0)) - 90.0) < 1.0) {
    std::swap(result.width, result.height);
    std::swap(result.sar_num, result.sar_den);
  }
  return result;
}

} // namespace slopfab::cli
