#pragma once

#include <cstdint>
#include <memory>

#include "vidfab/video/y4m.h"

namespace vidfab::vulkan {

// Vulkan accelerates output colour conversion only. Model inference remains
// CUDA and this class does not provide a Vulkan inference backend.
class Yuv420Converter final : public video::FrameConverter {
 public:
  explicit Yuv420Converter(uint32_t device_index = 0);
  ~Yuv420Converter() override;
  Yuv420Converter(Yuv420Converter&&) noexcept;
  Yuv420Converter& operator=(Yuv420Converter&&) noexcept;
  Yuv420Converter(const Yuv420Converter&) = delete;
  Yuv420Converter& operator=(const Yuv420Converter&) = delete;

  void convert(const float* r, const float* g, const float* b, int height, int width,
               uint8_t* y_plane, int y_stride, uint8_t* u_plane, int u_stride,
               uint8_t* v_plane, int v_stride) override;

  uint64_t reserved_bytes() const;
  uint64_t high_water_bytes() const;
  uint64_t capacity_pixels() const;
  const char* device_name() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vidfab::vulkan
