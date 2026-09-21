#pragma once

#include <cstdint>

namespace slopfab::vulkan::detail {

constexpr bool norm_dispatch_fits(uint64_t rows, uint32_t max_workgroups_x) noexcept {
  return rows != 0 && rows <= max_workgroups_x;
}

struct GemmDispatchGeometry {
  uint32_t x = 0;
  uint32_t y = 0;
};

// Checked ceil-division for a two-dimensional output grid. The quotient form
// avoids the conventional value + tile - 1 overflow at API boundaries.
constexpr bool gemm_dispatch_geometry(uint64_t rows, uint64_t columns, uint32_t tile_rows,
                                      uint32_t tile_columns, uint32_t max_x, uint32_t max_y,
                                      GemmDispatchGeometry* result) noexcept {
  if (rows == 0 || columns == 0 || tile_rows == 0 || tile_columns == 0 || max_x == 0 ||
      max_y == 0 || result == nullptr) {
    return false;
  }
  const uint64_t x = columns / tile_columns + (columns % tile_columns != 0 ? 1u : 0u);
  const uint64_t y = rows / tile_rows + (rows % tile_rows != 0 ? 1u : 0u);
  if (x > max_x || y > max_y)
    return false;
  result->x = static_cast<uint32_t>(x);
  result->y = static_cast<uint32_t>(y);
  return true;
}

// Exact rsqrt behavior is not a portable Vulkan capability. This tuple is the
// only device/driver combination on which the checked-in norm modules have
// completed the CUDA bit-parity matrix. Extend only with recorded evidence.
constexpr bool known_exact_vae_norm_device(uint32_t vendor_id, uint32_t device_id,
                                           uint32_t driver_version) noexcept {
  return vendor_id == 0x10deu && device_id == 0x2b85u &&
         driver_version == 0x98960000u; // RTX 5090, NVIDIA 610.88
}

// Kept distinct from normalization even while the first qualified tuple is
// the same. Pointwise exactness pins three independently generated modules and
// must not silently broaden when a future normalization tuple is added.
constexpr bool known_exact_vae_pointwise_device(uint32_t vendor_id, uint32_t device_id,
                                                uint32_t driver_version) noexcept {
  return vendor_id == 0x10deu && device_id == 0x2b85u &&
         driver_version == 0x98960000u; // RTX 5090, NVIDIA 610.88
}

} // namespace slopfab::vulkan::detail
