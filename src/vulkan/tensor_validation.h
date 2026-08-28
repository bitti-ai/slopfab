#pragma once

#include <cstdint>

namespace vidfab::vulkan::detail {

constexpr bool norm_dispatch_fits(uint64_t rows, uint32_t max_workgroups_x) noexcept {
  return rows != 0 && rows <= max_workgroups_x;
}

// Exact rsqrt behavior is not a portable Vulkan capability. This tuple is the
// only device/driver combination on which the checked-in norm modules have
// completed the CUDA bit-parity matrix. Extend only with recorded evidence.
constexpr bool known_exact_vae_norm_device(uint32_t vendor_id, uint32_t device_id,
                                           uint32_t driver_version) noexcept {
  return vendor_id == 0x10deu && device_id == 0x2b85u &&
         driver_version == 0x98960000u;  // RTX 5090, NVIDIA 610.88
}

}  // namespace vidfab::vulkan::detail
