#pragma once

#include <cstdint>

namespace vidfab::vulkan::detail {

constexpr bool norm_dispatch_fits(uint64_t rows, uint32_t max_workgroups_x) noexcept {
  return rows != 0 && rows <= max_workgroups_x;
}

}  // namespace vidfab::vulkan::detail
