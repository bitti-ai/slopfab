#pragma once
#include <algorithm>
#include <stdexcept>
#include "slopfab/vulkan/tensor.h"

namespace slopfab::vulkan::detail {
// Declared shared arrays, independent of driver lifetime/alias optimizations.
inline SageAttentionConfiguration sage_kernel_configuration(uint32_t kernel, uint32_t subgroup) {
  SageAttentionConfiguration c;
  c.kernel = kernel;
  c.query_rows = kernel >= 3 ? 64 : 32;
  c.value_rows = kernel == 2 || kernel == 4 ? 64 : 16;
  c.local_size = c.query_rows / 16 * 4 * subgroup;
  c.shared_bytes = c.query_rows * 128 + 64 * 128 + c.value_rows * 128 * 2 + c.query_rows * 64 * 6 +
                   c.query_rows * 12;
  return c;
}

inline bool sage_kernel_fits(const DeviceInfo& info, uint32_t kernel) {
  if (kernel < 1 || kernel > 4 || (info.subgroup_size != 32 && info.subgroup_size != 64))
    return false;
  const auto c = sage_kernel_configuration(kernel, info.subgroup_size);
  return c.local_size <= info.max_compute_workgroup_invocations &&
         c.local_size <= info.max_compute_workgroup_size[0] &&
         c.shared_bytes <= info.max_compute_shared_memory_bytes;
}

inline SageAttentionConfiguration select_sage_configuration(const DeviceInfo& info,
                                                            const H3AttentionPlanDesc& d,
                                                            uint64_t extra_budget) {
  if (!sage_kernel_fits(info, 1))
    throw std::runtime_error("vulkan Sage: compact kernel exceeds device limits");
  uint32_t kernel = d.sage_kernel;
  if (kernel == 0) {
    // Keep compact V staging: fewer barriers with full V can cost occupancy.
    // The wide policy is measured on subgroup32; wave64 starts conservatively.
    // Overrides expose all supported tiles for device-specific measurements.
    kernel = info.subgroup_size == 32 && d.sequence >= 256 && sage_kernel_fits(info, 3) ? 3 : 1;
  }
  if (!sage_kernel_fits(info, kernel))
    throw std::invalid_argument("vulkan Sage: requested kernel exceeds device limits");
  auto c = sage_kernel_configuration(kernel, info.subgroup_size);
  const uint64_t elements = uint64_t(d.sequence) * d.heads * d.head_dim;
  const uint64_t aux =
      uint64_t(d.heads) * (d.head_dim + 2ull * ((uint64_t(d.sequence) + 15) / 16)) * 4;
  c.required_workspace_bytes = elements * 2 + aux;
  const uint64_t partials =
      uint64_t(d.heads) * d.head_dim * ((uint64_t(d.sequence) + 255) / 256) * 4;
  c.parallel_mean = d.sequence >= 512 && partials <= extra_budget &&
                    aux + partials <= info.max_storage_buffer_bytes &&
                    aux + partials <= info.max_allocation_bytes;
  if (c.parallel_mean)
    c.extra_workspace_bytes += partials;
  c.prepared_value = elements * 2 <= extra_budget - c.extra_workspace_bytes &&
                     elements * 2 <= info.max_storage_buffer_bytes &&
                     elements * 2 <= info.max_allocation_bytes;
  if (c.prepared_value)
    c.extra_workspace_bytes += elements * 2;
  return c;
}
} // namespace slopfab::vulkan::detail
