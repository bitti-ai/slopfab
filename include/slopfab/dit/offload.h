#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace slopfab::dit {

struct BlockOffloadPlan {
  size_t first = 0, count = 0;
  size_t resident_bytes = 0, host_bytes = 0, slot_bytes = 0, device_bytes = 0;

  size_t slots() const {
    return std::min<size_t>(count, 2);
  }
};

// Select the shortest suffix that fits, including two alternating transfer
// buffers. Block sizes include associated adapter factors and alignment.
inline BlockOffloadPlan plan_block_offload(const std::vector<size_t>& blocks, size_t fixed,
                                           size_t budget, int forced_count = -1) {
  if (forced_count < -1 || forced_count > static_cast<int>(blocks.size()))
    throw std::invalid_argument("transformer offload: invalid block count");
  auto add = [](size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a)
      throw std::overflow_error("transformer offload: byte count overflow");
    return a + b;
  };
  BlockOffloadPlan p;
  p.first = blocks.size();
  p.resident_bytes = fixed;
  for (size_t b : blocks)
    p.resident_bytes = add(p.resident_bytes, b);
  for (;;) {
    p.device_bytes = add(p.resident_bytes, add(p.slot_bytes, p.slots() == 2 ? p.slot_bytes : 0));
    if ((forced_count < 0 || p.count == static_cast<size_t>(forced_count)) &&
        p.device_bytes <= budget)
      return p;
    if (p.count == blocks.size() ||
        (forced_count >= 0 && p.count == static_cast<size_t>(forced_count)))
      throw std::runtime_error(
          "transformer offload: weights and transfer buffers exceed the available VRAM budget");
    const size_t bytes = blocks[--p.first];
    ++p.count;
    p.resident_bytes -= bytes;
    p.host_bytes = add(p.host_bytes, bytes);
    p.slot_bytes = std::max(p.slot_bytes, bytes);
  }
}
} // namespace slopfab::dit
