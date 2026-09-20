#pragma once
#include "runtime_state.h"

namespace slopfab::vulkan {
struct BufferPool::Impl : std::enable_shared_from_this<BufferPool::Impl> {
  struct Range { uint64_t offset; uint64_t bytes; };
  struct Block {
    std::shared_ptr<detail::DeviceState> device;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint64_t bytes = 0;
    uint32_t memory_type = 0;
    VkMemoryPropertyFlags properties = 0;
    void* mapped = nullptr;
    bool dedicated = false;
    bool addressable = false;
    bool recycle_failed = false;
    uint64_t used = 0;
    std::vector<Range> free;
    mutable std::mutex mapped_mutex;

    ~Block() {
      if (memory == VK_NULL_HANDLE) return;
      if (mapped != nullptr) device->unmap_memory(device->device, memory);
      device->free_memory(device->device, memory, nullptr);
    }
  };

  std::shared_ptr<detail::DeviceState> device;
  uint64_t block_bytes = 0;
  mutable std::mutex mutex;
  std::vector<std::shared_ptr<Block>> blocks;

  uint32_t choose_memory_type(uint32_t bits, MemoryUsage usage) const {
    VkMemoryPropertyFlags required = 0;
    VkMemoryPropertyFlags preferred = 0;
    if (usage == MemoryUsage::kDevice) {
      required = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    } else {
      required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
      preferred = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
      if (usage == MemoryUsage::kReadback) preferred |= VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    }
    int best = -1;
    int best_score = -1;
    for (uint32_t i = 0; i < device->memory.memoryTypeCount; ++i) {
      if (!(bits & (1u << i))) continue;
      const auto flags = device->memory.memoryTypes[i].propertyFlags;
      if ((flags & required) != required) continue;
      int score = 0;
      for (uint32_t bit = 1; bit != 0; bit <<= 1) {
        if ((preferred & bit) && (flags & bit)) ++score;
      }
      if (score > best_score) { best = static_cast<int>(i); best_score = score; }
    }
    if (best < 0) throw std::runtime_error("vulkan: no compatible memory type for buffer");
    return static_cast<uint32_t>(best);
  }

  static bool take_range(Block& block, uint64_t bytes, uint64_t alignment, uint64_t* offset) {
    for (size_t i = 0; i < block.free.size(); ++i) {
      const Range range = block.free[i];
      const uint64_t aligned = detail::align_up(range.offset, alignment);
      if (aligned < range.offset || aligned - range.offset > range.bytes ||
          bytes > range.bytes - (aligned - range.offset)) continue;
      const uint64_t prefix = aligned - range.offset;
      const uint64_t suffix = range.bytes - prefix - bytes;
      // A split grows the vector by one. Reserve before changing the free list
      // so allocation failure leaves the original span intact.
      if (prefix != 0 && suffix != 0) block.free.reserve(block.free.size() + 1);
      block.free.erase(block.free.begin() + static_cast<std::ptrdiff_t>(i));
      if (suffix != 0) block.free.insert(block.free.begin() + static_cast<std::ptrdiff_t>(i),
                                        {aligned + bytes, suffix});
      if (prefix != 0) block.free.insert(block.free.begin() + static_cast<std::ptrdiff_t>(i),
                                        {range.offset, prefix});
      block.used += bytes;
      *offset = aligned;
      return true;
    }
    return false;
  }

  std::shared_ptr<Block> make_block(uint64_t bytes, uint32_t memory_type, bool dedicated,
                                    bool addressable, VkBuffer dedicated_buffer) {
    auto block = std::make_shared<Block>();
    block->device = device;
    block->bytes = bytes;
    block->memory_type = memory_type;
    block->properties = device->memory.memoryTypes[memory_type].propertyFlags;
    block->dedicated = dedicated;
    block->addressable = addressable;
    VkMemoryDedicatedAllocateInfo dedicated_info{};
    dedicated_info.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
    dedicated_info.buffer = dedicated_buffer;
    VkMemoryAllocateFlagsInfo flags_info{};
    flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags_info.pNext = dedicated ? &dedicated_info : nullptr;
    flags_info.flags = addressable ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT : 0;
    VkMemoryAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.pNext = addressable ? static_cast<const void*>(&flags_info)
                                 : (dedicated ? static_cast<const void*>(&dedicated_info)
                                              : nullptr);
    allocate.allocationSize = bytes;
    allocate.memoryTypeIndex = memory_type;
    const VkResult allocation_result =
        device->allocate_memory(device->device, &allocate, nullptr, &block->memory);
    if (allocation_result != VK_SUCCESS) {
      // Keep the failed request and this pool's residency in the exception:
      // the caller may only retain the final generation error, not stdout.
      uint64_t reserved = 0, used = 0;
      for (const auto& existing : blocks) {
        reserved += existing->bytes;
        used += existing->used;
      }
      const uint32_t heap = device->memory.memoryTypes[memory_type].heapIndex;
      const std::string operation = "vkAllocateMemory [requested=" +
          std::to_string(bytes) + " bytes, memory_type=" + std::to_string(memory_type) +
          ", heap=" + std::to_string(heap) + ", heap_size=" +
          std::to_string(device->memory.memoryHeaps[heap].size) +
          " bytes, pool_reserved=" + std::to_string(reserved) +
          " bytes, pool_used=" + std::to_string(used) + " bytes]";
      detail::fail(operation.c_str(), allocation_result);
    }
    if (block->properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
      try {
        detail::check(device->map_memory(device->device, block->memory, 0, bytes, 0,
                                         &block->mapped), "vkMapMemory");
      } catch (...) {
        device->free_memory(device->device, block->memory, nullptr);
        block->memory = VK_NULL_HANDLE;
        throw;
      }
    }
    block->free.push_back({0, bytes});
    blocks.push_back(block);
    return block;
  }

  void release(const std::shared_ptr<Block>& block, uint64_t offset, uint64_t bytes) noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    block->used -= bytes;
    if (!block->recycle_failed) {
      try {
        // Reserve before mutation. If metadata allocation fails, retire this
        // whole block from reuse; its remaining live buffers stay valid and
        // the block is freed as soon as the last one is released.
        block->free.reserve(block->free.size() + 1);
        auto pos = std::lower_bound(block->free.begin(), block->free.end(), offset,
                                    [](const Range& range, uint64_t value) {
                                      return range.offset < value;
                                    });
        pos = block->free.insert(pos, {offset, bytes});
        if (pos != block->free.begin()) {
          auto previous = pos - 1;
          if (previous->offset + previous->bytes == pos->offset) {
            previous->bytes += pos->bytes;
            pos = block->free.erase(pos) - 1;
          }
        }
        if (pos + 1 != block->free.end() &&
            pos->offset + pos->bytes == (pos + 1)->offset) {
          pos->bytes += (pos + 1)->bytes;
          block->free.erase(pos + 1);
        }
      } catch (...) {
        block->recycle_failed = true;
        block->free.clear();
      }
    }
    if ((block->dedicated || block->recycle_failed) && block->used == 0) {
      blocks.erase(std::remove(blocks.begin(), blocks.end(), block), blocks.end());
    }
  }
};

struct Buffer::Impl {
  std::shared_ptr<BufferPool::Impl> pool;
  std::shared_ptr<BufferPool::Impl::Block> block;
  VkBuffer buffer = VK_NULL_HANDLE;
  uint64_t bytes = 0;
  uint64_t allocation_bytes = 0;
  uint64_t offset = 0;
  MemoryUsage usage = MemoryUsage::kDevice;
  BufferUsage buffer_usage = static_cast<BufferUsage>(0);

  ~Impl() {
    if (buffer != VK_NULL_HANDLE) pool->device->destroy_buffer(pool->device->device, buffer, nullptr);
    if (block) pool->release(block, offset, allocation_bytes);
  }

  void check_range(uint64_t range_offset, uint64_t range_bytes) const {
    if (range_offset > bytes || range_bytes > bytes - range_offset) {
      throw std::out_of_range("vulkan: buffer range exceeds allocation");
    }
  }

  VkMappedMemoryRange mapped_range(uint64_t range_offset, uint64_t range_bytes) const {
    check_range(range_offset, range_bytes);
    const uint64_t atom = pool->device->non_coherent_atom_size;
    const uint64_t absolute = offset + range_offset;
    const uint64_t begin = absolute - absolute % atom;
    const uint64_t end_unaligned = absolute + range_bytes;
    const uint64_t end = std::min<uint64_t>(offset + allocation_bytes,
                                            detail::align_up(end_unaligned, atom));
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = block->memory;
    range.offset = begin;
    range.size = end - begin;
    return range;
  }

  void flush_range(uint64_t range_offset, uint64_t range_bytes) const {
    if (block->properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) return;
    const auto range = mapped_range(range_offset, range_bytes);
    detail::check(pool->device->flush_mapped_ranges(pool->device->device, 1, &range),
                  "vkFlushMappedMemoryRanges");
  }

  void invalidate_range(uint64_t range_offset, uint64_t range_bytes) const {
    if (block->properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) return;
    const auto range = mapped_range(range_offset, range_bytes);
    detail::check(pool->device->invalidate_mapped_ranges(pool->device->device, 1, &range),
                  "vkInvalidateMappedMemoryRanges");
  }
};


}  // namespace slopfab::vulkan
