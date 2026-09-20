#include "runtime_buffer_state.h"

namespace slopfab::vulkan {
namespace detail {
VkBufferUsageFlags buffer_usage_flags(BufferUsage usage) {
  const uint32_t value = static_cast<uint32_t>(usage);
  VkBufferUsageFlags result = 0;
  if (value & static_cast<uint32_t>(BufferUsage::kStorage)) result |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  if (value & static_cast<uint32_t>(BufferUsage::kUniform)) result |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
  if (value & static_cast<uint32_t>(BufferUsage::kTransferSource)) result |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if (value & static_cast<uint32_t>(BufferUsage::kTransferDestination)) result |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (value & static_cast<uint32_t>(BufferUsage::kDeviceAddress)) result |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  return result;
}

}  // namespace detail
BufferPool::BufferPool(const Device& device, uint64_t block_bytes) {
  if (!device.impl_) throw std::invalid_argument("vulkan: BufferPool requires a device");
  if (block_bytes == 0) throw std::invalid_argument("vulkan: BufferPool block size is zero");
  impl_ = std::make_shared<Impl>();
  impl_->device = device.impl_->state;
  impl_->block_bytes = block_bytes;
}
BufferPool::~BufferPool() = default;
BufferPool::BufferPool(BufferPool&&) noexcept = default;
BufferPool& BufferPool::operator=(BufferPool&&) noexcept = default;

Buffer BufferPool::allocate(uint64_t bytes, BufferUsage usage, MemoryUsage memory) {
  if (!impl_) throw std::logic_error("vulkan: empty BufferPool");
  if (bytes == 0) throw std::invalid_argument("vulkan: zero-sized buffers are not supported");
  const VkBufferUsageFlags flags = detail::buffer_usage_flags(usage);
  if (flags == 0) throw std::invalid_argument("vulkan: buffer usage is empty");
  const bool addressable =
      (static_cast<uint32_t>(usage) & static_cast<uint32_t>(BufferUsage::kDeviceAddress)) != 0;
  if (addressable && !impl_->device->buffer_device_address_enabled) {
    throw std::logic_error(
        "vulkan: kDeviceAddress requires enable_buffer_device_address at device creation");
  }

  VkBufferCreateInfo create{};
  create.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  create.size = bytes;
  create.usage = flags;
  create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VkBuffer buffer = VK_NULL_HANDLE;
  detail::check(impl_->device->create_buffer(impl_->device->device, &create, nullptr, &buffer),
                "vkCreateBuffer");
  std::shared_ptr<Impl::Block> block;
  uint64_t offset = 0;
  uint64_t suballocation_bytes = 0;
  bool span_reserved = false;
  VkMemoryRequirements requirements{};
  try {
    bool dedicated = false;
    if (impl_->device->get_buffer_requirements2 != nullptr) {
      VkBufferMemoryRequirementsInfo2 request{};
      request.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
      request.buffer = buffer;
      VkMemoryDedicatedRequirements dedicated_requirements{};
      dedicated_requirements.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
      VkMemoryRequirements2 requirements2{};
      requirements2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
      requirements2.pNext = &dedicated_requirements;
      impl_->device->get_buffer_requirements2(impl_->device->device, &request, &requirements2);
      requirements = requirements2.memoryRequirements;
      dedicated = dedicated_requirements.requiresDedicatedAllocation == VK_TRUE;
    } else {
      impl_->device->get_buffer_requirements(impl_->device->device, buffer, &requirements);
    }
    const uint32_t type = impl_->choose_memory_type(requirements.memoryTypeBits, memory);
    const bool host_visible = (impl_->device->memory.memoryTypes[type].propertyFlags &
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    const uint64_t suballocation_alignment = host_visible
        ? std::max<uint64_t>(requirements.alignment, impl_->device->non_coherent_atom_size)
        : requirements.alignment;
    suballocation_bytes = host_visible
        ? detail::align_up(requirements.size, impl_->device->non_coherent_atom_size)
        : requirements.size;
    {
      std::lock_guard<std::mutex> lock(impl_->mutex);
      if (!dedicated) {
        for (const auto& candidate : impl_->blocks) {
          if (!candidate->dedicated && !candidate->recycle_failed &&
              candidate->memory_type == type &&
              (!addressable || candidate->addressable) &&
              Impl::take_range(*candidate, suballocation_bytes, suballocation_alignment, &offset)) {
            block = candidate;
            span_reserved = true;
            break;
          }
        }
      }
      if (!block) {
        uint64_t allocation_bytes = suballocation_bytes;
        if (!dedicated) {
          const uint32_t heap_index = impl_->device->memory.memoryTypes[type].heapIndex;
          const uint64_t heap_bytes = impl_->device->memory.memoryHeaps[heap_index].size;
          const uint64_t economical_block = std::max<uint64_t>(suballocation_bytes,
              std::min<uint64_t>(impl_->block_bytes,
                                 std::max<uint64_t>(heap_bytes / 8, suballocation_bytes)));
          allocation_bytes = detail::align_up(economical_block, suballocation_alignment);
        }
        block = impl_->make_block(allocation_bytes, type, dedicated, addressable, buffer);
        if (!Impl::take_range(*block, suballocation_bytes, suballocation_alignment, &offset)) {
          throw std::logic_error("vulkan: new memory block cannot satisfy its buffer");
        }
        span_reserved = true;
      }
    }
    detail::check(impl_->device->bind_buffer_memory(impl_->device->device, buffer,
                                                    block->memory, offset),
                  "vkBindBufferMemory");
    auto result = std::make_shared<Buffer::Impl>();
    result->pool = impl_;
    result->block = std::move(block);
    result->buffer = buffer;
    result->bytes = bytes;
    result->allocation_bytes = suballocation_bytes;
    result->offset = offset;
    result->usage = memory;
    result->buffer_usage = usage;
    return Buffer(std::move(result));
  } catch (...) {
    impl_->device->destroy_buffer(impl_->device->device, buffer, nullptr);
    if (span_reserved) impl_->release(block, offset, suballocation_bytes);
    throw;
  }
}

void BufferPool::trim() {
  if (!impl_) return;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->blocks.erase(std::remove_if(impl_->blocks.begin(), impl_->blocks.end(),
                                    [](const auto& block) { return block->used == 0; }),
                      impl_->blocks.end());
}
uint64_t BufferPool::reserved_bytes() const {
  if (!impl_) return 0;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  uint64_t total = 0;
  for (const auto& block : impl_->blocks) {
    if (block->bytes > std::numeric_limits<uint64_t>::max() - total) {
      throw std::overflow_error("vulkan: reserved byte count overflow");
    }
    total += block->bytes;
  }
  return total;
}
uint64_t BufferPool::used_bytes() const {
  if (!impl_) return 0;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  uint64_t total = 0;
  for (const auto& block : impl_->blocks) {
    if (block->used > std::numeric_limits<uint64_t>::max() - total) {
      throw std::overflow_error("vulkan: used byte count overflow");
    }
    total += block->used;
  }
  return total;
}

Buffer::Buffer() = default;
Buffer::~Buffer() = default;
Buffer::Buffer(Buffer&&) noexcept = default;
Buffer& Buffer::operator=(Buffer&&) noexcept = default;
Buffer::Buffer(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
uint64_t Buffer::size() const noexcept { return impl_ ? impl_->bytes : 0; }
BufferUsage Buffer::usage() const noexcept {
  return impl_ ? impl_->buffer_usage : static_cast<BufferUsage>(0);
}
MemoryUsage Buffer::memory_usage() const noexcept {
  return impl_ ? impl_->usage : MemoryUsage::kDevice;
}
void* Buffer::mapped_data() noexcept {
  if (!impl_ || impl_->block->mapped == nullptr) return nullptr;
  return static_cast<unsigned char*>(impl_->block->mapped) + impl_->offset;
}
const void* Buffer::mapped_data() const noexcept {
  if (!impl_ || impl_->block->mapped == nullptr) return nullptr;
  return static_cast<const unsigned char*>(impl_->block->mapped) + impl_->offset;
}
void Buffer::flush(uint64_t offset, uint64_t bytes) {
  if (!impl_) throw std::logic_error("vulkan: empty Buffer");
  if (bytes == ~uint64_t{0}) bytes = impl_->bytes - offset;
  impl_->check_range(offset, bytes);
  if (bytes == 0) return;
  if (!(impl_->block->properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
    throw std::logic_error("vulkan: device-local buffer is not mapped");
  }
  std::lock_guard<std::mutex> lock(impl_->block->mapped_mutex);
  impl_->flush_range(offset, bytes);
}
void Buffer::invalidate(uint64_t offset, uint64_t bytes) const {
  if (!impl_) throw std::logic_error("vulkan: empty Buffer");
  if (bytes == ~uint64_t{0}) bytes = impl_->bytes - offset;
  impl_->check_range(offset, bytes);
  if (bytes == 0) return;
  if (!(impl_->block->properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
    throw std::logic_error("vulkan: device-local buffer is not mapped");
  }
  std::lock_guard<std::mutex> lock(impl_->block->mapped_mutex);
  impl_->invalidate_range(offset, bytes);
}
void Buffer::write(uint64_t offset, const void* data, uint64_t bytes) {
  if (data == nullptr && bytes != 0) throw std::invalid_argument("vulkan: null buffer write source");
  if (mapped_data() == nullptr) throw std::logic_error("vulkan: buffer is not host-visible");
  impl_->check_range(offset, bytes);
  if (bytes == 0) return;
  std::lock_guard<std::mutex> lock(impl_->block->mapped_mutex);
  std::memcpy(static_cast<unsigned char*>(mapped_data()) + offset, data,
              static_cast<size_t>(bytes));
  impl_->flush_range(offset, bytes);
}
void Buffer::read(uint64_t offset, void* data, uint64_t bytes) const {
  if (data == nullptr && bytes != 0) throw std::invalid_argument("vulkan: null buffer read destination");
  if (mapped_data() == nullptr) throw std::logic_error("vulkan: buffer is not host-visible");
  impl_->check_range(offset, bytes);
  if (bytes == 0) return;
  std::lock_guard<std::mutex> lock(impl_->block->mapped_mutex);
  impl_->invalidate_range(offset, bytes);
  std::memcpy(data, static_cast<const unsigned char*>(mapped_data()) + offset,
              static_cast<size_t>(bytes));
}
void Buffer::reset_after_idle(const Queue& queue) {
  if (!impl_) return;
  if (!queue.impl_ || queue.impl_->state != impl_->pool->device) {
    throw std::invalid_argument("vulkan: buffer and queue belong to different devices");
  }
  queue.wait_idle();
  impl_.reset();
}
uintptr_t Buffer::native_handle() const noexcept {
  return impl_ ? reinterpret_cast<uintptr_t>(impl_->buffer) : 0;
}
uint64_t Buffer::memory_offset() const noexcept { return impl_ ? impl_->offset : 0; }
Buffer::operator bool() const noexcept { return impl_ != nullptr; }


}  // namespace slopfab::vulkan
