#pragma once
#include "runtime_buffer_state.h"

namespace slopfab::vulkan {
struct TimestampQuery::Impl {
  std::shared_ptr<detail::DeviceState> device;
  VkQueryPool pool = VK_NULL_HANDLE;
  uint32_t count = 0;
  uint32_t valid_bits = 0;
  float period = 0;
  PFN_vkDestroyQueryPool destroy = nullptr;
  PFN_vkGetQueryPoolResults results = nullptr;
  PFN_vkCmdResetQueryPool reset = nullptr;
  PFN_vkCmdWriteTimestamp write = nullptr;

  ~Impl() {
    if (pool)
      destroy(device->device, pool, nullptr);
  }
};

struct ComputePipeline::Impl {
  std::shared_ptr<detail::DeviceState> device;
  VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  ComputePipelineOptions options;
  DeviceInfo info;
  PFN_vkDestroyDescriptorSetLayout destroy_descriptor_layout = nullptr;
  PFN_vkDestroyPipelineLayout destroy_pipeline_layout = nullptr;
  PFN_vkDestroyPipeline destroy_pipeline = nullptr;

  ~Impl() {
    if (pipeline != VK_NULL_HANDLE)
      destroy_pipeline(device->device, pipeline, nullptr);
    if (pipeline_layout != VK_NULL_HANDLE) {
      destroy_pipeline_layout(device->device, pipeline_layout, nullptr);
    }
    if (descriptor_layout != VK_NULL_HANDLE) {
      destroy_descriptor_layout(device->device, descriptor_layout, nullptr);
    }
  }
};

namespace detail {

struct ComputeFns {
  PFN_vkCreateCommandPool create_command_pool = nullptr;
  PFN_vkDestroyCommandPool destroy_command_pool = nullptr;
  PFN_vkAllocateCommandBuffers allocate_command_buffers = nullptr;
  PFN_vkResetCommandBuffer reset_command_buffer = nullptr;
  PFN_vkBeginCommandBuffer begin_command_buffer = nullptr;
  PFN_vkEndCommandBuffer end_command_buffer = nullptr;
  PFN_vkCreateDescriptorPool create_descriptor_pool = nullptr;
  PFN_vkDestroyDescriptorPool destroy_descriptor_pool = nullptr;
  PFN_vkResetDescriptorPool reset_descriptor_pool = nullptr;
  PFN_vkAllocateDescriptorSets allocate_descriptor_sets = nullptr;
  PFN_vkUpdateDescriptorSets update_descriptor_sets = nullptr;
  PFN_vkCmdCopyBuffer cmd_copy_buffer = nullptr;
  PFN_vkCmdPipelineBarrier cmd_pipeline_barrier = nullptr;
  PFN_vkCmdBindPipeline cmd_bind_pipeline = nullptr;
  PFN_vkCmdBindDescriptorSets cmd_bind_descriptor_sets = nullptr;
  PFN_vkCmdPushConstants cmd_push_constants = nullptr;
  PFN_vkCmdDispatch cmd_dispatch = nullptr;
  PFN_vkCreateSemaphore create_semaphore = nullptr;
  PFN_vkDestroySemaphore destroy_semaphore = nullptr;
  PFN_vkGetSemaphoreCounterValue get_semaphore_counter = nullptr;
  PFN_vkWaitSemaphores wait_semaphores = nullptr;
  PFN_vkQueueSubmit queue_submit = nullptr;
};

inline ComputeFns load_compute_fns(const std::shared_ptr<DeviceState>& device) {
  ComputeFns f;
#define SLOPFAB_LOAD_DEVICE(member, type, name)                                                    \
  f.member = load_device<type>(*device->instance, device->device, name)
  SLOPFAB_LOAD_DEVICE(create_command_pool, PFN_vkCreateCommandPool, "vkCreateCommandPool");
  SLOPFAB_LOAD_DEVICE(destroy_command_pool, PFN_vkDestroyCommandPool, "vkDestroyCommandPool");
  SLOPFAB_LOAD_DEVICE(allocate_command_buffers, PFN_vkAllocateCommandBuffers,
                      "vkAllocateCommandBuffers");
  SLOPFAB_LOAD_DEVICE(reset_command_buffer, PFN_vkResetCommandBuffer, "vkResetCommandBuffer");
  SLOPFAB_LOAD_DEVICE(begin_command_buffer, PFN_vkBeginCommandBuffer, "vkBeginCommandBuffer");
  SLOPFAB_LOAD_DEVICE(end_command_buffer, PFN_vkEndCommandBuffer, "vkEndCommandBuffer");
  SLOPFAB_LOAD_DEVICE(create_descriptor_pool, PFN_vkCreateDescriptorPool, "vkCreateDescriptorPool");
  SLOPFAB_LOAD_DEVICE(destroy_descriptor_pool, PFN_vkDestroyDescriptorPool,
                      "vkDestroyDescriptorPool");
  SLOPFAB_LOAD_DEVICE(reset_descriptor_pool, PFN_vkResetDescriptorPool, "vkResetDescriptorPool");
  SLOPFAB_LOAD_DEVICE(allocate_descriptor_sets, PFN_vkAllocateDescriptorSets,
                      "vkAllocateDescriptorSets");
  SLOPFAB_LOAD_DEVICE(update_descriptor_sets, PFN_vkUpdateDescriptorSets, "vkUpdateDescriptorSets");
  SLOPFAB_LOAD_DEVICE(cmd_copy_buffer, PFN_vkCmdCopyBuffer, "vkCmdCopyBuffer");
  SLOPFAB_LOAD_DEVICE(cmd_pipeline_barrier, PFN_vkCmdPipelineBarrier, "vkCmdPipelineBarrier");
  SLOPFAB_LOAD_DEVICE(cmd_bind_pipeline, PFN_vkCmdBindPipeline, "vkCmdBindPipeline");
  SLOPFAB_LOAD_DEVICE(cmd_bind_descriptor_sets, PFN_vkCmdBindDescriptorSets,
                      "vkCmdBindDescriptorSets");
  SLOPFAB_LOAD_DEVICE(cmd_push_constants, PFN_vkCmdPushConstants, "vkCmdPushConstants");
  SLOPFAB_LOAD_DEVICE(cmd_dispatch, PFN_vkCmdDispatch, "vkCmdDispatch");
  SLOPFAB_LOAD_DEVICE(create_semaphore, PFN_vkCreateSemaphore, "vkCreateSemaphore");
  SLOPFAB_LOAD_DEVICE(destroy_semaphore, PFN_vkDestroySemaphore, "vkDestroySemaphore");
  SLOPFAB_LOAD_DEVICE(get_semaphore_counter, PFN_vkGetSemaphoreCounterValue,
                      "vkGetSemaphoreCounterValue");
  SLOPFAB_LOAD_DEVICE(wait_semaphores, PFN_vkWaitSemaphores, "vkWaitSemaphores");
  SLOPFAB_LOAD_DEVICE(queue_submit, PFN_vkQueueSubmit, "vkQueueSubmit");
#undef SLOPFAB_LOAD_DEVICE
  return f;
}

struct ComputeSlot {
  VkCommandPool command_pool = VK_NULL_HANDLE;
  VkCommandBuffer commands = VK_NULL_HANDLE;
  VkDescriptorPool descriptors = VK_NULL_HANDLE;
  std::vector<VkDescriptorSet> descriptor_sets;
  std::vector<std::shared_ptr<void>> descriptor_pipelines;
  uint64_t value = 0;
  bool reserved = false;
  bool poisoned = false;
  std::vector<std::shared_ptr<void>> resources;
  std::vector<uint8_t> seen_bindings;
  std::vector<VkDescriptorBufferInfo> descriptor_infos;
  std::vector<VkWriteDescriptorSet> descriptor_writes;
};

struct ComputeState : std::enable_shared_from_this<ComputeState> {
  std::shared_ptr<DeviceState> device;
  ComputeFns f;
  VkSemaphore timeline = VK_NULL_HANDLE;
  uint32_t max_bindings = 0;
  uint32_t max_compute_binds = 0;
  std::atomic<uint64_t> descriptor_allocations{0};
  uint64_t next_value = 1;
  std::vector<ComputeSlot> slots;
  mutable std::mutex mutex;

  ~ComputeState() {
    if (timeline != VK_NULL_HANDLE && next_value > 1 && f.wait_semaphores != nullptr) {
      VkSemaphoreWaitInfo wait{};
      wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
      wait.semaphoreCount = 1;
      wait.pSemaphores = &timeline;
      const uint64_t tail = next_value - 1;
      wait.pValues = &tail;
      f.wait_semaphores(device->device, &wait, std::numeric_limits<uint64_t>::max());
    }
    for (auto& slot : slots) {
      slot.resources.clear();
      if (slot.descriptors != VK_NULL_HANDLE) {
        f.destroy_descriptor_pool(device->device, slot.descriptors, nullptr);
      }
      slot.descriptor_pipelines.clear();
      if (slot.command_pool != VK_NULL_HANDLE) {
        f.destroy_command_pool(device->device, slot.command_pool, nullptr);
      }
    }
    if (timeline != VK_NULL_HANDLE)
      f.destroy_semaphore(device->device, timeline, nullptr);
  }

  uint64_t completed() const {
    uint64_t value = 0;
    check(f.get_semaphore_counter(device->device, timeline, &value), "vkGetSemaphoreCounterValue");
    return value;
  }

  void wait_value(uint64_t value) const {
    VkSemaphoreWaitInfo wait{};
    wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait.semaphoreCount = 1;
    wait.pSemaphores = &timeline;
    wait.pValues = &value;
    check(f.wait_semaphores(device->device, &wait, std::numeric_limits<uint64_t>::max()),
          "vkWaitSemaphores");
  }

  void recycle_locked(uint64_t completed_value) {
    for (auto& slot : slots) {
      if (!slot.reserved || slot.value == 0 || slot.value > completed_value)
        continue;
      slot.resources.clear();
      check(f.reset_command_buffer(slot.commands, 0), "vkResetCommandBuffer");
      slot.reserved = false;
      slot.value = 0;
    }
  }
};

struct AccessInfo {
  VkPipelineStageFlags stage;
  VkAccessFlags access;
};

inline AccessInfo access_info(BufferAccess access) {
  switch (access) {
  case BufferAccess::kHostWrite:
    return {VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_WRITE_BIT};
  case BufferAccess::kHostRead:
    return {VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT};
  case BufferAccess::kTransferRead:
    return {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT};
  case BufferAccess::kTransferWrite:
    return {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT};
  case BufferAccess::kComputeRead:
    return {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT};
  case BufferAccess::kComputeWrite:
    return {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT};
  case BufferAccess::kComputeReadWrite:
    return {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
  }
  throw std::logic_error("vulkan: invalid buffer access");
}

inline bool has_usage(BufferUsage value, BufferUsage flag) {
  return (static_cast<uint32_t>(value) & static_cast<uint32_t>(flag)) != 0;
}

struct SpirvShape {
  uint32_t local_size[3] = {};
  std::vector<uint32_t> bindings;
  bool has_local_size = false;
};

inline SpirvShape inspect_spirv(const std::vector<uint32_t>& words) {
  SpirvShape shape;
  for (size_t at = 5; at < words.size();) {
    const uint16_t count = static_cast<uint16_t>(words[at] >> 16);
    const uint16_t opcode = static_cast<uint16_t>(words[at] & 0xffffu);
    if (count == 0 || at + count > words.size()) {
      throw std::invalid_argument("vulkan: malformed SPIR-V instruction stream");
    }
    // OpExecutionMode %entry LocalSize x y z
    if (opcode == 16 && count >= 6 && words[at + 2] == 17) {
      shape.local_size[0] = words[at + 3];
      shape.local_size[1] = words[at + 4];
      shape.local_size[2] = words[at + 5];
      shape.has_local_size = true;
    }
    // OpDecorate %target Binding binding-number
    if (opcode == 71 && count >= 4 && words[at + 2] == 33) {
      shape.bindings.push_back(words[at + 3]);
    }
    at += count;
  }
  std::sort(shape.bindings.begin(), shape.bindings.end());
  shape.bindings.erase(std::unique(shape.bindings.begin(), shape.bindings.end()),
                       shape.bindings.end());
  return shape;
}

} // namespace detail

struct ComputeContext::Impl {
  std::shared_ptr<detail::ComputeState> state;
};

struct Submission::Impl {
  std::shared_ptr<detail::ComputeState> state;
  uint64_t value = 0;
};

struct CommandList::Impl {
  std::shared_ptr<detail::ComputeState> state;
  size_t slot = 0;
  bool recording = false;
  uint32_t compute_bind_count = 0;
  bool push_constants_set = false;
  std::shared_ptr<ComputePipeline::Impl> pipeline;
  std::vector<std::shared_ptr<void>> resources;

  void retain(const std::shared_ptr<void>& resource) {
    if (std::find(resources.begin(), resources.end(), resource) == resources.end()) {
      resources.push_back(resource);
    }
  }

  ~Impl() {
    if (!recording || !state)
      return;
    std::lock_guard<std::mutex> lock(state->mutex);
    auto& owned = state->slots[slot];
    if (state->f.reset_command_buffer(owned.commands, 0) != VK_SUCCESS) {
      owned.poisoned = true;
    }
    owned.resources.swap(resources);
    owned.resources.clear();
    owned.reserved = false;
    owned.value = 0;
  }
};

} // namespace slopfab::vulkan
