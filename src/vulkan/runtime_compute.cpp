#include "runtime_compute_state.h"

namespace slopfab::vulkan {
ComputeContext::ComputeContext(const Device& device, const ComputeContextOptions& options) {
  if (!device.impl_)
    throw std::invalid_argument("vulkan: ComputeContext requires a device");
  if (!device.impl_->state->timeline_semaphore_enabled) {
    throw std::invalid_argument("vulkan: ComputeContext requires timeline semaphores enabled");
  }
  if (options.max_in_flight == 0 || options.max_storage_bindings == 0 ||
      options.max_compute_binds_per_job == 0) {
    throw std::invalid_argument("vulkan: compute context limits must be nonzero");
  }
  const uint64_t job_descriptor_capacity =
      static_cast<uint64_t>(options.max_storage_bindings) * options.max_compute_binds_per_job;
  const uint64_t cached_set_capacity =
      static_cast<uint64_t>(options.max_storage_bindings) * options.max_compute_binds_per_job;
  const uint64_t cached_descriptor_capacity =
      static_cast<uint64_t>(options.max_compute_binds_per_job) * options.max_storage_bindings *
      (options.max_storage_bindings + 1ull) / 2ull;
  if (job_descriptor_capacity > std::numeric_limits<uint32_t>::max() ||
      cached_set_capacity > std::numeric_limits<uint32_t>::max() ||
      cached_descriptor_capacity > std::numeric_limits<uint32_t>::max()) {
    throw std::invalid_argument("vulkan: compute descriptor capacity overflows uint32");
  }
  auto state = std::make_shared<detail::ComputeState>();
  state->device = device.impl_->state;
  state->f = detail::load_compute_fns(state->device);
  state->max_bindings = options.max_storage_bindings;
  state->max_compute_binds = options.max_compute_binds_per_job;

  try {
    state->slots.resize(options.max_in_flight);
    for (uint32_t i = 0; i < options.max_in_flight; ++i) {
      auto& slot = state->slots[i];
      slot.resources.reserve(static_cast<size_t>(job_descriptor_capacity) +
                             options.max_compute_binds_per_job + 8);
      slot.seen_bindings.reserve(options.max_storage_bindings);
      slot.descriptor_infos.reserve(options.max_storage_bindings);
      slot.descriptor_writes.reserve(options.max_storage_bindings);
      slot.descriptor_sets.resize(static_cast<size_t>(cached_set_capacity), VK_NULL_HANDLE);
      slot.descriptor_pipelines.resize(static_cast<size_t>(cached_set_capacity));
      VkCommandPoolCreateInfo pool_create{};
      pool_create.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
      pool_create.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
      pool_create.queueFamilyIndex = state->device->queue_family;
      detail::check(state->f.create_command_pool(state->device->device, &pool_create, nullptr,
                                                 &slot.command_pool),
                    "vkCreateCommandPool");
      VkCommandBufferAllocateInfo command_allocate{};
      command_allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      command_allocate.commandPool = slot.command_pool;
      command_allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      command_allocate.commandBufferCount = 1;
      detail::check(state->f.allocate_command_buffers(state->device->device, &command_allocate,
                                                      &slot.commands),
                    "vkAllocateCommandBuffers");
      VkDescriptorPoolSize size{};
      size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      size.descriptorCount = static_cast<uint32_t>(cached_descriptor_capacity);
      VkDescriptorPoolCreateInfo descriptor_create{};
      descriptor_create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      descriptor_create.maxSets = static_cast<uint32_t>(cached_set_capacity);
      descriptor_create.poolSizeCount = 1;
      descriptor_create.pPoolSizes = &size;
      detail::check(state->f.create_descriptor_pool(state->device->device, &descriptor_create,
                                                    nullptr, &slot.descriptors),
                    "vkCreateDescriptorPool");
    }
    VkSemaphoreTypeCreateInfo timeline_type{};
    timeline_type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timeline_type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphore_create{};
    semaphore_create.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphore_create.pNext = &timeline_type;
    detail::check(state->f.create_semaphore(state->device->device, &semaphore_create, nullptr,
                                            &state->timeline),
                  "vkCreateSemaphore");
  } catch (...) {
    state.reset();
    throw;
  }
  impl_ = std::make_shared<Impl>();
  impl_->state = std::move(state);
}

ComputeContext::~ComputeContext() = default;
ComputeContext::ComputeContext(ComputeContext&&) noexcept = default;
ComputeContext& ComputeContext::operator=(ComputeContext&&) noexcept = default;

CommandList ComputeContext::begin() {
  if (!impl_)
    throw std::logic_error("vulkan: empty ComputeContext");
  auto state = impl_->state;
  size_t selected = state->slots.size();
  for (;;) {
    uint64_t wait_for = std::numeric_limits<uint64_t>::max();
    {
      std::lock_guard<std::mutex> lock(state->mutex);
      state->recycle_locked(state->completed());
      for (size_t i = 0; i < state->slots.size(); ++i) {
        if (!state->slots[i].reserved && !state->slots[i].poisoned) {
          selected = i;
          break;
        }
        if (state->slots[i].value != 0)
          wait_for = std::min(wait_for, state->slots[i].value);
      }
      if (selected != state->slots.size()) {
        state->slots[selected].reserved = true;
        break;
      }
    }
    if (wait_for == std::numeric_limits<uint64_t>::max()) {
      throw std::logic_error("vulkan: no usable command slot is available");
    }
    state->wait_value(wait_for); // bounded backpressure: oldest submitted slot
  }
  VkCommandBufferBeginInfo begin{};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  std::unique_ptr<CommandList::Impl> commands;
  try {
    // Allocate host ownership before beginning Vulkan recording. If this
    // allocation fails the reserved slot is still in its clean initial state.
    commands = std::make_unique<CommandList::Impl>();
    commands->state = state;
    commands->slot = selected;
    commands->resources.swap(state->slots[selected].resources);
    detail::check(state->f.begin_command_buffer(state->slots[selected].commands, &begin),
                  "vkBeginCommandBuffer");
    commands->recording = true;
    return CommandList(std::move(commands));
  } catch (...) {
    std::lock_guard<std::mutex> lock(state->mutex);
    // vkBeginCommandBuffer failure leaves state unspecified; resetting makes
    // the slot usable regardless of whether recording actually began.
    if (state->f.reset_command_buffer(state->slots[selected].commands, 0) != VK_SUCCESS) {
      state->slots[selected].poisoned = true;
    }
    if (commands)
      state->slots[selected].resources.swap(commands->resources);
    state->slots[selected].reserved = false;
    throw;
  }
}

Submission ComputeContext::submit(CommandList&& commands) {
  if (!impl_ || !commands.impl_)
    throw std::invalid_argument("vulkan: empty command submission");
  if (commands.impl_->state != impl_->state) {
    throw std::invalid_argument("vulkan: command list belongs to another context");
  }
  auto state = impl_->state;
  auto& slot = state->slots[commands.impl_->slot];
  detail::check(state->f.end_command_buffer(slot.commands), "vkEndCommandBuffer");
  uint64_t value = 0;
  VkTimelineSemaphoreSubmitInfo timeline{};
  timeline.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
  timeline.signalSemaphoreValueCount = 1;
  VkSubmitInfo submit{};
  submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.pNext = &timeline;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &slot.commands;
  submit.signalSemaphoreCount = 1;
  submit.pSignalSemaphores = &state->timeline;
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    value = state->next_value;
    timeline.pSignalSemaphoreValues = &value;
    std::lock_guard<std::mutex> queue_lock(state->device->queue_mutex);
    detail::check(state->f.queue_submit(state->device->queue, 1, &submit, VK_NULL_HANDLE),
                  "vkQueueSubmit");
    ++state->next_value;
    slot.value = value;
    slot.resources = std::move(commands.impl_->resources);
  }
  commands.impl_->recording = false;
  commands.impl_.reset();
  auto token = std::make_shared<Submission::Impl>();
  token->state = std::move(state);
  token->value = value;
  return Submission(std::move(token));
}

void ComputeContext::collect() {
  if (!impl_)
    return;
  std::lock_guard<std::mutex> lock(impl_->state->mutex);
  impl_->state->recycle_locked(impl_->state->completed());
}

uint32_t ComputeContext::in_flight() const {
  if (!impl_)
    return 0;
  std::lock_guard<std::mutex> lock(impl_->state->mutex);
  uint32_t count = 0;
  for (const auto& slot : impl_->state->slots)
    if (slot.reserved)
      ++count;
  return count;
}

uint64_t ComputeContext::descriptor_set_allocations() const noexcept {
  return impl_ ? impl_->state->descriptor_allocations.load(std::memory_order_relaxed) : 0;
}

Submission::Submission() = default;

Submission::Submission(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {
}

Submission::operator bool() const noexcept {
  return impl_ != nullptr;
}

uint64_t Submission::value() const noexcept {
  return impl_ ? impl_->value : 0;
}

bool Submission::ready() const {
  if (!impl_)
    return false;
  return impl_->state->completed() >= impl_->value;
}

void Submission::wait() const {
  if (!impl_)
    throw std::logic_error("vulkan: empty Submission");
  impl_->state->wait_value(impl_->value);
}

CommandList::CommandList() = default;
CommandList::~CommandList() = default;
CommandList::CommandList(CommandList&&) noexcept = default;
CommandList& CommandList::operator=(CommandList&&) noexcept = default;

CommandList::CommandList(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {
}

CommandList::operator bool() const noexcept {
  return impl_ != nullptr;
}

void CommandList::reset_timestamps(TimestampQuery& queries) {
  if (!impl_ || !queries.impl_ || queries.impl_->device != impl_->state->device)
    throw std::invalid_argument("vulkan timestamps: incompatible query pool");
  impl_->retain(queries.impl_);
  queries.impl_->reset(impl_->state->slots[impl_->slot].commands, queries.impl_->pool, 0,
                       queries.impl_->count);
}

void CommandList::write_timestamp(TimestampQuery& queries, uint32_t index) {
  if (!impl_ || !queries.impl_ || queries.impl_->device != impl_->state->device ||
      index >= queries.impl_->count)
    throw std::invalid_argument("vulkan timestamps: incompatible query or index");
  impl_->retain(queries.impl_);
  queries.impl_->write(impl_->state->slots[impl_->slot].commands,
                       VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries.impl_->pool, index);
}

void CommandList::copy_buffer(Buffer& source, Buffer& destination, uint64_t bytes,
                              uint64_t source_offset, uint64_t destination_offset) {
  if (!impl_ || !source.impl_ || !destination.impl_) {
    throw std::invalid_argument("vulkan: copy requires live command list and buffers");
  }
  if (source.impl_->pool->device != impl_->state->device ||
      destination.impl_->pool->device != impl_->state->device) {
    throw std::invalid_argument("vulkan: copy buffer belongs to another device");
  }
  if (!detail::has_usage(source.impl_->buffer_usage, BufferUsage::kTransferSource) ||
      !detail::has_usage(destination.impl_->buffer_usage, BufferUsage::kTransferDestination)) {
    throw std::invalid_argument("vulkan: copy buffer lacks transfer usage");
  }
  source.impl_->check_range(source_offset, bytes);
  destination.impl_->check_range(destination_offset, bytes);
  if (bytes == 0)
    throw std::invalid_argument("vulkan: zero-sized copy");
  if (((source_offset | destination_offset | bytes) & 3u) != 0) {
    throw std::invalid_argument("vulkan: copy offsets and size must be four-byte aligned");
  }
  if (source.impl_->buffer == destination.impl_->buffer &&
      source_offset < destination_offset + bytes && destination_offset < source_offset + bytes) {
    throw std::invalid_argument("vulkan: same-buffer copy ranges overlap");
  }
  impl_->resources.reserve(impl_->resources.size() + 2);
  impl_->retain(source.impl_);
  impl_->retain(destination.impl_);
  VkBufferCopy region{source_offset, destination_offset, bytes};
  impl_->state->f.cmd_copy_buffer(impl_->state->slots[impl_->slot].commands, source.impl_->buffer,
                                  destination.impl_->buffer, 1, &region);
}

void CommandList::barrier(Buffer& buffer, BufferAccess before, BufferAccess after, uint64_t offset,
                          uint64_t bytes) {
  if (!impl_ || !buffer.impl_)
    throw std::invalid_argument("vulkan: barrier requires buffer");
  if (buffer.impl_->pool->device != impl_->state->device) {
    throw std::invalid_argument("vulkan: barrier buffer belongs to another device");
  }
  if (bytes == ~uint64_t{0})
    bytes = buffer.impl_->bytes - offset;
  buffer.impl_->check_range(offset, bytes);
  if (bytes == 0)
    throw std::invalid_argument("vulkan: zero-sized barrier");
  const auto src = detail::access_info(before);
  const auto dst = detail::access_info(after);
  impl_->resources.reserve(impl_->resources.size() + 1);
  impl_->retain(buffer.impl_);
  VkBufferMemoryBarrier barrier{};
  barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  barrier.srcAccessMask = src.access;
  barrier.dstAccessMask = dst.access;
  barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.buffer = buffer.impl_->buffer;
  barrier.offset = offset;
  barrier.size = bytes;
  impl_->state->f.cmd_pipeline_barrier(impl_->state->slots[impl_->slot].commands, src.stage,
                                       dst.stage, 0, 0, nullptr, 1, &barrier, 0, nullptr);
}

void CommandList::bind_compute(ComputePipeline& pipeline,
                               const std::vector<StorageBinding>& bindings) {
  if (!impl_ || !pipeline.impl_)
    throw std::invalid_argument("vulkan: bind requires a supported, prepared pipeline set");
  if (pipeline.impl_->device != impl_->state->device) {
    throw std::invalid_argument("vulkan: pipeline belongs to another device");
  }
  if (impl_->compute_bind_count >= impl_->state->max_compute_binds) {
    throw std::logic_error("vulkan: compute binds exceed per-job limit");
  }
  if (bindings.size() != pipeline.impl_->options.storage_binding_count ||
      bindings.size() > impl_->state->max_bindings) {
    throw std::invalid_argument("vulkan: storage binding count mismatch");
  }
  auto& slot = impl_->state->slots[impl_->slot];
  slot.seen_bindings.assign(bindings.size(), 0);
  slot.descriptor_infos.resize(bindings.size());
  slot.descriptor_writes.resize(bindings.size());
  for (size_t i = 0; i < bindings.size(); ++i) {
    const auto& binding = bindings[i];
    if (binding.binding >= bindings.size() || slot.seen_bindings[binding.binding] ||
        binding.buffer == nullptr || !binding.buffer->impl_) {
      throw std::invalid_argument("vulkan: invalid or duplicate storage binding");
    }
    slot.seen_bindings[binding.binding] = 1;
    if (binding.buffer->impl_->pool->device != impl_->state->device ||
        !detail::has_usage(binding.buffer->impl_->buffer_usage, BufferUsage::kStorage)) {
      throw std::invalid_argument("vulkan: storage binding has wrong device or usage");
    }
    uint64_t range = binding.bytes;
    if (range == ~uint64_t{0})
      range = binding.buffer->impl_->bytes - binding.offset;
    binding.buffer->impl_->check_range(binding.offset, range);
    if (range == 0 || range > pipeline.impl_->info.max_storage_buffer_bytes) {
      throw std::invalid_argument("vulkan: storage binding range exceeds device limit");
    }
    if ((binding.offset % pipeline.impl_->info.min_storage_buffer_offset_alignment) != 0) {
      throw std::invalid_argument("vulkan: storage binding offset is misaligned");
    }
    slot.descriptor_infos[i] = {binding.buffer->impl_->buffer, binding.offset, range};
    slot.descriptor_writes[i] = {};
    slot.descriptor_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    slot.descriptor_writes[i].dstBinding = binding.binding;
    slot.descriptor_writes[i].descriptorCount = 1;
    slot.descriptor_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    slot.descriptor_writes[i].pBufferInfo = &slot.descriptor_infos[i];
  }
  // Retain only after every binding has validated. This second small pass is
  // allocation-free because the job vector is pre-reserved from its slot.
  impl_->resources.reserve(impl_->resources.size() + bindings.size() + 1);
  for (const auto& binding : bindings)
    impl_->retain(binding.buffer->impl_);
  impl_->retain(pipeline.impl_);
  const uint32_t bind_index = impl_->compute_bind_count;
  // All layouts contain only contiguous storage-buffer bindings, so their
  // binding count is a complete compatibility key. Caching one set for each
  // bounded (bind position, layout count) pair supports mixed operator graphs
  // without resetting a pool referenced earlier in the command buffer and
  // without steady-state descriptor allocation.
  const uint32_t binding_count = pipeline.impl_->options.storage_binding_count;
  const size_t set_index =
      static_cast<size_t>(bind_index) * impl_->state->max_bindings + (binding_count - 1);
  if (slot.descriptor_sets[set_index] == VK_NULL_HANDLE) {
    VkDescriptorSetAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate.descriptorPool = slot.descriptors;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &pipeline.impl_->descriptor_layout;
    detail::check(impl_->state->f.allocate_descriptor_sets(impl_->state->device->device, &allocate,
                                                           &slot.descriptor_sets[set_index]),
                  "vkAllocateDescriptorSets");
    impl_->state->descriptor_allocations.fetch_add(1, std::memory_order_relaxed);
    slot.descriptor_pipelines[set_index] = pipeline.impl_;
  }
  for (auto& write : slot.descriptor_writes) {
    write.dstSet = slot.descriptor_sets[set_index];
  }
  impl_->state->f.update_descriptor_sets(impl_->state->device->device,
                                         static_cast<uint32_t>(slot.descriptor_writes.size()),
                                         slot.descriptor_writes.data(), 0, nullptr);
  const VkCommandBuffer commands = slot.commands;
  impl_->state->f.cmd_bind_pipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline.impl_->pipeline);
  impl_->state->f.cmd_bind_descriptor_sets(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                                           pipeline.impl_->pipeline_layout, 0, 1,
                                           &slot.descriptor_sets[set_index], 0, nullptr);
  impl_->pipeline = pipeline.impl_;
  impl_->push_constants_set = false;
  ++impl_->compute_bind_count;
}

void CommandList::push_constants(const void* data, uint32_t bytes) {
  if (!impl_ || !impl_->pipeline)
    throw std::logic_error("vulkan: bind pipeline before push constants");
  if (bytes != impl_->pipeline->options.push_constant_bytes || (data == nullptr && bytes != 0)) {
    throw std::invalid_argument("vulkan: push constant size mismatch");
  }
  if (bytes != 0) {
    impl_->state->f.cmd_push_constants(impl_->state->slots[impl_->slot].commands,
                                       impl_->pipeline->pipeline_layout,
                                       VK_SHADER_STAGE_COMPUTE_BIT, 0, bytes, data);
  }
  impl_->push_constants_set = true;
}

void CommandList::dispatch(uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
  if (!impl_ || !impl_->pipeline)
    throw std::logic_error("vulkan: bind pipeline before dispatch");
  if (impl_->pipeline->options.push_constant_bytes != 0 && !impl_->push_constants_set) {
    throw std::logic_error("vulkan: set required push constants before dispatch");
  }
  const uint32_t groups[3] = {groups_x, groups_y, groups_z};
  // Group-count limits come from the same physical device retained by the
  // pipeline; validate nonzero here, while vkCmdDispatch validates no state.
  for (int i = 0; i < 3; ++i) {
    if (groups[i] == 0 || groups[i] > impl_->pipeline->info.max_compute_workgroup_count[i]) {
      throw std::invalid_argument("vulkan: dispatch group count exceeds device limit");
    }
  }
  impl_->state->f.cmd_dispatch(impl_->state->slots[impl_->slot].commands, groups_x, groups_y,
                               groups_z);
}

} // namespace slopfab::vulkan
