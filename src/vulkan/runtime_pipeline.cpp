#include "runtime_compute_state.h"

namespace slopfab::vulkan {
TimestampQuery TimestampQuery::create(const Device& device, uint32_t count) {
  if (!device.impl_ || count == 0 || count > 256)
    throw std::invalid_argument("vulkan timestamps: count must be in [1,256]");
  if (!device.info().timestamp_valid_bits || device.info().timestamp_period_ns <= 0)
    throw std::runtime_error("vulkan timestamps: compute queue does not support timestamps");
  TimestampQuery query;
  query.impl_ = std::make_shared<Impl>();
  auto& s = *query.impl_;
  s.device = device.impl_->state;
  s.count = count;
  s.valid_bits = device.info().timestamp_valid_bits;
  s.period = device.info().timestamp_period_ns;
#define QUERY_FN(TYPE, NAME) detail::load_device<TYPE>(*s.device->instance, s.device->device, NAME)
  s.destroy = QUERY_FN(PFN_vkDestroyQueryPool, "vkDestroyQueryPool");
  s.results = QUERY_FN(PFN_vkGetQueryPoolResults, "vkGetQueryPoolResults");
  s.reset = QUERY_FN(PFN_vkCmdResetQueryPool, "vkCmdResetQueryPool");
  s.write = QUERY_FN(PFN_vkCmdWriteTimestamp, "vkCmdWriteTimestamp");
  const auto create = QUERY_FN(PFN_vkCreateQueryPool, "vkCreateQueryPool");
#undef QUERY_FN
  VkQueryPoolCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  info.queryCount = count;
  detail::check(create(s.device->device, &info, nullptr, &s.pool), "vkCreateQueryPool");
  return query;
}

uint32_t TimestampQuery::count() const noexcept {
  return impl_ ? impl_->count : 0;
}

double TimestampQuery::elapsed_milliseconds(uint32_t first, uint32_t last) const {
  if (!impl_ || first >= last || last >= impl_->count)
    throw std::invalid_argument("vulkan timestamps: invalid interval");
  uint64_t begin = 0, end = 0;
  detail::check(impl_->results(impl_->device->device, impl_->pool, first, 1, sizeof(begin), &begin,
                               sizeof(begin), VK_QUERY_RESULT_64_BIT),
                "vkGetQueryPoolResults(begin)");
  detail::check(impl_->results(impl_->device->device, impl_->pool, last, 1, sizeof(end), &end,
                               sizeof(end), VK_QUERY_RESULT_64_BIT),
                "vkGetQueryPoolResults(end)");
  const uint64_t mask =
      impl_->valid_bits == 64 ? ~uint64_t(0) : (uint64_t(1) << impl_->valid_bits) - 1;
  return double((end - begin) & mask) * impl_->period / 1.0e6;
}

ComputePipeline::ComputePipeline() = default;
ComputePipeline::~ComputePipeline() = default;
ComputePipeline::ComputePipeline(ComputePipeline&&) noexcept = default;
ComputePipeline& ComputePipeline::operator=(ComputePipeline&&) noexcept = default;

ComputePipeline::ComputePipeline(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {
}

ComputePipeline::operator bool() const noexcept {
  return impl_ != nullptr;
}

uint32_t ComputePipeline::storage_binding_count() const noexcept {
  return impl_ ? impl_->options.storage_binding_count : 0;
}

uint32_t ComputePipeline::push_constant_bytes() const noexcept {
  return impl_ ? impl_->options.push_constant_bytes : 0;
}

ComputePipeline ComputePipeline::create(const Device& device, const std::vector<uint32_t>& spirv,
                                        const ComputePipelineOptions& options) {
  if (!device.impl_)
    throw std::invalid_argument("vulkan: ComputePipeline requires a device");
  if (spirv.size() < 5 || spirv[0] != 0x07230203u) {
    throw std::invalid_argument("vulkan: invalid SPIR-V module");
  }
  if (options.storage_binding_count == 0) {
    throw std::invalid_argument("vulkan: compute pipeline has no storage bindings");
  }
  if (options.storage_binding_count > device.info().max_storage_buffer_bindings) {
    throw std::invalid_argument("vulkan: storage binding count exceeds device limit");
  }
  const detail::SpirvShape shape = detail::inspect_spirv(spirv);
  if (!shape.has_local_size) {
    throw std::invalid_argument("vulkan: SPIR-V has no literal LocalSize execution mode");
  }
  for (int i = 0; i < 3; ++i) {
    if (shape.local_size[i] != options.local_size[i]) {
      throw std::invalid_argument("vulkan: declared local size does not match SPIR-V");
    }
  }
  if (shape.bindings.size() != options.storage_binding_count) {
    throw std::invalid_argument("vulkan: declared storage binding count does not match SPIR-V");
  }
  for (uint32_t i = 0; i < options.storage_binding_count; ++i) {
    if (shape.bindings[i] != i) {
      throw std::invalid_argument("vulkan: SPIR-V storage bindings must be contiguous from zero");
    }
  }
  if (options.push_constant_bytes > device.info().max_push_constant_bytes ||
      (options.push_constant_bytes & 3u) != 0) {
    throw std::invalid_argument("vulkan: invalid push constant size");
  }
  uint64_t invocations = 1;
  for (int i = 0; i < 3; ++i) {
    if (options.local_size[i] == 0 ||
        options.local_size[i] > device.info().max_compute_workgroup_size[i]) {
      throw std::invalid_argument("vulkan: compute local size exceeds device limit");
    }
    invocations *= options.local_size[i];
  }
  if (invocations > device.info().max_compute_workgroup_invocations) {
    throw std::invalid_argument("vulkan: compute local invocation count exceeds device limit");
  }
  std::vector<VkSpecializationMapEntry> specialization_entries;
  std::vector<uint32_t> specialization_values;
  for (const auto& constant : options.specialization_constants) {
    for (const auto& entry : specialization_entries) {
      if (entry.constantID == constant.id)
        throw std::invalid_argument("vulkan: duplicate specialization constant ID");
    }
    specialization_entries.push_back(
        {constant.id, static_cast<uint32_t>(specialization_values.size() * sizeof(uint32_t)),
         sizeof(uint32_t)});
    specialization_values.push_back(constant.value);
  }
  VkSpecializationInfo specialization{};
  specialization.mapEntryCount = static_cast<uint32_t>(specialization_entries.size());
  specialization.pMapEntries = specialization_entries.data();
  specialization.dataSize = specialization_values.size() * sizeof(uint32_t);
  specialization.pData = specialization_values.data();

  // The cache key contains the complete module and every semantic option.
  // Arithmetic variants live in different modules; no driver/name heuristic
  // can accidentally reuse a pipeline compiled for another contract.
  std::string cache_key;
  auto append = [&](const void* data, size_t bytes) {
    cache_key.append(static_cast<const char*>(data), bytes);
  };
  auto append_size = [&](size_t size) {
    append(&size, sizeof(size));
  };
  append_size(spirv.size());
  append(spirv.data(), spirv.size() * sizeof(uint32_t));
  append(&options.storage_binding_count, sizeof(options.storage_binding_count));
  append(&options.push_constant_bytes, sizeof(options.push_constant_bytes));
  append(options.local_size, sizeof(options.local_size));
  append_size(options.entry_point.size());
  append(options.entry_point.data(), options.entry_point.size());
  append_size(options.specialization_constants.size());
  for (const auto& constant : options.specialization_constants) {
    append(&constant.id, sizeof(constant.id));
    append(&constant.value, sizeof(constant.value));
  }
  auto& device_state = *device.impl_->state;
  std::lock_guard<std::mutex> cache_lock(device_state.pipeline_mutex);
  if (auto found = device_state.pipeline_cache.find(cache_key);
      found != device_state.pipeline_cache.end()) {
    if (auto cached = found->second.lock()) {
      device_state.pipeline_cache_hits.fetch_add(1, std::memory_order_relaxed);
      return ComputePipeline(std::static_pointer_cast<Impl>(cached));
    }
  }
  for (auto it = device_state.pipeline_cache.begin(); it != device_state.pipeline_cache.end();) {
    if (it->second.expired())
      it = device_state.pipeline_cache.erase(it);
    else
      ++it;
  }
  auto result = std::make_shared<Impl>();
  result->device = device.impl_->state;
  result->options = options;
  result->info = device.info();
  const auto& state = *result->device;
  const auto create_shader = detail::load_device<PFN_vkCreateShaderModule>(
      *state.instance, state.device, "vkCreateShaderModule");
  const auto destroy_shader = detail::load_device<PFN_vkDestroyShaderModule>(
      *state.instance, state.device, "vkDestroyShaderModule");
  const auto create_descriptor_layout = detail::load_device<PFN_vkCreateDescriptorSetLayout>(
      *state.instance, state.device, "vkCreateDescriptorSetLayout");
  result->destroy_descriptor_layout = detail::load_device<PFN_vkDestroyDescriptorSetLayout>(
      *state.instance, state.device, "vkDestroyDescriptorSetLayout");
  const auto create_pipeline_layout = detail::load_device<PFN_vkCreatePipelineLayout>(
      *state.instance, state.device, "vkCreatePipelineLayout");
  result->destroy_pipeline_layout = detail::load_device<PFN_vkDestroyPipelineLayout>(
      *state.instance, state.device, "vkDestroyPipelineLayout");
  const auto create_pipelines = detail::load_device<PFN_vkCreateComputePipelines>(
      *state.instance, state.device, "vkCreateComputePipelines");
  result->destroy_pipeline = detail::load_device<PFN_vkDestroyPipeline>(
      *state.instance, state.device, "vkDestroyPipeline");

  VkShaderModule shader = VK_NULL_HANDLE;
  VkShaderModuleCreateInfo shader_create{};
  shader_create.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  shader_create.codeSize = spirv.size() * sizeof(uint32_t);
  shader_create.pCode = spirv.data();
  detail::check(create_shader(state.device, &shader_create, nullptr, &shader),
                "vkCreateShaderModule");
  try {
    std::vector<VkDescriptorSetLayoutBinding> bindings(options.storage_binding_count);
    for (uint32_t i = 0; i < options.storage_binding_count; ++i) {
      bindings[i].binding = i;
      bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[i].descriptorCount = 1;
      bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo descriptor_create{};
    descriptor_create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_create.bindingCount = static_cast<uint32_t>(bindings.size());
    descriptor_create.pBindings = bindings.data();
    detail::check(create_descriptor_layout(state.device, &descriptor_create, nullptr,
                                           &result->descriptor_layout),
                  "vkCreateDescriptorSetLayout");

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.size = options.push_constant_bytes;
    VkPipelineLayoutCreateInfo layout_create{};
    layout_create.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_create.setLayoutCount = 1;
    layout_create.pSetLayouts = &result->descriptor_layout;
    layout_create.pushConstantRangeCount = options.push_constant_bytes == 0 ? 0 : 1;
    layout_create.pPushConstantRanges = options.push_constant_bytes == 0 ? nullptr : &push;
    detail::check(
        create_pipeline_layout(state.device, &layout_create, nullptr, &result->pipeline_layout),
        "vkCreatePipelineLayout");

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = options.entry_point.c_str();
    stage.pSpecializationInfo = specialization_entries.empty() ? nullptr : &specialization;
    VkComputePipelineCreateInfo pipeline_create{};
    pipeline_create.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_create.stage = stage;
    pipeline_create.layout = result->pipeline_layout;
    detail::check(create_pipelines(state.device, VK_NULL_HANDLE, 1, &pipeline_create, nullptr,
                                   &result->pipeline),
                  "vkCreateComputePipelines");
  } catch (...) {
    destroy_shader(state.device, shader, nullptr);
    throw;
  }
  destroy_shader(state.device, shader, nullptr);
  device_state.pipeline_cache[std::move(cache_key)] = result;
  return ComputePipeline(std::move(result));
}

} // namespace slopfab::vulkan
