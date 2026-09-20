#include "runtime_state.h"

namespace slopfab::vulkan {
namespace detail {
bool has_extension(const std::vector<std::string>& extensions, const char* name) {
  return std::find(extensions.begin(), extensions.end(), name) != extensions.end();
}

std::vector<std::string> enumerate_extensions(const InstanceState& state,
                                              VkPhysicalDevice physical) {
  std::vector<VkExtensionProperties> props;
  for (;;) {
    uint32_t count = 0;
    check(state.enumerate_device_extensions(physical, nullptr, &count, nullptr),
          "vkEnumerateDeviceExtensionProperties(count)");
    props.resize(count);
    const VkResult result = state.enumerate_device_extensions(
        physical, nullptr, &count, props.empty() ? nullptr : props.data());
    props.resize(count);
    if (result == VK_SUCCESS) break;
    if (result != VK_INCOMPLETE) fail("vkEnumerateDeviceExtensionProperties", result);
  }
  std::vector<std::string> result;
  result.reserve(props.size());
  for (const auto& prop : props) result.emplace_back(prop.extensionName);
  std::sort(result.begin(), result.end());
  return result;
}

uint32_t choose_compute_queue(const InstanceState& state, VkPhysicalDevice physical,
                              uint32_t* queue_count, uint32_t* timestamp_bits) {
  uint32_t count = 0;
  state.get_queue_family_properties(physical, &count, nullptr);
  std::vector<VkQueueFamilyProperties> families(count);
  state.get_queue_family_properties(physical, &count, families.data());
  families.resize(count);

  uint32_t fallback = std::numeric_limits<uint32_t>::max();
  for (uint32_t i = 0; i < count; ++i) {
    if (families[i].queueCount == 0 || !(families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
    if (fallback == std::numeric_limits<uint32_t>::max()) fallback = i;
    if (!(families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
      *queue_count = families[i].queueCount;
      *timestamp_bits = families[i].timestampValidBits;
      return i;
    }
  }
  if (fallback == std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("vulkan: physical device has no compute queue");
  }
  *queue_count = families[fallback].queueCount;
  *timestamp_bits = families[fallback].timestampValidBits;
  return fallback;
}

DeviceInfo inspect_device(const std::shared_ptr<InstanceState>& state, VkPhysicalDevice physical) {
  VkPhysicalDeviceProperties properties{};
  state->get_physical_device_properties(physical, &properties);
  if (properties.apiVersion < VK_API_VERSION_1_2) {
    throw std::runtime_error("vulkan: compute device requires Vulkan 1.2");
  }
  VkPhysicalDeviceMemoryProperties memory{};
  state->get_physical_device_memory_properties(physical, &memory);
  VkPhysicalDeviceFeatures base_features{};
  state->get_physical_device_features(physical, &base_features);

  DeviceInfo info;
  info.name = properties.deviceName;
  info.vendor_id = properties.vendorID;
  info.device_id = properties.deviceID;
  info.driver_version = properties.driverVersion;
  info.api_version = unpack_version(properties.apiVersion);
  info.discrete = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
  info.compute_queue_family = choose_compute_queue(*state, physical, &info.compute_queue_count,
                                                   &info.timestamp_valid_bits);
  info.timestamp_period_ns = properties.limits.timestampPeriod;
  info.max_compute_workgroup_invocations = properties.limits.maxComputeWorkGroupInvocations;
  for (int i = 0; i < 3; ++i) {
    info.max_compute_workgroup_count[i] = properties.limits.maxComputeWorkGroupCount[i];
    info.max_compute_workgroup_size[i] = properties.limits.maxComputeWorkGroupSize[i];
  }
  info.max_compute_shared_memory_bytes = properties.limits.maxComputeSharedMemorySize;
  info.max_push_constant_bytes = properties.limits.maxPushConstantsSize;
  info.max_storage_buffer_bytes = properties.limits.maxStorageBufferRange;
  info.min_storage_buffer_offset_alignment =
      std::max<VkDeviceSize>(1, properties.limits.minStorageBufferOffsetAlignment);
  info.max_storage_buffer_bindings = std::min(
      properties.limits.maxPerStageDescriptorStorageBuffers,
      properties.limits.maxDescriptorSetStorageBuffers);
  info.non_coherent_atom_bytes = std::max<VkDeviceSize>(1, properties.limits.nonCoherentAtomSize);
  info.extensions = enumerate_extensions(*state, physical);

  for (uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
    const bool local = (memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
    info.memory_heaps.push_back({memory.memoryHeaps[i].size, local});
  }

  if (state->get_physical_device_features2 != nullptr) {
    VkPhysicalDeviceVulkan11Features features11{};
    features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features11.pNext = &features12;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative{};
    cooperative.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16{};
    bfloat16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
    if (info.supports_extension(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)) {
      cooperative.pNext = info.supports_extension(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME)
          ? static_cast<void*>(&bfloat16) : static_cast<void*>(&features11);
      bfloat16.pNext = &features11;
    }
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = info.supports_extension(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)
        ? static_cast<void*>(&cooperative) : static_cast<void*>(&features11);
    state->get_physical_device_features2(physical, &features2);
    info.shader_int64 = features2.features.shaderInt64 == VK_TRUE;
    info.storage_buffer_16bit = features11.storageBuffer16BitAccess == VK_TRUE;
    info.storage_buffer_8bit = features12.storageBuffer8BitAccess == VK_TRUE;
    info.shader_float16 = features12.shaderFloat16 == VK_TRUE;
    info.shader_int8 = features12.shaderInt8 == VK_TRUE;
    info.timeline_semaphore = features12.timelineSemaphore == VK_TRUE;
    info.buffer_device_address = features12.bufferDeviceAddress == VK_TRUE;
    info.runtime_descriptor_array = features12.runtimeDescriptorArray == VK_TRUE;
    info.descriptor_binding_partially_bound =
        features12.descriptorBindingPartiallyBound == VK_TRUE;
    info.descriptor_binding_variable_count =
        features12.descriptorBindingVariableDescriptorCount == VK_TRUE;
    info.storage_buffer_non_uniform_indexing =
        features12.shaderStorageBufferArrayNonUniformIndexing == VK_TRUE;
    info.descriptor_indexing = features12.descriptorIndexing == VK_TRUE &&
                               info.runtime_descriptor_array &&
                               info.descriptor_binding_partially_bound &&
                               info.descriptor_binding_variable_count &&
                               info.storage_buffer_non_uniform_indexing;
    info.cooperative_matrix = cooperative.cooperativeMatrix == VK_TRUE;
    info.shader_bfloat16_type = bfloat16.shaderBFloat16Type == VK_TRUE;
    info.shader_bfloat16_cooperative_matrix =
        bfloat16.shaderBFloat16CooperativeMatrix == VK_TRUE;
  }
  if (state->get_physical_device_properties2 != nullptr) {
    VkPhysicalDeviceVulkan11Properties properties11{};
    properties11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;
    VkPhysicalDeviceVulkan12Properties properties12{};
    properties12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
    properties11.pNext = &properties12;
    VkPhysicalDeviceProperties2 properties2{};
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties2.pNext = &properties11;
    state->get_physical_device_properties2(physical, &properties2);
    info.max_allocation_bytes = properties11.maxMemoryAllocationSize;
    info.subgroup_size = properties11.subgroupSize;
    info.compute_subgroup_shuffle =
        (properties11.subgroupSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
        (properties11.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT);
    info.compute_subgroup_arithmetic =
        (properties11.subgroupSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) &&
        (properties11.subgroupSupportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);
    std::memcpy(info.driver_uuid, properties11.driverUUID, VK_UUID_SIZE);
    info.fp32_denorm_preserve = properties12.shaderDenormPreserveFloat32 == VK_TRUE;
    info.fp32_signed_zero_inf_nan_preserve =
        properties12.shaderSignedZeroInfNanPreserveFloat32 == VK_TRUE;
    info.fp32_rounding_rte = properties12.shaderRoundingModeRTEFloat32 == VK_TRUE;
  }
  if (info.cooperative_matrix && state->get_cooperative_matrix_properties) {
    uint32_t count = 0;
    if (state->get_cooperative_matrix_properties(physical, &count, nullptr) == VK_SUCCESS) {
      std::vector<VkCooperativeMatrixPropertiesKHR> tuples(count);
      for (auto& tuple : tuples) {
        tuple.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
      }
      if (state->get_cooperative_matrix_properties(physical, &count, tuples.data()) == VK_SUCCESS) {
        for (const auto& tuple : tuples) {
          info.cooperative_matrix_i8_i32_16x16x32 |=
              tuple.MSize == 16 && tuple.NSize == 16 && tuple.KSize == 32 &&
              tuple.scope == VK_SCOPE_SUBGROUP_KHR &&
              tuple.AType == VK_COMPONENT_TYPE_SINT8_KHR &&
              tuple.BType == VK_COMPONENT_TYPE_SINT8_KHR &&
              tuple.CType == VK_COMPONENT_TYPE_SINT32_KHR &&
              tuple.ResultType == VK_COMPONENT_TYPE_SINT32_KHR &&
              tuple.saturatingAccumulation == VK_FALSE;
          const bool common = tuple.MSize == 16 && tuple.NSize == 16 &&
              tuple.KSize == 16 && tuple.scope == VK_SCOPE_SUBGROUP_KHR &&
              tuple.CType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
              tuple.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
              tuple.saturatingAccumulation == VK_FALSE;
          info.cooperative_matrix_bf16_f32_16x16x16 |= common &&
              tuple.AType == VK_COMPONENT_TYPE_BFLOAT16_KHR &&
              tuple.BType == VK_COMPONENT_TYPE_BFLOAT16_KHR;
          info.cooperative_matrix_f16_f32_16x16x16 |= common &&
              tuple.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
              tuple.BType == VK_COMPONENT_TYPE_FLOAT16_KHR;
        }
      }
    }
  }
  return info;
}

}  // namespace detail
bool DeviceInfo::supports_extension(const std::string& extension_name) const {
  return std::binary_search(extensions.begin(), extensions.end(), extension_name);
}

uint64_t DeviceInfo::device_local_bytes() const {
  uint64_t total = 0;
  for (const auto& heap : memory_heaps) {
    if (heap.device_local && heap.bytes <= std::numeric_limits<uint64_t>::max() - total) {
      total += heap.bytes;
    }
  }
  return total;
}

Instance::Instance() = default;
Instance::~Instance() = default;
Instance::Instance(Instance&&) noexcept = default;
Instance& Instance::operator=(Instance&&) noexcept = default;
Instance::Instance(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}

bool Instance::available(std::string* diagnostic) noexcept {
  try {
    auto loader = std::make_shared<detail::Loader>();
    loader->global<PFN_vkCreateInstance>("vkCreateInstance");
    if (diagnostic != nullptr) diagnostic->clear();
    return true;
  } catch (const std::exception& error) {
    if (diagnostic != nullptr) *diagnostic = error.what();
    return false;
  }
}

Instance Instance::create(const InstanceOptions& options) {
  auto loader = std::make_shared<detail::Loader>();
  const auto create_instance = loader->global<PFN_vkCreateInstance>("vkCreateInstance");
  const auto enumerate_version =
      loader->global<PFN_vkEnumerateInstanceVersion>("vkEnumerateInstanceVersion", false);
  uint32_t loader_version = VK_API_VERSION_1_0;
  if (enumerate_version != nullptr) detail::check(enumerate_version(&loader_version),
                                                  "vkEnumerateInstanceVersion");
  const uint32_t requested = options.api_version == 0
                                 ? std::min(loader_version, VK_API_VERSION_1_3)
                                 : options.api_version;
  if (requested < VK_API_VERSION_1_2) {
    throw std::runtime_error("vulkan: slopfab requires Vulkan instance API 1.2 or newer");
  }
  if (requested > loader_version) {
    throw std::runtime_error("vulkan: requested instance API exceeds loader API");
  }

  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = options.application_name.c_str();
  app.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
  app.pEngineName = "slopfab";
  app.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
  app.apiVersion = requested;
  VkInstanceCreateInfo create{};
  create.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create.pApplicationInfo = &app;

  VkInstance instance = VK_NULL_HANDLE;
  detail::check(create_instance(&create, nullptr, &instance), "vkCreateInstance");
  const auto get = loader->get_instance_proc_addr();
  detail::InstanceHandleGuard handle_guard;
  handle_guard.instance = instance;
  handle_guard.destroy =
      detail::load_instance<PFN_vkDestroyInstance>(get, instance, "vkDestroyInstance");
  auto state = std::make_shared<detail::InstanceState>();
  state->loader = loader;
  state->instance = instance;
  state->loader_api_version = loader_version;
  state->instance_api_version = requested;
  state->destroy_instance = handle_guard.destroy;
  handle_guard.instance = VK_NULL_HANDLE;
  try {
    state->enumerate_physical_devices = detail::load_instance<PFN_vkEnumeratePhysicalDevices>(get, instance, "vkEnumeratePhysicalDevices");
    state->get_physical_device_properties = detail::load_instance<PFN_vkGetPhysicalDeviceProperties>(get, instance, "vkGetPhysicalDeviceProperties");
    state->get_physical_device_properties2 = detail::load_instance<PFN_vkGetPhysicalDeviceProperties2>(get, instance, "vkGetPhysicalDeviceProperties2");
    state->get_physical_device_features = detail::load_instance<PFN_vkGetPhysicalDeviceFeatures>(get, instance, "vkGetPhysicalDeviceFeatures");
    state->get_physical_device_features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(get(instance, "vkGetPhysicalDeviceFeatures2"));
    state->get_cooperative_matrix_properties =
        reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
            get(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    state->get_physical_device_memory_properties = detail::load_instance<PFN_vkGetPhysicalDeviceMemoryProperties>(get, instance, "vkGetPhysicalDeviceMemoryProperties");
    state->get_queue_family_properties = detail::load_instance<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(get, instance, "vkGetPhysicalDeviceQueueFamilyProperties");
    state->enumerate_device_extensions = detail::load_instance<PFN_vkEnumerateDeviceExtensionProperties>(get, instance, "vkEnumerateDeviceExtensionProperties");
    state->create_device = detail::load_instance<PFN_vkCreateDevice>(get, instance, "vkCreateDevice");
    state->get_device_proc_addr = detail::load_instance<PFN_vkGetDeviceProcAddr>(get, instance, "vkGetDeviceProcAddr");
    auto result = std::make_shared<Impl>();
    result->state = std::move(state);
    return Instance(std::move(result));
  } catch (...) {
    state.reset();
    throw;
  }
}

Version Instance::loader_version() const {
  if (!impl_) throw std::logic_error("vulkan: empty Instance");
  return detail::unpack_version(impl_->state->loader_api_version);
}
Version Instance::api_version() const {
  if (!impl_) throw std::logic_error("vulkan: empty Instance");
  return detail::unpack_version(impl_->state->instance_api_version);
}

std::vector<PhysicalDevice> Instance::enumerate_devices() const {
  if (!impl_) throw std::logic_error("vulkan: empty Instance");
  std::vector<VkPhysicalDevice> handles;
  for (;;) {
    uint32_t count = 0;
    detail::check(impl_->state->enumerate_physical_devices(impl_->state->instance, &count, nullptr),
                  "vkEnumeratePhysicalDevices(count)");
    handles.resize(count);
    const VkResult result = impl_->state->enumerate_physical_devices(
        impl_->state->instance, &count, handles.empty() ? nullptr : handles.data());
    handles.resize(count);
    if (result == VK_SUCCESS) break;
    if (result != VK_INCOMPLETE) detail::fail("vkEnumeratePhysicalDevices", result);
  }
  std::vector<PhysicalDevice> devices;
  devices.reserve(handles.size());
  for (VkPhysicalDevice handle : handles) {
    try {
      auto device = std::make_shared<PhysicalDevice::Impl>();
      device->state = impl_->state;
      device->physical = handle;
      device->info = detail::inspect_device(impl_->state, handle);
      devices.emplace_back(PhysicalDevice(std::move(device)));
    } catch (const std::runtime_error&) {
      // A graphics-only or otherwise unusable adapter is not a compute device.
    }
  }
  return devices;
}

Instance::operator bool() const noexcept { return impl_ != nullptr; }

PhysicalDevice::PhysicalDevice() = default;
PhysicalDevice::PhysicalDevice(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
PhysicalDevice::operator bool() const noexcept { return impl_ != nullptr; }
const DeviceInfo& PhysicalDevice::info() const {
  if (!impl_) throw std::logic_error("vulkan: empty PhysicalDevice");
  return impl_->info;
}

Device PhysicalDevice::create_device(const DeviceOptions& options) const {
  if (!impl_) throw std::logic_error("vulkan: empty PhysicalDevice");
  auto require = [&](bool requested, bool supported, const char* feature) {
    if (requested && !supported) {
      throw std::runtime_error(std::string("vulkan: device does not support requested ") + feature);
    }
  };
  require(options.enable_shader_float16, impl_->info.shader_float16, "shaderFloat16");
  require(options.enable_shader_int8, impl_->info.shader_int8, "shaderInt8");
  require(options.enable_shader_int64, impl_->info.shader_int64, "shaderInt64");
  require(options.enable_storage_buffer_16bit, impl_->info.storage_buffer_16bit,
          "storageBuffer16BitAccess");
  require(options.enable_storage_buffer_8bit, impl_->info.storage_buffer_8bit,
          "storageBuffer8BitAccess");
  require(options.enable_timeline_semaphore, impl_->info.timeline_semaphore, "timelineSemaphore");
  require(options.enable_buffer_device_address, impl_->info.buffer_device_address, "bufferDeviceAddress");
  require(options.enable_descriptor_indexing, impl_->info.descriptor_indexing, "descriptorIndexing");
  require(options.enable_cooperative_matrix, impl_->info.cooperative_matrix,
          "cooperativeMatrix");
  const bool enable_bfloat16 = options.enable_cooperative_matrix &&
      impl_->info.shader_bfloat16_type && impl_->info.shader_bfloat16_cooperative_matrix;

  std::vector<std::string> names = options.extensions;
  if (options.enable_cooperative_matrix) {
    names.emplace_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    if (enable_bfloat16) names.emplace_back(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
  }
  std::sort(names.begin(), names.end());
  names.erase(std::unique(names.begin(), names.end()), names.end());
  for (const auto& name : names) {
    if (!impl_->info.supports_extension(name)) {
      throw std::runtime_error("vulkan: device extension is unavailable: " + name);
    }
  }
  std::vector<const char*> extension_names;
  extension_names.reserve(names.size());
  for (const auto& name : names) extension_names.push_back(name.c_str());

  const float priority = 1.0f;
  VkDeviceQueueCreateInfo queue{};
  queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue.queueFamilyIndex = impl_->info.compute_queue_family;
  queue.queueCount = 1;
  queue.pQueuePriorities = &priority;

  VkPhysicalDeviceVulkan11Features features11{};
  features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
  features11.storageBuffer16BitAccess = options.enable_storage_buffer_16bit;
  VkPhysicalDeviceVulkan12Features features12{};
  features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features11.pNext = &features12;
  features12.storageBuffer8BitAccess = options.enable_storage_buffer_8bit;
  features12.shaderFloat16 = options.enable_shader_float16;
  features12.shaderInt8 = options.enable_shader_int8;
  features12.timelineSemaphore = options.enable_timeline_semaphore;
  features12.bufferDeviceAddress = options.enable_buffer_device_address;
  features12.descriptorIndexing = options.enable_descriptor_indexing;
  features12.runtimeDescriptorArray = options.enable_descriptor_indexing;
  features12.descriptorBindingPartiallyBound = options.enable_descriptor_indexing;
  features12.descriptorBindingVariableDescriptorCount = options.enable_descriptor_indexing;
  features12.shaderStorageBufferArrayNonUniformIndexing = options.enable_descriptor_indexing;
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative{};
  cooperative.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
  cooperative.cooperativeMatrix = options.enable_cooperative_matrix;
  VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloat16{};
  bfloat16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
  bfloat16.shaderBFloat16Type = enable_bfloat16;
  bfloat16.shaderBFloat16CooperativeMatrix = enable_bfloat16;
  bfloat16.pNext = &features11;
  cooperative.pNext = enable_bfloat16 ? static_cast<void*>(&bfloat16) : static_cast<void*>(&features11);
  VkPhysicalDeviceFeatures core_features{};
  core_features.shaderInt64 = options.enable_shader_int64;
  const bool any_features = options.enable_shader_float16 || options.enable_shader_int8 ||
                            options.enable_storage_buffer_16bit ||
                            options.enable_storage_buffer_8bit ||
                            options.enable_timeline_semaphore ||
                            options.enable_buffer_device_address ||
                            options.enable_descriptor_indexing ||
                            options.enable_cooperative_matrix;

  VkDeviceCreateInfo create{};
  create.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create.pNext = !any_features ? nullptr : options.enable_cooperative_matrix
      ? static_cast<void*>(&cooperative) : static_cast<void*>(&features11);
  create.pEnabledFeatures = options.enable_shader_int64 ? &core_features : nullptr;
  create.queueCreateInfoCount = 1;
  create.pQueueCreateInfos = &queue;
  create.enabledExtensionCount = static_cast<uint32_t>(extension_names.size());
  create.ppEnabledExtensionNames = extension_names.data();

  VkDevice handle = VK_NULL_HANDLE;
  detail::check(impl_->state->create_device(impl_->physical, &create, nullptr, &handle),
                "vkCreateDevice");
  detail::DeviceHandleGuard handle_guard;
  handle_guard.device = handle;
  handle_guard.destroy = reinterpret_cast<PFN_vkDestroyDevice>(
      impl_->state->get_device_proc_addr(handle, "vkDestroyDevice"));
  if (handle_guard.destroy == nullptr) {
    throw std::runtime_error("vulkan: device has no vkDestroyDevice");
  }
  auto state = std::make_shared<detail::DeviceState>();
  state->instance = impl_->state;
  state->physical = impl_->physical;
  state->device = handle;
  state->queue_family = impl_->info.compute_queue_family;
  state->buffer_device_address_enabled = options.enable_buffer_device_address;
  state->timeline_semaphore_enabled = options.enable_timeline_semaphore;
  state->destroy_device = handle_guard.destroy;
  handle_guard.device = VK_NULL_HANDLE;
  impl_->state->get_physical_device_memory_properties(impl_->physical, &state->memory);
  state->non_coherent_atom_size = impl_->info.non_coherent_atom_bytes;
  try {
    state->device_wait_idle = detail::load_device<PFN_vkDeviceWaitIdle>(*impl_->state, handle, "vkDeviceWaitIdle");
    state->queue_wait_idle = detail::load_device<PFN_vkQueueWaitIdle>(*impl_->state, handle, "vkQueueWaitIdle");
    state->create_buffer = detail::load_device<PFN_vkCreateBuffer>(*impl_->state, handle, "vkCreateBuffer");
    state->destroy_buffer = detail::load_device<PFN_vkDestroyBuffer>(*impl_->state, handle, "vkDestroyBuffer");
    state->get_buffer_requirements = detail::load_device<PFN_vkGetBufferMemoryRequirements>(*impl_->state, handle, "vkGetBufferMemoryRequirements");
    state->get_buffer_requirements2 = detail::load_device<PFN_vkGetBufferMemoryRequirements2>(*impl_->state, handle, "vkGetBufferMemoryRequirements2", false);
    state->allocate_memory = detail::load_device<PFN_vkAllocateMemory>(*impl_->state, handle, "vkAllocateMemory");
    state->free_memory = detail::load_device<PFN_vkFreeMemory>(*impl_->state, handle, "vkFreeMemory");
    state->bind_buffer_memory = detail::load_device<PFN_vkBindBufferMemory>(*impl_->state, handle, "vkBindBufferMemory");
    state->map_memory = detail::load_device<PFN_vkMapMemory>(*impl_->state, handle, "vkMapMemory");
    state->unmap_memory = detail::load_device<PFN_vkUnmapMemory>(*impl_->state, handle, "vkUnmapMemory");
    state->flush_mapped_ranges = detail::load_device<PFN_vkFlushMappedMemoryRanges>(*impl_->state, handle, "vkFlushMappedMemoryRanges");
    state->invalidate_mapped_ranges = detail::load_device<PFN_vkInvalidateMappedMemoryRanges>(*impl_->state, handle, "vkInvalidateMappedMemoryRanges");
    const auto get_queue = detail::load_device<PFN_vkGetDeviceQueue>(*impl_->state, handle, "vkGetDeviceQueue");
    get_queue(handle, state->queue_family, 0, &state->queue);
    auto result = std::make_shared<Device::Impl>();
    result->state = std::move(state);
    result->info = impl_->info;
    result->info.shader_float16_enabled = options.enable_shader_float16;
    result->info.shader_int8_enabled = options.enable_shader_int8;
    result->info.shader_int64_enabled = options.enable_shader_int64;
    result->info.storage_buffer_16bit_enabled = options.enable_storage_buffer_16bit;
    result->info.cooperative_matrix_enabled = options.enable_cooperative_matrix;
    return Device(std::move(result));
  } catch (...) {
    state.reset();
    throw;
  }
}

Device::Device() = default;
Device::~Device() = default;
Device::Device(Device&&) noexcept = default;
Device& Device::operator=(Device&&) noexcept = default;
Device::Device(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
Device::operator bool() const noexcept { return impl_ != nullptr; }
uint64_t Device::pipeline_cache_hits() const noexcept {
  return impl_ ? impl_->state->pipeline_cache_hits.load(std::memory_order_relaxed) : 0;
}
const DeviceInfo& Device::info() const {
  if (!impl_) throw std::logic_error("vulkan: empty Device");
  return impl_->info;
}
Queue Device::compute_queue() const {
  if (!impl_) throw std::logic_error("vulkan: empty Device");
  auto queue = std::make_shared<Queue::Impl>();
  queue->state = impl_->state;
  return Queue(std::move(queue));
}
void Device::wait_idle() const {
  if (!impl_) throw std::logic_error("vulkan: empty Device");
  std::lock_guard<std::mutex> lock(impl_->state->queue_mutex);
  detail::check(impl_->state->device_wait_idle(impl_->state->device), "vkDeviceWaitIdle");
}
void* Device::native_handle() const noexcept {
  return impl_ ? reinterpret_cast<void*>(impl_->state->device) : nullptr;
}

Queue::Queue() = default;
Queue::Queue(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
Queue::operator bool() const noexcept { return impl_ != nullptr; }
uint32_t Queue::family_index() const {
  if (!impl_) throw std::logic_error("vulkan: empty Queue");
  return impl_->state->queue_family;
}
void Queue::wait_idle() const {
  if (!impl_) throw std::logic_error("vulkan: empty Queue");
  std::lock_guard<std::mutex> lock(impl_->state->queue_mutex);
  detail::check(impl_->state->queue_wait_idle(impl_->state->queue), "vkQueueWaitIdle");
}
void* Queue::native_handle() const noexcept {
  return impl_ ? reinterpret_cast<void*>(impl_->state->queue) : nullptr;
}


}  // namespace slopfab::vulkan
