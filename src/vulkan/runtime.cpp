#include "vidfab/vulkan/runtime.h"
#include "vidfab/vulkan/compute.h"

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace vidfab::vulkan {
namespace detail {

[[noreturn]] void fail(const char* operation, VkResult result) {
  throw std::runtime_error(std::string("vulkan: ") + operation + " failed (VkResult " +
                           std::to_string(static_cast<int>(result)) + ")");
}

void check(VkResult result, const char* operation) {
  if (result != VK_SUCCESS) fail(operation, result);
}

Version unpack_version(uint32_t version) {
  return {VK_API_VERSION_MAJOR(version), VK_API_VERSION_MINOR(version),
          VK_API_VERSION_PATCH(version)};
}

class Loader {
 public:
  Loader() {
#ifdef _WIN32
    module_ = LoadLibraryW(L"vulkan-1.dll");
    if (module_ == nullptr) throw std::runtime_error("vulkan: vulkan-1.dll is not installed");
    get_instance_proc_addr_ = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        GetProcAddress(module_, "vkGetInstanceProcAddr"));
#elif defined(__APPLE__)
    module_ = dlopen("libvulkan.1.dylib", RTLD_NOW | RTLD_LOCAL);
    if (module_ == nullptr) module_ = dlopen("libvulkan.dylib", RTLD_NOW | RTLD_LOCAL);
    if (module_ == nullptr) throw std::runtime_error(std::string("vulkan: ") + dlerror());
    get_instance_proc_addr_ = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        dlsym(module_, "vkGetInstanceProcAddr"));
#else
    module_ = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (module_ == nullptr) throw std::runtime_error(std::string("vulkan: ") + dlerror());
    get_instance_proc_addr_ = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        dlsym(module_, "vkGetInstanceProcAddr"));
#endif
    if (get_instance_proc_addr_ == nullptr) {
      close();
      throw std::runtime_error("vulkan: loader has no vkGetInstanceProcAddr");
    }
  }

  ~Loader() { close(); }
  Loader(const Loader&) = delete;
  Loader& operator=(const Loader&) = delete;

  template <typename T>
  T global(const char* name, bool required = true) const {
    T fn = reinterpret_cast<T>(get_instance_proc_addr_(VK_NULL_HANDLE, name));
    if (required && fn == nullptr) {
      throw std::runtime_error(std::string("vulkan: loader has no ") + name);
    }
    return fn;
  }

  PFN_vkGetInstanceProcAddr get_instance_proc_addr() const { return get_instance_proc_addr_; }

 private:
  void close() noexcept {
    if (module_ == nullptr) return;
#ifdef _WIN32
    FreeLibrary(module_);
#else
    dlclose(module_);
#endif
    module_ = nullptr;
  }

#ifdef _WIN32
  HMODULE module_ = nullptr;
#else
  void* module_ = nullptr;
#endif
  PFN_vkGetInstanceProcAddr get_instance_proc_addr_ = nullptr;
};

template <typename T>
T load_instance(PFN_vkGetInstanceProcAddr get, VkInstance instance, const char* name) {
  T fn = reinterpret_cast<T>(get(instance, name));
  if (fn == nullptr) throw std::runtime_error(std::string("vulkan: loader has no ") + name);
  return fn;
}

// Stack-only guards close the small OOM window between a successful Vulkan
// create call and allocation of the shared state that will own its handle.
struct InstanceHandleGuard {
  VkInstance instance = VK_NULL_HANDLE;
  PFN_vkDestroyInstance destroy = nullptr;
  ~InstanceHandleGuard() {
    if (instance != VK_NULL_HANDLE && destroy != nullptr) destroy(instance, nullptr);
  }
};

struct DeviceHandleGuard {
  VkDevice device = VK_NULL_HANDLE;
  PFN_vkDestroyDevice destroy = nullptr;
  ~DeviceHandleGuard() {
    if (device != VK_NULL_HANDLE && destroy != nullptr) destroy(device, nullptr);
  }
};

struct InstanceState {
  std::shared_ptr<Loader> loader;
  VkInstance instance = VK_NULL_HANDLE;
  uint32_t loader_api_version = VK_API_VERSION_1_0;
  uint32_t instance_api_version = VK_API_VERSION_1_0;
  PFN_vkDestroyInstance destroy_instance = nullptr;
  PFN_vkEnumeratePhysicalDevices enumerate_physical_devices = nullptr;
  PFN_vkGetPhysicalDeviceProperties get_physical_device_properties = nullptr;
  PFN_vkGetPhysicalDeviceProperties2 get_physical_device_properties2 = nullptr;
  PFN_vkGetPhysicalDeviceFeatures get_physical_device_features = nullptr;
  PFN_vkGetPhysicalDeviceFeatures2 get_physical_device_features2 = nullptr;
  PFN_vkGetPhysicalDeviceMemoryProperties get_physical_device_memory_properties = nullptr;
  PFN_vkGetPhysicalDeviceQueueFamilyProperties get_queue_family_properties = nullptr;
  PFN_vkEnumerateDeviceExtensionProperties enumerate_device_extensions = nullptr;
  PFN_vkCreateDevice create_device = nullptr;
  PFN_vkGetDeviceProcAddr get_device_proc_addr = nullptr;

  ~InstanceState() {
    if (instance != VK_NULL_HANDLE && destroy_instance != nullptr) {
      destroy_instance(instance, nullptr);
    }
  }
};

template <typename T>
T load_device(const InstanceState& instance, VkDevice device, const char* name,
              bool required = true) {
  T fn = reinterpret_cast<T>(instance.get_device_proc_addr(device, name));
  if (required && fn == nullptr) {
    throw std::runtime_error(std::string("vulkan: device has no ") + name);
  }
  return fn;
}

struct DeviceState {
  std::shared_ptr<InstanceState> instance;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queue_family = 0;
  bool buffer_device_address_enabled = false;
  bool timeline_semaphore_enabled = false;
  VkPhysicalDeviceMemoryProperties memory{};
  uint64_t non_coherent_atom_size = 1;
  PFN_vkDestroyDevice destroy_device = nullptr;
  PFN_vkDeviceWaitIdle device_wait_idle = nullptr;
  PFN_vkQueueWaitIdle queue_wait_idle = nullptr;
  PFN_vkCreateBuffer create_buffer = nullptr;
  PFN_vkDestroyBuffer destroy_buffer = nullptr;
  PFN_vkGetBufferMemoryRequirements get_buffer_requirements = nullptr;
  PFN_vkGetBufferMemoryRequirements2 get_buffer_requirements2 = nullptr;
  PFN_vkAllocateMemory allocate_memory = nullptr;
  PFN_vkFreeMemory free_memory = nullptr;
  PFN_vkBindBufferMemory bind_buffer_memory = nullptr;
  PFN_vkMapMemory map_memory = nullptr;
  PFN_vkUnmapMemory unmap_memory = nullptr;
  PFN_vkFlushMappedMemoryRanges flush_mapped_ranges = nullptr;
  PFN_vkInvalidateMappedMemoryRanges invalidate_mapped_ranges = nullptr;
  std::mutex queue_mutex;

  ~DeviceState() {
    if (device != VK_NULL_HANDLE && destroy_device != nullptr) {
      if (device_wait_idle != nullptr) device_wait_idle(device);
      destroy_device(device, nullptr);
    }
  }
};

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
                              uint32_t* queue_count) {
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
      return i;
    }
  }
  if (fallback == std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("vulkan: physical device has no compute queue");
  }
  *queue_count = families[fallback].queueCount;
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
  info.compute_queue_family = choose_compute_queue(*state, physical, &info.compute_queue_count);
  info.max_compute_workgroup_invocations = properties.limits.maxComputeWorkGroupInvocations;
  for (int i = 0; i < 3; ++i) {
    info.max_compute_workgroup_count[i] = properties.limits.maxComputeWorkGroupCount[i];
    info.max_compute_workgroup_size[i] = properties.limits.maxComputeWorkGroupSize[i];
  }
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
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &features11;
    state->get_physical_device_features2(physical, &features2);
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
    info.fp32_denorm_preserve = properties12.shaderDenormPreserveFloat32 == VK_TRUE;
    info.fp32_signed_zero_inf_nan_preserve =
        properties12.shaderSignedZeroInfNanPreserveFloat32 == VK_TRUE;
    info.fp32_rounding_rte = properties12.shaderRoundingModeRTEFloat32 == VK_TRUE;
  }
  return info;
}

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

uint64_t align_up(uint64_t value, uint64_t alignment) {
  if (alignment <= 1) return value;
  const uint64_t remainder = value % alignment;
  if (remainder == 0) return value;
  if (value > std::numeric_limits<uint64_t>::max() - (alignment - remainder)) {
    throw std::overflow_error("vulkan: allocation size overflow");
  }
  return value + alignment - remainder;
}

}  // namespace detail

struct Instance::Impl {
  std::shared_ptr<detail::InstanceState> state;
};

struct PhysicalDevice::Impl {
  std::shared_ptr<detail::InstanceState> state;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  DeviceInfo info;
};

struct Device::Impl {
  std::shared_ptr<detail::DeviceState> state;
  DeviceInfo info;
};

struct Queue::Impl {
  std::shared_ptr<detail::DeviceState> state;
};

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
    throw std::runtime_error("vulkan: vidfab requires Vulkan instance API 1.2 or newer");
  }
  if (requested > loader_version) {
    throw std::runtime_error("vulkan: requested instance API exceeds loader API");
  }

  VkApplicationInfo app{};
  app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app.pApplicationName = options.application_name.c_str();
  app.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
  app.pEngineName = "vidfab";
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
  require(options.enable_storage_buffer_16bit, impl_->info.storage_buffer_16bit,
          "storageBuffer16BitAccess");
  require(options.enable_storage_buffer_8bit, impl_->info.storage_buffer_8bit,
          "storageBuffer8BitAccess");
  require(options.enable_timeline_semaphore, impl_->info.timeline_semaphore, "timelineSemaphore");
  require(options.enable_buffer_device_address, impl_->info.buffer_device_address, "bufferDeviceAddress");
  require(options.enable_descriptor_indexing, impl_->info.descriptor_indexing, "descriptorIndexing");

  std::vector<std::string> names = options.extensions;
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
  const bool any_features = options.enable_shader_float16 || options.enable_shader_int8 ||
                            options.enable_storage_buffer_16bit ||
                            options.enable_storage_buffer_8bit ||
                            options.enable_timeline_semaphore ||
                            options.enable_buffer_device_address ||
                            options.enable_descriptor_indexing;

  VkDeviceCreateInfo create{};
  create.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  create.pNext = any_features ? &features11 : nullptr;
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
    detail::check(device->allocate_memory(device->device, &allocate, nullptr, &block->memory),
                  "vkAllocateMemory");
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

// --- reusable compute submission ------------------------------------------

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
    if (pipeline != VK_NULL_HANDLE) destroy_pipeline(device->device, pipeline, nullptr);
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

ComputeFns load_compute_fns(const std::shared_ptr<DeviceState>& device) {
  ComputeFns f;
#define VIDFAB_LOAD_DEVICE(member, type, name) \
  f.member = load_device<type>(*device->instance, device->device, name)
  VIDFAB_LOAD_DEVICE(create_command_pool, PFN_vkCreateCommandPool, "vkCreateCommandPool");
  VIDFAB_LOAD_DEVICE(destroy_command_pool, PFN_vkDestroyCommandPool, "vkDestroyCommandPool");
  VIDFAB_LOAD_DEVICE(allocate_command_buffers, PFN_vkAllocateCommandBuffers,
                     "vkAllocateCommandBuffers");
  VIDFAB_LOAD_DEVICE(reset_command_buffer, PFN_vkResetCommandBuffer, "vkResetCommandBuffer");
  VIDFAB_LOAD_DEVICE(begin_command_buffer, PFN_vkBeginCommandBuffer, "vkBeginCommandBuffer");
  VIDFAB_LOAD_DEVICE(end_command_buffer, PFN_vkEndCommandBuffer, "vkEndCommandBuffer");
  VIDFAB_LOAD_DEVICE(create_descriptor_pool, PFN_vkCreateDescriptorPool,
                     "vkCreateDescriptorPool");
  VIDFAB_LOAD_DEVICE(destroy_descriptor_pool, PFN_vkDestroyDescriptorPool,
                     "vkDestroyDescriptorPool");
  VIDFAB_LOAD_DEVICE(reset_descriptor_pool, PFN_vkResetDescriptorPool,
                     "vkResetDescriptorPool");
  VIDFAB_LOAD_DEVICE(allocate_descriptor_sets, PFN_vkAllocateDescriptorSets,
                     "vkAllocateDescriptorSets");
  VIDFAB_LOAD_DEVICE(update_descriptor_sets, PFN_vkUpdateDescriptorSets,
                     "vkUpdateDescriptorSets");
  VIDFAB_LOAD_DEVICE(cmd_copy_buffer, PFN_vkCmdCopyBuffer, "vkCmdCopyBuffer");
  VIDFAB_LOAD_DEVICE(cmd_pipeline_barrier, PFN_vkCmdPipelineBarrier,
                     "vkCmdPipelineBarrier");
  VIDFAB_LOAD_DEVICE(cmd_bind_pipeline, PFN_vkCmdBindPipeline, "vkCmdBindPipeline");
  VIDFAB_LOAD_DEVICE(cmd_bind_descriptor_sets, PFN_vkCmdBindDescriptorSets,
                     "vkCmdBindDescriptorSets");
  VIDFAB_LOAD_DEVICE(cmd_push_constants, PFN_vkCmdPushConstants, "vkCmdPushConstants");
  VIDFAB_LOAD_DEVICE(cmd_dispatch, PFN_vkCmdDispatch, "vkCmdDispatch");
  VIDFAB_LOAD_DEVICE(create_semaphore, PFN_vkCreateSemaphore, "vkCreateSemaphore");
  VIDFAB_LOAD_DEVICE(destroy_semaphore, PFN_vkDestroySemaphore, "vkDestroySemaphore");
  VIDFAB_LOAD_DEVICE(get_semaphore_counter, PFN_vkGetSemaphoreCounterValue,
                     "vkGetSemaphoreCounterValue");
  VIDFAB_LOAD_DEVICE(wait_semaphores, PFN_vkWaitSemaphores, "vkWaitSemaphores");
  VIDFAB_LOAD_DEVICE(queue_submit, PFN_vkQueueSubmit, "vkQueueSubmit");
#undef VIDFAB_LOAD_DEVICE
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
    if (timeline != VK_NULL_HANDLE) f.destroy_semaphore(device->device, timeline, nullptr);
  }

  uint64_t completed() const {
    uint64_t value = 0;
    check(f.get_semaphore_counter(device->device, timeline, &value),
          "vkGetSemaphoreCounterValue");
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
      if (!slot.reserved || slot.value == 0 || slot.value > completed_value) continue;
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

AccessInfo access_info(BufferAccess access) {
  switch (access) {
    case BufferAccess::kHostWrite: return {VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_WRITE_BIT};
    case BufferAccess::kHostRead: return {VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT};
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

bool has_usage(BufferUsage value, BufferUsage flag) {
  return (static_cast<uint32_t>(value) & static_cast<uint32_t>(flag)) != 0;
}

struct SpirvShape {
  uint32_t local_size[3] = {};
  std::vector<uint32_t> bindings;
  bool has_local_size = false;
};

SpirvShape inspect_spirv(const std::vector<uint32_t>& words) {
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

}  // namespace detail

struct ComputeContext::Impl { std::shared_ptr<detail::ComputeState> state; };
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
    if (!recording || !state) return;
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

ComputePipeline::ComputePipeline() = default;
ComputePipeline::~ComputePipeline() = default;
ComputePipeline::ComputePipeline(ComputePipeline&&) noexcept = default;
ComputePipeline& ComputePipeline::operator=(ComputePipeline&&) noexcept = default;
ComputePipeline::ComputePipeline(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
ComputePipeline::operator bool() const noexcept { return impl_ != nullptr; }
uint32_t ComputePipeline::storage_binding_count() const noexcept {
  return impl_ ? impl_->options.storage_binding_count : 0;
}
uint32_t ComputePipeline::push_constant_bytes() const noexcept {
  return impl_ ? impl_->options.push_constant_bytes : 0;
}

ComputePipeline ComputePipeline::create(const Device& device,
                                        const std::vector<uint32_t>& spirv,
                                        const ComputePipelineOptions& options) {
  if (!device.impl_) throw std::invalid_argument("vulkan: ComputePipeline requires a device");
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
    detail::check(create_pipeline_layout(state.device, &layout_create, nullptr,
                                         &result->pipeline_layout),
                  "vkCreatePipelineLayout");

    VkPipelineShaderStageCreateInfo stage{};
    stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = shader;
    stage.pName = options.entry_point.c_str();
    VkComputePipelineCreateInfo pipeline_create{};
    pipeline_create.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_create.stage = stage;
    pipeline_create.layout = result->pipeline_layout;
    detail::check(create_pipelines(state.device, VK_NULL_HANDLE, 1, &pipeline_create,
                                   nullptr, &result->pipeline),
                  "vkCreateComputePipelines");
  } catch (...) {
    destroy_shader(state.device, shader, nullptr);
    throw;
  }
  destroy_shader(state.device, shader, nullptr);
  return ComputePipeline(std::move(result));
}

ComputeContext::ComputeContext(const Device& device, const ComputeContextOptions& options) {
  if (!device.impl_) throw std::invalid_argument("vulkan: ComputeContext requires a device");
  if (!device.impl_->state->timeline_semaphore_enabled) {
    throw std::invalid_argument("vulkan: ComputeContext requires timeline semaphores enabled");
  }
  if (options.max_in_flight == 0 || options.max_storage_bindings == 0 ||
      options.max_compute_binds_per_job == 0) {
    throw std::invalid_argument("vulkan: compute context limits must be nonzero");
  }
  const uint64_t descriptor_capacity =
      static_cast<uint64_t>(options.max_storage_bindings) *
      options.max_compute_binds_per_job;
  if (descriptor_capacity > std::numeric_limits<uint32_t>::max()) {
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
      slot.resources.reserve(static_cast<size_t>(descriptor_capacity) +
                             options.max_compute_binds_per_job + 8);
      slot.seen_bindings.reserve(options.max_storage_bindings);
      slot.descriptor_infos.reserve(options.max_storage_bindings);
      slot.descriptor_writes.reserve(options.max_storage_bindings);
      slot.descriptor_sets.resize(options.max_compute_binds_per_job, VK_NULL_HANDLE);
      slot.descriptor_pipelines.resize(options.max_compute_binds_per_job);
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
      size.descriptorCount = static_cast<uint32_t>(descriptor_capacity);
      VkDescriptorPoolCreateInfo descriptor_create{};
      descriptor_create.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      descriptor_create.maxSets = options.max_compute_binds_per_job;
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
  if (!impl_) throw std::logic_error("vulkan: empty ComputeContext");
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
        if (state->slots[i].value != 0) wait_for = std::min(wait_for, state->slots[i].value);
      }
      if (selected != state->slots.size()) {
        state->slots[selected].reserved = true;
        break;
      }
    }
    if (wait_for == std::numeric_limits<uint64_t>::max()) {
      throw std::logic_error("vulkan: no usable command slot is available");
    }
    state->wait_value(wait_for);  // bounded backpressure: oldest submitted slot
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
    if (commands) state->slots[selected].resources.swap(commands->resources);
    state->slots[selected].reserved = false;
    throw;
  }
}

Submission ComputeContext::submit(CommandList&& commands) {
  if (!impl_ || !commands.impl_) throw std::invalid_argument("vulkan: empty command submission");
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
  if (!impl_) return;
  std::lock_guard<std::mutex> lock(impl_->state->mutex);
  impl_->state->recycle_locked(impl_->state->completed());
}
uint32_t ComputeContext::in_flight() const {
  if (!impl_) return 0;
  std::lock_guard<std::mutex> lock(impl_->state->mutex);
  uint32_t count = 0;
  for (const auto& slot : impl_->state->slots) if (slot.reserved) ++count;
  return count;
}
uint64_t ComputeContext::descriptor_set_allocations() const noexcept {
  return impl_ ? impl_->state->descriptor_allocations.load(std::memory_order_relaxed) : 0;
}

Submission::Submission() = default;
Submission::Submission(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
Submission::operator bool() const noexcept { return impl_ != nullptr; }
uint64_t Submission::value() const noexcept { return impl_ ? impl_->value : 0; }
bool Submission::ready() const {
  if (!impl_) return false;
  return impl_->state->completed() >= impl_->value;
}
void Submission::wait() const {
  if (!impl_) throw std::logic_error("vulkan: empty Submission");
  impl_->state->wait_value(impl_->value);
}

CommandList::CommandList() = default;
CommandList::~CommandList() = default;
CommandList::CommandList(CommandList&&) noexcept = default;
CommandList& CommandList::operator=(CommandList&&) noexcept = default;
CommandList::CommandList(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CommandList::operator bool() const noexcept { return impl_ != nullptr; }

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
  if (bytes == 0) throw std::invalid_argument("vulkan: zero-sized copy");
  if (((source_offset | destination_offset | bytes) & 3u) != 0) {
    throw std::invalid_argument("vulkan: copy offsets and size must be four-byte aligned");
  }
  if (source.impl_->buffer == destination.impl_->buffer &&
      source_offset < destination_offset + bytes &&
      destination_offset < source_offset + bytes) {
    throw std::invalid_argument("vulkan: same-buffer copy ranges overlap");
  }
  impl_->resources.reserve(impl_->resources.size() + 2);
  impl_->retain(source.impl_);
  impl_->retain(destination.impl_);
  VkBufferCopy region{source_offset, destination_offset, bytes};
  impl_->state->f.cmd_copy_buffer(impl_->state->slots[impl_->slot].commands,
                                  source.impl_->buffer, destination.impl_->buffer, 1, &region);
}

void CommandList::barrier(Buffer& buffer, BufferAccess before, BufferAccess after,
                          uint64_t offset, uint64_t bytes) {
  if (!impl_ || !buffer.impl_) throw std::invalid_argument("vulkan: barrier requires buffer");
  if (buffer.impl_->pool->device != impl_->state->device) {
    throw std::invalid_argument("vulkan: barrier buffer belongs to another device");
  }
  if (bytes == ~uint64_t{0}) bytes = buffer.impl_->bytes - offset;
  buffer.impl_->check_range(offset, bytes);
  if (bytes == 0) throw std::invalid_argument("vulkan: zero-sized barrier");
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
  impl_->state->f.cmd_pipeline_barrier(impl_->state->slots[impl_->slot].commands,
                                       src.stage, dst.stage, 0, 0, nullptr, 1, &barrier,
                                       0, nullptr);
}

void CommandList::bind_compute(ComputePipeline& pipeline,
                               const std::vector<StorageBinding>& bindings) {
  if (!impl_ || !pipeline.impl_) throw std::invalid_argument("vulkan: bind requires pipeline");
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
        binding.buffer == nullptr ||
        !binding.buffer->impl_) {
      throw std::invalid_argument("vulkan: invalid or duplicate storage binding");
    }
    slot.seen_bindings[binding.binding] = 1;
    if (binding.buffer->impl_->pool->device != impl_->state->device ||
        !detail::has_usage(binding.buffer->impl_->buffer_usage, BufferUsage::kStorage)) {
      throw std::invalid_argument("vulkan: storage binding has wrong device or usage");
    }
    uint64_t range = binding.bytes;
    if (range == ~uint64_t{0}) range = binding.buffer->impl_->bytes - binding.offset;
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
  for (const auto& binding : bindings) impl_->retain(binding.buffer->impl_);
  impl_->retain(pipeline.impl_);
  const uint32_t set_index = impl_->compute_bind_count;
  if (slot.descriptor_pipelines[set_index].get() != pipeline.impl_.get()) {
    if (set_index == 0) {
      // Safe only before this command list refers to a descriptor set. Stable
      // operator sequences take the allocation-free path across submissions.
      detail::check(impl_->state->f.reset_descriptor_pool(impl_->state->device->device,
                                                          slot.descriptors, 0),
                    "vkResetDescriptorPool");
      std::fill(slot.descriptor_sets.begin(), slot.descriptor_sets.end(), VK_NULL_HANDLE);
      std::fill(slot.descriptor_pipelines.begin(), slot.descriptor_pipelines.end(), nullptr);
    } else if (slot.descriptor_sets[set_index] != VK_NULL_HANDLE) {
      throw std::logic_error(
          "vulkan: compute pipeline sequence changed within cached descriptor capacity");
    }
  }
  if (slot.descriptor_sets[set_index] == VK_NULL_HANDLE) {
    VkDescriptorSetAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate.descriptorPool = slot.descriptors;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &pipeline.impl_->descriptor_layout;
    detail::check(impl_->state->f.allocate_descriptor_sets(
                      impl_->state->device->device, &allocate,
                      &slot.descriptor_sets[set_index]),
                  "vkAllocateDescriptorSets");
    impl_->state->descriptor_allocations.fetch_add(1, std::memory_order_relaxed);
    slot.descriptor_pipelines[set_index] = pipeline.impl_;
  }
  for (auto& write : slot.descriptor_writes) {
    write.dstSet = slot.descriptor_sets[set_index];
  }
  impl_->state->f.update_descriptor_sets(impl_->state->device->device,
      static_cast<uint32_t>(slot.descriptor_writes.size()), slot.descriptor_writes.data(),
      0, nullptr);
  const VkCommandBuffer commands = slot.commands;
  impl_->state->f.cmd_bind_pipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                                    pipeline.impl_->pipeline);
  impl_->state->f.cmd_bind_descriptor_sets(commands, VK_PIPELINE_BIND_POINT_COMPUTE,
                                           pipeline.impl_->pipeline_layout, 0, 1,
                                           &slot.descriptor_sets[set_index],
                                           0, nullptr);
  impl_->pipeline = pipeline.impl_;
  impl_->push_constants_set = false;
  ++impl_->compute_bind_count;
}

void CommandList::push_constants(const void* data, uint32_t bytes) {
  if (!impl_ || !impl_->pipeline) throw std::logic_error("vulkan: bind pipeline before push constants");
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
  if (!impl_ || !impl_->pipeline) throw std::logic_error("vulkan: bind pipeline before dispatch");
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
  impl_->state->f.cmd_dispatch(impl_->state->slots[impl_->slot].commands,
                               groups_x, groups_y, groups_z);
}

}  // namespace vidfab::vulkan
