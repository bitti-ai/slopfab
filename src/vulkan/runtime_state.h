#pragma once
#include "slopfab/vulkan/runtime.h"
#include "slopfab/vulkan/compute.h"

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_set>
#include <unordered_map>
#include <utility>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace slopfab::vulkan {
namespace detail {

[[noreturn]] inline void fail(const char* operation, VkResult result) {
  throw std::runtime_error(std::string("vulkan: ") + operation + " failed (VkResult " +
                           std::to_string(static_cast<int>(result)) + ")" +
                           (result == VK_ERROR_OUT_OF_DEVICE_MEMORY ? ": out of memory (device)"
                            : result == VK_ERROR_OUT_OF_HOST_MEMORY ? ": out of memory (host)"
                                                                    : ""));
}

inline void check(VkResult result, const char* operation) {
  if (result != VK_SUCCESS)
    fail(operation, result);
}

inline Version unpack_version(uint32_t version) {
  return {VK_API_VERSION_MAJOR(version), VK_API_VERSION_MINOR(version),
          VK_API_VERSION_PATCH(version)};
}

class Loader {
public:
  Loader() {
#ifdef _WIN32
    module_ = LoadLibraryW(L"vulkan-1.dll");
    if (module_ == nullptr)
      throw std::runtime_error("vulkan: vulkan-1.dll is not installed");
    get_instance_proc_addr_ = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        GetProcAddress(module_, "vkGetInstanceProcAddr"));
#elif defined(__APPLE__)
    module_ = dlopen("libvulkan.1.dylib", RTLD_NOW | RTLD_LOCAL);
    if (module_ == nullptr)
      module_ = dlopen("libvulkan.dylib", RTLD_NOW | RTLD_LOCAL);
    if (module_ == nullptr)
      throw std::runtime_error(std::string("vulkan: ") + dlerror());
    get_instance_proc_addr_ =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(module_, "vkGetInstanceProcAddr"));
#else
    module_ = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (module_ == nullptr)
      throw std::runtime_error(std::string("vulkan: ") + dlerror());
    get_instance_proc_addr_ =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(module_, "vkGetInstanceProcAddr"));
#endif
    if (get_instance_proc_addr_ == nullptr) {
      close();
      throw std::runtime_error("vulkan: loader has no vkGetInstanceProcAddr");
    }
  }

  ~Loader() {
    close();
  }

  Loader(const Loader&) = delete;
  Loader& operator=(const Loader&) = delete;

  template <typename T> T global(const char* name, bool required = true) const {
    T fn = reinterpret_cast<T>(get_instance_proc_addr_(VK_NULL_HANDLE, name));
    if (required && fn == nullptr) {
      throw std::runtime_error(std::string("vulkan: loader has no ") + name);
    }
    return fn;
  }

  PFN_vkGetInstanceProcAddr get_instance_proc_addr() const {
    return get_instance_proc_addr_;
  }

private:
  void close() noexcept {
    if (module_ == nullptr)
      return;
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
  if (fn == nullptr)
    throw std::runtime_error(std::string("vulkan: loader has no ") + name);
  return fn;
}

// Stack-only guards close the small OOM window between a successful Vulkan
// create call and allocation of the shared state that will own its handle.
struct InstanceHandleGuard {
  VkInstance instance = VK_NULL_HANDLE;
  PFN_vkDestroyInstance destroy = nullptr;

  ~InstanceHandleGuard() {
    if (instance != VK_NULL_HANDLE && destroy != nullptr)
      destroy(instance, nullptr);
  }
};

struct DeviceHandleGuard {
  VkDevice device = VK_NULL_HANDLE;
  PFN_vkDestroyDevice destroy = nullptr;

  ~DeviceHandleGuard() {
    if (device != VK_NULL_HANDLE && destroy != nullptr)
      destroy(device, nullptr);
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
  PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR get_cooperative_matrix_properties = nullptr;
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
  // Weak entries avoid a device -> pipeline -> device ownership cycle.
  std::mutex pipeline_mutex;
  std::unordered_map<std::string, std::weak_ptr<void>> pipeline_cache;
  std::atomic<uint64_t> pipeline_cache_hits{0};

  ~DeviceState() {
    if (device != VK_NULL_HANDLE && destroy_device != nullptr) {
      if (device_wait_idle != nullptr)
        device_wait_idle(device);
      destroy_device(device, nullptr);
    }
  }
};

inline uint64_t align_up(uint64_t value, uint64_t alignment) {
  if (alignment <= 1)
    return value;
  const uint64_t remainder = value % alignment;
  if (remainder == 0)
    return value;
  if (value > std::numeric_limits<uint64_t>::max() - (alignment - remainder)) {
    throw std::overflow_error("vulkan: allocation size overflow");
  }
  return value + alignment - remainder;
}

} // namespace detail

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

} // namespace slopfab::vulkan
