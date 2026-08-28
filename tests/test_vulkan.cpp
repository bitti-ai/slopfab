#include "harness.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "vidfab/vulkan/runtime.h"
#include "vidfab/vulkan/compute.h"
#include "vidfab/vulkan/yuv_converter.h"
#include "vidfab/video/y4m.h"

namespace {

std::vector<uint32_t> load_spirv(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error(std::string("cannot open SPIR-V: ") + path);
  const std::streamoff length = file.tellg();
  if (length <= 0 || (length % 4) != 0) throw std::runtime_error("invalid SPIR-V byte size");
  file.seekg(0);
  std::vector<uint32_t> words(static_cast<size_t>(length) / 4);
  file.read(reinterpret_cast<char*>(words.data()), length);
  if (!file) throw std::runtime_error("cannot read SPIR-V");
  return words;
}

VIDFAB_TEST(vulkan_runtime_and_pool) {
  using namespace vidfab::vulkan;

  std::string diagnostic;
  if (!Instance::available(&diagnostic)) {
    // Absence is a supported runtime state. The important contract is a useful
    // diagnostic rather than a load-time process failure.
    CHECK(!diagnostic.empty());
    return;
  }

  bool old_api_rejected = false;
  try {
    InstanceOptions old_api;
    old_api.api_version = (1u << 22) | (1u << 12);  // Vulkan 1.1.0
    (void)Instance::create(old_api);
  } catch (const std::runtime_error&) {
    old_api_rejected = true;
  }
  CHECK(old_api_rejected);

  Instance instance = Instance::create();
  CHECK(static_cast<bool>(instance));
  CHECK(instance.loader_version().major >= 1);
  CHECK(instance.api_version().major > 1 || instance.api_version().minor >= 2);
  const std::vector<PhysicalDevice> physical = instance.enumerate_devices();
  if (physical.empty()) {
    CHECK(physical.empty());
    return;
  }

  const DeviceInfo& info = physical.front().info();
  CHECK(!info.name.empty());
  CHECK(info.api_version.major >= 1);
  CHECK(info.compute_queue_count > 0);
  CHECK(info.max_compute_workgroup_invocations > 0);
  CHECK(!info.memory_heaps.empty());

  DeviceOptions feature_options;
  feature_options.enable_shader_float16 = info.shader_float16;
  feature_options.enable_shader_int8 = info.shader_int8;
  feature_options.enable_storage_buffer_16bit = info.storage_buffer_16bit;
  feature_options.enable_storage_buffer_8bit = info.storage_buffer_8bit;
  feature_options.enable_timeline_semaphore = info.timeline_semaphore;
  feature_options.enable_buffer_device_address = info.buffer_device_address;
  feature_options.enable_descriptor_indexing = info.descriptor_indexing;
  Device feature_device = physical.front().create_device(feature_options);
  CHECK(static_cast<bool>(feature_device));
  if (info.buffer_device_address) {
    BufferPool address_pool(feature_device, 64 * 1024);
    {
      Buffer addressable = address_pool.allocate(
          257, BufferUsage::kStorage | BufferUsage::kDeviceAddress,
          MemoryUsage::kDevice);
      CHECK(static_cast<bool>(addressable));
      CHECK(address_pool.used_bytes() >= addressable.size());
    }
    CHECK(address_pool.used_bytes() == 0);
    address_pool.trim();
    CHECK(address_pool.reserved_bytes() == 0);
  }
  feature_device.wait_idle();
  feature_device = Device();

  Device device = physical.front().create_device();
  CHECK(static_cast<bool>(device));
  CHECK(device.native_handle() != nullptr);
  Queue queue = device.compute_queue();
  CHECK(static_cast<bool>(queue));
  CHECK(queue.native_handle() != nullptr);
  CHECK(queue.family_index() == info.compute_queue_family);

  // A deliberately small block makes reuse and trimming observable without
  // reserving meaningful VRAM in the unit suite.
  BufferPool pool(device, 64 * 1024);
  bool disabled_address_rejected = false;
  try {
    (void)pool.allocate(256, BufferUsage::kStorage | BufferUsage::kDeviceAddress,
                        MemoryUsage::kDevice);
  } catch (const std::logic_error&) {
    disabled_address_rejected = true;
  }
  CHECK(disabled_address_rejected);
  const BufferUsage transfer = BufferUsage::kTransferSource |
                               BufferUsage::kTransferDestination;
  uintptr_t tiny_native = 0;
  uint64_t tiny_offset = 0;
  {
    Buffer tiny = pool.allocate(1, transfer, MemoryUsage::kUpload);
    Buffer unaligned = pool.allocate(3, transfer, MemoryUsage::kUpload);
    tiny_native = tiny.native_handle();
    tiny_offset = tiny.memory_offset();
    CHECK(unaligned.memory_offset() >= tiny.memory_offset() +
          info.non_coherent_atom_bytes);
    CHECK(pool.used_bytes() >= 4);
  }
  CHECK(pool.used_bytes() == 0);
  Buffer reused = pool.allocate(1, transfer, MemoryUsage::kUpload);
  CHECK(reused.memory_offset() == tiny_offset);
  // Vulkan buffer objects themselves need not be recycled; their pooled
  // memory spans are. Keep this check merely to ensure a real handle exists.
  CHECK(tiny_native != 0);
  reused.reset_after_idle(queue);
  CHECK(!static_cast<bool>(reused));
  CHECK(pool.used_bytes() == 0);

  Buffer first = pool.allocate(4096, transfer, MemoryUsage::kUpload);
  Buffer second = pool.allocate(4096, transfer, MemoryUsage::kUpload);
  CHECK(first.mapped_data() != nullptr);
  CHECK(second.mapped_data() != nullptr);
  CHECK(pool.used_bytes() >= 8192);
  CHECK(pool.reserved_bytes() >= pool.used_bytes());

  std::vector<uint32_t> source(1024);
  for (uint32_t i = 0; i < source.size(); ++i) source[i] = i * 2654435761u;
  first.write(0, source.data(), source.size() * sizeof(uint32_t));
  std::vector<uint32_t> copy(source.size());
  first.read(0, copy.data(), copy.size() * sizeof(uint32_t));
  CHECK(std::memcmp(source.data(), copy.data(), source.size() * sizeof(uint32_t)) == 0);

  first = Buffer();
  second = Buffer();
  CHECK(pool.used_bytes() == 0);
  pool.trim();
  CHECK(pool.reserved_bytes() == 0);
  queue.wait_idle();
}

VIDFAB_TEST(vulkan_compute_submission) {
  using namespace vidfab::vulkan;
  if (!Instance::available()) return;
  Instance instance = Instance::create();
  std::vector<PhysicalDevice> physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore) return;

  DeviceOptions device_options;
  device_options.enable_timeline_semaphore = true;
  Device device = physical.front().create_device(device_options);
  ComputeContextOptions context_options;
  context_options.max_in_flight = 2;
  context_options.max_storage_bindings = 2;
  ComputeContext context(device, context_options);
  ComputeContext other_context(device, context_options);

  const std::vector<uint32_t> spirv = load_spirv(VIDFAB_TEST_AFFINE_SPV_PATH);
  ComputePipelineOptions pipeline_options;
  pipeline_options.storage_binding_count = 2;
  pipeline_options.push_constant_bytes = 12;
  pipeline_options.local_size[0] = 64;
  ComputePipeline pipeline = ComputePipeline::create(device, spirv, pipeline_options);

  bool local_size_rejected = false;
  try {
    ComputePipelineOptions wrong = pipeline_options;
    wrong.local_size[0] = 32;
    (void)ComputePipeline::create(device, spirv, wrong);
  } catch (const std::invalid_argument&) {
    local_size_rejected = true;
  }
  CHECK(local_size_rejected);

  // A list is context-owned and cannot be submitted through a different
  // timeline. Its destructor safely returns the still-recording slot.
  bool foreign_rejected = false;
  {
    CommandList foreign = other_context.begin();
    try {
      (void)context.submit(std::move(foreign));
    } catch (const std::invalid_argument&) {
      foreign_rejected = true;
    }
  }
  CHECK(foreign_rejected);
  CHECK(other_context.in_flight() == 0);

  constexpr uint32_t count = 1003;  // deliberately not a multiple of local_size_x
  constexpr uint64_t bytes = static_cast<uint64_t>(count) * sizeof(float);
  BufferPool pool(device, 64 * 1024);
  const BufferUsage upload_usage = BufferUsage::kTransferSource;
  const BufferUsage input_usage = BufferUsage::kTransferDestination | BufferUsage::kStorage;
  const BufferUsage output_usage = BufferUsage::kStorage | BufferUsage::kTransferSource;
  const BufferUsage readback_usage = BufferUsage::kTransferDestination;

  // Range and usage failures happen before any Vulkan command is emitted and
  // the abandoned list remains recyclable.
  {
    Buffer tiny_src = pool.allocate(16, upload_usage, MemoryUsage::kUpload);
    Buffer tiny_dst = pool.allocate(16, input_usage, MemoryUsage::kDevice);
    Buffer tiny_out = pool.allocate(16, output_usage, MemoryUsage::kDevice);
    CommandList invalid = context.begin();
    bool range_rejected = false;
    try {
      invalid.copy_buffer(tiny_src, tiny_dst, 17);
    } catch (const std::out_of_range&) {
      range_rejected = true;
    }
    CHECK(range_rejected);
    bool descriptor_range_rejected = false;
    try {
      invalid.bind_compute(pipeline, {{0, &tiny_dst, 0, 17},
                                      {1, &tiny_out, 0, 16}});
    } catch (const std::out_of_range&) {
      descriptor_range_rejected = true;
    }
    CHECK(descriptor_range_rejected);
    if (physical.front().info().min_storage_buffer_offset_alignment > 1) {
      bool descriptor_alignment_rejected = false;
      try {
        invalid.bind_compute(pipeline, {{0, &tiny_dst, 1, 4},
                                        {1, &tiny_out, 0, 16}});
      } catch (const std::invalid_argument&) {
        descriptor_alignment_rejected = true;
      }
      CHECK(descriptor_alignment_rejected);
    }
    Buffer self_copy = pool.allocate(
        32, BufferUsage::kTransferSource | BufferUsage::kTransferDestination,
        MemoryUsage::kUpload);
    bool overlap_rejected = false;
    try {
      invalid.copy_buffer(self_copy, self_copy, 16, 0, 8);
    } catch (const std::invalid_argument&) {
      overlap_rejected = true;
    }
    CHECK(overlap_rejected);
    // Non-overlapping regions in one VkBuffer are legal.
    invalid.copy_buffer(self_copy, self_copy, 16, 0, 16);
  }
  CHECK(context.in_flight() == 0);
  CHECK(pool.used_bytes() == 0);

  struct Parameters { float scale; float bias; uint32_t count; };
  struct Job {
    Submission completion;
    Buffer readback;
    float scale = 0;
    float bias = 0;
  };
  std::vector<float> input(count);
  for (uint32_t i = 0; i < count; ++i) input[i] = static_cast<float>(i) * 0.125f - 7.0f;
  std::vector<Job> jobs;

  for (uint32_t iteration = 0; iteration < 5; ++iteration) {
    Buffer upload = pool.allocate(bytes, upload_usage, MemoryUsage::kUpload);
    Buffer device_input = pool.allocate(bytes, input_usage, MemoryUsage::kDevice);
    Buffer device_output = pool.allocate(bytes, output_usage, MemoryUsage::kDevice);
    Buffer readback = pool.allocate(bytes, readback_usage, MemoryUsage::kReadback);
    upload.write(0, input.data(), bytes);

    const Parameters parameters{1.25f + iteration * 0.5f,
                                -3.0f + static_cast<float>(iteration), count};
    CommandList commands = context.begin();
    commands.barrier(upload, BufferAccess::kHostWrite, BufferAccess::kTransferRead);
    commands.copy_buffer(upload, device_input, bytes);
    commands.barrier(device_input, BufferAccess::kTransferWrite, BufferAccess::kComputeRead);
    commands.bind_compute(pipeline, {{0, &device_input, 0, bytes},
                                     {1, &device_output, 0, bytes}});
    commands.push_constants(&parameters, sizeof(parameters));
    commands.dispatch((count + 63) / 64);
    commands.barrier(device_output, BufferAccess::kComputeWrite,
                     BufferAccess::kTransferRead);
    commands.copy_buffer(device_output, readback, bytes);
    commands.barrier(readback, BufferAccess::kTransferWrite, BufferAccess::kHostRead);
    Submission completion = context.submit(std::move(commands));
    CHECK(completion.value() != 0);
    CHECK(context.in_flight() <= 2);
    jobs.push_back({std::move(completion), std::move(readback),
                    parameters.scale, parameters.bias});
    // upload/device_input/device_output wrappers die here. The submitted slot
    // retains their allocations until its exact timeline value completes.
  }
  pipeline = ComputePipeline();  // in-flight jobs retain the pipeline too

  uint64_t previous_value = 0;
  for (Job& job : jobs) {
    CHECK(job.completion.value() > previous_value);
    previous_value = job.completion.value();
    job.completion.wait();
    context.collect();
    std::vector<float> output(count);
    job.readback.read(0, output.data(), bytes);
    for (uint32_t i = 0; i < count; ++i) {
      CHECK_NEAR(output[i], input[i] * job.scale + job.bias, 1e-5);
    }
  }
  CHECK(context.in_flight() == 0);
  jobs.clear();
  context.collect();
  CHECK(pool.used_bytes() == 0);
  pool.trim();
  CHECK(pool.reserved_bytes() == 0);
}

VIDFAB_TEST(vulkan_yuv420_output) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!Instance::available()) return;
  Instance probe = Instance::create();
  const auto devices = probe.enumerate_devices();
  if (devices.empty() || !devices.front().info().timeline_semaphore) return;

  Yuv420Converter converter;
  bool unavailable_device_rejected = false;
  try {
    Yuv420Converter unavailable(std::numeric_limits<uint32_t>::max());
  } catch (const std::runtime_error& error) {
    unavailable_device_rejected = std::string(error.what()).find("unavailable") != std::string::npos;
  }
  CHECK(unavailable_device_rejected);
  struct Extent { int width; int height; };
  const Extent extents[] = {{2, 2}, {10, 6}, {128, 66}};
  uint64_t previous_high_water = 0;
  for (const Extent extent : extents) {
    const size_t pixels = static_cast<size_t>(extent.width) * extent.height;
    std::vector<float> r(pixels), g(pixels), b(pixels);
    for (size_t i = 0; i < pixels; ++i) {
      r[i] = static_cast<float>((i * 17) % 113) / 97.0f - 0.08f;
      g[i] = static_cast<float>((i * 29 + 3) % 127) / 109.0f;
      b[i] = static_cast<float>((i * 43 + 11) % 139) / 101.0f - 0.12f;
    }
    const int ys = extent.width + 13;
    const int cs = extent.width / 2 + 7;
    std::vector<uint8_t> cpu_y(static_cast<size_t>(ys) * extent.height, 0xa5);
    std::vector<uint8_t> cpu_u(static_cast<size_t>(cs) * (extent.height / 2), 0xa5);
    std::vector<uint8_t> cpu_v(cpu_u.size(), 0xa5);
    std::vector<uint8_t> vk_y(cpu_y.size(), 0xa5), vk_u(cpu_u.size(), 0xa5), vk_v(cpu_v.size(), 0xa5);
    video::rgb_frame_to_yuv420(r.data(), g.data(), b.data(), extent.height, extent.width,
                               cpu_y.data(), ys, cpu_u.data(), cs, cpu_v.data(), cs);
    converter.convert(r.data(), g.data(), b.data(), extent.height, extent.width,
                      vk_y.data(), ys, vk_u.data(), cs, vk_v.data(), cs);

    int differences = 0;
    int max_difference = 0;
    auto compare_plane = [&](const std::vector<uint8_t>& expected,
                             const std::vector<uint8_t>& actual, int rows, int columns,
                             int stride) {
      for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < columns; ++x) {
          const int delta = std::abs(static_cast<int>(expected[static_cast<size_t>(y) * stride + x]) -
                                     static_cast<int>(actual[static_cast<size_t>(y) * stride + x]));
          if (delta != 0) ++differences;
          max_difference = std::max(max_difference, delta);
        }
        for (int x = columns; x < stride; ++x) {
          CHECK(actual[static_cast<size_t>(y) * stride + x] == 0xa5);
        }
      }
    };
    compare_plane(cpu_y, vk_y, extent.height, extent.width, ys);
    compare_plane(cpu_u, vk_u, extent.height / 2, extent.width / 2, cs);
    compare_plane(cpu_v, vk_v, extent.height / 2, extent.width / 2, cs);
    CHECK_MSG(max_difference <= 1,
              "Vulkan YUV differs by %d (changed samples %d) at %dx%d",
              max_difference, differences, extent.width, extent.height);
    CHECK(converter.capacity_pixels() >= pixels);
    CHECK(converter.high_water_bytes() >= previous_high_water);
    previous_high_water = converter.high_water_bytes();

    const uint64_t reserved = converter.reserved_bytes();
    converter.convert(r.data(), g.data(), b.data(), extent.height, extent.width,
                      vk_y.data(), ys, vk_u.data(), cs, vk_v.data(), cs);
    CHECK(converter.reserved_bytes() == reserved);
  }

  bool odd_rejected = false;
  uint8_t byte = 0;
  float sample = 0;
  try {
    converter.convert(&sample, &sample, &sample, 2, 3, &byte, 3, &byte, 1, &byte, 1);
  } catch (const std::invalid_argument&) {
    odd_rejected = true;
  }
  CHECK(odd_rejected);

  bool null_rejected = false;
  try {
    converter.convert(nullptr, &sample, &sample, 2, 2, &byte, 2, &byte, 1, &byte, 1);
  } catch (const std::invalid_argument&) {
    null_rejected = true;
  }
  CHECK(null_rejected);

  bool stride_rejected = false;
  try {
    converter.convert(&sample, &sample, &sample, 2, 2, &byte, 1, &byte, 1, &byte, 1);
  } catch (const std::invalid_argument&) {
    stride_rejected = true;
  }
  CHECK(stride_rejected);
}

}  // namespace

int main() { return ::vidfab::test::run_all(); }
