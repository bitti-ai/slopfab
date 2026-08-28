#include "vidfab/vulkan/yuv_converter.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "embedded_yuv_spv.h"
#include "vidfab/vulkan/compute.h"
#include "vidfab/vulkan/runtime.h"

namespace vidfab::vulkan {

struct Yuv420Converter::Impl {
  struct Geometry {
    uint32_t width;
    uint32_t height;
    uint32_t pixels;
  };

  Instance instance;
  Device device;
  std::unique_ptr<ComputeContext> context;
  std::unique_ptr<BufferPool> pool;
  ComputePipeline pipeline;
  Buffer upload;
  Buffer input;
  Buffer output;
  Buffer readback;
  std::vector<uint32_t> host_output;
  std::vector<StorageBinding> bindings;
  DeviceInfo limits;
  uint64_t capacity = 0;
  uint64_t high_water = 0;
  std::string name;

  explicit Impl(uint32_t device_index) : bindings(2) {
    instance = Instance::create();
    std::vector<PhysicalDevice> physical = instance.enumerate_devices();
    if (device_index >= physical.size()) {
      throw std::runtime_error("vulkan output: device index " + std::to_string(device_index) +
                               " is unavailable (" + std::to_string(physical.size()) +
                               " compute devices visible)");
    }
    if (!physical[device_index].info().timeline_semaphore) {
      throw std::runtime_error("vulkan output: selected device has no timeline semaphore");
    }
    limits = physical[device_index].info();
    name = limits.name;
    DeviceOptions device_options;
    device_options.enable_timeline_semaphore = true;
    device = physical[device_index].create_device(device_options);
    ComputeContextOptions context_options;
    context_options.max_in_flight = 1;
    context_options.max_storage_bindings = 2;
    context = std::make_unique<ComputeContext>(device, context_options);
    pool = std::make_unique<BufferPool>(device, 4ull << 20);

    static_assert(sizeof(detail::kRgbToYuvSpirv) % sizeof(uint32_t) == 0,
                  "embedded SPIR-V must contain whole words");
    std::vector<uint32_t> spirv(sizeof(detail::kRgbToYuvSpirv) / sizeof(uint32_t));
    std::memcpy(spirv.data(), detail::kRgbToYuvSpirv, sizeof(detail::kRgbToYuvSpirv));
    ComputePipelineOptions pipeline_options;
    pipeline_options.storage_binding_count = 2;
    pipeline_options.push_constant_bytes = sizeof(Geometry);
    pipeline_options.local_size[0] = 64;
    pipeline = ComputePipeline::create(device, spirv, pipeline_options);
    bindings[0].binding = 0;
    bindings[0].buffer = &input;
    bindings[1].binding = 1;
    bindings[1].buffer = &output;
  }

  void ensure_capacity(uint64_t pixels) {
    if (pixels <= capacity) return;
    context->collect();
    const uint64_t rgb_bytes = pixels * 3 * sizeof(float);
    const uint64_t yuv_values = pixels + pixels / 2;
    const uint64_t yuv_words = (yuv_values + 3) / 4;
    const uint64_t yuv_bytes = yuv_words * sizeof(uint32_t);
    try {
      Buffer new_upload =
          pool->allocate(rgb_bytes, BufferUsage::kTransferSource, MemoryUsage::kUpload);
      Buffer new_input = pool->allocate(rgb_bytes,
                                        BufferUsage::kTransferDestination | BufferUsage::kStorage,
                                        MemoryUsage::kDevice);
      Buffer new_output = pool->allocate(yuv_bytes,
                                         BufferUsage::kStorage | BufferUsage::kTransferSource,
                                         MemoryUsage::kDevice);
      Buffer new_readback = pool->allocate(yuv_bytes, BufferUsage::kTransferDestination,
                                           MemoryUsage::kReadback);
      std::vector<uint32_t> new_host_output(static_cast<size_t>(yuv_words));
      upload = std::move(new_upload);
      input = std::move(new_input);
      output = std::move(new_output);
      readback = std::move(new_readback);
      host_output = std::move(new_host_output);
      capacity = pixels;
    } catch (...) {
      pool->trim();
      throw;
    }
    pool->trim();
    high_water = std::max(high_water, pool->reserved_bytes());
  }
};

Yuv420Converter::Yuv420Converter(uint32_t device_index)
    : impl_(std::make_unique<Impl>(device_index)) {}
Yuv420Converter::~Yuv420Converter() = default;
Yuv420Converter::Yuv420Converter(Yuv420Converter&&) noexcept = default;
Yuv420Converter& Yuv420Converter::operator=(Yuv420Converter&&) noexcept = default;

void Yuv420Converter::convert(const float* r, const float* g, const float* b,
                              int height, int width, uint8_t* y_plane, int y_stride,
                              uint8_t* u_plane, int u_stride, uint8_t* v_plane,
                              int v_stride) {
  if (!impl_) throw std::logic_error("vulkan output: empty converter");
  if (r == nullptr || g == nullptr || b == nullptr || y_plane == nullptr ||
      u_plane == nullptr || v_plane == nullptr) {
    throw std::invalid_argument("vulkan output: RGB and YUV plane pointers must be non-null");
  }
  if (width <= 0 || height <= 0 || (width & 1) != 0 || (height & 1) != 0) {
    throw std::invalid_argument("vulkan output: YUV420 requires positive even dimensions");
  }
  if (y_stride < width || u_stride < width / 2 || v_stride < width / 2) {
    throw std::invalid_argument("vulkan output: destination stride is smaller than its plane");
  }
  const uint64_t pixels = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
  const uint64_t rgb_values = pixels * 3;
  const uint64_t yuv_values = pixels + pixels / 2;
  const uint64_t yuv_words = (yuv_values + 3) / 4;
  if (rgb_values > std::numeric_limits<uint32_t>::max() ||
      yuv_values > std::numeric_limits<uint32_t>::max() ||
      yuv_words > std::numeric_limits<size_t>::max() / sizeof(uint32_t)) {
    throw std::overflow_error("vulkan output: frame dimensions overflow converter limits");
  }
  const uint64_t rgb_bytes = rgb_values * sizeof(float);
  const uint64_t yuv_bytes = yuv_words * sizeof(uint32_t);
  const uint64_t dispatch_groups = (yuv_words + 63) / 64;
  const DeviceInfo& limits = impl_->limits;
  if ((limits.max_storage_buffer_bytes != 0 &&
       (rgb_bytes > limits.max_storage_buffer_bytes ||
        yuv_bytes > limits.max_storage_buffer_bytes)) ||
      (limits.max_allocation_bytes != 0 &&
       (rgb_bytes > limits.max_allocation_bytes || yuv_bytes > limits.max_allocation_bytes)) ||
      dispatch_groups > limits.max_compute_workgroup_count[0]) {
    throw std::length_error("vulkan output: frame exceeds selected device compute limits");
  }
  impl_->ensure_capacity(pixels);
  const uint64_t plane_bytes = pixels * sizeof(float);
  impl_->upload.write(0, r, plane_bytes);
  impl_->upload.write(plane_bytes, g, plane_bytes);
  impl_->upload.write(2 * plane_bytes, b, plane_bytes);

  Impl::Geometry geometry{static_cast<uint32_t>(width), static_cast<uint32_t>(height),
                          static_cast<uint32_t>(pixels)};
  CommandList commands = impl_->context->begin();
  commands.barrier(impl_->upload, BufferAccess::kHostWrite, BufferAccess::kTransferRead,
                   0, rgb_bytes);
  commands.copy_buffer(impl_->upload, impl_->input, rgb_bytes);
  commands.barrier(impl_->input, BufferAccess::kTransferWrite, BufferAccess::kComputeRead,
                   0, rgb_bytes);
  impl_->bindings[0].bytes = rgb_bytes;
  impl_->bindings[1].bytes = yuv_bytes;
  commands.bind_compute(impl_->pipeline, impl_->bindings);
  commands.push_constants(&geometry, sizeof(geometry));
  commands.dispatch(static_cast<uint32_t>(dispatch_groups));
  commands.barrier(impl_->output, BufferAccess::kComputeWrite, BufferAccess::kTransferRead,
                   0, yuv_bytes);
  commands.copy_buffer(impl_->output, impl_->readback, yuv_bytes);
  commands.barrier(impl_->readback, BufferAccess::kTransferWrite, BufferAccess::kHostRead,
                   0, yuv_bytes);
  Submission completion = impl_->context->submit(std::move(commands));
  completion.wait();
  impl_->context->collect();
  impl_->readback.read(0, impl_->host_output.data(), yuv_bytes);

  const auto byte_at = [&](uint64_t index) {
    const uint32_t word = impl_->host_output[static_cast<size_t>(index / 4)];
    return static_cast<uint8_t>((word >> ((index % 4) * 8)) & 0xffu);
  };
  for (int row = 0; row < height; ++row) {
    uint8_t* destination = y_plane + static_cast<size_t>(row) * y_stride;
    const uint64_t source = static_cast<uint64_t>(row) * width;
    for (int x = 0; x < width; ++x) destination[x] = byte_at(source + x);
  }
  for (int row = 0; row < height / 2; ++row) {
    uint8_t* ud = u_plane + static_cast<size_t>(row) * u_stride;
    uint8_t* vd = v_plane + static_cast<size_t>(row) * v_stride;
    const uint64_t chroma_row = static_cast<uint64_t>(row) * (width / 2);
    for (int x = 0; x < width / 2; ++x) {
      ud[x] = byte_at(pixels + chroma_row + x);
      vd[x] = byte_at(pixels + pixels / 4 + chroma_row + x);
    }
  }
}

uint64_t Yuv420Converter::reserved_bytes() const {
  return impl_ ? impl_->pool->reserved_bytes() : 0;
}
uint64_t Yuv420Converter::high_water_bytes() const { return impl_ ? impl_->high_water : 0; }
uint64_t Yuv420Converter::capacity_pixels() const { return impl_ ? impl_->capacity : 0; }
const char* Yuv420Converter::device_name() const { return impl_ ? impl_->name.c_str() : ""; }

}  // namespace vidfab::vulkan
