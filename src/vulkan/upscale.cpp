#include "../upscale/backend.h"
#include "slopfab/vulkan/compute.h"
#include "embedded_upscale_spv.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace slopfab::upscale_detail {
namespace {
using namespace vulkan;

struct VulkanTensor : Tensor {
  vulkan::Buffer data;
  BufferAccess access = BufferAccess::kComputeWrite;
};

VulkanTensor& tensor(const Buffer& b) {
  return static_cast<VulkanTensor&>(*b);
}

class VulkanBackend final : public Backend {
  Instance instance_;
  Device device_;
  DeviceInfo limits_;
  std::unique_ptr<BufferPool> pool_;
  std::unique_ptr<ComputeContext> context_;
  ComputePipeline pipeline_;

public:
  VulkanBackend() {
    instance_ = Instance::create();
    const auto devices = instance_.enumerate_devices();
    if (devices.empty())
      throw std::runtime_error("Real-ESRGAN: no Vulkan compute device");
    limits_ = devices[0].info();
    DeviceOptions options;
    options.enable_timeline_semaphore = true;
    device_ = devices[0].create_device(options);
    pool_ = std::make_unique<BufferPool>(device_, 4ull << 20);
    ComputeContextOptions context_options;
    context_options.max_in_flight = 1;
    context_options.max_storage_bindings = 4;
    context_ = std::make_unique<ComputeContext>(device_, context_options);
    std::vector<uint32_t> spirv(sizeof(kUpscaleSpirv) / 4);
    std::memcpy(spirv.data(), kUpscaleSpirv, sizeof(kUpscaleSpirv));
    ComputePipelineOptions pipeline_options;
    pipeline_options.storage_binding_count = 4;
    pipeline_options.push_constant_bytes = sizeof(Parameters);
    pipeline_options.local_size[0] = pipeline_options.local_size[1] = 16;
    pipeline_ = ComputePipeline::create(device_, spirv, pipeline_options);
  }

  Buffer allocate(size_t count) override {
    const uint64_t bytes = uint64_t(count) * sizeof(float);
    if (bytes > limits_.max_storage_buffer_bytes)
      throw std::length_error("Real-ESRGAN: Vulkan storage limit exceeded; reduce tile size");
    auto t = std::make_shared<VulkanTensor>();
    t->data = pool_->allocate(bytes,
                              BufferUsage::kStorage | BufferUsage::kTransferSource |
                                  BufferUsage::kTransferDestination,
                              MemoryUsage::kDevice);
    return t;
  }

  Buffer upload(const std::vector<float>& v) override {
    auto dst = allocate(v.size());
    const size_t bytes = v.size() * sizeof(float);
    auto staging = pool_->allocate(bytes, BufferUsage::kTransferSource, MemoryUsage::kUpload);
    staging.write(0, v.data(), bytes);
    auto cmd = context_->begin();
    cmd.barrier(staging, BufferAccess::kHostWrite, BufferAccess::kTransferRead);
    cmd.copy_buffer(staging, tensor(dst).data, bytes);
    context_->submit(std::move(cmd)).wait();
    tensor(dst).access = BufferAccess::kTransferWrite;
    return dst;
  }

  std::vector<float> download(const Buffer& src, size_t count) override {
    std::vector<float> result(count);
    const size_t bytes = count * sizeof(float);
    auto staging =
        pool_->allocate(bytes, BufferUsage::kTransferDestination, MemoryUsage::kReadback);
    auto cmd = context_->begin();
    auto& t = tensor(src);
    cmd.barrier(t.data, t.access, BufferAccess::kTransferRead);
    cmd.copy_buffer(t.data, staging, bytes);
    cmd.barrier(staging, BufferAccess::kTransferWrite, BufferAccess::kHostRead);
    context_->submit(std::move(cmd)).wait();
    t.access = BufferAccess::kTransferRead;
    staging.read(0, result.data(), bytes);
    return result;
  }

  void run(const Parameters& p, const Buffer& x, const Buffer& y, const Buffer& bias,
           const Buffer& out) override {
    uint32_t gx, gy;
    if (p.op == kConv) {
      gx = (p.height * p.width + 15) / 16;
      gy = (p.output + 15) / 16;
    } else {
      const uint32_t groups = (p.count + 255) / 256;
      gx = std::min(groups, limits_.max_compute_workgroup_count[0]);
      gy = (groups + gx - 1) / gx;
    }
    if (gx > limits_.max_compute_workgroup_count[0] || gy > limits_.max_compute_workgroup_count[1])
      throw std::length_error("Real-ESRGAN: Vulkan dispatch limit exceeded; reduce tile size");
    auto cmd = context_->begin();
    for (auto t : {x, y, bias}) {
      auto& v = tensor(t);
      cmd.barrier(v.data, v.access, BufferAccess::kComputeRead);
      v.access = BufferAccess::kComputeRead;
    }
    auto& dst = tensor(out);
    cmd.barrier(dst.data, dst.access, BufferAccess::kComputeWrite);
    cmd.bind_compute(
        pipeline_,
        {{0, &tensor(x).data}, {1, &tensor(y).data}, {2, &tensor(bias).data}, {3, &dst.data}});
    cmd.push_constants(&p, sizeof(p));
    cmd.dispatch(gx, gy);
    context_->submit(std::move(cmd)).wait();
    dst.access = BufferAccess::kComputeWrite;
  }
};
} // namespace

std::unique_ptr<Backend> make_vulkan_backend() {
  return std::make_unique<VulkanBackend>();
}
} // namespace slopfab::upscale_detail
