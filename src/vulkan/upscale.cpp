#include "../upscale/backend.h"
#include "slopfab/vulkan/compute.h"
#include "embedded_upscale_spv.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
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
  CommandList commands_;
  std::vector<Buffer> retained_;
  uint32_t pending_operations_ = 0, batch_limit_ = 16;
  uint64_t pending_bytes_ = 0;
  static constexpr uint64_t kBatchBytes = 64ull << 20;

  void flush() {
    if (!commands_)
      return;
    context_->submit(std::move(commands_)).wait();
    context_->collect();
    retained_.clear();
    pending_operations_ = 0;
    pending_bytes_ = 0;
  }

  CommandList& commands(std::initializer_list<Buffer> resources, uint64_t staging_bytes = 0) {
    uint64_t additional = staging_bytes;
    std::vector<Buffer> added;
    for (const auto& buffer : resources)
      if (std::find(retained_.begin(), retained_.end(), buffer) == retained_.end() &&
          std::find(added.begin(), added.end(), buffer) == added.end()) {
        added.push_back(buffer);
        additional += tensor(buffer).data.size();
      }
    if (commands_ && additional > kBatchBytes - std::min(pending_bytes_, kBatchBytes))
      flush();
    if (!commands_)
      commands_ = context_->begin();
    for (const auto& buffer : resources)
      if (std::find(retained_.begin(), retained_.end(), buffer) == retained_.end()) {
        retained_.push_back(buffer);
        pending_bytes_ += tensor(buffer).data.size();
      }
    pending_bytes_ += staging_bytes;
    return commands_;
  }

  void recorded() {
    // Bound both descriptors and retained activation/upload storage. A single
    // large operator may exceed the byte budget, and is submitted immediately.
    if (++pending_operations_ >= batch_limit_ || pending_bytes_ >= kBatchBytes)
      flush();
  }

public:
  VulkanBackend() {
    // Keep the original per-operator submission available for parity/benchmarks.
    if (const char* batch = std::getenv("SLOPFAB_REALESRGAN_VULKAN_BATCH")) {
      size_t end = 0;
      const auto count = std::stoul(batch, &end);
      if (batch[end] || count < 1 || count > 64)
        throw std::invalid_argument("Real-ESRGAN: Vulkan batch must be between 1 and 64");
      batch_limit_ = uint32_t(count);
    }
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
    context_options.max_compute_binds_per_job = batch_limit_;
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
    // Release completed batch temporaries before a large graph allocation.
    if (commands_ && bytes > kBatchBytes - std::min(pending_bytes_, kBatchBytes))
      flush();
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
    auto& cmd = commands({dst}, bytes);
    cmd.barrier(staging, BufferAccess::kHostWrite, BufferAccess::kTransferRead);
    cmd.copy_buffer(staging, tensor(dst).data, bytes);
    tensor(dst).access = BufferAccess::kTransferWrite;
    recorded();
    return dst;
  }

  std::vector<float> download(const Buffer& src, size_t count) override {
    std::vector<float> result(count);
    const size_t bytes = count * sizeof(float);
    auto staging =
        pool_->allocate(bytes, BufferUsage::kTransferDestination, MemoryUsage::kReadback);
    auto& cmd = commands({src}, bytes);
    auto& t = tensor(src);
    cmd.barrier(t.data, t.access, BufferAccess::kTransferRead);
    cmd.copy_buffer(t.data, staging, bytes);
    cmd.barrier(staging, BufferAccess::kTransferWrite, BufferAccess::kHostRead);
    t.access = BufferAccess::kTransferRead;
    flush();
    staging.read(0, result.data(), bytes);
    return result;
  }

  void run(const Parameters& p, const Buffer& x, const Buffer& y, const Buffer& bias,
           const Buffer& out) override {
    uint32_t gx, gy;
    if (p.op == kGroupNormSilu) {
      gx = 32;
      gy = 1;
    } else if (p.op == kConv || p.op == kConv3d) {
      gx = (p.frames * p.height * p.width + 15) / 16;
      gy = (p.output + 15) / 16;
    } else {
      const uint32_t groups = (p.count + 255) / 256;
      gx = std::min(groups, limits_.max_compute_workgroup_count[0]);
      gy = (groups + gx - 1) / gx;
    }
    if (gx > limits_.max_compute_workgroup_count[0] || gy > limits_.max_compute_workgroup_count[1])
      throw std::length_error("Real-ESRGAN: Vulkan dispatch limit exceeded; reduce tile size");
    auto& cmd = commands({x, y, bias, out});
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
    dst.access = BufferAccess::kComputeWrite;
    recorded();
  }
};
} // namespace

std::unique_ptr<Backend> make_vulkan_backend() {
  return std::make_unique<VulkanBackend>();
}
} // namespace slopfab::upscale_detail
