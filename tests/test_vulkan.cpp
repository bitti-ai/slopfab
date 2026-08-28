#include "harness.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "vidfab/vulkan/runtime.h"
#include "vidfab/vulkan/compute.h"
#include "vidfab/vulkan/tensor.h"
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

std::vector<uint8_t> read_bytes(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
}

uint32_t float_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

uint16_t reference_bf16(float value) {
  uint32_t bits = float_bits(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return 0x7fffu;
  }
  return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
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

VIDFAB_TEST(vulkan_tensor_batch_and_workspace) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!Instance::available()) return;
  Instance instance = Instance::create();
  const auto devices = instance.enumerate_devices();
  if (devices.empty() || !devices.front().info().timeline_semaphore) return;
  DeviceOptions options;
  options.enable_timeline_semaphore = true;
  Device device = devices.front().create_device(options);
  TensorContext tensors(device);
  TensorContext other(device);
  CHECK(tensors.full_fp32_add_exactness() ==
        devices.front().info().fp32_denorm_preserve);
  bool exact_gate_rejected = false;
  try {
    tensors.require_full_fp32_add_exactness();
  } catch (const std::runtime_error&) {
    exact_gate_rejected = true;
  }
  CHECK(exact_gate_rejected == !devices.front().info().fp32_denorm_preserve);
  const uint64_t extent = 257;
  const TensorLayout layout = TensorLayout::contiguous(&extent, 1);
  DeviceTensor a = tensors.allocate(layout);
  DeviceTensor b = tensors.allocate(layout);
  DeviceTensor sum = tensors.allocate(layout);
  DeviceTensor copied = tensors.allocate(layout);
  DeviceTensor tail = tensors.allocate(layout);

  std::vector<float> host_a(extent), host_b(extent);
  for (size_t i = 0; i < host_a.size(); ++i) {
    host_a[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 16.0f;
    host_b[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 32.0f;
  }
  const uint32_t special_a[] = {
      0x00000000u, 0x80000000u, 0x00000001u, 0x00800000u,
      0x3f800000u, 0x3f800000u, 0x7f800000u, 0xff800000u,
      0x7fc12345u, 0x7fa54321u};
  const uint32_t special_b[] = {
      0x80000000u, 0x80000000u, 0x00000001u, 0x807fffffu,
      0x33800000u, 0x33800001u, 0x3f800000u, 0xbf800000u,
      0x00000000u, 0x00000000u};
  static_assert(sizeof(float) == sizeof(uint32_t));
  std::memcpy(host_a.data(), special_a, sizeof(special_a));
  std::memcpy(host_b.data(), special_b, sizeof(special_b));
  tensors.upload(a, host_a.data(), extent);
  tensors.upload(b, host_b.data(), extent);

  // Access state is speculative during recording, so a context admits exactly
  // one open recorder/boundary operation. Discard releases the lease and rolls
  // state back; successful submit releases it before GPU completion.
  {
    TensorBatch open = tensors.begin_batch();
    open.add(a, b, sum);
    bool second_begin_rejected = false;
    try {
      (void)tensors.begin_batch();
    } catch (const std::logic_error&) {
      second_begin_rejected = true;
    }
    CHECK(second_begin_rejected);
    bool upload_rejected = false;
    try {
      tensors.upload(a, host_a.data(), extent);
    } catch (const std::logic_error&) {
      upload_rejected = true;
    }
    CHECK(upload_rejected);
    bool download_rejected = false;
    try {
      tensors.download(a, host_a.data(), extent);
    } catch (const std::logic_error&) {
      download_rejected = true;
    }
    CHECK(download_rejected);
  }
  tensors.upload(a, host_a.data(), extent);
  TensorBatch releasing = tensors.begin_batch();
  releasing.add(a, b, sum);
  Submission releasing_token = releasing.submit();
  TensorBatch after_submit = tensors.begin_batch();
  after_submit.add(a, b, copied);
  Submission after_submit_token = after_submit.submit();
  releasing_token.wait();
  after_submit_token.wait();

  // Four operators, one command buffer, one timeline submission. The first
  // copy covers payload NaNs, signed zero, subnormal and infinities exactly;
  // arithmetic comparisons below use the ordinary finite tail, while the
  // CUDA/Vulkan test covers special-value arithmetic on both devices.
  TensorBatch batch = tensors.begin_batch();
  batch.copy(a, copied);
  batch.add(a, b, sum);
  batch.copy(sum, tail);
  batch.add(sum, b, tail);
  Submission done = batch.submit();
  CHECK(done.value() != 0);
  done.wait();

  std::vector<float> got_sum(extent), got_copy(extent);
  tensors.download(sum, got_sum.data(), extent);
  tensors.download(copied, got_copy.data(), extent);
  CHECK(std::memcmp(host_a.data(), got_copy.data(), extent * sizeof(float)) == 0);
  for (size_t i = 0; i < host_a.size(); ++i) {
    const uint32_t input_bits = [&] {
      uint32_t bits = 0;
      std::memcpy(&bits, &host_a[i], sizeof(bits));
      return bits;
    }();
    if ((input_bits & 0x7f800000u) == 0x7f800000u &&
        (input_bits & 0x007fffffu) != 0) continue;
    if (!tensors.full_fp32_add_exactness() && (i == 2 || i == 3)) continue;
    const float expected = host_a[i] + host_b[i];
    uint32_t expected_bits = 0, actual_bits = 0;
    std::memcpy(&expected_bits, &expected, sizeof(expected_bits));
    std::memcpy(&actual_bits, &got_sum[i], sizeof(actual_bits));
    CHECK_MSG(expected_bits == actual_bits,
              "fp32 add bit mismatch at %zu: %08x != %08x", i,
              expected_bits, actual_bits);
  }

  // An abandoned recording restores speculative access tracking and releases
  // its command slot; the same tensors remain immediately usable.
  {
    TensorBatch abandoned = tensors.begin_batch();
    abandoned.add(a, b, sum);
  }
  tensors.copy(a, copied);
  tensors.download(copied, got_copy.data(), extent);
  CHECK(std::memcmp(got_copy.data(), host_a.data(), extent * sizeof(float)) == 0);

  bool cross_context_rejected = false;
  try {
    DeviceTensor foreign = other.allocate(layout);
    TensorBatch invalid = tensors.begin_batch();
    invalid.copy(a, foreign);
  } catch (const std::invalid_argument&) {
    cross_context_rejected = true;
  }
  CHECK(cross_context_rejected);
  CHECK(a.view().context != other.allocate(layout).view().context);

  const uint64_t tail_extent = 8;
  const TensorLayout tail_layout = TensorLayout::contiguous(&tail_extent, 1);
  DeviceTensorView tail_view = a.view().slice(16, tail_layout, 16);
  CHECK(tail_view.byte_offset == 16);
  bool overflow_rejected = false;
  try {
    (void)tail_view.slice(std::numeric_limits<uint64_t>::max(), tail_layout, 4);
  } catch (const std::overflow_error&) {
    overflow_rejected = true;
  }
  CHECK(overflow_rejected);

  TensorWorkspace& workspace = tensors.workspace();
  const uint64_t before_workspace = tensors.reserved_bytes();
  workspace.reserve(256);
  WorkspaceSpan old_span = workspace.allocate(64, 32);
  CHECK(workspace.valid(old_span));
  TensorWorkspace foreign_workspace(device, 1024);
  foreign_workspace.reserve(256);
  CHECK(!foreign_workspace.valid(old_span));
  workspace.reset();
  workspace.reserve(512);
  CHECK(!workspace.valid(old_span));
  CHECK(workspace.generation() != old_span.generation);
  CHECK(tensors.reserved_bytes() > before_workspace);

  // Descriptor sets and pooled buffers reach a fixed high-water after warmup.
  tensors.add(a, b, sum);
  const uint64_t warm_reserved = tensors.reserved_bytes();
  const uint64_t warm_descriptors = tensors.descriptor_set_allocations();
  for (int repeat = 0; repeat < 8; ++repeat) {
    TensorBatch repeated = tensors.begin_batch();
    repeated.add(a, b, sum);
    repeated.copy(sum, copied);
    repeated.submit().wait();
    CHECK(tensors.reserved_bytes() == warm_reserved);
    CHECK(tensors.descriptor_set_allocations() == warm_descriptors);
  }

  // The advertised 32-op bound is exact. A rejected 33rd record causes the
  // whole scope to be abandoned; rollback leaves the next batch usable.
  bool thirty_third_rejected = false;
  {
    TensorBatch bounded = tensors.begin_batch();
    for (int i = 0; i < 32; ++i) bounded.add(a, b, sum);
    try {
      bounded.add(a, b, sum);
    } catch (const std::logic_error&) {
      thirty_third_rejected = true;
    }
    bool poisoned_submit_rejected = false;
    try {
      (void)bounded.submit();
    } catch (const std::logic_error&) {
      poisoned_submit_rejected = true;
    }
    CHECK(poisoned_submit_rejected);
  }
  CHECK(thirty_third_rejected);
  tensors.add(a, b, sum);

  // Two full graph chunks can be outstanding on separate bounded flight
  // slots. The second begin neither waits for nor resets the first slot.
  const uint64_t before_chunks = tensors.reserved_bytes();
  TensorBatch first_chunk = tensors.begin_batch();
  for (int i = 0; i < 32; ++i) first_chunk.add(a, b, sum);
  Submission first_token = first_chunk.submit();
  TensorBatch second_chunk = tensors.begin_batch();
  for (int i = 0; i < 32; ++i) second_chunk.add(a, b, copied);
  Submission second_token = second_chunk.submit();
  CHECK(second_token.value() > first_token.value());
  // A third begin applies oldest-slot timeline backpressure, then reuses that
  // slot and its descriptor cache rather than growing resources.
  TensorBatch third_chunk = tensors.begin_batch();
  third_chunk.add(a, b, tail);
  Submission third_token = third_chunk.submit();
  first_token.wait();
  second_token.wait();
  third_token.wait();
  CHECK(third_token.value() > second_token.value());
  CHECK(tensors.reserved_bytes() == before_chunks);
  const uint64_t descriptor_high_water = tensors.descriptor_set_allocations();
  TensorBatch reuse_first = tensors.begin_batch();
  for (int i = 0; i < 32; ++i) reuse_first.add(a, b, sum);
  Submission reuse_first_token = reuse_first.submit();
  TensorBatch reuse_second = tensors.begin_batch();
  for (int i = 0; i < 32; ++i) reuse_second.add(a, b, copied);
  Submission reuse_second_token = reuse_second.submit();
  reuse_first_token.wait();
  reuse_second_token.wait();
  CHECK(tensors.descriptor_set_allocations() == descriptor_high_water);

  // Numeric identities never recycle even after owners and native handles are
  // destroyed. This covers both tensor contexts and scratch workspaces.
  uintptr_t old_tensor_context = 0;
  {
    TensorContext temporary(device);
    DeviceTensor old = temporary.allocate(layout);
    old_tensor_context = old.view().context;
  }
  TensorContext replacement_context(device);
  CHECK(replacement_context.allocate(layout).view().context != old_tensor_context);
  uintptr_t old_workspace_context = 0;
  {
    TensorWorkspace old_workspace(device, 1024);
    old_workspace.reserve(64);
    old_workspace_context = old_workspace.allocate(16, 4).context;
  }
  TensorWorkspace replacement_workspace(device, 1024);
  replacement_workspace.reserve(64);
  CHECK(replacement_workspace.allocate(16, 4).context != old_workspace_context);

  // Submitted jobs own their tensors and Vulkan resources through completion,
  // even when every caller wrapper and the TensorContext are dropped.
  Submission retained;
  {
    TensorContext temporary(device);
    DeviceTensor left = temporary.allocate(layout);
    DeviceTensor right = temporary.allocate(layout);
    DeviceTensor output = temporary.allocate(layout);
    temporary.upload(left, host_a.data(), extent);
    temporary.upload(right, host_b.data(), extent);
    TensorBatch live = temporary.begin_batch();
    live.add(left, right, output);
    retained = live.submit();
  }
  retained.wait();

  TensorWorkspace movable_workspace(device, 1024);
  TensorWorkspace moved_workspace = std::move(movable_workspace);
  CHECK(moved_workspace.capacity() == 0);
  CHECK(movable_workspace.capacity() == 0);
  CHECK(movable_workspace.used() == 0);
  CHECK(movable_workspace.generation() == 0);
  CHECK(movable_workspace.reserved_bytes() == 0);
  movable_workspace.reset();
  bool moved_workspace_rejected = false;
  try {
    movable_workspace.reserve(16);
  } catch (const std::logic_error&) {
    moved_workspace_rejected = true;
  }
  CHECK(moved_workspace_rejected);

  TensorContext movable_context(device);
  TensorContext moved_context = std::move(movable_context);
  CHECK(moved_context.descriptor_set_allocations() == 0);
  CHECK(movable_context.reserved_bytes() == 0);
  CHECK(!movable_context.full_fp32_add_exactness());
  bool moved_context_rejected = false;
  try {
    (void)movable_context.allocate(layout);
  } catch (const std::logic_error&) {
    moved_context_rejected = true;
  }
  CHECK(moved_context_rejected);
}

VIDFAB_TEST(vulkan_tensor_layout_and_conversion_ops) {
  using namespace vidfab;
  using namespace vidfab::vulkan;
  if (!Instance::available()) return;
  Instance instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty() || !physical.front().info().timeline_semaphore ||
      !physical.front().info().fp32_signed_zero_inf_nan_preserve ||
      !physical.front().info().fp32_rounding_rte) return;
  DeviceOptions options; options.enable_timeline_semaphore = true;
  Device device = physical.front().create_device(options);
  TensorContext tensors(device);

  const uint64_t shape_data[] = {3, 5};
  const uint64_t transposed_data[] = {5, 3};
  const TensorLayout shape = TensorLayout::contiguous(shape_data, 2);
  const TensorLayout transposed_shape = TensorLayout::contiguous(transposed_data, 2);
  DeviceTensor input = tensors.allocate(shape);
  DeviceTensor bf16 = tensors.allocate(shape, ScalarType::kBFloat16);
  DeviceTensor fp16 = tensors.allocate(shape, ScalarType::kFloat16);
  DeviceTensor bf16_back = tensors.allocate(shape);
  DeviceTensor fp16_back = tensors.allocate(shape);
  DeviceTensor transposed = tensors.allocate(transposed_shape);
  DeviceTensor biased = tensors.allocate(shape);
  const uint64_t bias_extent = 5;
  DeviceTensor bias = tensors.allocate(TensorLayout::contiguous(&bias_extent, 1));

  const uint32_t patterns[] = {
      0x00000000u, 0x80000000u, 0x3f800000u, 0xc0000000u, 0x477fe000u,
      0x7f800000u, 0xff800000u, 0x33800000u, 0x33800001u, 0x00000001u,
      0x7fc12345u, 0x7fa54321u, 0x38800000u, 0x3f000000u, 0xbf000000u};
  std::vector<float> host(std::size(patterns));
  std::memcpy(host.data(), patterns, sizeof(patterns));
  const float host_bias[] = {1.0f, -1.0f, 0.25f, -0.25f, 2.0f};
  tensors.upload(input, host.data(), host.size());
  tensors.upload(bias, host_bias, std::size(host_bias));

  TensorBatch batch = tensors.begin_batch();
  batch.convert(input, bf16);
  batch.convert(bf16, bf16_back);
  batch.convert(input, fp16);
  batch.convert(fp16, fp16_back);
  batch.transpose_2d(input, transposed);
  batch.add_bias(input, bias, biased);
  Submission done = batch.submit();
  done.wait();

  std::vector<uint16_t> got_bf16(host.size()), got_fp16(host.size());
  std::vector<float> got_bf16_back(host.size()), got_fp16_back(host.size());
  std::vector<float> got_transposed(host.size()), got_biased(host.size());
  tensors.download_bytes(bf16, got_bf16.data(), got_bf16.size() * sizeof(uint16_t));
  tensors.download_bytes(fp16, got_fp16.data(), got_fp16.size() * sizeof(uint16_t));
  tensors.download(bf16_back, got_bf16_back.data(), got_bf16_back.size());
  tensors.download(fp16_back, got_fp16_back.data(), got_fp16_back.size());
  tensors.download(transposed, got_transposed.data(), got_transposed.size());
  tensors.download(biased, got_biased.data(), got_biased.size());
  for (size_t i = 0; i < host.size(); ++i) {
    CHECK(got_bf16[i] == reference_bf16(host[i]));
    CHECK(float_bits(got_bf16_back[i]) == (static_cast<uint32_t>(got_bf16[i]) << 16));
  }
  const uint16_t expected_half[] = {
      0x0000u, 0x8000u, 0x3c00u, 0xc000u, 0x7bffu, 0x7c00u, 0xfc00u,
      0x0001u, 0x0001u, 0x0000u, 0x7fffu, 0x7fffu, 0x0400u, 0x3800u, 0xb800u};
  for (size_t i = 0; i < host.size(); ++i) {
    CHECK_MSG(got_fp16[i] == expected_half[i], "fp16 mismatch %zu: %04x != %04x",
              i, got_fp16[i], expected_half[i]);
  }
  for (size_t row = 0; row < 3; ++row) {
    for (size_t col = 0; col < 5; ++col) {
      CHECK(float_bits(got_transposed[col * 3 + row]) == float_bits(host[row * 5 + col]));
      if (std::isfinite(host[row * 5 + col])) {
        CHECK(float_bits(got_biased[row * 5 + col]) ==
              float_bits(host[row * 5 + col] + host_bias[col]));
      }
    }
  }

  const uint64_t matrix_shape_data[] = {3, 4};
  const uint64_t selected_shape_data[] = {2, 4};
  const TensorLayout matrix_shape = TensorLayout::contiguous(matrix_shape_data, 2);
  const TensorLayout selected_shape = TensorLayout::contiguous(selected_shape_data, 2);
  const uint64_t index_count = 2;
  DeviceTensor matrix = tensors.allocate(matrix_shape);
  DeviceTensor indices = tensors.allocate(TensorLayout::contiguous(&index_count, 1),
                                          ScalarType::kInt32);
  DeviceTensor gathered = tensors.allocate(selected_shape);
  DeviceTensor scattered = tensors.allocate(matrix_shape);
  std::vector<float> matrix_host(12), scatter_initial(12, -99.0f);
  for (size_t i = 0; i < matrix_host.size(); ++i) matrix_host[i] = static_cast<float>(i + 1);
  const int32_t host_indices[] = {2, 0};
  tensors.upload(matrix, matrix_host.data(), matrix_host.size());
  tensors.upload_bytes(indices, host_indices, sizeof(host_indices));
  tensors.upload(scattered, scatter_initial.data(), scatter_initial.size());
  TensorBatch indexed = tensors.begin_batch();
  indexed.gather_rows(matrix, indices, gathered);
  indexed.scatter_rows(gathered, indices, scattered);
  indexed.submit().wait();
  std::vector<float> gathered_host(8), scattered_host(12);
  tensors.download(gathered, gathered_host.data(), gathered_host.size());
  tensors.download(scattered, scattered_host.data(), scattered_host.size());
  for (size_t col = 0; col < 4; ++col) {
    CHECK(gathered_host[col] == matrix_host[8 + col]);
    CHECK(gathered_host[4 + col] == matrix_host[col]);
    CHECK(scattered_host[8 + col] == matrix_host[8 + col]);
    CHECK(scattered_host[col] == matrix_host[col]);
    CHECK(scattered_host[4 + col] == -99.0f);
  }

  const uint64_t heads_elements = 24;
  DeviceTensor heads_input = tensors.allocate(TensorLayout::contiguous(&heads_elements, 1));
  DeviceTensor token_bf16 = tensors.allocate(TensorLayout::contiguous(&heads_elements, 1),
                                             ScalarType::kBFloat16);
  std::vector<float> heads_host(heads_elements);
  for (size_t i = 0; i < heads_host.size(); ++i) heads_host[i] = static_cast<float>(i) / 8.0f;
  tensors.upload(heads_input, heads_host.data(), heads_host.size());
  TensorBatch layout_batch = tensors.begin_batch();
  layout_batch.heads_to_tokens_bf16(heads_input, token_bf16, 2, 3, 4);
  layout_batch.submit().wait();
  std::vector<uint16_t> token_host(heads_elements);
  tensors.download_bytes(token_bf16, token_host.data(), token_host.size() * 2);
  for (uint32_t s = 0; s < 3; ++s) for (uint32_t h = 0; h < 2; ++h)
    for (uint32_t d = 0; d < 4; ++d) {
      const size_t out = (s * 2 + h) * 4 + d;
      const size_t in = (h * 3 + s) * 4 + d;
      CHECK(token_host[out] == reference_bf16(heads_host[in]));
    }

  const uint64_t depth_elements = 8;
  DeviceTensor depth_in = tensors.allocate(TensorLayout::contiguous(&depth_elements, 1));
  DeviceTensor depth_out = tensors.allocate(TensorLayout::contiguous(&depth_elements, 1));
  const float depth_values[] = {0, 1, 2, 3, 4, 5, 6, 7};
  tensors.upload(depth_in, depth_values, depth_elements);
  TensorBatch depth_batch = tensors.begin_batch();
  depth_batch.depth_to_space(depth_in, depth_out, 1, 1, 2, 1, 1, 2);
  depth_batch.submit().wait();
  std::vector<float> depth_result(depth_elements);
  tensors.download(depth_out, depth_result.data(), depth_elements);
  const float depth_expected[] = {0, 1, 4, 5, 2, 3, 6, 7};
  CHECK(std::memcmp(depth_result.data(), depth_expected, sizeof(depth_expected)) == 0);

  // Deterministic validation happens before access state or command recording,
  // so a caller may catch it and still submit the earlier valid operation.
  bool alias_rejected = false;
  TensorBatch recoverable = tensors.begin_batch();
  recoverable.convert(input, bf16);
  try {
    recoverable.transpose_2d(input, input);
  } catch (const std::invalid_argument&) {
    alias_rejected = true;
  }
  CHECK(alias_rejected);
  recoverable.submit().wait();

  TensorContext other(device);
  DeviceTensor foreign = other.allocate(shape);
  bool context_rejected = false;
  try {
    TensorBatch invalid = tensors.begin_batch();
    invalid.convert(foreign, bf16);
  } catch (const std::invalid_argument&) {
    context_rejected = true;
  }
  CHECK(context_rejected);

  bool overflow_rejected = false;
  try {
    TensorBatch invalid = tensors.begin_batch();
    invalid.heads_to_tokens_bf16(input, bf16,
                                 std::numeric_limits<uint32_t>::max(),
                                 std::numeric_limits<uint32_t>::max(),
                                 std::numeric_limits<uint32_t>::max());
  } catch (const std::overflow_error&) {
    overflow_rejected = true;
  }
  CHECK(overflow_rejected);

  // Invalid trusted device indices are still memory-safe: gather writes zero
  // and scatter preserves the destination row. Producers remain responsible
  // for satisfying the documented range/uniqueness contract.
  const int32_t unsafe_indices[] = {2, -1};
  tensors.upload_bytes(indices, unsafe_indices, sizeof(unsafe_indices));
  tensors.upload(scattered, scatter_initial.data(), scatter_initial.size());
  TensorBatch safe_indexing = tensors.begin_batch();
  safe_indexing.gather_rows(matrix, indices, gathered);
  safe_indexing.scatter_rows(gathered, indices, scattered);
  safe_indexing.submit().wait();
  tensors.download(gathered, gathered_host.data(), gathered_host.size());
  tensors.download(scattered, scattered_host.data(), scattered_host.size());
  for (size_t col = 0; col < 4; ++col) {
    CHECK(gathered_host[col] == matrix_host[8 + col]);
    CHECK(float_bits(gathered_host[4 + col]) == 0u);
    CHECK(scattered_host[8 + col] == matrix_host[8 + col]);
    CHECK(scattered_host[col] == -99.0f);
    CHECK(scattered_host[4 + col] == -99.0f);
  }

  // Warm both bounded flight slots, then prove that repeated conversion and
  // layout batches reuse descriptor sets and pooled allocations exactly.
  for (int warm = 0; warm < 2; ++warm) {
    TensorBatch repeated = tensors.begin_batch();
    repeated.convert(input, bf16);
    repeated.convert(bf16, bf16_back);
    repeated.transpose_2d(input, transposed);
    repeated.add_bias(input, bias, biased);
    repeated.submit().wait();
  }
  const uint64_t warm_reserved = tensors.reserved_bytes();
  const uint64_t warm_descriptors = tensors.descriptor_set_allocations();
  for (int repeat = 0; repeat < 8; ++repeat) {
    TensorBatch repeated = tensors.begin_batch();
    repeated.convert(input, bf16);
    repeated.convert(bf16, bf16_back);
    repeated.transpose_2d(input, transposed);
    repeated.add_bias(input, bias, biased);
    repeated.submit().wait();
    CHECK(tensors.reserved_bytes() == warm_reserved);
    CHECK(tensors.descriptor_set_allocations() == warm_descriptors);
  }
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
  const Extent extents[] = {{2, 2}, {10, 6}, {128, 66}, {1280, 768}};
  uint64_t previous_high_water = 0;
  for (const Extent extent : extents) {
    const size_t pixels = static_cast<size_t>(extent.width) * extent.height;
    std::vector<float> r(pixels), g(pixels), b(pixels);
    for (size_t i = 0; i < pixels; ++i) {
      r[i] = static_cast<float>((i * 17) % 113) / 97.0f - 0.08f;
      g[i] = static_cast<float>((i * 29 + 3) % 127) / 109.0f;
      b[i] = static_cast<float>((i * 43 + 11) % 139) / 101.0f - 0.12f;
    }
    size_t boundary_block = 0;
    for (int by = 0; by < extent.height && boundary_block < 96; by += 2) {
      for (int bx = 0; bx < extent.width && boundary_block < 96; bx += 2, ++boundary_block) {
        const float boundary =
            (static_cast<float>(96 + (boundary_block % 64)) + 0.5f - 128.0f) / 112.0f;
        const int direction = static_cast<int>(boundary_block % 3) - 1;
        const float value = direction < 0 ? std::nextafter(boundary, -INFINITY)
                            : direction > 0 ? std::nextafter(boundary, INFINITY)
                                            : boundary;
        for (int dy = 0; dy < 2; ++dy) {
          for (int dx = 0; dx < 2; ++dx) {
            const size_t i = static_cast<size_t>(by + dy) * extent.width + bx + dx;
            r[i] = (boundary_block & 1) != 0 ? value : 0.0f;
            g[i] = 0.0f;
            b[i] = (boundary_block & 1) == 0 ? value : 0.0f;
          }
        }
      }
    }
    // Keep luma half-step fixtures at the opposite end of larger frames so
    // the chroma-boundary blocks above cannot overwrite them.
    if (pixels >= 512) {
      for (size_t fixture = 0; fixture < 192; ++fixture) {
        const size_t i = pixels - 192 + fixture;
        const float boundary =
            (static_cast<float>(16 + (fixture % 220)) + 0.5f - 16.0f) / 219.0f;
        const int direction = static_cast<int>(fixture % 3) - 1;
        const float value = direction < 0 ? std::nextafter(boundary, -INFINITY)
                            : direction > 0 ? std::nextafter(boundary, INFINITY)
                                            : boundary;
        r[i] = value;
        g[i] = value;
        b[i] = value;
      }
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
    CHECK_MSG(differences == 0,
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
  CHECK_MSG(converter.high_water_bytes() <= (48ull << 20),
            "Vulkan YUV persistent high-water is %llu bytes",
            static_cast<unsigned long long>(converter.high_water_bytes()));

  const uint64_t capacity_before_reject = converter.capacity_pixels();
  const uint64_t reserved_before_reject = converter.reserved_bytes();
  bool index_overflow_rejected = false;
  uint8_t reject_byte = 0;
  float reject_sample = 0.0f;
  try {
    converter.convert(&reject_sample, &reject_sample, &reject_sample, 32768, 65536,
                      &reject_byte, 65536, &reject_byte, 32768, &reject_byte, 32768);
  } catch (const std::overflow_error&) {
    index_overflow_rejected = true;
  }
  CHECK(index_overflow_rejected);
  CHECK(converter.capacity_pixels() == capacity_before_reject);
  CHECK(converter.reserved_bytes() == reserved_before_reject);

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

  const int frames = 3;
  const int y4m_width = 10;
  const int y4m_height = 6;
  const size_t frame_pixels = static_cast<size_t>(y4m_width) * y4m_height;
  PixelBuffer clip(static_cast<size_t>(3) * frames * frame_pixels);
  const size_t plane = static_cast<size_t>(frames) * frame_pixels;
  for (int frame = 0; frame < frames; ++frame) {
    for (size_t i = 0; i < frame_pixels; ++i) {
      const size_t offset = static_cast<size_t>(frame) * frame_pixels + i;
      clip[offset] = static_cast<float>((i * 11 + frame * 7) % 101) / 100.0f;
      clip[plane + offset] = static_cast<float>((i * 23 + frame * 5) % 103) / 102.0f;
      clip[2 * plane + offset] = static_cast<float>((i * 37 + frame * 3) % 107) / 106.0f;
    }
  }
  const auto cpu_path = std::filesystem::temp_directory_path() / "vidfab_vulkan_cpu.y4m";
  const auto vk_path = std::filesystem::temp_directory_path() / "vidfab_vulkan_output.y4m";
  std::filesystem::remove(cpu_path);
  std::filesystem::remove(vk_path);
  video::write_y4m(cpu_path.string(), clip, frames, y4m_height, y4m_width);
  const uint64_t before_y4m = converter.reserved_bytes();
  video::write_y4m(vk_path.string(), clip, frames, y4m_height, y4m_width, {}, &converter);
  CHECK(read_bytes(cpu_path) == read_bytes(vk_path));
  CHECK(converter.reserved_bytes() == before_y4m);
  std::filesystem::remove(cpu_path);
  std::filesystem::remove(vk_path);
}

}  // namespace

int main() { return ::vidfab::test::run_all(); }
