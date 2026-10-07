#include "harness.h"
#include "../src/seedvr2/runtime.cuh"
#include "slopfab/safetensors_write.h"
#include "slopfab/tensor_convert.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>

namespace {
using namespace slopfab;
using namespace slopfab::seedvr2;

bool have_cuda() {
  if (cuda::device_count() > 0)
    return true;
  SKIP_UNSUPPORTED_HARDWARE("CUDA required");
  return false;
}

struct Archive {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("slopfab_seedvr2_runtime_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
       ".safetensors");
  SafeTensors file;

  Archive(const char* dtype, size_t count, const std::vector<uint8_t>& bytes) {
    std::string header = "{\"weight\":{\"dtype\":\"" + std::string(dtype) + "\",\"shape\":[" +
                         std::to_string(count) + "],\"data_offsets\":[0," +
                         std::to_string(bytes.size()) + "]}}";
    while (header.size() % 8)
      header += ' ';
    std::ofstream out(path, std::ios::binary);
    const uint64_t size = header.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(header.data(), std::streamsize(header.size()));
    out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    out.close();
    if (!out)
      throw std::runtime_error("cannot write SeedVR2 test fixture");
    file.open(path.u8string());
  }

  explicit Archive(float scale) {
    write_safetensors(
        path.u8string(),
        {{"proj.weight", {4, 4}, {scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale, 0, 0, 0, 0, scale}},
         {"proj.bias", {4}, {0.25f, -0.5f, 0.75f, -1.0f}}});
    file.open(path.u8string());
  }

  ~Archive() {
    file.close();
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
  }
};

template <typename T> std::vector<uint8_t> bytes_of(const std::vector<T>& values) {
  std::vector<uint8_t> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

void check_weight(Runtime& runtime, SafeTensors& file) {
  runtime.clear(file);
  const auto expected = to_f32(file.at("weight"));
  const auto actual = runtime.download(runtime.weight("weight"));
  CHECK(actual.size() == expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    const float rounded = bf16_to_f32(f32_to_bf16(expected[i]));
    if (std::isnan(rounded)) {
      CHECK(std::isnan(actual[i]));
    } else {
      CHECK(actual[i] == rounded);
      CHECK(std::signbit(actual[i]) == std::signbit(rounded));
    }
  }
}

__global__ void write_values(uint32_t* values, size_t n, uint32_t value) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    values[i] = value;
}

__global__ void increment_values(uint32_t* values, size_t n) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    ++values[i];
}
} // namespace

SLOPFAB_TEST(seedvr2_runtime_fp8_all_encodings) {
  if (!have_cuda())
    return;
  std::vector<uint8_t> bytes(256);
  for (size_t i = 0; i < bytes.size(); ++i)
    bytes[i] = uint8_t(i);
  Archive fixture("F8_E4M3", bytes.size(), bytes);
  Runtime runtime;
  runtime.resident_budget = 0;
  check_weight(runtime, fixture.file);
  CHECK(runtime.resident.empty());
}

SLOPFAB_TEST(seedvr2_runtime_weight_conversion_dtypes) {
  if (!have_cuda())
    return;
  const std::vector<float> values{0.0f,
                                  -0.0f,
                                  1.0f,
                                  -1.0f,
                                  1.00390625f,
                                  1.01171875f,
                                  -1.00390625f,
                                  65504.0f,
                                  0x1p-24f,
                                  -0x1p-24f,
                                  1e-7f,
                                  std::numeric_limits<float>::infinity(),
                                  -std::numeric_limits<float>::infinity(),
                                  std::numeric_limits<float>::quiet_NaN()};
  std::vector<uint16_t> halves, bfloat;
  for (float value : values) {
    halves.push_back(f32_to_f16(value));
    bfloat.push_back(f32_to_bf16(value));
  }
  Archive f32("F32", values.size(), bytes_of(values));
  Archive f16("F16", values.size(), bytes_of(halves));
  Archive bf16("BF16", values.size(), bytes_of(bfloat));
  Runtime runtime;
  runtime.resident_budget = 0;
  check_weight(runtime, f32.file);
  check_weight(runtime, f16.file);
  check_weight(runtime, bf16.file);
}

SLOPFAB_TEST(seedvr2_runtime_pool_reuse_is_stream_ordered) {
  if (!have_cuda())
    return;
  auto pool = std::make_shared<cuda::ReferenceBufferPool>();
  constexpr size_t count = 1 << 18;
  uint32_t* pointer = nullptr;
  {
    Buffer<uint32_t> first(count, pool);
    pointer = first.get();
    write_values<<<int((count + 255) / 256), 256>>>(first.get(), count, 41);
    SLOPFAB_CUDA_CHECK(cudaGetLastError());
  }
  Buffer<uint32_t> second(count, pool);
  CHECK(second.get() == pointer);
  CHECK(pool->allocations() == 1);
  increment_values<<<int((count + 255) / 256), 256>>>(second.get(), count);
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
  std::vector<uint32_t> actual(count);
  second.copy_to_host(actual.data(), actual.size());
  CHECK(std::all_of(actual.begin(), actual.end(), [](uint32_t v) {
    return v == 42;
  }));
  Buffer<uint32_t> concurrent(count / 2, pool);
  CHECK(concurrent.get() != second.get());
  CHECK(pool->allocations() == 2);
  CHECK(pool->reserved_bytes() == (count + count / 2) * sizeof(uint32_t));
}

SLOPFAB_TEST(seedvr2_runtime_tensor_outlives_runtime_and_moves) {
  if (!have_cuda())
    return;
  Tensor survivor;
  std::weak_ptr<cuda::ReferenceBufferPool> pool;
  const std::vector<float> values{1, -2, 3.5f, 0};
  {
    Runtime runtime;
    runtime.activations = std::make_shared<cuda::ReferenceBufferPool>();
    pool = runtime.activations;
    survivor = runtime.upload(values, 1, 1, 1, int(values.size()));
  }
  CHECK(!pool.expired());
  std::vector<BFloat> actual(values.size());
  survivor.data.copy_to_host(actual.data(), actual.size());
  for (size_t i = 0; i < values.size(); ++i)
    CHECK(__bfloat162float(actual[i]) == values[i]);
  Tensor moved(std::move(survivor));
  CHECK(survivor.size() == 0);
  CHECK(moved.size() == values.size());
  Tensor target(1, 1, 1, 2, std::make_shared<cuda::ReferenceBufferPool>());
  target = std::move(moved); // release old lease before releasing its final pool owner
  CHECK(moved.size() == 0);
  CHECK(!pool.expired());
  target = Tensor();
  CHECK(pool.expired());
}

SLOPFAB_TEST(seedvr2_runtime_cache_separates_checkpoints) {
  if (!have_cuda())
    return;
  Archive first(1), second(2);
  Runtime runtime;
  runtime.resident_budget = 1024;
  runtime.clear(first.file);
  const auto* first_pointer = runtime.weight("proj.weight").data.get();
  const auto first_values = runtime.download(runtime.weight("proj.weight"));
  runtime.clear(second.file);
  const auto* second_pointer = runtime.weight("proj.weight").data.get();
  const auto second_values = runtime.download(runtime.weight("proj.weight"));
  CHECK(first_pointer != second_pointer);
  CHECK(first_values[0] == 1);
  CHECK(second_values[0] == 2);
  runtime.clear(first.file);
  CHECK(runtime.weight("proj.weight").data.get() == first_pointer);
  CHECK(runtime.resident.size() == 2);
  CHECK(runtime.resident_bytes == 2 * 16 * sizeof(BFloat));
}

SLOPFAB_TEST(seedvr2_runtime_cache_budget_preserves_results) {
  if (!have_cuda())
    return;
  Archive fixture(1.5f);
  std::vector<float> input(8 * 4);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = (float(i) - 16) / 8;
  Runtime streaming, cached;
  streaming.resident_budget = 0;
  cached.resident_budget = 16 * sizeof(BFloat); // weights fit exactly; bias must stream
  for (int segment = 0; segment < 3; ++segment) {
    streaming.clear(fixture.file);
    cached.clear(fixture.file);
    auto a = streaming.linear(streaming.upload(input, 1, 1, 8, 4), "proj");
    auto b = cached.linear(cached.upload(input, 1, 1, 8, 4), "proj");
    CHECK(streaming.download(a) == cached.download(b));
    CHECK(streaming.resident.empty());
    CHECK(cached.resident.size() == 1);
    CHECK(cached.weights.count("proj.bias") == 1);
    CHECK(cached.resident_bytes <= cached.resident_budget);
    streaming.end_segment();
    cached.end_segment();
    CHECK(streaming.weights.empty());
    CHECK(cached.weights.empty());
  }
}
