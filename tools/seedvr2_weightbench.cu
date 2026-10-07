// Synthetic FP8 loader benchmark. Archive creation/checks/downloads are untimed.
#include "../src/seedvr2/runtime.cuh"
#include "slopfab/tensor_convert.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

using namespace slopfab;
using namespace slopfab::seedvr2;

namespace {
__global__ void convert_original(const float* input, BFloat* output, size_t count) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count)
    output[i] = __float2bfloat16(input[i]);
}

// Preserve the old loader's redundant initial BF16 allocation and upload
// temporary lifetimes; using today's pooled Runtime::upload would alter it.
Tensor original_upload(const std::vector<float>& input) {
  Tensor output(1, 1, 1, int(input.size()));
  cuda::DeviceBuffer<float> staging(input.size());
  staging.copy_from_host(input.data(), input.size());
  convert_original<<<int((input.size() + 255) / 256), 256>>>(staging.get(), output.data.get(),
                                                            input.size());
  SLOPFAB_CUDA_CHECK(cudaGetLastError());
  return output;
}

struct Archive {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("slopfab_seedvr2_fp8bench_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".safetensors");
  SafeTensors file;
  explicit Archive(size_t count) {
    std::string header = "{\"weight\":{\"dtype\":\"F8_E4M3\",\"shape\":[" +
                         std::to_string(count) + "],\"data_offsets\":[0," +
                         std::to_string(count) + "]}}";
    while (header.size() % 8)
      header += ' ';
    std::vector<uint8_t> bytes(count);
    for (size_t i = 0; i < count; ++i) {
      uint8_t value = uint8_t(i);
      if ((value & 127) == 127)
        --value; // Both FP8 NaN encodings become finite maximum magnitudes.
      bytes[i] = value;
    }
    std::ofstream out(path, std::ios::binary);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    const uint64_t length = header.size();
    out.write(reinterpret_cast<const char*>(&length), sizeof(length));
    out.write(header.data(), header.size());
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    out.close();
    file.open(path.u8string());
  }
  ~Archive() {
    file.close();
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
  }
};

void verify(const Tensor& tensor, const std::vector<BFloat>& expected) {
  std::vector<BFloat> actual(tensor.size());
  tensor.data.copy_to_host(actual.data(), actual.size());
  if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(BFloat)))
    throw std::runtime_error("FP8 conversion changed BF16 bits");
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}
}

int main() {
  try {
    cuda::set_device(0);
    constexpr size_t count = size_t(20) << 20;
    Archive archive(count);
    const auto& view = archive.file.at("weight");
    std::vector<BFloat> expected(count);
    {
      auto values = to_f32(view);
      for (size_t i = 0; i < count; ++i)
        expected[i] = __float2bfloat16(values[i]);
    }
    std::vector<double> old_times, new_times;
    for (int repeat = 0; repeat < 6; ++repeat) {
      const auto start = std::chrono::steady_clock::now();
      Tensor output(1, 1, 1, int(count));
      output = original_upload(to_f32(view));
      SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start).count();
      if (repeat)
        old_times.push_back(ms);
      verify(output, expected);
    }
    Runtime runtime;
    runtime.resident_budget = 0;
    for (int repeat = 0; repeat < 6; ++repeat) {
      runtime.clear(archive.file);
      SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
      const auto start = std::chrono::steady_clock::now();
      const auto& output = runtime.weight("weight");
      SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
      const double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start).count();
      if (repeat)
        new_times.push_back(ms);
      verify(output, expected);
    }
    std::printf("FP8 elements=%zu old_median_ms=%.4f new_median_ms=%.4f outputs_bit_identical=yes\n",
                count, median(old_times), median(new_times));
    std::printf("Logical loader allocation peaks (exclude context/diagnostics): "
                "old_device_mib=%.1f new_device_mib=%.1f; new_temporary_pool_peak_mib=%.1f "
                "reserved_mib=%.1f allocations=%zu; host_float_vector_avoided_mib=%.1f; "
                "H2D_old_mib=%.1f H2D_new_mib=%.1f\n",
                count * 8.0 / 1048576, count * 3.0 / 1048576,
                runtime.temporaries->peak_bytes() / 1048576.0,
                runtime.temporaries->reserved_bytes() / 1048576.0,
                runtime.temporaries->allocations(), count * 4.0 / 1048576,
                count * 4.0 / 1048576, count / 1048576.0);
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
