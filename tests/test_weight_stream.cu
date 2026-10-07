#include "harness.h"
#include "../src/seedvr2/weight_stream.cuh"
#include "slopfab/cuda/device.h"
#include "slopfab/tensor_convert.h"
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>

namespace {
using namespace slopfab;
using namespace slopfab::seedvr2;
constexpr size_t kCount = 8193;

struct Environment {
  std::string old;
  bool present;

  Environment() : present(std::getenv("SLOPFAB_SEEDVR2_PREFETCH") != nullptr) {
    if (present)
      old = std::getenv("SLOPFAB_SEEDVR2_PREFETCH");
    set("1");
  }

  void set(const char* value) {
#ifdef _WIN32
    _putenv_s("SLOPFAB_SEEDVR2_PREFETCH", value ? value : "");
#else
    if (value)
      setenv("SLOPFAB_SEEDVR2_PREFETCH", value, 1);
    else
      unsetenv("SLOPFAB_SEEDVR2_PREFETCH");
#endif
  }

  ~Environment() {
    set(present ? old.c_str() : nullptr);
  }
};

struct Archive {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("slopfab_weight_stream_" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
       ".safetensors");
  SafeTensors file;

  Archive() {
    std::string header = "{";
    std::vector<uint8_t> data;
    const std::array<const char*, 4> types{"F16", "BF16", "F32", "F8_E4M3"};
    for (int block = 0; block < 32; ++block) {
      for (int dtype = 0; dtype < 4; ++dtype) {
        size_t start = data.size(), width = dtype == 2 ? 4 : dtype == 3 ? 1 : 2;
        data.resize(start + kCount * width);
        for (size_t i = 0; i < kCount; ++i) {
          float value = float(int(i % 100) - 50 + block) * .125f;
          if (i % 509 == 0)
            value = -0.0f;
          if (i % 509 == 1)
            value = std::numeric_limits<float>::infinity();
          if (i % 509 == 2)
            value = std::numeric_limits<float>::quiet_NaN();
          auto* out = data.data() + start + i * width;
          if (dtype == 2)
            std::memcpy(out, &value, 4);
          else if (dtype == 3)
            *out = uint8_t(i + block);
          else {
            uint16_t bits = dtype == 0 ? f32_to_f16(value) : f32_to_bf16(value);
            std::memcpy(out, &bits, 2);
          }
        }
        if (header.size() > 1)
          header += ',';
        header += "\"model.diffusion_model.blocks." + std::to_string(block) + "." + types[dtype] +
                  "\":{\"dtype\":\"" + types[dtype] + "\",\"shape\":[" + std::to_string(kCount) +
                  "],\"data_offsets\":[" + std::to_string(start) + "," +
                  std::to_string(data.size()) + "]}";
      }
    }
    header += '}';
    while (header.size() % 8)
      header += ' ';
    std::ofstream out(path, std::ios::binary);
    const uint64_t length = header.size();
    out.write(reinterpret_cast<const char*>(&length), sizeof(length));
    out.write(header.data(), std::streamsize(header.size()));
    out.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
    out.close();
    if (!out)
      throw std::runtime_error("cannot write weight stream fixture");
    file.open(path.u8string());
  }

  ~Archive() {
    file.close();
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
  }
};

void verify(WeightStream& stream, SafeTensors& file, int blocks) {
  cuda::DeviceBuffer<__nv_bfloat16> result(size_t(blocks) * 4 * kCount);
  const std::array<const char*, 4> types{"F16", "BF16", "F32", "F8_E4M3"};
  stream.begin();
  for (int block = 0; block < blocks; ++block) {
    stream.begin_block(block);
    for (int dtype = 0; dtype < 4; ++dtype) {
      auto view = stream.find("blocks." + std::to_string(block) + "." + types[dtype]);
      CHECK(view.count == kCount);
      CHECK(view.data != nullptr);
      if (!view.data)
        throw std::runtime_error("missing streamed weight");
      // No host synchronization here: the next block may be requested before
      // this read finishes. Slot reuse must wait for its default-stream event.
      SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(result.get() + (block * 4 + dtype) * kCount, view.data,
                                         kCount * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
    }
    CHECK(stream.find("missing").data == nullptr);
  }
  stream.finish();
  CHECK(stream.find("blocks.0.F16").data == nullptr);
  std::vector<__nv_bfloat16> actual(result.size());
  result.copy_to_host(actual.data(), actual.size());
  for (int block = 0; block < blocks; ++block)
    for (int dtype = 0; dtype < 4; ++dtype) {
      auto expected = to_f32(file.at("blocks." + std::to_string(block) + "." + types[dtype]));
      bool match = true;
      for (size_t i = 0; i < kCount; ++i) {
        float a = __bfloat162float(actual[(block * 4 + dtype) * kCount + i]);
        float b = bf16_to_f32(f32_to_bf16(expected[i]));
        match &= std::isnan(b) ? std::isnan(a) : a == b && std::signbit(a) == std::signbit(b);
      }
      CHECK(match);
    }
}
} // namespace

SLOPFAB_TEST(seedvr2_weight_stream_reuse_and_conversion) {
  if (cuda::device_count() == 0) {
    SKIP_UNSUPPORTED_HARDWARE("CUDA required");
    return;
  }
  Environment environment;
  Archive archive;
  int device = 0;
  SLOPFAB_CUDA_CHECK(cudaGetDevice(&device));
  constexpr size_t budget = size_t(64) << 20;
  CHECK(!WeightStream::create(archive.file, device, 0));
  environment.set("0");
  CHECK(!WeightStream::create(archive.file, device, budget));
  environment.set("1");
  auto stream = WeightStream::create(archive.file, device, budget);
  if (!stream) {
    SKIP_INSUFFICIENT_VRAM("Weight stream needs 17 MiB VRAM and 32 MiB pinned host memory");
    return;
  }
  CHECK(stream->device_bytes() <= budget);
  bool threw = false;
  try {
    stream->begin_block(0);
  } catch (const std::logic_error&) {
    threw = true;
  }
  CHECK(threw);
  verify(*stream, archive.file, 32);
  verify(*stream, archive.file, 3);
  verify(*stream, archive.file, 32);
  stream->begin();
  stream->begin_block(0);
  threw = false;
  try {
    stream->begin_block(2);
  } catch (const std::logic_error&) {
    threw = true;
  }
  CHECK(threw);
  // Joining with a prefetch in flight must also be safe without finish().
  stream.reset();
}
