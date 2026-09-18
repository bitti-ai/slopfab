#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "harness.h"
#include "slopfab/cuda/nf4_weight.cuh"
#include "slopfab/int8_weight.h"
#include "slopfab/safetensors_write.h"

namespace {

struct Fixture {
  std::string path = (std::filesystem::temp_directory_path() /
                      "slopfab_int8_weight_test.safetensors").string();
  std::vector<int8_t> codes;
  std::vector<float> scales;

  ~Fixture() { std::filesystem::remove(path); }

  void write(int columns, const std::string& tag, bool include_scale = true) {
    const int rows = 3;
    codes.resize(rows * columns);
    for (size_t i = 0; i < codes.size(); ++i)
      codes[i] = static_cast<int8_t>(static_cast<int>((i * 37 + i / columns * 17) % 255) - 127);
    if (scales.empty()) scales = {0.000017f, 0.0023f, 0.041f};
    const size_t scale_bytes = scales.size() * sizeof(float);
    std::string header = "{\"test.weight\":{\"dtype\":\"I8\",\"shape\":[3," +
        std::to_string(columns) + "],\"data_offsets\":[0," + std::to_string(codes.size()) + "]}";
    size_t offset = codes.size();
    if (include_scale) {
      header += ",\"test.weight_scale\":{\"dtype\":\"F32\",\"shape\":[" +
          std::to_string(scales.size()) + ",1],\"data_offsets\":[" +
          std::to_string(offset) + "," + std::to_string(offset + scale_bytes) + "]}";
      offset += scale_bytes;
    }
    if (!tag.empty()) {
      header += ",\"test.comfy_quant\":{\"dtype\":\"U8\",\"shape\":[" +
          std::to_string(tag.size()) + "],\"data_offsets\":[" +
          std::to_string(offset) + "," + std::to_string(offset + tag.size()) + "]}";
    }
    header += "}";
    while (header.size() % 8) header += ' ';
    const uint64_t length = header.size();
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(&length), sizeof(length));
    file.write(header.data(), header.size());
    file.write(reinterpret_cast<const char*>(codes.data()), codes.size());
    if (include_scale)
      file.write(reinterpret_cast<const char*>(scales.data()), scale_bytes);
    file.write(tag.data(), tag.size());
    if (!file) throw std::runtime_error("could not write INT8 test checkpoint");
  }
};

std::string metadata(int group) {
  return "{\"format\":\"int8_tensorwise\",\"convrot\":" +
      std::string(group == 1 ? "false" : "true") +
      ",\"convrot_groupsize\":" + std::to_string(group) + "}";
}

// Independent dense Kronecker-product oracle, not a butterfly implementation.
std::vector<uint16_t> reference(const Fixture& fixture, int columns, int group,
                                bool canonicalize) {
  std::vector<uint16_t> result(fixture.codes.size());
  for (size_t index = 0; index < result.size(); ++index) {
    const int column = static_cast<int>(index % columns);
    const int row = static_cast<int>(index / columns);
    const int base = column / group * group;
    int sum = 0;
    for (int k = 0; k < group; ++k) {
      int sign = 1;
      for (int stride = 1; stride < group; stride *= 4)
        if ((k / stride % 4) + (column % group / stride % 4) == 3) sign = -sign;
      sum += fixture.codes[row * columns + base + k] * sign;
    }
    uint16_t bits = slopfab::f32_to_f16(sum * (fixture.scales[row] / std::sqrt(float(group))));
    if (canonicalize && (bits & 0x7c00u) == 0) bits &= 0x8000u;
    result[index] = bits;
  }
  return result;
}

}  // namespace

SLOPFAB_TEST(int8_vae_weights_match_dense_hadamard) {
  Fixture fixture;
  slopfab::cuda::Stream stream;
  slopfab::cuda::F16Weight weight;
  // Multiple rows/groups, partial CUDA blocks, unrotated rows and the
  // quantizer's nondivisible-width fallback all exercise different indexing.
  for (const int group : {1, 4, 16, 64, 256}) {
    for (const int columns : {group * 2, 267}) {
      fixture.write(columns, metadata(group));
      slopfab::SafeTensors checkpoint;
      checkpoint.open(fixture.path);
      const int effective_group = columns % group == 0 ? group : 1;
      for (const bool canonicalize : {false, true}) {
        const auto expected = reference(fixture, columns, effective_group, canonicalize);
        const auto state = slopfab::read_int8_weight(checkpoint, "test.weight", "test");
        CHECK(state.rotation_group == effective_group);
        CHECK(slopfab::unpack_int8_weight(state, canonicalize) == expected);
        weight.load(checkpoint, "test.weight", expected.size(), stream.get(), "test", canonicalize);
        CHECK(weight.packed_int8());
        CHECK(weight.stored_bytes() == expected.size() + 3 * sizeof(float));
        slopfab::cuda::DeviceBuffer<__half> workspace(expected.size());
        const __half* data = weight.materialize(workspace.get(), workspace.size(), stream.get());
        std::vector<uint16_t> actual(expected.size());
        SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(actual.data(), data, actual.size() * 2,
                                          cudaMemcpyDeviceToHost, stream.get()));
        stream.synchronize();
        CHECK(actual == expected);
        bool rejected = false;
        try { weight.materialize(workspace.get(), expected.size() - 1, stream.get()); }
        catch (const std::runtime_error&) { rejected = true; }
        CHECK(rejected);
      }
    }
  }
  // Reloading a formerly packed weight must select its new dense storage.
  slopfab::write_safetensors(fixture.path, {{"test.weight", {1, 2}, {0.5f, -2.0f}, slopfab::DType::kF16}});
  slopfab::SafeTensors dense;
  dense.open(fixture.path);
  weight.load(dense, "test.weight", 2, stream.get(), "test");
  CHECK(!weight.packed_int8());
  uint16_t values[2];
  SLOPFAB_CUDA_CHECK(cudaMemcpyAsync(values, weight.materialize(nullptr, 0, stream.get()),
                                    sizeof(values), cudaMemcpyDeviceToHost, stream.get()));
  stream.synchronize();
  CHECK(values[0] == slopfab::f32_to_f16(0.5f));
  CHECK(values[1] == slopfab::f32_to_f16(-2.0f));
}

SLOPFAB_TEST(int8_vae_weights_reject_incomplete_quantization) {
  Fixture fixture;
  slopfab::cuda::Stream stream;
  for (int failure = 0; failure < 7; ++failure) {
    std::string tag = metadata(256);
    fixture.scales = {0.01f, 0.02f, 0.03f};
    if (failure == 0) tag.clear();
    if (failure == 1) tag = "{}";
    if (failure == 2) tag = metadata(128);
    if (failure == 4) fixture.scales.resize(2);
    if (failure == 5) fixture.scales[1] = std::numeric_limits<float>::infinity();
    if (failure == 6) fixture.scales[1] = -0.01f;
    fixture.write(512, tag, failure != 3);
    slopfab::SafeTensors checkpoint;
    checkpoint.open(fixture.path);
    bool rejected = false;
    try {
      slopfab::cuda::F16Weight weight;
      weight.load(checkpoint, "test.weight", fixture.codes.size(), stream.get(), "test");
    } catch (const std::runtime_error&) { rejected = true; }
    CHECK(rejected);
  }
}
