#pragma once
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "slopfab/dtype.h"

// Real safetensors container with embedded metadata; usable by DLL-only tests.
struct RefModFixture {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("slopfab-refmod-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
       ".safetensors");

  ~RefModFixture() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
    auto sidecar = path;
    sidecar.replace_extension(".json");
    std::filesystem::remove(sidecar, ec);
  }

  void write(const std::string& meta =
                 R"({"kind":"image","name":"test","latent_h":4,"latent_w":4,"latent_t":1})",
             std::vector<int64_t> shape = {1, 24, 1, 4, 4}, std::vector<float> values = {},
             slopfab::DType dtype = slopfab::DType::kF32) {
    size_t n = 1;
    std::string dims;
    for (auto d : shape) {
      n *= static_cast<size_t>(d);
      if (!dims.empty())
        dims += ',';
      dims += std::to_string(d);
    }
    if (values.empty()) {
      values.resize(n);
      for (size_t i = 0; i < n; ++i)
        values[i] = float(i % 97) / 16;
    }
    const bool fp32 = dtype == slopfab::DType::kF32;
    const char* type = fp32 ? "F32" : dtype == slopfab::DType::kF16 ? "F16" : "BF16";
    std::string header = "{";
    if (!meta.empty()) {
      std::string escaped;
      for (char c : meta) {
        if (c == '"' || c == '\\')
          escaped += '\\';
        escaped += c;
      }
      header += "\"__metadata__\":{\"refmod_meta\":\"" + escaped + "\"},";
    }
    header += "\"latent\":{\"dtype\":\"" + std::string(type) + "\",\"shape\":[" + dims +
              "],\"data_offsets\":[0," + std::to_string(values.size() * (fp32 ? 4 : 2)) + "]}}";
    while (header.size() % 8)
      header += ' ';
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const uint64_t length = header.size();
    out.write(reinterpret_cast<const char*>(&length), 8);
    out.write(header.data(), header.size());
    for (float x : values) {
      if (fp32)
        out.write(reinterpret_cast<const char*>(&x), 4);
      else {
        const uint16_t bits =
            dtype == slopfab::DType::kF16 ? slopfab::f32_to_f16(x) : slopfab::f32_to_bf16(x);
        out.write(reinterpret_cast<const char*>(&bits), 2);
      }
    }
    if (!out)
      throw std::runtime_error("cannot write refmod fixture");
  }
};
