#pragma once
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Independent archive fixture for tests that link only the public DLL.
struct LcsFixture {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("slopfab-lcs-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
       ".safetensors");

  ~LcsFixture() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }

  void write() const {
    std::vector<float> data;
    std::string header =
        "{\"__metadata__\":{\"slopfab_lcs\":\"1\",\"vae_sha256\":\"" + std::string(64, 'a') + "\"}";
    const auto add = [&](const char* name, const char* shape, const std::vector<float>& values) {
      const size_t begin = data.size() * 4;
      data.insert(data.end(), values.begin(), values.end());
      header += ",\"" + std::string(name) + "\":{\"dtype\":\"F32\",\"shape\":" + shape +
                ",\"data_offsets\":[" + std::to_string(begin) + "," +
                std::to_string(data.size() * 4) + "]}";
    };
    std::vector<float> basis(288, 0);
    basis[0] = basis[4] = basis[8] = 1;
    add("basis", "[96,3]", basis);
    add("mean", "[96]", std::vector<float>(96, 0));
    add("anchor_lcs", "[8,3]",
        {1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1});
    add("latent_mean", "[24]", std::vector<float>(24, 0));
    add("latent_std", "[24]", std::vector<float>(24, 1));
    add("explained_variance", "[1]", {1});
    header += '}';
    while (header.size() % 8)
      header += ' ';
    const uint64_t n = header.size();
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&n), 8);
    out.write(header.data(), header.size());
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * 4);
  }
};
