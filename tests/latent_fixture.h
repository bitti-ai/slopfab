#pragma once
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Independent archive writer, also usable by tests linked only to the DLL.
struct LatentFixture {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("slopfab-latents-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".safetensors");
  ~LatentFixture() { std::error_code ec; std::filesystem::remove(path, ec); }
  void write(int frames = 39, bool sampled = true, const std::string& schema = "h3-av-v1") {
    const int f = (frames - 5) / 17 * 5 + 2;
    const int a = (frames * 5 + 1) / 3;
    std::vector<float> video(size_t(f) * 2 * 96), audio(size_t(a) * 64);
    for (size_t i = 0; i < video.size(); ++i) video[i] = float(i) / 64;
    for (size_t i = 0; i < audio.size(); ++i) audio[i] = float(i) / 32;
    const size_t vb = video.size() * 4, ab = audio.size() * 4;
    std::string header = "{\"__metadata__\":{\"slopfab_latents\":\"" + schema +
        "\",\"width\":\"64\",\"height\":\"32\",\"frames\":\"" + std::to_string(frames) +
        "\",\"fps\":\"24\",\"sampled\":\"" + (sampled ? "1" : "0") +
        "\"},\"video_rows\":{\"dtype\":\"F32\",\"shape\":[" + std::to_string(f * 2) +
        ",96],\"data_offsets\":[0," + std::to_string(vb) +
        "]},\"audio_rows\":{\"dtype\":\"F32\",\"shape\":[" + std::to_string(a * 2) +
        ",32],\"data_offsets\":[" + std::to_string(vb) + ',' + std::to_string(vb + ab) + "]}}";
    while (header.size() % 8) header += ' ';
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const uint64_t n = header.size();
    out.write(reinterpret_cast<const char*>(&n), 8);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    out.write(reinterpret_cast<const char*>(video.data()), static_cast<std::streamsize>(vb));
    out.write(reinterpret_cast<const char*>(audio.data()), static_cast<std::streamsize>(ab));
    if (!out) throw std::runtime_error("latent fixture write failed");
  }
};
