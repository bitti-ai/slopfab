#include "../src/syncnet/network.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace slopfab::syncnet;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void write(const std::filesystem::path& p, const std::vector<float>& v) {
  std::ofstream f(p, std::ios::binary);
  f.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size() * 4));
  require(bool(f), "write fixture");
}
int main(int argc, char** argv) {
  try {
    std::vector<std::vector<float>> v(80, std::vector<float>(1024));
    for (int i = 0; i < 80; ++i) for (int c = 0; c < 1024; ++c)
      v[i][c] = float(std::sin(i * 1.737 + c * .271));
    auto a = v;
    require(score(v, a, 10).offset_frames == 0, "aligned offset");
    for (int i = 0; i < 80; ++i) a[i] = v[std::max(0, i - 4)];
    require(score(v, a, 10).offset_frames == -4, "delayed audio must advance");
    require(score(a, v, 10).offset_frames == 4, "early audio must delay");
    require(score(v, a, 4).search_boundary, "search boundary");
    Image x(2, 2, 1); x.data = {1, 2, 3, 4};
    auto p = Network::pool(x, 2, 2, 2, 2);
    require(p.data == std::vector<float>{4}, "pool");
    auto m = mfcc(std::vector<int16_t>(640, 0));
    require(m.size() == 39 && std::abs(m[0] - std::log(2.220446049250313e-16)) < 1e-5,
            "silent MFCC log energy");
    auto bad = std::filesystem::temp_directory_path() / "slopfab-syncnet-invalid.model";
    { std::ofstream f(bad); f << "not pickle"; }
    bool rejected = false;
    try { load_weights(bad.u8string()); } catch (const std::exception& e) {
      rejected = std::string(e.what()).find("961e8696") != std::string::npos;
    }
    std::filesystem::remove(bad);
    require(rejected, "hash rejection");
    if (argc >= 2) {
      auto root = std::filesystem::u8path(argv[1]);
      Network net(load_weights((root / "syncnet_v2.model").u8string()), argc >= 3 ? argv[2] : "cpu");
      Image audio(13, 20, 1), lip(224, 224, 15);
      for (size_t i = 0; i < audio.data.size(); ++i) audio.data[i] = float(int(i % 37) - 18);
      for (size_t i = 0; i < lip.data.size(); ++i) lip.data[i] = float(i % 251);
      write(root / "native-audio.f32", net.embedding(std::move(audio), true));
      write(root / "native-lip.f32", net.embedding(std::move(lip), false));
      std::vector<int16_t> pcm(16327);
      for (size_t i = 0; i < pcm.size(); ++i) pcm[i] = int16_t(int(i * 7919 % 60001) - 30000);
      write(root / "native-mfcc.f32", mfcc(pcm));
      Network detector(load_weights((root / "sfd_face.pth").u8string(), true), net.backend());
      Image rgb(64, 96, 3);
      for (size_t i = 0; i < rgb.data.size(); ++i) rgb.data[i] = float(i % 251) - 117;
      auto first = detector.conv(rgb, "vgg.0", 1, 1, 1, true);
      write(root / "native-detector-first.f32", first.data);
      require(detect(detector, rgb).empty(), "noise should not produce faces");
      std::cout << "Reference outputs written (" << net.backend() << ")\n";
    }
    std::cout << "SyncNet tests passed\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
