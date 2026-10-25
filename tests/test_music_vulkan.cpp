#include "harness.h"
#define SLOPFAB_MUSIC_VULKAN 1
#include "../src/music/runtime.cuh"
#include "slopfab/safetensors_write.h"
#include <chrono>
#include <cmath>
#include <filesystem>

SLOPFAB_TEST_CATEGORY(vulkan_music_attention_and_yue_norm, "synthetic") {
  using namespace slopfab;
  namespace gpu = music_gpu;
  if (!vulkan::Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("Vulkan unavailable");
    return;
  }
  auto devices = vulkan::Instance::create().enumerate_devices();
  if (devices.empty() || !devices[0].info().storage_buffer_16bit ||
      !devices[0].info().shader_int64 || !devices[0].info().timeline_semaphore) {
    SKIP_UNSUPPORTED_HARDWARE("Music Vulkan prerequisites unavailable");
    return;
  }
  auto path = std::filesystem::temp_directory_path() /
              ("slopfab-music-" +
               std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
               ".safetensors");

  struct Cleanup {
    std::filesystem::path path;

    ~Cleanup() {
      std::error_code error;
      std::filesystem::remove(path, error);
    }
  } cleanup{path};

  write_safetensors(path.string(), {{"norm", {128}, std::vector<float>(128, 1)}});
  gpu::Runtime rt(path.string());
  std::vector<float> values(3 * 128);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = float(int(i % 13) - 6) / 8;
  auto x = gpu::from_host(values, 3, 128);
  auto actual = gpu::to_host(music_vulkan::gpu_yue_norm(rt, x, "norm", 128));
  for (int r = 0; r < 3; ++r) {
    double sum = 0;
    for (int c = 0; c < 128; ++c)
      sum += double(values[r * 128 + c]) * values[r * 128 + c];
    float inv = gpu::round_bf(float(1 / std::sqrt(sum / 128 + 1e-6)));
    for (int c = 0; c < 128; ++c)
      CHECK(actual[r * 128 + c] == gpu::round_bf(values[r * 128 + c] * inv));
  }
  // GQA with unequal query/key lengths and 256-wide heads exercises both
  // cache reads and the MiniMax local decoder's larger head dimension.
  const int dim = 256, heads = 2, qr = 3, kr = 5;
  std::vector<float> q(qr * heads * dim), k(kr * dim), v(kr * dim);
  for (size_t i = 0; i < q.size(); ++i)
    q[i] = float(int(i % 17) - 8) / 16;
  for (size_t i = 0; i < k.size(); ++i) {
    k[i] = float(int(i % 11) - 5) / 16;
    v[i] = float(int(i % 23) - 11) / 16;
  }
  auto qt = gpu::from_host(q, qr, heads * dim), kt = gpu::from_host(k, kr, dim),
       vt = gpu::from_host(v, kr, dim);
  // Reserve a larger KV allocation than the live prefix, as production AR does.
  gpu::Tensor cache(9, dim);
  music_vulkan::gpu_copy_rows(kt, cache, 0, 0, kr);
  cache.rows = kr;
  for (bool fp32 : {false, true}) {
    auto out = gpu::to_host(rt.attend(qt, cache, vt, heads, 1, false, fp32));
    for (int r = 0; r < qr; ++r)
      for (int h = 0; h < heads; ++h) {
        double probabilities[kr], sum = 0;
        for (int j = 0; j < kr; ++j) {
          double score = 0;
          for (int c = 0; c < dim; ++c)
            score += double(q[(r * heads + h) * dim + c]) * k[j * dim + c];
          sum += (probabilities[j] = std::exp(score / std::sqrt(double(dim))));
        }
        for (auto& p : probabilities)
          p = fp32 ? float(p / sum) : gpu::round_bf(float(p / sum));
        for (int c = 0; c < dim; ++c) {
          double expected = 0;
          for (int j = 0; j < kr; ++j)
            expected += probabilities[j] * v[j * dim + c];
          CHECK(std::abs(out[(r * heads + h) * dim + c] - gpu::round_bf(float(expected))) <= .002f);
        }
      }
  }
}
