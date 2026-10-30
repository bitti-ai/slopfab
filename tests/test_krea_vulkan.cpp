#include "harness.h"
#include "../src/krea/vulkan_runtime.h"
#include "slopfab/safetensors_write.h"
#include <chrono>
#include <cmath>
#include <filesystem>

SLOPFAB_TEST_CATEGORY(krea_vulkan_primitives, "synthetic") {
  using namespace slopfab;
  using namespace krea_vulkan;
  if (!vk::Instance::available() || vk::Instance::create().enumerate_devices().empty()) {
    SKIP_UNSUPPORTED_HARDWARE("Vulkan unavailable");
    return;
  }
  const auto path = std::filesystem::temp_directory_path() /
                    ("slopfab-krea-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                     ".safetensors");

  struct Cleanup {
    std::filesystem::path path;

    ~Cleanup() {
      std::error_code ec;
      std::filesystem::remove(path, ec);
    }
  } cleanup{path};

  write_safetensors(path.string(), {{"scale", {128}, std::vector<float>(128, .125f)}});
  Runtime rt(path.string());
  std::vector<float> values(3 * 128);
  for (size_t i = 0; i < values.size(); ++i)
    values[i] = float(int(i % 17) - 8) / 8;
  auto x = from_host(values, 3, 128);
  auto actual = to_host(norm(rt, x, "scale", 128));
  for (int r = 0; r < 3; ++r) {
    double sum = 0;
    for (int c = 0; c < 128; ++c)
      sum += double(values[r * 128 + c]) * values[r * 128 + c];
    float inv = float(1 / std::sqrt(sum / 128 + 1e-5));
    for (int c = 0; c < 128; ++c)
      CHECK_NEAR(actual[r * 128 + c], round_bf(values[r * 128 + c] * inv * 1.125f), .008);
  }
  // With zero logits each token averages its own twelve layers. The second
  // token has a distant offset so a missing batch boundary cannot pass.
  auto q = from_host(std::vector<float>(24 * 2560, 0), 24, 2560);
  auto k = from_host(std::vector<float>(24 * 2560, 0), 24, 2560);
  std::vector<float> v(24 * 2560);
  for (int r = 0; r < 24; ++r)
    for (int c = 0; c < 2560; ++c)
      v[r * 2560 + c] = float(r % 12) + 100 * (r / 12);
  auto vt = from_host(v, 24, 2560);
  actual = to_host(gpu_layer_attention(q, k, vt));
  for (int r = 0; r < 24; ++r)
    for (int c = 0; c < 2560; ++c)
      CHECK(actual[r * 2560 + c] == 5.5f + 100 * (r / 12));

  std::vector<float> cosine(3 * 64, 0), sine(3 * 64, 1);
  ops::rope(x, 1, cosine, sine, true);
  actual = to_host(x);
  for (int i = 0; i < 3 * 128; i += 2) {
    CHECK(actual[i] == -values[i + 1]);
    CHECK(actual[i + 1] == values[i]);
  }
  // Grouped-query causal attention must exclude future values for both heads.
  q = from_host(std::vector<float>(5 * 256, 0), 5, 256);
  k = from_host(std::vector<float>(5 * 128, 0), 5, 128);
  v.resize(5 * 128);
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 128; ++c)
      v[r * 128 + c] = float(r);
  vt = from_host(v, 5, 128);
  actual = to_host(rt.attend(q, k, vt, 2, 1, true));
  for (int r = 0; r < 5; ++r)
    for (int c = 0; c < 256; ++c)
      CHECK_NEAR(actual[r * 256 + c], r * .5f, .016);
  actual = to_host(rt.attend(q, k, vt, 2, 1, false));
  for (float value : actual)
    CHECK_NEAR(value, 2, .016);
  // Nonuniform logits exercise the scale, softmax and query-to-KV head mapping.
  std::vector<float> qh(5 * 256), kh(5 * 128);
  for (size_t i = 0; i < qh.size(); ++i)
    qh[i] = round_bf(std::sin(float(i) * .17f));
  for (size_t i = 0; i < kh.size(); ++i) {
    kh[i] = round_bf(std::cos(float(i) * .11f));
    v[i] = round_bf(std::sin(float(i) * .07f));
  }
  q = from_host(qh, 5, 256);
  k = from_host(kh, 5, 128);
  vt = from_host(v, 5, 128);
  for (bool causal : {false, true}) {
    actual = to_host(rt.attend(q, k, vt, 2, 1, causal));
    for (int r = 0; r < 5; ++r)
      for (int head = 0; head < 2; ++head) {
        double probabilities[5] = {}, sum = 0;
        int count = causal ? r + 1 : 5;
        for (int j = 0; j < count; ++j) {
          double score = 0;
          for (int c = 0; c < 128; ++c)
            score += double(qh[r * 256 + head * 128 + c]) * kh[j * 128 + c];
          probabilities[j] = std::exp(score / std::sqrt(128.));
          sum += probabilities[j];
        }
        for (int c = 0; c < 128; ++c) {
          double expected = 0;
          for (int j = 0; j < count; ++j)
            expected += probabilities[j] / sum * v[j * 128 + c];
          CHECK_NEAR(actual[r * 256 + head * 128 + c], expected, .008);
        }
      }
  }
}
