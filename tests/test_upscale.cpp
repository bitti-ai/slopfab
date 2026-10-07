#include "harness.h"
#include "slopfab/upscale.h"
#include "slopfab/tensor_convert.h"
#include "slopfab/safetensors_write.h"
#include "../src/upscale/backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>

#if SLOPFAB_WITH_CUDA
#include "slopfab/cuda/device.h"
#endif
#if SLOPFAB_WITH_VULKAN
#include "slopfab/vulkan/runtime.h"
#endif

namespace {
using namespace slopfab;
using namespace slopfab::upscale_detail;

std::vector<DeviceBackend> backends() {
  std::vector<DeviceBackend> result;
#if SLOPFAB_WITH_CUDA
  int devices = 0;
  if (cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0)
    result.push_back(DeviceBackend::kCuda);
  else
    SKIP_UNSUPPORTED_HARDWARE("no CUDA device");
#endif
#if SLOPFAB_WITH_VULKAN
  if (vulkan::Instance::available() && !vulkan::Instance::create().enumerate_devices().empty())
    result.push_back(DeviceBackend::kVulkan);
  else
    SKIP_UNSUPPORTED_HARDWARE("no Vulkan compute device");
#endif
  return result;
}

template <class Fn> bool rejected(Fn fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

struct Fixture {
  std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("slopfab-upscale-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
       ".safetensors");

  ~Fixture() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};
}

SLOPFAB_TEST(upscale_validation) {
  CHECK(parse_upscale_method("seedvr2") == UpscaleMethod::kSeedVr2);
  UpscaleOptions seed;
  seed.width = 128; seed.height = 64; seed.segment_frames = 5;
  CHECK(upscale_output_elements(15, 32, 32, UpscaleMethod::kSeedVr2, seed) == 15 * 128 * 64 * 3);
  CHECK(rejected([&] { upscale_output_elements(15, 32, 32, UpscaleMethod::kRealEsrgan, seed); }));
  seed.segment_frames = 6;
  CHECK(rejected([&] { validate_upscale_options(seed, UpscaleMethod::kSeedVr2); }));
  CHECK(rejected([&] { make_upscaler(UpscaleMethod::kSeedVr2, "missing", DeviceBackend::kVulkan); }));
#if SLOPFAB_WITH_CUDA
  auto restorer = make_upscaler(UpscaleMethod::kSeedVr2, "missing", DeviceBackend::kCuda);
  bool cancelled = false;
  try { restorer->upscale(PixelBuffer(16 * 16 * 3), 1, 16, 16, {}, [](int, int) { return false; }); }
  catch (const UpscaleCancelled&) { cancelled = true; }
  CHECK(cancelled); // Cancellation before loading weights works through the common factory.
#endif
  CHECK(parse_upscale_method("realesrgan") == UpscaleMethod::kRealEsrgan);
  CHECK(std::string(upscale_method_name(UpscaleMethod::kRealEsrgan)) == "realesrgan");
  CHECK(upscale_scale_factor(UpscaleMethod::kRealEsrgan) == 4);
  CHECK(rejected([] {
    parse_upscale_method("unknown");
  }));
  CHECK(rejected([] {
    upscale_scale_factor(static_cast<UpscaleMethod>(999));
  }));
  CHECK(rejected([] {
    make_upscaler(static_cast<UpscaleMethod>(999), "missing.safetensors", DeviceBackend::kCuda);
  }));
  CHECK(upscale_output_elements(2, 3, 5) == 1440);
  CHECK(rejected([] {
    upscale_output_elements(1, 0, 5);
  }));
  CHECK(rejected([] {
    upscale_output_elements(INT32_MAX, INT32_MAX, INT32_MAX);
  }));
  CHECK(rejected([] {
    validate_upscale_options({-1, 10, 10});
  }));
  CHECK(rejected([] {
    validate_upscale_options({128, 257, 10});
  }));
  CHECK(rejected([] {
    validate_upscale_options({128, 10, -1});
  }));
  SafeTensors empty;
  CHECK(rejected([&] {
    validate_realesrgan_checkpoint(empty);
  }));
}

SLOPFAB_TEST(upscale_gpu_operators) {
  for (auto device : backends()) {
    auto backend = device == DeviceBackend::kCuda
#if SLOPFAB_WITH_CUDA
                       ? make_cuda_backend()
#else
                       ? std::unique_ptr<Backend>{}
#endif
#if SLOPFAB_WITH_VULKAN
                       : make_vulkan_backend();
#else
                       : std::unique_ptr<Backend>{};
#endif
    const int h = 3, w = 5, ci = 3, co = 5;
    const auto input = test::make_data(h * w * ci, 13, .5f);
    const auto weight = test::make_data(co * ci * 9, 17, .2f);
    const auto bias = test::make_data(co, 23, .1f);
    std::vector<float> expected(h * w * co);
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x)
        for (int oc = 0; oc < co; ++oc) {
          float v = 0;
          for (int ic = 0; ic < ci; ++ic)
            for (int ky = 0; ky < 3; ++ky)
              for (int kx = 0; kx < 3; ++kx) {
                const int iy = y + ky - 1, ix = x + kx - 1;
                if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                  v += input[(iy * w + ix) * ci + ic] * weight[((oc * ci + ic) * 3 + ky) * 3 + kx];
              }
          v += bias[oc];
          expected[(y * w + x) * co + oc] = v < 0 ? v * .2f : v;
        }
    auto a = backend->upload(input), b = backend->upload(weight), c = backend->upload(bias);
    auto out = backend->allocate(expected.size());
    backend->run({kConv, h, w, ci, co, uint32_t(expected.size()), 1, 0}, a, b, c, out);
    CHECK_CLOSE(expected, backend->download(out, expected.size()), 2e-6, "3x3 convolution + leaky");
    auto concatenated = backend->allocate(h * w * (ci + co));
    backend->run({kConcat, h, w, ci, ci + co, h * w * (ci + co), 0, 0}, a, out, out, concatenated);
    auto joined = backend->download(concatenated, h * w * (ci + co));
    for (int p = 0; p < h * w; ++p) {
      for (int i = 0; i < ci; ++i)
        CHECK_NEAR(joined[p * (ci + co) + i], input[p * ci + i], 0);
      for (int i = 0; i < co; ++i)
        CHECK_NEAR(joined[p * (ci + co) + ci + i], expected[p * co + i], 2e-6);
    }
    auto residual = backend->allocate(input.size());
    backend->run({kResidual, h, w, ci, ci, uint32_t(input.size()), 0, .2f}, a, a, a, residual);
    auto values = backend->download(residual, input.size());
    for (size_t i = 0; i < input.size(); ++i)
      CHECK_NEAR(values[i], input[i] * 1.2f, 1e-7);
  }
}

SLOPFAB_TEST(upscale_cuda_chunk_boundary_and_buffer_lifetime) {
#if SLOPFAB_WITH_CUDA
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    SKIP_UNSUPPORTED_HARDWARE("no CUDA device");
    return;
  }
  // 34,427 pixels: the 32,768-pixel chunk boundary splits an odd-width row,
  // and the final chunk is partial. Three input / five output channels also
  // catch incorrect output offsets that accidentally use the input stride.
  constexpr int h = 199, w = 173, ci = 3, co = 5;
  static_assert(h * w > 32768 && 32768 % w != 0);
  const auto input = test::make_data(size_t(h) * w * ci, 47, 0.5f);
  const auto weight = test::make_data(size_t(co) * ci * 9, 53, 0.2f);
  const auto bias = test::make_data(co, 59, 0.1f);
  std::vector<float> expected(size_t(h) * w * co);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x)
      for (int oc = 0; oc < co; ++oc) {
        float value = 0;
        for (int ic = 0; ic < ci; ++ic)
          for (int ky = 0; ky < 3; ++ky)
            for (int kx = 0; kx < 3; ++kx) {
              const int sy = y + ky - 1, sx = x + kx - 1;
              if (sy >= 0 && sx >= 0 && sy < h && sx < w)
                value += input[(sy * w + sx) * ci + ic] *
                         weight[((oc * ci + ic) * 3 + ky) * 3 + kx];
            }
        value += bias[oc];
        expected[(y * w + x) * co + oc] = value < 0 ? value * 0.2f : value;
      }
  auto backend = make_cuda_backend();
  auto a = backend->upload(input), b = backend->upload(weight), c = backend->upload(bias);
  auto intermediate = backend->allocate(input.size());
  const Parameters residual{kResidual, h, w, ci, ci, uint32_t(input.size()), 0, 0};
  backend->run(residual, a, a, a, intermediate);
  auto output = backend->allocate(expected.size());
  backend->run({kConv, h, w, ci, co, uint32_t(expected.size()), 1, 0}, intermediate, b, c, output);
  // Return a just-consumed activation before synchronizing. Reusing its pool
  // slot must stay ordered after the convolution that still reads it.
  intermediate.reset();
  auto reused = backend->allocate(input.size());
  auto overwrite = residual;
  overwrite.scale = 1;
  backend->run(overwrite, a, a, a, reused);
  // Outputs must own their pool leases even after the producing backend dies.
  backend.reset();
  auto reader = make_cuda_backend();
  CHECK_CLOSE(expected, reader->download(output, expected.size()), 2e-6,
              "chunked convolution preserves every pixel and survives backend destruction");
  auto doubled = input;
  for (float& value : doubled)
    value *= 2;
  CHECK_CLOSE(doubled, reader->download(reused, input.size()), 0,
              "queued activation reuse preserves stream ordering");
#else
  SKIP_UNSUPPORTED_HARDWARE("CUDA backend not compiled");
#endif
}

SLOPFAB_TEST(upscale_full_graph_tiling_frames_and_cancellation) {
  Fixture fixture;
  std::vector<TensorWrite> tensors;
  for (const auto& spec : model_convolutions()) {
    std::vector<float> weight(size_t(spec.output) * spec.input * 9);
    if (spec.name == "conv_first" || spec.name == "conv_up1" || spec.name == "conv_up2" ||
        spec.name == "conv_hr" || spec.name == "conv_last")
      for (int c = 0; c < 3; ++c)
        weight[(size_t(c) * spec.input + c) * 9 + 4] = 1;
    tensors.push_back({spec.name + ".weight", {spec.output, spec.input, 3, 3}, std::move(weight)});
    tensors.push_back({spec.name + ".bias", {spec.output}, std::vector<float>(spec.output)});
  }
  write_safetensors(fixture.path.u8string(), tensors);
  tensors.clear();
  PixelBuffer input(3 * 2 * 3 * 5);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = float(i + 1) / float(input.size() + 1);
  for (auto backend : backends()) {
    auto model = make_upscaler(UpscaleMethod::kRealEsrgan, fixture.path.u8string(), backend);
    for (int tile : {0, 3}) {
      int done = -1, total = 0;
      auto output = model->upscale(input, 2, 3, 5, {tile, 1, 2}, [&](int d, int t) {
        CHECK(d >= done);
        done = d;
        total = t;
        return true;
      });
      CHECK(done == total && total > 0);
      CHECK(output.size() == input.size() * 16);
      for (int c = 0; c < 3; ++c)
        for (int f = 0; f < 2; ++f)
          for (int y = 0; y < 12; ++y)
            for (int x = 0; x < 20; ++x)
              CHECK_NEAR(output[((c * 2 + f) * 12 + y) * 20 + x],
                         input[((c * 2 + f) * 3 + y / 4) * 5 + x / 4], 1e-6);
    }
    bool cancelled = false;
    try {
      model->upscale(input, 2, 3, 5, {}, [](int, int) {
        return false;
      });
    } catch (const UpscaleCancelled&) {
      cancelled = true;
    }
    CHECK(cancelled);
    cancelled = false;
    int last = -1;
    try {
      model->upscale(input, 2, 3, 5, {3, 1, 2}, [&](int done, int) {
        last = done;
        return done < 1;
      });
    } catch (const UpscaleCancelled&) {
      cancelled = true;
    }
    CHECK(cancelled && last == 1);
    CHECK(rejected([&] {
      model->upscale(input, 1, 3, 5);
    }));
    auto bad = input;
    bad[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK(rejected([&] {
      model->upscale(bad, 2, 3, 5);
    }));
  }
}

SLOPFAB_TEST_CATEGORY(upscale_real_checkpoint_reference, "integration") {
  const char* model_path = std::getenv("SLOPFAB_REALESRGAN_MODEL");
  const char* reference_path = std::getenv("SLOPFAB_REALESRGAN_REFERENCE");
  if (!model_path || !reference_path) {
    SKIP_MISSING_FIXTURE("set SLOPFAB_REALESRGAN_MODEL and SLOPFAB_REALESRGAN_REFERENCE");
    return;
  }
  SafeTensors reference;
  reference.open(reference_path);
  const auto& view = reference.at("input");
  CHECK(view.shape.size() == 4 && view.shape[0] == 3);
  const int frames = int(view.shape[1]), h = int(view.shape[2]), w = int(view.shape[3]);
  const auto values = to_f32(view);
  const PixelBuffer input(values.begin(), values.end());
  for (auto backend : backends()) {
    auto model = make_upscaler(UpscaleMethod::kRealEsrgan, model_path, backend);
    for (int tile : {0, 7}) {
      const auto actual = model->upscale(input, frames, h, w, {tile, 3, 3});
      const auto expected = to_f32(reference.at(tile ? "tiled" : "full"));
      CHECK_CLOSE(expected, std::vector<float>(actual.begin(), actual.end()), 2e-4,
                  "PyTorch RRDBNet reference");
    }
  }
}

SLOPFAB_TEST_CATEGORY(upscale_real_checkpoint_frame_independence, "integration") {
  const char* model_path = std::getenv("SLOPFAB_REALESRGAN_MODEL");
  if (!model_path || !std::filesystem::is_regular_file(std::filesystem::u8path(model_path))) {
    SKIP_MISSING_FIXTURE("Set SLOPFAB_REALESRGAN_MODEL to the real x4plus checkpoint");
    return;
  }
  constexpr int height = 23, width = 31;
  constexpr size_t plane = size_t(height) * width, output_plane = plane * 16;
  PixelBuffer a(plane * 3), b(plane * 3);
  for (int c = 0; c < 3; ++c)
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width; ++x) {
        const size_t i = (size_t(c) * height + y) * width + x;
        a[i] = float((x * (c + 3) + y * (7 - c) + c * 31) % 251) / 250.0f;
        b[i] = 0.85f - 0.7f * a[i];
      }
  // [channel,frame,y,x]: A/B/A catches both frame-indexing mistakes and
  // unintended temporal state. Distinct RGB planes catch planar-layout bugs.
  PixelBuffer video(plane * 3 * 3);
  for (size_t c = 0; c < 3; ++c)
    for (size_t f = 0; f < 3; ++f)
      std::copy_n((f == 1 ? b : a).data() + c * plane, plane,
                  video.data() + (c * 3 + f) * plane);
  constexpr int other_height = 35, other_width = 41;
  PixelBuffer other(size_t(other_height) * other_width * 3);
  for (size_t i = 0; i < other.size(); ++i)
    other[i] = float((i * 43 + 19) % 251) / 250.0f;

  for (auto backend : backends()) {
    const char* name = backend == DeviceBackend::kCuda ? "CUDA" : "Vulkan";
    auto same_bits = [name](const PixelBuffer& expected, const PixelBuffer& actual,
                            const char* context) {
      CHECK_MSG(expected.size() == actual.size(), "%s %s: output sizes differ", name, context);
      if (expected.size() == actual.size())
        CHECK_MSG(std::memcmp(expected.data(), actual.data(), expected.size() * sizeof(float)) == 0,
                  "%s %s: identical frames/settings changed output bits", name, context);
    };
    auto model = make_upscaler(UpscaleMethod::kRealEsrgan, model_path, backend);
    PixelBuffer last_reference;
    UpscaleOptions last_options;
    for (int tile : {0, 17}) {
      const UpscaleOptions options{tile, 2, 3};
      const auto expected_a = model->upscale(a, 1, height, width, options);
      const auto expected_b = model->upscale(b, 1, height, width, options);
      CHECK_MSG(expected_a != expected_b, "%s fixture must restore distinct A and B frames", name);

      auto ignored_options = options;
      ignored_options.seed = std::numeric_limits<uint64_t>::max();
      ignored_options.segment_frames = 129;
      ignored_options.color_match = false;
      const auto repeated = model->upscale(a, 1, height, width, ignored_options);
      same_bits(expected_a, repeated, "A after B with changed SeedVR2-only options");

      // Change both geometry and tiling between calls so the activation pool
      // grows/reuses different shapes before the original configuration returns.
      const UpscaleOptions other_options{tile ? 0 : 13, 3, 1};
      const auto resized = model->upscale(other, 1, other_height, other_width, other_options);
      CHECK(resized.size() == other.size() * 16);
      const auto actual_video = model->upscale(video, 3, height, width, ignored_options);
      PixelBuffer expected_video(output_plane * 3 * 3);
      for (size_t c = 0; c < 3; ++c)
        for (size_t f = 0; f < 3; ++f)
          std::copy_n((f == 1 ? expected_b : expected_a).data() + c * output_plane,
                      output_plane, expected_video.data() + (c * 3 + f) * output_plane);
      same_bits(expected_video, actual_video, "A/B/A video versus separate calls after shape reuse");
      last_reference = expected_a;
      last_options = options;
    }
    model.reset();
    auto fresh = make_upscaler(UpscaleMethod::kRealEsrgan, model_path, backend);
    same_bits(last_reference, fresh->upscale(a, 1, height, width, last_options),
              "fresh model versus reused model");
  }
}
