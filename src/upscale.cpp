#include "slopfab/upscale.h"
#include "slopfab/tensor_convert.h"
#include "upscale/backend.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace slopfab {
using namespace upscale_detail;

#if SLOPFAB_WITH_CUDA
std::unique_ptr<Upscaler> make_seedvr2_upscaler(const std::string& checkpoint);
#endif

std::unique_ptr<Upscaler> make_upscaler(UpscaleMethod method, const std::string& checkpoint,
                                        DeviceBackend backend) {
  switch (method) {
  case UpscaleMethod::kRealEsrgan:
    return std::make_unique<RealEsrgan>(checkpoint, backend);
  case UpscaleMethod::kSeedVr2:
#if SLOPFAB_WITH_CUDA
    if (backend == DeviceBackend::kCuda) return make_seedvr2_upscaler(checkpoint);
#endif
    throw std::invalid_argument("SeedVR2 requires the CUDA backend");
  }
  throw std::invalid_argument("unknown upscale method");
}

struct RealEsrgan::Impl {
  struct Conv {
    Buffer weight, bias;
    int input, output;
  };

  std::unique_ptr<Backend> backend;
  std::vector<Conv> convolutions;

  Impl(const std::string& path, DeviceBackend device) {
    SafeTensors file;
    file.open(path);
    validate_realesrgan_checkpoint(file);
    switch (device) {
#if SLOPFAB_WITH_CUDA
    case DeviceBackend::kCuda:
      backend = make_cuda_backend();
      break;
#endif
#if SLOPFAB_WITH_VULKAN
    case DeviceBackend::kVulkan:
      backend = make_vulkan_backend();
      break;
#endif
    default:
      throw std::invalid_argument("Real-ESRGAN: requested backend was not compiled in");
    }
    auto upload = [&](const std::string& name) {
      auto data = to_f32(file.at(name));
      for (float v : data)
        if (!std::isfinite(v))
          throw std::invalid_argument("Real-ESRGAN: non-finite weight in " + name);
      return backend->upload(data);
    };
    for (const auto& spec : model_convolutions())
      convolutions.push_back(
          {upload(spec.name + ".weight"), upload(spec.name + ".bias"), spec.input, spec.output});
  }

  Buffer operation(Operation op, const Buffer& x, const Buffer& y, int h, int w, int in, int out,
                   float scale = 0) {
    const size_t n = size_t(h) * w * out;
    auto dst = backend->allocate(n);
    backend->run({op, uint32_t(h), uint32_t(w), uint32_t(in), uint32_t(out), uint32_t(n), 0, scale},
                 x, y, y, dst);
    return dst;
  }

  std::vector<float> tile(const std::vector<float>& pixels, int h, int w) {
    // All GPU kernels use 32-bit indices, including their largest im2col workspace.
    if (uint64_t(h) * w > std::numeric_limits<int>::max() / (16 * 64 * 9))
      throw std::length_error("Real-ESRGAN: tile is too large; reduce --upscale-tile");
    size_t layer = 0;
    auto conv = [&](Buffer x, bool leaky, bool nearest = false) {
      const auto& c = convolutions.at(layer++);
      const size_t n = size_t(h) * w * c.output;
      if (nearest)
        return backend->nearest_conv(
            std::move(x), c.weight, c.bias,
            {kConv, uint32_t(h), uint32_t(w), uint32_t(c.input), uint32_t(c.output),
             uint32_t(n), uint32_t(leaky), 0});
      auto dst = backend->allocate(n);
      backend->run({kConv, uint32_t(h), uint32_t(w), uint32_t(c.input), uint32_t(c.output),
                    uint32_t(n), uint32_t(leaky), 0},
                   x, c.weight, c.bias, dst);
      return dst;
    };
    auto first = conv(backend->upload(pixels), false);
    auto x = first;
    for (int block = 0; block < 23; ++block) {
      auto skip = x;
      for (int rdb = 0; rdb < 3; ++rdb) {
        auto dense = x;
        for (int i = 0; i < 4; ++i) {
          auto feature = conv(dense, true);
          dense = operation(kConcat, dense, feature, h, w, 64 + i * 32, 96 + i * 32);
        }
        x = operation(kResidual, x, conv(dense, false), h, w, 64, 64, 0.2f);
      }
      x = operation(kResidual, skip, x, h, w, 64, 64, 0.2f);
    }
    x = operation(kResidual, first, conv(x, false), h, w, 64, 64, 1.0f);
    first.reset();
    for (int i = 0; i < 2; ++i) {
      h *= 2;
      w *= 2;
      x = conv(std::move(x), true, true);
    }
    x = conv(x, true);
    x = conv(x, false);
    return backend->download(x, size_t(h) * w * 3);
  }
};

RealEsrgan::RealEsrgan(const std::string& path, DeviceBackend backend)
    : impl_(std::make_unique<Impl>(path, backend)) {
}

RealEsrgan::~RealEsrgan() = default;

PixelBuffer RealEsrgan::upscale(const PixelBuffer& input, int frames, int height, int width,
                                const UpscaleOptions& o,
                                const std::function<bool(int, int)>& progress) {
  validate_upscale_options(o);
  const size_t count = upscale_output_elements(frames, height, width);
  if (input.size() != count / 16)
    throw std::invalid_argument("Real-ESRGAN: input buffer does not match RGB dimensions");
  for (float v : input)
    if (!std::isfinite(v))
      throw std::invalid_argument("Real-ESRGAN: non-finite input pixel");
  const int ph = height + o.pre_pad, pw = width + o.pre_pad;
  const int tile = o.tile_size ? o.tile_size : std::max(ph, pw);
  const uint64_t max_height = std::min(ph, tile + 2 * o.tile_pad);
  const uint64_t max_width = std::min(pw, tile + 2 * o.tile_pad);
  if (max_height * max_width > std::numeric_limits<int>::max() / (16 * 64 * 9))
    throw std::length_error("Real-ESRGAN: tile is too large; reduce --upscale-tile");
  const int64_t tiles = int64_t((ph - 1) / tile + 1) * ((pw - 1) / tile + 1) * frames;
  if (tiles > std::numeric_limits<int>::max())
    throw std::length_error("Real-ESRGAN: too many tiles");
  auto notify = [&](int done) {
    if (progress && !progress(done, int(tiles)))
      throw UpscaleCancelled();
  };
  notify(0);
  PixelBuffer output(count);
  // Repeated reflection also defines padding for images smaller than pre_pad.
  auto reflect = [](int p, int size) {
    if (size == 1)
      return 0;
    const int64_t period = 2 * int64_t(size - 1);
    const int64_t q = p % period;
    return int(q < size ? q : period - q);
  };
  int done = 0;
  for (int f = 0; f < frames; ++f)
    for (int y = 0; y < ph; y += tile)
      for (int x = 0; x < pw; x += tile) {
        notify(done);
        const int x0 = std::max(0, x - o.tile_pad), y0 = std::max(0, y - o.tile_pad);
        const int xe = std::min(pw, x + tile), ye = std::min(ph, y + tile);
        const int tw = std::min(pw, xe + o.tile_pad) - x0;
        const int th = std::min(ph, ye + o.tile_pad) - y0;
        std::vector<float> pixels(size_t(tw) * th * 3);
        for (int ty = 0; ty < th; ++ty)
          for (int tx = 0; tx < tw; ++tx)
            for (int c = 0; c < 3; ++c)
              pixels[(size_t(ty) * tw + tx) * 3 + c] = std::clamp(
                  input[((size_t(c) * frames + f) * height + reflect(y0 + ty, height)) * width +
                        reflect(x0 + tx, width)],
                  0.0f, 1.0f);
        const auto up = impl_->tile(pixels, th, tw);
        for (int oy = y * 4; oy < std::min(ye, height) * 4; ++oy)
          for (int ox = x * 4; ox < std::min(xe, width) * 4; ++ox)
            for (int c = 0; c < 3; ++c) {
              const float v = up[(size_t(oy - y0 * 4) * tw * 4 + ox - x0 * 4) * 3 + c];
              if (!std::isfinite(v))
                throw std::runtime_error("Real-ESRGAN: non-finite output pixel");
              output[((size_t(c) * frames + f) * height * 4 + oy) * width * 4 + ox] =
                  std::clamp(v, 0.0f, 1.0f);
            }
        ++done;
      }
  notify(done);
  return output;
}
} // namespace slopfab
