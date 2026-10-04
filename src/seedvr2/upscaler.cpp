#include "slopfab/upscale.h"
#include "slopfab/image.h"
#include <algorithm>
#include <cmath>

namespace slopfab {
namespace {
class SeedVr2Upscaler final : public Upscaler {
  std::string checkpoint_;

public:
  explicit SeedVr2Upscaler(std::string checkpoint) : checkpoint_(std::move(checkpoint)) {
  }

  PixelBuffer upscale(const PixelBuffer& input, int frames, int height, int width,
                      const UpscaleOptions& options,
                      const std::function<bool(int, int)>& progress) override {
    const auto count =
        upscale_output_elements(frames, height, width, UpscaleMethod::kSeedVr2, options);
    if (input.size() != size_t(frames) * height * width * 3)
      throw std::invalid_argument("SeedVR2: input buffer does not match RGB dimensions");
    auto o = seedvr2_options(options, height, width);
    o.transformer = checkpoint_;
    int read = 0, written = 0;
    auto notify = [&] {
      if (progress && !progress(written, frames))
        throw UpscaleCancelled();
    };
    notify();
    seedvr2::Restorer model(o);
    model.progress = [&](const std::string&) {
      notify();
    };
    PixelBuffer output(count);
    seedvr2::stream(
        o,
        [&](seedvr2::Frame& frame) {
          if (read == frames)
            return false;
          notify();
          RGBImage image;
          image.width = width;
          image.height = height;
          image.pixels.resize(size_t(width) * height * 3);
          for (size_t p = 0; p < size_t(width) * height; ++p)
            for (size_t c = 0; c < 3; ++c) {
              const float v = input[(c * frames + read) * height * width + p];
              if (!std::isfinite(v))
                throw std::invalid_argument("SeedVR2: nonfinite input pixel");
              image.pixels[p * 3 + c] = uint8_t(std::lround(std::clamp(v, 0.0f, 1.0f) * 255));
            }
          image = resize_reference_lanczos(image, o.width, o.height);
          frame.resize(image.pixels.size());
          for (size_t i = 0; i < frame.size(); ++i)
            frame[i] = image.pixels[i] / 255.0f;
          ++read;
          return true;
        },
        [&](const seedvr2::Frame& frame) {
          notify();
          const size_t plane = size_t(o.width) * o.height;
          for (size_t p = 0; p < plane; ++p)
            for (size_t c = 0; c < 3; ++c)
              output[(c * frames + written) * plane + p] = frame[p * 3 + c];
          ++written;
        },
        [&](const std::vector<seedvr2::Frame>& segment, uint64_t first) {
          return model.restore(segment, first);
        });
    notify();
    return output;
  }
};
}

std::unique_ptr<Upscaler> make_seedvr2_upscaler(const std::string& checkpoint) {
  return std::make_unique<SeedVr2Upscaler>(checkpoint);
}
}
