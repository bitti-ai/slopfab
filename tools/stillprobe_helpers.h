#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "slopfab/vae/vit_decoder.h"

namespace slopfab::probe {

inline std::vector<float> repeat_still_latent(const std::vector<float>& latent, int channels,
                                              int frames) {
  if (channels <= 0 || frames <= 0 || latent.empty() ||
      latent.size() % static_cast<size_t>(channels) != 0)
    throw std::invalid_argument("invalid still latent dimensions");
  const size_t pixels = latent.size() / channels;
  std::vector<float> repeated(latent.size() * frames);
  for (int channel = 0; channel < channels; ++channel) {
    for (int frame = 0; frame < frames; ++frame) {
      std::copy_n(latent.data() + channel * pixels, pixels,
                  repeated.data() + (static_cast<size_t>(channel) * frames + frame) * pixels);
    }
  }
  return repeated;
}

inline vae::DecodedVideo first_frame(const vae::DecodedVideo& video) {
  if (video.channels <= 0 || video.frames <= 0 || video.height <= 0 || video.width <= 0 ||
      video.data.size() !=
          static_cast<size_t>(video.channels) * video.frames * video.height * video.width)
    throw std::invalid_argument("invalid decoded video dimensions");
  vae::DecodedVideo image;
  image.channels = video.channels;
  image.frames = 1;
  image.height = video.height;
  image.width = video.width;
  const size_t pixels = image.frame_stride();
  image.data.resize(image.channels * pixels);
  for (int channel = 0; channel < image.channels; ++channel) {
    std::copy_n(video.data.data() + static_cast<size_t>(channel) * video.frames * pixels, pixels,
                image.data.data() + channel * pixels);
  }
  return image;
}

}
