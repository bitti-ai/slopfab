#include "reference_decode.h"
#include "native_media.h"

namespace slopfab::cli {
ReferenceMedia decode_reference_file(const std::string& path, bool video, const char*) {
  ReferenceMedia result;
  if (video) {
    auto input = read_y4m(path);
    const double dt = double(input.fps.denominator) / input.fps.numerator;
    result = ReferenceMedia::video(input.frames.size() * dt);
    for (size_t i = 0; i < input.frames.size(); ++i)
      result.append_frame(input.frames[i].data(), input.frames[i].size(), input.width, input.height,
                          size_t(input.width) * 3, 3, i * dt);
  } else {
    auto input = read_wav(path);
    result = ReferenceMedia::audio(input.interleaved.data(), input.interleaved.size(),
                                   input.channels, input.sample_rate);
  }
  result.validate();
  return result;
}
}
