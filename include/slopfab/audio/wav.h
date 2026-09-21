// RIFF/WAVE writer.
//
// The audio counterpart of the .y4m writer, and there for the same reason: it
// puts the samples on disk with nothing between them and the ear, so a wrong
// decode sounds wrong instead of being masked by a codec. It is also the
// fallback when ffmpeg is unavailable.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace slopfab::audio {

enum class SampleFormat {
  kPcm16,   // WAVE_FORMAT_PCM, the universally readable choice
  kFloat32, // WAVE_FORMAT_IEEE_FLOAT, lossless for debugging
};

// Writes interleaved float samples in [-1, 1]. Values outside the range are
// clamped for kPcm16 and passed through for kFloat32 — clipping is a decode
// bug worth hearing, not worth silently normalising away.
void write_wav(const std::string& path, const std::vector<float>& interleaved, int channels,
               int sample_rate, SampleFormat format = SampleFormat::kPcm16);

} // namespace slopfab::audio
