#include "slopfab/sampler/noise.h"

#include <cmath>
#include <stdexcept>

namespace slopfab::sampler {
namespace {

// Golden-ratio odd constants, the usual choice for a Weyl/Threefry mix. They
// only need to be odd and to have well-mixed bit patterns.
constexpr uint64_t kPhi0 = 0x9E3779B97F4A7C15ull;
constexpr uint64_t kPhi1 = 0xBF58476D1CE4E5B9ull;
constexpr uint64_t kPhi2 = 0x94D049BB133111EBull;

inline uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

// Four rounds of a Threefry-2x64-style mix. Enough diffusion that flipping one
// input bit changes about half the output bits, which is all that is needed
// here — this generates noise, not keys.
inline void mix(uint64_t& a, uint64_t& b) {
  static constexpr int kRot[4] = {16, 42, 12, 31};
  for (int r = 0; r < 4; ++r) {
    a += b;
    b = rotl(b, kRot[r]) ^ a;
  }
}

// splitmix64, to spread a small seed across the whole key space. Without it,
// seeds 0 and 1 would produce visibly related first draws.
inline uint64_t splitmix(uint64_t x) {
  x += kPhi0;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * kPhi1;
  z = (z ^ (z >> 27)) * kPhi2;
  return z ^ (z >> 31);
}

// Open interval (0, 1]: excludes zero so log() below never sees it. Uses 24
// bits, which is the fp32 mantissa.
inline float to_unit(uint64_t bits) {
  const uint32_t m = static_cast<uint32_t>(bits >> 40) & 0xFFFFFFu;  // 24 bits
  return (static_cast<float>(m) + 0.5f) * (1.0f / 16777216.0f);
}

}  // namespace

CounterRng::CounterRng(uint64_t seed, NoiseStream stream)
    : key0_(splitmix(seed)), key1_(splitmix(seed ^ (static_cast<uint64_t>(stream) + 0x5DEECE66Dull))) {}

void CounterRng::uniform2(uint64_t index, float* out) const {
  uint64_t a = key0_ ^ index;
  uint64_t b = key1_ + index * kPhi0;
  mix(a, b);
  out[0] = to_unit(a);
  out[1] = to_unit(b);
}

void CounterRng::normal2(uint64_t index, float* out) const {
  float u[2];
  uniform2(index, u);
  // Box-Muller. u[0] is in (0, 1] so the log is finite; the radius is computed
  // in double because -2*log(u) is ill-conditioned for u near 1, where the
  // result is near zero and fp32 cancellation would bite.
  const double r = std::sqrt(-2.0 * std::log(static_cast<double>(u[0])));
  const double theta = 6.283185307179586476925286766559 * static_cast<double>(u[1]);
  out[0] = static_cast<float>(r * std::cos(theta));
  out[1] = static_cast<float>(r * std::sin(theta));
}

void fill_normal(uint64_t seed, NoiseStream stream, float* out, size_t count) {
  const CounterRng rng(seed, stream);
  const size_t pairs = count / 2;
  for (size_t p = 0; p < pairs; ++p) {
    rng.normal2(p, out + p * 2);
  }
  if (count % 2 != 0) {
    float pair[2];
    rng.normal2(pairs, pair);
    out[count - 1] = pair[0];
  }
}

std::vector<float> video_noise(uint64_t seed, int latent_frames, int latent_height,
                               int latent_width, int channels) {
  if (latent_frames <= 0 || latent_height <= 0 || latent_width <= 0 || channels <= 0) {
    throw std::runtime_error("video_noise: every dimension must be positive");
  }
  const size_t n = static_cast<size_t>(channels) * latent_frames * latent_height * latent_width;
  std::vector<float> out(n);
  fill_normal(seed, NoiseStream::kVideoLatents, out.data(), n);
  return out;
}

std::vector<float> audio_noise(uint64_t seed, int num_audio_latents, int latent_channels) {
  if (num_audio_latents <= 0 || latent_channels <= 0) {
    throw std::runtime_error("audio_noise: every dimension must be positive");
  }
  const size_t rows = static_cast<size_t>(num_audio_latents) * 2;  // stereo, channel-major
  const size_t n = rows * static_cast<size_t>(latent_channels);
  std::vector<float> out(n);
  fill_normal(seed, NoiseStream::kAudioLatents, out.data(), n);
  return out;
}

}  // namespace slopfab::sampler
