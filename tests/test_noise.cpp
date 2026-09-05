// Latent noise initialisation.
//
// This is not tested against PyTorch — see the header for why the project
// deliberately does not reproduce torch's RNG. What is tested is everything
// that would make the noise *wrong* rather than merely different: biased
// moments, correlated neighbours, a stream that depends on draw order, or a
// generator whose output collapses for adjacent seeds.

#include <cmath>
#include <cstdint>
#include <vector>

#include "harness.h"
#include "slopfab/sampler/noise.h"

namespace {

using namespace slopfab::sampler;

struct Moments {
  double mean = 0.0;
  double variance = 0.0;
  double skew = 0.0;
  double kurtosis = 0.0;  // excess
  double min = 0.0;
  double max = 0.0;
};

Moments moments(const std::vector<float>& v) {
  Moments m;
  const double n = static_cast<double>(v.size());
  double s1 = 0.0;
  for (float x : v) s1 += x;
  m.mean = s1 / n;
  double s2 = 0.0;
  double s3 = 0.0;
  double s4 = 0.0;
  m.min = v[0];
  m.max = v[0];
  for (float x : v) {
    const double d = static_cast<double>(x) - m.mean;
    s2 += d * d;
    s3 += d * d * d;
    s4 += d * d * d * d;
    if (x < m.min) m.min = x;
    if (x > m.max) m.max = x;
  }
  m.variance = s2 / n;
  const double sd = std::sqrt(m.variance);
  m.skew = (s3 / n) / (sd * sd * sd);
  m.kurtosis = (s4 / n) / (m.variance * m.variance) - 3.0;
  return m;
}

SLOPFAB_TEST(noise_moments) {
  std::vector<float> v(1 << 20);
  fill_normal(1234, NoiseStream::kVideoLatents, v.data(), v.size());
  const Moments m = moments(v);

  // With a million samples the standard error of the mean is 1e-3, so 5e-3 is
  // a five-sigma band: tight enough to catch a real bias, loose enough not to
  // flake.
  CHECK_MSG(std::fabs(m.mean) < 5e-3, "mean %.6f is too far from 0", m.mean);
  CHECK_MSG(std::fabs(m.variance - 1.0) < 1e-2, "variance %.6f is too far from 1", m.variance);
  CHECK_MSG(std::fabs(m.skew) < 2e-2, "skew %.6f is too far from 0", m.skew);
  CHECK_MSG(std::fabs(m.kurtosis) < 5e-2, "excess kurtosis %.6f is too far from 0", m.kurtosis);

  // A Gaussian million-sample draw reaches roughly +-5 sigma. A generator
  // stuck in a narrow band, or one emitting infinities from a log(0), fails
  // one of these.
  CHECK_MSG(m.max > 3.5 && m.max < 7.0, "max %.4f is implausible", m.max);
  CHECK_MSG(m.min < -3.5 && m.min > -7.0, "min %.4f is implausible", m.min);

  bool finite = true;
  for (float x : v) finite = finite && std::isfinite(x);
  CHECK(finite);
}

SLOPFAB_TEST(noise_is_position_independent) {
  // Element i must depend only on (seed, stream, i). Filling a long buffer and
  // a short one must agree on the overlap — if it does not, the generator is
  // carrying state and the audio noise would shift whenever the video
  // resolution changed.
  std::vector<float> a(10000);
  std::vector<float> b(37);
  fill_normal(99, NoiseStream::kVideoLatents, a.data(), a.size());
  fill_normal(99, NoiseStream::kVideoLatents, b.data(), b.size());
  bool same = true;
  for (size_t i = 0; i < b.size(); ++i) same = same && a[i] == b[i];
  CHECK(same);

  // An odd count must not disturb the values before it.
  std::vector<float> c(101);
  fill_normal(99, NoiseStream::kVideoLatents, c.data(), c.size());
  bool prefix = true;
  for (size_t i = 0; i < c.size(); ++i) prefix = prefix && a[i] == c[i];
  CHECK(prefix);
}

SLOPFAB_TEST(noise_streams_and_seeds_are_independent) {
  const size_t n = 1 << 16;
  std::vector<float> video(n);
  std::vector<float> audio(n);
  fill_normal(7, NoiseStream::kVideoLatents, video.data(), n);
  fill_normal(7, NoiseStream::kAudioLatents, audio.data(), n);

  // The two streams off one seed must not be the same sequence, nor a shift of
  // it. Correlation is the direct test.
  double dot = 0.0;
  for (size_t i = 0; i < n; ++i) dot += static_cast<double>(video[i]) * audio[i];
  const double corr = dot / static_cast<double>(n);
  CHECK_MSG(std::fabs(corr) < 0.02, "video and audio streams correlate at %.5f", corr);

  bool identical = true;
  for (size_t i = 0; i < n; ++i) identical = identical && video[i] == audio[i];
  CHECK(!identical);

  // Adjacent seeds must decorrelate too — this is what splitmix on the seed is
  // for, and without it seeds 0 and 1 produce visibly related draws.
  std::vector<float> s0(n);
  std::vector<float> s1(n);
  fill_normal(0, NoiseStream::kVideoLatents, s0.data(), n);
  fill_normal(1, NoiseStream::kVideoLatents, s1.data(), n);
  double dot01 = 0.0;
  for (size_t i = 0; i < n; ++i) dot01 += static_cast<double>(s0[i]) * s1[i];
  CHECK_MSG(std::fabs(dot01 / static_cast<double>(n)) < 0.02,
            "seeds 0 and 1 correlate at %.5f", dot01 / static_cast<double>(n));
}

SLOPFAB_TEST(noise_has_no_short_range_structure) {
  // Autocorrelation at small lags catches a generator whose counter mixing is
  // too weak — the classic failure of a hand-rolled counter-based RNG, and one
  // that would print as visible structure in the first denoising step.
  const size_t n = 1 << 18;
  std::vector<float> v(n);
  fill_normal(4242, NoiseStream::kVideoLatents, v.data(), n);

  for (int lag = 1; lag <= 16; ++lag) {
    double dot = 0.0;
    for (size_t i = 0; i + lag < n; ++i) dot += static_cast<double>(v[i]) * v[i + lag];
    const double corr = dot / static_cast<double>(n - lag);
    CHECK_MSG(std::fabs(corr) < 0.01, "autocorrelation at lag %d is %.5f", lag, corr);
  }
}

SLOPFAB_TEST(noise_shapes) {
  // Video noise is channel-major (24, F, Hl, Wl), the layout patchify_video
  // reads.
  const std::vector<float> v = video_noise(11, 37, 48, 84);
  CHECK(v.size() == static_cast<size_t>(24) * 37 * 48 * 84);

  // Audio noise is drawn directly in row layout (2A, 32), not (2, 32, A).
  const std::vector<float> a = audio_noise(11, 207);
  CHECK(a.size() == static_cast<size_t>(2 * 207) * 32);

  // Determinism across calls.
  const std::vector<float> v2 = video_noise(11, 37, 48, 84);
  CHECK(v == v2);

  // A different seed gives different noise.
  const std::vector<float> v3 = video_noise(12, 37, 48, 84);
  CHECK(v != v3);

  // Changing the video geometry must not change the audio noise: the two are
  // separate streams, so audio is stable under a resolution change.
  const std::vector<float> a2 = audio_noise(11, 207);
  CHECK(a == a2);

  CHECK(::slopfab::test::throws([] { video_noise(1, 0, 48, 84); }));
  CHECK(::slopfab::test::throws([] { audio_noise(1, -1); }));
}

}  // namespace
