// Deterministic Gaussian noise for latent initialisation.
//
// **This deliberately does not reproduce PyTorch's RNG.** Matching a torch
// seed bit for bit would mean reimplementing Philox 10-round with torch's
// exact offset bookkeeping and its exact Box-Muller pairing and consumption
// order, and the payoff would be reproducing a *specific reference sample*.
// That is not the project's correctness criterion: the criterion is that the
// model matches the reference to 1e-3 absolute / 1e-2 relative *given the same
// input*, and a different seed is a different valid sample, not a wrong one.
//
// What does matter, and what this provides:
//
//   - **Determinism.** The same seed gives the same video on the same build,
//     which is what makes a bug reproducible.
//   - **Position independence.** Element `i` depends only on `(seed, stream,
//     i)`, not on how many values were drawn before it. A counter-based
//     generator gets this for free, and it means the draw order of the video
//     and audio noise cannot silently couple: changing the video resolution
//     does not change the audio noise.
//   - **Correct statistics.** Unit-variance, zero-mean, and no structure a
//     32x32 correlation test can find.
//
// The reference's draw order is documented in docs/transformer_spec.md section
// 1.3 and is reproduced here in structure — video first, then audio, audio
// drawn directly in row layout `(Sa, 32)` rather than as `(2, 32, A)` and
// permuted — because that ordering is part of the *shape* contract even when
// the values differ.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace slopfab::sampler {

// Independent streams off one seed, so adding a draw site never shifts an
// existing one.
enum class NoiseStream : uint32_t {
  kVideoLatents = 0,
  kAudioLatents = 1,
};

// Counter-based generator: `value(i)` is a pure function of the key and the
// index. Threefry-style, which is cheap, passes the usual statistical batteries
// at this quality bar, and needs no state.
class CounterRng {
public:
  CounterRng(uint64_t seed, NoiseStream stream);

  // Two independent uniforms in [0, 1) for counter `index`.
  void uniform2(uint64_t index, float* out) const;

  // Two independent standard normals for counter `index`, by Box-Muller.
  void normal2(uint64_t index, float* out) const;

private:
  uint64_t key0_;
  uint64_t key1_;
};

// Fills `count` standard normals. `out` must have room. Element `i` depends
// only on `(seed, stream, i)`.
void fill_normal(uint64_t seed, NoiseStream stream, float* out, size_t count);

// Video latent noise, `(24, F, Hl, Wl)` fp32 in channel-major layout — the
// layout `patchify_video` consumes.
std::vector<float> video_noise(uint64_t seed, int latent_frames, int latent_height,
                               int latent_width, int channels = 24);

// Audio latent noise, drawn **directly in row layout** `(2*A, 32)`.
//
// This is not the same as drawing `(2, 32, A)` and permuting: the reference
// draws the row-major form (before_denoise.py:319-324), and although our
// values differ from torch's anyway, keeping the shape contract identical
// means the two can be swapped without touching anything downstream.
std::vector<float> audio_noise(uint64_t seed, int num_audio_latents, int latent_channels = 32);

} // namespace slopfab::sampler
