#include "vidfab/dit/packing.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace vidfab::dit {
namespace {

// packing.py:47-95. These are checkpoint contracts, not tunables.
constexpr int kFps = 24;
constexpr int kShortEdge = 768;
constexpr int kMaxPixels = 768 * 1344;
constexpr int kCanvasMultiple = 32;
constexpr double kMinAspect = 1.0 / 4.0;
constexpr double kMaxAspect = 4.0;
constexpr int kFramesPerChunk = 17;
constexpr int kLatentsPerChunk = 5;
constexpr int kAudioLatentsPerSecond = 40;
constexpr int kAudioChannels = 2;

constexpr double kRopeFrameRescale = 5.0 / 3.0;
constexpr int kRopeFramesPerLatent[5] = {1, 4, 4, 4, 4};
constexpr double kRopeSpatialScale = 32.0;

// Python's round() is round-half-to-**even**, and the canvas arithmetic can
// land exactly on a half (a 2:1 ratio puts the long edge at 1536, but an
// awkward ratio need not). std::round is round-half-away-from-zero, so it is
// the wrong function here by one multiple of 32 on those inputs.
double round_half_even(double v) {
  const double floor_v = std::floor(v);
  const double frac = v - floor_v;
  if (frac > 0.5) return floor_v + 1.0;
  if (frac < 0.5) return floor_v;
  return (std::fmod(floor_v, 2.0) == 0.0) ? floor_v : floor_v + 1.0;
}

// np.linspace(left, left + ratio, n, endpoint=False) * 32.
//
// numpy computes `arange(n) * step + start` with `step = (stop - start) / n`,
// then the caller scales by 32 as a separate multiply. torch.linspace computes
// a different expression, and packing.py:338-340 warns about exactly this. The
// operation order below is the numpy one and must not be folded.
//
// The subtlety worth spelling out: `delta` is recomputed as `(left + ratio) -
// left` rather than reused as `ratio`, because that is literally what numpy
// does — `delta = stop - start` — and in float64 the round trip is not the
// identity. Using `ratio` directly leaves the width grid one ulp off for a
// 48x84 latent canvas. Verified bitwise against numpy for every even
// height/width in [16, 200).
std::vector<double> spatial_position_grid(int dim, int patch, double sqrt_area) {
  const int n = dim / patch;
  const double ratio = static_cast<double>(dim) / sqrt_area;
  const double left = (1.0 - ratio) / 2.0;
  const double delta = (left + ratio) - left;
  const double step = delta / static_cast<double>(n);

  std::vector<double> grid(static_cast<size_t>(n));
  for (int k = 0; k < n; ++k) {
    grid[static_cast<size_t>(k)] = (static_cast<double>(k) * step + left) * kRopeSpatialScale;
  }
  return grid;
}

// packing.py:344-353. Non-uniform spacing: 5/3 * (1, 4, 4, 4, 4) repeating,
// mirroring the VAE's 17-pixel-frames-to-5-latent-frames grouping. T(0) = 0.
std::vector<double> temporal_position_grid(int num_latent_frames, double origin) {
  std::vector<double> t(static_cast<size_t>(num_latent_frames));
  double acc = 0.0;
  for (int f = 0; f < num_latent_frames; ++f) {
    t[static_cast<size_t>(f)] = origin + acc;
    acc += kRopeFrameRescale * kRopeFramesPerLatent[f % 5];
  }
  return t;
}

}  // namespace

void resolve_canvas_size(double aspect_w, double aspect_h, int* out_h, int* out_w) {
  if (aspect_w <= 0.0 || aspect_h <= 0.0) {
    throw std::runtime_error("resolve_canvas_size: aspect ratio must be positive");
  }
  const double ratio = aspect_w / aspect_h;
  if (!(ratio >= kMinAspect && ratio <= kMaxAspect)) {
    throw std::runtime_error("resolve_canvas_size: MiniMax-H3 supports 1:4 to 4:1, got ratio " +
                             std::to_string(ratio));
  }

  double width;
  double height;
  if (ratio >= 1.0) {
    width = kShortEdge * ratio;
    height = static_cast<double>(kShortEdge);
  } else {
    width = static_cast<double>(kShortEdge);
    height = kShortEdge / ratio;
  }

  const double area = width * height;
  if (area > kMaxPixels) {
    const double scale = std::sqrt(kMaxPixels / area);
    width *= scale;
    height *= scale;
  }

  *out_h = std::max(kCanvasMultiple,
                    static_cast<int>(round_half_even(height / kCanvasMultiple)) * kCanvasMultiple);
  *out_w = std::max(kCanvasMultiple,
                    static_cast<int>(round_half_even(width / kCanvasMultiple)) * kCanvasMultiple);
}

int align_num_frames(int num_frames) {
  if (num_frames < 1) {
    throw std::runtime_error("align_num_frames: num_frames must be positive");
  }
  while (num_frames % kFramesPerChunk != kLatentsPerChunk) ++num_frames;
  return num_frames;
}

int video_latent_num_frames(int aligned_frames) {
  if (aligned_frames % kFramesPerChunk != kLatentsPerChunk) {
    throw std::runtime_error("video_latent_num_frames: expected 17*k + 5, got " +
                             std::to_string(aligned_frames));
  }
  return (aligned_frames - kLatentsPerChunk) / kFramesPerChunk * kLatentsPerChunk + 2;
}

int audio_latents_for_frames(int aligned_frames) {
  // round(frames / fps * 40), matching before_denoise.py's Python round.
  const double latents =
      static_cast<double>(aligned_frames) / kFps * static_cast<double>(kAudioLatentsPerSecond);
  return static_cast<int>(round_half_even(latents));
}

PackedIndices build_indices(const SequenceLayout& layout) {
  const int condition_start = layout.condition_start();
  const int audio_start = layout.audio_start();
  const int video_start = layout.video_start();
  const int total = layout.total_rows();

  PackedIndices out;
  out.text.reserve(static_cast<size_t>(layout.num_text));
  for (int i = 0; i < layout.num_text; ++i) out.text.push_back(i);

  out.audio.reserve(static_cast<size_t>(layout.num_audio_rows));
  for (int i = audio_start; i < video_start; ++i) out.audio.push_back(i);

  // Conditioning rows first, then targets — so `latents` is [conditions |
  // targets] and `latents[C:]` is the generated part. C is 0 for t2va, but the
  // ordering is what makes the denoise loop's slicing correct when it is not.
  out.video.reserve(static_cast<size_t>(layout.num_condition_video + layout.num_video_rows));
  for (int i = condition_start; i < audio_start; ++i) out.video.push_back(i);
  for (int i = video_start; i < total; ++i) out.video.push_back(i);

  // The assignment order matters: a text row belonging to a keyframe's vision
  // block is tagged 0 (video) even though it lives in the text region, and the
  // video pass overwrites the text pass for those rows. For t2va every text
  // row is simply 1.
  out.tags.assign(static_cast<size_t>(total), kTagText);
  for (int i : out.audio) out.tags[static_cast<size_t>(i)] = kTagAudio;
  for (int i : out.video) out.tags[static_cast<size_t>(i)] = kTagVideo;

  return out;
}

std::vector<double> build_position_ids(const SequenceLayout& layout) {
  const int total = layout.total_rows();
  const int L = layout.num_text;
  const int audio_start = layout.audio_start();
  const int video_start = layout.video_start();
  const int Hl = layout.latent_height;
  const int Wl = layout.latent_width;
  const int rows_per_frame = layout.rows_per_frame();

  std::vector<double> pos(static_cast<size_t>(total) * 3, 0.0);

  // Text: (i, 0, 0). The media clock then continues from L, so prompt length
  // shifts every media coordinate on the time axis.
  for (int i = 0; i < L; ++i) pos[static_cast<size_t>(i) * 3 + 0] = static_cast<double>(i);

  const double sqrt_area = std::sqrt(static_cast<double>(Hl) * static_cast<double>(Wl));
  const std::vector<double> h_grid = spatial_position_grid(Hl, 2, sqrt_area);
  const std::vector<double> w_grid = spatial_position_grid(Wl, 2, sqrt_area);

  // Audio: channel-major, sharing the video's 40-units-per-second clock. No
  // height coordinate; the two stereo channels are distinguished only by being
  // pinned to opposite extremes of the width grid.
  //
  // Note the origin is `L`, not `audio_start`: keyframe condition rows do not
  // advance the audio clock (packing.py:433).
  const int A = layout.num_audio_latents;
  for (int c = 0; c < kAudioChannels; ++c) {
    const double w = (c == 0) ? w_grid.front() : w_grid.back();
    for (int a = 0; a < A; ++a) {
      const size_t row = static_cast<size_t>(audio_start + c * A + a);
      pos[row * 3 + 0] = static_cast<double>(L) + static_cast<double>(a);
      pos[row * 3 + 1] = 0.0;
      pos[row * 3 + 2] = w;
    }
  }

  // Video: frame-major, then the same hh*(Wl/2)+ww ordering the patchifier
  // uses. A mismatch between the two scrambles the spatial rotary without
  // changing any shape.
  const std::vector<double> t_grid = temporal_position_grid(layout.num_latent_frames, static_cast<double>(L));
  const int half_w = Wl / 2;
  for (int f = 0; f < layout.num_latent_frames; ++f) {
    for (int r = 0; r < rows_per_frame; ++r) {
      const int hh = r / half_w;
      const int ww = r % half_w;
      const size_t row = static_cast<size_t>(video_start + f * rows_per_frame + r);
      pos[row * 3 + 0] = t_grid[static_cast<size_t>(f)];
      pos[row * 3 + 1] = h_grid[static_cast<size_t>(hh)];
      pos[row * 3 + 2] = w_grid[static_cast<size_t>(ww)];
    }
  }

  return pos;
}

void patchify_video(const float* latents, const SequenceLayout& layout, float* rows_out) {
  const int C = 24;
  const int F = layout.num_latent_frames;
  const int Hl = layout.latent_height;
  const int Wl = layout.latent_width;
  const int half_h = Hl / 2;
  const int half_w = Wl / 2;
  const int feature_dim = C * 4;

  const size_t frame_stride = static_cast<size_t>(Hl) * Wl;
  const size_t channel_stride = frame_stride * F;

  for (int f = 0; f < F; ++f) {
    for (int hh = 0; hh < half_h; ++hh) {
      for (int ww = 0; ww < half_w; ++ww) {
        const size_t row = (static_cast<size_t>(f) * half_h + hh) * half_w + ww;
        float* dst = rows_out + row * feature_dim;
        for (int c = 0; c < C; ++c) {
          for (int dh = 0; dh < 2; ++dh) {
            for (int dw = 0; dw < 2; ++dw) {
              const size_t src = static_cast<size_t>(c) * channel_stride +
                                 static_cast<size_t>(f) * frame_stride +
                                 static_cast<size_t>(2 * hh + dh) * Wl + (2 * ww + dw);
              dst[c * 4 + dh * 2 + dw] = latents[src];
            }
          }
        }
      }
    }
  }
}

void unpatchify_video(const float* rows, const SequenceLayout& layout, float* latents_out) {
  const int C = 24;
  const int F = layout.num_latent_frames;
  const int Hl = layout.latent_height;
  const int Wl = layout.latent_width;
  const int half_h = Hl / 2;
  const int half_w = Wl / 2;
  const int feature_dim = C * 4;

  const size_t frame_stride = static_cast<size_t>(Hl) * Wl;
  const size_t channel_stride = frame_stride * F;

  for (int f = 0; f < F; ++f) {
    for (int hh = 0; hh < half_h; ++hh) {
      for (int ww = 0; ww < half_w; ++ww) {
        const size_t row = (static_cast<size_t>(f) * half_h + hh) * half_w + ww;
        const float* src = rows + row * feature_dim;
        for (int c = 0; c < C; ++c) {
          for (int dh = 0; dh < 2; ++dh) {
            for (int dw = 0; dw < 2; ++dw) {
              const size_t dst = static_cast<size_t>(c) * channel_stride +
                                 static_cast<size_t>(f) * frame_stride +
                                 static_cast<size_t>(2 * hh + dh) * Wl + (2 * ww + dw);
              latents_out[dst] = src[c * 4 + dh * 2 + dw];
            }
          }
        }
      }
    }
  }
}

void unpack_audio(const float* rows, int num_audio_latents, float* out) {
  // (2A, 32) -> (2, A, 32) -> permute(0, 2, 1) -> (2, 32, A)
  const int channels = kAudioChannels;
  const int dim = 32;
  for (int c = 0; c < channels; ++c) {
    for (int a = 0; a < num_audio_latents; ++a) {
      for (int d = 0; d < dim; ++d) {
        const size_t src = (static_cast<size_t>(c) * num_audio_latents + a) * dim + d;
        const size_t dst = (static_cast<size_t>(c) * dim + d) * num_audio_latents + a;
        out[dst] = rows[src];
      }
    }
  }
}

RowTimesteps build_row_timesteps(const SequenceLayout& layout, const PackedIndices& idx,
                                 float video_t, float audio_t) {
  const int total = layout.total_rows();

  // Every row defaults to the video timestep — including text rows, which are
  // never overridden and so inherit it.
  std::vector<float> row_t(static_cast<size_t>(total), video_t);
  for (size_t i = static_cast<size_t>(layout.num_condition_video); i < idx.audio.size(); ++i) {
    row_t[static_cast<size_t>(idx.audio[i])] = audio_t;
  }

  // torch.unique(sorted=True, return_inverse=True). Ascending, so which of the
  // two timesteps is index 0 flips over the schedule as the shifted video and
  // audio grids cross. That feeds the AdaLN index, so it has to be reproduced
  // rather than fixed by convention.
  RowTimesteps out;
  out.unique = row_t;
  std::sort(out.unique.begin(), out.unique.end());
  out.unique.erase(std::unique(out.unique.begin(), out.unique.end()), out.unique.end());

  out.indices.resize(static_cast<size_t>(total));
  out.adaln.resize(static_cast<size_t>(total));
  for (int s = 0; s < total; ++s) {
    const auto it = std::lower_bound(out.unique.begin(), out.unique.end(), row_t[static_cast<size_t>(s)]);
    const int32_t ti = static_cast<int32_t>(it - out.unique.begin());
    out.indices[static_cast<size_t>(s)] = ti;
    out.adaln[static_cast<size_t>(s)] = ti * 3 + std::max(idx.tags[static_cast<size_t>(s)], 0);
  }
  return out;
}

}  // namespace vidfab::dit
