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

void validate_canvas_size(int height, int width) {
  if (height <= 0 || width <= 0) {
    throw std::runtime_error("resolution: both axes must be positive, got " +
                             std::to_string(width) + "x" + std::to_string(height));
  }
  // 32 rather than 16: the VAE compresses by 16 and the patchifier then walks
  // 2x2 patches, so an axis that is a multiple of 16 but not 32 produces a
  // latent with an odd extent and a patch grid that silently drops its last
  // row or column.
  if (height % kCanvasMultiple != 0 || width % kCanvasMultiple != 0) {
    throw std::runtime_error("resolution: both axes must be a multiple of " +
                             std::to_string(kCanvasMultiple) + ", got " + std::to_string(width) +
                             "x" + std::to_string(height));
  }
  const double ratio = static_cast<double>(width) / static_cast<double>(height);
  if (!(ratio >= kMinAspect && ratio <= kMaxAspect)) {
    throw std::runtime_error("resolution: MiniMax-H3 supports 1:4 to 4:1, and " +
                             std::to_string(width) + "x" + std::to_string(height) + " is " +
                             std::to_string(ratio));
  }
}

bool canvas_exceeds_trained_area(int height, int width) {
  return static_cast<long long>(height) * width > static_cast<long long>(kMaxPixels);
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

  // This reproduces `torch.unique(row_t, sorted=True, return_inverse=True)`
  // over a per-row timestep vector that takes **exactly two values**: every row
  // defaults to the video timestep — including text rows, which are never
  // overridden and so inherit it — and the audio rows from
  // `num_condition_video` onward are set to the audio timestep. So the sorted
  // unique set is the ascending dedupe of at most `{video_t, audio_t}` and the
  // per-row index is one comparison. Neither the sort nor the S-element `row_t`
  // temporary is needed; both used to be here, and at S = 37 710 the sort was
  // the whole cost of the function.
  //
  // Ascending is load-bearing, not cosmetic: which of the two timesteps is
  // index 0 flips as the shifted video (shift 12) and audio (shift 3) grids
  // cross, and that index feeds the AdaLN table row (spec sections 3.2, 3.4 and
  // 7.5). It has to be reproduced rather than fixed by convention.

  // The audio loop deliberately starts at `num_condition_video` into
  // `idx.audio` — conditioning audio rows are not stepped and keep the video
  // timestep. C is 0 for t2va, but the offset is what makes fl2va right.
  // The unsigned cast is the old loop's, kept deliberately: a negative C wraps
  // to a huge value, the clamp makes it `size()`, and the loop runs zero times
  // exactly as the old `for (size_t i = C; i < audio.size(); ++i)` did.
  const size_t audio_begin =
      std::min(static_cast<size_t>(layout.num_condition_video), idx.audio.size());
  const size_t num_overridden = idx.audio.size() - audio_begin;

  // Which of the two values actually occurs. `video_t` is absent only when
  // every row is an overridden audio row; `audio_t` is absent when none is.
  // Getting this wrong would leave a value in `unique` that no row carries,
  // which shifts every index by one.
  //
  // Counting rather than scanning relies on `idx.audio` holding distinct rows,
  // which is what `build_indices` produces — it is the half-open range
  // [audio_start, video_start). That is the only precondition here beyond the
  // one the old code already had, which was that those indices are in range.
  const bool has_video = static_cast<size_t>(total) > num_overridden;
  const bool has_audio = num_overridden > 0;

  RowTimesteps out;
  int32_t video_index = 0;
  int32_t audio_index = 0;
  if (has_video && has_audio) {
    // `<` and `==` exactly as std::sort and std::unique used them, so the
    // equal case collapses to a single entry and every index is 0. (The two
    // are `==` but not bitwise identical only for +0.0 against -0.0, where the
    // old code's answer was whatever std::sort happened to leave first —
    // unspecified. Outside the schedule's range either way.)
    if (audio_t < video_t) {
      out.unique = {audio_t, video_t};
      video_index = 1;
    } else if (video_t < audio_t) {
      out.unique = {video_t, audio_t};
      audio_index = 1;
    } else {
      out.unique = {video_t};
    }
  } else if (has_video) {
    out.unique = {video_t};
  } else if (has_audio) {
    out.unique = {audio_t};
  }

  // `max(tag, 0)` clamps padding rows (tag = -1) so they cannot index
  // backwards into the modulation table.
  out.indices.assign(static_cast<size_t>(total), video_index);
  out.adaln.resize(static_cast<size_t>(total));
  const int32_t video_base = video_index * 3;
  for (int s = 0; s < total; ++s) {
    out.adaln[static_cast<size_t>(s)] = video_base + std::max(idx.tags[static_cast<size_t>(s)], 0);
  }

  const int32_t audio_base = audio_index * 3;
  for (size_t i = audio_begin; i < idx.audio.size(); ++i) {
    const size_t s = static_cast<size_t>(idx.audio[i]);
    out.indices[s] = audio_index;
    out.adaln[s] = audio_base + std::max(idx.tags[s], 0);
  }
  return out;
}

BandedKeyRanges build_banded_key_ranges(const SequenceLayout& layout, int band_frames,
                                        int query_tile, int key_align) {
  if (query_tile <= 0 || key_align <= 0) {
    throw std::runtime_error("banded attention: query_tile and key_align must be positive");
  }
  const int S = layout.total_rows();
  const int R = layout.rows_per_frame();
  const int F = layout.num_latent_frames;
  const int video_start = layout.video_start();

  BandedKeyRanges out;
  out.query_tile = query_tile;
  out.num_query_tiles = (S + query_tile - 1) / query_tile;
  out.ranges.assign(static_cast<size_t>(out.num_query_tiles) * 4, 0);

  const auto round_down = [key_align](int x) { return x / key_align * key_align; };
  const auto round_up = [key_align](int x) { return (x + key_align - 1) / key_align * key_align; };
  const int seq_end = round_up(S);

  for (int t = 0; t < out.num_query_tiles; ++t) {
    const int q0 = t * query_tile;
    const int q_last = std::min(q0 + query_tile, S) - 1;
    int lo0 = 0, hi0 = 0, lo1 = 0, hi1 = 0;

    // A tile is banded only if every row in it is video. `q0 < video_start`
    // catches both the pure text/audio tiles and the one that straddles the
    // boundary, and gives them global attention -- which is what the
    // conditioning rows need and what makes the frame arithmetic below safe,
    // since `q0 - video_start` is then never negative.
    if (band_frames <= 0 || R <= 0 || F <= 0 || q0 < video_start) {
      lo0 = 0;
      hi0 = seq_end;
    } else {
      const int f_first = (q0 - video_start) / R;
      const int f_last = (q_last - video_start) / R;
      const int f_lo = std::max(0, f_first - band_frames);
      const int f_hi = std::min(F, f_last + band_frames + 1);

      lo0 = 0;
      hi0 = round_up(video_start);  // text + audio: the conditioning, always
      lo1 = round_down(video_start + f_lo * R);
      hi1 = std::min(seq_end, round_up(video_start + f_hi * R));
      // Touching or overlapping ranges become one, so the kernel never stages a
      // key block twice -- which would double-count it in the softmax sum.
      if (lo1 <= hi0) {
        hi0 = std::max(hi0, hi1);
        lo1 = hi1 = 0;
      }
    }
    out.ranges[static_cast<size_t>(t) * 4 + 0] = lo0;
    out.ranges[static_cast<size_t>(t) * 4 + 1] = hi0;
    out.ranges[static_cast<size_t>(t) * 4 + 2] = lo1;
    out.ranges[static_cast<size_t>(t) * 4 + 3] = hi1;
  }
  return out;
}

}  // namespace vidfab::dit
