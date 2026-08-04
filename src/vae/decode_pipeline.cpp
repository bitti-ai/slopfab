// Host-side orchestration around the ViT decoder: latent de-normalisation,
// temporal chunking, spatial tiling, cross-fade stitching, and the final
// ImageNet de-normalisation. See docs/vae_decoder_spec.md sections 1.3-1.6.

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "vidfab/vae/vit_decoder.h"

namespace vidfab::vae {
namespace {

// ImageNet statistics used by the reference VAEProcessor (normalize.py:9-10).
constexpr float kImagenetMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kImagenetStd[3] = {0.229f, 0.224f, 0.225f};

struct TileLayout {
  std::vector<int> starts;    // tile start position in pixels
  std::vector<int> extents;   // tile length in pixels
  std::vector<int> overlaps;  // overlap between tile i and i+1, size = N-1
};

// Mirrors split_tiles(..., is_decoder=True) (klvae.py:192-218). Positions are
// computed in pixel space; the caller divides by vae_ratio to slice latents.
TileLayout split_tiles(int input_len, int tile_size, int overlap_min, int ratio) {
  TileLayout layout;
  if (tile_size >= input_len) {
    layout.starts.push_back(0);
    layout.extents.push_back(input_len);
    return layout;
  }

  int n = (input_len + tile_size - 1) / tile_size;
  while (tile_size * n - overlap_min * (n - 1) - input_len < 0) ++n;

  std::vector<int> overlaps(static_cast<size_t>(n - 1), overlap_min);
  int surplus = tile_size * n - overlap_min * (n - 1) - input_len;
  // The surplus is absorbed by widening overlaps in whole latent units,
  // round-robin, so every tile boundary stays aligned to the latent grid.
  for (int i = 0; surplus > 0; i = (i + 1) % (n - 1)) {
    const int bump = std::min(surplus, ratio);
    overlaps[static_cast<size_t>(i)] += bump;
    surplus -= bump;
  }

  int pos = 0;
  for (int i = 0; i < n; ++i) {
    layout.starts.push_back(pos);
    layout.extents.push_back(tile_size);
    if (i < n - 1) pos += tile_size - overlaps[static_cast<size_t>(i)];
  }
  layout.overlaps = std::move(overlaps);
  return layout;
}

// blend(a, b, overlap) (klvae.py:220-250). The result has b's shape: the first
// `overlap` slices cross-fade from a to b, the rest is b verbatim.
//
// The ramp is asymmetric — weight_b runs 0, 1/n, ... (n-1)/n and never reaches
// 1 — so the last blended slice retains a 1/n contribution from `a`.
void blend_axis(const float* a, const float* b, float* out, int lead, int overlap, int blend_len,
                int trail) {
  // Layout is [lead][blend_len][trail]; `a` supplies its final `overlap`
  // slices along the blended axis.
  for (int l = 0; l < lead; ++l) {
    for (int i = 0; i < blend_len; ++i) {
      const float wb = (i < overlap) ? static_cast<float>(i) / static_cast<float>(overlap) : 1.0f;
      const float wa = 1.0f - wb;
      for (int t = 0; t < trail; ++t) {
        const size_t bi = (static_cast<size_t>(l) * blend_len + i) * trail + t;
        if (i < overlap) {
          const int a_index = blend_len - overlap + i;  // a's tail slice
          const size_t ai = (static_cast<size_t>(l) * blend_len + a_index) * trail + t;
          out[bi] = a[ai] * wa + b[bi] * wb;
        } else {
          out[bi] = b[bi];
        }
      }
    }
  }
}

}  // namespace

DecodedVideo ViTDecoder::decode(const float* z_norm, int T_lat, int H_lat, int W_lat,
                                const std::vector<float>& latents_mean,
                                const std::vector<float>& latents_std,
                                const DecodeSchedule& schedule) {
  const ViTConfig& cfg = config();
  const int ch = cfg.in_channels;
  if (static_cast<int>(latents_mean.size()) != ch ||
      static_cast<int>(latents_std.size()) != ch) {
    throw std::runtime_error("vae: latents_mean/std must have " + std::to_string(ch) +
                             " entries");
  }
  if (T_lat <= 0 || H_lat <= 0 || W_lat <= 0) {
    throw std::runtime_error("vae: latent extents must be positive");
  }

  const size_t voxels_per_frame = static_cast<size_t>(H_lat) * W_lat;

  // (1) De-normalise: z = z_norm * std + mean, per channel. Done in fp32 from
  // the config literals rather than the fp16 tensors in the checkpoint.
  std::vector<float> z(static_cast<size_t>(ch) * T_lat * voxels_per_frame);
  for (int c = 0; c < ch; ++c) {
    const float m = latents_mean[static_cast<size_t>(c)];
    const float s = latents_std[static_cast<size_t>(c)];
    const size_t base = static_cast<size_t>(c) * T_lat * voxels_per_frame;
    for (size_t i = 0; i < static_cast<size_t>(T_lat) * voxels_per_frame; ++i) {
      z[base + i] = z_norm[base + i] * s + m;
    }
  }

  // (2) Temporal padding: repeat the final latent frame until the pseudo token
  // count is a multiple of tokens_chunk_size.
  const int chunk = schedule.tokens_chunk_size;
  const int window = schedule.tokens_per_window();
  int pseudo_total = T_lat + schedule.frame_pre_padding;
  const int remainder = pseudo_total % chunk;
  const int pad_tokens = (remainder != 0) ? (chunk - remainder) : 0;

  int T_padded = T_lat;
  if (pad_tokens > 0) {
    std::vector<float> padded(static_cast<size_t>(ch) * (T_lat + pad_tokens) * voxels_per_frame);
    for (int c = 0; c < ch; ++c) {
      const size_t src_base = static_cast<size_t>(c) * T_lat * voxels_per_frame;
      const size_t dst_base = static_cast<size_t>(c) * (T_lat + pad_tokens) * voxels_per_frame;
      std::copy_n(z.begin() + static_cast<long long>(src_base),
                  static_cast<size_t>(T_lat) * voxels_per_frame,
                  padded.begin() + static_cast<long long>(dst_base));
      const size_t last = src_base + static_cast<size_t>(T_lat - 1) * voxels_per_frame;
      for (int p = 0; p < pad_tokens; ++p) {
        std::copy_n(z.begin() + static_cast<long long>(last), voxels_per_frame,
                    padded.begin() +
                        static_cast<long long>(dst_base +
                                               static_cast<size_t>(T_lat + p) * voxels_per_frame));
      }
    }
    z = std::move(padded);
    T_padded = T_lat + pad_tokens;
    pseudo_total += pad_tokens;
  }

  const int num_chunks = pseudo_total / chunk - 1;
  if (num_chunks <= 0) {
    throw std::runtime_error("vae: latent is too short to decode (need at least " +
                             std::to_string(2 * chunk - schedule.frame_pre_padding) +
                             " latent frames)");
  }

  const int H_px = H_lat * cfg.patch;
  const int W_px = W_lat * cfg.patch;
  const int frames_per_chunk = schedule.chunk_dec - schedule.frame_pre_padding;  // 17
  const size_t frame_pixels = static_cast<size_t>(H_px) * W_px;

  DecodedVideo video;
  video.height = H_px;
  video.width = W_px;
  video.frames = num_chunks * frames_per_chunk + schedule.frame_overlap;

  // Assembled as [frames][3][H][W] first, then transposed to planar at the end.
  std::vector<float> assembled(static_cast<size_t>(video.frames) * 3 * frame_pixels);
  std::vector<float> carry;  // trailing overlap frames from the previous chunk
  int written = 0;

  const TileLayout ytiles =
      schedule.tiling_enabled
          ? split_tiles(H_px, schedule.tile_size, schedule.tile_overlap_min, cfg.patch)
          : split_tiles(H_px, H_px, 0, cfg.patch);
  const TileLayout xtiles =
      schedule.tiling_enabled
          ? split_tiles(W_px, schedule.tile_size, schedule.tile_overlap_min, cfg.patch)
          : split_tiles(W_px, W_px, 0, cfg.patch);

  std::vector<float> clip(static_cast<size_t>(ch) * window * voxels_per_frame);

  // Tile buffers are hoisted out of the chunk loop and reused: moving out of
  // them each chunk would leave them empty, so forward_window's resize() would
  // reallocate and zero-fill the whole output on every single call.
  std::vector<std::vector<float>> tiles(ytiles.starts.size() * xtiles.starts.size());

  for (int c = 0; c < num_chunks; ++c) {
    const int t_start = c * chunk;

    // Gather this chunk's latent window.
    for (int ci = 0; ci < ch; ++ci) {
      const size_t src = static_cast<size_t>(ci) * T_padded * voxels_per_frame +
                         static_cast<size_t>(t_start) * voxels_per_frame;
      const size_t dst = static_cast<size_t>(ci) * window * voxels_per_frame;
      std::copy_n(z.begin() + static_cast<long long>(src),
                  static_cast<size_t>(window) * voxels_per_frame,
                  clip.begin() + static_cast<long long>(dst));
    }

    const int out_frames = window * cfg.patch_t;  // 28
    std::vector<float> chunk_pixels(static_cast<size_t>(3) * out_frames * frame_pixels);

    // Spatial tiling. Tiles are decoded independently, then blended against
    // their raw (unblended) neighbours and trimmed.
    std::vector<int> tile_h(ytiles.starts.size());
    std::vector<int> tile_w(xtiles.starts.size());

    for (size_t ti = 0; ti < ytiles.starts.size(); ++ti) {
      for (size_t tj = 0; tj < xtiles.starts.size(); ++tj) {
        const int y0 = ytiles.starts[ti] / cfg.patch;
        const int x0 = xtiles.starts[tj] / cfg.patch;
        const int th = std::min(ytiles.extents[ti], H_px - ytiles.starts[ti]) / cfg.patch;
        const int tw = std::min(xtiles.extents[tj], W_px - xtiles.starts[tj]) / cfg.patch;
        tile_h[ti] = th * cfg.patch;
        tile_w[tj] = tw * cfg.patch;

        std::vector<float> z_tile(static_cast<size_t>(ch) * window * th * tw);
        for (int ci = 0; ci < ch; ++ci) {
          for (int t = 0; t < window; ++t) {
            for (int y = 0; y < th; ++y) {
              const size_t src = ((static_cast<size_t>(ci) * window + t) * H_lat + (y0 + y)) *
                                     W_lat + x0;
              const size_t dst = ((static_cast<size_t>(ci) * window + t) * th + y) * tw;
              std::copy_n(clip.begin() + static_cast<long long>(src), tw,
                          z_tile.begin() + static_cast<long long>(dst));
            }
          }
        }

        forward_window(z_tile.data(), window, th, tw, tiles[ti * xtiles.starts.size() + tj]);
      }
    }

    // Merge tiles into the chunk's pixel buffer.
    int y_cursor = 0;
    for (size_t ti = 0; ti < ytiles.starts.size(); ++ti) {
      const int th = tile_h[ti];
      const int keep_h =
          (ti + 1 < ytiles.starts.size()) ? th - ytiles.overlaps[ti] : th;
      int x_cursor = 0;
      for (size_t tj = 0; tj < xtiles.starts.size(); ++tj) {
        const int tw = tile_w[tj];
        const int keep_w =
            (tj + 1 < xtiles.starts.size()) ? tw - xtiles.overlaps[tj] : tw;

        std::vector<float> tile = tiles[ti * xtiles.starts.size() + tj];
        const size_t plane = static_cast<size_t>(th) * tw;

        if (ti > 0) {
          const std::vector<float>& above = tiles[(ti - 1) * xtiles.starts.size() + tj];
          std::vector<float> merged(tile.size());
          blend_axis(above.data(), tile.data(), merged.data(), 3 * out_frames,
                     ytiles.overlaps[ti - 1], th, tw);
          tile = std::move(merged);
        }
        if (tj > 0) {
          const std::vector<float>& left = tiles[ti * xtiles.starts.size() + (tj - 1)];
          std::vector<float> merged(tile.size());
          blend_axis(left.data(), tile.data(), merged.data(), 3 * out_frames * th,
                     xtiles.overlaps[tj - 1], tw, 1);
          tile = std::move(merged);
        }

        for (int p = 0; p < 3 * out_frames; ++p) {
          for (int y = 0; y < keep_h; ++y) {
            const size_t src = static_cast<size_t>(p) * plane + static_cast<size_t>(y) * tw;
            const int plane_index = p % out_frames;
            const int channel = p / out_frames;
            const size_t dst =
                (static_cast<size_t>(plane_index) * 3 + channel) * frame_pixels +
                static_cast<size_t>(y_cursor + y) * W_px + x_cursor;
            std::copy_n(tile.begin() + static_cast<long long>(src), keep_w,
                        chunk_pixels.begin() + static_cast<long long>(dst));
          }
        }
        x_cursor += keep_w;
      }
      y_cursor += keep_h;
    }

    // Split the 28 decoded frames: [3:20] is the primary block, [23:28] is
    // carried into the next chunk as overlap.
    const int pre = schedule.frame_pre_padding;
    const int overlap = schedule.frame_overlap;
    std::vector<float> primary(static_cast<size_t>(frames_per_chunk) * 3 * frame_pixels);
    std::copy_n(chunk_pixels.begin() + static_cast<long long>(pre * 3 * frame_pixels),
                static_cast<size_t>(frames_per_chunk) * 3 * frame_pixels, primary.begin());

    std::vector<float> next_carry(static_cast<size_t>(overlap) * 3 * frame_pixels);
    std::copy_n(chunk_pixels.begin() +
                    static_cast<long long>((schedule.chunk_dec + pre) * 3 * frame_pixels),
                static_cast<size_t>(overlap) * 3 * frame_pixels, next_carry.begin());

    // Cross-fade the leading frames against the previous chunk's carry.
    if (!carry.empty()) {
      for (int f = 0; f < overlap; ++f) {
        const float wb = static_cast<float>(f) / static_cast<float>(overlap);
        const float wa = 1.0f - wb;
        for (size_t i = 0; i < 3 * frame_pixels; ++i) {
          const size_t pi = static_cast<size_t>(f) * 3 * frame_pixels + i;
          primary[pi] = carry[pi] * wa + primary[pi] * wb;
        }
      }
    }

    std::copy_n(primary.begin(), primary.size(),
                assembled.begin() + static_cast<long long>(static_cast<size_t>(written) * 3 *
                                                           frame_pixels));
    written += frames_per_chunk;
    carry = std::move(next_carry);
  }

  // The final carry is appended verbatim.
  std::copy_n(carry.begin(), carry.size(),
              assembled.begin() +
                  static_cast<long long>(static_cast<size_t>(written) * 3 * frame_pixels));
  written += schedule.frame_overlap;

  // Drop frames that came only from repeated padding tokens.
  int pad_frames = 0;
  for (int k = 0; k < pad_tokens; ++k) {
    pad_frames += ((T_lat + k) % chunk == 0) ? 1 : cfg.patch_t;
  }
  const int final_frames = std::max(0, written - pad_frames);
  video.frames = final_frames;

  // (8) Pixel de-normalisation, then transpose to planar [3][T][H][W].
  video.data.assign(static_cast<size_t>(3) * final_frames * frame_pixels, 0.0f);
  for (int f = 0; f < final_frames; ++f) {
    for (int c = 0; c < 3; ++c) {
      const float m = kImagenetMean[c];
      const float s = kImagenetStd[c];
      const size_t src = (static_cast<size_t>(f) * 3 + c) * frame_pixels;
      const size_t dst = (static_cast<size_t>(c) * final_frames + f) * frame_pixels;
      for (size_t i = 0; i < frame_pixels; ++i) {
        video.data[dst + i] = std::min(1.0f, std::max(0.0f, assembled[src + i] * s + m));
      }
    }
  }
  return video;
}

}  // namespace vidfab::vae
