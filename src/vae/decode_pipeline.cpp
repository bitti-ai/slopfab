// Host-side orchestration around the ViT decoder: latent de-normalisation,
// temporal chunking, spatial tiling, cross-fade stitching, and the final
// ImageNet de-normalisation. See docs/vae_decoder_spec.md sections 1.3-1.6.

#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "vidfab/cuda/profile.h"
#include "vidfab/vae/tile_merge.h"
#include "vidfab/vae/vit_decoder.h"

namespace vidfab::vae {

const std::vector<float>& default_video_latents_mean() {
  static const std::vector<float> values = {
      .8580903411f, -.9606591463f, 1.0661640167f, -.5090325475f, -.2727581859f,
      -1.3675414324f, -.2553254962f, -.2690755427f, -.5376840830f, -.0464097299f,
      .6657370329f, .1969012767f, -.5460608006f, -.4035342038f, -.2368302494f,
      .2592845261f, -.3013394475f, .2113419920f, -1.1206848621f, .3581933379f,
      -.0422514379f, .2604829967f, .2286409289f, .7056031823f};
  return values;
}

const std::vector<float>& default_video_latents_std() {
  static const std::vector<float> values = {
      1.2223774195f, 1.2767263651f, 1.6831774712f, 1.7549455166f, 1.5636216402f,
      2.1941435337f, .9653137922f, 1.0569885969f, .8419489264f, .7729952931f,
      1.8955937624f, .9468418360f, .7996809483f, .4498890042f, .7197399735f,
      .6936293244f, 2.9610950947f, 2.7694199085f, 3.0496184826f, 2.1088054180f,
      3.2762262821f, 3.1627357006f, 2.2816812992f, 2.6127843857f};
  return values;
}
namespace {

// ImageNet statistics used by the reference VAEProcessor (normalize.py:9-10).
constexpr float kImagenetMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kImagenetStd[3] = {0.229f, 0.224f, 0.225f};

// blend(a, b, overlap) (klvae.py:220-250) cross-fades the first `overlap`
// slices from a to b and leaves the rest of b alone. The ramp is asymmetric —
// weight_b runs 0, 1/n, ... (n-1)/n and never reaches 1 — so the last blended
// slice retains a 1/n contribution from `a`. TileMerge in vae/tile_merge.h
// applies it to a decoded tile without materialising the untouched interior.

}  // namespace

DecodedVideo decode_video(VideoVaeWindowBackend& backend, const float* z_norm,
                          int T_lat, int H_lat, int W_lat,
                          const std::vector<float>& latents_mean,
                          const std::vector<float>& latents_std,
                          const DecodeSchedule& schedule) {
  const ViTConfig& cfg = backend.config();
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

  cuda::PhaseSpan s_denorm("latent de-normalise");

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

  s_denorm.stop();

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
  // Deliberately not a vector: every element is written before it is read —
  // the chunks' primary blocks tile it exactly and the final carry fills the
  // tail — so value-initialising it would be a gigabytes-wide memset of values
  // nothing ever looks at.
  cuda::PhaseSpan s_alloc("alloc assembled");
  const std::unique_ptr<float[]> assembled(
      new float[static_cast<size_t>(video.frames) * 3 * frame_pixels]);
  s_alloc.stop();
  // Trailing overlap frames from the previous chunk. Allocated up front and
  // swapped with `next_carry` each chunk, so `have_carry` rather than
  // emptiness is what says whether there is a previous chunk to fade against.
  std::vector<float> carry(static_cast<size_t>(schedule.frame_overlap) * 3 * frame_pixels);
  bool have_carry = false;
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
  // reallocate and zero-fill the whole output on every single call. The decoder
  // writes into these slots directly for the same reason.
  std::vector<std::vector<float>> tiles(ytiles.starts.size() * xtiles.starts.size());

  // forward_windows page-locks those slots so each tile is DMA'd into its final
  // home. Declared *after* `tiles` so it is destroyed *before* it: the locks
  // must go while the memory they cover is still alive, on the throwing path as
  // much as the normal one.
  struct RegistrationScope {
    VideoVaeWindowBackend* backend;
    ~RegistrationScope() { backend->release_host_registrations(); }
  } registration_scope{&backend};

  // Every per-chunk working buffer is hoisted for the same reason as `tiles`:
  // each is written in full before it is read, so a fresh allocation per chunk
  // would only buy a zero-fill of a few hundred megabytes that the next line
  // overwrites. `merge` holds the two overlap slabs, which is all the
  // cross-fade ever changes; the raw neighbour tiles are read straight out of
  // `tiles`.
  const int out_frames = window * cfg.patch_t;  // 28
  TileMerge merge;
  std::vector<float> z_batch;
  std::vector<float> next_carry(static_cast<size_t>(schedule.frame_overlap) * 3 * frame_pixels);
  // Where each of the chunk's 28 decoded frames belongs, rebuilt per chunk
  // because `primary` advances and `next_carry` is swapped. Six of the frames
  // belong nowhere and are marked null; see the comment where it is filled.
  std::vector<float*> frame_dst(static_cast<size_t>(out_frames), nullptr);

  // Spatial tiling. Tiles are decoded independently, then blended against
  // their raw (unblended) neighbours and trimmed. Every input here — the two
  // tile layouts, the pixel extents and the patch size — is fixed for the whole
  // decode, so the geometry and the batching are resolved once rather than
  // rebuilt, std::map and all, on each of the chunks.
  std::vector<int> tile_h;
  std::vector<int> tile_w;
  std::map<std::pair<int, int>, std::vector<size_t>> shape_groups;
  tile_shape_groups(ytiles, xtiles, H_px, W_px, cfg.patch, &tile_h, &tile_w, &shape_groups);

  for (int c = 0; c < num_chunks; ++c) {
    const int t_start = c * chunk;

    // Gather this chunk's latent window.
    cuda::PhaseSpan s_clip("chunk latent gather");
    for (int ci = 0; ci < ch; ++ci) {
      const size_t src = static_cast<size_t>(ci) * T_padded * voxels_per_frame +
                         static_cast<size_t>(t_start) * voxels_per_frame;
      const size_t dst = static_cast<size_t>(ci) * window * voxels_per_frame;
      std::copy_n(z.begin() + static_cast<long long>(src),
                  static_cast<size_t>(window) * voxels_per_frame,
                  clip.begin() + static_cast<long long>(dst));
    }

    s_clip.stop();

    // Equal-shape tiles share the weight-heavy token projections. Ragged edge
    // shapes form their own batches so attention geometry and stitching stay
    // exactly the same as independent forward_window calls.
    for (const auto& [shape, ids] : shape_groups) {
      const int th = shape.first;
      const int tw = shape.second;
      const size_t tile_voxels = static_cast<size_t>(window) * th * tw;
      cuda::PhaseSpan s_gather("tile latent gather");
      // Grown, never value-initialised: the gather below writes every element
      // of [0, needed) — the loops cover every (batch item, channel, t, y) and
      // each writes a whole row — so the zero-fill a fresh vector performs is
      // 14 MiB of stores per shape group per chunk that the next loop
      // overwrites in full.
      const size_t needed = static_cast<size_t>(ids.size()) * ch * tile_voxels;
      if (z_batch.size() < needed) z_batch.resize(needed);
      for (size_t bi = 0; bi < ids.size(); ++bi) {
        const size_t id = ids[bi];
        const size_t ti = id / xtiles.starts.size();
        const size_t tj = id % xtiles.starts.size();
        const int y0 = ytiles.starts[ti] / cfg.patch;
        const int x0 = xtiles.starts[tj] / cfg.patch;
        for (int ci = 0; ci < ch; ++ci) {
          for (int t = 0; t < window; ++t) {
            for (int y = 0; y < th; ++y) {
              const size_t src = ((static_cast<size_t>(ci) * window + t) * H_lat + (y0 + y)) *
                                     W_lat + x0;
              const size_t dst = (bi * ch + ci) * tile_voxels +
                                 (static_cast<size_t>(t) * th + y) * tw;
              std::copy_n(clip.begin() + static_cast<long long>(src), tw,
                          z_batch.begin() + static_cast<long long>(dst));
            }
          }
        }
      }
      s_gather.stop();
      // Decoded straight into the hoisted slots, so each tile lands in the
      // buffer it used last chunk and its resize is a no-op.
      backend.forward_windows(z_batch.data(), static_cast<int>(ids.size()),
                              window, th, tw, tiles, ids.data());
    }

    // Where the chunk's 28 decoded frames go. This removes the 330 MiB staging
    // buffer the stitch used to fill and the two copies that emptied it, and
    // skips the six frames nothing downstream reads. The cross-fade below then
    // mutates `assembled` in place, which is what it was always trying to
    // express.
    const int pre = schedule.frame_pre_padding;
    const int overlap = schedule.frame_overlap;
    float* primary = assembled.get() + static_cast<size_t>(written) * 3 * frame_pixels;
    chunk_frame_destinations(out_frames, pre, frames_per_chunk, schedule.chunk_dec, overlap,
                             3 * frame_pixels, primary, next_carry.data(), &frame_dst);

    // Merge tiles into their destination frames.
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

        const std::vector<float>& raw = tiles[ti * xtiles.starts.size() + tj];

        // Only the overlap slabs are computed; with no neighbour on either axis
        // nothing is computed at all and the stitch reads the raw tile.
        cuda::PhaseSpan s_blend("tile blend");
        const float* above =
            (ti > 0) ? tiles[(ti - 1) * xtiles.starts.size() + tj].data() : nullptr;
        const float* left =
            (tj > 0) ? tiles[ti * xtiles.starts.size() + (tj - 1)].data() : nullptr;
        merge.prepare(raw.data(), above, left, 3 * out_frames, th, tw,
                      (ti > 0) ? ytiles.overlaps[ti - 1] : 0,
                      (tj > 0) ? xtiles.overlaps[tj - 1] : 0);
        s_blend.stop();

        cuda::PhaseSpan s_stitch("tile stitch");
        for (int p = 0; p < 3 * out_frames; ++p) {
          float* base = frame_dst[static_cast<size_t>(p % out_frames)];
          if (base == nullptr) continue;  // a frame nothing downstream reads
          const int channel = p / out_frames;
          float* row0 = base + static_cast<size_t>(channel) * frame_pixels +
                        static_cast<size_t>(y_cursor) * W_px + x_cursor;
          for (int y = 0; y < keep_h; ++y) {
            merge.copy_row(p, y, keep_w, row0 + static_cast<size_t>(y) * W_px);
          }
        }
        s_stitch.stop();
        x_cursor += keep_w;
      }
      y_cursor += keep_h;
    }

    // Cross-fade the leading frames against the previous chunk's carry. The
    // stitch has already put them in `assembled`, so this mutates the final
    // buffer in place.
    cuda::PhaseSpan s_fade("chunk cross-fade");
    if (have_carry) {
      for (int f = 0; f < overlap; ++f) {
        const float wb = static_cast<float>(f) / static_cast<float>(overlap);
        const float wa = 1.0f - wb;
        for (size_t i = 0; i < 3 * frame_pixels; ++i) {
          const size_t pi = static_cast<size_t>(f) * 3 * frame_pixels + i;
          primary[pi] = carry[pi] * wa + primary[pi] * wb;
        }
      }
    }
    s_fade.stop();

    written += frames_per_chunk;
    // Swapped rather than moved, so both buffers keep their allocation.
    carry.swap(next_carry);
    have_carry = true;
  }

  // The final carry is appended verbatim.
  std::copy_n(carry.begin(), carry.size(),
              assembled.get() + static_cast<size_t>(written) * 3 * frame_pixels);
  written += schedule.frame_overlap;

  // Drop frames that came only from repeated padding tokens.
  int pad_frames = 0;
  for (int k = 0; k < pad_tokens; ++k) {
    pad_frames += ((T_lat + k) % chunk == 0) ? 1 : cfg.patch_t;
  }
  const int final_frames = std::max(0, written - pad_frames);
  video.frames = final_frames;

  // (8) Pixel de-normalisation, then transpose to planar [3][T][H][W].
  // resize, not assign(n, 0): PixelBuffer default-initialises, and the loop
  // below writes every element of it. Zeroing first was a full-width memset of
  // the whole decoded video for nothing.
  cuda::PhaseSpan s_alloc_out("alloc output");
  video.data.resize(static_cast<size_t>(3) * final_frames * frame_pixels);
  s_alloc_out.stop();

  // Roughly six gigabytes of traffic at the heavy config for two flops per
  // element, so this is bandwidth and not arithmetic, and one core cannot
  // saturate the memory system. Each (frame, channel) plane reads and writes a
  // region no other plane touches, so the work is split into contiguous
  // *static* ranges of planes: every output element is computed by exactly the
  // same expression from exactly the same inputs as before, in the same order
  // within a plane. Nothing is reduced and nothing is reordered across
  // elements, so the result is bit-identical to the serial loop; a dynamic
  // schedule would be too, but a static one keeps that obvious.
  const size_t planes = static_cast<size_t>(final_frames) * 3;
  cuda::PhaseSpan s_out("pixel de-normalise");
  const auto plane_range = [&](size_t begin, size_t end) {
    for (size_t p = begin; p < end; ++p) {
      const int f = static_cast<int>(p / 3);
      const int c = static_cast<int>(p % 3);
      const float m = kImagenetMean[c];
      const float s = kImagenetStd[c];
      const size_t src = p * frame_pixels;
      const size_t dst = (static_cast<size_t>(c) * final_frames + f) * frame_pixels;
      for (size_t i = 0; i < frame_pixels; ++i) {
        video.data[dst + i] = std::min(1.0f, std::max(0.0f, assembled[src + i] * s + m));
      }
    }
  };

  // Thread count is capped well below the core count on purpose. The pass is
  // DRAM-bandwidth-bound, and the memory system saturates around eight streams
  // on this class of machine, so extra workers buy nothing and SMT siblings
  // actively contend. Spawning is not free either — a create/join pair costs
  // tens of microseconds, so a wide pool is a fixed tax on a phase that at the
  // small end only runs for tens of milliseconds. Below a threshold the tax
  // exceeds the saving outright and the serial path is simply faster.
  constexpr unsigned kMaxWorkers = 8;
  constexpr size_t kMinParallelBytes = 32u << 20;  // 32 MiB of output

  const size_t out_bytes = planes * frame_pixels * sizeof(float);
  unsigned workers = std::thread::hardware_concurrency();
  if (workers == 0) workers = 1;
  workers = std::min(workers, kMaxWorkers);
  workers = static_cast<unsigned>(std::min<size_t>(workers, std::max<size_t>(planes, 1)));
  if (out_bytes < kMinParallelBytes) workers = 1;

  if (workers <= 1) {
    plane_range(0, planes);
  } else {
    // Joins in the destructor as well as on the happy path: if `emplace_back`
    // throws part-way through, a plain vector of threads would run ~thread on
    // still-joinable threads and call std::terminate.
    struct JoiningPool {
      std::vector<std::thread> threads;
      ~JoiningPool() {
        for (std::thread& t : threads) {
          if (t.joinable()) t.join();
        }
      }
    } pool;
    pool.threads.reserve(workers - 1);
    const size_t share = (planes + workers - 1) / workers;
    for (unsigned w = 1; w < workers; ++w) {
      const size_t begin = std::min(planes, share * w);
      const size_t end = std::min(planes, begin + share);
      if (begin == end) break;
      pool.threads.emplace_back(plane_range, begin, end);
    }
    plane_range(0, std::min(planes, share));
  }
  s_out.stop();
  return video;
}

DecodedVideo ViTDecoder::decode(const float* z_norm, int T_lat, int H_lat,
                                int W_lat,
                                const std::vector<float>& latents_mean,
                                const std::vector<float>& latents_std,
                                const DecodeSchedule& schedule) {
  return decode_video(*this, z_norm, T_lat, H_lat, W_lat, latents_mean,
                      latents_std, schedule);
}

}  // namespace vidfab::vae
