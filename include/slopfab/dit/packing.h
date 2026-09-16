// Token packing for the H3 omni transformer.
//
// One request is one packed sequence. There are no separator tokens, no
// BOS/EOS, no learned modality embeddings and no padding. Modality is carried
// entirely by which input projection wrote the row, the per-row token tag, and
// the rotary coordinate.
//
//   [ text (L) | conditions (C=0 for t2va) | audio (Sa = 2A) | video (V = F*R) ]
//
// This header is host-side index arithmetic only — no CUDA. It is unit-tested
// on the host against the worked examples in docs/transformer_spec.md sections
// 1.1 and 2.3, which is the cheapest possible guard against the class of bug
// that scrambles the spatial RoPE without changing any shape.
#pragma once

#include <cstdint>
#include <vector>

namespace slopfab::dit {

// Modality tags. The values are the modality axis of the AdaLN table and must
// not be renumbered.
enum : int32_t {
  kTagVideo = 0,
  kTagText = 1,
  kTagAudio = 2,
};

struct SequenceLayout {
  int num_text = 0;             // L
  int num_condition_video = 0;  // C, always 0 for t2va
  int num_condition_audio = 0;  // reference audio rows, 0 for t2va/fl2va
  // False preserves the legacy fl2va convention where num_condition_video is
  // also the prefix of idx.audio kept at video_t. Ref2VA sets this true even
  // when it has zero audio anchors (for example an image-only request).
  bool condition_audio_is_explicit = false;
  int num_audio_rows = 0;       // Sa = 2 * num_audio_latents
  int num_video_rows = 0;       // V = F * R
  int num_audio_latents = 0;    // A, per channel
  int num_latent_frames = 0;    // F
  int latent_height = 0;        // Hl
  int latent_width = 0;         // Wl

  int rows_per_frame() const { return (latent_height / 2) * (latent_width / 2); }  // R
  int condition_start() const { return num_text; }
  int audio_start() const { return num_text + num_condition_video + num_condition_audio; }
  int video_start() const { return audio_start() + num_audio_rows; }
  int total_rows() const { return video_start() + num_video_rows; }  // S
};

// Canvas resolution from a display aspect ratio. Only the ratio of the two
// arguments matters: the short edge is fixed at 768, the area is capped at
// 768*1344, and both axes are then rounded to the nearest multiple of 32 — so
// the final area can land slightly *above* the pre-rounding budget. Throws for
// ratios outside 1:4 .. 4:1.
void resolve_canvas_size(double aspect_w, double aspect_h, int* out_h, int* out_w,
                         int short_edge = 768, int max_pixels = 768 * 1344);

// Checks a canvas the caller chose outright, rather than deriving one from a
// ratio. Enforces what the pipeline cannot work without — positive axes, both a
// multiple of 32, and a ratio inside 1:4 .. 4:1 — and nothing else. In
// particular the area is **not** capped: `resolve_canvas_size` scales a request
// down to the trained budget because the caller only asked for a shape, whereas
// a caller naming 1920x1088 has asked for that canvas and gets it. The cost of
// exceeding the budget is quadratic in area and belongs to whoever typed it, so
// `canvas_exceeds_trained_area` exists to warn rather than to refuse.
void validate_canvas_size(int height, int width);

// True when a canvas is larger than the 768*1344 the released model was trained
// at. Packed rows grow with area and attention grows with their square, so this
// is the difference between a slow run and an unusable one.
bool canvas_exceeds_trained_area(int height, int width);

// Snaps a frame count **up** to the next `17*k + 5` the video VAE can encode.
int align_num_frames(int num_frames);

// Latent frames for an aligned pixel-frame count: F = 5*k + 2 for 17*k + 5.
// Throws if `aligned_frames % 17 != 5`.
int video_latent_num_frames(int aligned_frames);

// Audio latents per channel: 40 per second at 24 fps.
int audio_latents_for_frames(int aligned_frames);

struct PackedIndices {
  std::vector<int32_t> text;   // [0, L)
  std::vector<int32_t> audio;  // [audio_start, video_start)
  std::vector<int32_t> video;  // conditions, then targets
  std::vector<int32_t> tags;   // [S], one of kTag*
};

PackedIndices build_indices(const SequenceLayout& layout);

// Rotary coordinates, `[S, 3]` in (t, h, w) order.
//
// **Built in float64 and kept in float64 here.** The reference builds them in
// float64 and casts to float32 inside rope.forward; the cast happens there,
// not here. The spatial grids use numpy's linspace formula
// `start + arange(n)*(stop-start)/n`, which is not torch.linspace, and the
// temporal grid is a cumulative sum over (5/3)*FRAMES_PER_LATENT with
// FRAMES_PER_LATENT = (1,4,4,4,4). Spec section 2.3.
std::vector<double> build_position_ids(const SequenceLayout& layout);

// Patchify: `(24, F, Hl, Wl)` fp32 -> `(V, 96)` rows.
//
//   row     = (f * (Hl/2) + hh) * (Wl/2) + ww
//   feature = c * 4 + dh * 2 + dw
//   value   = latents[c, f, 2*hh + dh, 2*ww + dw]
//
// Frame-major then latent-row-major, channel-major within the 2x2 patch. The
// same `hh*(Wl/2) + ww` ordering as the rotary spatial grid — a mismatch
// between the two silently scrambles the spatial RoPE.
void patchify_video(const float* latents, const SequenceLayout& layout, float* rows_out);
void unpatchify_video(const float* rows, const SequenceLayout& layout, float* latents_out);

// Audio: `(Sa, 32)` rows -> `(2, 32, A)`, channel-major.
void unpack_audio(const float* rows, int num_audio_latents, float* out);

// Per-row timesteps for one denoising step, then the ascending unique and the
// per-row index into it.
//
// `torch.unique(sorted=True)` sorts ascending, so which of the video and audio
// timesteps is index 0 flips over the schedule. That feeds the AdaLN index, so
// it must be reproduced rather than fixed. Spec section 7.5.
struct RowTimesteps {
  std::vector<float> unique;     // ascending, normally 2 entries for t2va
  std::vector<int32_t> indices;  // [S], into `unique`
  std::vector<int32_t> adaln;    // [S], = indices[s]*3 + max(tag[s], 0)
};

RowTimesteps build_row_timesteps(const SequenceLayout& layout, const PackedIndices& idx,
                                 float video_t, float audio_t);
RowTimesteps build_row_timesteps(const SequenceLayout& layout, const PackedIndices& idx,
                                 float video_t, float audio_t, float condition_video_t,
                                 float condition_audio_t);

// --- frame-banded attention -------------------------------------------------
//
// Which keys one query tile may attend to when the band is on. Host-side index
// arithmetic, computed once per request; the kernel reads it and changes only
// its loop bounds.
//
// **The band is frame-granular, not a window over the packed row index.** Video
// packs frame-major, so a latent frame is a contiguous run of `R` rows (1008 at
// the default geometry) and a naive band over row index would cut *within* a
// frame -- a horizontal strip of one frame plus a horizontal strip of the next.
// That is an anisotropic spatial prior that would show up as a banding artefact
// and be blamed on the method rather than on the indexing. Frame-granular means
// every query attends to whole frames.
//
// **Two ranges per tile, not one, and the second is the reason.** The packed
// sequence is `[ text | conditions | audio | video ]`, so text and audio sit at
// the *front*. A single [lo, hi) window around a late video frame would exclude
// them and cut every video row off from the prompt -- the conditioning path.
// So a video tile gets the text/audio prefix **plus** its band. That prefix is
// 431 of 37727 rows at the default geometry, 1.14%, so always including it costs
// essentially nothing and removing it would not be a speedup, it would be a
// different model.
//
// Tiles containing any text or audio row attend globally: they are a tiny
// fraction of S, banding them buys nothing, and they are the conditioning.
//
// Ranges are rounded outwards to `key_align` so the kernel's key loop steps
// over whole blocks. Rounding out rather than in is deliberate -- it attends to
// at most `key_align - 1` extra rows per edge, which is conservative (closer to
// full attention), never lossy, and deterministic.
struct BandedKeyRanges {
  int query_tile = 0;
  int num_query_tiles = 0;
  // Two half-open [lo, hi) ranges per tile, flattened as
  // `[t*4+0] = lo0, [t*4+1] = hi0, [t*4+2] = lo1, [t*4+3] = hi1`.
  // The second is empty (lo1 == hi1 == 0) when the two would touch and have
  // been merged, and when the tile attends globally.
  std::vector<int32_t> ranges;

  // Keys this tile attends to, for the cost model and the tests.
  int keys_for_tile(int t) const {
    return (ranges[size_t(t) * 4 + 1] - ranges[size_t(t) * 4 + 0]) +
           (ranges[size_t(t) * 4 + 3] - ranges[size_t(t) * 4 + 2]);
  }
};

// `band_frames` is a half-width: a query in latent frame f attends to frames
// [f - band_frames, f + band_frames], so 2*band_frames + 1 frames in all.
// 0 disables banding and every tile attends globally.
BandedKeyRanges build_banded_key_ranges(const SequenceLayout& layout, int band_frames,
                                        int query_tile, int key_align);

}  // namespace slopfab::dit
