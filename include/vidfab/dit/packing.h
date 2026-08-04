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

namespace vidfab::dit {

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
  int num_audio_rows = 0;       // Sa = 2 * num_audio_latents
  int num_video_rows = 0;       // V = F * R
  int num_audio_latents = 0;    // A, per channel
  int num_latent_frames = 0;    // F
  int latent_height = 0;        // Hl
  int latent_width = 0;         // Wl

  int rows_per_frame() const { return (latent_height / 2) * (latent_width / 2); }  // R
  int condition_start() const { return num_text; }
  int audio_start() const { return num_text + num_condition_video; }
  int video_start() const { return audio_start() + num_audio_rows; }
  int total_rows() const { return video_start() + num_video_rows; }  // S
};

// Canvas resolution from a display aspect ratio. Only the ratio of the two
// arguments matters: the short edge is fixed at 768, the area is capped at
// 768*1344, and both axes are then rounded to the nearest multiple of 32 — so
// the final area can land slightly *above* the pre-rounding budget. Throws for
// ratios outside 1:4 .. 4:1.
void resolve_canvas_size(double aspect_w, double aspect_h, int* out_h, int* out_w);

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

}  // namespace vidfab::dit
