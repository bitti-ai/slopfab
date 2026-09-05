// Temporal chunking of one request into several shorter ones.
//
// This exists for a **quality probe**, not for production. It answers a
// question the frame-banded attention work needs answered before its kernel is
// written: does this model tolerate losing distant temporal attention at all?
//
// Chunking is the crude, kernel-free way to ask. A request is split into
// several overlapping runs of the ordinary pipeline and the resulting latents
// are cross-faded back together. That is **strictly more damaging** than a
// band, in three separate ways, and every one of them is deliberate:
//
//   - a band keeps attention within +/-N frames; chunking has *no* cross-chunk
//     attention at all, which is a hard cut rather than a soft window;
//   - each chunk runs its own audio schedule, so audio desynchronises across a
//     boundary in a way a band never causes;
//   - each chunk re-derives its rotary coordinates from its own local frame
//     indices, so the temporal position grid restarts at every boundary.
//
// So a clean result here de-risks the band. A bad one does not condemn it, but
// says the model is sensitive to temporal locality, which is worth knowing.
//
// Two pieces of arithmetic here are silently-wrong-shaped and are what the
// tests point at:
//
//   - **Video rows are frame-major.** Latent frame `f` is the contiguous run of
//     `R = (Hl/2)*(Wl/2)` rows starting at `f*R` (spec 1.4). A chunk boundary
//     must land on a latent-frame boundary; an arbitrary row split would cut a
//     frame in half and no shape check would notice.
//   - **The rotary temporal grid is non-uniform**: `span[j] = (5/3) *
//     FRAMES_PER_LATENT[j % 5]` with `FRAMES_PER_LATENT = (1,4,4,4,4)` (spec
//     2.3). A chunk offset that is *not* a multiple of 5 gives the chunk a
//     different internal spacing from the frames it is standing in for, which
//     would confound "no cross-chunk attention" with "wrong rotary spacing".
//     `resolve_chunk_plan` rejects it rather than measuring it.
#pragma once

#include <cstdint>
#include <vector>

#include "slopfab/dit/packing.h"

namespace slopfab::dit {

// Where each chunk sits inside the full request. Video offsets are in latent
// frames, audio offsets in audio latents per channel.
struct ChunkPlan {
  int num_chunks = 0;

  int full_latent_frames = 0;   // F
  int chunk_latent_frames = 0;  // Fc
  int latent_stride = 0;        // (F - Fc) / (num_chunks - 1), a multiple of 5
  int frame_overlap = 0;        // Fc - latent_stride, in latent frames

  int full_audio_latents = 0;   // A
  int chunk_audio_latents = 0;  // Ac
  int audio_stride = 0;         // (A - Ac) / (num_chunks - 1)
  int audio_overlap = 0;        // Ac - audio_stride

  std::vector<int> frame_offset;  // [num_chunks], in latent frames
  std::vector<int> audio_offset;  // [num_chunks], in audio latents

  // The audio clock runs at 5/3 latents per pixel frame while the video clock
  // advances in the non-uniform (1,4,4,4,4) pattern, so an integer video-frame
  // stride generally does not land on an integer audio-latent stride. This is
  // the residual, in audio latents, for the worst chunk — reported rather than
  // hidden because it is a real (small) contribution to the audio desync.
  double worst_audio_drift_latents = 0.0;
};

// Resolves the placement of `num_chunks` chunks of `chunk` geometry inside a
// `full` request, both already resolved by `resolve_plan`.
//
// Throws with a specific message when the two geometries cannot tile: a
// non-integer or non-positive stride, a stride that is not a multiple of 5, a
// zero or negative overlap, or a spatial canvas that differs between the two.
ChunkPlan resolve_chunk_plan(const SequenceLayout& full, const SequenceLayout& chunk,
                             int num_chunks);

// Chunk `index`'s initial latents, taken as a **slice of the full request's own
// noise draw** rather than as an independent draw.
//
// This is the deliberate choice and it makes the probe *less* harsh than the
// alternative, which is why it is the one taken: a band restricts attention
// over one noise field, so slicing one full-length draw is the closer analogue.
// Independent per-chunk draws would additionally guarantee that no two chunks
// could ever agree, which would measure the noise rather than the attention.
//
// `video_rows_out` is `[Fc*Rc, 96]` and `audio_rows_out` is `[2*Ac, 32]`, i.e.
// exactly what the denoise loop would otherwise have drawn for the chunk.
void slice_chunk_noise(uint64_t seed, const SequenceLayout& full, const SequenceLayout& chunk,
                       const ChunkPlan& plan, int index, std::vector<float>* video_rows_out,
                       std::vector<float>* audio_rows_out);

// Cross-fades the chunks' denoised latents back into one full-geometry pair of
// row buffers.
//
// The window is a normalised trapezoid: a chunk's weight is 1 across its
// interior and ramps as `(j + 0.5) / overlap` into a neighbour, so two chunks
// meeting over an overlap of `n` frames contribute `(0.5, 1.5, ...)/n` and
// `(n-0.5, n-1.5, ...)/n` and the pair sums to exactly 1 at every frame. The
// result is then divided by the accumulated weight, which is 1 everywhere for
// a plan that tiles — the division is there so a plan that does not tile
// degrades to an average rather than to a dark band.
//
// The symmetric `+0.5` offset is deliberately *not* the VAE decoder's ramp
// (`blend_axis`, which uses `i/n` and never reaches 1, reproducing klvae). That
// one puts its whole discontinuity at one end of the overlap. Here the overlap
// can be as short as two frames and the question being asked is where the
// seams are, so a ramp that concentrates the step at one edge would be
// answering with its own artefact.
void blend_chunks(const SequenceLayout& full, const SequenceLayout& chunk, const ChunkPlan& plan,
                  const std::vector<std::vector<float>>& chunk_video,
                  const std::vector<std::vector<float>>& chunk_audio,
                  std::vector<float>* video_rows_out, std::vector<float>* audio_rows_out);

}  // namespace slopfab::dit
