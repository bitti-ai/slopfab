#pragma once
#include <cstdint>
#include <vector>
#include "slopfab/dit/packing.h"
namespace slopfab::dit {
enum class ReferenceKind { kImage, kVideo, kAudio };
struct ReferenceGeometry {
  ReferenceKind kind = ReferenceKind::kImage;
  int num_latent_frames = 1;
  int latent_height = 0;
  int latent_width = 0;
  int num_audio_latents = 0;
  // A final temporal guide shares the target origin instead of advancing it.
  bool target_aligned = false;
  // Temporal guides may sit immediately before/after the generated timeline.
  double target_time_offset = 0;
  int video_rows() const;
  int audio_rows() const;
};
struct Ref2VAPackedSequence {
  SequenceLayout layout;
  PackedIndices indices;
  std::vector<double> position_ids;
};
Ref2VAPackedSequence build_ref2va_packed_sequence(
    const std::vector<int32_t>& text_tags, const std::vector<ReferenceGeometry>& references,
    int num_latent_frames, int latent_height, int latent_width, int num_audio_latents);
void resolve_reference_image_size(int width, int height, int* out_h, int* out_w,
                                  int short_edge = 2048);
}  // namespace slopfab::dit
