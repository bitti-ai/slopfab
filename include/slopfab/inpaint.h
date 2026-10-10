#pragma once

#include <memory>
#include <functional>
#include <vector>
#include "slopfab/image.h"
#include "slopfab/pixel_buffer.h"
#include "slopfab/dit/ref2va.h"

namespace slopfab {

// Source pixels are an immutable snapshot. The half-open box uses source
// coordinates; feathering stays inside it. Internal padding never changes
// output dimensions. A null image disables editing.
struct ImageEdit {
  std::shared_ptr<const RGBImage> image;
  int x = 0, y = 0, width = 0, height = 0;
  float strength = 1.0f; // (0,1], fraction of schedule evaluations retained
  int feather = 0;       // pixels, inward from the box boundary
  bool invert_mask = false; // outpaint: box is source context; generate its entire surround
  int blend_overlap = 9; // outpaint only: odd dilation/blur kernel, 1..51; 1 = hard edge
  int langevin_steps = 5; // outpaint only: extra model evaluations per step, 0..100
  void validate() const;
};

RGBImage pad_edit_image(const ImageEdit& edit, int width, int height);
RGBImage outpaint_source_image(const ImageEdit& edit);
std::vector<float> edit_mask_rows(const ImageEdit& edit, int width, int height);
dit::ReferenceGeometry outpaint_keyframe_geometry(const ImageEdit& edit);
void append_outpaint_keyframe(const ImageEdit& edit, int width, int height,
                             const std::vector<float>& original,
                             std::vector<dit::ReferenceGeometry>& geometry,
                             std::vector<float>& rows);
PixelBuffer composite_image_edit(const ImageEdit& edit, const PixelBuffer& generated, int width,
                                 int height);

// Packed target rows only; reference/condition rows are never constrained.
struct InpaintConstraint {
  std::vector<float> original, noise, mask;
  int langevin_steps = 0; // disabled for ordinary image edits
  void validate(size_t count) const;
  std::vector<float> initial(float sigma) const;
  // Re-noising supplies the same fresh row-layout noise as the scheduler.
  // Otherwise reuse initial noise. At sigma zero no noise is read.
  void apply(float* rows, size_t count, float sigma, const float* step_noise = nullptr) const;
  // Flow-model LanPaint refinement at a fixed sigma. evaluate returns H3's
  // velocity (x0 - noise); false cancels before the next inner evaluation.
  // Only target rows are passed, so clean reference anchors remain untouched.
  bool refine(float* rows, size_t count, float sigma, uint64_t seed, int outer_step,
              const std::function<bool(const float*, float*)>& evaluate) const;
};

} // namespace slopfab
