#include "slopfab/inpaint.h"
#include "slopfab/dit/packing.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace slopfab {
void ImageEdit::validate() const {
  if (!image) {
    if (x || y || width || height || feather || strength != 1.0f || invert_mask)
      throw std::invalid_argument("image edit settings require a source image");
    return;
  }
  if (image->width <= 0 || image->height <= 0 || image->width > 8192 || image->height > 8192 ||
      image->pixels.size() != size_t(image->width) * image->height * 3)
    throw std::invalid_argument("image edit requires valid RGB pixels, at most 8192 per axis");
  if (x < 0 || y < 0 || width <= 0 || height <= 0 || int64_t(x) + width > image->width ||
      int64_t(y) + height > image->height)
    throw std::invalid_argument("edit box must be nonempty and inside the source image");
  if (!std::isfinite(strength) || strength <= 0 || strength > 1 || feather < 0)
    throw std::invalid_argument("edit strength must be in (0,1] and feather nonnegative");
  if (invert_mask && (feather != 0 || (x == 0 && y == 0 && width == image->width && height == image->height)))
    throw std::invalid_argument("outpainting needs space outside the preserved box and zero feather");
  if (invert_mask && ((x + 15) / 16 >= (x + width) / 16 || (y + 15) / 16 >= (y + height) / 16))
    throw std::invalid_argument("outpainting needs at least one complete latent cell of original context");
}

RGBImage pad_edit_image(const ImageEdit& edit, int width, int height) {
  edit.validate();
  if (!edit.image || width < edit.image->width || height < edit.image->height || width > 8192 ||
      height > 8192)
    throw std::invalid_argument("invalid image edit canvas");
  const auto& src = *edit.image;
  RGBImage out{width, height, std::vector<uint8_t>(size_t(width) * height * 3)};
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      for (int c = 0; c < 3; ++c)
        out.pixels[(size_t(y) * width + x) * 3 + c] =
            src.pixels[(size_t(std::min(y, src.height - 1)) * src.width +
                        std::min(x, src.width - 1)) *
                           3 +
                       c];
  return out;
}

RGBImage outpaint_source_image(const ImageEdit& edit) {
  edit.validate();
  if (!edit.image || !edit.invert_mask)
    throw std::invalid_argument("source scene requires an outpainting edit");
  RGBImage source{edit.width, edit.height, std::vector<uint8_t>(size_t(edit.width) * edit.height * 3)};
  for (int y = 0; y < edit.height; ++y)
    std::copy_n(edit.image->pixels.data() + (size_t(y + edit.y) * edit.image->width + edit.x) * 3,
                size_t(edit.width) * 3, source.pixels.data() + size_t(y) * edit.width * 3);
  return source;
}

dit::ReferenceGeometry outpaint_keyframe_geometry(const ImageEdit& edit) {
  edit.validate();
  if (!edit.image || !edit.invert_mask)
    throw std::invalid_argument("positioned source requires an outpainting edit");
  const int left = (edit.x + 31) / 32, top = (edit.y + 31) / 32;
  const int right = (edit.x + edit.width) / 32, bottom = (edit.y + edit.height) / 32;
  dit::ReferenceGeometry geometry{dit::ReferenceKind::kImage, 1,
      std::max(0, bottom - top) * 2, std::max(0, right - left) * 2, 0};
  geometry.target_aligned = true;
  geometry.target_latent_x = left * 2;
  geometry.target_latent_y = top * 2;
  return geometry;
}

void append_outpaint_keyframe(const ImageEdit& edit, int width, int height,
                             const std::vector<float>& original,
                             std::vector<dit::ReferenceGeometry>& geometry,
                             std::vector<float>& rows) {
  const auto keyframe = outpaint_keyframe_geometry(edit);
  if (width <= 0 || height <= 0 || width % 32 || height % 32 ||
      original.size() != size_t(width / 32) * (height / 32) * 96 ||
      width < edit.image->width || height < edit.image->height)
    throw std::invalid_argument("outpaint keyframe latent shape mismatch");
  // Tiny legacy masks can still use latent locking; the UI requires a full
  // patch for the stronger positioned anchor. Never anchor boundary padding.
  if (keyframe.video_rows() == 0) return;
  for (int y = keyframe.target_latent_y / 2;
       y < (keyframe.target_latent_y + keyframe.latent_height) / 2; ++y) {
    const auto start = original.begin() + (size_t(y) * (width / 32) + keyframe.target_latent_x / 2) * 96;
    rows.insert(rows.end(), start, start + size_t(keyframe.latent_width / 2) * 96);
  }
  geometry.push_back(keyframe);
}

std::vector<float> edit_mask_rows(const ImageEdit& edit, int width, int height) {
  edit.validate();
  if (!edit.image || width < edit.image->width || height < edit.image->height || width > 8192 ||
      height > 8192 || width % 32 || height % 32)
    throw std::invalid_argument("image edit canvas must contain complete H3 patches");
  dit::SequenceLayout layout;
  layout.num_latent_frames = 1;
  layout.latent_width = width / 16;
  layout.latent_height = height / 16;
  const size_t plane = size_t(layout.latent_width) * layout.latent_height;
  std::vector<float> mask(24 * plane, edit.invert_mask ? 1.0f : 0.0f), rows(mask.size());
  // Any overlap makes a latent cell editable. Pixel-space compositing enforces
  // the exact box even for sub-cell selections and unaligned edges.
  // In outpaint mode only cells fully inside the original stay locked, so
  // every new pixel (including unaligned seams) can be generated together.
  const int left = (edit.x + (edit.invert_mask ? 15 : 0)) / 16;
  const int top = (edit.y + (edit.invert_mask ? 15 : 0)) / 16;
  const int right = (edit.x + edit.width + (edit.invert_mask ? 0 : 15)) / 16;
  const int bottom = (edit.y + edit.height + (edit.invert_mask ? 0 : 15)) / 16;
  for (int y = top; y < bottom; ++y)
    for (int x = left; x < right; ++x)
      for (int c = 0; c < 24; ++c)
        mask[c * plane + size_t(y) * layout.latent_width + x] = edit.invert_mask ? 0.0f : 1.0f;
  dit::patchify_video(mask.data(), layout, rows.data());
  return rows;
}

PixelBuffer composite_image_edit(const ImageEdit& edit, const PixelBuffer& generated, int width,
                                 int height) {
  edit.validate();
  if (!edit.image || width < edit.image->width || height < edit.image->height ||
      generated.size() != size_t(width) * height * 3)
    throw std::invalid_argument("image edit decoded shape mismatch");
  const auto& src = *edit.image;
  const size_t plane = size_t(src.width) * src.height;
  PixelBuffer out(3 * plane);
  for (int y = 0; y < src.height; ++y) {
    for (int x = 0; x < src.width; ++x) {
      float alpha = 0;
      if (x >= edit.x && x < edit.x + edit.width && y >= edit.y && y < edit.y + edit.height) {
        alpha = 1;
        if (edit.feather) {
          const float distance = float(std::min({x - edit.x, edit.x + edit.width - 1 - x,
                                                 y - edit.y, edit.y + edit.height - 1 - y})) +
                                 .5f;
          alpha = std::min(1.0f, distance / edit.feather);
        }
      }
      if (edit.invert_mask)
        alpha = 1 - alpha;
      for (int c = 0; c < 3; ++c) {
        const size_t p = size_t(y) * src.width + x;
        const float original = src.pixels[3 * p + c] / 255.0f;
        // Branching also prevents unused NaNs from corrupting preserved pixels.
        out[c * plane + p] = alpha == 0 ? original
                                        : alpha * generated[(size_t(c) * height + y) * width + x] +
                                              (1 - alpha) * original;
      }
    }
  }
  return out;
}

void InpaintConstraint::validate(size_t count) const {
  if (!count || original.size() != count || noise.size() != count || mask.size() != count)
    throw std::invalid_argument("inpainting constraint shape mismatch");
  for (size_t i = 0; i < count; ++i)
    if (!std::isfinite(original[i]) || !std::isfinite(noise[i]) || (mask[i] != 0 && mask[i] != 1))
      throw std::invalid_argument("inpainting needs finite latents and a binary mask");
}

std::vector<float> InpaintConstraint::initial(float sigma) const {
  validate(original.size());
  if (!std::isfinite(sigma) || sigma < 0 || sigma > 1)
    throw std::invalid_argument("invalid inpainting sigma");
  std::vector<float> out(original.size());
  for (size_t i = 0; i < out.size(); ++i)
    out[i] = (1 - sigma) * original[i] + sigma * noise[i];
  return out;
}

void InpaintConstraint::apply(float* rows, size_t count, float sigma, const float* step_noise) const {
  if (!rows || count != original.size() || noise.size() != count || mask.size() != count ||
      !std::isfinite(sigma) || sigma < 0 || sigma > 1)
    throw std::invalid_argument("invalid inpainting update");
  const float* preserved_noise = step_noise ? step_noise : noise.data();
  for (size_t i = 0; i < count; ++i)
    if (mask[i] == 0)
      rows[i] = sigma == 0 ? original[i] : (1 - sigma) * original[i] + sigma * preserved_noise[i];
}
} // namespace slopfab
