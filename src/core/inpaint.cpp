#include "slopfab/inpaint.h"
#include "slopfab/dit/packing.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace slopfab {
void ImageEdit::validate() const {
  if (!image) {
    if (x || y || width || height || feather || strength != 1.0f)
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
  std::vector<float> mask(24 * plane, 0.0f), rows(mask.size());
  // Any overlap makes a latent cell editable. Pixel-space compositing enforces
  // the exact box even for sub-cell selections and unaligned edges.
  for (int y = edit.y / 16; y < (edit.y + edit.height + 15) / 16; ++y)
    for (int x = edit.x / 16; x < (edit.x + edit.width + 15) / 16; ++x)
      for (int c = 0; c < 24; ++c)
        mask[c * plane + size_t(y) * layout.latent_width + x] = 1.0f;
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

void InpaintConstraint::apply(float* rows, size_t count, float sigma) const {
  if (!rows || count != original.size() || noise.size() != count || mask.size() != count ||
      !std::isfinite(sigma) || sigma < 0 || sigma > 1)
    throw std::invalid_argument("invalid inpainting update");
  for (size_t i = 0; i < count; ++i)
    if (mask[i] == 0)
      rows[i] = sigma == 0 ? original[i] : (1 - sigma) * original[i] + sigma * noise[i];
}
} // namespace slopfab
