#pragma once

#include <memory>
#include <string>
#include <vector>

#include "slopfab/dit/ref2va.h"
#include "slopfab/dtype.h"

namespace slopfab {

// Standalone ComfyUI H3RefMod: already-normalized VAE latents. Loading owns a
// snapshot, so queued requests neither retain a file mapping nor reread a file.
class RefMod {
public:
  static std::shared_ptr<const RefMod> load(const std::string& path);

  const std::string& path() const {
    return path_;
  }

  const std::string& name() const {
    return name_;
  }

  const std::string& description() const {
    return description_;
  }

  const dit::ReferenceGeometry& geometry() const {
    return geometry_;
  }

  int token_count() const {
    return geometry_.video_rows() + geometry_.audio_rows();
  }

  // Clean packed rows, before the pipeline's visual condition noise. Strength
  // blends towards an 8x downsampled/upsampled latent, preserving its mean.
  std::vector<float> rows(float strength = 1.0f) const;

private:
  RefMod() = default;
  std::string path_, name_, description_;
  dit::ReferenceGeometry geometry_;
  DType dtype_ = DType::kF32;
  std::vector<float> latent_;
};

struct RefModReference {
  std::shared_ptr<const RefMod> mod;
  float strength = 1.0f; // [0,1]; zero omits the reference entirely
  int copies = 1;        // [1,10]; each copy gets its own packed time position

  bool enabled() const {
    return mod && strength > 0.0f;
  }
};

void validate_refmods(const std::vector<RefModReference>& references);
void append_refmod_conditions(const std::vector<RefModReference>& references, uint64_t seed,
                              std::vector<dit::ReferenceGeometry>& geometry,
                              std::vector<float>& video_rows, std::vector<float>& audio_rows);

} // namespace slopfab
