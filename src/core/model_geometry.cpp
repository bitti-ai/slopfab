#include "slopfab/model_geometry.h"
#include "slopfab/json.h"
#include <cmath>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <map>

namespace slopfab {
namespace {
using IntField = int LatentGeometry::*;
const std::map<std::string, IntField> ints = {
    {"fps", &LatentGeometry::fps},
    {"spatial_compression", &LatentGeometry::spatial_compression},
    {"video_channels", &LatentGeometry::video_channels},
    {"audio_features", &LatentGeometry::audio_features},
    {"audio_channels", &LatentGeometry::audio_channels},
    {"audio_latents_per_second", &LatentGeometry::audio_latents_per_second},
    {"audio_sample_rate", &LatentGeometry::audio_sample_rate},
    {"patch_height", &LatentGeometry::patch_height},
    {"patch_width", &LatentGeometry::patch_width},
    {"frames_per_chunk", &LatentGeometry::frames_per_chunk},
    {"latents_per_chunk", &LatentGeometry::latents_per_chunk},
    {"frame_offset", &LatentGeometry::frame_offset},
    {"latent_frame_offset", &LatentGeometry::latent_frame_offset},
    {"canvas_multiple", &LatentGeometry::canvas_multiple},
    {"trained_max_pixels", &LatentGeometry::trained_max_pixels}};
const std::map<std::string, double LatentGeometry::*> reals = {
    {"min_aspect", &LatentGeometry::min_aspect},
    {"max_aspect", &LatentGeometry::max_aspect},
    {"rope_frame_rescale", &LatentGeometry::rope_frame_rescale},
    {"rope_spatial_scale", &LatentGeometry::rope_spatial_scale}};

void require(bool ok, const std::string& detail) {
  if (!ok)
    throw std::runtime_error("geometry: " + detail);
}

int integer(const json::Value& value) {
  const double number = value.as_number();
  require(std::isfinite(number) && number > 0 && std::floor(number) == number &&
              number <= std::numeric_limits<int>::max(),
          "expected a positive integer");
  return static_cast<int>(number);
}
}

const LatentGeometry& h3_latent_geometry() {
  static const LatentGeometry geometry;
  return geometry;
}

void validate_latent_geometry(const LatentGeometry& g) {
  require(g.version == 1, "unsupported version");
  for (const auto& f : ints)
    require(g.*f.second > 0, f.first + " must be positive");
  for (const auto& f : reals)
    require(std::isfinite(g.*f.second) && g.*f.second > 0,
            f.first + " must be positive and finite");
  require(g.min_aspect <= g.max_aspect, "aspect bounds are inverted");
  require(g.frame_offset < g.frames_per_chunk,
          "frame_offset must be smaller than frames_per_chunk");
  require(int64_t(g.patch_height) * g.patch_width <=
              std::numeric_limits<int>::max() / g.video_channels,
          "video patch dimension overflows");
  require(int64_t(g.spatial_compression) * g.patch_height <= g.canvas_multiple &&
              int64_t(g.spatial_compression) * g.patch_width <= g.canvas_multiple &&
              g.canvas_multiple % (g.spatial_compression * g.patch_height) == 0 &&
              g.canvas_multiple % (g.spatial_compression * g.patch_width) == 0,
          "canvas_multiple must cover complete compressed patches");
  require(!g.rope_frames_per_latent.empty() && g.rope_frames_per_latent.size() <= 4096,
          "invalid temporal position pattern");
  for (int n : g.rope_frames_per_latent)
    require(n > 0, "temporal position steps must be positive");
}

std::string geometry_json(const LatentGeometry& g) {
  validate_latent_geometry(g);
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out.precision(std::numeric_limits<double>::max_digits10);
  out << "{\"version\":1";
  for (const auto& f : ints)
    out << ",\"" << f.first << "\":" << g.*f.second;
  for (const auto& f : reals)
    out << ",\"" << f.first << "\":" << g.*f.second;
  out << ",\"rope_frames_per_latent\":[";
  for (size_t i = 0; i < g.rope_frames_per_latent.size(); ++i) {
    if (i)
      out << ',';
    out << g.rope_frames_per_latent[i];
  }
  return out.str() + "]}";
}

std::string LatentGeometry::fingerprint() const {
  return geometry_json(*this);
}

void require_h3_latent_geometry(const LatentGeometry& g) {
  validate_latent_geometry(g);
  // Canvas policy does not change tensor layout, codec arithmetic or positions.
  // The planner consumes these profile defaults before allocating model state.
  auto structural = g;
  const auto& h3 = h3_latent_geometry();
  structural.trained_max_pixels = h3.trained_max_pixels;
  structural.min_aspect = h3.min_aspect;
  structural.max_aspect = h3.max_aspect;
  require(structural.fingerprint() == h3.fingerprint(),
          "checkpoint geometry is not supported by the H3 transformer and codecs");
}

LatentGeometry parse_model_geometry(std::string_view text) {
  const auto root = json::parse(text);
  require(root.is_object(), "expected an object");
  const auto* version = root.find("version");
  require(version && version->as_number() == 1, "unsupported or missing version");
  LatentGeometry geometry;
  for (const auto& field : root.as_object()) {
    if (field.first == "version")
      continue;
    if (const auto it = ints.find(field.first); it != ints.end())
      geometry.*it->second = integer(field.second);
    else if (const auto it = reals.find(field.first); it != reals.end())
      geometry.*it->second = field.second.as_number();
    else if (field.first == "rope_frames_per_latent") {
      geometry.rope_frames_per_latent.clear();
      for (const auto& item : field.second.as_array())
        geometry.rope_frames_per_latent.push_back(integer(item));
    } else
      throw std::runtime_error("geometry: unknown field " + field.first);
  }
  validate_latent_geometry(geometry);
  return geometry;
}

LatentGeometry read_model_geometry(const SafeTensors& checkpoint) {
  const auto it = checkpoint.metadata().find("slopfab.geometry");
  return it == checkpoint.metadata().end() ? h3_latent_geometry()
                                           : parse_model_geometry(it->second);
}
}
