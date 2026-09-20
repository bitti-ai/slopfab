#include "harness.h"
#include "slopfab/model_geometry.h"
#include "slopfab/dit/packing.h"
#include "slopfab/continuation.h"
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace {
template<class Fn> bool rejects(Fn fn) { try { fn(); return false; } catch (const std::exception&) { return true; } }
}
SLOPFAB_TEST(model_geometry_canonical_metadata_and_validation) {
  using namespace slopfab;
  const auto& h3 = h3_latent_geometry();
  const auto parsed = parse_model_geometry(geometry_json(h3));
  CHECK(parsed.fingerprint() == h3.fingerprint());
  require_h3_latent_geometry(parsed);
  const auto other = parse_model_geometry(R"({"version":1,"video_channels":8,"patch_height":1})");
  CHECK(other.video_patch_dim() == 16);
  CHECK(rejects([&] { require_h3_latent_geometry(other); }));
  for (const char* invalid : {R"({"version":2})", R"({"version":1,"patch_width":0})",
      R"({"version":1,"fps":23.5})", R"({"version":1,"unknown":0})",
      R"({"version":1,"patch_width":3})", R"({"version":1,"rope_frames_per_latent":[]})"})
    CHECK(rejects([&] { parse_model_geometry(invalid); }));
}
SLOPFAB_TEST(model_geometry_alternative_patches_and_audio_channels) {
  using namespace slopfab;
  LatentGeometry geometry;
  geometry.video_channels = 3;
  geometry.patch_height = 1;
  geometry.audio_channels = 1;
  geometry.audio_features = 5;
  dit::SequenceLayout layout;
  layout.num_latent_frames = 2;
  layout.latent_height = 3;
  layout.latent_width = 4;
  layout.num_video_rows = layout.num_latent_frames * layout.rows_per_frame(geometry);
  layout.num_audio_latents = 2;
  layout.num_audio_rows = 2;
  std::vector<float> latent(3*2*3*4);
  for (size_t i=0;i<latent.size();++i) latent[i]=static_cast<float>(i);
  std::vector<float> packed(latent.size()), restored(latent.size());
  dit::patchify_video(latent.data(),layout,packed.data(),geometry);
  CHECK(packed[0] == 0 && packed[1] == 1 && packed[2] == 24 && packed[4] == 48);
  dit::unpatchify_video(packed.data(),layout,restored.data(),geometry);
  CHECK(latent == restored);
  const auto positions = dit::build_position_ids(layout,geometry);
  CHECK(positions.size() == size_t(layout.total_rows())*3);
  std::vector<float> audio_rows = {0,1,2,3,4,5,6,7,8,9}, audio(10);
  dit::unpack_audio(audio_rows.data(),2,audio.data(),geometry);
  CHECK(audio == std::vector<float>({0,5,1,6,2,7,3,8,4,9}));
}
SLOPFAB_TEST(model_geometry_temporal_law_and_overflow) {
  using namespace slopfab;
  LatentGeometry geometry;
  geometry.fps = 30;
  geometry.audio_latents_per_second = 60;
  geometry.frames_per_chunk = 4;
  geometry.frame_offset = 1;
  geometry.latents_per_chunk = 2;
  geometry.latent_frame_offset = 1;
  CHECK(dit::align_num_frames(6,geometry) == 9);
  CHECK(dit::video_latent_num_frames(9,geometry) == 5);
  CHECK(dit::audio_latents_for_frames(9,geometry) == 18);
  CHECK(rejects([&] { dit::align_num_frames(std::numeric_limits<int>::max(),geometry); }));
  CHECK(rejects([&] { dit::video_latent_num_frames(8,geometry); }));
}
SLOPFAB_TEST(model_geometry_archive_preserves_contract) {
  using namespace slopfab;
  LatentClip clip;
  clip.width = clip.height = 32;
  clip.frames = 1;
  clip.video_rows.resize(96,1.0f);
  const auto path = std::filesystem::temp_directory_path()/"slopfab_geometry_archive.safetensors";
  clip.save(path.string());
  const auto loaded = LatentClip::load(path.string());
  CHECK(loaded->geometry.fingerprint() == clip.geometry.fingerprint());
  CHECK(loaded->video_rows == clip.video_rows);
  std::filesystem::remove(path);
  clip.geometry.fps=30;
  CHECK(rejects([&] { clip.validate(); }));
}
