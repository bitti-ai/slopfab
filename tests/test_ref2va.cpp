#include "harness.h"
#include "vidfab/dit/checkpoint.h"
#include "vidfab/dit/ref2va.h"
#include "vidfab/safetensors.h"
#include "vidfab/safetensors_write.h"
#include "vidfab/vae/keyframe_encoder.h"

#include <filesystem>

using namespace vidfab::dit;

namespace {
std::string checkpoint_fixture(
    const char* stem, std::vector<vidfab::TensorWrite> tensors) {
  const auto path = std::filesystem::temp_directory_path() /
                    (std::string("vidfab_") + stem + ".safetensors");
  vidfab::write_safetensors(path.string(), tensors);
  return path.string();
}
}  // namespace

VIDFAB_TEST(ref2va_transformer_checkpoint_detection) {
  const auto pruned_path = checkpoint_fixture(
      "pruned_kind", {{"adaln_t_table", {1}, {0.0f}},
                       {"blocks.0.adaln_proj.linear.weight", {1}, {0.0f}}});
  vidfab::SafeTensors pruned;
  pruned.open(pruned_path);
  CHECK(detect_transformer_checkpoint(pruned) == TransformerCheckpointKind::kPrunedTable);
  bool rejected = false;
  try {
    require_ref2va_transformer(pruned, 1);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  CHECK(rejected);
  require_ref2va_transformer(pruned, 0);

  const auto ref_path = checkpoint_fixture(
      "ref2va_kind", {{"time_embedder.proj_in.weight", {1}, {0.0f}},
                       {"time_embedder.proj_out.weight", {1}, {0.0f}},
                       {"blocks.0.adaln_proj.linear.weight.quant_state.bitsandbytes__nf4",
                        {1}, {0.0f}},
                       {"blocks.0.adaln_proj.linear.weight.absmax", {1}, {0.0f}}});
  vidfab::SafeTensors ref;
  ref.open(ref_path);
  CHECK(detect_transformer_checkpoint(ref) ==
        TransformerCheckpointKind::kRef2VABitsAndBytesNF4);
  require_ref2va_transformer(ref, 1);

  const auto unknown_path = checkpoint_fixture("unknown_kind", {{"x", {1}, {0.0f}}});
  vidfab::SafeTensors unknown;
  unknown.open(unknown_path);
  CHECK(detect_transformer_checkpoint(unknown) == TransformerCheckpointKind::kUnknown);

}

VIDFAB_TEST(ref2va_image_size) {
  int h = 0, w = 0;

  // Ref2VA preserves the source aspect ratio around a 2048-pixel short edge.
  resolve_reference_image_size(100, 50, &h, &w);
  CHECK(h == 2048);
  CHECK(w == 4096);
  resolve_reference_image_size(50, 100, &h, &w);
  CHECK(h == 4096);
  CHECK(w == 2048);
  resolve_reference_image_size(17, 17, &h, &w);
  CHECK(h == 2048);
  CHECK(w == 2048);

  // Python round is ties-to-even. 64.5 multiples rounds down to 64, while
  // 65.5 rounds up to 66. These are small inputs but exact rational ties after
  // setting the 128-pixel short edge to 2048.
  resolve_reference_image_size(129, 128, &h, &w);
  CHECK(h == 2048);
  CHECK(w == 2048);
  resolve_reference_image_size(131, 128, &h, &w);
  CHECK(h == 2048);
  CHECK(w == 2112);

  // The supported range is inclusive and is validated before resizing.
  resolve_reference_image_size(4, 1, &h, &w);
  CHECK(h == 2048 && w == 8192);
  resolve_reference_image_size(1, 4, &h, &w);
  CHECK(h == 8192 && w == 2048);
  CHECK(::vidfab::test::throws([] { int a, b; resolve_reference_image_size(401, 100, &a, &b); }));
  CHECK(::vidfab::test::throws([] { int a, b; resolve_reference_image_size(100, 401, &a, &b); }));
  CHECK(::vidfab::test::throws([] { int a, b; resolve_reference_image_size(0, 1, &a, &b); }));
  CHECK(::vidfab::test::throws([] { int a, b; resolve_reference_image_size(1, -1, &a, &b); }));
  CHECK(::vidfab::test::throws([] { int b; resolve_reference_image_size(1, 1, nullptr, &b); }));
}

VIDFAB_TEST(ref2va_keyframe_vae_contract) {
  vidfab::RGBImage image;
  image.width = 1;
  image.height = 1;
  image.pixels = {255, 0, 128};
  const auto pixels = vidfab::vae::prepare_keyframe_pixels(image);
  CHECK(pixels.size() == 3);
  CHECK(std::abs(pixels[0] - (1.0f - 0.485f) / 0.229f) < 1e-6f);
  CHECK(std::abs(pixels[1] - (0.0f - 0.456f) / 0.224f) < 1e-6f);

  std::vector<float> moments(48, 0.0f), normal(24, 0.0f), mean(24, 1.0f), sd(24, 2.0f);
  moments[0] = 3.0f;
  const auto latent = vidfab::vae::sample_keyframe_latents(
      moments.data(), normal.data(), 1, 1, mean, sd);
  CHECK(latent.size() == 24);
  CHECK(std::abs(latent[0] - 1.0f) < 1e-6f);
  CHECK(std::abs(latent[1] + 0.5f) < 1e-6f);
}
VIDFAB_TEST(ref2va_order_positions_and_timesteps) {
  const ReferenceGeometry image{ReferenceKind::kImage, 1, 4, 6, 0};  // 6 video rows
  const ReferenceGeometry audio{ReferenceKind::kAudio, 1, 0, 0, 2};  // 4 audio rows
  const ReferenceGeometry clip{ReferenceKind::kVideo, 2, 4, 4, 3};   // 6 audio, 8 video
  const auto p = build_ref2va_packed_sequence(
      {kTagText, kTagVideo, kTagText}, {image, audio, clip}, 2, 4, 4, 2);

  CHECK(p.layout.num_text == 3);
  CHECK(p.layout.num_condition_video == 14);
  CHECK(p.layout.num_condition_audio == 10);
  CHECK(p.layout.condition_audio_is_explicit);
  CHECK(p.layout.num_audio_rows == 4);
  CHECK(p.layout.num_video_rows == 8);
  CHECK(p.layout.total_rows() == 39);

  // Exact reference order: text | image video | audio reference | clip audio |
  // clip video | target audio | target video. Modality index vectors retain
  // that packed order even though each contains non-contiguous runs.
  CHECK(p.indices.video.size() == 22);
  CHECK(p.indices.audio.size() == 14);
  CHECK(p.indices.video[0] == 3);
  CHECK(p.indices.video[5] == 8);
  CHECK(p.indices.audio[0] == 9);
  CHECK(p.indices.audio[3] == 12);
  CHECK(p.indices.audio[4] == 13);
  CHECK(p.indices.audio[9] == 18);
  CHECK(p.indices.video[6] == 19);
  CHECK(p.indices.video[13] == 26);
  CHECK(p.indices.audio[10] == 27);
  CHECK(p.indices.video[14] == 31);

  CHECK(p.indices.tags[1] == kTagVideo);  // text-region vision token is preserved
  CHECK(p.indices.tags[3] == kTagVideo);
  CHECK(p.indices.tags[9] == kTagAudio);
  CHECK(p.indices.tags[19] == kTagVideo);
  CHECK(p.indices.tags[27] == kTagAudio);

  // Each reference advances one shared media clock. An image advances by one;
  // audio by A; a video by max(A, its non-uniform video span).
  CHECK_NEAR(p.position_ids[3 * 3], 3.0, 0.0);    // image origin
  CHECK_NEAR(p.position_ids[9 * 3], 4.0, 0.0);    // audio reference origin
  CHECK_NEAR(p.position_ids[13 * 3], 6.0, 0.0);   // clip audio origin
  CHECK_NEAR(p.position_ids[19 * 3], 6.0, 0.0);   // clip video shares it
  CHECK_NEAR(p.position_ids[23 * 3], 7.666666666666667, 1e-12);  // clip frame 1
  CHECK_NEAR(p.position_ids[27 * 3], 14.333333333333334, 1e-12); // target audio
  CHECK_NEAR(p.position_ids[31 * 3], 14.333333333333334, 1e-12); // target video

  // Conditioning rows of either modality are anchored to video_t. Only the
  // generated target audio is stepped on audio_t.
  const RowTimesteps rt = build_row_timesteps(p.layout, p.indices, 0.8f, 0.2f);
  CHECK(rt.unique.size() == 2);
  if (rt.unique.size() == 2) {
    CHECK_NEAR(rt.unique[0], 0.2, 1e-7);
    CHECK_NEAR(rt.unique[1], 0.8, 1e-7);
  }
  for (int row = 0; row < 27; ++row) CHECK(rt.indices[static_cast<size_t>(row)] == 1);
  for (int row = 27; row < 31; ++row) CHECK(rt.indices[static_cast<size_t>(row)] == 0);
  for (int row = 31; row < 39; ++row) CHECK(rt.indices[static_cast<size_t>(row)] == 1);
}

VIDFAB_TEST(ref2va_interleaved_reference_timesteps) {
  ReferenceGeometry image{ReferenceKind::kImage,1,4,4,0};
  ReferenceGeometry sound{ReferenceKind::kAudio,1,0,0,2};
  ReferenceGeometry clip{ReferenceKind::kVideo,1,4,4,1};
  auto p=build_ref2va_packed_sequence({kTagText},{image,sound,clip},1,4,4,2);
  auto rt=build_row_timesteps(p.layout,p.indices,0.75f,0.25f);
  // Indices are condition-first within each modality despite packed interleaving.
  for(int i=0;i<p.layout.num_condition_audio;++i)
    CHECK(rt.indices[static_cast<size_t>(p.indices.audio[i])]==1);
  for(size_t i=static_cast<size_t>(p.layout.num_condition_audio);i<p.indices.audio.size();++i)
    CHECK(rt.indices[static_cast<size_t>(p.indices.audio[i])]==0);
  for(int i:p.indices.video) CHECK(rt.indices[static_cast<size_t>(i)]==1);

  // Image-only references have no audio anchors: every audio row is generated.
  auto image_only=build_ref2va_packed_sequence({kTagText},{image},1,4,4,2);
  auto image_rt=build_row_timesteps(image_only.layout,image_only.indices,0.75f,0.25f);
  for(int i:image_only.indices.audio) CHECK(image_rt.indices[static_cast<size_t>(i)]==0);
}
