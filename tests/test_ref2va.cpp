#include "harness.h"
#include "slopfab/dit/checkpoint.h"
#include "slopfab/dit/ref2va.h"
#include "slopfab/safetensors.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/vae/keyframe_encoder.h"

#include <cstdint>
#include <filesystem>
#include <fstream>

using namespace slopfab::dit;

namespace {
std::string checkpoint_fixture(
    const char* stem, std::vector<slopfab::TensorWrite> tensors) {
  const auto path = std::filesystem::temp_directory_path() /
                    (std::string("slopfab_") + stem + ".safetensors");
  slopfab::write_safetensors(path.string(), tensors);
  return path.string();
}

std::string int8_transformer_fixture() {
  const auto path = std::filesystem::temp_directory_path() /
                    "slopfab_int8_transformer_kind.safetensors";
  std::string header =
      "{\"blocks.0.attn.qkv_proj.weight\":{\"dtype\":\"I8\","
      "\"shape\":[3,2],\"data_offsets\":[0,6]},"
      "\"blocks.0.attn.qkv_proj.weight_scale\":{\"dtype\":\"F32\","
      "\"shape\":[3,1],\"data_offsets\":[6,18]}}";
  while ((8 + header.size()) % 8 != 0) header += ' ';
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  const uint64_t header_bytes = header.size();
  const int8_t weights[6] = {-3, -2, -1, 1, 2, 3};
  const float scales[3] = {0.25f, 0.5f, 1.0f};
  out.write(reinterpret_cast<const char*>(&header_bytes), sizeof(header_bytes));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  out.write(reinterpret_cast<const char*>(weights), sizeof(weights));
  out.write(reinterpret_cast<const char*>(scales), sizeof(scales));
  if (!out) throw std::runtime_error("failed writing INT8 transformer fixture");
  return path.string();
}
}  // namespace

SLOPFAB_TEST(ref2va_transformer_checkpoint_detection) {
  CHECK(is_pruned_table_architecture(TransformerArchitecture::kPrunedTable));
  CHECK(is_pruned_table_architecture(TransformerArchitecture::kRef2VAPrunedTable));
  CHECK(!is_pruned_table_architecture(TransformerArchitecture::kRef2VAFullAdaLN));
  CHECK(!is_pruned_table_architecture(TransformerArchitecture::kUnknown));
  const auto pruned_path = checkpoint_fixture(
      "pruned_kind", {{"adaln_t_table", {1}, {0.0f}},
                       {"blocks.0.adaln_proj.linear.weight", {1}, {0.0f}}});
  slopfab::SafeTensors pruned;
  pruned.open(pruned_path);
  CHECK(detect_transformer_architecture(pruned) == TransformerArchitecture::kPrunedTable);
  bool rejected = false;
  try {
    require_ref2va_transformer(pruned, 1);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  CHECK(rejected);
  require_ref2va_transformer(pruned, 0);

  const auto pruned_ref_path = checkpoint_fixture(
      "ref2va_pruned_fp8", {{"adaln_t_table", {1025, 8}, std::vector<float>(1025 * 8)},
                             {"blocks.0.adaln_proj.linear.weight", {6, 8},
                              std::vector<float>(6 * 8)}});
  slopfab::SafeTensors pruned_ref;
  pruned_ref.open(pruned_ref_path);
  CHECK(detect_transformer_architecture(pruned_ref) ==
        TransformerArchitecture::kRef2VAPrunedTable);
  CHECK(std::string(transformer_architecture_name(
            TransformerArchitecture::kRef2VAPrunedTable)) ==
        "pruned AdaLN-table Ref2VA transformer");
  require_ref2va_transformer(pruned_ref, 1);

  const auto ref_path = checkpoint_fixture(
      "ref2va_kind", {{"time_embedder.proj_in.weight", {1}, {0.0f}},
                       {"time_embedder.proj_out.weight", {1}, {0.0f}},
                       {"blocks.0.adaln_proj.linear.weight", {1}, {0.0f}},
                       {"blocks.0.attn.qkv_proj.weight", {1}, {0.0f}},
                       {"blocks.0.adaln_proj.linear.weight.quant_state.bitsandbytes__nf4",
                        {1}, {0.0f}},
                       {"blocks.0.attn.qkv_proj.weight.quant_state.bitsandbytes__nf4",
                        {1}, {0.0f}},
                       {"blocks.0.adaln_proj.linear.weight.absmax", {1}, {0.0f}}});
  slopfab::SafeTensors ref;
  ref.open(ref_path);
  CHECK(detect_transformer_architecture(ref) == TransformerArchitecture::kRef2VAFullAdaLN);
  CHECK(detect_transformer_quantization(ref) == TransformerQuantization::kBitsAndBytesNF4);
  CHECK(std::string(transformer_quantization_name(
            TransformerQuantization::kInt8ConvRot)) == "int8 ConvRot");

  slopfab::SafeTensors int8;
  int8.open(int8_transformer_fixture());
  CHECK(detect_transformer_quantization(int8) ==
        TransformerQuantization::kInt8ConvRot);
  require_ref2va_transformer(ref, 1);

  const auto unknown_path = checkpoint_fixture("unknown_kind", {{"x", {1}, {0.0f}}});
  slopfab::SafeTensors unknown;
  unknown.open(unknown_path);
  CHECK(detect_transformer_architecture(unknown) == TransformerArchitecture::kUnknown);

}

SLOPFAB_TEST(ref2va_image_size) {
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
  CHECK(::slopfab::test::throws([] { int a, b; resolve_reference_image_size(401, 100, &a, &b); }));
  CHECK(::slopfab::test::throws([] { int a, b; resolve_reference_image_size(100, 401, &a, &b); }));
  CHECK(::slopfab::test::throws([] { int a, b; resolve_reference_image_size(0, 1, &a, &b); }));
  CHECK(::slopfab::test::throws([] { int a, b; resolve_reference_image_size(1, -1, &a, &b); }));
  CHECK(::slopfab::test::throws([] { int b; resolve_reference_image_size(1, 1, nullptr, &b); }));
}

SLOPFAB_TEST(ref2va_keyframe_vae_contract) {
  slopfab::RGBImage image;
  image.width = 1;
  image.height = 1;
  image.pixels = {255, 0, 128};
  const auto pixels = slopfab::vae::prepare_keyframe_pixels(image);
  CHECK(pixels.size() == 3);
  CHECK(std::abs(pixels[0] - (1.0f - 0.485f) / 0.229f) < 1e-6f);
  CHECK(std::abs(pixels[1] - (0.0f - 0.456f) / 0.224f) < 1e-6f);

  std::vector<float> moments(48, 0.0f), normal(24, 0.0f), mean(24, 1.0f), sd(24, 2.0f);
  moments[0] = 3.0f;
  const auto latent = slopfab::vae::sample_keyframe_latents(
      moments.data(), normal.data(), 1, 1, mean, sd);
  CHECK(latent.size() == 24);
  CHECK(std::abs(latent[0] - 1.0f) < 1e-6f);
  CHECK(std::abs(latent[1] + 0.5f) < 1e-6f);
}

SLOPFAB_TEST(ref2va_keyframe_patchify) {
  std::vector<float> latent(24 * 2 * 4);
  for (size_t i = 0; i < latent.size(); ++i) latent[i] = static_cast<float>(i);
  const auto rows = slopfab::vae::patchify_keyframe_latents(latent.data(), 2, 4);
  CHECK(rows.size() == 2 * 96);
  CHECK(rows[0] == 0.0f);
  CHECK(rows[1] == 1.0f);
  CHECK(rows[2] == 4.0f);
  CHECK(rows[3] == 5.0f);
  CHECK(rows[4] == 8.0f);
  CHECK(rows[96] == 2.0f);
  CHECK(rows[99] == 7.0f);
  CHECK(::slopfab::test::throws([] {
    float latent[24 * 4] = {};
    slopfab::vae::patchify_keyframe_latents(latent, 1, 4);
  }));
}

SLOPFAB_TEST(ref2va_torch_cpu_seed42_normal) {
  const auto n = slopfab::vae::torch_cpu_normal_seed42(16);
  // torch.manual_seed(42); torch.randn(16), contiguous CPU float kernel.
  const float golden[] = {1.92691541f, 1.48728406f, 0.90071720f, -2.10552096f,
                          0.67841846f, -1.23454487f, -0.04306748f, -1.60466695f,
                          -0.75213528f, 1.64872301f, -0.39247864f, -1.40360725f,
                          -0.72788125f, -0.55943018f, -0.76883894f, 0.76244539f};
  for (int i = 0; i < 16; ++i) CHECK_NEAR(n[i], golden[i], 3e-6);
}
SLOPFAB_TEST(ref2va_order_positions_and_timesteps) {
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

SLOPFAB_TEST(ref2va_interleaved_reference_timesteps) {
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

SLOPFAB_TEST(ref2va_fixed_condition_noise_levels) {
  const ReferenceGeometry image{ReferenceKind::kImage, 1, 4, 4, 0};
  const auto p = build_ref2va_packed_sequence({kTagText}, {image}, 1, 4, 4, 2);
  const auto rt = build_row_timesteps(p.layout, p.indices, 0.7f, 0.3f, 0.999f, 1.0f);
  CHECK(rt.unique.size() == 3);
  CHECK_NEAR(rt.unique[0], 0.3f, 0.0);
  CHECK_NEAR(rt.unique[1], 0.7f, 0.0);
  CHECK_NEAR(rt.unique[2], 0.999f, 0.0);
  for (int i = 0; i < p.layout.num_condition_video; ++i) {
    const int row = p.indices.video[static_cast<size_t>(i)];
    CHECK(rt.indices[static_cast<size_t>(row)] == 2);
    CHECK(rt.adaln[static_cast<size_t>(row)] == 2 * 3 + kTagVideo);
  }
  const int target_video = p.indices.video[static_cast<size_t>(p.layout.num_condition_video)];
  CHECK(rt.indices[static_cast<size_t>(target_video)] == 1);
  const int target_audio = p.indices.audio[0];
  CHECK(rt.indices[static_cast<size_t>(target_audio)] == 0);
}
