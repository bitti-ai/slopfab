#include "harness.h"
#include "latent_fixture.h"
#include "slopfab/h3_image.h"
#include "slopfab/dit/adaln.h"
#include "slopfab/model_profile.h"
#include "slopfab/generate.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/dit/pdd.h"
#include "slopfab/vae/vit_block.h"
#include "slopfab/tensor_convert.h"
#include <cstdlib>
#include <cmath>
#include <limits>
#include <fstream>
#include "slopfab/vae/keyframe_encoder.h"
#include "slopfab/dit/ref2va.h"

namespace {
bool rejects(const slopfab::GenerateRequest& request) {
  try { (void)slopfab::resolve_plan(request); }
  catch (const std::exception&) { return true; }
  return false;
}
}

SLOPFAB_TEST(h3_image_plan_matches_published_workflow) {
  using namespace slopfab;
  GenerateRequest r;
  r.model_family = "h3_image";
  const auto p = resolve_plan(r);
  CHECK(p.model.family == "h3_image" && !p.checkpoint_available);
  CHECK(model_profile("h3_image").image_only);
  CHECK(model_profile("h3_image").conditioner_width == 5120);
  CHECK(p.canvas_width == 4096 && p.canvas_height == 4096);
  CHECK(p.layout.num_latent_frames == 1 && p.aligned_frames == 1);
  CHECK(p.layout.num_video_rows == 128 * 128);
  CHECK(p.layout.num_audio_latents == 0 && p.layout.num_audio_rows == 0);
  CHECK(p.video_sigma_shift == 1 && p.audio_sigma_shift == 1);
  CHECK(p.num_inference_steps == 6 && p.num_model_evaluations() == 6);
  CHECK(p.fixed_sampling_grid && p.sampler == sampler::SamplerKind::kEuler);
  const std::vector<float> expected{1.f, .9675236940383911f, .8875485062599182f,
      .75f, .5327739119529724f, .23200887441635132f, 0.f};
  CHECK(p.video_sigmas == expected && p.audio_sigmas == expected);
  for (size_t i = 0; i < 6; ++i) {
    CHECK_NEAR(p.video_timesteps[i], 1 - expected[i], 1e-7);
    CHECK(p.audio_timesteps[i] == p.video_timesteps[i]);
  }
  CHECK(describe_plan(r, p).find("MiniMax-H3-Image") != std::string::npos);
  r.canvas_width = 3872;
  r.canvas_height = 2176;
  r.num_inference_steps = 6;
  r.guidance_scale_set = true;
  r.guidance_scale = 1;
  r.sampling.base_sigmas = expected;
  CHECK(resolve_plan(r).layout.num_video_rows == 121 * 68);
}

SLOPFAB_TEST(h3_image_rejects_incompatible_requests) {
  using namespace slopfab;
  GenerateRequest base;
  base.model_family = "h3_image";
  auto r = base;
  r.num_inference_steps = 4; CHECK(rejects(r));
  r = base; r.sampling.video_sigma_shift = 12; CHECK(rejects(r));
  r = base; r.sampling.audio_steps = 4; CHECK(rejects(r));
  r = base; r.sampling.base_sigmas = std::vector<float>{1, .5f, 0}; CHECK(rejects(r));
  r = base; r.schedule = sampler::ScheduleKind::kBeta; CHECK(rejects(r));
  r = base; r.sampling.sampler = sampler::SamplerKind::kAb2; CHECK(rejects(r));
  r = base; r.canvas_width = 1024; CHECK(rejects(r));
  r.canvas_height = 1000; CHECK(rejects(r));
  r.canvas_width = r.canvas_height = 8192; CHECK(rejects(r));
  r = base; r.negative_prompt = "bad"; CHECK(rejects(r));
  r = base; r.guidance_scale_set = true; CHECK(rejects(r));
  r.guidance_scale = std::numeric_limits<float>::quiet_NaN(); CHECK(rejects(r));
  r = base; r.reference_image_paths.assign(17, "unused.png"); CHECK(rejects(r));
  r = base; r.end_anchor_path = "unused.png"; CHECK(rejects(r));
  r = base; r.loras = {{"unused.safetensors", 1}}; CHECK(rejects(r));
  r = base; r.cache_threshold = .1f; CHECK(rejects(r));
  r = base; r.conditioning.pin_target_audio = true; CHECK(rejects(r));
  const auto p = resolve_plan(base);
  bool rejected = false;
  try { validate_sampling_sampler(p, sampler::SamplerKind::kRenoise); }
  catch (const std::exception&) { rejected = true; }
  CHECK(rejected);
  RunOptions options;
  options.attention_mode = AttentionMode::kFlash2;
  options.vulkan_portable_arithmetic = true;
  validate_generation_options(base, p, options);
  options.sampler = sampler::SamplerKind::kAb2;
  rejected = false;
  try { validate_generation_options(base, p, options); }
  catch (const std::exception&) { rejected = true; }
  CHECK(rejected);
}

SLOPFAB_TEST(h3_image_checkpoint_identity_and_shape_validation) {
  using namespace slopfab;
  LatentFixture fixture; // an unrelated filename must not affect recognition
  std::vector<TensorWrite> tensors;
  for (const auto& item : std::vector<std::pair<std::string, std::vector<int64_t>>>{
           {"adaln_t_table", {1025, 8}}, {"video_patch_proj.weight", {5376, 96}},
           {"condition_proj.weight", {5376, 5120}},
           {"blocks.0.adaln_proj.linear.weight", {96768, 8}},
           {"final_layer.adaln_proj.linear.weight", {10752, 8}},
           {"final_layer.video_out.weight", {576, 5376}},
           {"final_layer.video_out.bias", {576, 1}}})
    tensors.push_back({item.first, item.second,
                      std::vector<float>(size_t(item.second[0]) * item.second[1]), DType::kBF16});
  tensors.back().shape = {576};
  write_safetensors(fixture.path.string(), tensors);
  SafeTensors st;
  st.open(fixture.path.string());
  CHECK(is_h3_image_transformer(st));
  CHECK(std::string(checkpoint_model_profile(st).family) == "h3_image");
  const auto model = dit::resolve_model_descriptor(st);
  CHECK(model.family == "h3_image");
  CHECK(model.compatibility_architecture == dit::TransformerArchitecture::kH3Image);
  CHECK(dit::is_pruned_table_architecture(model.compatibility_architecture));
  GenerateRequest r;
  r.transformer_path = fixture.path.string();
  CHECK(request_model_family(r) == "h3_image");
  CHECK(resolve_plan(r).checkpoint_available);
  r.model_family = "h3";
  CHECK(rejects(r));
  st.close();
  write_safetensors(fixture.path.string(), tensors,
                    {{"config", R"({"transformer":{"pdd_steps":8}})"}});
  st.open(fixture.path.string());
  bool bad_metadata = false;
  try { (void)dit::resolve_model_descriptor(st); }
  catch (const std::exception&) { bad_metadata = true; }
  CHECK(bad_metadata);
  st.close();
  tensors.pop_back(); // missing output bias must fail before inference
  write_safetensors(fixture.path.string(), tensors);
  st.open(fixture.path.string());
  bool rejected = false;
  try { (void)dit::resolve_model_descriptor(st); }
  catch (const std::exception&) { rejected = true; }
  CHECK(rejected);
}

SLOPFAB_TEST(h3_image_local_checkpoint) {
  const char* path = std::getenv("SLOPFAB_H3_IMAGE_CHECKPOINT");
  if (!path || !*path) {
    SKIP_MISSING_FIXTURE("set SLOPFAB_H3_IMAGE_CHECKPOINT to inspect a downloaded transformer");
    return;
  }
  slopfab::GenerateRequest r;
  r.transformer_path = path;
  CHECK(slopfab::request_model_family(r) == "h3_image");
  const auto p = slopfab::resolve_plan(r);
  CHECK(p.checkpoint_available && p.num_model_evaluations() == 6);
  slopfab::SafeTensors checkpoint;
  checkpoint.open(path);
  slopfab::dit::AdaLNTable table;
  table.load(checkpoint);
  CHECK(table.rank() == 8 && table.rows() == 1025);
  for (float t : p.video_timesteps)
    CHECK(table.lookup_code(t).size() == 8);
}


SLOPFAB_TEST(h3_image_pdd_heads_and_spatial_rope) {
  using namespace slopfab;
  const std::vector<float> bank{2, 3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
  TensorView tensor;
  tensor.dtype = DType::kF32;
  tensor.shape = {6, 2};
  tensor.data = reinterpret_cast<const uint8_t*>(bank.data());
  tensor.nbytes = bank.size() * sizeof(float);
  const auto fused = dit::fuse_h3_image_heads(tensor);
  CHECK(fused == (std::vector<float>{2,3,12,23,32,43,52,63,72,83,92,103}));
  for (int i = 0; i < 6; ++i)
    CHECK(dit::pdd_step(6, 1 - h3_image_sigmas()[i]) == i);
  bool rejected = false;
  try { (void)dit::pdd_step(6, .5f); } catch (const std::exception&) { rejected = true; }
  CHECK(rejected);
  const auto c = h3_image_vae_config();
  CHECK(c.dim == 1024 && c.heads == 16 && c.num_layers == 24 && c.ffn_inner == 4096);
  CHECK(c.patch_t == 1 && c.rope_dim == 32 && c.rope_dims == 2 && !c.ph3_image);
  const auto rope = vae::build_vit_rope_tables(1, 2, 3, 5, 32, 100, false, 2);
  for (int token = 0; token < 11; ++token) {
    for (int axis = 0; axis < 2; ++axis) {
      const int extent = axis == 0 ? 2 : 3;
      const int index = axis == 0 ? token / 3 : token % 3;
      for (int f = 0; f < 8; ++f) {
        const double coord = 2.f * ((float(index) + .5f) / extent) - 1.f;
        const double angle = token >= 6 ? 0 : 6.283185307179586 * coord / std::pow(100., f / 8.);
        for (int half : {0, 16}) {
          CHECK_NEAR(rope.cosine[token * 32 + half + axis * 8 + f], std::cos(angle), 1e-6);
          CHECK_NEAR(rope.sine[token * 32 + half + axis * 8 + f], std::sin(angle), 1e-6);
        }
      }
    }
  }
}

SLOPFAB_TEST(h3_image_reference_sizes_labels_and_plan) {
  using namespace slopfab;
  int h = 0, w = 0;
  h3_image_reference_size(640, 320, &h, &w);
  CHECK(h == 1024 && w == 2048);
  h3_image_reference_size(1920, 1080, &h, &w);
  CHECK(h == 1088 && w == 1920);
  h3_image_reference_size(1920, 1080, &h, &w, true);
  CHECK(h == 1024 && w == 1824);
  h3_image_reference_size(6000, 3000, &h, &w);
  CHECK(h == 2048 && w == 4096);
  h3_image_reference_size(6000, 3000, &h, &w, true);
  CHECK(h == 1024 && w == 2048);
  CHECK(h3_image_picture_label(0) == "Picture 1: ");
  CHECK(h3_image_picture_label(15) == "Picture 16: ");
  LatentFixture fixture;
  fixture.path.replace_extension(".ppm");
  {
    std::ofstream image(fixture.path, std::ios::binary);
    image << "P6\n64 32\n255\n";
    const std::string pixels(64 * 32 * 3, char(90));
    image.write(pixels.data(), pixels.size());
  }
  GenerateRequest r;
  r.model_family = "h3_image";
  r.reference_image_paths = {fixture.path.string()};
  const auto p = resolve_plan(r);
  CHECK(p.canvas_width == 2048 && p.canvas_height == 1024);
  CHECK(p.model.supports_references && p.layout.num_audio_rows == 0);
  CHECK(describe_plan(r, p).find("image editing (1 references)") != std::string::npos);
  r.canvas_width = 512; r.canvas_height = 768;
  r.reference_image_paths.assign(16, fixture.path.string());
  CHECK(resolve_plan(r).canvas_width == 512);
  CHECK(resolve_plan(r).num_model_evaluations() == 6);
  CHECK(describe_plan(r, resolve_plan(r)).find("16 references") != std::string::npos);
  r.reference_image_paths.push_back(fixture.path.string());
  CHECK(rejects(r));
  // Different picture labels/posterior rules cannot hit video H3's cache.
  r.reference_image_paths.clear();
  const auto text_key = conditioning_cache_key(r), ref_key = reference_cache_key(r);
  r.model_family = "h3";
  CHECK(text_key != conditioning_cache_key(r));
  CHECK(ref_key != reference_cache_key(r));
}

SLOPFAB_TEST(h3_image_multi_reference_positions_and_posterior) {
  using namespace slopfab;
  std::vector<dit::ReferenceGeometry> refs;
  for (int i = 0; i < 16; ++i) refs.push_back({dit::ReferenceKind::kImage, 1, 2, 4, 0});
  const auto packed = dit::build_ref2va_packed_sequence({1,0,1}, refs, 1, 4, 4, 0);
  CHECK(packed.layout.num_condition_video == 32 && packed.layout.num_video_rows == 4);
  CHECK(packed.indices.audio.empty() && packed.layout.num_audio_rows == 0);
  for (int i = 0; i < 16; ++i) {
    CHECK(packed.position_ids[(3 + 2 * i) * 3] == 3 + i);
    CHECK(packed.position_ids[(4 + 2 * i) * 3] == 3 + i);
  }
  CHECK(packed.position_ids[35 * 3] == 19);
  const auto times = dit::build_row_timesteps(packed.layout, packed.indices, .25f, .25f, .999f, 1.f);
  for (int row = 3; row < 35; ++row)
    CHECK(times.unique[times.indices[row]] == .999f);
  CHECK(times.unique[times.indices[35]] == .25f);
  std::vector<float> moments(48, 0), normal(24, 0), mean(24, 0), stddev(24, 1);
  moments[0] = 1.0001f;
  const auto image = vae::sample_keyframe_latents(moments.data(), normal.data(), 1, 1, mean, stddev, false);
  const auto video = vae::sample_keyframe_latents(moments.data(), normal.data(), 1, 1, mean, stddev);
  CHECK(image[0] == moments[0] && video[0] != image[0]);
  CHECK(vae::torch_cpu_normal(96, 42) == vae::torch_cpu_normal_seed42(96));
  CHECK(vae::torch_cpu_normal(96, 43) != vae::torch_cpu_normal(96, 42));
}

SLOPFAB_TEST(h3_image_outpaint_canvas_and_fixed_schedule) {
  using namespace slopfab;
  RGBImage source;
  source.width = 65; source.height = 63;
  source.pixels.assign(65 * 63 * 3, 123);
  GenerateRequest r;
  r.model_family = "h3_image";
  r.image_edit = make_outpaint_edit(source, 32, 17, 31, 16);
  CHECK(rejects(r)); // General H3 Langevin default is incompatible with six PDD evaluations.
  r.image_edit.langevin_steps = 0;
  const auto p = resolve_plan(r);
  CHECK(p.canvas_width == 128 && p.canvas_height == 96);
  CHECK(p.layout.num_audio_rows == 0 && p.layout.num_latent_frames == 1);
  CHECK(p.video_sigmas == h3_image_sigmas() && p.num_model_evaluations() == 6);
  CHECK(describe_plan(r, p).find("outpainting") != std::string::npos);
  CHECK(outpaint_source_image(r.image_edit).pixels == source.pixels);
  const auto anchor = outpaint_keyframe_geometry(r.image_edit);
  CHECK(anchor.target_aligned && anchor.target_latent_x == 2 && anchor.target_latent_y == 2);
  r.reference_image_paths.assign(15, "unused.png");
  CHECK(resolve_plan(r).canvas_width == 128); // Never derive canvas from extra pictures.
  r.reference_image_paths.push_back("unused.png"); CHECK(rejects(r));
  r.reference_image_paths.clear();
  r.canvas_width = r.canvas_height = 512; CHECK(rejects(r));
  r.canvas_width = 128; r.canvas_height = 96; CHECK(!rejects(r));
  r.image_edit.strength = .5f; CHECK(rejects(r));
  r.image_edit.strength = 1;
  r.image_edit.invert_mask = false; CHECK(rejects(r));
}
