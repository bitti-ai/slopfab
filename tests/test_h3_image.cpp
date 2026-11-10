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
  r = base; r.reference_image_paths = {"unused.png"}; CHECK(rejects(r));
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
  tensors.pop_back(); // missing output projection must fail before inference
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
