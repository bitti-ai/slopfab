#include "harness.h"
#include "latent_fixture.h"
#include "slopfab/ph3_image.h"
#include "slopfab/dit/adaln.h"
#include "slopfab/model_profile.h"
#include "slopfab/generate.h"
#include "slopfab/safetensors_write.h"
#include "slopfab/text/clip_projection.h"
#include "slopfab/tensor_convert.h"
#include <cstdlib>
#include <limits>

namespace {
bool rejects(const slopfab::GenerateRequest& request) {
  try { (void)slopfab::resolve_plan(request); }
  catch (const std::exception&) { return true; }
  return false;
}
}

SLOPFAB_TEST(ph3_image_plan_matches_published_workflow) {
  using namespace slopfab;
  GenerateRequest r;
  r.model_family = "ph3_image";
  const auto p = resolve_plan(r);
  CHECK(p.model.family == "ph3_image" && !p.checkpoint_available);
  CHECK(model_profile("ph3_image").image_only);
  CHECK(model_profile("ph3_image").conditioner_width == 5120);
  CHECK(p.canvas_width == 2912 && p.canvas_height == 2912);
  CHECK(p.layout.num_latent_frames == 1 && p.aligned_frames == 1);
  CHECK(p.layout.num_video_rows == 91 * 91);
  CHECK(p.layout.num_audio_latents == 2 && p.layout.num_audio_rows == 4);
  CHECK(p.video_sigma_shift == 1 && p.audio_sigma_shift == 1);
  CHECK(p.num_inference_steps == 8 && p.num_model_evaluations() == 8);
  CHECK(p.fixed_sampling_grid && p.sampler == sampler::SamplerKind::kEuler);
  const std::vector<float> expected{1.f, .984753f, .967905f, .951688f, .925372f,
                                    .881923f, .810083f, .528718f, 0.f};
  CHECK(p.video_sigmas == expected && p.audio_sigmas == expected);
  for (size_t i = 0; i < 8; ++i) {
    CHECK_NEAR(p.video_timesteps[i], 1 - expected[i], 1e-7);
    CHECK(p.audio_timesteps[i] == p.video_timesteps[i]);
  }
  CHECK(describe_plan(r, p).find("P-H3 Image") != std::string::npos);
  r.canvas_width = 3872;
  r.canvas_height = 2176;
  r.num_inference_steps = 8;
  r.guidance_scale_set = true;
  r.guidance_scale = 1;
  r.sampling.base_sigmas = expected;
  CHECK(resolve_plan(r).layout.num_video_rows == 121 * 68);
}

SLOPFAB_TEST(ph3_image_rejects_incompatible_requests) {
  using namespace slopfab;
  GenerateRequest base;
  base.model_family = "ph3_image";
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

SLOPFAB_TEST(ph3_image_checkpoint_identity_and_shape_validation) {
  using namespace slopfab;
  LatentFixture fixture; // an unrelated filename must not affect recognition
  std::vector<TensorWrite> tensors;
  for (const auto& item : std::vector<std::pair<std::string, std::vector<int64_t>>>{
           {"adaln_t_table", {4097, 16}}, {"video_patch_proj.weight", {5376, 96}},
           {"audio_patch_proj.weight", {5376, 32}}, {"condition_proj.weight", {5376, 5120}},
           {"blocks.0.adaln_proj.linear.weight", {96768, 16}},
           {"final_layer.adaln_proj.linear.weight", {10752, 16}},
           {"final_layer.video_out.weight", {96, 5376}},
           {"final_layer.audio_out.weight", {32, 5376}}})
    tensors.push_back({item.first, item.second,
                      std::vector<float>(size_t(item.second[0]) * item.second[1]), DType::kBF16});
  write_safetensors(fixture.path.string(), tensors);
  SafeTensors st;
  st.open(fixture.path.string());
  CHECK(is_ph3_image_transformer(st));
  CHECK(std::string(checkpoint_model_profile(st).family) == "ph3_image");
  const auto model = dit::resolve_model_descriptor(st);
  CHECK(model.family == "ph3_image");
  CHECK(model.compatibility_architecture == dit::TransformerArchitecture::kPH3Image);
  CHECK(dit::is_pruned_table_architecture(model.compatibility_architecture));
  GenerateRequest r;
  r.transformer_path = fixture.path.string();
  CHECK(request_model_family(r) == "ph3_image");
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

SLOPFAB_TEST(ph3_image_local_checkpoint) {
  const char* path = std::getenv("SLOPFAB_PH3_IMAGE_CHECKPOINT");
  if (!path || !*path) {
    SKIP_MISSING_FIXTURE("set SLOPFAB_PH3_IMAGE_CHECKPOINT to inspect a downloaded transformer");
    return;
  }
  slopfab::GenerateRequest r;
  r.transformer_path = path;
  CHECK(slopfab::request_model_family(r) == "ph3_image");
  const auto p = slopfab::resolve_plan(r);
  CHECK(p.checkpoint_available && p.num_model_evaluations() == 8);
  slopfab::SafeTensors checkpoint;
  checkpoint.open(path);
  slopfab::dit::AdaLNTable table;
  table.load(checkpoint);
  CHECK(table.rank() == 16 && table.rows() == 4097);
  for (float t : p.video_timesteps)
    CHECK(table.lookup_code(t).size() == 16);
}

SLOPFAB_TEST(ph3_image_local_text_connector) {
  const char* path = std::getenv("SLOPFAB_PH3_TEXT_ENCODER");
  if (!path || !*path) {
    SKIP_MISSING_FIXTURE("set SLOPFAB_PH3_TEXT_ENCODER to validate the downloaded connector");
    return;
  }
  using namespace slopfab;
  SafeTensors checkpoint;
  checkpoint.open(path);
  text::ClipProjection connector(path);
  CHECK(connector.ph3_image());
  CHECK(connector.hidden_taps() == (std::vector<int>{13, 19, 25, 31, 36}));
  const auto cfg = text::validate_clip_student(checkpoint, connector);
  CHECK(cfg.width == 2560 && cfg.layers == 36);
  const auto mean = to_f32(checkpoint.at("text_connector.input_mean"));
  const auto stddev = to_f32(checkpoint.at("text_connector.input_std"));
  const auto output_mean = to_f32(checkpoint.at("text_connector.output_mean"));
  const auto output_std = to_f32(checkpoint.at("text_connector.output_std"));
  const auto sink = to_f32(checkpoint.at("text_connector.sink_embedding"));
  text::PromptEmbedding h;
  h.num_tokens = 2;
  h.hidden_size = 12800;
  h.modality_tags = {1, 1};
  h.data = mean;
  h.data.insert(h.data.end(), mean.begin(), mean.end());
  // One distinct component in each of the five taps makes omitted/reordered
  // taps and a transposed projection observable in the expected output.
  const int channels[] = {7, 2569, 5131, 7697, 10261};
  for (int i = 0; i < 5; ++i) h.data[12800 + channels[i]] += (i + 1) * stddev[channels[i]];
  const auto out = connector.apply(h);
  CHECK(out.num_tokens == 2 && out.hidden_size == 5120);
  const auto* weight = static_cast<const float*>(checkpoint.at("text_connector.proj.weight").data);
  for (int row = 0; row < 5120; ++row) {
    CHECK(out.data[row] == sink[row]);
    double expected = 0;
    for (int c : channels)
      expected += weight[size_t(row) * 12800 + c] * ((h.data[12800 + c] - mean[c]) / stddev[c]);
    CHECK_NEAR(out.data[5120 + row], expected * output_std[row] + output_mean[row], 1e-4);
  }
}
