#include <cmath>
#include <limits>
#include <memory>
#include <exception>
#include <vector>

#include "harness.h"
#include "slopfab/pipeline.h"
#include "slopfab/reference_media.h"
#include "slopfab/reference_conditioning.h"
#include "slopfab/text/qwen_vision.h"

namespace {
using slopfab::ReferenceMedia;
template <typename Fn> bool throws(Fn&& fn) {
  try { fn(); } catch (const std::exception&) { return true; }
  return false;
}

ReferenceMedia clip(double duration = 2) {
  auto result = ReferenceMedia::video(duration);
  const uint8_t rgb[] = {10, 20, 30};
  result.append_frame(rgb, sizeof(rgb), 1, 1, 3, 3, 0);
  return result;
}
}  // namespace

SLOPFAB_TEST(reference_media_pixels_and_snapshot) {
  auto video = ReferenceMedia::video(2);
  // Two 1-pixel rows: padding and alpha are deliberately distinct. The last
  // row needs only its visible bytes, not trailing row padding.
  uint8_t rgba[] = {1, 2, 3, 99, 88, 77, 4, 5, 6, 66};
  video.append_frame(rgba, sizeof(rgba), 1, 2, 6, 4, 0);
  CHECK(video.frames()[0]->image.pixels == std::vector<uint8_t>({1, 2, 3, 4, 5, 6}));
  const auto snapshot = std::make_shared<const ReferenceMedia>(video);
  const auto before = slopfab::reference_media_identity(*snapshot);
  rgba[0] = 100;
  video.append_frame(rgba, sizeof(rgba), 1, 2, 6, 4, 1);
  CHECK(snapshot->frames().size() == 1);
  CHECK(snapshot->frames()[0]->image.pixels[0] == 1);
  CHECK(slopfab::reference_media_identity(*snapshot) == before);
  CHECK(slopfab::reference_media_identity(video) != before);

  auto rgb_video = ReferenceMedia::video(2);
  const uint8_t rgb[] = {1, 2, 3, 4, 5, 6};
  rgb_video.append_frame(rgb, sizeof(rgb), 1, 2, 3, 3, 0);
  CHECK(slopfab::reference_media_identity(rgb_video) == before);
}

SLOPFAB_TEST(reference_media_rejects_invalid_frames_transactionally) {
  auto video = clip();
  const auto before = slopfab::reference_media_identity(video);
  const uint8_t pixels[12] = {};
  CHECK(throws([&] { video.append_frame(pixels, 3, 1, 1, 3, 3, 0); }));
  CHECK(throws([&] { video.append_frame(pixels, 3, 1, 1, 3, 3, 2); }));
  CHECK(throws([&] { video.append_frame(pixels, 3, 1, 1, 3, 3, NAN); }));
  CHECK(throws([&] { video.append_frame(pixels, 2, 1, 1, 3, 3, 1); }));
  CHECK(throws([&] { video.append_frame(pixels, 3, 1, 1, 2, 3, 1); }));
  CHECK(throws([&] { video.append_frame(nullptr, 3, 1, 1, 3, 3, 1); }));
  CHECK(throws([&] { video.append_frame(pixels, 12, 2, 2, 6, 3, 1); }));
  CHECK(slopfab::reference_media_identity(video) == before);
  auto empty = ReferenceMedia::video(2);
  CHECK(throws([&] { empty.append_frame(pixels, 3, 1, 1, 3, 3, .1); }));
  CHECK(throws([&] { empty.append_frame(pixels, 12, 1, 3,
      std::numeric_limits<size_t>::max(), 3, 0); }));
  CHECK(throws([&] { empty.validate(); }));
  CHECK(throws([] { ReferenceMedia::video(INFINITY); }));
  CHECK(throws([] { ReferenceMedia::video(1.99); }));
  CHECK(throws([] { ReferenceMedia::video(15.01); }));
}

SLOPFAB_TEST(reference_media_audio_ownership_and_validation) {
  auto video = clip(3);
  std::vector<float> pcm(64000 * 2, .25f);
  video.set_audio(pcm.data(), pcm.size(), 2, 32000, .5);
  const auto snapshot = video;
  const auto before = slopfab::reference_media_identity(snapshot);
  pcm[0] = .5f;
  video.set_audio(pcm.data(), pcm.size(), 2, 32000, 0);
  CHECK(snapshot.soundtrack()->interleaved[0] == .25f);
  CHECK(snapshot.soundtrack()->start_seconds == .5);
  CHECK(slopfab::reference_media_identity(video) != before);
  const auto valid = slopfab::reference_media_identity(video);
  CHECK(throws([&] { video.set_audio(pcm.data(), pcm.size(), 2, 32000, 2); }));
  CHECK(throws([&] { video.set_audio(pcm.data(), pcm.size() - 1, 2, 32000); }));
  CHECK(throws([&] { video.set_audio(pcm.data(), pcm.size(), 3, 32000); }));
  CHECK(throws([&] { video.set_audio(pcm.data(), pcm.size(), 2, 0); }));
  CHECK(throws([&] { video.set_audio(pcm.data(), pcm.size(), 2, 32000, NAN); }));
  pcm[0] = NAN;
  CHECK(throws([&] { video.set_audio(pcm.data(), pcm.size(), 2, 32000); }));
  pcm[0] = 1.01f;
  CHECK(throws([&] { video.set_audio(pcm.data(), pcm.size(), 2, 32000); }));
  CHECK(slopfab::reference_media_identity(video) == valid);
  pcm[0] = 0;
  auto audio = ReferenceMedia::audio(pcm.data(), pcm.size(), 2, 32000);
  CHECK(!audio.is_video());
  CHECK(audio.duration_seconds() == 2);
  CHECK(throws([&] { ReferenceMedia::audio(pcm.data(), 2, 2, 32000); }));
}

SLOPFAB_TEST(reference_media_request_limits_and_cache) {
  slopfab::GenerateRequest request;
  request.reference_media.push_back(std::make_shared<const ReferenceMedia>(clip()));
  const auto plan = slopfab::resolve_plan(request);
  CHECK(slopfab::describe_plan(request, plan).find("reference videos    1") != std::string::npos);
  const auto key = slopfab::reference_cache_key(request);
  const auto conditioning = slopfab::conditioning_cache_key(request);
  request.audio_vae_path = "another-audio-checkpoint";
  CHECK(slopfab::reference_cache_key(request) != key);
  CHECK(slopfab::conditioning_cache_key(request) == conditioning);
  auto changed = clip();
  const uint8_t other[] = {10, 21, 30};
  changed.append_frame(other, 3, 1, 1, 3, 3, 1);
  request.reference_media[0] = std::make_shared<const ReferenceMedia>(changed);
  CHECK(slopfab::conditioning_cache_key(request) != conditioning);
  CHECK(slopfab::reference_cache_key(request, {}) == slopfab::reference_cache_key(request));
  CHECK(slopfab::conditioning_cache_key(request, {}) == slopfab::conditioning_cache_key(request));

  request.reference_media.assign(3, std::make_shared<const ReferenceMedia>(clip(5)));
  slopfab::resolve_plan(request);
  request.reference_media.push_back(std::make_shared<const ReferenceMedia>(clip()));
  CHECK(throws([&] { slopfab::resolve_plan(request); }));
  request.reference_media.assign(2, std::make_shared<const ReferenceMedia>(clip(8)));
  CHECK(throws([&] { slopfab::resolve_plan(request); }));
  request.reference_media.assign(3, std::make_shared<const ReferenceMedia>(clip()));
  request.reference_image_paths.assign(9, "image.ppm");
  slopfab::resolve_plan(request);
  const std::vector<float> pcm(64000, 0);
  auto audio = std::make_shared<const ReferenceMedia>(ReferenceMedia::audio(pcm.data(), pcm.size(), 1, 32000));
  request.reference_media.push_back(audio);
  CHECK(throws([&] { slopfab::resolve_plan(request); }));
  request.reference_image_paths.clear();
  request.reference_media.assign(1, audio);
  CHECK(throws([&] { slopfab::resolve_plan(request); }));
  request.reference_image_paths.push_back("image.ppm");
  slopfab::resolve_plan(request);
  request.reference_media.push_back(nullptr);
  CHECK(throws([&] { slopfab::resolve_plan(request); }));
}

SLOPFAB_TEST(encoded_media_cache_identity_and_seed) {
  using Authority = slopfab::ReferenceEncoderAuthority;
  slopfab::GenerateRequest request;
  request.reference_media.push_back(std::make_shared<const ReferenceMedia>(clip()));
  const auto key = slopfab::media_encoding_cache_key(request, Authority::kCudaFp32);
  request.seed += 1;
  request.prompt = "a different prompt";
  request.reference_image_paths.push_back("irrelevant-image.ppm");
  CHECK(slopfab::media_encoding_cache_key(request, Authority::kCudaFp32) == key);
  CHECK(slopfab::media_encoding_cache_key(request, Authority::kCudaFp16) != key);
  CHECK(slopfab::media_encoding_cache_key(request, Authority::kVulkanFp32) != key);
  auto changed = request;
  changed.num_frames += 17;
  CHECK(slopfab::media_encoding_cache_key(changed, Authority::kCudaFp32) != key);
  changed = request;
  changed.video_vae_path = "other-video";
  CHECK(slopfab::media_encoding_cache_key(changed, Authority::kCudaFp32) != key);
  changed = request;
  changed.audio_vae_path = "other-audio";
  CHECK(slopfab::media_encoding_cache_key(changed, Authority::kCudaFp32) != key);
  changed = request;
  changed.reference_media.push_back(std::make_shared<const ReferenceMedia>(clip(3)));
  CHECK(slopfab::media_encoding_cache_key(changed, Authority::kCudaFp32) != key);

  slopfab::EncodedReferenceCondition encoded;
  encoded.geometry = {slopfab::dit::ReferenceKind::kVideo, 7, 2, 2, 1};
  encoded.video_rows.assign(7 * 96, .25f);
  encoded.audio_rows.assign(2 * 32, .5f);
  slopfab::EncodedMediaCache cache;
  CHECK(cache.find(key) == nullptr);
  cache.store(key, {encoded});
  CHECK(cache.find("miss") == nullptr);
  CHECK(cache.find(key) != nullptr);
  std::vector<float> v1, v2, v3, a1, a2, a3;
  const auto& hit = cache.find(key)->front();
  slopfab::append_encoded_reference_condition(hit, 42, 0, v1, a1);
  slopfab::append_encoded_reference_condition(hit, 43, 0, v2, a2);
  slopfab::append_encoded_reference_condition(hit, 42, 0, v3, a3);
  CHECK(v1 != v2);
  CHECK(v1 == v3);
  CHECK(a1 == encoded.audio_rows && a2 == a1 && a3 == a1);
  CHECK(hit.video_rows == encoded.video_rows);
  cache.store("replacement", {encoded});
  CHECK(cache.find(key) == nullptr);
  cache.clear();
  CHECK(cache.find("replacement") == nullptr);
}

SLOPFAB_TEST(encoded_media_cache_prepares_identical_qwen_frames) {
  auto video = clip();
  std::vector<float> pcm(64000, .25f);
  video.set_audio(pcm.data(), pcm.size(), 1, 32000);
  const auto full = slopfab::prepare_reference_condition(video, 22.0 / 24);
  const auto hit = slopfab::prepare_reference_condition(video, 22.0 / 24, false);
  CHECK(hit.plan.audio_samples == full.plan.audio_samples);
  CHECK(hit.audio.empty() && !full.audio.empty());
  CHECK(hit.frames.size() == full.frames.size());
  for (size_t i = 0; i < full.frames.size(); ++i) {
    if (i % 12 == 0) {
      CHECK(hit.frames[i].pixels == full.frames[i].pixels);
      CHECK(hit.frames[i].width == full.frames[i].width);
      CHECK(hit.frames[i].height == full.frames[i].height);
    } else CHECK(hit.frames[i].pixels.empty());
  }
}

SLOPFAB_TEST(reference_conditioning_temporal_geometry_and_audio) {
  const auto video = clip(2);
  auto plan = slopfab::reference_condition_plan(video, 5);
  CHECK(plan.frames == 48);
  CHECK(plan.encoding_frames == 39);
  CHECK(plan.geometry.num_latent_frames == 12);
  plan = slopfab::reference_condition_plan(video, 22.0 / 24);
  CHECK(plan.encoding_frames == 22 && plan.geometry.num_latent_frames == 7);
  CHECK(plan.width == 768 && plan.height == 768);
  CHECK(throws([&] { slopfab::reference_condition_plan(video, 1.0 / 24); }));

  std::vector<float> pcm(64000);
  for (size_t i = 0; i < pcm.size(); ++i) pcm[i] = float(i % 100) / 100;
  auto audio = ReferenceMedia::audio(pcm.data(), pcm.size(), 1, 32000);
  auto prepared = slopfab::prepare_reference_condition(audio, .02501);
  CHECK(prepared.plan.audio_samples == 801);
  CHECK(prepared.plan.geometry.num_audio_latents == 2);
  CHECK(prepared.audio.size() == 1602);
  for (int i = 0; i < 800; ++i) {
    CHECK(prepared.audio[i] == pcm[i]);
    CHECK(prepared.audio[801 + i] == pcm[i]);
  }
  // Truncate at native sample precision, then zero-fill a fractional tail.
  CHECK(prepared.audio[800] == 0 && prepared.audio[1601] == 0);
  slopfab::GenerateRequest request;
  request.reference_media.push_back(std::make_shared<const ReferenceMedia>(video));
  request.reference_media.push_back(std::make_shared<const ReferenceMedia>(audio));
  request.num_frames = 22;
  const auto short_key = slopfab::conditioning_cache_key(request);
  auto resolved = slopfab::resolve_plan(request);
  CHECK(resolved.layout.num_condition_video == 7 * 24 * 24);
  CHECK(resolved.layout.num_condition_audio == 74);
  request.num_frames = 124;
  CHECK(slopfab::conditioning_cache_key(request) != short_key);
}

SLOPFAB_TEST(reference_video_qwen_pair_uses_both_frames) {
  std::vector<uint8_t> black(32 * 32 * 3, 0), white(black.size(), 255);
  auto pair = slopfab::text::qwen3vl_patchify_resized_rgb_pair(black, white, 32, 32);
  CHECK(pair.rows.size() == 4 * 1536);
  CHECK(pair.rows[0] == -1);
  CHECK(pair.rows[255] == -1);
  CHECK(pair.rows[256] == 1);
  CHECK(pair.rows[511] == 1);
  auto still = slopfab::text::qwen3vl_patchify_resized_rgb(black, 32, 32);
  auto duplicated = slopfab::text::qwen3vl_patchify_resized_rgb_pair(black, black, 32, 32);
  CHECK(still.rows == duplicated.rows);
  const auto ids = slopfab::text::qwen3vl_image_block({10}, 1, 151652, 151656, 151653);
  auto multimodal = slopfab::text::qwen3vl_multimodal_plan(ids, {pair.grid});
  CHECK(multimodal.image_rows == std::vector<int32_t>({2}));
  std::vector<float> latents(24 * 2 * 2 * 2);
  for (size_t i = 0; i < latents.size(); ++i) latents[i] = float(i);
  auto rows = slopfab::patchify_reference_video(latents.data(), 2, 2, 2);
  CHECK(rows[0] == 0 && rows[4] == 8 && rows[96] == 4 && rows[100] == 12);
}

SLOPFAB_TEST(animate_geometry_order_and_audio_contract) {
  auto video = ReferenceMedia::video(2);
  const std::vector<uint8_t> pixels(64 * 96 * 3, 80);
  video.append_frame(pixels.data(), pixels.size(), 64, 96, 64 * 3, 3, 0);
  std::vector<float> pcm(32000, .25f);
  video.set_audio(pcm.data(), pcm.size(), 1, 32000);
  slopfab::GenerateRequest r;
  r.animate = true;
  r.num_inference_steps = 4;
  r.num_frames = 39;
  r.reference_image_paths = {"repainted.png"};
  r.reference_media = {std::make_shared<const ReferenceMedia>(video)};
  const auto p = slopfab::resolve_plan(r);
  CHECK(p.canvas_width == 64 && p.canvas_height == 96);
  CHECK(p.video_sigma_shift == 3);
  CHECK(p.layout.num_condition_audio == 0);
  CHECK(p.layout.num_audio_rows == 130);
  const auto options = slopfab::animate_reference_options(p.canvas_width, p.canvas_height);
  const auto prepared = slopfab::prepare_reference_condition(video, p.duration_seconds, true, options);
  CHECK(prepared.plan.width == 64 && prepared.plan.height == 96);
  CHECK(prepared.plan.encoding_frames == 39);
  CHECK(prepared.audio.empty());
  CHECK(prepared.plan.geometry.num_audio_latents == 0);
  int h, w;
  slopfab::dit::resolve_reference_image_size(300, 450, &h, &w, options.short_edge);
  CHECK(w == 64 && h == 96);

  std::vector<slopfab::dit::ReferenceGeometry> geometry = {
      {slopfab::dit::ReferenceKind::kImage, 1, 6, 4, 0}, prepared.plan.geometry};
  const size_t image_values = size_t(geometry.front().video_rows()) * 96;
  const size_t video_values = size_t(geometry.back().video_rows()) * 96;
  std::vector<float> rows(image_values, 1);
  rows.insert(rows.end(), video_values, 2);
  slopfab::order_animate_references(geometry, rows);
  CHECK(geometry.front().kind == slopfab::dit::ReferenceKind::kVideo);
  CHECK(std::vector<float>(rows.begin(), rows.begin() + video_values) == std::vector<float>(video_values, 2));
  CHECK(std::vector<float>(rows.begin() + video_values, rows.end()) == std::vector<float>(image_values, 1));
  const auto packed = slopfab::dit::build_ref2va_packed_sequence(std::vector<int32_t>(362, 1),
      geometry, p.layout.num_latent_frames, 6, 4, p.layout.num_audio_latents);
  CHECK(packed.position_ids[size_t(packed.indices.video.front()) * 3] == 362);
  CHECK(packed.position_ids[size_t(packed.indices.video[geometry.front().video_rows()]) * 3] > 362);
  CHECK(packed.layout.num_condition_audio == 0);
  CHECK(packed.layout.num_audio_rows == p.layout.num_audio_rows);

  const auto key = slopfab::media_encoding_cache_key(r, slopfab::ReferenceEncoderAuthority::kCudaFp32);
  r.preserve_driving_audio = true;
  CHECK(key != slopfab::media_encoding_cache_key(r, slopfab::ReferenceEncoderAuthority::kCudaFp32));
  CHECK(slopfab::resolve_plan(r).layout.num_condition_audio == 0);
  const auto image_key = slopfab::reference_cache_key(r);
  r.canvas_width = 128; r.canvas_height = 192;
  CHECK(image_key != slopfab::reference_cache_key(r));
  r.reference_image_paths.clear();
  CHECK(throws([&] { slopfab::resolve_plan(r); }));
  r.reference_image_paths = {"repainted.png"};
  r.reference_media = {std::make_shared<const ReferenceMedia>(clip())};
  CHECK(throws([&] { slopfab::resolve_plan(r); }));
}

SLOPFAB_TEST(animate_target_audio_padding_and_channel_crop) {
  auto video = clip();
  const std::vector<float> pcm(32000, .25f);
  video.set_audio(pcm.data(), pcm.size(), 1, 32000);
  const auto wav = slopfab::prepare_target_audio(*video.soundtrack(), 39);
  CHECK(wav.size() == 2 * 53333);
  for (int channel = 0; channel < 2; ++channel) {
    CHECK(wav[channel * 53333] == .25f);
    CHECK(wav[channel * 53333 + 31999] == .25f);
    CHECK(wav[channel * 53333 + 32000] == 0);
    CHECK(wav[channel * 53333 + 53332] == 0);
  }
  std::vector<float> encoded(2 * 4 * 32);
  for (size_t i = 0; i < encoded.size(); ++i) encoded[i] = float(i);
  const auto rows = slopfab::target_audio_rows(encoded, 3);
  CHECK(rows.size() == 2 * 3 * 32);
  CHECK(rows[95] == 95);
  CHECK(rows[96] == 128);  // Right channel starts after all four left latents.
  CHECK(rows.back() == 223);
  CHECK(throws([&] { slopfab::target_audio_rows(encoded, 5); }));
}
