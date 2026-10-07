#include "detail/vulkan_fixture.h"
#include "slopfab/sampler/noise.h"

SLOPFAB_TEST_CATEGORY(vulkan_animate_pinned_audio_boundaries, "synthetic") {
  using namespace slopfab;
  using namespace slopfab::vulkan;
  if (!Instance::available()) {
    SKIP_UNSUPPORTED_HARDWARE("Vulkan unavailable");
    return;
  }
  auto instance = Instance::create();
  const auto physical = instance.enumerate_devices();
  if (physical.empty()) {
    SKIP_UNSUPPORTED_HARDWARE("No Vulkan device");
    return;
  }
  const auto& info = physical.front().info();
  if (!info.cooperative_matrix_bf16_f32_16x16x16) {
    SKIP_UNSUPPORTED_HARDWARE("Exact attention unavailable");
    return;
  }
  DeviceOptions opts;
  opts.enable_timeline_semaphore = opts.enable_shader_int64 = true;
  opts.enable_shader_float16 = opts.enable_storage_buffer_16bit = opts.enable_cooperative_matrix =
      true;
  auto device = physical.front().create_device(opts);
  TensorContextOptions context_options;
  context_options.max_batch_operators = 128;
  TensorContext context(device, context_options);
  std::vector<TensorWrite> tensors;
  auto add = [&](std::string name, std::vector<int64_t> shape, float value = 0,
                 DType dtype = DType::kF32) {
    size_t count = 1;
    for (auto n : shape)
      count *= n;
    tensors.push_back({name, shape, std::vector<float>(count, value), dtype});
  };
  for (const std::string prefix : {"blocks.0.", "token_refiner.blocks.0."}) {
    for (const char* norm :
         {"norm1.weight", "norm2.weight", "attn.q_norm.weight", "attn.k_norm.weight"})
      add(prefix + norm, {128}, 1);
    add(prefix + "attn.qkv_proj.weight", {384, 128});
    add(prefix + "attn.out_proj.weight", {128, 128});
    add(prefix + "mlp.fc1.weight", {256, 128});
    add(prefix + "mlp.fc2.weight", {128, 128});
  }
  add("blocks.0.adaln_proj.linear.weight", {3 * 6 * 128, 8});
  add("blocks.0.adaln_proj.linear.bias", {3 * 6 * 128});
  add("adaln_t_table", {1025, 8});
  add("condition_proj.weight", {128, 16}, 0, DType::kBF16);
  add("condition_proj.bias", {128}, 0, DType::kBF16);
  add("video_patch_proj.weight", {128, 4});
  add("video_patch_proj.bias", {128});
  add("audio_patch_proj.weight", {128, 2});
  add("audio_patch_proj.bias", {128});
  add("token_refiner.final_norm.weight", {128}, 1, DType::kBF16);
  add("final_layer.norm.weight", {128}, 1, DType::kBF16);
  add("final_layer.adaln_proj.linear.weight", {256, 8}, 0, DType::kF16);
  add("final_layer.adaln_proj.linear.bias", {256}, 0, DType::kF16);
  add("final_layer.video_out.weight", {4, 128});
  add("final_layer.video_out.bias", {4}, .25f);
  add("final_layer.audio_out.weight", {2, 128});
  add("final_layer.audio_out.bias", {2}, 10);
  const auto path = std::filesystem::temp_directory_path() / "slopfab_pinned_audio.safetensors";
  write_safetensors(path.string(), tensors);
  SafeTensors checkpoint;
  checkpoint.open(path.string());
  const auto packed = dit::build_ref2va_packed_sequence(
      {1, 1, 1},
      {{dit::ReferenceKind::kVideo, 7, 4, 4, 0}, {dit::ReferenceKind::kImage, 1, 4, 4, 0}}, 1, 4, 4,
      1);
  ExactH3DenoiseConfig config;
  config.layout = packed.layout;
  config.indices = packed.indices;
  config.position_ids = packed.position_ids;
  config.pin_target_audio = true;
  auto& t = config.transformer;
  t.main.layers = 1;
  t.main.block.hidden = 128;
  t.main.block.heads = 1;
  t.main.block.head_dim = 128;
  t.main.block.ffn = 128;
  t.main.block.sequence = packed.layout.total_rows();
  t.main.block.timesteps = 4;
  t.text_rows = 3;
  t.text_dim = 16;
  t.video_dim = 4;
  t.audio_dim = 2;
  t.refiner_layers = 1;
  t.video_rows = static_cast<uint32_t>(packed.indices.video.size());
  t.audio_rows = 2;
  t.video_output_rows = packed.layout.num_video_rows;
  t.audio_output_rows = 2;
  t.video_output_start = packed.layout.video_start();
  t.audio_output_start = packed.layout.audio_start();
  auto model = ExactH3Denoiser::create(context, config);
  model.load(checkpoint);
  const std::vector<float> prompt(3 * 16, .5f), video(t.video_rows * 4, .5f),
      audio{.1f, .2f, .3f, .4f};
  model.prepare(prompt.data(), prompt.size(), video.data(), video.size(), audio.data(),
                audio.size());
  sampler::FlowScheduler vs(3), as(3);
  vs.set_timesteps(4);
  as.set_timesteps(4);
  int boundaries = 0;
  const auto result =
      model.run(vs, as, {}, [&](uint32_t, const std::vector<float>&, const std::vector<float>& a) {
        ++boundaries;
        CHECK(a == audio);
      });
  CHECK(boundaries == 3 && result.steps_completed == 3);
  CHECK(result.audio_rows == audio);
  CHECK(result.video_rows != std::vector<float>(t.video_output_rows * 4, .5f));
  model.unload();
  // Reuse the constant-velocity fixture to check both conditioned and plain
  // target storage. Preserved cells follow the source noise trajectory;
  // editable cells retain the actual GPU Euler update.
  for (bool conditioned : {true, false}) {
    auto edit_config = config;
    edit_config.pin_target_audio = false;
    if (!conditioned) {
      auto& l = edit_config.layout;
      l.num_condition_video = l.num_condition_audio = 0;
      l.condition_audio_is_explicit = false;
      edit_config.indices = dit::build_indices(l);
      edit_config.position_ids = dit::build_position_ids(l);
      auto& et = edit_config.transformer;
      et.main.block.sequence = l.total_rows();
      et.main.block.timesteps = 2;
      et.video_rows = l.num_video_rows;
      et.video_output_rows = et.audio_output_rows = 0;
      et.video_output_start = et.audio_output_start = 0;
    }
    auto constraint = std::make_shared<InpaintConstraint>();
    const size_t n = size_t(edit_config.layout.num_video_rows) * 4;
    constraint->original.assign(n, 2);
    constraint->noise.assign(n, 10);
    constraint->mask.assign(n, 0);
    constraint->mask[1] = 1;
    edit_config.inpaint = constraint;
    auto editor = ExactH3Denoiser::create(context, edit_config);
    editor.load(checkpoint);
    std::vector<float> initial(edit_config.transformer.video_rows * 4, .5f);
    editor.prepare(prompt.data(), prompt.size(), initial.data(), initial.size(), audio.data(),
                   audio.size());
    vs.set_sigmas({.75f, .5f, .25f, 0});
    as.set_sigmas({.75f, .5f, .25f, 0});
    int updates = 0;
    const auto edited = editor.run(
        vs, as, {}, [&](uint32_t step, const std::vector<float>& v, const std::vector<float>&) {
          ++updates;
          CHECK(v[0] == 2 + 8 * vs.sigmas()[step + 1]);
          CHECK_NEAR(v[1], 8 + .0625f * (step + 1), 1e-5);
        });
    CHECK(updates == 3 && edited.video_rows[0] == 2);
    editor.prepare(prompt.data(), prompt.size(), initial.data(), initial.size(), audio.data(),
                   audio.size());
    const auto cancelled = editor.run(vs, as, [](uint32_t, uint32_t) {
      return false;
    });
    CHECK(cancelled.cancelled && cancelled.steps_completed == 1);
    CHECK(cancelled.video_rows[0] == 6);
    editor.unload();

    // Constant GPU velocities provide an independent oracle for the re-noise
    // dispatch, including different video/audio sigmas and target row slices.
    edit_config.inpaint.reset();
    edit_config.seed = 42;
    auto student = ExactH3Denoiser::create(context, edit_config);
    student.load(checkpoint);
    student.prepare(prompt.data(), prompt.size(), initial.data(), initial.size(), audio.data(),
                    audio.size());
    vs = sampler::FlowScheduler(12);
    as = sampler::FlowScheduler(2);
    vs.set_timesteps(5);
    as.set_timesteps(5);
    vs.set_sampler(sampler::SamplerKind::kRenoise);
    as.set_sampler(sampler::SamplerKind::kRenoise);
    std::vector<float> expected_video(n, .5f), expected_audio = audio;
    const auto check_boundary = [&](uint32_t step, const std::vector<float>& v,
                                    const std::vector<float>& a) {
      const auto advance = [&](std::vector<float>& x, const sampler::FlowScheduler& schedule,
                               sampler::NoiseStream stream, float velocity) {
        std::vector<float> noise(x.size());
        sampler::fill_renoise_normal(42, step, stream, noise.data(), noise.size());
        const float sigma = schedule.sigmas()[step], next = schedule.sigmas()[step + 1];
        for (size_t i = 0; i < x.size(); ++i)
          x[i] = (1 - next) * (x[i] + sigma * velocity) + next * noise[i];
      };
      advance(expected_video, vs, sampler::NoiseStream::kVideoLatents, .25f);
      advance(expected_audio, as, sampler::NoiseStream::kAudioLatents, 10);
      CHECK_CLOSE(expected_video, v, 1e-6, "Vulkan DMAD video formula");
      CHECK_CLOSE(expected_audio, a, 1e-6, "Vulkan DMAD audio formula");
    };
    const auto generated = student.run(vs, as, {}, check_boundary);
    CHECK(generated.steps_completed == 4);
    student.prepare(prompt.data(), prompt.size(), initial.data(), initial.size(), audio.data(),
                    audio.size());
    const auto repeated = student.run(vs, as);
    CHECK(generated.video_rows == repeated.video_rows);
    CHECK(generated.audio_rows == repeated.audio_rows);
    student.unload();

    // Exercise fresh-noise preservation with both target storage layouts and
    // generated audio, which reuses the denoiser's scratch noise buffer.
    for (bool invert : {false, true}) {
      for (const std::vector<float>& sigmas :
           {std::vector<float>{1, .75f, .5f, .25f, 0},
            std::vector<float>{.5f, .25f, 0}, std::vector<float>{.25f, 0}}) {
        std::fill(constraint->mask.begin(), constraint->mask.end(), invert ? 1.f : 0.f);
        constraint->mask[1] = invert ? 0.f : 1.f;
        edit_config.inpaint = constraint;
        auto renoise_editor = ExactH3Denoiser::create(context, edit_config);
        renoise_editor.load(checkpoint);
        vs.set_base_sigmas(sigmas);
        as.set_base_sigmas(sigmas);
        auto expected = constraint->initial(vs.sigmas().front());
        int edit_boundaries = 0;
        const auto edit_boundary = [&](uint32_t step, const std::vector<float>& v,
                                        const std::vector<float>&) {
          ++edit_boundaries;
          std::vector<float> noise(n);
          sampler::fill_renoise_normal(42, step, sampler::NoiseStream::kVideoLatents,
                                       noise.data(), n);
          const float sigma = vs.sigmas()[step], next = vs.sigmas()[step + 1];
          for (size_t i = 0; i < n; ++i) {
            const float clean = constraint->mask[i] ? expected[i] + sigma * .25f : 2.f;
            expected[i] = (1 - next) * clean + next * noise[i];
          }
          CHECK_CLOSE(expected, v, 1e-6, "Vulkan DMAD edit boundary");
        };
        const auto prepare = [&] {
          renoise_editor.prepare(prompt.data(), prompt.size(), initial.data(), initial.size(),
                                   audio.data(), audio.size());
          expected = constraint->initial(vs.sigmas().front());
          edit_boundaries = 0;
        };
        prepare();
        const auto renoise_edited = renoise_editor.run(vs, as, {}, edit_boundary);
        CHECK(edit_boundaries == int(vs.num_steps()));
        prepare();
        const auto repeated_edit = renoise_editor.run(vs, as, {}, edit_boundary);
        CHECK(renoise_edited.video_rows == repeated_edit.video_rows);
        CHECK(renoise_edited.audio_rows == repeated_edit.audio_rows);
        prepare();
        const auto cancelled_edit = renoise_editor.run(
            vs, as, [](uint32_t, uint32_t) { return false; }, edit_boundary);
        CHECK(cancelled_edit.cancelled && edit_boundaries == 1);
        renoise_editor.unload();
      }
    }
    vs.set_sampler(sampler::SamplerKind::kEuler);
    as.set_sampler(sampler::SamplerKind::kEuler);
  }
  checkpoint.close();
  std::filesystem::remove(path);
}
