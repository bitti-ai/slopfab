#include "detail/transformer_fixture.h"

SLOPFAB_TEST_CATEGORY(transformer_real_qkv_is_contiguous, "checkpoint") {
  const std::string path = find_checkpoint();
  if (path.empty()) {
    SKIP_MISSING_FIXTURE("  transformer checkpoint not present; skipping\n");
    return;
  }
  slopfab::SafeTensors st;
  st.open(path);

  const int inner = 7168;
  const int hidden = 5376;
  const int head_dim = 128;
  const int heads = inner / head_dim;

  for (int block : {0, 49}) {
    const std::string name = "blocks." + std::to_string(block) + ".attn.qkv_proj.weight";
    const slopfab::TensorView& w = st.at(name);
    CHECK(w.dtype == slopfab::DType::kF8E4M3);
    CHECK(w.shape == std::vector<int64_t>({3 * inner, hidden}));
    const uint8_t* raw = static_cast<const uint8_t*>(w.data);

    // Every 8th row, as spec 10.3 measured it. The scale cancels in the
    // comparison, so the fp8 codes are used directly.
    auto mean_abs = [&](int part, bool interleaved) {
      double acc = 0.0;
      size_t n = 0;
      for (int r = 0; r < 3 * inner; r += 8) {
        const int which = interleaved ? (r / head_dim) % 3 : r / inner;
        if (which != part) continue;
        const uint8_t* row = raw + static_cast<size_t>(r) * hidden;
        for (int c = 0; c < hidden; ++c) acc += std::fabs(slopfab::f8_e4m3_to_f32(row[c]));
        n += static_cast<size_t>(hidden);
      }
      return n == 0 ? 0.0 : acc / static_cast<double>(n);
    };

    double contiguous[3];
    double interleaved[3];
    for (int p = 0; p < 3; ++p) {
      contiguous[p] = mean_abs(p, false);
      interleaved[p] = mean_abs(p, true);
    }
    const double c_spread =
        (*std::max_element(contiguous, contiguous + 3) -
         *std::min_element(contiguous, contiguous + 3)) /
        *std::max_element(contiguous, contiguous + 3);
    const double i_spread =
        (*std::max_element(interleaved, interleaved + 3) -
         *std::min_element(interleaved, interleaved + 3)) /
        *std::max_element(interleaved, interleaved + 3);
    std::printf("  block %2d contiguous [q,k,v] = %.3f, %.3f, %.3f (spread %.3f)\n", block,
                contiguous[0], contiguous[1], contiguous[2], c_spread);
    std::printf("           interleaved        = %.3f, %.3f, %.3f (spread %.3f)\n",
                interleaved[0], interleaved[1], interleaved[2], i_spread);
    CHECK_MSG(c_spread > 5.0 * i_spread,
              "block %d: the contiguous partition (spread %.4f) does not separate more than the "
              "interleaved one (%.4f) — reverify spec 8.1 before trusting the qkv split",
              block, c_spread, i_spread);
    // Spec 10.3's own numbers put the interleaved spread at 6.3 % on block 49,
    // so "flat" here means an order of magnitude below the contiguous split,
    // not literally zero.
    CHECK(i_spread < 0.15);
    (void)heads;
  }
}

SLOPFAB_TEST_CATEGORY(transformer_real_checkpoint, "checkpoint") {
  const std::string path = find_checkpoint();
  if (path.empty()) {
    SKIP_MISSING_FIXTURE("  transformer checkpoint not present; skipping\n");
    return;
  }

  size_t free_before = 0, total_device = 0;
  cudaMemGetInfo(&free_before, &total_device);

  slopfab::SafeTensors st;
  st.open(path);
  CHECK_MSG(st.tensor_count() == 1082, "checkpoint has %zu tensors, expected 1082",
            st.tensor_count());

  Transformer model;
  const auto load_start = std::chrono::steady_clock::now();
  model.load(st, TransformerConfig{});
  const double load_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count();
  std::printf("  load %.2f s, resident %.3f GiB\n", load_seconds,
              static_cast<double>(model.weight_bytes()) / (1024.0 * 1024.0 * 1024.0));
  CHECK(model.weight_bytes() > 19ull * 1024 * 1024 * 1024);

  // --- spec 3.2 layout, on the real weights --------------------------------
  //
  // The `1 + scale` parameterisation centres the two scale slices away from
  // zero and leaves the shifts and gates on it. That signature lands on slots 1
  // and 4 and nowhere else, so it is a direct test of the parameter order.
  {
    const int hidden = model.config().hidden_size;
    const std::vector<float> mod = model.debug_modulation(0, {0.0f});
    CHECK(mod.size() == static_cast<size_t>(6) * 3 * hidden);

    const char* names[6] = {"shift_msa", "scale_msa", "gate_msa",
                            "shift_mlp", "scale_mlp", "gate_mlp"};
    const char* modality[3] = {"video", "text", "audio"};
    // Measured (spec 3.2): video -0.8161 / -0.8062, text -0.4275 / +0.2217,
    // audio -0.6696 / -0.9126, everything else within 0.004 of zero.
    const double want_scale[3][2] = {{-0.8161, -0.8062}, {-0.4275, 0.2217}, {-0.6696, -0.9126}};
    for (int m = 0; m < 3; ++m) {
      for (int p = 0; p < 6; ++p) {
        // Parameter-outer, then the T*3 modulation rows; T is 1 here.
        const size_t base = (static_cast<size_t>(p) * 3 + m) * hidden;
        double sum = 0.0;
        for (int i = 0; i < hidden; ++i) sum += mod[base + i];
        const double mean = sum / hidden;
        std::printf("  %-5s %-9s mean %+.4f\n", modality[m], names[p], mean);
        if (p == 1 || p == 4) {
          const double want = want_scale[m][p == 1 ? 0 : 1];
          CHECK_MSG(std::fabs(mean - want) < 0.02,
                    "%s %s mean %+.4f, expected %+.4f — the spec 3.2 parameter order is wrong",
                    modality[m], names[p], mean, want);
        } else {
          CHECK_MSG(std::fabs(mean) < 0.01,
                    "%s %s mean %+.4f should sit on zero; a scale slice has landed in a "
                    "shift or gate slot",
                    modality[m], names[p], mean);
        }
      }
    }
  }

  // --- a real forward pass --------------------------------------------------
  const bool full = full_run_requested();
  int canvas_h = 0, canvas_w = 0;
  slopfab::dit::resolve_canvas_size(16, 9, &canvas_h, &canvas_w);
  const int aligned = slopfab::dit::align_num_frames(124);

  SequenceLayout layout;
  layout.num_text = 64;
  layout.latent_height = canvas_h / 16;
  layout.latent_width = canvas_w / 16;
  layout.num_audio_latents = slopfab::dit::audio_latents_for_frames(aligned);
  layout.num_audio_rows = 2 * layout.num_audio_latents;
  // The default run uses a shorter clip so an ordinary test pass stays in the
  // minutes; SLOPFAB_TRANSFORMER_FULL=1 runs the real 124-frame geometry.
  layout.num_latent_frames = full ? slopfab::dit::video_latent_num_frames(aligned) : 4;
  layout.num_video_rows = layout.num_latent_frames * layout.rows_per_frame();

  const PackedIndices idx = slopfab::dit::build_indices(layout);
  const std::vector<double> pos = slopfab::dit::build_position_ids(layout);
  std::printf("  sequence %d rows (%d video, %d audio, %d text), activations %.3f GiB\n",
              layout.total_rows(), layout.num_video_rows, layout.num_audio_rows, layout.num_text,
              static_cast<double>(model.activation_bytes(layout)) / (1024.0 * 1024.0 * 1024.0));

  const std::vector<float> prompt =
      make_data(static_cast<size_t>(layout.num_text) * 5120, 7, 1.0f);
  model.prepare_text(prompt.data(), layout.num_text);
  model.prepare_sequence(layout, idx, pos);

  const std::vector<float> video_rows =
      make_data(idx.video.size() * 96, 8, 1.0f);
  const std::vector<float> audio_rows = make_data(idx.audio.size() * 32, 9, 1.0f);
  std::vector<float> video_velocity(video_rows.size());
  std::vector<float> audio_velocity(audio_rows.size());

  const RowTimesteps rt = slopfab::dit::build_row_timesteps(layout, idx, 0.5f, 0.35f);
  const auto step_start = std::chrono::steady_clock::now();
  model.forward(video_rows.data(), audio_rows.data(), rt, video_velocity.data(),
                audio_velocity.data());
  const double step_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - step_start)
          .count();

  size_t free_after = 0;
  cudaMemGetInfo(&free_after, &total_device);
  std::printf("  forward %.1f ms, peak device %.3f GiB of %.3f GiB\n", step_ms,
              static_cast<double>(free_before - free_after) / (1024.0 * 1024.0 * 1024.0),
              static_cast<double>(total_device) / (1024.0 * 1024.0 * 1024.0));

  CHECK(all_finite(video_velocity));
  CHECK(all_finite(audio_velocity));
  const double video_rms = rms(video_velocity);
  const double audio_rms = rms(audio_velocity);
  std::printf("  velocity rms: video %.4f, audio %.4f\n", video_rms, audio_rms);
  // A rectified-flow velocity against unit-variance latents lives within an
  // order of magnitude of 1. Zero would mean a dead residual stream; hundreds
  // would mean a missing normalisation.
  CHECK_MSG(video_rms > 0.02 && video_rms < 50.0, "video velocity rms %.4f is implausible",
            video_rms);
  CHECK_MSG(audio_rms > 0.02 && audio_rms < 50.0, "audio velocity rms %.4f is implausible",
            audio_rms);

  // Determinism at the model level: the same inputs must give the same bytes.
  {
    std::vector<float> again(video_rows.size());
    std::vector<float> again_audio(audio_rows.size());
    model.forward(video_rows.data(), audio_rows.data(), rt, again.data(), again_audio.data());
    CHECK_CLOSE(video_velocity, again, 0.0, "repeated forward is bitwise identical");
  }

  if (!full) {
    SKIP_OPT_IN("set SLOPFAB_TRANSFORMER_FULL=1 for the 124-frame geometry and a 49-step loop");
    return;
  }

  // --- the whole loop -------------------------------------------------------
  slopfab::sampler::FlowScheduler video_sched(12.0f), audio_sched(3.0f);
  video_sched.set_timesteps(50);
  audio_sched.set_timesteps(50);
  CHECK(video_sched.timesteps().size() == 49);

  slopfab::dit::DenoiseInputs in;
  in.layout = &layout;
  in.indices = &idx;
  in.video_timesteps = &video_sched.timesteps();
  in.audio_timesteps = &audio_sched.timesteps();
  in.video_scheduler = &video_sched;
  in.audio_scheduler = &audio_sched;
  in.seed = 20260804;

  const auto loop_start = std::chrono::steady_clock::now();
  const slopfab::dit::DenoiseOutputs out =
      slopfab::dit::denoise(model, in, [&](int step, int total) {
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - loop_start).count();
        std::printf("  step %2d/%d  %.1f s elapsed (%.1f s/step)\n", step + 1, total, elapsed,
                    elapsed / (step + 1));
        std::fflush(stdout);
        return true;
      });
  CHECK(all_finite(out.video_rows));
  CHECK(all_finite(out.audio_rows));
  std::printf("  49-step denoise: video rms %.4f, audio rms %.4f\n", rms(out.video_rows),
              rms(out.audio_rows));
}

SLOPFAB_TEST_CATEGORY(transformer_real_ref2va_nf4_checkpoint_load, "checkpoint") {
  const std::string path = find_ref2va_nf4_checkpoint();
  if (path.empty()) {
    SKIP_MISSING_FIXTURE("  Ref2VA NF4 transformer checkpoint not present; skipping\n");
    return;
  }

  slopfab::SafeTensors st;
  st.open(path);
  CHECK_MSG(st.tensor_count() == 1830, "NF4 checkpoint has %zu tensors, expected 1830",
            st.tensor_count());

  size_t states = 0;
  for (const auto& kv : st.tensors()) {
    if (kv.first.find(".weight.quant_state.bitsandbytes__nf4") != std::string::npos) ++states;
  }
  CHECK_MSG(states == 259, "expected 259 NF4 matrices, found %zu", states);

  Transformer model;
  bool rejected = false;
  try {
    model.load(st, TransformerConfig{});
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  CHECK(rejected);
}

SLOPFAB_TEST_CATEGORY(transformer_real_nvfp4_checkpoint, "checkpoint") {
  const std::string path = find_nvfp4_checkpoint();
  if (path.empty()) {
    SKIP_MISSING_FIXTURE("  nvfp4 transformer checkpoint not present; skipping\n");
    return;
  }

  slopfab::SafeTensors st;
  st.open(path);
  CHECK_MSG(st.tensor_count() == 1132, "nvfp4 checkpoint has %zu tensors, expected 1132",
            st.tensor_count());

  // 16 top level + 2 refiner blocks x 8 + 50 blocks x (6 + 4 linears x 4). The
  // fp8 file's 1082 differ by exactly +200 weight_scale_2 and -150 input_scale.
  size_t scale2 = 0, input_scale = 0;
  for (const auto& kv : st.tensors()) {
    if (kv.first.size() > 15 && kv.first.rfind(".weight_scale_2") == kv.first.size() - 15) ++scale2;
    if (kv.first.size() > 12 && kv.first.rfind(".input_scale") == kv.first.size() - 12) {
      ++input_scale;
    }
  }
  CHECK_MSG(scale2 == 200, "expected 200 weight_scale_2 tensors, found %zu", scale2);
  CHECK_MSG(input_scale == 0,
            "the nvfp4 transformer must carry no input_scale at all, found %zu", input_scale);

  size_t free_before = 0, total_device = 0;
  cudaMemGetInfo(&free_before, &total_device);

  Transformer model;
  const auto load_start = std::chrono::steady_clock::now();
  model.load(st, TransformerConfig{});
  const double load_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count();
  const double resident_gib =
      static_cast<double>(model.weight_bytes()) / (1024.0 * 1024.0 * 1024.0);
  std::printf("  load %.2f s, resident %.3f GiB\n", load_seconds, resident_gib);

  // Nibbles plus block scales are 9/16 of a byte per weight against fp8's one,
  // and the 200 quantised linears are most but not all of the model, so the
  // 19.60 GiB fp8 arena should land near 12.5 GiB rather than at half.
  CHECK_MSG(resident_gib > 11.0 && resident_gib < 14.0,
            "nvfp4 weights occupy %.3f GiB, expected about 12.5", resident_gib);

  int canvas_h = 0, canvas_w = 0;
  slopfab::dit::resolve_canvas_size(16, 9, &canvas_h, &canvas_w);
  const int aligned = slopfab::dit::align_num_frames(124);

  SequenceLayout layout;
  layout.num_text = 64;
  layout.latent_height = canvas_h / 16;
  layout.latent_width = canvas_w / 16;
  layout.num_audio_latents = slopfab::dit::audio_latents_for_frames(aligned);
  layout.num_audio_rows = 2 * layout.num_audio_latents;
  layout.num_latent_frames = 4;
  layout.num_video_rows = layout.num_latent_frames * layout.rows_per_frame();

  const PackedIndices idx = slopfab::dit::build_indices(layout);
  const std::vector<double> pos = slopfab::dit::build_position_ids(layout);
  const std::vector<float> prompt =
      make_data(static_cast<size_t>(layout.num_text) * 5120, 7, 1.0f);
  const std::vector<float> video_rows = make_data(idx.video.size() * 96, 8, 1.0f);
  const std::vector<float> audio_rows = make_data(idx.audio.size() * 32, 9, 1.0f);
  const RowTimesteps rt = slopfab::dit::build_row_timesteps(layout, idx, 0.5f, 0.35f);

  std::vector<float> video_velocity(video_rows.size());
  std::vector<float> audio_velocity(audio_rows.size());

  model.prepare_text(prompt.data(), layout.num_text);
  model.prepare_sequence(layout, idx, pos);
  const auto step_start = std::chrono::steady_clock::now();
  model.forward(video_rows.data(), audio_rows.data(), rt, video_velocity.data(),
                audio_velocity.data());
  const double step_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - step_start)
          .count();

  size_t free_after = 0;
  cudaMemGetInfo(&free_after, &total_device);
  std::printf("  forward %.1f ms, peak device %.3f GiB of %.3f GiB\n", step_ms,
              static_cast<double>(free_before - free_after) / (1024.0 * 1024.0 * 1024.0),
              static_cast<double>(total_device) / (1024.0 * 1024.0 * 1024.0));

  CHECK(all_finite(video_velocity));
  CHECK(all_finite(audio_velocity));
  const double video_rms = rms(video_velocity);
  const double audio_rms = rms(audio_velocity);
  std::printf("  velocity rms: video %.4f, audio %.4f\n", video_rms, audio_rms);
  CHECK_MSG(video_rms > 0.02 && video_rms < 50.0, "video velocity rms %.4f is implausible",
            video_rms);
  CHECK_MSG(audio_rms > 0.02 && audio_rms < 50.0, "audio velocity rms %.4f is implausible",
            audio_rms);

  // --- against the fp8 build of the same weights ----------------------------
  const std::string fp8_path = find_checkpoint();
  if (fp8_path.empty()) {
    SKIP_MISSING_FIXTURE("  fp8 transformer not present; skipping the cross-check\n");
    return;
  }
  // 19.6 GiB and 12.5 GiB do not fit on one card together, so the nvfp4 model
  // is released before the fp8 one is loaded.
  model.unload();

  slopfab::SafeTensors fp8;
  fp8.open(fp8_path);
  Transformer reference;
  reference.load(fp8, TransformerConfig{});
  std::vector<float> ref_video(video_rows.size());
  std::vector<float> ref_audio(audio_rows.size());
  reference.prepare_text(prompt.data(), layout.num_text);
  reference.prepare_sequence(layout, idx, pos);
  reference.forward(video_rows.data(), audio_rows.data(), rt, ref_video.data(), ref_audio.data());

  auto correlation = [](const std::vector<float>& a, const std::vector<float>& b) {
    double sa = 0.0, sb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      sa += a[i];
      sb += b[i];
    }
    const double ma = sa / a.size(), mb = sb / b.size();
    double num = 0.0, da = 0.0, db = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
      const double x = a[i] - ma, y = b[i] - mb;
      num += x * y;
      da += x * x;
      db += y * y;
    }
    return num / std::sqrt(da * db);
  };

  const double video_corr = correlation(video_velocity, ref_video);
  const double audio_corr = correlation(audio_velocity, ref_audio);
  std::printf("  vs fp8: rms video %.4f/%.4f audio %.4f/%.4f, correlation video %.5f audio %.5f\n",
              video_rms, rms(ref_video), audio_rms, rms(ref_audio), video_corr, audio_corr);

  // Fifty blocks of accumulated fp4-against-fp8 disagreement, so this is not a
  // tolerance check. What it separates is a correct dequantisation from a
  // plausible wrong one: every layout error measured on the raw weights sits at
  // a correlation of 0.00003, and this runs fifty layers on top of that.
  CHECK_MSG(video_corr > 0.9, "nvfp4 video velocity correlates %.5f with the fp8 build of the "
                              "same weights; a layout error would sit near zero", video_corr);
  CHECK_MSG(audio_corr > 0.9, "nvfp4 audio velocity correlates %.5f with the fp8 build of the "
                              "same weights; a layout error would sit near zero", audio_corr);
}
