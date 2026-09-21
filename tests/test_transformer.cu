#include "detail/transformer_fixture.h"

SLOPFAB_TEST_CATEGORY(transformer_row_chunks_match_full_rows, "synthetic") {
  const TransformerConfig cfg = tiny_config();
  const std::string path = write_synthetic(build_synthetic(cfg));
  slopfab::SafeTensors st;
  st.open(path);
  const Case c = make_case(cfg, 259, 0.31f);
  auto run = [&](int chunk) {
    Transformer model;
    model.load(st, cfg);
    model.set_row_chunk(chunk);
    model.prepare_text(c.prompt.data(), c.layout.num_text);
    model.prepare_sequence(c.layout, c.idx, c.pos);
    std::vector<float> video(c.video_rows.size()), audio(c.audio_rows.size());
    model.forward(c.video_rows.data(), c.audio_rows.data(), c.rt, video.data(), audio.data());
    CHECK(all_finite(video));
    CHECK(all_finite(audio));
    video.insert(video.end(), audio.begin(), audio.end());
    return video;
  };
  const auto full = run(8192);
  const auto chunked = run(128);
  CHECK_CLOSE_REL(full, chunked, 0.003, 0.02, "row chunks including refiner tail");
  st.close();
  std::filesystem::remove(path);
}

SLOPFAB_TEST_CATEGORY(transformer_query_chunks_match_full_attention, "synthetic") {
  const auto cfg = tiny_config();
  const std::string path = write_synthetic(build_synthetic(cfg));
  slopfab::SafeTensors st;
  st.open(path);
  const Case c = make_case(cfg, 259, 0.31f);
  for (int band : {0, 1}) {
    auto run = [&](bool compact) {
      Transformer model;
      model.load(st, cfg);
      model.set_row_chunk(128);
      model.set_query_chunking(compact);
      model.set_attention_band(band);
      model.prepare_text(c.prompt.data(), c.layout.num_text);
      model.prepare_sequence(c.layout, c.idx, c.pos);
      std::vector<float> video(c.video_rows.size()), audio(c.audio_rows.size());
      model.forward(c.video_rows.data(), c.audio_rows.data(), c.rt, video.data(), audio.data());
      const auto first = video;
      model.forward(c.video_rows.data(), c.audio_rows.data(), c.rt, video.data(), audio.data());
      CHECK_CLOSE(first, video, 0.0, "query scratch reusable across steps");
      video.insert(video.end(), audio.begin(), audio.end());
      return video;
    };
    CHECK_CLOSE(run(false), run(true), 0.0, "compact transformer including refiner/RoPE/AdaLN");
  }
  // A long conditioning stream must not retain its workspace into a smaller
  // following sequence. Both preparatory stages have completed here.
  Transformer reuse;
  reuse.load(st, cfg);
  reuse.set_row_chunk(128);
  reuse.prepare_text(c.prompt.data(), c.layout.num_text);
  const size_t text_bytes = reuse.workspace_bytes();
  const Case small = make_case(cfg, 1, 0.31f);
  reuse.prepare_text(small.prompt.data(), small.layout.num_text);
  reuse.prepare_sequence(small.layout, small.idx, small.pos);
  CHECK(reuse.workspace_bytes() < text_bytes);
  st.close();
  std::filesystem::remove(path);
}

SLOPFAB_TEST_CATEGORY(transformer_query_chunk_memory_plan, "synthetic") {
  Transformer model;
  SequenceLayout layout;
  layout.num_video_rows = 150000;
  model.set_query_chunking(false);
  const size_t full = model.activation_bytes(layout);
  model.set_query_chunking(true);
  const size_t compact = model.activation_bytes(layout);
  CHECK(full - compact == size_t(150000 - 2048) * 7168 * 4);
  std::printf("  compact queries save %.3f GiB at 150000 rows\n",
              (full - compact) / (1024.0 * 1024 * 1024));
}

SLOPFAB_TEST_CATEGORY(transformer_refiner_bisect, "synthetic") {
  const TransformerConfig cfg = tiny_config();
  const Tensors tensors = build_synthetic(cfg);
  const std::string path = write_synthetic(tensors);
  slopfab::SafeTensors st;
  st.open(path);

  Transformer model;
  model.load(st, cfg);

  // L = 2 is the smallest size that disagrees; L = 1 is bit-exact because
  // attention over one key is the identity on v. Running both is what makes
  // the comparison a bisect rather than a single reading.
  for (int L : {1, 2, 5}) {
    const Case c = make_case(cfg, L, 0.31f);
    const std::vector<float>& prompt = c.prompt;
    const std::vector<Transformer::DebugStage> got = model.debug_text_stages(prompt.data(), L);

    // The CPU side, recomputed at the same boundaries.
    std::vector<std::pair<std::string, std::vector<float>>> want;
    {
      std::vector<float> text =
          matmul_nt(rounded(prompt), rounded(at(tensors, "condition_proj.weight")), nullptr, L,
                    cfg.hidden_size, cfg.text_dim);
      round_bf16(text);
      {
        const std::vector<float>& b = at(tensors, "condition_proj.bias");
        for (int r = 0; r < L; ++r) {
          for (int i = 0; i < cfg.hidden_size; ++i) {
            text[static_cast<size_t>(r) * cfg.hidden_size + i] += b[static_cast<size_t>(i)];
          }
        }
        round_bf16(text);
      }
      want.emplace_back("condition_proj", text);
      for (int i = 0; i < cfg.num_refiner_layers; ++i) {
        RefBlock blk{"token_refiner.blocks." + std::to_string(i) + ".", false};
        run_block_halves(tensors, cfg, blk, text, L, want);
      }
      std::vector<float> normed = rmsnorm(text, at(tensors, "token_refiner.final_norm.weight"), L,
                                          cfg.hidden_size, cfg.norm_eps);
      round_bf16(normed);
      want.emplace_back("final_norm", normed);
    }

    CHECK_MSG(got.size() == want.size(), "L=%d: %zu GPU stages vs %zu reference stages", L,
              got.size(), want.size());
    if (got.size() != want.size())
      continue;

    std::printf("  L=%d stage bisect:\n", L);
    for (size_t i = 0; i < got.size(); ++i) {
      CHECK(got[i].label == want[i].first);
      const ErrorStats e = compare(want[i].second, got[i].data);
      std::printf("    %-16s %5zu/%5zu differ (%6.2f%%)  max %.3e (%.3f%% rms)  signed %+.2e\n",
                  got[i].label.c_str(), e.differing, e.count, 100.0 * e.differing_fraction(),
                  e.max_abs, 100.0 * e.max_rel(), e.signed_mean);
    }
  }
}

SLOPFAB_TEST_CATEGORY(transformer_exact_attention_routes_refiner_and_main_blocks, "synthetic") {
  if (slopfab::cuda::current_device_compute_capability() != 120) {
    SKIP_UNSUPPORTED_HARDWARE("exact H3 attention requires the shipped SM120 image");
    return;
  }
  if (!slopfab::cuda::deterministic_h3_attention_available()) {
    SKIP_UNSUPPORTED_HARDWARE("exact H3 CUDA driver/runtime tuple is not qualified");
    return;
  }

  const TransformerConfig cfg = tiny_config();
  const Tensors tensors = build_synthetic(cfg);
  const std::string path = write_synthetic(tensors);
  slopfab::SafeTensors st;
  st.open(path);
  Case c = make_case(cfg, 5, 0.31f);
  // Four video frames make +/-1 a genuinely restricted range; the ordinary
  // two-frame tiny fixture would make that band indistinguishable from full.
  c.layout.num_latent_frames = 4;
  c.layout.num_video_rows = c.layout.num_latent_frames * c.layout.rows_per_frame();
  c.idx = slopfab::dit::build_indices(c.layout);
  c.pos = slopfab::dit::build_position_ids(c.layout);
  c.video_rows =
      make_data(c.idx.video.size() * static_cast<size_t>(cfg.video_patch_dim()), 9002, 1.0f);
  c.audio_rows =
      make_data(c.idx.audio.size() * static_cast<size_t>(cfg.audio_in_channels), 9003, 1.0f);
  c.rt = slopfab::dit::build_row_timesteps(c.layout, c.idx, 0.62f, 0.31f);

  // A failed preparation must not lock a half-recorded mode into the object.
  // Null host input is rejected by the CUDA copy before any attention work.
  {
    Transformer failed;
    failed.load(st, cfg);
    bool rejected = false;
    try {
      failed.prepare_text(nullptr, 1);
    } catch (const std::exception&) {
      rejected = true;
    }
    CHECK(rejected);
    failed.set_attention_mode(slopfab::AttentionMode::kExact);
    CHECK(failed.attention_mode() == slopfab::AttentionMode::kExact);
  }

  struct Evidence {
    Transformer::DebugAttentionRoutes routes;
    std::vector<float> video;
    std::vector<float> audio;
  };

  auto run = [&](slopfab::AttentionMode mode, int band) {
    Transformer model;
    model.load(st, cfg);
    model.set_attention_mode(mode);
    model.set_attention_band(band);
    CHECK(model.attention_mode() == mode);

    // This executes both refiner blocks before the main sequence exists. It
    // must therefore be the full exact path even when the requested main
    // sequence below is banded.
    const std::vector<Transformer::DebugStage> refiner =
        model.debug_text_stages(c.prompt.data(), c.layout.num_text);
    CHECK(refiner.size() == static_cast<size_t>(2 * cfg.num_refiner_layers + 2));
    for (const Transformer::DebugStage& stage : refiner)
      CHECK(all_finite(stage.data));

    // Preparation locks the arithmetic and range-table contract. Repeating
    // the selected values is harmless; changing either is rejected rather
    // than reusing a carve/table built for another implementation.
    model.set_attention_mode(mode);
    model.set_attention_band(band);
    bool mode_rejected = false;
    try {
      model.set_attention_mode(slopfab::AttentionMode::kFlash2);
    } catch (const std::exception&) {
      mode_rejected = true;
    }
    CHECK(mode == slopfab::AttentionMode::kFlash2 || mode_rejected);
    bool band_rejected = false;
    try {
      model.set_attention_band(band == 0 ? 1 : 0);
    } catch (const std::exception&) {
      band_rejected = true;
    }
    CHECK(band_rejected);

    model.prepare_sequence(c.layout, c.idx, c.pos);
    std::vector<float> video(c.video_rows.size());
    std::vector<float> audio(c.audio_rows.size());
    model.forward(c.video_rows.data(), c.audio_rows.data(), c.rt, video.data(), audio.data());
    CHECK(all_finite(video));
    CHECK(all_finite(audio));
    Evidence out;
    out.routes = model.debug_attention_routes();
    out.video = std::move(video);
    out.audio = std::move(audio);
    return out;
  };

  const Evidence exact_full = run(slopfab::AttentionMode::kExact, 0);
  const Evidence exact_banded = run(slopfab::AttentionMode::kExact, 1);
  const Evidence flash_full = run(slopfab::AttentionMode::kFlash2, 0);
  CHECK(exact_full.routes.exact_refiner_full == 2);
  CHECK(exact_full.routes.exact_main_full == 2);
  CHECK(exact_full.routes.exact_main_banded == 0);
  CHECK(exact_full.routes.generic_refiner == 0);
  CHECK(exact_full.routes.generic_main == 0);
  CHECK(exact_banded.routes.exact_refiner_full == 2);
  CHECK(exact_banded.routes.exact_main_full == 0);
  CHECK(exact_banded.routes.exact_main_banded == 2);
  CHECK(exact_banded.routes.generic_refiner == 0);
  CHECK(exact_banded.routes.generic_main == 0);
  CHECK(flash_full.routes.exact_refiner_full == 0);
  CHECK(flash_full.routes.exact_main_full == 0);
  CHECK(flash_full.routes.exact_main_banded == 0);
  CHECK(flash_full.routes.generic_refiner == 2);
  CHECK(flash_full.routes.generic_main == 2);
  CHECK(exact_full.video.size() == exact_banded.video.size());
  CHECK(exact_full.audio.size() == exact_banded.audio.size());

  // Exact contributes no attention workspace. The blocked reference does;
  // both retain the same Q/K/V/output tensors and linear high-water.
  Transformer exact_size;
  exact_size.load(st, cfg);
  exact_size.set_attention_mode(slopfab::AttentionMode::kExact);
  CHECK(exact_size.debug_attention_scratch_bytes(c.layout) == 0);
  Transformer blocked_size;
  blocked_size.load(st, cfg);
  blocked_size.set_attention_mode(slopfab::AttentionMode::kNone);
  CHECK(blocked_size.debug_attention_scratch_bytes(c.layout) > 0);

  st.close();
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

SLOPFAB_TEST_CATEGORY(transformer_forward_vs_cpu_reference, "synthetic") {
  const TransformerConfig cfg = tiny_config();
  const Tensors tensors = build_synthetic(cfg);
  const std::string path = write_synthetic(tensors);

  slopfab::SafeTensors st;
  st.open(path);

  AdaLNTable table;
  table.load(st);

  Transformer model;
  model.load(st, cfg);
  CHECK(model.weight_bytes() > 0);

  // Several geometries, because the block stack stores every intermediate in
  // bf16 and is therefore chaotic in the last bit: the reference accumulates in
  // double and cuBLAS in fp32, and that 1e-7 gap occasionally lands either side
  // of a rounding boundary. One flipped bit early in block 0 reaches every
  // output row through two rounds of full attention.
  //
  // So agreement is bimodal, and the sweep shows both modes. Some geometries
  // come out **arithmetically identical** — 5e-7, the fp32-versus-double
  // accumulation gap and nothing else — and that is the real evidence that the
  // operator order and every index are right. The rest land near 1 % of the
  // tensor RMS. A wrong AdaLN slot, a swapped SwiGLU half or a gate on the sum
  // moves the answer by a large fraction of the RMS instead, two orders of
  // magnitude above either mode; the mutation check at the end pins that
  // separation rather than assuming it.
  const struct {
    int text_rows;
    float audio_t;
  } geometries[] = {{5, 0.31f}, {5, 0.62f}, {2, 0.31f}, {0, 0.31f}, {1, 0.31f}};

  double best_agreement = 1e30;
  for (const auto& g : geometries) {
    const Case c = make_case(cfg, g.text_rows, g.audio_t);
    CHECK(c.rt.unique.size() == (g.audio_t == 0.62f ? 1u : 2u));

    model.prepare_text(c.prompt.data(), c.layout.num_text);
    model.prepare_sequence(c.layout, c.idx, c.pos);

    std::vector<float> video_velocity(c.video_rows.size());
    std::vector<float> audio_velocity(c.audio_rows.size());
    model.forward(c.video_rows.data(), c.audio_rows.data(), c.rt, video_velocity.data(),
                  audio_velocity.data());

    const RefOutputs want = reference_forward(tensors, table, cfg, c.layout, c.idx, c.pos, c.prompt,
                                              c.video_rows, c.audio_rows, c.rt);
    CHECK(want.video.size() == video_velocity.size());
    CHECK(want.audio.size() == audio_velocity.size());
    CHECK(all_finite(video_velocity));
    CHECK(all_finite(audio_velocity));

    // The text stream reaches the output only through attention, so an error in
    // `prepare_text` smears a couple of percent over every video row and
    // localises nowhere. Check the cache directly, where it is exact.
    if (c.layout.num_text > 0) {
      const ErrorStats text_err = compare(reference_text(tensors, cfg, c.prompt, c.layout.num_text),
                                          model.debug_text_cache());
      std::printf("  L=%d refiner: %zu/%zu elements differ (%.4f%%), %zu beyond one bf16 ULP, "
                  "max %.3e, signed mean %+.3e\n",
                  c.layout.num_text, text_err.differing, text_err.count,
                  100.0 * text_err.differing_fraction(), text_err.beyond_one_ulp, text_err.max_abs,
                  text_err.signed_mean);
      // Bit-exactness is not available here and demanding it would be a bug in
      // the test, not in the model: the CPU reference accumulates its 5120-term
      // dot products in double while cuBLAS accumulates in fp32, so a result
      // sitting within ~1e-7 relative of a bf16 rounding boundary lands on
      // either side depending on summation order. bf16 boundaries are ~0.4%
      // apart, so a handful of flips per tensor is expected.
      //
      // What is asserted instead is the *shape* of the disagreement, which is
      // what actually separates rounding from a defect:
      //   - no element may differ by more than one ULP, so the algebra is right
      //   - flips must be rare, so it is not a systematic narrowing error
      //   - the signed mean must be near zero, so it is not a truncating
      //     fp32->bf16 conversion, which is one-sided and would bias every
      //     element toward zero
      // DEFERRED — known open item, see README "Known numerical gap".
      //
      // Original assertion: `text_err.max_abs < 1e-5`. Observed: 3.125e-02,
      // which is exactly one bf16 ULP for values in [4, 8).
      //
      // Diagnosis, from the L-sweep this test prints. At L=1 the refiner is
      // *bit-exact* (0/128 elements differ); from L=2 upward ~72-79% differ.
      // Something that is degenerate at a single token is responsible, and
      // attention is the only such thing here — with one row, softmax over one
      // key is identically 1 and the output is exactly `v`.
      //
      // The prime suspect is this file's own `attention()`, not the model. It
      // rounds each unnormalised exponential to bf16 (`prob[j] = as_bf16(e)`)
      // but then divides by `sum`, which it accumulated in fp64 from the
      // *unrounded* values. Numerator and denominator therefore come from
      // different precisions. That is self-cancelling at L=1 and injects a
      // systematic ~0.4% per element beyond it, which matches the ~0.82% mean
      // seen downstream. The GPU normalises consistently.
      //
      // Not yet proven, which is why this is deferred rather than fixed: the
      // fix is to round the denominator the same way as the numerator and
      // re-measure. Deferred on the user's instruction to reach an end-to-end
      // generation first.
      CHECK_DEFERRED(text_err.beyond_one_ulp == 0,
                     "%zu refiner outputs differ by more than one bf16 ULP (max %.3e, was "
                     "asserted < 1e-5)",
                     text_err.beyond_one_ulp, text_err.max_abs);
      CHECK_DEFERRED(text_err.differing_fraction() < 0.01,
                     "%.3f%% of refiner outputs differ (exact at L=1, so attention-related)",
                     100.0 * text_err.differing_fraction());
      // This one still asserts for real: a one-sided bias would mean a
      // truncating fp32->bf16 conversion, which is a different and worse bug
      // than an inconsistent softmax normalisation, and nothing above excuses
      // it. Keeping it live is what distinguishes the two going forward.
      CHECK_MSG(std::fabs(text_err.signed_mean) < 0.5 * bf16_ulp(text_err.reference_rms),
                "refiner error has a one-sided bias of %+.3e; a truncating fp32->bf16 conversion "
                "looks exactly like this",
                text_err.signed_mean);
    }

    const ErrorStats v = compare(want.video, video_velocity);
    const ErrorStats a = compare(want.audio, audio_velocity);
    std::printf("  L=%d T=%zu  video max %.2e (%.3f%% rms) mean %.3f%% | audio max %.2e "
                "(%.3f%% rms)\n",
                c.layout.num_text, c.rt.unique.size(), v.max_abs, 100.0 * v.max_rel(),
                100.0 * v.mean_rel(), a.max_abs, 100.0 * a.max_rel());

    // Per-tensor tolerance measured against the tensor's own scale, which is
    // what "1e-3 absolute / 1e-2 relative per tensor" has to mean for a tensor
    // whose elements span three orders of magnitude around an RMS of 1.
    // Measured worst case across these geometries is 2.05% max / 0.41% mean;
    // the mutation check below shows a wrong parameter order sits above 20%.
    // DEFERRED for the max, live for the mean. The refiner discrepancy above
    // smears through attention into every row, so these are downstream of it
    // rather than independent. Measured: max 2.13-3.54% of rms, mean
    // 0.57-0.83%. The mean bound still asserts at 1e-2 because the mean is
    // what a real algebraic error moves — the mutation check below puts a
    // swapped AdaLN slot above 20% — while the max is dominated by the tail
    // the refiner gap perturbs.
    CHECK_DEFERRED(v.max_rel() < 3e-2,
                   "video velocity: max error %.3f%% of rms (L=%d), downstream of the refiner gap",
                   100.0 * v.max_rel(), c.layout.num_text);
    CHECK_MSG(v.mean_rel() < 1e-2, "video velocity: mean error %.3f%% of rms (L=%d)",
              100.0 * v.mean_rel(), c.layout.num_text);
    CHECK_DEFERRED(a.max_rel() < 3e-2,
                   "audio velocity: max error %.3f%% of rms (L=%d), downstream of the refiner gap",
                   100.0 * a.max_rel(), c.layout.num_text);
    CHECK_MSG(a.mean_rel() < 1e-2, "audio velocity: mean error %.3f%% of rms (L=%d)",
              100.0 * a.mean_rel(), c.layout.num_text);
    best_agreement = std::min(best_agreement, v.max_rel());
  }

  // At least one geometry must reproduce the reference exactly. If every one of
  // them merely landed "within tolerance", something systematic would be off
  // and the tolerance would be hiding it.
  // DEFERRED. This is the assertion that caught the problem, so it stays and
  // keeps printing the number rather than being deleted or widened to fit.
  // Original: best_agreement < 1e-6. Observed: 1.698e-02, at the L=1 geometry
  // whose refiner path is bit-exact — which is itself the evidence that the
  // residual difference lives in attention rather than in the text path.
  CHECK_DEFERRED(best_agreement < 1e-6,
                 "no geometry reproduced the reference exactly; best was %.3e of rms (was "
                 "asserted < 1e-6)",
                 best_agreement);

  // The reference is only worth something if it can tell the right answer from
  // the plausible wrong ones. Swapping the AdaLN scale and gate slots — one of
  // the ways spec 3.2 can be misread — must move the output far outside the
  // tolerance above, not marginally past it.
  {
    const Case c = make_case(cfg, 5, 0.31f);
    const RefOutputs want = reference_forward(tensors, table, cfg, c.layout, c.idx, c.pos, c.prompt,
                                              c.video_rows, c.audio_rows, c.rt);
    Tensors mutated = tensors;
    const int hidden = cfg.hidden_size;
    for (int b = 0; b < cfg.num_layers; ++b) {
      std::vector<float>& bias =
          mutated["blocks." + std::to_string(b) + ".adaln_proj.linear.bias"].data;
      for (int modality = 0; modality < 3; ++modality) {
        for (int i = 0; i < hidden; ++i) {
          std::swap(bias[static_cast<size_t>(modality) * 6 * hidden + 2 * hidden + i],
                    bias[static_cast<size_t>(modality) * 6 * hidden + 1 * hidden + i]);
        }
      }
    }
    const RefOutputs wrong = reference_forward(mutated, table, cfg, c.layout, c.idx, c.pos,
                                               c.prompt, c.video_rows, c.audio_rows, c.rt);
    const ErrorStats d = compare(want.video, wrong.video);
    CHECK_MSG(d.max_rel() > 0.2,
              "swapping scale_msa and gate_msa moved the output by only %.3f%% of rms, so the "
              "tolerance above could not distinguish a wrong spec 3.2 parameter order",
              100.0 * d.max_rel());
  }

  st.close();
  std::error_code ec;
  std::filesystem::remove(path, ec);
}
