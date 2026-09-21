#include "detail/encoder_fixture.h"

SLOPFAB_TEST_CATEGORY(encoder_real_encode, "integration") {
  const std::string path = find_checkpoint();
  if (path.empty()) {
    SKIP_MISSING_FIXTURE("  text encoder checkpoint not present; skipping\n");
    return;
  }

  slopfab::SafeTensors st;
  st.open(path);

  // A real prompt when the tokenizer is available, a synthetic id run
  // otherwise. Either way the causality test needs one token list that is a
  // strict prefix of another.
  std::vector<int32_t> ids;
  const std::string tok_path = find_tokenizer();
  if (!tok_path.empty()) {
    slopfab::text::Tokenizer tokenizer;
    tokenizer.load(tok_path);
    // No BOS, no EOS, no chat template: Tokenizer::encode adds nothing, and
    // adding one later would shift every RoPE position (spec section 1.2).
    ids = tokenizer.encode(
        "A slow aerial shot over a rain-slicked city at night, neon signs reflected in the "
        "puddles, a lone figure walking beneath the overpass while a tram passes overhead. "
        "The camera drifts forward and tilts down as the light changes from red to green, "
        "steam rising from a vent, distant sirens, the hum of traffic on a wet road. "
        "Cut to a narrow street market, awnings dripping, a vendor folding a tarp over "
        "crates of fruit while two children run past kicking up spray. The lens racks focus "
        "from the foreground puddle to a bus pulling away, its windows fogged, passengers "
        "silhouetted against the interior light. Overhead wires sway. A cat crosses the "
        "frame and disappears into a doorway. The shot holds on the empty street as the rain "
        "eases and the reflections settle, then pushes in slowly on a single lit window "
        "three floors up, where a figure stands with their back to the glass.");
    std::printf("  tokenised the prompt to %zu tokens\n", ids.size());
  } else {
    SKIP_MISSING_FIXTURE("  ref/ tokenizer.json not present; using synthetic token ids\n");
    for (int i = 0; i < 200; ++i)
      ids.push_back(1000 + i);
  }
  CHECK(!ids.empty());

  std::vector<int32_t> longer = ids;
  for (int i = 0; i < 37; ++i)
    longer.push_back(5000 + i * 7);

  size_t free_before = 0;
  size_t total = 0;
  SLOPFAB_CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
  std::printf("  device: %.2f GB free of %.2f GB\n", double(free_before) / (1 << 30),
              double(total) / (1 << 30));

  // Skip rather than fail when the card is busy. Streaming residency needs
  // ~1.2 GB and resident needs 24.4, so on a shared or contended GPU this test
  // can legitimately have nowhere to run — and a red test that means "someone
  // else is using the card" trains people to ignore red tests. The resident
  // block below already degrades to a printed note on its own; this guard
  // covers the streaming block, which has no fallback beneath it.
  constexpr size_t kStreamingHeadroom = size_t(2) << 30;
  if (free_before < kStreamingHeadroom) {
    SKIP_INSUFFICIENT_VRAM("only %.2f GB free; need ~2 GB even to stream",
                           double(free_before) / (1 << 30));
    return;
  }

  std::vector<float> resident_out;

  // --- residency mode 1: everything on the device.
  {
    slopfab::text::Encoder encoder;
    slopfab::text::EncoderConfig cfg;
    cfg.residency = slopfab::text::Residency::kResident;

    bool loaded = true;
    try {
      encoder.load(st, cfg);
    } catch (const std::exception& e) {
      loaded = false;
      std::printf("  resident load failed (%s); the card cannot hold 24.4 GB right now\n",
                  e.what());
      // load() owns every queued upload. A failed resident transaction must
      // drain and free it immediately, not leave the next streaming encoder to
      // discover a stale async OOM at cublasCreate.
      CHECK(cudaGetLastError() == cudaSuccess);
      size_t free_after_failed_load = 0;
      size_t ignored_total = 0;
      CHECK(cudaMemGetInfo(&free_after_failed_load, &ignored_total) == cudaSuccess);
      constexpr size_t kRollbackTolerance = size_t(512) << 20;
      const size_t retained =
          free_before > free_after_failed_load ? free_before - free_after_failed_load : 0;
      CHECK_MSG(free_after_failed_load + kRollbackTolerance >= free_before,
                "failed resident load retained %.2f GB of device memory",
                double(retained) / (1 << 30));
    }

    if (loaded) {
      const slopfab::text::PromptEmbedding a = encoder.encode(ids);
      size_t free_after = 0;
      SLOPFAB_CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
      const slopfab::text::EncoderStats& s = encoder.stats();
      std::printf("  resident: load %.2f s, encode %.3f s for %d tokens, weights %.2f GB, "
                  "workspace %.2f GB, accounted peak %.2f GB, measured %.2f GB\n",
                  s.load_seconds, s.last_encode_seconds, s.last_num_tokens,
                  double(s.weight_bytes) / (1 << 30), double(s.workspace_bytes) / (1 << 30),
                  double(s.peak_device_bytes) / (1 << 30),
                  double(free_before - free_after) / (1 << 30));

      CHECK(encoder.residency() == slopfab::text::Residency::kResident);
      CHECK(a.num_tokens == int(ids.size()));
      CHECK(a.hidden_size == 5120);
      CHECK(a.data.size() == size_t(a.num_tokens) * 5120);

      size_t nonfinite = 0;
      for (float value : a.data) {
        if (!std::isfinite(value))
          ++nonfinite;
      }
      CHECK_MSG(nonfinite == 0, "%zu of %zu output values are not finite", nonfinite,
                a.data.size());

      // The layer-49 residual is a raw pre-norm stream. The discriminator
      // against a final norm having crept in — spec section 1.4's trap, and
      // the exact failure encoders.py:142-149 guards against — is not the
      // absolute RMS but its *spread across rows*: an RMSNorm divides every
      // row by its own scale, so a normalised stream has row RMS ~ RMS(w) for
      // every row, flat to a few percent. A raw residual does not.
      //
      // Note the spec says the per-row RMS should be "hundreds". Measured, it
      // is the per-row L2 *norm* that reaches the hundreds; the RMS over 5120
      // channels is a few, except on token 0.
      std::vector<double> row_rms(a.num_tokens, 0.0);
      for (int r = 0; r < a.num_tokens; ++r) {
        double acc = 0.0;
        for (int i = 0; i < 5120; ++i) {
          const double v = a.data[size_t(r) * 5120 + i];
          acc += v * v;
        }
        row_rms[size_t(r)] = std::sqrt(acc / 5120.0);
      }
      const double rms_min = *std::min_element(row_rms.begin(), row_rms.end());
      const double rms_max = *std::max_element(row_rms.begin(), row_rms.end());
      const double rms_first = row_rms.front();
      const double rms_last = row_rms.back();
      std::printf("  row RMS: first %.2f, last %.2f, min %.2f, max %.2f (L2 norm of the last row "
                  "%.1f)\n",
                  rms_first, rms_last, rms_min, rms_max, rms_last * std::sqrt(5120.0));
      CHECK_MSG(rms_min > 0.5, "row RMS falls to %.4f; the stream has collapsed", rms_min);
      CHECK_MSG(rms_max < 1e5, "row RMS reaches %.4f, which is implausibly large", rms_max);
      CHECK_MSG(rms_max / rms_min > 3.0,
                "row RMS is nearly constant across rows (min %.3f, max %.3f). That is what an "
                "RMSNorm output looks like — no final norm may be applied here",
                rms_min, rms_max);

      // --- causality, the free test from spec section 3. Row 0 attends only
      // to itself, so appending tokens cannot change it. Bidirectional
      // attention is otherwise completely silent.
      const slopfab::text::PromptEmbedding b = encoder.encode(longer);
      std::printf("  resident: warm encode %.3f s for %d tokens\n", s.last_encode_seconds,
                  s.last_num_tokens);
      CHECK(b.num_tokens == int(longer.size()));

      // Measured per row as ||a_r - b_r|| / ||a_r||. It is not bitwise: L
      // changes the GEMM shapes, cuBLAS picks a different reduction order for
      // a different M, and 50 residual layers amplify that. Bidirectional
      // attention would instead make row 0 an average over the whole prompt,
      // moving it by order one — two orders of magnitude above this floor.
      double row0_relative = 0.0;
      double worst_relative = 0.0;
      int worst_row = 0;
      for (int r = 0; r < a.num_tokens; ++r) {
        double num = 0.0;
        double den = 0.0;
        for (int i = 0; i < 5120; ++i) {
          const size_t idx = size_t(r) * 5120 + i;
          const double diff = double(a.data[idx]) - b.data[idx];
          num += diff * diff;
          den += double(a.data[idx]) * a.data[idx];
        }
        const double relative = std::sqrt(num / std::max(1e-30, den));
        if (r == 0)
          row0_relative = relative;
        if (relative > worst_relative) {
          worst_relative = relative;
          worst_row = r;
        }
      }
      std::printf("  causality: row 0 moved %.3e of its norm; worst prefix row %d moved %.3e\n",
                  row0_relative, worst_row, worst_relative);
      CHECK_MSG(row0_relative < 1e-2,
                "prompt_embeds[0] moved by %.3e of its norm when tokens were appended; attention "
                "is not causal",
                row0_relative);
      // Every row of the shared prefix is likewise a function of that prefix
      // alone.
      CHECK_MSG(worst_relative < 5e-2,
                "row %d of the shared prefix moved %.3e of its norm when tokens were appended",
                worst_row, worst_relative);

      resident_out = a.data;
      encoder.unload();
    }
  }

  // --- residency mode 2: stream each layer from the mapping just before use.
  {
    slopfab::text::Encoder encoder;
    slopfab::text::EncoderConfig cfg;
    cfg.residency = slopfab::text::Residency::kStreaming;
    encoder.load(st, cfg);
    CHECK(encoder.residency() == slopfab::text::Residency::kStreaming);

    const slopfab::text::PromptEmbedding c = encoder.encode(ids);
    size_t free_after = 0;
    SLOPFAB_CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
    const slopfab::text::EncoderStats& s = encoder.stats();
    std::printf("  streaming: load %.2f s, encode %.3f s for %d tokens, layer buffers %.2f GB, "
                "workspace %.2f GB, accounted peak %.2f GB, measured %.2f GB\n",
                s.load_seconds, s.last_encode_seconds, s.last_num_tokens,
                double(s.weight_bytes) / (1 << 30), double(s.workspace_bytes) / (1 << 30),
                double(s.peak_device_bytes) / (1 << 30),
                double(free_before - free_after) / (1 << 30));

    CHECK(c.num_tokens == int(ids.size()));
    size_t nonfinite = 0;
    for (float value : c.data) {
      if (!std::isfinite(value))
        ++nonfinite;
    }
    CHECK(nonfinite == 0);

    // The two modes differ only in when the weights arrive, so they must agree
    // bit for bit. A double-buffering race would show up here and nowhere else.
    if (!resident_out.empty()) {
      size_t mismatches = 0;
      for (size_t i = 0; i < c.data.size(); ++i) {
        if (c.data[i] != resident_out[i])
          ++mismatches;
      }
      CHECK_MSG(mismatches == 0,
                "%zu of %zu values differ between the resident and streaming paths (max abs %.4g)",
                mismatches, c.data.size(), max_abs_diff(resident_out, c.data));
    }

    const slopfab::text::PromptEmbedding d = encoder.encode(ids);
    std::printf("  streaming: warm encode %.3f s for %d tokens (page cache warm)\n",
                s.last_encode_seconds, s.last_num_tokens);
    CHECK(d.num_tokens == int(ids.size()));

    // Rejections at the API boundary.
    bool empty_threw = false;
    try {
      encoder.encode(std::vector<int32_t>{});
    } catch (const std::exception&) {
      empty_threw = true;
    }
    CHECK(empty_threw);

    bool oob_threw = false;
    try {
      encoder.encode(std::vector<int32_t>{5, 151936});
    } catch (const std::exception&) {
      oob_threw = true;
    }
    CHECK(oob_threw);

    encoder.unload();
  }
}

SLOPFAB_TEST_CATEGORY(encoder_nvfp4_real_encode, "integration") {
  const std::string path = find_nvfp4_checkpoint();
  if (path.empty()) {
    SKIP_MISSING_FIXTURE("  nvfp4 text encoder checkpoint not present; skipping\n");
    return;
  }

  slopfab::SafeTensors st;
  st.open(path);
  // 1700 layer tensors at 34 per layer, 3 for the embedding, 351 visual.* that
  // are present and never loaded.
  CHECK(st.tensor_count() == 2054);
  CHECK(slopfab::text::detect_weight_format(st) == slopfab::text::WeightFormat::kNVFP4Awq);

  slopfab::text::EncoderConfig cfg;
  slopfab::text::validate_checkpoint(st, cfg);

  // Asking for the other build must be refused rather than half-read.
  cfg.format = slopfab::text::WeightFormat::kI8ConvRot;
  bool threw = false;
  try {
    slopfab::text::validate_checkpoint(st, cfg);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);

  const std::vector<int32_t> ids = prompt_ids();
  CHECK(!ids.empty());

  size_t free_before = 0;
  size_t total = 0;
  SLOPFAB_CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
  std::printf("  device: %.2f GB free of %.2f GB\n", double(free_before) / (1 << 30),
              double(total) / (1 << 30));
  if (free_before < (size_t(2) << 30)) {
    SKIP_INSUFFICIENT_VRAM("only %.2f GB free; need ~2 GB even to stream",
                           double(free_before) / (1 << 30));
    return;
  }

  const EncodeRun resident =
      run_encoder(st, slopfab::text::Residency::kResident, ids, "nvfp4 resident");
  const EncodeRun streaming =
      run_encoder(st, slopfab::text::Residency::kStreaming, ids, "nvfp4 streaming");

  if (resident.ok)
    check_residual_stream_shape(resident.out, "nvfp4 resident");
  if (streaming.ok)
    check_residual_stream_shape(streaming.out, "nvfp4 streaming");

  // The two modes differ only in when the weights arrive, so they must agree
  // bit for bit. A double-buffering race would show up here and nowhere else.
  if (resident.ok && streaming.ok) {
    size_t mismatches = 0;
    for (size_t i = 0; i < resident.out.data.size(); ++i) {
      if (resident.out.data[i] != streaming.out.data[i])
        ++mismatches;
    }
    CHECK_MSG(mismatches == 0,
              "%zu of %zu values differ between the residency modes (max abs %.4g)", mismatches,
              resident.out.data.size(), max_abs_diff(resident.out.data, streaming.out.data));
  }

  // --- the cross-checkpoint comparison, which is what actually catches a wrong
  // AWQ direction or a wrong block stride. Both are silent: they leave the
  // output finite, correctly shaped and plausibly scaled, and they pass every
  // structural check above. Neither survives being compared with the other
  // build of the same model.
  //
  // Two *different quantisations* agree to a few percent, not to 1e-3 — that is
  // a different bar from the project's per-tensor tolerance and must not be
  // confused with it. A wrong fold direction misses by order one.
  const std::string int8_path = find_checkpoint();
  if (int8_path.empty() || !resident.ok) {
    SKIP_MISSING_FIXTURE(
        "  int8 checkpoint not present; skipping the cross-checkpoint comparison\n");
    return;
  }
  slopfab::SafeTensors i8;
  i8.open(int8_path);
  const EncodeRun other =
      run_encoder(i8, slopfab::text::Residency::kStreaming, ids, "int8 streaming");
  if (!other.ok)
    return;
  check_residual_stream_shape(other.out, "int8 streaming");

  double worst_row = 0.0;
  int worst_index = 0;
  double total_num = 0.0;
  double total_den = 0.0;
  for (int r = 0; r < resident.out.num_tokens; ++r) {
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < 5120; ++i) {
      const size_t k = size_t(r) * 5120 + i;
      const double d = double(resident.out.data[k]) - other.out.data[k];
      num += d * d;
      den += double(other.out.data[k]) * other.out.data[k];
    }
    total_num += num;
    total_den += den;
    const double rel = std::sqrt(num / std::max(1e-30, den));
    if (rel > worst_row) {
      worst_row = rel;
      worst_index = r;
    }
  }
  const double overall = std::sqrt(total_num / std::max(1e-30, total_den));
  std::printf("  nvfp4 vs int8 hidden_states[50]: %.4f overall, worst row %d at %.4f\n", overall,
              worst_index, worst_row);
  CHECK_MSG(overall < 0.25,
            "the two builds of one model disagree by %.4f of the int8 output's norm. A few percent "
            "is the expected quantisation gap; order one means the AWQ fold direction or the block "
            "scale mapping is wrong, and neither would look wrong on its own",
            overall);
  CHECK_MSG(overall > 1e-4,
            "the two builds agree to %.3e, which is closer than two different quantisations of one "
            "model can be. Something is comparing an output with itself",
            overall);
}
