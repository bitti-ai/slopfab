#include "detail/encoder_fixture.h"

SLOPFAB_TEST_CATEGORY(encoder_rope_inv_freq_and_tables, "synthetic") {
  const std::vector<float> inv = slopfab::text::rope_inv_freq(128, 5.0e6f);
  CHECK(inv.size() == 64);

  // Values transcribed from spec section 2.5, which computed them from
  // `inv_freq[j] = 5e6 ** (-j/64)` independently of this code.
  CHECK_NEAR(inv[0], 1.0, 1e-9);
  CHECK_NEAR(inv[1], 7.858300209e-01, 1e-7);
  CHECK_NEAR(inv[2], 6.175287366e-01, 1e-7);
  CHECK_NEAR(inv[4], 3.813417554e-01, 1e-7);
  CHECK_NEAR(inv[8], 1.454215497e-01, 1e-7);
  // Tolerances are ~1e-7 relative: the value is an fp32 rounding of a power,
  // and spec section 6.4 measures the fp32-vs-fp64 spread at 1.19e-7 relative.
  CHECK_NEAR(inv[16], 2.114742622e-02, 1e-8);
  CHECK_NEAR(inv[24], 3.075291170e-03, 1e-9);
  CHECK_NEAR(inv[32], 4.472136206e-04, 1e-10);
  CHECK_NEAR(inv[63], 2.545079667e-07, 1e-13);

  // head_dim comes from the config (128), not from hidden/num_heads (which
  // would be 80). A port that used 80 would give a different inv_freq[1].
  const std::vector<float> wrong = slopfab::text::rope_inv_freq(80, 5.0e6f);
  CHECK(wrong.size() == 40);
  CHECK(std::fabs(wrong[1] - inv[1]) > 1e-3);

  std::vector<float> cos;
  std::vector<float> sin;
  slopfab::text::build_rope_tables(6, inv, cos, sin);
  CHECK(cos.size() == 6 * 128);
  CHECK(sin.size() == 6 * 128);

  double worst_angle = 0.0;
  size_t duplication_breaks = 0;
  for (int s = 0; s < 6; ++s) {
    for (int j = 0; j < 64; ++j) {
      const double angle = double(s) * inv[j];
      worst_angle = std::max(worst_angle, std::fabs(cos[size_t(s) * 128 + j] - std::cos(angle)));
      worst_angle = std::max(worst_angle, std::fabs(sin[size_t(s) * 128 + j] - std::sin(angle)));
      // The half period is duplicated, not interleaved: cos[j] == cos[j+64]
      // exactly. That duplication is what makes the (j, j+64) pairing correct,
      // and it is the reference's own `cat((freqs, freqs), -1)`.
      if (cos[size_t(s) * 128 + j + 64] != cos[size_t(s) * 128 + j]) ++duplication_breaks;
      if (sin[size_t(s) * 128 + j + 64] != sin[size_t(s) * 128 + j]) ++duplication_breaks;
    }
  }
  CHECK_MSG(worst_angle < 1e-5, "rope table worst deviation from cos/sin(s * inv_freq) is %.3e",
            worst_angle);
  CHECK_MSG(duplication_breaks == 0, "%zu of 768 table entries are not duplicated at j and j+64",
            duplication_breaks);

  // Smoke test from spec section 2.5: with theta = 5e6 only the low channels
  // complete a turn, so cos must still be ~1 for large j even far into the
  // sequence. If it is not, the exponent sign or the /64 is wrong.
  std::vector<float> cos_far;
  std::vector<float> sin_far;
  slopfab::text::build_rope_tables(4096, inv, cos_far, sin_far);
  CHECK(cos_far[size_t(4095) * 128 + 63] > 0.9999f);
  CHECK(cos_far[size_t(4095) * 128 + 48] > 0.99f);
  // ...and the low channels must have turned many times.
  CHECK(std::fabs(sin_far[size_t(4095) * 128 + 0]) <= 1.0f);
  CHECK(cos_far[size_t(4095) * 128 + 0] < 0.9999f);
}

SLOPFAB_TEST_CATEGORY(encoder_rope_neox_rotates_all_128_dims, "synthetic") {
  const int rows = 512;
  const int heads = 3;
  const int head_dim = 128;

  const std::vector<float> inv = slopfab::text::rope_inv_freq(head_dim, 5.0e6f);
  std::vector<float> cos;
  std::vector<float> sin;
  slopfab::text::build_rope_tables(rows, inv, cos, sin);

  const std::vector<float> x =
      bf16_round(make_data(size_t(rows) * heads * head_dim, 4001u, 2.0f));

  // All 128 dims rotate and j pairs with j+64.
  const std::vector<float> want = cpu_rope(x, cos, sin, rows, heads, head_dim, 128);
  // The hazard: the H3 DiT's convention — 96 of 128 rotated, j paired with
  // j+48, dims 96..127 passed through (spec section 9 item 10).
  const std::vector<float> hazard = cpu_rope(x, cos, sin, rows, heads, head_dim, 96);

  BfBuf dx(x);
  DeviceBuffer<float> dcos = to_device(cos);
  DeviceBuffer<float> dsin = to_device(sin);
  slopfab::cuda::launch_rope_neox(dx.p(), dcos.get(), dsin.get(), rows, heads, head_dim, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = dx.host();

  CHECK_CLOSE_REL(want, got, 1e-3, 1e-2, "rope_neox over all 128 dims, pairing (j, j+64)");
  CHECK_MSG(max_abs_diff(hazard, got) > 0.1,
            "RoPE must rotate all 128 dims with pairing (j, j+64), not the DiT's 96 of 128 "
            "(max diff from the hazard form %.4g)",
            max_abs_diff(hazard, got));

  // Directly: the top 32 dims must not be passed through. At row 511 they pair
  // with dims 32..63, whose angles are small but not negligible.
  double top_change = 0.0;
  for (int h = 0; h < heads; ++h) {
    for (int d = 96; d < 128; ++d) {
      const size_t i = (size_t(rows - 1) * heads + h) * head_dim + d;
      top_change = std::max(top_change, std::fabs(double(got[i]) - x[i]));
    }
  }
  CHECK_MSG(top_change > 1e-3, "dims 96..127 were passed through unrotated (max change %.4g)",
            top_change);
}

SLOPFAB_TEST_CATEGORY(encoder_causal_attention, "synthetic") {
  CublasScope cb;
  const int seq = 300;
  const int heads = 8;
  const int kv_heads = 2;
  const int head_dim = 64;
  const int qld = heads * head_dim;
  const int kvld = kv_heads * head_dim;

  const std::vector<float> q = bf16_round(make_data(size_t(seq) * qld, 4101u, 0.3f));
  const std::vector<float> k = bf16_round(make_data(size_t(seq) * kvld, 4102u, 0.3f));
  const std::vector<float> v = bf16_round(make_data(size_t(seq) * kvld, 4103u, 1.0f));

  BfBuf dq(q), dk(k), dv(v);

  slopfab::text::CausalAttentionConfig cfg;
  cfg.seq_len = seq;
  cfg.num_heads = heads;
  cfg.num_kv_heads = kv_heads;
  cfg.head_dim = head_dim;
  CHECK_NEAR(slopfab::text::causal_attention_scale(cfg), 1.0 / std::sqrt(64.0), 1e-7);

  const float scale = slopfab::text::causal_attention_scale(cfg);
  const std::vector<float> want = cpu_attention(q, k, v, seq, heads, kv_heads, head_dim, scale, true);
  const std::vector<float> bidirectional =
      cpu_attention(q, k, v, seq, heads, kv_heads, head_dim, scale, false);

  // The result must not depend on the tiling. A broken online-softmax rescale
  // or a mask computed from the wrong row index is silent under one block size.
  std::vector<std::vector<float>> results;
  for (int bq : {32, 128, 300}) {
    cfg.query_block = bq;
    BfBuf dout(size_t(seq) * qld);
    Workspace ws;
    ws.reserve(slopfab::text::causal_attention_workspace_bytes(cfg));
    slopfab::text::causal_attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                           ws);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    results.push_back(dout.host());
    CHECK_CLOSE_REL(want, results.back(), 1e-3, 1e-2,
                    ("causal gqa attention, query_block " + std::to_string(bq)).c_str());
  }
  CHECK_CLOSE_REL(results[0], results[1], 1e-3, 1e-2, "causal attention query_block 32 == 128");
  CHECK_CLOSE_REL(results[0], results[2], 1e-3, 1e-2, "causal attention query_block 32 == 300");

  const std::vector<float>& got = results[1];

  // Hazard 1: bidirectional attention. It changes every row but the last.
  CHECK_MSG(max_abs_diff(bidirectional, got) > 0.05,
            "attention must be causal, not bidirectional (max diff %.4g)",
            max_abs_diff(bidirectional, got));

  // Hazard 2: the grouped-query mapping. Reading kv head `h mod kv_heads`
  // instead of `h / group` is finite and correctly shaped.
  {
    std::vector<float> k_perm(k.size());
    std::vector<float> v_perm(v.size());
    for (int t = 0; t < seq; ++t) {
      for (int kv = 0; kv < kv_heads; ++kv) {
        const int other = (kv + 1) % kv_heads;
        for (int d = 0; d < head_dim; ++d) {
          k_perm[size_t(t) * kvld + kv * head_dim + d] = k[size_t(t) * kvld + other * head_dim + d];
          v_perm[size_t(t) * kvld + kv * head_dim + d] = v[size_t(t) * kvld + other * head_dim + d];
        }
      }
    }
    const std::vector<float> wrong =
        cpu_attention(q, k_perm, v_perm, seq, heads, kv_heads, head_dim, scale, true);
    CHECK_MSG(max_abs_diff(wrong, got) > 0.01,
              "query head h must read kv head h/(H/Hkv) (max diff %.4g)",
              max_abs_diff(wrong, got));
  }

  // Hazard 3: the cutoff itself. Truncating k and v to the first `t+1` rows
  // must not change row `t`'s output — a token cannot see its successors.
  {
    const int t = 97;
    slopfab::text::CausalAttentionConfig short_cfg = cfg;
    short_cfg.seq_len = t + 1;
    short_cfg.query_block = 128;
    BfBuf dshort(size_t(t + 1) * qld);
    Workspace ws;
    ws.reserve(slopfab::text::causal_attention_workspace_bytes(short_cfg));
    slopfab::text::causal_attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dshort.p(),
                                           short_cfg, ws);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> truncated = dshort.host();

    double worst = 0.0;
    for (int i = 0; i < qld; ++i) {
      worst = std::max(worst, std::fabs(double(truncated[size_t(t) * qld + i]) -
                                        got[size_t(t) * qld + i]));
    }
    CHECK_MSG(worst < 1e-2, "row %d changed by %.4g when the keys after it were removed; the "
                            "causal cutoff is not being applied",
              t, worst);
  }

  // Row 0 attends only to itself, so its output is exactly v[0] per head.
  double row0 = 0.0;
  for (int h = 0; h < heads; ++h) {
    const int kv = h / (heads / kv_heads);
    for (int d = 0; d < head_dim; ++d) {
      row0 = std::max(row0, std::fabs(double(got[size_t(h) * head_dim + d]) -
                                      v[size_t(kv) * head_dim + d]));
    }
  }
  CHECK_MSG(row0 < 1e-2, "row 0 must equal v[0] exactly; it differs by %.4g", row0);
}

SLOPFAB_TEST_CATEGORY(encoder_convrot_cross_check, "synthetic") {
  // Spec section 5.4. H is involutory, so de-rotating a weight and running it
  // against an *unrotated* activation must agree with the normal rotated path.
  // A wrong ConvRot is silent — it computes `x H W^T`, well-scaled noise — so
  // this is the highest-value bring-up check in the module.
  CublasScope cb;
  const int rows = 19;
  const int in_features = 512;  // two 256-wide groups
  const int out_features = 96;
  const int group = 256;

  // Random int8 and random per-row scales *are* the rotated weight, exactly:
  // the quantiser is symmetric per row at /127, so `int8 * scale` is not an
  // approximation of anything (spec section 5.1).
  std::vector<int8_t> w_int8(size_t(out_features) * in_features);
  uint32_t seed = 4201u;
  for (size_t i = 0; i < w_int8.size(); ++i) {
    seed = seed * 1664525u + 1013904223u;
    w_int8[i] = int8_t(int(seed >> 24) - 128);
  }
  std::vector<float> scales(out_features);
  for (int o = 0; o < out_features; ++o) scales[o] = 1e-3f * float(1 + o % 7);

  std::vector<float> w_rot(w_int8.size());
  for (int o = 0; o < out_features; ++o) {
    for (int i = 0; i < in_features; ++i) {
      w_rot[size_t(o) * in_features + i] = float(w_int8[size_t(o) * in_features + i]) * scales[o];
    }
  }

  const std::vector<float> x = bf16_round(make_data(size_t(rows) * in_features, 4202u, 0.1f));

  // (b) de-rotate the weight, multiply the unrotated activation.
  const std::vector<float> H = hadamard(group);
  const std::vector<float> w_plain = rotate_rows(w_rot, out_features, in_features, group, H);
  const std::vector<float> want = cpu_matmul_nt(x, w_plain, rows, out_features, in_features);

  // The hazard: no activation rotation at all, i.e. x W_rot^T.
  const std::vector<float> hazard = cpu_matmul_nt(x, w_rot, rows, out_features, in_features);

  // (a) the shipped path: int8 + ConvRot on the activation.
  DeviceBuffer<int8_t> dw(w_int8.size());
  dw.copy_from_host(w_int8.data(), w_int8.size());
  DeviceBuffer<float> dscale = to_device(scales);
  BfBuf dx(x);
  BfBuf dy(size_t(rows) * out_features);

  QuantWeight qw;
  qw.format = QuantFormat::kI8;
  qw.data = dw.get();
  qw.out_features = out_features;
  qw.in_features = in_features;
  qw.weight_scale = dscale.get();
  qw.per_channel_scale = true;
  qw.convrot = true;
  qw.convrot_group = group;

  slopfab::cuda::LinearRunner runner;
  runner.init(cb.h, nullptr);
  Workspace ws;
  ws.reserve(slopfab::cuda::linear_workspace_bytes(qw, rows, slopfab::cuda::ComputeType::kBF16));
  runner.forward(qw, dx.p(), rows, dy.p(), ws);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = dy.host();

  const double err = rms_relative_error(want, got);
  CHECK_MSG(err < 2e-2,
            "ConvRot: rotated activation must equal de-rotated weight; worst error is %.3e of the "
            "output RMS (%.4g)",
            err, rms(want));
  CHECK_MSG(max_abs_diff(hazard, got) > 0.05,
            "the activation must be rotated online; skipping it computes x H W^T, which is "
            "well-scaled noise (max diff %.4g)",
            max_abs_diff(hazard, got));

  // And the butterfly is its own inverse, which catches a stride or sign error
  // in the transform independently of any weight.
  const std::vector<float> twice = rotate_rows(rotate_rows(x, rows, in_features, group, H), rows,
                                               in_features, group, H);
  CHECK_CLOSE(x, twice, 1e-4, "H is involutory: H(H(v)) == v");
}

SLOPFAB_TEST_CATEGORY(encoder_layer_vs_cpu_reference, "synthetic") {
  // One full decoder layer at small synthetic dimensions, written from spec
  // section 4 and checked against a CPU implementation of the same. Widths are
  // multiples of the 256-wide ConvRot group where they are contracted over.
  CublasScope cb;
  const int L = 24;
  const int hidden = 256;
  const int heads = 4;
  const int kv_heads = 2;
  const int head_dim = 64;
  const int inner = 512;
  const int q_width = heads * head_dim;    // 256
  const int kv_width = kv_heads * head_dim;  // 128
  const float eps = 1e-6f;
  const int group = 256;
  const std::vector<float> H = hadamard(group);

  // Each projection: random int8 plus per-row scales, which *is* the rotated
  // weight; the CPU reference then de-rotates it and multiplies the unrotated
  // activation, so the two agree exactly up to arithmetic precision.
  struct Proj {
    std::vector<int8_t> i8;
    std::vector<float> scale;
    std::vector<float> plain;  // de-rotated, for the CPU reference
    DeviceBuffer<int8_t> d_w;
    DeviceBuffer<float> d_s;
    QuantWeight qw;
  };

  auto make_proj = [&](int out_features, int in_features, uint32_t seed) {
    Proj p;
    p.i8.resize(size_t(out_features) * in_features);
    uint32_t s = seed;
    for (size_t i = 0; i < p.i8.size(); ++i) {
      s = s * 1664525u + 1013904223u;
      p.i8[i] = int8_t(int(s >> 24) - 128);
    }
    p.scale.resize(out_features);
    for (int o = 0; o < out_features; ++o) p.scale[o] = 2e-3f * float(1 + (o * 13) % 5);

    std::vector<float> rot(p.i8.size());
    for (int o = 0; o < out_features; ++o) {
      for (int i = 0; i < in_features; ++i) {
        rot[size_t(o) * in_features + i] = float(p.i8[size_t(o) * in_features + i]) * p.scale[o];
      }
    }
    p.plain = rotate_rows(rot, out_features, in_features, group, H);

    p.d_w.allocate(p.i8.size());
    p.d_w.copy_from_host(p.i8.data(), p.i8.size());
    p.d_s.allocate(p.scale.size());
    p.d_s.copy_from_host(p.scale.data(), p.scale.size());

    p.qw.format = QuantFormat::kI8;
    p.qw.data = p.d_w.get();
    p.qw.out_features = out_features;
    p.qw.in_features = in_features;
    p.qw.weight_scale = p.d_s.get();
    p.qw.per_channel_scale = true;
    p.qw.convrot = true;
    p.qw.convrot_group = group;
    return p;
  };

  Proj wq = make_proj(q_width, hidden, 4301u);
  Proj wk = make_proj(kv_width, hidden, 4302u);
  Proj wv = make_proj(kv_width, hidden, 4303u);
  Proj wo = make_proj(hidden, q_width, 4304u);
  Proj wg = make_proj(inner, hidden, 4305u);
  Proj wu = make_proj(inner, hidden, 4306u);
  Proj wd = make_proj(hidden, inner, 4307u);

  const std::vector<float> x0 = bf16_round(make_data(size_t(L) * hidden, 4310u, 1.0f));
  const std::vector<float> ln_in = bf16_round(make_data(hidden, 4311u, 0.5f));
  const std::vector<float> ln_post = bf16_round(make_data(hidden, 4312u, 0.5f));
  const std::vector<float> q_norm = bf16_round(make_data(head_dim, 4313u, 1.0f));
  const std::vector<float> k_norm = bf16_round(make_data(head_dim, 4314u, 1.0f));

  const std::vector<float> inv = slopfab::text::rope_inv_freq(head_dim, 5.0e6f);
  std::vector<float> cos;
  std::vector<float> sin;
  slopfab::text::build_rope_tables(L, inv, cos, sin);

  const float scale = 1.0f / std::sqrt(float(head_dim));

  // --- CPU reference. `qk_before_rope` false and `causal` false are the two
  // silent hazards this test exists to catch.
  auto cpu_layer = [&](bool qk_before_rope, bool causal, bool silu_on_gate) {
    std::vector<float> x = x0;
    const std::vector<float> n = cpu_rmsnorm(x, ln_in, L, hidden, eps);
    std::vector<float> q = cpu_matmul_nt(n, wq.plain, L, q_width, hidden);
    std::vector<float> k = cpu_matmul_nt(n, wk.plain, L, kv_width, hidden);
    const std::vector<float> v = cpu_matmul_nt(n, wv.plain, L, kv_width, hidden);

    if (qk_before_rope) {
      q = cpu_rmsnorm(q, q_norm, L * heads, head_dim, eps);
      k = cpu_rmsnorm(k, k_norm, L * kv_heads, head_dim, eps);
      q = cpu_rope(q, cos, sin, L, heads, head_dim, head_dim);
      k = cpu_rope(k, cos, sin, L, kv_heads, head_dim, head_dim);
    } else {
      q = cpu_rope(q, cos, sin, L, heads, head_dim, head_dim);
      k = cpu_rope(k, cos, sin, L, kv_heads, head_dim, head_dim);
      q = cpu_rmsnorm(q, q_norm, L * heads, head_dim, eps);
      k = cpu_rmsnorm(k, k_norm, L * kv_heads, head_dim, eps);
    }

    const std::vector<float> a =
        cpu_attention(q, k, v, L, heads, kv_heads, head_dim, scale, causal);
    const std::vector<float> o = cpu_matmul_nt(a, wo.plain, L, hidden, q_width);
    for (size_t i = 0; i < x.size(); ++i) x[i] += o[i];

    const std::vector<float> n2 = cpu_rmsnorm(x, ln_post, L, hidden, eps);
    const std::vector<float> g = cpu_matmul_nt(n2, wg.plain, L, inner, hidden);
    const std::vector<float> u = cpu_matmul_nt(n2, wu.plain, L, inner, hidden);
    std::vector<float> h(g.size());
    for (size_t i = 0; i < h.size(); ++i) {
      h[i] = silu_on_gate ? silu(g[i]) * u[i] : g[i] * silu(u[i]);
    }
    const std::vector<float> d = cpu_matmul_nt(h, wd.plain, L, hidden, inner);
    for (size_t i = 0; i < x.size(); ++i) x[i] += d[i];
    return x;
  };

  const std::vector<float> want = cpu_layer(true, true, true);
  const std::vector<float> hazard_rope_first = cpu_layer(false, true, true);
  const std::vector<float> hazard_bidirectional = cpu_layer(true, false, true);
  const std::vector<float> hazard_silu_on_up = cpu_layer(true, true, false);

  // --- GPU.
  BfBuf dx(x0);
  BfBuf d_ln_in(ln_in), d_ln_post(ln_post), d_qnorm(q_norm), d_knorm(k_norm);
  DeviceBuffer<float> dcos = to_device(cos);
  DeviceBuffer<float> dsin = to_device(sin);

  slopfab::text::LayerWeights w;
  w.q_proj = wq.qw;
  w.k_proj = wk.qw;
  w.v_proj = wv.qw;
  w.o_proj = wo.qw;
  w.gate_proj = wg.qw;
  w.up_proj = wu.qw;
  w.down_proj = wd.qw;
  w.input_layernorm = d_ln_in.p();
  w.post_attention_layernorm = d_ln_post.p();
  w.q_norm = d_qnorm.p();
  w.k_norm = d_knorm.p();

  slopfab::text::LayerDims dims;
  dims.num_tokens = L;
  dims.hidden = hidden;
  dims.num_heads = heads;
  dims.num_kv_heads = kv_heads;
  dims.head_dim = head_dim;
  dims.intermediate = inner;
  dims.rms_norm_eps = eps;

  slopfab::cuda::LinearRunner runner;
  runner.init(cb.h, nullptr);
  Workspace ws;
  ws.reserve(slopfab::text::layer_workspace_bytes(dims));
  slopfab::text::encoder_layer_forward(cb.h, nullptr, runner, w, dims, dcos.get(), dsin.get(),
                                      dx.p(), ws);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = dx.host();

  // bf16 activations at every stage and eleven GEMMs deep, so a percent or so
  // of the output RMS is the floor. Each hazard below is two orders of
  // magnitude further away than that, which is the point.
  const double err_rms = rms_error_ratio(want, got);
  const double err_max = rms_relative_error(want, got);
  std::printf("  decoder layer vs CPU reference: %.3e rms, %.3e worst (output RMS %.4g)\n", err_rms,
              err_max, rms(want));
  // Eleven bf16 GEMMs deep with cancellation in every one; ~1.6e-2 rms is the
  // measured floor and matches the per-element bf16 error propagated through
  // the 512-term down_proj contraction.
  CHECK_MSG(err_rms < 3e-2, "one decoder layer vs CPU reference: %.3e rms error", err_rms);
  CHECK_MSG(err_max < 1.5e-1, "one decoder layer vs CPU reference: %.3e worst error", err_max);

  const double rope_first = rms_error_ratio(want, hazard_rope_first);
  const double bidirectional = rms_error_ratio(want, hazard_bidirectional);
  const double silu_on_up = rms_error_ratio(want, hazard_silu_on_up);
  std::printf("  hazard distances (rms): qk-after-rope %.3g, bidirectional %.3g, silu-on-up %.3g\n",
              rope_first, bidirectional, silu_on_up);
  CHECK_MSG(rms_error_ratio(hazard_rope_first, got) > err_rms * 20.0,
            "QK-norm must be applied BEFORE RoPE (spec 4.2)");
  CHECK_MSG(rms_error_ratio(hazard_bidirectional, got) > err_rms * 20.0,
            "attention inside the layer must be causal (spec 3)");
  CHECK_MSG(rms_error_ratio(hazard_silu_on_up, got) > err_rms * 20.0,
            "SiLU goes on gate_proj, not on up_proj (spec 4.3)");
}
