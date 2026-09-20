#include "detail/encoder_fixture.h"

SLOPFAB_TEST_CATEGORY(encoder_real_checkpoint_convrot_cross_check, "checkpoint") {
  const std::string path = find_checkpoint();
  if (path.empty()) {
    SKIP_MISSING_FIXTURE("  text encoder checkpoint not present; skipping\n");
    return;
  }
  CublasScope cb;

  slopfab::SafeTensors st;
  st.open(path);
  slopfab::text::EncoderConfig cfg;
  slopfab::text::validate_checkpoint(st, cfg);
  CHECK(st.tensor_count() == 1602);

  // Layer 0's k_proj: [1024, 5120] int8, small enough to run both ways.
  const slopfab::TensorView& wv = st.at("model.layers.0.self_attn.k_proj.weight");
  const slopfab::TensorView& sv = st.at("model.layers.0.self_attn.k_proj.weight_scale");
  const int out_features = 1024;
  const int in_features = 5120;
  const int rows = 8;

  DeviceBuffer<int8_t> dw(size_t(out_features) * in_features);
  dw.copy_from_host(static_cast<const int8_t*>(wv.data), dw.size());
  DeviceBuffer<float> dscale(out_features);
  dscale.copy_from_host(static_cast<const float*>(sv.data), dscale.size());

  // Every int8 row's max absolute value is exactly 127 — the quantiser is
  // symmetric per row at /127 (spec section 5.1). Worth pinning: it is what
  // makes `int8 * weight_scale` exact rather than approximate.
  const int8_t* w_host = static_cast<const int8_t*>(wv.data);
  int rows_at_127 = 0;
  for (int o = 0; o < out_features; ++o) {
    int amax = 0;
    for (int i = 0; i < in_features; ++i) {
      amax = std::max(amax, std::abs(int(w_host[size_t(o) * in_features + i])));
    }
    if (amax == 127) ++rows_at_127;
  }
  CHECK_MSG(rows_at_127 == out_features, "%d of %d int8 rows reach 127", rows_at_127, out_features);

  const std::vector<float> x = bf16_round(make_data(size_t(rows) * in_features, 4401u, 0.05f));
  BfBuf dx(x);

  slopfab::cuda::LinearRunner runner;
  runner.init(cb.h, nullptr);
  Workspace ws;

  // (a) the shipped path: int8 weight, activation rotated online.
  QuantWeight qa;
  qa.format = QuantFormat::kI8;
  qa.data = dw.get();
  qa.out_features = out_features;
  qa.in_features = in_features;
  qa.weight_scale = dscale.get();
  qa.per_channel_scale = true;
  qa.convrot = true;
  qa.convrot_group = 256;

  BfBuf ya(size_t(rows) * out_features);
  ws.reserve(slopfab::cuda::linear_workspace_bytes(qa, rows, slopfab::cuda::ComputeType::kBF16));
  runner.forward(qa, dx.p(), rows, ya.p(), ws);

  // (b) de-rotate the dequantised weight and use the unrotated activation.
  // H is involutory, so the same butterfly undoes it.
  BfBuf w_dequant(size_t(out_features) * in_features);
  BfBuf w_plain(size_t(out_features) * in_features);
  slopfab::cuda::launch_dequant_i8_per_channel(dw.get(), dscale.get(), w_dequant.p(), out_features,
                                              in_features, nullptr);
  slopfab::cuda::launch_convrot(w_dequant.p(), w_plain.p(), out_features, in_features, 256, nullptr);

  QuantWeight qb;
  qb.format = QuantFormat::kBF16;
  qb.data = w_plain.p();
  qb.out_features = out_features;
  qb.in_features = in_features;
  qb.convrot = false;

  BfBuf yb(size_t(rows) * out_features);
  runner.forward(qb, dx.p(), rows, yb.p(), ws);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());

  const std::vector<float> a = ya.host();
  const std::vector<float> b = yb.host();
  const double err = rms_relative_error(a, b);
  std::printf("  convrot cross-check on real layer-0 k_proj: worst error %.3e of output RMS "
              "(%.4g), max abs %.3e\n",
              err, rms(a), max_abs_diff(a, b));
  // Path (b) rounds the de-rotated weight to bf16, which redistributes the int8
  // quantisation noise; the two are not bitwise identical by construction
  // (spec section 5.4).
  CHECK_MSG(err < 3e-2,
            "ConvRot cross-check on real weights: worst error %.3e of output RMS", err);

  // The measured k_norm extreme from spec section 4.1, which is why attention
  // scores need fp32 accumulation. Corruption would look the same.
  std::vector<float> knorm;
  slopfab::to_f32(st.at("model.layers.0.self_attn.k_norm.weight"), knorm);
  CHECK(knorm.size() == 128);
  const float kmax = *std::max_element(knorm.begin(), knorm.end());
  CHECK_MSG(kmax > 20.0f && kmax < 21.0f, "layer 0 k_norm max is %.4f, expected ~20.75", kmax);
}
