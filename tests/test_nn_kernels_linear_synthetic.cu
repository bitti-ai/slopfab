#include "detail/nn_kernels_fixture.h"
#include "detail/nn_weight_fixture.h"

SLOPFAB_TEST_CATEGORY(nn_dequant_f8e4m3_all_patterns, "synthetic") {
  std::vector<uint8_t> patterns(256);
  for (int i = 0; i < 256; ++i) patterns[i] = uint8_t(i);

  DeviceBuffer<uint8_t> dsrc(256);
  dsrc.copy_from_host(patterns.data(), 256);
  const std::vector<float> one(1, 1.0f);
  DeviceBuffer<float> dscale = to_device(one);
  BfBuf ddst(256);
  slopfab::cuda::launch_dequant_f8e4m3(dsrc.get(), dscale.get(), ddst.p(), 256, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = ddst.host();

  // e4m3 has 3 mantissa bits and bf16 has 7, so the round trip through bf16 is
  // lossless and exact equality is the right bar. 0x7F and 0xFF are NaN, whose
  // payload is not something either side promises.
  size_t mismatches = 0;
  int first = -1;
  size_t nan_checked = 0;
  for (int i = 0; i < 256; ++i) {
    const float want = slopfab::f8_e4m3_to_f32(patterns[i]);
    if (std::isnan(want)) {
      ++nan_checked;
      if (!std::isnan(got[i])) {
        if (first < 0) first = i;
        ++mismatches;
      }
      continue;
    }
    if (std::memcmp(&want, &got[i], sizeof(float)) != 0) {
      if (first < 0) first = i;
      ++mismatches;
    }
  }
  CHECK_MSG(mismatches == 0,
            "dequant_f8e4m3: %zu of 256 patterns differ from dtype.h; first at 0x%02X",
            mismatches, first < 0 ? 0 : first);
  CHECK(nan_checked == 2);

  // A non-unit scale multiplies, it does not re-quantise.
  const std::vector<float> s(1, 0.0081264f);
  DeviceBuffer<float> dscale2 = to_device(s);
  slopfab::cuda::launch_dequant_f8e4m3(dsrc.get(), dscale2.get(), ddst.p(), 256, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> scaled = ddst.host();
  std::vector<float> want_scaled(256);
  std::vector<float> got_scaled(256);
  size_t kept = 0;
  for (int i = 0; i < 256; ++i) {
    const float w = slopfab::f8_e4m3_to_f32(patterns[i]);
    if (std::isnan(w)) continue;
    want_scaled[kept] = slopfab::bf16_to_f32(slopfab::f32_to_bf16(w * s[0]));
    got_scaled[kept] = scaled[i];
    ++kept;
  }
  want_scaled.resize(kept);
  got_scaled.resize(kept);
  CHECK_CLOSE(want_scaled, got_scaled, 0.0, "dequant_f8e4m3 with weight_scale");
}

SLOPFAB_TEST_CATEGORY(nn_quantize_f8e4m3_roundtrip, "synthetic") {
  std::vector<float> values;
  std::vector<uint8_t> expect;
  for (int i = 0; i < 256; ++i) {
    const float v = slopfab::f8_e4m3_to_f32(uint8_t(i));
    if (std::isnan(v)) continue;
    if (v == 0.0f && (i & 0x80)) continue;  // -0 encodes back as +0's pattern only by sign
    values.push_back(v);
    expect.push_back(uint8_t(i));
  }

  BfBuf dsrc(values);
  DeviceBuffer<uint8_t> ddst(values.size());
  slopfab::cuda::launch_quantize_f8e4m3(dsrc.p(), 1.0f, ddst.get(), values.size(), nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint8_t> got(values.size());
  ddst.copy_to_host(got.data(), got.size());

  size_t mismatches = 0;
  int first = -1;
  for (size_t i = 0; i < got.size(); ++i) {
    if (got[i] != expect[i]) {
      if (first < 0) first = int(i);
      ++mismatches;
    }
  }
  CHECK_MSG(mismatches == 0, "quantize_f8e4m3 round trip: %zu of %zu differ, first expect 0x%02X",
            mismatches, got.size(), first < 0 ? 0 : expect[first]);

  // Saturation, not wraparound, and never the NaN encoding.
  const std::vector<float> big = {1000.0f, -1000.0f, 448.0f, 449.0f, 460.0f};
  BfBuf dbig(big);
  DeviceBuffer<uint8_t> dbigq(big.size());
  slopfab::cuda::launch_quantize_f8e4m3(dbig.p(), 1.0f, dbigq.get(), big.size(), nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint8_t> bq(big.size());
  dbigq.copy_to_host(bq.data(), bq.size());
  CHECK(bq[0] == 0x7E);
  CHECK(bq[1] == 0xFE);
  CHECK(bq[2] == 0x7E);
  CHECK(bq[3] == 0x7E);
  CHECK(bq[4] == 0x7E);

  // A zero input_scale marks a full-precision layer; quantising is a bug.
  bool threw = false;
  try {
    slopfab::cuda::launch_quantize_f8e4m3(dbig.p(), 0.0f, dbigq.get(), big.size(), nullptr);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}

SLOPFAB_TEST_CATEGORY(nn_dequant_i8_per_channel, "synthetic") {
  const int out_features = 7;
  const int in_features = 40;  // multiple of 8 so the packed path runs
  std::vector<int8_t> w(size_t(out_features) * in_features);
  for (size_t i = 0; i < w.size(); ++i) w[i] = int8_t(int(i * 37 % 255) - 127);
  std::vector<float> scale(out_features);
  for (int o = 0; o < out_features; ++o) scale[o] = 1e-3f * float(o + 1);

  DeviceBuffer<int8_t> dw(w.size());
  dw.copy_from_host(w.data(), w.size());
  DeviceBuffer<float> dscale = to_device(scale);
  BfBuf ddst(w.size());
  slopfab::cuda::launch_dequant_i8_per_channel(dw.get(), dscale.get(), ddst.p(), out_features,
                                              in_features, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> want(w.size());
  for (int o = 0; o < out_features; ++o) {
    for (int i = 0; i < in_features; ++i) {
      want[size_t(o) * in_features + i] = slopfab::bf16_to_f32(
          slopfab::f32_to_bf16(float(w[size_t(o) * in_features + i]) * scale[o]));
    }
  }
  CHECK_CLOSE(want, ddst.host(), 0.0, "dequant_i8 per output channel");
}

SLOPFAB_TEST_CATEGORY(nn_w4a8_weight_and_activation_decode, "synthetic") {
  const int rows = 2;
  const int out_features = 3;
  const int in_features = 256;
  const int group_size = 16;

  std::vector<int8_t> packed(size_t(out_features) * in_features / 2);
  std::vector<int8_t> expected_weight(size_t(out_features) * in_features);
  for (int o = 0; o < out_features; ++o) {
    for (int i = 0; i < in_features; i += 2) {
      const int lo = (o * 5 + i) & 15;
      const int hi = (o * 7 + i + 1) & 15;
      packed[size_t(o) * (in_features / 2) + i / 2] =
          static_cast<int8_t>(lo | (hi << 4));
      expected_weight[size_t(o) * in_features + i] = static_cast<int8_t>(lo - 8);
      expected_weight[size_t(o) * in_features + i + 1] = static_cast<int8_t>(hi - 8);
    }
  }
  const std::vector<uint8_t> group_scale(
      size_t(out_features) * in_features / group_size, 0x38);  // E4M3 1.0
  std::vector<float> codebook(16);
  for (int i = 0; i < 16; ++i) codebook[i] = static_cast<float>(i - 8);

  DeviceBuffer<int8_t> dpacked(packed.size()), dweight(expected_weight.size());
  DeviceBuffer<uint8_t> dgroup(group_scale.size());
  dpacked.copy_from_host(packed.data(), packed.size());
  dgroup.copy_from_host(group_scale.data(), group_scale.size());
  DeviceBuffer<float> dcodebook = to_device(codebook);
  slopfab::cuda::launch_dequant_w4a8_weight(
      dpacked.get(), dgroup.get(), dcodebook.get(), dweight.get(), out_features,
      in_features, group_size, nullptr);
  std::vector<int8_t> got_weight(expected_weight.size());
  dweight.copy_to_host(got_weight.data(), got_weight.size());
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(got_weight == expected_weight);

  // Two exactly representable rows make the independently constructed
  // Hadamard reference unambiguous after dynamic INT8 quantization.
  std::vector<float> x(size_t(rows) * in_features, 0.0f);
  x[0] = 1.0f;
  std::fill(x.begin() + in_features, x.end(), 1.0f);
  std::vector<uint16_t> xbits(x.size());
  for (size_t i = 0; i < x.size(); ++i) xbits[i] = slopfab::f32_to_f16(x[i]);
  DeviceBuffer<uint16_t> dx(xbits.size());
  DeviceBuffer<int8_t> dq(x.size());
  DeviceBuffer<float> dscale(rows);
  dx.copy_from_host(xbits.data(), xbits.size());
  slopfab::cuda::launch_quantize_w4a8_activation(
      reinterpret_cast<const __half*>(dx.get()), dq.get(), dscale.get(), rows,
      in_features, nullptr);
  std::vector<int8_t> q(x.size());
  std::vector<float> scale(rows);
  dq.copy_to_host(q.data(), q.size());
  dscale.copy_to_host(scale.data(), scale.size());
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> H = kron_power(&kH4[0][0], 4, 4);
  for (float& value : H) value /= 16.0f;
  for (int r = 0; r < rows; ++r) {
    float amax = 0.0f;
    std::vector<float> rotated(in_features);
    for (int i = 0; i < in_features; ++i) {
      float sum = 0.0f;
      for (int j = 0; j < in_features; ++j)
        sum += H[size_t(i) * in_features + j] * x[size_t(r) * in_features + j];
      rotated[i] = sum;
      amax = std::max(amax, std::fabs(sum));
    }
    const float want_scale = std::max(amax / 127.0f, 1.0e-30f);
    CHECK_NEAR(scale[r], want_scale, 1e-8);
    for (int i = 0; i < in_features; ++i) {
      const int want = std::max(-128, std::min(127,
          static_cast<int>(std::nearbyint(rotated[i] / want_scale))));
      CHECK(q[size_t(r) * in_features + i] == want);
    }
  }

  std::vector<int32_t> accum = {100, -200, 300, -400, 500, -600};
  const std::vector<float> activation_scale = {0.25f, 0.5f};
  const std::vector<float> weight_scale = {0.1f, 0.2f, 0.3f};
  DeviceBuffer<int32_t> daccum = to_device_i32(accum);
  DeviceBuffer<float> dax = to_device(activation_scale);
  DeviceBuffer<float> dwx = to_device(weight_scale);
  slopfab::cuda::launch_dequant_w4a8_output(
      daccum.get(), dax.get(), dwx.get(), rows, out_features, nullptr);
  std::vector<float> result(accum.size());
  SLOPFAB_CUDA_CHECK(cudaMemcpy(result.data(), daccum.get(),
                               result.size() * sizeof(float),
                               cudaMemcpyDeviceToHost));
  for (int r = 0; r < rows; ++r) {
    for (int o = 0; o < out_features; ++o) {
      const size_t i = size_t(r) * out_features + o;
      CHECK_NEAR(result[i], static_cast<float>(accum[i]) *
                                activation_scale[r] * weight_scale[o], 1e-6);
    }
  }
}

SLOPFAB_TEST_CATEGORY(nn_convrot_matches_regular_hadamard, "synthetic") {
  const int group = 256;
  const int rows = 5;

  // H = kron^4(h4) / 16, built from the matrix quoted in docs/convrot_notes.md.
  std::vector<float> H = kron_power(&kH4[0][0], 4, 4);
  for (float& v : H) v /= 16.0f;

  // Defining properties: symmetric, orthogonal, involutory.
  double sym = 0.0;
  for (int i = 0; i < group; ++i) {
    for (int j = 0; j < group; ++j) {
      sym = std::max(sym, std::fabs(double(H[size_t(i) * group + j]) - H[size_t(j) * group + i]));
    }
  }
  CHECK_MSG(sym == 0.0, "reference H must be symmetric (max asymmetry %.3g)", sym);

  double off = 0.0;
  double diag = 0.0;
  for (int i = 0; i < group; ++i) {
    for (int j = 0; j < group; ++j) {
      double acc = 0.0;
      for (int k = 0; k < group; ++k) acc += double(H[size_t(i) * group + k]) * H[size_t(k) * group + j];
      if (i == j) {
        diag = std::max(diag, std::fabs(acc - 1.0));
      } else {
        off = std::max(off, std::fabs(acc));
      }
    }
  }
  CHECK_MSG(diag < 1e-6 && off < 1e-6, "reference H must be involutory (diag err %.3g, off %.3g)",
            diag, off);

  // The kernel, against a plain CPU matmul with that H.
  const std::vector<float> x = make_data(size_t(rows) * group, 131u, 1.0f);
  std::vector<float> want(x.size());
  for (int r = 0; r < rows; ++r) {
    for (int i = 0; i < group; ++i) {
      double acc = 0.0;
      for (int j = 0; j < group; ++j) {
        acc += double(H[size_t(i) * group + j]) * x[size_t(r) * group + j];
      }
      want[size_t(r) * group + i] = float(acc);
    }
  }

  DeviceBuffer<float> dx = to_device(x);
  DeviceBuffer<float> dy(x.size());
  slopfab::cuda::launch_convrot_f32(dx.get(), dy.get(), rows, group, group, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = to_host(dy);
  CHECK_CLOSE(want, got, 1e-5, "convrot == kron^4(regular h4)/16");

  // And explicitly not the Sylvester / Walsh-Hadamard transform.
  std::vector<float> W = kron_power(&kSylvester2[0][0], 2, 8);
  for (float& v : W) v /= 16.0f;
  std::vector<float> sylvester(x.size());
  for (int r = 0; r < rows; ++r) {
    for (int i = 0; i < group; ++i) {
      double acc = 0.0;
      for (int j = 0; j < group; ++j) {
        acc += double(W[size_t(i) * group + j]) * x[size_t(r) * group + j];
      }
      sylvester[size_t(r) * group + i] = float(acc);
    }
  }
  CHECK_MSG(max_abs_diff(sylvester, got) > 0.1,
            "convrot must not be the Sylvester transform (max diff %.4g)",
            max_abs_diff(sylvester, got));

  // Involutory in practice: applying it twice returns the input.
  DeviceBuffer<float> dz(x.size());
  slopfab::cuda::launch_convrot_f32(dy.get(), dz.get(), rows, group, group, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(x, to_host(dz), 1e-5, "convrot applied twice is the identity");

  // Orthogonal: the row norm is preserved.
  for (int r = 0; r < rows; ++r) {
    double n_in = 0.0;
    double n_out = 0.0;
    for (int i = 0; i < group; ++i) {
      n_in += double(x[size_t(r) * group + i]) * x[size_t(r) * group + i];
      n_out += double(got[size_t(r) * group + i]) * got[size_t(r) * group + i];
    }
    CHECK_MSG(std::fabs(std::sqrt(n_in) - std::sqrt(n_out)) < 1e-5 * std::sqrt(n_in),
              "convrot row %d norm %.9g -> %.9g", r, std::sqrt(n_in), std::sqrt(n_out));
  }

  // A multi-group row, so the block-diagonal structure is exercised.
  const int dim = 3 * group;
  const std::vector<float> xm = make_data(size_t(rows) * dim, 132u, 1.0f);
  DeviceBuffer<float> dxm = to_device(xm);
  DeviceBuffer<float> dym(xm.size());
  slopfab::cuda::launch_convrot_f32(dxm.get(), dym.get(), rows, dim, group, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> gm = to_host(dym);
  std::vector<float> wm(xm.size());
  for (int r = 0; r < rows; ++r) {
    for (int g = 0; g < dim / group; ++g) {
      const size_t base = size_t(r) * dim + size_t(g) * group;
      for (int i = 0; i < group; ++i) {
        double acc = 0.0;
        for (int j = 0; j < group; ++j) acc += double(H[size_t(i) * group + j]) * xm[base + j];
        wm[base + i] = float(acc);
      }
    }
  }
  CHECK_CLOSE(wm, gm, 1e-5, "convrot over multiple groups");

  // Group 16 = 4^2 exercises a different stage count.
  const std::vector<float> x16 = make_data(size_t(rows) * 16, 133u, 1.0f);
  std::vector<float> H16 = kron_power(&kH4[0][0], 4, 2);
  for (float& v : H16) v /= 4.0f;
  std::vector<float> w16(x16.size());
  for (int r = 0; r < rows; ++r) {
    for (int i = 0; i < 16; ++i) {
      double acc = 0.0;
      for (int j = 0; j < 16; ++j) acc += double(H16[size_t(i) * 16 + j]) * x16[size_t(r) * 16 + j];
      w16[size_t(r) * 16 + i] = float(acc);
    }
  }
  DeviceBuffer<float> dx16 = to_device(x16);
  DeviceBuffer<float> dy16(x16.size());
  slopfab::cuda::launch_convrot_f32(dx16.get(), dy16.get(), rows, 16, 16, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(w16, to_host(dy16), 1e-6, "convrot at group 16");
}

SLOPFAB_TEST_CATEGORY(linear_bf16_and_fp8, "synthetic") {
  CublasScope cb;
  const int rows = 33;
  const int in_features = 256;
  const int out_features = 96;

  const std::vector<float> x = bf16_round(make_data(size_t(rows) * in_features, 141u, 0.1f));
  BfBuf dx(x);
  BfBuf dy(size_t(rows) * out_features);

  slopfab::cuda::LinearRunner runner;
  runner.init(cb.h, nullptr);

  // 1. dense bf16, with a bias.
  {
    const std::vector<float> w =
        bf16_round(make_data(size_t(out_features) * in_features, 142u, 0.1f));
    const std::vector<float> bias = make_data(out_features, 143u, 0.5f);
    BfBuf dw(w);
    DeviceBuffer<float> dbias = to_device(bias);

    slopfab::cuda::QuantWeight qw;
    qw.format = slopfab::cuda::QuantFormat::kBF16;
    qw.data = dw.p();
    qw.out_features = out_features;
    qw.in_features = in_features;
    qw.bias = dbias.get();
    qw.bias_format = slopfab::cuda::QuantFormat::kF32;
    CHECK(qw.stored_bytes() == size_t(out_features) * in_features * 2);

    Workspace ws;
    ws.reserve(slopfab::cuda::linear_workspace_bytes(qw, rows, slopfab::cuda::ComputeType::kBF16) +
               256);
    runner.forward(qw, dx.p(), rows, dy.p(), ws);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> want = cpu_matmul_nt(x, w, rows, out_features, in_features);
    for (int r = 0; r < rows; ++r) {
      for (int o = 0; o < out_features; ++o) want[size_t(r) * out_features + o] += bias[o];
    }
    CHECK_CLOSE_REL(want, dy.host(), 1e-3, 1e-2, "linear bf16 + bias");
  }

  // 2. fp8 e4m3 with a per-tensor weight_scale. The stored bytes are real fp8
  // patterns, so the reference dequantises with dtype.h rather than assuming.
  {
    std::vector<uint8_t> raw(size_t(out_features) * in_features);
    uint32_t s = 12345u;
    for (size_t i = 0; i < raw.size(); ++i) {
      s ^= s << 13;
      s ^= s >> 17;
      s ^= s << 5;
      uint8_t b = uint8_t(s & 0xFFu);
      if ((b & 0x7F) == 0x7F) b &= 0x7Eu;  // avoid the NaN encodings
      raw[i] = b;
    }
    const float wscale = 8.1264e-3f;
    DeviceBuffer<uint8_t> draw(raw.size());
    draw.copy_from_host(raw.data(), raw.size());
    const std::vector<float> sv(1, wscale);
    DeviceBuffer<float> dscale = to_device(sv);

    slopfab::cuda::QuantWeight qw;
    qw.format = slopfab::cuda::QuantFormat::kF8E4M3;
    qw.data = draw.get();
    qw.out_features = out_features;
    qw.in_features = in_features;
    qw.weight_scale = dscale.get();
    qw.input_scale = 3.53655e-2f;
    CHECK(qw.stored_bytes() == size_t(out_features) * in_features);

    std::vector<float> w(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
      w[i] = slopfab::bf16_to_f32(slopfab::f32_to_bf16(slopfab::f8_e4m3_to_f32(raw[i]) * wscale));
    }
    const std::vector<float> want = cpu_matmul_nt(x, w, rows, out_features, in_features);

    Workspace ws;
    ws.reserve(slopfab::cuda::linear_workspace_bytes(qw, rows, slopfab::cuda::ComputeType::kBF16) +
               256);
    runner.forward(qw, dx.p(), rows, dy.p(), ws);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> got_dequant = dy.host();
    CHECK_CLOSE_REL(want, got_dequant, 1e-3, 1e-2, "linear fp8 dequantise-then-GEMM");

    // set_native must not change the answer beyond tolerance. It currently
    // falls back to the same path; this pins the contract for when it does not.
    runner.set_native(true);
    runner.forward(qw, dx.p(), rows, dy.p(), ws);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CLOSE_REL(got_dequant, dy.host(), 1e-3, 1e-2, "linear fp8 native == dequantised");

    // The load-bearing case: no input_scale means full precision. It must still
    // produce the same result with native enabled.
    qw.input_scale = 0.0f;
    runner.forward(qw, dx.p(), rows, dy.p(), ws);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CLOSE_REL(got_dequant, dy.host(), 1e-3, 1e-2,
                    "linear fp8 with no input_scale stays full precision");
    runner.set_native(false);
  }

  // 3. fp32 path.
  {
    const std::vector<float> w = make_data(size_t(out_features) * in_features, 144u, 0.1f);
    DeviceBuffer<float> dw = to_device(w);
    slopfab::cuda::QuantWeight qw;
    qw.format = slopfab::cuda::QuantFormat::kF32;
    qw.data = dw.get();
    qw.out_features = out_features;
    qw.in_features = in_features;

    DeviceBuffer<float> fx = to_device(x);
    DeviceBuffer<float> fy(size_t(rows) * out_features);
    Workspace ws;
    ws.reserve(slopfab::cuda::linear_workspace_bytes(qw, rows, slopfab::cuda::ComputeType::kF32) +
               256);
    runner.forward_f32(qw, fx.get(), rows, fy.get(), ws);
    SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CLOSE(cpu_matmul_nt(x, w, rows, out_features, in_features), to_host(fy), 1e-3,
                "linear fp32");
  }
}

SLOPFAB_TEST_CATEGORY(linear_prepared_matches_per_chunk, "synthetic") {
  CublasScope cb;
  const int rows = 33;
  const int chunk = 13;  // 13 + 13 + 7: three chunks, the last one ragged
  const int in_features = 256;
  const int out_features = 96;

  const std::vector<float> x = bf16_round(make_data(size_t(rows) * in_features, 271u, 0.1f));
  BfBuf dx(x);
  BfBuf dref(size_t(rows) * out_features);
  BfBuf dgot(size_t(rows) * out_features);

  // fp8, so `prepare` has real dequantisation to do — a bf16 weight would make
  // the test vacuous by returning the checkpoint pointer both ways.
  std::vector<uint8_t> raw(size_t(out_features) * in_features);
  uint32_t s = 99991u;
  for (size_t i = 0; i < raw.size(); ++i) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    uint8_t b = uint8_t(s & 0xFFu);
    if ((b & 0x7F) == 0x7F) b &= 0x7Eu;  // avoid the NaN encodings
    raw[i] = b;
  }
  DeviceBuffer<uint8_t> draw(raw.size());
  draw.copy_from_host(raw.data(), raw.size());
  const std::vector<float> sv(1, 8.1264e-3f);
  DeviceBuffer<float> dscale = to_device(sv);
  const std::vector<float> bias = make_data(out_features, 272u, 0.5f);
  DeviceBuffer<float> dbias = to_device(bias);

  slopfab::cuda::QuantWeight qw;
  qw.format = slopfab::cuda::QuantFormat::kF8E4M3;
  qw.data = draw.get();
  qw.out_features = out_features;
  qw.in_features = in_features;
  qw.weight_scale = dscale.get();
  qw.bias = dbias.get();
  qw.bias_format = slopfab::cuda::QuantFormat::kF32;

  slopfab::cuda::LinearRunner runner;
  runner.init(cb.h, nullptr);

  Workspace ws;
  ws.reserve(slopfab::cuda::linear_workspace_bytes(qw, chunk, slopfab::cuda::ComputeType::kBF16) +
             256);

  // Reference: dequantise inside the loop, once per chunk.
  for (int start = 0; start < rows; start += chunk) {
    const int n = std::min(chunk, rows - start);
    runner.forward(qw, dx.p() + size_t(start) * in_features, n,
                   dref.p() + size_t(start) * out_features, ws);
  }
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());

  // Hoisted: dequantise once, then run every chunk against that copy. Sized
  // from the split rather than from `linear_workspace_bytes`, which is the
  // arithmetic `plan_carve` now relies on.
  Workspace hw;
  hw.reserve(slopfab::cuda::linear_dense_weight_bytes(qw) +
             slopfab::cuda::linear_activation_workspace_bytes(qw, chunk,
                                                             slopfab::cuda::ComputeType::kBF16) +
             256);
  {
    Workspace::Scope scope(hw);
    const __nv_bfloat16* dense = runner.prepare(qw, hw);
    CHECK(dense != nullptr);
    for (int start = 0; start < rows; start += chunk) {
      const int n = std::min(chunk, rows - start);
      runner.forward_prepared(qw, dense, dx.p() + size_t(start) * in_features, n,
                              dgot.p() + size_t(start) * out_features, hw);
    }
  }
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());

  const std::vector<uint16_t> want = dref.bits();
  const std::vector<uint16_t> got = dgot.bits();
  size_t differing = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    if (want[i] != got[i]) ++differing;
  }
  CHECK_MSG(differing == 0, "prepared path differs from per-chunk in %zu of %zu bf16 words",
            differing, want.size());

  // A null dense weight means "this layer takes the native nvfp4 GEMM". This
  // fp8 weight does not, so passing null is a caller bug and must throw rather
  // than reach the fp4 GEMM and return plausible nonsense.
  bool threw = false;
  try {
    runner.forward_prepared(qw, nullptr, dx.p(), chunk, dgot.p(), hw);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK_MSG(threw, "forward_prepared accepted a null dense weight for a non-nvfp4 layer");
}

SLOPFAB_TEST_CATEGORY(linear_int8_convrot, "synthetic") {
  CublasScope cb;
  const int rows = 21;
  const int in_features = 512;  // two 256-wide groups
  const int out_features = 64;
  const int group = 256;

  std::vector<int8_t> w_rot(size_t(out_features) * in_features);
  uint32_t s = 999u;
  for (size_t i = 0; i < w_rot.size(); ++i) {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    w_rot[i] = int8_t(int(s & 0xFFu) - 128);
  }
  std::vector<float> wscale(out_features);
  for (int o = 0; o < out_features; ++o) wscale[o] = 7.675675e-4f * (1.0f + 0.01f * float(o));

  const std::vector<float> x = bf16_round(make_data(size_t(rows) * in_features, 151u, 0.1f));

  // Reference: rotate x on the host with the explicit H, dequantise the weight
  // per output channel, then a plain matmul.
  std::vector<float> H = kron_power(&kH4[0][0], 4, 4);
  for (float& v : H) v /= 16.0f;
  std::vector<float> xr(x.size());
  for (int r = 0; r < rows; ++r) {
    for (int g = 0; g < in_features / group; ++g) {
      const size_t base = size_t(r) * in_features + size_t(g) * group;
      for (int i = 0; i < group; ++i) {
        double acc = 0.0;
        for (int j = 0; j < group; ++j) acc += double(H[size_t(i) * group + j]) * x[base + j];
        xr[base + i] = float(acc);
      }
    }
  }
  std::vector<float> wdq(w_rot.size());
  for (int o = 0; o < out_features; ++o) {
    for (int i = 0; i < in_features; ++i) {
      wdq[size_t(o) * in_features + i] = float(w_rot[size_t(o) * in_features + i]) * wscale[o];
    }
  }
  const std::vector<float> want = cpu_matmul_nt(xr, wdq, rows, out_features, in_features);
  // Skipping the online rotation computes this instead:
  const std::vector<float> unrotated = cpu_matmul_nt(x, wdq, rows, out_features, in_features);

  DeviceBuffer<int8_t> dw(w_rot.size());
  dw.copy_from_host(w_rot.data(), w_rot.size());
  DeviceBuffer<float> dscale = to_device(wscale);
  BfBuf dx(x), dy(size_t(rows) * out_features);

  slopfab::cuda::QuantWeight qw;
  qw.format = slopfab::cuda::QuantFormat::kI8;
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
  ws.reserve(slopfab::cuda::linear_workspace_bytes(qw, rows, slopfab::cuda::ComputeType::kBF16) +
             256);
  runner.forward(qw, dx.p(), rows, dy.p(), ws);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = dy.host();

  CHECK_CLOSE_REL(want, got, 2e-3, 2e-2, "int8 convrot linear");
  CHECK_MSG(max_abs_diff(unrotated, got) > 0.1,
            "convrot linear must rotate the activation online (max diff %.4g)",
            max_abs_diff(unrotated, got));

  // in_features not divisible by the group means the quantiser skipped the
  // rotation, so the runner must skip it too.
  const int odd_in = 300;
  std::vector<int8_t> w_odd(size_t(out_features) * odd_in, 3);
  DeviceBuffer<int8_t> dwo(w_odd.size());
  dwo.copy_from_host(w_odd.data(), w_odd.size());
  const std::vector<float> xo = bf16_round(make_data(size_t(rows) * odd_in, 152u, 0.1f));
  BfBuf dxo(xo), dyo(size_t(rows) * out_features);
  slopfab::cuda::QuantWeight qo = qw;
  qo.data = dwo.get();
  qo.in_features = odd_in;
  Workspace ws2;
  ws2.reserve(slopfab::cuda::linear_workspace_bytes(qo, rows, slopfab::cuda::ComputeType::kBF16) +
              256);
  runner.forward(qo, dxo.p(), rows, dyo.p(), ws2);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> wo(w_odd.size());
  for (int o = 0; o < out_features; ++o) {
    for (int i = 0; i < odd_in; ++i) wo[size_t(o) * odd_in + i] = 3.0f * wscale[o];
  }
  CHECK_CLOSE_REL(cpu_matmul_nt(xo, wo, rows, out_features, odd_in), dyo.host(), 1e-3, 1e-2,
                  "convrot skipped when in_features % group != 0");
}

SLOPFAB_TEST_CATEGORY(nn_dequant_nf4_double_quant, "synthetic") {
  // 258 weight-scale blocks cross a 256-scale nested boundary.
  const Nf4Weight w = make_nf4(129, 128);
  DeviceBuffer<uint8_t> dp(w.packed.size()), da(w.absmax.size());
  dp.copy_from_host(w.packed.data(), w.packed.size());
  da.copy_from_host(w.absmax.data(), w.absmax.size());
  auto dm = to_device(w.map), dnm = to_device(w.nested_map), dna = to_device(w.nested_absmax);
  BfBuf got(size_t(w.out) * w.in);
  slopfab::cuda::launch_dequant_nf4(dp.get(), da.get(), dm.get(), dnm.get(), dna.get(), 64, 256,
                                   w.offset, got.p(), w.out, w.in, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(nf4_reference(w), got.host(), 0.0, "double-quant NF4 dequantisation");
  CHECK_MSG(max_abs_diff(nf4_reference(w, true), got.host()) > 0.1,
            "NF4 even element must use HIGH nibble (max diff %.4g)",
            max_abs_diff(nf4_reference(w, true), got.host()));
}

SLOPFAB_TEST_CATEGORY(nn_dequant_nf4_vector_matches_scalar, "synthetic") {
  const Nf4Weight w = make_nf4(129, 128);
  const size_t n = size_t(w.out) * w.in;
  CHECK(n % 8 == 0);
  DeviceBuffer<uint8_t> dp(w.packed.size()), da(w.absmax.size());
  dp.copy_from_host(w.packed.data(), w.packed.size());
  da.copy_from_host(w.absmax.data(), w.absmax.size());
  auto dm = to_device(w.map), dnm = to_device(w.nested_map), dna = to_device(w.nested_absmax);

  DeviceBuffer<uint16_t> fast(n), slow(n + 8);
  slopfab::cuda::launch_dequant_nf4(dp.get(), da.get(), dm.get(), dnm.get(), dna.get(), 64, 256,
                                   w.offset, reinterpret_cast<__nv_bfloat16*>(fast.get()), w.out,
                                   w.in, nullptr);
  slopfab::cuda::launch_dequant_nf4(dp.get(), da.get(), dm.get(), dnm.get(), dna.get(), 64, 256,
                                   w.offset, reinterpret_cast<__nv_bfloat16*>(slow.get() + 1),
                                   w.out, w.in, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> hf(n), hs(n + 8);
  fast.copy_to_host(hf.data(), n);
  slow.copy_to_host(hs.data(), n + 8);
  size_t bad = 0;
  for (size_t i = 0; i < n; ++i) bad += hf[i] != hs[i + 1];
  CHECK_MSG(bad == 0, "NF4 bf16: eight-wide and two-wide kernels differ in %zu of %zu bf16", bad,
            n);

  // The f16 pair, which had no coverage at all before: same structure, and the
  // packed store has to round exactly as `__float2half_rn` did.
  DeviceBuffer<uint16_t> ffast(n), fslow(n + 8);
  slopfab::cuda::launch_dequant_nf4_f16(dp.get(), da.get(), dm.get(), dnm.get(), dna.get(), 64, 256,
                                       w.offset, reinterpret_cast<__half*>(ffast.get()), n,
                                       nullptr);
  slopfab::cuda::launch_dequant_nf4_f16(dp.get(), da.get(), dm.get(), dnm.get(), dna.get(), 64, 256,
                                       w.offset, reinterpret_cast<__half*>(fslow.get() + 1), n,
                                       nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> gf(n), gs(n + 8);
  ffast.copy_to_host(gf.data(), n);
  fslow.copy_to_host(gs.data(), n + 8);
  bad = 0;
  for (size_t i = 0; i < n; ++i) bad += gf[i] != gs[i + 1];
  CHECK_MSG(bad == 0, "NF4 f16: eight-wide and two-wide kernels differ in %zu of %zu half", bad, n);

  // A ragged element count cannot use the eight-wide kernel at all, and the
  // fallback still has to be right -- this is the path a future shape lands on.
  const Nf4Weight r = make_nf4(3, 44);  // 132 elements, not a multiple of eight
  const size_t rn = size_t(r.out) * r.in;
  CHECK(rn % 8 != 0);
  DeviceBuffer<uint8_t> rp(r.packed.size()), ra(r.absmax.size());
  rp.copy_from_host(r.packed.data(), r.packed.size());
  ra.copy_from_host(r.absmax.data(), r.absmax.size());
  auto rm = to_device(r.map), rnm = to_device(r.nested_map), rna = to_device(r.nested_absmax);
  BfBuf ragged(rn);
  slopfab::cuda::launch_dequant_nf4(rp.get(), ra.get(), rm.get(), rnm.get(), rna.get(), 64, 256,
                                   r.offset, ragged.p(), r.out, r.in, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(nf4_reference(r), ragged.host(), 0.0, "NF4 ragged tail via the scalar kernel");
}

SLOPFAB_TEST_CATEGORY(linear_nf4_double_quant, "synthetic") {
  CublasScope cb;
  const int rows = 7;
  const Nf4Weight w = make_nf4(16, 128);
  DeviceBuffer<uint8_t> dp(w.packed.size()), da(w.absmax.size());
  dp.copy_from_host(w.packed.data(), w.packed.size());
  da.copy_from_host(w.absmax.data(), w.absmax.size());
  auto dm = to_device(w.map), dnm = to_device(w.nested_map), dna = to_device(w.nested_absmax);
  const auto x = bf16_round(make_data(size_t(rows) * w.in, 20260807u, 0.1f));
  BfBuf dx(x), dy(size_t(rows) * w.out);
  slopfab::cuda::QuantWeight qw;
  qw.format = slopfab::cuda::QuantFormat::kNF4; qw.data = dp.get();
  qw.out_features = w.out; qw.in_features = w.in; qw.nf4_absmax = da.get();
  qw.nf4_quant_map = dm.get(); qw.nf4_nested_quant_map = dnm.get();
  qw.nf4_nested_absmax = dna.get(); qw.nf4_nested_offset = w.offset;
  CHECK(qw.stored_bytes() == w.packed.size());
  slopfab::cuda::LinearRunner runner; runner.init(cb.h, nullptr);
  Workspace ws; ws.reserve(slopfab::cuda::linear_workspace_bytes(
      qw, rows, slopfab::cuda::ComputeType::kBF16) + 256);
  runner.forward(qw, dx.p(), rows, dy.p(), ws);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const auto want = cpu_matmul_nt(x, nf4_reference(w), rows, w.out, w.in);
  CHECK_CLOSE_REL(want, dy.host(), 1e-3, 1e-2, "linear NF4 dequantise-then-GEMM");
}
