#include "detail/nn_kernels_fixture.h"

SLOPFAB_TEST_CATEGORY(nn_rmsnorm, "synthetic") {
  const int rows = 19;
  const int dim = 5376;  // the real residual width, and a multiple of 8
  const float eps = 1e-5f;

  const std::vector<float> x = bf16_round(make_data(size_t(rows) * dim, 11u, 3.0f));
  const std::vector<float> w = bf16_round(make_data(dim, 22u, 1.0f));

  BfBuf dx(x), dw(w), dout(x.size());
  slopfab::cuda::launch_rmsnorm(dx.p(), dw.p(), dout.p(), rows, dim, eps, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE_REL(cpu_rmsnorm(x, w, rows, dim, eps), dout.host(), 1e-3, 1e-2, "rmsnorm bf16");

  // The scalar path, on a width that is not a multiple of 8.
  const int odd = 501;
  const std::vector<float> xo = bf16_round(make_data(size_t(rows) * odd, 33u, 3.0f));
  const std::vector<float> wo = bf16_round(make_data(odd, 44u, 1.0f));
  BfBuf dxo(xo), dwo(wo), douto(xo.size());
  slopfab::cuda::launch_rmsnorm(dxo.p(), dwo.p(), douto.p(), rows, odd, eps, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE_REL(cpu_rmsnorm(xo, wo, rows, odd, eps), douto.host(), 1e-3, 1e-2,
                  "rmsnorm bf16 scalar width");

  // fp32 variant.
  DeviceBuffer<float> fx = to_device(x);
  DeviceBuffer<float> fw = to_device(w);
  DeviceBuffer<float> fout(x.size());
  slopfab::cuda::launch_rmsnorm_f32(fx.get(), fw.get(), fout.get(), rows, dim, eps, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(cpu_rmsnorm(x, w, rows, dim, eps), to_host(fout), 1e-4, "rmsnorm fp32");

  // eps must be inside the sqrt and added to the mean of squares. With a tiny
  // row it dominates, so the two placements differ by orders of magnitude.
  const int tiny_dim = 8;
  const std::vector<float> tiny(size_t(1) * tiny_dim, 1e-4f);
  const std::vector<float> ones(tiny_dim, 1.0f);
  BfBuf dt(tiny), dones(ones), dtout(tiny.size());
  slopfab::cuda::launch_rmsnorm(dt.p(), dones.p(), dtout.p(), 1, tiny_dim, eps, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = dtout.host();
  const double inside = 1e-4 / std::sqrt(1e-8 + 1e-5);
  const double outside = 1e-4 / (std::sqrt(1e-8) + 1e-5);
  // The two placements differ by a factor of 29, so a loose bound still
  // separates them; the bound has to clear one bf16 ulp of the output (0.4 %).
  CHECK_MSG(std::fabs(got[0] - inside) < 2e-2 * std::fabs(inside),
            "rmsnorm eps inside sqrt: got %.6g, inside %.6g, outside would be %.6g", got[0],
            inside, outside);
}

SLOPFAB_TEST_CATEGORY(nn_rmsnorm_modulate, "synthetic") {
  const int rows = 23;
  const int dim = 512;
  const int mod_rows = 3;  // video / text / audio
  const float eps = 1e-5f;

  const std::vector<float> x = bf16_round(make_data(size_t(rows) * dim, 51u, 2.0f));
  const std::vector<float> w = bf16_round(make_data(dim, 52u, 1.0f));
  const std::vector<float> scale = make_data(size_t(mod_rows) * dim, 53u, 0.8f);
  const std::vector<float> shift = make_data(size_t(mod_rows) * dim, 54u, 0.5f);
  std::vector<int32_t> a(rows);
  for (int r = 0; r < rows; ++r) a[r] = r % mod_rows;

  BfBuf dx(x), dw(w), dout(x.size());
  DeviceBuffer<float> dscale = to_device(scale);
  DeviceBuffer<float> dshift = to_device(shift);
  DeviceBuffer<int32_t> da = to_device_i32(a);

  slopfab::cuda::launch_rmsnorm_modulate(dx.p(), dw.p(), dscale.get(), dshift.get(), da.get(),
                                        dout.p(), rows, dim, eps, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());

  const std::vector<float> want = cpu_modulate(x, w, scale, shift, a, rows, dim, eps, true);
  const std::vector<float> hazard = cpu_modulate(x, w, scale, shift, a, rows, dim, eps, false);
  const std::vector<float> actual = dout.host();

  CHECK_CLOSE_REL(want, actual, 1e-3, 1e-2, "rmsnorm_modulate n*(1+scale)+shift");
  // Re-associating `n*(1+s)+shift` into `n*s + (n+shift)` only changes rounding;
  // the form that changes the *answer* is dropping the `1 +`, and that is what
  // this leg pins.
  CHECK_MSG(max_abs_diff(want, hazard) > 0.1,
            "modulate hazard form must differ from the spec form (max diff %.4g)",
            max_abs_diff(want, hazard));
  CHECK_MSG(max_abs_diff(hazard, actual) > 0.1,
            "kernel must not match the missing-`1 +` form (max diff %.4g)",
            max_abs_diff(hazard, actual));

  // fp32 activation / bf16 norm weight variant, one shared modulation row as
  // the final layer uses.
  std::vector<int32_t> single(rows, 0);
  DeviceBuffer<int32_t> dsingle = to_device_i32(single);
  DeviceBuffer<float> fx = to_device(x);
  DeviceBuffer<float> fout(x.size());
  slopfab::cuda::launch_rmsnorm_modulate_f32(fx.get(), dw.p(), dscale.get(), dshift.get(),
                                            dsingle.get(), fout.get(), rows, dim, eps, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(cpu_modulate(x, w, scale, shift, single, rows, dim, eps, true), to_host(fout), 1e-4,
              "rmsnorm_modulate_f32");
}

SLOPFAB_TEST_CATEGORY(nn_add_gated, "synthetic") {
  const int rows = 30;
  const int dim = 256;
  const int mod_rows = 3;

  const std::vector<float> x = bf16_round(make_data(size_t(rows) * dim, 61u, 1.0f));
  const std::vector<float> branch = bf16_round(make_data(size_t(rows) * dim, 62u, 1.0f));

  // Deliberately very different gate rows: an off-by-one or a row-major/
  // column-major slip in the `a[]` lookup then shows up as an O(1) error rather
  // than as noise.
  std::vector<float> gate(size_t(mod_rows) * dim);
  for (int m = 0; m < mod_rows; ++m) {
    for (int i = 0; i < dim; ++i) gate[size_t(m) * dim + i] = float(m + 1) * (m == 1 ? -1.0f : 1.0f);
  }
  std::vector<int32_t> a(rows);
  for (int r = 0; r < rows; ++r) a[r] = (r * 2 + 1) % mod_rows;

  std::vector<float> want(x.size());
  for (int r = 0; r < rows; ++r) {
    for (int i = 0; i < dim; ++i) {
      const size_t idx = size_t(r) * dim + i;
      // Gate multiplies the branch only; the residual is ungated (spec 9.4.3).
      want[idx] = x[idx] + gate[size_t(a[r]) * dim + i] * branch[idx];
    }
  }
  // The hazard: gating the sum instead of the branch.
  std::vector<float> hazard(x.size());
  for (int r = 0; r < rows; ++r) {
    for (int i = 0; i < dim; ++i) {
      const size_t idx = size_t(r) * dim + i;
      hazard[idx] = gate[size_t(a[r]) * dim + i] * (x[idx] + branch[idx]);
    }
  }

  BfBuf dx(x), dbranch(branch);
  DeviceBuffer<float> dgate = to_device(gate);
  DeviceBuffer<int32_t> da = to_device_i32(a);
  slopfab::cuda::launch_add_gated(dx.p(), dbranch.p(), dgate.get(), da.get(), rows, dim, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> actual = dx.host();

  CHECK_CLOSE_REL(want, actual, 1e-3, 1e-2, "add_gated");
  CHECK_MSG(max_abs_diff(hazard, actual) > 0.1,
            "add_gated must gate the branch, not the sum (max diff %.4g)",
            max_abs_diff(hazard, actual));
}

SLOPFAB_TEST_CATEGORY(nn_swiglu, "synthetic") {
  const int rows = 13;
  const int inner = 1024;

  // Asymmetric by construction: the first half is scaled down and the second
  // up, so swapping the halves changes every element. A symmetric random input
  // passes either way and proves nothing.
  std::vector<float> fused(size_t(rows) * 2 * inner);
  const std::vector<float> noise = make_data(fused.size(), 71u, 1.0f);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < inner; ++c) {
      fused[size_t(r) * 2 * inner + c] = 0.25f * noise[size_t(r) * 2 * inner + c] - 1.0f;
      fused[size_t(r) * 2 * inner + inner + c] =
          4.0f * noise[size_t(r) * 2 * inner + inner + c] + 3.0f;
    }
  }
  fused = bf16_round(fused);

  std::vector<float> want(size_t(rows) * inner);
  std::vector<float> swapped(size_t(rows) * inner);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < inner; ++c) {
      // Gate is the FIRST half — `mlp.fc1` original naming (spec 4.4).
      const float gate = fused[size_t(r) * 2 * inner + c];
      const float value = fused[size_t(r) * 2 * inner + inner + c];
      want[size_t(r) * inner + c] = (gate / (1.0f + std::exp(-gate))) * value;
      swapped[size_t(r) * inner + c] = (value / (1.0f + std::exp(-value))) * gate;
    }
  }

  BfBuf dfused(fused), dout(size_t(rows) * inner);
  slopfab::cuda::launch_swiglu(dfused.p(), dout.p(), rows, inner, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> actual = dout.host();

  CHECK_CLOSE_REL(want, actual, 1e-3, 1e-2, "swiglu, gate = first half");
  CHECK_MSG(max_abs_diff(swapped, actual) > 1.0,
            "swiglu halves must not be swappable (max diff %.4g)", max_abs_diff(swapped, actual));
}

SLOPFAB_TEST_CATEGORY(nn_rope_h3, "synthetic") {
  const int rows = 11;
  const int heads = 3;
  const int head_dim = 128;
  const int half = 48;
  const int rot = 96;

  const std::vector<float> x =
      bf16_round(make_data(size_t(rows) * heads * head_dim, 81u, 2.0f));
  const std::vector<float> cos_t = make_data(size_t(rows) * rot, 82u, 1.0f);
  const std::vector<float> sin_t = make_data(size_t(rows) * rot, 83u, 1.0f);

  // Spec 5.3: pair j with j+48 over the 96 rotary channels; 96..127 untouched.
  std::vector<float> want = x;
  for (int r = 0; r < rows; ++r) {
    for (int h = 0; h < heads; ++h) {
      const size_t base = (size_t(r) * heads + h) * head_dim;
      for (int j = 0; j < half; ++j) {
        const float c = cos_t[size_t(r) * rot + j];
        const float s = sin_t[size_t(r) * rot + j];
        const float lo = x[base + j];
        const float hi = x[base + j + half];
        want[base + j] = lo * c - hi * s;
        want[base + j + half] = hi * c + lo * s;
      }
    }
  }
  // The hazard: NeoX pairing over all 128 channels, j with j+64.
  std::vector<float> hazard = x;
  for (int r = 0; r < rows; ++r) {
    for (int h = 0; h < heads; ++h) {
      const size_t base = (size_t(r) * heads + h) * head_dim;
      for (int j = 0; j < 64; ++j) {
        const float c = cos_t[size_t(r) * rot + (j % rot)];
        const float s = sin_t[size_t(r) * rot + (j % rot)];
        const float lo = x[base + j];
        const float hi = x[base + j + 64];
        hazard[base + j] = lo * c - hi * s;
        hazard[base + j + 64] = hi * c + lo * s;
      }
    }
  }

  BfBuf dx(x);
  DeviceBuffer<float> dcos = to_device(cos_t);
  DeviceBuffer<float> dsin = to_device(sin_t);
  slopfab::cuda::launch_rope_h3(dx.p(), dcos.get(), dsin.get(), rows, heads, head_dim, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> actual = dx.host();

  CHECK_CLOSE_REL(want, actual, 1e-3, 1e-2, "rope_h3 pairs j with j+48");
  CHECK_MSG(max_abs_diff(hazard, actual) > 0.5,
            "rope_h3 must not use the j/j+64 pairing (max diff %.4g)",
            max_abs_diff(hazard, actual));

  // Channels 96..127 pass through bit for bit, not merely within tolerance.
  BfBuf dref(x);
  const std::vector<uint16_t> before = dref.bits();
  const std::vector<uint16_t> after = dx.bits();
  size_t tail_diffs = 0;
  for (int r = 0; r < rows; ++r) {
    for (int h = 0; h < heads; ++h) {
      const size_t base = (size_t(r) * heads + h) * head_dim;
      for (int d = rot; d < head_dim; ++d) {
        if (before[base + d] != after[base + d]) ++tail_diffs;
      }
    }
  }
  CHECK_MSG(tail_diffs == 0, "rope_h3: %zu of the 32 pass-through channels were modified",
            tail_diffs);
}

#if 0  // Removed: canonical H3 tables are host-built and tested in test_packing.cpp.
SLOPFAB_TEST_CATEGORY(nn_rope_tables_h3, "synthetic") {
  const int rows = 9;
  const int freq_dim = 16;
  const float theta = 10000.0f;
  const int half = 3 * freq_dim;
  const int full = 2 * half;

  // Mix small spatial coordinates with the large T values a long prompt plus a
  // 124-frame clip reaches — that is where the fp64->fp32 cast point matters
  // (spec 9.2).
  std::vector<double> pos(size_t(rows) * 3);
  for (int r = 0; r < rows; ++r) {
    pos[size_t(r) * 3 + 0] = 3000.0 + 1.6666666666666667 * r;
    pos[size_t(r) * 3 + 1] = -5.25 + 0.5 * r;
    pos[size_t(r) * 3 + 2] = 31.125 - 0.25 * r;
  }

  DeviceBuffer<float> dcos(size_t(rows) * full);
  DeviceBuffer<float> dsin(size_t(rows) * full);
  slopfab::cuda::build_rope_tables_h3(pos.data(), rows, theta, freq_dim, dcos.get(), dsin.get(),
                                     nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> gc = to_host(dcos);
  const std::vector<float> gs = to_host(dsin);

  std::vector<float> wc(size_t(rows) * full);
  std::vector<float> ws(size_t(rows) * full);
  for (int r = 0; r < rows; ++r) {
    for (int j = 0; j < half; ++j) {
      const int axis = j / freq_dim;
      const int k = j % freq_dim;
      const float inv = float(1.0 / std::pow(double(theta), double(k) / double(freq_dim)));
      // The cast to fp32 happens here, before the multiply — not later.
      const float angle = float(pos[size_t(r) * 3 + axis]) * inv;
      wc[size_t(r) * full + j] = float(std::cos(double(angle)));
      ws[size_t(r) * full + j] = float(std::sin(double(angle)));
      wc[size_t(r) * full + j + half] = wc[size_t(r) * full + j];
      ws[size_t(r) * full + j + half] = ws[size_t(r) * full + j];
    }
  }
  CHECK_CLOSE(wc, gc, 1e-5, "rope cos table");
  CHECK_CLOSE(ws, gs, 1e-5, "rope sin table");

  // The 48-wide half period is duplicated to 96 exactly, which is what makes
  // the j/j+48 pairing read a single cos value.
  size_t dup_diffs = 0;
  for (int r = 0; r < rows; ++r) {
    for (int j = 0; j < half; ++j) {
      if (gc[size_t(r) * full + j] != gc[size_t(r) * full + j + half]) ++dup_diffs;
      if (gs[size_t(r) * full + j] != gs[size_t(r) * full + j + half]) ++dup_diffs;
    }
  }
  CHECK_MSG(dup_diffs == 0, "rope tables: %zu entries are not duplicated across the halves",
            dup_diffs);

  // The 48-wide half period is [T(16) | H(16) | W(16)] in that order. Channel 0
  // carries inv_freq[0] = 1, so cos[0] is cos(pos.t) outright; channel 16 is
  // the first H channel and must track pos.h, not pos.t.
  CHECK_NEAR(gc[0], std::cos(double(float(pos[0]))), 1e-5);
  CHECK_NEAR(gc[freq_dim], std::cos(double(float(pos[1]))), 1e-5);
  CHECK_NEAR(gc[2 * freq_dim], std::cos(double(float(pos[2]))), 1e-5);
}
#endif


SLOPFAB_TEST_CATEGORY(nn_rope_neox, "synthetic") {
  const int rows = 7;
  const int heads = 2;
  const int head_dim = 64;
  const int half = head_dim / 2;

  const std::vector<float> x = bf16_round(make_data(size_t(rows) * heads * head_dim, 91u, 2.0f));
  const std::vector<float> cos_t = make_data(size_t(rows) * head_dim, 92u, 1.0f);
  const std::vector<float> sin_t = make_data(size_t(rows) * head_dim, 93u, 1.0f);

  std::vector<float> want = x;
  for (int r = 0; r < rows; ++r) {
    for (int h = 0; h < heads; ++h) {
      const size_t base = (size_t(r) * heads + h) * head_dim;
      for (int j = 0; j < half; ++j) {
        const float lo = x[base + j];
        const float hi = x[base + j + half];
        want[base + j] = lo * cos_t[size_t(r) * head_dim + j] - hi * sin_t[size_t(r) * head_dim + j];
        want[base + j + half] = hi * cos_t[size_t(r) * head_dim + j + half] +
                                lo * sin_t[size_t(r) * head_dim + j + half];
      }
    }
  }

  BfBuf dx(x);
  DeviceBuffer<float> dcos = to_device(cos_t);
  DeviceBuffer<float> dsin = to_device(sin_t);
  slopfab::cuda::launch_rope_neox(dx.p(), dcos.get(), dsin.get(), rows, heads, head_dim, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE_REL(want, dx.host(), 1e-3, 1e-2, "rope_neox");
}

SLOPFAB_TEST_CATEGORY(nn_head_rmsnorm, "synthetic") {
  const int rows = 17;
  const int heads = 5;
  const int dim = 128;
  const float eps = 1e-5f;

  const std::vector<float> x = bf16_round(make_data(size_t(rows) * heads * dim, 101u, 2.0f));
  const std::vector<float> w = bf16_round(make_data(dim, 102u, 1.0f));

  // Normalises over the 128-wide head dim, not over heads*dim (spec 9.3).
  const std::vector<float> want = cpu_rmsnorm(x, w, rows * heads, dim, eps);
  // The hazard: normalising over the whole 640-wide concatenation.
  const std::vector<float> hazard = cpu_rmsnorm(x, std::vector<float>(size_t(heads) * dim, 1.0f),
                                                rows, heads * dim, eps);

  BfBuf dx(x), dw(w);
  slopfab::cuda::launch_head_rmsnorm(dx.p(), dw.p(), rows, heads, dim, eps, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> actual = dx.host();
  CHECK_CLOSE_REL(want, actual, 1e-3, 1e-2, "head_rmsnorm over head_dim");
  CHECK_MSG(max_abs_diff(hazard, actual) > 0.1,
            "head_rmsnorm must reduce over head_dim only (max diff %.4g)",
            max_abs_diff(hazard, actual));
}

SLOPFAB_TEST_CATEGORY(nn_gather_scatter, "synthetic") {
  const int n = 40;
  const int dim = 96;
  const std::vector<float> src = bf16_round(make_data(size_t(n) * dim, 111u, 1.0f));
  std::vector<int32_t> index(n);
  for (int i = 0; i < n; ++i) index[i] = (n - 1) - i;  // reversal, so identity cannot pass

  std::vector<float> want(src.size());
  for (int i = 0; i < n; ++i) {
    for (int d = 0; d < dim; ++d) want[size_t(i) * dim + d] = src[size_t(index[i]) * dim + d];
  }

  BfBuf dsrc(src), ddst(src.size());
  DeviceBuffer<int32_t> didx = to_device_i32(index);
  slopfab::cuda::launch_gather_rows(dsrc.p(), didx.get(), ddst.p(), n, dim, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want, ddst.host(), 0.0, "gather_rows");

  // scatter is the inverse of gather for a permutation index.
  BfBuf dback(src.size());
  slopfab::cuda::launch_scatter_rows(ddst.p(), didx.get(), dback.p(), n, dim, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(src, dback.host(), 0.0, "scatter_rows undoes gather_rows");

  DeviceBuffer<float> fsrc = to_device(src);
  DeviceBuffer<float> fdst(src.size());
  slopfab::cuda::launch_gather_rows_f32(fsrc.get(), didx.get(), fdst.get(), n, dim, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want, to_host(fdst), 0.0, "gather_rows_f32");
}

SLOPFAB_TEST_CATEGORY(nn_elementwise, "synthetic") {
  const size_t n = 4097;
  const std::vector<float> a = make_data(n, 121u, 2.0f);
  const std::vector<float> b = make_data(n, 122u, 2.0f);

  std::vector<float> want_add(n);
  std::vector<float> want_axpby(n);
  std::vector<float> want_silu(n);
  for (size_t i = 0; i < n; ++i) {
    want_add[i] = a[i] + b[i];
    want_axpby[i] = 1.5f * a[i] + (-0.25f) * b[i];
    want_silu[i] = a[i] / (1.0f + std::exp(-a[i]));
  }

  DeviceBuffer<float> da = to_device(a);
  DeviceBuffer<float> db = to_device(b);
  DeviceBuffer<float> dout(n);
  slopfab::cuda::launch_add(da.get(), db.get(), dout.get(), n, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want_add, to_host(dout), 1e-6, "add");

  slopfab::cuda::launch_axpby(da.get(), 1.5f, db.get(), -0.25f, dout.get(), n, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want_axpby, to_host(dout), 1e-6, "axpby");

  slopfab::cuda::launch_silu(da.get(), dout.get(), n, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want_silu, to_host(dout), 1e-5, "silu");
}

SLOPFAB_TEST_CATEGORY(nn_sub_bf16_aliasing_and_tails, "synthetic") {
  // Deliberately not a multiple of 8, so the vectorised bulk and the scalar
  // tail split inside the launcher are both exercised. That split is the only
  // branch in the function with no other coverage.
  const size_t n = 4099;
  const std::vector<float> a = bf16_round(make_data(n, 191u, 2.0f));
  const std::vector<float> b = bf16_round(make_data(n, 192u, 2.0f));

  // Rounded to bf16, because that is what the kernel stores. `a` and `b` are
  // already bf16 values, so their difference is exact in the fp32 the kernel
  // subtracts in, and the only error is the single rounding on the way out —
  // which makes this an exact expectation rather than an approximate one. A
  // plain fp32 difference here would be off by up to one bf16 ulp (7.8e-3 at
  // magnitude 2.5) and would need a tolerance loose enough to hide real bugs.
  std::vector<float> want(n);
  for (size_t i = 0; i < n; ++i) want[i] = a[i] - b[i];
  want = bf16_round(want);

  // Out of place first, to establish what the answer is.
  BfBuf da(a), db(b), dout(n);
  slopfab::cuda::launch_sub_bf16(da.p(), db.p(), dout.p(), n, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> disjoint = dout.host();
  CHECK_CLOSE(want, disjoint, 1e-6, "sub_bf16 out of place");

  // In place over `b`, which is what the block cache does. Must agree with the
  // out-of-place result exactly — not approximately: the same arithmetic on the
  // same inputs, so any difference is an aliasing bug and not rounding.
  BfBuf da2(a), db2(b);
  slopfab::cuda::launch_sub_bf16(da2.p(), db2.p(), db2.p(), n, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(db2.bits() == dout.bits());

  // `a` must survive the in-place call: it is the live residual stream and the
  // rest of the stack runs on it.
  CHECK(da2.bits() == da.bits());

  // The misaligned fallback. Offsetting all three pointers by one bf16 element
  // puts them 2 bytes off a 16-byte boundary, which would fault on the `uint4`
  // path; the launcher must notice and take the scalar kernel instead.
  const size_t m = n - 1;
  std::vector<float> want_off(m);
  for (size_t i = 0; i < m; ++i) want_off[i] = a[i + 1] - b[i + 1];
  want_off = bf16_round(want_off);
  BfBuf da3(a), db3(b), dout3(n);
  slopfab::cuda::launch_sub_bf16(da3.p() + 1, db3.p() + 1, dout3.p() + 1, m, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  // `host()` returns by value, so it must be held in a named local: taking
  // `.begin()` from one call and `.end()` from another walks between two
  // different temporaries.
  const std::vector<float> h3 = dout3.host();
  const std::vector<float> got_off(h3.begin() + 1, h3.end());
  CHECK_CLOSE(want_off, got_off, 1e-6, "sub_bf16 misaligned fallback");

  // A zero count must be a no-op rather than a launch with a zero grid.
  slopfab::cuda::launch_sub_bf16(da.p(), db.p(), dout.p(), 0, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(dout.bits() == BfBuf(disjoint).bits());
}

SLOPFAB_TEST_CATEGORY(nn_pre_quant_scale, "synthetic") {
  const int rows = 17;
  const int dim = 96;
  const std::vector<float> x = bf16_round(make_data(size_t(rows) * dim, 771u, 1.0f));
  const std::vector<float> scale = bf16_round(make_data(dim, 772u, 0.5f));

  BfBuf dx(x), dscale(scale), dy(x.size());
  slopfab::cuda::launch_pre_quant_scale(dx.p(), dscale.p(), dy.p(), rows, dim, nullptr);
  SLOPFAB_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> want(x.size());
  for (int r = 0; r < rows; ++r) {
    for (int i = 0; i < dim; ++i) {
      want[size_t(r) * dim + i] =
          slopfab::bf16_to_f32(slopfab::f32_to_bf16(x[size_t(r) * dim + i] * scale[i]));
    }
  }
  CHECK_CLOSE(want, dy.host(), 0.0, "pre_quant_scale");

  // Scaling by row instead of by column is the plausible transposition.
  std::vector<float> by_row(x.size());
  for (int r = 0; r < rows; ++r) {
    for (int i = 0; i < dim; ++i) {
      by_row[size_t(r) * dim + i] = x[size_t(r) * dim + i] * scale[r % dim];
    }
  }
  CHECK_MSG(max_abs_diff(by_row, dy.host()) > 1e-4,
            "pre_quant_scale must scale per input channel, not per row (max diff %.4g)",
            max_abs_diff(by_row, dy.host()));
}
