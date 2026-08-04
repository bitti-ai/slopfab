// Tests for the shared transformer kernel layer.
//
// Every reference here is written from docs/transformer_spec.md and
// docs/convrot_notes.md, never from the kernel it checks. The bugs this file
// exists to catch do not crash and do not produce NaNs — a RoPE pairing of
// j with j+64 instead of j+48, a SwiGLU with the halves swapped, a Sylvester
// Hadamard instead of a regular one, a gate applied to the sum instead of the
// branch. All four give finite, well-scaled, wrong output.
//
// Where a hazard has a plausible wrong form, the test computes that form too
// and asserts the kernel does *not* match it. A test that only checks the right
// answer cannot tell you whether it had any power to detect the wrong one.

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "harness.h"
#include "vidfab/cuda/attention.cuh"
#include "vidfab/cuda/device.h"
#include "vidfab/cuda/gemm.cuh"
#include "vidfab/cuda/linear.cuh"
#include "vidfab/cuda/nn_kernels.cuh"
#include "vidfab/cuda/workspace.cuh"
#include "vidfab/dtype.h"

namespace {

using vidfab::cuda::DeviceBuffer;
using vidfab::cuda::Workspace;
using vidfab::test::make_data;

// --- host/device plumbing ---------------------------------------------------

// bf16 is carried around as raw uint16 so the host side uses the already-tested
// `vidfab::f32_to_bf16`, which rounds to nearest even exactly as the device
// intrinsic does.
struct BfBuf {
  DeviceBuffer<uint16_t> raw;

  explicit BfBuf(size_t n) : raw(n) {}
  explicit BfBuf(const std::vector<float>& host) : raw(host.size()) {
    std::vector<uint16_t> bits(host.size());
    for (size_t i = 0; i < host.size(); ++i) bits[i] = vidfab::f32_to_bf16(host[i]);
    raw.copy_from_host(bits.data(), bits.size());
  }

  __nv_bfloat16* p() { return reinterpret_cast<__nv_bfloat16*>(raw.get()); }

  std::vector<uint16_t> bits() const {
    std::vector<uint16_t> h(raw.size());
    raw.copy_to_host(h.data(), h.size());
    return h;
  }

  std::vector<float> host() const {
    const std::vector<uint16_t> b = bits();
    std::vector<float> out(b.size());
    for (size_t i = 0; i < b.size(); ++i) out[i] = vidfab::bf16_to_f32(b[i]);
    return out;
  }
};

// The reference must see the same inputs the kernel does, so anything destined
// for a bf16 buffer is rounded on the host first.
std::vector<float> bf16_round(const std::vector<float>& v) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = vidfab::bf16_to_f32(vidfab::f32_to_bf16(v[i]));
  return out;
}

DeviceBuffer<float> to_device(const std::vector<float>& host) {
  DeviceBuffer<float> d(host.size());
  d.copy_from_host(host.data(), host.size());
  return d;
}

std::vector<float> to_host(const DeviceBuffer<float>& d) {
  std::vector<float> h(d.size());
  d.copy_to_host(h.data(), h.size());
  return h;
}

DeviceBuffer<int32_t> to_device_i32(const std::vector<int32_t>& host) {
  DeviceBuffer<int32_t> d(host.size());
  d.copy_from_host(host.data(), host.size());
  return d;
}

// Largest elementwise absolute difference; used where the point of the check is
// that two candidate references are far apart.
double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(double(a[i]) - b[i]));
  return worst;
}

// --- CPU references ---------------------------------------------------------

// x * rsqrt(mean(x^2) + eps) * w. eps inside the sqrt, added to the mean of
// squares; no mean subtraction, no bias (spec 9.3).
std::vector<float> cpu_rmsnorm(const std::vector<float>& x, const std::vector<float>& w, int rows,
                               int dim, float eps) {
  std::vector<float> out(x.size());
  for (int r = 0; r < rows; ++r) {
    double sum_sq = 0.0;
    for (int i = 0; i < dim; ++i) {
      const double v = x[size_t(r) * dim + i];
      sum_sq += v * v;
    }
    const double inv = 1.0 / std::sqrt(sum_sq / dim + eps);
    for (int i = 0; i < dim; ++i) {
      out[size_t(r) * dim + i] = float(x[size_t(r) * dim + i] * inv * w[i]);
    }
  }
  return out;
}

// `one_plus` false is the hazard form: the modulation applied without the
// `1 +`, which the spec spells out at 3.3 and 9.4.1.
std::vector<float> cpu_modulate(const std::vector<float>& x, const std::vector<float>& w,
                                const std::vector<float>& scale, const std::vector<float>& shift,
                                const std::vector<int32_t>& a, int rows, int dim, float eps,
                                bool one_plus) {
  const std::vector<float> n = cpu_rmsnorm(x, w, rows, dim, eps);
  std::vector<float> out(x.size());
  for (int r = 0; r < rows; ++r) {
    const size_t mod = size_t(a[r]) * dim;
    for (int i = 0; i < dim; ++i) {
      const float s = scale[mod + i];
      out[size_t(r) * dim + i] =
          n[size_t(r) * dim + i] * (one_plus ? (1.0f + s) : s) + shift[mod + i];
    }
  }
  return out;
}

// Dense attention, no mask, no causality (spec 2.2). O(S^2) but the test sizes
// are chosen so this runs in well under a second.
std::vector<float> cpu_attention(const std::vector<float>& q, const std::vector<float>& k,
                                 const std::vector<float>& v, int seq, int heads, int kv_heads,
                                 int head_dim, float scale) {
  const int qld = heads * head_dim;
  const int kvld = kv_heads * head_dim;
  const int group = heads / kv_heads;
  std::vector<float> out(size_t(seq) * qld, 0.0f);
  std::vector<double> p(seq);

  for (int h = 0; h < heads; ++h) {
    const int kv = h / group;
    for (int i = 0; i < seq; ++i) {
      double m = -1e300;
      for (int j = 0; j < seq; ++j) {
        double dot = 0.0;
        for (int d = 0; d < head_dim; ++d) {
          dot += double(q[size_t(i) * qld + h * head_dim + d]) *
                 k[size_t(j) * kvld + kv * head_dim + d];
        }
        p[j] = dot * scale;
        m = std::max(m, p[j]);
      }
      double sum = 0.0;
      for (int j = 0; j < seq; ++j) {
        p[j] = std::exp(p[j] - m);
        sum += p[j];
      }
      for (int d = 0; d < head_dim; ++d) {
        double acc = 0.0;
        for (int j = 0; j < seq; ++j) {
          acc += p[j] * v[size_t(j) * kvld + kv * head_dim + d];
        }
        out[size_t(i) * qld + h * head_dim + d] = float(acc / sum);
      }
    }
  }
  return out;
}

std::vector<float> cpu_matmul_nt(const std::vector<float>& A, const std::vector<float>& B, int M,
                                 int N, int K) {
  std::vector<float> C(size_t(M) * N);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      double acc = 0.0;
      for (int k = 0; k < K; ++k) acc += double(A[size_t(m) * K + k]) * B[size_t(n) * K + k];
      C[size_t(m) * N + n] = float(acc);
    }
  }
  return C;
}

// --- the regular Hadamard, built the long way -------------------------------
//
// docs/convrot_notes.md: h4 is the *regular* Hadamard matrix (symmetric,
// constant row sums 2), not the Sylvester one. H = kron^4(h4)/16.
const int kH4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
const int kSylvester2[2][2] = {{1, 1}, {1, -1}};

// Kronecker power of a small matrix, built explicitly so the test shares no
// code at all with the butterfly it checks.
std::vector<float> kron_power(const int* base, int base_n, int times) {
  std::vector<float> m(size_t(base_n) * base_n);
  for (int i = 0; i < base_n * base_n; ++i) m[i] = float(base[i]);
  int n = base_n;
  for (int t = 1; t < times; ++t) {
    const int next = n * base_n;
    std::vector<float> out(size_t(next) * next);
    for (int i = 0; i < base_n; ++i) {
      for (int j = 0; j < base_n; ++j) {
        for (int r = 0; r < n; ++r) {
          for (int c = 0; c < n; ++c) {
            out[size_t(i * n + r) * next + (j * n + c)] = float(base[i * base_n + j]) *
                                                          m[size_t(r) * n + c];
          }
        }
      }
    }
    m.swap(out);
    n = next;
  }
  return m;
}

// --- tests ------------------------------------------------------------------

VIDFAB_TEST(nn_rmsnorm) {
  const int rows = 19;
  const int dim = 5376;  // the real residual width, and a multiple of 8
  const float eps = 1e-5f;

  const std::vector<float> x = bf16_round(make_data(size_t(rows) * dim, 11u, 3.0f));
  const std::vector<float> w = bf16_round(make_data(dim, 22u, 1.0f));

  BfBuf dx(x), dw(w), dout(x.size());
  vidfab::cuda::launch_rmsnorm(dx.p(), dw.p(), dout.p(), rows, dim, eps, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE_REL(cpu_rmsnorm(x, w, rows, dim, eps), dout.host(), 1e-3, 1e-2, "rmsnorm bf16");

  // The scalar path, on a width that is not a multiple of 8.
  const int odd = 501;
  const std::vector<float> xo = bf16_round(make_data(size_t(rows) * odd, 33u, 3.0f));
  const std::vector<float> wo = bf16_round(make_data(odd, 44u, 1.0f));
  BfBuf dxo(xo), dwo(wo), douto(xo.size());
  vidfab::cuda::launch_rmsnorm(dxo.p(), dwo.p(), douto.p(), rows, odd, eps, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE_REL(cpu_rmsnorm(xo, wo, rows, odd, eps), douto.host(), 1e-3, 1e-2,
                  "rmsnorm bf16 scalar width");

  // fp32 variant.
  DeviceBuffer<float> fx = to_device(x);
  DeviceBuffer<float> fw = to_device(w);
  DeviceBuffer<float> fout(x.size());
  vidfab::cuda::launch_rmsnorm_f32(fx.get(), fw.get(), fout.get(), rows, dim, eps, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(cpu_rmsnorm(x, w, rows, dim, eps), to_host(fout), 1e-4, "rmsnorm fp32");

  // eps must be inside the sqrt and added to the mean of squares. With a tiny
  // row it dominates, so the two placements differ by orders of magnitude.
  const int tiny_dim = 8;
  const std::vector<float> tiny(size_t(1) * tiny_dim, 1e-4f);
  const std::vector<float> ones(tiny_dim, 1.0f);
  BfBuf dt(tiny), dones(ones), dtout(tiny.size());
  vidfab::cuda::launch_rmsnorm(dt.p(), dones.p(), dtout.p(), 1, tiny_dim, eps, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = dtout.host();
  const double inside = 1e-4 / std::sqrt(1e-8 + 1e-5);
  const double outside = 1e-4 / (std::sqrt(1e-8) + 1e-5);
  // The two placements differ by a factor of 29, so a loose bound still
  // separates them; the bound has to clear one bf16 ulp of the output (0.4 %).
  CHECK_MSG(std::fabs(got[0] - inside) < 2e-2 * std::fabs(inside),
            "rmsnorm eps inside sqrt: got %.6g, inside %.6g, outside would be %.6g", got[0],
            inside, outside);
}

VIDFAB_TEST(nn_rmsnorm_modulate) {
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

  vidfab::cuda::launch_rmsnorm_modulate(dx.p(), dw.p(), dscale.get(), dshift.get(), da.get(),
                                        dout.p(), rows, dim, eps, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());

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
  vidfab::cuda::launch_rmsnorm_modulate_f32(fx.get(), dw.p(), dscale.get(), dshift.get(),
                                            dsingle.get(), fout.get(), rows, dim, eps, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(cpu_modulate(x, w, scale, shift, single, rows, dim, eps, true), to_host(fout), 1e-4,
              "rmsnorm_modulate_f32");
}

VIDFAB_TEST(nn_add_gated) {
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
  vidfab::cuda::launch_add_gated(dx.p(), dbranch.p(), dgate.get(), da.get(), rows, dim, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> actual = dx.host();

  CHECK_CLOSE_REL(want, actual, 1e-3, 1e-2, "add_gated");
  CHECK_MSG(max_abs_diff(hazard, actual) > 0.1,
            "add_gated must gate the branch, not the sum (max diff %.4g)",
            max_abs_diff(hazard, actual));
}

VIDFAB_TEST(nn_swiglu) {
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
  vidfab::cuda::launch_swiglu(dfused.p(), dout.p(), rows, inner, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> actual = dout.host();

  CHECK_CLOSE_REL(want, actual, 1e-3, 1e-2, "swiglu, gate = first half");
  CHECK_MSG(max_abs_diff(swapped, actual) > 1.0,
            "swiglu halves must not be swappable (max diff %.4g)", max_abs_diff(swapped, actual));
}

VIDFAB_TEST(nn_rope_h3) {
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
  vidfab::cuda::launch_rope_h3(dx.p(), dcos.get(), dsin.get(), rows, heads, head_dim, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
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

VIDFAB_TEST(nn_rope_tables_h3) {
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
  vidfab::cuda::build_rope_tables_h3(pos.data(), rows, theta, freq_dim, dcos.get(), dsin.get(),
                                     nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
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

VIDFAB_TEST(nn_rope_neox) {
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
  vidfab::cuda::launch_rope_neox(dx.p(), dcos.get(), dsin.get(), rows, heads, head_dim, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE_REL(want, dx.host(), 1e-3, 1e-2, "rope_neox");
}

VIDFAB_TEST(nn_head_rmsnorm) {
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
  vidfab::cuda::launch_head_rmsnorm(dx.p(), dw.p(), rows, heads, dim, eps, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> actual = dx.host();
  CHECK_CLOSE_REL(want, actual, 1e-3, 1e-2, "head_rmsnorm over head_dim");
  CHECK_MSG(max_abs_diff(hazard, actual) > 0.1,
            "head_rmsnorm must reduce over head_dim only (max diff %.4g)",
            max_abs_diff(hazard, actual));
}

VIDFAB_TEST(nn_gather_scatter) {
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
  vidfab::cuda::launch_gather_rows(dsrc.p(), didx.get(), ddst.p(), n, dim, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want, ddst.host(), 0.0, "gather_rows");

  // scatter is the inverse of gather for a permutation index.
  BfBuf dback(src.size());
  vidfab::cuda::launch_scatter_rows(ddst.p(), didx.get(), dback.p(), n, dim, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(src, dback.host(), 0.0, "scatter_rows undoes gather_rows");

  DeviceBuffer<float> fsrc = to_device(src);
  DeviceBuffer<float> fdst(src.size());
  vidfab::cuda::launch_gather_rows_f32(fsrc.get(), didx.get(), fdst.get(), n, dim, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want, to_host(fdst), 0.0, "gather_rows_f32");
}

VIDFAB_TEST(nn_elementwise) {
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
  vidfab::cuda::launch_add(da.get(), db.get(), dout.get(), n, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want_add, to_host(dout), 1e-6, "add");

  vidfab::cuda::launch_axpby(da.get(), 1.5f, db.get(), -0.25f, dout.get(), n, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want_axpby, to_host(dout), 1e-6, "axpby");

  vidfab::cuda::launch_silu(da.get(), dout.get(), n, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(want_silu, to_host(dout), 1e-5, "silu");
}

// Every e4m3 bit pattern must dequantise to exactly what dtype.h's host
// reference produces. The two implementations are written out separately, so
// this is the check that keeps them in step.
VIDFAB_TEST(nn_dequant_f8e4m3_all_patterns) {
  std::vector<uint8_t> patterns(256);
  for (int i = 0; i < 256; ++i) patterns[i] = uint8_t(i);

  DeviceBuffer<uint8_t> dsrc(256);
  dsrc.copy_from_host(patterns.data(), 256);
  const std::vector<float> one(1, 1.0f);
  DeviceBuffer<float> dscale = to_device(one);
  BfBuf ddst(256);
  vidfab::cuda::launch_dequant_f8e4m3(dsrc.get(), dscale.get(), ddst.p(), 256, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = ddst.host();

  // e4m3 has 3 mantissa bits and bf16 has 7, so the round trip through bf16 is
  // lossless and exact equality is the right bar. 0x7F and 0xFF are NaN, whose
  // payload is not something either side promises.
  size_t mismatches = 0;
  int first = -1;
  size_t nan_checked = 0;
  for (int i = 0; i < 256; ++i) {
    const float want = vidfab::f8_e4m3_to_f32(patterns[i]);
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
  vidfab::cuda::launch_dequant_f8e4m3(dsrc.get(), dscale2.get(), ddst.p(), 256, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> scaled = ddst.host();
  std::vector<float> want_scaled(256);
  std::vector<float> got_scaled(256);
  size_t kept = 0;
  for (int i = 0; i < 256; ++i) {
    const float w = vidfab::f8_e4m3_to_f32(patterns[i]);
    if (std::isnan(w)) continue;
    want_scaled[kept] = vidfab::bf16_to_f32(vidfab::f32_to_bf16(w * s[0]));
    got_scaled[kept] = scaled[i];
    ++kept;
  }
  want_scaled.resize(kept);
  got_scaled.resize(kept);
  CHECK_CLOSE(want_scaled, got_scaled, 0.0, "dequant_f8e4m3 with weight_scale");
}

// The e4m3 encoder is only used by the (deferred) native fp8 path, but it has
// to be right before that path can be trusted. Every finite pattern must
// survive dequantise -> quantise unchanged.
VIDFAB_TEST(nn_quantize_f8e4m3_roundtrip) {
  std::vector<float> values;
  std::vector<uint8_t> expect;
  for (int i = 0; i < 256; ++i) {
    const float v = vidfab::f8_e4m3_to_f32(uint8_t(i));
    if (std::isnan(v)) continue;
    if (v == 0.0f && (i & 0x80)) continue;  // -0 encodes back as +0's pattern only by sign
    values.push_back(v);
    expect.push_back(uint8_t(i));
  }

  BfBuf dsrc(values);
  DeviceBuffer<uint8_t> ddst(values.size());
  vidfab::cuda::launch_quantize_f8e4m3(dsrc.p(), 1.0f, ddst.get(), values.size(), nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
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
  vidfab::cuda::launch_quantize_f8e4m3(dbig.p(), 1.0f, dbigq.get(), big.size(), nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
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
    vidfab::cuda::launch_quantize_f8e4m3(dbig.p(), 0.0f, dbigq.get(), big.size(), nullptr);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}

VIDFAB_TEST(nn_dequant_i8_per_channel) {
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
  vidfab::cuda::launch_dequant_i8_per_channel(dw.get(), dscale.get(), ddst.p(), out_features,
                                              in_features, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> want(w.size());
  for (int o = 0; o < out_features; ++o) {
    for (int i = 0; i < in_features; ++i) {
      want[size_t(o) * in_features + i] = vidfab::bf16_to_f32(
          vidfab::f32_to_bf16(float(w[size_t(o) * in_features + i]) * scale[o]));
    }
  }
  CHECK_CLOSE(want, ddst.host(), 0.0, "dequant_i8 per output channel");
}

// The ConvRot rotation, against an explicitly constructed 256x256 matrix. This
// is the test that separates the regular Hadamard from the Sylvester one — the
// wrong choice gives relative error 1.4, not a crash.
VIDFAB_TEST(nn_convrot_matches_regular_hadamard) {
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
  vidfab::cuda::launch_convrot_f32(dx.get(), dy.get(), rows, group, group, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
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
  vidfab::cuda::launch_convrot_f32(dy.get(), dz.get(), rows, group, group, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
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
  vidfab::cuda::launch_convrot_f32(dxm.get(), dym.get(), rows, dim, group, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
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
  vidfab::cuda::launch_convrot_f32(dx16.get(), dy16.get(), rows, 16, 16, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(w16, to_host(dy16), 1e-6, "convrot at group 16");
}

// --- linear -----------------------------------------------------------------

struct CublasScope {
  cublasHandle_t h = nullptr;
  CublasScope() { VIDFAB_CUBLAS_CHECK(cublasCreate(&h)); }
  ~CublasScope() { cublasDestroy(h); }
};

VIDFAB_TEST(linear_bf16_and_fp8) {
  CublasScope cb;
  const int rows = 33;
  const int in_features = 256;
  const int out_features = 96;

  const std::vector<float> x = bf16_round(make_data(size_t(rows) * in_features, 141u, 0.1f));
  BfBuf dx(x);
  BfBuf dy(size_t(rows) * out_features);

  vidfab::cuda::LinearRunner runner;
  runner.init(cb.h, nullptr);

  // 1. dense bf16, with a bias.
  {
    const std::vector<float> w =
        bf16_round(make_data(size_t(out_features) * in_features, 142u, 0.1f));
    const std::vector<float> bias = make_data(out_features, 143u, 0.5f);
    BfBuf dw(w);
    DeviceBuffer<float> dbias = to_device(bias);

    vidfab::cuda::QuantWeight qw;
    qw.format = vidfab::cuda::QuantFormat::kBF16;
    qw.data = dw.p();
    qw.out_features = out_features;
    qw.in_features = in_features;
    qw.bias = dbias.get();
    qw.bias_format = vidfab::cuda::QuantFormat::kF32;
    CHECK(qw.stored_bytes() == size_t(out_features) * in_features * 2);

    Workspace ws;
    ws.reserve(vidfab::cuda::linear_workspace_bytes(qw, rows, vidfab::cuda::ComputeType::kBF16) +
               256);
    runner.forward(qw, dx.p(), rows, dy.p(), ws);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());

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

    vidfab::cuda::QuantWeight qw;
    qw.format = vidfab::cuda::QuantFormat::kF8E4M3;
    qw.data = draw.get();
    qw.out_features = out_features;
    qw.in_features = in_features;
    qw.weight_scale = dscale.get();
    qw.input_scale = 3.53655e-2f;
    CHECK(qw.stored_bytes() == size_t(out_features) * in_features);

    std::vector<float> w(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
      w[i] = vidfab::bf16_to_f32(vidfab::f32_to_bf16(vidfab::f8_e4m3_to_f32(raw[i]) * wscale));
    }
    const std::vector<float> want = cpu_matmul_nt(x, w, rows, out_features, in_features);

    Workspace ws;
    ws.reserve(vidfab::cuda::linear_workspace_bytes(qw, rows, vidfab::cuda::ComputeType::kBF16) +
               256);
    runner.forward(qw, dx.p(), rows, dy.p(), ws);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> got_dequant = dy.host();
    CHECK_CLOSE_REL(want, got_dequant, 1e-3, 1e-2, "linear fp8 dequantise-then-GEMM");

    // set_native must not change the answer beyond tolerance. It currently
    // falls back to the same path; this pins the contract for when it does not.
    runner.set_native(true);
    runner.forward(qw, dx.p(), rows, dy.p(), ws);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CLOSE_REL(got_dequant, dy.host(), 1e-3, 1e-2, "linear fp8 native == dequantised");

    // The load-bearing case: no input_scale means full precision. It must still
    // produce the same result with native enabled.
    qw.input_scale = 0.0f;
    runner.forward(qw, dx.p(), rows, dy.p(), ws);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CLOSE_REL(got_dequant, dy.host(), 1e-3, 1e-2,
                    "linear fp8 with no input_scale stays full precision");
    runner.set_native(false);
  }

  // 3. fp32 path.
  {
    const std::vector<float> w = make_data(size_t(out_features) * in_features, 144u, 0.1f);
    DeviceBuffer<float> dw = to_device(w);
    vidfab::cuda::QuantWeight qw;
    qw.format = vidfab::cuda::QuantFormat::kF32;
    qw.data = dw.get();
    qw.out_features = out_features;
    qw.in_features = in_features;

    DeviceBuffer<float> fx = to_device(x);
    DeviceBuffer<float> fy(size_t(rows) * out_features);
    Workspace ws;
    ws.reserve(vidfab::cuda::linear_workspace_bytes(qw, rows, vidfab::cuda::ComputeType::kF32) +
               256);
    runner.forward_f32(qw, fx.get(), rows, fy.get(), ws);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CLOSE(cpu_matmul_nt(x, w, rows, out_features, in_features), to_host(fy), 1e-3,
                "linear fp32");
  }
}

// The end-to-end ConvRot path: the stored weight is already rotated, so the
// activation must be rotated online or the result is `x H W^T`, which is
// well-scaled noise (docs/convrot_notes.md).
VIDFAB_TEST(linear_int8_convrot) {
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

  vidfab::cuda::QuantWeight qw;
  qw.format = vidfab::cuda::QuantFormat::kI8;
  qw.data = dw.get();
  qw.out_features = out_features;
  qw.in_features = in_features;
  qw.weight_scale = dscale.get();
  qw.per_channel_scale = true;
  qw.convrot = true;
  qw.convrot_group = group;

  vidfab::cuda::LinearRunner runner;
  runner.init(cb.h, nullptr);
  Workspace ws;
  ws.reserve(vidfab::cuda::linear_workspace_bytes(qw, rows, vidfab::cuda::ComputeType::kBF16) +
             256);
  runner.forward(qw, dx.p(), rows, dy.p(), ws);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
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
  vidfab::cuda::QuantWeight qo = qw;
  qo.data = dwo.get();
  qo.in_features = odd_in;
  Workspace ws2;
  ws2.reserve(vidfab::cuda::linear_workspace_bytes(qo, rows, vidfab::cuda::ComputeType::kBF16) +
              256);
  runner.forward(qo, dxo.p(), rows, dyo.p(), ws2);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> wo(w_odd.size());
  for (int o = 0; o < out_features; ++o) {
    for (int i = 0; i < odd_in; ++i) wo[size_t(o) * odd_in + i] = 3.0f * wscale[o];
  }
  CHECK_CLOSE_REL(cpu_matmul_nt(xo, wo, rows, out_features, odd_in), dyo.host(), 1e-3, 1e-2,
                  "convrot skipped when in_features % group != 0");
}

// --- nvfp4 storage ----------------------------------------------------------
//
// Two facts about how the shipped checkpoints store an nvfp4 weight are not
// inferable from the file and were measured against the fp8 build of the same
// model (docs/transformer_spec.md 8.6):
//
//   - the **high** nibble of each byte is the even-indexed element;
//   - block scales are written in a 128x4 tile layout, not row-major.
//
// Both are silent when wrong. They leave the value histogram intact and the
// output finite and correctly scaled, and they cost elementwise correlation
// against the reference — 0.995 becomes 0.00003 — while every summary statistic
// stays where it was. So each wrong form is constructed below and the kernel is
// required *not* to match it; a test that only checked the right answer would
// pass under all four combinations.

// The 128x4 tile map, written out as an explicit walk of the tile and its
// interior rather than as the kernel's packed shift expression, so that the two
// are genuinely independent statements of the same layout.
size_t nvfp4_scale_slot(int o, int k, int blocks_per_row) {
  const int tiles_per_row = blocks_per_row / 4;
  const size_t tile = size_t(o / 128) * tiles_per_row + size_t(k / 4);
  const int row_in_tile = o % 128;
  // The 128 rows of a tile are visited as four groups of 32, the group index
  // moving slower than the row inside it.
  const size_t inside = size_t(row_in_tile % 32) * 16 + size_t(row_in_tile / 32) * 4 + size_t(k % 4);
  return tile * 128 * 4 + inside;
}

// A weight in the checkpoint's storage form, plus everything needed to state
// what it should dequantise to.
struct Nvfp4Weight {
  int out_features = 0;
  int in_features = 0;
  float global = 0.0f;
  std::vector<uint8_t> codes;    // one E2M1 code per element, [out, in]
  std::vector<uint8_t> scales;   // one e4m3 byte per 16 elements, unswizzled [out, in/16]
  std::vector<uint8_t> packed;   // [out, in/2], even element in the high nibble
  std::vector<uint8_t> stored;   // `scales` written through the 128x4 tile map
};

Nvfp4Weight make_nvfp4(int out_features, int in_features, float global, uint32_t seed) {
  Nvfp4Weight w;
  w.out_features = out_features;
  w.in_features = in_features;
  w.global = global;
  const int blocks_per_row = in_features / int(vidfab::cuda::kNVFP4BlockSize);

  uint32_t s = seed;
  auto next = [&]() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return s;
  };

  w.codes.resize(size_t(out_features) * in_features);
  for (size_t i = 0; i < w.codes.size(); ++i) w.codes[i] = uint8_t(next() & 0x0Fu);

  w.scales.resize(size_t(out_features) * blocks_per_row);
  for (size_t i = 0; i < w.scales.size(); ++i) {
    // Exponents 4..11 keep the scales well clear of the subnormals and of the
    // 0x7F/0xFF NaN encodings, and spread over three orders of magnitude so a
    // scale landing on the wrong block cannot go unnoticed.
    const uint32_t r = next();
    w.scales[i] = uint8_t(((4u + (r % 8u)) << 3) | (r >> 8 & 0x07u));
  }

  w.packed.assign(size_t(out_features) * in_features / 2, 0);
  for (int o = 0; o < out_features; ++o) {
    for (int i = 0; i < in_features; i += 2) {
      const size_t e = size_t(o) * in_features + i;
      w.packed[size_t(o) * (in_features / 2) + i / 2] =
          uint8_t((w.codes[e] << 4) | w.codes[e + 1]);
    }
  }

  w.stored.assign(w.scales.size(), 0);
  for (int o = 0; o < out_features; ++o) {
    for (int k = 0; k < blocks_per_row; ++k) {
      w.stored[nvfp4_scale_slot(o, k, blocks_per_row)] = w.scales[size_t(o) * blocks_per_row + k];
    }
  }
  return w;
}

// How the weight should come out. `block` and `swap_nibbles` exist so the same
// function can produce the wrong forms the kernel must be shown to reject.
std::vector<float> nvfp4_reference(const Nvfp4Weight& w, int block = 16,
                                   bool swap_nibbles = false) {
  const int blocks_per_row = w.in_features / int(vidfab::cuda::kNVFP4BlockSize);
  std::vector<float> out(w.codes.size());
  for (int o = 0; o < w.out_features; ++o) {
    for (int i = 0; i < w.in_features; ++i) {
      const int k = std::min(i / block, blocks_per_row - 1);
      // Same order of operations as the kernel, so bf16 rounding matches and
      // exact equality is the right bar.
      const float scale =
          vidfab::f8_e4m3_to_f32(w.scales[size_t(o) * blocks_per_row + k]) * w.global;
      const int src = swap_nibbles ? (i ^ 1) : i;
      const float v = vidfab::f4_e2m1_to_f32(w.codes[size_t(o) * w.in_features + src]) * scale;
      out[size_t(o) * w.in_features + i] = vidfab::bf16_to_f32(vidfab::f32_to_bf16(v));
    }
  }
  return out;
}

VIDFAB_TEST(nn_dequant_nvfp4) {
  // 256 rows spans two row-tiles and 128 columns gives eight scale blocks, so
  // both halves of the tile index vary; a one-tile case would pass under a
  // layout that ignored the tiling entirely.
  const int out_features = 256;
  const int in_features = 128;
  const float global = 1.3580322e-3f;  // the qkv_proj block 0 value, not a power of two
  const Nvfp4Weight w = make_nvfp4(out_features, in_features, global, 20260804u);

  // All sixteen E2M1 codes have to appear or the pattern coverage claim below
  // is empty.
  bool seen[16] = {false};
  for (uint8_t c : w.codes) seen[c] = true;
  for (int c = 0; c < 16; ++c) CHECK(seen[c]);

  DeviceBuffer<uint8_t> dw(w.packed.size());
  dw.copy_from_host(w.packed.data(), w.packed.size());
  DeviceBuffer<uint8_t> dsc(w.stored.size());
  dsc.copy_from_host(w.stored.data(), w.stored.size());
  BfBuf ddst(w.codes.size());

  vidfab::cuda::launch_dequant_nvfp4(dw.get(), dsc.get(), global, ddst.p(), out_features,
                                     in_features, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = ddst.host();

  // E2M1 has one mantissa bit and e4m3 three, so the product carries at most
  // five significant bits and bf16 holds eight: with the same operation order
  // the two sides agree bit for bit.
  CHECK_CLOSE(nvfp4_reference(w), got, 0.0, "dequant nvfp4");

  // Each wrong form, and the distance from it. These are the checks that give
  // the one above any power.
  const std::vector<float> swapped = nvfp4_reference(w, 16, /*swap_nibbles=*/true);
  CHECK_MSG(max_abs_diff(swapped, got) > 1e-4,
            "dequant nvfp4 must take the even element from the HIGH nibble (max diff %.4g)",
            max_abs_diff(swapped, got));

  const std::vector<float> stride32 = nvfp4_reference(w, 32);
  CHECK_MSG(max_abs_diff(stride32, got) > 1e-4,
            "dequant nvfp4 must scale in blocks of 16, not 32 (max diff %.4g)",
            max_abs_diff(stride32, got));

  // Block scales read row-major instead of through the tile map.
  {
    Nvfp4Weight flat = w;
    flat.stored = w.scales;  // as if the file were plain [out, in/16]
    DeviceBuffer<uint8_t> dflat(flat.stored.size());
    dflat.copy_from_host(flat.stored.data(), flat.stored.size());
    BfBuf dout(w.codes.size());
    vidfab::cuda::launch_dequant_nvfp4(dw.get(), dflat.get(), global, dout.p(), out_features,
                                       in_features, nullptr);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_MSG(max_abs_diff(dout.host(), got) > 1e-4,
              "dequant nvfp4 must unswizzle the 128x4 block-scale tiling (max diff %.4g)",
              max_abs_diff(dout.host(), got));
  }

  // A shape the tile map cannot address must be refused rather than silently
  // mis-offset: 200 does not divide by 128, 96 does not divide by 64.
  for (auto bad : {std::pair<int, int>(200, 128), std::pair<int, int>(256, 96)}) {
    bool threw = false;
    try {
      vidfab::cuda::launch_dequant_nvfp4(dw.get(), dsc.get(), global, ddst.p(), bad.first,
                                         bad.second, nullptr);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK_MSG(threw, "launch_dequant_nvfp4 must refuse %dx%d", bad.first, bad.second);
  }
}

// The AWQ per-input-channel activation scale. Only the text encoder's weights
// carry one; the transformer's are all null, which is a positive statement that
// the quantiser folded the scale into the preceding norm.
VIDFAB_TEST(nn_pre_quant_scale) {
  const int rows = 17;
  const int dim = 96;
  const std::vector<float> x = bf16_round(make_data(size_t(rows) * dim, 771u, 1.0f));
  const std::vector<float> scale = bf16_round(make_data(dim, 772u, 0.5f));

  BfBuf dx(x), dscale(scale), dy(x.size());
  vidfab::cuda::launch_pre_quant_scale(dx.p(), dscale.p(), dy.p(), rows, dim, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> want(x.size());
  for (int r = 0; r < rows; ++r) {
    for (int i = 0; i < dim; ++i) {
      want[size_t(r) * dim + i] =
          vidfab::bf16_to_f32(vidfab::f32_to_bf16(x[size_t(r) * dim + i] * scale[i]));
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

// The whole path: a stored nvfp4 weight through LinearRunner against a host
// matmul of the dequantised reference.
VIDFAB_TEST(linear_nvfp4) {
  CublasScope cb;
  const int rows = 29;
  const int in_features = 128;
  const int out_features = 256;
  const float global = 2.899169921875e-3f;

  const Nvfp4Weight w = make_nvfp4(out_features, in_features, global, 4242u);
  const std::vector<float> wdq = nvfp4_reference(w);
  const std::vector<float> x = bf16_round(make_data(size_t(rows) * in_features, 4243u, 0.1f));

  DeviceBuffer<uint8_t> dw(w.packed.size());
  dw.copy_from_host(w.packed.data(), w.packed.size());
  DeviceBuffer<uint8_t> dsc(w.stored.size());
  dsc.copy_from_host(w.stored.data(), w.stored.size());
  BfBuf dx(x), dy(size_t(rows) * out_features);

  vidfab::cuda::QuantWeight qw;
  qw.format = vidfab::cuda::QuantFormat::kNVFP4;
  qw.data = dw.get();
  qw.out_features = out_features;
  qw.in_features = in_features;
  qw.block_scale = dsc.get();
  qw.global_scale = global;
  CHECK(qw.stored_bytes() == size_t(out_features) * in_features / 2);

  vidfab::cuda::LinearRunner runner;
  runner.init(cb.h, nullptr);
  Workspace ws;
  ws.reserve(vidfab::cuda::linear_workspace_bytes(qw, rows, vidfab::cuda::ComputeType::kBF16) +
             256);
  runner.forward(qw, dx.p(), rows, dy.p(), ws);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());

  const std::vector<float> want = cpu_matmul_nt(x, wdq, rows, out_features, in_features);
  CHECK_CLOSE_REL(want, dy.host(), 1e-3, 1e-2, "linear nvfp4 dequantise-then-GEMM");

  // A GEMM against the nibble-swapped weight is well scaled and completely
  // wrong, which is the shape of the failure this format invites.
  const std::vector<float> wrong = cpu_matmul_nt(
      x, nvfp4_reference(w, 16, /*swap_nibbles=*/true), rows, out_features, in_features);
  CHECK_MSG(max_abs_diff(wrong, dy.host()) > 1e-3,
            "linear nvfp4 must not match the nibble-swapped weight (max diff %.4g)",
            max_abs_diff(wrong, dy.host()));

  // A weight the checkpoint did not flag full_precision may take a native path
  // later; enabling it must not move the answer.
  const std::vector<float> got = dy.host();
  runner.set_native(true);
  runner.forward(qw, dx.p(), rows, dy.p(), ws);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE_REL(got, dy.host(), 1e-3, 1e-2, "linear nvfp4 native == dequantised");
  runner.set_native(false);

  // With an AWQ activation scale the runner must scale the activation, not the
  // weight — the two differ because the GEMM is not symmetric in them.
  {
    const std::vector<float> pqs = bf16_round(make_data(in_features, 4244u, 0.5f));
    BfBuf dpqs(pqs);
    vidfab::cuda::QuantWeight aw = qw;
    aw.pre_quant_scale = dpqs.p();

    std::vector<float> xs(x.size());
    for (int r = 0; r < rows; ++r) {
      for (int i = 0; i < in_features; ++i) {
        xs[size_t(r) * in_features + i] = vidfab::bf16_to_f32(
            vidfab::f32_to_bf16(x[size_t(r) * in_features + i] * pqs[i]));
      }
    }
    Workspace ws2;
    ws2.reserve(vidfab::cuda::linear_workspace_bytes(aw, rows, vidfab::cuda::ComputeType::kBF16) +
                256);
    runner.forward(aw, dx.p(), rows, dy.p(), ws2);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CLOSE_REL(cpu_matmul_nt(xs, wdq, rows, out_features, in_features), dy.host(), 1e-3,
                    1e-2, "linear nvfp4 with pre_quant_scale");
    CHECK_MSG(max_abs_diff(got, dy.host()) > 1e-3,
              "pre_quant_scale must actually reach the activation (max diff %.4g)",
              max_abs_diff(got, dy.host()));
  }
}

// --- nvfp4 tensor core ------------------------------------------------------
//
// The shipped nvfp4 checkpoints are packed for
//
//   mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale
//       .scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
//
// and every byte of that instruction's operand layout has to be right before a
// GEMM built on it can be trusted. The A/B packing is inferable from the 8-bit
// m16n8k32 layout; the block-scale operand is not, so it was established by
// experiment and is pinned here. Getting either wrong yields finite, plausibly
// scaled garbage -- the failure mode this whole project keeps running into.

__device__ __host__ inline float e2m1_ref(uint8_t n) {
  const float m[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float v = m[n & 7];
  return (n & 8) ? -v : v;
}

__global__ void nvfp4_mma_kernel(const uint32_t* a, const uint32_t* b, const uint32_t* sa,
                                 float* out) {
  const int lane = threadIdx.x;
  const uint32_t ra[4] = {a[lane * 4], a[lane * 4 + 1], a[lane * 4 + 2], a[lane * 4 + 3]};
  const uint32_t rb[2] = {b[lane * 2], b[lane * 2 + 1]};
  const uint32_t s_a = sa[lane];
  const uint32_t s_b = 0x38383838u;  // four e4m3 1.0 scales
  float c[4] = {0, 0, 0, 0};
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X"
      ".f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]), "r"(s_a),
        "r"(s_b));
  const int gid = lane >> 2, tig = lane & 3;
  out[gid * 8 + tig * 2] = c[0];
  out[gid * 8 + tig * 2 + 1] = c[1];
  out[(gid + 8) * 8 + tig * 2] = c[2];
  out[(gid + 8) * 8 + tig * 2 + 1] = c[3];
}

// The lane that carries row r's block scales. Rows 0-7 sit on lane 4r, rows
// 8-15 on lane 4(r-8)+1; lanes 4g+2 and 4g+3 carry nothing, which is why only
// 64 of the warp's 128 scale bytes are live.
int scale_lane_for_row(int r) { return r < 8 ? 4 * r : 4 * (r - 8) + 1; }

VIDFAB_TEST(nvfp4_mma_operand_layout) {
  uint8_t A[16][64], B[64][8];
  for (int r = 0; r < 16; ++r)
    for (int k = 0; k < 64; ++k) A[r][k] = static_cast<uint8_t>((r * 7 + k * 3) % 15);
  for (int k = 0; k < 64; ++k)
    for (int c = 0; c < 8; ++c) B[k][c] = static_cast<uint8_t>((k * 5 + c * 11) % 15);

  // Each register packs eight consecutive-k nibbles; the register pair splits
  // rows at 8 and the pair-of-pairs splits k at 32.
  std::vector<uint32_t> ha(32 * 4, 0), hb(32 * 2, 0), hs(32, 0x38383838u);
  for (int lane = 0; lane < 32; ++lane) {
    const int gid = lane >> 2, tig = lane & 3;
    for (int i = 0; i < 8; ++i) {
      ha[lane * 4 + 0] |= uint32_t(A[gid][tig * 8 + i] & 15) << (4 * i);
      ha[lane * 4 + 1] |= uint32_t(A[gid + 8][tig * 8 + i] & 15) << (4 * i);
      ha[lane * 4 + 2] |= uint32_t(A[gid][32 + tig * 8 + i] & 15) << (4 * i);
      ha[lane * 4 + 3] |= uint32_t(A[gid + 8][32 + tig * 8 + i] & 15) << (4 * i);
      hb[lane * 2 + 0] |= uint32_t(B[tig * 8 + i][gid] & 15) << (4 * i);
      hb[lane * 2 + 1] |= uint32_t(B[32 + tig * 8 + i][gid] & 15) << (4 * i);
    }
  }

  DeviceBuffer<uint32_t> da(ha.size()), db(hb.size()), ds(hs.size());
  da.copy_from_host(ha.data(), ha.size());
  db.copy_from_host(hb.data(), hb.size());
  ds.copy_from_host(hs.data(), hs.size());
  DeviceBuffer<float> dout(128);
  nvfp4_mma_kernel<<<1, 32>>>(da.get(), db.get(), ds.get(), dout.get());
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = to_host(dout);

  std::vector<float> want(128, 0.0f);
  for (int r = 0; r < 16; ++r)
    for (int c = 0; c < 8; ++c)
      for (int k = 0; k < 64; ++k) want[r * 8 + c] += e2m1_ref(A[r][k]) * e2m1_ref(B[k][c]);
  // fp4 products of these magnitudes accumulate exactly in fp32, so this is an
  // equality, not a tolerance.
  CHECK_CLOSE_REL(want, got, 0.0, 0.0, "nvfp4 m16n8k64 A/B/accumulator layout");

  // Block scales: A and B all 1.0, so every row sums to 4 blocks x 16 = 64.
  // Doubling one (lane, byte) must lift exactly its own row by exactly 16.
  const std::vector<uint32_t> ones(32 * 4, 0x22222222u), onesb(32 * 2, 0x22222222u);
  DeviceBuffer<uint32_t> ua(ones.size()), ub(onesb.size()), dsx(32);
  ua.copy_from_host(ones.data(), ones.size());
  ub.copy_from_host(onesb.data(), onesb.size());
  for (int r : {0, 3, 7, 8, 11, 15}) {
    for (int blk : {0, 3}) {
      std::vector<uint32_t> s(32, 0x38383838u);
      const int lane = scale_lane_for_row(r);
      s[lane] = (s[lane] & ~(0xFFu << (8 * blk))) | (uint32_t(0x40) << (8 * blk));  // e4m3 2.0
      dsx.copy_from_host(s.data(), s.size());
      nvfp4_mma_kernel<<<1, 32>>>(ua.get(), ub.get(), dsx.get(), dout.get());
      VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
      const std::vector<float> g = to_host(dout);
      for (int rr = 0; rr < 16; ++rr) {
        const float expect = rr == r ? 80.0f : 64.0f;  // 64 + 16 on the scaled row
        CHECK_NEAR(g[rr * 8], expect, 1e-3);
      }
    }
  }
}

// --- attention --------------------------------------------------------------

VIDFAB_TEST(attention_blocked) {
  CublasScope cb;
  const int seq = 512;
  const int heads = 4;
  const int head_dim = 128;
  const int width = heads * head_dim;

  const std::vector<float> q = bf16_round(make_data(size_t(seq) * width, 161u, 0.3f));
  const std::vector<float> k = bf16_round(make_data(size_t(seq) * width, 162u, 0.3f));
  const std::vector<float> v = bf16_round(make_data(size_t(seq) * width, 163u, 1.0f));

  BfBuf dq(q), dk(k), dv(v);

  vidfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq;
  cfg.num_heads = heads;
  cfg.head_dim = head_dim;
  CHECK_NEAR(cfg.effective_scale(), 1.0 / std::sqrt(128.0), 1e-7);  // computed in fp32

  const std::vector<float> want =
      cpu_attention(q, k, v, seq, heads, heads, head_dim, cfg.effective_scale());

  // The result must not depend on query_block. A broken online-softmax rescale
  // is silent under a single block size and only shows up here.
  std::vector<std::vector<float>> results;
  const int blocks[] = {64, 128, 512};
  for (int bq : blocks) {
    cfg.query_block = bq;
    BfBuf dout(size_t(seq) * width);
    Workspace ws;
    ws.reserve(vidfab::cuda::attention_workspace_bytes(cfg, vidfab::cuda::AttentionBackend::kBlocked));
    vidfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                    vidfab::cuda::AttentionBackend::kBlocked, ws);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    results.push_back(dout.host());
    CHECK_CLOSE_REL(want, results.back(), 1e-3, 1e-2,
                    ("attention vs dense CPU, query_block " + std::to_string(bq)).c_str());
  }
  CHECK_CLOSE_REL(results[0], results[1], 1e-3, 1e-2, "attention query_block 64 == 128");
  CHECK_CLOSE_REL(results[0], results[2], 1e-3, 1e-2, "attention query_block 64 == 512");

  // The fused backend must agree with the dense CPU reference and with the
  // blocked backend it replaces. It takes no workspace, so pass an empty one --
  // if it ever starts allocating, this fails rather than silently reading
  // whatever the caller happened to leave reserved.
  {
    cfg.query_block = 1024;
    BfBuf dfused(size_t(seq) * width);
    Workspace ws;
    CHECK(vidfab::cuda::attention_workspace_bytes(cfg, vidfab::cuda::AttentionBackend::kFused) == 0);
    vidfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dfused.p(), cfg,
                                    vidfab::cuda::AttentionBackend::kFused, ws);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> got = dfused.host();
    CHECK_CLOSE_REL(want, got, 1e-3, 1e-2, "fused attention vs dense CPU");
    CHECK_CLOSE_REL(results[0], got, 1e-3, 1e-2, "fused attention == blocked attention");
  }

  // head_dim outside the instantiated set must be refused, not silently wrong.
  {
    vidfab::cuda::AttentionConfig odd = cfg;
    odd.head_dim = 96;
    bool threw = false;
    try {
      Workspace ws;
      BfBuf dodd(size_t(seq) * heads * 96);
      vidfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dodd.p(), odd,
                                      vidfab::cuda::AttentionBackend::kFused, ws);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw);
  }
}

// The fused kernel's tail handling is invisible at a sequence that divides the
// block sizes evenly. kBr is 64 and kBc is 32, so a prime sequence exercises a
// ragged query tile and a ragged key step at once -- the case where a masked
// column would otherwise contribute exp(0) = 1 to the denominator, which is
// wrong by a factor that grows with how much of the tile is padding.
VIDFAB_TEST(attention_fused_ragged_tail) {
  CublasScope cb;
  const int heads = 3;
  const int head_dim = 128;
  const int width = heads * head_dim;

  for (int seq : {17, 61, 127, 199}) {
    const std::vector<float> q = bf16_round(make_data(size_t(seq) * width, 401u + seq, 0.3f));
    const std::vector<float> k = bf16_round(make_data(size_t(seq) * width, 402u + seq, 0.3f));
    const std::vector<float> v = bf16_round(make_data(size_t(seq) * width, 403u + seq, 1.0f));
    BfBuf dq(q), dk(k), dv(v), dout(size_t(seq) * width);

    vidfab::cuda::AttentionConfig cfg;
    cfg.seq_len = seq;
    cfg.num_heads = heads;
    cfg.head_dim = head_dim;

    const std::vector<float> want =
        cpu_attention(q, k, v, seq, heads, heads, head_dim, cfg.effective_scale());

    Workspace ws;
    vidfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                    vidfab::cuda::AttentionBackend::kFused, ws);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CLOSE_REL(want, dout.host(), 1e-3, 1e-2,
                    ("fused attention, ragged seq " + std::to_string(seq)).c_str());
  }
}

// Grouped-query: 6 query heads share 2 kv heads. Getting the kv head index wrong
// still produces finite, plausibly-scaled output, so it is pinned against the
// CPU reference rather than against a shape check.
VIDFAB_TEST(attention_fused_gqa) {
  CublasScope cb;
  const int seq = 96;
  const int heads = 6;
  const int kv_heads = 2;
  const int head_dim = 128;

  const std::vector<float> q = bf16_round(make_data(size_t(seq) * heads * head_dim, 511u, 0.3f));
  const std::vector<float> k = bf16_round(make_data(size_t(seq) * kv_heads * head_dim, 512u, 0.3f));
  const std::vector<float> v = bf16_round(make_data(size_t(seq) * kv_heads * head_dim, 513u, 1.0f));
  BfBuf dq(q), dk(k), dv(v), dout(size_t(seq) * heads * head_dim);

  vidfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq;
  cfg.num_heads = heads;
  cfg.head_dim = head_dim;

  const std::vector<float> want =
      cpu_attention(q, k, v, seq, heads, kv_heads, head_dim, cfg.effective_scale());

  Workspace ws;
  vidfab::cuda::attention_forward_gqa(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                      kv_heads, vidfab::cuda::AttentionBackend::kFused, ws);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE_REL(want, dout.host(), 1e-3, 1e-2, "fused gqa attention vs dense CPU");
}

// The score tile is fp16 and doubles as the probability buffer. Two things have
// to hold and neither is visible from the shapes above, where the tile budget
// swallows the whole sequence in a single key block:
//
//   - the online softmax must give the same answer however the keys are split,
//     which is what exercises the running max, the correction factor and the
//     accumulator rescale at all; and
//   - the fp16 tile has to stay inside tolerance against an fp64-ordered
//     reference. The error is *recorded* here, not just bounded, so that a
//     future format change has a number to beat rather than an assertion to
//     satisfy.
VIDFAB_TEST(attention_fp16_score_tile) {
  CublasScope cb;
  const int seq = 1024;
  const int heads = 4;
  const int head_dim = 128;
  const int width = heads * head_dim;

  // Amplitude 0.5 over 128 channels puts scores at a few units — the same order
  // as post-q_norm/k_norm H3, and far from the 65504 where fp16 would saturate.
  const std::vector<float> q = bf16_round(make_data(size_t(seq) * width, 181u, 0.5f));
  const std::vector<float> k = bf16_round(make_data(size_t(seq) * width, 182u, 0.5f));
  const std::vector<float> v = bf16_round(make_data(size_t(seq) * width, 183u, 1.0f));
  BfBuf dq(q), dk(k), dv(v);

  vidfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq;
  cfg.num_heads = heads;
  cfg.head_dim = head_dim;

  const std::vector<float> want =
      cpu_attention(q, k, v, seq, heads, heads, head_dim, cfg.effective_scale());

  double norm_sq = 0.0;
  for (float f : want) norm_sq += double(f) * f;
  const double want_rms = std::sqrt(norm_sq / double(want.size()));

  // `out` is bf16, so *any* correct implementation is at least this far from the
  // fp64 reference. Measuring the kernel against a fixed constant would mostly
  // measure that floor — one bf16 ulp is 0.4 % relative — so the bound below is
  // expressed as a multiple of it instead.
  const std::vector<float> rounded = bf16_round(want);
  double floor_sq = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double d = double(rounded[i]) - want[i];
    floor_sq += d * d;
  }
  const double output_floor_rel = std::sqrt(floor_sq / double(want.size())) / want_rms;

  // Key blocks that do and do not divide the sequence, plus one that leaves a
  // short tail, crossed with two query blocks.
  const int key_blocks[] = {0, 128, 256, 384, 1024};
  const int query_blocks[] = {256, 1024};
  std::vector<std::vector<float>> results;
  double worst_rms_rel = 0.0;
  double worst_abs = 0.0;

  for (int kb : key_blocks) {
    for (int qb : query_blocks) {
      cfg.key_block = kb;
      cfg.query_block = qb;
      BfBuf dout(size_t(seq) * width);
      Workspace ws;
      ws.reserve(
          vidfab::cuda::attention_workspace_bytes(cfg, vidfab::cuda::AttentionBackend::kBlocked));
      vidfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                      vidfab::cuda::AttentionBackend::kBlocked, ws);
      VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
      const std::vector<float> got = dout.host();

      double err_sq = 0.0;
      for (size_t i = 0; i < got.size(); ++i) {
        const double d = double(got[i]) - want[i];
        err_sq += d * d;
        worst_abs = std::max(worst_abs, std::fabs(d));
      }
      worst_rms_rel = std::max(worst_rms_rel, std::sqrt(err_sq / double(got.size())) / want_rms);

      CHECK_CLOSE_REL(want, got, 1e-3, 1e-2,
                      ("attention key_block " + std::to_string(kb) + " query_block " +
                       std::to_string(qb))
                          .c_str());
      results.push_back(got);
    }
  }

  // Every split must agree to the same tolerance. A correction factor applied to
  // the wrong operand, or skipped when the running max happens not to move,
  // shows up here and nowhere else in this file.
  for (size_t i = 1; i < results.size(); ++i) {
    CHECK_CLOSE_REL(results[0], results[i], 1e-3, 1e-2,
                    "attention result independent of key/query block");
  }

  // Recorded, not merely bounded. Measured: rms_rel 1.67e-3 against a 1.66e-3
  // bf16 output floor, a ratio of 1.01 — the score path contributes
  // sqrt(1.67^2 - 1.66^2) = 1.8e-4, well under the quantisation of the format it
  // is written to. At this output precision an fp16 score tile is
  // indistinguishable from exact arithmetic.
  //
  // Be clear about what the assertion is worth: at ratio 1.01 it has no power to
  // separate fp16 probabilities from bf16 ones, because neither is visible
  // through a bf16 output. It catches gross regressions — an fp8 tile, a lost
  // fp32 accumulator, a correction factor applied to the wrong operand — and the
  // key/query block sweep above is what actually pins the online softmax.
  std::printf("  fp16 score tile: worst rms_rel %.3g (bf16 output floor %.3g, ratio %.2f), "
              "worst abs %.3g, output rms %.3g\n",
              worst_rms_rel, output_floor_rel, worst_rms_rel / output_floor_rel, worst_abs,
              want_rms);
  CHECK_MSG(worst_rms_rel < 1.3 * output_floor_rel,
            "fp16 score tile rms_rel %.4g exceeds 1.3x the %.4g bf16 output floor", worst_rms_rel,
            output_floor_rel);
}

VIDFAB_TEST(attention_gqa) {
  CublasScope cb;
  const int seq = 256;
  const int heads = 8;
  const int kv_heads = 2;
  const int head_dim = 64;

  const std::vector<float> q = bf16_round(make_data(size_t(seq) * heads * head_dim, 171u, 0.3f));
  const std::vector<float> k = bf16_round(make_data(size_t(seq) * kv_heads * head_dim, 172u, 0.3f));
  const std::vector<float> v = bf16_round(make_data(size_t(seq) * kv_heads * head_dim, 173u, 1.0f));

  BfBuf dq(q), dk(k), dv(v);
  BfBuf dout(size_t(seq) * heads * head_dim);

  vidfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq;
  cfg.num_heads = heads;
  cfg.head_dim = head_dim;
  cfg.query_block = 64;

  Workspace ws;
  ws.reserve(vidfab::cuda::attention_workspace_bytes(cfg, vidfab::cuda::AttentionBackend::kBlocked));
  vidfab::cuda::attention_forward_gqa(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                      kv_heads, vidfab::cuda::AttentionBackend::kBlocked, ws);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());

  const std::vector<float> want =
      cpu_attention(q, k, v, seq, heads, kv_heads, head_dim, cfg.effective_scale());
  CHECK_CLOSE_REL(want, dout.host(), 1e-3, 1e-2, "gqa attention, 8 query heads over 2 kv heads");

  // A wrong query-head -> kv-head mapping (say h % kv_heads instead of
  // h / group) would still be finite, so pin that it is not what happened.
  std::vector<float> wrong(size_t(seq) * heads * head_dim);
  {
    std::vector<float> k_perm(k.size());
    std::vector<float> v_perm(k.size());
    for (int t = 0; t < seq; ++t) {
      for (int kv = 0; kv < kv_heads; ++kv) {
        const int other = (kv + 1) % kv_heads;
        for (int d = 0; d < head_dim; ++d) {
          k_perm[size_t(t) * kv_heads * head_dim + kv * head_dim + d] =
              k[size_t(t) * kv_heads * head_dim + other * head_dim + d];
          v_perm[size_t(t) * kv_heads * head_dim + kv * head_dim + d] =
              v[size_t(t) * kv_heads * head_dim + other * head_dim + d];
        }
      }
    }
    wrong = cpu_attention(q, k_perm, v_perm, seq, heads, kv_heads, head_dim, cfg.effective_scale());
  }
  CHECK_MSG(max_abs_diff(wrong, dout.host()) > 0.01,
            "gqa must map query head h to kv head h/(H/Hkv) (max diff %.4g)",
            max_abs_diff(wrong, dout.host()));
}

VIDFAB_TEST(workspace_arena) {
  Workspace ws;
  ws.reserve(4096);
  CHECK(ws.capacity() >= 4096);
  void* a = ws.alloc(100);
  void* b = ws.alloc(100);
  // 256-byte grain, so consecutive carvings are cuBLAS-aligned.
  CHECK(reinterpret_cast<uintptr_t>(b) - reinterpret_cast<uintptr_t>(a) == 256);
  CHECK(reinterpret_cast<uintptr_t>(a) % 256 == 0);
  {
    Workspace::Scope scope(ws);
    ws.alloc(1024);
    CHECK(ws.used() > 512);
  }
  CHECK(ws.used() == 356);
  bool threw = false;
  try {
    ws.alloc(1u << 30);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
  ws.clear();
  CHECK(ws.used() == 0);
}

// --- timings ----------------------------------------------------------------
//
// Not an assertion, a measurement. The numbers go to the CUDA critic alongside
// the kernels; the only thing checked here is that the production shapes run at
// all on this card.

struct Timer {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  Timer() {
    VIDFAB_CUDA_CHECK(cudaEventCreate(&start));
    VIDFAB_CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~Timer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  template <class F>
  float measure(F&& f, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) f();
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    VIDFAB_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) f();
    VIDFAB_CUDA_CHECK(cudaEventRecord(stop));
    VIDFAB_CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    VIDFAB_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms / float(iters);
  }
};

// The dequantiser runs once per GEMM on every one of the 200 quantised linears,
// fifty times a step, so its cost is not incidental. It is pure streaming and
// should sit near the card's bandwidth: 2 bytes written and 9/16 read per
// element, of which the store is the overwhelming majority.
VIDFAB_TEST(nvfp4_dequant_timings) {
  Timer timer;
  const struct {
    const char* name;
    int out_features;
    int in_features;
  } shapes[] = {
      {"blocks.N.attn.qkv_proj", 21504, 5376},
      {"blocks.N.attn.out_proj", 5376, 7168},
      {"blocks.N.mlp.fc1", 28672, 5376},
      {"blocks.N.mlp.fc2", 5376, 14336},
  };

  for (const auto& s : shapes) {
    const size_t n = size_t(s.out_features) * s.in_features;
    DeviceBuffer<uint8_t> dw(n / 2);
    DeviceBuffer<uint8_t> dsc(n / size_t(vidfab::cuda::kNVFP4BlockSize));
    BfBuf ddst(n);
    // 0x38 is e4m3 1.0; the codes themselves do not affect timing.
    VIDFAB_CUDA_CHECK(cudaMemset(dw.get(), 0x52, dw.nbytes()));
    VIDFAB_CUDA_CHECK(cudaMemset(dsc.get(), 0x38, dsc.nbytes()));

    float ms = 1e30f;
    for (int pass = 0; pass < 3; ++pass) {
      ms = std::min(ms, timer.measure(
                            [&] {
                              vidfab::cuda::launch_dequant_nvfp4(dw.get(), dsc.get(), 1.0f,
                                                                 ddst.p(), s.out_features,
                                                                 s.in_features, nullptr);
                            },
                            3, 20));
    }
    const double bytes = double(n) * 2.0 + double(n) / 2.0 + double(n) / 16.0;
    std::printf("  %-24s %6d x %5d  %7.3f ms  %7.1f GB/s\n", s.name, s.out_features,
                s.in_features, ms, bytes / (double(ms) * 1e-3) / 1e9);
    CHECK(ms > 0.0f);
  }
}

VIDFAB_TEST(production_shape_timings) {
  CublasScope cb;
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  VIDFAB_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  std::printf("  device free %.2f GiB of %.2f GiB\n", double(free_bytes) / (1 << 30),
              double(total_bytes) / (1 << 30));

  // The default 124-frame request packs 37710 rows. If the card cannot hold the
  // buffers alongside everything else, fall back and say so rather than skip.
  int seq = 37710;
  const int heads = 56;
  const int head_dim = 128;
  const int width = heads * head_dim;
  const int model_dim = 5376;

  // q,k,v,out at [seq, 7168] bf16 plus the score tile and accumulator.
  while (seq > 1024 && size_t(seq) * width * 2ull * 4ull + (400ull << 20) > free_bytes * 3 / 4) {
    seq /= 2;
  }
  if (seq != 37710) {
    std::printf("  NOTE: attention measured at seq %d, not 37710, to fit in free memory\n", seq);
  }

  Timer timer;

  {
    BfBuf q(size_t(seq) * width), k(size_t(seq) * width), v(size_t(seq) * width),
        out(size_t(seq) * width);
    VIDFAB_CUDA_CHECK(cudaMemset(q.raw.get(), 0x3C, q.raw.nbytes()));
    VIDFAB_CUDA_CHECK(cudaMemset(k.raw.get(), 0x3B, k.raw.nbytes()));
    VIDFAB_CUDA_CHECK(cudaMemset(v.raw.get(), 0x3D, v.raw.nbytes()));

    vidfab::cuda::AttentionConfig cfg;
    cfg.seq_len = seq;
    cfg.num_heads = heads;
    cfg.head_dim = head_dim;
    cfg.query_block = 1024;
    const size_t ws_bytes =
        vidfab::cuda::attention_workspace_bytes(cfg, vidfab::cuda::AttentionBackend::kBlocked);
    Workspace ws;
    ws.reserve(ws_bytes);
    std::printf("  attention workspace %.2f GiB (score tile, accumulator, fp16 k/v/q)\n",
                double(ws_bytes) / (1 << 30));
    // Something else on this box touches the GPU intermittently: roughly one run
    // in six comes back at half throughput. Take the best of three passes rather
    // than believing a single number.
    float ms = 1e30f;
    for (int pass = 0; pass < 3; ++pass) {
      ms = std::min(ms, timer.measure(
                            [&] {
                              vidfab::cuda::attention_forward(cb.h, nullptr, q.p(), k.p(), v.p(),
                                                              out.p(), cfg,
                                                              vidfab::cuda::AttentionBackend::kBlocked,
                                                              ws);
                            },
                            1, 3));
    }
    const double flops = 4.0 * double(seq) * seq * head_dim * heads;
    std::printf("  attention   seq=%-6d heads=56 head_dim=128  %8.2f ms  (%.1f TFLOP/s)\n", seq,
                ms, flops / (ms * 1e-3) / 1e12);
    CHECK(ms > 0.0f);

    // Same buffers, same timer, same best-of-three: the only honest way to
    // report what removing the HBM round trip actually bought.
    Workspace none;
    float fms = 1e30f;
    for (int pass = 0; pass < 3; ++pass) {
      fms = std::min(fms, timer.measure(
                              [&] {
                                vidfab::cuda::attention_forward(
                                    cb.h, nullptr, q.p(), k.p(), v.p(), out.p(), cfg,
                                    vidfab::cuda::AttentionBackend::kFused, none);
                              },
                              1, 3));
    }
    std::printf("  attention   fused (no workspace)                %8.2f ms  (%.1f TFLOP/s)\n", fms,
                flops / (fms * 1e-3) / 1e12);
    std::printf("  fused speedup %.2fx over blocked, workspace %.2f GiB -> 0\n", ms / fms,
                double(ws_bytes) / (1 << 30));
    CHECK(fms > 0.0f);
  }

  {
    // q_norm/k_norm: 128-wide rows, the narrow-row path. Bytes are read+write of
    // the activation; the 128-element weight stays in L2.
    const int qk_heads = 56;
    const int qk_dim = 128;
    const size_t n = size_t(seq) * qk_heads * qk_dim;
    const size_t wn = size_t(qk_dim);
    BfBuf x(n), w(wn);
    VIDFAB_CUDA_CHECK(cudaMemset(x.raw.get(), 0x3C, x.raw.nbytes()));
    VIDFAB_CUDA_CHECK(cudaMemset(w.raw.get(), 0x3F, w.raw.nbytes()));
    float ms = 1e30f;
    for (int pass = 0; pass < 3; ++pass) {
      ms = std::min(ms, timer.measure(
                            [&] {
                              vidfab::cuda::launch_head_rmsnorm(x.p(), w.p(), seq, qk_heads, qk_dim,
                                                                1e-5f, nullptr);
                            },
                            3, 20));
    }
    const double bytes = 2.0 * double(n) * 2.0;
    std::printf("  head_rmsnorm rows=%-6d heads=56 dim=128     %8.3f ms  (%.0f GB/s)\n", seq, ms,
                bytes / (ms * 1e-3) / 1e9);
    CHECK(ms > 0.0f);
  }

  // The four transformer GEMM shapes, at the same row count.
  struct Shape {
    const char* name;
    int out_features;
    int in_features;
  };
  const Shape shapes[] = {
      {"qkv_proj", 21504, 5376},
      {"out_proj", 5376, 7168},
      {"mlp.fc1", 28672, 5376},
      {"mlp.fc2", 5376, 14336},
  };

  vidfab::cuda::LinearRunner runner;
  runner.init(cb.h, nullptr);
  for (const Shape& sh : shapes) {
    const size_t xn = size_t(seq) * sh.in_features;
    const size_t yn = size_t(seq) * sh.out_features;
    const size_t wn = size_t(sh.out_features) * sh.in_features;
    VIDFAB_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if ((xn * 2 + yn * 2 + wn * 3) > free_bytes * 3 / 4) {
      std::printf("  linear %-9s SKIPPED, needs more than the free memory\n", sh.name);
      continue;
    }
    DeviceBuffer<uint8_t> raw(wn);
    VIDFAB_CUDA_CHECK(cudaMemset(raw.get(), 0x38, raw.nbytes()));
    const std::vector<float> sv(1, 8.1264e-3f);
    DeviceBuffer<float> dscale = to_device(sv);
    BfBuf dx(xn), dy(yn);
    VIDFAB_CUDA_CHECK(cudaMemset(dx.raw.get(), 0x3C, dx.raw.nbytes()));

    vidfab::cuda::QuantWeight qw;
    qw.format = vidfab::cuda::QuantFormat::kF8E4M3;
    qw.data = raw.get();
    qw.out_features = sh.out_features;
    qw.in_features = sh.in_features;
    qw.weight_scale = dscale.get();

    Workspace ws;
    ws.reserve(vidfab::cuda::linear_workspace_bytes(qw, seq, vidfab::cuda::ComputeType::kBF16) +
               256);
    float ms = 1e30f;
    for (int pass = 0; pass < 2; ++pass) {
      ms = std::min(ms, timer.measure([&] { runner.forward(qw, dx.p(), seq, dy.p(), ws); }, 2, 10));
    }
    const double flops = 2.0 * double(seq) * sh.out_features * sh.in_features;
    std::printf("  linear %-9s [%5d,%5d] rows=%-6d  %8.3f ms  (%.1f TFLOP/s)\n", sh.name,
                sh.out_features, sh.in_features, seq, ms, flops / (ms * 1e-3) / 1e12);
    CHECK(ms > 0.0f);
  }

  {
    const size_t norm_n = size_t(seq) * model_dim;
    const size_t weight_n = size_t(model_dim);
    BfBuf x(norm_n), out(norm_n), w(weight_n);
    VIDFAB_CUDA_CHECK(cudaMemset(x.raw.get(), 0x3C, x.raw.nbytes()));
    VIDFAB_CUDA_CHECK(cudaMemset(w.raw.get(), 0x3F, w.raw.nbytes()));
    float ms_norm = 1e30f;
    for (int pass = 0; pass < 3; ++pass) {
      ms_norm = std::min(ms_norm, timer.measure(
                                      [&] {
                                        vidfab::cuda::launch_rmsnorm(x.p(), w.p(), out.p(), seq,
                                                                     model_dim, 1e-5f, nullptr);
                                      },
                                      3, 20));
    }
    const double bytes = 3.0 * double(seq) * model_dim * 2.0;
    std::printf("  rmsnorm     rows=%-6d dim=5376              %8.3f ms  (%.0f GB/s)\n", seq,
                ms_norm, bytes / (ms_norm * 1e-3) / 1e9);

    const int mod_rows = 3;
    DeviceBuffer<float> scale(size_t(mod_rows) * model_dim);
    DeviceBuffer<float> shift(size_t(mod_rows) * model_dim);
    scale.zero();
    shift.zero();
    std::vector<int32_t> a(seq);
    for (int r = 0; r < seq; ++r) a[r] = r % mod_rows;
    DeviceBuffer<int32_t> da = to_device_i32(a);
    float ms_mod = 1e30f;
    for (int pass = 0; pass < 3; ++pass) {
      ms_mod = std::min(ms_mod,
                        timer.measure(
                            [&] {
                              vidfab::cuda::launch_rmsnorm_modulate(x.p(), w.p(), scale.get(),
                                                                    shift.get(), da.get(), out.p(),
                                                                    seq, model_dim, 1e-5f, nullptr);
                            },
                            3, 20));
    }
    std::printf("  rmsnorm_mod rows=%-6d dim=5376              %8.3f ms\n", seq, ms_mod);
    CHECK(ms_norm > 0.0f && ms_mod > 0.0f);
  }
}

}  // namespace
