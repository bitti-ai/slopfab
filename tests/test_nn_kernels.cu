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
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "harness.h"
#include "vidfab/cuda/attention.cuh"
#include "vidfab/cuda/sol_attention.cuh"
#include "vidfab/cuda/device.h"
#include "vidfab/cuda/gemm.cuh"
#include "vidfab/cuda/linear.cuh"
#include "vidfab/cuda/nn_kernels.cuh"
#include "vidfab/cuda/nvfp4_gemm.cuh"
#include "vidfab/cuda/sage_attention.cuh"
#include "vidfab/cuda/workspace.cuh"
#include "vidfab/dit/packing.h"
#include "vidfab/dtype.h"
#include "vidfab/text/qwen_vision.h"
#include "vidfab/safetensors.h"

namespace {

using vidfab::cuda::DeviceBuffer;
using vidfab::cuda::Workspace;
using vidfab::test::make_data;

bool test_is_sm120() {
  return vidfab::cuda::current_device_compute_capability() == 120;
}

#define REQUIRE_SM120_TEST(feature)                                      \
  do {                                                                    \
    if (!test_is_sm120()) {                                               \
      SKIP_UNSUPPORTED_HARDWARE("%s requires the shipped SM120 image", feature); \
      return;                                                             \
    }                                                                     \
  } while (false)

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

// Independent one-head Sol-Attn oracle for routing/correction tests.
std::vector<float> cpu_sol_attention(const std::vector<float>& q, const std::vector<float>& k,
                                     const std::vector<float>& v, int seq, int prefix,
                                     float scale, float beta, int* selected, int* rejected) {
  constexpr int block = 64, dim = 128;
  const int nb = (seq + block - 1) / block;
  std::vector<float> km(size_t(nb) * dim), vs(size_t(nb) * dim), mean(dim), var(dim);
  for (int kb = 0; kb < nb; ++kb) {
    const int lo = kb * block, hi = std::min(lo + block, seq);
    for (int d = 0; d < dim; ++d) {
      for (int r = lo; r < hi; ++r) {
        km[size_t(kb) * dim + d] += k[size_t(r) * dim + d];
        vs[size_t(kb) * dim + d] += v[size_t(r) * dim + d];
      }
      km[size_t(kb) * dim + d] = vidfab::bf16_to_f32(vidfab::f32_to_bf16(
          km[size_t(kb) * dim + d] / float(hi - lo)));
    }
  }
  for (int d = 0; d < dim; ++d) {
    for (int kb = 0; kb < nb; ++kb) mean[d] += km[size_t(kb) * dim + d] / float(nb);
    for (int kb = 0; kb < nb; ++kb) {
      const float x = km[size_t(kb) * dim + d] - mean[d];
      var[d] += x * x / float(nb);
    }
  }
  std::vector<float> out(size_t(seq) * dim), qm(dim), logits(seq);
  for (int row = 0; row < seq; ++row) {
    const int qb = row / block, qlo = qb * block, qhi = std::min(qlo + block, seq);
    std::fill(qm.begin(), qm.end(), 0.0f);
    double mu = 0, vv = 0;
    for (int d = 0; d < dim; ++d) {
      for (int r = qlo; r < qhi; ++r) qm[d] += q[size_t(r) * dim + d];
      qm[d] /= float(qhi - qlo);
      mu += double(qm[d]) * mean[d];
      vv += double(qm[d]) * qm[d] * var[d];
    }
    const double tau = scale * (mu + beta * std::sqrt(std::max(vv, 0.0)));
    double max_logit = -1e300;
    std::vector<uint8_t> take(nb);
    for (int kb = 0; kb < nb; ++kb) {
      double proxy = 0;
      for (int d = 0; d < dim; ++d) proxy += double(qm[d]) * km[size_t(kb) * dim + d];
      take[kb] = qlo < prefix || kb * block < prefix || std::abs(qb - kb) <= 1 ||
                 proxy * scale > tau;
      if (row == qlo) take[kb] ? ++*selected : ++*rejected;
      const int lo = kb * block, hi = std::min(lo + block, seq);
      if (take[kb]) {
        for (int kr = lo; kr < hi; ++kr) {
          double dot = 0;
          for (int d = 0; d < dim; ++d) dot += double(q[size_t(row) * dim + d]) * k[size_t(kr) * dim + d];
          logits[kr] = float(dot * scale);
          max_logit = std::max(max_logit, double(logits[kr]));
        }
      } else {
        double dot = 0;
        for (int d = 0; d < dim; ++d) dot += double(q[size_t(row) * dim + d]) * km[size_t(kb) * dim + d];
        logits[lo] = float(dot * scale);
        max_logit = std::max(max_logit, double(logits[lo]));
      }
    }
    double denom = 0;
    for (int kb = 0; kb < nb; ++kb) {
      const int lo = kb * block, hi = std::min(lo + block, seq);
      if (take[kb]) for (int kr = lo; kr < hi; ++kr) denom += std::exp(logits[kr] - max_logit);
      else denom += (hi - lo) * std::exp(logits[lo] - max_logit);
    }
    for (int d = 0; d < dim; ++d) {
      double num = 0;
      for (int kb = 0; kb < nb; ++kb) {
        const int lo = kb * block, hi = std::min(lo + block, seq);
        if (take[kb]) for (int kr = lo; kr < hi; ++kr) num += std::exp(logits[kr] - max_logit) * v[size_t(kr) * dim + d];
        else num += std::exp(logits[lo] - max_logit) * vs[size_t(kb) * dim + d];
      }
      out[size_t(row) * dim + d] = float(num / denom);
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

#if 0  // Removed: canonical H3 tables are host-built and tested in test_packing.cpp.
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

#endif

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

// `launch_sub_bf16`, and specifically the aliasing contract the block cache
// depends on (dit/block_cache.h).
//
// The block cache holds *one* residual-stream-sized buffer instead of two —
// 844 MB at the production geometry — by parking the pre-span state in the
// delta buffer and subtracting in place, `sub(x, delta, delta)`. That is only
// sound because the kernel is elementwise at a single index: each thread reads
// and writes the same elements, so no thread reads what another has written.
// Nothing else in the tree checks it, and an end-to-end byte comparison — which
// is how it was validated once — would not survive a rewrite of the kernel into
// a grid-stride or shared-memory-staged form, both of which break `out == b`
// silently and produce plausible, finite, wrong deltas.
VIDFAB_TEST(nn_sub_bf16_aliasing_and_tails) {
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
  vidfab::cuda::launch_sub_bf16(da.p(), db.p(), dout.p(), n, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> disjoint = dout.host();
  CHECK_CLOSE(want, disjoint, 1e-6, "sub_bf16 out of place");

  // In place over `b`, which is what the block cache does. Must agree with the
  // out-of-place result exactly — not approximately: the same arithmetic on the
  // same inputs, so any difference is an aliasing bug and not rounding.
  BfBuf da2(a), db2(b);
  vidfab::cuda::launch_sub_bf16(da2.p(), db2.p(), db2.p(), n, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
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
  vidfab::cuda::launch_sub_bf16(da3.p() + 1, db3.p() + 1, dout3.p() + 1, m, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  // `host()` returns by value, so it must be held in a named local: taking
  // `.begin()` from one call and `.end()` from another walks between two
  // different temporaries.
  const std::vector<float> h3 = dout3.host();
  const std::vector<float> got_off(h3.begin() + 1, h3.end());
  CHECK_CLOSE(want_off, got_off, 1e-6, "sub_bf16 misaligned fallback");

  // A zero count must be a no-op rather than a launch with a zero grid.
  vidfab::cuda::launch_sub_bf16(da.p(), db.p(), dout.p(), 0, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(dout.bits() == BfBuf(disjoint).bits());
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

// Hoisting the dequantisation out of a row-chunk loop must be BIT-identical to
// dequantising per chunk, not merely close: it is the whole justification for
// `prepare` + `forward_prepared`, and a tolerance-based check here would pass
// on a version that dequantised a stale or mis-sliced weight.
VIDFAB_TEST(linear_prepared_matches_per_chunk) {
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

  vidfab::cuda::QuantWeight qw;
  qw.format = vidfab::cuda::QuantFormat::kF8E4M3;
  qw.data = draw.get();
  qw.out_features = out_features;
  qw.in_features = in_features;
  qw.weight_scale = dscale.get();
  qw.bias = dbias.get();
  qw.bias_format = vidfab::cuda::QuantFormat::kF32;

  vidfab::cuda::LinearRunner runner;
  runner.init(cb.h, nullptr);

  Workspace ws;
  ws.reserve(vidfab::cuda::linear_workspace_bytes(qw, chunk, vidfab::cuda::ComputeType::kBF16) +
             256);

  // Reference: dequantise inside the loop, once per chunk.
  for (int start = 0; start < rows; start += chunk) {
    const int n = std::min(chunk, rows - start);
    runner.forward(qw, dx.p() + size_t(start) * in_features, n,
                   dref.p() + size_t(start) * out_features, ws);
  }
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());

  // Hoisted: dequantise once, then run every chunk against that copy. Sized
  // from the split rather than from `linear_workspace_bytes`, which is the
  // arithmetic `plan_carve` now relies on.
  Workspace hw;
  hw.reserve(vidfab::cuda::linear_dense_weight_bytes(qw) +
             vidfab::cuda::linear_activation_workspace_bytes(qw, chunk,
                                                             vidfab::cuda::ComputeType::kBF16) +
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
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());

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

// `nn_dequant_nvfp4` pins the two layout facts on one 256x128 weight. This
// sweeps the tile counts instead, because the swizzle's address map is the part
// that varies with shape and the part a rewrite gets wrong: a tile count of one
// in a dimension hides a missing tile stride, and a count that is a power of two
// hides a missing multiply.
//
// Deliberately written against the launcher rather than any one kernel's
// indexing, so it survives a change of strategy inside `launch_dequant_nvfp4`.
// It was added alongside a tile-per-block dequant that was then reverted for
// being slower, and it passed unchanged across both — which is the property
// wanted from it.
//
// Exact equality is the bar, not a tolerance: the reference below is built from
// the independent tile walk in `nvfp4_scale_slot` and multiplies in the same
// order the kernel does.
VIDFAB_TEST(nn_dequant_nvfp4_tile_shapes) {
  const float global = 1.3580322e-3f;
  struct Shape { int out; int in; };
  const Shape shapes[] = {
      {128, 64},    // exactly one tile, the degenerate case
      {128, 512},   // one row-tile, eight k-tiles
      {384, 128},   // three row-tiles: not a power of two
      {256, 320},   // five k-tiles, likewise
      {512, 64},    // one k-tile, four row-tiles
  };
  for (int i = 0; i < 5; ++i) {
    const int out_features = shapes[i].out, in_features = shapes[i].in;
    const Nvfp4Weight w = make_nvfp4(out_features, in_features, global, 90210u + 7u * i);
    DeviceBuffer<uint8_t> dw(w.packed.size());
    dw.copy_from_host(w.packed.data(), w.packed.size());
    DeviceBuffer<uint8_t> dsc(w.stored.size());
    dsc.copy_from_host(w.stored.data(), w.stored.size());
    BfBuf ddst(w.codes.size());
    vidfab::cuda::launch_dequant_nvfp4(dw.get(), dsc.get(), global, ddst.p(), out_features,
                                       in_features, nullptr);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> got = ddst.host();
    const std::vector<float> want = nvfp4_reference(w);
    CHECK_MSG(max_abs_diff(want, got) == 0.0,
              "dequant nvfp4 %dx%d (tiles %dx%d) is not exact: max diff %.6g", out_features,
              in_features, out_features / 128, in_features / 64, max_abs_diff(want, got));

    // A scale that lands on the wrong tile still produces plausible output, so
    // check that this shape can tell the difference at all: perturb one stored
    // scale byte and require the result to move.
    if (w.stored.size() > 1) {
      std::vector<uint8_t> poked = w.stored;
      const size_t at = poked.size() / 3;
      poked[at] = uint8_t(poked[at] ^ 0x08u);  // one exponent step
      DeviceBuffer<uint8_t> dpoke(poked.size());
      dpoke.copy_from_host(poked.data(), poked.size());
      BfBuf dalt(w.codes.size());
      vidfab::cuda::launch_dequant_nvfp4(dw.get(), dpoke.get(), global, dalt.p(), out_features,
                                         in_features, nullptr);
      VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
      CHECK_MSG(max_abs_diff(dalt.host(), got) > 0.0,
                "dequant nvfp4 %dx%d ignored a perturbed block scale at byte %zu", out_features,
                in_features, at);
    }
  }
}

struct Nf4Weight {
  int out = 0, in = 0;
  float offset = 0.0f;
  std::vector<uint8_t> packed, absmax;
  std::vector<float> map, nested_map, nested_absmax;
};

Nf4Weight make_nf4(int out, int in) {
  Nf4Weight w;
  w.out = out; w.in = in; w.offset = 0.21360844373703003f;
  w.map = {-1.0f, -0.6961928f, -0.52507305f, -0.39491749f, -0.28444138f,
           -0.18477343f, -0.09105004f, 0.0f, 0.07958030f, 0.16093020f,
           0.24611230f, 0.33791524f, 0.44070983f, 0.56261700f, 0.72295684f, 1.0f};
  w.nested_map.resize(256);
  for (int i = 0; i < 256; ++i) w.nested_map[i] = (float(i) - 127.0f) / 128.0f;
  const size_t n = size_t(out) * in;
  const size_t blocks = (n + 63) / 64;
  w.nested_absmax.resize((blocks + 255) / 256);
  for (size_t i = 0; i < w.nested_absmax.size(); ++i) w.nested_absmax[i] = 0.75f + float(i) * 1.25f;
  w.absmax.resize(blocks);
  for (size_t i = 0; i < blocks; ++i) w.absmax[i] = uint8_t((i * 73 + 19) & 255);
  w.packed.resize((n + 1) / 2);
  for (size_t i = 0; i < w.packed.size(); ++i) {
    // Deliberately different nibbles; a swapped implementation cannot pass.
    w.packed[i] = uint8_t((((i * 5 + 3) & 15) << 4) | ((i * 11 + 9) & 15));
  }
  return w;
}

std::vector<float> nf4_reference(const Nf4Weight& w, bool swap = false) {
  const size_t n = size_t(w.out) * w.in;
  std::vector<float> result(n);
  for (size_t i = 0; i < n; ++i) {
    const size_t block = i / 64;
    const float scale = w.nested_map[w.absmax[block]] * w.nested_absmax[block / 256] + w.offset;
    const uint8_t byte = w.packed[i / 2];
    const bool high = ((i & 1) == 0) != swap;
    const uint8_t code = high ? byte >> 4 : byte & 15;
    result[i] = vidfab::bf16_to_f32(vidfab::f32_to_bf16(w.map[code] * scale));
  }
  return result;
}

VIDFAB_TEST(nn_dequant_nf4_double_quant) {
  // 258 weight-scale blocks cross a 256-scale nested boundary.
  const Nf4Weight w = make_nf4(129, 128);
  DeviceBuffer<uint8_t> dp(w.packed.size()), da(w.absmax.size());
  dp.copy_from_host(w.packed.data(), w.packed.size());
  da.copy_from_host(w.absmax.data(), w.absmax.size());
  auto dm = to_device(w.map), dnm = to_device(w.nested_map), dna = to_device(w.nested_absmax);
  BfBuf got(size_t(w.out) * w.in);
  vidfab::cuda::launch_dequant_nf4(dp.get(), da.get(), dm.get(), dnm.get(), dna.get(), 64, 256,
                                   w.offset, got.p(), w.out, w.in, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(nf4_reference(w), got.host(), 0.0, "double-quant NF4 dequantisation");
  CHECK_MSG(max_abs_diff(nf4_reference(w, true), got.host()) > 0.1,
            "NF4 even element must use HIGH nibble (max diff %.4g)",
            max_abs_diff(nf4_reference(w, true), got.host()));
}

// NF4 has two kernels now: an eight-wide one that loads four packed bytes at a
// time and a two-wide fallback for the shapes and alignments it cannot serve.
// Which one runs is decided entirely by `nf4_vector_eligible` in the launcher,
// so the two cases below are not guesses: 129x128 is 16512 elements out of two
// 256-byte-aligned DeviceBuffers with block sizes 64 and 256, which satisfies
// every clause, and offsetting the destination by one bf16 makes it 2-byte
// aligned, which fails the 16-byte clause and nothing else.
//
// Comparing the raw payloads is the point. The scalar kernel is the reference
// implementation this replaces; "same to within tolerance" would not establish
// anything, because the whole claim is that no bit moves.
VIDFAB_TEST(nn_dequant_nf4_vector_matches_scalar) {
  const Nf4Weight w = make_nf4(129, 128);
  const size_t n = size_t(w.out) * w.in;
  CHECK(n % 8 == 0);
  DeviceBuffer<uint8_t> dp(w.packed.size()), da(w.absmax.size());
  dp.copy_from_host(w.packed.data(), w.packed.size());
  da.copy_from_host(w.absmax.data(), w.absmax.size());
  auto dm = to_device(w.map), dnm = to_device(w.nested_map), dna = to_device(w.nested_absmax);

  DeviceBuffer<uint16_t> fast(n), slow(n + 8);
  vidfab::cuda::launch_dequant_nf4(dp.get(), da.get(), dm.get(), dnm.get(), dna.get(), 64, 256,
                                   w.offset, reinterpret_cast<__nv_bfloat16*>(fast.get()), w.out,
                                   w.in, nullptr);
  vidfab::cuda::launch_dequant_nf4(dp.get(), da.get(), dm.get(), dnm.get(), dna.get(), 64, 256,
                                   w.offset, reinterpret_cast<__nv_bfloat16*>(slow.get() + 1),
                                   w.out, w.in, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
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
  vidfab::cuda::launch_dequant_nf4_f16(dp.get(), da.get(), dm.get(), dnm.get(), dna.get(), 64, 256,
                                       w.offset, reinterpret_cast<__half*>(ffast.get()), n,
                                       nullptr);
  vidfab::cuda::launch_dequant_nf4_f16(dp.get(), da.get(), dm.get(), dnm.get(), dna.get(), 64, 256,
                                       w.offset, reinterpret_cast<__half*>(fslow.get() + 1), n,
                                       nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
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
  vidfab::cuda::launch_dequant_nf4(rp.get(), ra.get(), rm.get(), rnm.get(), rna.get(), 64, 256,
                                   r.offset, ragged.p(), r.out, r.in, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(nf4_reference(r), ragged.host(), 0.0, "NF4 ragged tail via the scalar kernel");
}

VIDFAB_TEST(linear_nf4_double_quant) {
  CublasScope cb;
  const int rows = 7;
  const Nf4Weight w = make_nf4(16, 128);
  DeviceBuffer<uint8_t> dp(w.packed.size()), da(w.absmax.size());
  dp.copy_from_host(w.packed.data(), w.packed.size());
  da.copy_from_host(w.absmax.data(), w.absmax.size());
  auto dm = to_device(w.map), dnm = to_device(w.nested_map), dna = to_device(w.nested_absmax);
  const auto x = bf16_round(make_data(size_t(rows) * w.in, 20260807u, 0.1f));
  BfBuf dx(x), dy(size_t(rows) * w.out);
  vidfab::cuda::QuantWeight qw;
  qw.format = vidfab::cuda::QuantFormat::kNF4; qw.data = dp.get();
  qw.out_features = w.out; qw.in_features = w.in; qw.nf4_absmax = da.get();
  qw.nf4_quant_map = dm.get(); qw.nf4_nested_quant_map = dnm.get();
  qw.nf4_nested_absmax = dna.get(); qw.nf4_nested_offset = w.offset;
  CHECK(qw.stored_bytes() == w.packed.size());
  vidfab::cuda::LinearRunner runner; runner.init(cb.h, nullptr);
  Workspace ws; ws.reserve(vidfab::cuda::linear_workspace_bytes(
      qw, rows, vidfab::cuda::ComputeType::kBF16) + 256);
  runner.forward(qw, dx.p(), rows, dy.p(), ws);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const auto want = cpu_matmul_nt(x, nf4_reference(w), rows, w.out, w.in);
  CHECK_CLOSE_REL(want, dy.host(), 1e-3, 1e-2, "linear NF4 dequantise-then-GEMM");
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

// Defined with the native-GEMM tests at the end of this file, where the rule
// they implement is written down. Declared here so `linear_nvfp4` can hold the
// native path to an fp4-activation reference without moving the definitions out
// of the block they belong to.
std::vector<float> host_quantise_act(const std::vector<float>& x, int rows, int dim);
double rms_rel(const std::vector<float>& want, const std::vector<float>& got);
double correlation(const std::vector<float>& a, const std::vector<float>& b);

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

  // A weight the checkpoint did not flag full_precision takes the native
  // tensor-core path when `set_native` is on. **That path does not compute the
  // same thing as this one, and the difference is arithmetic rather than
  // implementation**: it quantises the activation to nvfp4 too, because neither
  // shipped checkpoint carries an input_scale. E2M1 has one mantissa bit, so
  // the composed result sits about 9% rms from a bf16-activation reference, at
  // every K -- signal and error both grow as sqrt(K), so a dot product cannot
  // average it away. Asserting agreement with the dequantised path here would
  // be asserting that 4-bit activations are free.
  //
  // So the bound is pointed at what it can describe. The native path is held to
  // the same 1e-3 / 1e-2 against a reference that quantises the activation the
  // same way, which is a real assertion -- it passes at ~2e-3 and fails on any
  // operand, stride or scale error -- and the distance from the bf16 path is
  // reported rather than asserted. `nvfp4_activation_cost` measures that
  // distance properly, across three distributions and four values of K;
  // `nvfp4_gemm_exact_fp4_activations` is the control that separates the two.
  const std::vector<float> got = dy.host();
  if (test_is_sm120()) {
    runner.set_native(true);
    runner.forward(qw, dx.p(), rows, dy.p(), ws);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<float> native = dy.host();
    runner.set_native(false);
    CHECK_CLOSE_REL(cpu_matmul_nt(host_quantise_act(x, rows, in_features), wdq, rows,
                                  out_features, in_features),
                    native, 1e-3, 1e-2,
                    "linear nvfp4 native vs an fp4-activation reference");
    std::printf("  linear nvfp4 native vs dequantised: rms_rel %.4f (4-bit activations)\n",
                rms_rel(got, native));
    // Still the same matrix, and still the same one the dequantised path
    // computes: a layout error would take the correlation to ~0, not to 0.99.
    CHECK(correlation(got, native) > 0.99);
  } else {
    SKIP_UNSUPPORTED_HARDWARE(
        "native half of linear_nvfp4 requires the shipped SM120 image");
  }

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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X"
      ".f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]), "r"(s_a),
        "r"(s_b));
#endif
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
  REQUIRE_SM120_TEST("NVFP4 MMA operand-layout test");
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

VIDFAB_TEST(attention_sol) {
  REQUIRE_SM120_TEST("Sol attention");
  CublasScope cb;
  const int seq = 263;  // four full blocks, local routes, and a ragged tail
  const int heads = 2;
  const int dim = 128;
  const int width = heads * dim;
  const std::vector<float> q = bf16_round(make_data(size_t(seq) * width, 911u, 0.3f));
  std::vector<float> k = bf16_round(make_data(size_t(seq) * width, 912u, 0.3f));
  const std::vector<float> v = bf16_round(make_data(size_t(seq) * width, 913u, 1.0f));

  vidfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq;
  cfg.num_heads = heads;
  cfg.head_dim = dim;
  cfg.sol_pipeline = true;
  CHECK_NEAR(cfg.sol_beta, 1.0f, 0.0);
  const size_t bytes =
      vidfab::cuda::attention_workspace_bytes(cfg, vidfab::cuda::AttentionBackend::kSol);
  CHECK(bytes > 0);
  CHECK(bytes < size_t(seq) * heads * dim * sizeof(float));

  auto run = [&](const std::vector<float>& keys, float beta) {
    BfBuf dq(q), dk(keys), dv(v), dout(size_t(seq) * width);
    cfg.sol_beta = beta;
    Workspace ws;
    ws.reserve(bytes);
    vidfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                    vidfab::cuda::AttentionBackend::kSol, ws);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    return dout.host();
  };

  // A cutoff below every finite proxy selects every block, reducing Sol-Attn
  // to exact attention. This pins the common online-softmax state and tail.
  const std::vector<float> want =
      cpu_attention(q, k, v, seq, heads, heads, dim, cfg.effective_scale());
  CHECK_CLOSE_REL(want, run(k, -1.0e6f), 1e-3, 1e-2,
                  "Sol-Attn all-selected equals dense attention");

  // If K is constant within each physical block, the zeroth-order correction
  // is mathematically exact even when every routable block is rejected. This
  // independently catches mean-vs-sum and ragged-tail multiplicity mistakes.
  for (int row = 0; row < seq; ++row) {
    const int source = (row / 64) * 64;
    for (int x = 0; x < width; ++x) k[size_t(row) * width + x] = k[size_t(source) * width + x];
  }
  const std::vector<float> constant_want =
      cpu_attention(q, k, v, seq, heads, heads, dim, cfg.effective_scale());
  CHECK_CLOSE_REL(constant_want, run(k, 1.0e6f), 1e-3, 1e-2,
                  "Sol-Attn rejected constant-K blocks equal dense attention");

  // Forced-exact prefix includes the crossing physical block, so selecting an
  // arbitrary multimodal boundary cannot approximate any prefix key.
  cfg.exact_prefix = 70;
  CHECK_CLOSE_REL(constant_want, run(k, 1.0e6f), 1e-3, 1e-2,
                  "Sol-Attn forced-exact prefix");

  // Nonconstant mixed-route oracle: six blocks ensure prefix, local and
  // threshold-selected routes coexist with corrected rejected routes.
  {
    const int mixed_seq = 321;
    std::vector<float> mq = bf16_round(make_data(size_t(mixed_seq) * dim, 921u, 0.8f));
    std::vector<float> mk = bf16_round(make_data(size_t(mixed_seq) * dim, 922u, 0.8f));
    std::vector<float> mv = bf16_round(make_data(size_t(mixed_seq) * dim, 923u, 1.0f));
    int selected = 0, rejected = 0;
    const std::vector<float> oracle = cpu_sol_attention(
        mq, mk, mv, mixed_seq, 70, 1.0f / std::sqrt(float(dim)), 1.0f, &selected, &rejected);
    CHECK(selected > 0 && rejected > 0);
    BfBuf dqm(mq), dkm(mk), dvm(mv), dom(size_t(mixed_seq) * dim);
    vidfab::cuda::AttentionConfig mixed;
    mixed.seq_len = mixed_seq;
    mixed.num_heads = 1;
    mixed.head_dim = dim;
    mixed.exact_prefix = 70;
    mixed.sol_pipeline = true;
    Workspace mixed_ws;
    mixed_ws.reserve(vidfab::cuda::attention_workspace_bytes(
        mixed, vidfab::cuda::AttentionBackend::kSol));
    vidfab::cuda::attention_forward(cb.h, nullptr, dqm.p(), dkm.p(), dvm.p(), dom.p(), mixed,
                                    vidfab::cuda::AttentionBackend::kSol, mixed_ws);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CLOSE_REL(oracle, dom.host(), 2e-3, 2e-2, "Sol-Attn mixed-route CPU oracle");
  }
}

VIDFAB_TEST(attention_sol_pipeline_exact) {
  REQUIRE_SM120_TEST("Sol TMA pipeline");
  CublasScope cb;
  const int seq = 128, heads = 1, dim = 128;
  const auto q = bf16_round(make_data(size_t(seq) * dim, 921u, 0.3f));
  const auto k = bf16_round(make_data(size_t(seq) * dim, 922u, 0.3f));
  const auto v = bf16_round(make_data(size_t(seq) * dim, 923u, 1.0f));
  const auto want = cpu_attention(q, k, v, seq, heads, heads, dim,
                                  1.0f / std::sqrt(float(dim)));
  BfBuf dq(q), dk(k), dv(v), dout(size_t(seq) * dim);
  vidfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq; cfg.num_heads = heads; cfg.head_dim = dim;
  cfg.exact_prefix = seq; cfg.sol_pipeline = true;
  Workspace ws;
  ws.reserve(vidfab::cuda::attention_workspace_bytes(
      cfg, vidfab::cuda::AttentionBackend::kSol));
  vidfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                  vidfab::cuda::AttentionBackend::kSol, ws);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE_REL(want, dout.host(), 2e-3, 2e-2,
                  "Sol SM120 pipeline exact oracle");
}

VIDFAB_TEST(attention_sol_pipeline_mixed) {
  REQUIRE_SM120_TEST("Sol TMA pipeline");
  CublasScope cb;
  const int seq=384, dim=128;
  const auto q=bf16_round(make_data(size_t(seq)*dim,931u,0.8f));
  const auto k=bf16_round(make_data(size_t(seq)*dim,932u,0.8f));
  const auto v=bf16_round(make_data(size_t(seq)*dim,933u,1.0f));
  int selected=0,rejected=0;
  const auto want=cpu_sol_attention(q,k,v,seq,70,1.0f/std::sqrt(float(dim)),
                                    1.0f,&selected,&rejected);
  CHECK(selected>0 && rejected>0);
  BfBuf dq(q),dk(k),dv(v),dout(size_t(seq)*dim);
  vidfab::cuda::AttentionConfig cfg;
  cfg.seq_len=seq; cfg.num_heads=1; cfg.head_dim=dim;
  cfg.exact_prefix=70; cfg.sol_pipeline=true;
  Workspace ws; ws.reserve(vidfab::cuda::attention_workspace_bytes(
      cfg,vidfab::cuda::AttentionBackend::kSol));
  vidfab::cuda::attention_forward(cb.h,nullptr,dq.p(),dk.p(),dv.p(),dout.p(),cfg,
                                  vidfab::cuda::AttentionBackend::kSol,ws);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE_REL(want,dout.host(),2e-3,2e-2,"Sol pipeline mixed-route CPU oracle");
}

VIDFAB_TEST(attention_sol_pipeline_real_scale_finite) {
  REQUIRE_SM120_TEST("Sol TMA pipeline");
  CublasScope cb;
  // More than 64 physical blocks exercises multiple compacted approximate
  // groups and a ragged tail with activation ranges observed in H3 captures.
  const int seq=4097,dim=128;
  const auto q=bf16_round(make_data(size_t(seq)*dim,941u,14.0f));
  const auto k=bf16_round(make_data(size_t(seq)*dim,942u,12.0f));
  const auto v=bf16_round(make_data(size_t(seq)*dim,943u,72.0f));
  BfBuf dq(q),dk(k),dv(v),dout(size_t(seq)*dim);
  vidfab::cuda::AttentionConfig cfg;
  cfg.seq_len=seq;cfg.num_heads=1;cfg.head_dim=dim;
  cfg.exact_prefix=419;cfg.sol_pipeline=true;cfg.sol_beta=1.0f;
  Workspace ws;ws.reserve(vidfab::cuda::attention_workspace_bytes(
      cfg,vidfab::cuda::AttentionBackend::kSol));
  vidfab::cuda::attention_forward(cb.h,nullptr,dq.p(),dk.p(),dv.p(),dout.p(),cfg,
                                  vidfab::cuda::AttentionBackend::kSol,ws);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const auto got=dout.host();
  size_t bad=0;
  for(float x:got) bad+=!std::isfinite(x);
  CHECK_MSG(bad==0,"Sol real-scale grouped pipeline produced %zu non-finite values",bad);
}

VIDFAB_TEST(attention_sol_pipeline_large_pooled_v) {
  REQUIRE_SM120_TEST("Sol TMA pipeline");
  CublasScope cb;
  // A 64-row pooled V sum is ~131k here: well beyond BF16's finite range,
  // while the corrected attention result remains a perfectly finite ~2k.
  // This reproduces the end-to-end failure that motivated keeping the
  // approximate V contraction in FP32.
  const int seq=321,dim=128;
  const auto q=bf16_round(make_data(size_t(seq)*dim,951u,0.4f));
  auto k=bf16_round(make_data(size_t(seq)*dim,952u,0.4f));
  std::vector<float> v(size_t(seq)*dim);
  for(int row=0;row<seq;++row) {
    const int source=(row/64)*64;
    for(int x=0;x<dim;++x) {
      k[size_t(row)*dim+x]=k[size_t(source)*dim+x];
      v[size_t(row)*dim+x]=2048.0f+float((x%7)-3)*8.0f;
    }
  }
  const auto want=cpu_attention(q,k,v,seq,1,1,dim,1.0f/std::sqrt(float(dim)));
  BfBuf dq(q),dk(k),dv(v),dout(size_t(seq)*dim);
  vidfab::cuda::AttentionConfig cfg;
  cfg.seq_len=seq;cfg.num_heads=1;cfg.head_dim=dim;
  cfg.exact_prefix=0;cfg.sol_pipeline=true;cfg.sol_beta=1.0e6f;
  Workspace ws;ws.reserve(vidfab::cuda::attention_workspace_bytes(
      cfg,vidfab::cuda::AttentionBackend::kSol));
  vidfab::cuda::attention_forward(cb.h,nullptr,dq.p(),dk.p(),dv.p(),dout.p(),cfg,
                                  vidfab::cuda::AttentionBackend::kSol,ws);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const auto got=dout.host();
  size_t bad=0;
  for(float x:got) bad+=!std::isfinite(x);
  CHECK_MSG(bad==0,"Sol large pooled-V pipeline produced %zu non-finite values",bad);
  CHECK_CLOSE_REL(want,got,2e-3,2e-2,"Sol pipeline large pooled-V FP32 correction");
}

VIDFAB_TEST(attention_sol_rejects_invalid_error_weights) {
  REQUIRE_SM120_TEST("Sol attention");
  vidfab::cuda::AttentionConfig cfg;
  cfg.seq_len=64;cfg.num_heads=1;cfg.head_dim=128;
  vidfab::cuda::Workspace ws;
  auto rejected=[&]() {
    try {
      vidfab::cuda::sol_attention_forward(nullptr,nullptr,nullptr,nullptr,nullptr,cfg,ws);
      return false;
    } catch(const std::runtime_error&) { return true; }
  };
  cfg.sol_error_k=-1.0f;CHECK(rejected());
  cfg.sol_error_k=std::numeric_limits<float>::infinity();CHECK(rejected());
  cfg.sol_error_k=0.0f;cfg.sol_error_v=-1.0f;CHECK(rejected());
  cfg.sol_error_v=std::numeric_limits<float>::quiet_NaN();CHECK(rejected());
}

VIDFAB_TEST(attention_sol_zero_error_weight_ignores_infinite_residual) {
  REQUIRE_SM120_TEST("Sol attention");
  CublasScope cb;
  const int seq=321,dim=128;
  std::vector<float> q(size_t(seq)*dim,0.0f),k(size_t(seq)*dim),v;
  v=bf16_round(make_data(size_t(seq)*dim,961u,1.0f));
  // Squaring this finite BF16 value overflows FP32 residual preprocessing,
  // while alternating signs keep the centroid and zero-Q proxy finite.
  for(int row=0;row<seq;++row)for(int d=0;d<dim;++d)
    k[size_t(row)*dim+d]=((row+d)&1)?-3.0e38f:3.0e38f;
  k=bf16_round(k);
  BfBuf dq(q),dk(k),dv(v),dout(size_t(seq)*dim);
  vidfab::cuda::AttentionConfig cfg;
  cfg.seq_len=seq;cfg.num_heads=1;cfg.head_dim=dim;cfg.sol_pipeline=true;
  cfg.sol_beta=1.0e6f;cfg.sol_error_k=0.0f;cfg.sol_error_v=0.0f;
  Workspace ws;ws.reserve(vidfab::cuda::attention_workspace_bytes(
      cfg,vidfab::cuda::AttentionBackend::kSol));
  vidfab::cuda::attention_forward(cb.h,nullptr,dq.p(),dk.p(),dv.p(),dout.p(),cfg,
                                  vidfab::cuda::AttentionBackend::kSol,ws);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  size_t bad=0;for(float x:dout.host())bad+=!std::isfinite(x);
  CHECK_MSG(bad==0,"zero Sol error weights consumed infinite residual: %zu nonfinite",bad);
}

VIDFAB_TEST(attention_sage2_architecture_dispatch) {
  using Variant = vidfab::cuda::Sage2KernelVariant;
  CHECK(vidfab::cuda::sage2_variant_for_compute_capability(79) == Variant::kUnsupported);
  CHECK(vidfab::cuda::sage2_variant_for_compute_capability(80) == Variant::kUnsupported);
  CHECK(vidfab::cuda::sage2_variant_for_compute_capability(85) == Variant::kUnsupported);
  CHECK(vidfab::cuda::sage2_variant_for_compute_capability(86) == Variant::kAmpereFp16);
  CHECK(vidfab::cuda::sage2_variant_for_compute_capability(88) == Variant::kAmpereFp16);
  CHECK(vidfab::cuda::sage2_variant_for_compute_capability(89) == Variant::kUnsupported);
  CHECK(vidfab::cuda::sage2_variant_for_compute_capability(90) == Variant::kUnsupported);
  CHECK(vidfab::cuda::sage2_variant_for_compute_capability(120) == Variant::kBlackwellFp8);
  CHECK(vidfab::cuda::sage2_variant_for_compute_capability(121) == Variant::kUnsupported);
}

VIDFAB_TEST(attention_sage2) {
  CublasScope cb;
  const int seq = 199;
  const int heads = 2;
  const int head_dim = 128;
  const int width = heads * head_dim;
  const std::vector<float> q = bf16_round(make_data(size_t(seq) * width, 871u, 0.3f));
  const std::vector<float> k = bf16_round(make_data(size_t(seq) * width, 872u, 0.3f));
  const std::vector<float> v = bf16_round(make_data(size_t(seq) * width, 873u, 1.0f));
  const std::vector<float> want =
      cpu_attention(q, k, v, seq, heads, heads, head_dim, 1.0f / std::sqrt(128.0f));
  BfBuf dq(q), dk(k), dv(v), dout(size_t(seq) * width);
  vidfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq;
  cfg.num_heads = heads;
  cfg.head_dim = head_dim;
  Workspace ws;
  const size_t bytes =
      vidfab::cuda::attention_workspace_bytes(cfg, vidfab::cuda::AttentionBackend::kSage2);
  CHECK(bytes > 0);
  ws.reserve(bytes);
  vidfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                  vidfab::cuda::AttentionBackend::kSage2, ws);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = dout.host();
  CHECK_CLOSE_REL(want, got, 2.5e-2, 1e-1, "sage2 attention vs dense CPU");

  // Quantization is deterministic, including the ragged Q/K/V padding.
  BfBuf again(size_t(seq) * width);
  ws.clear();
  vidfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), again.p(), cfg,
                                  vidfab::cuda::AttentionBackend::kSage2, ws);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK(dout.bits() == again.bits());

  cfg.band_ranges = reinterpret_cast<const int32_t*>(uintptr_t{16});
  bool threw = false;
  try {
    ws.clear();
    vidfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), again.p(), cfg,
                                    vidfab::cuda::AttentionBackend::kSage2, ws);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK(threw);
}

// sage2's three preparation kernels take head dim as a template parameter, so
// the failure this guards is an index split that is right at one D and wrong at
// the other — invisible unless both widths run in the same process, hence the
// alternating loop.
//
// **The digests are frozen, not printed.** They were captured by running these
// six shapes against a build of this file linked to the pre-refactor
// `sage2_attention_forward` and again against the current one; both produced the
// values below. Comparing round 1 against round 0 of the same build would only
// have proved determinism — a deterministic bit change passes that and moves
// these constants, which is exactly the regression worth catching.
//
// A failure here is not necessarily a bug: it means the numerics moved, and the
// question is whether that was intended. If it was, re-derive the table the same
// way rather than pasting whatever the new build prints.
VIDFAB_TEST(attention_sage2_head_dims_bit_stable) {
  const int heads = 8, kv_heads = 2;
  const bool check_blackwell_hashes = test_is_sm120();
  if (!check_blackwell_hashes) {
    SKIP_UNSUPPORTED_HARDWARE(
        "Blackwell Sage hashes are SM120-specific; dense-reference and determinism still run");
  }
  struct Shape { int seq; int head_dim; uint64_t digest; };
  const Shape shapes[] = {
      {64, 64, 0x3dece381009ce78cull},  {199, 128, 0xc755489173f5999cull},
      {130, 64, 0x51488ef18c4440adull}, {256, 128, 0xd01e0a63a85a68a9ull},
      {65, 64, 0x00fca10d10e3d6b9ull},  {321, 128, 0x0a67856f531f85c4ull},
  };

  std::vector<uint16_t> first[6];
  for (int round = 0; round < 2; ++round) {
    for (int si = 0; si < 6; ++si) {
      const int seq = shapes[si].seq, head_dim = shapes[si].head_dim;
      const size_t qn = size_t(seq) * heads * head_dim;
      const size_t kn = size_t(seq) * kv_heads * head_dim;
      const std::vector<float> q = bf16_round(make_data(qn, 4101u + si, 0.3f));
      const std::vector<float> k = bf16_round(make_data(kn, 4201u + si, 0.3f));
      const std::vector<float> v = bf16_round(make_data(kn, 4301u + si, 1.0f));
      BfBuf dq(q), dk(k), dv(v), dout(qn);
      vidfab::cuda::AttentionConfig cfg;
      cfg.seq_len = seq;
      cfg.num_heads = heads;
      cfg.head_dim = head_dim;
      Workspace ws;
      ws.reserve(vidfab::cuda::sage2_workspace_bytes(cfg, kv_heads));
      vidfab::cuda::sage2_attention_forward(nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                            kv_heads, ws);
      VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
      const std::vector<uint16_t> bits = dout.bits();

      if (round == 0) {
        const std::vector<float> want =
            cpu_attention(q, k, v, seq, heads, kv_heads, head_dim,
                          1.0f / std::sqrt(float(head_dim)));
        CHECK_CLOSE_REL(want, dout.host(), 3.0e-2, 1.2e-1, "sage2 GQA vs dense CPU");
        first[si] = bits;
        // FNV-1a over the raw bf16 payload, against the frozen table above.
        uint64_t h = 1469598103934665603ull;
        for (uint16_t b : bits) {
          h = (h ^ (b & 0xffu)) * 1099511628211ull;
          h = (h ^ (b >> 8)) * 1099511628211ull;
        }
        if (check_blackwell_hashes) {
          CHECK_MSG(h == shapes[si].digest,
                    "sage2 output moved at seq=%d D=%d kv=%d: digest %016llx, expected %016llx",
                    seq, head_dim, kv_heads, static_cast<unsigned long long>(h),
                    static_cast<unsigned long long>(shapes[si].digest));
        }
      } else {
        size_t bad = 0;
        for (size_t i = 0; i < bits.size(); ++i) bad += bits[i] != first[si][i];
        CHECK_MSG(bad == 0,
                  "sage2 seq=%d D=%d drifted across interleaved head dims: %zu of %zu bf16 differ",
                  seq, head_dim, bad, bits.size());
      }
    }
  }
}

// head_dim 64 is the other instantiation `supported()` accepts, and until this
// existed nothing exercised it -- every fused test above is 128-wide. The two
// differ in more than a constant: the staging tiles the key block into 2 passes
// rather than 4 and each warp covers a 32-column group of a 64-wide head rather
// than of a 128-wide one, so a mapping that covers D=128 exactly can still
// double-write or skip columns at D=64. A ragged sequence checks that against
// the tail at the same time.
VIDFAB_TEST(attention_fused_head_dim_64) {
  CublasScope cb;
  const int heads = 5;
  const int head_dim = 64;
  const int width = heads * head_dim;

  for (int seq : {64, 129, 200}) {
    const std::vector<float> q = bf16_round(make_data(size_t(seq) * width, 601u + seq, 0.3f));
    const std::vector<float> k = bf16_round(make_data(size_t(seq) * width, 602u + seq, 0.3f));
    const std::vector<float> v = bf16_round(make_data(size_t(seq) * width, 603u + seq, 1.0f));
    BfBuf dq(q), dk(k), dv(v), dout(size_t(seq) * width);

    vidfab::cuda::AttentionConfig cfg;
    cfg.seq_len = seq;
    cfg.num_heads = heads;
    cfg.head_dim = head_dim;
    CHECK(vidfab::cuda::attention_preferred_backend(cfg) == vidfab::cuda::AttentionBackend::kFused);

    const std::vector<float> want =
        cpu_attention(q, k, v, seq, heads, heads, head_dim, cfg.effective_scale());

    Workspace ws;
    vidfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                    vidfab::cuda::AttentionBackend::kFused, ws);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_CLOSE_REL(want, dout.host(), 1e-3, 1e-2,
                    ("fused attention, head_dim 64, seq " + std::to_string(seq)).c_str());
  }

  // Grouped-query at 64 wide as well: `kvld` is num_kv_heads*64 there, which is
  // the stride the staging walks, and a kv head offset of `kv_head*64` is the
  // one place a 128-wide assumption would survive every test above.
  {
    const int seq = 96;
    const int kv_heads = 1;
    const std::vector<float> q = bf16_round(make_data(size_t(seq) * heads * head_dim, 611u, 0.3f));
    const std::vector<float> k =
        bf16_round(make_data(size_t(seq) * kv_heads * head_dim, 612u, 0.3f));
    const std::vector<float> v =
        bf16_round(make_data(size_t(seq) * kv_heads * head_dim, 613u, 1.0f));
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
    CHECK_CLOSE_REL(want, dout.host(), 1e-3, 1e-2, "fused gqa attention, head_dim 64");
  }
}

// Dense attention restricted to an explicit key set, for frame banding.
//
// It takes the *ranges* rather than a band width, deliberately. The ranges are
// the specification of what the kernel must do; whether they describe the right
// band is a separate question, answered on the host by
// `packing_banded_key_ranges`. Re-deriving an idealised band here would test the
// two implementations against each other's opinion of the rounding, and the
// first disagreement would get resolved by tuning the reference until it
// matched -- which is backwards, and would silently accept a kernel that
// attends to nearly the right keys.
std::vector<float> cpu_attention_banded(const std::vector<float>& q, const std::vector<float>& k,
                                        const std::vector<float>& v, int seq, int heads,
                                        int kv_heads, int head_dim, float scale,
                                        const std::vector<int32_t>& ranges, int query_tile) {
  const int qld = heads * head_dim;
  const int kvld = kv_heads * head_dim;
  const int group = heads / kv_heads;
  std::vector<float> out(size_t(seq) * qld, 0.0f);

  for (int h = 0; h < heads; ++h) {
    const int kv = h / group;
    for (int i = 0; i < seq; ++i) {
      const size_t t = size_t(i / query_tile) * 4;
      const int lo0 = ranges[t + 0], hi0 = std::min(ranges[t + 1], seq);
      const int lo1 = ranges[t + 2], hi1 = std::min(ranges[t + 3], seq);

      double m = -1e300;
      std::vector<std::pair<int, double>> p;
      p.reserve(size_t(hi0 - lo0) + size_t(std::max(0, hi1 - lo1)));
      const auto score = [&](int j) {
        double dot = 0.0;
        for (int d = 0; d < head_dim; ++d) {
          dot += double(q[size_t(i) * qld + h * head_dim + d]) *
                 k[size_t(j) * kvld + kv * head_dim + d];
        }
        const double s = dot * scale;
        m = std::max(m, s);
        p.emplace_back(j, s);
      };
      for (int j = lo0; j < hi0; ++j) score(j);
      for (int j = lo1; j < hi1; ++j) score(j);

      double sum = 0.0;
      for (auto& e : p) {
        e.second = std::exp(e.second - m);
        sum += e.second;
      }
      for (auto& e : p) {
        const double w = e.second / sum;
        for (int d = 0; d < head_dim; ++d) {
          out[size_t(i) * qld + h * head_dim + d] +=
              float(w * v[size_t(e.first) * kvld + kv * head_dim + d]);
        }
      }
    }
  }
  return out;
}

// Does the kernel attend to *exactly* the keys the host asked for?
//
// Nothing else can answer this. A band one key block too narrow produces
// finite, plausibly-scaled output, passes every norm and shape check, and is
// **faster** -- so it would show up as a speedup that beats its own forecast,
// which is the last thing anyone questions. The band edges here are chosen so
// the rounding is live: R = 42 rows per frame against a 64-row key block, so no
// frame boundary lands on a block boundary and every range is rounded outwards.
VIDFAB_TEST(attention_fused_banded) {
  CublasScope cb;
  using vidfab::dit::SequenceLayout;

  SequenceLayout layout;
  layout.num_text = 5;
  layout.num_audio_rows = 8;
  layout.num_latent_frames = 12;
  layout.latent_height = 12;
  layout.latent_width = 14;  // R = 6*7 = 42, coprime-ish with the 64-row block
  layout.num_video_rows = layout.num_latent_frames * layout.rows_per_frame();

  const int seq = layout.total_rows();
  const int heads = 2;
  const int head_dim = 128;
  const int width = heads * head_dim;
  const int tile = vidfab::cuda::attention_fused_query_tile();
  const int align = vidfab::cuda::attention_fused_key_align();
  CHECK(layout.rows_per_frame() == 42);
  CHECK(layout.rows_per_frame() % align != 0);  // the rounding must be exercised

  const std::vector<float> q = bf16_round(make_data(size_t(seq) * width, 701u, 0.3f));
  const std::vector<float> k = bf16_round(make_data(size_t(seq) * width, 702u, 0.3f));
  const std::vector<float> v = bf16_round(make_data(size_t(seq) * width, 703u, 1.0f));
  BfBuf dq(q), dk(k), dv(v);

  vidfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq;
  cfg.num_heads = heads;
  cfg.head_dim = head_dim;

  for (int band : {2, 3, 5, 64}) {
    const vidfab::dit::BandedKeyRanges r =
        vidfab::dit::build_banded_key_ranges(layout, band, tile, align);
    DeviceBuffer<int32_t> dranges(r.ranges.size());
    dranges.copy_from_host(r.ranges.data(), r.ranges.size());

    BfBuf dout(size_t(seq) * width);
    vidfab::cuda::AttentionConfig banded = cfg;
    banded.band_ranges = dranges.get();
    Workspace ws;
    vidfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), banded,
                                    vidfab::cuda::AttentionBackend::kFused, ws);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());

    const std::vector<float> want = cpu_attention_banded(
        q, k, v, seq, heads, heads, head_dim, cfg.effective_scale(), r.ranges, tile);
    CHECK_CLOSE_REL(want, dout.host(), 1e-3, 1e-2,
                    ("banded fused attention, band +/-" + std::to_string(band)).c_str());
  }

  // A band wide enough to span the sequence must reproduce full attention
  // exactly -- same bytes, not merely the same tolerance. If it does not, the
  // banded path differs from the unbanded one for reasons that have nothing to
  // do with banding.
  {
    const vidfab::dit::BandedKeyRanges wide =
        vidfab::dit::build_banded_key_ranges(layout, 64, tile, align);
    DeviceBuffer<int32_t> dranges(wide.ranges.size());
    dranges.copy_from_host(wide.ranges.data(), wide.ranges.size());

    BfBuf dfull(size_t(seq) * width), dwide(size_t(seq) * width);
    Workspace ws;
    vidfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dfull.p(), cfg,
                                    vidfab::cuda::AttentionBackend::kFused, ws);
    vidfab::cuda::AttentionConfig banded = cfg;
    banded.band_ranges = dranges.get();
    vidfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dwide.p(), banded,
                                    vidfab::cuda::AttentionBackend::kFused, ws);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_MSG(dfull.bits() == dwide.bits(),
              "a band covering the sequence is not bit-identical to full attention");
  }

  // A narrow band must actually change the answer. Without this the test above
  // would pass just as well against a kernel that ignored `band_ranges`
  // entirely -- which is the failure the blocked backend throws to avoid.
  {
    const vidfab::dit::BandedKeyRanges narrow =
        vidfab::dit::build_banded_key_ranges(layout, 1, tile, align);
    DeviceBuffer<int32_t> dranges(narrow.ranges.size());
    dranges.copy_from_host(narrow.ranges.data(), narrow.ranges.size());
    BfBuf dfull(size_t(seq) * width), dnarrow(size_t(seq) * width);
    Workspace ws;
    vidfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dfull.p(), cfg,
                                    vidfab::cuda::AttentionBackend::kFused, ws);
    vidfab::cuda::AttentionConfig banded = cfg;
    banded.band_ranges = dranges.get();
    vidfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dnarrow.p(), banded,
                                    vidfab::cuda::AttentionBackend::kFused, ws);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    CHECK_MSG(dfull.bits() != dnarrow.bits(), "a +/-1 frame band did not change the output");
  }

  // The blocked backend has no banding and must refuse rather than quietly
  // returning full attention under a banded caller's name.
  {
    DeviceBuffer<int32_t> dranges(4);
    const std::vector<int32_t> z = {0, seq, 0, 0};
    dranges.copy_from_host(z.data(), z.size());
    vidfab::cuda::AttentionConfig banded = cfg;
    banded.band_ranges = dranges.get();
    BfBuf dout(size_t(seq) * width);
    Workspace ws;
    ws.reserve(vidfab::cuda::attention_workspace_bytes(banded,
                                                       vidfab::cuda::AttentionBackend::kBlocked));
    bool threw = false;
    try {
      vidfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), banded,
                                      vidfab::cuda::AttentionBackend::kBlocked, ws);
    } catch (const std::exception&) {
      threw = true;
    }
    CHECK(threw);
  }
}

// The fused kernel's probability precision is pinned by nothing else in this
// file, and that is a hole rather than an oversight of one change.
//
// `attention_fp16_score_tile` below measures exactly the right quantity, and
// passes `AttentionBackend::kBlocked` at both its workspace sizing and its
// forward call -- it never runs `kFused` at any configuration. Every other
// fused test bounds 1e-3 absolute / 1e-2 relative against the CPU reference,
// and a host model of this kernel's own schedule says fp32, fp16 **and** bf16
// probabilities all pass that bound at every depth from 512 to 32768. So those
// tests cannot see P at all: they would go green whatever it was carried in.
//
// What that leaves unguarded: P feeds a second `mma` whose two operands must
// share a type, so any change that stages V as bf16 -- which is what a
// `cp.async` byte copy of V would require -- silently drops P from fp16's 11
// mantissa bits to bf16's 8. The file header records that fp16 was chosen over
// bf16 deliberately, for exactly those bits. Nothing was checking.
//
// **The bar is 1.10x and it is chosen, not inherited.** A host model of this
// schedule puts fp16 P at 1.00-1.01x the bf16 output floor and bf16 P at
// 1.21-1.27x, flat in K depth from 512 to 32768 -- signal and error both grow
// as sqrt(K), so it does not compound. 1.10x sits between them. Two things make
// that model trustworthy enough to set a bar from: it reproduces the 1.66e-3
// bf16 output floor and the 1.01x fp16 ratio that `attention_fp16_score_tile`
// measured independently, neither of which it was fitted to.
//
// At *this* test's amplitudes the same model puts fp16 P at 1.006x and bf16 P
// at 1.307x, so the two figures to hold in mind are 1.01x and ~1.3x.
//
// Reusing the blocked path's 1.3x would have been the wrong move, and the
// reason is sharper than "it is too loose": bf16 P lands at 1.307x here and at
// 1.21-1.27x at other amplitudes, so it sits *astride* that bar and which side
// it falls depends on the data. A bar a regression clears or misses by seven
// thousandths is not evidence either way -- it is a coin flip wearing whichever
// label it lands on, and a green one ends the conversation. 1.10x is chosen to
// sit clear of both figures: fp16 has ~9 points of headroom, bf16 ~20 points of
// exceedance, and neither depends on the draw.
//
// That test's own comment says its 1.3x "has no power to separate fp16
// probabilities from bf16 ones, because neither is visible through a bf16
// output". The first half is right about its assertion. The second half is too
// strong: the *measurement* separates them cleanly, 1.31x against 1.01x. Both
// comments are left standing because they are about different things.
VIDFAB_TEST(attention_fused_probability_precision) {
  CublasScope cb;
  const int seq = 1024;
  const int heads = 4;
  const int head_dim = 128;
  const int width = heads * head_dim;

  // Same amplitudes as the blocked path's test, so the two ratios are directly
  // comparable: scores at a few units, the order post-q_norm/k_norm H3 produces.
  const std::vector<float> q = bf16_round(make_data(size_t(seq) * width, 181u, 0.5f));
  const std::vector<float> k = bf16_round(make_data(size_t(seq) * width, 182u, 0.5f));
  const std::vector<float> v = bf16_round(make_data(size_t(seq) * width, 183u, 1.0f));
  BfBuf dq(q), dk(k), dv(v), dout(size_t(seq) * width);

  vidfab::cuda::AttentionConfig cfg;
  cfg.seq_len = seq;
  cfg.num_heads = heads;
  cfg.head_dim = head_dim;
  CHECK(vidfab::cuda::attention_preferred_backend(cfg) == vidfab::cuda::AttentionBackend::kFused);

  const std::vector<float> want =
      cpu_attention(q, k, v, seq, heads, heads, head_dim, cfg.effective_scale());

  double norm_sq = 0.0;
  for (double x : want) norm_sq += x * x;
  const double want_rms = std::sqrt(norm_sq / double(want.size()));

  // The floor: what rounding the exact answer to the output's own format costs.
  // Error below this is invisible by construction, whatever produced it.
  const std::vector<float> rounded = bf16_round(want);
  double floor_sq = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double d = double(rounded[i]) - want[i];
    floor_sq += d * d;
  }
  const double output_floor_rel = std::sqrt(floor_sq / double(want.size())) / want_rms;

  Workspace ws;
  vidfab::cuda::attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                  vidfab::cuda::AttentionBackend::kFused, ws);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  const std::vector<float> got = dout.host();

  double err_sq = 0.0;
  double worst_abs = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = double(got[i]) - want[i];
    err_sq += d * d;
    worst_abs = std::max(worst_abs, std::fabs(d));
  }
  const double rms_rel = std::sqrt(err_sq / double(got.size())) / want_rms;

  std::printf("  fused P precision: rms_rel %.3g (bf16 output floor %.3g, ratio %.2f), "
              "worst abs %.3g\n",
              rms_rel, output_floor_rel, rms_rel / output_floor_rel, worst_abs);
  CHECK_MSG(rms_rel < 1.10 * output_floor_rel,
            "fused probabilities: rms_rel %.4g is %.2fx the %.4g bf16 output floor, over the 1.10x "
            "bar; bf16 probabilities model at 1.21-1.27x",
            rms_rel, rms_rel / output_floor_rel, output_floor_rel);
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

// --- operands for a timing kernel -------------------------------------------
//
// **Constant operands are not a neutral choice, and this benchmark used to make
// it.** A `cudaMemset` fills a buffer with one repeated byte, so a GEMM over two
// such buffers multiplies the same pair of 16-bit values in every lane on every
// cycle and the tensor-core multiplier array barely toggles. Dynamic power *is*
// switching activity, so the card stops drawing its 575 W limit and boosts to
// the top of its V/F curve — measured here at 2865 MHz against the ~2.5 GHz it
// holds on real data, which is worth about 14% of throughput.
//
// That is enough to have put the reported figures **above this card's own
// sustained bf16 ceiling** (216-222 TFLOP/s): the four linears read 234-237
// TFLOP/s in the table this file feeds, a physical impossibility that sat in
// README.md unnoticed while the same table declared the ceiling two lines above.
//
// So the operands are filled with pseudo-random bits instead. The fill runs on
// the device because the largest of these buffers is 1.08 GB and staging it
// through the host would cost more than the measurement.
__global__ void fill_bf16_kernel(uint16_t* __restrict__ dst, size_t n, uint32_t seed) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t s = static_cast<uint32_t>(i) * 2654435761u + seed;
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  // Sign and all seven mantissa bits random; exponent in [126, 128], i.e. |x| in
  // [0.5, 4). Broad enough to toggle the datapath, tight enough that neither a
  // K = 14336 dot product nor a 37710-key softmax can leave range.
  const uint32_t exp = 126u + (s >> 28) % 3u;
  dst[i] = static_cast<uint16_t>((s & 0x8000u) | (exp << 7) | ((s >> 8) & 0x7Fu));
}

// e4m3 codes. The exponent is kept near the format's bias of 7 and the two NaN
// patterns (|code| == 0x7F) are unreachable by construction rather than masked
// out afterwards, so no fill can turn a timing run into a NaN propagation study.
__global__ void fill_f8_kernel(uint8_t* __restrict__ dst, size_t n, uint32_t seed) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t s = static_cast<uint32_t>(i) * 2246822519u + seed;
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  const uint32_t e = 5u + (s >> 27) % 5u;  // 2^-2 .. 2^2 before the tensor scale
  dst[i] = static_cast<uint8_t>((s & 0x80u) | (e << 3) | ((s >> 4) & 0x07u));
}

void fill_random(BfBuf& b, uint32_t seed) {
  const size_t n = b.raw.size();
  fill_bf16_kernel<<<static_cast<int>((n + 255) / 256), 256>>>(b.raw.get(), n, seed);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

void fill_random_f8(DeviceBuffer<uint8_t>& b, uint32_t seed) {
  const size_t n = b.size();
  fill_f8_kernel<<<static_cast<int>((n + 255) / 256), 256>>>(b.get(), n, seed);
  VIDFAB_CUDA_CHECK(cudaGetLastError());
}

// The dequantiser runs once per GEMM on every one of the 200 quantised linears,
// fifty times a step, so its cost is not incidental. It is pure streaming and
// should sit near the card's bandwidth: 2 bytes written and 9/16 read per
// element, of which the store is the overwhelming majority.
//
// This one keeps its `cudaMemset` fills deliberately: it is bound by how many
// bytes cross the memory system, not by multiplier switching, and it is the
// control that says so.
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
    // Random, not memset: see fill_bf16_kernel. Constant q and k also make every
    // score identical, so the online softmax never rescales and its running
    // maximum never moves — the kernel would be measured on the one input that
    // exercises none of its data-dependent work.
    fill_random(q, 0x5EEDu);
    fill_random(k, 0xA17Eu);
    fill_random(v, 0xC0FFu);

    vidfab::cuda::AttentionConfig cfg;
    cfg.seq_len = seq;
    cfg.num_heads = heads;
    cfg.head_dim = head_dim;
    cfg.query_block = 1024;
    // The production shape must dispatch to the fused kernel on its own, not
    // only when a benchmark names it. A silent fall back to the blocked path
    // would still produce correct output and a believable number -- it is only
    // visible if something asserts which backend the pipeline would pick.
    CHECK(vidfab::cuda::attention_preferred_backend(cfg) == vidfab::cuda::AttentionBackend::kFused);

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

    // Independent acceptance measurement: conversion/smoothing and the
    // quantized attention kernel are one timed operation.
    const std::vector<uint16_t> flash_bits = out.bits();
    const size_t sage_ws_bytes = vidfab::cuda::attention_workspace_bytes(
        cfg, vidfab::cuda::AttentionBackend::kSage2);
    Workspace sage_ws;
    sage_ws.reserve(sage_ws_bytes);
    float sms = 1e30f;
    for (int pass = 0; pass < 3; ++pass) {
      sms = std::min(sms, timer.measure(
          [&] { vidfab::cuda::attention_forward(cb.h, nullptr, q.p(), k.p(), v.p(), out.p(), cfg,
                                                vidfab::cuda::AttentionBackend::kSage2, sage_ws); },
          1, 3));
    }
    const std::vector<uint16_t> sage_bits = out.bits();
    vidfab::cuda::attention_forward(cb.h, nullptr, q.p(), k.p(), v.p(), out.p(), cfg,
                                    vidfab::cuda::AttentionBackend::kSage2, sage_ws);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<uint16_t> sage_repeat = out.bits();
    double err2 = 0.0, ref2 = 0.0, dot = 0.0, got2 = 0.0, max_abs = 0.0;
    size_t mismatches = 0;
    for (size_t i = 0; i < flash_bits.size(); ++i) {
      const double a = vidfab::bf16_to_f32(flash_bits[i]);
      const double b = vidfab::bf16_to_f32(sage_bits[i]);
      const double e = b - a;
      err2 += e * e; ref2 += a * a; dot += a * b; got2 += b * b;
      max_abs = std::max(max_abs, std::abs(e));
      mismatches += sage_bits[i] != sage_repeat[i];
    }
    std::printf("  attention   sage2 (all conversions)             %8.2f ms  (%.2fx flash2)\n",
                sms, fms / sms);
    std::printf("  sage2 workspace %.3f GiB, rel_L2 %.6f corr %.6f max_abs %.6g repeat_mismatch %zu\n",
                double(sage_ws_bytes) / (1 << 30), std::sqrt(err2 / ref2),
                dot / std::sqrt(ref2 * got2), max_abs, mismatches);
    CHECK(sms > 0.0f);
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
    fill_random_f8(raw, 0x1234u);
    const std::vector<float> sv(1, 8.1264e-3f);
    DeviceBuffer<float> dscale = to_device(sv);
    BfBuf dx(xn), dy(yn);
    fill_random(dx, 0x9ABCu);

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

// --- native nvfp4 GEMM ------------------------------------------------------
//
// Everything from here to the end of the file belongs to the native
// block-scaled nvfp4 GEMM in src/cuda/nvfp4_gemm.cu. It is one block so that
// the three tracks working on this file merge cleanly.
//
// Three separable things can be wrong and any two of them can cancel: the
// register operand layout (pinned by `nvfp4_mma_operand_layout`), the on-disk
// convention (pinned by `nn_dequant_nvfp4`), and the dynamic activation
// quantisation, which is this project's own choice and belongs to no
// checkpoint. Each is checked on its own here.
//
// The last of the three is not free and is not a bug. `nvfp4_activation_cost`
// measures it, and the reason it is measured rather than asserted is written
// out there.

// The e4m3 and e2m1 encoders, by brute force over the representable set. A
// reference exists to be obviously right; round-to-nearest-*even* on a
// four-bit exponent is easy to get subtly wrong by hand, and a reference that
// shares a bug with the thing it checks proves nothing.
// `nvfp4_rounding_reference` pins both against the hardware converters the
// kernel actually issues, before anything is built on them.
uint8_t host_e4m3(float v) {
  uint8_t best = 0;
  double bd = 1e300;
  for (int c = 0; c < 0x7F; ++c) {  // non-negative only; 0x7F is NaN
    const double d = std::fabs(double(vidfab::f8_e4m3_to_f32(uint8_t(c))) - double(v));
    if (d < bd || (d == bd && (c & 1) == 0)) {
      bd = d;
      best = static_cast<uint8_t>(c);
    }
  }
  return best;
}

// The sign is carried, not searched for. Rounding over all sixteen codes at
// once makes +0.1 a tie between +0.0 and -0.0, and which of the two comes back
// is a question about the sign of zero rather than about rounding -- the
// hardware keeps the input's sign, so this does too. It is the only place the
// two ever disagreed, over 0 of 270 *values* and 8 of 270 codes.
uint8_t host_e2m1(float v) {
  uint8_t mag = 0;
  double bd = 1e300;
  for (int c = 0; c < 8; ++c) {
    const double d = std::fabs(double(vidfab::f4_e2m1_to_f32(uint8_t(c))) - std::fabs(double(v)));
    if (d < bd || (d == bd && (c & 1) == 0)) {
      bd = d;
      mag = static_cast<uint8_t>(c);
    }
  }
  return static_cast<uint8_t>(mag | (std::signbit(v) ? 8u : 0u));
}

// A dense matrix put into the checkpoint's storage form, plus exactly what
// those bytes mean. `make_nvfp4` above starts from random codes, which is the
// right shape for a layout test; this starts from real values, which is what a
// numerical one needs.
//
// `dense` is the reference the GEMM is judged against. It is not the input
// matrix: it is the input matrix after nvfp4 has had its way with it.
struct NvfpPacked {
  std::vector<uint8_t> data;
  std::vector<uint8_t> scale;
  std::vector<float> dense;
};

// `high_even` and `swizzled` select the on-disk convention. Both shipped
// checkpoints are (true, true); the other three exist so a test can show the
// kernel tells them apart rather than happening to agree on symmetric data.
NvfpPacked pack_nvfp4(const std::vector<float>& w, int rows, int cols, float global,
                      bool high_even, bool swizzled) {
  const int kb = cols / 16;
  NvfpPacked t;
  t.data.assign(size_t(rows) * cols / 2, 0);
  t.scale.assign(size_t(rows) * kb, 0);
  t.dense.assign(size_t(rows) * cols, 0.0f);
  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < kb; ++b) {
      float amax = 0.0f;
      for (int i = 0; i < 16; ++i) {
        amax = std::max(amax, std::fabs(w[size_t(r) * cols + b * 16 + i]));
      }
      const uint8_t s8 = host_e4m3(amax / 6.0f / global);
      const float sd = vidfab::f8_e4m3_to_f32(s8) * global;
      const float inv = sd > 0.0f ? 1.0f / sd : 0.0f;
      t.scale[swizzled ? nvfp4_scale_slot(r, b, kb) : size_t(r) * kb + b] = s8;
      for (int i = 0; i < 16; ++i) {
        const int col = b * 16 + i;
        const size_t flat = size_t(r) * cols + col;
        const uint8_t q = host_e2m1(w[flat] * inv);
        t.data[flat / 2] |= static_cast<uint8_t>(q << (((col % 2 == 0) == high_even) ? 4 : 0));
        t.dense[flat] = vidfab::f4_e2m1_to_f32(q) * sd;
      }
    }
  }
  return t;
}

// The kernel's activation rule restated: amax/6 rounded to e4m3, then the
// elements divided by the *decoded* scale and rounded to e2m1. In float rather
// than double so the ties fall the same way.
std::vector<float> host_quantise_act(const std::vector<float>& x, int rows, int dim) {
  std::vector<float> out(x.size(), 0.0f);
  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < dim / 16; ++b) {
      float amax = 0.0f;
      for (int i = 0; i < 16; ++i) {
        amax = std::max(amax, std::fabs(x[size_t(r) * dim + b * 16 + i]));
      }
      const uint8_t s8 = host_e4m3(amax * (1.0f / 6.0f));
      const float sd = vidfab::f8_e4m3_to_f32(s8);
      const float inv = sd > 0.0f ? 1.0f / sd : 0.0f;
      for (int i = 0; i < 16; ++i) {
        const size_t flat = size_t(r) * dim + b * 16 + i;
        out[flat] = vidfab::f4_e2m1_to_f32(host_e2m1(x[flat] * inv)) * sd;
      }
    }
  }
  return out;
}

// Box-Muller over the harness's own generator, because post-norm activations
// are Gaussian-ish and the harness's uniform is not.
//
// The reason is realism, not pessimism, and the naive argument for it is
// backwards. One would expect uniform data to flatter a block-scaled format --
// every element sits near its own block maximum, so a shared scale wastes
// nothing. Measured, uniform comes out *worse*: 0.101 against Gaussian's
// 0.095 (`nvfp4_activation_cost`). E2M1's grid is finer near zero, with a step
// of 0.5 below 2 and of 2 above 4, so a distribution that puts most of its
// mass well inside its own maximum is the one the format suits.
std::vector<float> make_gaussian(size_t n, uint32_t seed, float sigma) {
  const std::vector<float> u = make_data(2 * n, seed, 0.5f);  // (-0.5, 0.5)
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) {
    const float a = std::max(1e-7f, u[2 * i] + 0.5f);
    v[i] = sigma * std::sqrt(-2.0f * std::log(a)) * std::cos(6.2831853f * (u[2 * i + 1] + 0.5f));
  }
  return v;
}

double rms_rel(const std::vector<float>& want, const std::vector<float>& got) {
  double num = 0.0;
  double den = 0.0;
  for (size_t i = 0; i < want.size(); ++i) {
    const double d = double(got[i]) - want[i];
    num += d * d;
    den += double(want[i]) * want[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : 0.0;
}

// Pearson correlation. The discriminator between quantisation noise and a
// layout bug: noise leaves this at 0.99-something, a misread operand collapses
// it towards zero while leaving every summary statistic looking healthy.
double correlation(const std::vector<float>& a, const std::vector<float>& b) {
  double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
  const double n = double(a.size());
  for (size_t i = 0; i < a.size(); ++i) {
    sa += a[i];
    sb += b[i];
    saa += double(a[i]) * a[i];
    sbb += double(b[i]) * b[i];
    sab += double(a[i]) * b[i];
  }
  const double cov = sab / n - (sa / n) * (sb / n);
  const double va = saa / n - (sa / n) * (sa / n);
  const double vb = sbb / n - (sb / n) * (sb / n);
  return (va > 0 && vb > 0) ? cov / std::sqrt(va * vb) : 0.0;
}

std::vector<float> run_native_nvfp4(const std::vector<float>& x, const NvfpPacked& w, int rows,
                                    int out_features, int in_features, float global) {
  BfBuf dx(x), dy(size_t(rows) * out_features);
  DeviceBuffer<uint8_t> dw(w.data.size()), dws(w.scale.size());
  dw.copy_from_host(w.data.data(), w.data.size());
  dws.copy_from_host(w.scale.data(), w.scale.size());
  Workspace ws;
  ws.reserve(vidfab::cuda::nvfp4_gemm_workspace_bytes(rows, in_features) + 4096);
  vidfab::cuda::nvfp4_gemm_forward(dx.p(), dw.get(), dws.get(), global, dy.p(), rows, out_features,
                                   in_features, ws, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  return dy.host();
}

// --- the B-side block-scale operand -----------------------------------------

__global__ void nvfp4_mma_bscale_kernel(const uint32_t* a, const uint32_t* b, const uint32_t* sa,
                                        const uint32_t* sb, float* out) {
  const int lane = threadIdx.x;
  const uint32_t ra[4] = {a[lane * 4], a[lane * 4 + 1], a[lane * 4 + 2], a[lane * 4 + 3]};
  const uint32_t rb[2] = {b[lane * 2], b[lane * 2 + 1]};
  const uint32_t s_a = sa[lane];
  const uint32_t s_b = sb[lane];
  float c[4] = {0, 0, 0, 0};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 1200
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X"
      ".f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{0,0},{%11},{0,0};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]), "r"(rb[0]), "r"(rb[1]), "r"(s_a),
        "r"(s_b));
#endif
  const int gid = lane >> 2, tig = lane & 3;
  out[gid * 8 + tig * 2] = c[0];
  out[gid * 8 + tig * 2 + 1] = c[1];
  out[(gid + 8) * 8 + tig * 2] = c[2];
  out[(gid + 8) * 8 + tig * 2 + 1] = c[3];
}

// `nvfp4_mma_operand_layout` pins the A side: row r's four block scales are the
// four bytes of lane `r < 8 ? 4r : 4(r-8)+1`. The B side needs its own
// experiment and does not follow by symmetry. A has sixteen rows and uses
// sixteen lanes; B has eight columns and uses **eight** — column c's scales are
// the four bytes of lane 4c, and lanes 4c+1..4c+3 carry nothing at all.
//
// Halving the A rule instead — putting columns 0-3 on lanes 4g and 4-7 on lanes
// 4g+1, which is the shape one reaches for — writes half the scales into lanes
// the instruction ignores. It does not crash and does not produce zeros. It
// produces a well-formed matrix with four of its eight columns scaled wrong.
VIDFAB_TEST(nvfp4_mma_b_scale_operand_layout) {
  REQUIRE_SM120_TEST("NVFP4 MMA block-scale test");
  const std::vector<uint32_t> ones(32 * 4, 0x22222222u), onesb(32 * 2, 0x22222222u);
  DeviceBuffer<uint32_t> da(ones.size()), db(onesb.size()), dsa(32), dsb(32);
  da.copy_from_host(ones.data(), ones.size());
  db.copy_from_host(onesb.data(), onesb.size());
  const std::vector<uint32_t> unit(32, 0x38383838u);  // four e4m3 1.0 scales
  dsa.copy_from_host(unit.data(), unit.size());
  DeviceBuffer<float> dout(128);

  // Every element and every scale 1.0, so each output is 4 blocks x 16 = 64.
  dsb.copy_from_host(unit.data(), unit.size());
  nvfp4_mma_bscale_kernel<<<1, 32>>>(da.get(), db.get(), dsa.get(), dsb.get(), dout.get());
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  CHECK_CLOSE(std::vector<float>(128, 64.0f), to_host(dout), 0.0, "nvfp4 B unit scales");

  int live_lanes = 0;
  for (int lane = 0; lane < 32; ++lane) {
    for (int blk : {0, 2, 3}) {
      std::vector<uint32_t> s(32, 0x38383838u);
      s[lane] = (s[lane] & ~(0xFFu << (8 * blk))) | (uint32_t(0x40) << (8 * blk));  // e4m3 2.0
      dsb.copy_from_host(s.data(), s.size());
      nvfp4_mma_bscale_kernel<<<1, 32>>>(da.get(), db.get(), dsa.get(), dsb.get(), dout.get());
      VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
      const std::vector<float> g = to_host(dout);
      const int live_col = (lane % 4 == 0) ? lane / 4 : -1;
      bool ok = true;
      for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 8; ++c) {
          ok = ok && std::fabs(g[r * 8 + c] - (c == live_col ? 80.0f : 64.0f)) < 1e-3;
        }
      }
      if (live_col >= 0 && blk == 0) ++live_lanes;
      CHECK_MSG(ok, "B scale lane %d byte %d: expected %s", lane, blk,
                live_col >= 0 ? "its own column lifted by one block" : "no effect at all");
    }
  }
  CHECK_MSG(live_lanes == 8, "exactly eight B scale lanes should be live, saw %d", live_lanes);
}

// --- rounding ---------------------------------------------------------------

__global__ void nvfp4_cvt_probe_kernel(const float* in, uint8_t* e2m1, uint8_t* e4m3, int pairs) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pairs) return;
  e2m1[i] = static_cast<uint8_t>(__nv_cvt_float2_to_fp4x2(make_float2(in[i * 2], in[i * 2 + 1]),
                                                          __NV_E2M1, cudaRoundNearest));
  e4m3[i] = __nv_cvt_float_to_fp8(in[i * 2], __NV_SATFINITE, __NV_E4M3);
}

// The awkward inputs are the ties — 0.25, 0.75, 1.75, 3.5, 5.0 — where
// round-to-nearest-even and round-half-away-from-zero disagree, and the
// saturating end, where e2m1 must clamp to +-6 rather than wrap.
VIDFAB_TEST(nvfp4_rounding_reference) {
  std::vector<float> vals = {0.0f, 0.25f, 0.75f, 1.25f, 1.75f, 2.5f,
                             3.5f, 5.0f,  6.0f,  1e4f,  -1e4f, 0.0f};
  const std::vector<float> r = make_data(512, 909u, 8.0f);
  vals.insert(vals.end(), r.begin(), r.end());

  DeviceBuffer<float> din(vals.size());
  din.copy_from_host(vals.data(), vals.size());
  const int pairs = int(vals.size() / 2);
  DeviceBuffer<uint8_t> d4(pairs), d8(pairs);
  nvfp4_cvt_probe_kernel<<<(pairs + 127) / 128, 128>>>(din.get(), d4.get(), d8.get(), pairs);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint8_t> h4(pairs), h8(pairs);
  d4.copy_to_host(h4.data(), h4.size());
  d8.copy_to_host(h8.data(), h8.size());

  int bad4 = 0, bad8 = 0;
  for (int i = 0; i < pairs; ++i) {
    // The instruction packs the even index in the LOW nibble. That the
    // checkpoint does the opposite is a separate statement about a separate
    // layer, and conflating the two is exactly the trap this file exists for.
    if ((h4[i] & 0x0F) != host_e2m1(vals[i * 2])) ++bad4;
    if ((h4[i] >> 4) != host_e2m1(vals[i * 2 + 1])) ++bad4;
    if (vals[i * 2] >= 0.0f && h8[i] != host_e4m3(vals[i * 2])) ++bad8;
  }
  CHECK_MSG(bad4 == 0, "host e2m1 encoder disagrees with cvt.rn.satfinite on %d of %d", bad4,
            pairs * 2);
  CHECK_MSG(bad8 == 0, "host e4m3 encoder disagrees with cvt.rn.satfinite on %d of %d", bad8,
            pairs);
}

// --- activation quantisation ------------------------------------------------

VIDFAB_TEST(nvfp4_activation_quantisation) {
  const int rows = 7;
  const int dim = 64;
  std::vector<float> x = make_gaussian(size_t(rows) * dim, 4242u, 0.7f);
  // Three blocks with a story: all zeros, far under e4m3's smallest scale, and
  // far over its largest. None may produce a NaN and none may wrap.
  for (int i = 0; i < 16; ++i) x[0 * dim + i] = 0.0f;
  for (int i = 0; i < 16; ++i) x[1 * dim + 16 + i] = 1e-6f * float(i + 1);
  for (int i = 0; i < 16; ++i) x[2 * dim + 32 + i] = 5000.0f;
  const std::vector<float> xr = bf16_round(x);

  BfBuf dx(x);
  DeviceBuffer<uint8_t> dq(size_t(rows) * dim / 2), ds(size_t(rows) * dim / 16);
  vidfab::cuda::launch_quantize_nvfp4_activations(dx.p(), dq.get(), ds.get(), rows, dim, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint8_t> hq(dq.size()), hs(ds.size());
  dq.copy_to_host(hq.data(), hq.size());
  ds.copy_to_host(hs.data(), hs.size());

  // The activation buffer is written by this project, for this instruction, so
  // it owes the checkpoint's conventions nothing: low nibble is the even
  // element and the scales are plain row-major.
  int bad_scale = 0, bad_nibble = 0;
  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < dim / 16; ++b) {
      float amax = 0.0f;
      for (int i = 0; i < 16; ++i) {
        amax = std::max(amax, std::fabs(xr[size_t(r) * dim + b * 16 + i]));
      }
      const uint8_t s8 = host_e4m3(amax * (1.0f / 6.0f));
      if (hs[size_t(r) * (dim / 16) + b] != s8) ++bad_scale;
      const float sd = vidfab::f8_e4m3_to_f32(s8);
      const float inv = sd > 0.0f ? 1.0f / sd : 0.0f;
      for (int i = 0; i < 16; ++i) {
        const size_t flat = size_t(r) * dim + b * 16 + i;
        const uint8_t got = (i % 2 == 0) ? (hq[flat / 2] & 0x0F) : (hq[flat / 2] >> 4);
        if (host_e2m1(xr[flat] * inv) != got) ++bad_nibble;
      }
    }
  }
  CHECK_MSG(bad_scale == 0, "%d of %d block scales are not amax/6 rounded to e4m3", bad_scale,
            int(hs.size()));
  CHECK_MSG(bad_nibble == 0, "%d of %d nibbles differ from the host rule", bad_nibble,
            int(xr.size()));

  // Zero block: scale zero, nibbles zero, and no NaN out of the reciprocal.
  CHECK(hs[0] == 0);
  bool zeros = true;
  for (int i = 0; i < 8; ++i) zeros = zeros && hq[i] == 0;
  CHECK(zeros);

  // Underflow flushes the whole block rather than clipping it onto the grid.
  // e4m3's smallest positive is 2^-9, so a block whose largest element is under
  // 6 * 2^-10 has no representable scale and becomes zero. Documented, not
  // accidental: it is why the header argues a per-tensor activation scale buys
  // nothing for post-norm activations, whose blocks are nowhere near this.
  CHECK(hs[dim / 16 + 1] == 0);

  // Overflow saturates. 5000 wants a scale of 833, e4m3 stops at 448, and the
  // elements then clip at 6 — finite and too small, never wrapped, never NaN.
  CHECK(vidfab::f8_e4m3_to_f32(hs[2 * (dim / 16) + 2]) == 448.0f);
  bool clipped = true;
  for (int i = 0; i < 8; ++i) {
    const uint8_t byte = hq[(size_t(2) * dim + 32) / 2 + i];
    clipped = clipped && (byte & 0x0F) == 7 && (byte >> 4) == 7;  // +6 both halves
  }
  CHECK(clipped);
}

// --- the GEMM against a CPU reference ---------------------------------------

VIDFAB_TEST(nvfp4_gemm_matches_cpu_reference) {
  REQUIRE_SM120_TEST("native NVFP4 GEMM");
  struct Shape {
    int rows, out, in;
  };
  // Ragged row counts on purpose: 1 leaves 127 rows of a tile as padding, 200
  // leaves a 72-row tail, 129 leaves a one-row second tile. The contraction
  // covers half a staging tile (64), one (128), one and a half (192) and two
  // and a half (320).
  const Shape shapes[] = {{1, 128, 64},    {17, 128, 128}, {128, 256, 192},
                          {200, 128, 320}, {129, 256, 64}};

  for (const Shape& s : shapes) {
    const std::vector<float> x = bf16_round(make_gaussian(size_t(s.rows) * s.in, 71u + s.in, 0.8f));
    const std::vector<float> wd = make_gaussian(size_t(s.out) * s.in, 33u + s.out, 0.05f);
    const NvfpPacked w = pack_nvfp4(wd, s.out, s.in, 0.7f, true, true);
    // The reference sees the same activation the kernel does. This check is
    // about the GEMM; what 4-bit activations cost is a different question,
    // measured in `nvfp4_activation_cost`.
    CHECK_CLOSE_REL(cpu_matmul_nt(host_quantise_act(x, s.rows, s.in), w.dense, s.rows, s.out, s.in),
                    run_native_nvfp4(x, w, s.rows, s.out, s.in, 0.7f), 1e-3, 1e-2,
                    "native nvfp4 GEMM vs CPU reference");
  }
}

// The load path from stored bytes to mma registers is not the identity, and
// every way of getting it wrong is silent. Each wrong form is constructed here
// and the kernel required not to match it.
VIDFAB_TEST(nvfp4_gemm_disk_layout) {
  REQUIRE_SM120_TEST("native NVFP4 GEMM");
  const int rows = 64, out = 128, in = 192;
  const std::vector<float> x = bf16_round(make_gaussian(size_t(rows) * in, 515u, 0.8f));
  const std::vector<float> wd = make_gaussian(size_t(out) * in, 616u, 0.05f);
  const NvfpPacked right = pack_nvfp4(wd, out, in, 1.0f, true, true);
  const std::vector<float> want =
      cpu_matmul_nt(host_quantise_act(x, rows, in), right.dense, rows, out, in);

  CHECK_CLOSE_REL(want, run_native_nvfp4(x, right, rows, out, in, 1.0f), 1e-3, 1e-2,
                  "checkpoint layout: high nibble even, block scales swizzled");

  const struct {
    bool high_even, swizzled;
    const char* what;
  } wrong[] = {
      {false, true, "low nibble even"},
      {true, false, "block scales row-major rather than 128x4 tiled"},
      {false, false, "both conventions inverted"},
  };
  for (const auto& c : wrong) {
    const NvfpPacked bad = pack_nvfp4(wd, out, in, 1.0f, c.high_even, c.swizzled);
    const double rel = rms_rel(want, run_native_nvfp4(x, bad, rows, out, in, 1.0f));
    CHECK_MSG(rel > 0.2, "%s must disagree with the checkpoint layout, rms_rel %.4f", c.what, rel);
  }

  // A block stride error inside an otherwise correct swizzle: every row's block
  // scales rotated by one. Every byte is still present and still in the right
  // tile, so nothing about the size or the value histogram gives it away.
  NvfpPacked rot = right;
  const int kb = in / 16;
  for (int m = 0; m < out; ++m) {
    for (int b = 0; b < kb; ++b) {
      rot.scale[nvfp4_scale_slot(m, b, kb)] = right.scale[nvfp4_scale_slot(m, (b + 1) % kb, kb)];
    }
  }
  const double rel = rms_rel(want, run_native_nvfp4(x, rot, rows, out, in, 1.0f));
  CHECK_MSG(rel > 0.2, "block scales rotated by one block must disagree, rms_rel %.4f", rel);

  // The shapes the 128x4 tiling cannot address are refused, not guessed at.
  CHECK(vidfab::cuda::nvfp4_gemm_supported(out, in));
  CHECK(!vidfab::cuda::nvfp4_gemm_supported(out, in + 16));   // in % 64 != 0
  CHECK(!vidfab::cuda::nvfp4_gemm_supported(out + 64, in));   // out % 128 != 0
}

// The global scale multiplies the whole tensor and is folded into the epilogue,
// so it has to appear exactly once. Twice, or not at all, still gives a
// well-formed matrix.
VIDFAB_TEST(nvfp4_gemm_global_scale) {
  REQUIRE_SM120_TEST("native NVFP4 GEMM");
  const int rows = 32, out = 128, in = 128;
  const std::vector<float> x = bf16_round(make_gaussian(size_t(rows) * in, 808u, 0.8f));
  const std::vector<float> wd = make_gaussian(size_t(out) * in, 909u, 0.05f);
  const NvfpPacked w = pack_nvfp4(wd, out, in, 1.0f, true, true);

  // The same stored bytes twice, with only the scalar changed, so anything the
  // quantiser does cancels and what is left is the epilogue's multiply.
  const std::vector<float> at_one = run_native_nvfp4(x, w, rows, out, in, 1.0f);
  const float g = 3.25f;
  const std::vector<float> at_g = run_native_nvfp4(x, w, rows, out, in, g);

  std::vector<float> lifted(at_one.size()), twice(at_one.size());
  for (size_t i = 0; i < at_one.size(); ++i) {
    lifted[i] = at_one[i] * g;
    twice[i] = at_one[i] * g * g;
  }
  // Not equality, and the reason is worth stating because the tempting
  // argument is wrong: g has three significant bits, but `at_one` is already
  // rounded to bf16, so scaling it multiplies a half-ULP error by 3.25 before
  // `at_g`'s own rounding is added. Two bf16 ULP apart is correct behaviour.
  // The bar still has all the power it needs: getting the count wrong moves
  // the answer by a factor of 3.25.
  CHECK_CLOSE_REL(lifted, at_g, 1e-3, 1e-2, "global scale applied exactly once");
  CHECK(rms_rel(at_one, at_g) > 0.5);  // not zero times
  CHECK(rms_rel(twice, at_g) > 0.5);   // not twice

  // And the scalar the checkpoint actually carries reaches the same answer
  // whether it is folded into the stored scales or passed alongside them.
  const NvfpPacked folded = pack_nvfp4(wd, out, in, g, true, true);
  CHECK_CLOSE_REL(cpu_matmul_nt(host_quantise_act(x, rows, in), folded.dense, rows, out, in),
                  run_native_nvfp4(x, folded, rows, out, in, g), 1e-3, 1e-2,
                  "global scale against a weight packed for it");
}

// --- what the operand path costs, separated from what the format costs -------

// The decisive experiment. Every activation here is already exactly on the fp4
// grid: each block is built from a power-of-two scale and the eight E2M1
// magnitudes, with at least one element at 6s so `amax/6` recovers `s` exactly.
// The kernel's dynamic quantiser is therefore the identity on this input, and
// what remains is the operand path alone.
//
// If this agrees and `nvfp4_activation_cost` does not, the activation
// quantisation is the entire story and the load path is correct. That is the
// one measurement that tells a numerical limit apart from a layout bug.
VIDFAB_TEST(nvfp4_gemm_exact_fp4_activations) {
  REQUIRE_SM120_TEST("native NVFP4 GEMM");
  const int rows = 128, out = 256, in = 512;
  const float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

  std::vector<float> x(size_t(rows) * in);
  const std::vector<float> u = make_data(x.size(), 31337u, 0.5f);
  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < in / 16; ++b) {
      // Powers of two from 2^-4 to 2^3, all exactly e4m3.
      const float s = std::ldexp(1.0f, ((r * 7 + b * 3) % 8) - 4);
      for (int i = 0; i < 16; ++i) {
        const size_t flat = size_t(r) * in + b * 16 + i;
        const int code = int((u[flat] + 0.5f) * 8.0f) & 7;
        const float sign = (int((u[flat] + 0.5f) * 64.0f) & 1) ? -1.0f : 1.0f;
        x[flat] = sign * grid[code] * s;
      }
      x[size_t(r) * in + b * 16] = 6.0f * s;  // pin amax so the scale round-trips
    }
  }
  // Every value is a small multiple of a power of two, so bf16 holds it exactly
  // and the buffer the kernel reads is the buffer built here.
  CHECK(bf16_round(x) == x);
  CHECK(host_quantise_act(x, rows, in) == x);

  const std::vector<float> wd = make_gaussian(size_t(out) * in, 2468u, 0.05f);
  const NvfpPacked w = pack_nvfp4(wd, out, in, 1.0f, true, true);
  const std::vector<float> got = run_native_nvfp4(x, w, rows, out, in, 1.0f);
  const std::vector<float> want = cpu_matmul_nt(x, w.dense, rows, out, in);

  std::printf("  nvfp4 exact-fp4 activations: rms_rel %.6f  corr %.6f\n", rms_rel(want, got),
              correlation(want, got));
  CHECK_CLOSE_REL(want, got, 1e-3, 1e-2, "native nvfp4 GEMM on fp4-exact activations");
}

// The measurement, not an assertion.
//
// `set_native(true)` replaces bf16 activations with 4-bit ones, and that is a
// change to the arithmetic, not to the implementation of it. E2M1 has eight
// magnitudes; a 16-element block sharing one scale carries a per-element
// relative error of order 10%, and a dot product does not average it away —
// both the signal and the error grow as sqrt(K), so the output's relative error
// stays where the input's was. Nothing about the kernel changes that and no
// value of K rescues it, which is what the sweep below is for.
//
// So this reports rather than asserts. `nvfp4_gemm_exact_fp4_activations` is
// the assertion that the operand path is right; this is the cost of the format
// on top of it, and whether that cost is acceptable is a modelling decision.
VIDFAB_TEST(nvfp4_activation_cost) {
  REQUIRE_SM120_TEST("native NVFP4 GEMM");
  const int rows = 128, out = 256;
  const std::vector<float> wd_full = make_gaussian(size_t(out) * 5376, 2468u, 0.05f);

  std::printf("  nvfp4 activation cost vs a bf16-activation reference:\n");
  std::printf("      %-26s %6s %8s %8s %8s %9s\n", "distribution", "K", "rms_rel", "median",
              "p99", "corr");

  struct Case {
    const char* name;
    int seed;
    int hot_stride;  // 0 = none
    bool uniform;
  };
  const Case cases[] = {{"gaussian", 1234, 0, false},
                        {"gaussian + hot channels", 5678, 61, false},
                        {"uniform", 4321, 0, true}};

  for (const Case& c : cases) {
    for (int in : {128, 512, 2048, 5376}) {
      std::vector<float> x = c.uniform ? make_data(size_t(rows) * in, uint32_t(c.seed), 1.0f)
                                       : make_gaussian(size_t(rows) * in, uint32_t(c.seed), 1.0f);
      // Post-norm transformer activations are not clean Gaussians: a handful of
      // channels run an order of magnitude hot and persist across rows. That is
      // the case block scaling exists for, and no uniform generator produces it.
      if (c.hot_stride) {
        for (int ch = 0; ch < in; ch += c.hot_stride) {
          for (int r = 0; r < rows; ++r) x[size_t(r) * in + ch] *= 20.0f;
        }
      }
      x = bf16_round(x);

      const std::vector<float> wd(wd_full.begin(), wd_full.begin() + size_t(out) * in);
      const NvfpPacked w = pack_nvfp4(wd, out, in, 1.0f, true, true);
      const std::vector<float> got = run_native_nvfp4(x, w, rows, out, in, 1.0f);

      // Same weight both sides, so the only difference is the activation.
      const std::vector<float> ref = cpu_matmul_nt(x, w.dense, rows, out, in);

      std::vector<double> rel;
      rel.reserve(got.size());
      double num = 0.0, den = 0.0, hi_err = 0.0, lo_err = 0.0;
      int hi_n = 0, lo_n = 0;
      double amax = 0.0;
      for (float v : ref) amax = std::max(amax, std::fabs(double(v)));
      for (size_t i = 0; i < got.size(); ++i) {
        const double a = std::fabs(double(ref[i]));
        const double d = std::fabs(double(got[i]) - ref[i]);
        num += d * d;
        den += double(ref[i]) * ref[i];
        if (a > 1e-6) rel.push_back(d / a);
        // Error against output magnitude: quantisation noise is roughly flat in
        // absolute terms, so it shows up as a huge *relative* error wherever
        // the output cancelled towards zero and a small one where it did not.
        if (a > 0.5 * amax) {
          hi_err += d;
          ++hi_n;
        } else if (a < 0.05 * amax) {
          lo_err += d;
          ++lo_n;
        }
      }
      std::sort(rel.begin(), rel.end());
      const double med = rel.empty() ? 0.0 : rel[rel.size() / 2];
      const double p99 = rel.empty() ? 0.0 : rel[rel.size() * 99 / 100];
      std::printf("      %-26s %6d %8.4f %8.4f %8.4f %9.6f\n", c.name, in,
                  std::sqrt(num / std::max(den, 1e-30)), med, p99, correlation(ref, got));
      if (in == 5376) {
        std::printf("          mean |err| on the largest half of outputs %.3e, "
                    "on the smallest twentieth %.3e\n",
                    hi_n ? hi_err / hi_n : 0.0, lo_n ? lo_err / lo_n : 0.0);
      }
      CHECK(std::sqrt(num / std::max(den, 1e-30)) < 1.0);  // still the same matrix
    }
  }
}

// The same measurement on a real tensor. Synthetic weights cannot say whether
// the shipped block scales are benign; these are the bytes the model ships.
VIDFAB_TEST(nvfp4_activation_cost_real_weights) {
  REQUIRE_SM120_TEST("native NVFP4 GEMM");
  std::string path;
  for (const char* prefix : {"", "../", "../../", "../../../"}) {
    const std::string p =
        std::string(prefix) + "weights/transformer/MiniMax_H3_FL2VA_pruned_nvfp4.safetensors";
    if (std::filesystem::exists(p)) {
      path = p;
      break;
    }
  }
  if (path.empty()) {
    std::printf("  nvfp4 real-weight error: checkpoint absent, skipped\n");
    return;
  }

  vidfab::SafeTensors st;
  st.open(path);
  const int rows = 128;
  const char* names[] = {"blocks.0.attn.qkv_proj", "blocks.0.mlp.fc2"};
  for (const char* name : names) {
    const vidfab::TensorView& wv = st.at(std::string(name) + ".weight");
    const vidfab::TensorView& sv = st.at(std::string(name) + ".weight_scale");
    const vidfab::TensorView& gv = st.at(std::string(name) + ".weight_scale_2");
    const int out = int(wv.shape[0]);
    const int in = int(wv.shape[1]) * 2;
    float global = 1.0f;
    std::memcpy(&global, gv.data, sizeof(float));

    // One 128-row slab, which is a whole number of scale tiles and so slices
    // cleanly out of both arrays.
    const int slab = 128;
    const int kb = in / 16;
    std::vector<uint8_t> wpacked(static_cast<const uint8_t*>(wv.data),
                                 static_cast<const uint8_t*>(wv.data) + size_t(slab) * in / 2);
    std::vector<uint8_t> wscale(static_cast<const uint8_t*>(sv.data),
                                static_cast<const uint8_t*>(sv.data) + size_t(slab) * kb);

    // What those bytes mean, straight from the file.
    std::vector<float> dense(size_t(slab) * in);
    for (int o = 0; o < slab; ++o) {
      for (int i = 0; i < in; ++i) {
        const float s = vidfab::f8_e4m3_to_f32(wscale[nvfp4_scale_slot(o, i / 16, kb)]) * global;
        const uint8_t byte = wpacked[(size_t(o) * in + i) / 2];
        dense[size_t(o) * in + i] =
            vidfab::f4_e2m1_to_f32(i % 2 == 0 ? (byte >> 4) : (byte & 0x0F)) * s;
      }
    }

    // Post-RMSNorm activations: unit RMS per row is what the norm produces, and
    // the hot channels are what makes the case hard.
    std::vector<float> x = make_gaussian(size_t(rows) * in, 777u, 1.0f);
    for (int ch = 0; ch < in; ch += 61) {
      for (int r = 0; r < rows; ++r) x[size_t(r) * in + ch] *= 20.0f;
    }
    x = bf16_round(x);

    NvfpPacked w;
    w.data = std::move(wpacked);
    w.scale = std::move(wscale);
    w.dense = dense;
    const std::vector<float> got = run_native_nvfp4(x, w, rows, slab, in, global);
    const std::vector<float> ref = cpu_matmul_nt(x, dense, rows, slab, in);
    const std::vector<float> fp4_ref =
        cpu_matmul_nt(host_quantise_act(x, rows, in), dense, rows, slab, in);

    std::printf("  %-24s out=%-6d in=%-6d  vs bf16 act: rms_rel %.4f corr %.6f | "
                "vs fp4 act: rms_rel %.6f\n",
                name, out, in, rms_rel(ref, got), correlation(ref, got), rms_rel(fp4_ref, got));
    // The operand path on the shipped bytes, with the format's own cost taken
    // out of both sides. This one is an assertion.
    CHECK_CLOSE_REL(fp4_ref, got, 1e-3, 1e-2, "native nvfp4 GEMM on a shipped weight");
  }
}

// --- timings ----------------------------------------------------------------

VIDFAB_TEST(nvfp4_gemm_production_timings) {
  REQUIRE_SM120_TEST("native NVFP4 GEMM timings");
  CublasScope cb;
  cudaDeviceProp prop{};
  VIDFAB_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("  %s: %d SMs, %d KB shared per SM\n", prop.name, prop.multiProcessorCount,
              int(prop.sharedMemPerMultiprocessor >> 10));

  struct Shape {
    const char* name;
    int out, in;
  };
  const Shape shapes[] = {{"qkv_proj", 21504, 5376},
                          {"attn.out_proj", 5376, 7168},
                          {"mlp.fc1", 28672, 5376},
                          {"mlp.fc2", 5376, 14336}};

  Timer timer;
  for (const Shape& s : shapes) {
    // The bytes never leave the device and their values do not affect timing.
    DeviceBuffer<uint8_t> dw(size_t(s.out) * s.in / 2), dws(size_t(s.out) * s.in / 16);
    VIDFAB_CUDA_CHECK(cudaMemset(dw.get(), 0x25, dw.nbytes()));
    VIDFAB_CUDA_CHECK(cudaMemset(dws.get(), 0x38, dws.nbytes()));

    for (int rows : {512, 2048, 8192}) {
      BfBuf dx(size_t(rows) * s.in), dy(size_t(rows) * s.out);
      VIDFAB_CUDA_CHECK(cudaMemset(dx.raw.get(), 0x3C, dx.raw.nbytes()));

      Workspace ws;
      ws.reserve(vidfab::cuda::nvfp4_gemm_workspace_bytes(rows, s.in) + (1u << 20));
      Workspace dqws;
      dqws.reserve(size_t(s.out) * s.in * sizeof(__nv_bfloat16) + 4096);

      const double flops = 2.0 * rows * s.out * s.in;
      // What the native path must move: the weight, its scales, the activation
      // in and the result out. The reference path moves the dequantised weight
      // twice on top of all of it.
      const double bytes = double(s.out) * s.in / 2 + double(s.out) * s.in / 16 +
                           2.0 * rows * s.in + 2.0 * rows * s.out;

      float ms = 1e30f;
      for (int pass = 0; pass < 3; ++pass) {
        ms = std::min(ms, timer.measure(
                              [&] {
                                vidfab::cuda::nvfp4_gemm_forward(dx.p(), dw.get(), dws.get(), 1.0f,
                                                                 dy.p(), rows, s.out, s.in, ws,
                                                                 nullptr);
                              },
                              2, 5));
      }

      // The reference path, same shape, same process: the production
      // dequantiser followed by a cuBLAS bf16 GEMM.
      float dq_ms = 1e30f;
      for (int pass = 0; pass < 3; ++pass) {
        dq_ms = std::min(dq_ms, timer.measure(
                                    [&] {
                                      Workspace::Scope scope(dqws);
                                      __nv_bfloat16* wb =
                                          dqws.alloc_n<__nv_bfloat16>(size_t(s.out) * s.in);
                                      vidfab::cuda::launch_dequant_nvfp4(dw.get(), dws.get(), 1.0f,
                                                                         wb, s.out, s.in, nullptr);
                                      const float alpha = 1.0f, beta = 0.0f;
                                      VIDFAB_CUBLAS_CHECK(cublasGemmEx(
                                          cb.h, CUBLAS_OP_T, CUBLAS_OP_N, s.out, rows, s.in,
                                          &alpha, wb, CUDA_R_16BF, s.in, dx.p(), CUDA_R_16BF, s.in,
                                          &beta, dy.p(), CUDA_R_16BF, s.out, CUBLAS_COMPUTE_32F,
                                          CUBLAS_GEMM_DEFAULT));
                                    },
                                    2, 5));
      }

      std::printf("  %-14s out=%-6d in=%-6d rows=%-5d  native %8.3f ms %6.1f TFLOP/s %6.0f GB/s"
                  "  |  dequant+cuBLAS %8.3f ms %6.1f TFLOP/s  |  x%.2f\n",
                  s.name, s.out, s.in, rows, ms, flops / (ms * 1e-3) / 1e12,
                  bytes / (ms * 1e-3) / 1e9, dq_ms, flops / (dq_ms * 1e-3) / 1e12, dq_ms / ms);
      CHECK(ms > 0.0f && dq_ms > 0.0f);
    }
  }
}

}  // namespace

VIDFAB_TEST(qwen_vision_layernorm_and_gelu) {
  const int rows = 2, dim = 7;
  const auto x = bf16_round(make_data(rows * dim, 8101, 2.0f));
  const auto w = bf16_round(make_data(dim, 8102, 0.3f));
  const auto b = bf16_round(make_data(dim, 8103, 0.2f));
  std::vector<float> want(rows * dim);
  for (int r = 0; r < rows; ++r) {
    float mean = 0, var = 0;
    for (int j = 0; j < dim; ++j) mean += x[r * dim + j];
    mean /= dim;
    for (int j = 0; j < dim; ++j) { float d = x[r * dim + j] - mean; var += d * d; }
    const float inv = 1.0f / std::sqrt(var / dim + 1e-6f);
    for (int j = 0; j < dim; ++j)
      want[r * dim + j] = (x[r * dim + j] - mean) * inv * w[j] + b[j];
  }
  BfBuf dx(x), dw(w), db(b), out(rows * dim);
  vidfab::cuda::launch_layernorm_affine(dx.p(), dw.p(), db.p(), out.p(), rows, dim, 1e-6f, nullptr);
  CHECK_CLOSE(want, out.host(), 2e-2, "vision layernorm");

  const auto gx = bf16_round(std::vector<float>{-3, -1, 0, 0.5f, 2});
  std::vector<float> gw(gx.size());
  for (size_t i = 0; i < gx.size(); ++i) {
    const float v = gx[i];
    gw[i] = .5f * v * (1 + std::tanh(0.7978845608028654f * (v + .044715f * v * v * v)));
  }
  BfBuf dg(gx);
  vidfab::cuda::launch_gelu_tanh(dg.p(), gx.size(), nullptr);
  CHECK_CLOSE(gw, dg.host(), 1e-2, "vision gelu tanh");
}

// Workspace's two reserve behaviours, neither of which was asserted anywhere
// before the Qwen vision encoder started carving per-image activations:
//
//   - `reserve` above the current capacity frees the old buffer and resets the
//     cursor, so every pointer handed out becomes dangling. Silent: the
//     arithmetic downstream keeps running on freed memory and returns plausible
//     numbers;
//   - `reserve` below it is a no-op, preserving both pointers and cursor.
//
// The first is why the encoder keeps its activations in an arena the tower can
// never reach: qwen_vision.cu reserves the arena it is handed from inside the
// call, so anything carved in that same arena is one undersized reserve away
// from being freed underneath the caller. Separating them is what makes the
// hazard structural rather than arithmetic -- but the property that motivates
// the separation is worth pinning, because if `reserve` ever stopped resetting
// the cursor a future reader would conclude the two arenas are redundant.
//
// The over-carve check is the other half: an arena that is simply too small
// throws rather than overruns.
VIDFAB_TEST(workspace_reserve_below_capacity_preserves_carved_pointers) {
  Workspace ws;
  ws.reserve(1 << 20);
  const size_t capacity = ws.capacity();
  CHECK(capacity >= (1u << 20));

  void* a = ws.alloc(4096);
  void* b = ws.alloc(4096);
  const size_t used = ws.used();
  CHECK(a != nullptr && b != nullptr && a != b);
  CHECK(reinterpret_cast<uintptr_t>(a) % 256 == 0);
  CHECK(reinterpret_cast<uintptr_t>(b) % 256 == 0);

  // Exactly the shape of the inner reserve: a smaller request against an arena
  // that is already big enough.
  ws.reserve(capacity / 2);
  CHECK_MSG(ws.capacity() == capacity, "reserve below capacity reallocated: %zu -> %zu", capacity,
            ws.capacity());
  CHECK_MSG(ws.used() == used, "reserve below capacity moved the cursor: %zu -> %zu", used,
            ws.used());
  void* c = ws.alloc(4096);
  CHECK_MSG(c != a && c != b, "reserve below capacity handed back a live pointer");

  // And the case the encoder's sizing exists to prevent, so the test states
  // what "too small a reserve" would actually have done.
  ws.reserve(capacity * 2);
  CHECK(ws.capacity() >= capacity * 2);
  CHECK_MSG(ws.used() == 0, "a growing reserve must reset the cursor, got %zu", ws.used());

  // An over-carve is a loud throw, not a silent overrun -- the other half of
  // why an under-sized reserve cannot corrupt results quietly.
  Workspace tight;
  tight.reserve(1024);
  bool threw = false;
  try {
    tight.alloc(1 << 20);
  } catch (const std::exception&) {
    threw = true;
  }
  CHECK_MSG(threw, "Workspace::alloc past the end must throw");
}

// The whole vision tower on the shipped weights, twice.
//
// `QwenVisionEncoder::encode` used to build fifteen DeviceBuffers per image and
// free them at the end of the iteration; it now carves them out of its arena,
// which is a change to *where* every activation lives and to nothing else. The
// only assertion worth making about that is that the numbers did not move, and
// the only way to make it is to run the real thing: the tower is 27 blocks of
// weights that no synthetic fixture reproduces.
//
// Two images in one call, so the second one exercises a re-carve against an
// arena that already has a high-water mark, which the single-image probe in
// tools/qwenvisionprobe.cpp does not reach. The two are the same picture, so
// their embeddings must come back bit for bit identical -- that is what a stale
// or overlapping carve would break, and it is checked on raw bf16 rather than
// on a norm because a norm survives a permutation.
VIDFAB_TEST(qwen_vision_encode_reuses_arena_across_images) {
  // Five levels, because the natural place to run the exe is
  // build/Release, which is three below the tree root, and a ctest run from
  // build/ is two. A search that stops short resolves nothing, and a skip that
  // only printf's is indistinguishable from a pass -- this campaign has already
  // been bitten once by a fixture-dependent case that quietly skipped for
  // months. CHECK_DEFERRED reports on every run without failing the suite, and
  // the resolved path is printed so "found" is never taken on trust either.
  std::string path;
  for (const char* prefix : {"", "../", "../../", "../../../", "../../../../"}) {
    const std::string p =
        std::string(prefix) + "weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors";
    if (std::filesystem::exists(p)) {
      path = p;
      break;
    }
  }
  if (path.empty()) {
    CHECK_DEFERRED(false,
                   "qwen vision: no text encoder checkpoint under any of ./ .. ../.. ../../.. "
                   "../../../.. -- the whole-tower carve is NOT being exercised");
    return;
  }
  std::printf("  qwen vision: using %s\n", path.c_str());

  std::vector<uint8_t> rgb(256 * 256 * 3);
  for (int y = 0; y < 256; ++y) {
    for (int x = 0; x < 256; ++x) {
      const size_t i = (size_t(y) * 256 + x) * 3;
      rgb[i] = uint8_t((x * 17 + y * 3) & 255);
      rgb[i + 1] = uint8_t((x * 5 + y * 11) & 255);
      rgb[i + 2] = uint8_t((x ^ y) & 255);
    }
  }
  const auto pixels = vidfab::text::qwen3vl_patchify_resized_rgb(rgb, 256, 256);

  vidfab::SafeTensors checkpoint;
  checkpoint.open(path);
  vidfab::text::QwenVisionEncoder encoder;
  encoder.load(checkpoint);
  const vidfab::text::QwenVisionEmbedding out = encoder.encode({pixels, pixels});

  CHECK(out.tokens == 128);
  CHECK(out.hidden == 5120);
  CHECK(out.main.size() == 128ull * 5120);
  const size_t half = out.main.size() / 2;
  size_t bad = 0;
  for (size_t i = 0; i < half; ++i) bad += out.main[i] != out.main[half + i];
  CHECK_MSG(bad == 0, "second image's main embedding differs in %zu of %zu bf16", bad, half);
  for (int d = 0; d < 3; ++d) {
    CHECK(out.deepstack[d].size() == out.main.size());
    size_t dbad = 0;
    for (size_t i = 0; i < half; ++i) dbad += out.deepstack[d][i] != out.deepstack[d][half + i];
    CHECK_MSG(dbad == 0, "deepstack %d differs between identical images in %zu of %zu bf16", d,
              dbad, half);
  }

  // Every value finite, and the rms printed so a future carve bug that shifts
  // an activation shows up as a number rather than as silence. These are the
  // same four figures qwenvisionprobe reports.
  size_t nonfinite = 0;
  auto rms = [&](const std::vector<uint16_t>& v) {
    long double acc = 0;
    for (uint16_t b : v) {
      const float x = vidfab::bf16_to_f32(b);
      nonfinite += !std::isfinite(x);
      acc += static_cast<long double>(x) * x;
    }
    return std::sqrt(double(acc / v.size()));
  };
  std::printf("  qwen vision main rms %.9g  deepstack %.9g %.9g %.9g\n", rms(out.main),
              rms(out.deepstack[0]), rms(out.deepstack[1]), rms(out.deepstack[2]));
  CHECK_MSG(nonfinite == 0, "qwen vision produced %zu nonfinite values", nonfinite);
}

VIDFAB_TEST(qwen_vision_merge_and_deepstack_scatter) {
  const int groups = 2, dim = 3;
  const auto x = bf16_round(make_data(groups * 4 * dim, 8110, 1.0f));
  BfBuf dx(x), merged(x.size());
  vidfab::cuda::launch_merge_four_rows(dx.p(), merged.p(), groups, dim, nullptr);
  CHECK_CLOSE(x, merged.host(), 0, "merge four contiguous rows");

  const std::vector<int32_t> rows = {1, 4};
  const auto add = bf16_round(std::vector<float>{1, 2, 3, -1, -.5f, .25f});
  auto base = bf16_round(make_data(6 * dim, 8111, .2f));
  auto want = base;
  for (int r = 0; r < 2; ++r)
    for (int d = 0; d < dim; ++d) want[rows[r] * dim + d] += add[r * dim + d];
  BfBuf dadd(add), dbase(base); auto didx = to_device_i32(rows);
  vidfab::cuda::launch_scatter_add_rows(dadd.p(), didx.get(), dbase.p(), 2, dim, nullptr);
  CHECK_CLOSE(want, dbase.host(), 2e-2, "deepstack additive scatter");
}
