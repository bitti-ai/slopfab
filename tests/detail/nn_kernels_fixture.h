#pragma once

// Private shared fixtures for the nn_kernels suites.
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

#include "../harness.h"
#include "slopfab/cuda/attention.cuh"
#include "slopfab/cuda/sol_attention.cuh"
#include "slopfab/cuda/device.h"
#include "slopfab/cuda/gemm.cuh"
#include "slopfab/cuda/linear.cuh"
#include "slopfab/cuda/nn_kernels.cuh"
#include "slopfab/cuda/nvfp4_gemm.cuh"
#include "slopfab/cuda/sage_attention.cuh"
#include "slopfab/cuda/workspace.cuh"
#include "slopfab/cuda/w4a8.cuh"
#include "slopfab/dit/packing.h"
#include "slopfab/dtype.h"
#include "slopfab/text/qwen_vision.h"
#include "slopfab/safetensors.h"


namespace {

using slopfab::cuda::DeviceBuffer;
using slopfab::cuda::Workspace;
using slopfab::test::make_data;

struct CublasScope {
  cublasHandle_t h = nullptr;
  CublasScope() { SLOPFAB_CUBLAS_CHECK(slopfab::cuda::cublas_create(&h)); }
  ~CublasScope() { slopfab::cuda::cublas_destroy(h); }
};

bool test_is_sm120() {
  return slopfab::cuda::current_device_compute_capability() == 120;
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
// `slopfab::f32_to_bf16`, which rounds to nearest even exactly as the device
// intrinsic does.
struct BfBuf {
  DeviceBuffer<uint16_t> raw;

  explicit BfBuf(size_t n) : raw(n) {}
  explicit BfBuf(const std::vector<float>& host) : raw(host.size()) {
    std::vector<uint16_t> bits(host.size());
    for (size_t i = 0; i < host.size(); ++i) bits[i] = slopfab::f32_to_bf16(host[i]);
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
    for (size_t i = 0; i < b.size(); ++i) out[i] = slopfab::bf16_to_f32(b[i]);
    return out;
  }
};

// The reference must see the same inputs the kernel does, so anything destined
// for a bf16 buffer is rounded on the host first.
std::vector<float> bf16_round(const std::vector<float>& v) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = slopfab::bf16_to_f32(slopfab::f32_to_bf16(v[i]));
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
      km[size_t(kb) * dim + d] = slopfab::bf16_to_f32(slopfab::f32_to_bf16(
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

#if 0  // Removed: canonical H3 tables are host-built and tested in test_packing.cpp.


#endif

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


// Every e4m3 bit pattern must dequantise to exactly what dtype.h's host
// reference produces. The two implementations are written out separately, so
// this is the check that keeps them in step.


// The e4m3 encoder is only used by the (deferred) native fp8 path, but it has
// to be right before that path can be trusted. Every finite pattern must
// survive dequantise -> quantise unchanged.

// The ConvRot rotation, against an explicitly constructed 256x256 matrix. This
// is the test that separates the regular Hadamard from the Sylvester one — the
// wrong choice gives relative error 1.4, not a crash.



}  // namespace
