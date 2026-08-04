// Tests for the Qwen3-VL text conditioner.
//
// Every reference here is written from docs/text_encoder_spec.md, never from
// the code it checks. Section 9 of that spec is a list of fifteen ways to get
// this module wrong, and *none of them crash*: bidirectional attention, a
// skipped ConvRot, the DiT's 96-of-128 RoPE, `h % 8` grouped-query mapping,
// SiLU on the wrong projection — all produce finite, well-scaled, plausible
// output. So wherever a hazard has a specific wrong form, the test computes
// that form too and asserts the implementation does *not* match it. A test
// that only checks the right answer cannot tell you whether it had any power
// to detect the wrong one.
//
// The checkpoint-dependent cases skip cleanly when
// weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors is absent; it is
// licence-restricted and not committed.

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "harness.h"
#include "vidfab/cuda/device.h"
#include "vidfab/cuda/gemm.cuh"
#include "vidfab/cuda/linear.cuh"
#include "vidfab/cuda/nn_kernels.cuh"
#include "vidfab/cuda/workspace.cuh"
#include "vidfab/dtype.h"
#include "vidfab/safetensors.h"
#include "vidfab/safetensors_write.h"
#include "vidfab/tensor_convert.h"
#include "vidfab/text/encoder.h"
#include "vidfab/text/tokenizer.h"

namespace {

using vidfab::cuda::DeviceBuffer;
using vidfab::cuda::QuantFormat;
using vidfab::cuda::QuantWeight;
using vidfab::cuda::Workspace;
using vidfab::test::make_data;

// --- host/device plumbing ---------------------------------------------------

struct BfBuf {
  DeviceBuffer<uint16_t> raw;

  explicit BfBuf(size_t n) : raw(n) {}
  explicit BfBuf(const std::vector<float>& host) : raw(host.size()) {
    std::vector<uint16_t> bits(host.size());
    for (size_t i = 0; i < host.size(); ++i) bits[i] = vidfab::f32_to_bf16(host[i]);
    raw.copy_from_host(bits.data(), bits.size());
  }

  __nv_bfloat16* p() { return reinterpret_cast<__nv_bfloat16*>(raw.get()); }

  std::vector<float> host() const {
    std::vector<uint16_t> b(raw.size());
    raw.copy_to_host(b.data(), b.size());
    std::vector<float> out(b.size());
    for (size_t i = 0; i < b.size(); ++i) out[i] = vidfab::bf16_to_f32(b[i]);
    return out;
  }
};

struct CublasScope {
  cublasHandle_t h = nullptr;
  CublasScope() { VIDFAB_CUBLAS_CHECK(cublasCreate(&h)); }
  ~CublasScope() { cublasDestroy(h); }
};

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

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
  double worst = 0.0;
  for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(double(a[i]) - b[i]));
  return worst;
}

double rms(const std::vector<float>& v) {
  double acc = 0.0;
  for (float x : v) acc += double(x) * x;
  return std::sqrt(acc / std::max<size_t>(1, v.size()));
}

// Worst absolute error as a fraction of the reference's RMS.
//
// Elementwise relative error is the wrong yardstick for a bf16 GEMM with
// cancellation: a 512-term dot product whose terms sum to 1.5 and whose result
// is 0.07 carries the *absolute* error of the terms, so the elements nearest
// zero always look catastrophic in relative terms while carrying no
// information. The error that matters is measured against the scale of the
// output as a whole, and it is what the 1e-3 / 1e-2 project tolerance means
// once cancellation is in play.
double rms_relative_error(const std::vector<float>& want, const std::vector<float>& got) {
  return max_abs_diff(want, got) / std::max(1e-30, rms(want));
}

// RMS of the error over RMS of the reference. The max-based measure above is
// dominated by the worst of tens of thousands of samples — about 4 sigma for
// Gaussian rounding noise — so quoting both separates "one element is unlucky"
// from "the whole tensor is wrong".
double rms_error_ratio(const std::vector<float>& want, const std::vector<float>& got) {
  std::vector<float> diff(want.size());
  for (size_t i = 0; i < want.size(); ++i) diff[i] = want[i] - got[i];
  return rms(diff) / std::max(1e-30, rms(want));
}

std::string find_checkpoint() {
  // The worktree layout puts the repository three levels up, so the search
  // reaches further than test_adaln's.
  for (const char* prefix : {"", "../", "../../", "../../../", "../../../../"}) {
    const std::string p =
        std::string(prefix) + "weights/text_encoder/qwen3vl_32b_int8_convrot.safetensors";
    if (std::filesystem::exists(p)) return p;
  }
  return {};
}

std::string find_tokenizer() {
  for (const char* prefix : {"", "../", "../../", "../../../", "../../../../"}) {
    const std::string p = std::string(prefix) + "ref/FL2VA/text_encoder/tokenizer.json";
    if (std::filesystem::exists(p)) return p;
  }
  return {};
}

// --- CPU references ---------------------------------------------------------

// y = w * (x * rsqrt(mean(x^2) + eps)). eps inside the root, added to the mean
// of squares; no mean subtraction, no bias. Accumulated in double, which is at
// least as good as the reference's mandatory fp32 upcast (spec 4.1).
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
    for (int i = 0; i < dim; ++i) out[size_t(r) * dim + i] = float(x[size_t(r) * dim + i] * inv * w[i]);
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

// GPT-NeoX half-split rotation. `rotary` lets the test build the *hazard* form
// as well — the H3 DiT rotates 96 of 128 and pairs j with j+48, and porting
// that habit here is spec section 9 item 10.
std::vector<float> cpu_rope(const std::vector<float>& x, const std::vector<float>& cos,
                            const std::vector<float>& sin, int rows, int heads, int head_dim,
                            int rotary) {
  std::vector<float> out = x;
  const int half = rotary / 2;
  for (int r = 0; r < rows; ++r) {
    for (int h = 0; h < heads; ++h) {
      const size_t base = (size_t(r) * heads + h) * head_dim;
      for (int j = 0; j < half; ++j) {
        const float lo = x[base + j];
        const float hi = x[base + j + half];
        const float c = cos[size_t(r) * head_dim + j];
        const float s = sin[size_t(r) * head_dim + j];
        out[base + j] = lo * c - hi * s;
        out[base + j + half] = hi * c + lo * s;
      }
    }
  }
  return out;
}

// Dense attention. `causal` false is the hazard: bidirectional attention
// changes every row but the last and is completely silent (spec section 3).
std::vector<float> cpu_attention(const std::vector<float>& q, const std::vector<float>& k,
                                 const std::vector<float>& v, int seq, int heads, int kv_heads,
                                 int head_dim, float scale, bool causal) {
  const int qld = heads * head_dim;
  const int kvld = kv_heads * head_dim;
  const int group = heads / kv_heads;
  std::vector<float> out(size_t(seq) * qld, 0.0f);
  std::vector<double> p(seq);

  for (int h = 0; h < heads; ++h) {
    // Query head h reads kv head h/group — contiguous blocks. `h % kv_heads`
    // is the plausible wrong answer and no shape check catches it.
    const int kv = h / group;
    for (int i = 0; i < seq; ++i) {
      const int last = causal ? i : seq - 1;
      double m = -1e300;
      for (int j = 0; j <= last; ++j) {
        double dot = 0.0;
        for (int d = 0; d < head_dim; ++d) {
          dot += double(q[size_t(i) * qld + h * head_dim + d]) *
                 k[size_t(j) * kvld + kv * head_dim + d];
        }
        p[j] = dot * scale;
        m = std::max(m, p[j]);
      }
      double sum = 0.0;
      for (int j = 0; j <= last; ++j) {
        p[j] = std::exp(p[j] - m);
        sum += p[j];
      }
      for (int d = 0; d < head_dim; ++d) {
        double acc = 0.0;
        for (int j = 0; j <= last; ++j) acc += p[j] * v[size_t(j) * kvld + kv * head_dim + d];
        out[size_t(i) * qld + h * head_dim + d] = float(acc / sum);
      }
    }
  }
  return out;
}

// --- the regular Hadamard, built the long way -------------------------------
//
// h4 is the *regular* Hadamard matrix (symmetric, constant row sums 2), not the
// Sylvester one; H = kron^4(h4)/16. Built here as an explicit 256x256 matrix
// and applied as a matmul, so it shares no code with the butterfly it checks
// (docs/convrot_notes.md).
std::vector<float> hadamard(int size) {
  const int h4[4][4] = {{1, 1, 1, -1}, {1, 1, -1, 1}, {1, -1, 1, 1}, {-1, 1, 1, 1}};
  std::vector<float> m{1.0f};
  int n = 1;
  while (n < size) {
    std::vector<float> next(size_t(n * 4) * (n * 4));
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        for (int r = 0; r < n; ++r) {
          for (int c = 0; c < n; ++c) {
            next[size_t(i * n + r) * (n * 4) + (j * n + c)] = float(h4[i][j]) * m[size_t(r) * n + c];
          }
        }
      }
    }
    m.swap(next);
    n *= 4;
  }
  const float norm = 1.0f / std::sqrt(float(size));
  for (float& v : m) v *= norm;
  return m;
}

// x_rot[g] = x[g] @ H, group by group along the trailing axis. H is symmetric
// and involutory, so the same routine rotates and de-rotates.
std::vector<float> rotate_rows(const std::vector<float>& x, int rows, int dim, int group,
                               const std::vector<float>& H) {
  std::vector<float> out(x.size());
  const int blocks = dim / group;
  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < blocks; ++b) {
      const size_t base = size_t(r) * dim + size_t(b) * group;
      for (int c = 0; c < group; ++c) {
        double acc = 0.0;
        for (int i = 0; i < group; ++i) acc += double(x[base + i]) * H[size_t(i) * group + c];
        out[base + c] = float(acc);
      }
    }
  }
  return out;
}

float silu(float z) { return z / (1.0f + std::exp(-z)); }

// --- tests ------------------------------------------------------------------

VIDFAB_TEST(encoder_rope_inv_freq_and_tables) {
  const std::vector<float> inv = vidfab::text::rope_inv_freq(128, 5.0e6f);
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
  const std::vector<float> wrong = vidfab::text::rope_inv_freq(80, 5.0e6f);
  CHECK(wrong.size() == 40);
  CHECK(std::fabs(wrong[1] - inv[1]) > 1e-3);

  std::vector<float> cos;
  std::vector<float> sin;
  vidfab::text::build_rope_tables(6, inv, cos, sin);
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
  vidfab::text::build_rope_tables(4096, inv, cos_far, sin_far);
  CHECK(cos_far[size_t(4095) * 128 + 63] > 0.9999f);
  CHECK(cos_far[size_t(4095) * 128 + 48] > 0.99f);
  // ...and the low channels must have turned many times.
  CHECK(std::fabs(sin_far[size_t(4095) * 128 + 0]) <= 1.0f);
  CHECK(cos_far[size_t(4095) * 128 + 0] < 0.9999f);
}

VIDFAB_TEST(encoder_rope_neox_rotates_all_128_dims) {
  const int rows = 512;
  const int heads = 3;
  const int head_dim = 128;

  const std::vector<float> inv = vidfab::text::rope_inv_freq(head_dim, 5.0e6f);
  std::vector<float> cos;
  std::vector<float> sin;
  vidfab::text::build_rope_tables(rows, inv, cos, sin);

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
  vidfab::cuda::launch_rope_neox(dx.p(), dcos.get(), dsin.get(), rows, heads, head_dim, nullptr);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
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

VIDFAB_TEST(encoder_causal_attention) {
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

  vidfab::text::CausalAttentionConfig cfg;
  cfg.seq_len = seq;
  cfg.num_heads = heads;
  cfg.num_kv_heads = kv_heads;
  cfg.head_dim = head_dim;
  CHECK_NEAR(vidfab::text::causal_attention_scale(cfg), 1.0 / std::sqrt(64.0), 1e-7);

  const float scale = vidfab::text::causal_attention_scale(cfg);
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
    ws.reserve(vidfab::text::causal_attention_workspace_bytes(cfg));
    vidfab::text::causal_attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dout.p(), cfg,
                                           ws);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
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
    vidfab::text::CausalAttentionConfig short_cfg = cfg;
    short_cfg.seq_len = t + 1;
    short_cfg.query_block = 128;
    BfBuf dshort(size_t(t + 1) * qld);
    Workspace ws;
    ws.reserve(vidfab::text::causal_attention_workspace_bytes(short_cfg));
    vidfab::text::causal_attention_forward(cb.h, nullptr, dq.p(), dk.p(), dv.p(), dshort.p(),
                                           short_cfg, ws);
    VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
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

VIDFAB_TEST(encoder_convrot_cross_check) {
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

  vidfab::cuda::LinearRunner runner;
  runner.init(cb.h, nullptr);
  Workspace ws;
  ws.reserve(vidfab::cuda::linear_workspace_bytes(qw, rows, vidfab::cuda::ComputeType::kBF16));
  runner.forward(qw, dx.p(), rows, dy.p(), ws);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
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

VIDFAB_TEST(encoder_layer_vs_cpu_reference) {
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

  const std::vector<float> inv = vidfab::text::rope_inv_freq(head_dim, 5.0e6f);
  std::vector<float> cos;
  std::vector<float> sin;
  vidfab::text::build_rope_tables(L, inv, cos, sin);

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

  vidfab::text::LayerWeights w;
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

  vidfab::text::LayerDims dims;
  dims.num_tokens = L;
  dims.hidden = hidden;
  dims.num_heads = heads;
  dims.num_kv_heads = kv_heads;
  dims.head_dim = head_dim;
  dims.intermediate = inner;
  dims.rms_norm_eps = eps;

  vidfab::cuda::LinearRunner runner;
  runner.init(cb.h, nullptr);
  Workspace ws;
  ws.reserve(vidfab::text::layer_workspace_bytes(dims));
  vidfab::text::encoder_layer_forward(cb.h, nullptr, runner, w, dims, dcos.get(), dsin.get(),
                                      dx.p(), ws);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());
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

VIDFAB_TEST(encoder_layer_layout) {
  vidfab::text::EncoderConfig cfg;
  const vidfab::text::LayerLayout layout = vidfab::text::make_layer_layout(cfg);

  const size_t q = size_t(8192) * 5120;
  const size_t kv = size_t(1024) * 5120;
  const size_t o = size_t(5120) * 8192;
  const size_t mlp = size_t(25600) * 5120;

  CHECK(layout.bytes[int(vidfab::text::LayerTensor::kQWeight)] == q);
  CHECK(layout.bytes[int(vidfab::text::LayerTensor::kKWeight)] == kv);
  CHECK(layout.bytes[int(vidfab::text::LayerTensor::kOWeight)] == o);
  CHECK(layout.bytes[int(vidfab::text::LayerTensor::kDownWeight)] == mlp);
  CHECK(layout.bytes[int(vidfab::text::LayerTensor::kQScale)] == 8192 * sizeof(float));
  CHECK(layout.bytes[int(vidfab::text::LayerTensor::kQNorm)] == 128 * 2);
  CHECK(layout.bytes[int(vidfab::text::LayerTensor::kInputLayerNorm)] == 5120 * 2);

  // Spec section 7: 487 587 840 B of int8 weights, 286 720 B of scales and
  // 20 992 B of norms per layer. The blob rounds each up to 256 bytes.
  size_t weights = 0;
  size_t scales = 0;
  size_t norms = 0;
  for (int i = 0; i < vidfab::text::kLayerTensorCount; ++i) {
    const vidfab::text::TensorSpec spec =
        vidfab::text::layer_tensor_spec(cfg, static_cast<vidfab::text::LayerTensor>(i));
    if (spec.dtype == vidfab::DType::kI8) weights += layout.bytes[i];
    else if (spec.dtype == vidfab::DType::kF32) scales += layout.bytes[i];
    else norms += layout.bytes[i];
  }
  CHECK(weights == 487587840);
  CHECK(scales == 286720);
  CHECK(norms == 20992);
  CHECK(layout.total_bytes >= weights + scales + norms);
  CHECK(layout.total_bytes % 256 == 0);

  // Every offset is 256-byte aligned and no two tensors overlap.
  for (int i = 0; i < vidfab::text::kLayerTensorCount; ++i) {
    CHECK(layout.offset[i] % 256 == 0);
    if (i > 0) CHECK(layout.offset[i] >= layout.offset[i - 1] + layout.bytes[i - 1]);
  }

  // Every contraction width is a multiple of the ConvRot group, so there is no
  // skip-when-not-divisible case in this checkpoint (spec section 5.2).
  for (int in_features : {5120, 8192, 25600}) CHECK(in_features % 256 == 0);
}

VIDFAB_TEST(encoder_validation_rejects_a_foreign_checkpoint) {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "vidfab_encoder_test";
  std::filesystem::create_directories(dir);
  const std::string path = (dir / "fake.safetensors").string();

  std::vector<vidfab::TensorWrite> tensors;
  tensors.push_back(
      {"model.embed_tokens.weight", {4, 5120}, std::vector<float>(4 * 5120, 0.5f)});
  vidfab::write_safetensors(path, tensors);

  vidfab::SafeTensors st;
  st.open(path);

  // Real widths, tiny vocabulary: the file is F32 where the checkpoint is
  // BF16, so validation must reject it on dtype and say which tensor.
  vidfab::text::EncoderConfig cfg;
  cfg.vocab_size = 4;
  cfg.num_layers = 1;

  bool threw = false;
  std::string message;
  try {
    vidfab::text::validate_checkpoint(st, cfg);
  } catch (const std::exception& e) {
    threw = true;
    message = e.what();
  }
  CHECK(threw);
  // The message must name the offending tensor, not just say "bad checkpoint".
  CHECK_MSG(message.find("model.embed_tokens.weight") != std::string::npos,
            "validation message does not name the tensor: %s", message.c_str());

  // The gather refuses the same file for the same reason, and refuses an id
  // outside the vocabulary.
  std::vector<uint16_t> out;
  bool gather_threw = false;
  try {
    vidfab::text::gather_embedding_rows(st.at("model.embed_tokens.weight"), {0, 1}, out);
  } catch (const std::exception&) {
    gather_threw = true;
  }
  CHECK(gather_threw);

  st.close();
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

// --- checkpoint-dependent ----------------------------------------------------

VIDFAB_TEST(encoder_real_checkpoint_convrot_cross_check) {
  const std::string path = find_checkpoint();
  if (path.empty()) {
    std::printf("  text encoder checkpoint not present; skipping\n");
    return;
  }
  CublasScope cb;

  vidfab::SafeTensors st;
  st.open(path);
  vidfab::text::EncoderConfig cfg;
  vidfab::text::validate_checkpoint(st, cfg);
  CHECK(st.tensor_count() == 1602);

  // Layer 0's k_proj: [1024, 5120] int8, small enough to run both ways.
  const vidfab::TensorView& wv = st.at("model.layers.0.self_attn.k_proj.weight");
  const vidfab::TensorView& sv = st.at("model.layers.0.self_attn.k_proj.weight_scale");
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

  vidfab::cuda::LinearRunner runner;
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
  ws.reserve(vidfab::cuda::linear_workspace_bytes(qa, rows, vidfab::cuda::ComputeType::kBF16));
  runner.forward(qa, dx.p(), rows, ya.p(), ws);

  // (b) de-rotate the dequantised weight and use the unrotated activation.
  // H is involutory, so the same butterfly undoes it.
  BfBuf w_dequant(size_t(out_features) * in_features);
  BfBuf w_plain(size_t(out_features) * in_features);
  vidfab::cuda::launch_dequant_i8_per_channel(dw.get(), dscale.get(), w_dequant.p(), out_features,
                                              in_features, nullptr);
  vidfab::cuda::launch_convrot(w_dequant.p(), w_plain.p(), out_features, in_features, 256, nullptr);

  QuantWeight qb;
  qb.format = QuantFormat::kBF16;
  qb.data = w_plain.p();
  qb.out_features = out_features;
  qb.in_features = in_features;
  qb.convrot = false;

  BfBuf yb(size_t(rows) * out_features);
  runner.forward(qb, dx.p(), rows, yb.p(), ws);
  VIDFAB_CUDA_CHECK(cudaDeviceSynchronize());

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
  vidfab::to_f32(st.at("model.layers.0.self_attn.k_norm.weight"), knorm);
  CHECK(knorm.size() == 128);
  const float kmax = *std::max_element(knorm.begin(), knorm.end());
  CHECK_MSG(kmax > 20.0f && kmax < 21.0f, "layer 0 k_norm max is %.4f, expected ~20.75", kmax);
}

VIDFAB_TEST(encoder_real_encode) {
  const std::string path = find_checkpoint();
  if (path.empty()) {
    std::printf("  text encoder checkpoint not present; skipping\n");
    return;
  }

  vidfab::SafeTensors st;
  st.open(path);

  // A real prompt when the tokenizer is available, a synthetic id run
  // otherwise. Either way the causality test needs one token list that is a
  // strict prefix of another.
  std::vector<int32_t> ids;
  const std::string tok_path = find_tokenizer();
  if (!tok_path.empty()) {
    vidfab::text::Tokenizer tokenizer;
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
    std::printf("  ref/ tokenizer.json not present; using synthetic token ids\n");
    for (int i = 0; i < 200; ++i) ids.push_back(1000 + i);
  }
  CHECK(!ids.empty());

  std::vector<int32_t> longer = ids;
  for (int i = 0; i < 37; ++i) longer.push_back(5000 + i * 7);

  size_t free_before = 0;
  size_t total = 0;
  VIDFAB_CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
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
    std::printf("  only %.2f GB free; need ~2 GB even to stream. Skipping.\n",
                double(free_before) / (1 << 30));
    return;
  }

  std::vector<float> resident_out;

  // --- residency mode 1: everything on the device.
  {
    vidfab::text::Encoder encoder;
    vidfab::text::EncoderConfig cfg;
    cfg.residency = vidfab::text::Residency::kResident;

    bool loaded = true;
    try {
      encoder.load(st, cfg);
    } catch (const std::exception& e) {
      loaded = false;
      std::printf("  resident load failed (%s); the card cannot hold 24.4 GB right now\n",
                  e.what());
    }

    if (loaded) {
      const vidfab::text::PromptEmbedding a = encoder.encode(ids);
      size_t free_after = 0;
      VIDFAB_CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
      const vidfab::text::EncoderStats& s = encoder.stats();
      std::printf(
          "  resident: load %.2f s, encode %.3f s for %d tokens, weights %.2f GB, "
          "workspace %.2f GB, accounted peak %.2f GB, measured %.2f GB\n",
          s.load_seconds, s.last_encode_seconds, s.last_num_tokens,
          double(s.weight_bytes) / (1 << 30), double(s.workspace_bytes) / (1 << 30),
          double(s.peak_device_bytes) / (1 << 30), double(free_before - free_after) / (1 << 30));

      CHECK(encoder.residency() == vidfab::text::Residency::kResident);
      CHECK(a.num_tokens == int(ids.size()));
      CHECK(a.hidden_size == 5120);
      CHECK(a.data.size() == size_t(a.num_tokens) * 5120);

      size_t nonfinite = 0;
      for (float value : a.data) {
        if (!std::isfinite(value)) ++nonfinite;
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
      const vidfab::text::PromptEmbedding b = encoder.encode(longer);
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
        if (r == 0) row0_relative = relative;
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
    vidfab::text::Encoder encoder;
    vidfab::text::EncoderConfig cfg;
    cfg.residency = vidfab::text::Residency::kStreaming;
    encoder.load(st, cfg);
    CHECK(encoder.residency() == vidfab::text::Residency::kStreaming);

    const vidfab::text::PromptEmbedding c = encoder.encode(ids);
    size_t free_after = 0;
    VIDFAB_CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
    const vidfab::text::EncoderStats& s = encoder.stats();
    std::printf(
        "  streaming: load %.2f s, encode %.3f s for %d tokens, layer buffers %.2f GB, "
        "workspace %.2f GB, accounted peak %.2f GB, measured %.2f GB\n",
        s.load_seconds, s.last_encode_seconds, s.last_num_tokens,
        double(s.weight_bytes) / (1 << 30), double(s.workspace_bytes) / (1 << 30),
        double(s.peak_device_bytes) / (1 << 30), double(free_before - free_after) / (1 << 30));

    CHECK(c.num_tokens == int(ids.size()));
    size_t nonfinite = 0;
    for (float value : c.data) {
      if (!std::isfinite(value)) ++nonfinite;
    }
    CHECK(nonfinite == 0);

    // The two modes differ only in when the weights arrive, so they must agree
    // bit for bit. A double-buffering race would show up here and nowhere else.
    if (!resident_out.empty()) {
      size_t mismatches = 0;
      for (size_t i = 0; i < c.data.size(); ++i) {
        if (c.data[i] != resident_out[i]) ++mismatches;
      }
      CHECK_MSG(mismatches == 0,
                "%zu of %zu values differ between the resident and streaming paths (max abs %.4g)",
                mismatches, c.data.size(), max_abs_diff(resident_out, c.data));
    }

    const vidfab::text::PromptEmbedding d = encoder.encode(ids);
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

}  // namespace
